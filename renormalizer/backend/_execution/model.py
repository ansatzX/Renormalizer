"""Immutable metadata for contraction execution plans."""

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from renormalizer.backend._execution.workspace import workspace_bytes_for_steps


_LAYOUTS = frozenset({"C", "F", "strided"})


def _as_tuple(value, name):
    try:
        return tuple(value)
    except TypeError as error:
        raise TypeError("{} must be an iterable".format(name)) from error


def _validate_modes(modes, *, allow_empty=True):
    modes = _as_tuple(modes, "modes")
    if not allow_empty and not modes:
        raise ValueError("modes must not be empty")
    if any(not isinstance(mode, str) or len(mode) != 1 for mode in modes):
        raise ValueError("modes must contain one-character strings")
    if len(set(modes)) != len(modes):
        raise ValueError("modes must be unique")
    return modes


def _normalize_dtype(dtype):
    try:
        normalized = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise ValueError("unsupported dtype: {!r}".format(dtype)) from error
    if normalized.hasobject or normalized.fields is not None or normalized.kind not in "biufc":
        raise ValueError("unsupported dtype: {!r}".format(dtype))
    return normalized.name


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str
    layout: str
    modes: tuple[str, ...]

    def __post_init__(self):
        shape = _as_tuple(self.shape, "shape")
        if any(isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) for dim in shape):
            raise TypeError("shape dimensions must be integers")
        shape = tuple(int(dim) for dim in shape)
        if any(dim < 0 for dim in shape):
            raise ValueError("shape dimensions must be non-negative")
        modes = _validate_modes(self.modes)
        if len(shape) != len(modes):
            raise ValueError("shape rank must match mode rank")
        if self.layout not in _LAYOUTS:
            raise ValueError("unsupported layout: {!r}".format(self.layout))
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", _normalize_dtype(self.dtype))
        object.__setattr__(self, "modes", modes)

    @property
    def nbytes(self):
        elements = 1
        for dim in self.shape:
            elements *= dim
        return elements * np.dtype(self.dtype).itemsize


@dataclass(frozen=True)
class BufferRef:
    key: str
    spec: TensorSpec

    def __post_init__(self):
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("buffer key must be a non-empty string")
        if not isinstance(self.spec, TensorSpec):
            raise TypeError("buffer spec must be a TensorSpec")


def _contiguous_strides(shape, layout):
    strides = [0] * len(shape)
    stride = 1
    axes = reversed(range(len(shape))) if layout == "C" else range(len(shape))
    for axis in axes:
        strides[axis] = stride
        stride *= shape[axis]
    return tuple(strides)


def _is_contiguous(shape, strides, layout):
    if not shape or any(dim == 0 for dim in shape):
        return True
    expected = 1
    axes = reversed(range(len(shape))) if layout == "C" else range(len(shape))
    for axis in axes:
        dim = shape[axis]
        if dim != 1 and strides[axis] != expected:
            return False
        expected *= dim
    return True


def _canonical_layout(shape, strides):
    if _is_contiguous(shape, strides, "C"):
        return "C"
    if _is_contiguous(shape, strides, "F"):
        return "F"
    return "strided"


def _transform_view_layout(input_spec, axes):
    output_shape = tuple(input_spec.shape[axis] for axis in axes)
    if input_spec.layout == "strided":
        if not output_shape or any(dim == 0 for dim in output_shape):
            return "C"
        return "strided"
    input_strides = _contiguous_strides(input_spec.shape, input_spec.layout)
    output_strides = tuple(input_strides[axis] for axis in axes)
    return _canonical_layout(output_shape, output_strides)


@dataclass(frozen=True)
class TransformStep:
    input: BufferRef
    output: BufferRef
    axes: tuple[int, ...]
    copy: bool

    def __post_init__(self):
        if not isinstance(self.input, BufferRef) or not isinstance(self.output, BufferRef):
            raise TypeError("transform buffers must be BufferRef instances")
        axes = _as_tuple(self.axes, "transform axes")
        rank = len(self.input.spec.shape)
        if any(isinstance(axis, bool) or not isinstance(axis, int) for axis in axes):
            raise TypeError("transform axes must be integers")
        if tuple(sorted(axes)) != tuple(range(rank)):
            raise ValueError("transform axes must be a permutation of the input rank")
        if not isinstance(self.copy, bool):
            raise TypeError("transform copy must be a boolean")
        if self.output.spec.dtype != self.input.spec.dtype:
            raise ValueError("transform output dtype must match input dtype")
        if self.output.spec.shape != tuple(self.input.spec.shape[axis] for axis in axes):
            raise ValueError("transform output shape does not match axes")
        if self.output.spec.modes != tuple(self.input.spec.modes[axis] for axis in axes):
            raise ValueError("transform output modes do not match axes")
        if self.copy:
            if self.output.spec.layout not in {"C", "F"}:
                raise ValueError("copied transform output layout must be C or F")
        else:
            expected_layout = _transform_view_layout(self.input.spec, axes)
            if self.output.spec.layout != expected_layout:
                raise ValueError(
                    "transform view output layout must be {!r}".format(expected_layout)
                )
        object.__setattr__(self, "axes", axes)

    @property
    def workspace_outputs(self):
        return (self.output,) if self.copy else ()


def _mode_dimensions(spec):
    return dict(zip(spec.modes, spec.shape))


def _validate_same_dtype(refs, context):
    dtypes = {ref.spec.dtype for ref in refs}
    if len(dtypes) != 1:
        raise ValueError("{} dtypes must match".format(context))


def _validate_pair_step(left, right, output, contracted_modes, batch_modes=()):
    if not all(isinstance(ref, BufferRef) for ref in (left, right, output)):
        raise TypeError("step buffers must be BufferRef instances")
    contracted_modes = _validate_modes(contracted_modes, allow_empty=False)
    batch_modes = _validate_modes(batch_modes)
    _validate_same_dtype((left, right, output), "pair step")

    left_dimensions = _mode_dimensions(left.spec)
    right_dimensions = _mode_dimensions(right.spec)
    output_mode_set = set(output.spec.modes)
    shared_modes = set(left.spec.modes) & set(right.spec.modes)
    for mode in left.spec.modes:
        if mode in shared_modes and left_dimensions[mode] != right_dimensions[mode]:
            raise ValueError("shared mode dimension mismatch for {!r}".format(mode))

    expected_contracted = tuple(
        mode
        for mode in left.spec.modes
        if mode in shared_modes and mode not in output_mode_set
    )
    if contracted_modes != expected_contracted:
        raise ValueError("contracted modes must list all shared non-output modes in left-input order")
    expected_batch = tuple(mode for mode in output.spec.modes if mode in shared_modes)
    if batch_modes != expected_batch:
        raise ValueError("batch modes must list all shared output modes in output order")

    expected_output_modes = (
        set(left.spec.modes) | set(right.spec.modes)
    ) - set(expected_contracted)
    if output_mode_set != expected_output_modes:
        raise ValueError("output modes do not match pair contraction modes")
    dimensions = {**left_dimensions, **right_dimensions}
    expected_shape = tuple(dimensions[mode] for mode in output.spec.modes)
    if output.spec.shape != expected_shape:
        raise ValueError("pair step output shape does not match output mode dimensions")
    left_free = tuple(mode for mode in left.spec.modes if mode not in shared_modes)
    right_free = tuple(mode for mode in right.spec.modes if mode not in shared_modes)
    if left.spec.modes != expected_batch + left_free + expected_contracted:
        raise ValueError("left pair input modes are not in canonical physical order")
    if right.spec.modes != expected_batch + expected_contracted + right_free:
        raise ValueError("right pair input modes are not in canonical physical order")
    if output.spec.modes != expected_batch + left_free + right_free:
        raise ValueError("pair output modes are not in canonical physical order")
    if any(ref.spec.layout != "C" for ref in (left, right, output)):
        raise ValueError("pair step buffers must use C layout")
    return expected_contracted, expected_batch


@dataclass(frozen=True)
class MatmulStep:
    left: BufferRef
    right: BufferRef
    output: BufferRef
    contracted_modes: tuple[str, ...]

    def __post_init__(self):
        contracted, _ = _validate_pair_step(
            self.left, self.right, self.output, self.contracted_modes
        )
        object.__setattr__(self, "contracted_modes", contracted)

    @property
    def workspace_outputs(self):
        return (self.output,)


@dataclass(frozen=True)
class BatchedMatmulStep:
    left: BufferRef
    right: BufferRef
    output: BufferRef
    batch_modes: tuple[str, ...]
    contracted_modes: tuple[str, ...]

    def __post_init__(self):
        contracted, batch = _validate_pair_step(
            self.left, self.right, self.output, self.contracted_modes, self.batch_modes
        )
        shared_output = tuple(
            mode
            for mode in self.output.spec.modes
            if mode in self.left.spec.modes and mode in self.right.spec.modes
        )
        if batch != shared_output:
            raise ValueError("batch modes must list all shared output modes in output order")
        object.__setattr__(self, "contracted_modes", contracted)
        object.__setattr__(self, "batch_modes", batch)

    @property
    def workspace_outputs(self):
        return (self.output,)


@dataclass(frozen=True)
class GroupedMatmulStep:
    operations: tuple[MatmulStep, ...]

    def __post_init__(self):
        operations = _as_tuple(self.operations, "grouped operations")
        if any(not isinstance(step, MatmulStep) for step in operations):
            raise ValueError("grouped operations must contain MatmulStep metadata")
        if len(operations) < 2:
            raise ValueError("grouped matmul requires at least two operations")
        output_keys = tuple(operation.output.key for operation in operations)
        if len(set(output_keys)) != len(output_keys):
            raise ValueError("grouped matmul operations require unique output keys")
        input_keys = {
            ref.key
            for operation in operations
            for ref in (operation.left, operation.right)
        }
        if input_keys & set(output_keys):
            raise ValueError("grouped matmul operations must have independent outputs")
        object.__setattr__(self, "operations", operations)

    @property
    def workspace_outputs(self):
        return tuple(operation.output for operation in self.operations)


@dataclass(frozen=True)
class ReductionStep:
    input: BufferRef
    output: BufferRef
    reduced_modes: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.input, BufferRef) or not isinstance(self.output, BufferRef):
            raise TypeError("reduction buffers must be BufferRef instances")
        reduced = _validate_modes(self.reduced_modes, allow_empty=False)
        if not set(reduced) <= set(self.input.spec.modes):
            raise ValueError("reduced modes must occur in the input")
        if self.output.spec.dtype != self.input.spec.dtype:
            raise ValueError("reduction output dtype must match input dtype")
        remaining = tuple(
            (mode, dim)
            for mode, dim in zip(self.input.spec.modes, self.input.spec.shape)
            if mode not in reduced
        )
        if self.output.spec.modes != tuple(mode for mode, _ in remaining):
            raise ValueError("reduction output modes do not match reduced modes")
        if self.output.spec.shape != tuple(dim for _, dim in remaining):
            raise ValueError("reduction output shape does not match unreduced dimensions")
        object.__setattr__(self, "reduced_modes", reduced)

    @property
    def workspace_outputs(self):
        return (self.output,)


Step = TransformStep | MatmulStep | BatchedMatmulStep | GroupedMatmulStep | ReductionStep


def _hash_value(value):
    if dataclasses.is_dataclass(value):
        return {
            "type": type(value).__name__,
            **{
                field.name: _hash_value(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, tuple):
        return [_hash_value(item) for item in value]
    return value


def _plan_hash(
    operation,
    inputs,
    output,
    steps,
    workspace_bytes,
    planner_source,
    oe_path,
    override_reason,
):
    payload = {
        "operation": operation,
        "inputs": _hash_value(inputs),
        "output": _hash_value(output),
        "steps": _hash_value(steps),
        "workspace_bytes": workspace_bytes,
        "planner_source": planner_source,
        "oe_path": _hash_value(oe_path),
        "override_reason": override_reason,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _step_operations(step):
    return step.operations if isinstance(step, GroupedMatmulStep) else (step,)


def _operation_inputs(operation):
    if isinstance(operation, (MatmulStep, BatchedMatmulStep)):
        return operation.left, operation.right
    return (operation.input,)


def _validate_step_graph(inputs, steps, output):
    available = {ref.key: ref for ref in inputs}
    last_outputs = ()
    for step in steps:
        produced = []
        for operation in _step_operations(step):
            for ref in _operation_inputs(operation):
                if ref.key not in available:
                    raise ValueError("plan step references unknown buffer {!r}".format(ref.key))
                if available[ref.key] != ref:
                    raise ValueError("plan step buffer specification is inconsistent")
            if operation.output.key in available or operation.output.key in {
                ref.key for ref in produced
            }:
                raise ValueError("plan steps produce a duplicate buffer key")
            produced.append(operation.output)
        available.update((ref.key, ref) for ref in produced)
        last_outputs = tuple(produced)
    if output.key not in {ref.key for ref in last_outputs}:
        raise ValueError("final plan step does not produce the declared output")
    if available[output.key] != output:
        raise ValueError("declared output specification is inconsistent")


def _normalize_oe_path(oe_path, input_count):
    oe_path = tuple(_as_tuple(indices, "path indices") for indices in oe_path)
    active = input_count
    for indices in oe_path:
        if len(indices) != 2 or len(set(indices)) != 2:
            raise ValueError("OE path must contain pair contractions")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise TypeError("OE path indices must be integers")
        if any(index < 0 or index >= active for index in indices):
            raise ValueError("OE path index is out of range")
        active -= 1
    if active != 1:
        raise ValueError("OE path does not reduce all inputs")
    return oe_path


def _validate_oe_path_dependencies(inputs, steps, oe_path, output):
    active = [ref.key for ref in inputs]
    path_index = 0
    for step in steps:
        for operation in _step_operations(step):
            if isinstance(operation, (TransformStep, ReductionStep)):
                try:
                    operand_index = active.index(operation.input.key)
                except ValueError as error:
                    raise ValueError(
                        "OE path preprocessing does not match active step dependencies"
                    ) from error
                active[operand_index] = operation.output.key
                continue

            if path_index >= len(oe_path):
                raise ValueError("OE path has fewer pairs than step dependencies")
            try:
                dependency_indices = (
                    active.index(operation.left.key),
                    active.index(operation.right.key),
                )
            except ValueError as error:
                raise ValueError(
                    "OE path pair selection does not match step dependencies"
                ) from error
            path_pair = oe_path[path_index]
            if set(dependency_indices) != set(path_pair):
                raise ValueError("OE path pair selection does not match step dependencies")
            for operand_index in sorted(path_pair, reverse=True):
                active.pop(operand_index)
            active.append(operation.output.key)
            path_index += 1
    if path_index != len(oe_path) or active != [output.key]:
        raise ValueError("OE path does not match pair-step dependency order")


@dataclass(frozen=True)
class ExecutionPlan:
    operation: str
    inputs: tuple[BufferRef, ...]
    output: BufferRef
    steps: tuple[object, ...]
    workspace_bytes: int
    planner_source: str
    oe_path: tuple[tuple[int, ...], ...]
    override_reason: str | None
    plan_hash: str

    def __post_init__(self):
        inputs = _as_tuple(self.inputs, "plan inputs")
        steps = _as_tuple(self.steps, "plan steps")
        if self.operation != "einsum":
            raise ValueError("unsupported plan operation: {!r}".format(self.operation))
        if not inputs or any(not isinstance(ref, BufferRef) for ref in inputs):
            raise TypeError("plan inputs must contain BufferRef instances")
        keys = tuple(ref.key for ref in inputs)
        if len(set(keys)) != len(keys):
            raise ValueError("plan inputs contain duplicate buffer keys")
        if not isinstance(self.output, BufferRef):
            raise TypeError("plan output must be a BufferRef")
        allowed_steps = (TransformStep, MatmulStep, BatchedMatmulStep, GroupedMatmulStep, ReductionStep)
        if not steps or any(not isinstance(step, allowed_steps) for step in steps):
            raise TypeError("plan step metadata has an unsupported type")
        _validate_step_graph(inputs, steps, self.output)
        if isinstance(self.workspace_bytes, bool) or not isinstance(self.workspace_bytes, int):
            raise TypeError("workspace bytes must be an integer")
        if self.workspace_bytes < 0:
            raise ValueError("workspace bytes must be non-negative")
        expected_workspace = workspace_bytes_for_steps(steps, self.output.key)
        if self.workspace_bytes != expected_workspace:
            raise ValueError(
                "workspace bytes must exactly match deterministic step metadata"
            )
        if self.planner_source not in {"opt_einsum", "specialized"}:
            raise ValueError("unsupported planner source")
        if self.planner_source == "specialized" and (
            not isinstance(self.override_reason, str) or not self.override_reason.strip()
        ):
            raise ValueError("specialized plans require a non-empty string override reason")
        if self.planner_source == "opt_einsum" and self.override_reason is not None:
            raise ValueError("opt_einsum plans must not record an override reason")
        oe_path = _normalize_oe_path(self.oe_path, len(inputs))
        if self.planner_source == "opt_einsum":
            _validate_oe_path_dependencies(inputs, steps, oe_path, self.output)
        if not isinstance(self.plan_hash, str) or len(self.plan_hash) != 64:
            raise ValueError("plan hash must be a 64-character hexadecimal digest")
        try:
            int(self.plan_hash, 16)
        except ValueError as error:
            raise ValueError("plan hash must be a 64-character hexadecimal digest") from error
        expected_hash = _plan_hash(
            self.operation,
            inputs,
            self.output,
            steps,
            self.workspace_bytes,
            self.planner_source,
            oe_path,
            self.override_reason,
        )
        if self.plan_hash != expected_hash:
            raise ValueError("plan hash does not match canonical metadata")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "oe_path", oe_path)


@dataclass(frozen=True)
class ExecutionBindings:
    arrays: Mapping[str, Any]

    def __post_init__(self):
        if not isinstance(self.arrays, Mapping):
            raise TypeError("execution bindings must be a mapping")
        copied = dict(self.arrays)
        if len(copied) != len(self.arrays):
            raise ValueError("execution bindings contain duplicate keys")
        if any(not isinstance(key, str) or not key for key in copied):
            raise ValueError("execution binding keys must be non-empty strings")
        object.__setattr__(self, "arrays", MappingProxyType(copied))

    def validate_for(self, plan):
        if not isinstance(plan, ExecutionPlan):
            raise TypeError("bindings can only be validated against an ExecutionPlan")
        expected = {ref.key for ref in plan.inputs}
        actual = set(self.arrays)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing:
            raise ValueError("missing execution binding keys: {}".format(", ".join(missing)))
        if unexpected:
            raise ValueError("unexpected execution binding keys: {}".format(", ".join(unexpected)))


__all__ = [
    "BatchedMatmulStep",
    "BufferRef",
    "ExecutionBindings",
    "ExecutionPlan",
    "GroupedMatmulStep",
    "MatmulStep",
    "ReductionStep",
    "TensorSpec",
    "TransformStep",
]
