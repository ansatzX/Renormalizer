"""Validated local execution for immutable contraction plans."""

import math
from contextlib import nullcontext

import numpy as np

from renormalizer.backend._execution.model import (
    BatchedMatmulStep,
    ExecutionBindings,
    ExecutionPlan,
    GroupedMatmulStep,
    MatmulStep,
    ReductionStep,
    TransformStep,
)


def _array_layout_description(array):
    flags = array.flags
    c_contiguous = bool(flags.c_contiguous)
    f_contiguous = bool(flags.f_contiguous)
    if c_contiguous and f_contiguous:
        return "C/F"
    if c_contiguous:
        return "C"
    if f_contiguous:
        return "F"
    return "strided"


def _matches_layout(array, expected_layout):
    c_contiguous = bool(array.flags.c_contiguous)
    f_contiguous = bool(array.flags.f_contiguous)
    if expected_layout == "C":
        return c_contiguous
    if expected_layout == "F":
        return f_contiguous
    return not c_contiguous and not f_contiguous


def _validate_array(backend, array, ref, context):
    backend._validate_execution_array(array)
    shape = tuple(array.shape)
    if shape != ref.spec.shape:
        raise ValueError(
            "{} shape {} does not match planned shape {}".format(
                context, shape, ref.spec.shape
            )
        )
    dtype = np.dtype(array.dtype).name
    if dtype != ref.spec.dtype:
        raise ValueError(
            "{} dtype {!r} does not match planned dtype {!r}".format(
                context, dtype, ref.spec.dtype
            )
        )
    if not _matches_layout(array, ref.spec.layout):
        raise ValueError(
            "{} layout {!r} does not match planned layout {!r}".format(
                context, _array_layout_description(array), ref.spec.layout
            )
        )


def _workspace_capacity(workspace):
    if workspace is None:
        return None
    if type(workspace) is int:
        capacity = workspace
    else:
        try:
            capacity = workspace.nbytes
        except AttributeError as error:
            raise TypeError(
                "workspace must be a non-negative Python int or expose nbytes"
            ) from error
        if type(capacity) is not int:
            raise TypeError("workspace nbytes must be a Python int")
    if capacity < 0:
        raise ValueError("workspace capacity must be non-negative")
    return capacity


class ExecutionAllocationScope:
    """Retain and publish execution destinations before any primitive launch."""

    def __init__(self, backend, *, owner=None, stream=None):
        from renormalizer.backend._distributed.async_owner import AsyncResourceOwner

        backend._require_execution_usable()
        if owner is not None and not isinstance(owner, AsyncResourceOwner):
            raise TypeError("execution allocation owner must be an AsyncResourceOwner")
        self.backend = backend
        self.owner = owner
        self.stream = stream
        self._arrays = []
        self._resources = []
        self._active = False
        self._failed = False

    @property
    def arrays(self):
        return tuple(self._arrays)

    @property
    def resources(self):
        return tuple(self._resources)

    def __enter__(self):
        if self._active:
            raise RuntimeError("execution allocation scope is already active")
        self.backend._require_execution_usable()
        self._active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._active = False
        if exc_value is not None:
            self._handle_primitive_failure(exc_value)
        if self.owner is None and not self._failed:
            self._arrays.clear()
            self._resources.clear()
        return False

    def _require_active(self):
        if not self._active:
            raise RuntimeError("execution allocation scope is not active")
        self.backend._require_execution_usable()

    def capture(self, array):
        self._require_active()
        if all(retained is not array for retained in self._arrays):
            self._arrays.append(array)
        if self.owner is not None:
            self.owner.capture_arrays(array)
        return array

    def capture_resources(self, *resources):
        self._require_active()
        for resource in resources:
            if resource is not None and all(
                retained is not resource for retained in self._resources
            ):
                self._resources.append(resource)
        if self.owner is not None:
            self.owner.capture_resources(*resources)

    def empty(self, shape, *, dtype, order="C"):
        self._require_active()
        destination = self.backend.empty(shape, dtype=dtype, order=order)
        return self.capture(destination)

    def _quarantine(self, owner):
        from renormalizer.backend._distributed.async_owner import (
            RuntimeTerminalQuarantine,
        )

        quarantine = self.backend._execution_terminal_quarantine
        if quarantine is None:
            quarantine = RuntimeTerminalQuarantine()
            self.backend._execution_terminal_quarantine = quarantine
        quarantine.retain(owner)
        if self.backend._execution_terminal_error is None:
            self.backend._execution_terminal_error = owner.error

    def _handle_primitive_failure(self, error):
        if self.owner is not None or self.backend.name != "cupy" or self._failed:
            return
        from renormalizer.backend._distributed.async_owner import AsyncResourceOwner

        self._failed = True
        owner = AsyncResourceOwner(
            "execution",
            arrays=self._arrays,
            resources=(self, *self._resources),
            drainer=lambda: self.backend._synchronize_execution_stream(self.stream),
            quarantine=self._quarantine,
        )
        owner.mark_enqueued()
        try:
            owner.fail(error)
        except BaseException as retained:
            if retained is not error:
                raise

    def _launch(self, method, *args, destination, **kwargs):
        self.capture(destination)
        try:
            result = method(*args, destination, **kwargs)
        except BaseException as error:
            self._handle_primitive_failure(error)
            raise
        if result is not None and not self.backend._is_exact_execution_destination(
            destination, result
        ):
            error = ValueError("backend primitive replaced its published destination")
            self._handle_primitive_failure(error)
            raise error
        return destination

    def copy_into(self, source, destination):
        return self._launch(
            self.backend._execution_copy_into,
            source,
            destination=destination,
        )

    def sum_into(self, source, destination, *, axis, dtype):
        return self._launch(
            self.backend._execution_sum_into,
            source,
            destination=destination,
            axis=axis,
            dtype=dtype,
        )

    def conjugate_into(self, source, destination):
        return self._launch(
            self.backend._execution_conjugate_into,
            source,
            destination=destination,
        )

    def multiply_into(self, left, right, destination):
        return self._launch(
            self.backend._execution_multiply_into,
            left,
            right,
            destination=destination,
        )

    def add_into(self, left, right, destination):
        return self._launch(
            self.backend._execution_add_into,
            left,
            right,
            destination=destination,
        )

    def matmul_into(self, left, right, destination, *, workspace=None, batched=False):
        method = (
            self.backend._execution_batched_matmul_into
            if batched
            else self.backend._execution_matmul_into
        )
        return self._launch(
            method,
            left,
            right,
            destination=destination,
            workspace=workspace,
        )


def _store_output(backend, buffers, ref, value):
    if ref.key in buffers:
        raise ValueError(
            "plan attempted to replace declared buffer {!r}".format(ref.key)
        )
    _validate_array(backend, value, ref, "step output {!r}".format(ref.key))
    buffers[ref.key] = value


def _exact_reshape_view(backend, value, shape):
    if tuple(value.shape) == tuple(shape):
        return value
    shape = tuple(shape)
    if _product(shape) != int(value.size):
        raise ValueError("planned contraction reshape has incompatible elements")
    if not bool(value.flags.c_contiguous):
        raise ValueError("planned contraction reshape requires C-contiguous storage")
    if int(value.size) and any(stride <= 0 for stride in value.strides):
        raise ValueError("planned contraction reshape has invalid source strides")
    result = backend.reshape(value, shape)
    if not backend._is_exact_execution_reshape(value, result):
        raise ValueError("planned contraction reshape unexpectedly copied data")
    if tuple(result.shape) != shape or result.dtype != value.dtype:
        raise ValueError("planned contraction reshape changed array metadata")
    return result


def _execute_transform(backend, step, buffers, scope):
    value = backend.transpose(buffers[step.input.key], axes=step.axes)
    if step.copy:
        destination = scope.empty(
            step.output.spec.shape,
            dtype=np.dtype(step.output.spec.dtype),
            order=step.output.spec.layout,
        )
        value = scope.copy_into(value, destination)
    _store_output(backend, buffers, step.output, value)


def _execute_reduction(backend, step, buffers, scope):
    axes = tuple(step.input.spec.modes.index(mode) for mode in step.reduced_modes)
    value = scope.empty(
        step.output.spec.shape,
        dtype=np.dtype(step.output.spec.dtype),
        order=step.output.spec.layout,
    )
    scope.sum_into(
        buffers[step.input.key],
        value,
        axis=axes,
        dtype=np.dtype(step.output.spec.dtype),
    )
    _store_output(backend, buffers, step.output, value)


def _product(values):
    return math.prod(values)


def _execute_matmul(backend, step, buffers, workspace, scope):
    left, right = _matmul_views(backend, step, buffers)
    matrix_shape = left.shape[:-2] + (left.shape[-2], right.shape[-1])
    value = scope.empty(matrix_shape, dtype=left.dtype, order="C")
    value = scope.matmul_into(
        left,
        right,
        value,
        workspace=workspace,
        batched=isinstance(step, BatchedMatmulStep),
    )
    value = _exact_reshape_view(backend, value, step.output.spec.shape)
    _store_output(backend, buffers, step.output, value)


def _matmul_views(backend, step, buffers):
    batch_count = len(step.batch_modes) if isinstance(step, BatchedMatmulStep) else 0
    contracted_count = len(step.contracted_modes)
    left_shape = step.left.spec.shape
    right_shape = step.right.spec.shape
    batch_shape = left_shape[:batch_count]
    left_free_shape = left_shape[batch_count : len(left_shape) - contracted_count]
    contracted_shape = left_shape[len(left_shape) - contracted_count :]
    right_free_shape = right_shape[batch_count + contracted_count :]
    left = _exact_reshape_view(
        backend,
        buffers[step.left.key],
        batch_shape + (_product(left_free_shape), _product(contracted_shape)),
    )
    right = _exact_reshape_view(
        backend,
        buffers[step.right.key],
        batch_shape + (_product(contracted_shape), _product(right_free_shape)),
    )
    return left, right


def _execute_grouped_matmul(backend, step, buffers, workspace, scope):
    from renormalizer.backend._gemm.descriptors import MatmulDesc
    from renormalizer.backend._gemm.executor import execute_grouped_gemm

    descriptors = []
    matrices = {}
    prepared = []
    for index, operation in enumerate(step.operations):
        left, right = _matmul_views(backend, operation, buffers)
        a_key = "grouped_a_{}".format(index)
        b_key = "grouped_b_{}".format(index)
        c_key = "grouped_c_{}".format(index)
        matrices[a_key] = left
        matrices[b_key] = right
        descriptors.append(
            MatmulDesc(
                a_key,
                b_key,
                c_key,
                left.shape[0],
                right.shape[1],
                left.shape[1],
            )
        )
        prepared.append(operation)

    matrix_outputs = execute_grouped_gemm(
        backend,
        tuple(descriptors),
        matrices,
        workspace=workspace,
        policy="execution_ir",
        _allocation_scope=scope,
    )
    outputs = []
    for operation, matrix in zip(prepared, matrix_outputs):
        if operation.output.key in buffers:
            raise ValueError(
                "plan attempted to replace declared buffer {!r}".format(
                    operation.output.key
                )
            )
        value = _exact_reshape_view(backend, matrix, operation.output.spec.shape)
        _validate_array(
            backend,
            value,
            operation.output,
            "step output {!r}".format(operation.output.key),
        )
        outputs.append((operation.output.key, value))
    buffers.update(outputs)


def execute_plan(
    backend,
    plan,
    bindings,
    *,
    stream=None,
    workspace=None,
    _allocation_scope=None,
):
    """Execute a trusted plan; ``workspace`` is a capacity bound, not reused storage."""
    backend._require_execution_usable()
    if not getattr(backend, "supports_execution_ir", False):
        raise NotImplementedError(
            "backend {!r} does not support execution IR".format(
                getattr(backend, "name", type(backend).__name__)
            )
        )
    if not isinstance(plan, ExecutionPlan):
        raise TypeError("plan must be an ExecutionPlan")
    if not isinstance(bindings, ExecutionBindings):
        raise TypeError("bindings must be ExecutionBindings")
    if _allocation_scope is not None and not isinstance(
        _allocation_scope, ExecutionAllocationScope
    ):
        raise TypeError("internal allocation scope is invalid")
    capacity = _workspace_capacity(workspace)
    bindings.validate_for(plan)
    backend._validate_execution_stream(stream)
    for ref in plan.inputs:
        _validate_array(
            backend, bindings.arrays[ref.key], ref, "binding {!r}".format(ref.key)
        )
    if capacity is not None and plan.workspace_bytes > capacity:
        raise ValueError(
            "plan workspace capacity {} exceeds supplied capacity {}".format(
                plan.workspace_bytes, capacity
            )
        )
    buffers = dict(bindings.arrays)
    with backend._execution_context(stream):
        created_scope = _allocation_scope is None
        scope = (
            ExecutionAllocationScope(backend, stream=stream)
            if created_scope
            else _allocation_scope
        )
        if scope.backend is not backend:
            raise ValueError("execution allocation scope backend does not match")
        scope_context = scope if created_scope else nullcontext(scope)
        with scope_context:
            scope.capture_resources(bindings, buffers)
            for array in buffers.values():
                scope.capture(array)
            for step in plan.steps:
                if isinstance(step, TransformStep):
                    _execute_transform(backend, step, buffers, scope)
                elif isinstance(step, ReductionStep):
                    _execute_reduction(backend, step, buffers, scope)
                elif isinstance(step, (MatmulStep, BatchedMatmulStep)):
                    _execute_matmul(backend, step, buffers, workspace, scope)
                elif isinstance(step, GroupedMatmulStep):
                    _execute_grouped_matmul(backend, step, buffers, workspace, scope)
                else:
                    raise TypeError("unsupported execution step type")
    return buffers[plan.output.key]


__all__ = ["execute_plan"]
