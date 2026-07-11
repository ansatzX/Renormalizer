"""Lower explicit einsum equations into immutable execution metadata."""

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import opt_einsum as oe

from renormalizer.backend._execution.model import (
    BatchedMatmulStep,
    BufferRef,
    ExecutionBindings,
    ExecutionPlan,
    MatmulStep,
    ReductionStep,
    TensorSpec,
    TransformStep,
    _plan_hash,
)
from renormalizer.backend._execution.workspace import workspace_bytes_for_steps


_TERM_PATTERN = re.compile(r"[A-Za-z]*\Z")


@dataclass(frozen=True)
class _EinsumPathMetadata:
    equation: str
    shapes: tuple
    optimize: object
    oe_path: tuple
    contraction_list: tuple
    fingerprint: str


def _parse_equation(equation):
    if not isinstance(equation, str):
        raise TypeError("einsum equation must be a string")
    equation = "".join(equation.split())
    if equation.count("->") != 1:
        raise ValueError("einsum equation must have an explicit output")
    input_text, output_text = equation.split("->")
    input_terms = input_text.split(",")
    if not input_terms or any(_TERM_PATTERN.fullmatch(term) is None for term in input_terms):
        raise ValueError("einsum input modes must use explicit alphabetic labels")
    if _TERM_PATTERN.fullmatch(output_text) is None:
        raise ValueError("einsum output modes must use explicit alphabetic labels")
    if any(len(set(term)) != len(term) for term in input_terms):
        raise ValueError("repeated modes within an operand are unsupported")
    if len(set(output_text)) != len(output_text):
        raise ValueError("output modes must be unique")
    all_input_modes = set("".join(input_terms))
    if not set(output_text) <= all_input_modes:
        raise ValueError("output modes must occur in an input")
    return equation, tuple(tuple(term) for term in input_terms), tuple(output_text)


def _validate_shapes(input_modes, shapes):
    try:
        shapes = tuple(tuple(shape) for shape in shapes)
    except TypeError as error:
        raise TypeError("operand shapes must be iterable") from error
    if len(shapes) != len(input_modes):
        raise ValueError("operand count does not match the einsum equation")
    mode_sizes = {}
    normalized = []
    for operand_index, (modes, shape) in enumerate(zip(input_modes, shapes)):
        if len(shape) != len(modes):
            raise ValueError("operand {} rank does not match equation modes".format(operand_index))
        clean_shape = []
        for mode, dim in zip(modes, shape):
            if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)):
                raise TypeError("shape dimensions must be integers")
            dim = int(dim)
            if dim < 0:
                raise ValueError("shape dimensions must be non-negative")
            previous = mode_sizes.setdefault(mode, dim)
            if previous != dim:
                raise ValueError("inconsistent dimension for mode {!r}".format(mode))
            clean_shape.append(dim)
        normalized.append(tuple(clean_shape))
    return tuple(normalized), mode_sizes


def _shape_for_modes(modes, mode_sizes):
    return tuple(mode_sizes[mode] for mode in modes)


def _new_ref(key, modes, mode_sizes, dtype, layout="C"):
    return BufferRef(key, TensorSpec(_shape_for_modes(modes, mode_sizes), dtype, layout, modes))


def _axes_for_modes(current_modes, target_modes):
    return tuple(current_modes.index(mode) for mode in target_modes)


def _canonical_optimizer(optimize):
    if isinstance(optimize, str):
        return optimize
    if not isinstance(optimize, (tuple, list)):
        raise TypeError(
            "reusable path metadata optimizer must be a strategy string or "
            "an explicit integer path"
        )
    canonical = []
    try:
        for step in optimize:
            step = tuple(step)
            if any(
                isinstance(index, bool)
                or not isinstance(index, (int, np.integer))
                for index in step
            ):
                raise TypeError
            canonical.append(tuple(int(index) for index in step))
    except TypeError as error:
        raise TypeError(
            "reusable path metadata explicit optimizer must contain integer steps"
        ) from error
    return tuple(canonical)


def _freeze_metadata_value(value):
    if isinstance(value, np.integer):
        return int(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_metadata_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_metadata_value(item) for item in value)
    raise TypeError(
        "opt_einsum contraction metadata contains unsupported value {!r}".format(
            type(value).__name__
        )
    )


def _fingerprint_value(value):
    if isinstance(value, tuple):
        return ("tuple", tuple(_fingerprint_value(item) for item in value))
    if isinstance(value, frozenset):
        items = tuple(
            sorted(
                (_fingerprint_value(item) for item in value),
                key=repr,
            )
        )
        return ("set", items)
    return (type(value).__name__, value)


def _path_metadata_fingerprint(
    equation, shapes, optimize, oe_path, contraction_list
):
    canonical = _fingerprint_value(
        (equation, shapes, optimize, oe_path, contraction_list)
    )
    return hashlib.sha256(repr(canonical).encode("utf-8")).hexdigest()


def _validate_path_metadata(metadata, equation, shapes, optimize):
    if not isinstance(metadata, _EinsumPathMetadata):
        raise TypeError("path metadata must come from _resolve_einsum_path")
    if metadata.equation != equation:
        raise ValueError("path metadata equation does not match lowering equation")
    if metadata.shapes != shapes:
        raise ValueError("path metadata shapes do not match lowering shapes")
    canonical_optimize = _canonical_optimizer(optimize)
    if metadata.optimize != canonical_optimize:
        raise ValueError("path metadata optimizer does not match lowering optimizer")
    if not isinstance(metadata.oe_path, tuple) or any(
        not isinstance(step, tuple)
        or any(not isinstance(index, int) for index in step)
        for step in metadata.oe_path
    ):
        raise ValueError("path metadata path must contain immutable integer steps")
    if not isinstance(metadata.contraction_list, tuple) or any(
        not isinstance(contraction, tuple) or len(contraction) != 5
        for contraction in metadata.contraction_list
    ):
        raise ValueError("path metadata contraction list is malformed")
    if len(metadata.contraction_list) != len(metadata.oe_path):
        raise ValueError("path metadata has inconsistent path and contraction lengths")
    expected_fingerprint = _path_metadata_fingerprint(
        metadata.equation,
        metadata.shapes,
        metadata.optimize,
        metadata.oe_path,
        metadata.contraction_list,
    )
    if metadata.fingerprint != expected_fingerprint:
        raise ValueError("path metadata integrity fingerprint does not match content")
    return metadata


def _search_einsum_path(equation, shapes, optimize):
    try:
        raw_path, path_info = oe.contract_path(
            equation, *shapes, shapes=True, optimize=optimize
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "opt_einsum rejected contraction metadata: {}".format(error)
        ) from error
    try:
        oe_path = tuple(tuple(int(index) for index in step) for step in raw_path)
        contraction_list = tuple(
            tuple(_freeze_metadata_value(value) for value in contraction)
            for contraction in path_info.contraction_list
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("opt_einsum returned malformed path metadata") from error
    if len(contraction_list) != len(oe_path):
        raise ValueError("opt_einsum returned inconsistent path metadata")
    return oe_path, contraction_list


def _resolve_einsum_path(equation, shapes, optimize="optimal"):
    equation, input_modes, _ = _parse_equation(equation)
    shapes, _ = _validate_shapes(input_modes, shapes)
    canonical_optimize = _canonical_optimizer(optimize)
    oe_path, contraction_list = _search_einsum_path(equation, shapes, optimize)
    fingerprint = _path_metadata_fingerprint(
        equation, shapes, canonical_optimize, oe_path, contraction_list
    )
    metadata = _EinsumPathMetadata(
        equation=equation,
        shapes=shapes,
        optimize=canonical_optimize,
        oe_path=oe_path,
        contraction_list=contraction_list,
        fingerprint=fingerprint,
    )
    return _validate_path_metadata(metadata, equation, shapes, optimize)


def lower_einsum_path(
    equation,
    shapes,
    dtype="float64",
    optimize="optimal",
    *,
    layouts=None,
    _path_metadata=None,
):
    equation, input_modes, declared_output_modes = _parse_equation(equation)
    shapes, mode_sizes = _validate_shapes(input_modes, shapes)
    dtype = TensorSpec((), dtype, "C", ()).dtype
    if layouts is None:
        layouts = ("C",) * len(input_modes)
    elif isinstance(layouts, str):
        raise TypeError("operand layouts must be an iterable of layout names")
    else:
        try:
            layouts = tuple(layouts)
        except TypeError as error:
            raise TypeError("operand layouts must be iterable") from error
    if len(layouts) != len(input_modes):
        raise ValueError("operand layout count does not match the einsum equation")
    inputs = tuple(
        BufferRef(
            "input_{}".format(index), TensorSpec(shape, dtype, layout, modes)
        )
        for index, (shape, modes, layout) in enumerate(
            zip(shapes, input_modes, layouts)
        )
    )
    if _path_metadata is None:
        oe_path, contraction_list = _search_einsum_path(
            equation, shapes, optimize
        )
    else:
        path_metadata = _validate_path_metadata(
            _path_metadata, equation, shapes, optimize
        )
        oe_path = path_metadata.oe_path
        contraction_list = path_metadata.contraction_list

    active = [(ref, ref.spec.modes) for ref in inputs]
    steps = []
    temporary_index = 0

    def new_temporary(modes):
        nonlocal temporary_index
        ref = _new_ref(
            "temporary_{}".format(temporary_index), modes, mode_sizes, dtype
        )
        temporary_index += 1
        return ref

    def pack_operand(ref, target_modes):
        if ref.spec.modes == target_modes and ref.spec.layout == "C":
            return ref
        packed = new_temporary(target_modes)
        steps.append(
            TransformStep(
                ref,
                packed,
                _axes_for_modes(ref.spec.modes, target_modes),
                True,
            )
        )
        return packed

    for path_step, contraction in zip(oe_path, contraction_list):
        if len(path_step) != 2 or len(set(path_step)) != 2:
            raise ValueError("unsupported opt_einsum path: only pair contractions can be lowered")
        if any(index < 0 or index >= len(active) for index in path_step):
            raise ValueError("opt_einsum path index is out of range")
        contraction_indices, _, step_equation, _, _ = contraction
        contraction_indices = tuple(int(index) for index in contraction_indices)
        if len(contraction_indices) != 2 or set(contraction_indices) != set(path_step):
            raise ValueError("opt_einsum path and contraction metadata disagree")
        _, step_input_modes, step_output_modes = _parse_equation(step_equation)
        selected = [active[index] for index in path_step]
        unmatched_modes = list(step_input_modes)
        for _, modes in selected:
            if modes not in unmatched_modes:
                raise ValueError(
                    "opt_einsum contraction modes disagree with active operands"
                )
            unmatched_modes.remove(modes)
        if unmatched_modes:
            raise ValueError("opt_einsum contraction modes disagree with active operands")

        pair_refs = [selected[0][0], selected[1][0]]
        pair_modes = [selected[0][1], selected[1][1]]
        output_mode_set = set(step_output_modes)
        for side in range(2):
            other_modes = set(pair_modes[1 - side])
            unique_reduced = tuple(
                mode
                for mode in pair_modes[side]
                if mode not in output_mode_set and mode not in other_modes
            )
            if unique_reduced:
                remaining_modes = tuple(
                    mode for mode in pair_modes[side] if mode not in unique_reduced
                )
                reduced_ref = new_temporary(remaining_modes)
                reduction = ReductionStep(
                    pair_refs[side], reduced_ref, unique_reduced
                )
                steps.append(reduction)
                pair_refs[side] = reduced_ref
                pair_modes[side] = reduced_ref.spec.modes

        shared = set(pair_modes[0]) & set(pair_modes[1])
        contracted_modes = tuple(
            mode for mode in pair_modes[0] if mode in shared and mode not in output_mode_set
        )
        if not contracted_modes:
            raise ValueError(
                "unsupported pair contraction without a contracted mode; no GEMM metadata was fabricated"
            )
        batch_modes = tuple(
            mode for mode in step_output_modes if mode in pair_modes[0] and mode in pair_modes[1]
        )
        left_free = tuple(mode for mode in pair_modes[0] if mode not in shared)
        right_free = tuple(mode for mode in pair_modes[1] if mode not in shared)
        left_modes = batch_modes + left_free + contracted_modes
        right_modes = batch_modes + contracted_modes + right_free
        pair_refs[0] = pack_operand(pair_refs[0], left_modes)
        pair_refs[1] = pack_operand(pair_refs[1], right_modes)
        natural_output_modes = batch_modes + left_free + right_free
        is_final = len(active) == 2
        needs_reorder = natural_output_modes != step_output_modes
        if needs_reorder:
            pair_output = new_temporary(natural_output_modes)
        elif is_final:
            pair_output = _new_ref("output", natural_output_modes, mode_sizes, dtype)
        else:
            pair_output = new_temporary(natural_output_modes)
        if batch_modes:
            pair_step = BatchedMatmulStep(
                pair_refs[0], pair_refs[1], pair_output, batch_modes, contracted_modes
            )
        else:
            pair_step = MatmulStep(
                pair_refs[0], pair_refs[1], pair_output, contracted_modes
            )
        steps.append(pair_step)

        if needs_reorder:
            if is_final:
                output_ref = _new_ref("output", step_output_modes, mode_sizes, dtype)
            else:
                output_ref = new_temporary(step_output_modes)
            steps.append(
                TransformStep(
                    pair_output,
                    output_ref,
                    _axes_for_modes(natural_output_modes, step_output_modes),
                    True,
                )
            )
        else:
            output_ref = pair_output

        for index in sorted(path_step, reverse=True):
            active.pop(index)
        active.append((output_ref, step_output_modes))

    if len(active) != 1 or active[0][1] != declared_output_modes:
        raise ValueError("lowered path does not produce the declared output modes")
    output = active[0][0]
    if output.key != "output":
        raise ValueError("lowered path did not produce a final output buffer")
    workspace_bytes = workspace_bytes_for_steps(tuple(steps), output.key)
    planner_source = "opt_einsum"
    override_reason = None
    plan_hash = _plan_hash(
        "einsum", inputs, output, tuple(steps), workspace_bytes,
        planner_source, oe_path, override_reason,
    )
    return ExecutionPlan(
        operation="einsum",
        inputs=inputs,
        output=output,
        steps=tuple(steps),
        workspace_bytes=workspace_bytes,
        planner_source=planner_source,
        oe_path=oe_path,
        override_reason=override_reason,
        plan_hash=plan_hash,
    )


def _array_layout(array):
    flags = getattr(array, "flags", None)
    if flags is not None and bool(getattr(flags, "c_contiguous", False)):
        return "C"
    if flags is not None and bool(getattr(flags, "f_contiguous", False)):
        return "F"
    return "strided"


def plan_einsum(equation, arrays, optimize="optimal"):
    if not isinstance(arrays, Mapping):
        raise TypeError("einsum arrays must be an ordered mapping")
    items = tuple(arrays.items())
    if any(not isinstance(key, str) or not key for key, _ in items):
        raise ValueError("source binding keys must be non-empty strings")
    _, input_modes, _ = _parse_equation(equation)
    if len(items) != len(input_modes):
        raise ValueError("operand count does not match the einsum equation")
    shapes = []
    dtypes = []
    layouts = []
    for index, (_, array) in enumerate(items):
        if not hasattr(array, "shape") or not hasattr(array, "dtype"):
            raise TypeError("operand {} is missing shape or dtype metadata".format(index))
        shapes.append(tuple(array.shape))
        dtypes.append(TensorSpec((), array.dtype, "C", ()).dtype)
        layouts.append(_array_layout(array))
    if len(set(dtypes)) != 1:
        raise ValueError("all operands must have the same dtype for execution planning")
    plan = lower_einsum_path(
        equation,
        tuple(shapes),
        dtypes[0],
        optimize=optimize,
        layouts=tuple(layouts),
    )
    bindings = ExecutionBindings(
        {ref.key: array for ref, (_, array) in zip(plan.inputs, items)}
    )
    bindings.validate_for(plan)
    return plan, bindings


__all__ = ["lower_einsum_path", "plan_einsum"]
