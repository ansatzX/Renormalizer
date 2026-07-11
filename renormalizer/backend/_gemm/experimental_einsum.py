"""Execution-IR adapters for explicit opt_einsum expressions."""

import itertools
import string
import threading
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from renormalizer.backend._execution.model import ExecutionBindings
from renormalizer.backend._execution.planner import (
    _resolve_einsum_path,
    lower_einsum_path,
)
from renormalizer.cons import backend
from renormalizer.utils import profiling


_STEP_NAMES = {
    "TransformStep": "transform",
    "ReductionStep": "reduction",
    "MatmulStep": "matmul",
    "BatchedMatmulStep": "batched_matmul",
    "GroupedMatmulStep": "grouped_matmul",
}

_SUPPORTED_DTYPES = tuple(
    np.dtype(dtype) for dtype in ("float32", "float64", "complex64", "complex128")
)
_VARIABLE_LAYOUTS = ("C", "F", "strided")
_MAX_EXECUTION_VARIANTS = 48
_EXECUTION_PLAN_SELECTOR = "key=(numpy_result_dtype_name, variable_layout_tuple)"


class UnsupportedRuntimeDtypeError(NotImplementedError):
    """The runtime result dtype is outside the construction-time plan set."""


@dataclass(frozen=True)
class ResolvedExecutionArtifact:
    execution_plan: object
    source_bindings: ExecutionBindings
    variable_key: str
    variable_index: int
    variable_array: object


def _ascii_equation(equation):
    if not isinstance(equation, str):
        raise NotImplementedError("execution IR requires a string einsum equation")
    equation = "".join(equation.split())
    if "..." in equation or equation.count("->") != 1:
        raise NotImplementedError(
            "execution IR requires an explicit einsum equation without ellipses"
        )
    symbols = []
    for character in equation.replace("->", "").replace(",", ""):
        if character not in symbols:
            symbols.append(character)
    if len(symbols) > len(string.ascii_letters):
        raise NotImplementedError(
            "execution IR supports at most {} distinct einsum modes".format(
                len(string.ascii_letters)
            )
        )
    translation = str.maketrans(dict(zip(symbols, string.ascii_letters)))
    return equation.translate(translation)


def _array_layout(value):
    flags = getattr(value, "flags", None)
    if flags is not None and bool(getattr(flags, "c_contiguous", False)):
        return "C"
    if flags is not None and bool(getattr(flags, "f_contiguous", False)):
        return "F"
    return "strided"


def _normalize_shape(shape, index):
    try:
        normalized = tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError) as error:
        raise NotImplementedError(
            "execution IR variable operand {} requires an explicit shape".format(index)
        ) from error
    if any(dimension < 0 for dimension in normalized):
        raise NotImplementedError("execution IR operand shapes must be non-negative")
    return normalized


def _is_execution_array(selected, value):
    try:
        selected._validate_execution_array(value)
    except (TypeError, ValueError):
        return False
    return True


def _variant_description(dtype, layouts):
    return "{}/{}".format(dtype, "+".join(layouts) if layouts else "-")


def _unsupported_variant(dtype, layouts, variants):
    available = ", ".join(
        _variant_description(variant_dtype, variant_layouts)
        for variant_dtype, variant_layouts in sorted(variants)
    )
    return UnsupportedRuntimeDtypeError(
        "execution IR runtime result dtype {!r} with layouts {!r} has no "
        "configured variant; available variants: {}".format(
            dtype, layouts, available
        )
    )


def _build_ir_expression(
    equation,
    operand_specs,
    constant_indices,
    *,
    network,
    center_kind,
    optimize,
    lowerer,
):
    selected = backend.current
    if not selected.supports_execution_ir:
        raise NotImplementedError(
            "backend {!r} does not support execution IR; select fallback_policy='legacy_oe' "
            "for compatibility".format(selected.name)
        )

    operand_specs = tuple(operand_specs)
    constant_indices = tuple(int(index) for index in constant_indices)
    if len(set(constant_indices)) != len(constant_indices) or any(
        index < 0 or index >= len(operand_specs) for index in constant_indices
    ):
        raise NotImplementedError("execution IR received invalid constant operands")
    constant_set = set(constant_indices)
    raw_constants = {
        index: operand_specs[index]
        for index in constant_indices
    }
    if not raw_constants:
        raise NotImplementedError(
            "experimental execution IR requires at least one constant operand to fix dtype"
        )
    if any(
        not hasattr(value, "shape") or not hasattr(value, "dtype")
        for value in raw_constants.values()
    ):
        raise NotImplementedError("execution IR constants must be backend arrays")
    raw_constant_dtypes = tuple(
        value.dtype for value in raw_constants.values()
    )
    variable_shapes = {}
    variable_indices = []
    for index, spec in enumerate(operand_specs):
        if index not in constant_set:
            variable_shapes[index] = _normalize_shape(spec, index)
            variable_indices.append(index)

    candidate_variable_dtypes = (
        _SUPPORTED_DTYPES
        if variable_indices
        else (np.result_type(*raw_constant_dtypes),)
    )
    supported_dtype_names = {dtype.name for dtype in _SUPPORTED_DTYPES}
    result_dtypes = []
    for variable_dtype in candidate_variable_dtypes:
        dtype = np.result_type(*raw_constant_dtypes, variable_dtype).name
        if dtype in supported_dtype_names and dtype not in result_dtypes:
            result_dtypes.append(dtype)
    if not result_dtypes:
        raise NotImplementedError(
            "execution IR constants do not produce a supported float32, float64, "
            "complex64, or complex128 result dtype"
        )
    variant_count = len(result_dtypes)
    for _ in variable_indices:
        variant_count *= len(_VARIABLE_LAYOUTS)
        if variant_count > _MAX_EXECUTION_VARIANTS:
            raise NotImplementedError(
                "{} execution variants exceeds bounded limit {}".format(
                    variant_count, _MAX_EXECUTION_VARIANTS
                )
            )
    layout_variants = tuple(
        itertools.product(_VARIABLE_LAYOUTS, repeat=len(variable_indices))
    )

    equation = _ascii_equation(equation)
    shapes = tuple(
        tuple(int(dimension) for dimension in raw_constants[index].shape)
        if index in constant_set
        else variable_shapes[index]
        for index in range(len(operand_specs))
    )
    try:
        path_metadata = _resolve_einsum_path(
            equation, shapes, optimize=optimize
        )
    except (TypeError, ValueError) as error:
        raise NotImplementedError(
            "execution IR could not resolve the opt_einsum path: {}".format(error)
        ) from error
    variants = {}
    for dtype in result_dtypes:
        constant_layouts = {
            index: (
                _array_layout(value)
                if np.dtype(value.dtype).name == dtype
                and _is_execution_array(selected, value)
                else "C"
            )
            for index, value in raw_constants.items()
        }
        for variable_layouts in layout_variants:
            variable_layout_by_index = dict(zip(variable_indices, variable_layouts))
            layouts = []
            for index in range(len(operand_specs)):
                if index in constant_set:
                    layouts.append(constant_layouts[index])
                else:
                    layouts.append(variable_layout_by_index[index])
            try:
                plan = lowerer(
                    equation,
                    shapes,
                    dtype=dtype,
                    optimize=optimize,
                    layouts=tuple(layouts),
                    _path_metadata=path_metadata,
                )
            except (TypeError, ValueError) as error:
                raise NotImplementedError(
                    "execution IR could not lower the opt_einsum path: {}".format(
                        error
                    )
                ) from error
            variants[(dtype, variable_layouts)] = plan

    constant_cache = {}
    constant_cache_lock = threading.Lock()

    def constants_for(dtype):
        cached = constant_cache.get(dtype)
        if cached is not None:
            return cached
        with constant_cache_lock:
            cached = constant_cache.get(dtype)
            if cached is None:
                target_dtype = np.dtype(dtype)
                cached = {
                    index: (
                        value
                        if np.dtype(value.dtype).name == dtype
                        and _is_execution_array(selected, value)
                        else selected.array(
                            value, dtype=target_dtype, copy=True, order="C"
                        )
                    )
                    for index, value in raw_constants.items()
                }
                constant_cache[dtype] = cached
        return cached

    def prepare(variable_operands):
        if len(variable_operands) != len(variable_indices):
            raise TypeError(
                "execution IR expression expected {} variable operands, got {}".format(
                    len(variable_indices), len(variable_operands)
                )
            )
        try:
            dtype = np.result_type(
                *raw_constant_dtypes,
                *(value.dtype for value in variable_operands),
            ).name
        except (AttributeError, TypeError, ValueError) as error:
            raise UnsupportedRuntimeDtypeError(
                "execution IR runtime operands require supported backend real/complex dtypes"
            ) from error
        prepared_variables = []
        for index, value in zip(variable_indices, variable_operands):
            try:
                value_dtype = np.dtype(value.dtype)
            except (AttributeError, TypeError, ValueError) as error:
                raise UnsupportedRuntimeDtypeError(
                    "execution IR runtime operands require supported backend "
                    "real/complex dtypes"
                ) from error
            if np.dtype(value.dtype).name != dtype:
                if not np.can_cast(value_dtype, np.dtype(dtype), casting="safe"):
                    raise UnsupportedRuntimeDtypeError(
                        "execution IR variable dtype {!r} cannot be represented by "
                        "result dtype {!r}".format(value_dtype.name, dtype)
                    )
                value = selected.asarray(value, dtype=np.dtype(dtype))
            elif not _is_execution_array(selected, value):
                value = selected.asarray(value, dtype=np.dtype(dtype))
            prepared_variables.append((index, value))

        variable_layouts = tuple(
            _array_layout(value) for _, value in prepared_variables
        )
        try:
            plan = variants[(dtype, variable_layouts)]
        except KeyError as error:
            raise _unsupported_variant(dtype, variable_layouts, variants) from error
        constants = constants_for(dtype)
        ordered = [None] * len(operand_specs)
        for index, value in constants.items():
            ordered[index] = value
        for index, value in prepared_variables:
            ordered[index] = value
        bindings = ExecutionBindings(
            {
                ref.key: value
                for ref, value in zip(plan.inputs, ordered)
            }
        )
        return plan, tuple(ordered), bindings, tuple(prepared_variables)

    def resolve_execution_artifact(variable_operand):
        if len(variable_indices) != 1:
            raise NotImplementedError(
                "distributed local H-v requires exactly one variable operand"
            )
        plan, _, bindings, prepared_variables = prepare((variable_operand,))
        variable_index, variable_array = prepared_variables[0]
        variable_key = plan.inputs[variable_index].key
        return ResolvedExecutionArtifact(
            execution_plan=plan,
            source_bindings=ExecutionBindings(
                {
                    key: value
                    for key, value in bindings.arrays.items()
                    if key != variable_key
                }
            ),
            variable_key=variable_key,
            variable_index=variable_index,
            variable_array=variable_array,
        )

    def expression(*variable_operands):
        plan, ordered, bindings, _ = prepare(variable_operands)
        if network is None or not profiling.enabled():
            return selected.execute_plan(plan, bindings)

        from time import perf_counter

        started = perf_counter()
        result = selected.execute_plan(plan, bindings)
        wall_s = perf_counter() - started

        from renormalizer.backend._execution.profiling import (
            completion_timing_metadata,
            local_hv_payload,
            oe_path_identity,
        )

        device = selected.current_device()
        timing = completion_timing_metadata(selected.name, device)
        payload = local_hv_payload(
            network=network,
            center_kind=center_kind,
            input_shapes=tuple(
                tuple(int(dimension) for dimension in value.shape)
                for value in ordered
            ),
            output_shape=tuple(int(dimension) for dimension in result.shape),
            requested_policy="execution_ir",
            actual_policy="execution_ir",
            planner_source=plan.planner_source,
            oe_path_hash=oe_path_identity(plan.oe_path),
            actual_steps=tuple(
                _STEP_NAMES[type(step).__name__] for step in plan.steps
            ),
            wall_s=wall_s,
            path_override_reason=plan.override_reason,
            fallback_reason=None,
            backend=selected.name,
            device=device,
            **timing,
        )
        profiling.record("local_hv_execute", **payload)
        return result

    expression.execution_plans = MappingProxyType(dict(variants))
    expression.execution_plan_selector = _EXECUTION_PLAN_SELECTOR
    expression.resolve_execution_artifact = resolve_execution_artifact
    expression.resolved_oe_path = path_metadata.oe_path
    if len(variants) == 1:
        expression.execution_plan = next(iter(variants.values()))
    return expression


def build_experimental_einsum(
    args, kwargs, *, profile_network=None, profile_center_kind=None
):
    """Attempt execution-IR translation of one opt_einsum expression."""
    args = tuple(args)
    if not args:
        raise NotImplementedError("execution IR requires an einsum equation")
    options = dict(kwargs)
    constants = options.pop("constants", ())
    optimize = options.pop("optimize", "optimal")
    if options:
        raise NotImplementedError(
            "experimental execution IR does not support opt_einsum options: {}".format(
                ", ".join(sorted(options))
            )
        )
    return _build_ir_expression(
        args[0],
        args[1:],
        constants,
        network=profile_network,
        center_kind=profile_center_kind,
        optimize=optimize,
        lowerer=lower_einsum_path,
    )


__all__ = [
    "ResolvedExecutionArtifact",
    "UnsupportedRuntimeDtypeError",
    "build_experimental_einsum",
]
