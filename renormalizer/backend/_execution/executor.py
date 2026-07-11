"""Validated local execution for immutable contraction plans."""

import math

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


def _store_output(backend, buffers, ref, value):
    if ref.key in buffers:
        raise ValueError("plan attempted to replace declared buffer {!r}".format(ref.key))
    _validate_array(backend, value, ref, "step output {!r}".format(ref.key))
    buffers[ref.key] = value


def _reshape_view(backend, value, shape):
    if tuple(value.shape) == tuple(shape):
        return value
    result = backend.reshape(value, shape)
    if not backend._is_exact_execution_reshape(value, result):
        raise ValueError("planned contraction reshape unexpectedly copied data")
    return result


def _execute_transform(backend, step, buffers):
    value = backend.transpose(buffers[step.input.key], axes=step.axes)
    if step.copy:
        value = backend.array(value, copy=True, order=step.output.spec.layout)
    _store_output(backend, buffers, step.output, value)


def _execute_reduction(backend, step, buffers):
    axes = tuple(step.input.spec.modes.index(mode) for mode in step.reduced_modes)
    value = backend.empty(
        step.output.spec.shape,
        dtype=np.dtype(step.output.spec.dtype),
        order=step.output.spec.layout,
    )
    result = backend.sum(
        buffers[step.input.key],
        out=value,
        axis=axes,
        dtype=np.dtype(step.output.spec.dtype),
    )
    if result is not value:
        raise ValueError("backend reduction did not return its planned output")
    _store_output(backend, buffers, step.output, value)


def _product(values):
    return math.prod(values)


def _execute_matmul(backend, step, buffers, workspace):
    left, right = _matmul_views(backend, step, buffers)
    value = backend.matmul(left, right, workspace=workspace)
    value = _reshape_view(backend, value, step.output.spec.shape)
    _store_output(backend, buffers, step.output, value)


def _matmul_views(backend, step, buffers):
    batch_count = len(step.batch_modes) if isinstance(step, BatchedMatmulStep) else 0
    contracted_count = len(step.contracted_modes)
    left_shape = step.left.spec.shape
    right_shape = step.right.spec.shape
    batch_shape = left_shape[:batch_count]
    left_free_shape = left_shape[batch_count:len(left_shape) - contracted_count]
    contracted_shape = left_shape[len(left_shape) - contracted_count:]
    right_free_shape = right_shape[batch_count + contracted_count:]
    left = _reshape_view(
        backend,
        buffers[step.left.key],
        batch_shape + (_product(left_free_shape), _product(contracted_shape)),
    )
    right = _reshape_view(
        backend,
        buffers[step.right.key],
        batch_shape + (_product(contracted_shape), _product(right_free_shape)),
    )
    return left, right


def _execute_grouped_matmul(backend, step, buffers, workspace):
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
    )
    outputs = []
    for operation, matrix in zip(prepared, matrix_outputs):
        if operation.output.key in buffers:
            raise ValueError(
                "plan attempted to replace declared buffer {!r}".format(
                    operation.output.key
                )
            )
        value = _reshape_view(backend, matrix, operation.output.spec.shape)
        _validate_array(
            backend,
            value,
            operation.output,
            "step output {!r}".format(operation.output.key),
        )
        outputs.append((operation.output.key, value))
    buffers.update(outputs)


def execute_plan(backend, plan, bindings, *, stream=None, workspace=None):
    """Execute a trusted plan; ``workspace`` is a capacity bound, not reused storage."""
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
        for step in plan.steps:
            if isinstance(step, TransformStep):
                _execute_transform(backend, step, buffers)
            elif isinstance(step, ReductionStep):
                _execute_reduction(backend, step, buffers)
            elif isinstance(step, (MatmulStep, BatchedMatmulStep)):
                _execute_matmul(backend, step, buffers, workspace)
            elif isinstance(step, GroupedMatmulStep):
                _execute_grouped_matmul(backend, step, buffers, workspace)
            else:
                raise TypeError("unsupported execution step type")
    return buffers[plan.output.key]


__all__ = ["execute_plan"]
