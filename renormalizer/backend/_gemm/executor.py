"""Validated scalar and bucketed-batched GEMM execution."""

from collections.abc import MutableMapping
from contextlib import nullcontext

import numpy as np

from renormalizer.backend._gemm.descriptors import MatmulDesc
from renormalizer.backend._gemm.grouping import GemmGroupKey, canonical_gemm_dtype
from renormalizer.utils import profiling


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


def _raw_shapes(descriptor):
    a_shape = (
        (descriptor.m, descriptor.k)
        if descriptor.trans_a == "N"
        else (descriptor.k, descriptor.m)
    )
    b_shape = (
        (descriptor.k, descriptor.n)
        if descriptor.trans_b == "N"
        else (descriptor.n, descriptor.k)
    )
    return a_shape, b_shape


def _validate_operand(backend, value, key, shape, dtype_name, role):
    backend._validate_execution_array(value)
    if tuple(value.shape) != shape:
        raise ValueError(
            "{} operand {!r} shape {} does not match {}".format(
                role, key, tuple(value.shape), shape
            )
        )
    actual_dtype = canonical_gemm_dtype(value.dtype)
    if dtype_name is not None and actual_dtype != dtype_name:
        raise ValueError(
            "{} operand {!r} dtype {!r} does not match {!r}".format(
                role, key, actual_dtype, dtype_name
            )
        )
    return actual_dtype


def _arrays_overlap(backend, left, right):
    if left is right:
        return True
    if backend.name == "numpy":
        return bool(np.shares_memory(left, right))
    shares_memory = getattr(getattr(backend, "_cupy", None), "shares_memory", None)
    if shares_memory is not None:
        return bool(shares_memory(left, right))
    return False


def _validate_scalars_for_dtype(descriptor, dtype_name):
    dtype = np.dtype(dtype_name)
    for field, identity in (("alpha", 1), ("beta", 0)):
        value = getattr(descriptor, field)
        if value == identity:
            continue
        scalar_dtype = np.asarray(value).dtype
        if not np.can_cast(scalar_dtype, dtype, casting="same_kind"):
            raise ValueError(
                "{} cannot be applied in-place to GEMM dtype {!r}".format(
                    field, dtype_name
                )
            )


def _validate_request(backend, descriptors, tensors, stream, workspace):
    if not isinstance(tensors, MutableMapping):
        raise TypeError("grouped GEMM tensors must be a mutable mapping")
    descriptors = tuple(descriptors)
    if any(not isinstance(descriptor, MatmulDesc) for descriptor in descriptors):
        raise TypeError("grouped GEMM tasks must be MatmulDesc instances")
    output_keys = tuple(descriptor.c_key for descriptor in descriptors)
    if len(set(output_keys)) != len(output_keys):
        raise ValueError("grouped GEMM requires unique output keys")
    input_keys = {
        key
        for descriptor in descriptors
        for key in (descriptor.a_key, descriptor.b_key)
    }
    collisions = sorted(set(output_keys) & input_keys)
    if collisions:
        raise ValueError(
            "grouped GEMM output keys must not collide with input keys: {}".format(
                ", ".join(collisions)
            )
        )

    backend._validate_execution_stream(stream)
    descriptor_dtypes = []
    output_arrays = []
    input_arrays = []
    protected_arrays = []
    for index, descriptor in enumerate(descriptors):
        missing = [
            key for key in (descriptor.a_key, descriptor.b_key) if key not in tensors
        ]
        if missing:
            raise ValueError(
                "missing grouped GEMM tensor keys: {}".format(", ".join(missing))
            )
        a_shape, b_shape = _raw_shapes(descriptor)
        left = tensors[descriptor.a_key]
        right = tensors[descriptor.b_key]
        left_dtype = _validate_operand(
            backend, left, descriptor.a_key, a_shape, None, "left"
        )
        _validate_operand(
            backend, right, descriptor.b_key, b_shape, left_dtype, "right"
        )
        _validate_scalars_for_dtype(descriptor, left_dtype)
        descriptor_dtypes.append(left_dtype)
        input_arrays.append((index, left, right))
        protected_arrays.extend((left, right))

        if descriptor.beta != 0 and descriptor.c_key not in tensors:
            raise ValueError(
                "nonzero beta requires existing C tensor {!r}".format(descriptor.c_key)
            )
        if descriptor.c_key in tensors:
            output = tensors[descriptor.c_key]
            _validate_operand(
                backend,
                output,
                descriptor.c_key,
                (descriptor.m, descriptor.n),
                left_dtype,
                "output",
            )
            output_arrays.append((index, descriptor.c_key, output))
            protected_arrays.append(output)

    for left_index, left_key, left in output_arrays:
        for right_index, right_key, right in output_arrays:
            if left_index < right_index and _arrays_overlap(backend, left, right):
                raise ValueError(
                    "grouped GEMM output tensors {!r} and {!r} overlap".format(
                        left_key, right_key
                    )
                )
        for input_index, input_left, input_right in input_arrays:
            if input_index == left_index:
                continue
            if _arrays_overlap(backend, left, input_left) or _arrays_overlap(
                backend, left, input_right
            ):
                raise ValueError(
                    "grouped GEMM output tensor {!r} overlaps another task input".format(
                        left_key
                    )
                )

    groups = {}
    for descriptor, dtype_name in zip(descriptors, descriptor_dtypes):
        key = GemmGroupKey(
            dtype_name,
            descriptor.m,
            descriptor.n,
            descriptor.k,
            descriptor.trans_a,
            descriptor.trans_b,
        )
        groups.setdefault(key, []).append(descriptor)
    groups = {key: tuple(group) for key, group in groups.items()}
    temporary_bytes = 0
    for key, group in groups.items():
        itemsize = np.dtype(key.dtype).itemsize
        if len(group) >= 2:
            temporary_bytes += len(group) * (key.m * key.k + key.k * key.n) * itemsize
        elif np.dtype(key.dtype).kind == "c":
            if key.trans_a == "C":
                temporary_bytes += key.m * key.k * itemsize
            if key.trans_b == "C":
                temporary_bytes += key.k * key.n * itemsize
        temporary_bytes += sum(
            descriptor.m * descriptor.n * itemsize
            for descriptor in group
            if descriptor.beta != 0
        )
    capacity = _workspace_capacity(workspace)
    if capacity is not None and temporary_bytes > capacity:
        raise ValueError(
            "grouped GEMM workspace capacity {} exceeds supplied capacity {}".format(
                temporary_bytes, capacity
            )
        )
    return descriptors, groups, tuple(protected_arrays)


def _transpose(backend, value, flag):
    if flag == "N":
        return value
    return backend.transpose(value, axes=(1, 0))


def _transform(backend, value, flag, scope):
    value = _transpose(backend, value, flag)
    if flag == "C" and np.dtype(value.dtype).kind == "c":
        destination = scope.empty(value.shape, dtype=value.dtype, order="C")
        value = scope.conjugate_into(value, destination)
    return value


def _conjugate_pack_in_place(value, flag, scope):
    if flag != "C" or np.dtype(value.dtype).kind != "c":
        return value
    return scope.conjugate_into(value, value)


def _apply_scalars(result, descriptor, tensors, scope):
    if descriptor.alpha != 1:
        scope.multiply_into(result, descriptor.alpha, result)
    if descriptor.beta != 0:
        scaled = scope.empty(result.shape, dtype=result.dtype, order="C")
        scope.multiply_into(tensors[descriptor.c_key], descriptor.beta, scaled)
        scope.add_into(result, scaled, result)
    return result


def _validate_result(backend, result, shape, dtype_name, context, protected_arrays):
    backend._validate_execution_array(result)
    if tuple(result.shape) != shape:
        raise ValueError(
            "{} shape {} does not match {}".format(context, tuple(result.shape), shape)
        )
    if canonical_gemm_dtype(result.dtype) != dtype_name:
        raise ValueError("{} dtype does not match grouped operands".format(context))
    if any(
        _arrays_overlap(backend, result, protected) for protected in protected_arrays
    ):
        raise ValueError("{} overlaps a protected task array".format(context))


def _execute_buckets(
    backend,
    descriptors,
    groups,
    tensors,
    workspace,
    protected_arrays,
    *,
    clock=None,
    allocation_scope=None,
):
    outputs = {}
    pack_wall_s = 0.0
    compute_wall_s = 0.0
    scatter_wall_s = 0.0
    executed_grouped = False

    for key, group in groups.items():
        if len(group) == 1:
            descriptor = group[0]
            left = _transform(
                backend,
                tensors[descriptor.a_key],
                descriptor.trans_a,
                allocation_scope,
            )
            right = _transform(
                backend,
                tensors[descriptor.b_key],
                descriptor.trans_b,
                allocation_scope,
            )
            started = clock() if clock is not None else None
            result = allocation_scope.empty(
                (descriptor.m, descriptor.n), dtype=np.dtype(key.dtype), order="C"
            )
            _validate_result(
                backend,
                result,
                (descriptor.m, descriptor.n),
                key.dtype,
                "scalar matmul result",
                protected_arrays,
            )
            allocation_scope.matmul_into(
                left, right, result, workspace=workspace, batched=False
            )
            if clock is not None:
                compute_wall_s += clock() - started
                started = clock()
            outputs[descriptor.c_key] = _apply_scalars(
                result, descriptor, tensors, allocation_scope
            )
            if clock is not None:
                scatter_wall_s += clock() - started
            continue

        executed_grouped = True
        started = clock() if clock is not None else None
        left_pack = allocation_scope.empty(
            (len(group), key.m, key.k), dtype=np.dtype(key.dtype), order="C"
        )
        for index, item in enumerate(group):
            allocation_scope.copy_into(
                _transpose(backend, tensors[item.a_key], item.trans_a), left_pack[index]
            )
        _conjugate_pack_in_place(left_pack, key.trans_a, allocation_scope)
        right_pack = allocation_scope.empty(
            (len(group), key.k, key.n), dtype=np.dtype(key.dtype), order="C"
        )
        for index, item in enumerate(group):
            allocation_scope.copy_into(
                _transpose(backend, tensors[item.b_key], item.trans_b),
                right_pack[index],
            )
        _conjugate_pack_in_place(right_pack, key.trans_b, allocation_scope)
        if clock is not None:
            pack_wall_s += clock() - started
            started = clock()
        result_pack = allocation_scope.empty(
            (len(group), key.m, key.n), dtype=np.dtype(key.dtype), order="C"
        )
        _validate_result(
            backend,
            result_pack,
            (len(group), key.m, key.n),
            key.dtype,
            "batched matmul result",
            protected_arrays,
        )
        allocation_scope.matmul_into(
            left_pack, right_pack, result_pack, workspace=workspace, batched=True
        )
        if clock is not None:
            compute_wall_s += clock() - started
            started = clock()
        for index, descriptor in enumerate(group):
            result_view = result_pack[index]
            outputs[descriptor.c_key] = _apply_scalars(
                result_view, descriptor, tensors, allocation_scope
            )
        if clock is not None:
            scatter_wall_s += clock() - started

    ordered = tuple(outputs[descriptor.c_key] for descriptor in descriptors)
    return ordered, executed_grouped, pack_wall_s, compute_wall_s, scatter_wall_s


def _execute_profiled(
    backend,
    descriptors,
    groups,
    tensors,
    workspace,
    protected_arrays,
    policy,
    allocation_scope,
):
    import time

    from renormalizer.backend._execution.profiling import completion_timing_metadata
    from renormalizer.backend._gemm.profiling import grouped_gemm_payload

    result = _execute_buckets(
        backend,
        descriptors,
        groups,
        tensors,
        workspace,
        protected_arrays,
        clock=time.perf_counter,
        allocation_scope=allocation_scope,
    )
    outputs, executed_grouped, pack_wall_s, compute_wall_s, scatter_wall_s = result
    shape_buckets = {}
    for key, group in groups.items():
        shape = (key.m, key.n, key.k)
        shape_buckets[shape] = shape_buckets.get(shape, 0) + len(group)
    timing = completion_timing_metadata(backend.name, backend.current_device())
    payload = grouped_gemm_payload(
        operation="gemm",
        task_count=len(descriptors),
        shape_buckets=shape_buckets,
        executed_grouped=executed_grouped,
        pack_wall_s=pack_wall_s,
        compute_wall_s=compute_wall_s,
        scatter_wall_s=scatter_wall_s,
        backend=backend.name,
        policy=policy,
        **timing,
    )
    profiling.record("grouped_gemm_execute", **payload)
    return outputs


def execute_grouped_gemm(
    backend,
    descriptors,
    tensors,
    *,
    stream=None,
    workspace=None,
    policy="direct",
    _allocation_scope=None,
):
    """Execute descriptors and publish every output only after full success."""
    backend._require_execution_usable()
    if not getattr(backend, "supports_grouped_gemm", False):
        raise NotImplementedError(
            "backend {!r} does not support grouped GEMM".format(backend.name)
        )
    if not isinstance(policy, str) or not policy:
        raise ValueError("grouped GEMM policy must be a non-empty string")
    from renormalizer.backend._execution.executor import ExecutionAllocationScope

    if _allocation_scope is not None and not isinstance(
        _allocation_scope, ExecutionAllocationScope
    ):
        raise TypeError("internal allocation scope is invalid")
    descriptors, groups, protected_arrays = _validate_request(
        backend, descriptors, tensors, stream, workspace
    )
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
            scope.capture_resources(tensors)
            for array in protected_arrays:
                scope.capture(array)
            if profiling.enabled():
                outputs = _execute_profiled(
                    backend,
                    descriptors,
                    groups,
                    tensors,
                    workspace,
                    protected_arrays,
                    policy,
                    scope,
                )
            else:
                outputs, _, _, _, _ = _execute_buckets(
                    backend,
                    descriptors,
                    groups,
                    tensors,
                    workspace,
                    protected_arrays,
                    allocation_scope=scope,
                )
            tensors.update(
                (descriptor.c_key, output)
                for descriptor, output in zip(descriptors, outputs)
            )
    return outputs


__all__ = ["execute_grouped_gemm"]
