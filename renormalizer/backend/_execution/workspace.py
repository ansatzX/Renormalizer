"""Deterministic workspace accounting for execution-plan metadata."""

import math

import numpy as np


MAX_WORKSPACE_BYTES = (1 << 63) - 1


def grouped_matmul_packing_bytes(operations):
    """Return stacked A/B bytes for repeated effective matrix shapes."""
    buckets = {}
    for operation in operations:
        contracted_count = len(operation.contracted_modes)
        left_shape = operation.left.spec.shape
        right_shape = operation.right.spec.shape
        left_free = left_shape[:-contracted_count]
        contracted = left_shape[-contracted_count:]
        right_free = right_shape[contracted_count:]
        key = (
            operation.left.spec.dtype,
            math.prod(left_free),
            math.prod(right_free),
            math.prod(contracted),
        )
        buckets[key] = buckets.get(key, 0) + 1

    total = 0
    for (dtype, m, n, k), count in buckets.items():
        if count < 2:
            continue
        total += count * (m * k + k * n) * np.dtype(dtype).itemsize
        if total > MAX_WORKSPACE_BYTES:
            raise OverflowError(
                "workspace requirement exceeds the supported metadata bound"
            )
    return total


def workspace_bytes_for_steps(steps, output_key):
    """Return a conservative sum of unique temporary output buffers."""
    sizes = {}
    for step in steps:
        grouped = getattr(step, "operations", None) is not None
        for output in step.workspace_outputs:
            if output.key == output_key and not grouped:
                continue
            previous = sizes.setdefault(output.key, output.spec.nbytes)
            if previous != output.spec.nbytes:
                raise ValueError(
                    "temporary buffer key has inconsistent specifications"
                )
    total = 0
    for size in sizes.values():
        total += size
        if total > MAX_WORKSPACE_BYTES:
            raise OverflowError(
                "workspace requirement exceeds the supported metadata bound"
            )
    for step in steps:
        operations = getattr(step, "operations", None)
        if operations is None:
            continue
        total += grouped_matmul_packing_bytes(operations)
        if total > MAX_WORKSPACE_BYTES:
            raise OverflowError(
                "workspace requirement exceeds the supported metadata bound"
            )
    return total


__all__ = ["grouped_matmul_packing_bytes", "workspace_bytes_for_steps"]
