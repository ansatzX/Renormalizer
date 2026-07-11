import math

from renormalizer.utils._profiling.events import (
    validate_distributed_record_fields,
    validate_working_set_record_fields,
)

MAX_SHAPE_RANK = 16


def _require_non_negative_int(value, field):
    if type(value) is not int:
        raise TypeError(f"{field} must be a Python integer")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")


def _require_non_negative_number(value, field):
    if type(value) not in (int, float):
        raise TypeError(f"{field} must be a Python number")
    if value < 0 or not math.isfinite(value):
        raise ValueError(f"{field} must be finite and non-negative")


def _validate_shape(shape, field):
    if type(shape) not in (list, tuple):
        raise TypeError(f"{field} must be Python shape metadata")
    if not shape:
        raise ValueError(f"{field} must not be empty")
    for dimension in shape:
        if type(dimension) is not int:
            raise TypeError("shape dimensions must be Python integers")
        if dimension <= 0:
            raise ValueError("shape dimensions must be positive")


def distributed_solve_payload(
    *,
    global_shape,
    local_shape,
    sharding_axis,
    hv_count,
    collective_calls,
    collective_bytes,
    collective_s,
    compute_s,
    synchronization_s,
    fallback_count,
):
    _validate_shape(global_shape, "global_shape")
    _validate_shape(local_shape, "local_shape")
    if len(global_shape) != len(local_shape):
        raise ValueError("global and local shapes must have the same rank")
    if type(sharding_axis) is not int:
        raise TypeError("sharding_axis must be a Python integer")
    if not 0 <= sharding_axis < len(global_shape):
        raise ValueError("sharding_axis must identify a shape dimension")
    for axis, (global_dimension, local_dimension) in enumerate(zip(global_shape, local_shape)):
        if axis == sharding_axis:
            if local_dimension > global_dimension:
                raise ValueError("local sharding dimension must not exceed global dimension")
        elif local_dimension != global_dimension:
            raise ValueError("unsharded local dimensions must equal global dimensions")

    for field, value in (
        ("hv_count", hv_count),
        ("collective_calls", collective_calls),
        ("collective_bytes", collective_bytes),
        ("fallback_count", fallback_count),
    ):
        _require_non_negative_int(value, field)
    for field, value in (
        ("collective_s", collective_s),
        ("compute_s", compute_s),
        ("synchronization_s", synchronization_s),
    ):
        _require_non_negative_number(value, field)
    if fallback_count > hv_count:
        raise ValueError("fallback_count must not exceed hv_count")

    payload = {
        "global_shape": list(global_shape[:MAX_SHAPE_RANK]),
        "local_shape": list(local_shape[:MAX_SHAPE_RANK]),
        "shape_rank": len(global_shape),
        "shapes_truncated": len(global_shape) > MAX_SHAPE_RANK,
        "sharding_axis": sharding_axis,
        "global_shard_extent": global_shape[sharding_axis],
        "local_shard_extent": local_shape[sharding_axis],
        "hv_count": hv_count,
        "collective_calls": collective_calls,
        "collective_bytes": collective_bytes,
        "collective_s": collective_s,
        "compute_s": compute_s,
        "synchronization_s": synchronization_s,
        "fallback_count": fallback_count,
    }
    validate_distributed_record_fields(payload)
    return payload


def working_set_payload(
    *,
    h2d_bytes,
    d2h_bytes,
    h2d_s,
    d2h_s,
    cache_hits,
    cache_misses,
    prefetch_overlap_s,
    prefetch_wait_s,
    dirty_writeback_bytes,
    dirty_writeback_s,
    peak_device_bytes,
    full_replica,
    wall_s=None,
):
    for field, value in (
        ("h2d_bytes", h2d_bytes),
        ("d2h_bytes", d2h_bytes),
        ("cache_hits", cache_hits),
        ("cache_misses", cache_misses),
        ("dirty_writeback_bytes", dirty_writeback_bytes),
        ("peak_device_bytes", peak_device_bytes),
    ):
        _require_non_negative_int(value, field)
    for field, value in (
        ("h2d_s", h2d_s),
        ("d2h_s", d2h_s),
        ("prefetch_overlap_s", prefetch_overlap_s),
        ("prefetch_wait_s", prefetch_wait_s),
        ("dirty_writeback_s", dirty_writeback_s),
    ):
        _require_non_negative_number(value, field)
    if type(full_replica) is not bool:
        raise TypeError("full_replica must be a Python boolean")
    if wall_s is not None:
        _require_non_negative_number(wall_s, "wall_s")

    payload = {
        "h2d_bytes": h2d_bytes,
        "d2h_bytes": d2h_bytes,
        "h2d_s": h2d_s,
        "d2h_s": d2h_s,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "prefetch_overlap_s": prefetch_overlap_s,
        "prefetch_wait_s": prefetch_wait_s,
        "dirty_writeback_bytes": dirty_writeback_bytes,
        "dirty_writeback_s": dirty_writeback_s,
        "peak_device_bytes": peak_device_bytes,
        "full_replica": full_replica,
    }
    if wall_s is not None:
        payload["wall_s"] = wall_s
    validate_working_set_record_fields(payload)
    return payload
