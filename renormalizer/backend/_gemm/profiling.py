MAX_SHAPE_BUCKETS = 32
GEMM_SHAPE_RANK = 3


def summarize_shape_buckets(buckets, *, limit=MAX_SHAPE_BUCKETS):
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("shape bucket limit must be a non-negative integer")
    bounded_limit = min(limit, MAX_SHAPE_BUCKETS)
    summary = []
    for shape, count in buckets.items():
        if type(shape) not in (list, tuple):
            raise TypeError("shape bucket keys must be lists or tuples")
        if len(shape) != GEMM_SHAPE_RANK:
            raise ValueError("GEMM shape buckets must contain exactly three dimensions")
        if not all(type(dimension) is int for dimension in shape):
            raise TypeError("shape bucket dimensions must be Python integers")
        if not isinstance(count, int) or isinstance(count, bool):
            raise TypeError("shape bucket counts must be Python integers")
        summary.append({"shape": list(shape), "count": count})
    summary.sort(key=lambda item: (-item["count"], item["shape"]))
    return summary[:bounded_limit]


def grouped_gemm_payload(
    *,
    operation,
    task_count,
    shape_buckets,
    executed_grouped=False,
    pack_wall_s=None,
    compute_wall_s=None,
    scatter_wall_s=None,
    backend=None,
    policy=None,
    timing_semantics=None,
    device_synchronized=None,
):
    if type(operation) is not str:
        raise TypeError("operation must be a Python string")
    if type(task_count) is not int:
        raise TypeError("task_count must be a Python integer")
    if task_count < 0:
        raise ValueError("task_count must be non-negative")
    if type(executed_grouped) is not bool:
        raise TypeError("executed_grouped must be a Python boolean")
    if executed_grouped and task_count < 2:
        raise ValueError("grouped GEMM execution requires at least two tasks")
    payload = {
        "operation": operation,
        "task_count": task_count,
        "grouped_execution": executed_grouped,
        "shape_buckets": summarize_shape_buckets(shape_buckets),
        "shape_bucket_count": len(shape_buckets),
        "shape_buckets_truncated": len(shape_buckets) > MAX_SHAPE_BUCKETS,
    }
    optional_strings = {
        "backend": backend,
        "policy": policy,
        "timing_semantics": timing_semantics,
    }
    for field, value in optional_strings.items():
        if value is not None:
            if type(value) is not str:
                raise TypeError("{} must be a Python string".format(field))
            payload[field] = value
    for field, value in {
        "pack_wall_s": pack_wall_s,
        "compute_wall_s": compute_wall_s,
        "scatter_wall_s": scatter_wall_s,
    }.items():
        if value is not None:
            if isinstance(value, bool) or type(value) not in {int, float}:
                raise TypeError("{} must be a Python number".format(field))
            if value < 0:
                raise ValueError("{} must be non-negative".format(field))
            payload[field] = value
    if device_synchronized is not None:
        if type(device_synchronized) is not bool:
            raise TypeError("device_synchronized must be a Python boolean")
        payload["device_synchronized"] = device_synchronized
    return payload
