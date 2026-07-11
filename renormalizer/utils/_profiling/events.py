import math


EVENT_NAMES = (
    "run_start",
    "phase_summary",
    "local_hv_execute",
    "grouped_gemm_execute",
    "distributed_solve_summary",
    "working_set_transfer",
    "run_summary",
)
SOURCE_EVENT_NAMES = EVENT_NAMES[:-1]
MAX_SHAPES = 32
MAX_SHAPE_RANK = 16
MAX_ACTUAL_STEPS = 32
MAX_SHAPE_BUCKETS = 32
MAX_STRING_BYTES = 1024
MAX_KEY_BYTES = 128
MAX_INTEGER_DIGITS = 64
MAX_CONTAINER_DEPTH = 4
MAX_CONTAINER_ITEMS = 64


def normalize_bounded_metadata(value, *, depth=0):
    value_type = type(value)
    if value is None or value_type is bool:
        return value
    if value_type is int:
        if len(str(abs(value))) > MAX_INTEGER_DIGITS:
            raise ValueError("profiling metadata integers exceed the digit limit")
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("profiling metadata floats must be finite")
        return value
    if value_type is str:
        if len(value.encode("utf-8")) > MAX_STRING_BYTES:
            raise ValueError("profiling metadata strings exceed the UTF-8 byte limit")
        return value
    if value_type.__module__ == "numpy" and value_type.__name__ != "ndarray":
        return normalize_bounded_metadata(value.item(), depth=depth)
    if hasattr(value, "shape"):
        raise TypeError("profiling metadata arrays are not supported")
    if value_type in (list, tuple):
        if depth >= MAX_CONTAINER_DEPTH:
            raise ValueError("profiling metadata exceeds the container depth limit")
        if len(value) > MAX_CONTAINER_ITEMS:
            raise ValueError("profiling metadata exceeds the container item limit")
        return [normalize_bounded_metadata(item, depth=depth + 1) for item in value]
    if value_type is dict:
        if depth >= MAX_CONTAINER_DEPTH:
            raise ValueError("profiling metadata exceeds the container depth limit")
        if len(value) > MAX_CONTAINER_ITEMS:
            raise ValueError("profiling metadata exceeds the container item limit")
        normalized = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("profiling metadata keys must be Python strings")
            if len(key.encode("utf-8")) > MAX_KEY_BYTES:
                raise ValueError("profiling metadata keys exceed the UTF-8 byte limit")
            normalized[key] = normalize_bounded_metadata(item, depth=depth + 1)
        return normalized
    raise TypeError(f"profiling metadata is not JSON serializable: {value_type.__name__}")


def normalize_event_filter(events):
    if events is None:
        return None
    if isinstance(events, str):
        events = (events,)
    events = frozenset(events)
    if not all(type(event) is str for event in events):
        raise TypeError("event filters must contain Python strings")
    unsupported = events.difference(EVENT_NAMES)
    if unsupported:
        raise ValueError(f"unsupported profiling event filters: {sorted(unsupported)}")
    return events


def should_write_event(event, events) -> bool:
    return events is None or event in events


def require_string(value, field):
    if type(value) is not str:
        raise TypeError(f"{field} must be a Python string")
    return value


def require_optional_string(value, field):
    if value is not None:
        require_string(value, field)
    return value


def require_non_negative_int(value, field):
    if type(value) is not int:
        raise TypeError(f"{field} must be a Python integer")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def require_positive_int(value, field):
    require_non_negative_int(value, field)
    if value == 0:
        raise ValueError(f"{field} must be positive")
    return value


def require_non_negative_number(value, field):
    if type(value) not in (int, float):
        raise TypeError(f"{field} must be a Python number")
    if value < 0 or not math.isfinite(value):
        raise ValueError(f"{field} must be finite and non-negative")
    return value


def require_bool(value, field):
    if type(value) is not bool:
        raise TypeError(f"{field} must be a Python boolean")
    return value


def require_shape(value, field, *, exact_rank=None):
    if type(value) is not list:
        raise TypeError(f"{field} must be a bounded Python list")
    if exact_rank is not None and len(value) != exact_rank:
        raise ValueError(f"{field} must have rank {exact_rank}")
    if len(value) > MAX_SHAPE_RANK:
        raise ValueError(f"{field} exceeds the retained shape rank")
    for dimension in value:
        require_positive_int(dimension, f"{field} dimension")
    return value


def _require_field(payload, field):
    if field not in payload:
        raise ValueError(f"{field} is required")
    return payload[field]


def validate_run_start_record(payload):
    require_string(_require_field(payload, "backend"), "backend")
    rank = require_non_negative_int(_require_field(payload, "rank"), "rank")
    world_size = require_positive_int(_require_field(payload, "world_size"), "world_size")
    if rank >= world_size:
        raise ValueError("rank must be less than world_size")


def validate_phase_summary_record(payload):
    for field in ("phase", "network", "operation"):
        require_string(_require_field(payload, field), field)
    require_non_negative_int(_require_field(payload, "operation_count"), "operation_count")
    require_non_negative_number(_require_field(payload, "wall_s"), "wall_s")


def validate_local_hv_record(payload):
    for field in (
        "network",
        "center_kind",
        "requested_policy",
        "actual_policy",
        "planner_source",
        "oe_path_hash",
        "timing_semantics",
    ):
        require_string(_require_field(payload, field), field)
    if payload["actual_policy"] not in {"execution_ir", "legacy_oe"}:
        raise ValueError("actual_policy is unsupported")
    if payload["timing_semantics"] not in {
        "host_elapsed_synchronous",
        "host_elapsed_unsynchronized",
    }:
        raise ValueError("timing_semantics is unsupported")
    for field in ("path_override_reason", "fallback", "fallback_reason"):
        require_optional_string(_require_field(payload, field), field)
    for field in ("backend", "device"):
        if field in payload:
            require_optional_string(payload[field], field)

    input_shapes = _require_field(payload, "input_shapes")
    if type(input_shapes) is not list or len(input_shapes) > MAX_SHAPES:
        raise ValueError("input_shapes must be a bounded Python list")
    for index, shape in enumerate(input_shapes):
        require_shape(shape, f"input_shapes[{index}]")
    input_shape_ranks = _require_field(payload, "input_shape_ranks")
    if type(input_shape_ranks) is not list or len(input_shape_ranks) != len(input_shapes):
        raise ValueError("input_shape_ranks must match retained input shapes")
    for index, (shape, original_rank) in enumerate(zip(input_shapes, input_shape_ranks)):
        require_non_negative_int(original_rank, f"input_shape_ranks[{index}]")
        if len(shape) != min(original_rank, MAX_SHAPE_RANK):
            raise ValueError("input shape rank evidence does not match retained shape")
    input_shape_count = require_non_negative_int(
        _require_field(payload, "input_shape_count"), "input_shape_count"
    )
    input_shapes_truncated = require_bool(
        _require_field(payload, "input_shapes_truncated"), "input_shapes_truncated"
    )
    expected_input_count = min(input_shape_count, MAX_SHAPES)
    if len(input_shapes) != expected_input_count:
        raise ValueError("retained input_shapes length does not match input_shape_count")
    expected_input_truncation = input_shape_count > MAX_SHAPES
    if input_shapes_truncated != expected_input_truncation:
        raise ValueError("input_shapes_truncated does not match retained evidence")
    output_shape = require_shape(_require_field(payload, "output_shape"), "output_shape")
    output_shape_rank = require_non_negative_int(
        _require_field(payload, "output_shape_rank"), "output_shape_rank"
    )
    if len(output_shape) != min(output_shape_rank, MAX_SHAPE_RANK):
        raise ValueError("output shape rank evidence does not match retained shape")
    output_shape_truncated = require_bool(
        _require_field(payload, "output_shape_truncated"), "output_shape_truncated"
    )
    if output_shape_truncated != (output_shape_rank > len(output_shape)):
        raise ValueError("output_shape_truncated does not match retained evidence")

    actual_steps = _require_field(payload, "actual_steps")
    if type(actual_steps) is not list or len(actual_steps) > MAX_ACTUAL_STEPS:
        raise ValueError("actual_steps must be a bounded Python list")
    for step in actual_steps:
        require_string(step, "actual_steps item")
    actual_step_count = require_non_negative_int(
        _require_field(payload, "actual_step_count"), "actual_step_count"
    )
    actual_steps_truncated = require_bool(
        _require_field(payload, "actual_steps_truncated"), "actual_steps_truncated"
    )
    expected_step_count = min(actual_step_count, MAX_ACTUAL_STEPS)
    if len(actual_steps) != expected_step_count:
        raise ValueError("retained actual_steps length does not match actual_step_count")
    if actual_steps_truncated != (actual_step_count > MAX_ACTUAL_STEPS):
        raise ValueError("actual_steps_truncated does not match retained evidence")
    require_bool(_require_field(payload, "device_synchronized"), "device_synchronized")
    require_non_negative_number(_require_field(payload, "wall_s"), "wall_s")


def validate_grouped_gemm_record(payload):
    require_string(_require_field(payload, "operation"), "operation")
    task_count = require_non_negative_int(_require_field(payload, "task_count"), "task_count")
    grouped_execution = require_bool(
        _require_field(payload, "grouped_execution"), "grouped_execution"
    )
    if grouped_execution and task_count < 2:
        raise ValueError("grouped execution requires at least two tasks")
    shape_buckets = _require_field(payload, "shape_buckets")
    if type(shape_buckets) is not list or len(shape_buckets) > MAX_SHAPE_BUCKETS:
        raise ValueError("shape_buckets must be a bounded Python list")
    for bucket in shape_buckets:
        if type(bucket) is not dict or set(bucket) != {"shape", "count"}:
            raise ValueError("shape bucket entries require shape and count")
        require_shape(bucket["shape"], "shape bucket", exact_rank=3)
        require_non_negative_int(bucket["count"], "shape bucket count")
    bucket_count = require_non_negative_int(
        _require_field(payload, "shape_bucket_count"), "shape_bucket_count"
    )
    buckets_truncated = require_bool(
        _require_field(payload, "shape_buckets_truncated"), "shape_buckets_truncated"
    )
    expected_bucket_count = min(bucket_count, MAX_SHAPE_BUCKETS)
    if len(shape_buckets) != expected_bucket_count:
        raise ValueError("retained shape_buckets length does not match shape_bucket_count")
    if buckets_truncated != (bucket_count > MAX_SHAPE_BUCKETS):
        raise ValueError("shape_buckets_truncated does not match retained evidence")


def validate_distributed_record_fields(payload):
    shape_rank = require_positive_int(_require_field(payload, "shape_rank"), "shape_rank")
    expected_retained_rank = min(shape_rank, MAX_SHAPE_RANK)
    global_shape = require_shape(
        _require_field(payload, "global_shape"),
        "global_shape",
        exact_rank=expected_retained_rank,
    )
    local_shape = require_shape(
        _require_field(payload, "local_shape"),
        "local_shape",
        exact_rank=expected_retained_rank,
    )
    shapes_truncated = require_bool(
        _require_field(payload, "shapes_truncated"), "shapes_truncated"
    )
    if shapes_truncated != (shape_rank > MAX_SHAPE_RANK):
        raise ValueError("shapes_truncated does not match shape_rank")
    sharding_axis = require_non_negative_int(
        _require_field(payload, "sharding_axis"), "sharding_axis"
    )
    if sharding_axis >= shape_rank:
        raise ValueError("sharding_axis must identify a shape dimension")
    global_extent = require_positive_int(
        _require_field(payload, "global_shard_extent"), "global_shard_extent"
    )
    local_extent = require_positive_int(
        _require_field(payload, "local_shard_extent"), "local_shard_extent"
    )
    if local_extent > global_extent:
        raise ValueError("local_shard_extent must not exceed global_shard_extent")
    for axis, (global_dimension, local_dimension) in enumerate(zip(global_shape, local_shape)):
        if axis == sharding_axis:
            if (global_dimension, local_dimension) != (global_extent, local_extent):
                raise ValueError("retained shard extents do not match shape evidence")
        elif global_dimension != local_dimension:
            raise ValueError("unsharded local dimensions must equal global dimensions")
    for field in ("hv_count", "collective_calls", "collective_bytes", "fallback_count"):
        require_non_negative_int(_require_field(payload, field), field)
    for field in ("collective_s", "compute_s", "synchronization_s"):
        require_non_negative_number(_require_field(payload, field), field)
    if payload["fallback_count"] > payload["hv_count"]:
        raise ValueError("fallback_count must not exceed hv_count")


def validate_working_set_record_fields(payload):
    for field in (
        "h2d_bytes",
        "d2h_bytes",
        "cache_hits",
        "cache_misses",
        "dirty_writeback_bytes",
        "peak_device_bytes",
    ):
        require_non_negative_int(_require_field(payload, field), field)
    for field in (
        "h2d_s",
        "d2h_s",
        "prefetch_overlap_s",
        "prefetch_wait_s",
        "dirty_writeback_s",
    ):
        require_non_negative_number(_require_field(payload, field), field)
    require_bool(_require_field(payload, "full_replica"), "full_replica")
    if "wall_s" in payload:
        require_non_negative_number(payload["wall_s"], "wall_s")


def validate_source_event(payload):
    if type(payload) is not dict:
        raise TypeError("profiling records must be Python dictionaries")
    event = require_string(_require_field(payload, "event"), "event")
    validators = {
        "run_start": validate_run_start_record,
        "phase_summary": validate_phase_summary_record,
        "local_hv_execute": validate_local_hv_record,
        "grouped_gemm_execute": validate_grouped_gemm_record,
        "distributed_solve_summary": validate_distributed_record_fields,
        "working_set_transfer": validate_working_set_record_fields,
    }
    if event not in validators:
        if event == "run_summary":
            raise ValueError("run_summary must be emitted with record_run_summary()")
        raise ValueError(f"unsupported profiling event: {event}")
    validators[event](payload)
    return normalize_bounded_metadata(payload)
