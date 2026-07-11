import hashlib
import json


MAX_SHAPES = 32
MAX_SHAPE_RANK = 16
MAX_ACTUAL_STEPS = 32


def _require_string(value, field):
    if not isinstance(value, str):
        raise TypeError(f"{field} must be Python metadata")


def _require_optional_string(value, field):
    if value is not None:
        _require_string(value, field)


def _require_number(value, field):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{field} must be Python metadata")


def completion_timing_metadata(backend_name, device):
    _require_string(backend_name, "backend_name")
    _require_string(device, "device")
    completion_guaranteed = backend_name in {"numpy", "torch"} and device == "cpu"
    return {
        "timing_semantics": (
            "host_elapsed_synchronous" if completion_guaranteed else "host_elapsed_unsynchronized"
        ),
        "device_synchronized": completion_guaranteed,
    }


def _validate_shape(shape):
    if type(shape) not in (list, tuple):
        raise TypeError("shape metadata must be a list or tuple")
    for dimension in shape:
        if type(dimension) is not int:
            raise TypeError("shape dimensions must be Python integers")


def _bounded_shape(shape):
    _validate_shape(shape)
    return list(shape[:MAX_SHAPE_RANK])


def _bounded_shapes(shapes):
    if type(shapes) not in (list, tuple):
        raise TypeError("input shape metadata must be a list or tuple")
    for shape in shapes:
        _validate_shape(shape)
    return [list(shape[:MAX_SHAPE_RANK]) for shape in shapes[:MAX_SHAPES]]


def oe_path_identity(path):
    if not isinstance(path, (list, tuple)):
        raise TypeError("OE path metadata must be a list or tuple")
    canonical_path = []
    for step in path:
        if not isinstance(step, (list, tuple)):
            raise TypeError("OE path steps must be lists or tuples")
        canonical_step = []
        for operand in step:
            if not isinstance(operand, int) or isinstance(operand, bool):
                raise TypeError("OE path operands must be Python integers")
            canonical_step.append(operand)
        canonical_path.append(canonical_step)
    encoded = json.dumps(canonical_path, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def local_hv_payload(
    *,
    network,
    center_kind,
    input_shapes,
    output_shape,
    requested_policy="legacy_oe",
    actual_policy="legacy_oe",
    planner_source,
    oe_path_hash,
    actual_steps,
    wall_s=None,
    timing_semantics,
    device_synchronized,
    path_override_reason=None,
    fallback_reason=None,
    backend=None,
    device=None,
):
    for field, value in (
        ("network", network),
        ("center_kind", center_kind),
        ("requested_policy", requested_policy),
        ("actual_policy", actual_policy),
        ("planner_source", planner_source),
        ("oe_path_hash", oe_path_hash),
        ("timing_semantics", timing_semantics),
    ):
        _require_string(value, field)
    for field, value in (
        ("path_override_reason", path_override_reason),
        ("fallback_reason", fallback_reason),
        ("backend", backend),
        ("device", device),
    ):
        _require_optional_string(value, field)
    if wall_s is not None:
        _require_number(wall_s, "wall_s")
    if type(device_synchronized) is not bool:
        raise TypeError("device_synchronized must be Python metadata")
    if timing_semantics not in {"host_elapsed_synchronous", "host_elapsed_unsynchronized"}:
        raise ValueError("unsupported timing semantics")
    if not isinstance(actual_steps, (list, tuple)) or not all(type(step) is str for step in actual_steps):
        raise TypeError("actual steps must be Python strings")
    bounded_inputs = _bounded_shapes(input_shapes)
    bounded_output = _bounded_shape(output_shape)
    payload = {
        "network": network,
        "center_kind": center_kind,
        "input_shapes": bounded_inputs,
        "input_shape_count": len(input_shapes),
        "input_shapes_truncated": len(input_shapes) > MAX_SHAPES
        or any(len(shape) > MAX_SHAPE_RANK for shape in input_shapes[:MAX_SHAPES]),
        "output_shape": bounded_output,
        "output_shape_truncated": len(output_shape) > MAX_SHAPE_RANK,
        "requested_policy": requested_policy,
        "actual_policy": actual_policy,
        "planner_source": planner_source,
        "oe_path_hash": oe_path_hash,
        "path_override_reason": path_override_reason,
        "actual_steps": list(actual_steps[:MAX_ACTUAL_STEPS]),
        "actual_step_count": len(actual_steps),
        "actual_steps_truncated": len(actual_steps) > MAX_ACTUAL_STEPS,
        "timing_semantics": timing_semantics,
        "device_synchronized": device_synchronized,
        "fallback": actual_policy if fallback_reason is not None else None,
        "fallback_reason": fallback_reason,
    }
    if backend is not None:
        payload["backend"] = backend
    if device is not None:
        payload["device"] = device
    if wall_s is not None:
        payload["wall_s"] = wall_s
    return payload


def phase_summary_payload(*, phase, network, operation, operation_count, wall_s):
    _require_string(phase, "phase")
    _require_string(network, "network")
    _require_string(operation, "operation")
    if not isinstance(operation_count, int) or isinstance(operation_count, bool):
        raise TypeError("operation_count must be Python metadata")
    _require_number(wall_s, "wall_s")
    return {
        "phase": phase,
        "network": network,
        "operation": operation,
        "operation_count": operation_count,
        "wall_s": wall_s,
    }
