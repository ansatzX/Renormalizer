# -*- coding: utf-8 -*-

"""Profiling boundary and structured event implementation.

This module is intentionally cheap to import.  Runtime state that needs
context variables, JSON helpers, locks, timers, and atexit hooks is initialized
only after profiling is enabled and an event/scope is actually recorded.
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

from renormalizer.utils.log import PROFILING


LOG_PREFIX = "RENORMALIZER_PROFILING "

logger = logging.getLogger("renormalizer.profiling")
_OP_EVENTS = frozenset({
    "tensordot",
    "oe_contract",
    "oe_contract_expression",
    "contraction_plan",
    "contraction_execute",
    "matmul_execute",
    "hmm_task_build",
    "grouped_gemm_prepack",
    "grouped_gemm_execute",
    "gemv_batch_execute",
    "svd_qn",
    "eigh_qn",
    "hop_expr",
})
_TRACE_EVENTS = frozenset({
    "mps_copy",
    "mps_to_complex",
    "mps_evolve",
    "tdvp_site",
    "expand_bond_dimension",
    "expand_bond_dimension_general",
    "environ_build",
    "environ_contract_site",
    "environ_getlr",
    "tree_copy_connection",
    "ttns_copy",
    "ttns_to_complex",
    "multi_tensor_contract",
    "ttno_build_summary",
    "ttn_environ_build",
    "ttn_environ_update",
    "ttn_environ_build_children_node",
    "ttn_environ_build_parent_node",
    "ttn_evolve_0site",
    "ttn_evolve_1site",
    "ttn_evolve_2site",
    "tn_hop_expr0",
    "tn_hop_expr1",
    "tn_hop_expr2",
})
_JSONL_EVENTS = _OP_EVENTS | _TRACE_EVENTS
_OVERHEAD_TEMPLATE = {
    "events_seen": 0,
    "events_logged": 0,
    "events_summarized": 0,
    "events_dropped": 0,
    "events_written": 0,
    "summary_rows_logged": 0,
    "record_overhead_s": 0.0,
    "summary_overhead_s": 0.0,
    "event_write_overhead_s": 0.0,
    "flush_overhead_s": 0.0,
}
_runtime = None
_default_event_output_path = None

COMPUTE_CLASS_TENSORDOT = "tensordot"
COMPUTE_CLASS_OE = "oe"
COMPUTE_CLASS_SVD = "svd"
COMPUTE_CLASS_CONTRACTION_PLAN = "contraction_plan"
COMPUTE_ROLE_COMPOSITE = "composite"
COMPUTE_ROLE_KERNEL = "kernel"
COMPUTE_ROLE_PLANNING = "planning"
COMPUTE_ACCOUNTING_PRIMARY = "primary"
COMPUTE_ACCOUNTING_INCLUSIVE = "inclusive"
COMPUTE_PROFILE_SCHEMA = "renormalizer.compute_profile.v1"
WORKLOAD_SIGNATURE_SCHEMA = "renormalizer.workload_signature.v1"
EXECUTION_BREAKDOWN_SCHEMA = "renormalizer.execution_breakdown.v1"
CORE_COMPUTE_PROFILE_SCHEMA = "renormalizer.core_compute_profile.v1"
CORE_KERNEL_SUMMARY_SCHEMA = "renormalizer.core_kernel_summary.v1"
CORE_OPERATION_SCHEMA = "renormalizer.core_operation.v1"
PRACTICAL_KERNEL_VIEW_SCHEMA = "renormalizer.practical_kernel_view.v1"
PRACTICAL_MEASUREMENT_PLAN_SCHEMA = "renormalizer.practical_measurement_plan.v1"
PRACTICAL_BUCKET_SCHEMA = "renormalizer.practical_bucket.v1"
KERNEL_OBSERVATION_SCHEMA = "renormalizer.kernel_observation.v1"
EXECUTION_MODEL_SCHEMA = "renormalizer.execution_model.v1"
EXECUTION_ROUTE_SCHEMA = "renormalizer.execution_route.v1"
OPTIMIZATION_PROFILE_SCHEMA = "renormalizer.optimization_profile.v1"
COST_FACTOR_ROLLUP_SCHEMA = "renormalizer.cost_factor_rollup.v1"
_OE_TENSORDOT_STEP_TYPES = frozenset({"TDOT", "TENSORDOT"})
_CORE_COMPUTE_FAMILIES = {
    COMPUTE_CLASS_TENSORDOT: "tensordot",
    COMPUTE_CLASS_OE: "oe_contract",
    COMPUTE_CLASS_SVD: "svd",
    COMPUTE_CLASS_CONTRACTION_PLAN: "contraction_plan",
}
_TENSORDOT_GEMM_FAMILY_KERNELS = frozenset({
    "gemm",
    "gemv",
    "dot",
    "batched_gemm",
    "strided_batched_gemm",
    "grouped_gemm",
    "block_grouped_gemm",
})
_BOTTLENECK_PRIORITY = {
    "python_rhs_loop": 100,
    "axis_permutation_copy": 95,
    "backend_fallback": 90,
    "fallback": 90,
    "non_gemm_path": 85,
    "oe_non_gemm_flops": 85,
    "oe_non_gemm_steps": 80,
    "large_intermediate": 75,
    "small_gemm": 75,
    "tiny_qn_blocks": 70,
    "tiny_svd_blocks": 70,
    "fragmented_qn_blocks": 65,
    "skinny_svd_blocks": 65,
    "batchable_qn_blocks": 60,
    "decomposition": 60,
    "memory_bandwidth": 55,
    "copy_overhead": 50,
    "copy": 50,
    "communication": 45,
    "blocked_decomposition": 40,
    "path_overhead": 40,
    "oe_mixed_path": 30,
    "oe_non_gemm_path": 30,
    "oe_gemm_only_path": 30,
    "oe_path": 10,
    "compute_kernel": 10,
    "planned_backend_lowering": 10,
}
_BOTTLENECK_RESOURCE = {
    "python_rhs_loop": "python_loop",
    "axis_permutation_copy": "layout_copy",
    "backend_fallback": "backend_fallback",
    "fallback": "backend_fallback",
    "non_gemm_path": "serial_path",
    "large_intermediate": "intermediate_memory",
    "small_gemm": "small_gemm",
    "tiny_qn_blocks": "tiny_block_overhead",
    "fragmented_qn_blocks": "shape_fragmentation",
    "batchable_qn_blocks": "batchable_blocks",
    "blocked_decomposition": "decomposition",
    "decomposition": "decomposition",
    "memory_bandwidth": "memory_bandwidth",
    "copy_overhead": "copy",
    "copy": "copy",
    "communication": "communication",
    "path_overhead": "path_overhead",
    "compute_kernel": "compute_kernel",
    "planned_backend_lowering": "planner",
}
_OPTIMIZATION_SCOPE_BY_ACTION = {
    "remove_backend_fallback": "backend_kernel_coverage",
    "batch_rhs_hop": "rhs_batching",
    "avoid_axis_permutation_copies": "layout",
    "batch_or_fuse_tiny_gemm": "small_gemm_batching",
    "optimize_skinny_gemm": "skinny_gemm_kernel",
    "reduce_communication": "distributed_layout",
    "reduce_layout_or_device_copies": "data_movement",
    "reduce_non_gemm_path_cost": "contraction_path",
    "limit_largest_intermediate": "contraction_path_memory",
    "specialize_shape_diverse_paths": "contraction_path_specialization",
    "batch_reused_qn_blocks": "qn_block_batching",
    "reduce_tiny_block_overhead": "qn_block_overhead",
    "preserve_qn_sparsity": "qn_sparsity",
    "manage_block_shape_fragmentation": "qn_block_layout",
    "inspect_tensordot_kernel": "inspection",
    "inspect_oe_contract_path": "inspection",
    "inspect_qn_decomposition": "inspection",
    "inspect_contraction_execute": "inspection",
    "inspect_contraction_plan": "inspection",
}


def compute_payload(compute_class, compute_subclass, compute_role, *, accounting=COMPUTE_ACCOUNTING_PRIMARY):
    return {
        "compute_class": str(compute_class),
        "compute_subclass": str(compute_subclass),
        "compute_role": str(compute_role),
        "compute_accounting": str(accounting),
    }


def _contraction_execute_compute_class(lowering=None):
    lowering = str(lowering or "")
    if lowering in {
        "gemm",
        "batched_gemm",
        "strided_batched_gemm",
        "grouped_gemm",
        "block_grouped_gemm",
        "batched_rhs_hop",
        "fallback_rhs_loop",
        "hmm_executor",
        "slice",
        "multi_step",
        "distributed",
        "distributed_contract",
    }:
        return COMPUTE_CLASS_CONTRACTION_PLAN
    return COMPUTE_CLASS_TENSORDOT


def _should_reclassify_legacy_contraction_execute(lowering=None):
    return str(lowering or "") in {
        "batched_rhs_hop",
        "hmm_executor",
        "multi_step",
        "distributed",
        "distributed_contract",
    }


def contraction_execute_compute_payload(lowering=None):
    payload = compute_payload(
        _contraction_execute_compute_class(lowering),
        "backend_execute",
        COMPUTE_ROLE_KERNEL,
    )
    payload["_compute_class_source"] = "contraction_execute_compute_payload"
    if lowering is not None:
        payload["_compute_payload_lowering"] = str(lowering)
    return payload


def _first_present(payload, names):
    for name in names:
        value = payload.get(name)
        if value is not None:
            return value
    return None


def _standard_communication_items(communication):
    if communication is None:
        return []
    if isinstance(communication, dict):
        items = (communication,)
    elif isinstance(communication, (list, tuple)):
        items = communication
    else:
        return communication
    normalized = []
    collective_primitives = {
        "broadcast",
        "allreduce",
        "reduce_scatter",
        "gather",
        "allgather",
        "alltoall",
    }
    for item in items:
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        kind = item.get("kind", item.get("collective"))
        primitive = item.get("primitive", kind)
        collective = item.get("collective", primitive)
        bytes_value = item.get("bytes", 0)
        num_messages = item.get("num_messages", 1)
        block_size = item.get("block_size")
        if block_size is None:
            try:
                block_size = int(bytes_value) // int(num_messages or 1)
            except (TypeError, ValueError, ZeroDivisionError):
                block_size = 0
        primitive_name = str(primitive) if primitive is not None else None
        standard = {
            "kind": str(kind) if kind is not None else primitive_name,
            "primitive": primitive_name,
            "collective": str(collective) if collective is not None else None,
            "is_collective": bool(
                item.get("is_collective")
                if item.get("is_collective") is not None
                else primitive_name in collective_primitives
            ),
            "is_point_to_point": bool(
                item.get("is_point_to_point")
                if item.get("is_point_to_point") is not None
                else primitive_name == "point_to_point"
            ),
            "bytes": bytes_value,
            "num_messages": num_messages,
            "block_size": block_size,
            "wall_s": item.get("wall_s", 0),
        }
        standard.update({
            key: value
            for key, value in item.items()
            if key not in {
                "kind",
                "primitive",
                "collective",
                "is_collective",
                "is_point_to_point",
                "bytes",
                "num_messages",
                "block_size",
                "wall_s",
            }
        })
        normalized.append(standard)
    return normalized


def _communication_metrics_from_items(communication):
    if isinstance(communication, dict):
        communication = (communication,)
    if not isinstance(communication, (list, tuple)):
        return {
            "bytes": 0,
            "num_messages": 0,
            "max_block_size": 0,
            "bytes_by_collective": {},
            "messages_by_collective": {},
            "wall_s_by_collective": {},
            "dominant_collective": None,
            "bytes_by_primitive": {},
            "messages_by_primitive": {},
            "wall_s_by_primitive": {},
            "dominant_primitive": None,
            "point_to_point_bytes": 0,
            "num_point_to_point_messages": 0,
        }
    total_bytes = 0
    total_messages = 0
    max_block_size = 0
    bytes_by_collective = {}
    messages_by_collective = {}
    wall_s_by_collective = {}
    bytes_by_primitive = {}
    messages_by_primitive = {}
    wall_s_by_primitive = {}
    point_to_point_bytes = 0
    point_to_point_messages = 0
    for item in communication:
        if not isinstance(item, dict):
            continue
        collective = item.get("collective")
        if collective is None:
            collective = item.get("kind")
        collective = str(collective) if collective is not None else "unknown"
        primitive = item.get("primitive")
        if primitive is None:
            primitive = item.get("kind", collective)
        primitive = str(primitive) if primitive is not None else "unknown"
        try:
            bytes_value = int(item.get("bytes", 0) or 0)
        except (TypeError, ValueError):
            bytes_value = 0
        try:
            num_messages = int(item.get("num_messages", 1) or 0)
        except (TypeError, ValueError):
            num_messages = 0
        try:
            block_size = int(item.get("block_size", 0) or 0)
        except (TypeError, ValueError):
            block_size = 0
        if not block_size and num_messages:
            block_size = bytes_value // num_messages
        total_bytes += bytes_value
        total_messages += num_messages
        bytes_by_collective[collective] = bytes_by_collective.get(collective, 0) + bytes_value
        messages_by_collective[collective] = messages_by_collective.get(collective, 0) + num_messages
        bytes_by_primitive[primitive] = bytes_by_primitive.get(primitive, 0) + bytes_value
        messages_by_primitive[primitive] = messages_by_primitive.get(primitive, 0) + num_messages
        if primitive == "point_to_point" or bool(item.get("is_point_to_point")):
            point_to_point_bytes += bytes_value
            point_to_point_messages += num_messages
        try:
            wall_s = float(item.get("wall_s", 0) or 0)
        except (TypeError, ValueError):
            wall_s = 0.0
        wall_s_by_collective[collective] = wall_s_by_collective.get(collective, 0.0) + wall_s
        wall_s_by_primitive[primitive] = wall_s_by_primitive.get(primitive, 0.0) + wall_s
        if block_size > max_block_size:
            max_block_size = block_size
    dominant_collective = None
    if bytes_by_collective:
        dominant_collective = max(
            sorted(bytes_by_collective),
            key=lambda key: bytes_by_collective[key],
        )
    dominant_primitive = None
    if bytes_by_primitive:
        dominant_primitive = max(
            sorted(bytes_by_primitive),
            key=lambda key: bytes_by_primitive[key],
        )
    return {
        "bytes": int(total_bytes),
        "num_messages": int(total_messages),
        "max_block_size": int(max_block_size),
        "bytes_by_collective": bytes_by_collective,
        "messages_by_collective": messages_by_collective,
        "wall_s_by_collective": wall_s_by_collective,
        "dominant_collective": dominant_collective,
        "bytes_by_primitive": bytes_by_primitive,
        "messages_by_primitive": messages_by_primitive,
        "wall_s_by_primitive": wall_s_by_primitive,
        "dominant_primitive": dominant_primitive,
        "point_to_point_bytes": int(point_to_point_bytes),
        "num_point_to_point_messages": int(point_to_point_messages),
    }


def _communication_bytes_from_items(communication):
    return _communication_metrics_from_items(communication)["bytes"]


def _profile_optional_shape(value):
    if value is None:
        return None
    return _profile_shape(value)


def _standard_communication_profile(payload):
    communication = payload.get("communication")
    metrics = _communication_metrics_from_items(communication)
    comm_bytes = metrics["bytes"]
    read_bytes = _int_profile_value(payload.get("read_bytes"))
    write_bytes = _int_profile_value(payload.get("write_bytes"))
    copy_bytes = _int_profile_value(payload.get("copy_bytes"))
    working_set_bytes = read_bytes + write_bytes + copy_bytes + comm_bytes
    distributed_modes = payload.get("distributed_modes") or ()
    if isinstance(distributed_modes, str):
        distributed_modes = (distributed_modes,)
    items = communication
    if isinstance(items, dict):
        items = (items,)
    num_collectives = len(items) if isinstance(items, (list, tuple)) else 0
    return {
        "communication_required": bool(comm_bytes or metrics["num_messages"]),
        "distributed": bool(payload.get("lowering") == "distributed" or distributed_modes),
        "distributed_modes": [str(mode) for mode in distributed_modes],
        "rank": payload.get("rank"),
        "world_size": payload.get("world_size"),
        "local_shape": _profile_optional_shape(payload.get("local_shape")),
        "global_shape": _profile_optional_shape(payload.get("global_shape")),
        "bytes": int(comm_bytes),
        "num_messages": int(metrics["num_messages"]),
        "num_collectives": int(num_collectives),
        "max_block_size": int(metrics["max_block_size"]),
        "bytes_by_collective": metrics["bytes_by_collective"],
        "messages_by_collective": metrics["messages_by_collective"],
        "wall_s_by_collective": metrics["wall_s_by_collective"],
        "wall_s": sum(float(value or 0.0) for value in metrics["wall_s_by_collective"].values()),
        "dominant_collective": metrics["dominant_collective"],
        "bytes_by_primitive": metrics["bytes_by_primitive"],
        "messages_by_primitive": metrics["messages_by_primitive"],
        "wall_s_by_primitive": metrics["wall_s_by_primitive"],
        "dominant_primitive": metrics["dominant_primitive"],
        "point_to_point_bytes": int(metrics["point_to_point_bytes"]),
        "num_point_to_point_messages": int(metrics["num_point_to_point_messages"]),
        "working_set_bytes": int(working_set_bytes),
        "communication_fraction_of_working_set": _safe_fraction(comm_bytes, working_set_bytes),
        "recommended_action": "reduce_communication" if comm_bytes or metrics["num_messages"] else None,
    }


def _execution_resource_profile(payload):
    workspace_required = int(payload.get("workspace_bytes") or 0)
    workspace_provided = payload.get("workspace_provided")
    workspace_nbytes = payload.get("workspace_nbytes")
    workspace_provided_bytes = None
    if workspace_nbytes is not None:
        try:
            workspace_provided_bytes = int(workspace_nbytes or 0)
        except (TypeError, ValueError):
            workspace_provided_bytes = 0
    elif workspace_provided:
        workspace_provided_bytes = 0
    workspace_slack = None
    if workspace_provided_bytes is not None:
        workspace_slack = workspace_provided_bytes - workspace_required
    profile = {
        "stream_provided": bool(payload.get("stream_provided")),
        "stream_type": payload.get("stream_type"),
        "workspace_provided": bool(payload.get("workspace_provided")),
        "workspace_required_bytes": workspace_required,
        "workspace_provided_bytes": workspace_provided_bytes,
        "workspace_slack_bytes": workspace_slack,
        "workspace_device_kind": payload.get("workspace_device_kind"),
        "workspace_device_index": payload.get("workspace_device_index"),
    }
    if "workspace_released" in payload:
        profile["workspace_released"] = bool(payload.get("workspace_released"))
    return profile


def _temporary_memory_profile(payload):
    resources = _execution_resource_profile(payload)
    return {
        "workspace_required_bytes": resources["workspace_required_bytes"],
        "workspace_provided_bytes": resources["workspace_provided_bytes"],
        "workspace_slack_bytes": resources["workspace_slack_bytes"],
        "largest_intermediate": int(payload.get("largest_intermediate") or 0),
    }


def _standard_workspace_profile(payload):
    resources = _execution_resource_profile(payload)
    required_bytes = resources["workspace_required_bytes"]
    provided_bytes = resources["workspace_provided_bytes"]
    workspace_sufficient = None
    if provided_bytes is not None:
        workspace_sufficient = provided_bytes >= required_bytes
    profile = {
        "workspace_required": bool(required_bytes),
        "workspace_required_bytes": required_bytes,
        "workspace_provided": resources["workspace_provided"],
        "workspace_provided_bytes": provided_bytes,
        "workspace_sufficient": workspace_sufficient,
        "workspace_slack_bytes": resources["workspace_slack_bytes"],
        "workspace_device_kind": resources["workspace_device_kind"],
        "workspace_device_index": resources["workspace_device_index"],
        "largest_intermediate": _int_profile_value(
            payload.get("largest_intermediate"),
            payload.get("largest_intermediate_elements"),
        ),
    }
    if "workspace_released" in resources:
        profile["workspace_released"] = resources["workspace_released"]
    return profile


def _standard_async_profile(payload):
    stream_provided = bool(payload.get("stream_provided"))
    return {
        "stream_provided": stream_provided,
        "stream_type": payload.get("stream_type"),
        "async_requested": stream_provided,
        "execution_mode": "external_stream" if stream_provided else "synchronous_or_default_stream",
    }


def _is_int_sequence(value):
    if not isinstance(value, (list, tuple)):
        return False
    for item in value:
        try:
            int(item)
        except (TypeError, ValueError):
            return False
    return True


def _profile_shape(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return value
    try:
        return tuple(int(dim) for dim in value)
    except (TypeError, ValueError):
        return tuple(value)


def _profile_shape_list(value):
    if value is None:
        return []
    if _is_int_sequence(value):
        return [_profile_shape(value)]
    if isinstance(value, (list, tuple)):
        return [_profile_shape(item) for item in value]
    return [_profile_shape(value)]


def _layout_attribute_at(values, index, count, default=None):
    if values is None:
        return default
    if count == 1 and _is_int_sequence(values):
        return values
    if isinstance(values, (list, tuple)) and not isinstance(values, (str, bytes)):
        if index < len(values):
            return values[index]
        return default
    return values


def _profile_strides(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return value
    try:
        return tuple(int(dim) for dim in value)
    except (TypeError, ValueError):
        return tuple(value)


def _profile_contiguous(value):
    if value is None:
        return None
    return bool(value)


def _standard_layout_item(shape, strides=None, order=None, contiguous=None):
    if isinstance(order, (list, tuple)) and not isinstance(order, (str, bytes)):
        order_value = [str(item) if item is not None else "unknown" for item in order]
    else:
        order_value = str(order) if order is not None else "unknown"
    if isinstance(contiguous, (list, tuple)) and not isinstance(contiguous, (str, bytes)):
        contiguous_value = [_profile_contiguous(item) for item in contiguous]
    else:
        contiguous_value = _profile_contiguous(contiguous)
    return {
        "shape": _profile_shape(shape),
        "strides": _profile_strides(strides),
        "order": order_value,
        "contiguous": contiguous_value,
    }


def _flatten_profile_operands(operands):
    if not isinstance(operands, (list, tuple)):
        return []
    flattened = []
    for operand in operands:
        if isinstance(operand, dict):
            flattened.append(operand)
        elif isinstance(operand, (list, tuple)):
            flattened.extend(_flatten_profile_operands(operand))
    return flattened


def _standard_layout_profile(payload):
    input_shapes = _profile_shape_list(payload.get("input_shapes"))
    operands = payload.get("operands") or payload.get("task_operands") or ()
    flat_operands = _flatten_profile_operands(operands)
    if flat_operands and (
        not input_shapes
        or payload.get("task_operands") is not None
        or len(flat_operands) != len(input_shapes)
    ):
        input_shapes = [
            _profile_shape(operand.get("shape"))
            for operand in flat_operands
            if operand.get("shape") is not None
        ]
    input_strides = payload.get("input_strides")
    input_orders = payload.get("input_orders")
    input_contiguous = payload.get("input_contiguous")
    input_layouts = []
    for index, shape in enumerate(input_shapes):
        operand = flat_operands[index] if index < len(flat_operands) else None
        if not isinstance(operand, dict):
            operand = {}
        strides = _layout_attribute_at(input_strides, index, len(input_shapes))
        order = _layout_attribute_at(input_orders, index, len(input_shapes))
        contiguous = _layout_attribute_at(input_contiguous, index, len(input_shapes))
        if strides is None:
            strides = operand.get("strides")
        if order is None:
            order = operand.get("order")
        if contiguous is None:
            contiguous = operand.get("contiguous")
        input_layouts.append(
            _standard_layout_item(
                shape,
                strides=strides,
                order=order,
                contiguous=contiguous,
            )
        )

    axis_copy_bytes = _int_profile_value(payload.get("axis_permutation_copy_bytes"))
    layout_transform_copy_bytes = _int_profile_value(payload.get("layout_transform_copy_bytes"))
    has_layout_copy_split = (
        payload.get("input_layout_transform_copy_bytes") is not None
        or payload.get("output_layout_transform_copy_bytes") is not None
    )
    input_layout_transform_copy_bytes = _int_profile_value(
        payload.get("input_layout_transform_copy_bytes")
    )
    output_layout_transform_copy_bytes = _int_profile_value(
        payload.get("output_layout_transform_copy_bytes")
    )
    copy_profile = payload.get("copy_profile") or {}
    copy_kind = copy_profile.get("copy_kind")
    if axis_copy_bytes:
        layout_transform_kind = "axis_permutation"
        estimated_layout_copy_bytes = axis_copy_bytes
    elif layout_transform_copy_bytes:
        layout_transform_kind = "layout_transform"
        estimated_layout_copy_bytes = layout_transform_copy_bytes
    elif copy_kind == "layout_transform":
        layout_transform_kind = "layout_transform"
        estimated_layout_copy_bytes = _int_profile_value(payload.get("copy_bytes"))
    else:
        layout_transform_kind = "none"
        estimated_layout_copy_bytes = 0

    profile = {
        "input_layouts": input_layouts,
        "output_layout": _standard_layout_item(
            payload.get("output_shape"),
            strides=payload.get("output_strides"),
            order=payload.get("output_order"),
            contiguous=payload.get("output_contiguous"),
        ),
        "layout_hint": payload.get("layout_hint"),
        "layout_transform_required": bool(estimated_layout_copy_bytes or layout_transform_kind != "none"),
        "layout_transform_kind": layout_transform_kind,
        "estimated_layout_copy_bytes": estimated_layout_copy_bytes,
    }
    if has_layout_copy_split:
        profile["input_layout_transform_copy_bytes"] = input_layout_transform_copy_bytes
        profile["output_layout_transform_copy_bytes"] = output_layout_transform_copy_bytes
    return profile


def _dtype_numeric_kind(dtype_name):
    if dtype_name is None:
        return "unknown"
    lowered = str(dtype_name).lower()
    if not lowered or lowered == "none":
        return "unknown"
    if "complex" in lowered:
        return "complex"
    if "float" in lowered or "double" in lowered or "half" in lowered:
        return "real"
    if "int" in lowered or "long" in lowered or "short" in lowered:
        return "integer"
    if "bool" in lowered:
        return "bool"
    return "unknown"


def _dtype_precision_bits(dtype_name):
    if dtype_name is None:
        return None
    lowered = str(dtype_name).lower()
    if not lowered or lowered == "none":
        return None
    for bits in (256, 128, 96, 80, 64, 32, 16, 8, 4, 1):
        if str(bits) in lowered:
            return bits
    return None


def _normalize_dtype_name(dtype_name):
    if dtype_name is None:
        return None
    name = str(dtype_name).strip()
    if not name:
        return None
    lowered = name.lower()
    if lowered == "none":
        return None
    for prefix in ("torch.", "jax.", "jnp.", "numpy.", "np.", "cupy.", "cp."):
        if lowered.startswith(prefix):
            return name[len(prefix):]
    return name


def _flatten_profile_sequence(values):
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    if isinstance(values, (list, tuple)):
        flattened = []
        for item in values:
            flattened.extend(_flatten_profile_sequence(item))
        return flattened
    return [values]


def _standard_dtype_profile(payload):
    dtype = payload.get("dtype")
    output_dtype = payload.get("output_dtype")
    input_dtypes = _flatten_profile_sequence(payload.get("input_dtypes") or ())
    input_dtypes = [
        normalized
        for normalized in (_normalize_dtype_name(item) for item in input_dtypes)
        if normalized is not None
    ]
    if dtype is None:
        dtype = output_dtype
    if dtype is None and input_dtypes:
        dtype = input_dtypes[0]
    dtype = _normalize_dtype_name(dtype)
    output_dtype = _normalize_dtype_name(output_dtype) if output_dtype is not None else dtype
    numeric_kind = _dtype_numeric_kind(dtype)
    precision_bits = _dtype_precision_bits(dtype)
    component_bits = precision_bits
    if numeric_kind == "complex" and precision_bits is not None:
        component_bits = precision_bits // 2
    return {
        "dtype": dtype,
        "output_dtype": output_dtype,
        "input_dtypes": input_dtypes,
        "input_dtype_set": sorted(set(input_dtypes)),
        "numeric_kind": numeric_kind,
        "precision_bits": precision_bits,
        "component_bits": component_bits,
        "mixed_input_dtypes": len(set(input_dtypes)) > 1,
    }


def _sync_dtype_payload_fields(payload):
    profile = payload.get("dtype_profile")
    if not isinstance(profile, dict):
        return
    payload["dtype"] = profile.get("dtype")
    payload["output_dtype"] = profile.get("output_dtype")


def _optional_int_profile_value(value):
    if value is None:
        return None
    return _int_profile_value(value)


def _device_kind_from_value(device):
    if device is None:
        return None
    kind = getattr(device, "kind", None)
    if kind is not None:
        return str(kind)
    lowered = str(device).lower()
    if lowered.startswith(("cuda", "gpu")):
        return "cuda"
    if lowered.startswith("cpu") or lowered == "host":
        return "cpu"
    for marker in ("rocm", "mps", "tpu", "distributed"):
        if lowered.startswith(marker):
            return marker
    return None


def _standard_device_profile(payload):
    device = payload.get("device")
    backend = payload.get("backend")
    device_kind = payload.get("device_kind")
    if device_kind is None:
        device_kind = _device_kind_from_value(device)
    device_index = _optional_int_profile_value(payload.get("device_index"))
    world_size = _optional_int_profile_value(payload.get("world_size"))
    rank = _optional_int_profile_value(payload.get("rank"))
    distributed = bool(
        payload.get("lowering") == "distributed"
        or payload.get("is_distributed")
        or payload.get("distributed_modes") is not None
        or (world_size is not None and world_size > 1)
    )
    normalized_kind = str(device_kind).lower() if device_kind is not None else None
    if distributed:
        array_location = "distributed"
    elif normalized_kind in {"cuda", "gpu", "rocm", "mps", "tpu"}:
        array_location = "device"
    elif normalized_kind in {"cpu", "host"} or backend == "numpy":
        array_location = "host"
    else:
        array_location = "unknown"
    return {
        "backend": backend,
        "device": str(device) if device is not None else None,
        "device_kind": device_kind,
        "device_index": device_index,
        "array_location": array_location,
        "distributed": distributed,
        "rank": rank,
        "world_size": world_size,
        "local_shape": payload.get("local_shape"),
        "global_shape": payload.get("global_shape"),
    }


def _contraction_execute_parallelism_hint(lowering, payload=None):
    payload = payload or {}
    if lowering == "distributed" or payload.get("distributed_modes"):
        return "distributed_contraction"
    if lowering in ("grouped_gemm", "block_grouped_gemm") or int(payload.get("num_grouped_tasks") or 0):
        return "grouped_backend_kernel"
    if lowering in ("batched_gemm", "strided_batched_gemm") or int(payload.get("num_batched_gemm") or 0):
        return "batched_backend_kernel"
    if lowering == "gemm":
        return "single_gemm_kernel"
    if lowering:
        return str(lowering)
    return "unknown_contraction_kernel"


def _infer_contraction_execute_fallback_target(payload):
    if not payload.get("fallback_reason"):
        return None
    lowering = payload.get("lowering")
    fallback_from = payload.get("fallback_from")
    if lowering in ("fallback_tensordot", "fallback_einsum"):
        return "loop_matmul"
    if lowering in ("grouped_gemm", "block_grouped_gemm") and fallback_from == "grouped_gemm":
        return "bucketed_grouped_gemm"
    if fallback_from in ("batched_gemm", "strided_batched_gemm", "gemm"):
        return "loop_matmul"
    return None


def _contraction_execute_event_practical_profile(payload):
    targets = []
    lowering = payload.get("lowering")
    compute_class = payload.get("compute_class") or _contraction_execute_compute_class(lowering)
    flops = payload.get("flops")
    read_bytes = int(payload.get("read_bytes") or 0)
    write_bytes = int(payload.get("write_bytes") or 0)
    copy_bytes = int(payload.get("copy_bytes") or 0)
    communication_metrics = _communication_metrics_from_items(payload.get("communication"))
    communication_bytes = communication_metrics["bytes"]
    memory_bytes = read_bytes + write_bytes
    working_set_bytes = memory_bytes + copy_bytes + communication_bytes
    execution_resources = _execution_resource_profile(payload)
    temporary_memory = _temporary_memory_profile(payload)
    num_gemm = int(payload.get("num_gemm") or 0)
    num_batched_gemm = int(payload.get("num_batched_gemm") or 0)
    num_grouped_tasks = int(payload.get("num_grouped_tasks") or 0)
    num_rhs_loop_calls = int(payload.get("num_rhs_loop_calls") or 0)
    rhs_count = _infer_rhs_count(payload)
    rhs_batching_candidate = rhs_count > 1

    if num_rhs_loop_calls:
        _append_unique(targets, "batch_rhs_hop")
    if payload.get("fallback_reason"):
        _append_unique(targets, "remove_backend_fallback")
    if communication_bytes:
        _append_unique(targets, "reduce_communication")
    if copy_bytes:
        _append_unique(targets, "reduce_layout_or_device_copies")

    if num_rhs_loop_calls:
        primary_issue = "python_rhs_loop"
        parallelism_hint = "rhs_batching_needed"
        measurement_focus = "rhs_loop"
    elif payload.get("fallback_reason"):
        primary_issue = "backend_fallback"
        parallelism_hint = "missing_backend_kernel"
        measurement_focus = "backend_fallback"
    elif communication_bytes:
        primary_issue = "communication"
        parallelism_hint = "distributed_contraction"
        measurement_focus = "communication"
    elif copy_bytes:
        primary_issue = "copy_overhead"
        parallelism_hint = _contraction_execute_parallelism_hint(lowering, payload)
        measurement_focus = "copy_overhead"
    else:
        primary_issue = "compute_kernel"
        parallelism_hint = _contraction_execute_parallelism_hint(lowering, payload)
        measurement_focus = "kernel"
    copy_fraction = _safe_fraction(copy_bytes, working_set_bytes)
    communication_fraction = _safe_fraction(communication_bytes, working_set_bytes)
    precision_evidence = {
        "lowering": lowering,
        "flops": int(flops or 0),
        "working_set_bytes_estimate": int(working_set_bytes),
        "copy_fraction": copy_fraction,
        "communication_fraction": communication_fraction,
        "num_gemm": num_gemm,
        "num_batched_gemm": num_batched_gemm,
        "num_grouped_tasks": num_grouped_tasks,
        "num_rhs_loop_calls": num_rhs_loop_calls,
    }
    if payload.get("fallback_from") is not None:
        precision_evidence["fallback_from"] = payload.get("fallback_from")
    if payload.get("fallback_to") is not None:
        precision_evidence["fallback_to"] = payload.get("fallback_to")

    return {
        "workload_class": compute_class,
        "measurement_scope": "backend_contraction_execute",
        "precision_level": "lowering_counters",
        "dominant_cost_kind": primary_issue,
        "primary_kernel": str(lowering) if lowering else None,
        "primary_issue": primary_issue,
        "parallelism_hint": parallelism_hint,
        "cost_model": _practical_cost_model(
            "backend_lowering_estimate",
            flops_estimate=flops,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            copy_bytes=copy_bytes,
            communication_bytes=communication_bytes,
            working_set_bytes=working_set_bytes,
        ),
        "dominant_work": {
            "unit": "backend_contraction_lowering",
            "lowering": str(lowering) if lowering else None,
            "num_gemm": num_gemm,
            "num_batched_gemm": num_batched_gemm,
            "num_grouped_tasks": num_grouped_tasks,
            "num_rhs_loop_calls": num_rhs_loop_calls,
        },
        "parallelization": {
            "unit": "backend_lowering",
            "backend_kernel": str(lowering) if lowering else None,
            "parallelism_hint": parallelism_hint,
            "batched_kernel": lowering in ("batched_gemm", "strided_batched_gemm") or bool(num_batched_gemm),
            "grouped_kernel": lowering in ("grouped_gemm", "block_grouped_gemm") or bool(num_grouped_tasks),
            "rhs_batching_candidate": rhs_batching_candidate,
        },
        "measurement_limits": _measurement_limits("python_wall_time_if_event_timed"),
        "dominant_operation": {
            "kind": str(lowering) if lowering else None,
            "num_gemm": num_gemm,
            "num_batched_gemm": num_batched_gemm,
            "num_grouped_tasks": num_grouped_tasks,
            "num_rhs_loop_calls": num_rhs_loop_calls,
        },
        "execution_resources": execution_resources,
        "temporary_memory": temporary_memory,
        "key_metrics": {
            "lowering": lowering,
            "fallback_from": payload.get("fallback_from"),
            "fallback_to": payload.get("fallback_to"),
            "fallback_policy": payload.get("fallback_policy"),
            "fallback_reason": payload.get("fallback_reason"),
            "num_gemm": int(payload.get("num_gemm") or 0),
            "num_batched_gemm": int(payload.get("num_batched_gemm") or 0),
            "num_grouped_tasks": int(payload.get("num_grouped_tasks") or 0),
            "num_blocks": int(payload.get("num_blocks") or 0),
            "num_shape_buckets": int(payload.get("num_shape_buckets") or 0),
            "copy_fraction_of_working_set": copy_fraction,
            "communication_bytes": communication_bytes,
            "communication_num_messages": communication_metrics["num_messages"],
            "max_communication_block_size": communication_metrics["max_block_size"],
            "communication_bytes_by_collective": communication_metrics["bytes_by_collective"],
            "communication_messages_by_collective": communication_metrics["messages_by_collective"],
            "communication_wall_s_by_collective": communication_metrics["wall_s_by_collective"],
            "dominant_communication_collective": communication_metrics["dominant_collective"],
            "communication_fraction_of_working_set": communication_fraction,
            "arithmetic_intensity_flops_per_byte": _safe_fraction(flops, memory_bytes),
            "workspace_required_bytes": execution_resources["workspace_required_bytes"],
            "workspace_provided_bytes": execution_resources["workspace_provided_bytes"],
            "workspace_slack_bytes": execution_resources["workspace_slack_bytes"],
            "largest_intermediate": temporary_memory["largest_intermediate"],
        },
        "optimization_targets": targets,
        "diagnosis": {
            "measurement_focus": measurement_focus,
            "primary_issue": primary_issue,
            "parallelism_hint": parallelism_hint,
            "fallback_candidate": bool(payload.get("fallback_reason")),
            "communication_bound_candidate": bool(communication_bytes),
            "copy_bound_candidate": bool(copy_bytes),
            "precision_evidence": {
                **precision_evidence,
                "execution_resources": execution_resources,
            },
            "recommended_action": _first_target(targets, "inspect_contraction_execute"),
        },
    }


def _standard_copy_profile(payload):
    copy_bytes = _int_profile_value(payload.get("copy_bytes"))
    axis_copy_bytes = _int_profile_value(payload.get("axis_permutation_copy_bytes"))
    layout_transform_copy_bytes = _int_profile_value(payload.get("layout_transform_copy_bytes"))
    has_layout_copy_split = (
        payload.get("input_layout_transform_copy_bytes") is not None
        or payload.get("output_layout_transform_copy_bytes") is not None
    )
    input_layout_transform_copy_bytes = _int_profile_value(
        payload.get("input_layout_transform_copy_bytes")
    )
    output_layout_transform_copy_bytes = _int_profile_value(
        payload.get("output_layout_transform_copy_bytes")
    )
    lowering = str(payload.get("lowering") or "")

    if axis_copy_bytes:
        copy_kind = "layout_transform"
        copy_source = "axis_permutation_copy_bytes"
    elif layout_transform_copy_bytes:
        copy_kind = "layout_transform"
        copy_source = "layout_transform_copy_bytes"
    elif copy_bytes and lowering in ("grouped_gemm", "block_grouped_gemm"):
        copy_kind = "packing_or_bucketed_execution"
        copy_source = "copy_bytes"
    elif copy_bytes and payload.get("fallback_reason"):
        copy_kind = "fallback_or_layout_transform"
        copy_source = "copy_bytes"
    elif copy_bytes:
        copy_kind = "unspecified"
        copy_source = "copy_bytes"
    else:
        copy_kind = "none"
        copy_source = None

    profile = {
        "copy_required": bool(copy_bytes),
        "copy_bytes": copy_bytes,
        "copy_kind": copy_kind,
        "copy_source": copy_source,
    }
    if has_layout_copy_split:
        profile["copy_breakdown"] = {
            "layout_transform_bytes": layout_transform_copy_bytes,
            "input_layout_transform_bytes": input_layout_transform_copy_bytes,
            "output_layout_transform_bytes": output_layout_transform_copy_bytes,
        }
    return profile


def _default_copy_location(payload):
    device_profile = payload.get("device_profile") or _standard_device_profile(payload)
    location = device_profile.get("array_location")
    if location in {"host", "device", "distributed"}:
        return location
    return None


def _copy_device_kind(payload, location, explicit_name):
    explicit = payload.get(explicit_name)
    if explicit is not None:
        return explicit
    if location == "host":
        return "cpu"
    if location in {"device", "distributed"}:
        device_profile = payload.get("device_profile") or _standard_device_profile(payload)
        return device_profile.get("device_kind")
    return None


def _copy_domain(copy_kind, source_location, destination_location):
    if copy_kind == "layout_transform":
        return "layout_transform"
    if source_location and destination_location and source_location != destination_location:
        return "{0}_to_{1}".format(source_location, destination_location)
    if source_location == "distributed" or destination_location == "distributed":
        return "distributed_redistribution"
    if source_location == "device" and destination_location == "device":
        return "device_to_device"
    if source_location == "host" and destination_location == "host":
        return "host_to_host"
    if source_location or destination_location:
        return "memory_copy"
    return "unknown"


def _copy_bandwidth_scope(copy_domain, copy_kind, source_location, destination_location):
    if copy_kind == "layout_transform":
        if source_location == "device" or destination_location == "device":
            return "device_layout"
        if source_location == "distributed" or destination_location == "distributed":
            return "distributed_layout"
        if source_location == "host" or destination_location == "host":
            return "host_layout"
        return "layout"
    if copy_domain in {"host_to_device", "device_to_host"}:
        return "host_device"
    if copy_domain == "device_to_device":
        return "device_device"
    if copy_domain == "distributed_redistribution":
        return "distributed_network"
    if copy_domain == "host_to_host":
        return "host_memory"
    return "unknown"


def _standard_copy_movement_profile(payload):
    copy_profile = payload.get("copy_profile") or _standard_copy_profile(payload)
    copy_required = bool(copy_profile.get("copy_required"))
    copy_kind = copy_profile.get("copy_kind")
    source_location = payload.get("copy_source_location")
    destination_location = payload.get("copy_destination_location")
    if source_location is None or destination_location is None:
        default_location = _default_copy_location(payload)
        if source_location is None:
            source_location = default_location
        if destination_location is None:
            destination_location = default_location
    if not copy_required:
        source_location = None
        destination_location = None
    copy_domain = "none"
    if copy_required:
        copy_domain = _copy_domain(copy_kind, source_location, destination_location)
    source_device_kind = None
    destination_device_kind = None
    if copy_required:
        source_device_kind = _copy_device_kind(payload, source_location, "copy_source_device_kind")
        destination_device_kind = _copy_device_kind(payload, destination_location, "copy_destination_device_kind")
    return {
        "copy_required": copy_required,
        "copy_bytes": _int_profile_value(copy_profile.get("copy_bytes")),
        "copy_kind": copy_kind,
        "copy_source": copy_profile.get("copy_source"),
        "copy_domain": copy_domain,
        "source_location": source_location,
        "destination_location": destination_location,
        "source_device_kind": source_device_kind,
        "destination_device_kind": destination_device_kind,
        "route": [source_location, destination_location] if copy_required else [],
        "bandwidth_scope": _copy_bandwidth_scope(
            copy_domain,
            copy_kind,
            source_location,
            destination_location,
        ) if copy_required else "none",
    }


def _optional_float_profile_value(*values):
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _cost_profile_comm_bytes(payload):
    comm_value = _first_present(payload, ("comm_bytes", "estimated_comm_bytes"))
    if comm_value is not None:
        return _int_profile_value(comm_value)
    communication_profile = payload.get("communication_profile")
    if isinstance(communication_profile, dict) and communication_profile.get("bytes") is not None:
        return _int_profile_value(communication_profile.get("bytes"))
    return _communication_bytes_from_items(payload.get("communication"))


def _standard_cost_profile(payload):
    flops = _int_profile_value(payload.get("flops"), payload.get("flops_estimate"), payload.get("estimated_flops"))
    read_bytes = _int_profile_value(payload.get("read_bytes"), payload.get("estimated_read_bytes"))
    write_bytes = _int_profile_value(payload.get("write_bytes"), payload.get("estimated_write_bytes"))
    copy_bytes = _int_profile_value(payload.get("copy_bytes"), payload.get("estimated_copy_bytes"))
    comm_bytes = _cost_profile_comm_bytes(payload)
    workspace_bytes = _int_profile_value(
        payload.get("workspace_bytes"),
        payload.get("required_workspace_bytes"),
        payload.get("estimated_workspace_bytes"),
    )
    peak_value = _first_present(payload, ("peak_bytes", "estimated_peak_bytes"))
    if peak_value is None:
        peak_bytes = write_bytes + copy_bytes + workspace_bytes
    else:
        peak_bytes = _int_profile_value(peak_value)
    memory_bytes = read_bytes + write_bytes
    working_set_bytes = memory_bytes + copy_bytes + comm_bytes
    return {
        "source": str(payload.get("cost_source") or "profile_fields"),
        "flops": flops,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "copy_bytes": copy_bytes,
        "comm_bytes": comm_bytes,
        "workspace_bytes": workspace_bytes,
        "peak_bytes": peak_bytes,
        "memory_bytes": memory_bytes,
        "working_set_bytes": working_set_bytes,
        "largest_intermediate": _int_profile_value(
            payload.get("largest_intermediate"),
            payload.get("largest_intermediate_elements"),
        ),
        "arithmetic_intensity_flops_per_byte": _safe_fraction(flops, memory_bytes),
        "effective_arithmetic_intensity_flops_per_byte": _safe_fraction(flops, working_set_bytes),
        "copy_fraction_of_working_set": _safe_fraction(copy_bytes, working_set_bytes),
        "communication_fraction_of_working_set": _safe_fraction(comm_bytes, working_set_bytes),
        "workspace_fraction_of_peak": _safe_fraction(workspace_bytes, peak_bytes),
        "compute_s": _optional_float_profile_value(payload.get("compute_s"), payload.get("estimated_compute_s")),
        "memory_s": _optional_float_profile_value(payload.get("memory_s"), payload.get("estimated_memory_s")),
        "copy_s": _optional_float_profile_value(payload.get("copy_s"), payload.get("estimated_copy_s")),
        "comm_s": _optional_float_profile_value(payload.get("comm_s"), payload.get("estimated_comm_s")),
        "total_s": _optional_float_profile_value(payload.get("total_s"), payload.get("estimated_total_s")),
        "estimated_time_s": _optional_float_profile_value(payload.get("estimated_time_s")),
    }


def _aggregate_cost_profile(payload, total_memory_bytes=None, derived_cost=None):
    derived_cost = derived_cost or {}
    read_bytes = _int_profile_value(payload.get("total_read_bytes"), payload.get("read_bytes"))
    write_bytes = _int_profile_value(payload.get("total_write_bytes"), payload.get("write_bytes"))
    copy_bytes = _int_profile_value(payload.get("total_copy_bytes"), payload.get("copy_bytes"))
    comm_bytes = _int_profile_value(payload.get("total_comm_bytes"), payload.get("comm_bytes"))
    workspace_bytes = _int_profile_value(payload.get("max_workspace_bytes"), payload.get("workspace_bytes"))
    peak_bytes = _int_profile_value(payload.get("max_peak_bytes"), payload.get("peak_bytes"))
    flops = _int_profile_value(
        payload.get("total_flops_estimate"),
        payload.get("flops_estimate"),
        payload.get("flops"),
    )
    largest_intermediate = _int_profile_value(
        payload.get("max_largest_intermediate_elements"),
        payload.get("largest_intermediate_elements"),
        payload.get("largest_intermediate"),
    )
    if total_memory_bytes is None:
        memory_bytes = read_bytes + write_bytes
    else:
        memory_bytes = _int_profile_value(total_memory_bytes)
    working_set_bytes = _int_profile_value(
        derived_cost.get("working_set_bytes_estimate"),
        derived_cost.get("working_set_bytes"),
    )
    if working_set_bytes == 0:
        working_set_bytes = memory_bytes + copy_bytes + comm_bytes
    if peak_bytes == 0:
        peak_bytes = write_bytes + copy_bytes + workspace_bytes
    return {
        "source": "aggregate_profile_fields",
        "flops": flops,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "copy_bytes": copy_bytes,
        "comm_bytes": comm_bytes,
        "workspace_bytes": workspace_bytes,
        "peak_bytes": peak_bytes,
        "memory_bytes": memory_bytes,
        "working_set_bytes": working_set_bytes,
        "largest_intermediate": largest_intermediate,
        "arithmetic_intensity_flops_per_byte": _safe_fraction(flops, memory_bytes),
        "effective_arithmetic_intensity_flops_per_byte": _safe_fraction(flops, working_set_bytes),
        "copy_fraction_of_working_set": _safe_fraction(copy_bytes, working_set_bytes),
        "communication_fraction_of_working_set": _safe_fraction(comm_bytes, working_set_bytes),
        "workspace_fraction_of_peak": _safe_fraction(workspace_bytes, peak_bytes),
        "compute_s": _optional_float_profile_value(payload.get("compute_s"), payload.get("estimated_compute_s")),
        "memory_s": _optional_float_profile_value(payload.get("memory_s"), payload.get("estimated_memory_s")),
        "copy_s": _optional_float_profile_value(payload.get("copy_s"), payload.get("estimated_copy_s")),
        "comm_s": _optional_float_profile_value(payload.get("comm_s"), payload.get("estimated_comm_s")),
        "total_s": _optional_float_profile_value(payload.get("total_s"), payload.get("estimated_total_s")),
        "estimated_time_s": _optional_float_profile_value(payload.get("estimated_time_s")),
    }


def _infer_rhs_count(payload):
    rhs_count = _int_profile_value(payload.get("num_rhs"))
    if rhs_count:
        return rhs_count
    input_shapes = payload.get("input_shapes") or ()
    if len(input_shapes) != 1:
        return 0
    try:
        shape = tuple(input_shapes[0])
    except TypeError:
        return 0
    if len(shape) == 2:
        return _int_profile_value(shape[-1])
    return 0


def _rhs_batched_execution_used(payload, lowering):
    if lowering in ("batched_gemm", "strided_batched_gemm"):
        return True
    if _int_profile_value(payload.get("num_batched_gemm")):
        return True
    primitives = payload.get("execution_primitives") or ()
    return any(str(item) in ("batched_matmul", "batched_rhs_expression") for item in primitives)


def _standard_rhs_profile(payload):
    rhs_count = _infer_rhs_count(payload)
    rhs_loop_calls = _int_profile_value(payload.get("num_rhs_loop_calls"))
    lowering = str(payload.get("lowering") or "")
    fallback_from = payload.get("fallback_from")
    fallback_to = payload.get("fallback_to")
    fallback_reason = payload.get("fallback_reason")
    batching_candidate = rhs_count > 1

    if rhs_loop_calls and (fallback_from == "batched_rhs_hop" or lowering == "fallback_rhs_loop"):
        batching_status = "fallback_to_loop"
    elif rhs_loop_calls:
        batching_status = "loop"
    elif batching_candidate and _rhs_batched_execution_used(payload, lowering):
        batching_status = "batched"
    elif batching_candidate:
        batching_status = "candidate_not_used"
    else:
        batching_status = "single_rhs"

    if rhs_loop_calls and batching_candidate:
        recommended_action = "batch_rhs_hop"
    elif batching_candidate and batching_status == "candidate_not_used":
        recommended_action = "inspect_rhs_batching"
    else:
        recommended_action = None

    return {
        "rhs_count": rhs_count,
        "rhs_loop_calls": rhs_loop_calls,
        "rhs_loop_required": bool(rhs_loop_calls),
        "batching_candidate": batching_candidate,
        "batching_status": batching_status,
        "fallback_from": fallback_from,
        "fallback_to": fallback_to,
        "fallback_reason": fallback_reason,
        "recommended_action": recommended_action,
    }


def _lowering_family(lowering):
    lowering = str(lowering or "")
    if lowering.startswith("fallback"):
        return "fallback"
    if lowering == "gemm":
        return "gemm"
    if lowering in ("batched_gemm", "strided_batched_gemm"):
        return "batched_gemm"
    if lowering in ("grouped_gemm", "block_grouped_gemm"):
        return "grouped_gemm"
    if lowering == "distributed":
        return "distributed"
    if lowering in ("tensordot", "einsum"):
        return lowering
    if lowering:
        return "custom"
    return "unknown"


def _lowering_family_from_payload(payload):
    lowering = payload.get("lowering")
    family = _lowering_family(lowering)
    if family != "custom":
        return family
    if _int_profile_value(payload.get("num_grouped_tasks")):
        return "grouped_gemm"
    if _int_profile_value(payload.get("num_batched_gemm")):
        return "batched_gemm"
    if _int_profile_value(payload.get("num_gemm")):
        return "gemm"
    return family


def _standard_lowering_profile(payload):
    lowering = payload.get("lowering")
    fallback_from = payload.get("fallback_from")
    fallback_to = payload.get("fallback_to")
    fallback_policy = payload.get("fallback_policy")
    fallback_reason = payload.get("fallback_reason")
    fallback_used = bool(
        str(lowering or "").startswith("fallback")
        or fallback_from is not None
        or fallback_to is not None
        or fallback_reason is not None
    )
    num_rhs_loop_calls = _int_profile_value(payload.get("num_rhs_loop_calls"))
    if fallback_used and fallback_from is not None and fallback_to is not None:
        lowering_route = [fallback_from, fallback_to]
    elif fallback_used:
        lowering_route = [value for value in (fallback_from, fallback_to) if value is not None]
    else:
        lowering_route = [lowering] if lowering is not None else []
    if num_rhs_loop_calls and (fallback_to == "rhs_loop" or lowering == "fallback_rhs_loop"):
        recommended_action = "batch_rhs_hop"
    elif fallback_used:
        recommended_action = "remove_backend_fallback"
    else:
        recommended_action = None
    return {
        "lowering": lowering,
        "lowering_family": _lowering_family_from_payload(payload),
        "fallback_used": fallback_used,
        "fallback_from": fallback_from,
        "fallback_to": fallback_to,
        "fallback_policy": fallback_policy,
        "fallback_reason": fallback_reason,
        "silent_fallback": bool(fallback_used and (not fallback_reason or fallback_policy == "silent")),
        "lowering_route": lowering_route,
        "kernel_counts": {
            "num_gemm": _int_profile_value(payload.get("num_gemm")),
            "num_batched_gemm": _int_profile_value(payload.get("num_batched_gemm")),
            "num_grouped_tasks": _int_profile_value(payload.get("num_grouped_tasks")),
            "num_blocks": _int_profile_value(payload.get("num_blocks")),
            "num_shape_buckets": _int_profile_value(payload.get("num_shape_buckets")),
            "num_rhs_loop_calls": num_rhs_loop_calls,
        },
        "recommended_action": recommended_action,
    }


def _should_add_rhs_profile(payload):
    return bool(
        payload.get("rhs_profile") is None
        and (
            payload.get("num_rhs") is not None
            or payload.get("num_rhs_loop_calls") is not None
            or payload.get("lowering") == "fallback_rhs_loop"
        )
    )


def _standard_contraction_execute_payload(payload):
    if payload.get("event") != "contraction_execute":
        return payload
    normalized = dict(payload)
    compute_class_source = normalized.pop("_compute_class_source", None)
    normalized.pop("_compute_payload_lowering", None)

    def set_default(name, default=None, aliases=()):
        if normalized.get(name) is not None:
            return
        alias_value = _first_present(normalized, aliases)
        normalized[name] = default if alias_value is None else alias_value

    for field in (
        "equation",
        "backend",
        "lowering",
        "input_shapes",
        "output_shape",
        "device",
        "fallback_from",
        "fallback_to",
        "fallback_policy",
        "fallback_reason",
    ):
        set_default(field, None)
    if normalized.get("fallback_to") is None:
        normalized["fallback_to"] = _infer_contraction_execute_fallback_target(normalized)
    set_default("dtype", None, aliases=("output_dtype",))
    expected_compute_class = _contraction_execute_compute_class(normalized.get("lowering"))
    if (
        normalized.get("compute_subclass") == "backend_execute"
        and (
            normalized.get("compute_class") is None
            or (
                normalized.get("compute_class") == COMPUTE_CLASS_TENSORDOT
                and compute_class_source == "contraction_execute_compute_payload"
                and _should_reclassify_legacy_contraction_execute(normalized.get("lowering"))
            )
        )
    ):
        normalized["compute_class"] = expected_compute_class
    if normalized.get("dtype_profile") is None:
        normalized["dtype_profile"] = _standard_dtype_profile(normalized)
    _sync_dtype_payload_fields(normalized)
    if normalized.get("device_profile") is None:
        normalized["device_profile"] = _standard_device_profile(normalized)
    set_default("flops", 0, aliases=("flops_estimate", "estimated_flops"))
    set_default("read_bytes", 0, aliases=("estimated_read_bytes",))
    set_default("write_bytes", 0, aliases=("estimated_write_bytes",))
    set_default("copy_bytes", 0, aliases=("estimated_copy_bytes",))
    if normalized.get("copy_profile") is None:
        normalized["copy_profile"] = _standard_copy_profile(normalized)
    if normalized.get("copy_movement_profile") is None:
        normalized["copy_movement_profile"] = _standard_copy_movement_profile(normalized)
    if normalized.get("layout_profile") is None:
        normalized["layout_profile"] = _standard_layout_profile(normalized)
    set_default("workspace_bytes", 0, aliases=("required_workspace_bytes", "estimated_workspace_bytes"))
    set_default("largest_intermediate", 0, aliases=("largest_intermediate_elements",))
    if normalized.get("workspace_profile") is None:
        normalized["workspace_profile"] = _standard_workspace_profile(normalized)
    if normalized.get("async_profile") is None:
        normalized["async_profile"] = _standard_async_profile(normalized)
    set_default("num_grouped_tasks", 0)
    set_default("num_blocks", 0)
    if normalized.get("num_shape_buckets") is None:
        shape_buckets = normalized.get("shape_buckets")
        normalized["num_shape_buckets"] = len(shape_buckets) if isinstance(shape_buckets, dict) else 0
    lowering = normalized.get("lowering")
    if normalized.get("num_gemm") is None:
        normalized["num_gemm"] = 1 if lowering in ("gemm", "grouped_gemm", "block_grouped_gemm") else 0
    if normalized.get("num_batched_gemm") is None:
        normalized["num_batched_gemm"] = 1 if lowering in ("batched_gemm", "strided_batched_gemm") else 0
    if _should_add_rhs_profile(normalized):
        normalized["rhs_profile"] = _standard_rhs_profile(normalized)
    if normalized.get("lowering_profile") is None:
        normalized["lowering_profile"] = _standard_lowering_profile(normalized)
    distributed_fields = {
        "distributed_modes",
        "rank",
        "world_size",
        "local_shape",
        "global_shape",
        "communication",
    }
    if lowering == "distributed" or any(field in normalized for field in distributed_fields):
        set_default("distributed_modes", [])
        set_default("rank", None)
        set_default("world_size", None)
        set_default("local_shape", None)
        set_default("global_shape", None, aliases=("output_shape",))
        set_default("communication", [])
        normalized["communication"] = _standard_communication_items(normalized.get("communication"))
    if normalized.get("communication_profile") is None:
        normalized["communication_profile"] = _standard_communication_profile(normalized)
    if normalized.get("cost_profile") is None:
        normalized["cost_profile"] = _standard_cost_profile(normalized)
    set_default("wall_s", 0)
    if normalized.get("practical_profile") is None:
        normalized["practical_profile"] = _contraction_execute_event_practical_profile(normalized)
    if normalized.get("compute_profile") is None:
        normalized["compute_profile"] = _event_compute_profile(normalized)
    return normalized


def _standard_contraction_plan_payload(payload):
    if payload.get("event") != "contraction_plan":
        return payload
    normalized = dict(payload)

    def set_default(name, default=0, aliases=()):
        if normalized.get(name) is not None:
            return
        alias_value = _first_present(normalized, aliases)
        normalized[name] = default if alias_value is None else alias_value

    set_default("dtype", None, aliases=("output_dtype",))
    if normalized.get("dtype_profile") is None:
        normalized["dtype_profile"] = _standard_dtype_profile(normalized)
    _sync_dtype_payload_fields(normalized)
    if normalized.get("device_profile") is None:
        normalized["device_profile"] = _standard_device_profile(normalized)
    set_default("flops", 0, aliases=("estimated_flops",))
    set_default("read_bytes", 0, aliases=("estimated_read_bytes",))
    set_default("write_bytes", 0, aliases=("estimated_write_bytes",))
    set_default("copy_bytes", 0, aliases=("estimated_copy_bytes",))
    if normalized.get("copy_profile") is None:
        normalized["copy_profile"] = _standard_copy_profile(normalized)
    if normalized.get("copy_movement_profile") is None:
        normalized["copy_movement_profile"] = _standard_copy_movement_profile(normalized)
    if normalized.get("layout_profile") is None:
        normalized["layout_profile"] = _standard_layout_profile(normalized)
    set_default("workspace_bytes", 0, aliases=("required_workspace_bytes", "estimated_workspace_bytes"))
    set_default("largest_intermediate", 0, aliases=("largest_intermediate_elements",))
    if normalized.get("workspace_profile") is None:
        normalized["workspace_profile"] = _standard_workspace_profile(normalized)
    set_default("num_gemm", 0)
    set_default("num_batched_gemm", 0)
    set_default("num_grouped_tasks", 0)
    set_default("num_blocks", 0)
    if normalized.get("num_shape_buckets") is None:
        shape_buckets = normalized.get("shape_buckets")
        normalized["num_shape_buckets"] = len(shape_buckets) if isinstance(shape_buckets, dict) else 0
    if normalized.get("lowering_profile") is None:
        normalized["lowering_profile"] = _standard_lowering_profile(normalized)
    if normalized.get("communication") is not None or normalized.get("lowering") == "distributed":
        set_default("communication", [])
        normalized["communication"] = _standard_communication_items(normalized.get("communication"))
    if normalized.get("communication_profile") is None:
        normalized["communication_profile"] = _standard_communication_profile(normalized)
    set_default("comm_bytes", None, aliases=("estimated_comm_bytes",))
    if normalized.get("comm_bytes") is None:
        normalized["comm_bytes"] = _communication_bytes_from_items(normalized.get("communication"))
    set_default("fallback_reason", None)
    if normalized.get("cost_profile") is None:
        normalized["cost_profile"] = _standard_cost_profile(normalized)
    if normalized.get("compute_profile") is None:
        normalized["compute_profile"] = _contraction_plan_compute_profile(normalized)
    return normalized


def _hmm_task_build_practical_profile(payload):
    read_bytes = int(payload.get("read_bytes") or 0)
    write_bytes = int(payload.get("write_bytes") or 0)
    copy_bytes = int(payload.get("copy_bytes") or 0)
    comm_bytes = int(payload.get("comm_bytes") or 0)
    working_set_bytes = read_bytes + write_bytes + copy_bytes + comm_bytes
    num_batches = int(payload.get("num_batches") or 0)
    batch_size = int(payload.get("batch_size") or 0)
    num_gemm_desc = int(payload.get("num_gemm_desc") or 0)
    num_gemv_desc = int(payload.get("num_gemv_desc") or 0)
    num_gemm = int(payload.get("num_gemm") or 0)
    num_batched = int(payload.get("num_batched_gemm") or 0)
    num_grouped = int(payload.get("num_grouped_tasks") or 0)
    num_blocks = int(payload.get("num_blocks") or payload.get("num_hx_blocks") or 0)
    num_shape_buckets = int(payload.get("num_shape_buckets") or 0)
    work = {
        "unit": "hmm_task_build",
        "lowering": payload.get("lowering"),
        "center_kind": payload.get("center_kind"),
        "num_batches": num_batches,
        "batch_size": batch_size,
        "num_gemm_desc": num_gemm_desc,
        "num_gemv_desc": num_gemv_desc,
        "num_gemm": num_gemm,
        "num_batched_gemm": num_batched,
        "num_grouped_tasks": num_grouped,
        "num_blocks": num_blocks,
        "num_shape_buckets": num_shape_buckets,
    }
    return {
        "workload_class": COMPUTE_CLASS_CONTRACTION_PLAN,
        "measurement_scope": "hmm_task_build",
        "precision_level": "hmm_build_lowering_counters",
        "primary_kernel": "hmm_task_build",
        "dominant_cost_kind": "planning_metadata",
        "cost_model": _practical_cost_model(
            "hmm_task_build_estimate",
            flops_estimate=payload.get("flops"),
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            copy_bytes=copy_bytes,
            communication_bytes=comm_bytes,
            working_set_bytes=working_set_bytes,
        ),
        "dominant_work": work,
        "parallelization": {
            "unit": "hmm_task_build",
            "backend_kernel": "hmm_task_build",
            "batched_kernel": bool(num_batched),
            "grouped_kernel": bool(num_grouped),
            "distributed": False,
            "num_batches": num_batches,
            "batch_size": batch_size,
            "num_gemm_desc": num_gemm_desc,
            "num_gemv_desc": num_gemv_desc,
            "num_gemm": num_gemm,
            "num_batched_gemm": num_batched,
            "num_grouped_tasks": num_grouped,
            "num_blocks": num_blocks,
            "num_shape_buckets": num_shape_buckets,
        },
        "measurement_limits": _measurement_limits("plan_estimate_only"),
        "optimization_targets": ["inspect_contraction_plan"],
        "diagnosis": {
            "measurement_focus": "hmm_task_build_metadata",
            "primary_issue": "planning_metadata",
            "parallelism_hint": (
                "shape_bucketed_grouped_gemm"
                if num_grouped
                else "batched_backend_kernel" if num_batched else "backend_kernel"
            ),
            "precision_evidence": {
                "flops": int(payload.get("flops") or 0),
                "working_set_bytes_estimate": int(working_set_bytes),
                "num_batches": num_batches,
                "batch_size": batch_size,
                "num_gemm_desc": num_gemm_desc,
                "num_gemv_desc": num_gemv_desc,
                "num_gemm": num_gemm,
                "num_batched_gemm": num_batched,
                "num_grouped_tasks": num_grouped,
                "num_blocks": num_blocks,
                "num_shape_buckets": num_shape_buckets,
            },
            "recommended_action": "inspect_contraction_plan",
        },
    }


def _standard_hmm_task_build_payload(payload):
    if payload.get("event") != "hmm_task_build":
        return payload
    normalized = dict(payload)

    def set_default(name, default=0, aliases=()):
        if normalized.get(name) is not None:
            return
        alias_value = _first_present(normalized, aliases)
        normalized[name] = default if alias_value is None else alias_value

    set_default("compute_class", COMPUTE_CLASS_CONTRACTION_PLAN)
    set_default("compute_subclass", "hmm_task_build")
    set_default("compute_role", COMPUTE_ROLE_PLANNING)
    set_default("compute_accounting", COMPUTE_ACCOUNTING_PRIMARY)
    set_default("backend", "hmm_task")
    set_default("device_kind", "planned")
    set_default("lowering", "hmm_task_build")
    set_default("flops", 0, aliases=("total_flops", "estimated_flops"))
    set_default("read_bytes", 0, aliases=("estimated_read_bytes",))
    set_default("write_bytes", 0, aliases=("estimated_write_bytes",))
    set_default("copy_bytes", 0, aliases=("estimated_copy_bytes",))
    set_default("comm_bytes", 0, aliases=("estimated_comm_bytes",))
    set_default("workspace_bytes", 0, aliases=("estimated_workspace_bytes", "required_workspace_bytes", "workspace_size"))
    set_default("largest_intermediate", 0, aliases=("largest_intermediate_elements", "inter_size"))
    set_default("num_blocks", 0, aliases=("num_hx_blocks",))
    set_default("num_gemm", 0)
    set_default("num_batched_gemm", 0)
    set_default("num_grouped_tasks", 0)
    if normalized.get("num_shape_buckets") is None:
        shape_buckets = normalized.get("shape_buckets")
        normalized["num_shape_buckets"] = len(shape_buckets) if isinstance(shape_buckets, (list, tuple, dict)) else 0
    set_default("fallback_reason", None)
    set_default("wall_s", 0)
    if normalized.get("dtype_profile") is None:
        normalized["dtype_profile"] = _standard_dtype_profile(normalized)
    if normalized.get("device_profile") is None:
        normalized["device_profile"] = _standard_device_profile(normalized)
    if normalized.get("copy_profile") is None:
        normalized["copy_profile"] = _standard_copy_profile(normalized)
    if normalized.get("copy_movement_profile") is None:
        normalized["copy_movement_profile"] = _standard_copy_movement_profile(normalized)
    if normalized.get("workspace_profile") is None:
        normalized["workspace_profile"] = _standard_workspace_profile(normalized)
    if normalized.get("communication_profile") is None:
        normalized["communication_profile"] = _standard_communication_profile(normalized)
    if normalized.get("cost_profile") is None:
        normalized["cost_profile"] = _standard_cost_profile(normalized)
    if normalized.get("practical_profile") is None:
        normalized["practical_profile"] = _hmm_task_build_practical_profile(normalized)
    return normalized


def _standard_grouped_gemm_execute_payload(payload):
    if payload.get("event") != "grouped_gemm_execute":
        return payload
    normalized = dict(payload)

    def set_default(name, default=0, aliases=()):
        if normalized.get(name) is not None:
            return
        alias_value = _first_present(normalized, aliases)
        normalized[name] = default if alias_value is None else alias_value

    set_default("compute_class", COMPUTE_CLASS_CONTRACTION_PLAN)
    set_default("compute_subclass", "grouped_gemm_execute")
    set_default("compute_role", COMPUTE_ROLE_KERNEL)
    set_default("compute_accounting", COMPUTE_ACCOUNTING_INCLUSIVE)
    set_default("lowering", "grouped_gemm")
    set_default("flops", 0, aliases=("total_flops", "flops_estimate", "estimated_flops"))
    set_default("read_bytes", 0, aliases=("estimated_read_bytes",))
    set_default("write_bytes", 0, aliases=("estimated_write_bytes",))
    set_default("copy_bytes", 0, aliases=("estimated_copy_bytes",))
    set_default("workspace_bytes", 0, aliases=("required_workspace_bytes", "estimated_workspace_bytes"))
    set_default("largest_intermediate", 0, aliases=("largest_intermediate_elements",))
    set_default("num_gemm", 0, aliases=("loop_task_count",))
    set_default("num_batched_gemm", 0, aliases=("batched_bucket_count",))
    set_default("num_grouped_tasks", 0, aliases=("num_tasks",))
    set_default("num_blocks", 0, aliases=("num_tasks",))
    if normalized.get("num_shape_buckets") is None:
        shape_buckets = normalized.get("shape_buckets")
        if isinstance(shape_buckets, (list, tuple, dict)):
            normalized["num_shape_buckets"] = len(shape_buckets)
        else:
            normalized["num_shape_buckets"] = _int_profile_value(normalized.get("num_groups"))
    set_default("wall_s", 0)
    set_default("fallback_reason", None)
    fallback_indicated = (
        normalized.get("fallback_reason") is not None
        or normalized.get("fallback_from") is not None
        or bool(normalized.get("requires_grouped_gemm_fallback"))
    )
    if normalized.get("fallback_to") is None:
        normalized["fallback_to"] = normalized.get("policy") if fallback_indicated else None
    if normalized.get("dtype_profile") is None:
        normalized["dtype_profile"] = _standard_dtype_profile(normalized)
    if normalized.get("device_profile") is None:
        normalized["device_profile"] = _standard_device_profile(normalized)
    if normalized.get("copy_profile") is None:
        normalized["copy_profile"] = _standard_copy_profile(normalized)
    if normalized.get("copy_movement_profile") is None:
        normalized["copy_movement_profile"] = _standard_copy_movement_profile(normalized)
    if normalized.get("layout_profile") is None:
        normalized["layout_profile"] = _standard_layout_profile(normalized)
    if normalized.get("workspace_profile") is None:
        normalized["workspace_profile"] = _standard_workspace_profile(normalized)
    if normalized.get("lowering_profile") is None:
        normalized["lowering_profile"] = _standard_lowering_profile(normalized)
    if normalized.get("communication_profile") is None:
        normalized["communication_profile"] = _standard_communication_profile(normalized)
    if normalized.get("cost_profile") is None:
        normalized["cost_profile"] = _standard_cost_profile(normalized)
    if normalized.get("practical_profile") is None:
        normalized["practical_profile"] = _contraction_execute_event_practical_profile(normalized)
    if normalized.get("compute_profile") is None:
        normalized["compute_profile"] = _event_compute_profile(normalized)
    return normalized


def _standard_matmul_execute_payload(payload):
    if payload.get("event") != "matmul_execute":
        return payload
    normalized = dict(payload)

    def set_default(name, default=0, aliases=()):
        if normalized.get(name) is not None:
            return
        alias_value = _first_present(normalized, aliases)
        normalized[name] = default if alias_value is None else alias_value

    set_default("compute_class", COMPUTE_CLASS_CONTRACTION_PLAN)
    set_default("compute_subclass", "matmul_execute")
    set_default("compute_role", COMPUTE_ROLE_KERNEL)
    set_default("compute_accounting", COMPUTE_ACCOUNTING_PRIMARY)
    set_default("lowering", "gemm")
    set_default("flops", 0, aliases=("flops_estimate", "estimated_flops"))
    set_default("read_bytes", 0, aliases=("estimated_read_bytes",))
    set_default("write_bytes", 0, aliases=("estimated_write_bytes",))
    set_default("copy_bytes", 0, aliases=("estimated_copy_bytes",))
    set_default("workspace_bytes", 0, aliases=("required_workspace_bytes", "estimated_workspace_bytes"))
    set_default("largest_intermediate", 0, aliases=("largest_intermediate_elements",))
    lowering = normalized.get("lowering")
    if normalized.get("num_gemm") is None:
        normalized["num_gemm"] = 1 if lowering == "gemm" else 0
    if normalized.get("num_batched_gemm") is None:
        normalized["num_batched_gemm"] = 1 if lowering in ("batched_gemm", "strided_batched_gemm") else 0
    set_default("num_grouped_tasks", 0)
    set_default("num_blocks", 0)
    set_default("num_shape_buckets", 0)
    set_default("wall_s", 0)
    set_default("fallback_reason", None)
    if normalized.get("dtype_profile") is None:
        normalized["dtype_profile"] = _standard_dtype_profile(normalized)
    if normalized.get("device_profile") is None:
        normalized["device_profile"] = _standard_device_profile(normalized)
    if normalized.get("copy_profile") is None:
        normalized["copy_profile"] = _standard_copy_profile(normalized)
    if normalized.get("copy_movement_profile") is None:
        normalized["copy_movement_profile"] = _standard_copy_movement_profile(normalized)
    if normalized.get("layout_profile") is None:
        normalized["layout_profile"] = _standard_layout_profile(normalized)
    if normalized.get("workspace_profile") is None:
        normalized["workspace_profile"] = _standard_workspace_profile(normalized)
    if normalized.get("async_profile") is None:
        normalized["async_profile"] = _standard_async_profile(normalized)
    if normalized.get("lowering_profile") is None:
        normalized["lowering_profile"] = _standard_lowering_profile(normalized)
    if normalized.get("communication_profile") is None:
        normalized["communication_profile"] = _standard_communication_profile(normalized)
    if normalized.get("cost_profile") is None:
        normalized["cost_profile"] = _standard_cost_profile(normalized)
    if normalized.get("practical_profile") is None:
        normalized["practical_profile"] = _contraction_execute_event_practical_profile(normalized)
    if normalized.get("compute_profile") is None:
        normalized["compute_profile"] = _event_compute_profile(normalized)
    return normalized


def _standard_grouped_gemm_prepack_payload(payload):
    if payload.get("event") != "grouped_gemm_prepack":
        return payload
    normalized = dict(payload)

    def set_default(name, default=0, aliases=()):
        if normalized.get(name) is not None:
            return
        alias_value = _first_present(normalized, aliases)
        normalized[name] = default if alias_value is None else alias_value

    set_default("compute_class", COMPUTE_CLASS_CONTRACTION_PLAN)
    set_default("compute_subclass", "backend_prepack")
    set_default("compute_role", COMPUTE_ROLE_PLANNING)
    set_default("compute_accounting", COMPUTE_ACCOUNTING_PRIMARY)
    set_default("lowering", "grouped_gemm")
    set_default("flops", 0, aliases=("total_flops", "flops_estimate", "estimated_flops"))
    set_default("read_bytes", 0, aliases=("estimated_read_bytes",))
    set_default("write_bytes", 0, aliases=("estimated_write_bytes",))
    set_default("copy_bytes", 0, aliases=("pack_bytes", "estimated_copy_bytes"))
    set_default("workspace_bytes", 0, aliases=("required_workspace_bytes", "estimated_workspace_bytes"))
    set_default("largest_intermediate", 0, aliases=("largest_intermediate_elements",))
    set_default("num_gemm", 0, aliases=("loop_task_count",))
    set_default("num_batched_gemm", 0, aliases=("batched_bucket_count",))
    set_default("num_grouped_tasks", 0, aliases=("num_tasks",))
    set_default("num_blocks", 0, aliases=("num_tasks",))
    if normalized.get("num_shape_buckets") is None:
        shape_buckets = normalized.get("shape_buckets")
        if isinstance(shape_buckets, (list, tuple, dict)):
            normalized["num_shape_buckets"] = len(shape_buckets)
        else:
            normalized["num_shape_buckets"] = _int_profile_value(normalized.get("num_groups"))
    set_default("wall_s", 0, aliases=("pack_s",))
    set_default("fallback_reason", None)
    fallback_indicated = (
        normalized.get("fallback_reason") is not None
        or normalized.get("fallback_from") is not None
        or bool(normalized.get("requires_grouped_gemm_fallback"))
    )
    if normalized.get("fallback_to") is None:
        normalized["fallback_to"] = normalized.get("policy") if fallback_indicated else None
    if normalized.get("dtype_profile") is None:
        normalized["dtype_profile"] = _standard_dtype_profile(normalized)
    if normalized.get("device_profile") is None:
        normalized["device_profile"] = _standard_device_profile(normalized)
    if normalized.get("copy_profile") is None:
        normalized["copy_profile"] = _standard_copy_profile(normalized)
    if normalized.get("copy_movement_profile") is None:
        normalized["copy_movement_profile"] = _standard_copy_movement_profile(normalized)
    if normalized.get("layout_profile") is None:
        normalized["layout_profile"] = _standard_layout_profile(normalized)
    if normalized.get("workspace_profile") is None:
        normalized["workspace_profile"] = _standard_workspace_profile(normalized)
    if normalized.get("async_profile") is None:
        normalized["async_profile"] = _standard_async_profile(normalized)
    if normalized.get("lowering_profile") is None:
        normalized["lowering_profile"] = _standard_lowering_profile(normalized)
    if normalized.get("communication_profile") is None:
        normalized["communication_profile"] = _standard_communication_profile(normalized)
    if normalized.get("cost_profile") is None:
        normalized["cost_profile"] = _standard_cost_profile(normalized)
    if normalized.get("practical_profile") is None:
        normalized["practical_profile"] = _contraction_execute_event_practical_profile(normalized)
    if normalized.get("compute_profile") is None:
        normalized["compute_profile"] = _event_compute_profile(normalized)
    return normalized


def enabled() -> bool:
    return logger.isEnabledFor(PROFILING)


def should_record_op() -> bool:
    return enabled()


def first_string(values):
    return values[0] if values and isinstance(values[0], str) else None


def array_shape(value):
    return tuple(value.shape) if hasattr(value, "shape") else None


def array_shapes(values):
    return [tuple(value.shape) for value in values if hasattr(value, "shape")]


def array_strides(value):
    strides = getattr(value, "strides", None)
    if strides is None:
        stride = getattr(value, "stride", None)
        if callable(stride):
            strides = stride()
    if strides is None:
        return None
    return tuple(int(dim) for dim in strides)


def array_contiguous(value):
    flags = getattr(value, "flags", None)
    for name in ("c_contiguous", "f_contiguous"):
        contiguous = getattr(flags, name, None)
        if contiguous is not None and bool(contiguous):
            return True
    if flags is not None:
        for name in ("C_CONTIGUOUS", "F_CONTIGUOUS"):
            try:
                if bool(flags[name]):
                    return True
            except (KeyError, TypeError, AttributeError):
                pass
        return False
    is_contiguous = getattr(value, "is_contiguous", None)
    if callable(is_contiguous):
        try:
            return bool(is_contiguous())
        except TypeError:
            return None
    return None


def array_order(value):
    flags = getattr(value, "flags", None)

    def flag_value(name):
        value = getattr(flags, name, None)
        if value is not None:
            return bool(value)
        if flags is None:
            return False
        try:
            return bool(flags[name])
        except (KeyError, TypeError, AttributeError):
            return False

    if flag_value("c_contiguous") or flag_value("C_CONTIGUOUS"):
        return "C"
    if flag_value("f_contiguous") or flag_value("F_CONTIGUOUS"):
        return "F"
    return "unknown"


def array_type_names(values):
    names = []
    for value in values:
        if not hasattr(value, "shape"):
            continue
        cls = value.__class__
        names.append(f"{cls.__module__}.{cls.__qualname__}")
    return names


def array_backend_name(value):
    module = value.__class__.__module__.split(".", 1)[0]
    if module == "jaxlib":
        return "jax"
    return module


def array_backend_names(values):
    return [array_backend_name(value) for value in values if hasattr(value, "shape")]


def array_device_kind(value):
    if not hasattr(value, "shape"):
        return None
    is_cuda = getattr(value, "is_cuda", None)
    if is_cuda is not None:
        return "cuda" if bool(is_cuda) else "cpu"
    device = getattr(value, "device", None)
    kind = _device_kind_from_value(device)
    if kind is not None:
        return "cuda" if kind == "gpu" else kind
    backend_name = array_backend_name(value)
    if backend_name == "numpy":
        return "cpu"
    if backend_name == "cupy":
        return "cuda"
    if backend_name in {"cupynumeric", "legate"}:
        return "distributed"
    return None


def array_device_kinds(values):
    return [array_device_kind(value) for value in values if hasattr(value, "shape")]


def array_is_distributed(value):
    return array_device_kind(value) == "distributed"


def array_location(value):
    kind = array_device_kind(value)
    if kind == "distributed":
        return "distributed"
    if kind in {"cuda", "rocm", "mps", "tpu"}:
        return "device"
    if kind in {"cpu", "host"}:
        return "host"
    return "unknown"


def array_locations(values):
    return [array_location(value) for value in values if hasattr(value, "shape")]


def array_is_host(value):
    return array_location(value) == "host"


def array_is_device(value):
    return array_location(value) == "device"


def array_dtype_names(values):
    return [str(getattr(value, "dtype", None)) for value in values if hasattr(value, "shape")]


def array_itemsize(value):
    itemsize = getattr(value, "itemsize", None)
    if itemsize is not None:
        return int(itemsize)
    dtype = getattr(value, "dtype", None)
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is not None:
        return int(itemsize)
    element_size = getattr(value, "element_size", None)
    if callable(element_size):
        return int(element_size())
    return 0


def _prod(values):
    result = 1
    for value in values:
        result *= int(value)
    return int(result)


def _kernel_kind_from_mnk(m, n, k):
    if not k:
        return None
    if m == 1 and n == 1:
        return "dot"
    if m == 1 or n == 1:
        return "gemv"
    return "gemm"


def _shape_key(shape):
    shape = tuple(int(dim) for dim in shape)
    if not shape:
        return "scalar"
    return "x".join(str(dim) for dim in shape)


def _shape_tuple(shape):
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None


def _shape_elements(shape):
    shape = _shape_tuple(shape)
    if shape is None:
        return 0
    return _prod(shape)


def _shape_key_or_unknown(shape):
    shape = _shape_tuple(shape)
    if shape is None:
        return "unknown"
    return _shape_key(shape)


def _problem_size_bin_from_scale(flops, memory_bytes):
    scale = max(float(flops or 0), float(memory_bytes or 0))
    if scale <= 0:
        return "unknown"
    if scale < 1_000_000:
        return "tiny"
    if scale < 100_000_000:
        return "small"
    if scale < 10_000_000_000:
        return "medium"
    if scale < 1_000_000_000_000:
        return "large"
    return "huge"


def _arithmetic_intensity_bin(arithmetic_intensity):
    if arithmetic_intensity is None:
        return "unknown"
    if arithmetic_intensity < 4.0:
        return "low"
    if arithmetic_intensity < 32.0:
        return "medium"
    return "high"


def _profile_tags(*groups):
    tags = []
    for group in groups:
        if group is None:
            continue
        if isinstance(group, str):
            values = (group,)
        else:
            values = group
        for value in values:
            if value is None:
                continue
            text = str(value)
            if text and text not in tags:
                tags.append(text)
    return tuple(tags)


def _bucket_dtype_key(input_dtypes, output_dtype):
    if input_dtypes is None:
        inputs = "unknown"
    elif isinstance(input_dtypes, str):
        inputs = input_dtypes
    else:
        inputs = ",".join(str(dtype) for dtype in input_dtypes) or "unknown"
    output = str(output_dtype) if output_dtype is not None else "unknown"
    return "{0}->{1}".format(inputs, output)


def _tensordot_practical_bucket(payload):
    kernel = payload.get("algorithmic_kernel") or payload.get("kernel_kind") or "unknown"
    dtype_key = _bucket_dtype_key(payload.get("input_dtypes"), payload.get("output_dtype"))
    m = int(payload.get("m") or 0)
    n = int(payload.get("n") or 0)
    k = int(payload.get("k") or 0)
    layout = payload.get("layout_hint") or "unknown"
    parallel_unit_by_kernel = {
        "gemm": "single_gemm",
        "gemv": "single_gemv",
        "dot": "single_dot",
        "outer": "single_outer_product",
    }
    aggregation_key = (
        "tensordot|kernel={0}|dtype={1}|m={2}|n={3}|k={4}|layout={5}".format(
            kernel,
            dtype_key,
            m,
            n,
            k,
            layout,
        )
    )
    return {
        "schema": PRACTICAL_BUCKET_SCHEMA,
        "family": "tensordot",
        "kernel": kernel,
        "precision_level": "mnk_layout_dtype",
        "measurement_unit": "single_call",
        "parallel_unit": parallel_unit_by_kernel.get(kernel, "single_backend_kernel"),
        "shape_signature": "m={0},n={1},k={2},layout={3}".format(m, n, k, layout),
        "dtype_key": dtype_key,
        "layout_key": layout,
        "batching_key": "kernel={0}|dtype={1}|m={2}|n={3}|k={4}".format(
            kernel,
            dtype_key,
            m,
            n,
            k,
        ),
        "comparison_key": payload.get("problem_signature"),
        "aggregation_key": aggregation_key,
        "comparison_axes": [
            "wall_s",
            "flops_estimate",
            "axis_permutation_copy_bytes",
            "effective_arithmetic_intensity_estimate",
        ],
    }


def _kernel_observation_cost(payload):
    cost = {
        "flops_estimate": _int_profile_value(payload.get("flops_estimate"), payload.get("flops")),
        "read_bytes": _int_profile_value(payload.get("read_bytes")),
        "write_bytes": _int_profile_value(payload.get("write_bytes")),
        "copy_bytes": _int_profile_value(payload.get("copy_bytes")),
    }
    workspace_bytes = _int_profile_value(payload.get("workspace_bytes"))
    if workspace_bytes:
        cost["workspace_bytes"] = workspace_bytes
    communication_bytes = _communication_bytes_from_items(payload.get("communication"))
    if communication_bytes:
        cost["communication_bytes"] = communication_bytes
    working_set_bytes = _int_profile_value(payload.get("working_set_bytes_estimate"))
    if not working_set_bytes:
        working_set_bytes = (
            cost["read_bytes"]
            + cost["write_bytes"]
            + cost["copy_bytes"]
            + cost.get("workspace_bytes", 0)
            + cost.get("communication_bytes", 0)
        )
    cost["working_set_bytes_estimate"] = int(working_set_bytes)
    return cost


def _kernel_observation_measurement(payload, cost_basis=None):
    return {
        "cost_basis": str(
            cost_basis
            or payload.get("cost_model")
            or payload.get("path_cost_model")
            or "profile_fields"
        ),
        "wall_time_basis": "caller_wall_s_if_recorded",
        "memory_basis": "array_nbytes_and_shape_estimate",
        "resource_timeseries": False,
        "external_telemetry_required": [
            "rss_pss_timeseries",
            "cpu_thread_context_switch_timeseries",
            "gpu_kernel_timeline",
            "host_device_transfer_timeline",
        ],
    }


def _kernel_observation_common(payload):
    bucket = payload.get("practical_bucket") or {}
    compute_class = payload.get("compute_class")
    return {
        "schema": KERNEL_OBSERVATION_SCHEMA,
        "family": bucket.get("family") or _CORE_COMPUTE_FAMILIES.get(compute_class, str(compute_class)),
        "accounting": payload.get("compute_accounting", COMPUTE_ACCOUNTING_PRIMARY),
        "kernel": payload.get("algorithmic_kernel") or payload.get("kernel_kind") or payload.get("lowering"),
        "aggregation_key": bucket.get("aggregation_key"),
        "comparison_key": bucket.get("comparison_key"),
        "cost": _kernel_observation_cost(payload),
        "measurement": _kernel_observation_measurement(payload),
    }


def _kernel_observation_data_motion(payload, *, copy_pressure=None, extra=None):
    cost = _kernel_observation_cost(payload)
    motion = {
        "working_set_bytes_estimate": int(cost.get("working_set_bytes_estimate") or 0),
        "copy_bytes": int(cost.get("copy_bytes") or 0),
        "copy_pressure": copy_pressure or ("copy" if int(cost.get("copy_bytes") or 0) else "none"),
        "peak_workspace_bytes_estimate": int(
            payload.get("peak_bytes")
            or payload.get("workspace_bytes")
            or cost.get("workspace_bytes")
            or 0
        ),
    }
    if extra:
        motion.update(extra)
    return motion


def _execution_unit(
    kind,
    count,
    *,
    role,
    source,
    flops_estimate=None,
    bytes_value=None,
    elements=None,
    serial_dependency=False,
    batchable=False,
):
    count = _int_profile_value(count)
    if count <= 0:
        return None
    unit = {
        "kind": str(kind),
        "role": str(role),
        "count": int(count),
        "source": str(source),
        "serial_dependency": bool(serial_dependency),
        "batchable": bool(batchable),
    }
    if flops_estimate is not None:
        unit["flops_estimate"] = _int_profile_value(flops_estimate)
    if bytes_value is not None:
        unit["bytes"] = _int_profile_value(bytes_value)
    if elements is not None:
        unit["elements"] = _int_profile_value(elements)
    return unit


def _append_execution_unit(units, *args, **kwargs):
    unit = _execution_unit(*args, **kwargs)
    if unit is not None:
        units.append(unit)


def _optimization_limit(kind, reason, *, evidence=None):
    limit = {
        "kind": str(kind),
        "reason": str(reason),
    }
    if evidence:
        limit["evidence"] = evidence
    return limit


def _tensordot_execution_units(payload, *, rhs_loop_calls):
    units = []
    kernel = payload.get("algorithmic_kernel") or payload.get("kernel_kind") or "tensordot"
    equivalent_gemm = bool(payload.get("equivalent_gemm") or payload.get("num_gemm"))
    if equivalent_gemm:
        primary_kind = "gemm_like"
        source = "mnk_shape_model"
    elif kernel == "outer":
        primary_kind = "outer_product"
        source = "outer_product_shape_model"
    else:
        primary_kind = str(kernel)
        source = "tensordot_shape_model"
    _append_execution_unit(
        units,
        primary_kind,
        rhs_loop_calls or 1,
        role="primary",
        source=source,
        flops_estimate=payload.get("flops_estimate"),
        serial_dependency=False,
        batchable=False,
    )
    if rhs_loop_calls:
        _append_execution_unit(
            units,
            "rhs_python_loop",
            rhs_loop_calls,
            role="overhead",
            source="num_rhs_loop_calls",
            serial_dependency=True,
            batchable=True,
        )
    copy_bytes = _int_profile_value(payload.get("copy_bytes"))
    if copy_bytes:
        _append_execution_unit(
            units,
            "layout_copy",
            1,
            role="overhead",
            source="axis_permutation_copy_bytes",
            bytes_value=copy_bytes,
            serial_dependency=False,
            batchable=False,
        )
    return units


def _tensordot_optimization_limits(payload, *, bucket, rhs_loop_calls, small_gemm_candidate):
    limits = []
    if rhs_loop_calls:
        limits.append(_optimization_limit(
            "rhs_loop_fallback",
            "multiple RHS vectors are executed through a Python loop unless lowered to a batched H*v path",
            evidence={"num_rhs_loop_calls": int(rhs_loop_calls)},
        ))
    elif small_gemm_candidate:
        limits.append(_optimization_limit(
            "cross_call_aggregation_required",
            "single event only contains one small GEMM-like contraction; batching needs repeated shape buckets",
            evidence={"bucket_key": bucket.get("batching_key")},
        ))
    copy_bytes = _int_profile_value(payload.get("copy_bytes"))
    if copy_bytes:
        limits.append(_optimization_limit(
            "layout_copy_estimate",
            "axis permutation copy bytes are shape-estimated; external telemetry is needed for copy wall time",
            evidence={"copy_bytes": int(copy_bytes), "layout_hint": payload.get("layout_hint")},
        ))
    return limits


def _tensordot_execution_model(payload, *, bucket, rhs_loop_calls, small_gemm_candidate):
    if rhs_loop_calls:
        batching = {
            "available_now": True,
            "scope": "within_call_rhs_batching",
            "grouping_basis": "rhs_axis",
            "bucket_key": bucket.get("batching_key"),
            "reason": "multiple_rhs_vectors_in_single_call",
        }
        batchable_work_units = int(rhs_loop_calls)
    elif small_gemm_candidate:
        batching = {
            "available_now": False,
            "scope": "cross_call_shape_aggregation_required",
            "grouping_basis": "m,n,k,dtype,layout",
            "bucket_key": bucket.get("batching_key"),
            "reason": "tiny_or_skinny_gemm_needs_repeated_shape_bucket",
        }
        batchable_work_units = 0
    else:
        batching = {
            "available_now": False,
            "scope": "none",
            "grouping_basis": "single_call",
            "bucket_key": bucket.get("batching_key"),
            "reason": "single_backend_kernel",
        }
        batchable_work_units = 0

    copy_pressure = "axis_permutation" if int(payload.get("copy_bytes") or 0) else "none"
    return {
        "schema": EXECUTION_MODEL_SCHEMA,
        "model": "single_backend_contraction",
        "primary_unit": bucket.get("parallel_unit") or "single_backend_kernel",
        "serial_dependency_units": 1,
        "independent_work_units": int(rhs_loop_calls or 1),
        "backend_kernel_units": int(rhs_loop_calls or 1),
        "batchable_work_units": int(batchable_work_units),
        "batching": batching,
        "data_motion": _kernel_observation_data_motion(payload, copy_pressure=copy_pressure),
    }


def _oe_execution_units(payload, *, gemm_steps, tensordot_steps, generic_einsum_steps, non_gemm_steps):
    units = []
    _append_execution_unit(
        units,
        "oe_gemm_step",
        gemm_steps,
        role="primary",
        source="path_step_profile",
        flops_estimate=payload.get("gemm_flops_estimate"),
        serial_dependency=True,
        batchable=False,
    )
    _append_execution_unit(
        units,
        "oe_tensordot_step",
        tensordot_steps,
        role="primary",
        source="path_step_profile",
        flops_estimate=payload.get("tensordot_flops_estimate"),
        serial_dependency=True,
        batchable=False,
    )
    _append_execution_unit(
        units,
        "oe_generic_einsum_step",
        generic_einsum_steps,
        role="primary",
        source="path_step_profile",
        flops_estimate=payload.get("generic_einsum_flops_estimate"),
        serial_dependency=True,
        batchable=False,
    )
    residual_non_gemm_steps = max(
        _int_profile_value(non_gemm_steps) - _int_profile_value(tensordot_steps) - _int_profile_value(generic_einsum_steps),
        0,
    )
    residual_non_gemm_flops = max(
        _int_profile_value(payload.get("non_gemm_flops_estimate"))
        - _int_profile_value(payload.get("tensordot_flops_estimate"))
        - _int_profile_value(payload.get("generic_einsum_flops_estimate")),
        0,
    )
    _append_execution_unit(
        units,
        "oe_other_non_gemm_step",
        residual_non_gemm_steps,
        role="primary",
        source="non_gemm_flops_estimate_minus_known_subcomponents",
        flops_estimate=residual_non_gemm_flops,
        serial_dependency=True,
        batchable=False,
    )
    largest_intermediate = _int_profile_value(
        payload.get("largest_intermediate_elements"),
        payload.get("largest_intermediate"),
    )
    if largest_intermediate:
        _append_execution_unit(
            units,
            "oe_intermediate",
            1,
            role="overhead",
            source="largest_intermediate_elements",
            elements=largest_intermediate,
            bytes_value=payload.get("largest_intermediate_bytes"),
            serial_dependency=False,
            batchable=False,
        )
    return units


def _oe_optimization_limits(
    payload,
    *,
    contraction_count,
    generic_einsum_steps,
    non_gemm_steps,
    batchable_groups,
):
    limits = []
    if contraction_count > 1:
        limits.append(_optimization_limit(
            "serial_path_dependency",
            "opt_einsum path steps are sequential because each step consumes intermediates from previous steps",
            evidence={"contraction_count": int(contraction_count)},
        ))
    if batchable_groups:
        limits.append(_optimization_limit(
            "same_step_bucket_detected",
            "repeated path-step signatures exist, but batching requires a lowering layer that materializes grouped GEMM tasks",
            evidence={"batchable_groups": int(batchable_groups)},
        ))
    if generic_einsum_steps:
        limits.append(_optimization_limit(
            "generic_einsum_not_gemm_lowered",
            "generic einsum path steps are not represented as explicit GEMM descriptors yet",
            evidence={"generic_einsum_step_count": int(generic_einsum_steps)},
        ))
    elif non_gemm_steps:
        limits.append(_optimization_limit(
            "non_gemm_path_not_gemm_lowered",
            "non-GEMM path steps need contraction IR lowering before backend GEMM batching can target them",
            evidence={"non_gemm_step_count": int(non_gemm_steps)},
        ))
    largest_ratio = payload.get("largest_intermediate_to_output_ratio")
    if largest_ratio is not None and largest_ratio >= 8:
        limits.append(_optimization_limit(
            "large_intermediate_pressure",
            "largest intermediate is much larger than the output and can dominate memory traffic",
            evidence={
                "largest_intermediate_elements": _int_profile_value(
                    payload.get("largest_intermediate_elements"),
                    payload.get("largest_intermediate"),
                ),
                "largest_intermediate_to_output_ratio": largest_ratio,
            },
        ))
    return limits


def _oe_execution_model(
    payload,
    *,
    contraction_count,
    gemm_steps,
    tensordot_steps,
    generic_einsum_steps,
    non_gemm_steps,
    batchable_groups,
):
    largest_ratio = payload.get("largest_intermediate_to_output_ratio")
    if largest_ratio is not None and largest_ratio >= 8:
        copy_pressure = "large_intermediate"
    elif int(payload.get("workspace_bytes") or 0):
        copy_pressure = "intermediate_workspace"
    else:
        copy_pressure = "none"
    return {
        "schema": EXECUTION_MODEL_SCHEMA,
        "model": "serial_oe_contraction_path",
        "primary_unit": "path_step",
        "serial_dependency_units": int(contraction_count),
        "independent_work_units": 0,
        "backend_kernel_units": int(gemm_steps + tensordot_steps + generic_einsum_steps),
        "batchable_work_units": int(batchable_groups),
        "kernel_units": {
            "gemm_steps": int(gemm_steps),
            "tensordot_steps": int(tensordot_steps),
            "generic_einsum_steps": int(generic_einsum_steps),
            "non_gemm_steps": int(non_gemm_steps),
        },
        "batching": {
            "available_now": bool(batchable_groups),
            "scope": "repeated_path_step_bucket" if batchable_groups else "none",
            "grouping_basis": "path_step_bucket_signature",
            "batchable_groups": int(batchable_groups),
        },
        "data_motion": _kernel_observation_data_motion(
            payload,
            copy_pressure=copy_pressure,
            extra={
                "largest_intermediate_elements": _int_profile_value(
                    payload.get("largest_intermediate_elements"),
                    payload.get("largest_intermediate"),
                ),
                "largest_intermediate_to_output_ratio": largest_ratio,
            },
        ),
    }


def _svd_execution_units(payload, *, block_count):
    units = []
    _append_execution_unit(
        units,
        "svd_qn_block",
        block_count,
        role="primary",
        source="block_count",
        flops_estimate=payload.get("block_flops_estimate"),
        serial_dependency=False,
        batchable=False,
    )
    _append_execution_unit(
        units,
        "batchable_qn_block",
        payload.get("batchable_block_count"),
        role="subset",
        source="batchable_block_count",
        flops_estimate=payload.get("batchable_flops_estimate"),
        serial_dependency=False,
        batchable=True,
    )
    _append_execution_unit(
        units,
        "tiny_qn_block",
        payload.get("tiny_block_count"),
        role="subset",
        source="tiny_block_count",
        serial_dependency=False,
        batchable=False,
    )
    _append_execution_unit(
        units,
        "skinny_qn_block",
        payload.get("skinny_block_count"),
        role="subset",
        source="skinny_block_count",
        serial_dependency=False,
        batchable=False,
    )
    return units


def _svd_optimization_limits(payload, *, unique_block_shape_count, batchable_group_count):
    limits = []
    if batchable_group_count:
        limits.append(_optimization_limit(
            "batching_requires_same_shape_driver",
            "same-shape QN blocks are visible, but a batched decomposition driver is needed to execute them together",
            evidence={"batchable_block_group_count": int(batchable_group_count)},
        ))
    tiny_blocks = _int_profile_value(payload.get("tiny_block_count"))
    if tiny_blocks:
        limits.append(_optimization_limit(
            "tiny_blocks_may_be_overhead_bound",
            "many QN decomposition blocks are small enough that dispatch overhead can dominate compute",
            evidence={"tiny_block_count": int(tiny_blocks)},
        ))
    if unique_block_shape_count > 1:
        limits.append(_optimization_limit(
            "shape_fragmentation",
            "multiple QN block shapes reduce same-shape batching reuse",
            evidence={"unique_block_shape_count": int(unique_block_shape_count)},
        ))
    return limits


def _svd_execution_model(payload, *, block_count, unique_block_shape_count, batchable_group_count):
    batchable_block_count = int(payload.get("batchable_block_count") or 0)
    tiny_block_count = int(payload.get("tiny_block_count") or 0)
    skinny_block_count = int(payload.get("skinny_block_count") or 0)
    return {
        "schema": EXECUTION_MODEL_SCHEMA,
        "model": "independent_qn_block_decomposition",
        "primary_unit": "qn_block",
        "serial_dependency_units": 0,
        "independent_work_units": int(block_count),
        "backend_kernel_units": int(block_count),
        "batchable_work_units": batchable_block_count,
        "kernel_units": {
            "block_count": int(block_count),
            "shape_groups": int(unique_block_shape_count),
            "tiny_blocks": tiny_block_count,
            "skinny_blocks": skinny_block_count,
        },
        "batching": {
            "available_now": bool(batchable_group_count),
            "scope": "same_shape_qn_blocks" if batchable_group_count else "none",
            "grouping_basis": "block_shape",
            "batchable_groups": int(batchable_group_count),
            "batchable_flop_fraction": payload.get("batchable_flop_fraction"),
        },
        "data_motion": _kernel_observation_data_motion(
            payload,
            copy_pressure="qn_sparse_block_density" if payload.get("qn_block_density") is not None else "none",
            extra={
                "matrix_elements": int(payload.get("matrix_elements") or 0),
                "total_block_elements": int(payload.get("total_block_elements") or 0),
                "qn_block_density": payload.get("qn_block_density"),
            },
        ),
    }


def _shape_key_list(shapes):
    return [_shape_key_or_unknown(shape) for shape in shapes or ()]


def _tensordot_kernel_observation(payload, input_shapes=None, output_shape=None):
    bucket = payload.get("practical_bucket") or {}
    tags = set(payload.get("profile_tags") or ())
    rhs_loop_calls = int(payload.get("num_rhs_loop_calls") or 0)
    small_gemm_candidate = bool({"tiny_gemm", "skinny_gemm"} & tags)
    if input_shapes is None:
        input_shapes = payload.get("input_shapes")
    if output_shape is None:
        output_shape = payload.get("output_shape")
    if rhs_loop_calls:
        parallel_candidate = "rhs_batching"
    elif small_gemm_candidate:
        parallel_candidate = "aggregate_repeated_small_gemm"
    else:
        parallel_candidate = "backend_kernel"
    observation = _kernel_observation_common(payload)
    observation.update({
        "scope": "single_call",
        "shape": {
            "m": int(payload.get("m") or 0),
            "n": int(payload.get("n") or 0),
            "k": int(payload.get("k") or 0),
            "layout": payload.get("layout_hint"),
            "input_shapes": _shape_key_list(input_shapes),
            "output_shape": _shape_key_or_unknown(output_shape),
            "dtype": bucket.get("dtype_key"),
        },
        "parallelism": {
            "unit": bucket.get("parallel_unit") or "single_backend_kernel",
            "candidate": parallel_candidate,
            "batchable": bool(rhs_loop_calls),
            "requires_aggregation": bool(small_gemm_candidate and not rhs_loop_calls),
            "independent_units": int(rhs_loop_calls or 1),
        },
        "execution_units": _tensordot_execution_units(
            payload,
            rhs_loop_calls=rhs_loop_calls,
        ),
        "optimization_limits": _tensordot_optimization_limits(
            payload,
            bucket=bucket,
            rhs_loop_calls=rhs_loop_calls,
            small_gemm_candidate=small_gemm_candidate,
        ),
        "execution_model": _tensordot_execution_model(
            payload,
            bucket=bucket,
            rhs_loop_calls=rhs_loop_calls,
            small_gemm_candidate=small_gemm_candidate,
        ),
    })
    return observation


def _oe_kernel_observation(payload):
    bucket = payload.get("practical_bucket") or {}
    contraction_count = int(payload.get("contraction_count") or bucket.get("contraction_count") or 0)
    gemm_steps = int(payload.get("gemm_step_count") or 0)
    tensordot_steps = int(payload.get("tensordot_step_count") or 0)
    generic_einsum_steps = int(payload.get("generic_einsum_step_count") or 0)
    non_gemm_steps = int(payload.get("non_gemm_step_count") or 0)
    step_costs_available = bool(payload.get("step_costs_available"))
    cost_basis = "oe_path_step_estimate" if step_costs_available else "oe_path_type_count_estimate"
    largest_ratio = payload.get("largest_intermediate_to_output_ratio")
    batchable_groups = sum(
        1
        for step_bucket in payload.get("step_buckets") or ()
        if isinstance(step_bucket, dict) and int(step_bucket.get("count") or 0) >= 2
    )
    if non_gemm_steps:
        primary_issue = "non_gemm_path"
        recommended_action = "reduce_non_gemm_path_cost"
    elif largest_ratio is not None and largest_ratio >= 8:
        primary_issue = "large_intermediate"
        recommended_action = "limit_largest_intermediate"
    else:
        primary_issue = "path_overhead"
        recommended_action = "inspect_oe_contract_path"
    observation = _kernel_observation_common(payload)
    observation["measurement"] = _kernel_observation_measurement(payload, cost_basis=cost_basis)
    observation.update({
        "scope": "path_call",
        "shape": {
            "path_type_signature": bucket.get("path_type_signature"),
            "contraction_count": contraction_count,
            "dominant_step_signature": bucket.get("dominant_step_signature"),
            "largest_intermediate_elements": _int_profile_value(
                payload.get("largest_intermediate_elements"),
                payload.get("largest_intermediate"),
            ),
            "unique_step_output_shape_count": int(payload.get("unique_step_output_shape_count") or 0),
        },
        "parallelism": {
            "unit": bucket.get("parallel_unit") or "sequential_path_step",
            "candidate": "path_kernel_mix",
            "batchable": bool(batchable_groups),
            "independent_units": 0,
            "serial_units": contraction_count,
            "backend_kernel_units": gemm_steps + tensordot_steps + generic_einsum_steps,
            "batchable_groups": int(batchable_groups),
        },
        "execution_units": _oe_execution_units(
            payload,
            gemm_steps=gemm_steps,
            tensordot_steps=tensordot_steps,
            generic_einsum_steps=generic_einsum_steps,
            non_gemm_steps=non_gemm_steps,
        ),
        "optimization_limits": _oe_optimization_limits(
            payload,
            contraction_count=contraction_count,
            generic_einsum_steps=generic_einsum_steps,
            non_gemm_steps=non_gemm_steps,
            batchable_groups=batchable_groups,
        ),
        "execution_model": _oe_execution_model(
            payload,
            contraction_count=contraction_count,
            gemm_steps=gemm_steps,
            tensordot_steps=tensordot_steps,
            generic_einsum_steps=generic_einsum_steps,
            non_gemm_steps=non_gemm_steps,
            batchable_groups=batchable_groups,
        ),
        "diagnosis": {
            "primary_issue": primary_issue,
            "recommended_action": recommended_action,
            "evidence": {
                "gemm_step_count": gemm_steps,
                "tensordot_step_count": tensordot_steps,
                "generic_einsum_step_count": generic_einsum_steps,
                "non_gemm_step_count": non_gemm_steps,
                "largest_intermediate_to_output_ratio": largest_ratio,
            },
        },
    })
    return observation


def _svd_kernel_observation(payload):
    bucket = payload.get("practical_bucket") or {}
    block_count = int(payload.get("block_count") or 0)
    unique_block_shape_count = int(payload.get("unique_block_shape_count") or 0)
    batchable_group_count = int(payload.get("batchable_block_group_count") or 0)
    if int(payload.get("tiny_block_count") or 0):
        primary_issue = "tiny_qn_blocks"
    elif unique_block_shape_count > 1:
        primary_issue = "fragmented_qn_blocks"
    elif batchable_group_count:
        primary_issue = "batchable_qn_blocks"
    else:
        primary_issue = "blocked_decomposition"
    if int(payload.get("batchable_flops_estimate") or 0):
        recommended_action = "batch_reused_qn_blocks"
    elif int(payload.get("tiny_block_count") or 0):
        recommended_action = "reduce_tiny_block_overhead"
    elif payload.get("qn_sparse_fraction"):
        recommended_action = "preserve_qn_sparsity"
    elif unique_block_shape_count > 1:
        recommended_action = "manage_block_shape_fragmentation"
    else:
        recommended_action = "inspect_qn_decomposition"
    observation = _kernel_observation_common(payload)
    observation.update({
        "scope": "decomposition_call",
        "shape": {
            "matrix_shape": bucket.get("matrix_shape"),
            "block_group_signature": bucket.get("block_group_signature"),
            "block_count": block_count,
            "unique_block_shape_count": unique_block_shape_count,
            "dominant_block_shape": payload.get("dominant_block_shape"),
        },
        "parallelism": {
            "unit": bucket.get("parallel_unit") or "qn_block_shape_group",
            "candidate": "block_shape_batching" if batchable_group_count else "independent_qn_blocks",
            "batchable": bool(batchable_group_count),
            "independent_units": block_count,
            "shape_groups": unique_block_shape_count,
            "batchable_groups": batchable_group_count,
        },
        "execution_units": _svd_execution_units(
            payload,
            block_count=block_count,
        ),
        "optimization_limits": _svd_optimization_limits(
            payload,
            unique_block_shape_count=unique_block_shape_count,
            batchable_group_count=batchable_group_count,
        ),
        "execution_model": _svd_execution_model(
            payload,
            block_count=block_count,
            unique_block_shape_count=unique_block_shape_count,
            batchable_group_count=batchable_group_count,
        ),
        "diagnosis": {
            "primary_issue": primary_issue,
            "recommended_action": recommended_action,
            "evidence": {
                "qn_block_density": payload.get("qn_block_density"),
                "batchable_block_fraction": payload.get("batchable_block_fraction"),
                "tiny_block_count": int(payload.get("tiny_block_count") or 0),
                "block_shape_fragmentation": payload.get("block_shape_fragmentation"),
            },
        },
    })
    return observation


def _kernel_observation(payload):
    compute_class = payload.get("compute_class")
    if compute_class == COMPUTE_CLASS_TENSORDOT:
        return _tensordot_kernel_observation(payload)
    if compute_class == COMPUTE_CLASS_OE:
        return _oe_kernel_observation(payload)
    if compute_class == COMPUTE_CLASS_SVD:
        return _svd_kernel_observation(payload)
    return None


def _safe_fraction(numerator, denominator):
    try:
        numerator = float(numerator or 0)
        denominator = float(denominator or 0)
    except (TypeError, ValueError):
        return None
    return numerator / denominator if denominator > 0 else None


def _append_unique(items, value):
    if value and value not in items:
        items.append(value)


def _first_target(targets, default):
    return targets[0] if targets else default


def _practical_cost_model(
    source,
    *,
    flops_estimate=0,
    read_bytes=0,
    write_bytes=0,
    copy_bytes=0,
    communication_bytes=0,
    working_set_bytes=None,
    extra=None,
):
    memory_bytes = int(read_bytes or 0) + int(write_bytes or 0)
    copy_bytes = int(copy_bytes or 0)
    communication_bytes = int(communication_bytes or 0)
    if working_set_bytes is None:
        working_set_bytes = memory_bytes + copy_bytes + communication_bytes
    payload = {
        "source": str(source),
        "flops_estimate": int(flops_estimate or 0),
        "read_bytes": int(read_bytes or 0),
        "write_bytes": int(write_bytes or 0),
        "copy_bytes": copy_bytes,
        "communication_bytes": communication_bytes,
        "working_set_bytes_estimate": int(working_set_bytes or 0),
        "arithmetic_intensity_flops_per_byte": _safe_fraction(flops_estimate, memory_bytes),
        "effective_arithmetic_intensity_flops_per_byte": _safe_fraction(
            flops_estimate, working_set_bytes
        ),
    }
    if extra:
        payload.update(extra)
    return payload


def _measurement_limits(runtime_scope):
    return {
        "runtime_scope": runtime_scope,
        "memory_scope": "array_nbytes_estimate",
        "cost_scope": "shape_derived_flop_estimate",
        "not_measured": [
            "rss_pss_time_series",
            "cpu_utilization_time_series",
            "context_switches",
            "gpu_kernel_timeline",
            "h2d_d2h_transfer_bytes",
            "per_backend_kernel_time",
        ],
    }


def _compact_estimated_cost(cost_model):
    cost_model = cost_model or {}
    return {
        "source": cost_model.get("source"),
        "flops": int(cost_model.get("flops_estimate") or 0),
        "read_bytes": int(cost_model.get("read_bytes") or 0),
        "write_bytes": int(cost_model.get("write_bytes") or 0),
        "copy_bytes": int(cost_model.get("copy_bytes") or 0),
        "communication_bytes": int(cost_model.get("communication_bytes") or 0),
        "working_set_bytes": int(cost_model.get("working_set_bytes_estimate") or 0),
        "arithmetic_intensity_flops_per_byte": cost_model.get("arithmetic_intensity_flops_per_byte"),
        "effective_arithmetic_intensity_flops_per_byte": cost_model.get(
            "effective_arithmetic_intensity_flops_per_byte"
        ),
    }


def _diagnosis_value(practical_profile, key, default=None):
    diagnosis = practical_profile.get("diagnosis")
    if isinstance(diagnosis, dict) and diagnosis.get(key) is not None:
        return diagnosis.get(key)
    return default


def _optimization_bottleneck(practical_profile):
    return (
        _diagnosis_value(practical_profile, "primary_issue")
        or practical_profile.get("primary_issue")
        or practical_profile.get("dominant_cost_kind")
        or practical_profile.get("dominant_bottleneck")
    )


def _optimization_recommended_action(practical_profile):
    return _diagnosis_value(
        practical_profile,
        "recommended_action",
        _first_target(practical_profile.get("optimization_targets") or (), None),
    )


def _optimization_actionability(practical_profile, estimated_cost, bottleneck, recommended_action):
    measurement_limits = practical_profile.get("measurement_limits") or {}
    return {
        "priority_score": int(_BOTTLENECK_PRIORITY.get(str(bottleneck), 0)),
        "dominant_resource": _BOTTLENECK_RESOURCE.get(str(bottleneck), "unknown"),
        "optimization_scope": _OPTIMIZATION_SCOPE_BY_ACTION.get(str(recommended_action), "inspection"),
        "evidence": {
            "cost_source": estimated_cost.get("source"),
            "precision_level": practical_profile.get("precision_level"),
            "runtime_scope": measurement_limits.get("runtime_scope"),
        },
    }


def _practical_next_measurements(compute_class, practical_profile):
    diagnosis = practical_profile.get("diagnosis") or {}
    issue = diagnosis.get("primary_issue") or practical_profile.get("primary_issue")
    if compute_class == COMPUTE_CLASS_TENSORDOT:
        if issue == "backend_fallback":
            return [
                "fallback_count_by_lowering",
                "backend_kernel_coverage",
                "wall_time_by_lowering",
            ]
        if issue == "python_rhs_loop":
            return [
                "rhs_loop_wall_time",
                "num_rhs_loop_calls",
                "batched_rhs_kernel_coverage",
            ]
        if issue == "communication":
            return [
                "communication_wall_time",
                "communication_bytes_by_collective",
                "backend_kernel_time",
            ]
        if issue in ("axis_permutation_copy", "copy_overhead"):
            return [
                "wall_time_by_shape",
                "axis_permutation_copy_bytes" if issue == "axis_permutation_copy" else "copy_bytes",
                "backend_kernel_time",
            ]
        if issue == "small_gemm":
            return [
                "wall_time_by_shape",
                "small_gemm_call_count",
                "batched_gemm_candidate_count",
            ]
        return [
            "wall_time_by_shape",
            "backend_kernel_time",
            "arithmetic_intensity",
        ]
    if compute_class == COMPUTE_CLASS_OE:
        if issue == "large_intermediate":
            return [
                "path_step_wall_time",
                "largest_intermediate_bytes",
                "workspace_bytes",
            ]
        return [
            "path_step_wall_time",
            "path_step_flops",
            "largest_intermediate_bytes",
        ]
    if compute_class == COMPUTE_CLASS_SVD:
        return [
            "block_shape_wall_time",
            "block_shape_groups",
            "decomposition_kernel_time",
        ]
    if compute_class == COMPUTE_CLASS_CONTRACTION_PLAN:
        if issue == "communication":
            return [
                "communication_wall_time",
                "communication_bytes_by_collective",
                "rank_local_work",
            ]
        return [
            "planned_lowering_counts",
            "workspace_bytes",
            "fallback_count_by_lowering",
        ]
    return [
        "wall_time",
        "working_set_bytes",
        "backend_kernel_time",
    ]


def _execution_model_from_practical(compute_class, practical_profile):
    diagnosis = practical_profile.get("diagnosis") or {}
    parallelism = practical_profile.get("parallelization") or {}
    issue = diagnosis.get("primary_issue") or practical_profile.get("primary_issue")
    scope = practical_profile.get("measurement_scope")
    primary_kernel = practical_profile.get("primary_kernel")

    if compute_class == COMPUTE_CLASS_TENSORDOT:
        if scope == "backend_contraction_execute":
            if issue == "python_rhs_loop":
                return "python_rhs_loop"
            if issue == "backend_fallback":
                return "backend_fallback_loop"
            if primary_kernel == "distributed":
                return "distributed_contraction"
            if parallelism.get("grouped_kernel"):
                return "grouped_backend_kernel"
            if parallelism.get("batched_kernel"):
                return "batched_backend_kernel"
            return "backend_lowering"
        return "single_backend_kernel"
    if compute_class == COMPUTE_CLASS_OE:
        return "sequential_path_with_backend_kernels"
    if compute_class == COMPUTE_CLASS_SVD:
        return "independent_qn_block_groups"
    if compute_class == COMPUTE_CLASS_CONTRACTION_PLAN:
        return "distributed_planner" if parallelism.get("distributed") else "backend_planner"
    return str(scope or "unknown")


def _parallelism_label_from_practical(compute_class, practical_profile):
    diagnosis = practical_profile.get("diagnosis") or {}
    parallelism = practical_profile.get("parallelization") or {}
    if compute_class == COMPUTE_CLASS_TENSORDOT:
        if practical_profile.get("measurement_scope") != "backend_contraction_execute":
            return parallelism.get("unit")
        return (
            diagnosis.get("parallelism_hint")
            or parallelism.get("parallelism_hint")
            or parallelism.get("unit")
        )
    if compute_class == COMPUTE_CLASS_OE:
        return "serial_path_steps" if parallelism.get("independent_path_steps") is False else parallelism.get("unit")
    if compute_class == COMPUTE_CLASS_SVD:
        return "independent_qn_block_groups"
    if compute_class == COMPUTE_CLASS_CONTRACTION_PLAN:
        return "distributed_contraction" if parallelism.get("distributed") else "backend_planner"
    return parallelism.get("unit")


def _execution_summary_from_practical(compute_class, practical_profile):
    practical_profile = practical_profile or {}
    diagnosis = practical_profile.get("diagnosis") or {}
    recommended_action = _optimization_recommended_action(practical_profile)
    return {
        "compute_class": compute_class,
        "execution_model": _execution_model_from_practical(compute_class, practical_profile),
        "primary_kernel": practical_profile.get("primary_kernel"),
        "dominant_cost_kind": practical_profile.get("dominant_cost_kind"),
        "recommended_action": recommended_action,
        "measurement_focus": diagnosis.get("measurement_focus"),
        "parallelism": _parallelism_label_from_practical(compute_class, practical_profile),
        "precision_level": practical_profile.get("precision_level"),
        "next_measurements": _practical_next_measurements(compute_class, practical_profile),
    }


def _int_profile_value(*values):
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _is_aggregate_practical_profile(practical_profile):
    scope = str(practical_profile.get("measurement_scope") or "")
    precision = str(practical_profile.get("precision_level") or "")
    return scope.endswith("_summary") or precision.startswith("aggregate_")


def _workload_call_granularity(compute_class, practical_profile):
    scope = practical_profile.get("measurement_scope")
    is_aggregate = _is_aggregate_practical_profile(practical_profile)
    if compute_class == COMPUTE_CLASS_TENSORDOT:
        if scope == "backend_contraction_execute":
            return "backend_lowering"
        return "aggregate" if is_aggregate else "single_kernel"
    if compute_class == COMPUTE_CLASS_OE:
        return "aggregate_path" if is_aggregate else "serial_path"
    if compute_class == COMPUTE_CLASS_SVD:
        return "aggregate_qn_block_groups" if is_aggregate else "qn_block_groups"
    if compute_class == COMPUTE_CLASS_CONTRACTION_PLAN:
        return "aggregate_plan" if is_aggregate else "plan"
    return str(scope or "unknown")


def _workload_parallelism_unit(compute_class, practical_profile):
    parallelism = practical_profile.get("parallelization") or {}
    if compute_class == COMPUTE_CLASS_TENSORDOT:
        if practical_profile.get("measurement_scope") == "backend_contraction_execute":
            return "backend_lowering"
        return parallelism.get("unit")
    if compute_class == COMPUTE_CLASS_OE:
        return "path_steps"
    if compute_class == COMPUTE_CLASS_SVD:
        return "qn_block_groups"
    if compute_class == COMPUTE_CLASS_CONTRACTION_PLAN:
        if parallelism.get("distributed"):
            return "distributed_contraction"
        if parallelism.get("grouped_kernel"):
            return "grouped_backend_kernel"
        if parallelism.get("batched_kernel"):
            return "batched_backend_kernel"
        return parallelism.get("unit") or "plan"
    return parallelism.get("unit")


def _workload_shape_signature(compute_class, practical_profile):
    work = practical_profile.get("dominant_work") or {}
    metrics = practical_profile.get("key_metrics") or {}
    parallelism = practical_profile.get("parallelization") or {}
    dominant_operation = practical_profile.get("dominant_operation") or {}
    is_aggregate = _is_aggregate_practical_profile(practical_profile)

    if compute_class == COMPUTE_CLASS_TENSORDOT:
        if practical_profile.get("measurement_scope") == "backend_contraction_execute":
            lowering = (
                work.get("lowering")
                or metrics.get("lowering")
                or practical_profile.get("primary_kernel")
            )
            return "lowering={0},rhs={1},gemm={2},batched={3},grouped={4}".format(
                lowering,
                _int_profile_value(work.get("num_rhs_loop_calls"), metrics.get("num_rhs_loop_calls")),
                _int_profile_value(work.get("num_gemm"), metrics.get("num_gemm")),
                _int_profile_value(work.get("num_batched_gemm"), metrics.get("num_batched_gemm")),
                _int_profile_value(work.get("num_grouped_tasks"), metrics.get("num_grouped_tasks")),
            )
        if is_aggregate:
            return "max_m={0},max_n={1},max_k={2}".format(
                _int_profile_value(work.get("max_m"), practical_profile.get("max_m"), dominant_operation.get("max_m")),
                _int_profile_value(work.get("max_n"), practical_profile.get("max_n"), dominant_operation.get("max_n")),
                _int_profile_value(work.get("max_k"), practical_profile.get("max_k"), dominant_operation.get("max_k")),
            )
        return "m={0},n={1},k={2},layout={3}".format(
            _int_profile_value(work.get("m"), metrics.get("m")),
            _int_profile_value(work.get("n"), metrics.get("n")),
            _int_profile_value(work.get("k"), metrics.get("k")),
            work.get("layout_hint") or metrics.get("layout_hint"),
        )

    if compute_class == COMPUTE_CLASS_OE:
        if is_aggregate:
            gemm_steps = _int_profile_value(parallelism.get("gemm_step_count"))
            non_gemm_steps = _int_profile_value(parallelism.get("non_gemm_step_count"))
            step_count = _int_profile_value(
                practical_profile.get("contraction_count"),
                metrics.get("contraction_count"),
                gemm_steps + non_gemm_steps,
            )
            return "steps={0},gemm={1},non_gemm={2},max_step_output={3}".format(
                step_count,
                gemm_steps,
                non_gemm_steps,
                _int_profile_value(
                    work.get("max_step_output_elements"),
                    practical_profile.get("max_step_output_elements"),
                    dominant_operation.get("max_step_output_elements"),
                ),
            )
        return "steps={0},gemm={1},non_gemm={2},dominant_step={3}:{4}".format(
            _int_profile_value(metrics.get("contraction_count")),
            _int_profile_value(parallelism.get("gemm_step_count")),
            _int_profile_value(parallelism.get("non_gemm_step_count")),
            work.get("step"),
            work.get("contraction_type"),
        )

    if compute_class == COMPUTE_CLASS_SVD:
        if is_aggregate:
            return "blocks={0},unique={1},batchable_groups={2}".format(
                _int_profile_value(practical_profile.get("block_count"), parallelism.get("block_count")),
                _int_profile_value(
                    practical_profile.get("unique_block_shape_count"),
                    parallelism.get("unique_block_shape_count"),
                ),
                _int_profile_value(
                    practical_profile.get("batchable_block_group_count"),
                    parallelism.get("batchable_block_group_count"),
                ),
            )
        return "blocks={0},unique={1},dominant={2},batchable_groups={3}".format(
            _int_profile_value(metrics.get("block_count"), parallelism.get("block_count")),
            _int_profile_value(metrics.get("unique_block_shape_count"), parallelism.get("unique_block_shape_count")),
            metrics.get("dominant_block_shape") or work.get("shape"),
            _int_profile_value(
                practical_profile.get("batchable_block_group_count"),
                parallelism.get("batchable_block_group_count"),
            ),
        )

    if compute_class == COMPUTE_CLASS_CONTRACTION_PLAN:
        return "lowering={0},gemm={1},batched={2},grouped={3},blocks={4},buckets={5}".format(
            work.get("lowering") or practical_profile.get("primary_kernel"),
            _int_profile_value(work.get("num_gemm"), metrics.get("num_gemm")),
            _int_profile_value(work.get("num_batched_gemm"), metrics.get("num_batched_gemm")),
            _int_profile_value(work.get("num_grouped_tasks"), metrics.get("num_grouped_tasks")),
            _int_profile_value(work.get("num_blocks"), metrics.get("num_blocks")),
            _int_profile_value(work.get("num_shape_buckets"), metrics.get("num_shape_buckets")),
        )

    return None


def _workload_signature_from_practical(compute_class, practical_profile, estimated_cost):
    if not compute_class:
        return None
    practical_profile = practical_profile or {}
    estimated_cost = estimated_cost or {}
    diagnosis = practical_profile.get("diagnosis") or {}
    parallelism = practical_profile.get("parallelization") or {}
    signature = {
        "schema": WORKLOAD_SIGNATURE_SCHEMA,
        "compute_class": compute_class,
        "measurement_scope": practical_profile.get("measurement_scope"),
        "call_granularity": _workload_call_granularity(compute_class, practical_profile),
        "kernel_family": practical_profile.get("primary_kernel"),
        "parallelism_unit": _workload_parallelism_unit(compute_class, practical_profile),
        "dominant_issue": diagnosis.get("primary_issue") or practical_profile.get("primary_issue"),
        "recommended_action": _optimization_recommended_action(practical_profile),
        "precision_level": practical_profile.get("precision_level"),
        "shape_signature": _workload_shape_signature(compute_class, practical_profile),
        "cost_source": estimated_cost.get("source"),
        "estimated_flops": int(estimated_cost.get("flops") or 0),
        "data_movement": {
            "read_bytes": int(estimated_cost.get("read_bytes") or 0),
            "write_bytes": int(estimated_cost.get("write_bytes") or 0),
            "copy_bytes": int(estimated_cost.get("copy_bytes") or 0),
            "communication_bytes": int(estimated_cost.get("communication_bytes") or 0),
        },
    }
    practical_bucket = practical_profile.get("practical_bucket") or {}
    if practical_bucket.get("aggregation_key") is not None:
        signature["practical_bucket_key"] = practical_bucket.get("aggregation_key")
    if practical_bucket.get("comparison_key") is not None:
        signature["practical_comparison_key"] = practical_bucket.get("comparison_key")
    if practical_bucket.get("parallel_unit") is not None:
        signature["practical_parallel_unit"] = practical_bucket.get("parallel_unit")
    if compute_class == COMPUTE_CLASS_CONTRACTION_PLAN:
        evidence = diagnosis.get("precision_evidence") or {}
        plan_hash = evidence.get("plan_hash")
        if plan_hash is not None:
            signature["plan_hash"] = plan_hash
        communication_num_messages = _int_profile_value(
            parallelism.get("communication_num_messages"),
            evidence.get("communication_num_messages"),
        )
        dominant_collective = (
            parallelism.get("dominant_communication_collective")
            or evidence.get("dominant_communication_collective")
        )
        if communication_num_messages or dominant_collective is not None:
            signature["communication"] = {
                "num_messages": communication_num_messages,
                "dominant_collective": dominant_collective,
            }
    return signature


def _scaled_int(value, scale):
    try:
        value = float(value or 0)
        scale = float(scale or 0)
    except (TypeError, ValueError):
        return 0
    return int(round(value * scale))


def _cost_factor(name, value, unit, source, priority, *, resource=None, accounting="primary"):
    try:
        numeric_value = float(value or 0)
    except (TypeError, ValueError):
        numeric_value = 0.0
    if numeric_value <= 0:
        return None
    return {
        "name": str(name),
        "value": int(numeric_value) if numeric_value.is_integer() else numeric_value,
        "unit": str(unit),
        "source": str(source),
        "resource": str(resource or unit),
        "accounting": str(accounting),
        "_priority": int(priority),
    }


def _factor_sort_key(item):
    return (-int(item.get("_priority", 0)), -float(item.get("value") or 0), str(item.get("name")))


def _public_cost_factors(factors):
    result = []
    for item in sorted((factor for factor in factors if factor), key=_factor_sort_key):
        public = dict(item)
        public.pop("_priority", None)
        result.append(public)
    return result


def _kernel_mix_item(
    kind,
    count=0,
    *,
    flops_estimate=0,
    bytes_value=0,
    resource=None,
    source=None,
    accounting="primary",
):
    count = int(count or 0)
    flops_estimate = int(flops_estimate or 0)
    bytes_value = int(bytes_value or 0)
    if count <= 0 and flops_estimate <= 0 and bytes_value <= 0:
        return None
    item = {
        "kind": str(kind),
        "count": count,
        "accounting": str(accounting),
    }
    if flops_estimate:
        item["flops_estimate"] = flops_estimate
    if bytes_value:
        item["bytes"] = bytes_value
    if resource is not None:
        item["resource"] = str(resource)
    if source is not None:
        item["source"] = str(source)
    return item


def _append_kernel_mix_item(items, item):
    if item is not None:
        items.append(item)


def _kernel_mix_wall_weight(item, basis):
    if basis == "flops_estimate":
        return int(item.get("flops_estimate") or 0)
    if basis == "count":
        return int(item.get("count") or 0)
    if basis == "bytes":
        return int(item.get("bytes") or 0)
    return 0


def _annotate_kernel_mix_wall_time(kernel_mix, total_wall_s):
    try:
        total_wall_s = float(total_wall_s)
    except (TypeError, ValueError):
        total_wall_s = None
    primary_items = [item for item in kernel_mix if item.get("accounting") == "primary"]
    non_additive = len(primary_items) != len(kernel_mix)
    if total_wall_s is None or total_wall_s < 0 or not primary_items:
        for item in kernel_mix:
            item["wall_attribution_basis"] = "not_attributed"
        return {
            "primary_kernel_mix_is_additive": bool(primary_items),
            "contains_non_additive_kernel_signals": non_additive,
            "time_attribution": [],
            "dominant_time_kind": None,
            "total_wall_s": total_wall_s,
        }

    basis = None
    for candidate in ("flops_estimate", "count", "bytes"):
        if sum(_kernel_mix_wall_weight(item, candidate) for item in primary_items) > 0:
            basis = candidate
            break

    if basis is None:
        for item in kernel_mix:
            item["wall_attribution_basis"] = "not_attributed"
        return {
            "primary_kernel_mix_is_additive": True,
            "contains_non_additive_kernel_signals": non_additive,
            "time_attribution": [],
            "dominant_time_kind": None,
            "total_wall_s": total_wall_s,
        }

    total_weight = sum(_kernel_mix_wall_weight(item, basis) for item in primary_items)
    attribution = []
    for item in kernel_mix:
        if item.get("accounting") != "primary":
            item["wall_s_estimate"] = None
            item["wall_fraction"] = None
            item["wall_attribution_basis"] = "not_attributed"
            continue
        weight = _kernel_mix_wall_weight(item, basis)
        fraction = _safe_fraction(weight, total_weight) or 0.0
        wall_s_estimate = total_wall_s * fraction
        item["wall_s_estimate"] = wall_s_estimate
        item["wall_fraction"] = fraction
        item["wall_attribution_basis"] = basis
        attribution.append({
            "kind": item["kind"],
            "accounting": "primary",
            "basis": basis,
            "weight": int(weight),
            "wall_s_estimate": wall_s_estimate,
            "wall_fraction": fraction,
        })
    attribution.sort(key=lambda item: (-float(item["wall_s_estimate"]), str(item["kind"])))
    return {
        "primary_kernel_mix_is_additive": True,
        "contains_non_additive_kernel_signals": non_additive,
        "time_attribution": attribution,
        "dominant_time_kind": attribution[0]["kind"] if attribution else None,
        "wall_attribution_basis": basis,
        "total_wall_s": total_wall_s,
    }


def _kernel_mix_groups(kernel_mix):
    groups = {
        "primary": [],
        "overhead": [],
        "subset": [],
    }
    for item in kernel_mix:
        accounting = item.get("accounting")
        if accounting in groups:
            groups[accounting].append(item["kind"])
    return groups


def _wall_time_measurement_kind(total_wall_s):
    try:
        total_wall_s = float(total_wall_s)
    except (TypeError, ValueError):
        return "not_measured"
    return "measured" if total_wall_s >= 0 else "not_measured"


def _per_kernel_wall_time_basis(wall_summary):
    if not wall_summary.get("time_attribution"):
        return "not_attributed"
    basis = wall_summary.get("wall_attribution_basis")
    if basis == "flops_estimate":
        return "estimated_from_primary_flops_estimate"
    if basis == "count":
        return "estimated_from_primary_count"
    if basis == "bytes":
        return "estimated_from_primary_bytes"
    return "estimated_from_primary_{0}".format(basis) if basis else "not_attributed"


def _core_measurement_basis(kernel_mix, wall_summary):
    non_additive = [
        item["kind"]
        for item in kernel_mix
        if item.get("accounting") != "primary"
    ]
    return {
        "total_wall_time": _wall_time_measurement_kind(wall_summary.get("total_wall_s")),
        "per_kernel_wall_time": _per_kernel_wall_time_basis(wall_summary),
        "primary_wall_time_is_additive": bool(wall_summary.get("primary_kernel_mix_is_additive")),
        "non_additive_signal_kinds": non_additive,
        "non_additive_signals_are_wall_attributed": False,
    }


def _kernel_signal_public_item(item, *, included_in_wall_ranking):
    public = {
        "kind": item["kind"],
        "count": int(item.get("count") or 0),
        "accounting": item.get("accounting"),
        "included_in_wall_ranking": bool(included_in_wall_ranking),
    }
    for key in (
        "flops_estimate",
        "bytes",
        "resource",
        "source",
        "wall_s_estimate",
        "wall_fraction",
        "wall_attribution_basis",
    ):
        if key in item:
            public[key] = item[key]
    return public


def _practical_wall_ranking_item(item):
    public = {
        "kind": item["kind"],
        "count": int(item.get("count") or 0),
    }
    for key in ("wall_s_estimate", "wall_fraction", "flops_estimate", "bytes"):
        value = item.get(key)
        if value is not None:
            public[key] = value
    return public


def _practical_diagnostic_signal_item(item):
    if item.get("bytes"):
        value = int(item.get("bytes") or 0)
        unit = "bytes"
    elif item.get("flops_estimate"):
        value = int(item.get("flops_estimate") or 0)
        unit = "flops"
    else:
        value = int(item.get("count") or 0)
        unit = "count"
    return {
        "kind": item["kind"],
        "signal_type": item.get("accounting"),
        "value": value,
        "unit": unit,
        "included_in_wall_ranking": False,
    }


def _item_kinds(items):
    return {str(item.get("kind")) for item in items}


def _diagnostic_focus_for_kind(kind):
    return {
        "rhs_python_loop": "batch_rhs_hop",
        "layout_copy": "avoid_axis_permutation_copies",
        "oe_intermediate": "limit_largest_intermediate",
        "batchable_qn_block": "batch_reused_qn_blocks",
        "tiny_qn_block": "reduce_tiny_block_overhead",
        "backend_plan_fallback": "remove_backend_fallback",
        "communication": "reduce_communication",
        "shape_bucket": "inspect_contraction_plan",
    }.get(str(kind))


def _practical_diagnostic_focuses(diagnostic_items):
    focuses = []
    for item in diagnostic_items:
        focus = _diagnostic_focus_for_kind(item.get("kind"))
        if focus and focus not in focuses:
            focuses.append(focus)
    return focuses


def _practical_recommended_focus(compute_class, dominant_kernel, primary_items, diagnostic_items):
    primary_kinds = _item_kinds(primary_items)
    diagnostic_kinds = _item_kinds(diagnostic_items)
    all_kinds = primary_kinds | diagnostic_kinds

    if "rhs_python_loop" in all_kinds:
        return "batch_rhs_hop"

    if compute_class == COMPUTE_CLASS_TENSORDOT:
        if dominant_kernel == "gemm_like":
            return "optimize_or_batch_gemm_like_tensordot"
        return "inspect_tensordot_kernel"

    if compute_class == COMPUTE_CLASS_OE:
        non_gemm_kinds = {"oe_tensordot_step", "oe_generic_einsum_step", "oe_non_gemm_step"}
        non_gemm_items = [
            item for item in primary_items
            if item.get("kind") in non_gemm_kinds
        ]
        primary_flops_available = any(int(item.get("flops_estimate") or 0) for item in primary_items)
        if dominant_kernel in non_gemm_kinds:
            return "reduce_non_gemm_path_cost"
        if non_gemm_items and not primary_flops_available:
            return "reduce_non_gemm_path_cost"
        if dominant_kernel == "oe_gemm_step":
            return "optimize_gemm_path"
        return "inspect_oe_contract_path"

    if compute_class == COMPUTE_CLASS_SVD:
        if "tiny_qn_block" in diagnostic_kinds:
            return "reduce_tiny_block_overhead"
        if "batchable_qn_block" in diagnostic_kinds:
            return "batch_reused_qn_blocks"
        return "inspect_qn_decomposition"

    if compute_class == COMPUTE_CLASS_CONTRACTION_PLAN:
        if "backend_plan_fallback" in diagnostic_kinds:
            return "remove_backend_fallback"
        if "communication" in diagnostic_kinds:
            return "reduce_communication"
        if dominant_kernel == "grouped_gemm_task":
            return "inspect_contraction_plan"
        return "inspect_contraction_plan"

    return "inspect_core_kernel"


def _practical_parallelization_hint(compute_class, primary_items, diagnostic_items):
    diagnostic_kinds = _item_kinds(diagnostic_items)
    if compute_class == COMPUTE_CLASS_TENSORDOT:
        if "rhs_python_loop" in diagnostic_kinds:
            return "rhs_batching_candidate"
        return "single_backend_kernel"
    if compute_class == COMPUTE_CLASS_OE:
        return "sequential_path_steps"
    if compute_class == COMPUTE_CLASS_SVD:
        if "batchable_qn_block" in diagnostic_kinds:
            return "batch_qn_block_groups"
        if sum(int(item.get("count") or 0) for item in primary_items) > 1:
            return "independent_qn_blocks"
        return "single_decomposition"
    if compute_class == COMPUTE_CLASS_CONTRACTION_PLAN:
        primary_kinds = _item_kinds(primary_items)
        if "grouped_gemm_task" in primary_kinds:
            return "shape_bucketed_grouped_gemm"
        if "batched_gemm_bucket" in primary_kinds:
            return "batched_gemm"
        if "communication" in diagnostic_kinds:
            return "distributed_contraction"
        return "backend_execution_plan"
    return "unknown"


def _practical_optimization_summary(compute_class, dominant_kernel, primary_items, diagnostic_items):
    return {
        "primary_bottleneck": dominant_kernel,
        "recommended_focus": _practical_recommended_focus(
            compute_class,
            dominant_kernel,
            primary_items,
            diagnostic_items,
        ),
        "diagnostic_focuses": _practical_diagnostic_focuses(diagnostic_items),
        "parallelization_hint": _practical_parallelization_hint(
            compute_class,
            primary_items,
            diagnostic_items,
        ),
        "diagnostic_signal_count": len(diagnostic_items),
    }


def _practical_view_headline(compute_class):
    if compute_class == COMPUTE_CLASS_TENSORDOT:
        return "single backend kernel"
    if compute_class == COMPUTE_CLASS_OE:
        return "serial contraction path"
    if compute_class == COMPUTE_CLASS_SVD:
        return "independent QN block decompositions"
    if compute_class == COMPUTE_CLASS_CONTRACTION_PLAN:
        return "backend execution plan"
    return str(compute_class or "unknown compute")


def _practical_kernel_view(compute_class, primary_items, overhead_items, subset_items, wall_summary):
    dominant_kernel = wall_summary.get("dominant_time_kind")
    if dominant_kernel is None and primary_items:
        dominant_kernel = primary_items[0].get("kind")
    diagnostic_items = [*overhead_items, *subset_items]
    return {
        "schema": PRACTICAL_KERNEL_VIEW_SCHEMA,
        "headline": _practical_view_headline(compute_class),
        "core_operation": _CORE_COMPUTE_FAMILIES.get(compute_class, str(compute_class or "unknown")),
        "dominant_kernel": dominant_kernel,
        "rank_basis": "wall_s_estimate" if wall_summary.get("time_attribution") else "not_attributed",
        "wall_time_ranking": [
            _practical_wall_ranking_item(item)
            for item in primary_items
        ],
        "diagnostic_signals": [
            _practical_diagnostic_signal_item(item)
            for item in diagnostic_items
        ],
        "accounting_model": {
            "primary_kernel_wall_time": "exclusive_estimate",
            "diagnostic_signal_wall_time": "not_attributed",
            "safe_to_sum_wall_time": ["primary_kernels"],
        },
        "optimization_summary": _practical_optimization_summary(
            compute_class,
            dominant_kernel,
            primary_items,
            diagnostic_items,
        ),
    }


def _kernel_signal_sort_key(item):
    wall_s = item.get("wall_s_estimate")
    if wall_s is None:
        wall_s = -1.0
    return (
        -float(wall_s),
        -int(item.get("flops_estimate") or 0),
        -int(item.get("bytes") or 0),
        -int(item.get("count") or 0),
        str(item.get("kind")),
    )


def _core_kernel_summary(compute_class, kernel_mix, wall_summary):
    primary_items = [item for item in kernel_mix if item.get("accounting") == "primary"]
    overhead_items = [item for item in kernel_mix if item.get("accounting") == "overhead"]
    subset_items = [item for item in kernel_mix if item.get("accounting") == "subset"]
    primary_items = sorted(primary_items, key=_kernel_signal_sort_key)
    overhead_items = sorted(overhead_items, key=_kernel_signal_sort_key)
    subset_items = sorted(subset_items, key=_kernel_signal_sort_key)

    time_attribution = wall_summary.get("time_attribution") or ()
    if time_attribution:
        rank_basis = "wall_s_estimate"
    else:
        rank_basis = "not_attributed"
    total_wall_s = wall_summary.get("total_wall_s")
    primary_wall_s = None
    if total_wall_s is not None and time_attribution:
        primary_wall_s = sum(
            float(item.get("wall_s_estimate") or 0.0)
            for item in primary_items
        )

    summary = {
        "schema": CORE_KERNEL_SUMMARY_SCHEMA,
        "compute_family": _CORE_COMPUTE_FAMILIES.get(compute_class, str(compute_class or "unknown")),
        "rank_basis": rank_basis,
        "dominant_primary_kernel": wall_summary.get("dominant_time_kind"),
        "primary_kernels": [
            _kernel_signal_public_item(item, included_in_wall_ranking=True)
            for item in primary_items
        ],
        "overhead_signals": [
            _kernel_signal_public_item(item, included_in_wall_ranking=False)
            for item in overhead_items
        ],
        "subset_signals": [
            _kernel_signal_public_item(item, included_in_wall_ranking=False)
            for item in subset_items
        ],
        "notes": {
            "primary_kernels": "ranked core compute kernels",
            "overhead_signals": "measured or estimated overhead, not additive with primary kernel wall time",
            "subset_signals": "diagnostic subsets of primary work, not additive with primary kernel wall time",
        },
    }
    summary["practical_view"] = _practical_kernel_view(
        compute_class,
        primary_items,
        overhead_items,
        subset_items,
        wall_summary,
    )
    if total_wall_s is not None:
        summary["total_wall_s"] = total_wall_s
    if primary_wall_s is not None:
        summary["primary_wall_s_estimate"] = primary_wall_s
        summary["unattributed_wall_s_estimate"] = max(float(total_wall_s) - primary_wall_s, 0.0)
    return summary


def _finalize_core_compute_profile(profile, kernel_mix, total_wall_s):
    wall_summary = _annotate_kernel_mix_wall_time(kernel_mix, total_wall_s)
    profile.update(wall_summary)
    profile["kernel_groups"] = _kernel_mix_groups(kernel_mix)
    profile["measurement_basis"] = _core_measurement_basis(kernel_mix, wall_summary)
    profile["core_kernel_summary"] = _core_kernel_summary(profile.get("compute_class"), kernel_mix, wall_summary)
    return profile


def _core_compute_profile(compute_class, practical_profile, estimated_cost, total_wall_s=None):
    practical_profile = practical_profile or {}
    estimated_cost = estimated_cost or {}
    if total_wall_s is None:
        total_wall_s = practical_profile.get("total_wall_s")
    breakdown = practical_profile.get("execution_breakdown") or {}
    parallelism = practical_profile.get("parallelization") or {}
    metrics = practical_profile.get("key_metrics") or {}
    flops = int(estimated_cost.get("flops") or 0)
    copy_bytes = int(estimated_cost.get("copy_bytes") or 0)
    communication_bytes = int(estimated_cost.get("communication_bytes") or 0)
    serial_depth = _int_profile_value(breakdown.get("serial_dependency_units"))
    independent_units = _int_profile_value(breakdown.get("independent_work_units"))
    batchable_units = _int_profile_value(breakdown.get("batchable_work_units"))
    kernel_mix = []

    if compute_class == COMPUTE_CLASS_TENSORDOT:
        dominant_work = practical_profile.get("dominant_work") or {}
        primary_kernel = practical_profile.get("primary_kernel") or dominant_work.get("kernel")
        if breakdown.get("gemm_like_work_units") is not None:
            gemm_like_units = _int_profile_value(breakdown.get("gemm_like_work_units"))
            gemm_like_source = "execution_breakdown"
        else:
            gemm_like_units = _int_profile_value(parallelism.get("total_gemm"))
            gemm_like_units += _int_profile_value(parallelism.get("total_batched_gemm"))
            gemm_like_units += _int_profile_value(parallelism.get("total_grouped_tasks"))
            gemm_like_source = "lowering_counts"
            if not gemm_like_units:
                if primary_kernel in _TENSORDOT_GEMM_FAMILY_KERNELS:
                    gemm_like_units = 1
                    gemm_like_source = "primary_kernel"
        if primary_kernel in _TENSORDOT_GEMM_FAMILY_KERNELS or gemm_like_units:
            primary_mix_kind = "gemm_like"
            primary_units = gemm_like_units
            primary_source = gemm_like_source
        elif primary_kernel == "outer":
            primary_mix_kind = "outer_product"
            primary_units = 1
            primary_source = "primary_kernel"
        elif str(primary_kernel or "").startswith("fallback"):
            primary_mix_kind = str(primary_kernel)
            primary_units = 0
            primary_source = "fallback"
        else:
            primary_mix_kind = str(primary_kernel or "tensordot_kernel")
            primary_units = independent_units or 1
            primary_source = "primary_kernel" if primary_kernel else "shape_model"
        primary_flops = 0 if primary_source == "fallback" else flops
        rhs_loop_calls = _int_profile_value(
            metrics.get("num_rhs_loop_calls"),
            parallelism.get("num_rhs_loop_calls"),
            breakdown.get("serial_dependency_units"),
        )
        _append_kernel_mix_item(
            kernel_mix,
            _kernel_mix_item(
                primary_mix_kind,
                primary_units,
                flops_estimate=primary_flops,
                resource="backend_kernel",
                source=primary_source,
            ),
        )
        _append_kernel_mix_item(
            kernel_mix,
            _kernel_mix_item(
                "rhs_python_loop",
                rhs_loop_calls,
                resource="python_loop",
                source="num_rhs_loop_calls",
                accounting="overhead",
            ),
        )
        _append_kernel_mix_item(
            kernel_mix,
            _kernel_mix_item(
                "layout_copy",
                1 if copy_bytes else 0,
                bytes_value=copy_bytes,
                resource="data_movement",
                source="copy_bytes",
                accounting="overhead",
            ),
        )
        profile = {
            "schema": CORE_COMPUTE_PROFILE_SCHEMA,
            "compute_class": compute_class,
            "measurement_scope": practical_profile.get("measurement_scope"),
            "execution_granularity": (
                "rhs_python_loop"
                if rhs_loop_calls
                else "single_backend_contraction"
            ),
            "primary_kernel": practical_profile.get("primary_kernel"),
            "serial_depth": rhs_loop_calls or 1,
            "independent_units": independent_units or 1,
            "batchable_units": batchable_units,
            "kernel_mix": kernel_mix,
            "data_movement": {
                "copy_bytes": copy_bytes,
                "communication_bytes": communication_bytes,
                "working_set_bytes": int(estimated_cost.get("working_set_bytes") or 0),
            },
            "precision_level": practical_profile.get("precision_level"),
        }
        if practical_profile.get("execution_resources") is not None:
            profile["execution_resources"] = practical_profile.get("execution_resources")
        if practical_profile.get("temporary_memory") is not None:
            profile["temporary_memory"] = practical_profile.get("temporary_memory")
        return _finalize_core_compute_profile(profile, kernel_mix, total_wall_s)

    if compute_class == COMPUTE_CLASS_OE:
        contraction_count = _int_profile_value(
            metrics.get("contraction_count"),
            practical_profile.get("contraction_count"),
            breakdown.get("serial_dependency_units"),
        )
        gemm_steps = _int_profile_value(parallelism.get("gemm_step_count"))
        tensordot_steps = _int_profile_value(
            parallelism.get("tensordot_step_count"),
            practical_profile.get("tensordot_step_count"),
            metrics.get("tensordot_step_count"),
        )
        generic_steps = _int_profile_value(
            parallelism.get("generic_einsum_step_count"),
            practical_profile.get("generic_einsum_step_count"),
            metrics.get("generic_einsum_step_count"),
        )
        non_gemm_steps = _int_profile_value(parallelism.get("non_gemm_step_count"))
        if not generic_steps and non_gemm_steps:
            generic_steps = max(non_gemm_steps - tensordot_steps, 0)
        gemm_flops = _int_profile_value(
            metrics.get("gemm_flops_estimate"),
            _scaled_int(practical_profile.get("gemm_flop_fraction"), flops),
        )
        tensordot_flops = _int_profile_value(
            metrics.get("tensordot_flops_estimate"),
            practical_profile.get("tensordot_flops_estimate"),
        )
        generic_flops = _int_profile_value(
            metrics.get("generic_einsum_flops_estimate"),
            practical_profile.get("generic_einsum_flops_estimate"),
        )
        non_gemm_flops = _int_profile_value(
            metrics.get("non_gemm_flops_estimate"),
            practical_profile.get("non_gemm_flops_estimate"),
        )
        residual_non_gemm_flops = max(non_gemm_flops - tensordot_flops - generic_flops, 0)
        residual_non_gemm_steps = max(non_gemm_steps - tensordot_steps - generic_steps, 0)
        _append_kernel_mix_item(
            kernel_mix,
            _kernel_mix_item(
                "oe_gemm_step",
                gemm_steps,
                flops_estimate=gemm_flops,
                resource="backend_kernel",
                source="contraction_type_counts",
            ),
        )
        _append_kernel_mix_item(
            kernel_mix,
            _kernel_mix_item(
                "oe_non_gemm_step",
                residual_non_gemm_steps,
                flops_estimate=residual_non_gemm_flops,
                resource="non_gemm_contraction",
                source="non_gemm_flops_estimate",
            ),
        )
        _append_kernel_mix_item(
            kernel_mix,
            _kernel_mix_item(
                "oe_tensordot_step",
                tensordot_steps,
                flops_estimate=tensordot_flops,
                resource="tensordot_kernel",
                source="contraction_type_counts",
            ),
        )
        _append_kernel_mix_item(
            kernel_mix,
            _kernel_mix_item(
                "oe_generic_einsum_step",
                generic_steps,
                flops_estimate=generic_flops,
                resource="generic_einsum",
                source="contraction_type_counts",
            ),
        )
        _append_kernel_mix_item(
            kernel_mix,
            _kernel_mix_item(
                "oe_intermediate",
                1 if int(practical_profile.get("max_largest_intermediate_elements") or 0) else 0,
                bytes_value=int((practical_profile.get("max_largest_intermediate_elements") or 0) * 8),
                resource="intermediate_memory",
                source="largest_intermediate_elements",
                accounting="overhead",
            ),
        )
        profile = {
            "schema": CORE_COMPUTE_PROFILE_SCHEMA,
            "compute_class": compute_class,
            "measurement_scope": practical_profile.get("measurement_scope"),
            "execution_granularity": "sequential_oe_contraction_path",
            "primary_kernel": practical_profile.get("primary_kernel"),
            "serial_depth": serial_depth or contraction_count,
            "independent_units": independent_units,
            "batchable_units": batchable_units,
            "kernel_mix": kernel_mix,
            "data_movement": {
                "copy_bytes": copy_bytes,
                "communication_bytes": communication_bytes,
                "working_set_bytes": int(estimated_cost.get("working_set_bytes") or 0),
            },
            "precision_level": practical_profile.get("precision_level"),
        }
        return _finalize_core_compute_profile(profile, kernel_mix, total_wall_s)

    if compute_class == COMPUTE_CLASS_SVD:
        block_count = _int_profile_value(
            metrics.get("block_count"),
            parallelism.get("block_count"),
            breakdown.get("independent_work_units"),
        )
        batchable_blocks = _int_profile_value(
            metrics.get("batchable_block_count"),
            parallelism.get("batchable_block_count"),
            breakdown.get("batchable_work_units"),
        )
        tiny_blocks = _int_profile_value(metrics.get("tiny_block_count"), practical_profile.get("tiny_block_count"))
        _append_kernel_mix_item(
            kernel_mix,
            _kernel_mix_item(
                "svd_qn_block",
                block_count,
                flops_estimate=flops,
                resource="decomposition",
                source="block_count",
            ),
        )
        _append_kernel_mix_item(
            kernel_mix,
            _kernel_mix_item(
                "batchable_qn_block",
                batchable_blocks,
                flops_estimate=_scaled_int(practical_profile.get("batchable_flop_fraction"), flops),
                resource="qn_block_batching",
                source="batchable_block_count",
                accounting="subset",
            ),
        )
        _append_kernel_mix_item(
            kernel_mix,
            _kernel_mix_item(
                "tiny_qn_block",
                tiny_blocks,
                resource="tiny_block_overhead",
                source="tiny_block_count",
                accounting="subset",
            ),
        )
        profile = {
            "schema": CORE_COMPUTE_PROFILE_SCHEMA,
            "compute_class": compute_class,
            "measurement_scope": practical_profile.get("measurement_scope"),
            "execution_granularity": "independent_qn_block_decompositions",
            "primary_kernel": practical_profile.get("primary_kernel"),
            "serial_depth": serial_depth,
            "independent_units": independent_units or block_count,
            "batchable_units": batchable_units or batchable_blocks,
            "kernel_mix": kernel_mix,
            "data_movement": {
                "copy_bytes": copy_bytes,
                "communication_bytes": communication_bytes,
                "working_set_bytes": int(estimated_cost.get("working_set_bytes") or 0),
            },
            "precision_level": practical_profile.get("precision_level"),
        }
        return _finalize_core_compute_profile(profile, kernel_mix, total_wall_s)

    if compute_class == COMPUTE_CLASS_CONTRACTION_PLAN:
        work = practical_profile.get("dominant_work") or {}
        num_gemm = _int_profile_value(work.get("num_gemm"), metrics.get("num_gemm"))
        num_batched = _int_profile_value(work.get("num_batched_gemm"), metrics.get("num_batched_gemm"))
        num_grouped = _int_profile_value(work.get("num_grouped_tasks"), metrics.get("num_grouped_tasks"))
        num_blocks = _int_profile_value(work.get("num_blocks"), metrics.get("num_blocks"))
        num_buckets = _int_profile_value(work.get("num_shape_buckets"), metrics.get("num_shape_buckets"))
        num_output_reductions = _int_profile_value(
            work.get("num_output_reduction_groups"),
            metrics.get("num_output_reduction_groups"),
        )
        num_scatter_add = _int_profile_value(
            work.get("num_scatter_add_tasks"),
            metrics.get("num_scatter_add_tasks"),
        )
        fallback_calls = _int_profile_value(
            practical_profile.get("fallback_calls"),
            metrics.get("fallback_calls"),
            1 if practical_profile.get("dominant_cost_kind") == "backend_fallback" else 0,
        )
        primary_work_units = max(num_grouped, num_batched, num_gemm)
        if num_grouped:
            _append_kernel_mix_item(
                kernel_mix,
                _kernel_mix_item(
                    "grouped_gemm_task",
                    num_grouped,
                    flops_estimate=flops,
                    resource="backend_grouped_kernel",
                    source="num_grouped_tasks",
                ),
            )
        elif num_batched:
            _append_kernel_mix_item(
                kernel_mix,
                _kernel_mix_item(
                    "batched_gemm_bucket",
                    num_batched,
                    flops_estimate=flops,
                    resource="backend_batched_kernel",
                    source="num_batched_gemm",
                ),
            )
        elif num_gemm:
            _append_kernel_mix_item(
                kernel_mix,
                _kernel_mix_item(
                    "gemm_task",
                    num_gemm,
                    flops_estimate=flops,
                    resource="backend_gemm_kernel",
                    source="num_gemm",
                ),
            )
        _append_kernel_mix_item(
            kernel_mix,
            _kernel_mix_item(
                "shape_bucket",
                num_buckets,
                resource="shape_grouping",
                source="num_shape_buckets",
                accounting="subset",
            ),
        )
        _append_kernel_mix_item(
            kernel_mix,
            _kernel_mix_item(
                "output_reduction",
                num_output_reductions or num_scatter_add,
                resource="scatter_add_or_reduce",
                source="output_reduction_groups",
                accounting="overhead",
            ),
        )
        _append_kernel_mix_item(
            kernel_mix,
            _kernel_mix_item(
                "backend_plan_fallback",
                fallback_calls,
                resource="missing_backend_kernel",
                source="fallback_calls",
                accounting="overhead",
            ),
        )
        _append_kernel_mix_item(
            kernel_mix,
            _kernel_mix_item(
                "layout_copy",
                1 if copy_bytes else 0,
                bytes_value=copy_bytes,
                resource="data_movement",
                source="copy_bytes",
                accounting="overhead",
            ),
        )
        _append_kernel_mix_item(
            kernel_mix,
            _kernel_mix_item(
                "communication",
                1 if communication_bytes else 0,
                bytes_value=communication_bytes,
                resource="communication",
                source="communication_bytes",
                accounting="overhead",
            ),
        )
        profile = {
            "schema": CORE_COMPUTE_PROFILE_SCHEMA,
            "compute_class": compute_class,
            "measurement_scope": practical_profile.get("measurement_scope"),
            "execution_granularity": (
                "distributed_execution_plan"
                if parallelism.get("distributed")
                else "backend_execution_plan"
            ),
            "primary_kernel": practical_profile.get("primary_kernel"),
            "serial_depth": serial_depth,
            "independent_units": independent_units or num_blocks or primary_work_units,
            "batchable_units": batchable_units or max(num_grouped, num_batched),
            "kernel_mix": kernel_mix,
            "data_movement": {
                "copy_bytes": copy_bytes,
                "communication_bytes": communication_bytes,
                "working_set_bytes": int(estimated_cost.get("working_set_bytes") or 0),
            },
            "precision_level": practical_profile.get("precision_level"),
        }
        if practical_profile.get("execution_resources") is not None:
            profile["execution_resources"] = practical_profile.get("execution_resources")
        if practical_profile.get("temporary_memory") is not None:
            profile["temporary_memory"] = practical_profile.get("temporary_memory")
        return _finalize_core_compute_profile(profile, kernel_mix, total_wall_s)

    return None


def _practical_execution_breakdown(compute_class, practical_profile, estimated_cost):
    practical_profile = practical_profile or {}
    estimated_cost = estimated_cost or {}
    metrics = practical_profile.get("key_metrics") or {}
    work = practical_profile.get("dominant_work") or {}
    parallelism = practical_profile.get("parallelization") or {}
    scope = practical_profile.get("measurement_scope")
    breakdown = {
        "schema": EXECUTION_BREAKDOWN_SCHEMA,
        "measurement_scope": scope,
        "work_unit": work.get("unit") or parallelism.get("unit") or compute_class,
        "estimated_flops": int(estimated_cost.get("flops") or 0),
        "working_set_bytes": int(estimated_cost.get("working_set_bytes") or 0),
        "copy_bytes": int(estimated_cost.get("copy_bytes") or 0),
        "serial_dependency_units": 0,
        "independent_work_units": 0,
        "gemm_like_work_units": 0,
        "non_gemm_work_units": 0,
        "batchable_work_units": 0,
        "batchable_group_count": 0,
    }

    if compute_class == COMPUTE_CLASS_TENSORDOT:
        call_count = _int_profile_value(parallelism.get("call_count"), 1)
        rhs_loop_calls = _int_profile_value(
            work.get("num_rhs_loop_calls"),
            metrics.get("num_rhs_loop_calls"),
            parallelism.get("num_rhs_loop_calls"),
            _scaled_int(practical_profile.get("rhs_loop_call_fraction"), call_count),
        )
        gemm_like_units = _int_profile_value(
            work.get("num_gemm"),
            metrics.get("num_gemm"),
            parallelism.get("total_gemm"),
        )
        gemm_like_units += _int_profile_value(
            work.get("num_batched_gemm"),
            metrics.get("num_batched_gemm"),
            parallelism.get("total_batched_gemm"),
        )
        gemm_like_units += _int_profile_value(
            work.get("num_grouped_tasks"),
            metrics.get("num_grouped_tasks"),
            parallelism.get("total_grouped_tasks"),
        )
        breakdown.update({
            "serial_dependency_units": rhs_loop_calls,
            "independent_work_units": rhs_loop_calls or call_count,
            "gemm_like_work_units": gemm_like_units,
            "batchable_work_units": (
                rhs_loop_calls
                if parallelism.get("rhs_batching_candidate")
                else (call_count if parallelism.get("batching_candidate") else 0)
            ),
        })
        return breakdown

    if compute_class == COMPUTE_CLASS_OE:
        gemm_steps = _int_profile_value(parallelism.get("gemm_step_count"))
        non_gemm_steps = _int_profile_value(parallelism.get("non_gemm_step_count"))
        step_count = _int_profile_value(metrics.get("contraction_count"), gemm_steps + non_gemm_steps)
        breakdown.update({
            "serial_dependency_units": step_count,
            "independent_work_units": 0,
            "gemm_like_work_units": gemm_steps,
            "non_gemm_work_units": non_gemm_steps,
        })
        return breakdown

    if compute_class == COMPUTE_CLASS_SVD:
        block_count = _int_profile_value(metrics.get("block_count"), parallelism.get("block_count"))
        batchable_blocks = _int_profile_value(
            metrics.get("batchable_block_count"),
            parallelism.get("batchable_block_count"),
        )
        batchable_groups = _int_profile_value(
            practical_profile.get("batchable_block_group_count"),
            metrics.get("batchable_block_group_count"),
            parallelism.get("batchable_block_group_count"),
        )
        breakdown.update({
            "independent_work_units": block_count,
            "batchable_work_units": batchable_blocks or batchable_groups,
            "batchable_group_count": batchable_groups,
        })
        return breakdown

    if compute_class == COMPUTE_CLASS_CONTRACTION_PLAN:
        num_gemm = _int_profile_value(work.get("num_gemm"), metrics.get("num_gemm"))
        num_batched = _int_profile_value(work.get("num_batched_gemm"), metrics.get("num_batched_gemm"))
        num_grouped = _int_profile_value(work.get("num_grouped_tasks"), metrics.get("num_grouped_tasks"))
        num_blocks = _int_profile_value(work.get("num_blocks"), metrics.get("num_blocks"))
        num_buckets = _int_profile_value(work.get("num_shape_buckets"), metrics.get("num_shape_buckets"))
        breakdown.update({
            "gemm_like_work_units": max(num_gemm, num_batched, num_grouped),
            "batchable_work_units": max(num_batched, num_grouped),
            "batchable_group_count": num_buckets,
            "independent_work_units": num_blocks or num_grouped or num_batched or num_gemm,
        })
    return breakdown


def _practical_parallelism_opportunity(compute_class, practical_profile):
    practical_profile = practical_profile or {}
    metrics = practical_profile.get("key_metrics") or {}
    parallelism = practical_profile.get("parallelization") or {}
    diagnosis = practical_profile.get("diagnosis") or {}

    if compute_class == COMPUTE_CLASS_TENSORDOT:
        call_count = _int_profile_value(parallelism.get("call_count"), 1)
        rhs_loop_calls = _int_profile_value(
            metrics.get("num_rhs_loop_calls"),
            parallelism.get("num_rhs_loop_calls"),
            _scaled_int(practical_profile.get("rhs_loop_call_fraction"), call_count),
        )
        if rhs_loop_calls:
            return {
                "kind": "rhs_batching",
                "candidate": True,
                "work_units": rhs_loop_calls,
                "reason": "python_rhs_loop",
            }
        if parallelism.get("batching_candidate"):
            return {
                "kind": "small_gemm_batching",
                "candidate": True,
                "work_units": call_count,
                "reason": diagnosis.get("parallelism_hint") or "small_gemm",
            }
        return {
            "kind": "backend_kernel",
            "candidate": False,
            "work_units": call_count,
            "reason": diagnosis.get("parallelism_hint") or "single_kernel",
        }

    if compute_class == COMPUTE_CLASS_OE:
        gemm_steps = _int_profile_value(parallelism.get("gemm_step_count"))
        non_gemm_steps = _int_profile_value(parallelism.get("non_gemm_step_count"))
        return {
            "kind": "path_kernel_specialization",
            "candidate": False,
            "work_units": _int_profile_value(metrics.get("contraction_count"), gemm_steps + non_gemm_steps),
            "reason": "sequential_oe_path",
        }

    if compute_class == COMPUTE_CLASS_SVD:
        batchable_blocks = _int_profile_value(
            metrics.get("batchable_block_count"),
            parallelism.get("batchable_block_count"),
        )
        batchable_groups = _int_profile_value(
            practical_profile.get("batchable_block_group_count"),
            metrics.get("batchable_block_group_count"),
            parallelism.get("batchable_block_group_count"),
        )
        if batchable_blocks or batchable_groups:
            return {
                "kind": "qn_block_grouping",
                "candidate": True,
                "work_units": batchable_blocks or batchable_groups,
                "reason": "reused_qn_block_shapes",
            }
        block_count = _int_profile_value(metrics.get("block_count"), parallelism.get("block_count"))
        return {
            "kind": "qn_block_parallelism",
            "candidate": block_count > 1,
            "work_units": block_count,
            "reason": "independent_qn_blocks" if block_count > 1 else "single_qn_block_group",
        }

    if compute_class == COMPUTE_CLASS_CONTRACTION_PLAN:
        work = practical_profile.get("dominant_work") or {}
        num_grouped = _int_profile_value(work.get("num_grouped_tasks"), metrics.get("num_grouped_tasks"))
        num_batched = _int_profile_value(work.get("num_batched_gemm"), metrics.get("num_batched_gemm"))
        num_gemm = _int_profile_value(work.get("num_gemm"), metrics.get("num_gemm"))
        num_buckets = _int_profile_value(work.get("num_shape_buckets"), metrics.get("num_shape_buckets"))
        if num_grouped:
            return {
                "kind": "grouped_gemm_execution",
                "candidate": bool(num_buckets),
                "work_units": num_grouped,
                "group_count": num_buckets,
                "reason": "shape_bucketed_grouped_gemm",
            }
        if num_batched:
            return {
                "kind": "batched_gemm_execution",
                "candidate": True,
                "work_units": num_batched,
                "group_count": num_buckets,
                "reason": "batched_backend_kernel",
            }
        return {
            "kind": "backend_execution_plan",
            "candidate": False,
            "work_units": num_gemm,
            "group_count": num_buckets,
            "reason": diagnosis.get("parallelism_hint") or "single_backend_plan",
        }

    return {
        "kind": str(compute_class or "unknown"),
        "candidate": False,
        "work_units": 0,
        "reason": "unclassified",
    }


def _practical_cost_factor_breakdown(compute_class, practical_profile, estimated_cost):
    practical_profile = practical_profile or {}
    estimated_cost = estimated_cost or {}
    metrics = practical_profile.get("key_metrics") or {}
    parallelism = practical_profile.get("parallelization") or {}
    factors = []
    diagnostic_factors = []

    if compute_class == COMPUTE_CLASS_TENSORDOT:
        call_count = _int_profile_value(parallelism.get("call_count"), 1)
        rhs_loop_calls = _int_profile_value(
            metrics.get("num_rhs_loop_calls"),
            parallelism.get("num_rhs_loop_calls"),
            _scaled_int(practical_profile.get("rhs_loop_call_fraction"), call_count),
        )
        factors.extend((
            _cost_factor("rhs_loop", rhs_loop_calls, "calls", "num_rhs_loop_calls", 100, resource="python_loop"),
            _cost_factor(
                "layout_copy",
                estimated_cost.get("copy_bytes"),
                "bytes",
                "copy_bytes",
                95,
                resource="data_movement",
            ),
            _cost_factor(
                "small_gemm_calls",
                _scaled_int(practical_profile.get("tiny_gemm_call_fraction"), call_count),
                "calls",
                "tiny_gemm_call_fraction",
                75,
                resource="small_gemm",
            ),
            _cost_factor(
                "compute_kernel_flops",
                estimated_cost.get("flops"),
                "flops",
                "estimated_flops",
                10,
                resource="compute",
            ),
        ))
    elif compute_class == COMPUTE_CLASS_OE:
        total_flops = int(estimated_cost.get("flops") or 0)
        gemm_flops = metrics.get("gemm_flops_estimate")
        non_gemm_flops = metrics.get("non_gemm_flops_estimate")
        tensordot_flops = metrics.get("tensordot_flops_estimate")
        generic_einsum_flops = metrics.get("generic_einsum_flops_estimate")
        non_gemm_steps = _int_profile_value(parallelism.get("non_gemm_step_count"))
        if gemm_flops is None:
            gemm_flops = _scaled_int(practical_profile.get("gemm_flop_fraction"), total_flops)
        if non_gemm_flops is None:
            non_gemm_flops = _scaled_int(practical_profile.get("non_gemm_flop_fraction"), total_flops)
        if tensordot_flops is None:
            tensordot_flops = practical_profile.get("tensordot_flops_estimate")
        if generic_einsum_flops is None:
            generic_einsum_flops = practical_profile.get("generic_einsum_flops_estimate")
        non_gemm_flops = _int_profile_value(non_gemm_flops)
        tensordot_flops = _int_profile_value(tensordot_flops)
        generic_einsum_flops = _int_profile_value(generic_einsum_flops)
        residual_non_gemm_flops = max(non_gemm_flops - tensordot_flops - generic_einsum_flops, 0)
        has_non_gemm_subcomponents = bool(tensordot_flops or generic_einsum_flops)
        if has_non_gemm_subcomponents:
            diagnostic_factors.append(
                _cost_factor(
                    "non_gemm_path_flops",
                    non_gemm_flops,
                    "flops",
                    "non_gemm_flops_estimate",
                    0,
                    resource="serial_path",
                    accounting="diagnostic_aggregate",
                )
            )
            factors.extend((
                _cost_factor(
                    "tensordot_path_flops",
                    tensordot_flops,
                    "flops",
                    "tensordot_flops_estimate",
                    85,
                    resource="tensordot_kernel",
                ),
                _cost_factor(
                    "generic_einsum_path_flops",
                    generic_einsum_flops,
                    "flops",
                    "generic_einsum_flops_estimate",
                    85,
                    resource="generic_einsum",
                ),
                _cost_factor(
                    "other_non_gemm_path_flops",
                    residual_non_gemm_flops,
                    "flops",
                    "non_gemm_flops_estimate_minus_known_subcomponents",
                    85,
                    resource="serial_path",
                ),
            ))
        else:
            factors.append(
                _cost_factor(
                    "non_gemm_path_flops",
                    non_gemm_flops,
                    "flops",
                    "non_gemm_flops_estimate",
                    85,
                    resource="serial_path",
                )
            )
        if non_gemm_steps and not non_gemm_flops:
            factors.append(
                _cost_factor(
                    "non_gemm_path_steps",
                    non_gemm_steps,
                    "steps",
                    "non_gemm_step_count",
                    80,
                    resource="serial_path",
                )
            )
        factors.extend((
            _cost_factor(
                "gemm_path_flops",
                gemm_flops,
                "flops",
                "gemm_flops_estimate",
                30,
                resource="backend_kernel",
            ),
            _cost_factor(
                "path_steps",
                _int_profile_value(
                    metrics.get("contraction_count"),
                    _int_profile_value(parallelism.get("gemm_step_count"))
                    + _int_profile_value(parallelism.get("non_gemm_step_count")),
                ),
                "steps",
                "contraction_count",
                20,
                resource="path_overhead",
            ),
            _cost_factor(
                "largest_intermediate",
                practical_profile.get("max_largest_intermediate_elements"),
                "elements",
                "largest_intermediate_elements",
                75,
                resource="intermediate_memory",
            ),
        ))
    elif compute_class == COMPUTE_CLASS_SVD:
        total_flops = int(estimated_cost.get("flops") or 0)
        cost_model = practical_profile.get("cost_model") or {}
        batchable_flops = metrics.get("batchable_flops_estimate")
        if batchable_flops is None:
            batchable_fraction = (
                metrics.get("batchable_flop_fraction"),
                practical_profile.get("batchable_flop_fraction"),
                cost_model.get("batchable_flop_fraction"),
            )
            batchable_flops = 0
            for value in batchable_fraction:
                if value is not None:
                    batchable_flops = _scaled_int(value, total_flops)
                    break
        factors.append(
            _cost_factor(
                "qn_block_decomposition_flops",
                total_flops,
                "flops",
                "estimated_flops",
                40,
                resource="decomposition",
            )
        )
        diagnostic_factors.extend((
            _cost_factor(
                "tiny_qn_blocks",
                metrics.get("tiny_block_count", practical_profile.get("tiny_block_count")),
                "blocks",
                "tiny_block_count",
                70,
                resource="tiny_block_overhead",
                accounting="diagnostic_subset",
            ),
            _cost_factor(
                "batchable_qn_block_flops",
                batchable_flops,
                "flops",
                "batchable_flops_estimate",
                60,
                resource="qn_block_batching",
                accounting="diagnostic_subset",
            ),
            _cost_factor(
                "shape_fragmentation",
                metrics.get("unique_block_shape_count", practical_profile.get("unique_block_shape_count")),
                "shapes",
                "unique_block_shape_count",
                55,
                resource="shape_fragmentation",
                accounting="diagnostic_signal",
            ),
        ))
    elif compute_class == COMPUTE_CLASS_CONTRACTION_PLAN:
        work = practical_profile.get("dominant_work") or {}
        fallback_calls = _int_profile_value(practical_profile.get("fallback_calls"), metrics.get("fallback_calls"))
        num_grouped = _int_profile_value(work.get("num_grouped_tasks"), metrics.get("num_grouped_tasks"))
        num_batched = _int_profile_value(work.get("num_batched_gemm"), metrics.get("num_batched_gemm"))
        num_gemm = _int_profile_value(work.get("num_gemm"), metrics.get("num_gemm"))
        num_buckets = _int_profile_value(work.get("num_shape_buckets"), metrics.get("num_shape_buckets"))
        num_reduction_groups = _int_profile_value(
            work.get("num_output_reduction_groups"),
            metrics.get("num_output_reduction_groups"),
        )
        num_scatter_add = _int_profile_value(
            work.get("num_scatter_add_tasks"),
            metrics.get("num_scatter_add_tasks"),
        )
        factors.extend((
            _cost_factor(
                "backend_plan_fallback",
                fallback_calls,
                "calls",
                "fallback_calls",
                100,
                resource="missing_backend_kernel",
            ),
            _cost_factor(
                "plan_copy_bytes",
                estimated_cost.get("copy_bytes"),
                "bytes",
                "copy_bytes",
                95,
                resource="data_movement",
            ),
            _cost_factor(
                "plan_communication_bytes",
                estimated_cost.get("communication_bytes"),
                "bytes",
                "communication_bytes",
                90,
                resource="communication",
            ),
            _cost_factor(
                "backend_plan_flops",
                estimated_cost.get("flops"),
                "flops",
                "estimated_flops",
                10,
                resource="compute",
            ),
        ))
        diagnostic_factors.extend((
            _cost_factor(
                "grouped_gemm_tasks",
                num_grouped,
                "tasks",
                "num_grouped_tasks",
                80,
                resource="backend_grouped_kernel",
                accounting="diagnostic_signal",
            ),
            _cost_factor(
                "batched_gemm_buckets",
                num_batched,
                "buckets",
                "num_batched_gemm",
                70,
                resource="backend_batched_kernel",
                accounting="diagnostic_signal",
            ),
            _cost_factor(
                "loop_gemm_tasks",
                num_gemm,
                "tasks",
                "num_gemm",
                65,
                resource="loop_matmul",
                accounting="diagnostic_signal",
            ),
            _cost_factor(
                "shape_buckets",
                num_buckets,
                "buckets",
                "num_shape_buckets",
                60,
                resource="shape_grouping",
                accounting="diagnostic_signal",
            ),
            _cost_factor(
                "output_reduction_groups",
                num_reduction_groups,
                "groups",
                "num_output_reduction_groups",
                55,
                resource="scatter_add_or_reduce",
                accounting="diagnostic_signal",
            ),
            _cost_factor(
                "scatter_add_tasks",
                num_scatter_add,
                "tasks",
                "num_scatter_add_tasks",
                50,
                resource="scatter_add_or_reduce",
                accounting="diagnostic_signal",
            ),
        ))
    else:
        factors.append(_cost_factor(
            "estimated_flops",
            estimated_cost.get("flops"),
            "flops",
            "estimated_flops",
            10,
            resource="compute",
        ))

    return {
        "rank_source": "bottleneck_priority_then_proxy_magnitude",
        "factors": _public_cost_factors(factors),
        "diagnostic_factors": _public_cost_factors(diagnostic_factors),
        "accounting_model": {
            "factors": "exclusive_primary_cost_factors",
            "diagnostic_factors": "non_additive_aggregate_or_subset_signals",
        },
        "limits": {
            "factor_wall_time": "not_measured",
            "weights": "proxy_values_may_mix_units",
        },
    }


def _cost_factor_numeric_value(item):
    try:
        return float(item.get("value") or 0)
    except (TypeError, ValueError):
        return 0.0


def _public_numeric_value(value):
    try:
        value = float(value or 0)
    except (TypeError, ValueError):
        return 0
    return int(value) if value.is_integer() else value


def _cost_factor_totals_by_unit(factors):
    totals = {}
    for item in factors or ():
        unit = item.get("unit")
        if unit is None:
            continue
        totals[str(unit)] = totals.get(str(unit), 0.0) + _cost_factor_numeric_value(item)
    return {key: _public_numeric_value(value) for key, value in totals.items() if value}


def _cost_factor_totals_by_resource(factors):
    totals = {}
    for item in factors or ():
        resource = item.get("resource") or item.get("unit")
        unit = item.get("unit")
        if resource is None or unit is None:
            continue
        resource_key = str(resource)
        unit_key = str(unit)
        resource_totals = totals.setdefault(resource_key, {})
        resource_totals[unit_key] = resource_totals.get(unit_key, 0.0) + _cost_factor_numeric_value(item)
    return {
        resource: {
            unit: _public_numeric_value(value)
            for unit, value in unit_totals.items()
            if value
        }
        for resource, unit_totals in totals.items()
    }


def _cost_factor_accounting_groups(factors):
    groups = {}
    for item in factors or ():
        accounting = str(item.get("accounting") or "diagnostic")
        group = groups.setdefault(accounting, {"items": []})
        group["items"].append(item)
    result = {}
    for accounting, group in groups.items():
        items = group["items"]
        result[accounting] = {
            "factor_names": [str(item.get("name")) for item in items],
            "totals_by_unit": _cost_factor_totals_by_unit(items),
            "totals_by_resource": _cost_factor_totals_by_resource(items),
        }
    return result


def _cost_factor_rollup(breakdown):
    breakdown = breakdown or {}
    primary = list(breakdown.get("factors") or ())
    diagnostic = list(breakdown.get("diagnostic_factors") or ())
    accounting_model = dict(breakdown.get("accounting_model") or {})
    return {
        "schema": COST_FACTOR_ROLLUP_SCHEMA,
        "rank_basis": accounting_model.get("factors") or "exclusive_primary_cost_factors",
        "ranking_method": breakdown.get("rank_source"),
        "accounting_model": accounting_model,
        "primary": {
            "factor_names": [str(item.get("name")) for item in primary],
            "top_factor": dict(primary[0]) if primary else None,
            "totals_by_unit": _cost_factor_totals_by_unit(primary),
            "totals_by_resource": _cost_factor_totals_by_resource(primary),
        },
        "diagnostic": {
            "factor_names": [str(item.get("name")) for item in diagnostic],
            "by_accounting": _cost_factor_accounting_groups(diagnostic),
        },
        "limits": dict(breakdown.get("limits") or {}),
    }


def _practical_kernel_view_from_core(core_compute_profile):
    if not isinstance(core_compute_profile, dict):
        return None
    core_summary = core_compute_profile.get("core_kernel_summary") or {}
    view = core_summary.get("practical_view")
    if isinstance(view, dict):
        return dict(view)
    return None


def _add_unique(items, value):
    if value and value not in items:
        items.append(value)


def _wall_time_status_from_core(core_compute_profile):
    measurement_basis = {}
    if isinstance(core_compute_profile, dict):
        measurement_basis = core_compute_profile.get("measurement_basis") or {}
    return {
        "event_wall_time": measurement_basis.get("total_wall_time") or "not_measured",
        "per_kernel_wall_time": measurement_basis.get("per_kernel_wall_time") or "not_attributed",
    }


def _measurement_plan_defaults(compute_class):
    if compute_class == COMPUTE_CLASS_TENSORDOT:
        return {
            "current_cost_proxy": "gemm_flops",
            "comparison_key": "m,n,k,layout_hint",
            "parallel_unit": "backend_kernel",
            "batching_key": "m,n,k,layout_hint",
            "critical_event_fields": [
                "m",
                "n",
                "k",
                "layout_hint",
                "flops_estimate",
                "axis_permutation_copy_bytes",
                "wall_s",
            ],
            "next_measurements": [
                "backend_kernel_wall_time_by_mnk_layout",
                "shape_bucket_call_count",
            ],
            "class_precision_gap": "backend_kernel_wall_time_by_mnk_layout",
        }
    if compute_class == COMPUTE_CLASS_OE:
        return {
            "current_cost_proxy": "path_type_counts",
            "comparison_key": "path_step_kernel_mix",
            "parallel_unit": "serial_path_steps",
            "batching_key": "step_contraction_type,input_shape,output_shape",
            "critical_event_fields": [
                "contraction_count",
                "step_costs",
                "top_step_costs",
                "largest_intermediate_elements",
                "wall_s",
            ],
            "next_measurements": [
                "per_path_step_wall_time",
                "actual_step_kernel_type",
                "intermediate_allocation_bytes",
            ],
            "class_precision_gap": "per_path_step_wall_time",
        }
    if compute_class == COMPUTE_CLASS_SVD:
        return {
            "current_cost_proxy": "block_shape_flops",
            "comparison_key": "qn_block_shape_groups",
            "parallel_unit": "independent_qn_blocks",
            "batching_key": "block_shape,dtype,mode",
            "critical_event_fields": [
                "block_count",
                "top_block_shape_groups",
                "batchable_block_group_count",
                "block_flops_estimate",
                "wall_s",
            ],
            "next_measurements": [
                "per_block_shape_wall_time",
                "block_shape_group_counts",
                "decomposition_kernel_wall_time",
            ],
            "class_precision_gap": "per_block_shape_wall_time",
        }
    if compute_class == COMPUTE_CLASS_CONTRACTION_PLAN:
        return {
            "current_cost_proxy": "planned_lowering_counts",
            "comparison_key": "lowering,num_blocks,num_shape_buckets",
            "parallel_unit": "backend_execution_plan",
            "batching_key": "shape_buckets",
            "critical_event_fields": [
                "lowering",
                "num_gemm",
                "num_batched_gemm",
                "num_grouped_tasks",
                "num_blocks",
                "num_shape_buckets",
                "flops",
                "read_bytes",
                "write_bytes",
                "copy_bytes",
                "workspace_bytes",
                "fallback_reason",
                "wall_s",
            ],
            "next_measurements": [
                "grouped_gemm_wall_time_by_shape_bucket",
                "bucket_occupancy",
                "fallback_count_by_lowering",
                "packing_copy_wall_time",
            ],
            "class_precision_gap": "per_shape_bucket_wall_time",
        }
    return {
        "current_cost_proxy": "estimated_flops",
        "comparison_key": "compute_class,primary_kernel",
        "parallel_unit": "unknown",
        "batching_key": "unknown",
        "critical_event_fields": ["flops_estimate", "wall_s"],
        "next_measurements": ["kernel_wall_time"],
        "class_precision_gap": "kernel_wall_time",
    }


def _contraction_execute_measurement_plan(practical_profile, estimated_cost, core_compute_profile):
    practical_profile = practical_profile or {}
    estimated_cost = estimated_cost or {}
    compute_class = practical_profile.get("workload_class") or COMPUTE_CLASS_TENSORDOT
    metrics = practical_profile.get("key_metrics") or {}
    work = practical_profile.get("dominant_work") or {}
    lowering = metrics.get("lowering") or work.get("lowering")
    wall_status = _wall_time_status_from_core(core_compute_profile)
    precision_gaps = []
    if wall_status["event_wall_time"] != "measured":
        _add_unique(precision_gaps, "total_event_wall_time")
    if wall_status["per_kernel_wall_time"] != "measured":
        _add_unique(precision_gaps, "per_kernel_wall_time")

    num_rhs_loop_calls = _int_profile_value(
        metrics.get("num_rhs_loop_calls"),
        work.get("num_rhs_loop_calls"),
    )
    num_grouped_tasks = _int_profile_value(
        metrics.get("num_grouped_tasks"),
        work.get("num_grouped_tasks"),
    )
    num_batched_gemm = _int_profile_value(
        metrics.get("num_batched_gemm"),
        work.get("num_batched_gemm"),
    )
    fallback_reason = metrics.get("fallback_reason")
    fallback_from = metrics.get("fallback_from")
    fallback_to = metrics.get("fallback_to")
    communication_bytes = _int_profile_value(
        metrics.get("communication_bytes"),
        estimated_cost.get("communication_bytes"),
    )
    copy_bytes = _int_profile_value(estimated_cost.get("copy_bytes"))

    if num_rhs_loop_calls:
        _add_unique(precision_gaps, "rhs_loop_wall_time")
        return {
            "schema": PRACTICAL_MEASUREMENT_PLAN_SCHEMA,
            "compute_class": compute_class,
            "measurement_scope": practical_profile.get("measurement_scope"),
            "precision_level": practical_profile.get("precision_level"),
            "cost_source": estimated_cost.get("source"),
            "current_cost_proxy": "rhs_loop_calls",
            "comparison_key": "lowering,num_rhs",
            "parallel_unit": "python_rhs_loop",
            "batching_key": "rhs_batch_axis",
            "wall_time_status": wall_status,
            "critical_event_fields": [
                "lowering",
                "num_rhs",
                "num_rhs_loop_calls",
                "fallback_reason",
                "input_shapes",
                "output_shape",
                "wall_s",
            ],
            "next_measurements": [
                "rhs_loop_wall_time",
                "batched_rhs_kernel_coverage",
                "rhs_batch_shape",
            ],
            "precision_gaps": precision_gaps,
        }

    if fallback_reason or str(lowering or "").startswith("fallback_"):
        _add_unique(precision_gaps, "fallback_kernel_wall_time")
        return {
            "schema": PRACTICAL_MEASUREMENT_PLAN_SCHEMA,
            "compute_class": compute_class,
            "measurement_scope": practical_profile.get("measurement_scope"),
            "precision_level": practical_profile.get("precision_level"),
            "cost_source": estimated_cost.get("source"),
            "current_cost_proxy": "fallback_route",
            "comparison_key": "fallback_from,fallback_to,lowering",
            "parallel_unit": "missing_backend_kernel",
            "batching_key": "fallback_from",
            "wall_time_status": wall_status,
            "critical_event_fields": [
                "lowering",
                "fallback_from",
                "fallback_to",
                "fallback_policy",
                "fallback_reason",
                "input_shapes",
                "output_shape",
                "wall_s",
            ],
            "next_measurements": [
                "fallback_count_by_route",
                "fallback_wall_time_by_route",
                "missing_backend_kernel_coverage",
            ],
            "precision_gaps": precision_gaps,
        }

    if communication_bytes:
        _add_unique(precision_gaps, "collective_wall_time")
        return {
            "schema": PRACTICAL_MEASUREMENT_PLAN_SCHEMA,
            "compute_class": compute_class,
            "measurement_scope": practical_profile.get("measurement_scope"),
            "precision_level": practical_profile.get("precision_level"),
            "cost_source": estimated_cost.get("source"),
            "current_cost_proxy": "communication_bytes",
            "comparison_key": "lowering,distributed_modes,collective",
            "parallel_unit": "distributed_collective",
            "batching_key": "distributed_modes",
            "wall_time_status": wall_status,
            "critical_event_fields": [
                "lowering",
                "distributed_modes",
                "communication",
                "rank",
                "world_size",
                "local_shape",
                "global_shape",
                "wall_s",
            ],
            "next_measurements": [
                "collective_wall_time_by_kind",
                "redistribution_bytes_by_mode",
                "local_contraction_wall_time",
            ],
            "precision_gaps": precision_gaps,
        }

    if lowering in ("grouped_gemm", "block_grouped_gemm") or num_grouped_tasks:
        _add_unique(precision_gaps, "grouped_gemm_wall_time_by_shape_bucket")
        if copy_bytes:
            _add_unique(precision_gaps, "packing_copy_wall_time")
        return {
            "schema": PRACTICAL_MEASUREMENT_PLAN_SCHEMA,
            "compute_class": compute_class,
            "measurement_scope": practical_profile.get("measurement_scope"),
            "precision_level": practical_profile.get("precision_level"),
            "cost_source": estimated_cost.get("source"),
            "current_cost_proxy": "grouped_gemm_tasks",
            "comparison_key": "lowering,num_blocks,num_shape_buckets",
            "parallel_unit": "grouped_backend_kernel",
            "batching_key": "shape_buckets",
            "wall_time_status": wall_status,
            "critical_event_fields": [
                "lowering",
                "input_shapes",
                "output_shape",
                "num_gemm",
                "num_batched_gemm",
                "num_grouped_tasks",
                "num_blocks",
                "num_shape_buckets",
                "flops",
                "copy_bytes",
                "workspace_bytes",
                "wall_s",
            ],
            "next_measurements": [
                "grouped_gemm_wall_time_by_shape_bucket",
                "bucket_occupancy",
                "packing_copy_wall_time",
            ],
            "precision_gaps": precision_gaps,
        }

    if lowering in ("batched_gemm", "strided_batched_gemm") or num_batched_gemm:
        _add_unique(precision_gaps, "batched_gemm_wall_time_by_shape")
        if copy_bytes:
            _add_unique(precision_gaps, "layout_or_pack_copy_wall_time")
        return {
            "schema": PRACTICAL_MEASUREMENT_PLAN_SCHEMA,
            "compute_class": compute_class,
            "measurement_scope": practical_profile.get("measurement_scope"),
            "precision_level": practical_profile.get("precision_level"),
            "cost_source": estimated_cost.get("source"),
            "current_cost_proxy": "batched_gemm_calls",
            "comparison_key": "lowering,input_shapes,output_shape",
            "parallel_unit": "batched_backend_kernel",
            "batching_key": "batch_shape,m,n,k",
            "wall_time_status": wall_status,
            "critical_event_fields": [
                "lowering",
                "input_shapes",
                "output_shape",
                "num_batched_gemm",
                "flops",
                "copy_bytes",
                "workspace_bytes",
                "wall_s",
            ],
            "next_measurements": [
                "batched_gemm_wall_time_by_shape",
                "batch_shape_bucket_count",
                "layout_or_pack_copy_wall_time",
            ],
            "precision_gaps": precision_gaps,
        }

    if copy_bytes:
        _add_unique(precision_gaps, "layout_copy_wall_time")
    return {
        "schema": PRACTICAL_MEASUREMENT_PLAN_SCHEMA,
        "compute_class": compute_class,
        "measurement_scope": practical_profile.get("measurement_scope"),
        "precision_level": practical_profile.get("precision_level"),
        "cost_source": estimated_cost.get("source"),
        "current_cost_proxy": "lowering_flops",
        "comparison_key": "lowering,input_shapes,output_shape,layout_hint",
        "parallel_unit": _contraction_execute_parallelism_hint(lowering),
        "batching_key": "m,n,k,layout_hint",
        "wall_time_status": wall_status,
        "critical_event_fields": [
            "lowering",
            "input_shapes",
            "output_shape",
            "num_gemm",
            "flops",
            "copy_bytes",
            "workspace_bytes",
            "wall_s",
        ],
        "next_measurements": [
            "lowering_wall_time_by_shape",
            "backend_kernel_wall_time",
            "layout_copy_wall_time",
        ],
        "precision_gaps": precision_gaps,
    }


def _practical_measurement_plan(compute_class, practical_profile, estimated_cost, core_compute_profile):
    practical_profile = practical_profile or {}
    estimated_cost = estimated_cost or {}
    if practical_profile.get("measurement_scope") == "backend_contraction_execute":
        return _contraction_execute_measurement_plan(
            practical_profile,
            estimated_cost,
            core_compute_profile,
        )
    defaults = _measurement_plan_defaults(compute_class)
    cost_proxy = defaults["current_cost_proxy"]
    next_measurements = list(defaults["next_measurements"])
    precision_gaps = []
    wall_status = _wall_time_status_from_core(core_compute_profile)

    if wall_status["event_wall_time"] != "measured":
        _add_unique(precision_gaps, "total_event_wall_time")
    if wall_status["per_kernel_wall_time"] != "measured":
        _add_unique(precision_gaps, "per_kernel_wall_time")

    if compute_class == COMPUTE_CLASS_TENSORDOT:
        if int(estimated_cost.get("copy_bytes") or 0):
            cost_proxy = "layout_copy_bytes"
            if "axis_permutation_copy_wall_time" not in next_measurements:
                next_measurements.insert(1, "axis_permutation_copy_wall_time")
            _add_unique(precision_gaps, "layout_copy_wall_time")
        parallelism = practical_profile.get("parallelization") or {}
        metrics = practical_profile.get("key_metrics") or {}
        rhs_loop_calls = _int_profile_value(
            metrics.get("num_rhs_loop_calls"),
            parallelism.get("num_rhs_loop_calls"),
        )
        if rhs_loop_calls:
            cost_proxy = "rhs_loop_calls"
            if "rhs_loop_wall_time" not in next_measurements:
                next_measurements.insert(0, "rhs_loop_wall_time")
            _add_unique(precision_gaps, "rhs_loop_wall_time")
    elif compute_class == COMPUTE_CLASS_OE:
        if practical_profile.get("precision_level") == "path_step_costs":
            cost_proxy = "path_step_flops"
        else:
            _add_unique(precision_gaps, "path_step_flops")
        _add_unique(precision_gaps, defaults["class_precision_gap"])
    elif compute_class == COMPUTE_CLASS_SVD:
        _add_unique(precision_gaps, defaults["class_precision_gap"])
    else:
        _add_unique(precision_gaps, defaults["class_precision_gap"])

    return {
        "schema": PRACTICAL_MEASUREMENT_PLAN_SCHEMA,
        "compute_class": compute_class,
        "measurement_scope": practical_profile.get("measurement_scope"),
        "precision_level": practical_profile.get("precision_level"),
        "cost_source": estimated_cost.get("source"),
        "current_cost_proxy": cost_proxy,
        "comparison_key": defaults["comparison_key"],
        "parallel_unit": defaults["parallel_unit"],
        "batching_key": defaults["batching_key"],
        "wall_time_status": wall_status,
        "critical_event_fields": list(defaults["critical_event_fields"]),
        "next_measurements": next_measurements,
        "precision_gaps": precision_gaps,
    }


def _optimization_profile_from_practical(compute_class, practical_profile, estimated_cost):
    practical_profile = practical_profile or {}
    estimated_cost = estimated_cost or {}
    cost_model = practical_profile.get("cost_model") or {}
    work = practical_profile.get("dominant_work") or {}
    parallelism = practical_profile.get("parallelization") or {}
    metrics = practical_profile.get("key_metrics") or {}
    bottleneck = _optimization_bottleneck(practical_profile)
    recommended_action = _optimization_recommended_action(practical_profile)
    communication_bytes = int(estimated_cost.get("communication_bytes") or 0)

    base = {
        "schema": OPTIMIZATION_PROFILE_SCHEMA,
        "bottleneck": bottleneck,
        "recommended_action": recommended_action,
        "actionability": _optimization_actionability(
            practical_profile,
            estimated_cost,
            bottleneck,
            recommended_action,
        ),
    }

    if compute_class == COMPUTE_CLASS_TENSORDOT:
        if practical_profile.get("measurement_scope") == "backend_contraction_execute":
            lowering = work.get("lowering") or metrics.get("lowering") or practical_profile.get("primary_kernel")
            num_rhs_loop_calls = int(work.get("num_rhs_loop_calls") or metrics.get("num_rhs_loop_calls") or 0)
            parallelism_model = {
                "unit": "backend_lowering",
                "kernel_parallelism": parallelism.get("parallelism_hint") or practical_profile.get("primary_kernel"),
                "lowering": lowering,
                "batched_kernel": bool(parallelism.get("batched_kernel")),
                "grouped_kernel": bool(parallelism.get("grouped_kernel")),
                "rhs_batching_candidate": bool(parallelism.get("rhs_batching_candidate")),
                "num_rhs_loop_calls": num_rhs_loop_calls,
            }
            if metrics.get("fallback_from") is not None:
                parallelism_model["fallback_from"] = metrics.get("fallback_from")
            if metrics.get("fallback_to") is not None:
                parallelism_model["fallback_to"] = metrics.get("fallback_to")
            base.update({
                "family": "tensor_contraction",
                "execution_model": _execution_model_from_practical(compute_class, practical_profile),
                "work_unit": {
                    "kind": lowering,
                    "num_gemm": int(work.get("num_gemm") or metrics.get("num_gemm") or 0),
                    "num_batched_gemm": int(
                        work.get("num_batched_gemm") or metrics.get("num_batched_gemm") or 0
                    ),
                    "num_grouped_tasks": int(
                        work.get("num_grouped_tasks") or metrics.get("num_grouped_tasks") or 0
                    ),
                    "num_rhs_loop_calls": num_rhs_loop_calls,
                },
                "parallelism_model": parallelism_model,
                "key_metrics": {
                    "flops": int(estimated_cost.get("flops") or 0),
                    "working_set_bytes": int(estimated_cost.get("working_set_bytes") or 0),
                    "copy_bytes": int(estimated_cost.get("copy_bytes") or 0),
                    "copy_fraction_of_working_set": metrics.get("copy_fraction_of_working_set"),
                    "communication_bytes": communication_bytes,
                    "fallback_reason": metrics.get("fallback_reason"),
                },
            })
            return base

        m = int(work.get("m") if work.get("m") is not None else work.get("max_m") or 0)
        n = int(work.get("n") if work.get("n") is not None else work.get("max_n") or 0)
        k = int(work.get("k") if work.get("k") is not None else work.get("max_k") or 0)
        copy_bytes = int(estimated_cost.get("copy_bytes") or metrics.get("axis_permutation_copy_bytes") or 0)
        working_set_bytes = int(estimated_cost.get("working_set_bytes") or 0)
        base.update({
            "family": "tensor_contraction",
            "execution_model": "single_backend_kernel",
            "work_unit": {
                "kind": work.get("kernel") or practical_profile.get("primary_kernel"),
                "m": m,
                "n": n,
                "k": k,
                "layout_hint": work.get("layout_hint"),
            },
            "parallelism_model": {
                "unit": "backend_kernel",
                "kernel_parallelism": parallelism.get("backend_kernel") or practical_profile.get("primary_kernel"),
                "independent_work_items": int(parallelism.get("independent_work_items") or 1),
                "batching_candidate": bool(parallelism.get("batching_candidate")),
                "rhs_batching_candidate": bool(parallelism.get("rhs_batching_candidate")),
            },
            "key_metrics": {
                "flops": int(estimated_cost.get("flops") or 0),
                "working_set_bytes": working_set_bytes,
                "effective_arithmetic_intensity_flops_per_byte": estimated_cost.get(
                    "effective_arithmetic_intensity_flops_per_byte"
                ),
                "copy_bytes": copy_bytes,
                "copy_fraction_of_working_set": (
                    metrics.get("copy_fraction_of_working_set")
                    if metrics.get("copy_fraction_of_working_set") is not None
                    else _safe_fraction(copy_bytes, working_set_bytes)
                ),
                "communication_bytes": communication_bytes,
            },
        })
        return base

    if compute_class == COMPUTE_CLASS_OE:
        dominant_step = work if isinstance(work, dict) else {}
        if dominant_step.get("unit") == "oe_path_type_mix":
            work_unit = {
                "kind": "path_type_mix",
                "contraction_count": int(dominant_step.get("contraction_count") or 0),
                "gemm_step_count": int(dominant_step.get("gemm_step_count") or 0),
                "tensordot_step_count": int(dominant_step.get("tensordot_step_count") or 0),
                "generic_einsum_step_count": int(dominant_step.get("generic_einsum_step_count") or 0),
                "non_gemm_step_count": int(dominant_step.get("non_gemm_step_count") or 0),
            }
        else:
            work_unit = {
                "kind": "path_step",
                "step": dominant_step.get("step"),
                "contraction_type": dominant_step.get("contraction_type"),
                "flops_estimate": int(dominant_step.get("flops_estimate") or 0),
                "flop_fraction": dominant_step.get("flop_fraction")
                if dominant_step.get("flop_fraction") is not None
                else practical_profile.get("dominant_step_flop_fraction"),
            }
        base.update({
            "family": "contraction_path",
            "execution_model": "sequential_path_with_backend_kernels",
            "work_unit": work_unit,
            "parallelism_model": {
                "unit": "path_steps",
                "path_steps_are_serial": not bool(parallelism.get("independent_path_steps")),
                "gemm_step_count": int(parallelism.get("gemm_step_count") or 0),
                "tensordot_step_count": int(parallelism.get("tensordot_step_count") or 0),
                "generic_einsum_step_count": int(parallelism.get("generic_einsum_step_count") or 0),
                "non_gemm_step_count": int(parallelism.get("non_gemm_step_count") or 0),
                "shape_diverse": bool(parallelism.get("shape_diverse")),
            },
            "key_metrics": {
                "flops": int(estimated_cost.get("flops") or 0),
                "gemm_flop_fraction": cost_model.get("gemm_flop_fraction")
                if cost_model.get("gemm_flop_fraction") is not None
                else practical_profile.get("gemm_flop_fraction"),
                "tensordot_flop_fraction": metrics.get("tensordot_flop_fraction")
                if metrics.get("tensordot_flop_fraction") is not None
                else practical_profile.get("tensordot_flop_fraction"),
                "generic_einsum_flop_fraction": metrics.get("generic_einsum_flop_fraction")
                if metrics.get("generic_einsum_flop_fraction") is not None
                else practical_profile.get("generic_einsum_flop_fraction"),
                "non_gemm_flop_fraction": cost_model.get("non_gemm_flop_fraction")
                if cost_model.get("non_gemm_flop_fraction") is not None
                else practical_profile.get("non_gemm_flop_fraction"),
                "dominant_step_flop_fraction": cost_model.get("dominant_step_flop_fraction")
                if cost_model.get("dominant_step_flop_fraction") is not None
                else practical_profile.get("dominant_step_flop_fraction"),
                "largest_intermediate_to_output_ratio": metrics.get("largest_intermediate_to_output_ratio"),
                "communication_bytes": communication_bytes,
            },
        })
        return base

    if compute_class == COMPUTE_CLASS_SVD:
        dominant_shape = work.get("shape") or practical_profile.get("dominant_block_shape")
        block_count = int(parallelism.get("block_count") or metrics.get("block_count") or 0)
        unique_shape_count = int(
            parallelism.get("unique_block_shape_count") or metrics.get("unique_block_shape_count") or 0
        )
        batchable_group_count = int(
            parallelism.get("batchable_block_group_count")
            or practical_profile.get("batchable_block_group_count")
            or 0
        )
        base.update({
            "family": "blocked_decomposition",
            "execution_model": "independent_qn_block_groups",
            "work_unit": {
                "kind": "qn_block_shape_group",
                "dominant_shape": dominant_shape,
                "block_count": block_count,
                "unique_block_shape_count": unique_shape_count,
                "batchable_block_group_count": batchable_group_count,
            },
            "parallelism_model": {
                "unit": "qn_block_groups",
                "independent_block_groups": True,
                "block_count": block_count,
                "unique_block_shape_count": unique_shape_count,
                "batchable_block_group_count": batchable_group_count,
                "shape_fragmentation": parallelism.get("shape_fragmentation"),
            },
            "key_metrics": {
                "flops": int(estimated_cost.get("flops") or 0),
                "qn_block_density": metrics.get("qn_block_density")
                if metrics.get("qn_block_density") is not None
                else practical_profile.get("qn_block_density"),
                "sparse_saved_flop_fraction": cost_model.get("sparse_saved_flop_fraction")
                if cost_model.get("sparse_saved_flop_fraction") is not None
                else practical_profile.get("sparse_saved_flop_fraction"),
                "batchable_flop_fraction": cost_model.get("batchable_flop_fraction")
                if cost_model.get("batchable_flop_fraction") is not None
                else practical_profile.get("batchable_flop_fraction"),
                "tiny_block_count": int(metrics.get("tiny_block_count") or practical_profile.get("tiny_block_count") or 0),
                "communication_bytes": communication_bytes,
            },
        })
        return base

    if compute_class == COMPUTE_CLASS_CONTRACTION_PLAN:
        communication_bytes = int(estimated_cost.get("communication_bytes") or 0)
        working_set_bytes = int(estimated_cost.get("working_set_bytes") or 0)
        parallelism_model = {
            "unit": parallelism.get("unit"),
            "distributed": bool(parallelism.get("distributed")),
        }
        if parallelism.get("communication_num_messages") is not None or parallelism.get(
            "dominant_communication_collective"
        ) is not None:
            parallelism_model.update({
                "communication_num_messages": int(parallelism.get("communication_num_messages") or 0),
                "dominant_communication_collective": parallelism.get("dominant_communication_collective"),
            })
        base.update({
            "family": "contraction_plan",
            "execution_model": "distributed_planner" if parallelism.get("distributed") else "backend_planner",
            "work_unit": dict(work),
            "parallelism_model": parallelism_model,
            "key_metrics": {
                "flops": int(estimated_cost.get("flops") or 0),
                "working_set_bytes": working_set_bytes,
                "copy_bytes": int(estimated_cost.get("copy_bytes") or 0),
                "communication_bytes": communication_bytes,
                "communication_fraction_of_working_set": _safe_fraction(
                    communication_bytes, working_set_bytes
                ),
            },
        })
        return base

    return None


def _event_top_work_items(payload, practical_profile):
    compute_class = payload.get("compute_class")
    if compute_class == COMPUTE_CLASS_OE:
        total_flops = int(payload.get("flops_estimate") or 0)
        items = []
        for item in payload.get("top_step_costs", ()) or ():
            if not isinstance(item, dict):
                continue
            step = {
                "step": item.get("step"),
                "contraction_type": item.get("contraction_type"),
                "flops_estimate": int(item.get("flops_estimate") or 0),
                "output_elements": int(item.get("output_elements") or 0),
            }
            step["flop_fraction"] = _safe_fraction(step["flops_estimate"], total_flops)
            items.append(step)
        if items:
            return items
    if compute_class == COMPUTE_CLASS_SVD:
        return [dict(item) for item in payload.get("top_block_shape_groups", ()) or () if isinstance(item, dict)]
    work = practical_profile.get("dominant_work")
    return [dict(work)] if isinstance(work, dict) and work else []


def _summary_top_work_items(practical_profile):
    items = practical_profile.get("top_problem_shapes") or practical_profile.get("top_kernels") or ()
    if items:
        return [dict(item) for item in items if isinstance(item, dict)]
    work = practical_profile.get("dominant_work")
    return [dict(work)] if isinstance(work, dict) and work else []


def _execution_route_from_profile(payload, practical_profile, estimated_cost, core_compute_profile):
    practical_profile = practical_profile or {}
    estimated_cost = estimated_cost or {}
    core_compute_profile = core_compute_profile or {}
    compute_class = payload.get("compute_class") or practical_profile.get("workload_class")
    parallelism = practical_profile.get("parallelization") or {}
    fallback_reasons = []
    for value in (
        payload.get("fallback_reason"),
        practical_profile.get("fallback_reason"),
    ):
        if value:
            _append_unique(fallback_reasons, str(value))
    for values in (
        payload.get("fallback_reasons"),
        practical_profile.get("fallback_reasons"),
    ):
        for value in values or ():
            if value:
                _append_unique(fallback_reasons, str(value))

    temporary_memory = practical_profile.get("temporary_memory") or {}
    workspace_bytes = _int_profile_value(
        payload.get("workspace_bytes"),
        practical_profile.get("workspace_bytes"),
        temporary_memory.get("max_workspace_required_bytes"),
        temporary_memory.get("max_workspace_provided_bytes"),
    )
    return {
        "schema": EXECUTION_ROUTE_SCHEMA,
        "compute_class": compute_class,
        "measurement_scope": practical_profile.get("measurement_scope"),
        "granularity": core_compute_profile.get("execution_granularity"),
        "primary_primitive": (
            practical_profile.get("primary_kernel")
            or core_compute_profile.get("primary_kernel")
            or payload.get("algorithmic_kernel")
            or payload.get("kernel_kind")
            or payload.get("lowering")
        ),
        "parallel_unit": parallelism.get("unit"),
        "serial_units": _int_profile_value(core_compute_profile.get("serial_depth")),
        "independent_units": _int_profile_value(core_compute_profile.get("independent_units")),
        "batchable_units": _int_profile_value(core_compute_profile.get("batchable_units")),
        "copy_bytes": int(estimated_cost.get("copy_bytes") or 0),
        "workspace_bytes": int(workspace_bytes),
        "communication_bytes": int(estimated_cost.get("communication_bytes") or 0),
        "working_set_bytes": int(estimated_cost.get("working_set_bytes") or 0),
        "fallback": bool(
            fallback_reasons
            or practical_profile.get("primary_issue") == "backend_fallback"
            or practical_profile.get("dominant_cost_kind") == "backend_fallback"
        ),
        "fallback_reasons": fallback_reasons,
    }


def _compute_profile_from_practical(
    payload,
    practical_profile,
    *,
    event_scope,
    top_work_items=None,
    call_count=None,
    total_wall_s=None,
    wall_fraction=None,
):
    practical_profile = practical_profile or {}
    compute_class = payload.get("compute_class") or practical_profile.get("workload_class")
    estimated_cost = _compact_estimated_cost(practical_profile.get("cost_model"))
    core_compute_profile = practical_profile.get("core_compute_profile") or _core_compute_profile(
        compute_class,
        practical_profile,
        estimated_cost,
        total_wall_s=total_wall_s,
    )
    workload_signature = _workload_signature_from_practical(
        compute_class,
        practical_profile,
        estimated_cost,
    )
    cost_factor_breakdown = _practical_cost_factor_breakdown(
        compute_class,
        practical_profile,
        estimated_cost,
    )
    practical_kernel_view = _practical_kernel_view_from_core(core_compute_profile)
    execution_route = _execution_route_from_profile(
        payload,
        practical_profile,
        estimated_cost,
        core_compute_profile,
    )
    result = {
        "schema": COMPUTE_PROFILE_SCHEMA,
        "event_scope": str(event_scope),
        "compute_class": compute_class,
        "compute_subclass": payload.get("compute_subclass"),
        "compute_role": payload.get("compute_role"),
        "compute_accounting": payload.get("compute_accounting"),
        "backend": payload.get("backend"),
        "device_kind": payload.get("device_kind"),
        "measurement_scope": practical_profile.get("measurement_scope"),
        "precision_level": practical_profile.get("precision_level"),
        "primary_kernel": practical_profile.get("primary_kernel"),
        "dominant_cost_kind": practical_profile.get("dominant_cost_kind"),
        "estimated_cost": estimated_cost,
        "work": practical_profile.get("dominant_work") or {},
        "parallelism": practical_profile.get("parallelization") or {},
        "top_work_items": top_work_items if top_work_items is not None else _summary_top_work_items(practical_profile),
        "optimization_targets": list(practical_profile.get("optimization_targets") or ()),
        "measurement_limits": practical_profile.get("measurement_limits") or {},
        "diagnosis": practical_profile.get("diagnosis") or {},
        "execution_summary": _execution_summary_from_practical(compute_class, practical_profile),
        "execution_breakdown": _practical_execution_breakdown(
            compute_class,
            practical_profile,
            estimated_cost,
        ),
        "execution_route": execution_route,
        "parallelism_opportunity": _practical_parallelism_opportunity(compute_class, practical_profile),
        "cost_factor_breakdown": cost_factor_breakdown,
        "cost_factor_rollup": _cost_factor_rollup(cost_factor_breakdown),
        "practical_measurement_plan": _practical_measurement_plan(
            compute_class,
            practical_profile,
            estimated_cost,
            core_compute_profile,
        ),
    }
    if practical_kernel_view is not None:
        result["practical_kernel_view"] = practical_kernel_view
    layout_profile = payload.get("layout_profile")
    if layout_profile is None and (
        payload.get("input_shapes") is not None
        or payload.get("output_shape") is not None
        or payload.get("layout_hint") is not None
    ):
        layout_profile = _standard_layout_profile(payload)
    if layout_profile is not None:
        result["layout_profile"] = layout_profile
    if practical_profile.get("execution_resources") is not None:
        result["execution_resources"] = practical_profile.get("execution_resources")
    if practical_profile.get("temporary_memory") is not None:
        result["temporary_memory"] = practical_profile.get("temporary_memory")
    if core_compute_profile is not None:
        result["core_compute_profile"] = core_compute_profile
    if workload_signature is not None:
        result["workload_signature"] = workload_signature
    if call_count is not None:
        result["call_count"] = int(call_count)
    if total_wall_s is not None:
        result["total_wall_s"] = float(total_wall_s)
    if wall_fraction is not None:
        result["wall_fraction"] = wall_fraction
    optimization_profile = _optimization_profile_from_practical(compute_class, practical_profile, estimated_cost)
    if optimization_profile is not None:
        result["optimization_profile"] = optimization_profile
    return result


def _event_compute_profile(payload):
    practical_profile = payload.get("practical_profile") or {}
    return _compute_profile_from_practical(
        payload,
        practical_profile,
        event_scope="single_event",
        top_work_items=_event_top_work_items(payload, practical_profile),
        total_wall_s=payload.get("wall_s"),
    )


def _primary_work_count_from_profiles(payload, core_summary):
    for name in (
        "num_grouped_tasks",
        "num_batched_gemm",
        "num_gemm",
        "block_count",
        "contraction_count",
    ):
        value = _int_profile_value(payload.get(name))
        if value:
            return value
    primary_kernels = core_summary.get("primary_kernels") or ()
    return sum(
        _int_profile_value(item.get("count"))
        for item in primary_kernels
        if isinstance(item, dict)
    )


def _core_operation_distributed_index(payload):
    communication_profile = payload.get("communication_profile")
    if not isinstance(communication_profile, dict):
        return None
    if not (
        communication_profile.get("distributed")
        or communication_profile.get("communication_required")
        or communication_profile.get("bytes")
    ):
        return None
    return {
        "distributed": bool(communication_profile.get("distributed")),
        "distributed_modes": list(communication_profile.get("distributed_modes") or ()),
        "rank": communication_profile.get("rank"),
        "world_size": communication_profile.get("world_size"),
        "local_shape": communication_profile.get("local_shape"),
        "global_shape": communication_profile.get("global_shape"),
        "communication": {
            "bytes": _int_profile_value(communication_profile.get("bytes")),
            "num_messages": _int_profile_value(communication_profile.get("num_messages")),
            "num_collectives": _int_profile_value(communication_profile.get("num_collectives")),
            "max_block_size": _int_profile_value(communication_profile.get("max_block_size")),
            "dominant_collective": communication_profile.get("dominant_collective"),
            "bytes_by_collective": dict(communication_profile.get("bytes_by_collective") or {}),
            "messages_by_collective": dict(communication_profile.get("messages_by_collective") or {}),
            "wall_s_by_collective": dict(communication_profile.get("wall_s_by_collective") or {}),
            "wall_s": communication_profile.get("wall_s"),
        },
    }


def _core_operation_fallback_index(payload):
    lowering_profile = payload.get("lowering_profile")
    if not isinstance(lowering_profile, dict):
        lowering_profile = {}
    fallback_from = lowering_profile.get("fallback_from", payload.get("fallback_from"))
    fallback_to = lowering_profile.get("fallback_to", payload.get("fallback_to"))
    fallback_policy = lowering_profile.get("fallback_policy", payload.get("fallback_policy"))
    fallback_reason = lowering_profile.get("fallback_reason", payload.get("fallback_reason"))
    fallback_used = bool(
        lowering_profile.get("fallback_used")
        or str(payload.get("lowering") or "").startswith("fallback")
        or fallback_from is not None
        or fallback_to is not None
        or fallback_reason is not None
    )
    if not fallback_used:
        return None
    lowering_route = lowering_profile.get("lowering_route")
    if isinstance(lowering_route, (list, tuple)):
        route = list(lowering_route)
    else:
        route = [value for value in (fallback_from, fallback_to) if value is not None]
    silent = lowering_profile.get("silent_fallback")
    if silent is None:
        silent = bool(fallback_used and (not fallback_reason or fallback_policy == "silent"))
    return {
        "used": True,
        "from": fallback_from,
        "to": fallback_to,
        "policy": fallback_policy,
        "reason": fallback_reason,
        "silent": bool(silent),
        "route": route,
    }


def _core_operation_index(payload):
    compute_profile = payload.get("compute_profile")
    if not isinstance(compute_profile, dict):
        return None
    practical_profile = payload.get("practical_profile")
    if not isinstance(practical_profile, dict):
        practical_profile = {}
    workload_signature = compute_profile.get("workload_signature") or {}
    execution_summary = compute_profile.get("execution_summary") or {}
    core_profile = (
        compute_profile.get("core_compute_profile")
        or practical_profile.get("core_compute_profile")
        or {}
    )
    core_summary = core_profile.get("core_kernel_summary") or {}
    practical_view = (
        compute_profile.get("practical_kernel_view")
        or core_summary.get("practical_view")
        or {}
    )
    observation = payload.get("kernel_observation") or {}
    observation_parallelism = observation.get("parallelism") or {}
    estimated_cost = compute_profile.get("estimated_cost") or {}
    compute_class = compute_profile.get("compute_class") or payload.get("compute_class")
    operation_family = (
        payload.get("operation_family")
        or practical_view.get("core_operation")
        or _CORE_COMPUTE_FAMILIES.get(compute_class, str(compute_class or "unknown"))
    )
    kernel = (
        payload.get("algorithmic_kernel")
        or compute_profile.get("primary_kernel")
        or execution_summary.get("primary_kernel")
    )
    parallelism_unit = workload_signature.get("parallelism_unit")
    if parallelism_unit == "backend_lowering":
        parallelism_unit = execution_summary.get("parallelism") or parallelism_unit
    result = {
        "schema": CORE_OPERATION_SCHEMA,
        "compute_class": compute_class,
        "operation_family": operation_family,
        "kernel": kernel,
        "dominant_kernel": (
            practical_view.get("dominant_kernel")
            or core_summary.get("dominant_primary_kernel")
        ),
        "execution_model": (
            execution_summary.get("execution_model")
            or core_profile.get("execution_granularity")
        ),
        "parallelism_unit": parallelism_unit or execution_summary.get("parallelism"),
        "practical_parallel_unit": (
            workload_signature.get("practical_parallel_unit")
            or observation_parallelism.get("unit")
            or execution_summary.get("parallelism")
        ),
        "shape_signature": workload_signature.get("shape_signature"),
        "practical_bucket_key": workload_signature.get("practical_bucket_key"),
        "primary_work_count": _primary_work_count_from_profiles(payload, core_summary),
        "serial_units": _int_profile_value(core_profile.get("serial_depth")),
        "independent_units": _int_profile_value(core_profile.get("independent_units")),
        "batchable_units": _int_profile_value(core_profile.get("batchable_units")),
        "flops": _int_profile_value(
            payload.get("flops"),
            payload.get("flops_estimate"),
            estimated_cost.get("flops"),
        ),
        "read_bytes": _int_profile_value(payload.get("read_bytes"), estimated_cost.get("read_bytes")),
        "write_bytes": _int_profile_value(payload.get("write_bytes"), estimated_cost.get("write_bytes")),
        "copy_bytes": _int_profile_value(payload.get("copy_bytes"), estimated_cost.get("copy_bytes")),
        "workspace_bytes": _int_profile_value(payload.get("workspace_bytes")),
        "communication_bytes": _int_profile_value(
            payload.get("comm_bytes"),
            payload.get("communication_bytes"),
            estimated_cost.get("communication_bytes"),
        ),
        "dominant_cost_kind": execution_summary.get("dominant_cost_kind"),
        "recommended_action": execution_summary.get("recommended_action"),
    }
    if payload.get("event") is not None:
        result["event"] = payload.get("event")
    distributed = _core_operation_distributed_index(payload)
    if distributed is not None:
        result["distributed"] = distributed
    fallback = _core_operation_fallback_index(payload)
    if fallback is not None:
        result["fallback"] = fallback
    return result


def _event_practical_profile_with_core(payload, practical_profile):
    if not isinstance(practical_profile, dict):
        return practical_profile
    compute_class = payload.get("compute_class") or practical_profile.get("workload_class")
    if not compute_class:
        return practical_profile
    profile = dict(practical_profile)
    core_profile = _core_compute_profile(
        compute_class,
        profile,
        _compact_estimated_cost(profile.get("cost_model")),
        total_wall_s=payload.get("wall_s"),
    )
    if core_profile is not None:
        profile["core_compute_profile"] = core_profile
    return profile


def _refresh_event_compute_profile(payload):
    if not payload.get("compute_class") or not isinstance(payload.get("practical_profile"), dict):
        return payload
    refreshed = dict(payload)
    refreshed["practical_profile"] = _event_practical_profile_with_core(
        refreshed,
        refreshed.get("practical_profile"),
    )
    refreshed["compute_profile"] = _event_compute_profile(refreshed)
    refreshed["core_operation"] = _core_operation_index(refreshed)
    return refreshed


def _contraction_plan_compute_profile(payload):
    lowering = payload.get("lowering")
    read_bytes = int(payload.get("read_bytes") or 0)
    write_bytes = int(payload.get("write_bytes") or 0)
    copy_bytes = int(payload.get("copy_bytes") or 0)
    communication_metrics = _communication_metrics_from_items(payload.get("communication"))
    comm_bytes = int(
        payload.get("comm_bytes")
        or payload.get("estimated_comm_bytes")
        or communication_metrics["bytes"]
        or 0
    )
    work = {
        "unit": "contraction_plan",
        "lowering": str(lowering) if lowering is not None else None,
        "num_gemm": int(payload.get("num_gemm") or 0),
        "num_batched_gemm": int(payload.get("num_batched_gemm") or 0),
        "num_grouped_tasks": int(payload.get("num_grouped_tasks") or 0),
        "num_blocks": int(payload.get("num_blocks") or 0),
        "num_shape_buckets": int(payload.get("num_shape_buckets") or 0),
    }
    targets = []
    if payload.get("fallback_reason"):
        _append_unique(targets, "remove_backend_fallback")
    if comm_bytes:
        _append_unique(targets, "reduce_communication")
    if copy_bytes:
        _append_unique(targets, "reduce_layout_or_device_copies")
    if payload.get("fallback_reason"):
        primary_issue = "backend_fallback"
    elif comm_bytes:
        primary_issue = "communication"
    elif copy_bytes:
        primary_issue = "copy_overhead"
    else:
        primary_issue = "planned_backend_lowering"
    parallelization = {
        "unit": "planned_backend_lowering",
        "backend_kernel": str(lowering) if lowering is not None else None,
        "batched_kernel": lowering in ("batched_gemm", "strided_batched_gemm"),
        "grouped_kernel": lowering in ("grouped_gemm", "block_grouped_gemm"),
        "distributed": lowering == "distributed",
    }
    if comm_bytes or communication_metrics["num_messages"]:
        parallelization.update({
            "communication_num_messages": int(communication_metrics["num_messages"]),
            "dominant_communication_collective": communication_metrics["dominant_collective"],
        })
    precision_evidence = {
        "plan_hash": payload.get("plan_hash"),
        "lowering": lowering,
        "flops": int(payload.get("flops") or 0),
        "working_set_bytes_estimate": read_bytes + write_bytes + copy_bytes + comm_bytes,
    }
    if comm_bytes or communication_metrics["num_messages"]:
        precision_evidence.update({
            "communication_bytes": comm_bytes,
            "communication_num_messages": int(communication_metrics["num_messages"]),
            "dominant_communication_collective": communication_metrics["dominant_collective"],
        })
    practical_profile = {
        "workload_class": COMPUTE_CLASS_CONTRACTION_PLAN,
        "measurement_scope": "contraction_plan",
        "precision_level": "planned_lowering_counters",
        "primary_kernel": str(lowering) if lowering is not None else None,
        "dominant_cost_kind": primary_issue,
        "cost_model": _practical_cost_model(
            "contraction_plan_estimate",
            flops_estimate=payload.get("flops"),
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            copy_bytes=copy_bytes,
            communication_bytes=comm_bytes,
            working_set_bytes=read_bytes + write_bytes + copy_bytes + comm_bytes,
        ),
        "dominant_work": work,
        "parallelization": parallelization,
        "measurement_limits": _measurement_limits("plan_estimate_only"),
        "optimization_targets": targets,
        "diagnosis": {
            "measurement_focus": "planning_metadata",
            "primary_issue": primary_issue,
            "precision_evidence": precision_evidence,
            "recommended_action": _first_target(targets, "inspect_contraction_plan"),
        },
    }
    return _compute_profile_from_practical(
        {
            "compute_class": COMPUTE_CLASS_CONTRACTION_PLAN,
            "backend": payload.get("backend"),
            "device_kind": payload.get("device_kind"),
        },
        practical_profile,
        event_scope="plan",
        top_work_items=[work],
    )


def _tensordot_event_practical_profile(payload):
    targets = []
    tags = set(payload.get("profile_tags") or ())
    copy_bytes = int(payload.get("axis_permutation_copy_bytes") or 0)
    working_set_bytes = int(payload.get("working_set_bytes_estimate") or 0)
    if payload.get("fallback_reason"):
        _append_unique(targets, "remove_backend_fallback")
    if int(payload.get("num_rhs_loop_calls") or 0):
        _append_unique(targets, "batch_rhs_hop")
    if copy_bytes:
        _append_unique(targets, "avoid_axis_permutation_copies")
    if "tiny_gemm" in tags:
        _append_unique(targets, "batch_or_fuse_tiny_gemm")
    if "skinny_gemm" in tags:
        _append_unique(targets, "optimize_skinny_gemm")

    if int(payload.get("num_rhs_loop_calls") or 0):
        primary_issue = "python_rhs_loop"
        parallelism_hint = "rhs_batching_needed"
    elif copy_bytes:
        primary_issue = "axis_permutation_copy"
        parallelism_hint = "tiny_or_skinny_gemm" if {"tiny_gemm", "skinny_gemm"} & tags else "layout_bound_gemm"
    elif {"tiny_gemm", "skinny_gemm"} & tags:
        primary_issue = "small_gemm"
        parallelism_hint = "tiny_or_skinny_gemm"
    elif payload.get("fallback_reason"):
        primary_issue = "backend_fallback"
        parallelism_hint = "missing_backend_kernel"
    else:
        primary_issue = "compute_kernel"
        parallelism_hint = "single_kernel"
    if int(payload.get("num_rhs_loop_calls") or 0):
        measurement_focus = "rhs_loop"
    elif copy_bytes:
        measurement_focus = "layout_copy"
    elif {"tiny_gemm", "skinny_gemm"} & tags:
        measurement_focus = "small_gemm"
    elif payload.get("fallback_reason"):
        measurement_focus = "backend_fallback"
    else:
        measurement_focus = "kernel"
    copy_fraction = _safe_fraction(copy_bytes, working_set_bytes)
    batching_candidate = bool(int(payload.get("num_rhs_loop_calls") or 0) or ({"tiny_gemm", "skinny_gemm"} & tags))

    profile = {
        "workload_class": COMPUTE_CLASS_TENSORDOT,
        "measurement_scope": "single_tensordot",
        "precision_level": "mnk_layout",
        "practical_bucket": payload.get("practical_bucket"),
        "dominant_cost_kind": primary_issue,
        "primary_kernel": payload.get("algorithmic_kernel") or payload.get("kernel_kind"),
        "primary_issue": primary_issue,
        "parallelism_hint": parallelism_hint,
        "cost_model": _practical_cost_model(
            "shape_estimate",
            flops_estimate=payload.get("flops_estimate"),
            read_bytes=payload.get("read_bytes"),
            write_bytes=payload.get("write_bytes"),
            copy_bytes=copy_bytes,
            working_set_bytes=working_set_bytes,
        ),
        "dominant_work": {
            "unit": "single_tensordot_kernel",
            "kernel": payload.get("algorithmic_kernel") or payload.get("kernel_kind"),
            "m": int(payload.get("m") or 0),
            "n": int(payload.get("n") or 0),
            "k": int(payload.get("k") or 0),
            "layout_hint": payload.get("layout_hint"),
        },
        "parallelization": {
            "unit": "backend_kernel",
            "backend_kernel": payload.get("algorithmic_kernel") or payload.get("kernel_kind"),
            "batching_candidate": batching_candidate,
            "rhs_batching_candidate": bool(int(payload.get("num_rhs_loop_calls") or 0)),
            "independent_work_items": 1,
        },
        "measurement_limits": _measurement_limits("python_wall_time_if_event_timed"),
        "dominant_operation": {
            "kind": payload.get("algorithmic_kernel") or payload.get("kernel_kind"),
            "m": int(payload.get("m") or 0),
            "n": int(payload.get("n") or 0),
            "k": int(payload.get("k") or 0),
            "layout_hint": payload.get("layout_hint"),
        },
        "key_metrics": {
            "m": int(payload.get("m") or 0),
            "n": int(payload.get("n") or 0),
            "k": int(payload.get("k") or 0),
            "axis_permutation_copy_bytes": copy_bytes,
            "copy_fraction_of_working_set": _safe_fraction(copy_bytes, working_set_bytes),
            "arithmetic_intensity_flops_per_byte": _safe_fraction(
                payload.get("flops_estimate"), payload.get("memory_bytes_estimate")
            ),
        },
        "optimization_targets": targets,
        "diagnosis": {
            "measurement_focus": measurement_focus,
            "primary_issue": primary_issue,
            "parallelism_hint": parallelism_hint,
            "batching_candidate": batching_candidate,
            "copy_bound_candidate": bool(copy_fraction),
            "precision_evidence": {
                "m": int(payload.get("m") or 0),
                "n": int(payload.get("n") or 0),
                "k": int(payload.get("k") or 0),
                "flops_estimate": int(payload.get("flops_estimate") or 0),
                "working_set_bytes_estimate": working_set_bytes,
                "copy_fraction": copy_fraction,
            },
            "recommended_action": _first_target(targets, "inspect_tensordot_kernel"),
        },
    }
    return _event_practical_profile_with_core(payload, profile)


def _oe_event_path_regime(gemm_fraction, non_gemm_fraction):
    if non_gemm_fraction is None:
        return "unknown_path"
    if non_gemm_fraction >= 0.5:
        return "non_gemm_flop_dominant"
    if non_gemm_fraction >= 0.1:
        return "mixed_flop_path"
    if gemm_fraction:
        return "gemm_flop_dominant"
    return "unknown_path"


def _oe_type_count_path_regime(gemm_steps, non_gemm_steps):
    if gemm_steps and non_gemm_steps:
        return "mixed_type_count_path"
    if non_gemm_steps:
        return "non_gemm_type_count_path"
    if gemm_steps:
        return "gemm_type_count_path"
    return "unknown_path"


def _oe_event_practical_profile(payload, contraction_count=None):
    targets = []
    if contraction_count is None:
        contraction_count = payload.get("contraction_count")
    contraction_count = int(contraction_count or 0)
    gemm_steps = int(payload.get("gemm_step_count") or 0)
    non_gemm_steps = int(payload.get("non_gemm_step_count") or 0)
    tensordot_steps = int(payload.get("tensordot_step_count") or 0)
    generic_einsum_steps = int(payload.get("generic_einsum_step_count") or 0)
    if not (tensordot_steps or generic_einsum_steps):
        counts = {
            str(key).upper(): int(value)
            for key, value in (payload.get("contraction_type_counts") or {}).items()
        }
        tensordot_steps = sum(counts.get(step_type, 0) for step_type in _OE_TENSORDOT_STEP_TYPES)
        generic_einsum_steps = max(
            int(sum(counts.values())) - int(counts.get("GEMM", 0)) - tensordot_steps,
            0,
        )
    gemm_flops = int(payload.get("gemm_flops_estimate") or 0)
    tensordot_flops = int(payload.get("tensordot_flops_estimate") or 0)
    generic_einsum_flops = int(payload.get("generic_einsum_flops_estimate") or 0)
    non_gemm_flops = int(payload.get("non_gemm_flops_estimate") or 0)
    total_flops = int(payload.get("flops_estimate") or 0) or gemm_flops + non_gemm_flops
    step_costs_available = bool(payload.get("step_costs_available"))
    flop_split_available = bool(payload.get("flop_split_available"))
    if flop_split_available:
        gemm_fraction = _safe_fraction(gemm_flops, total_flops)
        tensordot_fraction = _safe_fraction(tensordot_flops, total_flops)
        generic_einsum_fraction = _safe_fraction(generic_einsum_flops, total_flops)
        non_gemm_fraction = _safe_fraction(non_gemm_flops, total_flops)
        dominant_fraction = _safe_fraction(payload.get("dominant_step_flops_estimate"), total_flops)
        path_regime = _oe_event_path_regime(gemm_fraction, non_gemm_fraction)
        precision_level = "path_step_costs"
        cost_source = "oe_path_step_estimate"
    else:
        gemm_fraction = None
        tensordot_fraction = None
        generic_einsum_fraction = None
        non_gemm_fraction = None
        dominant_fraction = None
        path_regime = _oe_type_count_path_regime(gemm_steps, non_gemm_steps)
        precision_level = "path_type_counts" if contraction_count else "path_unknown"
        cost_source = "oe_path_type_count_estimate"
    largest_ratio = payload.get("largest_intermediate_to_output_ratio")
    shape_diverse = int(payload.get("unique_step_output_shape_count") or 0) > 1

    if non_gemm_steps or non_gemm_flops:
        _append_unique(targets, "reduce_non_gemm_path_cost")
    if largest_ratio is not None and largest_ratio >= 8:
        _append_unique(targets, "limit_largest_intermediate")

    if non_gemm_steps or non_gemm_flops:
        primary_issue = "non_gemm_path"
    elif largest_ratio is not None and largest_ratio >= 8:
        primary_issue = "large_intermediate"
    else:
        primary_issue = "path_overhead"
    if non_gemm_steps or non_gemm_flops:
        measurement_focus = "path_kernel_mix"
    elif largest_ratio is not None and largest_ratio >= 8:
        measurement_focus = "intermediate_pressure"
    else:
        measurement_focus = "path_overhead"

    if step_costs_available and payload.get("dominant_step"):
        dominant_work = {
            "unit": "oe_path_step",
            **payload.get("dominant_step"),
            "flop_fraction": dominant_fraction,
        }
        dominant_operation = payload.get("dominant_step")
    else:
        dominant_work = {
            "unit": "oe_path_type_mix",
            "contraction_count": contraction_count,
            "gemm_step_count": gemm_steps,
            "tensordot_step_count": tensordot_steps,
            "generic_einsum_step_count": generic_einsum_steps,
            "non_gemm_step_count": non_gemm_steps,
        }
        dominant_operation = dict(dominant_work)

    profile = {
        "workload_class": COMPUTE_CLASS_OE,
        "measurement_scope": "oe_path",
        "precision_level": precision_level,
        "practical_bucket": payload.get("practical_bucket"),
        "dominant_cost_kind": primary_issue,
        "primary_kernel": payload.get("algorithmic_kernel") or payload.get("kernel_kind"),
        "primary_issue": primary_issue,
        "path_regime": path_regime,
        "cost_model": _practical_cost_model(
            cost_source,
            flops_estimate=total_flops,
            read_bytes=payload.get("read_bytes"),
            write_bytes=payload.get("write_bytes"),
            copy_bytes=payload.get("copy_bytes"),
            working_set_bytes=(
                int(payload.get("read_bytes") or 0)
                + int(payload.get("write_bytes") or 0)
                + int(payload.get("workspace_bytes") or 0)
            ),
            extra={
                "gemm_flop_fraction": gemm_fraction,
                "non_gemm_flop_fraction": non_gemm_fraction,
                "dominant_step_flop_fraction": dominant_fraction,
            },
        ),
        "dominant_work": dominant_work,
        "parallelization": {
            "unit": "sequential_path_steps_with_backend_parallel_kernels",
            "independent_path_steps": False,
            "gemm_step_count": gemm_steps,
            "tensordot_step_count": tensordot_steps,
            "generic_einsum_step_count": generic_einsum_steps,
            "non_gemm_step_count": non_gemm_steps,
            "shape_diverse": shape_diverse,
        },
        "measurement_limits": _measurement_limits("python_wall_time_if_event_timed"),
        "dominant_operation": dominant_operation,
        "top_step_costs": payload.get("top_step_costs", ()),
        "key_metrics": {
            "contraction_count": contraction_count,
            "gemm_step_fraction": _safe_fraction(gemm_steps, contraction_count),
            "tensordot_step_fraction": _safe_fraction(tensordot_steps, contraction_count),
            "generic_einsum_step_fraction": _safe_fraction(generic_einsum_steps, contraction_count),
            "non_gemm_step_fraction": _safe_fraction(non_gemm_steps, contraction_count),
            "gemm_flop_fraction": gemm_fraction,
            "tensordot_flop_fraction": tensordot_fraction,
            "generic_einsum_flop_fraction": generic_einsum_fraction,
            "non_gemm_flop_fraction": non_gemm_fraction,
            "gemm_flops_estimate": gemm_flops,
            "tensordot_flops_estimate": tensordot_flops,
            "generic_einsum_flops_estimate": generic_einsum_flops,
            "non_gemm_flops_estimate": non_gemm_flops,
            "dominant_step_type": payload.get("dominant_step_type"),
            "dominant_step_flops_estimate": int(payload.get("dominant_step_flops_estimate") or 0),
            "dominant_step_flop_fraction": dominant_fraction,
            "max_step_output_elements": int(payload.get("max_step_output_elements") or 0),
            "total_step_output_elements": int(payload.get("total_step_output_elements") or 0),
            "unique_step_output_shape_count": int(payload.get("unique_step_output_shape_count") or 0),
            "largest_intermediate_to_output_ratio": largest_ratio,
        },
        "max_largest_intermediate_elements": int(payload.get("largest_intermediate_elements") or 0),
        "optimization_targets": targets,
        "diagnosis": {
            "measurement_focus": measurement_focus,
            "primary_issue": primary_issue,
            "path_regime": path_regime,
            "precision_evidence": {
                "precision_level": precision_level,
                "step_costs_available": step_costs_available,
                "flop_split_available": flop_split_available,
                "gemm_flop_fraction": gemm_fraction,
                "non_gemm_flop_fraction": non_gemm_fraction,
                "dominant_step_flop_fraction": dominant_fraction,
                "kernel_mix": {
                    "gemm_steps": gemm_steps,
                    "tensordot_steps": tensordot_steps,
                    "generic_einsum_steps": generic_einsum_steps,
                    "non_gemm_steps": non_gemm_steps,
                    "gemm_flops_estimate": gemm_flops,
                    "tensordot_flops_estimate": tensordot_flops,
                    "generic_einsum_flops_estimate": generic_einsum_flops,
                    "non_gemm_flops_estimate": non_gemm_flops,
                },
            },
            "recommended_action": _first_target(targets, "inspect_oe_contract_path"),
        },
    }
    return _event_practical_profile_with_core(payload, profile)


def _svd_event_practical_profile(payload):
    targets = []
    block_count = int(payload.get("block_count") or 0)
    unique_block_shapes = int(payload.get("unique_block_shape_count") or 0)
    qn_density = payload.get("qn_block_density")
    sparse_saved = None
    block_fraction = payload.get("block_flop_fraction")
    if block_fraction is not None:
        sparse_saved = 1.0 - float(block_fraction)
    batchable_fraction = payload.get("batchable_flop_fraction")

    if int(payload.get("batchable_flops_estimate") or 0):
        _append_unique(targets, "batch_reused_qn_blocks")
    if int(payload.get("tiny_block_count") or 0):
        _append_unique(targets, "reduce_tiny_block_overhead")
    if sparse_saved:
        _append_unique(targets, "preserve_qn_sparsity")
    if unique_block_shapes > 1:
        _append_unique(targets, "manage_block_shape_fragmentation")

    if sparse_saved and batchable_fraction:
        block_regime = "sparse_and_batchable"
    elif sparse_saved:
        block_regime = "sparse_qn_blocks"
    elif batchable_fraction:
        block_regime = "batchable_qn_blocks"
    elif qn_density is not None:
        block_regime = "dense_like_blocks"
    else:
        block_regime = "unknown_blocks"

    if int(payload.get("tiny_block_count") or 0):
        primary_issue = "tiny_qn_blocks"
    elif unique_block_shapes > 1:
        primary_issue = "fragmented_qn_blocks"
    elif batchable_fraction:
        primary_issue = "batchable_qn_blocks"
    else:
        primary_issue = "blocked_decomposition"
    batchable_group_count = int(payload.get("batchable_block_group_count") or 0)
    if batchable_group_count:
        measurement_focus = "qn_block_batching"
    elif int(payload.get("tiny_block_count") or 0):
        measurement_focus = "tiny_qn_blocks"
    elif unique_block_shapes > 1:
        measurement_focus = "shape_fragmentation"
    else:
        measurement_focus = "blocked_decomposition"
    top_block_shape_groups = payload.get("top_block_shape_groups", ()) or ()
    dominant_block_group = top_block_shape_groups[0] if top_block_shape_groups else None
    block_flops = int(payload.get("block_flops_estimate") or 0)
    dense_flops = int(payload.get("dense_flops_estimate") or 0)

    profile = {
        "workload_class": COMPUTE_CLASS_SVD,
        "measurement_scope": "qn_decomposition",
        "precision_level": "qn_block_groups",
        "practical_bucket": payload.get("practical_bucket"),
        "dominant_cost_kind": primary_issue,
        "primary_kernel": payload.get("algorithmic_kernel") or payload.get("kernel_kind"),
        "primary_issue": primary_issue,
        "block_regime": block_regime,
        "cost_model": _practical_cost_model(
            "qn_block_flop_estimate",
            flops_estimate=payload.get("flops_estimate"),
            read_bytes=payload.get("read_bytes"),
            write_bytes=payload.get("write_bytes"),
            extra={
                "block_flops_estimate": block_flops,
                "dense_flops_estimate": dense_flops,
                "block_flop_fraction": _safe_fraction(block_flops, dense_flops),
                "sparse_saved_flop_fraction": sparse_saved,
                "batchable_flop_fraction": batchable_fraction,
            },
        ),
        "dominant_work": {
            "unit": "qn_block_shape_group",
            **(dominant_block_group or {}),
        },
        "parallelization": {
            "unit": "independent_qn_block_groups",
            "batchable_block_group_count": batchable_group_count,
            "batchable_block_count": int(payload.get("batchable_block_count") or 0),
            "block_count": block_count,
            "unique_block_shape_count": unique_block_shapes,
            "shape_fragmentation": payload.get("block_shape_fragmentation"),
        },
        "measurement_limits": _measurement_limits("python_wall_time_if_event_timed"),
        "dominant_operation": dominant_block_group,
        "key_metrics": {
            "block_count": block_count,
            "unique_block_shape_count": unique_block_shapes,
            "qn_block_density": qn_density,
            "sparse_saved_flop_fraction": sparse_saved,
            "batchable_flop_fraction": batchable_fraction,
            "block_shape_reuse_fraction": payload.get("block_shape_reuse_fraction"),
            "tiny_block_count": int(payload.get("tiny_block_count") or 0),
            "skinny_block_count": int(payload.get("skinny_block_count") or 0),
            "dominant_block_shape": payload.get("dominant_block_shape"),
            "dominant_block_shape_count": int(payload.get("dominant_block_shape_count") or 0),
            "rank_sum": int(payload.get("rank_sum") or 0),
            "max_block_rank": int(payload.get("max_block_rank") or 0),
        },
        "optimization_targets": targets,
        "batchable_block_group_count": batchable_group_count,
        "dominant_block_shape_flop_fraction": payload.get("dominant_block_shape_flop_fraction"),
        "top_block_shape_groups": payload.get("top_block_shape_groups", ()),
        "diagnosis": {
            "measurement_focus": measurement_focus,
            "primary_issue": primary_issue,
            "block_regime": block_regime,
            "precision_evidence": {
                "qn_block_density": qn_density,
                "sparse_saved_flop_fraction": sparse_saved,
                "batchable_flop_fraction": batchable_fraction,
                "batchable_block_group_count": batchable_group_count,
                "dominant_block_shape_flop_fraction": payload.get("dominant_block_shape_flop_fraction"),
            },
            "recommended_action": _first_target(targets, "inspect_qn_decomposition"),
        },
    }
    return _event_practical_profile_with_core(payload, profile)


def _counter_from_values(values):
    counts = {}
    for value in values or ():
        key = str(value).upper()
        counts[key] = counts.get(key, 0) + 1
    return counts


def _dominant_count_key(counts):
    if not counts:
        return None
    return max(counts.items(), key=lambda item: (int(item[1]), str(item[0])))[0]


def _oe_path_profile_from_counts(counts):
    counts = {str(key).upper(): int(value) for key, value in (counts or {}).items()}
    step_count = int(sum(counts.values()))
    gemm_steps = int(counts.get("GEMM", 0))
    non_gemm_steps = int(max(step_count - gemm_steps, 0))
    if step_count == 0:
        path_kind = "unknown"
        kernel_kind = "oe_path"
    elif non_gemm_steps == 0:
        path_kind = "gemm_only"
        kernel_kind = "oe_gemm_path"
    elif gemm_steps:
        path_kind = "mixed"
        kernel_kind = "oe_mixed_path"
    else:
        path_kind = "non_gemm"
        kernel_kind = "oe_non_gemm_path"
    return {
        "path_kind": path_kind,
        "kernel_kind": kernel_kind,
        "dominant_contraction_type": _dominant_count_key(counts),
        "gemm_step_count": gemm_steps,
        "non_gemm_step_count": non_gemm_steps,
    }


def _oe_path_profile(path_summary):
    return _oe_path_profile_from_counts(_counter_from_values(path_summary.get("contraction_types", ())))


def _tensordot_layout_hint(left_axes, right_axes, left_ndim, right_ndim):
    contracted_ndim = len(left_axes)
    if contracted_ndim == 0:
        return "outer"
    expected_left = tuple(range(left_ndim - contracted_ndim, left_ndim))
    expected_right = tuple(range(contracted_ndim))
    needs_left = tuple(left_axes) != expected_left
    needs_right = tuple(right_axes) != expected_right
    if needs_left and needs_right:
        return "both_axis_permutation"
    if needs_left:
        return "left_axis_permutation"
    if needs_right:
        return "right_axis_permutation"
    return "reshape_only"


def _normalize_axes(axes, left_ndim, right_ndim):
    if isinstance(axes, int):
        if axes < 0:
            raise ValueError("tensordot axes must be non-negative when given as an integer")
        left_axes = list(range(left_ndim - axes, left_ndim))
        right_axes = list(range(axes))
    else:
        left_axes, right_axes = axes
        if isinstance(left_axes, int):
            left_axes = [left_axes]
        else:
            left_axes = list(left_axes)
        if isinstance(right_axes, int):
            right_axes = [right_axes]
        else:
            right_axes = list(right_axes)

    def normalize(axis, ndim):
        axis = int(axis)
        return axis + ndim if axis < 0 else axis

    return [normalize(axis, left_ndim) for axis in left_axes], [normalize(axis, right_ndim) for axis in right_axes]


def tensordot_compute_payload(left, right, axes, result):
    left_shape = tuple(int(dim) for dim in getattr(left, "shape", ()))
    right_shape = tuple(int(dim) for dim in getattr(right, "shape", ()))
    left_axes, right_axes = _normalize_axes(axes, len(left_shape), len(right_shape))
    left_axis_set = set(left_axes)
    right_axis_set = set(right_axes)
    left_free_axes = tuple(index for index in range(len(left_shape)) if index not in left_axis_set)
    right_free_axes = tuple(index for index in range(len(right_shape)) if index not in right_axis_set)
    left_free_shape = tuple(dim for index, dim in enumerate(left_shape) if index not in left_axis_set)
    right_free_shape = tuple(dim for index, dim in enumerate(right_shape) if index not in right_axis_set)
    left_contract_shape = tuple(left_shape[index] for index in left_axes)
    right_contract_shape = tuple(right_shape[index] for index in right_axes)
    contracted_shape = left_contract_shape if left_contract_shape == right_contract_shape else ()
    contracted_ndim = len(left_axes)
    contracted_size = _prod(contracted_shape) if contracted_shape or contracted_ndim == 0 else 0
    m = _prod(left_free_shape)
    n = _prod(right_free_shape)
    if contracted_ndim == 0:
        k = 0
        contraction_kind = "outer"
        equivalent_gemm = False
        kernel_kind = "outer"
        multiply_count = m * n
        add_count = 0
        flops = multiply_count
    elif left_contract_shape == right_contract_shape:
        k = contracted_size
        contraction_kind = "contract"
        equivalent_gemm = bool(k)
        kernel_kind = _kernel_kind_from_mnk(m, n, k)
        multiply_count = m * n * k if k else 0
        add_count = m * n * max(k - 1, 0) if k else 0
        flops = 2 * m * n * k if k else 0
    else:
        k = 0
        contraction_kind = "invalid"
        equivalent_gemm = False
        kernel_kind = None
        multiply_count = 0
        add_count = 0
        flops = 0
    output_shape = tuple(int(dim) for dim in getattr(result, "shape", ()))
    read_bytes = array_total_bytes((left, right))
    write_bytes = int(getattr(result, "nbytes", 0))
    layout_hint = _tensordot_layout_hint(left_axes, right_axes, len(left_shape), len(right_shape))
    left_permutation_bytes = (
        int(getattr(left, "nbytes", 0))
        if layout_hint in ("left_axis_permutation", "both_axis_permutation")
        else 0
    )
    right_permutation_bytes = (
        int(getattr(right, "nbytes", 0))
        if layout_hint in ("right_axis_permutation", "both_axis_permutation")
        else 0
    )
    axis_permutation_copy_bytes = left_permutation_bytes + right_permutation_bytes
    copy_bytes = axis_permutation_copy_bytes
    memory_bytes = read_bytes + write_bytes
    working_set_bytes = memory_bytes + copy_bytes
    arithmetic_intensity = flops / memory_bytes if memory_bytes else None
    effective_arithmetic_intensity = flops / working_set_bytes if working_set_bytes else None
    tags = []
    bottleneck_hints = []
    if equivalent_gemm:
        tags.append("gemm_like")
        if flops and flops < 1_000_000:
            tags.append("tiny_gemm")
            bottleneck_hints.append("small_gemm")
        if min(m, n, k) <= 4:
            tags.append("skinny_gemm")
            bottleneck_hints.append("skinny_gemm")
    elif contraction_kind == "outer":
        tags.append("outer_product")
    if layout_hint not in ("reshape_only", "outer"):
        tags.append("axis_permutation")
        bottleneck_hints.append("axis_permutation_copy")
    if len(left_free_shape) > 1 or len(right_free_shape) > 1:
        tags.append("high_rank_free")
    if arithmetic_intensity is not None and arithmetic_intensity < 4.0:
        tags.append("low_arithmetic_intensity")
        bottleneck_hints.append("memory_bandwidth")
    if not bottleneck_hints and equivalent_gemm:
        bottleneck_hints.append("compute")
    algorithmic_kernel = kernel_kind or contraction_kind
    problem_signature = "{0}:m={1},n={2},k={3},layout={4}".format(
        algorithmic_kernel,
        int(m),
        int(n),
        int(k),
        layout_hint,
    )
    payload = {
        **compute_payload(
            COMPUTE_CLASS_TENSORDOT,
            "api_tensordot",
            COMPUTE_ROLE_COMPOSITE,
            accounting=COMPUTE_ACCOUNTING_INCLUSIVE,
        ),
        "operation_family": "tensordot",
        "algorithmic_kernel": algorithmic_kernel,
        "problem_kind": f"tensordot:{algorithmic_kernel}",
        "problem_signature": problem_signature,
        "problem_size_bin": _problem_size_bin_from_scale(flops, memory_bytes),
        "input_shapes": [left_shape, right_shape],
        "input_strides": [array_strides(left), array_strides(right)],
        "input_orders": [array_order(left), array_order(right)],
        "input_contiguous": [array_contiguous(left), array_contiguous(right)],
        "input_backends": array_backend_names((left, right)),
        "input_device_kinds": array_device_kinds((left, right)),
        "input_locations": array_locations((left, right)),
        "input_is_host": [array_is_host(array) for array in (left, right) if hasattr(array, "shape")],
        "input_is_device": [array_is_device(array) for array in (left, right) if hasattr(array, "shape")],
        "input_is_distributed": [
            array_is_distributed(array)
            for array in (left, right)
            if hasattr(array, "shape")
        ],
        "output_shape": output_shape,
        "output_strides": array_strides(result),
        "output_order": array_order(result),
        "output_contiguous": array_contiguous(result),
        "output_backend": array_backend_name(result) if hasattr(result, "shape") else None,
        "output_device_kind": array_device_kind(result),
        "output_location": array_location(result) if hasattr(result, "shape") else None,
        "output_is_host": array_is_host(result) if hasattr(result, "shape") else None,
        "output_is_device": array_is_device(result) if hasattr(result, "shape") else None,
        "output_is_distributed": array_is_distributed(result) if hasattr(result, "shape") else None,
        "input_dtypes": array_dtype_names((left, right)),
        "output_dtype": str(getattr(result, "dtype", None)),
        "left_free_axes": left_free_axes,
        "right_free_axes": right_free_axes,
        "left_contract_axes": tuple(left_axes),
        "right_contract_axes": tuple(right_axes),
        "left_free_ndim": len(left_free_shape),
        "right_free_ndim": len(right_free_shape),
        "contracted_rank": int(contracted_ndim),
        "output_rank": len(output_shape),
        "left_free_shape": left_free_shape,
        "right_free_shape": right_free_shape,
        "contracted_shape": contracted_shape,
        "contraction_kind": contraction_kind,
        "contracted_ndim": int(contracted_ndim),
        "contracted_size": int(contracted_size),
        "left_free_size": int(m),
        "right_free_size": int(n),
        "output_elements": int(_prod(output_shape)),
        "m": int(m),
        "n": int(n),
        "k": int(k),
        "equivalent_gemm": equivalent_gemm,
        "kernel_kind": kernel_kind,
        "layout_hint": layout_hint,
        "requires_axis_permutation": layout_hint not in ("reshape_only", "outer"),
        "left_permutation_bytes": int(left_permutation_bytes),
        "right_permutation_bytes": int(right_permutation_bytes),
        "axis_permutation_copy_bytes": int(axis_permutation_copy_bytes),
        "copy_bytes": int(copy_bytes),
        "working_set_bytes_estimate": int(working_set_bytes),
        "cost_model": "single_{0}".format(algorithmic_kernel),
        "bottleneck_hints": _profile_tags(bottleneck_hints),
        "profile_tags": _profile_tags(tags),
        "num_gemm": 1 if equivalent_gemm and kernel_kind in ("gemm", "gemv", "dot") else 0,
        "multiply_count": int(multiply_count),
        "add_count": int(add_count),
        "flops_estimate": int(flops),
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "memory_bytes_estimate": int(memory_bytes),
        "arithmetic_intensity_estimate": arithmetic_intensity,
        "effective_arithmetic_intensity_estimate": effective_arithmetic_intensity,
        "arithmetic_intensity_bin": _arithmetic_intensity_bin(arithmetic_intensity),
    }
    payload["practical_bucket"] = _tensordot_practical_bucket(payload)
    payload["kernel_observation"] = _tensordot_kernel_observation(
        payload,
        input_shapes=(left_shape, right_shape),
        output_shape=output_shape,
    )
    payload["practical_profile"] = _tensordot_event_practical_profile(payload)
    payload["compute_profile"] = _event_compute_profile(payload)
    payload["core_operation"] = _core_operation_index(payload)
    return payload


def device_payload(device):
    return {
        "kind": getattr(device, "kind", None),
        "index": getattr(device, "index", None),
        "local_rank": getattr(device, "local_rank", None),
        "global_rank": getattr(device, "global_rank", None),
        "visible_id": getattr(device, "visible_id", None),
    }


def device_execution_payload(device):
    payload = device_payload(device)
    return {
        "device": str(device),
        "device_info": payload,
        "device_kind": payload["kind"],
        "device_index": payload["index"],
    }


def array_operand_payload(backend, name, array, modes):
    info = backend.array_info(array)
    return {
        "name": name,
        "modes": [str(mode) for mode in modes],
        "shape": info.shape,
        "dtype": str(info.dtype),
        "itemsize": info.itemsize,
        "size": info.size,
        "nbytes": info.nbytes,
        "ndim": info.ndim,
        "strides": info.strides,
        "order": info.order,
        "contiguous": info.contiguous,
        "writeable": info.writeable,
        "owns_data": info.owns_data,
        "backend": info.backend_name,
        "device": str(info.device),
        "device_kind": getattr(info.device, "kind", None),
        "device_index": getattr(info.device, "index", None),
        "is_host": info.is_host,
        "is_device": info.is_device,
        "is_distributed": info.is_distributed,
    }


def operand_shape(value):
    if hasattr(value, "shape"):
        return tuple(value.shape)
    if isinstance(value, (list, tuple)):
        try:
            return tuple(int(dim) for dim in value)
        except (TypeError, ValueError):
            return None
    return None


def json_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def expression_contraction_list_summary(expr):
    contraction_list = getattr(expr, "contraction_list", None) or []
    return {
        "path": [
            [int(index) for index in contraction[0]]
            for contraction in contraction_list
        ],
        "contraction_count": len(contraction_list),
        "contraction_types": [
            str(contraction[4])
            for contraction in contraction_list
            if len(contraction) > 4
        ],
        "contraction_steps": [],
        "flop_count": None,
        "largest_intermediate": None,
    }


def _einsum_shape_map(equation, shapes):
    if "->" not in equation or "..." in equation:
        return None, None
    input_text, output_modes = equation.split("->", 1)
    input_modes = [modes.strip() for modes in input_text.split(",")]
    output_modes = output_modes.strip()
    if len(input_modes) != len(shapes):
        return None, None
    mode_sizes = {}
    for modes, shape in zip(input_modes, shapes):
        if len(modes) != len(shape):
            return None, None
        for mode, size in zip(modes, shape):
            previous = mode_sizes.setdefault(mode, int(size))
            if previous != int(size):
                return None, None
    return mode_sizes, output_modes


def _shape_for_modes(modes, mode_sizes):
    shape = []
    for mode in modes:
        if mode not in mode_sizes:
            return None
        shape.append(int(mode_sizes[mode]))
    return shape


def _build_contraction_steps(equation, shapes, contraction_list, path, path_info):
    mode_sizes, _ = _einsum_shape_map(equation, shapes)
    if mode_sizes is None:
        return []

    scale_list = list(getattr(path_info, "scale_list", []) or [])
    size_list = list(getattr(path_info, "size_list", []) or [])
    steps = []
    for index, contraction in enumerate(contraction_list):
        if len(contraction) < 4:
            continue
        operand_positions, contracted_modes, step_equation, remaining_modes = contraction[:4]
        if "->" not in step_equation:
            continue
        step_input_text, step_output_modes = step_equation.split("->", 1)
        step_input_modes = [modes.strip() for modes in step_input_text.split(",")]
        step_output_modes = step_output_modes.strip()
        input_shapes = [_shape_for_modes(modes, mode_sizes) for modes in step_input_modes]
        output_shape = _shape_for_modes(step_output_modes, mode_sizes)
        remaining_shapes = [_shape_for_modes(modes, mode_sizes) for modes in remaining_modes]
        if output_shape is None or any(shape is None for shape in input_shapes + remaining_shapes):
            continue
        path_item = path[index] if index < len(path) else operand_positions
        step = {
            "step": int(index),
            "path": [int(item) for item in path_item],
            "operand_positions": [int(item) for item in operand_positions],
            "input_modes": [str(modes) for modes in step_input_modes],
            "input_shapes": input_shapes,
            "output_modes": str(step_output_modes),
            "output_shape": output_shape,
            "remaining_modes": [str(modes) for modes in remaining_modes],
            "remaining_shapes": remaining_shapes,
            "contracted_modes": [str(mode) for mode in sorted(contracted_modes)],
        }
        if len(contraction) > 4:
            step["contraction_type"] = str(contraction[4])
        if index < len(scale_list):
            step["scaling"] = json_int(scale_list[index])
        if index < len(size_list):
            step["size"] = json_int(size_list[index])
        steps.append(step)
    return steps


def contract_expression_path_summary(contract_path, args, kwargs, expr):
    summary = expression_contraction_list_summary(expr)
    if not args or not isinstance(args[0], str):
        return summary

    shapes = []
    for operand in args[1:]:
        shape = operand_shape(operand)
        if shape is None:
            return summary
        shapes.append(shape)

    path_kwargs = {"shapes": True, "optimize": kwargs.get("optimize")}
    if "memory_limit" in kwargs:
        path_kwargs["memory_limit"] = kwargs["memory_limit"]
    try:
        path, path_info = contract_path(args[0], *shapes, **path_kwargs)
    except Exception:
        return summary

    summary["path"] = [[int(index) for index in item] for item in path]
    summary["flop_count"] = json_int(getattr(path_info, "opt_cost", None))
    summary["largest_intermediate"] = json_int(getattr(path_info, "largest_intermediate", None))
    summary["contraction_steps"] = _build_contraction_steps(
        args[0],
        shapes,
        getattr(path_info, "contraction_list", None) or getattr(expr, "contraction_list", None) or [],
        summary["path"],
        path_info,
    )
    return summary


def contract_path_summary(contract_path, args, kwargs):
    summary = {
        "path": [],
        "contraction_count": 0,
        "contraction_types": [],
        "contraction_steps": [],
        "flop_count": None,
        "largest_intermediate": None,
    }
    if not args or not isinstance(args[0], str):
        return summary

    shapes = []
    for operand in args[1:]:
        shape = operand_shape(operand)
        if shape is None:
            return summary
        shapes.append(shape)

    path_kwargs = {"shapes": True, "optimize": kwargs.get("optimize")}
    if "memory_limit" in kwargs:
        path_kwargs["memory_limit"] = kwargs["memory_limit"]
    try:
        path, path_info = contract_path(args[0], *shapes, **path_kwargs)
    except Exception:
        return summary

    contraction_list = getattr(path_info, "contraction_list", None) or []
    summary["path"] = [[int(index) for index in item] for item in path]
    summary["contraction_count"] = len(contraction_list)
    summary["contraction_types"] = [
        str(contraction[4])
        for contraction in contraction_list
        if len(contraction) > 4
    ]
    summary["flop_count"] = json_int(getattr(path_info, "opt_cost", None))
    summary["largest_intermediate"] = json_int(getattr(path_info, "largest_intermediate", None))
    summary["contraction_steps"] = _build_contraction_steps(
        args[0],
        shapes,
        contraction_list,
        summary["path"],
        path_info,
    )
    return summary


def _oe_step_flops_estimate(step):
    explicit = json_int(step.get("flops_estimate"))
    if explicit is not None:
        return int(explicit)

    input_modes = step.get("input_modes") or ()
    input_shapes = step.get("input_shapes") or ()
    if len(input_modes) != len(input_shapes):
        return 0

    mode_sizes = {}
    ordered_modes = []
    for modes, shape in zip(input_modes, input_shapes):
        if isinstance(modes, str):
            modes = list(modes)
        else:
            modes = [str(mode) for mode in modes]
        shape = _shape_tuple(shape)
        if shape is None or len(modes) != len(shape):
            return 0
        for mode, size in zip(modes, shape):
            size = int(size)
            previous = mode_sizes.setdefault(mode, size)
            if previous != size:
                return 0
            if mode not in ordered_modes:
                ordered_modes.append(mode)

    if not ordered_modes:
        return 0
    operation_size = _prod(mode_sizes[mode] for mode in ordered_modes)
    if not operation_size:
        return 0
    contracted_modes = step.get("contracted_modes") or ()
    return int(2 * operation_size if contracted_modes else operation_size)


def _oe_path_cost_model(gemm_flops, non_gemm_flops):
    total = int(gemm_flops or 0) + int(non_gemm_flops or 0)
    if total <= 0:
        return "unknown"
    if non_gemm_flops <= 0:
        return "gemm_only"
    if gemm_flops <= 0:
        return "non_gemm_only"
    non_gemm_fraction = non_gemm_flops / total
    if non_gemm_fraction >= 0.5:
        return "non_gemm_flop_dominant"
    if non_gemm_fraction >= 0.1:
        return "mixed_flop_path"
    return "gemm_flop_dominant"


def _oe_step_bucket_key(step):
    contraction_type = str(step.get("contraction_type", "UNKNOWN")).upper()
    input_shape_key = "+".join(
        _shape_key_or_unknown(shape)
        for shape in step.get("input_shapes", ())
    )
    if not input_shape_key:
        input_shape_key = "unknown"
    output_shape = _shape_key_or_unknown(step.get("output_shape"))
    return {
        "signature": "{0}:{1}->{2}".format(contraction_type, input_shape_key, output_shape),
        "contraction_type": contraction_type,
        "input_shape_key": input_shape_key,
        "output_shape": output_shape,
    }


def _oe_step_buckets(steps):
    buckets = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        key = _oe_step_bucket_key(step)
        bucket = buckets.get(key["signature"])
        if bucket is None:
            bucket = {
                **key,
                "count": 0,
                "total_flops_estimate": 0,
                "total_output_elements": 0,
                "step_indices": [],
            }
            buckets[key["signature"]] = bucket
        bucket["count"] += 1
        bucket["total_flops_estimate"] += int(_oe_step_flops_estimate(step))
        bucket["total_output_elements"] += int(_shape_elements(step.get("output_shape")))
        step_index = json_int(step.get("step"))
        if step_index is not None and len(bucket["step_indices"]) < 8:
            bucket["step_indices"].append(step_index)

    return sorted(
        buckets.values(),
        key=lambda item: (
            int(item["total_flops_estimate"]),
            int(item["count"]),
            str(item["signature"]),
        ),
        reverse=True,
    )


def _oe_path_type_sequence(path_summary, step_type_counts):
    steps = path_summary.get("contraction_steps") or ()
    sequence = [
        str(step.get("contraction_type", "UNKNOWN")).upper()
        for step in steps
        if isinstance(step, dict)
    ]
    if sequence:
        return sequence
    types = path_summary.get("contraction_types") or ()
    sequence = [str(item).upper() for item in types]
    if sequence:
        return sequence
    return [
        str(key).upper()
        for key, count in sorted((step_type_counts or {}).items())
        for _ in range(int(count))
    ]


def _oe_practical_bucket(payload, contraction_count=None):
    kernel = payload.get("algorithmic_kernel") or payload.get("kernel_kind") or "unknown"
    type_sequence = payload.get("path_type_sequence") or ()
    type_signature = ">".join(str(item) for item in type_sequence) or "unknown"
    contraction_count = int(
        contraction_count
        if contraction_count is not None
        else payload.get("contraction_count") or 0
    )
    dominant_bucket = payload.get("dominant_step_bucket") or {}
    dominant_signature = dominant_bucket.get("signature") or "unknown"
    largest = int(payload.get("largest_intermediate_elements") or 0)
    aggregation_key = (
        "oe|kernel={0}|types={1}|steps={2}|dominant={3}".format(
            kernel,
            type_signature,
            contraction_count,
            dominant_signature,
        )
    )
    return {
        "schema": PRACTICAL_BUCKET_SCHEMA,
        "family": "oe_contract",
        "kernel": kernel,
        "precision_level": "path_step_signature",
        "measurement_unit": "path_call",
        "parallel_unit": "sequential_path_step",
        "contraction_count": contraction_count,
        "shape_signature": "types={0},steps={1},dominant={2}".format(
            type_signature,
            contraction_count,
            dominant_signature,
        ),
        "path_type_signature": type_signature,
        "dominant_step_signature": dominant_signature,
        "comparison_key": "oe_path:types={0};steps={1};largest={2}".format(
            type_signature,
            contraction_count,
            largest,
        ),
        "aggregation_key": aggregation_key,
        "comparison_axes": [
            "wall_s",
            "path_step_wall_time",
            "path_step_flops",
            "largest_intermediate_elements",
        ],
        "step_buckets": list(payload.get("step_buckets") or ())[:8],
    }


def _oe_step_profile(path_summary):
    steps = path_summary.get("contraction_steps") or []
    step_type_counts = {}
    output_shape_counts = {}
    total_output_elements = 0
    max_output_elements = 0
    non_gemm_steps = []
    step_costs = []
    gemm_flops = 0
    tensordot_flops = 0
    generic_einsum_flops = 0
    non_gemm_flops = 0
    dominant_step = None

    for step in steps:
        if not isinstance(step, dict):
            continue
        contraction_type = str(step.get("contraction_type", "UNKNOWN")).upper()
        step_type_counts[contraction_type] = step_type_counts.get(contraction_type, 0) + 1
        output_shape = _shape_tuple(step.get("output_shape"))
        output_elements = _shape_elements(output_shape)
        if output_shape is not None:
            key = _shape_key(output_shape)
            output_shape_counts[key] = output_shape_counts.get(key, 0) + 1
        total_output_elements += output_elements
        max_output_elements = max(max_output_elements, output_elements)
        step_flops = _oe_step_flops_estimate(step)
        step_cost = {
            "step": json_int(step.get("step")),
            "contraction_type": contraction_type,
            "flops_estimate": int(step_flops),
            "output_elements": int(output_elements),
        }
        step_costs.append(step_cost)
        if dominant_step is None or step_flops > dominant_step["flops_estimate"]:
            dominant_step = step_cost
        if contraction_type == "GEMM":
            gemm_flops += step_flops
            continue
        if contraction_type in _OE_TENSORDOT_STEP_TYPES:
            tensordot_flops += step_flops
        else:
            generic_einsum_flops += step_flops
        non_gemm_flops += step_flops
        non_gemm_steps.append({
            "step": json_int(step.get("step")),
            "contraction_type": contraction_type,
            "input_shapes": [
                _shape_tuple(shape)
                for shape in step.get("input_shapes", ())
            ],
            "output_shape": output_shape,
            "output_elements": int(output_elements),
            "size": json_int(step.get("size")),
            "scaling": json_int(step.get("scaling")),
        })

    if not step_type_counts:
        step_type_counts = _counter_from_values(path_summary.get("contraction_types", ()))

    tensordot_step_count = sum(
        int(step_type_counts.get(step_type, 0))
        for step_type in _OE_TENSORDOT_STEP_TYPES
    )
    gemm_step_count = int(step_type_counts.get("GEMM", 0))
    generic_einsum_step_count = max(
        int(sum(step_type_counts.values())) - gemm_step_count - tensordot_step_count,
        0,
    )

    def step_sort_key(item):
        step = item.get("step")
        return (
            -int(item.get("flops_estimate") or 0),
            int(step) if step is not None else 0,
        )

    top_step_costs = sorted(step_costs, key=step_sort_key)[:8]
    step_buckets = _oe_step_buckets(steps)
    dominant_step_bucket = step_buckets[0] if step_buckets else None
    path_type_sequence = _oe_path_type_sequence(path_summary, step_type_counts)
    tags = []
    path_profile = _oe_path_profile_from_counts(step_type_counts)
    if path_profile["path_kind"] != "unknown":
        tags.append("{0}_path".format(path_profile["path_kind"]))
    if non_gemm_steps:
        tags.append("non_gemm_steps")
    path_cost_model = _oe_path_cost_model(gemm_flops, non_gemm_flops)
    if path_cost_model != "unknown":
        tags.append(path_cost_model)
    if len(output_shape_counts) > 1:
        tags.append("shape_diverse")
    if max_output_elements and path_summary.get("largest_intermediate"):
        largest = json_int(path_summary.get("largest_intermediate")) or 0
        if largest > max_output_elements:
            tags.append("large_intermediate")

    return {
        "step_type_counts": step_type_counts,
        "step_output_shape_counts": output_shape_counts,
        "unique_step_output_shape_count": len(output_shape_counts),
        "max_step_output_elements": int(max_output_elements),
        "total_step_output_elements": int(total_output_elements),
        "non_gemm_steps": non_gemm_steps[:8],
        "path_type_sequence": path_type_sequence,
        "step_buckets": step_buckets[:16],
        "dominant_step_bucket": dominant_step_bucket,
        "step_costs": step_costs[:16],
        "dropped_step_cost_count": max(len(step_costs) - 16, 0),
        "dominant_step": top_step_costs[0] if top_step_costs else None,
        "top_step_costs": top_step_costs,
        "gemm_flops_estimate": int(gemm_flops),
        "tensordot_step_count": int(tensordot_step_count),
        "generic_einsum_step_count": int(generic_einsum_step_count),
        "tensordot_flops_estimate": int(tensordot_flops),
        "generic_einsum_flops_estimate": int(generic_einsum_flops),
        "non_gemm_flops_estimate": int(non_gemm_flops),
        "dominant_step_type": dominant_step["contraction_type"] if dominant_step is not None else None,
        "dominant_step_flops_estimate": int(dominant_step["flops_estimate"]) if dominant_step is not None else 0,
        "path_cost_model": path_cost_model,
        "profile_tags": _profile_tags(tags),
    }


def oe_compute_payload(subclass, operands, result, path_summary=None):
    path_summary = path_summary or {}
    largest_intermediate_elements = json_int(path_summary.get("largest_intermediate"))
    largest_intermediate_bytes = None
    if largest_intermediate_elements is not None:
        largest_intermediate_bytes = int(largest_intermediate_elements * array_itemsize(result))
    contraction_type_counts = _counter_from_values(path_summary.get("contraction_types", ()))
    step_profile = _oe_step_profile(path_summary)
    if not contraction_type_counts:
        contraction_type_counts = step_profile["step_type_counts"]
    path_profile = _oe_path_profile_from_counts(contraction_type_counts)
    contraction_count = json_int(path_summary.get("contraction_count")) or int(sum(contraction_type_counts.values()))
    gemm_steps = int(path_profile["gemm_step_count"])
    non_gemm_steps = int(path_profile["non_gemm_step_count"])
    output_elements = _shape_elements(getattr(result, "shape", ()))
    largest_to_output_ratio = (
        largest_intermediate_elements / output_elements
        if largest_intermediate_elements is not None and output_elements
        else None
    )
    read_bytes = array_total_bytes(operands)
    write_bytes = int(getattr(result, "nbytes", 0))
    flops = json_int(path_summary.get("flop_count")) or 0
    step_flops = step_profile["gemm_flops_estimate"] + step_profile["non_gemm_flops_estimate"]
    step_costs_available = bool(step_profile["step_costs"])
    flop_split_available = bool(step_flops)
    flop_denominator = flops or step_flops

    def step_flop_fraction(value):
        return value / flop_denominator if flop_split_available and flop_denominator else None

    bottleneck_hints = ["oe_path"]
    if path_profile["path_kind"] != "unknown":
        bottleneck_hints.append("oe_{0}_path".format(path_profile["path_kind"]))
    if non_gemm_steps:
        bottleneck_hints.append("oe_non_gemm_steps")
    if step_profile["non_gemm_flops_estimate"]:
        bottleneck_hints.append("oe_non_gemm_flops")
    if largest_to_output_ratio is not None and largest_to_output_ratio >= 8:
        bottleneck_hints.append("large_intermediate")
    payload = {
        **compute_payload(COMPUTE_CLASS_OE, subclass, COMPUTE_ROLE_COMPOSITE),
        "operation_family": "oe_contract",
        "algorithmic_kernel": path_profile["kernel_kind"],
        "problem_kind": "oe:{0}".format(path_profile["path_kind"]),
        "problem_signature": "{0}:steps={1},gemm={2},non_gemm={3},largest={4}".format(
            path_profile["kernel_kind"],
            int(contraction_count),
            gemm_steps,
            non_gemm_steps,
            int(largest_intermediate_elements or 0),
        ),
        "problem_size_bin": _problem_size_bin_from_scale(flops, read_bytes + write_bytes),
        "input_shapes": array_shapes(operands),
        "input_strides": [array_strides(array) for array in operands if hasattr(array, "shape")],
        "input_orders": [array_order(array) for array in operands if hasattr(array, "shape")],
        "input_contiguous": [array_contiguous(array) for array in operands if hasattr(array, "shape")],
        "input_backends": array_backend_names(operands),
        "input_device_kinds": array_device_kinds(operands),
        "input_locations": array_locations(operands),
        "input_is_host": [array_is_host(array) for array in operands if hasattr(array, "shape")],
        "input_is_device": [array_is_device(array) for array in operands if hasattr(array, "shape")],
        "input_is_distributed": [array_is_distributed(array) for array in operands if hasattr(array, "shape")],
        "output_shape": tuple(getattr(result, "shape", ())),
        "output_strides": array_strides(result),
        "output_order": array_order(result),
        "output_contiguous": array_contiguous(result),
        "output_backend": array_backend_name(result) if hasattr(result, "shape") else None,
        "output_device_kind": array_device_kind(result),
        "output_location": array_location(result) if hasattr(result, "shape") else None,
        "output_is_host": array_is_host(result) if hasattr(result, "shape") else None,
        "output_is_device": array_is_device(result) if hasattr(result, "shape") else None,
        "output_is_distributed": array_is_distributed(result) if hasattr(result, "shape") else None,
        "input_dtypes": array_dtype_names(operands),
        "output_dtype": str(getattr(result, "dtype", None)),
        **path_profile,
        "contraction_type_counts": contraction_type_counts,
        "step_type_counts": step_profile["step_type_counts"],
        "step_output_shape_counts": step_profile["step_output_shape_counts"],
        "unique_step_output_shape_count": step_profile["unique_step_output_shape_count"],
        "max_step_output_elements": step_profile["max_step_output_elements"],
        "total_step_output_elements": step_profile["total_step_output_elements"],
        "non_gemm_steps": step_profile["non_gemm_steps"],
        "path_type_sequence": step_profile["path_type_sequence"],
        "step_buckets": step_profile["step_buckets"],
        "dominant_step_bucket": step_profile["dominant_step_bucket"],
        "step_costs": step_profile["step_costs"],
        "dropped_step_cost_count": step_profile["dropped_step_cost_count"],
        "dominant_step": step_profile["dominant_step"],
        "top_step_costs": step_profile["top_step_costs"],
        "gemm_flops_estimate": step_profile["gemm_flops_estimate"],
        "tensordot_step_count": step_profile["tensordot_step_count"],
        "generic_einsum_step_count": step_profile["generic_einsum_step_count"],
        "tensordot_flops_estimate": step_profile["tensordot_flops_estimate"],
        "generic_einsum_flops_estimate": step_profile["generic_einsum_flops_estimate"],
        "non_gemm_flops_estimate": step_profile["non_gemm_flops_estimate"],
        "step_costs_available": step_costs_available,
        "flop_split_available": flop_split_available,
        "gemm_flop_fraction": step_flop_fraction(step_profile["gemm_flops_estimate"]),
        "tensordot_flop_fraction": step_flop_fraction(step_profile["tensordot_flops_estimate"]),
        "generic_einsum_flop_fraction": step_flop_fraction(step_profile["generic_einsum_flops_estimate"]),
        "non_gemm_flop_fraction": step_flop_fraction(step_profile["non_gemm_flops_estimate"]),
        "dominant_step_type": step_profile["dominant_step_type"],
        "dominant_step_flops_estimate": step_profile["dominant_step_flops_estimate"],
        "path_cost_model": step_profile["path_cost_model"],
        "gemm_step_fraction": gemm_steps / contraction_count if contraction_count else None,
        "tensordot_step_fraction": (
            step_profile["tensordot_step_count"] / contraction_count
            if contraction_count else None
        ),
        "generic_einsum_step_fraction": (
            step_profile["generic_einsum_step_count"] / contraction_count
            if contraction_count else None
        ),
        "non_gemm_step_fraction": non_gemm_steps / contraction_count if contraction_count else None,
        "largest_intermediate_to_output_ratio": largest_to_output_ratio,
        "bottleneck_hints": _profile_tags(bottleneck_hints),
        "profile_tags": _profile_tags(step_profile["profile_tags"]),
        "flops_estimate": flops,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "memory_bytes_estimate": int(read_bytes + write_bytes),
        "largest_intermediate_elements": largest_intermediate_elements,
        "largest_intermediate_bytes": largest_intermediate_bytes,
        "peak_bytes": largest_intermediate_bytes,
        "workspace_bytes": largest_intermediate_bytes,
    }
    payload["practical_bucket"] = _oe_practical_bucket(payload, contraction_count=contraction_count)
    payload["kernel_observation"] = _kernel_observation(payload)
    payload["practical_profile"] = _oe_event_practical_profile(payload, contraction_count=contraction_count)
    payload["compute_profile"] = _event_compute_profile(payload)
    payload["core_operation"] = _core_operation_index(payload)
    return payload


def qn_to_profile_list(qn):
    if hasattr(qn, "tolist"):
        qn = qn.tolist()
    if not isinstance(qn, (list, tuple)):
        qn = [qn]
    result = []
    for value in qn:
        if hasattr(value, "item"):
            value = value.item()
        result.append(int(value))
    return result


def svd_qn_block_payload(nl, nr, lset, rset, block, rank):
    return {
        "left_qn": qn_to_profile_list(nl),
        "right_qn": qn_to_profile_list(nr),
        "left_size": int(len(lset)),
        "right_size": int(len(rset)),
        "block_shape": tuple(block.shape),
        "rank": int(rank),
    }


def decomposition_flops_estimate(shape, mode):
    if len(shape) != 2:
        return 0
    m, n = (int(shape[0]), int(shape[1]))
    r = min(m, n)
    if r <= 0:
        return 0
    upper_mode = str(mode).upper()
    if upper_mode == "QR":
        return int(max(0, 2 * m * n * r - (2 * r * r * r) // 3))
    if upper_mode == "EIGH":
        return int((10 * r * r * r) // 3)
    return int(4 * m * n * r + (8 * r * r * r) // 3)


def _svd_block_shape_groups(blocks, block_shapes, block_flops_by_shape, total_block_flops):
    groups = {}
    for block, shape, flops in zip(blocks, block_shapes, block_flops_by_shape):
        key = _shape_key(shape)
        item = groups.get(key)
        if item is None:
            item = {
                "shape": key,
                "count": 0,
                "rank_sum": 0,
                "elements": int(_prod(shape)),
                "flops_per_block_estimate": int(flops),
                "total_flops_estimate": 0,
            }
            groups[key] = item
        item["count"] += 1
        item["total_flops_estimate"] += int(flops)
        if isinstance(block, dict) and block.get("rank") is not None:
            item["rank_sum"] += int(block.get("rank"))

    result = []
    for item in groups.values():
        item = dict(item)
        item["flop_fraction"] = (
            item["total_flops_estimate"] / total_block_flops
            if total_block_flops
            else None
        )
        item["batchable"] = item["count"] >= 2
        result.append(item)
    return sorted(
        result,
        key=lambda item: (
            int(item["total_flops_estimate"]),
            int(item["count"]),
            str(item["shape"]),
        ),
        reverse=True,
    )


def _svd_block_group_signature(block_shape_groups):
    parts = []
    for group in block_shape_groups or ():
        shape = group.get("shape")
        count = int(group.get("count") or 0)
        if shape is not None and count:
            parts.append("{0}:{1}".format(shape, count))
    return ",".join(parts) or "none"


def _svd_practical_bucket(payload, matrix_shape=None):
    kernel = payload.get("algorithmic_kernel") or payload.get("kernel_kind") or "unknown"
    matrix_shape = _shape_key_or_unknown(
        matrix_shape
        if matrix_shape is not None
        else payload.get("matrix_shape")
    )
    block_group_signature = payload.get("block_group_signature") or "none"
    aggregation_key = "svd|kernel={0}|matrix={1}|groups={2}".format(
        kernel,
        matrix_shape,
        block_group_signature,
    )
    return {
        "schema": PRACTICAL_BUCKET_SCHEMA,
        "family": "svd",
        "kernel": kernel,
        "precision_level": "qn_block_shape_groups",
        "measurement_unit": "decomposition_call",
        "parallel_unit": "qn_block_shape_group",
        "shape_signature": "matrix={0},groups={1}".format(matrix_shape, block_group_signature),
        "matrix_shape": matrix_shape,
        "block_group_signature": block_group_signature,
        "comparison_key": "svd_qn:kernel={0};matrix={1};groups={2}".format(
            kernel,
            matrix_shape,
            block_group_signature,
        ),
        "aggregation_key": aggregation_key,
        "comparison_axes": [
            "wall_s",
            "block_shape_wall_time",
            "block_shape_flops",
            "decomposition_kernel_time",
        ],
        "block_shape_groups": list(payload.get("top_block_shape_groups") or ())[:8],
    }


def svd_qn_compute_payload(mode, coef_array, coef_matrix, blocks, outputs):
    block_records = [
        block
        for block in (blocks or [])
        if block.get("block_shape") is not None
    ]
    block_shapes = [
        tuple(block.get("block_shape", ()))
        for block in block_records
    ]
    block_elements = [_prod(shape) for shape in block_shapes]
    block_shape_counts = {}
    for shape in block_shapes:
        key = _shape_key(shape)
        block_shape_counts[key] = block_shape_counts.get(key, 0) + 1
    block_count = len(block_shapes)
    dominant_block_shape = _dominant_count_key(block_shape_counts)
    dominant_block_shape_count = (
        int(block_shape_counts.get(dominant_block_shape, 0))
        if dominant_block_shape is not None
        else 0
    )
    batchable_block_count = sum(
        count
        for count in block_shape_counts.values()
        if int(count) >= 2
    )
    block_flops_by_shape = [
        decomposition_flops_estimate(shape, mode)
        for shape in block_shapes
    ]
    total_block_flops = sum(block_flops_by_shape)
    block_shape_groups = _svd_block_shape_groups(
        block_records,
        block_shapes,
        block_flops_by_shape,
        total_block_flops,
    )
    block_group_signature = _svd_block_group_signature(block_shape_groups)
    batchable_block_group_count = sum(1 for group in block_shape_groups if group["batchable"])
    dominant_block_shape_flop_fraction = (
        block_shape_groups[0]["flop_fraction"]
        if block_shape_groups
        else None
    )
    batchable_flops = sum(
        flops
        for shape, flops in zip(block_shapes, block_flops_by_shape)
        if block_shape_counts.get(_shape_key(shape), 0) >= 2
    )
    output_arrays = tuple(array for array in outputs if hasattr(array, "shape"))
    matrix_shape = tuple(getattr(coef_matrix, "shape", ()))
    matrix_elements = _prod(matrix_shape) if matrix_shape else 0
    total_block_elements = sum(block_elements)
    max_block_elements = max(block_elements, default=0)
    aspect_ratios = []
    for shape in block_shapes:
        if len(shape) < 2:
            continue
        small = min(int(shape[0]), int(shape[1]))
        large = max(int(shape[0]), int(shape[1]))
        if small:
            aspect_ratios.append(large / small)
    ranks = [
        int(block.get("rank"))
        for block in block_records
        if isinstance(block, dict) and block.get("rank") is not None
    ]
    tiny_block_count = sum(1 for elements in block_elements if elements < 4096)
    skinny_block_count = 0
    for shape in block_shapes:
        if len(shape) < 2:
            continue
        small = min(int(shape[0]), int(shape[1]))
        large = max(int(shape[0]), int(shape[1]))
        if small and large / small >= 4:
            skinny_block_count += 1
    tags = []
    if tiny_block_count:
        tags.append("tiny_svd_blocks")
    if skinny_block_count:
        tags.append("skinny_svd_blocks")
    if dominant_block_shape_count > 1:
        tags.append("reused_block_shapes")
    if batchable_block_count:
        tags.append("batchable_svd_blocks")
    if len(block_shape_counts) > 1:
        tags.append("shape_diverse")
    if matrix_elements and total_block_elements < matrix_elements:
        tags.append("qn_sparse_blocks")
    kernel_kind = _decomposition_subclass(mode)
    read_bytes = int(getattr(coef_array, "nbytes", 0))
    write_bytes = array_total_bytes(output_arrays)
    dense_flops = decomposition_flops_estimate(matrix_shape, mode)
    flops = int(total_block_flops or dense_flops)
    largest_block_flops = max(block_flops_by_shape, default=0)
    bottleneck_hints = list(tags)
    if block_count:
        bottleneck_hints.append("blocked_decomposition")
    if dense_flops and total_block_flops and total_block_flops < dense_flops:
        bottleneck_hints.append("qn_sparse_blocks")
    payload = {
        **compute_payload(COMPUTE_CLASS_SVD, kernel_kind, COMPUTE_ROLE_KERNEL),
        "operation_family": "decomposition",
        "algorithmic_kernel": kernel_kind,
        "problem_kind": "svd:qn_blocks",
        "problem_signature": "{0}:blocks={1},unique={2},dominant={3},rank={4}".format(
            kernel_kind,
            block_count,
            len(block_shape_counts),
            dominant_block_shape,
            int(sum(ranks)),
        ),
        "problem_size_bin": _problem_size_bin_from_scale(flops, read_bytes + write_bytes),
        "input_shapes": array_shapes((coef_array, coef_matrix)),
        "input_strides": [array_strides(array) for array in (coef_array, coef_matrix) if hasattr(array, "shape")],
        "input_orders": [array_order(array) for array in (coef_array, coef_matrix) if hasattr(array, "shape")],
        "input_contiguous": [array_contiguous(array) for array in (coef_array, coef_matrix) if hasattr(array, "shape")],
        "input_backends": array_backend_names((coef_array, coef_matrix)),
        "input_device_kinds": array_device_kinds((coef_array, coef_matrix)),
        "input_locations": array_locations((coef_array, coef_matrix)),
        "input_is_host": [array_is_host(array) for array in (coef_array, coef_matrix) if hasattr(array, "shape")],
        "input_is_device": [array_is_device(array) for array in (coef_array, coef_matrix) if hasattr(array, "shape")],
        "input_is_distributed": [
            array_is_distributed(array)
            for array in (coef_array, coef_matrix)
            if hasattr(array, "shape")
        ],
        "matrix_shape": matrix_shape,
        "matrix_strides": array_strides(coef_matrix),
        "matrix_order": array_order(coef_matrix),
        "matrix_contiguous": array_contiguous(coef_matrix),
        "matrix_backend": array_backend_name(coef_matrix) if hasattr(coef_matrix, "shape") else None,
        "matrix_device_kind": array_device_kind(coef_matrix),
        "matrix_location": array_location(coef_matrix) if hasattr(coef_matrix, "shape") else None,
        "output_shapes": array_shapes(output_arrays),
        "output_strides": [array_strides(array) for array in output_arrays],
        "output_orders": [array_order(array) for array in output_arrays],
        "output_contiguous": [array_contiguous(array) for array in output_arrays],
        "output_backends": array_backend_names(output_arrays),
        "output_device_kinds": array_device_kinds(output_arrays),
        "output_locations": array_locations(output_arrays),
        "output_is_host": [array_is_host(array) for array in output_arrays],
        "output_is_device": [array_is_device(array) for array in output_arrays],
        "output_is_distributed": [array_is_distributed(array) for array in output_arrays],
        "input_dtype": str(getattr(coef_array, "dtype", None)),
        "output_dtype": str(getattr(output_arrays[0], "dtype", None)) if output_arrays else None,
        "kernel_kind": kernel_kind,
        "matrix_elements": matrix_elements,
        "block_count": int(block_count),
        "max_block_m": max((shape[0] for shape in block_shapes if len(shape) >= 1), default=0),
        "max_block_n": max((shape[1] for shape in block_shapes if len(shape) >= 2), default=0),
        "max_block_elements": max_block_elements,
        "total_block_elements": total_block_elements,
        "block_flops_estimate": int(total_block_flops),
        "dense_flops_estimate": int(dense_flops),
        "block_flop_fraction": total_block_flops / dense_flops if dense_flops else None,
        "largest_block_flops_estimate": int(largest_block_flops),
        "largest_block_flop_fraction": (
            largest_block_flops / total_block_flops if total_block_flops else None
        ),
        "batchable_flops_estimate": int(batchable_flops),
        "batchable_flop_fraction": batchable_flops / total_block_flops if total_block_flops else None,
        "batchable_block_group_count": int(batchable_block_group_count),
        "dominant_block_shape_flop_fraction": dominant_block_shape_flop_fraction,
        "block_group_signature": block_group_signature,
        "top_block_shape_groups": block_shape_groups[:8],
        "block_shape_counts": block_shape_counts,
        "unique_block_shape_count": len(block_shape_counts),
        "dominant_block_shape": dominant_block_shape,
        "dominant_block_shape_count": dominant_block_shape_count,
        "block_shape_reuse_fraction": (
            dominant_block_shape_count / block_count if block_count else None
        ),
        "batchable_block_count": int(batchable_block_count),
        "batchable_block_fraction": batchable_block_count / block_count if block_count else None,
        "block_shape_fragmentation": len(block_shape_counts) / block_count if block_count else None,
        "block_element_fraction": (
            total_block_elements / matrix_elements if matrix_elements else None
        ),
        "qn_block_density": total_block_elements / matrix_elements if matrix_elements else None,
        "qn_sparse_fraction": 1 - (total_block_elements / matrix_elements) if matrix_elements else None,
        "largest_block_fraction": (
            max_block_elements / total_block_elements if total_block_elements else None
        ),
        "max_block_aspect_ratio": max(aspect_ratios, default=0.0),
        "tiny_block_count": int(tiny_block_count),
        "skinny_block_count": int(skinny_block_count),
        "rank_sum": int(sum(ranks)),
        "max_block_rank": max(ranks, default=0),
        "cost_model": "qn_sparse_blocked_decomposition" if block_count else "dense_decomposition",
        "bottleneck_hints": _profile_tags(bottleneck_hints),
        "profile_tags": _profile_tags(tags),
        "flops_estimate": flops,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "memory_bytes_estimate": int(read_bytes + write_bytes),
    }
    payload["practical_bucket"] = _svd_practical_bucket(payload, matrix_shape=matrix_shape)
    payload["kernel_observation"] = _kernel_observation(payload)
    payload["practical_profile"] = _svd_event_practical_profile(payload)
    payload["compute_profile"] = _event_compute_profile(payload)
    payload["core_operation"] = _core_operation_index(payload)
    return payload


def _decomposition_subclass(mode):
    upper_mode = str(mode).upper()
    if upper_mode == "QR":
        return "qr_qn"
    if upper_mode == "EIGH":
        return "eigh_qn"
    return "svd_qn"


def array_total_bytes(values) -> int:
    return int(sum(getattr(value, "nbytes", 0) for value in values if value is not None))


def mp_event_name(mp, action):
    if getattr(mp, "is_mps", False):
        return f"mps_{action}"
    if getattr(mp, "is_mpo", False):
        return f"mpo_{action}"
    if getattr(mp, "is_mpdm", False):
        return f"mpdm_{action}"
    return f"matrix_product_{action}"


def mp_tensor_shapes(mp):
    return array_shapes(mp)


def tree_edge_count(tree) -> int:
    return sum(len(node.children) for node in tree.node_list)


def tree_node_shapes(tree):
    return [tuple(node.shape) for node in tree]


def tree_total_bytes(tree) -> int:
    return int(sum(getattr(node.tensor, "nbytes", 0) for node in tree))


def _unique_profile_values(values):
    seen = set()
    unique = []
    for value in values:
        if value is None:
            continue
        value = str(value)
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def contraction_plan_summary(plan, *, equation=None):
    steps = []
    lowerings = []
    fallback_reasons = []
    num_gemm = 0
    num_batched_gemm = 0
    num_grouped_tasks = 0
    num_blocks = 0
    num_shape_buckets = 0
    for index, step in enumerate(getattr(plan, "steps", ()) or ()):
        nested_plan = getattr(step, "plan", None)
        lowering = getattr(nested_plan, "kind", None) or getattr(step, "kind", None)
        lowering = None if lowering is None else str(lowering)
        lowerings.append(lowering)
        descs = tuple(getattr(nested_plan, "descs", ()) or ())
        bucketed_by_shape = getattr(nested_plan, "bucketed_by_shape", None)
        shape_bucket_count = len(bucketed_by_shape) if isinstance(bucketed_by_shape, dict) else 0
        if lowering == "gemm":
            num_gemm += 1
        elif lowering in ("batched_gemm", "strided_batched_gemm"):
            num_batched_gemm += 1
        elif lowering == "grouped_gemm":
            num_grouped_tasks += len(descs)
            num_shape_buckets += shape_bucket_count
        fallback_reason = getattr(step, "fallback_reason", None) or getattr(nested_plan, "fallback_reason", None)
        if fallback_reason:
            fallback_reasons.append(str(fallback_reason))
        steps.append({
            "step_index": int(index),
            "kind": str(getattr(step, "kind", lowering or "unknown")),
            "lowering": lowering,
            "input_modes": [list(modes) for modes in getattr(step, "input_modes", ())],
            "output_modes": list(getattr(step, "output_modes", ())),
            "output_shape": _profile_shape(getattr(nested_plan, "output_shape", ())),
            "num_descs": len(descs),
            "num_shape_buckets": shape_bucket_count,
            "fallback_reason": None if fallback_reason is None else str(fallback_reason),
            "estimated_flops": _int_profile_value(getattr(step, "estimated_flops", 0)),
            "estimated_read_bytes": _int_profile_value(getattr(step, "estimated_read_bytes", 0)),
            "estimated_write_bytes": _int_profile_value(getattr(step, "estimated_write_bytes", 0)),
            "estimated_copy_bytes": _int_profile_value(getattr(step, "estimated_copy_bytes", 0)),
            "required_workspace_bytes": _int_profile_value(getattr(step, "required_workspace_bytes", 0)),
        })
    return {
        "equation": None if equation is None else str(equation).replace(" ", ""),
        "plan_hash": getattr(plan, "plan_hash", None),
        "contraction_count": len(steps),
        "steps": steps,
        "lowerings": _unique_profile_values(lowerings),
        "num_gemm": num_gemm,
        "num_batched_gemm": num_batched_gemm,
        "num_grouped_tasks": num_grouped_tasks,
        "num_blocks": num_blocks,
        "num_shape_buckets": num_shape_buckets,
        "fallback_reasons": _unique_profile_values(fallback_reasons),
        "estimated_flops": _int_profile_value(getattr(plan, "estimated_flops", 0)),
        "estimated_read_bytes": _int_profile_value(getattr(plan, "estimated_read_bytes", 0)),
        "estimated_write_bytes": _int_profile_value(getattr(plan, "estimated_write_bytes", 0)),
        "estimated_copy_bytes": _int_profile_value(getattr(plan, "estimated_copy_bytes", 0)),
        "required_workspace_bytes": _int_profile_value(getattr(plan, "required_workspace_bytes", 0)),
        "estimated_peak_bytes": _int_profile_value(getattr(plan, "estimated_peak_bytes", 0)),
    }


def aggregate_contraction_plan_summaries(plan_summaries):
    summaries = [dict(summary) for summary in (plan_summaries or ()) if summary]
    if not summaries:
        return {}
    lowerings = []
    fallback_reasons = []
    steps = []
    for summary in summaries:
        lowerings.extend(summary.get("lowerings", ()) or ())
        fallback_reasons.extend(summary.get("fallback_reasons", ()) or ())
        steps.extend(summary.get("steps", ()) or ())
    return {
        "lowering": "multi_tensor_contract",
        "num_contraction_steps": sum(_int_profile_value(summary.get("contraction_count")) for summary in summaries),
        "num_gemm": sum(_int_profile_value(summary.get("num_gemm")) for summary in summaries),
        "num_batched_gemm": sum(_int_profile_value(summary.get("num_batched_gemm")) for summary in summaries),
        "num_grouped_tasks": sum(_int_profile_value(summary.get("num_grouped_tasks")) for summary in summaries),
        "num_blocks": sum(_int_profile_value(summary.get("num_blocks")) for summary in summaries),
        "num_shape_buckets": sum(_int_profile_value(summary.get("num_shape_buckets")) for summary in summaries),
        "fallback_reason": "; ".join(_unique_profile_values(fallback_reasons)) or None,
        "estimated_flops": sum(_int_profile_value(summary.get("estimated_flops")) for summary in summaries),
        "estimated_read_bytes": sum(_int_profile_value(summary.get("estimated_read_bytes")) for summary in summaries),
        "estimated_write_bytes": sum(_int_profile_value(summary.get("estimated_write_bytes")) for summary in summaries),
        "estimated_copy_bytes": sum(_int_profile_value(summary.get("estimated_copy_bytes")) for summary in summaries),
        "required_workspace_bytes": max((_int_profile_value(summary.get("required_workspace_bytes")) for summary in summaries), default=0),
        "estimated_peak_bytes": max((_int_profile_value(summary.get("estimated_peak_bytes")) for summary in summaries), default=0),
        "contraction_plan_summary": {
            "contraction_count": sum(_int_profile_value(summary.get("contraction_count")) for summary in summaries),
            "lowerings": _unique_profile_values(lowerings),
            "fallback_reasons": _unique_profile_values(fallback_reasons),
            "steps": steps,
        },
    }


def multi_tensor_contract_payload(path, operands, result, plan_summaries=None):
    path_steps = []
    for indices, equation in path:
        path_steps.append({
            "indices": [int(index) for index in indices],
            "equation": str(equation).replace(" ", ""),
        })
    payload = {
        "contraction_count": len(path_steps),
        "path_steps": path_steps,
        "input_shapes": array_shapes(operands),
        "input_dtypes": array_dtype_names(operands),
        "operand_array_types": array_type_names(operands),
        "operand_array_backends": array_backend_names(operands),
        "output_shape": array_shape(result),
        "output_dtype": str(getattr(result, "dtype", None)),
        "output_backend": array_backend_name(result) if hasattr(result, "shape") else None,
        "read_bytes": array_total_bytes(operands),
        "write_bytes": int(getattr(result, "nbytes", 0)),
    }
    payload.update(aggregate_contraction_plan_summaries(plan_summaries))
    return payload


class _ProfilingRuntime:
    def __init__(self):
        import contextvars
        import json
        import threading
        import time

        self._json = json
        self._time = time
        self._scope_stack = contextvars.ContextVar("renormalizer_profiling_scope", default=())
        self._summary_lock = threading.Lock()
        self._summaries = {}
        self._compute_summaries = {}
        self._compute_class_summaries = {}
        self._overhead_lock = threading.Lock()
        self._overhead = dict(_OVERHEAD_TEMPLATE)
        self._event_output = None
        self._id_lock = threading.Lock()
        self._next_event_id = 1
        self._next_span_id = 1

    def perf_counter(self):
        return self._time.perf_counter()

    def time_ns(self):
        return self._time.time_ns()

    def _to_jsonable(self, value):
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]
        if isinstance(value, complex):
            return {"real": value.real, "imag": value.imag}
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if hasattr(value, "shape"):
            try:
                return {"shape": list(value.shape), "dtype": str(getattr(value, "dtype", None))}
            except Exception:
                return repr(value)
        return value

    def current_scope(self) -> dict:
        merged = {}
        for frame in self._scope_stack.get():
            merged.update(frame)
        return merged

    def push_scope(self, **fields):
        stack = self._scope_stack.get()
        return self._scope_stack.set(stack + (fields,))

    def pop_scope(self, token) -> None:
        self._scope_stack.reset(token)

    def _allocate_event_id(self):
        with self._id_lock:
            event_id = self._next_event_id
            self._next_event_id += 1
            return event_id

    def _allocate_span_id(self):
        with self._id_lock:
            span_id = self._next_span_id
            self._next_span_id += 1
            return span_id

    def push_span(self, name, **fields):
        parent_span_id = self.current_scope().get("span_id")
        span_id = self._allocate_span_id()
        token = self.push_scope(
            span_id=span_id,
            parent_span_id=parent_span_id,
            span_name=name,
            **fields,
        )
        return token, span_id

    def record(self, event: str, **fields) -> None:
        if not enabled():
            return
        started = self.perf_counter()
        try:
            payload = {
                **self.current_scope(),
                "event": event,
                **fields,
            }
            payload = {
                "event_id": self._allocate_event_id(),
                "timestamp_ns": self.time_ns(),
                **payload,
            }
            compute_accounting_override = payload.pop("compute_accounting_override", None)
            if compute_accounting_override is not None and payload.get("compute_class"):
                payload["compute_accounting"] = str(compute_accounting_override)
            payload = _standard_contraction_execute_payload(payload)
            payload = _standard_contraction_plan_payload(payload)
            payload = _standard_hmm_task_build_payload(payload)
            payload = _standard_grouped_gemm_execute_payload(payload)
            payload = _standard_matmul_execute_payload(payload)
            payload = _standard_grouped_gemm_prepack_payload(payload)
            if event in _JSONL_EVENTS:
                payload = _refresh_event_compute_profile(payload)
                self._bump_overhead("events_seen", 1)
                if self._event_output is not None:
                    self._write_event(payload)
                self._record_summary(payload)
                self._bump_overhead("events_summarized", 1)
                return
            text = self._json.dumps(self._to_jsonable(payload), sort_keys=True, separators=(",", ":"))
            logger.profiling("%s%s", LOG_PREFIX, text)
            self._bump_overhead("events_seen", 1)
            self._bump_overhead("events_logged", 1)
        finally:
            self._bump_overhead("record_overhead_s", self.perf_counter() - started)

    def _bump_overhead(self, key, value):
        with self._overhead_lock:
            self._overhead[key] += value

    def _signature_payload(self, payload):
        volatile_or_large_fields = {
            "event_id",
            "timestamp_ns",
            "span_id",
            "parent_span_id",
            "wall_s",
            "path",
            "path_steps",
            "contraction_types",
            "contraction_steps",
            "shape_buckets",
            "blocks",
            "operands",
            "task_operands",
            "task_specs",
            "communication",
            "input_states",
            "output_state",
            "output_block_keys",
            "unique_output_block_keys",
            "result_block_shapes",
            "non_gemm_steps",
            "step_costs",
            "top_step_costs",
            "dominant_step",
            "step_output_shape_counts",
            "block_shape_counts",
            "top_block_shape_groups",
            "practical_profile",
            "compute_profile",
            "core_operation",
            "kernel_observation",
        }
        return {
            key: value for key, value in payload.items()
            if key not in volatile_or_large_fields
        }

    def _record_summary(self, payload):
        started = self.perf_counter()
        signature = self._signature_payload(payload)
        compute_signature = self._compute_signature_payload(payload)
        try:
            key = self._json.dumps(self._to_jsonable(signature), sort_keys=True, separators=(",", ":"))
            wall_s = payload.get("wall_s")
            if wall_s is None:
                wall_s = 0.0
            wall_s = float(wall_s)
            with self._summary_lock:
                item = self._summaries.get(key)
                if item is None:
                    self._summaries[key] = {
                        "signature": signature,
                        "call_count": 1,
                        "total_wall_s": wall_s,
                        "min_wall_s": wall_s,
                        "max_wall_s": wall_s,
                    }
                else:
                    item["call_count"] += 1
                    item["total_wall_s"] += wall_s
                    item["min_wall_s"] = min(item["min_wall_s"], wall_s)
                    item["max_wall_s"] = max(item["max_wall_s"], wall_s)
                if compute_signature is not None:
                    self._record_compute_summary_unlocked(payload, compute_signature, wall_s)
                    class_signature = self._compute_class_signature_payload(payload)
                    if class_signature is not None:
                        self._record_compute_class_summary_unlocked(payload, class_signature, wall_s)
        finally:
            self._bump_overhead("summary_overhead_s", self.perf_counter() - started)

    @staticmethod
    def _number(payload, *names):
        for name in names:
            value = payload.get(name)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    @staticmethod
    def _hmm_step_desc_count(payload):
        steps = payload.get("hmm_steps")
        if not isinstance(steps, (list, tuple)):
            return 0.0
        total = 0.0
        for step in steps:
            if not isinstance(step, dict):
                continue
            descs = step.get("descs")
            if isinstance(descs, (list, tuple)):
                total += len(descs)
                continue
            total += _ProfilingRuntime._number(step, "num_desc")
        return total

    @staticmethod
    def _hmm_gemv_desc_count(payload):
        batches = payload.get("hmm_gemv_batches")
        if not isinstance(batches, dict):
            return 0.0
        total = 0.0
        for phase_batches in batches.values():
            if not isinstance(phase_batches, (list, tuple)):
                continue
            for batch in phase_batches:
                if not isinstance(batch, dict):
                    continue
                descs = batch.get("descs")
                if isinstance(descs, (list, tuple)):
                    total += len(descs)
                    continue
                total += _ProfilingRuntime._number(batch, "num_tasks")
        return total

    @staticmethod
    def _hmm_gemv_shape_bucket_count(payload):
        explicit = _ProfilingRuntime._number(payload, "num_gemv_shape_buckets", "hmm_num_gemv_shape_buckets")
        if explicit:
            return explicit
        buckets = payload.get("hmm_gemv_shape_buckets")
        if isinstance(buckets, (list, tuple)):
            return float(len(buckets))
        batches = payload.get("hmm_gemv_batches")
        if not isinstance(batches, dict):
            return 0.0
        total = 0.0
        for phase_batches in batches.values():
            if not isinstance(phase_batches, (list, tuple)):
                continue
            for batch in phase_batches:
                if not isinstance(batch, dict):
                    continue
                total += _ProfilingRuntime._number(batch, "num_groups")
        return total

    @staticmethod
    def _communication_bytes(payload):
        value = _ProfilingRuntime._number(payload, "comm_bytes", "estimated_comm_bytes")
        if value:
            return value
        communication = payload.get("communication")
        if isinstance(communication, dict):
            return _ProfilingRuntime._number(communication, "bytes")
        if isinstance(communication, (list, tuple)):
            total = 0.0
            for item in communication:
                if isinstance(item, dict):
                    total += _ProfilingRuntime._number(item, "bytes")
            return total
        return 0.0

    @staticmethod
    def _communication_breakdown(payload):
        bytes_by_collective = {}
        wall_s_by_collective = {}
        messages_by_collective = {}
        max_block_size_by_collective = {}

        def add_item(item):
            if not isinstance(item, dict):
                return
            collective = item.get("collective", item.get("kind"))
            if collective is None:
                return
            collective = str(collective)
            bytes_by_collective[collective] = (
                bytes_by_collective.get(collective, 0.0)
                + _ProfilingRuntime._number(item, "bytes")
            )
            wall_s_by_collective[collective] = (
                wall_s_by_collective.get(collective, 0.0)
                + _ProfilingRuntime._number(item, "wall_s")
            )
            num_messages = _ProfilingRuntime._number(item, "num_messages")
            if not num_messages:
                num_messages = 1.0
            messages_by_collective[collective] = messages_by_collective.get(collective, 0.0) + num_messages
            block_size = _ProfilingRuntime._number(item, "block_size")
            if block_size:
                max_block_size_by_collective[collective] = max(
                    max_block_size_by_collective.get(collective, 0.0),
                    block_size,
                )

        communication = payload.get("communication")
        if isinstance(communication, dict):
            add_item(communication)
        elif isinstance(communication, (list, tuple)):
            for item in communication:
                add_item(item)
        return {
            "comm_bytes_by_collective": bytes_by_collective,
            "comm_wall_s_by_collective": wall_s_by_collective,
            "comm_messages_by_collective": messages_by_collective,
            "max_comm_block_size_by_collective": max_block_size_by_collective,
        }

    def _compute_signature_payload(self, payload):
        if not payload.get("compute_class"):
            return None
        keys = (
            "compute_class",
            "compute_subclass",
            "compute_role",
            "compute_accounting",
            "backend",
            "stage",
            "method",
            "lowering",
            "device",
            "device_kind",
        )
        signature = {
            key: payload[key]
            for key in keys
            if key in payload and payload[key] is not None
        }
        kernel_kind = self._compute_kernel_kind(payload)
        if kernel_kind is not None:
            signature["kernel_kind"] = kernel_kind
        return signature

    def _compute_class_signature_payload(self, payload):
        if payload.get("compute_accounting", COMPUTE_ACCOUNTING_PRIMARY) != COMPUTE_ACCOUNTING_PRIMARY:
            return None
        keys = (
            "compute_class",
            "backend",
            "stage",
            "method",
            "device",
            "device_kind",
        )
        return {
            key: payload[key]
            for key in keys
            if key in payload and payload[key] is not None
        }

    def _compute_metrics(self, payload, wall_s=0.0):
        m, n, k = self._matrix_problem_dimensions(payload)
        contraction_type_counts = self._contraction_type_counts(payload)
        total_hmm_gemm_desc = self._number(payload, "num_gemm_desc", "hmm_num_gemm_desc")
        if not total_hmm_gemm_desc:
            total_hmm_gemm_desc = self._hmm_step_desc_count(payload)
        total_hmm_gemv_desc = self._number(payload, "num_gemv_desc", "hmm_num_gemv_desc")
        if not total_hmm_gemv_desc:
            total_hmm_gemv_desc = self._hmm_gemv_desc_count(payload)
        total_hmm_gemv_shape_buckets = self._hmm_gemv_shape_bucket_count(payload)
        total_oe_tensordot_steps = self._number(payload, "tensordot_step_count")
        if not total_oe_tensordot_steps:
            total_oe_tensordot_steps = sum(
                contraction_type_counts.get(step_type, 0.0)
                for step_type in _OE_TENSORDOT_STEP_TYPES
            )
        total_oe_gemm_steps = contraction_type_counts.get("GEMM", 0.0)
        total_oe_generic_einsum_steps = self._number(payload, "generic_einsum_step_count")
        if not total_oe_generic_einsum_steps:
            total_oe_generic_einsum_steps = max(
                self._number(payload, "contraction_count") - total_oe_gemm_steps - total_oe_tensordot_steps,
                0.0,
            )
        communication = self._communication_breakdown(payload)
        profile_tags = set(self._profile_tags_from_payload(payload))
        execution_resources = _execution_resource_profile(payload)
        workspace_provided_bytes = execution_resources["workspace_provided_bytes"] or 0
        workspace_slack_bytes = execution_resources["workspace_slack_bytes"] or 0
        metrics = {
            "total_flops_estimate": self._number(payload, "flops_estimate", "flops"),
            "total_read_bytes": self._number(payload, "read_bytes"),
            "total_write_bytes": self._number(payload, "write_bytes"),
            "total_copy_bytes": self._number(payload, "copy_bytes"),
            "total_comm_bytes": self._communication_bytes(payload),
            **communication,
            "max_workspace_bytes": self._number(payload, "workspace_bytes", "required_workspace_bytes"),
            "max_workspace_provided_bytes": float(workspace_provided_bytes),
            "max_workspace_slack_bytes": float(workspace_slack_bytes),
            "max_peak_bytes": self._number(payload, "peak_bytes", "largest_intermediate_bytes"),
            "max_largest_intermediate_elements": self._number(
                payload, "largest_intermediate_elements", "largest_intermediate"
            ),
            "max_largest_intermediate_bytes": self._number(payload, "largest_intermediate_bytes"),
            "total_stream_provided_calls": 1.0 if execution_resources["stream_provided"] else 0.0,
            "total_workspace_provided_calls": 1.0 if execution_resources["workspace_provided"] else 0.0,
            "total_contraction_steps": self._number(payload, "contraction_count"),
            "total_svd_blocks": self._number(payload, "block_count"),
            "total_decomposition_matrix_elements": self._number(payload, "matrix_elements"),
            "total_gemm": self._gemm_count(payload),
            "total_batched_gemm": self._number(payload, "num_batched_gemm"),
            "total_grouped_tasks": self._number(payload, "num_grouped_tasks"),
            "total_blocks": self._number(payload, "num_blocks"),
            "total_shape_buckets": self._number(payload, "num_shape_buckets"),
            "total_hmm_gemv_shape_buckets": total_hmm_gemv_shape_buckets,
            "total_hmm_batches": self._number(payload, "num_batches", "hmm_num_batches"),
            "max_hmm_batch_size": self._number(payload, "batch_size", "hmm_batch_size"),
            "total_hmm_gemm_desc": total_hmm_gemm_desc,
            "total_hmm_gemv_desc": total_hmm_gemv_desc,
            "total_output_reduction_groups": self._number(payload, "num_output_reduction_groups"),
            "total_scatter_add_tasks": self._number(payload, "num_scatter_add_tasks"),
            "max_output_contributions": self._number(payload, "max_output_contributions"),
            "scatter_add_calls": 1.0 if payload.get("scatter_add_required") else 0.0,
            "max_m": m,
            "max_n": n,
            "max_k": k,
            "max_svd_block_m": self._number(payload, "max_block_m"),
            "max_svd_block_n": self._number(payload, "max_block_n"),
            "max_svd_block_elements": self._number(payload, "max_block_elements"),
            "total_svd_block_elements": self._number(payload, "total_block_elements"),
            "max_output_rank": self._number(payload, "output_rank"),
            "total_singular_values": self._number(payload, "singular_value_count"),
            "total_oe_gemm_steps": total_oe_gemm_steps,
            "total_oe_tensordot_steps": total_oe_tensordot_steps,
            "total_oe_generic_einsum_steps": total_oe_generic_einsum_steps,
            "total_oe_non_gemm_steps": max(
                self._number(payload, "contraction_count") - total_oe_gemm_steps,
                0.0,
            ),
            "total_rhs_vectors": self._number(payload, "num_rhs"),
            "max_rhs": self._number(payload, "num_rhs"),
            "total_rhs_loop_calls": self._number(payload, "num_rhs_loop_calls"),
            "total_tensordot_axis_permutations": 1.0 if "axis_permutation" in profile_tags else 0.0,
            "total_tensordot_axis_permutation_copy_bytes": self._number(payload, "axis_permutation_copy_bytes"),
            "total_tensordot_high_rank_free": 1.0 if "high_rank_free" in profile_tags else 0.0,
            "total_tensordot_tiny_gemm": 1.0 if "tiny_gemm" in profile_tags else 0.0,
            "total_tensordot_skinny_gemm": 1.0 if "skinny_gemm" in profile_tags else 0.0,
            "max_oe_step_output_elements": self._number(payload, "max_step_output_elements"),
            "total_oe_step_output_elements": self._number(payload, "total_step_output_elements"),
            "max_oe_unique_step_output_shapes": self._number(payload, "unique_step_output_shape_count"),
            "total_oe_gemm_flops_estimate": self._number(payload, "gemm_flops_estimate"),
            "total_oe_tensordot_flops_estimate": self._number(payload, "tensordot_flops_estimate"),
            "total_oe_generic_einsum_flops_estimate": self._number(
                payload, "generic_einsum_flops_estimate"
            ),
            "total_oe_non_gemm_flops_estimate": self._number(payload, "non_gemm_flops_estimate"),
            "max_oe_dominant_step_flops_estimate": self._number(payload, "dominant_step_flops_estimate"),
            "max_svd_unique_block_shapes": self._number(payload, "unique_block_shape_count"),
            "max_svd_block_shape_reuse_fraction": self._number(payload, "block_shape_reuse_fraction"),
            "total_svd_tiny_blocks": self._number(payload, "tiny_block_count"),
            "total_svd_skinny_blocks": self._number(payload, "skinny_block_count"),
            "total_svd_batchable_blocks": self._number(payload, "batchable_block_count"),
            "total_svd_block_flops_estimate": self._number(payload, "block_flops_estimate"),
            "total_svd_dense_flops_estimate": self._number(payload, "dense_flops_estimate"),
            "total_svd_batchable_flops_estimate": self._number(payload, "batchable_flops_estimate"),
            "max_svd_batchable_flop_fraction": self._number(payload, "batchable_flop_fraction"),
            "max_svd_batchable_block_group_count": self._number(payload, "batchable_block_group_count"),
            "max_svd_dominant_block_shape_flop_fraction": self._number(
                payload, "dominant_block_shape_flop_fraction"
            ),
            "fallback_calls": 1 if payload.get("fallback_reason") else 0,
            **self._practical_profile_metrics(payload, wall_s),
        }
        compute_s = _optional_float_profile_value(payload.get("compute_s"), payload.get("estimated_compute_s"))
        if compute_s is not None:
            metrics["compute_s"] = compute_s
        pointer_setup_s = _optional_float_profile_value(payload.get("pointer_setup_s"))
        if pointer_setup_s is not None:
            metrics["pointer_setup_s"] = pointer_setup_s
        for key in ("pack_s", "kernel_s", "scatter_s", "loop_s"):
            value = _optional_float_profile_value(payload.get(key))
            if value is not None:
                metrics[key] = value
        return metrics

    @staticmethod
    def _counter_payload(key, value=1.0):
        if key is None:
            return {}
        return {str(key): float(value)}

    def _practical_profile_metrics(self, payload, wall_s):
        kernel_kind = self._compute_kernel_kind(payload)
        problem_size_bin = self._problem_size_bin(payload)
        problem_shape = self._problem_shape_key(payload)
        bottleneck_hints = self._bottleneck_hints(payload)
        profile_tags = self._profile_tags_from_payload(payload)
        return {
            "kernel_kind_counts": self._counter_payload(kernel_kind, 1.0),
            "kernel_kind_wall_s": self._counter_payload(kernel_kind, wall_s),
            "problem_size_bins": self._counter_payload(problem_size_bin, 1.0),
            "problem_shape_counts": self._counter_payload(problem_shape, 1.0),
            "problem_shape_wall_s": self._counter_payload(problem_shape, wall_s),
            "bottleneck_hint_counts": {
                str(hint): 1.0
                for hint in bottleneck_hints
            },
            "bottleneck_hint_wall_s": {
                str(hint): float(wall_s)
                for hint in bottleneck_hints
            },
            "profile_tag_counts": {
                str(tag): 1.0
                for tag in profile_tags
            },
            "profile_tag_wall_s": {
                str(tag): float(wall_s)
                for tag in profile_tags
            },
        }

    @staticmethod
    def _profile_tags_from_payload(payload):
        tags = payload.get("profile_tags") or ()
        if isinstance(tags, str):
            tags = (tags,)
        return tuple(dict.fromkeys(str(tag) for tag in tags if tag is not None))

    def _problem_size_bin(self, payload):
        explicit = payload.get("problem_size_bin")
        if explicit:
            return str(explicit)
        flops = self._number(payload, "flops_estimate", "flops")
        memory_bytes = (
            self._number(payload, "read_bytes")
            + self._number(payload, "write_bytes")
            + self._number(payload, "copy_bytes")
            + self._communication_bytes(payload)
        )
        return _problem_size_bin_from_scale(flops, memory_bytes)

    def _problem_shape_key(self, payload):
        explicit = payload.get("problem_signature")
        if explicit:
            return str(explicit)
        compute_class = payload.get("compute_class")
        kernel_kind = self._compute_kernel_kind(payload) or str(compute_class or "compute")
        if compute_class == COMPUTE_CLASS_SVD:
            block_count = int(self._number(payload, "block_count"))
            max_block_m = int(self._number(payload, "max_block_m"))
            max_block_n = int(self._number(payload, "max_block_n"))
            output_rank = int(self._number(payload, "output_rank"))
            return "{0}:blocks={1},max={2}x{3},rank={4}".format(
                kernel_kind,
                block_count,
                max_block_m,
                max_block_n,
                output_rank,
            )
        if compute_class == COMPUTE_CLASS_OE:
            contraction_count = int(self._number(payload, "contraction_count"))
            type_counts = self._contraction_type_counts(payload)
            if type_counts:
                type_text = "|".join(
                    "{0}:{1}".format(key, self._format_counter_value(value))
                    for key, value in sorted(type_counts.items())
                )
            else:
                type_text = "none"
            largest = int(self._number(payload, "largest_intermediate_elements", "largest_intermediate"))
            return "{0}:steps={1},types={2},largest={3}".format(
                kernel_kind,
                contraction_count,
                type_text,
                largest,
            )
        if compute_class == COMPUTE_CLASS_TENSORDOT:
            if self._number(payload, "num_rhs_loop_calls"):
                return "{0}:rhs={1}".format(kernel_kind, int(self._number(payload, "num_rhs")))
            m, n, k = self._matrix_problem_dimensions(payload)
            if m or n or k:
                return "{0}:m={1},n={2},k={3}".format(kernel_kind, int(m), int(n), int(k))
            input_shapes = payload.get("input_shapes")
            output_shape = payload.get("output_shape")
            if input_shapes is not None or output_shape is not None:
                return "{0}:inputs={1}->output={2}".format(kernel_kind, input_shapes, output_shape)
        return kernel_kind

    @staticmethod
    def _format_counter_value(value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if numeric.is_integer():
            return str(int(numeric))
        return str(numeric)

    def _bottleneck_hints(self, payload):
        explicit = payload.get("bottleneck_hints")
        if explicit:
            if isinstance(explicit, str):
                explicit = (explicit,)
            return tuple(dict.fromkeys(str(hint) for hint in explicit if hint is not None))
        hints = []
        if payload.get("fallback_reason"):
            hints.append("fallback")
        if self._number(payload, "num_rhs_loop_calls"):
            hints.append("python_rhs_loop")
        if self._communication_bytes(payload):
            hints.append("communication")
        copy_bytes = self._number(payload, "copy_bytes")
        profile_tags = set(self._profile_tags_from_payload(payload))
        if self._number(payload, "axis_permutation_copy_bytes") or "axis_permutation" in profile_tags:
            hints.append("axis_permutation_copy")
        elif copy_bytes:
            hints.append("copy")

        compute_class = payload.get("compute_class")
        if compute_class == COMPUTE_CLASS_SVD:
            hints.append("decomposition")
        elif compute_class == COMPUTE_CLASS_OE:
            hints.append("oe_path")
            path_profile = _oe_path_profile_from_counts(self._contraction_type_counts(payload))
            path_kind = payload.get("path_kind") or path_profile["path_kind"]
            if path_kind and path_kind != "unknown":
                hints.append("oe_{0}_path".format(path_kind))
            if self._number(payload, "non_gemm_step_count") or path_profile["non_gemm_step_count"]:
                hints.append("oe_non_gemm_steps")
            if self._number(payload, "non_gemm_flops_estimate"):
                hints.append("oe_non_gemm_flops")
            largest = self._number(payload, "largest_intermediate_bytes", "largest_intermediate_elements")
            memory_bytes = self._number(payload, "read_bytes") + self._number(payload, "write_bytes")
            if largest and memory_bytes and largest > memory_bytes:
                hints.append("large_intermediate")
        elif compute_class == COMPUTE_CLASS_TENSORDOT:
            loop_tasks = self._number(payload, "loop_task_count")
            grouped_tasks = self._number(payload, "num_grouped_tasks")
            if grouped_tasks and loop_tasks:
                hints.append("looped_grouped_gemm")
            if not hints:
                flops = self._number(payload, "flops_estimate", "flops")
                memory_bytes = self._number(payload, "read_bytes") + self._number(payload, "write_bytes")
                if flops and memory_bytes and flops / memory_bytes < 4.0:
                    hints.append("memory_bandwidth")
                elif flops:
                    hints.append("compute")
        if not hints:
            hints.append("unclassified")
        return tuple(dict.fromkeys(hints))

    @staticmethod
    def _contraction_type_counts(payload):
        counts = {}
        explicit = payload.get("contraction_type_counts")
        if isinstance(explicit, dict):
            for key, value in explicit.items():
                try:
                    counts[str(key)] = float(value)
                except (TypeError, ValueError):
                    counts[str(key)] = 0.0
            return counts
        for contraction_type in payload.get("contraction_types", ()) or ():
            key = str(contraction_type)
            counts[key] = counts.get(key, 0.0) + 1.0
        return counts

    def _gemm_count(self, payload):
        if payload.get("num_gemm") is not None:
            return self._number(payload, "num_gemm")
        kernel_kind = self._compute_kernel_kind(payload)
        if kernel_kind in {
            "gemm",
            "gemv",
            "dot",
            "batched_gemm",
            "strided_batched_gemm",
            "grouped_gemm",
            "block_grouped_gemm",
        }:
            return 1.0
        return 0.0

    @staticmethod
    def _normalize_profile_modes(modes):
        if modes is None:
            return []
        if isinstance(modes, str):
            return list(modes)
        return [str(mode) for mode in modes]

    @staticmethod
    def _matrix_problem_dimensions(payload):
        m = _ProfilingRuntime._number(payload, "m")
        n = _ProfilingRuntime._number(payload, "n")
        k = _ProfilingRuntime._number(payload, "k")
        if m or n or k:
            return m, n, k

        task_specs = payload.get("task_specs") or ()
        if task_specs:
            return (
                max((_ProfilingRuntime._number(task, "m") for task in task_specs), default=0.0),
                max((_ProfilingRuntime._number(task, "n") for task in task_specs), default=0.0),
                max((_ProfilingRuntime._number(task, "k") for task in task_specs), default=0.0),
            )

        lowering = payload.get("lowering")
        if lowering not in {
            "gemm",
            "batched_gemm",
            "strided_batched_gemm",
            "grouped_gemm",
            "block_grouped_gemm",
        }:
            return 0.0, 0.0, 0.0

        input_shapes = payload.get("input_shapes") or ()
        input_modes = payload.get("input_modes") or ()
        output_modes = payload.get("output_modes") or ()
        if len(input_shapes) < 2 or len(input_modes) < 2:
            return 0.0, 0.0, 0.0

        left_shape = tuple(input_shapes[0])
        right_shape = tuple(input_shapes[1])
        left_modes = _ProfilingRuntime._normalize_profile_modes(input_modes[0])
        right_modes = _ProfilingRuntime._normalize_profile_modes(input_modes[1])
        output_mode_set = set(_ProfilingRuntime._normalize_profile_modes(output_modes))
        if len(left_shape) != len(left_modes) or len(right_shape) != len(right_modes):
            return 0.0, 0.0, 0.0

        left_mode_set = set(left_modes)
        right_mode_set = set(right_modes)
        contracted_modes = (left_mode_set & right_mode_set) - output_mode_set
        left_free = [
            int(size) for mode, size in zip(left_modes, left_shape)
            if mode in output_mode_set and mode not in right_mode_set
        ]
        right_free = [
            int(size) for mode, size in zip(right_modes, right_shape)
            if mode in output_mode_set and mode not in left_mode_set
        ]
        contracted = [
            int(size) for mode, size in zip(left_modes, left_shape)
            if mode in contracted_modes
        ]
        return float(_prod(left_free)), float(_prod(right_free)), float(_prod(contracted))

    def _compute_kernel_kind(self, payload):
        value = payload.get("algorithmic_kernel")
        if value:
            return str(value)
        value = payload.get("kernel_kind")
        if value:
            return str(value)

        lowering = payload.get("lowering")
        if lowering:
            return str(lowering)

        compute_class = payload.get("compute_class")
        if compute_class == COMPUTE_CLASS_SVD:
            subclass = payload.get("compute_subclass")
            return str(subclass) if subclass else "decomposition"
        if compute_class == COMPUTE_CLASS_OE:
            return _oe_path_profile_from_counts(self._contraction_type_counts(payload))["kernel_kind"]

        if compute_class == COMPUTE_CLASS_TENSORDOT:
            m, n, k = self._matrix_problem_dimensions(payload)
            return _kernel_kind_from_mnk(int(m), int(n), int(k))
        return None

    def _new_compute_item(self, signature, payload, wall_s, metrics, extra_sets=()):
        source_event = payload.get("event")
        kernel_kind = self._compute_kernel_kind(payload)
        item = {
            "signature": signature,
            "source_events": {str(source_event)} if source_event is not None else set(),
            "kernel_kinds": {kernel_kind} if kernel_kind is not None else set(),
            "call_count": 1,
            "total_wall_s": wall_s,
            "min_wall_s": wall_s,
            "max_wall_s": wall_s,
            **metrics,
        }
        for field, payload_name in extra_sets:
            value = payload.get(payload_name)
            item[field] = {str(value)} if value is not None else set()
        return item

    def _accumulate_compute_item(self, item, payload, wall_s, metrics, extra_sets=()):
        source_event = payload.get("event")
        if source_event is not None:
            item["source_events"].add(str(source_event))
        kernel_kind = self._compute_kernel_kind(payload)
        if kernel_kind is not None:
            item["kernel_kinds"].add(kernel_kind)
        for field, payload_name in extra_sets:
            value = payload.get(payload_name)
            if value is not None:
                item[field].add(str(value))
        item["call_count"] += 1
        item["total_wall_s"] += wall_s
        item["min_wall_s"] = min(item["min_wall_s"], wall_s)
        item["max_wall_s"] = max(item["max_wall_s"], wall_s)
        for key, value in metrics.items():
            if isinstance(value, dict):
                target = item.setdefault(key, {})
                for dict_key, dict_value in value.items():
                    if key.startswith("max_"):
                        target[dict_key] = max(target.get(dict_key, 0.0), dict_value)
                    else:
                        target[dict_key] = target.get(dict_key, 0.0) + dict_value
            elif key.startswith("max_"):
                item[key] = max(item.get(key, 0.0), value)
            else:
                item[key] = item.get(key, 0.0) + value

    def _record_compute_summary_unlocked(self, payload, signature, wall_s):
        key = self._json.dumps(self._to_jsonable(signature), sort_keys=True, separators=(",", ":"))
        metrics = self._compute_metrics(payload, wall_s)
        extra_sets = (
            ("fallback_sources", "fallback_from"),
            ("fallback_targets", "fallback_to"),
            ("fallback_reasons", "fallback_reason"),
            ("fallback_policies", "fallback_policy"),
        )
        item = self._compute_summaries.get(key)
        if item is None:
            self._compute_summaries[key] = self._new_compute_item(
                signature, payload, wall_s, metrics, extra_sets=extra_sets
            )
            return
        self._accumulate_compute_item(item, payload, wall_s, metrics, extra_sets=extra_sets)

    def _record_compute_class_summary_unlocked(self, payload, signature, wall_s):
        key = self._json.dumps(self._to_jsonable(signature), sort_keys=True, separators=(",", ":"))
        metrics = self._compute_metrics(payload, wall_s)
        extra_sets = (
            ("source_subclasses", "compute_subclass"),
            ("source_roles", "compute_role"),
            ("lowerings", "lowering"),
            ("fallback_sources", "fallback_from"),
            ("fallback_targets", "fallback_to"),
            ("fallback_reasons", "fallback_reason"),
            ("fallback_policies", "fallback_policy"),
        )
        item = self._compute_class_summaries.get(key)
        if item is None:
            self._compute_class_summaries[key] = self._new_compute_item(
                signature, payload, wall_s, metrics, extra_sets=extra_sets
            )
            return
        self._accumulate_compute_item(item, payload, wall_s, metrics, extra_sets=extra_sets)

    @staticmethod
    def _sorted_int_counter(counter):
        return {
            str(key): int(value)
            for key, value in sorted((counter or {}).items())
        }

    @staticmethod
    def _sorted_float_counter(counter):
        return {
            str(key): float(value)
            for key, value in sorted((counter or {}).items())
        }

    @staticmethod
    def _dominant_counter_key(counter, priority=None):
        if not counter:
            return None
        priority = priority or {}
        return max(
            counter.items(),
            key=lambda item: (
                float(item[1]),
                int(priority.get(str(item[0]), 0)),
                str(item[0]),
            ),
        )[0]

    @staticmethod
    def _top_counter_items(counts, wall_s_by_key, limit=8, priority=None):
        priority = priority or {}
        keys = sorted(
            (counts or {}),
            key=lambda key: (
                float((wall_s_by_key or {}).get(key, 0.0)),
                float(counts[key]),
                int(priority.get(str(key), 0)),
                str(key),
            ),
            reverse=True,
        )
        return [
            {
                "key": str(key),
                "count": int(counts[key]),
                "wall_s": float((wall_s_by_key or {}).get(key, 0.0)),
            }
            for key in keys[:limit]
        ]

    def _practical_summary_payload(self, item):
        kernel_counts = item.get("kernel_kind_counts", {})
        kernel_wall_s = item.get("kernel_kind_wall_s", {})
        problem_size_bins = item.get("problem_size_bins", {})
        problem_shape_counts = item.get("problem_shape_counts", {})
        problem_shape_wall_s = item.get("problem_shape_wall_s", {})
        bottleneck_counts = item.get("bottleneck_hint_counts", {})
        bottleneck_wall_s = item.get("bottleneck_hint_wall_s", {})
        profile_tag_counts = item.get("profile_tag_counts", {})
        profile_tag_wall_s = item.get("profile_tag_wall_s", {})
        return {
            "kernel_kind_counts": self._sorted_int_counter(kernel_counts),
            "kernel_kind_wall_s": self._sorted_float_counter(kernel_wall_s),
            "dominant_kernel_kind": self._dominant_counter_key(kernel_wall_s),
            "top_kernel_kinds": self._top_counter_items(kernel_counts, kernel_wall_s),
            "problem_size_bins": self._sorted_int_counter(problem_size_bins),
            "top_problem_shapes": self._top_counter_items(problem_shape_counts, problem_shape_wall_s),
            "bottleneck_hints": sorted(str(key) for key in bottleneck_counts),
            "bottleneck_hint_counts": self._sorted_int_counter(bottleneck_counts),
            "bottleneck_hint_wall_s": self._sorted_float_counter(bottleneck_wall_s),
            "dominant_bottleneck_hint": self._dominant_counter_key(
                bottleneck_wall_s,
                priority=_BOTTLENECK_PRIORITY,
            ),
            "profile_tags": sorted(str(key) for key in profile_tag_counts),
            "profile_tag_counts": self._sorted_int_counter(profile_tag_counts),
            "profile_tag_wall_s": self._sorted_float_counter(profile_tag_wall_s),
            "dominant_profile_tag": self._dominant_counter_key(profile_tag_wall_s),
        }

    @staticmethod
    def _summary_phase_timing_payload(item):
        timing = {}
        for key in (
            "compute_s",
            "pointer_setup_s",
            "pack_s",
            "kernel_s",
            "scatter_s",
            "loop_s",
            "memory_s",
            "copy_s",
            "comm_s",
            "total_s",
            "estimated_time_s",
        ):
            value = item.get(key)
            if value is not None:
                timing[key] = float(value)
        return timing or None

    @staticmethod
    def _attach_summary_phase_timing(payload, phase_timing):
        if not phase_timing:
            return
        payload["phase_timing"] = phase_timing
        for key, value in phase_timing.items():
            payload["total_{0}".format(key)] = value

    @staticmethod
    def _safe_fraction(numerator, denominator):
        return numerator / denominator if denominator > 0 else None

    def _derived_compute_cost_payload(self, item, total_memory_bytes):
        total_copy_bytes = item["total_copy_bytes"]
        total_comm_bytes = item["total_comm_bytes"]
        working_set_bytes = total_memory_bytes + total_copy_bytes + total_comm_bytes
        oe_flops = item["total_oe_gemm_flops_estimate"] + item["total_oe_non_gemm_flops_estimate"]
        svd_block_flops = item["total_svd_block_flops_estimate"]
        svd_dense_flops = item["total_svd_dense_flops_estimate"]
        return {
            "working_set_bytes_estimate": int(working_set_bytes),
            "effective_arithmetic_intensity_flops_per_byte": self._safe_fraction(
                item["total_flops_estimate"], working_set_bytes
            ),
            "copy_fraction_of_working_set": self._safe_fraction(total_copy_bytes, working_set_bytes),
            "comm_fraction_of_working_set": self._safe_fraction(total_comm_bytes, working_set_bytes),
            "tensordot_axis_permutation_copy_fraction_of_working_set": self._safe_fraction(
                item["total_tensordot_axis_permutation_copy_bytes"], working_set_bytes
            ),
            "oe_flop_split_available": bool(oe_flops),
            "oe_total_split_flops": int(oe_flops),
            "oe_gemm_flop_fraction": self._safe_fraction(item["total_oe_gemm_flops_estimate"], oe_flops),
            "oe_non_gemm_flop_fraction": self._safe_fraction(
                item["total_oe_non_gemm_flops_estimate"], oe_flops
            ),
            "svd_block_flop_fraction": self._safe_fraction(svd_block_flops, svd_dense_flops),
            "svd_sparse_saved_flop_fraction": (
                1.0 - (svd_block_flops / svd_dense_flops) if svd_dense_flops > 0 else None
            ),
            "svd_batchable_flop_fraction": self._safe_fraction(
                item["total_svd_batchable_flops_estimate"], svd_block_flops
            ),
        }

    def _summary_execution_resources(self, item, call_count):
        stream_calls = int(item.get("total_stream_provided_calls") or 0)
        workspace_calls = int(item.get("total_workspace_provided_calls") or 0)
        max_required = int(item.get("max_workspace_bytes") or 0)
        max_provided = int(item.get("max_workspace_provided_bytes") or 0)
        max_slack = int(item.get("max_workspace_slack_bytes") or 0)
        if not any((stream_calls, workspace_calls, max_required, max_provided, max_slack)):
            return None
        return {
            "call_count": int(call_count or 0),
            "stream_call_count": stream_calls,
            "workspace_call_count": workspace_calls,
            "stream_fraction": self._safe_fraction(stream_calls, call_count),
            "workspace_fraction": self._safe_fraction(workspace_calls, call_count),
            "max_workspace_required_bytes": max_required,
            "max_workspace_provided_bytes": max_provided,
            "max_workspace_slack_bytes": max_slack,
        }

    @staticmethod
    def _summary_temporary_memory(item):
        max_required = int(item.get("max_workspace_bytes") or 0)
        max_provided = int(item.get("max_workspace_provided_bytes") or 0)
        max_slack = int(item.get("max_workspace_slack_bytes") or 0)
        max_largest_elements = int(item.get("max_largest_intermediate_elements") or 0)
        max_largest_bytes = int(item.get("max_largest_intermediate_bytes") or 0)
        if not any((max_required, max_provided, max_slack, max_largest_elements, max_largest_bytes)):
            return None
        return {
            "max_workspace_required_bytes": max_required,
            "max_workspace_provided_bytes": max_provided,
            "max_workspace_slack_bytes": max_slack,
            "max_largest_intermediate_elements": max_largest_elements,
            "max_largest_intermediate_bytes": max_largest_bytes,
        }

    @staticmethod
    def _append_unique(items, value):
        if value and value not in items:
            items.append(value)

    @staticmethod
    def _first_target(targets, default):
        return targets[0] if targets else default

    @staticmethod
    def _tensordot_diagnosis(profile, item, derived_cost, targets):
        if item.get("total_rhs_loop_calls", 0):
            primary_issue = "python_rhs_loop"
            parallelism_hint = "rhs_batching_needed"
        elif item.get("total_tensordot_axis_permutation_copy_bytes", 0):
            primary_issue = "axis_permutation_copy"
            parallelism_hint = (
                "tiny_or_skinny_gemm"
                if item.get("total_tensordot_tiny_gemm", 0) or item.get("total_tensordot_skinny_gemm", 0)
                else "layout_bound_gemm"
            )
        elif item.get("total_tensordot_tiny_gemm", 0) or item.get("total_tensordot_skinny_gemm", 0):
            primary_issue = "small_gemm"
            parallelism_hint = "tiny_or_skinny_gemm"
        elif item.get("fallback_calls", 0):
            primary_issue = "backend_fallback"
            parallelism_hint = "missing_backend_kernel"
        else:
            primary_issue = "compute_kernel"
            parallelism_hint = "single_kernel"
        return {
            "primary_issue": primary_issue,
            "parallelism_hint": parallelism_hint,
            "precision_evidence": {
                "max_m": int(item.get("max_m", 0)),
                "max_n": int(item.get("max_n", 0)),
                "max_k": int(item.get("max_k", 0)),
                "copy_fraction": derived_cost["tensordot_axis_permutation_copy_fraction_of_working_set"],
            },
            "recommended_action": _ProfilingRuntime._first_target(targets, "inspect_tensordot_kernel"),
        }

    @staticmethod
    def _oe_diagnosis(profile, item, derived_cost, targets):
        gemm_fraction = derived_cost["oe_gemm_flop_fraction"]
        non_gemm_fraction = derived_cost["oe_non_gemm_flop_fraction"]
        dominant_fraction = profile.get("dominant_step_flop_fraction")
        flop_split_available = bool(derived_cost.get("oe_flop_split_available"))
        if not flop_split_available:
            path_regime = _oe_type_count_path_regime(
                int(item.get("total_oe_gemm_steps", 0)),
                int(item.get("total_oe_non_gemm_steps", 0)),
            )
        elif non_gemm_fraction >= 0.5:
            path_regime = "non_gemm_flop_dominant"
        elif non_gemm_fraction >= 0.1:
            path_regime = "mixed_flop_path"
        elif gemm_fraction:
            path_regime = "gemm_flop_dominant"
        else:
            path_regime = "unknown_path"
        if item.get("total_oe_non_gemm_steps", 0) or item.get("total_oe_non_gemm_flops_estimate", 0):
            primary_issue = "non_gemm_path"
        elif item.get("max_largest_intermediate_elements", 0):
            primary_issue = "large_intermediate"
        else:
            primary_issue = "path_overhead"
        return {
            "primary_issue": primary_issue,
            "path_regime": path_regime,
            "precision_evidence": {
                "precision_level": profile.get("precision_level"),
                "flop_split_available": flop_split_available,
                "gemm_flop_fraction": gemm_fraction,
                "non_gemm_flop_fraction": non_gemm_fraction,
                "dominant_step_flop_fraction": dominant_fraction,
            },
            "recommended_action": _ProfilingRuntime._first_target(targets, "inspect_oe_contract_path"),
        }

    @staticmethod
    def _svd_diagnosis(profile, item, derived_cost, targets):
        sparse_saved = derived_cost["svd_sparse_saved_flop_fraction"]
        batchable = derived_cost["svd_batchable_flop_fraction"]
        if sparse_saved and batchable:
            block_regime = "sparse_and_batchable"
        elif sparse_saved:
            block_regime = "sparse_qn_blocks"
        elif batchable:
            block_regime = "batchable_qn_blocks"
        elif profile.get("qn_block_density") is not None:
            block_regime = "dense_like_blocks"
        else:
            block_regime = "unknown_blocks"
        if item.get("total_svd_tiny_blocks", 0):
            primary_issue = "tiny_qn_blocks"
        elif item.get("max_svd_unique_block_shapes", 0) > 1:
            primary_issue = "fragmented_qn_blocks"
        elif batchable:
            primary_issue = "batchable_qn_blocks"
        else:
            primary_issue = "blocked_decomposition"
        return {
            "primary_issue": primary_issue,
            "block_regime": block_regime,
            "precision_evidence": {
                "qn_block_density": profile.get("qn_block_density"),
                "sparse_saved_flop_fraction": sparse_saved,
                "batchable_flop_fraction": batchable,
            },
            "recommended_action": _ProfilingRuntime._first_target(targets, "inspect_qn_decomposition"),
        }

    def _practical_profile_payload(self, item, total_memory_bytes, derived_cost=None, wall_fraction=None):
        derived_cost = derived_cost or self._derived_compute_cost_payload(item, total_memory_bytes)
        signature = item.get("signature", {})
        compute_class = item.get("compute_class") or signature.get("compute_class")
        call_count = int(item.get("call_count", 0))
        dominant_kernel = self._dominant_counter_key(item.get("kernel_kind_wall_s", {}))
        top_bottlenecks = self._top_counter_items(
            item.get("bottleneck_hint_counts", {}),
            item.get("bottleneck_hint_wall_s", {}),
            priority=_BOTTLENECK_PRIORITY,
        )
        dominant_bottleneck = (
            top_bottlenecks[0]["key"]
            if top_bottlenecks
            else self._dominant_counter_key(item.get("bottleneck_hint_wall_s", {}))
        )
        signals = sorted(
            set(str(value) for value in item.get("bottleneck_hint_counts", {}))
            | set(str(value) for value in item.get("profile_tag_counts", {}))
        )
        profile = {
            "workload_class": compute_class,
            "primary_kernel": dominant_kernel,
            "dominant_bottleneck": dominant_bottleneck,
            "top_kernels": self._top_counter_items(
                item.get("kernel_kind_counts", {}),
                item.get("kernel_kind_wall_s", {}),
            ),
            "top_problem_shapes": self._top_counter_items(
                item.get("problem_shape_counts", {}),
                item.get("problem_shape_wall_s", {}),
            ),
            "top_bottlenecks": top_bottlenecks,
            "top_signals": self._top_counter_items(
                item.get("profile_tag_counts", {}),
                item.get("profile_tag_wall_s", {}),
            ),
            "signals": signals,
            "working_set_bytes_estimate": int(derived_cost["working_set_bytes_estimate"]),
            "effective_arithmetic_intensity_flops_per_byte": derived_cost[
                "effective_arithmetic_intensity_flops_per_byte"
            ],
            "total_wall_s": float(item.get("total_wall_s") or 0.0),
            "optimization_targets": [],
        }
        if wall_fraction is not None:
            profile["wall_fraction"] = wall_fraction
        execution_resources = self._summary_execution_resources(item, call_count)
        if execution_resources is not None:
            profile["execution_resources"] = execution_resources
        temporary_memory = self._summary_temporary_memory(item)
        if temporary_memory is not None:
            profile["temporary_memory"] = temporary_memory

        targets = profile["optimization_targets"]
        if item.get("fallback_calls", 0):
            self._append_unique(targets, "remove_backend_fallback")
        if item.get("total_comm_bytes", 0):
            self._append_unique(targets, "reduce_communication")
        if compute_class == COMPUTE_CLASS_TENSORDOT:
            axis_permutation_fraction = self._safe_fraction(
                item.get("total_tensordot_axis_permutations", 0), call_count
            )
            tiny_gemm_fraction = self._safe_fraction(item.get("total_tensordot_tiny_gemm", 0), call_count)
            skinny_gemm_fraction = self._safe_fraction(item.get("total_tensordot_skinny_gemm", 0), call_count)
            rhs_loop_fraction = self._safe_fraction(item.get("total_rhs_loop_calls", 0), call_count)
            profile.update({
                "gemm_call_fraction": self._safe_fraction(item.get("total_gemm", 0), call_count),
                "batched_gemm_call_fraction": self._safe_fraction(item.get("total_batched_gemm", 0), call_count),
                "rhs_loop_call_fraction": rhs_loop_fraction,
                "axis_permutation_call_fraction": axis_permutation_fraction,
                "axis_permutation_copy_fraction": derived_cost[
                    "tensordot_axis_permutation_copy_fraction_of_working_set"
                ],
                "tiny_gemm_call_fraction": tiny_gemm_fraction,
                "skinny_gemm_call_fraction": skinny_gemm_fraction,
                "max_m": int(item.get("max_m", 0)),
                "max_n": int(item.get("max_n", 0)),
                "max_k": int(item.get("max_k", 0)),
            })
            if item.get("total_rhs_loop_calls", 0):
                self._append_unique(targets, "batch_rhs_hop")
            if item.get("total_tensordot_axis_permutation_copy_bytes", 0):
                self._append_unique(targets, "avoid_axis_permutation_copies")
            if item.get("total_tensordot_tiny_gemm", 0):
                self._append_unique(targets, "batch_or_fuse_tiny_gemm")
            if item.get("total_tensordot_skinny_gemm", 0):
                self._append_unique(targets, "optimize_skinny_gemm")
            profile["diagnosis"] = self._tensordot_diagnosis(profile, item, derived_cost, targets)
            profile["measurement_scope"] = "tensordot_summary"
            profile["precision_level"] = "aggregate_mnk_layout"
            profile["dominant_cost_kind"] = profile["diagnosis"]["primary_issue"]
            profile["cost_model"] = _practical_cost_model(
                "aggregate_shape_estimate",
                flops_estimate=item.get("total_flops_estimate", 0),
                read_bytes=item.get("total_read_bytes", 0),
                write_bytes=item.get("total_write_bytes", 0),
                copy_bytes=item.get("total_copy_bytes", 0),
                communication_bytes=item.get("total_comm_bytes", 0),
                working_set_bytes=derived_cost["working_set_bytes_estimate"],
            )
            profile["dominant_work"] = {
                "unit": "tensordot_problem_shape",
                "kernel": dominant_kernel,
                "max_m": int(item.get("max_m", 0)),
                "max_n": int(item.get("max_n", 0)),
                "max_k": int(item.get("max_k", 0)),
            }
            profile["parallelization"] = {
                "unit": "backend_kernel",
                "backend_kernel": dominant_kernel,
                "batching_candidate": bool(
                    item.get("total_tensordot_tiny_gemm", 0)
                    or item.get("total_tensordot_skinny_gemm", 0)
                    or item.get("total_rhs_loop_calls", 0)
                ),
                "rhs_batching_candidate": bool(item.get("total_rhs_loop_calls", 0)),
                "call_count": call_count,
                "total_gemm": int(item.get("total_gemm", 0)),
                "total_batched_gemm": int(item.get("total_batched_gemm", 0)),
                "total_grouped_tasks": int(item.get("total_grouped_tasks", 0)),
            }
            profile["measurement_limits"] = _measurement_limits("aggregated_python_wall_time")
            profile["dominant_operation"] = {
                "kind": dominant_kernel,
                "max_m": int(item.get("max_m", 0)),
                "max_n": int(item.get("max_n", 0)),
                "max_k": int(item.get("max_k", 0)),
            }
        elif compute_class == COMPUTE_CLASS_OE:
            oe_flops = item.get("total_oe_gemm_flops_estimate", 0) + item.get(
                "total_oe_non_gemm_flops_estimate", 0
            )
            oe_flop_split_available = bool(derived_cost.get("oe_flop_split_available"))
            precision_level = (
                "aggregate_path_costs"
                if oe_flop_split_available
                else "aggregate_path_type_counts"
            )
            cost_source = (
                "aggregate_oe_path_estimate"
                if oe_flop_split_available
                else "aggregate_oe_path_type_count_estimate"
            )
            max_dominant_step_flops = item.get("max_oe_dominant_step_flops_estimate", 0)
            dominant_step_flop_fraction = (
                self._safe_fraction(max_dominant_step_flops, oe_flops)
                if oe_flop_split_available and max_dominant_step_flops
                else None
            )

            def oe_flop_fraction(value):
                return self._safe_fraction(value, oe_flops) if oe_flop_split_available else None

            profile.update({
                "gemm_step_fraction": self._safe_fraction(
                    item.get("total_oe_gemm_steps", 0), item.get("total_contraction_steps", 0)
                ),
                "tensordot_step_fraction": self._safe_fraction(
                    item.get("total_oe_tensordot_steps", 0), item.get("total_contraction_steps", 0)
                ),
                "generic_einsum_step_fraction": self._safe_fraction(
                    item.get("total_oe_generic_einsum_steps", 0), item.get("total_contraction_steps", 0)
                ),
                "non_gemm_step_fraction": self._safe_fraction(
                    item.get("total_oe_non_gemm_steps", 0), item.get("total_contraction_steps", 0)
                ),
                "gemm_flop_fraction": derived_cost["oe_gemm_flop_fraction"],
                "tensordot_flop_fraction": oe_flop_fraction(
                    item.get("total_oe_tensordot_flops_estimate", 0)
                ),
                "generic_einsum_flop_fraction": oe_flop_fraction(
                    item.get("total_oe_generic_einsum_flops_estimate", 0)
                ),
                "non_gemm_flop_fraction": derived_cost["oe_non_gemm_flop_fraction"],
                "tensordot_flops_estimate": int(item.get("total_oe_tensordot_flops_estimate", 0)),
                "generic_einsum_flops_estimate": int(item.get("total_oe_generic_einsum_flops_estimate", 0)),
                "non_gemm_flops_estimate": int(item.get("total_oe_non_gemm_flops_estimate", 0)),
                "dominant_step_flop_fraction": dominant_step_flop_fraction,
                "max_step_output_elements": int(item.get("max_oe_step_output_elements", 0)),
                "unique_step_output_shape_count": int(item.get("max_oe_unique_step_output_shapes", 0)),
                "max_largest_intermediate_elements": int(item.get("max_largest_intermediate_elements", 0)),
            })
            if item.get("total_oe_non_gemm_steps", 0) or item.get("total_oe_non_gemm_flops_estimate", 0):
                self._append_unique(targets, "reduce_non_gemm_path_cost")
            if item.get("max_largest_intermediate_elements", 0):
                self._append_unique(targets, "limit_largest_intermediate")
            if item.get("max_oe_unique_step_output_shapes", 0) > 1:
                self._append_unique(targets, "specialize_shape_diverse_paths")
            profile["measurement_scope"] = "oe_path_summary"
            profile["precision_level"] = precision_level
            profile["diagnosis"] = self._oe_diagnosis(profile, item, derived_cost, targets)
            profile["dominant_cost_kind"] = profile["diagnosis"]["primary_issue"]
            profile["cost_model"] = _practical_cost_model(
                cost_source,
                flops_estimate=item.get("total_flops_estimate", 0),
                read_bytes=item.get("total_read_bytes", 0),
                write_bytes=item.get("total_write_bytes", 0),
                copy_bytes=item.get("total_copy_bytes", 0),
                communication_bytes=item.get("total_comm_bytes", 0),
                working_set_bytes=derived_cost["working_set_bytes_estimate"],
                extra={
                    "gemm_flop_fraction": derived_cost["oe_gemm_flop_fraction"],
                    "non_gemm_flop_fraction": derived_cost["oe_non_gemm_flop_fraction"],
                    "dominant_step_flop_fraction": profile.get("dominant_step_flop_fraction"),
                },
            )
            if oe_flop_split_available:
                profile["dominant_work"] = {
                    "unit": "oe_path_step",
                    "kernel": dominant_kernel,
                    "dominant_step_flop_fraction": profile.get("dominant_step_flop_fraction"),
                    "max_step_output_elements": int(item.get("max_oe_step_output_elements", 0)),
                }
            else:
                profile["dominant_work"] = {
                    "unit": "oe_path_type_mix",
                    "contraction_count": int(item.get("total_contraction_steps", 0)),
                    "gemm_step_count": int(item.get("total_oe_gemm_steps", 0)),
                    "tensordot_step_count": int(item.get("total_oe_tensordot_steps", 0)),
                    "generic_einsum_step_count": int(item.get("total_oe_generic_einsum_steps", 0)),
                    "non_gemm_step_count": int(item.get("total_oe_non_gemm_steps", 0)),
                }
            profile["parallelization"] = {
                "unit": "sequential_path_steps_with_backend_parallel_kernels",
                "independent_path_steps": False,
                "gemm_step_count": int(item.get("total_oe_gemm_steps", 0)),
                "tensordot_step_count": int(item.get("total_oe_tensordot_steps", 0)),
                "generic_einsum_step_count": int(item.get("total_oe_generic_einsum_steps", 0)),
                "non_gemm_step_count": int(item.get("total_oe_non_gemm_steps", 0)),
                "shape_diverse": int(item.get("max_oe_unique_step_output_shapes", 0)) > 1,
            }
            profile["measurement_limits"] = _measurement_limits("aggregated_python_wall_time")
            if oe_flop_split_available:
                profile["dominant_operation"] = {
                    "kind": dominant_kernel,
                    "dominant_step_flop_fraction": profile.get("dominant_step_flop_fraction"),
                    "max_step_output_elements": int(item.get("max_oe_step_output_elements", 0)),
                }
            else:
                profile["dominant_operation"] = dict(profile["dominant_work"])
        elif compute_class == COMPUTE_CLASS_SVD:
            profile.update({
                "qn_block_density": self._safe_fraction(
                    item.get("total_svd_block_elements", 0), item.get("total_decomposition_matrix_elements", 0)
                ),
                "sparse_saved_flop_fraction": derived_cost["svd_sparse_saved_flop_fraction"],
                "batchable_flop_fraction": derived_cost["svd_batchable_flop_fraction"],
                "block_shape_reuse_fraction": item.get("max_svd_block_shape_reuse_fraction"),
                "unique_block_shape_count": int(item.get("max_svd_unique_block_shapes", 0)),
                "tiny_block_count": int(item.get("total_svd_tiny_blocks", 0)),
                "skinny_block_count": int(item.get("total_svd_skinny_blocks", 0)),
                "max_block_elements": int(item.get("max_svd_block_elements", 0)),
                "batchable_block_group_count": int(item.get("max_svd_batchable_block_group_count", 0)),
                "dominant_block_shape_flop_fraction": (
                    item.get("max_svd_dominant_block_shape_flop_fraction") or None
                ),
            })
            if item.get("total_svd_batchable_flops_estimate", 0):
                self._append_unique(targets, "batch_reused_qn_blocks")
            if item.get("total_svd_tiny_blocks", 0):
                self._append_unique(targets, "reduce_tiny_block_overhead")
            if derived_cost["svd_sparse_saved_flop_fraction"]:
                self._append_unique(targets, "preserve_qn_sparsity")
            if item.get("max_svd_unique_block_shapes", 0) > 1:
                self._append_unique(targets, "manage_block_shape_fragmentation")
            profile["diagnosis"] = self._svd_diagnosis(profile, item, derived_cost, targets)
            profile["measurement_scope"] = "qn_decomposition_summary"
            profile["precision_level"] = "aggregate_qn_block_groups"
            profile["dominant_cost_kind"] = profile["diagnosis"]["primary_issue"]
            profile["cost_model"] = _practical_cost_model(
                "aggregate_qn_block_estimate",
                flops_estimate=item.get("total_flops_estimate", 0),
                read_bytes=item.get("total_read_bytes", 0),
                write_bytes=item.get("total_write_bytes", 0),
                copy_bytes=item.get("total_copy_bytes", 0),
                communication_bytes=item.get("total_comm_bytes", 0),
                working_set_bytes=derived_cost["working_set_bytes_estimate"],
                extra={
                    "block_flop_fraction": derived_cost["svd_block_flop_fraction"],
                    "sparse_saved_flop_fraction": derived_cost["svd_sparse_saved_flop_fraction"],
                    "batchable_flop_fraction": derived_cost["svd_batchable_flop_fraction"],
                },
            )
            profile["dominant_work"] = {
                "unit": "qn_block_shape_groups",
                "kernel": dominant_kernel,
                "unique_block_shape_count": int(item.get("max_svd_unique_block_shapes", 0)),
                "batchable_block_group_count": int(item.get("max_svd_batchable_block_group_count", 0)),
                "dominant_block_shape_flop_fraction": (
                    item.get("max_svd_dominant_block_shape_flop_fraction") or None
                ),
            }
            profile["parallelization"] = {
                "unit": "independent_qn_block_groups",
                "batchable_block_group_count": int(item.get("max_svd_batchable_block_group_count", 0)),
                "batchable_block_count": int(item.get("total_svd_batchable_blocks", 0)),
                "block_count": int(item.get("total_svd_blocks", 0)),
                "unique_block_shape_count": int(item.get("max_svd_unique_block_shapes", 0)),
                "shape_fragmentation": self._safe_fraction(
                    item.get("max_svd_unique_block_shapes", 0),
                    item.get("total_svd_blocks", 0),
                ),
            }
            profile["measurement_limits"] = _measurement_limits("aggregated_python_wall_time")
            profile["dominant_operation"] = {
                "kind": dominant_kernel,
                "unique_block_shape_count": int(item.get("max_svd_unique_block_shapes", 0)),
                "batchable_block_group_count": int(item.get("max_svd_batchable_block_group_count", 0)),
                "dominant_block_shape_flop_fraction": (
                    item.get("max_svd_dominant_block_shape_flop_fraction") or None
                ),
            }
        elif compute_class == COMPUTE_CLASS_CONTRACTION_PLAN:
            lowerings = item.get("lowerings") or ()
            lowering = dominant_kernel
            if lowering is None and lowerings:
                lowering = sorted(str(value) for value in lowerings)[0]
            num_gemm = int(item.get("total_gemm", 0))
            num_batched = int(item.get("total_batched_gemm", 0))
            num_grouped = int(item.get("total_grouped_tasks", 0))
            num_blocks = int(item.get("total_blocks", 0))
            num_shape_buckets = int(item.get("total_shape_buckets", 0))
            num_hmm_gemv_shape_buckets = int(item.get("total_hmm_gemv_shape_buckets", 0))
            num_total_shape_buckets = num_shape_buckets + num_hmm_gemv_shape_buckets
            num_hmm_batches = int(item.get("total_hmm_batches", 0))
            max_hmm_batch_size = int(item.get("max_hmm_batch_size", 0))
            num_hmm_gemm_desc = int(item.get("total_hmm_gemm_desc", 0))
            num_hmm_gemv_desc = int(item.get("total_hmm_gemv_desc", 0))
            num_output_reduction_groups = int(item.get("total_output_reduction_groups", 0))
            num_scatter_add_tasks = int(item.get("total_scatter_add_tasks", 0))
            fallback_calls = int(item.get("fallback_calls", 0))
            communication_bytes = int(item.get("total_comm_bytes", 0))
            copy_bytes = int(item.get("total_copy_bytes", 0))
            if fallback_calls:
                self._append_unique(targets, "remove_backend_fallback")
            if communication_bytes:
                self._append_unique(targets, "reduce_communication")
            if copy_bytes:
                self._append_unique(targets, "reduce_layout_or_device_copies")
            if fallback_calls:
                primary_issue = "backend_fallback"
                measurement_focus = "fallback_route"
            elif communication_bytes:
                primary_issue = "communication"
                measurement_focus = "communication"
            elif copy_bytes:
                primary_issue = "copy_overhead"
                measurement_focus = "copy_overhead"
            else:
                primary_issue = "planned_backend_lowering"
                measurement_focus = "lowering_counters"
            working_set_bytes = int(derived_cost["working_set_bytes_estimate"])
            shape_bucket_reuse = self._safe_fraction(num_grouped, num_shape_buckets)
            work = {
                "unit": "backend_execution_plan",
                "lowering": lowering,
                "num_gemm": num_gemm,
                "num_batched_gemm": num_batched,
                "num_grouped_tasks": num_grouped,
                "num_blocks": num_blocks,
                "num_shape_buckets": num_shape_buckets,
                "num_gemv_shape_buckets": num_hmm_gemv_shape_buckets,
                "num_total_shape_buckets": num_total_shape_buckets,
                "num_batches": num_hmm_batches,
                "batch_size": max_hmm_batch_size,
                "num_gemm_desc": num_hmm_gemm_desc,
                "num_gemv_desc": num_hmm_gemv_desc,
                "num_output_reduction_groups": num_output_reduction_groups,
                "num_scatter_add_tasks": num_scatter_add_tasks,
                "max_m": int(item.get("max_m", 0)),
                "max_n": int(item.get("max_n", 0)),
                "max_k": int(item.get("max_k", 0)),
            }
            profile.update({
                "measurement_scope": "contraction_plan_summary",
                "precision_level": "aggregate_lowering_counters",
                "dominant_cost_kind": primary_issue,
                "primary_issue": primary_issue,
                "fallback_calls": fallback_calls,
                "num_gemm": num_gemm,
                "num_batched_gemm": num_batched,
                "num_grouped_tasks": num_grouped,
                "num_blocks": num_blocks,
                "num_shape_buckets": num_shape_buckets,
                "num_gemv_shape_buckets": num_hmm_gemv_shape_buckets,
                "num_total_shape_buckets": num_total_shape_buckets,
                "num_batches": num_hmm_batches,
                "batch_size": max_hmm_batch_size,
                "num_gemm_desc": num_hmm_gemm_desc,
                "num_gemv_desc": num_hmm_gemv_desc,
                "num_output_reduction_groups": num_output_reduction_groups,
                "num_scatter_add_tasks": num_scatter_add_tasks,
                "max_m": int(item.get("max_m", 0)),
                "max_n": int(item.get("max_n", 0)),
                "max_k": int(item.get("max_k", 0)),
                "shape_bucket_reuse": shape_bucket_reuse,
                "cost_model": _practical_cost_model(
                    "aggregate_contraction_plan_estimate",
                    flops_estimate=item.get("total_flops_estimate", 0),
                    read_bytes=item.get("total_read_bytes", 0),
                    write_bytes=item.get("total_write_bytes", 0),
                    copy_bytes=copy_bytes,
                    communication_bytes=communication_bytes,
                    working_set_bytes=working_set_bytes,
                    extra={
                        "shape_bucket_reuse": shape_bucket_reuse,
                        "fallback_call_fraction": self._safe_fraction(fallback_calls, call_count),
                    },
                ),
                "dominant_work": work,
                "parallelization": {
                    "unit": "backend_execution_plan",
                    "backend_kernel": lowering,
                    "batched_kernel": bool(
                        num_batched or lowering in ("batched_gemm", "strided_batched_gemm")
                    ),
                    "grouped_kernel": bool(
                        num_grouped or lowering in ("grouped_gemm", "block_grouped_gemm")
                    ),
                    "distributed": lowering in ("distributed", "distributed_contract"),
                    "num_gemm": num_gemm,
                    "num_batched_gemm": num_batched,
                    "num_grouped_tasks": num_grouped,
                    "num_blocks": num_blocks,
                    "num_shape_buckets": num_shape_buckets,
                    "num_batches": num_hmm_batches,
                    "batch_size": max_hmm_batch_size,
                    "num_gemm_desc": num_hmm_gemm_desc,
                    "num_gemv_desc": num_hmm_gemv_desc,
                    "shape_bucket_reuse": shape_bucket_reuse,
                    "fallback_calls": fallback_calls,
                    "communication_num_messages": int(
                        sum(item.get("comm_messages_by_collective", {}).values())
                    ),
                    "dominant_communication_collective": self._dominant_counter_key(
                        item.get("comm_wall_s_by_collective", {})
                    ),
                },
                "measurement_limits": _measurement_limits("aggregated_python_wall_time"),
                "dominant_operation": dict(work),
                "key_metrics": {
                    "lowering": lowering,
                    "num_gemm": num_gemm,
                    "num_batched_gemm": num_batched,
                    "num_grouped_tasks": num_grouped,
                    "num_blocks": num_blocks,
                    "num_shape_buckets": num_shape_buckets,
                    "num_batches": num_hmm_batches,
                    "batch_size": max_hmm_batch_size,
                    "num_gemm_desc": num_hmm_gemm_desc,
                    "num_gemv_desc": num_hmm_gemv_desc,
                    "num_output_reduction_groups": num_output_reduction_groups,
                    "num_scatter_add_tasks": num_scatter_add_tasks,
                    "fallback_calls": fallback_calls,
                    "max_output_contributions": int(item.get("max_output_contributions", 0)),
                    "communication_bytes": communication_bytes,
                    "copy_bytes": copy_bytes,
                    "working_set_bytes": working_set_bytes,
                    "shape_bucket_reuse": shape_bucket_reuse,
                },
                "diagnosis": {
                    "measurement_focus": measurement_focus,
                    "primary_issue": primary_issue,
                    "parallelism_hint": (
                        "distributed_contraction"
                        if lowering in ("distributed", "distributed_contract")
                        else (
                            "grouped_backend_kernel"
                            if num_grouped
                            else "batched_backend_kernel" if num_batched else "backend_kernel"
                        )
                    ),
                    "fallback_candidate": bool(fallback_calls),
                    "communication_bound_candidate": bool(communication_bytes),
                    "copy_bound_candidate": bool(copy_bytes),
                    "precision_evidence": {
                        "lowering": lowering,
                        "flops": int(item.get("total_flops_estimate", 0)),
                        "read_bytes": int(item.get("total_read_bytes", 0)),
                        "write_bytes": int(item.get("total_write_bytes", 0)),
                        "working_set_bytes_estimate": working_set_bytes,
                        "num_gemm": num_gemm,
                        "num_batched_gemm": num_batched,
                        "num_grouped_tasks": num_grouped,
                        "num_blocks": num_blocks,
                        "num_shape_buckets": num_shape_buckets,
                        "num_batches": num_hmm_batches,
                        "batch_size": max_hmm_batch_size,
                        "num_gemm_desc": num_hmm_gemm_desc,
                        "num_gemv_desc": num_hmm_gemv_desc,
                        "num_output_reduction_groups": num_output_reduction_groups,
                        "num_scatter_add_tasks": num_scatter_add_tasks,
                        "fallback_calls": fallback_calls,
                        "shape_bucket_reuse": shape_bucket_reuse,
                    },
                    "recommended_action": self._first_target(targets, "inspect_contraction_plan"),
                },
            })
        estimated_cost = _compact_estimated_cost(profile.get("cost_model"))
        profile["execution_breakdown"] = _practical_execution_breakdown(
            compute_class,
            profile,
            estimated_cost,
        )
        profile["parallelism_opportunity"] = _practical_parallelism_opportunity(compute_class, profile)
        profile["cost_factor_breakdown"] = _practical_cost_factor_breakdown(
            compute_class,
            profile,
            estimated_cost,
        )
        core_compute_profile = _core_compute_profile(
            compute_class,
            profile,
            estimated_cost,
            total_wall_s=item.get("total_wall_s"),
        )
        if core_compute_profile is not None:
            profile["core_compute_profile"] = core_compute_profile
        return profile

    def register_event_output(self, file_path, mode="w") -> None:
        self.close_event_output()
        self._event_output = open(os.fspath(file_path), mode, encoding="utf-8", buffering=1)

    def flush_event_output(self) -> None:
        if self._event_output is not None:
            self._event_output.flush()

    def close_event_output(self) -> None:
        if self._event_output is not None:
            self._event_output.close()
            self._event_output = None

    def _write_event(self, payload) -> None:
        if self._event_output is None:
            return
        started = self.perf_counter()
        try:
            text = self._json.dumps(self._to_jsonable(payload), sort_keys=True, separators=(",", ":"))
            self._event_output.write(text + "\n")
            self._bump_overhead("events_written", 1)
        finally:
            self._bump_overhead("event_write_overhead_s", self.perf_counter() - started)

    def flush_summaries(self) -> None:
        started = self.perf_counter()
        if not enabled():
            with self._summary_lock:
                self._summaries.clear()
                self._compute_summaries.clear()
                self._compute_class_summaries.clear()
            self._reset_overhead()
            self.flush_event_output()
            return
        with self._summary_lock:
            items = list(self._summaries.values())
            self._summaries.clear()
            compute_items = list(self._compute_summaries.values())
            self._compute_summaries.clear()
            compute_class_items = list(self._compute_class_summaries.values())
            self._compute_class_summaries.clear()
        if not items and not compute_items and not compute_class_items and not self._has_overhead_activity():
            self.flush_event_output()
            return
        for item in items:
            signature = item["signature"]
            call_count = item["call_count"]
            payload = {
                "event": "profile_summary",
                "source_event": signature.get("event"),
                "signature": signature,
                "call_count": call_count,
                "total_wall_s": item["total_wall_s"],
                "mean_wall_s": item["total_wall_s"] / call_count if call_count else None,
                "min_wall_s": item["min_wall_s"],
                "max_wall_s": item["max_wall_s"],
            }
            text = self._json.dumps(self._to_jsonable(payload), sort_keys=True, separators=(",", ":"))
            logger.profiling("%s%s", LOG_PREFIX, text)
            self._bump_overhead("summary_rows_logged", 1)
        for item in sorted(
            compute_items,
            key=lambda row: self._json.dumps(self._to_jsonable(row["signature"]), sort_keys=True),
        ):
            signature = item["signature"]
            call_count = item["call_count"]
            total_wall_s = item["total_wall_s"]
            total_flops = item["total_flops_estimate"]
            total_read_bytes = item["total_read_bytes"]
            total_write_bytes = item["total_write_bytes"]
            total_memory_bytes = total_read_bytes + total_write_bytes
            total_copy_bytes = item["total_copy_bytes"]
            total_comm_bytes = item["total_comm_bytes"]
            derived_cost = self._derived_compute_cost_payload(item, total_memory_bytes)

            def rate(value):
                return value / total_wall_s if total_wall_s > 0 else None

            practical_profile = self._practical_profile_payload(
                item, total_memory_bytes, derived_cost=derived_cost
            )
            cost_profile = _aggregate_cost_profile(item, total_memory_bytes, derived_cost)
            phase_timing = self._summary_phase_timing_payload(item)
            payload = {
                "event": "profile_compute_summary",
                **signature,
                "source_events": sorted(item["source_events"]),
                "kernel_kinds": sorted(item["kernel_kinds"]),
                "fallback_sources": sorted(item["fallback_sources"]),
                "fallback_targets": sorted(item["fallback_targets"]),
                "fallback_reasons": sorted(item["fallback_reasons"]),
                "fallback_policies": sorted(item["fallback_policies"]),
                **self._practical_summary_payload(item),
                "call_count": call_count,
                "total_wall_s": total_wall_s,
                "mean_wall_s": total_wall_s / call_count if call_count else None,
                "min_wall_s": item["min_wall_s"],
                "max_wall_s": item["max_wall_s"],
                "total_flops_estimate": int(total_flops),
                "total_read_bytes": int(total_read_bytes),
                "total_write_bytes": int(total_write_bytes),
                "total_memory_bytes": int(total_memory_bytes),
                "total_copy_bytes": int(total_copy_bytes),
                "total_comm_bytes": int(total_comm_bytes),
                "comm_bytes_by_collective": {
                    key: int(value) for key, value in sorted(item["comm_bytes_by_collective"].items())
                },
                "comm_wall_s_by_collective": dict(sorted(item["comm_wall_s_by_collective"].items())),
                "comm_messages_by_collective": {
                    key: int(value) for key, value in sorted(item["comm_messages_by_collective"].items())
                },
                "max_comm_block_size_by_collective": {
                    key: int(value) for key, value in sorted(item["max_comm_block_size_by_collective"].items())
                },
                "max_workspace_bytes": int(item["max_workspace_bytes"]),
                "max_workspace_provided_bytes": int(item["max_workspace_provided_bytes"]),
                "max_workspace_slack_bytes": int(item["max_workspace_slack_bytes"]),
                "total_stream_provided_calls": int(item["total_stream_provided_calls"]),
                "total_workspace_provided_calls": int(item["total_workspace_provided_calls"]),
                "max_peak_bytes": int(item["max_peak_bytes"]),
                "max_largest_intermediate_elements": int(item["max_largest_intermediate_elements"]),
                "max_largest_intermediate_bytes": int(item["max_largest_intermediate_bytes"]),
                "total_contraction_steps": int(item["total_contraction_steps"]),
                "total_svd_blocks": int(item["total_svd_blocks"]),
                "total_decomposition_matrix_elements": int(item["total_decomposition_matrix_elements"]),
                "total_gemm": int(item["total_gemm"]),
                "total_batched_gemm": int(item["total_batched_gemm"]),
                "total_grouped_tasks": int(item["total_grouped_tasks"]),
                "total_blocks": int(item["total_blocks"]),
                "total_shape_buckets": int(item["total_shape_buckets"]),
                "total_hmm_gemv_shape_buckets": int(item["total_hmm_gemv_shape_buckets"]),
                "total_hmm_batches": int(item["total_hmm_batches"]),
                "max_hmm_batch_size": int(item["max_hmm_batch_size"]),
                "total_hmm_gemm_desc": int(item["total_hmm_gemm_desc"]),
                "total_hmm_gemv_desc": int(item["total_hmm_gemv_desc"]),
                "total_output_reduction_groups": int(item["total_output_reduction_groups"]),
                "total_scatter_add_tasks": int(item["total_scatter_add_tasks"]),
                "max_output_contributions": int(item["max_output_contributions"]),
                "scatter_add_calls": int(item["scatter_add_calls"]),
                "max_m": int(item["max_m"]),
                "max_n": int(item["max_n"]),
                "max_k": int(item["max_k"]),
                "max_svd_block_m": int(item["max_svd_block_m"]),
                "max_svd_block_n": int(item["max_svd_block_n"]),
                "max_svd_block_elements": int(item["max_svd_block_elements"]),
                "total_svd_block_elements": int(item["total_svd_block_elements"]),
                "svd_block_density": (
                    item["total_svd_block_elements"] / item["total_decomposition_matrix_elements"]
                    if item["total_decomposition_matrix_elements"] > 0
                    else None
                ),
                "max_output_rank": int(item["max_output_rank"]),
                "total_singular_values": int(item["total_singular_values"]),
                "total_oe_gemm_steps": int(item["total_oe_gemm_steps"]),
                "total_oe_tensordot_steps": int(item["total_oe_tensordot_steps"]),
                "total_oe_generic_einsum_steps": int(item["total_oe_generic_einsum_steps"]),
                "total_oe_non_gemm_steps": int(item["total_oe_non_gemm_steps"]),
                "total_rhs_vectors": int(item["total_rhs_vectors"]),
                "max_rhs": int(item["max_rhs"]),
                "total_rhs_loop_calls": int(item["total_rhs_loop_calls"]),
                "total_tensordot_axis_permutations": int(item["total_tensordot_axis_permutations"]),
                "total_tensordot_axis_permutation_copy_bytes": int(
                    item["total_tensordot_axis_permutation_copy_bytes"]
                ),
                "total_tensordot_high_rank_free": int(item["total_tensordot_high_rank_free"]),
                "total_tensordot_tiny_gemm": int(item["total_tensordot_tiny_gemm"]),
                "total_tensordot_skinny_gemm": int(item["total_tensordot_skinny_gemm"]),
                "max_oe_step_output_elements": int(item["max_oe_step_output_elements"]),
                "total_oe_step_output_elements": int(item["total_oe_step_output_elements"]),
                "max_oe_unique_step_output_shapes": int(item["max_oe_unique_step_output_shapes"]),
                "total_oe_gemm_flops_estimate": int(item["total_oe_gemm_flops_estimate"]),
                "total_oe_tensordot_flops_estimate": int(item["total_oe_tensordot_flops_estimate"]),
                "total_oe_generic_einsum_flops_estimate": int(
                    item["total_oe_generic_einsum_flops_estimate"]
                ),
                "total_oe_non_gemm_flops_estimate": int(item["total_oe_non_gemm_flops_estimate"]),
                "max_oe_dominant_step_flops_estimate": int(item["max_oe_dominant_step_flops_estimate"]),
                "max_svd_unique_block_shapes": int(item["max_svd_unique_block_shapes"]),
                "max_svd_block_shape_reuse_fraction": item["max_svd_block_shape_reuse_fraction"],
                "total_svd_tiny_blocks": int(item["total_svd_tiny_blocks"]),
                "total_svd_skinny_blocks": int(item["total_svd_skinny_blocks"]),
                "total_svd_block_flops_estimate": int(item["total_svd_block_flops_estimate"]),
                "total_svd_dense_flops_estimate": int(item["total_svd_dense_flops_estimate"]),
                "total_svd_batchable_flops_estimate": int(item["total_svd_batchable_flops_estimate"]),
                "max_svd_batchable_flop_fraction": item["max_svd_batchable_flop_fraction"],
                "max_svd_batchable_block_group_count": int(item["max_svd_batchable_block_group_count"]),
                "max_svd_dominant_block_shape_flop_fraction": item[
                    "max_svd_dominant_block_shape_flop_fraction"
                ],
                "fallback_calls": int(item["fallback_calls"]),
                "flops_per_s_estimate": rate(total_flops),
                "read_bandwidth_Bps": rate(total_read_bytes),
                "write_bandwidth_Bps": rate(total_write_bytes),
                "memory_bandwidth_Bps": rate(total_memory_bytes),
                "copy_bandwidth_Bps": rate(total_copy_bytes),
                "comm_bandwidth_Bps": rate(total_comm_bytes),
                "arithmetic_intensity_flops_per_byte": (
                    total_flops / total_memory_bytes if total_memory_bytes > 0 else None
                ),
                **derived_cost,
                "cost_profile": cost_profile,
                "practical_profile": practical_profile,
                "compute_profile": _compute_profile_from_practical(
                    signature,
                    practical_profile,
                    event_scope="aggregate",
                    call_count=call_count,
                    total_wall_s=total_wall_s,
                ),
            }
            self._attach_summary_phase_timing(payload, phase_timing)
            text = self._json.dumps(self._to_jsonable(payload), sort_keys=True, separators=(",", ":"))
            logger.profiling("%s%s", LOG_PREFIX, text)
            self._bump_overhead("summary_rows_logged", 1)
        self._emit_compute_class_overview(compute_class_items)
        for item in sorted(
            compute_class_items,
            key=lambda row: self._json.dumps(self._to_jsonable(row["signature"]), sort_keys=True),
        ):
            signature = item["signature"]
            call_count = item["call_count"]
            total_wall_s = item["total_wall_s"]
            total_flops = item["total_flops_estimate"]
            total_read_bytes = item["total_read_bytes"]
            total_write_bytes = item["total_write_bytes"]
            total_memory_bytes = total_read_bytes + total_write_bytes
            total_copy_bytes = item["total_copy_bytes"]
            total_comm_bytes = item["total_comm_bytes"]
            derived_cost = self._derived_compute_cost_payload(item, total_memory_bytes)

            def rate(value):
                return value / total_wall_s if total_wall_s > 0 else None

            practical_profile = self._practical_profile_payload(
                item, total_memory_bytes, derived_cost=derived_cost
            )
            cost_profile = _aggregate_cost_profile(item, total_memory_bytes, derived_cost)
            phase_timing = self._summary_phase_timing_payload(item)
            payload = {
                "event": "profile_compute_class_summary",
                **signature,
                "source_events": sorted(item["source_events"]),
                "source_subclasses": sorted(item["source_subclasses"]),
                "source_roles": sorted(item["source_roles"]),
                "kernel_kinds": sorted(item["kernel_kinds"]),
                "lowerings": sorted(item["lowerings"]),
                "fallback_sources": sorted(item["fallback_sources"]),
                "fallback_targets": sorted(item["fallback_targets"]),
                "fallback_reasons": sorted(item["fallback_reasons"]),
                "fallback_policies": sorted(item["fallback_policies"]),
                **self._practical_summary_payload(item),
                "call_count": call_count,
                "total_wall_s": total_wall_s,
                "mean_wall_s": total_wall_s / call_count if call_count else None,
                "min_wall_s": item["min_wall_s"],
                "max_wall_s": item["max_wall_s"],
                "total_flops_estimate": int(total_flops),
                "total_read_bytes": int(total_read_bytes),
                "total_write_bytes": int(total_write_bytes),
                "total_memory_bytes": int(total_memory_bytes),
                "total_copy_bytes": int(total_copy_bytes),
                "total_comm_bytes": int(total_comm_bytes),
                "comm_bytes_by_collective": {
                    key: int(value) for key, value in sorted(item["comm_bytes_by_collective"].items())
                },
                "comm_wall_s_by_collective": dict(sorted(item["comm_wall_s_by_collective"].items())),
                "comm_messages_by_collective": {
                    key: int(value) for key, value in sorted(item["comm_messages_by_collective"].items())
                },
                "max_comm_block_size_by_collective": {
                    key: int(value) for key, value in sorted(item["max_comm_block_size_by_collective"].items())
                },
                "max_workspace_bytes": int(item["max_workspace_bytes"]),
                "max_workspace_provided_bytes": int(item["max_workspace_provided_bytes"]),
                "max_workspace_slack_bytes": int(item["max_workspace_slack_bytes"]),
                "total_stream_provided_calls": int(item["total_stream_provided_calls"]),
                "total_workspace_provided_calls": int(item["total_workspace_provided_calls"]),
                "max_peak_bytes": int(item["max_peak_bytes"]),
                "max_largest_intermediate_elements": int(item["max_largest_intermediate_elements"]),
                "max_largest_intermediate_bytes": int(item["max_largest_intermediate_bytes"]),
                "total_contraction_steps": int(item["total_contraction_steps"]),
                "total_svd_blocks": int(item["total_svd_blocks"]),
                "total_decomposition_matrix_elements": int(item["total_decomposition_matrix_elements"]),
                "total_gemm": int(item["total_gemm"]),
                "total_batched_gemm": int(item["total_batched_gemm"]),
                "total_grouped_tasks": int(item["total_grouped_tasks"]),
                "total_blocks": int(item["total_blocks"]),
                "total_shape_buckets": int(item["total_shape_buckets"]),
                "total_hmm_gemv_shape_buckets": int(item["total_hmm_gemv_shape_buckets"]),
                "total_hmm_batches": int(item["total_hmm_batches"]),
                "max_hmm_batch_size": int(item["max_hmm_batch_size"]),
                "total_hmm_gemm_desc": int(item["total_hmm_gemm_desc"]),
                "total_hmm_gemv_desc": int(item["total_hmm_gemv_desc"]),
                "total_output_reduction_groups": int(item["total_output_reduction_groups"]),
                "total_scatter_add_tasks": int(item["total_scatter_add_tasks"]),
                "max_output_contributions": int(item["max_output_contributions"]),
                "scatter_add_calls": int(item["scatter_add_calls"]),
                "max_m": int(item["max_m"]),
                "max_n": int(item["max_n"]),
                "max_k": int(item["max_k"]),
                "max_svd_block_m": int(item["max_svd_block_m"]),
                "max_svd_block_n": int(item["max_svd_block_n"]),
                "max_svd_block_elements": int(item["max_svd_block_elements"]),
                "total_svd_block_elements": int(item["total_svd_block_elements"]),
                "svd_block_density": (
                    item["total_svd_block_elements"] / item["total_decomposition_matrix_elements"]
                    if item["total_decomposition_matrix_elements"] > 0
                    else None
                ),
                "max_output_rank": int(item["max_output_rank"]),
                "total_singular_values": int(item["total_singular_values"]),
                "total_oe_gemm_steps": int(item["total_oe_gemm_steps"]),
                "total_oe_tensordot_steps": int(item["total_oe_tensordot_steps"]),
                "total_oe_generic_einsum_steps": int(item["total_oe_generic_einsum_steps"]),
                "total_oe_non_gemm_steps": int(item["total_oe_non_gemm_steps"]),
                "total_rhs_vectors": int(item["total_rhs_vectors"]),
                "max_rhs": int(item["max_rhs"]),
                "total_rhs_loop_calls": int(item["total_rhs_loop_calls"]),
                "total_tensordot_axis_permutations": int(item["total_tensordot_axis_permutations"]),
                "total_tensordot_axis_permutation_copy_bytes": int(
                    item["total_tensordot_axis_permutation_copy_bytes"]
                ),
                "total_tensordot_high_rank_free": int(item["total_tensordot_high_rank_free"]),
                "total_tensordot_tiny_gemm": int(item["total_tensordot_tiny_gemm"]),
                "total_tensordot_skinny_gemm": int(item["total_tensordot_skinny_gemm"]),
                "max_oe_step_output_elements": int(item["max_oe_step_output_elements"]),
                "total_oe_step_output_elements": int(item["total_oe_step_output_elements"]),
                "max_oe_unique_step_output_shapes": int(item["max_oe_unique_step_output_shapes"]),
                "total_oe_gemm_flops_estimate": int(item["total_oe_gemm_flops_estimate"]),
                "total_oe_tensordot_flops_estimate": int(item["total_oe_tensordot_flops_estimate"]),
                "total_oe_generic_einsum_flops_estimate": int(
                    item["total_oe_generic_einsum_flops_estimate"]
                ),
                "total_oe_non_gemm_flops_estimate": int(item["total_oe_non_gemm_flops_estimate"]),
                "max_oe_dominant_step_flops_estimate": int(item["max_oe_dominant_step_flops_estimate"]),
                "max_svd_unique_block_shapes": int(item["max_svd_unique_block_shapes"]),
                "max_svd_block_shape_reuse_fraction": item["max_svd_block_shape_reuse_fraction"],
                "total_svd_tiny_blocks": int(item["total_svd_tiny_blocks"]),
                "total_svd_skinny_blocks": int(item["total_svd_skinny_blocks"]),
                "total_svd_block_flops_estimate": int(item["total_svd_block_flops_estimate"]),
                "total_svd_dense_flops_estimate": int(item["total_svd_dense_flops_estimate"]),
                "total_svd_batchable_flops_estimate": int(item["total_svd_batchable_flops_estimate"]),
                "max_svd_batchable_flop_fraction": item["max_svd_batchable_flop_fraction"],
                "max_svd_batchable_block_group_count": int(item["max_svd_batchable_block_group_count"]),
                "max_svd_dominant_block_shape_flop_fraction": item[
                    "max_svd_dominant_block_shape_flop_fraction"
                ],
                "fallback_calls": int(item["fallback_calls"]),
                "flops_per_s_estimate": rate(total_flops),
                "read_bandwidth_Bps": rate(total_read_bytes),
                "write_bandwidth_Bps": rate(total_write_bytes),
                "memory_bandwidth_Bps": rate(total_memory_bytes),
                "copy_bandwidth_Bps": rate(total_copy_bytes),
                "comm_bandwidth_Bps": rate(total_comm_bytes),
                "arithmetic_intensity_flops_per_byte": (
                    total_flops / total_memory_bytes if total_memory_bytes > 0 else None
                ),
                **derived_cost,
                "cost_profile": cost_profile,
                "practical_profile": practical_profile,
                "compute_profile": _compute_profile_from_practical(
                    signature,
                    practical_profile,
                    event_scope="aggregate",
                    call_count=call_count,
                    total_wall_s=total_wall_s,
                ),
            }
            self._attach_summary_phase_timing(payload, phase_timing)
            text = self._json.dumps(self._to_jsonable(payload), sort_keys=True, separators=(",", ":"))
            logger.profiling("%s%s", LOG_PREFIX, text)
            self._bump_overhead("summary_rows_logged", 1)
        self._bump_overhead("flush_overhead_s", self.perf_counter() - started)
        self.flush_event_output()
        self._emit_overhead()

    def _emit_compute_class_overview(self, compute_class_items):
        if not compute_class_items:
            return

        by_class = {}
        for item in compute_class_items:
            signature = item["signature"]
            compute_class = signature.get("compute_class")
            if compute_class is None:
                continue
            aggregate = by_class.get(compute_class)
            if aggregate is None:
                aggregate = {
                    "compute_class": compute_class,
                    "backends": set(),
                    "devices": set(),
                    "device_kinds": set(),
                    "source_events": set(),
                    "source_subclasses": set(),
                    "source_roles": set(),
                    "kernel_kinds": set(),
                    "lowerings": set(),
                    "fallback_sources": set(),
                    "fallback_targets": set(),
                    "fallback_reasons": set(),
                    "fallback_policies": set(),
                    "call_count": 0,
                    "total_wall_s": 0.0,
                    "total_flops_estimate": 0.0,
                    "total_memory_bytes": 0.0,
                    "total_read_bytes": 0.0,
                    "total_write_bytes": 0.0,
                    "total_copy_bytes": 0.0,
                    "total_comm_bytes": 0.0,
                    "comm_bytes_by_collective": {},
                    "comm_wall_s_by_collective": {},
                    "comm_messages_by_collective": {},
                    "max_comm_block_size_by_collective": {},
                    "kernel_kind_counts": {},
                    "kernel_kind_wall_s": {},
                    "problem_size_bins": {},
                    "problem_shape_counts": {},
                    "problem_shape_wall_s": {},
                    "bottleneck_hint_counts": {},
                    "bottleneck_hint_wall_s": {},
                    "profile_tag_counts": {},
                    "profile_tag_wall_s": {},
                    "max_workspace_bytes": 0.0,
                    "max_workspace_provided_bytes": 0.0,
                    "max_workspace_slack_bytes": 0.0,
                    "max_peak_bytes": 0.0,
                    "max_largest_intermediate_elements": 0.0,
                    "max_largest_intermediate_bytes": 0.0,
                    "total_stream_provided_calls": 0.0,
                    "total_workspace_provided_calls": 0.0,
                    "total_contraction_steps": 0.0,
                    "total_svd_blocks": 0.0,
                    "total_decomposition_matrix_elements": 0.0,
                    "total_gemm": 0.0,
                    "total_batched_gemm": 0.0,
                    "total_grouped_tasks": 0.0,
                    "total_blocks": 0.0,
                    "total_shape_buckets": 0.0,
                    "total_hmm_gemv_shape_buckets": 0.0,
                    "total_output_reduction_groups": 0.0,
                    "total_scatter_add_tasks": 0.0,
                    "max_output_contributions": 0.0,
                    "scatter_add_calls": 0.0,
                    "max_m": 0.0,
                    "max_n": 0.0,
                    "max_k": 0.0,
                    "max_svd_block_m": 0.0,
                    "max_svd_block_n": 0.0,
                    "max_svd_block_elements": 0.0,
                    "total_svd_block_elements": 0.0,
                    "max_output_rank": 0.0,
                    "total_singular_values": 0.0,
                    "total_oe_gemm_steps": 0.0,
                    "total_oe_tensordot_steps": 0.0,
                    "total_oe_generic_einsum_steps": 0.0,
                    "total_oe_non_gemm_steps": 0.0,
                    "total_rhs_vectors": 0.0,
                    "max_rhs": 0.0,
                    "total_rhs_loop_calls": 0.0,
                    "total_tensordot_axis_permutations": 0.0,
                    "total_tensordot_axis_permutation_copy_bytes": 0.0,
                    "total_tensordot_high_rank_free": 0.0,
                    "total_tensordot_tiny_gemm": 0.0,
                    "total_tensordot_skinny_gemm": 0.0,
                    "max_oe_step_output_elements": 0.0,
                    "total_oe_step_output_elements": 0.0,
                    "max_oe_unique_step_output_shapes": 0.0,
                    "total_oe_gemm_flops_estimate": 0.0,
                    "total_oe_tensordot_flops_estimate": 0.0,
                    "total_oe_generic_einsum_flops_estimate": 0.0,
                    "total_oe_non_gemm_flops_estimate": 0.0,
                    "max_oe_dominant_step_flops_estimate": 0.0,
                    "max_svd_unique_block_shapes": 0.0,
                    "max_svd_block_shape_reuse_fraction": 0.0,
                    "total_svd_tiny_blocks": 0.0,
                    "total_svd_skinny_blocks": 0.0,
                    "total_svd_block_flops_estimate": 0.0,
                    "total_svd_dense_flops_estimate": 0.0,
                    "total_svd_batchable_flops_estimate": 0.0,
                    "max_svd_batchable_flop_fraction": 0.0,
                    "max_svd_batchable_block_group_count": 0.0,
                    "max_svd_dominant_block_shape_flop_fraction": 0.0,
                    "fallback_calls": 0.0,
                }
                by_class[compute_class] = aggregate
            for target, signature_key in (
                ("backends", "backend"),
                ("devices", "device"),
                ("device_kinds", "device_kind"),
            ):
                value = signature.get(signature_key)
                if value is not None:
                    aggregate[target].add(str(value))
            aggregate["source_events"].update(item["source_events"])
            aggregate["source_subclasses"].update(item["source_subclasses"])
            aggregate["source_roles"].update(item["source_roles"])
            aggregate["kernel_kinds"].update(item["kernel_kinds"])
            aggregate["lowerings"].update(item["lowerings"])
            aggregate["fallback_sources"].update(item["fallback_sources"])
            aggregate["fallback_targets"].update(item["fallback_targets"])
            aggregate["fallback_reasons"].update(item["fallback_reasons"])
            aggregate["fallback_policies"].update(item["fallback_policies"])
            aggregate["call_count"] += item["call_count"]
            aggregate["total_wall_s"] += item["total_wall_s"]
            aggregate["total_flops_estimate"] += item["total_flops_estimate"]
            aggregate["total_read_bytes"] += item["total_read_bytes"]
            aggregate["total_write_bytes"] += item["total_write_bytes"]
            aggregate["total_copy_bytes"] += item["total_copy_bytes"]
            aggregate["total_comm_bytes"] += item["total_comm_bytes"]
            aggregate["max_workspace_bytes"] = max(aggregate["max_workspace_bytes"], item["max_workspace_bytes"])
            aggregate["max_workspace_provided_bytes"] = max(
                aggregate["max_workspace_provided_bytes"], item["max_workspace_provided_bytes"]
            )
            aggregate["max_workspace_slack_bytes"] = max(
                aggregate["max_workspace_slack_bytes"], item["max_workspace_slack_bytes"]
            )
            aggregate["max_peak_bytes"] = max(aggregate["max_peak_bytes"], item["max_peak_bytes"])
            aggregate["max_largest_intermediate_elements"] = max(
                aggregate["max_largest_intermediate_elements"], item["max_largest_intermediate_elements"]
            )
            aggregate["max_largest_intermediate_bytes"] = max(
                aggregate["max_largest_intermediate_bytes"], item["max_largest_intermediate_bytes"]
            )
            aggregate["total_stream_provided_calls"] += item["total_stream_provided_calls"]
            aggregate["total_workspace_provided_calls"] += item["total_workspace_provided_calls"]
            aggregate["total_contraction_steps"] += item["total_contraction_steps"]
            aggregate["total_svd_blocks"] += item["total_svd_blocks"]
            aggregate["total_decomposition_matrix_elements"] += item["total_decomposition_matrix_elements"]
            for field in (
                "comm_bytes_by_collective",
                "comm_wall_s_by_collective",
                "comm_messages_by_collective",
                "max_comm_block_size_by_collective",
                "kernel_kind_counts",
                "kernel_kind_wall_s",
                "problem_size_bins",
                "problem_shape_counts",
                "problem_shape_wall_s",
                "bottleneck_hint_counts",
                "bottleneck_hint_wall_s",
                "profile_tag_counts",
                "profile_tag_wall_s",
            ):
                for key, value in item[field].items():
                    if field.startswith("max_"):
                        aggregate[field][key] = max(aggregate[field].get(key, 0.0), value)
                    else:
                        aggregate[field][key] = aggregate[field].get(key, 0.0) + value
            aggregate["total_gemm"] += item["total_gemm"]
            aggregate["total_batched_gemm"] += item["total_batched_gemm"]
            aggregate["total_grouped_tasks"] += item["total_grouped_tasks"]
            aggregate["total_blocks"] += item["total_blocks"]
            aggregate["total_shape_buckets"] += item["total_shape_buckets"]
            aggregate["total_hmm_gemv_shape_buckets"] += item["total_hmm_gemv_shape_buckets"]
            aggregate["total_output_reduction_groups"] += item["total_output_reduction_groups"]
            aggregate["total_scatter_add_tasks"] += item["total_scatter_add_tasks"]
            aggregate["max_output_contributions"] = max(
                aggregate["max_output_contributions"],
                item["max_output_contributions"],
            )
            aggregate["scatter_add_calls"] += item["scatter_add_calls"]
            aggregate["max_m"] = max(aggregate["max_m"], item["max_m"])
            aggregate["max_n"] = max(aggregate["max_n"], item["max_n"])
            aggregate["max_k"] = max(aggregate["max_k"], item["max_k"])
            aggregate["max_svd_block_m"] = max(aggregate["max_svd_block_m"], item["max_svd_block_m"])
            aggregate["max_svd_block_n"] = max(aggregate["max_svd_block_n"], item["max_svd_block_n"])
            aggregate["max_svd_block_elements"] = max(
                aggregate["max_svd_block_elements"], item["max_svd_block_elements"]
            )
            aggregate["total_svd_block_elements"] += item["total_svd_block_elements"]
            aggregate["max_output_rank"] = max(aggregate["max_output_rank"], item["max_output_rank"])
            aggregate["total_singular_values"] += item["total_singular_values"]
            aggregate["total_oe_gemm_steps"] += item["total_oe_gemm_steps"]
            aggregate["total_oe_tensordot_steps"] += item["total_oe_tensordot_steps"]
            aggregate["total_oe_generic_einsum_steps"] += item["total_oe_generic_einsum_steps"]
            aggregate["total_oe_non_gemm_steps"] += item["total_oe_non_gemm_steps"]
            aggregate["total_rhs_vectors"] += item["total_rhs_vectors"]
            aggregate["max_rhs"] = max(aggregate["max_rhs"], item["max_rhs"])
            aggregate["total_rhs_loop_calls"] += item["total_rhs_loop_calls"]
            aggregate["total_tensordot_axis_permutations"] += item["total_tensordot_axis_permutations"]
            aggregate["total_tensordot_axis_permutation_copy_bytes"] += item[
                "total_tensordot_axis_permutation_copy_bytes"
            ]
            aggregate["total_tensordot_high_rank_free"] += item["total_tensordot_high_rank_free"]
            aggregate["total_tensordot_tiny_gemm"] += item["total_tensordot_tiny_gemm"]
            aggregate["total_tensordot_skinny_gemm"] += item["total_tensordot_skinny_gemm"]
            aggregate["max_oe_step_output_elements"] = max(
                aggregate["max_oe_step_output_elements"], item["max_oe_step_output_elements"]
            )
            aggregate["total_oe_step_output_elements"] += item["total_oe_step_output_elements"]
            aggregate["max_oe_unique_step_output_shapes"] = max(
                aggregate["max_oe_unique_step_output_shapes"], item["max_oe_unique_step_output_shapes"]
            )
            aggregate["total_oe_gemm_flops_estimate"] += item["total_oe_gemm_flops_estimate"]
            aggregate["total_oe_tensordot_flops_estimate"] += item["total_oe_tensordot_flops_estimate"]
            aggregate["total_oe_generic_einsum_flops_estimate"] += item[
                "total_oe_generic_einsum_flops_estimate"
            ]
            aggregate["total_oe_non_gemm_flops_estimate"] += item["total_oe_non_gemm_flops_estimate"]
            aggregate["max_oe_dominant_step_flops_estimate"] = max(
                aggregate["max_oe_dominant_step_flops_estimate"],
                item["max_oe_dominant_step_flops_estimate"],
            )
            aggregate["max_svd_unique_block_shapes"] = max(
                aggregate["max_svd_unique_block_shapes"], item["max_svd_unique_block_shapes"]
            )
            aggregate["max_svd_block_shape_reuse_fraction"] = max(
                aggregate["max_svd_block_shape_reuse_fraction"], item["max_svd_block_shape_reuse_fraction"]
            )
            aggregate["total_svd_tiny_blocks"] += item["total_svd_tiny_blocks"]
            aggregate["total_svd_skinny_blocks"] += item["total_svd_skinny_blocks"]
            aggregate["total_svd_block_flops_estimate"] += item["total_svd_block_flops_estimate"]
            aggregate["total_svd_dense_flops_estimate"] += item["total_svd_dense_flops_estimate"]
            aggregate["total_svd_batchable_flops_estimate"] += item["total_svd_batchable_flops_estimate"]
            aggregate["max_svd_batchable_flop_fraction"] = max(
                aggregate["max_svd_batchable_flop_fraction"], item["max_svd_batchable_flop_fraction"]
            )
            aggregate["max_svd_batchable_block_group_count"] = max(
                aggregate["max_svd_batchable_block_group_count"],
                item["max_svd_batchable_block_group_count"],
            )
            aggregate["max_svd_dominant_block_shape_flop_fraction"] = max(
                aggregate["max_svd_dominant_block_shape_flop_fraction"],
                item["max_svd_dominant_block_shape_flop_fraction"],
            )
            aggregate["fallback_calls"] += item["fallback_calls"]

        rows = sorted(by_class.values(), key=lambda row: row["total_wall_s"], reverse=True)
        total_wall_s = sum(row["total_wall_s"] for row in rows)
        for row in rows:
            row["total_memory_bytes"] = row["total_read_bytes"] + row["total_write_bytes"]
            row["wall_fraction"] = row["total_wall_s"] / total_wall_s if total_wall_s > 0 else None
            class_wall_s = row["total_wall_s"]
            row["flops_per_s_estimate"] = (
                row["total_flops_estimate"] / class_wall_s if class_wall_s > 0 else None
            )
            row["read_bandwidth_Bps"] = (
                row["total_read_bytes"] / class_wall_s if class_wall_s > 0 else None
            )
            row["write_bandwidth_Bps"] = (
                row["total_write_bytes"] / class_wall_s if class_wall_s > 0 else None
            )
            row["memory_bandwidth_Bps"] = (
                row["total_memory_bytes"] / class_wall_s if class_wall_s > 0 else None
            )
            row["copy_bandwidth_Bps"] = (
                row["total_copy_bytes"] / class_wall_s if class_wall_s > 0 else None
            )
            row["comm_bandwidth_Bps"] = (
                row["total_comm_bytes"] / class_wall_s if class_wall_s > 0 else None
            )
            row["arithmetic_intensity_flops_per_byte"] = (
                row["total_flops_estimate"] / row["total_memory_bytes"]
                if row["total_memory_bytes"] > 0
                else None
            )
            row["svd_block_density"] = (
                row["total_svd_block_elements"] / row["total_decomposition_matrix_elements"]
                if row["total_decomposition_matrix_elements"] > 0
                else None
            )
            for key in (
                "backends",
                "devices",
                "device_kinds",
                "source_events",
                "source_subclasses",
                "source_roles",
                "kernel_kinds",
                "lowerings",
                "fallback_sources",
                "fallback_targets",
                "fallback_reasons",
                "fallback_policies",
            ):
                row[key] = sorted(row[key])
            for key in (
                "comm_bytes_by_collective",
                "comm_messages_by_collective",
                "max_comm_block_size_by_collective",
            ):
                row[key] = {dict_key: int(value) for dict_key, value in sorted(row[key].items())}
            row["comm_wall_s_by_collective"] = dict(sorted(row["comm_wall_s_by_collective"].items()))
            row.update(self._practical_summary_payload(row))
            for key in (
                "call_count",
                "total_flops_estimate",
                "total_memory_bytes",
                "total_read_bytes",
                "total_write_bytes",
                "total_copy_bytes",
                "total_comm_bytes",
                "max_workspace_bytes",
                "max_workspace_provided_bytes",
                "max_workspace_slack_bytes",
                "total_stream_provided_calls",
                "total_workspace_provided_calls",
                "max_peak_bytes",
                "max_largest_intermediate_elements",
                "max_largest_intermediate_bytes",
                "total_contraction_steps",
                "total_svd_blocks",
                "total_decomposition_matrix_elements",
                "total_gemm",
                "total_batched_gemm",
                "total_grouped_tasks",
                "total_blocks",
                "total_shape_buckets",
                "total_hmm_gemv_shape_buckets",
                "total_output_reduction_groups",
                "total_scatter_add_tasks",
                "max_output_contributions",
                "scatter_add_calls",
                "max_m",
                "max_n",
                "max_k",
                "max_svd_block_m",
                "max_svd_block_n",
                "max_svd_block_elements",
                "total_svd_block_elements",
                "max_output_rank",
                "total_singular_values",
                "total_oe_gemm_steps",
                "total_oe_tensordot_steps",
                "total_oe_generic_einsum_steps",
                "total_oe_non_gemm_steps",
                "total_rhs_vectors",
                "max_rhs",
                "total_rhs_loop_calls",
                "total_tensordot_axis_permutations",
                "total_tensordot_axis_permutation_copy_bytes",
                "total_tensordot_high_rank_free",
                "total_tensordot_tiny_gemm",
                "total_tensordot_skinny_gemm",
                "max_oe_step_output_elements",
                "total_oe_step_output_elements",
                "max_oe_unique_step_output_shapes",
                "total_oe_gemm_flops_estimate",
                "total_oe_tensordot_flops_estimate",
                "total_oe_generic_einsum_flops_estimate",
                "total_oe_non_gemm_flops_estimate",
                "max_oe_dominant_step_flops_estimate",
                "max_svd_unique_block_shapes",
                "total_svd_tiny_blocks",
                "total_svd_skinny_blocks",
                "total_svd_block_flops_estimate",
                "total_svd_dense_flops_estimate",
                "total_svd_batchable_flops_estimate",
                "max_svd_batchable_block_group_count",
                "fallback_calls",
            ):
                row[key] = int(row[key])
            derived_cost = self._derived_compute_cost_payload(row, row["total_memory_bytes"])
            row.update(derived_cost)
            row["cost_profile"] = _aggregate_cost_profile(row, row["total_memory_bytes"], derived_cost)
            row["practical_profile"] = self._practical_profile_payload(
                row,
                row["total_memory_bytes"],
                derived_cost=derived_cost,
                wall_fraction=row.get("wall_fraction"),
            )
            row["compute_profile"] = _compute_profile_from_practical(
                row,
                row["practical_profile"],
                event_scope="aggregate",
                call_count=row["call_count"],
                total_wall_s=row["total_wall_s"],
                wall_fraction=row.get("wall_fraction"),
            )
            core_operation_action = self._core_operation_action_from_row(row)
            if core_operation_action is not None:
                row["core_operation_action"] = core_operation_action
            cost_factors = self._row_cost_factors(row)
            if cost_factors:
                row["cost_factor_ranking"] = cost_factors
                row["top_cost_factor"] = cost_factors[0]

        payload = {
            "event": "profile_compute_class_overview",
            "accounting": COMPUTE_ACCOUNTING_PRIMARY,
            "class_count": len(rows),
            "total_wall_s": total_wall_s,
            "core_operation_ranking": self._core_operation_ranking(rows),
            "core_operation_factor_ranking": self._core_operation_factor_ranking(rows),
            "core_operation_action_plan": self._core_operation_action_plan(rows),
            "classes": rows,
        }
        text = self._json.dumps(self._to_jsonable(payload), sort_keys=True, separators=(",", ":"))
        logger.profiling("%s%s", LOG_PREFIX, text)
        self._bump_overhead("summary_rows_logged", 1)

    @staticmethod
    def _core_operation_ranking(rows):
        ranking = []
        for row in rows:
            compute_profile = row.get("compute_profile") or {}
            core_profile = compute_profile.get("core_compute_profile") or {}
            core_summary = core_profile.get("core_kernel_summary") or {}
            practical_view = core_summary.get("practical_view") or {}
            core_operation = practical_view.get("core_operation")
            if core_operation is None:
                continue
            wall_time_ranking = list(practical_view.get("wall_time_ranking") or ())
            diagnostic_signals = list(practical_view.get("diagnostic_signals") or ())
            item = {
                "compute_class": row.get("compute_class"),
                "core_operation": core_operation,
                "total_wall_s": float(row.get("total_wall_s") or 0.0),
                "wall_fraction": row.get("wall_fraction"),
                "dominant_kernel": practical_view.get("dominant_kernel"),
                "rank_basis": practical_view.get("rank_basis"),
                "primary_kernel_count": len(wall_time_ranking),
                "diagnostic_signal_count": len(diagnostic_signals),
            }
            if wall_time_ranking:
                top = wall_time_ranking[0]
                item["top_wall_time_kernel"] = {
                    key: top[key]
                    for key in ("kind", "count", "wall_s_estimate", "wall_fraction")
                    if key in top
                }
            ranking.append(item)
        return sorted(
            ranking,
            key=lambda item: (-item["total_wall_s"], str(item.get("core_operation"))),
        )

    @staticmethod
    def _row_cost_factors(row, limit=5):
        practical_profile = row.get("practical_profile") or {}
        breakdown = practical_profile.get("cost_factor_breakdown") or {}
        factors = breakdown.get("factors") or ()
        return [
            dict(factor)
            for factor in factors[:limit]
            if isinstance(factor, dict)
        ]

    @classmethod
    def _core_operation_factor_from_row(cls, row):
        compute_profile = row.get("compute_profile") or {}
        core_profile = compute_profile.get("core_compute_profile") or {}
        core_summary = core_profile.get("core_kernel_summary") or {}
        practical_view = core_summary.get("practical_view") or {}
        core_operation = practical_view.get("core_operation")
        cost_factors = cls._row_cost_factors(row)
        if core_operation is None or not cost_factors:
            return None
        return {
            "compute_class": row.get("compute_class"),
            "core_operation": core_operation,
            "total_wall_s": float(row.get("total_wall_s") or 0.0),
            "wall_fraction": row.get("wall_fraction"),
            "rank_basis": "cost_factor_breakdown",
            "factor_count": len(cost_factors),
            "top_factor": cost_factors[0],
            "factors": cost_factors,
        }

    @classmethod
    def _core_operation_factor_ranking(cls, rows):
        ranking = []
        for row in rows:
            item = cls._core_operation_factor_from_row(row)
            if item is not None:
                ranking.append(item)
        ranking = sorted(ranking, key=lambda item: (-item["total_wall_s"], str(item.get("core_operation"))))
        return [
            {
                **item,
                "rank": index,
            }
            for index, item in enumerate(ranking, 1)
        ]

    @staticmethod
    def _core_operation_action_from_row(row):
        compute_profile = row.get("compute_profile") or {}
        core_profile = compute_profile.get("core_compute_profile") or {}
        core_summary = core_profile.get("core_kernel_summary") or {}
        practical_view = core_summary.get("practical_view") or {}
        optimization = practical_view.get("optimization_summary") or {}
        core_operation = practical_view.get("core_operation")
        recommended_focus = optimization.get("recommended_focus")
        if core_operation is None or recommended_focus is None:
            return None
        return {
            "compute_class": row.get("compute_class"),
            "core_operation": core_operation,
            "total_wall_s": float(row.get("total_wall_s") or 0.0),
            "wall_fraction": row.get("wall_fraction"),
            "primary_bottleneck": optimization.get("primary_bottleneck"),
            "recommended_focus": recommended_focus,
            "diagnostic_focuses": list(optimization.get("diagnostic_focuses") or ()),
            "parallelization_hint": optimization.get("parallelization_hint"),
            "diagnostic_signal_count": int(optimization.get("diagnostic_signal_count") or 0),
            "evidence_kernel": practical_view.get("dominant_kernel"),
            "rank_basis": practical_view.get("rank_basis"),
        }

    @classmethod
    def _core_operation_action_plan(cls, rows):
        actions = []
        for row in rows:
            action = row.get("core_operation_action")
            if action is None:
                action = cls._core_operation_action_from_row(row)
            if action is not None:
                action = {
                    **action,
                    "total_wall_s": float(action.get("total_wall_s") or 0.0),
                }
                actions.append(action)
        actions = sorted(actions, key=lambda item: (-item["total_wall_s"], str(item.get("core_operation"))))
        return [
            {
                **action,
                "rank": index,
            }
            for index, action in enumerate(actions, 1)
        ]

    def _emit_overhead(self):
        with self._overhead_lock:
            payload = {
                "event": "profile_overhead",
                **self._overhead,
            }
            self._reset_overhead_unlocked()
        text = self._json.dumps(self._to_jsonable(payload), sort_keys=True, separators=(",", ":"))
        logger.profiling("%s%s", LOG_PREFIX, text)

    def _has_overhead_activity(self):
        with self._overhead_lock:
            return any(self._overhead.values())

    def _reset_overhead(self):
        with self._overhead_lock:
            self._reset_overhead_unlocked()

    def _reset_overhead_unlocked(self):
        for key in self._overhead:
            self._overhead[key] = 0.0 if key.endswith("_s") else 0


def _get_runtime():
    global _runtime
    if _runtime is None:
        import atexit

        _runtime = _ProfilingRuntime()
        atexit.register(flush_summaries)
    return _runtime


def _default_event_output():
    global _default_event_output_path
    if _default_event_output_path is None:
        import datetime

        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        _default_event_output_path = Path.cwd() / f"profiling-{stamp}.jsonl"
    return _default_event_output_path


def record(event: str, **fields) -> None:
    if enabled():
        _get_runtime().record(event, **fields)


def standardize_event_payload(payload_or_event, **fields):
    if isinstance(payload_or_event, dict):
        payload = dict(payload_or_event)
        if fields:
            payload.update(fields)
    else:
        payload = {"event": payload_or_event, **fields}
    payload = _standard_contraction_execute_payload(payload)
    payload = _standard_contraction_plan_payload(payload)
    payload = _standard_hmm_task_build_payload(payload)
    payload = _standard_grouped_gemm_execute_payload(payload)
    payload = _standard_matmul_execute_payload(payload)
    payload = _standard_grouped_gemm_prepack_payload(payload)
    if payload.get("event") in _JSONL_EVENTS:
        payload = _refresh_event_compute_profile(payload)
    return payload


def push_scope(**fields):
    if not enabled():
        return None
    return _get_runtime().push_scope(**fields)


def pop_scope(token) -> None:
    if token is not None and _runtime is not None:
        _runtime.pop_scope(token)


@contextlib.contextmanager
def span(name: str, **fields):
    if not enabled():
        yield None
        return
    runtime = _get_runtime()
    token, span_id = runtime.push_span(name, **fields)
    try:
        yield span_id
    finally:
        runtime.pop_scope(token)


@contextlib.contextmanager
def scope(**fields):
    if not enabled():
        yield
        return
    token = push_scope(**fields)
    try:
        yield
    finally:
        pop_scope(token)


@contextlib.contextmanager
def timed(event: str, **fields):
    if not enabled():
        yield
        return
    runtime = _get_runtime()
    started = runtime.perf_counter()
    try:
        yield
    finally:
        runtime.record(event, wall_s=runtime.perf_counter() - started, **fields)


def register_event_output(file_path=None, mode="w"):
    path = Path(file_path) if file_path is not None else _default_event_output()
    _get_runtime().register_event_output(path, mode=mode)
    return path


def flush_event_output() -> None:
    if _runtime is not None:
        _runtime.flush_event_output()


def close_event_output() -> None:
    if _runtime is not None:
        _runtime.close_event_output()


def flush_summaries() -> None:
    if _runtime is not None:
        _runtime.flush_summaries()
