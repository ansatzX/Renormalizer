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


def array_dtype_names(values):
    return [str(getattr(value, "dtype", None)) for value in values if hasattr(value, "shape")]


def _prod(values):
    result = 1
    for value in values:
        result *= int(value)
    return int(result)


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
    left_free_shape = tuple(dim for index, dim in enumerate(left_shape) if index not in left_axis_set)
    right_free_shape = tuple(dim for index, dim in enumerate(right_shape) if index not in right_axis_set)
    left_contract_shape = tuple(left_shape[index] for index in left_axes)
    right_contract_shape = tuple(right_shape[index] for index in right_axes)
    contracted_shape = left_contract_shape if left_contract_shape == right_contract_shape else ()
    m = _prod(left_free_shape)
    n = _prod(right_free_shape)
    k = _prod(left_contract_shape) if left_contract_shape == right_contract_shape else 0
    flops = 2 * m * n * k if k else 0
    return {
        "compute_class": "tensordot",
        "compute_subclass": "direct_tensordot",
        "input_dtypes": array_dtype_names((left, right)),
        "output_dtype": str(getattr(result, "dtype", None)),
        "left_free_shape": left_free_shape,
        "right_free_shape": right_free_shape,
        "contracted_shape": contracted_shape,
        "m": int(m),
        "n": int(n),
        "k": int(k),
        "equivalent_gemm": bool(k or left_contract_shape == right_contract_shape),
        "flops_estimate": int(flops),
        "read_bytes": array_total_bytes((left, right)),
        "write_bytes": int(getattr(result, "nbytes", 0)),
    }


def device_payload(device):
    return {
        "kind": getattr(device, "kind", None),
        "index": getattr(device, "index", None),
        "local_rank": getattr(device, "local_rank", None),
        "global_rank": getattr(device, "global_rank", None),
        "visible_id": getattr(device, "visible_id", None),
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


def oe_compute_payload(subclass, operands, result, path_summary=None):
    path_summary = path_summary or {}
    return {
        "compute_class": "oe",
        "compute_subclass": subclass,
        "input_dtypes": array_dtype_names(operands),
        "output_dtype": str(getattr(result, "dtype", None)),
        "flops_estimate": json_int(path_summary.get("flop_count")) or 0,
        "read_bytes": array_total_bytes(operands),
        "write_bytes": int(getattr(result, "nbytes", 0)),
    }


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


def svd_qn_compute_payload(mode, coef_array, coef_matrix, blocks, outputs):
    block_flops = sum(
        decomposition_flops_estimate(block.get("block_shape", ()), mode)
        for block in (blocks or [])
    )
    output_arrays = tuple(array for array in outputs if hasattr(array, "shape"))
    return {
        "compute_class": "svd",
        "compute_subclass": _decomposition_subclass(mode),
        "input_dtype": str(getattr(coef_array, "dtype", None)),
        "output_dtype": str(getattr(output_arrays[0], "dtype", None)) if output_arrays else None,
        "flops_estimate": int(block_flops or decomposition_flops_estimate(getattr(coef_matrix, "shape", ()), mode)),
        "read_bytes": int(getattr(coef_array, "nbytes", 0)),
        "write_bytes": array_total_bytes(output_arrays),
    }


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
            if event in _JSONL_EVENTS:
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

    def _compute_signature_payload(self, payload):
        if not payload.get("compute_class"):
            return None
        keys = (
            "compute_class",
            "compute_subclass",
            "backend",
            "stage",
            "method",
            "lowering",
            "device",
            "device_kind",
        )
        return {
            key: payload[key]
            for key in keys
            if key in payload and payload[key] is not None
        }

    def _record_compute_summary_unlocked(self, payload, signature, wall_s):
        key = self._json.dumps(self._to_jsonable(signature), sort_keys=True, separators=(",", ":"))
        source_event = payload.get("event")
        flops = self._number(payload, "flops_estimate", "flops")
        read_bytes = self._number(payload, "read_bytes")
        write_bytes = self._number(payload, "write_bytes")
        item = self._compute_summaries.get(key)
        if item is None:
            self._compute_summaries[key] = {
                "signature": signature,
                "source_events": {str(source_event)} if source_event is not None else set(),
                "call_count": 1,
                "total_wall_s": wall_s,
                "min_wall_s": wall_s,
                "max_wall_s": wall_s,
                "total_flops_estimate": flops,
                "total_read_bytes": read_bytes,
                "total_write_bytes": write_bytes,
            }
            return
        if source_event is not None:
            item["source_events"].add(str(source_event))
        item["call_count"] += 1
        item["total_wall_s"] += wall_s
        item["min_wall_s"] = min(item["min_wall_s"], wall_s)
        item["max_wall_s"] = max(item["max_wall_s"], wall_s)
        item["total_flops_estimate"] += flops
        item["total_read_bytes"] += read_bytes
        item["total_write_bytes"] += write_bytes

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
            self._reset_overhead()
            self.flush_event_output()
            return
        with self._summary_lock:
            items = list(self._summaries.values())
            self._summaries.clear()
            compute_items = list(self._compute_summaries.values())
            self._compute_summaries.clear()
        if not items and not compute_items and not self._has_overhead_activity():
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
            payload = {
                "event": "profile_compute_summary",
                **signature,
                "source_events": sorted(item["source_events"]),
                "call_count": call_count,
                "total_wall_s": item["total_wall_s"],
                "mean_wall_s": item["total_wall_s"] / call_count if call_count else None,
                "min_wall_s": item["min_wall_s"],
                "max_wall_s": item["max_wall_s"],
                "total_flops_estimate": int(item["total_flops_estimate"]),
                "total_read_bytes": int(item["total_read_bytes"]),
                "total_write_bytes": int(item["total_write_bytes"]),
            }
            text = self._json.dumps(self._to_jsonable(payload), sort_keys=True, separators=(",", ":"))
            logger.profiling("%s%s", LOG_PREFIX, text)
            self._bump_overhead("summary_rows_logged", 1)
        self._bump_overhead("flush_overhead_s", self.perf_counter() - started)
        self.flush_event_output()
        self._emit_overhead()

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
