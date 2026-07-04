# -*- coding: utf-8 -*-

import time

from renormalizer.mps.backend import backend
from renormalizer.mps.matrix import asxp
from renormalizer.mps.oe_contract_wrap import oe_contract_expression
from renormalizer.utils import profiling


def _record_hop_expr(expr, nsite, ancilla, twolayer, ltensor, rtensor, cmo, cshape, started):
    profiling.record(
        "hop_expr",
        nsite=nsite,
        ancilla=ancilla if nsite != 0 else None,
        twolayer=twolayer,
        l_shape=tuple(ltensor.shape),
        r_shape=tuple(rtensor.shape),
        mpo_shapes=[tuple(item.shape) for item in cmo],
        cshape=tuple(cshape),
        wall_s=time.perf_counter() - started,
    )
    return expr


def _record_hmm_scaffold(ltensor, rtensor, cmo, cshape, nrhs=None):
    try:
        dtype = getattr(ltensor, "dtype", None)
        if len(cmo) == 1:
            from renormalizer.backend.gemm import (
                build_batched_single_site_hmm_scaffold_from_shapes,
                build_single_site_hmm_scaffold_from_shapes,
                record_hmm_contraction_plan,
            )

            if nrhs is None:
                _hxlist, task = build_single_site_hmm_scaffold_from_shapes(
                    tuple(ltensor.shape),
                    tuple(cmo[0].shape),
                    tuple(rtensor.shape),
                    tuple(cshape),
                    dtype=dtype,
                )
            else:
                _hxlist, task = build_batched_single_site_hmm_scaffold_from_shapes(
                    tuple(ltensor.shape),
                    tuple(cmo[0].shape),
                    tuple(rtensor.shape),
                    tuple(cshape),
                    nrhs=nrhs,
                    dtype=dtype,
                )
            return _hmm_profile_metadata(record_hmm_contraction_plan(task), task)
        elif len(cmo) == 2:
            from renormalizer.backend.gemm import (
                build_batched_two_site_hmm_scaffold_from_shapes,
                build_two_site_hmm_scaffold_from_shapes,
                record_hmm_contraction_plan,
            )

            if nrhs is None:
                _hxlist, task = build_two_site_hmm_scaffold_from_shapes(
                    tuple(ltensor.shape),
                    tuple(cmo[0].shape),
                    tuple(cmo[1].shape),
                    tuple(rtensor.shape),
                    tuple(cshape),
                    dtype=dtype,
                )
            else:
                _hxlist, task = build_batched_two_site_hmm_scaffold_from_shapes(
                    tuple(ltensor.shape),
                    tuple(cmo[0].shape),
                    tuple(cmo[1].shape),
                    tuple(rtensor.shape),
                    tuple(cshape),
                    nrhs=nrhs,
                    dtype=dtype,
                )
            return _hmm_profile_metadata(record_hmm_contraction_plan(task), task)
    except Exception as exc:
        if str(exc).startswith("HMM profiling execution metadata failed"):
            raise
        raise RuntimeError(
            "HMM profiling scaffold failed for nsite={0}; H-v telemetry would be incomplete".format(len(cmo))
        ) from exc
    return {}


def _hmm_profile_metadata(plan, task):
    if plan is None:
        return {}
    step_lowerings = [str(step.kind) for step in plan.steps]
    num_grouped_tasks = sum(
        len(getattr(step.plan, "descs", ()) or ())
        for step in plan.steps
        if step.kind == "grouped_gemm"
    )
    dtype = getattr(plan.input_specs[0].array, "dtype", None) if plan.input_specs else None
    itemsize = _dtype_itemsize(dtype)
    workspace_bytes = int(getattr(task, "workspace_size", 0)) * itemsize
    largest_intermediate_elements = int(getattr(task, "inter_size", 0))
    largest_intermediate_bytes = largest_intermediate_elements * itemsize
    largest_intermediate = max(
        int(getattr(plan, "estimated_peak_bytes", 0)),
        workspace_bytes,
        largest_intermediate_bytes,
    )
    try:
        from renormalizer.backend.gemm import hmm_task_execution_metadata

        execution_metadata = hmm_task_execution_metadata(plan, task)
    except Exception as exc:
        raise RuntimeError(
            "HMM profiling execution metadata failed; H-v telemetry would be incomplete"
        ) from exc
    return {
        "hmm_plan_hash": plan.plan_hash,
        "hmm_matmul_plan_hashes": [
            step.plan.plan_hash
            for step in plan.steps
            if getattr(step, "plan", None) is not None
        ],
        "hmm_step_lowerings": step_lowerings,
        "hmm_num_shape_buckets": sum(
            len(stage_batch.groups)
            for task_batch in task.batches
            for stage_batch in task_batch
        ),
        "hmm_num_batched_gemm": sum(
            1
            for lowering in step_lowerings
            if lowering in ("batched_gemm", "strided_batched_gemm")
        ),
        "hmm_num_gemm": sum(1 for lowering in step_lowerings if lowering == "gemm"),
        "hmm_num_grouped_tasks": num_grouped_tasks,
        "hmm_num_blocks": int(getattr(task, "num_hx_blocks", 0)),
        "hmm_flops": int(getattr(plan, "estimated_flops", 0)),
        "hmm_read_bytes": int(getattr(plan, "estimated_read_bytes", 0)),
        "hmm_write_bytes": int(getattr(plan, "estimated_write_bytes", 0)),
        "hmm_copy_bytes": int(getattr(plan, "estimated_copy_bytes", 0)),
        "hmm_workspace_bytes": max(int(getattr(plan, "required_workspace_bytes", 0)), workspace_bytes),
        "hmm_largest_intermediate": largest_intermediate,
        "hmm_largest_intermediate_elements": largest_intermediate_elements,
        "hmm_largest_intermediate_bytes": largest_intermediate_bytes,
        "hmm_fallback_reason": next(
            (step.fallback_reason for step in plan.steps if step.fallback_reason),
            None,
        ),
        **execution_metadata,
        **_hmm_rhs_profile_metadata(task),
    }


def _hmm_rhs_profile_metadata(task):
    equation = getattr(task, "equation", None)
    output_shape = tuple(getattr(task, "output_shape", ()) or ())
    if not equation or "->" not in equation:
        return {}
    output_modes = equation.split("->", 1)[1].replace(" ", "")
    if not output_modes or output_modes[-1] != "r" or not output_shape:
        return {}
    return {
        "hmm_rhs_batch_mode": output_modes[-1],
        "hmm_num_rhs": int(output_shape[-1]),
        "hmm_num_rhs_loop_calls": 0,
    }


def _attach_hmm_profile_metadata(expr, metadata):
    if not metadata:
        return expr
    for key, value in metadata.items():
        setattr(expr, key, value)
    return expr


def _dtype_itemsize(dtype):
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is not None:
        return int(itemsize)
    if dtype is None:
        return 0
    try:
        import numpy as np

        return int(np.dtype(dtype).itemsize)
    except Exception:
        return 0


def _hmm_execution_profile_fields(expr):
    plan_hash = getattr(expr, "hmm_plan_hash", None)
    if plan_hash is None:
        return {}
    fields = {
        "hmm_plan_hash": plan_hash,
        "hmm_matmul_plan_hashes": list(getattr(expr, "hmm_matmul_plan_hashes", ())),
        "hmm_step_lowerings": list(getattr(expr, "hmm_step_lowerings", ())),
        "hmm_num_shape_buckets": int(getattr(expr, "hmm_num_shape_buckets", 0)),
        "hmm_num_batched_gemm": int(getattr(expr, "hmm_num_batched_gemm", 0)),
    }
    for name in (
        "hmm_flops",
        "hmm_read_bytes",
        "hmm_write_bytes",
        "hmm_copy_bytes",
        "hmm_workspace_bytes",
        "hmm_largest_intermediate",
        "hmm_steps",
        "hmm_shape_buckets",
        "hmm_gemv_shape_buckets",
        "hmm_hx_blocks",
        "hmm_batches",
        "hmm_gemv_batches",
        "hmm_execution_trace",
        "hmm_num_batches",
        "hmm_batch_size",
        "hmm_num_gemv_desc",
        "hmm_center_kind",
        "hmm_direct_intermediate",
    ):
        if hasattr(expr, name):
            fields[name] = getattr(expr, name)
    if hasattr(expr, "hmm_rhs_batch_mode"):
        rhs_batch_mode = str(getattr(expr, "hmm_rhs_batch_mode"))
        fields["rhs_batch_mode"] = rhs_batch_mode
        fields["hmm_rhs_batch_mode"] = rhs_batch_mode
    if hasattr(expr, "hmm_num_rhs"):
        num_rhs = int(getattr(expr, "hmm_num_rhs", 0))
        fields["num_rhs"] = num_rhs
        fields["hmm_num_rhs"] = num_rhs
    if hasattr(expr, "hmm_num_rhs_loop_calls"):
        num_rhs_loop_calls = int(getattr(expr, "hmm_num_rhs_loop_calls", 0))
        fields["num_rhs_loop_calls"] = num_rhs_loop_calls
        fields["hmm_num_rhs_loop_calls"] = num_rhs_loop_calls
    return fields


def _hmm_phase_breakdown_fields(expr):
    trace = getattr(expr, "hmm_execution_trace", ()) or ()
    return _hmm_phase_breakdown_fields_from_trace(trace)


def _append_unique(values, items):
    for item in items or ():
        if item not in values:
            values.append(item)


def _default_hmm_phase_execution(phase):
    if phase in {"inter_gemv", "reduce_gemv"}:
        return ["gemv"], ["loop_gemv"]
    return [], []


def _hmm_executor_kernel_counts(expr):
    lowerings = [str(lowering) for lowering in getattr(expr, "hmm_step_lowerings", ())]
    return {
        "num_gemm": int(getattr(expr, "hmm_num_gemm", sum(1 for lowering in lowerings if lowering == "gemm"))),
        "num_batched_gemm": int(
            getattr(
                expr,
                "hmm_num_batched_gemm",
                sum(1 for lowering in lowerings if lowering in ("batched_gemm", "strided_batched_gemm")),
            )
        ),
        "num_grouped_tasks": int(getattr(expr, "hmm_num_grouped_tasks", 0)),
        "num_blocks": int(getattr(expr, "hmm_num_blocks", 0)),
        "num_shape_buckets": int(getattr(expr, "hmm_num_shape_buckets", 0)),
    }


def _array_nbytes(array):
    nbytes = getattr(array, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    shape = getattr(array, "shape", ())
    dtype = getattr(array, "dtype", None)
    size = 1
    for dim in shape:
        size *= int(dim)
    return int(size) * _dtype_itemsize(dtype)


def _equation_input_modes(equation):
    if not equation or "->" not in equation:
        return ()
    left = equation.split("->", 1)[0]
    return tuple(
        tuple(mode for mode in operand.strip() if not mode.isspace())
        for operand in left.split(",")
    )


def _operand_modes(input_modes, index, array):
    if 0 <= index < len(input_modes):
        return input_modes[index]
    return tuple(range(len(tuple(getattr(array, "shape", ())))))


def _hmm_executor_operands(expr, center):
    static_operands = tuple(getattr(expr, "hmm_static_operands", ()))
    input_modes = _equation_input_modes(getattr(expr, "equation", None))
    operands = []
    for index, (name, array) in enumerate(static_operands):
        operands.append((name, array, _operand_modes(input_modes, index, array)))
    center_index = len(static_operands)
    operands.append(("center", center, _operand_modes(input_modes, center_index, center)))
    return tuple(operands)


def _execution_resource_fields(stream=None, workspace=None):
    workspace_device = getattr(workspace, "device", None)
    return {
        "stream_provided": stream is not None,
        "stream_type": type(stream).__name__ if stream is not None else None,
        "workspace_provided": workspace is not None,
        "workspace_nbytes": int(getattr(workspace, "nbytes", 0) or 0) if workspace is not None else None,
        "workspace_released": bool(getattr(workspace, "released", False)) if workspace is not None else None,
        "workspace_device_kind": getattr(workspace_device, "kind", None),
        "workspace_device_index": getattr(workspace_device, "index", None),
        "workspace_device_local_rank": getattr(workspace_device, "local_rank", None),
        "workspace_device_global_rank": getattr(workspace_device, "global_rank", None),
    }


def _cache_backend_execution_profile(payload):
    try:
        setattr(backend.current, "_last_execution_profile", dict(payload))
    except Exception:
        pass


def _normalize_equation(equation):
    return "" if equation is None else str(equation).replace(" ", "")


def _hmm_phase_breakdown_fields_from_trace(trace):
    phase_order = []
    counts = {}
    tasks = {}
    groups = {}
    flops = {}
    primitives = {}
    policies = {}
    fallback_reasons = {}
    for item in trace or ():
        if not isinstance(item, dict):
            continue
        phase = str(item.get("phase") or "unknown")
        if phase not in counts:
            phase_order.append(phase)
            counts[phase] = 0
            tasks[phase] = 0
            groups[phase] = 0
            flops[phase] = 0
            default_primitives, default_policies = _default_hmm_phase_execution(phase)
            primitives[phase] = list(default_primitives)
            policies[phase] = list(default_policies)
            fallback_reasons[phase] = []
        counts[phase] += 1
        tasks[phase] += int(item.get("num_tasks") or 0)
        groups[phase] += int(item.get("num_groups") or 0)
        flops[phase] += int(item.get("total_flops") or 0)
        _append_unique(primitives[phase], item.get("execution_primitives") or ())
        _append_unique(policies[phase], item.get("execution_policies") or ())
        _append_unique(fallback_reasons[phase], item.get("fallback_reasons") or ())
        if item.get("fallback_reason"):
            _append_unique(fallback_reasons[phase], (str(item["fallback_reason"]),))
    if not phase_order:
        return {}
    return {
        "hmm_phase_counts": {phase: int(counts[phase]) for phase in phase_order},
        "hmm_phase_tasks": {phase: int(tasks[phase]) for phase in phase_order},
        "hmm_phase_groups": {phase: int(groups[phase]) for phase in phase_order},
        "hmm_phase_flops": {phase: int(flops[phase]) for phase in phase_order},
        "hmm_phase_breakdown": [
            {
                "phase": phase,
                "occurrences": int(counts[phase]),
                "num_tasks": int(tasks[phase]),
                "num_groups": int(groups[phase]),
                "flops": int(flops[phase]),
                "execution_primitives": list(primitives[phase]),
                "execution_policies": list(policies[phase]),
                "fallback_reasons": list(fallback_reasons[phase]),
                "fallback_required": bool(fallback_reasons[phase]),
            }
            for phase in phase_order
        ],
    }


def _hmm_runtime_profile_fields(equation):
    try:
        profile = backend.last_execution_profile()
    except Exception:
        return {}
    if not isinstance(profile, dict):
        return {}
    if profile.get("lowering") != "hmm_task":
        return {}
    if _normalize_equation(profile.get("equation")) != _normalize_equation(equation):
        return {}

    keys = (
        "fallback_reason",
        "fallback_from",
        "fallback_to",
        "fallback_policy",
        "fallback_reasons",
        "fallback_sources",
        "fallback_targets",
        "fallback_policies",
        "grouped_gemm_policies",
        "grouped_gemm_implementations",
        "requires_grouped_gemm_fallback",
        "execution_primitives",
        "execution_policies",
        "planned_num_gemm",
        "planned_num_batched_gemm",
        "planned_num_grouped_tasks",
        "actual_lowerings",
        "actual_num_gemm",
        "actual_num_batched_gemm",
        "actual_num_grouped_gemm",
        "actual_num_loop_matmul",
        "actual_num_tensordot",
        "actual_num_einsum",
    )
    fields = {
        key: profile[key]
        for key in keys
        if key in profile
    }
    if profile.get("fallback_reason"):
        for key in ("hmm_batches", "hmm_execution_trace"):
            if key in profile:
                fields[key] = profile[key]
        if "hmm_execution_trace" in fields:
            fields.update(_hmm_phase_breakdown_fields_from_trace(fields["hmm_execution_trace"]))
    return fields


def _record_hmm_executor_execution(expr, center, result, started, *, stream=None, workspace=None):
    center_nbytes = _array_nbytes(center)
    result_nbytes = _array_nbytes(result)
    path_summary = getattr(expr, "path_summary", None) or {}
    kernel_counts = _hmm_executor_kernel_counts(expr)
    operands = _hmm_executor_operands(expr, center)
    largest_intermediate = int(
        getattr(
            expr,
            "hmm_largest_intermediate",
            max(center_nbytes, result_nbytes),
        )
    )
    largest_intermediate_bytes = int(
        getattr(expr, "hmm_largest_intermediate_bytes", largest_intermediate)
    )
    itemsize = _dtype_itemsize(getattr(result, "dtype", None))
    largest_intermediate_elements = int(
        getattr(
            expr,
            "hmm_largest_intermediate_elements",
            largest_intermediate_bytes // itemsize if itemsize else 0,
        )
    )
    payload = {
        "event": "contraction_execute",
        "backend": backend.name,
        **profiling.contraction_execute_compute_payload("hmm_executor"),
        "equation": getattr(expr, "equation", None),
        "lowering": "hmm_executor",
        "input_shapes": [tuple(getattr(array, "shape", ())) for _name, array, _modes in operands],
        "input_dtypes": [str(getattr(array, "dtype", None)) for _name, array, _modes in operands],
        "operands": [
            profiling.array_operand_payload(
                backend,
                name,
                array,
                modes,
            )
            for name, array, modes in operands
        ],
        "output_shape": tuple(getattr(result, "shape", ())),
        "dtype": str(getattr(result, "dtype", None)),
        **profiling.device_execution_payload(backend.current_device()),
        "flops": int(getattr(expr, "hmm_flops", path_summary.get("flop_count") or 0)),
        "read_bytes": int(getattr(expr, "hmm_read_bytes", center_nbytes)),
        "write_bytes": int(getattr(expr, "hmm_write_bytes", result_nbytes)),
        "copy_bytes": int(getattr(expr, "hmm_copy_bytes", 0)),
        "workspace_bytes": int(getattr(expr, "hmm_workspace_bytes", max(center_nbytes, result_nbytes))),
        "largest_intermediate": largest_intermediate,
        "largest_intermediate_elements": largest_intermediate_elements,
        "largest_intermediate_bytes": largest_intermediate_bytes,
        "hmm_largest_intermediate_elements": largest_intermediate_elements,
        "hmm_largest_intermediate_bytes": largest_intermediate_bytes,
        "fallback_reason": getattr(expr, "hmm_fallback_reason", None),
        **_execution_resource_fields(stream=stream, workspace=workspace),
        **kernel_counts,
        **_hmm_execution_profile_fields(expr),
        **_hmm_phase_breakdown_fields(expr),
        **_hmm_runtime_profile_fields(getattr(expr, "equation", None)),
        "wall_s": time.perf_counter() - started,
    }
    _cache_backend_execution_profile(payload)
    record_payload = dict(payload)
    record_payload.pop("event", None)
    profiling.record("contraction_execute", **record_payload)


def _hmm_path_summary(equation, *operands):
    if not profiling.should_record_op():
        return None
    contract_path = getattr(backend, "contract_path", None)
    if not callable(contract_path):
        return None
    return profiling.contract_path_summary(contract_path, (equation, *operands), {})


def _make_hmm_executor_expr(equation, executor, executor_name, path_summary=None, static_operands=()):
    def expr(center, *args, **kwargs):
        if args:
            raise TypeError("HMM executor expression accepts one center tensor argument")
        stream = kwargs.pop("stream", None)
        workspace = kwargs.pop("workspace", None)
        if kwargs:
            unexpected = ", ".join(sorted(str(key) for key in kwargs))
            raise TypeError("Unexpected HMM executor expression keyword(s): {0}".format(unexpected))
        profile_enabled = profiling.should_record_op()
        started = time.perf_counter() if profile_enabled else None
        result = executor(center, stream=stream, workspace=workspace)
        if profile_enabled:
            _record_hmm_executor_execution(expr, center, result, started, stream=stream, workspace=workspace)
        return result

    expr.equation = equation
    expr.path_summary = path_summary
    expr.hmm_executor = executor_name
    expr.hmm_static_operands = tuple(static_operands)
    return expr


def hop_expr(ltensor, rtensor, cmo, cshape, twolayer:bool=False):

    profile_enabled = profiling.should_record_op()
    started = time.perf_counter() if profile_enabled else None
    nsite = len(cmo)
    # whether have the ancilla
    ancilla = 2 * nsite + 2 == len(cshape)
    ancilla_for_profile = ancilla if nsite != 0 else None
    if not ancilla:
        assert nsite + 2 == len(cshape)

    ltensor = asxp(ltensor)
    rtensor = asxp(rtensor)
    for i in range(len(cmo)):
        cmo[i] = asxp(cmo[i])

    if nsite == 0:
        # ancilla not defined
        del ancilla

    if twolayer:
        assert nsite in [1, 2]
        # Only used in ground state algorithm
        # Hopefully generalize to CV in the future
        assert not ancilla
        if nsite == 1:
            #   S-a e j-S
            #   O-b-O-g-O
            #   |   f   |
            #   O-c-O-i-O
            #   S-d h k-S
            expr = oe_contract_expression(
                "abcd, befg, cfhi, jgik, aej -> dhk",
                ltensor, cmo[0], cmo[0], rtensor, cshape,
                constants=[0, 1, 2, 3]
            )
        else:
            #   S-a e   j o-S
            #   O-b-O-g-O-l-O
            #   |   f   k   |
            #   O-c-O-i-O-n-O
            #   S-d h   m p-S
            expr = oe_contract_expression(
                "abcd, befg, cfhi, gjkl, ikmn, olnp, aejo -> dhmp",
                ltensor, cmo[0], cmo[0], cmo[1], cmo[1], rtensor, cshape,
                constants=[0, 1, 2, 3, 4, 5],
            )
        # early return
        if profile_enabled:
            return _record_hop_expr(expr, nsite, ancilla_for_profile, twolayer, ltensor, rtensor, cmo, cshape, started)
        return expr

    # Single layer, the most common case
    # Could be written in an automatic way
    # But for now probably an overkill
    if nsite == 0:
        # S-a   l-S
        #
        # O-b - b-O
        #
        # S-c   k-S
        expr = oe_contract_expression(
            "abc, lbk, ck -> al",
            ltensor, rtensor, cshape,
            constants=[0, 1],
        )
    elif nsite == 1:
        if not ancilla:
            # S-a   l-S
            #     d
            # O-b-O-f-O
            #     e
            # S-c   k-S
            equation = "abc, bdef, lfk, cek -> adl"
            from renormalizer.backend.gemm import execute_single_site_hmm_action

            expr = _make_hmm_executor_expr(
                equation,
                lambda center, *, stream=None, workspace=None: execute_single_site_hmm_action(
                    backend,
                    ltensor,
                    cmo[0],
                    rtensor,
                    center,
                    stream=stream,
                    workspace=workspace,
                ),
                "single_site",
                _hmm_path_summary(equation, ltensor, cmo[0], rtensor, cshape),
                (
                    ("left_env", ltensor),
                    ("mpo0", cmo[0]),
                    ("right_env", rtensor),
                ),
            )
            if profile_enabled:
                _attach_hmm_profile_metadata(
                    expr,
                    _record_hmm_scaffold(ltensor, rtensor, cmo, cshape),
                )
        else:
            # S-a   l-S
            #     d
            # O-b-O-f-O
            #     e
            # S-c   k-S
            #     g
            expr = oe_contract_expression(
                "abc, bdef, lfk, cegk -> adgl",
                ltensor, cmo[0], rtensor, cshape,
                constants=[0, 1, 2],
            )
    else:
        if not ancilla:
            # S-a       l-S
            #     d   g
            # O-b-O-f-O-j-O
            #     e   h
            # S-c       k-S
            equation = "abc, bdef, fghj, ljk, cehk -> adgl"
            from renormalizer.backend.gemm import execute_two_site_hmm_action

            expr = _make_hmm_executor_expr(
                equation,
                lambda center, *, stream=None, workspace=None: execute_two_site_hmm_action(
                    backend,
                    ltensor,
                    cmo[0],
                    cmo[1],
                    rtensor,
                    center,
                    stream=stream,
                    workspace=workspace,
                ),
                "two_site",
                _hmm_path_summary(equation, ltensor, cmo[0], cmo[1], rtensor, cshape),
                (
                    ("left_env", ltensor),
                    ("mpo0", cmo[0]),
                    ("mpo1", cmo[1]),
                    ("right_env", rtensor),
                ),
            )
            if profile_enabled:
                _attach_hmm_profile_metadata(
                    expr,
                    _record_hmm_scaffold(ltensor, rtensor, cmo, cshape),
                )
        else:
            # S-a       l-S
            #     d   g
            # O-b-O-f-O-j-O
            #     e   h
            # S-c       k-S
            #     m   n
            expr = oe_contract_expression(
                "abc, bdef, fghj, ljk, cemhnk -> admgnl",
                ltensor, cmo[0], cmo[1], rtensor, cshape,
                constants=[0, 1, 2, 3],
            )

    if profile_enabled:
        return _record_hop_expr(expr, nsite, ancilla_for_profile, twolayer, ltensor, rtensor, cmo, cshape, started)
    return expr


def batched_hop_expr(ltensor, rtensor, cmo, cshape, nrhs, twolayer: bool = False):
    profile_enabled = profiling.should_record_op()
    started = time.perf_counter() if profile_enabled else None
    nsite = len(cmo)
    ancilla = 2 * nsite + 2 == len(cshape)
    ancilla_for_profile = ancilla if nsite != 0 else None
    if not ancilla:
        assert nsite + 2 == len(cshape)

    ltensor = asxp(ltensor)
    rtensor = asxp(rtensor)
    for i in range(len(cmo)):
        cmo[i] = asxp(cmo[i])

    batched_shape = tuple(cshape) + (int(nrhs),)

    if twolayer:
        assert nsite in [1, 2]
        assert not ancilla
        if nsite == 1:
            expr = oe_contract_expression(
                "abcd, befg, cfhi, jgik, aejr -> dhkr",
                ltensor, cmo[0], cmo[0], rtensor, batched_shape,
                constants=[0, 1, 2, 3],
            )
        else:
            expr = oe_contract_expression(
                "abcd, befg, cfhi, gjkl, ikmn, olnp, aejor -> dhmpr",
                ltensor, cmo[0], cmo[0], cmo[1], cmo[1], rtensor, batched_shape,
                constants=[0, 1, 2, 3, 4, 5],
            )
        if profile_enabled:
            return _record_hop_expr(expr, nsite, ancilla_for_profile, twolayer, ltensor, rtensor, cmo, batched_shape, started)
        return expr

    if nsite == 0:
        expr = oe_contract_expression(
            "abc, lbk, ckr -> alr",
            ltensor, rtensor, batched_shape,
            constants=[0, 1],
        )
    elif nsite == 1:
        if not ancilla:
            equation = "abc, bdef, lfk, cekr -> adlr"
            from renormalizer.backend.gemm import execute_single_site_hmm_action

            expr = _make_hmm_executor_expr(
                equation,
                lambda center, *, stream=None, workspace=None: execute_single_site_hmm_action(
                    backend,
                    ltensor,
                    cmo[0],
                    rtensor,
                    center,
                    stream=stream,
                    workspace=workspace,
                ),
                "single_site",
                _hmm_path_summary(equation, ltensor, cmo[0], rtensor, batched_shape),
                (
                    ("left_env", ltensor),
                    ("mpo0", cmo[0]),
                    ("right_env", rtensor),
                ),
            )
            if profile_enabled:
                _attach_hmm_profile_metadata(
                    expr,
                    _record_hmm_scaffold(ltensor, rtensor, cmo, cshape, nrhs=nrhs),
                )
        else:
            expr = oe_contract_expression(
                "abc, bdef, lfk, cegkr -> adglr",
                ltensor, cmo[0], rtensor, batched_shape,
                constants=[0, 1, 2],
            )
    else:
        if not ancilla:
            equation = "abc, bdef, fghj, ljk, cehkr -> adglr"
            from renormalizer.backend.gemm import execute_two_site_hmm_action

            expr = _make_hmm_executor_expr(
                equation,
                lambda center, *, stream=None, workspace=None: execute_two_site_hmm_action(
                    backend,
                    ltensor,
                    cmo[0],
                    cmo[1],
                    rtensor,
                    center,
                    stream=stream,
                    workspace=workspace,
                ),
                "two_site",
                _hmm_path_summary(equation, ltensor, cmo[0], cmo[1], rtensor, batched_shape),
                (
                    ("left_env", ltensor),
                    ("mpo0", cmo[0]),
                    ("mpo1", cmo[1]),
                    ("right_env", rtensor),
                ),
            )
            if profile_enabled:
                _attach_hmm_profile_metadata(
                    expr,
                    _record_hmm_scaffold(ltensor, rtensor, cmo, cshape, nrhs=nrhs),
                )
        else:
            expr = oe_contract_expression(
                "abc, bdef, fghj, ljk, cemhnkr -> admgnlr",
                ltensor, cmo[0], cmo[1], rtensor, batched_shape,
                constants=[0, 1, 2, 3],
            )

    if profile_enabled:
        return _record_hop_expr(expr, nsite, ancilla_for_profile, twolayer, ltensor, rtensor, cmo, batched_shape, started)
    return expr
