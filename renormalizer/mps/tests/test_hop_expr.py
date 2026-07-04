# -*- coding: utf-8 -*-

import json

import numpy as np
import pytest


def _rng():
    return np.random.default_rng(20260702)


def test_batched_hop_expr_matches_stacked_single_site_results():
    from renormalizer.mps.hop_expr import batched_hop_expr, hop_expr

    rng = _rng()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 6, 7))
    rtensor = rng.normal(size=(8, 7, 9))
    cshape = (4, 6, 9)
    center_batch = rng.normal(size=cshape + (3,))

    expr = hop_expr(ltensor, rtensor, [mpo], cshape)
    batched_expr = batched_hop_expr(ltensor, rtensor, [mpo], cshape, nrhs=3)

    expected = np.stack(
        [expr(center_batch[..., i]) for i in range(center_batch.shape[-1])],
        axis=-1,
    )
    actual = batched_expr(center_batch)

    assert actual.shape == expected.shape
    assert np.allclose(actual, expected)


def test_single_site_hop_expr_executes_backend_hmm_matmul(monkeypatch):
    from renormalizer.mps.backend import backend
    from renormalizer.mps.hop_expr import hop_expr

    rng = _rng()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 6, 7))
    rtensor = rng.normal(size=(8, 7, 9))
    cshape = (4, 6, 9)
    center = rng.normal(size=cshape)
    calls = {"matmul": 0, "batched_matmul": 0}

    original_matmul = backend.current.matmul
    original_batched = backend.current.batched_matmul

    def counted_matmul(*args, **kwargs):
        if not (len(args) == 1 and backend.current._is_matmul_desc(args[0])):
            calls["matmul"] += 1
        return original_matmul(*args, **kwargs)

    def counted_batched_matmul(*args, **kwargs):
        if not (len(args) == 1 and backend.current._is_matmul_desc(args[0])):
            calls["batched_matmul"] += 1
        return original_batched(*args, **kwargs)

    monkeypatch.setattr(backend.current, "matmul", counted_matmul)
    monkeypatch.setattr(backend.current, "batched_matmul", counted_batched_matmul)

    expr = hop_expr(ltensor, rtensor, [mpo], cshape)
    actual = expr(center)
    expected = np.einsum("abc,bdef,lfk,cek->adl", ltensor, mpo, rtensor, center)

    assert np.allclose(actual, expected)
    assert calls == {"matmul": 3, "batched_matmul": 0}
    assert getattr(expr, "hmm_executor", None) == "single_site"


def test_batched_single_site_hop_expr_executes_backend_hmm_batched_matmul(monkeypatch):
    from renormalizer.mps.backend import backend
    from renormalizer.mps.hop_expr import batched_hop_expr

    rng = _rng()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 6, 7))
    rtensor = rng.normal(size=(8, 7, 9))
    cshape = (4, 6, 9)
    center_batch = rng.normal(size=cshape + (3,))
    calls = {"matmul": 0, "batched_matmul": 0}

    original_matmul = backend.current.matmul
    original_batched = backend.current.batched_matmul

    def counted_matmul(*args, **kwargs):
        if not (len(args) == 1 and backend.current._is_matmul_desc(args[0])):
            calls["matmul"] += 1
        return original_matmul(*args, **kwargs)

    def counted_batched_matmul(*args, **kwargs):
        if not (len(args) == 1 and backend.current._is_matmul_desc(args[0])):
            calls["batched_matmul"] += 1
        return original_batched(*args, **kwargs)

    monkeypatch.setattr(backend.current, "matmul", counted_matmul)
    monkeypatch.setattr(backend.current, "batched_matmul", counted_batched_matmul)

    expr = batched_hop_expr(ltensor, rtensor, [mpo], cshape, nrhs=3)
    actual = expr(center_batch)
    expected = np.einsum("abc,bdef,lfk,cekr->adlr", ltensor, mpo, rtensor, center_batch)

    assert np.allclose(actual, expected)
    assert calls == {"matmul": 0, "batched_matmul": 3}
    assert getattr(expr, "hmm_executor", None) == "single_site"


def test_batched_hop_expr_profiles_batched_hmm_scaffold(tmp_path):
    from renormalizer.mps.hop_expr import batched_hop_expr
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    rng = _rng()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 6, 7))
    rtensor = rng.normal(size=(8, 7, 9))
    cshape = (4, 6, 9)
    center_batch = rng.normal(size=cshape + (3,))
    old_level = package_logger.level
    event_path = tmp_path / "events.jsonl"
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        expr = batched_hop_expr(ltensor, rtensor, [mpo], cshape, nrhs=3)
        result = expr(center_batch)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert result.shape == (2, 5, 8, 3)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    build_event = next(
        payload
        for payload in payloads
        if payload["event"] == "hmm_task_build"
        and payload["equation"] == "abc,bdef,lfk,cekr->adlr"
    )
    assert build_event["rhs_batch_mode"] == "r"
    assert build_event["num_rhs"] == 3
    assert build_event["num_rhs_loop_calls"] == 0
    assert build_event["num_gemm_desc"] == 3
    assert build_event["num_batched_gemm"] == 3
    assert build_event["num_shape_buckets"] == 3
    assert build_event["shape_buckets"][0]["batch_shape"] == [3]
    assert build_event["shape_buckets"][0]["batch_count"] == 3
    assert build_event["output_shape"] == [2, 5, 8, 3]

    plan_event = next(
        payload
        for payload in payloads
        if payload["event"] == "contraction_plan"
        and payload["equation"] == "abc,bdef,lfk,cekr->adlr"
        and payload["lowering"] == "hmm_task"
    )
    assert plan_event["rhs_batch_mode"] == "r"
    assert plan_event["num_rhs"] == 3
    assert plan_event["num_rhs_loop_calls"] == 0
    assert plan_event["step_count"] == 3
    assert plan_event["step_lowerings"] == ["batched_gemm", "batched_gemm", "batched_gemm"]
    assert plan_event["num_gemm"] == 0
    assert plan_event["num_batched_gemm"] == 3
    assert plan_event["num_grouped_tasks"] == 0
    assert plan_event["num_shape_buckets"] == 3
    assert plan_event["output_shape"] == [2, 5, 8, 3]
    assert plan_event["hmm_steps"][0]["descs"][0]["batch_shape"] == [3]
    assert plan_event["compute_profile"]["primary_kernel"] == "hmm_task"
    assert plan_event["compute_profile"]["work"]["num_batched_gemm"] == 3


def test_single_site_hop_expr_profiles_hmm_scaffold(tmp_path):
    from renormalizer.mps.hop_expr import hop_expr
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    rng = _rng()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 6, 7))
    rtensor = rng.normal(size=(8, 7, 9))
    cshape = (4, 6, 9)
    center = rng.normal(size=cshape)
    old_level = package_logger.level
    event_path = tmp_path / "events.jsonl"
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        expr = hop_expr(ltensor, rtensor, [mpo], cshape)
        result = expr(center)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert result.shape == (2, 5, 8)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in payloads if payload["event"] == "hmm_task_build")
    assert event["equation"] == "abc,bdef,lfk,cek->adl"
    assert event["center_kind"] == "onedot"
    assert event["num_hx_blocks"] == 1
    assert event["num_gemm_desc"] == 3
    assert event["stage_group_sizes"] == [[1], [1], [1]]
    assert event["operand_shapes"] == [[2, 3, 4], [3, 5, 6, 7], [8, 7, 9], [4, 6, 9]]
    assert event["output_shape"] == [2, 5, 8]
    assert event["largest_intermediate"] == 10752
    assert event["largest_intermediate_elements"] == 1344
    assert event["largest_intermediate_bytes"] == 10752
    plan_event = next(
        payload
        for payload in payloads
        if payload["event"] == "contraction_plan"
        and payload["equation"] == "abc,bdef,lfk,cek->adl"
        and payload["lowering"] == "hmm_task"
    )
    assert plan_event["hmm_center_kind"] == "onedot"
    assert plan_event["step_count"] == 3
    assert plan_event["step_lowerings"] == ["gemm", "gemm", "gemm"]
    assert plan_event["matmul_plan_hashes"] == [
        step["plan_hash"] for step in plan_event["hmm_steps"]
    ]
    assert plan_event["num_gemm"] == 3
    assert plan_event["num_batched_gemm"] == 0
    assert plan_event["num_grouped_tasks"] == 0
    assert plan_event["num_shape_buckets"] == 3
    assert plan_event["num_gemv_desc"] == 0
    assert plan_event["largest_intermediate"] == 10752
    assert plan_event["largest_intermediate_elements"] == 1344
    assert plan_event["largest_intermediate_bytes"] == 10752
    assert isinstance(plan_event["plan_hash"], str)
    assert plan_event["compute_profile"]["compute_class"] == "contraction_plan"
    assert plan_event["compute_profile"]["work"]["num_gemm"] == 3


def test_single_site_hop_expr_profiles_hmm_executor_execution(tmp_path, caplog):
    from renormalizer.mps.backend import backend
    from renormalizer.mps.hop_expr import hop_expr
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    rng = _rng()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 6, 7))
    rtensor = rng.normal(size=(8, 7, 9))
    cshape = (4, 6, 9)
    center = rng.normal(size=cshape)
    stream = object()
    workspace = backend.allocate_workspace(2048)
    old_level = package_logger.level
    event_path = tmp_path / "events.jsonl"
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        expr = hop_expr(ltensor, rtensor, [mpo], cshape)
        result = expr(center, stream=stream, workspace=workspace)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert result.shape == (2, 5, 8)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    plan_event = next(
        payload
        for payload in payloads
        if payload["event"] == "contraction_plan"
        and payload["equation"] == "abc,bdef,lfk,cek->adl"
        and payload["lowering"] == "hmm_task"
    )
    execute_event = next(
        payload
        for payload in payloads
        if payload["event"] == "contraction_execute"
        and payload["lowering"] == "hmm_executor"
        and payload["equation"].replace(" ", "") == "abc,bdef,lfk,cek->adl"
    )

    assert execute_event["backend"] == "numpy"
    assert execute_event["compute_class"] == "contraction_plan"
    assert execute_event["compute_subclass"] == "backend_execute"
    assert execute_event["compute_profile"]["compute_class"] == "contraction_plan"
    assert execute_event["hmm_plan_hash"] == plan_event["plan_hash"]
    assert execute_event["hmm_matmul_plan_hashes"] == plan_event["matmul_plan_hashes"]
    assert execute_event["hmm_step_lowerings"] == plan_event["step_lowerings"]
    assert execute_event["hmm_num_shape_buckets"] == plan_event["num_shape_buckets"]
    assert execute_event["hmm_num_batched_gemm"] == 0
    assert execute_event["hmm_steps"] == plan_event["hmm_steps"]
    assert execute_event["hmm_shape_buckets"] == plan_event["shape_buckets"]
    assert execute_event["hmm_gemv_shape_buckets"] == plan_event["hmm_gemv_shape_buckets"]
    assert execute_event["hmm_batches"] == plan_event["hmm_batches"]
    assert execute_event["hmm_hx_blocks"] == plan_event["hmm_hx_blocks"]
    assert execute_event["hmm_execution_trace"] == plan_event["hmm_execution_trace"]
    assert [item["phase"] for item in execute_event["hmm_execution_trace"]] == [
        "inter_gemv",
        "gemm_stage",
        "gemm_stage",
        "gemm_stage",
        "reduce_gemv",
    ]
    assert execute_event["hmm_phase_counts"] == {
        "inter_gemv": 1,
        "gemm_stage": 3,
        "reduce_gemv": 1,
    }
    assert execute_event["hmm_phase_tasks"] == {
        "inter_gemv": 0,
        "gemm_stage": 3,
        "reduce_gemv": 0,
    }
    assert execute_event["hmm_phase_groups"] == {
        "inter_gemv": 0,
        "gemm_stage": 3,
        "reduce_gemv": 0,
    }
    assert sum(execute_event["hmm_phase_flops"].values()) == execute_event["flops"]
    assert execute_event["hmm_phase_breakdown"] == [
        {
            "phase": "inter_gemv",
            "occurrences": 1,
            "num_tasks": 0,
            "num_groups": 0,
            "flops": 0,
            "execution_primitives": ["gemv"],
            "execution_policies": ["loop_gemv"],
            "fallback_reasons": [],
            "fallback_required": False,
        },
        {
            "phase": "gemm_stage",
            "occurrences": 3,
            "num_tasks": 3,
            "num_groups": 3,
            "flops": execute_event["flops"],
            "execution_primitives": ["matmul"],
            "execution_policies": ["backend_matmul"],
            "fallback_reasons": [],
            "fallback_required": False,
        },
        {
            "phase": "reduce_gemv",
            "occurrences": 1,
            "num_tasks": 0,
            "num_groups": 0,
            "flops": 0,
            "execution_primitives": ["gemv"],
            "execution_policies": ["loop_gemv"],
            "fallback_reasons": [],
            "fallback_required": False,
        },
    ]
    assert execute_event["hmm_execution_trace"][1]["plan_hashes"] == [
        plan_event["hmm_steps"][0]["plan_hash"]
    ]
    assert execute_event["hmm_execution_trace"][1]["execution_primitives"] == ["matmul"]
    assert execute_event["hmm_execution_trace"][1]["execution_policies"] == ["backend_matmul"]
    assert execute_event["hmm_execution_trace"][1]["fallback_reasons"] == []
    assert execute_event["hmm_batches"][0]["stages"][0]["group_sizes"] == [1]
    assert execute_event["hmm_batches"][0]["stages"][0]["execution_primitives"] == ["matmul"]
    assert execute_event["hmm_batches"][0]["stages"][0]["execution_policies"] == ["backend_matmul"]
    assert execute_event["hmm_batches"][0]["stages"][0]["fallback_reasons"] == []
    assert execute_event["hmm_batches"][0]["stages"][0]["gsta"] == [0, 1]
    assert execute_event["hmm_batches"][0]["stages"][0]["sorted_indices"] == [0]
    assert execute_event["hmm_steps"][0]["descs"][0]["m"] == 24
    assert execute_event["hmm_shape_buckets"][0]["task_count"] == 1
    assert execute_event["hmm_flops"] == plan_event["flops"]
    assert execute_event["hmm_read_bytes"] == plan_event["read_bytes"]
    assert execute_event["hmm_write_bytes"] == plan_event["write_bytes"]
    assert execute_event["hmm_copy_bytes"] == plan_event["copy_bytes"]
    assert execute_event["hmm_workspace_bytes"] == plan_event["workspace_bytes"]
    assert execute_event["hmm_largest_intermediate"] == plan_event["largest_intermediate"]
    assert execute_event["num_gemm"] == 3
    assert execute_event["num_batched_gemm"] == 0
    assert execute_event["num_shape_buckets"] == 3
    assert execute_event["input_shapes"] == [
        [2, 3, 4],
        [3, 5, 6, 7],
        [8, 7, 9],
        [4, 6, 9],
    ]
    assert [operand["name"] for operand in execute_event["operands"]] == [
        "left_env",
        "mpo0",
        "right_env",
        "center",
    ]
    assert execute_event["output_shape"] == [2, 5, 8]
    assert execute_event["largest_intermediate_elements"] == plan_event["largest_intermediate_elements"]
    assert execute_event["largest_intermediate_bytes"] == plan_event["largest_intermediate_bytes"]
    assert execute_event["hmm_largest_intermediate_elements"] == plan_event["largest_intermediate_elements"]
    assert execute_event["hmm_largest_intermediate_bytes"] == plan_event["largest_intermediate_bytes"]
    assert execute_event["stream_provided"] is True
    assert execute_event["stream_type"] == "object"
    assert execute_event["workspace_provided"] is True
    assert execute_event["workspace_nbytes"] == 2048
    assert execute_event["workspace_device_kind"] == "cpu"
    assert execute_event["workspace_device_index"] is None
    assert execute_event["fallback_reason"] is None
    assert execute_event["wall_s"] >= 0.0
    assert execute_event["hmm_num_batches"] == 1
    assert execute_event["hmm_batch_size"] == 1
    assert execute_event["hmm_num_gemv_desc"] == 0

    logged_payloads = [
        json.loads(record.getMessage()[len(profiling.LOG_PREFIX):])
        for record in caplog.records
        if record.getMessage().startswith(profiling.LOG_PREFIX)
    ]
    summary = next(
        payload
        for payload in logged_payloads
        if payload.get("event") == "profile_compute_class_summary"
        and payload.get("source_events") == ["contraction_execute"]
        and payload.get("compute_class") == "contraction_plan"
        and payload.get("source_subclasses") == ["backend_execute"]
    )
    assert summary["lowerings"] == ["hmm_executor"]
    assert summary["total_hmm_batches"] == 1
    assert summary["max_hmm_batch_size"] == 1
    assert summary["total_hmm_gemm_desc"] == 3
    assert summary["total_hmm_gemv_desc"] == 0
    assert summary["compute_profile"]["work"]["num_batches"] == 1
    assert summary["compute_profile"]["work"]["num_gemm_desc"] == 3
    assert summary["compute_profile"]["work"]["num_gemv_desc"] == 0

    profile = backend.last_execution_profile()
    assert profile["event"] == "contraction_execute"
    assert profile["lowering"] == "hmm_executor"
    assert profile["compute_class"] == "contraction_plan"
    assert profile["equation"] == execute_event["equation"]
    assert profile["hmm_plan_hash"] == plan_event["plan_hash"]
    assert profile["hmm_step_lowerings"] == plan_event["step_lowerings"]
    assert profile["hmm_gemv_shape_buckets"] == plan_event["hmm_gemv_shape_buckets"]
    assert profile["hmm_phase_counts"] == execute_event["hmm_phase_counts"]
    assert profile["hmm_phase_tasks"] == execute_event["hmm_phase_tasks"]
    assert profile["hmm_phase_flops"] == execute_event["hmm_phase_flops"]
    assert profile["hmm_flops"] == plan_event["flops"]
    assert profile["hmm_read_bytes"] == plan_event["read_bytes"]
    assert profile["hmm_write_bytes"] == plan_event["write_bytes"]
    assert profile["hmm_copy_bytes"] == plan_event["copy_bytes"]
    assert profile["hmm_workspace_bytes"] == plan_event["workspace_bytes"]
    assert profile["hmm_largest_intermediate"] == plan_event["largest_intermediate"]
    assert profile["num_gemm"] == 3
    assert profile["num_batched_gemm"] == 0
    assert profile["num_shape_buckets"] == 3
    assert profile["largest_intermediate_elements"] == plan_event["largest_intermediate_elements"]
    assert profile["largest_intermediate_bytes"] == plan_event["largest_intermediate_bytes"]
    assert profile["hmm_largest_intermediate_elements"] == plan_event["largest_intermediate_elements"]
    assert profile["hmm_largest_intermediate_bytes"] == plan_event["largest_intermediate_bytes"]
    assert profile["stream_provided"] is True
    assert profile["workspace_nbytes"] == 2048


def test_hop_expr_profiling_does_not_silently_drop_hmm_scaffold_failure(monkeypatch):
    from renormalizer.backend import gemm
    from renormalizer.mps.hop_expr import hop_expr
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    rng = _rng()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 6, 7))
    rtensor = rng.normal(size=(8, 7, 9))
    cshape = (4, 6, 9)
    old_level = package_logger.level

    def broken_scaffold(*_args, **_kwargs):
        raise RuntimeError("synthetic scaffold failure")

    monkeypatch.setattr(gemm, "build_single_site_hmm_scaffold_from_shapes", broken_scaffold)

    try:
        init_log(PROFILING)
        with pytest.raises(RuntimeError, match="HMM profiling scaffold failed"):
            hop_expr(ltensor, rtensor, [mpo], cshape)
    finally:
        init_log(old_level or DEBUG)


def test_hop_expr_profiling_does_not_silently_drop_hmm_execution_metadata_failure(monkeypatch):
    from renormalizer.backend import gemm
    from renormalizer.mps.hop_expr import hop_expr
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    rng = _rng()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 6, 7))
    rtensor = rng.normal(size=(8, 7, 9))
    cshape = (4, 6, 9)
    old_level = package_logger.level

    def broken_metadata(*_args, **_kwargs):
        raise RuntimeError("synthetic metadata failure")

    monkeypatch.setattr(gemm, "hmm_task_execution_metadata", broken_metadata)

    try:
        init_log(PROFILING)
        with pytest.raises(RuntimeError, match="HMM profiling execution metadata failed"):
            hop_expr(ltensor, rtensor, [mpo], cshape)
    finally:
        init_log(old_level or DEBUG)


def test_batched_single_site_hop_expr_executor_profiles_rhs_loop_counts(tmp_path):
    from renormalizer.mps.backend import backend
    from renormalizer.mps.hop_expr import batched_hop_expr
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    rng = _rng()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 6, 7))
    rtensor = rng.normal(size=(8, 7, 9))
    cshape = (4, 6, 9)
    center_batch = rng.normal(size=cshape + (3,))
    old_level = package_logger.level
    event_path = tmp_path / "events.jsonl"
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        expr = batched_hop_expr(ltensor, rtensor, [mpo], cshape, nrhs=3)
        result = expr(center_batch)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert result.shape == (2, 5, 8, 3)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    plan_event = next(
        payload
        for payload in payloads
        if payload["event"] == "contraction_plan"
        and payload["equation"] == "abc,bdef,lfk,cekr->adlr"
        and payload["lowering"] == "hmm_task"
    )
    execute_event = next(
        payload
        for payload in payloads
        if payload["event"] == "contraction_execute"
        and payload["lowering"] == "hmm_executor"
        and payload["equation"].replace(" ", "") == "abc,bdef,lfk,cekr->adlr"
    )

    assert execute_event["rhs_batch_mode"] == plan_event["rhs_batch_mode"]
    assert execute_event["num_rhs"] == 3
    assert execute_event["num_rhs_loop_calls"] == 0
    assert execute_event["hmm_rhs_batch_mode"] == plan_event["rhs_batch_mode"]
    assert execute_event["hmm_num_rhs"] == plan_event["num_rhs"]
    assert execute_event["hmm_num_rhs_loop_calls"] == plan_event["num_rhs_loop_calls"]
    assert execute_event["num_batched_gemm"] == 3
    assert execute_event["planned_num_batched_gemm"] == 3
    assert execute_event["actual_num_batched_gemm"] == 3
    assert execute_event["actual_num_loop_matmul"] == 0
    assert execute_event["actual_lowerings"] == ["batched_gemm"]
    assert execute_event["fallback_reason"] is None
    assert execute_event["hmm_execution_trace"][1]["execution_primitives"] == ["batched_matmul"]
    assert execute_event["hmm_execution_trace"][1]["execution_policies"] == ["backend_batched_matmul"]
    assert execute_event["hmm_execution_trace"][1]["fallback_reasons"] == []
    assert execute_event["hmm_batches"][0]["stages"][0]["execution_primitives"] == ["batched_matmul"]
    assert execute_event["hmm_batches"][0]["stages"][0]["execution_policies"] == ["backend_batched_matmul"]
    assert execute_event["hmm_batches"][0]["stages"][0]["fallback_reasons"] == []

    profile = backend.last_execution_profile()
    assert profile["event"] == "contraction_execute"
    assert profile["lowering"] == "hmm_executor"
    assert profile["rhs_batch_mode"] == "r"
    assert profile["num_rhs"] == 3
    assert profile["num_rhs_loop_calls"] == 0
    assert profile["hmm_rhs_batch_mode"] == plan_event["rhs_batch_mode"]
    assert profile["hmm_num_rhs_loop_calls"] == plan_event["num_rhs_loop_calls"]


def test_batched_hop_expr_executor_propagates_runtime_fallback_profile(tmp_path):
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.mps.backend import backend
    from renormalizer.mps.hop_expr import batched_hop_expr
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    class NoBatchedBackend(NumpyBackend):
        supports_batched_matmul = False

        def batched_matmul(self, *args, **kwargs):
            raise AssertionError("HMM executor fallback must not call backend batched_matmul")

    rng = _rng()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 6, 7))
    rtensor = rng.normal(size=(8, 7, 9))
    cshape = (4, 6, 9)
    center_batch = rng.normal(size=cshape + (3,))
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    old_backend = backend.current
    backend._manager.current = NoBatchedBackend(config=BackendConfig(fallback_policy="record"))
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        expr = batched_hop_expr(ltensor, rtensor, [mpo], cshape, nrhs=3)
        result = expr(center_batch)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)
        backend._manager.current = old_backend

    expected = np.einsum("abc,bdef,lfk,cekr->adlr", ltensor, mpo, rtensor, center_batch)
    assert np.allclose(result, expected)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    hmm_task_event = next(
        payload
        for payload in payloads
        if payload["event"] == "contraction_execute"
        and payload["lowering"] == "hmm_task"
        and payload["equation"] == "abc,bdef,lfk,cekr->adlr"
    )
    hmm_executor_event = next(
        payload
        for payload in payloads
        if payload["event"] == "contraction_execute"
        and payload["lowering"] == "hmm_executor"
        and payload["equation"].replace(" ", "") == "abc,bdef,lfk,cekr->adlr"
    )

    assert hmm_task_event["fallback_reason"] == "backend lacks batched_matmul for batch shape (3,)"
    assert hmm_executor_event["fallback_reason"] == hmm_task_event["fallback_reason"]
    assert hmm_executor_event["fallback_from"] == "batched_gemm"
    assert hmm_executor_event["fallback_to"] == "loop_matmul"
    assert hmm_executor_event["fallback_sources"] == ["batched_gemm"]
    assert hmm_executor_event["fallback_targets"] == ["loop_matmul"]
    assert hmm_executor_event["fallback_policies"] == ["record"]
    assert hmm_executor_event["requires_grouped_gemm_fallback"] is False
    assert hmm_executor_event["grouped_gemm_policies"] == []
    assert hmm_executor_event["grouped_gemm_implementations"] == []
    assert hmm_executor_event["planned_num_batched_gemm"] == 3
    assert hmm_executor_event["actual_num_batched_gemm"] == 0
    assert hmm_executor_event["actual_num_loop_matmul"] == 3
    assert hmm_executor_event["hmm_execution_trace"][1]["actual_lowerings"] == ["fallback_tensordot"]
    assert hmm_executor_event["hmm_execution_trace"][1]["fallback_reason"] == hmm_task_event["fallback_reason"]
    assert hmm_executor_event["hmm_execution_trace"][1]["fallback_from"] == "batched_gemm"
    assert hmm_executor_event["hmm_execution_trace"][1]["fallback_to"] == "loop_matmul"
    assert hmm_executor_event["hmm_execution_trace"][1]["fallback_policy"] == "record"
    assert hmm_executor_event["hmm_execution_trace"][1]["actual_num_loop_matmul"] == 1


def test_batched_hmm_action_records_batched_matmul_capability_fallback(tmp_path):
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.gemm import execute_single_site_hmm_action
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    class NoBatchedBackend(NumpyBackend):
        supports_batched_matmul = False

        def batched_matmul(self, *args, **kwargs):
            raise AssertionError("HMM fallback must not call backend batched_matmul")

    rng = _rng()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 6, 7))
    rtensor = rng.normal(size=(8, 7, 9))
    cshape = (4, 6, 9)
    center_batch = rng.normal(size=cshape + (3,))
    backend = NoBatchedBackend(config=BackendConfig(fallback_policy="record"))
    old_level = package_logger.level
    event_path = tmp_path / "events.jsonl"
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = execute_single_site_hmm_action(
            backend,
            ltensor,
            mpo,
            rtensor,
            center_batch,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    expected = np.einsum("abc,bdef,lfk,cekr->adlr", ltensor, mpo, rtensor, center_batch)
    assert np.allclose(result, expected)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(
        payload
        for payload in payloads
        if payload["event"] == "contraction_execute"
        and payload["lowering"] == "hmm_task"
        and payload["equation"] == "abc,bdef,lfk,cekr->adlr"
    )
    assert event["fallback_from"] == "batched_gemm"
    assert event["fallback_to"] == "loop_matmul"
    assert event["fallback_policy"] == "record"
    assert event["fallback_reason"] == "backend lacks batched_matmul for batch shape (3,)"
    assert event["fallback_reasons"] == [event["fallback_reason"]]
    assert event["fallback_sources"] == ["batched_gemm"]
    assert event["fallback_targets"] == ["loop_matmul"]
    assert event["fallback_policies"] == ["record"]
    assert event["requires_grouped_gemm_fallback"] is False
    assert event["grouped_gemm_policies"] == []
    assert event["grouped_gemm_implementations"] == []
    assert event["execution_primitives"] == ["matmul"]
    assert event["execution_policies"] == ["loop_matmul"]
    assert event["planned_num_batched_gemm"] == 3
    assert event["actual_num_batched_gemm"] == 0
    assert event["actual_num_loop_matmul"] == 3
    assert event["actual_num_gemm"] == 0
    assert event["hmm_execution_trace"][1]["fallback_reasons"] == [event["fallback_reason"]]
    assert event["hmm_execution_trace"][1]["fallback_reason"] == event["fallback_reason"]
    assert event["hmm_execution_trace"][1]["fallback_from"] == "batched_gemm"
    assert event["hmm_execution_trace"][1]["fallback_to"] == "loop_matmul"
    assert event["hmm_execution_trace"][1]["fallback_policy"] == "record"
    assert event["hmm_execution_trace"][1]["fallback_sources"] == ["batched_gemm"]
    assert event["hmm_execution_trace"][1]["fallback_targets"] == ["loop_matmul"]
    assert event["hmm_execution_trace"][1]["fallback_policies"] == ["record"]
    assert event["hmm_execution_trace"][1]["actual_lowerings"] == ["fallback_tensordot"]
    assert event["hmm_execution_trace"][1]["actual_num_loop_matmul"] == 1
    assert event["hmm_batches"][0]["stages"][0]["fallback_reasons"] == [event["fallback_reason"]]
    assert event["hmm_batches"][0]["stages"][0]["fallback_reason"] == event["fallback_reason"]
    assert event["hmm_batches"][0]["stages"][0]["fallback_from"] == "batched_gemm"
    assert event["hmm_batches"][0]["stages"][0]["fallback_to"] == "loop_matmul"
    assert event["hmm_batches"][0]["stages"][0]["fallback_policy"] == "record"
    assert event["hmm_batches"][0]["stages"][0]["fallback_sources"] == ["batched_gemm"]
    assert event["hmm_batches"][0]["stages"][0]["fallback_targets"] == ["loop_matmul"]
    assert event["hmm_batches"][0]["stages"][0]["fallback_policies"] == ["record"]
    assert event["hmm_batches"][0]["stages"][0]["actual_lowerings"] == ["fallback_tensordot"]
    assert event["hmm_batches"][0]["stages"][0]["actual_num_loop_matmul"] == 1


def test_scalar_hmm_action_records_matmul_capability_fallback(tmp_path):
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.gemm import execute_single_site_hmm_action
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    class NoMatmulBackend(NumpyBackend):
        supports_matmul = False

        def matmul(self, *args, **kwargs):
            raise AssertionError("HMM fallback must not call backend matmul")

    rng = _rng()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 6, 7))
    rtensor = rng.normal(size=(8, 7, 9))
    cshape = (4, 6, 9)
    center = rng.normal(size=cshape)
    backend = NoMatmulBackend(config=BackendConfig(fallback_policy="record"))
    old_level = package_logger.level
    event_path = tmp_path / "events.jsonl"
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = execute_single_site_hmm_action(
            backend,
            ltensor,
            mpo,
            rtensor,
            center,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    expected = np.einsum("abc,bdef,lfk,cek->adl", ltensor, mpo, rtensor, center)
    assert np.allclose(result, expected)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(
        payload
        for payload in payloads
        if payload["event"] == "contraction_execute"
        and payload["lowering"] == "hmm_task"
        and payload["equation"] == "abc,bdef,lfk,cek->adl"
    )
    assert event["fallback_from"] == "gemm"
    assert event["fallback_to"] == "tensordot"
    assert event["fallback_policy"] == "record"
    assert event["fallback_reason"] == "backend lacks matmul"
    assert event["fallback_reasons"] == [event["fallback_reason"]]
    assert event["execution_primitives"] == ["tensordot"]
    assert event["execution_policies"] == ["tensordot"]
    assert event["planned_num_gemm"] == 3
    assert event["actual_num_gemm"] == 0
    assert event["actual_num_tensordot"] == 3
    assert event["hmm_execution_trace"][1]["fallback_reasons"] == [event["fallback_reason"]]
    assert event["hmm_execution_trace"][1]["fallback_reason"] == event["fallback_reason"]
    assert event["hmm_execution_trace"][1]["fallback_from"] == "gemm"
    assert event["hmm_execution_trace"][1]["fallback_to"] == "tensordot"
    assert event["hmm_execution_trace"][1]["fallback_policy"] == "record"
    assert event["hmm_execution_trace"][1]["actual_lowerings"] == ["fallback_tensordot"]
    assert event["hmm_execution_trace"][1]["actual_num_tensordot"] == 1
    assert event["hmm_batches"][0]["stages"][0]["fallback_reasons"] == [event["fallback_reason"]]
    assert event["hmm_batches"][0]["stages"][0]["fallback_reason"] == event["fallback_reason"]
    assert event["hmm_batches"][0]["stages"][0]["fallback_from"] == "gemm"
    assert event["hmm_batches"][0]["stages"][0]["fallback_to"] == "tensordot"
    assert event["hmm_batches"][0]["stages"][0]["fallback_policy"] == "record"
    assert event["hmm_batches"][0]["stages"][0]["actual_lowerings"] == ["fallback_tensordot"]
    assert event["hmm_batches"][0]["stages"][0]["actual_num_tensordot"] == 1


def test_batched_hop_expr_matches_stacked_two_site_results():
    from renormalizer.mps.hop_expr import batched_hop_expr, hop_expr

    rng = _rng()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo0 = rng.normal(size=(3, 5, 6, 7))
    mpo1 = rng.normal(size=(7, 8, 9, 10))
    rtensor = rng.normal(size=(11, 10, 12))
    cshape = (4, 6, 9, 12)
    center_batch = rng.normal(size=cshape + (3,))

    expr = hop_expr(ltensor, rtensor, [mpo0, mpo1], cshape)
    batched_expr = batched_hop_expr(
        ltensor, rtensor, [mpo0, mpo1], cshape, nrhs=3
    )

    expected = np.stack(
        [expr(center_batch[..., i]) for i in range(center_batch.shape[-1])],
        axis=-1,
    )
    actual = batched_expr(center_batch)

    assert actual.shape == expected.shape
    assert np.allclose(actual, expected)


def test_two_site_hop_expr_executes_backend_hmm_matmul(monkeypatch):
    from renormalizer.mps.backend import backend
    from renormalizer.mps.hop_expr import hop_expr

    rng = _rng()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo0 = rng.normal(size=(3, 5, 6, 7))
    mpo1 = rng.normal(size=(7, 8, 9, 10))
    rtensor = rng.normal(size=(11, 10, 12))
    cshape = (4, 6, 9, 12)
    center = rng.normal(size=cshape)
    calls = {"matmul": 0, "batched_matmul": 0}

    original_matmul = backend.current.matmul
    original_batched = backend.current.batched_matmul

    def counted_matmul(*args, **kwargs):
        if not (len(args) == 1 and backend.current._is_matmul_desc(args[0])):
            calls["matmul"] += 1
        return original_matmul(*args, **kwargs)

    def counted_batched_matmul(*args, **kwargs):
        if not (len(args) == 1 and backend.current._is_matmul_desc(args[0])):
            calls["batched_matmul"] += 1
        return original_batched(*args, **kwargs)

    monkeypatch.setattr(backend.current, "matmul", counted_matmul)
    monkeypatch.setattr(backend.current, "batched_matmul", counted_batched_matmul)

    expr = hop_expr(ltensor, rtensor, [mpo0, mpo1], cshape)
    actual = expr(center)
    expected = np.einsum("abc,bdef,fghj,ljk,cehk->adgl", ltensor, mpo0, mpo1, rtensor, center)

    assert np.allclose(actual, expected)
    assert calls == {"matmul": 4, "batched_matmul": 0}
    assert getattr(expr, "hmm_executor", None) == "two_site"


def test_batched_two_site_hop_expr_executes_backend_hmm_batched_matmul(monkeypatch):
    from renormalizer.mps.backend import backend
    from renormalizer.mps.hop_expr import batched_hop_expr

    rng = _rng()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo0 = rng.normal(size=(3, 5, 6, 7))
    mpo1 = rng.normal(size=(7, 8, 9, 10))
    rtensor = rng.normal(size=(11, 10, 12))
    cshape = (4, 6, 9, 12)
    center_batch = rng.normal(size=cshape + (2,))
    calls = {"matmul": 0, "batched_matmul": 0}

    original_matmul = backend.current.matmul
    original_batched = backend.current.batched_matmul

    def counted_matmul(*args, **kwargs):
        if not (len(args) == 1 and backend.current._is_matmul_desc(args[0])):
            calls["matmul"] += 1
        return original_matmul(*args, **kwargs)

    def counted_batched_matmul(*args, **kwargs):
        if not (len(args) == 1 and backend.current._is_matmul_desc(args[0])):
            calls["batched_matmul"] += 1
        return original_batched(*args, **kwargs)

    monkeypatch.setattr(backend.current, "matmul", counted_matmul)
    monkeypatch.setattr(backend.current, "batched_matmul", counted_batched_matmul)

    expr = batched_hop_expr(ltensor, rtensor, [mpo0, mpo1], cshape, nrhs=2)
    actual = expr(center_batch)
    expected = np.einsum("abc,bdef,fghj,ljk,cehkr->adglr", ltensor, mpo0, mpo1, rtensor, center_batch)

    assert np.allclose(actual, expected)
    assert calls == {"matmul": 0, "batched_matmul": 4}
    assert getattr(expr, "hmm_executor", None) == "two_site"


def test_batched_two_site_hop_expr_profiles_batched_hmm_scaffold(tmp_path):
    from renormalizer.mps.hop_expr import batched_hop_expr
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    rng = _rng()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo0 = rng.normal(size=(3, 5, 6, 7))
    mpo1 = rng.normal(size=(7, 8, 9, 10))
    rtensor = rng.normal(size=(11, 10, 12))
    cshape = (4, 6, 9, 12)
    center_batch = rng.normal(size=cshape + (2,))
    old_level = package_logger.level
    event_path = tmp_path / "events.jsonl"
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        expr = batched_hop_expr(
            ltensor,
            rtensor,
            [mpo0, mpo1],
            cshape,
            nrhs=2,
        )
        result = expr(center_batch)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert result.shape == (2, 5, 8, 11, 2)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    build_event = next(
        payload
        for payload in payloads
        if payload["event"] == "hmm_task_build"
        and payload["equation"] == "abc,bdef,fghj,ljk,cehkr->adglr"
    )
    assert build_event["center_kind"] == "twodot"
    assert build_event["rhs_batch_mode"] == "r"
    assert build_event["num_rhs"] == 2
    assert build_event["num_rhs_loop_calls"] == 0
    assert build_event["num_hx_blocks"] == 1
    assert build_event["num_gemm_desc"] == 4
    assert build_event["num_batched_gemm"] == 4
    assert build_event["num_shape_buckets"] == 4
    assert build_event["stage_group_sizes"] == [[1], [1], [1], [1]]
    assert build_event["shape_buckets"][0]["batch_shape"] == [2]
    assert build_event["shape_buckets"][0]["batch_count"] == 2
    assert build_event["output_shape"] == [2, 5, 8, 11, 2]

    plan_event = next(
        payload
        for payload in payloads
        if payload["event"] == "contraction_plan"
        and payload["equation"] == "abc,bdef,fghj,ljk,cehkr->adglr"
        and payload["lowering"] == "hmm_task"
    )
    assert plan_event["hmm_center_kind"] == "twodot"
    assert plan_event["rhs_batch_mode"] == "r"
    assert plan_event["num_rhs"] == 2
    assert plan_event["num_rhs_loop_calls"] == 0
    assert plan_event["step_count"] == 4
    assert plan_event["step_lowerings"] == [
        "batched_gemm",
        "batched_gemm",
        "batched_gemm",
        "batched_gemm",
    ]
    assert plan_event["num_gemm"] == 0
    assert plan_event["num_batched_gemm"] == 4
    assert plan_event["num_grouped_tasks"] == 0
    assert plan_event["num_shape_buckets"] == 4
    assert plan_event["output_shape"] == [2, 5, 8, 11, 2]
    assert plan_event["hmm_steps"][0]["descs"][0]["batch_shape"] == [2]
    assert plan_event["compute_profile"]["primary_kernel"] == "hmm_task"
    assert plan_event["compute_profile"]["work"]["num_batched_gemm"] == 4


def test_two_site_hop_expr_profiles_hmm_scaffold(tmp_path):
    from renormalizer.mps.hop_expr import hop_expr
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    rng = _rng()
    ltensor = rng.normal(size=(2, 3, 4))
    mpo0 = rng.normal(size=(3, 5, 6, 7))
    mpo1 = rng.normal(size=(7, 8, 9, 10))
    rtensor = rng.normal(size=(11, 10, 12))
    cshape = (4, 6, 9, 12)
    center = rng.normal(size=cshape)
    old_level = package_logger.level
    event_path = tmp_path / "events.jsonl"
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        expr = hop_expr(ltensor, rtensor, [mpo0, mpo1], cshape)
        result = expr(center)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert result.shape == (2, 5, 8, 11)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(
        payload
        for payload in payloads
        if payload["event"] == "hmm_task_build"
        and payload["equation"] == "abc,bdef,fghj,ljk,cehk->adgl"
    )
    assert event["center_kind"] == "twodot"
    assert event["num_hx_blocks"] == 1
    assert event["num_gemm_desc"] == 4
    assert event["stage_group_sizes"] == [[1], [1], [1], [1]]
    assert event["operand_shapes"] == [
        [2, 3, 4],
        [3, 5, 6, 7],
        [7, 8, 9, 10],
        [11, 10, 12],
        [4, 6, 9, 12],
    ]
    assert event["output_shape"] == [2, 5, 8, 11]
    plan_event = next(
        payload
        for payload in payloads
        if payload["event"] == "contraction_plan"
        and payload["equation"] == "abc,bdef,fghj,ljk,cehk->adgl"
        and payload["lowering"] == "hmm_task"
    )
    assert plan_event["hmm_center_kind"] == "twodot"
    assert plan_event["step_count"] == 4
    assert plan_event["step_lowerings"] == ["gemm", "gemm", "gemm", "gemm"]
    assert plan_event["matmul_plan_hashes"] == [
        step["plan_hash"] for step in plan_event["hmm_steps"]
    ]
    assert plan_event["num_gemm"] == 4
    assert plan_event["num_batched_gemm"] == 0
    assert plan_event["num_grouped_tasks"] == 0
    assert plan_event["num_shape_buckets"] == 4
    assert plan_event["num_gemv_desc"] == 0
    assert isinstance(plan_event["plan_hash"], str)
    assert plan_event["compute_profile"]["compute_class"] == "contraction_plan"
    assert plan_event["compute_profile"]["work"]["num_gemm"] == 4


def test_apply_hop_to_packed_vectors_uses_batched_expression_for_matrix_rhs():
    from renormalizer.mps.gs import _apply_hop_to_packed_vectors

    mask = np.array([[True, False, True], [False, True, False]])
    packed = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    calls = {"scalar": 0, "batched": 0}

    def scalar_expr(_struct):
        calls["scalar"] += 1
        raise AssertionError("scalar expression should not be used for matrix RHS")

    def batched_expr(struct):
        calls["batched"] += 1
        return struct + 1.0

    result = _apply_hop_to_packed_vectors(
        packed,
        mask,
        scalar_expr,
        batched_expr,
        inverse=2.0,
    )

    assert calls == {"scalar": 0, "batched": 1}
    assert np.array_equal(result, (packed + 1.0) * 2.0)


def test_apply_hop_to_packed_vectors_profiles_batched_rhs_execution(tmp_path):
    from renormalizer.mps.gs import _apply_hop_to_packed_vectors
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    old_level = package_logger.level
    event_path = tmp_path / "events.jsonl"
    mask = np.array([[True, False, True], [False, True, False]])
    packed = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    def scalar_expr(_struct):
        raise AssertionError("scalar expression should not be used for matrix RHS")

    def batched_expr(struct):
        return struct + 1.0

    batched_expr.equation = "abc,lbk,ckr->alr"

    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = _apply_hop_to_packed_vectors(
            packed,
            mask,
            scalar_expr,
            batched_expr,
            inverse=2.0,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.array_equal(result, (packed + 1.0) * 2.0)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(payload for payload in payloads if payload["event"] == "contraction_execute")
    assert event["backend"] == "numpy"
    assert event["equation"] == "abc,lbk,ckr->alr"
    assert event["lowering"] == "batched_rhs_hop"
    assert event["compute_class"] == "contraction_plan"
    assert event["compute_profile"]["compute_class"] == "contraction_plan"
    assert event["compute_subclass"] == "backend_execute"
    assert event["compute_role"] == "kernel"
    assert event["input_shapes"] == [[3, 2]]
    assert event["input_dtypes"] == ["float64"]
    assert event["output_shape"] == [3, 2]
    assert event["output_strides"] == [16, 8]
    assert event["output_order"] == "C"
    assert event["output_contiguous"] is True
    assert event["output_location"] == "host"
    assert event["layout_profile"]["output_layout"]["shape"] == [3, 2]
    assert event["layout_profile"]["output_layout"]["strides"] == [16, 8]
    assert event["layout_profile"]["output_layout"]["order"] == "C"
    assert event["layout_profile"]["output_layout"]["contiguous"] is True
    assert event["device_info"] == {
        "kind": "cpu",
        "index": None,
        "local_rank": None,
        "global_rank": None,
        "visible_id": None,
    }
    assert event["center_shape"] == [2, 3]
    assert event["packed_dim"] == 3
    assert event["qn_mask_true_count"] == 3
    assert event["center_tensor_shape"] == [2, 3, 2]
    assert event["output_center_shape"] == [2, 3, 2]
    assert event["operands"][0]["name"] == "packed_rhs"
    assert event["operands"][0]["modes"] == ["packed", "rhs"]
    assert event["operands"][0]["shape"] == [3, 2]
    assert event["operands"][0]["itemsize"] == packed.itemsize
    assert event["operands"][0]["is_host"] is True
    assert event["operands"][0]["is_distributed"] is False
    assert event["num_rhs"] == 2
    assert event["num_rhs_loop_calls"] == 0
    assert event["num_batched_gemm"] == 1
    assert event["execution_primitives"] == ["batched_rhs_expression"]
    assert event["execution_policies"] == ["backend_batched_rhs_expression"]
    assert event["fallback_reasons"] == []
    assert event["fallback_from"] is None
    assert event["fallback_to"] is None
    assert event["fallback_policy"] is None
    assert event["fallback_reason"] is None
    assert event["wall_s"] >= 0.0


def test_apply_hop_to_packed_vectors_profiles_batched_rhs_path_metadata(tmp_path):
    from renormalizer.mps.gs import _apply_hop_to_packed_vectors
    from renormalizer.mps.hop_expr import batched_hop_expr, hop_expr
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    rng = _rng()
    ltensor = rng.normal(size=(2, 3, 2))
    rtensor = rng.normal(size=(5, 3, 5))
    cshape = (2, 5)
    mask = np.ones(cshape, dtype=bool)
    packed = rng.normal(size=(int(mask.sum()), 2))
    expr = hop_expr(ltensor, rtensor, [], cshape)
    batched_expr = batched_hop_expr(ltensor, rtensor, [], cshape, nrhs=2)

    old_level = package_logger.level
    event_path = tmp_path / "events.jsonl"
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = _apply_hop_to_packed_vectors(
            packed,
            mask,
            expr,
            batched_expr,
            inverse=1.0,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert result.shape == packed.shape

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(
        payload
        for payload in payloads
        if payload["event"] == "contraction_execute" and payload["lowering"] == "batched_rhs_hop"
    )
    assert event["equation"].replace(" ", "") == "abc,lbk,ckr->alr"
    assert event["rhs_batch_mode"] == "r"
    assert event["path"]
    assert event["contraction_count"] == len(event["contraction_steps"])
    assert event["contraction_types"]
    assert event["flops"] > 0
    assert event["expr_largest_intermediate"] > 0
    assert event["num_batched_gemm"] >= 1
    assert event["num_rhs_loop_calls"] == 0
    assert event["execution_primitives"] == ["batched_matmul"]
    assert event["execution_policies"] == ["oe_batched_rhs_expression"]
    assert event["fallback_reasons"] == []
    assert any("r" in step["output_modes"] for step in event["contraction_steps"])
    assert event["fallback_reason"] is None


def test_apply_hop_to_packed_vectors_forbid_policy_rejects_generic_batched_rhs_fallback():
    from renormalizer.backend import BackendConfig, BackendFeatureError
    from renormalizer.mps.backend import set_backend
    from renormalizer.mps.gs import _apply_hop_to_packed_vectors

    set_backend("numpy", config=BackendConfig(fallback_policy="forbid"))
    mask = np.array([[True, False, True], [False, True, False]])
    packed = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    calls = {"scalar": 0, "batched": 0}

    def scalar_expr(_struct):
        calls["scalar"] += 1
        raise AssertionError("scalar expression should not be used for matrix RHS")

    def batched_expr(struct):
        calls["batched"] += 1
        return struct + 1.0

    batched_expr.equation = "ab,bc->acr"
    batched_expr.path_summary = {
        "flop_count": 12,
        "largest_intermediate": 6,
        "contraction_types": ["GEMM"],
        "contraction_steps": [
            {
                "contraction_type": "GEMM",
                "input_modes": [["a", "b"], ["b", "c"]],
                "output_modes": ["a", "c"],
            }
        ],
    }

    try:
        with pytest.raises(BackendFeatureError, match="batched RHS expression path has no GEMM carrying RHS mode"):
            _apply_hop_to_packed_vectors(
                packed,
                mask,
                scalar_expr,
                batched_expr,
                inverse=1.0,
            )
    finally:
        set_backend("numpy")

    assert calls == {"scalar": 0, "batched": 0}


def test_apply_hop_to_packed_vectors_warn_policy_warns_for_generic_batched_rhs_fallback():
    from renormalizer.backend import BackendConfig
    from renormalizer.mps.backend import set_backend
    from renormalizer.mps.gs import _apply_hop_to_packed_vectors

    set_backend("numpy", config=BackendConfig(fallback_policy="warn"))
    mask = np.array([[True, False, True], [False, True, False]])
    packed = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    def scalar_expr(_struct):
        raise AssertionError("scalar expression should not be used for matrix RHS")

    def batched_expr(struct):
        return struct + 1.0

    batched_expr.equation = "ab,bc->acr"
    batched_expr.path_summary = {
        "flop_count": 12,
        "largest_intermediate": 6,
        "contraction_types": ["GEMM"],
        "contraction_steps": [
            {
                "contraction_type": "GEMM",
                "input_modes": [["a", "b"], ["b", "c"]],
                "output_modes": ["a", "c"],
            }
        ],
    }

    try:
        with pytest.warns(RuntimeWarning, match="batched RHS expression path has no GEMM carrying RHS mode"):
            result = _apply_hop_to_packed_vectors(
                packed,
                mask,
                scalar_expr,
                batched_expr,
                inverse=2.0,
            )
    finally:
        set_backend("numpy")

    assert np.array_equal(result, (packed + 1.0) * 2.0)


def test_apply_hop_to_packed_vectors_links_execution_to_hmm_plan(tmp_path):
    from renormalizer.mps.backend import backend
    from renormalizer.mps.gs import _apply_hop_to_packed_vectors
    from renormalizer.mps.hop_expr import batched_hop_expr, hop_expr
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    rng = _rng()
    ltensor = rng.normal(size=(4, 3, 4))
    mpo = rng.normal(size=(3, 6, 6, 7))
    rtensor = rng.normal(size=(9, 7, 9))
    cshape = (4, 6, 9)
    mask = np.ones(cshape, dtype=bool)
    packed = rng.normal(size=(int(mask.sum()), 2))

    old_level = package_logger.level
    event_path = tmp_path / "events.jsonl"
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        expr = hop_expr(ltensor, rtensor, [mpo], cshape)
        batched_expr = batched_hop_expr(ltensor, rtensor, [mpo], cshape, nrhs=2)
        result = _apply_hop_to_packed_vectors(
            packed,
            mask,
            expr,
            batched_expr,
            inverse=1.0,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert result.shape == packed.shape

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    plan_event = next(
        payload
        for payload in payloads
        if payload["event"] == "contraction_plan"
        and payload["equation"] == "abc,bdef,lfk,cekr->adlr"
        and payload["lowering"] == "hmm_task"
    )
    execute_event = next(
        payload
        for payload in payloads
        if payload["event"] == "contraction_execute"
        and payload["lowering"] == "batched_rhs_hop"
        and payload["equation"].replace(" ", "") == "abc,bdef,lfk,cekr->adlr"
    )

    assert execute_event["hmm_plan_hash"] == plan_event["plan_hash"]
    assert execute_event["hmm_matmul_plan_hashes"] == plan_event["matmul_plan_hashes"]
    assert execute_event["hmm_step_lowerings"] == plan_event["step_lowerings"]
    assert execute_event["hmm_num_shape_buckets"] == plan_event["num_shape_buckets"]
    assert execute_event["hmm_num_batched_gemm"] == plan_event["num_batched_gemm"]
    assert execute_event["hmm_steps"] == plan_event["hmm_steps"]
    assert execute_event["hmm_shape_buckets"] == plan_event["shape_buckets"]
    assert execute_event["hmm_gemv_shape_buckets"] == plan_event["hmm_gemv_shape_buckets"]
    assert execute_event["hmm_batches"] == plan_event["hmm_batches"]
    assert execute_event["hmm_hx_blocks"] == plan_event["hmm_hx_blocks"]
    assert execute_event["hmm_execution_trace"] == plan_event["hmm_execution_trace"]
    assert [item["phase"] for item in execute_event["hmm_execution_trace"]] == [
        "inter_gemv",
        "gemm_stage",
        "gemm_stage",
        "gemm_stage",
        "reduce_gemv",
    ]
    assert execute_event["hmm_phase_counts"] == {
        "inter_gemv": 1,
        "gemm_stage": 3,
        "reduce_gemv": 1,
    }
    assert execute_event["hmm_phase_tasks"]["gemm_stage"] == 3
    assert execute_event["hmm_phase_groups"]["gemm_stage"] == 3
    assert sum(execute_event["hmm_phase_flops"].values()) == plan_event["flops"]
    assert execute_event["num_rhs"] == 2
    assert execute_event["num_rhs_loop_calls"] == 0
    assert execute_event["hmm_num_rhs_loop_calls"] == plan_event["num_rhs_loop_calls"]
    assert execute_event["hmm_flops"] == plan_event["flops"]
    assert execute_event["hmm_read_bytes"] == plan_event["read_bytes"]
    assert execute_event["hmm_write_bytes"] == plan_event["write_bytes"]
    assert execute_event["hmm_copy_bytes"] == plan_event["copy_bytes"]
    assert execute_event["hmm_workspace_bytes"] == plan_event["workspace_bytes"]
    assert execute_event["workspace_bytes"] == plan_event["workspace_bytes"]
    assert execute_event["hmm_largest_intermediate"] == plan_event["largest_intermediate"]
    assert execute_event["largest_intermediate"] == plan_event["largest_intermediate_bytes"]
    assert execute_event["largest_intermediate_elements"] == plan_event["largest_intermediate_elements"]
    assert execute_event["largest_intermediate_bytes"] == plan_event["largest_intermediate_bytes"]
    assert execute_event["hmm_largest_intermediate_elements"] == plan_event["largest_intermediate_elements"]
    assert execute_event["hmm_largest_intermediate_bytes"] == plan_event["largest_intermediate_bytes"]

    profile = backend.last_execution_profile()
    assert profile["event"] == "contraction_execute"
    assert profile["lowering"] == "batched_rhs_hop"
    assert profile["compute_class"] == "contraction_plan"
    assert profile["compute_profile"]["compute_class"] == "contraction_plan"
    assert profile["compute_profile"]["workload_signature"]["compute_class"] == "contraction_plan"
    assert profile["equation"] == execute_event["equation"]
    assert profile["hmm_plan_hash"] == plan_event["plan_hash"]
    assert profile["hmm_step_lowerings"] == plan_event["step_lowerings"]
    assert profile["hmm_gemv_shape_buckets"] == plan_event["hmm_gemv_shape_buckets"]
    assert profile["hmm_batches"][0]["stages"][0]["planned_lowerings"] == ["batched_gemm"]
    assert profile["hmm_batches"][0]["stages"][0]["execution_primitives"] == ["batched_matmul"]
    assert profile["hmm_batches"][0]["stages"][0]["group_keys"][0]["batch_shape"] == (2,)
    assert [item["phase"] for item in profile["hmm_execution_trace"]] == [
        "inter_gemv",
        "gemm_stage",
        "gemm_stage",
        "gemm_stage",
        "reduce_gemv",
    ]
    assert profile["hmm_execution_trace"][1]["execution_primitives"] == ["batched_matmul"]
    assert profile["hmm_phase_counts"] == execute_event["hmm_phase_counts"]
    assert profile["num_rhs"] == 2
    assert profile["num_rhs_loop_calls"] == 0
    assert profile["hmm_num_rhs_loop_calls"] == plan_event["num_rhs_loop_calls"]
    assert profile["hmm_flops"] == plan_event["flops"]
    assert profile["hmm_read_bytes"] == plan_event["read_bytes"]
    assert profile["hmm_write_bytes"] == plan_event["write_bytes"]
    assert profile["hmm_copy_bytes"] == plan_event["copy_bytes"]
    assert profile["hmm_workspace_bytes"] == plan_event["workspace_bytes"]
    assert profile["workspace_bytes"] == plan_event["workspace_bytes"]
    assert profile["hmm_largest_intermediate"] == plan_event["largest_intermediate"]
    assert profile["largest_intermediate"] == plan_event["largest_intermediate_bytes"]
    assert profile["largest_intermediate_elements"] == plan_event["largest_intermediate_elements"]
    assert profile["largest_intermediate_bytes"] == plan_event["largest_intermediate_bytes"]
    assert profile["hmm_largest_intermediate_elements"] == plan_event["largest_intermediate_elements"]
    assert profile["hmm_largest_intermediate_bytes"] == plan_event["largest_intermediate_bytes"]


def test_apply_hop_to_packed_vectors_keeps_scalar_expression_for_vector_rhs():
    from renormalizer.mps.gs import _apply_hop_to_packed_vectors

    mask = np.array([[True, False, True], [False, True, False]])
    packed = np.array([1.0, 2.0, 3.0])
    calls = {"scalar": 0, "batched": 0}

    def scalar_expr(struct):
        calls["scalar"] += 1
        return struct + 1.0

    def batched_expr(_struct):
        calls["batched"] += 1
        raise AssertionError("batched expression should not be used for vector RHS")

    result = _apply_hop_to_packed_vectors(
        packed,
        mask,
        scalar_expr,
        batched_expr,
        inverse=2.0,
    )

    assert calls == {"scalar": 1, "batched": 0}
    assert np.array_equal(result, (packed + 1.0) * 2.0)
