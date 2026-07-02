# -*- coding: utf-8 -*-

import json

import numpy as np


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
    assert event["input_shapes"] == [[3, 2]]
    assert event["input_dtypes"] == ["float64"]
    assert event["output_shape"] == [3, 2]
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
    assert event["num_batched_gemm"] == 1
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
    assert any("r" in step["output_modes"] for step in event["contraction_steps"])
    assert event["fallback_reason"] is None


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
