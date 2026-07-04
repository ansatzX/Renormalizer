# -*- coding: utf-8 -*-

import json

import numpy as np
import pytest


def _jsonl_payloads(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def test_tn_apply_hop_to_packed_vectors_records_matrix_rhs_loop_fallback(caplog, tmp_path):
    from renormalizer.tn.gs import _apply_tn_hop_to_packed_vectors
    from renormalizer.utils import profiling
    from renormalizer.utils.log import PROFILING

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)
    mask = np.array([[True, False], [True, True]])
    x = np.arange(6, dtype=np.float64).reshape(3, 2)
    calls = []

    def expr(struct):
        calls.append(struct.copy())
        return struct + 1.0

    try:
        result = _apply_tn_hop_to_packed_vectors(x, mask, expr)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    assert np.array_equal(result, x + 1.0)
    assert len(calls) == 2

    event = next(
        payload
        for payload in _jsonl_payloads(event_path)
        if payload["event"] == "contraction_execute"
    )
    assert event["backend"] == "numpy"
    assert event["lowering"] == "fallback_rhs_loop"
    assert event["compute_class"] == "contraction_plan"
    assert event["compute_profile"]["compute_class"] == "contraction_plan"
    assert event["compute_profile"]["workload_signature"]["compute_class"] == "contraction_plan"
    assert event["compute_subclass"] == "backend_execute"
    assert event["compute_role"] == "kernel"
    assert event["input_shapes"] == [[3, 2]]
    assert event["input_dtypes"] == ["float64"]
    assert event["output_shape"] == [3, 2]
    assert event["output_strides"] == [16, 8]
    assert event["output_order"] == "C"
    assert event["output_contiguous"] is True
    assert event["output_location"] == "host"
    assert event["operands"][0]["name"] == "packed_rhs"
    assert event["operands"][0]["modes"] == ["packed", "rhs"]
    assert event["operands"][0]["shape"] == [3, 2]
    assert event["layout_profile"]["output_layout"]["shape"] == [3, 2]
    assert event["layout_profile"]["output_layout"]["strides"] == [16, 8]
    assert event["layout_profile"]["output_layout"]["order"] == "C"
    assert event["layout_profile"]["output_layout"]["contiguous"] is True
    assert event["num_rhs"] == 2
    assert event["num_rhs_loop_calls"] == 2
    assert event["num_batched_gemm"] == 0
    assert event["execution_primitives"] == ["rhs_loop"]
    assert event["execution_policies"] == ["fallback_rhs_loop"]
    assert event["fallback_from"] == "batched_rhs_hop"
    assert event["fallback_to"] == "rhs_loop"
    assert event["fallback_policy"] == "record"
    assert event["fallback_reason"] == "tn hop expression does not yet support batched RHS"
    assert event["fallback_reasons"] == [event["fallback_reason"]]
    assert event["wall_s"] >= 0.0


def test_tn_apply_hop_to_packed_vectors_keeps_vector_rhs_unprofiled(caplog, tmp_path):
    from renormalizer.tn.gs import _apply_tn_hop_to_packed_vectors
    from renormalizer.utils import profiling
    from renormalizer.utils.log import PROFILING

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)
    mask = np.array([[True, False], [True, True]])
    x = np.arange(3, dtype=np.float64)
    calls = []

    def expr(struct):
        calls.append(struct.copy())
        return struct + 1.0

    try:
        result = _apply_tn_hop_to_packed_vectors(x, mask, expr)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    assert np.array_equal(result, x + 1.0)
    assert len(calls) == 1
    assert not event_path.exists() or "contraction_execute" not in event_path.read_text()


def test_tn_apply_hop_to_packed_vectors_forbid_policy_rejects_rhs_loop_fallback():
    from renormalizer.backend import BackendConfig, BackendFeatureError
    from renormalizer.mps.backend import set_backend
    from renormalizer.tn.gs import _apply_tn_hop_to_packed_vectors

    set_backend("numpy", config=BackendConfig(fallback_policy="forbid"))
    mask = np.array([[True, False], [True, True]])
    x = np.arange(6, dtype=np.float64).reshape(3, 2)

    def expr(struct):
        return struct + 1.0

    try:
        with pytest.raises(BackendFeatureError, match="tn hop expression does not yet support batched RHS"):
            _apply_tn_hop_to_packed_vectors(x, mask, expr)
    finally:
        set_backend("numpy")


def test_tn_apply_hop_to_packed_vectors_warn_policy_warns_for_rhs_loop_fallback():
    from renormalizer.backend import BackendConfig
    from renormalizer.mps.backend import set_backend
    from renormalizer.tn.gs import _apply_tn_hop_to_packed_vectors

    set_backend("numpy", config=BackendConfig(fallback_policy="warn"))
    mask = np.array([[True, False], [True, True]])
    x = np.arange(6, dtype=np.float64).reshape(3, 2)

    def expr(struct):
        return struct + 1.0

    try:
        with pytest.warns(RuntimeWarning, match="tn hop expression does not yet support batched RHS"):
            result = _apply_tn_hop_to_packed_vectors(x, mask, expr)
    finally:
        set_backend("numpy")

    assert np.array_equal(result, x + 1.0)
