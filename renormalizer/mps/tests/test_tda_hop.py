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


def test_tda_multi_hop_records_matrix_rhs_loop_fallback(caplog, tmp_path):
    from renormalizer.mps.tda import _tda_multi_hop
    from renormalizer.utils import profiling
    from renormalizer.utils.log import PROFILING

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)
    calls = []

    def hop(vector):
        calls.append(vector.copy())
        return vector + 1.0

    x = np.arange(6, dtype=np.float64).reshape(3, 2)
    try:
        result = _tda_multi_hop(x, hop)
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
    assert event["operands"][0]["name"] == "packed_rhs"
    assert event["operands"][0]["modes"] == ["packed", "rhs"]
    assert event["operands"][0]["shape"] == [3, 2]
    assert event["operands"][0]["itemsize"] == x.itemsize
    assert event["operands"][0]["is_host"] is True
    assert event["operands"][0]["is_distributed"] is False
    assert event["num_rhs"] == 2
    assert event["num_rhs_loop_calls"] == 2
    assert event["num_batched_gemm"] == 0
    assert event["execution_primitives"] == ["rhs_loop"]
    assert event["execution_policies"] == ["fallback_rhs_loop"]
    assert event["fallback_reasons"] == ["tda matmat rebuilds ket-dependent environments per RHS"]
    assert event["fallback_from"] == "batched_rhs_hop"
    assert event["fallback_to"] == "rhs_loop"
    assert event["fallback_policy"] == "record"
    assert event["fallback_reason"] == "tda matmat rebuilds ket-dependent environments per RHS"
    assert event["wall_s"] >= 0.0


def test_tda_multi_hop_vector_rhs_uses_plain_hop_without_fallback_event(caplog, tmp_path):
    from renormalizer.mps.tda import _tda_multi_hop
    from renormalizer.utils import profiling
    from renormalizer.utils.log import PROFILING

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)
    calls = []

    def hop(vector):
        calls.append(vector.copy())
        return vector + 1.0

    x = np.arange(3, dtype=np.float64)
    try:
        result = _tda_multi_hop(x, hop)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    assert np.array_equal(result, x + 1.0)
    assert len(calls) == 1
    assert not event_path.exists() or "contraction_execute" not in event_path.read_text()


def test_tda_multi_hop_forbid_policy_rejects_rhs_loop_fallback():
    from renormalizer.backend import BackendConfig, BackendFeatureError
    from renormalizer.mps.backend import set_backend
    from renormalizer.mps.tda import _tda_multi_hop

    set_backend("numpy", config=BackendConfig(fallback_policy="forbid"))

    def hop(vector):
        return vector + 1.0

    x = np.arange(6, dtype=np.float64).reshape(3, 2)
    try:
        with pytest.raises(BackendFeatureError, match="tda matmat rebuilds ket-dependent environments per RHS"):
            _tda_multi_hop(x, hop)
    finally:
        set_backend("numpy")


def test_tda_multi_hop_warn_policy_warns_for_rhs_loop_fallback():
    from renormalizer.backend import BackendConfig
    from renormalizer.mps.backend import set_backend
    from renormalizer.mps.tda import _tda_multi_hop

    set_backend("numpy", config=BackendConfig(fallback_policy="warn"))

    def hop(vector):
        return vector + 1.0

    x = np.arange(6, dtype=np.float64).reshape(3, 2)
    try:
        with pytest.warns(RuntimeWarning, match="tda matmat rebuilds ket-dependent environments per RHS"):
            result = _tda_multi_hop(x, hop)
    finally:
        set_backend("numpy")

    assert np.array_equal(result, x + 1.0)


def test_tda_multi_hop_uses_vectorized_matmat_when_available(caplog, tmp_path):
    from renormalizer.mps.tda import _tda_multi_hop
    from renormalizer.utils import profiling
    from renormalizer.utils.log import PROFILING

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)
    hop_calls = []
    matmat_calls = []

    def hop(vector):
        hop_calls.append(vector.copy())
        return vector + 1.0

    def matmat(matrix):
        matmat_calls.append(matrix.copy())
        return matrix + 2.0

    x = np.arange(6, dtype=np.float64).reshape(3, 2)
    try:
        result = _tda_multi_hop(x, hop, matmat_hop=matmat)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    assert np.array_equal(result, x + 2.0)
    assert hop_calls == []
    assert len(matmat_calls) == 1

    event = next(
        payload
        for payload in _jsonl_payloads(event_path)
        if payload["event"] == "contraction_execute"
    )
    assert event["lowering"] == "batched_rhs_hop"
    assert event["num_rhs"] == 2
    assert event["num_rhs_loop_calls"] == 0
    assert event["output_strides"] == [16, 8]
    assert event["output_order"] == "C"
    assert event["output_contiguous"] is True
    assert event["output_location"] == "host"
    assert event["layout_profile"]["output_layout"]["order"] == "C"
    assert event["layout_profile"]["output_layout"]["contiguous"] is True
    assert event["execution_primitives"] == ["matmat_hop"]
    assert event["execution_policies"] == ["external_vectorized_matmat"]
    assert event["fallback_reasons"] == []
    assert event["fallback_reason"] is None
    assert event["fallback_from"] is None
    assert event["fallback_to"] is None
    assert event["fallback_policy"] is None
    assert event["num_batched_gemm"] == 1


def test_tda_multi_hop_rejects_vectorized_matmat_wrong_shape():
    from renormalizer.mps.tda import _tda_multi_hop

    x = np.arange(6, dtype=np.float64).reshape(3, 2)

    def hop(vector):
        return vector + 1.0

    def wrong_matmat(matrix):
        return np.ones((matrix.shape[0], matrix.shape[1] + 1))

    with pytest.raises(ValueError, match="TDA matmat_hop output shape must match input shape"):
        _tda_multi_hop(x, hop, matmat_hop=wrong_matmat)
