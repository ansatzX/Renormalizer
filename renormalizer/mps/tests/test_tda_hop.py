# -*- coding: utf-8 -*-

import json

import numpy as np


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
    assert event["compute_class"] == "tensordot"
    assert event["compute_subclass"] == "backend_execute"
    assert event["compute_role"] == "kernel"
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
    assert event["operands"][0]["name"] == "packed_rhs"
    assert event["operands"][0]["modes"] == ["packed", "rhs"]
    assert event["operands"][0]["shape"] == [3, 2]
    assert event["operands"][0]["itemsize"] == x.itemsize
    assert event["operands"][0]["is_host"] is True
    assert event["operands"][0]["is_distributed"] is False
    assert event["num_rhs"] == 2
    assert event["num_rhs_loop_calls"] == 2
    assert event["num_batched_gemm"] == 0
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
