# -*- coding: utf-8 -*-

import json

import numpy as np

import renormalizer.mps.svd_qn as svd_qn_module
from renormalizer.utils import log, profiling


def _no_qn_inputs():
    coef = np.arange(24, dtype=float).reshape(2, 3, 4)
    qnbigl = np.zeros((2, 3, 1), dtype=int)
    qnbigr = np.zeros((4, 1), dtype=int)
    qntot = np.zeros(1, dtype=int)
    return coef, qnbigl, qnbigr, qntot


def test_svd_qn_no_qn_svd_uses_dense_fast_path(monkeypatch):
    coef, qnbigl, qnbigr, qntot = _no_qn_inputs()

    def forbidden_get_qn_mask(*args, **kwargs):
        raise AssertionError("single-block no-QN SVD should not enumerate QN masks")

    monkeypatch.setattr(svd_qn_module, "get_qn_mask", forbidden_get_qn_mask)
    u, su, qnlset, v, sv, qnrset = svd_qn_module.svd_qn(
        coef,
        qnbigl,
        qnbigr,
        qntot,
        full_matrices=False,
    )

    matrix = coef.reshape(6, 4)
    assert np.allclose((u * su) @ v.T, matrix)
    assert np.allclose(su, sv)
    assert qnlset == [(0,)] * len(su)
    assert qnrset == [(0,)] * len(su)


def test_svd_qn_no_qn_qr_uses_dense_fast_path(monkeypatch):
    coef, qnbigl, qnbigr, qntot = _no_qn_inputs()

    def forbidden_get_qn_mask(*args, **kwargs):
        raise AssertionError("single-block no-QN QR should not enumerate QN masks")

    monkeypatch.setattr(svd_qn_module, "get_qn_mask", forbidden_get_qn_mask)
    u, qnlset, v, qnrset = svd_qn_module.svd_qn(
        coef,
        qnbigl,
        qnbigr,
        qntot,
        QR=True,
        system="L",
        full_matrices=False,
    )

    matrix = coef.reshape(6, 4)
    assert np.allclose(u @ v.T, matrix)
    assert qnlset == [(0,)] * u.shape[1]
    assert qnrset == [(0,)] * u.shape[1]


def test_svd_qn_no_qn_fast_path_profiles_single_dense_block(tmp_path):
    coef, qnbigl, qnbigr, qntot = _no_qn_inputs()
    previous_level = log.getLogger().level
    event_path = tmp_path / "profile-events.jsonl"
    log.init_log(log.PROFILING)
    profiling.register_event_output(event_path)
    try:
        svd_qn_module.svd_qn(
            coef,
            qnbigl,
            qnbigr,
            qntot,
            full_matrices=False,
        )
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()
        log.getLogger().setLevel(previous_level)

    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    svd_event = next(event for event in events if event["event"] == "svd_qn")
    assert svd_event["block_count"] == 1
    assert svd_event["unique_block_shape_count"] == 1
    assert svd_event["top_block_shape_groups"][0]["shape"] == "6x4"
