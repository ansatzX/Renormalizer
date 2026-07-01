# -*- coding: utf-8 -*-

import json


def test_parse_backend_names_expands_all_without_cupynumeric_by_default():
    from renormalizer.backend.gemm_smoke import parse_backend_names

    assert parse_backend_names("numpy,cupy") == ["numpy", "cupy"]
    assert parse_backend_names("all") == ["numpy", "cupy", "torch", "jax"]
    assert parse_backend_names("all,cupynumeric") == ["numpy", "cupy", "torch", "jax", "cupynumeric"]


def test_numpy_gemm_smoke_records_json_serializable_correctness():
    from renormalizer.backend.gemm_smoke import run_backend_smoke

    records = run_backend_smoke(
        "numpy",
        device="cpu",
        batch=4,
        small_dim=4,
        medium_dim=8,
        repeat=1,
        pack_threshold=2,
    )

    operations = {record["operation"] for record in records}
    assert operations == {"matmul", "batched_matmul", "grouped_gemm"}
    for record in records:
        assert record["backend"] == "numpy"
        assert record["device"] == "cpu"
        assert record["status"] == "passed"
        assert record["max_abs_error"] <= 1e-9
        assert record["wall_s"] >= 0.0
        assert record["gpu_count"] >= 0
        assert "cuda_visible_devices" in record
        json.dumps(record)
