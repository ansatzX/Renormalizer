# -*- coding: utf-8 -*-

import json


def _json_lines(text):
    return [json.loads(line) for line in text.splitlines() if line.strip().startswith("{")]


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
    grouped = next(record for record in records if record["operation"] == "grouped_gemm")
    assert grouped["num_shape_buckets"] >= 1
    assert grouped["num_grouped_tasks"] == 4
    assert "num_batched_gemm" in grouped
    assert "num_gemm" in grouped
    assert "copy_bytes" in grouped
    assert "loop_wall_s" in grouped
    assert "grouped_wall_s" in grouped
    assert "speedup_vs_loop" in grouped
    assert grouped["loop_wall_s"] >= 0.0
    assert grouped["grouped_wall_s"] >= 0.0


def test_gemm_smoke_skips_backend_incompatible_device():
    from renormalizer.backend.gemm_smoke import run_backend_smoke

    records = run_backend_smoke(
        "numpy",
        device="cuda",
        batch=2,
        small_dim=2,
        medium_dim=2,
        repeat=1,
        pack_threshold=2,
    )

    assert len(records) == 1
    record = records[0]
    assert record["backend"] == "numpy"
    assert record["device"] == "cuda"
    assert record["operation"] == "backend_create"
    assert record["status"] == "skipped"
    assert record["gpu_count"] >= 0
    assert "cuda_visible_devices" in record
    assert "does not support device" in record["error"]


def test_time_call_uses_warmup_and_median_trial_time(monkeypatch):
    from renormalizer.backend import gemm_smoke

    class DummyBackend:
        def sync(self):
            return None

    calls = []

    def fn():
        calls.append(len(calls))
        return len(calls)

    times = iter([0.0, 10.0, 0.0, 4.0, 0.0, 6.0])
    monkeypatch.setattr(gemm_smoke.time, "perf_counter", lambda: next(times))

    result, wall_s, samples = gemm_smoke._time_call(
        DummyBackend(),
        repeat=2,
        fn=fn,
        warmup=1,
        trials=3,
    )

    assert result == 8
    assert wall_s == 6.0
    assert samples == [10.0, 4.0, 6.0]
    assert len(calls) == 8


def test_grouped_gemm_benchmark_gate_requires_speedup_and_batched_bucket():
    from renormalizer.backend.gemm_smoke import evaluate_grouped_gemm_gate

    records = [
        {
            "backend": "torch",
            "device": "gpu",
            "operation": "grouped_gemm",
            "status": "passed",
            "speedup_vs_loop": 1.35,
            "num_batched_gemm": 1,
            "num_grouped_tasks": 64,
            "num_shape_buckets": 1,
        },
        {
            "backend": "cupy",
            "device": "gpu",
            "operation": "grouped_gemm",
            "status": "passed",
            "speedup_vs_loop": 0.92,
            "num_batched_gemm": 1,
            "num_grouped_tasks": 64,
            "num_shape_buckets": 1,
        },
    ]

    summary = evaluate_grouped_gemm_gate(records, min_speedup=1.05, require_backends=("torch", "cupy"))

    assert summary["status"] == "failed"
    assert summary["checked_count"] == 2
    assert summary["min_speedup"] == 1.05
    assert summary["failures"] == [
        {
            "backend": "cupy",
            "device": "gpu",
            "reason": "speedup below threshold",
            "speedup_vs_loop": 0.92,
        }
    ]


def test_grouped_gemm_benchmark_gate_reports_missing_backend():
    from renormalizer.backend.gemm_smoke import evaluate_grouped_gemm_gate

    summary = evaluate_grouped_gemm_gate(
        [
            {
                "backend": "torch",
                "device": "gpu",
                "operation": "grouped_gemm",
                "status": "passed",
                "speedup_vs_loop": 1.2,
                "num_batched_gemm": 1,
                "num_grouped_tasks": 32,
                "num_shape_buckets": 1,
            }
        ],
        min_speedup=1.0,
        require_backends=("torch", "cupy"),
    )

    assert summary["status"] == "failed"
    assert summary["failures"] == [
        {"backend": "cupy", "device": None, "reason": "missing grouped_gemm record"}
    ]


def test_grouped_gemm_benchmark_gate_filters_to_required_backends():
    from renormalizer.backend.gemm_smoke import evaluate_grouped_gemm_gate

    summary = evaluate_grouped_gemm_gate(
        [
            {
                "backend": "numpy",
                "device": "cpu",
                "operation": "grouped_gemm",
                "status": "passed",
                "speedup_vs_loop": 0.4,
                "num_batched_gemm": 0,
                "num_grouped_tasks": 32,
                "num_shape_buckets": 1,
            },
            {
                "backend": "cupy",
                "device": "cuda:0",
                "operation": "grouped_gemm",
                "status": "passed",
                "speedup_vs_loop": 1.4,
                "num_batched_gemm": 1,
                "num_grouped_tasks": 32,
                "num_shape_buckets": 1,
            },
        ],
        min_speedup=1.01,
        require_backends=("cupy",),
    )

    assert summary["status"] == "passed"
    assert summary["checked_count"] == 1
    assert summary["failures"] == []


def test_gemm_smoke_cli_returns_nonzero_when_grouped_gate_fails(monkeypatch, capsys):
    from renormalizer.backend import gemm_smoke

    monkeypatch.setattr(
        gemm_smoke,
        "run_backend_smoke",
        lambda *args, **kwargs: [
            {
                "backend": "torch",
                "device": "gpu",
                "operation": "grouped_gemm",
                "status": "passed",
                "speedup_vs_loop": 0.5,
                "num_batched_gemm": 1,
                "num_grouped_tasks": 4,
                "num_shape_buckets": 1,
            }
        ],
    )

    rc = gemm_smoke.main([
        "--backends",
        "torch",
        "--device",
        "gpu",
        "--grouped-speedup-gate",
        "--min-grouped-speedup",
        "1.1",
        "--require-grouped-backends",
        "torch",
    ])

    assert rc == 2
    payloads = _json_lines(capsys.readouterr().out)
    gate = payloads[-1]
    assert gate["operation"] == "grouped_gemm_gate"
    assert gate["status"] == "failed"
    assert gate["failures"][0]["reason"] == "speedup below threshold"


def test_grouped_gemm_smoke_uses_native_backend_batching_policy():
    from renormalizer.backend.gemm_smoke import _grouped_gemm_smoke
    from renormalizer.backend.numpy_backend import NumpyBackend

    class NativeGroupedNumpyBackend(NumpyBackend):
        supports_grouped_gemm = True

    backend = NativeGroupedNumpyBackend()

    record = _grouped_gemm_smoke(
        backend,
        "numpy",
        "cpu",
        gpu_count=0,
        repeat=1,
        batch=8,
        small_dim=64,
        pack_threshold=4,
    )

    assert record["num_batched_gemm"] == 1
    assert record["num_gemm"] == 0
