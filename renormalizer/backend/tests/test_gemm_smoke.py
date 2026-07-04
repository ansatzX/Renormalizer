# -*- coding: utf-8 -*-

import json

import pytest


def _json_lines(text):
    return [json.loads(line) for line in text.splitlines() if line.strip().startswith("{")]


def _grouped_record(**overrides):
    record = {
        "backend": "torch",
        "device": "gpu",
        "operation": "grouped_gemm",
        "status": "passed",
        "speedup_vs_loop": 1.35,
        "num_batched_gemm": 1,
        "num_gemm": 0,
        "num_grouped_tasks": 64,
        "num_shape_buckets": 1,
        "supports_grouped_gemm": True,
        "grouped_gemm_policy": "bucketed_batched_matmul",
        "grouped_gemm_implementation": "backend_bucketed_batched_matmul",
        "requires_grouped_gemm_fallback": False,
        "fallback_from": None,
        "fallback_to": None,
        "fallback_reason": None,
        "shape_buckets": [
            {
                "dtype_a": "float64",
                "dtype_b": "float64",
                "batch_shape": [],
                "m": 32,
                "n": 32,
                "k": 32,
                "trans_a": False,
                "trans_b": False,
                "conj_a": False,
                "conj_b": False,
                "execution": "batched",
                "task_count": 64,
            }
        ],
        "flops": 1048576,
        "read_bytes": 65536,
        "write_bytes": 32768,
        "copy_bytes": 0,
        "wall_s": 0.1,
        "grouped_wall_s": 0.1,
        "loop_wall_s": 0.135,
    }
    record.update(overrides)
    if "fallback_from" not in overrides:
        record["fallback_from"] = (
            "grouped_gemm" if record.get("requires_grouped_gemm_fallback") else None
        )
    if "fallback_to" not in overrides:
        record["fallback_to"] = (
            record.get("grouped_gemm_policy") if record.get("requires_grouped_gemm_fallback") else None
        )
    if "bucket_task_counts" not in overrides:
        record["bucket_task_counts"] = [int(record.get("num_grouped_tasks") or 0)]
    if "shape_buckets" not in overrides:
        record["shape_buckets"] = [
            {
                **record["shape_buckets"][0],
                "task_count": int(record["bucket_task_counts"][0]),
            }
        ]
    if "speedup_vs_loop" in overrides and "loop_wall_s" not in overrides:
        record["loop_wall_s"] = record["grouped_wall_s"] * record["speedup_vs_loop"]
    return record


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
    matmul = next(record for record in records if record["operation"] == "matmul")
    batched = next(record for record in records if record["operation"] == "batched_matmul")

    assert matmul["execution_profile"]["event"] == "contraction_execute"
    assert matmul["execution_profile"]["lowering"] == "gemm"
    assert matmul["execution_profile"]["num_gemm"] == 1
    assert matmul["execution_profile"]["num_batched_gemm"] == 0
    assert matmul["execution_profile"]["fallback_reason"] is None
    assert matmul["execution_profile"]["flops"] > 0
    assert matmul["execution_profile"]["wall_s"] >= 0.0

    assert batched["execution_profile"]["event"] == "contraction_execute"
    assert batched["execution_profile"]["lowering"] in ("batched_gemm", "strided_batched_gemm", "fallback_tensordot", "fallback_einsum")
    assert batched["execution_profile"]["num_gemm"] == 0
    assert batched["execution_profile"]["flops"] > 0
    assert batched["execution_profile"]["wall_s"] >= 0.0

    assert grouped["execution_profile"]["event"] == "contraction_execute"
    assert grouped["execution_profile"]["lowering"] == "grouped_gemm"
    assert grouped["execution_profile"]["num_grouped_tasks"] == 4
    assert grouped["execution_profile"]["num_shape_buckets"] == grouped["num_shape_buckets"]
    assert grouped["execution_profile"]["bucket_task_counts"] == grouped["bucket_task_counts"]
    assert grouped["execution_profile"]["group_sizes"] == grouped["bucket_task_counts"]
    assert grouped["execution_profile"]["gsta"] == [0, 4]
    assert grouped["execution_profile"]["sorted_indices"] == [0, 1, 2, 3]
    assert grouped["execution_profile"]["group_keys"][0]["m"] == 4
    assert grouped["execution_profile"]["group_keys"][0]["n"] == 4
    assert grouped["execution_profile"]["group_keys"][0]["k"] == 4
    assert grouped["execution_profile"]["fallback_from"] == "grouped_gemm"
    assert grouped["execution_profile"]["fallback_to"] == grouped["fallback_to"]
    assert grouped["execution_profile"]["fallback_reason"] == grouped["fallback_reason"]
    assert grouped["execution_profile"]["grouped_gemm_policy"] == grouped["grouped_gemm_policy"]
    assert grouped["execution_profile"]["grouped_gemm_implementation"] == grouped["grouped_gemm_implementation"]
    assert grouped["execution_profile"]["flops"] == grouped["flops"]
    assert grouped["execution_profile"]["num_gemm"] == grouped["num_gemm"]
    assert grouped["execution_profile"]["wall_s"] >= 0.0

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
    assert grouped["shape_buckets"] == [
        {
            "dtype_a": "float64",
            "dtype_b": "float64",
            "batch_shape": [],
            "m": 4,
            "n": 4,
            "k": 4,
            "trans_a": False,
            "trans_b": False,
            "conj_a": False,
            "conj_b": False,
            "execution": "loop",
            "task_count": 4,
        }
    ]


def test_run_backend_smoke_uses_grouped_dim_independent_of_small_dim():
    from renormalizer.backend.gemm_smoke import run_backend_smoke

    records = run_backend_smoke(
        "numpy",
        device="cpu",
        batch=4,
        small_dim=4,
        medium_dim=8,
        grouped_dim=6,
        repeat=1,
        pack_threshold=2,
    )

    batched = next(record for record in records if record["operation"] == "batched_matmul")
    grouped = next(record for record in records if record["operation"] == "grouped_gemm")

    assert batched["shape_a"] == (4, 4, 4)
    assert batched["shape_b"] == (4, 4, 4)
    assert grouped["shape"] == (6, 6, 6)
    assert grouped["shape_buckets"] == [
        {
            "dtype_a": "float64",
            "dtype_b": "float64",
            "batch_shape": [],
            "m": 6,
            "n": 6,
            "k": 6,
            "trans_a": False,
            "trans_b": False,
            "conj_a": False,
            "conj_b": False,
            "execution": "loop",
            "task_count": 4,
        }
    ]


def test_run_backend_smoke_records_prepacked_grouped_gemm_profile():
    from renormalizer.backend.gemm_smoke import run_backend_smoke

    records = run_backend_smoke(
        "numpy",
        device="cpu",
        batch=4,
        small_dim=4,
        medium_dim=8,
        grouped_dim=96,
        repeat=1,
        pack_threshold=2,
    )

    grouped = next(record for record in records if record["operation"] == "grouped_gemm")
    assert grouped["prepacked_wall_s"] >= 0.0
    assert grouped["prepacked_speedup_vs_loop"] is not None
    assert grouped["prepacked_speedup_vs_grouped"] is not None
    assert grouped["prepack_profile"]["event"] == "grouped_gemm_prepack"
    assert grouped["prepack_profile"]["pack_strategy"] == "prepack_once"
    assert grouped["prepack_profile"]["prepacked"] is True
    assert grouped["prepacked_execution_profile"]["event"] == "grouped_gemm_execute"
    assert grouped["prepacked_execution_profile"]["pack_strategy"] == "prepacked_reuse"
    assert grouped["prepacked_execution_profile"]["prepacked"] is True
    assert grouped["prepacked_execution_profile"]["pack_s"] == 0.0
    assert grouped["prepacked_execution_profile"]["pack_bytes"] == 0
    assert grouped["prepacked_execution_profile"]["reused_pack_bytes"] > 0
    assert grouped["prepacked_max_abs_error"] <= 1e-9
    assert grouped["raw_prepacked_wall_s"] >= 0.0
    assert grouped["raw_prepacked_speedup_vs_loop"] is not None
    assert grouped["raw_prepacked_max_abs_error"] <= 1e-9
    assert grouped["raw_prepacked_speedup_vs_loop"] == pytest.approx(
        grouped["loop_wall_s"] / grouped["raw_prepacked_wall_s"]
    )


def test_run_backend_smoke_sweeps_multiple_grouped_dims_once_per_dim():
    from renormalizer.backend.gemm_smoke import run_backend_smoke

    records = run_backend_smoke(
        "numpy",
        device="cpu",
        batch=4,
        small_dim=4,
        medium_dim=8,
        grouped_dims=(5, 6),
        repeat=1,
        pack_threshold=2,
    )

    assert [record["operation"] for record in records].count("matmul") == 1
    assert [record["operation"] for record in records].count("batched_matmul") == 1
    grouped_records = [record for record in records if record["operation"] == "grouped_gemm"]

    assert [record["grouped_dim"] for record in grouped_records] == [5, 6]
    assert [record["shape"] for record in grouped_records] == [(5, 5, 5), (6, 6, 6)]
    assert [record["grouped_dim_index"] for record in grouped_records] == [0, 1]
    assert [record["grouped_dim_count"] for record in grouped_records] == [2, 2]


def test_run_backend_smoke_can_generate_block_heavy_grouped_gemm_tasks():
    from renormalizer.backend.gemm_smoke import run_backend_smoke

    records = run_backend_smoke(
        "numpy",
        device="cpu",
        batch=8,
        small_dim=4,
        medium_dim=8,
        grouped_dim=12,
        grouped_shape_mode="block_heavy",
        repeat=1,
        pack_threshold=2,
    )

    grouped = next(record for record in records if record["operation"] == "grouped_gemm")
    assert grouped["grouped_shape_mode"] == "block_heavy"
    assert grouped["num_grouped_tasks"] == 8
    assert grouped["num_shape_buckets"] >= 3
    assert sum(grouped["bucket_task_counts"]) == grouped["num_grouped_tasks"]
    assert len({tuple(shape) for shape in grouped["grouped_task_shapes"]}) >= 3
    assert len({(bucket["m"], bucket["n"], bucket["k"]) for bucket in grouped["shape_buckets"]}) >= 3
    assert grouped["prepacked_execution_profile"]["event"] == "grouped_gemm_execute"
    assert "pack_strategy" in grouped["prepacked_execution_profile"]
    assert grouped["prepacked_max_abs_error"] <= 1e-9


def test_run_backend_smoke_can_generate_block_tensor_grouped_gemm_tasks():
    from renormalizer.backend.gemm_smoke import run_backend_smoke

    records = run_backend_smoke(
        "numpy",
        device="cpu",
        batch=8,
        small_dim=4,
        medium_dim=8,
        grouped_dim=12,
        grouped_shape_mode="block_heavy",
        grouped_source="block_tensor",
        repeat=1,
        pack_threshold=2,
    )

    grouped = next(record for record in records if record["operation"] == "grouped_gemm")
    assert grouped["grouped_source"] == "block_tensor"
    assert grouped["grouped_shape_mode"] == "block_heavy"
    assert grouped["block_tensor_dense_materialized"] is False
    assert grouped["block_plan_profile"]["event"] == "contraction_plan"
    assert grouped["block_plan_profile"]["lowering"] == "block_grouped_gemm"
    assert grouped["block_plan_profile"]["dense_materialized"] is False
    assert grouped["execution_profile"]["event"] == "contraction_execute"
    assert grouped["execution_profile"]["lowering"] == "block_grouped_gemm"
    assert grouped["execution_profile"]["dense_materialized"] is False
    assert grouped["num_grouped_tasks"] == 8
    assert grouped["num_shape_buckets"] >= 3
    assert sum(grouped["bucket_task_counts"]) == grouped["num_grouped_tasks"]
    assert len(grouped["result_block_shapes"]) == grouped["num_grouped_tasks"]
    assert len({tuple(shape) for shape in grouped["grouped_task_shapes"]}) >= 3
    assert grouped["prepacked_execution_profile"]["event"] == "grouped_gemm_execute"
    assert grouped["prepacked_max_abs_error"] <= 1e-9
    assert grouped["max_abs_error"] <= 1e-9


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
        _grouped_record(backend="torch", speedup_vs_loop=1.35),
        _grouped_record(
            backend="cupy",
            speedup_vs_loop=0.92,
        ),
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
            "speedup_metric": "speedup_vs_loop",
            "selected_speedup_vs_loop": 0.92,
            "min_speedup": 1.05,
        }
    ]


def test_grouped_gemm_phase5_gate_prefers_prepacked_reuse_speedup():
    from renormalizer.backend.gemm_smoke import evaluate_grouped_gemm_gate

    summary = evaluate_grouped_gemm_gate(
        [
            _grouped_record(
                backend="torch",
                speedup_vs_loop=0.9,
                grouped_wall_s=1.0,
                wall_s=1.0,
                loop_wall_s=0.9,
                prepacked_wall_s=0.45,
                prepacked_speedup_vs_loop=2.0,
                prepacked_max_abs_error=0.0,
            )
        ],
        min_speedup=1.1,
        require_backends=("torch",),
    )

    assert summary["status"] == "passed"
    assert summary["speedup_metrics"] == ["prepacked_speedup_vs_loop"]
    assert summary["failures"] == []


def test_grouped_gemm_phase5_gate_prefers_raw_prepacked_speedup():
    from renormalizer.backend.gemm_smoke import evaluate_grouped_gemm_gate

    summary = evaluate_grouped_gemm_gate(
        [
            _grouped_record(
                backend="cupy",
                speedup_vs_loop=0.2,
                grouped_wall_s=1.0,
                wall_s=1.0,
                loop_wall_s=0.2,
                prepacked_wall_s=0.5,
                prepacked_speedup_vs_loop=0.4,
                prepacked_max_abs_error=0.0,
                raw_prepacked_wall_s=0.1,
                raw_prepacked_speedup_vs_loop=2.0,
                raw_prepacked_max_abs_error=0.0,
            )
        ],
        min_speedup=1.5,
        require_backends=("cupy",),
    )

    assert summary["status"] == "passed"
    assert summary["speedup_metrics"] == ["raw_prepacked_speedup_vs_loop"]
    assert summary["failures"] == []


def test_grouped_gemm_phase5_gate_failure_reports_selected_speedup_metric():
    from renormalizer.backend.gemm_smoke import evaluate_grouped_gemm_gate

    summary = evaluate_grouped_gemm_gate(
        [
            _grouped_record(
                backend="cupy",
                speedup_vs_loop=2.5,
                grouped_wall_s=1.0,
                wall_s=1.0,
                loop_wall_s=2.5,
                prepacked_wall_s=2.0,
                prepacked_speedup_vs_loop=1.25,
                prepacked_max_abs_error=0.0,
            )
        ],
        min_speedup=1.5,
        require_backends=("cupy",),
    )

    assert summary["status"] == "failed"
    assert summary["failures"] == [
        {
            "backend": "cupy",
            "device": "gpu",
            "reason": "speedup below threshold",
            "speedup_vs_loop": 2.5,
            "speedup_metric": "prepacked_speedup_vs_loop",
            "selected_speedup_vs_loop": 1.25,
            "min_speedup": 1.5,
        }
    ]


def test_grouped_gemm_gate_rejects_inconsistent_prepacked_timing():
    from renormalizer.backend.gemm_smoke import evaluate_grouped_gemm_gate

    summary = evaluate_grouped_gemm_gate(
        [
            _grouped_record(
                backend="torch",
                speedup_vs_loop=1.2,
                grouped_wall_s=1.0,
                wall_s=1.0,
                loop_wall_s=1.2,
                prepacked_wall_s=0.5,
                prepacked_speedup_vs_loop=1.0,
                prepacked_max_abs_error=0.0,
            )
        ],
        min_speedup=0.5,
        require_backends=("torch",),
    )

    assert summary["status"] == "failed"
    assert summary["failures"] == [
        {
            "backend": "torch",
            "device": "gpu",
            "reason": "prepacked speedup does not match timing telemetry",
            "speedup_vs_loop": 1.2,
            "expected_prepacked_speedup_vs_loop": 2.4,
        }
    ]


def test_grouped_gemm_gate_rejects_wrong_prepacked_result_even_when_fast():
    from renormalizer.backend.gemm_smoke import evaluate_grouped_gemm_gate

    summary = evaluate_grouped_gemm_gate(
        [
            _grouped_record(
                backend="cupy",
                speedup_vs_loop=0.5,
                grouped_wall_s=1.0,
                wall_s=1.0,
                loop_wall_s=0.5,
                prepacked_wall_s=0.1,
                prepacked_speedup_vs_loop=5.0,
                prepacked_max_abs_error=1e-3,
            )
        ],
        min_speedup=1.5,
        require_backends=("cupy",),
    )

    assert summary["status"] == "failed"
    assert summary["failures"] == [
        {
            "backend": "cupy",
            "device": "gpu",
            "reason": "prepacked grouped_gemm correctness error exceeds tolerance",
            "speedup_vs_loop": 0.5,
            "prepacked_max_abs_error": 1e-3,
            "max_abs_error_tolerance": 1e-9,
        }
    ]


def test_grouped_gemm_gate_requires_execution_telemetry_fields():
    from renormalizer.backend.gemm_smoke import evaluate_grouped_gemm_gate

    incomplete = _grouped_record()
    del incomplete["flops"]
    del incomplete["grouped_gemm_policy"]
    del incomplete["fallback_from"]
    del incomplete["fallback_to"]

    summary = evaluate_grouped_gemm_gate(
        [incomplete],
        min_speedup=1.0,
        require_backends=("torch",),
    )

    assert summary["status"] == "failed"
    assert summary["failures"] == [
        {
            "backend": "torch",
            "device": "gpu",
            "reason": "missing grouped_gemm telemetry",
            "missing_fields": ["flops", "grouped_gemm_policy", "fallback_from", "fallback_to"],
            "speedup_vs_loop": 1.35,
        }
    ]


def test_grouped_gemm_gate_rejects_inconsistent_timing_telemetry():
    from renormalizer.backend.gemm_smoke import evaluate_grouped_gemm_gate

    summary = evaluate_grouped_gemm_gate(
        [
            _grouped_record(
                backend="torch",
                device="gpu",
                grouped_wall_s=0.25,
                wall_s=0.25,
                loop_wall_s=0.5,
                speedup_vs_loop=1.1,
            ),
            _grouped_record(
                backend="cupy",
                device="gpu",
                grouped_wall_s=0.25,
                wall_s=0.2,
                loop_wall_s=0.5,
                speedup_vs_loop=2.0,
            ),
        ],
        min_speedup=1.0,
        require_backends=("torch", "cupy"),
    )

    assert summary["status"] == "failed"
    assert summary["failures"] == [
        {
            "backend": "torch",
            "device": "gpu",
            "reason": "speedup does not match timing telemetry",
            "speedup_vs_loop": 1.1,
            "expected_speedup_vs_loop": 2.0,
        },
        {
            "backend": "cupy",
            "device": "gpu",
            "reason": "grouped_gemm wall time fields disagree",
            "speedup_vs_loop": 2.0,
            "wall_s": 0.2,
            "grouped_wall_s": 0.25,
        },
    ]


def test_grouped_gemm_gate_rejects_inconsistent_fallback_metadata():
    from renormalizer.backend.gemm_smoke import evaluate_grouped_gemm_gate

    summary = evaluate_grouped_gemm_gate(
        [
            _grouped_record(
                backend="torch",
                grouped_gemm_implementation="fallback_bucketed_batched_matmul",
                requires_grouped_gemm_fallback=False,
                fallback_reason=None,
            ),
            _grouped_record(
                backend="cupy",
                grouped_gemm_implementation="fallback_bucketed_batched_matmul",
                requires_grouped_gemm_fallback=True,
                fallback_reason=None,
            ),
            _grouped_record(
                backend="jax",
                grouped_gemm_implementation="backend_bucketed_batched_matmul",
                requires_grouped_gemm_fallback=False,
                fallback_reason="backend-owned grouped_gemm unavailable; used bucketed fallback",
            ),
        ],
        min_speedup=1.0,
        require_backends=("torch", "cupy", "jax"),
        require_backend_owned=False,
    )

    assert summary["status"] == "failed"
    assert summary["failures"] == [
        {
            "backend": "torch",
            "device": "gpu",
            "reason": "fallback flag does not match grouped_gemm implementation",
            "speedup_vs_loop": 1.35,
            "grouped_gemm_implementation": "fallback_bucketed_batched_matmul",
            "requires_grouped_gemm_fallback": False,
        },
        {
            "backend": "cupy",
            "device": "gpu",
            "reason": "missing grouped_gemm fallback reason",
            "speedup_vs_loop": 1.35,
        },
        {
            "backend": "jax",
            "device": "gpu",
            "reason": "unexpected grouped_gemm fallback reason",
            "speedup_vs_loop": 1.35,
            "fallback_reason": "backend-owned grouped_gemm unavailable; used bucketed fallback",
        },
    ]


def test_grouped_gemm_gate_rejects_inconsistent_fallback_route():
    from renormalizer.backend.gemm_smoke import evaluate_grouped_gemm_gate

    summary = evaluate_grouped_gemm_gate(
        [
            _grouped_record(
                backend="numpy",
                grouped_gemm_implementation="fallback_bucketed_batched_matmul",
                requires_grouped_gemm_fallback=True,
                fallback_from=None,
                fallback_to="bucketed_batched_matmul",
                fallback_reason="backend-owned grouped_gemm unavailable; used bucketed fallback",
            ),
            _grouped_record(
                backend="torch",
                grouped_gemm_implementation="backend_bucketed_batched_matmul",
                requires_grouped_gemm_fallback=False,
                fallback_from="grouped_gemm",
                fallback_to="bucketed_batched_matmul",
                fallback_reason=None,
            ),
        ],
        min_speedup=1.0,
        require_backends=("numpy", "torch"),
        require_backend_owned=False,
    )

    assert summary["status"] == "failed"
    assert summary["failures"] == [
        {
            "backend": "numpy",
            "device": "gpu",
            "reason": "missing grouped_gemm fallback route",
            "speedup_vs_loop": 1.35,
            "fallback_from": None,
            "fallback_to": "bucketed_batched_matmul",
        },
        {
            "backend": "torch",
            "device": "gpu",
            "reason": "unexpected grouped_gemm fallback route",
            "speedup_vs_loop": 1.35,
            "fallback_from": "grouped_gemm",
            "fallback_to": "bucketed_batched_matmul",
        },
    ]


def test_grouped_gemm_gate_rejects_inconsistent_bucket_telemetry():
    from renormalizer.backend.gemm_smoke import evaluate_grouped_gemm_gate

    summary = evaluate_grouped_gemm_gate(
        [
            _grouped_record(
                backend="torch",
                num_shape_buckets=2,
                num_grouped_tasks=64,
                bucket_task_counts=[64],
            ),
            _grouped_record(
                backend="cupy",
                num_shape_buckets=2,
                num_grouped_tasks=64,
                bucket_task_counts=[32, 16],
            ),
        ],
        min_speedup=1.0,
        require_backends=("torch", "cupy"),
    )

    assert summary["status"] == "failed"
    assert summary["failures"] == [
        {
            "backend": "torch",
            "device": "gpu",
            "reason": "bucket task counts do not match shape bucket count",
            "speedup_vs_loop": 1.35,
            "num_shape_buckets": 2,
            "bucket_task_counts": [64],
        },
        {
            "backend": "cupy",
            "device": "gpu",
            "reason": "bucket task counts do not match grouped task total",
            "speedup_vs_loop": 1.35,
            "num_grouped_tasks": 64,
            "bucket_task_counts": [32, 16],
        },
    ]


def test_grouped_gemm_gate_rejects_inconsistent_shape_bucket_telemetry():
    from renormalizer.backend.gemm_smoke import evaluate_grouped_gemm_gate

    summary = evaluate_grouped_gemm_gate(
        [
            _grouped_record(
                num_shape_buckets=2,
                bucket_task_counts=[32, 32],
                num_grouped_tasks=64,
                shape_buckets=[
                    {
                        "dtype_a": "float64",
                        "dtype_b": "float64",
                        "batch_shape": [],
                        "m": 32,
                        "n": 32,
                        "k": 32,
                        "trans_a": False,
                        "trans_b": False,
                        "conj_a": False,
                        "conj_b": False,
                        "execution": "batched",
                        "task_count": 64,
                    }
                ],
            )
        ],
        min_speedup=1.0,
        require_backends=("torch",),
    )

    assert summary["status"] == "failed"
    assert summary["failures"] == [
        {
            "backend": "torch",
            "device": "gpu",
            "reason": "shape bucket profiles do not match shape bucket count",
            "speedup_vs_loop": 1.35,
            "num_shape_buckets": 2,
            "shape_bucket_count": 1,
        }
    ]


def test_grouped_gemm_benchmark_gate_reports_missing_backend():
    from renormalizer.backend.gemm_smoke import evaluate_grouped_gemm_gate

    summary = evaluate_grouped_gemm_gate(
        [
            _grouped_record(backend="torch", speedup_vs_loop=1.2, num_grouped_tasks=32)
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
            _grouped_record(
                backend="numpy",
                device="cpu",
                speedup_vs_loop=0.4,
                num_batched_gemm=0,
                num_grouped_tasks=32,
                supports_grouped_gemm=False,
                grouped_gemm_implementation="fallback_bucketed_loop_matmul",
                requires_grouped_gemm_fallback=True,
                fallback_reason="backend-owned grouped_gemm unavailable; used bucketed fallback",
            ),
            _grouped_record(
                backend="cupy",
                device="cuda:0",
                speedup_vs_loop=1.4,
                num_grouped_tasks=32,
            ),
        ],
        min_speedup=1.01,
        require_backends=("cupy",),
    )

    assert summary["status"] == "passed"
    assert summary["checked_count"] == 1
    assert summary["failures"] == []


def test_grouped_gemm_phase5_gate_can_require_block_heavy_shape_mode():
    from renormalizer.backend.gemm_smoke import evaluate_grouped_gemm_gate

    summary = evaluate_grouped_gemm_gate(
        [
            _grouped_record(
                backend="cupy",
                device="cuda:0",
                grouped_shape_mode="same",
                speedup_vs_loop=1.5,
            )
        ],
        min_speedup=1.0,
        require_backends=("cupy",),
        require_grouped_shape_mode="block_heavy",
    )

    assert summary["status"] == "failed"
    assert summary["required_grouped_shape_mode"] == "block_heavy"
    assert summary["failures"] == [
        {
            "backend": "cupy",
            "device": "cuda:0",
            "reason": "missing grouped_gemm shape mode record",
            "grouped_shape_mode": "block_heavy",
        }
    ]


def test_grouped_gemm_phase5_gate_can_require_block_tensor_source():
    from renormalizer.backend.gemm_smoke import evaluate_grouped_gemm_gate

    summary = evaluate_grouped_gemm_gate(
        [
            _grouped_record(
                backend="cupy",
                device="cuda:0",
                grouped_source="descriptors",
                speedup_vs_loop=1.5,
            )
        ],
        min_speedup=1.0,
        require_backends=("cupy",),
        require_grouped_source="block_tensor",
    )

    assert summary["status"] == "failed"
    assert summary["required_grouped_source"] == "block_tensor"
    assert summary["failures"] == [
        {
            "backend": "cupy",
            "device": "cuda:0",
            "reason": "missing grouped_gemm source record",
            "grouped_source": "block_tensor",
        }
    ]


def test_gemm_smoke_cli_returns_nonzero_when_grouped_gate_fails(monkeypatch, capsys):
    from renormalizer.backend import gemm_smoke

    monkeypatch.setattr(
        gemm_smoke,
        "run_backend_smoke",
        lambda *args, **kwargs: [
            _grouped_record(speedup_vs_loop=0.5, num_grouped_tasks=4)
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


def test_grouped_gemm_phase5_gate_rejects_external_fallback_even_when_fast():
    from renormalizer.backend.gemm_smoke import evaluate_grouped_gemm_gate

    summary = evaluate_grouped_gemm_gate(
        [
            _grouped_record(
                speedup_vs_loop=3.0,
                num_grouped_tasks=32,
                supports_grouped_gemm=False,
                grouped_gemm_implementation="fallback_bucketed_batched_matmul",
                requires_grouped_gemm_fallback=True,
                fallback_reason="backend-owned grouped_gemm unavailable; used bucketed fallback",
            )
        ],
        min_speedup=1.0,
        require_backends=("torch",),
        require_native=True,
    )

    assert summary["status"] == "failed"
    assert summary["require_native"] is True
    assert summary["require_backend_owned"] is True
    assert summary["failures"] == [
        {
            "backend": "torch",
            "device": "gpu",
            "reason": "backend-owned grouped_gemm unavailable",
            "speedup_vs_loop": 3.0,
        }
    ]


def test_grouped_gemm_phase5_gate_requires_each_gpu_device_record():
    from renormalizer.backend.gemm_smoke import evaluate_grouped_gemm_gate

    summary = evaluate_grouped_gemm_gate(
        [
            _grouped_record(device="cuda:0", speedup_vs_loop=1.4, num_grouped_tasks=32)
        ],
        min_speedup=1.0,
        require_backends=("torch",),
        require_gpu_count=2,
    )

    assert summary["status"] == "failed"
    assert summary["required_gpu_count"] == 2
    assert summary["failures"] == [
        {
            "backend": "torch",
            "device": "cuda:1",
            "reason": "missing grouped_gemm record",
        }
    ]


def test_grouped_gemm_phase5_gate_requires_each_grouped_dim_per_gpu():
    from renormalizer.backend.gemm_smoke import evaluate_grouped_gemm_gate

    summary = evaluate_grouped_gemm_gate(
        [
            _grouped_record(device="cuda:0", grouped_dim=16, speedup_vs_loop=1.4),
            _grouped_record(device="cuda:1", grouped_dim=16, speedup_vs_loop=1.4),
            _grouped_record(device="cuda:1", grouped_dim=32, speedup_vs_loop=1.4),
        ],
        min_speedup=1.0,
        require_backends=("torch",),
        require_gpu_count=2,
        require_grouped_dims=(16, 32),
    )

    assert summary["status"] == "failed"
    assert summary["required_grouped_dims"] == [16, 32]
    assert summary["failures"] == [
        {
            "backend": "torch",
            "device": "cuda:0",
            "grouped_dim": 32,
            "reason": "missing grouped_gemm dimension record",
        }
    ]


def test_gemm_smoke_cli_all_gpus_expands_device_matrix_and_gates_coverage(monkeypatch, capsys):
    from renormalizer.backend import gemm_smoke

    calls = []

    monkeypatch.setattr(gemm_smoke, "detect_gpu_count", lambda: 2)

    def fake_run_backend_smoke(backend_name, **kwargs):
        calls.append((backend_name, kwargs["device"]))
        return [
            _grouped_record(backend=backend_name, device=kwargs["device"], speedup_vs_loop=1.5, num_grouped_tasks=kwargs["batch"])
        ]

    monkeypatch.setattr(gemm_smoke, "run_backend_smoke", fake_run_backend_smoke)

    rc = gemm_smoke.main([
        "--backends",
        "torch",
        "--device",
        "gpu",
        "--all-gpus",
        "--grouped-speedup-gate",
        "--require-grouped-backends",
        "torch",
        "--require-gpu-count",
        "2",
    ])

    assert rc == 0
    assert calls == [("torch", "cuda:0"), ("torch", "cuda:1")]
    payloads = _json_lines(capsys.readouterr().out)
    gate = payloads[-1]
    assert gate["operation"] == "grouped_gemm_gate"
    assert gate["status"] == "passed"
    assert gate["required_gpu_count"] == 2
    assert gate["required_gpu_indices"] == [0, 1]


def test_gemm_smoke_cli_runs_nvidia_dmon_around_smoke(monkeypatch, tmp_path, capsys):
    from renormalizer.backend import gemm_smoke

    calls = []
    process = None

    class FakeProcess:
        def __init__(self, stdout):
            self.stdout = stdout

        def terminate(self):
            calls.append(("terminate",))

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            return 0

    def fake_popen(argv, stdout=None, stderr=None, text=None):
        nonlocal process
        process = FakeProcess(stdout)
        calls.append(("popen", argv, stdout.name, stderr, text))
        return process

    def fake_run_backend_smoke(backend_name, **kwargs):
        calls.append(("run", backend_name, kwargs["device"]))
        assert process is not None
        assert process.stdout.closed is False
        return [_grouped_record(backend=backend_name, device=kwargs["device"])]

    dmon_path = tmp_path / "nvdmon.log"
    monkeypatch.setattr(gemm_smoke.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gemm_smoke, "run_backend_smoke", fake_run_backend_smoke)

    rc = gemm_smoke.main([
        "--backends",
        "torch",
        "--device",
        "gpu",
        "--nvidia-dmon-output",
        str(dmon_path),
    ])

    assert rc == 0
    assert calls == [
        (
            "popen",
            ["nvidia-smi", "dmon", "-s", "pucvmte", "-o", "T"],
            str(dmon_path),
            gemm_smoke.subprocess.DEVNULL,
            True,
        ),
        ("run", "torch", "gpu"),
        ("terminate",),
        ("wait", 5.0),
    ]
    assert process.stdout.closed is True
    payloads = _json_lines(capsys.readouterr().out)
    assert payloads[0]["operation"] == "grouped_gemm"


def test_gemm_smoke_cli_skips_nvidia_dmon_without_output(monkeypatch, capsys):
    from renormalizer.backend import gemm_smoke

    monkeypatch.setattr(
        gemm_smoke.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("nvidia dmon should not start"),
    )
    monkeypatch.setattr(
        gemm_smoke,
        "run_backend_smoke",
        lambda backend_name, **kwargs: [
            _grouped_record(backend=backend_name, device=kwargs["device"])
        ],
    )

    rc = gemm_smoke.main(["--backends", "torch", "--device", "gpu"])

    assert rc == 0
    payloads = _json_lines(capsys.readouterr().out)
    assert payloads[0]["operation"] == "grouped_gemm"


def test_gemm_smoke_cli_passes_required_grouped_dims_to_gate(monkeypatch, capsys):
    from renormalizer.backend import gemm_smoke

    def fake_run_backend_smoke(backend_name, **kwargs):
        return [
            _grouped_record(backend=backend_name, device=kwargs["device"], grouped_dim=16, speedup_vs_loop=1.5),
            _grouped_record(backend=backend_name, device=kwargs["device"], grouped_dim=32, speedup_vs_loop=1.5),
        ]

    monkeypatch.setattr(gemm_smoke, "run_backend_smoke", fake_run_backend_smoke)

    rc = gemm_smoke.main([
        "--backends",
        "torch",
        "--device",
        "cpu",
        "--grouped-speedup-gate",
        "--require-grouped-backends",
        "torch",
        "--require-grouped-dims",
        "16,32",
    ])

    assert rc == 0
    payloads = _json_lines(capsys.readouterr().out)
    gate = payloads[-1]
    assert gate["operation"] == "grouped_gemm_gate"
    assert gate["status"] == "passed"
    assert gate["required_grouped_dims"] == [16, 32]


def test_gemm_smoke_cli_passes_required_grouped_shape_mode_to_gate(monkeypatch, capsys):
    from renormalizer.backend import gemm_smoke

    monkeypatch.setattr(
        gemm_smoke,
        "run_backend_smoke",
        lambda backend_name, **kwargs: [
            _grouped_record(
                backend=backend_name,
                device=kwargs["device"],
                grouped_shape_mode="same",
                speedup_vs_loop=1.5,
            )
        ],
    )

    rc = gemm_smoke.main([
        "--backends",
        "torch",
        "--device",
        "cpu",
        "--grouped-speedup-gate",
        "--require-grouped-backends",
        "torch",
        "--require-grouped-shape-mode",
        "block_heavy",
    ])

    assert rc == 2
    payloads = _json_lines(capsys.readouterr().out)
    gate = payloads[-1]
    assert gate["operation"] == "grouped_gemm_gate"
    assert gate["required_grouped_shape_mode"] == "block_heavy"
    assert gate["failures"] == [
        {
            "backend": "torch",
            "device": "cpu",
            "reason": "missing grouped_gemm shape mode record",
            "grouped_shape_mode": "block_heavy",
        }
    ]


def test_gemm_smoke_cli_passes_required_grouped_source_to_gate(monkeypatch, capsys):
    from renormalizer.backend import gemm_smoke

    monkeypatch.setattr(
        gemm_smoke,
        "run_backend_smoke",
        lambda backend_name, **kwargs: [
            _grouped_record(
                backend=backend_name,
                device=kwargs["device"],
                grouped_source="descriptors",
                speedup_vs_loop=1.5,
            )
        ],
    )

    rc = gemm_smoke.main([
        "--backends",
        "torch",
        "--device",
        "cpu",
        "--grouped-speedup-gate",
        "--require-grouped-backends",
        "torch",
        "--require-grouped-source",
        "block_tensor",
    ])

    assert rc == 2
    payloads = _json_lines(capsys.readouterr().out)
    gate = payloads[-1]
    assert gate["operation"] == "grouped_gemm_gate"
    assert gate["required_grouped_source"] == "block_tensor"
    assert gate["failures"] == [
        {
            "backend": "torch",
            "device": "cpu",
            "reason": "missing grouped_gemm source record",
            "grouped_source": "block_tensor",
        }
    ]


def test_gemm_smoke_cli_passes_grouped_dim_to_backend_smoke(monkeypatch, capsys):
    from renormalizer.backend import gemm_smoke

    calls = []

    def fake_run_backend_smoke(backend_name, **kwargs):
        calls.append((backend_name, kwargs))
        return [
            _grouped_record(backend=backend_name, device=kwargs["device"], speedup_vs_loop=1.5)
        ]

    monkeypatch.setattr(gemm_smoke, "run_backend_smoke", fake_run_backend_smoke)

    rc = gemm_smoke.main([
        "--backends",
        "torch",
        "--device",
        "cpu",
        "--small-dim",
        "4",
        "--grouped-dim",
        "64",
    ])

    assert rc == 0
    assert len(calls) == 1
    assert calls[0][0] == "torch"
    assert calls[0][1]["small_dim"] == 4
    assert calls[0][1]["grouped_dim"] == 64
    payloads = _json_lines(capsys.readouterr().out)
    assert payloads[0]["operation"] == "grouped_gemm"


def test_gemm_smoke_cli_passes_grouped_shape_mode_to_backend_smoke(monkeypatch, capsys):
    from renormalizer.backend import gemm_smoke

    calls = []

    def fake_run_backend_smoke(backend_name, **kwargs):
        calls.append((backend_name, kwargs))
        return [
            _grouped_record(
                backend=backend_name,
                device=kwargs["device"],
                speedup_vs_loop=1.5,
                grouped_shape_mode=kwargs["grouped_shape_mode"],
            )
        ]

    monkeypatch.setattr(gemm_smoke, "run_backend_smoke", fake_run_backend_smoke)

    rc = gemm_smoke.main([
        "--backends",
        "torch",
        "--device",
        "cpu",
        "--grouped-shape-mode",
        "block_heavy",
    ])

    assert rc == 0
    assert len(calls) == 1
    assert calls[0][1]["grouped_shape_mode"] == "block_heavy"
    payloads = _json_lines(capsys.readouterr().out)
    assert payloads[0]["grouped_shape_mode"] == "block_heavy"


def test_gemm_smoke_cli_passes_grouped_source_to_backend_smoke(monkeypatch, capsys):
    from renormalizer.backend import gemm_smoke

    calls = []

    def fake_run_backend_smoke(backend_name, **kwargs):
        calls.append((backend_name, kwargs))
        return [
            _grouped_record(
                backend=backend_name,
                device=kwargs["device"],
                speedup_vs_loop=1.5,
                grouped_source=kwargs["grouped_source"],
            )
        ]

    monkeypatch.setattr(gemm_smoke, "run_backend_smoke", fake_run_backend_smoke)

    rc = gemm_smoke.main([
        "--backends",
        "torch",
        "--device",
        "cpu",
        "--grouped-source",
        "block_tensor",
    ])

    assert rc == 0
    assert len(calls) == 1
    assert calls[0][1]["grouped_source"] == "block_tensor"
    payloads = _json_lines(capsys.readouterr().out)
    assert payloads[0]["grouped_source"] == "block_tensor"


def test_gemm_smoke_cli_passes_grouped_dims_sweep_to_backend_smoke(monkeypatch, capsys):
    from renormalizer.backend import gemm_smoke

    calls = []

    def fake_run_backend_smoke(backend_name, **kwargs):
        calls.append((backend_name, kwargs))
        return [
            _grouped_record(backend=backend_name, device=kwargs["device"], speedup_vs_loop=1.5)
        ]

    monkeypatch.setattr(gemm_smoke, "run_backend_smoke", fake_run_backend_smoke)

    rc = gemm_smoke.main([
        "--backends",
        "torch",
        "--device",
        "cpu",
        "--grouped-dims",
        "16,32,64",
    ])

    assert rc == 0
    assert len(calls) == 1
    assert calls[0][1]["grouped_dims"] == (16, 32, 64)
    payloads = _json_lines(capsys.readouterr().out)
    assert payloads[0]["operation"] == "grouped_gemm"


def test_gemm_smoke_cli_rejects_grouped_dim_and_grouped_dims_together():
    from renormalizer.backend import gemm_smoke

    with pytest.raises(SystemExit) as exc_info:
        gemm_smoke.main([
            "--backends",
            "torch",
            "--grouped-dim",
            "16",
            "--grouped-dims",
            "16,32",
        ])

    assert exc_info.value.code == 2


def test_grouped_gemm_smoke_uses_backend_owned_batching_policy():
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
    assert record["supports_grouped_gemm"] is True
    assert record["grouped_gemm_policy"] == "bucketed_batched_matmul"
    assert record["grouped_gemm_implementation"] == "backend_bucketed_batched_matmul"
    assert record["requires_grouped_gemm_fallback"] is False
    assert record["fallback_reason"] is None


def test_grouped_gemm_smoke_execution_profile_uses_pack_threshold():
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
        batch=2,
        small_dim=64,
        pack_threshold=2,
    )

    assert record["num_batched_gemm"] == 1
    assert record["num_gemm"] == 0
    assert record["execution_profile"]["num_batched_gemm"] == 1
    assert record["execution_profile"]["num_gemm"] == 0
    assert record["execution_profile"]["fallback_reason"] is None


def test_grouped_gemm_smoke_records_bucketed_fallback_policy():
    from renormalizer.backend.gemm_smoke import _grouped_gemm_smoke
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()

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

    assert record["supports_grouped_gemm"] is False
    assert record["grouped_gemm_policy"] in ("bucketed_batched_matmul", "bucketed_loop_matmul")
    assert record["grouped_gemm_implementation"] in (
        "fallback_bucketed_batched_matmul",
        "fallback_bucketed_loop_matmul",
    )
    assert record["requires_grouped_gemm_fallback"] is True
    assert record["fallback_reason"] == "backend-owned grouped_gemm unavailable; used bucketed fallback"
