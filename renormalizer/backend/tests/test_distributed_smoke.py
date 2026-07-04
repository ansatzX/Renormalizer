# -*- coding: utf-8 -*-

import json

import pytest


def _passing_planner_cost_comparison():
    return {
        "distributed_estimated_time_s": 0.4,
        "slicing_estimated_time_s": 1.0,
        "estimated_speedup_vs_slicing": 2.5,
        "distributed_estimated_write_bytes": 512,
        "slicing_estimated_write_bytes": 1024,
        "distributed_estimated_comm_bytes": 256,
        "slicing_estimated_comm_bytes": 0,
    }


def test_distributed_speed_gate_requires_speedup_over_slicing():
    from renormalizer.backend.distributed_smoke import evaluate_distributed_speed_gate

    records = [
        {
            "operation": "row_sharded_speed_benchmark",
            "status": "passed",
            "rank": 0,
            "world_size": 8,
            "speedup_vs_slicing": 0.95,
            "distributed_wall_s": 0.19,
            "slicing_wall_s": 0.18,
            "max_abs_error": 0.0,
            **_passing_planner_cost_comparison(),
        }
    ]

    summary = evaluate_distributed_speed_gate(records, min_speedup=1.01, require_world_size=8)

    assert summary["operation"] == "distributed_speed_gate"
    assert summary["status"] == "failed"
    assert summary["checked_count"] == 1
    assert summary["failures"] == [
        {
            "operation": "row_sharded_speed_benchmark",
            "rank": 0,
            "reason": "speedup below threshold",
            "speedup_vs_slicing": 0.95,
        }
    ]


def test_distributed_speed_gate_can_require_real_runtime():
    from renormalizer.backend.distributed_smoke import evaluate_distributed_speed_gate

    records = [
        {
            "operation": "row_sharded_speed_benchmark",
            "status": "passed",
            "rank": 0,
            "world_size": 8,
            "is_distributed_runtime": False,
            "speedup_vs_slicing": 4.0,
            "distributed_wall_s": 0.25,
            "slicing_wall_s": 1.0,
            "max_abs_error": 0.0,
            **_passing_planner_cost_comparison(),
        }
    ]

    summary = evaluate_distributed_speed_gate(
        records,
        min_speedup=1.01,
        require_world_size=8,
        require_real_runtime=True,
    )

    assert summary["operation"] == "distributed_speed_gate"
    assert summary["status"] == "failed"
    assert summary["require_real_runtime"] is True
    assert summary["failures"] == [
        {
            "operation": "row_sharded_speed_benchmark",
            "rank": 0,
            "reason": "distributed runtime is simulated",
        }
    ]


def test_distributed_speed_gate_requires_planner_cost_comparison():
    from renormalizer.backend.distributed_smoke import evaluate_distributed_speed_gate

    records = [
        {
            "operation": "row_sharded_speed_benchmark",
            "status": "passed",
            "rank": 0,
            "world_size": 8,
            "is_distributed_runtime": True,
            "speedup_vs_slicing": 2.0,
            "distributed_wall_s": 0.5,
            "slicing_wall_s": 1.0,
            "max_abs_error": 0.0,
        }
    ]

    summary = evaluate_distributed_speed_gate(records, min_speedup=1.01, require_world_size=8)

    assert summary["operation"] == "distributed_speed_gate"
    assert summary["status"] == "failed"
    assert summary["failures"] == [
        {
            "operation": "row_sharded_speed_benchmark",
            "rank": 0,
            "reason": "missing planner cost comparison",
        }
    ]


def test_distributed_speed_gate_accepts_large_absolute_error_when_relative_error_is_small():
    from renormalizer.backend.distributed_smoke import evaluate_distributed_speed_gate

    records = [
        {
            "operation": "row_sharded_speed_benchmark",
            "status": "passed",
            "rank": 0,
            "world_size": 8,
            "is_distributed_runtime": True,
            "speedup_vs_slicing": 4.0,
            "distributed_wall_s": 0.25,
            "slicing_wall_s": 1.0,
            "max_abs_error": 1.0e-2,
            "max_relative_error": 1.0e-13,
            **_passing_planner_cost_comparison(),
        }
    ]

    summary = evaluate_distributed_speed_gate(
        records,
        min_speedup=1.01,
        require_world_size=8,
        max_abs_error=1.0e-9,
        max_relative_error=1.0e-12,
        require_real_runtime=True,
    )

    assert summary["operation"] == "distributed_speed_gate"
    assert summary["status"] == "passed"
    assert summary["max_abs_error"] == 1.0e-9
    assert summary["max_relative_error"] == 1.0e-12
    assert summary["failures"] == []


def test_distributed_speed_gate_rejects_when_absolute_and_relative_error_are_both_too_large():
    from renormalizer.backend.distributed_smoke import evaluate_distributed_speed_gate

    records = [
        {
            "operation": "row_sharded_speed_benchmark",
            "status": "passed",
            "rank": 0,
            "world_size": 8,
            "is_distributed_runtime": True,
            "speedup_vs_slicing": 4.0,
            "distributed_wall_s": 0.25,
            "slicing_wall_s": 1.0,
            "max_abs_error": 1.0e-2,
            "max_relative_error": 1.0e-10,
            **_passing_planner_cost_comparison(),
        }
    ]

    summary = evaluate_distributed_speed_gate(
        records,
        min_speedup=1.01,
        require_world_size=8,
        max_abs_error=1.0e-9,
        max_relative_error=1.0e-12,
        require_real_runtime=True,
    )

    assert summary["operation"] == "distributed_speed_gate"
    assert summary["status"] == "failed"
    assert summary["failures"] == [
        {
            "max_abs_error": 0.01,
            "max_relative_error": 1.0e-10,
            "operation": "row_sharded_speed_benchmark",
            "rank": 0,
            "reason": "error above absolute and relative thresholds",
            "relative_threshold": 1.0e-12,
            "threshold": 1.0e-9,
        }
    ]


def test_distributed_speed_cli_returns_nonzero_when_gate_fails(monkeypatch, capsys):
    from renormalizer.backend import distributed_smoke

    monkeypatch.setattr(distributed_smoke, "run_distributed_smoke", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        distributed_smoke,
        "run_distributed_speed_smoke",
        lambda *args, **kwargs: [
            {
                "operation": "row_sharded_speed_benchmark",
                "status": "passed",
                "rank": 0,
                "world_size": 8,
                "speedup_vs_slicing": 0.5,
                    "distributed_wall_s": 2.0,
                    "slicing_wall_s": 1.0,
                    "max_abs_error": 0.0,
                    **_passing_planner_cost_comparison(),
                }
            ],
    )

    rc = distributed_smoke.main([
        "--backends",
        "torch",
        "--device",
        "cuda",
        "--speed-benchmark",
        "--speed-gate",
        "--min-speedup",
        "1.01",
        "--require-world-size",
        "8",
    ])

    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    gate = payload["records"][-1]
    assert gate["operation"] == "distributed_speed_gate"
    assert gate["status"] == "failed"
    assert gate["failures"][0]["reason"] == "speedup below threshold"


def test_distributed_speed_cli_can_require_real_runtime(monkeypatch, capsys):
    from renormalizer.backend import distributed_smoke

    monkeypatch.setattr(distributed_smoke, "run_distributed_smoke", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        distributed_smoke,
        "run_distributed_speed_smoke",
        lambda *args, **kwargs: [
            {
                "operation": "row_sharded_speed_benchmark",
                "status": "passed",
                "rank": 0,
                "world_size": 8,
                "is_distributed_runtime": False,
                "speedup_vs_slicing": 4.0,
                    "distributed_wall_s": 0.25,
                    "slicing_wall_s": 1.0,
                    "max_abs_error": 0.0,
                    **_passing_planner_cost_comparison(),
                }
            ],
    )

    rc = distributed_smoke.main([
        "--backends",
        "torch",
        "--device",
        "cuda",
        "--speed-benchmark",
        "--speed-gate",
        "--min-speedup",
        "1.01",
        "--require-world-size",
        "8",
        "--require-real-runtime",
    ])

    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    gate = payload["records"][-1]
    assert gate["operation"] == "distributed_speed_gate"
    assert gate["status"] == "failed"
    assert gate["require_real_runtime"] is True
    assert gate["failures"] == [
        {
            "operation": "row_sharded_speed_benchmark",
            "rank": 0,
            "reason": "distributed runtime is simulated",
        }
    ]


def test_distributed_speed_smoke_skips_single_process_runtime():
    from renormalizer.backend.distributed_smoke import run_distributed_speed_smoke

    records = run_distributed_speed_smoke(
        "numpy",
        device="cpu",
        rows=8,
        shared_dim=8,
        cols=8,
        rank=0,
        world_size=1,
        repeat=1,
        warmup=0,
        trials=1,
    )

    assert len(records) == 1
    record = records[0]
    assert record["operation"] == "row_sharded_speed_benchmark"
    assert record["status"] == "skipped"
    assert record["rank"] == 0
    assert record["world_size"] == 1
    assert "world_size > 1" in record["error"]


def test_distributed_speed_smoke_records_runtime_kind_for_simulated_backend():
    from renormalizer.backend.distributed_smoke import run_distributed_speed_smoke

    records = run_distributed_speed_smoke(
        "numpy",
        device="cpu",
        rows=8,
        shared_dim=8,
        cols=8,
        rank=0,
        world_size=2,
        repeat=1,
        warmup=0,
        trials=1,
    )

    assert len(records) == 1
    record = records[0]
    assert record["operation"] == "row_sharded_speed_benchmark"
    assert record["status"] == "passed"
    assert record["world_size"] == 2
    assert record["is_distributed_runtime"] is False


def test_distributed_speed_smoke_records_planner_cost_comparison():
    from renormalizer.backend.distributed_smoke import run_distributed_speed_smoke

    records = run_distributed_speed_smoke(
        "numpy",
        device="cpu",
        rows=8,
        shared_dim=8,
        cols=8,
        rank=0,
        world_size=2,
        repeat=1,
        warmup=0,
        trials=1,
    )

    record = records[0]

    assert record["operation"] == "row_sharded_speed_benchmark"
    assert record["status"] == "passed"
    assert record["max_abs_error"] >= 0.0
    assert record["max_relative_error"] >= 0.0
    assert record["distributed_estimated_time_s"] > 0.0
    assert record["slicing_estimated_time_s"] > 0.0
    assert record["estimated_speedup_vs_slicing"] == pytest.approx(
        record["slicing_estimated_time_s"] / record["distributed_estimated_time_s"]
    )
    assert record["distributed_estimated_peak_bytes"] > 0
    assert record["slicing_estimated_peak_bytes"] > 0
    assert record["distributed_estimated_write_bytes"] < record["slicing_estimated_write_bytes"]
    assert record["distributed_estimated_comm_bytes"] > 0
    assert record["slicing_estimated_comm_bytes"] == 0


def test_distributed_profile_gate_requires_distributed_contraction_events():
    from renormalizer.backend.distributed_smoke import evaluate_distributed_profile_gate

    events = [
        {
            "event": "contraction_execute",
            "lowering": "distributed",
            "equation": "ik,kj->ij",
            "rank": 0,
            "world_size": 8,
            "global_shape": [32, 32],
            "local_shape": [4, 32],
            "distributed_modes": ["i"],
            "communication": [
                {"collective": "gather", "bytes": 8192, "wall_s": 0.01, "num_messages": 1, "block_size": 8192},
            ],
        },
        {
            "event": "contraction_execute",
            "lowering": "distributed",
            "equation": "ik,kj->ij",
            "rank": 0,
            "world_size": 8,
            "global_shape": [32, 32],
            "local_shape": [32, 32],
            "distributed_modes": [],
            "communication": [
                {"collective": "allreduce", "bytes": 8192, "wall_s": 0.02, "num_messages": 1, "block_size": 8192},
            ],
        },
        {
            "event": "contraction_execute",
            "lowering": "distributed",
            "equation": "ik,kj->ij",
            "rank": 0,
            "world_size": 8,
            "global_shape": [32, 32],
            "local_shape": [32, 4],
            "distributed_modes": ["j"],
            "communication": [
                {"collective": "reduce_scatter", "bytes": 8192, "wall_s": 0.025, "num_messages": 1, "block_size": 8192},
            ],
        },
        {
            "event": "contraction_execute",
            "lowering": "distributed",
            "equation": "ik,kj->ij",
            "rank": 0,
            "world_size": 8,
            "global_shape": [32, 32],
            "local_shape": [32, 4],
            "distributed_modes": ["j"],
            "communication": [
                {"collective": "allreduce", "bytes": 8192, "wall_s": 0.03, "num_messages": 1, "block_size": 8192},
            ],
        },
    ]

    summary = evaluate_distributed_profile_gate(events, require_world_size=8)

    assert summary["operation"] == "distributed_profile_gate"
    assert summary["status"] == "failed"
    assert summary["checked_count"] == 4
    assert summary["failures"] == [
        {
            "collective": "allreduce",
            "expected_collective": "broadcast",
            "reason": "missing distributed profile collective",
        },
        {
            "collective": "allreduce",
            "expected_collective": "alltoall",
            "reason": "missing distributed profile collective",
        }
    ]


def test_distributed_profile_gate_requires_communication_message_metadata():
    from renormalizer.backend.distributed_smoke import evaluate_distributed_profile_gate

    events = [
        {
            "event": "contraction_execute",
            "lowering": "distributed",
            "rank": 0,
            "world_size": 8,
            "global_shape": [32, 32],
            "local_shape": [32, 32],
            "distributed_modes": ["i"],
            "communication": [
                {"collective": "broadcast", "bytes": 8192, "wall_s": 0.01, "num_messages": 1, "block_size": 8192},
                {"collective": "allreduce", "bytes": 8192, "wall_s": 0.02, "block_size": 8192},
                {"collective": "reduce_scatter", "bytes": 8192, "wall_s": 0.025, "num_messages": 1, "block_size": 8192},
                {"collective": "alltoall", "bytes": 8192, "wall_s": 0.03, "num_messages": 1},
                {"collective": "gather", "bytes": 8192, "wall_s": 0.04, "num_messages": 1, "block_size": 8192},
            ],
        }
    ]

    summary = evaluate_distributed_profile_gate(events, require_world_size=8)

    assert summary["operation"] == "distributed_profile_gate"
    assert summary["status"] == "failed"
    assert summary["failures"] == [
        {
            "reason": "invalid communication num_messages",
            "rank": 0,
            "collective": "allreduce",
        },
        {
            "reason": "invalid communication block_size",
            "rank": 0,
            "collective": "alltoall",
        }
    ]


def test_distributed_profile_gate_rejects_inconsistent_communication_rollup():
    from renormalizer.backend.distributed_smoke import evaluate_distributed_profile_gate

    communication = [
        {"collective": "broadcast", "bytes": 1024, "wall_s": 0.01, "num_messages": 1, "block_size": 1024},
        {"collective": "allreduce", "bytes": 2048, "wall_s": 0.02, "num_messages": 1, "block_size": 2048},
        {"collective": "reduce_scatter", "bytes": 4096, "wall_s": 0.03, "num_messages": 1, "block_size": 4096},
        {"collective": "alltoall", "bytes": 8192, "wall_s": 0.04, "num_messages": 1, "block_size": 8192},
        {"collective": "gather", "bytes": 16384, "wall_s": 0.05, "num_messages": 1, "block_size": 16384},
    ]
    events = [
        {
            "event": "contraction_execute",
            "lowering": "distributed",
            "rank": 0,
            "world_size": 8,
            "global_shape": [32, 32],
            "local_shape": [4, 32],
            "distributed_modes": ["i"],
            "communication": communication,
            "communication_profile": {
                "bytes": 4096,
                "bytes_by_collective": {
                    "broadcast": 1024,
                    "allreduce": 2,
                    "reduce_scatter": 4096,
                    "alltoall": 8192,
                    "gather": 16384,
                },
                "num_messages": 4,
            },
        }
    ]

    summary = evaluate_distributed_profile_gate(events, require_world_size=8)

    assert summary["operation"] == "distributed_profile_gate"
    assert summary["status"] == "failed"
    assert summary["failures"] == [
        {
            "reason": "communication profile bytes do not match communication items",
            "rank": 0,
            "communication_profile_bytes": 4096,
            "communication_bytes": 31744,
        },
        {
            "reason": "communication profile collective bytes do not match communication items",
            "rank": 0,
            "communication_profile_bytes_by_collective": {
                "broadcast": 1024,
                "allreduce": 2,
                "reduce_scatter": 4096,
                "alltoall": 8192,
                "gather": 16384,
            },
            "communication_bytes_by_collective": {
                "broadcast": 1024,
                "allreduce": 2048,
                "reduce_scatter": 4096,
                "alltoall": 8192,
                "gather": 16384,
            },
        },
        {
            "reason": "communication profile num_messages does not match communication items",
            "rank": 0,
            "communication_profile_num_messages": 4,
            "communication_num_messages": 5,
        },
    ]


def test_distributed_profile_gate_rejects_unexpected_collective_names():
    from renormalizer.backend.distributed_smoke import evaluate_distributed_profile_gate

    events = [
        {
            "event": "contraction_execute",
            "lowering": "distributed",
            "rank": 0,
            "world_size": 8,
            "global_shape": [32, 32],
            "local_shape": [32, 32],
            "distributed_modes": ["i"],
            "communication": [
                {"collective": "broadcast", "bytes": 8192, "wall_s": 0.01, "num_messages": 1, "block_size": 8192},
                {"collective": "allreduce", "bytes": 8192, "wall_s": 0.02, "num_messages": 1, "block_size": 8192},
                {"collective": "reduce_scatter", "bytes": 8192, "wall_s": 0.025, "num_messages": 1, "block_size": 8192},
                {"collective": "alltoall", "bytes": 8192, "wall_s": 0.03, "num_messages": 1, "block_size": 8192},
                {"collective": "gather", "bytes": 8192, "wall_s": 0.04, "num_messages": 1, "block_size": 8192},
                {"collective": "mystery", "bytes": 8192, "wall_s": 0.05, "num_messages": 1, "block_size": 8192},
            ],
        }
    ]

    summary = evaluate_distributed_profile_gate(events, require_world_size=8)

    assert summary["operation"] == "distributed_profile_gate"
    assert summary["status"] == "failed"
    assert summary["failures"] == [
        {
            "reason": "unexpected distributed profile collective",
            "rank": 0,
            "collective": "mystery",
        }
    ]


def test_distributed_profile_gate_can_require_real_runtime():
    from renormalizer.backend.distributed_smoke import evaluate_distributed_profile_gate

    events = [
        {
            "event": "contraction_execute",
            "lowering": "distributed",
            "rank": 0,
            "world_size": 8,
            "is_distributed_runtime": False,
            "global_shape": [32, 32],
            "local_shape": [4, 32],
            "distributed_modes": ["i"],
            "communication": [
                {"collective": "broadcast", "bytes": 8192, "wall_s": 0.01, "num_messages": 1, "block_size": 8192},
                {"collective": "gather", "bytes": 8192, "wall_s": 0.02, "num_messages": 1, "block_size": 8192},
                {"collective": "allreduce", "bytes": 8192, "wall_s": 0.03, "num_messages": 1, "block_size": 8192},
                {"collective": "reduce_scatter", "bytes": 8192, "wall_s": 0.035, "num_messages": 1, "block_size": 8192},
                {"collective": "alltoall", "bytes": 8192, "wall_s": 0.04, "num_messages": 1, "block_size": 8192},
            ],
        }
    ]

    summary = evaluate_distributed_profile_gate(events, require_world_size=8, require_real_runtime=True)

    assert summary["operation"] == "distributed_profile_gate"
    assert summary["status"] == "failed"
    assert summary["require_real_runtime"] is True
    assert summary["failures"] == [
        {
            "reason": "distributed runtime is simulated",
            "rank": 0,
        }
    ]


def test_distributed_profile_cli_writes_ranked_profile_and_gates_it(monkeypatch, capsys, tmp_path):
    from renormalizer.backend import distributed_smoke

    profile_path = tmp_path / "profile.jsonl"

    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setattr(
        distributed_smoke,
        "run_distributed_smoke",
        lambda *args, **kwargs: [
            {
                "backend": "torch",
                "device": "cuda:3",
                "operation": "broadcast_tensor",
                "status": "passed",
                "rank": 3,
                "world_size": 8,
                "collectives": ["broadcast"],
                "max_abs_error": 0.0,
            },
            {
                "backend": "torch",
                "device": "cuda:3",
                "operation": "row_sharded_matmul",
                "status": "passed",
                "rank": 3,
                "world_size": 8,
                "collectives": ["gather"],
                "max_abs_error": 0.0,
            },
            {
                "backend": "torch",
                "device": "cuda:3",
                "operation": "contracted_sharded_allreduce",
                "status": "passed",
                "rank": 3,
                "world_size": 8,
                "collectives": ["allreduce"],
                "max_abs_error": 0.0,
            },
            {
                "backend": "torch",
                "device": "cuda:3",
                "operation": "redistribute_output_alltoall",
                "status": "passed",
                "rank": 3,
                "world_size": 8,
                "collectives": ["alltoall"],
                "max_abs_error": 0.0,
            },
        ],
    )
    monkeypatch.setattr(
        distributed_smoke,
        "read_jsonl",
        lambda path: [
            {
                "event": "contraction_execute",
                "lowering": "distributed",
                "rank": 3,
                "world_size": 8,
                "global_shape": [2, 3],
                "local_shape": [2, 3],
                "distributed_modes": [],
                "communication": [{"collective": "broadcast", "bytes": 48, "wall_s": 0.01, "num_messages": 1, "block_size": 48}],
            },
            {
                "event": "contraction_execute",
                "lowering": "distributed",
                "rank": 3,
                "world_size": 8,
                "global_shape": [32, 32],
                "local_shape": [4, 32],
                "distributed_modes": ["i"],
                "communication": [{"collective": "gather", "bytes": 8192, "wall_s": 0.01, "num_messages": 1, "block_size": 8192}],
            },
            {
                "event": "contraction_execute",
                "lowering": "distributed",
                "rank": 3,
                "world_size": 8,
                "global_shape": [32, 32],
                "local_shape": [32, 32],
                "distributed_modes": [],
                "communication": [{"collective": "allreduce", "bytes": 8192, "wall_s": 0.02, "num_messages": 1, "block_size": 8192}],
            },
            {
                "event": "contraction_execute",
                "lowering": "distributed",
                "rank": 3,
                "world_size": 8,
                "global_shape": [32, 32],
                "local_shape": [32, 4],
                "distributed_modes": ["j"],
                "communication": [{"collective": "reduce_scatter", "bytes": 8192, "wall_s": 0.025, "num_messages": 1, "block_size": 8192}],
            },
            {
                "event": "contraction_execute",
                "lowering": "distributed",
                "rank": 3,
                "world_size": 8,
                "global_shape": [32, 32],
                "local_shape": [32, 4],
                "distributed_modes": ["j"],
                "communication": [{"collective": "alltoall", "bytes": 8192, "wall_s": 0.03, "num_messages": 1, "block_size": 8192}],
            },
        ],
    )

    rc = distributed_smoke.main([
        "--backends",
        "torch",
        "--device",
        "cuda",
        "--profile-output",
        str(profile_path),
        "--profile-gate",
        "--require-world-size",
        "8",
    ])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    profile_gate = payload["records"][-1]
    assert profile_gate["operation"] == "distributed_profile_gate"
    assert profile_gate["status"] == "passed"
    assert profile_gate["profile_output"].endswith("profile-rank3.jsonl")


def test_distributed_profile_cli_can_require_real_runtime(monkeypatch, capsys, tmp_path):
    from renormalizer.backend import distributed_smoke

    profile_path = tmp_path / "profile.jsonl"

    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setattr(distributed_smoke, "run_distributed_smoke", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        distributed_smoke,
        "read_jsonl",
        lambda path: [
            {
                "event": "contraction_execute",
                "lowering": "distributed",
                "rank": 0,
                "world_size": 8,
                "is_distributed_runtime": False,
                "global_shape": [32, 32],
                "local_shape": [4, 32],
                "distributed_modes": ["i"],
                "communication": [
                    {"collective": "broadcast", "bytes": 8192, "wall_s": 0.01, "num_messages": 1, "block_size": 8192},
                    {"collective": "gather", "bytes": 8192, "wall_s": 0.02, "num_messages": 1, "block_size": 8192},
                    {"collective": "allreduce", "bytes": 8192, "wall_s": 0.03, "num_messages": 1, "block_size": 8192},
                    {"collective": "reduce_scatter", "bytes": 8192, "wall_s": 0.035, "num_messages": 1, "block_size": 8192},
                    {"collective": "alltoall", "bytes": 8192, "wall_s": 0.04, "num_messages": 1, "block_size": 8192},
                ],
            }
        ],
    )

    rc = distributed_smoke.main([
        "--backends",
        "torch",
        "--device",
        "cuda",
        "--profile-output",
        str(profile_path),
        "--profile-gate",
        "--require-world-size",
        "8",
        "--require-real-runtime",
    ])

    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    profile_gate = payload["records"][-1]
    assert profile_gate["operation"] == "distributed_profile_gate"
    assert profile_gate["status"] == "failed"
    assert profile_gate["require_real_runtime"] is True
    assert profile_gate["failures"] == [
        {
            "reason": "distributed runtime is simulated",
            "rank": 0,
        }
    ]


def test_distributed_profile_cli_suppresses_stream_profile_output(monkeypatch, capsys, tmp_path):
    from renormalizer.backend import distributed_smoke

    calls = []
    monkeypatch.setattr(distributed_smoke, "disable_stream_output", lambda: calls.append("disabled"))
    monkeypatch.setattr(distributed_smoke, "run_distributed_smoke", lambda *args, **kwargs: [])

    rc = distributed_smoke.main([
        "--backends",
        "torch",
        "--device",
        "cuda",
        "--profile-output",
        str(tmp_path / "profile.jsonl"),
    ])

    assert rc == 0
    assert calls == ["disabled"]
    assert json.loads(capsys.readouterr().out) == {"records": []}


def test_distributed_smoke_gate_requires_expected_collective_cases():
    from renormalizer.backend.distributed_smoke import evaluate_distributed_smoke_gate

    records = [
        {
            "operation": "row_sharded_matmul",
            "status": "passed",
            "world_size": 8,
            "collectives": ["gather"],
            "max_abs_error": 0.0,
        },
        {
            "operation": "contracted_sharded_allreduce",
            "status": "passed",
            "world_size": 8,
            "collectives": ["allreduce"],
            "max_abs_error": 0.0,
        },
        {
            "operation": "redistribute_output_alltoall",
            "status": "passed",
            "world_size": 8,
            "collectives": ["allreduce"],
            "max_abs_error": 0.0,
        },
    ]

    summary = evaluate_distributed_smoke_gate(records, require_world_size=8)

    assert summary["operation"] == "distributed_smoke_gate"
    assert summary["status"] == "failed"
    assert summary["checked_count"] == 3
    assert summary["failures"] == [
        {
            "operation": "broadcast_tensor",
            "reason": "missing distributed smoke record",
        },
        {
            "operation": "reduce_scatter_output",
            "reason": "missing distributed smoke record",
        },
        {
            "operation": "redistribute_output_alltoall",
            "reason": "unexpected collectives",
            "collectives": ["allreduce"],
            "expected_collectives": ["alltoall"],
        }
    ]


def test_distributed_smoke_gate_requires_collective_byte_rollups():
    from renormalizer.backend.distributed_smoke import evaluate_distributed_smoke_gate

    records = [
        {
            "operation": "broadcast_tensor",
            "status": "passed",
            "rank": 0,
            "world_size": 4,
            "collectives": ["broadcast"],
            "plan_comm_bytes": 48,
            "max_abs_error": 0.0,
        },
        {
            "operation": "row_sharded_matmul",
            "status": "passed",
            "rank": 0,
            "world_size": 4,
            "collectives": ["gather"],
            "plan_comm_bytes": 128,
            "plan_collective_bytes": {"gather": 64},
            "max_abs_error": 0.0,
        },
        {
            "operation": "contracted_sharded_allreduce",
            "status": "passed",
            "rank": 0,
            "world_size": 4,
            "collectives": ["allreduce"],
            "plan_comm_bytes": 256,
            "plan_collective_bytes": {"allreduce": 256},
            "max_abs_error": 0.0,
        },
        {
            "operation": "reduce_scatter_output",
            "status": "passed",
            "rank": 0,
            "world_size": 4,
            "collectives": ["reduce_scatter"],
            "plan_comm_bytes": 512,
            "plan_collective_bytes": {"reduce_scatter": 512},
            "max_abs_error": 0.0,
        },
        {
            "operation": "redistribute_output_alltoall",
            "status": "passed",
            "rank": 0,
            "world_size": 4,
            "collectives": ["alltoall"],
            "plan_comm_bytes": 1024,
            "plan_collective_bytes": {"alltoall": 1024},
            "max_abs_error": 0.0,
        },
    ]

    summary = evaluate_distributed_smoke_gate(records, require_world_size=4)

    assert summary["operation"] == "distributed_smoke_gate"
    assert summary["status"] == "failed"
    assert summary["failures"] == [
        {
            "operation": "broadcast_tensor",
            "rank": 0,
            "reason": "missing collective byte rollup",
        },
        {
            "operation": "row_sharded_matmul",
            "rank": 0,
            "reason": "collective byte rollup does not match total",
            "plan_comm_bytes": 128,
            "plan_collective_bytes": {"gather": 64},
        },
    ]


def test_distributed_smoke_gate_can_require_real_runtime():
    from renormalizer.backend.distributed_smoke import evaluate_distributed_smoke_gate

    records = [
        {
            "operation": "broadcast_tensor",
            "status": "passed",
            "rank": 0,
            "world_size": 4,
            "is_distributed_runtime": False,
            "collectives": ["broadcast"],
            "max_abs_error": 0.0,
        },
        {
            "operation": "row_sharded_matmul",
            "status": "passed",
            "rank": 0,
            "world_size": 4,
            "is_distributed_runtime": False,
            "collectives": ["gather"],
            "max_abs_error": 0.0,
        },
        {
            "operation": "contracted_sharded_allreduce",
            "status": "passed",
            "rank": 0,
            "world_size": 4,
            "is_distributed_runtime": False,
            "collectives": ["allreduce"],
            "max_abs_error": 0.0,
        },
        {
            "operation": "reduce_scatter_output",
            "status": "passed",
            "rank": 0,
            "world_size": 4,
            "is_distributed_runtime": False,
            "collectives": ["reduce_scatter"],
            "max_abs_error": 0.0,
        },
        {
            "operation": "redistribute_output_alltoall",
            "status": "passed",
            "rank": 0,
            "world_size": 4,
            "is_distributed_runtime": False,
            "collectives": ["alltoall"],
            "max_abs_error": 0.0,
        },
    ]

    summary = evaluate_distributed_smoke_gate(records, require_world_size=4, require_real_runtime=True)

    assert summary["operation"] == "distributed_smoke_gate"
    assert summary["status"] == "failed"
    assert summary["require_real_runtime"] is True
    assert summary["failures"] == [
        {
            "operation": "broadcast_tensor",
            "rank": 0,
            "reason": "distributed runtime is simulated",
        },
        {
            "operation": "row_sharded_matmul",
            "rank": 0,
            "reason": "distributed runtime is simulated",
        },
        {
            "operation": "contracted_sharded_allreduce",
            "rank": 0,
            "reason": "distributed runtime is simulated",
        },
        {
            "operation": "reduce_scatter_output",
            "rank": 0,
            "reason": "distributed runtime is simulated",
        },
        {
            "operation": "redistribute_output_alltoall",
            "rank": 0,
            "reason": "distributed runtime is simulated",
        },
    ]


def test_distributed_smoke_cli_returns_nonzero_when_gate_fails(monkeypatch, capsys):
    from renormalizer.backend import distributed_smoke

    monkeypatch.setattr(
        distributed_smoke,
        "run_distributed_smoke",
        lambda *args, **kwargs: [
            {
                "backend": "torch",
                "device": "cuda:0",
                "operation": "row_sharded_matmul",
                "status": "passed",
                "rank": 0,
                "world_size": 8,
                "collectives": ["gather"],
                "max_abs_error": 0.0,
            }
        ],
    )

    rc = distributed_smoke.main([
        "--backends",
        "torch",
        "--device",
        "cuda",
        "--distributed-gate",
        "--require-world-size",
        "8",
    ])

    assert rc == 2
    payload = json.loads(capsys.readouterr().out)
    gate = payload["records"][-1]
    assert gate["operation"] == "distributed_smoke_gate"
    assert gate["status"] == "failed"
    assert {failure["operation"] for failure in gate["failures"]} == {
        "broadcast_tensor",
        "contracted_sharded_allreduce",
        "reduce_scatter_output",
        "redistribute_output_alltoall",
    }


def test_numpy_distributed_smoke_records_planned_collective_cases():
    from renormalizer.backend.distributed_smoke import run_distributed_smoke

    records = run_distributed_smoke(
        "numpy",
        device="cpu",
        rows=8,
        shared_dim=4,
        cols=8,
        world_size=4,
    )

    operations = {record["operation"] for record in records}
    assert operations == {
        "broadcast_tensor",
        "row_sharded_matmul",
        "contracted_sharded_allreduce",
        "reduce_scatter_output",
        "redistribute_output_alltoall",
    }
    for record in records:
        assert record["backend"] == "numpy"
        assert record["device"] == "cpu"
        assert record["rank"] == 0
        assert record["world_size"] == 4
        assert record["is_distributed_runtime"] is False
        assert record["status"] == "passed"
        assert record["max_abs_error"] <= 1e-9
        assert record["global_shape"]
        assert record["local_shape"]
        assert record["plan_comm_bytes"] >= 0
        assert sorted(record["plan_collective_bytes"]) == sorted(record["collectives"])
        assert sum(record["plan_collective_bytes"].values()) == record["plan_comm_bytes"]
        json.dumps(record)

    by_operation = {record["operation"]: record for record in records}
    assert by_operation["broadcast_tensor"]["collectives"] == ["broadcast"]
    assert by_operation["broadcast_tensor"]["plan_collective_bytes"] == {"broadcast": 48}
    assert by_operation["row_sharded_matmul"]["collectives"] == ["gather"]
    assert by_operation["contracted_sharded_allreduce"]["collectives"] == ["allreduce"]
    assert by_operation["reduce_scatter_output"]["collectives"] == ["reduce_scatter"]
    assert by_operation["redistribute_output_alltoall"]["collectives"] == ["alltoall"]


def test_numpy_distributed_smoke_records_broadcast_profile_event(tmp_path):
    from renormalizer.backend.distributed_smoke import run_distributed_smoke
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    event_path = tmp_path / "profile.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)
        run_distributed_smoke(
            "numpy",
            device="cpu",
            rows=8,
            shared_dim=4,
            cols=8,
            world_size=4,
        )
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    events = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(
        event
        for event in events
        if event.get("event") == "contraction_execute"
        and event.get("lowering") == "distributed"
        and any(
            item.get("collective") == "broadcast"
            for item in event.get("communication") or []
        )
    )
    assert event["compute_class"] == "contraction_plan"
    assert event["compute_profile"]["compute_class"] == "contraction_plan"
    assert event["compute_profile"]["workload_signature"]["compute_class"] == "contraction_plan"
    assert event["compute_subclass"] == "backend_execute"
    assert event["compute_role"] == "kernel"
    assert event["output_strides"] == [24, 8]
    assert event["output_order"] == "C"
    assert event["output_contiguous"] is True
    assert event["output_backend"] == "numpy"
    assert event["output_location"] == "host"
    assert event["output_is_host"] is True
    assert event["output_is_device"] is False
    assert event["output_is_distributed"] is False
    assert event["layout_profile"]["output_layout"] == {
        "shape": [2, 3],
        "strides": [24, 8],
        "order": "C",
        "contiguous": True,
    }


def test_numpy_distributed_smoke_records_distributed_contraction_profile_events(tmp_path):
    from renormalizer.backend.distributed_smoke import run_distributed_smoke
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    event_path = tmp_path / "profile.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)
        run_distributed_smoke(
            "numpy",
            device="cpu",
            rows=8,
            shared_dim=4,
            cols=8,
            world_size=4,
        )
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    events = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    distributed_events = [
        event
        for event in events
        if event.get("event") == "contraction_execute"
        and event.get("lowering") == "distributed"
    ]
    by_collective = {
        item["collective"]: event
        for event in distributed_events
        for item in event.get("communication") or []
    }
    assert sorted(by_collective) == [
        "allreduce",
        "alltoall",
        "broadcast",
        "gather",
        "reduce_scatter",
    ]
    for collective, event in by_collective.items():
        assert event["compute_class"] == "contraction_plan"
        assert event["communication_profile"]["bytes_by_collective"][collective] > 0
        if collective == "broadcast":
            assert event["output_location"] == "host"
            continue
        assert event["output_location"] == "distributed"
        assert event["output_is_distributed"] is True
        assert event["output_is_host"] is False
        assert event["output_is_device"] is False
        assert event["output_local_shape"]
        assert event["output_local_strides"]
        assert len(event["output_local_strides"]) == len(event["output_local_shape"])
        assert event["output_local_order"] in {"C", "F", "unknown"}
        assert isinstance(event["output_local_contiguous"], bool)
