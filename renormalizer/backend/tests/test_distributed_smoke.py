# -*- coding: utf-8 -*-

import json


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
                {"collective": "allreduce", "bytes": 8192, "wall_s": 0.03, "num_messages": 1, "block_size": 8192},
            ],
        },
    ]

    summary = evaluate_distributed_profile_gate(events, require_world_size=8)

    assert summary["operation"] == "distributed_profile_gate"
    assert summary["status"] == "failed"
    assert summary["checked_count"] == 3
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
            "operation": "redistribute_output_alltoall",
            "reason": "unexpected collectives",
            "collectives": ["allreduce"],
            "expected_collectives": ["alltoall"],
        }
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
        "redistribute_output_alltoall",
    }
    for record in records:
        assert record["backend"] == "numpy"
        assert record["device"] == "cpu"
        assert record["rank"] == 0
        assert record["world_size"] == 4
        assert record["status"] == "passed"
        assert record["max_abs_error"] <= 1e-9
        assert record["global_shape"]
        assert record["local_shape"]
        assert record["plan_comm_bytes"] >= 0
        json.dumps(record)

    by_operation = {record["operation"]: record for record in records}
    assert by_operation["broadcast_tensor"]["collectives"] == ["broadcast"]
    assert by_operation["row_sharded_matmul"]["collectives"] == ["gather"]
    assert by_operation["contracted_sharded_allreduce"]["collectives"] == ["allreduce"]
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
    assert any(
        event.get("event") == "contraction_execute"
        and event.get("lowering") == "distributed"
        and any(
            item.get("collective") == "broadcast"
            for item in event.get("communication") or []
        )
        for event in events
    )
