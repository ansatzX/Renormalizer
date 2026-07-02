# -*- coding: utf-8 -*-

import json


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
                {"collective": "gather", "bytes": 8192, "wall_s": 0.01},
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
                {"collective": "allreduce", "bytes": 8192, "wall_s": 0.02},
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
                {"collective": "allreduce", "bytes": 8192, "wall_s": 0.03},
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
            "expected_collective": "alltoall",
            "reason": "missing distributed profile collective",
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
                "global_shape": [32, 32],
                "local_shape": [4, 32],
                "distributed_modes": ["i"],
                "communication": [{"collective": "gather", "bytes": 8192, "wall_s": 0.01}],
            },
            {
                "event": "contraction_execute",
                "lowering": "distributed",
                "rank": 3,
                "world_size": 8,
                "global_shape": [32, 32],
                "local_shape": [32, 32],
                "distributed_modes": [],
                "communication": [{"collective": "allreduce", "bytes": 8192, "wall_s": 0.02}],
            },
            {
                "event": "contraction_execute",
                "lowering": "distributed",
                "rank": 3,
                "world_size": 8,
                "global_shape": [32, 32],
                "local_shape": [32, 4],
                "distributed_modes": ["j"],
                "communication": [{"collective": "alltoall", "bytes": 8192, "wall_s": 0.03}],
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
        assert record["global_shape"] == [8, 8]
        assert record["local_shape"]
        assert record["plan_hash"]
        assert record["plan_comm_bytes"] >= 0
        json.dumps(record)

    by_operation = {record["operation"]: record for record in records}
    assert by_operation["row_sharded_matmul"]["collectives"] == ["gather"]
    assert by_operation["contracted_sharded_allreduce"]["collectives"] == ["allreduce"]
    assert by_operation["redistribute_output_alltoall"]["collectives"] == ["alltoall"]
