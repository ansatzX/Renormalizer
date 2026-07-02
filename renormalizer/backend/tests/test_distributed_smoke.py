# -*- coding: utf-8 -*-

import json


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
