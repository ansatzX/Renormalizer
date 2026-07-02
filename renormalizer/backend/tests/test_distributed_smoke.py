# -*- coding: utf-8 -*-

import json


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
