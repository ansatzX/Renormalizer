# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from renormalizer.backend.config import BackendConfig
from renormalizer.backend.execution import (
    DeviceMesh,
    DeviceSpec,
    DistributedContractionPlan,
    DistributedContractionSpec,
    ShardingSpec,
)
from renormalizer.backend.factory import create_backend, is_backend_available, normalize_backend_name


DEFAULT_BACKENDS = ("torch",)


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    return int(value)


def _env_rank_world(rank=None, world_size=None):
    env_rank = _env_int("LOCAL_RANK", _env_int("RANK", 0))
    env_world = _env_int("LOCAL_WORLD_SIZE", _env_int("WORLD_SIZE", 1))
    return (
        env_rank if rank is None else int(rank),
        env_world if world_size is None else int(world_size),
    )


def _device_for_rank(device, rank):
    value = str(device or "cpu").lower()
    if value in ("gpu", "cuda"):
        return "cuda:{0}".format(int(rank))
    return str(device or "cpu")


def _device_kind(device):
    value = str(device or "cpu").lower()
    if value in ("gpu", "cuda") or value.startswith(("cuda:", "gpu:")):
        return "cuda"
    return "cpu"


def _mesh_for_runtime(backend_name, device, rank, world_size):
    kind = _device_kind(device)
    devices = tuple(
        DeviceSpec(
            kind=kind,
            index=index if kind == "cuda" else None,
            local_rank=index,
            global_rank=index,
            visible_id=str(index) if kind == "cuda" else None,
        )
        for index in range(int(world_size))
    )
    return DeviceMesh(
        devices=devices,
        shape=(int(world_size),),
        axis_names=("rank",),
        backend=backend_name,
        local_rank=int(rank),
        global_rank=int(rank),
    )


def _parse_backend_names(value):
    names = []
    for item in str(value).split(","):
        item = item.strip().lower()
        if not item:
            continue
        candidates = DEFAULT_BACKENDS if item == "all" else (item,)
        for candidate in candidates:
            normalized = normalize_backend_name(candidate)
            if normalized not in names:
                names.append(normalized)
    return names


def _to_numpy(backend, value):
    return np.asarray(backend.to_numpy(value))


def _make_inputs(rows, shared_dim, cols):
    left = np.arange(int(rows) * int(shared_dim), dtype=np.float64).reshape(int(rows), int(shared_dim))
    right = np.arange(int(shared_dim) * int(cols), dtype=np.float64).reshape(int(shared_dim), int(cols))
    left = (left + 1.0) / max(int(shared_dim), 1)
    right = (right - 3.0) / max(int(cols), 1)
    return left, right


def _distributed_plan(plan):
    if not plan.steps:
        return None
    inner = plan.steps[0].plan
    if isinstance(inner, DistributedContractionPlan):
        return inner
    return None


def _collectives(plan):
    distributed_plan = _distributed_plan(plan)
    if distributed_plan is None or not distributed_plan.steps:
        return []
    return [item.kind for item in distributed_plan.steps[0].communication]


def _plan_comm_bytes(plan):
    distributed_plan = _distributed_plan(plan)
    if distributed_plan is None:
        return 0
    return int(distributed_plan.total_comm_bytes)


def _record_result(backend, backend_name, device, operation, plan, result, expected, rank, world_size):
    gathered = backend.gather_tensor(result)
    actual = _to_numpy(backend, gathered)
    error = float(np.max(np.abs(actual - expected)))
    local_shape = tuple(int(dim) for dim in getattr(result, "local_shape", ()))
    return {
        "backend": backend_name,
        "device": device,
        "device_spec": str(backend.current_device()),
        "operation": operation,
        "status": "passed",
        "rank": int(rank),
        "world_size": int(world_size),
        "global_shape": list(result.global_shape),
        "local_shape": list(local_shape),
        "output_sharded_modes": [str(mode) for mode in result.sharding.sharded_modes],
        "collectives": _collectives(plan),
        "plan_hash": plan.plan_hash,
        "plan_comm_bytes": _plan_comm_bytes(plan),
        "max_abs_error": error,
        "rank_local_arrays": result.rank_local_arrays is not None,
    }


def _close_distributed_runtime(backend):
    dist = getattr(backend, "_distributed", None)
    if dist is None:
        return
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        return


def _row_sharded_case(backend, backend_name, device, mesh, left_np, right_np, rank, world_size):
    left = backend.to_backend(left_np)
    right = backend.to_backend(right_np)
    left_spec = ShardingSpec(
        global_shape=left_np.shape,
        modes=("i", "k"),
        mesh=mesh,
        ranks_per_mode={"i": world_size},
        mode_to_mesh_axis={"i": "rank"},
    )
    spec = DistributedContractionSpec(
        equation="ik,kj->ij",
        operands=(backend.shard_tensor(left, left_spec), right),
    )
    plan = backend.plan_contraction(spec, allow_distribution=True)
    result = backend.distributed_contract(spec, plan=plan)
    return _record_result(
        backend,
        backend_name,
        device,
        "row_sharded_matmul",
        plan,
        result,
        left_np @ right_np,
        rank,
        world_size,
    )


def _contracted_sharded_case(backend, backend_name, device, mesh, left_np, right_np, rank, world_size):
    left = backend.to_backend(left_np)
    right = backend.to_backend(right_np)
    left_spec = ShardingSpec(
        global_shape=left_np.shape,
        modes=("i", "k"),
        mesh=mesh,
        ranks_per_mode={"k": world_size},
        mode_to_mesh_axis={"k": "rank"},
    )
    right_spec = ShardingSpec(
        global_shape=right_np.shape,
        modes=("k", "j"),
        mesh=mesh,
        ranks_per_mode={"k": world_size},
        mode_to_mesh_axis={"k": "rank"},
    )
    spec = DistributedContractionSpec(
        equation="ik,kj->ij",
        operands=(backend.shard_tensor(left, left_spec), backend.shard_tensor(right, right_spec)),
    )
    plan = backend.plan_contraction(spec, allow_distribution=True)
    result = backend.distributed_contract(spec, plan=plan)
    return _record_result(
        backend,
        backend_name,
        device,
        "contracted_sharded_allreduce",
        plan,
        result,
        left_np @ right_np,
        rank,
        world_size,
    )


def _redistribute_output_case(backend, backend_name, device, mesh, left_np, right_np, rank, world_size):
    left = backend.to_backend(left_np)
    right = backend.to_backend(right_np)
    left_spec = ShardingSpec(
        global_shape=left_np.shape,
        modes=("i", "k"),
        mesh=mesh,
        ranks_per_mode={"i": world_size},
        mode_to_mesh_axis={"i": "rank"},
    )
    output_spec = ShardingSpec(
        global_shape=(left_np.shape[0], right_np.shape[1]),
        modes=("i", "j"),
        mesh=mesh,
        ranks_per_mode={"j": world_size},
        mode_to_mesh_axis={"j": "rank"},
    )
    spec = DistributedContractionSpec(
        equation="ik,kj->ij",
        operands=(backend.shard_tensor(left, left_spec), right),
        output_sharding=output_spec,
    )
    plan = backend.plan_contraction(spec, allow_distribution=True)
    result = backend.distributed_contract(spec, plan=plan)
    return _record_result(
        backend,
        backend_name,
        device,
        "redistribute_output_alltoall",
        plan,
        result,
        left_np @ right_np,
        rank,
        world_size,
    )


def run_distributed_smoke(
    backend_name,
    *,
    device="cuda",
    rows=16,
    shared_dim=16,
    cols=16,
    rank=None,
    world_size=None,
):
    backend_name = normalize_backend_name(backend_name)
    if not is_backend_available(backend_name):
        return [
            {
                "backend": backend_name,
                "device": device,
                "operation": "backend_import",
                "status": "skipped",
                "error": "backend is not available",
            }
        ]
    rank, world_size = _env_rank_world(rank=rank, world_size=world_size)
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if rows < world_size or shared_dim < world_size or cols < world_size:
        raise ValueError("rows, shared_dim, and cols must be at least world_size")

    backend_device = _device_for_rank(device, rank)
    backend = create_backend(backend_name, config=BackendConfig(device=backend_device, precision=64))
    try:
        if getattr(backend, "is_distributed", False):
            rank = int(backend.rank)
            world_size = int(backend.size)
        mesh = _mesh_for_runtime(backend_name, backend_device, rank, world_size)
        left_np, right_np = _make_inputs(rows, shared_dim, cols)

        return [
            _row_sharded_case(backend, backend_name, backend_device, mesh, left_np, right_np, rank, world_size),
            _contracted_sharded_case(backend, backend_name, backend_device, mesh, left_np, right_np, rank, world_size),
            _redistribute_output_case(backend, backend_name, backend_device, mesh, left_np, right_np, rank, world_size),
        ]
    finally:
        _close_distributed_runtime(backend)


def write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description="Smoke-test backend distributed contraction primitives.")
    parser.add_argument("--backends", default="torch", help="Comma-separated backend list, or 'all'.")
    parser.add_argument("--device", default="cuda", help="Backend device, e.g. cpu, cuda, or gpu.")
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--shared-dim", type=int, default=16)
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--output", default=None, help="Optional JSONL output path.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    all_records = []
    for backend_name in _parse_backend_names(args.backends):
        records = run_distributed_smoke(
            backend_name,
            device=args.device,
            rows=args.rows,
            shared_dim=args.shared_dim,
            cols=args.cols,
        )
        all_records.extend(records)
    if args.output is not None:
        write_jsonl(all_records, args.output)
    print(json.dumps({"records": all_records}, sort_keys=True))
    if any(record.get("status") == "error" for record in all_records):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
