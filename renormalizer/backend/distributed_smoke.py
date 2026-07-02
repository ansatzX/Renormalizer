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
from renormalizer.utils.log import disable_stream_output


DEFAULT_BACKENDS = ("torch",)
EXPECTED_COLLECTIVES = {
    "broadcast_tensor": ("broadcast",),
    "row_sharded_matmul": ("gather",),
    "contracted_sharded_allreduce": ("allreduce",),
    "redistribute_output_alltoall": ("alltoall",),
}
EXPECTED_PROFILE_COLLECTIVES = ("allreduce", "alltoall", "gather")


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


def _sync_value(value):
    block_until_ready = getattr(value, "block_until_ready", None)
    if block_until_ready is not None:
        block_until_ready()


def _sync_backend(backend, value=None):
    try:
        if value is not None:
            if isinstance(value, (list, tuple)):
                for item in value:
                    _sync_value(item)
            else:
                _sync_value(value)
        backend.sync()
    except Exception:
        pass


def _run_repeated(repeat, fn):
    result = None
    for _ in range(int(repeat)):
        result = fn()
    return result


def _time_call(backend, repeat, fn, *, warmup=0, trials=1):
    import time

    repeat = int(repeat)
    warmup = max(0, int(warmup))
    trials = max(1, int(trials))
    _sync_backend(backend)
    result = None
    for _ in range(warmup):
        result = _run_repeated(repeat, fn)
        _sync_backend(backend, result)

    samples = []
    for _ in range(trials):
        _sync_backend(backend)
        started = time.perf_counter()
        result = _run_repeated(repeat, fn)
        _sync_backend(backend, result)
        samples.append(float(time.perf_counter() - started))
    return result, float(np.median(samples)), samples


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


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _ranked_profile_path(path, rank, world_size):
    path = Path(path)
    if "{rank}" in os.fspath(path):
        return Path(os.fspath(path).format(rank=int(rank), world_size=int(world_size)))
    if int(world_size) <= 1:
        return path
    return path.with_name("{0}-rank{1}{2}".format(path.stem, int(rank), path.suffix or ".jsonl"))


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


def _gate_failure(record, reason, **extra):
    failure = {
        "operation": record.get("operation"),
        "reason": reason,
    }
    if "backend" in record:
        failure["backend"] = record.get("backend")
    if "device" in record:
        failure["device"] = record.get("device")
    if "rank" in record:
        failure["rank"] = record.get("rank")
    failure.update(extra)
    return failure


def evaluate_distributed_smoke_gate(records, *, require_world_size=None, max_abs_error=1e-9):
    failures = []
    records_by_operation = {
        record.get("operation"): record
        for record in records
        if record.get("operation") in EXPECTED_COLLECTIVES
    }

    for operation, expected_collectives in EXPECTED_COLLECTIVES.items():
        record = records_by_operation.get(operation)
        if record is None:
            failures.append({
                "operation": operation,
                "reason": "missing distributed smoke record",
            })
            continue
        if record.get("status") != "passed":
            failures.append(_gate_failure(record, "distributed smoke record did not pass"))
            continue
        if require_world_size is not None and int(record.get("world_size") or 0) != int(require_world_size):
            failures.append(_gate_failure(
                record,
                "unexpected world size",
                world_size=record.get("world_size"),
                expected_world_size=int(require_world_size),
            ))
        collectives = list(record.get("collectives") or [])
        if collectives != list(expected_collectives):
            failures.append(_gate_failure(
                record,
                "unexpected collectives",
                collectives=collectives,
                expected_collectives=list(expected_collectives),
            ))
        error = record.get("max_abs_error")
        if error is None or float(error) > float(max_abs_error):
            failures.append(_gate_failure(
                record,
                "max_abs_error above threshold",
                max_abs_error=error,
                threshold=float(max_abs_error),
            ))

    return {
        "operation": "distributed_smoke_gate",
        "status": "failed" if failures else "passed",
        "checked_count": int(len(records_by_operation)),
        "required_world_size": None if require_world_size is None else int(require_world_size),
        "max_abs_error": float(max_abs_error),
        "required_operations": sorted(EXPECTED_COLLECTIVES),
        "failures": failures,
    }


def _distributed_profile_events(events):
    return [
        event for event in events
        if event.get("event") == "contraction_execute"
        and event.get("lowering") == "distributed"
    ]


def evaluate_distributed_profile_gate(events, *, require_world_size=None):
    profile_events = _distributed_profile_events(events)
    failures = []

    for expected_collective in EXPECTED_PROFILE_COLLECTIVES:
        matches = [
            event for event in profile_events
            for item in event.get("communication") or []
            if item.get("collective") == expected_collective
        ]
        if not matches:
            observed_collectives = [
                item.get("collective")
                for event in profile_events
                for item in event.get("communication") or []
            ]
            failures.append({
                "reason": "missing distributed profile collective",
                "collective": observed_collectives[-1] if observed_collectives else None,
                "expected_collective": expected_collective,
            })

    for event in profile_events:
        if require_world_size is not None and int(event.get("world_size") or 0) != int(require_world_size):
            failures.append({
                "reason": "unexpected world size",
                "rank": event.get("rank"),
                "world_size": event.get("world_size"),
                "expected_world_size": int(require_world_size),
            })
        if "rank" not in event:
            failures.append({"reason": "missing rank"})
        if not event.get("global_shape"):
            failures.append({"reason": "missing global shape", "rank": event.get("rank")})
        if not event.get("local_shape"):
            failures.append({"reason": "missing local shape", "rank": event.get("rank")})
        if "distributed_modes" not in event:
            failures.append({"reason": "missing distributed modes", "rank": event.get("rank")})
        communication = event.get("communication") or []
        if not communication:
            failures.append({"reason": "missing communication profile", "rank": event.get("rank")})
        for item in communication:
            if item.get("bytes") is None or int(item.get("bytes") or 0) < 0:
                failures.append({
                    "reason": "invalid communication bytes",
                    "rank": event.get("rank"),
                    "collective": item.get("collective"),
                })
            if item.get("wall_s") is None or float(item.get("wall_s") or 0.0) < 0.0:
                failures.append({
                    "reason": "invalid communication wall time",
                    "rank": event.get("rank"),
                    "collective": item.get("collective"),
                })

    return {
        "operation": "distributed_profile_gate",
        "status": "failed" if failures else "passed",
        "checked_count": int(len(profile_events)),
        "required_world_size": None if require_world_size is None else int(require_world_size),
        "required_collectives": list(EXPECTED_PROFILE_COLLECTIVES),
        "failures": failures,
    }


def _speed_records(records):
    return [
        record for record in records
        if record.get("operation") == "row_sharded_speed_benchmark"
    ]


def evaluate_distributed_speed_gate(records, *, min_speedup=1.0, require_world_size=None, max_abs_error=1e-9):
    speed_records = _speed_records(records)
    failures = []
    if not speed_records:
        failures.append({
            "operation": "row_sharded_speed_benchmark",
            "reason": "missing distributed speed benchmark record",
        })
    for record in speed_records:
        if record.get("status") != "passed":
            failures.append(_gate_failure(record, "distributed speed benchmark did not pass"))
            continue
        if require_world_size is not None and int(record.get("world_size") or 0) != int(require_world_size):
            failures.append(_gate_failure(
                record,
                "unexpected world size",
                world_size=record.get("world_size"),
                expected_world_size=int(require_world_size),
            ))
        error = record.get("max_abs_error")
        if error is None or float(error) > float(max_abs_error):
            failures.append(_gate_failure(
                record,
                "max_abs_error above threshold",
                max_abs_error=error,
                threshold=float(max_abs_error),
            ))
        speedup = record.get("speedup_vs_slicing")
        if speedup is None or float(speedup) < float(min_speedup):
            failures.append(_gate_failure(
                record,
                "speedup below threshold",
                speedup_vs_slicing=speedup,
            ))

    return {
        "operation": "distributed_speed_gate",
        "status": "failed" if failures else "passed",
        "checked_count": int(len(speed_records)),
        "min_speedup": float(min_speedup),
        "required_world_size": None if require_world_size is None else int(require_world_size),
        "max_abs_error": float(max_abs_error),
        "failures": failures,
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


def _broadcast_case(backend, backend_name, device, rank, world_size):
    root = 0
    expected = np.arange(6, dtype=np.float64).reshape(2, 3)
    local = expected if int(rank) == root else np.zeros_like(expected)
    value = backend.to_backend(local)
    result = backend.broadcast(value, root=root)
    _sync_backend(backend, result)
    actual = _to_numpy(backend, result)
    return {
        "backend": backend_name,
        "device": device,
        "device_spec": str(backend.current_device()),
        "operation": "broadcast_tensor",
        "status": "passed",
        "rank": int(rank),
        "world_size": int(world_size),
        "root": int(root),
        "global_shape": list(expected.shape),
        "local_shape": list(expected.shape),
        "collectives": ["broadcast"],
        "plan_comm_bytes": int(expected.nbytes),
        "max_abs_error": float(np.max(np.abs(actual - expected))),
    }


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
            _broadcast_case(backend, backend_name, backend_device, rank, world_size),
            _row_sharded_case(backend, backend_name, backend_device, mesh, left_np, right_np, rank, world_size),
            _contracted_sharded_case(backend, backend_name, backend_device, mesh, left_np, right_np, rank, world_size),
            _redistribute_output_case(backend, backend_name, backend_device, mesh, left_np, right_np, rank, world_size),
        ]
    finally:
        _close_distributed_runtime(backend)


def run_distributed_speed_smoke(
    backend_name,
    *,
    device="cuda",
    rows=1024,
    shared_dim=1024,
    cols=1024,
    rank=None,
    world_size=None,
    repeat=3,
    warmup=1,
    trials=3,
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
    if world_size == 1:
        return [
            {
                "backend": backend_name,
                "device": device,
                "operation": "row_sharded_speed_benchmark",
                "status": "skipped",
                "rank": int(rank),
                "world_size": int(world_size),
                "error": "distributed speed benchmark requires world_size > 1",
            }
        ]
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
        distributed_plan = _distributed_plan(plan)
        output_sharding = distributed_plan.output_sharding if distributed_plan is not None else None
        if output_sharding is None:
            raise RuntimeError("row-sharded speed benchmark requires an output sharding plan")
        output_slices = output_sharding.local_slices[int(rank)]

        def distributed_fn():
            return backend.distributed_contract(spec, plan=plan)

        def slicing_fn():
            full = backend.matmul(left, right)
            return full[tuple(output_slices)]

        distributed_result, distributed_wall_s, distributed_samples = _time_call(
            backend,
            repeat,
            distributed_fn,
            warmup=warmup,
            trials=trials,
        )
        slicing_result, slicing_wall_s, slicing_samples = _time_call(
            backend,
            repeat,
            slicing_fn,
            warmup=warmup,
            trials=trials,
        )
        del slicing_result
        expected_local = left_np[output_slices[0], :] @ right_np[:, output_slices[1]]
        actual_local = _to_numpy(backend, distributed_result.local_array)
        error = float(np.max(np.abs(actual_local - expected_local)))
        return [
            {
                "backend": backend_name,
                "device": backend_device,
                "device_spec": str(backend.current_device()),
                "operation": "row_sharded_speed_benchmark",
                "status": "passed",
                "rank": int(rank),
                "world_size": int(world_size),
                "shape": [int(rows), int(shared_dim), int(cols)],
                "local_shape": [int(dim) for dim in distributed_result.local_shape],
                "output_sharded_modes": [str(mode) for mode in distributed_result.sharding.sharded_modes],
                "collectives": _collectives(plan),
                "repeat": int(repeat),
                "warmup": int(warmup),
                "trials": int(trials),
                "distributed_wall_s": float(distributed_wall_s),
                "distributed_wall_s_samples": distributed_samples,
                "slicing_wall_s": float(slicing_wall_s),
                "slicing_wall_s_samples": slicing_samples,
                "speedup_vs_slicing": float(slicing_wall_s / distributed_wall_s) if distributed_wall_s > 0 else None,
                "max_abs_error": error,
            }
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
    parser.add_argument("--speed-rows", type=int, default=1024)
    parser.add_argument("--speed-shared-dim", type=int, default=1024)
    parser.add_argument("--speed-cols", type=int, default=1024)
    parser.add_argument("--speed-repeat", type=int, default=3)
    parser.add_argument("--speed-warmup", type=int, default=1)
    parser.add_argument("--speed-trials", type=int, default=3)
    parser.add_argument("--output", default=None, help="Optional JSONL output path.")
    parser.add_argument(
        "--profile-output",
        default=None,
        help="Optional profiling JSONL path. Multi-rank runs write rank-suffixed files unless {rank} is present.",
    )
    parser.add_argument(
        "--distributed-gate",
        action="store_true",
        help="Fail if distributed smoke records do not cover the expected collective cases.",
    )
    parser.add_argument(
        "--profile-gate",
        action="store_true",
        help="Fail if profiling JSONL does not contain distributed contraction communication events.",
    )
    parser.add_argument(
        "--speed-benchmark",
        action="store_true",
        help="Benchmark row-sharded distributed matmul against full dense matmul plus local slicing.",
    )
    parser.add_argument(
        "--speed-gate",
        action="store_true",
        help="Fail if the distributed speed benchmark does not beat the slicing baseline.",
    )
    parser.add_argument("--min-speedup", type=float, default=1.0)
    parser.add_argument("--require-world-size", type=int, default=None)
    parser.add_argument("--max-abs-error", type=float, default=1e-9)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rank, world_size = _env_rank_world()
    profile_output = None
    old_log_level = None
    if args.profile_output is not None:
        from renormalizer.utils import profiling
        from renormalizer.utils.log import PROFILING, init_log, package_logger

        old_log_level = package_logger.level
        profile_output = _ranked_profile_path(args.profile_output, rank, world_size)
        profile_output.parent.mkdir(parents=True, exist_ok=True)
        init_log(PROFILING)
        disable_stream_output()
        profiling.register_event_output(profile_output)
    all_records = []
    try:
        for backend_name in _parse_backend_names(args.backends):
            records = run_distributed_smoke(
                backend_name,
                device=args.device,
                rows=args.rows,
                shared_dim=args.shared_dim,
                cols=args.cols,
            )
            all_records.extend(records)
            if args.speed_benchmark:
                all_records.extend(
                    run_distributed_speed_smoke(
                        backend_name,
                        device=args.device,
                        rows=args.speed_rows,
                        shared_dim=args.speed_shared_dim,
                        cols=args.speed_cols,
                        repeat=args.speed_repeat,
                        warmup=args.speed_warmup,
                        trials=args.speed_trials,
                    )
                )
    finally:
        if args.profile_output is not None:
            from renormalizer.utils import profiling
            from renormalizer.utils.log import init_log

            profiling.close_event_output()
            profiling.flush_summaries()
            if old_log_level is not None:
                init_log(old_log_level)
    gate_summary = None
    if args.distributed_gate:
        gate_summary = evaluate_distributed_smoke_gate(
            all_records,
            require_world_size=args.require_world_size,
            max_abs_error=args.max_abs_error,
        )
        all_records.append(gate_summary)
    profile_gate_summary = None
    if args.profile_gate:
        profile_gate_summary = evaluate_distributed_profile_gate(
            read_jsonl(profile_output) if profile_output is not None else [],
            require_world_size=args.require_world_size,
        )
        if profile_output is not None:
            profile_gate_summary["profile_output"] = os.fspath(profile_output)
        all_records.append(profile_gate_summary)
    speed_gate_summary = None
    if args.speed_gate:
        speed_gate_summary = evaluate_distributed_speed_gate(
            all_records,
            min_speedup=args.min_speedup,
            require_world_size=args.require_world_size,
            max_abs_error=args.max_abs_error,
        )
        all_records.append(speed_gate_summary)
    if args.output is not None:
        write_jsonl(all_records, args.output)
    print(json.dumps({"records": all_records}, sort_keys=True))
    if any(record.get("status") == "error" for record in all_records):
        return 1
    if gate_summary is not None and gate_summary["status"] != "passed":
        return 2
    if profile_gate_summary is not None and profile_gate_summary["status"] != "passed":
        return 2
    if speed_gate_summary is not None and speed_gate_summary["status"] != "passed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
