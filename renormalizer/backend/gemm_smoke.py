# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from renormalizer.backend.config import BackendConfig
from renormalizer.backend.factory import create_backend, is_backend_available, normalize_backend_name
from renormalizer.backend.gemm import GemmTask, grouped_gemm_stats, run_gemm_task


DEFAULT_BACKENDS = ("numpy", "cupy", "torch", "jax")


def parse_backend_names(value):
    names = []
    for item in str(value).split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item == "all":
            candidates = DEFAULT_BACKENDS
        else:
            candidates = (item,)
        for candidate in candidates:
            normalized = normalize_backend_name(candidate)
            if normalized not in names:
                names.append(normalized)
    return names


def detect_gpu_count():
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return 0
    if completed.returncode != 0:
        return 0
    return sum(1 for line in completed.stdout.splitlines() if line.strip().startswith("GPU "))


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


def _record_base(backend, backend_name, device, operation, gpu_count, repeat, warmup, trials):
    return {
        "backend": backend_name,
        "device": device,
        "device_spec": str(backend.current_device()),
        "operation": operation,
        "status": "passed",
        "gpu_count": int(gpu_count),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "repeat": int(repeat),
        "warmup": int(warmup),
        "trials": int(trials),
    }


def _error_record(backend_name, device, operation, gpu_count, status, message):
    return {
        "backend": backend_name,
        "device": device,
        "operation": operation,
        "status": status,
        "gpu_count": int(gpu_count),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "error": str(message),
    }


def _random(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float64)


def _max_abs_error(actual, expected):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def _matmul_smoke(backend, backend_name, device, gpu_count, repeat, medium_dim, warmup=0, trials=1):
    a_np = _random((medium_dim, medium_dim), 1)
    b_np = _random((medium_dim, medium_dim), 2)
    a = backend.to_backend(a_np)
    b = backend.to_backend(b_np)
    result, wall_s, samples = _time_call(backend, repeat, lambda: backend.matmul(a, b), warmup=warmup, trials=trials)
    expected = a_np @ b_np
    record = _record_base(backend, backend_name, device, "matmul", gpu_count, repeat, warmup, trials)
    record.update({
        "shape_a": tuple(a_np.shape),
        "shape_b": tuple(b_np.shape),
        "wall_s": float(wall_s),
        "wall_s_samples": samples,
        "max_abs_error": _max_abs_error(_to_numpy(backend, result), expected),
    })
    return record


def _batched_matmul_smoke(backend, backend_name, device, gpu_count, repeat, batch, small_dim, warmup=0, trials=1):
    a_np = _random((batch, small_dim, small_dim), 3)
    b_np = _random((batch, small_dim, small_dim), 4)
    a = backend.to_backend(a_np)
    b = backend.to_backend(b_np)
    result, wall_s, samples = _time_call(backend, repeat, lambda: backend.batched_matmul(a, b), warmup=warmup, trials=trials)
    expected = np.matmul(a_np, b_np)
    record = _record_base(backend, backend_name, device, "batched_matmul", gpu_count, repeat, warmup, trials)
    record.update({
        "shape_a": tuple(a_np.shape),
        "shape_b": tuple(b_np.shape),
        "wall_s": float(wall_s),
        "wall_s_samples": samples,
        "max_abs_error": _max_abs_error(_to_numpy(backend, result), expected),
    })
    return record


def _grouped_gemm_smoke(
    backend,
    backend_name,
    device,
    gpu_count,
    repeat,
    batch,
    small_dim,
    pack_threshold,
    warmup=0,
    trials=1,
):
    tasks_np = []
    for index in range(batch):
        tasks_np.append((
            _random((small_dim, small_dim), 100 + index),
            _random((small_dim, small_dim), 200 + index),
        ))
    tasks = [
        GemmTask(backend.to_backend(a_np), backend.to_backend(b_np), tag=index)
        for index, (a_np, b_np) in enumerate(tasks_np)
    ]
    flop_copy_ratio = 0 if backend.supports_grouped_gemm else 10
    stats = grouped_gemm_stats(
        tasks,
        xp=backend.array_namespace,
        pack_threshold=pack_threshold,
        flop_copy_ratio=flop_copy_ratio,
    )
    _, loop_wall_s, loop_samples = _time_call(
        backend,
        repeat,
        lambda: [run_gemm_task(task, xp=backend.array_namespace) for task in tasks],
        warmup=warmup,
        trials=trials,
    )
    result, wall_s, grouped_samples = _time_call(
        backend,
        repeat,
        lambda: backend.grouped_gemm(tasks, pack_threshold=pack_threshold),
        warmup=warmup,
        trials=trials,
    )
    expected = [a_np @ b_np for a_np, b_np in tasks_np]
    actual = [_to_numpy(backend, item) for item in result]
    max_error = max(_max_abs_error(item, ref) for item, ref in zip(actual, expected))
    record = _record_base(backend, backend_name, device, "grouped_gemm", gpu_count, repeat, warmup, trials)
    record.update({
        "task_count": int(len(tasks)),
        "shape": (small_dim, small_dim, small_dim),
        "pack_threshold": int(pack_threshold),
        "num_shape_buckets": stats.shape_bucket_count,
        "num_batched_gemm": stats.batched_bucket_count,
        "num_gemm": stats.loop_task_count,
        "num_grouped_tasks": stats.task_count,
        "bucket_task_counts": stats.bucket_task_counts,
        "copy_bytes": stats.copy_bytes,
        "flops": stats.flops,
        "read_bytes": stats.read_bytes,
        "write_bytes": stats.write_bytes,
        "wall_s": float(wall_s),
        "wall_s_samples": grouped_samples,
        "grouped_wall_s": float(wall_s),
        "grouped_wall_s_samples": grouped_samples,
        "loop_wall_s": float(loop_wall_s),
        "loop_wall_s_samples": loop_samples,
        "speedup_vs_loop": float(loop_wall_s / wall_s) if wall_s > 0.0 else None,
        "max_abs_error": float(max_error),
    })
    return record


def _grouped_records(records):
    return [
        record for record in records
        if record.get("operation") == "grouped_gemm"
    ]


def _gate_failure(record, reason):
    failure = {
        "backend": record.get("backend"),
        "device": record.get("device"),
        "reason": reason,
    }
    if "speedup_vs_loop" in record:
        failure["speedup_vs_loop"] = record.get("speedup_vs_loop")
    return failure


def evaluate_grouped_gemm_gate(records, *, min_speedup=1.0, require_backends=()):
    """Evaluate whether grouped GEMM benchmark records satisfy the Phase 5 gate."""

    grouped_records = _grouped_records(records)
    required = tuple(parse_backend_names(",".join(require_backends))) if require_backends else ()
    checked_records = [
        record for record in grouped_records
        if not required or record.get("backend") in required
    ]
    failures = []

    for backend_name in required:
        if not any(record.get("backend") == backend_name for record in grouped_records):
            failures.append({
                "backend": backend_name,
                "device": None,
                "reason": "missing grouped_gemm record",
            })

    for record in checked_records:
        if record.get("status") != "passed":
            failures.append(_gate_failure(record, "grouped_gemm benchmark did not pass"))
            continue
        if int(record.get("num_grouped_tasks") or 0) <= 0:
            failures.append(_gate_failure(record, "no grouped GEMM tasks"))
            continue
        if int(record.get("num_batched_gemm") or 0) <= 0:
            failures.append(_gate_failure(record, "no batched GEMM bucket"))
            continue
        speedup = record.get("speedup_vs_loop")
        if speedup is None or float(speedup) < float(min_speedup):
            failures.append(_gate_failure(record, "speedup below threshold"))

    return {
        "operation": "grouped_gemm_gate",
        "status": "failed" if failures else "passed",
        "min_speedup": float(min_speedup),
        "checked_count": int(len(checked_records)),
        "required_backends": list(required),
        "failures": failures,
    }


def run_backend_smoke(
    backend_name,
    *,
    device,
    batch,
    small_dim,
    medium_dim,
    repeat,
    pack_threshold,
    warmup=0,
    trials=1,
):
    backend_name = normalize_backend_name(backend_name)
    gpu_count = detect_gpu_count()
    if not is_backend_available(backend_name):
        return [_error_record(backend_name, device, "backend_import", gpu_count, "skipped", "backend is not available")]
    try:
        backend = create_backend(backend_name, config=BackendConfig(device=device))
    except Exception as exc:
        return [_error_record(backend_name, device, "backend_create", gpu_count, "error", exc)]

    records = []
    for operation, fn in (
        ("matmul", lambda: _matmul_smoke(backend, backend_name, device, gpu_count, repeat, medium_dim, warmup, trials)),
        (
            "batched_matmul",
            lambda: _batched_matmul_smoke(
                backend,
                backend_name,
                device,
                gpu_count,
                repeat,
                batch,
                small_dim,
                warmup,
                trials,
            ),
        ),
        (
            "grouped_gemm",
            lambda: _grouped_gemm_smoke(
                backend,
                backend_name,
                device,
                gpu_count,
                repeat,
                batch,
                small_dim,
                pack_threshold,
                warmup,
                trials,
            ),
        ),
    ):
        try:
            records.append(fn())
        except Exception as exc:
            records.append(_error_record(backend_name, device, operation, gpu_count, "error", exc))
    return records


def write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description="Smoke-test backend GEMM primitives.")
    parser.add_argument("--backends", default="all", help="Comma-separated backend list, or 'all'.")
    parser.add_argument("--device", default="cpu", help="Backend device, e.g. cpu or gpu.")
    parser.add_argument("--gpu-index", type=int, default=None, help="Set CUDA_VISIBLE_DEVICES before backend creation.")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--small-dim", type=int, default=32)
    parser.add_argument("--medium-dim", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--pack-threshold", type=int, default=4)
    parser.add_argument("--output", default=None, help="Optional JSONL output path.")
    parser.add_argument(
        "--grouped-speedup-gate",
        action="store_true",
        help="Fail if grouped_gemm records do not beat the Python loop benchmark.",
    )
    parser.add_argument("--min-grouped-speedup", type=float, default=1.0)
    parser.add_argument(
        "--require-grouped-backends",
        default="",
        help="Comma-separated backend names that must have grouped_gemm records when the gate is enabled.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.gpu_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    all_records = []
    for backend_name in parse_backend_names(args.backends):
        records = run_backend_smoke(
            backend_name,
            device=args.device,
            batch=args.batch,
            small_dim=args.small_dim,
            medium_dim=args.medium_dim,
            repeat=args.repeat,
            pack_threshold=args.pack_threshold,
            warmup=args.warmup,
            trials=args.trials,
        )
        all_records.extend(records)

    gate_summary = None
    if args.grouped_speedup_gate:
        required = tuple(parse_backend_names(args.require_grouped_backends)) if args.require_grouped_backends else ()
        gate_summary = evaluate_grouped_gemm_gate(
            all_records,
            min_speedup=args.min_grouped_speedup,
            require_backends=required,
        )
        all_records.append(gate_summary)

    if args.output:
        write_jsonl(all_records, args.output)
    for record in all_records:
        print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    if any(record.get("status") == "error" for record in all_records):
        return 1
    if gate_summary is not None and gate_summary["status"] != "passed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
