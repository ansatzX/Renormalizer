# -*- coding: utf-8 -*-

"""Benchmark real Renormalizer examples across CPU thread counts and CuPy."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import html
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Iterable


THREAD_ENV_NAMES = (
    "RENO_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


@dataclass(frozen=True)
class ExampleCase:
    name: str
    script: Path
    args: tuple[str, ...] = ()
    assets: tuple[str, ...] = ()
    expected_steps: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class BenchmarkJob:
    case: ExampleCase
    suite: str
    backend: str
    device: str
    threads: int
    run_dir: Path
    repo_root: Path
    profile_mode: str = "off"

    @property
    def stdout_path(self) -> Path:
        return self.run_dir / "stdout.log"

    @property
    def stderr_path(self) -> Path:
        return self.run_dir / "stderr.log"

    @property
    def profile_events_path(self) -> Path:
        return self.run_dir / "profile-events.jsonl"

    @property
    def resource_samples_path(self) -> Path:
        return self.run_dir / "resource-samples.jsonl"

    @property
    def result_path(self) -> Path:
        return self.run_dir / "result.json"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_example_cases() -> list[ExampleCase]:
    """Return the example suite in the same order as ``example/run.sh``."""
    return [
        ExampleCase(
            "fmo",
            Path("example/fmo.py"),
            assets=("fmo_sdf.json",),
            expected_steps=251,
            notes="Charge diffusion FMO TDVP-PS, evolve_time/evolve_dt = 40000/160.",
        ),
        ExampleCase(
            "sbm",
            Path("example/sbm.py"),
            expected_steps=201,
            notes="Spin-boson TDVP, evolve_time/evolve_dt = 20/0.1.",
        ),
        ExampleCase(
            "h2o_qc",
            Path("example/h2o_qc.py"),
            assets=("h2o_fcidump.txt",),
            notes="Ground-state DMRG quantum chemistry example.",
        ),
        ExampleCase(
            "dynamics_std",
            Path("example/dynamics.py"),
            args=("std.yaml",),
            assets=("std.yaml",),
            expected_steps=301,
            notes="Transport dynamics from std.yaml, evolve_time/evolve_dt = 30000/100.",
        ),
        ExampleCase(
            "transport_kubo_std",
            Path("example/transport_kubo.py"),
            args=("std.yaml",),
            assets=("std.yaml",),
            expected_steps=301,
            notes="Transport Kubo dynamics from std.yaml.",
        ),
        ExampleCase(
            "ttns_junction_zt",
            Path("example/ttns/junction_zt.py"),
            expected_steps=100,
            notes="Zero-temperature TTNS junction example.",
        ),
        ExampleCase(
            "ttns_junction_ft",
            Path("example/ttns/junction_ft.py"),
            args=("32", "1", "100"),
            expected_steps=200,
            notes="Finite-temperature TTNS junction example.",
        ),
        ExampleCase(
            "ttns_sbm_zt",
            Path("example/ttns/sbm_zt.py"),
            args=("050", "001", "050"),
            expected_steps=200,
            notes="Zero-temperature TTNS spin-boson example.",
        ),
        ExampleCase(
            "ttns_sbm_ft",
            Path("example/ttns/sbm_ft.py"),
            expected_steps=400,
            notes="Finite-temperature TTNS spin-boson example.",
        ),
        ExampleCase(
            "ssh",
            Path("example/ssh.py"),
            notes="Optical SSH ground-state example.",
        ),
    ]


def extra_probe_cases() -> list[ExampleCase]:
    """Return opt-in benchmark probes that are not part of the default suite."""
    return [
        ExampleCase(
            "sbm_probe",
            Path("renormalizer/backend/probes/sbm_probe.py"),
            expected_steps=6,
            notes="Short SBM profiling probe, evolve_time/evolve_dt = 0.5/0.1.",
        ),
        ExampleCase(
            "holstein_multistate_probe",
            Path("renormalizer/backend/probes/holstein_multistate_probe.py"),
            notes="Multi-state Holstein DMRG probe for batched RHS HMM path coverage.",
        ),
    ]


def _sanitize_path_part(value: str) -> str:
    return (
        str(value)
        .replace(":", "-")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "-")
    )


def _normalize_threads(threads: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in threads)
    if not values:
        raise ValueError("at least one thread count is required")
    if any(value <= 0 for value in values):
        raise ValueError("thread counts must be positive")
    return values


def make_cpu_jobs(
    cases: Iterable[ExampleCase],
    output_dir,
    *,
    threads=(1, 48),
    backend="numpy",
    device="cpu",
    root: Path | None = None,
    profile_mode="off",
) -> list[BenchmarkJob]:
    output_dir = Path(output_dir).resolve()
    root = repo_root() if root is None else Path(root)
    jobs = []
    for case in cases:
        for nthreads in _normalize_threads(threads):
            run_dir = output_dir / "cpu" / backend / f"threads-{nthreads}" / case.name
            jobs.append(BenchmarkJob(case, "cpu", backend, device, int(nthreads), run_dir, root, str(profile_mode)))
    return jobs


def make_cupy_jobs(
    cases: Iterable[ExampleCase],
    output_dir,
    *,
    backend="cupy",
    device="cuda:0",
    threads=48,
    root: Path | None = None,
    profile_mode="off",
) -> list[BenchmarkJob]:
    output_dir = Path(output_dir).resolve()
    root = repo_root() if root is None else Path(root)
    device_part = _sanitize_path_part(device)
    return [
        BenchmarkJob(
            case,
            "cupy",
            backend,
            device,
            int(threads),
            output_dir / "cupy" / backend / device_part / f"threads-{int(threads)}" / case.name,
            root,
            str(profile_mode),
        )
        for case in cases
    ]


def build_thread_env(threads: int, *, base_env=None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    value = str(int(threads))
    for name in THREAD_ENV_NAMES:
        env[name] = value
    return env


def _env_with_repo_path(env: dict[str, str], root: Path) -> dict[str, str]:
    env = dict(env)
    current = env.get("PYTHONPATH")
    root_text = str(root)
    env["PYTHONPATH"] = root_text if not current else root_text + os.pathsep + current
    return env


def build_job_env(job: BenchmarkJob, *, base_env=None) -> dict[str, str]:
    env = _env_with_repo_path(build_thread_env(job.threads, base_env=base_env), job.repo_root)
    if job.profile_mode == "full":
        env["RENO_LOG_LEVEL"] = "PROFILING"
    return env


def build_child_command(job: BenchmarkJob) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "renormalizer.backend.example_runner",
        "--backend",
        str(job.backend),
        "--device",
        str(job.device),
        "--cwd",
        str(job.run_dir),
        "--script",
        str(job.repo_root / job.case.script),
    ]
    if job.profile_mode == "full":
        command.extend(["--profile-events", str(job.profile_events_path)])
    command.append("--")
    command.extend(job.case.args)
    return command


def parse_proc_status(text: str) -> dict[str, int]:
    key_map = {
        "VmRSS": "rss_kb",
        "VmHWM": "hwm_kb",
        "Threads": "threads",
        "voluntary_ctxt_switches": "voluntary_ctxt_switches",
        "nonvoluntary_ctxt_switches": "nonvoluntary_ctxt_switches",
    }
    result: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        raw_key, raw_value = line.split(":", 1)
        key = key_map.get(raw_key.strip())
        if key is None:
            continue
        parts = raw_value.strip().split()
        if parts:
            result[key] = int(parts[0])
    return result


def sample_process_status(pid: int) -> dict[str, int] | None:
    status_path = Path("/proc") / str(int(pid)) / "status"
    try:
        text = status_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return parse_proc_status(text)


def collect_profile_summary(profile_path) -> dict:
    profile_path = Path(profile_path)
    counts: Counter[str] = Counter()
    wall_s_by_event: defaultdict[str, float] = defaultdict(float)
    tensor_kernels: Counter[str] = Counter()
    tensordot_shapes: Counter[str] = Counter()
    bottleneck_hints: Counter[str] = Counter()
    arithmetic_intensity_bins: Counter[str] = Counter()
    contraction_lowerings: Counter[str] = Counter()
    grouped_pack_strategies: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    small_gemm_count = 0
    skinny_gemm_count = 0
    low_ai_count = 0
    tensordot_total = 0
    total = 0
    if not profile_path.exists():
        return {
            "total_events": 0,
            "event_counts": {},
            "wall_s_by_event": {},
            "profile_analysis": {},
        }
    with profile_path.open("r", encoding="utf-8", errors="replace") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = str(payload.get("event") or "unknown")
            counts[event] += 1
            total += 1
            try:
                wall_s_by_event[event] += float(payload.get("wall_s") or 0.0)
            except (TypeError, ValueError):
                pass
            if event == "tensordot":
                tensordot_total += 1
                kernel = payload.get("algorithmic_kernel") or payload.get("kernel")
                if kernel:
                    tensor_kernels[str(kernel)] += 1
                m = payload.get("m")
                n = payload.get("n")
                k = payload.get("k")
                if all(isinstance(value, int) for value in (m, n, k)):
                    tensordot_shapes[f"{m},{n},{k}"] += 1
                if payload.get("arithmetic_intensity_bin"):
                    arithmetic_intensity_bins[str(payload["arithmetic_intensity_bin"])] += 1
                if payload.get("arithmetic_intensity_bin") == "low":
                    low_ai_count += 1
                hints = payload.get("bottleneck_hints") or ()
                for hint in hints:
                    bottleneck_hints[str(hint)] += 1
                if "small_gemm" in hints:
                    small_gemm_count += 1
                if "skinny_gemm" in hints:
                    skinny_gemm_count += 1
            lowering = payload.get("lowering")
            if lowering:
                contraction_lowerings[str(lowering)] += 1
            pack_strategy = payload.get("pack_strategy")
            if pack_strategy:
                grouped_pack_strategies[str(pack_strategy)] += 1
            fallback_reason = payload.get("fallback_reason")
            if fallback_reason:
                fallback_reasons[str(fallback_reason)] += 1
    profile_analysis = {
        "tensordot_total": int(tensordot_total),
        "tensordot_kernel_counts": dict(tensor_kernels),
        "small_gemm_count": int(small_gemm_count),
        "small_gemm_pct": (float(small_gemm_count) / float(tensordot_total)) if tensordot_total else None,
        "skinny_gemm_count": int(skinny_gemm_count),
        "skinny_gemm_pct": (float(skinny_gemm_count) / float(tensordot_total)) if tensordot_total else None,
        "low_arithmetic_intensity_count": int(low_ai_count),
        "low_arithmetic_intensity_pct": (float(low_ai_count) / float(tensordot_total)) if tensordot_total else None,
        "arithmetic_intensity_bins": dict(arithmetic_intensity_bins),
        "bottleneck_hints": dict(bottleneck_hints),
        "top_tensordot_shapes": [
            {"shape_mnk": shape, "count": int(count)}
            for shape, count in tensordot_shapes.most_common(12)
        ],
        "contraction_lowerings": dict(contraction_lowerings),
        "grouped_gemm_execute_count": int(counts.get("grouped_gemm_execute", 0)),
        "grouped_pack_strategies": dict(grouped_pack_strategies),
        "fallback_reasons": dict(fallback_reasons),
    }
    return {
        "total_events": int(total),
        "event_counts": dict(counts),
        "wall_s_by_event": dict(wall_s_by_event),
        "profile_analysis": profile_analysis,
    }


def extract_tensordot_mnk_shapes(profile_path) -> list[tuple[int, int, int]]:
    """Extract concrete ``(m, n, k)`` tensordot shapes from a profiling JSONL file."""
    profile_path = Path(profile_path)
    shapes: list[tuple[int, int, int]] = []
    if not profile_path.exists():
        return shapes
    with profile_path.open("r", encoding="utf-8", errors="replace") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("event") != "tensordot":
                continue
            m = payload.get("m")
            n = payload.get("n")
            k = payload.get("k")
            if all(isinstance(value, int) and value > 0 for value in (m, n, k)):
                shapes.append((int(m), int(n), int(k)))
    return shapes


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _grouped_profile_dict(profile) -> dict:
    if hasattr(profile, "to_dict"):
        return profile.to_dict()
    return dict(profile)


def run_grouped_gemm_diagnostic_from_shapes(
    shapes: Iterable[tuple[int, int, int]],
    *,
    repeats=3,
    seed=123,
    pack_threshold=4,
) -> dict:
    """Replay recorded GEMM shapes through loop, grouped, and prepacked policies."""
    import numpy as np
    from renormalizer.backend.gemm import (
        GemmTask,
        execute_prepacked_grouped_gemm,
        grouped_gemm_bucketed_profiled,
        prepack_grouped_gemm,
        run_gemm_task,
    )

    normalized_shapes = [
        (int(m), int(n), int(k))
        for m, n, k in shapes
        if int(m) > 0 and int(n) > 0 and int(k) > 0
    ]
    rng = np.random.default_rng(int(seed))
    tasks = [
        GemmTask(
            rng.standard_normal((m, k)),
            rng.standard_normal((k, n)),
        )
        for m, n, k in normalized_shapes
    ]

    def bench_loop():
        started = time.perf_counter()
        for task in tasks:
            run_gemm_task(task, xp=np)
        return time.perf_counter() - started, {
            "pack_strategy": "none",
            "prepacked": False,
            "kernel_calls": len(tasks),
            "batched_kernel_calls": 0,
            "loop_kernel_calls": len(tasks),
        }

    def bench_default_grouped():
        started = time.perf_counter()
        _result, profile = grouped_gemm_bucketed_profiled(
            tasks,
            xp=np,
            pack_threshold=int(pack_threshold),
            allow_batched=True,
        )
        return time.perf_counter() - started, _grouped_profile_dict(profile)

    def bench_forced_stack_grouped():
        started = time.perf_counter()
        _result, profile = grouped_gemm_bucketed_profiled(
            tasks,
            xp=np,
            pack_threshold=int(pack_threshold),
            flop_copy_ratio=0,
            allow_batched=True,
        )
        return time.perf_counter() - started, _grouped_profile_dict(profile)

    prepack_started = time.perf_counter()
    prepacked_plan = prepack_grouped_gemm(
        tasks,
        xp=np,
        pack_threshold=int(pack_threshold),
        flop_copy_ratio=0,
        allow_batched=True,
    )
    prepack_s = time.perf_counter() - prepack_started

    def bench_prepacked_grouped():
        started = time.perf_counter()
        _result, profile = execute_prepacked_grouped_gemm(prepacked_plan, xp=np)
        profile = dict(profile)
        profile["prepack_s"] = float(prepack_s)
        return time.perf_counter() - started, profile

    timings = {}
    for name, runner in (
        ("loop", bench_loop),
        ("default_grouped", bench_default_grouped),
        ("forced_stack_grouped", bench_forced_stack_grouped),
        ("forced_prepacked_grouped", bench_prepacked_grouped),
    ):
        durations = []
        last_profile = {}
        for _ in range(int(repeats)):
            duration, last_profile = runner()
            durations.append(float(duration))
        timings[name] = {
            "durations_s": durations,
            "median_s": _median(durations),
            "profile": last_profile,
        }

    return {
        "task_count": len(tasks),
        "unique_shape_count": len(set(normalized_shapes)),
        "pack_threshold": int(pack_threshold),
        "repeats": int(repeats),
        "seed": int(seed),
        "timings": timings,
    }


def _dependency_versions(names=("numpy", "scipy", "h5py", "opt_einsum", "qutip", "cupy")) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_commit(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return "Unknown"
    if completed.returncode != 0:
        return "Unknown"
    return completed.stdout.strip()


def collect_environment_metadata(*, root: Path | None = None) -> dict:
    root = repo_root() if root is None else Path(root)
    return {
        "git_commit": _git_commit(root),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "dependencies": _dependency_versions(),
        "thread_env": {
            name: os.environ.get(name)
            for name in THREAD_ENV_NAMES
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _copy_case_assets(job: BenchmarkJob) -> None:
    job.run_dir.mkdir(parents=True, exist_ok=True)
    example_dir = job.repo_root / job.case.script.parent
    for asset in job.case.assets:
        src = example_dir / asset
        dst = job.run_dir / asset
        if src.resolve() == dst.resolve():
            continue
        if not src.exists():
            raise FileNotFoundError(f"missing benchmark asset {src}")
        shutil.copy2(src, dst)


def _max_sample_value(samples: list[dict], key: str) -> int | None:
    values = [
        int(sample[key])
        for sample in samples
        if sample.get(key) is not None
    ]
    return max(values) if values else None


_STEP_COMPLETE_RE = re.compile(r"step\s+(\d+)\s+complete,\s+time cost\s+([0-9:.]+)")


def _parse_duration_s(value: str) -> float | None:
    try:
        parts = [float(part) for part in str(value).strip().rstrip(".").split(":")]
    except ValueError:
        return None
    if not parts:
        return None
    total = 0.0
    for part in parts:
        total = total * 60.0 + part
    return total


def summarize_step_progress(stderr_path, *, expected_steps=None, wall_s=None) -> dict:
    stderr_path = Path(stderr_path)
    steps: list[tuple[int, float | None]] = []
    if stderr_path.exists():
        text = stderr_path.read_text(encoding="utf-8", errors="replace")
        for match in _STEP_COMPLETE_RE.finditer(text):
            steps.append((int(match.group(1)), _parse_duration_s(match.group(2))))

    completed = len(steps)
    last_step = steps[-1][0] if steps else None
    durations = [duration for _, duration in steps if duration is not None]
    progress_fraction = None
    if expected_steps and last_step is not None:
        progress_fraction = float(last_step) / float(expected_steps)
    step_throughput = None
    if wall_s and float(wall_s) > 0.0 and completed > 0:
        step_throughput = float(completed) / float(wall_s)
    return {
        "completed_steps": int(completed),
        "last_step": last_step,
        "expected_steps": expected_steps,
        "progress_fraction": progress_fraction,
        "mean_logged_step_s": (sum(durations) / len(durations)) if durations else None,
        "step_throughput_per_s": step_throughput,
    }


def run_job(job: BenchmarkJob, *, sample_interval=60.0, timeout=None, timeout_status="timeout", metadata=None) -> dict:
    _copy_case_assets(job)
    command = build_child_command(job)
    env = build_job_env(job)
    metadata = collect_environment_metadata(root=job.repo_root) if metadata is None else dict(metadata)

    started = time.perf_counter()
    samples: list[dict] = []
    timed_out = False
    with job.stdout_path.open("w", encoding="utf-8") as stdout, job.stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=str(job.repo_root),
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        with job.resource_samples_path.open("w", encoding="utf-8") as resources:
            while True:
                now = time.perf_counter()
                status = sample_process_status(process.pid)
                if status is not None:
                    status = {
                        "sample_time_s": float(now - started),
                        **status,
                    }
                    samples.append(status)
                    resources.write(json.dumps(status, sort_keys=True) + "\n")
                    resources.flush()

                if timeout is not None and now - started > float(timeout):
                    timed_out = True
                    process.terminate()
                    try:
                        process.wait(timeout=10.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break

                try:
                    process.wait(timeout=float(sample_interval))
                    break
                except subprocess.TimeoutExpired:
                    continue

    wall_s = float(time.perf_counter() - started)
    returncode = int(process.returncode if process.returncode is not None else -1)
    progress = summarize_step_progress(job.stderr_path, expected_steps=job.case.expected_steps, wall_s=wall_s)
    if returncode == 0 and not timed_out:
        status = "passed"
    elif timed_out and timeout_status == "sampled":
        status = "sampled"
    elif timed_out:
        status = "timeout"
    else:
        status = "failed"
    record = {
        "case": job.case.name,
        "suite": job.suite,
        "backend": job.backend,
        "device": job.device,
        "threads": int(job.threads),
        "status": status,
        "returncode": returncode,
        "timed_out": bool(timed_out),
        "wall_s": wall_s,
        "expected_steps": job.case.expected_steps,
        "command": command,
        "thread_env": {
            name: env.get(name)
            for name in THREAD_ENV_NAMES
        },
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "run_dir": str(job.run_dir),
        "stdout_path": str(job.stdout_path),
        "stderr_path": str(job.stderr_path),
        "profile_events_path": str(job.profile_events_path),
        "profile_mode": str(job.profile_mode),
        "resource_samples_path": str(job.resource_samples_path),
        "result_path": str(job.result_path),
        "max_rss_kb": _max_sample_value(samples, "rss_kb"),
        "max_hwm_kb": _max_sample_value(samples, "hwm_kb"),
        "max_threads": _max_sample_value(samples, "threads"),
        "metadata": metadata,
        "progress": progress,
        "profile_summary": collect_profile_summary(job.profile_events_path),
    }
    job.result_path.write_text(
        json.dumps(record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return record


def compare_cpu_results(records: Iterable[dict]) -> list[dict]:
    by_case: dict[str, dict[int, dict]] = defaultdict(dict)
    order: list[str] = []
    for record in records:
        if record.get("suite") != "cpu":
            continue
        case = str(record.get("case"))
        if case not in by_case:
            order.append(case)
        try:
            threads = int(record.get("threads"))
        except (TypeError, ValueError):
            continue
        by_case[case][threads] = record

    rows = []
    for case in order:
        one = by_case[case].get(1)
        forty_eight = by_case[case].get(48)
        row = {
            "case": case,
            "wall_s_1": one.get("wall_s") if one else None,
            "wall_s_48": forty_eight.get("wall_s") if forty_eight else None,
            "speedup_48_vs_1": None,
            "efficiency_48": None,
            "metric_basis": None,
            "status": "incomplete",
        }
        one_status = one.get("status") if one else None
        forty_eight_status = forty_eight.get("status") if forty_eight else None
        if (
            one
            and forty_eight
            and one_status == "passed"
            and forty_eight_status == "passed"
            and float(forty_eight.get("wall_s") or 0.0) > 0.0
        ):
            speedup = float(one["wall_s"]) / float(forty_eight["wall_s"])
            efficiency = speedup / 48.0
            row["speedup_48_vs_1"] = speedup
            row["efficiency_48"] = efficiency
            row["metric_basis"] = "completion_wall_time"
            if efficiency >= 0.50:
                row["status"] = "good"
            elif efficiency >= 0.25:
                row["status"] = "partial"
            else:
                row["status"] = "poor"
        elif (
            one
            and forty_eight
            and one_status in ("passed", "sampled")
            and forty_eight_status in ("passed", "sampled")
        ):
            one_rate = (one.get("progress") or {}).get("step_throughput_per_s")
            forty_eight_rate = (forty_eight.get("progress") or {}).get("step_throughput_per_s")
            if one_rate and forty_eight_rate and float(one_rate) > 0.0:
                speedup = float(forty_eight_rate) / float(one_rate)
                efficiency = speedup / 48.0
                row["speedup_48_vs_1"] = speedup
                row["efficiency_48"] = efficiency
                row["metric_basis"] = "progress_throughput"
        if row["efficiency_48"] is not None:
            efficiency = float(row["efficiency_48"])
            if efficiency >= 0.50:
                row["status"] = "good"
            elif efficiency >= 0.25:
                row["status"] = "partial"
            else:
                row["status"] = "poor"
        rows.append(row)
    return rows


def _fmt_float(value, digits=2):
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def _html_pre_json(value) -> str:
    return "<pre>{0}</pre>".format(html.escape(json.dumps(value, indent=2, sort_keys=True, default=str)))


def _fmt_pct(value, digits=1):
    if value is None:
        return ""
    try:
        return f"{100.0 * float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return ""


def generate_html_report(records: Iterable[dict], output_path, *, metadata=None, title="Renormalizer Example Benchmark") -> Path:
    records = list(records)
    metadata = collect_environment_metadata() if metadata is None else metadata
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparisons = compare_cpu_results(records)

    rows = []
    for record in records:
        profile_summary = record.get("profile_summary") or {}
        event_counts = profile_summary.get("event_counts") or {}
        progress = record.get("progress") or {}
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(record.get('case', '')))}</td>"
            f"<td>{html.escape(str(record.get('suite', '')))}</td>"
            f"<td>{html.escape(str(record.get('backend', '')))}</td>"
            f"<td>{html.escape(str(record.get('device', '')))}</td>"
            f"<td>{html.escape(str(record.get('threads', '')))}</td>"
            f"<td>{html.escape(str(record.get('status', '')))}</td>"
            f"<td>{_fmt_float(record.get('wall_s'))}</td>"
            f"<td>{html.escape(str(record.get('returncode', '')))}</td>"
            f"<td>{_fmt_float((record.get('max_rss_kb') or 0) / 1024.0)}</td>"
            f"<td>{html.escape(str(record.get('max_threads', '')))}</td>"
            f"<td>{html.escape(str(progress.get('completed_steps', '')))}</td>"
            f"<td>{_fmt_float(progress.get('step_throughput_per_s'), digits=6)}</td>"
            f"<td>{html.escape(json.dumps(event_counts, sort_keys=True))}</td>"
            "</tr>"
        )

    profile_rows = []
    for record in records:
        profile_summary = record.get("profile_summary") or {}
        analysis = profile_summary.get("profile_analysis") or {}
        if not profile_summary.get("total_events") and not analysis:
            continue
        top_shapes = analysis.get("top_tensordot_shapes") or []
        top_shape_text = ", ".join(
            "{0} x{1}".format(item.get("shape_mnk"), item.get("count"))
            for item in top_shapes[:5]
        )
        profile_rows.append(
            "<tr>"
            f"<td>{html.escape(str(record.get('case', '')))}</td>"
            f"<td>{html.escape(str(record.get('suite', '')))}</td>"
            f"<td>{html.escape(str(record.get('threads', '')))}</td>"
            f"<td>{html.escape(str(profile_summary.get('total_events', '')))}</td>"
            f"<td>{html.escape(str(analysis.get('tensordot_total', '')))}</td>"
            f"<td>{_fmt_pct(analysis.get('small_gemm_pct'))}</td>"
            f"<td>{_fmt_pct(analysis.get('low_arithmetic_intensity_pct'))}</td>"
            f"<td>{html.escape(json.dumps(analysis.get('tensordot_kernel_counts') or {}, sort_keys=True))}</td>"
            f"<td>{html.escape(json.dumps(analysis.get('contraction_lowerings') or {}, sort_keys=True))}</td>"
            f"<td>{html.escape(str(analysis.get('grouped_gemm_execute_count', '')))}</td>"
            f"<td>{html.escape(json.dumps(analysis.get('grouped_pack_strategies') or {}, sort_keys=True))}</td>"
            f"<td>{html.escape(top_shape_text)}</td>"
            "</tr>"
        )

    comparison_rows = []
    for row in comparisons:
        comparison_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['case']))}</td>"
            f"<td>{_fmt_float(row.get('wall_s_1'))}</td>"
            f"<td>{_fmt_float(row.get('wall_s_48'))}</td>"
            f"<td>{_fmt_float(row.get('speedup_48_vs_1'))}</td>"
            f"<td>{_fmt_float(row.get('efficiency_48'))}</td>"
            f"<td>{html.escape(str(row.get('metric_basis') or ''))}</td>"
            f"<td>{html.escape(str(row.get('status')))}</td>"
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; }}
    h1, h2 {{ margin: 0.8rem 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; font-size: 14px; }}
    th, td {{ border: 1px solid #ccd4dd; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f6; }}
    pre {{ background: #f5f7fa; border: 1px solid #d8dee6; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <h2>Environment</h2>
  {_html_pre_json(metadata)}
  <h2>CPU 1 Core vs 48 Core Comparison</h2>
  <table>
    <thead><tr><th>Case</th><th>1 core wall s</th><th>48 core wall s</th><th>Speedup</th><th>Efficiency</th><th>Metric basis</th><th>Status</th></tr></thead>
    <tbody>{''.join(comparison_rows)}</tbody>
  </table>
  <h2>Profile Root Cause Summary</h2>
  <table>
    <thead><tr><th>Case</th><th>Suite</th><th>Threads</th><th>Total events</th><th>Tensordot events</th><th>Small GEMM</th><th>Low arithmetic intensity</th><th>Tensordot kernels</th><th>Contraction lowerings</th><th>Grouped GEMM events</th><th>Grouped pack strategies</th><th>Top m,n,k shapes</th></tr></thead>
    <tbody>{''.join(profile_rows)}</tbody>
  </table>
  <h2>Runs</h2>
  <table>
    <thead><tr><th>Case</th><th>Suite</th><th>Backend</th><th>Device</th><th>Threads</th><th>Status</th><th>Wall s</th><th>Return code</th><th>Max RSS MiB</th><th>Max threads</th><th>Completed steps</th><th>Step throughput / s</th><th>Profile event counts</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Raw Records</h2>
  {_html_pre_json(records)}
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def _read_jsonl_records(path) -> list[dict]:
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8", errors="replace") as fin:
        for line in fin:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _first_grouped_event(stage_a_summary: dict) -> dict:
    for event in stage_a_summary.get("events") or []:
        if isinstance(event, dict):
            return event
    return {}


def _stage_b_sbm_summary(records: list[dict]) -> dict:
    rows = [row for row in compare_cpu_results(records) if row.get("case") == "sbm"]
    comparison = rows[0] if rows else {}
    by_thread = {}
    for record in records:
        if record.get("case") != "sbm":
            continue
        try:
            by_thread[int(record.get("threads"))] = record
        except (TypeError, ValueError):
            continue
    return {
        "records": [by_thread[key] for key in sorted(by_thread)],
        "speedup_48_vs_1": comparison.get("speedup_48_vs_1"),
        "efficiency_48": comparison.get("efficiency_48"),
        "metric_basis": comparison.get("metric_basis"),
        "status": comparison.get("status"),
    }


def _classify_sbm_batched_grouped(stage_a_summary: dict, stage_b_summary: dict) -> str:
    grouped_count = int(stage_a_summary.get("grouped_gemm_execute_count") or 0)
    if grouped_count <= 0:
        return "path-coverage failure"
    speedup = stage_b_summary.get("speedup_48_vs_1")
    if speedup is None:
        return "entered grouped path, but CPU thread scaling is unverified"
    if float(speedup) >= 1.20:
        return "promising strategy"
    return "entered grouped path, but CPU thread scaling remains poor"


def generate_sbm_batched_grouped_report(
    stage_a_summary_path,
    stage_b_results_path,
    output_path,
    *,
    metadata=None,
    title="SBM Batched/Grouped GEMM Report",
) -> dict:
    """Generate the spec report that separates SBM path and performance evidence."""
    stage_a_summary_path = Path(stage_a_summary_path)
    stage_b_results_path = Path(stage_b_results_path)
    output_path = Path(output_path)
    stage_a = json.loads(stage_a_summary_path.read_text(encoding="utf-8"))
    records = _read_jsonl_records(stage_b_results_path)
    metadata = collect_environment_metadata() if metadata is None else metadata
    first_event = _first_grouped_event(stage_a)
    stage_b = _stage_b_sbm_summary(records)
    classification = _classify_sbm_batched_grouped(stage_a, stage_b)

    stage_a_table = [
        ("grouped_gemm_execute events", stage_a.get("grouped_gemm_execute_count")),
        ("first grouped num_tasks", first_event.get("num_tasks")),
        ("first grouped num_groups", first_event.get("num_groups")),
        ("first grouped group_sizes", first_event.get("group_sizes")),
        ("first grouped policy", first_event.get("policy")),
        ("batched kernel calls", first_event.get("batched_kernel_calls")),
        ("loop kernel calls", first_event.get("loop_kernel_calls")),
        ("fallback reasons", first_event.get("bucket_fallback_reasons")),
    ]
    stage_b_table = [
        ("48 vs 1 speedup", stage_b.get("speedup_48_vs_1")),
        ("48-core efficiency", stage_b.get("efficiency_48")),
        ("metric basis", stage_b.get("metric_basis")),
        ("classification", classification),
    ]

    def table_rows(rows):
        rendered = []
        for key, value in rows:
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            rendered.append(
                "<tr><td>{0}</td><td>{1}</td></tr>".format(
                    html.escape(str(key)),
                    html.escape("" if value is None else str(value)),
                )
            )
        return "".join(rendered)

    run_rows = []
    command_rows = []
    for record in stage_b["records"]:
        progress = record.get("progress") or {}
        run_rows.append(
            "<tr>"
            f"<td>{html.escape(str(record.get('threads', '')))}</td>"
            f"<td>{html.escape(str(record.get('status', '')))}</td>"
            f"<td>{_fmt_float(record.get('wall_s'))}</td>"
            f"<td>{html.escape(str(progress.get('completed_steps', '')))}</td>"
            f"<td>{_fmt_float(progress.get('mean_logged_step_s'))}</td>"
            f"<td>{_fmt_float(progress.get('step_throughput_per_s'), digits=6)}</td>"
            f"<td>{_fmt_float((record.get('max_rss_kb') or 0) / 1024.0)}</td>"
            f"<td>{html.escape(str(record.get('max_threads', '')))}</td>"
            "</tr>"
        )
        command_rows.append(
            "<tr>"
            f"<td>{html.escape(str(record.get('threads', '')))}</td>"
            f"<td>{html.escape(' '.join(str(item) for item in (record.get('command') or [])))}</td>"
            f"<td>{html.escape(json.dumps(record.get('thread_env') or {}, sort_keys=True))}</td>"
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; }}
    h1, h2 {{ margin: 0.8rem 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; font-size: 14px; }}
    th, td {{ border: 1px solid #ccd4dd; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f6; }}
    pre {{ background: #f5f7fa; border: 1px solid #d8dee6; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <h2>Conclusion</h2>
  <p>{html.escape(classification)}</p>
  <h2>Stage A: Path Evidence</h2>
  <table><tbody>{table_rows(stage_a_table)}</tbody></table>
  <h2>Stage B: Performance Evidence</h2>
  <table><tbody>{table_rows(stage_b_table)}</tbody></table>
  <table>
    <thead><tr><th>Threads</th><th>Status</th><th>Wall s</th><th>Completed steps</th><th>Mean step s</th><th>Step throughput / s</th><th>Max RSS MiB</th><th>Max threads</th></tr></thead>
    <tbody>{''.join(run_rows)}</tbody>
  </table>
  <h2>Benchmark command and thread environment</h2>
  <table>
    <thead><tr><th>Threads</th><th>Command</th><th>Thread env</th></tr></thead>
    <tbody>{''.join(command_rows)}</tbody>
  </table>
  <h2>Environment</h2>
  {_html_pre_json(metadata)}
  <h2>Raw Stage A Summary</h2>
  {_html_pre_json(stage_a)}
  <h2>Raw Stage B Records</h2>
  {_html_pre_json(stage_b["records"])}
  <h2>Source Files</h2>
  <ul>
    <li>{html.escape(str(stage_a_summary_path))}</li>
    <li>{html.escape(str(stage_b_results_path))}</li>
  </ul>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")

    return {
        "classification": classification,
        "stage_a": {
            "grouped_gemm_execute_count": stage_a.get("grouped_gemm_execute_count"),
            "first_grouped_event": first_event,
        },
        "stage_b": {
            "speedup_48_vs_1": stage_b.get("speedup_48_vs_1"),
            "efficiency_48": stage_b.get("efficiency_48"),
            "metric_basis": stage_b.get("metric_basis"),
            "status": stage_b.get("status"),
        },
        "output_path": str(output_path),
    }


def _parse_thread_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def _select_cases(names: str | None) -> list[ExampleCase]:
    cases = default_example_cases()
    if not names:
        return cases
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    by_name = {case.name: case for case in [*cases, *extra_probe_cases()]}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise SystemExit("unknown example case(s): {0}".format(", ".join(missing)))
    return [by_name[name] for name in wanted]


def _jobs_from_args(args) -> list[BenchmarkJob]:
    cases = _select_cases(args.cases)
    if args.suite == "cpu":
        return make_cpu_jobs(
            cases,
            args.output_dir,
            threads=_parse_thread_list(args.threads),
            backend=args.backend,
            device=args.device,
            profile_mode=args.profile_mode,
        )
    if args.suite == "cupy":
        return make_cupy_jobs(
            cases,
            args.output_dir,
            backend=args.backend,
            device=args.device,
            threads=int(args.threads),
            profile_mode=args.profile_mode,
        )
    raise SystemExit("unknown suite {0!r}".format(args.suite))


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, sort_keys=True) + "\n")


def _cmd_plan(args) -> int:
    jobs = _jobs_from_args(args)
    for job in jobs:
        print(
            "{suite}\t{backend}\t{device}\tthreads={threads}\t{case}\t{script} {argv}".format(
                suite=job.suite,
                backend=job.backend,
                device=job.device,
                threads=job.threads,
                case=job.case.name,
                script=job.case.script,
                argv=" ".join(job.case.args),
            )
        )
    print("planned_jobs={0}".format(len(jobs)))
    return 0


def _cmd_run(args) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = collect_environment_metadata()
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    records = []
    for job in _jobs_from_args(args):
        print("running {0} {1} {2} threads={3}".format(job.suite, job.backend, job.case.name, job.threads), flush=True)
        records.append(
            run_job(
                job,
                sample_interval=float(args.sample_interval),
                timeout=args.timeout,
                timeout_status=args.timeout_status,
                metadata=metadata,
            )
        )
        _write_jsonl(output_dir / "results.jsonl", records)
    generate_html_report(records, output_dir / "report.html", metadata=metadata, title=args.title)
    print("wrote {0}".format(output_dir / "report.html"))
    success_statuses = {"passed", "sampled"} if args.timeout_status == "sampled" else {"passed"}
    return 0 if all(record.get("status") in success_statuses for record in records) else 1


def _make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser):
        subparser.add_argument("--suite", choices=("cpu", "cupy"), default="cpu")
        subparser.add_argument("--backend", default="numpy")
        subparser.add_argument("--device", default="cpu")
        subparser.add_argument("--threads", default="1,48")
        subparser.add_argument("--output-dir", default="docs/backend_report/runs/example-benchmark")
        subparser.add_argument("--cases", default=None, help="Comma-separated subset of example case names.")
        subparser.add_argument(
            "--profile-mode",
            choices=("off", "full"),
            default="off",
            help="Use 'full' only for short profiling probes; full JSONL can be very large.",
        )

    plan_parser = subparsers.add_parser("plan", help="Print planned jobs without running them.")
    add_common(plan_parser)
    plan_parser.set_defaults(func=_cmd_plan)

    run_parser = subparsers.add_parser("run", help="Run planned jobs and generate report.html.")
    add_common(run_parser)
    run_parser.add_argument("--sample-interval", type=float, default=60.0)
    run_parser.add_argument("--timeout", type=float, default=None)
    run_parser.add_argument(
        "--timeout-status",
        choices=("timeout", "sampled"),
        default="timeout",
        help="Use 'sampled' for bounded benchmark samples where timeout is expected data.",
    )
    run_parser.add_argument("--title", default="Renormalizer Example Benchmark")
    run_parser.set_defaults(func=_cmd_run)

    return parser


def main(argv=None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    if args.suite == "cupy":
        if args.backend == "numpy":
            args.backend = "cupy"
        if args.device == "cpu":
            args.device = "cuda:0"
        if str(args.threads) == "1,48":
            args.threads = "48"
        if "," in str(args.threads):
            raise SystemExit("cupy suite accepts a single --threads value")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
