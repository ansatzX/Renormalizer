# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from renormalizer.backend.config import BackendConfig
from renormalizer.backend.execution import (
    BlockContractionSpec,
    BlockKey,
    BlockTensor,
    DenseBlock,
    LayoutSpec,
    MatmulDesc,
    MatmulPlan,
)
from renormalizer.backend.factory import create_backend, is_backend_available, normalize_backend_name
from renormalizer.backend.gemm import (
    execute_prepacked_grouped_gemm as execute_prepacked_grouped_gemm_raw,
    grouped_gemm_stats,
    run_gemm_task,
)


DEFAULT_BACKENDS = ("numpy", "cupy", "torch", "jax")
GROUPED_GEMM_MAX_ABS_ERROR_TOLERANCE = 1e-9
GROUPED_SHAPE_MODES = ("same", "block_heavy")
GROUPED_SOURCES = ("descriptors", "block_tensor")


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


def parse_int_list(value):
    values = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parsed = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("invalid integer value {0!r}".format(item)) from exc
        if parsed <= 0:
            raise argparse.ArgumentTypeError("dimensions must be positive integers")
        values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return tuple(values)


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


@contextlib.contextmanager
def nvidia_dmon_monitor(output_path=None, *, fields="pucvmte"):
    if not output_path:
        yield None
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("w", encoding="utf-8")
    process = None
    try:
        process = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", str(fields), "-o", "T"],
            stdout=stream,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        yield process
    finally:
        if process is not None:
            poll = getattr(process, "poll", None)
            if poll is None or poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        stream.close()


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


def _matmul_estimates(a, b, output_shape, *, flops):
    itemsize = max(int(getattr(getattr(a, "dtype", None), "itemsize", 0) or 0), 1)
    return {
        "estimated_flops": int(flops),
        "estimated_read_bytes": int(getattr(a, "nbytes", 0) or 0) + int(getattr(b, "nbytes", 0) or 0),
        "estimated_write_bytes": int(np.prod(tuple(output_shape))) * itemsize,
    }


def _profile_summary(profile):
    if not profile:
        return None
    fields = (
        "event",
        "backend",
        "lowering",
        "plan_hash",
        "input_shapes",
        "output_shape",
        "dtype",
        "device_kind",
        "flops",
        "read_bytes",
        "write_bytes",
        "copy_bytes",
        "workspace_bytes",
        "largest_intermediate",
        "num_gemm",
        "num_batched_gemm",
        "num_grouped_tasks",
        "num_blocks",
        "num_shape_buckets",
        "scatter_add_required",
        "dense_materialized",
        "materialized_dense_bytes",
        "result_block_shapes",
        "bucket_task_counts",
        "shape_buckets",
        "group_sizes",
        "gsta",
        "sorted_indices",
        "group_keys",
        "supports_grouped_gemm",
        "grouped_gemm_policy",
        "grouped_gemm_implementation",
        "requires_grouped_gemm_fallback",
        "batched_bucket_count",
        "loop_task_count",
        "pack_threshold",
        "pack_strategy",
        "prepacked",
        "pack_s",
        "kernel_s",
        "scatter_s",
        "loop_s",
        "pack_bytes",
        "reused_pack_bytes",
        "kernel_calls",
        "batched_kernel_calls",
        "loop_kernel_calls",
        "bucket_execution_profiles",
        "fallback_reason",
        "fallback_from",
        "fallback_to",
        "wall_s",
    )
    return {
        field: profile.get(field)
        for field in fields
        if field in profile
    }


def _smoke_layout(shape, modes):
    shape = tuple(int(dim) for dim in shape)
    return LayoutSpec(
        logical_shape=shape,
        physical_shape=shape,
        logical_modes=tuple(modes),
        strides=None,
        order="C",
        contiguous_groups=(tuple(range(len(shape))),),
    )


def _grouped_gemm_fallback_reason(backend):
    if bool(getattr(backend, "supports_grouped_gemm", False)):
        return None
    return "backend-owned grouped_gemm unavailable; used bucketed fallback"


def _grouped_gemm_policy(stats, fallback_reason):
    if not int(getattr(stats, "task_count", 0) or 0):
        return "empty"
    if int(getattr(stats, "batched_task_count", 0) or 0) and int(getattr(stats, "loop_task_count", 0) or 0):
        return "bucketed_mixed_matmul"
    if int(getattr(stats, "batched_task_count", 0) or 0):
        return "bucketed_batched_matmul"
    return "bucketed_loop_matmul"


def _grouped_gemm_implementation(stats, fallback_reason):
    policy = _grouped_gemm_policy(stats, fallback_reason)
    if policy == "empty":
        return "empty"
    prefix = "fallback" if fallback_reason is not None else "backend"
    return "{0}_{1}".format(prefix, policy)


def _json_shape_buckets(stats):
    buckets = []
    for bucket in getattr(stats, "shape_buckets", ()):
        item = dict(bucket)
        item["batch_shape"] = [int(dim) for dim in item.get("batch_shape", ())]
        for field in ("m", "n", "k", "task_count"):
            item[field] = int(item.get(field) or 0)
        for field in ("trans_a", "trans_b", "conj_a", "conj_b"):
            item[field] = bool(item.get(field))
        item["execution"] = str(item.get("execution") or "")
        buckets.append(item)
    return buckets


def _grouped_shape_specs(batch, grouped_dim, grouped_shape_mode):
    batch = int(batch)
    grouped_dim = int(grouped_dim)
    if batch <= 0:
        raise ValueError("batch must be positive")
    if grouped_dim <= 0:
        raise ValueError("grouped_dim must be positive")
    mode = str(grouped_shape_mode)
    if mode not in GROUPED_SHAPE_MODES:
        raise ValueError("unknown grouped_shape_mode {0!r}".format(grouped_shape_mode))
    if mode == "same":
        return [(grouped_dim, grouped_dim, grouped_dim) for _ in range(batch)]

    dim = max(2, grouped_dim)
    bucket_shapes = (
        (dim, dim, dim),
        (max(2, dim // 2), dim, max(2, dim // 2)),
        (dim, max(2, dim // 2), max(2, dim // 3)),
        (max(2, dim // 3), max(2, dim // 3), dim),
    )
    return [bucket_shapes[index % len(bucket_shapes)] for index in range(batch)]


def _make_block_tensor_grouped_spec(backend, tasks_np, task_shapes):
    max_m = max(int(m) for m, _n, _k in task_shapes)
    max_n = max(int(n) for _m, n, _k in task_shapes)
    max_k = max(int(k) for _m, _n, k in task_shapes)
    left_blocks = {}
    right_blocks = {}
    expected_by_key = {}
    for index, ((a_np, b_np), (m, n, k)) in enumerate(zip(tasks_np, task_shapes)):
        left_key = BlockKey((int(index),), (int(index),))
        right_key = BlockKey((int(index),), (int(index),))
        output_key = BlockKey(left_key.qn_left, right_key.qn_right, extra=(int(index),))
        left_blocks[left_key] = DenseBlock(
            key=left_key,
            array=backend.to_backend(a_np),
            modes=("i", "k"),
            shape=(int(m), int(k)),
        )
        right_blocks[right_key] = DenseBlock(
            key=right_key,
            array=backend.to_backend(b_np),
            modes=("k", "j"),
            shape=(int(k), int(n)),
        )
        expected_by_key[output_key] = a_np @ b_np

    left = BlockTensor(
        left_blocks,
        global_shape=(int(max_m), int(max_k)),
        modes=("i", "k"),
        block_axis_meta={"source": "gemm_smoke", "side": "left"},
        backend=backend.name,
    )
    right = BlockTensor(
        right_blocks,
        global_shape=(int(max_k), int(max_n)),
        modes=("k", "j"),
        block_axis_meta={"source": "gemm_smoke", "side": "right"},
        backend=backend.name,
    )

    def qn_rule(left_key, right_key):
        if left_key.qn_right != right_key.qn_left:
            return None
        return BlockKey(left_key.qn_left, right_key.qn_right, extra=left_key.qn_left)

    return (
        BlockContractionSpec(left, right, output_modes=("i", "j"), qn_rule=qn_rule),
        expected_by_key,
    )


def _block_key_payload(key):
    payload = {
        "qn_left": [int(value) for value in getattr(key, "qn_left", ())],
        "qn_right": [int(value) for value in getattr(key, "qn_right", ())],
        "extra": list(getattr(key, "extra", ())),
    }
    return payload


def _block_plan_profile(plan):
    return {
        "event": "contraction_plan",
        "lowering": "block_grouped_gemm",
        "plan_hash": plan.plan_hash,
        "output_modes": [str(mode) for mode in plan.output_modes],
        "output_shape": tuple(plan.global_shape),
        "global_shape": tuple(plan.global_shape),
        "flops": int(plan.estimated_flops),
        "read_bytes": int(plan.estimated_read_bytes),
        "write_bytes": int(plan.estimated_write_bytes),
        "copy_bytes": int(plan.estimated_copy_bytes),
        "workspace_bytes": int(plan.estimated_workspace_bytes),
        "num_grouped_tasks": int(len(plan.tasks)),
        "num_blocks": int(len(set(plan.output_blocks))),
        "num_shape_buckets": int(len(plan.bucketed_by_shape)),
        "scatter_add_required": bool(plan.scatter_add_required),
        "dense_materialized": False,
        "materialized_dense_bytes": 0,
        "bucket_task_counts": [int(len(indices)) for indices in plan.bucketed_by_shape.values()],
        "shape_buckets": [
            {
                "m": int(shape[0]),
                "n": int(shape[1]),
                "k": int(shape[2]),
                "task_indices": [int(index) for index in indices],
                "task_count": int(len(indices)),
            }
            for shape, indices in plan.bucketed_by_shape.items()
        ],
        "output_block_keys": [_block_key_payload(key) for key in plan.output_blocks],
    }


def _result_block_shapes(block_tensor):
    return [
        {
            **_block_key_payload(key),
            "shape": tuple(block.shape),
        }
        for key, block in sorted(
            block_tensor.blocks.items(),
            key=lambda item: (
                tuple(getattr(item[0], "qn_left", ())),
                tuple(getattr(item[0], "qn_right", ())),
                tuple(getattr(item[0], "extra", ())),
            ),
        )
    ]


def _matmul_smoke(backend, backend_name, device, gpu_count, repeat, medium_dim, warmup=0, trials=1):
    a_np = _random((medium_dim, medium_dim), 1)
    b_np = _random((medium_dim, medium_dim), 2)
    a = backend.to_backend(a_np)
    b = backend.to_backend(b_np)
    estimates = _matmul_estimates(
        a,
        b,
        a_np.shape,
        flops=2 * int(medium_dim) * int(medium_dim) * int(medium_dim),
    )
    desc = MatmulDesc(
        a,
        b,
        None,
        int(medium_dim),
        int(medium_dim),
        int(medium_dim),
        layout_a=_smoke_layout(a_np.shape, ("i", "k")),
        layout_b=_smoke_layout(b_np.shape, ("k", "j")),
        layout_c=_smoke_layout(a_np.shape, ("i", "j")),
        **estimates,
    )
    plan = MatmulPlan(
        kind="gemm",
        descs=(desc,),
        pre_ops=(),
        post_ops=(),
        output_shape=tuple(a_np.shape),
        copy_bytes=0,
        workspace_bytes=desc.estimated_workspace_bytes,
        estimated_flops=desc.estimated_flops,
        estimated_time_s=None,
        reason="gemm_smoke matmul",
    )
    result, wall_s, samples = _time_call(
        backend,
        repeat,
        lambda: backend.execute_matmul_plan(
            plan,
            equation="ik,kj->ij",
            input_modes=(("i", "k"), ("k", "j")),
            output_modes=("i", "j"),
        ),
        warmup=warmup,
        trials=trials,
    )
    expected = a_np @ b_np
    record = _record_base(backend, backend_name, device, "matmul", gpu_count, repeat, warmup, trials)
    record.update({
        "shape_a": tuple(a_np.shape),
        "shape_b": tuple(b_np.shape),
        "execution_profile": _profile_summary(backend.last_execution_profile()),
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
    estimates = _matmul_estimates(
        a,
        b,
        a_np.shape,
        flops=2 * int(batch) * int(small_dim) * int(small_dim) * int(small_dim),
    )
    fallback_reason = None
    if bool(getattr(backend, "supports_batched_matmul", False)):
        kind = "batched_gemm"
    elif bool(getattr(backend, "supports_matmul", False)):
        kind = "fallback_tensordot"
        fallback_reason = "backend lacks batched_matmul for batch shape ({0},)".format(int(batch))
    else:
        kind = "fallback_einsum"
        fallback_reason = "backend lacks batched_matmul and matmul for batch shape ({0},)".format(int(batch))
    desc = MatmulDesc(
        a,
        b,
        None,
        int(small_dim),
        int(small_dim),
        int(small_dim),
        batch_shape=(int(batch),),
        layout_a=_smoke_layout(a_np.shape, ("b", "i", "k")),
        layout_b=_smoke_layout(b_np.shape, ("b", "k", "j")),
        layout_c=_smoke_layout(a_np.shape, ("b", "i", "j")),
        **estimates,
    )
    plan = MatmulPlan(
        kind=kind,
        descs=(desc,),
        pre_ops=(),
        post_ops=(),
        output_shape=tuple(a_np.shape),
        copy_bytes=0,
        workspace_bytes=desc.estimated_workspace_bytes,
        estimated_flops=desc.estimated_flops,
        estimated_time_s=None,
        reason="gemm_smoke batched_matmul",
        fallback_reason=fallback_reason,
    )
    result, wall_s, samples = _time_call(
        backend,
        repeat,
        lambda: backend.execute_matmul_plan(
            plan,
            equation="bik,bkj->bij",
            input_modes=(("b", "i", "k"), ("b", "k", "j")),
            output_modes=("b", "i", "j"),
        ),
        warmup=warmup,
        trials=trials,
    )
    expected = np.matmul(a_np, b_np)
    record = _record_base(backend, backend_name, device, "batched_matmul", gpu_count, repeat, warmup, trials)
    record.update({
        "shape_a": tuple(a_np.shape),
        "shape_b": tuple(b_np.shape),
        "execution_profile": _profile_summary(backend.last_execution_profile()),
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
    grouped_dim=None,
    grouped_shape_mode="same",
    grouped_source="descriptors",
):
    grouped_dim = small_dim if grouped_dim is None else int(grouped_dim)
    grouped_shape_mode = str(grouped_shape_mode)
    if grouped_shape_mode not in GROUPED_SHAPE_MODES:
        raise ValueError("unknown grouped_shape_mode {0!r}".format(grouped_shape_mode))
    grouped_source = str(grouped_source)
    if grouped_source not in GROUPED_SOURCES:
        raise ValueError("unknown grouped_source {0!r}".format(grouped_source))
    tasks_np = []
    task_shapes = _grouped_shape_specs(batch, grouped_dim, grouped_shape_mode)
    for index, (m, n, k) in enumerate(task_shapes):
        tasks_np.append((
            _random((m, k), 100 + index),
            _random((k, n), 200 + index),
        ))
    block_plan_profile = None
    result_block_shapes = []
    block_tensor_dense_materialized = None
    block_expected_by_key = {}
    block_plan = None
    if grouped_source == "block_tensor":
        block_spec, block_expected_by_key = _make_block_tensor_grouped_spec(backend, tasks_np, task_shapes)
        block_plan = backend.lower_block_contraction(block_spec)
        block_plan_profile = _block_plan_profile(block_plan)
        descs = tuple(block_plan.tasks)
        block_tensor_dense_materialized = False
    else:
        descs = []
        for _index, ((a_np, b_np), (m, n, k)) in enumerate(zip(tasks_np, task_shapes)):
            a = backend.to_backend(a_np)
            b = backend.to_backend(b_np)
            estimates = _matmul_estimates(
                a,
                b,
                (m, n),
                flops=2 * int(m) * int(n) * int(k),
            )
            descs.append(
                MatmulDesc(
                    a,
                    b,
                    None,
                    int(m),
                    int(n),
                    int(k),
                    layout_a=_smoke_layout(a_np.shape, ("i", "k")),
                    layout_b=_smoke_layout(b_np.shape, ("k", "j")),
                    layout_c=_smoke_layout((m, n), ("i", "j")),
                    **estimates,
                )
            )
        descs = tuple(descs)
    flop_copy_ratio = 0 if backend.supports_grouped_gemm else 10
    stats = grouped_gemm_stats(
        descs,
        xp=backend.array_namespace,
        pack_threshold=pack_threshold,
        flop_copy_ratio=flop_copy_ratio,
    )
    fallback_reason = _grouped_gemm_fallback_reason(backend)
    _, loop_wall_s, loop_samples = _time_call(
        backend,
        repeat,
        lambda: [run_gemm_task(desc, xp=backend.array_namespace) for desc in descs],
        warmup=warmup,
        trials=trials,
    )
    if grouped_source == "block_tensor":
        grouped_call = lambda: backend.execute_grouped_gemm_plan(block_plan, pack_threshold=pack_threshold)
    elif grouped_shape_mode == "same":
        plan = MatmulPlan(
            kind="grouped_gemm",
            descs=descs,
            pre_ops=(),
            post_ops=(),
            output_shape=(int(grouped_dim), int(grouped_dim)),
            copy_bytes=stats.copy_bytes,
            workspace_bytes=stats.workspace_bytes,
            estimated_flops=stats.flops,
            estimated_time_s=None,
            reason="gemm_smoke grouped_gemm",
            fallback_reason=fallback_reason,
        )
        grouped_call = lambda: backend.execute_matmul_plan(
            plan,
            pack_threshold=pack_threshold,
            equation="grouped:ik,kj->ij",
            input_modes=(("i", "k"), ("k", "j")),
            output_modes=("i", "j"),
        )
    else:
        grouped_call = lambda: backend.grouped_gemm(descs, pack_threshold=pack_threshold)
    if grouped_source == "block_tensor":
        from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

        old_level = package_logger.level
        try:
            init_log(PROFILING)
            result, wall_s, grouped_samples = _time_call(
                backend,
                repeat,
                grouped_call,
                warmup=warmup,
                trials=trials,
            )
        finally:
            init_log(old_level or DEBUG)
        result_block_shapes = _result_block_shapes(result)
    else:
        result, wall_s, grouped_samples = _time_call(
            backend,
            repeat,
            grouped_call,
            warmup=warmup,
            trials=trials,
        )
    grouped_execution_profile = backend.last_execution_profile()
    prepacked_plan = backend.prepack_grouped_gemm(descs, pack_threshold=pack_threshold)
    prepacked_result, prepacked_wall_s, prepacked_samples = _time_call(
        backend,
        repeat,
        lambda: backend.execute_prepacked_grouped_gemm(prepacked_plan),
        warmup=warmup,
        trials=trials,
    )
    raw_prepacked_profile = {}

    def raw_prepacked_call():
        raw_result, raw_profile = execute_prepacked_grouped_gemm_raw(
            prepacked_plan,
            xp=backend.array_namespace,
        )
        raw_prepacked_profile.clear()
        raw_prepacked_profile.update(raw_profile)
        return raw_result

    raw_prepacked_result, raw_prepacked_wall_s, raw_prepacked_samples = _time_call(
        backend,
        repeat,
        raw_prepacked_call,
        warmup=warmup,
        trials=trials,
    )
    if grouped_source == "block_tensor":
        expected = [block_expected_by_key[key] for key in block_plan.output_blocks]
        actual = [_to_numpy(backend, result.blocks[key].array) for key in block_plan.output_blocks]
    else:
        expected = [a_np @ b_np for a_np, b_np in tasks_np]
        actual = [_to_numpy(backend, item) for item in result]
    prepacked_actual = [_to_numpy(backend, item) for item in prepacked_result]
    raw_prepacked_actual = [_to_numpy(backend, item) for item in raw_prepacked_result]
    max_error = max(_max_abs_error(item, ref) for item, ref in zip(actual, expected))
    prepacked_max_error = max(_max_abs_error(item, ref) for item, ref in zip(prepacked_actual, expected))
    raw_prepacked_max_error = max(_max_abs_error(item, ref) for item, ref in zip(raw_prepacked_actual, expected))
    record = _record_base(backend, backend_name, device, "grouped_gemm", gpu_count, repeat, warmup, trials)
    record.update({
        "supports_grouped_gemm": bool(getattr(backend, "supports_grouped_gemm", False)),
        "grouped_gemm_policy": _grouped_gemm_policy(stats, fallback_reason),
        "grouped_gemm_implementation": _grouped_gemm_implementation(stats, fallback_reason),
        "requires_grouped_gemm_fallback": fallback_reason is not None,
        "fallback_from": "grouped_gemm" if fallback_reason is not None else None,
        "fallback_to": _grouped_gemm_policy(stats, fallback_reason) if fallback_reason is not None else None,
        "fallback_reason": fallback_reason,
        "task_count": int(len(descs)),
        "shape": (grouped_dim, grouped_dim, grouped_dim),
        "grouped_dim": int(grouped_dim),
        "grouped_shape_mode": grouped_shape_mode,
        "grouped_source": grouped_source,
        "grouped_task_shapes": [tuple(int(value) for value in shape) for shape in task_shapes],
        "pack_threshold": int(pack_threshold),
        "num_shape_buckets": stats.shape_bucket_count,
        "num_batched_gemm": stats.batched_bucket_count,
        "num_gemm": stats.loop_task_count,
        "num_grouped_tasks": stats.task_count,
        "bucket_task_counts": [int(value) for value in stats.bucket_task_counts],
        "shape_buckets": _json_shape_buckets(stats),
        "copy_bytes": stats.copy_bytes,
        "flops": stats.flops,
        "read_bytes": stats.read_bytes,
        "write_bytes": stats.write_bytes,
        "block_tensor_dense_materialized": block_tensor_dense_materialized,
        "block_plan_profile": block_plan_profile,
        "result_block_shapes": result_block_shapes,
        "execution_profile": _profile_summary(grouped_execution_profile),
        "prepack_profile": _profile_summary(prepacked_plan.profile),
        "prepacked_execution_profile": _profile_summary(backend.last_execution_profile()),
        "wall_s": float(wall_s),
        "wall_s_samples": grouped_samples,
        "grouped_wall_s": float(wall_s),
        "grouped_wall_s_samples": grouped_samples,
        "prepacked_wall_s": float(prepacked_wall_s),
        "prepacked_wall_s_samples": prepacked_samples,
        "raw_prepacked_wall_s": float(raw_prepacked_wall_s),
        "raw_prepacked_wall_s_samples": raw_prepacked_samples,
        "raw_prepacked_execution_profile": _profile_summary(raw_prepacked_profile),
        "loop_wall_s": float(loop_wall_s),
        "loop_wall_s_samples": loop_samples,
        "speedup_vs_loop": float(loop_wall_s / wall_s) if wall_s > 0.0 else None,
        "prepacked_speedup_vs_loop": float(loop_wall_s / prepacked_wall_s) if prepacked_wall_s > 0.0 else None,
        "raw_prepacked_speedup_vs_loop": (
            float(loop_wall_s / raw_prepacked_wall_s)
            if raw_prepacked_wall_s > 0.0
            else None
        ),
        "prepacked_speedup_vs_grouped": float(wall_s / prepacked_wall_s) if prepacked_wall_s > 0.0 else None,
        "max_abs_error": float(max_error),
        "prepacked_max_abs_error": float(prepacked_max_error),
        "raw_prepacked_max_abs_error": float(raw_prepacked_max_error),
    })
    return record


def _grouped_records(records):
    return [
        record for record in records
        if record.get("operation") == "grouped_gemm"
    ]


def _device_index(device):
    value = str(device)
    for prefix in ("cuda:", "gpu:"):
        if value.startswith(prefix):
            try:
                return int(value[len(prefix):])
            except ValueError:
                return None
    return None


def _record_grouped_dim(record):
    if record.get("grouped_dim") is not None:
        try:
            return int(record.get("grouped_dim"))
        except Exception:
            return None
    shape = record.get("shape")
    if isinstance(shape, (list, tuple)) and len(shape) == 3:
        try:
            dims = tuple(int(dim) for dim in shape)
        except Exception:
            return None
        if dims[0] == dims[1] == dims[2]:
            return dims[0]
    return None


_GROUPED_GEMM_TELEMETRY_FIELDS = (
    "flops",
    "read_bytes",
    "write_bytes",
    "copy_bytes",
    "wall_s",
    "grouped_wall_s",
    "loop_wall_s",
    "num_shape_buckets",
    "bucket_task_counts",
    "shape_buckets",
    "grouped_gemm_policy",
    "grouped_gemm_implementation",
    "requires_grouped_gemm_fallback",
    "fallback_from",
    "fallback_to",
    "fallback_reason",
)


def _gate_failure(record, reason, **extra):
    failure = {
        "backend": record.get("backend"),
        "device": record.get("device"),
        "reason": reason,
    }
    if "speedup_vs_loop" in record:
        failure["speedup_vs_loop"] = record.get("speedup_vs_loop")
    failure.update(extra)
    return failure


def _missing_grouped_gemm_telemetry(record):
    return [
        field
        for field in _GROUPED_GEMM_TELEMETRY_FIELDS
        if field not in record
    ]


def _float_record_value(record, field):
    try:
        return float(record.get(field))
    except Exception:
        return None


def _validate_grouped_gemm_timing(record):
    wall_s = _float_record_value(record, "wall_s")
    grouped_wall_s = _float_record_value(record, "grouped_wall_s")
    loop_wall_s = _float_record_value(record, "loop_wall_s")
    speedup = _float_record_value(record, "speedup_vs_loop")
    values = (wall_s, grouped_wall_s, loop_wall_s, speedup)
    if any(value is None or value < 0.0 for value in values):
        return [_gate_failure(record, "invalid grouped_gemm timing telemetry")]

    failures = []
    if not math.isclose(wall_s, grouped_wall_s, rel_tol=1e-9, abs_tol=1e-12):
        failures.append(_gate_failure(
            record,
            "grouped_gemm wall time fields disagree",
            wall_s=wall_s,
            grouped_wall_s=grouped_wall_s,
        ))
    if grouped_wall_s > 0.0:
        expected_speedup = loop_wall_s / grouped_wall_s
        if not math.isclose(speedup, expected_speedup, rel_tol=1e-9, abs_tol=1e-12):
            failures.append(_gate_failure(
                record,
                "speedup does not match timing telemetry",
                expected_speedup_vs_loop=expected_speedup,
            ))
    elif speedup is not None:
        failures.append(_gate_failure(record, "invalid grouped_gemm timing telemetry"))
    return failures


def _validate_prepacked_grouped_gemm_timing(record):
    fields = (
        "prepacked_wall_s",
        "prepacked_speedup_vs_loop",
        "prepacked_max_abs_error",
    )
    if not any(field in record for field in fields):
        return []

    missing = [
        field for field in fields
        if field not in record or record.get(field) is None
    ]
    if missing:
        return [_gate_failure(
            record,
            "missing prepacked grouped_gemm telemetry",
            missing_fields=missing,
        )]

    loop_wall_s = _float_record_value(record, "loop_wall_s")
    prepacked_wall_s = _float_record_value(record, "prepacked_wall_s")
    prepacked_speedup = _float_record_value(record, "prepacked_speedup_vs_loop")
    prepacked_max_error = _float_record_value(record, "prepacked_max_abs_error")
    if (
        loop_wall_s is None
        or loop_wall_s < 0.0
        or prepacked_wall_s is None
        or prepacked_wall_s <= 0.0
        or prepacked_speedup is None
        or prepacked_speedup < 0.0
        or prepacked_max_error is None
        or prepacked_max_error < 0.0
    ):
        return [_gate_failure(record, "invalid prepacked grouped_gemm timing telemetry")]

    expected_speedup = loop_wall_s / prepacked_wall_s
    if not math.isclose(prepacked_speedup, expected_speedup, rel_tol=1e-9, abs_tol=1e-12):
        return [_gate_failure(
            record,
            "prepacked speedup does not match timing telemetry",
            expected_prepacked_speedup_vs_loop=expected_speedup,
        )]
    return []


def _validate_raw_prepacked_grouped_gemm_timing(record):
    fields = (
        "raw_prepacked_wall_s",
        "raw_prepacked_speedup_vs_loop",
        "raw_prepacked_max_abs_error",
    )
    if not any(field in record for field in fields):
        return []

    missing = [
        field for field in fields
        if field not in record or record.get(field) is None
    ]
    if missing:
        return [_gate_failure(
            record,
            "missing raw prepacked grouped_gemm telemetry",
            missing_fields=missing,
        )]

    loop_wall_s = _float_record_value(record, "loop_wall_s")
    raw_prepacked_wall_s = _float_record_value(record, "raw_prepacked_wall_s")
    raw_prepacked_speedup = _float_record_value(record, "raw_prepacked_speedup_vs_loop")
    raw_prepacked_max_error = _float_record_value(record, "raw_prepacked_max_abs_error")
    if (
        loop_wall_s is None
        or loop_wall_s < 0.0
        or raw_prepacked_wall_s is None
        or raw_prepacked_wall_s <= 0.0
        or raw_prepacked_speedup is None
        or raw_prepacked_speedup < 0.0
        or raw_prepacked_max_error is None
        or raw_prepacked_max_error < 0.0
    ):
        return [_gate_failure(record, "invalid raw prepacked grouped_gemm timing telemetry")]

    expected_speedup = loop_wall_s / raw_prepacked_wall_s
    if not math.isclose(raw_prepacked_speedup, expected_speedup, rel_tol=1e-9, abs_tol=1e-12):
        return [_gate_failure(
            record,
            "raw prepacked speedup does not match timing telemetry",
            expected_raw_prepacked_speedup_vs_loop=expected_speedup,
        )]
    return []


def _validate_grouped_gemm_correctness(record):
    failures = []
    tolerance = float(GROUPED_GEMM_MAX_ABS_ERROR_TOLERANCE)
    if "max_abs_error" in record:
        max_error = _float_record_value(record, "max_abs_error")
        if max_error is None or max_error < 0.0:
            failures.append(_gate_failure(record, "invalid grouped_gemm correctness telemetry"))
        elif max_error > tolerance:
            failures.append(_gate_failure(
                record,
                "grouped_gemm correctness error exceeds tolerance",
                max_abs_error=max_error,
                max_abs_error_tolerance=tolerance,
            ))
    if "prepacked_max_abs_error" in record:
        prepacked_max_error = _float_record_value(record, "prepacked_max_abs_error")
        if prepacked_max_error is None or prepacked_max_error < 0.0:
            failures.append(_gate_failure(record, "invalid prepacked grouped_gemm correctness telemetry"))
        elif prepacked_max_error > tolerance:
            failures.append(_gate_failure(
                record,
                "prepacked grouped_gemm correctness error exceeds tolerance",
                prepacked_max_abs_error=prepacked_max_error,
                max_abs_error_tolerance=tolerance,
            ))
    if "raw_prepacked_max_abs_error" in record:
        raw_prepacked_max_error = _float_record_value(record, "raw_prepacked_max_abs_error")
        if raw_prepacked_max_error is None or raw_prepacked_max_error < 0.0:
            failures.append(_gate_failure(record, "invalid raw prepacked grouped_gemm correctness telemetry"))
        elif raw_prepacked_max_error > tolerance:
            failures.append(_gate_failure(
                record,
                "raw prepacked grouped_gemm correctness error exceeds tolerance",
                raw_prepacked_max_abs_error=raw_prepacked_max_error,
                max_abs_error_tolerance=tolerance,
            ))
    return failures


def _selected_grouped_gemm_speedup(record):
    if record.get("raw_prepacked_speedup_vs_loop") is not None:
        return "raw_prepacked_speedup_vs_loop", _float_record_value(record, "raw_prepacked_speedup_vs_loop")
    if record.get("prepacked_speedup_vs_loop") is not None:
        return "prepacked_speedup_vs_loop", _float_record_value(record, "prepacked_speedup_vs_loop")
    return "speedup_vs_loop", _float_record_value(record, "speedup_vs_loop")


def _validate_grouped_gemm_fallback_metadata(record):
    implementation = str(record.get("grouped_gemm_implementation") or "")
    requires_fallback = record.get("requires_grouped_gemm_fallback")
    fallback_reason = record.get("fallback_reason")
    fallback_from = record.get("fallback_from")
    fallback_to = record.get("fallback_to")
    failures = []
    if implementation.startswith("fallback_") and requires_fallback is not True:
        failures.append(_gate_failure(
            record,
            "fallback flag does not match grouped_gemm implementation",
            grouped_gemm_implementation=implementation,
            requires_grouped_gemm_fallback=requires_fallback,
        ))
    if implementation.startswith("backend_") and requires_fallback is not False:
        failures.append(_gate_failure(
            record,
            "fallback flag does not match grouped_gemm implementation",
            grouped_gemm_implementation=implementation,
            requires_grouped_gemm_fallback=requires_fallback,
        ))
    if requires_fallback is True and not fallback_reason:
        failures.append(_gate_failure(record, "missing grouped_gemm fallback reason"))
    if requires_fallback is True and (fallback_from != "grouped_gemm" or not fallback_to):
        failures.append(_gate_failure(
            record,
            "missing grouped_gemm fallback route",
            fallback_from=fallback_from,
            fallback_to=fallback_to,
        ))
    if requires_fallback is False and fallback_reason:
        failures.append(_gate_failure(
            record,
            "unexpected grouped_gemm fallback reason",
            fallback_reason=fallback_reason,
        ))
    if requires_fallback is False and (fallback_from is not None or fallback_to is not None):
        failures.append(_gate_failure(
            record,
            "unexpected grouped_gemm fallback route",
            fallback_from=fallback_from,
            fallback_to=fallback_to,
        ))
    return failures


def _validate_grouped_gemm_bucket_telemetry(record):
    try:
        bucket_task_counts = [int(value) for value in record.get("bucket_task_counts")]
    except Exception:
        return [_gate_failure(record, "invalid bucket task counts")]

    failures = []
    if any(value < 0 for value in bucket_task_counts):
        failures.append(_gate_failure(
            record,
            "invalid bucket task counts",
            bucket_task_counts=bucket_task_counts,
        ))
        return failures

    num_shape_buckets = int(record.get("num_shape_buckets") or 0)
    if len(bucket_task_counts) != num_shape_buckets:
        failures.append(_gate_failure(
            record,
            "bucket task counts do not match shape bucket count",
            num_shape_buckets=num_shape_buckets,
            bucket_task_counts=bucket_task_counts,
        ))

    num_grouped_tasks = int(record.get("num_grouped_tasks") or 0)
    if sum(bucket_task_counts) != num_grouped_tasks:
        failures.append(_gate_failure(
            record,
            "bucket task counts do not match grouped task total",
            num_grouped_tasks=num_grouped_tasks,
            bucket_task_counts=bucket_task_counts,
        ))
    return failures


def _validate_grouped_gemm_shape_bucket_telemetry(record):
    shape_buckets = record.get("shape_buckets")
    if not isinstance(shape_buckets, (list, tuple)):
        return [_gate_failure(record, "invalid shape bucket profiles")]

    failures = []
    num_shape_buckets = int(record.get("num_shape_buckets") or 0)
    if len(shape_buckets) != num_shape_buckets:
        failures.append(_gate_failure(
            record,
            "shape bucket profiles do not match shape bucket count",
            num_shape_buckets=num_shape_buckets,
            shape_bucket_count=len(shape_buckets),
        ))
        return failures

    task_counts = []
    batched_task_count = 0
    loop_task_count = 0
    for bucket in shape_buckets:
        if not isinstance(bucket, dict):
            failures.append(_gate_failure(record, "invalid shape bucket profiles"))
            return failures
        missing = [
            field
            for field in (
                "dtype_a",
                "dtype_b",
                "batch_shape",
                "m",
                "n",
                "k",
                "trans_a",
                "trans_b",
                "conj_a",
                "conj_b",
                "execution",
                "task_count",
            )
            if field not in bucket
        ]
        if missing:
            failures.append(_gate_failure(
                record,
                "shape bucket profile missing fields",
                missing_fields=missing,
            ))
            return failures
        try:
            dims = (int(bucket["m"]), int(bucket["n"]), int(bucket["k"]))
            task_count = int(bucket["task_count"])
            batch_shape = [int(dim) for dim in bucket.get("batch_shape") or []]
        except Exception:
            failures.append(_gate_failure(record, "invalid shape bucket profiles"))
            return failures
        if any(dim <= 0 for dim in dims) or any(dim < 0 for dim in batch_shape) or task_count <= 0:
            failures.append(_gate_failure(record, "invalid shape bucket profiles"))
            return failures
        execution = str(bucket.get("execution") or "")
        if execution not in ("batched", "loop"):
            failures.append(_gate_failure(
                record,
                "invalid shape bucket execution",
                execution=execution,
            ))
            return failures
        task_counts.append(task_count)
        if execution == "batched":
            batched_task_count += task_count
        else:
            loop_task_count += task_count

    if task_counts != [int(value) for value in record.get("bucket_task_counts")]:
        failures.append(_gate_failure(
            record,
            "shape bucket task counts do not match bucket task counts",
            shape_bucket_task_counts=task_counts,
            bucket_task_counts=[int(value) for value in record.get("bucket_task_counts")],
        ))
    if batched_task_count and int(record.get("num_batched_gemm") or 0) <= 0:
        failures.append(_gate_failure(record, "shape bucket execution missing batched GEMM count"))
    if loop_task_count != int(record.get("num_gemm") or 0):
        failures.append(_gate_failure(
            record,
            "shape bucket loop task count does not match loop GEMM count",
            shape_bucket_loop_tasks=loop_task_count,
            num_gemm=int(record.get("num_gemm") or 0),
        ))
    return failures


def evaluate_grouped_gemm_gate(
    records,
    *,
    min_speedup=1.0,
    require_backends=(),
    require_native=True,
    require_backend_owned=None,
    require_gpu_count=None,
    require_grouped_dims=(),
    require_grouped_shape_mode=None,
    require_grouped_source=None,
):
    """Evaluate whether grouped GEMM benchmark records satisfy the Phase 5 gate."""

    if require_backend_owned is None:
        require_backend_owned = bool(require_native)
    grouped_records = _grouped_records(records)
    required = tuple(parse_backend_names(",".join(require_backends))) if require_backends else ()
    required_gpu_indices = (
        tuple(range(int(require_gpu_count)))
        if require_gpu_count is not None
        else ()
    )
    required_grouped_dims = tuple(int(dim) for dim in (require_grouped_dims or ()))
    required_grouped_shape_mode = (
        None if require_grouped_shape_mode is None else str(require_grouped_shape_mode)
    )
    if required_grouped_shape_mode is not None and required_grouped_shape_mode not in GROUPED_SHAPE_MODES:
        raise ValueError("unknown grouped_shape_mode {0!r}".format(require_grouped_shape_mode))
    required_grouped_source = (
        None if require_grouped_source is None else str(require_grouped_source)
    )
    if required_grouped_source is not None and required_grouped_source not in GROUPED_SOURCES:
        raise ValueError("unknown grouped_source {0!r}".format(require_grouped_source))
    checked_records = [
        record for record in grouped_records
        if not required or record.get("backend") in required
        if required_grouped_shape_mode is None
        or record.get("grouped_shape_mode") == required_grouped_shape_mode
        if required_grouped_source is None
        or record.get("grouped_source") == required_grouped_source
    ]
    failures = []
    speedup_metrics = []

    for backend_name in required:
        if not any(record.get("backend") == backend_name for record in grouped_records):
            if not required_gpu_indices:
                failures.append({
                    "backend": backend_name,
                    "device": None,
                    "reason": "missing grouped_gemm record",
                })
        for gpu_index in required_gpu_indices:
            if not any(
                record.get("backend") == backend_name
                and _device_index(record.get("device")) == int(gpu_index)
                for record in grouped_records
            ):
                failures.append({
                    "backend": backend_name,
                    "device": "cuda:{0}".format(int(gpu_index)),
                    "reason": "missing grouped_gemm record",
                })
        if required_grouped_dims:
            backend_records = [
                record for record in grouped_records
                if record.get("backend") == backend_name
            ]
            if not backend_records:
                continue
            if required_gpu_indices:
                for gpu_index in required_gpu_indices:
                    device_records = [
                        record for record in backend_records
                        if _device_index(record.get("device")) == int(gpu_index)
                    ]
                    if not device_records:
                        continue
                    for dim in required_grouped_dims:
                        if not any(_record_grouped_dim(record) == int(dim) for record in device_records):
                            failures.append({
                                "backend": backend_name,
                                "device": "cuda:{0}".format(int(gpu_index)),
                                "grouped_dim": int(dim),
                                "reason": "missing grouped_gemm dimension record",
                            })
            else:
                for dim in required_grouped_dims:
                    if not any(_record_grouped_dim(record) == int(dim) for record in backend_records):
                        failures.append({
                            "backend": backend_name,
                            "device": None,
                            "grouped_dim": int(dim),
                            "reason": "missing grouped_gemm dimension record",
                        })
        if required_grouped_shape_mode is not None:
            backend_records = [
                record for record in grouped_records
                if record.get("backend") == backend_name
            ]
            if not backend_records:
                continue
            if required_gpu_indices:
                for gpu_index in required_gpu_indices:
                    device_records = [
                        record for record in backend_records
                        if _device_index(record.get("device")) == int(gpu_index)
                    ]
                    if not device_records:
                        continue
                    if not any(
                        record.get("grouped_shape_mode") == required_grouped_shape_mode
                        for record in device_records
                    ):
                        failures.append({
                            "backend": backend_name,
                            "device": "cuda:{0}".format(int(gpu_index)),
                            "reason": "missing grouped_gemm shape mode record",
                            "grouped_shape_mode": required_grouped_shape_mode,
                        })
            elif not any(
                record.get("grouped_shape_mode") == required_grouped_shape_mode
                for record in backend_records
            ):
                failures.append({
                    "backend": backend_name,
                    "device": backend_records[0].get("device"),
                    "reason": "missing grouped_gemm shape mode record",
                    "grouped_shape_mode": required_grouped_shape_mode,
                })
        if required_grouped_source is not None:
            backend_records = [
                record for record in grouped_records
                if record.get("backend") == backend_name
            ]
            if not backend_records:
                continue
            if required_gpu_indices:
                for gpu_index in required_gpu_indices:
                    device_records = [
                        record for record in backend_records
                        if _device_index(record.get("device")) == int(gpu_index)
                    ]
                    if not device_records:
                        continue
                    if not any(
                        record.get("grouped_source") == required_grouped_source
                        for record in device_records
                    ):
                        failures.append({
                            "backend": backend_name,
                            "device": "cuda:{0}".format(int(gpu_index)),
                            "reason": "missing grouped_gemm source record",
                            "grouped_source": required_grouped_source,
                        })
            elif not any(
                record.get("grouped_source") == required_grouped_source
                for record in backend_records
            ):
                failures.append({
                    "backend": backend_name,
                    "device": backend_records[0].get("device"),
                    "reason": "missing grouped_gemm source record",
                    "grouped_source": required_grouped_source,
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
        missing_telemetry = _missing_grouped_gemm_telemetry(record)
        if missing_telemetry:
            failures.append(_gate_failure(
                record,
                "missing grouped_gemm telemetry",
                missing_fields=missing_telemetry,
            ))
            continue
        bucket_telemetry_failures = _validate_grouped_gemm_bucket_telemetry(record)
        if bucket_telemetry_failures:
            failures.extend(bucket_telemetry_failures)
            continue
        shape_bucket_failures = _validate_grouped_gemm_shape_bucket_telemetry(record)
        if shape_bucket_failures:
            failures.extend(shape_bucket_failures)
            continue
        timing_failures = _validate_grouped_gemm_timing(record)
        if timing_failures:
            failures.extend(timing_failures)
            continue
        prepacked_timing_failures = _validate_prepacked_grouped_gemm_timing(record)
        if prepacked_timing_failures:
            failures.extend(prepacked_timing_failures)
            continue
        raw_prepacked_timing_failures = _validate_raw_prepacked_grouped_gemm_timing(record)
        if raw_prepacked_timing_failures:
            failures.extend(raw_prepacked_timing_failures)
            continue
        correctness_failures = _validate_grouped_gemm_correctness(record)
        if correctness_failures:
            failures.extend(correctness_failures)
            continue
        fallback_metadata_failures = _validate_grouped_gemm_fallback_metadata(record)
        if fallback_metadata_failures:
            failures.extend(fallback_metadata_failures)
            continue
        if bool(require_backend_owned) and record.get("supports_grouped_gemm") is not True:
            failures.append(_gate_failure(record, "backend-owned grouped_gemm unavailable"))
            continue
        speedup_metric, speedup = _selected_grouped_gemm_speedup(record)
        speedup_metrics.append(speedup_metric)
        if speedup is None or float(speedup) < float(min_speedup):
            failures.append(_gate_failure(
                record,
                "speedup below threshold",
                speedup_metric=speedup_metric,
                selected_speedup_vs_loop=None if speedup is None else float(speedup),
                min_speedup=float(min_speedup),
            ))

    return {
        "operation": "grouped_gemm_gate",
        "status": "failed" if failures else "passed",
        "min_speedup": float(min_speedup),
        "require_native": bool(require_native),
        "require_backend_owned": bool(require_backend_owned),
        "required_gpu_count": None if require_gpu_count is None else int(require_gpu_count),
        "required_gpu_indices": [int(index) for index in required_gpu_indices],
        "required_grouped_dims": [int(dim) for dim in required_grouped_dims],
        "required_grouped_shape_mode": required_grouped_shape_mode,
        "required_grouped_source": required_grouped_source,
        "checked_count": int(len(checked_records)),
        "speedup_metrics": sorted(set(speedup_metrics)),
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
    grouped_dim=None,
    grouped_dims=None,
    grouped_shape_mode="same",
    grouped_source="descriptors",
    warmup=0,
    trials=1,
):
    if grouped_dim is not None and grouped_dims is not None:
        raise ValueError("grouped_dim and grouped_dims are mutually exclusive")
    grouped_dim_values = (
        tuple(int(value) for value in grouped_dims)
        if grouped_dims is not None
        else (small_dim if grouped_dim is None else int(grouped_dim),)
    )
    backend_name = normalize_backend_name(backend_name)
    gpu_count = detect_gpu_count()
    if not is_backend_available(backend_name):
        return [_error_record(backend_name, device, "backend_import", gpu_count, "skipped", "backend is not available")]
    try:
        backend = create_backend(backend_name, config=BackendConfig(device=device))
    except Exception as exc:
        message = str(exc)
        status = "skipped" if "does not support device" in message else "error"
        return [_error_record(backend_name, device, "backend_create", gpu_count, status, exc)]

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
    ):
        try:
            records.append(fn())
        except Exception as exc:
            records.append(_error_record(backend_name, device, operation, gpu_count, "error", exc))
    grouped_dim_count = len(grouped_dim_values)
    for index, dim in enumerate(grouped_dim_values):
        try:
            record = _grouped_gemm_smoke(
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
                grouped_dim=dim,
                grouped_shape_mode=grouped_shape_mode,
                grouped_source=grouped_source,
            )
            if grouped_dim_count > 1:
                record["grouped_dim_index"] = int(index)
                record["grouped_dim_count"] = int(grouped_dim_count)
            records.append(record)
        except Exception as exc:
            records.append(_error_record(backend_name, device, "grouped_gemm", gpu_count, "error", exc))
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
    parser.add_argument(
        "--all-gpus",
        action="store_true",
        help="Run GPU smoke once per visible GPU using explicit cuda:N devices.",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--small-dim", type=int, default=32)
    parser.add_argument("--medium-dim", type=int, default=128)
    parser.add_argument(
        "--grouped-dim",
        type=int,
        default=None,
        help="Matrix dimension for grouped_gemm smoke; defaults to --small-dim.",
    )
    parser.add_argument(
        "--grouped-dims",
        type=parse_int_list,
        default=None,
        help="Comma-separated grouped_gemm dimensions to sweep, e.g. 16,32,64.",
    )
    parser.add_argument(
        "--grouped-shape-mode",
        choices=GROUPED_SHAPE_MODES,
        default="same",
        help="Grouped GEMM task shape profile: same or block_heavy ragged buckets.",
    )
    parser.add_argument(
        "--grouped-source",
        choices=GROUPED_SOURCES,
        default="descriptors",
        help="Grouped GEMM task source: direct descriptors or BlockTensor lowering.",
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--pack-threshold", type=int, default=4)
    parser.add_argument("--output", default=None, help="Optional JSONL output path.")
    parser.add_argument(
        "--nvidia-dmon-output",
        default=None,
        help="Optional nvidia-smi dmon output path for GPU resource telemetry.",
    )
    parser.add_argument(
        "--nvidia-dmon-fields",
        default="pucvmte",
        help="nvidia-smi dmon field selector used with --nvidia-dmon-output.",
    )
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
    parser.add_argument(
        "--require-gpu-count",
        type=int,
        default=None,
        help="Require grouped_gemm records for cuda:0..cuda:N-1 in the grouped gate.",
    )
    parser.add_argument(
        "--require-grouped-dims",
        type=parse_int_list,
        default=None,
        help="Require grouped_gemm records for each comma-separated dimension.",
    )
    parser.add_argument(
        "--require-grouped-shape-mode",
        choices=GROUPED_SHAPE_MODES,
        default=None,
        help="Require grouped_gemm records from a specific shape profile.",
    )
    parser.add_argument(
        "--require-grouped-source",
        choices=GROUPED_SOURCES,
        default=None,
        help="Require grouped_gemm records from a specific task source.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.all_gpus and args.gpu_index is not None:
        parser.error("--all-gpus cannot be combined with --gpu-index")
    if args.grouped_dim is not None and args.grouped_dims is not None:
        parser.error("--grouped-dim cannot be combined with --grouped-dims")
    if args.gpu_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    if args.all_gpus:
        gpu_count = detect_gpu_count()
        devices = ["cuda:{0}".format(index) for index in range(gpu_count)]
    else:
        devices = [args.device]

    all_records = []
    gate_summary = None
    with nvidia_dmon_monitor(args.nvidia_dmon_output, fields=args.nvidia_dmon_fields):
        for device in devices:
            for backend_name in parse_backend_names(args.backends):
                records = run_backend_smoke(
                    backend_name,
                    device=device,
                    batch=args.batch,
                    small_dim=args.small_dim,
                    medium_dim=args.medium_dim,
                    grouped_dim=args.grouped_dim,
                    grouped_dims=args.grouped_dims,
                    grouped_shape_mode=args.grouped_shape_mode,
                    grouped_source=args.grouped_source,
                    repeat=args.repeat,
                    pack_threshold=args.pack_threshold,
                    warmup=args.warmup,
                    trials=args.trials,
                )
                all_records.extend(records)

        if args.grouped_speedup_gate:
            required = tuple(parse_backend_names(args.require_grouped_backends)) if args.require_grouped_backends else ()
            gate_summary = evaluate_grouped_gemm_gate(
                all_records,
                min_speedup=args.min_grouped_speedup,
                require_backends=required,
                require_gpu_count=args.require_gpu_count,
                require_grouped_dims=args.require_grouped_dims or (),
                require_grouped_shape_mode=args.require_grouped_shape_mode,
                require_grouped_source=args.require_grouped_source,
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
