# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GemmTask:
    A: Any
    B: Any
    C: Any | None = None
    trans_a: bool = False
    trans_b: bool = False
    conj_a: bool = False
    conj_b: bool = False
    alpha: complex | float = 1.0
    beta: complex | float = 0.0
    tag: Any = None


@dataclass(frozen=True)
class GroupedGemmStats:
    task_count: int
    shape_bucket_count: int
    batched_bucket_count: int
    loop_bucket_count: int
    batched_task_count: int
    loop_task_count: int
    bucket_task_counts: tuple[int, ...]
    flops: int
    read_bytes: int
    write_bytes: int
    copy_bytes: int
    workspace_bytes: int = 0


def apply_gemm_flags(x, *, trans=False, conj=False, xp=None):
    if xp is None:
        import numpy as xp

    if conj:
        x = xp.conj(x)
    if trans:
        x = xp.swapaxes(x, -1, -2)
    return x


def array_nbytes(x) -> int:
    nbytes = getattr(x, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    numel = getattr(x, "numel", None)
    element_size = getattr(x, "element_size", None)
    if callable(numel) and callable(element_size):
        return int(numel() * element_size())
    shape = getattr(x, "shape", ())
    size = 1
    for dim in shape:
        size *= int(dim)
    dtype = getattr(x, "dtype", None)
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is not None:
        return int(size * itemsize)
    return 0


def gemm_task_key(task: GemmTask, *, xp=None):
    a = apply_gemm_flags(task.A, trans=task.trans_a, conj=task.conj_a, xp=xp)
    b = apply_gemm_flags(task.B, trans=task.trans_b, conj=task.conj_b, xp=xp)
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError("grouped_gemm tasks must contain rank-2 matrices")
    m, k_left = int(a.shape[-2]), int(a.shape[-1])
    k_right, n = int(b.shape[-2]), int(b.shape[-1])
    if k_left != k_right:
        raise ValueError("GEMM task has incompatible contracted dimensions")
    return (
        str(getattr(a, "dtype", None)),
        str(getattr(b, "dtype", None)),
        m,
        n,
        k_left,
        bool(task.trans_a),
        bool(task.trans_b),
        bool(task.conj_a),
        bool(task.conj_b),
    )


def group_tasks_by_shape(tasks, *, xp=None):
    buckets = defaultdict(list)
    for task in tasks:
        buckets[gemm_task_key(task, xp=xp)].append(task)
    return dict(buckets)


def _gemm_shape(task: GemmTask, *, xp=None):
    a = apply_gemm_flags(task.A, trans=task.trans_a, conj=task.conj_a, xp=xp)
    b = apply_gemm_flags(task.B, trans=task.trans_b, conj=task.conj_b, xp=xp)
    return int(a.shape[-2]), int(b.shape[-1]), int(a.shape[-1])


def should_batch(tasks, *, xp=None, pack_threshold=4, flop_copy_ratio=10):
    tasks = list(tasks)
    if len(tasks) < pack_threshold:
        return False
    flops = 0
    copy_bytes = 0
    for task in tasks:
        m, n, k = _gemm_shape(task, xp=xp)
        flops += 2 * m * n * k
        copy_bytes += array_nbytes(task.A)
        copy_bytes += array_nbytes(task.B)
        if task.C is not None:
            copy_bytes += array_nbytes(task.C)
    if copy_bytes and flops < flop_copy_ratio * copy_bytes:
        return False
    return True


def grouped_gemm_stats(tasks, *, xp=None, pack_threshold=4) -> GroupedGemmStats:
    tasks = list(tasks)
    buckets = group_tasks_by_shape(tasks, xp=xp)
    bucket_task_counts = tuple(sorted(len(group) for group in buckets.values()))
    batched_bucket_count = 0
    loop_bucket_count = 0
    batched_task_count = 0
    loop_task_count = 0
    flops = 0
    read_bytes = 0
    write_bytes = 0
    copy_bytes = 0
    for group in buckets.values():
        group_batched = should_batch(group, xp=xp, pack_threshold=pack_threshold)
        if group_batched:
            batched_bucket_count += 1
            batched_task_count += len(group)
        else:
            loop_bucket_count += 1
            loop_task_count += len(group)
        for task in group:
            m, n, k = _gemm_shape(task, xp=xp)
            flops += int(2 * m * n * k)
            read_bytes += array_nbytes(task.A) + array_nbytes(task.B)
            write_bytes += int(m * n * max(array_nbytes(task.A) // max(m * k, 1), array_nbytes(task.B) // max(k * n, 1)))
            if group_batched:
                copy_bytes += array_nbytes(task.A) + array_nbytes(task.B)
                if task.C is not None:
                    copy_bytes += array_nbytes(task.C)
    return GroupedGemmStats(
        task_count=len(tasks),
        shape_bucket_count=len(buckets),
        batched_bucket_count=batched_bucket_count,
        loop_bucket_count=loop_bucket_count,
        batched_task_count=batched_task_count,
        loop_task_count=loop_task_count,
        bucket_task_counts=bucket_task_counts,
        flops=int(flops),
        read_bytes=int(read_bytes),
        write_bytes=int(write_bytes),
        copy_bytes=int(copy_bytes),
    )


def run_gemm_task(task: GemmTask, *, xp=None):
    if xp is None:
        import numpy as xp

    a = apply_gemm_flags(task.A, trans=task.trans_a, conj=task.conj_a, xp=xp)
    b = apply_gemm_flags(task.B, trans=task.trans_b, conj=task.conj_b, xp=xp)
    result = xp.matmul(a, b)
    if task.alpha != 1.0:
        result = task.alpha * result
    if task.C is not None:
        if task.beta != 0.0:
            result = result + task.beta * task.C
        task.C[...] = result
        return task.C
    return result


def grouped_gemm_bucketed(tasks, *, xp=None, pack_threshold=4):
    if xp is None:
        import numpy as xp

    tasks = list(tasks)
    buckets = group_tasks_by_shape(tasks, xp=xp)
    results_by_task = {}
    for group in buckets.values():
        if not should_batch(group, xp=xp, pack_threshold=pack_threshold):
            for task in group:
                results_by_task[id(task)] = run_gemm_task(task, xp=xp)
            continue

        a_pack = xp.stack([
            apply_gemm_flags(task.A, trans=task.trans_a, conj=task.conj_a, xp=xp)
            for task in group
        ], axis=0)
        b_pack = xp.stack([
            apply_gemm_flags(task.B, trans=task.trans_b, conj=task.conj_b, xp=xp)
            for task in group
        ], axis=0)
        c_pack = xp.matmul(a_pack, b_pack)

        for index, task in enumerate(group):
            result = c_pack[index]
            if task.alpha != 1.0:
                result = task.alpha * result
            if task.C is not None:
                if task.beta != 0.0:
                    result = result + task.beta * task.C
                task.C[...] = result
                results_by_task[id(task)] = task.C
            else:
                results_by_task[id(task)] = result
    return [results_by_task[id(task)] for task in tasks]


def grouped_gemm_fallback(tasks, *, xp=None, pack_threshold=4):
    return grouped_gemm_bucketed(tasks, xp=xp, pack_threshold=pack_threshold)
