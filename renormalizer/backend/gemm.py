# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""Public bucketed GEMM metadata and execution helpers."""

from renormalizer.backend._gemm.descriptors import MatmulDesc
from renormalizer.backend._gemm.grouping import GemmGroupKey, group_descriptors


def batched_matmul(backend, a, b, *, stream=None, workspace=None):
    return backend.batched_matmul(a, b, stream=stream, workspace=workspace)


def grouped_gemm(
    backend, descriptors, tensors, *, stream=None, workspace=None, policy="direct"
):
    return backend.grouped_gemm(
        descriptors,
        tensors,
        stream=stream,
        workspace=workspace,
        policy=policy,
    )


__all__ = [
    "GemmGroupKey",
    "MatmulDesc",
    "batched_matmul",
    "group_descriptors",
    "grouped_gemm",
]
