"""Stable effective-shape bucketing for matrix multiplication descriptors."""

from dataclasses import dataclass

import numpy as np

from renormalizer.backend._gemm.descriptors import MatmulDesc


def canonical_gemm_dtype(dtype):
    try:
        normalized = np.dtype(dtype)
    except (TypeError, ValueError) as error:
        raise TypeError("GEMM dtype must be understood by NumPy") from error
    if normalized.kind not in "biufc":
        raise ValueError("GEMM dtype must be numeric")
    return normalized.name


@dataclass(frozen=True)
class GemmGroupKey:
    dtype: str
    m: int
    n: int
    k: int
    trans_a: str
    trans_b: str

    def __post_init__(self):
        object.__setattr__(self, "dtype", canonical_gemm_dtype(self.dtype))
        for field in ("m", "n", "k"):
            value = getattr(self, field)
            if type(value) is not int:
                raise TypeError("{} must be a Python integer".format(field))
            if value < 0:
                raise ValueError("{} must be non-negative".format(field))
        for field in ("trans_a", "trans_b"):
            value = getattr(self, field)
            if not isinstance(value, str) or value.upper() not in {"N", "T", "C"}:
                raise ValueError("{} must be one of N, T, or C".format(field))
            object.__setattr__(self, field, value.upper())


def group_descriptors(descriptors, *, dtype):
    """Return insertion-ordered buckets while preserving descriptor order."""
    canonical_dtype = canonical_gemm_dtype(dtype)
    buckets = {}
    for descriptor in descriptors:
        if not isinstance(descriptor, MatmulDesc):
            raise TypeError("grouped GEMM tasks must be MatmulDesc instances")
        key = GemmGroupKey(
            canonical_dtype,
            descriptor.m,
            descriptor.n,
            descriptor.k,
            descriptor.trans_a,
            descriptor.trans_b,
        )
        buckets.setdefault(key, []).append(descriptor)
    return {key: tuple(group) for key, group in buckets.items()}


__all__ = ["GemmGroupKey", "group_descriptors"]
