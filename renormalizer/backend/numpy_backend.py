# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""NumPy backend adapter."""

import numpy as np

from renormalizer.backend.abstract import AbstractBackend


class NumpyBackend(AbstractBackend):
    name = "numpy"
    array_namespace = np
    ndarray = np.ndarray
    host_array_types = (np.ndarray,)
    device_array_types = ()
    memory_errors = (MemoryError,)
    opt_einsum_name = "numpy"
    supports_execution_ir = True
    supports_batched_matmul = True
    supports_grouped_gemm = True

    def array(self, *args, **kwargs):
        if kwargs.get("copy", True) is None:
            kwargs = dict(kwargs)
            kwargs.pop("copy")
        return np.array(*args, **kwargs)

    def asarray(self, *args, **kwargs):
        return np.asarray(*args, **kwargs)

    def from_numpy(self, value):
        if value is None:
            return None
        return np.asarray(value)

    def to_numpy(self, value):
        if value is None:
            return None
        return np.asarray(value)

    def to_host(self, value):
        return self.to_numpy(value)

    def to_backend(self, value, *, dtype=None):
        if value is None:
            return None
        return np.asarray(value, dtype=dtype)

    def matmul(self, a, b, *, stream=None, workspace=None):
        if stream is not None:
            raise ValueError("NumPy execution does not accept a stream")
        return np.matmul(a, b)

    def batched_matmul(self, a, b, *, stream=None, workspace=None):
        if stream is not None:
            raise ValueError("NumPy execution does not accept a stream")
        return np.matmul(a, b)

    def grouped_gemm(
        self, descriptors, tensors, *, stream=None, workspace=None, policy="direct"
    ):
        from renormalizer.backend._gemm.executor import execute_grouped_gemm

        return execute_grouped_gemm(
            self,
            descriptors,
            tensors,
            stream=stream,
            workspace=workspace,
            policy=policy,
        )

    def _validate_execution_array(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError("NumPy execution binding must be a NumPy host array")
