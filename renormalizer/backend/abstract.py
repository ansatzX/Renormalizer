# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""Shared behavior for selectable numerical backends."""

from typing import Any

import numpy as np

from renormalizer.backend.config import BackendConfig
from renormalizer.backend.transforms import UnavailableTransforms


class AbstractBackend:
    name = "abstract"
    array_namespace = np
    ndarray = np.ndarray
    memory_errors = (MemoryError,)
    opt_einsum_name = "numpy"
    supports_gpu = False
    supports_autodiff = False
    supports_jit = False
    supports_sparse = False
    supports_functional_update = True
    host_array_types = (np.ndarray,)
    device_array_types = ()

    def __init__(self, config: BackendConfig):
        self.config = config
        self.device = config.device
        self.first_mp = False
        self.transforms = UnavailableTransforms(self.name)
        self._real_dtype = None
        self._complex_dtype = None
        if config.precision == 32:
            self.use_32bits()
        else:
            self.use_64bits()

    def __getattr__(self, name: str):
        return getattr(self.array_namespace, name)

    @property
    def random(self):
        return self.array_namespace.random

    @property
    def linalg(self):
        return self.array_namespace.linalg

    def use_32bits(self):
        self.dtypes = (np.float32, np.complex64)

    def use_64bits(self):
        self.dtypes = (np.float64, np.complex128)

    @property
    def is_32bits(self) -> bool:
        return self.real_dtype == np.float32

    @property
    def real_dtype(self):
        return self._real_dtype

    @real_dtype.setter
    def real_dtype(self, dtype):
        if self.first_mp:
            raise RuntimeError("Can't alter backend data type")
        self._real_dtype = dtype

    @property
    def complex_dtype(self):
        return self._complex_dtype

    @complex_dtype.setter
    def complex_dtype(self, dtype):
        if self.first_mp:
            raise RuntimeError("Can't alter backend data type")
        self._complex_dtype = dtype

    @property
    def dtypes(self):
        return self.real_dtype, self.complex_dtype

    @dtypes.setter
    def dtypes(self, dtypes):
        self.real_dtype, self.complex_dtype = dtypes

    @property
    def canonical_atol(self):
        return getattr(self, "_canonical_atol", 1e-4 if self.is_32bits else 1e-8)

    @canonical_atol.setter
    def canonical_atol(self, value):
        self._canonical_atol = self._validate_tolerance(value)

    @property
    def canonical_rtol(self):
        return getattr(self, "_canonical_rtol", 1e-2 if self.is_32bits else 1e-5)

    @canonical_rtol.setter
    def canonical_rtol(self, value):
        self._canonical_rtol = self._validate_tolerance(value)

    @staticmethod
    def _validate_tolerance(value):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("Tolerance must be a non-negative float number")
        return value

    def to_numpy(self, value: Any):
        if value is None:
            return None
        return np.asarray(value)

    def numpy(self, value: Any):
        return self.to_numpy(value)

    def from_numpy(self, value: np.ndarray):
        if value is None:
            return None
        return np.asarray(value)

    def to_host(self, value: Any):
        if value is None:
            return None
        return self.to_numpy(value)

    def to_backend(self, value: Any):
        if value is None:
            return None
        if self.is_host_array(value):
            return self.from_numpy(value)
        return self.asarray(value)

    def current_device(self):
        return self.device

    def tensordot(self, a, b, axes=2):
        return self.array_namespace.tensordot(a, b, axes=axes)

    def transpose(self, value, axes=None):
        return self.array_namespace.transpose(value, axes=axes)

    def matmul(self, a, b, *args, **kwargs):
        return self.array_namespace.matmul(a, b, *args, **kwargs)

    def is_array(self, value: Any) -> bool:
        return isinstance(value, self.ndarray)

    def is_host_array(self, value: Any) -> bool:
        return isinstance(value, self.host_array_types)

    def is_device_array(self, value: Any) -> bool:
        return isinstance(value, self.device_array_types)

    def sync(self):
        return None

    def free_all_blocks(self):
        return None

    def log_memory_usage(self, header=""):
        return None
