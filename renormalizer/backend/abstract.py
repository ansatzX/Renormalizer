# -*- coding: utf-8 -*-

import os
from typing import Any

import numpy as _np

from renormalizer.backend.config import BackendConfig
from renormalizer.backend.mpi import SingleProcessDistributedMixin
from renormalizer.backend.transforms import UnavailableTransforms


class AbstractBackend(SingleProcessDistributedMixin):
    name = "abstract"
    supported_device_kinds = ("cpu",)
    available_device_kinds = ("cpu",)
    supports_cpu = True
    array_namespace = None
    ndarray = ()
    memory_errors = (MemoryError,)
    opt_einsum_name = "numpy"
    supports_gpu = False
    supports_autodiff = False
    supports_jit = False
    supports_sparse = False
    supports_functional_update = True
    host_array_types = (_np.ndarray,)
    device_array_types = ()

    def __init__(self, config=None):
        self.config = BackendConfig.from_config(config)
        self.device = None
        self.first_mp = False
        self._real_dtype = None
        self._complex_dtype = None
        self.transforms = UnavailableTransforms(self.name)
        self.use_64bits()
        self._set_configured_device(self.supported_device_kinds, default="cpu")
        self._apply_precision_config()

    def _apply_precision_config(self):
        if self.config.precision == 32:
            self.use_32bits()
        elif self.config.precision == 64:
            self.use_64bits()
        elif os.environ.get("RENO_FP32") is not None:
            self.use_32bits()

    def _set_configured_device(self, supported, default=None, available=None):
        supported = tuple(supported)
        available = tuple(available or supported)
        self.supported_device_kinds = supported
        self.available_device_kinds = available
        requested = self.config.device
        if requested is None:
            self.device = default or (available[0] if available else None)
            return
        if requested not in supported:
            raise ValueError(
                "{0} backend does not support device '{1}'. Supported devices: {2}."
                .format(self.name, requested, ", ".join(supported) or "none")
            )
        if requested not in available:
            raise ValueError(
                "{0} backend device '{1}' was requested but is not available. "
                "Available devices: {2}."
                .format(self.name, requested, ", ".join(available) or "none")
            )
        self.device = requested

    def use_32bits(self):
        self.dtypes = (_np.float32, _np.complex64)

    def use_64bits(self):
        self.dtypes = (_np.float64, _np.complex128)

    @property
    def is_32bits(self) -> bool:
        return self._real_dtype == _np.float32

    @property
    def real_dtype(self):
        return self._real_dtype

    @real_dtype.setter
    def real_dtype(self, tp):
        if self.first_mp:
            raise RuntimeError("Can't alter backend data type")
        self._real_dtype = tp

    @property
    def complex_dtype(self):
        return self._complex_dtype

    @complex_dtype.setter
    def complex_dtype(self, tp):
        if self.first_mp:
            raise RuntimeError("Can't alter backend data type")
        self._complex_dtype = tp

    @property
    def dtypes(self):
        return self.real_dtype, self.complex_dtype

    @dtypes.setter
    def dtypes(self, target):
        self.real_dtype, self.complex_dtype = target

    @property
    def canonical_atol(self):
        return getattr(self, "_canonical_atol", 1e-4 if self.is_32bits else 1e-8)

    @canonical_atol.setter
    def canonical_atol(self, value):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f'canonical_atol must be a non-negative number, got {value!r}')
        self._canonical_atol = value

    @property
    def canonical_rtol(self):
        return getattr(self, "_canonical_rtol", 1e-4 if self.is_32bits else 1e-5)

    @canonical_rtol.setter
    def canonical_rtol(self, value):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f'canonical_rtol must be a non-negative number, got {value!r}')
        self._canonical_rtol = value

    def numpy(self, x: Any):
        raise NotImplementedError

    def from_numpy(self, x: _np.ndarray):
        raise NotImplementedError

    def to_numpy(self, x: Any):
        return self.numpy(x)

    def to_host(self, x: Any):
        return self.to_numpy(x)

    def to_backend(self, x: Any):
        if self.is_host_array(x):
            return self.from_numpy(x)
        return self.asarray(x)

    def is_array(self, x: Any) -> bool:
        return isinstance(x, self.ndarray)

    def is_host_array(self, x: Any) -> bool:
        return isinstance(x, self.host_array_types)

    def is_device_array(self, x: Any) -> bool:
        return isinstance(x, self.device_array_types)

    def sync(self):
        return None

    def free_all_blocks(self):
        return None

    def log_memory_usage(self, header=""):
        return None

    def at_set(self, x, idx, value):
        y = self.array(x, copy=True)
        y[idx] = value
        return y

    def at_add(self, x, idx, value):
        y = self.array(x, copy=True)
        y[idx] += value
        return y

    def at_sub(self, x, idx, value):
        y = self.array(x, copy=True)
        y[idx] -= value
        return y

    def at_mul(self, x, idx, value):
        y = self.array(x, copy=True)
        y[idx] *= value
        return y
