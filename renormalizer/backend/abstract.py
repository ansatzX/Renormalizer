# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""Shared behavior for selectable numerical backends."""

import functools
from contextlib import nullcontext
from types import ModuleType
from typing import Any

import numpy as np

from renormalizer.backend.config import BackendConfig
from renormalizer.backend.transforms import UnavailableTransforms


class _DeviceBoundNamespace:
    """Delegate namespace calls inside one configured device context."""

    def __init__(self, namespace, on_device):
        self._namespace = namespace
        self._on_device = on_device
        self._cache = {}

    def __getattr__(self, name):
        if name in self._cache:
            return self._cache[name]

        value = getattr(self._namespace, name)
        if isinstance(value, ModuleType):
            result = type(self)(value, self._on_device)
        elif callable(value) and not isinstance(value, type):
            def device_bound_call(*args, **kwargs):
                return self._on_device(value, *args, **kwargs)

            try:
                result = functools.update_wrapper(device_bound_call, value)
            except (AttributeError, TypeError):
                result = device_bound_call
        else:
            result = value
        self._cache[name] = result
        return result

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(self._namespace)))

    def __repr__(self):
        return repr(self._namespace)


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
    supports_execution_ir = False
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

    def to_backend(self, value: Any, *, dtype=None):
        if value is None:
            return None
        if dtype is not None:
            return self.asarray(value, dtype=dtype)
        if self.is_host_array(value):
            return self.from_numpy(value)
        return self.asarray(value)

    def activate(self):
        return None

    def deactivate(self):
        return None

    def current_device(self):
        return self.device

    def tensordot(self, a, b, axes=2):
        return self.array_namespace.tensordot(a, b, axes=axes)

    def transpose(self, value, axes=None):
        return self.array_namespace.transpose(value, axes=axes)

    def reshape(self, value, shape):
        return self.array_namespace.reshape(value, shape)

    def matmul(self, a, b, *, stream=None, workspace=None):
        if stream is not None:
            raise ValueError("backend {!r} does not accept a stream".format(self.name))
        return self.array_namespace.matmul(a, b)

    def execute_plan(self, plan, bindings, *, stream=None, workspace=None):
        if not self.supports_execution_ir:
            raise NotImplementedError(
                "backend {!r} does not support execution IR".format(self.name)
            )
        from renormalizer.backend._execution.executor import execute_plan

        return execute_plan(
            self, plan, bindings, stream=stream, workspace=workspace
        )

    def _validate_execution_stream(self, stream):
        if stream is not None:
            raise ValueError("backend {!r} does not accept a stream".format(self.name))

    def _execution_context(self, stream):
        self._validate_execution_stream(stream)
        return nullcontext()

    def _validate_execution_array(self, value):
        if not isinstance(value, self.ndarray):
            raise TypeError("execution binding must be a backend array")

    def _is_exact_execution_reshape(self, left, right):
        if (
            left.dtype != right.dtype
            or left.size != right.size
            or left.nbytes != right.nbytes
            or not left.flags.c_contiguous
            or not right.flags.c_contiguous
            or any(stride == 0 for stride in right.strides)
        ):
            return False

        def root_base(value):
            seen = set()
            while getattr(value, "base", None) is not None:
                if id(value) in seen:
                    break
                seen.add(id(value))
                value = value.base
            return value

        if root_base(left) is not root_base(right):
            return False
        if left.size == 0:
            return True
        return (
            left.__array_interface__["data"][0]
            == right.__array_interface__["data"][0]
        )

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
