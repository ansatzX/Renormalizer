# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""CuPy backend adapter with lazy package loading and indexed CUDA devices."""

import functools
import logging
from types import ModuleType

import numpy as np

from renormalizer.backend.abstract import AbstractBackend


logger = logging.getLogger(__name__)


def _import_cupy():
    try:
        import cupy
    except ImportError as error:
        raise ImportError(
            "CuPy is not installed. Install cupy or select the NumPy backend."
        ) from error
    return cupy


class _DeviceBoundNamespace:
    """Delegate CuPy namespace calls inside one configured device context."""

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


class CupyBackend(AbstractBackend):
    name = "cupy"
    opt_einsum_name = "cupy"
    supports_gpu = True
    host_array_types = (np.ndarray,)

    def __init__(self, config):
        cp = _import_cupy()
        if config.device != "gpu" and not config.device.startswith("cuda:"):
            raise ValueError("backend 'cupy' only supports device='gpu' or 'cuda:N'")

        self._cupy = cp
        self.ndarray = (np.ndarray, cp.ndarray)
        self.device_array_types = (cp.ndarray,)
        self.memory_errors = (MemoryError, cp.cuda.memory.OutOfMemoryError)
        super().__init__(config)
        self._device_index = self._resolve_device_index()
        cp.cuda.Device(self._device_index).use()
        self.array_namespace = _DeviceBoundNamespace(cp, self._on_device)

    def _resolve_device_index(self):
        if self.device == "gpu":
            index = int(self._cupy.cuda.runtime.getDevice())
        else:
            index = int(self.device.split(":", 1)[1])
        count = int(self._cupy.cuda.runtime.getDeviceCount())
        if index >= count:
            raise ValueError(
                "cupy backend CUDA device index {} is out of range for {} visible device(s)".format(
                    index, count
                )
            )
        return index

    def _on_device(self, function, *args, **kwargs):
        with self._cupy.cuda.Device(self._device_index):
            return function(*args, **kwargs)

    def current_device(self):
        return "cuda:{}".format(self._device_index)

    def array(self, *args, **kwargs):
        return self._on_device(self._cupy.array, *args, **kwargs)

    def asarray(self, *args, **kwargs):
        return self._on_device(self._cupy.asarray, *args, **kwargs)

    def from_numpy(self, value):
        if value is None:
            return None
        return self.asarray(value)

    def to_numpy(self, value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value
        return self._cupy.asnumpy(value)

    def to_host(self, value):
        return self.to_numpy(value)

    def to_backend(self, value):
        if value is None:
            return None
        return self.asarray(value)

    def tensordot(self, a, b, axes=2):
        return self._on_device(self._cupy.tensordot, a, b, axes=axes)

    def transpose(self, value, axes=None):
        return self._on_device(self._cupy.transpose, value, axes=axes)

    def matmul(self, a, b, *args, **kwargs):
        return self._on_device(self._cupy.matmul, a, b, *args, **kwargs)

    def sync(self):
        self._cupy.cuda.Device(self._device_index).synchronize()

    def free_all_blocks(self):
        with self._cupy.cuda.Device(self._device_index):
            self._cupy.get_default_memory_pool().free_all_blocks()

    def log_memory_usage(self, header=""):
        from renormalizer.utils.utils import sizeof_fmt

        with self._cupy.cuda.Device(self._device_index):
            memory_pool = self._cupy.get_default_memory_pool()
            logger.info(
                "%s GPU memory used/total: %s/%s",
                header,
                sizeof_fmt(memory_pool.used_bytes()),
                sizeof_fmt(memory_pool.total_bytes()),
            )
