# -*- coding: utf-8 -*-

"""CuPy backend — delegates to cupy if installed, raises clear error if not."""

import logging

import numpy as np

from renormalizer.backend.abstract import AbstractBackend
from renormalizer.backend.execution import DeviceSpec

logger = logging.getLogger(__name__)

_cupy = None
_cupy_available = False


def _try_import_cupy():
    global _cupy, _cupy_available
    try:
        import cupy as cp
        _cupy = cp
        _cupy_available = True
    except ImportError:
        _cupy_available = False


_try_import_cupy()


class CupyBackend(AbstractBackend):
    name = "cupy"
    supported_device_kinds = ("gpu",)
    available_device_kinds = ("gpu",)
    supports_cpu = False
    ndarray = (np.ndarray,)  # will be extended in __init__ if cupy available
    memory_errors = (MemoryError,)
    opt_einsum_name = "cupy"
    supports_gpu = True
    supports_grouped_gemm = True
    host_array_types = (np.ndarray,)

    def __init__(self, config=None):
        if not _cupy_available:
            raise ImportError(
                "CuPy is not installed. Install cupy or select another backend."
            )
        super().__init__(config=config)
        self._set_configured_device(("gpu",), default="gpu", available=("gpu",))
        self.array_namespace = _cupy
        self.ndarray = (np.ndarray, _cupy.ndarray)
        self.device_array_types = (_cupy.ndarray,)
        self.memory_errors = (MemoryError, _cupy.cuda.memory.OutOfMemoryError)
        self._activate_configured_device()

        self.linalg = _cupy.linalg
        self.random = _cupy.random

    def __getattr__(self, name):
        return getattr(_cupy, name)

    def _configured_cuda_index(self):
        spec = self.current_device()
        if spec.kind == "cuda" and spec.index is not None:
            index = int(spec.index)
            count = int(_cupy.cuda.runtime.getDeviceCount())
            if index < 0 or index >= count:
                raise ValueError(
                    "cupy backend CUDA device index {0} is out of range for {1} visible device(s)"
                    .format(index, count)
                )
            return index
        return None

    def _activate_configured_device(self):
        index = self._configured_cuda_index()
        if index is not None:
            _cupy.cuda.Device(index).use()

    def _on_configured_device(self, fn, *args, **kwargs):
        index = self._configured_cuda_index()
        if index is None:
            return fn(*args, **kwargs)
        with _cupy.cuda.Device(index):
            return fn(*args, **kwargs)

    def set_device(self, device):
        super().set_device(device)
        self._activate_configured_device()

    def device_count(self):
        return int(_cupy.cuda.runtime.getDeviceCount())

    def _device_spec_for_array(self, x):
        if isinstance(x, _cupy.ndarray):
            index = int(x.device.id)
            return DeviceSpec(kind="cuda", index=index, visible_id=str(index))
        return super()._device_spec_for_array(x)

    def array(self, *args, **kwargs):
        return self._on_configured_device(_cupy.array, *args, **kwargs)

    def asarray(self, *args, **kwargs):
        return self._on_configured_device(_cupy.asarray, *args, **kwargs)

    def from_numpy(self, x):
        return self._on_configured_device(_cupy.asarray, x)

    def numpy(self, x):
        return self.to_numpy(x)

    def to_numpy(self, x):
        """Convert ``x`` to a NumPy array on the host."""
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        return _cupy.asnumpy(x)

    def to_host(self, x):
        """Convert ``x`` to a host NumPy array."""
        return self.to_numpy(x)

    def to_backend(self, x):
        """Convert ``x`` to a CuPy array on the active device."""
        return self.asarray(x)

    def free_all_blocks(self):
        mempool = _cupy.get_default_memory_pool()
        mempool.free_all_blocks()

    def log_memory_usage(self, header=""):
        from renormalizer.utils.utils import sizeof_fmt
        mempool = _cupy.get_default_memory_pool()
        logger.info(
            f"{header} GPU memory used/Total: "
            f"{sizeof_fmt(mempool.used_bytes())}/{sizeof_fmt(mempool.total_bytes())}"
        )

    def sync(self):
        index = self._configured_cuda_index()
        if index is None:
            _cupy.cuda.Device().synchronize()
        else:
            _cupy.cuda.Device(index).synchronize()
