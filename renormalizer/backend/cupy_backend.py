# -*- coding: utf-8 -*-

"""CuPy backend — delegates to cupy if installed, raises clear error if not."""

import logging
import os

import numpy as np

from renormalizer.backend.abstract import AbstractBackend

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
    ndarray = (np.ndarray,)  # will be extended in __init__ if cupy available
    memory_errors = (MemoryError,)
    opt_einsum_name = "cupy"
    supports_gpu = True
    host_array_types = (np.ndarray,)

    def __init__(self):
        if not _cupy_available:
            raise ImportError(
                "CuPy is not installed. Install cupy or select another backend."
            )
        super().__init__()
        self.array_namespace = _cupy
        self.ndarray = (np.ndarray, _cupy.ndarray)
        self.device_array_types = (_cupy.ndarray,)
        self.memory_errors = (MemoryError, _cupy.cuda.memory.OutOfMemoryError)

        self.linalg = _cupy.linalg
        self.random = _cupy.random

        if os.environ.get("RENO_FP32") is not None:
            self.use_32bits()

    def __getattr__(self, name):
        return getattr(_cupy, name)

    def array(self, *args, **kwargs):
        return _cupy.array(*args, **kwargs)

    def asarray(self, *args, **kwargs):
        return _cupy.asarray(*args, **kwargs)

    def from_numpy(self, x):
        return _cupy.asarray(x)

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
        return _cupy.asarray(x)

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
        _cupy.cuda.Device(0).synchronize()
