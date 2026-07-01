# -*- coding: utf-8 -*-

import logging
import numpy as np

from renormalizer.backend.abstract import AbstractBackend
from renormalizer.backend.execution import BackendCopyError, CopyPolicy, parse_device_spec

logger = logging.getLogger(__name__)


class NumpyBackend(AbstractBackend):
    name = "numpy"
    array_namespace = np
    ndarray = np.ndarray
    host_array_types = (np.ndarray,)
    device_array_types = ()
    memory_errors = (MemoryError,)
    opt_einsum_name = "numpy"

    def __init__(self, config=None):
        super().__init__(config=config)
        self.linalg = np.linalg
        self.random = np.random

    def __getattr__(self, name):
        return getattr(np, name)

    def array(self, *args, **kwargs):
        if kwargs.get("copy", True) is None:
            kwargs = dict(kwargs)
            kwargs.pop("copy")
        return np.array(*args, **kwargs)

    def asarray(self, *args, **kwargs):
        return np.asarray(*args, **kwargs)

    def from_numpy(self, x):
        return np.asarray(x)

    def numpy(self, x):
        return self.to_numpy(x)

    def to_numpy(self, x):
        """Convert ``x`` to a NumPy array on the host."""
        if x is None:
            return None
        return np.asarray(x)

    def to_host(self, x):
        """Convert ``x`` to a host NumPy array."""
        return self.to_numpy(x)

    def to_backend(self, x, *, device=None, dtype=None, copy=CopyPolicy.IF_NEEDED):
        """Convert ``x`` to the NumPy backend representation."""
        if device is not None:
            spec = parse_device_spec(device)
            if spec is not None and spec.kind != "cpu":
                raise ValueError("numpy backend can only materialize CPU arrays")
        copy = CopyPolicy.from_value(copy)
        if isinstance(x, np.ndarray):
            dtype_requires_copy = dtype is not None and np.dtype(dtype) != x.dtype
            if copy is CopyPolicy.NEVER and dtype_requires_copy:
                raise BackendCopyError("to_backend would require a dtype conversion copy")
            if copy is CopyPolicy.ALWAYS:
                return np.array(x, dtype=dtype, copy=True)
            return np.asarray(x, dtype=dtype)
        if copy is CopyPolicy.NEVER:
            raise BackendCopyError("to_backend would require creating a NumPy array")
        return np.asarray(x, dtype=dtype)
