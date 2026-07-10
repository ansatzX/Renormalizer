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

    def to_backend(self, value):
        if value is None:
            return None
        return np.asarray(value)
