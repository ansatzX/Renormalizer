# -*- coding: utf-8 -*-

"""Optional cupynumeric backend extension point."""

import os

import numpy as np

from renormalizer.backend.abstract import AbstractBackend

try:
    import cupynumeric as cnp
    _IMPORT_ERROR = None
except (ImportError, OSError) as exc:
    cnp = None
    _IMPORT_ERROR = exc


class CupynumericBackend(AbstractBackend):
    name = "cupynumeric"
    array_namespace = None
    ndarray = (np.ndarray,)
    host_array_types = (np.ndarray,)
    device_array_types = ()
    memory_errors = (MemoryError,)
    opt_einsum_name = "numpy"
    supports_gpu = True

    def __init__(self):
        if cnp is None:
            raise ImportError(
                "cupynumeric is not installed. Install cupynumeric or select another backend."
            ) from _IMPORT_ERROR
        super().__init__()
        if os.environ.get("RENO_FP32") is not None:
            self.use_32bits()

        self.array_namespace = cnp
        self.linalg = cnp.linalg
        self.random = cnp.random

        cnp_ndarray = getattr(cnp, "ndarray", None)
        if cnp_ndarray is not None:
            self.device_array_types = (cnp_ndarray,)
            self.ndarray = (np.ndarray, cnp_ndarray)

    def __getattr__(self, name):
        return getattr(cnp, name)

    def array(self, *args, **kwargs):
        return cnp.array(*args, **kwargs)

    def asarray(self, *args, **kwargs):
        return cnp.asarray(*args, **kwargs)

    def from_numpy(self, x):
        return cnp.asarray(x)

    def numpy(self, x):
        return self.to_numpy(x)

    def to_numpy(self, x):
        """Convert ``x`` to a NumPy array on the host."""
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        asnumpy = getattr(cnp, "asnumpy", None)
        if asnumpy is not None:
            return asnumpy(x)
        return np.asarray(x)

    def to_host(self, x):
        """Convert ``x`` to a host NumPy array."""
        return self.to_numpy(x)

    def to_backend(self, x):
        """Convert ``x`` to the cupynumeric backend representation."""
        return cnp.asarray(x)
