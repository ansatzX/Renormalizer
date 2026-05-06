# -*- coding: utf-8 -*-

import logging
import os

import numpy as np

from renormalizer.backend.abstract import AbstractBackend

logger = logging.getLogger(__name__)


class NumpyBackend(AbstractBackend):
    name = "numpy"
    array_namespace = np
    ndarray = np.ndarray
    memory_errors = (MemoryError,)
    opt_einsum_name = "numpy"

    def __init__(self):
        super().__init__()
        if os.environ.get("RENO_FP32") is not None:
            self.use_32bits()

        self.linalg = np.linalg
        self.random = np.random

    def __getattr__(self, name):
        return getattr(np, name)

    def array(self, *args, **kwargs):
        copy = kwargs.pop("copy", None)
        result = np.array(*args, **kwargs)
        if copy:
            result = result.copy()
        return result

    def asarray(self, *args, **kwargs):
        return np.asarray(*args, **kwargs)

    def from_numpy(self, x):
        return np.asarray(x)

    def numpy(self, x):
        if x is None:
            return None
        return np.asarray(x)
