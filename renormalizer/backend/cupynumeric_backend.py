# -*- coding: utf-8 -*-

"""Optional cupynumeric backend extension point."""

import os

import numpy as np

from renormalizer.backend.abstract import AbstractBackend
from renormalizer.backend.config import BackendConfig


def _prepare_legate_import_env(config=None):
    device = None if config is None else config.device
    if device != "gpu" and "LEGATE_CONFIG" not in os.environ and "LEGATE_AUTO_CONFIG" not in os.environ:
        os.environ["LEGATE_AUTO_CONFIG"] = "0"


cnp = None
_IMPORT_ERROR = None


def _load_cupynumeric(config=None):
    global cnp, _IMPORT_ERROR
    if cnp is not None:
        return cnp
    if _IMPORT_ERROR is not None:
        return None
    try:
        _prepare_legate_import_env(config=config)
        import cupynumeric as cupynumeric_module
    except (ImportError, OSError, RuntimeError) as exc:
        _IMPORT_ERROR = exc
        return None
    cnp = cupynumeric_module
    return cnp


class CupynumericBackend(AbstractBackend):
    name = "cupynumeric"
    supported_device_kinds = ("cpu", "gpu")
    available_device_kinds = ("cpu", "gpu")
    array_namespace = None
    ndarray = (np.ndarray,)
    host_array_types = (np.ndarray,)
    device_array_types = ()
    memory_errors = (MemoryError,)
    opt_einsum_name = "numpy"
    supports_gpu = True

    def __init__(self, config=None):
        backend_config = BackendConfig.from_config(config)
        cupynumeric_module = _load_cupynumeric(config=backend_config)
        if cupynumeric_module is None:
            raise ImportError(
                "cupynumeric is not installed or failed to initialize. "
                "Install cupynumeric, configure Legate, or select another backend."
            ) from _IMPORT_ERROR
        super().__init__(config=backend_config)
        self._set_configured_device(("cpu", "gpu"), default="cpu", available=("cpu", "gpu"))

        self.array_namespace = cupynumeric_module
        self.linalg = cupynumeric_module.linalg
        self.random = cupynumeric_module.random

        cnp_ndarray = getattr(cupynumeric_module, "ndarray", None)
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
        if self.is_array(x):
            return x
        try:
            return cnp.asarray(x)
        except NotImplementedError as exc:
            message = str(exc)
            if "attach to array views" not in message:
                raise
            return cnp.asarray(np.array(x, copy=True))
