# -*- coding: utf-8 -*-

"""Legacy import compatibility for the process-wide backend facade."""

import numpy as np

from renormalizer.cons import backend, get_backend, get_git_commit_hash, runtime_backend, set_backend, xp


_primme = None
_primme_exception = None
_primme_checked = False


def _get_primme():
    global _primme, _primme_checked, _primme_exception
    if not _primme_checked:
        _primme_checked = True
        try:
            import primme as module
        except Exception as error:
            _primme_exception = error
        else:
            _primme = module
    return _primme, _primme_exception


def __getattr__(name):
    if name == "USE_GPU":
        return backend.supports_gpu
    if name == "GPU_ID":
        device = backend.current_device()
        if device.startswith("cuda:"):
            return int(device.split(":", 1)[1])
        return None
    if name == "OE_BACKEND":
        return backend.opt_einsum_name
    if name == "MEMORY_ERRORS":
        return backend.memory_errors
    if name == "ARRAY_TYPES":
        ndarray = backend.ndarray
        return ndarray if isinstance(ndarray, tuple) else (ndarray,)
    if name == "primme":
        return _get_primme()[0]
    if name == "IMPORT_PRIMME_EXCEPTION":
        return _get_primme()[1]
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


__all__ = [
    "np",
    "xp",
    "backend",
    "set_backend",
    "get_backend",
    "runtime_backend",
    "USE_GPU",
    "GPU_ID",
    "OE_BACKEND",
    "MEMORY_ERRORS",
    "ARRAY_TYPES",
    "primme",
    "IMPORT_PRIMME_EXCEPTION",
    "get_git_commit_hash",
]
