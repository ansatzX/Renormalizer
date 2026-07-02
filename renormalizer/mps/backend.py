# -*- coding: utf-8 -*-

import logging

import numpy as np

from renormalizer.cons import (
    SUPPORTED_BACKENDS,
    available_backends,
    backend,
    get_backend,
    is_backend_available,
    runtime_backend,
    set_backend,
    xp,
)

try:
    import primme
    IMPORT_PRIMME_EXCEPTION = None
except Exception as e:
    primme = None
    IMPORT_PRIMME_EXCEPTION = e


logger = logging.getLogger(__name__)


def get_git_commit_hash():
    from renormalizer.cons import get_git_commit_hash as _get_git_commit_hash

    return _get_git_commit_hash()


def use_gpu():
    """Return whether the active runtime backend is configured for GPU arrays."""
    if backend.device == "gpu":
        return True
    device = backend.current_device()
    return getattr(device, "kind", None) in ("cuda", "gpu", "rocm", "mps", "tpu")


def oe_backend():
    """Return the opt_einsum backend name for the active runtime backend."""
    return backend.opt_einsum_name


def memory_errors():
    """Return memory exception classes for the active runtime backend."""
    errors = backend.memory_errors
    return errors if isinstance(errors, tuple) else (errors,)


def array_types():
    """Return array classes recognized by the active runtime backend."""
    types = backend.ndarray
    return types if isinstance(types, tuple) else (types,)


def __getattr__(name):
    if name == "USE_GPU":
        return use_gpu()
    if name == "OE_BACKEND":
        return oe_backend()
    if name == "MEMORY_ERRORS":
        return memory_errors()
    if name == "ARRAY_TYPES":
        return array_types()
    raise AttributeError(name)

__all__ = [
    "np",
    "xp",
    "backend",
    "set_backend",
    "get_backend",
    "runtime_backend",
    "SUPPORTED_BACKENDS",
    "available_backends",
    "is_backend_available",
    "use_gpu",
    "oe_backend",
    "memory_errors",
    "array_types",
    "USE_GPU",
    "OE_BACKEND",
    "MEMORY_ERRORS",
    "ARRAY_TYPES",
    "primme",
    "IMPORT_PRIMME_EXCEPTION",
    "get_git_commit_hash",
]
