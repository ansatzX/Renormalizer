# -*- coding: utf-8 -*-

import logging

import numpy as np

from renormalizer.cons import backend, get_backend, runtime_backend, set_backend, xp

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


USE_GPU = False
OE_BACKEND = "numpy"
MEMORY_ERRORS = backend.memory_errors
ARRAY_TYPES = (np.ndarray,)

__all__ = [
    "np",
    "xp",
    "backend",
    "set_backend",
    "get_backend",
    "runtime_backend",
    "USE_GPU",
    "OE_BACKEND",
    "MEMORY_ERRORS",
    "ARRAY_TYPES",
    "primme",
    "IMPORT_PRIMME_EXCEPTION",
    "get_git_commit_hash",
]
