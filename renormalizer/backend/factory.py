# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""Backend construction without importing optional backend packages."""

import os
import importlib.util
from dataclasses import replace

from renormalizer.backend.config import BackendConfig
from renormalizer.backend.numpy_backend import NumpyBackend


SUPPORTED_BACKENDS = ("numpy", "cupy", "jax", "torch")
_BACKEND_ALIASES = {
    "np": "numpy",
    "numpy": "numpy",
    "cp": "cupy",
    "cupy": "cupy",
    "jax": "jax",
    "jnp": "jax",
    "pytorch": "torch",
    "torch": "torch",
}

_BACKEND_PACKAGES = {
    "cupy": "cupy",
    "jax": "jax",
    "torch": "torch",
}


def normalize_backend_name(name):
    normalized = "numpy" if name is None else str(name).lower().strip()
    try:
        return _BACKEND_ALIASES[normalized]
    except KeyError as error:
        raise ValueError(
            "unsupported backend {!r}; supported backends: {}".format(
                name, ", ".join(SUPPORTED_BACKENDS)
            )
        ) from error


def _make_config(config, options):
    if config is None:
        if "precision" not in options and os.environ.get("RENO_FP32") is not None:
            options = {**options, "precision": 32}
        return BackendConfig(**options)
    if not isinstance(config, BackendConfig):
        raise TypeError("config must be a BackendConfig or None")
    return replace(config, **options) if options else config


def create_backend(name=None, *, config=None, **options):
    normalized = normalize_backend_name(name)
    if normalized == "cupy" and config is None and "device" not in options:
        options = {**options, "device": "gpu"}
    backend_config = _make_config(config, options)
    if normalized == "numpy":
        if backend_config.device != "cpu":
            raise ValueError("backend 'numpy' only supports device='cpu' in Stage 1")
        return NumpyBackend(backend_config)
    if normalized == "cupy":
        from renormalizer.backend.cupy_backend import CupyBackend

        return CupyBackend(backend_config)
    if normalized == "jax":
        from renormalizer.backend.jax_backend import JaxBackend

        return JaxBackend(backend_config)
    from renormalizer.backend.torch_backend import TorchBackend

    return TorchBackend(backend_config)


def is_backend_available(name):
    normalized = normalize_backend_name(name)
    if normalized == "numpy":
        return True
    return importlib.util.find_spec(_BACKEND_PACKAGES[normalized]) is not None


def available_backends():
    return {name: is_backend_available(name) for name in SUPPORTED_BACKENDS}
