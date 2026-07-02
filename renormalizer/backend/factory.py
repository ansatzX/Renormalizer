# -*- coding: utf-8 -*-

import importlib.util
import os

from renormalizer.backend.config import BackendConfig
from renormalizer.backend.numpy_backend import NumpyBackend


SUPPORTED_BACKENDS = ("numpy", "cupy", "jax", "cupynumeric", "torch")

_BACKEND_ALIASES = {
    "np": "numpy",
    "numpy": "numpy",
    "cp": "cupy",
    "cupy": "cupy",
    "jnp": "jax",
    "jax": "jax",
    "cupynumeric": "cupynumeric",
    "cunumeric": "cupynumeric",
    "legate": "cupynumeric",
    "torch": "torch",
    "pytorch": "torch",
}

_BACKEND_PACKAGES = {
    "numpy": "numpy",
    "cupy": "cupy",
    "jax": "jax",
    "cupynumeric": "cupynumeric",
    "torch": "torch",
}

_BACKEND_ADAPTERS = {
    "cupynumeric": "renormalizer.backend.cupynumeric_backend",
    "torch": "renormalizer.backend.torch_backend",
}


_TRUE_VALUES = {"1", "true", "yes", "on"}


def normalize_backend_name(name):
    if name is None:
        return "numpy"
    normalized = str(name).lower().strip()
    if normalized in _BACKEND_ALIASES:
        return _BACKEND_ALIASES[normalized]
    raise ValueError(
        f"Unknown backend '{name}'. Supported backends: {', '.join(SUPPORTED_BACKENDS)}"
    )


def _experimental_cupynumeric_enabled():
    value = os.environ.get("RENO_ENABLE_EXPERIMENTAL_CUPYNUMERIC", "")
    return value.strip().lower() in _TRUE_VALUES


def is_backend_available(name):
    normalized = normalize_backend_name(name)
    if normalized == "cupynumeric" and not _experimental_cupynumeric_enabled():
        return False
    package_name = _BACKEND_PACKAGES[normalized]
    if importlib.util.find_spec(package_name) is None:
        return False
    adapter_name = _BACKEND_ADAPTERS.get(normalized)
    if adapter_name is not None and importlib.util.find_spec(adapter_name) is None:
        return False
    return True


def available_backends():
    return {name: is_backend_available(name) for name in SUPPORTED_BACKENDS}


def _raise_missing_backend(normalized):
    raise ImportError(
        f"{normalized} is not installed. Install the optional '{normalized}' backend "
        f"package before selecting this backend."
    )


def _raise_missing_adapter(normalized):
    raise ImportError(
        f"{normalized} backend adapter is not available in this Renormalizer build. "
        f"Install a version that includes renormalizer.backend.{normalized}_backend "
        f"before selecting this backend."
    )


def _raise_experimental_cupynumeric_disabled():
    raise ImportError(
        "cupynumeric backend is experimental and disabled by default; set "
        "RENO_ENABLE_EXPERIMENTAL_CUPYNUMERIC=1 to enable it for explicit probes."
    )


def _require_backend_package_and_adapter(normalized):
    package_name = _BACKEND_PACKAGES[normalized]
    if importlib.util.find_spec(package_name) is None:
        _raise_missing_backend(normalized)
    adapter_name = _BACKEND_ADAPTERS.get(normalized)
    if adapter_name is not None and importlib.util.find_spec(adapter_name) is None:
        _raise_missing_adapter(normalized)


def create_backend(name=None, *, explicit=True, config=None, **options):
    backend_config = BackendConfig.from_config(config, **options)
    normalized = normalize_backend_name(name)
    if normalized == "numpy":
        return NumpyBackend(config=backend_config)
    if normalized == "cupy":
        from renormalizer.backend.cupy_backend import CupyBackend
        return CupyBackend(config=backend_config)
    if normalized == "jax":
        from renormalizer.backend.jax_backend import JaxBackend
        return JaxBackend(config=backend_config)
    if normalized == "cupynumeric":
        package_name = _BACKEND_PACKAGES[normalized]
        if importlib.util.find_spec(package_name) is None:
            _raise_missing_backend(normalized)
        if not _experimental_cupynumeric_enabled():
            _raise_experimental_cupynumeric_disabled()
        _require_backend_package_and_adapter(normalized)
        from renormalizer.backend.cupynumeric_backend import CupynumericBackend
        return CupynumericBackend(config=backend_config)
    if normalized == "torch":
        _require_backend_package_and_adapter(normalized)
        from renormalizer.backend.torch_backend import TorchBackend
        return TorchBackend(config=backend_config)
    raise ValueError(
        f"Unknown backend '{name}'. Supported backends: {', '.join(SUPPORTED_BACKENDS)}"
    )
