# -*- coding: utf-8 -*-

import importlib.util

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


def normalize_backend_name(name):
    if name is None:
        return "numpy"
    normalized = str(name).lower().strip()
    if normalized in _BACKEND_ALIASES:
        return _BACKEND_ALIASES[normalized]
    raise ValueError(
        f"Unknown backend '{name}'. Supported backends: {', '.join(SUPPORTED_BACKENDS)}"
    )


def is_backend_available(name):
    normalized = normalize_backend_name(name)
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


def _require_backend_package_and_adapter(normalized):
    package_name = _BACKEND_PACKAGES[normalized]
    if importlib.util.find_spec(package_name) is None:
        _raise_missing_backend(normalized)
    adapter_name = _BACKEND_ADAPTERS.get(normalized)
    if adapter_name is not None and importlib.util.find_spec(adapter_name) is None:
        _raise_missing_adapter(normalized)


def create_backend(name=None, *, explicit=True):
    normalized = normalize_backend_name(name)
    if normalized == "numpy":
        return NumpyBackend()
    if normalized == "cupy":
        from renormalizer.backend.cupy_backend import CupyBackend
        return CupyBackend()
    if normalized == "jax":
        from renormalizer.backend.jax_backend import JaxBackend
        return JaxBackend()
    if normalized == "cupynumeric":
        _require_backend_package_and_adapter(normalized)
        from renormalizer.backend.cupynumeric_backend import CupynumericBackend
        return CupynumericBackend()
    if normalized == "torch":
        _require_backend_package_and_adapter(normalized)
        from renormalizer.backend.torch_backend import TorchBackend
        return TorchBackend()
    raise ValueError(
        f"Unknown backend '{name}'. Supported backends: {', '.join(SUPPORTED_BACKENDS)}"
    )
