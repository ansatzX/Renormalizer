# -*- coding: utf-8 -*-

from renormalizer.backend.numpy_backend import NumpyBackend


SUPPORTED_BACKENDS = ("numpy", "cupy", "jax")


def normalize_backend_name(name):
    if name is None:
        return "numpy"
    normalized = str(name).lower().strip()
    if normalized in {"np", "numpy"}:
        return "numpy"
    if normalized in {"cupy", "cp"}:
        return "cupy"
    if normalized in {"jax", "jnp"}:
        return "jax"
    raise ValueError(
        f"Unknown backend '{name}'. Supported backends: {', '.join(SUPPORTED_BACKENDS)}"
    )


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
    raise ValueError(
        f"Unknown backend '{name}'. Supported backends: {', '.join(SUPPORTED_BACKENDS)}"
    )
