# -*- coding: utf-8 -*-

"""Backend-neutral numeric boundary helpers.

These helpers define the small set of conversions that Renormalizer expects
when algorithm code crosses from backend arrays back to Python or creates
arrays that must match an existing backend tensor.
"""

import numpy as _np


def scalar_to_python(value, active_backend):
    """Convert a backend scalar to a Python ``float`` or ``complex``."""
    host_value = active_backend.to_numpy(value)
    scalar = _np.asarray(host_value).reshape(-1)[0]
    if _np.iscomplexobj(scalar):
        if _np.isclose(float(_np.imag(scalar)), 0):
            return float(_np.real(scalar))
        return complex(scalar)
    return float(scalar)


def eye_like(size, like_array, namespace):
    """Create an identity matrix matching a backend tensor's dtype and device."""
    kwargs = {"dtype": like_array.dtype}
    device = getattr(like_array, "device", None)
    if device is not None:
        kwargs["device"] = device
    try:
        return namespace.eye(size, **kwargs)
    except TypeError:
        kwargs.pop("device", None)
        return namespace.eye(size, **kwargs)


def flatten_backend(value, active_backend):
    """Convert ``value`` to the active backend and flatten it."""
    return active_backend.to_backend(value).reshape(-1)
