# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""Small helpers for crossing between backend arrays and Python values."""

import numpy as np


def to_numpy_dtype(dtype):
    if dtype is None:
        return None
    try:
        return np.dtype(dtype)
    except TypeError:
        name = str(dtype).rsplit(".", 1)[-1]
        try:
            return np.dtype(name)
        except (TypeError, ValueError) as error:
            raise TypeError("cannot convert {!r} to a NumPy dtype".format(dtype)) from error


def eye_like(size, operand, namespace):
    return namespace.eye(size, dtype=operand.dtype)


def scalar_to_python(value, active_backend):
    scalar = np.asarray(active_backend.to_numpy(value))
    if scalar.ndim != 0:
        raise ValueError("expected a scalar value, got shape {}".format(scalar.shape))
    return scalar.item()


def flatten_backend(value, active_backend):
    return active_backend.to_backend(value).reshape(-1)
