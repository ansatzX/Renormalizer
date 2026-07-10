# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""Small helpers for crossing between backend arrays and Python values."""

import numpy as np


def scalar_to_python(value, active_backend):
    scalar = np.asarray(active_backend.to_numpy(value)).reshape(-1)[0]
    if np.iscomplexobj(scalar) and not np.isclose(scalar.imag, 0):
        return complex(scalar)
    return float(np.real(scalar))


def flatten_backend(value, active_backend):
    return active_backend.to_backend(value).reshape(-1)
