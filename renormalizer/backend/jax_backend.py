# -*- coding: utf-8 -*-

"""JAX backend — delegates array operations to jax.numpy, exposes autodiff transforms."""

import logging
import os

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

from renormalizer.backend.abstract import AbstractBackend

logger = logging.getLogger(__name__)


class JaxTransforms:
    """Autodiff transform namespace backed by JAX."""

    @staticmethod
    def grad(f, *args, **kwargs):
        return jax.grad(f, *args, **kwargs)

    @staticmethod
    def value_and_grad(f, *args, **kwargs):
        return jax.value_and_grad(f, *args, **kwargs)

    @staticmethod
    def jit(f, *args, **kwargs):
        return jax.jit(f, *args, **kwargs)

    @staticmethod
    def vmap(f, *args, **kwargs):
        return jax.vmap(f, *args, **kwargs)

    @staticmethod
    def stop_gradient(x):
        return jax.lax.stop_gradient(x)


class JaxBackend(AbstractBackend):
    name = "jax"
    array_namespace = jnp
    ndarray = (jnp.ndarray, np.ndarray)
    memory_errors = (MemoryError,)
    opt_einsum_name = "jax"
    supports_autodiff = True
    supports_jit = True
    supports_functional_update = True

    def __init__(self):
        super().__init__()
        if os.environ.get("RENO_FP32") is not None:
            self.use_32bits()

        self.linalg = jnp.linalg
        self._rng_key = jr.PRNGKey(2019)
        self.transforms = JaxTransforms()

    def __getattr__(self, name):
        return getattr(jnp, name)

    @property
    def random(self):
        return _JaxRandomProxy(self)

    def _consume_key(self):
        key, subkey = jr.split(self._rng_key)
        self._rng_key = key
        return subkey

    def seed(self, seedval):
        self._rng_key = jr.PRNGKey(seedval)

    def array(self, *args, **kwargs):
        return jnp.array(*args, **kwargs)

    def asarray(self, *args, **kwargs):
        return jnp.asarray(*args, **kwargs)

    def from_numpy(self, x):
        return jnp.asarray(x)

    def numpy(self, x):
        if x is None:
            return None
        return np.asarray(x)

    def sync(self):
        return None

    def at_set(self, x, idx, value):
        return x.at[idx].set(value)

    def at_add(self, x, idx, value):
        return x.at[idx].add(value)

    def at_sub(self, x, idx, value):
        return x.at[idx].add(-value)

    def at_mul(self, x, idx, value):
        return x.at[idx].multiply(value)


class _JaxRandomProxy:
    """Proxy that provides a numpy.random-like interface backed by JAX PRNG.

    Each call consumes one key split.  Methods that sample (uniform, normal,
    randint, etc.) accept a ``size`` keyword rather than ``shape`` and use
    the backend's internal key.
    """

    def __init__(self, backend):
        object.__setattr__(self, "_backend", backend)

    def seed(self, seedval):
        self._backend.seed(seedval)

    def __getattr__(self, name):
        return getattr(jr, name)

    def _dispatch(self, name, size=None, **kwargs):
        fn = getattr(jr, name)
        key = self._backend._consume_key()
        if size is not None:
            return fn(key, shape=size, **kwargs)
        return fn(key, **kwargs)

    def uniform(self, low=0.0, high=1.0, size=None, dtype=None):
        key = self._backend._consume_key()
        if size is not None:
            x = jr.uniform(key, shape=size, minval=low, maxval=high, dtype=dtype)
        else:
            x = jr.uniform(key, shape=(), minval=low, maxval=high, dtype=dtype)
        return x

    def normal(self, loc=0.0, scale=1.0, size=None, dtype=None):
        key = self._backend._consume_key()
        if size is not None:
            x = jr.normal(key, shape=size, dtype=dtype) * scale + loc
        else:
            x = jr.normal(key, shape=(), dtype=dtype) * scale + loc
        return x

    def randint(self, low, high=None, size=None, dtype=int):
        key = self._backend._consume_key()
        if size is not None:
            x = jr.randint(key, shape=size, minval=low, maxval=high, dtype=dtype)
        else:
            x = jr.randint(key, shape=(), minval=low, maxval=high, dtype=dtype)
        return x

    def randn(self, *dims):
        key = self._backend._consume_key()
        return jr.normal(key, shape=dims if dims else ())

    def rand(self, *dims):
        key = self._backend._consume_key()
        return jr.uniform(key, shape=dims if dims else ())
