# -*- coding: utf-8 -*-

"""JAX backend — delegates array operations to jax.numpy, exposes autodiff transforms."""

import logging
import os

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    _IMPORT_ERROR = None
except (ImportError, OSError) as exc:
    jax = None
    jnp = None
    jr = None
    _IMPORT_ERROR = exc

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
    array_namespace = None
    ndarray = (np.ndarray,)
    host_array_types = (np.ndarray,)
    device_array_types = ()
    memory_errors = (MemoryError,)
    opt_einsum_name = "jax"
    supports_autodiff = True
    supports_jit = True
    supports_functional_update = True

    def __init__(self):
        if jax is None:
            raise ImportError(
                "jax is not installed. Install jax or select another backend."
            ) from _IMPORT_ERROR
        super().__init__()
        if os.environ.get("RENO_FP32") is not None:
            self.use_32bits()

        self.array_namespace = jnp
        self.ndarray = (jnp.ndarray, np.ndarray)
        self.device_array_types = (jnp.ndarray,)
        self.linalg = jnp.linalg
        self._rng_key = jr.PRNGKey(2019)
        self.transforms = JaxTransforms()

    def __getattr__(self, name):
        return getattr(jnp, name)

    @property
    def is_32bits(self):
        return self._real_dtype == jnp.float32

    def use_32bits(self):
        jax.config.update("jax_enable_x64", False)
        self.dtypes = (jnp.float32, jnp.complex64)

    def use_64bits(self):
        jax.config.update("jax_enable_x64", True)
        self.dtypes = (jnp.float64, jnp.complex128)

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
        return self.to_numpy(x)

    def to_numpy(self, x):
        """Convert ``x`` to a NumPy array on the host."""
        if x is None:
            return None
        return np.asarray(x)

    def to_host(self, x):
        """Convert ``x`` to a host NumPy array."""
        return self.to_numpy(x)

    def to_backend(self, x):
        """Convert ``x`` to the JAX backend array representation."""
        return jnp.asarray(x)

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
            return fn(key, shape=self._shape(size), **kwargs)
        return fn(key, **kwargs)

    def _shape(self, size):
        if size is None:
            return ()
        if isinstance(size, int):
            return (size,)
        return tuple(size)

    def random(self, size=None, dtype=None):
        return self.uniform(size=size, dtype=dtype)

    def uniform(self, low=0.0, high=1.0, size=None, dtype=None):
        key = self._backend._consume_key()
        if size is not None:
            x = jr.uniform(key, shape=self._shape(size), minval=low, maxval=high, dtype=dtype)
        else:
            x = jr.uniform(key, shape=(), minval=low, maxval=high, dtype=dtype)
        return x

    def normal(self, loc=0.0, scale=1.0, size=None, dtype=None):
        key = self._backend._consume_key()
        if size is not None:
            x = jr.normal(key, shape=self._shape(size), dtype=dtype) * scale + loc
        else:
            x = jr.normal(key, shape=(), dtype=dtype) * scale + loc
        return x

    def randint(self, low, high=None, size=None, dtype=int):
        if high is None:
            low, high = 0, low
        key = self._backend._consume_key()
        if size is not None:
            x = jr.randint(key, shape=self._shape(size), minval=low, maxval=high, dtype=dtype)
        else:
            x = jr.randint(key, shape=(), minval=low, maxval=high, dtype=dtype)
        return x

    def randn(self, *dims):
        key = self._backend._consume_key()
        return jr.normal(key, shape=dims if dims else ())

    def rand(self, *dims):
        key = self._backend._consume_key()
        return jr.uniform(key, shape=dims if dims else ())
