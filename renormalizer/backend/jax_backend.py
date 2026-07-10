# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""Legacy JAX backend adapter."""

import numpy as np

from renormalizer.backend.abstract import AbstractBackend, _DeviceBoundNamespace


def _import_jax():
    try:
        import jax
        import jax.numpy as jnp
        import jax.random as jrandom
    except (ImportError, OSError) as error:
        raise ImportError(
            "JAX is not installed. Install jax or select the NumPy backend."
        ) from error
    return jax, jnp, jrandom


class JaxBackend(AbstractBackend):
    name = "jax"
    opt_einsum_name = "jax"
    supports_autodiff = True
    supports_jit = True
    host_array_types = (np.ndarray,)
    memory_errors = (MemoryError,)

    def __init__(self, config):
        jax, jnp, jrandom = _import_jax()
        self._jax = jax
        self._jnp = jnp
        self._jrandom = jrandom
        self._jax_device = self._resolve_device(config.device)
        array_type = getattr(jax, "Array", jnp.ndarray)
        self.ndarray = (np.ndarray, array_type)
        self.device_array_types = (array_type,)
        self.supports_gpu = config.device != "cpu"
        self._rng_key = None
        self._active = False
        self._desired_x64 = config.precision == 64
        self._prior_x64 = None
        super().__init__(config)
        self.array_namespace = _DeviceBoundNamespace(jnp, self._on_device)

    def _resolve_device(self, configured):
        if configured == "cpu":
            platform = "cpu"
            index = 0
        elif configured == "gpu":
            platform = "gpu"
            index = 0
        elif configured.startswith("cuda:"):
            platform = "gpu"
            index = int(configured.split(":", 1)[1])
        else:
            raise ValueError("backend 'jax' does not support device={!r}".format(configured))
        try:
            devices = self._jax.devices(platform)
        except RuntimeError as error:
            raise ValueError(
                "JAX device {!r} is not available".format(configured)
            ) from error
        if index >= len(devices):
            raise ValueError(
                "JAX device index {} is out of range for {} visible {} device(s)".format(
                    index, len(devices), platform
                )
            )
        return devices[index]

    def _place(self, value):
        return self._jax.device_put(value, self._jax_device)

    def _on_device(self, function, *args, **kwargs):
        if "device" in kwargs:
            raise ValueError("device is fixed by BackendConfig")
        is_conversion = function is self._jnp.array or function is self._jnp.asarray
        with self._jax.default_device(self._jax_device):
            result = function(*args, **kwargs)
        if is_conversion:
            return self._place(result)
        return result

    def use_32bits(self):
        self.dtypes = (self._jnp.float32, self._jnp.complex64)
        self._desired_x64 = False
        if self._active:
            self._jax.config.update("jax_enable_x64", False)

    def use_64bits(self):
        self.dtypes = (self._jnp.float64, self._jnp.complex128)
        self._desired_x64 = True
        if self._active:
            self._jax.config.update("jax_enable_x64", True)

    def activate(self):
        if self._active:
            return
        prior_x64 = bool(self._jax.config.jax_enable_x64)
        try:
            self._jax.config.update("jax_enable_x64", self._desired_x64)
        except Exception:
            self._prior_x64 = None
            raise
        self._prior_x64 = prior_x64
        self._active = True

    def deactivate(self):
        if not self._active:
            return
        self._jax.config.update("jax_enable_x64", self._prior_x64)
        self._prior_x64 = None
        self._active = False

    @property
    def is_32bits(self):
        return self.real_dtype == self._jnp.float32

    @property
    def random(self):
        return _JaxRandom(self)

    def seed(self, seed):
        self._rng_key = self._on_device(self._jrandom.PRNGKey, seed)

    def _next_key(self):
        retained_key, sample_key = self._on_device(
            self._jrandom.split, self._rng_key
        )
        self._rng_key = self._place(retained_key)
        return self._place(sample_key)

    def current_device(self):
        if self._jax_device.platform == "cpu":
            return "cpu"
        return "cuda:{}".format(self._jax_device.id)

    def array(self, *args, **kwargs):
        return self._on_device(self._jnp.array, *args, **kwargs)

    def asarray(self, *args, **kwargs):
        return self._on_device(self._jnp.asarray, *args, **kwargs)

    def from_numpy(self, value):
        if value is None:
            return None
        return self.asarray(value)

    def to_numpy(self, value):
        if value is None:
            return None
        return np.asarray(value)

    def to_host(self, value):
        return self.to_numpy(value)

    def to_backend(self, value, *, dtype=None):
        if value is None:
            return None
        kwargs = {} if dtype is None else {"dtype": dtype}
        converted = self._on_device(self._jnp.asarray, value, **kwargs)
        return self._place(converted)


class _JaxRandom:
    def __init__(self, backend):
        self._backend = backend

    def seed(self, seed):
        self._backend.seed(seed)

    @staticmethod
    def _shape(size):
        if size is None:
            return ()
        if isinstance(size, (int, np.integer)):
            return (int(size),)
        return tuple(size)

    def random(self, size=None, dtype=None):
        if dtype is None:
            dtype = self._backend.real_dtype
        return self._backend._on_device(
            self._backend._jrandom.uniform,
            self._backend._next_key(), shape=self._shape(size), dtype=dtype
        )

    def rand(self, *dimensions):
        return self.random(dimensions)

    def randn(self, *dimensions):
        return self._backend._on_device(
            self._backend._jrandom.normal,
            self._backend._next_key(),
            shape=tuple(dimensions),
            dtype=self._backend.real_dtype,
        )

    def normal(self, loc=0.0, scale=1.0, size=None):
        return self.randn(*self._shape(size)) * scale + loc

    def randint(self, low, high=None, size=None, dtype=None):
        if high is None:
            low, high = 0, low
        if dtype is None:
            dtype = self._jnp.int64 if self._backend.config.precision == 64 else self._jnp.int32
        return self._backend._on_device(
            self._backend._jrandom.randint,
            self._backend._next_key(), self._shape(size), low, high, dtype=dtype
        )

    @property
    def _jnp(self):
        return self._backend._jnp
