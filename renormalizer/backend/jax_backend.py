# -*- coding: utf-8 -*-

"""JAX backend — delegates array operations to jax.numpy, exposes autodiff transforms."""

import logging
import os
import weakref

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
    ndarray = jax.Array
    array_types = (np.ndarray, jax.Array)
    device_array_types = (jax.Array,)
    memory_errors = (MemoryError,)
    opt_einsum_name = "jax"
    supports_autodiff = True
    supports_jit = True
    supports_functional_update = True

    def __init__(self, device=None):
        if device is None:
            self._device = jax.devices()[0]
        elif device == 'cpu':
            self._device = jax.devices('cpu')[0]
        elif device.startswith('cuda:') and device[5:].isdigit():
            try:
                devices = jax.devices('gpu')
                self._device = devices[int(device[5:])]
            except (RuntimeError, IndexError) as error:
                raise ValueError(f'JAX device {device} unavailable') from error
        else:
            raise ValueError(f'unsupported JAX device {device}')
        self._pending = {}
        super().__init__()
        if os.environ.get("RENO_FP32") is not None:
            self.use_32bits()

        self.linalg = _JaxLinalg(self)
        self._rng_key = jax.device_put(jr.PRNGKey(2019), self._device)
        self.transforms = JaxTransforms()

    def __getattr__(self, name):
        attribute = getattr(jnp, name)
        if not callable(attribute) or isinstance(attribute, type):
            return attribute
        def invoke(*args, **kwargs):
            with jax.default_device(self._device):
                return self.track(attribute(*args, **kwargs))
        return invoke

    def current_device(self):
        return 'cpu' if self._device.platform == 'cpu' else f'cuda:{self._device.id}'

    def is_array(self, x):
        return isinstance(x, self.array_types)

    def owns(self, x):
        return isinstance(x, jax.Array) and x.devices() == {self._device}

    def dtype_of(self, x):
        return np.dtype(x.dtype)

    def track(self, result):
        for value in jax.tree.leaves(result):
            if isinstance(value, jax.Array):
                key = id(value)
                self._pending[key] = weakref.ref(value, lambda ref, key=key: self._pending.pop(key, None))
        return result

    def tensordot(self, a, b, axes=2):
        # Basis contractions supply range objects; JAX versions require
        # concrete axis pairs. Scalar axes are contraction counts, not pairs.
        if isinstance(axes, (int, np.integer)):
            axes = int(axes)
        else:
            axes = tuple(int(axis) if isinstance(axis, (int, np.integer))
                         else tuple(axis) for axis in axes)
        return self.strict_call('tensordot', a, b, axes=axes)

    def strict_call(self, name, *args, **kwargs):
        if name == 'linalg.eigh':
            kwargs['symmetrize_input'] = False
        if name == 'astype':
            return self.track(args[0].astype(args[1], **kwargs))
        namespace = jnp
        for part in name.split('.'):
            namespace = getattr(namespace, part)
        with jax.default_device(self._device):
            return self.track(namespace(*args, **kwargs))

    def strict_update(self, name, x, idx, value):
        with jax.default_device(self._device):
            indexed = x.at[idx]
            if name == 'set':
                result = indexed.set(value)
            elif name == 'add':
                result = indexed.add(value)
            elif name == 'sub':
                result = indexed.add(-value)
            else:
                result = indexed.multiply(value)
            return self.track(result)

    @property
    def random(self):
        return _JaxRandomProxy(self)

    def _consume_key(self):
        key, subkey = jr.split(self._rng_key)
        self._rng_key = key
        return subkey

    def seed(self, seedval):
        self._rng_key = jax.device_put(jr.PRNGKey(seedval), self._device)

    def array(self, data, dtype=None, *, copy=True):
        if copy is not None and type(copy) is not bool:
            raise TypeError('copy must be None, True, or False')
        if copy is False:
            if not self.owns(data) or (dtype is not None and np.dtype(dtype) != data.dtype):
                raise ValueError('copy=False cannot transfer or convert a JAX array')
            return data
        with jax.default_device(self._device):
            return self.track(jnp.array(data, dtype=dtype, copy=True if copy is True else None, device=self._device))

    def asarray(self, data, dtype=None):
        return self.array(data, dtype=dtype, copy=None)

    def from_numpy(self, x, *, copy=None):
        return self.array(x, dtype=x.dtype, copy=copy)

    def to_numpy(self, x, *, copy=None):
        if copy is False:
            raise ValueError('copy=False host conversion of JAX arrays is not guaranteed')
        host = np.asarray(jax.device_get(x))
        return host.copy() if copy is True else host

    def numpy(self, x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        return self.to_numpy(x)

    def sync(self):
        for reference in list(self._pending.values()):
            value = reference()
            if value is not None:
                value.block_until_ready()

    def at_set(self, x, idx, value):
        return self.track(x.at[idx].set(value))

    def write_owned(self, x, idx, value):
        # JAX arrays remain immutable even for private solver workspaces;
        # callers retain this result, and sync must track the new array.
        return self.at_set(x, idx, value)

    def at_add(self, x, idx, value):
        return self.track(x.at[idx].add(value))

    def at_sub(self, x, idx, value):
        return self.track(x.at[idx].add(-value))

    def at_mul(self, x, idx, value):
        return self.track(x.at[idx].multiply(value))


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
            return self._backend.track(fn(key, shape=size, **kwargs))
        return self._backend.track(fn(key, **kwargs))

    def uniform(self, low=0.0, high=1.0, size=None, dtype=None):
        dtype = self._backend.real_dtype if dtype is None else dtype
        size = (size,) if isinstance(size, int) else size
        key = self._backend._consume_key()
        if size is not None:
            x = jr.uniform(key, shape=size, minval=low, maxval=high, dtype=dtype)
        else:
            x = jr.uniform(key, shape=(), minval=low, maxval=high, dtype=dtype)
        return self._backend.track(x)

    def normal(self, loc=0.0, scale=1.0, size=None, dtype=None):
        dtype = self._backend.real_dtype if dtype is None else dtype
        size = (size,) if isinstance(size, int) else size
        key = self._backend._consume_key()
        if size is not None:
            x = jr.normal(key, shape=size, dtype=dtype) * scale + loc
        else:
            x = jr.normal(key, shape=(), dtype=dtype) * scale + loc
        return self._backend.track(x)

    def randint(self, low, high=None, size=None, dtype=int):
        if high is None:
            low, high = 0, low
        size = (size,) if isinstance(size, int) else size
        key = self._backend._consume_key()
        if size is not None:
            x = jr.randint(key, shape=size, minval=low, maxval=high, dtype=dtype)
        else:
            x = jr.randint(key, shape=(), minval=low, maxval=high, dtype=dtype)
        return self._backend.track(x)

    def randn(self, *dims):
        key = self._backend._consume_key()
        return self._backend.track(jr.normal(key, shape=dims if dims else (), dtype=self._backend.real_dtype))

    def rand(self, *dims):
        key = self._backend._consume_key()
        return self._backend.track(jr.uniform(key, shape=dims if dims else (), dtype=self._backend.real_dtype))

    def random(self, size=None):
        return self.uniform(size=size)


class _JaxLinalg:
    def __init__(self, backend):
        self._backend = backend

    def __getattr__(self, name):
        function = getattr(jnp.linalg, name)
        if not callable(function):
            return function
        return lambda *args, **kwargs: self._backend.strict_call('linalg.' + name, *args, **kwargs)
