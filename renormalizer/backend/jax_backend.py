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

# The row index is traced, so one compilation per workspace shape serves every
# row; argument 0 is donated so the update happens in place.
_donated_row_set = jax.jit(lambda x, idx, value: x.at[idx].set(value), donate_argnums=0)


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

    def __init__(self, device=None, *, real_dtype=None):
        # Reno's scientific default is double precision. Configure JAX before
        # creating our arrays, without requiring a new user environment knob.
        # An explicit float32 context wins over the legacy environment default.
        dtype = np.dtype(real_dtype if real_dtype is not None else
                         ('float32' if os.environ.get('RENO_FP32') is not None else 'float64'))
        if dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError('real_dtype must be float32 or float64')
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
        if dtype == np.dtype('float64'):
            jax.config.update('jax_enable_x64', True)
            if not jax.config.x64_enabled:
                from renormalizer.backend.contracts import PrecisionError
                raise PrecisionError('JAX could not enable required double precision')
        self._pending = {}
        super().__init__()
        if dtype == np.dtype('float32'):
            self.use_32bits()

        self.linalg = _JaxLinalg(self)
        self._rng_key = jax.device_put(jr.PRNGKey(2019), self._device)
        self.transforms = JaxTransforms()
        from renormalizer.backend.contracts import check_actual_dtype
        # Do not publish a backend whose reported precision differs from its
        # actual allocations, including the legacy global-selection route.
        for requested in self.dtypes:
            check_actual_dtype(self.zeros((0,), dtype=requested).dtype, requested)

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
                if key not in self._pending:
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
        # Host storage (Matrix, environments) is uploaded at every contraction.
        # device_put places the exact host dtype on the captured device in a new
        # buffer, satisfying copy=None and copy=True at a fraction of the
        # jnp.array dispatch cost. Dtypes JAX would canonicalize (64-bit without
        # x64) keep the jnp.array path and its conversion semantics.
        if (copy is not False and type(data) is np.ndarray and data.dtype.kind in 'biufc'
                and (dtype is None or np.dtype(dtype) == data.dtype)
                and jax.dtypes.canonicalize_dtype(data.dtype) == data.dtype):
            return self.track(jax.device_put(data, self._device))
        # Tracers have no concrete device. An explicit dtype also removes JAX's
        # weak scalar typing, even when its storage dtype already matches.
        if (copy is None and not isinstance(data, jax.core.Tracer) and self.owns(data)
                and (dtype is None or (np.dtype(dtype) == data.dtype and not data.weak_type))):
            return self.track(data)
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
            # write_owned donates workspaces; a donated buffer has no result.
            if value is not None and not value.is_deleted():
                value.block_until_ready()

    def at_set(self, x, idx, value):
        return self.track(x.at[idx].set(value))

    def write_owned(self, x, idx, value):
        # JAX arrays remain immutable even for private solver workspaces;
        # callers retain this result, and sync must track the new array.
        # Eager at[].set copies the whole workspace on every write (one copy
        # of the Krylov basis per Lanczos step). The same update jitted with
        # x donated reuses its buffer in place; x is caller-owned scratch that
        # the caller replaces with the result, so invalidating it is within
        # the contract. Integer rows only: static slice bounds would compile
        # once per distinct slice.
        if (isinstance(idx, (int, np.integer)) and not isinstance(idx, bool)
                and isinstance(x, jax.Array) and not isinstance(x, jax.core.Tracer)):
            with jax.default_device(self._device):
                return self.track(_donated_row_set(x, idx, value))
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

    @staticmethod
    def _shape(size):
        return (int(size),) if isinstance(size, (int, np.integer)) else size

    def uniform(self, low=0.0, high=1.0, size=None, dtype=None):
        dtype = self._backend.real_dtype if dtype is None else dtype
        size = self._shape(size)
        key = self._backend._consume_key()
        if size is not None:
            x = jr.uniform(key, shape=size, minval=low, maxval=high, dtype=dtype)
        else:
            x = jr.uniform(key, shape=(), minval=low, maxval=high, dtype=dtype)
        return self._backend.track(x)

    def normal(self, loc=0.0, scale=1.0, size=None, dtype=None):
        dtype = self._backend.real_dtype if dtype is None else dtype
        size = self._shape(size)
        key = self._backend._consume_key()
        if size is not None:
            x = jr.normal(key, shape=size, dtype=dtype) * scale + loc
        else:
            x = jr.normal(key, shape=(), dtype=dtype) * scale + loc
        return self._backend.track(x)

    def randint(self, low, high=None, size=None, dtype=int):
        if high is None:
            low, high = 0, low
        size = self._shape(size)
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
