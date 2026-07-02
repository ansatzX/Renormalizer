# -*- coding: utf-8 -*-

"""JAX backend — delegates array operations to jax.numpy, exposes autodiff transforms."""

import logging

import numpy as np

from renormalizer.backend.execution import BackendCopyError, CopyPolicy, DeviceSpec, legacy_device_kind, parse_device_spec

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
from renormalizer.backend.config import BackendConfig

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
    supported_device_kinds = ("cpu", "gpu")
    array_namespace = None
    ndarray = (np.ndarray,)
    host_array_types = (np.ndarray,)
    device_array_types = ()
    memory_errors = (MemoryError,)
    opt_einsum_name = "jax"
    supports_cpu = True
    supports_gpu = True
    supports_device_index = True
    supports_autodiff = True
    supports_jit = True
    supports_functional_update = True

    def __init__(self, config=None):
        if jax is None:
            raise ImportError(
                "jax is not installed. Install jax or select another backend."
            ) from _IMPORT_ERROR
        self._jax_device = None
        self._jax_devices_by_kind = {}
        backend_config = BackendConfig.from_config(config)
        super().__init__(config=backend_config)
        available = self._detect_available_device_kinds()
        default = self._default_device_kind(available)
        self.supported_device_kinds = ("cpu", "gpu")
        self.available_device_kinds = available
        self.device = self.config.device or default
        if self.device not in self.supported_device_kinds:
            raise ValueError(
                "jax backend does not support device '{0}'. Supported devices: cpu, gpu."
                .format(self.device)
            )
        self._jax_device = self._select_jax_device(self.device)
        self.device_spec = self._device_spec_for_selected_device(self.device, self._jax_device)

        self.array_namespace = jnp
        self.ndarray = (jnp.ndarray, np.ndarray)
        self.device_array_types = (jnp.ndarray,)
        self.linalg = jnp.linalg
        self._rng_key = jr.PRNGKey(2019)
        self.transforms = JaxTransforms()

    def _device_kind(self, device):
        platform = getattr(device, "platform", None)
        if platform in ("gpu", "cuda", "rocm"):
            return "gpu"
        if platform == "cpu":
            return "cpu"
        return platform

    def _detect_available_device_kinds(self):
        devices_by_kind = {}
        for kind in self.supported_device_kinds:
            query_kind = "gpu" if kind == "gpu" else kind
            try:
                devices = jax.devices(query_kind)
            except Exception:
                devices = []
            if devices:
                devices_by_kind[kind] = tuple(devices)
        try:
            devices = jax.devices()
        except Exception:
            devices = []
        for device in devices:
            kind = self._device_kind(device)
            if kind in self.supported_device_kinds and kind not in devices_by_kind:
                devices_by_kind[kind] = (device,)
        self._jax_devices_by_kind = devices_by_kind
        available = tuple(kind for kind in self.supported_device_kinds if kind in devices_by_kind)
        return available or ("cpu",)

    def _default_device_kind(self, available):
        try:
            default = jax.default_backend()
        except Exception:
            default = None
        if default in ("cuda", "rocm"):
            default = "gpu"
        if default in available:
            return default
        return available[0] if available else "cpu"

    def _select_jax_device(self, kind):
        if kind not in self._jax_devices_by_kind:
            if kind == "gpu":
                raise ValueError(
                    "JAX GPU device was requested but no JAX GPU device is available. "
                    "Install a CUDA-enabled jaxlib or select device='cpu'."
                )
            if kind == "cpu":
                return None
            raise ValueError(
                "JAX device '{0}' was requested but is not available. Available devices: {1}."
                .format(kind, ", ".join(self.available_device_kinds) or "none")
            )
        devices = tuple(self._jax_devices_by_kind[kind])
        if kind == "gpu" and self.config.device_spec is not None and self.config.device_spec.index is not None:
            index = int(self.config.device_spec.index)
            if index < 0 or index >= len(devices):
                raise ValueError(
                    "jax backend CUDA device index {0} is out of range for {1} visible device(s)"
                    .format(index, len(devices))
                )
            return devices[index]
        return devices[0]

    def _jax_device_for_spec(self, spec):
        if spec is None:
            return self._jax_device
        kind = legacy_device_kind(spec)
        if kind == "gpu":
            devices = tuple(self._jax_devices_by_kind.get("gpu", ()))
            if not devices:
                raise ValueError("jax backend GPU device was requested but no JAX GPU device is available")
            if spec.index is None:
                return devices[0]
            index = int(spec.index)
            if index < 0 or index >= len(devices):
                raise ValueError(
                    "jax backend CUDA device index {0} is out of range for {1} visible device(s)"
                    .format(index, len(devices))
                )
            return devices[index]
        if kind == "cpu":
            devices = tuple(self._jax_devices_by_kind.get("cpu", ()))
            return devices[0] if devices else None
        raise ValueError("jax backend does not support device {0!r}".format(spec))

    def _jax_array_matches_spec(self, x, spec):
        if spec is None:
            return True
        devices = tuple(x.devices())
        if len(devices) != 1:
            return False
        device = devices[0]
        kind = legacy_device_kind(spec)
        if kind == "gpu":
            if self._device_kind(device) != "gpu":
                return False
            return spec.index is None or getattr(device, "id", None) == int(spec.index)
        if kind == "cpu":
            return self._device_kind(device) == "cpu"
        return False

    def _device_spec_for_selected_device(self, kind, device):
        if self.config.device_spec is not None:
            return self.config.device_spec
        if kind == "gpu":
            index = getattr(device, "id", None)
            return DeviceSpec(
                kind="cuda",
                index=index,
                visible_id=str(index) if index is not None else None,
            )
        return DeviceSpec(kind="cpu")

    def set_device(self, device):
        super().set_device(device)
        self.config = self.config.replace(device=device)
        self._jax_device = self._select_jax_device(self.device)
        self.device_spec = self._device_spec_for_selected_device(self.device, self._jax_device)

    def device_count(self):
        if self.device == "gpu":
            return len(self._jax_devices_by_kind.get("gpu", ()))
        return len(self._jax_devices_by_kind.get("cpu", ())) or 1

    def _device_spec_for_array(self, x):
        if isinstance(x, jnp.ndarray):
            devices = tuple(x.devices())
            if len(devices) == 1:
                device = devices[0]
                if self._device_kind(device) == "gpu":
                    index = getattr(device, "id", None)
                    return DeviceSpec(
                        kind="cuda",
                        index=index,
                        visible_id=str(index) if index is not None else None,
                    )
                if self._device_kind(device) == "cpu":
                    return DeviceSpec(kind="cpu")
        return super()._device_spec_for_array(x)

    def _place_on_configured_device(self, x):
        if self._jax_device is None:
            return x
        device_put = getattr(jax, "device_put", None)
        if device_put is None:
            return x
        return device_put(x, device=self._jax_device)

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
        return self._place_on_configured_device(jnp.array(*args, **kwargs))

    def asarray(self, *args, **kwargs):
        return self._place_on_configured_device(jnp.asarray(*args, **kwargs))

    def from_numpy(self, x):
        return self._place_on_configured_device(jnp.asarray(x))

    def numpy(self, x):
        return self.to_numpy(x)

    def to_numpy(self, x):
        """Convert ``x`` to a NumPy array on the host."""
        if x is None:
            return None
        return np.asarray(x)

    def to_host(self, x, *, copy=CopyPolicy.IF_NEEDED):
        """Convert ``x`` to a host NumPy array."""
        copy = CopyPolicy.from_value(copy)
        if isinstance(x, np.ndarray):
            if copy is CopyPolicy.ALWAYS:
                return np.array(x, copy=True)
            return x
        if isinstance(x, jnp.ndarray):
            devices = tuple(x.devices())
            on_cpu = len(devices) == 1 and self._device_kind(devices[0]) == "cpu"
            if not on_cpu and copy is CopyPolicy.NEVER:
                raise BackendCopyError("to_host would require a device-to-host copy")
            result = np.asarray(x)
            if copy is CopyPolicy.ALWAYS:
                return np.array(result, copy=True)
            return result
        if copy is CopyPolicy.NEVER:
            raise BackendCopyError("to_host would require creating a host array")
        return np.asarray(x)

    def to_backend(self, x, *, device=None, dtype=None, copy=CopyPolicy.IF_NEEDED):
        """Convert ``x`` to the JAX backend array representation."""
        copy = CopyPolicy.from_value(copy)
        spec = parse_device_spec(device) if device is not None else self.current_device()
        target_device = self._jax_device_for_spec(spec)
        if isinstance(x, jnp.ndarray):
            dtype_requires_copy = dtype is not None and np.dtype(dtype) != x.dtype
            device_requires_copy = not self._jax_array_matches_spec(x, spec)
            if copy is CopyPolicy.NEVER and (dtype_requires_copy or device_requires_copy):
                raise BackendCopyError("to_backend would require a dtype or device copy")
            if copy is CopyPolicy.NEVER:
                return x
            result = jnp.array(x, dtype=dtype, copy=(copy is CopyPolicy.ALWAYS))
            if device_requires_copy and target_device is not None:
                result = jax.device_put(result, device=target_device)
            return result
        if copy is CopyPolicy.NEVER:
            raise BackendCopyError("to_backend would require a host-to-device copy")
        result = jnp.asarray(x, dtype=dtype)
        if target_device is not None:
            result = jax.device_put(result, device=target_device)
        return result

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
            return self._backend._place_on_configured_device(fn(key, shape=self._shape(size), **kwargs))
        return self._backend._place_on_configured_device(fn(key, **kwargs))

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
        return self._backend._place_on_configured_device(x)

    def normal(self, loc=0.0, scale=1.0, size=None, dtype=None):
        key = self._backend._consume_key()
        if size is not None:
            x = jr.normal(key, shape=self._shape(size), dtype=dtype) * scale + loc
        else:
            x = jr.normal(key, shape=(), dtype=dtype) * scale + loc
        return self._backend._place_on_configured_device(x)

    def randint(self, low, high=None, size=None, dtype=int):
        if high is None:
            low, high = 0, low
        key = self._backend._consume_key()
        if size is not None:
            x = jr.randint(key, shape=self._shape(size), minval=low, maxval=high, dtype=dtype)
        else:
            x = jr.randint(key, shape=(), minval=low, maxval=high, dtype=dtype)
        return self._backend._place_on_configured_device(x)

    def randn(self, *dims):
        key = self._backend._consume_key()
        return self._backend._place_on_configured_device(jr.normal(key, shape=dims if dims else ()))

    def rand(self, *dims):
        key = self._backend._consume_key()
        return self._backend._place_on_configured_device(jr.uniform(key, shape=dims if dims else ()))
