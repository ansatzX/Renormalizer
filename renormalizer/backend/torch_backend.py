# -*- coding: utf-8 -*-

"""Optional PyTorch backend extension point."""

import numpy as np

from renormalizer.backend.abstract import AbstractBackend

try:
    import torch
    _IMPORT_ERROR = None
except (ImportError, OSError) as exc:
    torch = None
    _IMPORT_ERROR = exc


class TorchBackend(AbstractBackend):
    name = "torch"
    supported_device_kinds = ("cpu", "gpu")
    array_namespace = None
    ndarray = (np.ndarray,)
    host_array_types = (np.ndarray,)
    device_array_types = ()
    memory_errors = (MemoryError,)
    opt_einsum_name = "torch"
    supports_gpu = True

    def __init__(self, config=None):
        if torch is None:
            raise ImportError(
                "torch is not installed. Install torch or select another backend."
            ) from _IMPORT_ERROR
        self._torch_device = None
        super().__init__(config=config)
        available = ["cpu"]
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            available.append("gpu")
        self.supports_gpu = True
        self._set_configured_device(("cpu", "gpu"), default="cpu", available=tuple(available))
        torch_device = getattr(torch, "device", None)
        if torch_device is not None:
            self._torch_device = torch_device("cuda" if self.device == "gpu" else "cpu")

        self.array_namespace = torch
        self.linalg = torch.linalg
        self.random = _TorchRandomProxy(self)
        self.device_array_types = (torch.Tensor,)
        self.ndarray = (np.ndarray, torch.Tensor)
        torch_oom_error = getattr(torch, "OutOfMemoryError", None)
        if torch_oom_error is not None:
            self.memory_errors = (MemoryError, torch_oom_error)
        # Monkey-patch torch.tensordot to auto-promote mixed dtypes (e.g. float64 + complex128)
        if hasattr(torch, 'tensordot') and not getattr(torch.tensordot, "_renormalizer_patched", False):
            _original_tensordot = torch.tensordot
            def _tensordot_with_promote(a, b, dims=2, *args, **kwargs):
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype != b.dtype:
                    target = torch.promote_types(a.dtype, b.dtype)
                    a = a.to(dtype=target)
                    b = b.to(dtype=target)
                # Convert numpy-style axes=(i, j) to torch-style dims=([i], [j])
                if isinstance(dims, tuple) and len(dims) == 2:
                    if isinstance(dims[0], int):
                        dims = ([dims[0]], [dims[1]])
                    elif isinstance(dims[0], range):
                        dims = (list(dims[0]), list(dims[1]))
                return _original_tensordot(a, b, dims, *args, **kwargs)
            _tensordot_with_promote._renormalizer_patched = True
            torch.tensordot = _tensordot_with_promote

    def __getattr__(self, name):
        return getattr(torch, name)

    @property
    def is_32bits(self):
        return self._real_dtype is torch.float32

    def use_32bits(self):
        self.dtypes = (torch.float32, torch.complex64)

    def use_64bits(self):
        self.dtypes = (torch.float64, torch.complex128)

    def _default_dtype_for(self, x):
        if isinstance(x, torch.Tensor):
            return None
        try:
            dtype = np.asarray(x).dtype
        except (TypeError, ValueError):
            return None
        if np.issubdtype(dtype, np.floating):
            return self.real_dtype
        if np.issubdtype(dtype, np.complexfloating):
            return self.complex_dtype
        return None

    def _kwargs_with_default_dtype(self, args, kwargs):
        kwargs = dict(kwargs)
        if "dtype" not in kwargs and args:
            dtype = self._default_dtype_for(args[0])
            if dtype is not None:
                kwargs["dtype"] = dtype
        kwargs = self._kwargs_with_configured_device(kwargs)
        return kwargs

    def _kwargs_with_configured_device(self, kwargs):
        kwargs = dict(kwargs)
        if "device" not in kwargs and self._torch_device is not None:
            kwargs["device"] = self._torch_device
        return kwargs

    def array(self, *args, **kwargs):
        copy = kwargs.pop("copy", None)
        kwargs = self._kwargs_with_default_dtype(args, kwargs)
        result = torch.tensor(*args, **kwargs)
        if copy and isinstance(result, torch.Tensor):
            result = result.clone()
        return result

    def asarray(self, *args, **kwargs):
        kwargs = self._kwargs_with_default_dtype(args, kwargs)
        asarray = getattr(torch, "asarray", None)
        if asarray is not None:
            return asarray(*args, **kwargs)
        return torch.as_tensor(*args, **kwargs)

    def from_numpy(self, x):
        kwargs = {"dtype": self._default_dtype_for(x)}
        kwargs = self._kwargs_with_configured_device(kwargs)
        return torch.as_tensor(x, **kwargs)

    def numpy(self, x):
        return self.to_numpy(x)

    def to_numpy(self, x):
        """Convert ``x`` to a NumPy array on the host."""
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def to_host(self, x):
        """Convert ``x`` to a host NumPy array."""
        return self.to_numpy(x)

    def to_backend(self, x):
        """Convert ``x`` to a PyTorch tensor."""
        return self.asarray(x)


class _TorchRandomProxy:
    def __init__(self, backend):
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_torch", backend.array_namespace)
        object.__setattr__(self, "_random", backend.array_namespace.random)

    def seed(self, seedval):
        return self._torch.manual_seed(seedval)

    def _shape(self, size):
        if size is None:
            return ()
        if isinstance(size, (int, np.integer)):
            return (size,)
        return tuple(size)

    def random(self, size=None, dtype=None):
        if dtype is None:
            dtype = self._backend.real_dtype
        kwargs = self._backend._kwargs_with_configured_device({"dtype": dtype})
        return self._torch.rand(self._shape(size), **kwargs)

    def rand(self, *dims, **kwargs):
        dtype = kwargs.pop("dtype", None)
        if kwargs:
            raise TypeError("unexpected keyword argument(s): {0}".format(", ".join(kwargs)))
        if dtype is None:
            dtype = self._backend.real_dtype
        if not dims:
            kwargs = self._backend._kwargs_with_configured_device({"dtype": dtype})
            return self._torch.rand((), **kwargs)
        kwargs = self._backend._kwargs_with_configured_device({"dtype": dtype})
        return self._torch.rand(*dims, **kwargs)

    def randn(self, *dims, **kwargs):
        dtype = kwargs.pop("dtype", None)
        if kwargs:
            raise TypeError("unexpected keyword argument(s): {0}".format(", ".join(kwargs)))
        if dtype is None:
            dtype = self._backend.real_dtype
        if not dims:
            kwargs = self._backend._kwargs_with_configured_device({"dtype": dtype})
            return self._torch.randn((), **kwargs)
        kwargs = self._backend._kwargs_with_configured_device({"dtype": dtype})
        return self._torch.randn(*dims, **kwargs)

    def randint(self, low, high=None, size=None, dtype=None):
        if high is None:
            low, high = 0, low
        kwargs = {}
        if dtype is not None:
            kwargs["dtype"] = dtype
        kwargs = self._backend._kwargs_with_configured_device(kwargs)
        return self._torch.randint(low, high, self._shape(size), **kwargs)

    def __getattr__(self, name):
        return getattr(self._random, name)
        # Monkey-patch torch.tensordot to auto-promote mixed dtypes (e.g. float64 + complex128)
        if not getattr(torch.tensordot, "_renormalizer_patched", False):
            _original_tensordot = torch.tensordot
            def _tensordot_with_promote(a, b, *args, **kwargs):
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype != b.dtype:
                    target = torch.promote_types(a.dtype, b.dtype)
                    a = a.to(dtype=target)
                    b = b.to(dtype=target)
                return _original_tensordot(a, b, *args, **kwargs)
            _tensordot_with_promote._renormalizer_patched = True
            torch.tensordot = _tensordot_with_promote
