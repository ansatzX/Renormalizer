# -*- coding: utf-8 -*-

"""Optional PyTorch backend extension point."""

import os

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
    array_namespace = None
    ndarray = (np.ndarray,)
    host_array_types = (np.ndarray,)
    device_array_types = ()
    memory_errors = (MemoryError,)
    opt_einsum_name = "torch"
    supports_gpu = True

    def __init__(self):
        if torch is None:
            raise ImportError(
                "torch is not installed. Install torch or select another backend."
            ) from _IMPORT_ERROR
        super().__init__()
        if os.environ.get("RENO_FP32") is not None:
            self.use_32bits()

        self.array_namespace = torch
        self.linalg = torch.linalg
        self.random = _TorchRandomProxy(self)
        self.device_array_types = (torch.Tensor,)
        self.ndarray = (np.ndarray, torch.Tensor)
        torch_oom_error = getattr(torch, "OutOfMemoryError", None)
        if torch_oom_error is not None:
            self.memory_errors = (MemoryError, torch_oom_error)

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
        if "dtype" in kwargs or not args:
            return kwargs
        dtype = self._default_dtype_for(args[0])
        if dtype is None:
            return kwargs
        kwargs = dict(kwargs)
        kwargs["dtype"] = dtype
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
        return torch.as_tensor(x, dtype=self._default_dtype_for(x))

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
        return self._torch.rand(self._shape(size), dtype=dtype)

    def rand(self, *dims, **kwargs):
        dtype = kwargs.pop("dtype", None)
        if kwargs:
            raise TypeError("unexpected keyword argument(s): {0}".format(", ".join(kwargs)))
        if dtype is None:
            dtype = self._backend.real_dtype
        if not dims:
            return self._torch.rand((), dtype=dtype)
        return self._torch.rand(*dims, dtype=dtype)

    def randn(self, *dims, **kwargs):
        dtype = kwargs.pop("dtype", None)
        if kwargs:
            raise TypeError("unexpected keyword argument(s): {0}".format(", ".join(kwargs)))
        if dtype is None:
            dtype = self._backend.real_dtype
        if not dims:
            return self._torch.randn((), dtype=dtype)
        return self._torch.randn(*dims, dtype=dtype)

    def randint(self, low, high=None, size=None, dtype=None):
        if high is None:
            low, high = 0, low
        if dtype is None:
            return self._torch.randint(low, high, self._shape(size))
        return self._torch.randint(low, high, self._shape(size), dtype=dtype)

    def __getattr__(self, name):
        return getattr(self._random, name)
