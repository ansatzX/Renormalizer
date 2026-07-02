# -*- coding: utf-8 -*-

"""Optional PyTorch backend extension point."""

import numpy as np

from renormalizer.backend.abstract import AbstractBackend
from renormalizer.backend.mpi import TorchDistributedMixin

try:
    import torch
    _IMPORT_ERROR = None
except (ImportError, OSError) as exc:
    torch = None
    _IMPORT_ERROR = exc


class TorchBackend(TorchDistributedMixin, AbstractBackend):
    name = "torch"
    supported_device_kinds = ("cpu", "gpu")
    array_namespace = None
    ndarray = (np.ndarray,)
    host_array_types = (np.ndarray,)
    device_array_types = ()
    memory_errors = (MemoryError,)
    opt_einsum_name = "torch"
    supports_gpu = True
    supports_grouped_gemm = True

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
        self._torch_device = self._configured_torch_device()

        self.array_namespace = torch
        self.linalg = torch.linalg
        self.random = _TorchRandomProxy(self)
        self.device_array_types = (torch.Tensor,)
        self.ndarray = (np.ndarray, torch.Tensor)
        torch_oom_error = getattr(torch, "OutOfMemoryError", None)
        if torch_oom_error is not None:
            self.memory_errors = (MemoryError, torch_oom_error)
        self._init_distributed_runtime()

    def _configured_cuda_index(self):
        spec = self.current_device()
        if spec.kind == "cuda" and spec.index is not None:
            index = int(spec.index)
            if index < 0 or index >= torch.cuda.device_count():
                raise ValueError(
                    "torch backend CUDA device index {0} is out of range for {1} visible device(s)"
                    .format(index, torch.cuda.device_count())
                )
            return index
        return None

    def _configured_torch_device(self):
        torch_device = getattr(torch, "device", None)
        if torch_device is None:
            return None
        if self.device == "gpu":
            index = self._configured_cuda_index()
            if index is not None:
                torch.cuda.set_device(index)
                return torch_device("cuda:{0}".format(index))
            return torch_device("cuda")
        return torch_device("cpu")

    def set_device(self, device):
        super().set_device(device)
        self._torch_device = self._configured_torch_device()

    def device_count(self):
        if self.device == "gpu" and torch.cuda.is_available():
            return int(torch.cuda.device_count())
        return 1

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
        if args and isinstance(args[0], torch.Tensor):
            result = args[0].clone() if copy else args[0]
            dtype = kwargs.get("dtype")
            device = kwargs.get("device")
            if dtype is not None or device is not None:
                to = getattr(result, "to", None)
                if callable(to):
                    result = to(dtype=dtype, device=device)
            return result
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

    @staticmethod
    def _normalize_tensordot_dims(axes):
        if isinstance(axes, int):
            return axes
        if not (isinstance(axes, tuple) and len(axes) == 2):
            return axes
        left_axes, right_axes = axes
        if isinstance(left_axes, int):
            left_axes = (left_axes,)
        elif isinstance(left_axes, range):
            left_axes = tuple(left_axes)
        else:
            left_axes = tuple(left_axes)
        if isinstance(right_axes, int):
            right_axes = (right_axes,)
        elif isinstance(right_axes, range):
            right_axes = tuple(right_axes)
        else:
            right_axes = tuple(right_axes)
        return (left_axes, right_axes)

    def tensordot(self, a, b, axes=2):
        a, b = self._promote_tensordot_operands(a, b)
        return torch.tensordot(a, b, dims=self._normalize_tensordot_dims(axes))


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
