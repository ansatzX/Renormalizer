# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""Legacy PyTorch backend adapter."""

import numpy as np

from renormalizer.backend.abstract import AbstractBackend, _DeviceBoundNamespace


def _import_torch():
    try:
        import torch
    except (ImportError, OSError) as error:
        raise ImportError(
            "PyTorch is not installed. Install torch or select the NumPy backend."
        ) from error
    return torch


class TorchBackend(AbstractBackend):
    name = "torch"
    opt_einsum_name = "torch"
    host_array_types = (np.ndarray,)
    supports_gpu = True

    def __init__(self, config):
        torch = _import_torch()
        self._torch = torch
        self._torch_device = self._resolve_device(config.device)
        self.ndarray = (np.ndarray, torch.Tensor)
        self.device_array_types = (torch.Tensor,)
        self.supports_gpu = self._torch_device.type == "cuda"
        out_of_memory = getattr(torch, "OutOfMemoryError", None)
        self.memory_errors = (
            (MemoryError, out_of_memory) if out_of_memory is not None else (MemoryError,)
        )
        self._generator = torch.Generator(device=self._torch_device)
        self._random = _TorchRandom(self)
        super().__init__(config)
        self.array_namespace = _DeviceBoundNamespace(torch, self._on_device)

    def _resolve_device(self, configured):
        if configured == "cpu":
            return self._torch.device("cpu")
        if configured == "gpu":
            configured = "cuda:0"
        if not configured.startswith("cuda:"):
            raise ValueError("backend 'torch' does not support device={!r}".format(configured))
        index = int(configured.split(":", 1)[1])
        if not self._torch.cuda.is_available():
            raise ValueError("PyTorch CUDA device was requested but CUDA is unavailable")
        count = self._torch.cuda.device_count()
        if index >= count:
            raise ValueError(
                "PyTorch CUDA device index {} is out of range for {} visible device(s)".format(
                    index, count
                )
            )
        return self._torch.device("cuda:{}".format(index))

    def use_32bits(self):
        self.dtypes = (self._torch.float32, self._torch.complex64)

    def use_64bits(self):
        self.dtypes = (self._torch.float64, self._torch.complex128)

    def _on_device(self, function, *args, **kwargs):
        if "device" in kwargs:
            raise ValueError("device is fixed by BackendConfig")
        conversion_functions = (
            getattr(self._torch, "asarray", None),
            self._torch.as_tensor,
            self._torch.tensor,
        )
        is_conversion = any(function is candidate for candidate in conversion_functions)
        with self._torch.device(self._torch_device):
            result = function(*args, **kwargs)
        if is_conversion and isinstance(result, self._torch.Tensor):
            return result.to(device=self._torch_device)
        return result

    @property
    def is_32bits(self):
        return self.real_dtype is self._torch.float32

    @property
    def random(self):
        return self._random

    def current_device(self):
        return str(self._torch_device)

    def _default_dtype(self, value):
        if isinstance(value, self._torch.Tensor):
            return None
        try:
            dtype = np.asarray(value).dtype
        except (TypeError, ValueError):
            return None
        if np.issubdtype(dtype, np.floating):
            return self.real_dtype
        if np.issubdtype(dtype, np.complexfloating):
            return self.complex_dtype
        return None

    def _normalize_explicit_dtype(self, dtype):
        if isinstance(dtype, self._torch.dtype):
            return dtype
        try:
            numpy_dtype = np.dtype(dtype)
        except (TypeError, ValueError) as error:
            raise TypeError("unsupported Torch dtype: {!r}".format(dtype)) from error

        dtype_map = {
            np.dtype(np.bool_): self._torch.bool,
            np.dtype(np.uint8): self._torch.uint8,
            np.dtype(np.int8): self._torch.int8,
            np.dtype(np.int16): self._torch.int16,
            np.dtype(np.int32): self._torch.int32,
            np.dtype(np.int64): self._torch.int64,
            np.dtype(np.float16): self._torch.float16,
            np.dtype(np.float32): self._torch.float32,
            np.dtype(np.float64): self._torch.float64,
            np.dtype(np.complex64): self._torch.complex64,
            np.dtype(np.complex128): self._torch.complex128,
        }
        for name in ("uint16", "uint32", "uint64"):
            torch_dtype = getattr(self._torch, name, None)
            if torch_dtype is not None:
                dtype_map[np.dtype(name)] = torch_dtype
        try:
            return dtype_map[numpy_dtype]
        except KeyError as error:
            raise TypeError("unsupported Torch dtype: {!r}".format(dtype)) from error

    def _conversion_kwargs(self, value, args, kwargs):
        if len(args) > 1:
            raise TypeError(
                "only dtype may be supplied positionally to Torch conversion APIs"
            )
        kwargs = dict(kwargs)
        if "device" in kwargs:
            raise ValueError("device is fixed by BackendConfig")
        if args:
            if "dtype" in kwargs:
                raise TypeError("multiple values for dtype")
            kwargs["dtype"] = args[0]
        kwargs["device"] = self._torch_device
        if kwargs.get("dtype") is not None:
            kwargs["dtype"] = self._normalize_explicit_dtype(kwargs["dtype"])
        elif "dtype" not in kwargs:
            dtype = self._default_dtype(value)
            if dtype is not None:
                kwargs["dtype"] = dtype
        return kwargs

    @staticmethod
    def _compatible_layout(value):
        if isinstance(value, np.ndarray) and any(stride < 0 for stride in value.strides):
            return np.ascontiguousarray(value)
        return value

    def array(self, value, *args, **kwargs):
        copy = kwargs.pop("copy", True)
        value = self._compatible_layout(value)
        kwargs = self._conversion_kwargs(value, args, kwargs)
        result = self._torch.as_tensor(value, **kwargs)
        return result.clone() if copy else result

    def asarray(self, value, *args, **kwargs):
        value = self._compatible_layout(value)
        kwargs = self._conversion_kwargs(value, args, kwargs)
        return self._torch.as_tensor(value, **kwargs)

    def from_numpy(self, value):
        if value is None:
            return None
        return self.asarray(value)

    def to_numpy(self, value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, self._torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def to_host(self, value):
        return self.to_numpy(value)

    def to_backend(self, value, *, dtype=None):
        if value is None:
            return None
        if dtype is None:
            return self.asarray(value)
        return self.asarray(value, dtype=dtype)

    def tensordot(self, a, b, axes=2):
        a = self.to_backend(a)
        b = self.to_backend(b)
        if a.dtype != b.dtype:
            dtype = self._torch.promote_types(a.dtype, b.dtype)
            a = a.to(dtype=dtype)
            b = b.to(dtype=dtype)
        if isinstance(axes, tuple) and len(axes) == 2:
            left, right = axes
            if isinstance(left, (int, np.integer)):
                left = [int(left)]
            else:
                left = list(left)
            if isinstance(right, (int, np.integer)):
                right = [int(right)]
            else:
                right = list(right)
            axes = (left, right)
        return self._torch.tensordot(a, b, dims=axes)

    def transpose(self, value, axes=None):
        value = self.to_backend(value)
        if axes is None:
            axes = tuple(reversed(range(value.ndim)))
        return self._torch.permute(value, tuple(axes))

    def matmul(self, a, b, *args, **kwargs):
        return self._torch.matmul(self.to_backend(a), self.to_backend(b), *args, **kwargs)

    def sync(self):
        if self._torch_device.type == "cuda":
            self._torch.cuda.synchronize(self._torch_device)

    def free_all_blocks(self):
        if self._torch_device.type == "cuda":
            self._torch.cuda.empty_cache()


class _TorchRandom:
    def __init__(self, backend):
        self._backend = backend

    def seed(self, seed):
        return self._backend._generator.manual_seed(seed)

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
        return self._backend._torch.rand(
            self._shape(size),
            dtype=dtype,
            device=self._backend._torch_device,
            generator=self._backend._generator,
        )

    def rand(self, *dimensions):
        return self.random(dimensions)

    def randn(self, *dimensions):
        return self._backend._torch.randn(
            tuple(dimensions),
            dtype=self._backend.real_dtype,
            device=self._backend._torch_device,
            generator=self._backend._generator,
        )

    def normal(self, loc=0.0, scale=1.0, size=None):
        return self.randn(*self._shape(size)) * scale + loc

    def randint(self, low, high=None, size=None, dtype=None):
        if high is None:
            low, high = 0, low
        kwargs = {
            "device": self._backend._torch_device,
            "generator": self._backend._generator,
        }
        if dtype is not None:
            kwargs["dtype"] = dtype
        return self._backend._torch.randint(low, high, self._shape(size), **kwargs)
