# -*- coding: utf-8 -*-

"""Optional PyTorch backend extension point."""

import contextlib

import numpy as np

from renormalizer.backend.abstract import AbstractBackend
from renormalizer.backend.execution import BackendCopyError, CopyPolicy, DeviceSpec, StreamEvent, legacy_device_kind, parse_device_spec
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
    supports_streams = True
    supports_events = True

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

    def _torch_device_for_spec(self, spec):
        torch_device = getattr(torch, "device", None)
        if torch_device is None:
            return None
        if spec is None:
            return self._torch_device
        kind = legacy_device_kind(spec)
        if kind == "gpu":
            if spec.index is not None:
                index = int(spec.index)
                if index < 0 or index >= torch.cuda.device_count():
                    raise ValueError(
                        "torch backend CUDA device index {0} is out of range for {1} visible device(s)"
                        .format(index, torch.cuda.device_count())
                    )
                return torch_device("cuda:{0}".format(index))
            return torch_device("cuda")
        if kind == "cpu":
            return torch_device("cpu")
        raise ValueError("torch backend does not support device {0!r}".format(spec))

    @staticmethod
    def _tensor_matches_device_spec(x, spec):
        if spec is None:
            return True
        kind = legacy_device_kind(spec)
        if kind == "gpu":
            if x.device.type != "cuda":
                return False
            return spec.index is None or x.device.index == int(spec.index)
        if kind == "cpu":
            return x.device.type == "cpu"
        return False

    def set_device(self, device):
        super().set_device(device)
        self._torch_device = self._configured_torch_device()

    def device_count(self):
        if self.device == "gpu" and torch.cuda.is_available():
            return int(torch.cuda.device_count())
        return 1

    def default_stream(self):
        if self.device != "gpu" or not torch.cuda.is_available():
            return None
        return torch.cuda.current_stream(device=self._torch_device)

    def new_stream(self):
        if self.device != "gpu" or not torch.cuda.is_available():
            return None
        return torch.cuda.Stream(device=self._torch_device)

    def record_event(self, stream=None):
        if self.device != "gpu" or not torch.cuda.is_available():
            return super().record_event(stream=stream)
        stream = self.default_stream() if stream is None else stream
        event = torch.cuda.Event()
        event.record(stream)
        return StreamEvent(device=self.current_device(), stream=stream, token=event)

    def wait_event(self, event, stream=None):
        if self.device != "gpu" or not torch.cuda.is_available():
            return super().wait_event(event, stream=stream)
        stream = self.default_stream() if stream is None else stream
        token = event.token if isinstance(event, StreamEvent) else event
        stream.wait_event(token)
        return None

    def _stream_context(self, stream):
        if stream is None or self.device != "gpu" or not torch.cuda.is_available():
            return contextlib.nullcontext()
        return torch.cuda.stream(stream)

    def synchronize(self, device=None, stream=None):
        if self.device != "gpu" or not torch.cuda.is_available():
            return None
        if stream is not None:
            stream.synchronize()
            return None
        spec = parse_device_spec(device) if device is not None else self.current_device()
        torch_device = self._torch_device_for_spec(spec)
        torch.cuda.synchronize(device=torch_device)
        return None

    def sync(self):
        return self.synchronize()

    def _device_spec_for_array(self, x):
        if isinstance(x, torch.Tensor):
            device = x.device
            if device.type == "cuda":
                index = device.index
                return DeviceSpec(
                    kind="cuda",
                    index=index,
                    visible_id=str(index) if index is not None else None,
                )
            if device.type == "cpu":
                return DeviceSpec(kind="cpu")
        return super()._device_spec_for_array(x)

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

    def to_host(self, x, *, copy=CopyPolicy.IF_NEEDED):
        """Convert ``x`` to a host NumPy array."""
        copy = CopyPolicy.from_value(copy)
        if isinstance(x, np.ndarray):
            if copy is CopyPolicy.ALWAYS:
                return np.array(x, copy=True)
            return x
        if isinstance(x, torch.Tensor):
            if x.device.type != "cpu" and copy is CopyPolicy.NEVER:
                raise BackendCopyError("to_host would require a device-to-host copy")
            result = x.detach().cpu().numpy()
            if copy is CopyPolicy.ALWAYS:
                return np.array(result, copy=True)
            return result
        if copy is CopyPolicy.NEVER:
            raise BackendCopyError("to_host would require creating a host array")
        return np.asarray(x)

    def to_backend(self, x, *, device=None, dtype=None, copy=CopyPolicy.IF_NEEDED):
        """Convert ``x`` to a PyTorch tensor."""
        copy = CopyPolicy.from_value(copy)
        spec = parse_device_spec(device) if device is not None else self.current_device()
        target_device = self._torch_device_for_spec(spec)
        if isinstance(x, torch.Tensor):
            target_dtype = dtype
            dtype_requires_copy = target_dtype is not None and x.dtype != target_dtype
            device_requires_copy = not self._tensor_matches_device_spec(x, spec)
            if copy is CopyPolicy.NEVER and (dtype_requires_copy or device_requires_copy):
                raise BackendCopyError("to_backend would require a dtype or device copy")
            result = x
            if dtype_requires_copy or device_requires_copy:
                result = result.to(dtype=target_dtype, device=target_device)
            if copy is CopyPolicy.ALWAYS:
                result = result.clone()
            return result
        target_dtype = dtype or self._default_dtype_for(x)
        target_is_cuda = target_device is not None and getattr(target_device, "type", None) == "cuda"
        if copy is CopyPolicy.NEVER:
            if target_is_cuda:
                raise BackendCopyError("to_backend would require a host-to-device copy")
            if not isinstance(x, np.ndarray):
                raise BackendCopyError("to_backend would require creating a backend array")
            default_dtype = self._default_dtype_for(x)
            if target_dtype is not None and default_dtype is not None and target_dtype != default_dtype:
                raise BackendCopyError("to_backend would require a dtype conversion copy")
        return torch.as_tensor(x, dtype=target_dtype, device=target_device)

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
