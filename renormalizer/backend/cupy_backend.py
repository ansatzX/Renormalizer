# -*- coding: utf-8 -*-

"""CuPy backend — delegates to cupy if installed, raises clear error if not."""

import contextlib
import logging

import numpy as np

from renormalizer.backend.abstract import AbstractBackend
from renormalizer.backend.execution import BackendCopyError, CopyPolicy, DeviceSpec, StreamEvent, legacy_device_kind, parse_device_spec

logger = logging.getLogger(__name__)

_cupy = None
_cupy_available = False


def _try_import_cupy():
    global _cupy, _cupy_available
    try:
        import cupy as cp
        _cupy = cp
        _cupy_available = True
    except ImportError:
        _cupy_available = False


_try_import_cupy()


class CupyBackend(AbstractBackend):
    name = "cupy"
    supported_device_kinds = ("gpu",)
    available_device_kinds = ("gpu",)
    supports_cpu = False
    ndarray = (np.ndarray,)  # will be extended in __init__ if cupy available
    memory_errors = (MemoryError,)
    opt_einsum_name = "cupy"
    supports_gpu = True
    supports_device_index = True
    supports_streams = True
    supports_events = True
    supports_memory_pool = True
    supports_grouped_gemm = True
    host_array_types = (np.ndarray,)

    def __init__(self, config=None):
        if not _cupy_available:
            raise ImportError(
                "CuPy is not installed. Install cupy or select another backend."
            )
        super().__init__(config=config)
        self._set_configured_device(("gpu",), default="gpu", available=("gpu",))
        self.array_namespace = _cupy
        self.ndarray = (np.ndarray, _cupy.ndarray)
        self.device_array_types = (_cupy.ndarray,)
        self.memory_errors = (MemoryError, _cupy.cuda.memory.OutOfMemoryError)
        self._activate_configured_device()

        self.linalg = _cupy.linalg
        self.random = _cupy.random

    def __getattr__(self, name):
        return getattr(_cupy, name)

    def _configured_cuda_index(self):
        spec = self.current_device()
        if spec.kind == "cuda" and spec.index is not None:
            index = int(spec.index)
            count = int(_cupy.cuda.runtime.getDeviceCount())
            if index < 0 or index >= count:
                raise ValueError(
                    "cupy backend CUDA device index {0} is out of range for {1} visible device(s)"
                    .format(index, count)
                )
            return index
        return None

    def _activate_configured_device(self):
        index = self._configured_cuda_index()
        if index is not None:
            _cupy.cuda.Device(index).use()

    def _on_configured_device(self, fn, *args, **kwargs):
        index = self._configured_cuda_index()
        if index is None:
            return fn(*args, **kwargs)
        with _cupy.cuda.Device(index):
            return fn(*args, **kwargs)

    def _target_cuda_index(self, device):
        spec = parse_device_spec(device) if device is not None else self.current_device()
        if spec is None:
            return self._configured_cuda_index()
        kind = legacy_device_kind(spec)
        if kind == "cpu":
            raise ValueError("cupy backend cannot materialize CPU backend arrays")
        if kind != "gpu":
            raise ValueError("cupy backend does not support device {0!r}".format(spec))
        if spec.index is None:
            return self._configured_cuda_index()
        index = int(spec.index)
        count = int(_cupy.cuda.runtime.getDeviceCount())
        if index < 0 or index >= count:
            raise ValueError(
                "cupy backend CUDA device index {0} is out of range for {1} visible device(s)"
                .format(index, count)
            )
        return index

    def _on_cuda_index(self, index, fn, *args, **kwargs):
        if index is None:
            return fn(*args, **kwargs)
        with _cupy.cuda.Device(index):
            return fn(*args, **kwargs)

    def set_device(self, device):
        super().set_device(device)
        self._activate_configured_device()

    def device_count(self):
        return int(_cupy.cuda.runtime.getDeviceCount())

    def default_stream(self):
        return self._on_configured_device(_cupy.cuda.get_current_stream)

    def new_stream(self):
        return self._on_configured_device(_cupy.cuda.Stream, non_blocking=True)

    def record_event(self, stream=None):
        stream = self.default_stream() if stream is None else stream
        event = self._on_configured_device(_cupy.cuda.Event)
        event.record(stream)
        return StreamEvent(device=self.current_device(), stream=stream, token=event)

    def wait_event(self, event, stream=None):
        stream = self.default_stream() if stream is None else stream
        if isinstance(event, StreamEvent):
            self._validate_stream_event(event)
            token = event.token
        else:
            token = event
        stream.wait_event(token)
        return None

    def _stream_context(self, stream):
        if stream is None:
            return contextlib.nullcontext()
        return stream

    def _device_spec_for_array(self, x):
        if isinstance(x, _cupy.ndarray):
            index = int(x.device.id)
            return DeviceSpec(kind="cuda", index=index, visible_id=str(index))
        return super()._device_spec_for_array(x)

    def array(self, *args, **kwargs):
        kwargs = self._kwargs_with_default_dtype(args, kwargs)
        return self._on_configured_device(_cupy.array, *args, **kwargs)

    def asarray(self, *args, **kwargs):
        kwargs = self._kwargs_with_default_dtype(args, kwargs)
        return self._on_configured_device(_cupy.asarray, *args, **kwargs)

    def from_numpy(self, x):
        return self._on_configured_device(_cupy.asarray, x)

    def numpy(self, x):
        return self.to_numpy(x)

    def to_numpy(self, x):
        """Convert ``x`` to a NumPy array on the host."""
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        return _cupy.asnumpy(x)

    def to_host(self, x, *, copy=CopyPolicy.IF_NEEDED):
        """Convert ``x`` to a host NumPy array."""
        copy = CopyPolicy.from_value(copy)
        if isinstance(x, np.ndarray):
            if copy is CopyPolicy.ALWAYS:
                return np.array(x, copy=True)
            return x
        if isinstance(x, _cupy.ndarray):
            if copy is CopyPolicy.NEVER:
                raise BackendCopyError("to_host would require a device-to-host copy")
            return _cupy.asnumpy(x)
        if copy is CopyPolicy.NEVER:
            raise BackendCopyError("to_host would require creating a host array")
        return np.asarray(x)

    def to_backend(self, x, *, device=None, dtype=None, copy=CopyPolicy.IF_NEEDED):
        """Convert ``x`` to a CuPy array on the active device."""
        copy = CopyPolicy.from_value(copy)
        target_index = self._target_cuda_index(device)
        if isinstance(x, _cupy.ndarray):
            dtype_requires_copy = dtype is not None and np.dtype(dtype) != x.dtype
            device_requires_copy = target_index is not None and int(x.device.id) != target_index
            if copy is CopyPolicy.NEVER and (dtype_requires_copy or device_requires_copy):
                raise BackendCopyError("to_backend would require a dtype or device copy")
            if copy is CopyPolicy.NEVER:
                return x
            if copy is CopyPolicy.ALWAYS:
                return self._on_cuda_index(target_index, _cupy.array, x, dtype=dtype, copy=True)
            return self._on_cuda_index(target_index, _cupy.asarray, x, dtype=dtype)
        if copy is CopyPolicy.NEVER:
            raise BackendCopyError("to_backend would require a host-to-device copy")
        if dtype is None:
            dtype = self._default_dtype_for(x)
        return self._on_cuda_index(target_index, _cupy.asarray, x, dtype=dtype)

    def grouped_gemm(
        self,
        tasks,
        *,
        buffers=None,
        pack_threshold=4,
        stream=None,
        workspace=None,
        policy="auto",
        fallback_policy=None,
        profile_context=None,
    ):
        """Execute grouped GEMM through CuPy-owned bucketed matmul batches."""
        return super().grouped_gemm(
            tasks,
            buffers=buffers,
            pack_threshold=pack_threshold,
            stream=stream,
            workspace=workspace,
            policy=policy,
            fallback_policy=fallback_policy,
            profile_context=profile_context,
        )

    def free_all_blocks(self):
        mempool = _cupy.get_default_memory_pool()
        mempool.free_all_blocks()

    def log_memory_usage(self, header=""):
        from renormalizer.utils.utils import sizeof_fmt
        mempool = _cupy.get_default_memory_pool()
        logger.info(
            f"{header} GPU memory used/Total: "
            f"{sizeof_fmt(mempool.used_bytes())}/{sizeof_fmt(mempool.total_bytes())}"
        )

    def sync(self):
        index = self._configured_cuda_index()
        if index is None:
            _cupy.cuda.Device().synchronize()
        else:
            _cupy.cuda.Device(index).synchronize()

    def synchronize(self, device=None, stream=None):
        if stream is not None:
            stream.synchronize()
            return None
        if device is not None:
            index = self._target_cuda_index(device)
            if index is None:
                _cupy.cuda.Device().synchronize()
            else:
                _cupy.cuda.Device(index).synchronize()
            return None
        return self.sync()
