# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""CuPy backend adapter with lazy package loading and indexed CUDA devices."""

import logging
from contextlib import contextmanager

import numpy as np

from renormalizer.backend.abstract import AbstractBackend, _DeviceBoundNamespace


logger = logging.getLogger(__name__)


def _import_cupy():
    try:
        import cupy
    except ImportError as error:
        raise ImportError(
            "CuPy is not installed. Install cupy or select the NumPy backend."
        ) from error
    return cupy


class CupyBackend(AbstractBackend):
    name = "cupy"
    opt_einsum_name = "cupy"
    supports_gpu = True
    supports_execution_ir = True
    supports_batched_matmul = True
    supports_grouped_gemm = True
    host_array_types = (np.ndarray,)

    def __init__(self, config):
        cp = _import_cupy()
        if config.device != "gpu" and not config.device.startswith("cuda:"):
            raise ValueError("backend 'cupy' only supports device='gpu' or 'cuda:N'")

        self._cupy = cp
        self.ndarray = (np.ndarray, cp.ndarray)
        self.device_array_types = (cp.ndarray,)
        self.memory_errors = (MemoryError, cp.cuda.memory.OutOfMemoryError)
        super().__init__(config)
        self._device_index = self._resolve_device_index()
        cp.cuda.Device(self._device_index).use()
        self.array_namespace = _DeviceBoundNamespace(cp, self._on_device)

    def _resolve_device_index(self):
        if self.device == "gpu":
            index = int(self._cupy.cuda.runtime.getDevice())
        else:
            index = int(self.device.split(":", 1)[1])
        count = int(self._cupy.cuda.runtime.getDeviceCount())
        if index >= count:
            raise ValueError(
                "cupy backend CUDA device index {} is out of range for {} visible device(s)".format(
                    index, count
                )
            )
        return index

    def _on_device(self, function, *args, **kwargs):
        with self._cupy.cuda.Device(self._device_index):
            return function(*args, **kwargs)

    def current_device(self):
        return "cuda:{}".format(self._device_index)

    def array(self, *args, **kwargs):
        return self._on_device(self._cupy.array, *args, **kwargs)

    def asarray(self, *args, **kwargs):
        return self._on_device(self._cupy.asarray, *args, **kwargs)

    def from_numpy(self, value):
        if value is None:
            return None
        return self.asarray(value)

    def to_numpy(self, value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value
        return self._cupy.asnumpy(value)

    def to_host(self, value):
        return self.to_numpy(value)

    def to_backend(self, value, *, dtype=None):
        if value is None:
            return None
        return self.asarray(value, dtype=dtype)

    def tensordot(self, a, b, axes=2):
        return self._on_device(self._cupy.tensordot, a, b, axes=axes)

    def transpose(self, value, axes=None):
        return self._on_device(self._cupy.transpose, value, axes=axes)

    def matmul(self, a, b, *, stream=None, workspace=None):
        self._require_execution_usable()
        with self._execution_context(stream):
            return self._cupy.matmul(a, b)

    def batched_matmul(self, a, b, *, stream=None, workspace=None):
        self._require_execution_usable()
        with self._execution_context(stream):
            return self._cupy.matmul(a, b)

    def grouped_gemm(
        self, descriptors, tensors, *, stream=None, workspace=None, policy="direct"
    ):
        self._require_execution_usable()
        from renormalizer.backend._gemm.executor import execute_grouped_gemm

        return execute_grouped_gemm(
            self,
            descriptors,
            tensors,
            stream=stream,
            workspace=workspace,
            policy=policy,
        )

    def _validate_execution_stream(self, stream):
        if stream is None:
            return
        if not isinstance(stream, self._cupy.cuda.Stream):
            raise TypeError("CuPy execution stream must be a cupy.cuda.Stream")
        if stream.device_id not in {-1, self._device_index}:
            raise ValueError("CuPy stream must belong to the selected CUDA device")

    @contextmanager
    def _execution_context(self, stream):
        self._require_execution_usable()
        self._validate_execution_stream(stream)
        with self._cupy.cuda.Device(self._device_index):
            if stream is None:
                yield
            else:
                with stream:
                    yield

    def _synchronize_execution_stream(self, stream):
        self._validate_execution_stream(stream)
        with self._cupy.cuda.Device(self._device_index):
            selected = (
                self._cupy.cuda.get_current_stream() if stream is None else stream
            )
            selected.synchronize()

    def _is_exact_execution_destination(self, expected, result):
        if (
            expected.shape != result.shape
            or expected.dtype != result.dtype
            or expected.strides != result.strides
            or expected.data.mem is not result.data.mem
        ):
            return False
        if expected.size == 0:
            return True
        return expected.data.ptr == result.data.ptr

    def _validate_execution_array(self, value):
        if not isinstance(value, self._cupy.ndarray):
            raise TypeError("CuPy execution binding must be a CuPy device array")
        if value.device.id != self._device_index:
            raise ValueError(
                "CuPy execution binding is not on the selected CUDA device"
            )

    def _is_exact_execution_reshape(self, left, right):
        if (
            left.dtype != right.dtype
            or left.size != right.size
            or left.nbytes != right.nbytes
            or not left.flags.c_contiguous
            or not right.flags.c_contiguous
            or any(stride == 0 for stride in right.strides)
            or left.data.mem is not right.data.mem
        ):
            return False
        if left.size == 0:
            return True
        return left.data.ptr == right.data.ptr

    def sync(self):
        self._cupy.cuda.Device(self._device_index).synchronize()

    def free_all_blocks(self):
        with self._cupy.cuda.Device(self._device_index):
            self._cupy.get_default_memory_pool().free_all_blocks()

    def create_collective(self, context, *, host, port):
        if self._device_index != context.local_rank:
            raise ValueError(
                "CuPy backend device cuda:{} does not match local_rank {}".format(
                    self._device_index, context.local_rank
                )
            )
        from renormalizer.backend._distributed.collectives import CupyNcclCollective

        return CupyNcclCollective(
            context,
            cupy_module=self._cupy,
            host=host,
            port=port,
        )

    def log_memory_usage(self, header=""):
        from renormalizer.utils.utils import sizeof_fmt

        with self._cupy.cuda.Device(self._device_index):
            memory_pool = self._cupy.get_default_memory_pool()
            logger.info(
                "%s GPU memory used/total: %s/%s",
                header,
                sizeof_fmt(memory_pool.used_bytes()),
                sizeof_fmt(memory_pool.total_bytes()),
            )
