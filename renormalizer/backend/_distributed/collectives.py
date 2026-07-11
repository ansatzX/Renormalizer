"""Collective contracts and local collective behavior."""

from typing import Any, Protocol, runtime_checkable


_REDUCTION_OPS = frozenset({"sum", "prod", "min", "max"})


def _validate_reduction_op(op):
    if op not in _REDUCTION_OPS:
        raise ValueError("unsupported reduction op {!r}".format(op))


def _normalize_axis(array, axis):
    if type(axis) is not int:
        raise TypeError("axis must be an integer")
    ndim = int(array.ndim)
    if axis < -ndim or axis >= ndim:
        raise ValueError("axis {} is out of range for {} dimensions".format(axis, ndim))
    return axis % ndim


@runtime_checkable
class Collective(Protocol):
    rank: int
    size: int

    def barrier(self) -> None:
        ...

    def broadcast(self, array: Any, *, root: int) -> Any:
        ...

    def allreduce(self, array: Any, *, op: str = "sum") -> Any:
        ...

    def reduce_scatter(self, array: Any, *, axis: int, op: str = "sum") -> Any:
        ...

    def allgather(self, array: Any, *, axis: int) -> Any:
        ...

    def close(self) -> None:
        ...


class SingleProcessCollective:
    rank = 0
    size = 1

    def barrier(self):
        return None

    def broadcast(self, array, *, root):
        if root != 0:
            raise ValueError("single-process broadcast root must be zero")
        return array

    def allreduce(self, array, *, op="sum"):
        _validate_reduction_op(op)
        return array.copy()

    def reduce_scatter(self, array, *, axis, op="sum"):
        _normalize_axis(array, axis)
        _validate_reduction_op(op)
        return array.copy()

    def allgather(self, array, *, axis):
        _normalize_axis(array, axis)
        return array.copy()

    def close(self):
        return None


class CupyNcclCollective:
    """Device-bound owner of one ``cupyx.distributed`` NCCL backend.

    Broadcast mutates and returns its input. Other tensor collectives preserve
    their input and return a new array. Axis collectives require equal per-rank
    counts; uneven sharding is intentionally outside this interface.
    """

    _SUPPORTED_DTYPE_CHARS = frozenset("bBiIlLqQefdFD")

    def __init__(
        self,
        context,
        *,
        cupy_module=None,
        init_process_group=None,
        host=None,
        port=None,
    ):
        if not isinstance(host, str) or not host or host.strip() != host:
            raise ValueError("host must be a non-empty string")
        if type(port) is not int or port < 1 or port > 65535:
            raise ValueError("port must be an integer between 1 and 65535")
        if cupy_module is None:
            import cupy as cupy_module
        if init_process_group is None:
            from cupyx.distributed import init_process_group

        self._cupy = cupy_module
        self._context = context
        self.rank = context.rank
        self.size = context.world_size
        self._device_index = context.local_rank
        self._backend = None
        self._closed = False

        device_count = int(cupy_module.cuda.runtime.getDeviceCount())
        if self._device_index >= device_count:
            raise ValueError(
                "local_rank {} is out of range for {} visible CUDA device(s)".format(
                    self._device_index, device_count
                )
            )

        backend = None
        try:
            cupy_module.cuda.Device(self._device_index).use()
            options = {"backend": "nccl", "host": host, "port": port}
            backend = init_process_group(self.size, self.rank, **options)
            if int(backend.rank) != self.rank:
                raise RuntimeError(
                    "NCCL backend rank does not match distributed context"
                )
        except BaseException:
            self._closed = True
            if backend is not None:
                backend.stop()
            raise
        self._backend = backend

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _require_open(self):
        if self._closed:
            raise RuntimeError("collective is closed")

    def _validate_array(self, array, *, op=None):
        self._require_open()
        if not isinstance(array, self._cupy.ndarray):
            raise TypeError("CuPy NCCL collectives require a cupy.ndarray")
        if int(array.device.id) != self._device_index:
            raise ValueError("collective array is not on the selected CUDA device")
        if int(array.size) == 0:
            raise ValueError("collective arrays must not be empty")
        if array.dtype.char not in self._SUPPORTED_DTYPE_CHARS:
            raise TypeError(
                "dtype {} is not supported by NCCL".format(array.dtype.name)
            )
        if array.dtype.kind == "c" and op not in (None, "sum"):
            raise ValueError("complex arrays only support sum reduction")

    def _validate_root(self, root):
        if type(root) is not int:
            raise TypeError("root must be an integer")
        if root < 0 or root >= self.size:
            raise ValueError(
                "root {} is out of range for collective size {}".format(root, self.size)
            )

    def barrier(self):
        self._require_open()
        self._backend.barrier()

    def broadcast(self, array, *, root):
        self._validate_root(root)
        self._validate_array(array)
        cupy = self._cupy
        with cupy.cuda.Device(self._device_index):
            stream = cupy.cuda.get_current_stream()
            contiguous = cupy.ascontiguousarray(array)
            self._backend.broadcast(contiguous, root=root, stream=stream)
            if contiguous is not array:
                cupy.copyto(array, contiguous)
        return array

    def allreduce(self, array, *, op="sum"):
        _validate_reduction_op(op)
        self._validate_array(array, op=op)
        cupy = self._cupy
        with cupy.cuda.Device(self._device_index):
            stream = cupy.cuda.get_current_stream()
            input_buffer = cupy.ascontiguousarray(array)
            output_buffer = cupy.empty_like(input_buffer)
            self._backend.all_reduce(input_buffer, output_buffer, op=op, stream=stream)
        return output_buffer

    def reduce_scatter(self, array, *, axis, op="sum"):
        normalized_axis = _normalize_axis(array, axis)
        _validate_reduction_op(op)
        self._validate_array(array, op=op)
        axis_length = int(array.shape[normalized_axis])
        if axis_length % self.size != 0:
            raise ValueError(
                "axis length {} is not divisible by collective size {}".format(
                    axis_length, self.size
                )
            )

        cupy = self._cupy
        with cupy.cuda.Device(self._device_index):
            stream = cupy.cuda.get_current_stream()
            moved = cupy.moveaxis(array, normalized_axis, 0)
            input_buffer = cupy.ascontiguousarray(moved).reshape(-1)
            output_buffer = cupy.empty(
                input_buffer.size // self.size, dtype=input_buffer.dtype
            )
            self._backend.reduce_scatter(
                input_buffer,
                output_buffer,
                int(output_buffer.size),
                op=op,
                stream=stream,
            )
            output_shape = (axis_length // self.size,) + tuple(moved.shape[1:])
            result = cupy.moveaxis(
                output_buffer.reshape(output_shape), 0, normalized_axis
            )
        return result

    def allgather(self, array, *, axis):
        normalized_axis = _normalize_axis(array, axis)
        self._validate_array(array)
        cupy = self._cupy
        with cupy.cuda.Device(self._device_index):
            stream = cupy.cuda.get_current_stream()
            moved = cupy.moveaxis(array, normalized_axis, 0)
            input_buffer = cupy.ascontiguousarray(moved).reshape(-1)
            output_buffer = cupy.empty(
                input_buffer.size * self.size, dtype=input_buffer.dtype
            )
            self._backend.all_gather(
                input_buffer,
                output_buffer,
                int(input_buffer.size),
                stream=stream,
            )
            output_shape = (int(moved.shape[0]) * self.size,) + tuple(moved.shape[1:])
            result = cupy.moveaxis(
                output_buffer.reshape(output_shape), 0, normalized_axis
            )
        return result

    def close(self):
        if self._closed:
            return
        backend = self._backend
        backend.stop()
        self._backend = None
        self._closed = True
