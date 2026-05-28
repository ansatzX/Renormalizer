# -*- coding: utf-8 -*-

from typing import Any, Tuple, Type, Union


_BACKEND_PROTOCOL_RUNTIME_ATTRS = (
    "name",
    "array_namespace",
    "opt_einsum_name",
    "supports_gpu",
    "supports_autodiff",
    "supports_jit",
    "supports_sparse",
    "supports_functional_update",
    "host_array_types",
    "device_array_types",
    "ndarray",
    "memory_errors",
    "transforms",
    "random",
    "linalg",
    "rank",
    "size",
    "is_distributed",
    "is_32bits",
    "real_dtype",
    "complex_dtype",
    "dtypes",
    "canonical_atol",
    "canonical_rtol",
)

_BACKEND_PROTOCOL_RUNTIME_METHODS = (
    "use_32bits",
    "use_64bits",
    "array",
    "asarray",
    "from_numpy",
    "to_numpy",
    "numpy",
    "to_host",
    "to_backend",
    "is_array",
    "is_host_array",
    "is_device_array",
    "sync",
    "free_all_blocks",
    "log_memory_usage",
    "at_set",
    "at_add",
    "at_sub",
    "at_mul",
    "barrier",
    "allreduce",
    "broadcast",
    "gather",
    "allgather",
)

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    try:
        from typing_extensions import Protocol, runtime_checkable
    except ImportError:
        class _RuntimeBackendProtocolMeta(type):
            def __instancecheck__(cls, instance):
                return (
                    all(hasattr(instance, attr) for attr in _BACKEND_PROTOCOL_RUNTIME_ATTRS)
                    and all(callable(getattr(instance, method, None)) for method in _BACKEND_PROTOCOL_RUNTIME_METHODS)
                )

        class Protocol(object, metaclass=_RuntimeBackendProtocolMeta):
            pass

        def runtime_checkable(cls):
            return cls


@runtime_checkable
class BackendProtocol(Protocol):
    """Runtime-checkable structural contract for Renormalizer array backends.

    Backend implementations expose array creation, host/device conversion,
    capability flags, dtype configuration, memory hooks, and functional update
    helpers through this surface.  The protocol is intentionally structural so
    backend proxies and third-party backends can participate without inheriting
    from a Renormalizer base class.
    """

    name: str
    """Short backend identifier, such as ``"numpy"``, ``"cupy"``, or ``"jax"``."""

    array_namespace: Any
    """Array API namespace used by this backend."""

    opt_einsum_name: str
    """Backend name understood by opt_einsum."""

    supports_gpu: bool
    """Whether this backend can execute array operations on a GPU device."""

    supports_autodiff: bool
    """Whether this backend exposes automatic differentiation transforms."""

    supports_jit: bool
    """Whether this backend exposes just-in-time compilation transforms."""

    supports_sparse: bool
    """Whether this backend provides sparse array support."""

    supports_functional_update: bool
    """Whether update helpers return updated arrays without in-place mutation."""

    host_array_types: Tuple[Type[Any], ...]
    """Array classes that are already resident on host memory."""

    device_array_types: Tuple[Type[Any], ...]
    """Array classes that are resident on accelerator/device memory."""

    ndarray: Union[Type[Any], Tuple[Type[Any], ...]]
    """Array classes accepted by :meth:`is_array`."""

    memory_errors: Tuple[Type[BaseException], ...]
    """Exception types raised for backend memory exhaustion."""

    transforms: Any
    """Namespace exposing autodiff/JIT transform functions for the backend."""

    random: Any
    """Namespace exposing NumPy-like random helpers for the backend."""

    linalg: Any
    """Namespace exposing linear algebra helpers for the backend."""

    @property
    def is_32bits(self) -> bool:
        """Whether the backend is configured for 32-bit real/complex dtypes."""
        ...

    def use_32bits(self) -> None:
        """Switch the backend to 32-bit real/complex dtypes."""
        ...

    def use_64bits(self) -> None:
        """Switch the backend to 64-bit real/complex dtypes."""
        ...

    @property
    def real_dtype(self) -> Any:
        """Backend real floating dtype."""
        ...

    @real_dtype.setter
    def real_dtype(self, tp: Any) -> None:
        ...

    @property
    def complex_dtype(self) -> Any:
        """Backend complex floating dtype."""
        ...

    @complex_dtype.setter
    def complex_dtype(self, tp: Any) -> None:
        ...

    @property
    def dtypes(self) -> Tuple[Any, Any]:
        """Pair of configured real and complex dtypes."""
        ...

    @dtypes.setter
    def dtypes(self, target: Tuple[Any, Any]) -> None:
        ...

    @property
    def canonical_atol(self) -> float:
        """Default absolute tolerance for backend numerical comparisons."""
        ...

    @canonical_atol.setter
    def canonical_atol(self, value: float) -> None:
        ...

    @property
    def canonical_rtol(self) -> float:
        """Default relative tolerance for backend numerical comparisons."""
        ...

    @canonical_rtol.setter
    def canonical_rtol(self, value: float) -> None:
        ...

    def array(self, *args: Any, **kwargs: Any) -> Any:
        """Create a backend array."""
        ...

    def asarray(self, *args: Any, **kwargs: Any) -> Any:
        """Convert input to a backend array without unnecessary copies."""
        ...

    def from_numpy(self, x: Any) -> Any:
        """Convert a NumPy host array to this backend's array representation."""
        ...

    def to_numpy(self, x: Any) -> Any:
        """Convert a backend array to a NumPy host array."""
        ...

    def numpy(self, x: Any) -> Any:
        """Compatibility alias for converting an array to NumPy."""
        ...

    def to_host(self, x: Any) -> Any:
        """Convert a backend array to a host-resident array."""
        ...

    def to_backend(self, x: Any) -> Any:
        """Convert an array-like object to this backend's representation."""
        ...

    def is_array(self, x: Any) -> bool:
        """Return whether ``x`` is an array recognized by this backend."""
        ...

    def is_host_array(self, x: Any) -> bool:
        """Return whether ``x`` is a host-resident array."""
        ...

    def is_device_array(self, x: Any) -> bool:
        """Return whether ``x`` is a device-resident array."""
        ...

    def sync(self) -> None:
        """Synchronize pending backend work, if applicable."""
        ...

    def free_all_blocks(self) -> None:
        """Release cached backend memory blocks, if applicable."""
        ...

    def log_memory_usage(self, header: str = "") -> None:
        """Log backend memory usage, if the backend can report it."""
        ...

    def at_set(self, x: Any, idx: Any, value: Any) -> Any:
        """Return ``x`` with ``value`` set at ``idx``."""
        ...

    def at_add(self, x: Any, idx: Any, value: Any) -> Any:
        """Return ``x`` with ``value`` added at ``idx``."""
        ...

    def at_sub(self, x: Any, idx: Any, value: Any) -> Any:
        """Return ``x`` with ``value`` subtracted at ``idx``."""
        ...

    def at_mul(self, x: Any, idx: Any, value: Any) -> Any:
        """Return ``x`` with ``value`` multiplied at ``idx``."""
        ...

    rank: int
    """Distributed rank for the current process."""

    size: int
    """Distributed world size."""

    is_distributed: bool
    """Whether this backend is running in distributed mode."""

    def barrier(self) -> None:
        """Synchronize distributed workers, if any."""
        ...

    def allreduce(self, x: Any, op: str = "sum") -> Any:
        """Reduce a value across distributed workers."""
        ...

    def broadcast(self, x: Any, root: int = 0) -> Any:
        """Broadcast a value from one distributed worker."""
        ...

    def gather(self, x: Any, root: int = 0) -> Any:
        """Gather values to one distributed worker."""
        ...

    def allgather(self, x: Any) -> Any:
        """Gather values to all distributed workers."""
        ...
