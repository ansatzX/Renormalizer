# -*- coding: utf-8 -*-

from typing import Any, Tuple, Type, Union


_BACKEND_PROTOCOL_RUNTIME_ATTRS = (
    "name",
    "config",
    "device",
    "device_spec",
    "capabilities",
    "fallback_policy",
    "supported_device_kinds",
    "available_device_kinds",
    "supports_cpu",
    "array_namespace",
    "opt_einsum_name",
    "supports_gpu",
    "supports_autodiff",
    "supports_jit",
    "supports_sparse",
    "supports_functional_update",
    "supports_complex64",
    "supports_complex128",
    "supports_fp32",
    "supports_fp64",
    "supports_mixed_precision",
    "supports_device_index",
    "supports_matmul",
    "supports_batched_matmul",
    "supports_grouped_gemm",
    "supports_strided_batched_gemm",
    "supports_einsum",
    "supports_contract_expression",
    "supports_contraction_path",
    "supports_custom_contraction_plan",
    "supports_streams",
    "supports_events",
    "supports_memory_pool",
    "supports_block_sparse",
    "supports_packed_blocks",
    "supports_scatter_add",
    "supports_distributed",
    "supports_distributed_array",
    "supports_allreduce",
    "supports_broadcast",
    "supports_allgather",
    "supports_reduce_scatter",
    "supports_alltoall",
    "supports_point_to_point",
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
    "tensordot",
    "einsum",
    "is_array",
    "is_host_array",
    "is_device_array",
    "is_distributed_array",
    "array_info",
    "layout",
    "permute",
    "reshape_view",
    "can_reshape_view",
    "make_contiguous",
    "parse_einsum",
    "contract",
    "contract_expression",
    "contract_path",
    "plan_contraction",
    "plan_distributed_contraction_path",
    "estimate_matmul",
    "estimate_contraction",
    "estimate_redistribute",
    "default_stream",
    "new_stream",
    "record_event",
    "wait_event",
    "allocate_workspace",
    "release_workspace",
    "execute",
    "last_execution_profile",
    "unpack_masked_vectors",
    "pack_masked_vectors",
    "shard_tensor",
    "gather_tensor",
    "redistribute",
    "replicate_tensor",
    "distributed_contract",
    "astype",
    "ascontiguousarray",
    "current_device",
    "set_device",
    "device_count",
    "lower_pair_contraction_to_matmul",
    "lower_block_contraction",
    "execute_matmul_plan",
    "execute_grouped_gemm_plan",
    "matmul",
    "batched_matmul",
    "grouped_gemm",
    "gemv_batch",
    "prepack_grouped_gemm",
    "execute_prepacked_grouped_gemm",
    "synchronize",
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
    "reduce_scatter",
    "alltoall",
    "send",
    "recv",
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

    config: Any
    """Explicit backend configuration used to construct this backend."""

    device: str
    """Configured compute device kind, such as ``"cpu"`` or ``"gpu"``."""

    device_spec: Any
    """Structured device placement descriptor."""

    capabilities: Any
    """Explicit backend execution capabilities."""

    fallback_policy: Any
    """Policy controlling unsupported execution primitive fallback."""

    supported_device_kinds: Tuple[str, ...]
    """Device kinds supported by this backend implementation."""

    available_device_kinds: Tuple[str, ...]
    """Device kinds available in the current runtime."""

    supports_cpu: bool
    """Whether this backend implementation can execute on CPU devices."""

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

    supports_complex64: bool
    """Whether complex64 arrays are supported."""

    supports_complex128: bool
    """Whether complex128 arrays are supported."""

    supports_fp32: bool
    """Whether float32 arrays are supported."""

    supports_fp64: bool
    """Whether float64 arrays are supported."""

    supports_mixed_precision: bool
    """Whether mixed-precision execution is natively supported."""

    supports_device_index: bool
    """Whether indexed device selection is supported."""

    supports_matmul: bool
    """Whether dense matrix multiplication primitives are available."""

    supports_batched_matmul: bool
    """Whether same-shape stacked batched matmul is available."""

    supports_grouped_gemm: bool
    """Whether a backend-owned grouped GEMM entrypoint is available.

    This means the backend owns grouping/dispatch and does not rely on the
    AbstractBackend bucketed fallback.  The implementation may still be a
    bucketed batched-matmul path; vendor grouped BLAS is reported separately by
    profiling policy/implementation fields.
    """

    supports_strided_batched_gemm: bool
    """Whether native strided batched GEMM is available."""

    supports_einsum: bool
    """Whether direct einsum execution is available."""

    supports_contract_expression: bool
    """Whether reusable contraction expression execution is available."""

    supports_contraction_path: bool
    """Whether contraction path planning is available."""

    supports_custom_contraction_plan: bool
    """Whether backend execution plans can be produced and executed."""

    supports_streams: bool
    """Whether backend stream primitives are available."""

    supports_events: bool
    """Whether backend event primitives are available."""

    supports_memory_pool: bool
    """Whether native backend memory-pool hooks are available."""

    supports_block_sparse: bool
    """Whether backend block-sparse planning/execution APIs are available."""

    supports_packed_blocks: bool
    """Whether native packed-block storage is available."""

    supports_scatter_add: bool
    """Whether backend scatter-add/update primitives are available."""

    supports_distributed: bool
    """Whether this backend is currently participating in a distributed runtime."""

    supports_distributed_array: bool
    """Whether backend distributed tensor wrapper APIs are available."""

    supports_allreduce: bool
    """Whether allreduce communication primitives are available when distributed."""

    supports_broadcast: bool
    """Whether broadcast communication primitives are available when distributed."""

    supports_allgather: bool
    """Whether allgather communication primitives are available when distributed."""

    supports_reduce_scatter: bool
    """Whether reduce-scatter communication primitives are available when distributed."""

    supports_alltoall: bool
    """Whether all-to-all communication primitives are available when distributed."""

    supports_point_to_point: bool
    """Whether distributed point-to-point communication primitives are available."""

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

    def to_host(self, x: Any, *, copy: Any = None) -> Any:
        """Convert a backend array to a host-resident array."""
        ...

    def to_backend(self, x: Any, *, device: Any = None, dtype: Any = None, copy: Any = None) -> Any:
        """Convert an array-like object to this backend's representation."""
        ...

    def tensordot(
        self,
        a: Any,
        b: Any,
        axes: Any = 2,
        *,
        stream: Any = None,
        workspace: Any = None,
    ) -> Any:
        """Contract two tensors through the backend boundary."""
        ...

    def einsum(self, subscripts: Any, *operands: Any, **kwargs: Any) -> Any:
        """Execute an einsum contraction through the backend boundary."""
        ...

    def astype(self, x: Any, dtype: Any, *, copy: Any = None) -> Any:
        """Convert dtype using an explicit copy policy."""
        ...

    def ascontiguousarray(self, x: Any, *, copy: Any = None) -> Any:
        """Return a contiguous backend array using an explicit copy policy."""
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

    def is_distributed_array(self, x: Any) -> bool:
        """Return whether ``x`` is distributed across ranks/devices."""
        ...

    def array_info(self, x: Any) -> Any:
        """Return shape, dtype, device, and layout metadata for an array."""
        ...

    def layout(self, x: Any) -> Any:
        """Return backend layout metadata for an array."""
        ...

    def permute(self, x: Any, perm: Any, *, copy_policy: Any = None) -> Any:
        """Return an array with axes reordered by ``perm``."""
        ...

    def reshape_view(self, x: Any, shape: Any) -> Any:
        """Reshape an array only when the reshape can be represented as a view."""
        ...

    def can_reshape_view(self, x: Any, shape: Any) -> bool:
        """Return whether ``reshape_view`` can satisfy the requested shape."""
        ...

    def make_contiguous(self, x: Any, *, mode_groups: Any = None, copy_policy: Any = None) -> Any:
        """Return a C-contiguous backend array using an explicit copy policy."""
        ...

    def parse_einsum(self, equation: str, *operands: Any, constants: Any = (), optimize: Any = None) -> Any:
        """Parse an explicit einsum equation into backend contraction IR."""
        ...

    def contract(self, *args: Any, **kwargs: Any) -> Any:
        """Execute an opt_einsum contraction through this backend."""
        ...

    def contract_expression(self, *args: Any, **kwargs: Any) -> Any:
        """Create an opt_einsum expression that defaults to this backend."""
        ...

    def contract_path(self, *args: Any, **kwargs: Any) -> Any:
        """Plan an opt_einsum contraction path through the backend surface."""
        ...

    def plan_contraction(
        self,
        spec: Any,
        *,
        memory_limit: Any = None,
        prefer: str = "balanced",
        allow_slicing: bool = True,
        allow_distribution: bool = False,
        target_devices: Any = None,
        record_profile: bool = True,
    ) -> Any:
        """Plan an explicit contraction into backend execution steps."""
        ...

    def plan_distributed_contraction_path(
        self,
        path: Any,
        mesh: Any,
        *,
        memory_limit_per_device: Any = None,
        cost_model: Any = None,
    ) -> Any:
        """Plan distributed placement/communication for a contraction path."""
        ...

    def estimate_matmul(self, desc: Any, hw: Any = None) -> Any:
        """Estimate cost for one matmul descriptor or matmul plan."""
        ...

    def estimate_contraction(self, plan: Any, hw: Any = None) -> Any:
        """Estimate cost for a contraction plan."""
        ...

    def estimate_redistribute(
        self,
        src: Any,
        dst: Any,
        tensor_shape: Any,
        hw: Any = None,
        *,
        itemsize: int = 8,
    ) -> Any:
        """Estimate communication/copy cost for a redistribution."""
        ...

    def default_stream(self) -> Any:
        """Return the backend default stream token, or None for synchronous backends."""
        ...

    def new_stream(self) -> Any:
        """Create a new backend stream token, or None when streams are unavailable."""
        ...

    def record_event(self, stream: Any = None) -> Any:
        """Record a backend event on the given stream."""
        ...

    def wait_event(self, event: Any, stream: Any = None) -> Any:
        """Wait for a backend event from the given stream."""
        ...

    def allocate_workspace(self, nbytes: int, *, device: Any = None) -> Any:
        """Allocate a backend-resident workspace buffer."""
        ...

    def release_workspace(self, workspace: Any) -> Any:
        """Release a workspace buffer."""
        ...

    def execute(self, plan: Any, *, stream: Any = None, workspace: Any = None) -> Any:
        """Execute a backend plan through the unified plan executor."""
        ...

    def last_execution_profile(self) -> Any:
        """Return the latest structured execution profile payload, if any."""
        ...

    def unpack_masked_vectors(self, x: Any, spec: Any) -> Any:
        """Unpack one or more packed masked vectors into a structured center tensor."""
        ...

    def pack_masked_vectors(self, x_struct: Any, spec: Any) -> Any:
        """Pack a structured center tensor through its mask into vector form."""
        ...

    def shard_tensor(self, x: Any, spec: Any) -> Any:
        """Create a distributed tensor from a dense tensor and sharding spec."""
        ...

    def gather_tensor(self, x: Any, root: Any = None) -> Any:
        """Gather a distributed tensor to a dense tensor."""
        ...

    def redistribute(self, x: Any, new_spec: Any) -> Any:
        """Redistribute a distributed tensor according to a new sharding spec."""
        ...

    def replicate_tensor(self, x: Any, mesh: Any, *, modes: Any = None) -> Any:
        """Replicate a dense tensor across a device mesh."""
        ...

    def distributed_contract(self, spec: Any, *, plan: Any = None, stream: Any = None, workspace: Any = None) -> Any:
        """Execute a distributed contraction spec and return dense or distributed output."""
        ...

    def current_device(self) -> Any:
        """Return the structured current device descriptor."""
        ...

    def set_device(self, device: Any) -> None:
        """Set the active backend device."""
        ...

    def device_count(self) -> int:
        """Return the number of available devices for this backend."""
        ...

    def lower_pair_contraction_to_matmul(self, spec: Any, *, record_profile: bool = True) -> Any:
        """Lower a mode-labeled pair contraction to a matmul execution plan."""
        ...

    def lower_block_contraction(self, spec: Any) -> Any:
        """Lower a block-sparse contraction to a grouped GEMM plan."""
        ...

    def execute_matmul_plan(
        self,
        plan: Any,
        *,
        stream: Any = None,
        workspace: Any = None,
        pack_threshold: int | None = None,
        plan_hash: Any = None,
        record_profile: bool = True,
        equation: Any = None,
        input_modes: Any = None,
        output_modes: Any = None,
        fallback_policy: Any = None,
    ) -> Any:
        """Execute a matmul plan using this backend's primitives or recorded fallback."""
        ...

    def execute_grouped_gemm_plan(
        self,
        plan: Any,
        *,
        pack_threshold: int = 4,
        stream: Any = None,
        workspace: Any = None,
        policy: Any = "auto",
        fallback_policy: Any = None,
    ) -> Any:
        """Execute a grouped GEMM plan and return a sparse block tensor."""
        ...

    def matmul(
        self,
        A: Any,
        B: Any = None,
        *,
        C: Any = None,
        trans_a: bool = False,
        trans_b: bool = False,
        conj_a: bool = False,
        conj_b: bool = False,
        alpha: Any = 1.0,
        beta: Any = 0.0,
        stream: Any = None,
        workspace: Any = None,
        fallback_policy: Any = None,
    ) -> Any:
        """Execute one dense GEMM, or a descriptor for compatibility."""
        ...

    def batched_matmul(
        self,
        A: Any,
        B: Any = None,
        *,
        C: Any = None,
        trans_a: bool = False,
        trans_b: bool = False,
        conj_a: bool = False,
        conj_b: bool = False,
        alpha: Any = 1.0,
        beta: Any = 0.0,
        stream: Any = None,
        workspace: Any = None,
        fallback_policy: Any = None,
    ) -> Any:
        """Execute same-shape stacked batched GEMM, or a descriptor for compatibility."""
        ...

    def grouped_gemm(
        self,
        descs: Any,
        *,
        buffers: Any = None,
        pack_threshold: int = 4,
        stream: Any = None,
        workspace: Any = None,
        policy: Any = "auto",
        fallback_policy: Any = None,
        profile_context: Any = None,
    ) -> Any:
        """Execute grouped GEMM tasks through native support or bucketed fallback."""
        ...

    def gemv_batch(
        self,
        descs: Any,
        *,
        buffers: Any = None,
        role: Any = None,
        stream: Any = None,
        workspace: Any = None,
        fallback_policy: Any = None,
        profile_context: Any = None,
    ) -> Any:
        """Execute a FOCUS-style GEMV batch and record primitive metadata."""
        ...

    def prepack_grouped_gemm(
        self,
        descs: Any,
        *,
        buffers: Any = None,
        pack_threshold: int = 4,
        stream: Any = None,
        workspace: Any = None,
        policy: Any = "auto",
        fallback_policy: Any = None,
    ) -> Any:
        """Pack reusable same-shape grouped GEMM buckets once for repeated execution."""
        ...

    def execute_prepacked_grouped_gemm(
        self,
        plan: Any,
        *,
        stream: Any = None,
        workspace: Any = None,
        fallback_policy: Any = None,
    ) -> Any:
        """Execute a prepacked grouped GEMM plan without repacking A/B buckets."""
        ...

    def synchronize(self, device: Any = None, stream: Any = None) -> Any:
        """Synchronize backend work, optionally scoped to a device or stream."""
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

    def allgather(self, x: Any, axis: Any = None) -> Any:
        """Gather values to all workers, or concatenate tensor shards along ``axis``."""
        ...

    def reduce_scatter(self, x: Any, op: str = "sum", axis: Any = 0) -> Any:
        """Reduce and scatter a value across distributed workers."""
        ...

    def alltoall(self, x: Any, split_axis: Any = 0, concat_axis: Any = 0) -> Any:
        """Exchange tensor chunks across distributed workers."""
        ...

    def send(self, x: Any, *, dst: int, tag: int = 0) -> None:
        """Send a value to a peer distributed worker."""
        ...

    def recv(self, *, src: int, tag: int = 0, like: Any = None) -> Any:
        """Receive a value from a peer distributed worker."""
        ...
