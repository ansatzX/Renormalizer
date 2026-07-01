# -*- coding: utf-8 -*-

"""Backend execution metadata and contraction planning primitives.

This module is intentionally tensor-network agnostic.  MPS, TTNS, dense
operators, future block tensors, and distributed tensors should all enter the
backend through operands with explicit modes and metadata rather than through
algorithm-specific branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import reduce
from operator import mul
from typing import Any, Callable, Hashable, Mapping, Sequence, Tuple


class BackendFeatureError(RuntimeError):
    """Raised when a backend feature is required but unavailable."""


class BackendCopyError(RuntimeError):
    """Raised when a requested copy policy cannot be satisfied."""


class CopyPolicy(Enum):
    NEVER = "never"
    IF_NEEDED = "if_needed"
    ALWAYS = "always"

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.IF_NEEDED
        return cls(str(value).lower())


class FallbackPolicy(Enum):
    FORBID = "forbid"
    WARN = "warn"
    RECORD = "record"
    SILENT = "silent"

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.RECORD
        return cls(str(value).lower())


@dataclass(frozen=True)
class DeviceSpec:
    kind: str
    index: int | None = None
    local_rank: int | None = None
    global_rank: int | None = None
    visible_id: str | None = None


def parse_device_spec(device) -> DeviceSpec | None:
    if device is None:
        return None
    if isinstance(device, DeviceSpec):
        return device
    if not isinstance(device, str):
        raise ValueError("Unknown backend device {0!r}. Expected 'cpu', 'gpu', 'cuda', or 'cuda:N'.".format(device))
    value = device.lower().strip()
    if value in ("auto", "default"):
        return None
    if value in ("cpu", "host"):
        return DeviceSpec(kind="cpu")
    if value in ("gpu", "cuda"):
        return DeviceSpec(kind="cuda")
    for prefix in ("cuda:", "gpu:"):
        if value.startswith(prefix):
            visible = value.split(":", 1)[1]
            try:
                index = int(visible)
            except ValueError:
                raise ValueError("Unknown backend device {0!r}. Device index must be an integer.".format(device))
            return DeviceSpec(kind="cuda", index=index, visible_id=visible)
    raise ValueError("Unknown backend device {0!r}. Expected 'cpu', 'gpu', 'cuda', or 'cuda:N'.".format(device))


def legacy_device_kind(spec: DeviceSpec | None):
    if spec is None:
        return None
    if spec.kind == "cpu":
        return "cpu"
    if spec.kind in ("cuda", "gpu", "rocm", "mps", "tpu"):
        return "gpu"
    return spec.kind


@dataclass(frozen=True)
class BackendCapabilities:
    cpu: bool = True
    gpu: bool = False
    autodiff: bool = False
    jit: bool = False
    sparse: bool = False
    functional_update: bool = True
    complex64: bool = True
    complex128: bool = True
    fp32: bool = True
    fp64: bool = True
    mixed_precision: bool = False
    device_index: bool = False
    streams: bool = False
    events: bool = False
    memory_pool: bool = False
    matmul: bool = True
    batched_matmul: bool = False
    grouped_gemm: bool = False
    strided_batched_gemm: bool = False
    einsum: bool = True
    contract_expression: bool = True
    contraction_path: bool = True
    custom_contraction_plan: bool = False
    block_sparse: bool = False
    packed_blocks: bool = False
    scatter_add: bool = False
    distributed: bool = False
    distributed_array: bool = False
    allreduce: bool = False
    allgather: bool = False
    reduce_scatter: bool = False
    alltoall: bool = False
    point_to_point: bool = False


@dataclass(frozen=True)
class ArrayInfo:
    shape: tuple[int, ...]
    dtype: Any
    itemsize: int
    ndim: int
    size: int
    nbytes: int
    device: DeviceSpec
    is_host: bool
    is_device: bool
    is_distributed: bool
    strides: tuple[int, ...] | None
    order: str
    contiguous: bool
    writeable: bool
    owns_data: bool | None
    backend_name: str


@dataclass(frozen=True)
class LayoutSpec:
    logical_shape: tuple[int, ...]
    physical_shape: tuple[int, ...] | None
    logical_modes: tuple[Hashable, ...]
    strides: tuple[int, ...] | None
    order: str
    contiguous_groups: tuple[tuple[int, ...], ...]
    requires_transpose: bool = False
    transpose_perm: tuple[int, ...] | None = None
    estimated_copy_bytes: int = 0


def _prod(values) -> int:
    values = tuple(values)
    if not values:
        return 1
    return int(reduce(mul, values, 1))


def _shape_of(array) -> tuple[int, ...]:
    return tuple(int(dim) for dim in getattr(array, "shape", ()))


def _nbytes_of(array) -> int:
    nbytes = getattr(array, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    dtype = getattr(array, "dtype", None)
    itemsize = int(getattr(dtype, "itemsize", 0) or 0)
    return _prod(_shape_of(array)) * itemsize


def _itemsize_of(array) -> int:
    dtype = getattr(array, "dtype", None)
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is not None:
        return int(itemsize)
    nbytes = _nbytes_of(array)
    size = _prod(_shape_of(array))
    return int(nbytes // size) if size else 0


def _array_flags(array):
    flags = getattr(array, "flags", None)
    if flags is None:
        return {}
    result = {}
    for key in ("C_CONTIGUOUS", "F_CONTIGUOUS", "WRITEABLE", "OWNDATA"):
        try:
            result[key] = bool(flags[key])
        except Exception:
            result[key] = bool(getattr(flags, key.lower(), False))
    return result


def array_info_for_backend(backend, array, device: DeviceSpec) -> ArrayInfo:
    shape = _shape_of(array)
    flags = _array_flags(array)
    strides = getattr(array, "strides", None)
    if strides is not None:
        strides = tuple(int(stride) for stride in strides)
    c_contiguous = bool(flags.get("C_CONTIGUOUS", False))
    f_contiguous = bool(flags.get("F_CONTIGUOUS", False))
    if c_contiguous:
        order = "C"
    elif f_contiguous:
        order = "F"
    else:
        order = "unknown"
    return ArrayInfo(
        shape=shape,
        dtype=getattr(array, "dtype", None),
        itemsize=_itemsize_of(array),
        ndim=len(shape),
        size=_prod(shape),
        nbytes=_nbytes_of(array),
        device=device,
        is_host=backend.is_host_array(array),
        is_device=backend.is_device_array(array),
        is_distributed=getattr(backend, "is_distributed_array", lambda _: False)(array),
        strides=strides,
        order=order,
        contiguous=c_contiguous or f_contiguous,
        writeable=bool(flags.get("WRITEABLE", True)),
        owns_data=flags.get("OWNDATA"),
        backend_name=backend.name,
    )


def layout_from_array(array, modes=None) -> LayoutSpec:
    shape = _shape_of(array)
    strides = getattr(array, "strides", None)
    if strides is not None:
        strides = tuple(int(stride) for stride in strides)
    flags = _array_flags(array)
    if flags.get("C_CONTIGUOUS", False):
        order = "C"
    elif flags.get("F_CONTIGUOUS", False):
        order = "F"
    else:
        order = "unknown"
    return LayoutSpec(
        logical_shape=shape,
        physical_shape=shape,
        logical_modes=tuple(range(len(shape))) if modes is None else tuple(modes),
        strides=strides,
        order=order,
        contiguous_groups=(tuple(range(len(shape))),) if order in ("C", "F") else (),
        estimated_copy_bytes=0 if order in ("C", "F") else _nbytes_of(array),
    )


@dataclass(frozen=True)
class TensorOperand:
    array: Any
    modes: tuple[Hashable, ...]
    layout: LayoutSpec | None = None
    name: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "modes", tuple(self.modes))
        if self.layout is None:
            object.__setattr__(self, "layout", layout_from_array(self.array, self.modes))


@dataclass(frozen=True)
class EinsumSpec:
    operands: tuple[TensorOperand, ...]
    output_modes: tuple[Hashable, ...]
    constants: tuple[int, ...] = ()
    optimize: str | Any | None = None


@dataclass(frozen=True)
class PairContractionSpec:
    left: TensorOperand
    right: TensorOperand
    output_modes: tuple[Hashable, ...]
    left_batch_modes: tuple[Hashable, ...]
    right_batch_modes: tuple[Hashable, ...]
    contracted_modes: tuple[Hashable, ...]
    left_only_modes: tuple[Hashable, ...]
    right_only_modes: tuple[Hashable, ...]
    output_layout: LayoutSpec | None = None

    @classmethod
    def from_operands(cls, left: TensorOperand, right: TensorOperand, output_modes):
        output_modes = tuple(output_modes)
        right_modes = set(right.modes)
        left_modes = set(left.modes)
        output_set = set(output_modes)
        batch_modes = tuple(mode for mode in output_modes if mode in left_modes and mode in right_modes)
        contracted_modes = tuple(mode for mode in left.modes if mode in right_modes and mode not in output_set)
        left_only_modes = tuple(mode for mode in output_modes if mode in left_modes and mode not in right_modes)
        right_only_modes = tuple(mode for mode in output_modes if mode in right_modes and mode not in left_modes)
        return cls(
            left=left,
            right=right,
            output_modes=output_modes,
            left_batch_modes=batch_modes,
            right_batch_modes=batch_modes,
            contracted_modes=contracted_modes,
            left_only_modes=left_only_modes,
            right_only_modes=right_only_modes,
        )


@dataclass(frozen=True)
class MatmulDesc:
    A: Any
    B: Any
    C: Any | None
    m: int
    n: int
    k: int
    batch_shape: tuple[int, ...] = ()
    trans_a: bool = False
    trans_b: bool = False
    conj_a: bool = False
    conj_b: bool = False
    alpha: complex | float = 1.0
    beta: complex | float = 0.0
    dtype_compute: Any | None = None
    dtype_output: Any | None = None
    layout_a: LayoutSpec | None = None
    layout_b: LayoutSpec | None = None
    layout_c: LayoutSpec | None = None
    estimated_flops: int = 0
    estimated_read_bytes: int = 0
    estimated_write_bytes: int = 0
    estimated_workspace_bytes: int = 0


@dataclass(frozen=True)
class LayoutTransform:
    kind: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    copy_bytes: int = 0
    reason: str | None = None


@dataclass(frozen=True)
class MatmulPlan:
    kind: str
    descs: tuple[MatmulDesc, ...]
    pre_ops: tuple[LayoutTransform, ...]
    post_ops: tuple[LayoutTransform, ...]
    output_shape: tuple[int, ...]
    copy_bytes: int
    workspace_bytes: int
    estimated_flops: int
    estimated_time_s: float | None
    reason: str
    fallback_reason: str | None = None


def _mode_sizes(operand: TensorOperand) -> Mapping[Hashable, int]:
    return {mode: int(dim) for mode, dim in zip(operand.modes, _shape_of(operand.array))}


def _sizes_for_modes(modes, *maps):
    sizes = []
    for mode in modes:
        found = False
        for mapping in maps:
            if mode in mapping:
                sizes.append(mapping[mode])
                found = True
                break
        if not found:
            raise ValueError("mode {0!r} is not present in contraction operands".format(mode))
    return tuple(sizes)


def lower_pair_contraction_to_matmul(
    spec: PairContractionSpec,
    capabilities: BackendCapabilities,
) -> MatmulPlan:
    left_sizes = _mode_sizes(spec.left)
    right_sizes = _mode_sizes(spec.right)
    batch_shape = _sizes_for_modes(spec.left_batch_modes, left_sizes, right_sizes)
    m = _prod(_sizes_for_modes(spec.left_only_modes, left_sizes))
    n = _prod(_sizes_for_modes(spec.right_only_modes, right_sizes))
    k_left = _prod(_sizes_for_modes(spec.contracted_modes, left_sizes))
    k_right = _prod(_sizes_for_modes(spec.contracted_modes, right_sizes))
    if k_left != k_right:
        raise ValueError("contracted mode sizes do not match")
    output_shape = _sizes_for_modes(spec.output_modes, left_sizes, right_sizes)
    flops = int(2 * _prod(batch_shape) * m * n * k_left)
    read_bytes = _nbytes_of(spec.left.array) + _nbytes_of(spec.right.array)
    write_bytes = _prod(output_shape) * max(_itemsize_of(spec.left.array), _itemsize_of(spec.right.array))
    desc = MatmulDesc(
        A=spec.left.array,
        B=spec.right.array,
        C=None,
        m=m,
        n=n,
        k=k_left,
        batch_shape=batch_shape,
        layout_a=spec.left.layout,
        layout_b=spec.right.layout,
        estimated_flops=flops,
        estimated_read_bytes=read_bytes,
        estimated_write_bytes=write_bytes,
    )

    if not batch_shape:
        if capabilities.matmul:
            return MatmulPlan(
                kind="gemm",
                descs=(desc,),
                pre_ops=(),
                post_ops=(),
                output_shape=output_shape,
                copy_bytes=0,
                workspace_bytes=0,
                estimated_flops=flops,
                estimated_time_s=None,
                reason="pair contraction lowered to GEMM",
            )
        reason = "backend lacks matmul"
    elif capabilities.strided_batched_gemm:
        return MatmulPlan(
            kind="strided_batched_gemm",
            descs=(desc,),
            pre_ops=(),
            post_ops=(),
            output_shape=output_shape,
            copy_bytes=0,
            workspace_bytes=0,
            estimated_flops=flops,
            estimated_time_s=None,
            reason="pair contraction lowered to strided batched GEMM",
        )
    elif capabilities.batched_matmul:
        return MatmulPlan(
            kind="batched_gemm",
            descs=(desc,),
            pre_ops=(),
            post_ops=(),
            output_shape=output_shape,
            copy_bytes=0,
            workspace_bytes=0,
            estimated_flops=flops,
            estimated_time_s=None,
            reason="pair contraction lowered to batched GEMM",
        )
    else:
        reason = "backend lacks batched_matmul for batch shape {0}".format(batch_shape)

    return MatmulPlan(
        kind="fallback_tensordot",
        descs=(desc,),
        pre_ops=(),
        post_ops=(),
        output_shape=output_shape,
        copy_bytes=0,
        workspace_bytes=0,
        estimated_flops=flops,
        estimated_time_s=None,
        reason=reason,
        fallback_reason=reason,
    )


@dataclass
class DenseBlock:
    key: Any
    array: Any
    modes: tuple[Hashable, ...]
    shape: tuple[int, ...]
    offset: tuple[int, ...] | None = None


@dataclass
class BlockTensor:
    blocks: dict[Any, DenseBlock]
    global_shape: tuple[int, ...]
    modes: tuple[Hashable, ...]
    block_axis_meta: Any
    backend: str


@dataclass(frozen=True)
class BlockContractionSpec:
    left: BlockTensor
    right: BlockTensor
    output_modes: tuple[Hashable, ...]
    qn_rule: Callable[[Any, Any], Any | None]
    accumulate: bool = True
