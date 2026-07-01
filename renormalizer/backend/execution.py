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
import hashlib
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


def _split_slice(dim: int, parts: int, index: int) -> slice:
    base = int(dim) // int(parts)
    remainder = int(dim) % int(parts)
    start = int(index) * base + min(int(index), remainder)
    stop = start + base + (1 if int(index) < remainder else 0)
    return slice(start, stop)


def _rank_coordinates(rank: int, parts_by_mode: tuple[int, ...]) -> tuple[int, ...]:
    coords = []
    remaining = int(rank)
    for parts in reversed(parts_by_mode):
        coords.append(remaining % int(parts))
        remaining //= int(parts)
    return tuple(reversed(coords))


@dataclass(frozen=True)
class DeviceMesh:
    devices: tuple[DeviceSpec, ...]
    shape: tuple[int, ...]
    axis_names: tuple[str, ...]
    backend: str
    local_rank: int = 0
    global_rank: int = 0
    world_size: int | None = None

    def __post_init__(self):
        devices = tuple(parse_device_spec(device) if not isinstance(device, DeviceSpec) else device for device in self.devices)
        shape = tuple(int(dim) for dim in self.shape)
        axis_names = tuple(str(name) for name in self.axis_names)
        if any(dim <= 0 for dim in shape):
            raise ValueError("DeviceMesh shape dimensions must be positive")
        if len(axis_names) != len(shape):
            raise ValueError("DeviceMesh axis_names must match mesh rank")
        if _prod(shape) != len(devices):
            raise ValueError("DeviceMesh shape product must match number of devices")
        world_size = len(devices) if self.world_size is None else int(self.world_size)
        if world_size != len(devices):
            raise ValueError("DeviceMesh world_size must match number of devices")
        local_rank = int(self.local_rank)
        global_rank = int(self.global_rank)
        if local_rank < 0 or local_rank >= world_size:
            raise ValueError("DeviceMesh local_rank is out of range")
        if global_rank < 0 or global_rank >= world_size:
            raise ValueError("DeviceMesh global_rank is out of range")
        object.__setattr__(self, "devices", devices)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "axis_names", axis_names)
        object.__setattr__(self, "local_rank", local_rank)
        object.__setattr__(self, "global_rank", global_rank)
        object.__setattr__(self, "world_size", world_size)


@dataclass(frozen=True)
class ShardingSpec:
    global_shape: tuple[int, ...]
    modes: tuple[Hashable, ...]
    mesh: DeviceMesh
    ranks_per_mode: dict[Hashable, int]
    mode_to_mesh_axis: dict[Hashable, str]
    replicated_modes: tuple[Hashable, ...] = ()
    sharded_modes: tuple[Hashable, ...] = ()
    local_slices: dict[int, tuple[slice, ...]] | None = None

    def __post_init__(self):
        global_shape = tuple(int(dim) for dim in self.global_shape)
        modes = tuple(self.modes)
        if len(global_shape) != len(modes):
            raise ValueError("ShardingSpec modes must match global_shape rank")
        ranks_per_mode = {mode: int(count) for mode, count in self.ranks_per_mode.items()}
        if any(count <= 0 for count in ranks_per_mode.values()):
            raise ValueError("ranks_per_mode values must be positive")
        unknown_modes = set(ranks_per_mode) - set(modes)
        if unknown_modes:
            raise ValueError("ranks_per_mode contains modes not present in tensor modes: {0}".format(sorted(unknown_modes)))
        mode_to_mesh_axis = dict(self.mode_to_mesh_axis)
        unknown_axis_modes = set(mode_to_mesh_axis) - set(modes)
        if unknown_axis_modes:
            raise ValueError("mode_to_mesh_axis contains modes not present in tensor modes: {0}".format(sorted(unknown_axis_modes)))
        mesh_axis_to_index = {axis: index for index, axis in enumerate(self.mesh.axis_names)}
        unknown_axes = set(mode_to_mesh_axis.values()) - set(mesh_axis_to_index)
        if unknown_axes:
            raise ValueError("mode_to_mesh_axis contains unknown mesh axes: {0}".format(sorted(unknown_axes)))
        sharded_modes = tuple(self.sharded_modes) or tuple(mode for mode in modes if ranks_per_mode.get(mode, 1) > 1)
        replicated_modes = tuple(self.replicated_modes) or tuple(mode for mode in modes if mode not in sharded_modes)
        if set(sharded_modes) & set(replicated_modes):
            raise ValueError("sharded_modes and replicated_modes must not overlap")
        if set(sharded_modes) | set(replicated_modes) != set(modes):
            raise ValueError("sharded_modes and replicated_modes must cover all modes")
        mapped_axes = [mode_to_mesh_axis[mode] for mode in sharded_modes if mode in mode_to_mesh_axis]
        if len(mapped_axes) != len(set(mapped_axes)):
            raise ValueError("multiple sharded modes cannot map to the same mesh axis")
        for mode in sharded_modes:
            if mode not in mode_to_mesh_axis:
                continue
            axis = mode_to_mesh_axis[mode]
            axis_parts = self.mesh.shape[mesh_axis_to_index[axis]]
            requested_parts = ranks_per_mode.get(mode, axis_parts)
            if requested_parts != axis_parts:
                raise ValueError(
                    "ranks_per_mode for mode {0!r} must match mesh axis {1!r} size {2}"
                    .format(mode, axis, axis_parts)
                )
            ranks_per_mode[mode] = axis_parts
        shard_parts = tuple(ranks_per_mode.get(mode, 1) for mode in sharded_modes)
        shard_world = _prod(shard_parts)
        has_axis_mapping = all(mode in mode_to_mesh_axis for mode in sharded_modes)
        if sharded_modes and not has_axis_mapping and shard_world != self.mesh.world_size:
            raise ValueError("product of sharded ranks_per_mode must match mesh world_size")
        local_slices = self.local_slices
        if local_slices is None:
            local_slices = self._compute_local_slices(
                global_shape,
                modes,
                sharded_modes,
                shard_parts,
                mode_to_mesh_axis,
                mesh_axis_to_index,
            )
        else:
            local_slices = {int(rank): tuple(slices) for rank, slices in local_slices.items()}
        object.__setattr__(self, "global_shape", global_shape)
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "ranks_per_mode", ranks_per_mode)
        object.__setattr__(self, "mode_to_mesh_axis", mode_to_mesh_axis)
        object.__setattr__(self, "replicated_modes", replicated_modes)
        object.__setattr__(self, "sharded_modes", sharded_modes)
        object.__setattr__(self, "local_slices", local_slices)

    def _compute_local_slices(self, global_shape, modes, sharded_modes, shard_parts, mode_to_mesh_axis, mesh_axis_to_index):
        if not sharded_modes:
            full_slice = tuple(slice(None) for _ in global_shape)
            return {rank: full_slice for rank in range(self.mesh.world_size)}
        if all(mode in mode_to_mesh_axis for mode in sharded_modes):
            result = {}
            for rank in range(self.mesh.world_size):
                mesh_coords = _rank_coordinates(rank, self.mesh.shape)
                slices = []
                for axis, mode in enumerate(modes):
                    if mode in sharded_modes:
                        mesh_axis = mode_to_mesh_axis[mode]
                        mesh_position = mesh_axis_to_index[mesh_axis]
                        parts = self.mesh.shape[mesh_position]
                        coord = mesh_coords[mesh_position]
                        slices.append(_split_slice(global_shape[axis], parts, coord))
                    else:
                        slices.append(slice(None))
                result[rank] = tuple(slices)
            return result
        sharded_mode_to_position = {mode: index for index, mode in enumerate(sharded_modes)}
        result = {}
        for rank in range(self.mesh.world_size):
            coords = _rank_coordinates(rank, shard_parts)
            slices = []
            for axis, mode in enumerate(modes):
                if mode in sharded_mode_to_position:
                    position = sharded_mode_to_position[mode]
                    slices.append(_split_slice(global_shape[axis], shard_parts[position], coords[position]))
                else:
                    slices.append(slice(None))
            result[rank] = tuple(slices)
        return result


@dataclass
class DistributedTensor:
    local_array: Any
    global_shape: tuple[int, ...]
    modes: tuple[Hashable, ...]
    sharding: ShardingSpec
    mesh: DeviceMesh
    dtype: Any | None = None
    local_shape: tuple[int, ...] = ()
    local_nbytes: int = 0
    rank_local_arrays: dict[int, Any] | None = None

    def __post_init__(self):
        self.global_shape = tuple(int(dim) for dim in self.global_shape)
        self.modes = tuple(self.modes)
        if self.dtype is None:
            self.dtype = getattr(self.local_array, "dtype", None)
        if not self.local_shape:
            self.local_shape = _shape_of(self.local_array)
        else:
            self.local_shape = tuple(int(dim) for dim in self.local_shape)
        if not self.local_nbytes:
            self.local_nbytes = _nbytes_of(self.local_array)
        if self.rank_local_arrays is not None:
            self.rank_local_arrays = {int(rank): array for rank, array in self.rank_local_arrays.items()}

    @property
    def shape(self):
        return self.global_shape

    @property
    def ndim(self):
        return len(self.global_shape)

    @property
    def nbytes(self):
        itemsize = int(getattr(self.dtype, "itemsize", 0) or 0)
        return _prod(self.global_shape) * itemsize


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

    def __post_init__(self):
        object.__setattr__(self, "operands", tuple(self.operands))
        object.__setattr__(self, "output_modes", tuple(self.output_modes))
        object.__setattr__(self, "constants", tuple(int(index) for index in self.constants))


def _parse_einsum_modes(modes, *, context):
    if "..." in modes:
        raise ValueError("backend.parse_einsum requires explicit modes; ellipsis is not supported")
    return tuple(modes)


def parse_einsum_equation(equation) -> tuple[tuple[tuple[Hashable, ...], ...], tuple[Hashable, ...]]:
    normalized = "".join(str(equation).split())
    if normalized.count("->") != 1:
        raise ValueError("einsum equation requires an explicit output using '->'")
    lhs, rhs = normalized.split("->", 1)
    if not lhs:
        raise ValueError("einsum equation requires at least one input operand")
    return (
        tuple(_parse_einsum_modes(term, context="operand") for term in lhs.split(",")),
        _parse_einsum_modes(rhs, context="output"),
    )


def parse_einsum(equation, *operands, constants=(), optimize=None) -> EinsumSpec:
    input_modes, output_modes = parse_einsum_equation(equation)
    if len(input_modes) != len(operands):
        raise ValueError(
            "einsum operand count mismatch: equation has {0} operands but {1} arrays were provided"
            .format(len(input_modes), len(operands))
        )

    constants = tuple(int(index) for index in constants)
    for index in constants:
        if index < 0 or index >= len(operands):
            raise ValueError("einsum constant operand index {0} is out of range".format(index))

    tensor_operands = []
    input_mode_set = set()
    for index, (modes, array) in enumerate(zip(input_modes, operands)):
        shape = _shape_of(array)
        if len(modes) != len(shape):
            raise ValueError(
                "einsum operand {0} rank mismatch: equation has {1} modes but array rank is {2}"
                .format(index, len(modes), len(shape))
            )
        input_mode_set.update(modes)
        tensor_operands.append(TensorOperand(array, modes, name="operand{0}".format(index)))

    if len(output_modes) != len(set(output_modes)):
        raise ValueError("einsum output modes must be unique")
    missing = [mode for mode in output_modes if mode not in input_mode_set]
    if missing:
        raise ValueError("einsum output modes are not present in any input: {0}".format(missing))

    return EinsumSpec(
        operands=tuple(tensor_operands),
        output_modes=output_modes,
        constants=constants,
        optimize=optimize,
    )


@dataclass(frozen=True)
class PackedVectorSpec:
    qn_mask: Any
    center_shape: tuple[int, ...]
    batch_axis: int = -1
    packed_dim: int = 0
    nrhs: int = 1

    def __post_init__(self):
        center_shape = tuple(int(dim) for dim in self.center_shape)
        packed_dim = int(self.packed_dim)
        nrhs = int(self.nrhs)
        if any(dim < 0 for dim in center_shape):
            raise ValueError("center_shape dimensions must be non-negative")
        if packed_dim < 0:
            raise ValueError("packed_dim must be non-negative")
        if nrhs < 1:
            raise ValueError("nrhs must be positive")
        object.__setattr__(self, "center_shape", center_shape)
        object.__setattr__(self, "batch_axis", int(self.batch_axis))
        object.__setattr__(self, "packed_dim", packed_dim)
        object.__setattr__(self, "nrhs", nrhs)


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


@dataclass(frozen=True)
class ContractionStep:
    kind: str
    inputs: tuple[int, ...]
    output: int
    input_modes: tuple[tuple[Hashable, ...], ...]
    output_modes: tuple[Hashable, ...]
    plan: Any = None
    estimated_flops: int = 0
    estimated_read_bytes: int = 0
    estimated_write_bytes: int = 0
    estimated_copy_bytes: int = 0
    estimated_peak_bytes: int = 0
    estimated_comm_bytes: int = 0
    required_workspace_bytes: int = 0
    reason: str | None = None
    fallback_reason: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "inputs", tuple(int(index) for index in self.inputs))
        object.__setattr__(self, "input_modes", tuple(tuple(modes) for modes in self.input_modes))
        object.__setattr__(self, "output_modes", tuple(self.output_modes))


@dataclass(frozen=True)
class ContractionPlan:
    steps: tuple[ContractionStep, ...]
    input_specs: tuple[TensorOperand, ...]
    output_modes: tuple[Hashable, ...]
    estimated_flops: int = 0
    estimated_peak_bytes: int = 0
    estimated_read_bytes: int = 0
    estimated_write_bytes: int = 0
    estimated_copy_bytes: int = 0
    estimated_comm_bytes: int = 0
    required_workspace_bytes: int = 0
    sliced_modes: tuple[Hashable, ...] = ()
    distributed_modes: tuple[Hashable, ...] = ()
    plan_hash: str = ""

    def __post_init__(self):
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "input_specs", tuple(self.input_specs))
        object.__setattr__(self, "output_modes", tuple(self.output_modes))
        object.__setattr__(self, "sliced_modes", tuple(self.sliced_modes))
        object.__setattr__(self, "distributed_modes", tuple(self.distributed_modes))
        if not self.plan_hash:
            object.__setattr__(self, "plan_hash", _contraction_plan_hash(self))


@dataclass(frozen=True)
class DistributedContractionSpec:
    equation: str
    operands: tuple[Any, ...]
    output_modes: tuple[Hashable, ...] | None = None
    output_sharding: ShardingSpec | None = None
    optimize: str | Any | None = None

    def __post_init__(self):
        input_modes, parsed_output_modes = parse_einsum_equation(self.equation)
        if len(input_modes) != len(self.operands):
            raise ValueError(
                "einsum operand count mismatch: equation has {0} operands but {1} arrays were provided"
                .format(len(input_modes), len(self.operands))
            )
        output_modes = parsed_output_modes if self.output_modes is None else tuple(self.output_modes)
        if output_modes != parsed_output_modes:
            raise ValueError("DistributedContractionSpec output_modes must match equation output")
        object.__setattr__(self, "operands", tuple(self.operands))
        object.__setattr__(self, "output_modes", output_modes)


@dataclass(frozen=True)
class DistributionState:
    operand_index: int
    modes: tuple[Hashable, ...]
    sharding: ShardingSpec | None
    distributed_modes: tuple[Hashable, ...] = ()
    replicated_modes: tuple[Hashable, ...] = ()


@dataclass(frozen=True)
class CommunicationPlan:
    kind: str
    bytes: int
    modes: tuple[Hashable, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class DistributedStepPlan:
    local_step: ContractionStep
    input_states: tuple[DistributionState, ...]
    output_sharding: ShardingSpec | None
    communication: tuple[CommunicationPlan, ...] = ()
    estimated_compute_s: float = 0.0
    estimated_comm_s: float = 0.0
    estimated_total_s: float = 0.0
    kind: str = "distributed_contract"


@dataclass(frozen=True)
class DistributedContractionPlan:
    path: ContractionPlan
    steps: tuple[DistributedStepPlan, ...]
    output_sharding: ShardingSpec | None
    estimated_comm_bytes: int = 0
    equation: str | None = None
    peak_local_bytes: int = 0
    total_flops: int = 0
    total_comm_bytes: int = 0
    total_redistribute_bytes: int = 0
    total_allreduce_bytes: int = 0
    total_gather_bytes: int = 0


@dataclass(frozen=True)
class StreamEvent:
    device: DeviceSpec
    stream: Any = None
    token: Any = None


@dataclass
class Workspace:
    device: DeviceSpec
    nbytes: int
    buffer: Any


@dataclass(frozen=True)
class HardwareModel:
    flop_per_s: float | None = None
    memory_bandwidth_Bps: float | None = None
    h2d_bandwidth_Bps: float | None = None
    d2h_bandwidth_Bps: float | None = None
    p2p_bandwidth_Bps: float | None = None
    network_bandwidth_Bps: float | None = None
    latency_s: float = 0.0
    max_memory_bytes: int | None = None
    workspace_limit_bytes: int | None = None
    device_flop_s: float | None = None
    host_bandwidth_bytes_s: float | None = None
    device_bandwidth_bytes_s: float | None = None
    interconnect_bandwidth_bytes_s: float | None = None


@dataclass(frozen=True)
class CostEstimate:
    flops: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    copy_bytes: int = 0
    comm_bytes: int = 0
    workspace_bytes: int = 0
    peak_bytes: int = 0
    compute_s: float = 0.0
    memory_s: float = 0.0
    copy_s: float = 0.0
    comm_s: float = 0.0
    total_s: float = 0.0
    estimated_time_s: float | None = None

    def __post_init__(self):
        if self.estimated_time_s is None:
            object.__setattr__(self, "estimated_time_s", self.total_s)


def _contraction_plan_hash(plan: ContractionPlan) -> str:
    payload = (
        tuple(
            (
                step.kind,
                step.inputs,
                step.output,
                step.input_modes,
                step.output_modes,
                step.estimated_flops,
                step.estimated_peak_bytes,
                step.estimated_read_bytes,
                step.estimated_write_bytes,
                step.estimated_copy_bytes,
                step.estimated_comm_bytes,
            )
            for step in plan.steps
        ),
        tuple((operand.modes, _shape_of(operand.array), str(getattr(operand.array, "dtype", None))) for operand in plan.input_specs),
        plan.output_modes,
        plan.estimated_peak_bytes,
        plan.sliced_modes,
        plan.distributed_modes,
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class BlockKey:
    qn_left: tuple[int, ...]
    qn_right: tuple[int, ...]
    extra: tuple[Any, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "qn_left", tuple(self.qn_left))
        object.__setattr__(self, "qn_right", tuple(self.qn_right))
        object.__setattr__(self, "extra", tuple(self.extra))


@dataclass(frozen=True)
class GroupedGemmPlan:
    tasks: tuple[MatmulDesc, ...]
    output_blocks: tuple[Any, ...]
    bucketed_by_shape: dict[tuple[int, int, int], tuple[int, ...]]
    scatter_add_required: bool
    estimated_flops: int
    estimated_read_bytes: int
    estimated_write_bytes: int
    estimated_workspace_bytes: int
    output_modes: tuple[Hashable, ...] = ()
    global_shape: tuple[int, ...] = ()
    block_axis_meta: Any = None
    backend: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "output_blocks", tuple(self.output_blocks))
        object.__setattr__(self, "output_modes", tuple(self.output_modes))
        object.__setattr__(self, "global_shape", tuple(int(dim) for dim in self.global_shape))
        bucketed = {
            tuple(int(dim) for dim in shape): tuple(int(index) for index in indices)
            for shape, indices in self.bucketed_by_shape.items()
        }
        object.__setattr__(self, "bucketed_by_shape", bucketed)


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


def layout_from_modes_shape(modes, shape) -> LayoutSpec:
    shape = tuple(int(dim) for dim in shape)
    return LayoutSpec(
        logical_shape=shape,
        physical_shape=shape,
        logical_modes=tuple(modes),
        strides=None,
        order="C",
        contiguous_groups=(tuple(range(len(shape))),),
        estimated_copy_bytes=0,
    )


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
    output_layout = spec.output_layout or layout_from_modes_shape(spec.output_modes, output_shape)
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
        layout_c=output_layout,
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
