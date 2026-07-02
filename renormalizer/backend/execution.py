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
    VALID_KINDS = frozenset(("cpu", "cuda", "rocm", "mps", "tpu", "distributed"))

    kind: str
    index: int | None = None
    local_rank: int | None = None
    global_rank: int | None = None
    visible_id: str | None = None

    def __post_init__(self):
        kind = str(self.kind).lower()
        if kind == "gpu":
            kind = "cuda"
        if kind not in self.VALID_KINDS:
            raise ValueError("Unknown DeviceSpec kind {0!r}".format(self.kind))
        index = None if self.index is None else int(self.index)
        local_rank = None if self.local_rank is None else int(self.local_rank)
        global_rank = None if self.global_rank is None else int(self.global_rank)
        if index is not None and index < 0:
            raise ValueError("DeviceSpec index must be non-negative")
        if local_rank is not None and local_rank < 0:
            raise ValueError("DeviceSpec local_rank must be non-negative")
        if global_rank is not None and global_rank < 0:
            raise ValueError("DeviceSpec global_rank must be non-negative")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "local_rank", local_rank)
        object.__setattr__(self, "global_rank", global_rank)
        if self.visible_id is not None:
            object.__setattr__(self, "visible_id", str(self.visible_id))


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
    broadcast: bool = False
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

    def __post_init__(self):
        shape = tuple(int(dim) for dim in self.shape)
        itemsize = int(self.itemsize)
        ndim = int(self.ndim)
        size = int(self.size)
        nbytes = int(self.nbytes)
        if any(dim < 0 for dim in shape):
            raise ValueError("ArrayInfo shape dimensions must be non-negative")
        if itemsize < 0:
            raise ValueError("ArrayInfo itemsize must be non-negative")
        if ndim < 0:
            raise ValueError("ArrayInfo ndim must be non-negative")
        if size < 0:
            raise ValueError("ArrayInfo size must be non-negative")
        if nbytes < 0:
            raise ValueError("ArrayInfo nbytes must be non-negative")
        if ndim != len(shape):
            raise ValueError("ArrayInfo ndim must match shape rank")
        if size != _prod(shape):
            raise ValueError("ArrayInfo size must match shape product")
        if not self.is_distributed and nbytes != size * itemsize:
            raise ValueError("ArrayInfo nbytes must match size and itemsize")
        strides = None if self.strides is None else tuple(int(stride) for stride in self.strides)
        if strides is not None and len(strides) != len(shape):
            raise ValueError("ArrayInfo strides must match shape rank")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "itemsize", itemsize)
        object.__setattr__(self, "ndim", ndim)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "nbytes", nbytes)
        object.__setattr__(self, "strides", strides)
        object.__setattr__(self, "order", str(self.order))
        object.__setattr__(self, "contiguous", bool(self.contiguous))
        object.__setattr__(self, "writeable", bool(self.writeable))
        object.__setattr__(self, "is_host", bool(self.is_host))
        object.__setattr__(self, "is_device", bool(self.is_device))
        object.__setattr__(self, "is_distributed", bool(self.is_distributed))
        if self.owns_data is not None:
            object.__setattr__(self, "owns_data", bool(self.owns_data))
        object.__setattr__(self, "backend_name", str(self.backend_name))


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

    def __post_init__(self):
        logical_shape = tuple(int(dim) for dim in self.logical_shape)
        physical_shape = None if self.physical_shape is None else tuple(int(dim) for dim in self.physical_shape)
        physical_rank = len(logical_shape) if physical_shape is None else len(physical_shape)
        logical_modes = tuple(self.logical_modes)
        estimated_copy_bytes = int(self.estimated_copy_bytes)
        if any(dim < 0 for dim in logical_shape):
            raise ValueError("LayoutSpec logical_shape dimensions must be non-negative")
        if physical_shape is not None and any(dim < 0 for dim in physical_shape):
            raise ValueError("LayoutSpec physical_shape dimensions must be non-negative")
        if len(logical_modes) != len(logical_shape):
            raise ValueError("LayoutSpec logical_modes must match logical_shape rank")
        strides = None if self.strides is None else tuple(int(stride) for stride in self.strides)
        if strides is not None and len(strides) != physical_rank:
            raise ValueError("LayoutSpec strides must match physical shape rank")
        contiguous_groups = tuple(tuple(int(index) for index in group) for group in self.contiguous_groups)
        if any(index < 0 or index >= len(logical_shape) for group in contiguous_groups for index in group):
            raise ValueError("LayoutSpec contiguous_groups indices out of range")
        transpose_perm = None if self.transpose_perm is None else tuple(int(index) for index in self.transpose_perm)
        if transpose_perm is not None and sorted(transpose_perm) != list(range(len(logical_shape))):
            raise ValueError("LayoutSpec transpose_perm must be a permutation")
        if estimated_copy_bytes < 0:
            raise ValueError("LayoutSpec estimated_copy_bytes must be non-negative")
        object.__setattr__(self, "logical_shape", logical_shape)
        object.__setattr__(self, "physical_shape", physical_shape)
        object.__setattr__(self, "logical_modes", logical_modes)
        object.__setattr__(self, "strides", strides)
        object.__setattr__(self, "order", str(self.order))
        object.__setattr__(self, "contiguous_groups", contiguous_groups)
        object.__setattr__(self, "requires_transpose", bool(self.requires_transpose))
        object.__setattr__(self, "transpose_perm", transpose_perm)
        object.__setattr__(self, "estimated_copy_bytes", estimated_copy_bytes)


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
            if set(local_slices) != set(range(self.mesh.world_size)):
                raise ValueError("ShardingSpec local_slices ranks must match mesh ranks")
            for slices in local_slices.values():
                if len(slices) != len(global_shape):
                    raise ValueError("ShardingSpec local_slices entries must match global_shape rank")
                if any(not isinstance(item, slice) for item in slices):
                    raise ValueError("ShardingSpec local_slices entries must be slice objects")
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
        global_shape = tuple(int(dim) for dim in self.global_shape)
        modes = tuple(self.modes)
        if any(dim < 0 for dim in global_shape):
            raise ValueError("DistributedTensor global_shape dimensions must be non-negative")
        if len(modes) != len(global_shape):
            raise ValueError("DistributedTensor modes must match global_shape rank")
        if tuple(self.sharding.global_shape) != global_shape:
            raise ValueError("DistributedTensor sharding global_shape must match global_shape")
        if tuple(self.sharding.modes) != modes:
            raise ValueError("DistributedTensor sharding modes must match modes")
        if self.mesh != self.sharding.mesh:
            raise ValueError("DistributedTensor mesh must match sharding mesh")
        self.global_shape = global_shape
        self.modes = modes
        if self.dtype is None:
            self.dtype = getattr(self.local_array, "dtype", None)
        if not self.local_shape:
            local_shape = _shape_of(self.local_array)
        else:
            local_shape = tuple(int(dim) for dim in self.local_shape)
        if any(dim < 0 for dim in local_shape):
            raise ValueError("DistributedTensor local_shape dimensions must be non-negative")
        if len(local_shape) != len(global_shape):
            raise ValueError("DistributedTensor local_shape must match global_shape rank")
        self.local_shape = local_shape
        if not self.local_nbytes:
            local_nbytes = _nbytes_of(self.local_array)
        else:
            local_nbytes = int(self.local_nbytes)
        if local_nbytes < 0:
            raise ValueError("DistributedTensor local_nbytes must be non-negative")
        self.local_nbytes = local_nbytes
        if self.rank_local_arrays is not None:
            rank_local_arrays = {int(rank): array for rank, array in self.rank_local_arrays.items()}
            unknown_ranks = set(rank_local_arrays) - set(self.sharding.local_slices)
            if unknown_ranks:
                raise ValueError(
                    "DistributedTensor rank_local_arrays ranks must be present in sharding local_slices"
                )
            self.rank_local_arrays = rank_local_arrays

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


def _array_strides(array):
    strides = getattr(array, "strides", None)
    if callable(strides):
        strides = strides()
    if strides is not None:
        return tuple(int(stride) for stride in strides)

    stride = getattr(array, "stride", None)
    if callable(stride):
        try:
            strides = stride()
        except TypeError:
            strides = None
        if strides is not None:
            itemsize = _itemsize_of(array)
            return tuple(int(stride) * itemsize for stride in strides)
    return None


def _c_contiguous_from_strides(shape, strides, itemsize):
    if strides is None:
        return False
    expected = int(itemsize)
    for dim, stride in reversed(tuple(zip(shape, strides))):
        if int(dim) <= 1:
            continue
        if int(stride) != expected:
            return False
        expected *= int(dim)
    return True


def _f_contiguous_from_strides(shape, strides, itemsize):
    if strides is None:
        return False
    expected = int(itemsize)
    for dim, stride in tuple(zip(shape, strides)):
        if int(dim) <= 1:
            continue
        if int(stride) != expected:
            return False
        expected *= int(dim)
    return True


def _contiguous_order(array, shape, strides, flags):
    c_contiguous = bool(flags.get("C_CONTIGUOUS", False))
    f_contiguous = bool(flags.get("F_CONTIGUOUS", False))
    itemsize = _itemsize_of(array)
    if not c_contiguous:
        c_contiguous = _c_contiguous_from_strides(shape, strides, itemsize)
    if not f_contiguous:
        f_contiguous = _f_contiguous_from_strides(shape, strides, itemsize)

    is_contiguous = getattr(array, "is_contiguous", None)
    if callable(is_contiguous):
        try:
            c_contiguous = bool(is_contiguous()) or c_contiguous
        except TypeError:
            pass

    if c_contiguous:
        order = "C"
    elif f_contiguous:
        order = "F"
    else:
        order = "unknown"
    return order, bool(c_contiguous or f_contiguous)


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
    strides = _array_strides(array)
    order, contiguous = _contiguous_order(array, shape, strides, flags)
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
        contiguous=contiguous,
        writeable=bool(flags.get("WRITEABLE", True)),
        owns_data=flags.get("OWNDATA"),
        backend_name=backend.name,
    )


def layout_from_array(array, modes=None) -> LayoutSpec:
    shape = _shape_of(array)
    strides = _array_strides(array)
    flags = _array_flags(array)
    order, _ = _contiguous_order(array, shape, strides, flags)
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
        modes = tuple(self.modes)
        shape = _shape_of(self.array)
        if len(modes) != len(shape):
            raise ValueError("TensorOperand modes must match array rank")
        if self.layout is not None:
            if tuple(self.layout.logical_modes) != modes:
                raise ValueError("TensorOperand layout modes must match operand modes")
            if tuple(self.layout.logical_shape) != shape:
                raise ValueError("TensorOperand layout shape must match array shape")
        object.__setattr__(self, "modes", modes)
        if self.layout is None:
            object.__setattr__(self, "layout", layout_from_array(self.array, modes))


@dataclass(frozen=True)
class EinsumSpec:
    operands: tuple[TensorOperand, ...]
    output_modes: tuple[Hashable, ...]
    constants: tuple[int, ...] = ()
    optimize: str | Any | None = None

    def __post_init__(self):
        operands = tuple(self.operands)
        output_modes = tuple(self.output_modes)
        constants = tuple(int(index) for index in self.constants)
        if not operands:
            raise ValueError("EinsumSpec operands must be non-empty")
        if len(set(output_modes)) != len(output_modes):
            raise ValueError("EinsumSpec output_modes must be unique")
        input_modes = set()
        for operand in operands:
            input_modes.update(operand.modes)
        missing_modes = [mode for mode in output_modes if mode not in input_modes]
        if missing_modes:
            raise ValueError(
                "EinsumSpec output_modes are not present in operands: {0}"
                .format(missing_modes)
            )
        for index in constants:
            if index < 0 or index >= len(operands):
                raise ValueError("EinsumSpec constant operand index {0} is out of range".format(index))
        object.__setattr__(self, "operands", operands)
        object.__setattr__(self, "output_modes", output_modes)
        object.__setattr__(self, "constants", constants)


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
        batch_axis = int(self.batch_axis)
        if any(dim < 0 for dim in center_shape):
            raise ValueError("center_shape dimensions must be non-negative")
        if packed_dim < 0:
            raise ValueError("packed_dim must be non-negative")
        if nrhs < 1:
            raise ValueError("nrhs must be positive")
        batch_ndim = len(center_shape) + 1
        normalized_batch_axis = batch_axis + batch_ndim if batch_axis < 0 else batch_axis
        if normalized_batch_axis < 0 or normalized_batch_axis >= batch_ndim:
            raise ValueError(
                "PackedVectorSpec batch_axis {0} is out of bounds for ndim {1}"
                .format(batch_axis, batch_ndim)
            )
        mask_shape = getattr(self.qn_mask, "shape", None)
        if mask_shape is not None and tuple(int(dim) for dim in mask_shape) != center_shape:
            raise ValueError(
                "PackedVectorSpec qn_mask shape must match center_shape {0}; got {1}"
                .format(center_shape, tuple(int(dim) for dim in mask_shape))
            )
        mask_sum = getattr(self.qn_mask, "sum", None)
        if callable(mask_sum):
            true_count = int(mask_sum())
            if true_count != packed_dim:
                raise ValueError(
                    "PackedVectorSpec packed_dim must match qn_mask true count {0}; got {1}"
                    .format(true_count, packed_dim)
                )
        object.__setattr__(self, "center_shape", center_shape)
        object.__setattr__(self, "batch_axis", batch_axis)
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

    def __post_init__(self):
        output_modes = tuple(self.output_modes)
        left_batch_modes = tuple(self.left_batch_modes)
        right_batch_modes = tuple(self.right_batch_modes)
        contracted_modes = tuple(self.contracted_modes)
        left_only_modes = tuple(self.left_only_modes)
        right_only_modes = tuple(self.right_only_modes)

        if len(set(output_modes)) != len(output_modes):
            raise ValueError("PairContractionSpec output_modes must be unique")

        left_modes = set(self.left.modes)
        right_modes = set(self.right.modes)
        operand_modes = left_modes | right_modes
        missing_output = [mode for mode in output_modes if mode not in operand_modes]
        if missing_output:
            raise ValueError(
                "PairContractionSpec output_modes are not present in operands: {0}"
                .format(missing_output)
            )
        if left_batch_modes != right_batch_modes:
            raise ValueError("PairContractionSpec batch modes must match")
        missing_left_batch = [mode for mode in left_batch_modes if mode not in left_modes]
        if missing_left_batch:
            raise ValueError(
                "PairContractionSpec left_batch_modes are not present in left operand: {0}"
                .format(missing_left_batch)
            )
        missing_right_batch = [mode for mode in right_batch_modes if mode not in right_modes]
        if missing_right_batch:
            raise ValueError(
                "PairContractionSpec right_batch_modes are not present in right operand: {0}"
                .format(missing_right_batch)
            )
        missing_contracted = [
            mode for mode in contracted_modes if mode not in left_modes or mode not in right_modes
        ]
        if missing_contracted:
            raise ValueError(
                "PairContractionSpec contracted_modes are not present in both operands: {0}"
                .format(missing_contracted)
            )
        missing_left_only = [mode for mode in left_only_modes if mode not in left_modes]
        if missing_left_only:
            raise ValueError(
                "PairContractionSpec left_only_modes are not present in left operand: {0}"
                .format(missing_left_only)
            )
        missing_right_only = [mode for mode in right_only_modes if mode not in right_modes]
        if missing_right_only:
            raise ValueError(
                "PairContractionSpec right_only_modes are not present in right operand: {0}"
                .format(missing_right_only)
            )
        if any(mode in output_modes for mode in contracted_modes):
            raise ValueError("PairContractionSpec contracted_modes must not appear in output_modes")

        output_components = set(left_batch_modes) | set(left_only_modes) | set(right_only_modes)
        missing_components = [mode for mode in output_modes if mode not in output_components]
        if missing_components:
            raise ValueError(
                "PairContractionSpec output_modes are missing from contraction mode groups: {0}"
                .format(missing_components)
            )
        extra_components = [mode for mode in output_components if mode not in output_modes]
        if extra_components:
            raise ValueError(
                "PairContractionSpec contraction mode groups contain non-output modes: {0}"
                .format(extra_components)
            )
        if self.output_layout is not None:
            if len(self.output_layout.logical_shape) != len(output_modes):
                raise ValueError("PairContractionSpec output_layout rank must match output_modes")
            if tuple(self.output_layout.logical_modes) != output_modes:
                raise ValueError("PairContractionSpec output_layout modes must match output_modes")

        object.__setattr__(self, "output_modes", output_modes)
        object.__setattr__(self, "left_batch_modes", left_batch_modes)
        object.__setattr__(self, "right_batch_modes", right_batch_modes)
        object.__setattr__(self, "contracted_modes", contracted_modes)
        object.__setattr__(self, "left_only_modes", left_only_modes)
        object.__setattr__(self, "right_only_modes", right_only_modes)

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

    def __post_init__(self):
        for field in ("m", "n", "k"):
            value = int(getattr(self, field))
            if value < 0:
                raise ValueError("MatmulDesc {0} must be non-negative".format(field))
            object.__setattr__(self, field, value)
        batch_shape = tuple(int(dim) for dim in self.batch_shape)
        if any(dim < 0 for dim in batch_shape):
            raise ValueError("MatmulDesc batch_shape dimensions must be non-negative")
        object.__setattr__(self, "batch_shape", batch_shape)
        for field in (
            "estimated_flops",
            "estimated_read_bytes",
            "estimated_write_bytes",
            "estimated_workspace_bytes",
        ):
            value = int(getattr(self, field))
            if value < 0:
                raise ValueError("MatmulDesc {0} must be non-negative".format(field))
            object.__setattr__(self, field, value)


@dataclass(frozen=True)
class LayoutTransform:
    kind: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    copy_bytes: int = 0
    reason: str | None = None

    def __post_init__(self):
        input_shape = tuple(int(dim) for dim in self.input_shape)
        output_shape = tuple(int(dim) for dim in self.output_shape)
        copy_bytes = int(self.copy_bytes)
        if any(dim < 0 for dim in input_shape):
            raise ValueError("LayoutTransform input_shape dimensions must be non-negative")
        if any(dim < 0 for dim in output_shape):
            raise ValueError("LayoutTransform output_shape dimensions must be non-negative")
        if copy_bytes < 0:
            raise ValueError("LayoutTransform copy_bytes must be non-negative")
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "input_shape", input_shape)
        object.__setattr__(self, "output_shape", output_shape)
        object.__setattr__(self, "copy_bytes", copy_bytes)


@dataclass(frozen=True)
class MatmulPlan:
    VALID_KINDS = frozenset((
        "gemm",
        "strided_batched_gemm",
        "batched_gemm",
        "grouped_gemm",
        "fallback_tensordot",
        "fallback_einsum",
    ))

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
    plan_hash: str = ""

    def __post_init__(self):
        kind = str(self.kind)
        if kind not in self.VALID_KINDS:
            raise ValueError("Unknown MatmulPlan kind {0!r}".format(self.kind))
        descs = tuple(self.descs)
        if not descs:
            raise ValueError("MatmulPlan descs must be non-empty")
        output_shape = tuple(int(dim) for dim in self.output_shape)
        if any(dim < 0 for dim in output_shape):
            raise ValueError("MatmulPlan output_shape dimensions must be non-negative")
        for field in ("copy_bytes", "workspace_bytes", "estimated_flops"):
            value = int(getattr(self, field))
            if value < 0:
                raise ValueError("MatmulPlan {0} must be non-negative".format(field))
            object.__setattr__(self, field, value)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "descs", descs)
        object.__setattr__(self, "pre_ops", tuple(self.pre_ops))
        object.__setattr__(self, "post_ops", tuple(self.post_ops))
        object.__setattr__(self, "output_shape", output_shape)
        if not self.plan_hash:
            object.__setattr__(self, "plan_hash", _matmul_plan_hash(self))


@dataclass(frozen=True)
class ContractionStep:
    VALID_KINDS = frozenset((
        "gemm",
        "strided_batched_gemm",
        "batched_gemm",
        "grouped_gemm",
        "fallback_tensordot",
        "fallback_einsum",
        "tensordot",
        "einsum",
        "transpose",
        "reshape",
        "slice",
        "gather",
        "redistribute",
        "activate_distribution",
        "distributed_contract",
    ))

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
        kind = str(self.kind)
        if kind not in self.VALID_KINDS:
            raise ValueError("Unknown ContractionStep kind {0!r}".format(self.kind))
        inputs = tuple(int(index) for index in self.inputs)
        output = int(self.output)
        if any(index < 0 for index in inputs):
            raise ValueError("ContractionStep inputs must be non-negative")
        if output < 0:
            raise ValueError("ContractionStep output must be non-negative")
        input_modes = tuple(tuple(modes) for modes in self.input_modes)
        output_modes = tuple(self.output_modes)
        if len(input_modes) != len(inputs):
            raise ValueError("ContractionStep input_modes must match inputs length")
        if len(set(output_modes)) != len(output_modes):
            raise ValueError("ContractionStep output_modes must be unique")
        for field in (
            "estimated_flops",
            "estimated_read_bytes",
            "estimated_write_bytes",
            "estimated_copy_bytes",
            "estimated_peak_bytes",
            "estimated_comm_bytes",
            "required_workspace_bytes",
        ):
            value = int(getattr(self, field))
            if value < 0:
                raise ValueError("ContractionStep {0} must be non-negative".format(field))
            object.__setattr__(self, field, value)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "input_modes", input_modes)
        object.__setattr__(self, "output_modes", output_modes)


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
        steps = tuple(self.steps)
        if not steps:
            raise ValueError("ContractionPlan steps must be non-empty")
        input_specs = tuple(self.input_specs)
        output_modes = tuple(self.output_modes)
        if not input_specs:
            raise ValueError("ContractionPlan input_specs must be non-empty")
        if steps[-1].output_modes != output_modes:
            raise ValueError("ContractionPlan output_modes must match final step output_modes")
        for field in (
            "estimated_flops",
            "estimated_peak_bytes",
            "estimated_read_bytes",
            "estimated_write_bytes",
            "estimated_copy_bytes",
            "estimated_comm_bytes",
            "required_workspace_bytes",
        ):
            value = int(getattr(self, field))
            if value < 0:
                raise ValueError("ContractionPlan {0} must be non-negative".format(field))
            object.__setattr__(self, field, value)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "input_specs", input_specs)
        object.__setattr__(self, "output_modes", output_modes)
        object.__setattr__(self, "sliced_modes", tuple(self.sliced_modes))
        object.__setattr__(self, "distributed_modes", tuple(self.distributed_modes))
        if not self.plan_hash:
            object.__setattr__(self, "plan_hash", _contraction_plan_hash(self))


@dataclass(frozen=True)
class SlicedContractionPlan:
    base_plan: ContractionPlan
    sliced_mode: Hashable
    output_axis: int
    output_slices: tuple[tuple[slice, ...], ...]
    operand_slices: tuple[tuple[tuple[slice, ...], ...], ...]

    def __post_init__(self):
        output_axis = int(self.output_axis)
        output_rank = len(self.base_plan.output_modes)
        if output_axis < 0:
            output_axis += output_rank
        if output_axis < 0 or output_axis >= output_rank:
            raise ValueError("SlicedContractionPlan output_axis is out of bounds")
        output_slices = tuple(tuple(item) for item in self.output_slices)
        if not output_slices:
            raise ValueError("SlicedContractionPlan output_slices must be non-empty")
        operand_slices = tuple(tuple(tuple(slices) for slices in item) for item in self.operand_slices)
        if len(operand_slices) != len(output_slices):
            raise ValueError("SlicedContractionPlan operand_slices must match output_slices length")
        if any(len(group) != len(self.base_plan.input_specs) for group in operand_slices):
            raise ValueError("SlicedContractionPlan operand slice groups must match input_specs length")
        object.__setattr__(self, "output_axis", output_axis)
        object.__setattr__(self, "output_slices", output_slices)
        object.__setattr__(
            self,
            "operand_slices",
            operand_slices,
        )


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
    tensor_id: int | None = None
    shape: tuple[int, ...] = ()
    distributed_modes: tuple[Hashable, ...] = ()
    replicated_modes: tuple[Hashable, ...] = ()
    local_shape: tuple[int, ...] = ()
    local_nbytes: int = 0

    def __post_init__(self):
        operand_index = int(self.operand_index)
        tensor_id = int(operand_index) if self.tensor_id is None else int(self.tensor_id)
        modes = tuple(self.modes)
        shape = tuple(int(dim) for dim in self.shape)
        distributed_modes = tuple(self.distributed_modes)
        replicated_modes = tuple(self.replicated_modes)
        local_shape = tuple(int(dim) for dim in self.local_shape)
        local_nbytes = int(self.local_nbytes)

        if operand_index < 0:
            raise ValueError("DistributionState operand_index must be non-negative")
        if tensor_id < 0:
            raise ValueError("DistributionState tensor_id must be non-negative")
        if any(dim < 0 for dim in shape):
            raise ValueError("DistributionState shape dimensions must be non-negative")
        if len(modes) != len(shape):
            raise ValueError("DistributionState modes must match shape rank")
        if any(dim < 0 for dim in local_shape):
            raise ValueError("DistributionState local_shape dimensions must be non-negative")
        if len(local_shape) != len(shape):
            raise ValueError("DistributionState local_shape must match shape rank")
        if local_nbytes < 0:
            raise ValueError("DistributionState local_nbytes must be non-negative")

        mode_set = set(modes)
        if set(distributed_modes) - mode_set:
            raise ValueError("DistributionState distributed_modes contain modes not present in modes")
        if set(replicated_modes) - mode_set:
            raise ValueError("DistributionState replicated_modes contain modes not present in modes")
        if set(distributed_modes) & set(replicated_modes):
            raise ValueError("DistributionState distributed_modes and replicated_modes must not overlap")
        if set(distributed_modes) | set(replicated_modes) != mode_set:
            raise ValueError("DistributionState distributed_modes and replicated_modes must cover all modes")

        if self.sharding is not None:
            if tuple(self.sharding.global_shape) != shape:
                raise ValueError("DistributionState sharding global_shape must match shape")
            if tuple(self.sharding.modes) != modes:
                raise ValueError("DistributionState sharding modes must match modes")
            expected_distributed = tuple(mode for mode in modes if mode in self.sharding.sharded_modes)
            expected_replicated = tuple(mode for mode in modes if mode not in self.sharding.sharded_modes)
            if distributed_modes != expected_distributed:
                raise ValueError("DistributionState distributed_modes must match sharding sharded_modes")
            if replicated_modes != expected_replicated:
                raise ValueError("DistributionState replicated_modes must match sharding replicated_modes")
        elif distributed_modes:
            raise ValueError("DistributionState without sharding cannot have distributed_modes")

        object.__setattr__(self, "operand_index", operand_index)
        object.__setattr__(self, "tensor_id", tensor_id)
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "distributed_modes", distributed_modes)
        object.__setattr__(self, "replicated_modes", replicated_modes)
        object.__setattr__(self, "local_shape", local_shape)
        object.__setattr__(self, "local_nbytes", local_nbytes)


@dataclass(frozen=True)
class CommunicationPlan:
    VALID_KINDS = frozenset((
        "activate_distribution",
        "redistribute",
        "broadcast",
        "allreduce",
        "reduce_scatter",
        "gather",
        "allgather",
        "alltoall",
    ))

    kind: str
    bytes: int
    local_bytes: int | None = None
    modes: tuple[Hashable, ...] = ()
    reason: str | None = None
    num_messages: int = 1
    block_size: int | None = None

    def __post_init__(self):
        kind = str(self.kind)
        if kind not in self.VALID_KINDS:
            raise ValueError("Unknown CommunicationPlan kind {0!r}".format(self.kind))
        bytes_ = int(self.bytes)
        local_bytes = bytes_ if self.local_bytes is None else int(self.local_bytes)
        num_messages = int(self.num_messages)
        if bytes_ < 0:
            raise ValueError("CommunicationPlan bytes must be non-negative")
        if local_bytes < 0:
            raise ValueError("CommunicationPlan local_bytes must be non-negative")
        if num_messages < 1:
            raise ValueError("CommunicationPlan num_messages must be positive")
        block_size = local_bytes if self.block_size is None else int(self.block_size)
        if block_size < 0:
            raise ValueError("CommunicationPlan block_size must be non-negative")
        object.__setattr__(self, "bytes", bytes_)
        object.__setattr__(self, "local_bytes", local_bytes)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "modes", tuple(self.modes))
        object.__setattr__(self, "num_messages", num_messages)
        object.__setattr__(self, "block_size", block_size)


@dataclass(frozen=True)
class DistributedStepPlan:
    VALID_KINDS = frozenset((
        "activate_distribution",
        "keep_distribution",
        "redistribute",
        "distributed_contract",
        "gather",
        "replicate",
    ))

    local_step: ContractionStep
    input_states: tuple[DistributionState, ...]
    output_sharding: ShardingSpec | None
    output_state: DistributionState | None = None
    communication: tuple[CommunicationPlan, ...] = ()
    estimated_compute_s: float = 0.0
    estimated_comm_s: float = 0.0
    estimated_total_s: float = 0.0
    kind: str = "distributed_contract"

    def __post_init__(self):
        kind = str(self.kind)
        if kind not in self.VALID_KINDS:
            raise ValueError("Unknown DistributedStepPlan kind {0!r}".format(self.kind))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "input_states", tuple(self.input_states))
        object.__setattr__(self, "communication", tuple(self.communication))
        for field in ("estimated_compute_s", "estimated_comm_s", "estimated_total_s"):
            value = float(getattr(self, field))
            if value < 0:
                raise ValueError("DistributedStepPlan {0} must be non-negative".format(field))
            object.__setattr__(self, field, value)

    @property
    def local_contraction_plan(self):
        return self.local_step.plan

    @property
    def communication_plan(self):
        if not self.communication:
            return None
        return self.communication[0]


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

    def __post_init__(self):
        steps = tuple(self.steps)
        if not steps:
            raise ValueError("DistributedContractionPlan steps must be non-empty")
        object.__setattr__(self, "steps", steps)
        if self.equation is not None:
            object.__setattr__(self, "equation", str(self.equation))
        for field in (
            "estimated_comm_bytes",
            "peak_local_bytes",
            "total_flops",
            "total_comm_bytes",
            "total_redistribute_bytes",
            "total_allreduce_bytes",
            "total_gather_bytes",
        ):
            value = int(getattr(self, field))
            if value < 0:
                raise ValueError("DistributedContractionPlan {0} must be non-negative".format(field))
            object.__setattr__(self, field, value)


@dataclass(frozen=True)
class StreamEvent:
    device: DeviceSpec
    stream: Any = None
    token: Any = None

    def __post_init__(self):
        device = parse_device_spec(self.device)
        if device is None:
            raise ValueError("StreamEvent device must be explicit")
        object.__setattr__(self, "device", device)


@dataclass
class Workspace:
    device: DeviceSpec
    nbytes: int
    buffer: Any

    def __post_init__(self):
        device = parse_device_spec(self.device)
        if device is None:
            raise ValueError("Workspace device must be explicit")
        nbytes = int(self.nbytes)
        if nbytes < 0:
            raise ValueError("Workspace nbytes must be non-negative")
        self.device = device
        self.nbytes = nbytes


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

    def __post_init__(self):
        for field in (
            "flop_per_s",
            "memory_bandwidth_Bps",
            "h2d_bandwidth_Bps",
            "d2h_bandwidth_Bps",
            "p2p_bandwidth_Bps",
            "network_bandwidth_Bps",
            "latency_s",
            "max_memory_bytes",
            "workspace_limit_bytes",
            "device_flop_s",
            "host_bandwidth_bytes_s",
            "device_bandwidth_bytes_s",
            "interconnect_bandwidth_bytes_s",
        ):
            value = getattr(self, field)
            if value is not None and value < 0:
                raise ValueError("{0} must be non-negative".format(field))


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
        for field in (
            "flops",
            "read_bytes",
            "write_bytes",
            "copy_bytes",
            "comm_bytes",
            "workspace_bytes",
            "peak_bytes",
        ):
            value = int(getattr(self, field))
            if value < 0:
                raise ValueError("{0} must be non-negative".format(field))
            object.__setattr__(self, field, value)

        for field in ("compute_s", "memory_s", "copy_s", "comm_s", "total_s"):
            value = float(getattr(self, field))
            if value < 0:
                raise ValueError("{0} must be non-negative".format(field))
            object.__setattr__(self, field, value)

        if self.estimated_time_s is None:
            estimated_time_s = self.total_s
        else:
            estimated_time_s = float(self.estimated_time_s)
        if estimated_time_s < 0:
            raise ValueError("estimated_time_s must be non-negative")
        object.__setattr__(self, "estimated_time_s", estimated_time_s)


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
                step.required_workspace_bytes,
                step.reason,
                step.fallback_reason,
                _nested_step_plan_hash_payload(step.plan),
            )
            for step in plan.steps
        ),
        tuple(
            (
                operand.modes,
                _shape_of(operand.array),
                str(getattr(operand.array, "dtype", None)),
                _layout_hash_payload(operand.layout),
            )
            for operand in plan.input_specs
        ),
        plan.output_modes,
        plan.estimated_peak_bytes,
        plan.sliced_modes,
        plan.distributed_modes,
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:16]


def _nested_step_plan_hash_payload(plan):
    if plan is None:
        return None
    plan_hash = getattr(plan, "plan_hash", None)
    if plan_hash:
        return (type(plan).__name__, str(plan_hash))
    return (type(plan).__name__, repr(plan))


def _layout_hash_payload(layout):
    if layout is None:
        return None
    return (
        layout.logical_shape,
        layout.physical_shape,
        layout.logical_modes,
        layout.strides,
        layout.order,
        layout.contiguous_groups,
        layout.requires_transpose,
        layout.transpose_perm,
        layout.estimated_copy_bytes,
    )


def _layout_transform_hash_payload(transform):
    return (
        transform.kind,
        transform.input_shape,
        transform.output_shape,
        transform.copy_bytes,
        transform.reason,
    )


def _matmul_desc_hash_payload(desc):
    return (
        desc.m,
        desc.n,
        desc.k,
        desc.batch_shape,
        desc.trans_a,
        desc.trans_b,
        desc.conj_a,
        desc.conj_b,
        desc.alpha,
        desc.beta,
        str(desc.dtype_compute),
        str(desc.dtype_output),
        _layout_hash_payload(desc.layout_a),
        _layout_hash_payload(desc.layout_b),
        _layout_hash_payload(desc.layout_c),
        str(getattr(desc.A, "dtype", None)),
        str(getattr(desc.B, "dtype", None)),
        _shape_of(desc.A),
        _shape_of(desc.B),
        _shape_of(desc.C) if desc.C is not None else None,
        str(getattr(desc.C, "dtype", None)) if desc.C is not None else None,
        desc.estimated_flops,
        desc.estimated_read_bytes,
        desc.estimated_write_bytes,
        desc.estimated_workspace_bytes,
    )


def _matmul_plan_hash(plan: MatmulPlan) -> str:
    payload = (
        plan.kind,
        tuple(_matmul_desc_hash_payload(desc) for desc in plan.descs),
        tuple(_layout_transform_hash_payload(transform) for transform in plan.pre_ops),
        tuple(_layout_transform_hash_payload(transform) for transform in plan.post_ops),
        plan.output_shape,
        plan.copy_bytes,
        plan.workspace_bytes,
        plan.estimated_flops,
        plan.estimated_time_s,
        plan.reason,
        plan.fallback_reason,
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:16]


def _block_key_hash_payload(key):
    qn_left = getattr(key, "qn_left", None)
    qn_right = getattr(key, "qn_right", None)
    extra = getattr(key, "extra", None)
    if qn_left is not None and qn_right is not None:
        return (tuple(qn_left), tuple(qn_right), tuple(extra or ()))
    return repr(key)


def _grouped_gemm_plan_hash(plan) -> str:
    payload = (
        tuple(_matmul_desc_hash_payload(desc) for desc in plan.tasks),
        tuple(_block_key_hash_payload(key) for key in plan.output_blocks),
        tuple(sorted(plan.bucketed_by_shape.items())),
        plan.scatter_add_required,
        plan.estimated_flops,
        plan.estimated_read_bytes,
        plan.estimated_write_bytes,
        plan.estimated_workspace_bytes,
        plan.output_modes,
        plan.global_shape,
        plan.backend,
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
    plan_hash: str = ""

    def __post_init__(self):
        tasks = tuple(self.tasks)
        output_blocks = tuple(self.output_blocks)
        output_modes = tuple(self.output_modes)
        global_shape = tuple(int(dim) for dim in self.global_shape)
        if len(tasks) != len(output_blocks):
            raise ValueError("GroupedGemmPlan tasks and output_blocks must have the same length")
        if any(dim < 0 for dim in global_shape):
            raise ValueError("GroupedGemmPlan global_shape dimensions must be non-negative")
        if global_shape and len(output_modes) != len(global_shape):
            raise ValueError("GroupedGemmPlan output_modes must match global_shape rank")
        bucketed = {
            tuple(int(dim) for dim in shape): tuple(int(index) for index in indices)
            for shape, indices in self.bucketed_by_shape.items()
        }
        for shape, indices in bucketed.items():
            if len(shape) != 3:
                raise ValueError("GroupedGemmPlan bucket shapes must be rank-3")
            if any(dim < 0 for dim in shape):
                raise ValueError("GroupedGemmPlan bucket shapes must be non-negative")
            if any(index < 0 or index >= len(tasks) for index in indices):
                raise ValueError("GroupedGemmPlan bucket task indices out of range")
            for index in indices:
                task_shape = (tasks[index].m, tasks[index].n, tasks[index].k)
                if shape != task_shape:
                    raise ValueError("GroupedGemmPlan bucket shape must match task descriptor")
        bucket_indices = [
            index
            for indices in bucketed.values()
            for index in indices
        ]
        if sorted(bucket_indices) != list(range(len(tasks))):
            raise ValueError("GroupedGemmPlan bucketed_by_shape must cover each task exactly once")
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "output_blocks", output_blocks)
        object.__setattr__(self, "output_modes", output_modes)
        object.__setattr__(self, "global_shape", global_shape)
        object.__setattr__(self, "bucketed_by_shape", bucketed)
        object.__setattr__(self, "scatter_add_required", bool(self.scatter_add_required))
        if self.backend is not None:
            object.__setattr__(self, "backend", str(self.backend))
        for field in (
            "estimated_flops",
            "estimated_read_bytes",
            "estimated_write_bytes",
            "estimated_workspace_bytes",
        ):
            value = int(getattr(self, field))
            if value < 0:
                raise ValueError("GroupedGemmPlan {0} must be non-negative".format(field))
            object.__setattr__(self, field, value)
        if not self.plan_hash:
            object.__setattr__(self, "plan_hash", _grouped_gemm_plan_hash(self))


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

    def __post_init__(self):
        modes = tuple(self.modes)
        shape = tuple(int(dim) for dim in self.shape)
        if any(dim < 0 for dim in shape):
            raise ValueError("DenseBlock shape dimensions must be non-negative")
        if len(modes) != len(shape):
            raise ValueError("DenseBlock modes must match shape rank")
        array_shape = _shape_of(self.array)
        if array_shape and array_shape != shape:
            raise ValueError("DenseBlock shape must match array shape")
        if self.offset is None:
            offset = None
        else:
            offset = tuple(int(item) for item in self.offset)
            if len(offset) != len(shape):
                raise ValueError("DenseBlock offset must match shape rank")
            if any(item < 0 for item in offset):
                raise ValueError("DenseBlock offset entries must be non-negative")
        self.modes = modes
        self.shape = shape
        self.offset = offset


@dataclass
class BlockTensor:
    blocks: dict[Any, DenseBlock]
    global_shape: tuple[int, ...]
    modes: tuple[Hashable, ...]
    block_axis_meta: Any
    backend: str

    def __post_init__(self):
        global_shape = tuple(int(dim) for dim in self.global_shape)
        modes = tuple(self.modes)
        if any(dim < 0 for dim in global_shape):
            raise ValueError("BlockTensor global_shape dimensions must be non-negative")
        if len(modes) != len(global_shape):
            raise ValueError("BlockTensor modes must match global_shape rank")
        blocks = dict(self.blocks)
        for key, block in blocks.items():
            if key != block.key:
                raise ValueError("BlockTensor block dictionary key must match block.key")
            if tuple(block.modes) != modes:
                raise ValueError("BlockTensor block modes must match tensor modes")
        self.blocks = blocks
        self.global_shape = global_shape
        self.modes = modes
        self.backend = str(self.backend)


@dataclass(frozen=True)
class BlockContractionSpec:
    left: BlockTensor
    right: BlockTensor
    output_modes: tuple[Hashable, ...]
    qn_rule: Callable[[Any, Any], Any | None]
    accumulate: bool = True

    def __post_init__(self):
        output_modes = tuple(self.output_modes)
        if not callable(self.qn_rule):
            raise ValueError("BlockContractionSpec qn_rule must be callable")
        if len(set(output_modes)) != len(output_modes):
            raise ValueError("BlockContractionSpec output_modes must be unique")
        operand_modes = set(self.left.modes) | set(self.right.modes)
        missing_modes = set(output_modes) - operand_modes
        if missing_modes:
            raise ValueError(
                "BlockContractionSpec output_modes are not present in operands: {0}"
                .format(sorted(missing_modes))
            )
        if self.left.backend != self.right.backend:
            raise ValueError("BlockContractionSpec left and right backends must match")
        object.__setattr__(self, "output_modes", output_modes)
        object.__setattr__(self, "accumulate", bool(self.accumulate))
