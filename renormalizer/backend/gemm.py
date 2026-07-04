# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any


def _prod(shape) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return int(result)


def _backend_xp(backend):
    if backend is not None:
        xp = getattr(backend, "array_namespace", None)
        if xp is not None:
            return xp
    import numpy as xp

    return xp


def _reshape(xp, array, shape):
    shape = tuple(int(dim) for dim in shape)
    reshape = getattr(xp, "reshape", None)
    if callable(reshape):
        try:
            return reshape(array, shape)
        except TypeError:
            pass
    return array.reshape(shape)


def _transpose(xp, array, axes):
    axes = tuple(int(axis) for axis in axes)
    transpose = getattr(xp, "transpose", None)
    if callable(transpose):
        try:
            return transpose(array, axes)
        except TypeError:
            pass
    permute = getattr(array, "permute", None)
    if callable(permute):
        return permute(*axes)
    return array.transpose(axes)


def _broadcast_to(xp, array, shape):
    shape = tuple(int(dim) for dim in shape)
    broadcast_to = getattr(xp, "broadcast_to", None)
    if callable(broadcast_to):
        return broadcast_to(array, shape)
    expand = getattr(array, "expand", None)
    if callable(expand):
        return expand(*shape)
    return array + xp.zeros(shape, dtype=getattr(array, "dtype", None))


def _shape_tuple(array):
    return tuple(int(dim) for dim in getattr(array, "shape", ()))


@dataclass(frozen=True)
class _ShapeArray:
    shape: tuple[int, ...]
    dtype: Any = None

    def __post_init__(self):
        shape = tuple(int(dim) for dim in self.shape)
        if any(dim < 0 for dim in shape):
            raise ValueError("shape-only array dimensions must be non-negative")
        object.__setattr__(self, "shape", shape)

    @property
    def size(self):
        return _prod(self.shape)

    @property
    def nbytes(self):
        return self.size * _dtype_itemsize(self.dtype)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = tuple(int(dim) for dim in shape)
        if any(dim < 0 for dim in shape):
            raise ValueError("shape-only reshape does not support inferred or negative dimensions")
        if _prod(shape) != self.size:
            raise ValueError("shape-only reshape must preserve element count")
        return _ShapeArray(shape=shape, dtype=self.dtype)


def _dtype_itemsize(dtype) -> int:
    if dtype is None:
        return 0
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is not None:
        return int(itemsize)
    try:
        import numpy as np

        return int(np.dtype(dtype).itemsize)
    except Exception:
        return 0


class OutputMode(Enum):
    OVERWRITE = "overwrite"
    ADD = "add"
    REDUCE_BY_GEMV = "reduce_by_gemv"

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        return cls(str(value).lower())


@dataclass(frozen=True)
class GemmTask:
    A: Any
    B: Any
    C: Any | None = None
    trans_a: bool = False
    trans_b: bool = False
    conj_a: bool = False
    conj_b: bool = False
    alpha: complex | float = 1.0
    beta: complex | float = 0.0
    tag: Any = None


@dataclass(frozen=True)
class BufferRef:
    name: str
    array: Any
    device: str
    dtype: Any
    shape: tuple[int, ...]
    nbytes: int

    def __post_init__(self):
        shape = tuple(int(dim) for dim in self.shape)
        if any(dim < 0 for dim in shape):
            raise ValueError("BufferRef shape dimensions must be non-negative")
        nbytes = int(self.nbytes)
        if nbytes < 0:
            raise ValueError("BufferRef nbytes must be non-negative")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "device", str(self.device))
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "nbytes", nbytes)

    @classmethod
    def from_array(cls, name, array, *, device="unknown"):
        return cls(
            name=str(name),
            array=array,
            device=str(device),
            dtype=getattr(array, "dtype", None),
            shape=tuple(getattr(array, "shape", ())),
            nbytes=array_nbytes(array),
        )


@dataclass(frozen=True)
class BufferTable:
    refs: Any

    def __post_init__(self):
        refs = self.refs
        if isinstance(refs, BufferTable):
            refs = refs.refs
        if hasattr(refs, "items"):
            items = refs.items()
        else:
            items = ((getattr(ref, "name", None), ref) for ref in refs)
        normalized = {}
        for name, ref in items:
            if isinstance(ref, BufferRef):
                key = str(ref.name if name is None else name)
                if str(ref.name) != key:
                    raise ValueError(
                        "BufferTable key {0!r} must match BufferRef name {1!r}".format(
                            key,
                            ref.name,
                        )
                    )
            else:
                if name is None:
                    raise ValueError("BufferTable raw arrays require explicit names")
                key = str(name)
                ref = BufferRef.from_array(key, ref)
            if key in normalized:
                raise ValueError("BufferTable duplicate buffer name {0!r}".format(key))
            normalized[key] = ref
        object.__setattr__(self, "refs", normalized)

    @classmethod
    def from_refs(cls, refs):
        return cls(tuple(refs))

    @classmethod
    def from_arrays(cls, arrays, *, device="unknown"):
        return cls({
            str(name): BufferRef.from_array(name, array, device=device)
            for name, array in arrays.items()
        })

    def __getitem__(self, name):
        return self.refs[str(name)]

    def __iter__(self):
        return iter(self.refs)

    def __len__(self):
        return len(self.refs)

    def keys(self):
        return self.refs.keys()

    def items(self):
        return self.refs.items()

    def values(self):
        return self.refs.values()

    def get(self, name, default=None):
        return self.refs.get(str(name), default)


@dataclass(frozen=True)
class BufferSlice:
    buffer: str
    offset: int
    shape: tuple[int, ...]
    leading_dim: int | None = None

    def __post_init__(self):
        offset = int(self.offset)
        shape = tuple(int(dim) for dim in self.shape)
        if offset < 0:
            raise ValueError("BufferSlice offset must be non-negative")
        if any(dim < 0 for dim in shape):
            raise ValueError("BufferSlice shape dimensions must be non-negative")
        leading_dim = None if self.leading_dim is None else int(self.leading_dim)
        if leading_dim is not None and leading_dim < 0:
            raise ValueError("BufferSlice leading_dim must be non-negative")
        object.__setattr__(self, "buffer", str(self.buffer))
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "leading_dim", leading_dim)


@dataclass(frozen=True)
class MatmulDesc:
    A: BufferSlice
    B: BufferSlice
    C: BufferSlice
    trans_a: str
    trans_b: str
    m: int
    n: int
    k: int
    lda: int
    ldb: int
    ldc: int
    alpha: complex | float = 1.0
    beta: complex | float = 0.0
    dtype: Any | None = None
    tag: Any = None

    def __post_init__(self):
        trans_a = _normalize_transpose_flag(self.trans_a, field="trans_a")
        trans_b = _normalize_transpose_flag(self.trans_b, field="trans_b")
        for field in ("m", "n", "k", "lda", "ldb", "ldc"):
            value = int(getattr(self, field))
            if value < 0:
                raise ValueError("MatmulDesc {0} must be non-negative".format(field))
            object.__setattr__(self, field, value)
        object.__setattr__(self, "trans_a", trans_a)
        object.__setattr__(self, "trans_b", trans_b)


def _normalize_transpose_flag(value, *, field):
    value = str(value).upper()
    if value not in ("N", "T", "C"):
        raise ValueError("MatmulDesc {0} must be one of N, T, C".format(field))
    return value


@dataclass(frozen=True)
class GemmGroupKey:
    dtype: str
    trans_a: str
    trans_b: str
    conj_a: bool
    conj_b: bool
    m: int
    n: int
    k: int
    lda: int
    ldb: int
    ldc: int
    batch_shape: tuple[int, ...] = ()

    def __post_init__(self):
        for field in ("m", "n", "k", "lda", "ldb", "ldc"):
            value = int(getattr(self, field))
            if value < 0:
                raise ValueError("GemmGroupKey {0} must be non-negative".format(field))
            object.__setattr__(self, field, value)
        object.__setattr__(self, "dtype", str(self.dtype))
        object.__setattr__(self, "trans_a", str(self.trans_a))
        object.__setattr__(self, "trans_b", str(self.trans_b))
        object.__setattr__(self, "conj_a", bool(self.conj_a))
        object.__setattr__(self, "conj_b", bool(self.conj_b))
        batch_shape = tuple(int(dim) for dim in self.batch_shape)
        if any(dim < 0 for dim in batch_shape):
            raise ValueError("GemmGroupKey batch_shape dimensions must be non-negative")
        object.__setattr__(self, "batch_shape", batch_shape)

    @classmethod
    def from_task(cls, task: GemmTask, *, xp=None):
        a = apply_gemm_flags(task.A, trans=task.trans_a, conj=task.conj_a, xp=xp)
        b = apply_gemm_flags(task.B, trans=task.trans_b, conj=task.conj_b, xp=xp)
        if len(a.shape) < 2 or len(b.shape) < 2:
            raise ValueError("grouped_gemm tasks must contain matrices or batched matrices")
        batch_shape = tuple(int(dim) for dim in a.shape[:-2])
        b_batch_shape = tuple(int(dim) for dim in b.shape[:-2])
        if batch_shape != b_batch_shape:
            raise ValueError("GEMM task batch dimensions must match")
        m, k_left = int(a.shape[-2]), int(a.shape[-1])
        k_right, n = int(b.shape[-2]), int(b.shape[-1])
        if k_left != k_right:
            raise ValueError("GEMM task has incompatible contracted dimensions")
        dtype_a = str(getattr(a, "dtype", None))
        dtype_b = str(getattr(b, "dtype", None))
        dtype = dtype_a if dtype_a == dtype_b else "{0},{1}".format(dtype_a, dtype_b)
        return cls(
            dtype=dtype,
            trans_a="C" if task.conj_a and task.trans_a else ("T" if task.trans_a else "N"),
            trans_b="C" if task.conj_b and task.trans_b else ("T" if task.trans_b else "N"),
            conj_a=bool(task.conj_a),
            conj_b=bool(task.conj_b),
            batch_shape=batch_shape,
            m=m,
            n=n,
            k=k_left,
            lda=k_left,
            ldb=n,
            ldc=n,
        )

    @classmethod
    def from_desc(cls, desc: MatmulDesc):
        return cls(
            dtype=str(getattr(desc.dtype, "name", desc.dtype)),
            trans_a=desc.trans_a,
            trans_b=desc.trans_b,
            conj_a=desc.trans_a == "C",
            conj_b=desc.trans_b == "C",
            batch_shape=(),
            m=desc.m,
            n=desc.n,
            k=desc.k,
            lda=desc.lda,
            ldb=desc.ldb,
            ldc=desc.ldc,
        )

    def to_dict(self):
        return {
            "dtype": self.dtype,
            "trans_a": self.trans_a,
            "trans_b": self.trans_b,
            "conj_a": self.conj_a,
            "conj_b": self.conj_b,
            "batch_shape": self.batch_shape,
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "lda": self.lda,
            "ldb": self.ldb,
            "ldc": self.ldc,
        }


@dataclass(frozen=True)
class GemmBatch:
    descs: tuple[GemmTask, ...]
    groups: dict[GemmGroupKey, tuple[int, ...]]
    gsta: tuple[int, ...]
    sorted_indices: tuple[int, ...]
    total_flops: int
    max_m: int
    max_n: int
    max_k: int

    @classmethod
    def from_tasks(cls, tasks, *, xp=None):
        sortable = []
        for original_index, task in enumerate(tasks):
            key = _gemm_group_key_from_item(task, xp=xp)
            if key.m == 0 or key.n == 0 or key.k == 0:
                continue
            work = max(_prod(key.batch_shape), 1) * key.m * key.n * key.k
            sortable.append((original_index, task, key, work))
        sortable.sort(key=lambda item: (-item[3], _gemm_group_sort_key(item[2]), item[0]))

        descs = tuple(item[1] for item in sortable)
        sorted_indices = tuple(int(item[0]) for item in sortable)
        groups = {}
        group_starts = []
        total_flops = 0
        max_m = max_n = max_k = 0
        for sorted_index, (_original_index, _task, key, work) in enumerate(sortable):
            if key not in groups:
                groups[key] = []
                group_starts.append(sorted_index)
            groups[key].append(sorted_index)
            total_flops += int(2 * work)
            max_m = max(max_m, key.m)
            max_n = max(max_n, key.n)
            max_k = max(max_k, key.k)
        frozen_groups = {key: tuple(indices) for key, indices in groups.items()}
        gsta = tuple(group_starts + [len(sortable)])
        return cls(
            descs=descs,
            groups=frozen_groups,
            gsta=gsta,
            sorted_indices=sorted_indices,
            total_flops=int(total_flops),
            max_m=int(max_m),
            max_n=int(max_n),
            max_k=int(max_k),
        )

    @classmethod
    def from_descs(cls, descs):
        return cls.from_tasks(descs)

    @property
    def group_sizes(self):
        return tuple(len(indices) for indices in self.groups.values())


def _gemm_group_key_from_item(item, *, xp=None):
    if isinstance(item, MatmulDesc):
        return GemmGroupKey.from_desc(item)
    return GemmGroupKey.from_task(item, xp=xp)


def _gemm_group_sort_key(key: GemmGroupKey):
    return (
        key.dtype,
        key.trans_a,
        key.trans_b,
        key.conj_a,
        key.conj_b,
        key.batch_shape,
        key.m,
        key.n,
        key.k,
        key.lda,
        key.ldb,
        key.ldc,
    )


@dataclass(frozen=True)
class GemvDesc:
    A: BufferSlice
    x: BufferSlice
    y: BufferSlice
    trans_a: str
    m: int
    n: int
    lda: int
    incx: int = 1
    incy: int = 1
    alpha: complex | float = 1.0
    beta: complex | float = 0.0
    dtype: Any | None = None
    tag: Any = None

    def __post_init__(self):
        trans_a = str(self.trans_a).upper()
        if trans_a not in ("N", "T", "C"):
            raise ValueError("GemvDesc trans_a must be one of N, T, C")
        for field in ("m", "n", "lda"):
            value = int(getattr(self, field))
            if value < 0:
                raise ValueError("GemvDesc {0} must be non-negative".format(field))
            object.__setattr__(self, field, value)
        for field in ("incx", "incy"):
            value = int(getattr(self, field))
            if value <= 0:
                raise ValueError("GemvDesc {0} must be positive".format(field))
            object.__setattr__(self, field, value)
        object.__setattr__(self, "trans_a", trans_a)


@dataclass(frozen=True)
class GemvGroupKey:
    dtype: str
    trans_a: str
    m: int
    n: int
    lda: int
    incx: int
    incy: int

    def __post_init__(self):
        for field in ("m", "n", "lda"):
            value = int(getattr(self, field))
            if value < 0:
                raise ValueError("GemvGroupKey {0} must be non-negative".format(field))
            object.__setattr__(self, field, value)
        for field in ("incx", "incy"):
            value = int(getattr(self, field))
            if value <= 0:
                raise ValueError("GemvGroupKey {0} must be positive".format(field))
            object.__setattr__(self, field, value)
        trans_a = str(self.trans_a).upper()
        if trans_a not in ("N", "T", "C"):
            raise ValueError("GemvGroupKey trans_a must be one of N, T, C")
        object.__setattr__(self, "dtype", str(self.dtype))
        object.__setattr__(self, "trans_a", trans_a)

    @classmethod
    def from_desc(cls, desc: GemvDesc):
        return cls(
            dtype=str(getattr(desc.dtype, "name", desc.dtype)),
            trans_a=desc.trans_a,
            m=desc.m,
            n=desc.n,
            lda=desc.lda,
            incx=desc.incx,
            incy=desc.incy,
        )

    def to_dict(self):
        return {
            "dtype": self.dtype,
            "trans_a": self.trans_a,
            "conj_a": self.trans_a == "C",
            "m": self.m,
            "n": self.n,
            "lda": self.lda,
            "incx": self.incx,
            "incy": self.incy,
        }


@dataclass(frozen=True)
class GemvBatch:
    descs: tuple[GemvDesc, ...]
    groups: dict[GemvGroupKey, tuple[int, ...]]
    gsta: tuple[int, ...]
    sorted_indices: tuple[int, ...]
    total_flops: int
    max_m: int
    max_n: int

    @classmethod
    def from_descs(cls, descs):
        sortable = []
        for original_index, desc in enumerate(descs):
            key = GemvGroupKey.from_desc(desc)
            if key.m == 0 or key.n == 0:
                continue
            work = key.m * key.n
            sortable.append((original_index, desc, key, work))
        sortable.sort(key=lambda item: (-item[3], _gemv_group_sort_key(item[2]), item[0]))

        sorted_descs = tuple(item[1] for item in sortable)
        sorted_indices = tuple(int(item[0]) for item in sortable)
        groups = {}
        group_starts = []
        total_flops = 0
        max_m = max_n = 0
        for sorted_index, (_original_index, _desc, key, work) in enumerate(sortable):
            if key not in groups:
                groups[key] = []
                group_starts.append(sorted_index)
            groups[key].append(sorted_index)
            total_flops += int(2 * work)
            max_m = max(max_m, key.m)
            max_n = max(max_n, key.n)
        frozen_groups = {key: tuple(indices) for key, indices in groups.items()}
        gsta = tuple(group_starts + [len(sortable)])
        return cls(
            descs=sorted_descs,
            groups=frozen_groups,
            gsta=gsta,
            sorted_indices=sorted_indices,
            total_flops=int(total_flops),
            max_m=int(max_m),
            max_n=int(max_n),
        )

    @property
    def group_sizes(self):
        return tuple(len(indices) for indices in self.groups.values())


def _gemv_group_sort_key(key: GemvGroupKey):
    return (
        key.dtype,
        key.trans_a,
        key.m,
        key.n,
        key.lda,
        key.incx,
        key.incy,
    )


@dataclass(frozen=True)
class HxBlock:
    block_id: int
    input_slice: BufferSlice
    output_slice: BufferSlice
    coefficient: complex | float
    inter_slice: BufferSlice | None
    matmul_stages: tuple[tuple[GemmTask, ...], ...]
    gemv_inter: tuple[GemvDesc, ...]
    gemv_reduce: tuple[GemvDesc, ...]
    qn_key: Any | None = None
    term_key: Any | None = None
    cost: float = 0.0
    output_mode: OutputMode | str = OutputMode.ADD

    def __post_init__(self):
        block_id = int(self.block_id)
        if block_id < 0:
            raise ValueError("HxBlock block_id must be non-negative")
        matmul_stages = tuple(tuple(stage) for stage in self.matmul_stages)
        gemv_inter = tuple(self.gemv_inter)
        gemv_reduce = tuple(self.gemv_reduce)
        cost = float(self.cost)
        if cost < 0:
            raise ValueError("HxBlock cost must be non-negative")
        output_mode = OutputMode.from_value(self.output_mode)
        object.__setattr__(self, "block_id", block_id)
        object.__setattr__(self, "matmul_stages", matmul_stages)
        object.__setattr__(self, "gemv_inter", gemv_inter)
        object.__setattr__(self, "gemv_reduce", gemv_reduce)
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "output_mode", output_mode)


@dataclass(frozen=True)
class HxBlockList:
    blocks: tuple[HxBlock, ...]
    center_kind: str
    direct_intermediate: bool
    qn_adapted: bool
    total_cost: float | None = None
    max_block_size: int | None = None
    equation: str | None = None
    operand_shapes: tuple[tuple[int, ...], ...] = ()
    output_shape: tuple[int, ...] = ()

    VALID_CENTER_KINDS = frozenset(("onedot", "twodot"))

    def __post_init__(self):
        blocks = tuple(self.blocks)
        center_kind = str(self.center_kind)
        if center_kind not in self.VALID_CENTER_KINDS:
            raise ValueError("HxBlockList center_kind must be one of {0}".format(sorted(self.VALID_CENTER_KINDS)))
        total_cost = self.total_cost
        if total_cost is None:
            total_cost = sum(float(block.cost) for block in blocks)
        total_cost = float(total_cost)
        if total_cost < 0:
            raise ValueError("HxBlockList total_cost must be non-negative")
        max_block_size = self.max_block_size
        if max_block_size is None:
            max_block_size = max((_hx_block_size(block) for block in blocks), default=0)
        max_block_size = int(max_block_size)
        if max_block_size < 0:
            raise ValueError("HxBlockList max_block_size must be non-negative")
        operand_shapes = tuple(tuple(int(dim) for dim in shape) for shape in self.operand_shapes)
        output_shape = tuple(int(dim) for dim in self.output_shape)
        if any(dim < 0 for shape in operand_shapes for dim in shape):
            raise ValueError("HxBlockList operand_shapes dimensions must be non-negative")
        if any(dim < 0 for dim in output_shape):
            raise ValueError("HxBlockList output_shape dimensions must be non-negative")
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "center_kind", center_kind)
        object.__setattr__(self, "direct_intermediate", bool(self.direct_intermediate))
        object.__setattr__(self, "qn_adapted", bool(self.qn_adapted))
        object.__setattr__(self, "total_cost", total_cost)
        object.__setattr__(self, "max_block_size", max_block_size)
        object.__setattr__(self, "equation", None if self.equation is None else str(self.equation))
        object.__setattr__(self, "operand_shapes", operand_shapes)
        object.__setattr__(self, "output_shape", output_shape)

    @classmethod
    def from_blocks(
        cls,
        blocks,
        *,
        center_kind,
        direct_intermediate,
        qn_adapted,
        equation=None,
        operand_shapes=(),
        output_shape=(),
    ):
        return cls(
            blocks=tuple(blocks),
            center_kind=center_kind,
            direct_intermediate=direct_intermediate,
            qn_adapted=qn_adapted,
            equation=equation,
            operand_shapes=tuple(operand_shapes),
            output_shape=tuple(output_shape),
        )


@dataclass(frozen=True)
class HMMTask:
    batches: tuple[tuple[GemmBatch, ...], ...]
    inter_batches: tuple[GemvBatch, ...]
    reduce_batches: tuple[GemvBatch, ...]
    batch_size: int
    num_batches: int
    workspace_size: int
    inter_size: int
    total_cost: float
    direct_intermediate: bool
    center_kind: str
    equation: str | None = None
    operand_shapes: tuple[tuple[int, ...], ...] = ()
    output_shape: tuple[int, ...] = ()
    num_hx_blocks: int = 0
    output_modes: tuple[str, ...] = ()
    output_mode_counts: dict[str, int] | None = None
    scatter_add_required: bool = False
    reduce_by_gemv_required: bool = False
    output_contributions: tuple[OutputContribution, ...] = ()
    output_reduction_groups: tuple[OutputReductionGroup, ...] = ()
    hx_blocks: tuple[HxBlock, ...] = ()
    batch_size_auto: bool = False
    workspace_limit_bytes: int | None = None
    static_workspace_bytes: int = 0
    block_workspace_size: int = 0
    block_workspace_bytes: int = 0

    def __post_init__(self):
        batch_size = int(self.batch_size)
        num_batches = int(self.num_batches)
        workspace_size = int(self.workspace_size)
        inter_size = int(self.inter_size)
        num_hx_blocks = int(self.num_hx_blocks)
        workspace_limit_bytes = (
            None
            if self.workspace_limit_bytes is None
            else int(self.workspace_limit_bytes)
        )
        static_workspace_bytes = int(self.static_workspace_bytes)
        block_workspace_size = int(self.block_workspace_size)
        block_workspace_bytes = int(self.block_workspace_bytes)
        if batch_size <= 0:
            raise ValueError("HMMTask batch_size must be positive")
        for field, value in (
            ("num_batches", num_batches),
            ("workspace_size", workspace_size),
            ("inter_size", inter_size),
            ("num_hx_blocks", num_hx_blocks),
            ("static_workspace_bytes", static_workspace_bytes),
            ("block_workspace_size", block_workspace_size),
            ("block_workspace_bytes", block_workspace_bytes),
        ):
            if value < 0:
                raise ValueError("HMMTask {0} must be non-negative".format(field))
        if workspace_limit_bytes is not None and workspace_limit_bytes < 0:
            raise ValueError("HMMTask workspace_limit_bytes must be non-negative")
        batches = tuple(tuple(stage_batch for stage_batch in batch) for batch in self.batches)
        inter_batches = tuple(self.inter_batches)
        reduce_batches = tuple(self.reduce_batches)
        if len(batches) != num_batches:
            raise ValueError("HMMTask num_batches must match batches length")
        if len(inter_batches) != num_batches:
            raise ValueError("HMMTask num_batches must match inter_batches length")
        if len(reduce_batches) != num_batches:
            raise ValueError("HMMTask num_batches must match reduce_batches length")
        operand_shapes = tuple(tuple(int(dim) for dim in shape) for shape in self.operand_shapes)
        output_shape = tuple(int(dim) for dim in self.output_shape)
        if any(dim < 0 for shape in operand_shapes for dim in shape):
            raise ValueError("HMMTask operand_shapes dimensions must be non-negative")
        if any(dim < 0 for dim in output_shape):
            raise ValueError("HMMTask output_shape dimensions must be non-negative")
        output_modes = tuple(str(mode) for mode in self.output_modes)
        output_mode_counts = dict(self.output_mode_counts or _output_mode_counts(output_modes))
        for mode in output_modes:
            OutputMode.from_value(mode)
        for mode, count in output_mode_counts.items():
            OutputMode.from_value(mode)
            count = int(count)
            if count < 0:
                raise ValueError("HMMTask output_mode_counts values must be non-negative")
            output_mode_counts[mode] = count
        output_contributions = tuple(self.output_contributions)
        output_reduction_groups = tuple(self.output_reduction_groups)
        hx_blocks = tuple(self.hx_blocks)
        if hx_blocks and len(hx_blocks) != num_hx_blocks:
            raise ValueError("HMMTask hx_blocks length must match num_hx_blocks")
        object.__setattr__(self, "batches", batches)
        object.__setattr__(self, "inter_batches", inter_batches)
        object.__setattr__(self, "reduce_batches", reduce_batches)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "num_batches", num_batches)
        object.__setattr__(self, "workspace_size", workspace_size)
        object.__setattr__(self, "inter_size", inter_size)
        object.__setattr__(self, "total_cost", float(self.total_cost))
        object.__setattr__(self, "direct_intermediate", bool(self.direct_intermediate))
        object.__setattr__(self, "center_kind", str(self.center_kind))
        object.__setattr__(self, "equation", None if self.equation is None else str(self.equation))
        object.__setattr__(self, "operand_shapes", operand_shapes)
        object.__setattr__(self, "output_shape", output_shape)
        object.__setattr__(self, "num_hx_blocks", num_hx_blocks)
        object.__setattr__(self, "output_modes", output_modes)
        object.__setattr__(self, "output_mode_counts", output_mode_counts)
        object.__setattr__(self, "scatter_add_required", bool(self.scatter_add_required))
        object.__setattr__(self, "reduce_by_gemv_required", bool(self.reduce_by_gemv_required))
        object.__setattr__(self, "output_contributions", output_contributions)
        object.__setattr__(self, "output_reduction_groups", output_reduction_groups)
        object.__setattr__(self, "hx_blocks", hx_blocks)
        object.__setattr__(self, "batch_size_auto", bool(self.batch_size_auto))
        object.__setattr__(self, "workspace_limit_bytes", workspace_limit_bytes)
        object.__setattr__(self, "static_workspace_bytes", static_workspace_bytes)
        object.__setattr__(self, "block_workspace_size", block_workspace_size)
        object.__setattr__(self, "block_workspace_bytes", block_workspace_bytes)


def _output_mode_counts(output_modes):
    counts = {}
    for mode in output_modes:
        key = OutputMode.from_value(mode).value
        counts[key] = counts.get(key, 0) + 1
    return counts


def _output_modes_from_blocks(blocks):
    return tuple(block.output_mode.value for block in blocks)


def _hmm_scatter_add_required(blocks, output_modes):
    if any(mode == OutputMode.ADD.value for mode in output_modes):
        return True
    seen = set()
    for block in blocks:
        output_key = (block.output_slice.buffer, block.output_slice.offset, block.output_slice.shape)
        if output_key in seen:
            return True
        seen.add(output_key)
    return False


def _hmm_reduce_by_gemv_required(output_modes):
    return any(mode == OutputMode.REDUCE_BY_GEMV.value for mode in output_modes)


def _hmm_reduction_mode_from_modes(output_modes):
    unique_modes = tuple(dict.fromkeys(output_modes))
    if not unique_modes:
        return "none"
    if len(unique_modes) == 1:
        return unique_modes[0]
    return "mixed"


@dataclass(frozen=True)
class OutputContribution:
    block_id: int
    output_key: tuple[str, int, tuple[int, ...]]
    output_mode: str
    coefficient: complex | float

    def __post_init__(self):
        block_id = int(self.block_id)
        if block_id < 0:
            raise ValueError("OutputContribution block_id must be non-negative")
        buffer, offset, shape = self.output_key
        output_key = (str(buffer), int(offset), tuple(int(dim) for dim in shape))
        if output_key[1] < 0:
            raise ValueError("OutputContribution output offset must be non-negative")
        if any(dim < 0 for dim in output_key[2]):
            raise ValueError("OutputContribution output shape dimensions must be non-negative")
        output_mode = OutputMode.from_value(self.output_mode).value
        object.__setattr__(self, "block_id", block_id)
        object.__setattr__(self, "output_key", output_key)
        object.__setattr__(self, "output_mode", output_mode)


@dataclass(frozen=True)
class OutputReductionGroup:
    output_key: tuple[str, int, tuple[int, ...]]
    contribution_indices: tuple[int, ...]
    block_ids: tuple[int, ...]
    output_modes: tuple[str, ...]
    reduction_mode: str
    scatter_add_required: bool
    reduce_by_gemv_required: bool

    def __post_init__(self):
        buffer, offset, shape = self.output_key
        output_key = (str(buffer), int(offset), tuple(int(dim) for dim in shape))
        if output_key[1] < 0:
            raise ValueError("OutputReductionGroup output offset must be non-negative")
        if any(dim < 0 for dim in output_key[2]):
            raise ValueError("OutputReductionGroup output shape dimensions must be non-negative")
        contribution_indices = tuple(int(index) for index in self.contribution_indices)
        block_ids = tuple(int(block_id) for block_id in self.block_ids)
        output_modes = tuple(OutputMode.from_value(mode).value for mode in self.output_modes)
        if any(index < 0 for index in contribution_indices):
            raise ValueError("OutputReductionGroup contribution indices must be non-negative")
        if any(block_id < 0 for block_id in block_ids):
            raise ValueError("OutputReductionGroup block ids must be non-negative")
        object.__setattr__(self, "output_key", output_key)
        object.__setattr__(self, "contribution_indices", contribution_indices)
        object.__setattr__(self, "block_ids", block_ids)
        object.__setattr__(self, "output_modes", output_modes)
        object.__setattr__(self, "reduction_mode", str(self.reduction_mode))
        object.__setattr__(self, "scatter_add_required", bool(self.scatter_add_required))
        object.__setattr__(self, "reduce_by_gemv_required", bool(self.reduce_by_gemv_required))


def _output_key_from_slice(buffer_slice: BufferSlice):
    return (buffer_slice.buffer, buffer_slice.offset, buffer_slice.shape)


def _output_contributions_from_blocks(blocks):
    return tuple(
        OutputContribution(
            block_id=block.block_id,
            output_key=_output_key_from_slice(block.output_slice),
            output_mode=block.output_mode.value,
            coefficient=block.coefficient,
        )
        for block in blocks
    )


def _output_reduction_groups_from_contributions(contributions):
    grouped = {}
    order = []
    for index, contribution in enumerate(contributions):
        key = contribution.output_key
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append((index, contribution))
    groups = []
    for key in order:
        entries = grouped[key]
        output_modes = tuple(contribution.output_mode for _index, contribution in entries)
        groups.append(OutputReductionGroup(
            output_key=key,
            contribution_indices=tuple(index for index, _contribution in entries),
            block_ids=tuple(contribution.block_id for _index, contribution in entries),
            output_modes=output_modes,
            reduction_mode=_hmm_reduction_mode_from_modes(output_modes),
            scatter_add_required=(
                len(entries) > 1
                or any(mode == OutputMode.ADD.value for mode in output_modes)
            ),
            reduce_by_gemv_required=_hmm_reduce_by_gemv_required(output_modes),
        ))
    return tuple(groups)


def choose_batch_size(
    *,
    max_blocks: int,
    block_workspace_size: int,
    static_workspace_size: int = 0,
    memory_budget_bytes: int | None = None,
) -> int:
    """Choose an HMM batch size from a per-block workspace estimate."""

    max_blocks = int(max_blocks)
    block_workspace_size = int(block_workspace_size)
    static_workspace_size = int(static_workspace_size)
    if max_blocks < 0:
        raise ValueError("max_blocks must be non-negative")
    if block_workspace_size < 0:
        raise ValueError("block_workspace_size must be non-negative")
    if static_workspace_size < 0:
        raise ValueError("static_workspace_size must be non-negative")
    if max_blocks == 0:
        return 0
    if memory_budget_bytes is None:
        return max_blocks
    memory_budget_bytes = int(memory_budget_bytes)
    if memory_budget_bytes < 0:
        raise ValueError("memory_budget_bytes must be non-negative")
    if static_workspace_size > memory_budget_bytes:
        raise ValueError("memory_budget_bytes is insufficient for static workspace")
    if block_workspace_size == 0:
        return max_blocks
    available = memory_budget_bytes - static_workspace_size
    batch_size = available // block_workspace_size
    if batch_size <= 0:
        raise ValueError("memory_budget_bytes is insufficient for one HMM block")
    return min(max_blocks, int(batch_size))


def _hxlist_itemsize(hxlist: HxBlockList) -> int:
    dtype = None
    for block in hxlist.blocks:
        for stage in block.matmul_stages:
            for desc in stage:
                if isinstance(desc, MatmulDesc):
                    dtype = desc.dtype
                    break
                dtype = _merged_dtype(getattr(desc.A, "dtype", None), getattr(desc.B, "dtype", None))
                if dtype is not None:
                    break
            if dtype is not None:
                break
        if dtype is not None:
            break
    return _dtype_itemsize(dtype) or 8


def build_hmm_task(
    hxlist: HxBlockList,
    *,
    batch_size: int | None = None,
    workspace_limit_bytes: int | None = None,
    static_workspace_bytes: int = 0,
) -> HMMTask:
    if workspace_limit_bytes is not None and int(workspace_limit_bytes) < 0:
        raise ValueError("workspace_limit_bytes must be non-negative")
    if int(static_workspace_bytes) < 0:
        raise ValueError("static_workspace_bytes must be non-negative")

    hmm_batches = []
    inter_batches = []
    reduce_batches = []
    blocks = tuple(hxlist.blocks)
    batch_size_auto = batch_size is None
    block_workspace_size = int(hxlist.max_block_size)
    itemsize = _hxlist_itemsize(hxlist)
    block_workspace_bytes = block_workspace_size * itemsize
    if batch_size is None:
        batch_size = choose_batch_size(
            max_blocks=len(blocks),
            block_workspace_size=block_workspace_bytes,
            static_workspace_size=int(static_workspace_bytes),
            memory_budget_bytes=workspace_limit_bytes,
        )
        if batch_size == 0:
            batch_size = 1
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(blocks), batch_size):
        chunk = blocks[start:start + batch_size]
        stage_count = max((len(block.matmul_stages) for block in chunk), default=0)
        stage_batches = []
        for stage_index in range(stage_count):
            stage_tasks = []
            for block in chunk:
                if stage_index < len(block.matmul_stages):
                    stage_tasks.extend(block.matmul_stages[stage_index])
            stage_batches.append(GemmBatch.from_tasks(stage_tasks))
        hmm_batches.append(tuple(stage_batches))
        inter_batches.append(GemvBatch.from_descs(desc for block in chunk for desc in block.gemv_inter))
        reduce_batches.append(GemvBatch.from_descs(desc for block in chunk for desc in block.gemv_reduce))

    workspace_size = int(hxlist.max_block_size)
    if workspace_limit_bytes is not None:
        workspace_size = min(workspace_size, int(workspace_limit_bytes))
    inter_size = max((_prod(block.inter_slice.shape) if block.inter_slice is not None else 0 for block in blocks), default=0)
    output_modes = _output_modes_from_blocks(blocks)
    output_contributions = _output_contributions_from_blocks(blocks)
    output_reduction_groups = _output_reduction_groups_from_contributions(output_contributions)
    task = HMMTask(
        batches=tuple(hmm_batches),
        inter_batches=tuple(inter_batches),
        reduce_batches=tuple(reduce_batches),
        batch_size=batch_size,
        num_batches=len(hmm_batches),
        workspace_size=workspace_size,
        inter_size=inter_size,
        total_cost=hxlist.total_cost,
        direct_intermediate=hxlist.direct_intermediate,
        center_kind=hxlist.center_kind,
        equation=hxlist.equation,
        operand_shapes=hxlist.operand_shapes,
        output_shape=hxlist.output_shape,
        num_hx_blocks=len(blocks),
        output_modes=output_modes,
        output_mode_counts=_output_mode_counts(output_modes),
        scatter_add_required=_hmm_scatter_add_required(blocks, output_modes),
        reduce_by_gemv_required=_hmm_reduce_by_gemv_required(output_modes),
        output_contributions=output_contributions,
        output_reduction_groups=output_reduction_groups,
        hx_blocks=blocks,
        batch_size_auto=batch_size_auto,
        workspace_limit_bytes=workspace_limit_bytes,
        static_workspace_bytes=static_workspace_bytes,
        block_workspace_size=block_workspace_size,
        block_workspace_bytes=block_workspace_bytes,
    )
    _record_hmm_task_build(hxlist, task)
    return task


def _record_hmm_task_build(hxlist: HxBlockList, task: HMMTask):
    try:
        from renormalizer.utils import profiling

        if not profiling.should_record_op():
            return
        num_gemm_desc = sum(len(batch.descs) for task_batch in task.batches for batch in task_batch)
        num_gemv_desc = sum(len(batch.descs) for batch in task.inter_batches)
        num_gemv_desc += sum(len(batch.descs) for batch in task.reduce_batches)
        max_stage_count = max((len(task_batch) for task_batch in task.batches), default=0)
        stage_group_sizes = []
        num_gemm = 0
        num_batched_gemm = 0
        num_grouped_tasks = 0
        for stage_index in range(max_stage_count):
            group_sizes = []
            for task_batch in task.batches:
                if stage_index < len(task_batch):
                    stage_batch = task_batch[stage_index]
                    group_sizes.extend(int(size) for size in stage_batch.group_sizes)
                    for key, desc_indices in stage_batch.groups.items():
                        if key.batch_shape:
                            num_batched_gemm += 1
                        elif len(desc_indices) == 1:
                            num_gemm += 1
                        else:
                            num_grouped_tasks += len(desc_indices)
            stage_group_sizes.append(group_sizes)
        shape_buckets = _hmm_shape_bucket_profile_items(task)
        gemv_shape_buckets = _hmm_gemv_shape_bucket_profile_items(task)
        rhs_fields = _hmm_rhs_profile_fields(task)
        dtype = _task_dtype(task)
        dtype_name = None if dtype is None else str(dtype)
        itemsize = _dtype_itemsize(dtype)
        workspace_bytes = int(task.workspace_size) * itemsize
        largest_intermediate_elements = int(task.inter_size)
        largest_intermediate = int(task.inter_size) * itemsize
        plan = None
        try:
            plan = lower_hmm_task_to_contraction_plan(task)
        except Exception:
            plan = None
        read_bytes = int(getattr(plan, "estimated_read_bytes", 0) or 0)
        write_bytes = int(getattr(plan, "estimated_write_bytes", 0) or 0)
        copy_bytes = int(getattr(plan, "estimated_copy_bytes", 0) or 0)
        peak_bytes = max(
            int(getattr(plan, "estimated_peak_bytes", 0) or 0),
            workspace_bytes,
            largest_intermediate,
        )
        required_workspace_bytes = max(
            int(getattr(plan, "required_workspace_bytes", 0) or 0),
            workspace_bytes,
        )
        total_flops = int(
            getattr(plan, "estimated_flops", None)
            if plan is not None
            else sum(batch.total_flops for task_batch in task.batches for batch in task_batch)
        )
        output_fields = _hmm_output_profile_fields(task, include_output_modes_alias=True)
        profiling.record(
            "hmm_task_build",
            backend="hmm_task",
            equation=hxlist.equation,
            lowering="hmm_task_build",
            center_kind=hxlist.center_kind,
            direct_intermediate=hxlist.direct_intermediate,
            qn_adapted=hxlist.qn_adapted,
            dtype=dtype_name,
            device_kind="planned",
            num_hx_blocks=len(hxlist.blocks),
            num_blocks=len(hxlist.blocks),
            num_batches=task.num_batches,
            batch_size=task.batch_size,
            batch_size_auto=task.batch_size_auto,
            workspace_size=task.workspace_size,
            workspace_limit_bytes=task.workspace_limit_bytes,
            static_workspace_bytes=task.static_workspace_bytes,
            block_workspace_size=task.block_workspace_size,
            block_workspace_bytes=task.block_workspace_bytes,
            hmm_memory_plan=_hmm_memory_plan_profile_item(task),
            workspace_bytes=required_workspace_bytes,
            peak_bytes=peak_bytes,
            largest_intermediate=largest_intermediate,
            largest_intermediate_elements=largest_intermediate_elements,
            largest_intermediate_bytes=largest_intermediate,
            inter_size=task.inter_size,
            num_gemm_desc=num_gemm_desc,
            num_gemv_desc=num_gemv_desc,
            num_gemm=num_gemm,
            num_batched_gemm=num_batched_gemm,
            num_grouped_tasks=num_grouped_tasks,
            num_shape_buckets=len(shape_buckets),
            num_gemv_shape_buckets=len(gemv_shape_buckets),
            num_total_shape_buckets=len(shape_buckets) + len(gemv_shape_buckets),
            total_flops=total_flops,
            flops=total_flops,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            copy_bytes=copy_bytes,
            stage_group_sizes=stage_group_sizes,
            shape_buckets=shape_buckets,
            hmm_gemv_shape_buckets=gemv_shape_buckets,
            hmm_batches=_hmm_batch_profile_items(task, plan=plan),
            hmm_gemv_batches=_hmm_gemv_batch_profile_items(task),
            hmm_execution_trace=_hmm_execution_trace_items(task, plan=plan),
            hmm_hx_blocks=_hmm_hx_block_profile_items(task.hx_blocks),
            operand_shapes=[list(shape) for shape in hxlist.operand_shapes],
            output_shape=list(hxlist.output_shape),
            **output_fields,
            **rhs_fields,
        )
    except Exception:
        return


def _hmm_memory_plan_profile_item(task: HMMTask):
    return {
        "max_blocks": int(task.num_hx_blocks),
        "batch_size": int(task.batch_size),
        "batch_size_auto": bool(task.batch_size_auto),
        "workspace_limit_bytes": task.workspace_limit_bytes,
        "static_workspace_bytes": int(task.static_workspace_bytes),
        "block_workspace_size": int(task.block_workspace_size),
        "block_workspace_bytes": int(task.block_workspace_bytes),
    }


def lower_hmm_task_to_matmul_plans(task: HMMTask):
    """Lower an HMMTask scaffold into formal MatmulPlan IR.

    Return shape:
        ``batch -> stage -> one or more MatmulPlan``.

    A stage may contain multiple shape groups.  Each shape group becomes a
    single GEMM plan when it has one task, or a grouped GEMM plan when it has
    multiple same-shape tasks.
    """

    plan_batches = []
    for batch_index, task_batch in enumerate(task.batches):
        stage_plans = []
        for stage_index, stage_batch in enumerate(task_batch):
            plans = []
            for group_indices in stage_batch.groups.values():
                descs = tuple(
                    _gemm_task_to_matmul_desc(stage_batch.descs[index])
                    for index in group_indices
                )
                if not descs:
                    continue
                reason = _hmm_stage_reason(task, batch_index, stage_index)
                if any(desc.batch_shape for desc in descs):
                    for desc in descs:
                        plans.append(_matmul_plan(
                            kind="batched_gemm",
                            descs=(desc,),
                            output_shape=tuple(desc.batch_shape) + (desc.m, desc.n),
                            copy_bytes=_layout_transform_copy_bytes(desc),
                            workspace_bytes=desc.estimated_workspace_bytes,
                            estimated_flops=desc.estimated_flops,
                            reason=reason,
                        ))
                else:
                    kind = "gemm" if len(descs) == 1 else "grouped_gemm"
                    output_shape = (descs[0].m, descs[0].n)
                    estimated_flops = sum(desc.estimated_flops for desc in descs)
                    copy_bytes = sum(_layout_transform_copy_bytes(desc) for desc in descs)
                    workspace_bytes = sum(desc.estimated_workspace_bytes for desc in descs)
                    plans.append(_matmul_plan(
                        kind=kind,
                        descs=descs,
                        output_shape=output_shape,
                        copy_bytes=copy_bytes,
                        workspace_bytes=workspace_bytes,
                        estimated_flops=estimated_flops,
                        reason=reason,
                    ))
            stage_plans.append(tuple(plans))
        plan_batches.append(tuple(stage_plans))
    return tuple(plan_batches)


def _hmm_stage_reason(task, batch_index, stage_index):
    reason = "hmm_task_stage batch={0} stage={1}".format(batch_index, stage_index)
    if task.equation:
        reason += " equation={0}".format(task.equation)
    return reason


def lower_hmm_task_to_contraction_plan(task: HMMTask):
    """Lower an HMMTask scaffold into a formal ContractionPlan."""

    from renormalizer.backend.execution import ContractionPlan, ContractionStep, TensorOperand, parse_einsum_equation

    if not task.equation:
        raise ValueError("HMMTask equation is required for ContractionPlan lowering")
    input_modes, output_modes = parse_einsum_equation(task.equation)
    if len(input_modes) != len(task.operand_shapes):
        raise ValueError("HMMTask operand_shapes must match equation operands")
    input_specs = tuple(
        TensorOperand(_ShapeArray(shape, dtype=_task_dtype(task)), modes)
        for shape, modes in zip(task.operand_shapes, input_modes)
    )
    matmul_plan_batches = lower_hmm_task_to_matmul_plans(task)
    steps = []
    next_tensor_id = len(input_specs)
    for batch_index, task_batch in enumerate(matmul_plan_batches):
        for stage_index, stage_plans in enumerate(task_batch):
            for plan_index, plan in enumerate(stage_plans):
                is_final_step = (
                    batch_index == len(matmul_plan_batches) - 1
                    and stage_index == len(task_batch) - 1
                    and plan_index == len(stage_plans) - 1
                )
                step_output_modes = output_modes if is_final_step else _stage_output_modes(
                    batch_index,
                    stage_index,
                    plan_index,
                    plan.output_shape,
                )
                left_modes, right_modes = _stage_input_modes(batch_index, stage_index, plan_index)
                step = ContractionStep(
                    kind=plan.kind,
                    inputs=(0, 1) if not steps else (steps[-1].output, 1),
                    output=next_tensor_id,
                    input_modes=(left_modes, right_modes),
                    output_modes=step_output_modes,
                    plan=plan,
                    estimated_flops=plan.estimated_flops,
                    estimated_read_bytes=sum(desc.estimated_read_bytes for desc in plan.descs),
                    estimated_write_bytes=sum(desc.estimated_write_bytes for desc in plan.descs),
                    estimated_copy_bytes=plan.copy_bytes,
                    estimated_peak_bytes=max(
                        [plan.workspace_bytes, *(desc.estimated_write_bytes for desc in plan.descs)],
                        default=0,
                    ),
                    estimated_comm_bytes=0,
                    required_workspace_bytes=plan.workspace_bytes,
                    reason=plan.reason,
                    fallback_reason=plan.fallback_reason,
                )
                steps.append(step)
                next_tensor_id += 1
    if not steps:
        raise ValueError("HMMTask must contain at least one matmul stage")
    return ContractionPlan(
        steps=tuple(steps),
        input_specs=input_specs,
        output_modes=output_modes,
        estimated_flops=sum(step.estimated_flops for step in steps),
        estimated_peak_bytes=max((step.estimated_peak_bytes for step in steps), default=0),
        estimated_read_bytes=sum(step.estimated_read_bytes for step in steps),
        estimated_write_bytes=sum(step.estimated_write_bytes for step in steps),
        estimated_copy_bytes=sum(step.estimated_copy_bytes for step in steps),
        estimated_comm_bytes=0,
        required_workspace_bytes=max((step.required_workspace_bytes for step in steps), default=0),
    )


def record_hmm_contraction_plan(task: HMMTask):
    """Record the HMM scaffold as a contraction-plan profiling event."""

    from renormalizer.utils import profiling

    if not profiling.should_record_op():
        return None
    plan = lower_hmm_task_to_contraction_plan(task)
    dtype = _task_dtype(task)
    dtype_name = None if dtype is None else str(dtype)
    itemsize = _dtype_itemsize(dtype)
    workspace_bytes = int(task.workspace_size) * itemsize
    largest_intermediate_elements = int(task.inter_size)
    largest_intermediate = int(task.inter_size) * itemsize
    step_lowerings = [str(step.kind) for step in plan.steps]
    hmm_steps = _hmm_step_profile_items(plan)
    shape_buckets = _hmm_shape_bucket_profile_items(task)
    gemv_shape_buckets = _hmm_gemv_shape_bucket_profile_items(task)
    execution_trace = _hmm_execution_trace_items(task, plan=plan)
    rhs_fields = _hmm_rhs_profile_fields(task)
    num_grouped_tasks = sum(
        len(getattr(step.plan, "descs", ()) or ())
        for step in plan.steps
        if step.kind == "grouped_gemm"
    )
    num_gemv_desc = sum(len(batch.descs) for batch in task.inter_batches)
    num_gemv_desc += sum(len(batch.descs) for batch in task.reduce_batches)

    profiling.record(
        "contraction_plan",
        backend="hmm_task",
        equation=task.equation,
        lowering="hmm_task",
        step_lowerings=step_lowerings,
        step_count=len(plan.steps),
        plan_hash=plan.plan_hash,
        matmul_plan_hashes=[
            step["plan_hash"]
            for step in hmm_steps
        ],
        operands=_hmm_operand_profile_items(plan, dtype_name, itemsize),
        input_modes=[
            [str(mode) for mode in operand.modes]
            for operand in plan.input_specs
        ],
        output_modes=[str(mode) for mode in plan.output_modes],
        input_shapes=[tuple(getattr(operand.array, "shape", ())) for operand in plan.input_specs],
        output_shape=tuple(task.output_shape or _final_step_output_shape(plan)),
        input_dtypes=[dtype_name for _operand in plan.input_specs],
        dtype=dtype_name,
        device_kind="planned",
        flops=int(plan.estimated_flops),
        read_bytes=int(plan.estimated_read_bytes),
        write_bytes=int(plan.estimated_write_bytes),
        copy_bytes=int(plan.estimated_copy_bytes),
        workspace_bytes=max(int(plan.required_workspace_bytes), workspace_bytes),
        peak_bytes=max(
            int(plan.estimated_peak_bytes),
            workspace_bytes,
            largest_intermediate,
        ),
        largest_intermediate=largest_intermediate,
        largest_intermediate_elements=largest_intermediate_elements,
        largest_intermediate_bytes=largest_intermediate,
        num_gemm=sum(1 for lowering in step_lowerings if lowering == "gemm"),
        num_batched_gemm=sum(
            1
            for lowering in step_lowerings
            if lowering in ("batched_gemm", "strided_batched_gemm")
        ),
        num_grouped_tasks=num_grouped_tasks,
        num_blocks=int(task.num_hx_blocks),
        num_hx_blocks=int(task.num_hx_blocks),
        num_batches=int(task.num_batches),
        batch_size=int(task.batch_size),
        batch_size_auto=bool(task.batch_size_auto),
        workspace_limit_bytes=task.workspace_limit_bytes,
        static_workspace_bytes=int(task.static_workspace_bytes),
        block_workspace_size=int(task.block_workspace_size),
        block_workspace_bytes=int(task.block_workspace_bytes),
        hmm_memory_plan=_hmm_memory_plan_profile_item(task),
        num_shape_buckets=len(shape_buckets),
        num_gemv_shape_buckets=len(gemv_shape_buckets),
        num_total_shape_buckets=len(shape_buckets) + len(gemv_shape_buckets),
        num_gemv_desc=num_gemv_desc,
        hmm_center_kind=task.center_kind,
        direct_intermediate=task.direct_intermediate,
        hmm_steps=hmm_steps,
        shape_buckets=shape_buckets,
        hmm_gemv_shape_buckets=gemv_shape_buckets,
        hmm_batches=_hmm_batch_profile_items(task, plan=plan),
        hmm_gemv_batches=_hmm_gemv_batch_profile_items(task),
        hmm_execution_trace=execution_trace,
        hmm_hx_blocks=_hmm_hx_block_profile_items(task.hx_blocks),
        fallback_reason=next((step.fallback_reason for step in plan.steps if step.fallback_reason), None),
        **_hmm_output_profile_fields(task),
        **rhs_fields,
    )
    return plan


def hmm_task_execution_metadata(plan, task: HMMTask):
    """Return compact HMM plan details suitable for execution events."""

    num_gemv_desc = sum(len(batch.descs) for batch in task.inter_batches)
    num_gemv_desc += sum(len(batch.descs) for batch in task.reduce_batches)
    execution_trace = _hmm_execution_trace_items(task, plan=plan)
    rhs_fields = _hmm_rhs_profile_fields(task)
    return {
        "hmm_steps": _hmm_step_profile_items(plan),
        "hmm_shape_buckets": _hmm_shape_bucket_profile_items(task),
        "hmm_gemv_shape_buckets": _hmm_gemv_shape_bucket_profile_items(task),
        "hmm_batches": _hmm_batch_profile_items(task, plan=plan),
        "hmm_gemv_batches": _hmm_gemv_batch_profile_items(task),
        "hmm_execution_trace": execution_trace,
        "hmm_hx_blocks": _hmm_hx_block_profile_items(task.hx_blocks),
        "hmm_memory_plan": _hmm_memory_plan_profile_item(task),
        "hmm_num_batches": int(task.num_batches),
        "hmm_batch_size": int(task.batch_size),
        "hmm_batch_size_auto": bool(task.batch_size_auto),
        "hmm_workspace_limit_bytes": task.workspace_limit_bytes,
        "hmm_static_workspace_bytes": int(task.static_workspace_bytes),
        "hmm_block_workspace_size": int(task.block_workspace_size),
        "hmm_block_workspace_bytes": int(task.block_workspace_bytes),
        "hmm_num_gemv_desc": int(num_gemv_desc),
        "hmm_center_kind": task.center_kind,
        "hmm_direct_intermediate": bool(task.direct_intermediate),
        **_hmm_execution_summary_fields(execution_trace),
        **_hmm_output_profile_fields(task),
        **rhs_fields,
        **_hmm_prefixed_rhs_fields(rhs_fields),
    }


def _normalize_hmm_gemv_role(role):
    value = str(role).lower()
    if value in ("inter", "inter_gemv"):
        return "inter", "inter_gemv"
    if value in ("reduce", "reduce_gemv"):
        return "reduce", "reduce_gemv"
    raise ValueError("HMM GEMV role must be one of inter, inter_gemv, reduce, reduce_gemv")


def execute_hmm_gemv_batch(
    backend,
    task: HMMTask,
    *,
    batch_index: int,
    role,
    buffers,
    stream=None,
    workspace=None,
    fallback_policy=None,
):
    """Execute one HMM inter/reduce GEMV phase through the backend primitive."""

    role_key, phase = _normalize_hmm_gemv_role(role)
    batch_index = int(batch_index)
    if batch_index < 0 or batch_index >= int(task.num_batches):
        raise ValueError("HMM GEMV batch_index out of range")
    if role_key == "inter":
        batch = task.inter_batches[batch_index]
    else:
        batch = task.reduce_batches[batch_index]
    trace_item = _hmm_gemv_trace_item(phase, batch_index, batch, task=task, role=role_key)
    if role_key == "reduce":
        trace_item.update(_hmm_batch_output_reduction_profile_fields(task, batch_index))
    profile_context = {
        "hmm_phase": phase,
        "hmm_batch": batch_index,
        "hmm_center_kind": task.center_kind,
        "hmm_direct_intermediate": bool(task.direct_intermediate),
        "hmm_qn_adapted": bool(any(block.qn_key is not None for block in task.hx_blocks)),
        "hmm_num_batches": int(task.num_batches),
        "hmm_batch_size": int(task.batch_size),
        "hmm_trace_item": trace_item,
    }
    for key in (
        "task_block_ids",
        "task_block_gemv_indices",
        "output_reduction_group_indices",
        "output_reduction_groups",
    ):
        if key in trace_item:
            profile_context[key] = trace_item[key]
    return backend.gemv_batch(
        batch,
        buffers=buffers,
        role=phase,
        stream=stream,
        workspace=workspace,
        fallback_policy=fallback_policy,
        profile_context=profile_context,
    )


def _hmm_phase_summary(trace):
    phase_order = []
    counts = {}
    tasks = {}
    groups = {}
    flops = {}
    for item in trace:
        phase = str(item.get("phase"))
        if phase not in counts:
            phase_order.append(phase)
            counts[phase] = 0
            tasks[phase] = 0
            groups[phase] = 0
            flops[phase] = 0
        counts[phase] += 1
        tasks[phase] += int(item.get("num_tasks", 0) or 0)
        groups[phase] += int(item.get("num_groups", 0) or 0)
        flops[phase] += int(item.get("total_flops", 0) or 0)
    return {
        "phase_counts": {phase: int(counts[phase]) for phase in phase_order},
        "phase_tasks": {phase: int(tasks[phase]) for phase in phase_order},
        "phase_groups": {phase: int(groups[phase]) for phase in phase_order},
        "phase_flops": {phase: int(flops[phase]) for phase in phase_order},
        "hmm_phase_counts": {phase: int(counts[phase]) for phase in phase_order},
        "hmm_phase_tasks": {phase: int(tasks[phase]) for phase in phase_order},
        "hmm_phase_groups": {phase: int(groups[phase]) for phase in phase_order},
        "hmm_phase_flops": {phase: int(flops[phase]) for phase in phase_order},
    }


def _backend_last_execution_profile(backend):
    getter = getattr(backend, "last_execution_profile", None)
    if callable(getter):
        profile = getter()
        if profile is not None:
            return dict(profile)
    profile = getattr(backend, "_last_execution_profile", None)
    return dict(profile) if profile is not None else {}


def _hmm_gemm_stage_profile_context(task: HMMTask, batch_index, stage_index):
    trace_item = next(
        item
        for item in _hmm_execution_trace_items(task)
        if item.get("phase") == "gemm_stage"
        and int(item.get("batch", -1)) == int(batch_index)
        and int(item.get("stage", -1)) == int(stage_index)
    )
    profile_context = {
        "hmm_phase": "gemm_stage",
        "hmm_batch": int(batch_index),
        "hmm_stage": int(stage_index),
        "hmm_center_kind": task.center_kind,
        "hmm_direct_intermediate": bool(task.direct_intermediate),
        "hmm_qn_adapted": bool(any(block.qn_key is not None for block in task.hx_blocks)),
        "hmm_num_batches": int(task.num_batches),
        "hmm_batch_size": int(task.batch_size),
        "hmm_trace_item": trace_item,
    }
    for key in (
        "task_block_ids",
        "task_block_stage_task_indices",
    ):
        if key in trace_item:
            profile_context[key] = trace_item[key]
    return profile_context


def execute_hmm_task(
    backend,
    task: HMMTask,
    *,
    buffers,
    stream=None,
    workspace=None,
    pack_threshold=4,
    policy="auto",
    fallback_policy=None,
):
    """Execute an HMMTask buffer plan in FOCUS order through backend primitives."""

    import time

    started = time.perf_counter()
    phase_profiles = []
    phase_results = []
    for batch_index in range(task.num_batches):
        if task.inter_batches[batch_index].descs:
            phase_results.append(execute_hmm_gemv_batch(
                backend,
                task,
                batch_index=batch_index,
                role="inter",
                buffers=buffers,
                stream=stream,
                workspace=workspace,
                fallback_policy=fallback_policy,
            ))
            phase_profiles.append(_backend_last_execution_profile(backend))

        for stage_index, stage_batch in enumerate(task.batches[batch_index]):
            if not stage_batch.descs:
                continue
            profile_context = _hmm_gemm_stage_profile_context(task, batch_index, stage_index)
            result = backend.grouped_gemm(
                stage_batch,
                buffers=buffers,
                pack_threshold=pack_threshold,
                stream=stream,
                workspace=workspace,
                policy=policy,
                fallback_policy=fallback_policy,
                profile_context=profile_context,
            )
            profile = _backend_last_execution_profile(backend)
            profile.update(profile_context)
            phase_results.append(result)
            phase_profiles.append(profile)

        if task.reduce_batches[batch_index].descs:
            phase_results.append(execute_hmm_gemv_batch(
                backend,
                task,
                batch_index=batch_index,
                role="reduce",
                buffers=buffers,
                stream=stream,
                workspace=workspace,
                fallback_policy=fallback_policy,
            ))
            phase_profiles.append(_backend_last_execution_profile(backend))

    wall_s = time.perf_counter() - started
    trace = _hmm_execution_trace_items(task)
    phase_summary = _hmm_phase_summary(trace)
    num_gemm_desc = sum(len(stage_batch.descs) for batch in task.batches for stage_batch in batch)
    num_gemv_desc = sum(len(batch.descs) for batch in task.inter_batches)
    num_gemv_desc += sum(len(batch.descs) for batch in task.reduce_batches)
    phase_events = [
        str(profile.get("event"))
        for profile in phase_profiles
        if profile.get("event")
    ]
    read_bytes = sum(int(profile.get("read_bytes", 0) or 0) for profile in phase_profiles)
    write_bytes = sum(int(profile.get("write_bytes", 0) or 0) for profile in phase_profiles)
    copy_bytes = sum(int(profile.get("copy_bytes", 0) or 0) for profile in phase_profiles)
    workspace_bytes = max((int(profile.get("workspace_bytes", 0) or 0) for profile in phase_profiles), default=0)
    largest_intermediate = max((int(profile.get("largest_intermediate", 0) or 0) for profile in phase_profiles), default=0)
    total_flops = sum(int(item.get("total_flops", 0) or 0) for item in trace)
    execution_summary = _hmm_execution_summary_fields(phase_profiles)
    profile = {
        "event": "contraction_execute",
        "backend": getattr(backend, "name", type(backend).__name__),
        "equation": task.equation,
        "lowering": "hmm_task_buffers",
        "center_kind": task.center_kind,
        "hmm_center_kind": task.center_kind,
        "direct_intermediate": bool(task.direct_intermediate),
        "hmm_direct_intermediate": bool(task.direct_intermediate),
        "hmm_qn_adapted": bool(any(block.qn_key is not None for block in task.hx_blocks)),
        "num_batches": int(task.num_batches),
        "hmm_num_batches": int(task.num_batches),
        "batch_size": int(task.batch_size),
        "hmm_batch_size": int(task.batch_size),
        "batch_size_auto": bool(task.batch_size_auto),
        "hmm_batch_size_auto": bool(task.batch_size_auto),
        "workspace_limit_bytes": task.workspace_limit_bytes,
        "hmm_workspace_limit_bytes": task.workspace_limit_bytes,
        "static_workspace_bytes": int(task.static_workspace_bytes),
        "hmm_static_workspace_bytes": int(task.static_workspace_bytes),
        "block_workspace_size": int(task.block_workspace_size),
        "hmm_block_workspace_size": int(task.block_workspace_size),
        "block_workspace_bytes": int(task.block_workspace_bytes),
        "hmm_block_workspace_bytes": int(task.block_workspace_bytes),
        "hmm_memory_plan": _hmm_memory_plan_profile_item(task),
        "num_gemm_desc": int(num_gemm_desc),
        "num_gemv_desc": int(num_gemv_desc),
        "num_blocks": int(task.num_hx_blocks),
        "num_hx_blocks": int(task.num_hx_blocks),
        "num_shape_buckets": sum(len(stage_batch.groups) for batch in task.batches for stage_batch in batch),
        "num_gemv_shape_buckets": sum(len(batch.groups) for batch in task.inter_batches + task.reduce_batches),
        "hmm_execution_trace": trace,
        "phase_events": phase_events,
        "phase_profiles": phase_profiles,
        "flops": int(total_flops),
        "total_flops": int(total_flops),
        "read_bytes": int(read_bytes),
        "write_bytes": int(write_bytes),
        "copy_bytes": int(copy_bytes),
        "workspace_bytes": int(workspace_bytes),
        "peak_bytes": int(read_bytes + write_bytes + copy_bytes + workspace_bytes),
        "largest_intermediate": int(largest_intermediate),
        **execution_summary,
        "fallback_reason": next(
            (
                profile_item.get("fallback_reason")
                for profile_item in phase_profiles
                if profile_item.get("fallback_reason")
            ),
            None,
        ),
        "wall_s": float(wall_s),
        **phase_summary,
        **_hmm_output_profile_fields(task),
    }
    try:
        from renormalizer.utils import profiling

        profile.update(profiling.contraction_execute_compute_payload("hmm_task_buffers"))
        profile.update(_backend_execution_device_payload(backend))
        profile.update(_backend_execution_resource_payload(backend, stream=stream, workspace=workspace))
        profile = profiling.standardize_event_payload(profile)
        setattr(backend, "_last_execution_profile", dict(profile))
        if profiling.should_record_op():
            record_payload = dict(profile)
            record_payload.pop("event", None)
            profiling.record("contraction_execute", **record_payload)
    except Exception:
        setattr(backend, "_last_execution_profile", dict(profile))
    return profile


def _hmm_output_profile_fields(task: HMMTask, *, include_output_modes_alias=False):
    blocks_by_id = {
        int(block.block_id): block
        for block in task.hx_blocks
    }
    fields = {
        "output_contribution_modes": list(task.output_modes),
        "output_mode_counts": dict(task.output_mode_counts),
        "reduction_mode": _hmm_reduction_mode_from_modes(task.output_modes),
        "scatter_add_required": bool(task.scatter_add_required),
        "reduce_by_gemv_required": bool(task.reduce_by_gemv_required),
        "num_output_contributions": int(len(task.output_contributions)),
        "num_output_reduction_groups": int(len(task.output_reduction_groups)),
        "num_scatter_add_tasks": int(sum(
            len(group.contribution_indices)
            for group in task.output_reduction_groups
            if group.scatter_add_required
        )),
        "output_contributions": [
            _output_contribution_profile_item(contribution)
            for contribution in task.output_contributions
        ],
        "output_reduction_groups": [
            _output_reduction_group_profile_item(
                group,
                task.output_contributions,
                blocks_by_id,
            )
            for group in task.output_reduction_groups
        ],
    }
    if include_output_modes_alias:
        fields["output_modes"] = list(task.output_modes)
    return fields


def _output_key_profile_item(output_key):
    buffer, offset, shape = output_key
    return [str(buffer), int(offset), [int(dim) for dim in shape]]


def _buffer_slice_profile_item(buffer_slice):
    if buffer_slice is None:
        return None
    return [
        str(buffer_slice.buffer),
        int(buffer_slice.offset),
        [int(dim) for dim in buffer_slice.shape],
    ]


def _metadata_profile_item(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, (tuple, list)):
        return [_metadata_profile_item(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _metadata_profile_item(item) for key, item in value.items()}
    return str(value)


def _hmm_hx_block_profile_items(blocks):
    return [
        {
            "block_id": int(block.block_id),
            "input_slice": _buffer_slice_profile_item(block.input_slice),
            "output_slice": _buffer_slice_profile_item(block.output_slice),
            "inter_slice": _buffer_slice_profile_item(block.inter_slice),
            "coefficient": _serializable_scalar(block.coefficient),
            "output_mode": block.output_mode.value,
            "qn_key": _metadata_profile_item(block.qn_key),
            "term_key": _metadata_profile_item(block.term_key),
            "cost": float(block.cost),
            "matmul_stage_count": int(len(block.matmul_stages)),
            "matmul_stage_task_counts": [
                int(len(stage))
                for stage in block.matmul_stages
            ],
            "gemv_inter_count": int(len(block.gemv_inter)),
            "gemv_reduce_count": int(len(block.gemv_reduce)),
        }
        for block in blocks
    ]


def _output_contribution_profile_item(contribution: OutputContribution):
    return {
        "block_id": int(contribution.block_id),
        "output_key": _output_key_profile_item(contribution.output_key),
        "output_mode": contribution.output_mode,
        "coefficient": _serializable_scalar(contribution.coefficient),
    }


def _output_reduction_group_contribution_profile_item(index, contribution, block):
    item = {
        "contribution_index": int(index),
        "block_id": int(contribution.block_id),
        "output_mode": contribution.output_mode,
        "coefficient": _serializable_scalar(contribution.coefficient),
    }
    if block is not None:
        item.update({
            "input_slice": _buffer_slice_profile_item(block.input_slice),
            "output_slice": _buffer_slice_profile_item(block.output_slice),
            "inter_slice": _buffer_slice_profile_item(block.inter_slice),
        })
    else:
        item["output_slice"] = _output_key_profile_item(contribution.output_key)
    return item


def _output_reduction_group_profile_item(group: OutputReductionGroup, contributions=(), blocks_by_id=None):
    blocks_by_id = blocks_by_id or {}
    contributions = tuple(contributions)
    return {
        "output_key": _output_key_profile_item(group.output_key),
        "contribution_indices": [int(index) for index in group.contribution_indices],
        "block_ids": [int(block_id) for block_id in group.block_ids],
        "output_modes": list(group.output_modes),
        "reduction_mode": group.reduction_mode,
        "scatter_add_required": bool(group.scatter_add_required),
        "reduce_by_gemv_required": bool(group.reduce_by_gemv_required),
        "contributions": [
            _output_reduction_group_contribution_profile_item(
                index,
                contributions[index],
                blocks_by_id.get(int(contributions[index].block_id)),
            )
            for index in group.contribution_indices
            if 0 <= index < len(contributions)
        ],
    }


def _serializable_scalar(value):
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def _hmm_rhs_profile_fields(task: HMMTask):
    if not task.equation or "->" not in task.equation:
        return {}
    output_modes = task.equation.split("->", 1)[1].replace(" ", "")
    if not output_modes or output_modes[-1] != "r" or not task.output_shape:
        return {}
    return {
        "rhs_batch_mode": "r",
        "num_rhs": int(task.output_shape[-1]),
        "num_rhs_loop_calls": 0,
    }


def _hmm_prefixed_rhs_fields(rhs_fields):
    if not rhs_fields:
        return {}
    return {
        "hmm_rhs_batch_mode": rhs_fields.get("rhs_batch_mode"),
        "hmm_num_rhs": rhs_fields.get("num_rhs"),
        "hmm_num_rhs_loop_calls": rhs_fields.get("num_rhs_loop_calls"),
    }


def _hmm_execution_summary_fields(execution_trace):
    primitives = []
    policies = []
    fallback_reasons = []
    fallback_from = []
    fallback_to = []
    fallback_policies = []
    grouped_gemm_policies = []
    grouped_gemm_implementations = []
    requires_grouped_gemm_fallback = False
    for item in execution_trace:
        if not isinstance(item, dict):
            continue
        primitives.extend(item.get("execution_primitives") or ())
        policies.extend(item.get("execution_policies") or ())
        fallback_reasons.extend(item.get("fallback_reasons") or ())
        if item.get("fallback_from"):
            fallback_from.append(item["fallback_from"])
        if item.get("fallback_to"):
            fallback_to.append(item["fallback_to"])
        if item.get("fallback_policy"):
            fallback_policies.append(item["fallback_policy"])
        grouped_gemm_policy = item.get("grouped_gemm_policy")
        if grouped_gemm_policy:
            grouped_gemm_policies.append(grouped_gemm_policy)
        grouped_gemm_implementation = item.get("grouped_gemm_implementation")
        if grouped_gemm_implementation:
            grouped_gemm_implementations.append(grouped_gemm_implementation)
        if item.get("requires_grouped_gemm_fallback"):
            requires_grouped_gemm_fallback = True
    return {
        "execution_primitives": _unique_preserve_order(str(item) for item in primitives if item),
        "execution_policies": _unique_preserve_order(str(item) for item in policies if item),
        "fallback_reasons": _unique_preserve_order(str(item) for item in fallback_reasons if item),
        "fallback_from": str(fallback_from[0]) if fallback_from else None,
        "fallback_to": str(fallback_to[0]) if fallback_to else None,
        "fallback_policy": str(fallback_policies[0]) if fallback_policies else None,
        "fallback_sources": _unique_preserve_order(str(item) for item in fallback_from if item),
        "fallback_targets": _unique_preserve_order(str(item) for item in fallback_to if item),
        "fallback_policies": _unique_preserve_order(str(item) for item in fallback_policies if item),
        "grouped_gemm_policies": _unique_preserve_order(
            str(item) for item in grouped_gemm_policies if item
        ),
        "grouped_gemm_implementations": _unique_preserve_order(
            str(item) for item in grouped_gemm_implementations if item
        ),
        "requires_grouped_gemm_fallback": bool(requires_grouped_gemm_fallback),
    }


def _hmm_operand_profile_items(plan, dtype_name, itemsize):
    items = []
    for index, operand in enumerate(plan.input_specs):
        shape = tuple(getattr(operand.array, "shape", ()))
        items.append({
            "index": index,
            "name": operand.name,
            "modes": [str(mode) for mode in operand.modes],
            "shape": shape,
            "dtype": dtype_name,
            "nbytes": _prod(shape) * int(itemsize),
        })
    return items


def _hmm_step_profile_items(plan):
    items = []
    for index, step in enumerate(plan.steps):
        matmul_plan = step.plan
        descs = tuple(getattr(matmul_plan, "descs", ()) or ())
        items.append({
            "index": index,
            "kind": step.kind,
            "plan_hash": getattr(matmul_plan, "plan_hash", None),
            "output_shape": tuple(getattr(matmul_plan, "output_shape", ())),
            "output_modes": [str(mode) for mode in step.output_modes],
            "flops": int(step.estimated_flops),
            "read_bytes": int(step.estimated_read_bytes),
            "write_bytes": int(step.estimated_write_bytes),
            "copy_bytes": int(step.estimated_copy_bytes),
            "workspace_bytes": int(step.required_workspace_bytes),
            "fallback_reason": step.fallback_reason,
            "num_desc": len(descs),
            "descs": [_hmm_matmul_desc_profile_item(desc_index, desc) for desc_index, desc in enumerate(descs)],
        })
    return items


def _hmm_execution_primitive(lowering):
    if lowering == "gemm":
        return "matmul"
    if lowering in ("batched_gemm", "strided_batched_gemm"):
        return "batched_matmul"
    if lowering in ("grouped_gemm", "block_grouped_gemm"):
        return "grouped_gemm"
    if lowering == "fallback_tensordot":
        return "tensordot"
    if lowering == "fallback_einsum":
        return "einsum"
    return str(lowering)


def _hmm_execution_policy(lowering):
    primitive = _hmm_execution_primitive(lowering)
    if lowering in ("fallback_tensordot", "fallback_einsum"):
        return str(lowering)
    return "backend_{0}".format(primitive)


def _unique_preserve_order(items):
    result = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _bucket_profile_reasons(bucket_profiles, field):
    return _unique_preserve_order(
        str(item.get(field))
        for item in bucket_profiles
        if isinstance(item, dict) and item.get(field)
    )


def _bucket_kernel_profile(bucket_profiles):
    primitives = []
    calls_by_primitive = {}
    for item in bucket_profiles:
        if not isinstance(item, dict):
            continue
        execution = str(item.get("execution") or "")
        if execution == "batched":
            primitive = "batched_matmul"
        elif execution == "loop":
            primitive = "matmul"
        else:
            primitive = execution or "unknown"
        if primitive not in primitives:
            primitives.append(primitive)
        calls_by_primitive[primitive] = (
            calls_by_primitive.get(primitive, 0)
            + int(item.get("kernel_calls", 0) or 0)
        )
    return primitives, calls_by_primitive


def _hmm_matmul_desc_profile_item(index, desc):
    return {
        "index": index,
        "m": int(desc.m),
        "n": int(desc.n),
        "k": int(desc.k),
        "batch_shape": tuple(desc.batch_shape),
        "trans_a": bool(desc.trans_a),
        "trans_b": bool(desc.trans_b),
        "conj_a": bool(desc.conj_a),
        "conj_b": bool(desc.conj_b),
        "dtype_compute": None if desc.dtype_compute is None else str(desc.dtype_compute),
        "dtype_output": None if desc.dtype_output is None else str(desc.dtype_output),
        "estimated_flops": int(desc.estimated_flops),
        "estimated_read_bytes": int(desc.estimated_read_bytes),
        "estimated_write_bytes": int(desc.estimated_write_bytes),
        "estimated_workspace_bytes": int(desc.estimated_workspace_bytes),
        "a_shape": tuple(getattr(desc.A, "shape", ())),
        "b_shape": tuple(getattr(desc.B, "shape", ())),
        "c_shape": None if desc.C is None else tuple(getattr(desc.C, "shape", ())),
    }


def _hmm_shape_bucket_profile_items(task: HMMTask):
    buckets = []
    for batch_index, task_batch in enumerate(task.batches):
        for stage_index, stage_batch in enumerate(task_batch):
            provenance = _hmm_stage_task_provenance(task, batch_index, stage_index)
            for group_index, (key, desc_indices) in enumerate(stage_batch.groups.items()):
                task_block_ids, task_block_stage_task_indices = _hmm_group_provenance_items(
                    provenance,
                    stage_batch.sorted_indices,
                    desc_indices,
                )
                planned_lowering = _hmm_group_planned_lowering(key, desc_indices)
                buckets.append({
                    "batch": batch_index,
                    "stage": stage_index,
                    "group": group_index,
                    "dtype": key.dtype,
                    "trans_a": key.trans_a,
                    "trans_b": key.trans_b,
                    "conj_a": key.conj_a,
                    "conj_b": key.conj_b,
                    "batch_shape": key.batch_shape,
                    "batch_count": max(_prod(key.batch_shape), 1),
                    "m": key.m,
                    "n": key.n,
                    "k": key.k,
                    "lda": key.lda,
                    "ldb": key.ldb,
                    "ldc": key.ldc,
                    "task_indices": tuple(int(index) for index in desc_indices),
                    "task_count": len(desc_indices),
                    "task_block_ids": task_block_ids,
                    "task_block_stage_task_indices": task_block_stage_task_indices,
                    "planned_lowering": planned_lowering,
                    **_hmm_stage_execution_items([planned_lowering]),
                    "planned_plan_count": _hmm_group_plan_count(key, desc_indices),
                })
    return buckets


def _hmm_gemv_shape_bucket_profile_items(task: HMMTask):
    buckets = []
    for role, batches in (("inter", task.inter_batches), ("reduce", task.reduce_batches)):
        for batch_index, batch in enumerate(batches):
            provenance = _hmm_gemv_task_provenance(task, batch_index, role)
            for group_index, (key, desc_indices) in enumerate(batch.groups.items()):
                task_block_ids, task_block_gemv_indices = _hmm_group_provenance_items(
                    provenance,
                    batch.sorted_indices,
                    desc_indices,
                )
                buckets.append({
                    "phase": role,
                    "batch": int(batch_index),
                    "group": int(group_index),
                    "dtype": key.dtype,
                    "trans_a": key.trans_a,
                    "conj_a": key.trans_a == "C",
                    "m": int(key.m),
                    "n": int(key.n),
                    "lda": int(key.lda),
                    "incx": int(key.incx),
                    "incy": int(key.incy),
                    "task_indices": [int(index) for index in desc_indices],
                    "task_count": int(len(desc_indices)),
                    "task_block_ids": task_block_ids,
                    "task_block_gemv_indices": task_block_gemv_indices,
                    **_hmm_gemv_execution_items(batch),
                })
    return buckets


def _hmm_stage_task_provenance(task: HMMTask, batch_index, stage_index):
    start = int(batch_index) * int(task.batch_size)
    stop = min(start + int(task.batch_size), len(task.hx_blocks))
    provenance = []
    for block in task.hx_blocks[start:stop]:
        if int(stage_index) >= len(block.matmul_stages):
            continue
        for stage_task_index, _task in enumerate(block.matmul_stages[int(stage_index)]):
            provenance.append((int(block.block_id), int(stage_task_index)))
    return provenance


def _hmm_gemv_task_provenance(task: HMMTask, batch_index, role: str):
    if role == "inter":
        attr = "gemv_inter"
    elif role == "reduce":
        attr = "gemv_reduce"
    else:
        return []

    start = int(batch_index) * int(task.batch_size)
    stop = min(start + int(task.batch_size), len(task.hx_blocks))
    provenance = []
    for block in task.hx_blocks[start:stop]:
        for gemv_index, _desc in enumerate(getattr(block, attr)):
            provenance.append((int(block.block_id), int(gemv_index)))
    return provenance


def _hmm_group_provenance_items(provenance, sorted_indices, desc_indices):
    task_block_ids = []
    task_local_indices = []
    for desc_index in desc_indices:
        original_index = int(sorted_indices[int(desc_index)])
        if original_index >= len(provenance):
            continue
        block_id, local_index = provenance[original_index]
        task_block_ids.append(int(block_id))
        task_local_indices.append(int(local_index))
    return task_block_ids, task_local_indices


def _hmm_batch_profile_items(task: HMMTask, *, plan=None):
    plan_steps = _hmm_step_profile_items(plan) if plan is not None else ()
    plan_cursor = 0
    batches = []
    for batch_index, task_batch in enumerate(task.batches):
        stages = []
        for stage_index, stage_batch in enumerate(task_batch):
            plan_count = _hmm_stage_plan_count(stage_batch)
            planned_lowerings = _hmm_stage_planned_lowerings(stage_batch)
            stage_steps = ()
            plan_hashes = []
            if plan is not None:
                plan_end = min(plan_cursor + plan_count, len(plan.steps))
                stage_steps = tuple(plan.steps[index] for index in range(plan_cursor, plan_end))
                plan_hashes = [
                    plan_steps[index]["plan_hash"]
                    for index in range(plan_cursor, min(plan_cursor + plan_count, len(plan_steps)))
                ]
                plan_cursor += plan_count
            stage_item = {
                "stage": int(stage_index),
                "num_tasks": int(len(stage_batch.descs)),
                "num_groups": int(len(stage_batch.groups)),
                "group_sizes": [int(size) for size in stage_batch.group_sizes],
                "planned_lowerings": planned_lowerings,
                **_hmm_stage_execution_items(planned_lowerings, stage_steps=stage_steps),
                "planned_plan_counts": _hmm_stage_plan_counts(stage_batch),
                **_hmm_stage_task_block_profile_fields(task, batch_index, stage_index),
                "task_descriptors": _hmm_stage_task_descriptor_items(
                    task,
                    batch_index,
                    stage_index,
                    stage_batch,
                ),
                "gsta": [int(index) for index in stage_batch.gsta],
                "sorted_indices": [int(index) for index in stage_batch.sorted_indices],
                "group_keys": [
                    key.to_dict()
                    for key in stage_batch.groups
                ],
                "total_flops": int(stage_batch.total_flops),
                "max_m": int(stage_batch.max_m),
                "max_n": int(stage_batch.max_n),
                "max_k": int(stage_batch.max_k),
            }
            if plan is not None:
                stage_item["plan_hashes"] = plan_hashes
            stages.append(stage_item)
        batches.append({
            "batch": int(batch_index),
            "num_stages": int(len(stages)),
            "stages": stages,
        })
    return batches


def _hmm_gemv_batch_profile_items(task: HMMTask):
    return {
        "inter": _hmm_gemv_role_profile_items(task.inter_batches, task=task, role="inter"),
        "reduce": _hmm_gemv_role_profile_items(task.reduce_batches, task=task, role="reduce"),
    }


def _hmm_gemv_role_profile_items(batches, *, task: HMMTask | None = None, role: str | None = None):
    items = []
    for batch_index, batch in enumerate(batches):
        items.append({
            "batch": int(batch_index),
            "num_tasks": int(len(batch.descs)),
            "num_groups": int(len(batch.groups)),
            "group_sizes": [int(size) for size in batch.group_sizes],
            **_hmm_gemv_execution_items(batch),
            **_hmm_gemv_task_block_profile_fields(task, batch_index, role),
            "task_descriptors": _hmm_gemv_task_descriptor_items(
                task,
                batch_index,
                role,
                batch,
            ),
            "gsta": [int(index) for index in batch.gsta],
            "sorted_indices": [int(index) for index in batch.sorted_indices],
            "group_keys": [
                key.to_dict()
                for key in batch.groups
            ],
            "total_flops": int(batch.total_flops),
            "max_m": int(batch.max_m),
            "max_n": int(batch.max_n),
        })
    return items


def _hmm_execution_trace_items(task: HMMTask, *, plan=None):
    plan_steps = _hmm_step_profile_items(plan) if plan is not None else ()
    plan_cursor = 0
    items = []
    for batch_index in range(task.num_batches):
        inter_batch = task.inter_batches[batch_index]
        items.append(_hmm_gemv_trace_item("inter_gemv", batch_index, inter_batch, task=task, role="inter"))

        for stage_index, stage_batch in enumerate(task.batches[batch_index]):
            plan_count = _hmm_stage_plan_count(stage_batch)
            planned_lowerings = _hmm_stage_planned_lowerings(stage_batch)
            plan_hashes = []
            fallback_reasons = []
            if plan is not None:
                stage_steps = tuple(
                    plan.steps[index]
                    for index in range(plan_cursor, min(plan_cursor + plan_count, len(plan.steps)))
                )
                plan_hashes = [
                    plan_steps[index]["plan_hash"]
                    for index in range(plan_cursor, min(plan_cursor + plan_count, len(plan_steps)))
                ]
                fallback_reasons = [
                    str(step.fallback_reason)
                    for step in stage_steps
                    if step.fallback_reason
                ]
                plan_cursor += plan_count
            items.append({
                "phase": "gemm_stage",
                "batch": int(batch_index),
                "stage": int(stage_index),
                "num_tasks": int(len(stage_batch.descs)),
                "num_groups": int(len(stage_batch.groups)),
                "group_sizes": [int(size) for size in stage_batch.group_sizes],
                "planned_lowerings": planned_lowerings,
                **_hmm_stage_execution_items(planned_lowerings, fallback_reasons=fallback_reasons),
                "planned_plan_counts": _hmm_stage_plan_counts(stage_batch),
                **_hmm_stage_task_block_profile_fields(task, batch_index, stage_index),
                "task_descriptors": _hmm_stage_task_descriptor_items(
                    task,
                    batch_index,
                    stage_index,
                    stage_batch,
                ),
                "gsta": [int(index) for index in stage_batch.gsta],
                "sorted_indices": [int(index) for index in stage_batch.sorted_indices],
                "group_keys": [
                    _serializable_group_key(key)
                    for key in stage_batch.groups
                ],
                "total_flops": int(stage_batch.total_flops),
                "max_m": int(stage_batch.max_m),
                "max_n": int(stage_batch.max_n),
                "max_k": int(stage_batch.max_k),
                "plan_hashes": plan_hashes,
            })

        reduce_batch = task.reduce_batches[batch_index]
        reduce_item = _hmm_gemv_trace_item("reduce_gemv", batch_index, reduce_batch, task=task, role="reduce")
        reduce_item.update(_hmm_batch_output_reduction_profile_fields(task, batch_index))
        items.append(reduce_item)
    return items


def _hmm_stage_execution_items(planned_lowerings, *, stage_steps=(), fallback_reasons=()):
    if stage_steps:
        fallback_reasons = [
            str(step.fallback_reason)
            for step in stage_steps
            if step.fallback_reason
        ]
    return {
        "execution_primitives": _unique_preserve_order(
            _hmm_execution_primitive(lowering) for lowering in planned_lowerings
        ),
        "execution_policies": _unique_preserve_order(
            _hmm_execution_policy(lowering) for lowering in planned_lowerings
        ),
        "fallback_reasons": _unique_preserve_order(fallback_reasons),
    }


def _hmm_gemv_trace_item(phase, batch_index, batch, *, task: HMMTask | None = None, role: str | None = None):
    return {
        "phase": str(phase),
        "batch": int(batch_index),
        "num_tasks": int(len(batch.descs)),
        "num_groups": int(len(batch.groups)),
        "group_sizes": [int(size) for size in batch.group_sizes],
        **_hmm_gemv_execution_items(batch),
        **_hmm_gemv_task_block_profile_fields(task, batch_index, role),
        "task_descriptors": _hmm_gemv_task_descriptor_items(
            task,
            batch_index,
            role,
            batch,
        ),
        "gsta": [int(index) for index in batch.gsta],
        "sorted_indices": [int(index) for index in batch.sorted_indices],
        "group_keys": [
            _serializable_group_key(key)
            for key in batch.groups
        ],
        "total_flops": int(batch.total_flops),
        "max_m": int(batch.max_m),
        "max_n": int(batch.max_n),
    }


def _hmm_gemv_execution_items(batch):
    if not batch.descs:
        return {
            "execution_primitives": [],
            "execution_policies": [],
            "fallback_reasons": [],
        }
    return {
        "execution_primitives": ["gemv"],
        "execution_policies": ["backend_gemv"],
        "fallback_reasons": [],
    }


def _hmm_gemv_task_block_profile_fields(task: HMMTask | None, batch_index, role: str | None):
    if task is None or role is None:
        return {}
    if role == "inter":
        attr = "gemv_inter"
    elif role == "reduce":
        attr = "gemv_reduce"
    else:
        return {}

    start = int(batch_index) * int(task.batch_size)
    stop = min(start + int(task.batch_size), len(task.hx_blocks))
    task_block_ids = []
    task_block_gemv_indices = []
    for block in task.hx_blocks[start:stop]:
        for gemv_index, _desc in enumerate(getattr(block, attr)):
            task_block_ids.append(int(block.block_id))
            task_block_gemv_indices.append(int(gemv_index))
    return {
        "task_block_ids": task_block_ids,
        "task_block_gemv_indices": task_block_gemv_indices,
    }


def _hmm_stage_task_descriptor_items(task: HMMTask | None, batch_index, stage_index, stage_batch):
    provenance = []
    if task is not None:
        provenance = _hmm_stage_task_provenance(task, batch_index, stage_index)
    items = []
    for sorted_index, desc in enumerate(stage_batch.descs):
        original_index = _original_task_index(stage_batch.sorted_indices, sorted_index)
        block_id, local_index = _provenance_item(provenance, original_index)
        item = {
            "task_index": int(sorted_index),
            "original_index": int(original_index),
            "block_id": block_id,
            "stage_task_index": local_index,
        }
        item.update(_hmm_gemm_descriptor_profile_item(desc))
        items.append(item)
    return items


def _hmm_gemv_task_descriptor_items(task: HMMTask | None, batch_index, role: str | None, batch):
    provenance = []
    if task is not None and role is not None:
        provenance = _hmm_gemv_task_provenance(task, batch_index, role)
    items = []
    for sorted_index, desc in enumerate(batch.descs):
        original_index = _original_task_index(batch.sorted_indices, sorted_index)
        block_id, local_index = _provenance_item(provenance, original_index)
        item = {
            "task_index": int(sorted_index),
            "original_index": int(original_index),
            "block_id": block_id,
            "gemv_index": local_index,
        }
        item.update(_hmm_gemv_descriptor_profile_item(desc))
        items.append(item)
    return items


def _original_task_index(sorted_indices, sorted_index):
    if 0 <= int(sorted_index) < len(sorted_indices):
        return int(sorted_indices[int(sorted_index)])
    return int(sorted_index)


def _provenance_item(provenance, original_index):
    if 0 <= int(original_index) < len(provenance):
        block_id, local_index = provenance[int(original_index)]
        return int(block_id), int(local_index)
    return None, None


def _dtype_profile_item(dtype):
    if dtype is None:
        return None
    return str(getattr(dtype, "name", dtype))


def _array_shape_profile_item(array):
    return [int(dim) for dim in getattr(array, "shape", ())]


def _hmm_gemm_descriptor_profile_item(desc):
    if isinstance(desc, MatmulDesc):
        return _hmm_buffer_matmul_desc_profile_item(desc)
    return _hmm_gemm_task_profile_item(desc)


def _hmm_buffer_matmul_desc_profile_item(desc: MatmulDesc):
    return {
        "descriptor_kind": "matmul_desc",
        "A_slice": _buffer_slice_profile_item(desc.A),
        "B_slice": _buffer_slice_profile_item(desc.B),
        "C_slice": _buffer_slice_profile_item(desc.C),
        "m": int(desc.m),
        "n": int(desc.n),
        "k": int(desc.k),
        "batch_shape": [],
        "trans_a": desc.trans_a,
        "trans_b": desc.trans_b,
        "conj_a": desc.trans_a == "C",
        "conj_b": desc.trans_b == "C",
        "lda": int(desc.lda),
        "ldb": int(desc.ldb),
        "ldc": int(desc.ldc),
        "alpha": _serializable_scalar(desc.alpha),
        "beta": _serializable_scalar(desc.beta),
        "dtype": _dtype_profile_item(desc.dtype),
        "tag": _metadata_profile_item(desc.tag),
    }


def _hmm_gemm_task_profile_item(desc):
    a_shape = tuple(int(dim) for dim in getattr(desc.A, "shape", ()))
    b_shape = tuple(int(dim) for dim in getattr(desc.B, "shape", ()))
    if len(a_shape) >= 2 and len(b_shape) >= 2:
        a_matrix_shape = _effective_shape(a_shape[-2:], trans=desc.trans_a)
        b_matrix_shape = _effective_shape(b_shape[-2:], trans=desc.trans_b)
        batch_shape = a_shape[:-2]
        m = int(a_matrix_shape[0])
        k = int(a_matrix_shape[1])
        n = int(b_matrix_shape[1])
    else:
        batch_shape = ()
        m = n = k = 0
    return {
        "descriptor_kind": "gemm_task",
        "a_shape": _array_shape_profile_item(desc.A),
        "b_shape": _array_shape_profile_item(desc.B),
        "c_shape": None if desc.C is None else _array_shape_profile_item(desc.C),
        "m": int(m),
        "n": int(n),
        "k": int(k),
        "batch_shape": [int(dim) for dim in batch_shape],
        "trans_a": bool(desc.trans_a),
        "trans_b": bool(desc.trans_b),
        "conj_a": bool(desc.conj_a),
        "conj_b": bool(desc.conj_b),
        "alpha": _serializable_scalar(desc.alpha),
        "beta": _serializable_scalar(desc.beta),
        "dtype_a": _dtype_profile_item(getattr(desc.A, "dtype", None)),
        "dtype_b": _dtype_profile_item(getattr(desc.B, "dtype", None)),
        "dtype_c": None if desc.C is None else _dtype_profile_item(getattr(desc.C, "dtype", None)),
        "tag": _metadata_profile_item(desc.tag),
    }


def _hmm_gemv_descriptor_profile_item(desc: GemvDesc):
    return {
        "descriptor_kind": "gemv_desc",
        "A_slice": _buffer_slice_profile_item(desc.A),
        "x_slice": _buffer_slice_profile_item(desc.x),
        "y_slice": _buffer_slice_profile_item(desc.y),
        "trans_a": desc.trans_a,
        "conj_a": desc.trans_a == "C",
        "m": int(desc.m),
        "n": int(desc.n),
        "lda": int(desc.lda),
        "incx": int(desc.incx),
        "incy": int(desc.incy),
        "alpha": _serializable_scalar(desc.alpha),
        "beta": _serializable_scalar(desc.beta),
        "dtype": _dtype_profile_item(desc.dtype),
        "tag": _metadata_profile_item(desc.tag),
    }


def _hmm_stage_task_block_profile_fields(task: HMMTask, batch_index, stage_index):
    start = int(batch_index) * int(task.batch_size)
    stop = min(start + int(task.batch_size), len(task.hx_blocks))
    task_block_ids = []
    task_block_stage_task_indices = []
    for block in task.hx_blocks[start:stop]:
        if int(stage_index) >= len(block.matmul_stages):
            continue
        for stage_task_index, _task in enumerate(block.matmul_stages[int(stage_index)]):
            task_block_ids.append(int(block.block_id))
            task_block_stage_task_indices.append(int(stage_task_index))
    return {
        "task_block_ids": task_block_ids,
        "task_block_stage_task_indices": task_block_stage_task_indices,
    }


def _hmm_batch_output_reduction_profile_fields(task: HMMTask, batch_index):
    start = int(batch_index) * int(task.batch_size)
    stop = min(start + int(task.batch_size), len(task.hx_blocks))
    batch_block_ids = {
        int(block.block_id)
        for block in task.hx_blocks[start:stop]
    }
    blocks_by_id = {
        int(block.block_id): block
        for block in task.hx_blocks
    }
    group_indices = []
    groups = []
    for index, group in enumerate(task.output_reduction_groups):
        if not any(int(block_id) in batch_block_ids for block_id in group.block_ids):
            continue
        group_indices.append(int(index))
        groups.append(_output_reduction_group_profile_item(
            group,
            task.output_contributions,
            blocks_by_id,
        ))
    return {
        "output_reduction_group_indices": group_indices,
        "output_reduction_groups": groups,
    }


def _hmm_stage_plan_count(stage_batch):
    count = 0
    for key, desc_indices in stage_batch.groups.items():
        if key.batch_shape:
            count += len(desc_indices)
        elif desc_indices:
            count += 1
    return int(count)


def _hmm_stage_planned_lowerings(stage_batch):
    return [
        _hmm_group_planned_lowering(key, desc_indices)
        for key, desc_indices in stage_batch.groups.items()
    ]


def _hmm_stage_plan_counts(stage_batch):
    return [
        _hmm_group_plan_count(key, desc_indices)
        for key, desc_indices in stage_batch.groups.items()
    ]


def _hmm_group_planned_lowering(key, desc_indices):
    if key.batch_shape:
        return "batched_gemm"
    if len(desc_indices) == 1:
        return "gemm"
    return "grouped_gemm"


def _hmm_group_plan_count(key, desc_indices):
    if key.batch_shape:
        return int(len(desc_indices))
    return 1 if desc_indices else 0


def _serializable_group_key(key):
    item = key.to_dict()
    if "batch_shape" in item:
        item["batch_shape"] = [int(dim) for dim in item["batch_shape"]]
    return item


def _final_step_output_shape(plan):
    if not plan.steps:
        return ()
    return tuple(getattr(plan.steps[-1].plan, "output_shape", ()))


def _task_dtype(task: HMMTask):
    for task_batch in task.batches:
        for stage_batch in task_batch:
            for desc in stage_batch.descs:
                if isinstance(desc, MatmulDesc):
                    if desc.dtype is not None:
                        return desc.dtype
                    continue
                dtype = _merged_dtype(getattr(desc.A, "dtype", None), getattr(desc.B, "dtype", None))
                if dtype is not None:
                    return dtype
    return None


def _stage_input_modes(batch_index, stage_index, plan_index):
    prefix = "hmm_b{0}_s{1}_p{2}".format(batch_index, stage_index, plan_index)
    return (
        (prefix + "_m", prefix + "_k"),
        (prefix + "_k", prefix + "_n"),
    )


def _stage_output_modes(batch_index, stage_index, plan_index, output_shape):
    prefix = "hmm_b{0}_s{1}_p{2}".format(batch_index, stage_index, plan_index)
    return tuple("{0}_out{1}".format(prefix, axis) for axis, _dim in enumerate(output_shape))


def _matmul_plan(**kwargs):
    from renormalizer.backend.execution import MatmulPlan

    return MatmulPlan(
        pre_ops=(),
        post_ops=(),
        estimated_time_s=None,
        **kwargs,
    )


def _gemm_task_to_matmul_desc(task):
    from renormalizer.backend.execution import MatmulDesc as ExecutionMatmulDesc

    if isinstance(task, MatmulDesc):
        return _buffer_slice_matmul_desc_to_execution_desc(task, ExecutionMatmulDesc)

    a_batch_shape, a_shape = _matrix_shape_and_batch_shape(task.A, name="A")
    b_batch_shape, b_shape = _matrix_shape_and_batch_shape(task.B, name="B")
    if a_batch_shape != b_batch_shape:
        raise ValueError("GEMM task batch dimensions must match")
    a_effective = _effective_shape(a_shape, trans=task.trans_a)
    b_effective = _effective_shape(b_shape, trans=task.trans_b)
    m, k_left = a_effective
    k_right, n = b_effective
    if k_left != k_right:
        raise ValueError("GEMM task has incompatible contracted dimensions")
    dtype_compute = _merged_dtype(getattr(task.A, "dtype", None), getattr(task.B, "dtype", None))
    dtype_output = _merged_dtype(dtype_compute, getattr(task.C, "dtype", None))
    batch_count = max(_prod(a_batch_shape), 1)
    estimated_flops = int(batch_count * 2 * m * n * k_left)
    estimated_read_bytes = int(array_nbytes(task.A) + array_nbytes(task.B))
    if task.C is not None:
        estimated_write_bytes = int(array_nbytes(task.C))
    else:
        estimated_write_bytes = int(batch_count * m * n * _dtype_itemsize(dtype_output))
    return ExecutionMatmulDesc(
        task.A,
        task.B,
        task.C,
        m=m,
        n=n,
        k=k_left,
        batch_shape=a_batch_shape,
        trans_a=task.trans_a,
        trans_b=task.trans_b,
        conj_a=task.conj_a,
        conj_b=task.conj_b,
        alpha=task.alpha,
        beta=task.beta,
        dtype_compute=dtype_compute,
        dtype_output=dtype_output,
        estimated_flops=estimated_flops,
        estimated_read_bytes=estimated_read_bytes,
        estimated_write_bytes=estimated_write_bytes,
        estimated_workspace_bytes=0,
    )


def _buffer_slice_matmul_desc_to_execution_desc(desc: MatmulDesc, execution_desc_cls):
    dtype_compute = desc.dtype
    dtype_output = desc.dtype
    itemsize = _dtype_itemsize(dtype_output)
    estimated_flops = int(2 * desc.m * desc.n * desc.k)
    estimated_read_bytes = int((_prod(desc.A.shape) + _prod(desc.B.shape)) * itemsize)
    estimated_write_bytes = int(_prod(desc.C.shape) * itemsize)
    return execution_desc_cls(
        _ShapeArray(desc.A.shape, dtype=dtype_compute),
        _ShapeArray(desc.B.shape, dtype=dtype_compute),
        _ShapeArray(desc.C.shape, dtype=dtype_output),
        m=desc.m,
        n=desc.n,
        k=desc.k,
        batch_shape=(),
        trans_a=desc.trans_a in ("T", "C"),
        trans_b=desc.trans_b in ("T", "C"),
        conj_a=desc.trans_a == "C",
        conj_b=desc.trans_b == "C",
        alpha=desc.alpha,
        beta=desc.beta,
        dtype_compute=dtype_compute,
        dtype_output=dtype_output,
        estimated_flops=estimated_flops,
        estimated_read_bytes=estimated_read_bytes,
        estimated_write_bytes=estimated_write_bytes,
        estimated_workspace_bytes=0,
    )


def _matrix_shape_and_batch_shape(array, *, name):
    shape = tuple(int(dim) for dim in getattr(array, "shape", ()))
    if len(shape) < 2:
        raise ValueError("GEMM task {0} must be at least rank-2".format(name))
    return shape[:-2], shape[-2:]


def _effective_shape(shape, *, trans):
    return (shape[1], shape[0]) if trans else shape


def _merged_dtype(left, right):
    if right is None:
        return left
    if left is None:
        return right
    if left == right:
        return left
    try:
        import numpy as np

        return np.result_type(left, right)
    except Exception:
        return left


def _layout_transform_copy_bytes(desc):
    total = 0
    for layout in (desc.layout_a, desc.layout_b, desc.layout_c):
        if layout is not None:
            total += int(layout.estimated_copy_bytes)
    return total


def build_single_site_hmm_scaffold(ltensor, mpo, rtensor, center, *, batch_size: int = 1):
    """Build a shape-aware HMMTask scaffold for ``abc,bdef,lfk,cek->adl``.

    This does not replace the current optimized-einsum H*v execution path.  It
    exposes the local contraction as representative GEMM stages so profiling can
    report task counts, shape groups, and workspace size before a real lowering
    is connected.
    """

    l_shape = tuple(int(dim) for dim in ltensor.shape)
    mpo_shape = tuple(int(dim) for dim in mpo.shape)
    r_shape = tuple(int(dim) for dim in rtensor.shape)
    c_shape = tuple(int(dim) for dim in center.shape)
    dtype = getattr(center, "dtype", None)
    return build_single_site_hmm_scaffold_from_shapes(
        l_shape,
        mpo_shape,
        r_shape,
        c_shape,
        dtype=dtype,
        batch_size=batch_size,
    )


def build_single_site_hmm_scaffold_from_shapes(
    ltensor_shape,
    mpo_shape,
    rtensor_shape,
    center_shape,
    *,
    dtype=None,
    batch_size: int = 1,
):
    """Build the single-site HMM scaffold from shape metadata only."""

    l_shape = tuple(int(dim) for dim in ltensor_shape)
    mpo_shape = tuple(int(dim) for dim in mpo_shape)
    r_shape = tuple(int(dim) for dim in rtensor_shape)
    c_shape = tuple(int(dim) for dim in center_shape)
    if len(l_shape) != 3:
        raise ValueError("ltensor for single-site HMM scaffold must have shape (a, b, c)")
    if len(mpo_shape) != 4:
        raise ValueError("mpo for single-site HMM scaffold must have shape (b, d, e, f)")
    if len(r_shape) != 3:
        raise ValueError("rtensor for single-site HMM scaffold must have shape (l, f, k)")
    if len(c_shape) != 3:
        raise ValueError("center for single-site HMM scaffold must have shape (c, e, k)")

    a, b_left, c_left = l_shape
    b_mpo, d, e_mpo, f_mpo = mpo_shape
    l_right, f_right, k_right = r_shape
    c_center, e_center, k_center = c_shape
    if b_left != b_mpo:
        raise ValueError("ltensor b dimension must match mpo b dimension")
    if c_left != c_center:
        raise ValueError("ltensor c dimension must match center c dimension")
    if e_mpo != e_center:
        raise ValueError("mpo e dimension must match center e dimension")
    if f_mpo != f_right:
        raise ValueError("mpo f dimension must match rtensor f dimension")
    if k_right != k_center:
        raise ValueError("rtensor k dimension must match center k dimension")

    center_array = _ShapeArray(c_shape, dtype=dtype)
    mpo_array = _ShapeArray(mpo_shape, dtype=dtype)
    ltensor_array = _ShapeArray(l_shape, dtype=dtype)
    stage0_rhs = _ShapeArray((k_center, l_right * f_right), dtype=dtype)
    stage1_rhs = _ShapeArray((e_center * f_mpo, c_center * l_right), dtype=dtype)
    stage2_rhs = _ShapeArray((b_left * c_center, d * l_right), dtype=dtype)
    stage0 = GemmTask(
        center_array.reshape(c_center * e_center, k_center),
        stage0_rhs,
        tag={
            "equation": "abc,bdef,lfk,cek->adl",
            "stage": 0,
            "role": "center_right",
            "output_shape": (c_center * e_center, l_right * f_right),
        },
    )
    stage1 = GemmTask(
        mpo_array.reshape(b_mpo * d, e_mpo * f_mpo),
        stage1_rhs,
        tag={
            "equation": "abc,bdef,lfk,cek->adl",
            "stage": 1,
            "role": "mpo_intermediate",
            "output_shape": (b_mpo * d, c_center * l_right),
        },
    )
    stage2 = GemmTask(
        ltensor_array.reshape(a, b_left * c_left),
        stage2_rhs,
        tag={
            "equation": "abc,bdef,lfk,cek->adl",
            "stage": 2,
            "role": "left_reduce",
            "output_shape": (a, d * l_right),
        },
    )
    output_shape = (a, d, l_right)
    inter_shape = (c_center * e_center, l_right * f_right)
    cost = float(
        2 * (c_center * e_center) * (l_right * f_right) * k_center
        + 2 * (b_mpo * d) * (c_center * l_right) * (e_mpo * f_mpo)
        + 2 * a * (d * l_right) * (b_left * c_left)
    )
    hxlist = HxBlockList.from_blocks(
        [
            HxBlock(
                block_id=0,
                input_slice=BufferSlice("X", offset=0, shape=c_shape),
                output_slice=BufferSlice("Y", offset=0, shape=output_shape),
                coefficient=1.0,
                inter_slice=BufferSlice("INTER", offset=0, shape=inter_shape),
                matmul_stages=((stage0,), (stage1,), (stage2,)),
                gemv_inter=(),
                gemv_reduce=(),
                qn_key=None,
                term_key=("single_site",),
                cost=cost,
            )
        ],
        center_kind="onedot",
        direct_intermediate=True,
        qn_adapted=False,
        equation="abc,bdef,lfk,cek->adl",
        operand_shapes=(l_shape, mpo_shape, r_shape, c_shape),
        output_shape=output_shape,
    )
    return hxlist, build_hmm_task(hxlist, batch_size=batch_size)


def build_batched_single_site_hmm_scaffold(
    ltensor,
    mpo,
    rtensor,
    center,
    nrhs,
    *,
    batch_size: int = 1,
):
    """Build an HMMTask scaffold for ``abc,bdef,lfk,cekr->adlr``."""

    return build_batched_single_site_hmm_scaffold_from_shapes(
        tuple(int(dim) for dim in ltensor.shape),
        tuple(int(dim) for dim in mpo.shape),
        tuple(int(dim) for dim in rtensor.shape),
        tuple(int(dim) for dim in center.shape),
        nrhs=nrhs,
        dtype=getattr(center, "dtype", None),
        batch_size=batch_size,
    )


def build_batched_single_site_hmm_scaffold_from_shapes(
    ltensor_shape,
    mpo_shape,
    rtensor_shape,
    center_shape,
    *,
    nrhs,
    dtype=None,
    batch_size: int = 1,
):
    """Build the single-site batched-RHS HMM scaffold from shape metadata."""

    nrhs = _validate_nrhs(nrhs)
    l_shape = tuple(int(dim) for dim in ltensor_shape)
    mpo_shape = tuple(int(dim) for dim in mpo_shape)
    r_shape = tuple(int(dim) for dim in rtensor_shape)
    c_shape = tuple(int(dim) for dim in center_shape)
    if len(l_shape) != 3:
        raise ValueError("ltensor for batched single-site HMM scaffold must have shape (a, b, c)")
    if len(mpo_shape) != 4:
        raise ValueError("mpo for batched single-site HMM scaffold must have shape (b, d, e, f)")
    if len(r_shape) != 3:
        raise ValueError("rtensor for batched single-site HMM scaffold must have shape (l, f, k)")
    if len(c_shape) != 3:
        raise ValueError("center for batched single-site HMM scaffold must have shape (c, e, k)")

    a, b_left, c_left = l_shape
    b_mpo, d, e_mpo, f_mpo = mpo_shape
    l_right, f_right, k_right = r_shape
    c_center, e_center, k_center = c_shape
    if b_left != b_mpo:
        raise ValueError("ltensor b dimension must match mpo b dimension")
    if c_left != c_center:
        raise ValueError("ltensor c dimension must match center c dimension")
    if e_mpo != e_center:
        raise ValueError("mpo e dimension must match center e dimension")
    if f_mpo != f_right:
        raise ValueError("mpo f dimension must match rtensor f dimension")
    if k_right != k_center:
        raise ValueError("rtensor k dimension must match center k dimension")

    center_array = _ShapeArray((nrhs, c_center * e_center, k_center), dtype=dtype)
    stage0_rhs = _ShapeArray((nrhs, k_center, l_right * f_right), dtype=dtype)
    stage1_left = _ShapeArray((nrhs, b_mpo * d, e_mpo * f_mpo), dtype=dtype)
    stage1_rhs = _ShapeArray((nrhs, e_center * f_mpo, c_center * l_right), dtype=dtype)
    stage2_left = _ShapeArray((nrhs, a, b_left * c_left), dtype=dtype)
    stage2_rhs = _ShapeArray((nrhs, b_left * c_center, d * l_right), dtype=dtype)
    equation = "abc,bdef,lfk,cekr->adlr"
    stage0 = GemmTask(
        center_array,
        stage0_rhs,
        tag={
            "equation": equation,
            "stage": 0,
            "role": "center_right_batched_rhs",
            "output_shape": (nrhs, c_center * e_center, l_right * f_right),
        },
    )
    stage1 = GemmTask(
        stage1_left,
        stage1_rhs,
        tag={
            "equation": equation,
            "stage": 1,
            "role": "mpo_intermediate_batched_rhs",
            "output_shape": (nrhs, b_mpo * d, c_center * l_right),
        },
    )
    stage2 = GemmTask(
        stage2_left,
        stage2_rhs,
        tag={
            "equation": equation,
            "stage": 2,
            "role": "left_reduce_batched_rhs",
            "output_shape": (nrhs, a, d * l_right),
        },
    )
    output_shape = (a, d, l_right, nrhs)
    batched_center_shape = c_shape + (nrhs,)
    inter_shape = (nrhs, c_center * e_center, l_right * f_right)
    cost = float(
        nrhs * (
            2 * (c_center * e_center) * (l_right * f_right) * k_center
            + 2 * (b_mpo * d) * (c_center * l_right) * (e_mpo * f_mpo)
            + 2 * a * (d * l_right) * (b_left * c_left)
        )
    )
    hxlist = HxBlockList.from_blocks(
        [
            HxBlock(
                block_id=0,
                input_slice=BufferSlice("X", offset=0, shape=batched_center_shape),
                output_slice=BufferSlice("Y", offset=0, shape=output_shape),
                coefficient=1.0,
                inter_slice=BufferSlice("INTER", offset=0, shape=inter_shape),
                matmul_stages=((stage0,), (stage1,), (stage2,)),
                gemv_inter=(),
                gemv_reduce=(),
                qn_key=None,
                term_key=("single_site", "batched_rhs"),
                cost=cost,
            )
        ],
        center_kind="onedot",
        direct_intermediate=True,
        qn_adapted=False,
        equation=equation,
        operand_shapes=(l_shape, mpo_shape, r_shape, batched_center_shape),
        output_shape=output_shape,
    )
    return hxlist, build_hmm_task(hxlist, batch_size=batch_size)


def _validate_single_site_hmm_shapes(ltensor, mpo, rtensor, center):
    l_shape = _shape_tuple(ltensor)
    mpo_shape = _shape_tuple(mpo)
    r_shape = _shape_tuple(rtensor)
    c_shape = _shape_tuple(center)
    if len(l_shape) != 3:
        raise ValueError("ltensor for single-site HMM action must have shape (a, b, c)")
    if len(mpo_shape) != 4:
        raise ValueError("mpo for single-site HMM action must have shape (b, d, e, f)")
    if len(r_shape) != 3:
        raise ValueError("rtensor for single-site HMM action must have shape (l, f, k)")
    if len(c_shape) not in (3, 4):
        raise ValueError("center for single-site HMM action must have shape (c, e, k) or (c, e, k, nrhs)")

    a, b_left, c_left = l_shape
    b_mpo, d, e_mpo, f_mpo = mpo_shape
    l_right, f_right, k_right = r_shape
    c_center, e_center, k_center = c_shape[:3]
    if b_left != b_mpo:
        raise ValueError("ltensor b dimension must match mpo b dimension")
    if c_left != c_center:
        raise ValueError("ltensor c dimension must match center c dimension")
    if e_mpo != e_center:
        raise ValueError("mpo e dimension must match center e dimension")
    if f_mpo != f_right:
        raise ValueError("mpo f dimension must match rtensor f dimension")
    if k_right != k_center:
        raise ValueError("rtensor k dimension must match center k dimension")
    if len(c_shape) == 4 and c_shape[3] <= 0:
        raise ValueError("single-site HMM action nrhs dimension must be positive")

    return {
        "a": a,
        "b": b_left,
        "c": c_center,
        "d": d,
        "e": e_center,
        "f": f_mpo,
        "l": l_right,
        "k": k_center,
        "nrhs": c_shape[3] if len(c_shape) == 4 else None,
    }


def _execute_hmm_matmul_plan(
    backend,
    A,
    B,
    *,
    reason,
    stream=None,
    workspace=None,
):
    from renormalizer.backend.execution import LayoutSpec, MatmulDesc, MatmulPlan

    xp = _backend_xp(backend)
    a_shape = _shape_tuple(A)
    b_shape = _shape_tuple(B)
    if len(a_shape) < 2 or len(b_shape) < 2:
        raise ValueError("HMM GEMM stage operands must be at least rank-2")

    if len(a_shape) == 2 and len(b_shape) == 2:
        kind = "gemm"
        batch_shape = ()
    else:
        kind = "batched_gemm"
        a_batch = a_shape[:-2]
        b_batch = b_shape[:-2]
        if a_batch and b_batch and a_batch != b_batch:
            raise ValueError("HMM batched GEMM operands must have matching batch shape")
        batch_shape = a_batch or b_batch
        if not batch_shape:
            raise ValueError("HMM batched GEMM stage requires a batch shape")
        if not a_batch:
            A = _broadcast_to(xp, A, batch_shape + a_shape)
            a_shape = _shape_tuple(A)
        if not b_batch:
            B = _broadcast_to(xp, B, batch_shape + b_shape)
            b_shape = _shape_tuple(B)

    m = int(a_shape[-2])
    k = int(a_shape[-1])
    if int(b_shape[-2]) != k:
        raise ValueError("HMM GEMM stage contracted dimensions do not match")
    n = int(b_shape[-1])
    output_shape = tuple(batch_shape) + (m, n)
    if batch_shape:
        left_modes = ("batch", "m", "k")
        right_modes = ("batch", "k", "n")
        output_modes = ("batch", "m", "n")
    else:
        left_modes = ("m", "k")
        right_modes = ("k", "n")
        output_modes = ("m", "n")
    left_layout = LayoutSpec(
        logical_shape=a_shape,
        physical_shape=a_shape,
        logical_modes=left_modes,
        strides=getattr(A, "strides", None),
        order="C",
        contiguous_groups=(tuple(range(len(a_shape))),),
    )
    right_layout = LayoutSpec(
        logical_shape=b_shape,
        physical_shape=b_shape,
        logical_modes=right_modes,
        strides=getattr(B, "strides", None),
        order="C",
        contiguous_groups=(tuple(range(len(b_shape))),),
    )
    output_layout = LayoutSpec(
        logical_shape=output_shape,
        physical_shape=output_shape,
        logical_modes=output_modes,
        strides=None,
        order="C",
        contiguous_groups=(tuple(range(len(output_shape))),),
    )
    itemsize = max(_dtype_itemsize(getattr(A, "dtype", None)), _dtype_itemsize(getattr(B, "dtype", None)))
    write_bytes = _prod(output_shape) * itemsize
    desc = MatmulDesc(
        A,
        B,
        None,
        m,
        n,
        k,
        batch_shape=batch_shape,
        layout_a=left_layout,
        layout_b=right_layout,
        layout_c=output_layout,
        estimated_flops=2 * _prod(batch_shape or (1,)) * m * n * k,
        estimated_read_bytes=array_nbytes(A) + array_nbytes(B),
        estimated_write_bytes=write_bytes,
        estimated_workspace_bytes=0,
    )
    planned_kind = kind
    fallback_reason = None
    if kind == "gemm" and not bool(getattr(backend, "supports_matmul", False)):
        fallback_reason = "backend lacks matmul"
        kind = "fallback_tensordot"
    elif kind == "batched_gemm" and not bool(getattr(backend, "supports_batched_matmul", False)):
        fallback_reason = "backend lacks batched_matmul for batch shape {0}".format(tuple(batch_shape))
        kind = "fallback_tensordot" if bool(getattr(backend, "supports_matmul", False)) else "fallback_einsum"

    plan = MatmulPlan(
        kind=kind,
        descs=(desc,),
        pre_ops=(),
        post_ops=(),
        output_shape=output_shape,
        copy_bytes=0,
        workspace_bytes=0,
        estimated_flops=desc.estimated_flops,
        estimated_time_s=None,
        reason=reason,
        fallback_reason=fallback_reason,
    )
    result = backend.execute_matmul_plan(
        plan,
        stream=stream,
        workspace=workspace,
        record_profile=False,
    )
    _append_hmm_runtime_stage_profile(backend, planned_kind, plan)
    return result


def _append_hmm_runtime_stage_profile(backend, planned_kind, plan):
    profiles = getattr(backend, "_hmm_active_stage_profiles", None)
    if profiles is None:
        return
    try:
        execution_items = backend._matmul_plan_execution_items(plan)
        fallback_from = backend._fallback_source(plan)
        fallback_to = backend._fallback_target(plan)
        fallback_reason = getattr(plan, "fallback_reason", None)
        if fallback_reason is not None and fallback_to is not None:
            execution_items = dict(execution_items)
            if fallback_to == "loop_matmul":
                execution_items["execution_primitives"] = ["matmul"]
                execution_items["execution_policies"] = ["loop_matmul"]
            else:
                execution_items["execution_primitives"] = [fallback_to]
                execution_items["execution_policies"] = [fallback_to]
            execution_items["fallback_reasons"] = [str(fallback_reason)]
        fallback_policy = None
        if fallback_reason is not None:
            fallback_policy = backend.fallback_policy.value
        profile = {
            "planned_kind": str(planned_kind),
            "actual_kind": str(plan.kind),
            "actual_lowerings": [str(plan.kind)],
            "fallback_reason": None if fallback_reason is None else str(fallback_reason),
            "fallback_from": fallback_from,
            "fallback_to": fallback_to,
            "fallback_policy": fallback_policy,
            **execution_items,
        }
        profile.update(_hmm_runtime_stage_count_fields(profile))
        profiles.append(profile)
    except Exception:
        return


def _first_nonempty(values):
    for value in values:
        if value:
            return value
    return None


_HMM_RUNTIME_COUNT_FIELDS = (
    "actual_num_gemm",
    "actual_num_batched_gemm",
    "actual_num_grouped_gemm",
    "actual_num_loop_matmul",
    "actual_num_tensordot",
    "actual_num_einsum",
)


def _hmm_runtime_stage_count_fields(profile):
    actual_kind = str(profile.get("actual_kind") or "")
    primitives = set(str(item) for item in profile.get("execution_primitives", ()) if item)
    policies = set(str(item) for item in profile.get("execution_policies", ()) if item)
    fallback_reason = profile.get("fallback_reason")
    counts = {field: 0 for field in _HMM_RUNTIME_COUNT_FIELDS}
    if fallback_reason:
        if "loop_matmul" in policies:
            counts["actual_num_loop_matmul"] = 1
        elif "tensordot" in primitives or "tensordot" in policies:
            counts["actual_num_tensordot"] = 1
        elif "einsum" in primitives or "einsum" in policies:
            counts["actual_num_einsum"] = 1
        return counts

    if actual_kind == "gemm":
        counts["actual_num_gemm"] = 1
    elif actual_kind in ("batched_gemm", "strided_batched_gemm"):
        counts["actual_num_batched_gemm"] = 1
    elif actual_kind == "grouped_gemm":
        counts["actual_num_grouped_gemm"] = 1
    elif "tensordot" in primitives or "tensordot" in policies:
        counts["actual_num_tensordot"] = 1
    elif "einsum" in primitives or "einsum" in policies:
        counts["actual_num_einsum"] = 1
    return counts


def _sum_hmm_runtime_count_field(stage_profiles, field):
    return int(sum(int(profile.get(field, 0) or 0) for profile in stage_profiles))


def _hmm_runtime_fallback_summary(stage_profiles):
    stage_profiles = tuple(stage_profiles or ())
    fallback_reasons = _unique_preserve_order(
        profile.get("fallback_reason")
        for profile in stage_profiles
        if profile.get("fallback_reason")
    )
    fallback_sources = _unique_preserve_order(
        item
        for profile in stage_profiles
        for item in (profile.get("fallback_sources") or ((profile.get("fallback_from"),) if profile.get("fallback_from") else ()))
        if item
    )
    fallback_targets = _unique_preserve_order(
        item
        for profile in stage_profiles
        for item in (profile.get("fallback_targets") or ((profile.get("fallback_to"),) if profile.get("fallback_to") else ()))
        if item
    )
    fallback_policies = _unique_preserve_order(
        item
        for profile in stage_profiles
        for item in (profile.get("fallback_policies") or ((profile.get("fallback_policy"),) if profile.get("fallback_policy") else ()))
        if item
    )
    grouped_gemm_policies = _unique_preserve_order(
        item
        for profile in stage_profiles
        for item in (profile.get("grouped_gemm_policies") or ((profile.get("grouped_gemm_policy"),) if profile.get("grouped_gemm_policy") else ()))
        if item
    )
    grouped_gemm_implementations = _unique_preserve_order(
        item
        for profile in stage_profiles
        for item in (
            profile.get("grouped_gemm_implementations")
            or ((profile.get("grouped_gemm_implementation"),) if profile.get("grouped_gemm_implementation") else ())
        )
        if item
    )
    fallback_reason = fallback_reasons[0] if fallback_reasons else None
    return {
        "actual_lowerings": _unique_preserve_order(
            lowering
            for profile in stage_profiles
            for lowering in profile.get("actual_lowerings", ())
        ),
        "execution_primitives": _unique_preserve_order(
            primitive
            for profile in stage_profiles
            for primitive in profile.get("execution_primitives", ())
        ),
        "execution_policies": _unique_preserve_order(
            policy
            for profile in stage_profiles
            for policy in profile.get("execution_policies", ())
        ),
        "fallback_reasons": fallback_reasons,
        "fallback_reason": fallback_reason,
        "fallback_from": _first_nonempty(profile.get("fallback_from") for profile in stage_profiles),
        "fallback_to": _first_nonempty(profile.get("fallback_to") for profile in stage_profiles),
        "fallback_policy": _first_nonempty(profile.get("fallback_policy") for profile in stage_profiles),
        "fallback_sources": fallback_sources,
        "fallback_targets": fallback_targets,
        "fallback_policies": fallback_policies,
        "grouped_gemm_policies": grouped_gemm_policies,
        "grouped_gemm_implementations": grouped_gemm_implementations,
        "requires_grouped_gemm_fallback": any(
            bool(profile.get("requires_grouped_gemm_fallback"))
            for profile in stage_profiles
        ),
        **{
            field: _sum_hmm_runtime_count_field(stage_profiles, field)
            for field in _HMM_RUNTIME_COUNT_FIELDS
        },
    }


def _apply_hmm_runtime_stage_profiles(hmm_batches, execution_trace, stage_profiles):
    stage_profiles = list(stage_profiles or ())
    if not stage_profiles:
        return hmm_batches, execution_trace

    profile_index = 0
    merged_batches = []
    for batch in hmm_batches:
        batch = dict(batch)
        stages = []
        for stage in batch.get("stages", ()):
            stage = dict(stage)
            if profile_index < len(stage_profiles):
                profile = stage_profiles[profile_index]
                for key in (
                    "actual_lowerings",
                    "execution_primitives",
                    "execution_policies",
                    "fallback_reasons",
                    "fallback_sources",
                    "fallback_targets",
                    "fallback_policies",
                    "grouped_gemm_policies",
                    "grouped_gemm_implementations",
                ):
                    if key in profile:
                        stage[key] = list(profile[key])
                if "fallback_sources" not in stage and profile.get("fallback_from"):
                    stage["fallback_sources"] = [profile["fallback_from"]]
                if "fallback_targets" not in stage and profile.get("fallback_to"):
                    stage["fallback_targets"] = [profile["fallback_to"]]
                if "fallback_policies" not in stage and profile.get("fallback_policy"):
                    stage["fallback_policies"] = [profile["fallback_policy"]]
                for key in ("fallback_reason", "fallback_from", "fallback_to", "fallback_policy"):
                    if key in profile:
                        stage[key] = profile[key]
                if "requires_grouped_gemm_fallback" in profile:
                    stage["requires_grouped_gemm_fallback"] = bool(profile["requires_grouped_gemm_fallback"])
                for key in _HMM_RUNTIME_COUNT_FIELDS:
                    stage[key] = int(profile.get(key, 0) or 0)
                profile_index += 1
            stages.append(stage)
        batch["stages"] = stages
        merged_batches.append(batch)

    profile_index = 0
    merged_trace = []
    for item in execution_trace:
        item = dict(item)
        if item.get("phase") == "gemm_stage" and profile_index < len(stage_profiles):
            profile = stage_profiles[profile_index]
            for key in (
                "actual_lowerings",
                "execution_primitives",
                "execution_policies",
                "fallback_reasons",
                "fallback_sources",
                "fallback_targets",
                "fallback_policies",
                "grouped_gemm_policies",
                "grouped_gemm_implementations",
            ):
                if key in profile:
                    item[key] = list(profile[key])
            if "fallback_sources" not in item and profile.get("fallback_from"):
                item["fallback_sources"] = [profile["fallback_from"]]
            if "fallback_targets" not in item and profile.get("fallback_to"):
                item["fallback_targets"] = [profile["fallback_to"]]
            if "fallback_policies" not in item and profile.get("fallback_policy"):
                item["fallback_policies"] = [profile["fallback_policy"]]
            for key in ("fallback_reason", "fallback_from", "fallback_to", "fallback_policy"):
                if key in profile:
                    item[key] = profile[key]
            if "requires_grouped_gemm_fallback" in profile:
                item["requires_grouped_gemm_fallback"] = bool(profile["requires_grouped_gemm_fallback"])
            for key in _HMM_RUNTIME_COUNT_FIELDS:
                item[key] = int(profile.get(key, 0) or 0)
            profile_index += 1
        merged_trace.append(item)
    return merged_batches, merged_trace


def _backend_execution_device_payload(backend):
    try:
        current_device = backend.current_device()
    except Exception:
        current_device = None
    try:
        return backend._profile_device_execution(current_device)
    except Exception:
        try:
            from renormalizer.utils import profiling

            return profiling.device_execution_payload(current_device)
        except Exception:
            return {
                "device_kind": getattr(current_device, "kind", None),
                "device_index": getattr(current_device, "index", None),
                "device_local_rank": getattr(current_device, "local_rank", None),
                "device_global_rank": getattr(current_device, "global_rank", None),
            }


def _backend_execution_resource_payload(backend, *, stream=None, workspace=None):
    try:
        return backend._profile_execution_resources(stream=stream, workspace=workspace)
    except Exception:
        workspace_device = getattr(workspace, "device", None)
        return {
            "stream_provided": stream is not None,
            "stream_type": type(stream).__name__ if stream is not None else None,
            "workspace_provided": workspace is not None,
            "workspace_nbytes": int(getattr(workspace, "nbytes", 0) or 0) if workspace is not None else None,
            "workspace_released": bool(getattr(workspace, "released", False)) if workspace is not None else None,
            "workspace_device_kind": getattr(workspace_device, "kind", None),
            "workspace_device_index": getattr(workspace_device, "index", None),
            "workspace_device_local_rank": getattr(workspace_device, "local_rank", None),
            "workspace_device_global_rank": getattr(workspace_device, "global_rank", None),
        }


def _record_hmm_action_execution_profile(
    backend,
    task: HMMTask,
    result,
    wall_s,
    *,
    stream=None,
    workspace=None,
    stage_profiles=(),
):
    try:
        from renormalizer.utils import profiling

        plan = lower_hmm_task_to_contraction_plan(task)
        dtype = _task_dtype(task)
        dtype_name = None if dtype is None else str(dtype)
        itemsize = _dtype_itemsize(dtype)
        workspace_bytes = max(
            int(plan.required_workspace_bytes),
            int(task.workspace_size) * itemsize,
        )
        largest_intermediate_elements = max(
            int(task.inter_size),
            int(getattr(result, "size", 0) or 0),
        )
        largest_intermediate = max(
            int(task.inter_size) * itemsize,
            int(array_nbytes(result)),
        )
        shape_buckets = _hmm_shape_bucket_profile_items(task)
        gemv_shape_buckets = _hmm_gemv_shape_bucket_profile_items(task)
        step_lowerings = [str(step.kind) for step in plan.steps]
        hmm_steps = _hmm_step_profile_items(plan)
        hmm_batches, hmm_execution_trace = _apply_hmm_runtime_stage_profiles(
            _hmm_batch_profile_items(task, plan=plan),
            _hmm_execution_trace_items(task, plan=plan),
            stage_profiles,
        )
        runtime_summary = _hmm_runtime_fallback_summary(stage_profiles)
        num_grouped_tasks = sum(
            len(getattr(step.plan, "descs", ()) or ())
            for step in plan.steps
            if step.kind == "grouped_gemm"
        )
        num_gemv_desc = sum(len(batch.descs) for batch in task.inter_batches)
        num_gemv_desc += sum(len(batch.descs) for batch in task.reduce_batches)
        planned_num_gemm = sum(1 for lowering in step_lowerings if lowering == "gemm")
        planned_num_batched_gemm = sum(
            1
            for lowering in step_lowerings
            if lowering in ("batched_gemm", "strided_batched_gemm")
        )
        fallback_reason = (
            runtime_summary["fallback_reason"]
            or next((step.fallback_reason for step in plan.steps if step.fallback_reason), None)
        )
        payload = {
            "event": "contraction_execute",
            "backend": getattr(backend, "name", type(backend).__name__),
            **profiling.contraction_execute_compute_payload("hmm_task"),
            "equation": task.equation,
            "lowering": "hmm_task",
            "center_kind": task.center_kind,
            "hmm_center_kind": task.center_kind,
            "direct_intermediate": bool(task.direct_intermediate),
            "step_lowerings": step_lowerings,
            "step_count": len(plan.steps),
            "plan_hash": plan.plan_hash,
            "matmul_plan_hashes": [step["plan_hash"] for step in hmm_steps],
            "input_modes": [
                [str(mode) for mode in operand.modes]
                for operand in plan.input_specs
            ],
            "output_modes": [str(mode) for mode in plan.output_modes],
            "input_shapes": [
                tuple(getattr(operand.array, "shape", ()))
                for operand in plan.input_specs
            ],
            "output_shape": tuple(getattr(result, "shape", task.output_shape)),
            "input_dtypes": [dtype_name for _operand in plan.input_specs],
            "dtype": dtype_name or str(getattr(result, "dtype", None)),
            **_backend_execution_device_payload(backend),
            **_backend_execution_resource_payload(backend, stream=stream, workspace=workspace),
            "flops": int(plan.estimated_flops),
            "read_bytes": int(plan.estimated_read_bytes),
            "write_bytes": int(plan.estimated_write_bytes),
            "copy_bytes": int(plan.estimated_copy_bytes),
            "workspace_bytes": workspace_bytes,
            "peak_bytes": max(
                int(plan.estimated_peak_bytes),
                workspace_bytes,
                largest_intermediate,
            ),
            "largest_intermediate": largest_intermediate,
            "largest_intermediate_elements": largest_intermediate_elements,
            "largest_intermediate_bytes": largest_intermediate,
            "num_gemm": planned_num_gemm,
            "num_batched_gemm": planned_num_batched_gemm,
            "planned_num_gemm": planned_num_gemm,
            "planned_num_batched_gemm": planned_num_batched_gemm,
            "planned_num_grouped_tasks": int(num_grouped_tasks),
            "actual_lowerings": runtime_summary["actual_lowerings"],
            **{
                field: int(runtime_summary[field])
                for field in _HMM_RUNTIME_COUNT_FIELDS
            },
            "num_grouped_tasks": int(num_grouped_tasks),
            "num_blocks": int(task.num_hx_blocks),
            "num_hx_blocks": int(task.num_hx_blocks),
            "num_batches": int(task.num_batches),
            "batch_size": int(task.batch_size),
            "num_shape_buckets": len(shape_buckets),
            "num_gemv_shape_buckets": len(gemv_shape_buckets),
            "num_total_shape_buckets": len(shape_buckets) + len(gemv_shape_buckets),
            "num_gemv_desc": int(num_gemv_desc),
            "hmm_steps": hmm_steps,
            "shape_buckets": shape_buckets,
            **hmm_task_execution_metadata(plan, task),
            "fallback_reason": fallback_reason,
            "fallback_from": runtime_summary["fallback_from"],
            "fallback_to": runtime_summary["fallback_to"],
            "fallback_policy": runtime_summary["fallback_policy"],
            "fallback_reasons": runtime_summary["fallback_reasons"],
            "fallback_sources": runtime_summary["fallback_sources"],
            "fallback_targets": runtime_summary["fallback_targets"],
            "fallback_policies": runtime_summary["fallback_policies"],
            "grouped_gemm_policies": runtime_summary["grouped_gemm_policies"],
            "grouped_gemm_implementations": runtime_summary["grouped_gemm_implementations"],
            "requires_grouped_gemm_fallback": runtime_summary["requires_grouped_gemm_fallback"],
            "execution_primitives": runtime_summary["execution_primitives"],
            "execution_policies": runtime_summary["execution_policies"],
            "hmm_batches": hmm_batches,
            "hmm_execution_trace": hmm_execution_trace,
            "wall_s": float(wall_s),
        }
        setattr(backend, "_last_execution_profile", dict(payload))
        if profiling.should_record_op():
            record_payload = dict(payload)
            record_payload.pop("event")
            profiling.record("contraction_execute", **record_payload)
    except Exception:
        return


def _single_site_hmm_task_for_action(ltensor, mpo, rtensor, center, dims):
    dtype = getattr(center, "dtype", None)
    if dims["nrhs"] is None:
        _hxlist, task = build_single_site_hmm_scaffold_from_shapes(
            _shape_tuple(ltensor),
            _shape_tuple(mpo),
            _shape_tuple(rtensor),
            _shape_tuple(center),
            dtype=dtype,
            batch_size=1,
        )
        return task
    _hxlist, task = build_batched_single_site_hmm_scaffold_from_shapes(
        _shape_tuple(ltensor),
        _shape_tuple(mpo),
        _shape_tuple(rtensor),
        _shape_tuple(center)[:3],
        nrhs=dims["nrhs"],
        dtype=dtype,
        batch_size=1,
    )
    return task


def execute_single_site_hmm_action(
    backend,
    ltensor,
    mpo,
    rtensor,
    center,
    *,
    stream=None,
    workspace=None,
):
    """Execute ``abc,bdef,lfk,cek->adl`` or batched ``cekr->adlr`` via backend GEMM.

    This is the first executable HMM hot-path entrypoint.  It mirrors the
    scaffold stages so tests and profiling can distinguish scalar GEMM from
    batched-RHS GEMM execution.
    """

    import time

    dims = _validate_single_site_hmm_shapes(ltensor, mpo, rtensor, center)
    profile_task = _single_site_hmm_task_for_action(ltensor, mpo, rtensor, center, dims)
    started = time.perf_counter()
    stage_profiles = []
    previous_stage_profiles = getattr(backend, "_hmm_active_stage_profiles", None)
    setattr(backend, "_hmm_active_stage_profiles", stage_profiles)
    try:
        if dims["nrhs"] is None:
            result = _execute_single_site_hmm_action_scalar(
                backend,
                ltensor,
                mpo,
                rtensor,
                center,
                dims,
                stream=stream,
                workspace=workspace,
            )
        else:
            result = _execute_single_site_hmm_action_batched(
                backend,
                ltensor,
                mpo,
                rtensor,
                center,
                dims,
                stream=stream,
                workspace=workspace,
            )
    finally:
        if previous_stage_profiles is None:
            try:
                delattr(backend, "_hmm_active_stage_profiles")
            except AttributeError:
                pass
        else:
            setattr(backend, "_hmm_active_stage_profiles", previous_stage_profiles)
    _record_hmm_action_execution_profile(
        backend,
        profile_task,
        result,
        time.perf_counter() - started,
        stream=stream,
        workspace=workspace,
        stage_profiles=stage_profiles,
    )
    return result


def _execute_single_site_hmm_action_scalar(
    backend,
    ltensor,
    mpo,
    rtensor,
    center,
    dims,
    *,
    stream=None,
    workspace=None,
):
    xp = _backend_xp(backend)
    a = dims["a"]
    b = dims["b"]
    c = dims["c"]
    d = dims["d"]
    e = dims["e"]
    f = dims["f"]
    l_right = dims["l"]
    k = dims["k"]

    right0 = _reshape(xp, _transpose(xp, rtensor, (2, 0, 1)), (k, l_right * f))
    tmp0 = _execute_hmm_matmul_plan(
        backend,
        _reshape(xp, center, (c * e, k)),
        right0,
        reason="single_site_hmm_stage_0",
        stream=stream,
        workspace=workspace,
    )
    tmp0 = _reshape(xp, tmp0, (c, e, l_right, f))

    right1 = _reshape(xp, _transpose(xp, tmp0, (1, 3, 0, 2)), (e * f, c * l_right))
    tmp1 = _execute_hmm_matmul_plan(
        backend,
        _reshape(xp, mpo, (b * d, e * f)),
        right1,
        reason="single_site_hmm_stage_1",
        stream=stream,
        workspace=workspace,
    )
    tmp1 = _reshape(xp, tmp1, (b, d, c, l_right))

    right2 = _reshape(xp, _transpose(xp, tmp1, (0, 2, 1, 3)), (b * c, d * l_right))
    result = _execute_hmm_matmul_plan(
        backend,
        _reshape(xp, ltensor, (a, b * c)),
        right2,
        reason="single_site_hmm_stage_2",
        stream=stream,
        workspace=workspace,
    )
    return _reshape(xp, result, (a, d, l_right))


def _execute_single_site_hmm_action_batched(
    backend,
    ltensor,
    mpo,
    rtensor,
    center,
    dims,
    *,
    stream=None,
    workspace=None,
):
    xp = _backend_xp(backend)
    a = dims["a"]
    b = dims["b"]
    c = dims["c"]
    d = dims["d"]
    e = dims["e"]
    f = dims["f"]
    l_right = dims["l"]
    k = dims["k"]
    nrhs = dims["nrhs"]

    center_batch = _reshape(xp, _transpose(xp, center, (3, 0, 1, 2)), (nrhs, c * e, k))
    right0 = _reshape(xp, _transpose(xp, rtensor, (2, 0, 1)), (k, l_right * f))
    tmp0 = _execute_hmm_matmul_plan(
        backend,
        center_batch,
        right0,
        reason="single_site_hmm_batched_rhs_stage_0",
        stream=stream,
        workspace=workspace,
    )
    tmp0 = _reshape(xp, tmp0, (nrhs, c, e, l_right, f))

    right1 = _reshape(xp, _transpose(xp, tmp0, (0, 2, 4, 1, 3)), (nrhs, e * f, c * l_right))
    tmp1 = _execute_hmm_matmul_plan(
        backend,
        _reshape(xp, mpo, (b * d, e * f)),
        right1,
        reason="single_site_hmm_batched_rhs_stage_1",
        stream=stream,
        workspace=workspace,
    )
    tmp1 = _reshape(xp, tmp1, (nrhs, b, d, c, l_right))

    right2 = _reshape(xp, _transpose(xp, tmp1, (0, 1, 3, 2, 4)), (nrhs, b * c, d * l_right))
    result = _execute_hmm_matmul_plan(
        backend,
        _reshape(xp, ltensor, (a, b * c)),
        right2,
        reason="single_site_hmm_batched_rhs_stage_2",
        stream=stream,
        workspace=workspace,
    )
    result = _reshape(xp, result, (nrhs, a, d, l_right))
    return _transpose(xp, result, (1, 2, 3, 0))


def build_two_site_hmm_scaffold(ltensor, mpo0, mpo1, rtensor, center, *, batch_size: int = 1):
    """Build a shape-aware HMMTask scaffold for ``abc,bdef,fghj,ljk,cehk->adgl``."""

    return build_two_site_hmm_scaffold_from_shapes(
        tuple(int(dim) for dim in ltensor.shape),
        tuple(int(dim) for dim in mpo0.shape),
        tuple(int(dim) for dim in mpo1.shape),
        tuple(int(dim) for dim in rtensor.shape),
        tuple(int(dim) for dim in center.shape),
        dtype=getattr(center, "dtype", None),
        batch_size=batch_size,
    )


def build_two_site_hmm_scaffold_from_shapes(
    ltensor_shape,
    mpo0_shape,
    mpo1_shape,
    rtensor_shape,
    center_shape,
    *,
    dtype=None,
    batch_size: int = 1,
):
    """Build the two-site HMM scaffold from shape metadata only."""

    l_shape = tuple(int(dim) for dim in ltensor_shape)
    mpo0_shape = tuple(int(dim) for dim in mpo0_shape)
    mpo1_shape = tuple(int(dim) for dim in mpo1_shape)
    r_shape = tuple(int(dim) for dim in rtensor_shape)
    c_shape = tuple(int(dim) for dim in center_shape)
    if len(l_shape) != 3:
        raise ValueError("ltensor for two-site HMM scaffold must have shape (a, b, c)")
    if len(mpo0_shape) != 4:
        raise ValueError("first mpo for two-site HMM scaffold must have shape (b, d, e, f)")
    if len(mpo1_shape) != 4:
        raise ValueError("second mpo for two-site HMM scaffold must have shape (f, g, h, j)")
    if len(r_shape) != 3:
        raise ValueError("rtensor for two-site HMM scaffold must have shape (l, j, k)")
    if len(c_shape) != 4:
        raise ValueError("center for two-site HMM scaffold must have shape (c, e, h, k)")

    a, b_left, c_left = l_shape
    b_mpo, d, e_mpo, f_mpo0 = mpo0_shape
    f_mpo1, g, h_mpo, j_mpo = mpo1_shape
    l_right, j_right, k_right = r_shape
    c_center, e_center, h_center, k_center = c_shape
    if b_left != b_mpo:
        raise ValueError("ltensor b dimension must match first mpo b dimension")
    if c_left != c_center:
        raise ValueError("ltensor c dimension must match center c dimension")
    if e_mpo != e_center:
        raise ValueError("first mpo e dimension must match center e dimension")
    if f_mpo0 != f_mpo1:
        raise ValueError("first mpo f dimension must match second mpo f dimension")
    if h_mpo != h_center:
        raise ValueError("second mpo h dimension must match center h dimension")
    if j_mpo != j_right:
        raise ValueError("second mpo j dimension must match rtensor j dimension")
    if k_right != k_center:
        raise ValueError("rtensor k dimension must match center k dimension")

    center_array = _ShapeArray(c_shape, dtype=dtype)
    mpo0_array = _ShapeArray(mpo0_shape, dtype=dtype)
    mpo1_array = _ShapeArray(mpo1_shape, dtype=dtype)
    ltensor_array = _ShapeArray(l_shape, dtype=dtype)
    stage0_rhs = _ShapeArray((k_center, l_right * j_right), dtype=dtype)
    stage1_rhs = _ShapeArray((h_center * j_mpo, c_center * e_center * l_right), dtype=dtype)
    stage2_rhs = _ShapeArray((e_center * f_mpo0, c_center * g * l_right), dtype=dtype)
    stage3_rhs = _ShapeArray((b_left * c_center, d * g * l_right), dtype=dtype)
    equation = "abc,bdef,fghj,ljk,cehk->adgl"
    stage0 = GemmTask(
        center_array.reshape(c_center * e_center * h_center, k_center),
        stage0_rhs,
        tag={
            "equation": equation,
            "stage": 0,
            "role": "center_right",
            "output_shape": (c_center * e_center * h_center, l_right * j_right),
        },
    )
    stage1 = GemmTask(
        mpo1_array.reshape(f_mpo1 * g, h_mpo * j_mpo),
        stage1_rhs,
        tag={
            "equation": equation,
            "stage": 1,
            "role": "second_mpo_intermediate",
            "output_shape": (f_mpo1 * g, c_center * e_center * l_right),
        },
    )
    stage2 = GemmTask(
        mpo0_array.reshape(b_mpo * d, e_mpo * f_mpo0),
        stage2_rhs,
        tag={
            "equation": equation,
            "stage": 2,
            "role": "first_mpo_intermediate",
            "output_shape": (b_mpo * d, c_center * g * l_right),
        },
    )
    stage3 = GemmTask(
        ltensor_array.reshape(a, b_left * c_left),
        stage3_rhs,
        tag={
            "equation": equation,
            "stage": 3,
            "role": "left_reduce",
            "output_shape": (a, d * g * l_right),
        },
    )
    output_shape = (a, d, g, l_right)
    inter_shape = (c_center * e_center * h_center, l_right * j_right)
    cost = float(
        2 * (c_center * e_center * h_center) * (l_right * j_right) * k_center
        + 2 * (f_mpo1 * g) * (c_center * e_center * l_right) * (h_mpo * j_mpo)
        + 2 * (b_mpo * d) * (c_center * g * l_right) * (e_mpo * f_mpo0)
        + 2 * a * (d * g * l_right) * (b_left * c_left)
    )
    hxlist = HxBlockList.from_blocks(
        [
            HxBlock(
                block_id=0,
                input_slice=BufferSlice("X", offset=0, shape=c_shape),
                output_slice=BufferSlice("Y", offset=0, shape=output_shape),
                coefficient=1.0,
                inter_slice=BufferSlice("INTER", offset=0, shape=inter_shape),
                matmul_stages=((stage0,), (stage1,), (stage2,), (stage3,)),
                gemv_inter=(),
                gemv_reduce=(),
                qn_key=None,
                term_key=("two_site",),
                cost=cost,
            )
        ],
        center_kind="twodot",
        direct_intermediate=True,
        qn_adapted=False,
        equation=equation,
        operand_shapes=(l_shape, mpo0_shape, mpo1_shape, r_shape, c_shape),
        output_shape=output_shape,
    )
    return hxlist, build_hmm_task(hxlist, batch_size=batch_size)


def build_batched_two_site_hmm_scaffold(
    ltensor,
    mpo0,
    mpo1,
    rtensor,
    center,
    nrhs,
    *,
    batch_size: int = 1,
):
    """Build an HMMTask scaffold for ``abc,bdef,fghj,ljk,cehkr->adglr``."""

    return build_batched_two_site_hmm_scaffold_from_shapes(
        tuple(int(dim) for dim in ltensor.shape),
        tuple(int(dim) for dim in mpo0.shape),
        tuple(int(dim) for dim in mpo1.shape),
        tuple(int(dim) for dim in rtensor.shape),
        tuple(int(dim) for dim in center.shape),
        nrhs=nrhs,
        dtype=getattr(center, "dtype", None),
        batch_size=batch_size,
    )


def build_batched_two_site_hmm_scaffold_from_shapes(
    ltensor_shape,
    mpo0_shape,
    mpo1_shape,
    rtensor_shape,
    center_shape,
    *,
    nrhs,
    dtype=None,
    batch_size: int = 1,
):
    """Build the two-site batched-RHS HMM scaffold from shape metadata."""

    nrhs = _validate_nrhs(nrhs)
    l_shape = tuple(int(dim) for dim in ltensor_shape)
    mpo0_shape = tuple(int(dim) for dim in mpo0_shape)
    mpo1_shape = tuple(int(dim) for dim in mpo1_shape)
    r_shape = tuple(int(dim) for dim in rtensor_shape)
    c_shape = tuple(int(dim) for dim in center_shape)
    if len(l_shape) != 3:
        raise ValueError("ltensor for batched two-site HMM scaffold must have shape (a, b, c)")
    if len(mpo0_shape) != 4:
        raise ValueError("first mpo for batched two-site HMM scaffold must have shape (b, d, e, f)")
    if len(mpo1_shape) != 4:
        raise ValueError("second mpo for batched two-site HMM scaffold must have shape (f, g, h, j)")
    if len(r_shape) != 3:
        raise ValueError("rtensor for batched two-site HMM scaffold must have shape (l, j, k)")
    if len(c_shape) != 4:
        raise ValueError("center for batched two-site HMM scaffold must have shape (c, e, h, k)")

    a, b_left, c_left = l_shape
    b_mpo, d, e_mpo, f_mpo0 = mpo0_shape
    f_mpo1, g, h_mpo, j_mpo = mpo1_shape
    l_right, j_right, k_right = r_shape
    c_center, e_center, h_center, k_center = c_shape
    if b_left != b_mpo:
        raise ValueError("ltensor b dimension must match first mpo b dimension")
    if c_left != c_center:
        raise ValueError("ltensor c dimension must match center c dimension")
    if e_mpo != e_center:
        raise ValueError("first mpo e dimension must match center e dimension")
    if f_mpo0 != f_mpo1:
        raise ValueError("first mpo f dimension must match second mpo f dimension")
    if h_mpo != h_center:
        raise ValueError("second mpo h dimension must match center h dimension")
    if j_mpo != j_right:
        raise ValueError("second mpo j dimension must match rtensor j dimension")
    if k_right != k_center:
        raise ValueError("rtensor k dimension must match center k dimension")

    stage0_left = _ShapeArray((nrhs, c_center * e_center * h_center, k_center), dtype=dtype)
    stage0_rhs = _ShapeArray((nrhs, k_center, l_right * j_right), dtype=dtype)
    stage1_left = _ShapeArray((nrhs, f_mpo1 * g, h_mpo * j_mpo), dtype=dtype)
    stage1_rhs = _ShapeArray((nrhs, h_center * j_mpo, c_center * e_center * l_right), dtype=dtype)
    stage2_left = _ShapeArray((nrhs, b_mpo * d, e_mpo * f_mpo0), dtype=dtype)
    stage2_rhs = _ShapeArray((nrhs, e_center * f_mpo0, c_center * g * l_right), dtype=dtype)
    stage3_left = _ShapeArray((nrhs, a, b_left * c_left), dtype=dtype)
    stage3_rhs = _ShapeArray((nrhs, b_left * c_center, d * g * l_right), dtype=dtype)
    equation = "abc,bdef,fghj,ljk,cehkr->adglr"
    stage0 = GemmTask(
        stage0_left,
        stage0_rhs,
        tag={
            "equation": equation,
            "stage": 0,
            "role": "center_right_batched_rhs",
            "output_shape": (nrhs, c_center * e_center * h_center, l_right * j_right),
        },
    )
    stage1 = GemmTask(
        stage1_left,
        stage1_rhs,
        tag={
            "equation": equation,
            "stage": 1,
            "role": "second_mpo_intermediate_batched_rhs",
            "output_shape": (nrhs, f_mpo1 * g, c_center * e_center * l_right),
        },
    )
    stage2 = GemmTask(
        stage2_left,
        stage2_rhs,
        tag={
            "equation": equation,
            "stage": 2,
            "role": "first_mpo_intermediate_batched_rhs",
            "output_shape": (nrhs, b_mpo * d, c_center * g * l_right),
        },
    )
    stage3 = GemmTask(
        stage3_left,
        stage3_rhs,
        tag={
            "equation": equation,
            "stage": 3,
            "role": "left_reduce_batched_rhs",
            "output_shape": (nrhs, a, d * g * l_right),
        },
    )
    output_shape = (a, d, g, l_right, nrhs)
    batched_center_shape = c_shape + (nrhs,)
    inter_shape = (nrhs, c_center * e_center * h_center, l_right * j_right)
    cost = float(
        nrhs * (
            2 * (c_center * e_center * h_center) * (l_right * j_right) * k_center
            + 2 * (f_mpo1 * g) * (c_center * e_center * l_right) * (h_mpo * j_mpo)
            + 2 * (b_mpo * d) * (c_center * g * l_right) * (e_mpo * f_mpo0)
            + 2 * a * (d * g * l_right) * (b_left * c_left)
        )
    )
    hxlist = HxBlockList.from_blocks(
        [
            HxBlock(
                block_id=0,
                input_slice=BufferSlice("X", offset=0, shape=batched_center_shape),
                output_slice=BufferSlice("Y", offset=0, shape=output_shape),
                coefficient=1.0,
                inter_slice=BufferSlice("INTER", offset=0, shape=inter_shape),
                matmul_stages=((stage0,), (stage1,), (stage2,), (stage3,)),
                gemv_inter=(),
                gemv_reduce=(),
                qn_key=None,
                term_key=("two_site", "batched_rhs"),
                cost=cost,
            )
        ],
        center_kind="twodot",
        direct_intermediate=True,
        qn_adapted=False,
        equation=equation,
        operand_shapes=(l_shape, mpo0_shape, mpo1_shape, r_shape, batched_center_shape),
        output_shape=output_shape,
    )
    return hxlist, build_hmm_task(hxlist, batch_size=batch_size)


def _validate_two_site_hmm_shapes(ltensor, mpo0, mpo1, rtensor, center):
    l_shape = _shape_tuple(ltensor)
    mpo0_shape = _shape_tuple(mpo0)
    mpo1_shape = _shape_tuple(mpo1)
    r_shape = _shape_tuple(rtensor)
    c_shape = _shape_tuple(center)
    if len(l_shape) != 3:
        raise ValueError("ltensor for two-site HMM action must have shape (a, b, c)")
    if len(mpo0_shape) != 4:
        raise ValueError("first mpo for two-site HMM action must have shape (b, d, e, f)")
    if len(mpo1_shape) != 4:
        raise ValueError("second mpo for two-site HMM action must have shape (f, g, h, j)")
    if len(r_shape) != 3:
        raise ValueError("rtensor for two-site HMM action must have shape (l, j, k)")
    if len(c_shape) not in (4, 5):
        raise ValueError("center for two-site HMM action must have shape (c, e, h, k) or (c, e, h, k, nrhs)")

    a, b_left, c_left = l_shape
    b_mpo, d, e_mpo, f_mpo0 = mpo0_shape
    f_mpo1, g, h_mpo, j_mpo = mpo1_shape
    l_right, j_right, k_right = r_shape
    c_center, e_center, h_center, k_center = c_shape[:4]
    if b_left != b_mpo:
        raise ValueError("ltensor b dimension must match first mpo b dimension")
    if c_left != c_center:
        raise ValueError("ltensor c dimension must match center c dimension")
    if e_mpo != e_center:
        raise ValueError("first mpo e dimension must match center e dimension")
    if f_mpo0 != f_mpo1:
        raise ValueError("first mpo f dimension must match second mpo f dimension")
    if h_mpo != h_center:
        raise ValueError("second mpo h dimension must match center h dimension")
    if j_mpo != j_right:
        raise ValueError("second mpo j dimension must match rtensor j dimension")
    if k_right != k_center:
        raise ValueError("rtensor k dimension must match center k dimension")
    if len(c_shape) == 5 and c_shape[4] <= 0:
        raise ValueError("two-site HMM action nrhs dimension must be positive")

    return {
        "a": a,
        "b": b_left,
        "c": c_center,
        "d": d,
        "e": e_center,
        "f": f_mpo0,
        "g": g,
        "h": h_center,
        "j": j_mpo,
        "l": l_right,
        "k": k_center,
        "nrhs": c_shape[4] if len(c_shape) == 5 else None,
    }


def _two_site_hmm_task_for_action(ltensor, mpo0, mpo1, rtensor, center, dims):
    dtype = getattr(center, "dtype", None)
    if dims["nrhs"] is None:
        _hxlist, task = build_two_site_hmm_scaffold_from_shapes(
            _shape_tuple(ltensor),
            _shape_tuple(mpo0),
            _shape_tuple(mpo1),
            _shape_tuple(rtensor),
            _shape_tuple(center),
            dtype=dtype,
            batch_size=1,
        )
        return task
    _hxlist, task = build_batched_two_site_hmm_scaffold_from_shapes(
        _shape_tuple(ltensor),
        _shape_tuple(mpo0),
        _shape_tuple(mpo1),
        _shape_tuple(rtensor),
        _shape_tuple(center)[:4],
        nrhs=dims["nrhs"],
        dtype=dtype,
        batch_size=1,
    )
    return task


def execute_two_site_hmm_action(
    backend,
    ltensor,
    mpo0,
    mpo1,
    rtensor,
    center,
    *,
    stream=None,
    workspace=None,
):
    """Execute ``abc,bdef,fghj,ljk,cehk->adgl`` or batched ``cehkr->adglr``."""

    import time

    dims = _validate_two_site_hmm_shapes(ltensor, mpo0, mpo1, rtensor, center)
    profile_task = _two_site_hmm_task_for_action(ltensor, mpo0, mpo1, rtensor, center, dims)
    started = time.perf_counter()
    stage_profiles = []
    previous_stage_profiles = getattr(backend, "_hmm_active_stage_profiles", None)
    setattr(backend, "_hmm_active_stage_profiles", stage_profiles)
    try:
        if dims["nrhs"] is None:
            result = _execute_two_site_hmm_action_scalar(
                backend,
                ltensor,
                mpo0,
                mpo1,
                rtensor,
                center,
                dims,
                stream=stream,
                workspace=workspace,
            )
        else:
            result = _execute_two_site_hmm_action_batched(
                backend,
                ltensor,
                mpo0,
                mpo1,
                rtensor,
                center,
                dims,
                stream=stream,
                workspace=workspace,
            )
    finally:
        if previous_stage_profiles is None:
            try:
                delattr(backend, "_hmm_active_stage_profiles")
            except AttributeError:
                pass
        else:
            setattr(backend, "_hmm_active_stage_profiles", previous_stage_profiles)
    _record_hmm_action_execution_profile(
        backend,
        profile_task,
        result,
        time.perf_counter() - started,
        stream=stream,
        workspace=workspace,
        stage_profiles=stage_profiles,
    )
    return result


def _execute_two_site_hmm_action_scalar(
    backend,
    ltensor,
    mpo0,
    mpo1,
    rtensor,
    center,
    dims,
    *,
    stream=None,
    workspace=None,
):
    xp = _backend_xp(backend)
    a = dims["a"]
    b = dims["b"]
    c = dims["c"]
    d = dims["d"]
    e = dims["e"]
    f = dims["f"]
    g = dims["g"]
    h = dims["h"]
    j = dims["j"]
    l_right = dims["l"]
    k = dims["k"]

    right0 = _reshape(xp, _transpose(xp, rtensor, (2, 0, 1)), (k, l_right * j))
    tmp0 = _execute_hmm_matmul_plan(
        backend,
        _reshape(xp, center, (c * e * h, k)),
        right0,
        reason="two_site_hmm_stage_0",
        stream=stream,
        workspace=workspace,
    )
    tmp0 = _reshape(xp, tmp0, (c, e, h, l_right, j))

    right1 = _reshape(xp, _transpose(xp, tmp0, (2, 4, 0, 1, 3)), (h * j, c * e * l_right))
    tmp1 = _execute_hmm_matmul_plan(
        backend,
        _reshape(xp, mpo1, (f * g, h * j)),
        right1,
        reason="two_site_hmm_stage_1",
        stream=stream,
        workspace=workspace,
    )
    tmp1 = _reshape(xp, tmp1, (f, g, c, e, l_right))

    right2 = _reshape(xp, _transpose(xp, tmp1, (3, 0, 2, 1, 4)), (e * f, c * g * l_right))
    tmp2 = _execute_hmm_matmul_plan(
        backend,
        _reshape(xp, mpo0, (b * d, e * f)),
        right2,
        reason="two_site_hmm_stage_2",
        stream=stream,
        workspace=workspace,
    )
    tmp2 = _reshape(xp, tmp2, (b, d, c, g, l_right))

    right3 = _reshape(xp, _transpose(xp, tmp2, (0, 2, 1, 3, 4)), (b * c, d * g * l_right))
    result = _execute_hmm_matmul_plan(
        backend,
        _reshape(xp, ltensor, (a, b * c)),
        right3,
        reason="two_site_hmm_stage_3",
        stream=stream,
        workspace=workspace,
    )
    return _reshape(xp, result, (a, d, g, l_right))


def _execute_two_site_hmm_action_batched(
    backend,
    ltensor,
    mpo0,
    mpo1,
    rtensor,
    center,
    dims,
    *,
    stream=None,
    workspace=None,
):
    xp = _backend_xp(backend)
    a = dims["a"]
    b = dims["b"]
    c = dims["c"]
    d = dims["d"]
    e = dims["e"]
    f = dims["f"]
    g = dims["g"]
    h = dims["h"]
    j = dims["j"]
    l_right = dims["l"]
    k = dims["k"]
    nrhs = dims["nrhs"]

    center_batch = _reshape(xp, _transpose(xp, center, (4, 0, 1, 2, 3)), (nrhs, c * e * h, k))
    right0 = _reshape(xp, _transpose(xp, rtensor, (2, 0, 1)), (k, l_right * j))
    tmp0 = _execute_hmm_matmul_plan(
        backend,
        center_batch,
        right0,
        reason="two_site_hmm_batched_rhs_stage_0",
        stream=stream,
        workspace=workspace,
    )
    tmp0 = _reshape(xp, tmp0, (nrhs, c, e, h, l_right, j))

    right1 = _reshape(
        xp,
        _transpose(xp, tmp0, (0, 3, 5, 1, 2, 4)),
        (nrhs, h * j, c * e * l_right),
    )
    tmp1 = _execute_hmm_matmul_plan(
        backend,
        _reshape(xp, mpo1, (f * g, h * j)),
        right1,
        reason="two_site_hmm_batched_rhs_stage_1",
        stream=stream,
        workspace=workspace,
    )
    tmp1 = _reshape(xp, tmp1, (nrhs, f, g, c, e, l_right))

    right2 = _reshape(
        xp,
        _transpose(xp, tmp1, (0, 4, 1, 3, 2, 5)),
        (nrhs, e * f, c * g * l_right),
    )
    tmp2 = _execute_hmm_matmul_plan(
        backend,
        _reshape(xp, mpo0, (b * d, e * f)),
        right2,
        reason="two_site_hmm_batched_rhs_stage_2",
        stream=stream,
        workspace=workspace,
    )
    tmp2 = _reshape(xp, tmp2, (nrhs, b, d, c, g, l_right))

    right3 = _reshape(
        xp,
        _transpose(xp, tmp2, (0, 1, 3, 2, 4, 5)),
        (nrhs, b * c, d * g * l_right),
    )
    result = _execute_hmm_matmul_plan(
        backend,
        _reshape(xp, ltensor, (a, b * c)),
        right3,
        reason="two_site_hmm_batched_rhs_stage_3",
        stream=stream,
        workspace=workspace,
    )
    result = _reshape(xp, result, (nrhs, a, d, g, l_right))
    return _transpose(xp, result, (1, 2, 3, 4, 0))


def _validate_nrhs(nrhs):
    nrhs = int(nrhs)
    if nrhs <= 0:
        raise ValueError("nrhs must be positive")
    return nrhs


def _hx_block_size(block: HxBlock) -> int:
    sizes = [
        _prod(block.input_slice.shape),
        _prod(block.output_slice.shape),
    ]
    if block.inter_slice is not None:
        sizes.append(_prod(block.inter_slice.shape))
    return max(sizes)


@dataclass(frozen=True)
class GroupedGemmStats:
    task_count: int
    shape_bucket_count: int
    batched_bucket_count: int
    loop_bucket_count: int
    batched_task_count: int
    loop_task_count: int
    bucket_task_counts: tuple[int, ...]
    shape_buckets: tuple[dict[str, Any], ...]
    flops: int
    read_bytes: int
    write_bytes: int
    copy_bytes: int
    workspace_bytes: int = 0


@dataclass(frozen=True)
class GroupedGemmExecutionProfile:
    pack_strategy: str
    prepacked: bool
    pack_s: float
    kernel_s: float
    scatter_s: float
    loop_s: float
    pack_bytes: int
    kernel_calls: int
    batched_kernel_calls: int
    loop_kernel_calls: int
    bucket_execution_profiles: tuple[dict[str, Any], ...]

    def to_dict(self):
        bucket_profiles = [dict(item) for item in self.bucket_execution_profiles]
        bucket_kernel_primitives, bucket_kernel_calls_by_primitive = _bucket_kernel_profile(bucket_profiles)
        return {
            "pack_strategy": self.pack_strategy,
            "prepacked": bool(self.prepacked),
            "pack_s": float(self.pack_s),
            "kernel_s": float(self.kernel_s),
            "scatter_s": float(self.scatter_s),
            "loop_s": float(self.loop_s),
            "pack_bytes": int(self.pack_bytes),
            "kernel_calls": int(self.kernel_calls),
            "batched_kernel_calls": int(self.batched_kernel_calls),
            "loop_kernel_calls": int(self.loop_kernel_calls),
            "bucket_execution_profiles": bucket_profiles,
            "bucket_kernel_primitives": bucket_kernel_primitives,
            "bucket_kernel_calls_by_primitive": bucket_kernel_calls_by_primitive,
            "bucket_selection_reasons": _bucket_profile_reasons(bucket_profiles, "selection_reason"),
            "bucket_fallback_reasons": _bucket_profile_reasons(bucket_profiles, "fallback_reason"),
        }


@dataclass(frozen=True)
class PrepackedGemmBucket:
    task_indices: tuple[int, ...]
    tasks: tuple[GemmTask, ...]
    execution: str
    a_pack: Any | None = None
    b_pack: Any | None = None
    pack_bytes: int = 0
    pack_s: float = 0.0
    profile: dict[str, Any] | None = None

    def __post_init__(self):
        task_indices = tuple(int(index) for index in self.task_indices)
        if any(index < 0 for index in task_indices):
            raise ValueError("PrepackedGemmBucket task indices must be non-negative")
        execution = str(self.execution)
        if execution not in ("batched", "loop"):
            raise ValueError("PrepackedGemmBucket execution must be batched or loop")
        pack_bytes = int(self.pack_bytes)
        if pack_bytes < 0:
            raise ValueError("PrepackedGemmBucket pack_bytes must be non-negative")
        object.__setattr__(self, "task_indices", task_indices)
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "execution", execution)
        object.__setattr__(self, "pack_bytes", pack_bytes)
        object.__setattr__(self, "pack_s", float(self.pack_s))
        object.__setattr__(self, "profile", dict(self.profile or {}))


@dataclass(frozen=True)
class PrepackedGroupedGemm:
    tasks: tuple[GemmTask, ...]
    buckets: tuple[PrepackedGemmBucket, ...]
    stats: GroupedGemmStats
    pack_threshold: int
    profile: dict[str, Any]

    def __post_init__(self):
        pack_threshold = int(self.pack_threshold)
        if pack_threshold < 0:
            raise ValueError("PrepackedGroupedGemm pack_threshold must be non-negative")
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "buckets", tuple(self.buckets))
        object.__setattr__(self, "pack_threshold", pack_threshold)
        object.__setattr__(self, "profile", dict(self.profile))


def apply_gemm_flags(x, *, trans=False, conj=False, xp=None):
    if xp is None:
        import numpy as xp

    if conj:
        x = xp.conj(x)
    if trans:
        x = xp.swapaxes(x, -1, -2)
    return x


def array_nbytes(x) -> int:
    nbytes = getattr(x, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    numel = getattr(x, "numel", None)
    element_size = getattr(x, "element_size", None)
    if callable(numel) and callable(element_size):
        return int(numel() * element_size())
    shape = getattr(x, "shape", ())
    size = 1
    for dim in shape:
        size *= int(dim)
    dtype = getattr(x, "dtype", None)
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is not None:
        return int(size * itemsize)
    return 0


def _xp_matmul_provider_name(xp):
    module_name = getattr(xp, "__name__", None)
    if not module_name:
        module_name = type(xp).__module__
    return "{0}.matmul".format(module_name)


def gemm_task_key(task: GemmTask, *, xp=None):
    a = apply_gemm_flags(task.A, trans=task.trans_a, conj=task.conj_a, xp=xp)
    b = apply_gemm_flags(task.B, trans=task.trans_b, conj=task.conj_b, xp=xp)
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ValueError("grouped_gemm tasks must contain matrices or batched matrices")
    batch_shape = tuple(int(dim) for dim in a.shape[:-2])
    b_batch_shape = tuple(int(dim) for dim in b.shape[:-2])
    if batch_shape != b_batch_shape:
        raise ValueError("GEMM task batch dimensions must match")
    m, k_left = int(a.shape[-2]), int(a.shape[-1])
    k_right, n = int(b.shape[-2]), int(b.shape[-1])
    if k_left != k_right:
        raise ValueError("GEMM task has incompatible contracted dimensions")
    return (
        str(getattr(a, "dtype", None)),
        str(getattr(b, "dtype", None)),
        batch_shape,
        m,
        n,
        k_left,
        bool(task.trans_a),
        bool(task.trans_b),
        bool(task.conj_a),
        bool(task.conj_b),
    )


def group_tasks_by_shape(tasks, *, xp=None):
    buckets = defaultdict(list)
    for task in tasks:
        buckets[gemm_task_key(task, xp=xp)].append(task)
    return dict(buckets)


def _gemm_shape(task: GemmTask, *, xp=None):
    a = apply_gemm_flags(task.A, trans=task.trans_a, conj=task.conj_a, xp=xp)
    b = apply_gemm_flags(task.B, trans=task.trans_b, conj=task.conj_b, xp=xp)
    return int(a.shape[-2]), int(b.shape[-1]), int(a.shape[-1])


def _gemm_batch_shape(task: GemmTask, *, xp=None):
    a = apply_gemm_flags(task.A, trans=task.trans_a, conj=task.conj_a, xp=xp)
    b = apply_gemm_flags(task.B, trans=task.trans_b, conj=task.conj_b, xp=xp)
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ValueError("GEMM task must contain matrices or batched matrices")
    batch_shape = tuple(int(dim) for dim in a.shape[:-2])
    if batch_shape != tuple(int(dim) for dim in b.shape[:-2]):
        raise ValueError("GEMM task batch dimensions must match")
    return batch_shape


def grouped_gemm_bucket_selection(tasks, *, xp=None, pack_threshold=4, flop_copy_ratio=10, allow_batched=True):
    tasks = list(tasks)
    if not allow_batched:
        return False, "loop: batched execution disabled", "batched execution disabled"
    if len(tasks) < pack_threshold:
        return (
            False,
            "loop: task_count {0} below pack_threshold {1}".format(len(tasks), int(pack_threshold)),
            "task_count below pack_threshold",
        )
    flops = 0
    copy_bytes = 0
    for task in tasks:
        m, n, k = _gemm_shape(task, xp=xp)
        flops += max(_prod(_gemm_batch_shape(task, xp=xp)), 1) * 2 * m * n * k
        copy_bytes += array_nbytes(task.A)
        copy_bytes += array_nbytes(task.B)
        if task.C is not None:
            copy_bytes += array_nbytes(task.C)
    copy_threshold = int(flop_copy_ratio * copy_bytes)
    if copy_bytes and flops < copy_threshold:
        return (
            False,
            "loop: flops {0} below copy_threshold {1}".format(int(flops), copy_threshold),
            "flops below copy threshold",
        )
    return (
        True,
        "batched: task_count {0} >= pack_threshold {1} and flops {2} >= copy_threshold {3}".format(
            len(tasks),
            int(pack_threshold),
            int(flops),
            copy_threshold,
        ),
        None,
    )


def should_batch(tasks, *, xp=None, pack_threshold=4, flop_copy_ratio=10, allow_batched=True):
    return grouped_gemm_bucket_selection(
        tasks,
        xp=xp,
        pack_threshold=pack_threshold,
        flop_copy_ratio=flop_copy_ratio,
        allow_batched=allow_batched,
    )[0]


def grouped_gemm_stats(tasks, *, xp=None, pack_threshold=4, flop_copy_ratio=10, allow_batched=True) -> GroupedGemmStats:
    tasks = list(tasks)
    buckets = group_tasks_by_shape(tasks, xp=xp)
    batched_bucket_count = 0
    loop_bucket_count = 0
    batched_task_count = 0
    loop_task_count = 0
    shape_buckets = []
    flops = 0
    read_bytes = 0
    write_bytes = 0
    copy_bytes = 0
    for key, group in sorted(buckets.items()):
        group_batched = should_batch(
            group,
            xp=xp,
            pack_threshold=pack_threshold,
            flop_copy_ratio=flop_copy_ratio,
            allow_batched=allow_batched,
        )
        if group_batched:
            batched_bucket_count += 1
            batched_task_count += len(group)
        else:
            loop_bucket_count += 1
            loop_task_count += len(group)
        dtype_a, dtype_b, batch_shape, m, n, k, trans_a, trans_b, conj_a, conj_b = key
        shape_buckets.append({
            "dtype_a": dtype_a,
            "dtype_b": dtype_b,
            "batch_shape": tuple(int(dim) for dim in batch_shape),
            "m": int(m),
            "n": int(n),
            "k": int(k),
            "trans_a": bool(trans_a),
            "trans_b": bool(trans_b),
            "conj_a": bool(conj_a),
            "conj_b": bool(conj_b),
            "execution": "batched" if group_batched else "loop",
            "task_count": int(len(group)),
        })
        for task in group:
            m, n, k = _gemm_shape(task, xp=xp)
            batch_count = max(_prod(_gemm_batch_shape(task, xp=xp)), 1)
            flops += int(batch_count * 2 * m * n * k)
            read_bytes += array_nbytes(task.A) + array_nbytes(task.B)
            write_bytes += int(batch_count * m * n * max(array_nbytes(task.A) // max(batch_count * m * k, 1), array_nbytes(task.B) // max(batch_count * k * n, 1)))
            if group_batched:
                copy_bytes += array_nbytes(task.A) + array_nbytes(task.B)
                if task.C is not None:
                    copy_bytes += array_nbytes(task.C)
    bucket_task_counts = tuple(int(bucket["task_count"]) for bucket in shape_buckets)
    return GroupedGemmStats(
        task_count=len(tasks),
        shape_bucket_count=len(buckets),
        batched_bucket_count=batched_bucket_count,
        loop_bucket_count=loop_bucket_count,
        batched_task_count=batched_task_count,
        loop_task_count=loop_task_count,
        bucket_task_counts=bucket_task_counts,
        shape_buckets=tuple(shape_buckets),
        flops=int(flops),
        read_bytes=int(read_bytes),
        write_bytes=int(write_bytes),
        copy_bytes=int(copy_bytes),
    )


def run_gemm_task(task: GemmTask, *, xp=None):
    if xp is None:
        import numpy as xp

    a = apply_gemm_flags(task.A, trans=task.trans_a, conj=task.conj_a, xp=xp)
    b = apply_gemm_flags(task.B, trans=task.trans_b, conj=task.conj_b, xp=xp)
    result = xp.matmul(a, b)
    if task.alpha != 1.0:
        result = task.alpha * result
    if task.C is not None:
        if task.beta != 0.0:
            result = result + task.beta * task.C
        task.C[...] = result
        return task.C
    return result


def grouped_gemm_bucketed(
    tasks,
    *,
    xp=None,
    pack_threshold=4,
    flop_copy_ratio=10,
    allow_batched=True,
    batched_matmul=None,
    batched_matmul_provider=None,
):
    results, _profile = grouped_gemm_bucketed_profiled(
        tasks,
        xp=xp,
        pack_threshold=pack_threshold,
        flop_copy_ratio=flop_copy_ratio,
        allow_batched=allow_batched,
        batched_matmul=batched_matmul,
        batched_matmul_provider=batched_matmul_provider,
    )
    return results


def grouped_gemm_bucketed_profiled(
    tasks,
    *,
    xp=None,
    pack_threshold=4,
    flop_copy_ratio=10,
    allow_batched=True,
    batched_matmul=None,
    batched_matmul_provider=None,
):
    import time

    if xp is None:
        import numpy as xp

    tasks = list(tasks)
    buckets = group_tasks_by_shape(tasks, xp=xp)
    results_by_task = {}
    pack_s = 0.0
    kernel_s = 0.0
    scatter_s = 0.0
    loop_s = 0.0
    pack_bytes = 0
    batched_kernel_calls = 0
    loop_kernel_calls = 0
    bucket_profiles = []
    for group in buckets.values():
        group_batched, selection_reason, bucket_fallback_reason = grouped_gemm_bucket_selection(
            group,
            xp=xp,
            pack_threshold=pack_threshold,
            flop_copy_ratio=flop_copy_ratio,
            allow_batched=allow_batched,
        )
        if not group_batched:
            loop_started = time.perf_counter()
            for task in group:
                results_by_task[id(task)] = run_gemm_task(task, xp=xp)
            bucket_loop_s = time.perf_counter() - loop_started
            loop_s += bucket_loop_s
            loop_kernel_calls += len(group)
            bucket_profiles.append(_grouped_gemm_bucket_execution_profile(
                group,
                xp=xp,
                execution="loop",
                pack_strategy="none",
                pack_s=0.0,
                kernel_s=bucket_loop_s,
                scatter_s=0.0,
                pack_bytes=0,
                kernel_calls=len(group),
                selection_reason=selection_reason,
                fallback_reason=bucket_fallback_reason,
                batched_matmul_provider=None,
            ))
            continue

        bucket_pack_bytes = sum(array_nbytes(task.A) + array_nbytes(task.B) for task in group)
        pack_started = time.perf_counter()
        a_pack = xp.stack([
            apply_gemm_flags(task.A, trans=task.trans_a, conj=task.conj_a, xp=xp)
            for task in group
        ], axis=0)
        b_pack = xp.stack([
            apply_gemm_flags(task.B, trans=task.trans_b, conj=task.conj_b, xp=xp)
            for task in group
        ], axis=0)
        bucket_pack_s = time.perf_counter() - pack_started
        pack_s += bucket_pack_s
        pack_bytes += bucket_pack_bytes

        kernel_started = time.perf_counter()
        if batched_matmul is None:
            c_pack = xp.matmul(a_pack, b_pack)
        else:
            c_pack = batched_matmul(a_pack, b_pack)
        bucket_kernel_s = time.perf_counter() - kernel_started
        kernel_s += bucket_kernel_s
        batched_kernel_calls += 1

        scatter_started = time.perf_counter()
        for index, task in enumerate(group):
            result = c_pack[index]
            if task.alpha != 1.0:
                result = task.alpha * result
            if task.C is not None:
                if task.beta != 0.0:
                    result = result + task.beta * task.C
                task.C[...] = result
                results_by_task[id(task)] = task.C
            else:
                results_by_task[id(task)] = result
        bucket_scatter_s = time.perf_counter() - scatter_started
        scatter_s += bucket_scatter_s
        bucket_profiles.append(_grouped_gemm_bucket_execution_profile(
            group,
            xp=xp,
            execution="batched",
            pack_strategy="stack_per_call",
            pack_s=bucket_pack_s,
            kernel_s=bucket_kernel_s,
            scatter_s=bucket_scatter_s,
            pack_bytes=bucket_pack_bytes,
            kernel_calls=1,
            selection_reason=selection_reason,
            fallback_reason=bucket_fallback_reason,
            batched_matmul_provider=batched_matmul_provider or _xp_matmul_provider_name(xp),
        ))
    profile = GroupedGemmExecutionProfile(
        pack_strategy="stack_per_call" if batched_kernel_calls else "none",
        prepacked=False,
        pack_s=pack_s,
        kernel_s=kernel_s,
        scatter_s=scatter_s,
        loop_s=loop_s,
        pack_bytes=pack_bytes,
        kernel_calls=batched_kernel_calls + loop_kernel_calls,
        batched_kernel_calls=batched_kernel_calls,
        loop_kernel_calls=loop_kernel_calls,
        bucket_execution_profiles=tuple(bucket_profiles),
    )
    return [results_by_task[id(task)] for task in tasks], profile


def prepack_grouped_gemm(
    tasks,
    *,
    xp=None,
    pack_threshold=4,
    flop_copy_ratio=10,
    allow_batched=True,
    batched_matmul_provider=None,
):
    import time

    if xp is None:
        import numpy as xp

    tasks = tuple(tasks)
    stats = grouped_gemm_stats(
        tasks,
        xp=xp,
        pack_threshold=pack_threshold,
        flop_copy_ratio=flop_copy_ratio,
        allow_batched=allow_batched,
    )
    buckets_by_key = group_tasks_by_shape(tasks, xp=xp)
    task_id_to_index = {id(task): index for index, task in enumerate(tasks)}
    buckets = []
    pack_s = 0.0
    pack_bytes = 0
    batched_kernel_calls = 0
    loop_kernel_calls = 0
    bucket_profiles = []
    for group in buckets_by_key.values():
        indices = tuple(task_id_to_index[id(task)] for task in group)
        group_batched, selection_reason, bucket_fallback_reason = grouped_gemm_bucket_selection(
            group,
            xp=xp,
            pack_threshold=pack_threshold,
            flop_copy_ratio=flop_copy_ratio,
            allow_batched=allow_batched,
        )
        if not group_batched:
            loop_kernel_calls += len(group)
            bucket_profile = _grouped_gemm_bucket_execution_profile(
                group,
                xp=xp,
                execution="loop",
                pack_strategy="none",
                pack_s=0.0,
                kernel_s=0.0,
                scatter_s=0.0,
                pack_bytes=0,
                kernel_calls=len(group),
                selection_reason=selection_reason,
                fallback_reason=bucket_fallback_reason,
                batched_matmul_provider=None,
            )
            bucket_profiles.append(bucket_profile)
            buckets.append(PrepackedGemmBucket(
                task_indices=indices,
                tasks=tuple(group),
                execution="loop",
                profile=bucket_profile,
            ))
            continue

        bucket_pack_bytes = sum(array_nbytes(task.A) + array_nbytes(task.B) for task in group)
        pack_started = time.perf_counter()
        a_pack = xp.stack([
            apply_gemm_flags(task.A, trans=task.trans_a, conj=task.conj_a, xp=xp)
            for task in group
        ], axis=0)
        b_pack = xp.stack([
            apply_gemm_flags(task.B, trans=task.trans_b, conj=task.conj_b, xp=xp)
            for task in group
        ], axis=0)
        bucket_pack_s = time.perf_counter() - pack_started
        pack_s += bucket_pack_s
        pack_bytes += bucket_pack_bytes
        batched_kernel_calls += 1
        bucket_profile = _grouped_gemm_bucket_execution_profile(
            group,
            xp=xp,
            execution="batched",
            pack_strategy="prepack_once",
            pack_s=bucket_pack_s,
            kernel_s=0.0,
            scatter_s=0.0,
            pack_bytes=bucket_pack_bytes,
            kernel_calls=1,
            selection_reason=selection_reason,
            fallback_reason=bucket_fallback_reason,
            batched_matmul_provider=batched_matmul_provider or _xp_matmul_provider_name(xp),
        )
        bucket_profiles.append(bucket_profile)
        buckets.append(PrepackedGemmBucket(
            task_indices=indices,
            tasks=tuple(group),
            execution="batched",
            a_pack=a_pack,
            b_pack=b_pack,
            pack_bytes=bucket_pack_bytes,
            pack_s=bucket_pack_s,
            profile=bucket_profile,
        ))
    bucket_kernel_primitives, bucket_kernel_calls_by_primitive = _bucket_kernel_profile(bucket_profiles)
    profile = {
        "event": "grouped_gemm_prepack",
        "lowering": "grouped_gemm",
        "pack_strategy": "prepack_once" if batched_kernel_calls else "none",
        "prepacked": True,
        "pack_s": float(pack_s),
        "pack_bytes": int(pack_bytes),
        "flops": int(stats.flops),
        "total_flops": int(stats.flops),
        "read_bytes": int(stats.read_bytes),
        "write_bytes": int(stats.write_bytes),
        "copy_bytes": int(stats.copy_bytes),
        "workspace_bytes": int(stats.workspace_bytes),
        "num_tasks": int(len(tasks)),
        "num_grouped_tasks": int(stats.task_count),
        "num_groups": int(stats.shape_bucket_count),
        "num_shape_buckets": int(stats.shape_bucket_count),
        "num_batched_gemm": int(stats.batched_bucket_count),
        "num_gemm": int(stats.loop_task_count),
        "bucket_task_counts": tuple(int(count) for count in stats.bucket_task_counts),
        "shape_buckets": tuple(dict(bucket) for bucket in stats.shape_buckets),
        "kernel_calls": int(batched_kernel_calls + loop_kernel_calls),
        "batched_kernel_calls": int(batched_kernel_calls),
        "loop_kernel_calls": int(loop_kernel_calls),
        "bucket_execution_profiles": bucket_profiles,
        "bucket_kernel_primitives": bucket_kernel_primitives,
        "bucket_kernel_calls_by_primitive": bucket_kernel_calls_by_primitive,
        "bucket_selection_reasons": _bucket_profile_reasons(bucket_profiles, "selection_reason"),
        "bucket_fallback_reasons": _bucket_profile_reasons(bucket_profiles, "fallback_reason"),
        "batched_matmul_provider": (
            batched_matmul_provider or _xp_matmul_provider_name(xp)
            if batched_kernel_calls
            else None
        ),
    }
    return PrepackedGroupedGemm(
        tasks=tasks,
        buckets=tuple(buckets),
        stats=stats,
        pack_threshold=pack_threshold,
        profile=profile,
    )


def execute_prepacked_grouped_gemm(plan, *, xp=None, batched_matmul=None, batched_matmul_provider=None):
    import time

    if xp is None:
        import numpy as xp

    results_by_task = {}
    kernel_s = 0.0
    scatter_s = 0.0
    loop_s = 0.0
    batched_kernel_calls = 0
    loop_kernel_calls = 0
    bucket_profiles = []
    reused_pack_bytes = 0
    for bucket in plan.buckets:
        if bucket.execution == "loop":
            loop_started = time.perf_counter()
            for task in bucket.tasks:
                results_by_task[id(task)] = run_gemm_task(task, xp=xp)
            bucket_loop_s = time.perf_counter() - loop_started
            loop_s += bucket_loop_s
            loop_kernel_calls += len(bucket.tasks)
            bucket_profile = dict(bucket.profile)
            bucket_profile.update({
                "pack_strategy": "none",
                "pack_s": 0.0,
                "kernel_s": bucket_loop_s,
                "scatter_s": 0.0,
                "pack_bytes": 0,
                "kernel_calls": len(bucket.tasks),
                "batched_matmul_provider": None,
            })
            bucket_profiles.append(bucket_profile)
            continue

        kernel_started = time.perf_counter()
        if batched_matmul is None:
            c_pack = xp.matmul(bucket.a_pack, bucket.b_pack)
        else:
            c_pack = batched_matmul(bucket.a_pack, bucket.b_pack)
        bucket_kernel_s = time.perf_counter() - kernel_started
        kernel_s += bucket_kernel_s
        batched_kernel_calls += 1
        reused_pack_bytes += int(bucket.pack_bytes)

        scatter_started = time.perf_counter()
        for index, task in enumerate(bucket.tasks):
            result = c_pack[index]
            if task.alpha != 1.0:
                result = task.alpha * result
            if task.C is not None:
                if task.beta != 0.0:
                    result = result + task.beta * task.C
                task.C[...] = result
                results_by_task[id(task)] = task.C
            else:
                results_by_task[id(task)] = result
        bucket_scatter_s = time.perf_counter() - scatter_started
        scatter_s += bucket_scatter_s
        bucket_profile = dict(bucket.profile)
        bucket_profile.update({
            "pack_strategy": "prepacked_reuse",
            "pack_s": 0.0,
            "kernel_s": bucket_kernel_s,
            "scatter_s": bucket_scatter_s,
            "pack_bytes": 0,
            "reused_pack_bytes": int(bucket.pack_bytes),
            "kernel_calls": 1,
            "batched_matmul_provider": batched_matmul_provider or _xp_matmul_provider_name(xp),
        })
        bucket_profiles.append(bucket_profile)
    profile = GroupedGemmExecutionProfile(
        pack_strategy="prepacked_reuse" if batched_kernel_calls else "none",
        prepacked=True,
        pack_s=0.0,
        kernel_s=kernel_s,
        scatter_s=scatter_s,
        loop_s=loop_s,
        pack_bytes=0,
        kernel_calls=batched_kernel_calls + loop_kernel_calls,
        batched_kernel_calls=batched_kernel_calls,
        loop_kernel_calls=loop_kernel_calls,
        bucket_execution_profiles=tuple(bucket_profiles),
    ).to_dict()
    profile["reused_pack_bytes"] = int(reused_pack_bytes)
    profile["batched_matmul_provider"] = (
        batched_matmul_provider or _xp_matmul_provider_name(xp)
        if batched_kernel_calls
        else None
    )
    return [results_by_task[id(task)] for task in plan.tasks], profile


def _grouped_gemm_bucket_execution_profile(
    group,
    *,
    xp,
    execution,
    pack_strategy,
    pack_s,
    kernel_s,
    scatter_s,
    pack_bytes,
    kernel_calls,
    selection_reason,
    fallback_reason,
    batched_matmul_provider,
):
    (
        dtype_a,
        dtype_b,
        batch_shape,
        m,
        n,
        k,
        trans_a,
        trans_b,
        conj_a,
        conj_b,
    ) = gemm_task_key(group[0], xp=xp)
    return {
        "dtype_a": dtype_a,
        "dtype_b": dtype_b,
        "batch_shape": [int(dim) for dim in batch_shape],
        "m": int(m),
        "n": int(n),
        "k": int(k),
        "trans_a": bool(trans_a),
        "trans_b": bool(trans_b),
        "conj_a": bool(conj_a),
        "conj_b": bool(conj_b),
        "execution": str(execution),
        "task_count": int(len(group)),
        "pack_strategy": str(pack_strategy),
        "pack_s": float(pack_s),
        "kernel_s": float(kernel_s),
        "scatter_s": float(scatter_s),
        "pack_bytes": int(pack_bytes),
        "kernel_calls": int(kernel_calls),
        "selection_reason": str(selection_reason),
        "fallback_reason": fallback_reason,
        "batched_matmul_provider": (
            None
            if batched_matmul_provider is None
            else str(batched_matmul_provider)
        ),
    }


def grouped_gemm_fallback(tasks, *, xp=None, pack_threshold=4, allow_batched=True, batched_matmul=None):
    return grouped_gemm_bucketed(
        tasks,
        xp=xp,
        pack_threshold=pack_threshold,
        allow_batched=allow_batched,
        batched_matmul=batched_matmul,
    )
