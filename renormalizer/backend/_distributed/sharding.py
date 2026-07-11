"""Immutable axis-sharding metadata for distributed execution."""

from dataclasses import dataclass


def _normalize_global_shape(global_shape):
    try:
        shape = tuple(global_shape)
    except TypeError as error:
        raise TypeError("global_shape must be an iterable") from error
    if not shape:
        raise ValueError("global_shape must not be empty")
    if any(type(dimension) is not int or dimension <= 0 for dimension in shape):
        raise ValueError("global_shape dimensions must be positive integers")
    return shape


def _normalize_axis(axis, rank):
    if type(axis) is not int:
        raise TypeError("axis must be an integer")
    if axis < -rank or axis >= rank:
        raise ValueError("axis is out of range for global_shape")
    return axis % rank


def _is_full_slice(value):
    return isinstance(value, slice) and value == slice(None)


@dataclass(frozen=True)
class ShardingSpec:
    global_shape: tuple[int, ...]
    axis: int
    local_slices: tuple[tuple[slice, ...], ...]

    def __post_init__(self):
        shape = _normalize_global_shape(self.global_shape)
        axis = _normalize_axis(self.axis, len(shape))
        try:
            local_slices = tuple(
                tuple(local_slice) for local_slice in self.local_slices
            )
        except TypeError as error:
            raise TypeError("local_slices must contain iterable slices") from error
        if not local_slices:
            raise ValueError("local_slices must not be empty")
        for local_slice in local_slices:
            if len(local_slice) != len(shape):
                raise ValueError("local slice rank must match global_shape")
            for slice_axis, value in enumerate(local_slice):
                if not isinstance(value, slice):
                    raise TypeError("local_slices must contain slice objects")
                if slice_axis != axis and not _is_full_slice(value):
                    raise ValueError("non-sharded axes must use full slices")
        object.__setattr__(self, "global_shape", shape)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "local_slices", local_slices)
        if not self.coverage_is_complete():
            raise ValueError(
                "local slices must provide complete non-overlapping coverage"
            )

    @property
    def parts(self):
        return len(self.local_slices)

    @property
    def max_local_shape(self):
        return max(
            (self.local_shape(rank) for rank in range(self.parts)),
            key=lambda shape: tuple(shape),
        )

    @property
    def max_local_elements(self):
        elements = 1
        for dimension in self.max_local_shape:
            elements *= dimension
        return elements

    def local_shape(self, rank):
        if type(rank) is not int or rank < 0 or rank >= self.parts:
            raise ValueError("rank is out of range for sharding parts")
        selected = self.local_slices[rank][self.axis]
        return tuple(
            selected.stop - selected.start if index == self.axis else dimension
            for index, dimension in enumerate(self.global_shape)
        )

    def coverage_is_complete(self):
        expected_start = 0
        extent = self.global_shape[self.axis]
        for local_slice in self.local_slices:
            selected = local_slice[self.axis]
            if selected.step not in (None, 1):
                return False
            if type(selected.start) is not int or type(selected.stop) is not int:
                return False
            if selected.start != expected_start or selected.stop <= selected.start:
                return False
            if selected.stop > extent:
                return False
            expected_start = selected.stop
        return expected_start == extent


def shard_axis(global_shape, axis, parts):
    shape = _normalize_global_shape(global_shape)
    normalized_axis = _normalize_axis(axis, len(shape))
    if type(parts) is not int:
        raise TypeError("parts must be an integer")
    if parts <= 0:
        raise ValueError("parts must be positive")
    extent = shape[normalized_axis]
    if extent < parts:
        raise ValueError("sharded axis extent is smaller than parts")
    quotient, remainder = divmod(extent, parts)
    local_slices = []
    start = 0
    for rank in range(parts):
        stop = start + quotient + (1 if rank < remainder else 0)
        selected = [slice(None)] * len(shape)
        selected[normalized_axis] = slice(start, stop)
        local_slices.append(tuple(selected))
        start = stop
    return ShardingSpec(shape, normalized_axis, tuple(local_slices))


@dataclass(frozen=True)
class DistributedTensor:
    spec: ShardingSpec
    rank: int
    local_array: object

    def __post_init__(self):
        if not isinstance(self.spec, ShardingSpec):
            raise TypeError("spec must be a ShardingSpec")
        expected_shape = self.spec.local_shape(self.rank)
        try:
            actual_shape = tuple(self.local_array.shape)
        except AttributeError as error:
            raise TypeError("local_array must expose shape metadata") from error
        if actual_shape != expected_shape:
            raise ValueError(
                "local array shape {} does not match shard shape {}".format(
                    actual_shape, expected_shape
                )
            )

    @property
    def global_shape(self):
        return self.spec.global_shape


__all__ = ["DistributedTensor", "ShardingSpec", "shard_axis"]
