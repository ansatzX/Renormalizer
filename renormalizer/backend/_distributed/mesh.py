"""Deterministic row-major device mesh rank grouping."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceMesh:
    shape: tuple[int, ...]
    axis_names: tuple[str, ...]
    rank: int

    def __post_init__(self):
        shape = tuple(self.shape)
        axis_names = tuple(self.axis_names)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "axis_names", axis_names)
        if not shape:
            raise ValueError("shape must not be empty")
        if any(type(dimension) is not int or dimension <= 0 for dimension in shape):
            raise ValueError("shape dimensions must be positive integers")
        if len(axis_names) != len(shape):
            raise ValueError("axis_names must match shape")
        if any(not isinstance(name, str) or not name for name in axis_names):
            raise ValueError("axis_names must be non-empty strings")
        if len(set(axis_names)) != len(axis_names):
            raise ValueError("axis_names must be unique")
        if type(self.rank) is not int:
            raise TypeError("rank must be an integer")
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        if self.rank >= self.size:
            raise ValueError("rank must be smaller than mesh size")

    @property
    def size(self):
        return math.prod(self.shape)

    @property
    def coordinates(self):
        remainder = self.rank
        result = [0] * len(self.shape)
        for index in range(len(self.shape) - 1, -1, -1):
            remainder, result[index] = divmod(remainder, self.shape[index])
        return tuple(result)

    def axis_group_ranks(self, axis_name):
        try:
            axis = self.axis_names.index(axis_name)
        except ValueError as error:
            raise ValueError("unknown mesh axis {!r}".format(axis_name)) from error
        coordinates = list(self.coordinates)
        strides = [
            math.prod(self.shape[index + 1 :]) for index in range(len(self.shape))
        ]
        ranks = []
        for coordinate in range(self.shape[axis]):
            coordinates[axis] = coordinate
            ranks.append(
                sum(value * stride for value, stride in zip(coordinates, strides))
            )
        return tuple(ranks)
