from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from renormalizer.backend._distributed.sharding import (
    DistributedTensor,
    ShardingSpec,
    shard_axis,
)


def test_uneven_axis_shards_cover_without_overlap():
    spec = shard_axis(global_shape=(10, 6), axis=0, parts=4)

    assert spec.local_slices == (
        (slice(0, 3), slice(None)),
        (slice(3, 6), slice(None)),
        (slice(6, 8), slice(None)),
        (slice(8, 10), slice(None)),
    )
    assert spec.coverage_is_complete()


def test_sharding_reports_exact_local_shapes_and_is_immutable():
    spec = shard_axis(global_shape=(3, 10, 2), axis=1, parts=4)

    assert tuple(spec.local_shape(rank) for rank in range(4)) == (
        (3, 3, 2),
        (3, 3, 2),
        (3, 2, 2),
        (3, 2, 2),
    )
    assert spec.parts == 4
    with pytest.raises(FrozenInstanceError):
        spec.axis = 0


def test_sharding_reports_deterministic_maximum_local_storage():
    spec = shard_axis(global_shape=(3, 10, 2), axis=1, parts=4)

    assert spec.max_local_shape == (3, 3, 2)
    assert spec.max_local_elements == 18


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"global_shape": (), "axis": 0, "parts": 1}, "global_shape must not be empty"),
        ({"global_shape": (4, 3), "axis": 2, "parts": 1}, "axis is out of range"),
        ({"global_shape": (4, 3), "axis": True, "parts": 1}, "axis must be an integer"),
        ({"global_shape": (4, 3), "axis": 0, "parts": 0}, "parts must be positive"),
        ({"global_shape": (2, 3), "axis": 0, "parts": 3}, "smaller than parts"),
    ],
)
def test_shard_axis_rejects_invalid_or_empty_placements(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        shard_axis(**kwargs)


def test_sharding_spec_rejects_gaps_overlap_and_non_axis_slices():
    with pytest.raises(ValueError, match="complete non-overlapping coverage"):
        ShardingSpec(
            global_shape=(6, 2),
            axis=0,
            local_slices=(
                (slice(0, 4), slice(None)),
                (slice(3, 6), slice(None)),
            ),
        )
    with pytest.raises(ValueError, match="non-sharded axes must use full slices"):
        ShardingSpec(
            global_shape=(6, 2),
            axis=0,
            local_slices=(
                (slice(0, 3), slice(0, 1)),
                (slice(3, 6), slice(None)),
            ),
        )


def test_distributed_tensor_keeps_only_the_rank_local_array():
    spec = shard_axis(global_shape=(5, 3), axis=0, parts=2)
    local = np.arange(9).reshape(3, 3)

    tensor = DistributedTensor(spec=spec, rank=0, local_array=local)

    assert tensor.global_shape == (5, 3)
    assert tensor.local_array is local
    with pytest.raises(ValueError, match="local array shape"):
        DistributedTensor(spec=spec, rank=1, local_array=local)
