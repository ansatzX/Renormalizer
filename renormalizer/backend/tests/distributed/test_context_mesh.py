from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from renormalizer.backend._distributed.collectives import (
    Collective,
    SingleProcessCollective,
)
from renormalizer.backend._distributed.context import DistributedContext
from renormalizer.backend._distributed.mesh import DeviceMesh


def test_context_keeps_global_and_local_rank_distinct():
    context = DistributedContext(
        rank=9, local_rank=1, world_size=16, local_world_size=8
    )

    assert context.rank == 9
    assert context.local_rank == 1


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"rank": True}, "rank must be an integer"),
        ({"world_size": 0}, "world_size must be positive"),
        ({"rank": 2}, "rank must be smaller than world_size"),
        ({"local_world_size": 0}, "local_world_size must be positive"),
        ({"local_rank": 2}, "local_rank must be smaller than local_world_size"),
        ({"local_world_size": 3}, "local_world_size must not exceed world_size"),
    ],
)
def test_context_validates_rank_and_world_relationships(kwargs, message):
    values = {"rank": 1, "local_rank": 1, "world_size": 2, "local_world_size": 2}
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError), match=message):
        DistributedContext(**values)


def test_context_accepts_nondivisible_global_and_local_world_sizes():
    context = DistributedContext(rank=5, local_rank=3, world_size=6, local_world_size=4)

    assert context.rank == 5
    assert context.local_rank == 3


def test_context_parses_launcher_environment_without_equating_ranks():
    context = DistributedContext.from_environ(
        {
            "RANK": "9",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "16",
            "LOCAL_WORLD_SIZE": "8",
        }
    )

    assert context == DistributedContext(9, 1, 16, 8)


@pytest.mark.parametrize(
    "environ, message",
    [
        ({}, "missing distributed environment variable RANK"),
        (
            {
                "RANK": "zero",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
                "LOCAL_WORLD_SIZE": "1",
            },
            "RANK must be an integer",
        ),
    ],
)
def test_context_rejects_missing_or_malformed_environment(environ, message):
    with pytest.raises(ValueError, match=message):
        DistributedContext.from_environ(environ)


def test_rendezvous_precedence_is_explicit_then_cupyx_then_torchrun():
    from renormalizer.backend._distributed.context import DistributedRendezvous

    environ = {
        "CUPYX_DISTRIBUTED_HOST": "cupyx-host",
        "CUPYX_DISTRIBUTED_PORT": "24567",
        "MASTER_ADDR": "master-host",
        "MASTER_PORT": "29500",
    }

    explicit = DistributedRendezvous.from_environ(
        environ, host="explicit-host", port=25567
    )
    cupyx_override = DistributedRendezvous.from_environ(environ)
    torchrun_fallback = DistributedRendezvous.from_environ(
        {"MASTER_ADDR": "master-host", "MASTER_PORT": "29500"}
    )

    assert explicit == DistributedRendezvous("explicit-host", 25567)
    assert cupyx_override == DistributedRendezvous("cupyx-host", 24567)
    assert torchrun_fallback == DistributedRendezvous("master-host", 30500)
    with pytest.raises(FrozenInstanceError):
        torchrun_fallback.port = 12345


def test_master_port_fallback_avoids_cupy_default_port():
    from renormalizer.backend._distributed.context import DistributedRendezvous

    rendezvous = DistributedRendezvous.from_environ(
        {"MASTER_ADDR": "master-host", "MASTER_PORT": "12333"}
    )

    assert rendezvous.port == 13334
    assert rendezvous.port not in {12333, 13333}


@pytest.mark.parametrize("master_port", [1, 1023, 1024, 64535, 64536, 65535])
def test_master_port_fallback_is_deterministic_and_range_valid(master_port):
    from renormalizer.backend._distributed.context import DistributedRendezvous

    environ = {"MASTER_ADDR": "master-host", "MASTER_PORT": str(master_port)}

    first = DistributedRendezvous.from_environ(environ)
    second = DistributedRendezvous.from_environ(environ)

    assert first == second
    assert 1 <= first.port <= 65535
    assert first.port not in {master_port, 13333}


@pytest.mark.parametrize(
    "environ, options, message",
    [
        ({"MASTER_PORT": "29500"}, {}, "distributed rendezvous host is missing"),
        ({"MASTER_ADDR": "host"}, {}, "distributed rendezvous port is missing"),
        (
            {"MASTER_ADDR": "host", "MASTER_PORT": "invalid"},
            {},
            "MASTER_PORT must be an integer",
        ),
        (
            {"MASTER_ADDR": "host", "MASTER_PORT": "0"},
            {},
            "MASTER_PORT must be between 1 and 65535",
        ),
        (
            {"CUPYX_DISTRIBUTED_HOST": "host", "CUPYX_DISTRIBUTED_PORT": "0"},
            {},
            "CUPYX_DISTRIBUTED_PORT must be between 1 and 65535",
        ),
        (
            {"MASTER_ADDR": "host", "MASTER_PORT": "29500"},
            {"host": " "},
            "host must be a non-empty string",
        ),
        (
            {"MASTER_ADDR": "host", "MASTER_PORT": "29500"},
            {"port": True},
            "port must be an integer between 1 and 65535",
        ),
    ],
)
def test_rendezvous_rejects_invalid_values(environ, options, message):
    from renormalizer.backend._distributed.context import DistributedRendezvous

    with pytest.raises((TypeError, ValueError), match=message):
        DistributedRendezvous.from_environ(environ, **options)


def test_mesh_axis_grouping_is_deterministic():
    mesh = DeviceMesh(shape=(2, 4), axis_names=("node", "gpu"), rank=5)

    assert mesh.axis_group_ranks("gpu") == (4, 5, 6, 7)
    assert mesh.axis_group_ranks("node") == (1, 5)


def test_mesh_axis_grouping_supports_arbitrary_dimensions():
    mesh = DeviceMesh(shape=(2, 3, 4), axis_names=("x", "y", "z"), rank=18)

    assert mesh.coordinates == (1, 1, 2)
    assert mesh.axis_group_ranks("x") == (6, 18)
    assert mesh.axis_group_ranks("y") == (14, 18, 22)
    assert mesh.axis_group_ranks("z") == (16, 17, 18, 19)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"shape": ()}, "shape must not be empty"),
        ({"shape": (2, 0)}, "shape dimensions must be positive integers"),
        ({"axis_names": ("x",)}, "axis_names must match shape"),
        ({"axis_names": ("x", "x")}, "axis_names must be unique"),
        ({"rank": 8}, "rank must be smaller than mesh size"),
    ],
)
def test_mesh_validates_shape_axis_names_and_rank(kwargs, message):
    values = {"shape": (2, 4), "axis_names": ("x", "y"), "rank": 0}
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError), match=message):
        DeviceMesh(**values)


def test_mesh_rejects_unknown_axis_before_grouping():
    mesh = DeviceMesh(shape=(2, 2), axis_names=("x", "y"), rank=0)

    with pytest.raises(ValueError, match="unknown mesh axis"):
        mesh.axis_group_ranks("missing")


def test_single_process_collective_has_precise_copy_and_mutation_semantics():
    collective = SingleProcessCollective()
    source = np.arange(6).reshape(2, 3)

    broadcast = collective.broadcast(source, root=0)
    reduced = collective.allreduce(source)
    gathered = collective.allgather(source, axis=1)
    scattered = collective.reduce_scatter(source, axis=0)

    assert isinstance(collective, Collective)
    assert collective.rank == 0
    assert collective.size == 1
    assert collective.barrier() is None
    assert broadcast is source
    for result in (reduced, gathered, scattered):
        np.testing.assert_array_equal(result, source)
        assert result is not source
    assert collective.close() is None
    assert collective.close() is None


def test_single_process_inplace_allreduce_reuses_control_storage():
    collective = SingleProcessCollective()
    status = np.asarray([1], dtype=np.int32)

    result = collective.allreduce_inplace(status, op="max")

    assert result is status
    np.testing.assert_array_equal(status, [1])


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda c, a: c.broadcast(a, root=1), "root must be zero"),
        (lambda c, a: c.allreduce(a, op="mean"), "unsupported reduction op"),
        (lambda c, a: c.allgather(a, axis=2), "axis 2 is out of range"),
        (lambda c, a: c.reduce_scatter(a, axis=-3), "axis -3 is out of range"),
    ],
)
def test_single_process_collective_validates_arguments(call, message):
    collective = SingleProcessCollective()
    source = np.ones((2, 3))

    with pytest.raises(ValueError, match=message):
        call(collective, source)
