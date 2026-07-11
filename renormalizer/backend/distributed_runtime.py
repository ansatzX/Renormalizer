# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""Explicit lifecycle for launcher-configured CuPy distributed execution."""

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os

from renormalizer.backend._distributed.context import (
    DistributedContext,
    DistributedRendezvous,
)
from renormalizer.backend._distributed.mesh import DeviceMesh
from renormalizer.backend.config import BackendConfig
from renormalizer.backend.factory import create_backend


@dataclass
class CupyDistributedRuntime:
    backend: object
    context: DistributedContext
    rendezvous: DistributedRendezvous
    mesh: DeviceMesh
    collective: object
    _closed: bool = False

    @property
    def rank(self):
        return self.context.rank

    @property
    def local_rank(self):
        return self.context.local_rank

    @property
    def world_size(self):
        return self.context.world_size

    def barrier(self):
        return self.collective.barrier()

    def close(self):
        if self._closed:
            return
        self.collective.close()
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("distributed runtime is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _validate_expected_world_size(expected_world_size, actual_world_size):
    if expected_world_size is None:
        return
    if type(expected_world_size) is not int or expected_world_size <= 0:
        raise ValueError("expected_world_size must be a positive integer")
    if actual_world_size != expected_world_size:
        raise ValueError(
            "expected world size {}, got {}".format(
                expected_world_size, actual_world_size
            )
        )


def create_cupy_distributed_runtime(
    *,
    precision=64,
    expected_world_size=None,
    mesh_shape=None,
    axis_names=None,
    environ=None,
    host=None,
    port=None,
):
    environment = dict(os.environ if environ is None else environ)
    context = DistributedContext.from_environ(environment)
    _validate_expected_world_size(expected_world_size, context.world_size)
    rendezvous = DistributedRendezvous.from_environ(environment, host=host, port=port)

    shape = (context.world_size,) if mesh_shape is None else tuple(mesh_shape)
    names = ("rank",) if axis_names is None else tuple(axis_names)
    if math.prod(shape) != context.world_size:
        raise ValueError("mesh size must equal distributed world_size")
    mesh = DeviceMesh(shape=shape, axis_names=names, rank=context.rank)

    backend = create_backend(
        "cupy",
        config=BackendConfig(
            device="cuda:{}".format(context.local_rank), precision=precision
        ),
    )
    collective = backend.create_collective(
        context, host=rendezvous.host, port=rendezvous.port
    )
    return CupyDistributedRuntime(
        backend=backend,
        context=context,
        rendezvous=rendezvous,
        mesh=mesh,
        collective=collective,
    )


@contextmanager
def cupy_distributed_runtime(**kwargs):
    runtime = create_cupy_distributed_runtime(**kwargs)
    try:
        yield runtime
    finally:
        runtime.close()
