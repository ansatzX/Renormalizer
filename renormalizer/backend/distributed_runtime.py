# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""Explicit lifecycle for launcher-configured CuPy distributed execution."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
import os

import numpy as np

from renormalizer.backend._distributed.context import (
    DistributedContext,
    DistributedRendezvous,
)
from renormalizer.backend._distributed.center import (
    normalize_distributed_backend_metadata,
)
from renormalizer.backend._distributed.mesh import DeviceMesh
from renormalizer.backend._distributed.providers import DeviceResidentProvider
from renormalizer.backend._distributed.residency import MemoryBudgetResolution
from renormalizer.backend.config import BackendConfig, DistributedExecutionConfig
from renormalizer.backend.factory import create_backend


def _device_available_bytes(backend):
    cupy = getattr(backend, "_cupy", None)
    if cupy is None:
        namespace = getattr(backend, "array_namespace", None)
        if getattr(namespace, "__name__", None) == "cupy":
            cupy = namespace
    if cupy is None or not hasattr(cupy, "cuda"):
        raise RuntimeError("CuPy device memory availability is unavailable")
    device_index = getattr(backend, "_device_index", None)
    if device_index is None:
        device = str(getattr(backend, "device", ""))
        if not device.startswith("cuda:"):
            raise RuntimeError("CuPy device memory availability is unavailable")
        device_index = int(device.split(":", 1)[1])
    with cupy.cuda.Device(int(device_index)):
        free_bytes, _ = cupy.cuda.runtime.memGetInfo()
    return int(free_bytes)


def _host_available_bytes():
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError("host memory availability requires psutil") from error
    return int(psutil.virtual_memory().available)


@dataclass
class CupyDistributedRuntime:
    backend: object
    context: DistributedContext
    rendezvous: DistributedRendezvous
    mesh: DeviceMesh
    collective: object
    _closed: bool = False
    _auto_device_budget: object = field(default=None, init=False, repr=False)
    _auto_host_budget: object = field(default=None, init=False, repr=False)

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

    def execution_config(
        self,
        *,
        residency_policy="device_resident",
        device_memory_budget_bytes=None,
        host_memory_budget_bytes=None,
        prefetch_depth=1,
    ):
        if self._closed:
            raise RuntimeError("distributed runtime is closed")
        backend_metadata = self._synchronize_active_backend()
        device_resolution = self._resolve_budget(
            "device", device_memory_budget_bytes
        )
        host_resolution = self._resolve_budget("host", host_memory_budget_bytes)
        return DistributedExecutionConfig(
            context=self.context,
            mesh=self.mesh,
            collective=self.collective,
            provider=DeviceResidentProvider(),
            residency_policy=residency_policy,
            device_memory_budget_bytes=device_memory_budget_bytes,
            host_memory_budget_bytes=host_memory_budget_bytes,
            prefetch_depth=prefetch_depth,
            backend_name=backend_metadata[0] if backend_metadata is not None else None,
            backend_device=backend_metadata[1] if backend_metadata is not None else None,
            backend_precision=backend_metadata[2] if backend_metadata is not None else None,
            device_budget_resolution=device_resolution,
            host_budget_resolution=host_resolution,
        )

    def _control_array(self, values, dtype):
        converter = getattr(self.backend, "asarray", None)
        if callable(converter):
            return converter(values, dtype=dtype)
        return np.asarray(values, dtype=dtype)

    @staticmethod
    def _host_control(value):
        getter = getattr(value, "get", None)
        if callable(getter):
            value = getter()
        return np.asarray(value)

    def _requested_budget_agrees(self, requested, resource):
        local_error = None
        encoded = 0
        try:
            if requested is not None and (
                type(requested) is not int or requested <= 0
            ):
                raise ValueError(
                    "{}_memory_budget_bytes must be a positive integer or None".format(
                        resource
                    )
                )
            encoded = -1 if requested is None else requested
            if encoded > np.iinfo(np.int64).max:
                raise ValueError(
                    "requested memory budget exceeds supported integer range"
                )
        except BaseException as error:
            local_error = error
            encoded = 0

        if self.world_size == 1:
            if local_error is not None:
                raise local_error
            return
        status = self._control_array([int(local_error is not None)], np.int32)
        failed = int(
            self._host_control(
                self.collective.allreduce(status, op="max")
            ).reshape(-1)[0]
        )
        control = self._control_array([encoded], np.int64)
        minimum = self._host_control(
            self.collective.allreduce(control, op="min")
        ).reshape(-1)[0]
        maximum = self._host_control(
            self.collective.allreduce(control, op="max")
        ).reshape(-1)[0]
        if failed:
            raise ValueError(
                "{} memory budget request validation failed".format(resource)
            ) from local_error
        if int(minimum) != int(maximum):
            raise RuntimeError(
                "{} memory budget request disagreement".format(resource)
            )

    def _auto_available_snapshot(self, resource):
        cached_name = "_auto_{}_budget".format(resource)
        cached = getattr(self, cached_name)
        if cached is not None:
            return cached
        local_error = None
        available = 0
        try:
            available = (
                _device_available_bytes(self.backend)
                if resource == "device"
                else _host_available_bytes()
            )
            if type(available) is not int or available <= 0:
                raise ValueError("availability snapshot must be positive")
            if available > np.iinfo(np.int64).max:
                raise OverflowError("availability snapshot exceeds int64")
        except BaseException as error:
            local_error = error

        if self.world_size == 1:
            failed = int(local_error is not None)
        else:
            status = self._control_array([int(local_error is not None)], np.int32)
            failed = int(
                self._host_control(
                    self.collective.allreduce(status, op="max")
                ).reshape(-1)[0]
            )
        if failed:
            raise RuntimeError(
                "{} memory availability preflight failed".format(resource)
            ) from local_error
        if self.world_size > 1:
            control = self._control_array([available], np.int64)
            available = int(
                self._host_control(
                    self.collective.allreduce(control, op="min")
                ).reshape(-1)[0]
            )
        ratio = 85 if resource == "device" else 80
        resolved = available * ratio // 100
        if resolved <= 0:
            raise RuntimeError(
                "{} memory availability resolved to a nonpositive budget".format(
                    resource
                )
            )
        resolution = MemoryBudgetResolution(
            requested_bytes=None,
            resolved_bytes=resolved,
            source="auto",
            available_snapshot_bytes=available,
            resource=resource,
        )
        setattr(self, cached_name, resolution)
        return resolution

    def _resolve_budget(self, resource, requested):
        self._requested_budget_agrees(requested, resource)
        if requested is not None:
            return MemoryBudgetResolution(
                requested_bytes=requested,
                resolved_bytes=requested,
                source="explicit",
                available_snapshot_bytes=None,
                resource=resource,
            )
        return self._auto_available_snapshot(resource)

    def _synchronize_active_backend(self):
        expected_name = getattr(self.backend, "name", None)
        expected_device = getattr(self.backend, "device", None)
        expected_config = getattr(self.backend, "config", None)
        expected_precision = getattr(expected_config, "precision", None)
        if None in (expected_name, expected_device, expected_precision):
            return None

        from renormalizer.cons import get_backend

        local_error = None
        active_metadata = (None, None, None)
        try:
            active = get_backend()
            active_metadata = (
                str(active.name),
                str(active.device),
                int(active.config.precision),
            )
            expected = (
                str(expected_name),
                str(expected_device),
                int(expected_precision),
            )
            context_expected = (
                "cupy",
                "cuda:{}".format(self.local_rank),
                int(expected_precision),
            )
            if expected != context_expected or active_metadata != expected:
                raise ValueError(
                    "active backend name/device/precision does not match runtime"
                )
        except BaseException as error:
            local_error = error

        status = self.backend.asarray(
            [int(local_error is not None)], dtype=np.int32
        )
        failed = self.collective.allreduce(status, op="max")
        digest_payload = {
            "active": normalize_distributed_backend_metadata(
                active_metadata, local_rank=self.local_rank
            ),
            "expected": normalize_distributed_backend_metadata(
                (expected_name, expected_device, expected_precision),
                local_rank=self.local_rank,
            ),
            "local_error": None if local_error is None else type(local_error).__name__,
        }
        encoded = json.dumps(
            digest_payload, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        hexdigest = hashlib.sha256(encoded).hexdigest()
        words = np.asarray(
            [
                int(hexdigest[index : index + 16], 16)
                for index in range(0, 64, 16)
            ],
            dtype=np.uint64,
        )
        control = self.backend.asarray(words, dtype=np.uint64)
        minimum = self.collective.allreduce(control, op="min")
        maximum = self.collective.allreduce(control, op="max")
        host = lambda value: np.asarray(
            value.get() if callable(getattr(value, "get", None)) else value
        )
        if not np.array_equal(host(minimum), host(maximum)):
            raise RuntimeError("distributed runtime backend metadata disagreement")
        if int(host(failed).reshape(-1)[0]):
            raise RuntimeError(
                "distributed runtime backend validation failed"
            ) from local_error
        return (
            str(expected_name),
            str(expected_device),
            int(expected_precision),
        )

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
