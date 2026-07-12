from dataclasses import FrozenInstanceError
import os

import numpy as np
import pytest

from renormalizer.backend._distributed.context import (
    DistributedContext,
    DistributedRendezvous,
)
from renormalizer.backend._distributed.mesh import DeviceMesh
from renormalizer.backend._distributed.providers import DeviceResidentProvider
from renormalizer.backend._distributed.center import _resolved_execution_budget
from renormalizer.backend._distributed.residency import MemoryBudgetResolution
from renormalizer.backend.config import DistributedExecutionConfig
from renormalizer.backend.distributed_runtime import CupyDistributedRuntime
from renormalizer.utils.configs import EvolveConfig


class _Collective:
    rank = 0
    size = 1

    def __init__(self):
        self.calls = []

    def allreduce(self, value, *, op="sum"):
        self.calls.append((np.array(value, copy=True), op))
        return np.array(value, copy=True)

    def close(self):
        return None


class _FailClosedCollective(_Collective):
    size = 2

    def allreduce(self, value, *, op="sum"):
        copied = np.array(value, copy=True)
        self.calls.append((copied, op))
        return copied


class _Backend:
    name = "cupy"
    device = "cuda:0"

    class Config:
        precision = 64

    config = Config()

    @staticmethod
    def asarray(value, dtype=None):
        return np.asarray(value, dtype=dtype)


def _runtime():
    context = DistributedContext(0, 0, 1, 1)
    return CupyDistributedRuntime(
        backend=_Backend(),
        context=context,
        rendezvous=DistributedRendezvous("127.0.0.1", 2345),
        mesh=DeviceMesh((1,), ("rank",), 0),
        collective=_Collective(),
    )


def _two_rank_runtime(collective):
    context = DistributedContext(0, 0, 2, 2)
    return CupyDistributedRuntime(
        backend=_Backend(),
        context=context,
        rendezvous=DistributedRendezvous("127.0.0.1", 2345),
        mesh=DeviceMesh((2,), ("rank",), 0),
        collective=collective,
    )


def test_explicit_budgets_are_exact_and_do_not_query_availability(monkeypatch):
    runtime = _runtime()
    monkeypatch.setattr(
        runtime, "_synchronize_active_backend", lambda: ("cupy", "cuda:0", 64)
    )
    monkeypatch.setattr(
        "renormalizer.backend.distributed_runtime._device_available_bytes",
        lambda backend: (_ for _ in ()).throw(AssertionError("device queried")),
    )
    monkeypatch.setattr(
        "renormalizer.backend.distributed_runtime._host_available_bytes",
        lambda: (_ for _ in ()).throw(AssertionError("host queried")),
    )

    config = runtime.execution_config(
        device_memory_budget_bytes=1234,
        host_memory_budget_bytes=5678,
    )

    assert config.device_memory_budget_bytes == 1234
    assert config.host_memory_budget_bytes == 5678
    assert config.resolved_device_memory_budget_bytes == 1234
    assert config.resolved_host_memory_budget_bytes == 5678
    assert config.device_memory_budget_source == "explicit"
    assert config.host_memory_budget_source == "explicit"
    assert config.device_available_snapshot_bytes is None
    assert config.host_available_snapshot_bytes is None


def test_auto_budgets_query_once_use_integer_ratios_and_reuse_snapshot(monkeypatch):
    runtime = _runtime()
    monkeypatch.setattr(
        runtime, "_synchronize_active_backend", lambda: ("cupy", "cuda:0", 64)
    )
    calls = {"device": 0, "host": 0}

    def device(_backend):
        calls["device"] += 1
        return 10_003

    def host():
        calls["host"] += 1
        return 20_007

    monkeypatch.setattr(
        "renormalizer.backend.distributed_runtime._device_available_bytes", device
    )
    monkeypatch.setattr(
        "renormalizer.backend.distributed_runtime._host_available_bytes", host
    )

    first = runtime.execution_config()
    second = runtime.execution_config()

    assert calls == {"device": 1, "host": 1}
    assert first.device_memory_budget_bytes is None
    assert first.host_memory_budget_bytes is None
    assert first.resolved_device_memory_budget_bytes == 10_003 * 85 // 100
    assert first.resolved_host_memory_budget_bytes == 20_007 * 80 // 100
    assert first.device_available_snapshot_bytes == 10_003
    assert first.host_available_snapshot_bytes == 20_007
    assert first.device_memory_budget_source == "auto"
    assert first.host_memory_budget_source == "auto"
    assert first.device_budget_resolution is second.device_budget_resolution
    assert first.host_budget_resolution is second.host_budget_resolution


def test_wave7_budget_resolution_binds_resource_and_exact_auto_relation():
    device = MemoryBudgetResolution(None, 8_502, "auto", 10_003, "device")
    host = MemoryBudgetResolution(None, 16_005, "auto", 20_007, "host")

    assert device.resource == "device"
    assert host.resource == "host"
    with pytest.raises(ValueError, match="auto.*device"):
        MemoryBudgetResolution(None, 8_501, "auto", 10_003, "device")
    with pytest.raises(ValueError, match="auto.*host"):
        MemoryBudgetResolution(None, 16_004, "auto", 20_007, "host")
    with pytest.raises((TypeError, ValueError), match="resource"):
        MemoryBudgetResolution(None, 8_502, "auto", 10_003)


def test_wave7_explicit_budget_resource_is_bound_without_changing_value():
    device = MemoryBudgetResolution(4096, 4096, "explicit", None, "device")
    host = MemoryBudgetResolution(4096, 4096, "explicit", None, "host")

    assert device.resolved_bytes == host.resolved_bytes == 4096
    assert device != host


def test_wave8_budget_resolution_rejects_wrong_or_missing_resource():
    runtime = _runtime()
    device = MemoryBudgetResolution(4096, 4096, "explicit", None, "device")
    host = MemoryBudgetResolution(8192, 8192, "explicit", None, "host")

    with pytest.raises(ValueError, match="resource"):
        MemoryBudgetResolution(4096, 4096, "explicit", None, None)
    with pytest.raises(ValueError, match="device.*resource|resource.*device"):
        DistributedExecutionConfig(
            context=runtime.context,
            mesh=runtime.mesh,
            collective=runtime.collective,
            provider=DeviceResidentProvider(),
            device_memory_budget_bytes=host.requested_bytes,
            host_memory_budget_bytes=device.requested_bytes,
            device_budget_resolution=host,
            host_budget_resolution=device,
        )


def test_mixed_origins_query_only_auto_resource_and_are_frozen(monkeypatch):
    runtime = _runtime()
    monkeypatch.setattr(
        runtime, "_synchronize_active_backend", lambda: ("cupy", "cuda:0", 64)
    )
    monkeypatch.setattr(
        "renormalizer.backend.distributed_runtime._device_available_bytes",
        lambda backend: (_ for _ in ()).throw(AssertionError("device queried")),
    )
    monkeypatch.setattr(
        "renormalizer.backend.distributed_runtime._host_available_bytes", lambda: 4096
    )

    config = runtime.execution_config(
        device_memory_budget_bytes=2048,
        host_memory_budget_bytes=None,
    )

    assert config.device_memory_budget_source == "explicit"
    assert config.host_memory_budget_source == "auto"
    assert config.resolved_host_memory_budget_bytes == 4096 * 80 // 100
    with pytest.raises(FrozenInstanceError):
        config.host_budget_resolution = None

    assert _resolved_execution_budget(config, "device") == 2048
    assert _resolved_execution_budget(config, "host") == 4096 * 80 // 100


def test_direct_device_resident_config_may_remain_unresolved_for_compatibility():
    runtime = _runtime()

    config = DistributedExecutionConfig(
        context=runtime.context,
        mesh=runtime.mesh,
        collective=runtime.collective,
        provider=DeviceResidentProvider(),
    )

    assert config.device_budget_resolution is None
    assert config.host_budget_resolution is None
    assert config.resolved_device_memory_budget_bytes is None
    assert config.resolved_host_memory_budget_bytes is None


def test_active_factory_config_requires_both_frozen_budget_resolutions():
    class FactoryProvider:
        residency_policy = "active_working_set"
        provider_role = "factory"

        @staticmethod
        def acquire(request):
            raise RuntimeError("factory")

        @staticmethod
        def open_working_set(request, plan, store, receipt):
            raise AssertionError("not opened by config validation")

    runtime = _runtime()
    device = MemoryBudgetResolution(1024, 1024, "explicit", None, "device")
    host = MemoryBudgetResolution(2048, 2048, "explicit", None, "host")
    kwargs = {
        "context": runtime.context,
        "mesh": runtime.mesh,
        "collective": runtime.collective,
        "provider": FactoryProvider(),
        "residency_policy": "active_working_set",
        "device_memory_budget_bytes": 1024,
        "host_memory_budget_bytes": 2048,
    }

    with pytest.raises(ValueError, match="resolved device and host budgets"):
        DistributedExecutionConfig(**kwargs)

    config = DistributedExecutionConfig(
        **kwargs,
        device_budget_resolution=device,
        host_budget_resolution=host,
    )
    assert config.provider.provider_role == "factory"


def test_auto_query_failure_is_stable_preflight_failure(monkeypatch):
    runtime = _runtime()
    monkeypatch.setattr(
        runtime, "_synchronize_active_backend", lambda: ("cupy", "cuda:0", 64)
    )
    monkeypatch.setattr(
        "renormalizer.backend.distributed_runtime._device_available_bytes",
        lambda backend: 0,
    )
    monkeypatch.setattr(
        "renormalizer.backend.distributed_runtime._host_available_bytes", lambda: 4096
    )

    with pytest.raises(RuntimeError, match="device memory availability"):
        runtime.execution_config()


@pytest.mark.parametrize("requested", [0, np.iinfo(np.int64).max + 1])
def test_invalid_explicit_budget_completes_fixed_fail_closed_schedule(requested):
    collective = _FailClosedCollective()
    runtime = _two_rank_runtime(collective)

    with pytest.raises(ValueError, match="device memory budget request validation"):
        runtime._requested_budget_agrees(requested, "device")

    assert [(value.dtype, value.shape, op) for value, op in collective.calls] == [
        (np.dtype(np.int32), (1,), "max"),
        (np.dtype(np.int64), (1,), "min"),
        (np.dtype(np.int64), (1,), "max"),
    ]


def test_stable_config_display_includes_resolved_origins_without_handles(monkeypatch):
    runtime = _runtime()
    monkeypatch.setattr(
        runtime, "_synchronize_active_backend", lambda: ("cupy", "cuda:0", 64)
    )
    config = runtime.execution_config(
        device_memory_budget_bytes=1024,
        host_memory_budget_bytes=2048,
    )
    evolve = EvolveConfig()
    evolve.distributed_execution = config

    display = str(evolve)

    assert "device_memory_budget_source='explicit'" in display
    assert "resolved_device_memory_budget_bytes=1024" in display
    assert "host_memory_budget_source='explicit'" in display
    assert "resolved_host_memory_budget_bytes=2048" in display
    assert "provider" not in display
    assert "collective" not in display
    assert "object at" not in display


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2",
    reason="requires the exact two-rank torchrun command",
)
def test_real_two_rank_auto_budget_resolution_is_synchronized_and_cached():
    cupy = pytest.importorskip("cupy")
    from renormalizer import set_backend
    from renormalizer.backend.distributed_runtime import (
        create_cupy_distributed_runtime,
    )

    runtime = create_cupy_distributed_runtime(expected_world_size=2)
    try:
        set_backend(
            "cupy",
            device="cuda:{}".format(runtime.local_rank),
            precision=64,
        )
        first = runtime.execution_config()
        second = runtime.execution_config()
        assert first.device_budget_resolution is second.device_budget_resolution
        assert first.host_budget_resolution is second.host_budget_resolution
        values = cupy.asarray(
            [
                first.device_available_snapshot_bytes,
                first.resolved_device_memory_budget_bytes,
                first.host_available_snapshot_bytes,
                first.resolved_host_memory_budget_bytes,
            ],
            dtype=cupy.int64,
        )
        minimum = runtime.collective.allreduce(values, op="min")
        maximum = runtime.collective.allreduce(values, op="max")
        cupy.testing.assert_array_equal(minimum, maximum)
        assert first.resolved_device_memory_budget_bytes == (
            first.device_available_snapshot_bytes * 85 // 100
        )
        assert first.resolved_host_memory_budget_bytes == (
            first.host_available_snapshot_bytes * 80 // 100
        )
    finally:
        runtime.close()


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2",
    reason="requires the exact two-rank torchrun command",
)
def test_real_two_rank_invalid_explicit_budgets_fail_closed_without_desynchronizing():
    pytest.importorskip("cupy")
    from renormalizer import set_backend
    from renormalizer.backend.distributed_runtime import (
        create_cupy_distributed_runtime,
    )

    runtime = create_cupy_distributed_runtime(expected_world_size=2)
    try:
        set_backend(
            "cupy",
            device="cuda:{}".format(runtime.local_rank),
            precision=64,
        )
        invalid_requests = (
            0 if runtime.rank == 0 else 1024,
            np.iinfo(np.int64).max + 1 if runtime.rank == 0 else 1024,
        )
        for requested in invalid_requests:
            with pytest.raises(
                ValueError, match="device memory budget request validation"
            ):
                runtime.execution_config(
                    device_memory_budget_bytes=requested,
                    host_memory_budget_bytes=1024,
                )
            runtime.barrier()
    finally:
        runtime.close()
