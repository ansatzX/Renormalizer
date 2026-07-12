from types import SimpleNamespace

import numpy as np
import pytest

import renormalizer.backend._distributed.center as center_module


class _Collective:
    rank = 0
    size = 2

    def __init__(self):
        self.calls = []

    def allreduce(self, value, *, op="sum"):
        copied = np.array(value, copy=True)
        self.calls.append((copied, op))
        return copied

    def broadcast(self, value, *, root):
        self.calls.append((np.array(value, copy=True), "broadcast"))
        return value


class _Selected:
    name = "numpy"
    array_namespace = np

    @staticmethod
    def asarray(value, dtype=None):
        return np.asarray(value, dtype=dtype)


def _execution(collective, *, residency_policy="device_resident"):
    return SimpleNamespace(
        context=SimpleNamespace(world_size=2, local_world_size=2),
        collective=collective,
        residency_policy=residency_policy,
        resolved_device_memory_budget_bytes=1,
        resolved_host_memory_budget_bytes=1,
        device_memory_budget_bytes=None,
        host_memory_budget_bytes=None,
        device_budget_resolution=None,
        host_budget_resolution=None,
    )


def test_wave8_device_resident_root_fallback_is_unbounded_compatibility(
    monkeypatch,
):
    from renormalizer.utils import profiling

    collective = _Collective()
    execution = _execution(collective)
    counters = {}
    callback_calls = 0
    receive = np.empty(4, dtype=np.float64)

    approval = center_module.preflight_root_fallback_capacity(
        execution,
        _Selected(),
        device_bytes=1 << 80,
        host_bytes=1 << 80,
        counters=counters,
    )
    assert approval is None
    assert collective.calls == []

    def operation():
        nonlocal callback_calls
        callback_calls += 1
        return np.arange(4, dtype=np.float64)

    result = center_module.run_adapter_root_fallback(
        operation,
        execution,
        _Selected(),
        receive,
        estimated_device_bytes=1 << 80,
        estimated_host_bytes=1 << 80,
        counters=counters,
    )

    records = []
    monkeypatch.setattr(profiling, "enabled", lambda: True)
    monkeypatch.setattr(
        profiling,
        "record",
        lambda event, **payload: records.append({"event": event, **payload}),
    )
    center_module.record_fallback_solve(
        execution,
        solver_dtype=result.dtype,
        global_count=result.size,
        packed_qn=False,
        network="mps",
        center_kind="one_site",
        solver="krylov",
        hv_count=1,
        counters=counters,
        synchronization_s=0.0,
    )

    np.testing.assert_array_equal(result, np.arange(4, dtype=np.float64))
    assert callback_calls == 1
    assert counters["allreduce_calls"] == 6
    assert counters["capacity_proven"] is False
    assert "unbounded" in counters["capacity_reason"]
    assert all(
        value.dtype != np.dtype(np.int64)
        for value, operation_name in collective.calls
        if operation_name != "broadcast"
    )
    assert records[0]["capacity_proven"] is False
    assert records[0]["capacity_reason"] == counters["capacity_reason"]

    with pytest.raises(TypeError, match="approvals are unsupported"):
        center_module.run_adapter_root_fallback(
            operation,
            execution,
            _Selected(),
            receive,
            estimated_device_bytes=32,
            estimated_host_bytes=32,
            counters=counters,
            capacity_approval=object(),
        )
    assert callback_calls == 1
