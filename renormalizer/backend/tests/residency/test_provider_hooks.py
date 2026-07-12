from contextlib import contextmanager
import dataclasses
from dataclasses import replace
import os
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from renormalizer.backend._distributed.collectives import SingleProcessCollective
from renormalizer.backend._distributed.center import (
    CenterVectorMap,
    run_adapter_root_fallback,
)
from renormalizer.backend._distributed.context import DistributedContext
from renormalizer.backend._distributed.local_operator import DistributedLocalOperator
from renormalizer.backend._distributed.mesh import DeviceMesh
from renormalizer.backend._distributed.planner import plan_distributed_execution
from renormalizer.backend._distributed.providers import DeviceResidentProvider
from renormalizer.backend._distributed.providers import active_working_set_policy_error
from renormalizer.backend._distributed.residency import (
    HostTensorAllocation,
    HostTensorStore,
    MemoryBudgetResolution,
    ResidencyBudgetError,
    ResidencyPlanner,
    ResidencyRequest,
    StaleHostTensorRefError,
)
from renormalizer.backend._distributed.solvers import (
    build_krylov_memory_profile,
)
from renormalizer.backend._execution.model import ExecutionBindings, _plan_hash
from renormalizer.backend._execution.planner import lower_einsum_path
from renormalizer.backend._gemm.mps_lowering import build_mps_ir_hop
from renormalizer.backend.config import BackendConfig
from renormalizer.backend.factory import create_backend


def _operator(provider, source_binding):
    source = lower_einsum_path("ab,b->a", ((4, 4), (4,)), dtype="float64")
    plan = plan_distributed_execution(source, variable_key="input_1", world_size=1)
    context = DistributedContext(0, 0, 1, 1)
    backend = create_backend(
        "numpy",
        config=BackendConfig(device="cpu", execution_policy="execution_ir"),
    )
    return DistributedLocalOperator(
        plan=plan,
        provider=provider,
        collective=SingleProcessCollective(),
        counters={},
        backend=backend,
        context=context,
        source_bindings=ExecutionBindings({"input_0": source_binding}),
    )


def _explicit_budget(value, resource):
    return MemoryBudgetResolution(value, value, "explicit", None, resource)


def _execution_plan(*, specialized=True, dtype="float64"):
    plan = lower_einsum_path("ab,b->a", ((8, 8), (8,)), dtype=dtype)
    if not specialized:
        return plan
    return (
        build_mps_ir_hop(
            "ab,b->a",
            (np.ones((8, 8), dtype=dtype),),
            (8,),
            "one_site",
        )
        .resolve_execution_artifact(np.ones(8, dtype=dtype))
        .execution_plan
    )


def _relabeled_execution_plan(dtype="float64"):
    plan = lower_einsum_path("ab,b->a", ((8, 8), (8,)), dtype=dtype)
    override_reason = "wave8 active working set test"
    return replace(
        plan,
        planner_source="specialized",
        override_reason=override_reason,
        plan_hash=_plan_hash(
            plan.operation,
            plan.inputs,
            plan.output,
            plan.steps,
            plan.workspace_bytes,
            "specialized",
            plan.oe_path,
            override_reason,
        ),
    )


def _active_operator(
    provider,
    collective,
    *,
    backend_config=None,
    specialized=True,
    residency_request_transform=None,
    residency_plan_transform=None,
    context=None,
    mesh=None,
):
    source = _execution_plan(specialized=specialized)
    distributed = plan_distributed_execution(
        source, variable_key="input_1", world_size=2
    )
    store = HostTensorStore(store_id="active-run")
    matrix_ref = store.put("input_0", np.ones((8, 8), dtype=np.float64))
    center_ref = store.put("output", np.zeros(8, dtype=np.float64))
    snapshot = store.snapshot()
    output_allocation = HostTensorAllocation("output", (8,), "float64")
    request = ResidencyRequest(
        distributed_plan=distributed,
        host_refs={"input_0": matrix_ref, "output": center_ref},
        world_size=2,
        local_world_size=2,
        backend_name="numpy",
        store_bytes=None,
        external_host_bytes=(0, 0),
        transfer_staging_host_bytes=(0, 0),
        dirty_writeback_bytes=None,
        solver_input_sharding=distributed.input_sharding,
        solver_output_sharding=distributed.input_sharding,
        mapped_local_counts=(4, 4),
        solver_profile=build_krylov_memory_profile(
            distributed.input_sharding,
            "float64",
            max_krylov_vectors=2,
        ),
        materialization_policy="device",
        complete_center_bytes=center_ref.nbytes,
        prefetch_depth=1,
        future_plans=(),
        device_budget=_explicit_budget(1 << 30, "device"),
        host_budget=_explicit_budget(1 << 31, "host"),
        store_snapshots=(snapshot, snapshot),
        writeback_allocations=(
            (output_allocation,),
            (output_allocation,),
        ),
    )
    residency_plan = ResidencyPlanner().plan(request)
    if residency_plan_transform is not None:
        residency_plan = residency_plan_transform(residency_plan)
    operator_request = (
        request
        if residency_request_transform is None
        else residency_request_transform(request)
    )
    backend = create_backend(
        "numpy",
        config=(
            BackendConfig(device="cpu", execution_policy="execution_ir")
            if backend_config is None
            else backend_config
        ),
    )
    operator = DistributedLocalOperator(
        plan=distributed,
        provider=provider,
        collective=collective,
        counters={},
        backend=backend,
        context=(DistributedContext(0, 0, 2, 2) if context is None else context),
        mesh=(DeviceMesh((2,), ("rank",), 0) if mesh is None else mesh),
        source_bindings=ExecutionBindings({"input_0": matrix_ref}),
        residency_request=operator_request,
        residency_plan=residency_plan,
    )
    return operator, store


class _AgreeingCollective:
    rank = 0
    size = 2

    @staticmethod
    def allreduce(value, *, op="sum"):
        return np.array(value, copy=True)

    @staticmethod
    def allreduce_inplace(value, *, op="sum"):
        return value

    @staticmethod
    def broadcast(value, *, root):
        return value


def _real_active_case(
    runtime,
    store_id,
    device_budget,
    *,
    dtype="float64",
    coefficient=1.0,
    qn_mask_present=False,
):
    dtype = np.dtype(dtype)
    source = _execution_plan(specialized=True, dtype=dtype.name)
    distributed = plan_distributed_execution(
        source, variable_key="input_1", world_size=2
    )
    store = HostTensorStore(store_id=store_id)
    matrix_ref = store.put("input_0", np.ones((8, 8), dtype=dtype))
    center_ref = store.put("output", np.zeros(8, dtype=dtype))
    snapshot = store.snapshot()
    solver_profile = build_krylov_memory_profile(
        distributed.input_sharding,
        dtype,
        coefficient=coefficient,
        max_krylov_vectors=2,
    )
    output_allocation = HostTensorAllocation(
        "output", (8,), solver_profile.result_dtype
    )
    mask_identity = None
    if qn_mask_present:
        mask_identity = CenterVectorMap(
            distributed.input_sharding,
            distributed.output_sharding,
            np.ones(8, dtype=bool),
        ).qn_mask_identity
    request = ResidencyRequest(
        distributed_plan=distributed,
        host_refs={"input_0": matrix_ref, "output": center_ref},
        world_size=2,
        local_world_size=2,
        backend_name=runtime.backend.name,
        store_bytes=None,
        external_host_bytes=(0, 0),
        transfer_staging_host_bytes=(0, 0),
        dirty_writeback_bytes=None,
        solver_input_sharding=distributed.input_sharding,
        solver_output_sharding=distributed.input_sharding,
        mapped_local_counts=(4, 4),
        solver_profile=solver_profile,
        materialization_policy="device",
        complete_center_bytes=None,
        prefetch_depth=1,
        future_plans=(),
        device_budget=_explicit_budget(device_budget, "device"),
        host_budget=_explicit_budget(1 << 31, "host"),
        store_snapshots=(snapshot, snapshot),
        writeback_allocations=(
            (output_allocation,),
            (output_allocation,),
        ),
        qn_mask_present=qn_mask_present,
        qn_mask_identity=mask_identity,
    )
    try:
        residency_plan = ResidencyPlanner().plan(request)
    except BaseException:
        store.close()
        raise
    return distributed, store, matrix_ref, request, residency_plan


def _real_active_operator(
    runtime,
    provider,
    distributed,
    matrix_ref,
    residency_request,
    residency_plan,
):
    backend = create_backend(
        "cupy",
        config=replace(
            runtime.backend.config,
            execution_policy="execution_ir",
            fallback_policy="error",
        ),
    )
    return DistributedLocalOperator(
        plan=distributed,
        provider=provider,
        collective=runtime.collective,
        counters={},
        backend=backend,
        context=runtime.context,
        mesh=runtime.mesh,
        source_bindings=ExecutionBindings({"input_0": matrix_ref}),
        residency_request=residency_request,
        residency_plan=residency_plan,
    )


def test_device_resident_provider_uses_provider_neutral_validation_hooks():
    class RecordingProvider(DeviceResidentProvider):
        def __init__(self):
            self.events = []

        def validate_setup(self, distributed_plan, source_bindings, context):
            self.events.append(("setup", context.rank))
            return super().validate_setup(distributed_plan, source_bindings, context)

        def validate_request(self, request, residency_plan=None):
            self.events.append(("request", request.source_rank, residency_plan))
            return super().validate_request(request, residency_plan)

    provider = RecordingProvider()
    matrix = np.arange(16, dtype=np.float64).reshape(4, 4)
    operator = _operator(provider, matrix)

    result = operator(np.arange(4, dtype=np.float64))

    np.testing.assert_array_equal(result, matrix @ np.arange(4, dtype=np.float64))
    assert provider.events[:2] == [("setup", 0), ("request", 0, None)]


def test_provider_neutral_setup_does_not_inspect_active_source_ref_values():
    class ActiveMetadataProvider:
        residency_policy = "active_working_set"

        def __init__(self):
            self.setup_calls = 0
            self.acquire_calls = 0

        def validate_setup(self, distributed_plan, source_bindings, context):
            self.setup_calls += 1

        def validate_request(self, request, residency_plan=None):
            raise AssertionError("unresolved active plan reached request validation")

        @contextmanager
        def acquire(self, request):
            self.acquire_calls += 1
            yield None

    provider = ActiveMetadataProvider()
    operator = _operator(provider, object())

    with pytest.raises(
        ValueError, match="distributed setup preflight failed"
    ) as caught:
        operator.solver_preflight()

    assert "active_working_set" in str(caught.value.__cause__)
    assert "residency plan" in str(caught.value.__cause__)
    assert provider.setup_calls == 0
    assert provider.acquire_calls == 0


@pytest.mark.parametrize(
    "backend_config",
    [
        BackendConfig(
            device="cpu",
            execution_policy="legacy_oe",
            fallback_policy="error",
        ),
        BackendConfig(
            device="cpu",
            execution_policy="execution_ir",
            fallback_policy="legacy_oe",
        ),
    ],
)
def test_wave8_active_policy_rejects_legacy_and_fallback_before_acquire(
    backend_config,
):
    class Provider:
        residency_policy = "active_working_set"

        def __init__(self):
            self.setup_calls = 0
            self.request_calls = 0
            self.acquire_calls = 0

        def validate_setup(self, distributed_plan, source_bindings, context):
            self.setup_calls += 1

        def validate_request(self, request, residency_plan=None):
            self.request_calls += 1

        @contextmanager
        def acquire(self, request):
            self.acquire_calls += 1
            yield None

    provider = Provider()
    operator, store = _active_operator(
        provider,
        _AgreeingCollective(),
        backend_config=backend_config,
    )
    try:
        with pytest.raises(
            ValueError, match="distributed setup preflight failed"
        ) as caught:
            operator.solver_preflight()
    finally:
        store.close()

    assert "complete local-H-v execution contract" in str(caught.value.__cause__)
    assert provider.setup_calls == 0
    assert provider.request_calls == 0
    assert provider.acquire_calls == 0


def test_wave8_active_policy_rejects_incomplete_ir_before_callback():
    class Provider:
        residency_policy = "active_working_set"

        def __init__(self):
            self.callback_calls = 0
            self.acquire_calls = 0

        def validate_setup(self, distributed_plan, source_bindings, context):
            self.callback_calls += 1

        def validate_request(self, request, residency_plan=None):
            self.callback_calls += 1

        @contextmanager
        def acquire(self, request):
            self.acquire_calls += 1
            yield None

    provider = Provider()
    operator, store = _active_operator(
        provider,
        _AgreeingCollective(),
        specialized=False,
    )
    try:
        with pytest.raises(
            ValueError, match="distributed setup preflight failed"
        ) as caught:
            operator.solver_preflight()
    finally:
        store.close()

    assert "complete local-H-v execution contract" in str(caught.value.__cause__)
    assert provider.callback_calls == 0
    assert provider.acquire_calls == 0

    fallback_callback_calls = 0

    def fallback_callback():
        nonlocal fallback_callback_calls
        fallback_callback_calls += 1
        return np.ones(4, dtype=np.float64)

    with pytest.raises(
        NotImplementedError, match="complete local-H-v execution contract"
    ):
        run_adapter_root_fallback(
            fallback_callback,
            SimpleNamespace(residency_policy="active_working_set"),
            operator.backend,
            np.empty(4, dtype=np.float64),
            estimated_device_bytes=32,
            estimated_host_bytes=32,
            counters={},
        )

    assert fallback_callback_calls == 0


def test_wave9_active_policy_requires_real_local_hv_builder_contract():
    backend = create_backend(
        "numpy",
        config=BackendConfig(
            device="cpu",
            execution_policy="execution_ir",
            fallback_policy="error",
        ),
    )
    expression = build_mps_ir_hop(
        "ab,b->a",
        (np.ones((8, 8), dtype=np.float64),),
        (8,),
        "one_site",
    )
    actual = expression.resolve_execution_artifact(
        np.ones(8, dtype=np.float64)
    ).execution_plan
    actual_distributed = plan_distributed_execution(
        actual, variable_key="input_1", world_size=2
    )
    relabeled = _relabeled_execution_plan()
    relabeled_distributed = plan_distributed_execution(
        relabeled, variable_key="input_1", world_size=2
    )

    assert (
        active_working_set_policy_error(
            "active_working_set", backend, actual_distributed
        )
        is None
    )
    error = active_working_set_policy_error(
        "active_working_set", backend, relabeled_distributed
    )
    assert isinstance(error, NotImplementedError)
    assert "execution contract" in str(error)


def test_active_setup_rejects_replayed_request_before_provider_hooks():
    class Provider:
        residency_policy = "active_working_set"

        def __init__(self):
            self.setup_calls = 0
            self.request_calls = 0
            self.acquire_calls = 0

        def validate_setup(self, distributed_plan, source_bindings, context):
            self.setup_calls += 1

        def validate_request(self, request, residency_plan=None):
            self.request_calls += 1

        @contextmanager
        def acquire(self, request):
            self.acquire_calls += 1
            yield None

    def change_solver(request):
        return replace(
            request,
            solver_profile=build_krylov_memory_profile(
                request.solver_input_sharding,
                "float64",
                block_size=request.solver_profile.block_size + 1,
                max_krylov_vectors=request.solver_profile.max_krylov_vectors,
            ),
        )

    provider = Provider()
    operator, store = _active_operator(
        provider,
        _AgreeingCollective(),
        residency_request_transform=change_solver,
    )
    try:
        with pytest.raises(
            ValueError, match="distributed setup preflight failed"
        ) as caught:
            operator.solver_preflight()
    finally:
        store.close()

    assert "request hash" in str(caught.value.__cause__)
    assert provider.setup_calls == 0
    assert provider.request_calls == 0
    assert provider.acquire_calls == 0


def test_wave9_active_setup_rechecks_retained_plan_capacity_before_provider_hooks():
    class Provider:
        residency_policy = "active_working_set"

        def __init__(self):
            self.callback_calls = 0
            self.acquire_calls = 0

        def validate_setup(self, distributed_plan, source_bindings, context):
            self.callback_calls += 1

        def validate_request(self, request, residency_plan=None):
            self.callback_calls += 1

        @contextmanager
        def acquire(self, request):
            self.acquire_calls += 1
            yield None

    def bypass_constructor(plan):
        forged = object.__new__(type(plan))
        for field_info in dataclasses.fields(plan):
            value = getattr(plan, field_info.name)
            if field_info.name == "host_budget":
                value = _explicit_budget(plan.host_required_bytes - 1, "host")
            object.__setattr__(forged, field_info.name, value)
        return forged

    provider = Provider()
    operator, store = _active_operator(
        provider,
        _AgreeingCollective(),
        residency_plan_transform=bypass_constructor,
    )
    try:
        with pytest.raises(
            ValueError, match="distributed setup preflight failed"
        ) as caught:
            operator.solver_preflight()
    finally:
        store.close()

    assert isinstance(caught.value.__cause__, ResidencyBudgetError)
    assert caught.value.__cause__.resource == "host"
    assert provider.callback_calls == 0
    assert provider.acquire_calls == 0


@pytest.mark.parametrize(
    ("context", "mesh"),
    [
        (
            DistributedContext(0, 0, 2, 2),
            DeviceMesh((1, 2), ("node", "rank"), 0),
        ),
        (
            DistributedContext(0, 1, 2, 2),
            DeviceMesh((2,), ("rank",), 0),
        ),
    ],
)
def test_wave9_runtime_identity_mismatch_fails_before_provider_hooks(context, mesh):
    class Provider:
        residency_policy = "active_working_set"

        def __init__(self):
            self.callback_calls = 0
            self.acquire_calls = 0

        def validate_setup(self, distributed_plan, source_bindings, context):
            self.callback_calls += 1

        def validate_request(self, request, residency_plan=None):
            self.callback_calls += 1

        @contextmanager
        def acquire(self, request):
            self.acquire_calls += 1
            yield None

    provider = Provider()
    operator, store = _active_operator(
        provider,
        _AgreeingCollective(),
        context=context,
        mesh=mesh,
    )
    try:
        with pytest.raises(
            ValueError, match="distributed setup preflight failed"
        ) as caught:
            operator.solver_preflight()
    finally:
        store.close()

    assert "runtime identity" in str(caught.value.__cause__)
    assert provider.callback_calls == 0
    assert provider.acquire_calls == 0


def test_active_plan_rejects_policyless_provider_before_hooks():
    class Provider:
        def __init__(self):
            self.setup_calls = 0
            self.request_calls = 0
            self.acquire_calls = 0

        def validate_setup(self, distributed_plan, source_bindings, context):
            self.setup_calls += 1

        def validate_request(self, request, residency_plan=None):
            self.request_calls += 1

        @contextmanager
        def acquire(self, request):
            self.acquire_calls += 1
            yield None

    provider = Provider()
    operator, store = _active_operator(provider, _AgreeingCollective())
    try:
        with pytest.raises(
            ValueError, match="distributed setup preflight failed"
        ) as caught:
            operator.solver_preflight()
    finally:
        store.close()

    assert "provider policy" in str(caught.value.__cause__)
    assert provider.setup_calls == 0
    assert provider.request_calls == 0
    assert provider.acquire_calls == 0


def test_device_resident_validation_behavior_is_preserved_behind_setup_hook():
    operator = _operator(DeviceResidentProvider(), np.ones((3, 3)))

    with pytest.raises(
        ValueError, match="distributed setup preflight failed"
    ) as caught:
        operator.solver_preflight()

    assert "shape" in str(caught.value.__cause__)


def test_active_residency_hash_disagreement_fails_before_provider_acquisition():
    class Collective:
        rank = 0
        size = 2

        def __init__(self):
            self.calls = []

        def allreduce(self, value, *, op="sum"):
            copied = np.array(value, copy=True)
            self.calls.append((copied, op))
            if copied.dtype == np.dtype(np.uint64) and copied.size == 8 and op == "max":
                copied[4] ^= np.uint64(1)
            return copied

        def allreduce_inplace(self, value, *, op="sum"):
            raise AssertionError("execution allreduce must not be reached")

        def broadcast(self, value, *, root):
            raise AssertionError("data-path broadcast must not be reached")

    class Provider:
        residency_policy = "active_working_set"

        def __init__(self):
            self.setup_calls = 0
            self.request_calls = 0
            self.acquire_calls = 0

        def validate_setup(self, distributed_plan, source_bindings, context):
            self.setup_calls += 1

        def validate_request(self, request, residency_plan=None):
            self.request_calls += 1

        @contextmanager
        def acquire(self, request):
            self.acquire_calls += 1
            raise AssertionError("provider acquisition must not be reached")
            yield

    provider = Provider()
    operator, store = _active_operator(provider, Collective())
    try:
        with pytest.raises(ValueError, match="residency hash disagreement"):
            operator(np.ones(4, dtype=np.float64))
    finally:
        store.close()

    assert provider.setup_calls == 1
    assert provider.request_calls == 2
    assert provider.acquire_calls == 0


def test_wave8_requirement_disagreement_fails_closed():
    class Collective(_AgreeingCollective):
        def __init__(self):
            self.calls = []

        def allreduce(self, value, *, op="sum"):
            copied = np.array(value, copy=True)
            self.calls.append((copied, op))
            if copied.dtype == np.dtype(np.int64) and op == "max":
                copied[0] += 1
            return copied

    class Provider:
        residency_policy = "active_working_set"

        def __init__(self):
            self.acquire_calls = 0

        def validate_setup(self, distributed_plan, source_bindings, context):
            return None

        def validate_request(self, request, residency_plan=None):
            return None

        @contextmanager
        def acquire(self, request):
            self.acquire_calls += 1
            yield None

    collective = Collective()
    provider = Provider()
    operator, store = _active_operator(provider, collective)
    try:
        with pytest.raises(ValueError, match="residency requirement disagreement"):
            operator.solver_preflight()
    finally:
        store.close()

    assert any(value.dtype == np.dtype(np.int64) for value, _ in collective.calls)
    assert provider.acquire_calls == 0


@pytest.mark.parametrize(
    ("peer_control", "message"),
    [
        ((0, 0), "residency policy disagreement"),
        ((1, 0), "residency plan presence disagreement"),
    ],
)
def test_residency_presence_and_policy_agree_before_hash_collectives(
    peer_control, message
):
    class Collective:
        rank = 0
        size = 2

        def __init__(self):
            self.calls = []

        def allreduce(self, value, *, op="sum"):
            copied = np.array(value, copy=True)
            self.calls.append((copied, op))
            if copied.dtype == np.dtype(np.int32) and copied.size == 2:
                peer = np.asarray(peer_control, dtype=np.int32)
                if op == "min":
                    return np.minimum(copied, peer)
                if op == "max":
                    return np.maximum(copied, peer)
            if copied.dtype == np.dtype(np.uint64):
                raise AssertionError("hash comparison preceded residency controls")
            return copied

        def allreduce_inplace(self, value, *, op="sum"):
            raise AssertionError("execution allreduce must not be reached")

        def broadcast(self, value, *, root):
            raise AssertionError("data-path broadcast must not be reached")

    class Provider:
        residency_policy = "active_working_set"

        def validate_setup(self, distributed_plan, source_bindings, context):
            return None

        def validate_request(self, request, residency_plan=None):
            return None

        @contextmanager
        def acquire(self, request):
            raise AssertionError("provider acquisition must not be reached")
            yield

    collective = Collective()
    operator, store = _active_operator(Provider(), collective)
    try:
        with pytest.raises(ValueError, match=message):
            operator.solver_preflight()
    finally:
        store.close()

    assert [(value.dtype, value.shape, op) for value, op in collective.calls] == [
        (np.dtype(np.int32), (1,), "max"),
        (np.dtype(np.int32), (2,), "min"),
        (np.dtype(np.int32), (2,), "max"),
    ]


def test_active_residency_profile_bounds_retained_policy_and_hash_arrays(
    monkeypatch,
):
    import renormalizer.backend._distributed.local_operator as local_operator

    class TrackedArray(np.ndarray):
        pass

    class LifetimeTracker:
        def __init__(self):
            self.device_refs = []
            self.host_refs = []
            self.device_peak = 0
            self.host_peak = 0

        @staticmethod
        def _live_bytes(refs):
            return sum(
                array.nbytes for ref in refs for array in (ref(),) if array is not None
            )

        def track_device(self, value, dtype=None):
            array = np.array(value, dtype=dtype, copy=True).view(TrackedArray)
            self.device_refs.append(weakref.ref(array))
            self.device_peak = max(self.device_peak, self._live_bytes(self.device_refs))
            return array

        def track_host(self, value):
            array = np.array(value, copy=True).view(TrackedArray)
            self.host_refs.append(weakref.ref(array))
            self.host_peak = max(self.host_peak, self._live_bytes(self.host_refs))
            return array

    class Collective:
        rank = 0
        size = 2

        def __init__(self, tracker):
            self.tracker = tracker

        def allreduce(self, value, *, op="sum"):
            return self.tracker.track_device(value)

        def allreduce_inplace(self, value, *, op="sum"):
            raise AssertionError("execution allreduce must not be reached")

        def broadcast(self, value, *, root):
            raise AssertionError("data-path broadcast must not be reached")

    class Provider:
        residency_policy = "active_working_set"

        def validate_setup(self, distributed_plan, source_bindings, context):
            return None

        def validate_request(self, request, residency_plan=None):
            return None

        @contextmanager
        def acquire(self, request):
            raise AssertionError("provider acquisition must not be reached")
            yield

    tracker = LifetimeTracker()
    collective = Collective(tracker)
    monkeypatch.setattr(
        local_operator,
        "_control_array",
        lambda _collective, values, dtype: tracker.track_device(values, dtype),
    )
    monkeypatch.setattr(local_operator, "_array_to_numpy", tracker.track_host)
    operator, store = _active_operator(Provider(), collective)
    try:
        operator.solver_preflight()
    finally:
        store.close()

    estimate = operator.residency_plan.rank_estimates[0]
    assert tracker.device_peak == 216
    assert tracker.host_peak == 208
    assert estimate.residency_hash_control_device_bytes == tracker.device_peak
    assert estimate.residency_hash_control_host_bytes == tracker.host_peak


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2",
    reason="requires the exact two-rank torchrun command",
)
def test_real_two_rank_active_residency_metadata_preflight_and_peak_enforcement():
    cupy = pytest.importorskip("cupy")
    from renormalizer import set_backend
    from renormalizer.backend.distributed_runtime import (
        create_cupy_distributed_runtime,
    )

    class MetadataOnlyProvider:
        residency_policy = "active_working_set"

        def __init__(self, residency_plan, store):
            self.residency_plan = residency_plan
            self.store = store
            self.setup_calls = 0
            self.request_ranks = []
            self.acquire_calls = 0

        def validate_setup(self, distributed_plan, source_bindings, context):
            self.setup_calls += 1
            self.residency_plan.validate_store(self.store)

        def validate_request(self, request, residency_plan=None):
            assert residency_plan is self.residency_plan
            assert request.residency_plan is self.residency_plan
            self.residency_plan.validate_request(request.residency_request)
            self.request_ranks.append(request.source_rank)

        @contextmanager
        def acquire(self, request):
            self.acquire_calls += 1
            raise AssertionError("Task 18 acquisition must not be reached")
            yield

    class PolicyOnlyProvider:
        def __init__(self, residency_policy):
            self.residency_policy = residency_policy
            self.setup_calls = 0
            self.request_ranks = []
            self.acquire_calls = 0

        def validate_setup(self, distributed_plan, source_bindings, context):
            self.setup_calls += 1

        def validate_request(self, request, residency_plan=None):
            self.request_ranks.append(request.source_rank)

        @contextmanager
        def acquire(self, request):
            self.acquire_calls += 1
            raise AssertionError("Task 18 acquisition must not be reached")
            yield

    runtime = create_cupy_distributed_runtime(expected_world_size=2)
    stores = []
    try:
        set_backend(
            "cupy",
            device="cuda:{}".format(runtime.local_rank),
            precision=64,
        )

        with pytest.raises(ResidencyBudgetError) as budget_caught:
            _real_active_case(runtime, "real-active-low", 1)
        assert budget_caught.value.resource == "device"
        assert budget_caught.value.rank == 0
        assert budget_caught.value.required_bytes > 1

        promoted = _real_active_case(
            runtime,
            "real-promoted-qn-baseline",
            1 << 30,
            dtype="float32",
            coefficient=-0.125j,
            qn_mask_present=True,
        )
        distributed, store, matrix_ref, promoted_request, promoted_plan = promoted
        stores.append(store)
        promoted_estimate = promoted_plan.rank_estimates[runtime.rank]
        assert promoted_estimate.solver_input_bytes == 4 * 4
        assert promoted_estimate.solver_result_bytes == 4 * 16
        assert promoted_estimate.complete_input_center_bytes == 8 * 4
        assert promoted_estimate.complete_center_bytes == 8 * 16
        assert promoted_estimate.qn_hv_mask_device_bytes == 4
        assert promoted_estimate.qn_writeback_mask_device_bytes == 8
        assert promoted_estimate.state_agreement_device_bytes == 104
        assert promoted_estimate.state_agreement_host_bytes == 96
        exact_promoted_budget = max(promoted_plan.device_peak_bytes)

        with pytest.raises(ResidencyBudgetError) as promoted_budget_caught:
            _real_active_case(
                runtime,
                "real-promoted-qn-low",
                exact_promoted_budget - 1,
                dtype="float32",
                coefficient=-0.125j,
                qn_mask_present=True,
            )
        assert promoted_budget_caught.value.resource == "device"

        (
            distributed,
            store,
            matrix_ref,
            promoted_request,
            promoted_plan,
        ) = _real_active_case(
            runtime,
            "real-promoted-qn-exact",
            exact_promoted_budget,
            dtype="float32",
            coefficient=-0.125j,
            qn_mask_present=True,
        )
        stores.append(store)
        provider = MetadataOnlyProvider(promoted_plan, store)
        operator = _real_active_operator(
            runtime,
            provider,
            distributed,
            matrix_ref,
            promoted_request,
            promoted_plan,
        )
        operator.solver_preflight()
        assert provider.setup_calls == 1
        assert provider.request_ranks == [0, 1]
        assert provider.acquire_calls == 0
        runtime.barrier()

        (
            distributed,
            store,
            matrix_ref,
            residency_request,
            residency_plan,
        ) = _real_active_case(runtime, "real-policy-disagreement", 1 << 30)
        stores.append(store)
        policy = "active_working_set" if runtime.rank == 0 else "device_resident"
        provider = PolicyOnlyProvider(policy)
        operator = _real_active_operator(
            runtime,
            provider,
            distributed,
            matrix_ref,
            residency_request if runtime.rank == 0 else None,
            residency_plan if runtime.rank == 0 else None,
        )
        with pytest.raises(ValueError, match="residency policy disagreement"):
            operator.solver_preflight()
        assert provider.setup_calls == 1
        assert provider.request_ranks == [0, 1]
        assert provider.acquire_calls == 0
        runtime.barrier()

        (
            distributed,
            store,
            matrix_ref,
            residency_request,
            residency_plan,
        ) = _real_active_case(
            runtime,
            "real-active-disagreement",
            (1 << 30) + runtime.rank,
        )
        stores.append(store)
        provider = MetadataOnlyProvider(residency_plan, store)
        operator = _real_active_operator(
            runtime,
            provider,
            distributed,
            matrix_ref,
            residency_request,
            residency_plan,
        )
        with pytest.raises(ValueError, match="residency hash disagreement"):
            operator(cupy.ones(4, dtype=cupy.float64))
        assert provider.setup_calls == 1
        assert provider.request_ranks == [0, 1]
        assert provider.acquire_calls == 0

        (
            distributed,
            store,
            matrix_ref,
            residency_request,
            residency_plan,
        ) = _real_active_case(runtime, "real-active-stale", 1 << 30)
        stores.append(store)
        if runtime.rank == 0:
            store.update(
                "input_0",
                np.full((8, 8), 2.0, dtype=np.float64),
                expected_version=matrix_ref.version,
            )
        provider = MetadataOnlyProvider(residency_plan, store)
        operator = _real_active_operator(
            runtime,
            provider,
            distributed,
            matrix_ref,
            residency_request,
            residency_plan,
        )
        with pytest.raises(ValueError, match="distributed setup preflight failed"):
            operator.solver_preflight()
        if runtime.rank == 0:
            assert isinstance(operator._setup_error, StaleHostTensorRefError)
            assert provider.request_ranks == []
        else:
            assert operator._setup_error is None
            assert provider.request_ranks == [0, 1]
        assert provider.setup_calls == 1
        assert provider.acquire_calls == 0

        (
            distributed,
            store,
            matrix_ref,
            residency_request,
            residency_plan,
        ) = _real_active_case(runtime, "real-active-hooks", 1 << 30)
        stores.append(store)
        provider = MetadataOnlyProvider(residency_plan, store)
        operator = _real_active_operator(
            runtime,
            provider,
            distributed,
            matrix_ref,
            residency_request,
            residency_plan,
        )
        operator.solver_preflight()
        assert provider.setup_calls == 1
        assert provider.request_ranks == [0, 1]
        assert provider.acquire_calls == 0
    finally:
        for store in stores:
            if not store.closed:
                store.close()
        runtime.close()
