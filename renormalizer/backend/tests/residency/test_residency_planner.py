import dataclasses
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from renormalizer.backend._distributed.planner import plan_distributed_execution
from renormalizer.backend._distributed.center import (
    CenterVectorMap,
    center_materialization_memory_profile,
)
from renormalizer.backend._distributed.residency import (
    FutureResidencyPlan,
    HostTensorAllocation,
    HostTensorStore,
    HostTensorStoreSnapshot,
    MemoryBudgetResolution,
    ResidencyBudgetError,
    ResidencyPlanner,
    ResidencyRequest,
    ResidencyRuntimeIdentity,
    SliceRange,
    TensorPlacement,
    _RANK_COMPONENT_NAMES,
    _plan_payload,
    _rank_peaks,
    _sha256,
)
from renormalizer.backend._distributed.solvers import (
    build_davidson_memory_profile,
    build_krylov_memory_profile,
)
from renormalizer.backend._distributed.transfer import TransferProfile
from renormalizer.backend._distributed.sharding import shard_axis
from renormalizer.backend._execution.planner import lower_einsum_path


def _explicit_budget(value, resource):
    return MemoryBudgetResolution(
        requested_bytes=value,
        resolved_bytes=value,
        source="explicit",
        available_snapshot_bytes=None,
        resource=resource,
    )


def _solver_profile(sharding, kind="krylov"):
    if kind == "davidson":
        return build_davidson_memory_profile(sharding, "float64", max_space=4)
    return build_krylov_memory_profile(
        sharding, "float64", block_size=4, max_krylov_vectors=4
    )


def _request(
    *,
    kind="krylov",
    device_budget=1 << 30,
    host_budget=1 << 31,
    backend_name="cupy",
):
    source = lower_einsum_path("ab,b->a", ((8, 8), (8,)), dtype="float64")
    distributed = plan_distributed_execution(
        source, variable_key="input_1", world_size=2
    )
    store = HostTensorStore(store_id="shared-run")
    matrix = store.put("input_0", np.arange(64, dtype=np.float64).reshape(8, 8))
    center = store.put("output", np.zeros(8, dtype=np.float64))
    snapshot = store.snapshot()
    output_allocation = HostTensorAllocation("output", (8,), "float64")
    request = ResidencyRequest(
        distributed_plan=distributed,
        host_refs={"output": center, "input_0": matrix},
        world_size=2,
        local_world_size=2,
        backend_name=backend_name,
        store_bytes=None,
        external_host_bytes=(11, 13),
        transfer_staging_host_bytes=None,
        dirty_writeback_bytes=None,
        solver_input_sharding=distributed.input_sharding,
        solver_output_sharding=distributed.input_sharding,
        mapped_local_counts=(4, 4),
        solver_profile=_solver_profile(distributed.input_sharding, kind),
        materialization_policy="device",
        complete_center_bytes=64,
        prefetch_depth=1,
        future_plans=(),
        device_budget=_explicit_budget(device_budget, "device"),
        host_budget=_explicit_budget(host_budget, "host"),
        store_snapshots=(snapshot, snapshot),
        writeback_allocations=(
            (output_allocation,),
            (output_allocation,),
        ),
    )
    return request, store


def _expected_rank_host_peak(request, plan, rank):
    actual = plan.rank_estimates[rank]
    task14 = request.distributed_plan.memory_estimates[rank]
    solver = request.solver_profile.rank_estimates[rank]
    host_base = (
        request.store_bytes[rank]
        + request.external_host_bytes[rank]
        + request.transfer_staging_host_bytes[rank]
    )
    setup = host_base + max(
        task14.host_bytes,
        actual.residency_hash_control_host_bytes,
    )
    persistent = host_base + task14.host_control_status_bytes
    solver_phase = persistent + solver.host_peak_bytes
    materialization = persistent + actual.materialization_control_host_bytes
    state_digest = persistent + actual.state_digest_host_bytes
    state_agreement = persistent + actual.state_agreement_host_bytes
    writeback = (
        persistent
        + request.dirty_writeback_bytes[rank]
        + max(
            solver.host_peak_bytes,
            actual.materialization_control_host_bytes,
        )
    )
    return max(
        setup,
        solver_phase,
        materialization,
        state_digest,
        state_agreement,
        writeback,
        persistent,
    )


def _rehash_plan(plan, **changes):
    provisional = object.__new__(type(plan))
    for field_info in dataclasses.fields(plan):
        object.__setattr__(
            provisional,
            field_info.name,
            changes.get(field_info.name, getattr(plan, field_info.name)),
        )
    changes["plan_hash"] = _sha256(_plan_payload(provisional))
    return replace(plan, **changes)


def test_valid_lowered_plan_produces_exact_temporal_device_and_host_components():
    request, _ = _request(kind="davidson")

    plan = ResidencyPlanner().plan(request)

    assert plan.source_plan_hash == request.distributed_plan.execution_plan.plan_hash
    assert plan.placement_hash == request.distributed_plan.placement_hash
    assert plan.solver_kind == "davidson"
    assert plan.device_peak_bytes == tuple(
        estimate.device_peak_bytes for estimate in plan.rank_estimates
    )
    assert plan.host_peak_bytes == tuple(
        estimate.host_peak_bytes for estimate in plan.rank_estimates
    )
    assert plan.host_required_bytes == sum(plan.host_peak_bytes)
    for rank, actual in enumerate(plan.rank_estimates):
        stage4 = request.distributed_plan.memory_estimates[rank]
        solver = request.solver_profile.rank_estimates[rank]
        solver_bytes = request.mapped_local_counts[rank] * 8
        assert actual.dense_input_bytes == stage4.local_input_bytes
        assert actual.receive_buffer_bytes == stage4.receive_buffer_bytes
        assert actual.output_accumulator_bytes == stage4.output_accumulator_bytes
        assert actual.output_contribution_bytes == stage4.output_contribution_bytes
        assert actual.execution_workspace_bytes == stage4.workspace_bytes
        assert actual.mapped_packed_output_bytes == solver_bytes
        assert actual.solver_input_bytes == solver_bytes
        assert actual.solver_result_bytes == solver_bytes
        assert actual.solver_diagonal_bytes == solver_bytes
        assert actual.residency_hash_control_device_bytes == 3 * 8 * 8 + 3 * 2 * 4
        assert actual.residency_hash_control_host_bytes == 3 * 8 * 8 + 2 * 2 * 4
        assert actual.complete_center_bytes == 64
        assert actual.original_packed_guess_bytes == 64
        assert actual.original_packed_diagonal_bytes == 64
        assert actual.materialization_receive_bytes == 32
        assert actual.materialization_control_device_bytes == 12
        assert actual.materialization_control_host_bytes == 4
        assert actual.task14_host_control_bytes == stage4.host_bytes
        assert (
            actual.task14_persistent_execution_status_host_bytes
            == stage4.host_control_status_bytes
        )
        expected_preflight = (
            actual.current_static_bytes
            + actual.prefetched_static_bytes
            + actual.complete_input_center_bytes
            + actual.complete_diagonal_center_bytes
            + actual.original_packed_guess_bytes
            + actual.original_packed_diagonal_bytes
            + actual.solver_input_bytes
            + actual.solver_diagonal_bytes
            + actual.dense_input_bytes
            + actual.mapped_packed_output_bytes
            + max(
                stage4.preflight_hash_device_bytes,
                actual.residency_hash_control_device_bytes,
                stage4.preflight_status_device_bytes,
                solver.communication_device_bytes,
                actual.qn_setup_mask_device_bytes,
                actual.qn_extract_mask_device_bytes,
            )
        )
        expected_hv = (
            actual.current_static_bytes
            + actual.prefetched_static_bytes
            + stage4.local_input_bytes
            + stage4.receive_buffer_bytes
            + stage4.output_accumulator_bytes
            + stage4.control_status_bytes
            + solver_bytes
            + solver_bytes
            + solver_bytes
            + actual.complete_input_center_bytes
            + actual.complete_diagonal_center_bytes
            + actual.original_packed_guess_bytes
            + actual.original_packed_diagonal_bytes
            + solver.hv_retained_bytes
            + max(
                stage4.output_contribution_bytes
                + stage4.workspace_bytes
                + solver.hv_transient_bytes,
                stage4.preflight_status_device_bytes,
                solver.communication_device_bytes,
                actual.qn_hv_mask_device_bytes,
            )
        )
        expected_la = (
            actual.current_static_bytes
            + actual.prefetched_static_bytes
            + stage4.local_input_bytes
            + stage4.receive_buffer_bytes
            + stage4.output_accumulator_bytes
            + stage4.control_status_bytes
            + solver_bytes
            + solver_bytes
            + solver_bytes
            + actual.complete_input_center_bytes
            + actual.complete_diagonal_center_bytes
            + actual.original_packed_guess_bytes
            + actual.original_packed_diagonal_bytes
            + solver.la_retained_bytes
            + solver.la_transient_bytes
            + solver.communication_device_bytes
        )
        expected_materialize = (
            actual.current_static_bytes
            + actual.prefetched_static_bytes
            + stage4.local_input_bytes
            + stage4.receive_buffer_bytes
            + stage4.output_accumulator_bytes
            + stage4.control_status_bytes
            + solver_bytes
            + solver_bytes
            + solver_bytes
            + solver_bytes
            + actual.complete_input_center_bytes
            + actual.complete_diagonal_center_bytes
            + actual.original_packed_guess_bytes
            + actual.original_packed_diagonal_bytes
            + 64
            + 32
            + 12
            + actual.qn_materialization_mask_device_bytes
        )
        assert actual.preflight_device_peak_bytes == expected_preflight
        assert actual.hv_device_peak_bytes == expected_hv
        assert actual.solver_la_device_peak_bytes == expected_la
        assert actual.materialization_device_peak_bytes == expected_materialize
        assert actual.device_peak_bytes == max(
            expected_preflight, expected_hv, expected_la, expected_materialize
        )
        expected_host = _expected_rank_host_peak(request, plan, rank)
        assert actual.host_peak_bytes == expected_host


def test_hv_peak_combines_executor_workspace_with_concurrent_solver_state():
    request, store = _request(kind="krylov")
    try:
        plan = ResidencyPlanner().plan(request)
    finally:
        store.close()

    for rank, actual in enumerate(plan.rank_estimates):
        task14 = request.distributed_plan.memory_estimates[rank]
        solver = request.solver_profile.rank_estimates[rank]
        execution_base = (
            actual.current_static_bytes
            + actual.prefetched_static_bytes
            + actual.dense_input_bytes
            + actual.receive_buffer_bytes
            + actual.output_accumulator_bytes
            + actual.task14_execution_status_device_bytes
            + actual.mapped_packed_output_bytes
        )
        expected = (
            execution_base
            + actual.complete_input_center_bytes
            + actual.complete_diagonal_center_bytes
            + actual.solver_input_bytes
            + actual.solver_diagonal_bytes
            + solver.hv_retained_bytes
            + max(
                task14.output_contribution_bytes
                + task14.workspace_bytes
                + solver.hv_transient_bytes,
                task14.preflight_status_device_bytes,
                solver.communication_device_bytes,
            )
        )
        assert actual.hv_device_peak_bytes == expected


@pytest.mark.parametrize(
    ("host_components", "expected"),
    [
        (
            {
                "task14_host_control_bytes": 64,
                "residency_hash_control_host_bytes": 208,
                "task14_persistent_execution_status_host_bytes": 4,
                "solver_host_peak_bytes": 100,
            },
            208,
        ),
        (
            {
                "task14_host_control_bytes": 300,
                "residency_hash_control_host_bytes": 208,
                "task14_persistent_execution_status_host_bytes": 4,
                "solver_host_peak_bytes": 100,
                "materialization_control_host_bytes": 200,
            },
            300,
        ),
        (
            {
                "task14_host_control_bytes": 64,
                "task14_persistent_execution_status_host_bytes": 4,
                "solver_host_peak_bytes": 100,
            },
            104,
        ),
        (
            {
                "task14_host_control_bytes": 64,
                "task14_persistent_execution_status_host_bytes": 4,
                "solver_host_peak_bytes": 20,
                "materialization_control_host_bytes": 100,
            },
            104,
        ),
        (
            {
                "task14_host_control_bytes": 64,
                "task14_persistent_execution_status_host_bytes": 4,
                "solver_host_peak_bytes": 100,
                "dirty_writeback_bytes": 50,
            },
            154,
        ),
    ],
)
def test_host_phase_maxima_split_transient_and_persistent_controls(
    host_components, expected
):
    components = {name: 0 for name in _RANK_COMPONENT_NAMES}
    components.update(host_components)

    assert _rank_peaks(components)["host_peak_bytes"] == expected


def test_request_and_plan_are_canonical_immutable_and_hash_stable():
    first, _ = _request()
    second, _ = _request()
    second = replace(second, host_refs=dict(reversed(second.host_refs)))

    first_plan = ResidencyPlanner().plan(first)
    second_plan = ResidencyPlanner().plan(second)

    assert first == second
    assert first.request_hash == second.request_hash
    assert first_plan == second_plan
    assert first_plan.plan_hash == second_plan.plan_hash
    assert first_plan.local_slices == tuple(sorted(first_plan.local_slices))
    assert first_plan.required_refs == tuple(
        sorted(
            first_plan.required_refs,
            key=lambda ref: (ref.store_id, ref.key, ref.generation, ref.version),
        )
    )
    with pytest.raises(FrozenInstanceError):
        first_plan.prefetch_depth = 2


def test_transfer_profile_is_canonical_derived_and_bound_to_request_and_plan():
    request, _ = _request()
    plan = ResidencyPlanner().plan(request)

    assert isinstance(request.transfer_profile, TransferProfile)
    assert request.transfer_profile.lanes == 1
    assert request.transfer_profile.current_h2d_bytes == (128, 128)
    assert request.transfer_profile.future_h2d_bytes == (0, 0)
    assert request.transfer_profile.dirty_d2h_bytes == (64, 64)
    assert request.transfer_profile.rank_staging_bytes == (128, 128)
    assert request.transfer_staging_host_bytes == (128, 128)
    assert plan.transfer_profile == request.transfer_profile

    with pytest.raises(ValueError, match="transfer staging.*canonical profile"):
        replace(request, transfer_staging_host_bytes=(127, 128))

    replay = replace(request, transfer_staging_host_bytes=None)
    assert replay.transfer_profile == request.transfer_profile
    assert replay.request_hash == request.request_hash


def test_wave8_plan_binds_complete_request_and_rejects_solver_replay():
    request, store = _request()
    changed_profile = build_krylov_memory_profile(
        request.solver_input_sharding,
        "float64",
        block_size=request.solver_profile.block_size + 1,
        max_krylov_vectors=request.solver_profile.max_krylov_vectors,
    )
    changed = replace(request, solver_profile=changed_profile)
    try:
        plan = ResidencyPlanner().plan(request)
        plan.validate_request(request)

        with pytest.raises(ValueError, match="request hash"):
            plan.validate_request(changed)
    finally:
        store.close()

    assert request.request_hash != changed.request_hash
    assert plan.request_hash == request.request_hash
    assert _plan_payload(plan)["request_hash"] == request.request_hash


def test_validate_request_rejects_rehashed_internally_consistent_lower_peaks():
    request, store = _request()
    try:
        plan = ResidencyPlanner().plan(request)
        original = plan.rank_estimates[0]
        components = {name: getattr(original, name) for name in _RANK_COMPONENT_NAMES}
        components["current_static_bytes"] -= np.dtype(np.float64).itemsize
        forged_estimate = replace(
            original,
            **components,
            **_rank_peaks(components),
        )
        estimates = (forged_estimate,) + plan.rank_estimates[1:]
        device_peaks = tuple(value.device_peak_bytes for value in estimates)
        host_peaks = tuple(value.host_peak_bytes for value in estimates)
        forged = _rehash_plan(
            plan,
            rank_estimates=estimates,
            device_peak_bytes=device_peaks,
            host_peak_bytes=host_peaks,
            host_required_bytes=sum(host_peaks),
        )

        with pytest.raises(ValueError, match="deterministic residency plan"):
            forged.validate_request(request)
    finally:
        store.close()


def test_future_placement_prefetch_deduplicates_current_and_cross_plan_identity():
    request, store = _request()
    matrix = store.ref("input_0")
    future = store.put("next", np.arange(16, dtype=np.float64).reshape(4, 4))
    overlap = TensorPlacement(
        key="input_0",
        rank=0,
        source_rank=0,
        ranges=(SliceRange(0, 4), SliceRange(0, 4)),
        nbytes=128,
        layout="strided",
    )
    new = TensorPlacement(
        key="next",
        rank=0,
        source_rank=0,
        ranges=(SliceRange(0, 4), SliceRange(0, 4)),
        nbytes=128,
        layout="C",
    )
    future_plan = FutureResidencyPlan(
        source_plan_hash="1" * 64,
        required_refs=(future, matrix),
        local_slices=(new, overlap),
    )
    request = replace(
        request,
        store_bytes=None,
        store_snapshots=(store.snapshot(), store.snapshot()),
        prefetch_depth=2,
        future_plans=(future_plan, future_plan),
    )

    plan = ResidencyPlanner().plan(request)

    assert plan.prefetch_keys == ("input_0", "next")
    assert plan.future_plans == (future_plan, future_plan)
    assert plan.future_plan_hashes == (future_plan.plan_hash, future_plan.plan_hash)
    assert plan.rank_estimates[0].prefetched_static_bytes == 128
    assert plan.rank_estimates[1].prefetched_static_bytes == 0


def test_retained_future_placements_drive_replica_prediction_and_plan_hash():
    request, store = _request()
    matrix = store.ref("input_0")
    full_rank_0 = TensorPlacement(
        "input_0",
        0,
        0,
        (SliceRange(0, 8), SliceRange(0, 8)),
        matrix.nbytes,
        "C",
    )
    full_rank_1 = replace(full_rank_0, rank=1, source_rank=1)
    replicated = FutureResidencyPlan(
        source_plan_hash="2" * 64,
        required_refs=(matrix,),
        local_slices=(full_rank_1, full_rank_0),
    )
    partial = FutureResidencyPlan(
        source_plan_hash="2" * 64,
        required_refs=(matrix,),
        local_slices=(full_rank_0,),
    )
    overlapping_partial = FutureResidencyPlan(
        source_plan_hash="2" * 64,
        required_refs=(matrix,),
        local_slices=tuple(
            TensorPlacement(
                "input_0",
                rank,
                source_rank,
                ranges,
                rows * 8 * 8,
                "C",
            )
            for rank in range(2)
            for source_rank, ranges, rows in (
                (0, (SliceRange(0, 5), SliceRange(0, 8)), 5),
                (1, (SliceRange(3, 6), SliceRange(0, 8)), 3),
            )
        ),
    )

    replicated_plan = ResidencyPlanner().plan(
        replace(
            request,
            future_plans=(replicated,),
            transfer_staging_host_bytes=None,
        )
    )
    partial_plan = ResidencyPlanner().plan(
        replace(request, future_plans=(partial,), transfer_staging_host_bytes=None)
    )
    overlap_plan = ResidencyPlanner().plan(
        replace(
            request,
            future_plans=(overlapping_partial,),
            transfer_staging_host_bytes=None,
        )
    )

    assert replicated_plan.future_plans == (replicated,)
    assert replicated_plan.metadata()["full_replica_prediction"] is True
    assert partial_plan.metadata()["full_replica_prediction"] is False
    assert overlap_plan.metadata()["full_replica_prediction"] is False
    assert replicated_plan.plan_hash != partial_plan.plan_hash


def test_device_and_job_global_host_budget_errors_report_precise_context():
    request, _ = _request()
    baseline = ResidencyPlanner().plan(request)
    device_request = replace(
        request,
        device_budget=_explicit_budget(baseline.device_peak_bytes[0] - 1, "device"),
    )
    host_request = replace(
        request,
        host_budget=_explicit_budget(baseline.host_required_bytes - 1, "host"),
    )

    with pytest.raises(ResidencyBudgetError, match="device memory budget") as device:
        ResidencyPlanner().plan(device_request)
    with pytest.raises(ResidencyBudgetError, match="host memory budget") as host:
        ResidencyPlanner().plan(host_request)

    assert device.value.resource == "device"
    assert device.value.rank == 0
    assert device.value.required_bytes == baseline.device_peak_bytes[0]
    assert host.value.resource == "host"
    assert host.value.rank is None
    assert host.value.required_bytes == baseline.host_required_bytes


def test_host_budget_boundary_includes_persistent_execution_status():
    request, store = _request(kind="krylov")
    try:
        baseline = ResidencyPlanner().plan(request)
        expected_required = sum(
            _expected_rank_host_peak(request, baseline, rank)
            for rank in range(request.world_size)
        )

        with pytest.raises(ResidencyBudgetError, match="host memory budget"):
            ResidencyPlanner().plan(
                replace(
                    request,
                    host_budget=_explicit_budget(expected_required - 1, "host"),
                )
            )
        exact = ResidencyPlanner().plan(
            replace(
                request,
                host_budget=_explicit_budget(expected_required, "host"),
            )
        )
    finally:
        store.close()

    assert exact.host_required_bytes == expected_required


def test_planner_rejects_world_mismatch_empty_mapped_shard_and_unbounded_solver():
    request, _ = _request()

    with pytest.raises(ValueError, match="world_size"):
        ResidencyPlanner().plan(replace(request, local_world_size=1))
    with pytest.raises(NotImplementedError, match="empty mapped shard"):
        ResidencyPlanner().plan(replace(request, mapped_local_counts=(4, 0)))
    with pytest.raises(NotImplementedError, match="unbounded solver"):
        ResidencyPlanner().plan(
            replace(
                request,
                solver_profile=build_krylov_memory_profile(
                    request.solver_input_sharding,
                    "float64",
                    block_size=4,
                ),
            )
        )


def test_plan_store_validation_detects_version_change_without_reading_values():
    request, store = _request()
    plan = ResidencyPlanner().plan(request)

    plan.validate_store(store)
    store.update("output", np.ones(8), expected_version=0)

    with pytest.raises(RuntimeError, match="stale"):
        plan.validate_store(store)


def test_future_plan_rejects_ambiguous_refs_out_of_bounds_and_false_bytes():
    _, store = _request()
    ref = store.put("next", np.ones((4, 4), dtype=np.float64))
    foreign_store = HostTensorStore(store_id="other-run")
    duplicate_key = foreign_store.put("next", np.ones((4, 4), dtype=np.float64))

    with pytest.raises(ValueError, match="duplicate.*key"):
        FutureResidencyPlan(
            source_plan_hash="1" * 64,
            required_refs=(ref, duplicate_key),
            local_slices=(),
        )
    with pytest.raises(ValueError, match="bounds"):
        FutureResidencyPlan(
            source_plan_hash="1" * 64,
            required_refs=(ref,),
            local_slices=(
                TensorPlacement(
                    "next",
                    0,
                    0,
                    (SliceRange(0, 5), SliceRange(0, 4)),
                    160,
                    "C",
                ),
            ),
        )
    with pytest.raises(ValueError, match="nbytes"):
        FutureResidencyPlan(
            source_plan_hash="1" * 64,
            required_refs=(ref,),
            local_slices=(
                TensorPlacement(
                    "next",
                    0,
                    0,
                    (SliceRange(0, 4), SliceRange(0, 4)),
                    64,
                    "C",
                ),
            ),
        )


def test_request_rejects_host_ref_metadata_that_disagrees_with_lowered_plan():
    request, _ = _request()
    wrong_store = HostTensorStore(store_id="shared-run")
    wrong = wrong_store.put("input_0", np.ones((4, 4), dtype=np.float32))
    refs = dict(request.host_refs)
    refs["input_0"] = wrong

    with pytest.raises(ValueError, match="host ref metadata"):
        replace(request, host_refs=refs)


def test_request_derives_center_bytes_and_binds_solver_profile_identity():
    request, _ = _request()

    derived = replace(request, complete_center_bytes=None)

    assert derived.complete_center_bytes == 64
    assert derived.center_profile.complete_center_bytes == 64
    assert (
        derived.center_profile.input_sharding == request.distributed_plan.input_sharding
    )
    assert (
        derived.center_profile.output_sharding
        == request.distributed_plan.output_sharding
    )
    assert derived.center_profile.solver_sharding == request.solver_input_sharding
    assert derived.solver_profile.rank_counts == request.mapped_local_counts


def test_request_rejects_solver_shape_dtype_count_and_center_byte_mismatch():
    request, _ = _request()
    wrong_shape = build_krylov_memory_profile(
        shard_axis((10,), 0, 2),
        "float64",
        max_krylov_vectors=4,
    )
    wrong_dtype = build_krylov_memory_profile(
        request.solver_input_sharding,
        "float32",
        max_krylov_vectors=4,
    )

    with pytest.raises(ValueError, match="solver profile.*sharding"):
        replace(request, solver_profile=wrong_shape)
    with pytest.raises(ValueError, match="solver profile.*dtype"):
        replace(request, solver_profile=wrong_dtype)
    with pytest.raises(ValueError, match="complete_center_bytes"):
        replace(request, complete_center_bytes=32)


def test_planner_rebuilds_and_exact_compares_solver_profile_components():
    request, _ = _request()
    first_rank = request.solver_profile.rank_estimates[0]
    forged_rank = replace(
        request.solver_profile,
        rank_estimates=(
            replace(first_rank, hv_retained_bytes=first_rank.hv_retained_bytes + 8),
            request.solver_profile.rank_estimates[1],
        ),
    )
    forged_control = replace(request.solver_profile, block_size=5)

    with pytest.raises(ValueError, match="canonical solver profile"):
        ResidencyPlanner().plan(replace(request, solver_profile=forged_rank))
    with pytest.raises(ValueError, match="canonical solver profile"):
        ResidencyPlanner().plan(replace(request, solver_profile=forged_control))


def test_request_derives_store_and_transaction_overlap_from_complete_metadata():
    request, store = _request()
    snapshot = store.snapshot()
    allocation = HostTensorAllocation("output", (8,), "float64")
    snapshots = (snapshot, snapshot)
    allocations = ((allocation,), (allocation,))

    derived = replace(
        request,
        store_bytes=None,
        dirty_writeback_bytes=None,
        store_snapshots=snapshots,
        writeback_allocations=allocations,
    )

    assert derived.store_bytes == (snapshot.current_bytes,) * 2
    assert derived.dirty_writeback_bytes == (allocation.nbytes,) * 2
    assert (
        derived.transactional_store_bytes
        == (snapshot.current_bytes + allocation.nbytes,) * 2
    )
    with pytest.raises(ValueError, match="store_bytes.*canonical"):
        replace(
            derived,
            store_bytes=(snapshot.current_bytes + 1, snapshot.current_bytes),
        )
    incomplete = HostTensorStoreSnapshot(
        snapshot.store_id,
        tuple(ref for ref in snapshot.refs if ref.key != "output"),
    )
    with pytest.raises(ValueError, match="snapshot.*host refs"):
        replace(derived, store_snapshots=(incomplete, snapshot))
    wrong = HostTensorAllocation("output", (4,), "float64")
    with pytest.raises(ValueError, match="writeback allocation.*output"):
        replace(derived, writeback_allocations=((wrong,), (allocation,)))


def test_plan_metadata_is_bounded_and_omits_complete_keys_and_slices():
    request, _ = _request()
    plan = ResidencyPlanner().plan(request)

    metadata = plan.metadata()

    assert metadata["plan_hash"] == plan.plan_hash
    assert metadata["source_plan_hash"] == plan.source_plan_hash
    assert metadata["placement_hash"] == plan.placement_hash
    assert metadata["prefetch_key_count"] == 0
    assert metadata["store_ref_count"] == len(plan.required_refs)
    assert len(metadata["rank_estimates"]) == plan.world_size
    assert "required_refs" not in metadata
    assert "local_slices" not in metadata
    assert "prefetch_keys" not in metadata


def test_wave9_request_and_plan_hash_bind_complete_one_node_runtime_identity():
    request, store = _request()
    try:
        default_plan = ResidencyPlanner().plan(request)
        custom_identity = ResidencyRuntimeIdentity.one_node(
            world_size=2,
            backend_name="cupy",
            mesh_shape=(1, 2),
            mesh_axis_names=("node", "rank"),
        )
        custom_request = replace(request, runtime_identity=custom_identity)
        custom_plan = ResidencyPlanner().plan(custom_request)
    finally:
        store.close()

    identity = request.runtime_identity
    assert identity.mesh_shape == (2,)
    assert identity.mesh_axis_names == ("rank",)
    assert identity.rank_to_node == (0, 0)
    assert identity.rank_to_local_rank == (0, 1)
    assert identity.rank_to_device == ("cuda:0", "cuda:1")
    assert identity.backend_name == "cupy"
    assert identity.device_budget_bindings == (
        "device:cupy:cuda:0",
        "device:cupy:cuda:1",
    )
    assert identity.host_budget_bindings == ("host:node:0",)
    assert request.request_hash != custom_request.request_hash
    assert default_plan.runtime_identity == identity
    assert custom_plan.runtime_identity == custom_identity
    assert default_plan.plan_hash != custom_plan.plan_hash


def test_wave8_numpy_device_bytes_share_node_host_budget():
    request, store = _request(backend_name="numpy")
    generous = replace(
        request,
        host_budget=_explicit_budget(np.iinfo(np.int64).max, "host"),
    )
    try:
        baseline = ResidencyPlanner().plan(generous)
        expected = sum(baseline.host_peak_bytes) + sum(baseline.device_peak_bytes)

        with pytest.raises(ResidencyBudgetError, match="host memory budget"):
            ResidencyPlanner().plan(
                replace(
                    request,
                    host_budget=_explicit_budget(expected - 1, "host"),
                )
            )
        exact = ResidencyPlanner().plan(
            replace(
                request,
                device_budget=_explicit_budget(1, "device"),
                host_budget=_explicit_budget(expected, "host"),
            )
        )
    finally:
        store.close()

    assert exact.host_required_bytes == expected


@pytest.mark.parametrize(
    ("backend_name", "budget_name", "resource", "required"),
    [
        ("cupy", "device_budget", "device", lambda plan: max(plan.device_peak_bytes)),
        ("numpy", "host_budget", "host", lambda plan: plan.host_required_bytes),
    ],
)
def test_wave9_plan_construction_rejects_retained_peaks_above_bound_budget(
    backend_name, budget_name, resource, required
):
    request, store = _request(backend_name=backend_name)
    try:
        plan = ResidencyPlanner().plan(request)
        too_small = _explicit_budget(required(plan) - 1, resource)

        with pytest.raises(
            ResidencyBudgetError, match="{} memory budget".format(resource)
        ):
            _rehash_plan(plan, **{budget_name: too_small})
    finally:
        store.close()


def test_wave8_checked_memory_arithmetic_rejects_overflow():
    request, store = _request()
    maximum = np.iinfo(np.int64).max
    try:
        with pytest.raises(OverflowError, match="memory|int64|range"):
            ResidencyPlanner().plan(
                replace(
                    request,
                    external_host_bytes=(maximum, maximum),
                    host_budget=_explicit_budget(maximum, "host"),
                )
            )
    finally:
        store.close()


def _wave6_request(kind):
    dtype = np.dtype("float32")
    source = lower_einsum_path("ab,b->a", ((8, 8), (8,)), dtype=dtype.name)
    distributed = plan_distributed_execution(
        source, variable_key="input_1", world_size=2
    )
    vector_map = CenterVectorMap(
        distributed.input_sharding,
        distributed.output_sharding,
        np.asarray([True, True, True, False, True, True, True, False]),
    )
    solver_sharding = vector_map.solver_sharding
    if kind == "krylov":
        solver_profile = build_krylov_memory_profile(
            solver_sharding,
            dtype,
            coefficient=-0.125j,
            block_size=2,
            max_krylov_vectors=3,
        )
    else:
        solver_profile = build_davidson_memory_profile(
            solver_sharding, dtype, max_space=3
        )
    store = HostTensorStore(store_id="wave6-shared-run")
    matrix = store.put("input_0", np.eye(8, dtype=dtype))
    center = store.put("output", np.zeros(8, dtype=dtype))
    snapshot = store.snapshot()
    output_allocation = HostTensorAllocation(
        "output", (8,), solver_profile.result_dtype
    )
    request = ResidencyRequest(
        distributed_plan=distributed,
        host_refs={"output": center, "input_0": matrix},
        world_size=2,
        local_world_size=2,
        backend_name="cupy",
        store_bytes=None,
        external_host_bytes=(0, 0),
        transfer_staging_host_bytes=None,
        dirty_writeback_bytes=None,
        solver_input_sharding=solver_sharding,
        solver_output_sharding=solver_sharding,
        mapped_local_counts=(3, 3),
        solver_profile=solver_profile,
        materialization_policy="device",
        complete_center_bytes=8 * solver_profile.result_itemsize,
        prefetch_depth=1,
        future_plans=(),
        device_budget=_explicit_budget(1 << 30, "device"),
        host_budget=_explicit_budget(1 << 31, "host"),
        store_snapshots=(snapshot, snapshot),
        writeback_allocations=(
            (output_allocation,),
            (output_allocation,),
        ),
        qn_mask_present=True,
        qn_mask_identity=vector_map.qn_mask_identity,
    )
    return request, store


@pytest.mark.parametrize(
    ("kind", "result_dtype", "result_itemsize", "complete_input_count"),
    [
        ("krylov", "complex128", 16, 1),
        ("davidson", "float64", 8, 2),
    ],
)
def test_wave6_temporal_phases_retain_inputs_qn_results_and_state_controls(
    kind, result_dtype, result_itemsize, complete_input_count
):
    request, store = _wave6_request(kind)
    try:
        plan = ResidencyPlanner().plan(request)
    finally:
        store.close()

    assert request.solver_profile.result_dtype == result_dtype
    assert request.solver_profile.result_itemsize == result_itemsize
    for rank, estimate in enumerate(plan.rank_estimates):
        complete_input_bytes = 8 * 4
        assert estimate.complete_input_center_bytes == complete_input_bytes
        assert estimate.complete_diagonal_center_bytes == (
            complete_input_bytes if kind == "davidson" else 0
        )
        assert (
            estimate.complete_input_center_bytes
            + estimate.complete_diagonal_center_bytes
            == complete_input_count * complete_input_bytes
        )
        assert estimate.solver_input_bytes == 3 * 4
        assert estimate.solver_diagonal_bytes == (3 * 4 if kind == "davidson" else 0)
        assert estimate.solver_result_bytes == 3 * result_itemsize
        assert estimate.complete_center_bytes == 8 * result_itemsize
        assert estimate.qn_setup_mask_device_bytes == 8
        assert estimate.qn_extract_mask_device_bytes == 4
        assert estimate.qn_hv_mask_device_bytes == 4
        assert estimate.qn_materialization_mask_device_bytes == 4
        assert estimate.qn_writeback_mask_device_bytes == 8
        assert estimate.qn_writeback_packed_result_bytes == 6 * result_itemsize
        assert estimate.state_digest_host_bytes == 2 * 8 * result_itemsize
        assert estimate.state_agreement_device_bytes == 104
        assert estimate.state_agreement_host_bytes == 96

        execution = (
            estimate.current_static_bytes
            + estimate.prefetched_static_bytes
            + estimate.dense_input_bytes
            + estimate.receive_buffer_bytes
            + estimate.output_accumulator_bytes
            + estimate.task14_execution_status_device_bytes
            + estimate.mapped_packed_output_bytes
        )
        local_inputs = estimate.solver_input_bytes + estimate.solver_diagonal_bytes
        complete_inputs = (
            estimate.complete_input_center_bytes
            + estimate.complete_diagonal_center_bytes
            + estimate.original_packed_guess_bytes
            + estimate.original_packed_diagonal_bytes
        )
        post_solver = (
            execution
            + complete_inputs
            + local_inputs
            + estimate.solver_result_bytes
            + estimate.complete_center_bytes
        )
        assert estimate.materialization_device_peak_bytes == (
            post_solver
            + estimate.materialization_receive_bytes
            + estimate.materialization_control_device_bytes
            + estimate.qn_materialization_mask_device_bytes
        )
        assert estimate.state_digest_device_peak_bytes == post_solver
        assert estimate.state_agreement_device_peak_bytes == post_solver + 104
        assert estimate.writeback_device_peak_bytes == (
            post_solver
            + estimate.qn_writeback_mask_device_bytes
            + estimate.qn_writeback_packed_result_bytes
        )

        host_base = (
            estimate.store_bytes
            + estimate.external_host_bytes
            + estimate.transfer_staging_host_bytes
            + estimate.qn_persistent_mask_host_bytes
            + estimate.task14_persistent_execution_status_host_bytes
        )
        assert estimate.state_digest_host_peak_bytes == (
            host_base + estimate.state_digest_host_bytes
        )
        assert estimate.state_agreement_host_peak_bytes == host_base + 96


@pytest.mark.parametrize("kind", ["krylov", "davidson"])
def test_wave6_exact_budget_boundaries_include_state_and_promoted_writeback(kind):
    request, store = _wave6_request(kind)
    try:
        baseline = ResidencyPlanner().plan(request)
        exact_device = max(baseline.device_peak_bytes)
        exact_host = baseline.host_required_bytes
        exact = ResidencyPlanner().plan(
            replace(
                request,
                device_budget=_explicit_budget(exact_device, "device"),
                host_budget=_explicit_budget(exact_host, "host"),
            )
        )
        assert exact.device_peak_bytes == baseline.device_peak_bytes
        assert exact.host_required_bytes == baseline.host_required_bytes
        with pytest.raises(ResidencyBudgetError, match="device memory budget"):
            ResidencyPlanner().plan(
                replace(
                    request,
                    device_budget=_explicit_budget(exact_device - 1, "device"),
                )
            )
        with pytest.raises(ResidencyBudgetError, match="host memory budget"):
            ResidencyPlanner().plan(
                replace(
                    request,
                    host_budget=_explicit_budget(exact_host - 1, "host"),
                )
            )
    finally:
        store.close()


def test_wave6_canonical_profiles_hash_new_identity_and_reject_forgery():
    request, store = _wave6_request("krylov")
    try:
        with pytest.raises(ValueError, match="scalar dtype"):
            replace(
                request.solver_profile,
                scalar_dtype="complex64",
                scalar_itemsize=8,
                result_dtype="complex64",
                result_itemsize=8,
            )
        first_rank = request.solver_profile.rank_estimates[0]
        forged_solver = replace(
            request.solver_profile,
            rank_estimates=(
                replace(
                    first_rank,
                    la_transient_bytes=first_rank.la_transient_bytes + 1,
                ),
                request.solver_profile.rank_estimates[1],
            ),
        )
        forged_request = replace(request, solver_profile=forged_solver)
        with pytest.raises(ValueError, match="canonical solver profile"):
            ResidencyPlanner().plan(forged_request)

        dense_request, dense_store = _request()
        try:
            all_true_map = CenterVectorMap(
                dense_request.distributed_plan.input_sharding,
                dense_request.distributed_plan.output_sharding,
                np.ones(8, dtype=bool),
            )
            all_true_qn_request = replace(
                dense_request,
                qn_mask_present=True,
                qn_mask_identity=all_true_map.qn_mask_identity,
                center_profile=None,
            )
            assert all_true_qn_request.request_hash != dense_request.request_hash
            assert all_true_qn_request.center_profile != dense_request.center_profile
        finally:
            dense_store.close()
    finally:
        store.close()


def test_wave7_all_true_qn_identity_drives_exact_host_liveness_and_boundaries():
    dense_request, store = _request()
    vector_map = CenterVectorMap(
        dense_request.distributed_plan.input_sharding,
        dense_request.distributed_plan.output_sharding,
        np.ones(8, dtype=bool),
    )
    qn_profile = center_materialization_memory_profile(
        vector_map,
        dense_request.solver_profile.dtype,
        result_dtype=dense_request.solver_profile.result_dtype,
    )
    qn_request = replace(
        dense_request,
        qn_mask_present=True,
        qn_mask_identity=vector_map.qn_mask_identity,
        center_profile=qn_profile,
    )
    try:
        dense_plan = ResidencyPlanner().plan(dense_request)
        qn_plan = ResidencyPlanner().plan(qn_request)
        exact = ResidencyPlanner().plan(
            replace(
                qn_request,
                host_budget=_explicit_budget(qn_plan.host_required_bytes, "host"),
            )
        )
        assert exact.host_required_bytes == qn_plan.host_required_bytes
        with pytest.raises(ResidencyBudgetError, match="host memory budget"):
            ResidencyPlanner().plan(
                replace(
                    qn_request,
                    host_budget=_explicit_budget(
                        qn_plan.host_required_bytes - 1, "host"
                    ),
                )
            )
    finally:
        store.close()

    assert qn_request.request_hash != dense_request.request_hash
    for dense, qn in zip(dense_plan.rank_estimates, qn_plan.rank_estimates):
        assert qn.qn_setup_mask_device_bytes == 8
        assert qn.qn_extract_mask_device_bytes == 4
        assert qn.qn_persistent_mask_host_bytes == 16
        assert qn.qn_setup_digest_host_bytes == 8
        assert qn.setup_host_peak_bytes == dense.setup_host_peak_bytes + 24
        assert qn.hv_host_peak_bytes == dense.hv_host_peak_bytes + 16
        assert qn.solver_la_host_peak_bytes == dense.solver_la_host_peak_bytes + 16
        assert qn.state_digest_host_peak_bytes == (
            dense.state_digest_host_peak_bytes + 16
        )


def test_wave7_planner_hashes_exact_krylov_coefficient_identity():
    request, store = _request()
    first_profile = build_krylov_memory_profile(
        request.solver_input_sharding,
        "float64",
        coefficient=0.25,
        block_size=4,
        max_krylov_vectors=4,
    )
    second_profile = build_krylov_memory_profile(
        request.solver_input_sharding,
        "float64",
        coefficient=0.5,
        block_size=4,
        max_krylov_vectors=4,
    )
    try:
        first_request = replace(request, solver_profile=first_profile)
        second_request = replace(request, solver_profile=second_profile)
        first_plan = ResidencyPlanner().plan(first_request)
        second_plan = ResidencyPlanner().plan(second_request)
    finally:
        store.close()

    assert first_request.request_hash != second_request.request_hash
    assert first_plan.plan_hash != second_plan.plan_hash
    assert first_plan.krylov_coefficient_identity == first_profile.coefficient_identity
    assert (
        second_plan.krylov_coefficient_identity == second_profile.coefficient_identity
    )


def test_wave7_davidson_profiles_caller_owned_complete_packed_inputs():
    request, store = _request(kind="davidson")
    try:
        plan = ResidencyPlanner().plan(request)
    finally:
        store.close()

    for estimate in plan.rank_estimates:
        assert estimate.original_packed_guess_bytes == 8 * 8
        assert estimate.original_packed_diagonal_bytes == 8 * 8
        assert estimate.setup_device_peak_bytes >= (
            estimate.original_packed_guess_bytes
            + estimate.original_packed_diagonal_bytes
            + estimate.complete_input_center_bytes
            + estimate.complete_diagonal_center_bytes
        )


def test_wave7_residency_plan_rejects_noncanonical_current_refs_before_hashing():
    request, store = _request()
    try:
        plan = ResidencyPlanner().plan(request)
    finally:
        store.close()
    reversed_refs = tuple(reversed(plan.current_refs))
    assert reversed_refs != plan.current_refs
    provisional = object.__new__(type(plan))
    for field_info in dataclasses.fields(plan):
        object.__setattr__(
            provisional,
            field_info.name,
            reversed_refs
            if field_info.name == "current_refs"
            else getattr(plan, field_info.name),
        )
    forged_hash = _sha256(_plan_payload(provisional))

    with pytest.raises(ValueError, match="current_refs.*canonical"):
        replace(plan, current_refs=reversed_refs, plan_hash=forged_hash)


def test_wave7_plan_hash_binds_exact_equal_count_qn_mask_identity():
    first_request, store = _wave6_request("krylov")
    dense_sharding = first_request.distributed_plan.input_sharding
    second_map = CenterVectorMap(
        dense_sharding,
        first_request.distributed_plan.output_sharding,
        np.asarray([False, True, True, True, False, True, True, True]),
    )
    all_true_map = CenterVectorMap(
        dense_sharding,
        first_request.distributed_plan.output_sharding,
        np.ones(8, dtype=bool),
    )
    assert second_map.rank_counts == first_request.mapped_local_counts
    assert second_map.qn_mask_identity != first_request.qn_mask_identity

    try:
        second_request = replace(
            first_request,
            qn_mask_identity=second_map.qn_mask_identity,
            center_profile=None,
        )
        first_plan = ResidencyPlanner().plan(first_request)
        second_plan = ResidencyPlanner().plan(second_request)
        with pytest.raises(ValueError, match="QN mask identity.*request"):
            replace(
                first_request,
                qn_mask_identity=all_true_map.qn_mask_identity,
                center_profile=None,
            )
        with pytest.raises(ValueError, match="plan QN mask identity"):
            replace(first_plan, qn_mask_identity=all_true_map.qn_mask_identity)
    finally:
        store.close()

    assert first_request.request_hash != second_request.request_hash
    assert first_plan.qn_mask_identity == first_request.qn_mask_identity
    assert second_plan.qn_mask_identity == second_request.qn_mask_identity
    assert first_plan.plan_hash != second_plan.plan_hash
