import numpy as np
import pytest
from dataclasses import replace

import renormalizer.backend._distributed.center as center_module
import renormalizer.backend._distributed.solvers as solver_module

from renormalizer.backend._distributed.center import (
    CenterMaterializationMemoryProfile,
    CenterVectorMap,
    DavidsonRootFallbackMemoryProfile,
    KrylovRootFallbackMemoryProfile,
    center_materialization_memory_profile,
    davidson_root_fallback_memory_profile,
    krylov_root_fallback_memory_profile,
)
from renormalizer.backend._distributed.sharding import shard_axis
from renormalizer.backend._distributed.sharding import DistributedTensor
from renormalizer.backend._distributed.collectives import SingleProcessCollective
from renormalizer.backend._distributed.context import DistributedContext
from renormalizer.backend._distributed.solvers import (
    SolverMemoryProfile,
    SolverRankMemoryEstimate,
    build_davidson_memory_profile,
    build_krylov_memory_profile,
    run_sharded_krylov,
)
from renormalizer.backend.config import BackendConfig
from renormalizer.backend.factory import create_backend
from renormalizer.lib.davidson.backend import davidson_backend
from renormalizer.lib.krylov.krylov import expm_krylov


def test_profile_identity_fields_preserve_task15_positional_constructors():
    rank = SolverRankMemoryEstimate(1, 2, 3, 4, 5, 6)

    solver = SolverMemoryProfile("krylov", (rank,), True, 1)
    center = CenterMaterializationMemoryProfile("device", 8, (4,), (4,))
    fallback = DavidsonRootFallbackMemoryProfile(16, 32, 8)

    assert solver.sharding is None
    assert center.input_sharding is None
    assert fallback.dtype is None


def test_wave9_solver_global_size_product_rejects_oversized_metadata():
    with pytest.raises(OverflowError, match="int64|range"):
        solver_module._checked_shape_elements((1 << 62, 4), "solver global shape")


def test_wave6_identity_extensions_preserve_complete_pre_wave6_constructors():
    sharding = shard_axis((2,), 0, 1)
    rank = SolverRankMemoryEstimate(32, 16, 32, 16, 96, 128)

    solver = SolverMemoryProfile(
        "krylov",
        (rank,),
        True,
        2,
        global_shape=(2,),
        sharding=sharding,
        dtype="float32",
        itemsize=4,
        global_vector_count=2,
        rank_counts=(2,),
        allocation_vectors=2,
        projected_host_bytes=128,
        block_size=2,
        max_krylov_vectors=2,
    )
    center = CenterMaterializationMemoryProfile(
        "device",
        8,
        (8,),
        (12,),
        global_shape=(2,),
        input_sharding=sharding,
        output_sharding=sharding,
        solver_sharding=sharding,
        dtype="float32",
        itemsize=4,
        rank_counts=(2,),
        control_host_bytes=(4,),
    )
    krylov_fallback = KrylovRootFallbackMemoryProfile(
        "float32", 4, 2, 8, 2, 2, 2, 128, 512, 256
    )
    davidson_fallback = DavidsonRootFallbackMemoryProfile(
        64,
        512,
        256,
        dtype="float64",
        itemsize=8,
        packed_vector_count=2,
        packed_vector_bytes=16,
        full_center_bytes=16,
        concurrent_complete_center_bytes=80,
        mask_bytes=2,
        max_space=2,
    )

    assert solver.scalar_dtype == "float64"
    assert solver.result_dtype == "float64"
    assert center.input_dtype == "float32"
    assert center.qn_mask_present is False
    assert center.state_digest_host_bytes == 16
    assert center.state_agreement_device_bytes == (104,)
    assert krylov_fallback.scalar_dtype == "float64"
    assert krylov_fallback.result_dtype == "float64"
    assert krylov_fallback.result_vector_bytes == 16
    assert davidson_fallback.result_dtype == "float64"
    assert davidson_fallback.packed_result_bytes == 16
    assert davidson_fallback.full_result_center_bytes == 16

    with pytest.raises(ValueError, match="result itemsize"):
        replace(solver, result_dtype="float32")
    with pytest.raises(ValueError, match="identity metadata must be complete"):
        replace(
            solver,
            scalar_dtype=None,
            scalar_itemsize=None,
            result_dtype="float32",
            result_itemsize=None,
        )
    with pytest.raises(ValueError, match="identity metadata must be complete"):
        replace(center, state_digest_host_bytes=None)
    with pytest.raises(ValueError, match="scalar dtype"):
        replace(
            solver,
            scalar_dtype="complex64",
            scalar_itemsize=8,
            result_dtype="complex64",
            result_itemsize=8,
        )
    with pytest.raises(ValueError, match="scalar dtype"):
        replace(
            krylov_fallback,
            scalar_dtype="complex64",
            scalar_itemsize=8,
            result_dtype="complex64",
            result_itemsize=8,
            result_vector_bytes=16,
        )


def test_krylov_profile_requires_explicit_active_bound_and_counts_stacked_basis():
    sharding = shard_axis((10,), 0, 2)

    unbounded = build_krylov_memory_profile(
        sharding, np.dtype("complex128"), block_size=4
    )
    bounded = build_krylov_memory_profile(
        sharding,
        np.dtype("complex128"),
        block_size=4,
        max_krylov_vectors=3,
    )

    assert unbounded.bounded is False
    assert unbounded.basis_vectors is None
    assert bounded.bounded is True
    assert bounded.basis_vectors == 3
    assert bounded.global_shape == (10,)
    assert bounded.sharding == sharding
    assert bounded.dtype == "complex128"
    assert bounded.itemsize == 16
    assert bounded.global_vector_count == 10
    assert bounded.rank_counts == (5, 5)
    local_bytes = 5 * 16
    rank = bounded.rank_estimates[0]
    assert rank.hv_retained_bytes >= 3 * local_bytes
    assert rank.la_retained_bytes >= 3 * local_bytes
    assert rank.la_transient_bytes >= 3 * local_bytes
    assert rank.communication_device_bytes >= 2 * local_bytes


def test_davidson_profile_counts_basis_and_copied_applied_basis_exactly():
    sharding = shard_axis((10,), 0, 2)

    profile = build_davidson_memory_profile(sharding, np.dtype("float64"), max_space=4)

    assert profile.bounded is True
    assert profile.basis_vectors == 4
    local_bytes = 5 * 8
    rank = profile.rank_estimates[0]
    assert rank.hv_retained_bytes >= (2 * 4 + 4) * local_bytes + 5
    assert rank.la_retained_bytes >= (2 * 4 + 5) * local_bytes + 5
    assert rank.la_transient_bytes >= 2 * local_bytes
    assert rank.communication_device_bytes >= 2 * local_bytes


def test_davidson_projected_host_bound_uses_allocated_max_space_matrix():
    sharding = shard_axis((2,), 0, 2)

    profile = build_davidson_memory_profile(
        sharding, np.dtype("complex128"), max_space=12
    )

    # The recurrence allocates projected at max_space, even though K is only two.
    explicit_projected = 12 * 12 * 16
    eigh_input_and_vectors = 2 * 2 * 2 * 16
    eigenvalues = 2 * 8
    assert profile.projected_host_bytes >= (
        explicit_projected + eigh_input_and_vectors + eigenvalues
    )
    assert all(
        rank.host_peak_bytes >= profile.projected_host_bytes
        for rank in profile.rank_estimates
    )


def test_davidson_projected_host_bound_includes_scalar_dtype_promotion():
    sharding = shard_axis((2,), 0, 2)

    profile = build_davidson_memory_profile(sharding, np.dtype("float32"), max_space=12)

    promotion_peak = 12 * 12 * (4 + 8)
    assert profile.projected_host_bytes >= promotion_peak
    assert all(
        rank.host_peak_bytes == max(profile.projected_host_bytes, 3 * 5 * 8)
        for rank in profile.rank_estimates
    )


def test_solver_profiles_retain_controls_and_bound_simultaneous_control_arrays():
    sharding = shard_axis((2,), 0, 2)

    krylov = build_krylov_memory_profile(
        sharding,
        np.dtype("float64"),
        block_size=2,
        max_krylov_vectors=2,
    )
    davidson = build_davidson_memory_profile(sharding, np.dtype("float64"), max_space=2)

    assert krylov.block_size == 2
    assert krylov.max_krylov_vectors == 2
    assert krylov.max_space is None
    assert krylov.projected_host_bytes == 3 * 2 * 2 * 8 + 48 * 2 + 16 * 2
    assert all(
        rank.communication_device_bytes >= 3 * 4 * 8 + 2 * 4
        for rank in krylov.rank_estimates
    )
    assert all(rank.host_peak_bytes >= 3 * 4 * 8 for rank in krylov.rank_estimates)
    assert davidson.block_size is None
    assert davidson.max_krylov_vectors is None
    assert davidson.max_space == 2
    assert all(
        rank.communication_device_bytes >= 3 * 5 * 8 + 2 * 4
        for rank in davidson.rank_estimates
    )
    assert all(rank.host_peak_bytes >= 3 * 5 * 8 for rank in davidson.rank_estimates)

    forged = replace(
        krylov,
        rank_estimates=(
            replace(
                krylov.rank_estimates[0],
                communication_device_bytes=(
                    krylov.rank_estimates[0].communication_device_bytes + 1
                ),
            ),
            krylov.rank_estimates[1],
        ),
    )
    assert forged != krylov


def test_solver_profiles_bound_state_live_across_the_next_hv_call():
    sharding = shard_axis((8,), 0, 2)
    local_count = 4
    local_bytes = local_count * np.dtype(np.float64).itemsize

    krylov = build_krylov_memory_profile(
        sharding,
        np.dtype("float64"),
        block_size=2,
        max_krylov_vectors=3,
    ).rank_estimates[0]
    assert krylov.hv_retained_bytes == (3 + 2) * local_bytes
    assert krylov.hv_transient_bytes == 2 * local_bytes
    assert krylov.la_retained_bytes == (3 + 2) * local_bytes
    assert krylov.la_transient_bytes == (3 + 1) * local_bytes + 3 * 8
    assert krylov.communication_device_bytes == 3 * 4 * 8 + 2 * 4

    davidson = build_davidson_memory_profile(
        sharding, np.dtype("float64"), max_space=3
    ).rank_estimates[0]
    assert davidson.hv_retained_bytes == (2 * 3 + 4) * local_bytes + local_count
    assert davidson.hv_transient_bytes == 2 * local_bytes
    assert davidson.la_retained_bytes == (2 * 3 + 5) * local_bytes + local_count
    assert davidson.la_transient_bytes == 2 * local_bytes
    assert davidson.communication_device_bytes == 3 * 5 * 8 + 2 * 4


def test_profile_builders_reject_empty_or_incompatible_solver_metadata():
    sharding = shard_axis((4,), 0, 2)

    with pytest.raises(ValueError, match="max_krylov_vectors"):
        build_krylov_memory_profile(sharding, np.dtype("float64"), max_krylov_vectors=0)
    with pytest.raises(ValueError, match="max_space"):
        build_davidson_memory_profile(sharding, np.dtype("float64"), max_space=1)
    with pytest.raises(ValueError, match="dtype"):
        build_krylov_memory_profile(sharding, np.dtype("O"))


def test_explicit_krylov_bound_caps_recurrence_while_default_remains_global():
    matrix = np.diag(np.arange(1.0, 7.0))
    matrix += np.diag(np.full(5, -0.3), 1)
    matrix += np.diag(np.full(5, -0.3), -1)
    start = np.arange(1.0, 7.0)

    bounded, bounded_iterations = expm_krylov(
        matrix.dot,
        -0.1j,
        start,
        block_size=2,
        max_krylov_vectors=3,
    )
    default, default_iterations = expm_krylov(matrix.dot, -0.1j, start, block_size=2)

    assert bounded_iterations == 3
    assert bounded.shape == start.shape
    assert 3 <= default_iterations <= start.size
    np.testing.assert_array_equal(start, np.arange(1.0, 7.0))


def test_center_device_materialization_profile_matches_current_allocation_schedule():
    dense = shard_axis((8,), 0, 2)
    vector_map = CenterVectorMap(dense, dense)

    profile = center_materialization_memory_profile(
        vector_map, np.dtype("float64"), policy="device"
    )

    assert profile.policy == "device"
    assert profile.global_shape == (8,)
    assert profile.input_sharding == dense
    assert profile.output_sharding == dense
    assert profile.solver_sharding == vector_map.solver_sharding
    assert profile.dtype == "float64"
    assert profile.itemsize == 8
    assert profile.rank_counts == (4, 4)
    assert profile.complete_center_bytes == 64
    assert profile.receive_buffer_bytes == (32, 32)
    assert profile.control_device_bytes == (12, 12)
    assert profile.control_host_bytes == (4, 4)
    with pytest.raises(NotImplementedError, match="host center materialization"):
        center_materialization_memory_profile(
            vector_map, np.dtype("float64"), policy="host"
        )


def test_center_profile_rejects_dense_and_solver_world_size_mismatch():
    dense = shard_axis((8,), 0, 2)
    solver = shard_axis((8,), 0, 1)

    with pytest.raises(ValueError, match="parts"):
        CenterMaterializationMemoryProfile(
            "device",
            64,
            (64, 64),
            (8, 8),
            global_shape=(8,),
            input_sharding=dense,
            output_sharding=dense,
            solver_sharding=solver,
            dtype="float64",
            itemsize=8,
            rank_counts=(8,),
            control_host_bytes=(4, 4),
        )


def test_root_fallback_davidson_profile_counts_both_basis_lists():
    profile = davidson_root_fallback_memory_profile(
        packed_vector_bytes=80,
        packed_vector_count=10,
        full_center_bytes=160,
        mask_bytes=20,
        dtype=np.dtype("float64"),
        max_space=4,
    )

    assert profile.basis_bytes == 2 * 4 * 80
    assert profile.dtype == "float64"
    assert profile.itemsize == 8
    assert profile.packed_vector_count == 10
    assert profile.concurrent_complete_center_bytes == 5 * 160
    assert profile.device_peak_bytes >= profile.basis_bytes
    assert profile.host_peak_bytes > 0


def test_root_fallback_krylov_profile_bounds_three_fallback_matrices():
    profile = krylov_root_fallback_memory_profile(
        vector_bytes=16,
        vector_count=2,
        dtype=np.dtype("float64"),
        block_size=2,
        max_krylov_vectors=2,
    )

    assert profile.block_size == 2
    assert profile.max_krylov_vectors == 2
    assert profile.basis_vectors == 2
    assert profile.projected_host_bytes == 3 * 2 * 2 * 8 + 48 * 2 + 16 * 2
    assert profile.host_peak_bytes >= profile.projected_host_bytes
    assert profile.device_peak_bytes >= 2 * profile.vector_bytes


def test_narrow_solver_profiles_derive_actual_promoted_result_dtypes():
    sharding = shard_axis((4,), 0, 2)

    krylov = build_krylov_memory_profile(
        sharding,
        np.dtype("float32"),
        coefficient=-0.125j,
        block_size=2,
        max_krylov_vectors=2,
    )
    davidson = build_davidson_memory_profile(sharding, np.dtype("float32"), max_space=2)

    assert krylov.dtype == "float32"
    assert krylov.itemsize == 4
    assert krylov.scalar_dtype == "complex128"
    assert krylov.scalar_itemsize == 16
    assert krylov.result_dtype == "complex128"
    assert krylov.result_itemsize == 16
    assert davidson.dtype == "float32"
    assert davidson.itemsize == 4
    assert davidson.scalar_dtype == "float64"
    assert davidson.scalar_itemsize == 8
    assert davidson.result_dtype == "float64"
    assert davidson.result_itemsize == 8

    local_input_bytes = 2 * 4
    local_krylov_result_bytes = 2 * 16
    assert krylov.rank_estimates[0].hv_retained_bytes == (
        (krylov.basis_vectors + 1) * local_input_bytes + local_krylov_result_bytes
    )
    assert krylov.rank_estimates[0].la_transient_bytes == (
        krylov.basis_vectors * local_input_bytes
        + krylov.basis_vectors * krylov.scalar_itemsize
        + local_krylov_result_bytes
    )
    local_davidson_result_bytes = 2 * 8
    assert davidson.rank_estimates[0].hv_retained_bytes == (
        (2 * davidson.basis_vectors + 4) * local_davidson_result_bytes + 2
    )


def test_narrow_recurrences_return_promoted_results_without_narrowing():
    matrix = np.diag(np.asarray([1.0, 2.0], dtype=np.float32))
    start = np.asarray([1.0, 0.0], dtype=np.float32)

    krylov, _ = expm_krylov(
        matrix.dot,
        -0.125j,
        start,
        block_size=2,
        max_krylov_vectors=2,
    )
    _, davidson, _ = davidson_backend(
        matrix.dot,
        start,
        np.diag(matrix).copy(),
        max_cycle=2,
        max_space=2,
    )

    assert krylov.dtype == np.dtype("complex128")
    assert davidson.dtype == np.dtype("float64")


def test_krylov_profile_uses_recurrence_real_rule_for_zero_imaginary_scalar():
    sharding = shard_axis((2,), 0, 1)

    profile = build_krylov_memory_profile(
        sharding,
        np.dtype("float32"),
        coefficient=1.0 + 0.0j,
        block_size=2,
        max_krylov_vectors=2,
    )
    actual, _ = expm_krylov(
        np.eye(2, dtype=np.float32).dot,
        1.0 + 0.0j,
        np.asarray([1.0, 0.0], dtype=np.float32),
        block_size=2,
        max_krylov_vectors=2,
    )

    assert actual.dtype == np.dtype("float64")
    assert profile.scalar_dtype == "float64"
    assert profile.result_dtype == "float64"


def test_qn_materialization_profile_covers_all_mask_and_state_phases():
    dense = shard_axis((8,), 0, 2)
    mask = np.asarray([True, False, True, True, True, False, True, True])
    vector_map = CenterVectorMap(dense, dense, mask)

    profile = center_materialization_memory_profile(
        vector_map,
        np.dtype("float32"),
        result_dtype=np.dtype("complex128"),
        policy="device",
    )

    assert profile.input_dtype == "float32"
    assert profile.input_itemsize == 4
    assert profile.dtype == "complex128"
    assert profile.itemsize == 16
    assert profile.complete_center_bytes == 8 * 16
    assert profile.receive_buffer_bytes == (3 * 16, 3 * 16)
    assert profile.qn_mask_present is True
    assert profile.setup_qn_mask_bytes == 8
    assert profile.extract_qn_mask_bytes == (4, 4)
    assert profile.hv_qn_mask_bytes == (4, 4)
    assert profile.materialization_qn_mask_bytes == (4, 4)
    assert profile.writeback_qn_mask_bytes == 8
    assert profile.writeback_packed_result_bytes == 6 * 16
    assert profile.state_digest_host_bytes == 2 * 8 * 16
    assert profile.state_agreement_device_bytes == (104, 104)
    assert profile.state_agreement_host_bytes == (96, 96)

    with pytest.raises(ValueError, match="state agreement"):
        replace(profile, state_agreement_device_bytes=(100, 100))
    with pytest.raises(ValueError, match="QN mask"):
        replace(profile, materialization_qn_mask_bytes=(3, 3))


def test_root_fallback_profiles_size_promoted_results_and_mixed_centers():
    krylov = krylov_root_fallback_memory_profile(
        vector_bytes=16,
        vector_count=4,
        dtype=np.dtype("float32"),
        coefficient=-0.25j,
        block_size=2,
        max_krylov_vectors=2,
    )
    davidson = davidson_root_fallback_memory_profile(
        packed_vector_bytes=16,
        packed_vector_count=4,
        full_center_bytes=32,
        mask_bytes=8,
        dtype=np.dtype("float32"),
        max_space=2,
    )

    assert krylov.result_dtype == "complex128"
    assert krylov.result_itemsize == 16
    assert krylov.result_vector_bytes == 64
    assert krylov.device_peak_bytes >= krylov.vector_bytes + krylov.result_vector_bytes
    assert davidson.result_dtype == "float64"
    assert davidson.result_itemsize == 8
    assert davidson.packed_result_bytes == 32
    assert davidson.full_result_center_bytes == 64
    assert davidson.concurrent_complete_center_bytes == 2 * 32 + 3 * 64


def test_wave7_all_true_qn_identity_retains_device_and_exact_host_masks():
    dense = shard_axis((8,), 0, 2)
    no_mask = CenterVectorMap(dense, dense)
    all_true = CenterVectorMap(dense, dense, np.ones(8, dtype=bool))

    dense_profile = center_materialization_memory_profile(no_mask, np.float64)
    qn_profile = center_materialization_memory_profile(all_true, np.float64)

    assert no_mask.qn_mask_identity is None
    assert all_true.qn_mask_identity is not None
    assert all_true.qn_mask_identity.global_shape == (8,)
    assert all_true.qn_mask_identity.rank_counts == (4, 4)
    assert len(all_true.qn_mask_identity.digest) == 64
    assert dense_profile.qn_mask_identity is None
    assert qn_profile.qn_mask_identity == all_true.qn_mask_identity
    assert qn_profile.qn_mask_present is True
    assert qn_profile.setup_qn_mask_bytes == 8
    assert qn_profile.extract_qn_mask_bytes == (4, 4)
    assert qn_profile.persistent_qn_mask_host_bytes == 16
    assert qn_profile.setup_qn_digest_host_bytes == 8


def test_wave7_krylov_profile_binds_exact_recurrence_coefficient_identity():
    sharding = shard_axis((4,), 0, 2)
    real = build_krylov_memory_profile(
        sharding,
        np.float32,
        coefficient=0.25,
        block_size=2,
        max_krylov_vectors=2,
    )
    zero_imaginary = build_krylov_memory_profile(
        sharding,
        np.float32,
        coefficient=0.25 + 0.0j,
        block_size=2,
        max_krylov_vectors=2,
    )
    complex_profile = build_krylov_memory_profile(
        sharding,
        np.float32,
        coefficient=0.25 + 0.5j,
        block_size=2,
        max_krylov_vectors=2,
    )
    different_real = build_krylov_memory_profile(
        sharding,
        np.float32,
        coefficient=0.5,
        block_size=2,
        max_krylov_vectors=2,
    )

    assert real.coefficient_identity == zero_imaginary.coefficient_identity
    assert real.coefficient_identity != complex_profile.coefficient_identity
    assert real.coefficient_identity != different_real.coefficient_identity
    assert solver_module.canonical_solver_memory_profile(real) == real
    assert (
        solver_module.canonical_solver_memory_profile(complex_profile)
        == complex_profile
    )
    solver_module.validate_krylov_profile_coefficient(real, 0.25 + 0.0j)
    with pytest.raises(ValueError, match="coefficient"):
        solver_module.validate_krylov_profile_coefficient(real, 0.5)

    fallback = krylov_root_fallback_memory_profile(
        vector_bytes=16,
        vector_count=4,
        dtype=np.float32,
        coefficient=0.25 + 0.5j,
        block_size=2,
        max_krylov_vectors=2,
    )
    assert fallback.coefficient_identity == complex_profile.coefficient_identity


def test_wave7_sharded_krylov_rejects_coefficient_mismatched_residency_plan():
    sharding = shard_axis((2,), 0, 1)
    collective = SingleProcessCollective()
    backend = create_backend(
        "numpy",
        config=BackendConfig(device="cpu", execution_policy="execution_ir"),
    )

    class Operator:
        context = DistributedContext(0, 0, 1, 1)
        solver_input_sharding = sharding
        solver_output_sharding = sharding
        solver_dtype = np.dtype("float64")
        residency_plan = type(
            "Plan",
            (),
            {
                "krylov_coefficient_identity": (
                    solver_module.canonical_krylov_coefficient(0.5)
                )
            },
        )()

        def __init__(self):
            self.backend = backend
            self.collective = collective

        def solver_preflight(self):
            return None

        def __call__(self, local_vector):
            return local_vector.copy()

    vector = DistributedTensor(sharding, 0, np.asarray([1.0, 0.0]))

    with pytest.raises(ValueError, match="coefficient.*residency plan"):
        run_sharded_krylov(
            Operator(),
            vector,
            0.25,
            collective=collective,
            config={"block_size": 2, "max_krylov_vectors": 2},
        )


def test_wave7_root_davidson_counts_caller_owned_packed_device_arguments():
    profile = davidson_root_fallback_memory_profile(
        packed_vector_bytes=16,
        packed_vector_count=4,
        full_center_bytes=32,
        mask_bytes=8,
        dtype=np.float32,
        max_space=2,
    )

    assert profile.original_packed_guess_bytes == 16
    assert profile.original_packed_diagonal_bytes == 16
    assert profile.device_peak_bytes >= (
        profile.original_packed_guess_bytes
        + profile.original_packed_diagonal_bytes
        + profile.packed_vector_bytes
        + profile.packed_result_bytes
    )


def test_wave7_qn_and_coefficient_payload_helpers_are_canonical():
    dense = shard_axis((4,), 0, 1)
    vector_map = CenterVectorMap(dense, dense, np.ones(4, dtype=bool))
    coefficient = solver_module.canonical_krylov_coefficient(1.0 + 0.0j)

    assert center_module.qn_mask_identity_payload(vector_map.qn_mask_identity) == {
        "global_shape": [4],
        "rank_counts": [4],
        "digest": vector_map.qn_mask_identity.digest,
    }
    assert solver_module.krylov_coefficient_payload(coefficient)["kind"] == "real"
