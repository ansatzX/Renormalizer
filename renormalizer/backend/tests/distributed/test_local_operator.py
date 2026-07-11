import os
import weakref
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from renormalizer.backend._distributed.collectives import SingleProcessCollective
from renormalizer.backend._distributed.context import DistributedContext
from renormalizer.backend._distributed.local_operator import (
    DistributedLocalOperator,
    run_root_fallback,
)
from renormalizer.backend._distributed.mesh import DeviceMesh
from renormalizer.backend._distributed.planner import (
    DistributedBlockPlan,
    _execution_memory_profile,
    _placement_hash,
    plan_distributed_execution,
)
from renormalizer.backend._distributed.providers import (
    DeviceResidentProvider,
    OperandRequest,
)
from renormalizer.backend._execution.model import (
    BatchedMatmulStep,
    BufferRef,
    ExecutionBindings,
    ExecutionPlan,
    GroupedMatmulStep,
    MatmulStep,
    ReductionStep,
    TensorSpec,
    TransformStep,
    _plan_hash,
)
from renormalizer.backend._execution.planner import lower_einsum_path
from renormalizer.backend._execution.workspace import workspace_bytes_for_steps
from renormalizer.backend.config import BackendConfig, DistributedExecutionConfig
from renormalizer.backend.factory import create_backend


class _InjectedBaseException(BaseException):
    pass


_KNOWN_COUNTER_KEYS = (
    "broadcast_calls",
    "allreduce_calls",
    "allgather_calls",
    "execution_calls",
)


def _context(rank=0, size=1):
    return DistributedContext(
        rank=rank,
        local_rank=rank,
        world_size=size,
        local_world_size=size,
    )


def _numpy_backend():
    return create_backend(
        "numpy",
        config=BackendConfig(device="cpu", execution_policy="execution_ir"),
    )


def _all_step_refs(plan):
    refs = list(plan.inputs) + [plan.output]
    for step in plan.steps:
        operations = step.operations if isinstance(step, GroupedMatmulStep) else (step,)
        for operation in operations:
            if isinstance(operation, (MatmulStep, BatchedMatmulStep)):
                refs.extend((operation.left, operation.right, operation.output))
            elif isinstance(operation, (TransformStep, ReductionStep)):
                refs.extend((operation.input, operation.output))
    return refs


def _grouped_source_plan(*, pack_key_collision=False):
    third_key = "distributed_pack_0_input_0" if pack_key_collision else "input_2"
    inputs = (
        BufferRef("input_0", TensorSpec((8, 6), "float64", "C", ("a", "b"))),
        BufferRef("input_1", TensorSpec((6, 4), "float64", "C", ("b", "c"))),
        BufferRef(third_key, TensorSpec((3, 5), "float64", "C", ("d", "e"))),
        BufferRef("input_3", TensorSpec((5, 2), "float64", "C", ("e", "f"))),
    )
    first_output = BufferRef(
        "output", TensorSpec((8, 4), "float64", "C", ("a", "c"))
    )
    second_output = BufferRef(
        "other_output", TensorSpec((3, 2), "float64", "C", ("d", "f"))
    )
    grouped = GroupedMatmulStep(
        (
            MatmulStep(inputs[0], inputs[1], first_output, ("b",)),
            MatmulStep(inputs[2], inputs[3], second_output, ("e",)),
        )
    )
    steps = (grouped,)
    oe_path = ((0, 1), (0, 1), (0, 1))
    workspace = workspace_bytes_for_steps(steps, first_output.key)
    override_reason = "group independent contractions"
    return ExecutionPlan(
        operation="einsum",
        inputs=inputs,
        output=first_output,
        steps=steps,
        workspace_bytes=workspace,
        planner_source="specialized",
        oe_path=oe_path,
        override_reason=override_reason,
        plan_hash=_plan_hash(
            "einsum",
            inputs,
            first_output,
            steps,
            workspace,
            "specialized",
            oe_path,
            override_reason,
        ),
    )


def _multiple_grouped_steps_source_plan():
    def ref(key, shape, modes):
        return BufferRef(key, TensorSpec(shape, "float64", "C", modes))

    final_left = ref("input_0", (8, 3), ("r", "q"))
    variable = ref("input_1", (3,), ("q",))
    sibling_left = ref("input_2", (8, 3), ("r", "q"))
    sibling_right = ref("input_3", (3,), ("q",))
    first_bucket_inputs = (
        ref("input_4", (2, 3), ("a", "b")),
        ref("input_5", (3, 4), ("b", "c")),
        ref("input_6", (2, 3), ("d", "e")),
        ref("input_7", (3, 4), ("e", "f")),
    )
    second_bucket_inputs = (
        ref("input_8", (5, 2), ("g", "h")),
        ref("input_9", (2, 3), ("h", "i")),
        ref("input_10", (5, 2), ("j", "k")),
        ref("input_11", (2, 3), ("k", "l")),
    )
    inputs = (
        final_left,
        variable,
        sibling_left,
        sibling_right,
        *first_bucket_inputs,
        *second_bucket_inputs,
    )
    first_outputs = (
        ref("first_a0", (2, 4), ("a", "c")),
        ref("first_a1", (2, 4), ("d", "f")),
        ref("first_b0", (5, 3), ("g", "i")),
        ref("first_b1", (5, 3), ("j", "l")),
    )
    first_group = GroupedMatmulStep(
        (
            MatmulStep(inputs[4], inputs[5], first_outputs[0], ("b",)),
            MatmulStep(inputs[8], inputs[9], first_outputs[2], ("h",)),
            MatmulStep(inputs[6], inputs[7], first_outputs[1], ("e",)),
            MatmulStep(inputs[10], inputs[11], first_outputs[3], ("k",)),
        )
    )
    output = ref("output", (8,), ("r",))
    sibling_output = ref("sibling_output", (8,), ("r",))
    final_group = GroupedMatmulStep(
        (
            MatmulStep(final_left, variable, output, ("q",)),
            MatmulStep(
                sibling_left,
                sibling_right,
                sibling_output,
                ("q",),
            ),
        )
    )
    steps = (first_group, final_group)
    workspace = workspace_bytes_for_steps(steps, output.key)
    oe_path = tuple((0, 1) for _ in range(len(inputs) - 1))
    override_reason = "multiple grouped execution buckets"
    plan = ExecutionPlan(
        operation="einsum",
        inputs=inputs,
        output=output,
        steps=steps,
        workspace_bytes=workspace,
        planner_source="specialized",
        oe_path=oe_path,
        override_reason=override_reason,
        plan_hash=_plan_hash(
            "einsum",
            inputs,
            output,
            steps,
            workspace,
            "specialized",
            oe_path,
            override_reason,
        ),
    )
    arrays = {
        input_ref.key: (
            np.arange(
                int(np.prod(input_ref.spec.shape)), dtype=np.float64
            ).reshape(input_ref.spec.shape)
            + index
            + 1
        )
        / (index + 3)
        for index, input_ref in enumerate(inputs)
    }
    return plan, arrays


def _adversarial_grouped_liveness_plan():
    def ref(key, shape, modes):
        return BufferRef(key, TensorSpec(shape, "float64", "C", modes))

    inputs = (
        ref("input_0", (1, 100), ("a", "k")),
        ref("input_1", (100, 1), ("k", "b")),
        ref("input_2", (1, 100), ("c", "l")),
        ref("input_3", (100, 1), ("l", "d")),
        ref("input_4", (20, 1), ("e", "m")),
        ref("input_5", (1, 20), ("m", "f")),
    )
    first_output = ref("first_output", (1, 1), ("a", "b"))
    second_output = ref("second_output", (1, 1), ("c", "d"))
    output = ref("output", (20, 20), ("e", "f"))
    grouped = GroupedMatmulStep(
        (
            MatmulStep(inputs[0], inputs[1], first_output, ("k",)),
            MatmulStep(inputs[4], inputs[5], output, ("m",)),
            MatmulStep(inputs[2], inputs[3], second_output, ("l",)),
        )
    )
    steps = (grouped,)
    workspace = workspace_bytes_for_steps(steps, output.key)
    oe_path = tuple((0, 1) for _ in range(len(inputs) - 1))
    override_reason = "adversarial grouped bucket retention"
    plan = ExecutionPlan(
        operation="einsum",
        inputs=inputs,
        output=output,
        steps=steps,
        workspace_bytes=workspace,
        planner_source="specialized",
        oe_path=oe_path,
        override_reason=override_reason,
        plan_hash=_plan_hash(
            "einsum",
            inputs,
            output,
            steps,
            workspace,
            "specialized",
            oe_path,
            override_reason,
        ),
    )
    arrays = {
        input_ref.key: (
            np.arange(
                int(np.prod(input_ref.spec.shape)), dtype=np.float64
            ).reshape(input_ref.spec.shape)
            + index
            + 1
        )
        / (index + 7)
        for index, input_ref in enumerate(inputs)
    }
    return plan, arrays


def test_planner_rewrites_every_affected_spec_and_preserves_oe_path_order():
    plan = lower_einsum_path(
        "ab,bc,cd->ad",
        ((10, 7), (7, 7), (7, 5)),
        dtype="float64",
        optimize=((0, 1), (0, 1)),
    )

    distributed = plan_distributed_execution(
        plan, variable_key="input_2", world_size=4
    )
    block = distributed.block_plan(rank=3, source_rank=3)

    assert distributed.output_mode == "a"
    assert distributed.input_mode == "c"
    assert block.execution_plan.oe_path == plan.oe_path
    assert tuple(ref.key for ref in block.execution_plan.inputs) == tuple(
        ref.key for ref in plan.inputs
    )
    assert block.execution_plan.operation == plan.operation
    for ref in _all_step_refs(block.execution_plan):
        dimensions = dict(zip(ref.spec.modes, ref.spec.shape))
        if "a" in dimensions:
            assert dimensions["a"] == 2
        if "c" in dimensions:
            assert dimensions["c"] == 1


def test_planner_packs_strided_resident_slices_before_matmul():
    plan = lower_einsum_path("ab,b->a", ((10, 7), (7,)), dtype="float64")

    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=4
    )
    block = distributed.block_plan(rank=0, source_rank=3)

    matrix_input = block.execution_plan.inputs[0]
    assert matrix_input.spec.shape == (3, 1)
    assert matrix_input.spec.layout == "strided"
    assert isinstance(block.execution_plan.steps[0], TransformStep)
    assert block.execution_plan.steps[0].input == matrix_input
    assert block.execution_plan.steps[0].copy is True
    assert block.execution_plan.steps[0].output.spec.layout == "C"
    assert block.execution_plan.workspace_bytes > 0


def test_distributed_hash_is_canonical_and_includes_slice_offsets():
    plan = lower_einsum_path("ab,b->a", ((8, 8), (8,)), dtype="float64")

    first = plan_distributed_execution(plan, variable_key="input_1", world_size=4)
    second = plan_distributed_execution(plan, variable_key="input_1", world_size=4)

    assert first.placement_hash == second.placement_hash
    assert first.block_plan(0, 0).execution_plan.plan_hash == first.block_plan(
        1, 1
    ).execution_plan.plan_hash
    assert first.block_plan(0, 0).operand_slices != first.block_plan(
        1, 1
    ).operand_slices
    assert len(first.placement_hash) == 64
    int(first.placement_hash, 16)


def test_placement_hash_rejects_equal_shape_blocks_with_shifted_offsets():
    plan = lower_einsum_path("ab,b->a", ((8, 8), (8,)), dtype="float64")
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=4
    )
    original = distributed.block_plan(0, 0)
    shifted_slices = dict(original.operand_slices)
    shifted_slices["input_0"] = (slice(2, 4), slice(2, 4))
    shifted = DistributedBlockPlan(
        rank=original.rank,
        source_rank=original.source_rank,
        execution_plan=original.execution_plan,
        operand_slices=shifted_slices,
    )
    blocks = (shifted,) + distributed.block_plans[1:]

    shifted_hash = _placement_hash(
        distributed.execution_plan,
        distributed.variable_key,
        distributed.output_mode,
        distributed.input_mode,
        distributed.output_sharding,
        distributed.input_sharding,
        blocks,
        distributed.memory_estimates,
    )

    assert shifted.execution_plan.plan_hash == original.execution_plan.plan_hash
    assert shifted_hash != distributed.placement_hash
    with pytest.raises(ValueError, match="placement_hash does not match"):
        replace(distributed, block_plans=blocks)


def test_memory_estimates_count_unique_resident_and_reusable_state():
    plan = lower_einsum_path("ab,b->a", ((10, 7), (7,)), dtype="float64")

    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=4
    )
    rank_zero = distributed.memory_estimates[0]
    rank_three = distributed.memory_estimates[3]

    assert rank_zero.resident_static_bytes == 10 * 7 * 8
    assert rank_zero.local_input_bytes == 2 * 8
    assert rank_zero.receive_buffer_bytes == 2 * 8
    assert rank_zero.output_accumulator_bytes == 3 * 8
    assert rank_zero.output_contribution_bytes == 3 * 8
    assert rank_zero.control_status_bytes == np.dtype(np.int32).itemsize
    assert rank_zero.host_control_status_bytes == np.dtype(np.int32).itemsize
    assert rank_zero.preflight_hash_device_bytes == 3 * 4 * 8
    assert rank_zero.preflight_hash_host_bytes == 2 * 4 * 8
    assert rank_zero.preflight_status_device_bytes == 2 * 4
    assert rank_zero.preflight_status_host_bytes == 4
    assert rank_zero.workspace_bytes == max(
        distributed.block_plan(0, source).execution_plan.workspace_bytes
        for source in range(4)
    )
    resident_device_bytes = (
        rank_zero.resident_static_bytes + rank_zero.local_input_bytes
    )
    preflight_device_peak = resident_device_bytes + max(
        rank_zero.preflight_hash_device_bytes,
        rank_zero.preflight_status_device_bytes,
    )
    execution_device_peak = (
        resident_device_bytes
        + rank_zero.receive_buffer_bytes
        + rank_zero.output_accumulator_bytes
        + rank_zero.control_status_bytes
        + max(
            rank_zero.output_contribution_bytes + rank_zero.workspace_bytes,
            rank_zero.preflight_status_device_bytes,
        )
    )
    assert rank_zero.device_bytes == max(
        preflight_device_peak, execution_device_peak
    )
    assert rank_zero.host_bytes == max(
        rank_zero.preflight_hash_host_bytes,
        rank_zero.preflight_status_host_bytes,
        rank_zero.host_control_status_bytes
        + rank_zero.preflight_status_host_bytes,
    )
    assert rank_three.local_input_bytes == 8
    assert distributed.device_memory_estimates == tuple(
        estimate.device_bytes for estimate in distributed.memory_estimates
    )
    assert distributed.host_memory_estimates == (64, 64, 64, 64)


def test_preflight_control_peak_dominates_tiny_execution_state():
    plan = lower_einsum_path("ab,b->a", ((2, 2), (2,)), dtype="float64")
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=2
    )

    for estimate in distributed.memory_estimates:
        assert estimate.resident_static_bytes == 32
        assert estimate.local_input_bytes == 8
        assert estimate.preflight_hash_device_bytes == 96
        assert estimate.preflight_hash_host_bytes == 64
        assert estimate.preflight_status_device_bytes == 8
        assert estimate.preflight_status_host_bytes == 4
        assert estimate.device_bytes == 136
        assert estimate.host_bytes == 64


def test_grouped_memory_peak_excludes_separately_counted_final_output():
    source_plan = _grouped_source_plan()
    distributed = plan_distributed_execution(
        source_plan, variable_key="input_1", world_size=2
    )

    for rank, estimate in enumerate(distributed.memory_estimates):
        exact_capacities = [
            distributed.block_plan(rank, source).execution_plan.workspace_bytes
            for source in range(2)
        ]
        returned_output_bytes = [
            distributed.block_plan(rank, source).execution_plan.output.spec.nbytes
            for source in range(2)
        ]
        assert estimate.workspace_bytes == max(
            capacity - output_bytes
            for capacity, output_bytes in zip(
                exact_capacities, returned_output_bytes
            )
        )
        assert estimate.output_contribution_bytes == returned_output_bytes[0]
        assert all(
            capacity > estimate.workspace_bytes for capacity in exact_capacities
        )


def test_non_grouped_memory_peak_keeps_exact_block_workspace():
    source_plan = lower_einsum_path("ab,b->a", ((10, 7), (7,)))
    distributed = plan_distributed_execution(
        source_plan, variable_key="input_1", world_size=4
    )

    for rank, estimate in enumerate(distributed.memory_estimates):
        assert estimate.workspace_bytes == max(
            distributed.block_plan(rank, source).execution_plan.workspace_bytes
            for source in range(4)
        )


def test_multiple_grouped_buckets_and_steps_use_actual_live_execution_peak():
    source_plan, _ = _multiple_grouped_steps_source_plan()
    distributed = plan_distributed_execution(
        source_plan, variable_key="input_1", world_size=2
    )

    for rank, estimate in enumerate(distributed.memory_estimates):
        first_source = distributed.block_plan(rank, 0).execution_plan
        second_source = distributed.block_plan(rank, 1).execution_plan
        assert sum(
            isinstance(step, GroupedMatmulStep) for step in first_source.steps
        ) == 2
        assert first_source.workspace_bytes == 1264
        assert second_source.workspace_bytes == 1120
        assert estimate.output_contribution_bytes == 64
        assert estimate.workspace_bytes == 656
        assert (
            estimate.output_contribution_bytes + estimate.workspace_bytes
        ) == 720


def test_grouped_peak_tracks_python_pack_bindings_across_scalar_bucket():
    plan, arrays = _adversarial_grouped_liveness_plan()
    profile = _execution_memory_profile(plan)
    backend = _numpy_backend()
    pack_references = []
    result_pack_reference = []
    observed_live_peaks = []
    original_stack = backend.stack
    original_batched_matmul = backend.batched_matmul
    original_matmul = backend.matmul

    def recording_stack(values, *, axis=0):
        result = original_stack(values, axis=axis)
        pack_references.append(weakref.ref(result))
        return result

    def recording_batched_matmul(left, right, *, stream=None, workspace=None):
        result = original_batched_matmul(
            left, right, stream=stream, workspace=workspace
        )
        result_pack_reference.append(weakref.ref(result))
        return result

    def recording_matmul(left, right, *, stream=None, workspace=None):
        result = original_matmul(
            left, right, stream=stream, workspace=workspace
        )
        retained = [
            reference()
            for reference in (*pack_references, *result_pack_reference)
        ]
        assert len(retained) == 3
        assert all(array is not None for array in retained)
        observed_live_peaks.append(
            sum(array.nbytes for array in retained) + result.nbytes
        )
        return result

    backend.stack = recording_stack
    backend.batched_matmul = recording_batched_matmul
    backend.matmul = recording_matmul

    result = backend.execute_plan(
        plan,
        ExecutionBindings(arrays),
        workspace=plan.workspace_bytes,
    )

    assert plan.workspace_bytes == 6416
    assert profile.contribution_retained_bytes == 3200
    assert profile.peak_bytes == 6416
    assert observed_live_peaks == [6416]
    np.testing.assert_allclose(result, arrays["input_4"] @ arrays["input_5"])


def test_memory_estimates_are_trusted_placement_hash_metadata():
    plan = lower_einsum_path("ab,b->a", ((8, 8), (8,)), dtype="float64")
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=4
    )
    original = distributed.memory_estimates[0]
    changed = replace(
        original,
        resident_static_bytes=original.resident_static_bytes + 8,
        device_bytes=original.device_bytes + 8,
    )

    with pytest.raises(ValueError, match="memory estimates do not match"):
        replace(
            distributed,
            memory_estimates=(changed,) + distributed.memory_estimates[1:],
        )


def test_host_status_staging_changes_the_canonical_placement_hash():
    plan = lower_einsum_path("ab,b->a", ((8, 8), (8,)), dtype="float64")
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=4
    )
    original = distributed.memory_estimates[0]
    changed = replace(
        original,
        host_control_status_bytes=original.host_control_status_bytes + 4,
    )
    changed_hash = _placement_hash(
        distributed.execution_plan,
        distributed.variable_key,
        distributed.output_mode,
        distributed.input_mode,
        distributed.output_sharding,
        distributed.input_sharding,
        distributed.block_plans,
        (changed,) + distributed.memory_estimates[1:],
    )

    assert changed_hash != distributed.placement_hash


def test_preflight_control_components_change_the_canonical_placement_hash():
    plan = lower_einsum_path("ab,b->a", ((2, 2), (2,)), dtype="float64")
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=2
    )
    original = distributed.memory_estimates[0]
    changed = replace(
        original,
        preflight_hash_device_bytes=original.preflight_hash_device_bytes + 8,
        device_bytes=original.device_bytes + 8,
    )
    changed_hash = _placement_hash(
        distributed.execution_plan,
        distributed.variable_key,
        distributed.output_mode,
        distributed.input_mode,
        distributed.output_sharding,
        distributed.input_sharding,
        distributed.block_plans,
        (changed,) + distributed.memory_estimates[1:],
    )

    assert changed_hash != distributed.placement_hash


def test_distributed_plan_rejects_a_tampered_placement_hash():
    plan = lower_einsum_path("ab,b->a", ((8, 8), (8,)), dtype="float64")
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=4
    )

    with pytest.raises(ValueError, match="placement_hash does not match"):
        replace(distributed, placement_hash="0" * 64)


def test_planner_rewrites_every_grouped_matmul_operation():
    source_plan = _grouped_source_plan()

    distributed = plan_distributed_execution(
        source_plan, variable_key="input_1", world_size=2
    )
    block = distributed.block_plan(rank=0, source_rank=1)
    rewritten_group = block.execution_plan.steps[-1]

    assert isinstance(rewritten_group, GroupedMatmulStep)
    assert len(rewritten_group.operations) == 2
    assert rewritten_group.operations[0].output.spec.shape == (4, 4)
    assert rewritten_group.operations[1].output.spec.shape == (3, 2)
    assert block.execution_plan.oe_path == source_plan.oe_path


def test_generated_pack_keys_avoid_every_existing_plan_key():
    source_plan = _grouped_source_plan(pack_key_collision=True)

    distributed = plan_distributed_execution(
        source_plan, variable_key="input_1", world_size=2
    )
    block = distributed.block_plan(rank=0, source_rank=1)
    existing_keys = {ref.key for ref in _all_step_refs(source_plan)}
    pack_steps = [
        step
        for step in block.execution_plan.steps
        if isinstance(step, TransformStep) and step.copy
    ]

    assert pack_steps
    assert all(step.output.key not in existing_keys for step in pack_steps)


@pytest.mark.parametrize(
    "plan, variable_key, world_size, message",
    [
        (
            lower_einsum_path("ab,b->a", ((2, 4), (4,))),
            "input_1",
            3,
            "no legal distributed axis pair",
        ),
        (
            lower_einsum_path("ab,bc->a", ((4, 5), (5, 3))),
            "input_0",
            2,
            "no legal distributed axis pair",
        ),
        (
            lower_einsum_path("ab,b->a", ((4, 4), (4,))),
            "missing",
            2,
            "unknown variable input",
        ),
    ],
)
def test_planner_rejects_unsupported_placement_before_execution(
    plan, variable_key, world_size, message
):
    with pytest.raises(ValueError, match=message):
        plan_distributed_execution(
            plan, variable_key=variable_key, world_size=world_size
        )


def test_device_resident_provider_returns_views_without_transfer_or_full_variable():
    backend = _numpy_backend()
    plan = lower_einsum_path("ab,b->a", ((10, 7), (7,)), dtype="float64")
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=4
    )
    block = distributed.block_plan(rank=0, source_rank=3)
    matrix = np.arange(70.0).reshape(10, 7)
    variable = np.arange(1.0)
    provider = DeviceResidentProvider()
    request = OperandRequest(
        execution_plan=block.execution_plan,
        source_bindings=ExecutionBindings({"input_0": matrix}),
        distributed_plan=distributed,
        context=_context(rank=0, size=4),
        source_rank=3,
        broadcast_variable=variable,
        operand_slices=block.operand_slices,
    )

    with provider.acquire(request) as lease:
        matrix_view = lease.bindings.arrays["input_0"]
        assert matrix_view.shape == (3, 1)
        assert np.shares_memory(matrix_view, matrix)
        assert lease.bindings.arrays["input_1"] is variable
        lease.mark_dirty("input_0", matrix_view)

    with pytest.raises(RuntimeError, match="lease is closed"):
        lease.mark_dirty("input_0", matrix_view)


class _ReplayBroadcastCollective:
    def __init__(self, rank, source_shards):
        self.rank = rank
        self.size = len(source_shards)
        self.source_shards = source_shards
        self.broadcast_calls = []
        self.allreduce_calls = 0
        self.allreduce_trace = []
        self.inplace_allreduce_calls = 0
        self.allgather_calls = 0

    def barrier(self):
        return None

    def broadcast(self, array, *, root):
        expected = self.source_shards[root]
        assert array.shape == expected.shape
        array[...] = expected
        self.broadcast_calls.append((root, tuple(array.shape)))
        return array

    def allreduce(self, array, *, op="sum"):
        self.allreduce_calls += 1
        self.allreduce_trace.append((op, np.dtype(array.dtype).name, tuple(array.shape)))
        return array.copy()

    def allreduce_inplace(self, array, *, op="sum"):
        self.inplace_allreduce_calls += 1
        result = self.allreduce(array, op=op)
        array[...] = result
        return array

    def reduce_scatter(self, array, *, axis, op="sum"):
        raise AssertionError("local H-v must not reduce-scatter")

    def allgather(self, array, *, axis):
        self.allgather_calls += 1
        raise AssertionError("local H-v must not allgather")

    def close(self):
        return None


def _make_numpy_operator(distributed, matrix, rank, source_shards, **options):
    collective = _ReplayBroadcastCollective(rank, source_shards)
    counters = {}
    operator = DistributedLocalOperator(
        plan=distributed,
        provider=DeviceResidentProvider(),
        collective=collective,
        counters=counters,
        backend=_numpy_backend(),
        context=_context(rank=rank, size=len(source_shards)),
        source_bindings=ExecutionBindings({"input_0": matrix}),
        **options,
    )
    return operator, collective, counters


def test_local_operator_matches_full_hv_with_ordered_uneven_broadcasts():
    matrix = np.arange(70.0, dtype=np.float64).reshape(10, 7) / 17
    vector = np.arange(7.0, dtype=np.float64) / 5
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=4
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    local_results = []

    for rank in range(4):
        operator, collective, counters = _make_numpy_operator(
            distributed, matrix, rank, source_shards
        )
        local_results.append(operator(source_shards[rank]))
        assert collective.broadcast_calls == [
            (0, (2,)),
            (1, (2,)),
            (2, (2,)),
            (3, (1,)),
        ]
        assert collective.allgather_calls == 0
        assert counters["broadcast_calls"] == 4
        assert counters["allgather_calls"] == 0
        assert counters["execution_calls"] == 4
        assert counters["allreduce_calls"] == 10
        assert collective.allreduce_calls == 10
        assert collective.inplace_allreduce_calls == 4

    np.testing.assert_allclose(np.concatenate(local_results), matrix @ vector)


def test_operator_reuses_preallocated_receive_output_and_status_storage():
    matrix = np.arange(70.0, dtype=np.float64).reshape(10, 7)
    vector = np.arange(7.0, dtype=np.float64)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=4
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    operator, collective, counters = _make_numpy_operator(
        distributed, matrix, 3, source_shards
    )

    first = operator(source_shards[3])
    receive_storage = operator._receive_storage
    output_storage = operator._output_accumulator
    execution_status = operator._execution_status
    host_execution_status = operator._host_execution_status
    second = operator(source_shards[3])

    assert receive_storage.shape == (2,)
    assert output_storage.shape == (2,)
    assert operator._receive_storage is receive_storage
    assert operator._output_accumulator is output_storage
    assert operator._execution_status is execution_status
    assert operator._host_execution_status is host_execution_status
    assert host_execution_status.shape == (1,)
    assert host_execution_status.dtype == np.dtype(np.int32)
    assert first is output_storage
    assert second is output_storage
    assert counters["allreduce_calls"] == 16
    assert collective.allreduce_calls == 16
    assert collective.inplace_allreduce_calls == 8


def test_numpy_execution_status_copies_into_one_persistent_host_array(monkeypatch):
    matrix = np.arange(30.0).reshape(6, 5)
    vector = np.arange(5.0)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=2
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    operator, collective, _ = _make_numpy_operator(
        distributed, matrix, 0, source_shards
    )
    destinations = []
    original_copyto = np.copyto

    def recording_copyto(destination, source, *args, **kwargs):
        destinations.append(destination)
        return original_copyto(destination, source, *args, **kwargs)

    monkeypatch.setattr(np, "copyto", recording_copyto)

    operator(source_shards[0])
    host_status = operator._host_execution_status
    operator(source_shards[0])

    assert len(destinations) == 4
    assert all(destination is host_status for destination in destinations)
    assert collective.inplace_allreduce_calls == 4


def test_cupy_execution_status_uses_asnumpy_out_without_new_host_array(monkeypatch):
    cupy = pytest.importorskip("cupy")
    previous_device = int(cupy.cuda.runtime.getDevice())
    try:
        backend = create_backend(
            "cupy",
            config=BackendConfig(
                device="cuda:0", execution_policy="execution_ir"
            ),
        )
        matrix_host = np.arange(12.0).reshape(4, 3)
        vector_host = np.arange(3.0)
        source_plan = lower_einsum_path(
            "ab,b->a", (matrix_host.shape, vector_host.shape)
        )
        distributed = plan_distributed_execution(
            source_plan, variable_key="input_1", world_size=1
        )
        operator = DistributedLocalOperator(
            plan=distributed,
            provider=DeviceResidentProvider(),
            collective=SingleProcessCollective(),
            counters={},
            backend=backend,
            context=_context(),
            source_bindings=ExecutionBindings(
                {"input_0": cupy.asarray(matrix_host)}
            ),
        )
        local_vector = cupy.asarray(vector_host)
        original_asnumpy = cupy.asnumpy
        status_destinations = []

        def guarded_asnumpy(value, *args, **kwargs):
            if operator._execution_status is value:
                destination = kwargs.get("out")
                assert destination is operator._host_execution_status
                assert kwargs.get("blocking") is True
                result = original_asnumpy(value, *args, **kwargs)
                assert result is destination
                status_destinations.append(destination)
                return result
            return original_asnumpy(value, *args, **kwargs)

        monkeypatch.setattr(cupy, "asnumpy", guarded_asnumpy)

        operator(local_vector)
        host_status = operator._host_execution_status
        operator(local_vector)

        assert status_destinations == [host_status, host_status]
    finally:
        cupy.cuda.Device(previous_device).use()


def test_nonleading_uneven_input_axis_uses_c_contiguous_receive_prefixes():
    matrix = np.arange(70.0, dtype=np.float64).reshape(10, 7) / 17
    variable = np.arange(21.0, dtype=np.float64).reshape(3, 7) / 19
    plan = lower_einsum_path("ab,cb->ac", (matrix.shape, variable.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=4
    )
    source_shards = tuple(
        np.ascontiguousarray(variable[local_slice])
        for local_slice in distributed.input_sharding.local_slices
    )
    results = []

    for rank in range(4):
        operator, collective, _ = _make_numpy_operator(
            distributed, matrix, rank, source_shards
        )
        results.append(operator(source_shards[rank]).copy())
        assert operator._receive_storage.shape == (6,)
        assert collective.broadcast_calls == [
            (0, (3, 2)),
            (1, (3, 2)),
            (2, (3, 2)),
            (3, (3, 1)),
        ]

    np.testing.assert_allclose(
        np.concatenate(results), matrix @ variable.T, rtol=1e-12, atol=1e-12
    )


def test_operator_passes_each_block_exact_workspace_capacity():
    matrix = np.arange(70.0, dtype=np.float64).reshape(10, 7)
    vector = np.arange(7.0, dtype=np.float64)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=4
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    backend = _numpy_backend()
    calls = []
    original_execute = backend.execute_plan

    def recording_execute(plan, bindings, *, stream=None, workspace=None):
        calls.append(workspace)
        return original_execute(
            plan, bindings, stream=stream, workspace=workspace
        )

    backend.execute_plan = recording_execute
    collective = _ReplayBroadcastCollective(0, source_shards)
    operator = DistributedLocalOperator(
        plan=distributed,
        provider=DeviceResidentProvider(),
        collective=collective,
        counters={},
        backend=backend,
        context=_context(rank=0, size=4),
        source_bindings=ExecutionBindings({"input_0": matrix}),
    )

    operator(source_shards[0])

    assert calls == [
        distributed.block_plan(0, source).execution_plan.workspace_bytes
        for source in range(4)
    ]


def test_execute_baseexception_is_synchronized_before_next_source_broadcast():
    matrix = np.arange(30.0).reshape(6, 5)
    vector = np.arange(5.0)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=2
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    backend = _numpy_backend()

    def fail_execute(*args, **kwargs):
        raise _InjectedBaseException("injected execute failure")

    backend.execute_plan = fail_execute
    collective = _ReplayBroadcastCollective(0, source_shards)
    counters = {}
    operator = DistributedLocalOperator(
        plan=distributed,
        provider=DeviceResidentProvider(),
        collective=collective,
        counters=counters,
        backend=backend,
        context=_context(rank=0, size=2),
        source_bindings=ExecutionBindings({"input_0": matrix}),
    )

    with pytest.raises(RuntimeError, match="distributed block execution failed") as caught:
        operator(source_shards[0])

    assert isinstance(caught.value.__cause__, _InjectedBaseException)
    assert collective.broadcast_calls == [(0, (3,))]
    assert counters["allreduce_calls"] == 7
    assert counters["allgather_calls"] == 0


def test_accumulation_baseexception_is_synchronized_before_next_broadcast():
    matrix = np.arange(30.0).reshape(6, 5)
    vector = np.arange(5.0)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=2
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    operator, collective, counters = _make_numpy_operator(
        distributed, matrix, 0, source_shards
    )

    def fail_accumulation(output, contribution):
        raise _InjectedBaseException("injected accumulation failure")

    operator._accumulate_contribution = fail_accumulation

    with pytest.raises(RuntimeError, match="distributed block execution failed") as caught:
        operator(source_shards[0])

    assert isinstance(caught.value.__cause__, _InjectedBaseException)
    assert collective.broadcast_calls == [(0, (3,))]
    assert counters["allreduce_calls"] == 7
    assert operator._active_contribution is None


def test_contribution_reference_is_released_before_next_execute_call():
    matrix = np.arange(70.0).reshape(10, 7)
    vector = np.arange(7.0)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=4
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    backend = _numpy_backend()
    original_execute = backend.execute_plan
    observed_active_references = []

    def recording_execute(plan, bindings, *, stream=None, workspace=None):
        observed_active_references.append(operator._active_contribution)
        return original_execute(
            plan, bindings, stream=stream, workspace=workspace
        )

    backend.execute_plan = recording_execute
    collective = _ReplayBroadcastCollective(0, source_shards)
    operator = DistributedLocalOperator(
        plan=distributed,
        provider=DeviceResidentProvider(),
        collective=collective,
        counters={},
        backend=backend,
        context=_context(rank=0, size=4),
        source_bindings=ExecutionBindings({"input_0": matrix}),
    )

    operator(source_shards[0])

    assert observed_active_references == [None, None, None, None]
    assert operator._active_contribution is None


def test_transitive_multistep_rewrite_executes_numerically():
    left = np.arange(70.0).reshape(10, 7) / 19
    middle = (np.arange(49.0).reshape(7, 7) + 1) / 23
    variable = np.arange(35.0).reshape(7, 5) / 29
    source_plan = lower_einsum_path(
        "ab,bc,cd->ad",
        (left.shape, middle.shape, variable.shape),
        optimize=((0, 1), (0, 1)),
    )
    distributed = plan_distributed_execution(
        source_plan, variable_key="input_2", world_size=4
    )
    source_shards = tuple(
        variable[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    results = []

    for rank in range(4):
        collective = _ReplayBroadcastCollective(rank, source_shards)
        operator = DistributedLocalOperator(
            plan=distributed,
            provider=DeviceResidentProvider(),
            collective=collective,
            counters={},
            backend=_numpy_backend(),
            context=_context(rank=rank, size=4),
            source_bindings=ExecutionBindings(
                {"input_0": left, "input_1": middle}
            ),
        )
        results.append(operator(source_shards[rank]).copy())

    np.testing.assert_allclose(
        np.concatenate(results), left @ middle @ variable, rtol=1e-12, atol=1e-12
    )


def test_grouped_rewrite_executes_numerically():
    source_plan = _grouped_source_plan()
    matrix = np.arange(48.0).reshape(8, 6) / 13
    variable = np.arange(24.0).reshape(6, 4) / 17
    unrelated_left = np.arange(15.0).reshape(3, 5) / 7
    unrelated_right = np.arange(10.0).reshape(5, 2) / 11
    distributed = plan_distributed_execution(
        source_plan, variable_key="input_1", world_size=2
    )
    source_shards = tuple(
        variable[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    results = []

    for rank in range(2):
        collective = _ReplayBroadcastCollective(rank, source_shards)
        operator = DistributedLocalOperator(
            plan=distributed,
            provider=DeviceResidentProvider(),
            collective=collective,
            counters={},
            backend=_numpy_backend(),
            context=_context(rank=rank, size=2),
            source_bindings=ExecutionBindings(
                {
                    "input_0": matrix,
                    "input_2": unrelated_left,
                    "input_3": unrelated_right,
                }
            ),
        )
        results.append(operator(source_shards[rank]).copy())

    np.testing.assert_allclose(
        np.concatenate(results), matrix @ variable, rtol=1e-12, atol=1e-12
    )


def test_multiple_grouped_buckets_and_steps_execute_numerically():
    source_plan, arrays = _multiple_grouped_steps_source_plan()
    distributed = plan_distributed_execution(
        source_plan, variable_key="input_1", world_size=2
    )
    variable = arrays.pop("input_1")
    source_shards = tuple(
        variable[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    results = []

    for rank in range(2):
        collective = _ReplayBroadcastCollective(rank, source_shards)
        operator = DistributedLocalOperator(
            plan=distributed,
            provider=DeviceResidentProvider(),
            collective=collective,
            counters={},
            backend=_numpy_backend(),
            context=_context(rank=rank, size=2),
            source_bindings=ExecutionBindings(arrays),
        )
        results.append(operator(source_shards[rank]).copy())

    np.testing.assert_allclose(
        np.concatenate(results), arrays["input_0"] @ variable
    )


def test_local_vector_preflight_fails_before_first_data_path_collective():
    matrix = np.arange(30.0).reshape(6, 5)
    vector = np.arange(5.0)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=2
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    operator, collective, _ = _make_numpy_operator(
        distributed, matrix, 0, source_shards
    )

    with pytest.raises(ValueError, match="local vector preflight failed: shape"):
        operator(np.empty((1,), dtype=np.float64))

    assert collective.broadcast_calls == []
    assert collective.allgather_calls == 0
    assert collective.allreduce_calls == 4


def test_plan_hash_disagreement_fails_before_first_data_path_collective():
    class MismatchedPlanCollective(_ReplayBroadcastCollective):
        def allreduce(self, array, *, op="sum"):
            result = super().allreduce(array, op=op)
            if array.dtype == np.dtype("uint64") and op == "max":
                result[0] ^= np.uint64(1)
            return result

    matrix = np.arange(30.0).reshape(6, 5)
    vector = np.arange(5.0)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=2
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    collective = MismatchedPlanCollective(0, source_shards)
    operator = DistributedLocalOperator(
        plan=distributed,
        provider=DeviceResidentProvider(),
        collective=collective,
        counters={},
        backend=_numpy_backend(),
        context=_context(rank=0, size=2),
        source_bindings=ExecutionBindings({"input_0": matrix}),
    )

    with pytest.raises(ValueError, match="distributed placement hash disagreement"):
        operator(source_shards[0])

    assert collective.broadcast_calls == []
    assert collective.allgather_calls == 0
    assert collective.allreduce_calls == 3


def test_rank_local_resident_setup_failure_is_synchronized_before_broadcast():
    matrix = np.arange(30.0).reshape(6, 5)
    vector = np.arange(5.0)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=2
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    collective = _ReplayBroadcastCollective(0, source_shards)

    operator = DistributedLocalOperator(
        plan=distributed,
        provider=DeviceResidentProvider(),
        collective=collective,
        counters={},
        backend=_numpy_backend(),
        context=_context(rank=0, size=2),
        source_bindings=ExecutionBindings({"input_0": matrix[:, :-1]}),
    )

    with pytest.raises(
        ValueError, match="distributed setup preflight failed"
    ) as caught:
        operator(source_shards[0])

    assert caught.value.__cause__ is operator._setup_error
    assert collective.broadcast_calls == []
    assert collective.allreduce_calls == 1


def test_rank_local_setup_baseexception_is_synchronized_before_broadcast(
    monkeypatch,
):
    matrix = np.arange(30.0).reshape(6, 5)
    vector = np.arange(5.0)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=2
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    collective = _ReplayBroadcastCollective(0, source_shards)

    def fail_setup(self):
        raise _InjectedBaseException("injected rank-local setup failure")

    monkeypatch.setattr(DistributedLocalOperator, "_validate_setup", fail_setup)

    operator = DistributedLocalOperator(
        plan=distributed,
        provider=DeviceResidentProvider(),
        collective=collective,
        counters={},
        backend=_numpy_backend(),
        context=_context(rank=0, size=2),
        source_bindings=ExecutionBindings({"input_0": matrix}),
    )

    with pytest.raises(
        ValueError, match="distributed setup preflight failed"
    ) as caught:
        operator(source_shards[0])

    assert isinstance(operator._setup_error, _InjectedBaseException)
    assert caught.value.__cause__ is operator._setup_error
    assert collective.broadcast_calls == []
    assert collective.allreduce_calls == 1


def test_collective_size_mismatch_uses_context_schedule_before_placement():
    matrix = np.arange(30.0).reshape(6, 5)
    vector = np.arange(5.0)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=2
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    collective = _ReplayBroadcastCollective(0, source_shards)
    collective.size = 1
    operator = DistributedLocalOperator(
        plan=distributed,
        provider=DeviceResidentProvider(),
        collective=collective,
        counters={},
        backend=_numpy_backend(),
        context=_context(rank=0, size=2),
        source_bindings=ExecutionBindings({"input_0": matrix}),
    )

    with pytest.raises(
        ValueError, match="distributed setup preflight failed"
    ) as caught:
        operator(source_shards[0])

    assert caught.value.__cause__ is operator._setup_error
    assert collective.allreduce_trace == [("max", "int32", (1,))]
    assert collective.broadcast_calls == []
    assert operator._plan_preflight_complete is False


def test_noncallable_broadcast_uses_context_schedule_before_placement():
    matrix = np.arange(30.0).reshape(6, 5)
    vector = np.arange(5.0)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=2
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    collective = _ReplayBroadcastCollective(0, source_shards)
    collective.broadcast = None
    operator = DistributedLocalOperator(
        plan=distributed,
        provider=DeviceResidentProvider(),
        collective=collective,
        counters={},
        backend=_numpy_backend(),
        context=_context(rank=0, size=2),
        source_bindings=ExecutionBindings({"input_0": matrix}),
    )

    with pytest.raises(
        ValueError, match="distributed setup preflight failed"
    ) as caught:
        operator(source_shards[0])

    assert isinstance(operator._setup_error, TypeError)
    assert caught.value.__cause__ is operator._setup_error
    assert collective.allreduce_trace == [("max", "int32", (1,))]
    assert collective.broadcast_calls == []
    assert operator._plan_preflight_complete is False


def test_noncallable_allreduce_remains_an_early_structural_error():
    matrix = np.arange(30.0).reshape(6, 5)
    vector = np.arange(5.0)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=2
    )
    collective = _ReplayBroadcastCollective(0, (vector[:3], vector[3:]))
    collective.allreduce = None
    collective.size = 0

    with pytest.raises(TypeError, match="collective must provide a usable allreduce"):
        DistributedLocalOperator(
            plan=object(),
            provider=DeviceResidentProvider(),
            collective=collective,
            counters={},
            backend=_numpy_backend(),
            context=object(),
            source_bindings=ExecutionBindings({"input_0": matrix}),
        )

    assert collective.broadcast_calls == []


def _assert_invalid_constructor_field_uses_bootstrap(field):
    matrix = np.arange(30.0).reshape(6, 5)
    vector = np.arange(5.0)
    source_plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        source_plan, variable_key="input_1", world_size=2
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    collective = _ReplayBroadcastCollective(0, source_shards)
    invalid_counters = []
    options = {
        "plan": distributed,
        "context": _context(rank=0, size=2),
        "counters": {},
    }
    if field == "plan":
        options["plan"] = object()
    elif field == "context":
        options["context"] = object()
    elif field == "counters":
        options["counters"] = invalid_counters
    else:
        raise AssertionError("unknown invalid field")

    operator = DistributedLocalOperator(
        provider=DeviceResidentProvider(),
        collective=collective,
        backend=_numpy_backend(),
        source_bindings=ExecutionBindings({"input_0": matrix}),
        **options,
    )

    with pytest.raises(
        ValueError, match="distributed setup preflight failed"
    ) as caught:
        operator(source_shards[0])

    assert isinstance(operator._setup_error, TypeError)
    assert caught.value.__cause__ is operator._setup_error
    assert operator._bootstrap_world_size == 2
    assert operator._counters["allreduce_calls"] == 1
    assert collective.allreduce_trace == [("max", "int32", (1,))]
    assert collective.broadcast_calls == []
    assert operator._plan_preflight_complete is False
    if field == "counters":
        assert operator.counters is invalid_counters
        assert invalid_counters == []
        assert operator._counters is not invalid_counters
    else:
        assert operator._counters is not operator.counters
        assert operator.counters["allreduce_calls"] == 1


def test_invalid_plan_uses_context_bootstrap_for_setup_status():
    _assert_invalid_constructor_field_uses_bootstrap("plan")


def test_invalid_context_uses_plan_bootstrap_for_setup_status():
    _assert_invalid_constructor_field_uses_bootstrap("context")


def test_invalid_counters_use_private_sink_for_setup_status():
    _assert_invalid_constructor_field_uses_bootstrap("counters")


def test_invalid_plan_and_context_use_communicator_bootstrap():
    matrix = np.arange(30.0).reshape(6, 5)
    vector = np.arange(5.0)
    collective = _ReplayBroadcastCollective(0, (vector[:3], vector[3:]))
    operator = DistributedLocalOperator(
        plan=object(),
        provider=DeviceResidentProvider(),
        collective=collective,
        counters={},
        backend=_numpy_backend(),
        context=object(),
        source_bindings=ExecutionBindings({"input_0": matrix}),
    )

    with pytest.raises(
        ValueError, match="distributed setup preflight failed"
    ) as caught:
        operator(vector[:3])

    assert isinstance(caught.value.__cause__, TypeError)
    assert operator._bootstrap_world_size == 2
    assert operator._counters["allreduce_calls"] == 1
    assert collective.broadcast_calls == []


def test_missing_bootstrap_schedule_is_an_early_structural_error():
    vector = np.arange(5.0)
    collective = _ReplayBroadcastCollective(0, (vector,))
    collective.size = 0

    with pytest.raises(TypeError, match="bootstrap schedule size"):
        DistributedLocalOperator(
            plan=object(),
            provider=DeviceResidentProvider(),
            collective=collective,
            counters={},
            backend=_numpy_backend(),
            context=object(),
            source_bindings=ExecutionBindings({}),
        )

    assert collective.allreduce_calls == 0


@pytest.mark.parametrize("counter_key", _KNOWN_COUNTER_KEYS)
def test_malformed_known_counter_is_captured_before_placement(counter_key):
    matrix = np.arange(30.0).reshape(6, 5)
    vector = np.arange(5.0)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=2
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    collective = _ReplayBroadcastCollective(0, source_shards)
    metadata = object()
    counters = {"metadata": metadata, counter_key: "invalid"}
    original_counters = dict(counters)
    operator = DistributedLocalOperator(
        plan=distributed,
        provider=DeviceResidentProvider(),
        collective=collective,
        counters=counters,
        backend=_numpy_backend(),
        context=_context(rank=0, size=2),
        source_bindings=ExecutionBindings({"input_0": matrix}),
    )

    with pytest.raises(
        ValueError, match="distributed setup preflight failed"
    ) as caught:
        operator(source_shards[0])

    assert isinstance(caught.value.__cause__, TypeError)
    assert counters == original_counters
    assert counters["metadata"] is metadata
    assert operator._counters["allreduce_calls"] == 1
    assert all(
        type(operator._counters[key]) is int and operator._counters[key] >= 0
        for key in _KNOWN_COUNTER_KEYS
    )
    assert collective.allreduce_trace == [("max", "int32", (1,))]
    assert collective.broadcast_calls == []
    assert operator._plan_preflight_complete is False


@pytest.mark.parametrize(
    "value, cause_type",
    ((True, TypeError), (np.int64(1), TypeError), (-1, ValueError)),
)
def test_known_counter_requires_nonnegative_nonbool_python_int(value, cause_type):
    matrix = np.arange(30.0).reshape(6, 5)
    vector = np.arange(5.0)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=2
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    collective = _ReplayBroadcastCollective(0, source_shards)
    counters = {"allreduce_calls": value}
    operator = DistributedLocalOperator(
        plan=distributed,
        provider=DeviceResidentProvider(),
        collective=collective,
        counters=counters,
        backend=_numpy_backend(),
        context=_context(rank=0, size=2),
        source_bindings=ExecutionBindings({"input_0": matrix}),
    )

    with pytest.raises(
        ValueError, match="distributed setup preflight failed"
    ) as caught:
        operator(source_shards[0])

    assert isinstance(caught.value.__cause__, cause_type)
    assert counters["allreduce_calls"] is value
    assert operator._counters["allreduce_calls"] == 1
    assert collective.allreduce_calls == 1
    assert collective.broadcast_calls == []


def test_private_counter_sink_seeds_valid_values_and_preserves_metadata():
    matrix = np.arange(30.0).reshape(6, 5)
    vector = np.arange(5.0)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=2
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    metadata = object()
    counters = {
        "broadcast_calls": 5,
        "allreduce_calls": 7,
        "allgather_calls": 11,
        "execution_calls": 13,
        "metadata": metadata,
    }
    operator = DistributedLocalOperator(
        plan=distributed,
        provider=DeviceResidentProvider(),
        collective=_ReplayBroadcastCollective(0, source_shards),
        counters=counters,
        backend=_numpy_backend(),
        context=_context(rank=0, size=2),
        source_bindings=ExecutionBindings({"input_0": matrix}),
    )

    assert operator._counters is not counters
    assert operator._counters == {
        key: counters[key] for key in _KNOWN_COUNTER_KEYS
    }

    operator(source_shards[0])

    assert counters == {
        "broadcast_calls": 7,
        "allreduce_calls": 15,
        "allgather_calls": 11,
        "execution_calls": 15,
        "metadata": metadata,
    }
    assert operator._counters == {
        key: counters[key] for key in _KNOWN_COUNTER_KEYS
    }


@pytest.mark.parametrize("counter_key", _KNOWN_COUNTER_KEYS)
def test_external_known_counter_mutation_never_becomes_arithmetic(counter_key):
    matrix = np.arange(30.0).reshape(6, 5)
    vector = np.arange(5.0)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=2
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    metadata = object()
    counters = {"metadata": metadata}
    collective = _ReplayBroadcastCollective(0, source_shards)
    operator = DistributedLocalOperator(
        plan=distributed,
        provider=DeviceResidentProvider(),
        collective=collective,
        counters=counters,
        backend=_numpy_backend(),
        context=_context(rank=0, size=2),
        source_bindings=ExecutionBindings({"input_0": matrix}),
    )

    operator(source_shards[0])
    counters[counter_key] = "externally corrupted"

    result = operator(source_shards[0])

    expected_counters = {
        "broadcast_calls": 4,
        "allreduce_calls": 12,
        "allgather_calls": 0,
        "execution_calls": 4,
    }
    assert operator._counters == expected_counters
    assert all(counters[key] == value for key, value in expected_counters.items())
    assert counters["metadata"] is metadata
    assert collective.inplace_allreduce_calls == 4
    assert collective.allreduce_calls == 12
    assert len(collective.broadcast_calls) == 4
    np.testing.assert_allclose(
        result,
        (matrix @ vector)[distributed.output_sharding.local_slices[0]],
    )


def test_provider_acquire_failure_is_synchronized_before_broadcast():
    class FailingAcquireProvider(DeviceResidentProvider):
        @contextmanager
        def acquire(self, request):
            raise RuntimeError("injected acquire failure")
            yield

    matrix = np.arange(30.0).reshape(6, 5)
    vector = np.arange(5.0)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=2
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    collective = _ReplayBroadcastCollective(0, source_shards)
    operator = DistributedLocalOperator(
        plan=distributed,
        provider=FailingAcquireProvider(),
        collective=collective,
        counters={},
        backend=_numpy_backend(),
        context=_context(rank=0, size=2),
        source_bindings=ExecutionBindings({"input_0": matrix}),
    )

    with pytest.raises(ValueError, match="distributed resource preflight failed"):
        operator(source_shards[0])

    assert collective.broadcast_calls == []
    assert collective.allreduce_calls == 6


class _RecordingAcquireProvider(DeviceResidentProvider):
    def __init__(self, *, fail_on_source=None):
        self.fail_on_source = fail_on_source
        self.events = []

    def acquire(self, request):
        provider = self
        base_context = super().acquire(request)
        source_rank = request.source_rank

        class RecordingContext:
            def __enter__(self):
                provider.events.append(("enter", source_rank))
                if source_rank == provider.fail_on_source:
                    raise _InjectedBaseException(
                        "injected provider acquisition failure"
                    )
                return base_context.__enter__()

            def __exit__(self, exc_type, exc_value, traceback):
                provider.events.append(("exit", source_rank))
                return base_context.__exit__(exc_type, exc_value, traceback)

        return RecordingContext()


def _operator_for_cleanup_test(provider, collective):
    matrix = np.arange(30.0).reshape(6, 5)
    vector = np.arange(5.0)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=2
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    operator = DistributedLocalOperator(
        plan=distributed,
        provider=provider,
        collective=collective,
        counters={},
        backend=_numpy_backend(),
        context=_context(rank=0, size=2),
        source_bindings=ExecutionBindings({"input_0": matrix}),
    )
    return operator, source_shards[0]


def test_provider_baseexception_closes_already_entered_contexts():
    vector = np.arange(5.0)
    collective = _ReplayBroadcastCollective(0, (vector[:3], vector[3:]))
    provider = _RecordingAcquireProvider(fail_on_source=1)
    operator, local_vector = _operator_for_cleanup_test(provider, collective)

    with pytest.raises(ValueError, match="distributed resource preflight failed"):
        operator(local_vector)

    assert provider.events == [("enter", 0), ("enter", 1), ("exit", 0)]
    assert collective.broadcast_calls == []
    assert operator._host_execution_status is None


def test_resource_status_allreduce_failure_closes_all_provider_contexts():
    class FailingStatusCollective(_ReplayBroadcastCollective):
        def allreduce(self, array, *, op="sum"):
            if self.allreduce_calls == 5:
                raise RuntimeError("injected resource status failure")
            return super().allreduce(array, op=op)

    vector = np.arange(5.0)
    collective = FailingStatusCollective(0, (vector[:3], vector[3:]))
    provider = _RecordingAcquireProvider()
    operator, local_vector = _operator_for_cleanup_test(provider, collective)

    with pytest.raises(RuntimeError, match="injected resource status failure"):
        operator(local_vector)

    assert provider.events == [
        ("enter", 0),
        ("enter", 1),
        ("exit", 1),
        ("exit", 0),
    ]
    assert collective.broadcast_calls == []


def test_resource_status_conversion_failure_closes_all_provider_contexts():
    class InvalidStatusCollective(_ReplayBroadcastCollective):
        def allreduce(self, array, *, op="sum"):
            result = super().allreduce(array, op=op)
            if self.allreduce_calls == 6:
                return object()
            return result

    vector = np.arange(5.0)
    collective = InvalidStatusCollective(0, (vector[:3], vector[3:]))
    provider = _RecordingAcquireProvider()
    operator, local_vector = _operator_for_cleanup_test(provider, collective)

    with pytest.raises(TypeError):
        operator(local_vector)

    assert provider.events == [
        ("enter", 0),
        ("enter", 1),
        ("exit", 1),
        ("exit", 0),
    ]
    assert collective.broadcast_calls == []


def test_capacity_failure_is_synchronized_before_allocation_or_broadcast():
    matrix = np.arange(30.0).reshape(6, 5)
    vector = np.arange(5.0)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=2
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    estimate = distributed.memory_estimates[0]
    operator, collective, counters = _make_numpy_operator(
        distributed,
        matrix,
        0,
        source_shards,
        device_memory_budget_bytes=estimate.device_bytes - 1,
        host_memory_budget_bytes=1,
    )

    with pytest.raises(ValueError, match="distributed capacity preflight failed"):
        operator(source_shards[0])

    assert collective.broadcast_calls == []
    assert collective.allreduce_calls == 5
    assert counters["allreduce_calls"] == 5
    assert operator._receive_storage is None
    assert operator._output_accumulator is None


@pytest.mark.parametrize(
    "device_budget, host_budget",
    ((135, 64), (136, 63)),
)
def test_preflight_control_peak_is_enforced_by_tight_budgets(
    device_budget, host_budget
):
    matrix = np.arange(4.0).reshape(2, 2)
    vector = np.arange(2.0)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=2
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    operator, collective, counters = _make_numpy_operator(
        distributed,
        matrix,
        0,
        source_shards,
        device_memory_budget_bytes=device_budget,
        host_memory_budget_bytes=host_budget,
    )

    with pytest.raises(ValueError, match="distributed capacity preflight failed"):
        operator(source_shards[0])

    assert collective.broadcast_calls == []
    assert collective.allreduce_calls == 5
    assert counters["allreduce_calls"] == 5
    assert operator._receive_storage is None


def test_rank_local_allocation_failure_is_synchronized_before_broadcast():
    matrix = np.arange(30.0).reshape(6, 5)
    vector = np.arange(5.0)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=2
    )
    source_shards = tuple(
        vector[local_slice]
        for local_slice in distributed.input_sharding.local_slices
    )
    backend = _numpy_backend()

    def fail_allocation(*args, **kwargs):
        raise MemoryError("injected allocation failure")

    backend.empty = fail_allocation
    collective = _ReplayBroadcastCollective(0, source_shards)
    counters = {}
    operator = DistributedLocalOperator(
        plan=distributed,
        provider=DeviceResidentProvider(),
        collective=collective,
        counters=counters,
        backend=backend,
        context=_context(rank=0, size=2),
        source_bindings=ExecutionBindings({"input_0": matrix}),
    )

    with pytest.raises(ValueError, match="distributed resource preflight failed"):
        operator(source_shards[0])

    assert collective.broadcast_calls == []
    assert collective.allreduce_calls == 6
    assert counters["allreduce_calls"] == 6


def test_operator_rejects_a_complete_resident_variable_replica():
    matrix = np.arange(30.0).reshape(6, 5)
    vector = np.arange(5.0)
    plan = lower_einsum_path("ab,b->a", (matrix.shape, vector.shape))
    distributed = plan_distributed_execution(
        plan, variable_key="input_1", world_size=2
    )
    collective = _ReplayBroadcastCollective(0, (vector[:3], vector[3:]))

    operator = DistributedLocalOperator(
        plan=distributed,
        provider=DeviceResidentProvider(),
        collective=collective,
        counters={},
        backend=_numpy_backend(),
        context=_context(rank=0, size=2),
        source_bindings=ExecutionBindings(
            {"input_0": matrix, "input_1": vector}
        ),
    )

    with pytest.raises(
        ValueError, match="distributed setup preflight failed"
    ) as caught:
        operator(vector[:3])

    assert caught.value.__cause__ is operator._setup_error
    assert collective.broadcast_calls == []
    assert collective.allreduce_calls == 1


def test_distributed_config_accepts_only_device_resident_stage4_policy():
    context = _context()
    mesh = DeviceMesh(shape=(1,), axis_names=("rank",), rank=0)
    collective = SingleProcessCollective()
    provider = DeviceResidentProvider()

    config = DistributedExecutionConfig(
        context=context,
        mesh=mesh,
        collective=collective,
        provider=provider,
        device_memory_budget_bytes=1024,
        host_memory_budget_bytes=2048,
        prefetch_depth=2,
    )

    assert config.residency_policy == "device_resident"
    with pytest.raises(ValueError, match="only residency_policy='device_resident'"):
        DistributedExecutionConfig(
            context=context,
            mesh=mesh,
            collective=collective,
            provider=provider,
            residency_policy="active_working_set",
        )
    with pytest.raises(FrozenInstanceError):
        config.prefetch_depth = 3


def test_root_fallback_calls_operation_only_on_root_for_object_collective():
    calls = []

    result = run_root_fallback(
        lambda: calls.append("root") or {"answer": 42},
        SingleProcessCollective(),
    )

    assert result == {"answer": 42}
    assert calls == ["root"]


def test_root_fallback_synchronizes_root_failure_before_result_broadcast():
    class RecordingCollective(SingleProcessCollective):
        def __init__(self):
            self.broadcast_calls = 0

        def broadcast(self, array, *, root):
            self.broadcast_calls += 1
            return super().broadcast(array, root=root)

    collective = RecordingCollective()

    with pytest.raises(RuntimeError, match="root fallback operation failed"):
        run_root_fallback(
            lambda: (_ for _ in ()).throw(ValueError("root failed")),
            collective,
        )

    assert collective.broadcast_calls == 0


class _FallbackArrayCollective:
    rank = 0
    size = 2

    def __init__(self, *, mismatch_fingerprint=False, mismatch_presence=False):
        self.allreduce_calls = []
        self.broadcast_calls = 0
        self.mismatch_fingerprint = mismatch_fingerprint
        self.mismatch_presence = mismatch_presence

    def allreduce(self, array, *, op="sum"):
        self.allreduce_calls.append((op, np.dtype(array.dtype).name))
        result = array.copy()
        if (
            self.mismatch_presence
            and len(self.allreduce_calls) == 2
            and op == "max"
            and np.dtype(array.dtype) == np.dtype("int32")
        ):
            result[0] = 1
        if (
            self.mismatch_fingerprint
            and op == "max"
            and np.dtype(array.dtype) == np.dtype("uint64")
        ):
            result[0] ^= np.uint64(1)
        return result

    def broadcast(self, array, *, root):
        self.broadcast_calls += 1
        return array


def test_root_fallback_rejects_buffer_presence_disagreement_before_branching():
    collective = _FallbackArrayCollective(mismatch_presence=True)
    calls = []

    with pytest.raises(ValueError, match="receive-buffer presence disagreement"):
        run_root_fallback(
            lambda: calls.append("root") or {"answer": 42},
            collective,
        )

    assert calls == []
    assert collective.allreduce_calls == [("min", "int32"), ("max", "int32")]
    assert collective.broadcast_calls == 0


def test_root_fallback_synchronizes_receive_validation_before_root_execution():
    collective = _FallbackArrayCollective()
    calls = []

    with pytest.raises(ValueError, match="receive-buffer preflight failed"):
        run_root_fallback(
            lambda: calls.append("root") or np.ones(2),
            collective,
            receive_buffer=object(),
        )

    assert calls == []
    assert collective.allreduce_calls == [
        ("min", "int32"),
        ("max", "int32"),
        ("max", "int32"),
    ]
    assert collective.broadcast_calls == 0


def test_root_fallback_rejects_cross_rank_receive_fingerprint_before_root():
    collective = _FallbackArrayCollective(mismatch_fingerprint=True)
    calls = []

    with pytest.raises(ValueError, match="receive-buffer fingerprint disagreement"):
        run_root_fallback(
            lambda: calls.append("root") or np.ones(2),
            collective,
            receive_buffer=np.empty(2),
        )

    assert calls == []
    assert collective.allreduce_calls == [
        ("min", "int32"),
        ("max", "int32"),
        ("max", "int32"),
        ("min", "uint64"),
        ("max", "uint64"),
    ]
    assert collective.broadcast_calls == 0


def test_root_fallback_synchronizes_root_result_validation_before_broadcast():
    collective = _FallbackArrayCollective()

    with pytest.raises(RuntimeError, match="root fallback operation failed"):
        run_root_fallback(
            lambda: np.ones(3),
            collective,
            receive_buffer=np.empty(2),
        )

    assert collective.allreduce_calls == [
        ("min", "int32"),
        ("max", "int32"),
        ("max", "int32"),
        ("min", "uint64"),
        ("max", "uint64"),
        ("max", "int32"),
    ]
    assert collective.broadcast_calls == 0


def test_root_fallback_receive_buffer_success_uses_exact_control_schedule():
    collective = _FallbackArrayCollective()
    receive = np.empty(2)

    result = run_root_fallback(
        lambda: np.asarray([3.0, 5.0]),
        collective,
        receive_buffer=receive,
    )

    assert result is receive
    np.testing.assert_array_equal(result, [3.0, 5.0])
    assert collective.allreduce_calls == [
        ("min", "int32"),
        ("max", "int32"),
        ("max", "int32"),
        ("min", "uint64"),
        ("max", "uint64"),
        ("max", "int32"),
    ]
    assert collective.broadcast_calls == 1


def test_public_distributed_solver_facade_exports_generic_stage4_api():
    from renormalizer.backend.distributed_solver import (
        DeviceResidentProvider as FacadeProvider,
        DistributedLocalOperator as FacadeOperator,
        plan_distributed_execution as facade_plan,
    )

    assert FacadeProvider is DeviceResidentProvider
    assert FacadeOperator is DistributedLocalOperator
    assert facade_plan is plan_distributed_execution


@pytest.mark.parametrize("invalid_field", ("plan", "context", "counters"))
@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2",
    reason="requires the exact two-rank torchrun command",
)
def test_real_two_rank_invalid_constructor_field_uses_bootstrap(invalid_field):
    cupy = pytest.importorskip("cupy")
    from renormalizer.backend.distributed_runtime import (
        create_cupy_distributed_runtime,
    )

    runtime = create_cupy_distributed_runtime(expected_world_size=2)
    try:
        matrix_host = np.arange(35.0, dtype=np.float64).reshape(5, 7) / 11
        vector_host = np.arange(7.0, dtype=np.float64) / 13
        source_plan = lower_einsum_path(
            "ab,b->a", (matrix_host.shape, vector_host.shape)
        )
        distributed = plan_distributed_execution(
            source_plan, variable_key="input_1", world_size=2
        )
        rank = runtime.rank
        local_vector = cupy.asarray(
            vector_host[distributed.input_sharding.local_slices[rank]]
        )
        matrix = cupy.asarray(matrix_host)
        invalid_counters = []
        user_counters = (
            invalid_counters
            if rank == 0 and invalid_field == "counters"
            else {}
        )
        options = {
            "plan": (
                object()
                if rank == 0 and invalid_field == "plan"
                else distributed
            ),
            "context": (
                object()
                if rank == 0 and invalid_field == "context"
                else runtime.context
            ),
            "counters": user_counters,
        }
        operator = DistributedLocalOperator(
            provider=DeviceResidentProvider(),
            collective=runtime.collective,
            backend=runtime.backend,
            source_bindings=ExecutionBindings({"input_0": matrix}),
            **options,
        )

        with pytest.raises(
            ValueError, match="distributed setup preflight failed"
        ) as caught:
            operator(local_vector)

        if rank == 0:
            assert isinstance(operator._setup_error, TypeError)
            assert caught.value.__cause__ is operator._setup_error
        else:
            assert caught.value.__cause__ is None
        assert operator._bootstrap_world_size == 2
        assert operator._counters["allreduce_calls"] == 1
        assert operator._counters["broadcast_calls"] == 0
        assert operator._counters["allgather_calls"] == 0
        assert operator._plan_preflight_complete is False
        if rank == 0 and invalid_field == "counters":
            assert user_counters == []
            assert operator._counters is not user_counters
        else:
            assert operator._counters is not user_counters
            assert user_counters["allreduce_calls"] == 1
    finally:
        runtime.close()


@pytest.mark.parametrize("counter_key", _KNOWN_COUNTER_KEYS)
@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2",
    reason="requires the exact two-rank torchrun command",
)
def test_real_two_rank_malformed_known_counter_stops_in_setup(counter_key):
    cupy = pytest.importorskip("cupy")
    from renormalizer.backend.distributed_runtime import (
        create_cupy_distributed_runtime,
    )

    runtime = create_cupy_distributed_runtime(expected_world_size=2)
    try:
        matrix_host = np.arange(35.0, dtype=np.float64).reshape(5, 7) / 11
        vector_host = np.arange(7.0, dtype=np.float64) / 13
        source_plan = lower_einsum_path(
            "ab,b->a", (matrix_host.shape, vector_host.shape)
        )
        distributed = plan_distributed_execution(
            source_plan, variable_key="input_1", world_size=2
        )
        rank = runtime.rank
        local_vector = cupy.asarray(
            vector_host[distributed.input_sharding.local_slices[rank]]
        )
        matrix = cupy.asarray(matrix_host)
        counters = {"metadata": "preserved"}
        if rank == 0:
            counters[counter_key] = "invalid"
        original_counters = dict(counters)
        operator = DistributedLocalOperator(
            plan=distributed,
            provider=DeviceResidentProvider(),
            collective=runtime.collective,
            counters=counters,
            backend=runtime.backend,
            context=runtime.context,
            source_bindings=ExecutionBindings({"input_0": matrix}),
        )

        with pytest.raises(
            ValueError, match="distributed setup preflight failed"
        ) as caught:
            operator(local_vector)

        if rank == 0:
            assert isinstance(operator._setup_error, TypeError)
            assert caught.value.__cause__ is operator._setup_error
            assert counters == original_counters
        else:
            assert caught.value.__cause__ is None
            assert counters["metadata"] == "preserved"
            assert all(
                type(counters[key]) is int and counters[key] >= 0
                for key in _KNOWN_COUNTER_KEYS
            )
        assert operator._counters["allreduce_calls"] == 1
        assert all(
            type(operator._counters[key]) is int and operator._counters[key] >= 0
            for key in _KNOWN_COUNTER_KEYS
        )
        assert operator._counters["broadcast_calls"] == 0
        assert operator._counters["allgather_calls"] == 0
        assert operator._plan_preflight_complete is False
    finally:
        runtime.close()


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2",
    reason="requires the exact two-rank torchrun command",
)
def test_real_two_rank_cupy_local_hv_matches_full_result_without_allgather():
    cupy = pytest.importorskip("cupy")
    from renormalizer.backend.distributed_runtime import (
        create_cupy_distributed_runtime,
    )

    runtime = create_cupy_distributed_runtime(expected_world_size=2)
    try:
        matrix_host = np.arange(35.0, dtype=np.float64).reshape(5, 7) / 11
        vector_host = np.arange(7.0, dtype=np.float64) / 13
        source_plan = lower_einsum_path(
            "ab,b->a", (matrix_host.shape, vector_host.shape)
        )
        distributed = plan_distributed_execution(
            source_plan, variable_key="input_1", world_size=2
        )
        rank = runtime.rank
        input_slice = distributed.input_sharding.local_slices[rank]
        output_slice = distributed.output_sharding.local_slices[rank]
        matrix = cupy.asarray(matrix_host)
        local_vector = cupy.asarray(vector_host[input_slice])

        original_broadcast = runtime.collective.broadcast
        broadcast_counters = {}
        try:
            if rank == 0:
                runtime.collective.broadcast = None
            broadcast_operator = DistributedLocalOperator(
                plan=distributed,
                provider=DeviceResidentProvider(),
                collective=runtime.collective,
                counters=broadcast_counters,
                backend=runtime.backend,
                context=runtime.context,
                source_bindings=ExecutionBindings({"input_0": matrix}),
            )

            with pytest.raises(
                ValueError, match="distributed setup preflight failed"
            ) as broadcast_caught:
                broadcast_operator(local_vector)
        finally:
            runtime.collective.broadcast = original_broadcast

        if rank == 0:
            assert isinstance(broadcast_operator._setup_error, TypeError)
            assert (
                broadcast_caught.value.__cause__
                is broadcast_operator._setup_error
            )
        else:
            assert broadcast_caught.value.__cause__ is None
        assert broadcast_counters["allreduce_calls"] == 1
        assert broadcast_counters["broadcast_calls"] == 0
        assert broadcast_operator._plan_preflight_complete is False

        original_collective_size = runtime.collective.size
        mismatch_counters = {}
        try:
            if rank == 0:
                runtime.collective.size = 1
            mismatch_operator = DistributedLocalOperator(
                plan=distributed,
                provider=DeviceResidentProvider(),
                collective=runtime.collective,
                counters=mismatch_counters,
                backend=runtime.backend,
                context=runtime.context,
                source_bindings=ExecutionBindings({"input_0": matrix}),
            )

            with pytest.raises(
                ValueError, match="distributed setup preflight failed"
            ) as mismatch_caught:
                mismatch_operator(local_vector)
        finally:
            runtime.collective.size = original_collective_size

        if rank == 0:
            assert mismatch_caught.value.__cause__ is mismatch_operator._setup_error
        else:
            assert mismatch_caught.value.__cause__ is None
        assert mismatch_counters["allreduce_calls"] == 1
        assert mismatch_counters["broadcast_calls"] == 0
        assert mismatch_operator._plan_preflight_complete is False

        bad_counters = {}
        bad_matrix = matrix[:, :-1] if rank == 0 else matrix
        bad_operator = DistributedLocalOperator(
            plan=distributed,
            provider=DeviceResidentProvider(),
            collective=runtime.collective,
            counters=bad_counters,
            backend=runtime.backend,
            context=runtime.context,
            source_bindings=ExecutionBindings({"input_0": bad_matrix}),
        )

        with pytest.raises(
            ValueError, match="distributed setup preflight failed"
        ) as bad_caught:
            bad_operator(local_vector)

        if rank == 0:
            assert bad_caught.value.__cause__ is bad_operator._setup_error
        else:
            assert bad_caught.value.__cause__ is None
        assert bad_counters["allreduce_calls"] == 1
        assert bad_counters["broadcast_calls"] == 0

        counters = {}
        operator = DistributedLocalOperator(
            plan=distributed,
            provider=DeviceResidentProvider(),
            collective=runtime.collective,
            counters=counters,
            backend=runtime.backend,
            context=runtime.context,
            source_bindings=ExecutionBindings({"input_0": matrix}),
        )

        local_result = operator(local_vector)
        expected = (matrix_host @ vector_host)[output_slice]

        np.testing.assert_allclose(cupy.asnumpy(local_result), expected)
        assert counters["broadcast_calls"] == 2
        assert counters["allgather_calls"] == 0
        assert counters["allreduce_calls"] == 8

        original_execute = runtime.backend.execute_plan
        failure_attempts = 0

        def fail_rank_zero_first_source(*args, **kwargs):
            nonlocal failure_attempts
            failure_attempts += 1
            if rank == 0 and failure_attempts == 1:
                raise RuntimeError("injected rank-local execute failure")
            return original_execute(*args, **kwargs)

        runtime.backend.execute_plan = fail_rank_zero_first_source
        failure_counters = {}
        failure_operator = DistributedLocalOperator(
            plan=distributed,
            provider=DeviceResidentProvider(),
            collective=runtime.collective,
            counters=failure_counters,
            backend=runtime.backend,
            context=runtime.context,
            source_bindings=ExecutionBindings({"input_0": matrix}),
        )
        try:
            with pytest.raises(
                RuntimeError, match="distributed block execution failed"
            ) as caught:
                failure_operator(local_vector)
        finally:
            runtime.backend.execute_plan = original_execute

        if rank == 0:
            assert isinstance(caught.value.__cause__, RuntimeError)
        else:
            assert caught.value.__cause__ is None
        assert failure_counters["broadcast_calls"] == 1
        assert failure_counters["allreduce_calls"] == 7
        assert failure_counters["allgather_calls"] == 0

        fallback_calls = []
        presence_buffer = (
            cupy.empty((2,), dtype=cupy.float64) if rank == 0 else None
        )
        with pytest.raises(ValueError, match="receive-buffer presence disagreement"):
            run_root_fallback(
                lambda: fallback_calls.append("root") or cupy.ones(2),
                runtime.collective,
                receive_buffer=presence_buffer,
            )
        assert fallback_calls == []

        if rank == 0:
            with cupy.cuda.Device(1):
                invalid_buffer = cupy.empty((2,), dtype=cupy.float64)
        else:
            invalid_buffer = cupy.empty((2,), dtype=cupy.float64)
        with pytest.raises(ValueError, match="receive-buffer preflight failed"):
            run_root_fallback(
                lambda: fallback_calls.append("root") or cupy.ones(2),
                runtime.collective,
                receive_buffer=invalid_buffer,
            )
        assert fallback_calls == []

        mismatched_buffer = cupy.empty((rank + 2,), dtype=cupy.float64)
        with pytest.raises(
            ValueError, match="receive-buffer fingerprint disagreement"
        ):
            run_root_fallback(
                lambda: fallback_calls.append("root") or cupy.ones(2),
                runtime.collective,
                receive_buffer=mismatched_buffer,
            )
        assert fallback_calls == []

        fallback_buffer = cupy.empty((2,), dtype=cupy.float64)
        fallback = run_root_fallback(
            lambda: cupy.asarray([3.0, 5.0]),
            runtime.collective,
            receive_buffer=fallback_buffer,
        )
        np.testing.assert_array_equal(cupy.asnumpy(fallback), [3.0, 5.0])

        with pytest.raises(RuntimeError, match="root fallback operation failed"):
            run_root_fallback(
                lambda: cupy.ones(3),
                runtime.collective,
                receive_buffer=fallback_buffer,
            )

        with pytest.raises(RuntimeError, match="root fallback operation failed"):
            run_root_fallback(
                lambda: (_ for _ in ()).throw(ValueError("injected root failure")),
                runtime.collective,
                receive_buffer=fallback_buffer,
            )
    finally:
        runtime.close()
