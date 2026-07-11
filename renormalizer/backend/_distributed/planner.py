"""Deterministic rank/source rewrites for output-sharded contractions."""

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from renormalizer.backend._distributed.sharding import ShardingSpec, shard_axis
from renormalizer.backend._execution.model import (
    BatchedMatmulStep,
    BufferRef,
    ExecutionPlan,
    GroupedMatmulStep,
    MatmulStep,
    ReductionStep,
    TensorSpec,
    TransformStep,
    _canonical_layout,
    _contiguous_strides,
    _plan_hash,
    _transform_view_layout,
)
from renormalizer.backend._execution.workspace import workspace_bytes_for_steps


def _slice_payload(local_slice):
    return tuple((value.start, value.stop, value.step) for value in local_slice)


def _slice_layout(spec, local_slice, local_shape):
    if all(value == slice(None) for value in local_slice):
        return spec.layout
    if spec.layout == "strided":
        return "strided"
    strides = _contiguous_strides(spec.shape, spec.layout)
    return _canonical_layout(local_shape, strides)


def _mode_dimensions(plan):
    dimensions = {}
    for ref in plan.inputs + (plan.output,):
        for mode, dimension in zip(ref.spec.modes, ref.spec.shape):
            previous = dimensions.setdefault(mode, dimension)
            if previous != dimension:
                raise ValueError("mode dimension mismatch for {!r}".format(mode))
    return dimensions


@dataclass(frozen=True)
class DistributedBlockPlan:
    rank: int
    source_rank: int
    execution_plan: ExecutionPlan
    operand_slices: object

    def __post_init__(self):
        if type(self.rank) is not int or type(self.source_rank) is not int:
            raise TypeError("block ranks must be integers")
        if self.rank < 0 or self.source_rank < 0:
            raise ValueError("block ranks must be non-negative")
        if not isinstance(self.execution_plan, ExecutionPlan):
            raise TypeError("execution_plan must be an ExecutionPlan")
        try:
            copied = {
                key: tuple(local_slice)
                for key, local_slice in self.operand_slices.items()
            }
        except (AttributeError, TypeError) as error:
            raise TypeError("operand_slices must be a mapping of slices") from error
        expected_keys = {ref.key for ref in self.execution_plan.inputs}
        if set(copied) != expected_keys:
            raise ValueError("operand_slices must cover every block-plan input")
        for ref in self.execution_plan.inputs:
            local_slice = copied[ref.key]
            if len(local_slice) != len(ref.spec.shape) or any(
                not isinstance(value, slice) for value in local_slice
            ):
                raise ValueError("operand slice rank must match its block-plan input")
        object.__setattr__(self, "operand_slices", MappingProxyType(copied))


@dataclass(frozen=True)
class DistributedMemoryEstimate:
    resident_static_bytes: int
    local_input_bytes: int
    receive_buffer_bytes: int
    output_accumulator_bytes: int
    output_contribution_bytes: int
    control_status_bytes: int
    host_control_status_bytes: int
    preflight_hash_device_bytes: int
    preflight_hash_host_bytes: int
    preflight_status_device_bytes: int
    preflight_status_host_bytes: int
    workspace_bytes: int
    device_bytes: int
    host_bytes: int

    def __post_init__(self):
        names = (
            "resident_static_bytes",
            "local_input_bytes",
            "receive_buffer_bytes",
            "output_accumulator_bytes",
            "output_contribution_bytes",
            "control_status_bytes",
            "host_control_status_bytes",
            "preflight_hash_device_bytes",
            "preflight_hash_host_bytes",
            "preflight_status_device_bytes",
            "preflight_status_host_bytes",
            "workspace_bytes",
            "device_bytes",
            "host_bytes",
        )
        if any(type(getattr(self, name)) is not int for name in names):
            raise TypeError("memory estimate fields must be integers")
        if any(getattr(self, name) < 0 for name in names):
            raise ValueError("memory estimate fields must be non-negative")
        resident_device_bytes = (
            self.resident_static_bytes + self.local_input_bytes
        )
        preflight_device_peak = resident_device_bytes + max(
            self.preflight_hash_device_bytes,
            self.preflight_status_device_bytes,
        )
        execution_device_peak = (
            resident_device_bytes
            + self.receive_buffer_bytes
            + self.output_accumulator_bytes
            + self.control_status_bytes
            + max(
                self.output_contribution_bytes + self.workspace_bytes,
                self.preflight_status_device_bytes,
            )
        )
        expected_device = max(preflight_device_peak, execution_device_peak)
        if self.device_bytes != expected_device:
            raise ValueError("device_bytes must equal deterministic phase peak")
        expected_host = max(
            self.preflight_hash_host_bytes,
            self.preflight_status_host_bytes,
            self.host_control_status_bytes + self.preflight_status_host_bytes,
        )
        if self.host_bytes != expected_host:
            raise ValueError("host_bytes must equal deterministic components")


@dataclass(frozen=True)
class DistributedPlan:
    execution_plan: ExecutionPlan
    variable_key: str
    output_mode: str
    input_mode: str
    output_sharding: ShardingSpec
    input_sharding: ShardingSpec
    block_plans: tuple[DistributedBlockPlan, ...]
    memory_estimates: tuple[DistributedMemoryEstimate, ...]
    placement_hash: str

    def __post_init__(self):
        if not isinstance(self.execution_plan, ExecutionPlan):
            raise TypeError("execution_plan must be an ExecutionPlan")
        if not isinstance(self.output_sharding, ShardingSpec) or not isinstance(
            self.input_sharding, ShardingSpec
        ):
            raise TypeError("distributed shard metadata must use ShardingSpec")
        matching = tuple(
            ref for ref in self.execution_plan.inputs if ref.key == self.variable_key
        )
        if len(matching) != 1:
            raise ValueError("variable_key must identify exactly one plan input")
        variable_ref = matching[0]
        if self.output_mode not in self.execution_plan.output.spec.modes:
            raise ValueError("output_mode must occur in the execution output")
        if self.input_mode not in variable_ref.spec.modes:
            raise ValueError("input_mode must occur in the variable input")
        if self.output_mode in variable_ref.spec.modes:
            raise ValueError("output_mode must not occur in the variable input")
        if self.input_mode in self.execution_plan.output.spec.modes:
            raise ValueError("input_mode must not occur in the execution output")
        if self.output_sharding.global_shape != self.execution_plan.output.spec.shape:
            raise ValueError("output sharding shape does not match execution output")
        if self.input_sharding.global_shape != variable_ref.spec.shape:
            raise ValueError("input sharding shape does not match variable input")
        if (
            self.execution_plan.output.spec.modes[self.output_sharding.axis]
            != self.output_mode
            or variable_ref.spec.modes[self.input_sharding.axis] != self.input_mode
        ):
            raise ValueError("sharding axes do not match distributed modes")
        blocks = tuple(self.block_plans)
        size = self.output_sharding.parts
        if self.input_sharding.parts != size or len(blocks) != size * size:
            raise ValueError("distributed plan must contain every rank/source block")
        if any(not isinstance(block, DistributedBlockPlan) for block in blocks):
            raise TypeError("block_plans must contain DistributedBlockPlan metadata")
        expected = tuple(
            (rank, source) for rank in range(size) for source in range(size)
        )
        actual = tuple((block.rank, block.source_rank) for block in blocks)
        if actual != expected:
            raise ValueError("distributed block order must be rank-major and complete")
        memory_estimates = tuple(self.memory_estimates)
        if len(memory_estimates) != size or any(
            not isinstance(estimate, DistributedMemoryEstimate)
            for estimate in memory_estimates
        ):
            raise ValueError("memory_estimates must contain one estimate per rank")
        expected_estimates = _memory_estimates(
            self.execution_plan,
            self.variable_key,
            self.output_sharding,
            self.input_sharding,
            blocks,
        )
        if memory_estimates != expected_estimates:
            raise ValueError("memory estimates do not match distributed placement")
        if not isinstance(self.placement_hash, str) or len(self.placement_hash) != 64:
            raise ValueError("placement_hash must be a SHA-256 hexadecimal digest")
        try:
            int(self.placement_hash, 16)
        except ValueError as error:
            raise ValueError(
                "placement_hash must be a SHA-256 hexadecimal digest"
            ) from error
        expected_hash = _placement_hash(
            self.execution_plan,
            self.variable_key,
            self.output_mode,
            self.input_mode,
            self.output_sharding,
            self.input_sharding,
            blocks,
            memory_estimates,
        )
        if self.placement_hash != expected_hash:
            raise ValueError("placement_hash does not match canonical metadata")
        object.__setattr__(self, "block_plans", blocks)
        object.__setattr__(self, "memory_estimates", memory_estimates)

    @property
    def world_size(self):
        return self.output_sharding.parts

    @property
    def plan_hash(self):
        return self.placement_hash

    @property
    def device_memory_estimates(self):
        return tuple(estimate.device_bytes for estimate in self.memory_estimates)

    @property
    def host_memory_estimates(self):
        return tuple(estimate.host_bytes for estimate in self.memory_estimates)

    def block_plan(self, rank, source_rank):
        if type(rank) is not int or type(source_rank) is not int:
            raise TypeError("rank and source_rank must be integers")
        if rank < 0 or rank >= self.world_size:
            raise ValueError("rank is out of range for distributed plan")
        if source_rank < 0 or source_rank >= self.world_size:
            raise ValueError("source_rank is out of range for distributed plan")
        return self.block_plans[rank * self.world_size + source_rank]


def _new_spec(ref, dimensions, *, layout=None):
    shape = tuple(dimensions[mode] for mode in ref.spec.modes)
    return TensorSpec(
        shape,
        ref.spec.dtype,
        ref.spec.layout if layout is None else layout,
        ref.spec.modes,
    )


def _plan_buffer_keys(plan):
    keys = {ref.key for ref in plan.inputs}
    keys.add(plan.output.key)
    for step in plan.steps:
        operations = step.operations if isinstance(step, GroupedMatmulStep) else (step,)
        for operation in operations:
            if isinstance(operation, (MatmulStep, BatchedMatmulStep)):
                keys.update(
                    (operation.left.key, operation.right.key, operation.output.key)
                )
            else:
                keys.update((operation.input.key, operation.output.key))
    return keys


def _rewrite_block(
    plan,
    variable_key,
    output_mode,
    input_mode,
    output_slice,
    input_slice,
    rank,
    source_rank,
):
    global_dimensions = _mode_dimensions(plan)
    dimensions = dict(global_dimensions)
    dimensions[output_mode] = output_slice.stop - output_slice.start
    dimensions[input_mode] = input_slice.stop - input_slice.start
    operand_slices = {}
    current = {}
    for ref in plan.inputs:
        selected = [slice(None)] * len(ref.spec.shape)
        if output_mode in ref.spec.modes:
            selected[ref.spec.modes.index(output_mode)] = output_slice
        if input_mode in ref.spec.modes:
            selected[ref.spec.modes.index(input_mode)] = input_slice
        selected = tuple(selected)
        local_shape = tuple(dimensions[mode] for mode in ref.spec.modes)
        layout = "C" if ref.key == variable_key else _slice_layout(
            ref.spec, selected, local_shape
        )
        rewritten = BufferRef(
            ref.key,
            TensorSpec(local_shape, ref.spec.dtype, layout, ref.spec.modes),
        )
        operand_slices[ref.key] = selected
        current[ref.key] = rewritten

    steps = []
    pack_index = 0
    reserved_keys = _plan_buffer_keys(plan)

    def require_c(ref):
        nonlocal pack_index
        if ref.spec.layout == "C":
            return ref
        while True:
            key = "distributed_pack_{}_{}".format(pack_index, ref.key)
            pack_index += 1
            if key not in reserved_keys:
                break
        reserved_keys.add(key)
        packed = BufferRef(
            key,
            TensorSpec(ref.spec.shape, ref.spec.dtype, "C", ref.spec.modes),
        )
        steps.append(
            TransformStep(ref, packed, tuple(range(len(ref.spec.shape))), True)
        )
        current[ref.key] = packed
        return packed

    def rewrite_pair(operation):
        left = require_c(current[operation.left.key])
        right = require_c(current[operation.right.key])
        output = BufferRef(
            operation.output.key,
            _new_spec(operation.output, dimensions, layout="C"),
        )
        if isinstance(operation, BatchedMatmulStep):
            rewritten = BatchedMatmulStep(
                left,
                right,
                output,
                operation.batch_modes,
                operation.contracted_modes,
            )
        else:
            rewritten = MatmulStep(
                left, right, output, operation.contracted_modes
            )
        current[operation.output.key] = output
        return rewritten

    for step in plan.steps:
        if isinstance(step, TransformStep):
            input_ref = current[step.input.key]
            if step.copy:
                layout = step.output.spec.layout
            else:
                layout = _transform_view_layout(input_ref.spec, step.axes)
            output_ref = BufferRef(
                step.output.key,
                _new_spec(step.output, dimensions, layout=layout),
            )
            rewritten = TransformStep(input_ref, output_ref, step.axes, step.copy)
            current[step.output.key] = output_ref
            steps.append(rewritten)
        elif isinstance(step, ReductionStep):
            input_ref = current[step.input.key]
            output_ref = BufferRef(
                step.output.key, _new_spec(step.output, dimensions)
            )
            rewritten = ReductionStep(input_ref, output_ref, step.reduced_modes)
            current[step.output.key] = output_ref
            steps.append(rewritten)
        elif isinstance(step, (MatmulStep, BatchedMatmulStep)):
            steps.append(rewrite_pair(step))
        elif isinstance(step, GroupedMatmulStep):
            operations = tuple(rewrite_pair(operation) for operation in step.operations)
            steps.append(GroupedMatmulStep(operations))
        else:
            raise TypeError("unsupported execution step type in distributed rewrite")

    # Packing aliases replace entries in ``current``; declared inputs retain their
    # sliced specifications and are recovered from the first graph references.
    input_by_key = {}
    for step in steps:
        operations = step.operations if isinstance(step, GroupedMatmulStep) else (step,)
        for operation in operations:
            refs = (
                (operation.left, operation.right)
                if isinstance(operation, (MatmulStep, BatchedMatmulStep))
                else (operation.input,)
            )
            for ref in refs:
                if ref.key in operand_slices:
                    input_by_key.setdefault(ref.key, ref)
    for ref in plan.inputs:
        if ref.key not in input_by_key:
            selected = operand_slices[ref.key]
            local_shape = tuple(dimensions[mode] for mode in ref.spec.modes)
            layout = "C" if ref.key == variable_key else _slice_layout(
                ref.spec, selected, local_shape
            )
            input_by_key[ref.key] = BufferRef(
                ref.key, TensorSpec(local_shape, ref.spec.dtype, layout, ref.spec.modes)
            )
    inputs = tuple(input_by_key[ref.key] for ref in plan.inputs)
    output = current[plan.output.key]
    rewritten_steps = tuple(steps)
    workspace_bytes = workspace_bytes_for_steps(rewritten_steps, output.key)
    plan_hash = _plan_hash(
        plan.operation,
        inputs,
        output,
        rewritten_steps,
        workspace_bytes,
        plan.planner_source,
        plan.oe_path,
        plan.override_reason,
    )
    rewritten_plan = ExecutionPlan(
        operation=plan.operation,
        inputs=inputs,
        output=output,
        steps=rewritten_steps,
        workspace_bytes=workspace_bytes,
        planner_source=plan.planner_source,
        oe_path=plan.oe_path,
        override_reason=plan.override_reason,
        plan_hash=plan_hash,
    )
    return DistributedBlockPlan(
        rank=rank,
        source_rank=source_rank,
        execution_plan=rewritten_plan,
        operand_slices=operand_slices,
    )


def _placement_hash(
    plan,
    variable_key,
    output_mode,
    input_mode,
    output_sharding,
    input_sharding,
    blocks,
    memory_estimates,
):
    payload = {
        "source_plan_hash": plan.plan_hash,
        "variable_key": variable_key,
        "output_mode": output_mode,
        "input_mode": input_mode,
        "output_slices": [
            _slice_payload(value) for value in output_sharding.local_slices
        ],
        "input_slices": [
            _slice_payload(value) for value in input_sharding.local_slices
        ],
        "blocks": [
            {
                "rank": block.rank,
                "source_rank": block.source_rank,
                "plan_hash": block.execution_plan.plan_hash,
                "operand_slices": {
                    key: _slice_payload(value)
                    for key, value in sorted(block.operand_slices.items())
                },
            }
            for block in blocks
        ],
        "memory_estimates": [
            {
                "resident_static_bytes": estimate.resident_static_bytes,
                "local_input_bytes": estimate.local_input_bytes,
                "receive_buffer_bytes": estimate.receive_buffer_bytes,
                "output_accumulator_bytes": estimate.output_accumulator_bytes,
                "output_contribution_bytes": estimate.output_contribution_bytes,
                "control_status_bytes": estimate.control_status_bytes,
                "host_control_status_bytes": estimate.host_control_status_bytes,
                "preflight_hash_device_bytes": (
                    estimate.preflight_hash_device_bytes
                ),
                "preflight_hash_host_bytes": estimate.preflight_hash_host_bytes,
                "preflight_status_device_bytes": (
                    estimate.preflight_status_device_bytes
                ),
                "preflight_status_host_bytes": estimate.preflight_status_host_bytes,
                "workspace_bytes": estimate.workspace_bytes,
                "device_bytes": estimate.device_bytes,
                "host_bytes": estimate.host_bytes,
            }
            for estimate in memory_estimates
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _shape_nbytes(shape, dtype):
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * np.dtype(dtype).itemsize


@dataclass(frozen=True)
class _ExecutionMemoryProfile:
    contribution_retained_bytes: int
    peak_bytes: int


def _matmul_bucket_key(operation):
    contracted_count = len(operation.contracted_modes)
    left_shape = operation.left.spec.shape
    right_shape = operation.right.spec.shape
    left_free = left_shape[:-contracted_count]
    contracted = left_shape[-contracted_count:]
    right_free = right_shape[contracted_count:]
    return (
        operation.left.spec.dtype,
        int(np.prod(left_free)),
        int(np.prod(right_free)),
        int(np.prod(contracted)),
    )


def _execution_memory_profile(execution_plan):
    """Model device allocations retained by the executor's buffer mapping."""
    allocation_sizes = {}
    buffer_allocations = {ref.key: frozenset() for ref in execution_plan.inputs}
    retained_bytes = 0
    peak_bytes = 0
    next_allocation = 0

    def retain(output_refs, size):
        nonlocal next_allocation, retained_bytes, peak_bytes
        allocation = next_allocation
        next_allocation += 1
        peak_bytes = max(peak_bytes, retained_bytes + size)
        allocation_sizes[allocation] = size
        retained_bytes += size
        retained = frozenset((allocation,))
        for output_ref in output_refs:
            buffer_allocations[output_ref.key] = retained

    for step in execution_plan.steps:
        if isinstance(step, TransformStep):
            if step.copy:
                retain((step.output,), step.output.spec.nbytes)
            else:
                buffer_allocations[step.output.key] = buffer_allocations[
                    step.input.key
                ]
            continue
        if isinstance(step, (ReductionStep, MatmulStep, BatchedMatmulStep)):
            retain((step.output,), step.output.spec.nbytes)
            continue

        buckets = {}
        for operation in step.operations:
            buckets.setdefault(_matmul_bucket_key(operation), []).append(operation)
        left_pack_bytes = 0
        right_pack_bytes = 0
        for (dtype, m, n, k), operations in buckets.items():
            itemsize = np.dtype(dtype).itemsize
            if len(operations) == 1:
                result_bytes = m * n * itemsize
                peak_bytes = max(
                    peak_bytes,
                    retained_bytes
                    + left_pack_bytes
                    + right_pack_bytes
                    + result_bytes,
                )
                retain((operations[0].output,), result_bytes)
                continue
            count = len(operations)
            new_left_pack_bytes = count * m * k * itemsize
            new_right_pack_bytes = count * k * n * itemsize
            result_bytes = count * m * n * itemsize
            peak_bytes = max(
                peak_bytes,
                retained_bytes
                + left_pack_bytes
                + right_pack_bytes
                + new_left_pack_bytes,
            )
            left_pack_bytes = new_left_pack_bytes
            peak_bytes = max(
                peak_bytes,
                retained_bytes
                + left_pack_bytes
                + right_pack_bytes
                + new_right_pack_bytes,
            )
            right_pack_bytes = new_right_pack_bytes
            # Result views retain C; Python keeps current A/B locals after the bucket.
            peak_bytes = max(
                peak_bytes,
                retained_bytes
                + left_pack_bytes
                + right_pack_bytes
                + result_bytes,
            )
            retain(
                tuple(operation.output for operation in operations),
                result_bytes,
            )

    contribution_retained_bytes = sum(
        allocation_sizes[allocation]
        for allocation in buffer_allocations[execution_plan.output.key]
    )
    return _ExecutionMemoryProfile(
        contribution_retained_bytes=contribution_retained_bytes,
        peak_bytes=peak_bytes,
    )


def _memory_estimates(
    execution_plan,
    variable_key,
    output_sharding,
    input_sharding,
    blocks,
):
    variable_ref = next(
        ref for ref in execution_plan.inputs if ref.key == variable_key
    )
    resident_static_bytes = sum(
        ref.spec.nbytes for ref in execution_plan.inputs if ref.key != variable_key
    )
    receive_buffer_bytes = (
        input_sharding.max_local_elements * np.dtype(variable_ref.spec.dtype).itemsize
    )
    size = output_sharding.parts
    control_status_bytes = np.dtype(np.int32).itemsize
    if size > 1:
        preflight_hash_device_bytes = 3 * 4 * np.dtype(np.uint64).itemsize
        preflight_hash_host_bytes = 2 * 4 * np.dtype(np.uint64).itemsize
        preflight_status_device_bytes = 2 * control_status_bytes
        preflight_status_host_bytes = control_status_bytes
    else:
        preflight_hash_device_bytes = 0
        preflight_hash_host_bytes = 0
        preflight_status_device_bytes = 0
        preflight_status_host_bytes = 0
    estimates = []
    for rank in range(size):
        local_input_bytes = _shape_nbytes(
            input_sharding.local_shape(rank), variable_ref.spec.dtype
        )
        rank_blocks = blocks[rank * size : (rank + 1) * size]
        output_bytes = rank_blocks[0].execution_plan.output.spec.nbytes
        execution_profiles = tuple(
            _execution_memory_profile(block.execution_plan) for block in rank_blocks
        )
        output_contribution_bytes = max(
            profile.contribution_retained_bytes for profile in execution_profiles
        )
        execution_peak_bytes = max(
            profile.peak_bytes for profile in execution_profiles
        )
        workspace_bytes = execution_peak_bytes - output_contribution_bytes
        resident_device_bytes = resident_static_bytes + local_input_bytes
        preflight_device_peak = resident_device_bytes + max(
            preflight_hash_device_bytes,
            preflight_status_device_bytes,
        )
        execution_device_peak = (
            resident_device_bytes
            + receive_buffer_bytes
            + output_bytes
            + control_status_bytes
            + max(execution_peak_bytes, preflight_status_device_bytes)
        )
        device_bytes = max(preflight_device_peak, execution_device_peak)
        host_bytes = max(
            preflight_hash_host_bytes,
            preflight_status_host_bytes,
            control_status_bytes + preflight_status_host_bytes,
        )
        estimates.append(
            DistributedMemoryEstimate(
                resident_static_bytes=resident_static_bytes,
                local_input_bytes=local_input_bytes,
                receive_buffer_bytes=receive_buffer_bytes,
                output_accumulator_bytes=output_bytes,
                output_contribution_bytes=output_contribution_bytes,
                control_status_bytes=control_status_bytes,
                host_control_status_bytes=control_status_bytes,
                preflight_hash_device_bytes=preflight_hash_device_bytes,
                preflight_hash_host_bytes=preflight_hash_host_bytes,
                preflight_status_device_bytes=preflight_status_device_bytes,
                preflight_status_host_bytes=preflight_status_host_bytes,
                workspace_bytes=workspace_bytes,
                device_bytes=device_bytes,
                host_bytes=host_bytes,
            )
        )
    return tuple(estimates)


def plan_distributed_execution(
    execution_plan,
    *,
    variable_key,
    world_size,
    output_mode=None,
    input_mode=None,
):
    if not isinstance(execution_plan, ExecutionPlan):
        raise TypeError("execution_plan must be an ExecutionPlan")
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if not isinstance(variable_key, str) or not variable_key:
        raise ValueError("variable_key must be a non-empty string")
    matching = tuple(ref for ref in execution_plan.inputs if ref.key == variable_key)
    if not matching:
        raise ValueError("unknown variable input {!r}".format(variable_key))
    variable_ref = matching[0]
    dimensions = _mode_dimensions(execution_plan)
    output_candidates = tuple(
        mode
        for mode in execution_plan.output.spec.modes
        if mode not in variable_ref.spec.modes and dimensions[mode] >= world_size
    )
    input_candidates = tuple(
        mode
        for mode in variable_ref.spec.modes
        if mode not in execution_plan.output.spec.modes
        and dimensions[mode] >= world_size
    )
    if (output_mode is None) != (input_mode is None):
        raise ValueError("output_mode and input_mode must be provided together")
    if output_mode is None:
        if not output_candidates or not input_candidates:
            raise ValueError("no legal distributed axis pair")
        output_mode = output_candidates[0]
        input_mode = input_candidates[0]
    else:
        if output_mode not in output_candidates:
            raise ValueError("output_mode is not a legal distributed output mode")
        if input_mode not in input_candidates:
            raise ValueError("input_mode is not a legal distributed input mode")
    output_axis = execution_plan.output.spec.modes.index(output_mode)
    input_axis = variable_ref.spec.modes.index(input_mode)
    output_sharding = shard_axis(
        execution_plan.output.spec.shape, output_axis, world_size
    )
    input_sharding = shard_axis(variable_ref.spec.shape, input_axis, world_size)
    blocks = []
    for rank in range(world_size):
        output_slice = output_sharding.local_slices[rank][output_axis]
        for source_rank in range(world_size):
            input_slice = input_sharding.local_slices[source_rank][input_axis]
            blocks.append(
                _rewrite_block(
                    execution_plan,
                    variable_key,
                    output_mode,
                    input_mode,
                    output_slice,
                    input_slice,
                    rank,
                    source_rank,
                )
            )
    blocks = tuple(blocks)
    memory_estimates = _memory_estimates(
        execution_plan,
        variable_key,
        output_sharding,
        input_sharding,
        blocks,
    )
    placement_hash = _placement_hash(
        execution_plan,
        variable_key,
        output_mode,
        input_mode,
        output_sharding,
        input_sharding,
        blocks,
        memory_estimates,
    )
    return DistributedPlan(
        execution_plan=execution_plan,
        variable_key=variable_key,
        output_mode=output_mode,
        input_mode=input_mode,
        output_sharding=output_sharding,
        input_sharding=input_sharding,
        block_plans=blocks,
        memory_estimates=memory_estimates,
        placement_hash=placement_hash,
    )


__all__ = [
    "DistributedBlockPlan",
    "DistributedMemoryEstimate",
    "DistributedPlan",
    "plan_distributed_execution",
]
