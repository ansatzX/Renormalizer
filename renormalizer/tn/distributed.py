# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""TTNS adapters for mapped distributed Krylov and two-site Davidson."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from renormalizer.backend._distributed.center import (
    build_mapped_local_operator,
    collective_elapsed,
    copy_backend_metadata as _copy_metadata,
    coordinate_adapter_decision,
    host_array as _host_array,
    measured_execution as _measured_execution,
    record_phase_summary,
    record_fallback_solve,
    record_distributed_solve,
    resolve_solver_config as _resolve_solver_config,
    result_dtype as _result_dtype,
    run_adapter_root_fallback,
    run_davidson_root_fallback as _run_davidson_fallback,
    run_synchronized_setup_phase,
    run_synchronized_state_update,
    synchronize_state_update,
    validate_distributed_backend,
)
from renormalizer.backend._distributed.sharding import DistributedTensor
from renormalizer.backend._distributed.solvers import (
    run_sharded_davidson,
    run_sharded_krylov,
)
from renormalizer.backend.config import DistributedExecutionConfig
from renormalizer.cons import backend


@dataclass(frozen=True)
class TtnsCenterDescriptor:
    node_index: int
    parent_index: int
    child_index: int
    degree: int
    center_kind: str
    center_shape: tuple[int, ...]


def _validate_execution(distributed_execution):
    if not isinstance(distributed_execution, DistributedExecutionConfig):
        raise TypeError("distributed_execution must be a DistributedExecutionConfig")
    selected = backend.current
    return validate_distributed_backend(
        distributed_execution, selected, network="ttns"
    )


def _descriptor(
    *,
    node_index,
    parent_index,
    child_index,
    degree,
    center_kind,
    center_shape,
):
    return TtnsCenterDescriptor(
        int(node_index),
        int(parent_index),
        int(child_index),
        int(degree),
        str(center_kind),
        tuple(int(dimension) for dimension in center_shape),
    )


def coordinate_ttns_workflow_entry(
    distributed_execution,
    *,
    operation,
    node_count,
    center_kind,
    solver_controls,
    selectors,
    supported,
    fallback_reason_code,
):
    """Coordinate a tree workflow route before canonical or environment setup."""
    selected = _validate_execution(distributed_execution)
    return coordinate_adapter_decision(
        distributed_execution,
        selected,
        network="ttns",
        operation=operation,
        center_kind=center_kind,
        center_shape=(int(node_count),),
        topology={"node_count": int(node_count)},
        solver_controls=dict(solver_controls),
        selectors=dict(selectors),
        supported=bool(supported),
        fallback_reason_code=fallback_reason_code,
    )


def run_ttns_krylov(
    hop,
    *,
    distributed_execution,
    center,
    center_shape,
    node_index,
    parent_index,
    child_index,
    degree,
    center_kind,
    coefficient,
    solver_config=None,
    counters=None,
):
    selected = _validate_execution(distributed_execution)
    distributed_execution = _measured_execution(distributed_execution)
    def convert_center():
        descriptor = _descriptor(
            node_index=node_index,
            parent_index=parent_index,
            child_index=child_index,
            degree=degree,
            center_kind=center_kind,
            center_shape=center_shape,
        )
        return (
            descriptor,
            _resolve_solver_config(
                distributed_execution, "krylov", solver_config
            ),
            selected.reshape(center, descriptor.center_shape),
        )

    descriptor, options, full_center = run_synchronized_setup_phase(
        distributed_execution,
        convert_center,
        "ttns Krylov center conversion",
        counters=counters,
    )
    try:
        operator, vector_map, prepared_center = build_mapped_local_operator(
            hop,
            full_center,
            distributed_execution,
            selected,
            descriptor.center_shape,
            counters=counters,
            network="ttns",
            operation="krylov",
            center_kind=descriptor.center_kind,
            topology={
                "node_index": descriptor.node_index,
                "parent_index": descriptor.parent_index,
                "child_index": descriptor.child_index,
                "degree": descriptor.degree,
            },
            solver_controls={**options, "coefficient": coefficient},
            selectors={"solver": "krylov"},
        )
    except NotImplementedError:
        if selected.config.fallback_policy != "legacy_oe":
            raise
        from renormalizer.lib import expm_krylov
        if counters is None:
            counters = {}
        counters.setdefault("allgather_calls", 0)
        def allocate_fallback():
            metadata = selected.array_namespace.zeros(1, dtype=np.int64)
            receive = selected.array_namespace.empty(
                int(np.prod(descriptor.center_shape)), dtype=full_center.dtype
            )
            return (
                metadata,
                receive,
                getattr(hop, "legacy_fallback_expression", hop),
                options.get("block_size", 50),
            )

        metadata, receive, legacy_hop, block_size = (
            run_synchronized_setup_phase(
                distributed_execution,
                allocate_fallback,
                "ttns Krylov fallback center allocation",
                counters=counters,
            )
        )
        def operation():
            result, iterations = expm_krylov(
                lambda value: legacy_hop(
                    value.reshape(descriptor.center_shape)
                ).ravel(),
                coefficient,
                full_center.ravel(),
                block_size=block_size,
            )
            metadata[0] = iterations
            return result

        basis_bytes = receive.nbytes * (int(block_size) + 2)
        result = run_ttns_root_fallback(
            operation,
            distributed_execution,
            receive,
            metadata_buffers=(metadata,),
            estimated_device_bytes=basis_bytes,
            estimated_host_bytes=basis_bytes,
            counters=counters,
        )
        iterations = int(_host_array(metadata)[0])
        state_collective_start = collective_elapsed(
            distributed_execution.collective
        )
        synchronize_state_update(
            distributed_execution,
            [result],
            metadata=("ttns", "krylov", descriptor.center_kind, "fallback"),
        )
        state_collective_s = max(
            collective_elapsed(distributed_execution.collective)
            - state_collective_start,
            0.0,
        )
        counters["allreduce_calls"] = counters.get("allreduce_calls", 0) + 3
        record_fallback_solve(
            distributed_execution,
            solver_dtype=result.dtype,
            global_count=result.size,
            packed_qn=False,
            network="ttns",
            center_kind=descriptor.center_kind,
            solver="krylov",
            hv_count=iterations,
            counters=counters,
            synchronization_s=(
                counters.get("last_root_synchronization_s", 0.0)
                + state_collective_s
            ),
            compute_s=counters.get("last_root_compute_s", 0.0),
        )
        return result, iterations
    def extract_local():
        rank = distributed_execution.context.rank
        local = vector_map.extract_local(prepared_center, rank, selected)
        return DistributedTensor(vector_map.solver_sharding, rank, local)

    vector = run_synchronized_setup_phase(
        distributed_execution,
        extract_local,
        "ttns Krylov local extraction",
        counters=operator.counters,
    )
    solver_collective_start = collective_elapsed(distributed_execution.collective)
    solver_start = perf_counter()
    result, iterations = run_sharded_krylov(
        operator,
        vector,
        coefficient,
        collective=distributed_execution.collective,
        config=options,
    )
    solver_wall_s = perf_counter() - solver_start
    solver_collective_s = max(
        collective_elapsed(distributed_execution.collective)
        - solver_collective_start,
        0.0,
    )
    compute_s = max(solver_wall_s - solver_collective_s, 0.0)
    materialization_collective_start = collective_elapsed(
        distributed_execution.collective
    )
    materialization_start = perf_counter()
    full = vector_map.materialize(
        result.local_array, distributed_execution.collective, selected
    )
    materialization_s = perf_counter() - materialization_start
    materialization_collective_s = max(
        collective_elapsed(distributed_execution.collective)
        - materialization_collective_start,
        0.0,
    )
    synchronization_s = solver_collective_s + materialization_collective_s
    record_phase_summary(
        "ttns",
        "boundary_materialization",
        "ordered_broadcast",
        materialization_s,
    )
    operator.counters["boundary_materialization_broadcasts"] += (
        distributed_execution.context.world_size
    )
    state_collective_start = collective_elapsed(distributed_execution.collective)
    synchronize_state_update(
        distributed_execution,
        [full],
        metadata=("ttns", "krylov", descriptor.center_kind, "supported"),
    )
    synchronization_s += max(
        collective_elapsed(distributed_execution.collective)
        - state_collective_start,
        0.0,
    )
    record_distributed_solve(
        operator,
        vector_map,
        network="ttns",
        center_kind=descriptor.center_kind,
        solver="krylov",
        hv_count=iterations,
        compute_s=compute_s,
        synchronization_s=synchronization_s,
    )
    return full.ravel(), iterations


def run_ttns_davidson(
    hop,
    *,
    distributed_execution,
    qn_mask,
    initial_guess,
    diagonal,
    node_index,
    parent_index,
    child_index,
    degree,
    center_kind,
    solver_config=None,
    counters=None,
    decision_selectors=None,
):
    selected = _validate_execution(distributed_execution)
    distributed_execution = _measured_execution(distributed_execution)
    def convert_center():
        mask = np.asarray(qn_mask, dtype=bool)
        descriptor = _descriptor(
            node_index=node_index,
            parent_index=parent_index,
            child_index=child_index,
            degree=degree,
            center_kind=center_kind,
            center_shape=mask.shape,
        )
        selectors = {
            "solver": "davidson",
            **(
                {}
                if decision_selectors is None
                else dict(decision_selectors)
            ),
        }
        return (
            mask,
            descriptor,
            _resolve_solver_config(
                distributed_execution, "davidson", solver_config
            ),
            _result_dtype(initial_guess, diagonal),
            selectors,
        )

    mask, descriptor, solver_options, dtype, selectors = (
        run_synchronized_setup_phase(
            distributed_execution,
            convert_center,
            "ttns Davidson center conversion",
            counters=counters,
        )
    )
    if center_kind != "two_site":
        route = coordinate_adapter_decision(
            distributed_execution,
            selected,
            network="ttns",
            operation="davidson",
            center_kind=descriptor.center_kind,
            center_shape=descriptor.center_shape,
            topology={
                "node_index": descriptor.node_index,
                "parent_index": descriptor.parent_index,
                "child_index": descriptor.child_index,
                "degree": descriptor.degree,
            },
            qn_mask=mask,
            solver_controls=solver_options,
            selectors=selectors,
            supported=False,
            fallback_reason_code="unsupported_ttns_davidson_center_kind",
            counters=counters,
        )
        if route != "fallback":
            raise RuntimeError("unsupported TTNS center decision did not select fallback")
        raise NotImplementedError(
            "TTNS distributed ground state supports only existing two-site Davidson"
        )
    def allocate_center():
        namespace = selected.array_namespace
        full_guess = namespace.zeros(mask.shape, dtype=dtype)
        full_diagonal = namespace.zeros(mask.shape, dtype=dtype)
        backend_mask = selected.asarray(mask)
        full_guess[backend_mask] = selected.asarray(initial_guess, dtype=dtype)
        full_diagonal[backend_mask] = selected.asarray(diagonal, dtype=dtype)
        return full_guess, full_diagonal

    full_guess, full_diagonal = run_synchronized_setup_phase(
        distributed_execution,
        allocate_center,
        "ttns Davidson center allocation",
        counters=counters,
    )
    try:
        operator, vector_map, prepared_guess = build_mapped_local_operator(
            hop,
            full_guess,
            distributed_execution,
            selected,
            descriptor.center_shape,
            qn_mask=mask,
            counters=counters,
            network="ttns",
            operation="davidson",
            center_kind=descriptor.center_kind,
            topology={
                "node_index": descriptor.node_index,
                "parent_index": descriptor.parent_index,
                "child_index": descriptor.child_index,
                "degree": descriptor.degree,
            },
            solver_controls=solver_options,
            selectors=selectors,
        )
    except NotImplementedError:
        if selected.config.fallback_policy != "legacy_oe":
            raise
        if counters is None:
            counters = {}
        energy, vector, info = _run_davidson_fallback(
            hop,
            distributed_execution,
            selected,
            mask,
            full_guess,
            full_diagonal,
            coefficient=1.0,
            solver_config=solver_options,
            counters=counters,
        )
        record_phase_summary(
            "ttns",
            "root_fallback",
            "root_solver_and_broadcast",
            counters["last_root_fallback_s"],
        )
        def allocate_result():
            full = selected.array_namespace.zeros(mask.shape, dtype=vector.dtype)
            full[selected.asarray(mask)] = vector
            return full

        full = run_synchronized_setup_phase(
            distributed_execution,
            allocate_result,
            "ttns Davidson fallback result allocation",
            counters=counters,
        )
        state_collective_start = collective_elapsed(
            distributed_execution.collective
        )
        synchronize_state_update(
            distributed_execution,
            [full],
            metadata=("ttns", "davidson", descriptor.center_kind, "fallback"),
        )
        state_collective_s = max(
            collective_elapsed(distributed_execution.collective)
            - state_collective_start,
            0.0,
        )
        counters["allreduce_calls"] = counters.get("allreduce_calls", 0) + 3
        record_fallback_solve(
            distributed_execution,
            solver_dtype=vector.dtype,
            global_count=vector.size,
            packed_qn=True,
            network="ttns",
            center_kind=descriptor.center_kind,
            solver="davidson",
            hv_count=info.h_v_count,
            counters=counters,
            synchronization_s=(
                counters.get("last_root_synchronization_s", 0.0)
                + state_collective_s
            ),
            compute_s=counters.get("last_root_compute_s", 0.0),
            solver_residual_norm=info.residual_norm,
        )
        return energy, _host_array(full), info
    def extract_local():
        rank = distributed_execution.context.rank
        vector = DistributedTensor(
            vector_map.solver_sharding,
            rank,
            vector_map.extract_local(prepared_guess, rank, selected),
        )
        diagonal_tensor = DistributedTensor(
            vector_map.solver_sharding,
            rank,
            vector_map.extract_local(full_diagonal, rank, selected),
        )
        options = dict(solver_options)
        options["diagonal"] = diagonal_tensor
        return vector, options

    vector, options = run_synchronized_setup_phase(
        distributed_execution,
        extract_local,
        "ttns Davidson local extraction",
        counters=operator.counters,
    )
    solver_collective_start = collective_elapsed(distributed_execution.collective)
    solver_start = perf_counter()
    energy, result, info = run_sharded_davidson(
        operator,
        vector,
        collective=distributed_execution.collective,
        config=options,
    )
    solver_wall_s = perf_counter() - solver_start
    solver_collective_s = max(
        collective_elapsed(distributed_execution.collective)
        - solver_collective_start,
        0.0,
    )
    compute_s = max(solver_wall_s - solver_collective_s, 0.0)
    materialization_collective_start = collective_elapsed(
        distributed_execution.collective
    )
    materialization_start = perf_counter()
    full = vector_map.materialize(
        result.local_array, distributed_execution.collective, selected
    )
    materialization_s = perf_counter() - materialization_start
    materialization_collective_s = max(
        collective_elapsed(distributed_execution.collective)
        - materialization_collective_start,
        0.0,
    )
    synchronization_s = solver_collective_s + materialization_collective_s
    record_phase_summary(
        "ttns",
        "boundary_materialization",
        "ordered_broadcast",
        materialization_s,
    )
    operator.counters["boundary_materialization_broadcasts"] += (
        distributed_execution.context.world_size
    )
    state_collective_start = collective_elapsed(distributed_execution.collective)
    synchronize_state_update(
        distributed_execution,
        [full],
        metadata=("ttns", "davidson", descriptor.center_kind, "supported"),
    )
    synchronization_s += max(
        collective_elapsed(distributed_execution.collective)
        - state_collective_start,
        0.0,
    )
    record_distributed_solve(
        operator,
        vector_map,
        network="ttns",
        center_kind=descriptor.center_kind,
        solver="davidson",
        hv_count=info.h_v_count,
        compute_s=compute_s,
        synchronization_s=synchronization_s,
        solver_residual_norm=info.residual_norm,
    )
    return energy, _host_array(full), info


def run_ttns_root_fallback(*args, **kwargs):
    if len(args) > 3:
        raise TypeError("run_ttns_root_fallback accepts three positional arguments")
    names = ("operation", "distributed_execution", "receive_buffer")
    values = {}
    for index, name in enumerate(names):
        if index < len(args):
            if name in kwargs:
                raise TypeError("{} passed by position and keyword".format(name))
            values[name] = args[index]
        else:
            try:
                values[name] = kwargs.pop(name)
            except KeyError as error:
                raise TypeError("missing required argument: {}".format(name)) from error
    distributed_execution = values["distributed_execution"]
    counters = kwargs.get("counters")
    if counters is None:
        counters = {}
        kwargs["counters"] = counters
    selected = _validate_execution(distributed_execution)
    result = run_adapter_root_fallback(
        values["operation"],
        distributed_execution,
        selected,
        values["receive_buffer"],
        **kwargs,
    )
    record_phase_summary(
        "ttns",
        "root_fallback",
        "root_solver_and_broadcast",
        counters["last_root_fallback_s"],
    )
    return result


def run_ttns_ground_state_fallback(
    operation,
    *,
    distributed_execution,
    qn_mask,
    initial_guess,
    node_index,
    parent_index,
    child_index,
    degree,
    center_kind,
    solver_controls,
    selectors,
    counters=None,
):
    selected = _validate_execution(distributed_execution)
    distributed_execution = _measured_execution(distributed_execution)
    if counters is None:
        counters = {}
    mask, topology, controls, decision_selectors = run_synchronized_setup_phase(
        distributed_execution,
        lambda: (
            np.asarray(qn_mask, dtype=bool),
            {
                "node_index": int(node_index),
                "parent_index": int(parent_index),
                "child_index": int(child_index),
                "degree": int(degree),
            },
            dict(solver_controls),
            dict(selectors),
        ),
        "ttns ground-state center conversion",
        counters=counters,
    )
    route = coordinate_adapter_decision(
        distributed_execution,
        selected,
        network="ttns",
        operation="ground_state",
        center_kind=center_kind,
        center_shape=mask.shape,
        topology=topology,
        qn_mask=mask,
        solver_controls=controls,
        selectors=decision_selectors,
        supported=False,
        fallback_reason_code="unsupported_ground_state_mode",
        counters=counters,
    )
    if route != "fallback":
        raise RuntimeError("unsupported TTNS decision did not select fallback")
    if center_kind != "two_site":
        raise NotImplementedError(
            "TTNS ground-state fallback is limited to the existing two-site workflow"
        )
    def allocate():
        namespace = selected.array_namespace
        receive = namespace.empty(
            int(np.count_nonzero(mask)), dtype=np.dtype(initial_guess.dtype)
        )
        energy_buffer = namespace.zeros(1, dtype=np.float64)
        return namespace, receive, energy_buffer

    namespace, receive, energy_buffer = run_synchronized_setup_phase(
        distributed_execution,
        allocate,
        "ttns ground-state center allocation",
        counters=counters,
    )

    def root_operation():
        energy, vector = operation()
        vector = namespace.ascontiguousarray(
            selected.asarray(vector, dtype=receive.dtype).reshape(-1)
        )
        if tuple(vector.shape) != tuple(receive.shape):
            raise ValueError("TTNS fallback vector size mismatch")
        _copy_metadata(energy_buffer, (energy,), selected)
        return vector

    result = run_ttns_root_fallback(
        root_operation,
        distributed_execution,
        receive,
        metadata_buffers=(energy_buffer,),
        estimated_device_bytes=receive.nbytes * 6,
        estimated_host_bytes=receive.nbytes * 6,
        counters=counters,
    )
    record_fallback_solve(
        distributed_execution,
        solver_dtype=result.dtype,
        global_count=result.size,
        packed_qn=True,
        network="ttns",
        center_kind=center_kind,
        solver="ground_state:{}".format(solver_controls.get("algo", "unknown")),
        hv_count=0,
        counters=counters,
        synchronization_s=counters.get("last_root_synchronization_s", 0.0),
        compute_s=counters.get("last_root_compute_s", 0.0),
    )
    return float(_host_array(energy_buffer)[0]), _host_array(result).copy()


def synchronize_ttns_update(
    distributed_execution, tensors, *, qn_arrays=(), metadata=()
):
    return synchronize_state_update(
        distributed_execution,
        tensors,
        qn_arrays=qn_arrays,
        metadata=metadata,
    )


def _synchronize_ttns_state(
    distributed_execution, state, *, metadata=(), operation=None
):
    if operation is None:
        operation = lambda: None
    return run_synchronized_state_update(
        distributed_execution,
        operation,
        lambda: (
            [node.tensor for node in state],
            [node.qn for node in state],
        ),
        metadata=lambda: ("ttns", state.coeff, *metadata),
    )


def _synchronize_ttns_result(distributed_execution, operation, *, metadata=()):
    holder = {}

    def produce():
        holder["state"] = operation()
        return holder["state"]

    def payload():
        state = holder["state"]
        return [node.tensor for node in state], [node.qn for node in state]

    return run_synchronized_state_update(
        distributed_execution,
        produce,
        payload,
        metadata=lambda: ("ttns", holder["state"].coeff, *metadata),
    )


__all__ = [
    "TtnsCenterDescriptor",
    "coordinate_ttns_workflow_entry",
    "run_ttns_davidson",
    "run_ttns_ground_state_fallback",
    "run_ttns_krylov",
    "run_ttns_root_fallback",
    "synchronize_ttns_update",
]
