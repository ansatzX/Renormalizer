# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""MPS adapters for mapped distributed Krylov and Davidson solves."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from renormalizer.backend._distributed.center import (
    MappedDistributedLocalOperator,
    build_mapped_local_operator,
    collective_elapsed,
    copy_backend_metadata as _copy_metadata,
    coordinate_adapter_decision,
    host_array as _host_array,
    krylov_root_fallback_memory_profile,
    measured_execution as _measured_execution,
    preflight_root_fallback_capacity,
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
class MpsCenterDescriptor:
    site_indices: tuple[int, ...]
    center_kind: str
    center_shape: tuple[int, ...]


@dataclass
class _ScaledSolverOperator:
    base: MappedDistributedLocalOperator
    coefficient: object

    @property
    def backend(self):
        return self.base.backend

    @property
    def context(self):
        return self.base.context

    @property
    def collective(self):
        return self.base.collective

    @property
    def counters(self):
        return self.base.counters

    @property
    def solver_input_sharding(self):
        return self.base.solver_input_sharding

    @property
    def solver_output_sharding(self):
        return self.base.solver_output_sharding

    @property
    def solver_dtype(self):
        return self.base.solver_dtype

    def solver_preflight(self):
        self.base.solver_preflight()

    def __call__(self, local_vector):
        result = self.base(local_vector)
        result *= self.coefficient
        return result


def _mps_ground_mode_supported(*, algo, nroots, omega, stacked_mpo):
    return (
        str(algo) == "davidson"
        and int(nroots) == 1
        and omega is None
        and not bool(stacked_mpo)
    )


def coordinate_mps_workflow_entry(
    distributed_execution,
    *,
    operation,
    site_count,
    center_kind,
    solver_controls,
    selectors,
    supported,
    fallback_reason_code,
):
    """Coordinate a workflow route before state or environment setup."""
    selected = _validate_execution(distributed_execution)
    return coordinate_adapter_decision(
        distributed_execution,
        selected,
        network="mps",
        operation=operation,
        center_kind=center_kind,
        center_shape=(int(site_count),),
        topology={"site_count": int(site_count)},
        solver_controls=dict(solver_controls),
        selectors=dict(selectors),
        supported=bool(supported),
        fallback_reason_code=fallback_reason_code,
    )


def _validate_execution(distributed_execution):
    if not isinstance(distributed_execution, DistributedExecutionConfig):
        raise TypeError("distributed_execution must be a DistributedExecutionConfig")
    selected = backend.current
    return validate_distributed_backend(
        distributed_execution, selected, network="mps"
    )


def run_mps_root_fallback(
    operation,
    distributed_execution,
    receive_buffer,
    *,
    metadata_buffers=(),
    estimated_device_bytes,
    estimated_host_bytes,
    counters=None,
    capacity_approval=None,
):
    selected = _validate_execution(distributed_execution)
    if counters is None:
        counters = {}
    result = run_adapter_root_fallback(
        operation,
        distributed_execution,
        selected,
        receive_buffer,
        metadata_buffers=metadata_buffers,
        estimated_device_bytes=estimated_device_bytes,
        estimated_host_bytes=estimated_host_bytes,
        counters=counters,
        capacity_approval=capacity_approval,
    )
    record_phase_summary(
        "mps",
        "root_fallback",
        "root_solver_and_broadcast",
        counters["last_root_fallback_s"],
    )
    return result


def run_mps_ivp_fallback(
    operation,
    *,
    distributed_execution,
    center,
    center_shape,
    site_indices,
    center_kind,
    solver,
    solver_controls,
    counters=None,
):
    selected = _validate_execution(distributed_execution)
    distributed_execution = _measured_execution(distributed_execution)
    if counters is None:
        counters = {}
    center_shape, site_indices, decision_controls = run_synchronized_setup_phase(
        distributed_execution,
        lambda: (
            tuple(int(dimension) for dimension in center_shape),
            tuple(int(index) for index in site_indices),
            {"solver": str(solver), **dict(solver_controls)},
        ),
        "mps IVP center conversion",
        counters=counters,
    )
    route = coordinate_adapter_decision(
        distributed_execution,
        selected,
        network="mps",
        operation="ivp",
        center_kind=center_kind,
        center_shape=center_shape,
        topology={"site_indices": site_indices},
        solver_controls=decision_controls,
        selectors={"ivp_solver": str(solver)},
        supported=False,
        fallback_reason_code="unsupported_ivp_solver",
        counters=counters,
    )
    if route != "fallback":
        raise RuntimeError("unsupported IVP decision did not select fallback")
    def profile_fallback():
        flat_center = selected.reshape(center, (-1,))
        estimated_bytes = int(flat_center.size) * np.dtype(flat_center.dtype).itemsize * 4
        return flat_center, estimated_bytes

    flat_center, estimated_bytes = run_synchronized_setup_phase(
        distributed_execution,
        profile_fallback,
        "mps IVP fallback profile",
        counters=counters,
    )
    capacity_approval = preflight_root_fallback_capacity(
        distributed_execution,
        selected,
        device_bytes=estimated_bytes,
        host_bytes=estimated_bytes,
        counters=counters,
    )

    def allocate():
        namespace = selected.array_namespace
        receive = namespace.empty(flat_center.shape, dtype=flat_center.dtype)
        metadata = namespace.zeros(1, dtype=np.int64)
        return namespace, receive, metadata

    namespace, receive, metadata = run_synchronized_setup_phase(
        distributed_execution,
        allocate,
        "mps IVP center allocation",
        counters=counters,
    )

    def root_operation():
        result, evaluations = operation()
        result = selected.asarray(result, dtype=receive.dtype).reshape(-1)
        if tuple(result.shape) != tuple(receive.shape):
            raise ValueError("IVP fallback result shape does not match center")
        metadata[0] = int(evaluations)
        return result

    result = run_mps_root_fallback(
        root_operation,
        distributed_execution,
        receive,
        metadata_buffers=(metadata,),
        estimated_device_bytes=estimated_bytes,
        estimated_host_bytes=estimated_bytes,
        counters=counters,
        capacity_approval=capacity_approval,
    )
    evaluations = int(_host_array(metadata).reshape(-1)[0])
    record_fallback_solve(
        distributed_execution,
        solver_dtype=result.dtype,
        global_count=result.size,
        packed_qn=False,
        network="mps",
        center_kind=center_kind,
        solver="ivp:{}".format(solver),
        hv_count=evaluations,
        counters=counters,
        synchronization_s=counters.get("last_root_synchronization_s", 0.0),
        compute_s=counters.get("last_root_compute_s", 0.0),
    )
    return result, evaluations


def run_mps_ground_state_fallback(
    operation,
    *,
    distributed_execution,
    qn_mask,
    initial_guesses,
    site_indices,
    center_kind,
    solver_controls,
    selectors,
    counters=None,
):
    selected = _validate_execution(distributed_execution)
    distributed_execution = _measured_execution(distributed_execution)
    if counters is None:
        counters = {}
    mask, controls, decision_selectors, site_indices, nroots = (
        run_synchronized_setup_phase(
            distributed_execution,
            lambda: (
                np.asarray(qn_mask, dtype=bool),
                dict(solver_controls),
                dict(selectors),
                tuple(int(index) for index in site_indices),
                int(dict(solver_controls).get("nroots", len(initial_guesses))),
            ),
            "mps ground-state center conversion",
            counters=counters,
        )
    )
    route = coordinate_adapter_decision(
        distributed_execution,
        selected,
        network="mps",
        operation="ground_state",
        center_kind=center_kind,
        center_shape=mask.shape,
        topology={"site_indices": site_indices},
        qn_mask=mask,
        solver_controls=controls,
        selectors=decision_selectors,
        supported=False,
        fallback_reason_code="unsupported_ground_state_mode",
        counters=counters,
    )
    if route != "fallback":
        raise RuntimeError("unsupported ground-state decision did not select fallback")
    def profile_fallback():
        allowed_count = int(np.count_nonzero(mask))
        dtype = _result_dtype(*initial_guesses)
        receive_bytes = allowed_count * nroots * np.dtype(dtype).itemsize
        return allowed_count, dtype, receive_bytes

    allowed_count, dtype, estimated_bytes = run_synchronized_setup_phase(
        distributed_execution,
        profile_fallback,
        "mps ground-state fallback profile",
        counters=counters,
    )
    estimated_bytes *= 6
    capacity_approval = preflight_root_fallback_capacity(
        distributed_execution,
        selected,
        device_bytes=estimated_bytes,
        host_bytes=estimated_bytes,
        counters=counters,
    )

    def allocate():
        namespace = selected.array_namespace
        receive = namespace.empty(allowed_count * nroots, dtype=dtype)
        energies = namespace.zeros(nroots, dtype=np.float64)
        return namespace, receive, energies

    namespace, receive, energies = (
        run_synchronized_setup_phase(
            distributed_execution,
            allocate,
            "mps ground-state center allocation",
            counters=counters,
        )
    )

    def root_operation():
        energy, vectors = operation()
        energy_values = np.asarray(energy).reshape(-1)
        vector_values = vectors if isinstance(vectors, (list, tuple)) else [vectors]
        if len(energy_values) != nroots or len(vector_values) != nroots:
            raise ValueError("ground-state fallback root count mismatch")
        packed = np.concatenate(
            [np.asarray(value).reshape(-1) for value in vector_values]
        )
        if packed.size != receive.size:
            raise ValueError("ground-state fallback vector size mismatch")
        _copy_metadata(energies, energy_values, selected)
        return selected.asarray(packed, dtype=dtype)

    result = run_mps_root_fallback(
        root_operation,
        distributed_execution,
        receive,
        metadata_buffers=(energies,),
        estimated_device_bytes=estimated_bytes,
        estimated_host_bytes=estimated_bytes,
        counters=counters,
        capacity_approval=capacity_approval,
    )
    host_result = _host_array(result)
    host_energies = _host_array(energies)
    vectors = [
        host_result[index * allowed_count : (index + 1) * allowed_count].copy()
        for index in range(nroots)
    ]
    record_fallback_solve(
        distributed_execution,
        solver_dtype=result.dtype,
        global_count=result.size,
        packed_qn=True,
        network="mps",
        center_kind=center_kind,
        solver="ground_state:{}".format(controls.get("algo", "unknown")),
        hv_count=0,
        counters=counters,
        synchronization_s=counters.get("last_root_synchronization_s", 0.0),
        compute_s=counters.get("last_root_compute_s", 0.0),
    )
    if nroots == 1:
        return float(host_energies[0]), vectors[0]
    return host_energies.copy(), vectors


def _mapped_operator(
    hop,
    full_center,
    distributed_execution,
    descriptor,
    *,
    qn_mask=None,
    counters=None,
    operation,
    solver_controls,
    selectors,
):
    selected = _validate_execution(distributed_execution)
    return build_mapped_local_operator(
        hop,
        full_center,
        distributed_execution,
        selected,
        descriptor.center_shape,
        qn_mask=qn_mask,
        counters=counters,
        network="mps",
        operation=operation,
        center_kind=descriptor.center_kind,
        topology={"site_indices": descriptor.site_indices},
        solver_controls=solver_controls,
        selectors=selectors,
    )


def run_mps_krylov(
    hop,
    *,
    distributed_execution,
    center,
    center_shape,
    site_indices,
    center_kind,
    coefficient,
    solver_config=None,
    counters=None,
):
    selected = _validate_execution(distributed_execution)
    distributed_execution = _measured_execution(distributed_execution)
    descriptor, options, full_center = run_synchronized_setup_phase(
        distributed_execution,
        lambda: (
            MpsCenterDescriptor(
                tuple(int(index) for index in site_indices),
                str(center_kind),
                tuple(int(dimension) for dimension in center_shape),
            ),
            _resolve_solver_config(
                distributed_execution, "krylov", solver_config
            ),
            selected.reshape(
                center, tuple(int(dimension) for dimension in center_shape)
            ),
        ),
        "mps Krylov center conversion",
        counters=counters,
    )
    try:
        operator, vector_map, prepared_center = _mapped_operator(
            hop,
            full_center,
            distributed_execution,
            descriptor,
            counters=counters,
            operation="krylov",
            solver_controls={
                **options,
                "coefficient": coefficient,
            },
            selectors={"solver": "krylov"},
        )
    except NotImplementedError:
        if selected.config.fallback_policy != "legacy_oe":
            raise
        from renormalizer.lib import expm_krylov

        if counters is None:
            counters = {}
        counters.setdefault("allgather_calls", 0)
        def profile_fallback():
            unknown = set(options) - {"block_size", "max_krylov_vectors"}
            if unknown:
                raise ValueError(
                    "unknown Krylov config keys: {}".format(
                        ", ".join(sorted(map(str, unknown)))
                    )
                )
            memory_profile = krylov_root_fallback_memory_profile(
                vector_bytes=int(full_center.nbytes),
                vector_count=int(full_center.size),
                dtype=np.dtype(full_center.dtype),
                coefficient=coefficient,
                block_size=options.get("block_size", 50),
                max_krylov_vectors=options.get("max_krylov_vectors"),
            )
            return (
                getattr(hop, "legacy_fallback_expression", hop),
                memory_profile,
            )

        legacy_hop, memory_profile = run_synchronized_setup_phase(
            distributed_execution,
            profile_fallback,
            "mps Krylov fallback profile",
            counters=counters,
        )
        capacity_approval = preflight_root_fallback_capacity(
            distributed_execution,
            selected,
            device_bytes=memory_profile.device_peak_bytes,
            host_bytes=memory_profile.host_peak_bytes,
            counters=counters,
        )

        def allocate_fallback():
            metadata = selected.array_namespace.zeros(1, dtype=np.int64)
            receive = selected.array_namespace.empty(
                int(np.prod(descriptor.center_shape)),
                dtype=np.dtype(memory_profile.result_dtype),
            )
            return (
                metadata,
                receive,
            )

        metadata, receive = (
            run_synchronized_setup_phase(
                distributed_execution,
                allocate_fallback,
                "mps Krylov fallback center allocation",
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
                block_size=memory_profile.block_size,
                max_krylov_vectors=memory_profile.max_krylov_vectors,
            )
            metadata[0] = iterations
            return result

        result = run_mps_root_fallback(
            operation,
            distributed_execution,
            receive,
            metadata_buffers=(metadata,),
            estimated_device_bytes=memory_profile.device_peak_bytes,
            estimated_host_bytes=memory_profile.host_peak_bytes,
            counters=counters,
            capacity_approval=capacity_approval,
        )
        iterations = int(_host_array(metadata)[0])
        state_collective_start = collective_elapsed(
            distributed_execution.collective
        )
        synchronize_state_update(
            distributed_execution,
            [result],
            metadata=("mps", "krylov", descriptor.center_kind, "fallback"),
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
            network="mps",
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
        local = vector_map.extract_local(prepared_center, rank, operator.backend)
        return DistributedTensor(vector_map.solver_sharding, rank, local)

    vector = run_synchronized_setup_phase(
        distributed_execution,
        extract_local,
        "mps Krylov local extraction",
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
        result.local_array, distributed_execution.collective, operator.backend
    )
    materialization_s = perf_counter() - materialization_start
    materialization_collective_s = max(
        collective_elapsed(distributed_execution.collective)
        - materialization_collective_start,
        0.0,
    )
    synchronization_s = solver_collective_s + materialization_collective_s
    record_phase_summary(
        "mps",
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
        metadata=("mps", "krylov", descriptor.center_kind, "supported"),
    )
    synchronization_s += max(
        collective_elapsed(distributed_execution.collective)
        - state_collective_start,
        0.0,
    )
    record_distributed_solve(
        operator,
        vector_map,
        network="mps",
        center_kind=descriptor.center_kind,
        solver="krylov",
        hv_count=iterations,
        compute_s=compute_s,
        synchronization_s=synchronization_s,
    )
    return full.ravel(), iterations


def run_mps_davidson(
    hop,
    *,
    distributed_execution,
    qn_mask,
    initial_guess,
    diagonal,
    site_indices,
    center_kind,
    coefficient=1.0,
    solver_config=None,
    counters=None,
    decision_selectors=None,
):
    selected = _validate_execution(distributed_execution)
    distributed_execution = _measured_execution(distributed_execution)
    def convert_center():
        mask = np.asarray(qn_mask, dtype=bool)
        descriptor = MpsCenterDescriptor(
            tuple(int(index) for index in site_indices),
            str(center_kind),
            tuple(int(dimension) for dimension in mask.shape),
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
            "mps Davidson center conversion",
            counters=counters,
        )
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
        "mps Davidson center allocation",
        counters=counters,
    )
    try:
        operator, vector_map, prepared_guess = _mapped_operator(
            hop,
            full_guess,
            distributed_execution,
            descriptor,
            qn_mask=mask,
            counters=counters,
            operation="davidson",
            solver_controls={
                **solver_options,
                "coefficient": coefficient,
            },
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
            coefficient=coefficient,
            solver_config=solver_options,
            counters=counters,
        )
        record_phase_summary(
            "mps",
            "root_fallback",
            "root_solver_and_broadcast",
            counters["last_root_fallback_s"],
        )
        state_collective_start = collective_elapsed(
            distributed_execution.collective
        )
        synchronize_state_update(
            distributed_execution,
            [vector],
            metadata=("mps", "davidson", descriptor.center_kind, "fallback"),
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
            network="mps",
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
        return energy, _host_array(vector), info
    def extract_local():
        solver_operator = (
            operator
            if coefficient == 1
            else _ScaledSolverOperator(operator, coefficient)
        )
        rank = distributed_execution.context.rank
        local_guess = vector_map.extract_local(prepared_guess, rank, selected)
        local_diagonal = vector_map.extract_local(full_diagonal, rank, selected)
        vector = DistributedTensor(vector_map.solver_sharding, rank, local_guess)
        diagonal_tensor = DistributedTensor(
            vector_map.solver_sharding, rank, local_diagonal
        )
        options = dict(solver_options)
        options["diagonal"] = diagonal_tensor
        return solver_operator, vector, options

    solver_operator, vector, options = run_synchronized_setup_phase(
        distributed_execution,
        extract_local,
        "mps Davidson local extraction",
        counters=operator.counters,
    )
    solver_collective_start = collective_elapsed(distributed_execution.collective)
    solver_start = perf_counter()
    energy, result, info = run_sharded_davidson(
        solver_operator,
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
        "mps",
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
        metadata=("mps", "davidson", descriptor.center_kind, "supported"),
    )
    synchronization_s += max(
        collective_elapsed(distributed_execution.collective)
        - state_collective_start,
        0.0,
    )
    record_distributed_solve(
        operator,
        vector_map,
        network="mps",
        center_kind=descriptor.center_kind,
        solver="davidson",
        hv_count=info.h_v_count,
        compute_s=compute_s,
        synchronization_s=synchronization_s,
        solver_residual_norm=info.residual_norm,
    )
    return energy, _host_array(vector_map.pack_baseline(full)), info


def synchronize_mps_update(
    distributed_execution, tensors, *, qn_arrays=(), metadata=()
):
    return synchronize_state_update(
        distributed_execution,
        tensors,
        qn_arrays=qn_arrays,
        metadata=metadata,
    )


def _synchronize_mps_state(
    distributed_execution, state, *, metadata=(), operation=None
):
    if operation is None:
        operation = lambda: None
    return run_synchronized_state_update(
        distributed_execution,
        operation,
        lambda: (
            [matrix.array for matrix in state],
            state.qn,
        ),
        metadata=lambda: (
            "mps",
            state.qnidx,
            bool(state.to_right),
            state.coeff,
            *metadata,
        ),
    )


def _synchronize_mps_result(distributed_execution, operation, *, metadata=()):
    holder = {}

    def produce():
        holder["state"] = operation()
        return holder["state"]

    def payload():
        state = holder["state"]
        return [matrix.array for matrix in state], state.qn

    return run_synchronized_state_update(
        distributed_execution,
        produce,
        payload,
        metadata=lambda: (
            "mps",
            holder["state"].qnidx,
            bool(holder["state"].to_right),
            holder["state"].coeff,
            *metadata,
        ),
    )


def _deterministic_mps_update(state, *args, metadata, **kwargs):
    import hashlib
    import json

    encoded = json.dumps(
        list(metadata), sort_keys=True, separators=(",", ":"), default=repr
    ).encode("ascii")
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:4], "little")
    random_state = np.random.get_state()
    np.random.seed(seed)
    try:
        return state._update_mps(*args, **kwargs)
    finally:
        np.random.set_state(random_state)


__all__ = [
    "MpsCenterDescriptor",
    "coordinate_mps_workflow_entry",
    "run_mps_davidson",
    "run_mps_ground_state_fallback",
    "run_mps_ivp_fallback",
    "run_mps_krylov",
    "run_mps_root_fallback",
    "synchronize_mps_update",
]
