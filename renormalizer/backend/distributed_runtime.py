# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""Explicit lifecycle for launcher-configured CuPy distributed execution."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import os
import threading
import uuid
import weakref

import numpy as np

from renormalizer.backend._distributed.async_owner import RuntimeTerminalQuarantine
from renormalizer.backend._distributed.context import (
    DistributedContext,
    DistributedRendezvous,
)
from renormalizer.backend._distributed.center import (
    normalize_distributed_backend_metadata,
)
from renormalizer.backend._distributed.mesh import DeviceMesh
from renormalizer.backend._distributed.providers import DeviceResidentProvider
from renormalizer.backend._distributed.residency import (
    MemoryBudgetResolution,
    ResidencyPlan,
    ResidencyPreflightReceipt,
    ResidencyRequest,
    budget_resolution_hash,
)
from renormalizer.backend._distributed.terminal import (
    _FatalTransition,
    _TerminalLifecycleGate,
    _TerminalPhase,
    _TERMINAL_TIMEOUT_S,
)
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


class _RuntimeCommunicatorFatalHook:
    def __init__(self, runtime):
        self._runtime_ref = weakref.ref(runtime)

    def _runtime(self):
        runtime = self._runtime_ref()
        if runtime is None:
            raise RuntimeError("distributed runtime fatal hook is unavailable")
        return runtime

    def __call__(self, primary):
        runtime = self._runtime()
        discovering_token = runtime._terminal_gate._current_thread_admission()
        return runtime._enter_communicator_fatal(
            primary, discovering_token=discovering_token
        )

    def begin(self, primary, *, owner=None, discovering_token=None):
        return self._runtime()._begin_communicator_fatal(
            primary, owner=owner, discovering_token=discovering_token
        )

    def publish(self, transition):
        return self._runtime()._publish_communicator_fatal_transition(transition)

    def complete(self, transition, snapshot):
        self._runtime()._terminal_gate.publish_fatal(transition, snapshot)


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
    _runtime_id: str = field(
        default_factory=lambda: uuid.uuid4().hex, init=False, repr=False
    )
    _issued_receipts: dict = field(default_factory=dict, init=False, repr=False)
    _active_provider: object = field(default=None, init=False, repr=False)
    _terminal_quarantine: RuntimeTerminalQuarantine = field(
        default_factory=RuntimeTerminalQuarantine, init=False, repr=False
    )
    _terminal_error: object = field(default=None, init=False, repr=False)
    _terminal_secondary_errors: tuple = field(
        default_factory=tuple, init=False, repr=False
    )
    _terminal_gate: _TerminalLifecycleGate = field(
        default_factory=_TerminalLifecycleGate, init=False, repr=False
    )
    _terminal_state_lock: object = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    _communicator_fatal_hook: object = field(default=None, init=False, repr=False)
    _pending_fatal_context: object = field(default=None, init=False, repr=False)
    _pending_fatal_collective: object = field(default=None, init=False, repr=False)
    _legacy_fatal_publication_owner: int | None = field(
        default=None, init=False, repr=False
    )

    @property
    def rank(self):
        return self.context.rank

    @property
    def local_rank(self):
        return self.context.local_rank

    @property
    def world_size(self):
        return self.context.world_size

    @property
    def terminal_poisoned(self):
        return self._terminal_error is not None

    def _require_usable(self):
        primary = None
        with self._terminal_gate._condition:
            if self._closed:
                raise RuntimeError("distributed runtime is closed")
            transition = self._terminal_gate._fatal_transition
            if transition is not None and self._terminal_gate._phase in (
                _TerminalPhase.FATAL_PUBLISHED,
                _TerminalPhase.RUNTIME_CLOSED,
            ):
                primary = transition.primary
            else:
                with self._terminal_state_lock:
                    backend_error = getattr(
                        self.backend, "_execution_terminal_error", None
                    )
                    candidate = (
                        self._terminal_error
                        if self._terminal_error is not None
                        else backend_error
                    )
                    if isinstance(candidate, BaseException):
                        primary, provider, lease, owner = (
                            self._normalize_communicator_fatal(candidate, None)
                        )
                        self._force_terminal_primary_locked(
                            primary,
                            provider=provider,
                            lease=lease,
                            owner=owner,
                        )
        if primary is not None:
            raise RuntimeError(
                "distributed runtime is terminal-poisoned"
            ) from primary

    def _accept_async_quarantine(self, owner, primary=None):
        error = owner.error if primary is None else primary
        if not isinstance(error, BaseException):
            raise TypeError("async quarantine failure must be an exception")
        with self._terminal_gate._condition:
            transition = self._terminal_gate._fatal_transition
            if transition is not None and self._terminal_gate._phase in (
                _TerminalPhase.FATAL_PUBLISHED,
                _TerminalPhase.RUNTIME_CLOSED,
            ):
                return
            with self._terminal_state_lock:
                canonical, provider, lease, owner = (
                    self._normalize_communicator_fatal(error, owner)
                )
                self._force_terminal_primary_locked(
                    canonical,
                    provider=provider,
                    lease=lease,
                    owner=owner,
                    retain_owner=True,
                )

    def _arm_communicator_fatal(self):
        install = getattr(self.collective, "_install_fatal_handler", None)
        if not callable(install):
            raise RuntimeError("collective does not provide fatal observation")
        hook = _RuntimeCommunicatorFatalHook(self)
        self._communicator_fatal_hook = hook
        install(hook)

    @contextmanager
    def _communicator_fatal_reservation(self, primary):
        if not isinstance(primary, BaseException):
            raise TypeError("communicator fatal failure must be an exception")
        reserve = getattr(self.collective, "_communicator_fatal_reservation", None)
        boundary = (
            reserve(primary, join_existing=True)
            if callable(reserve)
            else nullcontext(None)
        )
        with boundary as reservation:
            if reservation is not None:
                primary = reservation.primary
            yield primary, reservation

    def _remember_terminal_secondary_locked(self, error, primary, owner=None):
        if not isinstance(error, BaseException):
            return
        if error is primary:
            return
        if not any(
            retained is error for retained in self._terminal_secondary_errors
        ):
            self._terminal_secondary_errors = (
                *self._terminal_secondary_errors,
                error,
            )
        call = None if owner is None else getattr(owner, "_operator_call", None)
        record_secondary = (
            None if call is None else getattr(call, "record_secondary", None)
        )
        if callable(record_secondary):
            record_secondary(error)
        elif owner is not None:
            remember_secondary = getattr(owner, "_remember_secondary", None)
            if callable(remember_secondary):
                remember_secondary(error)
        collective = self._pending_fatal_collective
        if collective is None:
            collective = self.collective
        record_collective = (
            None
            if collective is None
            else getattr(collective, "_record_fatal_secondary", None)
        )
        if callable(record_collective):
            record_collective(error)

    def _force_async_primary_locked(self, owner, primary):
        if owner is None:
            return
        call = getattr(owner, "_operator_call", None)
        owner_primary = getattr(owner, "error", None)
        call_primary = None if call is None else getattr(call, "primary_error", None)
        if hasattr(owner, "error"):
            owner.error = primary
        if call is not None and hasattr(call, "primary_error"):
            call.primary_error = primary
        for error in (owner_primary, call_primary):
            self._remember_terminal_secondary_locked(error, primary)
        if isinstance(owner_primary, BaseException) and owner_primary is not primary:
            remember_secondary = getattr(owner, "_remember_secondary", None)
            if callable(remember_secondary):
                remember_secondary(owner_primary)
        if isinstance(call_primary, BaseException) and call_primary is not primary:
            record_secondary = getattr(call, "record_secondary", None)
            if callable(record_secondary):
                record_secondary(call_primary)

    def _normalize_communicator_fatal(self, primary, owner):
        if not isinstance(primary, BaseException):
            raise TypeError("communicator fatal failure must be an exception")
        discovered = primary
        provider = self._active_provider
        lease = (
            None
            if provider is None
            else getattr(provider, "_active_lease", None)
        )
        if owner is None and lease is not None:
            owner = getattr(lease, "_active_operator_owner", None)
            if owner is None:
                status_workspace = getattr(lease, "_status_workspace", None)
                if status_workspace is not None:
                    owner = status_workspace.borrower
        call = None if owner is None else getattr(owner, "_operator_call", None)
        backend = self.backend
        transition = self._terminal_gate._fatal_transition
        candidates = (
            self._terminal_error,
            None
            if backend is None
            else getattr(backend, "_execution_terminal_error", None),
            self._terminal_quarantine.first_error,
            None
            if provider is None
            else getattr(provider, "_terminal_error", None),
            None if lease is None else getattr(lease, "_poisoned_error", None),
            None if call is None else call.primary_error,
            discovered,
        )
        if transition is not None:
            primary = transition.primary
        else:
            primary = next(
                (
                    candidate
                    for candidate in candidates
                    if isinstance(candidate, BaseException)
                ),
                discovered,
            )
        for candidate in candidates:
            self._remember_terminal_secondary_locked(candidate, primary, owner)
        return primary, provider, lease, owner

    def _force_terminal_primary_locked(
        self,
        primary,
        *,
        provider=None,
        lease=None,
        owner=None,
        retain_owner=False,
    ):
        if not isinstance(primary, BaseException):
            raise TypeError("terminal primary must be an exception")
        transition = self._terminal_gate._fatal_transition
        if transition is not None and transition.primary is not primary:
            self._remember_terminal_secondary_locked(primary, transition.primary, owner)
            primary = transition.primary
        self._force_async_primary_locked(owner, primary)
        if transition is not None and self._terminal_gate._phase in (
            _TerminalPhase.FATAL_PUBLISHED,
            _TerminalPhase.RUNTIME_CLOSED,
        ):
            return transition.primary
        backend = self.backend
        existing = (
            self._terminal_error,
            None
            if backend is None
            else getattr(backend, "_execution_terminal_error", None),
            self._terminal_quarantine.first_error,
            None
            if provider is None
            else getattr(provider, "_terminal_error", None),
            None if lease is None else getattr(lease, "_poisoned_error", None),
        )
        for error in existing:
            self._remember_terminal_secondary_locked(error, primary, owner)
        self._terminal_quarantine.first_error = primary
        self._terminal_quarantine.retain_error(primary)
        if retain_owner and owner is not None:
            self._terminal_quarantine.retain(owner, primary)
        self._terminal_error = primary
        if backend is not None and (
            hasattr(backend, "_execution_terminal_error")
            or hasattr(backend, "__dict__")
        ):
            backend._execution_terminal_error = primary
        if provider is not None:
            provider._terminal_error = primary
        if lease is not None:
            lease._poisoned_error = primary
        return primary

    def _begin_communicator_fatal(
        self, primary, *, owner=None, discovering_token=None
    ):
        gate = self._terminal_gate
        with gate._condition:
            transition = gate._fatal_transition
            if gate._phase in (
                _TerminalPhase.FATAL_PUBLISHED,
                _TerminalPhase.RUNTIME_CLOSED,
            ) or self._closed:
                if transition is not None:
                    return transition
                raise RuntimeError("distributed runtime is closed")
            if gate._runtime_close_commit_selected:
                raise RuntimeError("runtime close is committed")
            with self._terminal_state_lock:
                primary, provider, lease, owner = (
                    self._normalize_communicator_fatal(primary, owner)
                )
                collective = self._pending_fatal_collective
                if collective is None:
                    collective = self.collective
                publish = (
                    None
                    if collective is None
                    else getattr(collective, "_publish_communicator_fatal", None)
                )
                if not callable(publish):
                    raise RuntimeError(
                        "collective does not provide fatal publication"
                    )
                reserve_outcome = getattr(
                    collective, "_reserve_runtime_fatal_outcome", None
                )
                elected_here = False

                def retain_fatal_context():
                    if self._pending_fatal_collective is None:
                        self._pending_fatal_collective = collective
                    if self._pending_fatal_context is None:
                        self._pending_fatal_context = (provider, lease, owner)
                    elif (
                        owner is not None
                        and self._pending_fatal_context[2] is None
                    ):
                        retained_provider, retained_lease, _ = (
                            self._pending_fatal_context
                        )
                        self._pending_fatal_context = (
                            retained_provider,
                            retained_lease,
                            owner,
                        )

                if callable(reserve_outcome):
                    if discovering_token is not None:
                        gate._convertible_token_state(discovering_token)
                    pre_reserve = getattr(
                        collective, "_pre_reserve_fatal_publication", None
                    )
                    if not callable(pre_reserve):
                        raise RuntimeError(
                            "collective cannot pre-reserve fatal publication"
                        )

                    def elect_joinable_fatal():
                        nonlocal elected_here, transition
                        reserved_primary, started = pre_reserve(primary)
                        if reserved_primary is not primary or not started:
                            raise RuntimeError(
                                "communicator fatal owner was not reserved"
                            )
                        transition = gate.begin_fatal(
                            primary, discovering_token
                        )
                        if transition.primary is not primary:
                            raise RuntimeError(
                                "communicator fatal primary changed"
                            )
                        elected_here = True
                        retain_fatal_context()

                    outcome = reserve_outcome(
                        primary, before_select=elect_joinable_fatal
                    )
                    if outcome.primary is not primary:
                        self._remember_terminal_secondary_locked(
                            primary, outcome.primary, owner
                        )
                        primary = outcome.primary
                if not elected_here:
                    transition = gate.begin_fatal(primary, discovering_token)
                if transition.primary is not primary:
                    raise RuntimeError("communicator fatal primary changed")
                retain_fatal_context()
                return transition

    def _publish_communicator_fatal_transition(self, transition):
        if not isinstance(transition, _FatalTransition):
            raise TypeError("communicator fatal transition is invalid")
        primary = transition.primary
        owner = None
        try:
            self._terminal_gate.wait_for_admissions(
                transition, _TERMINAL_TIMEOUT_S
            )
            with self._terminal_state_lock:
                context = self._pending_fatal_context
            provider, lease, owner = (
                (None, None, None) if context is None else context
            )
            if owner is not None:
                if owner.state not in {"detached", "quarantined"}:
                    owner.force_quarantine(primary)
            with self._terminal_gate._condition:
                if self._terminal_gate._fatal_transition is not transition:
                    raise RuntimeError("communicator fatal transition changed")
                with self._terminal_state_lock:
                    primary = self._force_terminal_primary_locked(
                        transition.primary,
                        provider=provider,
                        lease=lease,
                        owner=owner,
                        retain_owner=(
                            owner is not None and owner.state == "quarantined"
                        ),
                    )
        except BaseException as error:
            if owner is not None:
                owner._remember_secondary(error)
            if error is not primary:
                raise primary
            raise
        return primary

    def _enter_communicator_fatal(
        self, primary, owner=None, discovering_token=None
    ):
        if not isinstance(primary, BaseException):
            raise TypeError("communicator fatal failure must be an exception")
        transition = self._begin_communicator_fatal(
            primary,
            owner=owner,
            discovering_token=discovering_token,
        )
        primary = transition.primary
        with self._terminal_gate._condition:
            phase = self._terminal_gate._phase
        if phase in (
            _TerminalPhase.FATAL_PUBLISHED,
            _TerminalPhase.RUNTIME_CLOSED,
        ):
            return primary
        with self._terminal_state_lock:
            context = self._pending_fatal_context
            if context is not None:
                owner = context[2]
            collective = self._pending_fatal_collective
        if collective is None:
            raise RuntimeError("fatal publication collective was not retained")
        publish = getattr(collective, "_publish_communicator_fatal", None)
        if not callable(publish):
            raise RuntimeError("collective does not provide fatal publication")
        if callable(getattr(collective, "_begin_fatal_publication", None)):
            active = getattr(
                collective._fatal_publication_local, "reservation", None
            )
            if active is not None and not isinstance(
                active.handler, _RuntimeCommunicatorFatalHook
            ):
                if active.primary is not transition.primary:
                    raise RuntimeError("communicator fatal primary changed")
                snapshot = self._publish_communicator_fatal_transition(transition)
                active.transition = transition
                active.snapshot = snapshot
                active.completion = lambda: self._terminal_gate.publish_fatal(
                    transition, snapshot
                )
                return snapshot
            hook = self._communicator_fatal_hook
            if hook is None:
                hook = _RuntimeCommunicatorFatalHook(self)
                self._communicator_fatal_hook = hook
            return publish(
                primary,
                fatal_owner=owner,
                discovering_token=discovering_token,
                join_existing=True,
                handler_override=hook,
                transition=transition,
            )

        thread_id = threading.get_ident()
        with self._terminal_state_lock:
            publication_owner = self._legacy_fatal_publication_owner
            owns_publication = publication_owner is None
            if owns_publication:
                self._legacy_fatal_publication_owner = thread_id
        if publication_owner == thread_id:
            return transition.primary
        if not owns_publication:
            return self._terminal_gate.wait_for_published(_TERMINAL_TIMEOUT_S)
        snapshot = self._publish_communicator_fatal_transition(transition)
        publish(snapshot)
        self._terminal_gate.publish_fatal(transition, snapshot)
        return snapshot

    def barrier(self):
        self._require_usable()
        return self.collective.barrier()

    def execution_config(
        self,
        *,
        residency_policy="device_resident",
        device_memory_budget_bytes=None,
        host_memory_budget_bytes=None,
        prefetch_depth=1,
    ):
        self._require_usable()
        backend_metadata = self._synchronize_active_backend()
        device_resolution = self._resolve_budget("device", device_memory_budget_bytes)
        host_resolution = self._resolve_budget("host", host_memory_budget_bytes)
        if residency_policy == "active_working_set":
            config = getattr(self.backend, "config", None)
            if (
                getattr(config, "execution_policy", None) != "execution_ir"
                or getattr(config, "fallback_policy", None) != "error"
            ):
                raise ValueError(
                    "active_working_set requires execution_ir with fallback_policy='error'"
                )
            from renormalizer.backend._distributed.providers import (
                ActiveWorkingSetProvider,
            )

            if self._active_provider is None:
                self._active_provider = ActiveWorkingSetProvider(
                    self,
                    device_budget_resolution=device_resolution,
                    host_budget_resolution=host_resolution,
                    prefetch_depth=prefetch_depth,
                )
            elif not self._active_provider.matches_config(
                device_resolution, host_resolution, prefetch_depth
            ):
                raise ValueError(
                    "active provider already has different frozen budget metadata"
                )
            provider = self._active_provider
        else:
            provider = DeviceResidentProvider()
        return DistributedExecutionConfig(
            context=self.context,
            mesh=self.mesh,
            collective=self.collective,
            provider=provider,
            residency_policy=residency_policy,
            device_memory_budget_bytes=device_memory_budget_bytes,
            host_memory_budget_bytes=host_memory_budget_bytes,
            prefetch_depth=prefetch_depth,
            backend_name=backend_metadata[0] if backend_metadata is not None else None,
            backend_device=backend_metadata[1]
            if backend_metadata is not None
            else None,
            backend_precision=backend_metadata[2]
            if backend_metadata is not None
            else None,
            device_budget_resolution=device_resolution,
            host_budget_resolution=host_resolution,
        )

    def preflight_residency(self, request, plan):
        """Run the fixed Stage 5 agreement schedule without creating resources."""
        self._require_usable()
        local_error = None
        try:
            if not isinstance(request, ResidencyRequest):
                raise TypeError("request must be a ResidencyRequest")
            if not isinstance(plan, ResidencyPlan):
                raise TypeError("plan must be a ResidencyPlan")
            plan.validate_request(request)
            plan.validate_capacity()
            plan.runtime_identity.validate_runtime(
                self.context, self.mesh, self.backend
            )
            config = getattr(self.backend, "config", None)
            if (
                getattr(config, "execution_policy", None) != "execution_ir"
                or getattr(config, "fallback_policy", None) != "error"
            ):
                raise ValueError(
                    "active_working_set requires execution_ir with fallback_policy='error'"
                )
            if (
                self._active_provider is not None
                and not self._active_provider.matches_config(
                    request.device_budget,
                    request.host_budget,
                    request.prefetch_depth,
                )
            ):
                raise ValueError("residency request budgets do not match the runtime")
        except BaseException as error:
            local_error = error

        world_size = self.world_size
        if world_size == 1:
            if local_error is not None:
                raise ValueError(
                    "active residency preflight validation failed"
                ) from local_error
            device_budget_hash = budget_resolution_hash(request.device_budget)
            host_budget_hash = budget_resolution_hash(request.host_budget)
            receipt = ResidencyPreflightReceipt.create(
                runtime_id=self._runtime_id,
                rank=self.rank,
                local_device=str(self.backend.current_device()),
                request_hash=request.request_hash,
                plan_hash=plan.plan_hash,
                device_budget_hash=device_budget_hash,
                host_budget_hash=host_budget_hash,
            )
            self._issued_receipts.clear()
            self._issued_receipts[id(receipt)] = receipt
            return receipt

        status = self._control_array([int(local_error is not None)], np.int32)
        failed = int(
            self._host_control(self.collective.allreduce(status, op="max")).reshape(-1)[
                0
            ]
        )
        del status

        policy = self._control_array(
            [
                1,
                int(
                    getattr(
                        getattr(self.backend, "config", None), "execution_policy", None
                    )
                    == "execution_ir"
                ),
                int(
                    getattr(
                        getattr(self.backend, "config", None), "fallback_policy", None
                    )
                    == "error"
                ),
                world_size,
                self.context.local_world_size,
            ],
            np.int32,
        )
        policy_minimum = self._host_control(self.collective.allreduce(policy, op="min"))
        policy_maximum = self._host_control(self.collective.allreduce(policy, op="max"))
        policy_disagreement = not np.array_equal(policy_minimum, policy_maximum)
        del policy, policy_minimum, policy_maximum

        hashes = (
            request.request_hash if isinstance(request, ResidencyRequest) else "0" * 64,
            plan.plan_hash if isinstance(plan, ResidencyPlan) else "0" * 64,
            (
                budget_resolution_hash(request.device_budget)
                if isinstance(request, ResidencyRequest)
                else "0" * 64
            ),
            (
                budget_resolution_hash(request.host_budget)
                if isinstance(request, ResidencyRequest)
                else "0" * 64
            ),
        )
        hash_control = self._control_array(
            [
                int(value[index : index + 16], 16)
                for value in hashes
                for index in range(0, 64, 16)
            ],
            np.uint64,
        )
        hash_minimum = self._host_control(
            self.collective.allreduce(hash_control, op="min")
        )
        hash_maximum = self._host_control(
            self.collective.allreduce(hash_control, op="max")
        )
        local_hashes = self._host_control(hash_control).reshape(4, 4)
        minimum_hashes = np.asarray(hash_minimum).reshape(4, 4)
        maximum_hashes = np.asarray(hash_maximum).reshape(4, 4)
        hash_disagreements = tuple(
            not (
                np.array_equal(minimum_hashes[index], local_hashes[index])
                and np.array_equal(maximum_hashes[index], local_hashes[index])
            )
            for index in range(4)
        )
        del hash_control, hash_minimum, hash_maximum
        del local_hashes, minimum_hashes, maximum_hashes

        if isinstance(plan, ResidencyPlan):
            requirement_values = (
                *plan.device_peak_bytes,
                *plan.host_peak_bytes,
                plan.host_required_bytes,
                plan.device_budget.resolved_bytes,
                plan.host_budget.resolved_bytes,
            )
        else:
            requirement_values = (0,) * (2 * world_size + 3)
        requirements = self._control_array(requirement_values, np.int64)
        requirement_minimum = self._host_control(
            self.collective.allreduce(requirements, op="min")
        )
        requirement_maximum = self._host_control(
            self.collective.allreduce(requirements, op="max")
        )
        local_requirements = self._host_control(requirements)
        requirement_count = 2 * world_size + 1
        requirement_disagreement = not (
            np.array_equal(
                np.asarray(requirement_minimum)[:requirement_count],
                local_requirements[:requirement_count],
            )
            and np.array_equal(
                np.asarray(requirement_maximum)[:requirement_count],
                local_requirements[:requirement_count],
            )
        )
        budget_disagreement = not (
            np.array_equal(
                np.asarray(requirement_minimum)[requirement_count:],
                local_requirements[requirement_count:],
            )
            and np.array_equal(
                np.asarray(requirement_maximum)[requirement_count:],
                local_requirements[requirement_count:],
            )
        )
        del requirements, requirement_minimum, requirement_maximum, local_requirements

        capacity_code = 0
        if isinstance(plan, ResidencyPlan):
            if (
                plan.backend_name == "cupy"
                and plan.device_peak_bytes[self.rank]
                > plan.device_budget.resolved_bytes
            ):
                capacity_code = 1
            elif plan.host_required_bytes > plan.host_budget.resolved_bytes:
                capacity_code = 2
        capacity = self._control_array([capacity_code], np.int32)
        capacity_failed = int(
            self._host_control(self.collective.allreduce(capacity, op="max")).reshape(
                -1
            )[0]
        )
        del capacity

        if policy_disagreement:
            raise ValueError("active residency policy disagreement")
        labels = ("request hash", "plan hash", "device budget", "host budget")
        for index, label in enumerate(labels):
            if hash_disagreements[index]:
                raise ValueError("{} disagreement".format(label))
        if requirement_disagreement:
            raise ValueError("requirement disagreement")
        if budget_disagreement:
            raise ValueError("budget disagreement")
        if failed:
            raise ValueError(
                "active residency preflight validation failed"
            ) from local_error
        if capacity_failed:
            raise ValueError("active residency capacity preflight failed")

        receipt = ResidencyPreflightReceipt.create(
            runtime_id=self._runtime_id,
            rank=self.rank,
            local_device=str(self.backend.current_device()),
            request_hash=request.request_hash,
            plan_hash=plan.plan_hash,
            device_budget_hash=hashes[2],
            host_budget_hash=hashes[3],
        )
        self._issued_receipts.clear()
        self._issued_receipts[id(receipt)] = receipt
        return receipt

    def consume_residency_receipt(self, receipt, request, plan):
        self._require_usable()
        if not isinstance(receipt, ResidencyPreflightReceipt):
            raise TypeError("receipt must be a ResidencyPreflightReceipt")
        issued = self._issued_receipts.pop(id(receipt), None)
        if issued is not receipt:
            raise ValueError(
                "residency preflight receipt was not issued by this runtime"
            )
        expected = (
            self._runtime_id,
            self.rank,
            str(self.backend.current_device()),
            request.request_hash,
            plan.plan_hash,
            budget_resolution_hash(request.device_budget),
            budget_resolution_hash(request.host_budget),
        )
        actual = (
            receipt.runtime_id,
            receipt.rank,
            receipt.local_device,
            receipt.request_hash,
            receipt.plan_hash,
            receipt.device_budget_hash,
            receipt.host_budget_hash,
        )
        if actual != expected:
            raise ValueError(
                "residency preflight receipt does not match the runtime tuple"
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
            if requested is not None and (type(requested) is not int or requested <= 0):
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
            self._host_control(self.collective.allreduce(status, op="max")).reshape(-1)[
                0
            ]
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
            raise RuntimeError("{} memory budget request disagreement".format(resource))

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
                self._host_control(self.collective.allreduce(status, op="max")).reshape(
                    -1
                )[0]
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

        status = self.backend.asarray([int(local_error is not None)], dtype=np.int32)
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
            [int(hexdigest[index : index + 16], 16) for index in range(0, 64, 16)],
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

    def _clear_runtime_references(
        self,
        error,
        *,
        publish_error=True,
        mark_closed=True,
        clear_collective=True,
    ):
        with self._terminal_gate._condition:
            with self._terminal_state_lock:
                transition = self._terminal_gate._fatal_transition
                fatal_is_public = (
                    transition is not None
                    and self._terminal_gate._phase
                    in (
                        _TerminalPhase.FATAL_PUBLISHED,
                        _TerminalPhase.RUNTIME_CLOSED,
                    )
                )
                if fatal_is_public:
                    error = transition.primary
                elif publish_error and isinstance(error, BaseException):
                    error, provider, lease, owner = (
                        self._normalize_communicator_fatal(error, None)
                    )
                    error = self._force_terminal_primary_locked(
                        error,
                        provider=provider,
                        lease=lease,
                        owner=owner,
                    )
                self._issued_receipts.clear()
                self._active_provider = None
                if clear_collective:
                    self.collective = None
                if mark_closed:
                    self._closed = True
        return error

    def _release_runtime_close_step(self, token):
        if token is None:
            return
        try:
            self._terminal_gate.release(token)
        except RuntimeError as error:
            if "converted" not in str(error):
                raise

    def _commit_fatal_runtime_close(self, transition, error):
        result = self._terminal_gate.commit_runtime_close(
            transition,
            lambda: self._clear_runtime_references(
                error,
                publish_error=False,
                mark_closed=False,
                clear_collective=False,
            ),
        )
        if not self._closed:
            self._clear_runtime_references(error)
        return result

    def close(self):
        if self._closed:
            return
        error = self._terminal_error
        if error is None:
            error = getattr(self.backend, "_execution_terminal_error", None)
        request = None
        try:
            request = self._terminal_gate.admit_runtime("begin_runtime_close")
        except RuntimeError:
            transition = self._terminal_gate.begin_runtime_close(None)
        else:
            transition = self._terminal_gate.begin_runtime_close(request)
        if isinstance(transition, _FatalTransition):
            error = transition.primary
            self._commit_fatal_runtime_close(transition, error)
            raise error
        if transition.owner_thread_id != threading.get_ident():
            result = self._terminal_gate.commit_runtime_close(
                transition, lambda: None
            )
            if isinstance(result, BaseException):
                raise result
            return result

        provider = self._active_provider
        collective = self.collective
        if provider is not None:
            provider_token = self._terminal_gate.admit_runtime_close(
                transition, "provider_close"
            )
            try:
                provider.close()
            except BaseException as caught:
                if error is None:
                    error = caught
            finally:
                self._release_runtime_close_step(provider_token)
        try:
            if collective is not None and self._terminal_gate.phase not in (
                _TerminalPhase.FATAL_PENDING,
                _TerminalPhase.FATAL_PUBLISHED,
            ):
                close_for_runtime = getattr(collective, "_close_for_runtime", None)
                if callable(close_for_runtime):
                    close_for_runtime(self._terminal_gate, transition)
                else:
                    collective_token = self._terminal_gate.admit_runtime_close(
                        transition, "collective_close"
                    )
                    try:
                        collective.close()
                    finally:
                        self._release_runtime_close_step(collective_token)
        except BaseException as caught:
            if error is None:
                error = caught

        if self._terminal_gate.phase in (
            _TerminalPhase.FATAL_PENDING,
            _TerminalPhase.FATAL_PUBLISHED,
        ):
            fatal_transition = self._terminal_gate.begin_runtime_close(None)
            error = fatal_transition.primary
            self._commit_fatal_runtime_close(fatal_transition, error)
            raise error
        if self._terminal_error is not None:
            error = self._terminal_error
        result = self._terminal_gate.commit_runtime_close(
            transition, lambda: self._clear_runtime_references(error)
        )
        if error is not None:
            raise error
        return result

    def resource_state(self):
        if self._active_provider is None:
            state = {
                "active_leases": 0,
                "cache_bytes": 0,
                "cache_refs": 0,
                "pinned_bytes": 0,
                "stream_count": 0,
                "event_count": 0,
            }
        else:
            state = self._active_provider.runtime_resource_state()
        if self._terminal_quarantine.poisoned:
            retained = self._terminal_quarantine.retained_resource_state()
            for key in ("cache_bytes", "pinned_bytes", "stream_count", "event_count"):
                state[key] = max(state[key], retained[key])
            state.update(self._terminal_quarantine.resource_state())
        return state

    def __enter__(self):
        self._require_usable()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is None:
            self.close()
        else:
            try:
                self.close()
            except BaseException:
                pass
        return False


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
    execution_policy="legacy_oe",
    fallback_policy="error",
    experimental_oe_ir=False,
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
            device="cuda:{}".format(context.local_rank),
            precision=precision,
            execution_policy=execution_policy,
            fallback_policy=fallback_policy,
            experimental_oe_ir=experimental_oe_ir,
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
    except BaseException:
        try:
            runtime.close()
        except BaseException:
            pass
        raise
    else:
        runtime.close()


@contextmanager
def active_working_set_execution(distributed_execution, request, plan, store):
    """Borrow an active config with one receipt-bound outer working-set lease."""
    if not isinstance(distributed_execution, DistributedExecutionConfig):
        raise TypeError("distributed_execution must be a DistributedExecutionConfig")
    if distributed_execution.residency_policy != "active_working_set":
        raise ValueError(
            "active working-set execution requires active residency policy"
        )
    factory = distributed_execution.provider
    if getattr(factory, "provider_role", None) != "factory":
        raise ValueError("active working-set execution requires a factory provider")
    runtime = getattr(factory, "runtime", None)
    if runtime is None or runtime._closed:
        raise RuntimeError("active provider runtime is unavailable")
    runtime._require_usable()
    receipt = runtime.preflight_residency(request, plan)
    working_set = factory.open_working_set(request, plan, store, receipt)
    try:
        yield replace(
            distributed_execution,
            provider=working_set,
            residency_request=request,
            residency_plan=plan,
            residency_receipt=receipt,
        )
    except BaseException:
        try:
            working_set.close()
        except BaseException:
            pass
        raise
    else:
        working_set.close()
