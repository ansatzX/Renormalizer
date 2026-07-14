# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""Explicit lifecycle for launcher-configured CuPy distributed execution."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
import hashlib
import inspect
import json
import math
import os
import threading
import time
import uuid
import weakref

import numpy as np

from renormalizer.backend._distributed.async_owner import (
    RuntimeTerminalQuarantine,
    _require_resource_admission,
)
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
    _MISSING as _TERMINAL_MISSING,
    _MonitorOutcome,
    _TerminalLifecycleGate,
    _TerminalPhase,
    _TERMINAL_TIMEOUT_S,
)
from renormalizer.backend.config import BackendConfig, DistributedExecutionConfig
from renormalizer.backend.factory import create_backend


_ADMISSION_UNSET = object()


def _call_admitted_private(
    callback,
    *args,
    _admission_token,
    _admission_validator,
    **kwargs,
):
    _require_resource_admission(
        _admission_token,
        _admission_validator,
    )
    parameters = inspect.signature(callback).parameters.values()
    accepts_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    names = {parameter.name for parameter in parameters}
    if accepts_keywords or "_admission_token" in names:
        kwargs["_admission_token"] = _admission_token
    if accepts_keywords or "_admission_validator" in names:
        kwargs["_admission_validator"] = _admission_validator
    return callback(*args, **kwargs)


class _ExactAdmissionValidator:
    """Revalidate one canonical admission immediately before resource access."""

    def __init__(
        self,
        runtime,
        token,
        *,
        scope,
        epoch,
        parent_sequence,
        transition_sequence,
    ):
        self._runtime = runtime
        self._sequence = token.sequence
        self._scope = scope
        self._epoch = epoch
        self._parent_sequence = parent_sequence
        self._transition_sequence = transition_sequence

    def __call__(self, token):
        return self._runtime._require_admission(
            token,
            scope=self._scope,
            epoch=self._epoch,
            sequence=self._sequence,
            parent_sequence=self._parent_sequence,
            transition_sequence=self._transition_sequence,
        )


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
        runtime = self._runtime()
        runtime._terminal_gate.publish_fatal(transition, snapshot)

    def secondaries(self, transition):
        return self._runtime()._structured_terminal_secondaries(transition)

    def fail(self, transition, failure):
        self._runtime()._terminal_gate._fail_fatal_publication(
            transition, failure
        )

    def failure_recorded(self, transition, failure):
        return self._runtime()._terminal_gate._fatal_publication_failed(
            transition, failure
        )

    def record_secondary(self, error):
        self._runtime()._record_structured_fatal_secondary(error)

    def _terminalize_failure(self, primary, diagnostics=(), transition=None):
        return self._runtime()._terminalize_communicator_fatal_failure(
            primary,
            diagnostics=diagnostics,
            transition=transition,
        )


_MISSING_TERMINAL_FIELD = object()


class _DeferredAsyncQuarantine:
    __slots__ = (
        "owner",
        "callback",
        "deferred_callback",
        "triggered",
        "completed",
    )

    def __init__(self, owner, callback):
        self.owner = owner
        self.callback = callback
        self.deferred_callback = None
        self.triggered = False
        self.completed = False


class _RecoverableFatalElection:
    __slots__ = (
        "runtime",
        "gate",
        "collective",
        "primary",
        "provider",
        "lease",
        "owner",
        "discovering_token",
        "stage",
        "prepared_transition",
        "prepared_outcome",
        "owner_reservation",
        "transition",
        "context_retained",
        "context_repair_attempted",
        "outcome",
        "recovery_errors",
        "recovery_complete",
    )

    _MAX_RECOVERY_ERRORS = 4

    def __init__(
        self,
        runtime,
        collective,
        primary,
        provider,
        lease,
        owner,
        discovering_token,
    ):
        self.runtime = runtime
        self.gate = runtime._terminal_gate
        self.collective = collective
        self.primary = primary
        self.provider = provider
        self.lease = lease
        self.owner = owner
        self.discovering_token = discovering_token
        self.stage = "created"
        self.prepared_transition = None
        self.prepared_outcome = None
        self.owner_reservation = None
        self.transition = None
        self.context_retained = False
        self.context_repair_attempted = False
        self.outcome = None
        self.recovery_errors = ()
        self.recovery_complete = False

    def _remember_recovery_error(self, error):
        if any(retained is error for retained in self.recovery_errors):
            return
        if len(self.recovery_errors) >= self._MAX_RECOVERY_ERRORS:
            return
        self.recovery_errors = (*self.recovery_errors, error)

    def _prepare_artifacts_locked(self):
        if self.prepared_transition is None:
            transition = self.gate._fatal_transition
            if transition is None:
                transition = _FatalTransition(
                    gate_id=self.gate._gate_id,
                    primary=self.primary,
                    sequence=self.gate._sequence(),
                )
            elif transition.primary is not self.primary:
                raise RuntimeError("communicator fatal transition changed")
            self.prepared_transition = transition
        if self.prepared_outcome is None:
            self.prepared_outcome = _MonitorOutcome(
                kind="fatal_elected", primary=self.primary
            )
        self.stage = "artifacts_prepared"

    def _retain_context_locked(self):
        runtime = self.runtime
        if runtime._pending_fatal_collective is None:
            runtime._pending_fatal_collective = self.collective
        if runtime._pending_fatal_context is None:
            runtime._pending_fatal_context = (
                self.provider,
                self.lease,
                self.owner,
            )
        elif (
            self.owner is not None
            and runtime._pending_fatal_context[2] is None
        ):
            retained_provider, retained_lease, _ = runtime._pending_fatal_context
            runtime._pending_fatal_context = (
                retained_provider,
                retained_lease,
                self.owner,
            )
        runtime._stage_active_broadcast_quarantine_locked(
            self.owner, self.collective
        )
        self.context_retained = True
        self.stage = "context_retained"

    def _preallocate_owner_locked(self):
        if self.owner_reservation is not None:
            return self.owner_reservation
        prepare_owner = getattr(
            self.collective,
            "_prepare_runtime_fatal_owner_reservation",
            None,
        )
        if not callable(prepare_owner):
            raise RuntimeError(
                "collective fatal owner reservation is not preparable"
            )
        self.owner_reservation = prepare_owner(self.primary, self)
        self.stage = "owner_preallocated"
        return self.owner_reservation

    def _repair_owner_locked(self):
        owner_reservation = self.owner_reservation
        if owner_reservation is None:
            raise RuntimeError("communicator fatal owner was not preallocated")
        collective = self.collective
        with collective._fatal_condition:
            retained = collective._fatal_publication_owner_reservation
            publications = collective._fatal_publications
            pending = collective._fatal_pending_primary
            if (
                owner_reservation.owner_thread
                is not threading.current_thread()
                or owner_reservation.primary is not self.primary
                or owner_reservation.election is not self
                or owner_reservation.adopted
            ):
                raise RuntimeError(
                    "communicator fatal owner reservation changed"
                )
            if retained is None:
                if publications != 0:
                    raise RuntimeError(
                        "communicator fatal owner reservation is inconsistent"
                    )
                if collective._fatal_protocol_completed or collective._closed:
                    raise RuntimeError(
                        "communicator fatal owner is no longer installable"
                    )
            elif retained is owner_reservation:
                if publications not in (0, 1):
                    raise RuntimeError(
                        "communicator fatal owner reservation is inconsistent"
                    )
            else:
                raise RuntimeError(
                    "communicator fatal owner reservation is inconsistent"
                )
            if pending is None:
                collective._fatal_pending_primary = self.primary
                collective._fatal_pending_origin_rank = None
                owner_reservation.installed = True
            elif pending is not self.primary:
                raise RuntimeError("communicator fatal owner primary changed")
            elif owner_reservation.installed is None:
                owner_reservation.installed = False
            if retained is None and publications == 0:
                collective._fatal_publication_owner_reservation = (
                    owner_reservation
                )
                collective._fatal_publications = 1
            elif retained is owner_reservation and publications == 0:
                collective._fatal_publications = 1
            elif retained is not owner_reservation or publications != 1:
                raise RuntimeError(
                    "communicator fatal owner reservation is inconsistent"
                )
            collective._fatal_condition.notify_all()
        self.stage = "owner_reserved"
        return owner_reservation

    def _prepare_locked(self):
        self._prepare_artifacts_locked()
        pre_reserve = getattr(
            self.collective, "_pre_reserve_fatal_publication", None
        )
        reserved_primary_for_thread = getattr(
            self.collective,
            "_current_thread_reserved_fatal_primary",
            None,
        )
        reserved_election_for_thread = getattr(
            self.collective,
            "_current_thread_reserved_fatal_election",
            None,
        )
        if (
            not callable(pre_reserve)
            or not callable(reserved_primary_for_thread)
            or not callable(reserved_election_for_thread)
        ):
            raise RuntimeError(
                "collective fatal owner reservation is not recoverable"
            )
        try:
            reserved_primary, _ = pre_reserve(self.primary, election=self)
            self.stage = "owner_reserved"
            owner_election = reserved_election_for_thread()
            owner_primary = reserved_primary_for_thread()
            self.stage = "owner_confirmed"
            if (
                reserved_primary is not self.primary
                or owner_primary is not self.primary
                or owner_election is not self
            ):
                raise RuntimeError("communicator fatal owner was not reserved")
            self.transition = self.gate._recover_fatal(
                self.primary,
                self.discovering_token,
                prepared_transition=self.prepared_transition,
            )
            self.stage = "token_converted"
            self._retain_context_locked()
        except BaseException:
            try:
                self._repair_owner_locked()
            except BaseException as repair_error:
                self._remember_recovery_error(repair_error)
            try:
                self._repair_gate_locked()
            except BaseException as repair_error:
                self._remember_recovery_error(repair_error)
            self._repair_context_locked()
            raise

    def _select_outcome_locked(self):
        self._prepare_artifacts_locked()
        self._preallocate_owner_locked()
        reserve_outcome = getattr(
            self.collective, "_reserve_runtime_fatal_outcome", None
        )
        if not callable(reserve_outcome):
            raise RuntimeError("collective fatal outcome is not reservable")
        outcome = reserve_outcome(
            self.primary,
            before_select=self._prepare_locked,
        )
        self.outcome = outcome
        self.stage = "handoff_selected"
        return outcome

    def _retained_owner_locked(self):
        with self.collective._fatal_condition:
            retained = self.collective._fatal_publication_owner_reservation
            if (
                self.collective._fatal_publications <= 0
                or retained is None
                or retained.owner_thread is not threading.current_thread()
                or retained.primary is not self.primary
                or retained.election is not self
            ):
                return None
            return retained

    def _repair_gate_locked(self):
        self._prepare_artifacts_locked()
        gate = self.gate
        transition = gate._fatal_transition
        if transition is None:
            transition = self.prepared_transition
            gate._fatal_transition = transition
        if transition.primary is not self.primary:
            raise RuntimeError("communicator fatal transition recovery changed")
        if gate._phase in (
            _TerminalPhase.HEALTHY,
            _TerminalPhase.RUNTIME_CLOSING,
        ):
            gate._phase = _TerminalPhase.FATAL_PENDING
            if gate._live_epoch is not None:
                gate._leases[gate._live_epoch].phase = "fatal_retained"
        token = self.discovering_token
        if token is not None:
            if token.gate_id != gate._gate_id:
                raise RuntimeError(
                    "resource admission belongs to another terminal gate"
                )
            token_state = gate._tokens.get(token.sequence)
            if token_state is None or token_state.token != token:
                raise RuntimeError("resource admission token is unknown or stale")
            if token_state.status == "released":
                raise RuntimeError(
                    "resource admission token was already released"
                )
            if token.thread_id != threading.get_ident():
                raise RuntimeError(
                    "resource admission belongs to another thread"
                )
            if token_state.status == "active":
                if (
                    token_state.async_capability is not _TERMINAL_MISSING
                    and not token_state.async_claimed
                ):
                    raise RuntimeError(
                        "unclaimed async admission cannot be converted"
                    )
                token_state.status = "converted"
                token_state.async_capability = _TERMINAL_MISSING
                gate._thread_tokens.pop(token.thread_id, None)
            elif token_state.status != "converted":
                raise RuntimeError("resource admission state is invalid")
        gate._condition.notify_all()
        self.transition = transition
        self.stage = "token_converted"

    def _repair_context_locked(self):
        runtime = self.runtime
        if runtime._pending_fatal_collective is None:
            runtime._pending_fatal_collective = self.collective
        if runtime._pending_fatal_context is None:
            runtime._pending_fatal_context = (
                self.provider,
                self.lease,
                self.owner,
            )
        elif self.owner is not None and runtime._pending_fatal_context[2] is None:
            provider, lease, _ = runtime._pending_fatal_context
            runtime._pending_fatal_context = (provider, lease, self.owner)
        if not self.context_retained and not self.context_repair_attempted:
            self.context_repair_attempted = True
            try:
                runtime._stage_active_broadcast_quarantine_locked(
                    self.owner, self.collective
                )
            except BaseException as error:
                self._remember_recovery_error(error)
        self.context_retained = True
        self.stage = "context_retained"

    def _repair_handoff_locked(self):
        handoff = self.collective._fatal_monitor_handoff
        outcome = handoff._outcome
        if outcome is None:
            outcome = self.prepared_outcome
            handoff._outcome = outcome
            handoff._condition.notify_all()
        if outcome.kind != "fatal_elected" or outcome.primary is not self.primary:
            raise RuntimeError("communicator fatal handoff recovery changed")
        self.outcome = outcome
        self.stage = "handoff_selected"

    def _recovery_state_complete_locked(self):
        transition = self.gate._fatal_transition
        if transition is None or transition.primary is not self.primary:
            return False
        if self.discovering_token is not None:
            token_state = self.gate._tokens.get(self.discovering_token.sequence)
            if token_state is None or token_state.status != "converted":
                return False
        handoff = self.collective._fatal_monitor_handoff
        outcome = handoff._outcome
        return (
            self.runtime._pending_fatal_collective is self.collective
            and self.runtime._pending_fatal_context is not None
            and outcome is not None
            and outcome.kind == "fatal_elected"
            and outcome.primary is self.primary
        )

    def recover_locked(self):
        handoff = self.collective._fatal_monitor_handoff
        with handoff._condition:
            if self._retained_owner_locked() is None:
                try:
                    self._repair_owner_locked()
                except BaseException as error:
                    self._remember_recovery_error(error)
                if self._retained_owner_locked() is None:
                    return False
            if not self.recovery_complete:
                try:
                    self._repair_gate_locked()
                except BaseException as error:
                    self._remember_recovery_error(error)
                self._repair_context_locked()
                try:
                    self._repair_handoff_locked()
                except BaseException as error:
                    self._remember_recovery_error(error)
                self.recovery_complete = self._recovery_state_complete_locked()
            return self.recovery_complete

    def recover(self):
        with self.gate._condition:
            with self.runtime._terminal_state_lock:
                return self.recover_locked()


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
    _terminal_pending_owner_diagnostics: tuple = field(
        default_factory=tuple, init=False, repr=False
    )
    _terminal_pending_direct_diagnostics: tuple = field(
        default_factory=tuple, init=False, repr=False
    )
    _terminal_pending_collective_diagnostics: tuple = field(
        default_factory=tuple, init=False, repr=False
    )
    _terminal_owner_diagnostic_dispatches: tuple = field(
        default_factory=tuple, init=False, repr=False
    )
    _terminal_collective_diagnostic_dispatches: tuple = field(
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
    _deferred_async_quarantines: list = field(
        default_factory=list, init=False, repr=False
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

    def _require_admission(
        self,
        token,
        *,
        scope,
        epoch,
        sequence=_ADMISSION_UNSET,
        parent_sequence=_ADMISSION_UNSET,
        transition_sequence=None,
    ):
        gate = self._terminal_gate
        with gate._condition:
            state = gate._token_state(token)
            gate._require_token_thread(state)
            if token.scope != scope or token.epoch != epoch:
                raise RuntimeError(
                    "resource admission does not match the required runtime scope"
                )
            if sequence is not _ADMISSION_UNSET and token.sequence != sequence:
                raise RuntimeError(
                    "resource admission does not match the required sequence"
                )
            if (
                parent_sequence is not _ADMISSION_UNSET
                and token.parent_sequence != parent_sequence
            ):
                raise RuntimeError(
                    "resource admission does not match the required parent"
                )
            if token.transition_sequence != transition_sequence:
                raise RuntimeError(
                    "resource admission does not match the required transition"
                )
            return token

    def _exact_admission_validator(
        self,
        token,
        *,
        scope,
        epoch,
        parent_sequence=None,
        transition_sequence=None,
    ):
        self._require_admission(
            token,
            scope=scope,
            epoch=epoch,
            sequence=token.sequence,
            parent_sequence=parent_sequence,
            transition_sequence=transition_sequence,
        )
        return _ExactAdmissionValidator(
            self,
            token,
            scope=scope,
            epoch=epoch,
            parent_sequence=parent_sequence,
            transition_sequence=transition_sequence,
        )

    def _resolve_admission_validator(
        self,
        token,
        validator,
        *,
        scope,
        epoch,
        parent_sequence=None,
        transition_sequence=None,
    ):
        if validator is None:
            validator = self._exact_admission_validator(
                token,
                scope=scope,
                epoch=epoch,
                parent_sequence=parent_sequence,
                transition_sequence=transition_sequence,
            )
        _require_resource_admission(token, validator)
        return validator

    def _release_admission(self, token):
        if token is None:
            return
        try:
            self._terminal_gate.release(token)
        except RuntimeError as error:
            if "converted" not in str(error):
                raise

    def _install_active_provider(
        self,
        provider,
        *,
        _admission_token,
        _admission_validator=None,
    ):
        self._resolve_admission_validator(
            _admission_token,
            _admission_validator,
            scope="runtime_setup",
            epoch=None,
        )
        existing = self._active_provider
        if existing is not None and existing is not provider:
            raise RuntimeError("runtime already owns an active working-set provider")
        self._active_provider = provider
        return provider

    def _publish_residency_receipt(
        self,
        receipt,
        *,
        _admission_token,
        _admission_validator=None,
    ):
        self._resolve_admission_validator(
            _admission_token,
            _admission_validator,
            scope="runtime_setup",
            epoch=None,
        )
        self._issued_receipts.clear()
        self._issued_receipts[id(receipt)] = receipt
        return receipt

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

    def _fatal_collective(self):
        collective = self._pending_fatal_collective
        return self.collective if collective is None else collective

    def _defer_current_active_broadcast_fatal(
        self, primary, *, owner=None, discovering_token=None
    ):
        collective = self._fatal_collective()
        defer = (
            None
            if collective is None
            else getattr(
                collective, "_defer_current_active_broadcast_fatal", None
            )
        )
        if not callable(defer):
            return None
        return defer(
            primary,
            origin_rank=self.rank,
            fatal_owner=owner,
            discovering_token=discovering_token,
        )

    def _current_fatal_publication_is_deferred(self):
        collective = self._fatal_collective()
        probe = (
            None
            if collective is None
            else getattr(
                collective, "_current_fatal_publication_is_deferred", None
            )
        )
        return bool(callable(probe) and probe())

    def _retain_deferred_terminal_error(self, error, owner=None):
        gate = self._terminal_gate
        with gate._condition:
            transition = gate._fatal_transition
            if transition is None:
                raise RuntimeError("deferred communicator fatal is not elected")
            with self._terminal_state_lock:
                primary, _, _, owner = self._normalize_communicator_fatal(
                    error, owner
                )
                self._force_async_primary_locked(owner, primary)
                self._terminal_quarantine.first_error = primary
                self._terminal_quarantine.retain_error(primary)
                if owner is not None:
                    self._terminal_quarantine.retain(owner, primary)
                return primary

    def _stage_active_broadcast_quarantine_locked(self, owner, collective):
        if owner is None:
            return
        active = getattr(
            collective, "_current_thread_has_active_broadcast", None
        )
        if not callable(active) or not active():
            return
        record = next(
            (
                retained
                for retained in self._deferred_async_quarantines
                if retained.owner is owner
            ),
            None,
        )
        if record is None:
            callback = getattr(owner, "_quarantine", None)
            if not callable(callback):
                return
            record = _DeferredAsyncQuarantine(owner, callback)
            self._deferred_async_quarantines.append(record)
        if record.deferred_callback is None:
            runtime_ref = weakref.ref(self)

            def defer_quarantine(retained_owner):
                runtime = runtime_ref()
                if runtime is None:
                    return record.callback(retained_owner)
                return runtime._accept_deferred_async_quarantine(
                    record, retained_owner
                )

            record.deferred_callback = defer_quarantine
        owner._quarantine = record.deferred_callback

    def _accept_deferred_async_quarantine(self, record, owner):
        if record.owner is not owner:
            raise RuntimeError("deferred async quarantine owner changed")
        primary = self._retain_deferred_terminal_error(owner.error, owner)
        with self._terminal_state_lock:
            record.triggered = True
        return primary

    def _publish_deferred_async_quarantines(self):
        with self._terminal_state_lock:
            records = tuple(
                record
                for record in self._deferred_async_quarantines
                if record.triggered and not record.completed
            )
        for record in records:
            record.callback(record.owner)
            with self._terminal_state_lock:
                record.completed = True

    @staticmethod
    def _terminal_field_snapshot(owner, name):
        if owner is None:
            return _MISSING_TERMINAL_FIELD
        return getattr(owner, name, _MISSING_TERMINAL_FIELD)

    @staticmethod
    def _restore_terminal_field(owner, name, value):
        if owner is None:
            return
        if value is _MISSING_TERMINAL_FIELD:
            if hasattr(owner, name):
                delattr(owner, name)
            return
        setattr(owner, name, value)

    def _begin_deferred_terminal_public_state(self, reservation):
        if reservation is None or not reservation.completion_deferred:
            return None
        gate = self._terminal_gate
        with gate._condition:
            if gate._phase is not _TerminalPhase.FATAL_PENDING:
                return None
            with self._terminal_state_lock:
                provider = self._active_provider
                lease = (
                    None
                    if provider is None
                    else getattr(provider, "_active_lease", None)
                )
                backend = self.backend
                snapshot = {
                    "runtime_error": self._terminal_error,
                    "backend": backend,
                    "backend_error": self._terminal_field_snapshot(
                        backend, "_execution_terminal_error"
                    ),
                    "provider": provider,
                    "provider_error": self._terminal_field_snapshot(
                        provider, "_terminal_error"
                    ),
                    "lease": lease,
                    "lease_error": self._terminal_field_snapshot(
                        lease, "_poisoned_error"
                    ),
                    "overrides": [],
                }

                def retain_only(error):
                    return self._retain_deferred_terminal_error(error)

                for target, name in (
                    (lease, "_poison"),
                    (provider, "_poison_terminal"),
                ):
                    namespace = getattr(target, "__dict__", None)
                    if namespace is None:
                        continue
                    previous = namespace.get(name, _MISSING_TERMINAL_FIELD)
                    setattr(target, name, retain_only)
                    snapshot["overrides"].append((target, name, previous))
                return snapshot

    def _restore_deferred_terminal_public_state(self, reservation, snapshot):
        if snapshot is None:
            return
        gate = self._terminal_gate
        with gate._condition:
            with self._terminal_state_lock:
                for target, name, previous in reversed(snapshot["overrides"]):
                    self._restore_terminal_field(target, name, previous)
                if (
                    not reservation.completion_deferred
                    or gate._phase is not _TerminalPhase.FATAL_PENDING
                ):
                    return
                self._terminal_error = snapshot["runtime_error"]
                self._restore_terminal_field(
                    snapshot["backend"],
                    "_execution_terminal_error",
                    snapshot["backend_error"],
                )
                self._restore_terminal_field(
                    snapshot["provider"],
                    "_terminal_error",
                    snapshot["provider_error"],
                )
                self._restore_terminal_field(
                    snapshot["lease"],
                    "_poisoned_error",
                    snapshot["lease_error"],
                )

    def _accept_async_quarantine(self, owner, primary=None):
        error = owner.error if primary is None else primary
        if not isinstance(error, BaseException):
            raise TypeError("async quarantine failure must be an exception")
        if self._current_fatal_publication_is_deferred():
            self._retain_deferred_terminal_error(error, owner)
            return
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
        deferred_primary = self._defer_current_active_broadcast_fatal(primary)
        if deferred_primary is not None:
            primary = deferred_primary
        collective = self._fatal_collective()
        reserve = (
            None
            if collective is None
            else getattr(collective, "_communicator_fatal_reservation", None)
        )
        boundary = (
            reserve(primary, join_existing=True)
            if callable(reserve)
            else nullcontext(None)
        )
        with boundary as reservation:
            if reservation is not None:
                primary = reservation.primary
            snapshot = self._begin_deferred_terminal_public_state(reservation)
            try:
                yield primary, reservation
            finally:
                self._restore_deferred_terminal_public_state(
                    reservation, snapshot
                )

    def _retain_terminal_secondary_locked(
        self,
        error,
        primary,
        owner=None,
        *,
        record_collective=True,
    ):
        if not isinstance(error, BaseException):
            return False
        if error is primary:
            return False
        retained = any(
            retained is error for retained in self._terminal_secondary_errors
        )
        if not retained:
            self._terminal_secondary_errors = (
                *self._terminal_secondary_errors,
                error,
            )
        if owner is not None and not any(
            retained_owner is owner and retained_error is error
            for retained_owner, retained_error in (
                self._terminal_pending_owner_diagnostics
            )
        ):
            self._terminal_pending_owner_diagnostics = (
                *self._terminal_pending_owner_diagnostics,
                (owner, error),
            )
        if record_collective and not any(
            retained_error is error
            for retained_error in self._terminal_pending_collective_diagnostics
        ):
            self._terminal_pending_collective_diagnostics = (
                *self._terminal_pending_collective_diagnostics,
                error,
            )
        return not retained

    def _retain_direct_terminal_diagnostic_locked(
        self, target, method_name, error, primary
    ):
        if (
            target is None
            or not isinstance(error, BaseException)
            or error is primary
        ):
            return
        if any(
            retained_target is target and retained_error is error
            for retained_target, _, retained_error in (
                self._terminal_pending_direct_diagnostics
            )
        ):
            return
        self._terminal_pending_direct_diagnostics = (
            *self._terminal_pending_direct_diagnostics,
            (target, method_name, error),
        )

    def _fatal_outcome_installed(self, transition):
        gate = self._terminal_gate
        with gate._condition:
            if transition is None or gate._fatal_transition is not transition:
                return False
            return gate._fatal_snapshot is not _TERMINAL_MISSING

    def _dispatch_terminal_owner_secondaries(
        self,
        errors,
        primary,
        owner,
        transition,
    ):
        if owner is None or not self._fatal_outcome_installed(transition):
            return
        for error in tuple(errors):
            if not isinstance(error, BaseException) or error is primary:
                continue
            try:
                call = getattr(owner, "_operator_call", None)
            except BaseException as diagnostic_error:
                with self._terminal_state_lock:
                    self._retain_terminal_secondary_locked(
                        diagnostic_error,
                        primary,
                        record_collective=False,
                    )
                continue
            if call is None:
                target = owner
                method_name = "_remember_secondary"
            else:
                target = call
                method_name = "record_secondary"
            self._dispatch_terminal_direct_secondary(
                target,
                method_name,
                error,
                primary,
                transition,
            )

    def _dispatch_terminal_direct_secondary(
        self,
        target,
        method_name,
        error,
        primary,
        transition,
    ):
        if target is None or not self._fatal_outcome_installed(transition):
            return
        if not isinstance(error, BaseException) or error is primary:
            return
        with self._terminal_state_lock:
            if any(
                retained_target is target and retained_error is error
                for retained_target, retained_error in (
                    self._terminal_owner_diagnostic_dispatches
                )
            ):
                return
            self._terminal_owner_diagnostic_dispatches = (
                *self._terminal_owner_diagnostic_dispatches,
                (target, error),
            )
        try:
            recorder = getattr(target, method_name, None)
        except BaseException as diagnostic_error:
            with self._terminal_state_lock:
                self._retain_terminal_secondary_locked(
                    diagnostic_error,
                    primary,
                    record_collective=False,
                )
            return
        if not callable(recorder):
            return
        try:
            recorder(error)
        except BaseException as diagnostic_error:
            with self._terminal_state_lock:
                self._retain_terminal_secondary_locked(
                    diagnostic_error,
                    primary,
                    record_collective=False,
                )

    def _dispatch_terminal_collective_secondaries(
        self,
        errors,
        primary,
        transition,
    ):
        if not self._fatal_outcome_installed(transition):
            return
        with self._terminal_state_lock:
            collective = self._pending_fatal_collective
            if collective is None:
                collective = self.collective
        if collective is None:
            return
        for error in tuple(errors):
            if not isinstance(error, BaseException) or error is primary:
                continue
            with self._terminal_state_lock:
                if any(
                    retained_collective is collective
                    and retained_error is error
                    for retained_collective, retained_error in (
                        self._terminal_collective_diagnostic_dispatches
                    )
                ):
                    continue
                self._terminal_collective_diagnostic_dispatches = (
                    *self._terminal_collective_diagnostic_dispatches,
                    (collective, error),
                )
            try:
                recorder = getattr(collective, "_record_fatal_secondary", None)
                if callable(recorder):
                    recorder(error)
            except BaseException as diagnostic_error:
                with self._terminal_state_lock:
                    self._retain_terminal_secondary_locked(
                        diagnostic_error,
                        primary,
                        record_collective=False,
                    )

    def _dispatch_terminal_secondaries_after_outcome(
        self, transition, *, include_collective=False
    ):
        if not self._fatal_outcome_installed(transition):
            return
        primary = transition.primary
        with self._terminal_state_lock:
            context = self._pending_fatal_context
            owner = None if context is None else context[2]
            errors = tuple(self._terminal_secondary_errors)
            owner_diagnostics = tuple(self._terminal_pending_owner_diagnostics)
            direct_diagnostics = tuple(
                self._terminal_pending_direct_diagnostics
            )
            collective_diagnostics = tuple(
                self._terminal_pending_collective_diagnostics
            )
        self._dispatch_terminal_owner_secondaries(
            errors,
            primary,
            owner,
            transition,
        )
        for retained_owner, error in owner_diagnostics:
            self._dispatch_terminal_owner_secondaries(
                (error,),
                primary,
                retained_owner,
                transition,
            )
        for target, method_name, error in direct_diagnostics:
            self._dispatch_terminal_direct_secondary(
                target,
                method_name,
                error,
                primary,
                transition,
            )
        if include_collective:
            self._dispatch_terminal_collective_secondaries(
                collective_diagnostics,
                primary,
                transition,
            )

    def _structured_terminal_secondaries(self, transition):
        if not self._fatal_outcome_installed(transition):
            return ()
        with self._terminal_state_lock:
            return tuple(self._terminal_pending_collective_diagnostics)

    def _record_structured_fatal_secondary(self, error):
        gate = self._terminal_gate
        with gate._condition:
            transition = gate._fatal_transition
            if transition is None:
                return
            primary = transition.primary
        with self._terminal_state_lock:
            self._retain_terminal_secondary_locked(
                error,
                primary,
                record_collective=False,
            )
        self._dispatch_terminal_secondaries_after_outcome(transition)

    def _terminalize_fatal_publication_failure(
        self,
        transition,
        secondary=None,
        *,
        owner=None,
    ):
        """Install the gate outcome and retain diagnostics without callbacks."""
        primary = transition.primary
        gate = self._terminal_gate
        with gate._condition:
            gate._require_fatal_transition(transition)
            if gate._fatal_publication_failure is _TERMINAL_MISSING:
                gate._fail_fatal_publication(transition, primary)
        try:
            with self._terminal_state_lock:
                self._retain_terminal_secondary_locked(secondary, primary)
        except BaseException as diagnostic_error:
            # The gate outcome is already terminal; diagnostics cannot replace it.
            try:
                with self._terminal_state_lock:
                    if (
                        diagnostic_error is not primary
                        and not any(
                            retained is diagnostic_error
                            for retained in self._terminal_secondary_errors
                        )
                    ):
                        self._terminal_secondary_errors = (
                            *self._terminal_secondary_errors,
                            diagnostic_error,
                        )
            except BaseException:
                pass
        return primary

    def _terminalize_communicator_fatal_failure(
        self,
        primary,
        *,
        diagnostics=(),
        transition=None,
    ):
        """Non-pluggable fallback for a failed structured fatal begin."""
        if not isinstance(primary, BaseException):
            raise TypeError("communicator fatal failure must be an exception")
        gate = self._terminal_gate
        discovering_token = gate._current_thread_admission()
        retained_transition = gate._terminalize_fatal_failure(
            primary,
            discovering_token,
        )
        canonical = retained_transition.primary
        with self._terminal_state_lock:
            prior = self._terminal_error
            self._terminal_error = canonical
            if self._pending_fatal_collective is None:
                self._pending_fatal_collective = self.collective
            for error in (primary, prior, *tuple(diagnostics)):
                self._retain_terminal_secondary_locked(error, canonical)
            if transition is not None and transition is not retained_transition:
                self._retain_terminal_secondary_locked(
                    RuntimeError("communicator fatal transition changed"),
                    canonical,
                )
        return retained_transition

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
            self._retain_terminal_secondary_locked(error, primary, owner)
        self._retain_direct_terminal_diagnostic_locked(
            owner,
            "_remember_secondary",
            owner_primary,
            primary,
        )
        self._retain_direct_terminal_diagnostic_locked(
            call,
            "record_secondary",
            call_primary,
            primary,
        )

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
            self._retain_terminal_secondary_locked(candidate, primary, owner)
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
            self._retain_terminal_secondary_locked(
                primary, transition.primary, owner
            )
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
            self._retain_terminal_secondary_locked(error, primary, owner)
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
        failure_context = [None]
        try:
            return self._elect_communicator_fatal(
                primary,
                owner=owner,
                discovering_token=discovering_token,
                failure_context=failure_context,
            )
        except BaseException as error:
            election = failure_context[0]
            if election is not None and election.recover():
                collective = election.collective
                terminalize = getattr(
                    collective,
                    "_terminalize_structured_fatal_failure",
                    None,
                )
                if not callable(terminalize):
                    raise RuntimeError(
                        "collective cannot terminalize fatal publication"
                    ) from error
                primary = terminalize(
                    error,
                    transition_handler=self._communicator_fatal_hook,
                    transition=election.transition,
                    diagnostics=election.recovery_errors,
                )
                raise primary
            raise

    def _elect_communicator_fatal(
        self,
        primary,
        *,
        owner=None,
        discovering_token=None,
        failure_context,
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
                election = _RecoverableFatalElection(
                    self,
                    collective,
                    primary,
                    provider,
                    lease,
                    owner,
                    discovering_token,
                )
                failure_context[0] = election
                if callable(reserve_outcome):
                    if discovering_token is not None:
                        gate._convertible_token_state(discovering_token)
                    try:
                        outcome = election._select_outcome_locked()
                    except BaseException:
                        election.recover_locked()
                        raise
                    if outcome.primary is not primary:
                        self._retain_terminal_secondary_locked(
                            primary, outcome.primary
                        )
                        primary = outcome.primary
                    if election.transition is None:
                        transition = gate.begin_fatal(
                            primary, discovering_token
                        )
                        election.transition = transition
                        election._retain_context_locked()
                    else:
                        transition = election.transition
                else:
                    transition = gate.begin_fatal(primary, discovering_token)
                    election.transition = transition
                    election._retain_context_locked()
                if transition.primary is not primary:
                    raise RuntimeError("communicator fatal primary changed")
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
                _, current_provider, current_lease, current_owner = (
                    self._normalize_communicator_fatal(primary, owner)
                )
                if provider is None:
                    provider = current_provider
                if lease is None and provider is current_provider:
                    lease = current_lease
                if owner is None:
                    owner = current_owner
                self._pending_fatal_context = (provider, lease, owner)
                self._issued_receipts.clear()
            retain_resources = (
                None
                if provider is None
                else getattr(provider, "_retain_transition_resources", None)
            )
            if callable(retain_resources):
                retain_resources(lease)
            if owner is not None:
                if owner.state not in {"detached", "quarantined"}:
                    owner.force_quarantine(primary)
            self._publish_deferred_async_quarantines()
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
            with self._terminal_state_lock:
                collective = self._pending_fatal_collective
                if collective is None:
                    collective = self.collective
            if callable(
                getattr(collective, "_begin_fatal_publication", None)
            ):
                raise
            raise self._terminalize_fatal_publication_failure(
                transition,
                error,
                owner=owner,
            )
        return primary

    def _enter_communicator_fatal_unchecked(
        self, primary, owner=None, discovering_token=None
    ):
        if not isinstance(primary, BaseException):
            raise TypeError("communicator fatal failure must be an exception")
        candidate = primary
        if discovering_token is None:
            discovering_token = self._terminal_gate._current_thread_admission()
        deferred_primary = self._defer_current_active_broadcast_fatal(
            primary,
            owner=owner,
            discovering_token=discovering_token,
        )
        if deferred_primary is not None:
            primary = deferred_primary
            discovering_token = None
        transition = self._begin_communicator_fatal(
            primary,
            owner=owner,
            discovering_token=discovering_token,
        )
        primary = transition.primary
        diagnostics = () if candidate is primary else (candidate,)
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
            if active is not None:
                if active.primary is not transition.primary:
                    raise RuntimeError("communicator fatal primary changed")
                active.transition = transition
                if active.transition_publish_started:
                    return transition.primary
            if active is not None and not isinstance(
                active.handler, _RuntimeCommunicatorFatalHook
            ):
                if active.completion_deferred:
                    return transition.primary
                if active.transition_published:
                    return active.snapshot
                snapshot = self._publish_communicator_fatal_transition(transition)
                active.snapshot = snapshot
                active.transition_published = True
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
                diagnostics=diagnostics,
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
        try:
            snapshot = self._publish_communicator_fatal_transition(transition)
            publish(snapshot)
            self._terminal_gate.publish_fatal(transition, snapshot)
            self._dispatch_terminal_secondaries_after_outcome(
                transition, include_collective=True
            )
        except BaseException as publication_error:
            raise self._terminalize_fatal_publication_failure(
                transition,
                publication_error,
                owner=owner,
            )
        return snapshot

    def _enter_communicator_fatal(
        self, primary, owner=None, discovering_token=None
    ):
        """Publish fatal or install the gate's bounded publication failure."""
        try:
            return self._enter_communicator_fatal_unchecked(
                primary,
                owner=owner,
                discovering_token=discovering_token,
            )
        except BaseException as publication_error:
            transition = self._terminal_gate._fatal_transition
            if transition is None:
                raise
            primary = transition.primary
            failure_was_recorded = (
                self._terminal_gate._fatal_publication_failed(
                    transition, primary
                )
            )
            with self._terminal_state_lock:
                collective = self._pending_fatal_collective
                if collective is None:
                    collective = self.collective
            structured_collective = callable(
                getattr(collective, "_begin_fatal_publication", None)
            )
            if structured_collective:
                try:
                    exit_claimed = (
                        collective._current_thread_claimed_fatal_hard_exit(
                            primary
                        )
                    )
                except BaseException as claim_error:
                    with self._terminal_state_lock:
                        self._retain_terminal_secondary_locked(
                            claim_error,
                            primary,
                        )
                    exit_claimed = False
                if exit_claimed:
                    raise
                terminalize = getattr(
                    collective,
                    "_terminalize_structured_fatal_failure",
                    None,
                )
                if not callable(terminalize):
                    if failure_was_recorded:
                        if publication_error is not primary:
                            raise
                        raise primary
                    raise RuntimeError(
                        "collective cannot terminalize fatal publication"
                    ) from publication_error
                primary = terminalize(
                    publication_error,
                    transition_handler=self._communicator_fatal_hook,
                    transition=transition,
                )
                if failure_was_recorded and publication_error is not primary:
                    raise
                raise primary
            primary = self._terminalize_fatal_publication_failure(
                transition,
                publication_error,
                owner=owner,
            )
            raise primary

    def barrier(self):
        self._require_usable()
        token = self._terminal_gate.admit_runtime("barrier_collective")
        try:
            validator = self._exact_admission_validator(
                token,
                scope="runtime",
                epoch=None,
            )
            return self._barrier_collective(
                _admission_token=token,
                _admission_validator=validator,
            )
        finally:
            self._release_admission(token)

    def _barrier_collective(
        self,
        *,
        _admission_token,
        _admission_validator=None,
    ):
        self._resolve_admission_validator(
            _admission_token,
            _admission_validator,
            scope="runtime",
            epoch=None,
        )
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
        token = self._terminal_gate.admit_runtime_setup("execution_config")
        try:
            return self._execution_config(
                residency_policy=residency_policy,
                device_memory_budget_bytes=device_memory_budget_bytes,
                host_memory_budget_bytes=host_memory_budget_bytes,
                prefetch_depth=prefetch_depth,
                _admission_token=token,
            )
        finally:
            self._release_admission(token)

    def _execution_config(
        self,
        *,
        residency_policy,
        device_memory_budget_bytes,
        host_memory_budget_bytes,
        prefetch_depth,
        _admission_token,
    ):
        validator = self._exact_admission_validator(
            _admission_token,
            scope="runtime_setup",
            epoch=None,
        )
        backend_metadata = _call_admitted_private(
            self._synchronize_active_backend,
            _admission_token=_admission_token,
            _admission_validator=validator,
        )
        device_resolution = self._resolve_budget(
            "device",
            device_memory_budget_bytes,
            _admission_token=_admission_token,
            _admission_validator=validator,
        )
        host_resolution = self._resolve_budget(
            "host",
            host_memory_budget_bytes,
            _admission_token=_admission_token,
            _admission_validator=validator,
        )
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
                provider = ActiveWorkingSetProvider(
                    self,
                    device_budget_resolution=device_resolution,
                    host_budget_resolution=host_resolution,
                    prefetch_depth=prefetch_depth,
                    _admission_token=_admission_token,
                    _admission_validator=validator,
                )
                self._install_active_provider(
                    provider,
                    _admission_token=_admission_token,
                    _admission_validator=validator,
                )
            elif not self._active_provider.matches_config(
                device_resolution,
                host_resolution,
                    prefetch_depth,
                    _admission_token=_admission_token,
                    _admission_validator=validator,
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
        token = self._terminal_gate.admit_runtime_setup("preflight_residency")
        try:
            return self._preflight_residency(
                request,
                plan,
                _admission_token=token,
            )
        finally:
            self._release_admission(token)

    def _preflight_residency(self, request, plan, *, _admission_token):
        validator = self._exact_admission_validator(
            _admission_token,
            scope="runtime_setup",
            epoch=None,
        )
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
                    _admission_token=_admission_token,
                    _admission_validator=validator,
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
            return self._publish_residency_receipt(
                receipt,
                _admission_token=_admission_token,
                _admission_validator=validator,
            )

        status = self._control_array(
            [int(local_error is not None)],
            np.int32,
            _admission_token=_admission_token,
            _admission_validator=validator,
        )
        failed = int(
            self._host_control(
                self._runtime_setup_allreduce(
                    status,
                    op="max",
                    _admission_token=_admission_token,
                    _admission_validator=validator,
                )
            ).reshape(-1)[0]
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
            _admission_token=_admission_token,
            _admission_validator=validator,
        )
        policy_minimum = self._host_control(
            self._runtime_setup_allreduce(
                policy,
                op="min",
                _admission_token=_admission_token,
                _admission_validator=validator,
            )
        )
        policy_maximum = self._host_control(
            self._runtime_setup_allreduce(
                policy,
                op="max",
                _admission_token=_admission_token,
                _admission_validator=validator,
            )
        )
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
            _admission_token=_admission_token,
            _admission_validator=validator,
        )
        hash_minimum = self._host_control(
            self._runtime_setup_allreduce(
                hash_control,
                op="min",
                _admission_token=_admission_token,
                _admission_validator=validator,
            )
        )
        hash_maximum = self._host_control(
            self._runtime_setup_allreduce(
                hash_control,
                op="max",
                _admission_token=_admission_token,
                _admission_validator=validator,
            )
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
        requirements = self._control_array(
            requirement_values,
            np.int64,
            _admission_token=_admission_token,
            _admission_validator=validator,
        )
        requirement_minimum = self._host_control(
            self._runtime_setup_allreduce(
                requirements,
                op="min",
                _admission_token=_admission_token,
                _admission_validator=validator,
            )
        )
        requirement_maximum = self._host_control(
            self._runtime_setup_allreduce(
                requirements,
                op="max",
                _admission_token=_admission_token,
                _admission_validator=validator,
            )
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
        capacity = self._control_array(
            [capacity_code],
            np.int32,
            _admission_token=_admission_token,
            _admission_validator=validator,
        )
        capacity_failed = int(
            self._host_control(
                self._runtime_setup_allreduce(
                    capacity,
                    op="max",
                    _admission_token=_admission_token,
                    _admission_validator=validator,
                )
            ).reshape(-1)[0]
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
        return self._publish_residency_receipt(
            receipt,
            _admission_token=_admission_token,
            _admission_validator=validator,
        )

    def consume_residency_receipt(
        self,
        receipt,
        request,
        plan,
        *,
        _admission_token,
        _admission_validator=None,
        _resource_recorder=None,
    ):
        _admission_validator = self._resolve_admission_validator(
            _admission_token,
            _admission_validator,
            scope="construction",
            epoch=_admission_token.epoch,
        )
        self._require_usable()
        token = _admission_token
        with self._terminal_gate._condition:
            lease = self._terminal_gate._lease_state(token.epoch)
            if (
                lease.phase != "constructing"
                or self._terminal_gate._live_epoch != token.epoch
            ):
                raise RuntimeError(
                    "receipt consumption requires a constructing lease epoch"
                )
        if not isinstance(receipt, ResidencyPreflightReceipt):
            raise TypeError("receipt must be a ResidencyPreflightReceipt")
        if _resource_recorder is not None and not callable(_resource_recorder):
            raise TypeError("receipt resource recorder must be callable")
        issued = self._issued_receipts.pop(id(receipt), None)
        if issued is not receipt:
            raise ValueError(
                "residency preflight receipt was not issued by this runtime"
            )
        if _resource_recorder is not None:
            try:
                _resource_recorder(issued)
            except BaseException:
                self._issued_receipts[id(issued)] = issued
                raise
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

    def _control_array(
        self,
        values,
        dtype,
        *,
        _admission_token,
        _admission_validator=None,
    ):
        self._resolve_admission_validator(
            _admission_token,
            _admission_validator,
            scope="runtime_setup",
            epoch=None,
        )
        converter = getattr(self.backend, "asarray", None)
        if callable(converter):
            return converter(values, dtype=dtype)
        return np.asarray(values, dtype=dtype)

    def _runtime_setup_allreduce(
        self,
        value,
        *,
        op,
        _admission_token,
        _admission_validator=None,
    ):
        self._resolve_admission_validator(
            _admission_token,
            _admission_validator,
            scope="runtime_setup",
            epoch=None,
        )
        return self.collective.allreduce(value, op=op)

    @staticmethod
    def _host_control(value):
        getter = getattr(value, "get", None)
        if callable(getter):
            value = getter()
        return np.asarray(value)

    def _requested_budget_agrees(
        self,
        requested,
        resource,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        if _admission_token is None:
            _admission_token = self._terminal_gate._current_thread_admission()
            if _admission_token is None:
                token = self._terminal_gate.admit_runtime_setup("budget_probe")
                try:
                    return self._requested_budget_agrees(
                        requested,
                        resource,
                        _admission_token=token,
                    )
                finally:
                    self._release_admission(token)
        _admission_validator = self._resolve_admission_validator(
            _admission_token,
            _admission_validator,
            scope="runtime_setup",
            epoch=None,
        )
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
        status = self._control_array(
            [int(local_error is not None)],
            np.int32,
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
        )
        failed = int(
            self._host_control(
                self._runtime_setup_allreduce(
                    status,
                    op="max",
                    _admission_token=_admission_token,
                    _admission_validator=_admission_validator,
                )
            ).reshape(-1)[0]
        )
        control = self._control_array(
            [encoded],
            np.int64,
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
        )
        minimum = self._host_control(
            self._runtime_setup_allreduce(
                control,
                op="min",
                _admission_token=_admission_token,
                _admission_validator=_admission_validator,
            )
        ).reshape(-1)[0]
        maximum = self._host_control(
            self._runtime_setup_allreduce(
                control,
                op="max",
                _admission_token=_admission_token,
                _admission_validator=_admission_validator,
            )
        ).reshape(-1)[0]
        if failed:
            raise ValueError(
                "{} memory budget request validation failed".format(resource)
            ) from local_error
        if int(minimum) != int(maximum):
            raise RuntimeError("{} memory budget request disagreement".format(resource))

    def _auto_available_snapshot(
        self,
        resource,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        if _admission_token is None:
            _admission_token = self._terminal_gate._current_thread_admission()
            if _admission_token is None:
                token = self._terminal_gate.admit_runtime_setup("budget_probe")
                try:
                    return self._auto_available_snapshot(
                        resource,
                        _admission_token=token,
                    )
                finally:
                    self._release_admission(token)
        _admission_validator = self._resolve_admission_validator(
            _admission_token,
            _admission_validator,
            scope="runtime_setup",
            epoch=None,
        )
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
            status = self._control_array(
                [int(local_error is not None)],
                np.int32,
                _admission_token=_admission_token,
                _admission_validator=_admission_validator,
            )
            failed = int(
                self._host_control(
                    self._runtime_setup_allreduce(
                        status,
                        op="max",
                        _admission_token=_admission_token,
                        _admission_validator=_admission_validator,
                    )
                ).reshape(-1)[0]
            )
        if failed:
            raise RuntimeError(
                "{} memory availability preflight failed".format(resource)
            ) from local_error
        if self.world_size > 1:
            control = self._control_array(
                [available],
                np.int64,
                _admission_token=_admission_token,
                _admission_validator=_admission_validator,
            )
            available = int(
                self._host_control(
                    self._runtime_setup_allreduce(
                        control,
                        op="min",
                        _admission_token=_admission_token,
                        _admission_validator=_admission_validator,
                    )
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

    def _resolve_budget(
        self,
        resource,
        requested,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        if _admission_token is None:
            _admission_token = self._terminal_gate._current_thread_admission()
            if _admission_token is None:
                token = self._terminal_gate.admit_runtime_setup("budget_probe")
                try:
                    return self._resolve_budget(
                        resource,
                        requested,
                        _admission_token=token,
                    )
                finally:
                    self._release_admission(token)
        _admission_validator = self._resolve_admission_validator(
            _admission_token,
            _admission_validator,
            scope="runtime_setup",
            epoch=None,
        )
        self._requested_budget_agrees(
            requested,
            resource,
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
        )
        if requested is not None:
            return MemoryBudgetResolution(
                requested_bytes=requested,
                resolved_bytes=requested,
                source="explicit",
                available_snapshot_bytes=None,
                resource=resource,
            )
        return self._auto_available_snapshot(
            resource,
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
        )

    def _synchronize_active_backend(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        if _admission_token is None:
            _admission_token = self._terminal_gate._current_thread_admission()
            if _admission_token is None:
                token = self._terminal_gate.admit_runtime_setup(
                    "execution_config_backend_sync"
                )
                try:
                    return self._synchronize_active_backend(
                        _admission_token=token
                    )
                finally:
                    self._release_admission(token)
        _admission_validator = self._resolve_admission_validator(
            _admission_token,
            _admission_validator,
            scope="runtime_setup",
            epoch=None,
        )
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

        status = self._control_array(
            [int(local_error is not None)],
            np.int32,
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
        )
        failed = self._runtime_setup_allreduce(
            status,
            op="max",
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
        )
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
        control = self._control_array(
            words,
            np.uint64,
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
        )
        minimum = self._runtime_setup_allreduce(
            control,
            op="min",
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
        )
        maximum = self._runtime_setup_allreduce(
            control,
            op="max",
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
        )
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

    def _admit_runtime_close_step(self, transition, operation):
        gate = self._terminal_gate
        try:
            return gate.admit_runtime_close(transition, operation)
        except BaseException:
            token = gate._recover_current_thread_admission(
                scope="runtime_close",
                operation=operation,
                epoch=None,
                transition=transition,
            )
            if token is not None:
                self._release_runtime_close_step(token)
            raise

    def _commit_fatal_runtime_close(self, transition, error):
        def finalize_pending():
            try:
                return self._clear_runtime_references(
                    error,
                    publish_error=False,
                    mark_closed=False,
                    clear_collective=False,
                )
            except BaseException as clear_error:
                with self._terminal_state_lock:
                    self._retain_terminal_secondary_locked(
                        clear_error,
                        transition.primary,
                    )
                return transition.primary

        try:
            result = self._terminal_gate.commit_runtime_close(
                transition,
                finalize_pending,
            )
        except BaseException:
            with self._terminal_gate._condition:
                publication_failed = (
                    self._terminal_gate._fatal_transition is transition
                    and self._terminal_gate._fatal_publication_failure
                    is transition.primary
                )
            if publication_failed:
                with self._terminal_state_lock:
                    collective = self._pending_fatal_collective
                join_publication = (
                    None
                    if collective is None
                    else getattr(
                        collective,
                        "_wait_for_joined_fatal_publication",
                        None,
                    )
                )
                if callable(join_publication):
                    join_publication()
                else:
                    self._finish_failed_fatal_runtime_close(transition)
            raise
        if not self._closed:
            try:
                self._clear_runtime_references(
                    error,
                    publish_error=False,
                    clear_collective=False,
                )
            except BaseException as clear_error:
                with self._terminal_state_lock:
                    self._retain_terminal_secondary_locked(
                        clear_error,
                        transition.primary,
                    )
        return result

    def _finish_failed_fatal_runtime_close(self, transition):
        primary = transition.primary
        self._terminal_gate._fail_fatal_publication(transition, primary)
        secondary = self._terminal_gate._complete_failed_fatal_runtime_close(
            transition,
            lambda: self._clear_runtime_references(
                primary,
                publish_error=False,
                clear_collective=False,
            ),
        )
        if secondary is not None:
            with self._terminal_state_lock:
                self._retain_terminal_secondary_locked(
                    secondary,
                    primary,
                )
        raise primary

    def _fail_stop_elected_runtime_close(self, transition, error):
        gate = self._terminal_gate
        with gate._condition:
            fatal_transition = gate._fatal_transition
            phase = gate._phase
            commit_selected = gate._runtime_close_commit_selected
        if fatal_transition is None and phase is _TerminalPhase.RUNTIME_CLOSED:
            raise error
        primary = (
            error if fatal_transition is None else fatal_transition.primary
        )
        if error is not primary:
            with self._terminal_state_lock:
                self._retain_terminal_secondary_locked(error, primary)
        if fatal_transition is None and commit_selected:
            secondary = gate._fail_selected_runtime_close(
                transition,
                primary,
                lambda: self._clear_runtime_references(primary),
            )
            if secondary is not None:
                with self._terminal_state_lock:
                    self._retain_terminal_secondary_locked(
                        secondary,
                        primary,
                    )
            raise primary
        try:
            primary = self._enter_communicator_fatal(primary)
        except BaseException as publication_error:
            fatal_transition = gate._fatal_transition
            if fatal_transition is None or fatal_transition.primary is not primary:
                raise primary
            if publication_error is not primary:
                with self._terminal_state_lock:
                    self._retain_terminal_secondary_locked(
                        publication_error,
                        primary,
                    )
            self._finish_failed_fatal_runtime_close(fatal_transition)
        fatal_transition = gate._fatal_transition
        if fatal_transition is None or fatal_transition.primary is not primary:
            raise RuntimeError("runtime close fatal transition changed") from primary
        self._commit_fatal_runtime_close(fatal_transition, primary)
        raise primary

    def _runtime_close_snapshot(self):
        with self._terminal_gate._condition:
            with self._terminal_state_lock:
                provider = self._active_provider
                lease = (
                    None
                    if provider is None
                    else getattr(provider, "_active_lease", None)
                )
                scheduler = None if lease is None else lease.scheduler
                return provider, lease, scheduler

    @staticmethod
    def _close_collective(
        collective,
        *,
        _admission_token,
        _admission_validator,
    ):
        _require_resource_admission(
            _admission_token,
            _admission_validator,
        )
        return collective.close()

    def _run_elected_runtime_close(self, transition, error):
        """Run every fallible elected-owner action under the caller's guard."""
        _, _, frozen_scheduler = self._runtime_close_snapshot()
        if frozen_scheduler is not None:
            frozen_scheduler._start_counted_completions()
        transition = self._terminal_gate._drain_runtime_close(transition)
        if isinstance(transition, _FatalTransition):
            primary = transition.primary
            self._commit_fatal_runtime_close(transition, primary)
            raise primary

        provider, lease, _ = self._runtime_close_snapshot()
        collective = self.collective
        if lease is not None:
            try:
                lease.close()
            except Exception as caught:
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
        if provider is not None:
            try:
                provider_token = self._admit_runtime_close_step(
                    transition, "provider_close"
                )
            except Exception as caught:
                if self._terminal_gate.phase in (
                    _TerminalPhase.FATAL_PENDING,
                    _TerminalPhase.FATAL_PUBLISHED,
                ):
                    fatal_transition = self._terminal_gate.begin_runtime_close(None)
                    error = fatal_transition.primary
                    self._commit_fatal_runtime_close(fatal_transition, error)
                    raise error
                if error is None:
                    error = caught
            else:
                try:
                    provider_validator = self._exact_admission_validator(
                        provider_token,
                        scope="runtime_close",
                        epoch=None,
                        transition_sequence=transition.sequence,
                    )
                    provider._close_for_runtime(
                        _admission_token=provider_token,
                        _admission_validator=provider_validator,
                    )
                except Exception as caught:
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
                    collective_token = self._admit_runtime_close_step(
                        transition, "collective_close"
                    )
                    try:
                        collective_validator = self._exact_admission_validator(
                            collective_token,
                            scope="runtime_close",
                            epoch=None,
                            transition_sequence=transition.sequence,
                        )
                        self._close_collective(
                            collective,
                            _admission_token=collective_token,
                            _admission_validator=collective_validator,
                        )
                    finally:
                        self._release_runtime_close_step(collective_token)
        except Exception as caught:
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
        if isinstance(result, BaseException):
            fatal_transition = self._terminal_gate._fatal_transition
            if (
                fatal_transition is not None
                and fatal_transition.primary is result
            ):
                self._commit_fatal_runtime_close(fatal_transition, result)
            raise result
        if error is not None:
            raise error
        return result

    def close(self):
        if self._closed:
            return
        error = self._terminal_error
        if error is None:
            error = getattr(self.backend, "_execution_terminal_error", None)
        request = None
        entry_error = None
        try:
            request = self._terminal_gate.admit_runtime("begin_runtime_close")
        except BaseException as caught:
            request = self._terminal_gate._recover_current_thread_admission(
                scope="runtime",
                operation="begin_runtime_close",
                epoch=None,
            )
            entry_error = (
                self._terminal_gate._classify_runtime_close_admission_failure(
                    caught,
                    request,
                )
            )
        try:
            transition, elected = self._terminal_gate._freeze_runtime_close(request)
        except BaseException as caught:
            if entry_error is None:
                entry_error = caught
            elif caught is not entry_error:
                with self._terminal_state_lock:
                    self._retain_terminal_secondary_locked(caught, entry_error)
            transition, elected = (
                self._terminal_gate._recover_runtime_close_entry()
            )
            if transition is None:
                self._enter_communicator_fatal(
                    entry_error,
                    discovering_token=request,
                )
                transition = self._terminal_gate._fatal_transition
                error = transition.primary
                self._commit_fatal_runtime_close(transition, error)
                raise error
        if isinstance(transition, _FatalTransition):
            error = transition.primary
            if entry_error is not None and entry_error is not error:
                with self._terminal_state_lock:
                    self._retain_terminal_secondary_locked(
                        entry_error, error
                    )
            self._commit_fatal_runtime_close(transition, error)
            raise error

        if not elected:
            try:
                transition = self._terminal_gate._drain_runtime_close(
                    transition
                )
            except BaseException as caught:
                self._fail_stop_elected_runtime_close(transition, caught)
            if isinstance(transition, _FatalTransition):
                error = transition.primary
                try:
                    self._commit_fatal_runtime_close(transition, error)
                except BaseException as caught:
                    if caught is error:
                        raise
                    self._fail_stop_elected_runtime_close(transition, caught)
                raise error
            try:
                result = self._terminal_gate.commit_runtime_close(
                    transition, lambda: None
                )
            except BaseException as caught:
                self._fail_stop_elected_runtime_close(transition, caught)
            if isinstance(result, BaseException):
                raise result
            if entry_error is not None:
                raise entry_error
            return result

        try:
            if entry_error is not None:
                raise entry_error
            return self._run_elected_runtime_close(transition, error)
        except BaseException as caught:
            self._fail_stop_elected_runtime_close(transition, caught)

    def resource_state(self):
        gate = self._terminal_gate
        deadline = time.monotonic() + _TERMINAL_TIMEOUT_S

        def fail_terminal_wait(error):
            with gate._condition:
                transition = gate._fatal_transition
                primary = error if transition is None else transition.primary
            if error is not primary:
                with self._terminal_state_lock:
                    self._retain_terminal_secondary_locked(error, primary)
            try:
                published = self._enter_communicator_fatal(primary)
                if isinstance(published, BaseException):
                    primary = published
            except BaseException as publication_error:
                with gate._condition:
                    transition = gate._fatal_transition
                    if transition is not None:
                        primary = transition.primary
                if publication_error is not primary:
                    with self._terminal_state_lock:
                        self._retain_terminal_secondary_locked(
                            publication_error,
                            primary,
                        )
            raise primary

        while True:
            owner_thread = None
            with gate._condition:
                phase = gate._phase
                live_epoch = gate._live_epoch
                lease_state = (
                    None if live_epoch is None else gate._leases[live_epoch]
                )
                lease_phase = None if lease_state is None else lease_state.phase
                fatal = gate._fatal_transition
                if phase is _TerminalPhase.RUNTIME_CLOSING:
                    transition = gate._runtime_close_transition
                    owner_thread = (
                        None if transition is None else transition.owner_thread
                    )
                elif lease_phase == "constructing":
                    construction = lease_state.construction
                    owner_thread = (
                        None
                        if construction is None
                        else construction.owner_thread
                    )
                elif lease_phase == "closing":
                    transition = lease_state.transition
                    owner_thread = (
                        None if transition is None else transition.owner_thread
                    )
                elif phase is _TerminalPhase.FATAL_PENDING:
                    owner_thread = gate._fatal_runtime_finalizer_thread

            if phase is _TerminalPhase.FATAL_PENDING:
                if owner_thread is not None and not owner_thread.is_alive():
                    fail_terminal_wait(
                        RuntimeError("resource_state terminal owner exited")
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    fail_terminal_wait(
                        TimeoutError("resource_state terminal wait timed out")
                    )
                try:
                    gate.wait_for_published(remaining)
                except TimeoutError:
                    fail_terminal_wait(
                        TimeoutError("resource_state terminal wait timed out")
                    )
                continue
            if phase is _TerminalPhase.FATAL_PUBLISHED or (
                phase is _TerminalPhase.RUNTIME_CLOSED and fatal is not None
            ):
                return self._terminal_resource_state()
            if phase is _TerminalPhase.RUNTIME_CLOSED:
                return self._terminal_resource_state()
            if phase is _TerminalPhase.RUNTIME_CLOSING or lease_phase in (
                "constructing",
                "closing",
                "fatal_retained",
            ):
                with gate._condition:
                    current_epoch = gate._live_epoch
                    current_lease_phase = (
                        None
                        if current_epoch is None
                        else gate._leases[current_epoch].phase
                    )
                    if (
                        gate._phase is not phase
                        or current_epoch != live_epoch
                        or current_lease_phase != lease_phase
                    ):
                        continue
                    if owner_thread is not None and not owner_thread.is_alive():
                        wait_error = RuntimeError(
                            "resource_state terminal owner exited"
                        )
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            wait_error = TimeoutError(
                                "resource_state terminal wait timed out"
                            )
                        else:
                            wait_error = None
                            gate._condition.wait(remaining)
                if wait_error is not None:
                    fail_terminal_wait(wait_error)
                continue

            try:
                if live_epoch is None:
                    token = gate.admit_runtime("resource_state")
                    scope = "runtime"
                else:
                    token = gate.admit_lease(live_epoch, "resource_state")
                    scope = "lease"
            except RuntimeError:
                continue
            retry = False
            try:
                validator = self._exact_admission_validator(
                    token,
                    scope=scope,
                    epoch=live_epoch if scope == "lease" else None,
                )
                with gate._condition:
                    current_epoch = gate._live_epoch
                    current_phase = gate._phase
                    current_lease_phase = (
                        None
                        if current_epoch is None
                        else gate._leases[current_epoch].phase
                    )
                    retry = (
                        current_phase is not _TerminalPhase.HEALTHY
                        or current_epoch != live_epoch
                        or (
                            scope == "runtime"
                            and current_epoch is not None
                        )
                        or (
                            scope == "lease"
                            and current_lease_phase != "open"
                        )
                    )
                if retry:
                    continue
                provider = self._active_provider
                if provider is None:
                    state = self._empty_resource_state()
                else:
                    state = provider.runtime_resource_state(
                        _admission_token=token,
                        _admission_validator=validator,
                    )
            finally:
                self._release_admission(token)
            if retry:
                continue
            break

        if self._terminal_quarantine.poisoned:
            retained = self._terminal_quarantine.retained_resource_state()
            for key in ("cache_bytes", "pinned_bytes", "stream_count", "event_count"):
                state[key] = max(state[key], retained[key])
            state.update(self._terminal_quarantine.resource_state())
        return state

    @staticmethod
    def _empty_resource_state():
        return {
            "active_leases": 0,
            "cache_bytes": 0,
            "cache_refs": 0,
            "pinned_bytes": 0,
            "stream_count": 0,
            "event_count": 0,
        }

    def _terminal_resource_state(self):
        state = self._empty_resource_state()
        retained = self._terminal_quarantine.retained_resource_state()
        for key in ("cache_bytes", "pinned_bytes", "stream_count", "event_count"):
            state[key] = retained[key]
        if self._terminal_quarantine.poisoned:
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
