"""Resource-neutral coordination for distributed runtime terminal transitions."""

from dataclasses import dataclass, field, replace
import enum
import math
import threading
import time
from typing import Literal


_TERMINAL_TIMEOUT_S = 5.0


class _TerminalPhase(enum.Enum):
    HEALTHY = "healthy"
    FATAL_PENDING = "fatal_pending"
    FATAL_PUBLISHED = "fatal_published"
    RUNTIME_CLOSING = "runtime_closing"
    RUNTIME_CLOSED = "runtime_closed"


@dataclass(frozen=True)
class _ResourceAdmission:
    gate_id: int
    scope: Literal[
        "runtime",
        "runtime_setup",
        "construction",
        "lease",
        "lease_close",
        "runtime_close",
    ]
    epoch: int | None
    thread_id: int
    operation: str
    sequence: int
    parent_sequence: int | None = None
    transition_sequence: int | None = None


@dataclass(frozen=True)
class _FatalTransition:
    gate_id: int
    primary: BaseException
    sequence: int


@dataclass(frozen=True)
class _LeaseCloseTransition:
    gate_id: int
    epoch: int
    owner_thread_id: int
    owner_thread: threading.Thread
    sequence: int


@dataclass(frozen=True)
class _RuntimeCloseTransition:
    gate_id: int
    owner_thread_id: int
    owner_thread: threading.Thread
    sequence: int


@dataclass(frozen=True)
class _MonitorOutcome:
    kind: Literal["fatal_elected", "stopped_clean"]
    primary: BaseException | None = None
    generation: int | None = None


class _FatalMonitorHandoff:
    """Linearizes monitor store reads, outcome selection, and exit acknowledgment."""

    def __init__(self):
        self._condition = threading.Condition()
        self._next_generation = 1
        self._requested_generation: int | None = None
        self._next_read_generation = 1
        self._inflight_read = None
        self._fatal_reservation: _MonitorOutcome | None = None
        self._outcome: _MonitorOutcome | None = None
        self._exited_outcome: _MonitorOutcome | None = None

    def request_stop(self) -> int:
        with self._condition:
            if self._requested_generation is None:
                self._requested_generation = self._next_generation
                self._next_generation += 1
            self._condition.notify_all()
            return self._requested_generation

    def observe_stop(self) -> int | None:
        with self._condition:
            return self._requested_generation

    def observe_outcome(self) -> _MonitorOutcome | None:
        with self._condition:
            return self._outcome

    def read_store_if_unselected(self, read):
        if not callable(read):
            raise TypeError("monitor store read must be callable")
        with self._condition:
            if self._outcome is not None:
                return self._outcome, None
            if self._fatal_reservation is not None:
                return self._complete_fatal_reservation_locked(), None
            if self._inflight_read is not None:
                raise RuntimeError("fatal monitor store read is already in flight")
            read_generation = self._next_read_generation
            self._next_read_generation += 1
            owner = threading.current_thread()
            stop_generation = self._requested_generation
            claim = (read_generation, owner)
            self._inflight_read = claim

        value = None
        read_error = None
        try:
            value = read()
        except BaseException as error:
            read_error = error

        with self._condition:
            if self._inflight_read != claim:
                raise RuntimeError("fatal monitor store read ownership changed")
            self._inflight_read = None
            self._condition.notify_all()
            if self._outcome is not None:
                return self._outcome, None
            if self._fatal_reservation is not None:
                return self._complete_fatal_reservation_locked(), None
            if self._requested_generation != stop_generation:
                return None, None
        if read_error is not None:
            raise read_error
        return None, value

    def _complete_fatal_reservation_locked(self):
        reservation = self._fatal_reservation
        if reservation is None:
            raise RuntimeError("monitor fatal outcome is not reserved")
        if self._outcome is None:
            self._outcome = reservation
            self._condition.notify_all()
        elif self._outcome is not reservation:
            raise RuntimeError("monitor fatal outcome changed")
        return self._outcome

    def select_fatal(
        self,
        primary: BaseException,
        *,
        before_select=None,
    ) -> _MonitorOutcome:
        if not isinstance(primary, BaseException):
            raise TypeError("monitor fatal primary must be an exception")
        if before_select is not None and not callable(before_select):
            raise TypeError("monitor fatal preparation must be callable")
        with self._condition:
            if self._outcome is None:
                reservation = self._fatal_reservation
                if reservation is None:
                    reservation = _MonitorOutcome(
                        kind="fatal_elected", primary=primary
                    )
                    try:
                        self._fatal_reservation = reservation
                        if before_select is not None:
                            before_select()
                        self._outcome = reservation
                    finally:
                        if self._outcome is None:
                            self._complete_fatal_reservation_locked()
                elif reservation.primary is not primary:
                    raise RuntimeError("monitor fatal primary changed")
                else:
                    self._complete_fatal_reservation_locked()
                self._condition.notify_all()
            return self._outcome

    def select_clean(self, generation: int) -> _MonitorOutcome:
        with self._condition:
            if generation != self._requested_generation:
                raise RuntimeError("monitor stop generation is not current")
            if self._outcome is None:
                if self._fatal_reservation is not None:
                    self._complete_fatal_reservation_locked()
                else:
                    self._outcome = _MonitorOutcome(
                        kind="stopped_clean", generation=generation
                    )
                    self._condition.notify_all()
            return self._outcome

    def wait_for_outcome(
        self, generation: int, timeout_s: float
    ) -> _MonitorOutcome:
        deadline = _TerminalLifecycleGate._deadline(timeout_s)
        with self._condition:
            if generation != self._requested_generation:
                raise RuntimeError("monitor stop generation is not current")
            while self._outcome is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("fatal monitor acknowledgment timed out")
                self._condition.wait(remaining)
            return self._outcome

    def wait_for_selection(self, timeout_s: float) -> _MonitorOutcome:
        deadline = _TerminalLifecycleGate._deadline(timeout_s)
        with self._condition:
            while self._outcome is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("fatal monitor outcome selection timed out")
                self._condition.wait(remaining)
            return self._outcome

    def acknowledge_exit(self, outcome: _MonitorOutcome) -> _MonitorOutcome:
        with self._condition:
            if outcome is not self._outcome:
                raise RuntimeError("monitor exit outcome is not selected")
            if self._exited_outcome is None:
                self._exited_outcome = outcome
                self._condition.notify_all()
            elif self._exited_outcome is not outcome:
                raise RuntimeError("monitor exit outcome changed")
            return self._exited_outcome

    def wait_for_exit(
        self, outcome: _MonitorOutcome, timeout_s: float
    ) -> _MonitorOutcome:
        deadline = _TerminalLifecycleGate._deadline(timeout_s)
        with self._condition:
            if outcome is not self._outcome:
                raise RuntimeError("monitor exit outcome is not selected")
            while self._exited_outcome is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "fatal monitor exit acknowledgment timed out"
                    )
                self._condition.wait(remaining)
            if self._exited_outcome is not outcome:
                raise RuntimeError("monitor exit outcome changed")
            return self._exited_outcome


_MISSING = object()


@dataclass
class _TokenState:
    token: _ResourceAdmission
    status: Literal["active", "released", "converted"] = "active"
    async_capability: object = _MISSING
    async_claimed: bool = False


@dataclass
class _LeaseState:
    phase: Literal[
        "constructing", "open", "closing", "closed", "fatal_retained"
    ]
    transition: _LeaseCloseTransition | None = None
    result: object = None
    has_result: bool = False
    construction: object = None


@dataclass
class _LeaseConstructionTransaction:
    gate_id: int
    operation: str
    owner_thread_id: int
    owner_thread: threading.Thread
    state: Literal[
        "prepared", "active", "committed", "aborted", "fatal_retained"
    ] = "prepared"
    epoch: int | None = None
    token: _ResourceAdmission | None = None
    primary: BaseException | None = None
    result: object = _MISSING
    secondaries: tuple[BaseException, ...] = ()
    resource_slots: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _LeaseConstructionResourceSnapshot:
    name: str
    resource: object
    kind: str | None
    records: tuple
    events: tuple
    streams: tuple


class _LeaseConstructionResourceSlot:
    """Gate-owned, query-free ownership published by one real constructor."""

    def __init__(self, gate, transaction, name):
        self._gate = gate
        self._transaction = transaction
        self.name = name
        self._resource = None
        self._kind = None
        self._records = {}
        self._events = {}
        self._streams = {}

    def publish(
        self,
        resource=None,
        *,
        kind=None,
        records=(),
        events=(),
        streams=(),
    ):
        self._gate._publish_lease_construction_resource(
            self,
            resource=resource,
            kind=kind,
            records=records,
            events=events,
            streams=streams,
        )

    def snapshot(self):
        return self._gate._snapshot_lease_construction_resource(self)

    def clear(self):
        self._gate._clear_lease_construction_resource(self)


def _publish_lease_construction_resource(
    slot,
    resource=None,
    *,
    kind=None,
    records=(),
    events=(),
    streams=(),
):
    if slot is None:
        return
    if not isinstance(slot, _LeaseConstructionResourceSlot):
        raise TypeError("lease construction resource slot is invalid")
    slot.publish(
        resource,
        kind=kind,
        records=records,
        events=events,
        streams=streams,
    )


class _TerminalLifecycleGate:
    """Linearizes admissions, lease epochs, fatal publication, and close."""

    def __init__(self):
        self._condition = threading.Condition()
        self._gate_id = id(self)
        self._phase = _TerminalPhase.HEALTHY
        self._next_epoch = 1
        self._next_sequence = 1
        self._tokens: dict[int, _TokenState] = {}
        self._thread_tokens: dict[int, int] = {}
        self._leases: dict[int, _LeaseState] = {}
        self._live_epoch: int | None = None
        self._fatal_transition: _FatalTransition | None = None
        self._fatal_snapshot = _MISSING
        self._fatal_publication_failure = _MISSING
        self._fatal_runtime_finalizer_state = "none"
        self._fatal_runtime_finalizer_owner: threading.Thread | None = None
        self._fatal_runtime_finalizer_thread: threading.Thread | None = None
        self._runtime_close_transition: _RuntimeCloseTransition | None = None
        self._runtime_close_commit_selected = False
        self._runtime_close_result = _MISSING
        self._runtime_close_admission_rejections = {}

    @property
    def phase(self):
        with self._condition:
            return self._phase

    def _sequence(self):
        sequence = self._next_sequence
        self._next_sequence += 1
        return sequence

    @staticmethod
    def _require_operation(operation):
        if not isinstance(operation, str) or not operation:
            raise ValueError("admission operation must be a non-empty string")

    @staticmethod
    def _deadline(timeout_s):
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise TypeError("admission timeout must be a number")
        try:
            finite = math.isfinite(timeout_s)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("admission timeout must be finite")
        if timeout_s < 0:
            raise ValueError("admission timeout must be non-negative")
        return time.monotonic() + timeout_s

    def _raise_for_terminal_phase(self):
        if self._phase in (
            _TerminalPhase.FATAL_PENDING,
            _TerminalPhase.FATAL_PUBLISHED,
        ):
            error = RuntimeError("runtime is fatal")
            if self._fatal_transition is not None:
                raise error from self._fatal_transition.primary
            raise error
        if self._phase is _TerminalPhase.RUNTIME_CLOSING:
            raise RuntimeError("runtime is closing")
        if self._phase is _TerminalPhase.RUNTIME_CLOSED:
            raise RuntimeError("runtime is closed")

    def _require_no_nested_token(self, thread_id):
        if thread_id in self._thread_tokens:
            raise RuntimeError("nested resource admission is not allowed")

    def _require_no_held_admission(self, action):
        if threading.get_ident() in self._thread_tokens:
            raise RuntimeError(
                "cannot {} while holding an admission".format(action)
            )

    def _has_closing_lease(self):
        return any(lease.phase == "closing" for lease in self._leases.values())

    def _has_active_tokens(self):
        return any(state.status == "active" for state in self._tokens.values())

    def _has_active_scope(self, scope):
        return any(
            state.status == "active" and state.token.scope == scope
            for state in self._tokens.values()
        )

    def _new_token(
        self,
        scope,
        epoch,
        operation,
        *,
        parent_sequence=None,
        transition_sequence=None,
        async_capability=_MISSING,
    ):
        self._require_operation(operation)
        thread_id = threading.get_ident()
        if parent_sequence is None:
            self._require_no_nested_token(thread_id)
        token = _ResourceAdmission(
            gate_id=self._gate_id,
            scope=scope,
            epoch=epoch,
            thread_id=thread_id,
            operation=operation,
            sequence=self._sequence(),
            parent_sequence=parent_sequence,
            transition_sequence=transition_sequence,
        )
        self._tokens[token.sequence] = _TokenState(
            token=token,
            async_capability=async_capability,
        )
        if parent_sequence is None:
            self._thread_tokens[thread_id] = token.sequence
        return token

    def _token_state(self, token):
        if not isinstance(token, _ResourceAdmission):
            raise TypeError("resource admission token is required")
        if token.gate_id != self._gate_id:
            raise RuntimeError("resource admission belongs to another terminal gate")
        state = self._tokens.get(token.sequence)
        if state is None:
            raise RuntimeError("resource admission token is unknown")
        if state.status == "released":
            raise RuntimeError("resource admission token was already released")
        if state.status == "converted":
            raise RuntimeError("resource admission token was already converted")
        if state.token != token:
            raise RuntimeError("stale resource admission token")
        return state

    def _require_token_thread(self, state):
        thread_id = threading.get_ident()
        if state.token.thread_id != thread_id:
            raise RuntimeError("resource admission belongs to another thread")
        if state.async_capability is _MISSING or state.async_claimed:
            if self._thread_tokens.get(thread_id) != state.token.sequence:
                raise RuntimeError("resource admission thread state is inconsistent")

    def _convertible_token_state(self, token):
        state = self._token_state(token)
        self._require_token_thread(state)
        if state.async_capability is not _MISSING and not state.async_claimed:
            raise RuntimeError("unclaimed async admission cannot be converted")
        return state

    def _recoverable_fatal_token_state(self, token):
        if not isinstance(token, _ResourceAdmission):
            raise TypeError("resource admission token is required")
        if token.gate_id != self._gate_id:
            raise RuntimeError("resource admission belongs to another terminal gate")
        state = self._tokens.get(token.sequence)
        if state is None:
            raise RuntimeError("resource admission token is unknown")
        if state.status == "released":
            raise RuntimeError("resource admission token was already released")
        if state.token != token:
            raise RuntimeError("stale resource admission token")
        if state.status == "active":
            self._require_token_thread(state)
            if state.async_capability is not _MISSING and not state.async_claimed:
                raise RuntimeError("unclaimed async admission cannot be converted")
        elif state.status == "converted":
            if state.token.thread_id != threading.get_ident():
                raise RuntimeError("resource admission belongs to another thread")
        else:
            raise RuntimeError("resource admission state is invalid")
        return state

    def _convert_token_state(self, state):
        state.status = "converted"
        state.async_capability = _MISSING
        self._thread_tokens.pop(state.token.thread_id, None)
        self._condition.notify_all()

    def _convert_token(self, token):
        state = self._convertible_token_state(token)
        self._convert_token_state(state)

    def _wait_until_deadline(
        self,
        predicate,
        deadline,
        message,
        *,
        owner_thread=None,
        owner_message=None,
    ):
        while not predicate():
            if owner_thread is not None and not owner_thread.is_alive():
                raise RuntimeError(owner_message or "terminal owner exited")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(message)
            self._condition.wait(remaining)

    def _wait_until(
        self,
        predicate,
        timeout_s,
        message,
        *,
        owner_thread=None,
        owner_message=None,
    ):
        self._wait_until_deadline(
            predicate,
            self._deadline(timeout_s),
            message,
            owner_thread=owner_thread,
            owner_message=owner_message,
        )

    def _require_fatal_transition(self, transition):
        if not isinstance(transition, _FatalTransition):
            raise TypeError("fatal transition is required")
        if transition.gate_id != self._gate_id:
            raise RuntimeError("fatal transition belongs to another terminal gate")
        if transition != self._fatal_transition:
            raise RuntimeError("stale fatal transition")

    def _lease_state(self, epoch):
        if type(epoch) is not int or epoch <= 0:
            raise RuntimeError("stale lease epoch")
        lease = self._leases.get(epoch)
        if lease is None:
            raise RuntimeError("stale lease epoch")
        return lease

    def _require_lease_transition(self, transition):
        if not isinstance(transition, _LeaseCloseTransition):
            raise TypeError("lease close transition is required")
        if transition.gate_id != self._gate_id:
            raise RuntimeError("lease close transition belongs to another terminal gate")
        lease = self._lease_state(transition.epoch)
        if lease.transition != transition:
            raise RuntimeError("stale lease close transition")
        return lease

    def _require_lease_close_owner(self, transition):
        if transition.owner_thread is not threading.current_thread():
            raise RuntimeError("only the elected lease close owner may run close steps")

    def _require_runtime_transition(self, transition):
        if not isinstance(transition, _RuntimeCloseTransition):
            raise TypeError("runtime close transition is required")
        if transition.gate_id != self._gate_id:
            raise RuntimeError("runtime close transition belongs to another terminal gate")
        if transition != self._runtime_close_transition:
            raise RuntimeError("stale runtime close transition")

    def admit_runtime(self, operation: str) -> _ResourceAdmission:
        with self._condition:
            current_thread = threading.current_thread()
            if operation == "begin_runtime_close":
                self._runtime_close_admission_rejections.pop(
                    current_thread, None
                )
            try:
                self._raise_for_terminal_phase()
                if self._has_closing_lease():
                    raise RuntimeError("lease is closing")
            except BaseException as error:
                if operation == "begin_runtime_close":
                    self._runtime_close_admission_rejections[
                        current_thread
                    ] = error
                raise
            return self._new_token("runtime", None, operation)

    def admit_runtime_setup(self, operation: str) -> _ResourceAdmission:
        with self._condition:
            self._raise_for_terminal_phase()
            if self._live_epoch is not None:
                raise RuntimeError("runtime setup requires no live lease")
            return self._new_token("runtime_setup", None, operation)

    def _prepare_lease_construction(self, operation, *, resource_names=()):
        self._require_operation(operation)
        transaction = _LeaseConstructionTransaction(
            gate_id=self._gate_id,
            operation=operation,
            owner_thread_id=threading.get_ident(),
            owner_thread=threading.current_thread(),
        )
        for name in tuple(resource_names):
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "lease construction resource name must be non-empty"
                )
            if name in transaction.resource_slots:
                raise ValueError("lease construction resource names must be unique")
            transaction.resource_slots[name] = _LeaseConstructionResourceSlot(
                self, transaction, name
            )
        return transaction

    def _require_lease_construction_resource_slot(self, slot):
        if not isinstance(slot, _LeaseConstructionResourceSlot):
            raise TypeError("lease construction resource slot is required")
        if slot._gate is not self:
            raise RuntimeError(
                "lease construction resource slot belongs to another terminal gate"
            )
        transaction = slot._transaction
        retained = transaction.resource_slots.get(slot.name)
        if retained is not slot:
            raise RuntimeError("stale lease construction resource slot")
        return transaction

    @staticmethod
    def _merge_construction_records(target, records):
        for record in tuple(records):
            identity = getattr(record, "identity", None)
            capacity = getattr(record, "capacity_bytes", None)
            if identity is None or capacity is None:
                raise TypeError("construction allocation record is invalid")
            retained = target.get(identity)
            if retained is None or capacity > retained.capacity_bytes:
                target[identity] = record

    def _publish_lease_construction_resource(
        self,
        slot,
        *,
        resource=None,
        kind=None,
        records=(),
        events=(),
        streams=(),
    ):
        with self._condition:
            transaction = self._require_lease_construction_resource_slot(slot)
            self._require_lease_construction(transaction, active=True)
            if resource is not None:
                if slot._resource is not None and slot._resource is not resource:
                    raise RuntimeError(
                        "lease construction resource identity changed"
                    )
                slot._resource = resource
            if kind is not None:
                if not isinstance(kind, str) or not kind:
                    raise ValueError("construction resource kind must be non-empty")
                if slot._kind is not None and slot._kind != kind:
                    raise RuntimeError("construction resource kind changed")
                slot._kind = kind
            self._merge_construction_records(slot._records, records)
            for event in tuple(events):
                if event is not None:
                    slot._events[id(event)] = event
            for stream in tuple(streams):
                if stream is not None:
                    slot._streams[id(stream)] = stream

    def _snapshot_lease_construction_resource(self, slot):
        with self._condition:
            self._require_lease_construction_resource_slot(slot)
            return _LeaseConstructionResourceSnapshot(
                name=slot.name,
                resource=slot._resource,
                kind=slot._kind,
                records=tuple(slot._records.values()),
                events=tuple(slot._events.values()),
                streams=tuple(slot._streams.values()),
            )

    def _clear_lease_construction_resource(self, slot):
        with self._condition:
            self._require_lease_construction_resource_slot(slot)
            slot._resource = None
            slot._kind = None
            slot._records.clear()
            slot._events.clear()
            slot._streams.clear()

    def _require_lease_construction(self, transaction, *, active=False):
        if not isinstance(transaction, _LeaseConstructionTransaction):
            raise TypeError("lease construction transaction is required")
        if transaction.gate_id != self._gate_id:
            raise RuntimeError(
                "lease construction transaction belongs to another terminal gate"
            )
        if transaction.owner_thread_id != threading.get_ident():
            raise RuntimeError(
                "lease construction transaction belongs to another thread"
            )
        if transaction.epoch is None:
            if transaction.token is not None or transaction.state not in {
                "prepared",
                "aborted",
                "fatal_retained",
            }:
                raise RuntimeError(
                    "lease construction transaction state is inconsistent"
                )
            if active:
                raise RuntimeError("lease construction transaction has not begun")
            return None
        lease = self._lease_state(transaction.epoch)
        if lease.construction is not transaction:
            raise RuntimeError("stale lease construction transaction")
        if active and transaction.state != "active":
            raise RuntimeError("lease construction transaction is already resolved")
        return lease

    def begin_lease(
        self,
        operation: str,
        *,
        _transaction: _LeaseConstructionTransaction | None = None,
    ) -> tuple[int, _ResourceAdmission]:
        transaction = (
            self._prepare_lease_construction(operation)
            if _transaction is None
            else _transaction
        )
        with self._condition:
            self._require_operation(operation)
            self._require_lease_construction(transaction)
            if transaction.operation != operation:
                raise RuntimeError("lease construction operation changed")
            self._raise_for_terminal_phase()
            if self._live_epoch is not None:
                raise RuntimeError("a lease epoch is already live")
            if self._has_active_scope("runtime_setup"):
                raise RuntimeError("runtime setup is active")
            if self._has_active_scope("runtime"):
                raise RuntimeError("runtime admission is active")
            self._require_no_nested_token(threading.get_ident())
            epoch = self._next_epoch
            self._next_epoch += 1
            self._leases[epoch] = _LeaseState(
                "constructing", construction=transaction
            )
            self._live_epoch = epoch
            transaction.epoch = epoch
            transaction.state = "active"
            expected_sequence = self._next_sequence
            try:
                token = self._new_token("construction", epoch, operation)
            except BaseException:
                token_state = self._tokens.get(expected_sequence)
                if (
                    token_state is not None
                    and token_state.token.scope == "construction"
                    and token_state.token.epoch == epoch
                    and token_state.token.operation == operation
                ):
                    transaction.token = token_state.token
                raise
            transaction.token = token
            return epoch, token

    @staticmethod
    def _remember_construction_secondary_locked(transaction, error, primary):
        if (
            not isinstance(error, BaseException)
            or error is primary
            or any(retained is error for retained in transaction.secondaries)
        ):
            return
        transaction.secondaries = (*transaction.secondaries, error)

    def _consume_construction_token_locked(self, transaction):
        token = transaction.token
        if token is None:
            raise RuntimeError("lease construction admission is unavailable")
        state = self._tokens.get(token.sequence)
        if state is None or state.token != token:
            raise RuntimeError("lease construction admission is unknown or stale")
        if state.status == "released":
            return
        if state.status not in {"active", "converted"}:
            raise RuntimeError("lease construction admission is not consumable")
        if token.thread_id != threading.get_ident():
            raise RuntimeError("lease construction admission belongs to another thread")
        state.status = "released"
        state.async_capability = _MISSING
        self._thread_tokens.pop(token.thread_id, None)

    def _retain_failed_lease_construction_locked(
        self,
        transaction,
        primary,
        finalizer,
        *,
        secondary=None,
    ):
        lease = self._require_lease_construction(
            transaction, active=transaction.state == "active"
        )
        token = transaction.token
        token_state = (
            None if token is None else self._tokens.get(token.sequence)
        )
        discovering_state = (
            token_state
            if token_state is not None and token_state.status == "active"
            else None
        )
        transition = self._begin_fatal_locked(primary, discovering_state)
        canonical = transition.primary
        self._remember_construction_secondary_locked(
            transaction, primary, canonical
        )
        self._remember_construction_secondary_locked(
            transaction, secondary, canonical
        )
        try:
            finalizer(canonical)
        except BaseException as error:
            self._remember_construction_secondary_locked(
                transaction, error, canonical
            )
        if lease is not None:
            lease.phase = "fatal_retained"
        transaction.state = "fatal_retained"
        transaction.primary = canonical
        transaction.result = canonical
        if token is not None:
            self._consume_construction_token_locked(transaction)
        self._condition.notify_all()
        return canonical

    def _fail_lease_construction(
        self,
        transaction,
        primary,
        finalizer,
        *,
        secondary=None,
    ):
        if not isinstance(primary, BaseException):
            raise TypeError("lease construction failure must be an exception")
        if not callable(finalizer):
            raise TypeError("lease construction fatal finalizer must be callable")
        with self._condition:
            if transaction.state not in {"prepared", "active"}:
                self._require_lease_construction(transaction)
                return transaction.primary
            return self._retain_failed_lease_construction_locked(
                transaction,
                primary,
                finalizer,
                secondary=secondary,
            )

    def _abort_lease_construction(
        self,
        transaction,
        primary,
        finalizer,
        fatal_finalizer,
    ):
        if not isinstance(primary, BaseException):
            raise TypeError("lease construction rollback primary must be an exception")
        if not callable(finalizer) or not callable(fatal_finalizer):
            raise TypeError("lease construction finalizers must be callable")
        with self._condition:
            if transaction.state not in {"prepared", "active"}:
                self._require_lease_construction(transaction)
                return transaction.primary
            lease = self._require_lease_construction(
                transaction, active=transaction.state == "active"
            )
            if self._fatal_transition is not None or not isinstance(
                primary, Exception
            ):
                return self._retain_failed_lease_construction_locked(
                    transaction,
                    primary,
                    fatal_finalizer,
                )
            try:
                result = finalizer()
            except BaseException as error:
                return self._retain_failed_lease_construction_locked(
                    transaction,
                    primary,
                    fatal_finalizer,
                    secondary=error,
                )
            if lease is not None:
                lease.phase = "closed"
                lease.result = primary
                lease.has_result = True
            transaction.state = "aborted"
            transaction.primary = primary
            transaction.result = result
            if lease is not None:
                self._live_epoch = None
                if transaction.token is not None:
                    self._consume_construction_token_locked(transaction)
            self._condition.notify_all()
            return primary

    def _commit_lease_construction(
        self,
        transaction,
        result,
        finalizer,
        fatal_finalizer,
    ):
        if not callable(finalizer) or not callable(fatal_finalizer):
            raise TypeError("lease construction finalizers must be callable")
        with self._condition:
            if transaction.state != "active":
                self._require_lease_construction(transaction)
                return transaction.result
            lease = self._require_lease_construction(transaction, active=True)
            if self._fatal_transition is not None:
                return self._retain_failed_lease_construction_locked(
                    transaction,
                    RuntimeError("fatal transition preempted lease construction"),
                    fatal_finalizer,
                )
            if self._phase is not _TerminalPhase.HEALTHY:
                self._raise_for_terminal_phase()
            try:
                finalizer()
            except BaseException as error:
                return self._retain_failed_lease_construction_locked(
                    transaction,
                    error,
                    fatal_finalizer,
                )
            lease.phase = "open"
            transaction.state = "committed"
            transaction.result = result
            self._consume_construction_token_locked(transaction)
            self._condition.notify_all()
            return result

    def _lease_construction_outcome(self, transaction):
        with self._condition:
            self._require_lease_construction(transaction)
            return (
                transaction.state,
                transaction.primary,
                transaction.result,
                transaction.secondaries,
            )

    def activate_lease(self, epoch: int, token: _ResourceAdmission) -> None:
        with self._condition:
            self._raise_for_terminal_phase()
            lease = self._lease_state(epoch)
            if lease.phase != "constructing" or self._live_epoch != epoch:
                raise RuntimeError("lease epoch is not constructing")
            state = self._token_state(token)
            self._require_token_thread(state)
            if token.scope != "construction" or token.epoch != epoch:
                raise RuntimeError("lease activation requires its construction admission")
            lease.phase = "open"
            self._condition.notify_all()

    def admit_lease(self, epoch: int, operation: str) -> _ResourceAdmission:
        with self._condition:
            self._raise_for_terminal_phase()
            lease = self._lease_state(epoch)
            if lease.phase == "closed" or self._live_epoch != epoch:
                raise RuntimeError("stale lease epoch")
            if lease.phase == "closing":
                raise RuntimeError("lease is closing")
            if lease.phase != "open":
                raise RuntimeError("lease is not open")
            return self._new_token("lease", epoch, operation)

    def begin_lease_close(self, epoch: int) -> tuple[_LeaseCloseTransition, bool]:
        with self._condition:
            lease = self._lease_state(epoch)
            if lease.transition is not None:
                return lease.transition, False
            if lease.phase == "closed" or self._live_epoch != epoch:
                raise RuntimeError("stale lease epoch")
            if self._phase in (
                _TerminalPhase.FATAL_PENDING,
                _TerminalPhase.FATAL_PUBLISHED,
                _TerminalPhase.RUNTIME_CLOSED,
            ):
                self._raise_for_terminal_phase()
            transition = _LeaseCloseTransition(
                gate_id=self._gate_id,
                epoch=epoch,
                owner_thread_id=threading.get_ident(),
                owner_thread=threading.current_thread(),
                sequence=self._sequence(),
            )
            lease.phase = "closing"
            lease.transition = transition
            self._condition.notify_all()
            return transition, True

    def _recover_lease_close_entry(self, epoch):
        """Return an installed lease-close election without adopting its owner."""
        with self._condition:
            lease = self._lease_state(epoch)
            transition = lease.transition
            if transition is None:
                return None, False
            return (
                transition,
                transition.owner_thread is threading.current_thread(),
            )

    def wait_for_lease_admissions(self, transition, timeout_s) -> None:
        with self._condition:
            self._require_lease_transition(transition)
            self._require_no_held_admission("drain admissions")
            self._wait_until(
                lambda: not self._has_active_tokens(),
                timeout_s,
                "lease admission drain timed out",
            )

    def admit_lease_close(
        self, transition, operation: str
    ) -> _ResourceAdmission:
        with self._condition:
            lease = self._require_lease_transition(transition)
            self._require_lease_close_owner(transition)
            if self._phase in (
                _TerminalPhase.FATAL_PENDING,
                _TerminalPhase.FATAL_PUBLISHED,
            ):
                raise RuntimeError("fatal transition preempted lease close") from (
                    self._fatal_transition.primary
                    if self._fatal_transition is not None
                    else None
                )
            if self._phase is _TerminalPhase.RUNTIME_CLOSED:
                raise RuntimeError("runtime is closed")
            if lease.phase != "closing":
                raise RuntimeError("lease is not closing")
            if self._has_active_tokens():
                raise RuntimeError("lease admissions have not drained")
            return self._new_token(
                "lease_close",
                transition.epoch,
                operation,
                transition_sequence=transition.sequence,
            )

    def wait_for_lease_closed(self, transition, timeout_s) -> object:
        with self._condition:
            lease = self._require_lease_transition(transition)
            self._require_no_held_admission("join lease close")
            deadline = self._deadline(timeout_s)
            self._wait_until_deadline(
                lambda: (
                    lease.phase == "closed"
                    or self._phase
                    in (
                        _TerminalPhase.FATAL_PENDING,
                        _TerminalPhase.FATAL_PUBLISHED,
                    )
                    or (
                        self._phase is _TerminalPhase.RUNTIME_CLOSED
                        and self._fatal_transition is not None
                    )
                ),
                deadline,
                "lease close join timed out",
                owner_thread=transition.owner_thread,
                owner_message="lease close owner exited",
            )
            if lease.phase != "closed":
                raise RuntimeError("fatal transition preempted lease close") from (
                    self._fatal_transition.primary
                    if self._fatal_transition is not None
                    else None
                )
            return lease.result

    def commit_lease_close(self, transition, finalizer) -> object:
        if not callable(finalizer):
            raise TypeError("lease close finalizer must be callable")
        with self._condition:
            lease = self._require_lease_transition(transition)
            self._require_lease_close_owner(transition)
            if self._phase in (
                _TerminalPhase.FATAL_PENDING,
                _TerminalPhase.FATAL_PUBLISHED,
            ):
                raise RuntimeError("fatal transition preempted lease close") from (
                    self._fatal_transition.primary
                    if self._fatal_transition is not None
                    else None
                )
            if self._phase is _TerminalPhase.RUNTIME_CLOSED:
                raise RuntimeError("runtime is closed")
            if lease.phase != "closing":
                raise RuntimeError("lease is not closing")
            if self._has_active_tokens():
                raise RuntimeError("lease close commit requires zero live admissions")
            result = finalizer()
            lease.phase = "closed"
            lease.result = result
            lease.has_result = True
            self._live_epoch = None
            self._condition.notify_all()
            return result

    def _fail_elected_lease_close(self, transition, primary) -> None:
        """Complete a failed elected close without invoking resource callbacks."""
        if not isinstance(primary, BaseException):
            raise TypeError("failed lease close primary must be an exception")
        with self._condition:
            lease = self._require_lease_transition(transition)
            self._require_lease_close_owner(transition)
            if self._has_active_tokens():
                raise RuntimeError(
                    "failed lease close completion requires zero live admissions"
                )
            if lease.phase == "closed":
                if not lease.has_result or lease.result is not primary:
                    raise RuntimeError("failed lease close primary changed")
            elif lease.phase in {"closing", "fatal_retained"}:
                lease.phase = "closed"
                lease.result = primary
                lease.has_result = True
            else:
                raise RuntimeError("lease close is not completable")
            if self._live_epoch == transition.epoch:
                self._live_epoch = None
            self._condition.notify_all()

    def spawn_async(
        self, parent: _ResourceAdmission, owner_identity: object
    ) -> _ResourceAdmission:
        with self._condition:
            state = self._token_state(parent)
            self._require_token_thread(state)
            if state.async_capability is not _MISSING:
                raise RuntimeError("async descendants cannot spawn nested descendants")
            if any(
                retained.token.parent_sequence == parent.sequence
                and retained.async_capability is owner_identity
                for retained in self._tokens.values()
            ):
                raise RuntimeError(
                    "async claim capability already has a descendant admission"
                )
            if parent.scope in ("runtime", "runtime_setup"):
                self._raise_for_terminal_phase()
                if self._has_closing_lease():
                    raise RuntimeError("lease is closing")
            elif parent.scope == "construction":
                self._raise_for_terminal_phase()
                lease = self._lease_state(parent.epoch)
                if lease.phase not in ("constructing", "open"):
                    raise RuntimeError("construction epoch is closing")
            elif parent.scope == "lease":
                self._raise_for_terminal_phase()
                lease = self._lease_state(parent.epoch)
                if lease.phase != "open":
                    raise RuntimeError("lease is closing")
            elif parent.scope == "lease_close":
                lease = self._lease_state(parent.epoch)
                if (
                    lease.phase != "closing"
                    or lease.transition is None
                    or parent.transition_sequence != lease.transition.sequence
                ):
                    raise RuntimeError("stale lease close admission")
                if self._phase in (
                    _TerminalPhase.FATAL_PENDING,
                    _TerminalPhase.FATAL_PUBLISHED,
                    _TerminalPhase.RUNTIME_CLOSED,
                ):
                    self._raise_for_terminal_phase()
            elif parent.scope == "runtime_close":
                if (
                    self._phase is not _TerminalPhase.RUNTIME_CLOSING
                    or self._runtime_close_transition is None
                    or parent.transition_sequence
                    != self._runtime_close_transition.sequence
                ):
                    raise RuntimeError("stale runtime close admission")
            return self._new_token(
                parent.scope,
                parent.epoch,
                parent.operation,
                parent_sequence=parent.sequence,
                transition_sequence=parent.transition_sequence,
                async_capability=owner_identity,
            )

    def claim_async(
        self,
        token: _ResourceAdmission,
        owner_identity: object,
        operation: str,
    ) -> _ResourceAdmission:
        with self._condition:
            self._require_operation(operation)
            state = self._token_state(token)
            if state.async_capability is _MISSING:
                raise RuntimeError("resource admission is not an async descendant")
            if state.async_capability is not owner_identity:
                raise RuntimeError("async admission claim capability does not match")
            if state.async_claimed:
                raise RuntimeError("async admission was already claimed")
            thread_id = threading.get_ident()
            self._require_no_nested_token(thread_id)
            claimed = replace(token, thread_id=thread_id, operation=operation)
            state.token = claimed
            state.async_claimed = True
            self._thread_tokens[thread_id] = claimed.sequence
            return claimed

    def cancel_async(self, token, owner_identity) -> bool:
        """Release one unclaimed descendant from any terminal-management thread."""
        with self._condition:
            state = self._token_state(token)
            if state.async_capability is _MISSING:
                raise RuntimeError("resource admission is not an async descendant")
            if state.async_capability is not owner_identity:
                raise RuntimeError(
                    "async admission cancel capability does not match"
                )
            if state.async_claimed:
                return False
            state.status = "released"
            state.async_capability = _MISSING
            self._condition.notify_all()
            return True

    def release(self, token: _ResourceAdmission) -> None:
        with self._condition:
            state = self._token_state(token)
            self._require_token_thread(state)
            state.status = "released"
            if state.async_capability is _MISSING or state.async_claimed:
                self._thread_tokens.pop(state.token.thread_id, None)
            state.async_capability = _MISSING
            self._condition.notify_all()

    def _current_thread_admission(self) -> _ResourceAdmission | None:
        with self._condition:
            sequence = self._thread_tokens.get(threading.get_ident())
            if sequence is None:
                return None
            state = self._tokens.get(sequence)
            if state is None or state.status != "active":
                raise RuntimeError("resource admission thread state is inconsistent")
            return state.token

    def _recover_current_thread_admission(
        self,
        *,
        scope,
        operation,
        epoch=None,
        transition=None,
    ):
        """Recover only an exact active token installed by the current thread."""
        with self._condition:
            self._require_operation(operation)
            sequence = self._thread_tokens.get(threading.get_ident())
            if sequence is None:
                return None
            state = self._tokens.get(sequence)
            if state is None or state.status != "active":
                raise RuntimeError("resource admission thread state is inconsistent")
            token = state.token
            expected_transition = None
            if transition is not None:
                if scope == "lease_close":
                    self._require_lease_transition(transition)
                elif scope == "runtime_close":
                    self._require_runtime_transition(transition)
                else:
                    raise RuntimeError(
                        "only close admissions may bind a transition"
                    )
                expected_transition = transition.sequence
            if (
                token.scope != scope
                or token.epoch != epoch
                or token.operation != operation
                or token.transition_sequence != expected_transition
                or token.parent_sequence is not None
            ):
                raise RuntimeError(
                    "current resource admission does not match close recovery"
                )
            self._require_token_thread(state)
            return token

    def _begin_fatal_locked(
        self,
        primary,
        discovering_state,
        *,
        prepared_transition=None,
    ):
        if (
            self._phase is _TerminalPhase.RUNTIME_CLOSED
            and self._fatal_transition is None
        ):
            raise RuntimeError("runtime is closed")
        if self._runtime_close_commit_selected and self._fatal_transition is None:
            if (
                discovering_state is not None
                and discovering_state.status == "active"
            ):
                self._convert_token_state(discovering_state)
                self._condition.notify_all()
            raise RuntimeError("runtime close is committed")
        if self._fatal_transition is None:
            if prepared_transition is None:
                prepared_transition = _FatalTransition(
                    gate_id=self._gate_id,
                    primary=primary,
                    sequence=self._sequence(),
                )
            if (
                not isinstance(prepared_transition, _FatalTransition)
                or prepared_transition.gate_id != self._gate_id
                or prepared_transition.primary is not primary
            ):
                raise RuntimeError("prepared fatal transition is invalid")
            self._fatal_transition = prepared_transition
        if self._phase in (
            _TerminalPhase.HEALTHY,
            _TerminalPhase.RUNTIME_CLOSING,
        ):
            self._phase = _TerminalPhase.FATAL_PENDING
            if self._live_epoch is not None:
                self._leases[self._live_epoch].phase = "fatal_retained"
        if (
            discovering_state is not None
            and discovering_state.status == "active"
        ):
            self._convert_token_state(discovering_state)
        self._condition.notify_all()
        return self._fatal_transition

    def begin_fatal(
        self,
        primary: BaseException,
        discovering_token: _ResourceAdmission | None = None,
    ) -> _FatalTransition:
        if not isinstance(primary, BaseException):
            raise TypeError("fatal primary must be an exception")
        with self._condition:
            discovering_state = None
            if discovering_token is not None:
                discovering_state = self._convertible_token_state(discovering_token)
            return self._begin_fatal_locked(primary, discovering_state)

    def _recover_fatal(
        self,
        primary,
        discovering_token=None,
        *,
        prepared_transition=None,
    ):
        if not isinstance(primary, BaseException):
            raise TypeError("fatal primary must be an exception")
        with self._condition:
            discovering_state = None
            if discovering_token is not None:
                discovering_state = self._recoverable_fatal_token_state(
                    discovering_token
                )
            transition = self._begin_fatal_locked(
                primary,
                discovering_state,
                prepared_transition=prepared_transition,
            )
            if transition.primary is not primary:
                raise RuntimeError("fatal primary changed during recovery")
            return transition

    def _terminalize_fatal_failure(
        self,
        primary,
        discovering_token=None,
    ):
        """Install one canonical fatal failure outcome without callbacks."""
        if not isinstance(primary, BaseException):
            raise TypeError("fatal primary must be an exception")
        with self._condition:
            discovering_state = None
            if discovering_token is not None:
                discovering_state = self._recoverable_fatal_token_state(
                    discovering_token
                )
            transition = self._begin_fatal_locked(primary, discovering_state)
            primary = transition.primary
            if self._fatal_snapshot is _MISSING:
                if self._fatal_publication_failure is _MISSING:
                    self._fatal_publication_failure = primary
                elif self._fatal_publication_failure is not primary:
                    raise RuntimeError("fatal publication failure changed")
            self._condition.notify_all()
            return transition

    def wait_for_admissions(
        self, transition: _FatalTransition, timeout_s: float
    ) -> None:
        with self._condition:
            self._require_fatal_transition(transition)
            self._require_no_held_admission("drain admissions")
            self._wait_until(
                lambda: not self._has_active_tokens(),
                timeout_s,
                "fatal admission drain timed out",
            )

    def publish_fatal(self, transition, snapshot) -> None:
        with self._condition:
            self._require_fatal_transition(transition)
            if self._phase is not _TerminalPhase.FATAL_PENDING:
                raise RuntimeError("fatal transition is not pending")
            self._require_no_held_admission("publish fatal")
            if self._has_active_tokens():
                raise RuntimeError("fatal publication requires zero live admissions")
            self._wait_until(
                lambda: self._fatal_runtime_finalizer_state != "pending",
                _TERMINAL_TIMEOUT_S,
                "fatal runtime finalizer wait timed out",
                owner_thread=self._fatal_runtime_finalizer_thread,
                owner_message="fatal runtime finalizer owner exited",
            )
            if self._fatal_publication_failure is not _MISSING:
                raise self._fatal_publication_failure
            if self._phase is not _TerminalPhase.FATAL_PENDING:
                raise RuntimeError("fatal transition is not pending")
            self._fatal_snapshot = snapshot
            self._phase = _TerminalPhase.FATAL_PUBLISHED
            self._condition.notify_all()

    def _fail_fatal_publication(self, transition, failure):
        if not isinstance(failure, BaseException):
            raise TypeError("fatal publication failure must be an exception")
        with self._condition:
            self._require_fatal_transition(transition)
            if failure is not transition.primary:
                raise RuntimeError("fatal publication primary changed")
            if self._fatal_snapshot is not _MISSING:
                return
            if self._fatal_publication_failure is _MISSING:
                self._fatal_publication_failure = failure
            elif self._fatal_publication_failure is not failure:
                raise RuntimeError("fatal publication failure changed")
            self._condition.notify_all()

    def _fatal_publication_failed(self, transition, failure):
        with self._condition:
            self._require_fatal_transition(transition)
            return self._fatal_publication_failure is failure

    def wait_for_published(self, timeout_s: float):
        with self._condition:
            self._require_no_held_admission("wait for fatal publication")
            self._wait_until(
                lambda: (
                    self._fatal_snapshot is not _MISSING
                    or self._fatal_publication_failure is not _MISSING
                ),
                timeout_s,
                "fatal publication wait timed out",
            )
            if self._fatal_publication_failure is not _MISSING:
                raise self._fatal_publication_failure
            return self._fatal_snapshot

    def _freeze_runtime_close(
        self, discovering_token: _ResourceAdmission | None
    ):
        with self._condition:
            thread_id = threading.get_ident()
            discovering_state = None
            if discovering_token is not None:
                discovering_state = self._convertible_token_state(discovering_token)
                if discovering_token.scope != "runtime":
                    raise RuntimeError(
                        "runtime close requires an initiating runtime admission"
                    )
            elif thread_id in self._thread_tokens:
                self._require_no_held_admission("begin runtime close")

            if self._phase is _TerminalPhase.RUNTIME_CLOSED:
                if self._fatal_transition is not None:
                    return self._fatal_transition, False
                return self._runtime_close_transition, False
            if self._phase in (
                _TerminalPhase.FATAL_PENDING,
                _TerminalPhase.FATAL_PUBLISHED,
            ):
                if discovering_state is not None:
                    self._convert_token_state(discovering_state)
                return self._fatal_transition, False
            elected = False
            if self._phase is _TerminalPhase.HEALTHY:
                self._runtime_close_transition = _RuntimeCloseTransition(
                    gate_id=self._gate_id,
                    owner_thread_id=thread_id,
                    owner_thread=threading.current_thread(),
                    sequence=self._sequence(),
                )
                self._phase = _TerminalPhase.RUNTIME_CLOSING
                elected = True
            if discovering_state is not None:
                self._convert_token_state(discovering_state)
            self._condition.notify_all()
            return self._runtime_close_transition, elected

    def _recover_runtime_close_entry(self):
        """Recover the current runtime/fatal transition without owner adoption."""
        with self._condition:
            if self._fatal_transition is not None:
                return self._fatal_transition, False
            transition = self._runtime_close_transition
            if transition is None:
                return None, False
            return (
                transition,
                transition.owner_thread is threading.current_thread(),
            )

    def _classify_runtime_close_admission_failure(self, error, token):
        """Separate expected gate rejection from a fallible wrapper failure."""
        if not isinstance(error, BaseException):
            raise TypeError("runtime close entry failure must be an exception")
        with self._condition:
            rejection = self._runtime_close_admission_rejections.pop(
                threading.current_thread(), None
            )
            if rejection is error:
                return None
            if token is not None or self._fatal_transition is not None:
                return error
            if self._phase in (
                _TerminalPhase.RUNTIME_CLOSING,
                _TerminalPhase.RUNTIME_CLOSED,
            ) or self._has_closing_lease():
                return None
            return error

    def _drain_runtime_close(
        self,
        transition,
        *,
        timeout_s=_TERMINAL_TIMEOUT_S,
    ):
        with self._condition:
            deadline = self._deadline(timeout_s)
            self._require_no_held_admission("drain runtime close")
            if isinstance(transition, _FatalTransition):
                self._require_fatal_transition(transition)
                return transition
            if self._phase in (
                _TerminalPhase.FATAL_PENDING,
                _TerminalPhase.FATAL_PUBLISHED,
            ):
                return self._fatal_transition
            if self._phase is _TerminalPhase.RUNTIME_CLOSED:
                if self._fatal_transition is not None:
                    return self._fatal_transition
                return self._runtime_close_transition
            self._require_runtime_transition(transition)

            self._wait_until_deadline(
                lambda: (
                    not self._has_active_tokens()
                    or self._fatal_transition is not None
                ),
                deadline,
                "runtime close admission drain timed out",
            )
            if self._fatal_transition is not None:
                return self._fatal_transition

            if self._live_epoch is not None:
                lease = self._leases[self._live_epoch]
                if (
                    lease.phase == "closing"
                    and lease.transition is not None
                    and lease.transition.owner_thread
                    is not threading.current_thread()
                ):
                    self._wait_until_deadline(
                        lambda: (
                            lease.phase == "closed"
                            or self._fatal_transition is not None
                        ),
                        deadline,
                        "foreign lease close wait timed out",
                        owner_thread=lease.transition.owner_thread,
                        owner_message="lease close owner exited",
                    )
            if self._fatal_transition is not None:
                return self._fatal_transition
            return self._runtime_close_transition

    def begin_runtime_close(
        self, discovering_token: _ResourceAdmission | None
    ):
        transition, _ = self._freeze_runtime_close(discovering_token)
        return self._drain_runtime_close(transition)

    def admit_runtime_close(self, transition, operation: str):
        with self._condition:
            if self._phase in (
                _TerminalPhase.FATAL_PENDING,
                _TerminalPhase.FATAL_PUBLISHED,
            ):
                raise RuntimeError("fatal transition preempted runtime close") from (
                    self._fatal_transition.primary
                    if self._fatal_transition is not None
                    else None
                )
            self._require_runtime_transition(transition)
            if transition.owner_thread is not threading.current_thread():
                raise RuntimeError(
                    "only the elected runtime close owner may run close steps"
                )
            if self._phase is _TerminalPhase.RUNTIME_CLOSED:
                raise RuntimeError("runtime is closed")
            if self._phase is not _TerminalPhase.RUNTIME_CLOSING:
                raise RuntimeError("runtime is not closing")
            if self._live_epoch is not None:
                raise RuntimeError("runtime close requires no live lease")
            if self._has_active_tokens():
                raise RuntimeError("runtime admissions have not drained")
            return self._new_token(
                "runtime_close",
                None,
                operation,
                transition_sequence=transition.sequence,
            )

    def _require_runtime_close_admission(self, token, operation):
        """Prove one exact elected-owner token for a destructive close step."""
        with self._condition:
            self._require_operation(operation)
            if self._fatal_transition is not None and self._phase in (
                _TerminalPhase.FATAL_PENDING,
                _TerminalPhase.FATAL_PUBLISHED,
                _TerminalPhase.RUNTIME_CLOSED,
            ):
                raise self._fatal_transition.primary
            transition = self._runtime_close_transition
            self._require_runtime_transition(transition)
            if transition.owner_thread is not threading.current_thread():
                raise RuntimeError(
                    "only the elected runtime close owner may run close steps"
                )
            if self._phase is not _TerminalPhase.RUNTIME_CLOSING:
                raise RuntimeError("runtime is not closing")
            state = self._token_state(token)
            self._require_token_thread(state)
            if (
                token.scope != "runtime_close"
                or token.epoch is not None
                or token.operation != operation
                or token.parent_sequence is not None
                or token.transition_sequence != transition.sequence
            ):
                raise RuntimeError(
                    "runtime close admission does not match the elected step"
                )
            return token

    def select_runtime_close_commit(self, transition):
        """Make healthy runtime close irrevocable before destructive close work."""
        with self._condition:
            self._require_runtime_transition(transition)
            if transition.owner_thread is not threading.current_thread():
                raise RuntimeError(
                    "only the elected runtime close owner may select close commit"
                )
            if self._phase in (
                _TerminalPhase.FATAL_PENDING,
                _TerminalPhase.FATAL_PUBLISHED,
            ):
                return self._fatal_transition
            if self._phase is _TerminalPhase.RUNTIME_CLOSED:
                return transition
            if self._phase is not _TerminalPhase.RUNTIME_CLOSING:
                raise RuntimeError("runtime is not closing")
            if self._live_epoch is not None:
                raise RuntimeError("runtime close commit requires no live lease")
            if self._has_active_tokens():
                raise RuntimeError(
                    "runtime close commit requires zero live admissions"
                )
            self._runtime_close_commit_selected = True
            self._condition.notify_all()
            return transition

    def _commit_fatal_runtime_close(
        self,
        transition,
        finalizer,
        *,
        may_finalize=True,
        timeout_s=_TERMINAL_TIMEOUT_S,
    ):
        self._require_fatal_transition(transition)
        self._require_no_held_admission("commit runtime close")
        if self._phase is _TerminalPhase.RUNTIME_CLOSED:
            return self._runtime_close_result
        if self._phase is _TerminalPhase.FATAL_PUBLISHED:
            self._phase = _TerminalPhase.RUNTIME_CLOSED
            self._runtime_close_result = self._fatal_snapshot
            self._condition.notify_all()
            return self._runtime_close_result
        current_thread = threading.current_thread()
        deadline = self._deadline(timeout_s)
        if may_finalize and self._fatal_runtime_finalizer_state == "none":
            self._fatal_runtime_finalizer_state = "pending"
            self._fatal_runtime_finalizer_owner = current_thread
            self._fatal_runtime_finalizer_thread = current_thread
        if self._fatal_runtime_finalizer_owner is current_thread:
            self._wait_until_deadline(
                lambda: not self._has_active_tokens(),
                deadline,
                "fatal runtime close admission drain timed out",
            )
            finalizer()
            self._fatal_runtime_finalizer_state = "complete"
            self._condition.notify_all()
        else:
            self._wait_until_deadline(
                lambda: (
                    self._fatal_runtime_finalizer_state != "pending"
                    or self._fatal_snapshot is not _MISSING
                    or self._fatal_publication_failure is not _MISSING
                ),
                deadline,
                "fatal runtime finalizer join timed out",
                owner_thread=self._fatal_runtime_finalizer_thread,
                owner_message="fatal runtime finalizer owner exited",
            )
            if self._fatal_publication_failure is not _MISSING:
                raise self._fatal_publication_failure
        self._wait_until_deadline(
            lambda: (
                self._fatal_snapshot is not _MISSING
                or self._fatal_publication_failure is not _MISSING
            ),
            deadline,
            "fatal publication wait timed out",
        )
        if self._fatal_snapshot is _MISSING:
            if self._fatal_publication_failure is not _MISSING:
                raise self._fatal_publication_failure
        if self._phase is _TerminalPhase.RUNTIME_CLOSED:
            return self._runtime_close_result
        self._phase = _TerminalPhase.RUNTIME_CLOSED
        self._runtime_close_result = self._fatal_snapshot
        self._condition.notify_all()
        return self._runtime_close_result

    def commit_runtime_close(
        self,
        transition,
        finalizer,
        *,
        timeout_s=_TERMINAL_TIMEOUT_S,
    ):
        if not callable(finalizer):
            raise TypeError("runtime close finalizer must be callable")
        with self._condition:
            if isinstance(transition, _FatalTransition):
                return self._commit_fatal_runtime_close(
                    transition,
                    finalizer,
                    timeout_s=timeout_s,
                )
            self._require_runtime_transition(transition)
            if transition.owner_thread is not threading.current_thread():
                self._require_no_held_admission("join runtime close")
                self._wait_until(
                    lambda: self._phase is not _TerminalPhase.RUNTIME_CLOSING,
                    timeout_s,
                    "runtime close join timed out",
                    owner_thread=transition.owner_thread,
                    owner_message="runtime close owner exited",
                )
                if self._phase is _TerminalPhase.RUNTIME_CLOSED:
                    return self._runtime_close_result
                return self._commit_fatal_runtime_close(
                    self._fatal_transition,
                    lambda: None,
                    may_finalize=False,
                    timeout_s=timeout_s,
                )
            if self._phase in (
                _TerminalPhase.FATAL_PENDING,
                _TerminalPhase.FATAL_PUBLISHED,
            ):
                return self._commit_fatal_runtime_close(
                    self._fatal_transition,
                    finalizer,
                    timeout_s=timeout_s,
                )
            if self._phase is _TerminalPhase.RUNTIME_CLOSED:
                return self._runtime_close_result
            if self._live_epoch is not None:
                raise RuntimeError("runtime close commit requires no live lease")
            if self._has_active_tokens():
                raise RuntimeError("runtime close commit requires zero live admissions")
            self._runtime_close_commit_selected = True
            result = finalizer()
            self._phase = _TerminalPhase.RUNTIME_CLOSED
            self._runtime_close_result = result
            self._condition.notify_all()
            return result

    def _fail_selected_runtime_close(self, transition, primary, finalizer):
        """Finish an irrevocable healthy close after its elected owner fails."""
        if not isinstance(primary, BaseException):
            raise TypeError("runtime close failure must be an exception")
        if not callable(finalizer):
            raise TypeError("runtime close failure finalizer must be callable")
        with self._condition:
            self._require_runtime_transition(transition)
            if transition.owner_thread is not threading.current_thread():
                raise RuntimeError(
                    "only the elected runtime close owner may fail close"
                )
            if self._phase is _TerminalPhase.RUNTIME_CLOSED:
                return None
            if not self._runtime_close_commit_selected:
                raise RuntimeError("runtime close is not irrevocably selected")
            sequence = self._thread_tokens.get(threading.get_ident())
            if sequence is not None:
                self._convert_token_state(self._tokens[sequence])
            secondary = None
            try:
                finalizer()
            except BaseException as error:
                secondary = error
            self._phase = _TerminalPhase.RUNTIME_CLOSED
            self._runtime_close_result = primary
            self._condition.notify_all()
            return secondary

    def _complete_failed_fatal_runtime_close(self, transition, finalizer):
        """Close a runtime whose fatal publisher installed the failure marker."""
        if not callable(finalizer):
            raise TypeError("fatal runtime close finalizer must be callable")
        with self._condition:
            self._require_fatal_transition(transition)
            if self._fatal_publication_failure is not transition.primary:
                raise RuntimeError("fatal publication failure is not recorded")
            if self._phase is _TerminalPhase.RUNTIME_CLOSED:
                return None
            self._require_no_held_admission("complete failed fatal runtime close")
            current_thread = threading.current_thread()
            run_finalizer = self._fatal_runtime_finalizer_state == "none"
            if run_finalizer:
                self._fatal_runtime_finalizer_state = "pending"
                self._fatal_runtime_finalizer_owner = current_thread
                self._fatal_runtime_finalizer_thread = current_thread
            elif self._fatal_runtime_finalizer_owner is current_thread:
                run_finalizer = True
            secondary = None
            if run_finalizer:
                try:
                    finalizer()
                except BaseException as error:
                    secondary = error
            self._fatal_runtime_finalizer_state = "complete"
            self._phase = _TerminalPhase.RUNTIME_CLOSED
            self._runtime_close_result = transition.primary
            self._condition.notify_all()
            return secondary
