"""Resource-neutral coordination for distributed runtime terminal transitions."""

from dataclasses import dataclass, replace
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
    sequence: int


@dataclass(frozen=True)
class _RuntimeCloseTransition:
    gate_id: int
    owner_thread_id: int
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
            return None, read()

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
                if before_select is not None:
                    before_select()
                self._outcome = _MonitorOutcome(
                    kind="fatal_elected", primary=primary
                )
                self._condition.notify_all()
            return self._outcome

    def select_clean(self, generation: int) -> _MonitorOutcome:
        with self._condition:
            if generation != self._requested_generation:
                raise RuntimeError("monitor stop generation is not current")
            if self._outcome is None:
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
        self._fatal_runtime_finalizer_owner: int | None = None
        self._runtime_close_transition: _RuntimeCloseTransition | None = None
        self._runtime_close_commit_selected = False
        self._runtime_close_result = _MISSING

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

    def _wait_until(self, predicate, timeout_s, message):
        deadline = self._deadline(timeout_s)
        while not predicate():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(message)
            self._condition.wait(remaining)

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
        if transition.owner_thread_id != threading.get_ident():
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
            self._raise_for_terminal_phase()
            if self._has_closing_lease():
                raise RuntimeError("lease is closing")
            return self._new_token("runtime", None, operation)

    def admit_runtime_setup(self, operation: str) -> _ResourceAdmission:
        with self._condition:
            self._raise_for_terminal_phase()
            if self._live_epoch is not None:
                raise RuntimeError("runtime setup requires no live lease")
            return self._new_token("runtime_setup", None, operation)

    def begin_lease(self, operation: str) -> tuple[int, _ResourceAdmission]:
        with self._condition:
            self._require_operation(operation)
            self._raise_for_terminal_phase()
            if self._live_epoch is not None:
                raise RuntimeError("a lease epoch is already live")
            if self._has_active_scope("runtime_setup"):
                raise RuntimeError("runtime setup is active")
            self._require_no_nested_token(threading.get_ident())
            epoch = self._next_epoch
            self._next_epoch += 1
            self._leases[epoch] = _LeaseState("constructing")
            self._live_epoch = epoch
            token = self._new_token("construction", epoch, operation)
            return epoch, token

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
                sequence=self._sequence(),
            )
            lease.phase = "closing"
            lease.transition = transition
            self._condition.notify_all()
            return transition, True

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
            while lease.phase != "closed":
                if self._phase in (
                    _TerminalPhase.FATAL_PENDING,
                    _TerminalPhase.FATAL_PUBLISHED,
                ):
                    raise RuntimeError("fatal transition preempted lease close") from (
                        self._fatal_transition.primary
                        if self._fatal_transition is not None
                        else None
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("lease close join timed out")
                self._condition.wait(remaining)
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
            while self._fatal_runtime_finalizer_state == "pending":
                self._condition.wait()
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

    def begin_runtime_close(
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
                    return self._fatal_transition
                return self._runtime_close_transition
            if self._phase in (
                _TerminalPhase.FATAL_PENDING,
                _TerminalPhase.FATAL_PUBLISHED,
            ):
                if discovering_state is not None:
                    self._convert_token_state(discovering_state)
                return self._fatal_transition
            if self._phase is _TerminalPhase.HEALTHY:
                self._runtime_close_transition = _RuntimeCloseTransition(
                    gate_id=self._gate_id,
                    owner_thread_id=thread_id,
                    sequence=self._sequence(),
                )
                self._phase = _TerminalPhase.RUNTIME_CLOSING
            if discovering_state is not None:
                self._convert_token_state(discovering_state)
            self._condition.notify_all()

            while self._has_active_tokens():
                self._condition.wait()
                if self._fatal_transition is not None:
                    return self._fatal_transition
            if self._fatal_transition is not None:
                return self._fatal_transition

            if self._live_epoch is not None:
                lease = self._leases[self._live_epoch]
                if (
                    lease.phase == "closing"
                    and lease.transition is not None
                    and lease.transition.owner_thread_id != thread_id
                ):
                    while lease.phase != "closed":
                        self._condition.wait()
                        if self._fatal_transition is not None:
                            return self._fatal_transition
            if self._fatal_transition is not None:
                return self._fatal_transition
            return self._runtime_close_transition

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
            if transition.owner_thread_id != threading.get_ident():
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

    def select_runtime_close_commit(self, transition):
        """Make healthy runtime close irrevocable before destructive close work."""
        with self._condition:
            self._require_runtime_transition(transition)
            if transition.owner_thread_id != threading.get_ident():
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
        self, transition, finalizer, *, may_finalize=True
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
        thread_id = threading.get_ident()
        if may_finalize and self._fatal_runtime_finalizer_state == "none":
            self._fatal_runtime_finalizer_state = "pending"
            self._fatal_runtime_finalizer_owner = thread_id
        if self._fatal_runtime_finalizer_owner == thread_id:
            while self._has_active_tokens():
                self._condition.wait()
            finalizer()
            self._fatal_runtime_finalizer_state = "complete"
            self._condition.notify_all()
        else:
            while (
                self._fatal_runtime_finalizer_state == "pending"
                and self._fatal_snapshot is _MISSING
            ):
                if self._fatal_publication_failure is not _MISSING:
                    raise self._fatal_publication_failure
                self._condition.wait()
        while self._fatal_snapshot is _MISSING:
            if self._fatal_publication_failure is not _MISSING:
                raise self._fatal_publication_failure
            self._condition.wait()
        if self._phase is _TerminalPhase.RUNTIME_CLOSED:
            return self._runtime_close_result
        self._phase = _TerminalPhase.RUNTIME_CLOSED
        self._runtime_close_result = self._fatal_snapshot
        self._condition.notify_all()
        return self._runtime_close_result

    def commit_runtime_close(self, transition, finalizer):
        if not callable(finalizer):
            raise TypeError("runtime close finalizer must be callable")
        with self._condition:
            if isinstance(transition, _FatalTransition):
                return self._commit_fatal_runtime_close(transition, finalizer)
            self._require_runtime_transition(transition)
            if transition.owner_thread_id != threading.get_ident():
                self._require_no_held_admission("join runtime close")
                while self._phase is _TerminalPhase.RUNTIME_CLOSING:
                    self._condition.wait()
                if self._phase is _TerminalPhase.RUNTIME_CLOSED:
                    return self._runtime_close_result
                return self._commit_fatal_runtime_close(
                    self._fatal_transition, lambda: None, may_finalize=False
                )
            if self._phase in (
                _TerminalPhase.FATAL_PENDING,
                _TerminalPhase.FATAL_PUBLISHED,
            ):
                return self._commit_fatal_runtime_close(
                    self._fatal_transition, finalizer
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
