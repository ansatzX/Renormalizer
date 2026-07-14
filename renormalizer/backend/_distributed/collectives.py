"""Collective contracts and local collective behavior."""

from contextlib import contextmanager
from ctypes import sizeof
from dataclasses import dataclass
import os
import socket
import struct
import threading
import time
from typing import Any, Protocol, runtime_checkable
import weakref

from renormalizer.backend._distributed.terminal import (
    _FatalMonitorHandoff,
    _TERMINAL_TIMEOUT_S,
    _remaining_lifecycle_time,
)


_REDUCTION_OPS = frozenset({"sum", "prod", "min", "max"})
_STATUS_WORKSPACE_KEY_PREFIX = "renormalizer.status_workspace.rank."
_FATAL_KEY_PREFIX = "renormalizer.fatal.rank."
_FATAL_ACK_KEY_PREFIX = "renormalizer.fatal_ack.rank."
_ACTIVE_B_KEY_PREFIX = "renormalizer.active_b.rank."
_CLOSE_READY_KEY_PREFIX = "renormalizer.close_ready.rank."
_MONITOR_STOPPED_KEY_PREFIX = "renormalizer.monitor_stopped.rank."
_CLOSE_CONSUMED_KEY_PREFIX = "renormalizer.close_consumed.rank."
_CLOSE_RELEASE_KEY = "renormalizer.close_release"
_FATAL_CAPABILITY_ERROR = 4
_FATAL_TIMEOUT_S = _TERMINAL_TIMEOUT_S
_FATAL_EXIT_CODE = 86
_ACTIVE_B_RECORD = struct.Struct("!QB")
_FATAL_GATE_FAILURE_IDLE = "idle"
_FATAL_GATE_FAILURE_INFLIGHT = "inflight"
_FATAL_GATE_FAILURE_SIGNALED = "signaled"


@dataclass(frozen=True)
class _FatalGateFailureClaim:
    state: str
    owner: Any = None
    deadline: Any = None

    def __post_init__(self):
        if self.state == _FATAL_GATE_FAILURE_INFLIGHT:
            if not isinstance(self.owner, threading.Thread):
                raise TypeError("inflight gate failure claim requires a thread")
            if type(self.deadline) is not float:
                raise TypeError("inflight gate failure claim requires a deadline")
            return
        if self.state not in {
            _FATAL_GATE_FAILURE_IDLE,
            _FATAL_GATE_FAILURE_SIGNALED,
        }:
            raise ValueError("invalid gate failure claim state")
        if self.owner is not None or self.deadline is not None:
            raise ValueError("settled gate failure claim cannot retain an owner")


_FATAL_GATE_FAILURE_IDLE_CLAIM = _FatalGateFailureClaim(
    _FATAL_GATE_FAILURE_IDLE
)
_FATAL_GATE_FAILURE_SIGNALED_CLAIM = _FatalGateFailureClaim(
    _FATAL_GATE_FAILURE_SIGNALED
)


class _RemoteCommunicatorFailure(RuntimeError):
    def __init__(self, origin_rank):
        self.origin_rank = origin_rank
        super().__init__("remote communicator failure from rank {}".format(origin_rank))


class _FatalPublicationReservation:
    __slots__ = (
        "primary",
        "installed",
        "started",
        "joined",
        "monitor_deferred",
        "completion_deferred",
        "handler",
        "transition_handler",
        "handler_started",
        "transition_publish_started",
        "transition_published",
        "failure_signaled",
        "fail_stopped",
        "transition",
        "snapshot",
        "completion",
        "diagnostics",
    )

    def __init__(
        self,
        primary,
        installed,
        started,
        joined=False,
        monitor_deferred=False,
        completion_deferred=False,
        handler=None,
        transition_handler=None,
        transition=None,
    ):
        self.primary = primary
        self.installed = installed
        self.started = started
        self.joined = joined
        self.monitor_deferred = monitor_deferred
        self.completion_deferred = completion_deferred
        self.handler = handler
        self.transition_handler = transition_handler
        self.handler_started = False
        self.transition_publish_started = False
        self.transition_published = False
        self.failure_signaled = False
        self.fail_stopped = False
        self.transition = transition
        self.snapshot = None
        self.completion = None
        self.diagnostics = ()

    def retain_diagnostics(self, errors):
        for error in tuple(errors):
            if not isinstance(error, BaseException):
                continue
            if any(retained is error for retained in self.diagnostics):
                continue
            self.diagnostics = (*self.diagnostics, error)


class _DeferredFatalPublication:
    __slots__ = ("boundary", "reservation", "completed")

    def __init__(self, boundary, reservation):
        self.boundary = boundary
        self.reservation = reservation
        self.completed = False

    @property
    def primary(self):
        return self.reservation.primary

    def complete(self):
        if self.completed:
            return self.primary
        self.boundary.__exit__(None, None, None)
        self.completed = True
        return self.primary


class _FatalPublicationOwnerReservation:
    __slots__ = (
        "primary",
        "owner_thread",
        "owner_thread_id",
        "installed",
        "adopted",
        "election",
    )

    def __init__(self, primary, owner_thread, installed, election=None):
        self.primary = primary
        self.owner_thread = owner_thread
        self.owner_thread_id = owner_thread.ident
        self.installed = installed
        self.adopted = False
        self.election = election


def _validate_reduction_op(op):
    if op not in _REDUCTION_OPS:
        raise ValueError("unsupported reduction op {!r}".format(op))


def _normalize_axis(array, axis):
    if type(axis) is not int:
        raise TypeError("axis must be an integer")
    ndim = int(array.ndim)
    if axis < -ndim or axis >= ndim:
        raise ValueError("axis {} is out of range for {} dimensions".format(axis, ndim))
    return axis % ndim


@runtime_checkable
class Collective(Protocol):
    rank: int
    size: int

    def barrier(self) -> None:
        ...

    def broadcast(self, array: Any, *, root: int) -> Any:
        ...

    def allreduce(self, array: Any, *, op: str = "sum") -> Any:
        ...

    def allreduce_inplace(self, array: Any, *, op: str = "sum") -> Any:
        ...

    def reduce_scatter(self, array: Any, *, axis: int, op: str = "sum") -> Any:
        ...

    def allgather(self, array: Any, *, axis: int) -> Any:
        ...

    def close(self) -> None:
        ...


class SingleProcessCollective:
    _terminal_lifecycle_deadlines = True
    rank = 0
    size = 1

    def __init__(self):
        self._fatal_error = None
        self._fatal_handler = None
        self._active_broadcast_sequence = 1

    def _require_operational(self):
        fatal_error = getattr(self, "_fatal_error", None)
        if fatal_error is not None:
            raise RuntimeError("collective is terminal-aborted") from fatal_error

    def barrier(self):
        self._require_operational()
        return None

    def broadcast(self, array, *, root):
        self._require_operational()
        if root != 0:
            raise ValueError("single-process broadcast root must be zero")
        return array

    def allreduce(self, array, *, op="sum"):
        self._require_operational()
        _validate_reduction_op(op)
        return array.copy()

    def allreduce_inplace(self, array, *, op="sum"):
        self._require_operational()
        _validate_reduction_op(op)
        return array

    @staticmethod
    def _bootstrap_fatal_control(*, _deadline=None):
        if _deadline is not None and time.monotonic() >= _deadline:
            raise TimeoutError("terminal lifecycle deadline expired")
        return 0

    def _install_fatal_handler(self, handler):
        if not callable(handler):
            raise TypeError("fatal handler must be callable")
        self._fatal_handler = handler

    def _publish_communicator_fatal(self, error):
        if not isinstance(error, BaseException):
            raise TypeError("communicator fatal failure must be an exception")
        if getattr(self, "_fatal_error", None) is None:
            self._fatal_error = error
        return self._fatal_error

    def _agree_active_broadcast(self, failed):
        if type(failed) is not bool:
            raise TypeError("active broadcast failure flag must be a boolean")
        self._active_broadcast_sequence = (
            getattr(self, "_active_broadcast_sequence", 1) + 1
        )
        if failed:
            if getattr(self, "_fatal_error", None) is None:
                self._fatal_error = RuntimeError("active broadcast failed")
            raise RuntimeError("collective is terminal-aborted") from self._fatal_error
        self._require_operational()

    @staticmethod
    def _bootstrap_status_or(local_code, *, _deadline=None):
        if _deadline is not None and time.monotonic() >= _deadline:
            raise TimeoutError("terminal lifecycle deadline expired")
        if type(local_code) is not int or local_code < 0:
            raise ValueError("bootstrap status code must be a non-negative integer")
        return local_code

    def reduce_scatter(self, array, *, axis, op="sum"):
        self._require_operational()
        _normalize_axis(array, axis)
        _validate_reduction_op(op)
        return array.copy()

    def allgather(self, array, *, axis):
        self._require_operational()
        _normalize_axis(array, axis)
        return array.copy()

    def close(self):
        return None


class CupyNcclCollective:
    """Device-bound owner of one ``cupyx.distributed`` NCCL backend.

    Broadcast mutates and returns its input. Other tensor collectives preserve
    their input and return a new array. Axis collectives require equal per-rank
    counts; uneven sharding is intentionally outside this interface.
    """

    _SUPPORTED_DTYPE_CHARS = frozenset("bBiIlLqQefdFD")
    _terminal_lifecycle_deadlines = True

    def __init__(
        self,
        context,
        *,
        cupy_module=None,
        init_process_group=None,
        host=None,
        port=None,
    ):
        if not isinstance(host, str) or not host or host.strip() != host:
            raise ValueError("host must be a non-empty string")
        if type(port) is not int or port < 1 or port > 65535:
            raise ValueError("port must be an integer between 1 and 65535")
        if cupy_module is None:
            import cupy as cupy_module
        if init_process_group is None:
            from cupyx.distributed import init_process_group

        self._cupy = cupy_module
        self._context = context
        self.rank = context.rank
        self.size = context.world_size
        self._device_index = context.local_rank
        self._backend = None
        self._bootstrap_store_proxy = None
        self._closed = False
        self._close_lock = threading.Lock()
        self._close_in_progress = False
        self._fatal_lock = threading.RLock()
        self._fatal_condition = threading.Condition(self._fatal_lock)
        self._fatal_publication_lock = threading.RLock()
        self._fatal_publication_local = threading.local()
        self._active_broadcast_local = threading.local()
        self._admitted_operations = 0
        self._active_broadcast_agreements = 0
        self._fatal_publications = 0
        self._fatal_publication_failure = None
        self._fatal_publication_gate_failure_claim = (
            _FATAL_GATE_FAILURE_IDLE_CLAIM
        )
        self._fatal_publication_owner_reservation = None
        self._closing = False
        self._fatal_pending_primary = None
        self._fatal_pending_origin_rank = None
        self._fatal_error = None
        self._fatal_origin_rank = None
        self._fatal_secondary_errors = ()
        self._fatal_diagnostic_dispatches = ()
        self._fatal_transition_diagnostic_dispatches = ()
        self._fatal_store_announced = False
        self._fatal_abort_started = False
        self._fatal_abort_completed = False
        self._fatal_abort_event = threading.Event()
        self._fatal_abort_error = None
        self._fatal_abort_thread = None
        self._fatal_abort_state = "none"
        self._fatal_protocol_completed = False
        self._fatal_handler = None
        self._fatal_transition_handler = None
        self._fatal_transition_fallback = None
        self._terminal_gate_fallback = None
        self._terminal_runtime_fallback = None
        self._fatal_monitor_stop = threading.Event()
        self._fatal_monitor_thread = None
        self._fatal_monitor_state = "none"
        self._fatal_monitor_handoff = _FatalMonitorHandoff()
        self._fatal_monitor_outcome = None
        self._fatal_monitor_outcome_confirmed = False
        self._fatal_monitor_publisher_thread = None
        self._fatal_monitor_publisher_state = "none"
        self._fatal_monitor_publisher_diagnostics = ()
        self._fatal_hard_exit_primary = None
        self._fatal_hard_exit_claimant = None
        self._fatal_control_initialized = False
        self._active_broadcast_sequence = 1

        device_count = int(cupy_module.cuda.runtime.getDeviceCount())
        if self._device_index >= device_count:
            raise ValueError(
                "local_rank {} is out of range for {} visible CUDA device(s)".format(
                    self._device_index, device_count
                )
            )

        backend = None
        try:
            cupy_module.cuda.Device(self._device_index).use()
            options = {"backend": "nccl", "host": host, "port": port}
            backend = init_process_group(self.size, self.rank, **options)
            if int(backend.rank) != self.rank:
                raise RuntimeError(
                    "NCCL backend rank does not match distributed context"
                )
        except BaseException:
            self._closed = True
            if backend is not None:
                backend.stop()
            raise
        self._backend = backend
        self._bootstrap_store_proxy = self._independent_store_proxy(backend, host, port)

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _require_open(self):
        if self._closed:
            raise RuntimeError("collective is closed")

    def _require_operational(self):
        with self._fatal_lock:
            fatal_error = self._fatal_error
            pending_primary = self._fatal_pending_primary
            closing = self._closing
        if fatal_error is not None:
            raise RuntimeError("collective is terminal-aborted") from fatal_error
        if pending_primary is not None:
            raise RuntimeError(
                "collective fatal publication is pending"
            ) from pending_primary
        if closing:
            raise RuntimeError("collective is closing")
        self._require_open()

    def _begin_collective_operation(self, *, active_broadcast=False):
        with self._fatal_condition:
            if self._fatal_error is not None:
                raise RuntimeError(
                    "collective is terminal-aborted"
                ) from self._fatal_error
            if self._fatal_pending_primary is not None:
                raise RuntimeError(
                    "collective fatal publication is pending"
                ) from self._fatal_pending_primary
            if self._closed:
                raise RuntimeError("collective is closed")
            if self._closing:
                raise RuntimeError("collective is closing")
            self._admitted_operations += 1
            if active_broadcast:
                self._active_broadcast_agreements += 1

    def _finish_collective_operation(self, *, active_broadcast=False):
        with self._fatal_condition:
            if self._admitted_operations <= 0:
                raise RuntimeError("collective operation count is inconsistent")
            if active_broadcast:
                if self._active_broadcast_agreements <= 0:
                    raise RuntimeError(
                        "active broadcast agreement count is inconsistent"
                    )
                self._active_broadcast_agreements -= 1
            self._admitted_operations -= 1
            self._fatal_condition.notify_all()

    def _execute_admitted_collective(self, operation):
        try:
            return operation()
        except BaseException as error:
            primary = self._enter_observed_fatal(error, self.rank, admitted=True)
            if primary is error:
                raise
            raise primary

    def _raise_terminal(self):
        with self._fatal_lock:
            error = self._fatal_error
            pending = self._fatal_pending_primary
        if error is not None:
            raise RuntimeError("collective is terminal-aborted") from error
        raise RuntimeError("collective fatal publication is pending") from pending

    def _record_fatal_secondary(self, error):
        if not isinstance(error, BaseException):
            raise TypeError("fatal secondary failure must be an exception")
        with self._fatal_lock:
            if (
                error is self._fatal_error
                or error is self._fatal_pending_primary
                or any(
                    retained is error for retained in self._fatal_secondary_errors
                )
            ):
                return
            self._fatal_secondary_errors = (*self._fatal_secondary_errors, error)

    def _retain_fatal_secondary_direct(self, error):
        if not isinstance(error, BaseException):
            return
        with self._fatal_condition:
            if (
                error is self._fatal_error
                or error is self._fatal_pending_primary
                or any(
                    retained is error for retained in self._fatal_secondary_errors
                )
            ):
                return
            self._fatal_secondary_errors = (*self._fatal_secondary_errors, error)

    def _retain_fatal_secondaries_direct(self, errors):
        for error in tuple(errors):
            self._retain_fatal_secondary_direct(error)

    def _dispatch_fatal_secondaries(self, errors):
        for error in errors:
            if not isinstance(error, BaseException):
                continue
            with self._fatal_condition:
                if (
                    error is self._fatal_error
                    or error is self._fatal_pending_primary
                    or any(
                        dispatched is error
                        for dispatched in self._fatal_diagnostic_dispatches
                    )
                ):
                    continue
                self._fatal_diagnostic_dispatches = (
                    *self._fatal_diagnostic_dispatches,
                    error,
                )
            try:
                recorder = getattr(self, "_record_fatal_secondary")
            except BaseException as diagnostic_error:
                self._retain_fatal_secondary_direct(error)
                self._retain_fatal_secondary_direct(diagnostic_error)
                continue
            try:
                recorder(error)
            except BaseException as diagnostic_error:
                self._retain_fatal_secondary_direct(diagnostic_error)
            finally:
                self._retain_fatal_secondary_direct(error)

    def _dispatch_fatal_transition_secondaries(self, handler, errors):
        if handler is None:
            return
        for error in errors:
            if not isinstance(error, BaseException):
                continue
            with self._fatal_condition:
                if any(
                    retained_handler is handler and retained_error is error
                    for retained_handler, retained_error in (
                        self._fatal_transition_diagnostic_dispatches
                    )
                ):
                    continue
                self._fatal_transition_diagnostic_dispatches = (
                    *self._fatal_transition_diagnostic_dispatches,
                    (handler, error),
                )
            try:
                recorder = getattr(handler, "record_secondary", None)
            except BaseException as diagnostic_error:
                self._retain_fatal_secondary_direct(diagnostic_error)
                continue
            if not callable(recorder):
                continue
            try:
                recorder(error)
            except BaseException as diagnostic_error:
                self._retain_fatal_secondary_direct(diagnostic_error)

    def _collect_fatal_transition_secondaries(self, handler, transition):
        if handler is None or transition is None:
            return ()
        try:
            collector = getattr(handler, "secondaries", None)
        except BaseException as diagnostic_error:
            return (diagnostic_error,)
        if not callable(collector):
            return ()
        try:
            return tuple(collector(transition))
        except BaseException as diagnostic_error:
            return (diagnostic_error,)

    def _claim_fatal_hard_exit(self, error):
        if not isinstance(error, BaseException):
            raise TypeError("fatal hard-exit primary must be an exception")
        with self._fatal_condition:
            owner = self._fatal_publication_owner_reservation
            canonical = self._fatal_publication_failure
            if canonical is None and owner is not None:
                canonical = owner.primary
            if canonical is None:
                canonical = self._fatal_pending_primary
            if canonical is None:
                canonical = self._fatal_error
            if canonical is None:
                canonical = error
            if self._fatal_hard_exit_primary is None:
                self._fatal_hard_exit_primary = canonical
            else:
                canonical = self._fatal_hard_exit_primary
            winner = self._fatal_hard_exit_claimant is None
            if winner:
                self._fatal_hard_exit_claimant = threading.current_thread()
            self._fatal_condition.notify_all()
        if error is not canonical:
            self._retain_fatal_secondary_direct(error)
        return winner, canonical

    def _direct_terminal_gate_failure_marker(self, error):
        reference = self._terminal_gate_fallback
        gate = None if reference is None else reference()
        if gate is None:
            return None
        discovering_token = gate._current_thread_admission()
        return gate._terminalize_fatal_failure(error, discovering_token)

    def _ensure_fail_stop_markers(self, error, diagnostics=()):
        retained = list(diagnostics)
        with self._fatal_condition:
            retain_discovered = (
                error is not self._fatal_error
                and error is not self._fatal_pending_primary
                and not any(
                    existing is error
                    for existing in self._fatal_secondary_errors
                )
            )
        transition = None
        try:
            transition = self._direct_terminal_gate_failure_marker(error)
        except BaseException as gate_error:
            retained.append(gate_error)
        primary = error if transition is None else transition.primary
        if error is not primary:
            retained.append(error)
        primary, _ = self._install_fatal_failure_marker(primary, retained)
        with self._fatal_condition:
            observations = (
                (error, *retained) if retain_discovered else tuple(retained)
            )
            for observed in observations:
                if (
                    not isinstance(observed, BaseException)
                    or (
                        observed is primary
                        and not (retain_discovered and observed is error)
                    )
                    or any(
                    existing is observed
                    for existing in self._fatal_secondary_errors
                    )
                ):
                    continue
                self._fatal_secondary_errors = (
                    *self._fatal_secondary_errors,
                    observed,
                )
        runtime_reference = self._terminal_runtime_fallback
        runtime = None if runtime_reference is None else runtime_reference()
        if runtime is not None:
            with runtime._terminal_state_lock:
                for observed in retained:
                    if (
                        not isinstance(observed, BaseException)
                        or observed is primary
                        or any(
                            existing is observed
                            for existing in runtime._terminal_secondary_errors
                        )
                    ):
                        continue
                    runtime._terminal_secondary_errors = (
                        *runtime._terminal_secondary_errors,
                        observed,
                    )
        return primary

    def _hard_exit_once(self, error):
        primary = self._ensure_fail_stop_markers(error)
        winner, primary = self._claim_fatal_hard_exit(primary)
        if not winner:
            raise primary
        self._fatal_hard_exit()
        raise primary

    def _current_thread_claimed_fatal_hard_exit(self, primary=None):
        with self._fatal_condition:
            return (
                self._fatal_hard_exit_claimant is threading.current_thread()
                and (
                    primary is None
                    or self._fatal_hard_exit_primary is primary
                )
            )

    def _diagnose_and_hard_exit(self, error):
        """Install both terminal markers before claiming the hard exit."""
        return self._fail_stop_fatal_path(error)

    def _install_fatal_failure_marker(self, primary, diagnostics=()):
        """Publish the collective failure outcome and release its owner."""
        if not isinstance(primary, BaseException):
            raise TypeError("fatal publication failure must be an exception")
        retained = list(diagnostics)
        with self._fatal_condition:
            existing = self._fatal_publication_failure
            if existing is not None:
                if primary is not existing:
                    retained.append(primary)
                primary = existing
            owner = self._fatal_publication_owner_reservation
            if owner is not None and owner.primary is not primary:
                retained.append(owner.primary)
            pending = self._fatal_pending_primary
            if pending is not None and pending is not primary:
                retained.append(pending)
            self._fatal_pending_primary = primary
            first_failure = self._fatal_publication_failure is None
            self._fatal_publication_failure = primary
            self._fatal_condition.notify_all()
        self._retain_fatal_secondaries_direct(retained)
        return primary, first_failure

    def _terminalize_unowned_fatal_failure(self, error, diagnostics=()):
        primary = self._ensure_fail_stop_markers(error, diagnostics)
        self._hard_exit_once(primary)

    def _terminalize_pre_owner_fatal_failure(self, error, diagnostics=()):
        """Elect one structured owner before marker-first fail-stop publication."""
        retained = list(diagnostics)
        transition_handler = None
        transition = None
        try:
            transition_handler = self._fatal_transition_handler_callback()
        except BaseException as handler_error:
            retained.append(handler_error)
        begin = None
        if transition_handler is not None:
            try:
                begin = getattr(transition_handler, "begin", None)
            except BaseException as begin_lookup_error:
                retained.append(begin_lookup_error)
        if callable(begin):
            try:
                transition = begin(error)
            except BaseException as begin_error:
                retained.append(begin_error)
        if transition is None:
            return self._terminalize_unowned_fatal_failure(
                error,
                diagnostics=tuple(retained),
            )
        primary = transition.primary
        if error is not primary:
            retained.append(error)
        return self._terminalize_structured_fatal_failure(
            primary,
            transition_handler=transition_handler,
            transition=transition,
            diagnostics=tuple(retained),
        )

    def _fail_stop_fatal_path(self, error):
        """Use structured ownership when present, otherwise exit directly."""
        with self._fatal_condition:
            owner = self._fatal_publication_owner_reservation
            owned = (
                self._fatal_publications > 0
                and owner is not None
                and owner.owner_thread is threading.current_thread()
            )
        if owned:
            primary = self._terminalize_structured_fatal_failure(error)
            raise primary
        if owner is None:
            return self._terminalize_pre_owner_fatal_failure(error)

        deadline = self._inherited_fatal_deadline()
        with self._fatal_condition:
            while (
                self._fatal_publication_failure is None
                and self._fatal_error is None
                and owner.owner_thread.is_alive()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._fatal_condition.wait(remaining)
            primary = owner.primary
        return self._terminalize_unowned_fatal_failure(
            primary,
            diagnostics=(error,),
        )

    def _install_fatal(self, error, origin_rank):
        if not isinstance(error, BaseException):
            raise TypeError("communicator fatal failure must be an exception")
        with self._fatal_lock:
            installed = self._fatal_pending_primary is None
            if installed:
                self._fatal_pending_primary = error
                self._fatal_pending_origin_rank = origin_rank
            return self._fatal_pending_primary, installed

    def _pre_reserve_fatal_publication(
        self, error, origin_rank=None, *, election=None
    ):
        """Install one joinable owner before exposing fatal monitor selection."""
        if not isinstance(error, BaseException):
            raise TypeError("communicator fatal failure must be an exception")
        owner_reservation = None
        if election is not None:
            owner_reservation = getattr(election, "owner_reservation", None)
            if (
                owner_reservation is None
                or owner_reservation.primary is not error
                or owner_reservation.election is not election
            ):
                raise RuntimeError(
                    "communicator fatal owner was not preallocated"
                )
        return self._install_prepared_fatal_publication_owner(
            error,
            origin_rank,
            owner_reservation=owner_reservation,
            election=election,
        )

    def _prepare_runtime_fatal_owner_reservation(self, error, election):
        if not isinstance(error, BaseException):
            raise TypeError("communicator fatal failure must be an exception")
        if election is None:
            raise TypeError("runtime fatal election is required")
        return _FatalPublicationOwnerReservation(
            error,
            threading.current_thread(),
            None,
            election,
        )

    def _install_prepared_fatal_publication_owner(
        self,
        error,
        origin_rank=None,
        *,
        owner_reservation=None,
        election=None,
    ):
        """Install a prepared exact owner without invoking election callbacks."""
        if not isinstance(error, BaseException):
            raise TypeError("communicator fatal failure must be an exception")
        with self._fatal_condition:
            current_owner = self._fatal_publication_owner_reservation
            if current_owner is owner_reservation and current_owner is not None:
                if (
                    self._fatal_publications != 1
                    or current_owner.primary is not error
                    or current_owner.election is not election
                ):
                    raise RuntimeError(
                        "communicator fatal owner reservation is inconsistent"
                    )
                return current_owner.primary, False
            if self._fatal_protocol_completed:
                return self._fatal_pending_primary, False
            if self._fatal_publications > 0:
                return self._fatal_pending_primary, False
            if self._closed:
                raise RuntimeError("collective is closed")
            if owner_reservation is not None:
                owner_reservation.installed = (
                    self._fatal_pending_primary is None
                )
            primary, installed = self._install_fatal(error, origin_rank)
            if owner_reservation is None:
                owner_reservation = _FatalPublicationOwnerReservation(
                    primary,
                    threading.current_thread(),
                    installed,
                    election,
                )
            elif (
                owner_reservation.owner_thread is not threading.current_thread()
                or owner_reservation.primary is not primary
                or owner_reservation.election is not election
                or owner_reservation.adopted
            ):
                raise RuntimeError(
                    "communicator fatal owner reservation changed"
                )
            else:
                owner_reservation.installed = installed
            self._fatal_publication_owner_reservation = owner_reservation
            self._fatal_publications += 1
            self._fatal_condition.notify_all()
            return primary, True

    def _current_thread_reserved_fatal_primary(self):
        with self._fatal_condition:
            owner = self._fatal_publication_owner_reservation
            if (
                self._fatal_publications <= 0
                or owner is None
                or owner.owner_thread is not threading.current_thread()
            ):
                return None
            return owner.primary

    def _current_thread_reserved_fatal_election(self):
        with self._fatal_condition:
            owner = self._fatal_publication_owner_reservation
            if (
                self._fatal_publications <= 0
                or owner is None
                or owner.owner_thread is not threading.current_thread()
            ):
                return None
            return owner.election

    def _begin_fatal_publication(
        self,
        error,
        origin_rank,
        *,
        admitted=False,
        release_admitted=False,
        join_existing=False,
    ):
        if not isinstance(error, BaseException):
            raise TypeError("communicator fatal failure must be an exception")
        with self._fatal_condition:
            owner_reservation = self._fatal_publication_owner_reservation
            if (
                owner_reservation is not None
                and owner_reservation.owner_thread
                is threading.current_thread()
            ):
                if owner_reservation.adopted:
                    raise RuntimeError(
                        "communicator fatal owner was already adopted"
                    )
                if owner_reservation.primary is not error:
                    raise RuntimeError("communicator fatal primary changed")
                if self._fatal_pending_origin_rank is None:
                    self._fatal_pending_origin_rank = origin_rank
                owner_reservation.adopted = True
                return (
                    owner_reservation.primary,
                    owner_reservation.installed,
                    True,
                    False,
                    False,
                )
            if self._fatal_protocol_completed:
                return (
                    self._fatal_pending_primary,
                    False,
                    False,
                    self._fatal_publications > 0,
                    False,
                )
            if (
                self._fatal_pending_primary is not None
                and self._fatal_publications > 0
            ):
                monitor_deferred = (
                    threading.current_thread() is self._fatal_monitor_thread
                )
                return (
                    self._fatal_pending_primary,
                    False,
                    False,
                    not monitor_deferred,
                    monitor_deferred,
                )
            needed_by_admitted_operation = (
                release_admitted and self._admitted_operations > 0
            )
            if (
                self._closing
                and not admitted
                and not needed_by_admitted_operation
                and not join_existing
            ) or self._closed:
                return (
                    self._fatal_pending_primary
                    if self._fatal_pending_primary is not None
                    else error,
                    False,
                    False,
                    False,
                    False,
                )
            primary, installed = self._install_fatal(error, origin_rank)
            owner_reservation = _FatalPublicationOwnerReservation(
                primary,
                threading.current_thread(),
                installed,
            )
            owner_reservation.adopted = True
            self._fatal_publication_owner_reservation = owner_reservation
            self._fatal_publications += 1
            return primary, installed, True, False, False

    def _finish_fatal_publication(self):
        with self._fatal_condition:
            if self._fatal_publications <= 0:
                raise RuntimeError("fatal publication count is inconsistent")
            self._fatal_publications -= 1
            if self._fatal_publications == 0:
                self._fatal_publication_owner_reservation = None
            self._fatal_condition.notify_all()

    def _wait_for_joined_fatal_publication(self, *, _deadline=None):
        deadline = self._inherited_fatal_deadline(_deadline)
        publication_failure = None
        fail_stop_error = None
        with self._fatal_condition:
            while self._fatal_publications:
                publication_failure = self._fatal_publication_failure
                if publication_failure is not None:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    fail_stop_error = RuntimeError(
                        "communicator fatal publication join timed out"
                    )
                    break
                self._fatal_condition.wait(remaining)
            if (
                fail_stop_error is None
                and publication_failure is None
                and not self._fatal_protocol_completed
            ):
                fail_stop_error = RuntimeError(
                    "communicator fatal publication ended without completion"
                )
        if fail_stop_error is not None:
            self._fail_stop_fatal_path(fail_stop_error)
        if publication_failure is not None:
            self._fail_stop_fatal_path(publication_failure)

    def _wait_for_active_broadcast_agreements(self, *, _deadline=None):
        deadline = self._inherited_fatal_deadline(_deadline)
        fail_stop_error = None
        with self._fatal_condition:
            while self._active_broadcast_agreements:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    fail_stop_error = RuntimeError(
                        "active broadcast publication barrier timed out"
                    )
                    break
                self._fatal_condition.wait(remaining)
        if fail_stop_error is not None:
            self._fail_stop_fatal_path(fail_stop_error)

    @contextmanager
    def _communicator_fatal_reservation(
        self,
        error,
        *,
        origin_rank=None,
        admitted=False,
        release_admitted=False,
        join_existing=False,
        fatal_owner=None,
        discovering_token=None,
        handler_override=None,
        transition=None,
        defer_completion=False,
        diagnostics=(),
    ):
        active = getattr(self._fatal_publication_local, "reservation", None)
        if active is not None:
            if transition is not None:
                if active.transition is None:
                    active.transition = transition
                elif active.transition is not transition:
                    raise RuntimeError("communicator fatal transition changed")
                if transition.primary is not active.primary:
                    raise RuntimeError("communicator fatal primary changed")
            active.retain_diagnostics(diagnostics)
            if error is not active.primary:
                active.retain_diagnostics((error,))
            self._refresh_fatal_handler(
                active,
                fatal_owner=fatal_owner,
                discovering_token=discovering_token,
            )
            yield active
            return
        if origin_rank is None:
            origin_rank = self.rank
        transition_handler = handler_override
        try:
            handler = (
                handler_override
                if handler_override is not None
                else self._fatal_handler_callback()
            )
            transition_handler = handler
            begin = None if handler is None else getattr(handler, "begin", None)
            if not callable(begin):
                transition_handler = self._fatal_transition_handler_callback()
                begin = (
                    None
                    if transition_handler is None
                    else getattr(transition_handler, "begin", None)
                )
            if handler is None:
                handler = transition_handler
            if transition is None and callable(begin):
                transition = begin(
                    error,
                    owner=fatal_owner,
                    discovering_token=discovering_token,
                )
            if transition is not None:
                primary = transition.primary
                if not isinstance(primary, BaseException):
                    raise TypeError(
                        "communicator fatal transition primary is invalid"
                    )
                if error is not primary:
                    diagnostics = (*tuple(diagnostics), error)
                outcome = self._observe_fatal_monitor_outcome()
                if (
                    outcome is None
                    or outcome.kind != "fatal_elected"
                    or outcome.primary is not primary
                ):
                    raise RuntimeError(
                        "communicator fatal handoff was not reserved"
                    )
            else:
                outcome = self._elect_collective_fatal_outcome(
                    error, origin_rank
                )
                primary = outcome.primary
            primary, installed, started, joined, monitor_deferred = (
                self._begin_fatal_publication(
                    primary,
                    origin_rank,
                    admitted=admitted,
                    release_admitted=release_admitted,
                    join_existing=join_existing,
                )
            )
            reservation = _FatalPublicationReservation(
                primary,
                installed,
                started,
                joined=joined,
                monitor_deferred=monitor_deferred,
                completion_deferred=defer_completion,
                handler=handler,
                transition_handler=transition_handler,
                transition=transition,
            )
            reservation.retain_diagnostics(diagnostics)
            deadline = self._inherited_fatal_deadline(
                None if transition is None else transition.deadline
            )
        except BaseException as setup_error:
            if self._current_thread_claimed_fatal_hard_exit():
                raise
            with self._fatal_condition:
                marker_installed = self._fatal_publication_failure is not None
            primary = self._terminalize_structured_fatal_failure(
                setup_error,
                transition_handler=transition_handler,
                transition=transition,
                diagnostics=diagnostics,
            )
            if marker_installed and setup_error is not primary:
                raise
            raise primary
        if not started:
            if monitor_deferred:
                outcome = self._wait_for_fatal_monitor_selection(
                    _deadline=deadline,
                )
                if outcome.kind != "fatal_elected" or outcome.primary is not primary:
                    raise RuntimeError("deferred monitor fatal outcome changed")
                self._fatal_monitor_outcome = outcome
            if joined:
                self._refresh_fatal_handler(
                    reservation,
                    fatal_owner=fatal_owner,
                    discovering_token=discovering_token,
                )
            attach_deferred_join = joined and defer_completion
            joined_succeeded = False
            try:
                if attach_deferred_join:
                    self._fatal_publication_local.reservation = reservation
                yield reservation
                if joined:
                    self._wait_for_joined_fatal_publication(
                        _deadline=(
                            None
                            if reservation.transition is None
                            else reservation.transition.deadline
                        ),
                    )
                joined_succeeded = True
            finally:
                if attach_deferred_join:
                    del self._fatal_publication_local.reservation
                if joined and not joined_succeeded:
                    self._retain_fatal_secondaries_direct(
                        reservation.diagnostics
                    )
            if joined_succeeded:
                self._dispatch_fatal_secondaries(reservation.diagnostics)
            return
        publication_succeeded = False
        local_installed = False
        try:
            self._fatal_publication_local.reservation = reservation
            local_installed = True
            if transition is not None and transition.primary is not primary:
                raise RuntimeError("communicator fatal primary changed")
            if transition is not None:
                self._confirm_runtime_fatal_outcome(
                    primary,
                    _deadline=deadline,
                )
            self._refresh_fatal_handler(
                reservation,
                fatal_owner=fatal_owner,
                discovering_token=discovering_token,
            )
            if not defer_completion:
                self._wait_for_active_broadcast_agreements(
                    _deadline=deadline,
                )
                self._prepare_local_fatal_locked(_deadline=deadline)
            yield reservation
            if defer_completion:
                self._wait_for_active_broadcast_agreements(
                    _deadline=deadline,
                )
                self._prepare_local_fatal_locked(_deadline=deadline)
            self._run_fatal_handler(reservation)
            self._publish_communicator_fatal_locked(
                primary,
                _deadline=deadline,
            )
            self._complete_fatal_handler(reservation)
            transition_errors = self._collect_fatal_transition_secondaries(
                reservation.transition_handler,
                reservation.transition,
            )
            reservation.retain_diagnostics(transition_errors)
            self._dispatch_fatal_secondaries(reservation.diagnostics)
            self._dispatch_fatal_transition_secondaries(
                reservation.transition_handler,
                reservation.diagnostics,
            )
            publication_succeeded = True
        except BaseException as error:
            if self._current_thread_claimed_fatal_hard_exit():
                raise
            with self._fatal_condition:
                marker_installed = self._fatal_publication_failure is not None
            primary = self._fail_fatal_publication(reservation, error)
            if marker_installed and error is not primary:
                raise
            raise primary
        finally:
            try:
                if local_installed:
                    del self._fatal_publication_local.reservation
            finally:
                if publication_succeeded:
                    self._finish_fatal_publication()

    @contextmanager
    def _adopt_communicator_fatal_reservation_locked(self, error, origin_rank):
        with self._communicator_fatal_reservation(
            error, origin_rank=origin_rank, join_existing=True
        ) as reservation:
            yield reservation

    def _install_fatal_handler(self, handler):
        if not callable(handler):
            raise TypeError("fatal handler must be callable")
        try:
            reference = weakref.WeakMethod(handler)
        except TypeError:
            reference = weakref.ref(handler)
        fallback = getattr(handler, "_terminalize_failure", None)
        if callable(fallback):
            try:
                fallback_reference = weakref.WeakMethod(fallback)
            except TypeError:
                fallback_reference = weakref.ref(fallback)
        else:
            fallback_reference = None
        with self._fatal_lock:
            self._fatal_handler = reference
            if callable(getattr(handler, "begin", None)):
                self._fatal_transition_handler = reference
            self._fatal_transition_fallback = fallback_reference

    def _fatal_handler_callback(self):
        with self._fatal_lock:
            reference = self._fatal_handler
        return None if reference is None else reference()

    def _fatal_transition_handler_callback(self):
        with self._fatal_lock:
            reference = self._fatal_transition_handler
        return None if reference is None else reference()

    def _fatal_transition_fallback_callback(self):
        with self._fatal_lock:
            reference = self._fatal_transition_fallback
        return None if reference is None else reference()

    def _refresh_fatal_handler(
        self,
        reservation,
        *,
        fatal_owner=None,
        discovering_token=None,
    ):
        handler = reservation.transition_handler
        begin = None if handler is None else getattr(handler, "begin", None)
        if not callable(begin):
            return
        if reservation.transition is not None:
            if reservation.transition.primary is not reservation.primary:
                raise RuntimeError("communicator fatal primary changed")
            return
        transition = begin(
            reservation.primary,
            owner=fatal_owner,
            discovering_token=discovering_token,
        )
        if reservation.transition is None:
            reservation.transition = transition
        elif reservation.transition is not transition:
            raise RuntimeError("communicator fatal transition changed")
        if transition.primary is not reservation.primary:
            raise RuntimeError("communicator fatal primary changed")

    def _run_fatal_handler(self, reservation):
        if reservation.handler_started or reservation.handler is None:
            reservation.completion_deferred = False
            return reservation.snapshot
        reservation.handler_started = True
        publish_transition_first = (
            reservation.completion_deferred
            and reservation.transition is not None
            and reservation.transition_handler is not None
            and reservation.transition_handler is not reservation.handler
        )
        reservation.completion_deferred = False
        if publish_transition_first:
            publish_transition = getattr(
                reservation.transition_handler, "publish", None
            )
            if callable(publish_transition):
                reservation.transition_publish_started = True
                try:
                    reservation.snapshot = publish_transition(
                        reservation.transition
                    )
                finally:
                    reservation.transition_publish_started = False
                reservation.transition_published = True
        publish = getattr(reservation.handler, "publish", None)
        if callable(publish):
            reservation.transition_publish_started = True
            try:
                reservation.snapshot = publish(reservation.transition)
            finally:
                reservation.transition_publish_started = False
            if reservation.handler is reservation.transition_handler:
                reservation.transition_published = True
        else:
            handler_snapshot = reservation.handler(reservation.primary)
            if reservation.snapshot is None:
                reservation.snapshot = handler_snapshot
        if (
            reservation.transition is not None
            and reservation.transition_handler is not None
            and reservation.transition_handler is not reservation.handler
            and not reservation.transition_published
        ):
            publish_transition = getattr(
                reservation.transition_handler, "publish", None
            )
            if callable(publish_transition):
                reservation.transition_publish_started = True
                try:
                    reservation.snapshot = publish_transition(
                        reservation.transition
                    )
                finally:
                    reservation.transition_publish_started = False
                reservation.transition_published = True
        return reservation.snapshot

    @staticmethod
    def _invoke_fatal_gate_failure_callback(
        failure_handler, transition, primary
    ):
        return failure_handler(transition, primary)

    @staticmethod
    def _verify_fatal_gate_failure_callback(
        failure_confirmed, transition, primary
    ):
        return bool(failure_confirmed(transition, primary))

    def _repair_fatal_gate_failure_claim(self, claim, confirmed):
        with self._fatal_condition:
            current = self._fatal_publication_gate_failure_claim
            if (
                isinstance(current, _FatalGateFailureClaim)
                and current.state == _FATAL_GATE_FAILURE_SIGNALED
            ):
                self._fatal_publication_gate_failure_claim = (
                    _FATAL_GATE_FAILURE_SIGNALED_CLAIM
                )
                result = True
            elif confirmed:
                self._fatal_publication_gate_failure_claim = (
                    _FATAL_GATE_FAILURE_SIGNALED_CLAIM
                )
                result = True
            elif (
                current is claim
                or not isinstance(current, _FatalGateFailureClaim)
                or current.state == _FATAL_GATE_FAILURE_IDLE
            ):
                self._fatal_publication_gate_failure_claim = (
                    _FATAL_GATE_FAILURE_IDLE_CLAIM
                )
                result = False
            else:
                result = False
            self._fatal_condition.notify_all()
            return result

    def _settle_fatal_gate_failure_claim(self, claim, confirmed):
        return self._repair_fatal_gate_failure_claim(claim, confirmed)

    def _signal_reserved_fatal_gate_failure(
        self,
        primary,
        transition,
        failure_handler,
        failure_confirmed,
    ):
        current_thread = threading.current_thread()
        protocol_deadline = self._inherited_fatal_deadline(
            transition.deadline,
        )
        signal_errors = []
        for _attempt in range(2):
            claim = _FatalGateFailureClaim(
                _FATAL_GATE_FAILURE_INFLIGHT,
                current_thread,
                protocol_deadline,
            )
            claim_owned = False
            claim_published = False
            confirmed = False
            verification_conclusive = False
            decision = None
            try:
                with self._fatal_condition:
                    current = self._fatal_publication_gate_failure_claim
                    if not isinstance(current, _FatalGateFailureClaim):
                        claim_owned = True
                        self._fatal_publication_gate_failure_claim = claim
                        claim_published = True
                        self._fatal_condition.notify_all()
                        decision = "verify_before_callback"
                    elif current.state == _FATAL_GATE_FAILURE_SIGNALED:
                        decision = _FATAL_GATE_FAILURE_SIGNALED
                    elif current.state == _FATAL_GATE_FAILURE_IDLE:
                        claim_owned = True
                        self._fatal_publication_gate_failure_claim = claim
                        claim_published = True
                        self._fatal_condition.notify_all()
                        decision = "verify_before_callback"
                    elif current.owner is current_thread:
                        decision = "reentrant"
                    elif not current.owner.is_alive():
                        claim_owned = True
                        self._fatal_publication_gate_failure_claim = claim
                        claim_published = True
                        self._fatal_condition.notify_all()
                        decision = "verify_before_callback"
                    else:
                        remaining = max(
                            0.0, protocol_deadline - time.monotonic()
                        )
                        if remaining:
                            self._fatal_condition.wait(remaining)
                        observed = self._fatal_publication_gate_failure_claim
                        if not isinstance(observed, _FatalGateFailureClaim):
                            claim_owned = True
                            self._fatal_publication_gate_failure_claim = claim
                            claim_published = True
                            self._fatal_condition.notify_all()
                            decision = "verify_before_callback"
                        elif observed.state == _FATAL_GATE_FAILURE_SIGNALED:
                            decision = _FATAL_GATE_FAILURE_SIGNALED
                        elif observed.state == _FATAL_GATE_FAILURE_IDLE:
                            decision = "retry"
                        elif not observed.owner.is_alive():
                            claim_owned = True
                            self._fatal_publication_gate_failure_claim = claim
                            claim_published = True
                            self._fatal_condition.notify_all()
                            decision = "verify_before_callback"
                        else:
                            decision = "live_owner_timeout"
                if decision == _FATAL_GATE_FAILURE_SIGNALED:
                    return True, signal_errors
                if decision == "reentrant":
                    return False, signal_errors
                if decision == "live_owner_timeout":
                    signal_errors.append(
                        TimeoutError(
                            "communicator gate failure signal owner is still active"
                        )
                    )
                    return False, signal_errors
                if decision == "retry":
                    continue
                if decision == "verify_before_callback":
                    try:
                        confirmed = self._verify_fatal_gate_failure_callback(
                            failure_confirmed, transition, primary
                        )
                        verification_conclusive = True
                    except BaseException as error:
                        signal_errors.append(error)
                if verification_conclusive and not confirmed:
                    try:
                        self._invoke_fatal_gate_failure_callback(
                            failure_handler, transition, primary
                        )
                    except BaseException as error:
                        signal_errors.append(error)
                    try:
                        confirmed = self._verify_fatal_gate_failure_callback(
                            failure_confirmed, transition, primary
                        )
                        verification_conclusive = True
                    except BaseException as error:
                        signal_errors.append(error)
            except BaseException as error:
                signal_errors.append(error)
            finally:
                if claim_owned or claim_published:
                    if not confirmed:
                        try:
                            confirmed = (
                                self._verify_fatal_gate_failure_callback(
                                    failure_confirmed, transition, primary
                                )
                            )
                            verification_conclusive = True
                        except BaseException as error:
                            signal_errors.append(error)
                    try:
                        confirmed = self._settle_fatal_gate_failure_claim(
                            claim, confirmed
                        )
                    except BaseException as error:
                        signal_errors.append(error)
                        try:
                            confirmed = self._repair_fatal_gate_failure_claim(
                                claim, confirmed
                            )
                        except BaseException as repair_error:
                            signal_errors.append(repair_error)
            if confirmed:
                return True, signal_errors
        return False, signal_errors

    def _fail_reserved_fatal_publication(
        self,
        error,
        *,
        transition_handler=None,
        transition=None,
        diagnostics=(),
    ):
        if not isinstance(error, BaseException):
            raise TypeError("fatal publication failure must be an exception")
        consistency_error = None
        current_thread = threading.current_thread()
        with self._fatal_condition:
            owner = self._fatal_publication_owner_reservation
            if (
                self._fatal_publications <= 0
                or owner is None
            ):
                return False, False
            owned = owner.owner_thread is current_thread
            primary = owner.primary
            if not owned and self._fatal_publication_failure is not primary:
                return False, False
            if self._fatal_pending_primary is not primary:
                consistency_error = RuntimeError(
                    "communicator fatal publication primary changed"
                )
            first_failure = owned and self._fatal_publication_failure is None
            if owned and self._fatal_publication_failure is None:
                self._fatal_publication_failure = primary
            elif self._fatal_publication_failure is not primary:
                consistency_error = RuntimeError(
                    "communicator fatal publication failure changed"
                )
            self._fatal_condition.notify_all()
        signal_errors = []
        if transition_handler is None and transition is not None:
            try:
                transition_handler = self._fatal_transition_handler_callback()
            except BaseException as handler_error:
                signal_errors.append(handler_error)
        try:
            failure_handler = (
                None
                if transition_handler is None
                else getattr(transition_handler, "fail", None)
            )
        except BaseException as handler_error:
            signal_errors.append(handler_error)
            failure_handler = None
        try:
            failure_confirmed = (
                None
                if transition_handler is None
                else getattr(transition_handler, "failure_recorded", None)
            )
        except BaseException as handler_error:
            signal_errors.append(handler_error)
            failure_confirmed = None
        signal_gate = transition is not None and callable(failure_handler)
        gate_confirmed = False
        if signal_gate:
            if not callable(failure_confirmed):
                legacy_signal = [False]
                original_failure_handler = failure_handler

                def failure_handler(transition, failure):
                    original_failure_handler(transition, failure)
                    legacy_signal[0] = True

                def failure_confirmed(transition, failure):
                    return legacy_signal[0]

            try:
                gate_confirmed, caught_signal_errors = (
                    self._signal_reserved_fatal_gate_failure(
                        primary,
                        transition,
                        failure_handler,
                        failure_confirmed,
                    )
                )
            except BaseException as signal_error:
                signal_errors.append(signal_error)
            else:
                signal_errors.extend(caught_signal_errors)
        if transition is not None and not gate_confirmed:
            try:
                fallback = self._fatal_transition_fallback_callback()
            except BaseException as fallback_lookup_error:
                signal_errors.append(fallback_lookup_error)
                fallback = None
            if callable(fallback):
                try:
                    retained_transition = fallback(
                        primary,
                        diagnostics=tuple(signal_errors),
                        transition=transition,
                    )
                    gate_confirmed = retained_transition.primary is primary
                except BaseException as fallback_error:
                    signal_errors.append(fallback_error)
        diagnostic_errors = list(diagnostics)
        diagnostic_errors.extend(signal_errors)
        if first_failure and not (
            isinstance(error, SystemExit) and error.code == _FATAL_EXIT_CODE
        ):
            diagnostic_errors.append(error)
        if consistency_error is not None:
            diagnostic_errors.append(consistency_error)
        retained_primary, _ = self._install_fatal_failure_marker(
            primary,
            diagnostic_errors,
        )
        if retained_primary is not primary:
            self._retain_fatal_secondary_direct(primary)
        return owned, first_failure

    def _structured_fatal_failure_context(self):
        reservation = getattr(
            self._fatal_publication_local, "reservation", None
        )
        if reservation is not None:
            return reservation.transition_handler, reservation.transition
        with self._fatal_condition:
            owner = self._fatal_publication_owner_reservation
            election = None if owner is None else owner.election
            transition = None if election is None else election.transition
        return None, transition

    def _terminalize_structured_fatal_failure(
        self,
        error,
        *,
        transition_handler=None,
        transition=None,
        diagnostics=(),
    ):
        if not isinstance(error, BaseException):
            raise TypeError("structured fatal failure must be an exception")
        if transition_handler is None or transition is None:
            retained_handler, retained_transition = (
                self._structured_fatal_failure_context()
            )
            if transition_handler is None:
                transition_handler = retained_handler
            if transition is None:
                transition = retained_transition
        terminalization_errors = list(diagnostics)
        try:
            owned, _ = self._fail_reserved_fatal_publication(
                error,
                transition_handler=transition_handler,
                transition=transition,
                diagnostics=diagnostics,
            )
        except BaseException as terminalization_error:
            terminalization_errors.append(terminalization_error)
            owned = False
        with self._fatal_condition:
            primary = self._fatal_publication_failure
        if primary is None and transition is not None:
            primary = transition.primary
        if primary is None:
            primary = error
        if not owned and self._fatal_publication_failure is None:
            terminalization_errors.append(error)
            return self._terminalize_unowned_fatal_failure(
                primary,
                diagnostics=tuple(terminalization_errors),
            )
        terminalization_errors.append(error)
        self._retain_fatal_secondaries_direct(terminalization_errors)
        self._hard_exit_once(primary)

    def _fail_fatal_publication(self, reservation, error):
        if reservation.failure_signaled:
            return reservation.primary
        reservation.failure_signaled = True
        reservation.fail_stopped = True
        return self._terminalize_structured_fatal_failure(
            error,
            transition_handler=reservation.transition_handler,
            transition=reservation.transition,
            diagnostics=reservation.diagnostics,
        )

    @staticmethod
    def _complete_fatal_handler(reservation):
        if reservation.completion is not None:
            reservation.completion()
            return
        if reservation.transition is None:
            return
        handler = reservation.transition_handler
        if handler is None:
            handler = reservation.handler
        complete = None if handler is None else getattr(handler, "complete", None)
        if callable(complete):
            complete(reservation.transition, reservation.snapshot)

    @staticmethod
    def _fatal_key(rank):
        return "{}{}".format(_FATAL_KEY_PREFIX, rank)

    @staticmethod
    def _fatal_ack_key(rank):
        return "{}{}".format(_FATAL_ACK_KEY_PREFIX, rank)

    @staticmethod
    def _active_b_key(rank):
        return "{}{}".format(_ACTIVE_B_KEY_PREFIX, rank)

    @staticmethod
    def _close_ready_key(rank):
        return "{}{}".format(_CLOSE_READY_KEY_PREFIX, rank)

    @staticmethod
    def _monitor_stopped_key(rank):
        return "{}{}".format(_MONITOR_STOPPED_KEY_PREFIX, rank)

    @staticmethod
    def _close_consumed_key(rank):
        return "{}{}".format(_CLOSE_CONSUMED_KEY_PREFIX, rank)

    @staticmethod
    def _independent_store_proxy(backend, host, port):
        try:
            backend_proxy = getattr(backend, "_store_proxy")
        except BaseException:
            return None
        try:
            from cupyx.distributed._store import TCPStoreProxy
        except ImportError:
            return None
        if type(backend_proxy) is not TCPStoreProxy:
            return None
        try:
            proxy = TCPStoreProxy(host, port)
        except BaseException:
            return None
        if type(proxy) is not TCPStoreProxy or proxy is backend_proxy:
            return None
        return proxy

    @staticmethod
    def _encode_active_b(sequence, failed):
        return _ACTIVE_B_RECORD.pack(int(sequence), int(failed))

    @staticmethod
    def _decode_active_b(value):
        if not isinstance(value, (bytes, bytearray)):
            raise TypeError("active broadcast result record must be bytes")
        return _ACTIVE_B_RECORD.unpack(value)

    def _fatal_hard_exit(self):
        os._exit(_FATAL_EXIT_CODE)

    @staticmethod
    def _store_deadline(deadline):
        if deadline is None:
            return time.monotonic() + _FATAL_TIMEOUT_S
        if type(deadline) is not float:
            raise TypeError("terminal lifecycle deadline must be a float")
        if time.monotonic() >= deadline:
            raise TimeoutError("terminal lifecycle deadline expired")
        return deadline

    def _bounded_tcp_store_action(self, action, deadline):
        from cupyx.distributed import _klv_utils
        from cupyx.distributed._store import TCPStoreProxy

        proxy = self._bootstrap_store_proxy
        if type(proxy) is not TCPStoreProxy:
            raise TypeError("bounded TCP store requires an exact TCPStoreProxy")
        delay = float(proxy.DELAY_FOR_RETRY)
        for _ in range(int(proxy.MAX_NUM_RETRIES)):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("terminal lifecycle deadline expired")
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                    client.settimeout(remaining)
                    client.connect((proxy.host, proxy.port))
                    client.sendall(action.klv())
                    payload = client.recv(sizeof(_klv_utils.result_action_t))
                    if not payload:
                        continue
                    result = _klv_utils.result_action_t.from_buffer_copy(payload)
                    value = bytearray(result.value)[: result.length]
                    if result.status == 0:
                        return action.decode_result(value)
                    raise RuntimeError(value.decode("utf-8"))
            except ConnectionRefusedError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                threading.Event().wait(min(delay, remaining))
            except socket.timeout as error:
                raise TimeoutError(
                    "terminal lifecycle TCP store operation timed out"
                ) from error
        raise TimeoutError("terminal lifecycle TCP store operation timed out")

    def _store_set(self, key, value, *, _deadline=None):
        deadline = self._store_deadline(_deadline)
        proxy = self._bootstrap_store_proxy
        try:
            from cupyx.distributed import _store_actions
            from cupyx.distributed._store import TCPStoreProxy
        except ImportError:
            proxy[key] = value
            return None
        if type(proxy) is TCPStoreProxy:
            return self._bounded_tcp_store_action(
                _store_actions.Set(key, value),
                deadline,
            )
        proxy[key] = value
        if time.monotonic() > deadline:
            raise TimeoutError("terminal lifecycle deadline expired")
        return None

    def _store_get(self, key, *, _deadline=None):
        deadline = self._store_deadline(_deadline)
        proxy = self._bootstrap_store_proxy
        try:
            from cupyx.distributed import _store_actions
            from cupyx.distributed._store import TCPStoreProxy
        except ImportError:
            return proxy[key]
        if type(proxy) is TCPStoreProxy:
            return self._bounded_tcp_store_action(
                _store_actions.Get(key),
                deadline,
            )
        value = proxy[key]
        if time.monotonic() > deadline:
            raise TimeoutError("terminal lifecycle deadline expired")
        return value

    def _store_barrier(self, *, _deadline=None):
        deadline = self._store_deadline(_deadline)
        proxy = self._bootstrap_store_proxy
        try:
            from cupyx.distributed import _store_actions
            from cupyx.distributed._store import TCPStoreProxy
        except ImportError:
            return proxy.barrier()
        if type(proxy) is TCPStoreProxy:
            return self._bounded_tcp_store_action(
                _store_actions.Barrier(),
                deadline,
            )
        result = proxy.barrier()
        if time.monotonic() > deadline:
            raise TimeoutError("terminal lifecycle deadline expired")
        return result

    def _fatal_store_set(self, key, value, *, _deadline=None):
        try:
            self._store_set(
                key,
                value,
                _deadline=self._inherited_fatal_deadline(_deadline),
            )
        except BaseException as error:
            self._fail_stop_fatal_path(error)

    def _fatal_store_get(self, key, *, _deadline=None):
        try:
            return self._store_get(
                key,
                _deadline=self._inherited_fatal_deadline(_deadline),
            )
        except BaseException as error:
            self._fail_stop_fatal_path(error)

    def _inherited_fatal_deadline(self, explicit=None):
        if explicit is not None:
            return explicit
        reservation = getattr(
            self._fatal_publication_local,
            "reservation",
            None,
        )
        transition = (
            None if reservation is None else reservation.transition
        )
        if transition is not None and transition.deadline is not None:
            return transition.deadline
        reference = self._terminal_gate_fallback
        gate = None if reference is None else reference()
        if gate is not None:
            with gate._condition:
                transition = gate._fatal_transition
                if transition is None:
                    transition = gate._runtime_close_transition
                if transition is not None and transition.deadline is not None:
                    return transition.deadline
        return time.monotonic() + _FATAL_TIMEOUT_S

    def _abort_local_communicator(self, *, _deadline=None):
        deadline = self._inherited_fatal_deadline(_deadline)

        def abort():
            current = threading.current_thread()
            with self._fatal_condition:
                if (
                    self._fatal_abort_thread is not current
                    or self._fatal_abort_state != "starting"
                ):
                    return
                self._fatal_abort_state = "started"
                self._fatal_condition.notify_all()
            try:
                backend = self._backend
                raw_comm = getattr(backend, "_comm", None)
                abort_comm = getattr(raw_comm, "abort", None)
                if not callable(abort_comm):
                    raise RuntimeError(
                        "CuPy NCCL raw communicator abort is unavailable"
                    )
                abort_comm()
            except BaseException as error:
                with self._fatal_condition:
                    self._fatal_abort_error = error
            else:
                with self._fatal_condition:
                    self._fatal_abort_completed = True
            finally:
                with self._fatal_condition:
                    if self._fatal_abort_thread is current:
                        self._fatal_abort_state = "completed"
                    self._fatal_abort_event.set()
                    self._fatal_condition.notify_all()

        with self._fatal_condition:
            if self._fatal_abort_started:
                return self._fatal_abort_event
            self._fatal_abort_started = True
            thread = threading.Thread(
                target=abort,
                name="renormalizer-nccl-abort-rank-{}".format(self.rank),
                daemon=True,
            )
            self._fatal_abort_thread = thread
            self._fatal_abort_state = "starting"
            self._fatal_condition.notify_all()
        try:
            thread.start()
        except BaseException as start_error:
            with self._fatal_condition:
                claimed = (
                    self._fatal_abort_thread is thread
                    and self._fatal_abort_state in {"started", "completed"}
                )
                if not claimed and self._fatal_abort_thread is thread:
                    self._fatal_abort_thread = None
                    self._fatal_abort_state = "start_failed"
                    self._fatal_abort_error = start_error
                    self._fatal_abort_event.set()
                self._fatal_condition.notify_all()
            if claimed:
                self._retain_fatal_secondary_direct(start_error)
            elif thread.ident is not None:
                remaining = _remaining_lifecycle_time(
                    deadline,
                    "NCCL communicator abort start recovery timed out",
                )
                thread.join(remaining)
        return self._fatal_abort_event

    def _wait_for_local_communicator_abort(self, *, _deadline=None):
        deadline = self._inherited_fatal_deadline(_deadline)
        event = self._fatal_abort_event
        remaining = _remaining_lifecycle_time(
            deadline,
            "NCCL communicator abort timed out",
        )
        if not event.wait(remaining):
            error = RuntimeError("NCCL communicator abort timed out")
            self._fail_stop_fatal_path(error)
        with self._fatal_lock:
            error = self._fatal_abort_error
            completed = self._fatal_abort_completed
        if error is not None:
            self._fail_stop_fatal_path(error)
        if not completed:
            error = RuntimeError("NCCL communicator abort did not complete")
            self._fail_stop_fatal_path(error)

    def _wait_for_fatal_acknowledgments(self, *, _deadline=None):
        self._wait_for_all_control_records(
            self._fatal_ack_key,
            "communicator fatal acknowledgment timed out",
            _deadline=self._inherited_fatal_deadline(_deadline),
        )

    def _wait_for_all_control_records(
        self,
        key,
        timeout_message,
        *,
        _deadline=None,
    ):
        deadline = self._inherited_fatal_deadline(_deadline)
        while True:
            if all(
                int(self._fatal_store_get(key(rank), _deadline=deadline)) == 1
                for rank in range(self.size)
            ):
                return
            if time.monotonic() >= deadline:
                error = RuntimeError(timeout_message)
                self._fail_stop_fatal_path(error)
            time.sleep(0.001)

    def _prepare_local_fatal_locked(self, *, _deadline=None):
        deadline = self._inherited_fatal_deadline(_deadline)
        with self._fatal_lock:
            announce = not self._fatal_store_announced
            self._fatal_store_announced = True
            primary = self._fatal_pending_primary
            monitor = self._fatal_monitor_thread
            monitor_live = monitor is not None and monitor.is_alive()
        outcome = self._select_runtime_fatal_outcome(
            primary,
            _deadline=deadline,
        )
        if monitor is threading.current_thread():
            raise RuntimeError("fatal monitor cannot own communicator publication")
        elif monitor_live:
            self._wait_for_fatal_monitor_exit(
                outcome,
                _deadline=deadline,
            )
            self._join_fatal_monitor(monitor, _deadline=deadline)
        if announce:
            self._fatal_store_set(
                self._fatal_key(self.rank),
                1,
                _deadline=deadline,
            )
        self._abort_local_communicator(_deadline=deadline)

    def _publish_communicator_fatal_locked(self, primary, *, _deadline=None):
        deadline = self._inherited_fatal_deadline(_deadline)
        with self._fatal_lock:
            if self._fatal_protocol_completed:
                return primary
        if not self._fatal_control_initialized:
            missing = RuntimeError("communicator fatal control is unavailable")
            self._fail_stop_fatal_path(missing)
        self._wait_for_local_communicator_abort(_deadline=deadline)
        with self._fatal_lock:
            with self._fatal_publication_lock:
                if self._fatal_pending_primary is not primary:
                    raise RuntimeError("communicator fatal primary changed")
                if self._fatal_error is None:
                    self._fatal_error = primary
                    self._fatal_origin_rank = self._fatal_pending_origin_rank
        self._fatal_store_set(
            self._fatal_ack_key(self.rank),
            1,
            _deadline=deadline,
        )
        self._wait_for_fatal_acknowledgments(_deadline=deadline)
        with self._fatal_lock:
            self._fatal_protocol_completed = True
        return primary

    def _publish_communicator_fatal(
        self,
        error,
        *,
        admitted=False,
        fatal_owner=None,
        discovering_token=None,
        join_existing=True,
        handler_override=None,
        transition=None,
        diagnostics=(),
    ):
        with self._communicator_fatal_reservation(
            error,
            admitted=admitted,
            fatal_owner=fatal_owner,
            discovering_token=discovering_token,
            join_existing=join_existing,
            handler_override=handler_override,
            transition=transition,
            diagnostics=diagnostics,
        ) as reservation:
            if reservation.started and not reservation.completion_deferred:
                self._run_fatal_handler(reservation)
            return reservation.primary

    def _enter_observed_fatal_locked(self, error, origin_rank):
        return self._enter_observed_fatal(error, origin_rank, join_existing=True)

    def _enter_observed_fatal(
        self,
        error,
        origin_rank,
        *,
        admitted=False,
        release_admitted=False,
        join_existing=False,
        on_elected=None,
        discovering_token=None,
    ):
        if admitted and hasattr(self._active_broadcast_local, "deferral"):
            primary = self._defer_active_broadcast_fatal(error, origin_rank)
            if on_elected is not None:
                on_elected(primary)
            return primary
        with self._communicator_fatal_reservation(
            error,
            origin_rank=origin_rank,
            admitted=admitted,
            release_admitted=release_admitted,
            join_existing=join_existing,
            discovering_token=discovering_token,
        ) as reservation:
            if on_elected is not None:
                on_elected(reservation.primary)
            if not reservation.started or reservation.completion_deferred:
                return reservation.primary
            self._run_fatal_handler(reservation)
            return reservation.primary

    def _current_thread_has_active_broadcast(self):
        return hasattr(self._active_broadcast_local, "deferral")

    def _current_fatal_publication_is_deferred(self):
        reservation = getattr(self._fatal_publication_local, "reservation", None)
        return bool(
            reservation is not None and reservation.completion_deferred
        )

    def _defer_current_active_broadcast_fatal(
        self,
        error,
        *,
        origin_rank=None,
        fatal_owner=None,
        discovering_token=None,
        transition=None,
    ):
        if not self._current_thread_has_active_broadcast():
            return None
        if origin_rank is None:
            origin_rank = self.rank
        return self._defer_active_broadcast_fatal(
            error,
            origin_rank,
            fatal_owner=fatal_owner,
            discovering_token=discovering_token,
            transition=transition,
        )

    def _defer_active_broadcast_fatal(
        self,
        error,
        origin_rank,
        *,
        fatal_owner=None,
        discovering_token=None,
        transition=None,
    ):
        deferral = self._active_broadcast_local.deferral
        if deferral is not None:
            if error is not deferral.primary:
                self._retain_fatal_secondary_direct(error)
            reservation = deferral.reservation
            if transition is not None:
                if reservation.transition is None:
                    reservation.transition = transition
                elif reservation.transition is not transition:
                    raise RuntimeError("communicator fatal transition changed")
                if transition.primary is not reservation.primary:
                    raise RuntimeError("communicator fatal primary changed")
            self._refresh_fatal_handler(
                reservation,
                fatal_owner=fatal_owner,
                discovering_token=discovering_token,
            )
            return deferral.primary
        boundary = self._communicator_fatal_reservation(
            error,
            origin_rank=origin_rank,
            admitted=True,
            fatal_owner=fatal_owner,
            discovering_token=discovering_token,
            transition=transition,
            defer_completion=True,
        )
        reservation = boundary.__enter__()
        deferral = _DeferredFatalPublication(boundary, reservation)
        self._active_broadcast_local.deferral = deferral
        return deferral.primary

    def _read_fatal_origin(self, *, _deadline=None):
        deadline = self._inherited_fatal_deadline(_deadline)
        return next(
            (
                rank
                for rank in range(self.size)
                if int(
                    self._fatal_store_get(
                        self._fatal_key(rank),
                        _deadline=deadline,
                    )
                )
                == 1
            ),
            None,
        )

    def _observe_fatal_monitor_stop(self):
        return self._fatal_monitor_handoff.observe_stop()

    def _observe_fatal_monitor_outcome(self):
        return self._fatal_monitor_handoff.observe_outcome()

    def _request_fatal_monitor_stop(self):
        generation = self._fatal_monitor_handoff.request_stop()
        self._fatal_monitor_stop.set()
        return generation

    def _select_fatal_monitor_outcome(
        self, kind, value, *, before_select=None
    ):
        if kind == "fatal_elected":
            return self._fatal_monitor_handoff.select_fatal(
                value, before_select=before_select
            )
        if kind == "stopped_clean":
            if before_select is not None:
                raise ValueError("clean monitor outcome cannot prepare fatal state")
            return self._fatal_monitor_handoff.select_clean(value)
        raise ValueError("fatal monitor outcome is invalid")

    def _reserve_runtime_fatal_outcome(self, primary, *, before_select=None):
        outcome = self._fatal_monitor_handoff.select_fatal(
            primary,
            before_select=before_select,
        )
        if outcome.kind != "fatal_elected":
            raise RuntimeError("clean monitor stop preempted communicator fatal")
        if outcome.primary is not primary:
            self._retain_fatal_secondary_direct(primary)
        return outcome

    def _elect_collective_fatal_outcome(self, candidate, origin_rank):
        with self._fatal_lock:
            primary = (
                candidate
                if self._fatal_pending_primary is None
                else self._fatal_pending_primary
            )
        if primary is not candidate:
            self._retain_fatal_secondary_direct(candidate)

        def pre_reserve_owner():
            reserved, started = self._pre_reserve_fatal_publication(
                primary, origin_rank
            )
            if reserved is not primary:
                raise RuntimeError("communicator fatal primary changed")
            if not started:
                with self._fatal_lock:
                    if self._fatal_publications <= 0:
                        raise RuntimeError(
                            "communicator fatal publication is not joinable"
                        )

        outcome = self._select_fatal_monitor_outcome(
            "fatal_elected",
            primary,
            before_select=pre_reserve_owner,
        )
        if outcome.kind != "fatal_elected":
            raise RuntimeError("clean monitor stop preempted communicator fatal")
        if outcome.primary is not primary:
            self._retain_fatal_secondary_direct(primary)
        with self._fatal_lock:
            self._fatal_monitor_outcome = outcome
            self._fatal_monitor_outcome_confirmed = True
        self._fatal_monitor_stop.set()
        return outcome

    def _confirm_runtime_fatal_outcome(self, primary, *, _deadline=None):
        self._inherited_fatal_deadline(_deadline)
        outcome = self._select_fatal_monitor_outcome("fatal_elected", primary)
        if outcome.kind != "fatal_elected" or outcome.primary is not primary:
            raise RuntimeError("clean monitor stop preempted communicator fatal")
        with self._fatal_lock:
            self._fatal_monitor_outcome = outcome
            self._fatal_monitor_outcome_confirmed = True
        self._fatal_monitor_stop.set()
        return outcome

    def _select_runtime_fatal_outcome(self, primary, *, _deadline=None):
        deadline = self._inherited_fatal_deadline(_deadline)
        with self._fatal_lock:
            confirmed = self._fatal_monitor_outcome_confirmed
            outcome = self._fatal_monitor_outcome
        if not confirmed:
            return self._confirm_runtime_fatal_outcome(
                primary,
                _deadline=deadline,
            )
        if outcome.kind != "fatal_elected" or outcome.primary is not primary:
            raise RuntimeError("clean monitor stop preempted communicator fatal")
        self._fatal_monitor_stop.set()
        return outcome

    def _wait_for_fatal_monitor_outcome(self, generation, *, _deadline=None):
        try:
            deadline = self._inherited_fatal_deadline(_deadline)
            _remaining_lifecycle_time(
                deadline,
                "communicator fatal monitor stop timed out",
            )
            return self._fatal_monitor_handoff.wait_for_outcome(
                generation,
                _deadline=deadline,
            )
        except TimeoutError:
            error = RuntimeError("communicator fatal monitor stop timed out")
            self._fail_stop_fatal_path(error)

    def _wait_for_fatal_monitor_selection(self, *, _deadline=None):
        try:
            deadline = self._inherited_fatal_deadline(_deadline)
            return self._fatal_monitor_handoff.wait_for_selection(
                _deadline=deadline,
            )
        except TimeoutError:
            error = RuntimeError(
                "communicator fatal monitor outcome selection timed out"
            )
            self._fail_stop_fatal_path(error)

    def _acknowledge_fatal_monitor_exit(self, outcome):
        return self._fatal_monitor_handoff.acknowledge_exit(outcome)

    def _wait_for_fatal_monitor_exit(self, outcome, *, _deadline=None):
        try:
            deadline = self._inherited_fatal_deadline(_deadline)
            _remaining_lifecycle_time(
                deadline,
                "communicator fatal monitor exit acknowledgment timed out",
            )
            return self._fatal_monitor_handoff.wait_for_exit(
                outcome,
                _deadline=deadline,
            )
        except TimeoutError:
            error = RuntimeError(
                "communicator fatal monitor exit acknowledgment timed out"
            )
            self._fail_stop_fatal_path(error)

    def _join_fatal_monitor(self, thread, *, _deadline=None):
        if thread is None or thread is threading.current_thread():
            return
        deadline = self._inherited_fatal_deadline(_deadline)
        remaining = _remaining_lifecycle_time(
            deadline,
            "communicator fatal monitor stop timed out",
        )
        thread.join(remaining)
        if thread.is_alive():
            error = RuntimeError("communicator fatal monitor stop timed out")
            self._fail_stop_fatal_path(error)

    def _start_fatal_monitor_publication(self, error, origin_rank):
        deadline = self._inherited_fatal_deadline()

        def publish():
            current = threading.current_thread()
            diagnostics = ()
            publication_succeeded = False
            with self._fatal_condition:
                if (
                    self._fatal_monitor_publisher_thread is not current
                    or self._fatal_monitor_publisher_state != "starting"
                ):
                    return
                self._fatal_monitor_publisher_state = "started"
                self._fatal_condition.notify_all()
            try:
                self._enter_observed_fatal(
                    error,
                    origin_rank,
                    release_admitted=True,
                    join_existing=True,
                )
                publication_succeeded = True
            except BaseException as publication_error:
                if self._current_thread_claimed_fatal_hard_exit():
                    raise
                self._fail_stop_fatal_path(publication_error)
            finally:
                with self._fatal_condition:
                    diagnostics = self._fatal_monitor_publisher_diagnostics
                    self._fatal_monitor_publisher_diagnostics = ()
                    if self._fatal_monitor_publisher_thread is current:
                        self._fatal_monitor_publisher_state = "completed"
                    self._fatal_condition.notify_all()
                if publication_succeeded:
                    self._dispatch_fatal_secondaries(diagnostics)
                else:
                    self._retain_fatal_secondaries_direct(diagnostics)

        thread = threading.Thread(
            target=publish,
            name="renormalizer-fatal-publisher-rank-{}".format(self.rank),
            daemon=True,
        )
        with self._fatal_condition:
            existing = self._fatal_monitor_publisher_thread
            if existing is not None and self._fatal_monitor_publisher_state in {
                "starting",
                "started",
                "completed",
            }:
                return existing
            self._fatal_monitor_publisher_thread = thread
            self._fatal_monitor_publisher_state = "starting"
            self._fatal_condition.notify_all()
        try:
            thread.start()
        except BaseException as start_error:
            dispatch_directly = False
            with self._fatal_condition:
                claimed = (
                    self._fatal_monitor_publisher_thread is thread
                    and self._fatal_monitor_publisher_state
                    in {"started", "completed"}
                )
                if claimed:
                    if self._fatal_monitor_publisher_state == "completed":
                        dispatch_directly = (
                            self._fatal_publication_failure is None
                            and self._fatal_error is not None
                        )
                        if not dispatch_directly:
                            self._retain_fatal_secondary_direct(start_error)
                    elif all(
                        retained is not start_error
                        for retained in self._fatal_monitor_publisher_diagnostics
                    ):
                        self._fatal_monitor_publisher_diagnostics = (
                            *self._fatal_monitor_publisher_diagnostics,
                            start_error,
                        )
                elif self._fatal_monitor_publisher_thread is thread:
                    self._fatal_monitor_publisher_thread = None
                    self._fatal_monitor_publisher_state = "start_failed"
                self._fatal_condition.notify_all()
            if claimed:
                if dispatch_directly:
                    self._dispatch_fatal_secondaries((start_error,))
                return thread
            if thread.ident is not None:
                remaining = _remaining_lifecycle_time(
                    deadline,
                    "remote fatal publisher start recovery timed out",
                )
                thread.join(remaining)
            self._terminalize_pre_owner_fatal_failure(
                error,
                diagnostics=(start_error,),
            )
        return thread

    def _monitor_fatal_records(self):
        outcome = None
        try:
            while True:
                try:
                    outcome, origin_rank = (
                        self._fatal_monitor_handoff.read_store_if_unselected(
                            self._read_fatal_origin
                        )
                    )
                    if outcome is not None:
                        return
                    outcome = self._observe_fatal_monitor_outcome()
                    if outcome is not None:
                        return
                    generation = self._observe_fatal_monitor_stop()
                    if origin_rank is None and generation is not None:
                        outcome = self._observe_fatal_monitor_outcome()
                        if outcome is not None:
                            return
                        outcome, origin_rank = (
                            self._fatal_monitor_handoff.read_store_if_unselected(
                                self._read_fatal_origin
                            )
                        )
                        if outcome is not None:
                            return
                        outcome = self._observe_fatal_monitor_outcome()
                        if outcome is not None:
                            return
                except BaseException as error:
                    if self._current_thread_claimed_fatal_hard_exit():
                        raise
                    if self._fatal_monitor_stop.is_set():
                        outcome = self._observe_fatal_monitor_outcome()
                        return
                    self._fail_stop_fatal_path(error)
                if origin_rank is not None:
                    outcome = self._observe_fatal_monitor_outcome()
                    if outcome is not None:
                        return
                    with self._fatal_condition:
                        existing = (
                            self._fatal_pending_primary
                            if self._fatal_pending_primary is not None
                            else self._fatal_error
                        )
                    error = (
                        existing
                        if existing is not None
                        else _RemoteCommunicatorFailure(origin_rank)
                    )
                    self._start_fatal_monitor_publication(error, origin_rank)
                    outcome = self._wait_for_fatal_monitor_selection()
                    return
                if generation is not None:
                    outcome = self._select_fatal_monitor_outcome(
                        "stopped_clean", generation
                    )
                    return
                if self._fatal_monitor_stop.is_set():
                    outcome = self._observe_fatal_monitor_outcome()
                    return
                self._fatal_monitor_stop.wait(0.001)
        finally:
            if outcome is not None:
                self._fatal_monitor_outcome = outcome
                self._acknowledge_fatal_monitor_exit(outcome)

    def _start_fatal_monitor(self, *, _deadline=None):
        def monitor():
            current = threading.current_thread()
            with self._fatal_condition:
                if (
                    self._fatal_monitor_thread is not current
                    or self._fatal_monitor_state != "starting"
                ):
                    return
                self._fatal_monitor_state = "started"
                self._fatal_condition.notify_all()
            try:
                self._monitor_fatal_records()
            finally:
                with self._fatal_condition:
                    if self._fatal_monitor_thread is current:
                        self._fatal_monitor_state = "completed"
                    self._fatal_condition.notify_all()

        with self._fatal_condition:
            existing = self._fatal_monitor_thread
            if existing is not None and self._fatal_monitor_state in {
                "starting",
                "started",
            }:
                return existing
            self._fatal_monitor_stop.clear()
            thread = threading.Thread(
                target=monitor,
                name="renormalizer-fatal-monitor-rank-{}".format(self.rank),
                daemon=True,
            )
            self._fatal_monitor_thread = thread
            self._fatal_monitor_state = "starting"
            self._fatal_condition.notify_all()
        try:
            thread.start()
        except BaseException as start_error:
            with self._fatal_condition:
                claimed = (
                    self._fatal_monitor_thread is thread
                    and self._fatal_monitor_state in {"started", "completed"}
                )
                if not claimed and self._fatal_monitor_thread is thread:
                    self._fatal_monitor_thread = None
                    self._fatal_monitor_state = "start_failed"
                self._fatal_condition.notify_all()
            if claimed:
                self._retain_fatal_secondary_direct(start_error)
                return thread
            if thread.ident is not None:
                try:
                    remaining = _remaining_lifecycle_time(
                        _deadline,
                        "fatal monitor start recovery timed out",
                    )
                    thread.join(
                        _FATAL_TIMEOUT_S if remaining is None else remaining
                    )
                    if thread.is_alive():
                        raise TimeoutError(
                            "fatal monitor start recovery timed out"
                        )
                except BaseException as recovery_error:
                    self._retain_fatal_secondary_direct(recovery_error)
            raise
        return thread

    def _validate_array(self, array, *, op=None):
        if not isinstance(array, self._cupy.ndarray):
            raise TypeError("CuPy NCCL collectives require a cupy.ndarray")
        if int(array.device.id) != self._device_index:
            raise ValueError("collective array is not on the selected CUDA device")
        if int(array.size) == 0:
            raise ValueError("collective arrays must not be empty")
        if array.dtype.char not in self._SUPPORTED_DTYPE_CHARS:
            raise TypeError(
                "dtype {} is not supported by NCCL".format(array.dtype.name)
            )
        if array.dtype.kind == "c" and op not in (None, "sum"):
            raise ValueError("complex arrays only support sum reduction")

    def _validate_root(self, root):
        if type(root) is not int:
            raise TypeError("root must be an integer")
        if root < 0 or root >= self.size:
            raise ValueError(
                "root {} is out of range for collective size {}".format(root, self.size)
            )

    def barrier(self):
        self._begin_collective_operation()
        try:
            return self._execute_admitted_collective(self._backend.barrier)
        finally:
            self._finish_collective_operation()

    def _validate_bootstrap_status_or(self):
        self._require_operational()
        if not self._fatal_control_initialized:
            raise RuntimeError(
                "CuPy NCCL bootstrap store is unavailable for status provisioning"
            )

    def _local_fatal_capability_code(self, *, _deadline=None):
        local_code = 0
        try:
            from cupyx.distributed._store import TCPStoreProxy

            store_proxy = getattr(self._backend, "_store_proxy")
            proxy_type = type(store_proxy)
            if (
                store_proxy is None
                or proxy_type is not TCPStoreProxy
                or not callable(getattr(store_proxy, "barrier", None))
                or not callable(getattr(proxy_type, "__getitem__", None))
                or not callable(getattr(proxy_type, "__setitem__", None))
            ):
                raise RuntimeError("private store contract is unavailable")
            probe_key = "{}{}".format(_STATUS_WORKSPACE_KEY_PREFIX, self.rank)
            self._store_set(probe_key, 0, _deadline=_deadline)
            if int(self._store_get(probe_key, _deadline=_deadline)) != 0:
                raise RuntimeError("private store round trip failed")
        except BaseException:
            local_code |= _FATAL_CAPABILITY_ERROR
        try:
            raw_comm = getattr(self._backend, "_comm")
            if not callable(getattr(raw_comm, "abort", None)):
                raise RuntimeError("private communicator abort is unavailable")
        except BaseException:
            local_code |= _FATAL_CAPABILITY_ERROR
        return local_code

    def _bootstrap_fatal_control(self, *, _deadline=None):
        deadline = self._store_deadline(_deadline)
        self._require_operational()
        if self._fatal_control_initialized:
            return 0
        store_proxy = self._bootstrap_store_proxy
        proxy_type = type(store_proxy)
        if (
            store_proxy is None
            or not callable(getattr(store_proxy, "barrier", None))
            or not callable(getattr(proxy_type, "__getitem__", None))
            or not callable(getattr(proxy_type, "__setitem__", None))
        ):
            raise RuntimeError(
                "independent CuPy bootstrap control store is unavailable"
            )
        local_code = self._local_fatal_capability_code(_deadline=deadline)
        local_status_key = "{}{}".format(_STATUS_WORKSPACE_KEY_PREFIX, self.rank)
        self._store_set(local_status_key, local_code, _deadline=deadline)
        self._store_set(self._fatal_key(self.rank), 0, _deadline=deadline)
        self._store_set(self._fatal_ack_key(self.rank), 0, _deadline=deadline)
        self._store_set(
            self._active_b_key(self.rank),
            self._encode_active_b(0, 0),
            _deadline=deadline,
        )
        self._store_set(self._close_ready_key(self.rank), 0, _deadline=deadline)
        self._store_set(
            self._monitor_stopped_key(self.rank),
            0,
            _deadline=deadline,
        )
        self._store_set(
            self._close_consumed_key(self.rank),
            0,
            _deadline=deadline,
        )
        self._store_set(_CLOSE_RELEASE_KEY, 0, _deadline=deadline)
        self._store_barrier(_deadline=deadline)
        aggregate = 0
        for rank in range(self.size):
            aggregate |= int(
                self._store_get(
                    "{}{}".format(_STATUS_WORKSPACE_KEY_PREFIX, rank),
                    _deadline=deadline,
                )
            )
            int(self._store_get(self._fatal_key(rank), _deadline=deadline))
            int(self._store_get(self._fatal_ack_key(rank), _deadline=deadline))
            sequence, failed = self._decode_active_b(
                self._store_get(self._active_b_key(rank), _deadline=deadline)
            )
            if (int(sequence), int(failed)) != (0, 0):
                aggregate |= _FATAL_CAPABILITY_ERROR
            if (
                int(
                    self._store_get(
                        self._close_ready_key(rank),
                        _deadline=deadline,
                    )
                )
                != 0
                or int(
                    self._store_get(
                        self._monitor_stopped_key(rank),
                        _deadline=deadline,
                    )
                )
                != 0
                or int(
                    self._store_get(
                        self._close_consumed_key(rank),
                        _deadline=deadline,
                    )
                )
                != 0
            ):
                aggregate |= _FATAL_CAPABILITY_ERROR
        if int(self._store_get(_CLOSE_RELEASE_KEY, _deadline=deadline)) != 0:
            aggregate |= _FATAL_CAPABILITY_ERROR
        self._store_barrier(_deadline=deadline)
        if aggregate == 0:
            try:
                monitor = self._start_fatal_monitor(_deadline=deadline)
                with self._fatal_condition:
                    monitor_owned = (
                        monitor is not None
                        and self._fatal_monitor_thread is monitor
                        and self._fatal_monitor_state
                        in {"started", "completed"}
                    )
                if not monitor_owned:
                    raise RuntimeError(
                        "fatal monitor start returned without owned execution"
                    )
            except BaseException as start_error:
                with self._fatal_condition:
                    self._fatal_control_initialized = False
                    self._fatal_condition.notify_all()
                primary = self._terminalize_pre_owner_fatal_failure(start_error)
                raise primary
            with self._fatal_condition:
                self._fatal_control_initialized = True
                self._fatal_condition.notify_all()
        return aggregate

    def _bootstrap_status_or(self, local_code, *, _deadline=None):
        deadline = self._store_deadline(_deadline)
        if type(local_code) is not int or local_code < 0:
            raise ValueError("bootstrap status code must be a non-negative integer")
        self._validate_bootstrap_status_or()
        local_key = "{}{}".format(_STATUS_WORKSPACE_KEY_PREFIX, self.rank)
        self._store_set(local_key, local_code, _deadline=deadline)
        self._store_barrier(_deadline=deadline)
        aggregate = 0
        for rank in range(self.size):
            aggregate |= int(
                self._store_get(
                    "{}{}".format(_STATUS_WORKSPACE_KEY_PREFIX, rank),
                    _deadline=deadline,
                )
            )
        self._store_barrier(_deadline=deadline)
        return aggregate

    def _broadcast_unadmitted(self, array, *, root):
        try:
            self._validate_root(root)
            self._validate_array(array)
        except BaseException as error:
            if not self._current_thread_has_active_broadcast():
                raise
            primary = self._defer_active_broadcast_fatal(error, self.rank)
            if primary is error:
                raise
            raise primary

        def execute():
            cupy = self._cupy
            with cupy.cuda.Device(self._device_index):
                stream = cupy.cuda.get_current_stream()
                contiguous = cupy.ascontiguousarray(array)
                self._backend.broadcast(contiguous, root=root, stream=stream)
                if contiguous is not array:
                    cupy.copyto(array, contiguous)
            return array

        return self._execute_admitted_collective(execute)

    @contextmanager
    def _active_broadcast_admission(self):
        self._begin_collective_operation(active_broadcast=True)
        self._active_broadcast_local.deferral = None
        try:
            yield self._broadcast_unadmitted, self._agree_admitted_active_broadcast
        finally:
            deferral = self._active_broadcast_local.deferral
            try:
                self._finish_collective_operation(active_broadcast=True)
                if deferral is not None:
                    deferral.complete()
            finally:
                del self._active_broadcast_local.deferral

    def broadcast(self, array, *, root):
        self._begin_collective_operation()
        try:
            return self._broadcast_unadmitted(array, root=root)
        finally:
            self._finish_collective_operation()

    def allreduce(self, array, *, op="sum"):
        self._begin_collective_operation()
        try:
            _validate_reduction_op(op)
            self._validate_array(array, op=op)

            def execute():
                cupy = self._cupy
                with cupy.cuda.Device(self._device_index):
                    stream = cupy.cuda.get_current_stream()
                    input_buffer = cupy.ascontiguousarray(array)
                    output_buffer = cupy.empty_like(input_buffer)
                    self._backend.all_reduce(
                        input_buffer, output_buffer, op=op, stream=stream
                    )
                return output_buffer

            return self._execute_admitted_collective(execute)
        finally:
            self._finish_collective_operation()

    def allreduce_inplace(self, array, *, op="sum"):
        self._begin_collective_operation()
        try:
            _validate_reduction_op(op)
            self._validate_array(array, op=op)
            if not bool(array.flags.c_contiguous):
                raise ValueError("in-place allreduce requires a C-contiguous array")

            def execute():
                cupy = self._cupy
                with cupy.cuda.Device(self._device_index):
                    stream = cupy.cuda.get_current_stream()
                    self._backend.all_reduce(array, array, op=op, stream=stream)
                return array

            return self._execute_admitted_collective(execute)
        finally:
            self._finish_collective_operation()

    def reduce_scatter(self, array, *, axis, op="sum"):
        self._begin_collective_operation()
        try:
            normalized_axis = _normalize_axis(array, axis)
            _validate_reduction_op(op)
            self._validate_array(array, op=op)
            axis_length = int(array.shape[normalized_axis])
            if axis_length % self.size != 0:
                raise ValueError(
                    "axis length {} is not divisible by collective size {}".format(
                        axis_length, self.size
                    )
                )

            def execute():
                cupy = self._cupy
                with cupy.cuda.Device(self._device_index):
                    stream = cupy.cuda.get_current_stream()
                    moved = cupy.moveaxis(array, normalized_axis, 0)
                    input_buffer = cupy.ascontiguousarray(moved).reshape(-1)
                    output_buffer = cupy.empty(
                        input_buffer.size // self.size, dtype=input_buffer.dtype
                    )
                    self._backend.reduce_scatter(
                        input_buffer,
                        output_buffer,
                        int(output_buffer.size),
                        op=op,
                        stream=stream,
                    )
                    output_shape = (axis_length // self.size,) + tuple(moved.shape[1:])
                    result = cupy.moveaxis(
                        output_buffer.reshape(output_shape), 0, normalized_axis
                    )
                return result

            return self._execute_admitted_collective(execute)
        finally:
            self._finish_collective_operation()

    def allgather(self, array, *, axis):
        self._begin_collective_operation()
        try:
            normalized_axis = _normalize_axis(array, axis)
            self._validate_array(array)

            def execute():
                cupy = self._cupy
                with cupy.cuda.Device(self._device_index):
                    stream = cupy.cuda.get_current_stream()
                    moved = cupy.moveaxis(array, normalized_axis, 0)
                    input_buffer = cupy.ascontiguousarray(moved).reshape(-1)
                    output_buffer = cupy.empty(
                        input_buffer.size * self.size, dtype=input_buffer.dtype
                    )
                    self._backend.all_gather(
                        input_buffer,
                        output_buffer,
                        int(input_buffer.size),
                        stream=stream,
                    )
                    output_shape = (int(moved.shape[0]) * self.size,) + tuple(
                        moved.shape[1:]
                    )
                    result = cupy.moveaxis(
                        output_buffer.reshape(output_shape), 0, normalized_axis
                    )
                return result

            return self._execute_admitted_collective(execute)
        finally:
            self._finish_collective_operation()

    def _agree_active_broadcast(self, failed):
        return self._agree_active_broadcast_core(failed, admitted=False)

    def _agree_admitted_active_broadcast(self, failed):
        return self._agree_active_broadcast_core(failed, admitted=True)

    def _agree_active_broadcast_core(self, failed, *, admitted):
        if type(failed) is not bool:
            raise TypeError("active broadcast failure flag must be a boolean")
        with self._fatal_lock:
            sequence = self._active_broadcast_sequence
            self._active_broadcast_sequence += 1
        deadline = self._inherited_fatal_deadline()
        self._fatal_store_set(
            self._active_b_key(self.rank),
            self._encode_active_b(sequence, failed),
            _deadline=deadline,
        )
        records = None
        while records is None:
            observed = []
            complete = True
            for rank in range(self.size):
                value = self._fatal_store_get(
                    self._active_b_key(rank),
                    _deadline=deadline,
                )
                try:
                    observed_sequence, observed_failed = self._decode_active_b(value)
                    observed_sequence = int(observed_sequence)
                    observed_failed = int(observed_failed)
                except BaseException as error:
                    protocol_error = RuntimeError(
                        "active broadcast result record is invalid"
                    )
                    protocol_error.__cause__ = error
                    self._enter_observed_fatal(
                        protocol_error, self.rank, admitted=admitted
                    )
                    self._raise_terminal()
                if observed_sequence > sequence:
                    protocol_error = RuntimeError(
                        "active broadcast result sequence advanced unexpectedly"
                    )
                    self._enter_observed_fatal(
                        protocol_error, self.rank, admitted=admitted
                    )
                    self._raise_terminal()
                if observed_sequence != sequence:
                    complete = False
                observed.append((rank, bool(observed_failed)))
            if complete:
                records = observed
                break
            if time.monotonic() >= deadline:
                timeout_error = RuntimeError("active broadcast agreement timed out")
                self._enter_observed_fatal(timeout_error, self.rank, admitted=admitted)
                self._raise_terminal()
            time.sleep(0.001)
        failed_ranks = [rank for rank, rank_failed in records if rank_failed]
        if failed_ranks:
            with self._fatal_lock:
                primary = self._fatal_error
                if primary is None:
                    primary = self._fatal_pending_primary
            if primary is None:
                origin_rank = failed_ranks[0]
                primary = _RemoteCommunicatorFailure(origin_rank)
            else:
                origin_rank = self._fatal_pending_origin_rank
                if origin_rank is None:
                    origin_rank = failed_ranks[0]
            self._enter_observed_fatal(primary, origin_rank, admitted=admitted)
            self._raise_terminal()
        if admitted:
            with self._fatal_lock:
                fatal_error = self._fatal_error
                closed = self._closed
            if fatal_error is not None:
                raise RuntimeError("collective is terminal-aborted") from fatal_error
            if closed:
                raise RuntimeError("collective is closed")
        else:
            self._require_operational()

    def _wait_for_control_value(
        self,
        key,
        timeout_message,
        *,
        _deadline=None,
    ):
        deadline = (
            time.monotonic() + _FATAL_TIMEOUT_S
            if _deadline is None
            else _deadline
        )
        while int(self._fatal_store_get(key, _deadline=deadline)) != 1:
            if time.monotonic() >= deadline:
                error = RuntimeError(timeout_message)
                self._diagnose_and_hard_exit(error)
            time.sleep(0.001)

    def _consume_terminal_control_records(
        self,
        discovering_token=None,
        *,
        _deadline=None,
    ):
        fatal = []
        for rank in range(self.size):
            fatal.append(
                int(
                    self._fatal_store_get(
                        self._fatal_key(rank),
                        _deadline=_deadline,
                    )
                )
            )
            self._decode_active_b(
                self._fatal_store_get(
                    self._active_b_key(rank),
                    _deadline=_deadline,
                )
            )
        if any(fatal):
            origin_rank = fatal.index(1)
            with self._fatal_lock:
                existing = (
                    self._fatal_pending_primary
                    if self._fatal_pending_primary is not None
                    else self._fatal_error
                )
            error = (
                existing
                if existing is not None
                else _RemoteCommunicatorFailure(origin_rank)
            )
            self._enter_observed_fatal(
                error,
                origin_rank,
                join_existing=True,
                discovering_token=discovering_token,
            )
            return self._fatal_pending_primary
        return None

    def _wait_for_close_ready(self, discovering_token=None, *, _deadline=None):
        deadline = (
            time.monotonic() + _FATAL_TIMEOUT_S
            if _deadline is None
            else _deadline
        )
        while True:
            primary = self._consume_terminal_control_records(
                discovering_token,
                _deadline=deadline,
            )
            if primary is not None:
                return primary
            if all(
                int(
                    self._fatal_store_get(
                        self._close_ready_key(rank),
                        _deadline=deadline,
                    )
                )
                == 1
                for rank in range(self.size)
            ):
                return None
            if time.monotonic() >= deadline:
                error = RuntimeError("communicator close intent timed out")
                self._diagnose_and_hard_exit(error)
            time.sleep(0.001)

    def _quiesce_operations_for_close(self, *, _deadline=None):
        deadline = (
            time.monotonic() + _FATAL_TIMEOUT_S
            if _deadline is None
            else _deadline
        )
        fail_stop_error = None
        with self._fatal_condition:
            self._closing = True
            while self._admitted_operations or self._fatal_publications:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    fail_stop_error = RuntimeError(
                        "communicator fatal publication quiescence timed out"
                    )
                    break
                self._fatal_condition.wait(remaining)
        if fail_stop_error is not None:
            self._diagnose_and_hard_exit(fail_stop_error)

    @staticmethod
    def _release_runtime_close_step(gate, token):
        if gate is None or token is None:
            return
        try:
            gate.release(token)
        except RuntimeError as error:
            if "converted" not in str(error):
                raise

    def _admit_runtime_close_step(self, gate, transition, operation):
        if gate is None:
            return None
        return gate.admit_runtime_close(transition, operation)

    def _close_initialized_control(
        self,
        backend,
        gate=None,
        transition=None,
        *,
        _deadline=None,
    ):
        ready_token = self._admit_runtime_close_step(
            gate, transition, "collective_close_ready"
        )
        try:
            self._fatal_store_set(
                self._close_ready_key(self.rank),
                1,
                _deadline=_deadline,
            )
            primary = self._wait_for_close_ready(
                ready_token,
                _deadline=_deadline,
            )
            if primary is None:
                primary = self._consume_terminal_control_records(
                    ready_token,
                    _deadline=_deadline,
                )
        finally:
            self._release_runtime_close_step(gate, ready_token)

        if primary is not None:
            self._wait_for_joined_fatal_publication(_deadline=_deadline)
            return self._fatal_monitor_outcome

        stop_token = self._admit_runtime_close_step(
            gate, transition, "collective_monitor_stop"
        )
        try:
            generation = self._request_fatal_monitor_stop()
        finally:
            self._release_runtime_close_step(gate, stop_token)

        outcome = self._wait_for_fatal_monitor_outcome(
            generation,
            _deadline=_deadline,
        )
        self._wait_for_fatal_monitor_exit(outcome, _deadline=_deadline)
        thread = self._fatal_monitor_thread
        self._join_fatal_monitor(thread, _deadline=_deadline)
        if outcome.kind == "fatal_elected":
            self._wait_for_joined_fatal_publication(_deadline=_deadline)
            return outcome

        if gate is not None:
            selected = gate.select_runtime_close_commit(transition)
            if selected is not transition:
                self._wait_for_joined_fatal_publication(_deadline=_deadline)
                return self._fatal_monitor_outcome

        close_token = self._admit_runtime_close_step(
            gate, transition, "collective_close"
        )
        try:
            self._fatal_store_set(
                self._monitor_stopped_key(self.rank),
                1,
                _deadline=_deadline,
            )
            self._wait_for_all_control_records(
                self._monitor_stopped_key,
                "communicator monitor-stop agreement timed out",
                _deadline=_deadline,
            )

            if self.rank == 0:
                self._fatal_store_set(
                    _CLOSE_RELEASE_KEY,
                    1,
                    _deadline=_deadline,
                )
            else:
                self._wait_for_control_value(
                    _CLOSE_RELEASE_KEY,
                    "communicator close release timed out",
                    _deadline=_deadline,
                )
                backend.stop()
            self._fatal_store_set(
                self._close_consumed_key(self.rank),
                1,
                _deadline=_deadline,
            )
            if self.rank == 0:
                self._wait_for_all_control_records(
                    self._close_consumed_key,
                    "communicator close consumption timed out",
                    _deadline=_deadline,
                )
                backend.stop()
        finally:
            self._release_runtime_close_step(gate, close_token)
        return outcome

    def _claim_close_owner(self):
        with self._close_lock:
            if self._closed:
                return False, True
            if self._close_in_progress:
                return False, False
            self._close_in_progress = True
            return True, False

    def _release_close_owner(self):
        with self._close_lock:
            self._close_in_progress = False
        with self._fatal_condition:
            self._fatal_condition.notify_all()

    def _wait_for_close_owner(self, *, _deadline=None):
        deadline = (
            time.monotonic() + _FATAL_TIMEOUT_S
            if _deadline is None
            else _deadline
        )
        fail_stop_error = None
        with self._fatal_condition:
            while self._close_in_progress and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    fail_stop_error = RuntimeError(
                        "communicator close join timed out"
                    )
                    break
                self._fatal_condition.wait(remaining)
        if fail_stop_error is not None:
            self._diagnose_and_hard_exit(fail_stop_error)

    def _close_impl(self, gate=None, transition=None, *, _deadline=None):
        if _deadline is None:
            _deadline = (
                transition.deadline
                if transition is not None
                else time.monotonic() + _FATAL_TIMEOUT_S
            )
        while True:
            _remaining_lifecycle_time(
                _deadline,
                "communicator close lifecycle timed out before owner claim",
            )
            owns_close, closed = self._claim_close_owner()
            if closed:
                return None
            if owns_close:
                break
            self._wait_for_close_owner(_deadline=_deadline)
        try:
            backend = self._backend
            with self._fatal_lock:
                pending = self._fatal_pending_primary is not None
                thread = self._fatal_monitor_thread
            if (
                self._fatal_control_initialized
                and not pending
                and (thread is None or not thread.is_alive())
            ):
                restart_token = self._admit_runtime_close_step(
                    gate, transition, "collective_monitor_start"
                )
                try:
                    self._start_fatal_monitor(_deadline=_deadline)
                finally:
                    self._release_runtime_close_step(gate, restart_token)
            self._quiesce_operations_for_close(_deadline=_deadline)
            with self._fatal_lock:
                terminal = self._fatal_error is not None
            try:
                if terminal:
                    outcome = None
                elif self._fatal_control_initialized:
                    outcome = self._close_initialized_control(
                        backend,
                        gate=gate,
                        transition=transition,
                        _deadline=_deadline,
                    )
                else:
                    self._fatal_monitor_stop.set()
                    thread = self._fatal_monitor_thread
                    if thread is not None and thread is not threading.current_thread():
                        remaining = _remaining_lifecycle_time(
                            _deadline,
                            "communicator close lifecycle timed out joining monitor",
                        )
                        thread.join(remaining)
                    selected = (
                        transition
                        if gate is None
                        else gate.select_runtime_close_commit(transition)
                    )
                    if selected is transition:
                        backend.stop()
                        outcome = None
                    else:
                        self._wait_for_joined_fatal_publication(
                            _deadline=_deadline,
                        )
                        outcome = self._fatal_monitor_outcome
            except BaseException:
                if not self._fatal_control_initialized:
                    with self._fatal_condition:
                        self._closing = False
                        self._fatal_condition.notify_all()
                raise
            fatal_retained = self._fatal_error is not None or (
                outcome is not None and outcome.kind == "fatal_elected"
            )
            if not fatal_retained:
                self._backend = None
                self._bootstrap_store_proxy = None
                self._fatal_handler = None
                self._fatal_transition_handler = None
                self._fatal_monitor_thread = None
                self._closed = True
            return outcome
        finally:
            self._release_close_owner()

    def _close_for_runtime(self, gate, transition):
        return self._close_impl(
            gate=gate,
            transition=transition,
            _deadline=transition.deadline,
        )

    def close(self):
        self._close_impl()
