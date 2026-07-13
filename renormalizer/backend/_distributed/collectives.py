"""Collective contracts and local collective behavior."""

from contextlib import contextmanager
import os
import struct
import threading
import time
from typing import Any, Protocol, runtime_checkable
import weakref

from renormalizer.backend._distributed.terminal import (
    _FatalMonitorHandoff,
    _TERMINAL_TIMEOUT_S,
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
        "handler",
        "transition_handler",
        "handler_started",
        "transition",
        "snapshot",
        "completion",
    )

    def __init__(
        self,
        primary,
        installed,
        started,
        joined=False,
        monitor_deferred=False,
        handler=None,
        transition_handler=None,
        transition=None,
    ):
        self.primary = primary
        self.installed = installed
        self.started = started
        self.joined = joined
        self.monitor_deferred = monitor_deferred
        self.handler = handler
        self.transition_handler = transition_handler
        self.handler_started = False
        self.transition = transition
        self.snapshot = None
        self.completion = None


class _FatalPublicationOwnerReservation:
    __slots__ = ("primary", "owner_thread_id", "installed")

    def __init__(self, primary, owner_thread_id, installed):
        self.primary = primary
        self.owner_thread_id = owner_thread_id
        self.installed = installed


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
    def _bootstrap_fatal_control():
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
    def _bootstrap_status_or(local_code):
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
        self._admitted_operations = 0
        self._fatal_publications = 0
        self._fatal_publication_owner_reservation = None
        self._closing = False
        self._fatal_pending_primary = None
        self._fatal_pending_origin_rank = None
        self._fatal_error = None
        self._fatal_origin_rank = None
        self._fatal_secondary_errors = ()
        self._fatal_store_announced = False
        self._fatal_abort_started = False
        self._fatal_abort_completed = False
        self._fatal_abort_event = threading.Event()
        self._fatal_abort_error = None
        self._fatal_protocol_completed = False
        self._fatal_handler = None
        self._fatal_transition_handler = None
        self._fatal_monitor_stop = threading.Event()
        self._fatal_monitor_thread = None
        self._fatal_monitor_handoff = _FatalMonitorHandoff()
        self._fatal_monitor_outcome = None
        self._fatal_monitor_outcome_confirmed = False
        self._fatal_monitor_publisher_thread = None
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

    def _begin_collective_operation(self):
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

    def _finish_collective_operation(self):
        with self._fatal_condition:
            if self._admitted_operations <= 0:
                raise RuntimeError("collective operation count is inconsistent")
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

    def _install_fatal(self, error, origin_rank):
        if not isinstance(error, BaseException):
            raise TypeError("communicator fatal failure must be an exception")
        with self._fatal_lock:
            installed = self._fatal_pending_primary is None
            if installed:
                self._fatal_pending_primary = error
                self._fatal_pending_origin_rank = origin_rank
            return self._fatal_pending_primary, installed

    def _pre_reserve_fatal_publication(self, error, origin_rank=None):
        """Install one joinable owner before exposing fatal monitor selection."""
        if not isinstance(error, BaseException):
            raise TypeError("communicator fatal failure must be an exception")
        with self._fatal_condition:
            if self._fatal_protocol_completed:
                return self._fatal_pending_primary, False
            if self._fatal_publications > 0:
                return self._fatal_pending_primary, False
            if self._closed:
                raise RuntimeError("collective is closed")
            primary, installed = self._install_fatal(error, origin_rank)
            self._fatal_publications += 1
            self._fatal_publication_owner_reservation = (
                _FatalPublicationOwnerReservation(
                    primary,
                    threading.get_ident(),
                    installed,
                )
            )
            return primary, True

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
                and owner_reservation.owner_thread_id == threading.get_ident()
            ):
                if owner_reservation.primary is not error:
                    raise RuntimeError("communicator fatal primary changed")
                if self._fatal_pending_origin_rank is None:
                    self._fatal_pending_origin_rank = origin_rank
                self._fatal_publication_owner_reservation = None
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
            self._fatal_publications += 1
            return primary, installed, True, False, False

    def _finish_fatal_publication(self):
        with self._fatal_condition:
            if self._fatal_publications <= 0:
                raise RuntimeError("fatal publication count is inconsistent")
            self._fatal_publications -= 1
            self._fatal_condition.notify_all()

    def _wait_for_joined_fatal_publication(self):
        deadline = time.monotonic() + _FATAL_TIMEOUT_S
        with self._fatal_condition:
            while self._fatal_publications:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    error = RuntimeError(
                        "communicator fatal publication join timed out"
                    )
                    self._record_fatal_secondary(error)
                    self._fatal_hard_exit()
                self._fatal_condition.wait(remaining)
            if not self._fatal_protocol_completed:
                error = RuntimeError(
                    "communicator fatal publication ended without completion"
                )
                self._record_fatal_secondary(error)
                self._fatal_hard_exit()

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
            self._refresh_fatal_handler(
                active,
                fatal_owner=fatal_owner,
                discovering_token=discovering_token,
            )
            yield active
            return
        if origin_rank is None:
            origin_rank = self.rank
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
                raise TypeError("communicator fatal transition primary is invalid")
            if error is not primary:
                self._record_fatal_secondary(error)
            outcome = self._observe_fatal_monitor_outcome()
            if (
                outcome is None
                or outcome.kind != "fatal_elected"
                or outcome.primary is not primary
            ):
                raise RuntimeError("communicator fatal handoff was not reserved")
        else:
            outcome = self._elect_collective_fatal_outcome(error, origin_rank)
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
        if transition is not None and transition.primary is not primary:
            raise RuntimeError("communicator fatal primary changed")
        if transition is not None and started:
            self._confirm_runtime_fatal_outcome(primary)
        reservation = _FatalPublicationReservation(
            primary,
            installed,
            started,
            joined=joined,
            monitor_deferred=monitor_deferred,
            handler=handler,
            transition_handler=transition_handler,
            transition=transition,
        )
        if not started:
            if monitor_deferred:
                outcome = self._wait_for_fatal_monitor_selection()
                if outcome.kind != "fatal_elected" or outcome.primary is not primary:
                    raise RuntimeError("deferred monitor fatal outcome changed")
                self._fatal_monitor_outcome = outcome
            if joined:
                self._refresh_fatal_handler(
                    reservation,
                    fatal_owner=fatal_owner,
                    discovering_token=discovering_token,
                )
            try:
                yield reservation
            finally:
                if joined:
                    self._wait_for_joined_fatal_publication()
            return
        try:
            self._fatal_publication_local.reservation = reservation
            self._refresh_fatal_handler(
                reservation,
                fatal_owner=fatal_owner,
                discovering_token=discovering_token,
            )
            self._prepare_local_fatal_locked()
            try:
                yield reservation
            except TimeoutError as error:
                self._record_fatal_secondary(error)
                self._fatal_hard_exit()
                raise
            else:
                self._run_fatal_handler(reservation)
                self._publish_communicator_fatal_locked(primary)
                self._complete_fatal_handler(reservation)
        finally:
            try:
                del self._fatal_publication_local.reservation
            finally:
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
        with self._fatal_lock:
            self._fatal_handler = reference
            if callable(getattr(handler, "begin", None)):
                self._fatal_transition_handler = reference

    def _fatal_handler_callback(self):
        with self._fatal_lock:
            reference = self._fatal_handler
        return None if reference is None else reference()

    def _fatal_transition_handler_callback(self):
        with self._fatal_lock:
            reference = self._fatal_transition_handler
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
            return reservation.snapshot
        reservation.handler_started = True
        publish = getattr(reservation.handler, "publish", None)
        try:
            if callable(publish):
                reservation.snapshot = publish(reservation.transition)
            else:
                reservation.snapshot = reservation.handler(reservation.primary)
            if (
                reservation.transition is not None
                and reservation.transition_handler is not None
                and reservation.transition_handler is not reservation.handler
                and reservation.completion is None
            ):
                publish_transition = getattr(
                    reservation.transition_handler, "publish", None
                )
                if callable(publish_transition):
                    reservation.snapshot = publish_transition(reservation.transition)
        except TimeoutError as error:
            self._record_fatal_secondary(error)
            self._fatal_hard_exit()
            raise
        return reservation.snapshot

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

    def _fatal_store_set(self, key, value):
        try:
            self._bootstrap_store_proxy[key] = value
        except BaseException as error:
            self._record_fatal_secondary(error)
            self._fatal_hard_exit()

    def _fatal_store_get(self, key):
        try:
            return self._bootstrap_store_proxy[key]
        except BaseException as error:
            self._record_fatal_secondary(error)
            self._fatal_hard_exit()

    def _abort_local_communicator(self):
        with self._fatal_lock:
            if not self._fatal_abort_started:
                self._fatal_abort_started = True

                def abort():
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
                        with self._fatal_lock:
                            self._fatal_abort_error = error
                    else:
                        with self._fatal_lock:
                            self._fatal_abort_completed = True
                    finally:
                        self._fatal_abort_event.set()

                threading.Thread(
                    target=abort,
                    name="renormalizer-nccl-abort-rank-{}".format(self.rank),
                    daemon=True,
                ).start()
            return self._fatal_abort_event

    def _wait_for_local_communicator_abort(self):
        event = self._fatal_abort_event
        if not event.wait(_FATAL_TIMEOUT_S):
            error = RuntimeError("NCCL communicator abort timed out")
            self._record_fatal_secondary(error)
            self._fatal_hard_exit()
        with self._fatal_lock:
            error = self._fatal_abort_error
            completed = self._fatal_abort_completed
        if error is not None:
            self._record_fatal_secondary(error)
            self._fatal_hard_exit()
        if not completed:
            error = RuntimeError("NCCL communicator abort did not complete")
            self._record_fatal_secondary(error)
            self._fatal_hard_exit()

    def _wait_for_fatal_acknowledgments(self):
        self._wait_for_all_control_records(
            self._fatal_ack_key,
            "communicator fatal acknowledgment timed out",
        )

    def _wait_for_all_control_records(self, key, timeout_message):
        deadline = time.monotonic() + _FATAL_TIMEOUT_S
        while True:
            if all(
                int(self._fatal_store_get(key(rank))) == 1 for rank in range(self.size)
            ):
                return
            if time.monotonic() >= deadline:
                error = RuntimeError(timeout_message)
                self._record_fatal_secondary(error)
                self._fatal_hard_exit()
            time.sleep(0.001)

    def _prepare_local_fatal_locked(self):
        with self._fatal_lock:
            announce = not self._fatal_store_announced
            self._fatal_store_announced = True
            primary = self._fatal_pending_primary
            monitor = self._fatal_monitor_thread
            monitor_live = monitor is not None and monitor.is_alive()
        outcome = self._select_runtime_fatal_outcome(primary)
        if monitor is threading.current_thread():
            raise RuntimeError("fatal monitor cannot own communicator publication")
        elif monitor_live:
            self._wait_for_fatal_monitor_exit(outcome)
            self._join_fatal_monitor(monitor)
        if announce:
            self._fatal_store_set(self._fatal_key(self.rank), 1)
        self._abort_local_communicator()

    def _publish_communicator_fatal_locked(self, primary):
        with self._fatal_lock:
            if self._fatal_protocol_completed:
                return primary
        if not self._fatal_control_initialized:
            missing = RuntimeError("communicator fatal control is unavailable")
            self._record_fatal_secondary(missing)
            self._fatal_hard_exit()
        self._wait_for_local_communicator_abort()
        with self._fatal_lock:
            with self._fatal_publication_lock:
                if self._fatal_pending_primary is not primary:
                    raise RuntimeError("communicator fatal primary changed")
                if self._fatal_error is None:
                    self._fatal_error = primary
                    self._fatal_origin_rank = self._fatal_pending_origin_rank
        self._fatal_store_set(self._fatal_ack_key(self.rank), 1)
        self._wait_for_fatal_acknowledgments()
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
    ):
        with self._communicator_fatal_reservation(
            error,
            admitted=admitted,
            fatal_owner=fatal_owner,
            discovering_token=discovering_token,
            join_existing=join_existing,
            handler_override=handler_override,
            transition=transition,
        ) as reservation:
            if reservation.started:
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
            if not reservation.started:
                return reservation.primary
            self._run_fatal_handler(reservation)
            return reservation.primary

    def _read_fatal_origin(self):
        return next(
            (
                rank
                for rank in range(self.size)
                if int(self._bootstrap_store_proxy[self._fatal_key(rank)]) == 1
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
            primary, before_select=before_select
        )
        if outcome.kind != "fatal_elected":
            raise RuntimeError("clean monitor stop preempted communicator fatal")
        if outcome.primary is not primary:
            self._record_fatal_secondary(primary)
        return outcome

    def _elect_collective_fatal_outcome(self, candidate, origin_rank):
        with self._fatal_lock:
            primary = (
                candidate
                if self._fatal_pending_primary is None
                else self._fatal_pending_primary
            )
        if primary is not candidate:
            self._record_fatal_secondary(candidate)

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
            self._record_fatal_secondary(primary)
        with self._fatal_lock:
            self._fatal_monitor_outcome = outcome
            self._fatal_monitor_outcome_confirmed = True
        self._fatal_monitor_stop.set()
        return outcome

    def _confirm_runtime_fatal_outcome(self, primary):
        outcome = self._select_fatal_monitor_outcome("fatal_elected", primary)
        if outcome.kind != "fatal_elected" or outcome.primary is not primary:
            raise RuntimeError("clean monitor stop preempted communicator fatal")
        with self._fatal_lock:
            self._fatal_monitor_outcome = outcome
            self._fatal_monitor_outcome_confirmed = True
        self._fatal_monitor_stop.set()
        return outcome

    def _select_runtime_fatal_outcome(self, primary):
        with self._fatal_lock:
            confirmed = self._fatal_monitor_outcome_confirmed
            outcome = self._fatal_monitor_outcome
        if not confirmed:
            return self._confirm_runtime_fatal_outcome(primary)
        if outcome.kind != "fatal_elected" or outcome.primary is not primary:
            raise RuntimeError("clean monitor stop preempted communicator fatal")
        self._fatal_monitor_stop.set()
        return outcome

    def _wait_for_fatal_monitor_outcome(self, generation):
        try:
            return self._fatal_monitor_handoff.wait_for_outcome(
                generation, _FATAL_TIMEOUT_S
            )
        except TimeoutError:
            error = RuntimeError("communicator fatal monitor stop timed out")
            self._record_fatal_secondary(error)
            self._fatal_hard_exit()

    def _wait_for_fatal_monitor_selection(self):
        try:
            return self._fatal_monitor_handoff.wait_for_selection(
                _FATAL_TIMEOUT_S
            )
        except TimeoutError:
            error = RuntimeError(
                "communicator fatal monitor outcome selection timed out"
            )
            self._record_fatal_secondary(error)
            self._fatal_hard_exit()

    def _acknowledge_fatal_monitor_exit(self, outcome):
        return self._fatal_monitor_handoff.acknowledge_exit(outcome)

    def _wait_for_fatal_monitor_exit(self, outcome):
        try:
            return self._fatal_monitor_handoff.wait_for_exit(
                outcome, _FATAL_TIMEOUT_S
            )
        except TimeoutError:
            error = RuntimeError(
                "communicator fatal monitor exit acknowledgment timed out"
            )
            self._record_fatal_secondary(error)
            self._fatal_hard_exit()

    def _join_fatal_monitor(self, thread):
        if thread is None or thread is threading.current_thread():
            return
        thread.join(_FATAL_TIMEOUT_S)
        if thread.is_alive():
            error = RuntimeError("communicator fatal monitor stop timed out")
            self._record_fatal_secondary(error)
            self._fatal_hard_exit()

    def _start_fatal_monitor_publication(self, error, origin_rank):
        def publish():
            try:
                self._enter_observed_fatal(
                    error,
                    origin_rank,
                    release_admitted=True,
                    join_existing=True,
                )
            except BaseException as publication_error:
                if publication_error is not error:
                    self._record_fatal_secondary(publication_error)
                self._fatal_hard_exit()

        thread = threading.Thread(
            target=publish,
            name="renormalizer-fatal-publisher-rank-{}".format(self.rank),
            daemon=True,
        )
        with self._fatal_lock:
            self._fatal_monitor_publisher_thread = thread
        thread.start()
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
                    if self._fatal_monitor_stop.is_set():
                        outcome = self._observe_fatal_monitor_outcome()
                        return
                    self._record_fatal_secondary(error)
                    self._fatal_hard_exit()
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

    def _start_fatal_monitor(self):
        with self._fatal_lock:
            thread = self._fatal_monitor_thread
            if thread is not None and thread.is_alive():
                return
            self._fatal_monitor_stop.clear()
            thread = threading.Thread(
                target=self._monitor_fatal_records,
                name="renormalizer-fatal-monitor-rank-{}".format(self.rank),
                daemon=True,
            )
            self._fatal_monitor_thread = thread
            thread.start()

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

    def _local_fatal_capability_code(self):
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
            store_proxy[probe_key] = 0
            if int(store_proxy[probe_key]) != 0:
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

    def _bootstrap_fatal_control(self):
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
        local_code = self._local_fatal_capability_code()
        local_status_key = "{}{}".format(_STATUS_WORKSPACE_KEY_PREFIX, self.rank)
        store_proxy[local_status_key] = local_code
        store_proxy[self._fatal_key(self.rank)] = 0
        store_proxy[self._fatal_ack_key(self.rank)] = 0
        store_proxy[self._active_b_key(self.rank)] = self._encode_active_b(0, 0)
        store_proxy[self._close_ready_key(self.rank)] = 0
        store_proxy[self._monitor_stopped_key(self.rank)] = 0
        store_proxy[self._close_consumed_key(self.rank)] = 0
        store_proxy[_CLOSE_RELEASE_KEY] = 0
        store_proxy.barrier()
        aggregate = 0
        for rank in range(self.size):
            aggregate |= int(
                store_proxy["{}{}".format(_STATUS_WORKSPACE_KEY_PREFIX, rank)]
            )
            int(store_proxy[self._fatal_key(rank)])
            int(store_proxy[self._fatal_ack_key(rank)])
            sequence, failed = self._decode_active_b(
                store_proxy[self._active_b_key(rank)]
            )
            if (int(sequence), int(failed)) != (0, 0):
                aggregate |= _FATAL_CAPABILITY_ERROR
            if (
                int(store_proxy[self._close_ready_key(rank)]) != 0
                or int(store_proxy[self._monitor_stopped_key(rank)]) != 0
                or int(store_proxy[self._close_consumed_key(rank)]) != 0
            ):
                aggregate |= _FATAL_CAPABILITY_ERROR
        if int(store_proxy[_CLOSE_RELEASE_KEY]) != 0:
            aggregate |= _FATAL_CAPABILITY_ERROR
        store_proxy.barrier()
        if aggregate == 0:
            self._fatal_control_initialized = True
            self._start_fatal_monitor()
        return aggregate

    def _bootstrap_status_or(self, local_code):
        if type(local_code) is not int or local_code < 0:
            raise ValueError("bootstrap status code must be a non-negative integer")
        self._validate_bootstrap_status_or()
        store_proxy = self._bootstrap_store_proxy
        local_key = "{}{}".format(_STATUS_WORKSPACE_KEY_PREFIX, self.rank)
        store_proxy[local_key] = local_code
        store_proxy.barrier()
        aggregate = 0
        for rank in range(self.size):
            aggregate |= int(
                store_proxy["{}{}".format(_STATUS_WORKSPACE_KEY_PREFIX, rank)]
            )
        store_proxy.barrier()
        return aggregate

    def _broadcast_unadmitted(self, array, *, root):
        self._validate_root(root)
        self._validate_array(array)

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
        self._begin_collective_operation()
        try:
            yield self._broadcast_unadmitted, self._agree_admitted_active_broadcast
        finally:
            self._finish_collective_operation()

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
        self._fatal_store_set(
            self._active_b_key(self.rank), self._encode_active_b(sequence, failed)
        )
        deadline = time.monotonic() + _FATAL_TIMEOUT_S
        records = None
        while records is None:
            observed = []
            complete = True
            for rank in range(self.size):
                value = self._fatal_store_get(self._active_b_key(rank))
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
                origin_rank = failed_ranks[0]
                primary = _RemoteCommunicatorFailure(origin_rank)
                self._enter_observed_fatal(primary, origin_rank, admitted=admitted)
            else:
                self._publish_communicator_fatal(primary, admitted=admitted)
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

    def _wait_for_control_value(self, key, timeout_message):
        deadline = time.monotonic() + _FATAL_TIMEOUT_S
        while int(self._fatal_store_get(key)) != 1:
            if time.monotonic() >= deadline:
                error = RuntimeError(timeout_message)
                self._record_fatal_secondary(error)
                self._fatal_hard_exit()
            time.sleep(0.001)

    def _consume_terminal_control_records(self, discovering_token=None):
        fatal = []
        for rank in range(self.size):
            fatal.append(int(self._fatal_store_get(self._fatal_key(rank))))
            self._decode_active_b(self._fatal_store_get(self._active_b_key(rank)))
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

    def _wait_for_close_ready(self, discovering_token=None):
        deadline = time.monotonic() + _FATAL_TIMEOUT_S
        while True:
            primary = self._consume_terminal_control_records(discovering_token)
            if primary is not None:
                return primary
            if all(
                int(self._fatal_store_get(self._close_ready_key(rank))) == 1
                for rank in range(self.size)
            ):
                return None
            if time.monotonic() >= deadline:
                error = RuntimeError("communicator close intent timed out")
                self._record_fatal_secondary(error)
                self._fatal_hard_exit()
            time.sleep(0.001)

    def _quiesce_operations_for_close(self):
        deadline = time.monotonic() + _FATAL_TIMEOUT_S
        with self._fatal_condition:
            self._closing = True
            while self._admitted_operations or self._fatal_publications:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    error = RuntimeError(
                        "communicator fatal publication quiescence timed out"
                    )
                    self._record_fatal_secondary(error)
                    self._fatal_hard_exit()
                self._fatal_condition.wait(remaining)

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

    def _close_initialized_control(self, backend, gate=None, transition=None):
        ready_token = self._admit_runtime_close_step(
            gate, transition, "collective_close_ready"
        )
        try:
            self._fatal_store_set(self._close_ready_key(self.rank), 1)
            primary = self._wait_for_close_ready(ready_token)
            if primary is None:
                primary = self._consume_terminal_control_records(ready_token)
        finally:
            self._release_runtime_close_step(gate, ready_token)

        if primary is not None:
            self._wait_for_joined_fatal_publication()
            return self._fatal_monitor_outcome

        stop_token = self._admit_runtime_close_step(
            gate, transition, "collective_monitor_stop"
        )
        try:
            generation = self._request_fatal_monitor_stop()
        finally:
            self._release_runtime_close_step(gate, stop_token)

        outcome = self._wait_for_fatal_monitor_outcome(generation)
        self._wait_for_fatal_monitor_exit(outcome)
        thread = self._fatal_monitor_thread
        self._join_fatal_monitor(thread)
        if outcome.kind == "fatal_elected":
            self._wait_for_joined_fatal_publication()
            return outcome

        if gate is not None:
            selected = gate.select_runtime_close_commit(transition)
            if selected is not transition:
                self._wait_for_joined_fatal_publication()
                return self._fatal_monitor_outcome

        close_token = self._admit_runtime_close_step(
            gate, transition, "collective_close"
        )
        try:
            self._fatal_store_set(self._monitor_stopped_key(self.rank), 1)
            self._wait_for_all_control_records(
                self._monitor_stopped_key,
                "communicator monitor-stop agreement timed out",
            )

            if self.rank == 0:
                self._fatal_store_set(_CLOSE_RELEASE_KEY, 1)
            else:
                self._wait_for_control_value(
                    _CLOSE_RELEASE_KEY,
                    "communicator close release timed out",
                )
                backend.stop()
            self._fatal_store_set(self._close_consumed_key(self.rank), 1)
            if self.rank == 0:
                self._wait_for_all_control_records(
                    self._close_consumed_key,
                    "communicator close consumption timed out",
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

    def _wait_for_close_owner(self):
        deadline = time.monotonic() + _FATAL_TIMEOUT_S
        with self._fatal_condition:
            while self._close_in_progress and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    error = RuntimeError("communicator close join timed out")
                    self._record_fatal_secondary(error)
                    self._fatal_hard_exit()
                self._fatal_condition.wait(remaining)

    def _close_impl(self, gate=None, transition=None):
        while True:
            owns_close, closed = self._claim_close_owner()
            if closed:
                return None
            if owns_close:
                break
            self._wait_for_close_owner()
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
                    self._start_fatal_monitor()
                finally:
                    self._release_runtime_close_step(gate, restart_token)
            self._quiesce_operations_for_close()
            with self._fatal_lock:
                terminal = self._fatal_error is not None
            try:
                if terminal:
                    outcome = None
                elif self._fatal_control_initialized:
                    outcome = self._close_initialized_control(
                        backend, gate=gate, transition=transition
                    )
                else:
                    self._fatal_monitor_stop.set()
                    thread = self._fatal_monitor_thread
                    if thread is not None and thread is not threading.current_thread():
                        thread.join(_FATAL_TIMEOUT_S)
                    selected = (
                        transition
                        if gate is None
                        else gate.select_runtime_close_commit(transition)
                    )
                    if selected is transition:
                        backend.stop()
                        outcome = None
                    else:
                        self._wait_for_joined_fatal_publication()
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
        return self._close_impl(gate=gate, transition=transition)

    def close(self):
        self._close_impl()
