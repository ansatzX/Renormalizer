"""Single-owner lifecycle for resources touched by asynchronous CUDA work."""

from dataclasses import dataclass
import threading

import numpy as np

from renormalizer.backend._distributed.terminal import (
    _remaining_lifecycle_time,
    _require_managed_resource_admission,
)


def _require_resource_admission(token, validator):
    if token is None and validator is None:
        return None
    if token is None:
        raise TypeError("resource admission token is required")
    if not callable(validator):
        raise TypeError("resource admission validator is required")
    return validator(token)


def event_complete(event):
    query = getattr(event, "query", None)
    if callable(query):
        return bool(query())
    done = getattr(event, "done", None)
    if type(done) is bool:
        return done
    raise TypeError("completion event must provide query() or done")


def wait_event(event):
    synchronize = getattr(event, "synchronize", None)
    if not callable(synchronize):
        raise TypeError("completion event must provide synchronize()")
    synchronize()


def _unique(values):
    retained = []
    identities = set()
    for value in values:
        if value is None or id(value) in identities:
            continue
        identities.add(id(value))
        retained.append(value)
    return tuple(retained)


@dataclass(frozen=True)
class AsyncAllocationRecord:
    """Stable identity and full capacity for one retained backing allocation."""

    identity: tuple
    owner: object
    capacity_bytes: int


def _metadata_int(value):
    if callable(value):
        value = value()
    if type(value) is int:
        return value
    if value is None:
        return None
    return int(value)


def allocation_record(array=None, *, owner=None, pointer=None, capacity_bytes=None):
    if isinstance(array, AsyncAllocationRecord):
        return array
    metadata = getattr(array, "__renormalizer_allocation__", None)
    if metadata is not None:
        metadata = metadata() if callable(metadata) else metadata
        if isinstance(metadata, AsyncAllocationRecord):
            return metadata
        try:
            owner, pointer, capacity_bytes = metadata
        except (TypeError, ValueError) as error:
            raise TypeError(
                "explicit allocation metadata must contain owner, pointer, and capacity"
            ) from error
    if owner is not None or pointer is not None or capacity_bytes is not None:
        if owner is None or pointer is None or capacity_bytes is None:
            raise ValueError("explicit allocation metadata must be complete")
        pointer = _metadata_int(pointer)
        capacity_bytes = _metadata_int(capacity_bytes)
        if pointer < 0 or capacity_bytes < 0:
            raise ValueError("explicit allocation metadata must be non-negative")
        return AsyncAllocationRecord(("explicit", pointer), owner, capacity_bytes)

    data = getattr(array, "data", None)
    memory = getattr(data, "mem", None)
    if memory is not None:
        capacity = _metadata_int(getattr(memory, "size", None))
        root_pointer = _metadata_int(getattr(memory, "ptr", None))
        if capacity is None or root_pointer is None:
            raise TypeError("device allocation does not expose pointer and capacity")
        return AsyncAllocationRecord(("device", root_pointer), memory, capacity)

    owner = array
    identities = set()
    while id(owner) not in identities:
        identities.add(id(owner))
        base = getattr(owner, "base", None)
        if base is None and isinstance(owner, memoryview):
            base = owner.obj
        if base is None or base is owner:
            break
        owner = base
    owner_memory = getattr(owner, "mem", None)
    if owner_memory is not None:
        capacity = _metadata_int(getattr(owner_memory, "size", None))
        root_pointer = _metadata_int(getattr(owner_memory, "ptr", None))
        if capacity is not None and root_pointer is not None:
            return AsyncAllocationRecord(("host", root_pointer), owner_memory, capacity)

    try:
        root_view = memoryview(owner)
    except TypeError as error:
        raise TypeError("allocation owner does not expose trusted capacity") from error
    capacity = int(root_view.nbytes)
    interface = getattr(owner, "__array_interface__", None)
    if interface is not None:
        root_pointer = int(interface["data"][0])
    elif capacity == 0:
        root_pointer = id(owner)
    else:
        root_pointer = int(
            np.frombuffer(
                root_view, dtype=np.uint8, count=capacity
            ).__array_interface__["data"][0]
        )
    return AsyncAllocationRecord(("host", root_pointer), owner, capacity)


def allocation_records(arrays):
    records = {}
    for array in arrays:
        record = allocation_record(array)
        retained = records.get(record.identity)
        if retained is None or record.capacity_bytes > retained.capacity_bytes:
            records[record.identity] = record
    return tuple(records.values())


def merge_allocation_records(existing, arrays):
    records = {record.identity: record for record in existing}
    for record in allocation_records(arrays):
        retained = records.get(record.identity)
        if retained is None or record.capacity_bytes > retained.capacity_bytes:
            records[record.identity] = record
    return tuple(records.values())


def require_async_owner(value, context):
    owner = getattr(value, "owner", value)
    if not isinstance(owner, AsyncResourceOwner):
        raise TypeError("{} requires a pre-existing AsyncResourceOwner".format(context))
    return owner


class _CountedAsyncAdmission:
    """One pre-counted descendant claimed only by its completion worker."""

    def __init__(self, gate, parent, capability, operation=None):
        self._gate = gate
        self._parent = parent
        self._capability = capability
        self._operation = operation
        self._token = gate.spawn_async(parent, capability, operation)
        self._state_lock = threading.RLock()
        self._claimed = None
        self._released = False
        self._state = "installed"

    def _validate_claimed(self, claimed):
        parent = self._parent
        with self._gate._condition:
            state = self._gate._token_state(claimed)
            self._gate._require_token_thread(state)
            if (
                claimed.sequence != self._token.sequence
                or state.async_capability is not self._capability
                or claimed.scope != parent.scope
                or claimed.epoch != parent.epoch
                or claimed.parent_sequence != parent.sequence
                or claimed.transition_sequence != parent.transition_sequence
                or (
                    self._operation is not None
                    and claimed.operation != self._operation
                )
            ):
                raise RuntimeError(
                    "async admission does not match its canonical parent"
                )
        return claimed

    @staticmethod
    def _run_admitted_callback(
        callback,
        *,
        _admission_token,
        _admission_validator,
    ):
        _require_resource_admission(
            _admission_token,
            _admission_validator,
        )
        return callback(
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
        )

    @property
    def token(self):
        with self._state_lock:
            return self._token if self._claimed is None else self._claimed

    @property
    def close_owned(self):
        return self.token.scope == "lease_close"

    def consume_start_request(self):
        with self._state_lock:
            if self._state in {"cancelled", "released"}:
                return False
            sequence = self._token.sequence
            capability = self._capability
        return self._gate._consume_counted_async_start(
            sequence,
            capability,
        )

    @property
    def operation(self):
        return (
            "async_completion"
            if self._operation is None
            else self._operation
        )

    def run(self, operation, callback):
        # Admission state may enter the gate lock; it never enters an owner lock.
        with self._state_lock:
            if self._state != "installed":
                raise RuntimeError("async admission is no longer claimable")
            if self._operation is not None and operation != self._operation:
                raise RuntimeError("async operation does not match its family")
            claimed = self._gate.claim_async(
                self._token,
                self._capability,
                operation,
            )
            self._claimed = claimed
            self._state = "claimed"
        try:
            return self._run_admitted_callback(
                callback,
                _admission_token=claimed,
                _admission_validator=self._validate_claimed,
            )
        finally:
            try:
                self._gate.release(claimed)
            except RuntimeError as error:
                if "converted" not in str(error):
                    raise
            finally:
                with self._state_lock:
                    self._released = True
                    self._state = "released"

    def cancel(self):
        with self._state_lock:
            if self._state in {"cancelled", "released"}:
                return False
            if self._state == "claimed":
                return False
            cancelled = self._gate.cancel_async(
                self._token,
                self._capability,
            )
            if not cancelled:
                return False
            self._released = True
            self._state = "cancelled"
            return True

    def wake(self):
        with self._gate._condition:
            self._gate._condition.notify_all()

    def wait_for_execution(self, requested):
        with self._gate._condition:
            while not requested.is_set():
                phase = getattr(self._gate._phase, "value", None)
                if phase in {
                    "fatal_pending",
                    "fatal_published",
                    "runtime_closed",
                }:
                    return False
                self._gate._condition.wait()
            return True


class AsyncResourceOwner:
    """Own one async operation until completion, successful drain, or quarantine."""

    def __init__(
        self,
        kind,
        *,
        arrays=(),
        resources=(),
        streams=(),
        callback=None,
        accounting=None,
        nbytes=0,
        timer=None,
        started_at=None,
        elapsed_reader=None,
        drainer=None,
        detached=None,
        _detached_commit=None,
        quarantine=None,
        _async_admission=None,
        _resource_recorder=None,
        _resource_releaser=None,
        _defer_async_completion=False,
        _callback_requires_admission=False,
        _detached_requires_admission=False,
        _managed_guard=None,
        _managed_epoch=None,
    ):
        if not isinstance(kind, str) or not kind:
            raise ValueError("async owner kind must be a non-empty string")
        if type(nbytes) is not int or nbytes < 0:
            raise ValueError("async owner nbytes must be a non-negative integer")
        for value, name in (
            (callback, "callback"),
            (accounting, "accounting"),
            (timer, "timer"),
            (elapsed_reader, "elapsed_reader"),
            (drainer, "drainer"),
            (detached, "detached"),
            (_detached_commit, "detached commit"),
            (quarantine, "quarantine"),
            (_resource_recorder, "resource recorder"),
            (_resource_releaser, "resource releaser"),
        ):
            if value is not None and not callable(value):
                raise TypeError("{} must be callable".format(name))

        self.kind = kind
        self.direction = kind if kind in {"h2d", "d2h"} else None
        self.nbytes = nbytes
        self.elapsed_s = 0.0
        self.result = None
        self.error = None
        self.secondary_errors = ()
        self.state = "new"
        self.accounted = False

        self._arrays = _unique(arrays)
        self._allocations = allocation_records(self._arrays)
        self._resources = _unique(resources)
        self._streams = _unique(streams)
        self._events = []
        self._completion_event = None
        self._callback = callback
        self._callback_requires_admission = bool(
            _callback_requires_admission
        )
        self._detached_requires_admission = bool(
            _detached_requires_admission
        )
        self._accounting = accounting
        self._timer = timer
        self._started_at = (
            timer() if timer is not None and started_at is None else started_at
        )
        self._elapsed_reader = elapsed_reader
        self._drainer = drainer
        self._detached = detached
        self._detached_commit = _detached_commit
        self._quarantine = quarantine
        self._release_callbacks = []
        self._completion_armed = False
        self._async_admission = _async_admission
        self._resource_recorder = _resource_recorder
        self._resource_releaser = _resource_releaser
        self._managed_guard = _managed_guard
        self._managed_epoch = _managed_epoch
        # Terminal lock order is owner -> scheduler -> resource record. The
        # exact admission is established before terminal-transition election.
        self._async_lock = threading.RLock()
        self._async_worker = None
        self._async_worker_state = "none"
        self._async_done = threading.Event()
        self._async_requested = threading.Event()
        self._counted_quarantine_pending = False
        self._async_completion_deferred = bool(_defer_async_completion)
        self._async_start_pending = False
        self._async_admission_state = (
            "uninstalled" if _async_admission is None else "installed"
        )
        self._async_quarantine_requested = False
        self._detach_transition_capability = object()
        self._detach_transition_state = "available"
        self._completion_callback_done = callback is None
        self._detach_completion_done = False
        self._detach_scheduler_receipt = None
        if self._resource_recorder is not None:
            self._resource_recorder(
                resource=self,
                records=self._allocations,
                streams=self._streams,
            )

    @property
    def arrays(self):
        self._require_admission(
            allowed_operations=(
                "acquire",
                "child_close",
                "compute_completion",
                "d2h_completion",
                "h2d_completion",
                "load",
                "operator_call",
                "prefetch",
                "resource_state",
                "schedule_writeback",
            )
        )
        return tuple(self._arrays)

    @property
    def resources(self):
        self._require_admission(
            allowed_operations=(
                "acquire",
                "child_close",
                "compute_completion",
                "d2h_completion",
                "h2d_completion",
                "load",
                "operator_call",
                "prefetch",
                "resource_state",
                "schedule_writeback",
            )
        )
        return tuple(self._resources)

    @property
    def allocations(self):
        self._require_admission(
            allowed_operations=(
                "acquire",
                "child_close",
                "compute_completion",
                "d2h_completion",
                "h2d_completion",
                "load",
                "operator_call",
                "prefetch",
                "resource_state",
                "schedule_writeback",
            )
        )
        return tuple(self._allocations)

    @property
    def streams(self):
        self._require_admission(
            allowed_operations=(
                "acquire",
                "child_close",
                "compute_completion",
                "d2h_completion",
                "h2d_completion",
                "load",
                "operator_call",
                "prefetch",
                "resource_state",
                "schedule_writeback",
            )
        )
        return tuple(self._streams)

    @property
    def events(self):
        self._require_admission(
            allowed_operations=(
                "acquire",
                "cache_wait",
                "child_close",
                "compute_completion",
                "d2h_completion",
                "h2d_completion",
                "load",
                "operator_call",
                "pool_reap",
                "prefetch",
                "reap",
                "resource_state",
                "schedule_writeback",
                "scheduler_complete",
            )
        )
        return tuple(self._events)

    @property
    def completion_event(self):
        self._require_admission(
            allowed_operations=(
                "acquire",
                "cache_wait",
                "child_close",
                "compute_completion",
                "d2h_completion",
                "h2d_completion",
                "load",
                "operator_call",
                "pool_reap",
                "prefetch",
                "reap",
                "resource_state",
                "schedule_writeback",
                "scheduler_complete",
            )
        )
        return self._completion_event

    @property
    def completed(self):
        self._require_admission(
            allowed_operations=(
                "acquire",
                "cache_wait",
                "child_close",
                "close_progress",
                "compute_completion",
                "d2h_completion",
                "emit_profile",
                "h2d_completion",
                "load",
                "operator_call",
                "pool_reap",
                "prefetch",
                "reap",
                "resource_state",
                "schedule_writeback",
                "scheduler_complete",
            )
        )
        return self.state == "detached"

    @property
    def quarantined(self):
        self._require_admission(
            allowed_operations=(
                "acquire",
                "cache_wait",
                "child_close",
                "close_progress",
                "compute_completion",
                "d2h_completion",
                "h2d_completion",
                "load",
                "operator_call",
                "pool_reap",
                "prefetch",
                "reap",
                "resource_state",
                "schedule_writeback",
                "scheduler_complete",
            )
        )
        return self.state == "quarantined"

    def _require_admission(
        self,
        token=None,
        validator=None,
        *,
        allowed_operations=(),
    ):
        return _require_managed_resource_admission(
            self._managed_guard,
            token,
            validator,
            allowed_scopes=("lease", "lease_close"),
            allowed_operations=allowed_operations,
            epoch=lambda _token: self._managed_epoch,
        )

    def _terminal_resource_snapshot(self):
        """Query-free terminal ownership snapshot for the runtime owner."""
        return {
            "arrays": tuple(self._arrays),
            "resources": tuple(self._resources),
            "allocations": tuple(self._allocations),
            "streams": tuple(self._streams),
            "events": tuple(self._events),
            "completion_event": self._completion_event,
            "completed": self.state == "detached",
            "quarantined": self.state == "quarantined",
        }

    def capture_arrays(
        self,
        *arrays,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=("acquire", "child_close", "operator_call"),
        )
        if self.state not in {"new", "enqueued"}:
            raise RuntimeError("async owner can no longer capture arrays")
        self._arrays = _unique((*self._arrays, *arrays))
        self._allocations = merge_allocation_records(self._allocations, arrays)
        if self._resource_recorder is not None:
            self._resource_recorder(resource=self, records=self._allocations)

    def capture_allocations(
        self,
        *arrays,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=(
                "acquire",
                "close_progress",
                "load",
                "operator_call",
                "prefetch",
                "schedule_writeback",
            ),
        )
        if self.state not in {"new", "enqueued"}:
            raise RuntimeError("async owner can no longer capture allocations")
        self._allocations = merge_allocation_records(self._allocations, arrays)
        if self._resource_recorder is not None:
            self._resource_recorder(resource=self, records=self._allocations)

    def capture_resources(
        self,
        *resources,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=(
                "acquire",
                "child_close",
                "close_progress",
                "load",
                "operator_call",
                "prefetch",
                "schedule_writeback",
            ),
        )
        if self.state not in {"new", "enqueued"}:
            raise RuntimeError("async owner can no longer capture resources")
        self._resources = _unique((*self._resources, *resources))

    def capture_streams(
        self,
        *streams,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=(
                "acquire",
                "child_close",
                "close_progress",
                "load",
                "operator_call",
                "prefetch",
                "schedule_writeback",
            ),
        )
        if self.state not in {"new", "enqueued"}:
            raise RuntimeError("async owner can no longer capture streams")
        self._streams = _unique((*self._streams, *streams))
        if self._resource_recorder is not None:
            self._resource_recorder(resource=self, streams=streams)

    def add_event(
        self,
        event,
        *,
        completion=False,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=(
                "acquire",
                "child_close",
                "close_progress",
                "load",
                "operator_call",
                "prefetch",
                "schedule_writeback",
            ),
        )
        if self.state not in {"new", "enqueued"}:
            raise RuntimeError("async owner can no longer capture events")
        if event is None:
            raise TypeError("async owner event must not be None")
        if all(retained is not event for retained in self._events):
            self._events.append(event)
        if completion:
            if self._completion_event is not None:
                raise RuntimeError("async owner already has a completion event")
            self._completion_event = event
        if self._resource_recorder is not None:
            self._resource_recorder(resource=self, events=(event,))

    def add_release_callback(
        self,
        callback,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=(
                "acquire",
                "child_close",
                "close_progress",
                "load",
                "operator_call",
                "prefetch",
                "schedule_writeback",
            ),
        )
        if not callable(callback):
            raise TypeError("release callback must be callable")
        if self.state not in {"new", "enqueued"}:
            raise RuntimeError("async owner can no longer capture release callbacks")
        self._release_callbacks.append(callback)

    def arm_completion(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=(
                "acquire",
                "child_close",
                "close_progress",
                "load",
                "operator_call",
                "prefetch",
                "schedule_writeback",
            ),
        )
        if self.state not in {"new", "enqueued"}:
            raise RuntimeError("async owner can no longer arm completion")
        self._completion_armed = True
        self._watch_counted_completion()

    def _watch_counted_completion(self):
        worker = None
        with self._async_lock:
            if (
                self._async_admission is None
                or self._async_worker_state != "none"
                or not self._completion_armed
                or self._completion_event is None
                or self.state != "enqueued"
                or self._async_admission_state
                not in {"installed", "cancelling"}
            ):
                return
            admission = self._async_admission
            worker = threading.Thread(
                target=self._run_counted_completion,
                args=(admission,),
                name="renormalizer-async-completion",
                daemon=True,
            )
            self._async_worker = worker
            self._async_worker_state = "starting"
            self._async_admission_state = "worker"
            if admission.close_owned:
                self._async_start_pending = True
        try:
            worker.start()
        except BaseException as error:
            self._terminalize_counted_start_failure(worker, admission, error)
            raise error
        with self._async_lock:
            if self._async_worker_state == "starting":
                self._async_worker_state = "started"
        self._publish_counted_start_request()

    def _terminalize_counted_start_failure(self, worker, admission, primary):
        cancel = False
        wake = None
        with self._async_lock:
            if (
                self._async_worker is worker
                and self._async_worker_state == "starting"
            ):
                self._async_worker_state = "start_failed"
                self._async_admission_state = "cancelling"
                self._async_start_pending = False
                self._async_requested.set()
                cancel = True
            else:
                self._async_start_pending = False
                self._async_requested.set()
                wake = self._async_admission

        cancelled = False
        if cancel:
            try:
                cancelled = admission.cancel()
            except BaseException as error:
                self._remember_secondary(error)
            with self._async_lock:
                if cancelled:
                    self._async_admission_state = "cancelled"
                    self._async_admission = None
                    self._async_done.set()
                else:
                    wake = self._async_admission

        if wake is not None:
            try:
                wake.wake()
            except BaseException as error:
                self._remember_secondary(error)
        self._remember_error(primary)
        try:
            self._move_to_quarantine()
        except BaseException as error:
            self._remember_secondary(error)

    def _publish_counted_start_request(self):
        admission = None
        with self._async_lock:
            if (
                not self._async_start_pending
                or self._async_worker is None
                or self._async_worker_state != "started"
                or self._async_completion_deferred
                or self._async_requested.is_set()
                or self._async_admission is None
                or self._async_admission_state == "cancelled"
            ):
                return
            self._async_start_pending = False
            self._async_requested.set()
            admission = self._async_admission
        admission.wake()

    def _consume_counted_start_request(self, *, _close_transition=None):
        with self._async_lock:
            admission = self._async_admission
        if admission is None:
            if _close_transition is None:
                return False
            self._start_counted_completion(
                _close_transition=_close_transition,
            )
            return True
        if not admission.consume_start_request():
            return False
        with self._async_lock:
            if (
                self._async_admission is not admission
                or self._async_admission_state == "cancelled"
            ):
                return False
        self._start_counted_completion(_latched_admission=admission)
        return True

    def _start_counted_completion(
        self,
        *,
        _close_transition=None,
        _latched_admission=None,
    ):
        if self._managed_guard is not None:
            if _latched_admission is not None:
                with self._async_lock:
                    if self._async_admission is not _latched_admission:
                        return False
            elif _close_transition is None:
                self._require_admission(
                    allowed_operations=(
                        "child_close",
                        "close_progress",
                        "reap",
                        "scheduler_complete",
                    )
                )
            else:
                self._managed_guard.require_close_transition(
                    _close_transition,
                    epoch=lambda _transition: self._managed_epoch,
                )
        with self._async_lock:
            if not self._async_requested.is_set():
                self._async_start_pending = True
        self._watch_counted_completion()
        self._publish_counted_start_request()
        return True

    def publish_counted_completion(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=("acquire", "load", "operator_call", "prefetch"),
        )
        with self._async_lock:
            self._async_completion_deferred = False
        self._watch_counted_completion()
        self._publish_counted_start_request()

    def install_counted_admission(
        self,
        admission,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=("acquire", "child_close", "operator_call"),
        )
        with self._async_lock:
            if self._async_admission is not None or self._async_worker is not None:
                raise RuntimeError("async owner already has a counted admission")
            if self.state not in {"new", "enqueued"}:
                raise RuntimeError("async owner can no longer install an admission")
            self._async_admission = admission
            self._async_admission_state = "installed"

    def watch_counted_completion(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=("acquire", "child_close", "operator_call"),
        )
        self._watch_counted_completion()

    def _run_counted_completion(self, admission):
        with self._async_lock:
            if self._async_worker_state == "start_failed":
                self._async_done.set()
                return
            if self._async_worker_state == "starting":
                self._async_worker_state = "started"
        execution_requested = admission.wait_for_execution(self._async_requested)

        def complete(*, _admission_token, _admission_validator):
            with self._async_lock:
                terminal_requested = self._async_quarantine_requested or (
                    self.state in {"detached", "quarantined"}
                )
            if terminal_requested:
                self._counted_quarantine_pending = True
                return
            if not execution_requested:
                self._counted_quarantine_pending = True
                return
            try:
                wait_event(self._completion_event)
            except BaseException as error:
                self._resolve_counted_wait_failure(
                    error,
                    _admission_token=_admission_token,
                    _admission_validator=_admission_validator,
                )
                return
            if self.state in {"detached", "quarantined"}:
                return
            try:
                self._detach(
                    "completed",
                    _admission_token=_admission_token,
                    _admission_validator=_admission_validator,
                    _transition_capability=self._detach_transition_capability,
                )
            except BaseException:
                # _detach records callback/accounting failures before raising them.
                return

        try:
            admission.run(admission.operation, complete)
        except BaseException as error:
            self._remember_error(error)
            if self.state not in {"detached", "quarantined"}:
                self._counted_quarantine_pending = True
        finally:
            with self._async_lock:
                if self._async_admission_state != "cancelled":
                    self._async_admission_state = "terminal"
                if self._async_worker_state != "start_failed":
                    self._async_worker_state = "terminal"
            self._async_done.set()

    def _resolve_counted_wait_failure(
        self,
        error,
        *,
        _admission_token,
        _admission_validator,
    ):
        if self.error is None:
            self._remember_error(error)
        else:
            self._remember_secondary(error)
        try:
            self._drain()
        except BaseException as drain_error:
            self._remember_secondary(drain_error)
            self._counted_quarantine_pending = True
            return
        if self.state not in {"detached", "quarantined"}:
            try:
                self._detach(
                    "drained",
                    _admission_token=_admission_token,
                    _admission_validator=_admission_validator,
                    _transition_capability=self._detach_transition_capability,
                )
            except BaseException:
                pass

    def _cancel_unclaimed_async(self):
        with self._async_lock:
            admission = self._async_admission
            worker = self._async_worker
            if admission is None:
                return True
            worker_state = self._async_worker_state
            if worker_state in {"starting", "started", "terminal"}:
                self._async_start_pending = False
                self._async_requested.set()
            elif worker_state == "start_failed":
                return self._async_admission_state == "cancelled"
            else:
                self._async_admission_state = "cancelling"
        if worker_state in {"starting", "started", "terminal"}:
            admission.wake()
            return False

        cancelled = admission.cancel()
        wake = None
        with self._async_lock:
            if cancelled:
                self._async_admission_state = "cancelled"
                self._async_admission = None
                self._async_start_pending = False
                self._async_requested.set()
                wake = admission
                if self._async_worker is None:
                    self._async_done.set()
            else:
                if self._async_worker_state != "none":
                    self._async_admission_state = "worker"
                self._async_start_pending = False
                self._async_requested.set()
                wake = self._async_admission
        if wake is not None:
            wake.wake()
        return cancelled

    def start_counted_completion(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=(
                "child_close",
                "close_progress",
                "reap",
                "scheduler_complete",
            ),
        )
        self._start_counted_completion()

    def _prepare_counted_completion(
        self,
        *,
        wait,
        join_started=True,
        _deadline=None,
    ):
        self._require_admission(
            allowed_operations=(
                "acquire",
                "cache_wait",
                "child_close",
                "close_progress",
                "load",
                "operator_call",
                "pool_close",
                "pool_reap",
                "prefetch",
                "reap",
                "resource_state",
                "schedule_writeback",
                "scheduler_close",
                "scheduler_complete",
            )
        )
        if self._async_admission is None:
            return None
        if (
            not wait
            and not self._async_requested.is_set()
            and not self._async_done.is_set()
        ):
            try:
                if not event_complete(self._completion_event):
                    return False
            except BaseException as error:
                self._remember_error(error)
        if self._async_worker is None:
            if (
                self._async_completion_deferred
                or not self._completion_armed
                or self._completion_event is None
                or self.state != "enqueued"
            ):
                return False
            self._start_counted_completion()
        else:
            self._async_requested.set()
            self._async_admission.wake()
        return self._counted_completion_result(
            wait=wait or join_started,
            _deadline=_deadline,
        )

    def _counted_completion_result(self, *, wait, _deadline=None):
        self._require_admission(
            allowed_operations=(
                "acquire",
                "cache_wait",
                "child_close",
                "close_progress",
                "load",
                "operator_call",
                "pool_close",
                "pool_reap",
                "prefetch",
                "reap",
                "resource_state",
                "schedule_writeback",
                "scheduler_close",
                "scheduler_complete",
            )
        )
        worker = self._async_worker
        if worker is None:
            return None
        if threading.current_thread() is worker:
            return False
        if wait:
            remaining = _remaining_lifecycle_time(
                _deadline,
                "async completion lifecycle timed out before worker wait",
            )
            if not self._async_done.wait(remaining):
                raise TimeoutError("async completion lifecycle timed out")
        elif not self._async_done.is_set():
            return False
        if self._counted_quarantine_pending and self.state not in {
            "detached",
            "quarantined",
        }:
            self._counted_quarantine_pending = False
            self._move_to_quarantine()
        if self.state == "quarantined":
            raise self.error
        if self.state == "detached":
            if self.error is not None:
                raise self.error
            return True
        if self.error is not None and self._async_done.is_set():
            raise self.error
        return False

    def mark_enqueued(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=(
                "acquire",
                "child_close",
                "close_progress",
                "load",
                "operator_call",
                "prefetch",
                "schedule_writeback",
            ),
        )
        if self.state != "new":
            raise RuntimeError("async owner is not new")
        self.state = "enqueued"

    def _remember_error(self, error):
        if not isinstance(error, BaseException):
            raise TypeError("async owner failure must be an exception")
        with self._async_lock:
            if self.error is None:
                self.error = error

    def _remember_secondary(self, error):
        if not isinstance(error, BaseException):
            raise TypeError("async owner secondary failure must be an exception")
        with self._async_lock:
            if error is self.error or any(
                retained is error for retained in self.secondary_errors
            ):
                return
            self.secondary_errors = (*self.secondary_errors, error)

    def _drain(self):
        if self._drainer is not None:
            self._drainer()
            return
        if self._streams:
            first_error = None
            for stream in self._streams:
                try:
                    synchronize = getattr(stream, "synchronize", None)
                    if not callable(synchronize):
                        raise TypeError("owned stream must provide synchronize()")
                    synchronize()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
            if first_error is not None:
                raise first_error
            return
        if self._completion_event is not None:
            wait_event(self._completion_event)

    def _run_completion(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        first_error = self.error
        try:
            if self._elapsed_reader is not None:
                self.elapsed_s = self._elapsed_reader()
            elif self._timer is not None:
                self.elapsed_s = self._timer() - self._started_at
        except BaseException as error:
            if first_error is None:
                first_error = error
                self.error = error
            else:
                self._remember_secondary(error)
        if not self._completion_callback_done:
            try:
                if self._callback_requires_admission:
                    self.result = self._callback(
                        _admission_token=_admission_token,
                        _admission_validator=_admission_validator,
                    )
                else:
                    self.result = self._callback()
            except BaseException as error:
                if first_error is None:
                    first_error = error
                    self.error = error
                else:
                    self._remember_secondary(error)
                raise first_error
            self._completion_callback_done = True
        try:
            if self._accounting is not None and not self.accounted:
                self._accounting(self)
                self.accounted = True
        except BaseException as error:
            if first_error is None:
                first_error = error
                self.error = error
            else:
                self._remember_secondary(error)
        if first_error is not None:
            raise first_error

    def _detach(
        self,
        terminal_state,
        *,
        _admission_token=None,
        _admission_validator=None,
        _transition_capability=None,
    ):
        admission = self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=(
                "acquire",
                "cache_wait",
                "child_close",
                "close_progress",
                "compute_completion",
                "d2h_completion",
                "h2d_completion",
                "load",
                "operator_call",
                "pool_close",
                "pool_reap",
                "prefetch",
                "reap",
                "resource_state",
                "schedule_writeback",
                "scheduler_close",
                "scheduler_complete",
            ),
        )
        if terminal_state not in {"completed", "drained"}:
            raise ValueError("async owner terminal state is invalid")
        with self._async_lock:
            if (
                self._detach_transition_capability is None
                or _transition_capability
                is not self._detach_transition_capability
            ):
                raise RuntimeError(
                    "async owner transition capability does not own detach"
                )
            if self._detach_transition_state != "available":
                raise RuntimeError("async owner detach transition is not available")
            if self.state not in {"new", "enqueued"}:
                raise RuntimeError("async owner state cannot enter detach")
            if terminal_state == "completed" and self.state != "enqueued":
                raise RuntimeError("only an enqueued async owner can complete")
            prior_state = self.state
            self._detach_transition_state = "running"
            callback_token = _admission_token
            callback_validator = _admission_validator
            if admission is not None and callback_token is None:
                callback_token = admission

                def validate_callback_admission(candidate, expected=admission):
                    if candidate is not expected:
                        raise RuntimeError(
                            "async detach admission changed before callback"
                        )
                    return candidate

                callback_validator = validate_callback_admission

            self.state = terminal_state
            first_error = self.error
            if self._completion_armed and not self._detach_completion_done:
                try:
                    self._run_completion(
                        _admission_token=_admission_token,
                        _admission_validator=_admission_validator,
                    )
                except BaseException as error:
                    if first_error is None:
                        first_error = error
                        self.error = error
                    else:
                        self._remember_secondary(error)
                    if not self._completion_callback_done:
                        self.state = prior_state
                        self._detach_transition_state = "available"
                        raise first_error
                if self._completion_callback_done:
                    self._detach_completion_done = True

            if self._async_quarantine_requested:
                self.state = prior_state
                self._detach_transition_state = "available"
                self._move_to_quarantine()
                if self.error is not None:
                    raise self.error
                return False

            try:
                if self._detached is not None:
                    if self._detached_requires_admission:
                        receipt = self._detached(
                            self,
                            _admission_token=callback_token,
                            _admission_validator=callback_validator,
                        )
                    else:
                        receipt = self._detached(self)
                    self._detach_scheduler_receipt = receipt
                if self._detached_commit is not None:
                    receipt = self._detached_commit(
                        self,
                        self._detach_scheduler_receipt,
                    )
                    if receipt is not None:
                        self._detach_scheduler_receipt = receipt
            except BaseException as error:
                if first_error is None:
                    first_error = error
                    self.error = error
                else:
                    self._remember_secondary(error)
                self.state = prior_state
                self._detach_transition_state = "available"
                raise first_error

            for callback in tuple(self._release_callbacks):
                try:
                    callback()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
                        self.error = error
                    else:
                        self._remember_secondary(error)

            if self._resource_releaser is not None:
                try:
                    self._resource_releaser(self)
                except BaseException as error:
                    if first_error is None:
                        first_error = error
                        self.error = error
                    else:
                        self._remember_secondary(error)
            self.state = "detached"
            self._arrays = ()
            self._allocations = ()
            self._resources = ()
            self._streams = ()
            self._events = []
            self._completion_event = None
            self._callback = None
            self._accounting = None
            self._timer = None
            self._started_at = None
            self._elapsed_reader = None
            self._drainer = None
            self._detached = None
            self._detached_commit = None
            self._detached_requires_admission = False
            self._quarantine = None
            self._resource_recorder = None
            self._resource_releaser = None
            self._release_callbacks = []
            self._completion_armed = False
            self._detach_scheduler_receipt = None
            self._detach_transition_capability = None
            self._detach_transition_state = "complete"
            if first_error is not None:
                raise first_error
            return True

    def _move_to_quarantine(self):
        with self._async_lock:
            if self.state in {"detached", "quarantined"}:
                return False
            if self._detach_transition_state == "running":
                self._async_quarantine_requested = True
                return False
            if self._detach_transition_state == "quarantining":
                return False
            if self._detach_transition_state != "available":
                raise RuntimeError(
                    "async owner terminal transition cannot enter quarantine"
                )
            self._detach_transition_state = "quarantining"
            self.state = "quarantined"
            quarantine = self._quarantine
            if quarantine is not None:
                quarantine(self)
            self._detach_transition_capability = None
            self._detach_transition_state = "complete"
            self._detached = None
            self._detached_commit = None
            self._quarantine = None
            return True

    def force_quarantine(
        self,
        error,
        *,
        secondary_errors=(),
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=(
                "acquire",
                "cache_wait",
                "child_close",
                "close_progress",
                "load",
                "operator_call",
                "pool_close",
                "pool_reap",
                "prefetch",
                "reap",
                "schedule_writeback",
                "scheduler_close",
                "scheduler_complete",
            ),
        )
        return self._force_quarantine_terminal(
            error,
            secondary_errors=secondary_errors,
        )

    def _force_quarantine_terminal(self, error, *, secondary_errors=()):
        with self._async_lock:
            if self.state == "detached":
                return False
            self._remember_error(error)
            for secondary_error in secondary_errors:
                self._remember_secondary(secondary_error)
            if self.state == "quarantined":
                return False
            self._async_quarantine_requested = True
            if self._detach_transition_state == "running":
                return False
        self._cancel_unclaimed_async()
        return self._move_to_quarantine()

    def fail(
        self,
        error,
        *,
        secondary_errors=(),
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=(
                "acquire",
                "cache_wait",
                "child_close",
                "close_progress",
                "load",
                "operator_call",
                "pool_close",
                "pool_reap",
                "prefetch",
                "reap",
                "schedule_writeback",
                "scheduler_close",
                "scheduler_complete",
            ),
        )
        self._remember_error(error)
        for secondary_error in secondary_errors:
            self._remember_secondary(secondary_error)
        if self.state == "quarantined":
            raise self.error
        if self.state == "detached":
            raise self.error
        if self._async_worker is not None:
            self.force_quarantine(self.error, secondary_errors=secondary_errors)
            raise self.error
        self._cancel_unclaimed_async()
        if self.state == "new":
            try:
                self._detach(
                    "drained",
                    _admission_token=_admission_token,
                    _admission_validator=_admission_validator,
                    _transition_capability=self._detach_transition_capability,
                )
            except BaseException:
                pass
            raise self.error
        try:
            self._drain()
        except BaseException as drain_error:
            self._remember_secondary(drain_error)
            self._move_to_quarantine()
            raise self.error
        try:
            self._detach(
                "drained",
                _admission_token=_admission_token,
                _admission_validator=_admission_validator,
                _transition_capability=self._detach_transition_capability,
            )
        except BaseException:
            pass
        raise self.error

    def reap(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
        _wait_for_counted=True,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=(
                "acquire",
                "cache_wait",
                "child_close",
                "close_progress",
                "load",
                "operator_call",
                "pool_close",
                "pool_reap",
                "prefetch",
                "reap",
                "resource_state",
                "schedule_writeback",
                "scheduler_close",
                "scheduler_complete",
            ),
        )
        if type(_wait_for_counted) is not bool:
            raise TypeError("counted completion wait mode must be a boolean")
        counted = self._prepare_counted_completion(
            wait=False,
            join_started=_wait_for_counted,
        )
        if counted is not None:
            return counted
        if self.state == "quarantined":
            raise self.error
        if self.state == "detached":
            if self.error is not None:
                raise self.error
            return True
        if self._completion_event is None:
            return False
        try:
            complete = event_complete(self._completion_event)
        except BaseException as error:
            return self.fail(
                error,
                _admission_token=_admission_token,
                _admission_validator=_admission_validator,
            )
        if not complete:
            return False
        return self._detach(
            "completed",
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
            _transition_capability=self._detach_transition_capability,
        )

    def wait(
        self,
        *,
        _deadline=None,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=(
                "acquire",
                "cache_wait",
                "child_close",
                "close_progress",
                "load",
                "operator_call",
                "pool_close",
                "pool_reap",
                "prefetch",
                "reap",
                "schedule_writeback",
                "scheduler_close",
                "scheduler_complete",
            ),
        )
        counted = self._prepare_counted_completion(
            wait=True,
            _deadline=_deadline,
        )
        if counted is not None:
            return counted
        if self.state == "quarantined":
            raise self.error
        if self.state == "detached":
            if self.error is not None:
                raise self.error
            return True
        if self._completion_event is None:
            return False
        try:
            _remaining_lifecycle_time(
                _deadline,
                "async completion lifecycle timed out before event wait",
            )
            wait_event(self._completion_event)
            _remaining_lifecycle_time(
                _deadline,
                "async completion lifecycle timed out during event wait",
            )
        except BaseException as error:
            return self.fail(
                error,
                _admission_token=_admission_token,
                _admission_validator=_admission_validator,
            )
        return self._detach(
            "completed",
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
            _transition_capability=self._detach_transition_capability,
        )

    def drain(
        self,
        *,
        _deadline=None,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=("scheduler_close", "scheduler_complete"),
        )
        counted = self._prepare_counted_completion(
            wait=True,
            _deadline=_deadline,
        )
        if counted is not None:
            return counted
        if self.state == "quarantined":
            raise self.error
        if self.state == "detached":
            if self.error is not None:
                raise self.error
            return True
        try:
            _remaining_lifecycle_time(
                _deadline,
                "async completion lifecycle timed out before drain",
            )
            self._drain()
            _remaining_lifecycle_time(
                _deadline,
                "async completion lifecycle timed out during drain",
            )
        except BaseException as error:
            self._remember_error(error)
            self._move_to_quarantine()
            raise self.error
        return self._detach(
            "drained",
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
            _transition_capability=self._detach_transition_capability,
        )

    def take_result(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=("scheduler_complete",),
        )
        if self.state != "detached":
            raise RuntimeError("async owner is not complete")
        result = self.result
        self.result = None
        return result


class RuntimeTerminalQuarantine:
    """Terminal registry that intentionally retains unsafe async owners."""

    def __init__(self):
        self._owners = []
        self._allocations = {}
        self._cache_allocations = {}
        self._pinned_allocations = {}
        self._resources = []
        self._events = {}
        self._streams = {}
        self.first_error = None

    @property
    def poisoned(self):
        return self.first_error is not None

    @property
    def owners(self):
        return tuple(self._owners)

    @property
    def allocations(self):
        return tuple(self._allocations.values())

    def retain(self, owner, error=None):
        if not isinstance(owner, AsyncResourceOwner):
            raise TypeError("terminal quarantine requires an AsyncResourceOwner")
        retained_error = owner.error if error is None else error
        if retained_error is not None and self.first_error is None:
            self.first_error = retained_error
        if all(retained is not owner for retained in self._owners):
            self._owners.append(owner)
        snapshot = owner._terminal_resource_snapshot()
        self.retain_snapshot(
            resources=(owner,),
            allocations=snapshot["allocations"],
            events=snapshot["events"],
            streams=snapshot["streams"],
        )

    def retain_error(self, error):
        if not isinstance(error, BaseException):
            raise TypeError("terminal quarantine failure must be an exception")
        if self.first_error is None:
            self.first_error = error

    def retain_allocations(self, records):
        for record in records:
            if not isinstance(record, AsyncAllocationRecord):
                raise TypeError(
                    "terminal allocation retention requires AsyncAllocationRecord"
                )
            retained = self._allocations.get(record.identity)
            if retained is None or record.capacity_bytes > retained.capacity_bytes:
                self._allocations[record.identity] = record

    @staticmethod
    def _retain_typed_allocations(target, records):
        for record in records:
            if not isinstance(record, AsyncAllocationRecord):
                raise TypeError(
                    "terminal allocation retention requires AsyncAllocationRecord"
                )
            retained = target.get(record.identity)
            if retained is None or record.capacity_bytes > retained.capacity_bytes:
                target[record.identity] = record

    def retain_snapshot(
        self,
        *,
        resources=(),
        allocations=(),
        cache_allocations=(),
        pinned_allocations=(),
        events=(),
        streams=(),
    ):
        for resource in resources:
            if resource is not None and all(
                retained is not resource for retained in self._resources
            ):
                self._resources.append(resource)
        self.retain_allocations(allocations)
        self._retain_typed_allocations(
            self._cache_allocations,
            cache_allocations,
        )
        self._retain_typed_allocations(
            self._pinned_allocations,
            pinned_allocations,
        )
        for event in events:
            if event is not None:
                self._events[id(event)] = event
        for stream in streams:
            if stream is not None:
                self._streams[id(stream)] = stream

    def retain_resources(self, resources):
        for resource in resources:
            if resource is None:
                continue
            terminal_records = getattr(
                resource,
                "_terminal_allocation_records",
                None,
            )
            records = (
                tuple(terminal_records())
                if callable(terminal_records)
                else tuple(getattr(resource, "allocation_records", ()))
            )
            kind = getattr(resource, "_terminal_resource_kind", None)
            self.retain_snapshot(
                resources=(resource,),
                allocations=records,
                cache_allocations=records if kind == "cache" else (),
                pinned_allocations=records if kind == "pinned" else (),
                events=tuple(getattr(resource, "retained_events", ())),
                streams=tuple(getattr(resource, "retained_streams", ())),
            )

    def _retained_resource_allocations(self, kind=None):
        if kind == "cache":
            return dict(self._cache_allocations)
        if kind == "pinned":
            return dict(self._pinned_allocations)
        return dict(self._allocations)

    def _retained_events(self):
        return tuple(self._events.values())

    def _retained_streams(self):
        return tuple(self._streams.values())

    def retained_resource_state(self):
        cache_allocations = self._retained_resource_allocations("cache")
        pinned_allocations = self._retained_resource_allocations("pinned")
        return {
            "cache_bytes": sum(
                record.capacity_bytes for record in cache_allocations.values()
            ),
            "pinned_bytes": sum(
                record.capacity_bytes for record in pinned_allocations.values()
            ),
            "event_count": len(self._retained_events()),
            "stream_count": len(self._retained_streams()),
        }

    def resource_state(self):
        allocations = {record.identity: record for record in self.allocations}
        events = self._retained_events()
        streams = self._retained_streams()
        return {
            "quarantined_owner_count": len(self._owners),
            "quarantined_array_count": len(allocations),
            "quarantined_bytes": sum(
                allocation.capacity_bytes for allocation in allocations.values()
            ),
            "quarantined_event_count": len(events),
            "quarantined_stream_count": len(streams),
        }


__all__ = [
    "AsyncAllocationRecord",
    "AsyncResourceOwner",
    "RuntimeTerminalQuarantine",
    "allocation_record",
    "allocation_records",
    "merge_allocation_records",
    "require_async_owner",
]
