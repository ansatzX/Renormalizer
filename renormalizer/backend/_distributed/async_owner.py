"""Single-owner lifecycle for resources touched by asynchronous CUDA work."""

from dataclasses import dataclass
import threading

import numpy as np


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

    def __init__(self, gate, parent, capability):
        self._gate = gate
        self._parent = parent
        self._capability = capability
        self._token = gate.spawn_async(parent, capability)
        self._claimed = None
        self._released = False

    def _validate_claimed(self, claimed):
        parent = self._parent
        with self._gate._condition:
            state = self._gate._token_state(claimed)
            self._gate._require_token_thread(state)
            if (
                claimed.scope != parent.scope
                or claimed.epoch != parent.epoch
                or claimed.parent_sequence != parent.sequence
                or claimed.transition_sequence != parent.transition_sequence
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
        return self._token if self._claimed is None else self._claimed

    @property
    def close_owned(self):
        return self.token.scope == "lease_close"

    def run(self, operation, callback):
        if self._released or self._claimed is not None:
            raise RuntimeError("async admission is no longer claimable")
        claimed = self._gate.claim_async(
            self._token,
            self._capability,
            operation,
        )
        self._claimed = claimed
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
            self._released = True

    def cancel(self):
        if self._released:
            return
        if self._claimed is not None:
            raise RuntimeError("claimed async admission cannot be cancelled")
        self._gate.release(self._token)
        self._released = True

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
        quarantine=None,
        _async_admission=None,
        _resource_recorder=None,
        _resource_releaser=None,
        _defer_async_completion=False,
        _callback_requires_admission=False,
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
        self._accounting = accounting
        self._timer = timer
        self._started_at = (
            timer() if timer is not None and started_at is None else started_at
        )
        self._elapsed_reader = elapsed_reader
        self._drainer = drainer
        self._detached = detached
        self._quarantine = quarantine
        self._release_callbacks = []
        self._completion_armed = False
        self._async_admission = _async_admission
        self._resource_recorder = _resource_recorder
        self._resource_releaser = _resource_releaser
        self._async_worker = None
        self._async_done = threading.Event()
        self._async_requested = threading.Event()
        self._counted_quarantine_pending = False
        self._async_completion_deferred = bool(_defer_async_completion)
        self._async_start_pending = False
        if self._resource_recorder is not None:
            self._resource_recorder(
                resource=self,
                records=self._allocations,
                streams=self._streams,
            )

    @property
    def arrays(self):
        return tuple(self._arrays)

    @property
    def resources(self):
        return tuple(self._resources)

    @property
    def allocations(self):
        return tuple(self._allocations)

    @property
    def streams(self):
        return tuple(self._streams)

    @property
    def events(self):
        return tuple(self._events)

    @property
    def completion_event(self):
        return self._completion_event

    @property
    def completed(self):
        return self.state == "detached"

    @property
    def quarantined(self):
        return self.state == "quarantined"

    def capture_arrays(self, *arrays):
        if self.state not in {"new", "enqueued"}:
            raise RuntimeError("async owner can no longer capture arrays")
        self._arrays = _unique((*self._arrays, *arrays))
        self._allocations = merge_allocation_records(self._allocations, arrays)
        if self._resource_recorder is not None:
            self._resource_recorder(resource=self, records=self._allocations)

    def capture_allocations(self, *arrays):
        if self.state not in {"new", "enqueued"}:
            raise RuntimeError("async owner can no longer capture allocations")
        self._allocations = merge_allocation_records(self._allocations, arrays)
        if self._resource_recorder is not None:
            self._resource_recorder(resource=self, records=self._allocations)

    def capture_resources(self, *resources):
        if self.state not in {"new", "enqueued"}:
            raise RuntimeError("async owner can no longer capture resources")
        self._resources = _unique((*self._resources, *resources))

    def capture_streams(self, *streams):
        if self.state not in {"new", "enqueued"}:
            raise RuntimeError("async owner can no longer capture streams")
        self._streams = _unique((*self._streams, *streams))
        if self._resource_recorder is not None:
            self._resource_recorder(resource=self, streams=streams)

    def add_event(self, event, *, completion=False):
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

    def add_release_callback(self, callback):
        if not callable(callback):
            raise TypeError("release callback must be callable")
        if self.state not in {"new", "enqueued"}:
            raise RuntimeError("async owner can no longer capture release callbacks")
        self._release_callbacks.append(callback)

    def arm_completion(self):
        if self.state not in {"new", "enqueued"}:
            raise RuntimeError("async owner can no longer arm completion")
        self._completion_armed = True
        self._watch_counted_completion()

    def _watch_counted_completion(self):
        if (
            self._async_admission is None
            or self._async_worker is not None
            or not self._completion_armed
            or self._completion_event is None
            or self.state != "enqueued"
        ):
            return
        worker = threading.Thread(
            target=self._run_counted_completion,
            name="renormalizer-async-completion",
            daemon=True,
        )
        self._async_worker = worker
        worker.start()
        if self._async_admission.close_owned:
            self._start_counted_completion()

    def _start_counted_completion(self):
        self._watch_counted_completion()
        if self._async_worker is not None:
            if self._async_completion_deferred:
                self._async_start_pending = True
                return
            self._async_requested.set()
            self._async_admission.wake()

    def publish_counted_completion(self):
        self._async_completion_deferred = False
        self._watch_counted_completion()
        if self._async_start_pending and self._async_worker is not None:
            self._async_start_pending = False
            self._async_requested.set()
            self._async_admission.wake()

    def install_counted_admission(self, admission):
        if self._async_admission is not None or self._async_worker is not None:
            raise RuntimeError("async owner already has a counted admission")
        if self.state not in {"new", "enqueued"}:
            raise RuntimeError("async owner can no longer install an admission")
        self._async_admission = admission

    def watch_counted_completion(self):
        self._watch_counted_completion()

    def _run_counted_completion(self):
        admission = self._async_admission
        execution_requested = admission.wait_for_execution(self._async_requested)

        def complete(*, _admission_token, _admission_validator):
            if self.state in {"detached", "quarantined"}:
                return
            if not execution_requested:
                self._counted_quarantine_pending = True
                return
            try:
                wait_event(self._completion_event)
            except BaseException as error:
                self._resolve_counted_wait_failure(error)
                return
            if self.state in {"detached", "quarantined"}:
                return
            try:
                self._detach(
                    "completed",
                    _admission_token=_admission_token,
                    _admission_validator=_admission_validator,
                )
            except BaseException:
                # _detach records callback/accounting failures before raising them.
                return

        try:
            admission.run("async_completion", complete)
        except BaseException as error:
            self._remember_error(error)
            if self.state not in {"detached", "quarantined"}:
                self._counted_quarantine_pending = True
        finally:
            self._async_done.set()

    def _resolve_counted_wait_failure(self, error):
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
                self._detach("drained")
            except BaseException:
                pass

    def _cancel_unclaimed_async(self):
        admission = self._async_admission
        if admission is None:
            return
        if self._async_worker is not None:
            self._async_requested.set()
            admission.wake()
            return
        admission.cancel()
        self._async_admission = None
        self._async_done.set()

    def start_counted_completion(self):
        self._start_counted_completion()

    def _prepare_counted_completion(self, *, wait):
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
        return self._counted_completion_result(wait=True)

    def _counted_completion_result(self, *, wait):
        worker = self._async_worker
        if worker is None:
            return None
        if threading.current_thread() is worker:
            return False
        if wait:
            self._async_done.wait()
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
        return False

    def mark_enqueued(self):
        if self.state != "new":
            raise RuntimeError("async owner is not new")
        self.state = "enqueued"

    def _remember_error(self, error):
        if not isinstance(error, BaseException):
            raise TypeError("async owner failure must be an exception")
        if self.error is None:
            self.error = error

    def _remember_secondary(self, error):
        if not isinstance(error, BaseException):
            raise TypeError("async owner secondary failure must be an exception")
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
        try:
            if self._callback is not None:
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
    ):
        if terminal_state not in {"completed", "drained"}:
            raise ValueError("async owner terminal state is invalid")
        self.state = terminal_state
        first_error = self.error
        if self._completion_armed:
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

        self.state = "detached"
        for callback in tuple(self._release_callbacks):
            try:
                callback()
            except BaseException as error:
                if first_error is None:
                    first_error = error
                    self.error = error
                else:
                    self._remember_secondary(error)
        if self._detached is not None:
            try:
                self._detached(self)
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
        self._quarantine = None
        self._resource_recorder = None
        self._resource_releaser = None
        self._release_callbacks = []
        self._completion_armed = False
        if first_error is not None:
            raise first_error
        return True

    def _move_to_quarantine(self):
        self.state = "quarantined"
        quarantine = self._quarantine
        self._detached = None
        self._quarantine = None
        if quarantine is not None:
            quarantine(self)

    def force_quarantine(self, error, *, secondary_errors=()):
        self._remember_error(error)
        for secondary_error in secondary_errors:
            self._remember_secondary(secondary_error)
        if self.state == "quarantined":
            return
        if self.state == "detached":
            return
        self._cancel_unclaimed_async()
        self._move_to_quarantine()

    def fail(self, error, *, secondary_errors=()):
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
                self._detach("drained")
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
            self._detach("drained")
        except BaseException:
            pass
        raise self.error

    def reap(self):
        counted = self._prepare_counted_completion(wait=False)
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
            return self.fail(error)
        if not complete:
            return False
        return self._detach("completed")

    def wait(self):
        counted = self._prepare_counted_completion(wait=True)
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
            wait_event(self._completion_event)
        except BaseException as error:
            return self.fail(error)
        return self._detach("completed")

    def drain(self):
        counted = self._prepare_counted_completion(wait=True)
        if counted is not None:
            return counted
        if self.state == "quarantined":
            raise self.error
        if self.state == "detached":
            if self.error is not None:
                raise self.error
            return True
        try:
            self._drain()
        except BaseException as error:
            self._remember_error(error)
            self._move_to_quarantine()
            raise self.error
        return self._detach("drained")

    def take_result(self):
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
        self.retain_snapshot(
            resources=(owner,),
            allocations=owner.allocations,
            events=owner.events,
            streams=owner.streams,
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
            records = tuple(getattr(resource, "allocation_records", ()))
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
