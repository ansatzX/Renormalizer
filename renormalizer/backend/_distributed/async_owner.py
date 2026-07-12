"""Single-owner lifecycle for resources touched by asynchronous CUDA work."""

from dataclasses import dataclass

import numpy as np


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

    def capture_allocations(self, *arrays):
        if self.state not in {"new", "enqueued"}:
            raise RuntimeError("async owner can no longer capture allocations")
        self._allocations = merge_allocation_records(self._allocations, arrays)

    def capture_resources(self, *resources):
        if self.state not in {"new", "enqueued"}:
            raise RuntimeError("async owner can no longer capture resources")
        self._resources = _unique((*self._resources, *resources))

    def capture_streams(self, *streams):
        if self.state not in {"new", "enqueued"}:
            raise RuntimeError("async owner can no longer capture streams")
        self._streams = _unique((*self._streams, *streams))

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

    def _run_completion(self):
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

    def _detach(self, terminal_state):
        if terminal_state not in {"completed", "drained"}:
            raise ValueError("async owner terminal state is invalid")
        self.state = terminal_state
        first_error = self.error
        if self._completion_armed:
            try:
                self._run_completion()
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
        self._move_to_quarantine()

    def fail(self, error, *, secondary_errors=()):
        self._remember_error(error)
        for secondary_error in secondary_errors:
            self._remember_secondary(secondary_error)
        if self.state == "quarantined":
            raise self.error
        if self.state == "detached":
            raise self.error
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
        self._resources = []
        self.first_error = None

    @property
    def poisoned(self):
        return self.first_error is not None

    @property
    def owners(self):
        return tuple(self._owners)

    @property
    def allocations(self):
        allocations = dict(self._allocations)
        for identity, allocation in self._retained_resource_allocations().items():
            retained = allocations.get(identity)
            if retained is None or allocation.capacity_bytes > retained.capacity_bytes:
                allocations[identity] = allocation
        for owner in self._owners:
            for allocation in owner.allocations:
                retained = allocations.get(allocation.identity)
                if (
                    retained is None
                    or allocation.capacity_bytes > retained.capacity_bytes
                ):
                    allocations[allocation.identity] = allocation
        return tuple(allocations.values())

    def retain(self, owner, error=None):
        if not isinstance(owner, AsyncResourceOwner):
            raise TypeError("terminal quarantine requires an AsyncResourceOwner")
        retained_error = owner.error if error is None else error
        if retained_error is not None and self.first_error is None:
            self.first_error = retained_error
        if all(retained is not owner for retained in self._owners):
            self._owners.append(owner)

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

    def retain_resources(self, resources):
        for resource in resources:
            if resource is not None and all(
                retained is not resource for retained in self._resources
            ):
                self._resources.append(resource)
                self.retain_allocations(getattr(resource, "allocation_records", ()))

    def _retained_resource_allocations(self, kind=None):
        allocations = {}
        for resource in self._resources:
            if kind is not None and (
                getattr(resource, "_terminal_resource_kind", None) != kind
            ):
                continue
            for record in getattr(resource, "allocation_records", ()):
                retained = allocations.get(record.identity)
                if retained is None or record.capacity_bytes > retained.capacity_bytes:
                    allocations[record.identity] = record
        return allocations

    def _retained_events(self):
        return _unique(
            (
                *(event for owner in self._owners for event in owner.events),
                *(
                    event
                    for resource in self._resources
                    for event in getattr(resource, "retained_events", ())
                ),
            )
        )

    def _retained_streams(self):
        return _unique(
            (
                *(stream for owner in self._owners for stream in owner.streams),
                *(
                    stream
                    for resource in self._resources
                    for stream in getattr(resource, "retained_streams", ())
                ),
            )
        )

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
