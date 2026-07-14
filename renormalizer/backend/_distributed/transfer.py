"""Canonical staging bounds and completion-tracked host/device transfers."""

from dataclasses import dataclass, field
import hashlib
import json
import weakref

import numpy as np

from renormalizer.backend._distributed.async_owner import (
    AsyncResourceOwner,
    event_complete as _event_complete,
    require_async_owner,
    wait_event as _wait_event,
)
from renormalizer.backend._distributed.pinned import StagingSlot


def _sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _rank_bytes(values, name):
    try:
        values = tuple(values)
    except TypeError as error:
        raise TypeError("{} must be an iterable".format(name)) from error
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("{} must contain non-negative integers".format(name))
    return values


@dataclass(frozen=True)
class TransferProfile:
    lanes: int
    rank_staging_bytes: tuple[int, ...]
    current_h2d_bytes: tuple[int, ...]
    future_h2d_bytes: tuple[int, ...]
    dirty_d2h_bytes: tuple[int, ...]
    profile_hash: str = field(init=False)

    def __post_init__(self):
        if self.lanes != 1:
            raise ValueError("TransferProfile currently requires exactly one lane")
        fields = {
            name: _rank_bytes(getattr(self, name), name)
            for name in (
                "rank_staging_bytes",
                "current_h2d_bytes",
                "future_h2d_bytes",
                "dirty_d2h_bytes",
            )
        }
        world_size = len(fields["rank_staging_bytes"])
        if world_size == 0 or any(
            len(values) != world_size for values in fields.values()
        ):
            raise ValueError("transfer profile must contain one value per rank")
        expected = tuple(
            max(current, future, dirty)
            for current, future, dirty in zip(
                fields["current_h2d_bytes"],
                fields["future_h2d_bytes"],
                fields["dirty_d2h_bytes"],
            )
        )
        if fields["rank_staging_bytes"] != expected:
            raise ValueError(
                "rank staging bytes do not match canonical transfer bounds"
            )
        for name, values in fields.items():
            object.__setattr__(self, name, values)
        object.__setattr__(
            self,
            "profile_hash",
            _sha256(
                {
                    "schema": "renormalizer.transfer.profile.v1",
                    "lanes": self.lanes,
                    "rank_staging_bytes": list(expected),
                    "current_h2d_bytes": list(fields["current_h2d_bytes"]),
                    "future_h2d_bytes": list(fields["future_h2d_bytes"]),
                    "dirty_d2h_bytes": list(fields["dirty_d2h_bytes"]),
                }
            ),
        )

    @classmethod
    def build(cls, current_h2d_bytes, future_h2d_bytes, dirty_d2h_bytes):
        current = _rank_bytes(current_h2d_bytes, "current_h2d_bytes")
        future = _rank_bytes(future_h2d_bytes, "future_h2d_bytes")
        dirty = _rank_bytes(dirty_d2h_bytes, "dirty_d2h_bytes")
        if (
            len(current) == 0
            or len(current) != len(future)
            or len(current) != len(dirty)
        ):
            raise ValueError("transfer components must contain one value per rank")
        staging = tuple(max(values) for values in zip(current, future, dirty))
        return cls(1, staging, current, future, dirty)


@dataclass(frozen=True)
class TransferSource:
    ref: object
    local_slice: tuple[slice, ...] | None = None
    reverse_axis: int | None = None


class _ImmediateEvent:
    def record(self, stream=None):
        return None

    def query(self):
        return True

    def synchronize(self):
        return None


class AsyncCompletionHandle:
    """Non-owning API view of an AsyncResourceOwner."""

    def __init__(self, owner):
        if not isinstance(owner, AsyncResourceOwner):
            raise TypeError("completion handle requires an AsyncResourceOwner")
        self._owner = owner

    @property
    def owner(self):
        return self._owner

    @property
    def event(self):
        return self._owner.completion_event

    @property
    def completed(self):
        return self._owner.completed

    @property
    def error(self):
        return self._owner.error

    @property
    def terminal_poisoned(self):
        return self._owner.quarantined

    def owned_events(self):
        return self._owner.events

    def reap(self):
        return self._owner.reap()

    def wait(self):
        return self._owner.wait()

    def query(self):
        return self.reap()

    def synchronize(self):
        return self.wait()

    def close(self):
        return self.wait()


class TransferTicket(AsyncCompletionHandle):
    """Non-owning transfer view; the AsyncResourceOwner holds all resources."""

    def __init__(self, owner):
        owner = require_async_owner(owner, "transfer ticket")
        super().__init__(owner)

    @property
    def direction(self):
        return self._owner.direction

    @property
    def nbytes(self):
        return self._owner.nbytes

    @property
    def elapsed_s(self):
        return self._owner.elapsed_s

    @property
    def result(self):
        return self._owner.result

    @property
    def _accounted(self):
        return self._owner.accounted

    @_accounted.setter
    def _accounted(self, value):
        self._owner.accounted = bool(value)

    def take_result(self):
        return self._owner.take_result()

    def detach_terminal_callbacks(self):
        # Quarantine must retain the complete owner, including its callback.
        return None

    def _publish_completion(self):
        self._owner.publish_counted_completion()


class TransferScheduler:
    """Single-lane scheduler with one explicit transfer stream on CuPy."""

    def __init__(
        self,
        store,
        backend,
        pool,
        *,
        reservation=None,
        event_factory=None,
        profile_enabled=False,
        timing_event_factory=None,
        elapsed_time_reader=None,
        quarantine=None,
        _async_admission_factory=None,
        _resource_recorder=None,
        _resource_releaser=None,
    ):
        if not callable(getattr(store, "copy_into", None)):
            raise TypeError("transfer scheduler requires HostTensorStore.copy_into")
        if getattr(backend, "name", None) not in {"numpy", "cupy"}:
            raise ValueError("transfer scheduler backend is unsupported")
        if not callable(getattr(pool, "checkout", None)):
            raise TypeError("transfer scheduler requires a pinned buffer pool")
        if type(profile_enabled) is not bool:
            raise TypeError("profile_enabled must be a boolean")
        if quarantine is not None and not callable(quarantine):
            raise TypeError("quarantine must be callable")
        if _async_admission_factory is not None and not callable(
            _async_admission_factory
        ):
            raise TypeError("async admission factory must be callable")
        if _resource_recorder is not None and not callable(_resource_recorder):
            raise TypeError("resource recorder must be callable")
        if _resource_releaser is not None and not callable(_resource_releaser):
            raise TypeError("resource releaser must be callable")
        for supplied, name in (
            (event_factory, "event_factory"),
            (timing_event_factory, "timing_event_factory"),
            (elapsed_time_reader, "elapsed_time_reader"),
        ):
            if supplied is not None and not callable(supplied):
                raise TypeError("{} must be callable".format(name))
        self.store = store
        self.backend = backend
        self.pool = pool
        self.reservation = reservation
        self._cupy = getattr(backend, "_cupy", None) if backend.name == "cupy" else None
        if self._cupy is None:
            self._stream = None
            default_event_factory = _ImmediateEvent
        else:
            with self._cupy.cuda.Device(backend._device_index):
                self._stream = self._cupy.cuda.Stream(non_blocking=True)
            default_event_factory = lambda: self._cupy.cuda.Event(disable_timing=True)
        self._event_factory = (
            default_event_factory if event_factory is None else event_factory
        )
        self._profile_enabled = profile_enabled
        self._timer = None
        self._timing_event_factory = None
        self._elapsed_time_reader = None
        if profile_enabled and self._cupy is None:
            from time import perf_counter

            self._timer = perf_counter
        elif profile_enabled:
            self._timing_event_factory = (
                (lambda: self._cupy.cuda.Event())
                if timing_event_factory is None
                else timing_event_factory
            )
            self._elapsed_time_reader = (
                self._cupy.cuda.get_elapsed_time
                if elapsed_time_reader is None
                else elapsed_time_reader
            )
        self._tickets = []
        self._owners = {}
        self._events = []
        self._event_owners = {}
        self._quarantined_owners = []
        self._quarantine = quarantine
        self._async_admission_factory = _async_admission_factory
        self._resource_recorder = _resource_recorder
        self._resource_releaser = _resource_releaser
        self._quarantining = False
        self._closed = False
        self._poisoned_error = None
        self.h2d_bytes = 0
        self.d2h_bytes = 0
        self.h2d_s = 0.0
        self.d2h_s = 0.0
        self.last_compute_event = None
        if self._resource_recorder is not None:
            self._resource_recorder(resource=self, streams=(self._stream,))

    @property
    def stream_count(self):
        return int(self._stream is not None)

    @property
    def pending_ticket_count(self):
        return len(self._tickets)

    @property
    def retained_event_count(self):
        return len(self._event_owners) + sum(
            len(owner.events) for owner in self._quarantined_owners
        )

    @property
    def retained_streams(self):
        owners = (*self._owners.values(), *self._quarantined_owners)
        return tuple(
            {
                id(stream): stream
                for stream in (
                    self._stream,
                    *(stream for owner in owners for stream in owner.streams),
                )
                if stream is not None
            }.values()
        )

    @property
    def retained_events(self):
        owners = (*self._owners.values(), *self._quarantined_owners)
        return tuple(
            {
                id(event): event
                for event in (
                    *self._events,
                    *(event for owner in owners for event in owner.events),
                )
            }.values()
        )

    @property
    def poisoned(self):
        return self._poisoned_error is not None

    @property
    def event_count(self):
        self.reap_completed()
        return self.retained_event_count

    def _require_usable(self):
        if self._poisoned_error is not None:
            raise RuntimeError("transfer scheduler is terminal-poisoned") from (
                self._poisoned_error
            )
        if self._closed:
            raise RuntimeError("transfer scheduler is closed")

    def _poison(self, error):
        if self._poisoned_error is None:
            self._poisoned_error = error

    def _register_owner(
        self,
        kind,
        *,
        arrays=(),
        resources=(),
        streams=(),
        callback=None,
        nbytes=0,
        timer=None,
        started_at=None,
        elapsed_reader=None,
        defer_counted_completion=False,
        defer_counted_admission=False,
    ):
        scheduler_ref = weakref.ref(self)
        cupy = self._cupy
        device_index = getattr(self.backend, "_device_index", None)
        holder = {}

        def drain():
            owner = holder["owner"]
            if not owner.streams:
                if owner.completion_event is not None:
                    _wait_event(owner.completion_event)
                return
            first_error = None
            context = cupy.cuda.Device(device_index) if cupy is not None else None
            if context is None:
                entered = False
            else:
                context.__enter__()
                entered = True
            try:
                for stream in owner.streams:
                    try:
                        stream.synchronize()
                    except BaseException as error:
                        if first_error is None:
                            first_error = error
            finally:
                if entered:
                    context.__exit__(None, None, None)
            if first_error is not None:
                raise first_error

        def account(owner):
            scheduler = scheduler_ref()
            if scheduler is not None:
                scheduler._account_owner(owner)

        async_admission = None
        if (
            self._async_admission_factory is not None
            and not defer_counted_admission
        ):
            capability = object()
            async_admission = self._async_admission_factory(
                capability,
                "{}_completion".format(kind),
            )
        try:
            owner = AsyncResourceOwner(
                kind,
                arrays=arrays,
                resources=resources,
                streams=streams,
                callback=callback,
                accounting=account,
                nbytes=nbytes,
                timer=timer,
                started_at=started_at,
                elapsed_reader=elapsed_reader,
                drainer=drain,
                detached=self._owner_detached,
                quarantine=self._owner_quarantined,
                _async_admission=async_admission,
                _resource_recorder=self._resource_recorder,
                _resource_releaser=self._resource_releaser,
                _defer_async_completion=defer_counted_completion,
            )
        except BaseException:
            if async_admission is not None:
                async_admission.cancel()
            raise
        holder["owner"] = owner
        self._owners[id(owner)] = owner
        return owner

    def _owner_detached(self, owner):
        self._owners.pop(id(owner), None)
        identities = {id(event) for event in owner.events}
        self._events = [event for event in self._events if id(event) not in identities]
        for identity in identities:
            self._event_owners.pop(identity, None)
        if (
            self.last_compute_event is not None
            and self.last_compute_event.owner is owner
        ):
            self.last_compute_event = None

    def _retain_quarantined_owner(self, owner):
        self._poison(owner.error)
        self._owner_detached(owner)
        if all(retained is not owner for retained in self._quarantined_owners):
            self._quarantined_owners.append(owner)
        owns_pool_slot = any(
            type(resource) is StagingSlot and resource._pool is self.pool
            for resource in owner.resources
        )
        if owns_pool_slot:
            self.pool._poison(owner.error)

    def _notify_quarantined_owner(self, owner):
        if self._quarantine is not None:
            try:
                self._quarantine(owner)
            except BaseException as error:
                owner._remember_secondary(error)

    def _owner_quarantined(self, owner):
        if self._quarantining:
            self._retain_quarantined_owner(owner)
            return
        self._quarantining = True
        try:
            other_owners = tuple(
                retained for retained in self._owners.values() if retained is not owner
            )
            self._retain_quarantined_owner(owner)
            for retained in other_owners:
                retained.force_quarantine(owner.error)
            for retained in (owner, *other_owners):
                self._notify_quarantined_owner(retained)
        finally:
            self._quarantining = False

    def _new_event(self, owner, *, completion=False):
        return self._new_event_from(owner, self._event_factory, completion=completion)

    def _new_timing_event(self, owner, *, completion=False):
        return self._new_event_from(
            owner, self._timing_event_factory, completion=completion
        )

    def _new_event_from(self, owner, factory, *, completion=False):
        event = factory()
        owner.add_event(event, completion=completion)
        self._events.append(event)
        self._event_owners[id(event)] = owner
        if not callable(getattr(event, "record", None)):
            raise TypeError("transfer event must provide record()")
        if (
            not callable(getattr(event, "query", None))
            and type(getattr(event, "done", None)) is not bool
        ):
            raise TypeError("transfer event must provide query() or done")
        if not callable(getattr(event, "synchronize", None)):
            raise TypeError("transfer event must provide synchronize()")
        return event

    @staticmethod
    def _record(event, stream=None):
        event.record(stream)

    @staticmethod
    def _transfer_order(array):
        if bool(array.flags.f_contiguous) and not bool(array.flags.c_contiguous):
            return "F"
        return "C"

    def _host_view(self, slot, shape, dtype, *, order="C"):
        return slot.view(tuple(shape), np.dtype(dtype), order=order)

    def _copy_h2d(self, destination, staging):
        runtime = getattr(self._cupy.cuda, "runtime", None)
        memcpy_async = getattr(runtime, "memcpyAsync", None)
        stream_pointer = getattr(self._stream, "ptr", None)
        destination_pointer = getattr(getattr(destination, "data", None), "ptr", None)
        if (
            callable(memcpy_async)
            and stream_pointer is not None
            and destination_pointer is not None
        ):
            memcpy_async(
                int(destination_pointer),
                int(staging.ctypes.data),
                int(staging.nbytes),
                runtime.memcpyHostToDevice,
                int(stream_pointer),
            )
            return
        destination.set(staging, stream=self._stream)

    def _source(self, source_ref):
        if isinstance(source_ref, TransferSource):
            return (
                source_ref.ref,
                source_ref.local_slice,
                source_ref.reverse_axis,
            )
        return source_ref, None, None

    def stage_h2d(self, source_ref, destination, slot, *, cache_lease=None):
        self._require_usable()
        ref, local_slice, reverse_axis = self._source(source_ref)
        staging = self._host_view(
            slot,
            destination.shape,
            destination.dtype,
            order=self._transfer_order(destination),
        )
        copy_destination = (
            staging if reverse_axis is None else np.flip(staging, axis=reverse_axis)
        )
        start = self._timer() if self._timer is not None else None
        timing = {}
        elapsed_time_reader = self._elapsed_time_reader

        def elapsed_reader():
            return float(elapsed_time_reader(timing["start"], timing["end"])) / 1000.0

        owner = self._register_owner(
            "h2d",
            arrays=(staging, destination),
            resources=(source_ref, slot, cache_lease),
            streams=(self._stream,),
            nbytes=int(destination.nbytes),
            timer=self._timer if self._profile_enabled and self._cupy is None else None,
            started_at=start,
            elapsed_reader=(
                elapsed_reader
                if self._profile_enabled and self._cupy is not None
                else None
            ),
            defer_counted_completion=True,
        )
        ticket = TransferTicket(owner)
        self._tickets.append(ticket)
        try:
            slot.retain_until(owner)
            self.store.copy_into(ref, copy_destination, local_slice)
            if self._cupy is None:
                event = self._new_event(owner, completion=True)
                owner.mark_enqueued()
                np.copyto(destination, staging, casting="no")
                self._record(event)
                owner.arm_completion()
            else:
                with self._cupy.cuda.Device(self.backend._device_index):
                    if self._profile_enabled:
                        timing["start"] = self._new_timing_event(owner)
                    event = (
                        self._new_timing_event(owner, completion=True)
                        if self._profile_enabled
                        else self._new_event(owner, completion=True)
                    )
                    timing["end"] = event
                    owner.mark_enqueued()
                    if self._profile_enabled:
                        self._record(timing["start"], self._stream)
                    self._copy_h2d(destination, staging)
                    self._record(event, self._stream)
                    owner.arm_completion()
        except BaseException as error:
            try:
                owner.fail(error)
            finally:
                if ticket in self._tickets:
                    self._tickets.remove(ticket)
        return ticket

    def wait_for_h2d(self, event):
        self._require_usable()
        owner = self._event_owners.get(id(event))
        if owner is not None:
            if owner.reap():
                return
        else:
            try:
                if _event_complete(event):
                    return
            except BaseException:
                raise
        if self._cupy is None:
            if owner is not None:
                owner.wait()
            else:
                _wait_event(event)
            return
        with self._cupy.cuda.Device(self.backend._device_index):
            stream = self._cupy.cuda.get_current_stream()
            if owner is not None:
                owner.capture_streams(stream)
            try:
                stream.wait_event(event)
            except BaseException as error:
                if owner is not None:
                    owner.fail(error)
                try:
                    stream.synchronize()
                except BaseException:
                    self._poison(error)
                raise

    @staticmethod
    def _compute_arrays(bindings, arrays):
        captured_arrays = []
        identities = set()
        if bindings is not None:
            for array in bindings.arrays.values():
                if id(array) not in identities:
                    identities.add(id(array))
                    captured_arrays.append(array)
        for array in arrays:
            if id(array) not in identities:
                identities.add(id(array))
                captured_arrays.append(array)
        return tuple(captured_arrays)

    def begin_compute(self, *, cache_leases=(), bindings=None, arrays=(), resources=()):
        self._require_usable()
        cache_leases = tuple(cache_leases)
        resources = tuple(resources)
        captured_arrays = self._compute_arrays(bindings, arrays)
        owner = self._register_owner(
            "compute",
            arrays=captured_arrays,
            resources=(bindings, *cache_leases, *resources),
            nbytes=sum(int(array.nbytes) for array in captured_arrays),
            defer_counted_admission=True,
        )
        try:
            if self._cupy is not None:
                with self._cupy.cuda.Device(self.backend._device_index):
                    stream = self._cupy.cuda.get_current_stream()
                    owner.capture_streams(stream)
            for lease in cache_leases:
                transfer = getattr(lease, "transfer_to", None)
                if callable(transfer):
                    transfer(owner)
            owner.mark_enqueued()
            owner.arm_completion()
        except BaseException as error:
            owner.fail(error)
        return AsyncCompletionHandle(owner)

    def record_compute_completion(
        self,
        *,
        handle=None,
        cache_leases=(),
        bindings=None,
        arrays=(),
    ):
        self._require_usable()
        if handle is None:
            handle = self.begin_compute(
                cache_leases=cache_leases,
                bindings=bindings,
                arrays=arrays,
            )
            arrays = ()
        elif cache_leases or bindings is not None:
            raise ValueError(
                "an existing compute handle already owns bindings and cache leases"
            )
        if not isinstance(handle, AsyncCompletionHandle):
            raise TypeError("compute completion requires an AsyncCompletionHandle")
        owner = handle.owner
        if (
            owner.kind != "compute"
            or owner.state != "enqueued"
            or self._owners.get(id(owner)) is not owner
        ):
            raise RuntimeError("compute owner is not active")
        owner.capture_arrays(*arrays)
        try:
            if self._async_admission_factory is not None:
                capability = object()
                owner.install_counted_admission(
                    self._async_admission_factory(
                        capability,
                        "compute_completion",
                    )
                )
            event = self._new_event(owner, completion=True)
            stream = None if not owner.streams else owner.streams[0]
            self._record(event, stream)
            owner.watch_counted_completion()
        except BaseException as error:
            owner.fail(error)
        self.last_compute_event = handle
        return handle

    def writeback_d2h(self, source, destination_ref, slot):
        self._require_usable()
        staging = self._host_view(slot, source.shape, source.dtype)
        if tuple(source.shape) != tuple(destination_ref.shape):
            raise ValueError("dirty source shape does not match destination ref")
        start = self._timer() if self._timer is not None else None

        reservation = self.reservation
        store = self.store

        def commit():
            if reservation is not None:
                return reservation.commit(destination_ref, staging)
            return store.update(
                destination_ref.key,
                staging,
                expected_version=destination_ref.version,
            )

        timing = {}
        elapsed_time_reader = self._elapsed_time_reader

        def elapsed_reader():
            return float(elapsed_time_reader(timing["start"], timing["end"])) / 1000.0

        owner = self._register_owner(
            "d2h",
            arrays=(source, staging),
            resources=(destination_ref, slot),
            streams=(self._stream,),
            callback=commit,
            nbytes=int(source.nbytes),
            timer=self._timer if self._profile_enabled and self._cupy is None else None,
            started_at=start,
            elapsed_reader=(
                elapsed_reader
                if self._profile_enabled and self._cupy is not None
                else None
            ),
        )
        ticket = TransferTicket(owner)
        self._tickets.append(ticket)
        try:
            slot.retain_until(owner)
            if self._cupy is None:
                owner.mark_enqueued()
                compute_event = self._new_event(owner)
                transfer_event = self._new_event(owner, completion=True)
                self._record(compute_event)
                np.copyto(staging, source, casting="no")
                self._record(transfer_event)
                owner.arm_completion()
            else:
                with self._cupy.cuda.Device(self.backend._device_index):
                    compute_stream = self._cupy.cuda.get_current_stream()
                    owner.capture_streams(compute_stream)
                    owner.mark_enqueued()
                    compute_event = self._new_event(owner)
                    self._record(compute_event, compute_stream)
                    self._stream.wait_event(compute_event)
                    if self._profile_enabled:
                        timing["start"] = self._new_timing_event(owner)
                        self._record(timing["start"], self._stream)
                    transfer_event = (
                        self._new_timing_event(owner, completion=True)
                        if self._profile_enabled
                        else self._new_event(owner, completion=True)
                    )
                    timing["end"] = transfer_event
                    source.get(out=staging, stream=self._stream, blocking=False)
                    self._record(transfer_event, self._stream)
                    owner.arm_completion()
        except BaseException as error:
            try:
                owner.fail(error)
            finally:
                if ticket in self._tickets:
                    self._tickets.remove(ticket)
        return ticket

    def reap_completed(self):
        self._require_usable()
        errors = []
        for owner in tuple(self._owners.values()):
            try:
                owner.reap()
            except BaseException as error:
                errors.append(error)
        self._tickets = [ticket for ticket in self._tickets if not ticket.completed]
        if errors:
            raise errors[0]

    def _account_owner(self, owner):
        if owner.direction == "h2d":
            self.h2d_bytes += owner.nbytes
            self.h2d_s += owner.elapsed_s
        elif owner.direction == "d2h":
            self.d2h_bytes += owner.nbytes
            self.d2h_s += owner.elapsed_s

    def complete_all(self):
        self._require_usable()
        errors = []
        for owner in tuple(self._owners.values()):
            try:
                if owner.completion_event is None:
                    owner.drain()
                else:
                    owner.wait()
            except BaseException as error:
                errors.append(error)
        self._tickets = [ticket for ticket in self._tickets if not ticket.completed]
        if errors:
            raise errors[0]

    def _start_counted_completions(self):
        for owner in tuple(self._owners.values()):
            owner.start_counted_completion()

    def close(self):
        if self._closed:
            if self._poisoned_error is not None:
                raise self._poisoned_error
            return
        error = None
        if self._poisoned_error is None:
            try:
                self.complete_all()
            except BaseException as caught:
                error = caught
        else:
            error = self._poisoned_error
        self._tickets.clear()
        self._owners.clear()
        self._events.clear()
        self._event_owners.clear()
        self._stream = None
        self.last_compute_event = None
        self._cupy = None
        self.store = None
        self.backend = None
        self.pool = None
        self.reservation = None
        self._event_factory = None
        self._timer = None
        self._timing_event_factory = None
        self._elapsed_time_reader = None
        self._quarantine = None
        self._async_admission_factory = None
        self._resource_recorder = None
        self._resource_releaser = None
        self._closed = True
        if self._poisoned_error is None:
            self._profile_enabled = False
        else:
            if error is None:
                error = self._poisoned_error
        if error is not None:
            raise error


__all__ = [
    "TransferProfile",
    "TransferScheduler",
    "TransferSource",
    "TransferTicket",
]
