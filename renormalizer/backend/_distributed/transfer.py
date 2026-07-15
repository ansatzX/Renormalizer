"""Canonical staging bounds and completion-tracked host/device transfers."""

from dataclasses import dataclass, field
import hashlib
import json
import threading
import weakref

import numpy as np

from renormalizer.backend._distributed.async_owner import (
    AsyncResourceOwner,
    _require_resource_admission,
    event_complete as _event_complete,
    require_async_owner,
    wait_event as _wait_event,
)
from renormalizer.backend._distributed.pinned import StagingSlot
from renormalizer.backend._distributed.terminal import (
    _publish_lease_construction_resource,
    _publish_lease_construction_resource_direct,
    _remaining_lifecycle_time,
    _require_managed_resource_admission,
    _resolve_managed_resource_guard,
)


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
        self._managed_guard = owner._managed_guard
        self._managed_epoch = owner._managed_epoch

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

    @property
    def owner(self):
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
                "scheduler_complete",
            )
        )
        return self._owner

    @property
    def event(self):
        self._require_admission(
            allowed_operations=(
                "acquire",
                "cache_wait",
                "h2d_completion",
                "load",
                "operator_call",
                "prefetch",
                "reap",
                "resource_state",
                "schedule_writeback",
                "scheduler_complete",
            )
        )
        return self._owner.completion_event

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
        return self._owner.completed

    @property
    def error(self):
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
        return self._owner.error

    @property
    def terminal_poisoned(self):
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
        return self._owner.quarantined

    def owned_events(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=("resource_state",),
        )
        return self._owner.events

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
                "pool_reap",
                "prefetch",
                "reap",
                "schedule_writeback",
                "scheduler_complete",
            ),
        )
        return self._owner.reap(
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
            _wait_for_counted=_wait_for_counted,
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
                "load",
                "operator_call",
                "pool_reap",
                "prefetch",
                "scheduler_complete",
            ),
        )
        return self._owner.wait(
            _deadline=_deadline,
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
        )

    def query(self):
        self._require_admission(allowed_operations=("reap",))
        return self.reap()

    def synchronize(self):
        self._require_admission(allowed_operations=("scheduler_complete",))
        return self.wait()

    def close(self):
        self._require_admission(allowed_operations=("scheduler_complete",))
        return self.wait()


class TransferTicket(AsyncCompletionHandle):
    """Non-owning transfer view; the AsyncResourceOwner holds all resources."""

    def __init__(self, owner):
        owner = require_async_owner(owner, "transfer ticket")
        super().__init__(owner)

    @property
    def direction(self):
        self._require_admission(
            allowed_operations=("emit_profile", "resource_state", "scheduler_complete")
        )
        return self._owner.direction

    @property
    def nbytes(self):
        self._require_admission(
            allowed_operations=("emit_profile", "resource_state", "scheduler_complete")
        )
        return self._owner.nbytes

    @property
    def elapsed_s(self):
        self._require_admission(
            allowed_operations=("emit_profile", "resource_state", "scheduler_complete")
        )
        return self._owner.elapsed_s

    @property
    def result(self):
        self._require_admission(
            allowed_operations=("resource_state", "scheduler_complete")
        )
        return self._owner.result

    @property
    def _accounted(self):
        self._require_admission(allowed_operations=("scheduler_complete",))
        return self._owner.accounted

    @_accounted.setter
    def _accounted(self, value):
        self._require_admission(allowed_operations=("scheduler_complete",))
        self._owner.accounted = bool(value)

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
        return self._owner.take_result(
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
        )

    def detach_terminal_callbacks(self):
        # Quarantine must retain the complete owner, including its callback.
        return None

    def _publish_completion(self):
        self._owner.publish_counted_completion()


@dataclass(frozen=True)
class _OwnerDetachPreparation:
    owner_identity: int
    capability: object
    event_identities: tuple


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
        _admission_token=None,
        _admission_validator=None,
        _construction_slot=None,
        _managed_guard=None,
        _standalone=True,
        _stream_factory=None,
    ):
        _require_resource_admission(_admission_token, _admission_validator)
        self._managed_guard = _resolve_managed_resource_guard(
            standalone=_standalone,
            guard=_managed_guard,
        )
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
        if _stream_factory is not None and not callable(_stream_factory):
            raise TypeError("scheduler stream factory must be callable")
        self._managed = self._managed_guard is not None
        self._managed_epoch = (
            None if _admission_token is None else _admission_token.epoch
        )
        self.store = store
        self.backend = backend
        self.pool = pool
        self.reservation = reservation
        self._cupy = getattr(backend, "_cupy", None) if backend.name == "cupy" else None
        self._stream = None
        self._event_factory = event_factory
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
        # Lock order: async owner -> scheduler collections. Owner callbacks are
        # invoked only from collection snapshots taken before releasing this lock.
        self._collection_lock = threading.RLock()
        self._owners = {}
        self._owner_detach_capabilities = {}
        self._owner_detach_preparations = {}
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
        _publish_lease_construction_resource_direct(
            _construction_slot,
            self,
        )
        if self._cupy is None:
            default_event_factory = _ImmediateEvent
        else:
            stream_factory = (
                self._create_stream_owned
                if _stream_factory is None
                else _stream_factory
            )
            self._stream = stream_factory(
                cupy=self._cupy,
                device_index=backend._device_index,
                scheduler=self,
                construction_slot=_construction_slot,
            )
            default_event_factory = lambda: self._cupy.cuda.Event(disable_timing=True)
        self._event_factory = (
            default_event_factory if event_factory is None else event_factory
        )
        _publish_lease_construction_resource_direct(
            _construction_slot,
            self,
            streams=(self._stream,),
        )
        _publish_lease_construction_resource(
            _construction_slot,
            self,
            streams=(self._stream,),
        )
        if self._resource_recorder is not None:
            self._resource_recorder(resource=self, streams=(self._stream,))

    @staticmethod
    def _create_stream_owned(
        *,
        cupy,
        device_index,
        scheduler,
        construction_slot,
    ):
        with cupy.cuda.Device(device_index):
            stream = cupy.cuda.Stream(non_blocking=True)
        _publish_lease_construction_resource_direct(
            construction_slot,
            scheduler,
            streams=(stream,),
        )
        return stream

    @property
    def stream_count(self):
        self._require_admission(
            None,
            None,
            allowed_operations=(
                "lease_construction",
                "resource_state",
            ),
        )
        return int(self._stream is not None)

    @property
    def pending_ticket_count(self):
        self._require_admission(
            None,
            None,
            allowed_operations=("resource_state",),
        )
        with self._collection_lock:
            return len(self._tickets)

    @property
    def retained_event_count(self):
        self._require_admission(
            None,
            None,
            allowed_operations=("resource_state",),
        )
        with self._collection_lock:
            event_owner_count = len(self._event_owners)
            quarantined = tuple(self._quarantined_owners)
        return event_owner_count + sum(len(owner.events) for owner in quarantined)

    @property
    def retained_streams(self):
        self._require_admission(
            None,
            None,
            allowed_operations=(
                "compute_completion",
                "d2h_completion",
                "h2d_completion",
                "lease_construction",
                "resource_state",
            ),
        )
        with self._collection_lock:
            owners = (*self._owners.values(), *self._quarantined_owners)
            primary_stream = self._stream
        return tuple(
            {
                id(stream): stream
                for stream in (
                    primary_stream,
                    *(stream for owner in owners for stream in owner.streams),
                )
                if stream is not None
            }.values()
        )

    @property
    def retained_events(self):
        self._require_admission(
            None,
            None,
            allowed_operations=(
                "compute_completion",
                "d2h_completion",
                "h2d_completion",
                "lease_construction",
                "resource_state",
            ),
        )
        with self._collection_lock:
            owners = (*self._owners.values(), *self._quarantined_owners)
            events = tuple(self._events)
        return tuple(
            {
                id(event): event
                for event in (
                    *events,
                    *(event for owner in owners for event in owner.events),
                )
            }.values()
        )

    @property
    def poisoned(self):
        self._require_admission(
            None,
            None,
            allowed_operations=(
                "acquire",
                "child_close",
                "close_progress",
                "load",
                "operator_call",
                "prefetch",
                "reap",
                "resource_state",
                "schedule_writeback",
                "scheduler_close",
                "scheduler_complete",
            ),
        )
        with self._collection_lock:
            return self._poisoned_error is not None

    @property
    def event_count(self):
        self._require_admission(None, None, allowed_operations=("resource_state",))
        self.reap_completed()
        return self.retained_event_count

    def _runtime_event_count(
        self,
        *,
        _admission_token,
        _admission_validator,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=("resource_state",),
        )
        self.reap_completed(
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
        )
        return self.retained_event_count

    def _last_compute_completion(
        self,
        *,
        _admission_token,
        _admission_validator,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=("acquire", "child_close", "operator_call"),
        )
        with self._collection_lock:
            return self.last_compute_event

    def _require_usable(self):
        with self._collection_lock:
            poisoned_error = self._poisoned_error
            closed = self._closed
        if poisoned_error is not None:
            raise RuntimeError("transfer scheduler is terminal-poisoned") from (
                poisoned_error
            )
        if closed:
            raise RuntimeError("transfer scheduler is closed")

    def _require_admission(
        self,
        token,
        validator,
        *,
        allowed_operations=(),
    ):
        guard = getattr(self, "_managed_guard", None)
        if (
            guard is None
            and getattr(self, "_managed", False)
            and token is None
            and validator is None
        ):
            raise TypeError("managed transfer scheduler requires admission")
        return _require_managed_resource_admission(
            guard,
            token,
            validator,
            allowed_scopes=("construction", "lease", "lease_close"),
            allowed_operations=allowed_operations,
            epoch=lambda _token: self._managed_epoch,
        )

    def _poison(self, error):
        with self._collection_lock:
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
        callback_requires_admission=False,
        nbytes=0,
        timer=None,
        started_at=None,
        elapsed_reader=None,
        defer_counted_completion=False,
        defer_counted_admission=False,
    ):
        allowed_operations = {
            "compute": ("acquire", "child_close", "operator_call"),
            "d2h": ("close_progress", "schedule_writeback"),
            "h2d": ("acquire", "load", "operator_call", "prefetch"),
        }.get(kind)
        if allowed_operations is None:
            raise RuntimeError("unknown managed async owner family")
        self._require_admission(
            None,
            None,
            allowed_operations=allowed_operations,
        )
        scheduler_ref = weakref.ref(self)
        detach_capability = object()
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

        def detached(
            owner,
            *,
            _admission_token=None,
            _admission_validator=None,
        ):
            scheduler = scheduler_ref()
            if scheduler is None:
                return None
            return scheduler._owner_detached(
                owner,
                _detach_capability=detach_capability,
                _admission_token=_admission_token,
                _admission_validator=_admission_validator,
            )

        def commit_detached(owner, preparation):
            scheduler = scheduler_ref()
            if scheduler is not None:
                scheduler._commit_owner_detached(
                    owner,
                    preparation,
                    _detach_capability=detach_capability,
                )

        def quarantined(owner):
            scheduler = scheduler_ref()
            if scheduler is not None:
                scheduler._owner_quarantined(
                    owner,
                    _detach_capability=detach_capability,
                    _resource_snapshot=owner._terminal_resource_snapshot(),
                    _terminal_error=owner.error,
                    _terminal_state=owner.state,
                )

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
                detached=detached,
                _detached_commit=commit_detached,
                _detached_requires_admission=True,
                quarantine=quarantined,
                _async_admission=async_admission,
                _resource_recorder=self._resource_recorder,
                _resource_releaser=self._resource_releaser,
                _defer_async_completion=defer_counted_completion,
                _callback_requires_admission=callback_requires_admission,
                _managed_guard=self._managed_guard,
                _managed_epoch=self._managed_epoch,
            )
        except BaseException:
            if async_admission is not None:
                async_admission.cancel()
            raise
        holder["owner"] = owner
        with self._collection_lock:
            self._owners[id(owner)] = owner
            self._owner_detach_capabilities[id(owner)] = detach_capability
        owner._consume_counted_start_request()
        return owner

    def _owner_detached(
        self,
        owner,
        *,
        _detach_capability=None,
        _admission_token=None,
        _admission_validator=None,
    ):
        allowed_operations = (
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
        )
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=allowed_operations,
        )
        owner._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=allowed_operations,
        )
        identity = id(owner)
        event_identities = tuple(id(event) for event in owner._events)
        with self._collection_lock:
            if (
                self._owners.get(identity) is not owner
                or self._owner_detach_capabilities.get(identity)
                is not _detach_capability
            ):
                raise RuntimeError(
                    "scheduler owner detach capability does not own this transition"
                )
            preparation = self._owner_detach_preparations.get(identity)
            if preparation is None:
                preparation = _OwnerDetachPreparation(
                    identity,
                    _detach_capability,
                    event_identities,
                )
                self._owner_detach_preparations[identity] = preparation
            elif (
                preparation.capability is not _detach_capability
                or preparation.event_identities != event_identities
            ):
                raise RuntimeError("scheduler owner detach preparation changed")
            return preparation

    def _commit_owner_detached(
        self,
        owner,
        preparation,
        *,
        _detach_capability,
    ):
        identity = id(owner)
        with self._collection_lock:
            if (
                not isinstance(preparation, _OwnerDetachPreparation)
                or preparation.owner_identity != identity
                or preparation.capability is not _detach_capability
                or self._owner_detach_preparations.get(identity) is not preparation
                or self._owners.get(identity) is not owner
                or self._owner_detach_capabilities.get(identity)
                is not _detach_capability
            ):
                raise RuntimeError("scheduler owner detach preparation is stale")
            self._remove_owner_membership_locked(
                owner,
                preparation.event_identities,
            )

    def _remove_owner_membership_locked(self, owner, event_identities):
        identity = id(owner)
        self._owners.pop(identity, None)
        self._owner_detach_capabilities.pop(identity, None)
        self._owner_detach_preparations.pop(identity, None)
        identities = frozenset(event_identities)
        self._events[:] = [
            event for event in self._events if id(event) not in identities
        ]
        for event_identity in identities:
            self._event_owners.pop(event_identity, None)
        if (
            self.last_compute_event is not None
            and self.last_compute_event._owner is owner
        ):
            self.last_compute_event = None

    def _retain_quarantined_owner_locked(
        self,
        owner,
        detach_capability,
        resource_snapshot,
        terminal_error,
        terminal_state,
    ):
        identity = id(owner)
        if (
            terminal_state != "quarantined"
            or self._owners.get(identity) is not owner
            or self._owner_detach_capabilities.get(identity)
            is not detach_capability
        ):
            raise RuntimeError(
                "scheduler owner quarantine capability does not own this transition"
            )
        if self._poisoned_error is None:
            self._poisoned_error = terminal_error
        event_identities = tuple(
            id(event) for event in resource_snapshot["events"]
        )
        self._remove_owner_membership_locked(owner, event_identities)
        if all(retained is not owner for retained in self._quarantined_owners):
            self._quarantined_owners.append(owner)
        return any(
            type(resource) is StagingSlot and resource._pool is self.pool
            for resource in resource_snapshot["resources"]
        )

    def _notify_quarantined_owner(self, owner):
        with self._collection_lock:
            quarantine = self._quarantine
        if quarantine is not None:
            try:
                quarantine(owner)
            except BaseException as error:
                owner._remember_secondary(error)

    def _owner_quarantined(
        self,
        owner,
        *,
        _detach_capability=None,
        _resource_snapshot,
        _terminal_error,
        _terminal_state,
    ):
        with self._collection_lock:
            nested = self._quarantining
            if not nested:
                self._quarantining = True
                other_owners = tuple(
                    retained
                    for retained in self._owners.values()
                    if retained is not owner
                )
            else:
                other_owners = ()
            owns_pool_slot = self._retain_quarantined_owner_locked(
                owner,
                _detach_capability,
                _resource_snapshot,
                _terminal_error,
                _terminal_state,
            )
        if owns_pool_slot:
            self.pool._poison(_terminal_error)
        if nested:
            return
        try:
            for retained in other_owners:
                retained.force_quarantine(_terminal_error)
            for retained in (owner, *other_owners):
                self._notify_quarantined_owner(retained)
        finally:
            with self._collection_lock:
                self._quarantining = False

    def _new_event(self, owner, *, completion=False):
        return self._new_event_from(owner, self._event_factory, completion=completion)

    def _new_timing_event(self, owner, *, completion=False):
        return self._new_event_from(
            owner, self._timing_event_factory, completion=completion
        )

    def _new_event_from(self, owner, factory, *, completion=False):
        self._require_admission(
            None,
            None,
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
        event = self._create_event_owned(
            owner=owner,
            factory=factory,
            completion=completion,
        )
        with self._collection_lock:
            if self._owners.get(id(owner)) is not owner:
                raise RuntimeError("transfer event owner is no longer active")
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
    def _create_event_owned(*, owner, factory, completion):
        event = factory()
        owner.add_event(event, completion=completion)
        return event

    def _record(self, event, stream=None):
        self._require_admission(
            None,
            None,
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
        event.record(stream)

    @staticmethod
    def _transfer_order(array):
        if bool(array.flags.f_contiguous) and not bool(array.flags.c_contiguous):
            return "F"
        return "C"

    def _host_view(
        self,
        slot,
        shape,
        dtype,
        *,
        order="C",
        checkout_capability=None,
        admission_token=None,
        admission_validator=None,
    ):
        kwargs = {}
        if checkout_capability is not None:
            kwargs = {
                "_checkout_capability": checkout_capability,
                "_admission_token": admission_token,
                "_admission_validator": admission_validator,
            }
        return slot.view(tuple(shape), np.dtype(dtype), order=order, **kwargs)

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

    def stage_h2d(
        self,
        source_ref,
        destination,
        slot,
        *,
        cache_lease=None,
        _checkout_capability=None,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=("acquire", "load", "operator_call", "prefetch"),
        )
        self._require_usable()
        ref, local_slice, reverse_axis = self._source(source_ref)
        staging = self._host_view(
            slot,
            destination.shape,
            destination.dtype,
            order=self._transfer_order(destination),
            checkout_capability=_checkout_capability,
            admission_token=_admission_token,
            admission_validator=_admission_validator,
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
        with self._collection_lock:
            self._tickets.append(ticket)
        try:
            if _checkout_capability is None:
                slot.retain_until(owner)
            else:
                slot.retain_until(
                    owner,
                    _checkout_capability=_checkout_capability,
                    _admission_token=_admission_token,
                    _admission_validator=_admission_validator,
                )
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
                with self._collection_lock:
                    if ticket in self._tickets:
                        self._tickets.remove(ticket)
        return ticket

    def wait_for_h2d(
        self,
        event,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=("acquire", "load", "operator_call"),
        )
        self._require_usable()
        with self._collection_lock:
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

    def begin_compute(
        self,
        *,
        cache_leases=(),
        bindings=None,
        arrays=(),
        resources=(),
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=("acquire", "child_close", "operator_call"),
        )
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
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=("acquire", "child_close", "operator_call"),
        )
        self._require_usable()
        if handle is None:
            handle = self.begin_compute(
                cache_leases=cache_leases,
                bindings=bindings,
                arrays=arrays,
                _admission_token=_admission_token,
                _admission_validator=_admission_validator,
            )
            arrays = ()
        elif cache_leases or bindings is not None:
            raise ValueError(
                "an existing compute handle already owns bindings and cache leases"
            )
        if not isinstance(handle, AsyncCompletionHandle):
            raise TypeError("compute completion requires an AsyncCompletionHandle")
        owner = handle.owner
        with self._collection_lock:
            active_owner = self._owners.get(id(owner))
        if (
            owner.kind != "compute"
            or owner.state != "enqueued"
            or active_owner is not owner
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
        with self._collection_lock:
            self.last_compute_event = handle
        return handle

    def writeback_d2h(
        self,
        source,
        destination_ref,
        slot,
        *,
        _checkout_capability=None,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=("close_progress", "schedule_writeback"),
        )
        self._require_usable()
        staging = self._host_view(
            slot,
            source.shape,
            source.dtype,
            checkout_capability=_checkout_capability,
            admission_token=_admission_token,
            admission_validator=_admission_validator,
        )
        if tuple(source.shape) != tuple(destination_ref.shape):
            raise ValueError("dirty source shape does not match destination ref")
        start = self._timer() if self._timer is not None else None

        reservation = self.reservation
        store = self.store

        def commit(*, _admission_token=None, _admission_validator=None):
            return self._commit_writeback(
                reservation,
                store,
                destination_ref,
                staging,
                _admission_token=_admission_token,
                _admission_validator=_admission_validator,
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
            callback_requires_admission=True,
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
        with self._collection_lock:
            self._tickets.append(ticket)
        try:
            if _checkout_capability is None:
                slot.retain_until(owner)
            else:
                slot.retain_until(
                    owner,
                    _checkout_capability=_checkout_capability,
                    _admission_token=_admission_token,
                    _admission_validator=_admission_validator,
                )
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
                with self._collection_lock:
                    if ticket in self._tickets:
                        self._tickets.remove(ticket)
        return ticket

    @staticmethod
    def _commit_writeback(
        reservation,
        store,
        destination_ref,
        staging,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        _require_resource_admission(
            _admission_token,
            _admission_validator,
        )
        if reservation is not None:
            from renormalizer.backend._distributed.residency import (
                _HostTensorReservation,
            )

            if isinstance(reservation, _HostTensorReservation):
                return reservation.commit(
                    destination_ref,
                    staging,
                    _admission_token=_admission_token,
                    _admission_validator=_admission_validator,
                )
            return reservation.commit(destination_ref, staging)
        return store.update(
            destination_ref.key,
            staging,
            expected_version=destination_ref.version,
        )

    def reap_completed(
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
                "pool_reap",
                "prefetch",
                "reap",
                "resource_state",
                "schedule_writeback",
                "scheduler_close",
                "scheduler_complete",
            ),
        )
        self._require_usable()
        errors = []
        with self._collection_lock:
            owners = tuple(self._owners.values())
        for owner in owners:
            try:
                owner.reap(_wait_for_counted=_wait_for_counted)
            except BaseException as error:
                if self._managed:
                    raise
                errors.append(error)
        with self._collection_lock:
            tickets = tuple(self._tickets)
        completed = {id(ticket) for ticket in tickets if ticket.completed}
        with self._collection_lock:
            self._tickets[:] = [
                ticket for ticket in self._tickets if id(ticket) not in completed
            ]
        if errors:
            raise errors[0]

    def _account_owner(self, owner):
        with self._collection_lock:
            if owner.direction == "h2d":
                self.h2d_bytes += owner.nbytes
                self.h2d_s += owner.elapsed_s
            elif owner.direction == "d2h":
                self.d2h_bytes += owner.nbytes
                self.d2h_s += owner.elapsed_s

    def complete_all(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
        _deadline=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=(
                "lease_construction",
                "scheduler_close",
                "scheduler_complete",
            ),
        )
        _remaining_lifecycle_time(
            _deadline,
            "transfer scheduler lifecycle timed out before completion drain",
        )
        self._require_usable()
        errors = []
        with self._collection_lock:
            owners = tuple(self._owners.values())
        for owner in owners:
            try:
                if owner.completion_event is None:
                    owner.drain(_deadline=_deadline)
                else:
                    owner.wait(_deadline=_deadline)
            except BaseException as error:
                if self._managed:
                    raise
                errors.append(error)
        with self._collection_lock:
            tickets = tuple(self._tickets)
        completed = {id(ticket) for ticket in tickets if ticket.completed}
        with self._collection_lock:
            self._tickets[:] = [
                ticket for ticket in self._tickets if id(ticket) not in completed
            ]
        if errors:
            raise errors[0]

    def _start_counted_completions(self, *, _close_transition=None):
        if self._managed_guard is not None:
            requested = self._managed_guard.request_counted_starts(
                _close_transition,
                epoch=self._managed_epoch,
            )
            if requested is None:
                return False
        with self._collection_lock:
            owners = tuple(self._owners.values())
        for owner in owners:
            if self._managed_guard is None:
                owner._start_counted_completion()
            else:
                owner._consume_counted_start_request(
                    _close_transition=_close_transition,
                )
        return True

    def close(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
        _deadline=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=("lease_construction", "scheduler_close"),
        )
        _remaining_lifecycle_time(
            _deadline,
            "transfer scheduler lifecycle timed out before close",
        )
        with self._collection_lock:
            closed = self._closed
            poisoned_error = self._poisoned_error
        if closed:
            if poisoned_error is not None:
                raise poisoned_error
            return
        error = None
        if poisoned_error is None:
            try:
                self.complete_all(
                    _admission_token=_admission_token,
                    _admission_validator=_admission_validator,
                    _deadline=_deadline,
                )
            except BaseException as caught:
                if self._managed:
                    raise
                error = caught
        else:
            error = poisoned_error
            if self._managed:
                raise error
        _remaining_lifecycle_time(
            _deadline,
            "transfer scheduler lifecycle timed out before destructive close",
        )
        with self._collection_lock:
            self._tickets.clear()
            self._owners.clear()
            self._owner_detach_capabilities.clear()
            self._owner_detach_preparations.clear()
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
            elif error is None:
                error = self._poisoned_error
        if error is not None:
            raise error


__all__ = [
    "TransferProfile",
    "TransferScheduler",
    "TransferSource",
    "TransferTicket",
]
