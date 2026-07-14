"""Hard-bounded device tensor cache with exact lease reservations."""

from dataclasses import dataclass
from types import MappingProxyType
import weakref

import numpy as np

from renormalizer.backend._distributed.async_owner import (
    AsyncResourceOwner,
    _require_resource_admission,
    allocation_records,
    require_async_owner,
)
from renormalizer.backend._distributed.terminal import (
    _publish_lease_construction_resource,
    _publish_lease_construction_resource_direct,
    _remaining_lifecycle_time,
    _require_managed_resource_admission,
    _resolve_managed_resource_guard,
)


def _validate_identity(identity):
    if not isinstance(identity, tuple) or len(identity) != 9:
        raise TypeError("cache identity must be a canonical nine-field tuple")
    store_id, key, generation, version, rank, device, ranges, dtype, layout = identity
    if not isinstance(store_id, str) or not store_id:
        raise ValueError("cache identity store_id is invalid")
    if not isinstance(key, str) or not key:
        raise ValueError("cache identity key is invalid")
    if any(
        type(value) is not int or value < 0 for value in (generation, version, rank)
    ):
        raise ValueError("cache identity generation, version, and rank are invalid")
    if not isinstance(device, str) or not device:
        raise ValueError("cache identity device is invalid")
    if not isinstance(ranges, tuple) or any(
        not isinstance(value, tuple)
        or len(value) != 3
        or any(type(component) is not int for component in value)
        or value[0] < 0
        or value[1] <= value[0]
        or value[2] != 1
        for value in ranges
    ):
        raise ValueError("cache identity ranges are invalid")
    try:
        canonical_dtype = np.dtype(dtype).name
    except (TypeError, ValueError) as error:
        raise ValueError("cache identity dtype is invalid") from error
    if dtype != canonical_dtype:
        raise ValueError("cache identity dtype is noncanonical")
    if layout not in {"C", "F", "strided"}:
        raise ValueError("cache identity layout is invalid")
    return identity


@dataclass(frozen=True)
class CacheEntrySpec:
    identity: tuple
    shape: tuple[int, ...]
    dtype: str
    layout: str
    nbytes: int

    def __post_init__(self):
        _validate_identity(self.identity)
        try:
            shape = tuple(self.shape)
        except TypeError as error:
            raise TypeError("cache entry shape must be an iterable") from error
        if any(type(value) is not int or value < 0 for value in shape):
            raise ValueError("cache entry shape is invalid")
        try:
            dtype = np.dtype(self.dtype)
        except (TypeError, ValueError) as error:
            raise ValueError("cache entry dtype is invalid") from error
        if self.dtype != dtype.name or self.identity[7] != dtype.name:
            raise ValueError("cache entry dtype is noncanonical")
        if self.layout != self.identity[8] or self.layout not in {"C", "F", "strided"}:
            raise ValueError("cache entry layout is inconsistent")
        elements = 1
        for dimension in shape:
            elements *= dimension
        expected = elements * int(dtype.itemsize)
        if type(self.nbytes) is not int or self.nbytes != expected:
            raise ValueError("cache entry nbytes do not match shape and dtype")
        object.__setattr__(self, "shape", shape)


@dataclass(frozen=True)
class CacheAllocation:
    array: object
    transfer_array: object
    reverse_axis: int | None = None


class _CacheEntry:
    __slots__ = (
        "spec",
        "array",
        "transfer_array",
        "reverse_axis",
        "refcount",
        "state",
        "readiness_ticket",
        "error",
        "leases",
        "allocation_records",
    )

    def __init__(self, spec, allocation, records):
        self.spec = spec
        self.array = allocation.array
        self.transfer_array = allocation.transfer_array
        self.reverse_axis = allocation.reverse_axis
        self.refcount = 0
        self.state = "loading"
        self.readiness_ticket = None
        self.error = None
        self.leases = weakref.WeakSet()
        self.allocation_records = tuple(records)


class CacheEntryLease:
    """Reference to one cache entry, optionally released after an event."""

    def __init__(self, cache, entry, *, cache_hit):
        self._cache = cache
        self._managed = cache._managed
        self._managed_guard = cache._managed_guard
        self._entry = entry
        self.cache_hit = cache_hit
        self._failure = None
        self._closed = False
        self._owner = None
        entry.leases.add(self)

    def _require_entry(self):
        if self._entry is None:
            if self._failure is not None:
                raise RuntimeError("cache entry lease failed") from self._failure
            raise RuntimeError("cache entry lease is closed")
        return self._entry

    def _require_admission(self, token=None, validator=None):
        guard = getattr(self, "_managed_guard", None)
        if (
            guard is None
            and getattr(self, "_managed", False)
            and token is None
            and validator is None
        ):
            raise TypeError("managed cache entry lease requires admission")
        return _require_managed_resource_admission(
            guard,
            token,
            validator,
        )

    @property
    def identity(self):
        self._require_admission()
        return self._require_entry().spec.identity

    @property
    def state(self):
        self._require_admission()
        if self._entry is not None:
            return self._entry.state
        return "failed" if self._failure is not None else "closed"

    @property
    def array(self):
        self._require_admission()
        entry = self._require_entry()
        if entry.array is None:
            raise RuntimeError("cache entry lease failed") from entry.error
        return entry.array

    @property
    def transfer_array(self):
        self._require_admission()
        entry = self._require_entry()
        if entry.transfer_array is None:
            raise RuntimeError("cache entry lease failed") from entry.error
        return entry.transfer_array

    @property
    def reverse_axis(self):
        self._require_admission()
        return self._require_entry().reverse_axis

    def install_readiness(
        self,
        ticket,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(_admission_token, _admission_validator)
        entry = self._require_entry()
        self._cache._install_readiness(entry, ticket)

    def wait_for_ready(
        self,
        waiter,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(_admission_token, _admission_validator)
        entry = self._require_entry()
        self._cache._wait_for_ready(entry, waiter)

    def fail(
        self,
        error,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(_admission_token, _admission_validator)
        entry = self._require_entry()
        self._cache._fail_entry(entry, error)

    def _detach(self, failure=None):
        entry = self._entry
        if entry is not None:
            entry.leases.discard(self)
        self._entry = None
        self._cache = None
        self._failure = failure
        self._owner = None
        self._closed = True

    def transfer_to(
        self,
        owner,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(_admission_token, _admission_validator)
        if self._closed:
            return
        if not isinstance(owner, AsyncResourceOwner):
            raise TypeError("cache lease transfer requires an AsyncResourceOwner")
        if self._owner is not None:
            if self._owner is owner:
                return
            raise RuntimeError("cache entry lease already has an async owner")
        owner.capture_resources(self)
        self._owner = owner
        self._closed = True
        self._cache._release_to_owner(self._entry, self, owner)

    def close(
        self,
        event=None,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
        )
        if self._closed:
            return
        if event is not None:
            owner = require_async_owner(event, "cache lease transfer")
            self.transfer_to(
                owner,
                _admission_token=_admission_token,
                _admission_validator=_admission_validator,
            )
            return
        cache = self._cache
        entry = self._entry
        try:
            cache._release(entry)
        finally:
            self._detach()

    def __enter__(self):
        self._require_admission()
        if self._closed:
            raise RuntimeError("cache entry lease is closed")
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


class CacheReservation:
    """Exclusive exact allowlist installed for one outer working set."""

    def __init__(self, cache, allowlist, required_bytes, allocated_bytes):
        self._cache = cache
        self._managed = cache._managed
        self._managed_guard = cache._managed_guard
        self._allowlist = MappingProxyType(dict(allowlist))
        self.required_bytes = required_bytes
        self.peak_allocated_bytes = allocated_bytes
        self._closed = False

    @property
    def allowlist(self):
        guard = getattr(self, "_managed_guard", None)
        if guard is None and getattr(self, "_managed", False):
            raise TypeError("managed cache reservation requires admission")
        _require_managed_resource_admission(guard)
        return self._allowlist

    def close(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        guard = getattr(self, "_managed_guard", None)
        if (
            guard is None
            and getattr(self, "_managed", False)
            and _admission_token is None
            and _admission_validator is None
        ):
            raise TypeError("managed cache reservation requires admission")
        _require_managed_resource_admission(
            guard,
            _admission_token,
            _admission_validator,
        )
        if self._closed:
            return
        cache = self._cache
        if cache._managed:
            cache._release_reservation(self)
            self._cache = None
            self._allowlist = MappingProxyType({})
            self._closed = True
            return
        try:
            cache._release_reservation(self)
        finally:
            self._cache = None
            self._allowlist = MappingProxyType({})
            self._closed = True

    def __enter__(self):
        _require_managed_resource_admission(self._managed_guard)
        if self._closed:
            raise RuntimeError("cache reservation is closed")
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


class DeviceTensorCache:
    """Cache constrained by both a hard cap and an active exact allowlist."""

    _terminal_resource_kind = "cache"

    def __init__(
        self,
        capacity_bytes,
        *,
        allocator,
        _resource_recorder=None,
        _admission_token=None,
        _admission_validator=None,
        _construction_slot=None,
        _managed_guard=None,
        _standalone=True,
    ):
        _require_resource_admission(_admission_token, _admission_validator)
        self._managed_guard = _resolve_managed_resource_guard(
            standalone=_standalone,
            guard=_managed_guard,
        )
        if type(capacity_bytes) is not int or capacity_bytes < 0:
            raise ValueError("capacity_bytes must be a non-negative integer")
        if not callable(allocator):
            raise TypeError("allocator must be callable")
        if _resource_recorder is not None and not callable(_resource_recorder):
            raise TypeError("resource recorder must be callable")
        self._capacity_bytes = capacity_bytes
        self._allocator = allocator
        self._entries = {}
        self._owner_pending = []
        self._reservation = None
        self._allocated_bytes = 0
        self._allocation_refcounts = {}
        self._peak_allocated_bytes = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._poisoned_error = None
        self._closed = False
        self._managed = self._managed_guard is not None
        self._resource_recorder = _resource_recorder
        _publish_lease_construction_resource_direct(
            _construction_slot,
            self,
            kind="cache",
        )
        _publish_lease_construction_resource(
            _construction_slot,
            self,
            kind="cache",
        )
        if self._resource_recorder is not None:
            self._resource_recorder(
                resource=self,
                kind="cache",
                records=(),
                _replace_kind=True,
            )

    def _set_resource_recorder(
        self,
        recorder,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(_admission_token, _admission_validator)
        if recorder is not None and not callable(recorder):
            raise TypeError("resource recorder must be callable")
        self._resource_recorder = recorder
        self._record_allocation_snapshot()

    def _record_allocation_snapshot(self):
        if self._resource_recorder is not None:
            self._resource_recorder(
                resource=self,
                kind="cache",
                records=self.allocation_records,
                _replace_kind=True,
            )

    def _terminal_allocation_records(self):
        return tuple(retained[0] for retained in self._allocation_refcounts.values())

    @property
    def capacity_bytes(self):
        self._require_admission(None, None)
        return self._capacity_bytes

    @property
    def allocated_bytes(self):
        self._require_admission(None, None)
        return self._allocated_bytes

    @property
    def allocation_records(self):
        self._require_admission(None, None)
        return self._terminal_allocation_records()

    @property
    def peak_allocated_bytes(self):
        self._require_admission(None, None)
        return self._peak_allocated_bytes

    @property
    def reserved_bytes(self):
        self._require_admission(None, None)
        return 0 if self._reservation is None else self._reservation.required_bytes

    @property
    def cache_hits(self):
        self._require_admission(None, None)
        return self._cache_hits

    @property
    def cache_misses(self):
        self._require_admission(None, None)
        return self._cache_misses

    @property
    def entry_count(self):
        self._require_admission(None, None)
        return len(self._entries)

    @property
    def pending_release_count(self):
        self._require_admission(None, None)
        return len(self._owner_pending)

    @property
    def poisoned(self):
        self._require_admission(None, None)
        return self._poisoned_error is not None

    def _require_usable(self):
        if self._poisoned_error is not None:
            raise RuntimeError("device tensor cache is terminal-poisoned") from (
                self._poisoned_error
            )
        if self._closed:
            raise RuntimeError("device tensor cache is closed")

    def _require_admission(self, token, validator):
        guard = getattr(self, "_managed_guard", None)
        if (
            guard is None
            and getattr(self, "_managed", False)
            and token is None
            and validator is None
        ):
            raise TypeError("managed device tensor cache requires admission")
        return _require_managed_resource_admission(
            guard,
            token,
            validator,
        )

    def _poison(self, error):
        if self._poisoned_error is None:
            self._poisoned_error = error

    def reserve(
        self,
        allowlist,
        required_bytes,
        *,
        _resource_recorder=None,
        _admission_token=None,
        _admission_validator=None,
        _construction_slot=None,
    ):
        self._require_admission(_admission_token, _admission_validator)
        if _resource_recorder is not None and not callable(_resource_recorder):
            raise TypeError("resource recorder must be callable")
        self._require_usable()
        if self._reservation is not None:
            raise RuntimeError("device tensor cache already has an active reservation")
        if type(required_bytes) is not int or required_bytes < 0:
            raise ValueError("required_bytes must be a non-negative integer")
        try:
            items = tuple(allowlist.items())
        except AttributeError as error:
            raise TypeError(
                "allowlist must be a mapping of identities to specs"
            ) from error
        normalized = {}
        for identity, spec in items:
            _validate_identity(identity)
            if not isinstance(spec, CacheEntrySpec) or spec.identity != identity:
                raise ValueError(
                    "allowlist entries must match CacheEntrySpec identities"
                )
            if identity in normalized:
                raise ValueError("allowlist identities must be unique")
            normalized[identity] = spec
        expected = sum(spec.nbytes for spec in normalized.values())
        if required_bytes != expected:
            raise ValueError("required_bytes do not match the exact cache allowlist")
        if required_bytes > self._capacity_bytes:
            raise MemoryError("cache reservation exceeds fixed cache capacity")

        self.reap_completed(
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
        )
        outside = [
            (identity, entry)
            for identity, entry in self._entries.items()
            if identity not in normalized
        ]
        if any(entry.refcount or entry.state == "loading" for _, entry in outside):
            raise RuntimeError("an unauthorized cache entry is still in flight")
        for identity, entry in outside:
            self._evict(identity, entry)
        for identity, entry in self._entries.items():
            if entry.spec != normalized[identity]:
                raise ValueError(
                    "retained cache entry does not match its allowlist spec"
                )
        if self._allocated_bytes > required_bytes:
            raise MemoryError("cache physical allocation exceeds reservation capacity")

        reservation = CacheReservation(
            self, normalized, required_bytes, self._allocated_bytes
        )
        self._reservation = reservation
        _publish_lease_construction_resource_direct(
            _construction_slot,
            reservation,
        )
        _publish_lease_construction_resource(
            _construction_slot,
            reservation,
        )
        if _resource_recorder is not None:
            _resource_recorder(resource=reservation)
        return reservation

    def _release_reservation(self, reservation):
        if self._reservation is not reservation:
            raise RuntimeError("cache reservation is not active")
        self._reservation = None

    def _validate_array(self, array, spec, *, transfer=False):
        if tuple(array.shape) != spec.shape:
            raise ValueError("cache allocator returned the wrong shape")
        if np.dtype(array.dtype).name != spec.dtype:
            raise ValueError("cache allocator returned the wrong dtype")
        if int(array.nbytes) != spec.nbytes:
            raise ValueError("cache allocator returned the wrong nbytes")
        expected_layout = "C" if transfer and spec.layout == "strided" else spec.layout
        if expected_layout == "C" and not bool(array.flags.c_contiguous):
            raise ValueError("cache allocator returned a non-C-contiguous array")
        if expected_layout == "F" and not bool(array.flags.f_contiguous):
            raise ValueError("cache allocator returned a non-F-contiguous array")
        if expected_layout == "strided" and (
            bool(array.flags.c_contiguous) or bool(array.flags.f_contiguous)
        ):
            raise ValueError("cache allocator returned a contiguous strided entry")

    def _register_allocations(self, records):
        additional = 0
        for record in records:
            retained = self._allocation_refcounts.get(record.identity)
            if retained is None:
                additional += record.capacity_bytes
            elif record.capacity_bytes > retained[0].capacity_bytes:
                additional += record.capacity_bytes - retained[0].capacity_bytes
        if self._allocated_bytes + additional > self._capacity_bytes:
            raise MemoryError("cache physical allocation exceeds fixed cache capacity")
        reservation = self._reservation
        if (
            reservation is None
            or self._allocated_bytes + additional > reservation.required_bytes
        ):
            raise MemoryError("cache physical allocation exceeds reservation capacity")
        for record in records:
            retained = self._allocation_refcounts.get(record.identity)
            if retained is None:
                self._allocation_refcounts[record.identity] = [record, 1]
                continue
            if record.capacity_bytes > retained[0].capacity_bytes:
                retained[0] = record
            retained[1] += 1
        self._allocated_bytes += additional
        self._record_allocation_snapshot()

    def _release_allocations(self, entry):
        for record in entry.allocation_records:
            retained = self._allocation_refcounts.get(record.identity)
            if retained is None or retained[1] <= 0:
                raise RuntimeError("cache physical allocation count is inconsistent")
            retained[1] -= 1
            if retained[1] == 0:
                self._allocated_bytes -= retained[0].capacity_bytes
                del self._allocation_refcounts[record.identity]
        entry.allocation_records = ()
        self._record_allocation_snapshot()

    def acquire(
        self,
        identity,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(_admission_token, _admission_validator)
        self._require_usable()
        _validate_identity(identity)
        reservation = self._reservation
        if reservation is None or identity not in reservation.allowlist:
            raise ValueError("cache identity is outside the active allowlist")
        self.reap_completed(
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
        )
        entry = self._entries.get(identity)
        cache_hit = entry is not None
        if (
            entry is not None
            and entry.state == "loading"
            and entry.readiness_ticket is None
        ):
            raise RuntimeError("cache entry load has no readiness metadata")
        if entry is None:
            spec = reservation.allowlist[identity]
            allocated = self._allocator(spec)
            if isinstance(allocated, CacheAllocation):
                allocation = allocated
            else:
                allocation = CacheAllocation(allocated, allocated)
            self._validate_array(allocation.array, spec)
            self._validate_array(allocation.transfer_array, spec, transfer=True)
            if spec.layout == "strided":
                if type(allocation.reverse_axis) is not int:
                    raise ValueError("strided cache allocation requires a reverse axis")
                if allocation.reverse_axis < 0 or allocation.reverse_axis >= len(
                    spec.shape
                ):
                    raise ValueError("strided cache reverse axis is out of range")
            elif allocation.reverse_axis is not None:
                raise ValueError("contiguous cache allocation has a reverse axis")
            records = allocation_records((allocation.array, allocation.transfer_array))
            self._register_allocations(records)
            entry = _CacheEntry(spec, allocation, records)
            self._entries[identity] = entry
            self._peak_allocated_bytes = max(
                self._peak_allocated_bytes, self._allocated_bytes
            )
            reservation.peak_allocated_bytes = max(
                reservation.peak_allocated_bytes, self._allocated_bytes
            )
            self._cache_misses += 1
        else:
            self._cache_hits += 1
        entry.refcount += 1
        return CacheEntryLease(self, entry, cache_hit=cache_hit)

    def _refresh_entry(self, entry):
        if entry.state != "loading" or entry.readiness_ticket is None:
            return
        ticket = entry.readiness_ticket
        try:
            if getattr(ticket, "terminal_poisoned", False):
                raise ticket.error
            if ticket.completed:
                if ticket.error is not None:
                    raise ticket.error
                completed = True
            else:
                completed = bool(ticket.reap())
        except BaseException as error:
            if getattr(ticket, "terminal_poisoned", False):
                entry.state = "failed"
                entry.error = error
                self._poison(error)
            else:
                self._fail_entry(entry, error)
            raise
        if completed:
            entry.state = "ready"
            entry.readiness_ticket = None

    def _install_readiness(self, entry, ticket):
        if entry.state != "loading" or entry.readiness_ticket is not None:
            raise RuntimeError("cache entry readiness is already installed")
        if self._entries.get(entry.spec.identity) is not entry:
            raise RuntimeError("cache entry is not retained")
        if (
            not callable(getattr(ticket, "reap", None))
            or not callable(getattr(ticket, "wait", None))
            or getattr(ticket, "event", None) is None
        ):
            raise TypeError("cache readiness requires a transfer ticket")
        entry.readiness_ticket = ticket
        publish = getattr(ticket, "_publish_completion", None)
        if callable(publish):
            publish()
        self._refresh_entry(entry)

    def _wait_for_ready(self, entry, waiter):
        if not callable(waiter):
            raise TypeError("readiness waiter must be callable")
        self._refresh_entry(entry)
        if entry.state == "ready":
            return
        if entry.state == "failed":
            raise RuntimeError("cache entry load failed") from entry.error
        ticket = entry.readiness_ticket
        if ticket is None or ticket.event is None:
            raise RuntimeError("cache entry load has no readiness metadata")
        waiter(ticket.event)

    def _fail_entry(self, entry, error):
        if not isinstance(error, BaseException):
            raise TypeError("cache entry failure must be an exception")
        identity = entry.spec.identity
        if self._entries.get(identity) is entry:
            del self._entries[identity]
            self._release_allocations(entry)
        entry.state = "failed"
        entry.error = error
        entry.readiness_ticket = None
        entry.array = None
        entry.transfer_array = None
        entry.reverse_axis = None
        entry.refcount = 0
        for lease in tuple(entry.leases):
            lease._detach(error)
        entry.leases.clear()

    def _release(self, entry):
        if entry.state == "failed":
            return
        if entry.refcount <= 0:
            raise RuntimeError("cache entry reference count is inconsistent")
        entry.refcount -= 1

    def _release_to_owner(self, entry, lease, owner):
        retained = (owner, entry, lease)
        self._owner_pending.append(retained)

        def release():
            if lease._owner is not owner:
                return
            if entry.state != "failed":
                if entry.refcount <= 0:
                    raise RuntimeError("cache entry reference count is inconsistent")
                entry.refcount -= 1
            lease._detach()
            try:
                self._owner_pending.remove(retained)
            except ValueError:
                pass

        owner.add_release_callback(release)

    def reap_completed(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
        _deadline=None,
    ):
        self._require_admission(_admission_token, _admission_validator)
        _remaining_lifecycle_time(
            _deadline,
            "device cache lifecycle timed out before reap",
        )
        if self._closed and not self.poisoned:
            return
        self._require_usable()
        errors = []
        for owner in tuple(
            dict.fromkeys(retained[0] for retained in self._owner_pending)
        ):
            try:
                owner.reap()
            except BaseException as error:
                if owner.quarantined:
                    self._poison(error)
                if self._managed:
                    raise
                errors.append(error)
        for entry in tuple(self._entries.values()):
            try:
                self._refresh_entry(entry)
            except BaseException as error:
                if self._managed:
                    raise
                errors.append(error)
        if errors:
            raise errors[0]

    def wait_for_pending(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
        _deadline=None,
    ):
        self._require_admission(_admission_token, _admission_validator)
        _remaining_lifecycle_time(
            _deadline,
            "device cache lifecycle timed out before pending wait",
        )
        errors = []
        for owner in tuple(
            dict.fromkeys(retained[0] for retained in self._owner_pending)
        ):
            try:
                owner.wait(_deadline=_deadline)
            except BaseException as error:
                if owner.quarantined:
                    self._poison(error)
                if self._managed:
                    raise
                errors.append(error)
        for entry in tuple(self._entries.values()):
            ticket = entry.readiness_ticket
            if entry.state == "loading" and ticket is not None:
                try:
                    ticket.wait(_deadline=_deadline)
                    self._refresh_entry(entry)
                except BaseException as error:
                    if getattr(ticket, "terminal_poisoned", False):
                        entry.state = "failed"
                        entry.error = error
                        self._poison(error)
                    else:
                        self._fail_entry(entry, error)
                    if self._managed:
                        raise
                    errors.append(error)
        try:
            self.reap_completed(
                _admission_token=_admission_token,
                _admission_validator=_admission_validator,
                _deadline=_deadline,
            )
        except BaseException as error:
            if self._managed:
                raise
            errors.append(error)
        if errors:
            raise errors[0]

    def contains(
        self,
        identity,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(_admission_token, _admission_validator)
        return identity in self._entries

    def refcount(
        self,
        identity,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(_admission_token, _admission_validator)
        if isinstance(identity, str):
            return sum(
                entry.refcount
                for key, entry in self._entries.items()
                if key[1] == identity
            )
        entry = self._entries.get(identity)
        return 0 if entry is None else entry.refcount

    def _evict(self, identity, entry):
        self._refresh_entry(entry)
        if entry.state == "loading":
            raise RuntimeError("cache entry load is still in flight")
        if entry.refcount:
            raise RuntimeError("cache entry is still referenced")
        del self._entries[identity]
        self._release_allocations(entry)
        entry.array = None
        entry.transfer_array = None
        entry.reverse_axis = None

    def invalidate(
        self,
        identity,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(_admission_token, _admission_validator)
        entry = self._entries.get(identity)
        if entry is None:
            return
        self.reap_completed(
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
        )
        if entry.refcount:
            raise RuntimeError("cache entry is still referenced")
        self._evict(identity, entry)

    def invalidate_ref(
        self,
        ref,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        self._require_admission(_admission_token, _admission_validator)
        identities = [
            identity
            for identity in self._entries
            if identity[:4] == (ref.store_id, ref.key, ref.generation, ref.version)
        ]
        for identity in identities:
            self.invalidate(
                identity,
                _admission_token=_admission_token,
                _admission_validator=_admission_validator,
            )

    def close(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
        _deadline=None,
    ):
        self._require_admission(_admission_token, _admission_validator)
        _remaining_lifecycle_time(
            _deadline,
            "device cache lifecycle timed out before close",
        )
        if self._closed:
            return
        error = self._poisoned_error
        if self._poisoned_error is None:
            try:
                self.wait_for_pending(
                    _admission_token=_admission_token,
                    _admission_validator=_admission_validator,
                    _deadline=_deadline,
                )
            except BaseException as caught:
                if self._managed:
                    raise
                if error is None:
                    error = caught
        if self._reservation is not None and error is None:
            try:
                self._reservation.close(
                    _admission_token=_admission_token,
                    _admission_validator=_admission_validator,
                )
            except BaseException as caught:
                if self._managed:
                    raise
                error = caught
        if self._poisoned_error is not None:
            if self._managed:
                raise self._poisoned_error
            self._reservation = None
            self._allocator = None
            self._closed = True
            if error is not None:
                raise error
            return
        if any(entry.refcount for entry in self._entries.values()) and error is None:
            error = RuntimeError("cannot close cache with live entry leases")
            if self._managed:
                raise error
        _remaining_lifecycle_time(
            _deadline,
            "device cache lifecycle timed out before destructive close",
        )
        close_error = RuntimeError("device tensor cache closed with a live entry")
        for entry in tuple(self._entries.values()):
            entry.state = "failed"
            entry.error = close_error
            entry.readiness_ticket = None
            entry.array = None
            entry.transfer_array = None
            entry.reverse_axis = None
            self._release_allocations(entry)
            entry.refcount = 0
            for lease in tuple(entry.leases):
                lease._detach(close_error)
            entry.leases.clear()
        self._entries.clear()
        self._allocated_bytes = 0
        self._allocation_refcounts.clear()
        self._owner_pending.clear()
        self._reservation = None
        self._allocator = None
        self._resource_recorder = None
        self._closed = True
        if error is not None:
            raise error


__all__ = [
    "CacheAllocation",
    "CacheEntryLease",
    "CacheEntrySpec",
    "CacheReservation",
    "DeviceTensorCache",
]
