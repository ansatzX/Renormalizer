"""Hard-bounded device tensor cache with exact lease reservations."""

from dataclasses import dataclass
from types import MappingProxyType
import weakref

import numpy as np

from renormalizer.backend._distributed.async_owner import (
    AsyncResourceOwner,
    allocation_records,
    require_async_owner,
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

    @property
    def identity(self):
        return self._require_entry().spec.identity

    @property
    def state(self):
        if self._entry is not None:
            return self._entry.state
        return "failed" if self._failure is not None else "closed"

    @property
    def array(self):
        entry = self._require_entry()
        if entry.array is None:
            raise RuntimeError("cache entry lease failed") from entry.error
        return entry.array

    @property
    def transfer_array(self):
        entry = self._require_entry()
        if entry.transfer_array is None:
            raise RuntimeError("cache entry lease failed") from entry.error
        return entry.transfer_array

    @property
    def reverse_axis(self):
        return self._require_entry().reverse_axis

    def install_readiness(self, ticket):
        entry = self._require_entry()
        self._cache._install_readiness(entry, ticket)

    def wait_for_ready(self, waiter):
        entry = self._require_entry()
        self._cache._wait_for_ready(entry, waiter)

    def fail(self, error):
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

    def transfer_to(self, owner):
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

    def close(self, event=None):
        if self._closed:
            return
        if event is not None:
            owner = require_async_owner(event, "cache lease transfer")
            self.transfer_to(owner)
            return
        cache = self._cache
        entry = self._entry
        try:
            cache._release(entry)
        finally:
            self._detach()

    def __enter__(self):
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
        self._allowlist = MappingProxyType(dict(allowlist))
        self.required_bytes = required_bytes
        self.peak_allocated_bytes = allocated_bytes
        self._closed = False

    @property
    def allowlist(self):
        return self._allowlist

    def close(self):
        if self._closed:
            return
        cache = self._cache
        try:
            cache._release_reservation(self)
        finally:
            self._cache = None
            self._allowlist = MappingProxyType({})
            self._closed = True

    def __enter__(self):
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

    def __init__(self, capacity_bytes, *, allocator):
        if type(capacity_bytes) is not int or capacity_bytes < 0:
            raise ValueError("capacity_bytes must be a non-negative integer")
        if not callable(allocator):
            raise TypeError("allocator must be callable")
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

    @property
    def capacity_bytes(self):
        return self._capacity_bytes

    @property
    def allocated_bytes(self):
        return self._allocated_bytes

    @property
    def allocation_records(self):
        return tuple(retained[0] for retained in self._allocation_refcounts.values())

    @property
    def peak_allocated_bytes(self):
        return self._peak_allocated_bytes

    @property
    def reserved_bytes(self):
        return 0 if self._reservation is None else self._reservation.required_bytes

    @property
    def cache_hits(self):
        return self._cache_hits

    @property
    def cache_misses(self):
        return self._cache_misses

    @property
    def entry_count(self):
        return len(self._entries)

    @property
    def pending_release_count(self):
        return len(self._owner_pending)

    @property
    def poisoned(self):
        return self._poisoned_error is not None

    def _require_usable(self):
        if self._poisoned_error is not None:
            raise RuntimeError("device tensor cache is terminal-poisoned") from (
                self._poisoned_error
            )
        if self._closed:
            raise RuntimeError("device tensor cache is closed")

    def _poison(self, error):
        if self._poisoned_error is None:
            self._poisoned_error = error

    def reserve(self, allowlist, required_bytes):
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

        self.reap_completed()
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

    def acquire(self, identity):
        self._require_usable()
        _validate_identity(identity)
        reservation = self._reservation
        if reservation is None or identity not in reservation.allowlist:
            raise ValueError("cache identity is outside the active allowlist")
        self.reap_completed()
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

    def reap_completed(self):
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
                errors.append(error)
        for entry in tuple(self._entries.values()):
            try:
                self._refresh_entry(entry)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]

    def wait_for_pending(self):
        errors = []
        for owner in tuple(
            dict.fromkeys(retained[0] for retained in self._owner_pending)
        ):
            try:
                owner.wait()
            except BaseException as error:
                if owner.quarantined:
                    self._poison(error)
                errors.append(error)
        for entry in tuple(self._entries.values()):
            ticket = entry.readiness_ticket
            if entry.state == "loading" and ticket is not None:
                try:
                    ticket.wait()
                    self._refresh_entry(entry)
                except BaseException as error:
                    if getattr(ticket, "terminal_poisoned", False):
                        entry.state = "failed"
                        entry.error = error
                        self._poison(error)
                    else:
                        self._fail_entry(entry, error)
                    errors.append(error)
        try:
            self.reap_completed()
        except BaseException as error:
            errors.append(error)
        if errors:
            raise errors[0]

    def contains(self, identity):
        return identity in self._entries

    def refcount(self, identity):
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

    def invalidate(self, identity):
        entry = self._entries.get(identity)
        if entry is None:
            return
        self.reap_completed()
        if entry.refcount:
            raise RuntimeError("cache entry is still referenced")
        self._evict(identity, entry)

    def invalidate_ref(self, ref):
        identities = [
            identity
            for identity in self._entries
            if identity[:4] == (ref.store_id, ref.key, ref.generation, ref.version)
        ]
        for identity in identities:
            self.invalidate(identity)

    def close(self):
        if self._closed:
            return
        error = self._poisoned_error
        if self._reservation is not None:
            try:
                self._reservation.close()
            except BaseException as caught:
                if error is None:
                    error = caught
        if self._poisoned_error is None:
            try:
                self.wait_for_pending()
            except BaseException as caught:
                if error is None:
                    error = caught
        if self._poisoned_error is not None:
            self._reservation = None
            self._allocator = None
            self._closed = True
            if error is not None:
                raise error
            return
        if any(entry.refcount for entry in self._entries.values()) and error is None:
            error = RuntimeError("cannot close cache with live entry leases")
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
