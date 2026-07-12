import gc
import weakref

import numpy as np
import pytest

from renormalizer.backend._distributed.async_owner import AsyncResourceOwner
from renormalizer.backend._distributed.cache import (
    CacheEntrySpec,
    DeviceTensorCache,
)
from renormalizer.backend._distributed.transfer import TransferTicket


class _ManualEvent:
    def __init__(self):
        self.done = False

    def query(self):
        return self.done

    def synchronize(self):
        self.done = True

    def complete(self):
        self.done = True


class _QueryFailureEvent(_ManualEvent):
    def __init__(self, *, synchronize_fails=False):
        super().__init__()
        self.synchronize_calls = 0
        self.synchronize_fails = synchronize_fails

    def query(self):
        raise RuntimeError("injected cache query failure")

    def synchronize(self):
        self.synchronize_calls += 1
        if self.synchronize_fails:
            raise RuntimeError("injected cache drain failure")
        self.done = True


def _event_owner(event, *, kind="h2d", arrays=(), nbytes=0):
    owner = AsyncResourceOwner(kind, arrays=arrays, nbytes=nbytes)
    owner.add_event(event, completion=True)
    owner.mark_enqueued()
    return owner


def _identity(key, *, version=0, start=0, stop=4):
    return (
        "store-a",
        key,
        0,
        version,
        0,
        "cpu",
        ((start, stop, 1),),
        "float64",
        "C",
    )


def _spec(key, *, version=0, start=0, stop=4):
    identity = _identity(key, version=version, start=start, stop=stop)
    return CacheEntrySpec(
        identity=identity,
        shape=(stop - start,),
        dtype="float64",
        layout="C",
        nbytes=(stop - start) * 8,
    )


def _cache(capacity=64):
    return DeviceTensorCache(
        capacity,
        allocator=lambda spec: np.empty(
            spec.shape, dtype=np.dtype(spec.dtype), order=spec.layout
        ),
    )


def _complete_load(lease):
    event = _ManualEvent()
    event.complete()
    owner = _event_owner(event, arrays=(lease.array,), nbytes=int(lease.array.nbytes))
    lease.install_readiness(TransferTicket(owner))


def test_reservation_is_an_exact_allowlist_and_hard_byte_bound():
    cache = _cache(63)
    left = _spec("left")
    right = _spec("right")

    with pytest.raises(ValueError, match="required_bytes"):
        cache.reserve({left.identity: left}, 31)
    with pytest.raises(MemoryError, match="capacity"):
        cache.reserve({left.identity: left, right.identity: right}, 64)

    reservation = cache.reserve({left.identity: left}, 32)
    with pytest.raises(ValueError, match="allowlist"):
        cache.acquire(right.identity)
    assert cache.reserved_bytes == 32
    reservation.close()
    reservation.close()
    assert cache.reserved_bytes == 0
    cache.close()


def test_cache_rejects_logical_view_with_oversized_physical_backing():
    backing = np.empty(8, dtype=np.float64)
    spec = _spec("oversized")
    cache = DeviceTensorCache(32, allocator=lambda _spec: backing[:4])
    reservation = cache.reserve({spec.identity: spec}, spec.nbytes)

    with pytest.raises(MemoryError, match="physical.*capacity"):
        cache.acquire(spec.identity)

    assert cache.allocated_bytes == 0
    assert cache.entry_count == 0
    reservation.close()
    cache.close()


def test_cache_rejects_physical_backing_above_exact_reservation_below_global_cap():
    backing = np.empty(8, dtype=np.float64)
    spec = _spec("reservation-oversized")
    cache = DeviceTensorCache(128, allocator=lambda _spec: backing[:4])
    reservation = cache.reserve({spec.identity: spec}, spec.nbytes)

    with pytest.raises(MemoryError, match="physical.*reservation"):
        cache.acquire(spec.identity)

    assert cache.reserved_bytes == spec.nbytes
    assert cache.allocated_bytes == 0
    assert cache.entry_count == 0
    reservation.close()
    cache.close()


def test_cache_shared_backing_is_charged_once_until_final_alias_eviction():
    backing = np.empty(8, dtype=np.float64)
    left = _spec("left")
    right = _spec("right")
    arrays = iter((backing[:4], backing[4:]))
    cache = DeviceTensorCache(64, allocator=lambda _spec: next(arrays))
    reservation = cache.reserve(
        {left.identity: left, right.identity: right}, left.nbytes + right.nbytes
    )
    left_lease = cache.acquire(left.identity)
    right_lease = cache.acquire(right.identity)
    _complete_load(left_lease)
    _complete_load(right_lease)
    left_lease.close()
    right_lease.close()

    assert cache.allocated_bytes == backing.nbytes
    cache.invalidate(left.identity)
    assert cache.allocated_bytes == backing.nbytes
    cache.invalidate(right.identity)
    assert cache.allocated_bytes == 0
    reservation.close()
    cache.close()


def test_reservation_copies_allowlist_into_a_private_immutable_mapping():
    cache = _cache()
    spec = _spec("left")
    supplied = {spec.identity: spec}
    reservation = cache.reserve(supplied, spec.nbytes)

    supplied.clear()
    assert tuple(reservation.allowlist) == (spec.identity,)
    with pytest.raises(TypeError):
        reservation.allowlist[spec.identity] = spec

    reservation.close()
    cache.close()


def test_loading_entry_waits_only_when_consumed_and_becomes_ready_on_completion():
    cache = _cache()
    spec = _spec("left")
    reservation = cache.reserve({spec.identity: spec}, spec.nbytes)
    event = _ManualEvent()
    ticket = TransferTicket(_event_owner(event, nbytes=spec.nbytes))

    producer = cache.acquire(spec.identity)
    assert producer.state == "loading"
    producer.install_readiness(ticket)
    producer.close()

    consumer = cache.acquire(spec.identity)
    waits = []
    assert consumer.cache_hit is True
    assert consumer.state == "loading"
    consumer.wait_for_ready(waits.append)
    assert waits == [event]
    assert event.done is False

    event.complete()
    cache.reap_completed()
    assert consumer.state == "ready"
    consumer.close()
    reservation.close()
    cache.close()


def test_failed_loading_entry_is_evicted_and_detaches_device_array():
    arrays = []

    def allocate(spec):
        array = np.empty(spec.shape, dtype=np.dtype(spec.dtype))
        arrays.append(array)
        return array

    cache = DeviceTensorCache(64, allocator=allocate)
    spec = _spec("left")
    reservation = cache.reserve({spec.identity: spec}, spec.nbytes)
    lease = cache.acquire(spec.identity)
    array_ref = weakref.ref(lease.array)

    lease.fail(RuntimeError("injected H2D failure"))
    arrays.clear()
    gc.collect()

    assert cache.contains(spec.identity) is False
    assert cache.allocated_bytes == 0
    assert array_ref() is None
    with pytest.raises(RuntimeError, match="failed"):
        lease.array
    reservation.close()
    cache.close()


def test_readiness_wait_without_successful_owner_drain_retains_terminal_entry():
    class FailingEvent(_ManualEvent):
        def synchronize(self):
            raise RuntimeError("injected asynchronous H2D failure")

    cache = _cache()
    spec = _spec("left")
    reservation = cache.reserve({spec.identity: spec}, spec.nbytes)
    event = FailingEvent()
    ticket = TransferTicket(_event_owner(event, nbytes=spec.nbytes))
    lease = cache.acquire(spec.identity)
    lease.install_readiness(ticket)
    lease.close()

    with pytest.raises(RuntimeError, match="asynchronous H2D failure"):
        ticket.wait()
    with pytest.raises(RuntimeError, match="asynchronous H2D failure"):
        cache.reap_completed()

    assert ticket.terminal_poisoned is True
    assert cache.contains(spec.identity) is True
    assert cache.allocated_bytes == spec.nbytes
    assert cache.poisoned is True
    reservation.close()
    with pytest.raises(RuntimeError, match="asynchronous H2D failure"):
        cache.close()
    assert cache.allocated_bytes == spec.nbytes


def test_repeated_acquire_reuses_entry_and_counts_hit_without_growth():
    cache = _cache()
    spec = _spec("left")
    reservation = cache.reserve({spec.identity: spec}, spec.nbytes)

    first = cache.acquire(spec.identity)
    first_ptr = first.array.__array_interface__["data"][0]
    assert first.cache_hit is False
    _complete_load(first)
    first.close()
    second = cache.acquire(spec.identity)
    assert second.cache_hit is True
    assert second.array.__array_interface__["data"][0] == first_ptr
    second.close()

    assert cache.allocated_bytes == spec.nbytes
    assert cache.cache_hits == 1
    assert cache.cache_misses == 1
    reservation.close()
    cache.close()


def test_entry_ref_survives_event_and_blocks_unauthorized_replacement():
    cache = _cache(32)
    left = _spec("left")
    right = _spec("right")
    reservation = cache.reserve({left.identity: left}, left.nbytes)
    event = _ManualEvent()
    lease = cache.acquire(left.identity)
    _complete_load(lease)
    lease.close(_event_owner(event, kind="compute"))
    reservation.close()

    assert cache.refcount(left.identity) == 1
    with pytest.raises(RuntimeError, match="in flight"):
        cache.reserve({right.identity: right}, right.nbytes)
    assert cache.allocated_bytes == left.nbytes

    event.complete()
    cache.reap_completed()
    replacement = cache.reserve({right.identity: right}, right.nbytes)
    assert cache.contains(left.identity) is False
    right_lease = cache.acquire(right.identity)
    _complete_load(right_lease)
    right_lease.close()
    replacement.close()
    cache.close()


def test_cache_release_rejects_raw_event_without_registered_owner():
    cache = _cache(32)
    spec = _spec("left")
    reservation = cache.reserve({spec.identity: spec}, spec.nbytes)
    lease = cache.acquire(spec.identity)

    with pytest.raises(TypeError, match="pre-existing AsyncResourceOwner"):
        lease.close(_ManualEvent())

    lease.close()
    reservation.close()
    cache.close()


def test_cache_query_failure_drains_before_releasing_entry_reference():
    cache = _cache(32)
    spec = _spec("left")
    reservation = cache.reserve({spec.identity: spec}, spec.nbytes)
    lease = cache.acquire(spec.identity)
    _complete_load(lease)
    event = _QueryFailureEvent()
    owner = _event_owner(event, kind="compute")

    lease.close(owner)
    with pytest.raises(RuntimeError, match="cache query failure"):
        cache.reap_completed()

    assert event.synchronize_calls == 1
    assert cache.refcount(spec.identity) == 0
    assert cache.allocated_bytes == spec.nbytes
    assert cache.poisoned is False
    cache.invalidate(spec.identity)
    reservation.close()
    cache.close()


def test_cache_query_and_drain_failure_retains_entry_and_terminal_poison():
    cache = _cache(32)
    spec = _spec("left")
    reservation = cache.reserve({spec.identity: spec}, spec.nbytes)
    lease = cache.acquire(spec.identity)
    _complete_load(lease)
    array_ref = weakref.ref(lease.array)
    event = _QueryFailureEvent(synchronize_fails=True)
    owner = _event_owner(event, kind="compute")

    lease.close(owner)
    with pytest.raises(RuntimeError, match="cache query failure"):
        cache.reap_completed()

    assert event.synchronize_calls == 1
    assert cache.refcount(spec.identity) == 1
    assert cache.allocated_bytes == spec.nbytes
    assert cache.contains(spec.identity) is True
    assert cache.poisoned is True
    assert array_ref() is not None
    with pytest.raises(RuntimeError, match="terminal-poisoned"):
        cache.acquire(spec.identity)
    reservation.close()
    with pytest.raises(RuntimeError, match="cache query failure"):
        cache.close()
    assert cache.allocated_bytes == spec.nbytes
    assert array_ref() is not None


def test_old_version_is_invalidated_only_after_references_complete():
    cache = _cache()
    old = _spec("center", version=0)
    current = _spec("center", version=1)
    reservation = cache.reserve(
        {old.identity: old, current.identity: current},
        old.nbytes + current.nbytes,
    )
    event = _ManualEvent()
    lease = cache.acquire(old.identity)
    _complete_load(lease)
    lease.close(_event_owner(event, kind="compute"))

    with pytest.raises(RuntimeError, match="referenced"):
        cache.invalidate(old.identity)
    event.complete()
    cache.reap_completed()
    cache.invalidate(old.identity)
    assert cache.contains(old.identity) is False
    reservation.close()
    cache.close()


def test_close_is_idempotent_and_waits_for_pending_releases():
    cache = _cache()
    spec = _spec("left")
    reservation = cache.reserve({spec.identity: spec}, spec.nbytes)
    event = _ManualEvent()
    lease = cache.acquire(spec.identity)
    _complete_load(lease)
    lease.close(_event_owner(event, kind="compute"))
    reservation.close()

    cache.close()
    cache.close()

    assert event.done is True
    assert cache.allocated_bytes == 0


def test_close_preserves_first_terminal_error_identity_over_cleanup_failure():
    cache = _cache()
    first = RuntimeError("first terminal cache failure")
    later = RuntimeError("later reservation cleanup failure")

    class FailingReservation:
        required_bytes = 0

        def close(self):
            raise later

    cache._poison(first)
    cache._reservation = FailingReservation()

    with pytest.raises(RuntimeError) as caught:
        cache.close()
    assert caught.value is first
