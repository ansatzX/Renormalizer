import gc
import weakref

import numpy as np
import pytest

from renormalizer.backend._distributed.async_owner import AsyncResourceOwner
from renormalizer.backend._distributed.pinned import PinnedBufferPool


class _ManualEvent:
    def __init__(self):
        self.done = False
        self.synchronize_calls = 0

    def query(self):
        return self.done

    def synchronize(self):
        self.synchronize_calls += 1
        self.done = True

    def complete(self):
        self.done = True


class _QueryFailureEvent(_ManualEvent):
    def __init__(self, *, synchronize_fails=False):
        super().__init__()
        self.query_calls = 0
        self.synchronize_fails = synchronize_fails

    def query(self):
        self.query_calls += 1
        if self.query_calls == 1:
            return False
        raise RuntimeError("injected staging query failure")

    def synchronize(self):
        self.synchronize_calls += 1
        if self.synchronize_fails:
            raise RuntimeError("injected staging drain failure")
        self.done = True


def _event_owner(event, *, array=None):
    arrays = () if array is None else (array,)
    owner = AsyncResourceOwner("staging", arrays=arrays)
    owner.add_event(event, completion=True)
    owner.mark_enqueued()
    return owner


def test_checkout_is_exclusive_fixed_capacity_and_reuses_one_slot():
    allocations = []

    def allocate(nbytes):
        allocations.append(nbytes)
        return np.empty(nbytes, dtype=np.uint8)

    pool = PinnedBufferPool(64, pinned_allocator=allocate)
    with pool.checkout(32) as first:
        first_array = first.array
        assert first.pinned is True
        assert first.nbytes == 32
        with pytest.raises(RuntimeError, match="no staging slot"):
            with pool.checkout(1):
                pass
    with pool.checkout(64) as second:
        assert second.array is first_array
    with pytest.raises(MemoryError, match="capacity"):
        with pool.checkout(65):
            pass

    assert allocations == [64]
    assert pool.capacity_bytes == 64
    assert pool.peak_checked_out_bytes == 64
    pool.close()
    pool.close()


def test_pool_rejects_exact_view_with_oversized_physical_backing():
    pinned_backing = np.empty(64, dtype=np.uint8)
    pageable_backing = np.empty(64, dtype=np.uint8)

    with pytest.raises(MemoryError, match="physical.*capacity"):
        PinnedBufferPool(
            32,
            pinned_allocator=lambda _nbytes: pinned_backing[:32],
            pageable_allocator=lambda _nbytes: pageable_backing[:32],
        )


def test_staging_view_supports_exact_c_and_f_order_without_new_allocation(
    monkeypatch,
):
    pool = PinnedBufferPool(48)

    def fail_conversion(*args, **kwargs):
        raise AssertionError("staging order must not allocate a tensor conversion")

    monkeypatch.setattr(np, "ascontiguousarray", fail_conversion)
    monkeypatch.setattr(np, "asfortranarray", fail_conversion)
    with pool.checkout(48) as slot:
        raw = slot.array
        c_view = slot.view((2, 3), np.float64, order="C")
        f_view = slot.view((2, 3), np.float64, order="F")

        assert c_view.flags.c_contiguous
        assert f_view.flags.f_contiguous
        assert not f_view.flags.c_contiguous
        assert np.shares_memory(c_view, raw)
        assert np.shares_memory(f_view, raw)
        assert (
            c_view.__array_interface__["data"][0] == raw.__array_interface__["data"][0]
        )
        assert (
            f_view.__array_interface__["data"][0] == raw.__array_interface__["data"][0]
        )
        assert pool.checked_out_bytes == 48
        assert pool.allocated_bytes == 48

    assert pool.pending_bytes == 0
    pool.close()


def test_checkout_remains_live_until_completion_event_before_reuse():
    event = _ManualEvent()
    pool = PinnedBufferPool(
        64, pinned_allocator=lambda nbytes: np.empty(nbytes, dtype=np.uint8)
    )

    with pool.checkout(48) as first:
        first_array = first.array
        first.retain_until(_event_owner(event, array=first.array))

    assert pool.pending_bytes == 48
    pool.reap_completed()
    assert pool.pending_bytes == 48
    event.complete()
    pool.reap_completed()
    assert pool.pending_bytes == 0
    with pool.checkout(48) as second:
        assert second.array is first_array

    pool.close()


def test_staging_release_rejects_raw_event_without_registered_owner():
    pool = PinnedBufferPool(32)

    with pool.checkout(16) as slot:
        with pytest.raises(TypeError, match="pre-existing AsyncResourceOwner"):
            slot.retain_until(_ManualEvent())

    pool.close()


def test_staging_owner_records_full_pool_capacity_for_smaller_checkout():
    event = _ManualEvent()
    pool = PinnedBufferPool(64)

    with pool.checkout(16) as slot:
        owner = _event_owner(event)
        slot.retain_until(owner)
        assert len(owner.allocations) == 1
        assert owner.allocations[0].owner is slot.array
        assert owner.allocations[0].capacity_bytes == 64

    event.complete()
    pool.reap_completed()
    pool.close()


def test_query_failure_drains_then_finalizes_staging_without_early_reuse():
    event = _QueryFailureEvent()
    pool = PinnedBufferPool(32)

    with pool.checkout(16) as slot:
        slot.retain_until(_event_owner(event, array=slot.array))
    with pytest.raises(RuntimeError, match="staging query failure"):
        pool.reap_completed()

    assert event.synchronize_calls == 1
    assert pool.pending_bytes == 0
    assert pool.checked_out_bytes == 0
    assert pool.poisoned is False
    with pool.checkout(16):
        pass
    pool.close()


def test_query_and_drain_failure_poison_pool_and_retain_staging_ownership():
    event = _QueryFailureEvent(synchronize_fails=True)
    pool = PinnedBufferPool(32)
    raw_ref = weakref.ref(pool._slot._array)

    with pool.checkout(16) as slot:
        slot.retain_until(_event_owner(event, array=slot.array))
    with pytest.raises(RuntimeError, match="staging query failure"):
        pool.reap_completed()

    assert event.synchronize_calls == 1
    assert pool.pending_bytes == 16
    assert pool.allocated_bytes == 32
    assert raw_ref() is not None
    assert pool.poisoned is True
    with pytest.raises(RuntimeError, match="terminal-poisoned"):
        pool.checkout(1)
    with pytest.raises(RuntimeError, match="staging query failure"):
        pool.close()
    assert pool.pending_bytes == 16
    assert pool.allocated_bytes == 32
    assert raw_ref() is not None


def test_pinned_failure_substitutes_one_same_sized_pageable_slot():
    pinned_calls = []
    pageable_calls = []

    def fail_pinned(nbytes):
        pinned_calls.append(nbytes)
        raise MemoryError("pinning unavailable")

    def allocate_pageable(nbytes):
        pageable_calls.append(nbytes)
        return np.empty(nbytes, dtype=np.uint8)

    pool = PinnedBufferPool(
        96,
        pinned_allocator=fail_pinned,
        pageable_allocator=allocate_pageable,
    )
    with pool.checkout(80) as slot:
        assert slot.pinned is False
        assert slot.capacity_bytes == 96

    assert pinned_calls == [96]
    assert pageable_calls == [96]
    assert pool.pageable_fallback_count == 1
    assert pool.pageable_fallback_bytes == 96
    assert pool.allocated_bytes == 96
    pool.close()
    assert pool.allocated_bytes == 0


def test_close_waits_for_pending_completion_and_rejects_live_checkout():
    event = _ManualEvent()
    pool = PinnedBufferPool(
        32, pinned_allocator=lambda nbytes: np.empty(nbytes, dtype=np.uint8)
    )
    checkout = pool.checkout(16)
    slot = checkout.__enter__()
    with pytest.raises(RuntimeError, match="checked out"):
        pool.close()
    slot.retain_until(_event_owner(event, array=slot.array))
    checkout.__exit__(None, None, None)

    pool.close()

    assert event.synchronize_calls == 1
    assert pool.allocated_bytes == 0


def test_close_event_failure_retains_staging_allocation_as_terminal_owned():
    class FailingEvent(_ManualEvent):
        def synchronize(self):
            self.synchronize_calls += 1
            raise RuntimeError("injected staging event failure")

    event = FailingEvent()
    pool = PinnedBufferPool(32)
    array_ref = weakref.ref(pool._slot._array)
    with pool.checkout(16) as slot:
        slot.retain_until(_event_owner(event, array=slot.array))

    with pytest.raises(RuntimeError, match="staging event failure"):
        pool.close()
    gc.collect()

    assert pool.allocated_bytes == 32
    assert pool.pending_bytes == 16
    assert pool.poisoned is True
    assert array_ref() is not None
    pool.close()
