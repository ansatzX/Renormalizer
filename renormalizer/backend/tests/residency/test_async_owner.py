import gc
import weakref

import numpy as np
import pytest

from renormalizer.backend._distributed.async_owner import (
    AsyncResourceOwner,
    RuntimeTerminalQuarantine,
)


class _ManualEvent:
    def __init__(self, *, done=False):
        self.done = done

    def query(self):
        return self.done

    def synchronize(self):
        self.done = True


def test_owner_state_transition_is_the_only_resource_release_point():
    array = np.arange(4, dtype=np.float64)
    array_ref = weakref.ref(array)
    event = _ManualEvent()
    owner = AsyncResourceOwner("h2d", arrays=(array,), nbytes=array.nbytes)
    owner.add_event(event, completion=True)
    owner.mark_enqueued()
    del array

    assert owner.state == "enqueued"
    assert owner.reap() is False
    gc.collect()
    assert array_ref() is not None

    event.done = True
    assert owner.reap() is True
    gc.collect()
    assert owner.state == "detached"
    assert array_ref() is None


def test_drain_failure_moves_complete_owner_to_terminal_quarantine():
    quarantine = RuntimeTerminalQuarantine()
    array = np.arange(4, dtype=np.float64)
    array_ref = weakref.ref(array)
    event = _ManualEvent()
    first_error = RuntimeError("injected first completion failure")

    def fail_drain():
        raise RuntimeError("injected later drain failure")

    owner = AsyncResourceOwner(
        "compute",
        arrays=(array,),
        streams=(object(),),
        nbytes=array.nbytes,
        drainer=fail_drain,
        quarantine=quarantine.retain,
    )
    owner.add_event(event, completion=True)
    owner.mark_enqueued()
    del array

    with pytest.raises(RuntimeError, match="first completion failure"):
        owner.fail(first_error)
    gc.collect()

    assert owner.state == "quarantined"
    assert owner.error is first_error
    assert array_ref() is not None
    assert quarantine.first_error is first_error
    assert quarantine.resource_state() == {
        "quarantined_owner_count": 1,
        "quarantined_array_count": 1,
        "quarantined_bytes": 32,
        "quarantined_event_count": 1,
        "quarantined_stream_count": 1,
    }


def test_quarantine_counts_aliases_once_at_full_backing_capacity():
    quarantine = RuntimeTerminalQuarantine()
    backing = np.empty(128, dtype=np.uint8)
    small = backing[:16]
    alias = backing[8:24]
    error = RuntimeError("injected allocation quarantine")
    owner = AsyncResourceOwner(
        "compute",
        arrays=(small, alias),
        quarantine=quarantine.retain,
    )

    owner.force_quarantine(error)

    assert len(owner.allocations) == 1
    allocation = owner.allocations[0]
    assert allocation.owner is backing
    assert allocation.capacity_bytes == backing.nbytes
    assert quarantine.resource_state()["quarantined_array_count"] == 1
    assert quarantine.resource_state()["quarantined_bytes"] == backing.nbytes


def test_cross_owner_quarantine_keeps_maximum_capacity_when_small_alias_is_last():
    quarantine = RuntimeTerminalQuarantine()
    backing = bytearray(128)
    full = np.frombuffer(backing, dtype=np.uint8)
    small = full[:16]
    error = RuntimeError("injected cross-owner quarantine")
    full_owner = AsyncResourceOwner("compute", arrays=(full,))
    small_owner = AsyncResourceOwner("compute", arrays=(small,))

    full_owner.force_quarantine(error)
    small_owner.force_quarantine(error)
    quarantine.retain(full_owner)
    quarantine.retain(small_owner)

    assert quarantine.resource_state()["quarantined_array_count"] == 1
    assert quarantine.resource_state()["quarantined_bytes"] == 128
