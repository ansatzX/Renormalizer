"""Fixed-capacity host staging storage with completion-safe reuse."""

from contextlib import AbstractContextManager

import numpy as np

from renormalizer.backend._distributed.async_owner import (
    _require_resource_admission,
    allocation_record,
    require_async_owner,
)
from renormalizer.backend._distributed.terminal import (
    _publish_lease_construction_resource,
)


def _pageable_array(nbytes):
    return np.empty(nbytes, dtype=np.uint8)


def _validate_storage(value, nbytes):
    if type(value) is not np.ndarray:
        raise TypeError("staging allocator must return a NumPy ndarray")
    if (
        value.dtype != np.dtype(np.uint8)
        or value.ndim != 1
        or int(value.nbytes) != nbytes
        or not value.flags.c_contiguous
        or not value.flags.writeable
    ):
        raise ValueError("staging allocator returned inconsistent storage")
    record = allocation_record(value)
    if record.capacity_bytes != nbytes:
        raise MemoryError("staging physical allocation exceeds fixed pool capacity")
    return value, record


class StagingSlot:
    """One exclusive view onto a pool-owned fixed allocation."""

    def __init__(self, pool, array, *, pinned):
        self._pool = pool
        self._array = array
        self._pinned = pinned
        self._nbytes = 0
        self._owner = None
        self._checked_out = False

    @property
    def array(self):
        if self._array is None:
            raise RuntimeError("staging slot is closed")
        return self._array

    @property
    def pinned(self):
        return self._pinned

    @property
    def nbytes(self):
        return self._nbytes

    @property
    def capacity_bytes(self):
        return self._pool.capacity_bytes

    @property
    def completion_event(self):
        if self._owner is not None:
            return self._owner.completion_event
        return None

    def view(self, shape, dtype, *, order="C"):
        if not self._checked_out:
            raise RuntimeError("staging slot is not checked out")
        if order not in {"C", "F"}:
            raise ValueError("staging view order must be 'C' or 'F'")
        dtype = np.dtype(dtype)
        shape = tuple(shape)
        elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
        required = elements * int(dtype.itemsize)
        if required != self._nbytes:
            raise ValueError("staging view does not match checkout size")
        return np.ndarray(
            shape=shape,
            dtype=dtype,
            buffer=self.array,
            offset=0,
            order=order,
        )

    def retain_until(self, completion):
        if not self._checked_out:
            raise RuntimeError("staging slot is not checked out")
        if self._owner is not None:
            raise RuntimeError("staging slot already has a completion owner")
        owner = require_async_owner(completion, "staging lifetime transfer")
        self._owner = owner
        owner.capture_allocations(self.array)
        owner.capture_resources(self)
        owner.add_release_callback(
            lambda pool=self._pool, slot=self, retained=owner: pool._owner_detached(
                slot, retained
            )
        )


class _SlotCheckout(AbstractContextManager):
    def __init__(self, pool, nbytes):
        self._pool = pool
        self._nbytes = nbytes
        self._slot = None

    def __enter__(self):
        if self._slot is not None:
            raise RuntimeError("staging checkout is already entered")
        self._slot = self._pool._checkout(self._nbytes)
        return self._slot

    def __exit__(self, exc_type, exc_value, traceback):
        if self._slot is not None:
            try:
                self._pool._release(self._slot)
            except BaseException:
                if exc_value is None:
                    raise
            finally:
                self._slot = None
        return False


class PinnedBufferPool:
    """One-lane staging pool whose allocation can never grow."""

    _terminal_resource_kind = "pinned"

    def __init__(
        self,
        capacity_bytes,
        *,
        lanes=1,
        pinned_allocator=None,
        pageable_allocator=None,
        _resource_recorder=None,
        _admission_token=None,
        _admission_validator=None,
        _construction_slot=None,
    ):
        _require_resource_admission(_admission_token, _admission_validator)
        if type(capacity_bytes) is not int or capacity_bytes < 0:
            raise ValueError("capacity_bytes must be a non-negative integer")
        if lanes != 1:
            raise ValueError("PinnedBufferPool currently requires exactly one lane")
        if pinned_allocator is None:
            pinned_allocator = _pageable_array
        if pageable_allocator is None:
            pageable_allocator = _pageable_array
        if not callable(pinned_allocator) or not callable(pageable_allocator):
            raise TypeError("staging allocators must be callable")
        if _resource_recorder is not None and not callable(_resource_recorder):
            raise TypeError("resource recorder must be callable")

        pinned = True
        fallback_count = 0
        fallback_bytes = 0
        try:
            array, record = _validate_storage(
                pinned_allocator(capacity_bytes), capacity_bytes
            )
        except Exception:
            array, record = _validate_storage(
                pageable_allocator(capacity_bytes), capacity_bytes
            )
            pinned = False
            fallback_count = 1
            fallback_bytes = capacity_bytes

        self._capacity_bytes = capacity_bytes
        self._slot = StagingSlot(self, array, pinned=pinned)
        self._allocation_record = record
        self._closed = False
        self._checked_out_bytes = 0
        self._pending_bytes = 0
        self._peak_checked_out_bytes = 0
        self._pageable_fallback_count = fallback_count
        self._pageable_fallback_bytes = fallback_bytes
        self._poisoned_error = None
        self._resource_recorder = _resource_recorder
        _publish_lease_construction_resource(
            _construction_slot,
            self,
            kind="pinned",
            records=(record,),
        )
        if self._resource_recorder is not None:
            self._resource_recorder(
                resource=self,
                kind="pinned",
                records=(record,),
            )

    @property
    def capacity_bytes(self):
        return self._capacity_bytes

    @property
    def allocated_bytes(self):
        if self._slot._array is None:
            return 0
        return self._allocation_record.capacity_bytes

    @property
    def allocation_records(self):
        if self._allocation_record is None:
            return ()
        return (self._allocation_record,)

    @property
    def checked_out_bytes(self):
        return self._checked_out_bytes

    @property
    def pending_bytes(self):
        return self._pending_bytes

    @property
    def peak_checked_out_bytes(self):
        return self._peak_checked_out_bytes

    @property
    def pageable_fallback_count(self):
        return self._pageable_fallback_count

    @property
    def pageable_fallback_bytes(self):
        return self._pageable_fallback_bytes

    @property
    def poisoned(self):
        return self._poisoned_error is not None

    def _require_usable(self):
        if self._poisoned_error is not None:
            raise RuntimeError("pinned buffer pool is terminal-poisoned") from (
                self._poisoned_error
            )
        if self._closed:
            raise RuntimeError("pinned buffer pool is closed")

    def _poison(self, error):
        if self._poisoned_error is None:
            self._poisoned_error = error

    def _finalize_slot(self):
        self._slot._owner = None
        self._slot._nbytes = 0
        self._pending_bytes = 0

    def _owner_detached(self, slot, owner):
        if slot is not self._slot or slot._owner is not owner:
            return
        if not slot._checked_out:
            self._finalize_slot()

    def _owner_quarantined(self, owner):
        self._poison(owner.error)
        if self._slot._owner is owner:
            self._pending_bytes = self._slot._nbytes

    def checkout(
        self,
        nbytes,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        _require_resource_admission(_admission_token, _admission_validator)
        self._require_usable()
        if type(nbytes) is not int or nbytes <= 0:
            raise ValueError("checkout nbytes must be a positive integer")
        if nbytes > self._capacity_bytes:
            raise MemoryError("staging checkout exceeds fixed pool capacity")
        return _SlotCheckout(self, nbytes)

    def _checkout(self, nbytes):
        self._require_usable()
        self.reap_completed()
        slot = self._slot
        if slot._checked_out or slot._owner is not None:
            raise RuntimeError("no staging slot is available")
        slot._checked_out = True
        slot._nbytes = nbytes
        self._checked_out_bytes = nbytes
        self._peak_checked_out_bytes = max(self._peak_checked_out_bytes, nbytes)
        return slot

    def _release(self, slot):
        if slot is not self._slot or not slot._checked_out:
            raise RuntimeError("staging slot is not owned by this checkout")
        slot._checked_out = False
        self._checked_out_bytes = 0
        owner = slot._owner
        if owner is None:
            self._finalize_slot()
            return
        self._pending_bytes = slot._nbytes
        if owner.completed:
            self._finalize_slot()
            return
        if owner.quarantined:
            self._owner_quarantined(owner)
            if owner.kind == "staging":
                raise owner.error
            return
        if owner.kind != "staging":
            return
        try:
            owner.reap()
        except BaseException:
            if owner.quarantined:
                self._owner_quarantined(owner)
            raise
        if owner.completed:
            self._finalize_slot()

    def reap_completed(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        _require_resource_admission(_admission_token, _admission_validator)
        if self._closed and not self.poisoned:
            return
        self._require_usable()
        slot = self._slot
        owner = slot._owner
        if owner is None:
            return
        try:
            owner.reap()
        except BaseException:
            if owner.quarantined:
                self._owner_quarantined(owner)
            raise
        if owner.completed:
            self._finalize_slot()

    def wait_for_slot(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        _require_resource_admission(_admission_token, _admission_validator)
        self._require_usable()
        owner = self._slot._owner
        if owner is not None:
            try:
                owner.wait()
            except BaseException as error:
                if owner.quarantined:
                    self._owner_quarantined(owner)
                raise
            self._finalize_slot()

    def close(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        _require_resource_admission(_admission_token, _admission_validator)
        if self._closed:
            return
        if self._slot._checked_out:
            raise RuntimeError("cannot close pool with a checked out staging slot")
        error = None
        if self._poisoned_error is not None:
            error = self._poisoned_error
        else:
            try:
                self.wait_for_slot()
            except BaseException as caught:
                error = caught
        if self._poisoned_error is None:
            self._slot._array = None
            self._allocation_record = None
            self._slot._owner = None
            self._slot._nbytes = 0
            self._pending_bytes = 0
        self._resource_recorder = None
        self._slot._pool = None
        self._closed = True
        if error is not None:
            raise error


__all__ = ["PinnedBufferPool", "StagingSlot"]
