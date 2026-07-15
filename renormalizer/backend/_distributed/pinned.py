"""Fixed-capacity host staging storage with completion-safe reuse."""

from contextlib import AbstractContextManager
from dataclasses import dataclass
import threading

import numpy as np

from renormalizer.backend._distributed.async_owner import (
    _require_resource_admission,
    allocation_record,
    require_async_owner,
)
from renormalizer.backend._distributed.terminal import (
    _publish_lease_construction_resource,
    _publish_lease_construction_resource_direct,
    _remaining_lifecycle_time,
    _require_managed_resource_admission,
    _resolve_managed_resource_guard,
)


def _pageable_array(nbytes, *, _construction_slot=None):
    array = np.empty(nbytes, dtype=np.uint8)
    _publish_lease_construction_resource_direct(
        _construction_slot,
        kind="pinned",
        records=(allocation_record(array),),
    )
    return array


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


@dataclass(frozen=True)
class _StagingCheckoutCapability:
    admission: object
    thread_id: int
    generation: int


class StagingSlot:
    """One exclusive view onto a pool-owned fixed allocation."""

    def __init__(self, pool, array, *, pinned):
        self._pool = pool
        self._managed_guard = pool._managed_guard
        self._array = array
        self._pinned = pinned
        self._nbytes = 0
        self._owner = None
        self._checked_out = False
        self._checkout_admission = None
        self._checkout_capability = None

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
            allowed_scopes=("construction", "lease", "lease_close"),
            allowed_operations=allowed_operations,
            epoch=lambda _token: self._pool._managed_epoch,
        )

    def _require_checkout_capability(self, admission, capability):
        if self._managed_guard is None:
            return
        if not isinstance(capability, _StagingCheckoutCapability):
            raise RuntimeError("staging checkout capability is required")
        if (
            self._checkout_capability is not capability
            or capability.admission is not admission
            or capability.thread_id != threading.get_ident()
            or capability.generation != self._pool._slot_generation
        ):
            raise RuntimeError("staging checkout capability does not own the slot")

    @property
    def array(self):
        self._require_admission(
            allowed_operations=(
                "acquire",
                "close_progress",
                "d2h_completion",
                "h2d_completion",
                "load",
                "operator_call",
                "prefetch",
                "schedule_writeback",
            )
        )
        if self._array is None:
            raise RuntimeError("staging slot is closed")
        return self._array

    @property
    def pinned(self):
        self._require_admission(
            allowed_operations=(
                "acquire",
                "load",
                "operator_call",
                "prefetch",
                "resource_state",
                "schedule_writeback",
            )
        )
        return self._pinned

    @property
    def nbytes(self):
        self._require_admission(
            allowed_operations=(
                "acquire",
                "d2h_completion",
                "h2d_completion",
                "load",
                "operator_call",
                "prefetch",
                "resource_state",
                "schedule_writeback",
            )
        )
        return self._nbytes

    @property
    def capacity_bytes(self):
        self._require_admission(
            allowed_operations=(
                "acquire",
                "load",
                "operator_call",
                "prefetch",
                "resource_state",
                "schedule_writeback",
            )
        )
        return self._pool._capacity_bytes

    @property
    def completion_event(self):
        self._require_admission(
            allowed_operations=(
                "acquire",
                "d2h_completion",
                "h2d_completion",
                "load",
                "operator_call",
                "pool_reap",
                "prefetch",
                "reap",
                "schedule_writeback",
            )
        )
        if self._owner is not None:
            return self._owner.completion_event
        return None

    def view(
        self,
        shape,
        dtype,
        *,
        order="C",
        _admission_token=None,
        _admission_validator=None,
        _checkout_capability=None,
    ):
        admission = self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=(
                "acquire",
                "close_progress",
                "load",
                "operator_call",
                "prefetch",
                "schedule_writeback",
            ),
        )
        self._require_checkout_capability(admission, _checkout_capability)
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

    def retain_until(
        self,
        completion,
        *,
        _admission_token=None,
        _admission_validator=None,
        _checkout_capability=None,
    ):
        admission = self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=(
                "acquire",
                "close_progress",
                "load",
                "operator_call",
                "prefetch",
                "schedule_writeback",
            ),
        )
        self._require_checkout_capability(admission, _checkout_capability)
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
    def __init__(
        self,
        pool,
        nbytes,
        admission_token,
        admission_validator,
        checkout_capability,
        *,
        nonblocking,
    ):
        self._pool = pool
        self._nbytes = nbytes
        self._admission_token = admission_token
        self._admission_validator = admission_validator
        self._checkout_capability = checkout_capability
        self._nonblocking = nonblocking
        self._slot = None
        self._entered = False

    def __enter__(self):
        if self._entered:
            raise RuntimeError("staging checkout is already entered")
        self._entered = True
        self._slot = self._pool._checkout(
            self._nbytes,
            _admission_token=self._admission_token,
            _admission_validator=self._admission_validator,
            _checkout_capability=self._checkout_capability,
            _nonblocking=self._nonblocking,
        )
        return self._slot

    def __exit__(self, exc_type, exc_value, traceback):
        if self._slot is not None:
            try:
                self._pool._release(
                    self._slot,
                    _admission_token=self._admission_token,
                    _admission_validator=self._admission_validator,
                    _checkout_capability=self._checkout_capability,
                    _wait_for_counted=not self._nonblocking,
                )
            except BaseException:
                if exc_value is None:
                    raise
            finally:
                self._slot = None
        self._entered = False
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
        _managed_guard=None,
        _standalone=True,
        _allocator_owns_construction_slot=False,
    ):
        _require_resource_admission(_admission_token, _admission_validator)
        self._managed_guard = _resolve_managed_resource_guard(
            standalone=_standalone,
            guard=_managed_guard,
        )
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
        if type(_allocator_owns_construction_slot) is not bool:
            raise TypeError("allocator ownership mode must be a boolean")

        self._managed = self._managed_guard is not None
        self._managed_epoch = (
            None if _admission_token is None else _admission_token.epoch
        )
        self._capacity_bytes = capacity_bytes
        self._slot = StagingSlot(self, None, pinned=True)
        self._allocation_record = None
        self._closed = False
        self._checked_out_bytes = 0
        self._pending_bytes = 0
        self._slot_generation = 0
        self._peak_checked_out_bytes = 0
        self._pageable_fallback_count = 0
        self._pageable_fallback_bytes = 0
        self._poisoned_error = None
        self._resource_recorder = _resource_recorder
        _publish_lease_construction_resource_direct(
            _construction_slot,
            self,
            kind="pinned",
        )

        pinned = True
        try:
            if _allocator_owns_construction_slot:
                array = pinned_allocator(
                    capacity_bytes,
                    _construction_slot=_construction_slot,
                )
            else:
                array = pinned_allocator(capacity_bytes)
            self._slot._array = array
            array, record = _validate_storage(array, capacity_bytes)
        except Exception:
            if (
                _allocator_owns_construction_slot
                and _construction_slot is not None
                and _construction_slot.snapshot().records
            ):
                raise
            self._slot._array = None
            if _allocator_owns_construction_slot:
                array = pageable_allocator(
                    capacity_bytes,
                    _construction_slot=_construction_slot,
                )
            else:
                array = pageable_allocator(capacity_bytes)
            self._slot._array = array
            array, record = _validate_storage(array, capacity_bytes)
            pinned = False
            self._pageable_fallback_count = 1
            self._pageable_fallback_bytes = capacity_bytes

        self._slot._array = array
        self._slot._pinned = pinned
        self._allocation_record = record
        _publish_lease_construction_resource_direct(
            _construction_slot,
            self,
            kind="pinned",
            records=(record,),
        )
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
        self._require_admission(
            None,
            None,
            allowed_operations=(
                "lease_construction",
                "resource_state",
            ),
        )
        return self._capacity_bytes

    @property
    def allocated_bytes(self):
        self._require_admission(
            None,
            None,
            allowed_operations=(
                "acquire",
                "child_close",
                "lease_construction",
                "mark_dirty",
                "operator_call",
                "observe_peaks",
                "resource_state",
            ),
        )
        if self._slot._array is None:
            return 0
        return self._allocation_record.capacity_bytes

    @property
    def allocation_records(self):
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
        return self._terminal_allocation_records()

    def _terminal_allocation_records(self):
        if self._allocation_record is None:
            return ()
        return (self._allocation_record,)

    @property
    def checked_out_bytes(self):
        self._require_admission(
            None,
            None,
            allowed_operations=("observe_peaks", "resource_state"),
        )
        return self._checked_out_bytes

    @property
    def pending_bytes(self):
        self._require_admission(
            None,
            None,
            allowed_operations=(
                "acquire",
                "close_progress",
                "load",
                "operator_call",
                "prefetch",
                "resource_state",
                "schedule_writeback",
            ),
        )
        return self._pending_bytes

    @property
    def peak_checked_out_bytes(self):
        self._require_admission(
            None,
            None,
            allowed_operations=("emit_profile", "observe_peaks", "resource_state"),
        )
        return self._peak_checked_out_bytes

    @property
    def pageable_fallback_count(self):
        self._require_admission(
            None,
            None,
            allowed_operations=("emit_profile", "lease_construction", "resource_state"),
        )
        return self._pageable_fallback_count

    @property
    def pageable_fallback_bytes(self):
        self._require_admission(
            None,
            None,
            allowed_operations=("emit_profile", "lease_construction", "resource_state"),
        )
        return self._pageable_fallback_bytes

    @property
    def poisoned(self):
        self._require_admission(
            None,
            None,
            allowed_operations=(
                "acquire",
                "load",
                "operator_call",
                "pool_close",
                "pool_reap",
                "prefetch",
                "reap",
                "resource_state",
                "schedule_writeback",
            ),
        )
        return self._poisoned_error is not None

    def _require_usable(self):
        if self._poisoned_error is not None:
            raise RuntimeError("pinned buffer pool is terminal-poisoned") from (
                self._poisoned_error
            )
        if self._closed:
            raise RuntimeError("pinned buffer pool is closed")

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
            raise TypeError("managed pinned buffer pool requires admission")
        return _require_managed_resource_admission(
            guard,
            token,
            validator,
            allowed_scopes=("construction", "lease", "lease_close"),
            allowed_operations=allowed_operations,
            epoch=lambda _token: self._managed_epoch,
        )

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
        _nonblocking=False,
    ):
        admission = self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=(
                "acquire",
                "close_progress",
                "load",
                "operator_call",
                "prefetch",
                "schedule_writeback",
            ),
        )
        if type(_nonblocking) is not bool:
            raise TypeError("nonblocking checkout mode must be a boolean")
        self._require_usable()
        if type(nbytes) is not int or nbytes <= 0:
            raise ValueError("checkout nbytes must be a positive integer")
        if nbytes > self._capacity_bytes:
            raise MemoryError("staging checkout exceeds fixed pool capacity")
        capability = _StagingCheckoutCapability(
            admission=admission,
            thread_id=threading.get_ident(),
            generation=self._slot_generation + 1,
        )
        return _SlotCheckout(
            self,
            nbytes,
            _admission_token,
            _admission_validator,
            capability,
            nonblocking=_nonblocking,
        )

    def _checkout(
        self,
        nbytes,
        *,
        _admission_token=None,
        _admission_validator=None,
        _checkout_capability=None,
        _nonblocking=False,
    ):
        admission = self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=(
                "acquire",
                "close_progress",
                "load",
                "operator_call",
                "prefetch",
                "schedule_writeback",
            ),
        )
        if type(_nonblocking) is not bool:
            raise TypeError("nonblocking checkout mode must be a boolean")
        if self._managed:
            capability = _checkout_capability
            if not isinstance(capability, _StagingCheckoutCapability):
                raise RuntimeError("staging checkout capability is required")
            if (
                capability.admission is not admission
                or capability.thread_id != threading.get_ident()
                or capability.generation != self._slot_generation + 1
            ):
                raise RuntimeError(
                    "staging checkout capability does not own this checkout"
                )
        self._require_usable()
        self.reap_completed(
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
            _wait_for_counted=not _nonblocking,
        )
        slot = self._slot
        if slot._checked_out or slot._owner is not None:
            if _nonblocking:
                return None
            raise RuntimeError("no staging slot is available")
        self._slot_generation += 1
        slot._checked_out = True
        slot._nbytes = nbytes
        slot._checkout_admission = admission
        slot._checkout_capability = _checkout_capability
        self._checked_out_bytes = nbytes
        self._peak_checked_out_bytes = max(self._peak_checked_out_bytes, nbytes)
        return slot

    def _release(
        self,
        slot,
        *,
        _admission_token=None,
        _admission_validator=None,
        _checkout_capability=None,
        _wait_for_counted=True,
    ):
        admission = self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=(
                "acquire",
                "close_progress",
                "load",
                "operator_call",
                "prefetch",
                "schedule_writeback",
            ),
        )
        slot._require_checkout_capability(admission, _checkout_capability)
        if slot is not self._slot or not slot._checked_out:
            raise RuntimeError("staging slot is not owned by this checkout")
        slot._checkout_capability = None
        slot._checkout_admission = None
        slot._checked_out = False
        self._slot_generation += 1
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
            owner.reap(_wait_for_counted=_wait_for_counted)
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
        _deadline=None,
        _wait_for_counted=True,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=(
                "acquire",
                "close_progress",
                "load",
                "operator_call",
                "pool_close",
                "pool_reap",
                "prefetch",
                "reap",
                "schedule_writeback",
            ),
        )
        _remaining_lifecycle_time(
            _deadline,
            "pinned pool lifecycle timed out before reap",
        )
        if self._closed and not self.poisoned:
            return
        self._require_usable()
        slot = self._slot
        owner = slot._owner
        if owner is None:
            return
        try:
            owner.reap(_wait_for_counted=_wait_for_counted)
        except BaseException:
            if owner.quarantined:
                self._owner_quarantined(owner)
            raise
        if owner.completed:
            self._finalize_slot()
        _remaining_lifecycle_time(
            _deadline,
            "pinned pool lifecycle timed out during reap",
        )

    def wait_for_slot(
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
                "acquire",
                "close_progress",
                "lease_construction",
                "load",
                "operator_call",
                "pool_close",
                "prefetch",
                "schedule_writeback",
            ),
        )
        _remaining_lifecycle_time(
            _deadline,
            "pinned pool lifecycle timed out before owner wait",
        )
        self._require_usable()
        owner = self._slot._owner
        if owner is not None:
            try:
                owner.wait(_deadline=_deadline)
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
        _deadline=None,
    ):
        self._require_admission(
            _admission_token,
            _admission_validator,
            allowed_operations=("lease_construction", "pool_close"),
        )
        _remaining_lifecycle_time(
            _deadline,
            "pinned pool lifecycle timed out before close",
        )
        if self._closed:
            return
        if self._slot._checked_out:
            raise RuntimeError("cannot close pool with a checked out staging slot")
        error = self._poisoned_error
        if error is None:
            try:
                self.wait_for_slot(
                    _admission_token=_admission_token,
                    _admission_validator=_admission_validator,
                    _deadline=_deadline,
                )
            except BaseException as caught:
                if self._managed:
                    raise
                error = caught
        elif self._managed:
            raise error
        _remaining_lifecycle_time(
            _deadline,
            "pinned pool lifecycle timed out before destructive close",
        )
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
