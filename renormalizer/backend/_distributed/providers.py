"""Resident operand leases for distributed block execution."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import MappingProxyType
from typing import ContextManager, Protocol

import numpy as np

from renormalizer.backend._distributed.async_owner import (
    allocation_records,
    merge_allocation_records,
)
from renormalizer.backend._distributed.context import DistributedContext
from renormalizer.backend._distributed.cache import (
    CacheAllocation,
    CacheEntrySpec,
    CacheEntryLease,
    DeviceTensorCache,
)
from renormalizer.backend._distributed.pinned import PinnedBufferPool, StagingSlot
from renormalizer.backend._distributed.planner import DistributedPlan
from renormalizer.backend._distributed.residency import (
    HostTensorStore,
    ResidencyPlan,
    ResidencyPreflightReceipt,
    ResidencyRequest,
)
from renormalizer.backend._distributed.transfer import (
    TransferScheduler,
    TransferSource,
)
from renormalizer.backend._execution.model import ExecutionBindings, ExecutionPlan
from renormalizer.backend._execution.executor import _validate_array


@dataclass(frozen=True)
class OperandRequest:
    execution_plan: ExecutionPlan
    source_bindings: ExecutionBindings
    distributed_plan: DistributedPlan
    context: DistributedContext
    source_rank: int
    broadcast_variable: object
    operand_slices: object
    residency_request: object = None
    residency_plan: object = None

    def __post_init__(self):
        if not isinstance(self.execution_plan, ExecutionPlan):
            raise TypeError("execution_plan must be an ExecutionPlan")
        if not isinstance(self.source_bindings, ExecutionBindings):
            raise TypeError("source_bindings must be ExecutionBindings")
        if not isinstance(self.distributed_plan, DistributedPlan):
            raise TypeError("distributed_plan must be a DistributedPlan")
        if not isinstance(self.context, DistributedContext):
            raise TypeError("context must be a DistributedContext")
        if type(self.source_rank) is not int:
            raise TypeError("source_rank must be an integer")
        try:
            copied = {
                key: tuple(local_slice)
                for key, local_slice in self.operand_slices.items()
            }
        except (AttributeError, TypeError) as error:
            raise TypeError("operand_slices must be a mapping") from error
        if self.context.world_size != self.distributed_plan.world_size:
            raise ValueError("request context does not match distributed plan")
        if self.source_rank < 0 or self.source_rank >= self.context.world_size:
            raise ValueError("source_rank is out of range for request context")
        block = self.distributed_plan.block_plan(self.context.rank, self.source_rank)
        if self.execution_plan != block.execution_plan:
            raise ValueError("request execution_plan does not match rank/source block")
        if copied != dict(block.operand_slices):
            raise ValueError("request operand_slices do not match rank/source block")
        if self.residency_plan is None:
            if self.residency_request is not None:
                raise ValueError(
                    "residency_request requires an authorized residency plan"
                )
        else:
            from renormalizer.backend._distributed.residency import (
                ResidencyPlan,
                ResidencyRequest,
            )

            if not isinstance(self.residency_plan, ResidencyPlan):
                raise TypeError("residency_plan must be a ResidencyPlan or None")
            if not isinstance(self.residency_request, ResidencyRequest):
                raise TypeError(
                    "residency_request must be a ResidencyRequest with a plan"
                )
            if self.residency_plan.world_size != self.context.world_size:
                raise ValueError("residency plan world size does not match request")
            if (
                self.residency_plan.placement_hash
                != self.distributed_plan.placement_hash
            ):
                raise ValueError("residency plan placement does not match request")
            if self.residency_request.distributed_plan != self.distributed_plan:
                raise ValueError(
                    "residency request distributed plan does not match operand request"
                )
            self.residency_plan.validate_request(self.residency_request)
        object.__setattr__(self, "operand_slices", MappingProxyType(copied))


def active_working_set_policy_error(
    residency_policy,
    backend,
    distributed_plan=None,
):
    """Return the closed-world active-policy error without invoking user code."""
    if residency_policy != "active_working_set":
        return None
    config = getattr(backend, "config", None)
    execution_policy = getattr(config, "execution_policy", None)
    fallback_policy = getattr(config, "fallback_policy", None)
    message = (
        "active_working_set requires a complete local-H-v execution contract "
        "under execution_ir with fallback_policy='error'"
    )
    if execution_policy != "execution_ir" or fallback_policy != "error":
        return ValueError(message)
    if not isinstance(distributed_plan, DistributedPlan):
        return NotImplementedError(message)
    source = distributed_plan.execution_plan
    if source.execution_contract is None or any(
        block.execution_plan.execution_contract is None
        for block in distributed_plan.block_plans
    ):
        return NotImplementedError(message)
    return None


class OperandLease(Protocol):
    bindings: ExecutionBindings

    def mark_dirty(self, key: str, local_array) -> None:
        ...

    def close(self) -> None:
        ...


class OperandProvider(Protocol):
    def validate_setup(
        self,
        distributed_plan: DistributedPlan,
        source_bindings: ExecutionBindings,
        context: DistributedContext,
    ) -> None:
        ...

    def validate_request(self, request: OperandRequest, residency_plan=None) -> None:
        ...

    def acquire(self, request: OperandRequest) -> ContextManager[OperandLease]:
        ...


def _layout_matches(array, layout):
    c_contiguous = bool(array.flags.c_contiguous)
    f_contiguous = bool(array.flags.f_contiguous)
    if layout == "C":
        return c_contiguous
    if layout == "F":
        return f_contiguous
    return not c_contiguous and not f_contiguous


def _validate_dirty_array(backend, array, allocation):
    backend._validate_execution_array(array)
    if tuple(array.shape) != allocation.shape:
        raise ValueError("dirty array shape does not match host output")
    if np.dtype(array.dtype).name != allocation.dtype:
        raise ValueError("dirty array dtype does not match host output")
    if int(array.nbytes) != allocation.nbytes:
        raise ValueError("dirty array nbytes do not match host output")
    if not _layout_matches(array, allocation.layout):
        raise ValueError("dirty array layout does not match host output")


def _validate_resident_array(array, context):
    if isinstance(array, np.ndarray):
        return "numpy"
    module = type(array).__module__.split(".", 1)[0]
    if module != "cupy" or not hasattr(array, "device"):
        raise NotImplementedError(
            "DeviceResidentProvider supports only NumPy and CuPy arrays"
        )
    if int(array.device.id) != context.local_rank:
        raise ValueError("resident CuPy array is not on the selected local device")
    return "cupy"


def _shares_storage(view, source, kind):
    if kind == "numpy":
        return np.shares_memory(view, source)
    return view.data.mem is source.data.mem


class _DeviceResidentLease:
    def __init__(self, bindings):
        self.bindings = bindings
        self._closed = False
        self._dirty = {}

    def mark_dirty(self, key, local_array):
        if self._closed:
            raise RuntimeError("operand lease is closed")
        if key not in self.bindings.arrays:
            raise ValueError("unknown lease binding {!r}".format(key))
        if local_array is not self.bindings.arrays[key]:
            raise ValueError("dirty array must be the leased binding")
        self._dirty[key] = local_array

    def close(self):
        self._closed = True


class DeviceResidentProvider:
    """Lease existing backend arrays and their slices without transfer or copy."""

    residency_policy = "device_resident"
    provider_role = "resident"

    @staticmethod
    def validate_setup(distributed_plan, source_bindings, context):
        if not isinstance(distributed_plan, DistributedPlan):
            raise TypeError("distributed_plan must be a DistributedPlan")
        if not isinstance(source_bindings, ExecutionBindings):
            raise TypeError("source_bindings must be ExecutionBindings")
        if not isinstance(context, DistributedContext):
            raise TypeError("context must be a DistributedContext")
        if context.world_size != distributed_plan.world_size:
            raise ValueError("provider context does not match distributed plan")
        variable_key = distributed_plan.variable_key
        expected_refs = {
            ref.key: ref
            for ref in distributed_plan.execution_plan.inputs
            if ref.key != variable_key
        }
        if set(source_bindings.arrays) != set(expected_refs):
            raise ValueError("resident source binding coverage is incomplete")
        kinds = set()
        for key, ref in expected_refs.items():
            array = source_bindings.arrays[key]
            kinds.add(_validate_resident_array(array, context))
            if tuple(array.shape) != ref.spec.shape:
                raise ValueError("resident source shape does not match execution plan")
            if np.dtype(array.dtype).name != ref.spec.dtype:
                raise ValueError("resident source dtype does not match execution plan")
            if not _layout_matches(array, ref.spec.layout):
                raise ValueError("resident source layout does not match execution plan")
        if len(kinds) > 1:
            raise ValueError("resident operands must use one array backend")

    @staticmethod
    def _build_bindings(request, *, require_variable):
        expected_refs = {ref.key: ref for ref in request.execution_plan.inputs}
        variable_key = request.distributed_plan.variable_key
        source_keys = set(request.source_bindings.arrays)
        expected_source_keys = set(expected_refs) - {variable_key}
        missing = sorted(expected_source_keys - source_keys)
        unexpected = sorted(source_keys - expected_source_keys)
        if missing:
            raise ValueError(
                "missing resident binding keys: {}".format(", ".join(missing))
            )
        if unexpected:
            raise ValueError(
                "unexpected resident binding keys: {}".format(", ".join(unexpected))
            )

        arrays = {}
        kinds = set()
        for key in sorted(expected_source_keys):
            source = request.source_bindings.arrays[key]
            kind = _validate_resident_array(source, request.context)
            kinds.add(kind)
            try:
                local_array = source[request.operand_slices[key]]
            except (IndexError, KeyError, TypeError) as error:
                raise ValueError(
                    "resident operand {!r} cannot supply its planned slice".format(key)
                ) from error
            if not _shares_storage(local_array, source, kind):
                raise ValueError("resident operand slicing unexpectedly copied data")
            arrays[key] = local_array

        if require_variable:
            variable = request.broadcast_variable
            kind = _validate_resident_array(variable, request.context)
            kinds.add(kind)
            arrays[variable_key] = variable
        if len(kinds) > 1:
            raise ValueError("resident operands must use one array backend")

        for key, array in arrays.items():
            ref = expected_refs[key]
            if tuple(array.shape) != ref.spec.shape:
                raise ValueError(
                    "resident operand {!r} shape does not match block plan".format(key)
                )
            if np.dtype(array.dtype).name != ref.spec.dtype:
                raise ValueError(
                    "resident operand {!r} dtype does not match block plan".format(key)
                )
            if not _layout_matches(array, ref.spec.layout):
                raise ValueError(
                    "resident operand {!r} layout does not match block plan".format(key)
                )
        return ExecutionBindings(arrays)

    def validate_request(self, request, residency_plan=None):
        if not isinstance(request, OperandRequest):
            raise TypeError("request must be an OperandRequest")
        if residency_plan is not None and residency_plan is not request.residency_plan:
            raise ValueError("request residency plan identity does not match")
        self._build_bindings(request, require_variable=False)

    def validate_resident(self, request):
        self.validate_request(request, request.residency_plan)

    @contextmanager
    def acquire(self, request):
        if not isinstance(request, OperandRequest):
            raise TypeError("request must be an OperandRequest")
        self.validate_request(request, request.residency_plan)
        bindings = self._build_bindings(request, require_variable=True)
        bindings.validate_for(request.execution_plan)
        lease = _DeviceResidentLease(bindings)
        try:
            yield lease
        finally:
            lease.close()


def _placement_ranges(local_slice, shape):
    if len(local_slice) != len(shape):
        raise ValueError("operand slice rank does not match host tensor")
    normalized = []
    for selected, dimension in zip(local_slice, shape):
        start, stop, step = selected.indices(dimension)
        if step != 1 or start >= stop:
            raise ValueError(
                "operand slice must be canonical, non-empty, and unit-step"
            )
        normalized.append((start, stop, step))
    return tuple(normalized)


def _cache_identity(ref, placement, device):
    return (
        ref.store_id,
        ref.key,
        ref.generation,
        ref.version,
        placement.rank,
        device,
        tuple((value.start, value.stop, value.step) for value in placement.ranges),
        ref.dtype,
        placement.layout,
    )


def _slice_from_ranges(ranges):
    return tuple(slice(value.start, value.stop, value.step) for value in ranges)


class WorkingSetMetrics:
    def __init__(self):
        self.h2d_count = 0
        self.h2d_bytes = 0
        self.d2h_bytes = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.prefetch_overlap_s = 0.0
        self.prefetch_wait_s = 0.0
        self.dirty_writeback_count = 0
        self.dirty_writeback_bytes = 0
        self.dirty_writeback_s = 0.0
        self.pageable_fallback_count = 0
        self.pageable_fallback_bytes = 0
        self.full_replica = False

    def record_h2d(self, nbytes):
        self.h2d_count += 1
        self.h2d_bytes += nbytes


class _ActiveOperandLease:
    def __init__(
        self, working_set, bindings, cache_leases, compute_handle, *, shared_call=False
    ):
        self._working_set = working_set
        self.bindings = bindings
        self._cache_leases = tuple(cache_leases)
        self._compute_handle = compute_handle
        self._shared_call = shared_call
        self._closed = False

    def mark_dirty(self, key, local_array):
        if self._closed:
            raise RuntimeError("operand lease is closed")
        self._working_set.mark_dirty(key, local_array)

    def _capture_execution_allocation(self, array):
        if self._closed:
            raise RuntimeError("operand lease is closed")
        owner = self._working_set._active_operator_owner
        if owner is None:
            owner = self._compute_handle.owner
        owner.capture_arrays(array)

    def close(self):
        if self._closed:
            return
        working_set = self._working_set
        leases = self._cache_leases
        event = None
        error = None
        try:
            working_set._observe_peaks()
        except BaseException as caught:
            error = caught
        if not self._shared_call:
            try:
                dirty_arrays = (
                    () if working_set._dirty is None else (working_set._dirty[1],)
                )
                event = working_set.scheduler.record_compute_completion(
                    handle=self._compute_handle,
                    arrays=dirty_arrays,
                )
            except BaseException as caught:
                if error is None:
                    error = caught
                if working_set.scheduler.poisoned:
                    event = working_set.scheduler.last_compute_event
            for lease in leases:
                if not getattr(lease, "_closed", False):
                    try:
                        lease.close(event)
                    except BaseException as caught:
                        if error is None:
                            error = caught
        try:
            working_set._children.discard(self)
        finally:
            working_set._provider.last_compute_event = None
            self.bindings = None
            self._cache_leases = ()
            self._compute_handle = None
            self._shared_call = False
            self._working_set = None
            self._closed = True
        if error is not None:
            working_set._poison(error)
            raise error

    def __enter__(self):
        if self._closed:
            raise RuntimeError("operand lease is closed")
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


class _LeaseStatusWorkspace:
    def __init__(self, device_status, host_status):
        self.device_status = device_status
        self.host_status = host_status
        self._allocation_records = allocation_records((device_status, host_status))
        self._borrower = None
        self._closed = False

    @property
    def borrower(self):
        return self._borrower

    @property
    def allocation_records(self):
        return self._allocation_records

    def attach(self, owner):
        from renormalizer.backend._distributed.async_owner import AsyncResourceOwner

        if self._closed:
            raise RuntimeError("working-set status workspace is closed")
        if not isinstance(owner, AsyncResourceOwner):
            raise TypeError("status workspace requires an AsyncResourceOwner")
        if self._borrower is not None:
            raise RuntimeError("working-set status workspace is already borrowed")
        for array in (self.device_status, self.host_status):
            if not any(retained is array for retained in owner.arrays):
                raise RuntimeError("status workspace array is not owned by the call")
        owner.capture_resources(self)
        self._borrower = owner

        def release():
            if self._borrower is owner:
                self._borrower = None

        owner.add_release_callback(release)

    def close(self):
        if self._closed:
            return
        borrower = self._borrower
        if borrower is not None and not borrower.quarantined:
            raise RuntimeError("working-set status workspace is still borrowed")
        self.device_status = None
        self.host_status = None
        self._allocation_records = ()
        self._closed = True


class _OperatorCall:
    def __init__(self, owner, scope, status_workspace, *, entry_error=None):
        self.owner = owner
        self.scope = scope
        self.status_workspace = status_workspace
        self.entry_error = entry_error
        self.primary_error = entry_error
        self._secondary_errors = []
        self.disposition = "RECOVERABLE"

    @property
    def secondary_errors(self):
        return tuple(self._secondary_errors)

    def record_primary(self, error):
        if not isinstance(error, BaseException):
            raise TypeError("operator primary failure must be an exception")
        if self.primary_error is None:
            self.primary_error = error
        return self.primary_error

    def record_secondary(self, error):
        if not isinstance(error, BaseException):
            raise TypeError("operator secondary failure must be an exception")
        if error is not self.primary_error and all(
            retained is not error for retained in self._secondary_errors
        ):
            self._secondary_errors.append(error)

    def mark_communicator_fatal(self, error):
        primary = self.record_primary(error)
        self.disposition = "FATAL"
        return primary


class WorkingSetLease:
    """Receipt-bound outer lease spanning one complete rank-local solve."""

    residency_policy = "active_working_set"
    provider_role = "working_set"

    def __init__(
        self,
        provider,
        request,
        plan,
        store,
        receipt,
        store_reservation,
        cache_reservation,
        pool,
        scheduler,
        status_workspace,
        entries,
        current_lookup,
        future_identities,
        *,
        profile_enabled,
        timer,
        peak_sampler,
    ):
        self._provider = provider
        self.request = request
        self.plan = plan
        self.store = store
        self.receipt = receipt
        self.context = provider.runtime.context
        self.backend = provider.runtime.backend
        self._store_reservation = store_reservation
        self._cache_reservation = cache_reservation
        self.pool = pool
        self.scheduler = scheduler
        self._status_workspace = status_workspace
        self._entries = entries
        self._current_lookup = current_lookup
        self._future_queue = list(future_identities)
        self._prefetch_tickets = []
        self.cache_identities = tuple(entries)
        self._children = set()
        self._active_operator_owner = None
        self._active_operator_scope = None
        self._dirty = None
        self._dirty_allocation_records = ()
        self._writeback_ticket = None
        self._writeback_accounted = False
        self._writeback_invalidated = False
        self._poisoned_error = None
        self.metrics = WorkingSetMetrics()
        self._closing = False
        self._closed = False
        self._profile_enabled = profile_enabled
        self._timer = timer
        self._started_at = timer() if profile_enabled else None
        self._peak_sampler = peak_sampler
        self._baseline_device_bytes = 0
        self._baseline_host_bytes = 0
        self._observed_device_peak_bytes = 0
        self._observed_host_peak_bytes = 0
        if profile_enabled:
            (
                self._baseline_device_bytes,
                self._baseline_host_bytes,
            ) = peak_sampler()
            self._observed_host_peak_bytes = (
                request.store_bytes[self.context.rank] + pool.allocated_bytes
            )

    def __enter__(self):
        if self._closed:
            raise RuntimeError("working-set lease is closed")
        return self

    @property
    def allocation_records(self):
        workspace = self._status_workspace
        status_records = () if workspace is None else workspace.allocation_records
        return merge_allocation_records(status_records, self._dirty_allocation_records)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is None:
            self.close()
        else:
            try:
                self.close()
            except BaseException:
                pass
        return False

    def _handoff_published_terminal(self):
        provider = self._provider
        if provider is None:
            return
        runtime = provider.runtime
        terminal = None if runtime is None else runtime._terminal_error
        if terminal is None:
            terminal = provider._terminal_error
        if terminal is None:
            terminal = getattr(self.backend, "_execution_terminal_error", None)
        if terminal is None:
            return
        if provider._terminal_error is None:
            provider._terminal_error = terminal
        if runtime is not None and runtime._terminal_error is None:
            runtime._terminal_error = terminal
        try:
            provider.close()
        except BaseException as error:
            if error is terminal:
                raise
            raise terminal
        raise terminal

    def validate_setup(self, distributed_plan, source_bindings, context):
        if self._closed or self._closing:
            raise RuntimeError("working-set lease is closing")
        if distributed_plan is not self.request.distributed_plan:
            raise ValueError("distributed plan does not match working-set lease")
        if context != self.context:
            raise ValueError("context does not match working-set lease")
        if not isinstance(source_bindings, ExecutionBindings):
            raise TypeError("source_bindings must be ExecutionBindings")
        expected = {
            ref.key
            for ref in distributed_plan.execution_plan.inputs
            if ref.key != distributed_plan.variable_key
        }
        if set(source_bindings.arrays) != expected:
            raise ValueError("active source binding coverage is incomplete")

    def validate_request(self, request, residency_plan=None):
        if self._closed or self._closing:
            raise RuntimeError("working-set lease is closing")
        if not isinstance(request, OperandRequest):
            raise TypeError("request must be an OperandRequest")
        if residency_plan is not None and residency_plan is not self.plan:
            raise ValueError("operand residency plan identity does not match")
        if (
            request.residency_request is not self.request
            or request.residency_plan is not self.plan
        ):
            raise ValueError("operand request is not bound to this working set")
        if request.context != self.context:
            raise ValueError("operand context does not match working-set lease")
        if request.distributed_plan is not self.request.distributed_plan:
            raise ValueError(
                "operand distributed plan does not match working-set lease"
            )
        block = self.plan.world_size and request.distributed_plan.block_plan(
            self.context.rank, request.source_rank
        )
        if request.execution_plan is not block.execution_plan:
            raise ValueError("operand execution plan identity does not match")
        for key, local_slice in request.operand_slices.items():
            if key == request.distributed_plan.variable_key:
                continue
            ref = dict(self.request.host_refs)[key]
            lookup = (
                request.source_rank,
                key,
                _placement_ranges(local_slice, ref.shape),
            )
            if lookup not in self._current_lookup:
                raise ValueError("operand placement is outside the working-set plan")

    def _wait_for_staging(self):
        if self.pool.pending_bytes:
            started = self._timer() if self._profile_enabled else None
            self.pool.wait_for_slot()
            self.scheduler.reap_completed()
            if started is not None:
                self.metrics.prefetch_wait_s += self._timer() - started

    def _observe_peaks(self):
        if not self._profile_enabled:
            return
        device_bytes, host_bytes = self._peak_sampler()
        self._observed_device_peak_bytes = max(
            self._observed_device_peak_bytes,
            max(0, device_bytes - self._baseline_device_bytes),
        )
        self._observed_host_peak_bytes = max(
            self._observed_host_peak_bytes,
            self.request.store_bytes[self.context.rank]
            + self.pool.allocated_bytes
            + max(0, host_bytes - self._baseline_host_bytes),
        )

    @contextmanager
    def _operator_call(self, local_vector):
        if self._closed or self._closing:
            raise RuntimeError("working-set lease is closing")
        self._raise_if_poisoned()
        if self._active_operator_owner is not None:
            raise RuntimeError("working-set operator call is already active")
        prior_owner = self._status_workspace.borrower
        prior_error = None
        if prior_owner is not None:
            try:
                if not prior_owner.reap():
                    prior_owner.wait()
            except BaseException as error:
                if prior_owner.state == "detached":
                    prior_error = error
                else:
                    raise prior_owner.error
        from renormalizer.backend._execution.executor import ExecutionAllocationScope

        handle = None
        owner = None
        try:
            handle = self.scheduler.begin_compute(
                arrays=(
                    self._status_workspace.device_status,
                    self._status_workspace.host_status,
                ),
                resources=(local_vector,),
            )
            owner = handle.owner
            self._status_workspace.attach(owner)
            scope = ExecutionAllocationScope(self.backend, owner=owner)
            call = _OperatorCall(
                owner,
                scope,
                self._status_workspace,
                entry_error=prior_error,
            )
            owner._operator_call = call
        except BaseException as error:
            primary = prior_error if prior_error is not None else error
            if owner is not None:
                secondary = () if error is primary else (error,)
                owner.force_quarantine(primary, secondary_errors=secondary)
            self._provider.runtime._enter_communicator_fatal(primary, owner)
            raise primary
        self._active_operator_owner = owner
        self._active_operator_scope = scope
        try:
            with scope:
                yield call
        except BaseException as error:
            primary = call.record_primary(error)
            if call.disposition == "FATAL":
                owner.force_quarantine(primary, secondary_errors=call.secondary_errors)
                self._provider.runtime._enter_communicator_fatal(primary, owner)
            else:
                try:
                    owner.fail(primary, secondary_errors=call.secondary_errors)
                except BaseException:
                    pass
                raise
            raise primary
        else:
            try:
                self.scheduler.record_compute_completion(handle=handle)
            except BaseException as error:
                call.record_primary(error)
                if owner.state not in {"detached", "quarantined"}:
                    try:
                        owner.fail(
                            call.primary_error,
                            secondary_errors=call.secondary_errors,
                        )
                    except BaseException:
                        pass
                raise
        finally:
            self._active_operator_owner = None
            self._active_operator_scope = None

    def _load_identity(self, identity, *, prefetch=False):
        self._raise_if_poisoned()
        try:
            lease = self._provider.cache.acquire(identity)
        except BaseException as error:
            self._poison(error)
            raise
        if lease.cache_hit:
            self.metrics.cache_hits += 1
            if prefetch:
                lease.close()
                return None
            try:
                lease.wait_for_ready(self.scheduler.wait_for_h2d)
            except BaseException as error:
                lease.close()
                self._poison(error)
                raise
            return lease
        self.metrics.cache_misses += 1
        spec, ref, local_slice = self._entries[identity]
        ticket = None
        try:
            self._wait_for_staging()
            with self.pool.checkout(spec.nbytes) as slot:
                ticket = self.scheduler.stage_h2d(
                    TransferSource(ref, local_slice, lease.reverse_axis),
                    lease.transfer_array,
                    slot,
                    cache_lease=lease,
                )
            lease.install_readiness(ticket)
            if not prefetch:
                lease.wait_for_ready(self.scheduler.wait_for_h2d)
        except BaseException as error:
            first_error = error
            terminal = self.scheduler.poisoned or (
                ticket is not None and ticket.terminal_poisoned
            )
            if ticket is not None and not terminal:
                try:
                    ticket.wait()
                except BaseException:
                    pass
            if not terminal:
                try:
                    if lease.state not in {"closed", "failed"}:
                        lease.fail(first_error)
                finally:
                    self._poison(first_error)
            else:
                self._poison(first_error)
            raise
        self.metrics.record_h2d(spec.nbytes)
        if prefetch:
            if ticket.completed:
                lease.close()
            else:
                lease.transfer_to(ticket.owner)
            self._prefetch_tickets.append(ticket)
            return None
        return lease

    def _poison(self, error):
        if self._poisoned_error is None:
            self._poisoned_error = error

    def _poison_status_collective(self, error):
        self._poison(error)
        self._provider._poison_terminal(error)

    def _enter_communicator_fatal(self, error, owner=None):
        runtime = self._provider.runtime
        with runtime._communicator_fatal_reservation(error) as (primary, _):
            self._poison(primary)
            self._provider._poison_terminal(primary)
            return runtime._enter_communicator_fatal(primary, owner)

    def _raise_if_poisoned(self):
        if self._poisoned_error is not None:
            raise RuntimeError(
                "working-set lease is poisoned"
            ) from self._poisoned_error

    def _prefetch_one(self):
        while self._future_queue:
            identity = self._future_queue.pop(0)
            if self._provider.cache.contains(identity):
                continue
            self._load_identity(identity, prefetch=True)
            break

    def acquire(self, request):
        self.validate_request(request, self.plan)
        arrays = {}
        leases = []
        child = None
        variable_key = request.distributed_plan.variable_key
        call_owner = self._active_operator_owner
        try:
            for ref in request.execution_plan.inputs:
                if ref.key == variable_key:
                    arrays[ref.key] = request.broadcast_variable
                    _validate_array(
                        self.backend,
                        request.broadcast_variable,
                        ref,
                        "active broadcast variable",
                    )
                    continue
                host_ref = dict(self.request.host_refs)[ref.key]
                lookup = (
                    request.source_rank,
                    ref.key,
                    _placement_ranges(request.operand_slices[ref.key], host_ref.shape),
                )
                identity = self._current_lookup[lookup]
                cache_lease = self._load_identity(identity)
                leases.append(cache_lease)
                if call_owner is not None:
                    cache_lease.transfer_to(call_owner)
                arrays[ref.key] = cache_lease.array
                _validate_array(
                    self.backend, cache_lease.array, ref, "active static operand"
                )
            bindings = ExecutionBindings(arrays)
            bindings.validate_for(request.execution_plan)
            if call_owner is not None:
                call_owner.capture_resources(bindings)
                call_owner.capture_arrays(*bindings.arrays.values())
            self._prefetch_one()
            self._observe_peaks()
            compute_handle = None
            if call_owner is None:
                compute_handle = self.scheduler.begin_compute(
                    cache_leases=leases,
                    bindings=bindings,
                )
            child = _ActiveOperandLease(
                self,
                bindings,
                leases,
                compute_handle,
                shared_call=call_owner is not None,
            )
            self._children.add(child)
            return child
        except BaseException:
            if child is not None:
                try:
                    child.close()
                except BaseException:
                    pass
            else:
                for lease in leases:
                    try:
                        lease.close()
                    except BaseException:
                        pass
            raise

    def mark_dirty(self, key, local_array):
        if self._closed or self._closing:
            raise RuntimeError("working-set lease is closing")
        self._raise_if_poisoned()
        output = self.request.distributed_plan.execution_plan.output
        if key != output.key:
            raise ValueError("only the authorized distributed output may be dirty")
        ref = dict(self.request.host_refs)[key]
        allocation = self.request.writeback_allocations[self.context.rank][0]
        if allocation.key != key:
            raise ValueError("dirty allocation key does not match host output")
        _validate_dirty_array(self.backend, local_array, allocation)
        if self._dirty is not None:
            retained_ref, retained_array = self._dirty
            if retained_ref != ref or retained_array is not local_array:
                raise RuntimeError("working-set lease already has a dirty result")
        else:
            self._dirty_allocation_records = allocation_records((local_array,))
            self._dirty = (ref, local_array)
        self._observe_peaks()

    def reap_completed(self):
        error = None
        try:
            self.scheduler.reap_completed()
        except BaseException as caught:
            error = caught
        try:
            self.pool.reap_completed()
        except BaseException as caught:
            if error is None:
                error = caught
        try:
            self._provider.cache.reap_completed()
        except BaseException as caught:
            if error is None:
                error = caught
        if error is not None:
            self._poison(error)
            raise error

    def _schedule_writeback(self):
        if self._dirty is None or self._writeback_ticket is not None:
            return
        ref, array = self._dirty
        self._wait_for_staging()
        with self.pool.checkout(int(array.nbytes)) as slot:
            self._writeback_ticket = self.scheduler.writeback_d2h(array, ref, slot)

    def _emit_profile(self, planned_cache_bytes, pool):
        if not self._profile_enabled:
            return
        from renormalizer.utils import profiling

        rank = self.context.rank
        estimate = self.plan.rank_estimates[rank]
        metrics = self.metrics
        observed_cache_bytes = self._cache_reservation.peak_allocated_bytes
        payload = {
            "h2d_bytes": self.scheduler.h2d_bytes,
            "d2h_bytes": self.scheduler.d2h_bytes,
            "h2d_s": self.scheduler.h2d_s,
            "d2h_s": self.scheduler.d2h_s,
            "cache_hits": metrics.cache_hits,
            "cache_misses": metrics.cache_misses,
            "prefetch_overlap_s": metrics.prefetch_overlap_s,
            "prefetch_wait_s": metrics.prefetch_wait_s,
            "dirty_writeback_bytes": metrics.dirty_writeback_bytes,
            "dirty_writeback_s": metrics.dirty_writeback_s,
            "dirty_writeback_count": metrics.dirty_writeback_count,
            "peak_device_bytes": observed_cache_bytes,
            "planned_device_peak_bytes": self.plan.device_peak_bytes[rank],
            "observed_device_peak_bytes": self._observed_device_peak_bytes,
            "planned_host_peak_bytes": self.plan.host_peak_bytes[rank],
            "observed_host_peak_bytes": self._observed_host_peak_bytes,
            "planned_cache_peak_bytes": planned_cache_bytes,
            "observed_cache_peak_bytes": observed_cache_bytes,
            "planned_pinned_peak_bytes": self.request.transfer_profile.rank_staging_bytes[
                rank
            ],
            "observed_pinned_peak_bytes": pool.peak_checked_out_bytes,
            "pageable_fallback_count": pool.pageable_fallback_count,
            "pageable_fallback_bytes": pool.pageable_fallback_bytes,
            "full_replica": metrics.full_replica,
            "request_hash": self.request.request_hash,
            "plan_hash": self.plan.plan_hash,
            "profile_hash": self.request.transfer_profile.profile_hash,
            "rank": rank,
            "device": str(self.backend.current_device()),
            "wall_s": self._timer() - self._started_at,
        }
        profiling.record("working_set_transfer", **payload)

    def close(self, wait=True):
        if type(wait) is not bool:
            raise TypeError("wait must be a boolean")
        if self._closed:
            return
        self._handoff_published_terminal()
        self._closing = True
        if not wait:
            try:
                self._schedule_writeback()
                self.reap_completed()
                return
            except BaseException as caught:
                self._poison(caught)

        error = self._poisoned_error
        old_ref = None if self._dirty is None else self._dirty[0]
        planned_cache_bytes = self._cache_reservation.required_bytes
        pool = self.pool
        scheduler = self.scheduler
        status_workspace = self._status_workspace
        provider = self._provider
        cache_reservation = self._cache_reservation
        store_reservation = self._store_reservation

        if error is None:
            try:
                self._schedule_writeback()
            except BaseException as caught:
                error = caught

        for child in tuple(self._children):
            try:
                child.close()
            except BaseException as caught:
                if error is None:
                    error = caught
        try:
            self._observe_peaks()
        except BaseException as caught:
            if error is None:
                error = caught
        try:
            scheduler.complete_all()
        except BaseException as caught:
            if error is None:
                error = caught
        try:
            provider.cache.wait_for_pending()
        except BaseException as caught:
            if error is None:
                error = caught
        try:
            pool.reap_completed()
        except BaseException as caught:
            if error is None:
                error = caught

        self.metrics.prefetch_overlap_s += sum(
            ticket.elapsed_s for ticket in self._prefetch_tickets if ticket.completed
        )

        if (
            self._writeback_ticket is not None
            and self._writeback_ticket.completed
            and not self._writeback_accounted
        ):
            try:
                updated = self._writeback_ticket.take_result()
                if updated is None:
                    raise RuntimeError("dirty writeback completed without a CAS result")
                self.metrics.dirty_writeback_count += 1
                self.metrics.dirty_writeback_bytes += old_ref.nbytes
                self.metrics.dirty_writeback_s += self._writeback_ticket.elapsed_s
                self._writeback_accounted = True
            except BaseException as caught:
                if error is None:
                    error = caught
        if self._writeback_accounted and not self._writeback_invalidated:
            try:
                provider.cache.invalidate_ref(old_ref)
                self._writeback_invalidated = True
            except BaseException as caught:
                if error is None:
                    error = caught

        try:
            self._emit_profile(planned_cache_bytes, pool)
        except BaseException as caught:
            if error is None:
                error = caught
        try:
            status_workspace.close()
        except BaseException as caught:
            if error is None:
                error = caught
        try:
            scheduler.close()
        except BaseException as caught:
            if error is None:
                error = caught
        try:
            pool.close()
        except BaseException as caught:
            if error is None:
                error = caught
        try:
            cache_reservation.close()
        except BaseException as caught:
            if error is None:
                error = caught
        try:
            store_reservation.close()
        except BaseException as caught:
            if error is None:
                error = caught

        terminal_resources = tuple(
            resource
            for resource in (scheduler, pool, provider._cache)
            if resource is not None and getattr(resource, "poisoned", False)
        )
        if terminal_resources:
            provider._retain_terminal_resources(*terminal_resources)
        try:
            provider._lease_closed(self)
        except BaseException as caught:
            if error is None:
                error = caught
        finally:
            provider.last_compute_event = None
            self._children.clear()
            self._active_operator_owner = None
            self._active_operator_scope = None
            self._dirty = None
            self._dirty_allocation_records = ()
            self._writeback_ticket = None
            self._poisoned_error = error
            self._entries = {}
            self._current_lookup = {}
            self._future_queue = []
            self._prefetch_tickets = []
            self.cache_identities = ()
            self._cache_reservation = None
            self._store_reservation = None
            self.pool = None
            self.scheduler = None
            self._status_workspace = None
            self.store = None
            self.request = None
            self.plan = None
            self.receipt = None
            self.context = None
            self.backend = None
            self._provider = None
            self._peak_sampler = None
            self._timer = None
            self._started_at = None
            self._profile_enabled = False
            self._closing = False
            self._closed = True
        if error is not None:
            raise error


class ActiveWorkingSetProvider:
    """Runtime-scoped factory for exact, bounded outer working-set leases."""

    residency_policy = "active_working_set"
    provider_role = "factory"

    def __init__(
        self,
        runtime,
        *,
        device_budget_resolution,
        host_budget_resolution,
        prefetch_depth=1,
        cache_factory=DeviceTensorCache,
        pool_factory=PinnedBufferPool,
        scheduler_factory=TransferScheduler,
        event_factory=None,
    ):
        if getattr(runtime, "_closed", True):
            raise RuntimeError("active provider requires an open runtime")
        if type(prefetch_depth) is not int or prefetch_depth <= 0:
            raise ValueError("prefetch_depth must be a positive integer")
        self.runtime = runtime
        self.device_budget_resolution = device_budget_resolution
        self.host_budget_resolution = host_budget_resolution
        self.prefetch_depth = prefetch_depth
        self._cache_factory = cache_factory
        self._pool_factory = pool_factory
        self._scheduler_factory = scheduler_factory
        self._event_factory = event_factory
        self._cache = None
        self._active_lease = None
        self._terminal_resources = ()
        self._terminal_error = None
        self._terminal_quarantine = runtime._terminal_quarantine
        self._last_authorized_cache_bytes = 0
        self._closed = False
        self.metrics = WorkingSetMetrics()
        self.last_compute_event = None
        existing = getattr(runtime, "_active_provider", None)
        if existing is not None and existing is not self:
            raise RuntimeError("runtime already owns an active working-set provider")
        runtime._active_provider = self

    @property
    def closed(self):
        return self._closed

    @property
    def terminal_poisoned(self):
        return self._terminal_error is not None

    @property
    def cache(self):
        if self._cache is None:
            raise RuntimeError("active provider cache has not been allocated")
        return self._cache

    def _poison_terminal(self, error):
        if self._terminal_error is None:
            self._terminal_error = error
        runtime = self.runtime
        if runtime is not None and runtime._terminal_error is None:
            runtime._terminal_error = error

    def matches_config(self, device_budget, host_budget, prefetch_depth):
        return (
            device_budget == self.device_budget_resolution
            and host_budget == self.host_budget_resolution
            and prefetch_depth == self.prefetch_depth
        )

    def acquire(self, request):
        raise RuntimeError("factory provider cannot acquire operand blocks directly")

    def _allocate_status_device(self):
        backend = self.runtime.backend
        if backend.name == "numpy":
            return backend.empty((1,), dtype=np.int32, order="C")
        cupy = backend._cupy
        nbytes = np.dtype(np.int32).itemsize
        with cupy.cuda.Device(backend._device_index):
            memory = cupy.cuda.Memory(nbytes)
            pointer = cupy.cuda.MemoryPointer(memory, 0)
            return cupy.ndarray((1,), dtype=np.int32, memptr=pointer, order="C")

    @staticmethod
    def _allocate_status_host():
        return np.empty((1,), dtype=np.int32, order="C")

    def _provision_status_workspace(self):
        collective = self.runtime.collective
        self.runtime._arm_communicator_fatal()
        bootstrap_fatal = getattr(collective, "_bootstrap_fatal_control", None)
        if not callable(bootstrap_fatal):
            raise RuntimeError("collective does not provide fatal bootstrap")
        if bootstrap_fatal():
            raise RuntimeError("collective fatal capability bootstrap failed")
        bootstrap = getattr(collective, "_bootstrap_status_or", None)
        if not callable(bootstrap):
            error = RuntimeError(
                "collective does not provide status-workspace bootstrap"
            )
            if self._terminal_error is None:
                self._terminal_error = error
            if self.runtime._terminal_error is None:
                self.runtime._terminal_error = error
            raise error
        validate = getattr(collective, "_validate_bootstrap_status_or", None)
        try:
            if callable(validate):
                validate()
        except BaseException as error:
            if self._terminal_error is None:
                self._terminal_error = error
            if self.runtime._terminal_error is None:
                self.runtime._terminal_error = error
            raise

        device_status = None
        host_status = None
        local_error = None
        local_code = 0
        try:
            device_status = self._allocate_status_device()
        except BaseException as error:
            local_error = error
            local_code |= 1
        try:
            host_status = self._allocate_status_host()
        except BaseException as error:
            if local_error is None:
                local_error = error
            local_code |= 2

        try:
            aggregate_code = bootstrap(local_code)
        except BaseException as error:
            device_status = None
            host_status = None
            if self._terminal_error is None:
                self._terminal_error = error
            if self.runtime._terminal_error is None:
                self.runtime._terminal_error = error
            raise
        if aggregate_code:
            device_status = None
            host_status = None
            if local_error is not None:
                raise ValueError(
                    "working-set status allocation failed"
                ) from local_error
            raise ValueError("working-set status allocation failed")
        return _LeaseStatusWorkspace(device_status, host_status)

    def _empty_cache_array(self, spec, *, order):
        backend = self.runtime.backend
        if backend.name == "numpy":
            return backend.empty(spec.shape, dtype=np.dtype(spec.dtype), order=order)
        cupy = backend._cupy
        with cupy.cuda.Device(backend._device_index):
            memory = cupy.cuda.Memory(spec.nbytes)
            pointer = cupy.cuda.MemoryPointer(memory, 0)
            return cupy.ndarray(
                spec.shape,
                dtype=np.dtype(spec.dtype),
                memptr=pointer,
                order=order,
            )

    def _cache_allocator(self, spec):
        if spec.layout != "strided":
            return self._empty_cache_array(spec, order=spec.layout)
        try:
            reverse_axis = next(
                axis
                for axis in reversed(range(len(spec.shape)))
                if spec.shape[axis] > 1
            )
        except StopIteration as error:
            raise ValueError(
                "a singleton cache entry cannot realize strided layout"
            ) from error
        transfer_array = self._empty_cache_array(spec, order="C")
        selected = [slice(None)] * len(spec.shape)
        selected[reverse_axis] = slice(None, None, -1)
        array = transfer_array[tuple(selected)]
        return CacheAllocation(array, transfer_array, reverse_axis)

    def _pinned_allocator(self, nbytes):
        backend = self.runtime.backend
        if backend.name == "numpy":
            return np.empty(nbytes, dtype=np.uint8)
        cupy = backend._cupy
        with cupy.cuda.Device(backend._device_index):
            memory = cupy.cuda.PinnedMemory(nbytes)
            pointer = cupy.cuda.PinnedMemoryPointer(memory, 0)
        return np.frombuffer(pointer, dtype=np.uint8, count=nbytes)

    def _entries(self, request, plan):
        rank = self.runtime.rank
        device = str(self.runtime.backend.current_device())
        entries = {}
        current_lookup = {}
        future_identities = []

        def add(refs, placements, *, current):
            refs_by_key = {ref.key: ref for ref in refs}
            for placement in placements:
                if placement.rank != rank:
                    continue
                ref = refs_by_key[placement.key]
                identity = _cache_identity(ref, placement, device)
                shape = tuple(value.stop - value.start for value in placement.ranges)
                spec = CacheEntrySpec(
                    identity=identity,
                    shape=shape,
                    dtype=ref.dtype,
                    layout=placement.layout,
                    nbytes=placement.nbytes,
                )
                retained = (spec, ref, _slice_from_ranges(placement.ranges))
                if identity in entries and entries[identity] != retained:
                    raise ValueError(
                        "cache identity has inconsistent retained metadata"
                    )
                entries[identity] = retained
                if current:
                    lookup = (
                        placement.source_rank,
                        placement.key,
                        identity[6],
                    )
                    if lookup in current_lookup and current_lookup[lookup] != identity:
                        raise ValueError("current operand placement is ambiguous")
                    current_lookup[lookup] = identity
                elif identity not in future_identities:
                    future_identities.append(identity)

        add(plan.current_refs, plan.local_slices, current=True)
        for future in plan.future_plans[: plan.prefetch_depth]:
            add(future.required_refs, future.local_slices, current=False)
        return entries, current_lookup, future_identities

    def open_working_set(self, request, plan, store, receipt):
        if self._terminal_error is not None:
            raise RuntimeError(
                "active working-set provider is terminal-poisoned"
            ) from self._terminal_error
        if self._closed:
            raise RuntimeError("active working-set provider is closed")
        if self._active_lease is not None:
            raise RuntimeError(
                "overlapping outer working-set leases are not authorized"
            )
        if not isinstance(request, ResidencyRequest):
            raise TypeError("request must be a ResidencyRequest")
        if not isinstance(plan, ResidencyPlan):
            raise TypeError("plan must be a ResidencyPlan")
        if not isinstance(store, HostTensorStore):
            raise TypeError("store must be a HostTensorStore")
        if not isinstance(receipt, ResidencyPreflightReceipt):
            raise TypeError("receipt must be a ResidencyPreflightReceipt")
        if not self.matches_config(
            request.device_budget, request.host_budget, request.prefetch_depth
        ):
            raise ValueError("request budgets do not match active provider config")
        plan.validate_request(request)
        plan.runtime_identity.validate_runtime(
            self.runtime.context, self.runtime.mesh, self.runtime.backend
        )
        plan.validate_store(store)
        entries, current_lookup, future_identities = self._entries(request, plan)
        allowlist = {identity: retained[0] for identity, retained in entries.items()}
        required_bytes = sum(spec.nbytes for spec in allowlist.values())
        estimate = plan.rank_estimates[self.runtime.rank]
        if (
            required_bytes
            != estimate.current_static_bytes + estimate.prefetched_static_bytes
        ):
            raise ValueError("cache reservation does not match residency plan")

        self.runtime.consume_residency_receipt(receipt, request, plan)
        status_workspace = None
        store_reservation = None
        cache_reservation = None
        pool = None
        scheduler = None
        try:
            status_workspace = self._provision_status_workspace()
            dirty_key = request.distributed_plan.execution_plan.output.key
            dirty_ref = dict(request.host_refs)[dirty_key]
            store_reservation = store.reserve(
                request.store_snapshots[self.runtime.rank],
                dirty_ref=dirty_ref,
            )
            if self._cache is None:
                self._cache = self._cache_factory(
                    self.device_budget_resolution.resolved_bytes,
                    allocator=self._cache_allocator,
                )
            cache_reservation = self._cache.reserve(allowlist, required_bytes)
            pool = self._pool_factory(
                request.transfer_profile.rank_staging_bytes[self.runtime.rank],
                pinned_allocator=self._pinned_allocator,
            )
            from renormalizer.utils.log import PROFILING, get_logger

            profile_enabled = bool(get_logger().isEnabledFor(PROFILING))
            timer = None
            if profile_enabled:
                from time import perf_counter

                timer = perf_counter
            scheduler = self._scheduler_factory(
                store,
                self.runtime.backend,
                pool,
                reservation=store_reservation,
                event_factory=self._event_factory,
                profile_enabled=profile_enabled,
                quarantine=self._accept_async_quarantine,
            )
            peak_sampler = None
            if profile_enabled:
                import psutil

                process = psutil.Process()

                def peak_sampler():
                    if self.runtime.backend.name == "cupy":
                        with self.runtime.backend._cupy.cuda.Device(
                            self.runtime.backend._device_index
                        ):
                            (
                                free_bytes,
                                total_bytes,
                            ) = self.runtime.backend._cupy.cuda.runtime.memGetInfo()
                        device_bytes = int(total_bytes) - int(free_bytes)
                    else:
                        device_bytes = self._cache.allocated_bytes
                    return device_bytes, int(process.memory_info().rss)

            lease = WorkingSetLease(
                self,
                request,
                plan,
                store,
                receipt,
                store_reservation,
                cache_reservation,
                pool,
                scheduler,
                status_workspace,
                entries,
                current_lookup,
                future_identities,
                profile_enabled=profile_enabled,
                timer=timer,
                peak_sampler=peak_sampler,
            )
        except BaseException as error:
            first_error = error
            if scheduler is not None:
                try:
                    scheduler.close()
                except BaseException:
                    pass
            if pool is not None:
                try:
                    pool.close()
                except BaseException:
                    pass
            if cache_reservation is not None:
                try:
                    cache_reservation.close()
                except BaseException:
                    pass
            if store_reservation is not None:
                try:
                    store_reservation.close()
                except BaseException:
                    pass
            if status_workspace is not None:
                try:
                    status_workspace.close()
                except BaseException:
                    pass
            raise first_error
        self._active_lease = lease
        self._last_authorized_cache_bytes = required_bytes
        lease.metrics.full_replica = bool(plan.metadata()["full_replica_prediction"])
        lease.metrics.pageable_fallback_count = pool.pageable_fallback_count
        lease.metrics.pageable_fallback_bytes = pool.pageable_fallback_bytes
        self.metrics = lease.metrics
        return lease

    def _lease_closed(self, lease):
        if self._active_lease is lease:
            self._active_lease = None

    def _retain_terminal_resources(self, *resources):
        retained = list(self._terminal_resources)
        for resource in resources:
            if resource is not None and resource not in retained:
                retained.append(resource)
        self._terminal_resources = tuple(retained)
        self._terminal_quarantine.retain_resources(resources)
        runtime = self.runtime
        if runtime is not None:
            runtime._terminal_quarantine.retain_resources(resources)

    def _accept_async_quarantine(self, owner):
        error = owner.error
        lease = self._active_lease
        scheduler = None if lease is None else lease.scheduler
        cohort = (owner,) if scheduler is None else tuple(scheduler._quarantined_owners)
        if all(retained is not owner for retained in cohort):
            cohort = (*cohort, owner)
        runtime = self.runtime
        boundary = (
            runtime._communicator_fatal_reservation(error)
            if runtime is not None
            else nullcontext((error, None))
        )
        with boundary as (primary, _):
            for retained_owner in cohort:
                if self._terminal_error is None:
                    self._terminal_error = primary
                self._terminal_quarantine.retain(retained_owner, primary)
                if lease is not None and lease._poisoned_error is None:
                    lease._poisoned_error = primary
                if runtime is not None:
                    try:
                        runtime._accept_async_quarantine(retained_owner, primary)
                    except BaseException as callback_error:
                        retained_owner._remember_secondary(callback_error)
                owns_cache_lease = any(
                    type(resource) is CacheEntryLease and resource._cache is self._cache
                    for resource in retained_owner.resources
                )
                if self._cache is not None and owns_cache_lease:
                    self._cache._poison(primary)
                owns_pool_slot = lease is not None and any(
                    type(resource) is StagingSlot and resource._pool is lease.pool
                    for resource in retained_owner.resources
                )
                if owns_pool_slot and lease.pool is not None:
                    lease.pool._poison(primary)
            if runtime is not None:
                runtime._enter_communicator_fatal(primary, owner)

    def resource_state(self):
        cache = self._cache
        lease = self._active_lease
        pool = None if lease is None else lease.pool
        state = {
            "active_leases": int(lease is not None),
            "reserved_cache_bytes": 0 if cache is None else cache.reserved_bytes,
            "retained_cache_bytes": 0 if cache is None else cache.allocated_bytes,
            "hard_cache_capacity_bytes": self.device_budget_resolution.resolved_bytes,
            "authorized_cache_bytes": self._last_authorized_cache_bytes,
            "checked_out_pinned_bytes": 0 if pool is None else pool.checked_out_bytes,
            "pending_pinned_bytes": 0 if pool is None else pool.pending_bytes,
        }
        if self._terminal_quarantine.poisoned:
            state.update(self._terminal_quarantine.resource_state())
        return state

    def runtime_resource_state(self):
        cache = self._cache
        lease = self._active_lease
        scheduler = None if lease is None else lease.scheduler
        state = {
            "active_leases": int(lease is not None),
            "cache_bytes": 0 if cache is None else cache.allocated_bytes,
            "cache_refs": (
                0
                if cache is None
                else sum(entry.refcount for entry in cache._entries.values())
            ),
            "pinned_bytes": 0 if lease is None else lease.pool.allocated_bytes,
            "stream_count": 0 if scheduler is None else scheduler.stream_count,
            "event_count": (
                0
                if scheduler is None
                else (
                    scheduler.retained_event_count
                    if scheduler.poisoned
                    else scheduler.event_count
                )
            ),
        }
        if self._terminal_quarantine.poisoned:
            state.update(self._terminal_quarantine.resource_state())
        return state

    def close(self):
        if self._closed:
            return
        error = self._terminal_error
        runtime = self.runtime
        lease = self._active_lease
        cache = self._cache
        if error is not None and isinstance(lease, WorkingSetLease):
            terminal_resources = (lease.scheduler, lease.pool, cache, lease)
            retained_resources = list(self._terminal_resources)
            for resource in terminal_resources:
                if resource is not None and all(
                    retained is not resource for retained in retained_resources
                ):
                    retained_resources.append(resource)
            self._terminal_resources = tuple(retained_resources)
            self._retain_terminal_resources(*terminal_resources)
            store_reservation = lease._store_reservation
            if store_reservation is not None:
                try:
                    store_reservation.close()
                except BaseException as caught:
                    if error is None:
                        error = caught
                lease._store_reservation = None
        elif lease is not None:
            try:
                lease.close()
            except BaseException as caught:
                if error is None:
                    error = caught
        if cache is not None and error is None:
            try:
                cache.close()
            except BaseException as caught:
                if error is None:
                    error = caught
        try:
            if cache is not None and getattr(cache, "poisoned", False):
                self._retain_terminal_resources(cache)
        finally:
            if (
                runtime is not None
                and getattr(runtime, "_active_provider", None) is self
            ):
                runtime._active_provider = None
            self._active_lease = None
            self._cache = None
            self._cache_factory = None
            self._pool_factory = None
            self._scheduler_factory = None
            self._event_factory = None
            self.last_compute_event = None
            self.runtime = None
            if self._terminal_error is None:
                self._terminal_error = error
            self._closed = True
        if error is not None:
            raise error


__all__ = [
    "ActiveWorkingSetProvider",
    "active_working_set_policy_error",
    "DeviceResidentProvider",
    "OperandLease",
    "OperandProvider",
    "OperandRequest",
    "WorkingSetLease",
]
