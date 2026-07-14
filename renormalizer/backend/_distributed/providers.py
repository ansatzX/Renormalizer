"""Resident operand leases for distributed block execution."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import MappingProxyType
from typing import ContextManager, Protocol

import numpy as np

from renormalizer.backend._distributed.async_owner import (
    _CountedAsyncAdmission,
    _require_resource_admission,
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
from renormalizer.backend._distributed.terminal import (
    _TerminalPhase,
    _TERMINAL_TIMEOUT_S,
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


class _LeaseResourceRecord:
    """Strong, incrementally captured lease resources for query-free handoff."""

    def __init__(self, epoch):
        self.epoch = epoch
        self._resources = []
        self._consumed_receipts = {}
        self._allocations = {}
        self._cache_allocations = {}
        self._pinned_allocations = {}
        self._events = {}
        self._streams = {}

    @property
    def resources(self):
        return tuple(self._resources)

    @property
    def allocations(self):
        records = dict(self._allocations)
        self._merge(records, self._cache_allocations.values())
        self._merge(records, self._pinned_allocations.values())
        return tuple(records.values())

    @property
    def cache_allocations(self):
        return tuple(self._cache_allocations.values())

    @property
    def pinned_allocations(self):
        return tuple(self._pinned_allocations.values())

    @property
    def events(self):
        return tuple(self._events.values())

    @property
    def streams(self):
        return tuple(self._streams.values())

    @property
    def consumed_receipts(self):
        return tuple(self._consumed_receipts.values())

    @staticmethod
    def _merge(target, records):
        for record in records:
            retained = target.get(record.identity)
            if retained is None or record.capacity_bytes > retained.capacity_bytes:
                target[record.identity] = record

    def capture(
        self,
        resource=None,
        *,
        kind=None,
        records=(),
        events=(),
        streams=(),
        _replace_kind=False,
    ):
        if resource is not None and all(
            retained is not resource for retained in self._resources
        ):
            self._resources.append(resource)
        records = tuple(records)
        if resource is not None and not _replace_kind:
            records = (*records, *getattr(resource, "allocation_records", ()))
            events = (*tuple(events), *getattr(resource, "retained_events", ()))
            streams = (*tuple(streams), *getattr(resource, "retained_streams", ()))
        if kind == "cache":
            if _replace_kind:
                self._cache_allocations.clear()
            self._merge(self._cache_allocations, records)
        elif kind == "pinned":
            if _replace_kind:
                self._pinned_allocations.clear()
            self._merge(self._pinned_allocations, records)
        else:
            self._merge(self._allocations, records)
        for event in events:
            if event is not None:
                self._events[id(event)] = event
        for stream in streams:
            if stream is not None:
                self._streams[id(stream)] = stream

    def capture_consumed_receipt(self, receipt):
        self._consumed_receipts[id(receipt)] = receipt
        self.capture(receipt)

    def restore_consumed_receipts(self, runtime):
        for identity, receipt in tuple(self._consumed_receipts.items()):
            issued = runtime._issued_receipts.get(identity)
            if issued is not None and issued is not receipt:
                raise RuntimeError("consumed residency receipt identity changed")
            runtime._issued_receipts[identity] = receipt
            self._resources = [
                resource for resource in self._resources if resource is not receipt
            ]
        self._consumed_receipts.clear()

    def clear(self):
        self._resources.clear()
        self._consumed_receipts.clear()
        self._allocations.clear()
        self._cache_allocations.clear()
        self._pinned_allocations.clear()
        self._events.clear()
        self._streams.clear()

    def release(self, resource):
        self._resources = [
            retained for retained in self._resources if retained is not resource
        ]
        self._allocations.clear()
        self._cache_allocations.clear()
        self._pinned_allocations.clear()
        self._events.clear()
        self._streams.clear()
        for retained in tuple(self._resources):
            records = getattr(retained, "allocation_records", None)
            if records is None:
                records = getattr(retained, "allocations", ())
            events = getattr(retained, "retained_events", None)
            if events is None:
                events = getattr(retained, "events", ())
            streams = getattr(retained, "retained_streams", None)
            if streams is None:
                streams = getattr(retained, "streams", ())
            self.capture(
                records=tuple(records),
                kind=getattr(retained, "_terminal_resource_kind", None),
                events=tuple(events),
                streams=tuple(streams),
            )


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

    def close(self, *, _admission_token=None):
        if self._closed:
            return
        working_set = self._working_set
        working_set._raise_terminal_close_preemption()
        try:
            with working_set._child_close_admission(
                _admission_token
            ) as admitted_token:
                validator = working_set._exact_admission_validator(
                    admitted_token
                )
                return self._close_admitted(
                    admitted_token,
                    validator,
                )
        except BaseException:
            working_set._raise_terminal_close_preemption()
            raise

    def _close_admitted(self, admission_token, admission_validator):
        _require_resource_admission(
            admission_token,
            admission_validator,
        )
        working_set = self._working_set
        leases = self._cache_leases
        event = None
        error = None
        try:
            working_set._observe_peaks(
                _admission_token=admission_token,
                _admission_validator=admission_validator,
            )
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
                    _admission_token=admission_token,
                    _admission_validator=admission_validator,
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

    def close(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        _require_resource_admission(
            _admission_token,
            _admission_validator,
        )
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
        epoch,
        resource_record,
        profile_enabled,
        timer,
        peak_sampler,
    ):
        self._provider = provider
        self._epoch = epoch
        self._resource_record = resource_record
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

    @contextmanager
    def _lease_admission(self, operation, token=None):
        provider = self._provider
        if provider is None or provider.runtime is None:
            raise RuntimeError("working-set lease runtime is unavailable")
        runtime = provider.runtime
        gate = runtime._terminal_gate
        owned = False
        if token is None:
            current = gate._current_thread_admission()
            if current is None:
                token = gate.admit_lease(self._epoch, operation)
                owned = True
            else:
                token = current
        runtime._require_admission(
            token,
            scope="lease",
            epoch=self._epoch,
        )
        try:
            yield token
        finally:
            if owned:
                runtime._release_admission(token)

    @contextmanager
    def _child_close_admission(self, token=None):
        if token is not None and token.scope == "lease_close":
            runtime = self._provider.runtime
            runtime._require_admission(
                token,
                scope="lease_close",
                epoch=self._epoch,
                transition_sequence=token.transition_sequence,
            )
            yield token
            return
        with self._lease_admission("child_close", token) as admitted:
            yield admitted

    def _spawn_async_admission(self, capability, operation):
        return self._provider._spawn_async_admission(
            self._epoch,
            capability,
            operation,
        )

    def _record_resource(self, **captured):
        self._resource_record.capture(**captured)

    def _exact_admission_validator(self, token):
        runtime = self._provider.runtime
        if token.scope == "lease":
            transition_sequence = None
        elif token.scope == "lease_close":
            transition_sequence = token.transition_sequence
        else:
            raise RuntimeError(
                "working-set resource requires lease ownership"
            )
        return runtime._exact_admission_validator(
            token,
            scope=token.scope,
            epoch=self._epoch,
            parent_sequence=token.parent_sequence,
            transition_sequence=transition_sequence,
        )

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
        self._retain_terminal_lease_safely(provider, terminal)
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

    def _wait_for_staging(self, *, _admission_token, _admission_validator):
        if self.pool.pending_bytes:
            started = self._timer() if self._profile_enabled else None
            self.pool.wait_for_slot(
                _admission_token=_admission_token,
                _admission_validator=_admission_validator,
            )
            self.scheduler.reap_completed(
                _admission_token=_admission_token,
                _admission_validator=_admission_validator,
            )
            if started is not None:
                self.metrics.prefetch_wait_s += self._timer() - started

    def _raise_if_admitted_transition_preempted(self):
        runtime = self._provider.runtime
        gate = runtime._terminal_gate
        with gate._condition:
            transition = gate._fatal_transition
            if transition is not None and gate._phase in (
                _TerminalPhase.FATAL_PENDING,
                _TerminalPhase.FATAL_PUBLISHED,
                _TerminalPhase.RUNTIME_CLOSED,
            ):
                raise transition.primary
            if gate._phase is _TerminalPhase.RUNTIME_CLOSING:
                raise RuntimeError("runtime is closing")
            if gate._phase is _TerminalPhase.RUNTIME_CLOSED:
                raise RuntimeError("runtime is closed")
            lease = gate._lease_state(self._epoch)
            if lease.phase != "open":
                raise RuntimeError("lease is closing")

    def _observe_peaks(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        _require_resource_admission(
            _admission_token,
            _admission_validator,
        )
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
    def _operator_call(self, local_vector, *, _admission_token=None):
        self._raise_if_poisoned()
        with self._lease_admission(
            "operator_call", _admission_token
        ) as admission_token:
            validator = self._exact_admission_validator(admission_token)
            with self._operator_call_admitted(
                local_vector,
                _admission_token=admission_token,
                _admission_validator=validator,
            ) as call:
                yield call

    @contextmanager
    def _operator_call_admitted(
        self,
        local_vector,
        *,
        _admission_token,
        _admission_validator,
    ):
        _require_resource_admission(
            _admission_token,
            _admission_validator,
        )
        validator = _admission_validator
        self._raise_if_admitted_transition_preempted()
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
                _admission_token=_admission_token,
                _admission_validator=validator,
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
                self.scheduler.record_compute_completion(
                    handle=handle,
                    _admission_token=_admission_token,
                    _admission_validator=validator,
                )
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

    def _load_identity(
        self, identity, *, prefetch=False, _admission_token=None
    ):
        operation = "prefetch" if prefetch else "load"
        with self._lease_admission(operation, _admission_token) as admission_token:
            validator = self._exact_admission_validator(admission_token)
            return self._load_identity_admitted(
                identity,
                prefetch=prefetch,
                _admission_token=admission_token,
                _admission_validator=validator,
            )

    def _load_identity_admitted(
        self,
        identity,
        *,
        prefetch,
        _admission_token,
        _admission_validator,
    ):
        _require_resource_admission(
            _admission_token,
            _admission_validator,
        )
        validator = _admission_validator
        self._raise_if_admitted_transition_preempted()
        self._raise_if_poisoned()
        try:
            lease = self._provider.cache.acquire(
                identity,
                _admission_token=_admission_token,
                _admission_validator=validator,
            )
        except BaseException as error:
            self._poison(error)
            raise
        if lease.cache_hit:
            self.metrics.cache_hits += 1
            if prefetch:
                lease.close()
                return None
            try:
                lease.wait_for_ready(
                    lambda event: self.scheduler.wait_for_h2d(
                        event,
                        _admission_token=_admission_token,
                        _admission_validator=validator,
                    )
                )
            except BaseException as error:
                lease.close()
                self._poison(error)
                raise
            return lease
        self.metrics.cache_misses += 1
        spec, ref, local_slice = self._entries[identity]
        ticket = None
        try:
            self._wait_for_staging(
                _admission_token=_admission_token,
                _admission_validator=validator,
            )
            with self.pool.checkout(
                spec.nbytes,
                _admission_token=_admission_token,
                _admission_validator=validator,
            ) as slot:
                ticket = self.scheduler.stage_h2d(
                    TransferSource(ref, local_slice, lease.reverse_axis),
                    lease.transfer_array,
                    slot,
                    cache_lease=lease,
                    _admission_token=_admission_token,
                    _admission_validator=validator,
                )
            lease.install_readiness(ticket)
            if not prefetch:
                lease.wait_for_ready(
                    lambda event: self.scheduler.wait_for_h2d(
                        event,
                        _admission_token=_admission_token,
                        _admission_validator=validator,
                    )
                )
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

    def _prefetch_one(self, *, _admission_token=None):
        with self._lease_admission(
            "prefetch", _admission_token
        ) as admission_token:
            validator = self._exact_admission_validator(admission_token)
            return self._prefetch_one_admitted(
                _admission_token=admission_token,
                _admission_validator=validator,
            )

    def _prefetch_one_admitted(
        self,
        *,
        _admission_token,
        _admission_validator,
    ):
        _require_resource_admission(
            _admission_token,
            _admission_validator,
        )
        self._raise_if_admitted_transition_preempted()
        while self._future_queue:
            identity = self._future_queue.pop(0)
            if self._provider.cache.contains(
                identity,
                _admission_token=_admission_token,
                _admission_validator=_admission_validator,
            ):
                continue
            self._load_identity(
                identity,
                prefetch=True,
                _admission_token=_admission_token,
            )
            break

    def acquire(self, request, *, _admission_token=None):
        with self._lease_admission(
            "acquire", _admission_token
        ) as admission_token:
            validator = self._exact_admission_validator(admission_token)
            return self._acquire_admitted(
                request,
                _admission_token=admission_token,
                _admission_validator=validator,
            )

    def _acquire_admitted(
        self,
        request,
        *,
        _admission_token,
        _admission_validator,
    ):
        _require_resource_admission(
            _admission_token,
            _admission_validator,
        )
        validator = _admission_validator
        self._raise_if_admitted_transition_preempted()
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
                cache_lease = self._load_identity(
                    identity, _admission_token=_admission_token
                )
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
            self._prefetch_one(_admission_token=_admission_token)
            self._observe_peaks(
                _admission_token=_admission_token,
                _admission_validator=validator,
            )
            compute_handle = None
            if call_owner is None:
                compute_handle = self.scheduler.begin_compute(
                    cache_leases=leases,
                    bindings=bindings,
                    _admission_token=_admission_token,
                    _admission_validator=validator,
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
                    child.close(_admission_token=_admission_token)
                except BaseException:
                    pass
            else:
                for lease in leases:
                    try:
                        lease.close()
                    except BaseException:
                        pass
            raise

    def mark_dirty(self, key, local_array, *, _admission_token=None):
        with self._lease_admission(
            "mark_dirty", _admission_token
        ) as admission_token:
            validator = self._exact_admission_validator(admission_token)
            return self._mark_dirty_admitted(
                key,
                local_array,
                _admission_token=admission_token,
                _admission_validator=validator,
            )

    def _mark_dirty_admitted(
        self,
        key,
        local_array,
        *,
        _admission_token,
        _admission_validator,
    ):
        _require_resource_admission(
            _admission_token,
            _admission_validator,
        )
        validator = _admission_validator
        self._raise_if_admitted_transition_preempted()
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
            self._resource_record.capture(records=self._dirty_allocation_records)
            self._dirty = (ref, local_array)
        self._observe_peaks(
            _admission_token=_admission_token,
            _admission_validator=validator,
        )

    def reap_completed(self, *, _admission_token=None):
        with self._lease_admission(
            "reap", _admission_token
        ) as admission_token:
            validator = self._exact_admission_validator(admission_token)
            return self._reap_completed_admitted(
                _admission_token=admission_token,
                _admission_validator=validator,
            )

    def _reap_completed_admitted(
        self,
        *,
        _admission_token,
        _admission_validator,
    ):
        _require_resource_admission(
            _admission_token,
            _admission_validator,
        )
        validator = _admission_validator
        self._raise_if_admitted_transition_preempted()
        error = None
        try:
            self.scheduler.reap_completed(
                _admission_token=_admission_token,
                _admission_validator=validator,
            )
        except BaseException as caught:
            error = caught
        try:
            self.pool.reap_completed(
                _admission_token=_admission_token,
                _admission_validator=validator,
            )
        except BaseException as caught:
            if error is None:
                error = caught
        try:
            self._provider.cache.reap_completed(
                _admission_token=_admission_token,
                _admission_validator=validator,
            )
        except BaseException as caught:
            if error is None:
                error = caught
        if error is not None:
            self._poison(error)
            raise error

    def _schedule_writeback(
        self,
        *,
        _admission_token,
        _admission_validator,
    ):
        _require_resource_admission(
            _admission_token,
            _admission_validator,
        )
        if self._dirty is None or self._writeback_ticket is not None:
            return
        ref, array = self._dirty
        self._wait_for_staging(
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
        )
        with self.pool.checkout(
            int(array.nbytes),
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
        ) as slot:
            self._writeback_ticket = self.scheduler.writeback_d2h(
                array,
                ref,
                slot,
                _admission_token=_admission_token,
                _admission_validator=_admission_validator,
            )

    @staticmethod
    def _close_store_reservation(
        reservation,
        *,
        _admission_token,
        _admission_validator,
    ):
        _require_resource_admission(
            _admission_token,
            _admission_validator,
        )
        return reservation.close()

    def _emit_profile(
        self,
        planned_cache_bytes,
        pool,
        *,
        _admission_token,
        _admission_validator,
    ):
        _require_resource_admission(
            _admission_token,
            _admission_validator,
        )
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

    def _lease_close_step(self, transition, operation, callback):
        runtime = self._provider.runtime
        gate = runtime._terminal_gate
        gate.wait_for_lease_admissions(transition, _TERMINAL_TIMEOUT_S)
        token = gate.admit_lease_close(transition, operation)
        try:
            validator = runtime._exact_admission_validator(
                token,
                scope="lease_close",
                epoch=transition.epoch,
                transition_sequence=transition.sequence,
            )
            return callback(token, validator)
        finally:
            runtime._release_admission(token)

    def _terminal_close_primary(self):
        runtime = self._provider.runtime
        transition = runtime._terminal_gate._fatal_transition
        if transition is not None:
            return transition.primary
        for error in (
            runtime._terminal_error,
            self._provider._terminal_error,
            self._poisoned_error,
        ):
            if isinstance(error, BaseException):
                return error
        return RuntimeError("terminal transition preempted working-set close")

    def _raise_terminal_close_preemption(self):
        provider = self._provider
        runtime = None if provider is None else provider.runtime
        if runtime is None:
            return
        if runtime._terminal_gate.phase not in (
            _TerminalPhase.FATAL_PENDING,
            _TerminalPhase.FATAL_PUBLISHED,
            _TerminalPhase.RUNTIME_CLOSED,
        ):
            return
        primary = self._terminal_close_primary()
        self._retain_terminal_lease_safely(provider, primary)
        raise primary

    def _retain_terminal_lease_safely(self, provider, primary):
        runtime = provider.runtime
        try:
            provider._retain_terminal_lease(self, primary)
        except BaseException as retention_error:
            with runtime._terminal_state_lock:
                runtime._remember_terminal_secondary_locked(
                    retention_error,
                    primary,
                )

    def _fail_stop_elected_close(self, error, *, provider, runtime):
        gate = runtime._terminal_gate
        with gate._condition:
            fatal_transition = gate._fatal_transition
            lease_phase = gate._leases[self._epoch].phase
        if fatal_transition is None and lease_phase == "closed":
            raise error
        primary = (
            error if fatal_transition is None else fatal_transition.primary
        )
        try:
            primary = runtime._enter_communicator_fatal(primary)
        except BaseException as publication_error:
            fatal_transition = gate._fatal_transition
            if fatal_transition is None or fatal_transition.primary is not primary:
                raise primary
            if publication_error is not primary:
                with runtime._terminal_state_lock:
                    runtime._remember_terminal_secondary_locked(
                        publication_error,
                        primary,
                    )
            self._retain_terminal_lease_safely(provider, primary)
            runtime._finish_failed_fatal_runtime_close(fatal_transition)
        self._retain_terminal_lease_safely(provider, primary)
        raise primary

    def _run_elected_close(self, transition, *, provider, runtime):
        """Run every fallible elected-owner action under the caller's guard."""
        gate = runtime._terminal_gate
        scheduler = self.scheduler
        scheduler._start_counted_completions()
        error = self._poisoned_error
        old_ref = None if self._dirty is None else self._dirty[0]
        planned_cache_bytes = self._cache_reservation.required_bytes
        pool = self.pool
        status_workspace = self._status_workspace
        cache_reservation = self._cache_reservation
        store_reservation = self._store_reservation

        def remember(caught):
            nonlocal error
            if error is None:
                error = caught

        def run_step(operation, callback):
            try:
                return self._lease_close_step(
                    transition,
                    operation,
                    callback,
                )
            except Exception as caught:
                if gate.phase in (
                    _TerminalPhase.FATAL_PENDING,
                    _TerminalPhase.FATAL_PUBLISHED,
                    _TerminalPhase.RUNTIME_CLOSED,
                ):
                    primary = self._terminal_close_primary()
                    self._retain_terminal_lease_safely(provider, primary)
                    raise primary
                remember(caught)
                return None

        def close_children(token, validator):
            child_error = None
            for child in tuple(self._children):
                try:
                    child.close(_admission_token=token)
                except Exception as caught:
                    if child_error is None:
                        child_error = caught
            if child_error is not None:
                raise child_error

        run_step("child_close", close_children)
        if error is None:
            run_step(
                "schedule_writeback",
                lambda token, validator: self._schedule_writeback(
                    _admission_token=token,
                    _admission_validator=validator,
                ),
            )
        run_step(
            "observe_peaks",
            lambda token, validator: self._observe_peaks(
                _admission_token=token,
                _admission_validator=validator,
            ),
        )

        def complete_scheduler(token, validator):
            scheduler.complete_all(
                _admission_token=token,
                _admission_validator=validator,
            )
            self.metrics.prefetch_overlap_s += sum(
                ticket.elapsed_s
                for ticket in self._prefetch_tickets
                if ticket.completed
            )
            if (
                self._writeback_ticket is not None
                and self._writeback_ticket.completed
                and not self._writeback_accounted
            ):
                if self._writeback_ticket.error is not None:
                    raise self._writeback_ticket.error
                updated = self._writeback_ticket.take_result()
                if updated is None:
                    raise RuntimeError(
                        "dirty writeback completed without a CAS result"
                    )
                self.metrics.dirty_writeback_count += 1
                self.metrics.dirty_writeback_bytes += old_ref.nbytes
                self.metrics.dirty_writeback_s += self._writeback_ticket.elapsed_s
                self._writeback_accounted = True

        run_step("scheduler_complete", complete_scheduler)
        run_step(
            "cache_wait",
            lambda token, validator: provider.cache.wait_for_pending(
                _admission_token=token,
                _admission_validator=validator,
            ),
        )
        run_step(
            "pool_reap",
            lambda token, validator: pool.reap_completed(
                _admission_token=token,
                _admission_validator=validator,
            ),
        )
        if self._writeback_accounted and not self._writeback_invalidated:
            def invalidate(token, validator):
                provider.cache.invalidate_ref(
                    old_ref,
                    _admission_token=token,
                    _admission_validator=validator,
                )
                self._writeback_invalidated = True

            run_step("cache_invalidate", invalidate)
        run_step(
            "emit_profile",
            lambda token, validator: self._emit_profile(
                planned_cache_bytes,
                pool,
                _admission_token=token,
                _admission_validator=validator,
            ),
        )
        run_step(
            "status_close",
            lambda token, validator: status_workspace.close(
                _admission_token=token,
                _admission_validator=validator,
            ),
        )
        run_step(
            "scheduler_close",
            lambda token, validator: scheduler.close(
                _admission_token=token,
                _admission_validator=validator,
            ),
        )
        run_step(
            "pool_close",
            lambda token, validator: pool.close(
                _admission_token=token,
                _admission_validator=validator,
            ),
        )
        run_step(
            "cache_reservation_close",
            lambda token, validator: cache_reservation.close(
                _admission_token=token,
                _admission_validator=validator,
            ),
        )
        run_step(
            "cache_lifetime_reconcile",
            lambda token, validator: provider._reconcile_lease_cache(
                self._resource_record,
                _admission_token=token,
                _admission_validator=validator,
            ),
        )
        run_step(
            "store_reservation_close",
            lambda token, validator: self._close_store_reservation(
                store_reservation,
                _admission_token=token,
                _admission_validator=validator,
            ),
        )

        gate.wait_for_lease_admissions(transition, _TERMINAL_TIMEOUT_S)
        result = gate.commit_lease_close(
            transition,
            lambda: self._commit_close_references(provider, error),
        )
        if isinstance(result, BaseException):
            raise result
        return result

    def _commit_close_references(self, provider, error):
        provider._lease_closed(self)
        provider.last_compute_event = None
        self._resource_record.clear()
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
        return error

    def close(self, wait=True):
        if type(wait) is not bool:
            raise TypeError("wait must be a boolean")
        if self._closed:
            return
        self._handoff_published_terminal()
        if not wait:
            try:
                with self._lease_admission("close_progress") as token:
                    validator = self._exact_admission_validator(token)
                    self._schedule_writeback(
                        _admission_token=token,
                        _admission_validator=validator,
                    )
                    self.reap_completed(_admission_token=token)
                return
            except BaseException as caught:
                self._poison(caught)
                return

        provider = self._provider
        runtime = provider.runtime
        gate = runtime._terminal_gate
        try:
            transition, elected = gate.begin_lease_close(self._epoch)
        except BaseException:
            primary = self._terminal_close_primary()
            self._retain_terminal_lease_safely(provider, primary)
            raise primary
        self._closing = True
        if not elected:
            try:
                result = gate.wait_for_lease_closed(
                    transition,
                    _TERMINAL_TIMEOUT_S,
                )
            except BaseException:
                self._raise_terminal_close_preemption()
                raise
            if isinstance(result, BaseException):
                raise result
            return result

        try:
            return self._run_elected_close(
                transition,
                provider=provider,
                runtime=runtime,
            )
        except BaseException as caught:
            self._fail_stop_elected_close(
                caught,
                provider=provider,
                runtime=runtime,
            )


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
        _admission_token=None,
        _admission_validator=None,
    ):
        if _admission_token is not None:
            if _admission_validator is None:
                _admission_validator = runtime._exact_admission_validator(
                    _admission_token,
                    scope="runtime_setup",
                    epoch=None,
                )
            _require_resource_admission(
                _admission_token,
                _admission_validator,
            )
        elif _admission_validator is not None:
            raise TypeError("provider construction admission token is required")
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
        self._provisional_resources = None
        self._persistent_resources = _LeaseResourceRecord(None)
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
        if _admission_token is None:
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

    def matches_config(
        self,
        device_budget,
        host_budget,
        prefetch_depth,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        if _admission_token is not None:
            if _admission_validator is None:
                _admission_validator = self.runtime._exact_admission_validator(
                    _admission_token,
                    scope="runtime_setup",
                    epoch=None,
                )
            _require_resource_admission(
                _admission_token,
                _admission_validator,
            )
        elif _admission_validator is not None:
            raise TypeError("provider config admission token is required")
        return (
            device_budget == self.device_budget_resolution
            and host_budget == self.host_budget_resolution
            and prefetch_depth == self.prefetch_depth
        )

    def _spawn_async_admission(self, epoch, capability, operation):
        runtime = self.runtime
        gate = runtime._terminal_gate
        parent = gate._current_thread_admission()
        if parent is None:
            raise RuntimeError("async scheduling requires a resource admission")
        if parent.scope == "lease":
            runtime._require_admission(
                parent,
                scope="lease",
                epoch=epoch,
            )
        elif parent.scope == "lease_close":
            runtime._require_admission(
                parent,
                scope="lease_close",
                epoch=epoch,
                transition_sequence=parent.transition_sequence,
            )
        else:
            raise RuntimeError(
                "working-set async scheduling requires lease ownership"
            )
        return _CountedAsyncAdmission(gate, parent, capability)

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

    def _provision_status_workspace(
        self,
        *,
        _admission_token,
        _admission_validator,
        _resource_recorder,
    ):
        _require_resource_admission(
            _admission_token,
            _admission_validator,
        )
        if not callable(_resource_recorder):
            raise TypeError("status resource recorder must be callable")
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
            _resource_recorder(
                records=allocation_records((device_status,)),
            )
        except BaseException as error:
            local_error = error
            local_code |= 1
        try:
            host_status = self._allocate_status_host()
            _resource_recorder(
                records=allocation_records((host_status,)),
            )
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

    def _lease_cache_recorder(self, record):
        def capture(**captured):
            record.capture(**captured)
            self._persistent_resources.capture(**captured)

        return capture

    def _reconcile_lease_cache(
        self,
        record,
        *,
        _admission_token,
        _admission_validator,
    ):
        _require_resource_admission(
            _admission_token,
            _admission_validator,
        )
        cache = self._cache
        if cache is None:
            return
        cache._set_resource_recorder(self._persistent_resources.capture)
        record.capture(
            resource=cache,
            kind="cache",
            records=cache.allocation_records,
            _replace_kind=True,
        )

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

    @staticmethod
    def _reserve_store(
        store,
        snapshot,
        *,
        dirty_ref,
        _admission_token,
        _admission_validator,
    ):
        _require_resource_admission(
            _admission_token,
            _admission_validator,
        )
        return store.reserve(snapshot, dirty_ref=dirty_ref)

    def _activate_lease(
        self,
        epoch,
        *,
        _admission_token,
        _admission_validator,
    ):
        _require_resource_admission(
            _admission_token,
            _admission_validator,
        )
        return self.runtime._terminal_gate.activate_lease(
            epoch,
            _admission_token,
        )

    def _rollback_partial(
        self,
        token,
        first_error,
        *,
        record,
        _admission_validator,
        scheduler,
        pool,
        cache_reservation,
        store_reservation,
        status_workspace,
    ):
        _require_resource_admission(
            token,
            _admission_validator,
        )
        error = first_error
        try:
            record.restore_consumed_receipts(self.runtime)
        except BaseException as caught:
            if error is None:
                error = caught
        for resource in (
            scheduler,
            pool,
            cache_reservation,
            store_reservation,
            status_workspace,
        ):
            if resource is None:
                continue
            try:
                if resource is store_reservation:
                    resource.close()
                else:
                    resource.close(
                        _admission_token=token,
                        _admission_validator=_admission_validator,
                    )
            except BaseException as caught:
                if error is None:
                    error = caught
        if self._cache is not None:
            try:
                self._cache._set_resource_recorder(
                    self._persistent_resources.capture
                )
            except BaseException as caught:
                if error is None:
                    error = caught
        return error

    def _raise_construction_transition_preemption(self):
        gate = self.runtime._terminal_gate
        with gate._condition:
            if gate._phase in (
                _TerminalPhase.FATAL_PENDING,
                _TerminalPhase.FATAL_PUBLISHED,
                _TerminalPhase.RUNTIME_CLOSED,
            ):
                transition = gate._fatal_transition
                if transition is not None:
                    raise transition.primary
                gate._raise_for_terminal_phase()
            if gate._phase is _TerminalPhase.RUNTIME_CLOSING:
                raise RuntimeError("runtime is closing")

    def _commit_construction_rollback(self, epoch, record):
        gate = self.runtime._terminal_gate
        transition, elected = gate.begin_lease_close(epoch)
        if not elected:
            return gate.wait_for_lease_closed(transition, _TERMINAL_TIMEOUT_S)
        gate.wait_for_lease_admissions(transition, _TERMINAL_TIMEOUT_S)

        def finalize():
            if self._active_lease is not None and (
                getattr(self._active_lease, "_resource_record", None) is record
            ):
                self._active_lease = None
            if self._provisional_resources is record:
                self._provisional_resources = None
            record.clear()

        return gate.commit_lease_close(transition, finalize)

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

        gate = self.runtime._terminal_gate
        epoch, construction_token = gate.begin_lease("lease_construction")
        construction_validator = self.runtime._exact_admission_validator(
            construction_token,
            scope="construction",
            epoch=epoch,
        )
        record = _LeaseResourceRecord(epoch)
        self._provisional_resources = record
        status_workspace = None
        store_reservation = None
        cache_reservation = None
        pool = None
        scheduler = None
        lease = None
        profile_enabled = False
        timer = None
        peak_sampler = None
        cache_recorder = self._lease_cache_recorder(record)
        try:
            self.runtime.consume_residency_receipt(
                receipt,
                request,
                plan,
                _admission_token=construction_token,
                _admission_validator=construction_validator,
                _resource_recorder=record.capture_consumed_receipt,
            )
            self._raise_construction_transition_preemption()
            status_workspace = self._provision_status_workspace(
                _admission_token=construction_token,
                _admission_validator=construction_validator,
                _resource_recorder=record.capture,
            )
            record.capture(status_workspace)
            self._raise_construction_transition_preemption()
            dirty_key = request.distributed_plan.execution_plan.output.key
            dirty_ref = dict(request.host_refs)[dirty_key]
            store_reservation = self._reserve_store(
                store,
                request.store_snapshots[self.runtime.rank],
                dirty_ref=dirty_ref,
                _admission_token=construction_token,
                _admission_validator=construction_validator,
            )
            record.capture(store_reservation)
            self._raise_construction_transition_preemption()
            if self._cache is None:
                self._cache = self._cache_factory(
                    self.device_budget_resolution.resolved_bytes,
                    allocator=self._cache_allocator,
                    _resource_recorder=cache_recorder,
                    _admission_token=construction_token,
                    _admission_validator=construction_validator,
                )
            set_recorder = getattr(self._cache, "_set_resource_recorder", None)
            if callable(set_recorder):
                set_recorder(cache_recorder)
            record.capture(self._cache, kind="cache")
            cache_reservation = self._cache.reserve(
                allowlist,
                required_bytes,
                _admission_token=construction_token,
                _admission_validator=construction_validator,
            )
            record.capture(cache_reservation)
            self._raise_construction_transition_preemption()
            pool = self._pool_factory(
                request.transfer_profile.rank_staging_bytes[self.runtime.rank],
                pinned_allocator=self._pinned_allocator,
                _resource_recorder=record.capture,
                _admission_token=construction_token,
                _admission_validator=construction_validator,
            )
            record.capture(pool, kind="pinned")
            self._raise_construction_transition_preemption()
            from renormalizer.utils.log import PROFILING, get_logger

            profile_enabled = bool(get_logger().isEnabledFor(PROFILING))
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
                _async_admission_factory=lambda capability, operation: (
                    self._spawn_async_admission(
                        epoch,
                        capability,
                        operation,
                    )
                ),
                _resource_recorder=record.capture,
                _resource_releaser=record.release,
                _admission_token=construction_token,
                _admission_validator=construction_validator,
            )
            record.capture(scheduler)
            self._raise_construction_transition_preemption()
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
                epoch=epoch,
                resource_record=record,
                profile_enabled=profile_enabled,
                timer=timer,
                peak_sampler=peak_sampler,
            )
            record.capture(lease)
            self._active_lease = lease
            self._last_authorized_cache_bytes = required_bytes
            lease.metrics.full_replica = bool(
                plan.metadata()["full_replica_prediction"]
            )
            lease.metrics.pageable_fallback_count = pool.pageable_fallback_count
            lease.metrics.pageable_fallback_bytes = pool.pageable_fallback_bytes
            self.metrics = lease.metrics
            self._activate_lease(
                epoch,
                _admission_token=construction_token,
                _admission_validator=construction_validator,
            )
            self._raise_construction_transition_preemption()
        except BaseException as error:
            first_error = error
            terminal_fatal = gate.phase in (
                _TerminalPhase.FATAL_PENDING,
                _TerminalPhase.FATAL_PUBLISHED,
                _TerminalPhase.RUNTIME_CLOSED,
            )
            if terminal_fatal and gate._fatal_transition is not None:
                first_error = gate._fatal_transition.primary
            if not terminal_fatal:
                first_error = self._rollback_partial(
                    construction_token,
                    first_error,
                    record=record,
                    _admission_validator=construction_validator,
                    scheduler=scheduler,
                    pool=pool,
                    cache_reservation=cache_reservation,
                    store_reservation=store_reservation,
                    status_workspace=status_workspace,
                )
            self.runtime._release_admission(construction_token)
            if gate.phase in (
                _TerminalPhase.FATAL_PENDING,
                _TerminalPhase.FATAL_PUBLISHED,
                _TerminalPhase.RUNTIME_CLOSED,
            ) and gate._fatal_transition is not None:
                first_error = gate._fatal_transition.primary
            if not terminal_fatal and gate.phase not in (
                _TerminalPhase.FATAL_PENDING,
                _TerminalPhase.FATAL_PUBLISHED,
                _TerminalPhase.RUNTIME_CLOSED,
            ):
                self._commit_construction_rollback(epoch, record)
            raise first_error
        self.runtime._release_admission(construction_token)
        self._provisional_resources = None
        return lease

    def _lease_closed(self, lease):
        if self._active_lease is lease:
            self._active_lease = None

    def _retain_terminal_snapshot(self, record):
        if record is None:
            return
        retained = list(self._terminal_resources)
        for resource in record.resources:
            if resource is not None and all(
                existing is not resource for existing in retained
            ):
                retained.append(resource)
        self._terminal_resources = tuple(retained)
        self._terminal_quarantine.retain_snapshot(
            resources=record.resources,
            allocations=record.allocations,
            cache_allocations=record.cache_allocations,
            pinned_allocations=record.pinned_allocations,
            events=record.events,
            streams=record.streams,
        )

    def _retain_transition_resources(self, lease):
        self._retain_terminal_snapshot(self._persistent_resources)
        record = (
            self._provisional_resources
            if lease is None
            else getattr(lease, "_resource_record", None)
        )
        self._retain_terminal_snapshot(record)

    def _retain_terminal_record(self, record, error):
        if record is None:
            return
        if isinstance(error, BaseException):
            if self._terminal_error is None:
                self._terminal_error = error
            self._terminal_quarantine.retain_error(error)
        self._retain_terminal_snapshot(record)

    def _retain_terminal_lease(self, lease, error):
        record = getattr(lease, "_resource_record", None)
        self._retain_terminal_snapshot(self._persistent_resources)
        self._retain_terminal_record(record, error)
        runtime = self.runtime
        if runtime is not None and runtime._terminal_error is None:
            runtime._terminal_error = error

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
                runtime._enter_communicator_fatal(
                    primary,
                    owner,
                    discovering_token=(
                        runtime._terminal_gate._current_thread_admission()
                    ),
                )

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

    def runtime_resource_state(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        _require_resource_admission(
            _admission_token,
            _admission_validator,
        )
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

    def close(
        self,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
        _require_resource_admission(
            _admission_token,
            _admission_validator,
        )
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
