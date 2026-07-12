"""Resident operand leases for distributed block execution."""

from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import ContextManager, Protocol

import numpy as np

from renormalizer.backend._distributed.context import DistributedContext
from renormalizer.backend._distributed.planner import DistributedPlan
from renormalizer.backend._execution.model import ExecutionBindings, ExecutionPlan


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


__all__ = [
    "active_working_set_policy_error",
    "DeviceResidentProvider",
    "OperandLease",
    "OperandProvider",
    "OperandRequest",
]
