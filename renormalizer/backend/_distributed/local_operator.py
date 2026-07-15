"""Ordered source-broadcast execution for output-sharded local H-v."""

from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from renormalizer.backend._distributed.context import DistributedContext
from renormalizer.backend._distributed.planner import DistributedPlan
from renormalizer.backend._distributed.providers import (
    OperandRequest,
    active_working_set_policy_error,
)
from renormalizer.backend._execution.model import ExecutionBindings
from renormalizer.backend._execution.executor import _exact_reshape_view


_PREFLIGHT_ERRORS = {
    1: "type",
    2: "device",
    3: "shape",
    4: "dtype",
    5: "layout",
}

_COUNTER_KEYS = (
    "broadcast_calls",
    "allreduce_calls",
    "allgather_calls",
    "execution_calls",
)


def _counter_validation_error(counters):
    if type(counters) is not dict:
        return TypeError("counters must be a dict")
    for key in _COUNTER_KEYS:
        if key not in counters:
            continue
        value = counters[key]
        if type(value) is not int:
            return TypeError(
                "counter {!r} must be a non-negative Python int".format(key)
            )
        if value < 0:
            return ValueError("counter {!r} must be non-negative".format(key))
    return None


def _array_scalar(array):
    module = type(array).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy

        return int(cupy.asnumpy(array).reshape(-1)[0])
    return int(np.asarray(array).reshape(-1)[0])


def _array_to_numpy(array):
    module = type(array).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy

        return cupy.asnumpy(array)
    return np.asarray(array)


def _control_array(collective, values, dtype):
    cupy = getattr(collective, "_cupy", None)
    if cupy is not None:
        with cupy.cuda.Device(int(collective._device_index)):
            return cupy.asarray(values, dtype=dtype)
    return np.asarray(values, dtype=dtype)


def _fallback_status(collective, failed):
    failure = _control_array(collective, [1 if failed else 0], np.int32)
    synchronized = collective.allreduce(failure, op="max")
    return _array_scalar(synchronized)


def _validate_fallback_buffer(buffer, collective):
    cupy_collective = getattr(collective, "_cupy", None) is not None
    if cupy_collective:
        collective._validate_array(buffer)
        device_kind = "cupy-local-device"
    else:
        if not isinstance(buffer, np.ndarray):
            raise TypeError("receive_buffer must be a NumPy or CuPy array")
        if buffer.size == 0:
            raise ValueError("receive_buffer must not be empty")
        device_kind = "numpy-host"
    if not bool(buffer.flags.c_contiguous):
        raise ValueError("receive_buffer must be C contiguous")
    dtype = np.dtype(buffer.dtype)
    if dtype.hasobject or dtype.fields is not None or dtype.kind not in "biufc":
        raise TypeError("receive_buffer dtype is unsupported")
    return {
        "shape": tuple(int(dimension) for dimension in buffer.shape),
        "dtype": dtype.str,
        "device": device_kind,
        "count": int(buffer.size),
    }


def _fallback_fingerprint(metadata):
    encoded = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    digest = hashlib.sha256(encoded).hexdigest()
    return [int(digest[index : index + 16], 16) for index in range(0, 64, 16)]


def _fallback_fingerprint_agrees(collective, metadata):
    if collective.size == 1:
        return True
    fingerprint = _control_array(collective, _fallback_fingerprint(metadata), np.uint64)
    minimum = collective.allreduce(fingerprint, op="min")
    maximum = collective.allreduce(fingerprint, op="max")
    local = _array_to_numpy(fingerprint)
    return np.array_equal(_array_to_numpy(minimum), local) and np.array_equal(
        _array_to_numpy(maximum), local
    )


def _fallback_buffer_presence_agrees(collective, receive_buffer):
    if collective.size == 1:
        return True
    presence = _control_array(
        collective, [1 if receive_buffer is not None else 0], np.int32
    )
    minimum = collective.allreduce(presence, op="min")
    maximum = collective.allreduce(presence, op="max")
    return _array_scalar(minimum) == _array_scalar(maximum)


def run_root_fallback(operation, collective, *, receive_buffer=None):
    """Run one root operation and synchronize failure before broadcasting."""
    if not callable(operation):
        raise TypeError("operation must be callable")
    if not hasattr(collective, "rank") or not hasattr(collective, "size"):
        raise TypeError("collective must expose rank and size")
    if not _fallback_buffer_presence_agrees(collective, receive_buffer):
        raise ValueError("root fallback receive-buffer presence disagreement")
    cupy_collective = getattr(collective, "_cupy", None) is not None
    buffer_metadata = None
    if receive_buffer is not None or (cupy_collective and collective.size > 1):
        buffer_error = None
        try:
            if receive_buffer is None:
                raise ValueError(
                    "real multi-rank CuPy fallback requires receive_buffer"
                )
            buffer_metadata = _validate_fallback_buffer(receive_buffer, collective)
        except Exception as error:
            buffer_error = error
        if _fallback_status(collective, buffer_error is not None):
            raise ValueError("root fallback receive-buffer preflight failed")
        if not _fallback_fingerprint_agrees(collective, buffer_metadata):
            raise ValueError("root fallback receive-buffer fingerprint disagreement")

    result = None
    root_error = None
    if collective.rank == 0:
        try:
            result = operation()
            if receive_buffer is not None:
                result_metadata = _validate_fallback_buffer(result, collective)
                if result_metadata != buffer_metadata:
                    raise ValueError(
                        "root fallback result metadata does not match receive_buffer"
                    )
                if result is not receive_buffer:
                    receive_buffer[...] = result
                result = receive_buffer
        except BaseException as error:
            root_error = error

    if _fallback_status(collective, root_error is not None):
        if root_error is not None:
            raise RuntimeError("root fallback operation failed") from root_error
        raise RuntimeError("root fallback operation failed")

    if receive_buffer is not None:
        result = receive_buffer
    return collective.broadcast(result, root=0)


def _bootstrap_schedule_size(context, plan, collective):
    """Choose the fixed setup-status schedule from valid coordination metadata."""
    if isinstance(context, DistributedContext):
        return context.world_size
    if isinstance(plan, DistributedPlan):
        return plan.world_size
    collective_size = getattr(collective, "size", None)
    if type(collective_size) is int and collective_size > 0:
        return collective_size
    raise TypeError("operator bootstrap schedule size is unavailable")


@dataclass
class DistributedLocalOperator:
    plan: DistributedPlan
    provider: object
    collective: object
    counters: dict[str, int]
    backend: object
    context: DistributedContext
    source_bindings: ExecutionBindings
    device_memory_budget_bytes: int | None = None
    host_memory_budget_bytes: int | None = None
    residency_request: object = None
    residency_plan: object = None
    residency_receipt: object = None
    mesh: object = None

    def __post_init__(self):
        if not callable(getattr(self.collective, "allreduce", None)):
            raise TypeError("collective must provide a usable allreduce")
        self._bootstrap_world_size = _bootstrap_schedule_size(
            self.context, self.plan, self.collective
        )
        self._counter_error = _counter_validation_error(self.counters)
        self._counter_target = self.counters if self._counter_error is None else None
        self._counters = {key: 0 for key in _COUNTER_KEYS}
        if type(self.counters) is dict:
            for key in _COUNTER_KEYS:
                value = self.counters.get(key)
                if type(value) is int and value >= 0:
                    self._counters[key] = value
        self._setup_error = None
        try:
            self._validate_setup()
        except BaseException as error:
            self._setup_error = error
        self._plan_preflight_complete = False
        self._setup_preflight_complete = False
        self._capacity_preflight_complete = False
        self._receive_storage = None
        self._output_accumulator = None
        self._execution_status = None
        self._host_execution_status = None
        self._active_contribution = None
        self._active_allocation_scope = None
        self._resource_allocation_preflight_complete = False

    def _validate_setup(self):
        if not isinstance(self.plan, DistributedPlan):
            raise TypeError("plan must be a DistributedPlan")
        if not isinstance(self.context, DistributedContext):
            raise TypeError("context must be a DistributedContext")
        if self._counter_error is not None:
            raise self._counter_error
        for name in ("rank", "size"):
            if type(getattr(self.collective, name, None)) is not int:
                raise TypeError("collective must expose integer rank and size")
        if self.collective.size <= 0:
            raise ValueError("collective size must be positive")
        if not callable(getattr(self.collective, "broadcast", None)):
            raise TypeError("collective must provide broadcast")
        if not isinstance(self.source_bindings, ExecutionBindings):
            raise TypeError("source_bindings must be ExecutionBindings")
        if getattr(self.backend, "name", None) not in {"numpy", "cupy"}:
            raise NotImplementedError(
                "distributed local execution supports only NumPy and CuPy backends"
            )
        if not getattr(self.backend, "supports_execution_ir", False):
            raise NotImplementedError("backend does not support execution IR")
        if self.context.world_size != self.plan.world_size:
            raise ValueError("distributed plan world size does not match context")
        if (
            self.collective.rank != self.context.rank
            or self.collective.size != self.context.world_size
        ):
            raise ValueError("collective rank or size does not match context")
        if not callable(getattr(self.provider, "acquire", None)):
            raise TypeError("provider must implement acquire")
        if not callable(getattr(self.provider, "validate_setup", None)):
            raise TypeError("provider must implement validate_setup")
        if not callable(getattr(self.provider, "validate_request", None)):
            raise TypeError("provider must implement validate_request")
        if not callable(getattr(self.collective, "allreduce_inplace", None)):
            raise TypeError("collective must implement in-place allreduce")
        for name in ("device_memory_budget_bytes", "host_memory_budget_bytes"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(
                    "{} must be a non-negative integer or None".format(name)
                )
        if self.plan.variable_key in self.source_bindings.arrays:
            raise ValueError(
                "resident source bindings must not contain the variable input"
            )
        expected_keys = {ref.key for ref in self.plan.execution_plan.inputs} - {
            self.plan.variable_key
        }
        actual_keys = set(self.source_bindings.arrays)
        if actual_keys != expected_keys:
            raise ValueError("source binding coverage is incomplete")
        provider_policy = getattr(self.provider, "residency_policy", None)
        provider_role = getattr(self.provider, "provider_role", None)
        if provider_role == "factory":
            raise ValueError("factory provider cannot execute operand blocks")
        if provider_policy == "active_working_set" and (
            self.residency_plan is None or self.residency_request is None
        ):
            raise ValueError(
                "active_working_set requires a resolved residency plan and request"
            )
        if provider_policy != "active_working_set" and (
            self.residency_plan is not None
            or self.residency_request is not None
            or self.residency_receipt is not None
        ):
            raise ValueError(
                "residency metadata requires active_working_set provider policy"
            )
        policy_error = active_working_set_policy_error(
            provider_policy, self.backend, self.plan
        )
        if policy_error is not None:
            raise policy_error
        if self.residency_plan is not None:
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
            if (
                self.residency_plan.world_size != self.plan.world_size
                or self.residency_plan.placement_hash != self.plan.placement_hash
            ):
                raise ValueError("residency plan does not match distributed plan")
            if self.residency_request.distributed_plan != self.plan:
                raise ValueError("residency request does not match distributed plan")
            if self.residency_request.backend_name != self.backend.name:
                raise ValueError("residency request backend does not match operator")
            self.residency_plan.validate_capacity()
            self.residency_plan.runtime_identity.validate_runtime(
                self.context, self.mesh, self.backend
            )
            self.residency_plan.validate_request(self.residency_request)
            if provider_role == "working_set":
                from renormalizer.backend._distributed.residency import (
                    ResidencyPreflightReceipt,
                )

                if not isinstance(self.residency_receipt, ResidencyPreflightReceipt):
                    raise TypeError("working-set provider requires a preflight receipt")
                if (
                    getattr(self.provider, "request", None)
                    is not self.residency_request
                    or getattr(self.provider, "plan", None) is not self.residency_plan
                    or getattr(self.provider, "receipt", None)
                    is not self.residency_receipt
                ):
                    raise ValueError(
                        "working-set provider metadata does not match operator"
                    )
        self.provider.validate_setup(self.plan, self.source_bindings, self.context)
        self._validate_provider_requests()

    @property
    def solver_input_sharding(self):
        return self.plan.input_sharding

    @property
    def solver_output_sharding(self):
        return self.plan.output_sharding

    @property
    def solver_dtype(self):
        variable_ref = next(
            ref
            for ref in self.plan.execution_plan.inputs
            if ref.key == self.plan.variable_key
        )
        return np.dtype(variable_ref.spec.dtype)

    def solver_preflight(self):
        if callable(getattr(self.provider, "_operator_call", None)):
            if (
                self.residency_receipt is None
                or self.provider.request is not self.residency_request
                or self.provider.plan is not self.residency_plan
                or self.provider.receipt is not self.residency_receipt
            ):
                raise ValueError("working-set receipt metadata does not match operator")
            return
        self._preflight_setup()
        self._preflight_plan_agreement()
        self._preflight_capacity()

    def _validate_provider_requests(self):
        for source_rank in range(self.plan.world_size):
            block = self.plan.block_plan(self.context.rank, source_rank)
            request = OperandRequest(
                execution_plan=block.execution_plan,
                source_bindings=self.source_bindings,
                distributed_plan=self.plan,
                context=self.context,
                source_rank=source_rank,
                broadcast_variable=None,
                operand_slices=block.operand_slices,
                residency_request=self.residency_request,
                residency_plan=self.residency_plan,
            )
            self.provider.validate_request(request, self.residency_plan)

    def _local_vector_error(self, local_vector):
        try:
            self.backend._validate_execution_array(local_vector)
        except TypeError:
            return 1
        except ValueError:
            return 2
        local_ref = next(
            ref
            for ref in self.plan.block_plan(
                self.context.rank, self.context.rank
            ).execution_plan.inputs
            if ref.key == self.plan.variable_key
        )
        if tuple(local_vector.shape) != local_ref.spec.shape:
            return 3
        if np.dtype(local_vector.dtype).name != local_ref.spec.dtype:
            return 4
        if not bool(local_vector.flags.c_contiguous):
            return 5
        return 0

    def _mirror_counters(self):
        if self._counter_target is None:
            return
        for key in _COUNTER_KEYS:
            self._counter_target[key] = self._counters[key]

    def _increment_counter(self, key, amount=1, *, mirror=True):
        self._counters[key] = self._counters[key] + amount
        if mirror:
            self._mirror_counters()

    def _allreduce_status(
        self, local_code, *, schedule_size=None, mirror_counters=True
    ):
        if schedule_size is None:
            schedule_size = self.context.world_size
        if schedule_size == 1:
            return local_code
        status = _control_array(self.collective, [local_code], np.int32)
        synchronized = self.collective.allreduce(status, op="max")
        self._increment_counter("allreduce_calls", mirror=mirror_counters)
        return _array_scalar(synchronized)

    def _preflight_setup(self):
        if self._setup_preflight_complete:
            return
        error_code = self._allreduce_status(
            1 if self._setup_error is not None else 0,
            schedule_size=self._bootstrap_world_size,
            mirror_counters=False,
        )
        self._mirror_counters()
        if error_code:
            if self._setup_error is not None:
                raise ValueError(
                    "distributed setup preflight failed"
                ) from self._setup_error
            raise ValueError("distributed setup preflight failed")
        self._setup_preflight_complete = True

    def _preflight_local_vector(self, local_vector):
        error_code = self._local_vector_error(local_vector)
        error_code = self._allreduce_status(error_code)
        if error_code:
            raise ValueError(
                "local vector preflight failed: {}".format(
                    _PREFLIGHT_ERRORS.get(error_code, "unknown")
                )
            )

    def _preflight_plan_agreement(self):
        if self._plan_preflight_complete or self.context.world_size == 1:
            self._plan_preflight_complete = True
            return
        policy = getattr(self.provider, "residency_policy", None)
        local_policy = _control_array(
            self.collective,
            [
                int(policy == "active_working_set"),
                int(self.residency_plan is not None),
            ],
            np.int32,
        )
        policy_minimum = self.collective.allreduce(local_policy, op="min")
        policy_maximum = self.collective.allreduce(local_policy, op="max")
        self._increment_counter("allreduce_calls", amount=2)
        policy_minimum_values = _array_to_numpy(policy_minimum)
        policy_maximum_values = _array_to_numpy(policy_maximum)
        if policy_minimum_values[0] != policy_maximum_values[0]:
            raise ValueError("distributed residency policy disagreement")
        if policy_minimum_values[1] != policy_maximum_values[1]:
            raise ValueError("distributed residency plan presence disagreement")

        chunks = [
            int(self.plan.placement_hash[index : index + 16], 16)
            for index in range(0, 64, 16)
        ]
        if self.residency_plan is not None:
            chunks.extend(
                int(self.residency_plan.plan_hash[index : index + 16], 16)
                for index in range(0, 64, 16)
            )
        local_hash = _control_array(self.collective, chunks, np.uint64)
        minimum = self.collective.allreduce(local_hash, op="min")
        maximum = self.collective.allreduce(local_hash, op="max")
        self._increment_counter("allreduce_calls", amount=2)
        local_values = _array_to_numpy(local_hash)
        minimum_values = _array_to_numpy(minimum)
        maximum_values = _array_to_numpy(maximum)
        if not (
            np.array_equal(minimum_values[:4], local_values[:4])
            and np.array_equal(maximum_values[:4], local_values[:4])
        ):
            raise ValueError("distributed placement hash disagreement")
        if self.residency_plan is not None and not (
            np.array_equal(minimum_values[4:], local_values[4:])
            and np.array_equal(maximum_values[4:], local_values[4:])
        ):
            raise ValueError("distributed residency hash disagreement")
        if self.residency_plan is not None:
            del (
                local_policy,
                policy_minimum,
                policy_maximum,
                policy_minimum_values,
                policy_maximum_values,
                local_hash,
                minimum,
                maximum,
                local_values,
                minimum_values,
                maximum_values,
            )
            requirements = (
                *self.residency_plan.device_peak_bytes,
                *self.residency_plan.host_peak_bytes,
                self.residency_plan.host_required_bytes,
            )
            budgets = (
                self.residency_plan.device_budget.resolved_bytes,
                self.residency_plan.host_budget.resolved_bytes,
            )
            control = _control_array(
                self.collective, (*requirements, *budgets), np.int64
            )
            minimum = self.collective.allreduce(control, op="min")
            maximum = self.collective.allreduce(control, op="max")
            self._increment_counter("allreduce_calls", amount=2)
            local_values = _array_to_numpy(control)
            minimum_values = _array_to_numpy(minimum)
            maximum_values = _array_to_numpy(maximum)
            requirement_count = len(requirements)
            if not (
                np.array_equal(
                    minimum_values[:requirement_count],
                    local_values[:requirement_count],
                )
                and np.array_equal(
                    maximum_values[:requirement_count],
                    local_values[:requirement_count],
                )
            ):
                raise ValueError("distributed residency requirement disagreement")
            if not (
                np.array_equal(
                    minimum_values[requirement_count:],
                    local_values[requirement_count:],
                )
                and np.array_equal(
                    maximum_values[requirement_count:],
                    local_values[requirement_count:],
                )
            ):
                raise ValueError("distributed residency budget disagreement")
        self._plan_preflight_complete = True

    def _preflight_capacity(self):
        if self._capacity_preflight_complete:
            return
        local_code = 0
        if self.residency_plan is not None:
            if (
                self.residency_plan.backend_name == "cupy"
                and self.residency_plan.device_peak_bytes[self.context.rank]
                > self.residency_plan.device_budget.resolved_bytes
            ):
                local_code = 1
            elif (
                self.residency_plan.host_required_bytes
                > self.residency_plan.host_budget.resolved_bytes
            ):
                local_code = 2
        else:
            estimate = self.plan.memory_estimates[self.context.rank]
            if (
                self.device_memory_budget_bytes is not None
                and estimate.device_bytes > self.device_memory_budget_bytes
            ):
                local_code = 1
            elif (
                self.host_memory_budget_bytes is not None
                and estimate.host_bytes > self.host_memory_budget_bytes
            ):
                local_code = 2
        if self._allreduce_status(local_code):
            raise ValueError("distributed capacity preflight failed")
        self._capacity_preflight_complete = True

    @staticmethod
    def _shape_elements(shape):
        elements = 1
        for dimension in shape:
            elements *= dimension
        return elements

    def _receive_view(self, storage, shape):
        elements = self._shape_elements(shape)
        prefix = storage[:elements]
        view = _exact_reshape_view(self.backend, prefix, shape)
        if not bool(view.flags.c_contiguous):
            raise ValueError("reusable receive view must be C contiguous")
        return view

    def _discard_execution_resources(self):
        self._receive_storage = None
        self._output_accumulator = None
        self._execution_status = None
        self._host_execution_status = None
        self._resource_allocation_preflight_complete = False

    def _copy_execution_status_to_host(self):
        return self._copy_status_to_host(
            self._execution_status, self._host_execution_status
        )

    def _copy_status_to_host(self, status, host_status):
        if self.backend.name == "cupy":
            cupy = self.backend._cupy
            with cupy.cuda.Device(self.backend._device_index):
                copied = cupy.asnumpy(status, out=host_status, blocking=True)
            if copied is not host_status:
                raise RuntimeError("CuPy status copy replaced host staging storage")
        else:
            np.copyto(host_status, status, casting="no")
        return int(host_status[0])

    def _allreduce_active_status(self, call, local_error):
        runtime = self.provider._provider.runtime
        terminal = runtime._terminal_error or getattr(
            self.collective, "_fatal_error", None
        )
        if terminal is not None:
            primary = call.mark_communicator_fatal(terminal)
            self.provider._enter_communicator_fatal(primary, call.owner)
            raise primary
        status = call.status_workspace.device_status
        host_status = call.status_workspace.host_status
        try:
            status.fill(1 if local_error is not None else 0)
            synchronized = self.collective.allreduce_inplace(status, op="max")
            self._increment_counter("allreduce_calls")
            if synchronized is not status:
                raise RuntimeError(
                    "in-place allreduce replaced execution status storage"
                )
            terminal = runtime._terminal_error or getattr(
                self.collective, "_fatal_error", None
            )
            if terminal is not None:
                primary = call.mark_communicator_fatal(terminal)
                self.provider._enter_communicator_fatal(primary, call.owner)
                raise primary
            result = self._copy_status_to_host(status, host_status)
            terminal = runtime._terminal_error or getattr(
                self.collective, "_fatal_error", None
            )
            if terminal is not None:
                primary = call.mark_communicator_fatal(terminal)
                self.provider._enter_communicator_fatal(primary, call.owner)
                raise primary
            return result
        except BaseException as error:
            fatal_error = getattr(self.collective, "_fatal_error", None)
            if fatal_error is not None and error.__cause__ is fatal_error:
                error = fatal_error
            primary = call.record_primary(error)
            if primary is not error:
                call.record_secondary(error)
            call.mark_communicator_fatal(primary)
            self.provider._enter_communicator_fatal(primary, call.owner)
            if primary is error:
                raise
            raise primary

    def _allreduce_execution_status(self, local_error, *, synchronize=True):
        status = self._execution_status
        status.fill(1 if local_error is not None else 0)
        if synchronize:
            synchronized = self.collective.allreduce_inplace(status, op="max")
            self._increment_counter("allreduce_calls")
            if synchronized is not status:
                raise RuntimeError(
                    "in-place allreduce replaced execution status storage"
                )
        return self._copy_execution_status_to_host()

    def _accumulate_contribution(self, output, contribution):
        if self._active_allocation_scope is None:
            output += contribution
            return
        self._active_allocation_scope.add_into(output, contribution, output)

    def _ensure_resident_call_resources(self, allocation_scope=None):
        resources = (
            self._receive_storage,
            self._output_accumulator,
            self._execution_status,
            self._host_execution_status,
        )
        if all(resource is not None for resource in resources):
            return
        if any(resource is not None for resource in resources):
            self._discard_execution_resources()
        variable_ref = next(
            ref
            for ref in self.plan.execution_plan.inputs
            if ref.key == self.plan.variable_key
        )
        self._execution_status = self.backend.empty((1,), dtype=np.int32, order="C")
        if allocation_scope is not None:
            allocation_scope.capture(self._execution_status)
        self._host_execution_status = np.empty((1,), dtype=np.int32, order="C")
        if allocation_scope is not None:
            allocation_scope.capture(self._host_execution_status)
        self._receive_storage = self.backend.empty(
            (self.plan.input_sharding.max_local_elements,),
            dtype=np.dtype(variable_ref.spec.dtype),
            order="C",
        )
        if allocation_scope is not None:
            allocation_scope.capture(self._receive_storage)
        self._output_accumulator = self.backend.empty(
            self.plan.output_sharding.local_shape(self.context.rank),
            dtype=np.dtype(self.plan.execution_plan.output.spec.dtype),
            order="C",
        )
        if allocation_scope is not None:
            allocation_scope.capture(self._output_accumulator)

    def _discard_active_call_resources(self):
        self._receive_storage = None
        self._output_accumulator = None

    def _ensure_call_resources(self, allocation_scope):
        resources = (self._receive_storage, self._output_accumulator)
        if all(resource is not None for resource in resources):
            for resource in resources:
                allocation_scope.capture(resource)
            return
        if any(resource is not None for resource in resources):
            self._discard_active_call_resources()
        variable_ref = next(
            ref
            for ref in self.plan.execution_plan.inputs
            if ref.key == self.plan.variable_key
        )
        self._receive_storage = self.backend.empty(
            (self.plan.input_sharding.max_local_elements,),
            dtype=np.dtype(variable_ref.spec.dtype),
            order="C",
        )
        allocation_scope.capture(self._receive_storage)
        self._output_accumulator = self.backend.empty(
            self.plan.output_sharding.local_shape(self.context.rank),
            dtype=np.dtype(self.plan.execution_plan.output.spec.dtype),
            order="C",
        )
        allocation_scope.capture(self._output_accumulator)

    def _prepare_call_bindings(self, local_vector, call=None):
        stack = ExitStack()
        if call is not None:
            call.owner.capture_resources(stack)
        leases = []
        local_error = None
        try:
            self._output_accumulator.fill(0)
            for source_rank in range(self.plan.world_size):
                block = self.plan.block_plan(self.context.rank, source_rank)
                if source_rank == self.context.rank:
                    broadcast_variable = local_vector
                else:
                    broadcast_variable = self._receive_view(
                        self._receive_storage,
                        self.plan.input_sharding.local_shape(source_rank),
                    )
                request = OperandRequest(
                    execution_plan=block.execution_plan,
                    source_bindings=self.source_bindings,
                    distributed_plan=self.plan,
                    context=self.context,
                    source_rank=source_rank,
                    broadcast_variable=broadcast_variable,
                    operand_slices=block.operand_slices,
                    residency_request=self.residency_request,
                    residency_plan=self.residency_plan,
                )
                leases.append(stack.enter_context(self.provider.acquire(request)))
        except BaseException as error:
            local_error = error
            if call is not None:
                call.record_primary(error)

        if call is not None and (
            call.owner.state == "quarantined"
            or self.provider._provider._terminal_error is not None
            or self.provider._provider.runtime._terminal_error is not None
            or getattr(self.collective, "_fatal_error", None) is not None
        ):
            primary = call.mark_communicator_fatal(
                call.primary_error
                or local_error
                or self.provider._provider.runtime._terminal_error
                or getattr(self.collective, "_fatal_error", None)
            )
            self.provider._enter_communicator_fatal(primary, call.owner)
            raise primary

        try:
            error_code = (
                self._allreduce_execution_status(
                    local_error, synchronize=self.context.world_size > 1
                )
                if call is None
                else self._allreduce_active_status(call, local_error)
            )
        except BaseException as status_error:
            if call is not None:
                call.record_primary(status_error)
            try:
                stack.close()
            except BaseException as cleanup_error:
                if call is not None:
                    call.record_secondary(cleanup_error)
            raise
        if error_code:
            try:
                stack.close()
            except BaseException as cleanup_error:
                if call is not None:
                    if call.primary_error is None:
                        call.record_primary(cleanup_error)
                    else:
                        call.record_secondary(cleanup_error)
                    raise call.primary_error
                raise
            if local_error is not None:
                raise ValueError(
                    "distributed resource preflight failed"
                ) from local_error
            raise ValueError("distributed resource preflight failed")
        return stack, tuple(leases)

    def _active_call_ready_error(self, local_vector):
        if self._setup_error is not None:
            return self._setup_error
        try:
            if (
                self.provider.request is not self.residency_request
                or self.provider.plan is not self.residency_plan
                or self.provider.receipt is not self.residency_receipt
            ):
                raise ValueError("working-set receipt metadata does not match operator")
            self.residency_plan.validate_capacity()
            rank = self.context.rank
            if (
                self.residency_plan.device_peak_bytes[rank]
                > self.residency_plan.device_budget.resolved_bytes
                or self.residency_plan.host_required_bytes
                > self.residency_plan.host_budget.resolved_bytes
            ):
                raise ValueError("working-set capacity does not match operator")
            local_code = self._local_vector_error(local_vector)
            if local_code:
                raise ValueError(
                    "local vector preflight failed: {}".format(
                        _PREFLIGHT_ERRORS.get(local_code, "unknown")
                    )
                )
            from renormalizer.backend._distributed.async_owner import allocation_record

            allocation_record(local_vector)
        except BaseException as error:
            return error
        return None

    def _call_active(self, local_vector, operator_call):
        try:
            with operator_call(local_vector) as call:
                self._active_allocation_scope = call.scope
                ready_error = (
                    call.entry_error
                    if call.entry_error is not None
                    else self._active_call_ready_error(local_vector)
                )
                if ready_error is not None:
                    call.record_primary(ready_error)
                if self._allreduce_active_status(call, ready_error):
                    if call.entry_error is not None:
                        raise call.entry_error
                    if ready_error is not None:
                        raise ValueError(
                            "distributed call-ready preflight failed"
                        ) from ready_error
                    raise ValueError("distributed call-ready preflight failed")

                allocation_error = None
                try:
                    call.owner.capture_arrays(local_vector)
                    self._ensure_call_resources(call.scope)
                except BaseException as error:
                    allocation_error = error
                    call.record_primary(error)
                if self._allreduce_active_status(call, allocation_error):
                    if allocation_error is not None:
                        raise ValueError(
                            "distributed call-storage preflight failed"
                        ) from allocation_error
                    raise ValueError("distributed call-storage preflight failed")

                stack, leases = self._prepare_call_bindings(local_vector, call)
                local_output = self._output_accumulator
                with stack:
                    for source_rank, lease in enumerate(leases):
                        block = self.plan.block_plan(self.context.rank, source_rank)
                        broadcast_variable = lease.bindings.arrays[
                            self.plan.variable_key
                        ]
                        admission = getattr(
                            self.collective, "_active_broadcast_admission", None
                        )
                        boundary = (
                            nullcontext(
                                (
                                    self.collective.broadcast,
                                    self.collective._agree_active_broadcast,
                                )
                            )
                            if admission is None
                            else admission()
                        )
                        with boundary as (broadcast, agree):
                            broadcast_error = None
                            try:
                                broadcast(broadcast_variable, root=source_rank)
                            except BaseException as error:
                                broadcast_error = error
                                primary = call.mark_communicator_fatal(error)
                                if call.owner.state not in {
                                    "detached",
                                    "quarantined",
                                }:
                                    call.owner.force_quarantine(primary)
                                self.provider._enter_communicator_fatal(
                                    primary, call.owner
                                )
                            try:
                                agree(broadcast_error is not None)
                            except BaseException as error:
                                fatal_error = getattr(
                                    self.collective, "_fatal_error", None
                                )
                                if (
                                    fatal_error is not None
                                    and error.__cause__ is fatal_error
                                ):
                                    error = fatal_error
                                if broadcast_error is None:
                                    primary = call.mark_communicator_fatal(error)
                                    self.provider._enter_communicator_fatal(
                                        primary, call.owner
                                    )
                                else:
                                    primary = call.primary_error
                                    call.record_secondary(error)
                                raise primary
                            if broadcast_error is not None:
                                raise call.primary_error
                        self._increment_counter("broadcast_calls")
                        local_error = None
                        contribution = None
                        try:
                            contribution = self.backend._execute_plan_with_scope(
                                block.execution_plan,
                                lease.bindings,
                                workspace=block.execution_plan.workspace_bytes,
                                allocation_scope=call.scope,
                            )
                            self._increment_counter("execution_calls")
                            self._active_contribution = contribution
                            self._accumulate_contribution(local_output, contribution)
                        except BaseException as error:
                            local_error = error
                            call.record_primary(error)
                        finally:
                            self._active_contribution = None
                            contribution = None
                        failed = self._allreduce_active_status(call, local_error)
                        if failed:
                            if local_error is not None:
                                raise RuntimeError(
                                    "distributed block execution failed"
                                ) from local_error
                            raise RuntimeError("distributed block execution failed")
                return local_output
        except BaseException:
            self._discard_active_call_resources()
            raise
        finally:
            self._active_allocation_scope = None

    def _call_device_resident(self, local_vector):
        self._preflight_setup()
        self._preflight_plan_agreement()
        self._preflight_local_vector(local_vector)
        self._preflight_capacity()
        try:
            with nullcontext(None) as allocation_scope:
                self._active_allocation_scope = allocation_scope
                allocation_error = None
                try:
                    self._ensure_resident_call_resources(allocation_scope)
                except BaseException as error:
                    allocation_error = error
                if not self._resource_allocation_preflight_complete:
                    if (
                        self._execution_status is None
                        or self._host_execution_status is None
                    ):
                        if allocation_scope is not None:
                            raise ValueError(
                                "distributed resource preflight failed"
                            ) from allocation_error
                        error_code = self._allreduce_status(1)
                    else:
                        error_code = self._allreduce_execution_status(
                            allocation_error,
                            synchronize=self.context.world_size > 1,
                        )
                    if error_code:
                        if allocation_error is not None:
                            raise ValueError(
                                "distributed resource preflight failed"
                            ) from allocation_error
                        raise ValueError("distributed resource preflight failed")
                    self._resource_allocation_preflight_complete = True
                stack, leases = self._prepare_call_bindings(local_vector)
                local_output = self._output_accumulator
                with stack:
                    for source_rank, lease in enumerate(leases):
                        block = self.plan.block_plan(self.context.rank, source_rank)
                        broadcast_variable = lease.bindings.arrays[
                            self.plan.variable_key
                        ]
                        self.collective.broadcast(broadcast_variable, root=source_rank)
                        self._increment_counter("broadcast_calls")
                        local_error = None
                        contribution = None
                        try:
                            contribution = self.backend.execute_plan(
                                block.execution_plan,
                                lease.bindings,
                                workspace=block.execution_plan.workspace_bytes,
                            )
                            self._increment_counter("execution_calls")
                            self._active_contribution = contribution
                            self._accumulate_contribution(local_output, contribution)
                        except BaseException as error:
                            local_error = error
                        finally:
                            self._active_contribution = None
                            contribution = None
                        failed = self._allreduce_execution_status(local_error)
                        if failed:
                            if local_error is not None:
                                raise RuntimeError(
                                    "distributed block execution failed"
                                ) from local_error
                            raise RuntimeError("distributed block execution failed")
                return local_output
        except BaseException:
            self._discard_execution_resources()
            raise
        finally:
            self._active_allocation_scope = None

    def __call__(self, local_vector):
        operator_call = getattr(self.provider, "_operator_call", None)
        if callable(operator_call):
            return self._call_active(local_vector, operator_call)
        return self._call_device_resident(local_vector)


__all__ = ["DistributedLocalOperator", "run_root_fallback"]
