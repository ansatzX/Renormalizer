"""Ordered source-broadcast execution for output-sharded local H-v."""

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from renormalizer.backend._distributed.context import DistributedContext
from renormalizer.backend._distributed.planner import DistributedPlan
from renormalizer.backend._distributed.providers import OperandRequest
from renormalizer.backend._execution.model import ExecutionBindings


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
    fingerprint = _control_array(
        collective, _fallback_fingerprint(metadata), np.uint64
    )
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

    def __post_init__(self):
        if not callable(getattr(self.collective, "allreduce", None)):
            raise TypeError("collective must provide a usable allreduce")
        self._bootstrap_world_size = _bootstrap_schedule_size(
            self.context, self.plan, self.collective
        )
        self._counter_error = _counter_validation_error(self.counters)
        self._counter_target = (
            self.counters if self._counter_error is None else None
        )
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
        if not callable(getattr(self.collective, "allreduce_inplace", None)):
            raise TypeError("collective must implement in-place allreduce")
        for name in ("device_memory_budget_bytes", "host_memory_budget_bytes"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("{} must be a non-negative integer or None".format(name))
        if self.plan.variable_key in self.source_bindings.arrays:
            raise ValueError(
                "resident source bindings must not contain the variable input"
            )
        expected_keys = {
            ref.key for ref in self.plan.execution_plan.inputs
        } - {self.plan.variable_key}
        actual_keys = set(self.source_bindings.arrays)
        if actual_keys != expected_keys:
            raise ValueError("resident source binding coverage is incomplete")
        self._validate_source_arrays()
        self._validate_resident_views()

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
        self._preflight_setup()
        self._preflight_plan_agreement()
        self._preflight_capacity()

    def _validate_source_arrays(self):
        refs = {ref.key: ref for ref in self.plan.execution_plan.inputs}
        for key, array in self.source_bindings.arrays.items():
            ref = refs[key]
            self.backend._validate_execution_array(array)
            if tuple(array.shape) != ref.spec.shape:
                raise ValueError("resident source shape does not match execution plan")
            if np.dtype(array.dtype).name != ref.spec.dtype:
                raise ValueError("resident source dtype does not match execution plan")

    def _validate_resident_views(self):
        validator = getattr(self.provider, "validate_resident", None)
        if validator is None:
            raise NotImplementedError(
                "Stage 4 operand providers must support deterministic resident preflight"
            )
        for source_rank in range(self.plan.world_size):
            block = self.plan.block_plan(self.context.rank, source_rank)
            validator(
                OperandRequest(
                    execution_plan=block.execution_plan,
                    source_bindings=self.source_bindings,
                    distributed_plan=self.plan,
                    context=self.context,
                    source_rank=source_rank,
                    broadcast_variable=None,
                    operand_slices=block.operand_slices,
                )
            )

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
        self._increment_counter(
            "allreduce_calls", mirror=mirror_counters
        )
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
        chunks = [
            int(self.plan.placement_hash[index : index + 16], 16)
            for index in range(0, 64, 16)
        ]
        local_hash = _control_array(self.collective, chunks, np.uint64)
        minimum = self.collective.allreduce(local_hash, op="min")
        maximum = self.collective.allreduce(local_hash, op="max")
        self._increment_counter("allreduce_calls", amount=2)
        local_values = _array_to_numpy(local_hash)
        if not (
            np.array_equal(_array_to_numpy(minimum), local_values)
            and np.array_equal(_array_to_numpy(maximum), local_values)
        ):
            raise ValueError("distributed placement hash disagreement")
        self._plan_preflight_complete = True

    def _preflight_capacity(self):
        if self._capacity_preflight_complete:
            return
        estimate = self.plan.memory_estimates[self.context.rank]
        local_code = 0
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
        view = self.backend.reshape(prefix, shape)
        if not self.backend._is_exact_execution_reshape(prefix, view):
            raise ValueError("reusable receive reshape unexpectedly copied data")
        if not bool(view.flags.c_contiguous):
            raise ValueError("reusable receive view must be C contiguous")
        return view

    def _discard_execution_resources(self):
        self._receive_storage = None
        self._output_accumulator = None
        self._execution_status = None
        self._host_execution_status = None

    def _copy_execution_status_to_host(self):
        status = self._execution_status
        host_status = self._host_execution_status
        if self.backend.name == "cupy":
            cupy = self.backend._cupy
            with cupy.cuda.Device(self.backend._device_index):
                copied = cupy.asnumpy(status, out=host_status, blocking=True)
            if copied is not host_status:
                raise RuntimeError("CuPy status copy replaced host staging storage")
        else:
            np.copyto(host_status, status, casting="no")
        return int(host_status[0])

    def _allreduce_execution_status(self, local_error):
        status = self._execution_status
        status.fill(1 if local_error is not None else 0)
        synchronized = self.collective.allreduce_inplace(status, op="max")
        self._increment_counter("allreduce_calls")
        if synchronized is not status:
            raise RuntimeError("in-place allreduce replaced execution status storage")
        return self._copy_execution_status_to_host()

    @staticmethod
    def _accumulate_contribution(output, contribution):
        output += contribution

    def _prepare_call_bindings(self, local_vector):
        new_resources = self._receive_storage is None
        receive_storage = self._receive_storage
        output_accumulator = self._output_accumulator
        execution_status = self._execution_status
        host_execution_status = self._host_execution_status
        stack = ExitStack()
        leases = []
        local_error = None
        try:
            variable_ref = next(
                ref
                for ref in self.plan.execution_plan.inputs
                if ref.key == self.plan.variable_key
            )
            if new_resources:
                receive_storage = self.backend.empty(
                    (self.plan.input_sharding.max_local_elements,),
                    dtype=np.dtype(variable_ref.spec.dtype),
                    order="C",
                )
                output_accumulator = self.backend.empty(
                    self.plan.output_sharding.local_shape(self.context.rank),
                    dtype=np.dtype(self.plan.execution_plan.output.spec.dtype),
                    order="C",
                )
                execution_status = self.backend.empty(
                    (1,), dtype=np.int32, order="C"
                )
                host_execution_status = np.empty(
                    (1,), dtype=np.int32, order="C"
                )
            output_accumulator.fill(0)
            for source_rank in range(self.plan.world_size):
                block = self.plan.block_plan(self.context.rank, source_rank)
                if source_rank == self.context.rank:
                    broadcast_variable = local_vector
                else:
                    broadcast_variable = self._receive_view(
                        receive_storage,
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
                )
                leases.append(stack.enter_context(self.provider.acquire(request)))
        except BaseException as error:
            local_error = error

        try:
            error_code = self._allreduce_status(1 if local_error is not None else 0)
        except BaseException as status_error:
            try:
                stack.close()
            except BaseException as cleanup_error:
                self._discard_execution_resources()
                raise status_error from cleanup_error
            self._discard_execution_resources()
            raise
        if error_code:
            stack.close()
            self._discard_execution_resources()
            if local_error is not None:
                raise ValueError("distributed resource preflight failed") from local_error
            raise ValueError("distributed resource preflight failed")
        self._receive_storage = receive_storage
        self._output_accumulator = output_accumulator
        self._execution_status = execution_status
        self._host_execution_status = host_execution_status
        return stack, tuple(leases)

    def __call__(self, local_vector):
        self._preflight_setup()
        self._preflight_plan_agreement()
        self._preflight_local_vector(local_vector)
        self._preflight_capacity()
        stack, leases = self._prepare_call_bindings(local_vector)
        local_output = self._output_accumulator
        with stack:
            for source_rank, lease in enumerate(leases):
                block = self.plan.block_plan(self.context.rank, source_rank)
                broadcast_variable = lease.bindings.arrays[self.plan.variable_key]
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


__all__ = ["DistributedLocalOperator", "run_root_fallback"]
