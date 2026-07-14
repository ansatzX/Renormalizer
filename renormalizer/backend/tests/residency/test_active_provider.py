from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
import builtins
import gc
import inspect
import json
import os
import threading
import time
from types import MethodType, SimpleNamespace
import weakref

import numpy as np
import pytest

from renormalizer.backend._distributed.center import CenterVectorMap
from renormalizer.backend._distributed.async_owner import (
    AsyncResourceOwner,
    RuntimeTerminalQuarantine,
    allocation_record,
)
from renormalizer.backend._distributed.cache import CacheEntrySpec, DeviceTensorCache
from renormalizer.backend._distributed.collectives import SingleProcessCollective
from renormalizer.backend._distributed.context import (
    DistributedContext,
    DistributedRendezvous,
)
from renormalizer.backend._distributed.local_operator import DistributedLocalOperator
from renormalizer.backend._distributed.mesh import DeviceMesh
from renormalizer.backend._distributed.planner import plan_distributed_execution
from renormalizer.backend._distributed.providers import (
    ActiveWorkingSetProvider,
    OperandRequest,
)
from renormalizer.backend._distributed.pinned import PinnedBufferPool
from renormalizer.backend._distributed.residency import (
    HostTensorError,
    HostTensorAllocation,
    HostTensorStore,
    MemoryBudgetResolution,
    ResidencyPlanner,
    ResidencyRequest,
)
from renormalizer.backend._distributed.solvers import build_krylov_memory_profile
from renormalizer.backend._distributed.terminal import _TerminalPhase
from renormalizer.backend._distributed.transfer import (
    TransferScheduler,
    TransferTicket,
)
from renormalizer.backend._execution.model import ExecutionBindings
from renormalizer.backend._gemm.mps_lowering import build_mps_ir_hop
from renormalizer.backend.abstract import AbstractBackend
from renormalizer.backend.config import BackendConfig, DistributedExecutionConfig
from renormalizer.backend.cupy_backend import CupyBackend
from renormalizer.backend.distributed_runtime import (
    CupyDistributedRuntime,
    active_working_set_execution,
)
from renormalizer.backend.factory import create_backend


def _has_one_visible_gpu():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return False
    return len([item for item in visible.split(",") if item.strip()]) == 1


class _ManualEvent:
    def __init__(self, *, done=False):
        self.done = done
        self.record_calls = 0

    def record(self, stream=None):
        self.record_calls += 1

    def query(self):
        return self.done

    def synchronize(self):
        self.done = True

    def complete(self):
        self.done = True


class _FakeDeviceContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _FakeTransferStream:
    def __init__(self, *, synchronize_fails=False):
        self.synchronize_calls = 0
        self.synchronize_fails = synchronize_fails

    def wait_event(self, event):
        pass

    def synchronize(self):
        self.synchronize_calls += 1
        if self.synchronize_fails:
            raise RuntimeError("injected scheduler drain failure")


class _FakeCuda:
    def __init__(self, *, synchronize_fails=False):
        self.stream = _FakeTransferStream(synchronize_fails=synchronize_fails)

    def Device(self, index):
        return _FakeDeviceContext()

    def Stream(self, *, non_blocking):
        assert non_blocking is True
        return self.stream

    def get_current_stream(self):
        return self.stream


class _FakeCupy:
    def __init__(self, *, synchronize_fails=False):
        self.cuda = _FakeCuda(synchronize_fails=synchronize_fails)


class _FakeCupyBackend:
    name = "cupy"
    _device_index = 0

    def __init__(self, *, synchronize_fails=False):
        self._cupy = _FakeCupy(synchronize_fails=synchronize_fails)


class _FakeDestination:
    shape = (4,)
    dtype = np.dtype(np.float64)
    nbytes = 32

    class flags:
        c_contiguous = True
        f_contiguous = True

    def __init__(self):
        self.enqueued = False

    def __renormalizer_allocation__(self):
        return self, id(self), self.nbytes

    def set(self, source, *, stream):
        self.enqueued = True


class _FakeSource:
    shape = (4,)
    dtype = np.dtype(np.float64)
    nbytes = 32

    class flags:
        c_contiguous = True
        f_contiguous = True

    def __init__(self):
        self.enqueued = False

    def __renormalizer_allocation__(self):
        return self, id(self), self.nbytes

    def get(self, *, out, stream, blocking):
        assert blocking is False
        self.enqueued = True


class _RecordFailureEvent(_ManualEvent):
    def record(self, stream=None):
        super().record(stream)
        raise RuntimeError("injected post-enqueue event failure")


class _RecordAndDrainFailureEvent(_ManualEvent):
    def record(self, stream=None):
        super().record(stream)
        raise RuntimeError("injected H2D enqueue failure")

    def synchronize(self):
        raise RuntimeError("injected H2D drain failure")


class _PriorOwnerFailureEvent(_ManualEvent):
    def __init__(self, primary, drain_error, *, failure_operation, drain_fails):
        super().__init__(done=False)
        self.primary = primary
        self.drain_error = drain_error
        self.failure_operation = failure_operation
        self.drain_fails = drain_fails
        self.synchronize_calls = 0

    def query(self):
        if self.failure_operation == "query":
            raise self.primary
        return False

    def synchronize(self):
        self.synchronize_calls += 1
        if self.failure_operation == "wait" and self.synchronize_calls == 1:
            raise self.primary
        if self.drain_fails:
            raise self.drain_error
        self.done = True


class _SchedulerQueryFailureEvent(_ManualEvent):
    def __init__(self, *, synchronize_fails=False):
        super().__init__()
        self.synchronize_calls = 0
        self.synchronize_fails = synchronize_fails

    def query(self):
        raise RuntimeError("injected scheduler query failure")

    def synchronize(self):
        self.synchronize_calls += 1
        if self.synchronize_fails:
            raise RuntimeError("injected scheduler event wait failure")
        self.done = True


class _SchedulerWaitFailureEvent(_ManualEvent):
    def synchronize(self):
        raise RuntimeError("injected scheduler wait failure")


class _FakeStagingSlot:
    pinned = True

    def __init__(self):
        self.event = None

    def view(self, shape, dtype, *, order="C"):
        return np.empty(shape, dtype=dtype, order=order)

    def retain_until(self, event):
        self.event = event


class _LoopbackCollective:
    def __init__(self, size):
        self.rank = 0
        self.size = size
        self._fatal_error = None
        self._fatal_handler = None
        self._active_broadcast_sequence = 0

    def allreduce(self, array, *, op="sum"):
        return np.array(array, copy=True)

    def allreduce_inplace(self, array, *, op="sum"):
        return array

    def _bootstrap_status_or(self, local_code):
        return local_code

    def _bootstrap_fatal_control(self):
        return 0

    def _install_fatal_handler(self, handler):
        self._fatal_handler = handler

    def _publish_communicator_fatal(self, error):
        if self._fatal_error is None:
            self._fatal_error = error

    def _agree_active_broadcast(self, failed):
        self._active_broadcast_sequence += 1
        if failed:
            raise RuntimeError("collective is terminal-aborted") from self._fatal_error

    def broadcast(self, array, *, root):
        return array

    def close(self):
        return None


class _TwoRankCollectiveGroup:
    def __init__(self, *, timeout=3.0):
        self.timeout = timeout
        self._condition = threading.Condition()
        self._operations = {}
        self._bootstrap_codes = [None, None]
        self._bootstrap_publish = threading.Barrier(2, timeout=timeout)
        self._bootstrap_consume = threading.Barrier(2, timeout=timeout)
        self.active = False
        self.status_failures = {}
        self.status_hook = None
        self.fatal_capability_codes = [0, 0]
        self.broadcast_failures = {}
        self.fatal_hook = None
        self.fatal = None
        self.fatal_publications = []
        self.abort_calls = [0, 0]
        self.fatal_acks = [False, False]
        self._fatal_notifications = set()
        self._fatal_threads = []
        self.active_broadcast_records = {0: (0, False), 1: (0, False)}
        self.active_broadcast_consumed = [0, 0]
        self.active_broadcast_overwrites = []
        self.close_ready = [False, False]
        self.monitor_stopped = [False, False]
        self.close_release = False
        self.close_consumed = [False, False]
        self.store_stop_rank = None
        self.endpoints = tuple(_TwoRankCollective(self, rank) for rank in range(2))

    def exchange(self, rank, index, kind, value, *, op=None, root=None):
        payload = np.array(value, copy=True)
        fatal = None
        with self._condition:
            slot = self._operations.setdefault(index, {"calls": {}, "readers": 0})
            if rank in slot["calls"]:
                raise AssertionError("rank repeated a fake collective slot")
            slot["calls"][rank] = (kind, op, root, payload)
            self._condition.notify_all()
            if not self._condition.wait_for(
                lambda: len(slot["calls"]) == 2 or self.fatal is not None,
                timeout=self.timeout,
            ):
                raise AssertionError("two-rank fake collective did not terminate")
            if len(slot["calls"]) != 2:
                fatal = self.fatal
            else:
                calls = slot["calls"]
                descriptors = {(call[0], call[1], call[2]) for call in calls.values()}
                if len(descriptors) != 1:
                    raise AssertionError(
                        "two-rank collective schedule diverged: {!r}".format(
                            descriptors
                        )
                    )
                if "result" not in slot:
                    left = calls[0][3]
                    right = calls[1][3]
                    if kind == "broadcast":
                        slot["result"] = np.array(calls[root][3], copy=True)
                    elif op == "max":
                        slot["result"] = np.maximum(left, right)
                    elif op == "min":
                        slot["result"] = np.minimum(left, right)
                    elif op == "sum":
                        slot["result"] = left + right
                    else:
                        raise AssertionError(
                            "unsupported fake reduction {!r}".format(op)
                        )
                result = np.array(slot["result"], copy=True)
                slot["readers"] += 1
                if slot["readers"] == 2:
                    del self._operations[index]
                    self._condition.notify_all()
                return result
        origin_rank, origin_error = fatal
        self.endpoints[rank]._observe_remote_fatal(origin_rank, origin_error)
        self.endpoints[rank]._raise_terminal()

    def bootstrap_status_or(self, rank, local_code, trace):
        self._bootstrap_codes[rank] = local_code
        trace.append(("BOOTSTRAP_SET", local_code))
        self._bootstrap_publish.wait()
        trace.append(("BOOTSTRAP_BARRIER", 1))
        aggregate = 0
        for source_rank in range(2):
            trace.append(("BOOTSTRAP_GET", source_rank))
            aggregate |= self._bootstrap_codes[source_rank]
        self._bootstrap_consume.wait()
        trace.append(("BOOTSTRAP_BARRIER", 2))
        return aggregate

    def publish_fatal(self, rank, error):
        notify = None
        with self._condition:
            if self.fatal is None:
                self.fatal = (rank, error)
            if self.abort_calls[rank] == 0:
                if self.fatal_hook is not None:
                    self.fatal_hook(rank, error)
                self.abort_calls[rank] = 1
                self.fatal_publications.append(rank)
            self.fatal_acks[rank] = True
            peer_rank = 1 - rank
            peer_pending = self.endpoints[peer_rank]._pending_fatal_error
            if (
                not self.status_failures
                and peer_pending is None
                and peer_rank not in self._fatal_notifications
            ):
                self._fatal_notifications.add(peer_rank)
                notify = (peer_rank, *self.fatal)
            self._condition.notify_all()
        if notify is not None:
            peer_rank, origin_rank, origin_error = notify
            thread = threading.Thread(
                target=self.endpoints[peer_rank]._observe_remote_fatal,
                args=(origin_rank, origin_error),
                daemon=True,
            )
            self._fatal_threads.append(thread)
            thread.start()

    def join_fatal_threads(self):
        for thread in tuple(self._fatal_threads):
            thread.join(self.timeout)
            assert not thread.is_alive()

    def wait_for_fatal_acks(self):
        with self._condition:
            if not self._condition.wait_for(
                lambda: all(self.fatal_acks), timeout=self.timeout
            ):
                raise AssertionError("two-rank fatal acknowledgment did not terminate")

    def publish_active_broadcast(self, rank, sequence, failed):
        with self._condition:
            current_sequence, _ = self.active_broadcast_records[rank]
            if sequence <= current_sequence:
                return self.fatal, RuntimeError(
                    "active broadcast result sequence is stale"
                )
            if sequence > current_sequence + 1:
                return self.fatal, RuntimeError(
                    "active broadcast result sequence advanced unexpectedly"
                )
            if current_sequence and not self._condition.wait_for(
                lambda: all(
                    consumed >= current_sequence
                    for consumed in self.active_broadcast_consumed
                ),
                timeout=self.timeout,
            ):
                raise AssertionError(
                    "active broadcast overwrite waited for incomplete consumption"
                )
            if current_sequence:
                self.active_broadcast_overwrites.append(
                    (rank, sequence, tuple(self.active_broadcast_consumed))
                )
            self.active_broadcast_records[rank] = (sequence, bool(failed))
            self._condition.notify_all()
            return self.fatal, None

    def wait_active_broadcast(self, rank, sequence):
        with self._condition:
            if not self._condition.wait_for(
                lambda: all(
                    record_sequence >= sequence
                    for record_sequence, _ in self.active_broadcast_records.values()
                ),
                timeout=self.timeout,
            ):
                raise AssertionError("active broadcast agreement did not terminate")
            advanced = any(
                record_sequence > sequence
                for record_sequence, _ in self.active_broadcast_records.values()
            )
            if advanced:
                return (
                    False,
                    self.fatal,
                    RuntimeError(
                        "active broadcast result sequence advanced unexpectedly"
                    ),
                )
            self.active_broadcast_consumed[rank] = sequence
            failed = any(
                record_failed
                for record_sequence, record_failed in self.active_broadcast_records.values()
                if record_sequence == sequence
            )
            self._condition.notify_all()
            return failed, self.fatal, None

    def close_endpoint(self, rank):
        with self._condition:
            self.close_ready[rank] = True
            self._condition.notify_all()
            if not self._condition.wait_for(
                lambda: all(self.close_ready), timeout=self.timeout
            ):
                raise AssertionError("two-rank close intent did not terminate")
            if self.fatal is not None and not all(self.fatal_acks):
                raise AssertionError("two-rank close preceded fatal consumption")
            self.monitor_stopped[rank] = True
            self._condition.notify_all()
            if not self._condition.wait_for(
                lambda: all(self.monitor_stopped), timeout=self.timeout
            ):
                raise AssertionError("two-rank monitor stop did not terminate")
            if rank == 0:
                self.close_release = True
                self._condition.notify_all()
            elif not self._condition.wait_for(
                lambda: self.close_release, timeout=self.timeout
            ):
                raise AssertionError("two-rank close release did not terminate")
            self.close_consumed[rank] = True
            self._condition.notify_all()
            if rank == 0:
                if not self._condition.wait_for(
                    lambda: all(self.close_consumed), timeout=self.timeout
                ):
                    raise AssertionError("two-rank close consumption did not terminate")
                self.store_stop_rank = 0

    def start_active(self):
        if self._operations:
            raise AssertionError("preflight fake collectives are still pending")
        self.active = True
        for endpoint in self.endpoints:
            endpoint.reset_active_trace()


class _TwoRankCollective:
    size = 2

    def __init__(self, group, rank):
        self.group = group
        self.rank = rank
        self.trace = []
        self.status_arrays = []
        self._operation_index = 0
        self._status_index = 0
        self._active_broadcast_sequence = 1
        self._fatal_error = None
        self._fatal_handler = None
        self._pending_fatal_error = None
        self._fatal_control_initialized = False
        self._closed = False
        self._active_broadcast_local = threading.local()

    def _exchange(self, kind, value, *, op=None, root=None):
        self._require_operational()
        index = self._operation_index
        self._operation_index += 1
        return self.group.exchange(self.rank, index, kind, value, op=op, root=root)

    def reset_active_trace(self):
        self.trace = []
        self.status_arrays = []
        self._operation_index = 0
        self._status_index = 0

    def _bootstrap_fatal_control(self):
        aggregate = self.group.bootstrap_status_or(
            self.rank, self.group.fatal_capability_codes[self.rank], self.trace
        )
        self._fatal_control_initialized = aggregate == 0
        return aggregate

    def _install_fatal_handler(self, handler):
        self._fatal_handler = handler

    def _raise_terminal(self):
        raise RuntimeError("collective is terminal-aborted") from self._fatal_error

    def _observe_remote_fatal(self, origin_rank, origin_error):
        if self._fatal_error is None:
            self._fatal_error = RuntimeError(
                "remote communicator failure from rank {}".format(origin_rank)
            )
            if self._fatal_handler is not None:
                self._fatal_handler(self._fatal_error)
        self.group.publish_fatal(self.rank, self._fatal_error)
        self.group.wait_for_fatal_acks()

    def _require_operational(self):
        if self._fatal_error is not None:
            self._raise_terminal()
        fatal = self.group.fatal
        if fatal is not None:
            self._observe_remote_fatal(*fatal)
            self._raise_terminal()

    def _publish_communicator_fatal_now(self, error):
        if self._fatal_error is None:
            self._fatal_error = error
        self.group.publish_fatal(self.rank, self._fatal_error)
        self.group.wait_for_fatal_acks()

    def _publish_communicator_fatal(self, error):
        admission = getattr(self._active_broadcast_local, "admission", None)
        if admission is not None:
            if admission[0] is None:
                admission[0] = error
            if self._fatal_error is None:
                self._fatal_error = admission[0]
            return
        self._publish_communicator_fatal_now(error)

    @contextmanager
    def _active_broadcast_admission(self):
        if hasattr(self._active_broadcast_local, "admission"):
            raise RuntimeError("active broadcast admission cannot be nested")
        admission = [None]
        self._active_broadcast_local.admission = admission
        try:
            yield self.broadcast, self._agree_active_broadcast
        finally:
            del self._active_broadcast_local.admission
            if admission[0] is not None:
                self._publish_communicator_fatal_now(admission[0])

    def _agree_active_broadcast(self, failed):
        sequence = self._active_broadcast_sequence
        self._active_broadcast_sequence += 1
        fatal, protocol_error = self.group.publish_active_broadcast(
            self.rank, sequence, failed
        )
        if protocol_error is not None:
            self._publish_communicator_fatal(protocol_error)
            self._raise_terminal()
        if fatal is not None and self._fatal_error is None:
            self._observe_remote_fatal(*fatal)
        any_failed, fatal, protocol_error = self.group.wait_active_broadcast(
            self.rank, sequence
        )
        if protocol_error is not None:
            self._publish_communicator_fatal(protocol_error)
            self._raise_terminal()
        if fatal is not None and self._fatal_error is None:
            self._observe_remote_fatal(*fatal)
        if any_failed or self._fatal_error is not None:
            self._raise_terminal()

    def _bootstrap_status_or(self, local_code):
        return self.group.bootstrap_status_or(self.rank, local_code, self.trace)

    def allreduce(self, array, *, op="sum"):
        self._require_operational()
        self.trace.append(("ALLREDUCE", op))
        if self.group.active:
            raise AssertionError("active H-v used allocating allreduce")
        return self._exchange("allreduce", array, op=op)

    def allreduce_inplace(self, array, *, op="sum"):
        self._require_operational()
        status_index = self._status_index
        self._status_index += 1
        label = (
            ("P0", "P1", "P2")[status_index]
            if status_index < 3
            else "E({})".format(status_index - 3)
        )
        self.trace.append((label, op))
        self.status_arrays.append(array)
        result = self._exchange("inplace:{}".format(label), array, op=op)
        np.copyto(array, result, casting="no")
        if label in self.group.status_failures:
            if self.group.status_hook is not None:
                self.group.status_hook(self.rank, label, array)
            error = self.group.status_failures[label][self.rank]
            self._pending_fatal_error = error
            raise error
        return array

    def broadcast(self, array, *, root):
        self._require_operational()
        label = "B({})".format(root)
        self.trace.append((label, root))
        result = self._exchange("broadcast", array, root=root)
        np.copyto(array, result, casting="no")
        failure = self.group.broadcast_failures.get((label, self.rank))
        if failure is not None:
            raise failure
        return array

    def barrier(self):
        raise AssertionError("fake NCCL barrier must not provision status")

    def close(self):
        if self._closed:
            return
        if self._fatal_control_initialized:
            self.group.close_endpoint(self.rank)
        self._closed = True


def _run_two_rank_threads(operations, *, timeout=5.0):
    start = threading.Barrier(3, timeout=timeout)
    results = [None, None]
    errors = [None, None]

    def run(rank):
        try:
            start.wait()
            results[rank] = operations[rank]()
        except BaseException as error:
            errors[rank] = error

    threads = [
        threading.Thread(target=run, args=(rank,), daemon=True) for rank in range(2)
    ]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout)
    assert all(not thread.is_alive() for thread in threads)
    return results, errors


def test_two_rank_fake_active_broadcast_reuses_fixed_records_after_consumption(
    monkeypatch,
):
    group = _TwoRankCollectiveGroup()
    rank_one_entered = threading.Event()
    release_rank_one = threading.Event()
    publish = group.publish_active_broadcast

    def stagger_first_publish(rank, sequence, failed):
        if rank == 1 and sequence == 1:
            rank_one_entered.set()
            assert release_rank_one.wait(3.0)
        return publish(rank, sequence, failed)

    monkeypatch.setattr(group, "publish_active_broadcast", stagger_first_publish)
    first_results = [None, None]
    first_errors = [None, None]

    def first(rank):
        try:
            first_results[rank] = group.endpoints[rank]._agree_active_broadcast(False)
        except BaseException as error:
            first_errors[rank] = error

    first_threads = [threading.Thread(target=first, args=(rank,)) for rank in range(2)]
    for thread in first_threads:
        thread.start()
    assert rank_one_entered.wait(1.0)
    first_threads[0].join(0.05)
    assert first_threads[0].is_alive()
    release_rank_one.set()
    for thread in first_threads:
        thread.join(5.0)
    assert all(not thread.is_alive() for thread in first_threads)
    assert first_results == [None, None]
    assert first_errors == [None, None]

    for _ in range(2):
        _, errors = _run_two_rank_threads(
            tuple(
                lambda rank=rank: group.endpoints[rank]._agree_active_broadcast(False)
                for rank in range(2)
            )
        )
        assert errors == [None, None]

    assert group.active_broadcast_records == {
        0: (3, False),
        1: (3, False),
    }
    assert group.active_broadcast_consumed == [3, 3]
    assert len(group.active_broadcast_overwrites) == 4
    assert all(
        consumed[0] >= sequence - 1 and consumed[1] >= sequence - 1
        for _, sequence, consumed in group.active_broadcast_overwrites
    )


def test_two_rank_fake_active_broadcast_sequence_ahead_is_fatal():
    group = _TwoRankCollectiveGroup()
    group.active_broadcast_records[1] = (2, False)

    with pytest.raises(RuntimeError, match="terminal-aborted"):
        group.endpoints[0]._agree_active_broadcast(False)

    group.join_fatal_threads()
    assert group.abort_calls == [1, 1]
    assert group.fatal_acks == [True, True]
    assert isinstance(group.endpoints[0]._fatal_error, RuntimeError)
    assert str(group.endpoints[0]._fatal_error) == (
        "active broadcast result sequence advanced unexpectedly"
    )


def _explicit_budget(value, resource):
    return MemoryBudgetResolution(value, value, "explicit", None, resource)


def _specialized_plan(world_size=1, dtype="float64"):
    artifact = build_mps_ir_hop(
        "ab,b->a",
        (np.ones((4, 4), dtype=dtype),),
        (4,),
        "one_site",
    ).resolve_execution_artifact(np.ones(4, dtype=dtype))
    return plan_distributed_execution(
        artifact.execution_plan,
        variable_key=artifact.variable_key,
        world_size=world_size,
    )


def _runtime(*, world_size=1, rank=0, collective=None, backend_name="numpy"):
    context = DistributedContext(rank, rank, world_size, world_size)
    backend = create_backend(
        backend_name,
        config=BackendConfig(
            device="cpu" if backend_name == "numpy" else "cuda:{}".format(rank),
            execution_policy="execution_ir",
            fallback_policy="error",
        ),
    )
    return CupyDistributedRuntime(
        backend=backend,
        context=context,
        rendezvous=DistributedRendezvous("127.0.0.1", 23456),
        mesh=DeviceMesh((world_size,), ("rank",), rank),
        collective=(SingleProcessCollective() if collective is None else collective),
    )


def _active_case(runtime, *, store_id="active-case", dtype="float64", coefficient=1.0):
    dtype = np.dtype(dtype)
    distributed = _specialized_plan(runtime.world_size, dtype.name)
    store = HostTensorStore(store_id=store_id)
    matrix = store.put("input_0", np.arange(16, dtype=dtype).reshape(4, 4))
    center = store.put("output", np.zeros(4, dtype=dtype))
    snapshot = store.snapshot()
    solver_profile = build_krylov_memory_profile(
        distributed.input_sharding,
        dtype.name,
        coefficient=coefficient,
        max_krylov_vectors=2,
    )
    output = HostTensorAllocation("output", (4,), solver_profile.result_dtype)
    request = ResidencyRequest(
        distributed_plan=distributed,
        host_refs={"input_0": matrix, "output": center},
        world_size=runtime.world_size,
        local_world_size=runtime.world_size,
        backend_name=runtime.backend.name,
        store_bytes=None,
        external_host_bytes=(0,) * runtime.world_size,
        transfer_staging_host_bytes=None,
        dirty_writeback_bytes=None,
        solver_input_sharding=distributed.input_sharding,
        solver_output_sharding=distributed.output_sharding,
        mapped_local_counts=tuple(
            distributed.input_sharding.local_shape(rank)[0]
            for rank in range(runtime.world_size)
        ),
        solver_profile=solver_profile,
        materialization_policy="device",
        complete_center_bytes=None,
        prefetch_depth=1,
        future_plans=(),
        device_budget=_explicit_budget(1 << 30, "device"),
        host_budget=_explicit_budget(1 << 31, "host"),
        store_snapshots=(snapshot,) * runtime.world_size,
        writeback_allocations=((output,),) * runtime.world_size,
    )
    plan = ResidencyPlanner().plan(request)
    return request, plan, store, matrix, center


def _block_request(working_set, source_rank=0, broadcast_variable=None):
    request = working_set.request
    distributed = request.distributed_plan
    block = distributed.block_plan(working_set.context.rank, source_rank)
    if broadcast_variable is None:
        broadcast_variable = working_set.backend.zeros(
            distributed.input_sharding.local_shape(source_rank),
            dtype=np.float64,
        )
    return OperandRequest(
        execution_plan=block.execution_plan,
        source_bindings=ExecutionBindings(
            {"input_0": dict(request.host_refs)["input_0"]}
        ),
        distributed_plan=distributed,
        context=working_set.context,
        source_rank=source_rank,
        broadcast_variable=broadcast_variable,
        operand_slices=block.operand_slices,
        residency_request=request,
        residency_plan=working_set.plan,
    )


def _provider(runtime, request, **kwargs):
    return ActiveWorkingSetProvider(
        runtime,
        device_budget_resolution=request.device_budget,
        host_budget_resolution=request.host_budget,
        prefetch_depth=request.prefetch_depth,
        **kwargs,
    )


def _active_execution_config(runtime, request, provider):
    return DistributedExecutionConfig(
        context=runtime.context,
        mesh=runtime.mesh,
        collective=runtime.collective,
        provider=provider,
        residency_policy="active_working_set",
        device_memory_budget_bytes=request.device_budget.requested_bytes,
        host_memory_budget_bytes=request.host_budget.requested_bytes,
        prefetch_depth=request.prefetch_depth,
        backend_name=runtime.backend.name,
        backend_device=str(runtime.backend.current_device()),
        backend_precision=64,
        device_budget_resolution=request.device_budget,
        host_budget_resolution=request.host_budget,
    )


def _active_operator_case(world_size=1, *, store_id, backend_name="numpy"):
    collective = _LoopbackCollective(world_size)
    runtime = _runtime(
        world_size=world_size,
        collective=collective,
        backend_name=backend_name,
    )
    request, plan, store, _, _ = _active_case(runtime, store_id=store_id)
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    runtime._active_provider = provider
    working_set = provider.open_working_set(request, plan, store, receipt).__enter__()
    distributed = request.distributed_plan
    operator = DistributedLocalOperator(
        plan=distributed,
        provider=working_set,
        collective=collective,
        counters={},
        backend=runtime.backend,
        context=runtime.context,
        source_bindings=ExecutionBindings(
            {"input_0": dict(request.host_refs)["input_0"]}
        ),
        residency_request=request,
        residency_plan=plan,
        residency_receipt=receipt,
        mesh=runtime.mesh,
    )
    local_vector = runtime.backend.ones(
        distributed.input_sharding.local_shape(0), dtype=np.float64
    )
    return runtime, store, provider, working_set, operator, local_vector


def _two_rank_unopened_cases(*, store_id):
    group = _TwoRankCollectiveGroup()
    states = [None, None]

    def prepare(rank):
        runtime = _runtime(
            world_size=2,
            rank=rank,
            collective=group.endpoints[rank],
        )
        request, plan, store, _, _ = _active_case(runtime, store_id=store_id)
        receipt = runtime.preflight_residency(request, plan)
        provider = _provider(runtime, request)
        state = {
            "runtime": runtime,
            "request": request,
            "plan": plan,
            "store": store,
            "receipt": receipt,
            "provider": provider,
            "working_set": None,
            "operator": None,
            "local_vector": None,
        }
        states[rank] = state
        return state

    _, errors = _run_two_rank_threads(
        tuple(lambda rank=rank: prepare(rank) for rank in range(2))
    )
    assert errors == [None, None]
    for endpoint in group.endpoints:
        endpoint.trace.clear()
    return group, states


def _open_two_rank_cases(states):
    def open_rank(rank):
        state = states[rank]
        working_set = state["provider"].open_working_set(
            state["request"],
            state["plan"],
            state["store"],
            state["receipt"],
        )
        state["working_set"] = working_set
        distributed = state["request"].distributed_plan
        state["operator"] = DistributedLocalOperator(
            plan=distributed,
            provider=working_set,
            collective=state["runtime"].collective,
            counters={},
            backend=state["runtime"].backend,
            context=state["runtime"].context,
            source_bindings=ExecutionBindings(
                {"input_0": dict(state["request"].host_refs)["input_0"]}
            ),
            residency_request=state["request"],
            residency_plan=state["plan"],
            residency_receipt=state["receipt"],
            mesh=state["runtime"].mesh,
        )
        state["local_vector"] = state["runtime"].backend.ones(
            distributed.input_sharding.local_shape(rank), dtype=np.float64
        )
        return working_set

    return _run_two_rank_threads(
        tuple(lambda rank=rank: open_rank(rank) for rank in range(2))
    )


def _open_two_rank_outer_context_cases(states, context_kind):
    def open_rank(rank):
        state = states[rank]
        if context_kind == "generic":
            manager = active_working_set_execution(
                _active_execution_config(
                    state["runtime"], state["request"], state["provider"]
                ),
                state["request"],
                state["plan"],
                state["store"],
            )
            borrowed = manager.__enter__()
            working_set = borrowed.provider
            receipt = borrowed.residency_receipt
        else:
            manager = state["provider"].open_working_set(
                state["request"],
                state["plan"],
                state["store"],
                state["receipt"],
            )
            working_set = manager.__enter__()
            receipt = state["receipt"]
        state["outer_context"] = manager
        state["working_set"] = working_set
        distributed = state["request"].distributed_plan
        state["operator"] = DistributedLocalOperator(
            plan=distributed,
            provider=working_set,
            collective=state["runtime"].collective,
            counters={},
            backend=state["runtime"].backend,
            context=state["runtime"].context,
            source_bindings=ExecutionBindings(
                {"input_0": dict(state["request"].host_refs)["input_0"]}
            ),
            residency_request=state["request"],
            residency_plan=state["plan"],
            residency_receipt=receipt,
            mesh=state["runtime"].mesh,
        )
        state["local_vector"] = state["runtime"].backend.ones(
            distributed.input_sharding.local_shape(rank), dtype=np.float64
        )
        return working_set

    return _run_two_rank_threads(
        tuple(lambda rank=rank: open_rank(rank) for rank in range(2))
    )


def _close_two_rank_cases(states):
    groups = {
        state["runtime"].collective.group
        for state in states
        if state["runtime"].collective is not None
        and isinstance(state["runtime"].collective, _TwoRankCollective)
    }
    for group in groups:
        group.join_fatal_threads()

    def close_rank(rank):
        runtime = states[rank]["runtime"]
        if not runtime._closed:
            try:
                runtime.close()
            except BaseException:
                pass

    _, close_errors = _run_two_rank_threads(
        tuple(lambda rank=rank: close_rank(rank) for rank in range(2))
    )
    assert close_errors == [None, None]
    for group in groups:
        if all(endpoint._fatal_control_initialized for endpoint in group.endpoints):
            if group.fatal is None:
                assert group.monitor_stopped == [True, True]
                assert group.close_consumed == [True, True]
                assert group.store_stop_rank == 0
            else:
                assert group.monitor_stopped == [False, False]
                assert group.close_consumed == [False, False]
                assert group.store_stop_rank is None
    for state in states:
        runtime = state["runtime"]
        store = state["store"]
        if not store.closed:
            try:
                store.close()
            except BaseException:
                pass


def test_operator_call_owner_precedes_accumulator_fill_and_retains_through_drain(
    monkeypatch,
):
    (
        runtime,
        store,
        provider,
        working_set,
        operator,
        local_vector,
    ) = _active_operator_case(store_id="operator-fill-owner")
    backend = runtime.backend
    original_empty = backend.empty
    allocations = 0
    observed_owner = []
    drained_arrays = []
    error = RuntimeError("injected accumulator fill launch failure")

    class FailingFillArray(np.ndarray):
        def fill(self, value):
            owner = working_set._active_operator_owner
            assert owner.state == "enqueued"
            assert any(array is self for array in owner.arrays)
            assert any(array is local_vector for array in owner.arrays)
            assert len(owner.arrays) >= 5
            observed_owner.append(owner)
            owner._drainer = lambda: drained_arrays.append(tuple(owner.arrays))
            raise error

    def allocate(shape, dtype=float, order="C"):
        nonlocal allocations
        allocations += 1
        array = original_empty(shape, dtype=dtype, order=order)
        return array.view(FailingFillArray) if allocations == 2 else array

    monkeypatch.setattr(backend, "empty", allocate)
    try:
        with pytest.raises(ValueError, match="distributed resource preflight failed"):
            operator(local_vector)

        assert len(observed_owner) == 1
        owner = observed_owner[0]
        assert owner.error is error
        assert owner.state == "detached"
        assert len(drained_arrays) == 1
        assert any(array is local_vector for array in drained_arrays[0])
        assert any(isinstance(array, FailingFillArray) for array in drained_arrays[0])
        assert provider.last_compute_event is None
    finally:
        if not working_set._closed:
            working_set.close()
        store.close()
        runtime.close()


def test_execute_plan_public_signature_remains_exact():
    assert str(inspect.signature(AbstractBackend.execute_plan)) == (
        "(self, plan, bindings, *, stream=None, workspace=None)"
    )


def test_active_operator_supports_old_execute_plan_override_signature(monkeypatch):
    (
        runtime,
        store,
        _,
        working_set,
        operator,
        local_vector,
    ) = _active_operator_case(store_id="old-execute-plan-override")
    backend = runtime.backend
    original_execute_plan = backend.execute_plan
    calls = []

    def old_execute_plan(self, plan, bindings, *, stream=None, workspace=None):
        calls.append((plan, bindings, stream, workspace))
        return original_execute_plan(plan, bindings, stream=stream, workspace=workspace)

    monkeypatch.setattr(backend, "execute_plan", MethodType(old_execute_plan, backend))
    try:
        result = operator(local_vector)
        assert result.shape == operator.plan.output_sharding.local_shape(
            operator.context.rank
        )
        assert len(calls) == operator.plan.world_size
    finally:
        if not working_set._closed:
            working_set.close()
        store.close()
        runtime.close()


def test_two_rank_missing_abort_capability_rejects_before_status_allocation(
    monkeypatch,
):
    group, states = _two_rank_unopened_cases(store_id="two-rank-missing-abort")
    group.fatal_capability_codes[0] = 4
    allocations = []
    for state in states:
        monkeypatch.setattr(
            state["provider"],
            "_allocate_status_device",
            lambda: allocations.append("device"),
        )
        monkeypatch.setattr(
            state["provider"],
            "_allocate_status_host",
            lambda: allocations.append("host"),
        )
    try:
        results, errors = _open_two_rank_cases(states)
        assert results == [None, None]
        assert all(
            isinstance(error, RuntimeError)
            and str(error) == "collective fatal capability bootstrap failed"
            for error in errors
        )
        assert allocations == []
        assert all(state["provider"]._active_lease is None for state in states)
        assert all(state["runtime"]._terminal_error is None for state in states)
    finally:
        _close_two_rank_cases(states)


@pytest.mark.parametrize(
    ("failure_point", "failure_code"),
    (("device_status", 1), ("host_status", 2)),
)
def test_two_rank_status_pair_partial_allocation_uses_cpu_schedule_only(
    monkeypatch, failure_point, failure_code
):
    group, states = _two_rank_unopened_cases(
        store_id="two-rank-status-{}".format(failure_point)
    )
    allocation_error = RuntimeError(
        "injected {} allocation failure".format(failure_point)
    )
    allocation_calls = [[], []]
    allocation_refs = []

    class StatusArray(np.ndarray):
        pass

    def allocate(rank, kind):
        allocation_calls[rank].append(kind)
        if rank == 0 and kind == failure_point:
            raise allocation_error
        array = np.empty((1,), dtype=np.int32).view(StatusArray)
        allocation_refs.append(weakref.ref(array))
        return array

    for rank, state in enumerate(states):
        monkeypatch.setattr(
            state["provider"],
            "_allocate_status_device",
            lambda rank=rank: allocate(rank, "device_status"),
            raising=False,
        )
        monkeypatch.setattr(
            state["provider"],
            "_allocate_status_host",
            lambda rank=rank: allocate(rank, "host_status"),
            raising=False,
        )

    try:
        results, errors = _open_two_rank_cases(states)
        assert results == [None, None]
        assert all(
            isinstance(error, ValueError)
            and str(error) == "working-set status allocation failed"
            for error in errors
        )
        assert errors[0].__cause__ is allocation_error
        assert errors[1].__cause__ is None
        assert allocation_calls == [
            ["device_status", "host_status"],
            ["device_status", "host_status"],
        ]
        for rank, endpoint in enumerate(group.endpoints):
            assert endpoint.trace == [
                ("BOOTSTRAP_SET", 0),
                ("BOOTSTRAP_BARRIER", 1),
                ("BOOTSTRAP_GET", 0),
                ("BOOTSTRAP_GET", 1),
                ("BOOTSTRAP_BARRIER", 2),
                ("BOOTSTRAP_SET", failure_code if rank == 0 else 0),
                ("BOOTSTRAP_BARRIER", 1),
                ("BOOTSTRAP_GET", 0),
                ("BOOTSTRAP_GET", 1),
                ("BOOTSTRAP_BARRIER", 2),
            ]
            assert states[rank]["provider"]._active_lease is None
        errors = None
        gc.collect()
        assert allocation_refs
        assert all(reference() is None for reference in allocation_refs)
    finally:
        _close_two_rank_cases(states)


def test_two_rank_active_hv_uses_fixed_preallocated_inplace_schedule():
    group, states = _two_rank_unopened_cases(store_id="two-rank-fixed-schedule")
    try:
        _, open_errors = _open_two_rank_cases(states)
        assert open_errors == [None, None]
        group.start_active()
        _, call_errors = _run_two_rank_threads(
            tuple(
                lambda rank=rank: states[rank]["operator"](states[rank]["local_vector"])
                for rank in range(2)
            )
        )
        assert call_errors == [None, None]
        expected = ["P0", "P1", "P2", "B(0)", "E(0)", "B(1)", "E(1)"]
        for rank, endpoint in enumerate(group.endpoints):
            assert [entry[0] for entry in endpoint.trace] == expected
            workspace = states[rank]["working_set"]._status_workspace
            assert len(endpoint.status_arrays) == 5
            assert all(
                array is workspace.device_status for array in endpoint.status_arrays
            )
            assert states[rank]["operator"]._counters["allreduce_calls"] == 5
    finally:
        _close_two_rank_cases(states)


def test_two_rank_h2d_failed_drain_aborts_before_p2_and_retains_complete_cohort(
    monkeypatch,
):
    group, states = _two_rank_unopened_cases(store_id="two-rank-h2d-fatal")
    captured_calls = [[], []]
    forbidden = []
    fatal_snapshot = {}
    try:
        _, open_errors = _open_two_rank_cases(states)
        assert open_errors == [None, None]
        for rank, state in enumerate(states):
            working_set = state["working_set"]
            original_operator_call = working_set._operator_call

            @contextmanager
            def capture_call(local_vector, rank=rank, original=original_operator_call):
                with original(local_vector) as call:
                    captured_calls[rank].append(call)
                    yield call

            monkeypatch.setattr(working_set, "_operator_call", capture_call)

            operator = state["operator"]
            original_copy = operator._copy_status_to_host

            def guarded_copy(
                status,
                host_status,
                rank=rank,
                original=original_copy,
                runtime=state["runtime"],
            ):
                if runtime._terminal_error is not None:
                    forbidden.append((rank, "status_d2h"))
                return original(status, host_status)

            monkeypatch.setattr(operator, "_copy_status_to_host", guarded_copy)
            original_execute = state["runtime"].backend._execute_plan_with_scope

            def guarded_execute(
                *args,
                rank=rank,
                original=original_execute,
                runtime=state["runtime"],
                **kwargs,
            ):
                if runtime._terminal_error is not None:
                    forbidden.append((rank, "backend"))
                return original(*args, **kwargs)

            monkeypatch.setattr(
                state["runtime"].backend,
                "_execute_plan_with_scope",
                guarded_execute,
            )

        states[0]["working_set"].scheduler._event_factory = _RecordAndDrainFailureEvent

        def inspect_fatal_publication(rank, error):
            if rank != 0:
                return
            state = states[0]
            scheduler = state["working_set"].scheduler
            owners = tuple(scheduler._quarantined_owners)
            fatal_snapshot.update(
                owners=owners,
                kinds={owner.kind for owner in owners},
                provider=state["provider"]._terminal_error,
                runtime=state["runtime"]._terminal_error,
                backend=state["runtime"].backend._execution_terminal_error,
                error=error,
            )

        group.fatal_hook = inspect_fatal_publication
        group.start_active()
        _, call_errors = _run_two_rank_threads(
            tuple(
                lambda rank=rank: states[rank]["operator"](states[rank]["local_vector"])
                for rank in range(2)
            )
        )

        origin = call_errors[0]
        assert isinstance(origin, RuntimeError)
        assert str(origin) == "injected H2D enqueue failure"
        assert fatal_snapshot["kinds"] >= {"h2d", "compute"}
        assert all(owner.state == "quarantined" for owner in fatal_snapshot["owners"])
        assert fatal_snapshot["provider"] is origin
        assert fatal_snapshot["runtime"] is origin
        assert fatal_snapshot["backend"] is origin
        assert fatal_snapshot["error"] is origin
        assert group.abort_calls == [1, 1]
        assert group.fatal_acks == [True, True]
        assert "P2" not in [entry[0] for entry in group.endpoints[0].trace]
        assert all(
            "P2" not in [entry[0] for entry in endpoint.trace]
            or endpoint._fatal_error is not None
            for endpoint in group.endpoints
        )
        assert forbidden == []
        call = captured_calls[0][0]
        assert call.owner.error is origin
        assert any(isinstance(resource, ExitStack) for resource in call.owner.resources)
        assert call.owner in states[0]["runtime"]._terminal_quarantine.owners
    finally:
        _close_two_rank_cases(states)


@pytest.mark.parametrize("failure_operation", ("query", "wait"))
@pytest.mark.parametrize("drain_fails", (False, True))
def test_two_rank_prior_owner_failure_reaches_p0_or_aborts_without_divergence(
    failure_operation,
    drain_fails,
):
    group, states = _two_rank_unopened_cases(
        store_id="two-rank-prior-{}-{}".format(failure_operation, drain_fails)
    )
    primary = RuntimeError("injected prior {} failure".format(failure_operation))
    drain_error = RuntimeError("injected prior owner drain failure")
    try:
        _, open_errors = _open_two_rank_cases(states)
        assert open_errors == [None, None]
        group.start_active()
        _, first_errors = _run_two_rank_threads(
            tuple(
                lambda rank=rank: states[rank]["operator"](states[rank]["local_vector"])
                for rank in range(2)
            )
        )
        assert first_errors == [None, None]

        prior_owner = states[0]["working_set"]._status_workspace.borrower
        assert prior_owner is not None
        event = _PriorOwnerFailureEvent(
            primary,
            drain_error,
            failure_operation=failure_operation,
            drain_fails=drain_fails,
        )
        prior_owner._completion_event = event
        prior_owner._events = [event]

        group.start_active()
        _, second_errors = _run_two_rank_threads(
            tuple(
                lambda rank=rank: states[rank]["operator"](states[rank]["local_vector"])
                for rank in range(2)
            )
        )

        assert second_errors[0] is primary
        traces = [
            [entry[0] for entry in endpoint.trace] for endpoint in group.endpoints
        ]
        if drain_fails:
            assert group.abort_calls == [1, 1]
            assert all("P1" not in trace for trace in traces)
            assert prior_owner.state == "quarantined"
            assert prior_owner.error is primary
            assert prior_owner.secondary_errors == (drain_error,)
            for rank, state in enumerate(states):
                local_primary = state["runtime"]._terminal_error
                assert local_primary is not None
                assert state["provider"]._terminal_error is local_primary
                assert state["working_set"]._poisoned_error is local_primary
                assert (
                    state["runtime"].backend._execution_terminal_error is local_primary
                )
                assert (
                    state["runtime"]._terminal_quarantine.first_error is local_primary
                )
                assert group.endpoints[rank]._fatal_error is local_primary
        else:
            assert traces == [["P0"], ["P0"]]
            assert prior_owner.state == "detached"
            assert prior_owner.error is primary
            assert group.abort_calls == [0, 0]
            for state in states:
                assert state["working_set"]._poisoned_error is None
                assert state["provider"]._terminal_error is None
                assert state["runtime"]._terminal_error is None
            group.start_active()
            _, third_errors = _run_two_rank_threads(
                tuple(
                    lambda rank=rank: states[rank]["operator"](
                        states[rank]["local_vector"]
                    )
                    for rank in range(2)
                )
            )
            assert third_errors == [None, None]
    finally:
        _close_two_rank_cases(states)


@pytest.mark.parametrize(
    "explicit_reap", (False, True), ids=("borrower-live", "after-explicit-reap")
)
def test_two_rank_peer_fatal_between_calls_quarantines_prior_borrower_exactly(
    monkeypatch, explicit_reap
):
    group, states = _two_rank_unopened_cases(
        store_id="two-rank-between-call-peer-fatal"
    )
    forbidden = []
    expected = []
    try:
        _, open_errors = _open_two_rank_cases(states)
        assert open_errors == [None, None]
        group.start_active()
        _, first_errors = _run_two_rank_threads(
            tuple(
                lambda rank=rank: states[rank]["operator"](states[rank]["local_vector"])
                for rank in range(2)
            )
        )
        assert first_errors == [None, None]

        for rank, state in enumerate(states):
            runtime = state["runtime"]
            working_set = state["working_set"]
            provider = state["provider"]
            owner = working_set._status_workspace.borrower
            assert owner is not None
            assert working_set._active_operator_owner is None
            status_pair = (
                working_set._status_workspace.device_status,
                working_set._status_workspace.host_status,
            )
            assert all(
                any(retained is status for retained in owner.arrays)
                for status in status_pair
            )
            status_records = tuple(allocation_record(status) for status in status_pair)
            assert len({record.identity for record in status_records}) == 2
            assert all(record.capacity_bytes == 4 for record in status_records)
            scheduler = working_set.scheduler
            cache = provider.cache
            pool = working_set.pool
            if explicit_reap:
                working_set.reap_completed()
                assert owner.state == "detached"
                assert working_set._status_workspace.borrower is None
            records = {}
            for record in (
                *(status_records if explicit_reap else owner.allocations),
                *cache.allocation_records,
                *pool.allocation_records,
            ):
                retained = records.get(record.identity)
                if retained is None or record.capacity_bytes > retained.capacity_bytes:
                    records[record.identity] = record
            expected.append(
                {
                    "owner": owner,
                    "arrays": tuple(owner.arrays),
                    "status_pair": status_pair,
                    "status_records": status_records,
                    "records": records,
                    "cache_records": {
                        record.identity: record for record in cache.allocation_records
                    },
                    "pool_records": {
                        record.identity: record for record in pool.allocation_records
                    },
                    "streams": {id(stream) for stream in scheduler.retained_streams},
                    "events": {id(event) for event in scheduler.retained_events},
                }
            )

            event = owner.completion_event
            if event is not None:
                original_query = event.query
                original_synchronize = event.synchronize

                def guarded_query(rank=rank, runtime=runtime, original=original_query):
                    if runtime._terminal_error is not None:
                        forbidden.append((rank, "event.query"))
                    return original()

                def guarded_synchronize(
                    rank=rank, runtime=runtime, original=original_synchronize
                ):
                    if runtime._terminal_error is not None:
                        forbidden.append((rank, "event.synchronize"))
                    return original()

                monkeypatch.setattr(event, "query", guarded_query)
                monkeypatch.setattr(event, "synchronize", guarded_synchronize)
            guarded_resources = (
                ("cache", cache, ("close", "wait_for_pending", "reap_completed")),
                ("scheduler", scheduler, ("close", "complete_all", "reap_completed")),
                ("pool", pool, ("close", "reap_completed")),
            )
            for resource_name, resource, operation_names in guarded_resources:
                for name in operation_names:
                    original = getattr(resource, name)

                    def guarded_operation(
                        *args,
                        rank=rank,
                        runtime=runtime,
                        resource_name=resource_name,
                        name=name,
                        original=original,
                        **kwargs,
                    ):
                        if runtime._terminal_error is not None:
                            forbidden.append(
                                (rank, "{}.{}".format(resource_name, name))
                            )
                        return original(*args, **kwargs)

                    monkeypatch.setattr(resource, name, guarded_operation)

        primary = RuntimeError("injected between-call communicator failure")
        fatal_errors = []

        def publish_fatal():
            try:
                states[0]["runtime"]._enter_communicator_fatal(primary)
            except BaseException as error:
                fatal_errors.append(error)

        publisher = threading.Thread(target=publish_fatal, daemon=True)
        publisher.start()
        publisher.join(5.0)
        assert not publisher.is_alive()
        group.join_fatal_threads()
        assert fatal_errors == []
        assert group.abort_calls == [1, 1]
        assert group.fatal_acks == [True, True]

        terminals = [state["runtime"]._terminal_error for state in states]

        def close_rank(rank):
            try:
                states[rank]["runtime"].close()
            except BaseException as error:
                return error

        close_results, close_errors = _run_two_rank_threads(
            tuple(lambda rank=rank: close_rank(rank) for rank in range(2))
        )
        assert close_errors == [None, None]
        assert close_results == terminals
        assert forbidden == []

        for rank, state in enumerate(states):
            runtime = state["runtime"]
            snapshot = expected[rank]
            owner = snapshot["owner"]
            if explicit_reap:
                assert owner.state == "detached"
                assert owner not in runtime._terminal_quarantine.owners
                assert state["working_set"]._status_workspace.borrower is None
            else:
                assert owner.state == "quarantined"
                assert owner in runtime._terminal_quarantine.owners
                assert tuple(owner.arrays) == snapshot["arrays"]
                assert all(
                    any(retained is status for retained in owner.arrays)
                    for status in snapshot["status_pair"]
                )
            retained = {
                record.identity: record
                for record in runtime._terminal_quarantine.allocations
            }
            assert set(retained) == set(snapshot["records"])
            assert all(
                retained[identity].capacity_bytes == record.capacity_bytes
                for identity, record in snapshot["records"].items()
            )
            for status_record in snapshot["status_records"]:
                assert (
                    sum(
                        record.identity == status_record.identity
                        for record in runtime._terminal_quarantine.allocations
                    )
                    == 1
                )
            resource_state = runtime.resource_state()
            assert resource_state["cache_bytes"] == sum(
                record.capacity_bytes for record in snapshot["cache_records"].values()
            )
            assert resource_state["pinned_bytes"] == sum(
                record.capacity_bytes for record in snapshot["pool_records"].values()
            )
            assert resource_state["stream_count"] == len(snapshot["streams"])
            assert resource_state["event_count"] == len(snapshot["events"])
            assert resource_state["quarantined_bytes"] == sum(
                record.capacity_bytes for record in snapshot["records"].values()
            )
    finally:
        _close_two_rank_cases(states)


@pytest.mark.parametrize("context_kind", ("generic", "direct"))
def test_terminal_outer_context_handoff_retains_lease_without_cleanup(
    monkeypatch, context_kind
):
    group, states = _two_rank_unopened_cases(
        store_id="two-rank-terminal-{}-context".format(context_kind)
    )
    forbidden = []
    snapshots = []
    try:
        _, open_errors = _open_two_rank_outer_context_cases(states, context_kind)
        assert open_errors == [None, None]
        group.start_active()
        _, first_errors = _run_two_rank_threads(
            tuple(
                lambda rank=rank: states[rank]["operator"](states[rank]["local_vector"])
                for rank in range(2)
            )
        )
        assert first_errors == [None, None]

        for rank, state in enumerate(states):
            runtime = state["runtime"]
            working_set = state["working_set"]
            provider = state["provider"]
            status_workspace = working_set._status_workspace
            prior_owner = status_workspace.borrower
            assert prior_owner is not None
            working_set.reap_completed()
            assert prior_owner.state == "detached"
            assert status_workspace.borrower is None

            adjacent_event = _ManualEvent(done=False)
            working_set.scheduler._event_factory = lambda event=adjacent_event: event
            output_ref = dict(working_set.request.host_refs)["output"]
            adjacent_destination = working_set.backend.empty(
                output_ref.shape, dtype=np.dtype(output_ref.dtype), order="C"
            )
            with working_set._lease_admission("prefetch"):
                with working_set.pool.checkout(adjacent_destination.nbytes) as slot:
                    adjacent_ticket = working_set.scheduler.stage_h2d(
                        output_ref, adjacent_destination, slot
                    )
            assert adjacent_ticket.completed is False

            expected_records = {}
            for record in (
                *working_set.allocation_records,
                *provider.cache.allocation_records,
                *working_set.pool.allocation_records,
                allocation_record(adjacent_destination),
            ):
                retained = expected_records.get(record.identity)
                if retained is None or record.capacity_bytes > retained.capacity_bytes:
                    expected_records[record.identity] = record

            def guarded_event_operation(
                operation, original, *, rank=rank, runtime=runtime
            ):
                def guarded(*args, **kwargs):
                    if runtime._terminal_error is not None:
                        forbidden.append((rank, "event.{}".format(operation)))
                    return original(*args, **kwargs)

                return guarded

            for operation in ("query", "synchronize"):
                monkeypatch.setattr(
                    adjacent_event,
                    operation,
                    guarded_event_operation(
                        operation, getattr(adjacent_event, operation)
                    ),
                )
            guarded_resources = (
                (
                    "cache",
                    provider.cache,
                    ("close", "wait_for_pending", "reap_completed"),
                ),
                (
                    "scheduler",
                    working_set.scheduler,
                    ("close", "complete_all", "reap_completed"),
                ),
                ("pool", working_set.pool, ("close", "reap_completed")),
            )
            for resource_name, resource, operations in guarded_resources:
                for operation in operations:
                    original = getattr(resource, operation)

                    def guarded_resource_operation(
                        *args,
                        rank=rank,
                        runtime=runtime,
                        resource_name=resource_name,
                        operation=operation,
                        original=original,
                        **kwargs,
                    ):
                        if runtime._terminal_error is not None:
                            forbidden.append(
                                (rank, "{}.{}".format(resource_name, operation))
                            )
                        return original(*args, **kwargs)

                    monkeypatch.setattr(resource, operation, guarded_resource_operation)
            snapshots.append(
                {
                    "lease": working_set,
                    "provider": provider,
                    "scheduler": working_set.scheduler,
                    "pool": working_set.pool,
                    "cache": provider.cache,
                    "status_workspace": status_workspace,
                    "status_records": status_workspace.allocation_records,
                    "adjacent_event": adjacent_event,
                    "adjacent_ticket": adjacent_ticket,
                    "expected_records": expected_records,
                }
            )

        primary = RuntimeError("injected peer fatal before outer context exit")
        states[0]["runtime"]._enter_communicator_fatal(primary)
        group.join_fatal_threads()
        terminals = [state["runtime"]._terminal_error for state in states]
        assert all(terminal is not None for terminal in terminals)

        exit_results, exit_errors = _run_two_rank_threads(
            tuple(
                lambda rank=rank: states[rank]["outer_context"].__exit__(
                    type(terminals[rank]), terminals[rank], None
                )
                for rank in range(2)
            )
        )
        assert exit_errors == [None, None]
        assert exit_results == [False, False]
        assert forbidden == []

        for rank, state in enumerate(states):
            runtime = state["runtime"]
            snapshot = snapshots[rank]
            lease = snapshot["lease"]
            status_workspace = snapshot["status_workspace"]
            assert snapshot["provider"].closed is False
            assert snapshot["provider"]._active_lease is lease
            assert lease._closed is False
            assert status_workspace._closed is False
            assert status_workspace.allocation_records == snapshot["status_records"]
            assert snapshot["adjacent_ticket"].completed is False
            assert snapshot["adjacent_event"].done is False
            assert all(
                retained is expected
                for retained, expected in zip(
                    lease.allocation_records, snapshot["status_records"]
                )
            )
            assert lease in snapshot["provider"]._terminal_resources
            assert snapshot["scheduler"] in snapshot["provider"]._terminal_resources
            assert snapshot["pool"] in snapshot["provider"]._terminal_resources
            assert snapshot["cache"] in snapshot["provider"]._terminal_resources
            retained = {
                record.identity: record
                for record in runtime._terminal_quarantine.allocations
            }
            assert set(retained) == set(snapshot["expected_records"])
            assert all(
                retained[identity].capacity_bytes == record.capacity_bytes
                for identity, record in snapshot["expected_records"].items()
            )

        def close_rank(rank):
            try:
                states[rank]["runtime"].close()
            except BaseException as error:
                return error

        close_results, close_errors = _run_two_rank_threads(
            tuple(lambda rank=rank: close_rank(rank) for rank in range(2))
        )
        assert close_errors == [None, None]
        assert close_results == terminals
    finally:
        _close_two_rank_cases(states)


@pytest.mark.skipif(
    not _has_one_visible_gpu(),
    reason="requires exactly one visible CUDA device",
)
def test_profiled_cupy_terminal_context_exit_avoids_peak_query(monkeypatch):
    from renormalizer.utils.log import DEBUG, PROFILING, init_log

    runtime = _runtime(backend_name="cupy")
    request, plan, store, _, _ = _active_case(
        runtime, store_id="profiled-cupy-terminal-context"
    )
    provider = _provider(runtime, request)
    manager = active_working_set_execution(
        _active_execution_config(runtime, request, provider),
        request,
        plan,
        store,
    )
    init_log(PROFILING)
    try:
        borrowed = manager.__enter__()
        working_set = borrowed.provider
        peak_queries = []

        def forbidden_peak_query():
            peak_queries.append("memGetInfo")
            raise AssertionError("terminal context queried CUDA memory")

        monkeypatch.setattr(
            runtime.backend._cupy.cuda.runtime,
            "memGetInfo",
            forbidden_peak_query,
        )
        primary = RuntimeError("injected profiled communicator fatal")
        assert runtime._enter_communicator_fatal(primary) is primary
        assert manager.__exit__(type(primary), primary, None) is False

        assert peak_queries == []
        assert working_set._closed is False
        assert working_set._status_workspace._closed is False
        assert working_set in provider._terminal_resources
    finally:
        init_log(DEBUG)
        if not runtime._closed:
            try:
                runtime.close()
            except BaseException:
                pass
        if not store.closed:
            if runtime._terminal_error is None:
                store.close()
            else:
                assert store._reservations


def test_two_rank_broadcast_enqueue_then_raise_aborts_before_execution_or_e():
    group, states = _two_rank_unopened_cases(store_id="two-rank-broadcast-fatal")
    injected = RuntimeError("injected post-enqueue broadcast failure")
    try:
        _, open_errors = _open_two_rank_cases(states)
        assert open_errors == [None, None]
        group.broadcast_failures[("B(0)", 0)] = injected
        group.start_active()
        _, call_errors = _run_two_rank_threads(
            tuple(
                lambda rank=rank: states[rank]["operator"](states[rank]["local_vector"])
                for rank in range(2)
            )
        )

        assert call_errors[0] is injected
        assert group.abort_calls == [1, 1]
        assert group.fatal_acks == [True, True]
        assert group.active_broadcast_records == {
            0: (1, True),
            1: (1, False),
        }
        assert group.active_broadcast_consumed == [1, 1]
        for rank, state in enumerate(states):
            trace = [entry[0] for entry in group.endpoints[rank].trace]
            assert trace == ["P0", "P1", "P2", "B(0)"]
            assert state["operator"]._counters.get("broadcast_calls", 0) == 0
            assert state["operator"]._counters.get("execution_calls", 0) == 0
            local_primary = state["runtime"]._terminal_error
            assert local_primary is group.endpoints[rank]._fatal_error
            assert state["provider"]._terminal_error is local_primary
            assert state["working_set"]._poisoned_error is local_primary
            assert state["runtime"].backend._execution_terminal_error is local_primary

        group.start_active()
        _, retry_errors = _run_two_rank_threads(
            tuple(
                lambda rank=rank: states[rank]["operator"](states[rank]["local_vector"])
                for rank in range(2)
            )
        )
        assert all(isinstance(error, RuntimeError) for error in retry_errors)
        assert all(endpoint.trace == [] for endpoint in group.endpoints)
        assert group.abort_calls == [1, 1]
    finally:
        _close_two_rank_cases(states)


def test_two_rank_malformed_vector_stops_both_actors_through_p0():
    group, states = _two_rank_unopened_cases(store_id="two-rank-malformed-vector")
    try:
        _, open_errors = _open_two_rank_cases(states)
        assert open_errors == [None, None]
        states[0]["local_vector"] = object()
        group.start_active()

        _, call_errors = _run_two_rank_threads(
            tuple(
                lambda rank=rank: states[rank]["operator"](states[rank]["local_vector"])
                for rank in range(2)
            )
        )

        traces = [
            [entry[0] for entry in endpoint.trace] for endpoint in group.endpoints
        ]
        assert all(
            isinstance(error, ValueError)
            and str(error) == "distributed call-ready preflight failed"
            for error in call_errors
        ), (call_errors, traces)
        for rank, endpoint in enumerate(group.endpoints):
            assert traces[rank] == ["P0"]
            workspace = states[rank]["working_set"]._status_workspace
            assert endpoint.status_arrays == [workspace.device_status]
    finally:
        _close_two_rank_cases(states)


@pytest.mark.parametrize("failure_slot", ("P0", "P1"))
def test_two_rank_status_enqueue_failure_retains_published_owner_resources(
    failure_slot,
):
    group, states = _two_rank_unopened_cases(
        store_id="two-rank-enqueue-{}".format(failure_slot.lower())
    )
    status_errors = [
        RuntimeError("injected rank {} status enqueue failure".format(rank))
        for rank in range(2)
    ]
    drain_errors = [
        RuntimeError("injected rank {} status drain failure".format(rank))
        for rank in range(2)
    ]
    captured = [None, None]
    try:
        _, open_errors = _open_two_rank_cases(states)
        assert open_errors == [None, None]
        group.start_active()
        group.status_failures[failure_slot] = status_errors

        def inspect_owner(rank, label, status):
            state = states[rank]
            working_set = state["working_set"]
            operator = state["operator"]
            workspace = working_set._status_workspace
            owner = working_set._active_operator_owner
            assert label == failure_slot
            assert owner is not None
            assert status is workspace.device_status
            for array in (
                workspace.device_status,
                workspace.host_status,
            ):
                assert any(retained is array for retained in owner.arrays)
            if label == "P0":
                assert any(
                    retained is state["local_vector"] for retained in owner.resources
                )
                assert all(
                    retained is not state["local_vector"] for retained in owner.arrays
                )
            else:
                assert any(
                    retained is state["local_vector"] for retained in owner.arrays
                )
            if label == "P1":
                assert any(
                    retained is operator._receive_storage for retained in owner.arrays
                )
                assert any(
                    retained is operator._output_accumulator
                    for retained in owner.arrays
                )

            def fail_drain():
                raise drain_errors[rank]

            owner._drainer = fail_drain
            captured[rank] = (
                owner,
                operator._receive_storage,
                operator._output_accumulator,
            )

        group.status_hook = inspect_owner
        _, call_errors = _run_two_rank_threads(
            tuple(
                lambda rank=rank: states[rank]["operator"](states[rank]["local_vector"])
                for rank in range(2)
            )
        )
        assert all(call_errors[rank] is status_errors[rank] for rank in range(2))
        expected_trace = ["P0"] if failure_slot == "P0" else ["P0", "P1"]
        for rank, state in enumerate(states):
            assert [entry[0] for entry in group.endpoints[rank].trace] == expected_trace
            owner, receive_storage, output_accumulator = captured[rank]
            quarantine = state["runtime"]._terminal_quarantine
            workspace = state["working_set"]._status_workspace
            assert owner.state == "quarantined"
            assert owner.error is status_errors[rank]
            assert owner in quarantine.owners
            assert state["provider"]._terminal_error is status_errors[rank]
            assert state["runtime"]._terminal_error is status_errors[rank]
            assert quarantine.first_error is status_errors[rank]
            assert any(retained is workspace.device_status for retained in owner.arrays)
            assert any(retained is workspace.host_status for retained in owner.arrays)
            if failure_slot == "P1":
                assert any(retained is receive_storage for retained in owner.arrays)
                assert any(retained is output_accumulator for retained in owner.arrays)
    finally:
        _close_two_rank_cases(states)


def test_two_rank_status_failure_with_successful_drain_terminally_rejects_reuse():
    group, states = _two_rank_unopened_cases(
        store_id="two-rank-successful-drain-terminal"
    )
    status_errors = [
        RuntimeError("injected rank {} terminal status failure".format(rank))
        for rank in range(2)
    ]
    captured_owners = [None, None]
    try:
        _, open_errors = _open_two_rank_cases(states)
        assert open_errors == [None, None]
        group.start_active()
        group.status_failures["P0"] = status_errors

        def capture_owner(rank, label, status):
            assert label == "P0"
            captured_owners[rank] = states[rank]["working_set"]._active_operator_owner

        group.status_hook = capture_owner
        _, call_errors = _run_two_rank_threads(
            tuple(
                lambda rank=rank: states[rank]["operator"](states[rank]["local_vector"])
                for rank in range(2)
            )
        )

        assert all(call_errors[rank] is status_errors[rank] for rank in range(2))
        for rank, state in enumerate(states):
            owner = captured_owners[rank]
            assert owner.state == "quarantined"
            assert owner.arrays
            assert owner.resources
            assert state["working_set"]._status_workspace.borrower is owner
            assert state["working_set"]._poisoned_error is status_errors[rank]
            assert state["provider"]._terminal_error is status_errors[rank]
            assert state["runtime"]._terminal_error is status_errors[rank]
            assert state["runtime"]._terminal_quarantine.owners == (owner,)
            assert [entry[0] for entry in group.endpoints[rank].trace] == ["P0"]
        assert group.abort_calls == [1, 1]

        group.status_failures.clear()
        group.status_hook = None
        group.start_active()
        _, retry_errors = _run_two_rank_threads(
            tuple(
                lambda rank=rank: states[rank]["operator"](states[rank]["local_vector"])
                for rank in range(2)
            )
        )
        for rank, state in enumerate(states):
            assert isinstance(retry_errors[rank], RuntimeError)
            assert str(retry_errors[rank]) == "working-set lease is poisoned"
            assert retry_errors[rank].__cause__ is status_errors[rank]

            with pytest.raises(RuntimeError, match="terminal-poisoned") as rejected:
                state["provider"].open_working_set(None, None, None, None)
            assert rejected.value.__cause__ is status_errors[rank]
            with pytest.raises(RuntimeError, match="terminal-poisoned") as rejected:
                state["runtime"].barrier()
            assert rejected.value.__cause__ is status_errors[rank]
            with pytest.raises(RuntimeError, match="terminal-poisoned") as rejected:
                state["runtime"].execution_config()
            assert rejected.value.__cause__ is status_errors[rank]
            assert group.endpoints[rank].trace == []
    finally:
        _close_two_rank_cases(states)


def test_two_rank_first_status_error_identity_survives_cleanup_and_drain(
    monkeypatch,
):
    group, states = _two_rank_unopened_cases(store_id="two-rank-error-identity")
    status_errors = [
        RuntimeError("injected rank {} acquisition status failure".format(rank))
        for rank in range(2)
    ]
    cleanup_error = RuntimeError("injected child cleanup failure")
    drain_error = RuntimeError("injected owner drain failure")
    misleading_cause = RuntimeError("must not become the primary error")
    status_errors[0].__cause__ = misleading_cause
    captured_calls = [[], []]
    try:
        _, open_errors = _open_two_rank_cases(states)
        assert open_errors == [None, None]
        for rank, state in enumerate(states):
            working_set = state["working_set"]
            original_operator_call = working_set._operator_call

            @contextmanager
            def capture_call(local_vector, rank=rank, original=original_operator_call):
                with original(local_vector) as call:
                    captured_calls[rank].append(call)
                    yield call

            monkeypatch.setattr(working_set, "_operator_call", capture_call)

        working_set = states[0]["working_set"]
        original_acquire = working_set.acquire
        acquisition_count = 0

        class CleanupFailureContext:
            def __init__(self, child):
                self.child = child

            def __enter__(self):
                return self.child.__enter__()

            def __exit__(self, exc_type, exc_value, traceback):
                self.child.__exit__(exc_type, exc_value, traceback)
                raise cleanup_error

        def acquire_with_cleanup_failure(request):
            nonlocal acquisition_count
            child = original_acquire(request)
            acquisition_count += 1
            if acquisition_count == 1:
                return CleanupFailureContext(child)
            return child

        monkeypatch.setattr(working_set, "acquire", acquire_with_cleanup_failure)
        group.start_active()
        group.status_failures["P2"] = status_errors

        def fail_rank_zero_drain(rank, label, status):
            if rank != 0:
                return
            owner = states[rank]["working_set"]._active_operator_owner

            def fail_drain():
                raise drain_error

            owner._drainer = fail_drain

        group.status_hook = fail_rank_zero_drain
        _, call_errors = _run_two_rank_threads(
            tuple(
                lambda rank=rank: states[rank]["operator"](states[rank]["local_vector"])
                for rank in range(2)
            )
        )

        status_error = status_errors[0]
        call = captured_calls[0][0]
        owner = call.owner
        provider = states[0]["provider"]
        runtime = states[0]["runtime"]
        quarantine = runtime._terminal_quarantine
        assert call_errors[0] is status_error
        assert call.primary_error is status_error
        assert owner.error is status_error
        assert provider._terminal_error is status_error
        assert runtime._terminal_error is status_error
        assert quarantine.first_error is status_error
        assert owner.secondary_errors == (cleanup_error,)
        assert owner.state == "quarantined"
        assert drain_error not in owner.secondary_errors
        assert misleading_cause not in owner.secondary_errors
        assert [entry[0] for entry in group.endpoints[0].trace] == [
            "P0",
            "P1",
            "P2",
        ]
    finally:
        _close_two_rank_cases(states)


@pytest.mark.parametrize("drain_fails", (False, True))
def test_two_rank_mixed_local_and_status_failure_preserves_first_error(
    monkeypatch, drain_fails
):
    group, states = _two_rank_unopened_cases(
        store_id="two-rank-mixed-status-{}-drain".format(
            "failed" if drain_fails else "successful"
        )
    )
    local_error = RuntimeError("injected rank 0 local acquisition failure")
    status_errors = [
        RuntimeError("injected rank {} later status failure".format(rank))
        for rank in range(2)
    ]
    drain_error = RuntimeError("injected rank 0 later drain failure")
    captured_calls = [[], []]
    try:
        _, open_errors = _open_two_rank_cases(states)
        assert open_errors == [None, None]
        for rank, state in enumerate(states):
            working_set = state["working_set"]
            original_operator_call = working_set._operator_call

            @contextmanager
            def capture_call(local_vector, rank=rank, original=original_operator_call):
                with original(local_vector) as call:
                    captured_calls[rank].append(call)
                    yield call

            monkeypatch.setattr(working_set, "_operator_call", capture_call)

        def fail_local_acquisition(request):
            raise local_error

        monkeypatch.setattr(states[0]["working_set"], "acquire", fail_local_acquisition)
        group.start_active()
        group.status_failures["P2"] = status_errors

        def configure_rank_zero_drain(rank, label, status):
            if rank == 0 and drain_fails:
                owner = states[rank]["working_set"]._active_operator_owner

                def fail_drain():
                    raise drain_error

                owner._drainer = fail_drain

        group.status_hook = configure_rank_zero_drain
        _, call_errors = _run_two_rank_threads(
            tuple(
                lambda rank=rank: states[rank]["operator"](states[rank]["local_vector"])
                for rank in range(2)
            )
        )

        first_errors = (local_error, status_errors[1])
        for rank, state in enumerate(states):
            first_error = first_errors[rank]
            call = captured_calls[rank][0]
            owner = call.owner
            quarantine = state["runtime"]._terminal_quarantine
            assert call_errors[rank] is first_error
            assert call.primary_error is first_error
            assert owner.error is first_error
            assert state["working_set"]._poisoned_error is first_error
            assert state["provider"]._terminal_error is first_error
            assert state["runtime"]._terminal_error is first_error
            assert call.secondary_errors == ((status_errors[0],) if rank == 0 else ())
            assert [entry[0] for entry in group.endpoints[rank].trace] == [
                "P0",
                "P1",
                "P2",
            ]
            assert owner.state == "quarantined"
            assert owner.secondary_errors == ((status_errors[0],) if rank == 0 else ())
            assert drain_error not in owner.secondary_errors
            assert quarantine.owners == (owner,)
            assert quarantine.first_error is first_error
        assert group.abort_calls == [1, 1]
    finally:
        _close_two_rank_cases(states)


@pytest.mark.parametrize("drain_fails", (False, True))
def test_two_rank_acquisition_cleanup_failure_rethrows_call_primary(
    monkeypatch, drain_fails
):
    group, states = _two_rank_unopened_cases(
        store_id="two-rank-acquire-cleanup-{}-drain".format(
            "failed" if drain_fails else "successful"
        )
    )
    acquisition_error = RuntimeError("injected rank 0 acquisition failure")
    cleanup_error = RuntimeError("injected rank 0 child cleanup failure")
    drain_error = RuntimeError("injected rank 0 owner drain failure")
    captured_calls = [[], []]
    try:
        _, open_errors = _open_two_rank_cases(states)
        assert open_errors == [None, None]
        for rank, state in enumerate(states):
            working_set = state["working_set"]
            original_operator_call = working_set._operator_call

            @contextmanager
            def capture_call(local_vector, rank=rank, original=original_operator_call):
                with original(local_vector) as call:
                    captured_calls[rank].append(call)
                    yield call

            monkeypatch.setattr(working_set, "_operator_call", capture_call)

        working_set = states[0]["working_set"]
        original_acquire = working_set.acquire
        acquisition_count = 0

        class CleanupFailureContext:
            def __init__(self, child):
                self.child = child

            def __enter__(self):
                return self.child.__enter__()

            def __exit__(self, exc_type, exc_value, traceback):
                self.child.__exit__(exc_type, exc_value, traceback)
                raise cleanup_error

        def acquire_then_fail(request):
            nonlocal acquisition_count
            acquisition_count += 1
            if acquisition_count == 2:
                if drain_fails:
                    owner = working_set._active_operator_owner
                    owner._drainer = lambda: (_ for _ in ()).throw(drain_error)
                raise acquisition_error
            return CleanupFailureContext(original_acquire(request))

        monkeypatch.setattr(working_set, "acquire", acquire_then_fail)
        group.start_active()
        _, call_errors = _run_two_rank_threads(
            tuple(
                lambda rank=rank: states[rank]["operator"](states[rank]["local_vector"])
                for rank in range(2)
            )
        )

        call = captured_calls[0][0]
        owner = call.owner
        quarantine = states[0]["runtime"]._terminal_quarantine
        assert call_errors[0] is acquisition_error
        if isinstance(call_errors[1], ValueError):
            assert str(call_errors[1]) == "distributed resource preflight failed"
        else:
            assert drain_fails is True
            assert call_errors[1] is group.endpoints[1]._fatal_error
        assert call.primary_error is acquisition_error
        assert call.secondary_errors == (cleanup_error,)
        assert owner.error is acquisition_error
        assert [entry[0] for entry in group.endpoints[0].trace] == ["P0", "P1", "P2"]
        assert [entry[0] for entry in group.endpoints[1].trace] == ["P0", "P1", "P2"]
        if drain_fails:
            assert owner.state == "quarantined"
            assert owner.secondary_errors == (cleanup_error, drain_error)
            assert quarantine.owners == (owner,)
            assert quarantine.first_error is acquisition_error
        else:
            assert owner.state == "detached"
            assert owner.secondary_errors == (cleanup_error,)
            assert quarantine.owners == ()
    finally:
        _close_two_rank_cases(states)


def test_partial_acquisition_transfers_first_cache_lease_to_call_owner(monkeypatch):
    runtime, store, _, working_set, operator, local_vector = _active_operator_case(
        2, store_id="operator-partial-acquire"
    )
    original_load = working_set._load_identity
    loaded = []
    drained_resources = []
    error = RuntimeError("injected second operand acquisition failure")

    def fail_second(identity, *, prefetch=False, _admission_token=None):
        if not prefetch and loaded:
            owner = working_set._active_operator_owner
            first = loaded[0]
            assert first._owner is owner
            assert any(resource is first for resource in owner.resources)
            owner._drainer = lambda: drained_resources.append(tuple(owner.resources))
            raise error
        lease = original_load(
            identity,
            prefetch=prefetch,
            _admission_token=_admission_token,
        )
        if not prefetch:
            loaded.append(lease)
        return lease

    monkeypatch.setattr(working_set, "_load_identity", fail_second)
    try:
        with pytest.raises(ValueError, match="distributed resource preflight failed"):
            operator(local_vector)

        assert len(loaded) == 1
        assert len(drained_resources) == 1
        assert any(resource is loaded[0] for resource in drained_resources[0])
        assert loaded[0].state == "closed"
        assert working_set._active_operator_owner is None
    finally:
        if not working_set._closed:
            working_set.close()
        store.close()
        runtime.close()


def test_operator_call_rejects_nested_active_calls():
    runtime, store, _, working_set, _, local_vector = _active_operator_case(
        store_id="operator-nested-call"
    )
    try:
        with working_set._operator_call(local_vector):
            with pytest.raises(RuntimeError, match="already active"):
                with working_set._operator_call(local_vector):
                    pass
    finally:
        if not working_set._closed:
            working_set.close()
        store.close()
        runtime.close()


def test_active_preflight_failure_allocates_no_resources():
    class RequirementDisagreementCollective:
        rank = 0
        size = 2

        def allreduce(self, value, *, op="sum"):
            result = np.array(value, copy=True)
            if result.dtype == np.int64 and result.size == 7 and op == "max":
                result[0] += 1
            return result

        def close(self):
            pass

    runtime = _runtime(world_size=2, collective=RequirementDisagreementCollective())
    request, plan, store, _, _ = _active_case(runtime)
    allocations = []
    provider = _provider(
        runtime,
        request,
        cache_factory=lambda *args, **kwargs: allocations.append("cache"),
        pool_factory=lambda *args, **kwargs: allocations.append("pool"),
        scheduler_factory=lambda *args, **kwargs: allocations.append("scheduler"),
    )
    runtime._active_provider = provider

    with pytest.raises(ValueError, match="requirement disagreement"):
        runtime.preflight_residency(request, plan)

    assert allocations == []
    assert provider.resource_state()["active_leases"] == 0
    store.close()
    runtime.close()


def test_open_rejects_unrelated_store_mutation_before_resource_allocation():
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(runtime, store_id="snapshot-open")
    receipt = runtime.preflight_residency(request, plan)
    allocations = []
    provider = _provider(
        runtime,
        request,
        cache_factory=lambda *args, **kwargs: allocations.append("cache"),
    )
    store.put("unrelated", np.ones(1, dtype=np.float64))

    with pytest.raises(HostTensorError, match="complete snapshot"):
        provider.open_working_set(request, plan, store, receipt)

    assert allocations == []
    assert provider.resource_state()["active_leases"] == 0
    store.close()
    runtime.close()


def test_single_rank_preflight_issues_receipt_without_control_array_allocation(
    monkeypatch,
):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(runtime, store_id="single-preflight")

    def fail_control_array(*args, **kwargs):
        raise AssertionError("single-rank preflight must not allocate controls")

    monkeypatch.setattr(runtime, "_control_array", fail_control_array)
    receipt = runtime.preflight_residency(request, plan)

    assert receipt.request_hash == request.request_hash
    assert receipt.plan_hash == plan.plan_hash
    store.close()
    runtime.close()


def test_outer_lease_reuses_static_slice_across_repeated_child_leases():
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(runtime)
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)

    with provider.open_working_set(request, plan, store, receipt) as working_set:
        with working_set.acquire(_block_request(working_set)) as first:
            first_ptr = first.bindings.arrays["input_0"].__array_interface__["data"][0]
        with working_set.acquire(_block_request(working_set)) as second:
            second_ptr = second.bindings.arrays["input_0"].__array_interface__["data"][
                0
            ]
        assert second_ptr == first_ptr

    assert provider.metrics.h2d_count == 1
    state = provider.resource_state()
    assert state["reserved_cache_bytes"] == 0
    assert state["checked_out_pinned_bytes"] == 0
    store.close()
    runtime.close()


def test_sequential_outer_lease_epochs_reuse_retained_cache_entry():
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime, store_id="sequential-outer-epochs"
    )
    provider = _provider(runtime, request)

    first_receipt = runtime.preflight_residency(request, plan)
    with provider.open_working_set(
        request, plan, store, first_receipt
    ) as first_working_set:
        first_epoch = first_working_set._epoch
        with first_working_set.acquire(_block_request(first_working_set)) as first:
            first_ptr = first.bindings.arrays["input_0"].__array_interface__["data"][0]

    second_receipt = runtime.preflight_residency(request, plan)
    with provider.open_working_set(
        request, plan, store, second_receipt
    ) as second_working_set:
        second_epoch = second_working_set._epoch
        with second_working_set.acquire(_block_request(second_working_set)) as second:
            second_ptr = second.bindings.arrays["input_0"].__array_interface__["data"][0]

    assert second_epoch > first_epoch
    assert second_ptr == first_ptr
    assert provider.metrics.cache_hits == 1
    assert provider.metrics.cache_misses == 0
    assert provider.metrics.h2d_count == 0
    assert runtime._terminal_gate.phase is _TerminalPhase.HEALTHY
    store.close()
    runtime.close()


def test_child_close_waits_for_compute_event_before_cache_reuse():
    compute_event = _ManualEvent()
    events = iter((_ManualEvent(done=True), compute_event))
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(runtime)
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request, event_factory=lambda: next(events))
    working_set = provider.open_working_set(request, plan, store, receipt).__enter__()
    child = working_set.acquire(_block_request(working_set)).__enter__()
    identity = next(
        identity
        for identity in working_set.cache_identities
        if identity[1] == "input_0"
    )

    child.close()
    assert provider.last_compute_event is None
    assert provider.cache.refcount(identity) == 1
    working_set.reap_completed()
    assert provider.cache.refcount(identity) == 1
    compute_event.complete()
    working_set.reap_completed()
    assert provider.cache.refcount(identity) == 0

    working_set.close()
    store.close()
    runtime.close()


def test_terminal_compute_drain_failure_retains_outer_resource_ownership(monkeypatch):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime, store_id="terminal-compute-drain"
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    runtime._active_provider = provider
    working_set = provider.open_working_set(request, plan, store, receipt).__enter__()
    child = working_set.acquire(_block_request(working_set)).__enter__()
    identity = next(
        identity
        for identity in working_set.cache_identities
        if identity[1] == "input_0"
    )
    scheduler = working_set.scheduler
    cache = provider.cache
    event = _SchedulerQueryFailureEvent(synchronize_fails=True)
    fake_backend = _FakeCupyBackend(synchronize_fails=True)
    scheduler.backend = fake_backend
    scheduler._cupy = fake_backend._cupy
    scheduler._stream = fake_backend._cupy.cuda.stream
    scheduler._event_factory = lambda: event

    child.close()
    with pytest.raises(RuntimeError, match="scheduler query failure"):
        working_set.reap_completed()

    assert child.bindings is None
    assert provider.cache.refcount(identity) == 1
    assert provider.cache.poisoned is True
    assert scheduler.poisoned is True
    with pytest.raises(RuntimeError, match="scheduler query failure"):
        working_set.close()

    assert working_set._closed is False
    assert str(working_set._poisoned_error) == "injected scheduler query failure"
    assert working_set._provider is provider
    assert working_set.scheduler is scheduler
    assert working_set.pool is not None
    assert working_set._status_workspace._closed is False
    assert provider._active_lease is working_set
    assert working_set in provider._terminal_resources
    assert scheduler in provider._terminal_resources
    with pytest.raises(RuntimeError, match="scheduler query failure"):
        provider.close()
    assert provider.closed is False
    assert provider.runtime is runtime
    assert provider._cache is cache
    assert provider._active_lease is working_set
    assert working_set._store_reservation is not None
    assert cache in provider._terminal_resources
    with pytest.raises(RuntimeError, match="scheduler query failure"):
        runtime.close()


def test_prefetch_inserts_compute_wait_only_when_loading_entry_is_consumed(
    monkeypatch,
):
    events = []

    def event_factory():
        event = _ManualEvent()
        events.append(event)
        return event

    runtime = _runtime()
    request, plan, store, _, _ = _active_case(runtime, store_id="prefetch-wait")
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request, event_factory=event_factory)
    working_set = provider.open_working_set(request, plan, store, receipt).__enter__()
    identity = next(
        value for value in working_set.cache_identities if value[1] == "input_0"
    )
    waits = []
    monkeypatch.setattr(
        working_set.scheduler,
        "wait_for_h2d",
        lambda event, **_kwargs: waits.append(event),
        raising=False,
    )

    working_set._load_identity(identity, prefetch=True)
    assert waits == []
    with working_set.acquire(_block_request(working_set)):
        assert waits == [events[0]]

    for event in events:
        event.complete()
    working_set.close()
    store.close()
    runtime.close()


def test_h2d_failure_after_enqueue_synchronizes_stream_and_drops_event():
    store = HostTensorStore(store_id="enqueue-failure")
    ref = store.put("input", np.arange(4, dtype=np.float64))
    backend = _FakeCupyBackend()
    pool = PinnedBufferPool(32)
    scheduler = TransferScheduler(
        store,
        backend,
        pool,
        event_factory=_RecordFailureEvent,
    )
    destination = _FakeDestination()
    try:
        with pool.checkout(32) as slot:
            with pytest.raises(RuntimeError, match="post-enqueue"):
                scheduler.stage_h2d(ref, destination, slot)

        assert destination.enqueued is True
        assert backend._cupy.cuda.stream.synchronize_calls == 1
        assert scheduler.event_count == 0
        assert scheduler.h2d_bytes == 0
    finally:
        scheduler.close()
        pool.close()
        store.close()


def test_d2h_transfer_event_record_failure_never_commits_or_accounts():
    store = HostTensorStore(store_id="d2h-record-failure")
    destination_ref = store.put("output", np.zeros(4, dtype=np.float64))
    backend = _FakeCupyBackend()
    pool = PinnedBufferPool(32)
    events = iter((_ManualEvent(), _RecordFailureEvent()))
    scheduler = TransferScheduler(
        store,
        backend,
        pool,
        event_factory=lambda: next(events),
    )
    source = _FakeSource()
    try:
        with pool.checkout(32) as slot:
            with pytest.raises(RuntimeError, match="post-enqueue event failure"):
                scheduler.writeback_d2h(source, destination_ref, slot)

        assert source.enqueued is True
        assert backend._cupy.cuda.stream.synchronize_calls == 1
        assert store.ref("output") == destination_ref
        assert scheduler.d2h_bytes == 0
    finally:
        scheduler.close()
        pool.close()
        store.close()


def test_h2d_owner_is_registered_before_event_creation_failure():
    store = HostTensorStore(store_id="h2d-event-creation")
    ref = store.put("input", np.arange(4, dtype=np.float64))
    backend = _FakeCupyBackend()
    pool = PinnedBufferPool(32)
    scheduler = None
    destination = _FakeDestination()
    observed = {}

    def fail_event_creation():
        owner = next(iter(scheduler._owners.values()))
        observed["kind"] = owner.kind
        observed["destination"] = any(array is destination for array in owner.arrays)
        observed["stream"] = backend._cupy.cuda.stream in owner.streams
        raise RuntimeError("injected event creation failure")

    scheduler = TransferScheduler(
        store,
        backend,
        pool,
        event_factory=fail_event_creation,
    )
    try:
        with pool.checkout(32) as slot:
            with pytest.raises(RuntimeError, match="event creation failure"):
                scheduler.stage_h2d(ref, destination, slot)

        assert observed == {"kind": "h2d", "destination": True, "stream": True}
        assert scheduler._owners == {}
        assert scheduler._event_owners == {}
        assert scheduler.h2d_bytes == 0
        assert pool.pending_bytes == 0
    finally:
        scheduler.close()
        pool.close()
        store.close()


def test_d2h_event_creation_failure_drains_compute_without_committing():
    store = HostTensorStore(store_id="d2h-event-creation")
    destination_ref = store.put("output", np.zeros(4, dtype=np.float64))
    backend = _FakeCupyBackend()
    pool = PinnedBufferPool(32)
    scheduler = None
    observed = {}

    def fail_event_creation():
        owner = next(iter(scheduler._owners.values()))
        observed["kind"] = owner.kind
        observed["source_count"] = len(owner.arrays)
        observed["stream"] = backend._cupy.cuda.stream in owner.streams
        raise RuntimeError("injected D2H event creation failure")

    scheduler = TransferScheduler(
        store,
        backend,
        pool,
        event_factory=fail_event_creation,
    )
    source = _FakeSource()
    try:
        with pool.checkout(32) as slot:
            with pytest.raises(RuntimeError, match="D2H event creation failure"):
                scheduler.writeback_d2h(source, destination_ref, slot)

        assert observed == {"kind": "d2h", "source_count": 2, "stream": True}
        assert backend._cupy.cuda.stream.synchronize_calls == 1
        assert source.enqueued is False
        assert store.ref("output") == destination_ref
        assert scheduler.d2h_bytes == 0
        assert scheduler._owners == {}
        assert scheduler._event_owners == {}
        assert pool.pending_bytes == 0
    finally:
        scheduler.close()
        pool.close()
        store.close()


def test_first_h2d_query_failure_runs_through_owner_before_slot_release():
    store = HostTensorStore(store_id="h2d-first-query")
    ref = store.put("input", np.arange(4, dtype=np.float64))
    backend = _FakeCupyBackend()
    pool = PinnedBufferPool(32)
    event = _SchedulerQueryFailureEvent()
    scheduler = TransferScheduler(
        store,
        backend,
        pool,
        event_factory=lambda: event,
    )

    with pool.checkout(32) as slot:
        ticket = scheduler.stage_h2d(ref, _FakeDestination(), slot)
    assert pool.pending_bytes == 32

    with pytest.raises(RuntimeError, match="scheduler query failure"):
        scheduler.reap_completed()

    assert ticket.completed is True
    assert backend._cupy.cuda.stream.synchronize_calls == 1
    assert pool.pending_bytes == 0
    assert scheduler._owners == {}
    assert scheduler._event_owners == {}
    scheduler.close()
    pool.close()
    store.close()


def test_wait_on_consumption_drain_failure_quarantines_h2d_owner():
    store = HostTensorStore(store_id="h2d-wait-owner-quarantine")
    ref = store.put("input", np.arange(4, dtype=np.float64))
    backend = _FakeCupyBackend(synchronize_fails=True)
    pool = PinnedBufferPool(32)
    event = _ManualEvent()
    quarantine = RuntimeTerminalQuarantine()
    scheduler = TransferScheduler(
        store,
        backend,
        pool,
        event_factory=lambda: event,
        quarantine=quarantine.retain,
    )
    with pool.checkout(32) as slot:
        ticket = scheduler.stage_h2d(ref, _FakeDestination(), slot)

    def fail_wait(retained):
        assert retained is event
        raise RuntimeError("injected wait enqueue failure")

    backend._cupy.cuda.stream.wait_event = fail_wait
    with pytest.raises(RuntimeError, match="wait enqueue failure"):
        scheduler.wait_for_h2d(event)

    assert ticket.terminal_poisoned is True
    assert scheduler.poisoned is True
    assert quarantine.resource_state()["quarantined_owner_count"] == 1
    assert quarantine.resource_state()["quarantined_stream_count"] == 1
    with pytest.raises(RuntimeError, match="wait enqueue failure"):
        scheduler.close()
    with pytest.raises(RuntimeError, match="wait enqueue failure"):
        pool.close()
    store.close()


def test_h2d_post_enqueue_drain_failure_quarantines_complete_owner():
    store = HostTensorStore(store_id="h2d-owner-quarantine")
    ref = store.put("input", np.arange(4, dtype=np.float64))
    backend = _FakeCupyBackend(synchronize_fails=True)
    pool = PinnedBufferPool(32)
    quarantine = RuntimeTerminalQuarantine()
    scheduler = TransferScheduler(
        store,
        backend,
        pool,
        event_factory=_RecordFailureEvent,
        quarantine=quarantine.retain,
    )
    destination = _FakeDestination()
    destination_ref = weakref.ref(destination)
    staging_ref = weakref.ref(pool._slot._array)

    with pool.checkout(32) as slot:
        with pytest.raises(RuntimeError, match="post-enqueue event failure"):
            scheduler.stage_h2d(ref, destination, slot)
    del destination
    gc.collect()

    assert quarantine.first_error.args == ("injected post-enqueue event failure",)
    assert quarantine.resource_state() == {
        "quarantined_owner_count": 1,
        "quarantined_array_count": 2,
        "quarantined_bytes": 64,
        "quarantined_event_count": 1,
        "quarantined_stream_count": 1,
    }
    assert destination_ref() is not None
    assert staging_ref() is not None
    assert scheduler.poisoned is True
    assert pool.poisoned is True
    with pytest.raises(RuntimeError, match="post-enqueue event failure"):
        scheduler.close()
    with pytest.raises(RuntimeError, match="post-enqueue event failure"):
        pool.close()
    store.close()


def test_d2h_post_enqueue_drain_failure_quarantines_source_staging_and_callback():
    store = HostTensorStore(store_id="d2h-owner-quarantine")
    destination_ref = store.put("output", np.zeros(4, dtype=np.float64))
    backend = _FakeCupyBackend(synchronize_fails=True)
    pool = PinnedBufferPool(32)
    quarantine = RuntimeTerminalQuarantine()
    events = iter((_ManualEvent(), _RecordFailureEvent()))
    scheduler = TransferScheduler(
        store,
        backend,
        pool,
        event_factory=lambda: next(events),
        quarantine=quarantine.retain,
    )
    source = _FakeSource()
    source_ref = weakref.ref(source)
    staging_ref = weakref.ref(pool._slot._array)

    with pool.checkout(32) as slot:
        with pytest.raises(RuntimeError, match="post-enqueue event failure"):
            scheduler.writeback_d2h(source, destination_ref, slot)
    assert source.enqueued is True
    del source
    gc.collect()

    assert quarantine.resource_state() == {
        "quarantined_owner_count": 1,
        "quarantined_array_count": 2,
        "quarantined_bytes": 64,
        "quarantined_event_count": 2,
        "quarantined_stream_count": 1,
    }
    assert source_ref() is not None
    assert staging_ref() is not None
    assert store.ref("output") == destination_ref
    with pytest.raises(RuntimeError, match="post-enqueue event failure"):
        scheduler.close()
    with pytest.raises(RuntimeError, match="post-enqueue event failure"):
        pool.close()
    store.close()


def test_compute_owner_captures_stream_bindings_and_leases_before_event_creation():
    class CacheLease:
        pass

    store = HostTensorStore(store_id="compute-owner-capture")
    backend = _FakeCupyBackend()
    pool = PinnedBufferPool(32)
    array = np.arange(4, dtype=np.float64)
    output = np.arange(4, dtype=np.float64) + 1
    bindings = ExecutionBindings({"input": array})
    lease = CacheLease()
    scheduler = None
    observed = {}

    def fail_event_creation():
        owner = next(iter(scheduler._owners.values()))
        observed["resources"] = (
            bindings in owner.resources,
            lease in owner.resources,
        )
        observed["array"] = any(retained is array for retained in owner.arrays)
        observed["output"] = any(retained is output for retained in owner.arrays)
        observed["stream"] = backend._cupy.cuda.stream in owner.streams
        raise RuntimeError("injected compute event creation failure")

    scheduler = TransferScheduler(
        store,
        backend,
        pool,
        event_factory=fail_event_creation,
    )
    with pytest.raises(RuntimeError, match="compute event creation failure"):
        scheduler.record_compute_completion(
            cache_leases=(lease,), bindings=bindings, arrays=(output,)
        )

    assert observed == {
        "resources": (True, True),
        "array": True,
        "output": True,
        "stream": True,
    }
    assert backend._cupy.cuda.stream.synchronize_calls == 1
    assert scheduler._owners == {}
    scheduler.close()
    pool.close()
    store.close()


def test_compute_owner_is_registered_before_execution_begins():
    class CacheLease:
        pass

    store = HostTensorStore(store_id="compute-owner-begin")
    backend = _FakeCupyBackend()
    pool = PinnedBufferPool(32)
    array = np.arange(4, dtype=np.float64)
    bindings = ExecutionBindings({"input": array})
    lease = CacheLease()
    event_creations = []

    def forbidden_event_creation():
        event_creations.append(None)
        raise AssertionError("begin_compute created a completion event")

    scheduler = TransferScheduler(
        store,
        backend,
        pool,
        event_factory=forbidden_event_creation,
    )
    handle = scheduler.begin_compute(cache_leases=(lease,), bindings=bindings)
    owner = handle.owner

    assert event_creations == []
    assert scheduler._owners == {id(owner): owner}
    assert owner.state == "enqueued"
    assert owner.events == ()
    assert bindings in owner.resources
    assert lease in owner.resources
    assert any(retained is array for retained in owner.arrays)
    assert backend._cupy.cuda.stream in owner.streams

    scheduler._event_factory = lambda: _ManualEvent(done=True)
    assert scheduler.record_compute_completion(handle=handle) is handle
    scheduler.reap_completed()
    assert owner.state == "detached"
    assert scheduler._owners == {}
    scheduler.close()
    pool.close()
    store.close()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"event_factory": object()}, "event_factory"),
        (
            {"profile_enabled": True, "timing_event_factory": object()},
            "timing_event_factory",
        ),
        (
            {"profile_enabled": True, "elapsed_time_reader": object()},
            "elapsed_time_reader",
        ),
    ],
)
def test_scheduler_validates_supplied_factories_before_cupy_stream(options, message):
    store = HostTensorStore(store_id="factory-validation-before-stream")
    backend = _FakeCupyBackend()
    stream_calls = []
    pool = PinnedBufferPool(32)

    def forbidden_stream(*args, **kwargs):
        stream_calls.append((args, kwargs))
        raise AssertionError("stream created before supplied factory validation")

    backend._cupy.cuda.Stream = forbidden_stream

    with pytest.raises(TypeError, match=message):
        TransferScheduler(store, backend, pool, **options)

    assert stream_calls == []
    pool.close()
    store.close()


def test_scheduler_close_drains_unfinished_compute_owner_before_detach():
    store = HostTensorStore(store_id="compute-owner-close-drain")
    backend = _FakeCupyBackend()
    pool = PinnedBufferPool(32)
    scheduler = TransferScheduler(store, backend, pool)
    array = np.arange(4, dtype=np.float64)
    array_ref = weakref.ref(array)
    handle = scheduler.begin_compute(arrays=(array,))
    del array

    scheduler.close()
    gc.collect()

    assert backend._cupy.cuda.stream.synchronize_calls == 1
    assert handle.owner.state == "detached"
    assert scheduler._owners == {}
    assert array_ref() is None
    pool.close()
    store.close()


def test_terminal_stream_failure_quarantines_every_registered_owner():
    store = HostTensorStore(store_id="all-owner-quarantine")
    backend = _FakeCupyBackend(synchronize_fails=True)
    pool = PinnedBufferPool(32)
    quarantine = RuntimeTerminalQuarantine()
    scheduler = TransferScheduler(
        store,
        backend,
        pool,
        event_factory=_RecordFailureEvent,
        quarantine=quarantine.retain,
    )
    first_array = np.arange(4, dtype=np.float64)
    second_array = np.arange(4, dtype=np.float64) + 1
    first = scheduler.begin_compute(arrays=(first_array,))
    second = scheduler.begin_compute(arrays=(second_array,))

    with pytest.raises(RuntimeError, match="post-enqueue event failure"):
        scheduler.record_compute_completion(handle=first)

    assert first.owner.quarantined is True
    assert second.owner.quarantined is True
    assert scheduler._owners == {}
    assert quarantine.resource_state() == {
        "quarantined_owner_count": 2,
        "quarantined_array_count": 2,
        "quarantined_bytes": 64,
        "quarantined_event_count": 1,
        "quarantined_stream_count": 1,
    }
    with pytest.raises(RuntimeError, match="post-enqueue event failure"):
        scheduler.close()
    pool.close()
    store.close()


def test_compute_post_enqueue_drain_failure_reaches_runtime_quarantine():
    class RecordingCollective:
        rank = 0
        size = 1

        def __init__(self):
            self.close_calls = 0
            self._fatal_error = None

        @staticmethod
        def _bootstrap_fatal_control():
            return 0

        def _install_fatal_handler(self, handler):
            self._fatal_handler = handler

        def _publish_communicator_fatal(self, error):
            if self._fatal_error is None:
                self._fatal_error = error

        @staticmethod
        def _bootstrap_status_or(local_code):
            return local_code

        def close(self):
            self.close_calls += 1

    collective = RecordingCollective()
    runtime = _runtime(collective=collective)
    request, plan, store, _, _ = _active_case(
        runtime, store_id="runtime-owner-quarantine"
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    runtime._active_provider = provider
    working_set = provider.open_working_set(request, plan, store, receipt).__enter__()
    child = working_set.acquire(_block_request(working_set)).__enter__()
    owner = child._compute_handle.owner
    expected_records = {}
    for record in (
        *working_set._status_workspace.allocation_records,
        *provider.cache.allocation_records,
        *working_set.pool.allocation_records,
        *owner.allocations,
    ):
        retained = expected_records.get(record.identity)
        if retained is None or record.capacity_bytes > retained.capacity_bytes:
            expected_records[record.identity] = record
    retained_array = next(iter(child.bindings.arrays.values()))
    retained_ref = weakref.ref(retained_array)
    scheduler = working_set.scheduler
    fake_backend = _FakeCupyBackend(synchronize_fails=True)
    scheduler.backend = fake_backend
    scheduler._cupy = fake_backend._cupy
    scheduler._stream = fake_backend._cupy.cuda.stream
    owner.capture_streams(fake_backend._cupy.cuda.stream)
    scheduler._event_factory = _RecordFailureEvent

    with pytest.raises(RuntimeError, match="post-enqueue event failure"):
        child.close()
    del retained_array
    gc.collect()

    assert provider.terminal_poisoned is True
    assert runtime.terminal_poisoned is True
    expected_state = {
        "quarantined_owner_count": 1,
        "quarantined_array_count": len(expected_records),
        "quarantined_bytes": sum(
            record.capacity_bytes for record in expected_records.values()
        ),
        "quarantined_event_count": 1,
        "quarantined_stream_count": 1,
    }
    for key, value in expected_state.items():
        assert provider.resource_state()[key] == value
        assert runtime.resource_state()[key] == value
    with pytest.raises(RuntimeError, match="terminal-poisoned"):
        provider.open_working_set(request, plan, store, receipt)
    with pytest.raises(RuntimeError, match="terminal-poisoned"):
        runtime.preflight_residency(request, plan)
    with pytest.raises(RuntimeError, match="terminal-poisoned"):
        runtime.execution_config(residency_policy="active_working_set")

    with pytest.raises(RuntimeError, match="post-enqueue event failure"):
        runtime.close()
    assert collective.close_calls == 0
    runtime.close()
    assert collective.close_calls == 0
    for key, value in expected_state.items():
        assert provider.resource_state()[key] == value
        assert runtime.resource_state()[key] == value
    gc.collect()
    assert retained_ref() is not None
    assert store.closed is False
    assert store._reservations


def test_repeated_completed_events_do_not_grow_owner_mapping():
    store = HostTensorStore(store_id="bounded-event-owners")
    pool = PinnedBufferPool(32)
    runtime = _runtime()
    scheduler = TransferScheduler(store, runtime.backend, pool)
    event_refs = []

    for _ in range(64):
        event = scheduler.record_compute_completion()
        event_refs.append(weakref.ref(event))
        scheduler.reap_completed()
        assert scheduler._event_owners == {}

    del event
    gc.collect()
    assert all(reference() is None for reference in event_refs)
    scheduler.close()
    pool.close()
    store.close()
    runtime.close()


def test_scheduler_query_failure_drains_stream_before_finalizing_ticket():
    store = HostTensorStore(store_id="scheduler-query-drain")
    ref = store.put("input", np.arange(4, dtype=np.float64))
    backend = _FakeCupyBackend()
    pool = PinnedBufferPool(32)
    event = _SchedulerQueryFailureEvent()
    scheduler = TransferScheduler(
        store,
        backend,
        pool,
        event_factory=lambda: event,
    )
    ticket = scheduler.stage_h2d(ref, _FakeDestination(), _FakeStagingSlot())

    with pytest.raises(RuntimeError, match="scheduler query failure"):
        scheduler.reap_completed()

    assert backend._cupy.cuda.stream.synchronize_calls == 1
    assert event.synchronize_calls == 0
    assert ticket.completed is True
    assert scheduler.pending_ticket_count == 0
    assert scheduler.retained_event_count == 0
    assert scheduler.poisoned is False
    scheduler.close()
    pool.close()
    store.close()


def test_scheduler_query_and_stream_drain_failure_retains_terminal_ownership():
    store = HostTensorStore(store_id="scheduler-query-poison")
    ref = store.put("input", np.arange(4, dtype=np.float64))
    backend = _FakeCupyBackend(synchronize_fails=True)
    pool = PinnedBufferPool(32)
    event = _SchedulerQueryFailureEvent(synchronize_fails=True)
    scheduler = TransferScheduler(
        store,
        backend,
        pool,
        event_factory=lambda: event,
    )
    ticket = scheduler.stage_h2d(ref, _FakeDestination(), _FakeStagingSlot())

    with pytest.raises(RuntimeError, match="scheduler query failure"):
        scheduler.reap_completed()

    assert backend._cupy.cuda.stream.synchronize_calls == 1
    assert ticket.completed is False
    assert scheduler.pending_ticket_count == 1
    assert scheduler.retained_event_count == 1
    assert scheduler.poisoned is True
    with pytest.raises(RuntimeError, match="terminal-poisoned"):
        scheduler.stage_h2d(ref, _FakeDestination(), _FakeStagingSlot())
    with pytest.raises(RuntimeError, match="scheduler query failure"):
        scheduler.close()
    assert scheduler.pending_ticket_count == 0
    assert scheduler.retained_event_count == 1
    pool.close()
    store.close()


def test_scheduler_wait_and_stream_drain_failure_poison_immediately():
    store = HostTensorStore(store_id="scheduler-wait-poison")
    ref = store.put("input", np.arange(4, dtype=np.float64))
    backend = _FakeCupyBackend(synchronize_fails=True)
    event = _SchedulerWaitFailureEvent()
    pool = PinnedBufferPool(32)
    scheduler = TransferScheduler(
        store,
        backend,
        pool,
        event_factory=lambda: event,
    )
    ticket = scheduler.stage_h2d(ref, _FakeDestination(), _FakeStagingSlot())

    with pytest.raises(RuntimeError, match="scheduler wait failure"):
        ticket.wait()

    assert backend._cupy.cuda.stream.synchronize_calls == 1
    assert ticket.terminal_poisoned is True
    assert ticket.completed is False
    assert scheduler.poisoned is True
    with pytest.raises(RuntimeError, match="terminal-poisoned"):
        scheduler.record_compute_completion()
    with pytest.raises(RuntimeError, match="scheduler wait failure"):
        scheduler.close()
    pool.close()
    store.close()


def test_h2d_uses_destination_order_for_same_allocation_staging(monkeypatch):
    runtime = _runtime()
    store = HostTensorStore(store_id="f-order-staging")
    source = np.arange(6, dtype=np.float64).reshape(2, 3)
    ref = store.put("input", source)
    pool = PinnedBufferPool(source.nbytes)
    scheduler = TransferScheduler(store, runtime.backend, pool)
    destination = np.empty(source.shape, dtype=source.dtype, order="F")
    observed = []
    original_copy_into = store.copy_into

    def copy_into(source_ref, staging, local_slice=None):
        observed.append(
            (
                staging.flags.f_contiguous,
                staging.flags.c_contiguous,
                np.shares_memory(staging, pool._slot._array),
                staging.__array_interface__["data"][0],
            )
        )
        return original_copy_into(source_ref, staging, local_slice)

    monkeypatch.setattr(store, "copy_into", copy_into)
    try:
        with pool.checkout(source.nbytes) as slot:
            ticket = scheduler.stage_h2d(ref, destination, slot)
        ticket.wait()

        assert observed == [
            (
                True,
                False,
                True,
                pool._slot._array.__array_interface__["data"][0],
            )
        ]
        np.testing.assert_array_equal(destination, source)
        assert pool.capacity_bytes == source.nbytes
        assert pool.pending_bytes == 0
    finally:
        scheduler.close()
        pool.close()
        store.close()
        runtime.close()


def test_h2d_failure_before_enqueue_evicts_entry_and_poison_is_sticky():
    class FailingScheduler(TransferScheduler):
        def stage_h2d(self, source_ref, destination, slot, **kwargs):
            raise RuntimeError("injected pre-enqueue H2D failure")

    runtime = _runtime()
    request, plan, store, _, _ = _active_case(runtime, store_id="load-failure")
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request, scheduler_factory=FailingScheduler)
    working_set = provider.open_working_set(request, plan, store, receipt).__enter__()
    try:
        with pytest.raises(RuntimeError, match="pre-enqueue H2D failure"):
            working_set.acquire(_block_request(working_set))
        assert provider.cache.allocated_bytes == 0
        assert provider.cache.entry_count == 0
        with pytest.raises(RuntimeError, match="poisoned"):
            working_set.acquire(_block_request(working_set))
        with pytest.raises(RuntimeError, match="pre-enqueue H2D failure"):
            working_set.close()
    finally:
        if not working_set._closed:
            working_set._poisoned_error = None
            working_set.close()
        store.close()
        runtime.close()


def test_staging_failure_after_cache_allocation_evicts_and_poisons(monkeypatch):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(runtime, store_id="staging-failure")
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    working_set = provider.open_working_set(request, plan, store, receipt).__enter__()

    def fail_staging(**_kwargs):
        raise RuntimeError("injected staging availability failure")

    monkeypatch.setattr(working_set, "_wait_for_staging", fail_staging)
    try:
        with pytest.raises(RuntimeError, match="staging availability failure"):
            working_set.acquire(_block_request(working_set))
        assert provider.cache.entry_count == 0
        assert provider.cache.allocated_bytes == 0
        with pytest.raises(RuntimeError, match="poisoned"):
            working_set.acquire(_block_request(working_set))
    finally:
        working_set._poisoned_error = None
        working_set.close()
        store.close()
        runtime.close()


def test_transfer_ticket_times_completion_and_releases_callback_and_event():
    class CallbackOwner:
        pass

    event = _ManualEvent()
    owner = CallbackOwner()
    owner_ref = weakref.ref(owner)
    ticks = iter((2.0, 7.5))

    def callback(retained=owner):
        assert retained is not None

    async_owner = AsyncResourceOwner(
        "h2d",
        nbytes=32,
        callback=callback,
        timer=lambda: next(ticks),
    )
    async_owner.add_event(event, completion=True)
    async_owner.mark_enqueued()
    async_owner.arm_completion()
    ticket = TransferTicket(async_owner)
    del owner, callback

    assert ticket.elapsed_s == 0.0
    assert ticket.reap() is False
    event.complete()
    assert ticket.reap() is True
    gc.collect()

    assert ticket.elapsed_s == 5.5
    assert ticket.event is None
    assert owner_ref() is None


def test_transfer_ticket_rejects_raw_event_owner_construction():
    with pytest.raises(TypeError, match="pre-existing AsyncResourceOwner"):
        TransferTicket(_ManualEvent())


def test_terminal_ticket_retains_callback_owner_and_unsafe_array():
    class CallbackOwner:
        pass

    event = _SchedulerQueryFailureEvent(synchronize_fails=True)
    array = np.arange(4, dtype=np.float64)
    owner = CallbackOwner()
    array_ref = weakref.ref(array)
    owner_ref = weakref.ref(owner)

    def callback(retained=owner):
        assert retained is not None

    def fail_drain():
        raise RuntimeError("injected ticket owner drain failure")

    async_owner = AsyncResourceOwner(
        "d2h",
        arrays=(array,),
        resources=(array,),
        nbytes=array.nbytes,
        callback=callback,
        drainer=fail_drain,
    )
    async_owner.add_event(event, completion=True)
    async_owner.mark_enqueued()
    async_owner.arm_completion()
    ticket = TransferTicket(async_owner)
    with pytest.raises(RuntimeError, match="scheduler query failure"):
        ticket.reap()
    del array, owner, callback
    gc.collect()

    assert ticket.terminal_poisoned is True
    assert array_ref() is not None
    assert owner_ref() is not None
    ticket.detach_terminal_callbacks()
    gc.collect()
    assert array_ref() is not None
    assert owner_ref() is not None
    assert ticket.event is event


def test_profiled_cupy_transfer_reads_elapsed_from_completion_events():
    store = HostTensorStore(store_id="profiled-cupy-events")
    ref = store.put("input", np.arange(4, dtype=np.float64))
    events = []
    elapsed_reads = []

    def timing_event_factory():
        event = _ManualEvent()
        events.append(event)
        return event

    def elapsed_time_reader(start, end):
        elapsed_reads.append((start, end))
        return 4.25

    scheduler = TransferScheduler(
        store,
        _FakeCupyBackend(),
        PinnedBufferPool(32),
        event_factory=lambda: pytest.fail("separate completion event was created"),
        profile_enabled=True,
        timing_event_factory=timing_event_factory,
        elapsed_time_reader=elapsed_time_reader,
    )
    ticket = scheduler.stage_h2d(ref, _FakeDestination(), _FakeStagingSlot())

    assert len(events) == 2
    assert [event.record_calls for event in events] == [1, 1]
    assert elapsed_reads == []
    assert ticket.reap() is False
    events[1].complete()
    scheduler.reap_completed()

    assert ticket.completed is True
    assert ticket.elapsed_s == pytest.approx(0.00425)
    assert elapsed_reads == [(events[0], events[1])]
    assert scheduler.h2d_s == pytest.approx(0.00425)
    scheduler.close()
    store.close()


def test_unprofiled_cupy_transfer_creates_no_timing_events_or_elapsed_reads():
    store = HostTensorStore(store_id="unprofiled-cupy-events")
    ref = store.put("input", np.arange(4, dtype=np.float64))
    completion = _ManualEvent(done=True)

    def forbidden(*args, **kwargs):
        pytest.fail("disabled profiling used a timing helper")

    scheduler = TransferScheduler(
        store,
        _FakeCupyBackend(),
        PinnedBufferPool(32),
        event_factory=lambda: completion,
        profile_enabled=False,
        timing_event_factory=forbidden,
        elapsed_time_reader=forbidden,
    )
    ticket = scheduler.stage_h2d(ref, _FakeDestination(), _FakeStagingSlot())
    scheduler.reap_completed()

    assert ticket.elapsed_s == 0.0
    scheduler.close()
    store.close()


def test_dirty_version_changes_only_after_d2h_completion():
    events = []

    def event_factory():
        event = _ManualEvent()
        events.append(event)
        return event

    runtime = _runtime()
    request, plan, store, _, center = _active_case(runtime)
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request, event_factory=event_factory)
    working_set = provider.open_working_set(request, plan, store, receipt).__enter__()
    before = store.ref("output")
    working_set.mark_dirty("output", np.arange(4.0))

    working_set.close(wait=False)
    assert store.ref("output") == before
    for event in events:
        event.complete()
    working_set.close()

    assert store.ref("output").version == center.version + 1
    np.testing.assert_array_equal(store.read(store.ref("output")), np.arange(4.0))
    assert provider.metrics.dirty_writeback_count == 1
    store.close()
    runtime.close()


def test_dirty_mark_rejects_wrong_backend_device_and_layout(monkeypatch):
    class ArrayLike:
        shape = (4,)
        dtype = np.dtype(np.float64)
        nbytes = 32

        class flags:
            c_contiguous = True
            f_contiguous = True

    runtime = _runtime()
    request, plan, store, _, _ = _active_case(runtime, store_id="dirty-validation")
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    working_set = provider.open_working_set(request, plan, store, receipt).__enter__()
    try:
        with pytest.raises(TypeError, match="NumPy execution binding"):
            working_set.mark_dirty("output", ArrayLike())

        strided = np.arange(8, dtype=np.float64)[::2]
        with pytest.raises(ValueError, match="layout"):
            working_set.mark_dirty("output", strided)

        def reject_device(value):
            raise ValueError("array is not on the selected local device")

        monkeypatch.setattr(runtime.backend, "_validate_execution_array", reject_device)
        with pytest.raises(ValueError, match="selected local device"):
            working_set.mark_dirty("output", np.arange(4, dtype=np.float64))
    finally:
        working_set._dirty = None
        working_set.close()
        store.close()
        runtime.close()


def test_child_close_detaches_bindings_and_provider_event_reference():
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(runtime, store_id="child-release")
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    working_set = provider.open_working_set(request, plan, store, receipt).__enter__()
    broadcast = np.arange(4, dtype=np.float64)
    broadcast_ref = weakref.ref(broadcast)
    child = working_set.acquire(
        _block_request(working_set, broadcast_variable=broadcast)
    )
    bindings_ref = weakref.ref(child.bindings)

    child.close()
    del broadcast
    gc.collect()

    assert child.bindings is None
    assert bindings_ref() is not None
    assert broadcast_ref() is not None
    working_set.reap_completed()
    gc.collect()
    assert bindings_ref() is None
    assert broadcast_ref() is None
    assert provider.last_compute_event is None
    working_set.close()
    store.close()
    runtime.close()


def test_dirty_writeback_uses_authorized_promoted_result_dtype():
    runtime = _runtime()
    request, plan, store, _, center = _active_case(
        runtime,
        store_id="promoted-dirty",
        dtype="float32",
        coefficient=-0.125j,
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    promoted = np.arange(4, dtype=np.complex128) * (1 - 2j)

    with provider.open_working_set(request, plan, store, receipt) as working_set:
        working_set.mark_dirty("output", promoted)

    updated = store.ref("output")
    assert updated.version == center.version + 1
    assert updated.dtype == "complex128"
    np.testing.assert_array_equal(store.read(updated), promoted)
    store.close()
    runtime.close()


@pytest.mark.parametrize("aliases_cache", (False, True), ids=("unique", "alias"))
@pytest.mark.parametrize("backend_name", ("numpy", "cupy"))
def test_terminal_dirty_allocation_is_accounted_after_explicit_owner_reap(
    backend_name, aliases_cache
):
    if backend_name == "cupy" and not _has_one_visible_gpu():
        pytest.skip("requires exactly one visible CUDA device")
    (
        runtime,
        store,
        provider,
        working_set,
        operator,
        local_vector,
    ) = _active_operator_case(
        store_id="terminal-dirty-{}-{}".format(
            backend_name, "alias" if aliases_cache else "unique"
        ),
        backend_name=backend_name,
    )
    try:
        operator(local_vector)
        prior_owner = working_set._status_workspace.borrower
        assert prior_owner is not None
        working_set.reap_completed()
        assert prior_owner.state == "detached"
        assert working_set._status_workspace.borrower is None

        if aliases_cache:
            cache_entry = next(iter(provider.cache._entries.values()))
            dirty = cache_entry.array[0]
        else:
            dirty = working_set.backend.ones((4,), dtype=np.float64)
        working_set.mark_dirty("output", dirty)
        dirty_record = allocation_record(dirty)

        expected_records = {}
        for record in (
            *working_set._status_workspace.allocation_records,
            *provider.cache.allocation_records,
            *working_set.pool.allocation_records,
            dirty_record,
        ):
            retained = expected_records.get(record.identity)
            if retained is None or record.capacity_bytes > retained.capacity_bytes:
                expected_records[record.identity] = record
        cache_identities = {
            record.identity for record in provider.cache.allocation_records
        }
        assert (dirty_record.identity in cache_identities) is aliases_cache

        primary = RuntimeError("injected terminal dirty accounting failure")
        assert runtime._enter_communicator_fatal(primary) is primary
        assert working_set.__exit__(type(primary), primary, None) is False

        retained = {
            record.identity: record
            for record in runtime._terminal_quarantine.allocations
        }
        assert set(retained) == set(expected_records)
        assert all(
            retained[identity].capacity_bytes == record.capacity_bytes
            for identity, record in expected_records.items()
        )
        assert (
            sum(
                record.identity == dirty_record.identity
                for record in runtime._terminal_quarantine.allocations
            )
            == 1
        )
        assert runtime.resource_state()["quarantined_bytes"] == sum(
            record.capacity_bytes for record in expected_records.values()
        )
        assert working_set._dirty[1] is dirty
        assert any(
            record.identity == dirty_record.identity
            for record in working_set.allocation_records
        )
        with pytest.raises(RuntimeError) as caught:
            runtime.close()
        assert caught.value is primary
    finally:
        if not runtime._closed:
            try:
                runtime.close()
            except BaseException:
                pass
        if not store.closed:
            if runtime._terminal_error is None:
                store.close()
            else:
                assert store._reservations


def test_working_set_close_releases_dirty_array_ticket_and_staging_storage():
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(runtime, store_id="outer-release")
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    working_set = provider.open_working_set(request, plan, store, receipt).__enter__()
    dirty = np.arange(4, dtype=np.float64)
    dirty_ref = weakref.ref(dirty)
    staging_ref = weakref.ref(working_set.pool._slot._array)
    working_set.mark_dirty("output", dirty)
    del dirty

    working_set.close()
    gc.collect()

    assert working_set._dirty is None
    assert working_set._dirty_allocation_records == ()
    assert working_set._writeback_ticket is None
    assert working_set.pool is None
    assert working_set.scheduler is None
    assert dirty_ref() is None
    assert staging_ref() is None
    assert provider.last_compute_event is None
    store.close()
    runtime.close()


def _assert_cleanup_failure_is_fatal_retained(
    runtime,
    provider,
    working_set,
    primary,
):
    gate = runtime._terminal_gate
    assert provider._active_lease is working_set
    assert provider.resource_state()["active_leases"] == 1
    assert working_set._store_reservation is not None
    assert working_set._store_reservation._closed is False
    with gate._condition:
        assert gate._fatal_transition.primary is primary
        assert gate._phase is _TerminalPhase.FATAL_PUBLISHED
        assert gate._leases[working_set._epoch].phase == "fatal_retained"
        assert not gate._has_active_tokens()


def _release_retained_working_set_test_case(runtime, store, working_set):
    reservation = working_set._store_reservation
    if reservation is not None and not reservation._closed:
        reservation.close()
    runtime._closed = True
    if not store.closed:
        store.close()


def test_writeback_scheduling_failure_preserves_error_and_retains_all_owners(
    monkeypatch,
):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(runtime, store_id="close-failure")
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    working_set = provider.open_working_set(request, plan, store, receipt).__enter__()
    working_set.mark_dirty("output", np.arange(4, dtype=np.float64))

    primary = RuntimeError("injected writeback scheduling failure")

    def fail_writeback(*args, **kwargs):
        raise primary

    monkeypatch.setattr(working_set.scheduler, "writeback_d2h", fail_writeback)
    try:
        with pytest.raises(RuntimeError, match="writeback scheduling failure") as caught:
            working_set.close()
        assert caught.value is primary
        _assert_cleanup_failure_is_fatal_retained(
            runtime,
            provider,
            working_set,
            primary,
        )
        assert working_set._dirty is not None
    finally:
        monkeypatch.undo()
        _release_retained_working_set_test_case(runtime, store, working_set)


def test_checkout_release_uncertainty_retains_writeback_ticket(monkeypatch):
    runtime = _runtime()
    request, plan, store, _, center = _active_case(
        runtime, store_id="writeback-release-failure"
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    working_set = provider.open_working_set(request, plan, store, receipt).__enter__()
    dirty = np.arange(4, dtype=np.float64) + 11
    working_set.mark_dirty("output", dirty)
    release = working_set.pool._release
    commit = working_set._store_reservation.commit
    invalidate = provider.cache.invalidate_ref
    commits = []
    invalidations = []
    primary = RuntimeError("injected checkout release failure")

    def record_commit(ref, value):
        commits.append(ref)
        return commit(ref, value)

    def fail_after_release(slot):
        release(slot)
        raise primary

    def record_invalidation(ref, **kwargs):
        invalidations.append(ref)
        return invalidate(ref, **kwargs)

    monkeypatch.setattr(working_set._store_reservation, "commit", record_commit)
    monkeypatch.setattr(working_set.pool, "_release", fail_after_release)
    monkeypatch.setattr(provider.cache, "invalidate_ref", record_invalidation)

    try:
        with pytest.raises(RuntimeError, match="checkout release failure") as caught:
            working_set.close()
        assert caught.value is primary
        updated = store.ref("output")
        assert updated.version == center.version + 1
        np.testing.assert_array_equal(store.read(updated), dirty)
        assert commits == [center]
        assert invalidations == []
        assert working_set.metrics.dirty_writeback_count == 0
        assert working_set._writeback_ticket is not None
        _assert_cleanup_failure_is_fatal_retained(
            runtime,
            provider,
            working_set,
            primary,
        )
    finally:
        monkeypatch.undo()
        _release_retained_working_set_test_case(runtime, store, working_set)


def test_successful_dirty_cas_is_accounted_once_before_invalidation_failure(
    monkeypatch,
):
    runtime = _runtime()
    request, plan, store, _, center = _active_case(
        runtime, store_id="writeback-invalidation-failure"
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    working_set = provider.open_working_set(request, plan, store, receipt).__enter__()
    dirty = np.arange(4, dtype=np.float64) + 19
    working_set.mark_dirty("output", dirty)
    commit = working_set._store_reservation.commit
    commits = []
    invalidations = []
    primary = RuntimeError("injected cache invalidation failure")

    def record_commit(ref, value):
        commits.append(ref)
        return commit(ref, value)

    def fail_invalidation(ref, **_kwargs):
        invalidations.append(ref)
        raise primary

    monkeypatch.setattr(working_set._store_reservation, "commit", record_commit)
    monkeypatch.setattr(provider.cache, "invalidate_ref", fail_invalidation)

    try:
        with pytest.raises(RuntimeError, match="cache invalidation failure") as caught:
            working_set.close()
        assert caught.value is primary
        updated = store.ref("output")
        assert updated.version == center.version + 1
        np.testing.assert_array_equal(store.read(updated), dirty)
        assert commits == [center]
        assert invalidations == [center]
        assert working_set.metrics.dirty_writeback_count == 1
        assert working_set.metrics.dirty_writeback_bytes == center.nbytes
        assert working_set.metrics.dirty_writeback_s >= 0.0
        _assert_cleanup_failure_is_fatal_retained(
            runtime,
            provider,
            working_set,
            primary,
        )
    finally:
        monkeypatch.undo()
        _release_retained_working_set_test_case(runtime, store, working_set)


def test_writeback_callback_uncertainty_retains_dirty_refs_and_reservation(
    monkeypatch,
):
    runtime = _runtime()
    request, plan, store, _, center = _active_case(
        runtime, store_id="writeback-callback-failure"
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    working_set = provider.open_working_set(request, plan, store, receipt).__enter__()
    dirty = np.arange(4, dtype=np.float64)
    dirty_ref = weakref.ref(dirty)
    working_set.mark_dirty("output", dirty)
    del dirty

    primary = RuntimeError("injected dirty CAS callback failure")

    def fail_commit(ref, value):
        raise primary

    monkeypatch.setattr(working_set._store_reservation, "commit", fail_commit)
    try:
        with pytest.raises(RuntimeError, match="dirty CAS callback failure") as caught:
            working_set.close()
        assert caught.value is primary
        gc.collect()
        assert store.ref("output") == center
        assert working_set._dirty is not None
        assert working_set._writeback_ticket is not None
        assert dirty_ref() is not None
        _assert_cleanup_failure_is_fatal_retained(
            runtime,
            provider,
            working_set,
            primary,
        )
    finally:
        monkeypatch.undo()
        _release_retained_working_set_test_case(runtime, store, working_set)


def test_metrics_are_bounded_scalars_replaced_for_each_outer_lease():
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(runtime, store_id="lease-metrics")
    provider = _provider(runtime, request)

    first_receipt = runtime.preflight_residency(request, plan)
    with provider.open_working_set(request, plan, store, first_receipt) as first:
        with first.acquire(_block_request(first)):
            pass
    first_metrics = provider.metrics
    assert first_metrics.h2d_count == 1
    assert not hasattr(first_metrics, "h2d_count_by_key")

    second_receipt = runtime.preflight_residency(request, plan)
    with provider.open_working_set(request, plan, store, second_receipt) as second:
        with second.acquire(_block_request(second)):
            pass
    second_metrics = provider.metrics
    assert second_metrics is not first_metrics
    assert second_metrics.h2d_count == 0
    assert second_metrics.cache_hits == 1

    store.close()
    runtime.close()


def test_factory_to_working_set_borrowed_config_handoff_preserves_owners():
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(runtime)
    provider = _provider(runtime, request)
    runtime._active_provider = provider
    config = DistributedExecutionConfig(
        context=runtime.context,
        mesh=runtime.mesh,
        collective=runtime.collective,
        provider=provider,
        residency_policy="active_working_set",
        device_memory_budget_bytes=request.device_budget.requested_bytes,
        host_memory_budget_bytes=request.host_budget.requested_bytes,
        prefetch_depth=1,
        backend_name="numpy",
        backend_device="cpu",
        backend_precision=64,
        device_budget_resolution=request.device_budget,
        host_budget_resolution=request.host_budget,
    )

    with pytest.raises(RuntimeError, match="factory"):
        provider.acquire(None)
    with active_working_set_execution(config, request, plan, store) as borrowed:
        assert borrowed is not config
        assert borrowed.provider.provider_role == "working_set"
        assert borrowed.context is config.context
        assert borrowed.mesh is config.mesh
        assert borrowed.collective is config.collective
        assert borrowed.residency_request is request
        assert borrowed.residency_plan is plan
        assert borrowed.residency_receipt is borrowed.provider.receipt
    assert provider.closed is False
    assert runtime._closed is False
    store.close()
    runtime.close()


def test_runtime_close_preserves_provider_error_but_still_closes_collective():
    class FailingOnceProvider:
        def __init__(self):
            self.close_calls = 0

        def _close_for_runtime(self, **_kwargs):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("injected provider close failure")

    class RecordingCollective:
        rank = 0
        size = 1

        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    collective = RecordingCollective()
    runtime = _runtime(collective=collective)
    provider = FailingOnceProvider()
    runtime._active_provider = provider
    runtime._issued_receipts[1] = object()

    with pytest.raises(RuntimeError, match="provider close failure"):
        runtime.close()

    assert provider.close_calls == 1
    assert collective.close_calls == 1
    assert runtime._issued_receipts == {}
    assert runtime._active_provider is None
    assert runtime.collective is None
    assert runtime._closed is True
    runtime.close()
    assert provider.close_calls == 1
    assert collective.close_calls == 1
    with pytest.raises(RuntimeError, match="distributed runtime is closed"):
        runtime.barrier()
    with pytest.raises(RuntimeError, match="distributed runtime is closed"):
        runtime.execution_config()


def test_runtime_terminal_guard_rejects_barrier_and_device_resident_config():
    class RecordingCollective:
        rank = 0
        size = 1

        def __init__(self):
            self.barrier_calls = 0

        def barrier(self):
            self.barrier_calls += 1

        def close(self):
            pass

    collective = RecordingCollective()
    runtime = _runtime(collective=collective)
    first_error = RuntimeError("injected terminal owner failure")
    runtime._terminal_error = first_error

    for operation in (
        runtime.barrier,
        lambda: runtime.execution_config(residency_policy="device_resident"),
    ):
        with pytest.raises(RuntimeError, match="terminal-poisoned") as caught:
            operation()
        assert caught.value.__cause__ is first_error

    assert collective.barrier_calls == 0
    with pytest.raises(RuntimeError, match="terminal owner failure"):
        runtime.close()


def test_scheduler_quarantine_retains_opaque_owner_when_downstream_callback_fails():
    class OpaqueResource:
        def __init__(self, error):
            self.error = error
            self.accesses = []

        def __getattr__(self, name):
            self.accesses.append(name)
            raise self.error

    store = HostTensorStore(store_id="scheduler-local-opaque-quarantine")
    backend = _FakeCupyBackend(synchronize_fails=True)
    pool = PinnedBufferPool(32)
    callback_error = RuntimeError("injected downstream quarantine failure")
    callback_calls = []

    def fail_quarantine(owner):
        callback_calls.append(owner)
        raise callback_error

    scheduler = TransferScheduler(
        store,
        backend,
        pool,
        event_factory=_ManualEvent,
        quarantine=fail_quarantine,
    )
    primary_error = RuntimeError("injected scheduler owner failure")
    classification_error = RuntimeError("opaque resource attribute access")
    resource = OpaqueResource(classification_error)
    array = np.empty(1)
    handle = scheduler.begin_compute(arrays=(array,), resources=(resource,))

    with pytest.raises(RuntimeError) as caught:
        handle.owner.fail(primary_error)

    assert caught.value is primary_error
    assert handle.owner.error is primary_error
    assert callback_calls == [handle.owner]
    assert callback_error in handle.owner.secondary_errors
    assert resource.accesses == []
    assert scheduler._quarantined_owners == [handle.owner]
    assert scheduler._owners == {}
    assert handle.owner.arrays == (array,)
    assert handle.owner.resources == (resource,)
    assert handle.owner.streams == (backend._cupy.cuda.stream,)

    with pytest.raises(RuntimeError) as caught:
        scheduler.close()
    assert caught.value is primary_error
    pool.close()
    store.close()


def test_terminal_cupy_execution_entries_reject_before_any_backend_access():
    from renormalizer.backend._execution.executor import execute_plan
    from renormalizer.backend._gemm.executor import execute_grouped_gemm

    class ExplodingCupy:
        def __init__(self):
            self.accesses = []

        def __getattr__(self, name):
            self.accesses.append(name)
            raise AssertionError("terminal backend touched CuPy")

    class ExplodingSentinel:
        def __init__(self, name):
            self.name = name
            self.accesses = []

        def explode(self, operation):
            self.accesses.append(operation)
            raise AssertionError(
                "terminal backend touched {} via {}".format(self.name, operation)
            )

        def __getattr__(self, name):
            self.explode("attribute {}".format(name))

        def __iter__(self):
            self.explode("iteration")

        def __len__(self):
            self.explode("length")

        def __getitem__(self, key):
            self.explode("item {!r}".format(key))

        def __contains__(self, key):
            self.explode("contains {!r}".format(key))

    class MalformedOpaqueVector(np.ndarray):
        @property
        def shape(self):
            self.accesses.append("shape")
            raise self.primary_error

        def __getattr__(self, name):
            self.accesses.append(name)
            raise self.classification_error

    (
        runtime,
        store,
        provider,
        working_set,
        operator,
        _,
    ) = _active_operator_case(store_id="active-terminal-cupy-guard")
    backend = object.__new__(CupyBackend)
    backend._cupy = _FakeCupy(synchronize_fails=True)
    backend._device_index = 0
    backend._execution_terminal_error = None
    backend._execution_terminal_quarantine = None
    runtime.backend = backend
    runtime._active_provider = provider
    scheduler = working_set.scheduler
    scheduler.backend = backend
    scheduler._cupy = backend._cupy
    scheduler._stream = backend._cupy.cuda.stream
    primary_error = RuntimeError("injected malformed H-v failure")
    classification_error = RuntimeError("opaque H-v attribute access")
    local_vector = np.empty(4).view(MalformedOpaqueVector)
    local_vector.primary_error = primary_error
    local_vector.classification_error = classification_error
    local_vector.accesses = []

    with pytest.raises(ValueError, match="call-ready preflight") as caught:
        operator(local_vector)
    assert caught.value.__cause__ is primary_error
    assert runtime._terminal_quarantine.owners
    owner = runtime._terminal_quarantine.owners[0]
    assert owner.state == "quarantined"
    assert owner.error is primary_error
    assert scheduler._quarantined_owners == [owner]
    assert any(resource is local_vector for resource in owner.resources)
    assert any(
        resource is working_set._status_workspace for resource in owner.resources
    )
    assert working_set._status_workspace.device_status in owner.arrays
    assert working_set._status_workspace.host_status in owner.arrays
    assert owner.streams == (scheduler._stream,)
    assert provider._terminal_error is primary_error
    assert runtime._terminal_quarantine.owners == (owner,)
    assert runtime._terminal_quarantine.first_error is primary_error
    assert runtime._terminal_error is primary_error
    assert backend._execution_terminal_error is primary_error
    assert local_vector.accesses == ["shape"]
    backend._cupy = ExplodingCupy()

    stream = ExplodingSentinel("stream")
    operand = ExplodingSentinel("operand")
    plan = ExplodingSentinel("plan")
    bindings = ExplodingSentinel("bindings")
    descriptors = ExplodingSentinel("descriptors")
    tensors = ExplodingSentinel("tensors")
    operations = (
        lambda: backend.matmul(operand, operand, stream=stream),
        lambda: backend.batched_matmul(operand, operand, stream=stream),
        lambda: backend.execute_plan(plan, bindings, stream=stream),
        lambda: backend.grouped_gemm(descriptors, tensors, stream=stream),
        lambda: execute_plan(backend, plan, bindings, stream=stream),
        lambda: execute_grouped_gemm(backend, descriptors, tensors, stream=stream),
    )
    for operation in operations:
        with pytest.raises(RuntimeError, match="terminal-poisoned") as rejected:
            operation()
        assert rejected.value.__cause__ is primary_error
    assert backend._cupy.accesses == []
    assert all(
        sentinel.accesses == []
        for sentinel in (stream, operand, plan, bindings, descriptors, tensors)
    )
    with pytest.raises(RuntimeError) as caught:
        runtime.close()
    assert caught.value is primary_error
    assert store.closed is False
    assert store._reservations


def test_terminal_close_merges_cache_and_owner_physical_backings_once():
    runtime = _runtime()
    request, _, store, _, _ = _active_case(
        runtime, store_id="terminal-physical-cache-accounting"
    )
    provider = _provider(runtime, request)
    runtime._active_provider = provider
    allocations = []

    def allocate(spec):
        array = np.empty(spec.shape, dtype=spec.dtype, order=spec.layout)
        allocations.append(weakref.ref(array))
        return array

    cache = DeviceTensorCache(64, allocator=allocate)
    provider._cache = cache

    def spec(key, start):
        identity = (
            store.store_id,
            key,
            0,
            0,
            0,
            "cpu",
            ((start, start + 4, 1),),
            "float64",
            "C",
        )
        return CacheEntrySpec(identity, (4,), "float64", "C", 32)

    left = spec("left", 0)
    right = spec("right", 4)
    reservation = cache.reserve({left.identity: left, right.identity: right}, 64)

    def complete_load(lease):
        event = _ManualEvent(done=True)
        owner = AsyncResourceOwner("h2d", arrays=(lease.array,))
        owner.add_event(event, completion=True)
        owner.mark_enqueued()
        lease.install_readiness(TransferTicket(owner))

    unowned = cache.acquire(left.identity)
    complete_load(unowned)
    left_record = allocation_record(unowned.array)
    unowned.close()

    held = cache.acquire(right.identity)
    complete_load(held)
    right_record = allocation_record(held.array)
    assert left_record.identity != right_record.identity
    assert left_record.capacity_bytes == right_record.capacity_bytes == 32
    terminal_error = RuntimeError("injected owner failure")
    drain_error = RuntimeError("injected owner drain failure")
    owner = AsyncResourceOwner(
        "compute",
        arrays=(held.array,),
        drainer=lambda: (_ for _ in ()).throw(drain_error),
        quarantine=provider._accept_async_quarantine,
    )
    owner.mark_enqueued()
    held.transfer_to(owner)
    with pytest.raises(RuntimeError) as caught:
        owner.fail(terminal_error)
    assert caught.value is terminal_error
    assert {record.identity for record in owner.allocations} == {right_record.identity}

    with pytest.raises(RuntimeError) as caught:
        provider.close()
    assert caught.value is terminal_error
    assert runtime.resource_state()["quarantined_bytes"] == 32
    assert {record.identity for record in runtime._terminal_quarantine.allocations} == {
        right_record.identity,
    }

    with pytest.raises(RuntimeError) as caught:
        runtime.close()
    assert caught.value is terminal_error
    assert runtime.resource_state()["quarantined_bytes"] == 32

    reservation = None
    unowned = None
    held = None
    owner = None
    cache = None
    provider = None
    left_record = None
    right_record = None
    gc.collect()
    assert all(reference() is not None for reference in allocations)
    store.close()


def test_terminal_close_accounts_completed_h2d_pool_cache_and_broadcast_streams(
    monkeypatch,
):
    (
        runtime,
        store,
        provider,
        working_set,
        operator,
        local_vector,
    ) = _active_operator_case(store_id="terminal-completed-h2d-broadcast")
    scheduler = working_set.scheduler
    with working_set.acquire(_block_request(working_set)):
        pass
    transfer_stream = _FakeTransferStream()
    compute_stream = _FakeTransferStream()

    class DualStreamCuda:
        @staticmethod
        def Device(index):
            return _FakeDeviceContext()

        @staticmethod
        def get_current_stream():
            return compute_stream

    scheduler._cupy = SimpleNamespace(cuda=DualStreamCuda())
    scheduler.backend = SimpleNamespace(_device_index=0)
    scheduler._stream = transfer_stream
    injected = RuntimeError("injected post-enqueue broadcast failure")
    broadcast = runtime.collective.broadcast

    def broadcast_then_raise(array, *, root):
        broadcast(array, root=root)
        raise injected

    monkeypatch.setattr(runtime.collective, "broadcast", broadcast_then_raise)
    with pytest.raises(RuntimeError) as caught:
        operator(local_vector)
    assert caught.value is injected
    assert working_set.pool.checked_out_bytes == 0
    assert working_set.pool.pending_bytes == 0

    pool_records = working_set.pool.allocation_records
    cache_records = provider.cache.allocation_records
    owners = runtime._terminal_quarantine.owners
    expected_records = {}
    for record in (
        *pool_records,
        *cache_records,
        *(record for owner in owners for record in owner.allocations),
    ):
        retained = expected_records.get(record.identity)
        if retained is None or record.capacity_bytes > retained.capacity_bytes:
            expected_records[record.identity] = record
    expected_streams = {
        id(stream)
        for stream in (
            transfer_stream,
            *(stream for owner in owners for stream in owner.streams),
        )
    }
    expected_events = {id(event) for owner in owners for event in owner.events}
    expected_cache_bytes = sum(
        record.capacity_bytes
        for record in {record.identity: record for record in cache_records}.values()
    )
    expected_pinned_bytes = sum(
        record.capacity_bytes
        for record in {record.identity: record for record in pool_records}.values()
    )

    try:
        runtime.close()
    except BaseException as error:
        assert error is injected
    state = runtime.resource_state()
    assert state["cache_bytes"] == expected_cache_bytes
    assert state["pinned_bytes"] == expected_pinned_bytes
    assert state["stream_count"] == len(expected_streams)
    assert state["event_count"] == len(expected_events)
    assert state["quarantined_bytes"] == sum(
        record.capacity_bytes for record in expected_records.values()
    )
    assert state["quarantined_stream_count"] == len(expected_streams)
    assert state["quarantined_event_count"] == len(expected_events)
    assert {
        record.identity for record in runtime._terminal_quarantine.allocations
    } == set(expected_records)
    assert store.closed is False
    assert store._reservations


def test_runtime_close_keeps_first_terminal_error_over_provider_cleanup_error():
    first_error = RuntimeError("injected first runtime terminal failure")

    class FailingProvider:
        def close(self):
            raise RuntimeError("injected later provider cleanup failure")

    runtime = _runtime()
    runtime._terminal_error = first_error
    runtime._active_provider = FailingProvider()

    with pytest.raises(RuntimeError, match="first runtime terminal failure") as caught:
        runtime.close()

    assert caught.value is first_error


def test_runtime_close_keeps_terminal_installed_during_collective_close():
    primary = RuntimeError("injected terminal during collective close")
    cleanup = RuntimeError("injected later collective close failure")

    class ClosingCollective(_LoopbackCollective):
        def close(self):
            self._fatal_handler(primary)
            raise cleanup

    runtime = _runtime(collective=ClosingCollective(1))
    runtime._arm_communicator_fatal()

    with pytest.raises(RuntimeError) as caught:
        runtime.close()

    assert caught.value is primary
    assert runtime._terminal_error is primary


def test_runtime_close_adopts_backend_poison_before_collective_cleanup_failure():
    first_error = RuntimeError("injected backend terminal failure")
    cleanup_error = RuntimeError("injected later collective close failure")

    class FailingCollective:
        rank = 0
        size = 1

        def close(self):
            raise cleanup_error

    runtime = _runtime(collective=FailingCollective())
    runtime.backend._execution_terminal_error = first_error

    with pytest.raises(RuntimeError) as caught:
        runtime.close()

    assert caught.value is first_error
    assert runtime._terminal_error is first_error


def test_provider_close_failure_is_terminal_and_detaches_owned_factories(monkeypatch):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime, store_id="provider-terminal-close"
    )
    provider = _provider(runtime, request)
    runtime._active_provider = provider
    receipt = runtime.preflight_residency(request, plan)
    provider.open_working_set(request, plan, store, receipt).__enter__().close()
    cache = provider.cache
    close_cache = cache.close
    close_calls = []

    def fail_after_cache_close(**kwargs):
        close_calls.append(None)
        close_cache(**kwargs)
        raise RuntimeError("injected provider cache close failure")

    monkeypatch.setattr(cache, "close", fail_after_cache_close)
    with pytest.raises(RuntimeError, match="provider cache close failure"):
        provider.close()

    assert provider.closed is True
    assert provider.runtime is None
    assert provider._active_lease is None
    assert provider._cache is None
    assert provider._cache_factory is None
    assert provider._pool_factory is None
    assert provider._scheduler_factory is None
    assert provider._event_factory is None
    assert runtime._active_provider is None
    with pytest.raises(RuntimeError, match="provider cache close failure"):
        provider.close()
    assert len(close_calls) == 1
    store.close()
    runtime.close()


def test_provider_close_keeps_first_terminal_error_over_lease_cleanup_error():
    first_error = RuntimeError("injected first provider terminal failure")

    class FailingLease:
        scheduler = None

        def close(self):
            raise RuntimeError("injected later lease cleanup failure")

    runtime = _runtime()
    request, _, store, _, _ = _active_case(
        runtime, store_id="provider-first-terminal-error"
    )
    provider = _provider(runtime, request)
    provider._terminal_error = first_error
    provider._active_lease = FailingLease()
    runtime._terminal_error = first_error
    runtime._active_provider = provider

    with pytest.raises(RuntimeError, match="first provider terminal failure") as caught:
        provider.close()

    assert caught.value is first_error
    store.close()
    runtime.close()


def test_active_provider_emits_bounded_planned_and_observed_telemetry(tmp_path):
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log

    runtime = _runtime()
    request, plan, store, _, _ = _active_case(runtime, store_id="profiled-active")
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    path = tmp_path / "active.jsonl"
    init_log(PROFILING)
    profiling.register_event_output(path, events={"working_set_transfer"})
    try:
        with provider.open_working_set(request, plan, store, receipt) as working_set:
            with working_set.acquire(_block_request(working_set)):
                pass
            working_set.mark_dirty("output", np.arange(4.0))
    finally:
        profiling.close_event_output()
        init_log(DEBUG)

    record = json.loads(path.read_text())
    assert record["event"] == "working_set_transfer"
    assert record["planned_cache_peak_bytes"] == (
        plan.rank_estimates[0].current_static_bytes
        + plan.rank_estimates[0].prefetched_static_bytes
    )
    assert record["observed_cache_peak_bytes"] <= record["planned_cache_peak_bytes"]
    assert record["observed_pinned_peak_bytes"] <= record["planned_pinned_peak_bytes"]
    assert record["dirty_writeback_count"] == 1
    assert record["request_hash"] == request.request_hash
    assert record["plan_hash"] == plan.plan_hash
    assert record["profile_hash"] == request.transfer_profile.profile_hash
    assert not {"keys", "slices", "masks", "tensor_values"}.intersection(record)
    store.close()
    runtime.close()


def test_disabled_profiling_imports_no_payload_or_timing_helpers(monkeypatch):
    from renormalizer.utils.log import DEBUG, init_log

    runtime = _runtime()
    request, plan, store, _, _ = _active_case(runtime, store_id="unprofiled-active")
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    init_log(DEBUG)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"time", "renormalizer.utils.profiling"}:
            raise AssertionError("disabled profiling imported a cost helper")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with provider.open_working_set(request, plan, store, receipt) as working_set:
        with working_set.acquire(_block_request(working_set)):
            pass

    store.close()
    runtime.close()


@pytest.mark.skipif(
    not _has_one_visible_gpu(),
    reason="requires exactly one visible GPU",
)
def test_real_one_gpu_transfer_reuses_cache_commits_dirty_and_cleans_up():
    cupy = pytest.importorskip("cupy")
    runtime = _runtime(backend_name="cupy")
    request, plan, store, _, center = _active_case(runtime, store_id="real-one-gpu")
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)

    try:
        with provider.open_working_set(request, plan, store, receipt) as working_set:
            with working_set.acquire(_block_request(working_set)) as first:
                first_ptr = first.bindings.arrays["input_0"].data.ptr
                assert isinstance(first.bindings.arrays["input_0"], cupy.ndarray)
            with working_set.acquire(_block_request(working_set)) as second:
                assert second.bindings.arrays["input_0"].data.ptr == first_ptr
            working_set.mark_dirty("output", cupy.arange(4, dtype=cupy.float64))

        assert provider.metrics.h2d_count == 1
        assert store.ref("output").version == center.version + 1
        np.testing.assert_array_equal(store.read(store.ref("output")), np.arange(4.0))
        assert provider.resource_state()["reserved_cache_bytes"] == 0
    finally:
        if not store.closed:
            store.close()
        runtime.close()
        cupy.cuda.Stream.null.synchronize()

    assert runtime.resource_state() == {
        "active_leases": 0,
        "cache_bytes": 0,
        "cache_refs": 0,
        "pinned_bytes": 0,
        "stream_count": 0,
        "event_count": 0,
    }


@pytest.mark.skipif(
    not _has_one_visible_gpu(),
    reason="requires exactly one visible GPU",
)
def test_real_one_gpu_pageable_fallback_transfers_and_cleans_up():
    cupy = pytest.importorskip("cupy")
    runtime = _runtime(backend_name="cupy")
    request, plan, store, _, _ = _active_case(
        runtime, store_id="real-pageable-fallback"
    )
    receipt = runtime.preflight_residency(request, plan)

    def pool_factory(capacity_bytes, **kwargs):
        def fail_pinned(nbytes):
            raise MemoryError("injected real pageable fallback")

        return PinnedBufferPool(
            capacity_bytes,
            pinned_allocator=fail_pinned,
            pageable_allocator=lambda nbytes: np.empty(nbytes, dtype=np.uint8),
        )

    provider = _provider(runtime, request, pool_factory=pool_factory)
    try:
        with provider.open_working_set(request, plan, store, receipt) as working_set:
            assert working_set.pool._slot.pinned is False
            with working_set.acquire(_block_request(working_set)) as child:
                cupy.testing.assert_array_equal(
                    child.bindings.arrays["input_0"],
                    cupy.asarray(np.arange(16.0).reshape(4, 4)),
                )
            working_set.mark_dirty("output", cupy.arange(4, dtype=cupy.float64))

        assert provider.metrics.pageable_fallback_count == 1
        assert provider.metrics.pageable_fallback_bytes == (
            request.transfer_profile.rank_staging_bytes[0]
        )
    finally:
        if not store.closed:
            store.close()
        runtime.close()
        cupy.cuda.Stream.null.synchronize()

    assert runtime.resource_state() == {
        "active_leases": 0,
        "cache_bytes": 0,
        "cache_refs": 0,
        "pinned_bytes": 0,
        "stream_count": 0,
        "event_count": 0,
    }


@pytest.mark.skipif(
    not _has_one_visible_gpu(),
    reason="requires exactly one visible GPU",
)
def test_real_one_gpu_f_order_h2d_uses_bounded_pinned_staging(monkeypatch):
    cupy = pytest.importorskip("cupy")
    runtime = _runtime(backend_name="cupy")
    source = np.arange(12, dtype=np.float64).reshape((3, 4), order="F")
    store = HostTensorStore(store_id="real-f-order-h2d")
    ref = store.put("input", source)

    def allocate_pinned(nbytes):
        memory = cupy.cuda.PinnedMemory(nbytes)
        pointer = cupy.cuda.PinnedMemoryPointer(memory, 0)
        return np.frombuffer(pointer, dtype=np.uint8, count=nbytes)

    pool = PinnedBufferPool(source.nbytes, pinned_allocator=allocate_pinned)
    scheduler = TransferScheduler(
        store,
        runtime.backend,
        pool,
        profile_enabled=True,
    )
    destination = cupy.empty(source.shape, dtype=source.dtype, order="F")
    raw_ref = weakref.ref(pool._slot._array)
    observed = []
    copy_into = store.copy_into

    def observe_copy(source_ref, staging, local_slice=None):
        observed.append(
            (
                bool(staging.flags.f_contiguous),
                bool(staging.flags.c_contiguous),
                np.shares_memory(staging, pool._slot._array),
                staging.__array_interface__["data"][0],
            )
        )
        return copy_into(source_ref, staging, local_slice)

    def forbidden_conversion(*args, **kwargs):
        pytest.fail("F-order H2D created a hidden tensor-sized NumPy conversion")

    monkeypatch.setattr(store, "copy_into", observe_copy)
    monkeypatch.setattr(np, "asfortranarray", forbidden_conversion)
    monkeypatch.setattr(np, "ascontiguousarray", forbidden_conversion)
    try:
        with pool.checkout(source.nbytes) as slot:
            assert slot.pinned is True
            ticket = scheduler.stage_h2d(ref, destination, slot)
        ticket.wait()
        scheduler.reap_completed()
        pool.reap_completed()

        cupy.testing.assert_array_equal(destination, cupy.asarray(source, order="F"))
        assert bool(destination.flags.f_contiguous) is True
        assert observed == [
            (
                True,
                False,
                True,
                pool._slot._array.__array_interface__["data"][0],
            )
        ]
        assert pool.capacity_bytes == source.nbytes
        assert pool.allocated_bytes == source.nbytes
        assert pool.checked_out_bytes == 0
        assert pool.pending_bytes == 0
        assert scheduler.pending_ticket_count == 0
        assert scheduler.retained_event_count == 0
        assert ticket.elapsed_s >= 0.0
    finally:
        scheduler.close()
        pool.close()
        store.close()
        runtime.close()
        cupy.cuda.Stream.null.synchronize()
    gc.collect()
    assert raw_ref() is None


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2",
    reason="requires the exact two-rank torchrun command",
)
def test_real_two_rank_h2d_failed_drain_aborts_without_post_terminal_access(
    monkeypatch,
):
    cupy = pytest.importorskip("cupy")
    (
        runtime,
        store,
        working_set,
        operator,
        local_vector,
    ) = _real_two_rank_active_operator_case("real-two-rank-h2d-fatal")
    rank = runtime.rank
    scheduler = working_set.scheduler
    status_calls = []
    forbidden = []
    original_status = operator._allreduce_active_status

    def guarded_status(call, local_error):
        if runtime._terminal_error is not None:
            forbidden.append("status")
        status_calls.append(len(status_calls))
        return original_status(call, local_error)

    monkeypatch.setattr(operator, "_allreduce_active_status", guarded_status)
    original_execute = runtime.backend._execute_plan_with_scope

    def guarded_execute(*args, **kwargs):
        if runtime._terminal_error is not None:
            forbidden.append("backend")
        return original_execute(*args, **kwargs)

    monkeypatch.setattr(runtime.backend, "_execute_plan_with_scope", guarded_execute)
    enqueue_error = RuntimeError("injected real H2D enqueue failure")
    drain_error = RuntimeError("injected real H2D drain failure")
    if rank == 0:
        original_copy_h2d = scheduler._copy_h2d

        def enqueue_then_raise(destination, staging):
            owner = next(
                retained
                for retained in scheduler._owners.values()
                if retained.kind == "h2d"
            )

            def fail_drain():
                raise drain_error

            owner._drainer = fail_drain
            original_copy_h2d(destination, staging)
            raise enqueue_error

        monkeypatch.setattr(scheduler, "_copy_h2d", enqueue_then_raise)

    caught = None
    try:
        operator(local_vector)
    except BaseException as error:
        caught = error
    assert caught is not None
    assert runtime.terminal_poisoned is True
    assert runtime.collective._fatal_abort_started is True
    assert runtime.collective._fatal_abort_completed is True, (
        runtime.rank,
        runtime.collective._fatal_abort_event.is_set(),
        runtime.collective._fatal_abort_error,
        runtime.collective._fatal_error,
        caught,
        status_calls,
        runtime.collective._fatal_secondary_errors,
    )
    assert runtime.collective._fatal_abort_error is None
    assert forbidden == []
    if rank == 0:
        assert caught is enqueue_error
        assert len(status_calls) == 2
        owners = runtime._terminal_quarantine.owners
        assert {owner.kind for owner in owners} >= {"h2d", "compute"}
        assert all(owner.state == "quarantined" for owner in owners)
        h2d_owner = next(owner for owner in owners if owner.kind == "h2d")
        assert h2d_owner.error is enqueue_error
        assert h2d_owner.secondary_errors == (drain_error,)
    before_retry = len(status_calls)
    with pytest.raises(RuntimeError, match="poisoned"):
        operator(local_vector)
    assert len(status_calls) == before_retry

    terminal = runtime._terminal_error
    try:
        runtime.close()
    except BaseException as error:
        assert error is terminal
    store.close()
    assert runtime._closed is True


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2",
    reason="requires the exact two-rank torchrun command",
)
def test_real_two_rank_failed_compute_drain_runtime_fatal_reserves_before_close(
    monkeypatch,
):
    (
        runtime,
        store,
        working_set,
        _,
        local_vector,
    ) = _real_two_rank_active_operator_case("real-two-rank-compute-drain-close")
    rank = runtime.rank
    collective = runtime.collective
    control = collective._bootstrap_store_proxy
    backend = collective._backend
    scheduler = working_set.scheduler
    paused_key = "renormalizer.test.review19.failed_drain.paused"
    closing_prefix = "renormalizer.test.review19.failed_drain.closing."
    if rank == 0:
        control[paused_key] = 0
        for peer in range(2):
            control["{}{}".format(closing_prefix, peer)] = 0
    control.barrier()

    callbacks = []
    original_handler = runtime._enter_communicator_fatal

    def record_callback(error):
        callbacks.append(error)
        return original_handler(error)

    collective._install_fatal_handler(record_callback)
    abort_calls = []
    original_abort = collective._abort_local_communicator

    def record_abort():
        abort_calls.append(1)
        return original_abort()

    monkeypatch.setattr(collective, "_abort_local_communicator", record_abort)
    hard_exits = []

    def hard_exit():
        hard_exits.append(1)
        raise AssertionError("unexpected communicator hard exit")

    monkeypatch.setattr(collective, "_fatal_hard_exit", hard_exit)
    stop_snapshots = []
    original_stop = backend.stop

    def record_stop():
        stop_snapshots.append(
            {
                "close_consumed": tuple(
                    int(control[collective._close_consumed_key(peer)])
                    for peer in range(2)
                ),
                "fatal": tuple(
                    int(control[collective._fatal_key(peer)]) for peer in range(2)
                ),
                "ack": tuple(
                    int(control[collective._fatal_ack_key(peer)]) for peer in range(2)
                ),
                "admitted": collective._admitted_operations,
                "publications": collective._fatal_publications,
            }
        )
        return original_stop()

    monkeypatch.setattr(backend, "stop", record_stop)

    primary = RuntimeError("injected real close-race compute failure")
    drain_error = RuntimeError("injected real close-race compute drain failure")
    failed_owner = None
    if rank == 0:
        failed_owner = scheduler.begin_compute(arrays=(local_vector,)).owner
        failed_owner._drainer = lambda: (_ for _ in ()).throw(drain_error)

    handoff_entered = threading.Event()
    release_handoff = threading.Event()
    if rank == 0:
        original_publish = collective._publish_communicator_fatal

        def pause_runtime_handoff(error, **kwargs):
            handoff_entered.set()
            assert release_handoff.wait(10.0)
            return original_publish(error, **kwargs)

        monkeypatch.setattr(
            collective, "_publish_communicator_fatal", pause_runtime_handoff
        )

    operation_errors = []
    close_errors = []

    def run_operation():
        try:
            failed_owner.fail(primary)
        except BaseException as error:
            operation_errors.append(error)

    def close_runtime():
        try:
            runtime.close()
        except BaseException as error:
            close_errors.append(error)

    operation = (
        threading.Thread(target=run_operation, daemon=True) if rank == 0 else None
    )
    closer = threading.Thread(target=close_runtime, daemon=True)
    state_at_visibility = None
    close_joined_reservation = None
    try:
        if rank == 0:
            operation.start()
            assert handoff_entered.wait(10.0)
            state_at_visibility = (
                collective._fatal_publications,
                collective._fatal_abort_started,
                int(control[collective._fatal_key(0)]),
            )
            control[paused_key] = 1
        else:
            deadline = time.monotonic() + 10.0
            while int(control[paused_key]) != 1:
                assert time.monotonic() < deadline
                time.sleep(0.001)

        closer.start()
        deadline = time.monotonic() + 10.0
        while not collective._closing:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        control["{}{}".format(closing_prefix, rank)] = 1
        deadline = time.monotonic() + 10.0
        while not all(
            int(control["{}{}".format(closing_prefix, peer)]) == 1 for peer in range(2)
        ):
            assert time.monotonic() < deadline
            time.sleep(0.001)
        closer.join(0.05)
        close_joined_reservation = closer.is_alive()

        release_handoff.set()
        if operation is not None:
            operation.join(20.0)
        closer.join(20.0)
    finally:
        release_handoff.set()
        if operation is not None and operation.ident is not None:
            operation.join(20.0)
        if closer.ident is not None:
            closer.join(20.0)
        if not runtime._closed:
            try:
                runtime.close()
            except BaseException:
                pass
        store.close()

    if rank == 0:
        assert state_at_visibility == (1, True, 1)
    else:
        assert state_at_visibility is None
    assert close_joined_reservation is True
    assert operation is None or not operation.is_alive()
    assert not closer.is_alive()
    terminal = runtime._terminal_error
    assert terminal is not None
    assert close_errors == [terminal]
    if rank == 0:
        assert terminal is primary
        assert operation_errors == [primary]
        assert callbacks == []
        assert failed_owner in runtime._terminal_quarantine.owners
        assert failed_owner.error is primary
        assert failed_owner.secondary_errors == (drain_error,)
    else:
        assert operation_errors == []
        assert len(callbacks) == 1
        assert callbacks[0].origin_rank == 0
    assert abort_calls == [1]
    assert hard_exits == []
    assert collective._admitted_operations == 0
    assert collective._fatal_publications == 0
    assert collective._fatal_abort_started is True
    assert collective._fatal_abort_completed is True
    assert collective._closed is True
    assert stop_snapshots
    assert stop_snapshots[0]["fatal"] == (1, 1)
    assert stop_snapshots[0]["ack"] == (1, 1)
    assert stop_snapshots[0]["admitted"] == 0
    assert stop_snapshots[0]["publications"] == 0
    if rank == 0:
        assert stop_snapshots[0]["close_consumed"] == (1, 1)


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2",
    reason="requires the exact two-rank torchrun command",
)
@pytest.mark.parametrize("failure_operation", ("query", "wait"))
@pytest.mark.parametrize("drain_fails", (False, True))
def test_real_two_rank_prior_owner_failure_reaches_p0_or_aborts(
    failure_operation,
    drain_fails,
):
    cupy = pytest.importorskip("cupy")
    (
        runtime,
        store,
        working_set,
        operator,
        local_vector,
    ) = _real_two_rank_active_operator_case(
        "real-two-rank-prior-{}-{}".format(failure_operation, drain_fails)
    )
    operator(local_vector)
    prior_owner = working_set._status_workspace.borrower
    assert prior_owner is not None
    primary = RuntimeError("injected real prior {} failure".format(failure_operation))
    drain_error = RuntimeError("injected real prior owner drain failure")

    class EventWrapper:
        def __init__(self, event):
            self.event = event
            self.synchronize_calls = 0

        def query(self):
            if failure_operation == "query":
                raise primary
            return False

        def synchronize(self):
            self.synchronize_calls += 1
            if failure_operation == "wait" and self.synchronize_calls == 1:
                raise primary
            self.event.synchronize()

    if runtime.rank == 0:
        wrapped_event = EventWrapper(prior_owner.completion_event)
        prior_owner._completion_event = wrapped_event
        prior_owner._events = [wrapped_event]
        if drain_fails:
            prior_owner._drainer = lambda: (_ for _ in ()).throw(drain_error)

    status_calls = []
    original_status = operator._allreduce_active_status

    def record_status(call, local_error):
        status_calls.append(len(status_calls))
        return original_status(call, local_error)

    operator._allreduce_active_status = record_status
    caught = None
    try:
        operator(local_vector)
    except BaseException as error:
        caught = error
    assert caught is not None
    if drain_fails:
        assert runtime.terminal_poisoned is True
        assert runtime.collective._fatal_abort_started is True
        assert runtime.collective._fatal_abort_completed is True
        assert len(status_calls) <= 1
        if runtime.rank == 0:
            assert caught is primary
            assert prior_owner.state == "quarantined"
            assert prior_owner.error is primary
            assert prior_owner.secondary_errors == (drain_error,)
        terminal = runtime._terminal_error
        try:
            runtime.close()
        except BaseException as error:
            assert error is terminal
    else:
        assert status_calls == [0]
        if runtime.rank == 0:
            assert caught is primary
        else:
            assert isinstance(caught, ValueError)
            assert str(caught) == "distributed call-ready preflight failed"
        assert runtime._terminal_error is None
        assert prior_owner.state == "detached"
        result = operator(local_vector)
        assert result.shape == local_vector.shape
        runtime.close()
    store.close()


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2",
    reason="requires the exact two-rank torchrun command",
)
@pytest.mark.parametrize(
    "explicit_reap", (False, True), ids=("borrower-live", "after-explicit-reap")
)
def test_real_two_rank_peer_fatal_between_calls_quarantines_without_cuda(
    monkeypatch, explicit_reap
):
    (
        runtime,
        store,
        working_set,
        operator,
        local_vector,
    ) = _real_two_rank_active_operator_case("real-two-rank-between-call-fatal")
    operator(local_vector)
    collective = runtime.collective
    control = collective._bootstrap_store_proxy
    owner = working_set._status_workspace.borrower
    assert owner is not None
    assert working_set._active_operator_owner is None
    status_pair = (
        working_set._status_workspace.device_status,
        working_set._status_workspace.host_status,
    )
    assert all(
        any(retained is status for retained in owner.arrays) for status in status_pair
    )
    status_records = tuple(allocation_record(status) for status in status_pair)
    assert len({record.identity for record in status_records}) == 2
    assert all(record.capacity_bytes == 4 for record in status_records)
    if explicit_reap:
        working_set.reap_completed()
        assert owner.state == "detached"
        assert working_set._status_workspace.borrower is None

    forbidden = []
    completion = owner.completion_event

    class GuardedEvent:
        def query(self):
            if runtime._terminal_error is not None:
                forbidden.append("event.query")
            return completion.query()

        def synchronize(self):
            if runtime._terminal_error is not None:
                forbidden.append("event.synchronize")
            completion.synchronize()

    scheduler = working_set.scheduler
    if completion is not None:
        guarded_event = GuardedEvent()
        owner._completion_event = guarded_event
        owner._events = [
            guarded_event if event is completion else event for event in owner._events
        ]
        scheduler._events = [
            guarded_event if event is completion else event
            for event in scheduler._events
        ]
        event_owner = scheduler._event_owners.pop(id(completion), None)
        if event_owner is not None:
            scheduler._event_owners[id(guarded_event)] = event_owner

    cache = working_set._provider.cache
    pool = working_set.pool
    guarded_resources = (
        ("cache", cache, ("close", "wait_for_pending", "reap_completed")),
        ("scheduler", scheduler, ("close", "complete_all", "reap_completed")),
        ("pool", pool, ("close", "reap_completed")),
    )
    for resource_name, resource, operation_names in guarded_resources:
        for name in operation_names:
            original = getattr(resource, name)

            def guarded_operation(
                *args,
                resource_name=resource_name,
                name=name,
                original=original,
                **kwargs,
            ):
                if runtime._terminal_error is not None:
                    forbidden.append("{}.{}".format(resource_name, name))
                return original(*args, **kwargs)

            monkeypatch.setattr(resource, name, guarded_operation)

    records = {}
    for record in (
        *(status_records if explicit_reap else owner.allocations),
        *cache.allocation_records,
        *pool.allocation_records,
    ):
        retained = records.get(record.identity)
        if retained is None or record.capacity_bytes > retained.capacity_bytes:
            records[record.identity] = record
    cache_records = {record.identity: record for record in cache.allocation_records}
    pool_records = {record.identity: record for record in pool.allocation_records}
    expected_arrays = tuple(owner.arrays)
    expected_streams = {id(stream) for stream in scheduler.retained_streams}
    expected_events = {id(event) for event in scheduler.retained_events}

    ready_prefix = "renormalizer.test.review14.between_call.ready."
    if runtime.rank == 0:
        for rank in range(2):
            control["{}{}".format(ready_prefix, rank)] = 0
    control.barrier()
    control["{}{}".format(ready_prefix, runtime.rank)] = 1
    control.barrier()
    assert all(
        int(control["{}{}".format(ready_prefix, rank)]) == 1 for rank in range(2)
    )

    primary = RuntimeError("injected real between-call communicator failure")
    if runtime.rank == 0:
        assert runtime._enter_communicator_fatal(primary) is primary
    else:
        deadline = time.monotonic() + 10.0
        while runtime._terminal_error is None:
            assert time.monotonic() < deadline
            time.sleep(0.001)

    terminal = runtime._terminal_error
    assert terminal is not None
    if explicit_reap:
        assert owner.state == "detached"
        assert owner not in runtime._terminal_quarantine.owners
        assert working_set._status_workspace.borrower is None
    else:
        assert owner.state == "quarantined"
        assert owner in runtime._terminal_quarantine.owners
        assert tuple(owner.arrays) == expected_arrays
        assert all(
            any(retained is status for retained in owner.arrays)
            for status in status_pair
        )
    if not explicit_reap:
        assert cache.poisoned is True
        assert scheduler.poisoned is True
    assert collective._fatal_abort_started is True
    assert collective._fatal_abort_event.wait(10.0)
    assert collective._fatal_abort_completed is True
    deadline = time.monotonic() + 10.0
    while not collective._fatal_protocol_completed:
        assert time.monotonic() < deadline
        time.sleep(0.001)

    try:
        runtime.close()
    except BaseException as error:
        assert error is terminal
    assert forbidden == []
    retained = {
        record.identity: record for record in runtime._terminal_quarantine.allocations
    }
    assert set(retained) == set(records)
    assert all(
        retained[identity].capacity_bytes == record.capacity_bytes
        for identity, record in records.items()
    )
    for status_record in status_records:
        assert (
            sum(
                record.identity == status_record.identity
                for record in runtime._terminal_quarantine.allocations
            )
            == 1
        )
    state = runtime.resource_state()
    assert state["cache_bytes"] == sum(
        record.capacity_bytes for record in cache_records.values()
    )
    assert state["pinned_bytes"] == sum(
        record.capacity_bytes for record in pool_records.values()
    )
    assert state["stream_count"] == len(expected_streams)
    assert state["event_count"] == len(expected_events)
    assert state["quarantined_bytes"] == sum(
        record.capacity_bytes for record in records.values()
    )
    store.close()


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2",
    reason="requires the exact two-rank torchrun command",
)
@pytest.mark.parametrize("origin_rank", (0, 1))
def test_real_two_rank_broadcast_enqueue_then_raise_aborts_before_execution_or_e(
    monkeypatch, origin_rank
):
    cupy = pytest.importorskip("cupy")
    (
        runtime,
        store,
        working_set,
        operator,
        local_vector,
    ) = _real_two_rank_active_operator_case(
        "real-two-rank-broadcast-fatal-origin-{}".format(origin_rank)
    )
    collective = runtime.collective
    raw_backend = collective._backend
    backend_broadcast = raw_backend.broadcast
    backend_broadcast_calls = []
    injected = RuntimeError("injected real post-enqueue broadcast failure")

    def broadcast_then_maybe_raise(*args, **kwargs):
        backend_broadcast_calls.append(1)
        result = backend_broadcast(*args, **kwargs)
        if runtime.rank == origin_rank:
            raise injected
        return result

    monkeypatch.setattr(raw_backend, "broadcast", broadcast_then_maybe_raise)
    status_calls = []
    original_status = operator._allreduce_active_status

    def record_status(call, local_error):
        status_calls.append(len(status_calls))
        return original_status(call, local_error)

    monkeypatch.setattr(operator, "_allreduce_active_status", record_status)
    caught = None
    try:
        operator(local_vector)
    except BaseException as error:
        caught = error
    assert caught is not None
    if runtime.rank == origin_rank:
        assert caught is injected
    assert backend_broadcast_calls == [1]
    assert status_calls == [0, 1, 2]
    assert operator._counters.get("broadcast_calls", 0) == 0
    assert operator._counters.get("execution_calls", 0) == 0
    assert collective._fatal_abort_started is True
    assert collective._fatal_abort_completed is True
    before_retry = (len(status_calls), len(backend_broadcast_calls))
    with pytest.raises(RuntimeError, match="poisoned"):
        operator(local_vector)
    assert (len(status_calls), len(backend_broadcast_calls)) == before_retry

    pool_records = working_set.pool.allocation_records
    cache_records = working_set._provider.cache.allocation_records
    owners = runtime._terminal_quarantine.owners
    expected_records = {}
    for record in (
        *pool_records,
        *cache_records,
        *(record for owner in owners for record in owner.allocations),
    ):
        retained = expected_records.get(record.identity)
        if retained is None or record.capacity_bytes > retained.capacity_bytes:
            expected_records[record.identity] = record
    expected_streams = {
        id(stream)
        for stream in (
            working_set.scheduler._stream,
            *(stream for owner in owners for stream in owner.streams),
        )
        if stream is not None
    }
    expected_events = {
        id(event)
        for event in (
            *working_set.scheduler._events,
            *(event for owner in owners for event in owner.events),
        )
    }
    expected_cache_bytes = sum(
        record.capacity_bytes
        for record in {record.identity: record for record in cache_records}.values()
    )
    expected_pinned_bytes = sum(
        record.capacity_bytes
        for record in {record.identity: record for record in pool_records}.values()
    )
    terminal = runtime._terminal_error
    try:
        runtime.close()
    except BaseException as error:
        assert error is terminal
    state = runtime.resource_state()
    assert state["cache_bytes"] == expected_cache_bytes
    assert state["pinned_bytes"] == expected_pinned_bytes
    assert state["stream_count"] == len(expected_streams)
    assert state["event_count"] == len(expected_events)
    assert state["quarantined_bytes"] == sum(
        record.capacity_bytes for record in expected_records.values()
    )
    assert state["quarantined_stream_count"] == len(expected_streams)
    assert state["quarantined_event_count"] == len(expected_events)
    assert {
        record.identity for record in runtime._terminal_quarantine.allocations
    } == set(expected_records)
    store.close()


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2",
    reason="requires the exact two-rank torchrun command",
)
def test_real_two_rank_asymmetric_private_store_loss_rejects_all_ranks(
    monkeypatch,
):
    from cupyx.distributed._store import TCPStoreProxy

    from renormalizer import set_backend
    from renormalizer.backend.distributed_runtime import (
        create_cupy_distributed_runtime,
    )

    runtime = create_cupy_distributed_runtime(
        expected_world_size=2,
        execution_policy="execution_ir",
        fallback_policy="error",
    )
    store = None
    try:
        set_backend(
            "cupy",
            device="cuda:{}".format(runtime.local_rank),
            precision=64,
            execution_policy="execution_ir",
            fallback_policy="error",
        )
        config = runtime.execution_config(
            residency_policy="active_working_set",
            device_memory_budget_bytes=1 << 30,
            host_memory_budget_bytes=1 << 31,
        )
        request, _, store, _, _ = _active_case(
            runtime, store_id="real-two-rank-asymmetric-store-loss"
        )
        request = replace(
            request,
            device_budget=config.device_budget_resolution,
            host_budget=config.host_budget_resolution,
        )
        plan = ResidencyPlanner().plan(request)
        receipt = runtime.preflight_residency(request, plan)
        collective = runtime.collective
        backend_store = collective._backend._store_proxy
        control_store = collective._bootstrap_store_proxy
        assert type(backend_store) is TCPStoreProxy
        assert type(control_store) is TCPStoreProxy
        assert control_store is not backend_store
        if runtime.rank == 1:
            del collective._backend._store_proxy

        allocation_calls = []

        def forbidden_status_allocation():
            allocation_calls.append(1)
            raise AssertionError("status allocation followed capability rejection")

        monkeypatch.setattr(
            config.provider, "_allocate_status_device", forbidden_status_allocation
        )
        monkeypatch.setattr(
            config.provider, "_allocate_status_host", forbidden_status_allocation
        )
        with pytest.raises(
            RuntimeError, match="collective fatal capability bootstrap failed"
        ):
            config.provider.open_working_set(request, plan, store, receipt)
        assert allocation_calls == []
        assert collective._fatal_control_initialized is False
        assert collective._fatal_abort_started is False
        assert config.provider._active_lease is None
    finally:
        if store is not None and not store.closed:
            store.close()
        runtime.close()


def _real_two_rank_active_operator_case(store_id):
    cupy = pytest.importorskip("cupy")
    from renormalizer import set_backend
    from renormalizer.backend.distributed_runtime import (
        create_cupy_distributed_runtime,
    )

    runtime = create_cupy_distributed_runtime(
        expected_world_size=2,
        execution_policy="execution_ir",
        fallback_policy="error",
    )
    set_backend(
        "cupy",
        device="cuda:{}".format(runtime.local_rank),
        precision=64,
        execution_policy="execution_ir",
        fallback_policy="error",
    )
    config = runtime.execution_config(
        residency_policy="active_working_set",
        device_memory_budget_bytes=1 << 30,
        host_memory_budget_bytes=1 << 31,
    )
    request, plan, store, _, _ = _active_case(runtime, store_id=store_id)
    request = replace(
        request,
        device_budget=config.device_budget_resolution,
        host_budget=config.host_budget_resolution,
    )
    plan = ResidencyPlanner().plan(request)
    receipt = runtime.preflight_residency(request, plan)
    working_set = config.provider.open_working_set(request, plan, store, receipt)
    distributed = request.distributed_plan
    operator = DistributedLocalOperator(
        plan=distributed,
        provider=working_set,
        collective=runtime.collective,
        counters={},
        backend=runtime.backend,
        context=runtime.context,
        source_bindings=ExecutionBindings(
            {"input_0": dict(request.host_refs)["input_0"]}
        ),
        residency_request=request,
        residency_plan=plan,
        residency_receipt=receipt,
        mesh=runtime.mesh,
    )
    local_vector = cupy.ones(
        distributed.input_sharding.local_shape(runtime.rank), dtype=cupy.float64
    )
    return runtime, store, working_set, operator, local_vector


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2",
    reason="requires the exact two-rank torchrun command",
)
def test_real_two_rank_staggered_healthy_close_stops_rank_zero_store_last():
    (
        runtime,
        store,
        _,
        operator,
        local_vector,
    ) = _real_two_rank_active_operator_case("real-two-rank-staggered-close")
    operator(local_vector)
    collective = runtime.collective
    if runtime.rank == 1:
        time.sleep(0.5)
    started = time.monotonic()
    runtime.close()
    elapsed = time.monotonic() - started
    if runtime.rank == 0:
        assert elapsed >= 0.2
    assert collective._closed is True
    assert collective._backend is None
    store.close()


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2",
    reason="requires the exact two-rank torchrun command",
)
def test_real_two_rank_close_ready_joins_unpolled_nonzero_fatal(monkeypatch):
    (
        runtime,
        store,
        _,
        _,
        _,
    ) = _real_two_rank_active_operator_case("real-two-rank-close-fatal-race")
    collective = runtime.collective
    control = collective._bootstrap_store_proxy
    backend = collective._backend
    stop_snapshots = []
    original_stop = backend.stop

    def record_stop():
        stop_snapshots.append(
            tuple(
                int(control[collective._close_consumed_key(rank)]) for rank in range(2)
            )
        )
        return original_stop()

    monkeypatch.setattr(backend, "stop", record_stop)
    release_monitor = threading.Event()
    if runtime.rank == 0:
        monitor = collective._fatal_monitor_thread
        collective._fatal_monitor_stop.set()
        monitor.join(10.0)
        assert not monitor.is_alive()
        collective._fatal_monitor_stop.clear()
        original_monitor = collective._monitor_fatal_records

        def delayed_monitor():
            assert release_monitor.wait(10.0)
            original_monitor()

        monitor = threading.Thread(target=delayed_monitor, daemon=True)
        collective._fatal_monitor_thread = monitor
        monitor.start()

    ready_prefix = "renormalizer.test.review14.close_race.ready."
    published_key = "renormalizer.test.review14.close_race.published"
    if runtime.rank == 0:
        for rank in range(2):
            control["{}{}".format(ready_prefix, rank)] = 0
        control[published_key] = 0
    control.barrier()
    control["{}{}".format(ready_prefix, runtime.rank)] = 1
    control.barrier()
    assert all(
        int(control["{}{}".format(ready_prefix, rank)]) == 1 for rank in range(2)
    )

    primary = RuntimeError("injected real rank-1 close-ready race")
    publish_errors = []

    def publish_fatal():
        try:
            runtime._enter_communicator_fatal(primary)
        except BaseException as error:
            publish_errors.append(error)

    publisher = None
    if runtime.rank == 1:
        publisher = threading.Thread(target=publish_fatal, daemon=True)
        publisher.start()
        deadline = time.monotonic() + 10.0
        while int(control[collective._fatal_key(1)]) != 1:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        control[published_key] = 1
    else:
        deadline = time.monotonic() + 10.0
        while int(control[published_key]) != 1:
            assert time.monotonic() < deadline
            time.sleep(0.001)

    close_errors = []

    def close_runtime():
        try:
            runtime.close()
        except BaseException as error:
            close_errors.append(error)

    closer = threading.Thread(target=close_runtime, daemon=True)
    closer.start()
    if runtime.rank == 0:
        deadline = time.monotonic() + 10.0
        while int(control[collective._fatal_ack_key(0)]) != 1:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert collective._fatal_abort_started is True
        assert collective._fatal_abort_completed is True
        release_monitor.set()
    else:
        publisher.join(10.0)
        assert not publisher.is_alive()
        assert publish_errors == []

    closer.join(10.0)
    assert not closer.is_alive()
    terminal = runtime._terminal_error
    assert terminal is not None
    assert close_errors == [terminal]
    assert collective._fatal_abort_started is True
    assert collective._fatal_abort_completed is True
    assert collective._closed is True
    assert stop_snapshots
    if runtime.rank == 0:
        assert stop_snapshots == [(1, 1)]
    store.close()


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2",
    reason="requires the exact two-rank torchrun command",
)
def test_real_two_rank_close_intent_waits_for_admitted_operation_fatal(monkeypatch):
    (
        runtime,
        store,
        _,
        _,
        _,
    ) = _real_two_rank_active_operator_case("real-two-rank-admitted-close-fatal")
    collective = runtime.collective
    control = collective._bootstrap_store_proxy
    backend = collective._backend
    stop_snapshots = []
    original_stop = backend.stop

    def record_stop():
        stop_snapshots.append(
            tuple(
                int(control[collective._close_consumed_key(rank)]) for rank in range(2)
            )
        )
        return original_stop()

    monkeypatch.setattr(backend, "stop", record_stop)
    intent_key = "renormalizer.test.review15.close_intent"
    rank_zero_started_key = "renormalizer.test.review15.rank_zero_started"
    if runtime.rank == 0:
        control[intent_key] = 0
        control[rank_zero_started_key] = 0
    control.barrier()

    operation_entered = threading.Event()
    release_operation = threading.Event()
    primary = RuntimeError("injected real admitted rank-1 communicator failure")
    operation_errors = []
    if runtime.rank == 1:

        def fail_barrier():
            operation_entered.set()
            assert release_operation.wait(10.0)
            raise primary

        monkeypatch.setattr(backend, "barrier", fail_barrier)

    def run_operation():
        try:
            collective.barrier()
        except BaseException as error:
            operation_errors.append(error)

    operation = None
    if runtime.rank == 1:
        operation = threading.Thread(target=run_operation, daemon=True)
        operation.start()
        assert operation_entered.wait(10.0)

    close_errors = []

    def close_runtime():
        try:
            runtime.close()
        except BaseException as error:
            close_errors.append(error)

    closer = threading.Thread(target=close_runtime, daemon=True)
    if runtime.rank == 1:
        closer.start()
        deadline = time.monotonic() + 10.0
        while not collective._closing:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        control[intent_key] = 1
        deadline = time.monotonic() + 10.0
        while int(control[rank_zero_started_key]) != 1:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        release_operation.set()
    else:
        deadline = time.monotonic() + 10.0
        while int(control[intent_key]) != 1:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        control[rank_zero_started_key] = 1
        closer.start()

    if operation is not None:
        operation.join(10.0)
        assert not operation.is_alive()
        assert operation_errors == [primary]
    closer.join(10.0)
    assert not closer.is_alive()
    terminal = runtime._terminal_error
    assert terminal is not None
    assert close_errors == [terminal]
    assert collective._fatal_abort_started is True
    assert collective._fatal_abort_completed is True
    assert collective._closed is True
    assert stop_snapshots
    if runtime.rank == 0:
        assert stop_snapshots == [(1, 1)]
    store.close()


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2",
    reason="requires the exact two-rank torchrun command",
)
def test_real_two_rank_close_joins_monitor_fatal_for_admitted_peer(monkeypatch):
    cupy = pytest.importorskip("cupy")
    (
        runtime,
        store,
        _,
        _,
        _,
    ) = _real_two_rank_active_operator_case("real-two-rank-both-admitted-close-fatal")
    collective = runtime.collective
    control = collective._bootstrap_store_proxy
    backend = collective._backend

    entered_prefix = "renormalizer.test.review16.entered."
    closing_prefix = "renormalizer.test.review16.closing."
    if runtime.rank == 0:
        for rank in range(2):
            control["{}{}".format(entered_prefix, rank)] = 0
            control["{}{}".format(closing_prefix, rank)] = 0
    control.barrier()

    callbacks = []
    original_handler = runtime._enter_communicator_fatal

    def record_callback(error):
        callbacks.append(error)
        return original_handler(error)

    collective._install_fatal_handler(record_callback)

    abort_calls = []
    original_abort = collective._abort_local_communicator

    def record_abort():
        abort_calls.append(1)
        return original_abort()

    monkeypatch.setattr(collective, "_abort_local_communicator", record_abort)

    hard_exits = []

    def hard_exit():
        hard_exits.append(1)
        raise AssertionError("unexpected communicator hard exit")

    monkeypatch.setattr(collective, "_fatal_hard_exit", hard_exit)

    stop_snapshots = []
    actual_threads = []
    actual_errors = []
    original_stop = backend.stop

    def record_stop():
        for thread in actual_threads:
            thread.join(3.0)
        stop_snapshots.append(
            {
                "close_consumed": tuple(
                    int(control[collective._close_consumed_key(rank)])
                    for rank in range(2)
                ),
                "fatal": tuple(
                    int(control[collective._fatal_key(rank)]) for rank in range(2)
                ),
                "ack": tuple(
                    int(control[collective._fatal_ack_key(rank)]) for rank in range(2)
                ),
                "admitted": collective._admitted_operations,
                "publications": collective._fatal_publications,
                "actual_alive": sum(thread.is_alive() for thread in actual_threads),
            }
        )
        return original_stop()

    monkeypatch.setattr(backend, "stop", record_stop)

    operation_entered = threading.Event()
    release_failure = threading.Event()
    rank_zero_release_timed_out = threading.Event()
    primary = RuntimeError("injected real both-admitted rank-1 communicator failure")
    original_all_reduce = backend.all_reduce

    def controlled_all_reduce(*args, **kwargs):
        actual_called = threading.Event()

        def invoke_actual_all_reduce():
            try:
                with cupy.cuda.Device(runtime.local_rank):
                    actual_called.set()
                    original_all_reduce(*args, **kwargs)
            except BaseException as error:
                actual_errors.append(error)

        actual = threading.Thread(target=invoke_actual_all_reduce, daemon=True)
        actual_threads.append(actual)
        actual.start()
        assert actual_called.wait(10.0)
        operation_entered.set()
        if runtime.rank == 0:
            if not collective._fatal_abort_event.wait(3.0):
                rank_zero_release_timed_out.set()
                raise RuntimeError("rank-0 admitted NCCL operation abort timed out")
        else:
            assert release_failure.wait(10.0)
            raise primary

    monkeypatch.setattr(backend, "all_reduce", controlled_all_reduce)

    operation_errors = []

    def run_operation():
        try:
            with cupy.cuda.Device(runtime.local_rank):
                value = cupy.asarray([runtime.rank + 1], dtype=cupy.float64)
                collective.allreduce(value)
        except BaseException as error:
            operation_errors.append(error)

    close_errors = []

    def close_runtime():
        try:
            runtime.close()
        except BaseException as error:
            close_errors.append(error)

    operation = threading.Thread(target=run_operation, daemon=True)
    closer = threading.Thread(target=close_runtime, daemon=True)
    try:
        operation.start()
        assert operation_entered.wait(10.0)
        assert collective._admitted_operations == 1
        control["{}{}".format(entered_prefix, runtime.rank)] = 1
        deadline = time.monotonic() + 10.0
        while not all(
            int(control["{}{}".format(entered_prefix, rank)]) == 1 for rank in range(2)
        ):
            assert time.monotonic() < deadline
            time.sleep(0.001)

        closer.start()
        deadline = time.monotonic() + 10.0
        while not collective._closing:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        control["{}{}".format(closing_prefix, runtime.rank)] = 1
        deadline = time.monotonic() + 10.0
        while not all(
            int(control["{}{}".format(closing_prefix, rank)]) == 1 for rank in range(2)
        ):
            assert time.monotonic() < deadline
            time.sleep(0.001)
        if runtime.rank == 1:
            release_failure.set()

        operation.join(10.0)
        closer.join(10.0)
    finally:
        release_failure.set()
        if operation.ident is not None:
            operation.join(10.0)
        if closer.ident is not None:
            closer.join(10.0)

    assert not operation.is_alive()
    assert not closer.is_alive()
    assert rank_zero_release_timed_out.is_set() is False
    terminal = runtime._terminal_error
    assert terminal is not None
    if runtime.rank == 0:
        assert operation_errors == []
        assert callbacks[0].origin_rank == 1
    else:
        assert operation_errors == [primary]
        assert callbacks == [primary]
    assert len(callbacks) == 1
    assert close_errors == [terminal]
    assert abort_calls == [1]
    assert hard_exits == []
    assert collective._admitted_operations == 0
    assert collective._fatal_publications == 0
    assert collective._fatal_abort_started is True
    assert collective._fatal_abort_completed is True
    assert collective._closed is True
    assert len(actual_threads) == 1
    assert all(not thread.is_alive() for thread in actual_threads)
    assert stop_snapshots
    assert stop_snapshots[0]["fatal"] == (1, 1)
    assert stop_snapshots[0]["ack"] == (1, 1)
    assert stop_snapshots[0]["admitted"] == 0
    assert stop_snapshots[0]["publications"] == 0
    assert stop_snapshots[0]["actual_alive"] == 0
    if runtime.rank == 0:
        assert stop_snapshots == [
            {
                "close_consumed": (1, 1),
                "fatal": (1, 1),
                "ack": (1, 1),
                "admitted": 0,
                "publications": 0,
                "actual_alive": 0,
            }
        ]
    store.close()


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2",
    reason="requires the exact two-rank torchrun command",
)
@pytest.mark.parametrize(
    "fatal", (os.environ.get("RENORMALIZER_TEST_REVIEW17_FATAL") == "1",)
)
def test_real_two_rank_close_waits_for_complete_active_broadcast_agreement(
    monkeypatch, fatal
):
    cupy = pytest.importorskip("cupy")
    (
        runtime,
        store,
        _,
        _,
        _,
    ) = _real_two_rank_active_operator_case(
        "real-two-rank-active-b-close-{}".format("fatal" if fatal else "healthy")
    )
    collective = runtime.collective
    control = collective._bootstrap_store_proxy
    backend = collective._backend

    entered_prefix = "renormalizer.test.review17.active_b_entered."
    closing_prefix = "renormalizer.test.review17.active_b_closing."
    if runtime.rank == 0:
        for rank in range(2):
            control["{}{}".format(entered_prefix, rank)] = 0
            control["{}{}".format(closing_prefix, rank)] = 0
    control.barrier()

    primary = RuntimeError("injected real active broadcast communicator failure")
    if fatal and runtime.rank == 1:
        original_broadcast = backend.broadcast

        def fail_after_broadcast(*args, **kwargs):
            original_broadcast(*args, **kwargs)
            raise primary

        monkeypatch.setattr(backend, "broadcast", fail_after_broadcast)

    callbacks = []
    original_handler = runtime._enter_communicator_fatal

    def record_callback(error):
        callbacks.append(error)
        return original_handler(error)

    collective._install_fatal_handler(record_callback)

    abort_calls = []
    original_abort = collective._abort_local_communicator

    def record_abort():
        abort_calls.append(1)
        return original_abort()

    monkeypatch.setattr(collective, "_abort_local_communicator", record_abort)

    hard_exits = []

    def hard_exit():
        hard_exits.append(1)
        raise AssertionError("unexpected communicator hard exit")

    monkeypatch.setattr(collective, "_fatal_hard_exit", hard_exit)

    original_set = collective._fatal_store_set
    release_agreement = threading.Event()

    def pause_active_record(key, value):
        if key == collective._active_b_key(runtime.rank):
            sequence, _ = collective._decode_active_b(value)
            if int(sequence) == 1:
                control["{}{}".format(entered_prefix, runtime.rank)] = 1
                deadline = time.monotonic() + 10.0
                while not all(
                    int(control["{}{}".format(entered_prefix, rank)]) == 1
                    for rank in range(2)
                ):
                    assert time.monotonic() < deadline
                    time.sleep(0.001)
                assert release_agreement.wait(10.0)
        return original_set(key, value)

    monkeypatch.setattr(collective, "_fatal_store_set", pause_active_record)

    stop_snapshots = []
    original_stop = backend.stop

    def record_stop():
        stop_snapshots.append(
            {
                "close_consumed": tuple(
                    int(control[collective._close_consumed_key(rank)])
                    for rank in range(2)
                ),
                "admitted": collective._admitted_operations,
                "publications": collective._fatal_publications,
            }
        )
        return original_stop()

    monkeypatch.setattr(backend, "stop", record_stop)

    operation_errors = []

    def run_active_broadcast():
        admission = getattr(collective, "_active_broadcast_admission", None)
        boundary = (
            nullcontext((collective.broadcast, collective._agree_active_broadcast))
            if admission is None
            else admission()
        )
        try:
            with cupy.cuda.Device(runtime.local_rank):
                value = cupy.asarray([runtime.rank + 1], dtype=cupy.float64)
                with boundary as (broadcast, agree):
                    broadcast_error = None
                    try:
                        broadcast(value, root=0)
                    except BaseException as error:
                        broadcast_error = error
                    try:
                        agree(broadcast_error is not None)
                    except BaseException:
                        if broadcast_error is not None:
                            raise broadcast_error
                        raise
                    if broadcast_error is not None:
                        raise broadcast_error
        except BaseException as error:
            operation_errors.append(error)

    close_errors = []

    def close_runtime():
        try:
            runtime.close()
        except BaseException as error:
            close_errors.append(error)

    operation = threading.Thread(target=run_active_broadcast, daemon=True)
    closer = threading.Thread(target=close_runtime, daemon=True)
    try:
        operation.start()
        deadline = time.monotonic() + 10.0
        while int(control["{}{}".format(entered_prefix, runtime.rank)]) != 1:
            assert time.monotonic() < deadline
            time.sleep(0.001)

        closer.start()
        while not collective._closing:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        control["{}{}".format(closing_prefix, runtime.rank)] = 1
        while not all(
            int(control["{}{}".format(closing_prefix, rank)]) == 1 for rank in range(2)
        ):
            assert time.monotonic() < deadline
            time.sleep(0.001)

        closer.join(0.1)
        assert closer.is_alive()
        assert stop_snapshots == []
        assert collective._admitted_operations == 1
        release_agreement.set()

        operation.join(10.0)
        closer.join(10.0)
    finally:
        release_agreement.set()
        if operation.ident is not None:
            operation.join(10.0)
        if closer.ident is not None:
            closer.join(10.0)

    assert not operation.is_alive()
    assert not closer.is_alive()
    if fatal:
        assert len(operation_errors) == 1
        if runtime.rank == 1:
            assert operation_errors == [primary]
            assert callbacks == [primary]
        else:
            assert callbacks[0].origin_rank == 1
        assert len(callbacks) == 1
        assert len(close_errors) == 1
        assert close_errors[0] is runtime._terminal_error
        assert abort_calls == [1]
    else:
        assert operation_errors == []
        assert callbacks == []
        assert close_errors == []
        assert abort_calls == []
    assert hard_exits == []
    assert collective._admitted_operations == 0
    assert collective._fatal_publications == 0
    assert collective._closed is True
    assert stop_snapshots
    assert stop_snapshots[0]["admitted"] == 0
    assert stop_snapshots[0]["publications"] == 0
    if runtime.rank == 0:
        assert stop_snapshots == [
            {
                "close_consumed": (1, 1),
                "admitted": 0,
                "publications": 0,
            }
        ]
    store.close()


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2",
    reason="requires the exact two-rank torchrun command",
)
def test_real_two_rank_outer_lease_reuses_static_shards_and_cleans_up():
    cupy = pytest.importorskip("cupy")
    from renormalizer import set_backend
    from renormalizer.backend.distributed_runtime import (
        create_cupy_distributed_runtime,
    )

    runtime = create_cupy_distributed_runtime(
        expected_world_size=2,
        execution_policy="execution_ir",
        fallback_policy="error",
    )
    store = None
    try:
        set_backend(
            "cupy",
            device="cuda:{}".format(runtime.local_rank),
            precision=64,
            execution_policy="execution_ir",
            fallback_policy="error",
        )
        config = runtime.execution_config(
            residency_policy="active_working_set",
            device_memory_budget_bytes=1 << 30,
            host_memory_budget_bytes=1 << 31,
        )
        request, plan, store, _, center = _active_case(
            runtime, store_id="real-two-rank-active"
        )
        request = replace(
            request,
            device_budget=config.device_budget_resolution,
            host_budget=config.host_budget_resolution,
        )
        plan = ResidencyPlanner().plan(request)

        with active_working_set_execution(config, request, plan, store) as borrowed:
            working_set = borrowed.provider
            first_pointers = {}
            for _ in range(2):
                for source_rank in range(runtime.world_size):
                    with working_set.acquire(
                        _block_request(working_set, source_rank)
                    ) as child:
                        local_matrix = child.bindings.arrays["input_0"]
                        pointer = local_matrix.data.ptr
                        previous = first_pointers.setdefault(source_rank, pointer)
                        assert pointer == previous
                        block = request.distributed_plan.block_plan(
                            runtime.rank, source_rank
                        )
                        cupy.testing.assert_array_equal(
                            local_matrix,
                            cupy.asarray(
                                np.arange(16.0).reshape(4, 4)[
                                    block.operand_slices["input_0"]
                                ]
                            ),
                        )
            counters = {}
            operator = DistributedLocalOperator(
                plan=request.distributed_plan,
                provider=working_set,
                collective=runtime.collective,
                counters=counters,
                backend=runtime.backend,
                context=runtime.context,
                source_bindings=ExecutionBindings(
                    {"input_0": dict(request.host_refs)["input_0"]}
                ),
                residency_request=request,
                residency_plan=plan,
                residency_receipt=borrowed.residency_receipt,
                mesh=runtime.mesh,
            )
            local_vector = cupy.ones(
                request.distributed_plan.input_sharding.local_shape(runtime.rank),
                dtype=cupy.float64,
            )
            local_result = operator(local_vector)
            expected_result = np.arange(16.0).reshape(4, 4).dot(np.ones(4))
            output_slice = request.distributed_plan.output_sharding.local_slices[
                runtime.rank
            ]
            cupy.testing.assert_array_equal(
                local_result, cupy.asarray(expected_result[output_slice])
            )
            assert counters["allreduce_calls"] == 5
            assert counters["broadcast_calls"] == 2
            assert counters["execution_calls"] == 2
            other_device = (runtime.local_rank + 1) % runtime.world_size
            with cupy.cuda.Device(other_device):
                wrong_device_dirty = cupy.arange(4, dtype=cupy.float64)
            with pytest.raises(ValueError, match="selected CUDA device"):
                working_set.mark_dirty("output", wrong_device_dirty)
            del wrong_device_dirty
            working_set.mark_dirty(
                "output", cupy.full((4,), runtime.rank + 1, dtype=cupy.float64)
            )
        metrics = config.provider.metrics
        expected_h2d = len(first_pointers)
        assert metrics.h2d_count == expected_h2d
        assert metrics.full_replica is False
        assert store.ref("output").version == center.version + 1
        state = config.provider.resource_state()
        assert state["active_leases"] == 0
        assert state["reserved_cache_bytes"] == 0
        assert state["checked_out_pinned_bytes"] == 0
        assert state["retained_cache_bytes"] <= state["hard_cache_capacity_bytes"]
        assert state["authorized_cache_bytes"] == (
            plan.rank_estimates[runtime.rank].current_static_bytes
            + plan.rank_estimates[runtime.rank].prefetched_static_bytes
        )
        runtime.barrier()
    finally:
        if store is not None and not store.closed:
            store.close()
        runtime.close()
        cupy.cuda.Stream.null.synchronize()

    assert runtime.resource_state() == {
        "active_leases": 0,
        "cache_bytes": 0,
        "cache_refs": 0,
        "pinned_bytes": 0,
        "stream_count": 0,
        "event_count": 0,
    }
