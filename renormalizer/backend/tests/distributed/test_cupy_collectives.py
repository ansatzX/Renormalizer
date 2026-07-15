from contextlib import nullcontext
import gc
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from renormalizer.backend._distributed.context import DistributedContext
from renormalizer.backend._distributed.terminal import (
    _FatalTransition,
    _TERMINAL_TIMEOUT_S,
    _TerminalPhase,
)


cupy = pytest.importorskip("cupy")


class FakeNcclBackend:
    def __init__(self, size, rank):
        self.size = size
        self.rank = rank
        self.calls = []
        self.stop_calls = 0

    def barrier(self):
        self.calls.append(("barrier",))

    def broadcast(self, array, root=0, stream=None):
        self.calls.append(("broadcast", root, stream))
        array[...] = root + 10

    def all_reduce(self, in_array, out_array, op="sum", stream=None):
        self.calls.append(("all_reduce", op, stream))
        out_array[...] = in_array * self.size

    def all_gather(self, in_array, out_array, count, stream=None):
        self.calls.append(("all_gather", count, stream))
        for rank in range(self.size):
            out_array[rank * count : (rank + 1) * count] = in_array + rank * 100

    def reduce_scatter(self, in_array, out_array, count, op="sum", stream=None):
        self.calls.append(("reduce_scatter", count, op, stream))
        start = self.rank * count
        out_array[...] = in_array[start : start + count] * self.size

    def stop(self):
        self.stop_calls += 1

    def close(self):
        self.stop()


class FailingOnceNcclBackend(FakeNcclBackend):
    def stop(self):
        self.stop_calls += 1
        if self.stop_calls == 1:
            raise RuntimeError("injected stop failure")


class _CpuDevice:
    def use(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _CpuCudaRuntime:
    @staticmethod
    def getDeviceCount():
        return 2


class _CpuCuda:
    runtime = _CpuCudaRuntime()

    @staticmethod
    def Device(index):
        return _CpuDevice()

    @staticmethod
    def get_current_stream():
        return None


class _CpuOnlyCupy:
    cuda = _CpuCuda()

    ndarray = np.ndarray
    ascontiguousarray = staticmethod(np.ascontiguousarray)
    copyto = staticmethod(np.copyto)
    empty_like = staticmethod(np.empty_like)
    empty = staticmethod(np.empty)
    moveaxis = staticmethod(np.moveaxis)


class _SharedStore:
    def __init__(self, parties):
        self._values = {}
        self._condition = threading.Condition()
        self._barrier = threading.Barrier(parties, timeout=3.0)

    def open_handle(self):
        return _SharedStoreHandle(self)

    def set(self, key, value):
        with self._condition:
            self._values[key] = value
            self._condition.notify_all()

    def get(self, key):
        with self._condition:
            return self._values[key]

    def barrier(self):
        self._barrier.wait()

    def wait_for(self, predicate, *, timeout=3.0):
        with self._condition:
            return self._condition.wait_for(
                lambda: predicate(dict(self._values)), timeout=timeout
            )


class _SharedStoreHandle:
    def __init__(self, store):
        self._store = store
        self._closed = False

    def __setitem__(self, key, value):
        if self._closed:
            raise RuntimeError("shared store handle is closed")
        self._store.set(key, value)

    def __getitem__(self, key):
        if self._closed:
            raise RuntimeError("shared store handle is closed")
        return self._store.get(key)

    def barrier(self):
        if self._closed:
            raise RuntimeError("shared store handle is closed")
        self._store.barrier()

    def close(self):
        self._closed = True


class _RawComm:
    def __init__(self, *, abort_error=None):
        self.abort_error = abort_error
        self.abort_calls = 0
        self.abort_event = threading.Event()
        self.destroy_calls = 0

    def abort(self):
        self.abort_calls += 1
        self.abort_event.set()
        if self.abort_error is not None:
            raise self.abort_error

    def destroy(self):
        self.destroy_calls += 1


class _CpuNcclBackend(FakeNcclBackend):
    def __init__(self, size, rank, store, raw_comm):
        super().__init__(size, rank)
        self._store_proxy = store
        self._comm = raw_comm


def _cpu_collective(rank, size, store, raw_comm):
    from renormalizer.backend._distributed.collectives import CupyNcclCollective

    context = DistributedContext(rank, rank, size, size)
    backend = _CpuNcclBackend(size, rank, store.open_handle(), raw_comm)
    wrapper = CupyNcclCollective(
        context,
        cupy_module=_CpuOnlyCupy(),
        init_process_group=lambda *args, **kwargs: backend,
        host="127.0.0.1",
        port=23456,
    )
    wrapper._bootstrap_store_proxy = store.open_handle()

    def local_fatal_capability_code(*, _deadline=None):
        local_code = 0
        try:
            backend_store = getattr(backend, "_store_proxy")
            if type(backend_store) is not _SharedStoreHandle:
                raise RuntimeError("test private store contract is unavailable")
            probe_key = "test.status_workspace.rank.{}".format(rank)
            backend_store[probe_key] = 0
            if backend_store[probe_key] != 0:
                raise RuntimeError("test private store round trip failed")
        except BaseException:
            local_code |= 4
        try:
            if not callable(getattr(backend._comm, "abort", None)):
                raise RuntimeError("test private communicator abort is unavailable")
        except BaseException:
            local_code |= 4
        return local_code

    wrapper._local_fatal_capability_code = local_fatal_capability_code
    return wrapper, backend


@pytest.fixture
def context():
    return DistributedContext(rank=1, local_rank=0, world_size=2, local_world_size=2)


@pytest.fixture
def collective(context):
    from renormalizer.backend._distributed.collectives import CupyNcclCollective

    backend = FakeNcclBackend(context.world_size, context.rank)
    collective = CupyNcclCollective(
        context,
        cupy_module=cupy,
        init_process_group=lambda *args, **kwargs: backend,
        host="127.0.0.1",
        port=23456,
    )
    try:
        yield collective, backend
    finally:
        collective.close()


def test_constructor_uses_global_rank_and_selects_local_device(context):
    from renormalizer.backend._distributed.collectives import CupyNcclCollective

    calls = []

    def initialize(*args, **kwargs):
        calls.append((args, kwargs, cupy.cuda.runtime.getDevice()))
        return FakeNcclBackend(context.world_size, context.rank)

    cupy.cuda.Device(1).use()
    collective = CupyNcclCollective(
        context,
        cupy_module=cupy,
        init_process_group=initialize,
        host="127.0.0.9",
        port=23456,
    )
    try:
        assert calls == [
            (
                (context.world_size, context.rank),
                {"backend": "nccl", "host": "127.0.0.9", "port": 23456},
                context.local_rank,
            )
        ]
        assert collective.rank == context.rank
        assert collective.size == context.world_size
    finally:
        collective.close()


def test_constructor_stops_acquired_backend_when_validation_fails(context):
    from renormalizer.backend._distributed.collectives import CupyNcclCollective

    backend = FakeNcclBackend(context.world_size, rank=0)

    with pytest.raises(RuntimeError, match="rank does not match"):
        CupyNcclCollective(
            context,
            cupy_module=cupy,
            init_process_group=lambda *args, **kwargs: backend,
            host="127.0.0.1",
            port=23456,
        )

    assert backend.stop_calls == 1


@pytest.mark.parametrize(
    "options, message",
    [
        ({"host": "", "port": 23456}, "host must be a non-empty string"),
        (
            {"host": "127.0.0.1", "port": 0},
            "port must be an integer between 1 and 65535",
        ),
        (
            {"host": "127.0.0.1", "port": True},
            "port must be an integer between 1 and 65535",
        ),
    ],
)
def test_constructor_rejects_invalid_rendezvous_before_initialization(
    context, options, message
):
    from renormalizer.backend._distributed.collectives import CupyNcclCollective

    called = False

    def initialize(*args, **kwargs):
        nonlocal called
        called = True

    with pytest.raises((TypeError, ValueError), match=message):
        CupyNcclCollective(
            context,
            cupy_module=cupy,
            init_process_group=initialize,
            **options,
        )

    assert called is False


@pytest.mark.parametrize(
    "options, message",
    [
        ({"host": None, "port": 23456}, "host must be a non-empty string"),
        (
            {"host": "127.0.0.1", "port": None},
            "port must be an integer between 1 and 65535",
        ),
    ],
)
def test_constructor_requires_concrete_rendezvous_before_initialization(
    context, options, message
):
    from renormalizer.backend._distributed.collectives import CupyNcclCollective

    called = False

    def initialize(*args, **kwargs):
        nonlocal called
        called = True

    with pytest.raises((TypeError, ValueError), match=message):
        CupyNcclCollective(
            context,
            cupy_module=cupy,
            init_process_group=initialize,
            **options,
        )

    assert called is False


def test_constructor_accepts_backend_without_private_size_attribute(context):
    from renormalizer.backend._distributed.collectives import CupyNcclCollective

    backend = FakeNcclBackend(context.world_size, context.rank)

    collective = CupyNcclCollective(
        context,
        cupy_module=cupy,
        init_process_group=lambda *args, **kwargs: backend,
        host="127.0.0.1",
        port=23456,
    )
    collective.close()

    assert backend.stop_calls == 1


def test_close_is_idempotent_and_rejects_later_calls(collective):
    wrapper, backend = collective

    wrapper.close()
    wrapper.close()

    assert backend.stop_calls == 1
    with pytest.raises(RuntimeError, match="collective is closed"):
        wrapper.barrier()


def test_collective_close_retries_without_dropping_backend(context):
    from renormalizer.backend._distributed.collectives import CupyNcclCollective

    backend = FailingOnceNcclBackend(context.world_size, context.rank)
    collective = CupyNcclCollective(
        context,
        cupy_module=cupy,
        init_process_group=lambda *args, **kwargs: backend,
        host="127.0.0.1",
        port=23456,
    )

    with pytest.raises(RuntimeError, match="injected stop failure"):
        collective.close()

    assert collective._backend is backend
    assert collective._closed is False
    collective.barrier()

    collective.close()
    collective.close()

    assert backend.stop_calls == 2
    assert collective._backend is None
    assert collective._closed is True


def test_fatal_bootstrap_rejects_every_rank_when_raw_abort_is_missing():
    store = _SharedStore(2)
    wrappers = [_cpu_collective(rank, 2, store, object())[0] for rank in range(2)]
    results = [None, None]
    errors = [None, None]

    def run(rank):
        try:
            results[rank] = wrappers[rank]._bootstrap_fatal_control()
        except BaseException as error:
            errors[rank] = error

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(2)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5.0)
        assert all(not thread.is_alive() for thread in threads)
        assert errors == [None, None]
        assert all(result != 0 for result in results)
    finally:
        for wrapper in wrappers:
            wrapper.close()


def test_independent_store_proxy_requires_exact_installed_type(monkeypatch):
    from cupyx.distributed import _store

    from renormalizer.backend._distributed.collectives import CupyNcclCollective

    constructions = []

    class InstalledTCPStoreProxy:
        def __init__(self, host="127.0.0.1", port=13333):
            constructions.append((host, port, self))

    class ChangedTCPStoreProxy(InstalledTCPStoreProxy):
        pass

    class RaisingBackend:
        @property
        def _store_proxy(self):
            raise RuntimeError("injected private store access failure")

    monkeypatch.setattr(_store, "TCPStoreProxy", InstalledTCPStoreProxy)
    installed = InstalledTCPStoreProxy()
    created = CupyNcclCollective._independent_store_proxy(
        SimpleNamespace(_store_proxy=installed), "127.0.0.9", 24567
    )
    assert type(created) is InstalledTCPStoreProxy
    assert created is not installed
    assert constructions[-1][:2] == ("127.0.0.9", 24567)

    invalid_backends = (
        SimpleNamespace(),
        SimpleNamespace(_store_proxy=ChangedTCPStoreProxy()),
        SimpleNamespace(_store_proxy=object()),
        RaisingBackend(),
    )
    assert all(
        CupyNcclCollective._independent_store_proxy(backend, "127.0.0.9", 24567) is None
        for backend in invalid_backends
    )


def test_cpu_control_store_handle_has_separate_identity_and_lifetime():
    store = _SharedStore(1)
    wrapper, backend = _cpu_collective(0, 1, store, _RawComm())
    try:
        assert wrapper._bootstrap_store_proxy is not backend._store_proxy
        backend._store_proxy.close()
        wrapper._bootstrap_store_proxy["independent-control"] = 7
        assert wrapper._bootstrap_store_proxy["independent-control"] == 7
    finally:
        wrapper.close()


@pytest.mark.parametrize("missing_capability", ("store", "raising_store", "abort"))
def test_fatal_bootstrap_rejects_every_rank_for_one_rank_capability_loss(
    missing_capability,
):
    store = _SharedStore(2)
    raw_comms = [_RawComm(), _RawComm()]
    pairs = [_cpu_collective(rank, 2, store, raw_comms[rank]) for rank in range(2)]
    wrappers = [pair[0] for pair in pairs]
    backends = [pair[1] for pair in pairs]
    results = [None, None]
    errors = [None, None]

    if missing_capability == "store":
        del backends[1]._store_proxy
    elif missing_capability == "raising_store":

        class RaisingStoreBackend(type(backends[1])):
            @property
            def _store_proxy(self):
                raise RuntimeError("injected private store access failure")

        backends[1].__class__ = RaisingStoreBackend
    else:
        backends[1]._comm = object()

    def run(rank):
        try:
            results[rank] = wrappers[rank]._bootstrap_fatal_control()
        except BaseException as error:
            errors[rank] = error

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(2)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5.0)
        assert all(not thread.is_alive() for thread in threads)
        assert errors == [None, None]
        assert all(result != 0 for result in results)
        assert [raw.abort_calls for raw in raw_comms] == [0, 0]
        assert all(backend.calls == [] for backend in backends)
    finally:
        for wrapper in wrappers:
            wrapper.close()


def _bootstrap_cpu_wrappers(wrappers):
    errors = [None, None]

    def bootstrap(rank):
        try:
            assert wrappers[rank]._bootstrap_fatal_control() == 0
        except BaseException as error:
            errors[rank] = error

    threads = [threading.Thread(target=bootstrap, args=(rank,)) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5.0)
    assert all(not thread.is_alive() for thread in threads)
    assert errors == [None, None]


def _close_cpu_wrappers(wrappers):
    errors = [None, None]

    def close(rank):
        try:
            wrappers[rank].close()
        except BaseException as error:
            errors[rank] = error

    threads = [threading.Thread(target=close, args=(rank,)) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5.0)
    assert all(not thread.is_alive() for thread in threads)
    assert errors == [None, None]


_TASK_18_2_TIMEOUT_S = 3.0


def _start_task_18_2_call(call, *, name=None):
    results = []
    errors = []
    done = threading.Event()

    def run():
        try:
            results.append(call())
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    thread = threading.Thread(target=run, name=name, daemon=True)
    thread.start()
    return thread, results, errors, done


def _join_task_18_2_call(thread, done):
    assert done.wait(_TASK_18_2_TIMEOUT_S)
    thread.join(_TASK_18_2_TIMEOUT_S)
    assert not thread.is_alive()


def _task_18_2_runtime(wrapper, backend):
    from renormalizer.backend._distributed.context import DistributedRendezvous
    from renormalizer.backend._distributed.mesh import DeviceMesh
    from renormalizer.backend._distributed.terminal import _TerminalLifecycleGate
    from renormalizer.backend.distributed_runtime import CupyDistributedRuntime

    runtime = CupyDistributedRuntime(
        backend=backend,
        context=wrapper._context,
        rendezvous=DistributedRendezvous("127.0.0.1", 23456),
        mesh=DeviceMesh((wrapper.size,), ("rank",), wrapper.rank),
        collective=wrapper,
    )
    runtime._terminal_gate = _TerminalLifecycleGate()
    runtime._arm_communicator_fatal()
    return runtime


def _completed_test_monitor(wrapper):
    thread = threading.Thread(
        target=lambda: None,
        name="renormalizer-test-completed-fatal-monitor-rank-{}".format(
            wrapper.rank
        ),
        daemon=True,
    )
    with wrapper._fatal_condition:
        wrapper._fatal_monitor_thread = thread
        wrapper._fatal_monitor_state = "starting"
    thread.start()
    thread.join(_TASK_18_2_TIMEOUT_S)
    assert not thread.is_alive()
    with wrapper._fatal_condition:
        wrapper._fatal_monitor_state = "completed"
        wrapper._fatal_condition.notify_all()
    return thread


def _single_rank_task_18_2_runtime(monkeypatch, *, start_monitor=False):
    store = _SharedStore(1)
    raw_comm = _RawComm()
    wrapper, backend = _cpu_collective(0, 1, store, raw_comm)
    start_fatal_monitor = wrapper._start_fatal_monitor
    monkeypatch.setattr(
        wrapper,
        "_start_fatal_monitor",
        lambda **_kwargs: _completed_test_monitor(wrapper),
    )
    assert wrapper._bootstrap_fatal_control() == 0
    monkeypatch.setattr(wrapper, "_start_fatal_monitor", start_fatal_monitor)
    runtime = _task_18_2_runtime(wrapper, backend)
    if start_monitor:
        wrapper._start_fatal_monitor()
    return runtime, wrapper, backend, store, raw_comm


def test_collective_pending_primary_is_not_public_terminal_error(monkeypatch):
    _, wrapper, _, _, raw_comm = _single_rank_task_18_2_runtime(monkeypatch)
    primary = RuntimeError("pending communicator primary")
    callback_entered = threading.Event()
    release_callback = threading.Event()
    callback_errors = []

    def handler(error):
        try:
            assert error is primary
            assert raw_comm.abort_event.is_set()
            callback_entered.set()
            assert release_callback.wait(_TASK_18_2_TIMEOUT_S)
        except BaseException as caught:
            callback_errors.append(caught)
            callback_entered.set()
            raise

    wrapper._install_fatal_handler(handler)
    publisher, results, errors, done = _start_task_18_2_call(
        lambda: wrapper._enter_observed_fatal(primary, wrapper.rank)
    )
    try:
        assert callback_entered.wait(_TASK_18_2_TIMEOUT_S)
        assert getattr(wrapper, "_fatal_pending_primary", None) is primary
        assert wrapper._fatal_error is None
        assert wrapper._fatal_origin_rank is None
        with pytest.raises(RuntimeError, match="pending"):
            wrapper._begin_collective_operation()
        assert wrapper._admitted_operations == 0
    finally:
        release_callback.set()
        _join_task_18_2_call(publisher, done)

    assert callback_errors == []
    assert errors == []
    assert results == [primary]
    assert wrapper._fatal_error is primary
    assert wrapper._fatal_origin_rank == wrapper.rank


def test_fatal_waits_for_runtime_admission_outside_publication_lock(monkeypatch):
    runtime, wrapper, backend, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    blocker = gate.admit_runtime("blocked_runtime_operation")
    primary = RuntimeError("fatal waits for runtime admission")
    order = []
    drain_entered = threading.Event()
    blocker_released = False

    original_abort = wrapper._abort_local_communicator

    def record_abort_start(**kwargs):
        event = original_abort(**kwargs)
        order.append(("abort_start", wrapper._fatal_publication_lock._is_owned()))
        return event

    monkeypatch.setattr(wrapper, "_abort_local_communicator", record_abort_start)
    original_wait = gate.wait_for_admissions

    def record_admission_drain(transition, timeout_s):
        order.append(("admission_drain", wrapper._fatal_publication_lock._is_owned()))
        drain_entered.set()
        return original_wait(transition, timeout_s)

    monkeypatch.setattr(gate, "wait_for_admissions", record_admission_drain)
    original_retain_error = runtime._terminal_quarantine.retain_error

    def record_quarantine(error):
        order.append(("quarantine", wrapper._fatal_publication_lock._is_owned()))
        assert runtime._terminal_error is None
        assert getattr(backend, "_execution_terminal_error", None) is None
        assert wrapper._fatal_error is None
        return original_retain_error(error)

    monkeypatch.setattr(
        runtime._terminal_quarantine, "retain_error", record_quarantine
    )
    original_store_set = wrapper._fatal_store_set

    def record_store_set(key, value, **kwargs):
        if key == wrapper._fatal_ack_key(wrapper.rank):
            order.append(
                (
                    "ack",
                    wrapper._fatal_publication_lock._is_owned(),
                    runtime._terminal_error,
                    getattr(backend, "_execution_terminal_error", None),
                    wrapper._fatal_error,
                )
            )
        return original_store_set(key, value, **kwargs)

    monkeypatch.setattr(wrapper, "_fatal_store_set", record_store_set)
    publisher, results, errors, done = _start_task_18_2_call(
        lambda: runtime._enter_communicator_fatal(primary)
    )
    try:
        assert drain_entered.wait(_TASK_18_2_TIMEOUT_S)
        assert [entry[0] for entry in order] == ["abort_start", "admission_drain"]
        assert order[1][1] is False
        assert wrapper._fatal_error is None
        assert runtime._terminal_error is None
        gate.release(blocker)
        blocker_released = True
        _join_task_18_2_call(publisher, done)
    finally:
        if not blocker_released:
            gate.release(blocker)
        if publisher.is_alive():
            _join_task_18_2_call(publisher, done)

    assert errors == []
    assert results == [primary]
    assert [entry[0] for entry in order] == [
        "abort_start",
        "admission_drain",
        "quarantine",
        "ack",
    ]
    assert order[2][1] is False
    assert order[3][1] is False
    assert order[3][2:] == (primary, primary, primary)


def test_admitted_discoverer_converts_token_without_self_deadlock(monkeypatch):
    runtime, wrapper, _, _, raw_comm = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    primary = RuntimeError("publication owner primary")
    later = RuntimeError("admitted discoverer primary")
    token_admitted = threading.Event()
    owner_pending = threading.Event()
    conversion_errors = []
    joined_primaries = []

    original_begin_fatal = gate._recover_fatal

    def observe_pending(error, discovering_token=None, **kwargs):
        transition = original_begin_fatal(
            error, discovering_token, **kwargs
        )
        owner_pending.set()
        return transition

    monkeypatch.setattr(gate, "_recover_fatal", observe_pending)

    def discover():
        token = gate.admit_runtime("fatal_discoverer")
        token_admitted.set()
        assert owner_pending.wait(_TASK_18_2_TIMEOUT_S)
        with wrapper._communicator_fatal_reservation(
            later,
            join_existing=True,
            discovering_token=token,
            handler_override=runtime._communicator_fatal_hook,
        ) as reservation:
            joined_primaries.append(reservation.primary)
            try:
                gate.release(token)
            except BaseException as error:
                conversion_errors.append(error)
        return reservation.primary

    discoverer, results, errors, done = _start_task_18_2_call(discover)
    assert token_admitted.wait(_TASK_18_2_TIMEOUT_S)
    owner, owner_results, owner_errors, owner_done = _start_task_18_2_call(
        lambda: runtime._enter_communicator_fatal(primary)
    )
    _join_task_18_2_call(discoverer, done)
    _join_task_18_2_call(owner, owner_done)

    assert errors == []
    assert results == [primary]
    assert owner_errors == []
    assert owner_results == [primary]
    assert joined_primaries == [primary]
    assert len(conversion_errors) == 1
    assert "converted" in str(conversion_errors[0])
    assert raw_comm.abort_calls == 1
    assert wrapper._fatal_error is primary


def test_remote_monitor_and_local_failure_join_one_transition(monkeypatch):
    store = _SharedStore(2)
    raw_comms = [_RawComm(), _RawComm()]
    pairs = [_cpu_collective(rank, 2, store, raw_comms[rank]) for rank in range(2)]
    wrappers = [pair[0] for pair in pairs]
    backends = [pair[1] for pair in pairs]
    for wrapper in wrappers:
        monkeypatch.setattr(
            wrapper,
            "_start_fatal_monitor",
            lambda retained=wrapper, **_kwargs: _completed_test_monitor(retained),
        )
    _bootstrap_cpu_wrappers(wrappers)
    runtime = _task_18_2_runtime(wrappers[1], backends[1])
    gate = runtime._terminal_gate
    blocker = gate.admit_runtime("hold_local_publication")
    primary = RuntimeError("local failure wins monitor race")
    pending_entered = threading.Event()
    monitor_acknowledged = threading.Event()
    blocker_released = False
    transitions = []
    monitor_outcomes = []
    monitor_publication_entries = []
    monitor = None

    original_begin_fatal = gate._recover_fatal

    def record_begin_fatal(error, discovering_token=None, **kwargs):
        transition = original_begin_fatal(
            error, discovering_token, **kwargs
        )
        transitions.append(transition)
        pending_entered.set()
        return transition

    monkeypatch.setattr(gate, "_recover_fatal", record_begin_fatal)
    original_begin_publication = wrappers[1]._begin_fatal_publication

    def record_begin_publication(*args, **kwargs):
        result = original_begin_publication(*args, **kwargs)
        if threading.current_thread().name == "task-18.2-remote-monitor":
            monitor_publication_entries.append(result)
        return result

    monkeypatch.setattr(
        wrappers[1], "_begin_fatal_publication", record_begin_publication
    )
    original_acknowledge = wrappers[1]._acknowledge_fatal_monitor_exit

    def record_acknowledgment(outcome):
        monitor_outcomes.append(outcome)
        monitor_acknowledged.set()
        return original_acknowledge(outcome)

    monkeypatch.setattr(
        wrappers[1], "_acknowledge_fatal_monitor_exit", record_acknowledgment
    )
    store.set(wrappers[1]._fatal_ack_key(0), 1)
    local, local_results, local_errors, local_done = _start_task_18_2_call(
        lambda: runtime._enter_communicator_fatal(primary),
        name="task-18.2-local-failure",
    )
    try:
        assert pending_entered.wait(_TASK_18_2_TIMEOUT_S)
        assert raw_comms[1].abort_event.wait(_TASK_18_2_TIMEOUT_S)
        store.set(wrappers[1]._fatal_key(0), 1)
        monitor = threading.Thread(
            target=wrappers[1]._monitor_fatal_records,
            name="task-18.2-remote-monitor",
            daemon=True,
        )
        wrappers[1]._fatal_monitor_thread = monitor
        monitor.start()
        assert monitor_acknowledged.wait(_TASK_18_2_TIMEOUT_S)
        assert len(transitions) == 1
        assert transitions[0] is gate._fatal_transition
        assert transitions[0].primary is primary
        assert len(monitor_outcomes) == 1
        assert monitor_outcomes[0].kind == "fatal_elected"
        assert monitor_outcomes[0].primary is primary
        assert monitor_publication_entries == []
        gate.release(blocker)
        blocker_released = True
        _join_task_18_2_call(local, local_done)
        monitor.join(_TASK_18_2_TIMEOUT_S)
        assert not monitor.is_alive()
    finally:
        if not blocker_released:
            gate.release(blocker)
        store.set(wrappers[1]._fatal_ack_key(0), 1)
        if local.is_alive():
            _join_task_18_2_call(local, local_done)
        if monitor is not None and monitor.is_alive():
            wrappers[1]._fatal_monitor_stop.set()
            monitor.join(_TASK_18_2_TIMEOUT_S)

    assert local_errors == []
    assert local_results == [primary]
    assert wrappers[1]._fatal_error is primary
    assert raw_comms[1].abort_calls == 1


def test_abort_ack_and_first_primary_are_exactly_once(monkeypatch):
    runtime, wrapper, _, _, raw_comm = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    blocker = gate.admit_runtime("hold_first_publication")
    first = RuntimeError("first fatal primary")
    later = RuntimeError("later fatal primary")
    pending_entered = threading.Event()
    later_joined = threading.Event()
    blocker_released = False
    writes = []
    monitor_stopped_before_ack = []

    original_begin_fatal = gate._recover_fatal

    def record_begin_fatal(error, discovering_token=None, **kwargs):
        transition = original_begin_fatal(
            error, discovering_token, **kwargs
        )
        pending_entered.set()
        return transition

    monkeypatch.setattr(gate, "_recover_fatal", record_begin_fatal)
    original_store_set = wrapper._fatal_store_set

    def record_store_set(key, value, **kwargs):
        if int(value) == 1:
            writes.append(key)
        if key == wrapper._fatal_ack_key(wrapper.rank):
            monitor_stopped_before_ack.append(wrapper._fatal_monitor_stop.is_set())
        return original_store_set(key, value, **kwargs)

    monkeypatch.setattr(wrapper, "_fatal_store_set", record_store_set)
    later_thread_name = "task-18.2-later-primary"
    later_thread = None
    original_begin_publication = wrapper._begin_fatal_publication

    def record_begin_publication(*args, **kwargs):
        result = original_begin_publication(*args, **kwargs)
        if threading.current_thread().name == later_thread_name:
            later_joined.set()
        return result

    monkeypatch.setattr(wrapper, "_begin_fatal_publication", record_begin_publication)
    first_thread, first_results, first_errors, first_done = _start_task_18_2_call(
        lambda: runtime._enter_communicator_fatal(first),
        name="task-18.2-first-primary",
    )
    later_results = []
    later_errors = []
    later_done = threading.Event()
    try:
        assert pending_entered.wait(_TASK_18_2_TIMEOUT_S)
        later_thread, later_results, later_errors, later_done = _start_task_18_2_call(
            lambda: runtime._enter_communicator_fatal(later),
            name=later_thread_name,
        )
        assert later_joined.wait(_TASK_18_2_TIMEOUT_S)
        gate.release(blocker)
        blocker_released = True
        _join_task_18_2_call(first_thread, first_done)
        _join_task_18_2_call(later_thread, later_done)
        assert runtime._enter_communicator_fatal(
            RuntimeError("post-publication failure")
        ) is first
    finally:
        if not blocker_released:
            gate.release(blocker)
        if first_thread.is_alive():
            _join_task_18_2_call(first_thread, first_done)
        if later_thread is not None and later_thread.is_alive():
            _join_task_18_2_call(later_thread, later_done)

    assert first_errors == []
    assert later_errors == []
    assert first_results == [first]
    assert later_results == [first]
    assert gate._fatal_transition.primary is first
    assert wrapper._fatal_error is first
    assert raw_comm.abort_calls == 1
    assert writes.count(wrapper._fatal_key(wrapper.rank)) == 1
    assert writes.count(wrapper._fatal_ack_key(wrapper.rank)) == 1
    assert monitor_stopped_before_ack == [True]


def test_monitor_detected_fatal_retires_before_every_publication_sentinel(
    monkeypatch,
):
    from renormalizer.backend._distributed.terminal import _TerminalPhase

    runtime, wrapper, _, store, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    sentinels = []
    store_access_phases = []
    monitor = None

    def record_sentinel(name):
        with gate._condition:
            phase = gate._phase
        sentinels.append(
            (
                name,
                monitor.is_alive(),
                threading.current_thread() is monitor,
                phase,
            )
        )

    original_store_set = wrapper._fatal_store_set

    def record_store_set(key, value, **kwargs):
        with gate._condition:
            store_access_phases.append(("set", key, gate._phase))
        if key == wrapper._fatal_key(wrapper.rank):
            record_sentinel("fatal_key")
        elif key == wrapper._fatal_ack_key(wrapper.rank):
            record_sentinel("ack")
        return original_store_set(key, value, **kwargs)

    monkeypatch.setattr(wrapper, "_fatal_store_set", record_store_set)
    original_store_get = wrapper._fatal_store_get

    def record_store_get(key, **kwargs):
        with gate._condition:
            store_access_phases.append(("get", key, gate._phase))
        return original_store_get(key, **kwargs)

    monkeypatch.setattr(wrapper, "_fatal_store_get", record_store_get)
    original_read = wrapper._read_fatal_origin

    def record_monitor_read(**kwargs):
        with gate._condition:
            store_access_phases.append(("monitor_read", None, gate._phase))
        return original_read(**kwargs)

    monkeypatch.setattr(wrapper, "_read_fatal_origin", record_monitor_read)
    original_abort = wrapper._abort_local_communicator

    def record_abort(**kwargs):
        record_sentinel("abort")
        return original_abort(**kwargs)

    monkeypatch.setattr(wrapper, "_abort_local_communicator", record_abort)
    original_runtime_publish = runtime._publish_communicator_fatal_transition

    def record_runtime_publish(transition):
        record_sentinel("runtime_public")
        return original_runtime_publish(transition)

    monkeypatch.setattr(
        runtime,
        "_publish_communicator_fatal_transition",
        record_runtime_publish,
    )
    original_collective_publish = wrapper._publish_communicator_fatal_locked

    def record_collective_publish(primary, **kwargs):
        record_sentinel("collective_public")
        return original_collective_publish(primary, **kwargs)

    monkeypatch.setattr(
        wrapper,
        "_publish_communicator_fatal_locked",
        record_collective_publish,
    )
    original_gate_publish = gate.publish_fatal

    def record_gate_publish(transition, snapshot):
        record_sentinel("gate_public")
        return original_gate_publish(transition, snapshot)

    monkeypatch.setattr(gate, "publish_fatal", record_gate_publish)

    wrapper._start_fatal_monitor()
    monitor = wrapper._fatal_monitor_thread
    store.set(wrapper._fatal_key(wrapper.rank), 1)
    snapshot = gate.wait_for_published(_TASK_18_2_TIMEOUT_S)
    wrapper._wait_for_joined_fatal_publication()
    monitor.join(_TASK_18_2_TIMEOUT_S)

    assert not monitor.is_alive()
    assert snapshot is gate._fatal_transition.primary
    assert [entry[0] for entry in sentinels] == [
        "fatal_key",
        "abort",
        "runtime_public",
        "collective_public",
        "ack",
        "gate_public",
    ]
    assert all(not alive and not is_monitor for _, alive, is_monitor, _ in sentinels)
    assert all(phase is _TerminalPhase.FATAL_PENDING for *_, phase in sentinels)
    assert all(
        phase not in {_TerminalPhase.FATAL_PUBLISHED, _TerminalPhase.RUNTIME_CLOSED}
        for *_, phase in store_access_phases
    )


def test_close_joins_local_fatal_publication_before_clearing_collective(
    monkeypatch,
):
    runtime, wrapper, backend, _, raw_comm = _single_rank_task_18_2_runtime(
        monkeypatch
    )
    retained_store = wrapper._bootstrap_store_proxy
    monitor_before_stop = threading.Event()
    release_monitor = threading.Event()
    stop_requested = threading.Event()
    fatal_selected = threading.Event()
    owner_after_monitor_exit = threading.Event()
    release_owner = threading.Event()
    close_waiting_for_publication = threading.Event()
    owner_name = "task-18.2-local-fatal-after-stop"
    close_name = "task-18.2-close-joins-local-fatal"

    original_observe_stop = wrapper._observe_fatal_monitor_stop

    def pause_before_stop_observation():
        monitor_before_stop.set()
        assert release_monitor.wait(_TASK_18_2_TIMEOUT_S * 2)
        return original_observe_stop()

    monkeypatch.setattr(
        wrapper, "_observe_fatal_monitor_stop", pause_before_stop_observation
    )
    original_request_stop = wrapper._request_fatal_monitor_stop

    def record_stop_request():
        generation = original_request_stop()
        stop_requested.set()
        return generation

    monkeypatch.setattr(wrapper, "_request_fatal_monitor_stop", record_stop_request)
    original_select = wrapper._select_fatal_monitor_outcome

    def record_fatal_selection(kind, value, **kwargs):
        outcome = original_select(kind, value, **kwargs)
        if kind == "fatal_elected":
            fatal_selected.set()
        return outcome

    monkeypatch.setattr(
        wrapper, "_select_fatal_monitor_outcome", record_fatal_selection
    )
    original_join = wrapper._join_fatal_monitor

    def block_owner_after_monitor_join(thread, **kwargs):
        result = original_join(thread, **kwargs)
        if threading.current_thread().name == owner_name:
            owner_after_monitor_exit.set()
            assert release_owner.wait(_TASK_18_2_TIMEOUT_S * 2)
        return result

    monkeypatch.setattr(wrapper, "_join_fatal_monitor", block_owner_after_monitor_join)
    original_wait_for_publication = wrapper._wait_for_joined_fatal_publication

    def record_publication_join(**kwargs):
        if threading.current_thread().name == close_name:
            close_waiting_for_publication.set()
        return original_wait_for_publication(**kwargs)

    monkeypatch.setattr(
        wrapper, "_wait_for_joined_fatal_publication", record_publication_join
    )

    # Keep RED cleanup inside pytest even if the buggy close clears live fields.
    monkeypatch.setattr(
        wrapper,
        "_fatal_store_set",
        lambda key, value, **_kwargs: retained_store.__setitem__(key, value),
    )
    monkeypatch.setattr(
        wrapper,
        "_fatal_store_get",
        lambda key, **_kwargs: retained_store[key],
    )

    def retained_abort(**_kwargs):
        with wrapper._fatal_lock:
            if not wrapper._fatal_abort_started:
                wrapper._fatal_abort_started = True
                raw_comm.abort()
                wrapper._fatal_abort_completed = True
                wrapper._fatal_abort_event.set()
            return wrapper._fatal_abort_event

    monkeypatch.setattr(wrapper, "_abort_local_communicator", retained_abort)

    wrapper._start_fatal_monitor()
    monitor = wrapper._fatal_monitor_thread
    assert monitor_before_stop.wait(_TASK_18_2_TIMEOUT_S)
    closer, close_results, close_errors, close_done = _start_task_18_2_call(
        runtime.close, name=close_name
    )
    owner = None
    primary = RuntimeError("local fatal after monitor stop request")
    try:
        assert stop_requested.wait(_TASK_18_2_TIMEOUT_S)
        owner, owner_results, owner_errors, owner_done = _start_task_18_2_call(
            lambda: runtime._enter_communicator_fatal(primary), name=owner_name
        )
        assert fatal_selected.wait(_TASK_18_2_TIMEOUT_S)
        release_monitor.set()
        assert owner_after_monitor_exit.wait(_TASK_18_2_TIMEOUT_S)
        assert close_waiting_for_publication.wait(_TASK_18_2_TIMEOUT_S)

        assert not monitor.is_alive()
        assert not close_done.is_set()
        assert wrapper._backend is backend
        assert wrapper._bootstrap_store_proxy is not None
        assert wrapper._closed is False
        assert runtime.collective is wrapper
        assert runtime._closed is False
    finally:
        release_monitor.set()
        release_owner.set()
        if owner is not None and owner.is_alive():
            _join_task_18_2_call(owner, owner_done)
        if closer.is_alive():
            _join_task_18_2_call(closer, close_done)
        if monitor.is_alive():
            monitor.join(_TASK_18_2_TIMEOUT_S)

    assert owner_errors == []
    assert owner_results == [primary]
    assert close_results == []
    assert close_errors == [primary]
    assert runtime._closed is True


def test_visible_fatal_handoff_has_joinable_pre_reserved_owner(monkeypatch):
    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    monitor_before_stop = threading.Event()
    release_monitor = threading.Event()
    stop_requested = threading.Event()
    ordinary_begin_entered = threading.Event()
    release_ordinary_begin = threading.Event()
    close_join_entered = threading.Event()
    close_join_observations = []
    hard_exits = []
    owner_name = "task-18.2-pre-reserved-fatal-owner"
    close_name = "task-18.2-pre-reserved-fatal-close"

    original_observe_stop = wrapper._observe_fatal_monitor_stop

    def pause_before_stop_observation():
        monitor_before_stop.set()
        assert release_monitor.wait(_TASK_18_2_TIMEOUT_S * 2)
        return original_observe_stop()

    monkeypatch.setattr(
        wrapper, "_observe_fatal_monitor_stop", pause_before_stop_observation
    )
    original_request_stop = wrapper._request_fatal_monitor_stop

    def record_stop_request():
        generation = original_request_stop()
        stop_requested.set()
        return generation

    monkeypatch.setattr(wrapper, "_request_fatal_monitor_stop", record_stop_request)
    original_begin = wrapper._begin_fatal_publication

    def pause_before_ordinary_begin(*args, **kwargs):
        if threading.current_thread().name == owner_name:
            ordinary_begin_entered.set()
            assert release_ordinary_begin.wait(_TASK_18_2_TIMEOUT_S * 2)
        return original_begin(*args, **kwargs)

    monkeypatch.setattr(wrapper, "_begin_fatal_publication", pause_before_ordinary_begin)
    original_join = wrapper._wait_for_joined_fatal_publication

    def record_close_join(**kwargs):
        if threading.current_thread().name == close_name:
            with wrapper._fatal_condition:
                close_join_observations.append(
                    (
                        wrapper._fatal_publications,
                        wrapper._fatal_protocol_completed,
                    )
                )
            close_join_entered.set()
        return original_join(**kwargs)

    monkeypatch.setattr(
        wrapper, "_wait_for_joined_fatal_publication", record_close_join
    )

    def hard_exit():
        hard_exits.append(threading.current_thread().name)
        raise AssertionError("unexpected communicator hard exit")

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    wrapper._start_fatal_monitor()
    monitor = wrapper._fatal_monitor_thread
    assert monitor_before_stop.wait(_TASK_18_2_TIMEOUT_S)
    closer, close_results, close_errors, close_done = _start_task_18_2_call(
        runtime.close, name=close_name
    )
    owner = None
    primary = RuntimeError("fatal handoff owns publication before visibility")
    try:
        assert stop_requested.wait(_TASK_18_2_TIMEOUT_S)
        owner, owner_results, owner_errors, owner_done = _start_task_18_2_call(
            lambda: runtime._enter_communicator_fatal(primary), name=owner_name
        )
        assert ordinary_begin_entered.wait(_TASK_18_2_TIMEOUT_S)
        release_monitor.set()
        assert close_join_entered.wait(_TASK_18_2_TIMEOUT_S)
        assert close_join_observations == [(1, False)]
        assert hard_exits == []
        assert not close_done.is_set()
    finally:
        release_monitor.set()
        release_ordinary_begin.set()
        if owner is not None:
            _join_task_18_2_call(owner, owner_done)
        _join_task_18_2_call(closer, close_done)
        if monitor.is_alive():
            monitor.join(_TASK_18_2_TIMEOUT_S)

    assert owner_errors == []
    assert owner_results == [primary]
    assert close_results == []
    assert close_errors == [primary]
    assert close_join_observations == [(1, False)]
    assert hard_exits == []
    assert wrapper._fatal_publications == 0


def test_active_operator_primary_is_canonical_before_collective_publication(
    monkeypatch,
):
    from renormalizer.backend._distributed.async_owner import AsyncResourceOwner

    runtime, wrapper, backend, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    earlier = RuntimeError("earlier active operator primary")
    later = RuntimeError("later communicator failure")

    class OperatorCall:
        def __init__(self):
            self.primary_error = earlier
            self.secondary_errors = []

        def record_secondary(self, error):
            if error is not self.primary_error and error not in self.secondary_errors:
                self.secondary_errors.append(error)

    call = OperatorCall()
    owner = AsyncResourceOwner("compute")
    owner._operator_call = call
    owner.mark_enqueued()
    owner.force_quarantine(earlier)
    lease = SimpleNamespace(
        _active_operator_owner=owner,
        _status_workspace=None,
        _poisoned_error=None,
    )
    provider = SimpleNamespace(_active_lease=lease, _terminal_error=None)
    runtime._active_provider = provider
    selections = []
    acknowledgments = []
    original_select = wrapper._select_fatal_monitor_outcome

    def record_selection(kind, value, **kwargs):
        outcome = original_select(kind, value, **kwargs)
        if kind == "fatal_elected":
            selections.append(
                (
                    value,
                    wrapper._fatal_pending_primary,
                    runtime._terminal_gate._fatal_transition.primary,
                )
            )
        return outcome

    monkeypatch.setattr(wrapper, "_select_fatal_monitor_outcome", record_selection)
    original_store_set = wrapper._fatal_store_set

    def record_acknowledgment(key, value, **kwargs):
        if key == wrapper._fatal_ack_key(wrapper.rank):
            acknowledgments.append(
                (
                    wrapper._fatal_pending_primary,
                    wrapper._fatal_error,
                    runtime._terminal_error,
                    provider._terminal_error,
                    lease._poisoned_error,
                    backend._execution_terminal_error,
                )
            )
        return original_store_set(key, value, **kwargs)

    monkeypatch.setattr(wrapper, "_fatal_store_set", record_acknowledgment)

    result = wrapper._enter_observed_fatal(
        later, wrapper.rank, join_existing=True
    )

    assert result is earlier
    assert selections == [(earlier, earlier, earlier)]
    assert runtime._terminal_gate._fatal_transition.primary is earlier
    assert wrapper._fatal_pending_primary is earlier
    assert wrapper._fatal_error is earlier
    assert runtime._terminal_error is earlier
    assert provider._terminal_error is earlier
    assert lease._poisoned_error is earlier
    assert backend._execution_terminal_error is earlier
    assert acknowledgments == [(earlier,) * 6]
    assert call.secondary_errors == [later]
    assert wrapper._fatal_secondary_errors == (later,)


def test_runtime_owner_diagnostics_are_post_marker_unlocked_and_exact_once(
    monkeypatch,
):
    from renormalizer.backend._distributed.async_owner import AsyncResourceOwner

    runtime, wrapper, backend, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    primary = RuntimeError("runtime diagnostic canonical primary")
    owner_error = RuntimeError("runtime diagnostic owner secondary")
    call_error = RuntimeError("runtime diagnostic call secondary")
    discovered = RuntimeError("runtime diagnostic discovered secondary")
    diagnostic_failure = RuntimeError("runtime diagnostic callback failure")
    callback_entered = threading.Event()
    release_callback = threading.Event()
    callback_observations = []

    class OperatorCall:
        def __init__(self):
            self.primary_error = call_error
            self.secondary_errors = []

        def record_secondary(self, error):
            callback_observations.append(
                (
                    error,
                    gate.phase,
                    wrapper._fatal_error,
                    gate._condition._is_owned(),
                    runtime._terminal_state_lock._is_owned(),
                    wrapper._fatal_condition._is_owned(),
                    wrapper._fatal_lock._is_owned(),
                )
            )
            self.secondary_errors.append(error)
            if not callback_entered.is_set():
                callback_entered.set()
                assert release_callback.wait(_TASK_18_2_TIMEOUT_S * 2)
                raise diagnostic_failure

    call = OperatorCall()
    owner = AsyncResourceOwner("runtime-diagnostic-owner")
    owner._operator_call = call
    owner.mark_enqueued()
    owner.force_quarantine(owner_error)
    lease = SimpleNamespace(
        _active_operator_owner=owner,
        _status_workspace=None,
        _poisoned_error=None,
    )
    provider = SimpleNamespace(_active_lease=lease, _terminal_error=None)
    runtime._active_provider = provider
    runtime._terminal_error = primary
    backend._execution_terminal_error = primary

    publisher, publish_results, publish_errors, publish_done = (
        _start_task_18_2_call(
            lambda: wrapper._enter_observed_fatal(
                discovered,
                wrapper.rank,
                join_existing=True,
            ),
            name="task-18.3-runtime-diagnostic-publisher",
        )
    )
    waiter = None
    try:
        assert callback_entered.wait(_TASK_18_2_TIMEOUT_S)
        waiter, wait_results, wait_errors, wait_done = _start_task_18_2_call(
            lambda: gate.wait_for_published(_TASK_18_2_TIMEOUT_S),
            name="task-18.3-runtime-diagnostic-waiter",
        )
        assert wait_done.wait(_TASK_18_2_TIMEOUT_S)
        _join_task_18_2_call(waiter, wait_done)
        assert wait_results == [primary]
        assert wait_errors == []
    finally:
        release_callback.set()
        _join_task_18_2_call(publisher, publish_done)
        if waiter is not None and waiter.is_alive():
            _join_task_18_2_call(waiter, wait_done)

    assert publish_results == [primary]
    assert publish_errors == []
    assert callback_observations
    assert all(
        observation[1:] == (
            _TerminalPhase.FATAL_PUBLISHED,
            primary,
            False,
            False,
            False,
            False,
        )
        for observation in callback_observations
    )
    dispatched = [observation[0] for observation in callback_observations]
    for error in (owner_error, call_error, discovered):
        assert sum(retained is error for retained in dispatched) == 1
    assert diagnostic_failure in runtime._terminal_secondary_errors


def test_preexisting_backend_poison_is_canonical_for_fatal_publication(
    monkeypatch,
):
    runtime, wrapper, backend, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    earlier = RuntimeError("preexisting backend terminal poison")
    later = RuntimeError("later communicator failure")
    backend._execution_terminal_error = earlier
    diagnostic_observations = []
    original_record_secondary = wrapper._record_fatal_secondary

    def record_secondary(error):
        diagnostic_observations.append(
            (
                error,
                runtime._terminal_gate.phase,
                wrapper._fatal_error,
                runtime._terminal_gate._condition._is_owned(),
                runtime._terminal_state_lock._is_owned(),
                wrapper._fatal_condition._is_owned(),
                wrapper._fatal_lock._is_owned(),
            )
        )
        original_record_secondary(error)

    monkeypatch.setattr(wrapper, "_record_fatal_secondary", record_secondary)

    result = runtime._enter_communicator_fatal(later)
    snapshot = runtime._terminal_gate.wait_for_published(_TASK_18_2_TIMEOUT_S)

    assert result is earlier
    assert snapshot is earlier
    assert runtime._terminal_gate._fatal_transition.primary is earlier
    assert wrapper._fatal_pending_primary is earlier
    assert wrapper._fatal_error is earlier
    assert runtime._terminal_error is earlier
    assert backend._execution_terminal_error is earlier
    assert runtime._terminal_quarantine.first_error is earlier
    assert wrapper._fatal_secondary_errors == (later,)
    assert diagnostic_observations == [
        (
            later,
            _TerminalPhase.FATAL_PUBLISHED,
            earlier,
            False,
            False,
            False,
            False,
        )
    ]


def test_quarantine_after_election_cannot_replace_canonical_primary(monkeypatch):
    from renormalizer.backend._distributed.async_owner import AsyncResourceOwner

    runtime, wrapper, backend, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    primary = RuntimeError("elected communicator primary")
    later = RuntimeError("late concurrent quarantine failure")
    drain_entered = threading.Event()
    release_drain = threading.Event()
    original_wait = gate.wait_for_admissions

    def pause_after_election(transition, timeout_s):
        assert transition.primary is primary
        drain_entered.set()
        assert release_drain.wait(_TASK_18_2_TIMEOUT_S * 2)
        return original_wait(transition, timeout_s)

    monkeypatch.setattr(gate, "wait_for_admissions", pause_after_election)
    owner = AsyncResourceOwner("late-quarantine")
    owner.mark_enqueued()
    owner.force_quarantine(later)
    publisher, results, errors, done = _start_task_18_2_call(
        lambda: runtime._enter_communicator_fatal(primary),
        name="task-18.2-canonical-publication-owner",
    )
    quarantine = None
    try:
        assert drain_entered.wait(_TASK_18_2_TIMEOUT_S)
        quarantine, quarantine_results, quarantine_errors, quarantine_done = (
            _start_task_18_2_call(
                lambda: runtime._accept_async_quarantine(owner, later),
                name="task-18.2-late-quarantine-writer",
            )
        )
        _join_task_18_2_call(quarantine, quarantine_done)
        pending_identities = (
            gate._fatal_transition.primary,
            wrapper._fatal_pending_primary,
            runtime._terminal_error,
            backend._execution_terminal_error,
            runtime._terminal_quarantine.first_error,
        )
    finally:
        release_drain.set()
        _join_task_18_2_call(publisher, done)
        if quarantine is not None and quarantine.is_alive():
            _join_task_18_2_call(quarantine, quarantine_done)

    snapshot = gate.wait_for_published(_TASK_18_2_TIMEOUT_S)
    assert quarantine_errors == []
    assert quarantine_results == [None]
    assert all(value is primary for value in pending_identities)
    assert errors == []
    assert results == [primary]
    assert snapshot is primary
    assert gate._fatal_transition.primary is primary
    assert wrapper._fatal_pending_primary is primary
    assert wrapper._fatal_error is primary
    assert runtime._terminal_error is primary
    assert backend._execution_terminal_error is primary
    assert runtime._terminal_quarantine.first_error is primary
    assert runtime._terminal_quarantine.owners == (owner,)
    assert wrapper._fatal_secondary_errors == (later,)


def test_post_election_owner_fail_adopts_canonical_operator_primary(monkeypatch):
    from renormalizer.backend._distributed.async_owner import AsyncResourceOwner
    from renormalizer.backend._distributed.providers import _OperatorCall

    runtime, wrapper, backend, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    primary = RuntimeError("elected owner failure primary")
    later = RuntimeError("post-election owner failure")
    publication_waiting = threading.Event()
    release_publication = threading.Event()
    original_wait = gate.wait_for_admissions

    def pause_publication(transition, timeout_s):
        assert transition.primary is primary
        publication_waiting.set()
        assert release_publication.wait(_TASK_18_2_TIMEOUT_S * 2)
        return original_wait(transition, timeout_s)

    monkeypatch.setattr(gate, "wait_for_admissions", pause_publication)

    def fail_drain():
        raise later

    def accept_quarantine(owner):
        runtime._accept_async_quarantine(owner, owner.error)

    owner = AsyncResourceOwner(
        "post-election-owner",
        drainer=fail_drain,
        quarantine=accept_quarantine,
    )
    call = _OperatorCall(owner, None, None, entry_error=later)
    owner._operator_call = call
    owner.mark_enqueued()

    publisher, publish_results, publish_errors, publish_done = (
        _start_task_18_2_call(
            lambda: runtime._enter_communicator_fatal(primary),
            name="task-18.2-post-election-publication-owner",
        )
    )
    failing_owner = None
    try:
        assert publication_waiting.wait(_TASK_18_2_TIMEOUT_S)
        failing_owner, fail_results, fail_errors, fail_done = _start_task_18_2_call(
            lambda: owner.fail(later),
            name="task-18.2-post-election-owner-fail",
        )
        _join_task_18_2_call(failing_owner, fail_done)
        pending_identities = (
            gate._fatal_transition.primary,
            wrapper._fatal_pending_primary,
            runtime._terminal_error,
            backend._execution_terminal_error,
            runtime._terminal_quarantine.first_error,
            owner.error,
            call.primary_error,
        )
    finally:
        release_publication.set()
        _join_task_18_2_call(publisher, publish_done)
        if failing_owner is not None and failing_owner.is_alive():
            _join_task_18_2_call(failing_owner, fail_done)

    snapshot = gate.wait_for_published(_TASK_18_2_TIMEOUT_S)
    assert fail_results == []
    assert fail_errors == [primary]
    assert all(value is primary for value in pending_identities)
    assert owner.secondary_errors == (later,)
    assert call.secondary_errors == (later,)
    assert publish_errors == []
    assert publish_results == [primary]
    assert snapshot is primary
    assert wrapper._fatal_error is primary
    assert runtime._terminal_error is primary
    assert backend._execution_terminal_error is primary
    assert runtime._terminal_quarantine.owners == (owner,)
    assert runtime._terminal_secondary_errors == (later,)
    assert wrapper._fatal_secondary_errors == (later,)


def test_fatal_election_retains_collective_across_pending_close_finalizer(
    monkeypatch,
):
    runtime, wrapper, backend, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    primary = RuntimeError("fatal retained across pending close")
    elected = threading.Event()
    release_resolution = threading.Event()
    pending_finalizer = threading.Event()
    original_begin = runtime._begin_communicator_fatal

    def pause_after_election(*args, **kwargs):
        transition = original_begin(*args, **kwargs)
        elected.set()
        assert release_resolution.wait(_TASK_18_2_TIMEOUT_S * 2)
        return transition

    monkeypatch.setattr(runtime, "_begin_communicator_fatal", pause_after_election)
    original_clear = runtime._clear_runtime_references

    def record_pending_finalizer(*args, **kwargs):
        result = original_clear(*args, **kwargs)
        if kwargs.get("mark_closed") is False:
            pending_finalizer.set()
        return result

    monkeypatch.setattr(runtime, "_clear_runtime_references", record_pending_finalizer)

    publisher, results, errors, done = _start_task_18_2_call(
        lambda: runtime._enter_communicator_fatal(primary),
        name="task-18.2-retained-fatal-publisher",
    )
    assert elected.wait(_TASK_18_2_TIMEOUT_S)
    closer, close_results, close_errors, close_done = _start_task_18_2_call(
        runtime.close, name="task-18.2-retained-fatal-closer"
    )
    try:
        assert pending_finalizer.wait(_TASK_18_2_TIMEOUT_S)
        assert runtime.collective is wrapper
        assert runtime._pending_fatal_collective is wrapper
        assert wrapper._backend is backend
        assert wrapper._closed is False
        assert not close_done.is_set()
    finally:
        release_resolution.set()
        _join_task_18_2_call(publisher, done)
        with gate._condition:
            if gate._phase is _TerminalPhase.FATAL_PENDING:
                gate.publish_fatal(gate._fatal_transition, primary)
        _join_task_18_2_call(closer, close_done)

    assert errors == []
    assert results == [primary]
    assert close_results == []
    assert close_errors == [primary]
    assert gate.phase is _TerminalPhase.RUNTIME_CLOSED


def test_runtime_close_selects_clean_before_collective_teardown_gap(monkeypatch):
    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    teardown_complete = threading.Event()
    release_runtime_commit = threading.Event()
    publication_calls = []
    original_close = wrapper._close_for_runtime

    def pause_after_collective_teardown(close_gate, transition):
        result = original_close(close_gate, transition)
        teardown_complete.set()
        assert release_runtime_commit.wait(_TASK_18_2_TIMEOUT_S * 2)
        return result

    monkeypatch.setattr(wrapper, "_close_for_runtime", pause_after_collective_teardown)
    original_publish = wrapper._publish_communicator_fatal

    def record_publication(*args, **kwargs):
        publication_calls.append((args, kwargs))
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(wrapper, "_publish_communicator_fatal", record_publication)

    closer, close_results, close_errors, close_done = _start_task_18_2_call(
        runtime.close, name="task-18.2-close-before-runtime-commit"
    )
    assert teardown_complete.wait(_TASK_18_2_TIMEOUT_S)
    primary = RuntimeError("fatal after collective teardown")
    publisher, results, errors, done = _start_task_18_2_call(
        lambda: runtime._enter_communicator_fatal(primary),
        name="task-18.2-fatal-after-collective-teardown",
    )
    try:
        _join_task_18_2_call(publisher, done)
        assert results == []
        assert len(errors) == 1
        assert "committed" in str(errors[0])
        assert publication_calls == []
        assert gate._fatal_transition is None
        assert gate._runtime_close_commit_selected is True
        assert not close_done.is_set()
    finally:
        with gate._condition:
            if gate._phase is _TerminalPhase.FATAL_PENDING:
                gate.publish_fatal(gate._fatal_transition, primary)
        release_runtime_commit.set()
        _join_task_18_2_call(closer, close_done)

    assert close_errors == []
    assert close_results == [None]
    assert gate.phase is _TerminalPhase.RUNTIME_CLOSED


def test_gate_fatal_election_preselects_monitor_before_collective_prepare(
    monkeypatch,
):
    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    primary = RuntimeError("gate and monitor share fatal winner")
    prepare_entered = threading.Event()
    release_prepare = threading.Event()
    original_prepare = wrapper._prepare_local_fatal_locked

    def pause_collective_prepare(**kwargs):
        prepare_entered.set()
        assert release_prepare.wait(_TASK_18_2_TIMEOUT_S * 2)
        return original_prepare(**kwargs)

    monkeypatch.setattr(wrapper, "_prepare_local_fatal_locked", pause_collective_prepare)
    publisher, results, errors, done = _start_task_18_2_call(
        lambda: runtime._enter_communicator_fatal(primary),
        name="task-18.2-gate-monitor-election",
    )
    try:
        assert prepare_entered.wait(_TASK_18_2_TIMEOUT_S)
        generation = wrapper._request_fatal_monitor_stop()
        clean_attempt = wrapper._select_fatal_monitor_outcome(
            "stopped_clean", generation
        )
        assert gate._fatal_transition.primary is primary
        assert wrapper._fatal_pending_primary is primary
        assert clean_attempt.kind == "fatal_elected"
        assert clean_attempt.primary is primary
        assert wrapper._observe_fatal_monitor_outcome() is clean_attempt
    finally:
        release_prepare.set()
        _join_task_18_2_call(publisher, done)

    assert errors == []
    assert results == [primary]
    assert gate.phase is _TerminalPhase.FATAL_PUBLISHED


def test_protocol_completed_reporter_joins_until_gate_publication(monkeypatch):
    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    primary = RuntimeError("reporter joins immutable gate publication")
    gate_publish_entered = threading.Event()
    release_gate_publish = threading.Event()
    reporter_joining = threading.Event()
    original_gate_publish = gate.publish_fatal

    def pause_gate_publish(transition, snapshot):
        gate_publish_entered.set()
        assert release_gate_publish.wait(_TASK_18_2_TIMEOUT_S * 2)
        return original_gate_publish(transition, snapshot)

    monkeypatch.setattr(gate, "publish_fatal", pause_gate_publish)
    original_join = wrapper._wait_for_joined_fatal_publication

    def record_reporter_join(**kwargs):
        if threading.current_thread().name == "task-18.2-protocol-reporter":
            reporter_joining.set()
        return original_join(**kwargs)

    monkeypatch.setattr(
        wrapper, "_wait_for_joined_fatal_publication", record_reporter_join
    )

    owner, owner_results, owner_errors, owner_done = _start_task_18_2_call(
        lambda: runtime._enter_communicator_fatal(primary),
        name="task-18.2-protocol-owner",
    )
    assert gate_publish_entered.wait(_TASK_18_2_TIMEOUT_S)
    assert wrapper._fatal_protocol_completed is True
    assert wrapper._fatal_publications == 1
    assert gate.phase is _TerminalPhase.FATAL_PENDING
    reporter, reporter_results, reporter_errors, reporter_done = (
        _start_task_18_2_call(
            lambda: runtime._enter_communicator_fatal(primary),
            name="task-18.2-protocol-reporter",
        )
    )
    try:
        assert reporter_joining.wait(_TASK_18_2_TIMEOUT_S)
        assert not reporter_done.is_set()
        assert gate.phase is _TerminalPhase.FATAL_PENDING
    finally:
        release_gate_publish.set()
        _join_task_18_2_call(owner, owner_done)
        _join_task_18_2_call(reporter, reporter_done)

    assert owner_errors == []
    assert reporter_errors == []
    assert owner_results == [primary]
    assert reporter_results == [primary]
    assert gate.phase is _TerminalPhase.FATAL_PUBLISHED


def test_legacy_callable_adopts_active_operator_primary_before_publication(
    monkeypatch,
):
    from renormalizer.backend._distributed.async_owner import AsyncResourceOwner

    runtime, wrapper, backend, _, raw_comm = _single_rank_task_18_2_runtime(
        monkeypatch
    )
    earlier = RuntimeError("legacy earlier operator primary")
    later = RuntimeError("legacy later communicator failure")

    class OperatorCall:
        def __init__(self):
            self.primary_error = earlier
            self.secondary_errors = []

        def record_secondary(self, error):
            if error is not self.primary_error and error not in self.secondary_errors:
                self.secondary_errors.append(error)

    call = OperatorCall()
    owner = AsyncResourceOwner("legacy-compute")
    owner._operator_call = call
    owner.mark_enqueued()
    owner.force_quarantine(earlier)
    lease = SimpleNamespace(
        _active_operator_owner=owner,
        _status_workspace=None,
        _poisoned_error=None,
    )
    provider = SimpleNamespace(_active_lease=lease, _terminal_error=None)
    runtime._active_provider = provider
    sentinels = []
    original_store_set = wrapper._fatal_store_set

    def record_store(key, value, **kwargs):
        if key == wrapper._fatal_key(wrapper.rank):
            sentinels.append(
                (
                    "store",
                    wrapper._fatal_pending_primary,
                    runtime._terminal_gate._fatal_transition.primary,
                )
            )
        return original_store_set(key, value, **kwargs)

    monkeypatch.setattr(wrapper, "_fatal_store_set", record_store)
    original_abort = wrapper._abort_local_communicator

    def record_abort(**kwargs):
        sentinels.append(
            (
                "abort",
                wrapper._fatal_pending_primary,
                runtime._terminal_gate._fatal_transition.primary,
            )
        )
        return original_abort(**kwargs)

    monkeypatch.setattr(wrapper, "_abort_local_communicator", record_abort)
    callbacks = []

    def legacy_handler(error):
        callbacks.append(error)
        sentinels.append(
            (
                "handler",
                wrapper._fatal_pending_primary,
                runtime._terminal_gate._fatal_transition.primary,
            )
        )
        return runtime._enter_communicator_fatal(error)

    wrapper._install_fatal_handler(legacy_handler)

    result = wrapper._enter_observed_fatal(later, wrapper.rank, join_existing=True)

    assert result is earlier
    assert callbacks == [earlier]
    assert sentinels == [
        ("store", earlier, earlier),
        ("abort", earlier, earlier),
        ("handler", earlier, earlier),
    ]
    assert runtime._terminal_gate._fatal_transition.primary is earlier
    assert wrapper._fatal_pending_primary is earlier
    assert wrapper._fatal_error is earlier
    assert runtime._terminal_error is earlier
    assert provider._terminal_error is earlier
    assert lease._poisoned_error is earlier
    assert backend._execution_terminal_error is earlier
    assert call.secondary_errors == [later]
    assert wrapper._fatal_secondary_errors == (later,)
    assert raw_comm.abort_calls == 1


def _install_task_18_2_monitor_barriers(monkeypatch, wrapper):
    before_stop_observation = threading.Event()
    release_stop_observation = threading.Event()
    stop_requested = threading.Event()
    outcome_selected = threading.Event()
    release_outcome = threading.Event()
    selected = []

    original_observe = getattr(wrapper, "_observe_fatal_monitor_stop", lambda: None)

    def observe_stop():
        before_stop_observation.set()
        assert release_stop_observation.wait(_TASK_18_2_TIMEOUT_S)
        return original_observe()

    monkeypatch.setattr(
        wrapper, "_observe_fatal_monitor_stop", observe_stop, raising=False
    )
    original_request = getattr(
        wrapper, "_request_fatal_monitor_stop", lambda: None
    )

    def request_stop():
        generation = original_request()
        stop_requested.set()
        return generation

    monkeypatch.setattr(
        wrapper, "_request_fatal_monitor_stop", request_stop, raising=False
    )
    original_select = getattr(
        wrapper, "_select_fatal_monitor_outcome", lambda kind, value: None
    )

    def select_outcome(kind, value, **kwargs):
        outcome = original_select(kind, value, **kwargs)
        selected.append(outcome)
        outcome_selected.set()
        assert release_outcome.wait(_TASK_18_2_TIMEOUT_S)
        return outcome

    monkeypatch.setattr(
        wrapper, "_select_fatal_monitor_outcome", select_outcome, raising=False
    )
    return SimpleNamespace(
        before_stop_observation=before_stop_observation,
        release_stop_observation=release_stop_observation,
        stop_requested=stop_requested,
        outcome_selected=outcome_selected,
        release_outcome=release_outcome,
        selected=selected,
    )


def test_local_fatal_waits_for_delayed_monitor_exit_before_publication(
    monkeypatch,
):
    from renormalizer.backend._distributed.terminal import _TerminalPhase

    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    monitor_between_reads = threading.Event()
    release_monitor = threading.Event()
    outcome_selected = threading.Event()
    release_selection = threading.Event()
    monitor_acknowledged = threading.Event()
    store_read_phases = []
    monitor_wait_calls = 0
    monitor_wait_lock = threading.Lock()
    publisher = None

    original_read = wrapper._read_fatal_origin

    def record_monitor_store_read(**kwargs):
        with gate._condition:
            store_read_phases.append(gate._phase)
        return original_read(**kwargs)

    monkeypatch.setattr(wrapper, "_read_fatal_origin", record_monitor_store_read)
    original_monitor_wait = wrapper._fatal_monitor_stop.wait

    def wait_between_monitor_reads(timeout=None):
        nonlocal monitor_wait_calls
        if threading.current_thread().name == "renormalizer-fatal-monitor-rank-0":
            with monitor_wait_lock:
                monitor_wait_calls += 1
                first_wait = monitor_wait_calls == 1
            if first_wait:
                monitor_between_reads.set()
                assert release_monitor.wait(_TASK_18_2_TIMEOUT_S * 2)
                return wrapper._fatal_monitor_stop.is_set()
        return original_monitor_wait(timeout)

    monkeypatch.setattr(
        wrapper._fatal_monitor_stop, "wait", wait_between_monitor_reads
    )
    original_select = wrapper._select_fatal_monitor_outcome

    def select_outcome(kind, value, **kwargs):
        outcome = original_select(kind, value, **kwargs)
        if threading.current_thread().name == "task-18.2-delayed-monitor-owner":
            outcome_selected.set()
            assert release_selection.wait(_TASK_18_2_TIMEOUT_S * 2)
        return outcome

    monkeypatch.setattr(wrapper, "_select_fatal_monitor_outcome", select_outcome)
    original_acknowledge = getattr(
        wrapper, "_acknowledge_fatal_monitor_exit", lambda outcome: outcome
    )

    def acknowledge_monitor_exit(outcome):
        result = original_acknowledge(outcome)
        monitor_acknowledged.set()
        return result

    monkeypatch.setattr(
        wrapper,
        "_acknowledge_fatal_monitor_exit",
        acknowledge_monitor_exit,
        raising=False,
    )

    wrapper._start_fatal_monitor()
    monitor = wrapper._fatal_monitor_thread
    assert monitor_between_reads.wait(_TASK_18_2_TIMEOUT_S)
    primary = RuntimeError("local fatal with delayed monitor")
    publisher, results, errors, done = _start_task_18_2_call(
        lambda: runtime._enter_communicator_fatal(primary),
        name="task-18.2-delayed-monitor-owner",
    )
    try:
        assert outcome_selected.wait(_TASK_18_2_TIMEOUT_S)
        with gate._condition:
            assert gate._phase is _TerminalPhase.FATAL_PENDING
        release_monitor.set()
        monitor.join(_TASK_18_2_TIMEOUT_S)
        assert not monitor.is_alive()
        assert monitor_acknowledged.wait(_TASK_18_2_TIMEOUT_S)
        assert store_read_phases == [_TerminalPhase.HEALTHY]
        release_selection.set()
        _join_task_18_2_call(publisher, done)
    finally:
        release_monitor.set()
        release_selection.set()
        if publisher is not None and publisher.is_alive():
            _join_task_18_2_call(publisher, done)
        if monitor.is_alive():
            monitor.join(_TASK_18_2_TIMEOUT_S)

    assert errors == []
    assert results == [primary]
    assert gate._fatal_transition.primary is primary
    assert gate.phase is _TerminalPhase.FATAL_PUBLISHED
    assert all(
        phase not in {_TerminalPhase.FATAL_PUBLISHED, _TerminalPhase.RUNTIME_CLOSED}
        for phase in store_read_phases
    )


def test_close_ready_fatal_joins_monitor_before_publication_and_prevents_late_store_access(
    monkeypatch,
):
    from renormalizer.backend._distributed.terminal import _TerminalPhase

    runtime, wrapper, _, store, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    monitor_waiting = threading.Event()
    release_monitor = threading.Event()
    outcome_selected = threading.Event()
    store_accesses = []
    selected_outcomes = []
    exit_waits = []
    joins = []
    closer = None

    original_monitor = wrapper._monitor_fatal_records

    def delayed_monitor():
        monitor_waiting.set()
        assert release_monitor.wait(_TASK_18_2_TIMEOUT_S)
        original_monitor()

    monitor = threading.Thread(
        target=delayed_monitor,
        name="task-18.2-delayed-close-ready-monitor",
        daemon=True,
    )
    wrapper._fatal_monitor_thread = monitor
    monitor.start()
    assert monitor_waiting.wait(_TASK_18_2_TIMEOUT_S)
    store.set(wrapper._fatal_key(wrapper.rank), 1)

    def record_store_access(operation):
        with gate._condition:
            store_accesses.append((operation, gate._phase))

    original_store_get = wrapper._fatal_store_get

    def store_get(key, **kwargs):
        record_store_access(("get", key))
        return original_store_get(key, **kwargs)

    monkeypatch.setattr(wrapper, "_fatal_store_get", store_get)
    original_store_set = wrapper._fatal_store_set

    def store_set(key, value, **kwargs):
        record_store_access(("set", key))
        return original_store_set(key, value, **kwargs)

    monkeypatch.setattr(wrapper, "_fatal_store_set", store_set)
    original_read = wrapper._read_fatal_origin

    def read_fatal_origin(**kwargs):
        record_store_access("monitor_read")
        return original_read(**kwargs)

    monkeypatch.setattr(wrapper, "_read_fatal_origin", read_fatal_origin)
    original_select = wrapper._select_fatal_monitor_outcome

    def select_outcome(kind, value, **kwargs):
        outcome = original_select(kind, value, **kwargs)
        selected_outcomes.append(outcome)
        outcome_selected.set()
        return outcome

    monkeypatch.setattr(wrapper, "_select_fatal_monitor_outcome", select_outcome)
    original_wait_for_exit = getattr(
        wrapper, "_wait_for_fatal_monitor_exit", lambda outcome, **_kwargs: outcome
    )

    def wait_for_exit(outcome, **kwargs):
        exit_waits.append(outcome)
        return original_wait_for_exit(outcome, **kwargs)

    monkeypatch.setattr(
        wrapper, "_wait_for_fatal_monitor_exit", wait_for_exit, raising=False
    )
    original_join = wrapper._join_fatal_monitor

    def join_monitor(thread, **kwargs):
        joins.append(thread)
        return original_join(thread, **kwargs)

    monkeypatch.setattr(wrapper, "_join_fatal_monitor", join_monitor)

    closer, close_results, close_errors, close_done = _start_task_18_2_call(
        runtime.close, name="task-18.2-close-ready-fatal"
    )
    try:
        assert outcome_selected.wait(_TASK_18_2_TIMEOUT_S)
        release_monitor.set()
        _join_task_18_2_call(closer, close_done)
        monitor.join(_TASK_18_2_TIMEOUT_S)
        assert not monitor.is_alive()
    finally:
        release_monitor.set()
        if closer is not None and closer.is_alive():
            _join_task_18_2_call(closer, close_done)
        if monitor.is_alive():
            monitor.join(_TASK_18_2_TIMEOUT_S)

    primary = gate._fatal_transition.primary
    assert close_results == []
    assert close_errors == [primary]
    assert selected_outcomes[0].kind == "fatal_elected"
    assert selected_outcomes[0].primary is primary
    assert exit_waits == [selected_outcomes[0]]
    assert joins == [monitor]
    assert runtime._closed is True
    assert gate.phase is _TerminalPhase.RUNTIME_CLOSED
    assert all(
        phase not in {_TerminalPhase.FATAL_PUBLISHED, _TerminalPhase.RUNTIME_CLOSED}
        for _, phase in store_accesses
    )


def test_monitor_fatal_before_stop_observation_preempts_close(monkeypatch):
    runtime, wrapper, _, store, raw_comm = _single_rank_task_18_2_runtime(monkeypatch)
    barriers = _install_task_18_2_monitor_barriers(monkeypatch, wrapper)
    gate = runtime._terminal_gate
    close_joining_monitor = threading.Event()
    closer = None
    original_join = getattr(
        wrapper, "_join_fatal_monitor", lambda thread, **_kwargs: None
    )

    def observe_join(thread, **kwargs):
        close_joining_monitor.set()
        return original_join(thread, **kwargs)

    monkeypatch.setattr(wrapper, "_join_fatal_monitor", observe_join, raising=False)
    wrapper._start_fatal_monitor()
    close_results = []
    close_errors = []
    close_done = threading.Event()
    try:
        assert barriers.before_stop_observation.wait(_TASK_18_2_TIMEOUT_S)
        closer, close_results, close_errors, close_done = _start_task_18_2_call(
            runtime.close, name="task-18.2-fatal-before-monitor-stop"
        )
        assert barriers.stop_requested.wait(_TASK_18_2_TIMEOUT_S)
        store.set(wrapper._fatal_key(wrapper.rank), 1)
        barriers.release_stop_observation.set()
        assert barriers.outcome_selected.wait(_TASK_18_2_TIMEOUT_S)
        outcome = barriers.selected[0]
        assert outcome.kind == "fatal_elected"
        assert outcome.primary is gate._fatal_transition.primary
        barriers.release_outcome.set()
        assert close_joining_monitor.wait(_TASK_18_2_TIMEOUT_S)
        _join_task_18_2_call(closer, close_done)
    finally:
        barriers.release_stop_observation.set()
        barriers.release_outcome.set()
        wrapper._fatal_monitor_stop.set()
        thread = wrapper._fatal_monitor_thread
        if thread is not None:
            thread.join(_TASK_18_2_TIMEOUT_S)
        if closer is not None and closer.is_alive():
            _join_task_18_2_call(closer, close_done)

    assert close_results == []
    assert close_errors == [gate._fatal_transition.primary]
    assert runtime._terminal_error is gate._fatal_transition.primary
    assert wrapper._fatal_error is gate._fatal_transition.primary
    assert raw_comm.abort_calls == 1


def test_monitor_clean_stop_wins_before_late_fatal(monkeypatch):
    from renormalizer.backend._distributed.terminal import _TerminalPhase

    runtime, wrapper, backend, store, raw_comm = _single_rank_task_18_2_runtime(
        monkeypatch
    )
    barriers = _install_task_18_2_monitor_barriers(monkeypatch, wrapper)
    wrapper._start_fatal_monitor()
    closer = None
    close_results = []
    close_errors = []
    close_done = threading.Event()
    try:
        assert barriers.before_stop_observation.wait(_TASK_18_2_TIMEOUT_S)
        closer, close_results, close_errors, close_done = _start_task_18_2_call(
            runtime.close, name="task-18.2-clean-monitor-stop"
        )
        assert barriers.stop_requested.wait(_TASK_18_2_TIMEOUT_S)
        barriers.release_stop_observation.set()
        assert barriers.outcome_selected.wait(_TASK_18_2_TIMEOUT_S)
        outcome = barriers.selected[0]
        assert outcome.kind == "stopped_clean"
        store.set(wrapper._fatal_key(wrapper.rank), 1)
        barriers.release_outcome.set()
        _join_task_18_2_call(closer, close_done)
    finally:
        barriers.release_stop_observation.set()
        barriers.release_outcome.set()
        wrapper._fatal_monitor_stop.set()
        thread = wrapper._fatal_monitor_thread
        if thread is not None:
            thread.join(_TASK_18_2_TIMEOUT_S)
        if closer is not None and closer.is_alive():
            _join_task_18_2_call(closer, close_done)

    assert close_errors == []
    assert close_results == [None]
    assert runtime._terminal_gate.phase is _TerminalPhase.RUNTIME_CLOSED
    assert runtime._terminal_error is None
    assert wrapper._fatal_error is None
    assert raw_comm.abort_calls == 0
    assert backend.stop_calls == 1


def test_monitor_ack_and_join_hold_no_gate_admission_or_publication_lock(
    monkeypatch,
):
    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    observations = []
    original_wait = getattr(
        wrapper, "_wait_for_fatal_monitor_exit", lambda outcome, **_kwargs: outcome
    )

    def checked_wait(outcome, **kwargs):
        with gate._condition:
            holds_admission = threading.get_ident() in gate._thread_tokens
        observations.append(
            (
                "ack",
                holds_admission,
                wrapper._fatal_publication_lock._is_owned(),
                wrapper._fatal_lock._is_owned(),
                wrapper._close_lock.locked(),
            )
        )
        return original_wait(outcome, **kwargs)

    monkeypatch.setattr(
        wrapper, "_wait_for_fatal_monitor_exit", checked_wait, raising=False
    )
    original_join = getattr(
        wrapper, "_join_fatal_monitor", lambda thread, **_kwargs: None
    )

    def checked_join(thread, **kwargs):
        with gate._condition:
            holds_admission = threading.get_ident() in gate._thread_tokens
        observations.append(
            (
                "join",
                holds_admission,
                wrapper._fatal_publication_lock._is_owned(),
                wrapper._fatal_lock._is_owned(),
                wrapper._close_lock.locked(),
            )
        )
        return original_join(thread, **kwargs)

    monkeypatch.setattr(wrapper, "_join_fatal_monitor", checked_join, raising=False)
    wrapper._start_fatal_monitor()

    primary = RuntimeError("fatal monitor lock ordering")

    assert runtime._enter_communicator_fatal(primary) is primary

    assert observations == [
        ("ack", False, False, False, False),
        ("join", False, False, False, False),
    ]


def test_staggered_close_stops_rank_zero_store_after_all_close_consumers():
    store = _SharedStore(2)
    raw_comms = [_RawComm(), _RawComm()]
    pairs = [_cpu_collective(rank, 2, store, raw_comms[rank]) for rank in range(2)]
    wrappers = [pair[0] for pair in pairs]
    backends = [pair[1] for pair in pairs]
    _bootstrap_cpu_wrappers(wrappers)
    errors = [None, None]

    def close(rank):
        try:
            wrappers[rank].close()
        except BaseException as error:
            errors[rank] = error

    threads = [threading.Thread(target=close, args=(rank,)) for rank in range(2)]
    try:
        threads[0].start()
        threads[0].join(0.05)
        assert threads[0].is_alive()
        assert wrappers[0]._fatal_monitor_stop.is_set() is False
        assert backends[0].stop_calls == 0

        threads[1].start()
        for thread in threads:
            thread.join(5.0)
        assert all(not thread.is_alive() for thread in threads)
        assert errors == [None, None]
        assert [backend.stop_calls for backend in backends] == [1, 1]
        assert wrappers[0]._closed is True
        assert wrappers[1]._closed is True
    finally:
        for thread in threads:
            if thread.ident is not None:
                thread.join(5.0)
        for wrapper in wrappers:
            if not wrapper._closed:
                wrapper.close()


def test_nonzero_fatal_origin_cannot_be_overtaken_by_rank_zero_close(monkeypatch):
    store = _SharedStore(2)
    raw_comms = [_RawComm(), _RawComm()]
    pairs = [_cpu_collective(rank, 2, store, raw_comms[rank]) for rank in range(2)]
    wrappers = [pair[0] for pair in pairs]
    backends = [pair[1] for pair in pairs]
    _bootstrap_cpu_wrappers(wrappers)
    origin_waiting = threading.Event()
    release_origin = threading.Event()
    close_waiting = threading.Event()
    original_store_set = wrappers[1]._fatal_store_set

    def staggered_origin_ack(key, value, **kwargs):
        if key == wrappers[1]._fatal_ack_key(1):
            origin_waiting.set()
            assert release_origin.wait(3.0)
        return original_store_set(key, value, **kwargs)

    monkeypatch.setattr(
        wrappers[1], "_fatal_store_set", staggered_origin_ack
    )
    original_condition_wait = wrappers[0]._fatal_condition.wait

    def observe_close_wait(timeout=None):
        close_waiting.set()
        return original_condition_wait(timeout)

    wrappers[0]._fatal_condition.wait = observe_close_wait
    primary = RuntimeError("injected rank-1 communicator failure")
    publish_errors = []
    close_errors = [None, None]

    def publish():
        try:
            wrappers[1]._publish_communicator_fatal(primary)
        except BaseException as error:
            publish_errors.append(error)

    def close(rank):
        try:
            wrappers[rank].close()
        except BaseException as error:
            close_errors[rank] = error

    publisher = threading.Thread(target=publish)
    closers = [threading.Thread(target=close, args=(rank,)) for rank in range(2)]
    try:
        publisher.start()
        assert origin_waiting.wait(3.0)
        assert raw_comms[0].abort_event.wait(3.0)
        assert [raw.abort_calls for raw in raw_comms] == [1, 1]

        closers[0].start()
        assert close_waiting.wait(3.0)
        assert closers[0].is_alive()
        assert wrappers[0]._fatal_monitor_stop.is_set() is True
        assert backends[0].stop_calls == 0

        release_origin.set()
        publisher.join(5.0)
        assert not publisher.is_alive()
        closers[1].start()
        for closer in closers:
            closer.join(5.0)
        assert all(not closer.is_alive() for closer in closers)
        assert publish_errors == []
        assert close_errors == [None, None]
        assert wrappers[1]._fatal_error is primary
        assert wrappers[0]._fatal_origin_rank == 1
        assert [backend.stop_calls for backend in backends] == [0, 0]
    finally:
        release_origin.set()
        publisher.join(5.0)
        for closer in closers:
            if closer.ident is not None:
                closer.join(5.0)
        for wrapper in wrappers:
            if not wrapper._closed:
                wrapper.close()


def test_close_ready_cannot_overtake_unpolled_nonzero_fatal(monkeypatch):
    store = _SharedStore(2)
    raw_comms = [_RawComm(), _RawComm()]
    pairs = [_cpu_collective(rank, 2, store, raw_comms[rank]) for rank in range(2)]
    wrappers = [pair[0] for pair in pairs]
    backends = [pair[1] for pair in pairs]
    _bootstrap_cpu_wrappers(wrappers)

    rank_zero_monitor = wrappers[0]._fatal_monitor_thread
    wrappers[0]._fatal_monitor_stop.set()
    rank_zero_monitor.join(3.0)
    assert not rank_zero_monitor.is_alive()
    wrappers[0]._fatal_monitor_stop.clear()

    monitor_waiting = threading.Event()
    release_monitor = threading.Event()
    original_monitor = wrappers[0]._monitor_fatal_records

    def delayed_monitor():
        monitor_waiting.set()
        assert release_monitor.wait(3.0)
        original_monitor()

    rank_zero_monitor = threading.Thread(target=delayed_monitor, daemon=True)
    wrappers[0]._fatal_monitor_thread = rank_zero_monitor
    rank_zero_monitor.start()
    assert monitor_waiting.wait(3.0)

    primary = RuntimeError("injected rank-1 close-ready race")
    publish_errors = []
    close_errors = [None, None]
    rank_zero_stop_requests = []
    rank_zero_outcome_selected = threading.Event()
    original_request_stop = wrappers[0]._request_fatal_monitor_stop

    def record_stop_request():
        rank_zero_stop_requests.append(None)
        return original_request_stop()

    monkeypatch.setattr(
        wrappers[0], "_request_fatal_monitor_stop", record_stop_request
    )
    original_select_outcome = wrappers[0]._select_fatal_monitor_outcome

    def record_outcome_selection(kind, value, **kwargs):
        outcome = original_select_outcome(kind, value, **kwargs)
        if outcome.kind == "fatal_elected":
            rank_zero_outcome_selected.set()
        return outcome

    monkeypatch.setattr(
        wrappers[0],
        "_select_fatal_monitor_outcome",
        record_outcome_selection,
    )

    def publish():
        try:
            wrappers[1]._publish_communicator_fatal(primary)
        except BaseException as error:
            publish_errors.append(error)

    def close(rank):
        try:
            wrappers[rank].close()
        except BaseException as error:
            close_errors[rank] = error

    publisher = threading.Thread(target=publish, daemon=True)
    closers = [
        threading.Thread(target=close, args=(rank,), daemon=True) for rank in range(2)
    ]
    observed_ack = None
    observed_aborts = None
    try:
        publisher.start()
        assert store.wait_for(lambda values: values.get(wrappers[1]._fatal_key(1)) == 1)
        assert raw_comms[1].abort_event.wait(3.0)
        assert raw_comms[1].abort_calls == 1
        assert raw_comms[0].abort_calls == 0

        for closer in closers:
            closer.start()
        assert rank_zero_outcome_selected.wait(3.0)
        release_monitor.set()
        assert store.wait_for(
            lambda values: values.get(wrappers[0]._fatal_ack_key(0)) == 1
        )
        assert raw_comms[0].abort_event.wait(3.0)
        observed_ack = store.get(wrappers[0]._fatal_ack_key(0))
        observed_aborts = [raw.abort_calls for raw in raw_comms]
    finally:
        if store.get(wrappers[0]._fatal_ack_key(0)) != 1:
            store.set(wrappers[0]._fatal_ack_key(0), 1)
        release_monitor.set()
        publisher.join(5.0)
        for closer in closers:
            if closer.ident is not None:
                closer.join(5.0)
        for wrapper in wrappers:
            if not wrapper._closed:
                wrapper._fatal_control_initialized = False
                wrapper.close()

    assert observed_ack == 1
    assert observed_aborts == [1, 1]
    assert not publisher.is_alive()
    assert all(not closer.is_alive() for closer in closers)
    assert publish_errors == []
    assert close_errors == [None, None]
    assert [backend.stop_calls for backend in backends] == [0, 0]
    assert rank_zero_stop_requests == []


@pytest.mark.parametrize(
    ("operation_name", "backend_method"),
    (
        ("barrier", "barrier"),
        ("broadcast", "broadcast"),
        ("allreduce", "all_reduce"),
        ("allreduce_inplace", "all_reduce"),
        ("reduce_scatter", "reduce_scatter"),
        ("allgather", "all_gather"),
    ),
)
def test_close_intent_waits_for_admitted_operation_fatal_publication(
    monkeypatch, operation_name, backend_method
):
    store = _SharedStore(2)
    raw_comms = [_RawComm(), _RawComm()]
    pairs = [_cpu_collective(rank, 2, store, raw_comms[rank]) for rank in range(2)]
    wrappers = [pair[0] for pair in pairs]
    backends = [pair[1] for pair in pairs]
    _bootstrap_cpu_wrappers(wrappers)

    operation_entered = threading.Event()
    release_operation = threading.Event()
    primary = RuntimeError("injected admitted rank-1 communicator failure")

    def fail_collective(*args, **kwargs):
        operation_entered.set()
        assert release_operation.wait(3.0)
        raise primary

    monkeypatch.setattr(wrappers[1], "_validate_array", lambda *args, **kwargs: None)
    monkeypatch.setattr(backends[1], backend_method, fail_collective)
    callbacks = [[], []]
    handlers = []
    for rank in range(2):

        def handler(error, rank=rank):
            callbacks[rank].append(error)
            wrappers[rank]._publish_communicator_fatal(error)

        handlers.append(handler)
        wrappers[rank]._install_fatal_handler(handler)

    stop_order = []
    for rank, backend in enumerate(backends):
        original_stop = backend.stop

        def record_stop(rank=rank, original_stop=original_stop):
            stop_order.append(rank)
            return original_stop()

        monkeypatch.setattr(backend, "stop", record_stop)

    operation_errors = []
    close_errors = [None, None]

    def run_operation():
        try:
            array = np.ones((2,), dtype=np.float64)
            if operation_name == "barrier":
                wrappers[1].barrier()
            elif operation_name == "broadcast":
                wrappers[1].broadcast(array, root=0)
            elif operation_name in {"allreduce", "allreduce_inplace"}:
                getattr(wrappers[1], operation_name)(array)
            else:
                getattr(wrappers[1], operation_name)(array, axis=0)
        except BaseException as error:
            operation_errors.append(error)

    def close(rank):
        try:
            wrappers[rank].close()
        except BaseException as error:
            close_errors[rank] = error

    operation = threading.Thread(target=run_operation, daemon=True)
    closers = [
        threading.Thread(target=close, args=(rank,), daemon=True) for rank in range(2)
    ]
    try:
        operation.start()
        assert operation_entered.wait(3.0)

        closers[1].start()
        assert store.wait_for(lambda values: wrappers[1]._closing)
        closers[0].start()
        release_operation.set()

        operation.join(5.0)
        for closer in closers:
            closer.join(5.0)
    finally:
        release_operation.set()
        operation.join(5.0)
        for closer in closers:
            if closer.ident is not None:
                closer.join(5.0)
        for wrapper in wrappers:
            if not wrapper._closed:
                wrapper._fatal_control_initialized = False
                wrapper.close()

    assert not operation.is_alive()
    assert all(not closer.is_alive() for closer in closers)
    assert operation_errors == [primary]
    assert close_errors == [None, None]
    assert len(callbacks[0]) == 1
    assert callbacks[0][0].origin_rank == 1
    assert callbacks[1] == [primary]
    assert [raw.abort_calls for raw in raw_comms] == [1, 1]
    assert [
        store.get(wrapper._fatal_key(rank)) for rank, wrapper in enumerate(wrappers)
    ] == [
        1,
        1,
    ]
    assert [
        store.get(wrapper._fatal_ack_key(rank)) for rank, wrapper in enumerate(wrappers)
    ] == [1, 1]
    assert stop_order == []


def test_close_joins_monitor_fatal_needed_by_admitted_peer(monkeypatch):
    store = _SharedStore(2)
    raw_comms = [_RawComm(), _RawComm()]
    pairs = [_cpu_collective(rank, 2, store, raw_comms[rank]) for rank in range(2)]
    wrappers = [pair[0] for pair in pairs]
    backends = [pair[1] for pair in pairs]
    _bootstrap_cpu_wrappers(wrappers)

    operation_entered = [threading.Event(), threading.Event()]
    release_failure = threading.Event()
    release_rank_zero_for_cleanup = threading.Event()
    primary = RuntimeError("injected admitted rank-1 communicator failure")

    def block_rank_zero():
        operation_entered[0].set()
        deadline = time.monotonic() + 5.0
        while not (
            raw_comms[0].abort_event.is_set() or release_rank_zero_for_cleanup.is_set()
        ):
            assert time.monotonic() < deadline
            time.sleep(0.001)

    def fail_rank_one():
        operation_entered[1].set()
        assert release_failure.wait(3.0)
        raise primary

    monkeypatch.setattr(backends[0], "barrier", block_rank_zero)
    monkeypatch.setattr(backends[1], "barrier", fail_rank_one)

    callbacks = [[], []]
    handlers = []
    for rank in range(2):

        def handler(error, rank=rank):
            callbacks[rank].append(error)
            wrappers[rank]._publish_communicator_fatal(error)

        handlers.append(handler)
        wrappers[rank]._install_fatal_handler(handler)

    stop_order = []
    for rank, backend in enumerate(backends):
        original_stop = backend.stop

        def record_stop(rank=rank, original_stop=original_stop):
            stop_order.append(rank)
            return original_stop()

        monkeypatch.setattr(backend, "stop", record_stop)

    hard_exits = []
    for rank, wrapper in enumerate(wrappers):

        def hard_exit(rank=rank):
            hard_exits.append(rank)
            raise AssertionError(
                "unexpected communicator hard exit on rank {}".format(rank)
            )

        monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    operation_errors = [[], []]
    close_errors = [[], []]

    def run_operation(rank):
        try:
            wrappers[rank].barrier()
        except BaseException as error:
            operation_errors[rank].append(error)

    def close(rank):
        try:
            wrappers[rank].close()
        except BaseException as error:
            close_errors[rank].append(error)

    operations = [
        threading.Thread(target=run_operation, args=(rank,), daemon=True)
        for rank in range(2)
    ]
    closers = [
        threading.Thread(target=close, args=(rank,), daemon=True) for rank in range(2)
    ]
    needed_external_cleanup = False
    try:
        for operation in operations:
            operation.start()
        assert all(entered.wait(3.0) for entered in operation_entered)
        assert [wrapper._admitted_operations for wrapper in wrappers] == [1, 1]

        for closer in closers:
            closer.start()
        assert store.wait_for(
            lambda values: all(wrapper._closing for wrapper in wrappers)
        )
        release_failure.set()

        for actor in (*operations, *closers):
            actor.join(1.0)
        needed_external_cleanup = any(
            actor.is_alive() for actor in (*operations, *closers)
        )
    finally:
        release_failure.set()
        if any(actor.is_alive() for actor in (*operations, *closers)):
            store.set(wrappers[0]._fatal_ack_key(0), 1)
            release_rank_zero_for_cleanup.set()
        for actor in (*operations, *closers):
            if actor.ident is not None:
                actor.join(5.0)
        for wrapper in wrappers:
            if not wrapper._closed:
                wrapper._fatal_control_initialized = False
                wrapper.close()

    assert needed_external_cleanup is False
    assert all(not actor.is_alive() for actor in (*operations, *closers))
    assert operation_errors == [[], [primary]]
    assert close_errors == [[], []]
    assert len(callbacks[0]) == 1
    assert callbacks[0][0].origin_rank == 1
    assert callbacks[1] == [primary]
    assert [raw.abort_calls for raw in raw_comms] == [1, 1]
    assert [
        store.get(wrapper._fatal_key(rank)) for rank, wrapper in enumerate(wrappers)
    ] == [
        1,
        1,
    ]
    assert [
        store.get(wrapper._fatal_ack_key(rank)) for rank, wrapper in enumerate(wrappers)
    ] == [1, 1]
    assert [wrapper._admitted_operations for wrapper in wrappers] == [0, 0]
    assert [wrapper._fatal_publications for wrapper in wrappers] == [0, 0]
    assert hard_exits == []
    assert stop_order == []


@pytest.mark.parametrize("fatal", (False, True))
def test_close_waits_for_complete_active_broadcast_agreement(monkeypatch, fatal):
    store = _SharedStore(2)
    raw_comms = [_RawComm(), _RawComm()]
    pairs = [_cpu_collective(rank, 2, store, raw_comms[rank]) for rank in range(2)]
    wrappers = [pair[0] for pair in pairs]
    backends = [pair[1] for pair in pairs]
    _bootstrap_cpu_wrappers(wrappers)
    for wrapper in wrappers:
        monkeypatch.setattr(wrapper, "_validate_array", lambda *args, **kwargs: None)

    primary = RuntimeError("injected active broadcast communicator failure")
    if fatal:
        original_broadcast = backends[1].broadcast

        def fail_after_broadcast(*args, **kwargs):
            original_broadcast(*args, **kwargs)
            raise primary

        monkeypatch.setattr(backends[1], "broadcast", fail_after_broadcast)

    callbacks = [[], []]
    handlers = []
    for rank, wrapper in enumerate(wrappers):

        def handler(error, rank=rank, wrapper=wrapper):
            callbacks[rank].append(error)
            wrapper._publish_communicator_fatal(error)

        handlers.append(handler)
        wrapper._install_fatal_handler(handler)

    agreement_entered = [threading.Event(), threading.Event()]
    release_agreement = threading.Event()
    for rank, wrapper in enumerate(wrappers):
        original_set = wrapper._fatal_store_set

        def pause_active_record(
            key,
            value,
            rank=rank,
            original_set=original_set,
            **kwargs,
        ):
            if key == wrappers[rank]._active_b_key(rank):
                sequence, _ = wrappers[rank]._decode_active_b(value)
                if int(sequence) == 1:
                    agreement_entered[rank].set()
                    assert release_agreement.wait(3.0)
            return original_set(key, value, **kwargs)

        monkeypatch.setattr(wrapper, "_fatal_store_set", pause_active_record)

    hard_exits = []
    for rank, wrapper in enumerate(wrappers):

        def hard_exit(rank=rank):
            hard_exits.append(rank)
            raise AssertionError(
                "unexpected communicator hard exit on rank {}".format(rank)
            )

        monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    stop_order = []
    for rank, backend in enumerate(backends):
        original_stop = backend.stop

        def record_stop(rank=rank, original_stop=original_stop):
            stop_order.append(rank)
            return original_stop()

        monkeypatch.setattr(backend, "stop", record_stop)

    operation_errors = [[], []]
    close_errors = [[], []]

    def run_active_broadcast(rank):
        wrapper = wrappers[rank]
        admission = getattr(wrapper, "_active_broadcast_admission", None)
        boundary = (
            nullcontext((wrapper.broadcast, wrapper._agree_active_broadcast))
            if admission is None
            else admission()
        )
        try:
            with boundary as (broadcast, agree):
                broadcast_error = None
                try:
                    broadcast(np.ones((2,), dtype=np.float64), root=0)
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
            operation_errors[rank].append(error)

    def close(rank):
        try:
            wrappers[rank].close()
        except BaseException as error:
            close_errors[rank].append(error)

    operations = [
        threading.Thread(target=run_active_broadcast, args=(rank,), daemon=True)
        for rank in range(2)
    ]
    closers = [
        threading.Thread(target=close, args=(rank,), daemon=True) for rank in range(2)
    ]
    try:
        for operation in operations:
            operation.start()
        assert all(entered.wait(3.0) for entered in agreement_entered)

        for closer in closers:
            closer.start()
        assert store.wait_for(
            lambda values: all(wrapper._closing for wrapper in wrappers)
        )
        for closer in closers:
            closer.join(0.05)
        assert all(closer.is_alive() for closer in closers)
        assert [backend.stop_calls for backend in backends] == [0, 0]
        assert [wrapper._admitted_operations for wrapper in wrappers] == [1, 1]

        release_agreement.set()
        for actor in (*operations, *closers):
            actor.join(5.0)
    finally:
        release_agreement.set()
        for actor in (*operations, *closers):
            if actor.ident is not None:
                actor.join(5.0)
        for wrapper in wrappers:
            if not wrapper._closed:
                wrapper._fatal_control_initialized = False
                wrapper.close()

    assert all(not actor.is_alive() for actor in (*operations, *closers))
    if fatal:
        assert operation_errors[1] == [primary]
        assert len(operation_errors[0]) == 1
        assert callbacks[1] == [primary]
        assert len(callbacks[0]) == 1
        assert callbacks[0][0].origin_rank == 1
        assert [raw.abort_calls for raw in raw_comms] == [1, 1]
    else:
        assert operation_errors == [[], []]
        assert callbacks == [[], []]
        assert [raw.abort_calls for raw in raw_comms] == [0, 0]
    assert close_errors == [[], []]
    assert [wrapper._admitted_operations for wrapper in wrappers] == [0, 0]
    assert [wrapper._fatal_publications for wrapper in wrappers] == [0, 0]
    assert hard_exits == []
    assert stop_order == ([] if fatal else [1, 0])


def test_active_broadcast_fatal_publication_waits_for_admitted_agreement(
    monkeypatch,
):
    runtime, wrapper, backend, _, raw_comm = _single_rank_task_18_2_runtime(
        monkeypatch
    )
    gate = runtime._terminal_gate
    primary = RuntimeError("active broadcast deferred fatal")
    before_agree = threading.Event()
    release_agree = threading.Event()
    active_store_entered = threading.Event()
    release_active_store = threading.Event()
    close_waiting = threading.Event()
    store_trace = []
    publications = []
    hard_exits = []

    monkeypatch.setattr(wrapper, "_validate_array", lambda *args, **kwargs: None)

    def fail_broadcast(*args, **kwargs):
        raise primary

    monkeypatch.setattr(backend, "broadcast", fail_broadcast)
    original_store_set = wrapper._fatal_store_set
    original_store_get = wrapper._fatal_store_get

    def trace_store_set(key, value, **kwargs):
        if key == wrapper._active_b_key(wrapper.rank):
            active_store_entered.set()
            assert release_active_store.wait(_TASK_18_2_TIMEOUT_S * 2)
        store_trace.append(("set", key, gate.phase, wrapper._fatal_error))
        return original_store_set(key, value, **kwargs)

    def trace_store_get(key, **kwargs):
        store_trace.append(("get", key, gate.phase, wrapper._fatal_error))
        return original_store_get(key, **kwargs)

    monkeypatch.setattr(wrapper, "_fatal_store_set", trace_store_set)
    monkeypatch.setattr(wrapper, "_fatal_store_get", trace_store_get)
    original_publish = gate.publish_fatal

    def record_publication(transition, snapshot):
        publications.append((transition, snapshot, tuple(store_trace)))
        return original_publish(transition, snapshot)

    monkeypatch.setattr(gate, "publish_fatal", record_publication)

    def hard_exit():
        hard_exits.append(True)
        raise AssertionError("unexpected active broadcast hard exit")

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    close_thread_name = "task-18.2-active-b-close"
    original_condition_wait = gate._condition.wait

    def observe_close_wait(timeout=None):
        if threading.current_thread().name == close_thread_name:
            close_waiting.set()
        return original_condition_wait(timeout)

    monkeypatch.setattr(gate._condition, "wait", observe_close_wait)

    def run_active_broadcast():
        with wrapper._active_broadcast_admission() as (broadcast, agree):
            broadcast_error = None
            try:
                broadcast(np.ones((2,), dtype=np.float64), root=0)
            except BaseException as error:
                broadcast_error = error
                assert runtime._enter_communicator_fatal(error) is primary
                before_agree.set()
                assert release_agree.wait(_TASK_18_2_TIMEOUT_S * 2)
            try:
                agree(broadcast_error is not None)
            except BaseException:
                if broadcast_error is not None:
                    raise broadcast_error
                raise
            if broadcast_error is not None:
                raise broadcast_error

    worker, worker_results, worker_errors, worker_done = _start_task_18_2_call(
        run_active_broadcast,
        name="task-18.2-active-b-discoverer",
    )
    closer = None
    try:
        assert before_agree.wait(_TASK_18_2_TIMEOUT_S)
        before_state = (
            gate.phase,
            wrapper._fatal_pending_primary,
            wrapper._fatal_error,
            runtime._terminal_error,
            wrapper._admitted_operations,
            wrapper._fatal_publications,
        )
        closer, close_results, close_errors, close_done = _start_task_18_2_call(
            runtime.close,
            name=close_thread_name,
        )
        assert before_state == (
            _TerminalPhase.FATAL_PENDING,
            primary,
            None,
            None,
            1,
            1,
        )
        assert close_waiting.wait(_TASK_18_2_TIMEOUT_S)
        assert not close_done.is_set()

        release_agree.set()
        assert active_store_entered.wait(_TASK_18_2_TIMEOUT_S)
        assert gate.phase is _TerminalPhase.FATAL_PENDING
        assert wrapper._fatal_error is None
        assert runtime._terminal_error is None
        assert wrapper._admitted_operations == 1
        assert wrapper._fatal_publications == 1
        assert not close_done.is_set()

        release_active_store.set()
        _join_task_18_2_call(worker, worker_done)
        _join_task_18_2_call(closer, close_done)
    finally:
        release_agree.set()
        release_active_store.set()
        if worker.is_alive():
            _join_task_18_2_call(worker, worker_done)
        if closer is not None and closer.is_alive():
            _join_task_18_2_call(closer, close_done)

    active_key = wrapper._active_b_key(wrapper.rank)
    fatal_key = wrapper._fatal_key(wrapper.rank)
    fatal_ack_key = wrapper._fatal_ack_key(wrapper.rank)
    store_keys = [(operation, key) for operation, key, _, _ in store_trace]
    assert worker_results == []
    assert worker_errors == [primary]
    assert close_results == []
    assert close_errors == [primary]
    assert len(publications) == 1
    assert publications[0][0].primary is primary
    assert publications[0][1] is primary
    assert all(
        phase is _TerminalPhase.FATAL_PENDING
        for _, _, phase, _ in store_trace
    )
    assert all(
        collective_error is None
        for _, key, _, collective_error in store_trace
        if key == active_key
    )
    assert store_keys.index(("set", active_key)) < store_keys.index(
        ("set", fatal_key)
    )
    assert max(
        index
        for index, (_, key) in enumerate(store_keys)
        if key == active_key
    ) < store_keys.index(("set", fatal_key))
    assert store_keys.index(("set", fatal_key)) < store_keys.index(
        ("set", fatal_ack_key)
    )
    assert gate.phase is _TerminalPhase.RUNTIME_CLOSED
    assert wrapper._fatal_error is primary
    assert wrapper._fatal_publications == 0
    assert wrapper._admitted_operations == 0
    assert raw_comm.abort_calls == 1
    assert hard_exits == []


def test_active_broadcast_joiner_blocks_existing_owner_until_agreement(
    monkeypatch,
):
    runtime, wrapper, backend, _, raw_comm = _single_rank_task_18_2_runtime(
        monkeypatch
    )
    gate = runtime._terminal_gate
    primary = RuntimeError("existing active-b publication owner")
    later = RuntimeError("joined active-b broadcast failure")
    admission_entered = threading.Event()
    release_broadcast = threading.Event()
    before_agree = threading.Event()
    release_agree = threading.Event()
    active_store_entered = threading.Event()
    release_active_store = threading.Event()
    owner_waiting = threading.Event()
    close_waiting = threading.Event()
    store_trace = []
    publications = []
    hard_exits = []
    owner_name = "task-18.2-active-b-existing-owner"
    close_name = "task-18.2-active-b-joined-close"

    monkeypatch.setattr(wrapper, "_validate_array", lambda *args, **kwargs: None)

    def fail_broadcast(*args, **kwargs):
        raise later

    monkeypatch.setattr(backend, "broadcast", fail_broadcast)
    original_condition_wait = wrapper._fatal_condition.wait

    def observe_publication_wait(timeout=None):
        thread_name = threading.current_thread().name
        if thread_name == owner_name:
            owner_waiting.set()
        return original_condition_wait(timeout)

    monkeypatch.setattr(
        wrapper._fatal_condition, "wait", observe_publication_wait
    )
    original_gate_wait = gate._condition.wait

    def observe_close_wait(timeout=None):
        if threading.current_thread().name == close_name:
            close_waiting.set()
        return original_gate_wait(timeout)

    monkeypatch.setattr(gate._condition, "wait", observe_close_wait)
    original_store_set = wrapper._fatal_store_set
    original_store_get = wrapper._fatal_store_get

    def trace_store_set(key, value, **kwargs):
        if key == wrapper._active_b_key(wrapper.rank):
            active_store_entered.set()
            assert release_active_store.wait(_TASK_18_2_TIMEOUT_S * 2)
        store_trace.append(("set", key, gate.phase, wrapper._fatal_error))
        return original_store_set(key, value, **kwargs)

    def trace_store_get(key, **kwargs):
        store_trace.append(("get", key, gate.phase, wrapper._fatal_error))
        return original_store_get(key, **kwargs)

    monkeypatch.setattr(wrapper, "_fatal_store_set", trace_store_set)
    monkeypatch.setattr(wrapper, "_fatal_store_get", trace_store_get)
    original_publish = gate.publish_fatal

    def record_publication(transition, snapshot):
        publications.append((transition, snapshot, tuple(store_trace)))
        return original_publish(transition, snapshot)

    monkeypatch.setattr(gate, "publish_fatal", record_publication)

    def hard_exit():
        hard_exits.append(threading.current_thread().name)
        raise AssertionError("unexpected joined active-b hard exit")

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    def run_active_broadcast():
        with wrapper._active_broadcast_admission() as (broadcast, agree):
            admission_entered.set()
            assert release_broadcast.wait(_TASK_18_2_TIMEOUT_S * 2)
            broadcast_error = None
            try:
                broadcast(np.ones((2,), dtype=np.float64), root=0)
            except BaseException as error:
                broadcast_error = error
                assert error is primary
                assert runtime._enter_communicator_fatal(error) is primary
                before_agree.set()
                assert release_agree.wait(_TASK_18_2_TIMEOUT_S * 2)
            try:
                agree(broadcast_error is not None)
            except BaseException:
                if broadcast_error is not None:
                    raise broadcast_error
                raise
            if broadcast_error is not None:
                raise broadcast_error

    worker, worker_results, worker_errors, worker_done = _start_task_18_2_call(
        run_active_broadcast,
        name="task-18.2-active-b-joined-discoverer",
    )
    owner = None
    closer = None
    try:
        assert admission_entered.wait(_TASK_18_2_TIMEOUT_S)
        owner, owner_results, owner_errors, owner_done = _start_task_18_2_call(
            lambda: runtime._enter_communicator_fatal(primary),
            name=owner_name,
        )
        assert owner_waiting.wait(_TASK_18_2_TIMEOUT_S)
        assert not owner_done.is_set()

        release_broadcast.set()
        assert before_agree.wait(_TASK_18_2_TIMEOUT_S)
        closer, close_results, close_errors, close_done = _start_task_18_2_call(
            runtime.close,
            name=close_name,
        )
        assert close_waiting.wait(_TASK_18_2_TIMEOUT_S)
        assert gate.phase is _TerminalPhase.FATAL_PENDING
        assert wrapper._fatal_pending_primary is primary
        assert wrapper._fatal_error is None
        assert runtime._terminal_error is None
        assert wrapper._active_broadcast_agreements == 1
        assert wrapper._admitted_operations == 1
        assert wrapper._fatal_publications == 1
        assert not owner_done.is_set()
        assert not close_done.is_set()

        release_agree.set()
        assert active_store_entered.wait(_TASK_18_2_TIMEOUT_S)
        assert gate.phase is _TerminalPhase.FATAL_PENDING
        assert wrapper._fatal_error is None
        assert runtime._terminal_error is None
        assert wrapper._active_broadcast_agreements == 1
        assert wrapper._fatal_publications == 1
        assert not owner_done.is_set()
        assert not close_done.is_set()

        release_active_store.set()
        _join_task_18_2_call(worker, worker_done)
        _join_task_18_2_call(owner, owner_done)
        _join_task_18_2_call(closer, close_done)
    finally:
        release_broadcast.set()
        release_agree.set()
        release_active_store.set()
        if worker.is_alive():
            _join_task_18_2_call(worker, worker_done)
        if owner is not None and owner.is_alive():
            _join_task_18_2_call(owner, owner_done)
        if closer is not None and closer.is_alive():
            _join_task_18_2_call(closer, close_done)

    active_key = wrapper._active_b_key(wrapper.rank)
    fatal_key = wrapper._fatal_key(wrapper.rank)
    store_keys = [(operation, key) for operation, key, _, _ in store_trace]
    assert worker_results == []
    assert worker_errors == [primary]
    assert owner_results == [primary]
    assert owner_errors == []
    assert close_results == []
    assert close_errors == [primary]
    assert len(publications) == 1
    assert publications[0][0].primary is primary
    assert publications[0][1] is primary
    assert all(
        phase is _TerminalPhase.FATAL_PENDING
        for _, _, phase, _ in store_trace
    )
    assert max(
        index
        for index, (_, key) in enumerate(store_keys)
        if key == active_key
    ) < store_keys.index(("set", fatal_key))
    assert gate.phase is _TerminalPhase.RUNTIME_CLOSED
    assert wrapper._fatal_error is primary
    assert wrapper._active_broadcast_agreements == 0
    assert wrapper._admitted_operations == 0
    assert wrapper._fatal_publications == 0
    assert wrapper._fatal_secondary_errors == (later,)
    assert raw_comm.abort_calls == 1
    assert hard_exits == []


def test_legacy_active_broadcast_defers_one_transition_publish_until_agreement(
    monkeypatch,
):
    runtime, wrapper, backend, _, raw_comm = _single_rank_task_18_2_runtime(
        monkeypatch
    )
    gate = runtime._terminal_gate
    primary = RuntimeError("legacy active-b communicator failure")
    before_agree = threading.Event()
    release_agree = threading.Event()
    active_store_entered = threading.Event()
    release_active_store = threading.Event()
    transition_publications = []
    legacy_callbacks = []
    gate_publications = []
    order = []
    hard_exits = []

    monkeypatch.setattr(wrapper, "_validate_array", lambda *args, **kwargs: None)

    def fail_broadcast(*args, **kwargs):
        raise primary

    monkeypatch.setattr(backend, "broadcast", fail_broadcast)
    original_transition_publish = runtime._publish_communicator_fatal_transition

    def record_transition_publish(transition):
        order.append("transition_publish")
        transition_publications.append(
            (
                transition,
                gate.phase,
                runtime._terminal_error,
                getattr(backend, "_execution_terminal_error", None),
                wrapper._fatal_error,
            )
        )
        return original_transition_publish(transition)

    monkeypatch.setattr(
        runtime,
        "_publish_communicator_fatal_transition",
        record_transition_publish,
    )

    def legacy_handler(error):
        order.append("legacy_handler")
        legacy_callbacks.append(
            (
                error,
                gate.phase,
                runtime._terminal_error,
                getattr(backend, "_execution_terminal_error", None),
                wrapper._fatal_error,
            )
        )
        return runtime._enter_communicator_fatal(error)

    wrapper._install_fatal_handler(legacy_handler)
    original_store_set = wrapper._fatal_store_set

    def observe_store_set(key, value, **kwargs):
        if key == wrapper._active_b_key(wrapper.rank):
            active_store_entered.set()
            assert release_active_store.wait(_TASK_18_2_TIMEOUT_S * 2)
            order.append("active_b_set")
        elif key == wrapper._fatal_ack_key(wrapper.rank):
            order.append("fatal_ack")
        return original_store_set(key, value, **kwargs)

    monkeypatch.setattr(wrapper, "_fatal_store_set", observe_store_set)
    original_gate_publish = gate.publish_fatal

    def record_gate_publish(transition, snapshot):
        order.append("gate_publish")
        gate_publications.append((transition, snapshot))
        return original_gate_publish(transition, snapshot)

    monkeypatch.setattr(gate, "publish_fatal", record_gate_publish)

    def hard_exit():
        hard_exits.append(threading.current_thread().name)
        raise AssertionError("unexpected legacy active-b hard exit")

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    def run_active_broadcast():
        with wrapper._active_broadcast_admission() as (broadcast, agree):
            broadcast_error = None
            try:
                broadcast(np.ones((2,), dtype=np.float64), root=0)
            except BaseException as error:
                broadcast_error = error
                assert error is primary
                assert runtime._enter_communicator_fatal(error) is primary
                before_agree.set()
                assert release_agree.wait(_TASK_18_2_TIMEOUT_S * 2)
            try:
                agree(broadcast_error is not None)
            except BaseException:
                if broadcast_error is not None:
                    raise broadcast_error
                raise
            if broadcast_error is not None:
                raise broadcast_error

    worker, worker_results, worker_errors, worker_done = _start_task_18_2_call(
        run_active_broadcast,
        name="task-18.2-legacy-active-b-discoverer",
    )
    try:
        assert before_agree.wait(_TASK_18_2_TIMEOUT_S)
        assert gate.phase is _TerminalPhase.FATAL_PENDING
        assert wrapper._fatal_pending_primary is primary
        assert wrapper._fatal_error is None
        assert runtime._terminal_error is None
        assert getattr(backend, "_execution_terminal_error", None) is None
        assert transition_publications == []
        assert legacy_callbacks == []
        assert wrapper._fatal_publications == 1

        release_agree.set()
        assert active_store_entered.wait(_TASK_18_2_TIMEOUT_S)
        assert gate.phase is _TerminalPhase.FATAL_PENDING
        assert wrapper._fatal_error is None
        assert runtime._terminal_error is None
        assert getattr(backend, "_execution_terminal_error", None) is None
        assert transition_publications == []
        assert legacy_callbacks == []

        release_active_store.set()
        _join_task_18_2_call(worker, worker_done)
    finally:
        release_agree.set()
        release_active_store.set()
        if worker.is_alive():
            _join_task_18_2_call(worker, worker_done)

    assert worker_results == []
    assert worker_errors == [primary]
    assert len(transition_publications) == 1
    transition, phase, runtime_error, backend_error, collective_error = (
        transition_publications[0]
    )
    assert transition.primary is primary
    assert phase is _TerminalPhase.FATAL_PENDING
    assert runtime_error is None
    assert backend_error is None
    assert collective_error is None
    assert legacy_callbacks == [
        (
            primary,
            _TerminalPhase.FATAL_PENDING,
            primary,
            primary,
            None,
        )
    ]
    assert gate_publications == [(transition, primary)]
    assert order.index("active_b_set") < order.index("transition_publish")
    assert order.index("transition_publish") < order.index("legacy_handler")
    assert order.index("legacy_handler") < order.index("fatal_ack")
    assert order.index("fatal_ack") < order.index("gate_publish")
    assert gate.phase is _TerminalPhase.FATAL_PUBLISHED
    assert wrapper._fatal_error is primary
    assert runtime._terminal_error is primary
    assert backend._execution_terminal_error is primary
    assert wrapper._fatal_publications == 0
    assert raw_comm.abort_calls == 1
    assert hard_exits == []


def test_deferred_quarantine_failure_retains_owner_and_fail_stops_close(
    monkeypatch,
):
    from renormalizer.backend._distributed.async_owner import AsyncResourceOwner

    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, backend, _, raw_comm = _single_rank_task_18_2_runtime(
        monkeypatch
    )
    gate = runtime._terminal_gate
    primary = RuntimeError("deferred quarantine fatal primary")
    callback_error = RuntimeError("injected quarantine callback failure")
    before_agree = threading.Event()
    release_agree = threading.Event()
    active_store_entered = threading.Event()
    release_active_store = threading.Event()
    publication_hard_exit = threading.Event()
    close_hard_exit = threading.Event()
    callback_calls = []
    hard_exit_observations = []
    deferrals = []
    worker_name = "task-18.2-deferred-quarantine-owner"
    close_name = "task-18.2-deferred-quarantine-close"

    monkeypatch.setattr(wrapper, "_validate_array", lambda *args, **kwargs: None)

    def fail_broadcast(*args, **kwargs):
        raise primary

    monkeypatch.setattr(backend, "broadcast", fail_broadcast)

    def fail_quarantine(owner):
        callback_calls.append(owner)
        raise callback_error

    owner = AsyncResourceOwner(
        "deferred-quarantine-failure",
        quarantine=fail_quarantine,
    )
    owner.mark_enqueued()
    original_store_set = wrapper._fatal_store_set

    def pause_active_store(key, value, **kwargs):
        if key == wrapper._active_b_key(wrapper.rank):
            active_store_entered.set()
            assert release_active_store.wait(_TASK_18_2_TIMEOUT_S * 2)
        return original_store_set(key, value, **kwargs)

    monkeypatch.setattr(wrapper, "_fatal_store_set", pause_active_store)

    def hard_exit():
        thread_name = threading.current_thread().name
        hard_exit_observations.append(
            (
                thread_name,
                wrapper._fatal_publications,
                getattr(wrapper, "_active_broadcast_agreements", None),
                gate.phase,
                wrapper._fatal_protocol_completed,
                getattr(wrapper, "_fatal_publication_failure", None),
            )
        )
        if thread_name == worker_name:
            publication_hard_exit.set()
        elif thread_name == close_name:
            close_hard_exit.set()
        raise FatalHardExit(thread_name)

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    def run_active_broadcast():
        with wrapper._active_broadcast_admission() as (broadcast, agree):
            broadcast_error = None
            try:
                broadcast(np.ones((2,), dtype=np.float64), root=0)
            except BaseException as error:
                broadcast_error = error
                assert error is primary
                assert runtime._enter_communicator_fatal(error, owner) is primary
                deferrals.append(wrapper._active_broadcast_local.deferral)
                before_agree.set()
                assert release_agree.wait(_TASK_18_2_TIMEOUT_S * 2)
            try:
                agree(broadcast_error is not None)
            except BaseException:
                if broadcast_error is not None:
                    raise broadcast_error
                raise
            if broadcast_error is not None:
                raise broadcast_error

    worker, worker_results, worker_errors, worker_done = _start_task_18_2_call(
        run_active_broadcast,
        name=worker_name,
    )
    closer = None
    try:
        assert before_agree.wait(_TASK_18_2_TIMEOUT_S)
        assert gate.phase is _TerminalPhase.FATAL_PENDING
        assert runtime._terminal_error is None
        assert wrapper._fatal_error is None
        assert callback_calls == []
        assert wrapper._fatal_publications == 1

        release_agree.set()
        assert active_store_entered.wait(_TASK_18_2_TIMEOUT_S)
        assert gate.phase is _TerminalPhase.FATAL_PENDING
        assert callback_calls == []
        assert hard_exit_observations == []
        assert wrapper._fatal_publications == 1

        release_active_store.set()
        assert publication_hard_exit.wait(_TASK_18_2_TIMEOUT_S)
        _join_task_18_2_call(worker, worker_done)
        assert wrapper._fatal_publications == 1
        assert wrapper._fatal_protocol_completed is False
        assert gate.phase is _TerminalPhase.FATAL_PENDING
        assert deferrals[0].completed is False

        closer, close_results, close_errors, close_done = _start_task_18_2_call(
            runtime.close,
            name=close_name,
        )
        _join_task_18_2_call(closer, close_done)
    finally:
        release_agree.set()
        release_active_store.set()
        if worker.is_alive():
            _join_task_18_2_call(worker, worker_done)
        if closer is not None and closer.is_alive():
            _join_task_18_2_call(closer, close_done)

    assert worker_results == []
    assert len(worker_errors) == 1
    assert isinstance(worker_errors[0], FatalHardExit)
    assert close_results == []
    assert len(close_errors) == 1
    assert close_errors[0] is primary
    assert close_hard_exit.is_set() is False
    assert callback_calls == [owner]
    assert owner.state == "quarantined"
    assert owner.error is primary
    assert owner.secondary_errors == ()
    assert runtime._terminal_secondary_errors == ()
    assert wrapper._fatal_secondary_errors == (callback_error,)
    assert wrapper._fatal_publication_failure is primary
    assert hard_exit_observations == [
        (
            worker_name,
            1,
            0,
            _TerminalPhase.FATAL_PENDING,
            False,
            primary,
        ),
    ]
    assert runtime._terminal_error is None
    assert getattr(backend, "_execution_terminal_error", None) is None
    assert wrapper._fatal_error is None
    assert runtime._closed is False
    assert raw_comm.abort_calls == 1


def test_runtime_origin_fatal_reservation_precedes_terminal_visibility_and_close(
    monkeypatch,
):
    from renormalizer.backend._distributed.context import DistributedRendezvous
    from renormalizer.backend._distributed.mesh import DeviceMesh
    from renormalizer.backend.distributed_runtime import CupyDistributedRuntime

    store = _SharedStore(2)
    raw_comms = [_RawComm(), _RawComm()]
    pairs = [_cpu_collective(rank, 2, store, raw_comms[rank]) for rank in range(2)]
    wrappers = [pair[0] for pair in pairs]
    backends = [pair[1] for pair in pairs]
    _bootstrap_cpu_wrappers(wrappers)
    runtimes = [
        CupyDistributedRuntime(
            backend=backends[rank],
            context=wrappers[rank]._context,
            rendezvous=DistributedRendezvous("127.0.0.1", 23456),
            mesh=DeviceMesh((2,), ("rank",), rank),
            collective=wrappers[rank],
        )
        for rank in range(2)
    ]
    callbacks = [[], []]
    callback_entered = [threading.Event(), threading.Event()]
    handlers = []
    for rank, (runtime, wrapper) in enumerate(zip(runtimes, wrappers)):

        def handler(error, rank=rank, runtime=runtime):
            callbacks[rank].append(error)
            callback_entered[rank].set()
            return runtime._enter_communicator_fatal(error)

        handlers.append(handler)
        wrapper._install_fatal_handler(handler)

    publication_starts = [[], []]
    for rank, wrapper in enumerate(wrappers):
        original_begin = wrapper._begin_fatal_publication

        def record_begin(
            *args, rank=rank, wrapper=wrapper, original=original_begin, **kwargs
        ):
            result = original(*args, **kwargs)
            if result[2]:
                publication_starts[rank].append(wrapper._fatal_publications)
            return result

        monkeypatch.setattr(wrapper, "_begin_fatal_publication", record_begin)

    handoff_entered = threading.Event()
    release_handoff = threading.Event()
    original_publish = runtimes[1]._publish_communicator_fatal_transition

    def pause_runtime_handoff(transition):
        handoff_entered.set()
        assert release_handoff.wait(3.0)
        return original_publish(transition)

    monkeypatch.setattr(
        runtimes[1], "_publish_communicator_fatal_transition", pause_runtime_handoff
    )

    hard_exits = []
    for rank, wrapper in enumerate(wrappers):

        def hard_exit(rank=rank):
            hard_exits.append(rank)
            raise AssertionError(
                "unexpected communicator hard exit on rank {}".format(rank)
            )

        monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    stop_order = []
    for rank, backend in enumerate(backends):
        original_stop = backend.stop

        def record_stop(rank=rank, original=original_stop):
            stop_order.append(rank)
            return original()

        monkeypatch.setattr(backend, "stop", record_stop)

    primary = RuntimeError("injected direct runtime communicator failure")
    fatal_errors = []
    close_errors = [[], []]

    def enter_fatal():
        try:
            assert runtimes[1]._enter_communicator_fatal(primary) is primary
        except BaseException as error:
            fatal_errors.append(error)

    def close(rank):
        try:
            runtimes[rank].close()
        except BaseException as error:
            close_errors[rank].append(error)

    publisher = threading.Thread(target=enter_fatal, daemon=True)
    closers = [
        threading.Thread(target=close, args=(rank,), daemon=True) for rank in range(2)
    ]
    close_waiting_for_publication = threading.Event()
    original_gate_wait = runtimes[1]._terminal_gate._condition.wait

    def observe_gate_wait(timeout=None):
        if threading.current_thread() is closers[1]:
            close_waiting_for_publication.set()
        return original_gate_wait(timeout)

    runtimes[1]._terminal_gate._condition.wait = observe_gate_wait
    state_at_visibility = None
    callback_before_close = None
    close_joined_reservation = None
    try:
        publisher.start()
        assert handoff_entered.wait(3.0)
        state_at_visibility = (
            wrappers[1]._fatal_publications,
            wrappers[1]._fatal_abort_started,
            store.get(wrappers[1]._fatal_key(1)),
        )
        callback_before_close = callback_entered[0].wait(0.5)

        for closer in closers:
            closer.start()
        assert close_waiting_for_publication.wait(3.0)
        close_joined_reservation = closers[1].is_alive()
        assert runtimes[1]._terminal_error is None
        assert wrappers[1]._fatal_error is None
        assert runtimes[1]._closed is False

        release_handoff.set()
        publisher.join(5.0)
        for closer in closers:
            closer.join(5.0)
    finally:
        release_handoff.set()
        if publisher.ident is not None:
            publisher.join(5.0)
        for closer in closers:
            if closer.ident is not None:
                closer.join(5.0)
        for runtime in runtimes:
            if not runtime._closed:
                try:
                    runtime.close()
                except BaseException:
                    pass

    assert state_at_visibility == (1, True, 1)
    assert callback_before_close is True
    assert close_joined_reservation is True
    assert not publisher.is_alive()
    assert all(not closer.is_alive() for closer in closers)
    assert fatal_errors == []
    assert callbacks[0][0].origin_rank == 1
    assert callbacks[1] == []
    assert len(callbacks[0]) == 1
    assert publication_starts == [[1], [1]]
    assert [raw.abort_calls for raw in raw_comms] == [1, 1]
    assert [
        store.get(wrapper._fatal_key(rank)) for rank, wrapper in enumerate(wrappers)
    ] == [
        1,
        1,
    ]
    assert [
        store.get(wrapper._fatal_ack_key(rank)) for rank, wrapper in enumerate(wrappers)
    ] == [1, 1]
    assert [wrapper._admitted_operations for wrapper in wrappers] == [0, 0]
    assert [wrapper._fatal_publications for wrapper in wrappers] == [0, 0]
    assert runtimes[1]._terminal_error is primary
    assert close_errors == [
        [runtimes[0]._terminal_error],
        [primary],
    ]
    assert hard_exits == []
    assert stop_order == []


def test_failed_owner_terminal_slot_holds_fatal_reservation_against_close(monkeypatch):
    from renormalizer.backend._distributed.async_owner import AsyncResourceOwner
    from renormalizer.backend._distributed.context import DistributedRendezvous
    from renormalizer.backend._distributed.mesh import DeviceMesh
    from renormalizer.backend._distributed.providers import ActiveWorkingSetProvider
    from renormalizer.backend.distributed_runtime import CupyDistributedRuntime

    store = _SharedStore(2)
    raw_comms = [_RawComm(), _RawComm()]
    pairs = [_cpu_collective(rank, 2, store, raw_comms[rank]) for rank in range(2)]
    wrappers = [pair[0] for pair in pairs]
    backends = [pair[1] for pair in pairs]
    _bootstrap_cpu_wrappers(wrappers)
    runtimes = [
        CupyDistributedRuntime(
            backend=backends[rank],
            context=wrappers[rank]._context,
            rendezvous=DistributedRendezvous("127.0.0.1", 23456),
            mesh=DeviceMesh((2,), ("rank",), rank),
            collective=wrappers[rank],
        )
        for rank in range(2)
    ]
    budget = SimpleNamespace(resolved_bytes=0)
    providers = [
        ActiveWorkingSetProvider(
            runtime,
            device_budget_resolution=budget,
            host_budget_resolution=budget,
            _standalone=True,
        )
        for runtime in runtimes
    ]
    for runtime in runtimes:
        runtime._arm_communicator_fatal()

    primary = RuntimeError("injected failed owner primary")
    drain_error = RuntimeError("injected failed owner drain failure")
    cohort_owner = AsyncResourceOwner("h2d")
    cohort_owner.force_quarantine(primary)
    failed_owner = AsyncResourceOwner(
        "compute",
        drainer=lambda: (_ for _ in ()).throw(drain_error),
        quarantine=providers[1]._accept_async_quarantine,
    )
    failed_owner.mark_enqueued()
    providers[1]._active_lease = SimpleNamespace(
        scheduler=SimpleNamespace(_quarantined_owners=[cohort_owner]),
        pool=None,
        _poisoned_error=None,
        _active_operator_owner=None,
        _status_workspace=None,
        close=lambda: None,
    )

    terminal_slot_visible = threading.Event()
    release_quarantine = threading.Event()
    retain_calls = []
    original_retain = runtimes[1]._terminal_quarantine.retain

    def pause_first_retain(owner, error=None):
        retain_calls.append((owner, error))
        if len(retain_calls) == 1:
            terminal_slot_visible.set()
            assert release_quarantine.wait(3.0)
        return original_retain(owner, error)

    monkeypatch.setattr(runtimes[1]._terminal_quarantine, "retain", pause_first_retain)

    publication_starts = [[], []]
    remote_publication_started = threading.Event()
    control_writes = [[], []]
    for rank, wrapper in enumerate(wrappers):
        original_begin = wrapper._begin_fatal_publication
        original_set = wrapper._fatal_store_set

        def record_begin(
            *args, rank=rank, wrapper=wrapper, original=original_begin, **kwargs
        ):
            result = original(*args, **kwargs)
            if result[2]:
                publication_starts[rank].append(wrapper._fatal_publications)
                if rank == 0:
                    remote_publication_started.set()
            return result

        def record_set(
            key,
            value,
            rank=rank,
            original=original_set,
            **kwargs,
        ):
            if int(value) == 1:
                control_writes[rank].append(key)
            return original(key, value, **kwargs)

        monkeypatch.setattr(wrapper, "_begin_fatal_publication", record_begin)
        monkeypatch.setattr(wrapper, "_fatal_store_set", record_set)

    hard_exits = []
    stop_order = []
    for rank, (wrapper, backend) in enumerate(zip(wrappers, backends)):
        original_stop = backend.stop

        def hard_exit(rank=rank):
            hard_exits.append(rank)
            raise AssertionError(
                "unexpected communicator hard exit on rank {}".format(rank)
            )

        def record_stop(rank=rank, original=original_stop):
            stop_order.append(rank)
            return original()

        monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)
        monkeypatch.setattr(backend, "stop", record_stop)

    operation_errors = []
    close_errors = [[], []]

    def fail_owner():
        try:
            failed_owner.fail(primary)
        except BaseException as error:
            operation_errors.append(error)

    def close(rank):
        try:
            runtimes[rank].close()
        except BaseException as error:
            close_errors[rank].append(error)

    operation = threading.Thread(target=fail_owner, daemon=True)
    closers = [
        threading.Thread(target=close, args=(rank,), daemon=True) for rank in range(2)
    ]
    close_waiting_for_publication = threading.Event()
    original_gate_wait = runtimes[1]._terminal_gate._condition.wait

    def observe_gate_wait(timeout=None):
        if threading.current_thread() is closers[1]:
            close_waiting_for_publication.set()
        return original_gate_wait(timeout)

    runtimes[1]._terminal_gate._condition.wait = observe_gate_wait
    state_at_first_terminal = None
    close_joined_reservation = None
    try:
        operation.start()
        assert terminal_slot_visible.wait(3.0)
        state_at_first_terminal = (
            providers[1]._terminal_error,
            runtimes[1]._terminal_error,
            getattr(backends[1], "_execution_terminal_error", None),
            wrappers[1]._fatal_publications,
            wrappers[1]._fatal_abort_started,
            store.get(wrappers[1]._fatal_key(1)),
        )
        assert remote_publication_started.wait(3.0)

        for closer in closers:
            closer.start()
        assert close_waiting_for_publication.wait(3.0)
        close_joined_reservation = closers[1].is_alive()

        release_quarantine.set()
        operation.join(5.0)
        for closer in closers:
            closer.join(5.0)
    finally:
        release_quarantine.set()
        if operation.ident is not None:
            operation.join(5.0)
        for closer in closers:
            if closer.ident is not None:
                closer.join(5.0)
        for runtime in runtimes:
            if not runtime._closed:
                try:
                    runtime.close()
                except BaseException:
                    pass

    assert state_at_first_terminal == (primary, None, None, 1, True, 1)
    assert close_joined_reservation is True
    assert not operation.is_alive()
    assert all(not closer.is_alive() for closer in closers)
    assert operation_errors == [primary]
    assert close_errors == [[runtimes[0]._terminal_error], [primary]]
    assert publication_starts == [[1], [1]]
    assert [raw.abort_calls for raw in raw_comms] == [1, 1]
    assert [
        writes.count(wrapper._fatal_key(rank))
        for rank, (wrapper, writes) in enumerate(zip(wrappers, control_writes))
    ] == [1, 1]
    assert [
        writes.count(wrapper._fatal_ack_key(rank))
        for rank, (wrapper, writes) in enumerate(zip(wrappers, control_writes))
    ] == [1, 1]
    assert [wrapper._fatal_publications for wrapper in wrappers] == [0, 0]
    assert set(runtimes[1]._terminal_quarantine.owners) == {
        cohort_owner,
        failed_owner,
    }
    assert runtimes[1]._terminal_quarantine.first_error is primary
    assert failed_owner.error is primary
    assert failed_owner.secondary_errors == (drain_error,)
    assert hard_exits == []
    assert stop_order == []


def test_monitor_atomically_hands_existing_local_fatal_to_worker(monkeypatch):
    store = _SharedStore(2)
    raw_comms = [_RawComm(), _RawComm()]
    pairs = [_cpu_collective(rank, 2, store, raw_comms[rank]) for rank in range(2)]
    wrappers = [pair[0] for pair in pairs]
    backends = [pair[1] for pair in pairs]
    _bootstrap_cpu_wrappers(wrappers)
    for wrapper in wrappers:
        wrapper._fatal_monitor_stop.set()
    for wrapper in wrappers:
        wrapper._fatal_monitor_thread.join(3.0)
        assert not wrapper._fatal_monitor_thread.is_alive()
        wrapper._fatal_monitor_stop.clear()

    store.set(wrappers[0]._fatal_key(0), 1)
    store.set(wrappers[0]._fatal_ack_key(0), 1)
    monitor_observed = threading.Event()
    release_monitor = threading.Event()
    worker_admitted = threading.Event()
    direct_owner_reserved = threading.Event()
    direct_live = threading.Event()
    release_direct_owner = threading.Event()
    original_enter = wrappers[1]._enter_observed_fatal

    def pause_monitor(error, origin_rank, **kwargs):
        monitor_observed.set()
        assert release_monitor.wait(3.0)
        return original_enter(error, origin_rank, **kwargs)

    monkeypatch.setattr(wrappers[1], "_enter_observed_fatal", pause_monitor)
    admissions = []
    direct_begins = []
    original_begin = wrappers[1]._begin_fatal_publication
    monitor = None

    def record_begin(*args, **kwargs):
        result = original_begin(*args, **kwargs)
        thread_name = threading.current_thread().name
        if thread_name == "task-18.2-direct-publisher":
            direct_begins.append(
                (result[2], result[3], result[4], wrappers[1]._fatal_publications)
            )
        elif thread_name == "renormalizer-fatal-publisher-rank-1":
            admissions.append(
                (
                    result[2],
                    result[3],
                    result[4],
                    wrappers[1]._fatal_publications,
                )
            )
            worker_admitted.set()
        return result

    monkeypatch.setattr(wrappers[1], "_begin_fatal_publication", record_begin)
    original_prepare = wrappers[1]._prepare_local_fatal_locked

    def hold_observable_direct_owner(**kwargs):
        if threading.current_thread().name == "task-18.2-direct-publisher":
            reservation = wrappers[1]._fatal_publication_local.reservation
            assert reservation.started is True
            assert direct_begins == [(True, False, False, 1)]
            direct_owner_reserved.set()
            assert release_direct_owner.wait(3.0)
        return original_prepare(**kwargs)

    monkeypatch.setattr(
        wrappers[1], "_prepare_local_fatal_locked", hold_observable_direct_owner
    )

    control_writes = [[], []]
    for rank, wrapper in enumerate(wrappers):
        original_set = wrapper._fatal_store_set

        def record_set(
            key,
            value,
            rank=rank,
            original=original_set,
            **kwargs,
        ):
            if int(value) == 1:
                control_writes[rank].append(key)
            return original(key, value, **kwargs)

        monkeypatch.setattr(wrapper, "_fatal_store_set", record_set)

    hard_exits = []
    stop_order = []
    for rank, (wrapper, backend) in enumerate(zip(wrappers, backends)):
        original_stop = backend.stop

        def hard_exit(rank=rank):
            hard_exits.append(rank)
            raise AssertionError(
                "unexpected communicator hard exit on rank {}".format(rank)
            )

        def record_stop(rank=rank, original=original_stop):
            stop_order.append(rank)
            return original()

        monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)
        monkeypatch.setattr(backend, "stop", record_stop)

    primary = RuntimeError("injected direct local fatal")
    direct_errors = []

    def publish_direct():
        try:
            with wrappers[1]._communicator_fatal_reservation(primary) as reservation:
                assert reservation.primary is primary
                direct_live.set()
        except BaseException as error:
            direct_errors.append(error)

    monitor = threading.Thread(
        target=wrappers[1]._monitor_fatal_records,
        name="task-18.2-deferred-monitor",
        daemon=True,
    )
    wrappers[1]._fatal_monitor_thread = monitor
    publisher = threading.Thread(
        target=publish_direct,
        name="task-18.2-direct-publisher",
        daemon=True,
    )
    try:
        monitor.start()
        assert monitor_observed.wait(3.0)
        publisher.start()
        assert direct_owner_reserved.wait(3.0)
        assert wrappers[1]._fatal_publications == 1

        release_monitor.set()
        assert worker_admitted.wait(3.0)
        worker = wrappers[1]._fatal_monitor_publisher_thread
        peak_publications = wrappers[1]._fatal_publications
        monitor.join(3.0)
        assert not monitor.is_alive()

        release_direct_owner.set()
        assert direct_live.wait(3.0)
        publisher.join(5.0)
        worker.join(5.0)
        monitor.join(5.0)
    finally:
        release_monitor.set()
        release_direct_owner.set()
        if publisher.ident is not None:
            publisher.join(5.0)
        if monitor.ident is not None:
            monitor.join(5.0)
        worker = wrappers[1]._fatal_monitor_publisher_thread
        if worker is not None and worker.ident is not None:
            worker.join(5.0)

    assert direct_begins == [(True, False, False, 1)]
    assert admissions == [(False, True, False, 1)]
    assert peak_publications == 1
    assert not publisher.is_alive()
    assert not monitor.is_alive()
    assert not worker.is_alive()
    assert direct_errors == []
    assert wrappers[1]._fatal_error is primary
    assert wrappers[1]._fatal_publications == 0
    assert raw_comms[1].abort_calls == 1
    assert control_writes[1].count(wrappers[1]._fatal_key(1)) == 1
    assert control_writes[1].count(wrappers[1]._fatal_ack_key(1)) == 1

    _close_cpu_wrappers(wrappers)

    assert [raw.abort_calls for raw in raw_comms] == [1, 1]
    assert [
        writes.count(wrapper._fatal_key(rank))
        for rank, (wrapper, writes) in enumerate(zip(wrappers, control_writes))
    ] == [1, 1]
    assert [
        writes.count(wrapper._fatal_ack_key(rank))
        for rank, (wrapper, writes) in enumerate(zip(wrappers, control_writes))
    ] == [1, 1]
    assert hard_exits == []
    assert stop_order == []


def test_uninitialized_close_waits_for_admitted_operation(monkeypatch):
    store = _SharedStore(1)
    raw_comm = _RawComm()
    wrapper, backend = _cpu_collective(0, 1, store, raw_comm)
    operation_entered = threading.Event()
    release_operation = threading.Event()

    def blocking_barrier():
        operation_entered.set()
        assert release_operation.wait(3.0)

    monkeypatch.setattr(backend, "barrier", blocking_barrier)
    operation_errors = []
    close_errors = []

    def run_operation():
        try:
            wrapper.barrier()
        except BaseException as error:
            operation_errors.append(error)

    def close():
        try:
            wrapper.close()
        except BaseException as error:
            close_errors.append(error)

    operation = threading.Thread(target=run_operation, daemon=True)
    closer = threading.Thread(target=close, daemon=True)
    try:
        operation.start()
        assert operation_entered.wait(3.0)
        closer.start()
        deadline = time.monotonic() + 3.0
        while not wrapper._closing:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        closer.join(0.05)
        assert closer.is_alive()
        assert backend.stop_calls == 0
        release_operation.set()
        operation.join(5.0)
        closer.join(5.0)
    finally:
        release_operation.set()
        operation.join(5.0)
        if closer.ident is not None:
            closer.join(5.0)
        if not wrapper._closed:
            wrapper.close()

    assert not operation.is_alive()
    assert not closer.is_alive()
    assert operation_errors == []
    assert close_errors == []
    assert backend.stop_calls == 1


def test_local_and_remote_fatal_abort_once_without_destroy_or_stop():
    store = _SharedStore(2)
    raw_comms = [_RawComm(), _RawComm()]
    wrappers = [
        _cpu_collective(rank, 2, store, raw_comms[rank])[0] for rank in range(2)
    ]
    bootstrap_errors = [None, None]

    def bootstrap(rank):
        try:
            assert wrappers[rank]._bootstrap_fatal_control() == 0
        except BaseException as error:
            bootstrap_errors[rank] = error

    threads = [threading.Thread(target=bootstrap, args=(rank,)) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5.0)
    assert all(not thread.is_alive() for thread in threads)
    assert bootstrap_errors == [None, None]

    primary = RuntimeError("injected local communicator failure")
    publish_error = []

    def publish():
        try:
            wrappers[0]._publish_communicator_fatal(primary)
        except BaseException as error:
            publish_error.append(error)

    publisher = threading.Thread(target=publish)
    try:
        publisher.start()
        publisher.join(5.0)
        assert not publisher.is_alive()
        assert publish_error == []
        assert wrappers[0]._fatal_error is primary
        assert wrappers[1]._fatal_error is not primary
        assert (
            str(wrappers[1]._fatal_error) == "remote communicator failure from rank 0"
        )
        assert [raw.abort_calls for raw in raw_comms] == [1, 1]
        assert [raw.destroy_calls for raw in raw_comms] == [0, 0]
        assert [wrapper._backend.stop_calls for wrapper in wrappers] == [0, 0]

        wrappers[0]._publish_communicator_fatal(RuntimeError("later failure"))
        assert [raw.abort_calls for raw in raw_comms] == [1, 1]
    finally:
        _close_cpu_wrappers(wrappers)


def test_fatal_handler_runs_after_abort_start_without_waiting_for_completion():
    store = _SharedStore(1)
    raw_comm = _RawComm()
    wrapper, _ = _cpu_collective(0, 1, store, raw_comm)
    assert wrapper._bootstrap_fatal_control() == 0
    wrapper._fatal_monitor_stop.set()
    wrapper._fatal_monitor_thread.join(5.0)
    abort_entered = threading.Event()
    release_abort = threading.Event()

    def blocking_abort():
        raw_comm.abort_calls += 1
        abort_entered.set()
        assert release_abort.wait(3.0)

    raw_comm.abort = blocking_abort
    observed = []

    def handler(error):
        observed.append(
            (
                error,
                wrapper._fatal_abort_started,
                wrapper._fatal_abort_completed,
            )
        )
        wrapper._publish_communicator_fatal(error)

    wrapper._install_fatal_handler(handler)
    primary = RuntimeError("injected remote communicator failure")
    errors = []

    def observe():
        try:
            wrapper._enter_observed_fatal(primary, 0)
        except BaseException as error:
            errors.append(error)

    observer = threading.Thread(target=observe)
    try:
        observer.start()
        assert abort_entered.wait(3.0)
        observer.join(0.05)
        assert observer.is_alive()
        release_abort.set()
        observer.join(5.0)
        assert not observer.is_alive()
        assert errors == []
        assert len(observed) == 1
        error, abort_started, abort_completed = observed[0]
        assert error is primary
        assert abort_started is True
        assert abort_completed is False
    finally:
        release_abort.set()
        observer.join(5.0)
        wrapper.close()


def test_terminal_collective_rejects_every_public_entry_before_sentinel_access():
    class Sentinel:
        def __getattribute__(self, name):
            raise AssertionError("terminal collective touched array sentinel")

    store = _SharedStore(1)
    raw_comm = _RawComm()
    wrapper, backend = _cpu_collective(0, 1, store, raw_comm)
    primary = RuntimeError("injected terminal communicator failure")
    assert wrapper._bootstrap_fatal_control() == 0
    wrapper._publish_communicator_fatal(primary)
    try:
        operations = (
            wrapper.barrier,
            lambda: wrapper.broadcast(Sentinel(), root=0),
            lambda: wrapper.allreduce(Sentinel()),
            lambda: wrapper.allreduce_inplace(Sentinel()),
            lambda: wrapper.reduce_scatter(Sentinel(), axis=0),
            lambda: wrapper.allgather(Sentinel(), axis=0),
        )
        for operation in operations:
            with pytest.raises(RuntimeError, match="terminal-aborted") as caught:
                operation()
            assert caught.value.__cause__ is primary
        assert backend.calls == []
        assert raw_comm.abort_calls == 1
        assert raw_comm.destroy_calls == 0
    finally:
        wrapper.close()


def test_abort_failure_is_secondary_to_immutable_fatal_primary(monkeypatch):
    abort_error = RuntimeError("injected raw abort failure")
    store = _SharedStore(1)
    raw_comm = _RawComm(abort_error=abort_error)
    wrapper, backend = _cpu_collective(0, 1, store, raw_comm)
    primary = RuntimeError("injected communicator primary")
    assert wrapper._bootstrap_fatal_control() == 0
    wrapper._fatal_monitor_stop.set()
    wrapper._fatal_monitor_thread.join(5.0)

    def hard_exit():
        raise SystemExit(86)

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)
    try:
        with pytest.raises(SystemExit) as caught:
            wrapper._publish_communicator_fatal(primary)
        assert caught.value.code == 86
        assert wrapper._fatal_pending_primary is primary
        assert wrapper._fatal_error is None
        assert wrapper._fatal_secondary_errors == (abort_error,)
        assert wrapper._fatal_publications == 1
        assert wrapper._fatal_publication_failure is primary
        assert raw_comm.abort_calls == 1
        assert raw_comm.destroy_calls == 0
        assert backend.stop_calls == 0
    finally:
        wrapper._fatal_control_initialized = False
        with wrapper._fatal_condition:
            wrapper._fatal_publications = 0
            wrapper._fatal_publication_failure = None
            wrapper._fatal_condition.notify_all()
        wrapper.close()


def test_abort_failure_hard_exits_subprocess_instead_of_returning():
    script = r"""
from renormalizer.backend._distributed.collectives import CupyNcclCollective
from renormalizer.backend._distributed.context import DistributedContext

class Device:
    def use(self):
        pass

class Runtime:
    @staticmethod
    def getDeviceCount():
        return 1

class Cuda:
    runtime = Runtime()
    Device = staticmethod(lambda index: Device())

class Cupy:
    cuda = Cuda()
    class ndarray:
        pass

class Store:
    def __init__(self):
        self.values = {}
    def __setitem__(self, key, value):
        self.values[key] = value
    def __getitem__(self, key):
        return self.values[key]
    def barrier(self):
        pass

class Raw:
    def abort(self):
        raise RuntimeError("injected raw abort failure")

class Backend:
    rank = 0
    def __init__(self):
        self._store_proxy = Store()
        self._comm = Raw()
    def stop(self):
        raise AssertionError("stop must not substitute for abort")

backend = Backend()
collective = CupyNcclCollective(
    DistributedContext(0, 0, 1, 1),
    cupy_module=Cupy(),
    init_process_group=lambda *args, **kwargs: backend,
    host="127.0.0.1",
    port=23456,
)
collective._bootstrap_store_proxy = Store()
collective._local_fatal_capability_code = lambda **_kwargs: 0
assert collective._bootstrap_fatal_control() == 0
collective._publish_communicator_fatal(RuntimeError("primary"))
raise AssertionError("abort failure returned to ordinary Python")
"""
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=15.0,
    )
    assert completed.returncode == 86, (completed.stdout, completed.stderr)


def test_runtime_close_failure_is_terminal_without_collective_retry():
    from renormalizer.backend._distributed.context import DistributedRendezvous
    from renormalizer.backend._distributed.mesh import DeviceMesh
    from renormalizer.backend.distributed_runtime import CupyDistributedRuntime

    class FailingOnceCollective:
        def __init__(self):
            self.close_calls = 0
            self.live = True

        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("injected collective close failure")
            self.live = False

    context = DistributedContext(0, 0, 1, 1)
    collective = FailingOnceCollective()
    runtime = CupyDistributedRuntime(
        backend=object(),
        context=context,
        rendezvous=DistributedRendezvous("127.0.0.1", 23456),
        mesh=DeviceMesh((1,), ("rank",), 0),
        collective=collective,
    )

    with pytest.raises(RuntimeError, match="injected collective close failure"):
        runtime.close()

    assert runtime._closed is True
    assert runtime.collective is collective
    assert collective.close_calls == 1
    assert collective.live is True

    runtime.close()
    runtime.close()

    assert runtime._closed is True
    assert runtime.collective is collective
    assert collective.live is True
    assert collective.close_calls == 1
    with pytest.raises(RuntimeError, match="distributed runtime is closed"):
        runtime.barrier()


def test_runtime_execution_config_is_a_borrowed_frozen_view():
    from dataclasses import FrozenInstanceError

    from renormalizer.backend._distributed.collectives import SingleProcessCollective
    from renormalizer.backend._distributed.context import DistributedRendezvous
    from renormalizer.backend._distributed.mesh import DeviceMesh
    from renormalizer.backend._distributed.providers import DeviceResidentProvider
    from renormalizer.backend.distributed_runtime import CupyDistributedRuntime

    context = DistributedContext(0, 0, 1, 1)
    collective = SingleProcessCollective()
    runtime = CupyDistributedRuntime(
        backend=object(),
        context=context,
        rendezvous=DistributedRendezvous("127.0.0.1", 23456),
        mesh=DeviceMesh((1,), ("rank",), 0),
        collective=collective,
    )

    config = runtime.execution_config(
        device_memory_budget_bytes=1024,
        host_memory_budget_bytes=2048,
        prefetch_depth=2,
    )

    assert config.context is context
    assert config.mesh is runtime.mesh
    assert config.collective is collective
    assert isinstance(config.provider, DeviceResidentProvider)
    assert config.device_memory_budget_bytes == 1024
    assert config.host_memory_budget_bytes == 2048
    assert config.prefetch_depth == 2
    with pytest.raises(FrozenInstanceError):
        config.prefetch_depth = 3

    runtime.close()
    with pytest.raises(RuntimeError, match="distributed runtime is closed"):
        runtime.execution_config()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "numpy"),
        ("device", "cuda:1"),
        ("precision", 32),
    ],
)
def test_runtime_execution_config_synchronizes_active_backend_mismatch(
    field, value, monkeypatch
):
    import renormalizer.backend.distributed_runtime as runtime_module
    from renormalizer.backend._distributed.context import DistributedRendezvous
    from renormalizer.backend._distributed.mesh import DeviceMesh

    class Collective:
        rank = 0
        size = 2

        def __init__(self):
            self.trace = []

        def allreduce(self, array, *, op="sum"):
            self.trace.append((op, np.asarray(array).dtype.str, np.asarray(array).size))
            return np.array(array, copy=True)

    expected = SimpleNamespace(
        name="cupy",
        device="cuda:0",
        config=SimpleNamespace(precision=64),
        array_namespace=np,
        asarray=np.asarray,
    )
    active = SimpleNamespace(
        name="cupy",
        device="cuda:0",
        config=SimpleNamespace(precision=64),
    )
    if field == "precision":
        active.config.precision = value
    else:
        setattr(active, field, value)
    monkeypatch.setattr("renormalizer.cons.get_backend", lambda: active)
    collective = Collective()
    context = DistributedContext(0, 0, 2, 2)
    runtime = runtime_module.CupyDistributedRuntime(
        backend=expected,
        context=context,
        rendezvous=DistributedRendezvous("127.0.0.1", 23456),
        mesh=DeviceMesh((2,), ("rank",), 0),
        collective=collective,
    )

    with pytest.raises(
        RuntimeError, match="distributed runtime backend validation failed"
    ):
        runtime.execution_config()

    assert collective.trace == [
        ("max", np.dtype(np.int32).str, 1),
        ("min", np.dtype(np.uint64).str, 4),
        ("max", np.dtype(np.uint64).str, 4),
    ]


def test_runtime_backend_digest_normalizes_valid_rank_local_devices(monkeypatch):
    import renormalizer.backend.distributed_runtime as runtime_module
    from renormalizer.backend._distributed.context import DistributedRendezvous
    from renormalizer.backend._distributed.mesh import DeviceMesh

    def digest_for_rank(rank):
        class Collective:
            size = 2

            def __init__(self):
                self.rank = rank
                self.digest = None

            def allreduce(self, value, *, op="sum"):
                value = np.asarray(value)
                if op == "min" and value.dtype == np.uint64:
                    self.digest = np.array(value, copy=True)
                return np.array(value, copy=True)

        selected = SimpleNamespace(
            name="cupy",
            device="cuda:{}".format(rank),
            config=SimpleNamespace(precision=64),
            array_namespace=np,
            asarray=np.asarray,
        )
        monkeypatch.setattr("renormalizer.cons.get_backend", lambda: selected)
        collective = Collective()
        context = DistributedContext(rank, rank, 2, 2)
        runtime = runtime_module.CupyDistributedRuntime(
            backend=selected,
            context=context,
            rendezvous=DistributedRendezvous("127.0.0.1", 23456),
            mesh=DeviceMesh((2,), ("rank",), rank),
            collective=collective,
        )
        config = runtime.execution_config(
            device_memory_budget_bytes=1024,
            host_memory_budget_bytes=2048,
        )
        assert config.backend_device == "cuda:{}".format(rank)
        return collective.digest

    np.testing.assert_array_equal(digest_for_rank(0), digest_for_rank(1))


def test_fake_collectives_have_explicit_numerical_and_mutation_semantics(collective):
    wrapper, backend = collective
    source = cupy.arange(12, dtype=cupy.float64).reshape(3, 4)
    original = source.copy()

    reduced = wrapper.allreduce(source)
    gathered = wrapper.allgather(source, axis=1)
    scattered = wrapper.reduce_scatter(source, axis=1)
    broadcast = wrapper.broadcast(source, root=0)

    cupy.testing.assert_array_equal(reduced, original * 2)
    cupy.testing.assert_array_equal(
        gathered,
        cupy.concatenate((original, original + 100), axis=1),
    )
    cupy.testing.assert_array_equal(scattered, original[:, 2:] * 2)
    assert broadcast is source
    cupy.testing.assert_array_equal(source, cupy.full_like(source, 10))
    assert [call[0] for call in backend.calls] == [
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "broadcast",
    ]


def test_inplace_allreduce_reuses_control_array_without_staging_allocation(
    collective, monkeypatch
):
    wrapper, backend = collective
    source = cupy.ones((1,), dtype=cupy.int32)
    allocation_calls = []
    original_empty_like = cupy.empty_like
    original_ascontiguousarray = cupy.ascontiguousarray

    def recording_empty_like(*args, **kwargs):
        allocation_calls.append("empty_like")
        return original_empty_like(*args, **kwargs)

    def recording_ascontiguousarray(*args, **kwargs):
        allocation_calls.append("ascontiguousarray")
        return original_ascontiguousarray(*args, **kwargs)

    monkeypatch.setattr(cupy, "empty_like", recording_empty_like)
    monkeypatch.setattr(cupy, "ascontiguousarray", recording_ascontiguousarray)

    result = wrapper.allreduce_inplace(source, op="max")

    assert result is source
    assert allocation_calls == []
    cupy.testing.assert_array_equal(source, cupy.asarray([2], dtype=cupy.int32))
    assert backend.calls == [("all_reduce", "max", cupy.cuda.Stream.null)]


def test_inplace_allreduce_uses_nondefault_stream_and_restores_active_device(
    collective,
):
    wrapper, backend = collective
    with cupy.cuda.Device(0):
        stream = cupy.cuda.Stream(non_blocking=True)
        with stream:
            source = cupy.ones((1,), dtype=cupy.int32)
            cupy.cuda.Device(1).use()

            result = wrapper.allreduce_inplace(source, op="max")

            assert int(cupy.cuda.runtime.getDevice()) == 1

    assert result is source
    assert backend.calls == [("all_reduce", "max", stream)]


def test_noncontiguous_arrays_are_staged_without_host_conversion(
    collective, monkeypatch
):
    wrapper, _ = collective
    source = cupy.arange(24, dtype=cupy.complex128).reshape(4, 6)[:, ::2]
    original = source.copy()
    monkeypatch.setattr(
        cupy,
        "asnumpy",
        lambda *args, **kwargs: pytest.fail("collective converted through the host"),
    )

    result = wrapper.allreduce(source, op="sum")

    assert bool(cupy.all(result == original * 2))
    assert bool(cupy.all(source == original))


def test_collectives_use_selected_device_current_stream(collective):
    wrapper, backend = collective
    with cupy.cuda.Device(0):
        stream = cupy.cuda.Stream(non_blocking=True)
        with stream:
            source = cupy.arange(4, dtype=cupy.float32)
            cupy.cuda.Device(1).use()
            wrapper.allreduce(source)

    operation, _, used_stream = backend.calls[-1]
    assert operation == "all_reduce"
    assert used_stream is stream


@pytest.mark.parametrize(
    "invoke, message",
    [
        (lambda c, a: c.broadcast(a, root=2), "root 2 is out of range"),
        (lambda c, a: c.allreduce(a, op="mean"), "unsupported reduction op"),
        (lambda c, a: c.allgather(a, axis=2), "axis 2 is out of range"),
        (
            lambda c, a: c.reduce_scatter(a, axis=1),
            "axis length 3 is not divisible by collective size 2",
        ),
    ],
)
def test_illegal_arguments_are_rejected_before_nccL(collective, invoke, message):
    wrapper, backend = collective
    source = cupy.ones((2, 3), dtype=cupy.float64)

    with pytest.raises(ValueError, match=message):
        invoke(wrapper, source)

    assert backend.calls == []


def test_unsupported_dtype_and_complex_op_are_rejected_before_nccl(collective):
    wrapper, backend = collective

    with pytest.raises(TypeError, match="dtype bool is not supported"):
        wrapper.allreduce(cupy.ones(2, dtype=cupy.bool_))
    with pytest.raises(ValueError, match="complex arrays only support sum"):
        wrapper.allreduce(cupy.ones(2, dtype=cupy.complex64), op="max")

    assert backend.calls == []


def test_wrong_device_array_is_rejected_before_nccl(collective):
    if cupy.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("requires two visible CUDA devices")
    wrapper, backend = collective
    with cupy.cuda.Device(1):
        source = cupy.ones(2)

    with pytest.raises(ValueError, match="selected CUDA device"):
        wrapper.allreduce(source)

    assert backend.calls == []


def test_runtime_parses_environment_once_and_preflights_world_size(monkeypatch):
    import renormalizer.backend.distributed_runtime as runtime_module

    parsed = []
    original = DistributedContext.from_environ.__func__

    def recording_parser(cls, environ):
        parsed.append(dict(environ))
        return original(cls, environ)

    monkeypatch.setattr(
        DistributedContext, "from_environ", classmethod(recording_parser)
    )
    backend_calls = []

    class FakeBackend:
        def create_collective(self, supplied_context, **kwargs):
            backend_calls.append((supplied_context, kwargs))
            return FakeNcclBackend(supplied_context.world_size, supplied_context.rank)

    monkeypatch.setattr(
        runtime_module,
        "create_backend",
        lambda *args, **kwargs: FakeBackend(),
    )
    environ = {
        "RANK": "9",
        "LOCAL_RANK": "1",
        "WORLD_SIZE": "16",
        "LOCAL_WORLD_SIZE": "8",
    }

    runtime = runtime_module.create_cupy_distributed_runtime(
        expected_world_size=16, environ=environ, host="host", port=1234
    )
    environ["RANK"] = "3"
    try:
        assert runtime.rank == 9
        assert runtime.local_rank == 1
        assert len(parsed) == 1
        assert backend_calls == [(runtime.context, {"host": "host", "port": 1234})]
    finally:
        runtime.close()


def test_runtime_snapshots_master_rendezvous_before_backend_creation(monkeypatch):
    import renormalizer.backend.distributed_runtime as runtime_module
    import cupyx.distributed as cupyx_distributed

    initializer_calls = []
    environ = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
        "MASTER_ADDR": "snapshot-host",
        "MASTER_PORT": "29500",
    }
    original_create_backend = runtime_module.create_backend

    def initialize(*args, **kwargs):
        initializer_calls.append((args, kwargs))
        return FakeNcclBackend(size=args[0], rank=args[1])

    def create_backend_after_environment_mutation(*args, **kwargs):
        environ["MASTER_ADDR"] = "mutated-supplied-host"
        environ["MASTER_PORT"] = "31000"
        monkeypatch.setenv("CUPYX_DISTRIBUTED_HOST", "mutated-ambient-host")
        monkeypatch.setenv("CUPYX_DISTRIBUTED_PORT", "32000")
        return original_create_backend(*args, **kwargs)

    monkeypatch.setattr(cupyx_distributed, "init_process_group", initialize)
    monkeypatch.setattr(
        runtime_module, "create_backend", create_backend_after_environment_mutation
    )

    runtime = runtime_module.create_cupy_distributed_runtime(environ=environ)
    try:
        assert runtime.rendezvous.host == "snapshot-host"
        assert runtime.rendezvous.port == 30500
        assert initializer_calls == [
            (
                (1, 0),
                {
                    "backend": "nccl",
                    "host": "snapshot-host",
                    "port": 30500,
                },
            )
        ]
    finally:
        runtime.close()


def test_runtime_uses_cupyx_rendezvous_override(monkeypatch):
    import renormalizer.backend.distributed_runtime as runtime_module

    backend_calls = []
    environ = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
        "CUPYX_DISTRIBUTED_HOST": "cupyx-host",
        "CUPYX_DISTRIBUTED_PORT": "24567",
        "MASTER_ADDR": "master-host",
        "MASTER_PORT": "29500",
    }

    class FakeBackend:
        def create_collective(self, supplied_context, **kwargs):
            backend_calls.append((supplied_context, kwargs))
            return FakeNcclBackend(supplied_context.world_size, supplied_context.rank)

    monkeypatch.setattr(
        runtime_module, "create_backend", lambda *args, **kwargs: FakeBackend()
    )

    runtime = runtime_module.create_cupy_distributed_runtime(environ=environ)
    try:
        assert backend_calls == [
            (runtime.context, {"host": "cupyx-host", "port": 24567})
        ]
    finally:
        runtime.close()


def test_runtime_rejects_invalid_rendezvous_before_backend_creation(monkeypatch):
    import renormalizer.backend.distributed_runtime as runtime_module

    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(runtime_module, "create_backend", fail_if_called)
    environ = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
        "MASTER_ADDR": "master-host",
        "MASTER_PORT": "invalid",
    }

    with pytest.raises(ValueError, match="MASTER_PORT must be an integer"):
        runtime_module.create_cupy_distributed_runtime(environ=environ)

    assert called is False


def test_runtime_context_closes_collective_after_body_failure(monkeypatch):
    import renormalizer.backend.distributed_runtime as runtime_module

    created = []

    class FakeBackend:
        def create_collective(self, supplied_context, **kwargs):
            collective = FakeNcclBackend(
                supplied_context.world_size, supplied_context.rank
            )
            created.append(collective)
            return collective

    monkeypatch.setattr(
        runtime_module, "create_backend", lambda *args, **kwargs: FakeBackend()
    )
    environ = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
        "MASTER_ADDR": "master-host",
        "MASTER_PORT": "29500",
    }

    with pytest.raises(RuntimeError, match="injected body failure"):
        with runtime_module.cupy_distributed_runtime(environ=environ):
            raise RuntimeError("injected body failure")

    assert len(created) == 1
    assert created[0].stop_calls == 1


def test_runtime_rejects_expected_world_size_before_backend_creation(monkeypatch):
    import renormalizer.backend.distributed_runtime as runtime_module

    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(runtime_module, "create_backend", fail_if_called)
    environ = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
    }

    with pytest.raises(ValueError, match="expected world size 4, got 2"):
        runtime_module.create_cupy_distributed_runtime(
            expected_world_size=4, environ=environ
        )

    assert called is False


def test_active_broadcast_validation_failure_self_defers_first_owner(monkeypatch):
    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    before_agree = threading.Event()
    release_agree = threading.Event()
    validation_errors = []
    self_wait_errors = []
    hard_exits = []
    worker_name = "task-18.2-validation-first-owner"
    original_wait = wrapper._wait_for_active_broadcast_agreements

    def forbid_active_broadcast_self_wait(**kwargs):
        if (
            threading.current_thread().name == worker_name
            and wrapper._active_broadcast_agreements
        ):
            error = AssertionError("active-B discoverer waited on itself")
            self_wait_errors.append(error)
            raise error
        return original_wait(**kwargs)

    def hard_exit():
        hard_exits.append(threading.current_thread().name)
        raise AssertionError("unexpected validation fatal hard exit")

    monkeypatch.setattr(
        wrapper,
        "_wait_for_active_broadcast_agreements",
        forbid_active_broadcast_self_wait,
    )
    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    def run_active_broadcast():
        with wrapper._active_broadcast_admission() as (broadcast, agree):
            try:
                broadcast(np.ones((2,), dtype=np.float64), root=1)
            except ValueError as error:
                validation_errors.append(error)
                try:
                    assert runtime._enter_communicator_fatal(error) is error
                except BaseException as publication_error:
                    self_wait_errors.append(publication_error)
                    before_agree.set()
                    raise
                assert wrapper._active_broadcast_local.deferral is not None
                before_agree.set()
                assert release_agree.wait(_TASK_18_2_TIMEOUT_S * 2)
                try:
                    agree(True)
                except BaseException:
                    raise error
                raise error
            raise AssertionError("invalid root was accepted")

    worker, results, errors, done = _start_task_18_2_call(
        run_active_broadcast, name=worker_name
    )
    try:
        assert before_agree.wait(_TASK_18_2_TIMEOUT_S)
        assert self_wait_errors == []
        assert gate.phase is _TerminalPhase.FATAL_PENDING
        assert wrapper._fatal_error is None
        assert runtime._terminal_error is None
        assert wrapper._active_broadcast_agreements == 1
    finally:
        release_agree.set()
        _join_task_18_2_call(worker, done)

    primary = validation_errors[0]
    assert results == []
    assert errors == [primary]
    assert gate.wait_for_published(_TASK_18_2_TIMEOUT_S) is primary
    assert wrapper._fatal_error is primary
    assert runtime._terminal_error is primary
    assert wrapper._active_broadcast_agreements == 0
    assert wrapper._fatal_publications == 0
    assert hard_exits == []


def test_active_broadcast_fatal_reservation_converts_discoverer_lease_admission(
    monkeypatch,
):
    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    epoch, construction = gate.begin_lease("active-broadcast-fatal-reservation")
    gate.activate_lease(epoch, construction)
    gate.release(construction)
    primary = RuntimeError("active broadcast fatal with lease admission")
    self_waits = []
    hard_exits = []
    original_wait = gate.wait_for_admissions

    def observe_self_wait(transition, timeout_s):
        current = gate._current_thread_admission()
        if current is not None:
            with gate._condition:
                state = gate._tokens[current.sequence]
                if state.status == "active":
                    self_waits.append(current)
        return original_wait(transition, timeout_s)

    def hard_exit():
        hard_exits.append(threading.current_thread().name)
        raise AssertionError("fatal reservation attempted a hard exit")

    monkeypatch.setattr(gate, "wait_for_admissions", observe_self_wait)
    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    def run_active_broadcast():
        token = gate.admit_lease(epoch, "operator_call")
        try:
            with wrapper._active_broadcast_admission() as (_, agree):
                with runtime._communicator_fatal_reservation(primary) as (
                    canonical,
                    reservation,
                ):
                    assert canonical is primary
                    assert reservation.completion_deferred is True
                    assert runtime._enter_communicator_fatal(primary) is primary
                with pytest.raises(
                    RuntimeError,
                    match="collective fatal publication is pending",
                ) as pending:
                    agree(True)
                assert pending.value.__cause__ is primary
                raise primary
        finally:
            runtime._release_admission(token)

    worker, results, errors, done = _start_task_18_2_call(
        run_active_broadcast,
        name="active-broadcast-fatal-with-lease-admission",
    )
    _join_task_18_2_call(worker, done)

    assert results == []
    assert errors == [primary]
    assert self_waits == []
    assert hard_exits == []
    assert gate.wait_for_published(_TASK_18_2_TIMEOUT_S) is primary
    assert wrapper._fatal_error is primary
    assert runtime._terminal_error is primary


def test_active_broadcast_backend_failure_converts_fallback_lease_admission(
    monkeypatch,
):
    runtime, wrapper, backend, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    epoch, construction = gate.begin_lease("active-broadcast-backend-failure")
    gate.activate_lease(epoch, construction)
    gate.release(construction)
    primary = RuntimeError("active broadcast backend failure with lease admission")
    hard_exits = []

    monkeypatch.setattr(wrapper, "_validate_array", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        backend,
        "broadcast",
        lambda *args, **kwargs: (_ for _ in ()).throw(primary),
    )

    def hard_exit():
        hard_exits.append(threading.current_thread().name)
        raise AssertionError("backend failure attempted a fatal hard exit")

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    def run_active_broadcast():
        token = gate.admit_lease(epoch, "operator_call")
        try:
            with wrapper._active_broadcast_admission() as (broadcast, agree):
                with pytest.raises(RuntimeError) as broadcast_error:
                    broadcast(np.ones((2,), dtype=np.float64), root=0)
                assert broadcast_error.value is primary
                with gate._condition:
                    assert gate._tokens[token.sequence].status == "converted"
                with pytest.raises(
                    RuntimeError,
                    match="collective fatal publication is pending",
                ) as pending:
                    agree(True)
                assert pending.value.__cause__ is primary
                raise primary
        finally:
            runtime._release_admission(token)

    worker, results, errors, done = _start_task_18_2_call(
        run_active_broadcast,
        name="active-broadcast-backend-failure-with-lease-admission",
    )
    _join_task_18_2_call(worker, done)

    assert results == []
    assert errors == [primary]
    assert hard_exits == []
    assert gate.wait_for_published(_TASK_18_2_TIMEOUT_S) is primary
    assert wrapper._fatal_error is primary
    assert runtime._terminal_error is primary


def test_active_broadcast_validation_failure_converts_fallback_lease_admission(
    monkeypatch,
):
    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    epoch, construction = gate.begin_lease("active-broadcast-validation-failure")
    gate.activate_lease(epoch, construction)
    gate.release(construction)
    validation_errors = []
    hard_exits = []

    def hard_exit():
        hard_exits.append(threading.current_thread().name)
        raise AssertionError("validation failure attempted a fatal hard exit")

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    def run_active_broadcast():
        token = gate.admit_lease(epoch, "operator_call")
        try:
            with wrapper._active_broadcast_admission() as (broadcast, agree):
                with pytest.raises(
                    ValueError, match="root 1 is out of range"
                ) as caught:
                    broadcast(np.ones((2,), dtype=np.float64), root=1)
                primary = caught.value
                validation_errors.append(primary)
                with gate._condition:
                    assert gate._tokens[token.sequence].status == "converted"
                with pytest.raises(
                    RuntimeError,
                    match="collective fatal publication is pending",
                ) as pending:
                    agree(True)
                assert pending.value.__cause__ is primary
                raise primary
        finally:
            runtime._release_admission(token)

    worker, results, errors, done = _start_task_18_2_call(
        run_active_broadcast,
        name="active-broadcast-validation-failure-with-lease-admission",
    )
    _join_task_18_2_call(worker, done)

    primary = validation_errors[0]
    assert results == []
    assert hard_exits == []
    assert errors == [primary]
    assert gate.wait_for_published(_TASK_18_2_TIMEOUT_S) is primary
    assert wrapper._fatal_error is primary
    assert runtime._terminal_error is primary


def test_active_broadcast_validation_failure_joins_existing_owner(monkeypatch):
    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    primary = RuntimeError("existing active-B publication owner")
    later = ValueError("later active-B root validation failure")
    admission_entered = threading.Event()
    release_validation = threading.Event()
    owner_waiting = threading.Event()
    before_agree = threading.Event()
    release_agree = threading.Event()
    validation_errors = []
    self_join_errors = []
    hard_exits = []
    owner_name = "task-18.2-validation-existing-owner"
    worker_name = "task-18.2-validation-joiner"
    original_barrier_wait = wrapper._wait_for_active_broadcast_agreements
    original_join = wrapper._wait_for_joined_fatal_publication

    def observe_owner_barrier(**kwargs):
        if threading.current_thread().name == owner_name:
            owner_waiting.set()
        return original_barrier_wait(**kwargs)

    def forbid_active_broadcast_self_join(**kwargs):
        if (
            threading.current_thread().name == worker_name
            and wrapper._active_broadcast_agreements
        ):
            error = AssertionError("active-B discoverer joined before agreement")
            self_join_errors.append(error)
            raise error
        return original_join(**kwargs)

    def hard_exit():
        hard_exits.append(threading.current_thread().name)
        raise AssertionError("unexpected joined validation fatal hard exit")

    monkeypatch.setattr(
        wrapper, "_wait_for_active_broadcast_agreements", observe_owner_barrier
    )
    monkeypatch.setattr(
        wrapper, "_wait_for_joined_fatal_publication", forbid_active_broadcast_self_join
    )
    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    def fail_root_validation(root):
        raise later

    monkeypatch.setattr(wrapper, "_validate_root", fail_root_validation)

    def run_active_broadcast():
        with wrapper._active_broadcast_admission() as (broadcast, agree):
            admission_entered.set()
            assert release_validation.wait(_TASK_18_2_TIMEOUT_S * 2)
            try:
                broadcast(np.ones((2,), dtype=np.float64), root=1)
            except BaseException as error:
                assert error is primary
                validation_errors.append(later)
                try:
                    with runtime._communicator_fatal_reservation(later) as (
                        canonical,
                        reservation,
                    ):
                        assert canonical is primary
                        assert reservation.completion_deferred is True
                        assert runtime._enter_communicator_fatal(later) is primary
                except BaseException as publication_error:
                    self_join_errors.append(publication_error)
                    before_agree.set()
                    raise
                before_agree.set()
                assert release_agree.wait(_TASK_18_2_TIMEOUT_S * 2)
                try:
                    agree(True)
                except BaseException:
                    raise primary
                raise primary
            raise AssertionError("invalid root was accepted")

    worker, worker_results, worker_errors, worker_done = _start_task_18_2_call(
        run_active_broadcast, name=worker_name
    )
    owner = None
    try:
        assert admission_entered.wait(_TASK_18_2_TIMEOUT_S)
        owner, owner_results, owner_errors, owner_done = _start_task_18_2_call(
            lambda: runtime._enter_communicator_fatal(primary), name=owner_name
        )
        assert owner_waiting.wait(_TASK_18_2_TIMEOUT_S)
        release_validation.set()
        assert before_agree.wait(_TASK_18_2_TIMEOUT_S)
        assert self_join_errors == []
        assert wrapper._active_broadcast_agreements == 1
        assert wrapper._fatal_publications == 1
        assert not owner_done.is_set()
    finally:
        release_validation.set()
        release_agree.set()
        _join_task_18_2_call(worker, worker_done)
        if owner is not None:
            _join_task_18_2_call(owner, owner_done)

    assert validation_errors == [later]
    assert worker_results == []
    assert worker_errors == [primary]
    assert owner_results == [primary]
    assert owner_errors == []
    assert wrapper._fatal_error is primary
    assert runtime._terminal_error is primary
    assert wrapper._fatal_secondary_errors == (later,)
    assert wrapper._active_broadcast_agreements == 0
    assert wrapper._fatal_publications == 0
    assert hard_exits == []


def test_real_operator_quarantine_stays_private_until_active_b_agreement(
    monkeypatch,
):
    from renormalizer.backend._distributed.async_owner import AsyncResourceOwner
    from renormalizer.backend._distributed.local_operator import (
        DistributedLocalOperator,
    )
    from renormalizer.backend._distributed.providers import (
        ActiveWorkingSetProvider,
        WorkingSetLease,
        _OperatorCall,
    )
    from renormalizer.backend._distributed.terminal import (
        _ManagedResourceAdmissionGuard,
    )
    from renormalizer.backend._execution.model import ExecutionBindings

    runtime, wrapper, backend, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    primary = RuntimeError("real operator deferred quarantine")
    before_agree = threading.Event()
    release_agree = threading.Event()
    transition_publications = []
    legacy_calls = []
    hard_exits = []
    budget = SimpleNamespace(resolved_bytes=0)
    epoch, construction = gate.begin_lease("real-operator-active-b")
    gate.activate_lease(epoch, construction)
    gate.release(construction)
    provider = ActiveWorkingSetProvider(
        runtime,
        device_budget_resolution=budget,
        host_budget_resolution=budget,
        _standalone=True,
    )
    owner = AsyncResourceOwner(
        "real-operator-active-b",
        quarantine=provider._accept_async_quarantine,
        _managed_guard=_ManagedResourceAdmissionGuard(gate),
        _managed_epoch=epoch,
    )
    setup_token = gate.admit_lease(epoch, "operator_call")
    try:
        owner.mark_enqueued()
    finally:
        gate.release(setup_token)
    call = _OperatorCall(owner, None, None)
    owner._operator_call = call
    lease = object.__new__(WorkingSetLease)
    lease._provider = provider
    lease._epoch = epoch
    lease._poisoned_error = None
    lease._active_operator_owner = owner
    lease._status_workspace = None
    lease.scheduler = SimpleNamespace(_quarantined_owners=[owner])
    lease.pool = None
    provider._active_lease = lease

    operator = object.__new__(DistributedLocalOperator)
    operator.plan = SimpleNamespace(
        variable_key="variable",
        block_plan=lambda local_rank, source_rank: SimpleNamespace(),
    )
    operator.provider = lease
    operator.collective = wrapper
    operator.backend = backend
    operator.context = wrapper._context
    operator._output_accumulator = np.zeros((2,), dtype=np.float64)
    operator._active_allocation_scope = None
    operator._active_contribution = None
    operator._active_call_ready_error = lambda local_vector: None
    operator._allreduce_active_status = lambda active_call, error: False
    operator._ensure_call_resources = lambda scope: None
    operator._discard_active_call_resources = lambda: None
    operator._prepare_call_bindings = lambda local_vector, active_call: (
        nullcontext(),
        (
            SimpleNamespace(
                bindings=ExecutionBindings(
                    {"variable": np.ones((2,), dtype=np.float64)}
                )
            ),
        ),
    )

    monkeypatch.setattr(wrapper, "_validate_array", lambda *args, **kwargs: None)

    def fail_broadcast(*args, **kwargs):
        raise primary

    monkeypatch.setattr(backend, "broadcast", fail_broadcast)
    original_transition_publish = runtime._publish_communicator_fatal_transition

    def record_transition_publish(transition):
        transition_publications.append(transition)
        return original_transition_publish(transition)

    monkeypatch.setattr(
        runtime,
        "_publish_communicator_fatal_transition",
        record_transition_publish,
    )

    def legacy_handler(error):
        legacy_calls.append(error)
        return runtime._enter_communicator_fatal(error)

    wrapper._install_fatal_handler(legacy_handler)
    original_agree = wrapper._agree_admitted_active_broadcast

    def pause_before_agree(failed):
        before_agree.set()
        assert release_agree.wait(_TASK_18_2_TIMEOUT_S * 2)
        return original_agree(failed)

    monkeypatch.setattr(
        wrapper, "_agree_admitted_active_broadcast", pause_before_agree
    )

    def hard_exit():
        hard_exits.append(threading.current_thread().name)
        raise AssertionError("unexpected real operator hard exit")

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    def run_operator():
        token = gate.admit_lease(epoch, "operator_call")
        try:
            return operator._call_active(
                np.ones((2,), dtype=np.float64),
                lambda local_vector: nullcontext(call),
            )
        finally:
            runtime._release_admission(token)

    worker, results, errors, done = _start_task_18_2_call(
        run_operator,
        name="task-18.2-real-operator-deferred-quarantine",
    )
    try:
        assert before_agree.wait(_TASK_18_2_TIMEOUT_S), errors
        assert gate.phase is _TerminalPhase.FATAL_PENDING
        assert owner.state == "quarantined"
        assert owner.error is primary
        assert call.primary_error is primary
        assert runtime._terminal_quarantine.first_error is primary
        assert runtime._terminal_quarantine.owners == (owner,)
        assert runtime._terminal_error is None
        assert getattr(backend, "_execution_terminal_error", None) is None
        assert provider._terminal_error is None
        assert lease._poisoned_error is None
        assert wrapper._fatal_error is None
        assert transition_publications == []
        assert legacy_calls == []
    finally:
        release_agree.set()
        _join_task_18_2_call(worker, done)

    assert results == []
    assert errors == [primary]
    assert len(transition_publications) == 1
    assert transition_publications[0].primary is primary
    assert legacy_calls == [primary]
    assert gate.wait_for_published(_TASK_18_2_TIMEOUT_S) is primary
    assert runtime._terminal_error is primary
    assert backend._execution_terminal_error is primary
    assert provider._terminal_error is primary
    assert lease._poisoned_error is primary
    assert wrapper._fatal_error is primary
    assert wrapper._fatal_publications == 0
    assert hard_exits == []


def test_fatal_owner_cancellation_retains_owner_and_bounds_close(monkeypatch):
    class PublicationCancelled(BaseException):
        pass

    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    primary = RuntimeError("cancelled fatal publication")
    cancellation = PublicationCancelled("cancel at active-B barrier")
    failure_signals = []
    hard_exit_observations = []
    publisher_name = "task-18.2-cancelled-fatal-owner"
    close_name = "task-18.2-cancelled-fatal-close"

    def cancel_barrier(**_kwargs):
        raise cancellation

    monkeypatch.setattr(
        wrapper, "_wait_for_active_broadcast_agreements", cancel_barrier
    )
    original_fail = gate._fail_fatal_publication

    def record_failure(transition, failure):
        failure_signals.append((transition, failure))
        return original_fail(transition, failure)

    monkeypatch.setattr(gate, "_fail_fatal_publication", record_failure)

    def hard_exit():
        hard_exit_observations.append(
            (
                threading.current_thread().name,
                wrapper._fatal_publications,
                wrapper._fatal_publication_failure,
                gate.phase,
            )
        )
        raise FatalHardExit(threading.current_thread().name)

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)
    publisher, publish_results, publish_errors, publish_done = (
        _start_task_18_2_call(
            lambda: runtime._enter_communicator_fatal(primary),
            name=publisher_name,
        )
    )
    _join_task_18_2_call(publisher, publish_done)

    assert publish_results == []
    assert len(publish_errors) == 1
    assert isinstance(publish_errors[0], FatalHardExit)
    assert wrapper._fatal_publications == 1
    assert wrapper._fatal_publication_failure is primary
    assert gate.phase is _TerminalPhase.FATAL_PENDING
    assert len(failure_signals) == 1
    assert failure_signals[0][0].primary is primary
    assert failure_signals[0][1] is primary

    closer, close_results, close_errors, close_done = _start_task_18_2_call(
        runtime.close, name=close_name
    )
    _join_task_18_2_call(closer, close_done)

    assert close_results == []
    assert len(close_errors) == 1
    assert close_errors[0] is primary
    assert hard_exit_observations == [
        (publisher_name, 1, primary, _TerminalPhase.FATAL_PENDING),
    ]
    assert wrapper._fatal_secondary_errors == (cancellation,)


def test_deferred_gate_completion_failure_retains_owner_and_bounds_close(
    monkeypatch,
):
    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, backend, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    primary = RuntimeError("deferred completion fatal primary")
    completion_error = RuntimeError("injected gate completion failure")
    completion_entered = threading.Event()
    failure_signals = []
    hard_exit_observations = []
    deferrals = []
    publisher_name = "task-18.2-deferred-completion-owner"
    close_name = "task-18.2-deferred-completion-close"

    monkeypatch.setattr(wrapper, "_validate_array", lambda *args, **kwargs: None)

    def fail_broadcast(*args, **kwargs):
        raise primary

    monkeypatch.setattr(backend, "broadcast", fail_broadcast)

    def fail_gate_completion(transition, snapshot):
        completion_entered.set()
        raise completion_error

    monkeypatch.setattr(gate, "publish_fatal", fail_gate_completion)
    original_fail = gate._fail_fatal_publication

    def record_failure(transition, failure):
        failure_signals.append((transition, failure))
        return original_fail(transition, failure)

    monkeypatch.setattr(gate, "_fail_fatal_publication", record_failure)

    def hard_exit():
        hard_exit_observations.append(
            (
                threading.current_thread().name,
                wrapper._fatal_publications,
                wrapper._fatal_publication_failure,
                wrapper._fatal_protocol_completed,
                gate.phase,
            )
        )
        raise FatalHardExit(threading.current_thread().name)

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    def run_active_broadcast():
        with wrapper._active_broadcast_admission() as (broadcast, agree):
            try:
                broadcast(np.ones((2,), dtype=np.float64), root=0)
            except BaseException as error:
                assert error is primary
                assert runtime._enter_communicator_fatal(error) is primary
                deferrals.append(wrapper._active_broadcast_local.deferral)
                try:
                    agree(True)
                except BaseException:
                    raise primary
                raise primary

    publisher, publish_results, publish_errors, publish_done = (
        _start_task_18_2_call(run_active_broadcast, name=publisher_name)
    )
    assert completion_entered.wait(_TASK_18_2_TIMEOUT_S)
    _join_task_18_2_call(publisher, publish_done)

    assert publish_results == []
    assert len(publish_errors) == 1
    assert isinstance(publish_errors[0], FatalHardExit)
    assert deferrals[0].completed is False
    assert wrapper._fatal_protocol_completed is True
    assert wrapper._fatal_error is primary
    assert wrapper._fatal_publications == 1
    assert wrapper._fatal_publication_failure is primary
    assert gate.phase is _TerminalPhase.FATAL_PENDING
    assert len(failure_signals) == 1
    assert failure_signals[0][0].primary is primary
    assert failure_signals[0][1] is primary

    closer, close_results, close_errors, close_done = _start_task_18_2_call(
        runtime.close, name=close_name
    )
    _join_task_18_2_call(closer, close_done)

    assert close_results == []
    assert len(close_errors) == 1
    assert close_errors[0] is primary
    assert hard_exit_observations == [
        (
            publisher_name,
            1,
            primary,
            True,
            _TerminalPhase.FATAL_PENDING,
        ),
    ]
    assert completion_error in wrapper._fatal_secondary_errors


def test_immediate_post_pre_reservation_failure_recovers_exact_gate_and_close(
    monkeypatch,
):
    class ElectionCancelled(BaseException):
        pass

    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, _, _, raw_comm = _single_rank_task_18_2_runtime(
        monkeypatch
    )
    gate = runtime._terminal_gate
    primary = RuntimeError("immediate post-reservation primary")
    later = RuntimeError("immediate post-reservation later reporter")
    cancellation = ElectionCancelled("cancel immediately after pre-reservation")
    publisher_name = "task-18.2-immediate-pre-reservation-owner"
    reporter_name = "task-18.2-immediate-pre-reservation-reporter"
    close_name = "task-18.2-immediate-pre-reservation-close"
    tokens = []
    injection_observations = []
    reporter_adoptions = []
    join_observations = []
    failure_signals = []
    hard_exit_observations = []

    original_pre_reserve = wrapper._pre_reserve_fatal_publication

    def cancel_after_real_pre_reservation(
        error, origin_rank=None, *, election=None
    ):
        result = original_pre_reserve(
            error, origin_rank, election=election
        )
        if threading.current_thread().name == publisher_name:
            with wrapper._fatal_condition:
                owner = wrapper._fatal_publication_owner_reservation
                owner_state = (
                    wrapper._fatal_publications,
                    owner,
                    None if owner is None else owner.adopted,
                    wrapper._fatal_pending_primary,
                )
            with gate._condition:
                token_state = gate._tokens[tokens[0].sequence]
                gate_state = (
                    gate._phase,
                    gate._fatal_transition,
                    token_state.status,
                )
            injection_observations.append((*owner_state, *gate_state))
            raise cancellation
        return result

    monkeypatch.setattr(
        wrapper,
        "_pre_reserve_fatal_publication",
        cancel_after_real_pre_reservation,
    )
    original_begin_publication = wrapper._begin_fatal_publication

    def record_reporter_adoption(*args, **kwargs):
        result = original_begin_publication(*args, **kwargs)
        if threading.current_thread().name == reporter_name:
            reporter_adoptions.append(result)
        return result

    monkeypatch.setattr(
        wrapper, "_begin_fatal_publication", record_reporter_adoption
    )
    original_wait = wrapper._wait_for_joined_fatal_publication

    def record_join(**kwargs):
        with wrapper._fatal_condition:
            join_observations.append(
                (
                    threading.current_thread().name,
                    wrapper._fatal_publications,
                    wrapper._fatal_publication_failure,
                )
            )
        return original_wait(**kwargs)

    monkeypatch.setattr(
        wrapper, "_wait_for_joined_fatal_publication", record_join
    )
    original_fail = gate._fail_fatal_publication

    def record_failure(transition, failure):
        failure_signals.append(
            (
                transition,
                failure,
                wrapper._fatal_condition._is_owned(),
                wrapper._fatal_lock._is_owned(),
            )
        )
        return original_fail(transition, failure)

    monkeypatch.setattr(gate, "_fail_fatal_publication", record_failure)

    def hard_exit():
        hard_exit_observations.append(
            (
                threading.current_thread().name,
                wrapper._fatal_publications,
                wrapper._fatal_publication_failure,
                wrapper._fatal_condition._is_owned(),
                wrapper._fatal_lock._is_owned(),
                gate._condition._is_owned(),
            )
        )
        raise FatalHardExit(threading.current_thread().name)

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    def publish_from_admission():
        token = gate.admit_runtime("immediate_pre_reservation_discoverer")
        tokens.append(token)
        return runtime._enter_communicator_fatal(
            primary, discovering_token=token
        )

    publisher, publish_results, publish_errors, publish_done = (
        _start_task_18_2_call(publish_from_admission, name=publisher_name)
    )
    _join_task_18_2_call(publisher, publish_done)

    assert publish_results == []
    assert len(publish_errors) == 1
    assert isinstance(publish_errors[0], FatalHardExit)
    assert len(injection_observations) == 1
    (
        owner_count,
        retained_owner,
        adopted,
        pending_primary,
        injection_phase,
        injection_transition,
        injection_token_status,
    ) = injection_observations[0]
    assert owner_count == 1
    assert retained_owner.owner_thread is publisher
    assert adopted is False
    assert pending_primary is primary
    assert injection_phase is _TerminalPhase.HEALTHY
    assert injection_transition is None
    assert injection_token_status == "active"

    transition = gate._fatal_transition
    assert transition.primary is primary
    assert transition is retained_owner.election.prepared_transition
    with gate._condition:
        assert gate._phase is _TerminalPhase.FATAL_PENDING
        assert gate._fatal_transition is transition
        assert gate._fatal_publication_failure is primary
        assert gate._tokens[tokens[0].sequence].status == "converted"
        assert gate._has_active_tokens() is False
    with pytest.raises(RuntimeError, match="converted"):
        gate.release(tokens[0])
    with wrapper._fatal_condition:
        assert wrapper._fatal_publications == 1
        assert wrapper._fatal_publication_owner_reservation is retained_owner
        assert wrapper._fatal_publication_failure is primary
    assert len(failure_signals) == 1
    assert failure_signals[0] == (transition, primary, False, False)

    reporter, report_results, report_errors, report_done = (
        _start_task_18_2_call(
            lambda: runtime._enter_communicator_fatal(later),
            name=reporter_name,
        )
    )
    _join_task_18_2_call(reporter, report_done)

    assert report_results == []
    assert len(report_errors) == 1
    assert report_errors[0] is primary
    assert len(reporter_adoptions) == 1
    assert reporter_adoptions[0][2:] == (False, True, False)

    closer, close_results, close_errors, close_done = _start_task_18_2_call(
        runtime.close, name=close_name
    )
    _join_task_18_2_call(closer, close_done)

    assert close_results == []
    assert len(close_errors) == 1
    assert close_errors[0] is primary
    assert join_observations == [
        (reporter_name, 1, primary),
        (close_name, 1, primary),
    ]
    assert hard_exit_observations == [
        (publisher_name, 1, primary, False, False, False),
    ]
    assert len(failure_signals) == 1
    assert cancellation in wrapper._fatal_secondary_errors
    assert raw_comm.abort_calls == 0
    assert wrapper._fatal_error is None
    assert runtime._terminal_error is None


def test_hidden_fatal_preparation_escape_repairs_owner_before_handoff_release(
    monkeypatch,
):
    from renormalizer.backend._distributed import collectives as collectives_module
    import renormalizer.backend.distributed_runtime as runtime_module

    class ElectionCancelled(BaseException):
        pass

    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, _, _, raw_comm = _single_rank_task_18_2_runtime(
        monkeypatch
    )
    gate = runtime._terminal_gate
    handoff = wrapper._fatal_monitor_handoff
    primary = RuntimeError("pre-side-effect fatal primary")
    later = RuntimeError("pre-side-effect later reporter")
    cancellation = ElectionCancelled("cancel before real pre-reservation")
    publisher_name = "task-18.2-pre-side-effect-owner"
    clean_name = "task-18.2-pre-side-effect-clean"
    reporter_name = "task-18.2-pre-side-effect-reporter"
    close_name = "task-18.2-pre-side-effect-close"
    pre_reserve_entered = threading.Event()
    clean_attempting = threading.Event()
    reporter_adoptions = []
    join_observations = []
    hard_exit_observations = []
    tokens = []

    monkeypatch.setattr(collectives_module, "_FATAL_TIMEOUT_S", 0.25)
    monkeypatch.setattr(runtime_module, "_TERMINAL_TIMEOUT_S", 0.25)
    generation = wrapper._request_fatal_monitor_stop()
    original_pre_reserve = wrapper._pre_reserve_fatal_publication

    def cancel_before_real_pre_reservation(*args, **kwargs):
        if threading.current_thread().name != publisher_name:
            return original_pre_reserve(*args, **kwargs)
        with wrapper._fatal_condition:
            assert wrapper._fatal_publications == 0
            assert wrapper._fatal_publication_owner_reservation is None
            assert wrapper._fatal_pending_primary is None
        pre_reserve_entered.set()
        assert clean_attempting.wait(_TASK_18_2_TIMEOUT_S)
        raise cancellation

    monkeypatch.setattr(
        wrapper,
        "_pre_reserve_fatal_publication",
        cancel_before_real_pre_reservation,
    )
    original_begin_publication = wrapper._begin_fatal_publication

    def record_reporter_adoption(*args, **kwargs):
        result = original_begin_publication(*args, **kwargs)
        if threading.current_thread().name == reporter_name:
            reporter_adoptions.append(result)
        return result

    monkeypatch.setattr(
        wrapper, "_begin_fatal_publication", record_reporter_adoption
    )
    original_join = wrapper._wait_for_joined_fatal_publication

    def record_join(**kwargs):
        with wrapper._fatal_condition:
            join_observations.append(
                (
                    threading.current_thread().name,
                    wrapper._fatal_publications,
                    wrapper._fatal_publication_failure,
                )
            )
        return original_join(**kwargs)

    monkeypatch.setattr(
        wrapper, "_wait_for_joined_fatal_publication", record_join
    )

    def hard_exit():
        hard_exit_observations.append(
            (
                threading.current_thread().name,
                wrapper._fatal_condition._is_owned(),
                wrapper._fatal_lock._is_owned(),
                gate._condition._is_owned(),
                handoff._condition._is_owned(),
            )
        )
        raise FatalHardExit(threading.current_thread().name)

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    def publish_before_pre_reservation():
        token = gate.admit_runtime("pre_side_effect_fatal")
        tokens.append(token)
        return runtime._enter_communicator_fatal(
            primary, discovering_token=token
        )

    publisher, publish_results, publish_errors, publish_done = (
        _start_task_18_2_call(
            publish_before_pre_reservation,
            name=publisher_name,
        )
    )
    cleaner = None
    try:
        assert pre_reserve_entered.wait(_TASK_18_2_TIMEOUT_S)

        def select_clean():
            clean_attempting.set()
            return wrapper._select_fatal_monitor_outcome(
                "stopped_clean", generation
            )

        cleaner, clean_results, clean_errors, clean_done = (
            _start_task_18_2_call(select_clean, name=clean_name)
        )
        _join_task_18_2_call(cleaner, clean_done)
        _join_task_18_2_call(publisher, publish_done)
    finally:
        clean_attempting.set()
        if cleaner is not None and cleaner.is_alive():
            _join_task_18_2_call(cleaner, clean_done)
        if publisher.is_alive():
            _join_task_18_2_call(publisher, publish_done)

    with wrapper._fatal_condition:
        retained_owner = wrapper._fatal_publication_owner_reservation
        owner_state_before_reporter = (
            wrapper._fatal_publications,
            retained_owner,
            wrapper._fatal_pending_primary,
            wrapper._fatal_publication_failure,
        )
    with gate._condition:
        transition_before_reporter = gate._fatal_transition
        gate_state_before_reporter = (
            gate._phase,
            transition_before_reporter,
            gate._fatal_publication_failure,
            gate._tokens[tokens[0].sequence].status,
        )

    reporter, report_results, report_errors, report_done = (
        _start_task_18_2_call(
            lambda: runtime._enter_communicator_fatal(later),
            name=reporter_name,
        )
    )
    _join_task_18_2_call(reporter, report_done)
    closer, close_results, close_errors, close_done = _start_task_18_2_call(
        runtime.close, name=close_name
    )
    _join_task_18_2_call(closer, close_done)

    assert clean_errors == []
    assert len(clean_results) == 1
    outcome = clean_results[0]
    assert outcome.kind == "fatal_elected"
    assert outcome.primary is primary
    assert handoff.observe_outcome() is outcome
    assert publish_results == []
    assert len(publish_errors) == 1
    assert isinstance(publish_errors[0], FatalHardExit)
    owner_count, retained_owner, pending, publication_failure = (
        owner_state_before_reporter
    )
    assert owner_count == 1
    assert retained_owner.owner_thread is publisher
    assert retained_owner.primary is primary
    assert retained_owner.election.primary is primary
    assert pending is primary
    assert publication_failure is primary
    phase, transition, gate_failure, token_status = gate_state_before_reporter
    assert phase is _TerminalPhase.FATAL_PENDING
    assert transition is transition_before_reporter
    assert transition.primary is primary
    assert gate_failure is primary
    assert token_status == "converted"
    assert runtime._pending_fatal_collective is wrapper
    assert runtime._pending_fatal_context is not None
    assert report_results == []
    assert len(report_errors) == 1
    assert report_errors[0] is primary
    assert len(reporter_adoptions) == 1
    assert reporter_adoptions[0][0] is primary
    assert reporter_adoptions[0][2:] == (False, True, False)
    assert close_results == []
    assert len(close_errors) == 1
    assert close_errors[0] is primary
    assert join_observations == [
        (reporter_name, 1, primary),
        (close_name, 1, primary),
    ]
    assert hard_exit_observations == [
        (publisher_name, False, False, False, False),
    ]
    assert cancellation in wrapper._fatal_secondary_errors
    assert later in wrapper._fatal_secondary_errors
    assert raw_comm.abort_calls == 0
    assert wrapper._fatal_error is None
    assert runtime._terminal_error is None


def test_pre_reserve_escape_backs_transition_token_and_context_before_handoff_release(
    monkeypatch,
):
    from renormalizer.backend._distributed import collectives as collectives_module
    import renormalizer.backend.distributed_runtime as runtime_module

    class ElectionCancelled(BaseException):
        pass

    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, _, _, raw_comm = _single_rank_task_18_2_runtime(
        monkeypatch
    )
    gate = runtime._terminal_gate
    handoff = wrapper._fatal_monitor_handoff
    primary = RuntimeError("fully backed pre-reserve fatal primary")
    later = RuntimeError("fully backed later reporter")
    cancellation = ElectionCancelled("cancel before real pre-reservation")
    publisher_name = "task-18.2-fully-backed-owner"
    clean_name = "task-18.2-fully-backed-clean"
    reporter_name = "task-18.2-fully-backed-reporter"
    close_name = "task-18.2-fully-backed-close"
    pre_reserve_entered = threading.Event()
    clean_attempting = threading.Event()
    outer_recovery_entered = threading.Event()
    release_outer_recovery = threading.Event()
    recovery_observations = []
    quarantine_observations = []
    collective_failure_signals = []
    gate_failure_signals = []
    reporter_adoptions = []
    join_observations = []
    hard_exit_observations = []
    tokens = []

    monkeypatch.setattr(collectives_module, "_FATAL_TIMEOUT_S", 0.25)
    monkeypatch.setattr(runtime_module, "_TERMINAL_TIMEOUT_S", 0.25)
    generation = wrapper._request_fatal_monitor_stop()
    original_pre_reserve = wrapper._pre_reserve_fatal_publication

    def cancel_before_real_pre_reservation(*args, **kwargs):
        if threading.current_thread().name != publisher_name:
            return original_pre_reserve(*args, **kwargs)
        with wrapper._fatal_condition:
            assert wrapper._fatal_publications == 0
            assert wrapper._fatal_publication_owner_reservation is None
            assert wrapper._fatal_pending_primary is None
        pre_reserve_entered.set()
        assert clean_attempting.wait(_TASK_18_2_TIMEOUT_S)
        raise cancellation

    monkeypatch.setattr(
        wrapper,
        "_pre_reserve_fatal_publication",
        cancel_before_real_pre_reservation,
    )
    original_recover_locked = (
        runtime_module._RecoverableFatalElection.recover_locked
    )

    def pause_first_outer_recovery(election):
        if (
            threading.current_thread().name == publisher_name
            and not recovery_observations
        ):
            recovery_observations.append(
                (
                    election,
                    gate._condition._is_owned(),
                    runtime._terminal_state_lock._is_owned(),
                    handoff._condition._is_owned(),
                    wrapper._fatal_condition._is_owned(),
                )
            )
            outer_recovery_entered.set()
            assert release_outer_recovery.wait(_TASK_18_2_TIMEOUT_S * 2)
        return original_recover_locked(election)

    monkeypatch.setattr(
        runtime_module._RecoverableFatalElection,
        "recover_locked",
        pause_first_outer_recovery,
    )
    original_stage_quarantine = (
        runtime._stage_active_broadcast_quarantine_locked
    )

    def record_quarantine_stage(owner, collective):
        token_state = gate._tokens[tokens[0].sequence]
        quarantine_observations.append(
            (
                threading.current_thread().name,
                runtime._pending_fatal_collective,
                runtime._pending_fatal_context,
                gate._phase,
                gate._fatal_transition,
                token_state.status,
                gate._condition._is_owned(),
                runtime._terminal_state_lock._is_owned(),
                handoff._condition._is_owned(),
                wrapper._fatal_condition._is_owned(),
            )
        )
        return original_stage_quarantine(owner, collective)

    monkeypatch.setattr(
        runtime,
        "_stage_active_broadcast_quarantine_locked",
        record_quarantine_stage,
    )
    original_fail_reserved = wrapper._fail_reserved_fatal_publication

    def record_collective_failure(*args, **kwargs):
        result = original_fail_reserved(*args, **kwargs)
        if result[1]:
            collective_failure_signals.append(
                (
                    threading.current_thread().name,
                    wrapper._fatal_publication_failure,
                    wrapper._fatal_condition._is_owned(),
                    gate._condition._is_owned(),
                    handoff._condition._is_owned(),
                )
            )
        return result

    monkeypatch.setattr(
        wrapper,
        "_fail_reserved_fatal_publication",
        record_collective_failure,
    )
    original_gate_fail = gate._fail_fatal_publication

    def record_gate_failure(transition, failure):
        gate_failure_signals.append(
            (
                transition,
                failure,
                wrapper._fatal_condition._is_owned(),
                gate._condition._is_owned(),
                handoff._condition._is_owned(),
            )
        )
        return original_gate_fail(transition, failure)

    monkeypatch.setattr(gate, "_fail_fatal_publication", record_gate_failure)
    original_begin_publication = wrapper._begin_fatal_publication

    def record_reporter_adoption(*args, **kwargs):
        result = original_begin_publication(*args, **kwargs)
        if threading.current_thread().name == reporter_name:
            reporter_adoptions.append(result)
        return result

    monkeypatch.setattr(
        wrapper, "_begin_fatal_publication", record_reporter_adoption
    )
    original_join = wrapper._wait_for_joined_fatal_publication

    def record_join(**kwargs):
        with wrapper._fatal_condition:
            join_observations.append(
                (
                    threading.current_thread().name,
                    wrapper._fatal_publications,
                    wrapper._fatal_publication_failure,
                )
            )
        return original_join(**kwargs)

    monkeypatch.setattr(
        wrapper, "_wait_for_joined_fatal_publication", record_join
    )

    def hard_exit():
        hard_exit_observations.append(
            (
                threading.current_thread().name,
                wrapper._fatal_condition._is_owned(),
                wrapper._fatal_lock._is_owned(),
                gate._condition._is_owned(),
                runtime._terminal_state_lock._is_owned(),
                handoff._condition._is_owned(),
            )
        )
        raise FatalHardExit(threading.current_thread().name)

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    def publish_before_pre_reservation():
        token = gate.admit_runtime("fully_back_pre_reserve_fatal")
        tokens.append(token)
        return runtime._enter_communicator_fatal(
            primary, discovering_token=token
        )

    publisher, publish_results, publish_errors, publish_done = (
        _start_task_18_2_call(
            publish_before_pre_reservation,
            name=publisher_name,
        )
    )
    cleaner = None
    try:
        assert pre_reserve_entered.wait(_TASK_18_2_TIMEOUT_S)

        def select_clean():
            clean_attempting.set()
            return wrapper._select_fatal_monitor_outcome(
                "stopped_clean", generation
            )

        cleaner, clean_results, clean_errors, clean_done = (
            _start_task_18_2_call(select_clean, name=clean_name)
        )
        assert outer_recovery_entered.wait(_TASK_18_2_TIMEOUT_S)
        _join_task_18_2_call(cleaner, clean_done)
        with wrapper._fatal_condition:
            retained_owner = wrapper._fatal_publication_owner_reservation
            owner_state = (
                wrapper._fatal_publications,
                retained_owner,
                wrapper._fatal_pending_primary,
                wrapper._fatal_publication_failure,
            )
        election = recovery_observations[0][0]
        pre_release_state = (
            handoff.observe_outcome(),
            gate._phase,
            gate._fatal_transition,
            gate._tokens[tokens[0].sequence].status,
            runtime._pending_fatal_collective,
            runtime._pending_fatal_context,
            election.transition,
            election.prepared_transition,
            election.context_retained,
            election.context_repair_attempted,
            tuple(quarantine_observations),
            tuple(collective_failure_signals),
            tuple(gate_failure_signals),
            publish_done.is_set(),
        )
    finally:
        clean_attempting.set()
        release_outer_recovery.set()
        if cleaner is not None and cleaner.is_alive():
            _join_task_18_2_call(cleaner, clean_done)
        if publisher.is_alive():
            _join_task_18_2_call(publisher, publish_done)

    reporter, report_results, report_errors, report_done = (
        _start_task_18_2_call(
            lambda: runtime._enter_communicator_fatal(later),
            name=reporter_name,
        )
    )
    _join_task_18_2_call(reporter, report_done)
    closer, close_results, close_errors, close_done = _start_task_18_2_call(
        runtime.close, name=close_name
    )
    _join_task_18_2_call(closer, close_done)

    assert clean_errors == []
    assert len(clean_results) == 1
    outcome = clean_results[0]
    assert outcome.kind == "fatal_elected"
    assert outcome.primary is primary
    assert pre_release_state[0] is outcome
    owner_count, retained_owner, pending, publication_failure = owner_state
    assert owner_count == 1
    assert retained_owner.owner_thread is publisher
    assert retained_owner.primary is primary
    assert retained_owner.election is recovery_observations[0][0]
    assert pending is primary
    assert publication_failure is None
    (
        _,
        phase,
        transition,
        token_status,
        pending_collective,
        pending_context,
        election_transition,
        prepared_transition,
        context_retained,
        context_repair_attempted,
        quarantine_before_outer_recovery,
        collective_failures_before_outer_recovery,
        gate_failures_before_outer_recovery,
        publisher_finished_before_outer_recovery,
    ) = pre_release_state
    assert phase is _TerminalPhase.FATAL_PENDING
    assert transition is prepared_transition
    assert election_transition is transition
    assert transition.primary is primary
    assert token_status == "converted"
    assert pending_collective is wrapper
    election = recovery_observations[0][0]
    assert pending_context == (
        election.provider,
        election.lease,
        election.owner,
    )
    assert context_retained is True
    assert context_repair_attempted is True
    assert quarantine_before_outer_recovery == (
        (
            publisher_name,
            wrapper,
            pending_context,
            _TerminalPhase.FATAL_PENDING,
            transition,
            "converted",
            True,
            True,
            True,
            False,
        ),
    )
    assert collective_failures_before_outer_recovery == ()
    assert gate_failures_before_outer_recovery == ()
    assert publisher_finished_before_outer_recovery is False
    assert recovery_observations[0][1:] == (True, True, False, False)
    assert publish_results == []
    assert len(publish_errors) == 1
    assert isinstance(publish_errors[0], FatalHardExit)
    assert report_results == []
    assert len(report_errors) == 1
    assert report_errors[0] is primary
    assert len(reporter_adoptions) == 1
    assert reporter_adoptions[0][0] is primary
    assert reporter_adoptions[0][2:] == (False, True, False)
    assert close_results == []
    assert len(close_errors) == 1
    assert close_errors[0] is primary
    assert collective_failure_signals == [
        (publisher_name, primary, False, False, False)
    ]
    assert gate_failure_signals == [
        (transition, primary, False, False, False)
    ]
    assert sum(
        observation[0] == publisher_name
        for observation in quarantine_observations
    ) == 1
    assert join_observations == [
        (reporter_name, 1, primary),
        (close_name, 1, primary),
    ]
    assert hard_exit_observations == [
        (publisher_name, False, False, False, False, False),
    ]
    assert sum(
        retained is cancellation
        for retained in wrapper._fatal_secondary_errors
    ) == 1
    assert sum(
        retained is later for retained in wrapper._fatal_secondary_errors
    ) == 1
    assert raw_comm.abort_calls == 0
    assert wrapper._fatal_error is None
    assert runtime._terminal_error is None


def test_hidden_fatal_reservation_preempts_clean_after_preparation_escape(
    monkeypatch,
):
    import inspect
    from renormalizer.backend._distributed import collectives as collectives_module
    import renormalizer.backend.distributed_runtime as runtime_module

    class ElectionCancelled(BaseException):
        pass

    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, _, _, raw_comm = _single_rank_task_18_2_runtime(
        monkeypatch
    )
    gate = runtime._terminal_gate
    handoff = wrapper._fatal_monitor_handoff
    primary = RuntimeError("monotonic hidden fatal primary")
    later = RuntimeError("monotonic hidden fatal later reporter")
    cancellation = ElectionCancelled("cancel before fatal outcome assignment")
    publisher_name = "task-18.2-monotonic-handoff-owner"
    clean_name = "task-18.2-monotonic-handoff-clean"
    reporter_name = "task-18.2-monotonic-handoff-reporter"
    close_name = "task-18.2-monotonic-handoff-close"
    preparation_completed = threading.Event()
    clean_attempting = threading.Event()
    clean_selected = threading.Event()
    recovery_entered = threading.Event()
    injected = []
    failure_signals = []
    join_observations = []
    hard_exit_observations = []
    close_commit_entered = threading.Event()
    teardown_failure_markers = []

    monkeypatch.setattr(collectives_module, "_FATAL_TIMEOUT_S", 0.25)
    monkeypatch.setattr(runtime_module, "_TERMINAL_TIMEOUT_S", 0.25)
    generation = wrapper._request_fatal_monitor_stop()

    select_source, select_start = inspect.getsourcelines(type(handoff).select_fatal)
    assignment_line = next(
        select_start + offset
        for offset, line in enumerate(select_source)
        if "self._outcome =" in line
    )
    select_code = type(handoff).select_fatal.__code__

    original_recover_locked = runtime_module._RecoverableFatalElection.recover_locked
    recovery_delayed = []

    def delay_recovery_until_clean_selection(election):
        if (
            threading.current_thread().name == publisher_name
            and not recovery_delayed
        ):
            recovery_delayed.append(election)
            recovery_entered.set()
            assert clean_selected.wait(_TASK_18_2_TIMEOUT_S)
        return original_recover_locked(election)

    monkeypatch.setattr(
        runtime_module._RecoverableFatalElection,
        "recover_locked",
        delay_recovery_until_clean_selection,
    )
    original_gate_fail = gate._fail_fatal_publication

    def record_gate_failure(transition, failure):
        failure_signals.append(
            (
                transition,
                failure,
                wrapper._fatal_condition._is_owned(),
                wrapper._fatal_lock._is_owned(),
                gate._condition._is_owned(),
                handoff._condition._is_owned(),
            )
        )
        return original_gate_fail(transition, failure)

    monkeypatch.setattr(gate, "_fail_fatal_publication", record_gate_failure)
    original_commit_runtime_close = gate.commit_runtime_close

    def record_close_commit(*args, **kwargs):
        close_commit_entered.set()
        return original_commit_runtime_close(*args, **kwargs)

    monkeypatch.setattr(gate, "commit_runtime_close", record_close_commit)
    original_join = wrapper._wait_for_joined_fatal_publication

    def record_join(**kwargs):
        with wrapper._fatal_condition:
            join_observations.append(
                (
                    threading.current_thread().name,
                    wrapper._fatal_publications,
                    wrapper._fatal_publication_failure,
                )
            )
        return original_join(**kwargs)

    monkeypatch.setattr(
        wrapper, "_wait_for_joined_fatal_publication", record_join
    )

    def hard_exit():
        hard_exit_observations.append(
            (
                threading.current_thread().name,
                wrapper._fatal_condition._is_owned(),
                wrapper._fatal_lock._is_owned(),
                gate._condition._is_owned(),
                handoff._condition._is_owned(),
            )
        )
        raise FatalHardExit(threading.current_thread().name)

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    def publish_with_boundary_cancellation():
        def cancel_at_assignment(frame, event, _arg):
            if (
                event == "line"
                and frame.f_code is select_code
                and frame.f_lineno == assignment_line
                and not injected
            ):
                injected.append(cancellation)
                preparation_completed.set()
                assert clean_attempting.wait(_TASK_18_2_TIMEOUT_S)
                raise cancellation
            return cancel_at_assignment

        sys.settrace(cancel_at_assignment)
        try:
            return runtime._enter_communicator_fatal(primary)
        finally:
            sys.settrace(None)

    publisher, publish_results, publish_errors, publish_done = (
        _start_task_18_2_call(
            publish_with_boundary_cancellation,
            name=publisher_name,
        )
    )
    cleaner = None
    try:
        assert preparation_completed.wait(_TASK_18_2_TIMEOUT_S)

        def select_clean():
            clean_attempting.set()
            try:
                return wrapper._select_fatal_monitor_outcome(
                    "stopped_clean", generation
                )
            finally:
                clean_selected.set()

        cleaner, clean_results, clean_errors, clean_done = (
            _start_task_18_2_call(select_clean, name=clean_name)
        )
        _join_task_18_2_call(cleaner, clean_done)
        _join_task_18_2_call(publisher, publish_done)
    finally:
        clean_attempting.set()
        clean_selected.set()
        if cleaner is not None and cleaner.is_alive():
            _join_task_18_2_call(cleaner, clean_done)
        if publisher.is_alive():
            _join_task_18_2_call(publisher, publish_done)

    reporter, report_results, report_errors, report_done = (
        _start_task_18_2_call(
            lambda: runtime._enter_communicator_fatal(later),
            name=reporter_name,
        )
    )
    _join_task_18_2_call(reporter, report_done)
    closer, close_results, close_errors, close_done = _start_task_18_2_call(
        runtime.close, name=close_name
    )
    assert close_commit_entered.wait(_TASK_18_2_TIMEOUT_S)
    with gate._condition:
        if gate._fatal_publication_failure is not primary:
            teardown_failure_markers.append(gate._fatal_publication_failure)
            original_gate_fail(gate._fatal_transition, primary)
    _join_task_18_2_call(closer, close_done)

    assert injected == [cancellation]
    assert teardown_failure_markers == []
    assert len(recovery_delayed) == 1
    assert recovery_entered.is_set()
    assert clean_errors == []
    assert len(clean_results) == 1
    outcome = clean_results[0]
    assert outcome.kind == "fatal_elected"
    assert outcome.primary is primary
    assert handoff.observe_outcome() is outcome
    assert publish_results == []
    assert len(publish_errors) == 1
    assert isinstance(publish_errors[0], FatalHardExit)
    assert report_results == []
    assert len(report_errors) == 1
    assert report_errors[0] is primary
    assert close_results == []
    assert len(close_errors) == 1
    assert close_errors[0] is primary
    with wrapper._fatal_condition:
        retained_owner = wrapper._fatal_publication_owner_reservation
        assert wrapper._fatal_publications == 1
        assert retained_owner.owner_thread is publisher
        assert retained_owner.primary is primary
        assert wrapper._fatal_publication_failure is primary
    with gate._condition:
        transition = gate._fatal_transition
        assert transition.primary is primary
        assert gate._fatal_publication_failure is primary
        assert gate._phase is _TerminalPhase.FATAL_PENDING
    assert failure_signals == [
        (transition, primary, False, False, False, False)
    ]
    assert join_observations == [
        (reporter_name, 1, primary),
        (close_name, 1, primary),
    ]
    assert hard_exit_observations == [
        (publisher_name, False, False, False, False),
    ]
    assert cancellation in wrapper._fatal_secondary_errors
    assert later in wrapper._fatal_secondary_errors
    assert raw_comm.abort_calls == 0
    assert wrapper._fatal_error is None
    assert runtime._terminal_error is None


@pytest.mark.parametrize(
    "failure_stage",
    [
        "after_pre_reserve",
        "after_owner_query",
        "after_transition_assignment",
        "during_context_retention",
        "before_handoff_outcome",
    ],
)
def test_hidden_fatal_election_recovers_every_reserved_owner_boundary(
    monkeypatch, failure_stage
):
    class ElectionCancelled(BaseException):
        pass

    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, _, _, raw_comm = _single_rank_task_18_2_runtime(
        monkeypatch
    )
    gate = runtime._terminal_gate
    primary = RuntimeError("recoverable hidden election primary")
    later = RuntimeError("recoverable hidden election later reporter")
    cancellation = ElectionCancelled(failure_stage)
    publisher_name = "task-18.2-hidden-election-{}".format(failure_stage)
    reporter_name = "task-18.2-hidden-election-reporter-{}".format(
        failure_stage
    )
    close_name = "task-18.2-hidden-election-close-{}".format(failure_stage)
    tokens = []
    pre_reserve_returns = []
    owner_query_returns = []
    injection_observations = []
    collective_failure_installs = []
    gate_failure_signals = []
    reporter_adoptions = []
    join_observations = []
    hard_exit_observations = []

    original_pre_reserve = wrapper._pre_reserve_fatal_publication

    def record_normal_pre_reserve(*args, **kwargs):
        result = original_pre_reserve(*args, **kwargs)
        if threading.current_thread().name == publisher_name:
            pre_reserve_returns.append(result)
            if (
                failure_stage == "after_pre_reserve"
                and not injection_observations
            ):
                with wrapper._fatal_condition:
                    election = (
                        wrapper._fatal_publication_owner_reservation.election
                    )
                cancel_at_operation(failure_stage, election)
        return result

    monkeypatch.setattr(
        wrapper, "_pre_reserve_fatal_publication", record_normal_pre_reserve
    )
    original_owner_query = wrapper._current_thread_reserved_fatal_primary

    def record_normal_owner_query():
        result = original_owner_query()
        if threading.current_thread().name == publisher_name:
            owner_query_returns.append(result)
            if (
                failure_stage == "after_owner_query"
                and not injection_observations
            ):
                with wrapper._fatal_condition:
                    election = (
                        wrapper._fatal_publication_owner_reservation.election
                    )
                cancel_at_operation(failure_stage, election)
        return result

    monkeypatch.setattr(
        wrapper,
        "_current_thread_reserved_fatal_primary",
        record_normal_owner_query,
    )

    def cancel_at_operation(stage, election):
        with wrapper._fatal_condition:
            retained_owner = wrapper._fatal_publication_owner_reservation
            owner_state = (
                wrapper._fatal_publications,
                retained_owner,
                wrapper._fatal_pending_primary,
            )
        with gate._condition:
            token_state = gate._tokens[tokens[0].sequence]
            gate_state = (
                gate._phase,
                gate._fatal_transition,
                token_state.status,
            )
        outcome = wrapper._fatal_monitor_handoff.observe_outcome()
        injection_observations.append(
            (
                stage,
                election,
                *owner_state,
                *gate_state,
                runtime._pending_fatal_collective,
                runtime._pending_fatal_context,
                outcome,
            )
        )
        raise cancellation

    def current_election():
        with wrapper._fatal_condition:
            retained_owner = wrapper._fatal_publication_owner_reservation
            assert retained_owner is not None
            return retained_owner.election

    original_begin_fatal_locked = gate._begin_fatal_locked

    def inject_partial_gate_transition(primary_arg, discovering_state, **kwargs):
        if (
            threading.current_thread().name == publisher_name
            and failure_stage == "after_transition_assignment"
            and not injection_observations
        ):
            election = current_election()
            transition = getattr(election, "prepared_transition", None)
            if transition is None:
                transition = _FatalTransition(
                    gate_id=gate._gate_id,
                    primary=primary_arg,
                    sequence=gate._sequence(),
                )
            gate._fatal_transition = transition
            election.transition = transition
            cancel_at_operation(failure_stage, election)
        return original_begin_fatal_locked(
            primary_arg, discovering_state, **kwargs
        )

    monkeypatch.setattr(
        gate, "_begin_fatal_locked", inject_partial_gate_transition
    )
    original_stage_quarantine = (
        runtime._stage_active_broadcast_quarantine_locked
    )

    def inject_context_retention(*args, **kwargs):
        result = original_stage_quarantine(*args, **kwargs)
        if (
            threading.current_thread().name == publisher_name
            and failure_stage == "during_context_retention"
            and not injection_observations
        ):
            cancel_at_operation(failure_stage, current_election())
        return result

    monkeypatch.setattr(
        runtime,
        "_stage_active_broadcast_quarantine_locked",
        inject_context_retention,
    )
    handoff = wrapper._fatal_monitor_handoff
    original_select_fatal = handoff.select_fatal

    def inject_before_handoff_outcome(
        primary_arg, *, before_select=None, **kwargs
    ):
        if (
            threading.current_thread().name == publisher_name
            and failure_stage == "before_handoff_outcome"
            and not injection_observations
        ):
            with handoff._condition:
                if handoff._outcome is None:
                    if before_select is not None:
                        before_select()
                    cancel_at_operation(failure_stage, current_election())
        return original_select_fatal(
            primary_arg, before_select=before_select, **kwargs
        )

    monkeypatch.setattr(
        handoff, "select_fatal", inject_before_handoff_outcome
    )
    original_collective_fail = wrapper._fail_reserved_fatal_publication

    def record_collective_failure(*args, **kwargs):
        with wrapper._fatal_condition:
            before = wrapper._fatal_publication_failure
        result = original_collective_fail(*args, **kwargs)
        with wrapper._fatal_condition:
            after = wrapper._fatal_publication_failure
        if before is None and after is primary:
            collective_failure_installs.append(
                (
                    threading.current_thread().name,
                    result,
                    wrapper._fatal_condition._is_owned(),
                    wrapper._fatal_lock._is_owned(),
                )
            )
        return result

    monkeypatch.setattr(
        wrapper,
        "_fail_reserved_fatal_publication",
        record_collective_failure,
    )
    original_gate_fail = gate._fail_fatal_publication

    def record_gate_failure(transition, failure):
        gate_failure_signals.append(
            (
                transition,
                failure,
                wrapper._fatal_condition._is_owned(),
                wrapper._fatal_lock._is_owned(),
                gate._condition._is_owned(),
            )
        )
        return original_gate_fail(transition, failure)

    monkeypatch.setattr(gate, "_fail_fatal_publication", record_gate_failure)
    original_begin_publication = wrapper._begin_fatal_publication

    def record_reporter_adoption(*args, **kwargs):
        result = original_begin_publication(*args, **kwargs)
        if threading.current_thread().name == reporter_name:
            reporter_adoptions.append(result)
        return result

    monkeypatch.setattr(
        wrapper, "_begin_fatal_publication", record_reporter_adoption
    )
    original_wait = wrapper._wait_for_joined_fatal_publication

    def record_join(**kwargs):
        with wrapper._fatal_condition:
            join_observations.append(
                (
                    threading.current_thread().name,
                    wrapper._fatal_publications,
                    wrapper._fatal_publication_failure,
                )
            )
        return original_wait(**kwargs)

    monkeypatch.setattr(
        wrapper, "_wait_for_joined_fatal_publication", record_join
    )

    def hard_exit():
        hard_exit_observations.append(
            (
                threading.current_thread().name,
                wrapper._fatal_condition._is_owned(),
                wrapper._fatal_lock._is_owned(),
                gate._condition._is_owned(),
            )
        )
        raise FatalHardExit(threading.current_thread().name)

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    def publish_from_admission():
        token = gate.admit_runtime("hidden_election_{}".format(failure_stage))
        tokens.append(token)
        return runtime._enter_communicator_fatal(
            primary, discovering_token=token
        )

    publisher, publish_results, publish_errors, publish_done = (
        _start_task_18_2_call(publish_from_admission, name=publisher_name)
    )
    _join_task_18_2_call(publisher, publish_done)

    assert publish_results == []
    assert len(publish_errors) == 1
    assert isinstance(publish_errors[0], FatalHardExit)
    assert pre_reserve_returns[0] == (primary, True)
    assert len(injection_observations) == 1
    (
        observed_stage,
        election,
        owner_count,
        retained_owner,
        pending_primary,
        injection_phase,
        injection_transition,
        injection_token_status,
        injection_collective,
        injection_context,
        injection_outcome,
    ) = injection_observations[0]
    assert observed_stage == failure_stage
    assert owner_count == 1
    assert retained_owner.owner_thread is publisher
    assert retained_owner.primary is primary
    assert retained_owner.election is election
    assert pending_primary is primary
    assert injection_outcome is None
    if failure_stage in {"after_pre_reserve", "after_owner_query"}:
        assert injection_phase is _TerminalPhase.HEALTHY
        assert injection_transition is None
        assert injection_token_status == "active"
        assert injection_collective is None
        assert injection_context is None
    elif failure_stage == "after_transition_assignment":
        assert injection_phase is _TerminalPhase.HEALTHY
        assert injection_transition.primary is primary
        assert injection_token_status == "active"
        assert injection_collective is None
        assert injection_context is None
    else:
        assert injection_phase is _TerminalPhase.FATAL_PENDING
        assert injection_transition.primary is primary
        assert injection_token_status == "converted"
        assert injection_collective is wrapper
        assert injection_context is not None
    if failure_stage != "after_pre_reserve":
        assert owner_query_returns[0] is primary

    with wrapper._fatal_condition:
        assert wrapper._fatal_publications == 1
        assert wrapper._fatal_publication_owner_reservation is retained_owner
        assert wrapper._fatal_publication_failure is primary
        assert wrapper._fatal_publication_gate_failure_claim.state == "signaled"
    with gate._condition:
        transition = gate._fatal_transition
        assert transition.primary is primary
        assert gate._phase is _TerminalPhase.FATAL_PENDING
        assert gate._fatal_publication_failure is primary
        assert gate._tokens[tokens[0].sequence].status == "converted"
        assert gate._has_active_tokens() is False
    outcome = wrapper._fatal_monitor_handoff.observe_outcome()
    assert outcome.kind == "fatal_elected"
    assert outcome.primary is primary
    assert runtime._pending_fatal_collective is wrapper
    assert runtime._pending_fatal_context is not None
    assert len(collective_failure_installs) == 1
    assert collective_failure_installs[0][0] == publisher_name
    assert collective_failure_installs[0][1] == (True, True)
    assert collective_failure_installs[0][2:] == (False, False)
    assert gate_failure_signals == [
        (transition, primary, False, False, False)
    ]
    with pytest.raises(RuntimeError, match="converted"):
        gate.release(tokens[0])

    reporter, report_results, report_errors, report_done = (
        _start_task_18_2_call(
            lambda: runtime._enter_communicator_fatal(later),
            name=reporter_name,
        )
    )
    _join_task_18_2_call(reporter, report_done)

    assert report_results == []
    assert len(report_errors) == 1
    assert report_errors[0] is primary
    assert len(reporter_adoptions) == 1
    assert reporter_adoptions[0][2:] == (False, True, False)

    closer, close_results, close_errors, close_done = _start_task_18_2_call(
        runtime.close, name=close_name
    )
    _join_task_18_2_call(closer, close_done)

    assert close_results == []
    assert len(close_errors) == 1
    assert close_errors[0] is primary
    assert join_observations == [
        (reporter_name, 1, primary),
        (close_name, 1, primary),
    ]
    assert hard_exit_observations == [
        (publisher_name, False, False, False),
    ]
    assert len(gate_failure_signals) == 1
    assert sum(
        retained is cancellation for retained in wrapper._fatal_secondary_errors
    ) == 1
    assert raw_comm.abort_calls == 0
    assert wrapper._fatal_error is None
    assert runtime._terminal_error is None


def test_hidden_fatal_election_has_no_dynamic_checkpoint_dispatch():
    import inspect
    import renormalizer.backend.distributed_runtime as runtime_module

    source = inspect.getsource(runtime_module._RecoverableFatalElection)
    recovery_source = inspect.getsource(
        runtime_module._RecoverableFatalElection.recover_locked
    )

    assert "_fatal_election_test_checkpoint" not in source
    assert "while True" not in recovery_source


def test_hidden_fatal_election_retains_a_bounded_recovery_error_set(monkeypatch):
    import renormalizer.backend.distributed_runtime as runtime_module

    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    primary = RuntimeError("bounded recovery primary")
    election = runtime_module._RecoverableFatalElection(
        runtime,
        wrapper,
        primary,
        None,
        None,
        None,
        None,
    )
    errors = tuple(RuntimeError("recovery {}".format(index)) for index in range(6))

    for error in errors:
        election._remember_recovery_error(error)

    assert election.recovery_errors == errors[:4]


def _assert_hidden_fatal_recovery_operation_is_bounded(monkeypatch, *, persistent):
    class ElectionCancelled(BaseException):
        pass

    class RecoveryCancelled(BaseException):
        pass

    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, _, _, raw_comm = _single_rank_task_18_2_runtime(
        monkeypatch
    )
    gate = runtime._terminal_gate
    primary = RuntimeError("finite hidden recovery primary")
    later = RuntimeError("finite hidden recovery later reporter")
    cancellation = ElectionCancelled("cancel after retained owner creation")
    recovery_failure = RecoveryCancelled("cancel recovery context operation")
    publisher_name = "task-18.2-finite-recovery-owner"
    reporter_name = "task-18.2-finite-recovery-reporter"
    close_name = "task-18.2-finite-recovery-close"
    recovery_progress = threading.Condition()
    recovery_calls = []
    release_persistent_failure = threading.Event()
    owner_at_hard_exit = threading.Event()
    release_owner = threading.Event()
    hard_exit_observations = []
    tokens = []

    original_pre_reserve = wrapper._pre_reserve_fatal_publication

    def cancel_after_owner_creation(*args, **kwargs):
        result = original_pre_reserve(*args, **kwargs)
        if threading.current_thread().name == publisher_name:
            raise cancellation
        return result

    monkeypatch.setattr(
        wrapper, "_pre_reserve_fatal_publication", cancel_after_owner_creation
    )
    original_stage_quarantine = (
        runtime._stage_active_broadcast_quarantine_locked
    )

    def fail_recovery_context_operation(*args, **kwargs):
        if threading.current_thread().name != publisher_name:
            return original_stage_quarantine(*args, **kwargs)
        with recovery_progress:
            recovery_calls.append(threading.current_thread())
            call_count = len(recovery_calls)
            recovery_progress.notify_all()
        if persistent and not release_persistent_failure.is_set():
            raise recovery_failure
        if not persistent and call_count == 1:
            raise recovery_failure
        return original_stage_quarantine(*args, **kwargs)

    monkeypatch.setattr(
        runtime,
        "_stage_active_broadcast_quarantine_locked",
        fail_recovery_context_operation,
    )

    def hard_exit():
        hard_exit_observations.append(
            (
                threading.current_thread().name,
                wrapper._fatal_condition._is_owned(),
                wrapper._fatal_lock._is_owned(),
                gate._condition._is_owned(),
            )
        )
        if threading.current_thread().name == publisher_name:
            owner_at_hard_exit.set()
            assert release_owner.wait(_TASK_18_2_TIMEOUT_S * 2)
        raise FatalHardExit(threading.current_thread().name)

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    def publish_from_admission():
        token = gate.admit_runtime("finite_hidden_recovery")
        tokens.append(token)
        return runtime._enter_communicator_fatal(
            primary, discovering_token=token
        )

    publisher, publish_results, publish_errors, publish_done = (
        _start_task_18_2_call(publish_from_admission, name=publisher_name)
    )
    reporter = closer = None
    try:
        with recovery_progress:
            assert recovery_progress.wait_for(
                lambda: owner_at_hard_exit.is_set()
                or len(recovery_calls) >= 3,
                timeout=_TASK_18_2_TIMEOUT_S,
            )
        if not owner_at_hard_exit.is_set():
            release_persistent_failure.set()
        assert owner_at_hard_exit.wait(_TASK_18_2_TIMEOUT_S)

        reporter, report_results, report_errors, report_done = (
            _start_task_18_2_call(
                lambda: runtime._enter_communicator_fatal(later),
                name=reporter_name,
            )
        )
        closer, close_results, close_errors, close_done = (
            _start_task_18_2_call(runtime.close, name=close_name)
        )
        _join_task_18_2_call(reporter, report_done)
        _join_task_18_2_call(closer, close_done)
    finally:
        release_persistent_failure.set()
        release_owner.set()
        if publisher.is_alive():
            _join_task_18_2_call(publisher, publish_done)

    assert publish_results == []
    assert len(publish_errors) == 1
    assert isinstance(publish_errors[0], FatalHardExit)
    assert report_results == []
    assert len(report_errors) == 1
    assert report_errors[0] is primary
    assert close_results == []
    assert len(close_errors) == 1
    assert close_errors[0] is primary
    assert len(recovery_calls) == 1
    with wrapper._fatal_condition:
        retained_owner = wrapper._fatal_publication_owner_reservation
        assert retained_owner.owner_thread is publisher
        assert retained_owner.primary is primary
        assert retained_owner.election.recovery_errors == (recovery_failure,)
        assert wrapper._fatal_publications == 1
        assert wrapper._fatal_publication_failure is primary
        assert wrapper._fatal_publication_gate_failure_claim.state == "signaled"
    with gate._condition:
        transition = gate._fatal_transition
        assert transition.primary is primary
        assert gate._fatal_publication_failure is primary
        assert gate._tokens[tokens[0].sequence].status == "converted"
        assert gate._has_active_tokens() is False
    outcome = wrapper._fatal_monitor_handoff.observe_outcome()
    assert outcome.kind == "fatal_elected"
    assert outcome.primary is primary
    assert runtime._pending_fatal_collective is wrapper
    assert runtime._pending_fatal_context is not None
    assert all(observation[1:] == (False, False, False) for observation in hard_exit_observations)
    assert [observation[0] for observation in hard_exit_observations] == [
        publisher_name
    ]
    assert sum(
        retained is recovery_failure
        for retained in wrapper._fatal_secondary_errors
    ) == 1
    assert raw_comm.abort_calls == 0


def test_hidden_fatal_recovery_operation_failure_is_repaired_once(monkeypatch):
    _assert_hidden_fatal_recovery_operation_is_bounded(
        monkeypatch, persistent=False
    )


def test_persistent_hidden_fatal_recovery_operation_uses_bounded_fail_stop(
    monkeypatch,
):
    _assert_hidden_fatal_recovery_operation_is_bounded(
        monkeypatch, persistent=True
    )


def test_gate_failure_callback_cancellation_falls_back_and_wakes_waiters(
    monkeypatch,
):
    class CallbackCancelled(BaseException):
        pass

    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, _, _, raw_comm = _single_rank_task_18_2_runtime(
        monkeypatch
    )
    gate = runtime._terminal_gate
    hook = runtime._communicator_fatal_hook
    primary = RuntimeError("retryable gate failure primary")
    callback_cancellation = CallbackCancelled("cancel first gate callback")
    owner_name = "task-18.2-retryable-gate-failure-owner"
    callback_observations = []
    reentrant_results = []
    gate_primitive_calls = []
    hard_exit_observations = []
    owner_state = []

    real_gate_primitive = gate._fail_fatal_publication

    def record_gate_primitive(transition, failure):
        gate_primitive_calls.append(
            (
                transition,
                failure,
                wrapper._fatal_condition._is_owned(),
                wrapper._fatal_lock._is_owned(),
                gate._condition._is_owned(),
            )
        )
        return real_gate_primitive(transition, failure)

    monkeypatch.setattr(gate, "_fail_fatal_publication", record_gate_primitive)
    real_hook_fail = hook.fail

    def fail_once_then_signal(transition, failure):
        callback_observations.append(
            (
                wrapper._fatal_condition._is_owned(),
                wrapper._fatal_lock._is_owned(),
                gate._condition._is_owned(),
            )
        )
        if len(callback_observations) == 1:
            reentrant_results.append(
                wrapper._fail_reserved_fatal_publication(
                    callback_cancellation,
                    transition_handler=hook,
                    transition=transition,
                )
            )
            raise callback_cancellation
        return real_hook_fail(transition, failure)

    monkeypatch.setattr(hook, "fail", fail_once_then_signal)

    def hard_exit():
        hard_exit_observations.append(
            (
                wrapper._fatal_condition._is_owned(),
                wrapper._fatal_lock._is_owned(),
                gate._condition._is_owned(),
            )
        )
        raise FatalHardExit(owner_name)

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)
    waiter, wait_results, wait_errors, wait_done = _start_task_18_2_call(
        lambda: gate.wait_for_published(0.5),
        name="task-18.2-retryable-gate-failure-waiter",
    )

    def fail_reserved_owner():
        token = gate.admit_runtime("retryable_gate_failure")
        reserved, started = wrapper._pre_reserve_fatal_publication(primary)
        assert reserved is primary
        assert started is True
        transition = gate.begin_fatal(primary, token)
        outcome = wrapper._reserve_runtime_fatal_outcome(primary)
        assert outcome.primary is primary
        with wrapper._fatal_condition:
            owner_state.append(wrapper._fatal_publication_owner_reservation)
        owned, first_failure = wrapper._fail_reserved_fatal_publication(
            callback_cancellation,
            transition_handler=hook,
            transition=transition,
        )
        assert (owned, first_failure) == (True, True)
        wrapper._fatal_hard_exit()

    owner, owner_results, owner_errors, owner_done = _start_task_18_2_call(
        fail_reserved_owner, name=owner_name
    )
    _join_task_18_2_call(owner, owner_done)
    _join_task_18_2_call(waiter, wait_done)

    assert owner_results == []
    assert len(owner_errors) == 1
    assert isinstance(owner_errors[0], FatalHardExit)
    assert wait_results == []
    assert wait_errors == [primary]
    assert owner_state[0].owner_thread is owner
    assert owner_state[0].primary is primary
    assert callback_observations == [(False, False, False)]
    assert reentrant_results == [(True, False)]
    assert gate_primitive_calls == []
    with wrapper._fatal_condition:
        assert wrapper._fatal_publications == 1
        assert wrapper._fatal_publication_failure is primary
        assert wrapper._fatal_publication_gate_failure_claim.state == "signaled"
    with gate._condition:
        assert gate._fatal_publication_failure is primary
        assert gate._tokens[next(iter(gate._tokens))].status == "converted"
    assert hard_exit_observations == [(False, False, False)]
    assert sum(
        retained is callback_cancellation
        for retained in wrapper._fatal_secondary_errors
    ) == 1
    assert raw_comm.abort_calls == 0


def test_gate_failure_signal_uses_one_immutable_finite_claim(monkeypatch):
    import ast
    from dataclasses import FrozenInstanceError
    import inspect
    import textwrap

    _, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    claim = wrapper._fatal_publication_gate_failure_claim
    signal_source = inspect.getsource(
        type(wrapper)._signal_reserved_fatal_gate_failure
    )
    signal_tree = ast.parse(textwrap.dedent(signal_source))

    assert not hasattr(wrapper, "_fatal_publication_gate_failure_state")
    assert not hasattr(wrapper, "_fatal_publication_gate_failure_owner")
    assert not hasattr(wrapper, "_fatal_publication_gate_failure_deadline")
    assert claim.state == "idle"
    assert claim.owner is None
    assert claim.deadline is None
    with pytest.raises(FrozenInstanceError):
        claim.state = "inflight"
    assert not any(isinstance(node, ast.While) for node in ast.walk(signal_tree))


def test_live_gate_failure_claim_is_not_stolen_after_deadline(monkeypatch):
    from renormalizer.backend._distributed import collectives as collectives_module
    from renormalizer.backend._distributed import terminal as terminal_module

    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, _, _, raw_comm = _single_rank_task_18_2_runtime(
        monkeypatch
    )
    gate = runtime._terminal_gate
    hook = runtime._communicator_fatal_hook
    primary = RuntimeError("live gate failure claim primary")
    trigger = RuntimeError("live gate failure claim trigger")
    owner_name = "task-18.2-live-gate-claim-owner"
    competitor_name = "task-18.2-live-gate-claim-competitor"
    close_name = "task-18.2-live-gate-claim-close"
    callback_entered = threading.Event()
    release_callback = threading.Event()
    competitor_entered = threading.Event()
    close_entered = threading.Event()
    callback_calls = []
    primitive_calls = []
    competing_results = []
    hard_exit_observations = []
    hard_exit_claimed = threading.Event()

    monkeypatch.setattr(collectives_module, "_FATAL_TIMEOUT_S", 0.05)
    monkeypatch.setattr(terminal_module, "_TERMINAL_TIMEOUT_S", 0.05)
    real_gate_primitive = gate._fail_fatal_publication

    def record_gate_primitive(transition, failure):
        primitive_calls.append(
            (
                threading.current_thread().name,
                transition,
                failure,
                wrapper._fatal_condition._is_owned(),
                gate._condition._is_owned(),
            )
        )
        return real_gate_primitive(transition, failure)

    monkeypatch.setattr(gate, "_fail_fatal_publication", record_gate_primitive)
    real_hook_fail = hook.fail

    def hold_live_owner_callback(transition, failure):
        callback_calls.append(threading.current_thread().name)
        if threading.current_thread().name == owner_name:
            callback_entered.set()
            assert release_callback.wait(_TASK_18_2_TIMEOUT_S * 2)
        return real_hook_fail(transition, failure)

    monkeypatch.setattr(hook, "fail", hold_live_owner_callback)
    original_commit_runtime_close = gate.commit_runtime_close

    def record_close_entry(*args, **kwargs):
        if threading.current_thread().name == close_name:
            close_entered.set()
        return original_commit_runtime_close(*args, **kwargs)

    monkeypatch.setattr(gate, "commit_runtime_close", record_close_entry)

    def hard_exit():
        hard_exit_observations.append(
            (
                threading.current_thread().name,
                wrapper._fatal_condition._is_owned(),
                wrapper._fatal_lock._is_owned(),
                gate._condition._is_owned(),
            )
        )
        hard_exit_claimed.set()
        raise FatalHardExit(threading.current_thread().name)

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    def reserve_and_fail_owner():
        token = gate.admit_runtime("live_gate_failure_claim")
        reserved, started = wrapper._pre_reserve_fatal_publication(primary)
        assert (reserved, started) == (primary, True)
        transition = gate.begin_fatal(primary, token)
        outcome = wrapper._reserve_runtime_fatal_outcome(primary)
        assert outcome.primary is primary
        with runtime._terminal_state_lock:
            runtime._pending_fatal_collective = wrapper
            runtime._pending_fatal_context = (None, None, None)
        assert wrapper._fail_reserved_fatal_publication(
            trigger,
            transition_handler=hook,
            transition=transition,
        ) == (True, True)
        wrapper._hard_exit_once(primary)

    owner, owner_results, owner_errors, owner_done = _start_task_18_2_call(
        reserve_and_fail_owner, name=owner_name
    )
    competitor = closer = None
    try:
        assert callback_entered.wait(_TASK_18_2_TIMEOUT_S)
        transition = gate._fatal_transition

        def compete_for_signal():
            competitor_entered.set()
            competing_results.append(
                wrapper._fail_reserved_fatal_publication(
                    trigger,
                    transition_handler=hook,
                    transition=transition,
                )
            )

        competitor, competitor_results, competitor_errors, competitor_done = (
            _start_task_18_2_call(
                compete_for_signal,
                name=competitor_name,
            )
        )
        closer, close_results, close_errors, close_done = (
            _start_task_18_2_call(runtime.close, name=close_name)
        )
        assert competitor_entered.wait(_TASK_18_2_TIMEOUT_S)
        assert close_entered.wait(_TASK_18_2_TIMEOUT_S)
        _join_task_18_2_call(competitor, competitor_done)
        assert hard_exit_claimed.wait(_TASK_18_2_TIMEOUT_S)
        _join_task_18_2_call(closer, close_done)
        before_release = (
            tuple(callback_calls),
            len(primitive_calls),
            close_done.is_set(),
        )
    finally:
        release_callback.set()
        if competitor is not None and competitor.is_alive():
            _join_task_18_2_call(competitor, competitor_done)
        if owner.is_alive():
            _join_task_18_2_call(owner, owner_done)
        if closer is not None and closer.is_alive():
            _join_task_18_2_call(closer, close_done)

    assert before_release == ((owner_name,), 0, True)
    assert owner_results == []
    assert owner_errors == [primary]
    assert competitor_results == [None]
    assert competitor_errors == []
    assert competing_results == [(False, False)]
    assert close_results == []
    assert len(close_errors) == 1
    assert isinstance(close_errors[0], FatalHardExit)
    assert callback_calls == [owner_name]
    assert len(primitive_calls) == 1
    assert primitive_calls[0][0] == owner_name
    assert primitive_calls[0][1] is gate._fatal_transition
    assert primitive_calls[0][2] is primary
    assert primitive_calls[0][3:] == (False, False)
    with wrapper._fatal_condition:
        claim = wrapper._fatal_publication_gate_failure_claim
        assert claim.state == "signaled"
        assert claim.owner is None
        assert claim.deadline is None
        assert wrapper._fatal_publication_failure is primary
    with gate._condition:
        assert gate._fatal_publication_failure is primary
    assert hard_exit_observations == [
        (close_name, False, False, False),
    ]
    assert raw_comm.abort_calls == 0


@pytest.mark.parametrize(
    "boundary",
    [
        "claim_publication",
        "before_callback",
        "after_callback",
        "after_verification",
        "cleanup_before",
        "cleanup_after_state",
        "cleanup_after_owner",
    ],
)
def test_gate_failure_claim_repairs_every_one_shot_boundary(
    monkeypatch, boundary
):
    import inspect

    class BoundaryCancelled(BaseException):
        pass

    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, _, _, raw_comm = _single_rank_task_18_2_runtime(
        monkeypatch
    )
    gate = runtime._terminal_gate
    hook = runtime._communicator_fatal_hook
    primary = RuntimeError("guarded gate failure claim primary")
    cancellation = BoundaryCancelled(boundary)
    owner_name = "task-18.2-gate-claim-owner-{}".format(boundary)
    close_name = "task-18.2-gate-claim-close-{}".format(boundary)
    injection_observations = []
    primitive_calls = []
    competing_results = []
    boundary_reached = threading.Event()
    release_boundary = threading.Event()
    competitor_entered = threading.Event()
    close_entered = threading.Event()
    owner_at_hard_exit = threading.Event()
    release_owner = threading.Event()
    hard_exit_observations = []
    transition_holder = []

    signal_source, signal_start = inspect.getsourcelines(
        type(wrapper)._signal_reserved_fatal_gate_failure
    )
    claim_publication_lines = {
        signal_start + offset
        for offset, line in enumerate(signal_source)
        if "claim_published = True" in line
    }
    assert claim_publication_lines
    signal_code = type(wrapper)._signal_reserved_fatal_gate_failure.__code__

    def cancel_at_boundary():
        injection_observations.append(boundary)
        boundary_reached.set()
        assert release_boundary.wait(_TASK_18_2_TIMEOUT_S * 2)
        raise cancellation

    real_gate_primitive = gate._fail_fatal_publication

    def record_gate_primitive(transition, failure):
        primitive_calls.append(
            (
                transition,
                failure,
                wrapper._fatal_condition._is_owned(),
                gate._condition._is_owned(),
            )
        )
        return real_gate_primitive(transition, failure)

    monkeypatch.setattr(gate, "_fail_fatal_publication", record_gate_primitive)
    original_invoke = getattr(
        wrapper, "_invoke_fatal_gate_failure_callback", None
    )

    def inject_callback(failure_handler, transition, failure):
        if boundary == "before_callback" and not injection_observations:
            cancel_at_boundary()
        if callable(original_invoke):
            result = original_invoke(failure_handler, transition, failure)
        else:
            result = failure_handler(transition, failure)
        if boundary == "after_callback" and not injection_observations:
            cancel_at_boundary()
        return result

    monkeypatch.setattr(
        wrapper,
        "_invoke_fatal_gate_failure_callback",
        inject_callback,
        raising=False,
    )
    original_verify = getattr(
        wrapper, "_verify_fatal_gate_failure_callback", None
    )

    def inject_verification(failure_confirmed, transition, failure):
        if callable(original_verify):
            result = original_verify(
                failure_confirmed, transition, failure
            )
        else:
            result = bool(failure_confirmed(transition, failure))
        if boundary == "after_verification" and not injection_observations:
            cancel_at_boundary()
        return result

    monkeypatch.setattr(
        wrapper,
        "_verify_fatal_gate_failure_callback",
        inject_verification,
        raising=False,
    )
    original_settle = getattr(
        wrapper, "_settle_fatal_gate_failure_claim", None
    )

    def inject_cleanup(claim, confirmed):
        if boundary == "cleanup_before" and not injection_observations:
            cancel_at_boundary()
        if boundary in {
            "cleanup_after_state",
            "cleanup_after_owner",
        } and not injection_observations:
            with wrapper._fatal_condition:
                wrapper._fatal_publication_gate_failure_claim = SimpleNamespace(
                    state="signaled" if confirmed else "idle",
                    owner=(
                        claim.owner
                        if boundary == "cleanup_after_state"
                        else None
                    ),
                    deadline=claim.deadline,
                )
            cancel_at_boundary()
        if callable(original_settle):
            result = original_settle(claim, confirmed)
        else:
            result = confirmed
        return result

    monkeypatch.setattr(
        wrapper,
        "_settle_fatal_gate_failure_claim",
        inject_cleanup,
        raising=False,
    )
    original_signal = wrapper._signal_reserved_fatal_gate_failure

    def record_competing_signaler(*args, **kwargs):
        if threading.current_thread().name.startswith(
            "task-18.2-gate-claim-competitor-"
        ):
            competitor_entered.set()
        return original_signal(*args, **kwargs)

    monkeypatch.setattr(
        wrapper,
        "_signal_reserved_fatal_gate_failure",
        record_competing_signaler,
    )
    original_commit_runtime_close = gate.commit_runtime_close

    def record_close_entry(*args, **kwargs):
        if threading.current_thread().name == close_name:
            close_entered.set()
        return original_commit_runtime_close(*args, **kwargs)

    monkeypatch.setattr(gate, "commit_runtime_close", record_close_entry)

    def hard_exit():
        hard_exit_observations.append(
            (
                threading.current_thread().name,
                wrapper._fatal_condition._is_owned(),
                wrapper._fatal_lock._is_owned(),
                gate._condition._is_owned(),
            )
        )
        if threading.current_thread().name == owner_name:
            owner_at_hard_exit.set()
            assert release_owner.wait(_TASK_18_2_TIMEOUT_S * 2)
        raise FatalHardExit(threading.current_thread().name)

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    def fail_owner():
        token = gate.admit_runtime("guarded_gate_failure_claim")
        reserved, started = wrapper._pre_reserve_fatal_publication(primary)
        assert reserved is primary
        assert started is True
        transition = gate.begin_fatal(primary, token)
        transition_holder.append(transition)
        outcome = wrapper._reserve_runtime_fatal_outcome(primary)
        assert outcome.primary is primary
        with runtime._terminal_state_lock:
            runtime._pending_fatal_collective = wrapper
            runtime._pending_fatal_context = (None, None, None)

        def cancel_claim_publication(frame, event, _arg):
            installed_claim = wrapper._fatal_publication_gate_failure_claim
            if (
                boundary == "claim_publication"
                and event == "line"
                and frame.f_code is signal_code
                and frame.f_lineno in claim_publication_lines
                and installed_claim is frame.f_locals.get("claim")
                and not injection_observations
            ):
                cancel_at_boundary()
            return cancel_claim_publication

        if boundary == "claim_publication":
            sys.settrace(cancel_claim_publication)
        try:
            result = wrapper._fail_reserved_fatal_publication(
                cancellation,
                transition_handler=hook,
                transition=transition,
            )
        finally:
            sys.settrace(None)
        assert result == (True, True)
        wrapper._fatal_hard_exit()

    owner, owner_results, owner_errors, owner_done = _start_task_18_2_call(
        fail_owner, name=owner_name
    )
    closer = competitor = None
    try:
        assert boundary_reached.wait(_TASK_18_2_TIMEOUT_S)
        transition = transition_holder[0]

        def compete_for_signal():
            competitor_entered.set()
            competing_results.append(
                wrapper._fail_reserved_fatal_publication(
                    cancellation,
                    transition_handler=hook,
                    transition=transition,
                )
            )

        competitor, competitor_results, competitor_errors, competitor_done = (
            _start_task_18_2_call(
                compete_for_signal,
                name="task-18.2-gate-claim-competitor-{}".format(boundary),
            )
        )
        closer, close_results, close_errors, close_done = (
            _start_task_18_2_call(runtime.close, name=close_name)
        )
        assert competitor_entered.wait(_TASK_18_2_TIMEOUT_S)
        assert close_entered.wait(_TASK_18_2_TIMEOUT_S)
        release_boundary.set()
        assert owner_at_hard_exit.wait(_TASK_18_2_TIMEOUT_S)
        _join_task_18_2_call(competitor, competitor_done)
        _join_task_18_2_call(closer, close_done)
    finally:
        release_boundary.set()
        release_owner.set()
        if competitor is not None and competitor.is_alive():
            _join_task_18_2_call(competitor, competitor_done)
        if closer is not None and closer.is_alive():
            _join_task_18_2_call(closer, close_done)
        if owner.is_alive():
            _join_task_18_2_call(owner, owner_done)

    assert owner_results == []
    assert len(owner_errors) == 1
    assert isinstance(owner_errors[0], FatalHardExit)
    assert competitor_results == [None]
    assert competitor_errors == []
    assert competing_results == [(False, False)]
    assert close_results == []
    assert len(close_errors) == 1
    assert isinstance(close_errors[0], FatalHardExit)
    assert injection_observations == [boundary]
    assert len(primitive_calls) == 1
    assert primitive_calls[0][0] is transition_holder[0]
    assert primitive_calls[0][1] is primary
    assert primitive_calls[0][2:] == (False, False)
    with wrapper._fatal_condition:
        claim = wrapper._fatal_publication_gate_failure_claim
        assert claim.state == "signaled"
        assert claim.owner is None
        assert claim.deadline is None
        assert wrapper._fatal_publication_failure is primary
    with gate._condition:
        assert gate._fatal_publication_failure is primary
    assert all(observation[1:] == (False, False, False) for observation in hard_exit_observations)
    assert {observation[0] for observation in hard_exit_observations} == {
        owner_name,
        close_name,
    }
    assert sum(
        retained is cancellation
        for retained in wrapper._fatal_secondary_errors
    ) == 1
    assert raw_comm.abort_calls == 0


def test_departed_successful_gate_claim_verify_escape_does_not_repeat_callback(
    monkeypatch,
):
    class VerificationInterrupted(BaseException):
        pass

    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    hook = runtime._communicator_fatal_hook
    primary = RuntimeError("departed successful gate claim primary")
    interruption = VerificationInterrupted("interrupt initial verification")
    callback_calls = []
    primitive_calls = []
    verification_calls = []
    hard_exit_observations = []
    departed_done = threading.Event()

    def depart():
        departed_done.set()

    departed = threading.Thread(
        target=depart,
        name="task-18.2-departed-successful-claim-owner",
        daemon=True,
    )
    departed.start()
    _join_task_18_2_call(departed, departed_done)

    real_gate_primitive = gate._fail_fatal_publication

    def record_gate_primitive(transition, failure):
        primitive_calls.append(
            (
                transition,
                failure,
                wrapper._fatal_condition._is_owned(),
                gate._condition._is_owned(),
            )
        )
        return real_gate_primitive(transition, failure)

    monkeypatch.setattr(gate, "_fail_fatal_publication", record_gate_primitive)
    real_hook_fail = hook.fail

    def record_hook_fail(transition, failure):
        callback_calls.append(
            (
                transition,
                failure,
                wrapper._fatal_condition._is_owned(),
                gate._condition._is_owned(),
            )
        )
        return real_hook_fail(transition, failure)

    monkeypatch.setattr(hook, "fail", record_hook_fail)
    token = gate.admit_runtime("departed_successful_gate_claim")
    reserved, started = wrapper._pre_reserve_fatal_publication(primary)
    assert (reserved, started) == (primary, True)
    transition = gate.begin_fatal(primary, token)
    outcome = wrapper._reserve_runtime_fatal_outcome(primary)
    assert outcome.primary is primary
    with runtime._terminal_state_lock:
        runtime._pending_fatal_collective = wrapper
        runtime._pending_fatal_context = (None, None, None)

    waiter, wait_results, wait_errors, wait_done = _start_task_18_2_call(
        lambda: gate.wait_for_published(_TASK_18_2_TIMEOUT_S),
        name="task-18.2-departed-successful-claim-waiter",
    )
    hook.fail(transition, primary)
    with wrapper._fatal_condition:
        claim_type = type(wrapper._fatal_publication_gate_failure_claim)
        wrapper._fatal_publication_failure = primary
        wrapper._fatal_publication_gate_failure_claim = claim_type(
            "inflight", departed, time.monotonic() + 5.0
        )

    real_verify = wrapper._verify_fatal_gate_failure_callback

    def interrupt_first_verification(failure_confirmed, selected, failure):
        verification_calls.append(
            (
                selected,
                failure,
                wrapper._fatal_condition._is_owned(),
                gate._condition._is_owned(),
            )
        )
        if len(verification_calls) == 1:
            raise interruption
        return real_verify(failure_confirmed, selected, failure)

    monkeypatch.setattr(
        wrapper,
        "_verify_fatal_gate_failure_callback",
        interrupt_first_verification,
    )

    assert wrapper._fail_reserved_fatal_publication(
        primary,
        transition_handler=hook,
        transition=transition,
    ) == (True, False)
    _join_task_18_2_call(waiter, wait_done)

    def hard_exit():
        hard_exit_observations.append(
            (
                wrapper._fatal_condition._is_owned(),
                wrapper._fatal_lock._is_owned(),
                gate._condition._is_owned(),
            )
        )
        raise FatalHardExit()

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)
    closer, close_results, close_errors, close_done = _start_task_18_2_call(
        runtime.close,
        name="task-18.2-departed-successful-claim-close",
    )
    _join_task_18_2_call(closer, close_done)

    assert wait_results == []
    assert wait_errors == [primary]
    assert close_results == []
    assert len(close_errors) == 1
    assert isinstance(close_errors[0], FatalHardExit)
    assert len(callback_calls) == 1
    assert callback_calls[0][0] is transition
    assert callback_calls[0][1] is primary
    assert callback_calls[0][2:] == (False, False)
    assert len(primitive_calls) == 1
    assert primitive_calls[0][0] is transition
    assert primitive_calls[0][1] is primary
    assert primitive_calls[0][2:] == (False, False)
    assert len(verification_calls) == 2
    assert all(call[0] is transition for call in verification_calls)
    assert all(call[1] is primary for call in verification_calls)
    assert all(call[2:] == (False, False) for call in verification_calls)
    with wrapper._fatal_condition:
        claim = wrapper._fatal_publication_gate_failure_claim
        assert claim.state == "signaled"
        assert claim.owner is None
        assert claim.deadline is None
    with gate._condition:
        assert gate._fatal_publication_failure is primary
    assert hard_exit_observations == [(False, False, False)]
    assert sum(
        retained is interruption
        for retained in wrapper._fatal_secondary_errors
    ) == 1


def test_absent_gate_claim_verify_escape_requires_fresh_verified_callback(
    monkeypatch,
):
    class VerificationInterrupted(BaseException):
        pass

    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    hook = runtime._communicator_fatal_hook
    primary = RuntimeError("departed absent gate claim primary")
    interruption = VerificationInterrupted("interrupt absent verification")
    trace = []
    callback_calls = []
    primitive_calls = []
    hard_exit_observations = []
    departed_done = threading.Event()

    def depart():
        departed_done.set()

    departed = threading.Thread(
        target=depart,
        name="task-18.2-departed-absent-claim-owner",
        daemon=True,
    )
    departed.start()
    _join_task_18_2_call(departed, departed_done)

    real_gate_primitive = gate._fail_fatal_publication

    def record_gate_primitive(transition, failure):
        trace.append("primitive")
        primitive_calls.append(
            (
                transition,
                failure,
                wrapper._fatal_condition._is_owned(),
                gate._condition._is_owned(),
            )
        )
        return real_gate_primitive(transition, failure)

    monkeypatch.setattr(gate, "_fail_fatal_publication", record_gate_primitive)
    real_hook_fail = hook.fail

    def record_hook_fail(transition, failure):
        trace.append("callback")
        callback_calls.append(
            (
                transition,
                failure,
                wrapper._fatal_condition._is_owned(),
                gate._condition._is_owned(),
            )
        )
        return real_hook_fail(transition, failure)

    monkeypatch.setattr(hook, "fail", record_hook_fail)
    token = gate.admit_runtime("departed_absent_gate_claim")
    reserved, started = wrapper._pre_reserve_fatal_publication(primary)
    assert (reserved, started) == (primary, True)
    transition = gate.begin_fatal(primary, token)
    outcome = wrapper._reserve_runtime_fatal_outcome(primary)
    assert outcome.primary is primary
    with runtime._terminal_state_lock:
        runtime._pending_fatal_collective = wrapper
        runtime._pending_fatal_context = (None, None, None)
    with wrapper._fatal_condition:
        claim_type = type(wrapper._fatal_publication_gate_failure_claim)
        wrapper._fatal_publication_failure = primary
        wrapper._fatal_publication_gate_failure_claim = claim_type(
            "inflight", departed, time.monotonic() + 5.0
        )

    waiter, wait_results, wait_errors, wait_done = _start_task_18_2_call(
        lambda: gate.wait_for_published(_TASK_18_2_TIMEOUT_S),
        name="task-18.2-departed-absent-claim-waiter",
    )
    real_verify = wrapper._verify_fatal_gate_failure_callback
    verification_count = [0]

    def interrupt_first_verification(failure_confirmed, selected, failure):
        verification_count[0] += 1
        if verification_count[0] == 1:
            trace.append("verify_interrupted")
            raise interruption
        result = real_verify(failure_confirmed, selected, failure)
        trace.append("verify_present" if result else "verify_absent")
        return result

    monkeypatch.setattr(
        wrapper,
        "_verify_fatal_gate_failure_callback",
        interrupt_first_verification,
    )
    real_settle = wrapper._settle_fatal_gate_failure_claim

    def record_settle(claim, confirmed):
        trace.append("settle_present" if confirmed else "settle_absent")
        return real_settle(claim, confirmed)

    monkeypatch.setattr(
        wrapper,
        "_settle_fatal_gate_failure_claim",
        record_settle,
    )

    assert wrapper._fail_reserved_fatal_publication(
        primary,
        transition_handler=hook,
        transition=transition,
    ) == (True, False)
    _join_task_18_2_call(waiter, wait_done)

    def hard_exit():
        hard_exit_observations.append(
            (
                wrapper._fatal_condition._is_owned(),
                wrapper._fatal_lock._is_owned(),
                gate._condition._is_owned(),
            )
        )
        raise FatalHardExit()

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)
    closer, close_results, close_errors, close_done = _start_task_18_2_call(
        runtime.close,
        name="task-18.2-departed-absent-claim-close",
    )
    _join_task_18_2_call(closer, close_done)

    assert wait_results == []
    assert wait_errors == [primary]
    assert close_results == []
    assert len(close_errors) == 1
    assert isinstance(close_errors[0], FatalHardExit)
    assert callback_calls == [
        (transition, primary, False, False),
    ]
    assert primitive_calls == [
        (transition, primary, False, False),
    ]
    assert trace == [
        "verify_interrupted",
        "verify_absent",
        "settle_absent",
        "verify_absent",
        "callback",
        "primitive",
        "verify_present",
        "settle_present",
    ]
    with wrapper._fatal_condition:
        claim = wrapper._fatal_publication_gate_failure_claim
        assert claim.state == "signaled"
        assert claim.owner is None
        assert claim.deadline is None
    with gate._condition:
        assert gate._fatal_publication_failure is primary
    assert hard_exit_observations == [(False, False, False)]
    assert sum(
        retained is interruption
        for retained in wrapper._fatal_secondary_errors
    ) == 1


def test_gate_failure_claim_reclaims_departed_thread_owner(monkeypatch):
    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    hook = runtime._communicator_fatal_hook
    primary = RuntimeError("departed gate claim owner primary")
    departed_done = threading.Event()

    def depart():
        departed_done.set()

    departed = threading.Thread(
        target=depart,
        name="task-18.2-departed-gate-claim-owner",
        daemon=True,
    )
    departed.start()
    _join_task_18_2_call(departed, departed_done)
    token = gate.admit_runtime("reclaim_departed_gate_claim")
    reserved, started = wrapper._pre_reserve_fatal_publication(primary)
    assert (reserved, started) == (primary, True)
    transition = gate.begin_fatal(primary, token)
    outcome = wrapper._reserve_runtime_fatal_outcome(primary)
    assert outcome.primary is primary
    with runtime._terminal_state_lock:
        runtime._pending_fatal_collective = wrapper
        runtime._pending_fatal_context = (None, None, None)
    with wrapper._fatal_condition:
        claim_type = type(wrapper._fatal_publication_gate_failure_claim)
        wrapper._fatal_publication_failure = primary
        wrapper._fatal_publication_gate_failure_claim = claim_type(
            "inflight", departed, time.monotonic() + 5.0
        )

    assert wrapper._fail_reserved_fatal_publication(
        primary,
        transition_handler=hook,
        transition=transition,
    ) == (True, False)

    with wrapper._fatal_condition:
        claim = wrapper._fatal_publication_gate_failure_claim
        assert claim.state == "signaled"
        assert claim.owner is None
        assert claim.deadline is None
    with gate._condition:
        assert gate._fatal_publication_failure is primary

    monkeypatch.setattr(
        wrapper,
        "_fatal_hard_exit",
        lambda: (_ for _ in ()).throw(FatalHardExit()),
    )
    closer, close_results, close_errors, close_done = _start_task_18_2_call(
        runtime.close, name="task-18.2-departed-gate-claim-close"
    )
    _join_task_18_2_call(closer, close_done)
    assert close_results == []
    assert len(close_errors) == 1
    assert isinstance(close_errors[0], FatalHardExit)


def test_gate_failure_claim_repairs_partial_idle_cleanup(monkeypatch):
    _, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    owner = threading.current_thread()

    with wrapper._fatal_condition:
        malformed = SimpleNamespace(
            state="idle",
            owner=owner,
            deadline=time.monotonic() + 5.0,
        )
        wrapper._fatal_publication_gate_failure_claim = malformed

    assert wrapper._repair_fatal_gate_failure_claim(malformed, False) is False

    with wrapper._fatal_condition:
        claim = wrapper._fatal_publication_gate_failure_claim
        assert claim.state == "idle"
        assert claim.owner is None
        assert claim.deadline is None


def test_pre_reserved_fatal_owner_failure_before_adoption_is_fail_stopped(
    monkeypatch,
):
    class PublicationCancelled(BaseException):
        pass

    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, _, _, raw_comm = _single_rank_task_18_2_runtime(
        monkeypatch
    )
    gate = runtime._terminal_gate
    primary = RuntimeError("pre-adoption fatal primary")
    later = RuntimeError("later fatal reporter")
    cancellation = PublicationCancelled(
        "cancel hidden election before owner adoption"
    )
    publisher_name = "task-18.2-pre-adoption-fatal-owner"
    reporter_name = "task-18.2-pre-adoption-fatal-reporter"
    close_name = "task-18.2-pre-adoption-fatal-close"
    injection_observations = []
    join_observations = []
    reporter_adoptions = []
    failure_signals = []
    hard_exit_observations = []

    original_reserve_outcome = wrapper._reserve_runtime_fatal_outcome

    def cancel_before_adoption(primary, *, before_select=None, **kwargs):
        outcome = original_reserve_outcome(
            primary,
            before_select=before_select,
            **kwargs,
        )
        if threading.current_thread().name == publisher_name:
            with wrapper._fatal_condition:
                owner = wrapper._fatal_publication_owner_reservation
                injection_observations.append(
                    (
                        wrapper._fatal_publications,
                        owner is not None,
                        None if owner is None else owner.owner_thread_id,
                        wrapper._fatal_pending_primary,
                    )
                )
            raise cancellation
        return outcome

    monkeypatch.setattr(
        wrapper,
        "_reserve_runtime_fatal_outcome",
        cancel_before_adoption,
    )
    original_begin = wrapper._begin_fatal_publication

    def record_reporter_adoption(error, origin_rank, **kwargs):
        result = original_begin(error, origin_rank, **kwargs)
        reporter_adoptions.append(result)
        return result

    monkeypatch.setattr(
        wrapper, "_begin_fatal_publication", record_reporter_adoption
    )
    original_wait = wrapper._wait_for_joined_fatal_publication

    def record_join(**kwargs):
        with wrapper._fatal_condition:
            join_observations.append(
                (
                    threading.current_thread().name,
                    wrapper._fatal_publications,
                    wrapper._fatal_publication_failure,
                )
            )
        return original_wait(**kwargs)

    monkeypatch.setattr(
        wrapper, "_wait_for_joined_fatal_publication", record_join
    )
    original_fail = gate._fail_fatal_publication

    def record_failure(transition, failure):
        failure_signals.append(
            (
                transition,
                failure,
                wrapper._fatal_lock._is_owned(),
            )
        )
        return original_fail(transition, failure)

    monkeypatch.setattr(gate, "_fail_fatal_publication", record_failure)

    def hard_exit():
        hard_exit_observations.append(
            (
                threading.current_thread().name,
                wrapper._fatal_publications,
                wrapper._fatal_publication_failure,
                wrapper._fatal_lock._is_owned(),
                gate._condition._is_owned(),
            )
        )
        raise FatalHardExit(threading.current_thread().name)

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    publisher, publish_results, publish_errors, publish_done = (
        _start_task_18_2_call(
            lambda: runtime._enter_communicator_fatal(primary),
            name=publisher_name,
        )
    )
    _join_task_18_2_call(publisher, publish_done)

    assert publish_results == []
    assert len(publish_errors) == 1
    assert isinstance(publish_errors[0], FatalHardExit)
    assert injection_observations == [
        (1, True, publisher.ident, primary),
    ]
    assert wrapper._fatal_publications == 1
    assert wrapper._fatal_publication_failure is primary
    with wrapper._fatal_condition:
        retained_owner = wrapper._fatal_publication_owner_reservation
        assert retained_owner.owner_thread is publisher
        assert retained_owner.adopted is False
    with gate._condition:
        assert gate._fatal_publication_failure is primary
    assert gate.phase is _TerminalPhase.FATAL_PENDING
    assert len(failure_signals) == 1
    assert failure_signals[0][0].primary is primary
    assert failure_signals[0][1] is primary
    assert failure_signals[0][2] is False

    reporter, report_results, report_errors, report_done = (
        _start_task_18_2_call(
            lambda: runtime._enter_communicator_fatal(later),
            name=reporter_name,
        )
    )
    _join_task_18_2_call(reporter, report_done)

    assert report_results == []
    assert len(report_errors) == 1
    assert report_errors[0] is primary
    assert len(reporter_adoptions) == 1
    assert reporter_adoptions[0][2:] == (False, True, False)

    closer, close_results, close_errors, close_done = _start_task_18_2_call(
        runtime.close, name=close_name
    )
    _join_task_18_2_call(closer, close_done)

    assert close_results == []
    assert len(close_errors) == 1
    assert close_errors[0] is primary
    assert join_observations == [
        (reporter_name, 1, primary),
        (close_name, 1, primary),
    ]
    assert hard_exit_observations == [
        (publisher_name, 1, primary, False, False),
    ]
    assert len(failure_signals) == 1
    assert wrapper._fatal_secondary_errors == (cancellation, later)
    assert raw_comm.abort_calls == 0
    assert wrapper._fatal_error is None
    assert runtime._terminal_error is None


def test_adopted_fatal_owner_construction_failure_is_fail_stopped(monkeypatch):
    from renormalizer.backend._distributed import collectives as collectives_module

    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, _, _, raw_comm = _single_rank_task_18_2_runtime(
        monkeypatch
    )
    gate = runtime._terminal_gate
    primary = RuntimeError("post-adoption fatal primary")
    later = RuntimeError("later post-adoption reporter")
    construction_failure = KeyboardInterrupt(
        "interrupt fatal reservation construction"
    )
    publisher_name = "task-18.2-post-adoption-fatal-owner"
    reporter_name = "task-18.2-post-adoption-fatal-reporter"
    close_name = "task-18.2-post-adoption-fatal-close"
    adoption_observations = []
    construction_observations = []
    join_observations = []
    failure_signals = []
    hard_exit_observations = []

    original_begin = wrapper._begin_fatal_publication

    def record_adoption(error, origin_rank, **kwargs):
        result = original_begin(error, origin_rank, **kwargs)
        if threading.current_thread().name == publisher_name:
            adoption_observations.append(result)
        return result

    monkeypatch.setattr(wrapper, "_begin_fatal_publication", record_adoption)
    reservation_type = collectives_module._FatalPublicationReservation

    def fail_owner_construction(*args, **kwargs):
        if threading.current_thread().name == publisher_name:
            with wrapper._fatal_condition:
                construction_observations.append(
                    (
                        wrapper._fatal_publications,
                        wrapper._fatal_pending_primary,
                    )
                )
            raise construction_failure
        return reservation_type(*args, **kwargs)

    monkeypatch.setattr(
        collectives_module,
        "_FatalPublicationReservation",
        fail_owner_construction,
    )
    original_wait = wrapper._wait_for_joined_fatal_publication

    def record_join(**kwargs):
        with wrapper._fatal_condition:
            join_observations.append(
                (
                    threading.current_thread().name,
                    wrapper._fatal_publications,
                    wrapper._fatal_publication_failure,
                )
            )
        return original_wait(**kwargs)

    monkeypatch.setattr(
        wrapper, "_wait_for_joined_fatal_publication", record_join
    )
    original_fail = gate._fail_fatal_publication

    def record_failure(transition, failure):
        failure_signals.append(
            (
                transition,
                failure,
                wrapper._fatal_lock._is_owned(),
            )
        )
        return original_fail(transition, failure)

    monkeypatch.setattr(gate, "_fail_fatal_publication", record_failure)

    def hard_exit():
        hard_exit_observations.append(
            (
                threading.current_thread().name,
                wrapper._fatal_publications,
                wrapper._fatal_publication_failure,
                wrapper._fatal_lock._is_owned(),
                gate._condition._is_owned(),
            )
        )
        raise FatalHardExit(threading.current_thread().name)

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    publisher, publish_results, publish_errors, publish_done = (
        _start_task_18_2_call(
            lambda: runtime._enter_communicator_fatal(primary),
            name=publisher_name,
        )
    )
    _join_task_18_2_call(publisher, publish_done)

    assert publish_results == []
    assert len(publish_errors) == 1
    assert isinstance(publish_errors[0], FatalHardExit)
    assert len(adoption_observations) == 1
    assert adoption_observations[0][2:] == (True, False, False)
    assert construction_observations == [(1, primary)]
    assert wrapper._fatal_publications == 1
    assert wrapper._fatal_publication_failure is primary
    with wrapper._fatal_condition:
        retained_owner = wrapper._fatal_publication_owner_reservation
        assert retained_owner.owner_thread is publisher
        assert retained_owner.adopted is True
    with gate._condition:
        assert gate._fatal_publication_failure is primary
    assert gate.phase is _TerminalPhase.FATAL_PENDING
    assert len(failure_signals) == 1
    assert failure_signals[0][0].primary is primary
    assert failure_signals[0][1] is primary
    assert failure_signals[0][2] is False

    reporter, report_results, report_errors, report_done = (
        _start_task_18_2_call(
            lambda: runtime._enter_communicator_fatal(later),
            name=reporter_name,
        )
    )
    _join_task_18_2_call(reporter, report_done)

    assert report_results == []
    assert len(report_errors) == 1
    assert report_errors[0] is primary

    closer, close_results, close_errors, close_done = _start_task_18_2_call(
        runtime.close, name=close_name
    )
    _join_task_18_2_call(closer, close_done)

    assert close_results == []
    assert len(close_errors) == 1
    assert close_errors[0] is primary
    assert join_observations == [
        (reporter_name, 1, primary),
        (close_name, 1, primary),
    ]
    assert hard_exit_observations == [
        (publisher_name, 1, primary, False, False),
    ]
    assert len(failure_signals) == 1
    assert wrapper._fatal_secondary_errors == (
        construction_failure,
        later,
    )
    assert raw_comm.abort_calls == 0
    assert wrapper._fatal_error is None
    assert runtime._terminal_error is None


@pytest.mark.parametrize(
    ("branch", "message"),
    [
        (
            "joined_timeout",
            "communicator fatal publication join timed out",
        ),
        (
            "joined_incomplete",
            "communicator fatal publication ended without completion",
        ),
        (
            "active_b_timeout",
            "active broadcast publication barrier timed out",
        ),
        (
            "close_quiescence_timeout",
            "communicator fatal publication quiescence timed out",
        ),
        ("close_owner_timeout", "communicator close join timed out"),
    ],
)
def test_fatal_wait_timeout_hard_exit_holds_no_fatal_lock(
    monkeypatch, branch, message
):
    from renormalizer.backend._distributed import collectives as collectives_module

    class FatalHardExit(BaseException):
        pass

    _, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    monkeypatch.setattr(collectives_module, "_FATAL_TIMEOUT_S", 0.0)
    hard_exit_observations = []

    if branch == "joined_timeout":
        with wrapper._fatal_condition:
            wrapper._fatal_publications = 1
        call = wrapper._wait_for_joined_fatal_publication
    elif branch == "joined_incomplete":
        call = wrapper._wait_for_joined_fatal_publication
    elif branch == "active_b_timeout":
        with wrapper._fatal_condition:
            wrapper._active_broadcast_agreements = 1
        call = wrapper._wait_for_active_broadcast_agreements
    elif branch == "close_quiescence_timeout":
        with wrapper._fatal_condition:
            wrapper._fatal_publications = 1
        call = wrapper._quiesce_operations_for_close
    else:
        with wrapper._close_lock:
            wrapper._close_in_progress = True
        call = wrapper._wait_for_close_owner

    def hard_exit():
        hard_exit_observations.append(
            (
                wrapper._fatal_condition._is_owned(),
                wrapper._fatal_lock._is_owned(),
            )
        )
        raise FatalHardExit(branch)

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    with pytest.raises(FatalHardExit):
        call()

    assert hard_exit_observations == [(False, False)]
    retained = [
        error
        for error in wrapper._fatal_secondary_errors
        if str(error) == message
    ]
    assert len(retained) == 1


@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE") != "2",
    reason="requires a two-rank torchrun launcher",
)
@pytest.mark.parametrize("dtype", [cupy.float64, cupy.complex128])
def test_real_two_rank_cupy_nccl_collectives_and_cleanup(dtype):
    from renormalizer.backend.distributed_runtime import (
        create_cupy_distributed_runtime,
    )

    assert "CUPYX_DISTRIBUTED_HOST" not in os.environ
    assert "CUPYX_DISTRIBUTED_PORT" not in os.environ
    runtime = create_cupy_distributed_runtime(expected_world_size=2)
    rank = runtime.rank
    local_rank = runtime.local_rank
    master_port = int(os.environ["MASTER_PORT"])
    assert runtime.rendezvous.host == os.environ["MASTER_ADDR"]
    assert runtime.rendezvous.port not in {master_port, 13333}
    collective = runtime.collective
    raw_backend = collective._backend
    backend_reference = weakref.ref(raw_backend)
    store_process = raw_backend._store._process if rank == 0 else None
    del raw_backend

    try:
        with cupy.cuda.Device(local_rank):
            stream = cupy.cuda.Stream(non_blocking=True)
            with stream:
                assert cupy.cuda.runtime.getDevice() == local_rank
                assert cupy.cuda.get_current_stream() is stream

                broadcast_source = cupy.arange(6, dtype=dtype).reshape(2, 3)
                if rank != 0:
                    broadcast_source.fill(-1)
                broadcast = collective.broadcast(broadcast_source, root=0)

                reduced_source = cupy.full((2, 3), rank + 1, dtype=dtype)
                if dtype == cupy.complex128:
                    reduced_source *= 1 + 2j
                reduced = collective.allreduce(reduced_source)

                gathered_source = cupy.arange(4, dtype=dtype).reshape(2, 2) + rank * 10
                if dtype == cupy.complex128:
                    gathered_source *= 1 + 1j
                gathered = collective.allgather(gathered_source, axis=1)

                scattered_source = (
                    cupy.arange(8, dtype=dtype).reshape(2, 4) + rank * 100
                )
                if dtype == cupy.complex128:
                    scattered_source *= 1 - 1j
                scattered = collective.reduce_scatter(scattered_source, axis=1)

                completion = cupy.cuda.Event()
                completion.record(stream)
                assert cupy.cuda.get_current_stream() is stream
            completion.synchronize()

            expected_broadcast = cupy.arange(6, dtype=dtype).reshape(2, 3)
            cupy.testing.assert_array_equal(broadcast, expected_broadcast)
            expected_reduced = cupy.full((2, 3), 3, dtype=dtype)
            if dtype == cupy.complex128:
                expected_reduced *= 1 + 2j
            cupy.testing.assert_array_equal(reduced, expected_reduced)

            gathered_parts = []
            for source_rank in range(2):
                part = cupy.arange(4, dtype=dtype).reshape(2, 2) + source_rank * 10
                if dtype == cupy.complex128:
                    part *= 1 + 1j
                gathered_parts.append(part)
            cupy.testing.assert_array_equal(
                gathered, cupy.concatenate(gathered_parts, axis=1)
            )

            reduced_full = cupy.arange(8, dtype=dtype).reshape(2, 4) * 2 + 100
            if dtype == cupy.complex128:
                reduced_full *= 1 - 1j
            cupy.testing.assert_array_equal(
                scattered, reduced_full[:, rank * 2 : (rank + 1) * 2]
            )

            uuid_bytes = cupy.cuda.runtime.getDeviceProperties(local_rank)["uuid"]
            uuid_hex = bytes(uuid_bytes).hex()
            gpu_uuid = "GPU-{}-{}-{}-{}-{}".format(
                uuid_hex[:8],
                uuid_hex[8:12],
                uuid_hex[12:16],
                uuid_hex[16:20],
                uuid_hex[20:],
            )

        runtime.barrier()
    finally:
        runtime.close()
    runtime.close()
    gc.collect()

    assert collective._backend is None
    assert backend_reference() is None
    if store_process is not None:
        assert not store_process.is_alive()
    print(
        "TASK13_GPU rank={} local_rank={} visible_index={} uuid={} "
        "master_port={} cupyx_port={} stream_ptr={} cleanup=closed".format(
            rank,
            local_rank,
            local_rank,
            gpu_uuid,
            master_port,
            runtime.rendezvous.port,
            stream.ptr,
        ),
        flush=True,
    )


def _reserve_task_18_3_structured_failure(runtime, wrapper, primary):
    gate = runtime._terminal_gate
    token = gate.admit_runtime("task_18_3_structured_failure")
    reserved, started = wrapper._pre_reserve_fatal_publication(primary)
    assert reserved is primary
    assert started is True
    transition = gate.begin_fatal(primary, token)
    outcome = wrapper._reserve_runtime_fatal_outcome(primary)
    assert outcome.primary is primary
    with wrapper._fatal_condition:
        owner = wrapper._fatal_publication_owner_reservation
        owner.election = SimpleNamespace(transition=transition)
    return transition


@pytest.mark.parametrize("catch_path", ("reserved", "monitor_read", "timeout"))
def test_structured_terminalizer_marks_before_diagnostic_and_hard_exits_once(
    monkeypatch,
    catch_path,
):
    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    primary = RuntimeError("{} structured primary".format(catch_path))
    trigger = RuntimeError("{} structured trigger".format(catch_path))
    diagnostic_calls = []
    first_action = []
    action_ready = threading.Event()
    release_diagnostic = threading.Event()
    hard_exits = []

    def block_diagnostic(error):
        diagnostic_calls.append(error)
        if not first_action:
            first_action.append("diagnostic")
            action_ready.set()
        assert release_diagnostic.wait(_TASK_18_2_TIMEOUT_S)

    def hard_exit():
        if not first_action:
            first_action.append("hard_exit")
            action_ready.set()
        hard_exits.append(
            (
                wrapper._fatal_condition._is_owned(),
                wrapper._fatal_lock._is_owned(),
                gate._condition._is_owned(),
            )
        )
        raise FatalHardExit(catch_path)

    monkeypatch.setattr(wrapper, "_record_fatal_secondary", block_diagnostic)
    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)
    if catch_path == "monitor_read":
        monkeypatch.setattr(
            wrapper,
            "_read_fatal_origin",
            lambda **_kwargs: (_ for _ in ()).throw(trigger),
        )

        def call():
            _reserve_task_18_3_structured_failure(runtime, wrapper, primary)
            handoff = wrapper._fatal_monitor_handoff
            with handoff._condition:
                handoff._fatal_reservation = None
                handoff._outcome = None
            return wrapper._monitor_fatal_records()

    elif catch_path == "timeout":
        monkeypatch.setattr(
            wrapper._fatal_monitor_handoff,
            "wait_for_selection",
            lambda *_args, **_kwargs: (
                _ for _ in ()
            ).throw(TimeoutError()),
        )

        def call():
            _reserve_task_18_3_structured_failure(runtime, wrapper, primary)
            return wrapper._wait_for_fatal_monitor_selection()

    else:

        def call():
            transition = _reserve_task_18_3_structured_failure(
                runtime, wrapper, primary
            )
            return wrapper._terminalize_structured_fatal_failure(
                trigger,
                transition_handler=runtime._communicator_fatal_hook,
                transition=transition,
            )

    worker, results, errors, done = _start_task_18_2_call(
        call,
        name="task-18.3-no-fail-stop-diagnostic-{}".format(catch_path),
    )
    try:
        assert action_ready.wait(_TASK_18_2_TIMEOUT_S)
    finally:
        release_diagnostic.set()
        _join_task_18_2_call(worker, done)

    assert results == []
    assert len(errors) == 1
    assert isinstance(errors[0], FatalHardExit)
    assert first_action == ["hard_exit"]
    assert diagnostic_calls == []
    assert hard_exits == [(False, False, False)]
    if catch_path == "timeout":
        assert any(
            "outcome selection timed out" in str(error)
            for error in wrapper._fatal_secondary_errors
        )
    else:
        assert trigger in wrapper._fatal_secondary_errors
    with wrapper._fatal_condition:
        assert wrapper._fatal_publication_failure is primary
    with gate._condition:
        assert gate._fatal_publication_failure is primary


def test_recovered_structured_election_terminalizes_before_diagnostics(
    monkeypatch,
):
    class ElectionFailure(BaseException):
        pass

    class RecoveryFailure(BaseException):
        pass

    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    primary = RuntimeError("recovered structured primary")
    election_failure = ElectionFailure("interrupt after owner reservation")
    recovery_failure = RecoveryFailure("recovery context failed")
    original_pre_reserve = wrapper._pre_reserve_fatal_publication
    diagnostic_calls = []
    marker_observations = []
    hard_exits = []

    def fail_after_owner(*args, **kwargs):
        original_pre_reserve(*args, **kwargs)
        raise election_failure

    def fail_recovery(*_args, **_kwargs):
        raise recovery_failure

    def fail_diagnostic(error):
        diagnostic_calls.append(error)
        with wrapper._fatal_condition:
            collective_marked = wrapper._fatal_publication_failure is primary
        with gate._condition:
            gate_marked = gate._fatal_publication_failure is primary
        marker_observations.append((collective_marked, gate_marked))
        raise AssertionError("fail-stop invoked a pluggable diagnostic")

    def hard_exit():
        hard_exits.append(
            (
                wrapper._fatal_condition._is_owned(),
                wrapper._fatal_lock._is_owned(),
                gate._condition._is_owned(),
            )
        )
        raise FatalHardExit("recovered election")

    monkeypatch.setattr(wrapper, "_pre_reserve_fatal_publication", fail_after_owner)
    monkeypatch.setattr(
        runtime, "_stage_active_broadcast_quarantine_locked", fail_recovery
    )
    monkeypatch.setattr(wrapper, "_record_fatal_secondary", fail_diagnostic)
    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    with pytest.raises(FatalHardExit):
        runtime._enter_communicator_fatal(primary)

    assert diagnostic_calls == []
    assert marker_observations == []
    assert hard_exits == [(False, False, False)]
    assert recovery_failure in wrapper._fatal_secondary_errors
    assert election_failure in wrapper._fatal_secondary_errors
    with wrapper._fatal_condition:
        assert wrapper._fatal_publication_failure is primary
    with gate._condition:
        assert gate._fatal_publication_failure is primary


@pytest.mark.parametrize("start_mode", ("before_real", "after_real"))
def test_remote_fatal_publisher_thread_start_is_transactional(
    monkeypatch,
    start_mode,
):
    class PublisherStartFailure(BaseException):
        pass

    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    primary = RuntimeError("remote publisher canonical primary")
    start_failure = PublisherStartFailure(
        "remote publisher {} start failure".format(start_mode)
    )
    worker_entered = threading.Event()
    release_worker = threading.Event()
    original_start = threading.Thread.start
    original_enter = wrapper._enter_observed_fatal
    original_record_secondary = wrapper._record_fatal_secondary
    diagnostic_observations = []
    hard_exits = []

    def hold_worker(*args, **kwargs):
        worker_entered.set()
        assert release_worker.wait(_TASK_18_2_TIMEOUT_S)
        return original_enter(*args, **kwargs)

    def fail_start(thread):
        if not thread.name.startswith("renormalizer-fatal-publisher-rank-"):
            return original_start(thread)
        if start_mode == "before_real":
            raise start_failure
        original_start(thread)
        assert worker_entered.wait(_TASK_18_2_TIMEOUT_S)
        raise start_failure

    def marker_state():
        with wrapper._fatal_condition:
            collective_marked = wrapper._fatal_publication_failure is primary
        with gate._condition:
            gate_marked = gate._fatal_publication_failure is primary
        return collective_marked, gate_marked

    def record_secondary(error):
        if error is start_failure:
            diagnostic_observations.append(marker_state())
        return original_record_secondary(error)

    def hard_exit():
        hard_exits.append((threading.current_thread(), marker_state()))
        raise FatalHardExit()

    monkeypatch.setattr(wrapper, "_enter_observed_fatal", hold_worker)
    monkeypatch.setattr(threading.Thread, "start", fail_start)
    monkeypatch.setattr(wrapper, "_record_fatal_secondary", record_secondary)
    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)
    publisher = None
    try:
        if start_mode == "before_real":
            with pytest.raises(FatalHardExit):
                wrapper._start_fatal_monitor_publication(primary, 0)
            with wrapper._fatal_condition:
                assert wrapper._fatal_monitor_publisher_thread is None
                assert wrapper._fatal_monitor_publisher_state == "start_failed"
                assert wrapper._fatal_publication_failure is primary
            with gate._condition:
                assert gate._fatal_transition.primary is primary
                assert gate._fatal_publication_failure is primary
            assert diagnostic_observations == []
            assert hard_exits == [
                (threading.current_thread(), (True, True))
            ]
            assert start_failure in wrapper._fatal_secondary_errors
        else:
            publisher = wrapper._start_fatal_monitor_publication(primary, 0)
            assert publisher is wrapper._fatal_monitor_publisher_thread
            release_worker.set()
            publisher.join(_TASK_18_2_TIMEOUT_S)
            assert not publisher.is_alive()
            with wrapper._fatal_condition:
                assert wrapper._fatal_monitor_publisher_state == "completed"
                assert wrapper._fatal_publication_failure is None
            with gate._condition:
                assert gate._fatal_transition.primary is primary
                assert gate._phase is _TerminalPhase.FATAL_PUBLISHED
            assert hard_exits == []
            assert diagnostic_observations == [(False, False)]
            assert start_failure in wrapper._fatal_secondary_errors
    finally:
        release_worker.set()
        if publisher is not None and publisher.ident is not None:
            publisher.join(_TASK_18_2_TIMEOUT_S)


def test_true_pre_owner_monitor_read_failure_marks_before_exit(monkeypatch):
    class MonitorReadFailure(BaseException):
        pass

    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    primary = MonitorReadFailure("fatal monitor read failed before election")
    dispatch_observations = []
    hard_exits = []
    original_dispatch = wrapper._dispatch_fatal_secondaries

    def fail_read(**_kwargs):
        raise primary

    def record_dispatch(errors):
        with wrapper._fatal_condition:
            collective_marked = wrapper._fatal_publication_failure is primary
        with gate._condition:
            gate_marked = gate._fatal_publication_failure is primary
        dispatch_observations.append((collective_marked, gate_marked))
        return original_dispatch(errors)

    def hard_exit():
        hard_exits.append(
            (
                wrapper._fatal_condition._is_owned(),
                wrapper._fatal_lock._is_owned(),
                gate._condition._is_owned(),
            )
        )
        raise FatalHardExit()

    monkeypatch.setattr(wrapper, "_read_fatal_origin", fail_read)
    monkeypatch.setattr(wrapper, "_dispatch_fatal_secondaries", record_dispatch)
    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    with pytest.raises(FatalHardExit):
        wrapper._monitor_fatal_records()

    assert dispatch_observations == []
    assert hard_exits == [(False, False, False)]
    with wrapper._fatal_condition:
        assert wrapper._fatal_publication_failure is primary
    with gate._condition:
        assert gate._fatal_transition.primary is primary
        assert gate._fatal_publication_failure is primary


def test_fatal_outcome_replacement_retains_without_locked_diagnostic(
    monkeypatch,
):
    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    primary = RuntimeError("already selected fatal primary")
    later = RuntimeError("later fatal election candidate")
    outcome = wrapper._fatal_monitor_handoff.select_fatal(primary)
    diagnostic_entered = threading.Event()
    release_diagnostic = threading.Event()
    action_ready = threading.Event()
    first_action = []
    diagnostic_calls = []

    def block_diagnostic(error):
        diagnostic_calls.append(error)
        if not first_action:
            first_action.append("diagnostic")
            action_ready.set()
        diagnostic_entered.set()
        assert release_diagnostic.wait(_TASK_18_2_TIMEOUT_S)

    def reserve_under_terminal_locks():
        with gate._condition:
            with runtime._terminal_state_lock:
                selected = wrapper._reserve_runtime_fatal_outcome(later)
        if not first_action:
            first_action.append("returned")
            action_ready.set()
        return selected

    monkeypatch.setattr(wrapper, "_record_fatal_secondary", block_diagnostic)
    worker, results, errors, done = _start_task_18_2_call(
        reserve_under_terminal_locks,
        name="task-18.3-locked-fatal-replacement",
    )
    try:
        assert action_ready.wait(_TASK_18_2_TIMEOUT_S)
    finally:
        release_diagnostic.set()
        _join_task_18_2_call(worker, done)

    assert errors == []
    assert results == [outcome]
    assert first_action == ["returned"]
    assert diagnostic_entered.is_set() is False
    assert diagnostic_calls == []
    assert later in wrapper._fatal_secondary_errors


def test_owner_and_non_owner_fail_stop_share_one_hard_exit(monkeypatch):
    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    primary = RuntimeError("shared hard-exit primary")
    later = RuntimeError("non-owner fail-stop reporter")
    owner_ready = threading.Event()
    race = threading.Barrier(2)
    hard_exits = []

    def hard_exit():
        hard_exits.append(threading.current_thread().name)
        raise FatalHardExit(threading.current_thread().name)

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    def owner_call():
        token = gate.admit_runtime("shared_hard_exit_owner")
        reserved, started = wrapper._pre_reserve_fatal_publication(primary)
        assert (reserved, started) == (primary, True)
        transition = gate.begin_fatal(primary, token)
        outcome = wrapper._reserve_runtime_fatal_outcome(primary)
        assert outcome.primary is primary
        with wrapper._fatal_condition:
            owner = wrapper._fatal_publication_owner_reservation
            owner.election = SimpleNamespace(transition=transition)
        owner_ready.set()
        race.wait(_TASK_18_2_TIMEOUT_S)
        return wrapper._terminalize_structured_fatal_failure(
            primary,
            transition_handler=runtime._communicator_fatal_hook,
            transition=transition,
        )

    def non_owner_call():
        assert owner_ready.wait(_TASK_18_2_TIMEOUT_S)
        race.wait(_TASK_18_2_TIMEOUT_S)
        return wrapper._fail_stop_fatal_path(later)

    owner, owner_results, owner_errors, owner_done = _start_task_18_2_call(
        owner_call,
        name="task-18.3-hard-exit-owner",
    )
    non_owner, non_owner_results, non_owner_errors, non_owner_done = (
        _start_task_18_2_call(
            non_owner_call,
            name="task-18.3-hard-exit-non-owner",
        )
    )
    _join_task_18_2_call(owner, owner_done)
    _join_task_18_2_call(non_owner, non_owner_done)

    assert owner_results == []
    assert non_owner_results == []
    combined = owner_errors + non_owner_errors
    assert sum(isinstance(error, FatalHardExit) for error in combined) == 1
    assert sum(error is primary for error in combined) == 1
    assert len(hard_exits) == 1
    with wrapper._fatal_condition:
        assert wrapper._fatal_publication_failure is primary
    with gate._condition:
        assert gate._fatal_publication_failure is primary


def test_blocked_monitor_read_does_not_block_stop_or_publish_late_result(
    monkeypatch,
):
    _, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    read_entered = threading.Event()
    release_read = threading.Event()
    stop_returned = threading.Event()
    publisher_calls = []
    stop_results = []
    read_calls = []

    def blocked_read(**_kwargs):
        read_calls.append(True)
        if len(read_calls) > 1:
            return None
        read_entered.set()
        assert release_read.wait(_TASK_18_2_TIMEOUT_S)
        return 0

    def unexpected_publisher(error, origin_rank):
        publisher_calls.append((error, origin_rank))
        return None

    monkeypatch.setattr(wrapper, "_read_fatal_origin", blocked_read)
    monkeypatch.setattr(
        wrapper, "_start_fatal_monitor_publication", unexpected_publisher
    )
    monitor = threading.Thread(
        target=wrapper._monitor_fatal_records,
        name="task-18.3-blocked-monitor-read",
        daemon=True,
    )
    wrapper._fatal_monitor_thread = monitor

    def request_stop():
        stop_results.append(wrapper._request_fatal_monitor_stop())
        stop_returned.set()

    stopper = threading.Thread(
        target=request_stop,
        name="task-18.3-monitor-stopper",
        daemon=True,
    )
    try:
        monitor.start()
        assert read_entered.wait(_TASK_18_2_TIMEOUT_S)
        stopper.start()
        assert stop_returned.wait(0.25)
        release_read.set()
        stopper.join(_TASK_18_2_TIMEOUT_S)
        monitor.join(_TASK_18_2_TIMEOUT_S)
        assert not stopper.is_alive()
        assert not monitor.is_alive()
    finally:
        release_read.set()
        if stopper.ident is not None:
            stopper.join(_TASK_18_2_TIMEOUT_S)
        if monitor.ident is not None:
            monitor.join(_TASK_18_2_TIMEOUT_S)

    assert len(stop_results) == 1
    outcome = wrapper._fatal_monitor_handoff.observe_outcome()
    assert outcome.kind == "stopped_clean"
    assert outcome.generation == stop_results[0]
    assert publisher_calls == []


@pytest.mark.parametrize("start_mode", ("before_real", "after_real"))
def test_fatal_abort_thread_start_is_transactional(monkeypatch, start_mode):
    class AbortStartFailure(BaseException):
        pass

    _, wrapper, _, _, raw_comm = _single_rank_task_18_2_runtime(monkeypatch)
    start_failure = AbortStartFailure("abort {} start".format(start_mode))
    abort_entered = threading.Event()
    release_abort = threading.Event()
    original_start = threading.Thread.start
    original_abort = raw_comm.abort

    def hold_abort():
        abort_entered.set()
        assert release_abort.wait(_TASK_18_2_TIMEOUT_S)
        return original_abort()

    def fail_start(thread):
        if not thread.name.startswith("renormalizer-nccl-abort-rank-"):
            return original_start(thread)
        if start_mode == "before_real":
            raise start_failure
        original_start(thread)
        assert abort_entered.wait(_TASK_18_2_TIMEOUT_S)
        raise start_failure

    monkeypatch.setattr(raw_comm, "abort", hold_abort)
    monkeypatch.setattr(threading.Thread, "start", fail_start)
    event = wrapper._abort_local_communicator()
    worker = wrapper._fatal_abort_thread
    try:
        if start_mode == "before_real":
            assert event.is_set()
            assert worker is None
            assert wrapper._fatal_abort_state == "start_failed"
            assert wrapper._fatal_abort_error is start_failure
        else:
            assert worker is not None
            assert wrapper._fatal_abort_state == "started"
            release_abort.set()
            assert event.wait(_TASK_18_2_TIMEOUT_S)
            worker.join(_TASK_18_2_TIMEOUT_S)
            assert not worker.is_alive()
            assert wrapper._fatal_abort_state == "completed"
            assert wrapper._fatal_abort_completed is True
            assert start_failure in wrapper._fatal_secondary_errors
    finally:
        release_abort.set()
        if worker is not None and worker.ident is not None:
            worker.join(_TASK_18_2_TIMEOUT_S)


@pytest.mark.parametrize("start_mode", ("before_real", "after_real"))
def test_fatal_monitor_thread_start_is_transactional(monkeypatch, start_mode):
    class MonitorStartFailure(BaseException):
        pass

    _, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    start_failure = MonitorStartFailure("monitor {} start".format(start_mode))
    monitor_entered = threading.Event()
    release_monitor = threading.Event()
    original_start = threading.Thread.start

    def hold_monitor():
        monitor_entered.set()
        assert release_monitor.wait(_TASK_18_2_TIMEOUT_S)

    def fail_start(thread):
        if not thread.name.startswith("renormalizer-fatal-monitor-rank-"):
            return original_start(thread)
        if start_mode == "before_real":
            raise start_failure
        original_start(thread)
        assert monitor_entered.wait(_TASK_18_2_TIMEOUT_S)
        raise start_failure

    monkeypatch.setattr(wrapper, "_monitor_fatal_records", hold_monitor)
    monkeypatch.setattr(threading.Thread, "start", fail_start)
    worker = None
    try:
        if start_mode == "before_real":
            with pytest.raises(MonitorStartFailure) as caught:
                wrapper._start_fatal_monitor()
            assert caught.value is start_failure
            assert wrapper._fatal_monitor_thread is None
            assert wrapper._fatal_monitor_state == "start_failed"
        else:
            worker = wrapper._start_fatal_monitor()
            assert worker is wrapper._fatal_monitor_thread
            assert wrapper._fatal_monitor_state == "started"
            release_monitor.set()
            worker.join(_TASK_18_2_TIMEOUT_S)
            assert not worker.is_alive()
            assert wrapper._fatal_monitor_state == "completed"
            assert start_failure in wrapper._fatal_secondary_errors
    finally:
        release_monitor.set()
        if worker is not None and worker.ident is not None:
            worker.join(_TASK_18_2_TIMEOUT_S)


@pytest.mark.parametrize("start_mode", ("before_real", "after_real"))
def test_fatal_control_initialization_commits_only_with_owned_monitor(
    monkeypatch,
    start_mode,
):
    class MonitorStartFailure(BaseException):
        pass

    class FatalHardExit(BaseException):
        pass

    store = _SharedStore(1)
    raw_comm = _RawComm()
    wrapper, backend = _cpu_collective(0, 1, store, raw_comm)
    runtime = _task_18_2_runtime(wrapper, backend)
    gate = runtime._terminal_gate
    start_failure = MonitorStartFailure(
        "bootstrap monitor {} start".format(start_mode)
    )
    monitor_entered = threading.Event()
    release_monitor = threading.Event()
    hard_exits = []
    original_start = threading.Thread.start

    def hold_monitor():
        monitor_entered.set()
        assert release_monitor.wait(_TASK_18_2_TIMEOUT_S)

    def fail_start(thread):
        if not thread.name.startswith("renormalizer-fatal-monitor-rank-"):
            return original_start(thread)
        if start_mode == "before_real":
            raise start_failure
        original_start(thread)
        assert monitor_entered.wait(_TASK_18_2_TIMEOUT_S)
        raise start_failure

    def hard_exit():
        with wrapper._fatal_condition:
            collective_marker = wrapper._fatal_publication_failure
        with gate._condition:
            gate_marker = gate._fatal_publication_failure
        hard_exits.append((collective_marker, gate_marker))
        raise FatalHardExit()

    monkeypatch.setattr(wrapper, "_monitor_fatal_records", hold_monitor)
    monkeypatch.setattr(threading.Thread, "start", fail_start)
    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)
    worker = None
    try:
        if start_mode == "before_real":
            with pytest.raises(FatalHardExit):
                wrapper._bootstrap_fatal_control()
            assert wrapper._fatal_control_initialized is False
            assert wrapper._fatal_monitor_thread is None
            assert wrapper._fatal_monitor_state == "start_failed"
            with wrapper._fatal_condition:
                assert wrapper._fatal_publication_failure is start_failure
            with gate._condition:
                assert gate._fatal_transition.primary is start_failure
                assert gate._fatal_publication_failure is start_failure
            assert hard_exits == [(start_failure, start_failure)]
            with pytest.raises(BaseException):
                wrapper._bootstrap_fatal_control()
        else:
            assert wrapper._bootstrap_fatal_control() == 0
            worker = wrapper._fatal_monitor_thread
            assert worker is not None
            assert wrapper._fatal_control_initialized is True
            assert wrapper._fatal_monitor_state == "started"
            assert start_failure in wrapper._fatal_secondary_errors
            with wrapper._fatal_condition:
                assert wrapper._fatal_publication_failure is None
            with gate._condition:
                assert gate._fatal_transition is None
            assert hard_exits == []
    finally:
        release_monitor.set()
        if worker is not None and worker.ident is not None:
            worker.join(_TASK_18_2_TIMEOUT_S)


@pytest.mark.parametrize(
    "begin_boundary",
    ("handler_lookup", "before_real_begin", "after_real_begin"),
)
def test_monitor_start_failure_totalizes_fatal_begin_failure(
    monkeypatch,
    begin_boundary,
):
    class MonitorStartFailure(BaseException):
        pass

    class FatalBeginFailure(BaseException):
        pass

    class FatalHardExit(BaseException):
        pass

    store = _SharedStore(1)
    raw_comm = _RawComm()
    wrapper, backend = _cpu_collective(0, 1, store, raw_comm)
    runtime = _task_18_2_runtime(wrapper, backend)
    gate = runtime._terminal_gate
    start_failure = MonitorStartFailure("fatal monitor did not start")
    begin_failure = FatalBeginFailure(
        "{} fatal begin failed".format(begin_boundary)
    )
    hard_exits = []
    diagnostic_calls = []
    original_start = threading.Thread.start
    original_begin = runtime._communicator_fatal_hook.begin

    def fail_monitor_start(thread):
        if thread.name.startswith("renormalizer-fatal-monitor-rank-"):
            raise start_failure
        return original_start(thread)

    if begin_boundary == "handler_lookup":
        monkeypatch.setattr(
            wrapper,
            "_fatal_transition_handler_callback",
            lambda: (_ for _ in ()).throw(begin_failure),
        )
    elif begin_boundary == "before_real_begin":
        monkeypatch.setattr(
            runtime._communicator_fatal_hook,
            "begin",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(begin_failure),
        )
    else:

        def fail_after_begin(*args, **kwargs):
            original_begin(*args, **kwargs)
            raise begin_failure

        monkeypatch.setattr(
            runtime._communicator_fatal_hook,
            "begin",
            fail_after_begin,
        )

    def diagnostic(error):
        diagnostic_calls.append(error)
        raise AssertionError("fatal-begin fail-stop dispatched diagnostics")

    def hard_exit():
        with wrapper._fatal_condition:
            collective_marker = wrapper._fatal_publication_failure
            publications = wrapper._fatal_publications
            owner = wrapper._fatal_publication_owner_reservation
        with gate._condition:
            transition = gate._fatal_transition
            gate_marker = gate._fatal_publication_failure
        hard_exits.append(
            (
                collective_marker,
                gate_marker,
                None if transition is None else transition.primary,
                publications,
                owner,
                wrapper._fatal_condition._is_owned(),
                gate._condition._is_owned(),
            )
        )
        raise FatalHardExit()

    monkeypatch.setattr(threading.Thread, "start", fail_monitor_start)
    monkeypatch.setattr(wrapper, "_record_fatal_secondary", diagnostic)
    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)

    with pytest.raises(FatalHardExit):
        wrapper._bootstrap_fatal_control()

    assert wrapper._fatal_control_initialized is False
    assert wrapper._fatal_monitor_thread is None
    assert wrapper._fatal_monitor_state == "start_failed"
    assert diagnostic_calls == []
    assert len(hard_exits) == 1
    (
        collective_marker,
        gate_marker,
        transition_primary,
        publications,
        owner,
        collective_locked,
        gate_locked,
    ) = hard_exits[0]
    assert (
        collective_marker,
        gate_marker,
        transition_primary,
        collective_locked,
        gate_locked,
    ) == (start_failure, start_failure, start_failure, False, False)
    if begin_boundary == "after_real_begin":
        assert publications == 1
        assert owner.primary is start_failure
        assert owner.owner_thread is threading.current_thread()
    else:
        assert publications == 0
        assert owner is None
    assert begin_failure in wrapper._fatal_secondary_errors
    assert begin_failure in runtime._terminal_secondary_errors


def test_direct_fail_stop_installs_collective_and_gate_markers_before_exit(
    monkeypatch,
):
    class FatalHardExit(BaseException):
        pass

    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    primary = RuntimeError("direct fail-stop primary")
    observations = []

    def hard_exit():
        with wrapper._fatal_condition:
            collective_marker = wrapper._fatal_publication_failure
        with gate._condition:
            transition = gate._fatal_transition
            gate_marker = gate._fatal_publication_failure
        observations.append(
            (
                collective_marker,
                gate_marker,
                None if transition is None else transition.primary,
                wrapper._fatal_condition._is_owned(),
                gate._condition._is_owned(),
            )
        )
        raise FatalHardExit()

    monkeypatch.setattr(wrapper, "_fatal_hard_exit", hard_exit)
    with pytest.raises(FatalHardExit):
        wrapper._diagnose_and_hard_exit(primary)

    assert observations == [(primary, primary, primary, False, False)]


@pytest.mark.parametrize(
    "operation",
    ("fatal_bootstrap", "status_bootstrap"),
)
def test_construction_store_barriers_reject_expired_lifecycle_deadline(
    monkeypatch,
    operation,
):
    from renormalizer.backend._distributed import collectives as collectives_module

    _, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    calls = []

    class Store:
        host = "127.0.0.1"
        port = 1

        def __getitem__(self, key):
            calls.append(("get", key))
            raise AssertionError("expired lifecycle performed a store read")

        def __setitem__(self, key, value):
            calls.append(("set", key, value))
            raise AssertionError("expired lifecycle performed a store write")

        def barrier(self):
            calls.append(("barrier",))
            raise AssertionError("expired lifecycle entered a store barrier")

    wrapper._bootstrap_store_proxy = Store()
    monkeypatch.setattr(collectives_module.time, "monotonic", lambda: 10.0)
    with pytest.raises(TimeoutError, match="lifecycle deadline"):
        if operation == "fatal_bootstrap":
            wrapper._bootstrap_fatal_control(_deadline=9.0)
        else:
            wrapper._fatal_control_initialized = True
            wrapper._bootstrap_status_or(0, _deadline=9.0)
    assert calls == []


def test_fatal_monitor_start_receives_construction_lifecycle_deadline(
    monkeypatch,
):
    class FatalHardExit(BaseException):
        pass

    store = _SharedStore(1)
    wrapper, backend = _cpu_collective(0, 1, store, _RawComm())
    _task_18_2_runtime(wrapper, backend)
    deadline = time.monotonic() + _TASK_18_2_TIMEOUT_S
    captured = []

    def start_monitor(*, _deadline):
        captured.append(_deadline)
        return _completed_test_monitor(wrapper)

    monkeypatch.setattr(wrapper, "_start_fatal_monitor", start_monitor)
    monkeypatch.setattr(
        wrapper,
        "_fatal_hard_exit",
        lambda: (_ for _ in ()).throw(FatalHardExit()),
    )

    assert wrapper._bootstrap_fatal_control(_deadline=deadline) == 0
    assert captured == [deadline]


def test_fatal_origin_read_uses_bounded_store_get_with_inherited_deadline(
    monkeypatch,
):
    _, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    deadline = time.monotonic() + _TASK_18_2_TIMEOUT_S
    calls = []

    class RawStore:
        def __getitem__(self, key):
            raise AssertionError("fatal origin used an unbounded raw store read")

    wrapper._bootstrap_store_proxy = RawStore()

    def bounded_get(key, *, _deadline=None):
        calls.append((key, _deadline))
        return 1

    monkeypatch.setattr(wrapper, "_fatal_store_get", bounded_get)

    assert wrapper._read_fatal_origin(_deadline=deadline) == 0
    assert calls == [(wrapper._fatal_key(0), deadline)]


def test_structured_fatal_election_inherits_current_lifecycle_deadline(
    monkeypatch,
):
    from renormalizer.backend._distributed import collectives as collectives_module

    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    deadline = time.monotonic() + _TASK_18_2_TIMEOUT_S
    monkeypatch.setattr(gate, "_deadline", lambda _timeout_s: deadline)
    request = gate.admit_runtime("begin_runtime_close")
    close_transition, elected = gate._freeze_runtime_close(request)
    assert elected is True
    assert close_transition.deadline == deadline

    primary = RuntimeError("structured election deadline")
    transition = runtime._begin_communicator_fatal(primary)
    with wrapper._fatal_condition:
        retained_owner = wrapper._fatal_publication_owner_reservation
    election = retained_owner.election

    assert transition is election.prepared_transition
    assert transition is gate._fatal_transition
    assert transition.deadline == deadline
    monkeypatch.setattr(
        collectives_module.time,
        "monotonic",
        lambda: (_ for _ in ()).throw(
            AssertionError("structured fatal wait refreshed its deadline")
        ),
    )
    assert wrapper._inherited_fatal_deadline() == deadline


def test_structured_fatal_election_ignores_expired_committed_construction_deadline(
    monkeypatch,
):
    from renormalizer.backend._distributed import collectives as collectives_module

    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    gate = runtime._terminal_gate
    transaction = gate._prepare_lease_construction("lease_construction")
    _, token = gate.begin_lease(
        "lease_construction",
        _transaction=transaction,
    )
    result = object()
    assert gate._commit_lease_construction(
        transaction,
        result,
        lambda: None,
        lambda _primary: None,
    ) is result
    assert transaction.state == "committed"
    assert gate._current_thread_admission() is None

    expired = time.monotonic() - 1.0
    transaction.deadline = expired
    fresh = time.monotonic() + _TASK_18_2_TIMEOUT_S
    requested = []

    def new_deadline(timeout_s):
        requested.append(timeout_s)
        return fresh

    monkeypatch.setattr(gate, "_deadline", new_deadline)
    primary = RuntimeError("structured election after committed construction")
    transition = runtime._begin_communicator_fatal(primary)
    with wrapper._fatal_condition:
        retained_owner = wrapper._fatal_publication_owner_reservation
    election = retained_owner.election

    assert requested == [pytest.approx(_TERMINAL_TIMEOUT_S)]
    assert transition is election.prepared_transition
    assert transition is gate._fatal_transition
    assert transition.deadline == fresh
    assert transition.deadline != expired
    monkeypatch.setattr(
        gate,
        "_deadline",
        lambda _timeout_s: (_ for _ in ()).throw(
            AssertionError("structured fatal lifecycle refreshed its deadline")
        ),
    )
    monkeypatch.setattr(
        collectives_module.time,
        "monotonic",
        lambda: (_ for _ in ()).throw(
            AssertionError("structured fatal wait refreshed its deadline")
        ),
    )
    assert wrapper._inherited_fatal_deadline() == fresh


@pytest.mark.parametrize(
    "family",
    (
        "abort_wait",
        "acknowledgments",
        "monitor_outcome",
        "monitor_selection",
        "monitor_exit",
        "monitor_join",
        "fatal_prepare",
        "fatal_publish",
    ),
)
def test_every_fatal_wait_family_consumes_one_transition_deadline(
    monkeypatch,
    family,
):
    runtime, wrapper, _, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    primary = RuntimeError("fatal deadline family {}".format(family))
    transition = runtime._terminal_gate.begin_fatal(primary)
    deadline = transition.deadline
    observed = []

    if family == "abort_wait":
        class AbortEvent:
            def wait(self, timeout):
                observed.append(timeout)
                return True

        wrapper._fatal_abort_event = AbortEvent()
        wrapper._fatal_abort_completed = True
        wrapper._fatal_abort_error = None
        monkeypatch.setattr(
            time,
            "monotonic",
            lambda: deadline - 1.25,
        )
        wrapper._wait_for_local_communicator_abort(_deadline=deadline)
        assert observed == [pytest.approx(1.25)]
    elif family == "acknowledgments":
        monkeypatch.setattr(
            wrapper,
            "_wait_for_all_control_records",
            lambda *_args, _deadline=None, **_kwargs: observed.append(_deadline),
        )
        wrapper._wait_for_fatal_acknowledgments(_deadline=deadline)
        assert observed == [deadline]
    elif family == "monitor_outcome":
        expected = object()
        monkeypatch.setattr(
            wrapper._fatal_monitor_handoff,
            "wait_for_outcome",
            lambda generation, *, _deadline: (
                observed.append((generation, _deadline)) or expected
            ),
        )
        assert wrapper._wait_for_fatal_monitor_outcome(
            7,
            _deadline=deadline,
        ) is expected
        assert observed == [(7, deadline)]
    elif family == "monitor_selection":
        expected = object()
        monkeypatch.setattr(
            wrapper._fatal_monitor_handoff,
            "wait_for_selection",
            lambda *, _deadline: observed.append(_deadline) or expected,
        )
        assert wrapper._wait_for_fatal_monitor_selection(
            _deadline=deadline,
        ) is expected
        assert observed == [deadline]
    elif family == "monitor_exit":
        outcome = object()
        monkeypatch.setattr(
            wrapper._fatal_monitor_handoff,
            "wait_for_exit",
            lambda retained, *, _deadline: (
                observed.append((retained, _deadline)) or retained
            ),
        )
        assert wrapper._wait_for_fatal_monitor_exit(
            outcome,
            _deadline=deadline,
        ) is outcome
        assert observed == [(outcome, deadline)]
    elif family == "monitor_join":
        class Monitor:
            def join(self, timeout):
                observed.append(timeout)

            @staticmethod
            def is_alive():
                return False

        monkeypatch.setattr(time, "monotonic", lambda: deadline - 2.0)
        wrapper._join_fatal_monitor(Monitor(), _deadline=deadline)
        assert observed == [pytest.approx(2.0)]
    elif family == "fatal_prepare":
        wrapper._fatal_pending_primary = primary
        wrapper._fatal_store_announced = True
        wrapper._fatal_monitor_thread = None
        outcome = SimpleNamespace(kind="fatal_elected", primary=primary)
        monkeypatch.setattr(
            wrapper,
            "_select_runtime_fatal_outcome",
            lambda retained, *, _deadline: (
                observed.append(("select", retained, _deadline)) or outcome
            ),
        )
        monkeypatch.setattr(
            wrapper,
            "_abort_local_communicator",
            lambda *, _deadline: observed.append(("abort", _deadline)),
        )
        wrapper._prepare_local_fatal_locked(_deadline=deadline)
        assert observed == [
            ("select", primary, deadline),
            ("abort", deadline),
        ]
    else:
        wrapper._fatal_control_initialized = True
        wrapper._fatal_pending_primary = primary
        monkeypatch.setattr(
            wrapper,
            "_wait_for_local_communicator_abort",
            lambda *, _deadline: observed.append(("abort", _deadline)),
        )
        monkeypatch.setattr(
            wrapper,
            "_fatal_store_set",
            lambda *_args, _deadline=None, **_kwargs: observed.append(
                ("store", _deadline)
            ),
        )
        monkeypatch.setattr(
            wrapper,
            "_wait_for_fatal_acknowledgments",
            lambda *, _deadline: observed.append(("ack", _deadline)),
        )
        assert wrapper._publish_communicator_fatal_locked(
            primary,
            _deadline=deadline,
        ) is primary
        assert observed == [
            ("abort", deadline),
            ("store", deadline),
            ("ack", deadline),
        ]
