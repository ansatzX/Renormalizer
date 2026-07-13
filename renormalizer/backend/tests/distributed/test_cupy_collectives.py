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
from renormalizer.backend._distributed.terminal import _TerminalPhase


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

    def local_fatal_capability_code():
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


def _single_rank_task_18_2_runtime(monkeypatch, *, start_monitor=False):
    store = _SharedStore(1)
    raw_comm = _RawComm()
    wrapper, backend = _cpu_collective(0, 1, store, raw_comm)
    start_fatal_monitor = wrapper._start_fatal_monitor
    monkeypatch.setattr(wrapper, "_start_fatal_monitor", lambda: None)
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

    def record_abort_start():
        event = original_abort()
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

    def record_store_set(key, value):
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
        return original_store_set(key, value)

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

    original_begin_fatal = gate.begin_fatal

    def observe_pending(error, discovering_token=None):
        transition = original_begin_fatal(error, discovering_token)
        owner_pending.set()
        return transition

    monkeypatch.setattr(gate, "begin_fatal", observe_pending)

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
        monkeypatch.setattr(wrapper, "_start_fatal_monitor", lambda: None)
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

    original_begin_fatal = gate.begin_fatal

    def record_begin_fatal(error, discovering_token=None):
        transition = original_begin_fatal(error, discovering_token)
        transitions.append(transition)
        pending_entered.set()
        return transition

    monkeypatch.setattr(gate, "begin_fatal", record_begin_fatal)
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

    original_begin_fatal = gate.begin_fatal

    def record_begin_fatal(error, discovering_token=None):
        transition = original_begin_fatal(error, discovering_token)
        pending_entered.set()
        return transition

    monkeypatch.setattr(gate, "begin_fatal", record_begin_fatal)
    original_store_set = wrapper._fatal_store_set

    def record_store_set(key, value):
        if int(value) == 1:
            writes.append(key)
        if key == wrapper._fatal_ack_key(wrapper.rank):
            monitor_stopped_before_ack.append(wrapper._fatal_monitor_stop.is_set())
        return original_store_set(key, value)

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

    def record_store_set(key, value):
        with gate._condition:
            store_access_phases.append(("set", key, gate._phase))
        if key == wrapper._fatal_key(wrapper.rank):
            record_sentinel("fatal_key")
        elif key == wrapper._fatal_ack_key(wrapper.rank):
            record_sentinel("ack")
        return original_store_set(key, value)

    monkeypatch.setattr(wrapper, "_fatal_store_set", record_store_set)
    original_store_get = wrapper._fatal_store_get

    def record_store_get(key):
        with gate._condition:
            store_access_phases.append(("get", key, gate._phase))
        return original_store_get(key)

    monkeypatch.setattr(wrapper, "_fatal_store_get", record_store_get)
    original_read = wrapper._read_fatal_origin

    def record_monitor_read():
        with gate._condition:
            store_access_phases.append(("monitor_read", None, gate._phase))
        return original_read()

    monkeypatch.setattr(wrapper, "_read_fatal_origin", record_monitor_read)
    original_abort = wrapper._abort_local_communicator

    def record_abort():
        record_sentinel("abort")
        return original_abort()

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

    def record_collective_publish(primary):
        record_sentinel("collective_public")
        return original_collective_publish(primary)

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

    def block_owner_after_monitor_join(thread):
        result = original_join(thread)
        if threading.current_thread().name == owner_name:
            owner_after_monitor_exit.set()
            assert release_owner.wait(_TASK_18_2_TIMEOUT_S * 2)
        return result

    monkeypatch.setattr(wrapper, "_join_fatal_monitor", block_owner_after_monitor_join)
    original_wait_for_publication = wrapper._wait_for_joined_fatal_publication

    def record_publication_join():
        if threading.current_thread().name == close_name:
            close_waiting_for_publication.set()
        return original_wait_for_publication()

    monkeypatch.setattr(
        wrapper, "_wait_for_joined_fatal_publication", record_publication_join
    )

    # Keep RED cleanup inside pytest even if the buggy close clears live fields.
    monkeypatch.setattr(
        wrapper,
        "_fatal_store_set",
        lambda key, value: retained_store.__setitem__(key, value),
    )
    monkeypatch.setattr(wrapper, "_fatal_store_get", retained_store.__getitem__)

    def retained_abort():
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

    def record_close_join():
        if threading.current_thread().name == close_name:
            with wrapper._fatal_condition:
                close_join_observations.append(
                    (
                        wrapper._fatal_publications,
                        wrapper._fatal_protocol_completed,
                    )
                )
            close_join_entered.set()
        return original_join()

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

    def record_acknowledgment(key, value):
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
        return original_store_set(key, value)

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


def test_preexisting_backend_poison_is_canonical_for_fatal_publication(
    monkeypatch,
):
    runtime, wrapper, backend, _, _ = _single_rank_task_18_2_runtime(monkeypatch)
    earlier = RuntimeError("preexisting backend terminal poison")
    later = RuntimeError("later communicator failure")
    backend._execution_terminal_error = earlier

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

    def pause_collective_prepare():
        prepare_entered.set()
        assert release_prepare.wait(_TASK_18_2_TIMEOUT_S * 2)
        return original_prepare()

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

    def record_reporter_join():
        if threading.current_thread().name == "task-18.2-protocol-reporter":
            reporter_joining.set()
        return original_join()

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

    def record_store(key, value):
        if key == wrapper._fatal_key(wrapper.rank):
            sentinels.append(
                (
                    "store",
                    wrapper._fatal_pending_primary,
                    runtime._terminal_gate._fatal_transition.primary,
                )
            )
        return original_store_set(key, value)

    monkeypatch.setattr(wrapper, "_fatal_store_set", record_store)
    original_abort = wrapper._abort_local_communicator

    def record_abort():
        sentinels.append(
            (
                "abort",
                wrapper._fatal_pending_primary,
                runtime._terminal_gate._fatal_transition.primary,
            )
        )
        return original_abort()

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

    def record_monitor_store_read():
        with gate._condition:
            store_read_phases.append(gate._phase)
        return original_read()

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

    def store_get(key):
        record_store_access(("get", key))
        return original_store_get(key)

    monkeypatch.setattr(wrapper, "_fatal_store_get", store_get)
    original_store_set = wrapper._fatal_store_set

    def store_set(key, value):
        record_store_access(("set", key))
        return original_store_set(key, value)

    monkeypatch.setattr(wrapper, "_fatal_store_set", store_set)
    original_read = wrapper._read_fatal_origin

    def read_fatal_origin():
        record_store_access("monitor_read")
        return original_read()

    monkeypatch.setattr(wrapper, "_read_fatal_origin", read_fatal_origin)
    original_select = wrapper._select_fatal_monitor_outcome

    def select_outcome(kind, value, **kwargs):
        outcome = original_select(kind, value, **kwargs)
        selected_outcomes.append(outcome)
        outcome_selected.set()
        return outcome

    monkeypatch.setattr(wrapper, "_select_fatal_monitor_outcome", select_outcome)
    original_wait_for_exit = getattr(
        wrapper, "_wait_for_fatal_monitor_exit", lambda outcome: outcome
    )

    def wait_for_exit(outcome):
        exit_waits.append(outcome)
        return original_wait_for_exit(outcome)

    monkeypatch.setattr(
        wrapper, "_wait_for_fatal_monitor_exit", wait_for_exit, raising=False
    )
    original_join = wrapper._join_fatal_monitor

    def join_monitor(thread):
        joins.append(thread)
        return original_join(thread)

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
    original_join = getattr(wrapper, "_join_fatal_monitor", lambda thread: None)

    def observe_join(thread):
        close_joining_monitor.set()
        return original_join(thread)

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
        wrapper, "_wait_for_fatal_monitor_exit", lambda outcome: outcome
    )

    def checked_wait(outcome):
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
        return original_wait(outcome)

    monkeypatch.setattr(
        wrapper, "_wait_for_fatal_monitor_exit", checked_wait, raising=False
    )
    original_join = getattr(wrapper, "_join_fatal_monitor", lambda thread: None)

    def checked_join(thread):
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
        return original_join(thread)

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

    def staggered_origin_ack(key, value):
        if key == wrappers[1]._fatal_ack_key(1):
            origin_waiting.set()
            assert release_origin.wait(3.0)
        return original_store_set(key, value)

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

        def pause_active_record(key, value, rank=rank, original_set=original_set):
            if key == wrappers[rank]._active_b_key(rank):
                sequence, _ = wrappers[rank]._decode_active_b(value)
                if int(sequence) == 1:
                    agreement_entered[rank].set()
                    assert release_agreement.wait(3.0)
            return original_set(key, value)

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

    def trace_store_set(key, value):
        if key == wrapper._active_b_key(wrapper.rank):
            active_store_entered.set()
            assert release_active_store.wait(_TASK_18_2_TIMEOUT_S * 2)
        store_trace.append(("set", key, gate.phase, wrapper._fatal_error))
        return original_store_set(key, value)

    def trace_store_get(key):
        store_trace.append(("get", key, gate.phase, wrapper._fatal_error))
        return original_store_get(key)

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

    def trace_store_set(key, value):
        if key == wrapper._active_b_key(wrapper.rank):
            active_store_entered.set()
            assert release_active_store.wait(_TASK_18_2_TIMEOUT_S * 2)
        store_trace.append(("set", key, gate.phase, wrapper._fatal_error))
        return original_store_set(key, value)

    def trace_store_get(key):
        store_trace.append(("get", key, gate.phase, wrapper._fatal_error))
        return original_store_get(key)

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

    def observe_store_set(key, value):
        if key == wrapper._active_b_key(wrapper.rank):
            active_store_entered.set()
            assert release_active_store.wait(_TASK_18_2_TIMEOUT_S * 2)
            order.append("active_b_set")
        elif key == wrapper._fatal_ack_key(wrapper.rank):
            order.append("fatal_ack")
        return original_store_set(key, value)

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

    def pause_active_store(key, value):
        if key == wrapper._active_b_key(wrapper.rank):
            active_store_entered.set()
            assert release_active_store.wait(_TASK_18_2_TIMEOUT_S * 2)
        return original_store_set(key, value)

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
        assert close_hard_exit.wait(_TASK_18_2_TIMEOUT_S)
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
    assert isinstance(close_errors[0], FatalHardExit)
    assert callback_calls == [owner]
    assert owner.state == "quarantined"
    assert owner.error is primary
    assert owner.secondary_errors == (callback_error,)
    assert runtime._terminal_secondary_errors == (callback_error,)
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
        (
            close_name,
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

        def record_set(key, value, rank=rank, original=original_set):
            if int(value) == 1:
                control_writes[rank].append(key)
            return original(key, value)

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

    def hold_observable_direct_owner():
        if threading.current_thread().name == "task-18.2-direct-publisher":
            reservation = wrappers[1]._fatal_publication_local.reservation
            assert reservation.started is True
            assert direct_begins == [(True, False, False, 1)]
            direct_owner_reserved.set()
            assert release_direct_owner.wait(3.0)
        return original_prepare()

    monkeypatch.setattr(
        wrappers[1], "_prepare_local_fatal_locked", hold_observable_direct_owner
    )

    control_writes = [[], []]
    for rank, wrapper in enumerate(wrappers):
        original_set = wrapper._fatal_store_set

        def record_set(key, value, rank=rank, original=original_set):
            if int(value) == 1:
                control_writes[rank].append(key)
            return original(key, value)

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
        assert raw_comm.abort_calls == 1
        assert raw_comm.destroy_calls == 0
        assert backend.stop_calls == 0
    finally:
        wrapper._fatal_control_initialized = False
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
collective._local_fatal_capability_code = lambda: 0
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
    assert runtime.collective is None
    assert collective.close_calls == 1
    assert collective.live is True

    runtime.close()
    runtime.close()

    assert runtime._closed is True
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
