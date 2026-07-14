import ast
from contextlib import contextmanager
from dataclasses import replace
import threading
import traceback

import numpy as np
import pytest

from renormalizer.backend._distributed.async_owner import AsyncAllocationRecord
from renormalizer.backend._distributed.cache import DeviceTensorCache
from renormalizer.backend._distributed.pinned import PinnedBufferPool
from renormalizer.backend._distributed.terminal import (
    _FatalTransition,
    _TerminalPhase,
)
from renormalizer.backend._distributed.transfer import TransferScheduler
from renormalizer.backend.distributed_runtime import CupyDistributedRuntime
from renormalizer.backend.tests.residency.test_active_provider import (
    _LoopbackCollective,
    _active_case,
    _block_request,
    _explicit_budget,
    _provider,
    _runtime,
)


_TIMEOUT_S = 3.0


ADMISSION_SENTINEL_MATRIX = {
    "barrier_collective": {"scope": "runtime", "epoch": None},
    "execution_config_backend_sync": {
        "scope": "runtime_setup",
        "epoch": None,
    },
    "budget_probe": {"scope": "runtime_setup", "epoch": None},
    "provider_construct": {"scope": "runtime_setup", "epoch": None},
    "provider_install": {"scope": "runtime_setup", "epoch": None},
    "provider_config_match": {"scope": "runtime_setup", "epoch": None},
    "preflight_control_allocation": {
        "scope": "runtime_setup",
        "epoch": None,
    },
    "preflight_collective": {"scope": "runtime_setup", "epoch": None},
    "receipt_publish": {"scope": "runtime_setup", "epoch": None},
    "receipt_consume": {
        "scope": "construction",
        "epoch": "provisional_epoch",
        "same_sequence": "construction_token.sequence",
    },
    "construct_status": {
        "scope": "construction",
        "epoch": "provisional_epoch",
        "same_sequence": "construction_token.sequence",
    },
    "construct_store_reservation": {
        "scope": "construction",
        "epoch": "provisional_epoch",
        "same_sequence": "construction_token.sequence",
    },
    "construct_cache_reservation": {
        "scope": "construction",
        "epoch": "provisional_epoch",
        "same_sequence": "construction_token.sequence",
    },
    "construct_pool": {
        "scope": "construction",
        "epoch": "provisional_epoch",
        "same_sequence": "construction_token.sequence",
    },
    "construct_scheduler": {
        "scope": "construction",
        "epoch": "provisional_epoch",
        "same_sequence": "construction_token.sequence",
    },
    "activate_lease": {
        "scope": "construction",
        "epoch": "provisional_epoch",
        "same_sequence": "construction_token.sequence",
    },
    "rollback_partial": {
        "scope": "construction",
        "epoch": "provisional_epoch",
        "same_sequence": "construction_token.sequence",
    },
    "operator_call": {"scope": "lease", "epoch": "active_epoch"},
    "acquire": {"scope": "lease", "epoch": "active_epoch"},
    "load": {"scope": "lease", "epoch": "active_epoch"},
    "prefetch": {"scope": "lease", "epoch": "active_epoch"},
    "mark_dirty": {"scope": "lease", "epoch": "active_epoch"},
    "reap": {"scope": "lease", "epoch": "active_epoch"},
    "resource_state_between_leases": {"scope": "runtime", "epoch": None},
    "resource_state_open_lease": {
        "scope": "lease",
        "epoch": "active_epoch",
    },
    "ordinary_async_quarantine": {
        "scope": "lease",
        "epoch": "active_epoch",
        "parent_sequence": "enqueue_token.sequence",
    },
    "child_close": {
        "scope": "lease_close",
        "epoch": "closing_epoch",
        "transition_sequence": "lease_close.sequence",
    },
    "schedule_writeback": {
        "scope": "lease_close",
        "epoch": "closing_epoch",
        "transition_sequence": "lease_close.sequence",
    },
    "dirty_writeback_callback": {
        "scope": "lease_close",
        "epoch": "closing_epoch",
        "transition_sequence": "lease_close.sequence",
        "parent_sequence": "schedule_writeback_token.sequence",
    },
    "observe_peaks": {
        "scope": "lease_close",
        "epoch": "closing_epoch",
        "transition_sequence": "lease_close.sequence",
    },
    "scheduler_complete": {
        "scope": "lease_close",
        "epoch": "closing_epoch",
        "transition_sequence": "lease_close.sequence",
    },
    "cache_wait": {
        "scope": "lease_close",
        "epoch": "closing_epoch",
        "transition_sequence": "lease_close.sequence",
    },
    "pool_reap": {
        "scope": "lease_close",
        "epoch": "closing_epoch",
        "transition_sequence": "lease_close.sequence",
    },
    "cache_invalidate": {
        "scope": "lease_close",
        "epoch": "closing_epoch",
        "transition_sequence": "lease_close.sequence",
    },
    "emit_profile": {
        "scope": "lease_close",
        "epoch": "closing_epoch",
        "transition_sequence": "lease_close.sequence",
    },
    "status_close": {
        "scope": "lease_close",
        "epoch": "closing_epoch",
        "transition_sequence": "lease_close.sequence",
    },
    "scheduler_close": {
        "scope": "lease_close",
        "epoch": "closing_epoch",
        "transition_sequence": "lease_close.sequence",
    },
    "pool_close": {
        "scope": "lease_close",
        "epoch": "closing_epoch",
        "transition_sequence": "lease_close.sequence",
    },
    "cache_reservation_close": {
        "scope": "lease_close",
        "epoch": "closing_epoch",
        "transition_sequence": "lease_close.sequence",
    },
    "store_reservation_close": {
        "scope": "lease_close",
        "epoch": "closing_epoch",
        "transition_sequence": "lease_close.sequence",
    },
    "provider_close": {
        "scope": "runtime_close",
        "epoch": None,
        "transition_sequence": "runtime_close.sequence",
    },
    "collective_close": {
        "scope": "runtime_close",
        "epoch": None,
        "transition_sequence": "runtime_close.sequence",
    },
}


TRANSITION_BOUNDARIES = (
    "begin_lease_close",
    "lease_close_commit",
    "begin_runtime_close",
    "runtime_close_commit",
)


CONSTRUCTION_MATRIX_ROWS = (
    "receipt_consume",
    "construct_status",
    "construct_store_reservation",
    "construct_cache_reservation",
    "construct_pool",
    "construct_scheduler",
    "activate_lease",
)


ORDINARY_LEASE_MATRIX_ROWS = (
    "operator_call",
    "acquire",
    "load",
    "prefetch",
    "mark_dirty",
    "reap",
)


EXECUTION_CONFIG_MATRIX_ROWS = (
    "execution_config_backend_sync",
    "budget_probe",
    "provider_construct",
    "provider_install",
    "provider_config_match",
)


PREFLIGHT_MATRIX_ROWS = (
    "preflight_control_allocation",
    "preflight_collective",
    "receipt_publish",
)


LEASE_CLOSE_MATRIX_ROWS = (
    "child_close",
    "schedule_writeback",
    "dirty_writeback_callback",
    "observe_peaks",
    "scheduler_complete",
    "cache_wait",
    "pool_reap",
    "cache_invalidate",
    "emit_profile",
    "status_close",
    "scheduler_close",
    "pool_close",
    "cache_reservation_close",
    "store_reservation_close",
)


def _current_token(runtime):
    return runtime._terminal_gate._current_thread_admission()


def _start(call):
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

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, results, errors, done


def _join(thread, done):
    assert done.wait(_TIMEOUT_S)
    thread.join(_TIMEOUT_S)
    assert not thread.is_alive()


def _wait_for_phase(runtime, phase):
    gate = runtime._terminal_gate
    with gate._condition:
        assert gate._condition.wait_for(
            lambda: gate._phase is phase,
            timeout=_TIMEOUT_S,
        )


def _assert_token(token, *, scope, epoch, transition=None, parent=None):
    assert token is not None
    assert token.scope == scope
    assert token.epoch == epoch
    if transition is not None:
        assert token.transition_sequence == transition.sequence
    if parent is not None:
        assert token.parent_sequence == parent.sequence


def _open_active_case(*, store_id):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(runtime, store_id=store_id)
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    lease = provider.open_working_set(request, plan, store, receipt).__enter__()
    return runtime, request, plan, store, provider, lease


def _close_case(runtime, store):
    if not runtime._closed:
        try:
            runtime.close()
        except BaseException:
            pass
    if not store.closed:
        try:
            store.close()
        except BaseException:
            pass


def _instrument_construction_matrix(
    monkeypatch,
    runtime,
    provider,
    store,
    intercept,
    capture=lambda name, resource: None,
):
    original_consume = runtime.consume_residency_receipt
    original_status = provider._provision_status_workspace
    original_store_reserve = store.reserve
    original_cache_factory = provider._cache_factory
    original_pool_factory = provider._pool_factory
    original_scheduler_factory = provider._scheduler_factory
    original_activate = runtime._terminal_gate.activate_lease

    def consume(*args, **kwargs):
        return intercept(
            "receipt_consume",
            lambda: original_consume(*args, **kwargs),
        )

    def status(*args, **kwargs):
        resource = intercept(
            "construct_status",
            lambda: original_status(*args, **kwargs),
        )
        capture("construct_status", resource)
        return resource

    def reserve_store(*args, **kwargs):
        resource = intercept(
            "construct_store_reservation",
            lambda: original_store_reserve(*args, **kwargs),
        )
        capture("construct_store_reservation", resource)
        return resource

    def cache_factory(*args, **kwargs):
        cache = original_cache_factory(*args, **kwargs)
        original_reserve = cache.reserve

        def reserve_cache(*reserve_args, **reserve_kwargs):
            resource = intercept(
                "construct_cache_reservation",
                lambda: original_reserve(*reserve_args, **reserve_kwargs),
            )
            capture("construct_cache_reservation", resource)
            return resource

        monkeypatch.setattr(cache, "reserve", reserve_cache)
        return cache

    def pool_factory(*args, **kwargs):
        resource = intercept(
            "construct_pool",
            lambda: original_pool_factory(*args, **kwargs),
        )
        capture("construct_pool", resource)
        return resource

    def scheduler_factory(*args, **kwargs):
        resource = intercept(
            "construct_scheduler",
            lambda: original_scheduler_factory(*args, **kwargs),
        )
        capture("construct_scheduler", resource)
        return resource

    def activate(epoch, token):
        return intercept(
            "activate_lease",
            lambda: original_activate(epoch, token),
        )

    monkeypatch.setattr(runtime, "consume_residency_receipt", consume)
    monkeypatch.setattr(provider, "_provision_status_workspace", status)
    monkeypatch.setattr(store, "reserve", reserve_store)
    provider._cache_factory = cache_factory
    provider._pool_factory = pool_factory
    provider._scheduler_factory = scheduler_factory
    monkeypatch.setattr(runtime._terminal_gate, "activate_lease", activate)


def _blocked_ordinary_operation(monkeypatch, lease, row, entered, release):
    def block():
        token = _current_token(lease._provider.runtime)
        _assert_token(token, scope="lease", epoch=lease._epoch)
        entered.set()
        assert release.wait(_TIMEOUT_S)

    if row == "operator_call":
        original = lease._operator_call_admitted

        @contextmanager
        def admitted(*args, **kwargs):
            block()
            with original(*args, **kwargs) as call:
                yield call

        monkeypatch.setattr(lease, "_operator_call_admitted", admitted)

        def operation():
            with lease._operator_call(np.ones(4, dtype=np.float64)):
                pass

        return operation

    if row == "acquire":
        original = lease._acquire_admitted

        def admitted(*args, **kwargs):
            block()
            return original(*args, **kwargs)

        monkeypatch.setattr(lease, "_acquire_admitted", admitted)
        request = _block_request(lease)

        def operation():
            with lease._lease_admission("acquire") as token:
                child = lease.acquire(request, _admission_token=token)
                child.close(_admission_token=token)

        return operation

    if row == "load":
        original = lease._load_identity_admitted

        def admitted(*args, **kwargs):
            block()
            return original(*args, **kwargs)

        monkeypatch.setattr(lease, "_load_identity_admitted", admitted)
        identity = lease.cache_identities[0]

        def operation():
            with lease._lease_admission("load") as token:
                loaded = lease._load_identity(
                    identity,
                    _admission_token=token,
                )
                loaded.close()

        return operation

    if row == "prefetch":
        identity = lease.cache_identities[0]
        lease._future_queue = [identity]
        original = lease._provider.cache.contains

        def contains(*args, **kwargs):
            block()
            return False

        monkeypatch.setattr(lease._provider.cache, "contains", contains)

        def operation():
            lease._prefetch_one()
            monkeypatch.setattr(lease._provider.cache, "contains", original)

        return operation

    if row == "mark_dirty":
        original = lease._mark_dirty_admitted

        def admitted(*args, **kwargs):
            block()
            return original(*args, **kwargs)

        monkeypatch.setattr(lease, "_mark_dirty_admitted", admitted)
        output = lease.backend.zeros((4,), dtype=np.float64)
        return lambda: lease.mark_dirty("output", output)

    if row == "reap":
        original = lease._reap_completed_admitted

        def admitted(*args, **kwargs):
            block()
            return original(*args, **kwargs)

        monkeypatch.setattr(lease, "_reap_completed_admitted", admitted)
        return lease.reap_completed

    raise AssertionError("unknown ordinary lease row {!r}".format(row))


def test_lifecycle_races_use_events_not_sleep():
    source = __file__
    with open(source, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())

    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(
        (isinstance(call, ast.Name) and call.id == "sleep")
        or (isinstance(call, ast.Attribute) and call.attr == "sleep")
        for call in calls
    )


def test_admission_sentinel_matrix_covers_every_approved_boundary():
    assert set(ADMISSION_SENTINEL_MATRIX) == {
        "barrier_collective",
        "execution_config_backend_sync",
        "budget_probe",
        "provider_construct",
        "provider_install",
        "provider_config_match",
        "preflight_control_allocation",
        "preflight_collective",
        "receipt_publish",
        "receipt_consume",
        "construct_status",
        "construct_store_reservation",
        "construct_cache_reservation",
        "construct_pool",
        "construct_scheduler",
        "activate_lease",
        "rollback_partial",
        "operator_call",
        "acquire",
        "load",
        "prefetch",
        "mark_dirty",
        "reap",
        "resource_state_between_leases",
        "resource_state_open_lease",
        "ordinary_async_quarantine",
        "child_close",
        "schedule_writeback",
        "dirty_writeback_callback",
        "observe_peaks",
        "scheduler_complete",
        "cache_wait",
        "pool_reap",
        "cache_invalidate",
        "emit_profile",
        "status_close",
        "scheduler_close",
        "pool_close",
        "cache_reservation_close",
        "store_reservation_close",
        "provider_close",
        "collective_close",
    }
    assert TRANSITION_BOUNDARIES == (
        "begin_lease_close",
        "lease_close_commit",
        "begin_runtime_close",
        "runtime_close_commit",
    )


def test_barrier_collective_uses_one_runtime_admission(monkeypatch):
    runtime = _runtime()
    observed = []

    def barrier():
        observed.append(_current_token(runtime))
        return "barrier-result"

    monkeypatch.setattr(runtime.collective, "barrier", barrier, raising=False)

    assert runtime.barrier() == "barrier-result"
    assert len(observed) == 1
    _assert_token(observed[0], scope="runtime", epoch=None)
    assert _current_token(runtime) is None


@pytest.mark.parametrize("transition_kind", ("fatal", "runtime_close"))
def test_barrier_admission_drains_before_terminal_transition(
    monkeypatch, transition_kind
):
    runtime = _runtime(collective=_LoopbackCollective(1))
    entered = threading.Event()
    release = threading.Event()

    def barrier():
        _assert_token(_current_token(runtime), scope="runtime", epoch=None)
        entered.set()
        assert release.wait(_TIMEOUT_S)
        return "barrier-result"

    monkeypatch.setattr(runtime.collective, "barrier", barrier, raising=False)
    worker, results, errors, done = _start(runtime.barrier)
    assert entered.wait(_TIMEOUT_S)
    primary = RuntimeError("fatal during barrier")
    transition = (
        (lambda: runtime._enter_communicator_fatal(primary))
        if transition_kind == "fatal"
        else runtime.close
    )
    terminal, terminal_results, terminal_errors, terminal_done = _start(transition)
    if transition_kind == "fatal":
        _wait_for_phase(runtime, _TerminalPhase.FATAL_PENDING)
    else:
        _wait_for_phase(runtime, _TerminalPhase.RUNTIME_CLOSING)
    try:
        assert not terminal_done.is_set()
        release.set()
        _join(worker, done)
        _join(terminal, terminal_done)
    finally:
        release.set()
        for thread, finished in ((worker, done), (terminal, terminal_done)):
            if thread.is_alive():
                _join(thread, finished)
        if transition_kind == "fatal":
            try:
                runtime.close()
            except BaseException:
                pass

    assert errors == []
    assert results == ["barrier-result"]
    if transition_kind == "fatal":
        assert terminal_errors == []
        assert terminal_results == [primary]
    else:
        assert terminal_errors == []
        assert terminal_results == [None]


def test_execution_config_uses_one_setup_token_per_call(monkeypatch):
    from renormalizer.backend._distributed import providers as provider_module

    runtime = _runtime()
    observed = []

    def record(name):
        observed.append((name, _current_token(runtime)))

    def synchronize_backend(*, _admission_token=None):
        record("execution_config_backend_sync")
        return "numpy", "cpu", 64

    def resolve_budget(resource, requested, *, _admission_token=None):
        record("budget_probe")
        return _explicit_budget(requested, resource)

    class Provider:
        residency_policy = "active_working_set"
        provider_role = "factory"

        def __init__(self, retained_runtime, **kwargs):
            assert retained_runtime is runtime
            self.runtime = runtime
            self.kwargs = kwargs
            record("provider_construct")

        def matches_config(self, *args, **kwargs):
            record("provider_config_match")
            return True

        @staticmethod
        def acquire(request):
            raise AssertionError("factory provider cannot acquire")

        @staticmethod
        def open_working_set(*args, **kwargs):
            raise AssertionError("fake provider does not open working sets")

    def install(provider, *, _admission_token=None):
        record("provider_install")
        runtime._active_provider = provider

    monkeypatch.setattr(runtime, "_synchronize_active_backend", synchronize_backend)
    monkeypatch.setattr(runtime, "_resolve_budget", resolve_budget)
    monkeypatch.setattr(provider_module, "ActiveWorkingSetProvider", Provider)
    monkeypatch.setattr(runtime, "_install_active_provider", install, raising=False)

    first = runtime.execution_config(
        residency_policy="active_working_set",
        device_memory_budget_bytes=1024,
        host_memory_budget_bytes=2048,
    )
    second = runtime.execution_config(
        residency_policy="active_working_set",
        device_memory_budget_bytes=1024,
        host_memory_budget_bytes=2048,
    )

    assert first.provider is second.provider
    expected = {
        "execution_config_backend_sync",
        "budget_probe",
        "provider_construct",
        "provider_install",
        "provider_config_match",
    }
    assert {name for name, _ in observed} == expected
    for name, token in observed:
        _assert_token(token, scope="runtime_setup", epoch=None)
    first_sequence = observed[0][1].sequence
    install_index = next(
        index for index, (name, _) in enumerate(observed) if name == "provider_install"
    )
    first_call = [token for _, token in observed[: install_index + 1]]
    assert all(token.sequence == first_sequence for token in first_call)
    assert observed[-1][1].sequence > first_sequence
    assert _current_token(runtime) is None


@pytest.mark.parametrize("row", EXECUTION_CONFIG_MATRIX_ROWS)
@pytest.mark.parametrize("transition_kind", ("fatal", "runtime_close"))
def test_execution_config_setup_row_drains_before_terminal_transition(
    monkeypatch, row, transition_kind
):
    from renormalizer.backend._distributed import providers as provider_module

    runtime = _runtime()
    entered = threading.Event()
    release = threading.Event()

    def block(name):
        token = _current_token(runtime)
        _assert_token(token, scope="runtime_setup", epoch=None)
        if name == row:
            entered.set()
            assert release.wait(_TIMEOUT_S)

    def synchronize_backend():
        block("execution_config_backend_sync")
        return "numpy", "cpu", 64

    def resolve_budget(resource, requested, *, _admission_token=None):
        block("budget_probe")
        return _explicit_budget(requested, resource)

    class Provider:
        residency_policy = "active_working_set"
        provider_role = "factory"
        _active_lease = None
        _terminal_error = None

        def __init__(self, retained_runtime, **kwargs):
            assert retained_runtime is runtime
            block("provider_construct")
            self.runtime = retained_runtime
            self.closed = False

        def matches_config(self, *args, **kwargs):
            block("provider_config_match")
            return True

        @staticmethod
        def acquire(request):
            raise AssertionError("factory provider cannot acquire")

        @staticmethod
        def open_working_set(*args, **kwargs):
            raise AssertionError("fake provider does not open working sets")

        def _retain_transition_resources(self, lease):
            assert lease is None

        def _finalize_terminal_runtime_close(self, error):
            self._terminal_error = error

        def close(self):
            self.closed = True

    original_install = runtime._install_active_provider

    def install(provider, *, _admission_token):
        block("provider_install")
        return original_install(provider, _admission_token=_admission_token)

    monkeypatch.setattr(runtime, "_synchronize_active_backend", synchronize_backend)
    monkeypatch.setattr(runtime, "_resolve_budget", resolve_budget)
    monkeypatch.setattr(provider_module, "ActiveWorkingSetProvider", Provider)
    monkeypatch.setattr(runtime, "_install_active_provider", install)
    configure = lambda: runtime.execution_config(
        residency_policy="active_working_set",
        device_memory_budget_bytes=1024,
        host_memory_budget_bytes=2048,
    )
    if row == "provider_config_match":
        configure()

    worker, results, errors, done = _start(configure)
    assert entered.wait(_TIMEOUT_S)
    primary = RuntimeError("fatal during {}".format(row))
    transition = (
        (lambda: runtime._enter_communicator_fatal(primary))
        if transition_kind == "fatal"
        else runtime.close
    )
    terminal, terminal_results, terminal_errors, terminal_done = _start(transition)
    if transition_kind == "fatal":
        _wait_for_phase(runtime, _TerminalPhase.FATAL_PENDING)
    else:
        _wait_for_phase(runtime, _TerminalPhase.RUNTIME_CLOSING)
    try:
        assert not terminal_done.is_set()
        release.set()
        _join(worker, done)
        _join(terminal, terminal_done)
    finally:
        release.set()
        for thread, finished in ((worker, done), (terminal, terminal_done)):
            if thread.is_alive():
                _join(thread, finished)
        if transition_kind == "fatal" and not runtime._closed:
            try:
                runtime.close()
            except BaseException:
                pass

    assert errors == []
    assert len(results) == 1
    provider = results[0].provider
    if transition_kind == "fatal":
        assert terminal_errors == []
        assert terminal_results == [primary]
        assert provider._terminal_error is primary
    else:
        assert terminal_errors == []
        assert terminal_results == [None]
        assert provider.closed is True


def test_preflight_control_collective_and_receipt_share_setup_token(monkeypatch):
    collective = _LoopbackCollective(2)
    runtime = _runtime(world_size=2, rank=0, collective=collective)
    request, plan, store, _, _ = _active_case(
        runtime, store_id="terminal-setup-preflight"
    )
    observed = []
    original_control = runtime._control_array
    original_allreduce = collective.allreduce

    def control(*args, **kwargs):
        observed.append(("preflight_control_allocation", _current_token(runtime)))
        return original_control(*args, **kwargs)

    def allreduce(*args, **kwargs):
        observed.append(("preflight_collective", _current_token(runtime)))
        return original_allreduce(*args, **kwargs)

    def publish(receipt, *, _admission_token=None):
        observed.append(("receipt_publish", _current_token(runtime)))
        runtime._issued_receipts.clear()
        runtime._issued_receipts[id(receipt)] = receipt
        return receipt

    monkeypatch.setattr(runtime, "_control_array", control)
    monkeypatch.setattr(collective, "allreduce", allreduce)
    monkeypatch.setattr(runtime, "_publish_residency_receipt", publish, raising=False)
    try:
        receipt = runtime.preflight_residency(request, plan)
        assert runtime._issued_receipts[id(receipt)] is receipt
        assert {name for name, _ in observed} == {
            "preflight_control_allocation",
            "preflight_collective",
            "receipt_publish",
        }
        sequence = observed[0][1].sequence
        for _, token in observed:
            _assert_token(token, scope="runtime_setup", epoch=None)
            assert token.sequence == sequence
    finally:
        _close_case(runtime, store)


@pytest.mark.parametrize("row", PREFLIGHT_MATRIX_ROWS)
@pytest.mark.parametrize("transition_kind", ("fatal", "runtime_close"))
def test_preflight_setup_row_drains_before_terminal_transition(
    monkeypatch, row, transition_kind
):
    collective = _LoopbackCollective(2)
    runtime = _runtime(world_size=2, rank=0, collective=collective)
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-preflight-{}-{}".format(transition_kind, row),
    )
    entered = threading.Event()
    release = threading.Event()
    original_control = runtime._control_array
    original_allreduce = collective.allreduce
    original_publish = runtime._publish_residency_receipt

    def block(name):
        token = _current_token(runtime)
        _assert_token(token, scope="runtime_setup", epoch=None)
        if name == row:
            entered.set()
            assert release.wait(_TIMEOUT_S)

    def control(*args, **kwargs):
        block("preflight_control_allocation")
        return original_control(*args, **kwargs)

    def allreduce(*args, **kwargs):
        block("preflight_collective")
        return original_allreduce(*args, **kwargs)

    def publish(receipt, *, _admission_token):
        block("receipt_publish")
        return original_publish(receipt, _admission_token=_admission_token)

    monkeypatch.setattr(runtime, "_control_array", control)
    monkeypatch.setattr(collective, "allreduce", allreduce)
    monkeypatch.setattr(runtime, "_publish_residency_receipt", publish)
    worker, results, errors, done = _start(
        lambda: runtime.preflight_residency(request, plan)
    )
    assert entered.wait(_TIMEOUT_S)
    primary = RuntimeError("fatal during {}".format(row))
    transition = (
        (lambda: runtime._enter_communicator_fatal(primary))
        if transition_kind == "fatal"
        else runtime.close
    )
    terminal, terminal_results, terminal_errors, terminal_done = _start(transition)
    if transition_kind == "fatal":
        _wait_for_phase(runtime, _TerminalPhase.FATAL_PENDING)
    else:
        _wait_for_phase(runtime, _TerminalPhase.RUNTIME_CLOSING)
    try:
        assert not terminal_done.is_set()
        release.set()
        _join(worker, done)
        _join(terminal, terminal_done)
    finally:
        release.set()
        for thread, finished in ((worker, done), (terminal, terminal_done)):
            if thread.is_alive():
                _join(thread, finished)
        if transition_kind == "fatal" and not runtime._closed:
            try:
                runtime.close()
            except BaseException:
                pass
        if not store.closed:
            store.close()

    assert errors == []
    assert len(results) == 1
    assert runtime._issued_receipts == {}
    if transition_kind == "fatal":
        assert terminal_errors == []
        assert terminal_results == [primary]
    else:
        assert terminal_errors == []
        assert terminal_results == [None]


def test_fatal_imports_provider_installed_by_draining_setup(monkeypatch):
    runtime = _runtime()
    monkeypatch.setattr(
        runtime,
        "_synchronize_active_backend",
        lambda: ("numpy", "cpu", 64),
    )
    install_entered = threading.Event()
    release_install = threading.Event()
    original_install = runtime._install_active_provider

    def install(provider, *, _admission_token):
        install_entered.set()
        assert release_install.wait(_TIMEOUT_S)
        return original_install(provider, _admission_token=_admission_token)

    monkeypatch.setattr(runtime, "_install_active_provider", install)
    setup, setup_results, setup_errors, setup_done = _start(
        lambda: runtime.execution_config(
            residency_policy="active_working_set",
            device_memory_budget_bytes=1024,
            host_memory_budget_bytes=2048,
        )
    )
    assert install_entered.wait(_TIMEOUT_S)
    primary = RuntimeError("fatal during provider installation")
    fatal, fatal_results, fatal_errors, fatal_done = _start(
        lambda: runtime._enter_communicator_fatal(primary)
    )
    _wait_for_phase(runtime, _TerminalPhase.FATAL_PENDING)
    try:
        release_install.set()
        _join(setup, setup_done)
        _join(fatal, fatal_done)
    finally:
        release_install.set()
        for thread, done in ((setup, setup_done), (fatal, fatal_done)):
            if thread.is_alive():
                _join(thread, done)

    assert setup_errors == []
    assert fatal_errors == []
    assert fatal_results == [primary]
    provider = setup_results[0].provider
    assert runtime._active_provider is provider
    assert provider._terminal_error is primary
    assert runtime._pending_fatal_context[:2] == (provider, None)


def test_fatal_clears_receipt_published_by_draining_setup(monkeypatch):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime, store_id="terminal-fatal-draining-receipt"
    )
    publish_entered = threading.Event()
    release_publish = threading.Event()
    original_publish = runtime._publish_residency_receipt

    def publish(receipt, *, _admission_token):
        publish_entered.set()
        assert release_publish.wait(_TIMEOUT_S)
        return original_publish(receipt, _admission_token=_admission_token)

    monkeypatch.setattr(runtime, "_publish_residency_receipt", publish)
    setup, setup_results, setup_errors, setup_done = _start(
        lambda: runtime.preflight_residency(request, plan)
    )
    assert publish_entered.wait(_TIMEOUT_S)
    primary = RuntimeError("fatal during receipt publication")
    fatal, fatal_results, fatal_errors, fatal_done = _start(
        lambda: runtime._enter_communicator_fatal(primary)
    )
    _wait_for_phase(runtime, _TerminalPhase.FATAL_PENDING)
    try:
        release_publish.set()
        _join(setup, setup_done)
        _join(fatal, fatal_done)
    finally:
        release_publish.set()
        for thread, done in ((setup, setup_done), (fatal, fatal_done)):
            if thread.is_alive():
                _join(thread, done)
        if not store.closed:
            store.close()

    assert setup_errors == []
    assert fatal_errors == []
    assert fatal_results == [primary]
    assert setup_results
    assert runtime._issued_receipts == {}


@pytest.mark.parametrize("phase", ("constructing", "open", "closing"))
@pytest.mark.parametrize("operation", ("execution_config", "preflight"))
def test_runtime_setup_rejects_live_lease_before_first_sentinel(
    monkeypatch, phase, operation
):
    runtime = _runtime()
    gate = runtime._terminal_gate
    epoch, construction = gate.begin_lease("test_setup_rejection")
    if phase in {"open", "closing"}:
        gate.activate_lease(epoch, construction)
        gate.release(construction)
    if phase == "closing":
        gate.begin_lease_close(epoch)
    sentinel_calls = []

    def sentinel(*args, **kwargs):
        sentinel_calls.append(True)
        raise AssertionError("runtime setup reached a resource sentinel")

    monkeypatch.setattr(runtime, "_synchronize_active_backend", sentinel)
    monkeypatch.setattr(runtime, "_control_array", sentinel)
    monkeypatch.setattr(runtime.collective, "allreduce", sentinel, raising=False)

    with pytest.raises(RuntimeError):
        if operation == "execution_config":
            runtime.execution_config()
        else:
            runtime.preflight_residency(object(), object())

    assert sentinel_calls == []


def test_provisional_epoch_and_resource_record_precede_first_allocation(monkeypatch):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime, store_id="terminal-provisional-before-allocation"
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    observed = {}

    class StopBeforeAllocation(RuntimeError):
        pass

    def stop_before_allocation(*, _admission_token=None):
        observed["token"] = _current_token(runtime)
        observed["epoch"] = runtime._terminal_gate._live_epoch
        observed["record"] = getattr(provider, "_provisional_resources", None)
        raise StopBeforeAllocation("first allocation sentinel")

    monkeypatch.setattr(provider, "_provision_status_workspace", stop_before_allocation)
    try:
        with pytest.raises(StopBeforeAllocation, match="first allocation sentinel"):
            provider.open_working_set(request, plan, store, receipt)

        token = observed["token"]
        _assert_token(token, scope="construction", epoch=observed["epoch"])
        assert observed["record"] is not None
        assert observed["record"].epoch == observed["epoch"]
        assert runtime._terminal_gate._live_epoch is None
        assert provider._active_lease is None
    finally:
        _close_case(runtime, store)


def test_complete_construction_reuses_exact_provisional_token(monkeypatch):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime, store_id="terminal-construction-token"
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    observed = []

    def record(name):
        observed.append((name, _current_token(runtime)))

    original_consume = runtime.consume_residency_receipt
    original_status = provider._provision_status_workspace
    original_store_reserve = store.reserve
    original_cache_factory = provider._cache_factory
    original_pool_factory = provider._pool_factory
    original_scheduler_factory = provider._scheduler_factory
    original_activate = runtime._terminal_gate.activate_lease

    def consume(*args, **kwargs):
        record("receipt_consume")
        return original_consume(*args, **kwargs)

    def status(*args, **kwargs):
        record("construct_status")
        return original_status(*args, **kwargs)

    def reserve_store(*args, **kwargs):
        record("construct_store_reservation")
        return original_store_reserve(*args, **kwargs)

    def cache_factory(*args, **kwargs):
        cache = original_cache_factory(*args, **kwargs)
        original_reserve = cache.reserve

        def reserve_cache(*reserve_args, **reserve_kwargs):
            record("construct_cache_reservation")
            return original_reserve(*reserve_args, **reserve_kwargs)

        monkeypatch.setattr(cache, "reserve", reserve_cache)
        return cache

    def pool_factory(*args, **kwargs):
        record("construct_pool")
        return original_pool_factory(*args, **kwargs)

    def scheduler_factory(*args, **kwargs):
        record("construct_scheduler")
        return original_scheduler_factory(*args, **kwargs)

    def activate(epoch, token):
        record("activate_lease")
        assert token is _current_token(runtime)
        return original_activate(epoch, token)

    monkeypatch.setattr(runtime, "consume_residency_receipt", consume)
    monkeypatch.setattr(provider, "_provision_status_workspace", status)
    monkeypatch.setattr(store, "reserve", reserve_store)
    provider._cache_factory = cache_factory
    provider._pool_factory = pool_factory
    provider._scheduler_factory = scheduler_factory
    monkeypatch.setattr(runtime._terminal_gate, "activate_lease", activate)

    lease = None
    try:
        lease = provider.open_working_set(request, plan, store, receipt).__enter__()
        expected = {
            "receipt_consume",
            "construct_status",
            "construct_store_reservation",
            "construct_cache_reservation",
            "construct_pool",
            "construct_scheduler",
            "activate_lease",
        }
        assert {name for name, _ in observed} == expected
        construction = observed[0][1]
        for _, token in observed:
            assert token is construction
            _assert_token(token, scope="construction", epoch=lease._epoch)
        assert provider._active_lease is lease
        assert runtime._terminal_gate._leases[lease._epoch].phase == "open"
    finally:
        if lease is not None:
            try:
                lease.close()
            except BaseException:
                pass
        _close_case(runtime, store)


@pytest.mark.parametrize("blocked_row", CONSTRUCTION_MATRIX_ROWS)
def test_fatal_stops_construction_after_admitted_matrix_row(
    monkeypatch, blocked_row
):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime, store_id="terminal-fatal-construction-{}".format(blocked_row)
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    row_entered = threading.Event()
    release_row = threading.Event()
    visited = []
    close_calls = []

    def capture(name, resource):
        original_close = resource.close

        def close(*args, **kwargs):
            close_calls.append((name, _current_token(runtime)))
            return original_close(*args, **kwargs)

        monkeypatch.setattr(resource, "close", close)

    def intercept(name, callback):
        visited.append(name)
        if name == blocked_row:
            row_entered.set()
            assert release_row.wait(_TIMEOUT_S)
        return callback()

    _instrument_construction_matrix(
        monkeypatch,
        runtime,
        provider,
        store,
        intercept,
        capture,
    )
    opener, open_results, open_errors, open_done = _start(
        lambda: provider.open_working_set(request, plan, store, receipt)
    )
    assert row_entered.wait(_TIMEOUT_S)
    record = provider._provisional_resources
    primary = RuntimeError("fatal during {}".format(blocked_row))
    fatal, fatal_results, fatal_errors, fatal_done = _start(
        lambda: runtime._enter_communicator_fatal(primary)
    )
    _wait_for_phase(runtime, _TerminalPhase.FATAL_PENDING)
    try:
        release_row.set()
        _join(opener, open_done)
        _join(fatal, fatal_done)

        boundary = CONSTRUCTION_MATRIX_ROWS.index(blocked_row) + 1
        assert visited == list(CONSTRUCTION_MATRIX_ROWS[:boundary])
        assert open_results == []
        assert open_errors == [primary]
        assert fatal_results == [primary]
        assert fatal_errors == []
        assert close_calls == []
        assert record is not None
        assert all(
            any(retained is resource for retained in provider._terminal_resources)
            for resource in record.resources
        )
        assert all(
            any(
                retained is resource
                for retained in runtime._terminal_quarantine._resources
            )
            for resource in record.resources
        )
    finally:
        release_row.set()
        for thread, done in ((opener, open_done), (fatal, fatal_done)):
            if thread.is_alive():
                _join(thread, done)
        _close_case(runtime, store)


@pytest.mark.parametrize("blocked_row", CONSTRUCTION_MATRIX_ROWS)
def test_runtime_close_stops_construction_and_joins_rollback(
    monkeypatch, blocked_row
):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime, store_id="terminal-runtime-close-construction-{}".format(blocked_row)
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    row_entered = threading.Event()
    release_row = threading.Event()
    visited = []
    close_tokens = []

    def capture(name, resource):
        original_close = resource.close

        def close(*args, **kwargs):
            close_tokens.append((name, _current_token(runtime)))
            return original_close(*args, **kwargs)

        monkeypatch.setattr(resource, "close", close)

    def intercept(name, callback):
        visited.append(name)
        if name == blocked_row:
            row_entered.set()
            assert release_row.wait(_TIMEOUT_S)
        return callback()

    _instrument_construction_matrix(
        monkeypatch,
        runtime,
        provider,
        store,
        intercept,
        capture,
    )
    opener, open_results, open_errors, open_done = _start(
        lambda: provider.open_working_set(request, plan, store, receipt)
    )
    assert row_entered.wait(_TIMEOUT_S)
    record = provider._provisional_resources
    closer, close_results, close_errors, close_done = _start(runtime.close)
    _wait_for_phase(runtime, _TerminalPhase.RUNTIME_CLOSING)
    try:
        assert not close_done.is_set()
        release_row.set()
        _join(opener, open_done)
        _join(closer, close_done)

        boundary = CONSTRUCTION_MATRIX_ROWS.index(blocked_row) + 1
        assert visited == list(CONSTRUCTION_MATRIX_ROWS[:boundary])
        assert open_results == []
        assert len(open_errors) == 1
        assert "runtime is closing" in str(open_errors[0])
        assert close_results == [None]
        assert close_errors == []
        assert record is not None
        assert all(
            token.scope == "construction"
            and token.epoch == record.epoch
            for _, token in close_tokens
        )
        assert provider._active_lease is None
        assert runtime._closed is True
    finally:
        release_row.set()
        for thread, done in ((opener, open_done), (closer, close_done)):
            if thread.is_alive():
                _join(thread, done)
        _close_case(runtime, store)


@pytest.mark.parametrize("failed_row", CONSTRUCTION_MATRIX_ROWS[1:])
def test_constructor_failure_rolls_back_under_exact_construction_token(
    monkeypatch, failed_row
):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime, store_id="terminal-rollback-{}".format(failed_row)
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    rollback_tokens = []
    close_tokens = []
    created_rows = []

    class InjectedConstructionFailure(RuntimeError):
        pass

    def capture(name, resource):
        created_rows.append(name)
        original_close = resource.close

        def close(*args, **kwargs):
            close_tokens.append((name, _current_token(runtime)))
            return original_close(*args, **kwargs)

        monkeypatch.setattr(resource, "close", close)

    def intercept(name, callback):
        if name == failed_row:
            if name == "activate_lease":
                callback()
            raise InjectedConstructionFailure(name)
        return callback()

    original_rollback = provider._rollback_partial

    def rollback(token, *args, **kwargs):
        rollback_tokens.append((token, _current_token(runtime)))
        return original_rollback(token, *args, **kwargs)

    monkeypatch.setattr(provider, "_rollback_partial", rollback)
    _instrument_construction_matrix(
        monkeypatch,
        runtime,
        provider,
        store,
        intercept,
        capture,
    )
    try:
        with pytest.raises(InjectedConstructionFailure, match=failed_row):
            provider.open_working_set(request, plan, store, receipt)

        assert len(rollback_tokens) == 1
        construction, current = rollback_tokens[0]
        assert construction is current
        _assert_token(
            construction,
            scope="construction",
            epoch=construction.epoch,
        )
        assert {name for name, _ in close_tokens} == set(created_rows)
        assert all(token is construction for _, token in close_tokens)
        assert provider._active_lease is None
        assert provider._provisional_resources is None
        assert runtime._terminal_gate._live_epoch is None
    finally:
        _close_case(runtime, store)


def test_live_operations_and_state_use_exact_epoch_admissions(monkeypatch):
    runtime, request, plan, store, provider, lease = _open_active_case(
        store_id="terminal-live-admissions"
    )
    observed = []

    def record(name):
        token = _current_token(runtime)
        observed.append((name, token))
        return token

    original_cache_acquire = provider.cache.acquire
    cache_operation = ["acquire"]

    def cache_acquire(*args, **kwargs):
        record(cache_operation[0])
        return original_cache_acquire(*args, **kwargs)

    monkeypatch.setattr(provider.cache, "acquire", cache_acquire)
    try:
        with lease._operator_call(np.ones(4, dtype=np.float64)):
            record("operator_call")

        child = lease.acquire(_block_request(lease))
        child.close()

        identity = lease.cache_identities[0]
        cache_operation[0] = "load"
        loaded = lease._load_identity(identity)
        loaded.close()

        original_contains = provider.cache.contains
        monkeypatch.setattr(provider.cache, "contains", lambda identity: False)
        monkeypatch.setattr(
            lease,
            "_load_identity",
            lambda *args, **kwargs: record("prefetch"),
        )
        lease._future_queue = [identity]
        lease._prefetch_one()
        monkeypatch.setattr(provider.cache, "contains", original_contains)

        output = runtime.backend.zeros((4,), dtype=np.float64)
        from renormalizer.backend._distributed import providers as provider_module

        original_validate_dirty = provider_module._validate_dirty_array

        def validate_dirty(*args, **kwargs):
            record("mark_dirty")
            return original_validate_dirty(*args, **kwargs)

        monkeypatch.setattr(provider_module, "_validate_dirty_array", validate_dirty)
        lease.mark_dirty("output", output)
        original_reap = lease.scheduler.reap_completed

        def reap():
            record("reap")
            return original_reap()

        monkeypatch.setattr(lease.scheduler, "reap_completed", reap)
        lease.reap_completed()

        runtime_state_impl = provider.runtime_resource_state

        def open_state():
            record("resource_state_open_lease")
            return runtime_state_impl()

        monkeypatch.setattr(provider, "runtime_resource_state", open_state)
        runtime.resource_state()

        for name, token in observed:
            _assert_token(token, scope="lease", epoch=lease._epoch)
        assert len({token.sequence for _, token in observed}) >= 6
    finally:
        try:
            lease.close()
        except BaseException:
            pass

    between = []
    def between_state():
        between.append(_current_token(runtime))
        return runtime_state_impl()

    monkeypatch.setattr(provider, "runtime_resource_state", between_state)
    try:
        runtime.resource_state()
        assert len(between) == 1
        _assert_token(between[0], scope="runtime", epoch=None)
    finally:
        _close_case(runtime, store)


@pytest.mark.parametrize(
    ("lease_open", "transition_kind"),
    (
        (True, "lease_close"),
        (True, "runtime_close"),
        (True, "fatal"),
        (False, "runtime_close"),
        (False, "fatal"),
    ),
)
def test_admitted_resource_state_drains_before_terminal_transition(
    monkeypatch, lease_open, transition_kind
):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-state-{}-{}".format(lease_open, transition_kind)
    )
    if not lease_open:
        lease.close()
    entered = threading.Event()
    release = threading.Event()
    original_state = provider.runtime_resource_state

    def state():
        token = _current_token(runtime)
        _assert_token(
            token,
            scope="lease" if lease_open else "runtime",
            epoch=lease._epoch if lease_open else None,
        )
        entered.set()
        assert release.wait(_TIMEOUT_S)
        return original_state()

    monkeypatch.setattr(provider, "runtime_resource_state", state)
    reader, state_results, state_errors, state_done = _start(runtime.resource_state)
    assert entered.wait(_TIMEOUT_S)
    primary = RuntimeError("fatal during resource state")
    transition = {
        "lease_close": lease.close,
        "runtime_close": runtime.close,
        "fatal": lambda: runtime._enter_communicator_fatal(primary),
    }[transition_kind]
    terminal, terminal_results, terminal_errors, terminal_done = _start(transition)
    if transition_kind == "lease_close":
        gate = runtime._terminal_gate
        with gate._condition:
            assert gate._condition.wait_for(
                lambda: gate._leases[lease._epoch].phase == "closing",
                timeout=_TIMEOUT_S,
            )
    elif transition_kind == "runtime_close":
        _wait_for_phase(runtime, _TerminalPhase.RUNTIME_CLOSING)
    else:
        _wait_for_phase(runtime, _TerminalPhase.FATAL_PENDING)
    try:
        assert not terminal_done.is_set()
        release.set()
        _join(reader, state_done)
        _join(terminal, terminal_done)

        assert state_errors == []
        assert len(state_results) == 1
        assert state_results[0]["active_leases"] == int(lease_open)
        if transition_kind == "fatal":
            assert terminal_results == [primary]
            assert terminal_errors == []
        else:
            assert terminal_results == [None]
            assert terminal_errors == []
    finally:
        release.set()
        for thread, done in ((reader, state_done), (terminal, terminal_done)):
            if thread.is_alive():
                _join(thread, done)
        _close_case(runtime, store)


@pytest.mark.parametrize("row", ORDINARY_LEASE_MATRIX_ROWS)
@pytest.mark.parametrize("close_kind", ("lease_close", "runtime_close"))
def test_healthy_close_stops_admitted_ordinary_row_before_mutation(
    monkeypatch, row, close_kind
):
    runtime, _, _, store, _, lease = _open_active_case(
        store_id="terminal-ordinary-{}-{}".format(close_kind, row)
    )
    entered = threading.Event()
    release = threading.Event()
    operation = _blocked_ordinary_operation(
        monkeypatch,
        lease,
        row,
        entered,
        release,
    )
    worker, operation_results, operation_errors, operation_done = _start(operation)
    assert entered.wait(_TIMEOUT_S)
    close = lease.close if close_kind == "lease_close" else runtime.close
    closer, close_results, close_errors, close_done = _start(close)
    gate = runtime._terminal_gate
    with gate._condition:
        if close_kind == "lease_close":
            assert gate._condition.wait_for(
                lambda: gate._leases[lease._epoch].phase == "closing",
                timeout=_TIMEOUT_S,
            )
        else:
            assert gate._condition.wait_for(
                lambda: gate._phase is _TerminalPhase.RUNTIME_CLOSING,
                timeout=_TIMEOUT_S,
            )
    try:
        assert not close_done.is_set()
        release.set()
        _join(worker, operation_done)
        _join(closer, close_done)
    finally:
        release.set()
        for thread, done in ((worker, operation_done), (closer, close_done)):
            if thread.is_alive():
                _join(thread, done)
        _close_case(runtime, store)

    assert operation_results == []
    assert len(operation_errors) == 1
    assert "closing" in str(operation_errors[0])
    assert close_errors == []
    assert close_results == [None]


@pytest.mark.parametrize("row", ORDINARY_LEASE_MATRIX_ROWS)
def test_fatal_stops_admitted_ordinary_row_with_same_primary(monkeypatch, row):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-fatal-ordinary-{}".format(row)
    )
    entered = threading.Event()
    release = threading.Event()
    operation = _blocked_ordinary_operation(
        monkeypatch,
        lease,
        row,
        entered,
        release,
    )
    baseline = (
        len(lease._children),
        lease._dirty,
        provider.cache.allocated_bytes,
        lease.pool.allocated_bytes,
        len(lease.scheduler._owners),
    )
    worker, operation_results, operation_errors, operation_done = _start(operation)
    assert entered.wait(_TIMEOUT_S)
    primary = RuntimeError("fatal during ordinary {}".format(row))
    fatal, fatal_results, fatal_errors, fatal_done = _start(
        lambda: runtime._enter_communicator_fatal(primary)
    )
    _wait_for_phase(runtime, _TerminalPhase.FATAL_PENDING)
    try:
        release.set()
        _join(worker, operation_done)
        _join(fatal, fatal_done)

        assert operation_results == []
        assert operation_errors == [primary]
        assert fatal_results == [primary]
        assert fatal_errors == []
        assert provider._active_lease is lease
        assert lease._closed is False
        assert lease._provider is provider
        assert (
            len(lease._children),
            lease._dirty,
            provider.cache.allocated_bytes,
            lease.pool.allocated_bytes,
            len(lease.scheduler._owners),
        ) == baseline
    finally:
        release.set()
        for thread, done in ((worker, operation_done), (fatal, fatal_done)):
            if thread.is_alive():
                _join(thread, done)
        _close_case(runtime, store)


class _BlockingEvent:
    def __init__(self):
        self.recorded = threading.Event()
        self.waiting = threading.Event()
        self.release = threading.Event()

    def record(self, stream=None):
        self.recorded.set()

    def query(self):
        return False

    def synchronize(self):
        self.waiting.set()
        assert self.release.wait(_TIMEOUT_S)


def test_async_descendant_is_counted_from_enqueue_through_callback(monkeypatch):
    runtime, _, _, store, _, lease = _open_active_case(
        store_id="terminal-counted-async"
    )
    event = _BlockingEvent()
    child = lease.acquire(_block_request(lease))
    lease.scheduler._event_factory = lambda: event
    child.close()
    assert event.recorded.wait(_TIMEOUT_S)

    gate = runtime._terminal_gate
    with gate._condition:
        descendants = [
            state.token
            for state in gate._tokens.values()
            if state.status == "active" and state.token.parent_sequence is not None
        ]
    assert len(descendants) == 1
    descendant = descendants[0]
    _assert_token(descendant, scope="lease", epoch=lease._epoch)

    closer, _, close_errors, close_done = _start(lease.close)
    try:
        assert event.waiting.wait(_TIMEOUT_S)
        assert not close_done.is_set()
        event.release.set()
        _join(closer, close_done)
    finally:
        event.release.set()
        if closer.is_alive():
            _join(closer, close_done)
        _close_case(runtime, store)
    assert close_errors == []


def test_runtime_close_starts_preexisting_counted_descendant_before_gate_drain():
    runtime, _, _, store, _, lease = _open_active_case(
        store_id="terminal-runtime-close-counted-async"
    )
    event = _BlockingEvent()
    child = lease.acquire(_block_request(lease))
    scheduler = lease.scheduler
    scheduler._event_factory = lambda: event
    child.close()
    assert event.recorded.wait(_TIMEOUT_S)

    owner = next(iter(scheduler._owners.values()))
    gate = runtime._terminal_gate
    with gate._condition:
        descendants = [
            state.token
            for state in gate._tokens.values()
            if state.status == "active" and state.token.parent_sequence is not None
        ]
    assert len(descendants) == 1
    descendant = descendants[0]
    _assert_token(descendant, scope="lease", epoch=lease._epoch)
    assert descendant.operation == "child_close"
    assert owner.kind == "compute"
    assert owner._async_admission.token is descendant
    assert not owner._async_requested.is_set()

    closer, _, close_errors, close_done = _start(runtime.close)
    try:
        _wait_for_phase(runtime, _TerminalPhase.RUNTIME_CLOSING)
        with gate._condition:
            assert gate._tokens[descendant.sequence].status == "active"
        assert not close_done.is_set()
        assert event.waiting.wait(_TIMEOUT_S)
        event.release.set()
        _join(closer, close_done)
        assert close_errors == []
    finally:
        scheduler._start_counted_completions()
        event.release.set()
        if closer.is_alive():
            _join(closer, close_done)
        _close_case(runtime, store)


def test_runtime_close_keeps_scheduler_snapshot_if_lease_closes_before_admit(
    monkeypatch,
):
    runtime, _, _, store, _, lease = _open_active_case(
        store_id="terminal-runtime-close-stale-lease"
    )
    gate = runtime._terminal_gate
    original_admit = gate.admit_runtime
    intercepted = []

    def admit(operation):
        intercepted.append(operation)
        lease.close()
        assert lease.scheduler is None
        return original_admit(operation)

    monkeypatch.setattr(gate, "admit_runtime", admit)
    try:
        runtime.close()
        assert intercepted == ["begin_runtime_close"]
        assert runtime._closed is True
    finally:
        _close_case(runtime, store)


def test_completed_event_before_lease_close_releases_counted_descendant():
    runtime, _, _, store, _, lease = _open_active_case(
        store_id="terminal-precompleted-async"
    )
    event = _BlockingEvent()
    child = lease.acquire(_block_request(lease))
    lease.scheduler._event_factory = lambda: event
    child.close()
    assert event.recorded.wait(_TIMEOUT_S)

    owner = next(iter(lease.scheduler._owners.values()))
    gate = runtime._terminal_gate
    with gate._condition:
        descendants = [
            state.token
            for state in gate._tokens.values()
            if state.status == "active" and state.token.parent_sequence is not None
        ]
    assert len(descendants) == 1
    _assert_token(descendants[0], scope="lease", epoch=lease._epoch)

    event.release.set()
    closer, _, close_errors, close_done = _start(lease.close)
    try:
        _join(closer, close_done)
        assert close_errors == []
        assert owner.state == "detached"
        assert owner._async_done.is_set()
        assert not owner._async_worker.is_alive()
        with gate._condition:
            assert not any(
                state.status == "active" for state in gate._tokens.values()
            )
    finally:
        event.release.set()
        _close_case(runtime, store)


def test_fatal_drains_h2d_descendant_before_readiness_publication(monkeypatch):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-deferred-h2d-fatal"
    )
    event = _BlockingEvent()
    lease.scheduler._event_factory = lambda: event
    output_ref = dict(lease.request.host_refs)["output"]
    destination = lease.backend.empty(
        output_ref.shape,
        dtype=np.dtype(output_ref.dtype),
        order="C",
    )
    with lease._lease_admission("prefetch"):
        with lease.pool.checkout(destination.nbytes) as slot:
            ticket = lease.scheduler.stage_h2d(output_ref, destination, slot)
    assert event.recorded.wait(_TIMEOUT_S)

    owner = ticket._owner
    assert owner._async_completion_deferred is True
    assert owner._async_worker is not None
    touched = []

    def forbidden(operation):
        def fail():
            touched.append(operation)
            raise AssertionError("fatal drain touched deferred H2D event")

        return fail

    monkeypatch.setattr(event, "query", forbidden("query"))
    monkeypatch.setattr(event, "synchronize", forbidden("synchronize"))
    primary = RuntimeError("fatal before H2D readiness publication")
    publisher, results, errors, published = _start(
        lambda: runtime._enter_communicator_fatal(primary)
    )
    try:
        _join(publisher, published)
        assert errors == []
        assert results == [primary]
        assert touched == []
        assert owner.state == "enqueued"
        assert owner._counted_quarantine_pending is True
        assert owner._async_done.is_set()
        assert not owner._async_worker.is_alive()
        with runtime._terminal_gate._condition:
            assert not any(
                state.status == "active"
                for state in runtime._terminal_gate._tokens.values()
            )
        retained = {
            record.identity
            for record in runtime._terminal_quarantine.allocations
        }
        assert {
            record.identity for record in owner.allocations
        }.issubset(retained)
    finally:
        event.release.set()
        _close_case(runtime, store)


def test_fatal_publisher_converts_its_implicit_lease_admission(monkeypatch):
    collective = _LoopbackCollective(1)
    runtime = _runtime(collective=collective)
    gate = runtime._terminal_gate
    epoch, construction = gate.begin_lease("fatal-self-admission")
    gate.activate_lease(epoch, construction)
    gate.release(construction)
    token = gate.admit_lease(epoch, "operator_call")
    primary = RuntimeError("fatal publisher owns lease admission")
    blocked = []
    original_wait = gate.wait_for_admissions

    def reject_self_wait(transition, timeout_s):
        with gate._condition:
            state = gate._tokens[token.sequence]
            if state.status == "active":
                blocked.append(state.token)
                raise AssertionError("fatal publisher waited on its own admission")
        return original_wait(transition, timeout_s)

    monkeypatch.setattr(gate, "wait_for_admissions", reject_self_wait)
    try:
        assert runtime._enter_communicator_fatal(primary) is primary
    finally:
        runtime._release_admission(token)

    assert blocked == []
    assert collective._fatal_error is primary
    assert gate.phase is _TerminalPhase.FATAL_PUBLISHED


def test_child_close_descendant_starts_during_elected_transition():
    runtime, _, _, store, _, lease = _open_active_case(
        store_id="terminal-child-close-descendant"
    )
    event = _BlockingEvent()
    child = lease.acquire(_block_request(lease))
    lease.scheduler._event_factory = lambda: event
    closer, _, close_errors, close_done = _start(lease.close)
    try:
        assert event.recorded.wait(_TIMEOUT_S)
        assert event.waiting.wait(_TIMEOUT_S)
        assert not close_done.is_set()
        gate = runtime._terminal_gate
        with gate._condition:
            transition = gate._leases[lease._epoch].transition
            descendants = [
                state.token
                for state in gate._tokens.values()
                if state.status == "active"
                and state.token.parent_sequence is not None
            ]
        assert len(descendants) == 1
        _assert_token(
            descendants[0],
            scope="lease_close",
            epoch=lease._epoch,
            transition=transition,
        )
        event.release.set()
        _join(closer, close_done)
        assert close_errors == []
    finally:
        event.release.set()
        if closer.is_alive():
            lease.scheduler._start_counted_completions()
            assert close_done.wait(_TIMEOUT_S)
            closer.join(_TIMEOUT_S)
        _close_case(runtime, store)


def test_elected_lease_close_binds_every_step_and_dirty_callback(monkeypatch):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-close-matrix"
    )
    output = runtime.backend.zeros((4,), dtype=np.float64)
    lease.mark_dirty("output", output)
    child = lease.acquire(_block_request(lease))
    observed = {}
    transitions = []

    def record(name):
        observed.setdefault(name, []).append(_current_token(runtime))

    def wrap(obj, attribute, name):
        original = getattr(obj, attribute)

        def call(*args, **kwargs):
            record(name)
            return original(*args, **kwargs)

        monkeypatch.setattr(obj, attribute, call)

    original_begin = runtime._terminal_gate.begin_lease_close

    def begin(epoch):
        result = original_begin(epoch)
        transitions.append(result)
        return result

    monkeypatch.setattr(runtime._terminal_gate, "begin_lease_close", begin)
    wrap(child, "close", "child_close")
    wrap(lease, "_schedule_writeback", "schedule_writeback")
    wrap(lease._store_reservation, "commit", "dirty_writeback_callback")
    wrap(lease, "_observe_peaks", "observe_peaks")
    wrap(lease.scheduler, "complete_all", "scheduler_complete")
    wrap(provider.cache, "wait_for_pending", "cache_wait")
    wrap(lease.pool, "reap_completed", "pool_reap")
    wrap(provider.cache, "invalidate_ref", "cache_invalidate")
    wrap(lease, "_emit_profile", "emit_profile")
    wrap(lease._status_workspace, "close", "status_close")
    wrap(lease.scheduler, "close", "scheduler_close")
    wrap(lease.pool, "close", "pool_close")
    wrap(lease._cache_reservation, "close", "cache_reservation_close")
    wrap(lease._store_reservation, "close", "store_reservation_close")

    try:
        lease.close()
        assert len(transitions) == 1
        transition, elected = transitions[0]
        assert elected is True
        assert transition.epoch == lease._epoch
        for name in (
            "child_close",
            "schedule_writeback",
            "dirty_writeback_callback",
            "observe_peaks",
            "scheduler_complete",
            "cache_wait",
            "pool_reap",
            "cache_invalidate",
            "emit_profile",
            "status_close",
            "scheduler_close",
            "pool_close",
            "cache_reservation_close",
            "store_reservation_close",
        ):
            assert name in observed
            token = observed[name][0]
            _assert_token(
                token,
                scope="lease_close",
                epoch=transition.epoch,
                transition=transition,
            )
        schedule = observed["schedule_writeback"][0]
        callback = observed["dirty_writeback_callback"][0]
        _assert_token(
            callback,
            scope="lease_close",
            epoch=transition.epoch,
            transition=transition,
            parent=schedule,
        )
        assert _current_token(runtime) is None
    finally:
        _close_case(runtime, store)


@pytest.mark.parametrize("row", LEASE_CLOSE_MATRIX_ROWS)
@pytest.mark.parametrize("transition_kind", ("fatal", "runtime_close"))
def test_elected_close_row_races_terminal_transition(
    monkeypatch, row, transition_kind
):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-close-row-{}-{}".format(transition_kind, row)
    )
    output = runtime.backend.zeros((4,), dtype=np.float64)
    lease.mark_dirty("output", output)
    child = lease.acquire(_block_request(lease))
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def wrap(obj, attribute, name):
        original = getattr(obj, attribute)
        blocked = [False]

        def call(*args, **kwargs):
            token = _current_token(runtime)
            calls.append(name)
            if name == row and not blocked[0]:
                blocked[0] = True
                _assert_token(
                    token,
                    scope="lease_close",
                    epoch=lease._epoch,
                    transition=runtime._terminal_gate._leases[
                        lease._epoch
                    ].transition,
                )
                entered.set()
                assert release.wait(_TIMEOUT_S)
            return original(*args, **kwargs)

        monkeypatch.setattr(obj, attribute, call)

    wrap(child, "close", "child_close")
    wrap(lease, "_schedule_writeback", "schedule_writeback")
    wrap(lease._store_reservation, "commit", "dirty_writeback_callback")
    wrap(lease, "_observe_peaks", "observe_peaks")
    wrap(lease.scheduler, "complete_all", "scheduler_complete")
    wrap(provider.cache, "wait_for_pending", "cache_wait")
    wrap(lease.pool, "reap_completed", "pool_reap")
    wrap(provider.cache, "invalidate_ref", "cache_invalidate")
    wrap(lease, "_emit_profile", "emit_profile")
    wrap(lease._status_workspace, "close", "status_close")
    wrap(lease.scheduler, "close", "scheduler_close")
    wrap(lease.pool, "close", "pool_close")
    wrap(lease._cache_reservation, "close", "cache_reservation_close")
    wrap(lease._store_reservation, "close", "store_reservation_close")

    closer, close_results, close_errors, close_done = _start(lease.close)
    assert entered.wait(_TIMEOUT_S)
    primary = RuntimeError("fatal during close row {}".format(row))
    transition = (
        (lambda: runtime._enter_communicator_fatal(primary))
        if transition_kind == "fatal"
        else runtime.close
    )
    terminal, terminal_results, terminal_errors, terminal_done = _start(transition)
    if transition_kind == "fatal":
        _wait_for_phase(runtime, _TerminalPhase.FATAL_PENDING)
    else:
        _wait_for_phase(runtime, _TerminalPhase.RUNTIME_CLOSING)
    try:
        assert not terminal_done.is_set()
        release.set()
        _join(closer, close_done)
        _join(terminal, terminal_done)

        assert row in calls
        if transition_kind == "fatal":
            assert close_results == []
            assert close_errors == [primary]
            assert terminal_results == [primary]
            assert terminal_errors == []
            assert provider._active_lease is lease
            assert lease._closed is False
            assert lease._provider is provider
        else:
            diagnostics = (
                tuple(
                    "".join(traceback.format_exception(error))
                    for error in close_errors
                ),
                tuple(
                    "".join(traceback.format_exception(error))
                    for error in terminal_errors
                ),
                calls,
            )
            assert close_results == [None], diagnostics
            assert close_errors == [], diagnostics
            assert terminal_results == [None], diagnostics
            assert terminal_errors == [], diagnostics
            assert runtime._closed is True
    finally:
        release.set()
        for thread, done in ((closer, close_done), (terminal, terminal_done)):
            if thread.is_alive():
                _join(thread, done)
        _close_case(runtime, store)


def test_second_lease_close_and_runtime_close_join_elected_owner(monkeypatch):
    runtime, _, _, store, _, lease = _open_active_case(
        store_id="terminal-close-election"
    )
    close_step_entered = threading.Event()
    release_close_step = threading.Event()
    joiner_began = threading.Event()
    close_step_owners = []
    begin_calls = []
    original_observe = lease._observe_peaks
    original_begin = runtime._terminal_gate.begin_lease_close

    def observe():
        close_step_owners.append(threading.get_ident())
        close_step_entered.set()
        assert release_close_step.wait(_TIMEOUT_S)
        return original_observe()

    def begin(epoch):
        result = original_begin(epoch)
        begin_calls.append(result)
        if result[1] is False:
            joiner_began.set()
        return result

    monkeypatch.setattr(lease, "_observe_peaks", observe)
    monkeypatch.setattr(runtime._terminal_gate, "begin_lease_close", begin)
    owner, owner_results, owner_errors, owner_done = _start(lease.close)
    assert close_step_entered.wait(_TIMEOUT_S)
    joiner, join_results, join_errors, join_done = _start(lease.close)
    assert joiner_began.wait(_TIMEOUT_S)
    runtime_closer, runtime_results, runtime_errors, runtime_done = _start(runtime.close)
    try:
        assert not owner_done.is_set()
        assert not join_done.is_set()
        assert not runtime_done.is_set()
        release_close_step.set()
        _join(owner, owner_done)
        _join(joiner, join_done)
        _join(runtime_closer, runtime_done)
    finally:
        release_close_step.set()
        for thread, done in (
            (owner, owner_done),
            (joiner, join_done),
            (runtime_closer, runtime_done),
        ):
            if thread.is_alive():
                _join(thread, done)
        _close_case(runtime, store)

    assert owner_errors == []
    assert join_errors == []
    assert runtime_errors == []
    assert owner_results == [None]
    assert join_results == [None]
    assert runtime_results == [None]
    assert len(close_step_owners) == 1
    assert begin_calls[0][1] is True
    assert all(transition == begin_calls[0][0] for transition, _ in begin_calls)


def test_fatal_preempts_lease_close_owner_and_joiner_with_same_primary(monkeypatch):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-fatal-close-joiner"
    )
    close_step_entered = threading.Event()
    release_close_step = threading.Event()
    joiner_waiting = threading.Event()
    original_observe = lease._observe_peaks
    original_wait = runtime._terminal_gate.wait_for_lease_closed

    def observe():
        close_step_entered.set()
        assert release_close_step.wait(_TIMEOUT_S)
        return original_observe()

    def wait_for_close(*args, **kwargs):
        joiner_waiting.set()
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(lease, "_observe_peaks", observe)
    monkeypatch.setattr(
        runtime._terminal_gate, "wait_for_lease_closed", wait_for_close
    )
    owner, _, owner_errors, owner_done = _start(lease.close)
    assert close_step_entered.wait(_TIMEOUT_S)
    joiner, _, joiner_errors, joiner_done = _start(lease.close)
    assert joiner_waiting.wait(_TIMEOUT_S)
    primary = RuntimeError("fatal preempted elected lease close")
    fatal, fatal_results, fatal_errors, fatal_done = _start(
        lambda: runtime._enter_communicator_fatal(primary)
    )
    _wait_for_phase(runtime, _TerminalPhase.FATAL_PENDING)
    try:
        release_close_step.set()
        for thread, done in (
            (owner, owner_done),
            (joiner, joiner_done),
            (fatal, fatal_done),
        ):
            _join(thread, done)
    finally:
        release_close_step.set()
        for thread, done in (
            (owner, owner_done),
            (joiner, joiner_done),
            (fatal, fatal_done),
        ):
            if thread.is_alive():
                _join(thread, done)
        _close_case(runtime, store)

    assert owner_errors == [primary]
    assert joiner_errors == [primary]
    assert fatal_errors == []
    assert fatal_results == [primary]
    assert provider._active_lease is lease
    assert lease._closed is False
    assert lease._provider is provider


def test_pending_resource_state_waits_query_free_for_published_snapshot(monkeypatch):
    runtime = _runtime()
    primary = RuntimeError("pending resource state")
    transition = runtime._terminal_gate.begin_fatal(primary)
    queried = []

    class ExplodingProvider:
        def runtime_resource_state(self):
            queried.append(True)
            raise AssertionError("pending state queried a live provider")

    runtime._active_provider = ExplodingProvider()
    wait_entered = threading.Event()
    original_wait = runtime._terminal_gate._condition.wait

    def observe_wait(timeout=None):
        wait_entered.set()
        return original_wait(timeout)

    monkeypatch.setattr(runtime._terminal_gate._condition, "wait", observe_wait)
    reader, results, errors, done = _start(runtime.resource_state)
    assert wait_entered.wait(_TIMEOUT_S)
    assert not done.is_set()
    allocation = AsyncAllocationRecord(("explicit", 99), object(), 4096)
    runtime._terminal_quarantine.retain_error(primary)
    runtime._terminal_quarantine.retain_allocations((allocation,))
    runtime._terminal_gate.publish_fatal(transition, primary)
    _join(reader, done)

    assert errors == []
    assert queried == []
    assert results[0]["quarantined_bytes"] == 4096
    assert results[0]["quarantined_array_count"] == 1


def test_fatal_runtime_close_releases_store_reservation_without_clearing_lease():
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-fatal-runtime-store-reservation"
    )
    reservation = lease._store_reservation
    primary = RuntimeError("fatal runtime store reservation")
    try:
        assert runtime._enter_communicator_fatal(primary) is primary
        with pytest.raises(RuntimeError) as caught:
            runtime.close()
        assert caught.value is primary

        assert reservation._closed is True
        assert lease._store_reservation is reservation
        assert provider._active_lease is lease
        assert lease._closed is False
        assert lease._provider is provider
        store.close()
        assert store.closed is True
    finally:
        if not reservation._closed:
            reservation.close()
        if not store.closed:
            store.close()


def test_runtime_close_provider_and_collective_bind_elected_transition(monkeypatch):
    runtime = _runtime()
    observed = []
    transitions = []

    class Provider:
        _active_lease = None
        _terminal_error = None

        def close(self, *args, **kwargs):
            observed.append(("provider_close", _current_token(runtime)))

    provider = Provider()
    runtime._active_provider = provider
    original_close = runtime.collective.close
    original_begin = runtime._terminal_gate.begin_runtime_close
    original_commit = runtime._terminal_gate.commit_runtime_close

    def collective_close():
        observed.append(("collective_close", _current_token(runtime)))
        return original_close()

    def begin(token):
        transition = original_begin(token)
        transitions.append(transition)
        return transition

    def commit(transition, finalizer):
        assert _current_token(runtime) is None
        return original_commit(transition, finalizer)

    monkeypatch.setattr(runtime.collective, "close", collective_close)
    monkeypatch.setattr(runtime._terminal_gate, "begin_runtime_close", begin)
    monkeypatch.setattr(runtime._terminal_gate, "commit_runtime_close", commit)

    runtime.close()

    assert len(transitions) == 1
    transition = transitions[0]
    assert not isinstance(transition, _FatalTransition)
    assert {name for name, _ in observed} == {
        "provider_close",
        "collective_close",
    }
    for _, token in observed:
        _assert_token(
            token,
            scope="runtime_close",
            epoch=None,
            transition=transition,
        )
    assert runtime._terminal_gate.phase is _TerminalPhase.RUNTIME_CLOSED


def test_fatal_preempts_runtime_provider_close_admission(monkeypatch):
    runtime = _runtime()
    provider_admit_entered = threading.Event()
    release_provider_admit = threading.Event()
    provider_close_calls = []
    terminal_finalizers = []

    class Provider:
        _active_lease = None
        _terminal_error = None

        def _retain_transition_resources(self, lease):
            assert lease is None

        def _finalize_terminal_runtime_close(self, error):
            terminal_finalizers.append(error)

        def close(self):
            provider_close_calls.append(True)

    runtime._active_provider = Provider()
    gate = runtime._terminal_gate
    original_admit = gate.admit_runtime_close

    def admit(transition, operation):
        if operation == "provider_close":
            provider_admit_entered.set()
            assert release_provider_admit.wait(_TIMEOUT_S)
        return original_admit(transition, operation)

    monkeypatch.setattr(gate, "admit_runtime_close", admit)
    closer, close_results, close_errors, close_done = _start(runtime.close)
    assert provider_admit_entered.wait(_TIMEOUT_S)
    primary = RuntimeError("fatal before provider close admission")
    fatal, fatal_results, fatal_errors, fatal_done = _start(
        lambda: runtime._enter_communicator_fatal(primary)
    )
    _wait_for_phase(runtime, _TerminalPhase.FATAL_PUBLISHED)
    try:
        release_provider_admit.set()
        _join(closer, close_done)
        _join(fatal, fatal_done)
    finally:
        release_provider_admit.set()
        for thread, done in ((closer, close_done), (fatal, fatal_done)):
            if thread.is_alive():
                _join(thread, done)

    assert close_results == []
    assert close_errors == [primary]
    assert fatal_results == [primary]
    assert fatal_errors == []
    assert provider_close_calls == []
    assert terminal_finalizers == [primary]
    assert runtime._closed is True
    assert gate.phase is _TerminalPhase.RUNTIME_CLOSED


def test_fatal_preempts_runtime_close_begin_and_joins_commit(monkeypatch):
    runtime = _runtime(collective=_LoopbackCollective(1))
    begin_entered = threading.Event()
    release_begin = threading.Event()
    gate = runtime._terminal_gate
    original_begin = gate.begin_runtime_close

    def begin(token):
        begin_entered.set()
        assert release_begin.wait(_TIMEOUT_S)
        return original_begin(token)

    monkeypatch.setattr(gate, "begin_runtime_close", begin)
    closer, close_results, close_errors, close_done = _start(runtime.close)
    assert begin_entered.wait(_TIMEOUT_S)
    primary = RuntimeError("fatal during runtime close begin")
    fatal, fatal_results, fatal_errors, fatal_done = _start(
        lambda: runtime._enter_communicator_fatal(primary)
    )
    _wait_for_phase(runtime, _TerminalPhase.FATAL_PENDING)
    try:
        release_begin.set()
        _join(closer, close_done)
        _join(fatal, fatal_done)
    finally:
        release_begin.set()
        for thread, done in ((closer, close_done), (fatal, fatal_done)):
            if thread.is_alive():
                _join(thread, done)

    assert close_results == []
    assert close_errors == [primary]
    assert fatal_results == [primary]
    assert fatal_errors == []
    assert runtime._closed is True
    assert gate.phase is _TerminalPhase.RUNTIME_CLOSED


def test_fatal_preempts_runtime_collective_close_admission(monkeypatch):
    collective = _LoopbackCollective(1)
    runtime = _runtime(collective=collective)
    collective_admit_entered = threading.Event()
    release_collective_admit = threading.Event()
    collective_close_calls = []
    gate = runtime._terminal_gate
    original_admit = gate.admit_runtime_close
    original_close = collective.close

    def admit(transition, operation):
        if operation == "collective_close":
            collective_admit_entered.set()
            assert release_collective_admit.wait(_TIMEOUT_S)
        return original_admit(transition, operation)

    def close_collective():
        collective_close_calls.append(True)
        return original_close()

    monkeypatch.setattr(gate, "admit_runtime_close", admit)
    monkeypatch.setattr(collective, "close", close_collective)
    closer, close_results, close_errors, close_done = _start(runtime.close)
    assert collective_admit_entered.wait(_TIMEOUT_S)
    primary = RuntimeError("fatal before collective close admission")
    fatal, fatal_results, fatal_errors, fatal_done = _start(
        lambda: runtime._enter_communicator_fatal(primary)
    )
    _wait_for_phase(runtime, _TerminalPhase.FATAL_PUBLISHED)
    try:
        release_collective_admit.set()
        _join(closer, close_done)
        _join(fatal, fatal_done)
    finally:
        release_collective_admit.set()
        for thread, done in ((closer, close_done), (fatal, fatal_done)):
            if thread.is_alive():
                _join(thread, done)

    assert close_results == []
    assert close_errors == [primary]
    assert fatal_results == [primary]
    assert fatal_errors == []
    assert collective_close_calls == [True]
    assert runtime._closed is True
    assert gate.phase is _TerminalPhase.RUNTIME_CLOSED


def test_fatal_preempts_runtime_close_commit_with_same_primary(monkeypatch):
    runtime = _runtime(collective=_LoopbackCollective(1))
    commit_entered = threading.Event()
    release_commit = threading.Event()
    gate = runtime._terminal_gate
    original_commit = gate.commit_runtime_close

    def commit(transition, finalizer):
        if not isinstance(transition, _FatalTransition):
            commit_entered.set()
            assert release_commit.wait(_TIMEOUT_S)
        return original_commit(transition, finalizer)

    monkeypatch.setattr(gate, "commit_runtime_close", commit)
    closer, close_results, close_errors, close_done = _start(runtime.close)
    assert commit_entered.wait(_TIMEOUT_S)
    primary = RuntimeError("fatal during runtime close commit")
    fatal, fatal_results, fatal_errors, fatal_done = _start(
        lambda: runtime._enter_communicator_fatal(primary)
    )
    _wait_for_phase(runtime, _TerminalPhase.FATAL_PUBLISHED)
    try:
        release_commit.set()
        _join(closer, close_done)
        _join(fatal, fatal_done)
    finally:
        release_commit.set()
        for thread, done in ((closer, close_done), (fatal, fatal_done)):
            if thread.is_alive():
                _join(thread, done)

    assert close_results == []
    assert close_errors == [primary]
    assert fatal_results == [primary]
    assert fatal_errors == []
    assert runtime._closed is True
    assert gate.phase is _TerminalPhase.RUNTIME_CLOSED


@pytest.mark.parametrize(
    "scope",
    ("runtime", "runtime_setup", "construction", "lease_close", "runtime_close"),
)
def test_wrong_canonical_scope_is_rejected_before_lease_resource(
    monkeypatch, scope
):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-wrong-scope-{}".format(scope)
    )
    gate = runtime._terminal_gate
    token = gate.admit_lease(lease._epoch, "valid_lease_fixture")
    wrong = replace(token, scope=scope)
    sentinel_calls = []
    original_acquire = provider.cache.acquire

    def sentinel(*args, **kwargs):
        sentinel_calls.append(True)
        return original_acquire(*args, **kwargs)

    monkeypatch.setattr(provider.cache, "acquire", sentinel)
    try:
        with pytest.raises((TypeError, RuntimeError)):
            lease._load_identity(
                lease.cache_identities[0], _admission_token=wrong
            )
        assert sentinel_calls == []
    finally:
        gate.release(token)
        lease.close()
        _close_case(runtime, store)



def test_noncanonical_scope_string_is_rejected_before_resource_sentinel(
    monkeypatch,
):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-noncanonical-scope"
    )
    gate = runtime._terminal_gate
    token = gate.admit_lease(lease._epoch, "valid_lease_fixture")
    malformed = replace(token, scope="lease-close")
    sentinel_calls = []
    original_acquire = provider.cache.acquire

    def sentinel(*args, **kwargs):
        sentinel_calls.append(True)
        return original_acquire(*args, **kwargs)

    monkeypatch.setattr(provider.cache, "acquire", sentinel)
    try:
        with pytest.raises((TypeError, RuntimeError)):
            lease._load_identity(
                lease.cache_identities[0], _admission_token=malformed
            )
        assert sentinel_calls == []
    finally:
        gate.release(token)
        lease.close()
        _close_case(runtime, store)
