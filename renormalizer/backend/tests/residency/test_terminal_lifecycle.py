import ast
from contextlib import contextmanager
from dataclasses import replace
import inspect
import threading
import traceback

import numpy as np
import pytest

from renormalizer.backend._distributed.async_owner import (
    AsyncAllocationRecord,
    AsyncResourceOwner,
    _CountedAsyncAdmission,
    allocation_record,
)
from renormalizer.backend._distributed.cache import (
    CacheReservation,
    DeviceTensorCache,
)
from renormalizer.backend._distributed.pinned import PinnedBufferPool
from renormalizer.backend._distributed.residency import ResidencyPlanner
from renormalizer.backend._distributed.providers import (
    ActiveWorkingSetProvider,
    WorkingSetLease,
    _ActiveOperandLease,
    _LeaseResourceRecord,
    _LeaseStatusWorkspace,
)
from renormalizer.backend._distributed import providers as providers_module
from renormalizer.backend._distributed.terminal import (
    _FatalTransition,
    _MISSING,
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
    "cache_lifetime_reconcile": {
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
    "cache_lifetime_reconcile",
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
    original_activate = provider._activate_lease

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

    def activate(epoch, **kwargs):
        return intercept(
            "activate_lease",
            lambda: original_activate(epoch, **kwargs),
        )

    monkeypatch.setattr(runtime, "consume_residency_receipt", consume)
    monkeypatch.setattr(provider, "_provision_status_workspace", status)
    monkeypatch.setattr(store, "reserve", reserve_store)
    provider._cache_factory = cache_factory
    provider._pool_factory = pool_factory
    provider._scheduler_factory = scheduler_factory
    monkeypatch.setattr(provider, "_activate_lease", activate)


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
        "cache_lifetime_reconcile",
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


def _matrix_admission_token(runtime, row):
    gate = runtime._terminal_gate
    specification = ADMISSION_SENTINEL_MATRIX[row]
    scope = specification["scope"]
    retained = []
    transition = None

    if scope == "runtime":
        token = gate.admit_runtime(row)
    elif scope == "runtime_setup":
        token = gate.admit_runtime_setup(row)
    elif scope == "construction":
        _, token = gate.begin_lease(row)
    elif scope == "lease":
        epoch, construction = gate.begin_lease("{}_construction".format(row))
        gate.activate_lease(epoch, construction)
        gate.release(construction)
        parent = gate.admit_lease(epoch, row)
        if "parent_sequence" in specification:
            capability = object()
            token = gate.spawn_async(parent, capability)
            retained.append(parent)
        else:
            token = parent
    elif scope == "lease_close":
        epoch, construction = gate.begin_lease("{}_construction".format(row))
        gate.activate_lease(epoch, construction)
        gate.release(construction)
        transition, elected = gate.begin_lease_close(epoch)
        assert elected is True
        parent = gate.admit_lease_close(transition, row)
        if "parent_sequence" in specification:
            capability = object()
            token = gate.spawn_async(parent, capability)
            retained.append(parent)
        else:
            token = parent
    elif scope == "runtime_close":
        transition = gate.begin_runtime_close(None)
        token = gate.admit_runtime_close(transition, row)
    else:
        raise AssertionError("unknown matrix scope {!r}".format(scope))
    return token, transition, retained


class _LowerAdmissionSentinel(Exception):
    pass


def _invoke_matrix_lower_helper(runtime, row, token, validator):
    provider = object.__new__(ActiveWorkingSetProvider)
    lease = object.__new__(WorkingSetLease)
    child = object.__new__(_ActiveOperandLease)
    cache = object.__new__(DeviceTensorCache)
    pool = object.__new__(PinnedBufferPool)
    scheduler = object.__new__(TransferScheduler)
    status = object.__new__(_LeaseStatusWorkspace)
    cache_reservation = object.__new__(CacheReservation)
    admitted = {
        "_admission_token": token,
        "_admission_validator": validator,
    }

    if row == "barrier_collective":
        return runtime._barrier_collective(**admitted)
    if row == "execution_config_backend_sync":
        return runtime._synchronize_active_backend(**admitted)
    if row == "budget_probe":
        return runtime._resolve_budget("device", 1, **admitted)
    if row == "provider_construct":
        return ActiveWorkingSetProvider(
            runtime,
            device_budget_resolution=None,
            host_budget_resolution=None,
            **admitted,
        )
    if row == "provider_install":
        return runtime._install_active_provider(provider, **admitted)
    if row == "provider_config_match":
        return ActiveWorkingSetProvider.matches_config(
            provider,
            None,
            None,
            1,
            **admitted,
        )
    if row == "preflight_control_allocation":
        return runtime._control_array([], np.int32, **admitted)
    if row == "preflight_collective":
        return runtime._runtime_setup_allreduce(
            np.zeros(1, dtype=np.int32),
            op="max",
            **admitted,
        )
    if row == "receipt_publish":
        return runtime._publish_residency_receipt(object(), **admitted)
    if row == "receipt_consume":
        return runtime.consume_residency_receipt(None, None, None, **admitted)
    if row == "construct_status":
        return ActiveWorkingSetProvider._provision_status_workspace(
            provider,
            _resource_recorder=None,
            **admitted,
        )
    if row == "construct_store_reservation":
        return ActiveWorkingSetProvider._reserve_store(
            None,
            None,
            dirty_ref=None,
            **admitted,
        )
    if row == "construct_cache_reservation":
        return DeviceTensorCache.reserve(cache, None, 0, **admitted)
    if row == "construct_pool":
        return PinnedBufferPool(0, **admitted)
    if row == "construct_scheduler":
        return TransferScheduler(None, None, None, **admitted)
    if row == "activate_lease":
        return ActiveWorkingSetProvider._activate_lease(
            provider,
            token.epoch,
            **admitted,
        )
    if row == "rollback_partial":
        return ActiveWorkingSetProvider._rollback_partial(
            provider,
            token,
            RuntimeError("rollback sentinel"),
            record=None,
            scheduler=None,
            pool=None,
            cache_reservation=None,
            store_reservation=None,
            status_workspace=None,
            _admission_validator=validator,
        )
    if row == "operator_call":
        context = WorkingSetLease._operator_call_admitted(
            lease,
            None,
            **admitted,
        )
        return context.__enter__()
    if row == "acquire":
        return WorkingSetLease._acquire_admitted(
            lease,
            None,
            **admitted,
        )
    if row == "load":
        return WorkingSetLease._load_identity_admitted(
            lease,
            None,
            prefetch=False,
            **admitted,
        )
    if row == "prefetch":
        return WorkingSetLease._prefetch_one_admitted(lease, **admitted)
    if row == "mark_dirty":
        return WorkingSetLease._mark_dirty_admitted(
            lease,
            None,
            None,
            **admitted,
        )
    if row == "reap":
        return WorkingSetLease._reap_completed_admitted(lease, **admitted)
    if row in {
        "resource_state_between_leases",
        "resource_state_open_lease",
    }:
        return ActiveWorkingSetProvider.runtime_resource_state(
            provider,
            **admitted,
        )
    if row == "ordinary_async_quarantine":
        return _CountedAsyncAdmission._run_admitted_callback(
            lambda: None,
            **admitted,
        )
    if row == "child_close":
        return _ActiveOperandLease._close_admitted(
            child,
            token,
            validator,
        )
    if row == "schedule_writeback":
        return WorkingSetLease._schedule_writeback(lease, **admitted)
    if row == "dirty_writeback_callback":
        return TransferScheduler._commit_writeback(
            None,
            None,
            None,
            None,
            **admitted,
        )
    if row == "observe_peaks":
        return WorkingSetLease._observe_peaks(lease, **admitted)
    if row == "scheduler_complete":
        return TransferScheduler.complete_all(scheduler, **admitted)
    if row == "cache_wait":
        return DeviceTensorCache.wait_for_pending(cache, **admitted)
    if row == "pool_reap":
        return PinnedBufferPool.reap_completed(pool, **admitted)
    if row == "cache_invalidate":
        return DeviceTensorCache.invalidate_ref(cache, None, **admitted)
    if row == "emit_profile":
        return WorkingSetLease._emit_profile(lease, 0, pool, **admitted)
    if row == "status_close":
        return _LeaseStatusWorkspace.close(status, **admitted)
    if row == "scheduler_close":
        return TransferScheduler.close(scheduler, **admitted)
    if row == "pool_close":
        return PinnedBufferPool.close(pool, **admitted)
    if row == "cache_reservation_close":
        return CacheReservation.close(cache_reservation, **admitted)
    if row == "cache_lifetime_reconcile":
        return ActiveWorkingSetProvider._reconcile_lease_cache(
            provider,
            None,
            **admitted,
        )
    if row == "store_reservation_close":
        return WorkingSetLease._close_store_reservation(None, **admitted)
    if row == "provider_close":
        return ActiveWorkingSetProvider._close_for_runtime(provider, **admitted)
    if row == "collective_close":
        return runtime._close_collective(None, **admitted)
    raise AssertionError("unknown matrix row {!r}".format(row))


@pytest.mark.parametrize("row", tuple(ADMISSION_SENTINEL_MATRIX))
@pytest.mark.parametrize(
    "token_kind",
    ("exact", "wrong_canonical", "noncanonical"),
)
def test_every_matrix_row_enforces_exact_lower_admission_before_sentinel(
    row,
    token_kind,
):
    runtime = _runtime()
    gate = runtime._terminal_gate
    token, transition, retained = _matrix_admission_token(runtime, row)
    specification = ADMISSION_SENTINEL_MATRIX[row]
    exact_validator = runtime._exact_admission_validator(
        token,
        scope=specification["scope"],
        epoch=token.epoch,
        parent_sequence=(
            token.parent_sequence
            if "parent_sequence" in specification
            else None
        ),
        transition_sequence=(
            transition.sequence
            if "transition_sequence" in specification
            else None
        ),
    )
    sentinel_calls = []

    def validator(candidate):
        exact_validator(candidate)
        sentinel_calls.append(row)
        raise _LowerAdmissionSentinel(row)

    candidate = token
    if token_kind != "exact":
        expected_epoch = token.epoch
        gate.release(token)
        token = None
        for retained_token in retained:
            gate.release(retained_token)
        retained = []
        candidate_scope = (
            next(
                scope
                for scope in (
                    "runtime",
                    "runtime_setup",
                    "construction",
                    "lease",
                    "lease_close",
                    "runtime_close",
                )
                if scope != specification["scope"]
            )
            if token_kind == "wrong_canonical"
            else "lease-close"
        )
        candidate_epoch = (
            None
            if candidate_scope in {
                "runtime",
                "runtime_setup",
                "runtime_close",
            }
            else expected_epoch
        )
        with gate._condition:
            candidate = gate._new_token(
                candidate_scope,
                candidate_epoch,
                "{}_candidate".format(token_kind),
            )

    try:
        if token_kind == "exact":
            with pytest.raises(_LowerAdmissionSentinel):
                _invoke_matrix_lower_helper(runtime, row, candidate, validator)
            assert sentinel_calls == [row]
        else:
            with pytest.raises((TypeError, RuntimeError)):
                _invoke_matrix_lower_helper(runtime, row, candidate, validator)
            assert sentinel_calls == []
    finally:
        if token is not None:
            gate.release(token)
        elif candidate is not None:
            gate.release(candidate)
        for retained_token in retained:
            gate.release(retained_token)


@pytest.mark.parametrize("async_kind", ("ordinary", "dirty_writeback"))
def test_counted_async_admission_rejects_real_sibling_before_lower_mutation(
    async_kind,
):
    runtime = _runtime()
    gate = runtime._terminal_gate
    epoch, construction = gate.begin_lease(
        "{}_async_construction".format(async_kind)
    )
    gate.activate_lease(epoch, construction)
    gate.release(construction)
    transition = None
    if async_kind == "ordinary":
        parent = gate.admit_lease(epoch, "ordinary_async_parent")
    else:
        transition, elected = gate.begin_lease_close(epoch)
        assert elected is True
        parent = gate.admit_lease_close(
            transition,
            "dirty_writeback_async_parent",
        )
    expected = _CountedAsyncAdmission(gate, parent, object())
    sibling = _CountedAsyncAdmission(gate, parent, object())
    gate.release(parent)
    mutations = []

    class Reservation:
        def __init__(self, label):
            self.label = label

        def commit(self, destination_ref, staging):
            mutations.append(self.label)
            return self.label

    def invoke_lower(label, token, validator):
        if async_kind == "ordinary":
            return _CountedAsyncAdmission._run_admitted_callback(
                lambda **kwargs: mutations.append(label),
                _admission_token=token,
                _admission_validator=validator,
            )
        return TransferScheduler._commit_writeback(
            Reservation(label),
            None,
            None,
            None,
            _admission_token=token,
            _admission_validator=validator,
        )

    expected.run(
        "{}_positive".format(async_kind),
        lambda **admitted: invoke_lower(
            "positive",
            admitted["_admission_token"],
            admitted["_admission_validator"],
        ),
    )
    assert mutations == ["positive"]

    with pytest.raises(RuntimeError, match="async admission"):
        sibling.run(
            "{}_sibling".format(async_kind),
            lambda **admitted: invoke_lower(
                "sibling",
                admitted["_admission_token"],
                expected._validate_claimed,
            ),
        )
    assert mutations == ["positive"]
    with gate._condition:
        assert gate._tokens[expected.token.sequence].status == "released"
        assert gate._tokens[sibling.token.sequence].status == "released"
        assert not any(state.status == "active" for state in gate._tokens.values())
    if transition is not None:
        gate.commit_lease_close(transition, lambda: None)


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

    def synchronize_backend(
        *, _admission_token=None, _admission_validator=None
    ):
        record("execution_config_backend_sync")
        return "numpy", "cpu", 64

    def resolve_budget(
        resource,
        requested,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
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

    def install(
        provider,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
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

    def synchronize_backend(
        *, _admission_token=None, _admission_validator=None
    ):
        block("execution_config_backend_sync")
        return "numpy", "cpu", 64

    def resolve_budget(
        resource,
        requested,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
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

        def _close_for_runtime(
            self, *, _admission_token, _admission_validator
        ):
            self.closed = True

    original_install = runtime._install_active_provider

    def install(provider, *, _admission_token, _admission_validator):
        block("provider_install")
        return original_install(
            provider,
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
        )

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

    def publish(
        receipt,
        *,
        _admission_token=None,
        _admission_validator=None,
    ):
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

    def publish(receipt, *, _admission_token, _admission_validator):
        block("receipt_publish")
        return original_publish(
            receipt,
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
        )

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
        lambda **kwargs: ("numpy", "cpu", 64),
    )
    install_entered = threading.Event()
    release_install = threading.Event()
    original_install = runtime._install_active_provider

    def install(provider, *, _admission_token, _admission_validator):
        install_entered.set()
        assert release_install.wait(_TIMEOUT_S)
        return original_install(
            provider,
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
        )

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

    def publish(receipt, *, _admission_token, _admission_validator):
        publish_entered.set()
        assert release_publish.wait(_TIMEOUT_S)
        return original_publish(
            receipt,
            _admission_token=_admission_token,
            _admission_validator=_admission_validator,
        )

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

    def stop_before_allocation(**kwargs):
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


@pytest.mark.parametrize(
    "failure_point",
    ("host_allocation", "bootstrap"),
)
def test_fatal_partial_status_construction_retains_allocations_and_receipt(
    monkeypatch,
    failure_point,
):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-partial-status-{}".format(failure_point),
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    entered = threading.Event()
    release = threading.Event()
    allocated = []
    failure = RuntimeError("injected {} failure".format(failure_point))
    original_device = provider._allocate_status_device
    original_host = provider._allocate_status_host
    original_bootstrap = runtime.collective._bootstrap_status_or

    def allocate_device():
        array = original_device()
        allocated.append(array)
        return array

    def allocate_host():
        if failure_point == "host_allocation":
            entered.set()
            assert release.wait(_TIMEOUT_S)
            raise failure
        array = original_host()
        allocated.append(array)
        return array

    def bootstrap(local_code):
        if failure_point == "bootstrap":
            entered.set()
            assert release.wait(_TIMEOUT_S)
            raise failure
        return original_bootstrap(local_code)

    monkeypatch.setattr(provider, "_allocate_status_device", allocate_device)
    monkeypatch.setattr(provider, "_allocate_status_host", allocate_host)
    monkeypatch.setattr(runtime.collective, "_bootstrap_status_or", bootstrap)
    opener, open_results, open_errors, open_done = _start(
        lambda: provider.open_working_set(request, plan, store, receipt)
    )
    assert entered.wait(_TIMEOUT_S)
    record = provider._provisional_resources
    primary = RuntimeError("fatal during partial status construction")
    fatal, fatal_results, fatal_errors, fatal_done = _start(
        lambda: runtime._enter_communicator_fatal(primary)
    )
    _wait_for_phase(runtime, _TerminalPhase.FATAL_PENDING)
    try:
        release.set()
        _join(opener, open_done)
        _join(fatal, fatal_done)

        identities = {allocation_record(array).identity for array in allocated}
        assert identities
        assert identities.issubset(
            {retained.identity for retained in record.allocations}
        )
        assert identities.issubset(
            {
                retained.identity
                for retained in runtime._terminal_quarantine.allocations
            }
        )
        assert any(resource is receipt for resource in record.resources)
        assert any(
            resource is receipt
            for resource in runtime._terminal_quarantine._resources
        )
        assert open_results == []
        assert open_errors == [primary]
        assert fatal_results == [primary]
        assert fatal_errors == []
    finally:
        release.set()
        for thread, done in ((opener, open_done), (fatal, fatal_done)):
            if thread.is_alive():
                _join(thread, done)
        _close_case(runtime, store)


def test_healthy_construction_rollback_restores_consumed_receipt(monkeypatch):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-restore-consumed-receipt",
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    failure = RuntimeError("injected construction failure after receipt consumption")

    def fail_status(*args, **kwargs):
        raise failure

    monkeypatch.setattr(provider, "_provision_status_workspace", fail_status)
    try:
        with pytest.raises(RuntimeError) as caught:
            provider.open_working_set(request, plan, store, receipt)
        assert caught.value is failure
        assert runtime._issued_receipts[id(receipt)] is receipt
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
    original_activate = provider._activate_lease

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

    def activate(epoch, **kwargs):
        record("activate_lease")
        assert kwargs["_admission_token"] is _current_token(runtime)
        return original_activate(epoch, **kwargs)

    monkeypatch.setattr(runtime, "consume_residency_receipt", consume)
    monkeypatch.setattr(provider, "_provision_status_workspace", status)
    monkeypatch.setattr(store, "reserve", reserve_store)
    provider._cache_factory = cache_factory
    provider._pool_factory = pool_factory
    provider._scheduler_factory = scheduler_factory
    monkeypatch.setattr(provider, "_activate_lease", activate)

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


@pytest.mark.parametrize(
    "failure_boundary",
    ("rollback_before", "finalizer_after", "abort_before"),
)
def test_construction_transaction_fail_stops_unprovable_cleanup(
    monkeypatch,
    failure_boundary,
):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-construction-total-{}".format(failure_boundary),
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    gate = runtime._terminal_gate
    primary = RuntimeError("construction failed")

    class RollbackSecondary(BaseException):
        pass

    secondary = RollbackSecondary(failure_boundary)
    monkeypatch.setattr(
        provider,
        "_provision_status_workspace",
        lambda **_kwargs: (_ for _ in ()).throw(primary),
    )
    if failure_boundary == "rollback_before":
        monkeypatch.setattr(
            provider,
            "_rollback_partial",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(secondary),
        )
    elif failure_boundary == "finalizer_after":
        original_clear = _LeaseResourceRecord.clear

        def fail_after_clear(record):
            original_clear(record)
            raise secondary

        monkeypatch.setattr(_LeaseResourceRecord, "clear", fail_after_clear)
    else:
        original_abort = gate._abort_lease_construction

        def fail_before_abort(*_args, **_kwargs):
            raise secondary

        monkeypatch.setattr(gate, "_abort_lease_construction", fail_before_abort)

    try:
        with pytest.raises(RuntimeError) as caught:
            provider.open_working_set(request, plan, store, receipt)
        assert caught.value is primary
        assert gate._fatal_transition.primary is primary
        assert secondary in runtime._terminal_secondary_errors
        with gate._condition:
            epoch = gate._live_epoch
            assert gate._leases[epoch].phase == "fatal_retained"
            assert not gate._has_active_tokens()
        assert provider._active_lease is None
        assert provider._provisional_resources is not None
    finally:
        _close_case(runtime, store)


def test_every_lease_close_election_uses_the_total_owner_primitive():
    with open(providers_module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())

    module_functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }

    def method(class_name, method_name):
        return next(
            node
            for node in classes[class_name].body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        )

    def calls_named(node, name):
        return [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and (
                isinstance(call.func, ast.Name)
                and call.func.id == name
                or isinstance(call.func, ast.Attribute)
                and call.func.attr == name
            )
        ]

    runner = module_functions["_run_total_elected_lease_close"]
    begin_calls = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if calls_named(node, "begin_lease_close"):
                begin_calls.append(node.name)
    assert begin_calls == ["_run_total_elected_lease_close"]
    assert len(calls_named(runner, "begin_lease_close")) == 1
    assert calls_named(method("WorkingSetLease", "close"), runner.name)
    construction = method(
        "ActiveWorkingSetProvider", "_resolve_construction_failure"
    )
    assert calls_named(construction, "_abort_lease_construction")
    assert calls_named(construction, "_fail_lease_construction")


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
        monkeypatch.setattr(
            provider.cache,
            "contains",
            lambda *args, **kwargs: False,
        )
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

        def reap(**kwargs):
            record("reap")
            return original_reap(**kwargs)

        monkeypatch.setattr(lease.scheduler, "reap_completed", reap)
        lease.reap_completed()

        runtime_state_impl = provider.runtime_resource_state

        def open_state(**kwargs):
            record("resource_state_open_lease")
            return runtime_state_impl(**kwargs)

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
    def between_state(**kwargs):
        between.append(_current_token(runtime))
        return runtime_state_impl(**kwargs)

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

    def state(**kwargs):
        token = _current_token(runtime)
        _assert_token(
            token,
            scope="lease" if lease_open else "runtime",
            epoch=lease._epoch if lease_open else None,
        )
        entered.set()
        assert release.wait(_TIMEOUT_S)
        return original_state(**kwargs)

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


def test_lease_close_latches_start_before_completion_worker_publication(
    monkeypatch,
):
    runtime, _, _, store, _, lease = _open_active_case(
        store_id="terminal-counted-start-before-worker"
    )
    scheduler = lease.scheduler
    event = _BlockingEvent()
    factory_entered = threading.Event()
    release_factory = threading.Event()
    child = lease.acquire(_block_request(lease))

    def event_factory():
        factory_entered.set()
        assert release_factory.wait(_TIMEOUT_S)
        return event

    scheduler._event_factory = event_factory
    child_closer, _, child_errors, child_done = _start(child.close)
    assert factory_entered.wait(_TIMEOUT_S)

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
    assert owner._async_admission.token is descendant
    assert owner._async_worker is None

    start_requested = threading.Event()
    start_calls = []
    original_start = scheduler._start_counted_completions

    def start_counted_completions():
        start_calls.append(threading.get_ident())
        result = original_start()
        start_requested.set()
        return result

    monkeypatch.setattr(
        scheduler,
        "_start_counted_completions",
        start_counted_completions,
    )
    closer, _, close_errors, close_done = _start(lease.close)
    assert start_requested.wait(_TIMEOUT_S)
    assert start_calls == [closer.ident]
    assert not close_done.is_set()

    try:
        release_factory.set()
        _join(child_closer, child_done)
        assert child_errors == []
        assert event.recorded.wait(_TIMEOUT_S)
        assert event.waiting.wait(_TIMEOUT_S)
        event.release.set()
        _join(closer, close_done)

        assert close_errors == []
        assert start_calls == [closer.ident]
        assert owner.state == "detached"
        assert owner._async_done.is_set()
        with gate._condition:
            assert gate._tokens[descendant.sequence].status == "released"
            assert not any(
                state.status == "active" for state in gate._tokens.values()
            )
    finally:
        release_factory.set()
        owner.start_counted_completion()
        event.release.set()
        if child_closer.is_alive():
            _join(child_closer, child_done)
        if closer.is_alive():
            _join(closer, close_done)
        _close_case(runtime, store)


@pytest.mark.parametrize("async_kind", ("ordinary", "close_owned"))
def test_counted_async_cancel_install_claim_is_one_terminal_transaction(
    monkeypatch,
    async_kind,
):
    runtime = _runtime()
    gate = runtime._terminal_gate
    epoch, construction = gate.begin_lease(
        "{}_cancel_race_construction".format(async_kind)
    )
    gate.activate_lease(epoch, construction)
    gate.release(construction)
    transition = None
    if async_kind == "ordinary":
        parent = gate.admit_lease(epoch, "ordinary_cancel_parent")
    else:
        transition, elected = gate.begin_lease_close(epoch)
        assert elected is True
        parent = gate.admit_lease_close(transition, "close_cancel_parent")
    capability = object()
    admission = _CountedAsyncAdmission(gate, parent, capability)
    descendant = admission.token
    gate.release(parent)

    event = _BlockingEvent()
    quarantined = []
    owner = AsyncResourceOwner(
        "compute",
        _async_admission=admission,
        quarantine=quarantined.append,
    )
    owner.mark_enqueued()
    owner.add_event(event, completion=True)

    cancel_entered = threading.Event()
    claim_completed = threading.Event()
    release_cancel = threading.Event()
    release_calls = []
    original_cancel = admission.cancel
    original_claim = gate.claim_async
    original_release = gate.release

    def pause_cancel():
        cancel_entered.set()
        assert release_cancel.wait(_TIMEOUT_S)
        return original_cancel()

    def record_release(token):
        result = original_release(token)
        if token.sequence == descendant.sequence:
            release_calls.append(token.sequence)
        return result

    def record_claim(*args, **kwargs):
        claimed = original_claim(*args, **kwargs)
        claim_completed.set()
        return claimed

    monkeypatch.setattr(admission, "cancel", pause_cancel)
    monkeypatch.setattr(gate, "claim_async", record_claim)
    monkeypatch.setattr(gate, "release", record_release)
    primary = RuntimeError("{} terminal quarantine".format(async_kind))
    quarantiner, _, quarantine_errors, quarantine_done = _start(
        lambda: owner.force_quarantine(primary)
    )
    assert cancel_entered.wait(_TIMEOUT_S)

    def arm_and_start():
        owner.arm_completion()
        if async_kind == "ordinary":
            owner.start_counted_completion()

    armer, _, armer_errors, armer_done = _start(arm_and_start)
    assert claim_completed.wait(_TIMEOUT_S)

    try:
        release_cancel.set()
        _join(quarantiner, quarantine_done)
        _join(armer, armer_done)
        event.release.set()
        assert owner._async_done.wait(_TIMEOUT_S)
        worker = owner._async_worker
        assert worker is not None
        worker.join(_TIMEOUT_S)
        assert not worker.is_alive()

        assert quarantine_errors == []
        assert armer_errors == []
        assert owner.state == "quarantined"
        assert owner.error is primary
        assert quarantined == [owner]
        assert release_calls == [descendant.sequence]
        with gate._condition:
            assert gate._tokens[descendant.sequence].status == "released"
            assert not any(
                state.status == "active" for state in gate._tokens.values()
            )
    finally:
        release_cancel.set()
        event.release.set()
        if quarantiner.is_alive():
            _join(quarantiner, quarantine_done)
        if armer.is_alive():
            _join(armer, armer_done)
        worker = owner._async_worker
        if worker is not None:
            worker.join(_TIMEOUT_S)
        with gate._condition:
            active = [
                state.token
                for state in gate._tokens.values()
                if state.status == "active"
            ]
        for token in active:
            try:
                gate.release(token)
            except BaseException:
                pass
        if transition is None:
            transition, elected = gate.begin_lease_close(epoch)
            assert elected is True
        with gate._condition:
            lease_phase = gate._leases[epoch].phase
        if lease_phase == "closing":
            gate.commit_lease_close(transition, lambda: None)
        runtime.close()


@pytest.mark.parametrize("async_kind", ("ordinary", "close_owned"))
def test_counted_async_cancellation_winner_prevents_late_worker_claim(
    monkeypatch,
    async_kind,
):
    runtime = _runtime()
    gate = runtime._terminal_gate
    epoch, construction = gate.begin_lease(
        "{}_cancel_winner_construction".format(async_kind)
    )
    gate.activate_lease(epoch, construction)
    gate.release(construction)
    transition = None
    if async_kind == "ordinary":
        parent = gate.admit_lease(epoch, "ordinary_cancel_winner")
    else:
        transition, elected = gate.begin_lease_close(epoch)
        assert elected is True
        parent = gate.admit_lease_close(transition, "close_cancel_winner")
    admission = _CountedAsyncAdmission(gate, parent, object())
    descendant = admission.token
    gate.release(parent)

    quarantined = []
    owner = AsyncResourceOwner(
        "compute",
        _async_admission=admission,
        quarantine=quarantined.append,
    )
    owner.mark_enqueued()
    owner.add_event(_BlockingEvent(), completion=True)
    cancel_entered = threading.Event()
    release_cancel = threading.Event()
    run_entered = threading.Event()
    release_run = threading.Event()
    publish_entered = threading.Event()
    release_publish = threading.Event()
    claim_calls = []
    cancel_calls = []
    release_calls = []
    original_cancel = admission.cancel
    original_run = admission.run
    original_publish = owner._publish_counted_start_request
    original_claim = gate.claim_async
    original_cancel_async = gate.cancel_async
    original_release = gate.release

    def pause_cancel():
        cancel_entered.set()
        assert release_cancel.wait(_TIMEOUT_S)
        return original_cancel()

    def pause_run(*args, **kwargs):
        run_entered.set()
        assert release_run.wait(_TIMEOUT_S)
        return original_run(*args, **kwargs)

    def pause_publish():
        if not publish_entered.is_set():
            publish_entered.set()
            assert release_publish.wait(_TIMEOUT_S)
        return original_publish()

    def record_claim(*args, **kwargs):
        claim_calls.append(descendant.sequence)
        return original_claim(*args, **kwargs)

    def record_cancel(*args, **kwargs):
        cancelled = original_cancel_async(*args, **kwargs)
        if cancelled:
            cancel_calls.append(descendant.sequence)
        return cancelled

    def record_release(token):
        result = original_release(token)
        if token.sequence == descendant.sequence:
            release_calls.append(token.sequence)
        return result

    monkeypatch.setattr(admission, "cancel", pause_cancel)
    monkeypatch.setattr(admission, "run", pause_run)
    monkeypatch.setattr(owner, "_publish_counted_start_request", pause_publish)
    monkeypatch.setattr(gate, "claim_async", record_claim)
    monkeypatch.setattr(gate, "cancel_async", record_cancel)
    monkeypatch.setattr(gate, "release", record_release)
    primary = RuntimeError("{} cancellation winner".format(async_kind))
    quarantiner, _, quarantine_errors, quarantine_done = _start(
        lambda: owner.force_quarantine(primary)
    )
    assert cancel_entered.wait(_TIMEOUT_S)

    def arm_and_start():
        owner.arm_completion()
        if async_kind == "ordinary":
            owner.start_counted_completion()

    armer, _, armer_errors, armer_done = _start(arm_and_start)
    assert publish_entered.wait(_TIMEOUT_S)
    recovery_wake = False

    try:
        release_cancel.set()
        _join(quarantiner, quarantine_done)
        release_publish.set()
        _join(armer, armer_done)
        if not run_entered.wait(_TIMEOUT_S):
            recovery_wake = True
            admission.wake()
        assert run_entered.wait(_TIMEOUT_S)
        release_run.set()
        assert owner._async_done.wait(_TIMEOUT_S)
        worker = owner._async_worker
        assert worker is not None
        worker.join(_TIMEOUT_S)
        assert not worker.is_alive()

        assert quarantine_errors == []
        assert armer_errors == []
        assert recovery_wake is False
        assert owner.state == "quarantined"
        assert owner.error is primary
        assert quarantined == [owner]
        assert claim_calls == []
        assert cancel_calls == [descendant.sequence]
        assert release_calls == []
        with gate._condition:
            assert gate._tokens[descendant.sequence].status == "released"
            assert not any(
                state.status == "active" for state in gate._tokens.values()
            )
    finally:
        release_cancel.set()
        release_publish.set()
        release_run.set()
        owner._async_requested.set()
        admission.wake()
        if quarantiner.is_alive():
            _join(quarantiner, quarantine_done)
        if armer.is_alive():
            _join(armer, armer_done)
        worker = owner._async_worker
        if worker is not None:
            worker.join(_TIMEOUT_S)
        with gate._condition:
            active = [
                state.token
                for state in gate._tokens.values()
                if state.status == "active"
            ]
        for token in active:
            try:
                gate.release(token)
            except BaseException:
                pass
        if transition is None:
            transition, elected = gate.begin_lease_close(epoch)
            assert elected is True
        with gate._condition:
            lease_phase = gate._leases[epoch].phase
        if lease_phase == "closing":
            gate.commit_lease_close(transition, lambda: None)
        runtime.close()


@pytest.mark.parametrize("async_kind", ("ordinary", "close_owned"))
@pytest.mark.parametrize(
    "start_mode",
    ("before_start", "after_start_before_claim", "after_claim"),
)
def test_counted_async_thread_start_is_transactional(
    monkeypatch,
    async_kind,
    start_mode,
):
    runtime = _runtime()
    gate = runtime._terminal_gate
    epoch, construction = gate.begin_lease(
        "{}_{}_construction".format(async_kind, start_mode)
    )
    gate.activate_lease(epoch, construction)
    gate.release(construction)
    transition = None
    if async_kind == "ordinary":
        parent = gate.admit_lease(epoch, "ordinary_start_parent")
    else:
        transition, elected = gate.begin_lease_close(epoch)
        assert elected is True
        parent = gate.admit_lease_close(transition, "close_start_parent")
    admission = _CountedAsyncAdmission(gate, parent, object())
    descendant = admission.token
    gate.release(parent)

    completion = _BlockingEvent()
    quarantined = []
    owner = AsyncResourceOwner(
        "compute",
        _async_admission=admission,
        quarantine=quarantined.append,
    )
    owner.mark_enqueued()
    owner.add_event(completion, completion=True)
    worker_entered = threading.Event()
    release_worker = threading.Event()
    claim_entered = threading.Event()
    workers = []
    cancel_calls = []
    claim_calls = []
    release_calls = []

    class StartFailure(BaseException):
        pass

    primary = StartFailure(
        "{} {} worker start failure".format(async_kind, start_mode)
    )
    original_worker_run = owner._run_counted_completion
    original_thread_start = threading.Thread.start
    original_cancel = gate.cancel_async
    original_claim = gate.claim_async
    original_release = gate.release

    if start_mode == "after_start_before_claim":

        def pause_worker(admitted):
            worker_entered.set()
            assert release_worker.wait(_TIMEOUT_S)
            return original_worker_run(admitted)

        monkeypatch.setattr(owner, "_run_counted_completion", pause_worker)
    elif start_mode == "after_claim":
        owner._async_requested.set()
        admission.wake()

    def record_cancel(*args, **kwargs):
        cancelled = original_cancel(*args, **kwargs)
        if cancelled:
            cancel_calls.append(descendant.sequence)
        return cancelled

    def record_claim(*args, **kwargs):
        claimed = original_claim(*args, **kwargs)
        claim_calls.append(descendant.sequence)
        claim_entered.set()
        return claimed

    def record_release(token):
        result = original_release(token)
        if token.sequence == descendant.sequence:
            release_calls.append(descendant.sequence)
        return result

    def fail_thread_start(worker):
        if worker.name != "renormalizer-async-completion":
            return original_thread_start(worker)
        workers.append(worker)
        if start_mode == "before_start":
            raise primary
        original_thread_start(worker)
        if start_mode == "after_start_before_claim":
            assert worker_entered.wait(_TIMEOUT_S)
        else:
            assert claim_entered.wait(_TIMEOUT_S)
        raise primary

    monkeypatch.setattr(gate, "cancel_async", record_cancel)
    monkeypatch.setattr(gate, "claim_async", record_claim)
    monkeypatch.setattr(gate, "release", record_release)
    monkeypatch.setattr(threading.Thread, "start", fail_thread_start)

    caller_error = None
    recovery_cancelled = False
    token_status = None
    done_before_recovery = None
    worker_state = None
    drain_errors = []
    try:
        try:
            owner.arm_completion()
        except BaseException as error:
            try:
                owner.fail(error)
            except BaseException as terminal_error:
                caller_error = terminal_error

        release_worker.set()
        completion.release.set()
        worker = workers[0]
        if worker.ident is not None:
            worker.join(_TIMEOUT_S)
            assert not worker.is_alive()
        with gate._condition:
            token_status = gate._tokens[descendant.sequence].status
        done_before_recovery = owner._async_done.is_set()
        worker_state = getattr(owner, "_async_worker_state", None)
        if token_status == "active":
            recovery_cancelled = admission.cancel()
            owner._async_done.set()
        if transition is None:
            transition, elected = gate.begin_lease_close(epoch)
            assert elected is True
        drain, _, drain_errors, drain_done = _start(
            lambda: gate.wait_for_lease_admissions(
                transition,
                _TIMEOUT_S,
            )
        )
        _join(drain, drain_done)
        gate.commit_lease_close(transition, lambda: None)
    finally:
        release_worker.set()
        completion.release.set()
        for worker in workers:
            if worker.ident is not None and worker.is_alive():
                worker.join(_TIMEOUT_S)
        with gate._condition:
            state = gate._tokens.get(descendant.sequence)
            active = state is not None and state.status == "active"
        if active:
            try:
                admission.cancel()
            except BaseException:
                pass
        if gate._live_epoch is not None:
            try:
                if transition is None:
                    transition, _ = gate.begin_lease_close(epoch)
                gate.commit_lease_close(transition, lambda: None)
            except BaseException:
                pass
        runtime.close()

    assert caller_error is primary
    assert recovery_cancelled is False
    assert token_status == "released"
    assert done_before_recovery is True
    assert owner.state == "quarantined"
    assert owner.error is primary
    assert quarantined == [owner]
    assert drain_errors == []
    assert gate._live_epoch is None
    if start_mode in {"before_start", "after_start_before_claim"}:
        assert worker_state == "start_failed"
        assert cancel_calls == [descendant.sequence]
        assert claim_calls == []
        assert release_calls == []
    else:
        assert worker_state == "terminal"
        assert cancel_calls == []
        assert claim_calls == [descendant.sequence]
        assert release_calls == [descendant.sequence]


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


def test_runtime_close_freezes_before_final_active_scheduler_snapshot(monkeypatch):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-runtime-close-final-scheduler-snapshot",
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    assert provider._active_lease is None

    gate = runtime._terminal_gate
    admit_entered = threading.Event()
    release_admit = threading.Event()
    original_admit = gate.admit_runtime

    def admit(operation):
        admit_entered.set()
        assert release_admit.wait(_TIMEOUT_S)
        return original_admit(operation)

    monkeypatch.setattr(gate, "admit_runtime", admit)
    closer, close_results, close_errors, close_done = _start(runtime.close)
    assert admit_entered.wait(_TIMEOUT_S)

    lease = provider.open_working_set(request, plan, store, receipt).__enter__()
    scheduler = lease.scheduler
    child = lease.acquire(_block_request(lease))
    event = _BlockingEvent()
    scheduler._event_factory = lambda: event
    child.close()
    assert event.recorded.wait(_TIMEOUT_S)
    parent = gate.admit_lease(lease._epoch, "held_before_runtime_freeze")

    try:
        release_admit.set()
        _wait_for_phase(runtime, _TerminalPhase.RUNTIME_CLOSING)
        with pytest.raises(RuntimeError, match="runtime is closing"):
            gate.admit_lease(lease._epoch, "after_runtime_freeze")
        with pytest.raises(RuntimeError, match="runtime is closing"):
            gate.spawn_async(parent, object())
        assert event.waiting.wait(_TIMEOUT_S)
        gate.release(parent)
        parent = None
        event.release.set()
        _join(closer, close_done)
        assert close_results == [None]
        assert close_errors == []
    finally:
        release_admit.set()
        if parent is not None:
            gate.release(parent)
        scheduler._start_counted_completions()
        event.release.set()
        if closer.is_alive():
            _join(closer, close_done)
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
    wrap(provider, "_reconcile_lease_cache", "cache_lifetime_reconcile")
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
            "cache_lifetime_reconcile",
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
    wrap(provider, "_reconcile_lease_cache", "cache_lifetime_reconcile")
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
        with runtime._terminal_gate._condition:
            abandoned_runtime_close = (
                runtime._terminal_gate._phase
                is _TerminalPhase.RUNTIME_CLOSING
                and terminal_done.is_set()
            )
        if abandoned_runtime_close and terminal_errors:
            raise terminal_errors[0]
        assert not abandoned_runtime_close, (
            "elected runtime close exited before commit",
            tuple(
                "".join(traceback.format_exception(error))
                for error in terminal_errors
            ),
            calls,
        )
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

    def observe(**kwargs):
        close_step_owners.append(threading.get_ident())
        close_step_entered.set()
        assert release_close_step.wait(_TIMEOUT_S)
        return original_observe(**kwargs)

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


def test_runtime_close_scheduler_start_baseexception_finishes_owner_and_joiner(
    monkeypatch,
):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-runtime-close-start-baseexception"
    )
    gate = runtime._terminal_gate
    scheduler = lease.scheduler
    pool = lease.pool
    cache = provider.cache
    start_entered = threading.Event()
    release_start = threading.Event()
    joiner_waiting = threading.Event()

    class SchedulerStartFailure(BaseException):
        pass

    primary = SchedulerStartFailure("runtime close scheduler start failed")

    def fail_start():
        start_entered.set()
        assert release_start.wait(_TIMEOUT_S)
        raise primary

    original_commit = gate.commit_runtime_close

    def commit(transition, finalizer):
        if (
            not isinstance(transition, _FatalTransition)
            and transition.owner_thread_id != threading.get_ident()
        ):
            joiner_waiting.set()
        return original_commit(transition, finalizer)

    monkeypatch.setattr(scheduler, "_start_counted_completions", fail_start)
    monkeypatch.setattr(gate, "commit_runtime_close", commit)
    owner, _, owner_errors, owner_done = _start(runtime.close)
    assert start_entered.wait(_TIMEOUT_S)
    joiner, _, joiner_errors, joiner_done = _start(runtime.close)
    assert joiner_waiting.wait(_TIMEOUT_S)

    try:
        release_start.set()
        _join(owner, owner_done)
        assert gate.phase is _TerminalPhase.RUNTIME_CLOSED
        _join(joiner, joiner_done)

        assert owner_errors == [primary]
        assert joiner_errors == [primary]
        assert gate._fatal_transition.primary is primary
        assert provider._active_lease is lease
        assert lease._closed is False
        assert lease.scheduler is scheduler
        assert lease.pool is pool
        assert provider.cache is cache
        retained = runtime._terminal_quarantine._resources
        assert all(
            any(resource is expected for resource in retained)
            for expected in (lease, scheduler, pool, cache)
        )
    finally:
        release_start.set()
        if gate.phase is _TerminalPhase.RUNTIME_CLOSING:
            runtime._enter_communicator_fatal(primary)
        for thread, done in ((owner, owner_done), (joiner, joiner_done)):
            if thread.is_alive():
                _join(thread, done)
        _close_case(runtime, store)


def test_lease_close_scheduler_start_baseexception_finishes_owner_and_joiner(
    monkeypatch,
):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-lease-close-start-baseexception"
    )
    gate = runtime._terminal_gate
    scheduler = lease.scheduler
    pool = lease.pool
    cache = provider.cache
    start_entered = threading.Event()
    release_start = threading.Event()
    joiner_waiting = threading.Event()

    class SchedulerStartFailure(BaseException):
        pass

    primary = SchedulerStartFailure("lease close scheduler start failed")

    def fail_start():
        start_entered.set()
        assert release_start.wait(_TIMEOUT_S)
        raise primary

    original_wait = gate.wait_for_lease_closed

    def wait_for_close(*args, **kwargs):
        joiner_waiting.set()
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(scheduler, "_start_counted_completions", fail_start)
    monkeypatch.setattr(gate, "wait_for_lease_closed", wait_for_close)
    owner, _, owner_errors, owner_done = _start(lease.close)
    assert start_entered.wait(_TIMEOUT_S)
    joiner, _, joiner_errors, joiner_done = _start(lease.close)
    assert joiner_waiting.wait(_TIMEOUT_S)

    try:
        release_start.set()
        _join(owner, owner_done)
        with gate._condition:
            assert gate._leases[lease._epoch].phase == "fatal_retained"
        _join(joiner, joiner_done)

        assert owner_errors == [primary]
        assert joiner_errors == [primary]
        assert gate.phase is _TerminalPhase.FATAL_PUBLISHED
        assert gate._fatal_transition.primary is primary
        assert provider._active_lease is lease
        assert lease._closed is False
        assert lease.scheduler is scheduler
        assert lease.pool is pool
        assert provider.cache is cache
        retained = runtime._terminal_quarantine._resources
        assert all(
            any(resource is expected for resource in retained)
            for expected in (lease, scheduler, pool, cache)
        )
    finally:
        release_start.set()
        with gate._condition:
            phase = gate._leases[lease._epoch].phase
        if phase == "closing":
            runtime._enter_communicator_fatal(primary)
        for thread, done in ((owner, owner_done), (joiner, joiner_done)):
            if thread.is_alive():
                _join(thread, done)
        _close_case(runtime, store)


@pytest.mark.parametrize("close_kind", ("lease", "runtime"))
@pytest.mark.parametrize(
    "failure_boundary",
    ("before_body", "in_body", "after_release"),
)
def test_elected_close_total_guard_covers_every_post_election_boundary(
    monkeypatch,
    close_kind,
    failure_boundary,
):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-total-{}-{}".format(close_kind, failure_boundary)
    )
    gate = runtime._terminal_gate
    entered = threading.Event()
    release_failure = threading.Event()
    joiner_waiting = threading.Event()

    class CloseBoundaryFailure(BaseException):
        pass

    primary = CloseBoundaryFailure(
        "{} close {} failure".format(close_kind, failure_boundary)
    )

    def fail():
        entered.set()
        assert release_failure.wait(_TIMEOUT_S)
        raise primary

    if failure_boundary == "before_body":
        monkeypatch.setattr(lease.scheduler, "_start_counted_completions", fail)
    elif failure_boundary == "in_body":
        if close_kind == "lease":
            monkeypatch.setattr(lease, "_observe_peaks", lambda **_kwargs: fail())
        else:
            monkeypatch.setattr(
                provider, "_close_for_runtime", lambda **_kwargs: fail()
            )
    else:
        target_operation = (
            "observe_peaks" if close_kind == "lease" else "provider_close"
        )
        original_release = gate.release
        failed = [False]

        def fail_after_release(token):
            if token.operation == target_operation and not failed[0]:
                failed[0] = True
                original_release(token)
                return fail()
            return original_release(token)

        monkeypatch.setattr(gate, "release", fail_after_release)

    if close_kind == "lease":
        original_join = gate.wait_for_lease_closed

        def observe_join(*args, **kwargs):
            joiner_waiting.set()
            return original_join(*args, **kwargs)

        monkeypatch.setattr(gate, "wait_for_lease_closed", observe_join)
        close = lease.close
    else:
        original_drain = gate._drain_runtime_close
        original_commit = gate.commit_runtime_close

        def observe_drain(transition):
            if (
                not isinstance(transition, _FatalTransition)
                and transition.owner_thread_id != threading.get_ident()
            ):
                joiner_waiting.set()
            return original_drain(transition)

        def observe_join(transition, finalizer):
            if (
                not isinstance(transition, _FatalTransition)
                and transition.owner_thread_id != threading.get_ident()
            ):
                joiner_waiting.set()
            return original_commit(transition, finalizer)

        monkeypatch.setattr(gate, "_drain_runtime_close", observe_drain)
        monkeypatch.setattr(gate, "commit_runtime_close", observe_join)
        close = runtime.close

    owner, _, owner_errors, owner_done = _start(close)
    assert entered.wait(_TIMEOUT_S)
    joiner, _, joiner_errors, joiner_done = _start(close)
    assert joiner_waiting.wait(_TIMEOUT_S)
    abandoned = False

    try:
        release_failure.set()
        _join(owner, owner_done)
        with gate._condition:
            lease_phase = gate._leases[lease._epoch].phase
            abandoned = (
                gate._phase is _TerminalPhase.RUNTIME_CLOSING
                if close_kind == "runtime"
                else lease_phase == "closing"
            )
        if abandoned:
            runtime._enter_communicator_fatal(primary)
        _join(joiner, joiner_done)
    finally:
        release_failure.set()
        if owner.is_alive():
            _join(owner, owner_done)
        with gate._condition:
            lease_phase = gate._leases[lease._epoch].phase
            needs_recovery = (
                gate._phase is _TerminalPhase.RUNTIME_CLOSING
                or lease_phase == "closing"
            )
        if needs_recovery:
            runtime._enter_communicator_fatal(primary)
        if joiner.is_alive():
            _join(joiner, joiner_done)
        _close_case(runtime, store)

    assert abandoned is False
    assert owner_errors == [primary]
    assert joiner_errors == [primary]
    assert gate._fatal_transition.primary is primary
    with gate._condition:
        assert not any(
            state.status == "active" for state in gate._tokens.values()
        )
        if close_kind == "runtime":
            assert gate._runtime_close_transition.owner_thread_id == owner.ident
        else:
            assert gate._leases[lease._epoch].transition.owner_thread_id == owner.ident
    if close_kind == "runtime":
        assert gate.phase is _TerminalPhase.RUNTIME_CLOSED
    else:
        assert lease_phase == "fatal_retained"
    if close_kind == "lease" or failure_boundary == "before_body":
        assert provider._active_lease is lease
        assert lease._closed is False


@pytest.mark.parametrize("close_kind", ("lease", "runtime"))
def test_elected_close_total_guard_includes_reference_finalizer_and_commit(
    monkeypatch,
    close_kind,
):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-finalizer-{}".format(close_kind)
    )
    gate = runtime._terminal_gate
    finalizer_entered = threading.Event()
    release_finalizer = threading.Event()
    joiner_started = threading.Event()
    finalizer_calls = []

    class FinalizerFailure(BaseException):
        pass

    primary = FinalizerFailure("{} reference finalizer failed".format(close_kind))
    if close_kind == "lease":

        def fail_finalizer(*_args, **_kwargs):
            finalizer_calls.append("lease")
            finalizer_entered.set()
            assert release_finalizer.wait(_TIMEOUT_S)
            raise primary

        monkeypatch.setattr(lease, "_commit_close_references", fail_finalizer)
        close = lease.close
    else:
        original_clear = runtime._clear_runtime_references

        def fail_once(*args, **kwargs):
            finalizer_calls.append("runtime")
            if len(finalizer_calls) == 1:
                finalizer_entered.set()
                assert release_finalizer.wait(_TIMEOUT_S)
                raise primary
            return original_clear(*args, **kwargs)

        monkeypatch.setattr(runtime, "_clear_runtime_references", fail_once)
        close = runtime.close

    owner, _, owner_errors, owner_done = _start(close)
    assert finalizer_entered.wait(_TIMEOUT_S)

    def join_close():
        joiner_started.set()
        return close()

    joiner, _, joiner_errors, joiner_done = _start(join_close)
    assert joiner_started.wait(_TIMEOUT_S)

    try:
        release_finalizer.set()
        _join(owner, owner_done)
        _join(joiner, joiner_done)
    finally:
        release_finalizer.set()
        if owner.is_alive():
            _join(owner, owner_done)
        if joiner.is_alive():
            _join(joiner, joiner_done)
        _close_case(runtime, store)

    assert owner_errors == [primary]
    assert joiner_errors == [primary]
    with gate._condition:
        assert not any(
            state.status == "active" for state in gate._tokens.values()
        )
    if close_kind == "lease":
        assert finalizer_calls == ["lease"]
        assert gate._fatal_transition.primary is primary
        assert gate._leases[lease._epoch].phase == "fatal_retained"
        assert provider._active_lease is lease
        assert lease._closed is False
    else:
        assert finalizer_calls == ["runtime", "runtime"]
        assert gate._fatal_transition is None
        assert gate.phase is _TerminalPhase.RUNTIME_CLOSED
        assert runtime._terminal_error is primary


@pytest.mark.parametrize("catch_layer", ("transition", "legacy", "outer"))
def test_publication_failure_terminalizer_marks_before_diagnostics(
    monkeypatch,
    catch_layer,
):
    runtime = _runtime()
    gate = runtime._terminal_gate
    primary = RuntimeError("{} publication primary".format(catch_layer))

    class PublishFailure(BaseException):
        pass

    class DiagnosticFailure(BaseException):
        pass

    publication_failure = PublishFailure(
        "{} publication failure".format(catch_layer)
    )
    diagnostic_failure = DiagnosticFailure(
        "{} diagnostic failure".format(catch_layer)
    )
    diagnostic_calls = []
    marker_observations = []
    close_commit_entered = threading.Event()

    def fail_diagnostic(error):
        diagnostic_calls.append(error)
        with gate._condition:
            transition = gate._fatal_transition
            marker_observations.append(
                transition is not None
                and gate._fatal_publication_failure is transition.primary
            )
        raise diagnostic_failure

    monkeypatch.setattr(
        runtime.collective,
        "_record_fatal_secondary",
        fail_diagnostic,
        raising=False,
    )
    if catch_layer == "transition":

        def fail_transition(*_args, **_kwargs):
            raise publication_failure

        monkeypatch.setattr(gate, "wait_for_admissions", fail_transition)
    elif catch_layer == "legacy":

        def fail_legacy(*_args, **_kwargs):
            raise publication_failure

        monkeypatch.setattr(
            runtime.collective,
            "_publish_communicator_fatal",
            fail_legacy,
        )
    else:
        original_begin = runtime._begin_communicator_fatal

        def fail_outer(*args, **kwargs):
            original_begin(*args, **kwargs)
            raise publication_failure

        monkeypatch.setattr(runtime, "_begin_communicator_fatal", fail_outer)

    original_commit = gate.commit_runtime_close

    def observe_commit(transition, finalizer):
        if isinstance(transition, _FatalTransition):
            close_commit_entered.set()
        return original_commit(transition, finalizer)

    monkeypatch.setattr(gate, "commit_runtime_close", observe_commit)
    publisher_error = None
    closer = None
    close_errors = []
    marker_before_recovery = False
    phase_before_recovery = None
    try:
        try:
            runtime._enter_communicator_fatal(primary)
        except BaseException as error:
            publisher_error = error
        closer, _, close_errors, close_done = _start(runtime.close)
        assert close_commit_entered.wait(_TIMEOUT_S)
        with gate._condition:
            transition = gate._fatal_transition
            marker_before_recovery = (
                transition is not None
                and gate._fatal_publication_failure is transition.primary
            )
            phase_before_recovery = gate._phase
        if marker_before_recovery:
            _join(closer, close_done)
    finally:
        with gate._condition:
            transition = gate._fatal_transition
            marker_recorded = (
                transition is not None
                and gate._fatal_publication_failure is transition.primary
            )
        if transition is not None and not marker_recorded:
            gate._fail_fatal_publication(transition, transition.primary)
        if closer is not None and closer.is_alive():
            _join(closer, close_done)

    assert publisher_error is primary
    assert marker_before_recovery is True
    assert phase_before_recovery is _TerminalPhase.RUNTIME_CLOSED
    assert close_errors == [primary]
    assert marker_observations == [True]
    assert diagnostic_calls == [publication_failure]
    assert publication_failure in runtime._terminal_secondary_errors
    assert diagnostic_failure in runtime._terminal_secondary_errors


@pytest.mark.parametrize("marker_preinstalled", (False, True))
def test_publication_failure_terminalizer_is_idempotent(
    monkeypatch,
    marker_preinstalled,
):
    runtime = _runtime()
    gate = runtime._terminal_gate
    primary = RuntimeError("terminalizer primary")
    secondary = RuntimeError("terminalizer secondary")
    transition = gate.begin_fatal(primary)
    calls = []
    marker_calls = []
    original_fail = gate._fail_fatal_publication

    def record_marker(selected, failure):
        marker_calls.append((selected, failure))
        return original_fail(selected, failure)

    monkeypatch.setattr(gate, "_fail_fatal_publication", record_marker)

    def record_secondary(error):
        calls.append(error)
        assert gate._fatal_publication_failure is primary

    monkeypatch.setattr(
        runtime.collective,
        "_record_fatal_secondary",
        record_secondary,
        raising=False,
    )
    if marker_preinstalled:
        gate._fail_fatal_publication(transition, primary)

    first = runtime._terminalize_fatal_publication_failure(
        transition,
        secondary,
    )
    second = runtime._terminalize_fatal_publication_failure(
        transition,
        secondary,
    )

    assert first is primary
    assert second is primary
    assert gate._fatal_publication_failure is primary
    assert calls == [secondary]
    assert marker_calls == [(transition, primary)]


def test_structured_publication_hard_exit_remains_owned_by_collective():
    class StructuredHardExit(BaseException):
        pass

    hard_exit = StructuredHardExit("structured fatal hard exit")

    class StructuredCollective(_LoopbackCollective):
        def __init__(self):
            super().__init__(1)
            self._fatal_publication_local = threading.local()
            self.runtime = None

        def _begin_fatal_publication(self, *_args, **_kwargs):
            raise AssertionError("structured presence sentinel must not run")

        def _publish_communicator_fatal(self, error, **kwargs):
            transition = kwargs["transition"]
            self.runtime._terminal_gate._fail_fatal_publication(
                transition,
                transition.primary,
            )
            raise hard_exit

    collective = StructuredCollective()
    runtime = _runtime(collective=collective)
    collective.runtime = runtime
    primary = RuntimeError("structured publication primary")

    with pytest.raises(StructuredHardExit) as caught:
        runtime._enter_communicator_fatal(primary)

    assert caught.value is hard_exit
    assert runtime._terminal_gate._fatal_publication_failure is primary


@pytest.mark.parametrize(
    "publication_failure",
    ("retention", "collective_callback"),
)
def test_elected_close_secondary_publication_failure_has_bounded_outcome(
    monkeypatch,
    publication_failure,
):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-publication-failure-{}".format(publication_failure)
    )
    gate = runtime._terminal_gate
    start_entered = threading.Event()
    release_start = threading.Event()
    joiner_waiting = threading.Event()
    failure_calls = []
    destructive_calls = []

    class SchedulerPrimary(BaseException):
        pass

    class PublicationSecondary(BaseException):
        pass

    primary = SchedulerPrimary("runtime close scheduler primary")
    secondary = PublicationSecondary(
        "{} publication secondary".format(publication_failure)
    )

    def fail_start():
        start_entered.set()
        assert release_start.wait(_TIMEOUT_S)
        raise primary

    def fail_publication(*_args, **_kwargs):
        failure_calls.append(publication_failure)
        raise secondary

    monkeypatch.setattr(lease.scheduler, "_start_counted_completions", fail_start)
    if publication_failure == "retention":
        monkeypatch.setattr(
            provider,
            "_retain_transition_resources",
            fail_publication,
        )
    else:
        monkeypatch.setattr(
            runtime.collective,
            "_publish_communicator_fatal",
            fail_publication,
        )

    monkeypatch.setattr(
        provider,
        "close",
        lambda **_kwargs: destructive_calls.append("provider_close"),
    )
    monkeypatch.setattr(
        runtime.collective,
        "close",
        lambda: destructive_calls.append("collective_close"),
    )
    original_drain = gate._drain_runtime_close

    def observe_joiner(transition):
        if (
            not isinstance(transition, _FatalTransition)
            and transition.owner_thread_id != threading.get_ident()
        ):
            joiner_waiting.set()
        return original_drain(transition)

    monkeypatch.setattr(gate, "_drain_runtime_close", observe_joiner)
    owner, _, owner_errors, owner_done = _start(runtime.close)
    assert start_entered.wait(_TIMEOUT_S)
    joiner, _, joiner_errors, joiner_done = _start(runtime.close)
    assert joiner_waiting.wait(_TIMEOUT_S)
    marker_before_recovery = False
    phase_before_recovery = None

    try:
        release_start.set()
        _join(owner, owner_done)
        with gate._condition:
            transition = gate._fatal_transition
            marker_before_recovery = (
                transition is not None
                and gate._fatal_publication_failure is transition.primary
            )
            phase_before_recovery = gate._phase
        if transition is not None and not marker_before_recovery:
            gate._fail_fatal_publication(transition, transition.primary)
        _join(joiner, joiner_done)
    finally:
        release_start.set()
        if owner.is_alive():
            _join(owner, owner_done)
        with gate._condition:
            transition = gate._fatal_transition
            marker_recorded = (
                transition is not None
                and gate._fatal_publication_failure is transition.primary
            )
        if transition is not None and not marker_recorded:
            gate._fail_fatal_publication(transition, transition.primary)
        if joiner.is_alive():
            _join(joiner, joiner_done)
        _close_case(runtime, store)

    assert marker_before_recovery is True
    assert phase_before_recovery is _TerminalPhase.RUNTIME_CLOSED
    assert owner_errors == [primary]
    assert joiner_errors == [primary]
    assert gate._fatal_transition.primary is primary
    assert gate._fatal_publication_failure is primary
    assert gate._fatal_snapshot is _MISSING
    assert secondary in runtime._terminal_secondary_errors
    assert failure_calls == [publication_failure]
    assert destructive_calls == []
    with gate._condition:
        assert not any(
            state.status == "active" for state in gate._tokens.values()
        )


def test_elected_lease_close_secondary_retention_failure_wakes_joiner(
    monkeypatch,
):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-lease-publication-failure"
    )
    gate = runtime._terminal_gate
    start_entered = threading.Event()
    release_start = threading.Event()
    joiner_waiting = threading.Event()

    class LeaseClosePrimary(BaseException):
        pass

    class RetentionSecondary(BaseException):
        pass

    primary = LeaseClosePrimary("lease close scheduler primary")
    secondary = RetentionSecondary("lease close retention secondary")

    def fail_start():
        start_entered.set()
        assert release_start.wait(_TIMEOUT_S)
        raise primary

    def fail_retention(_lease):
        raise secondary

    original_join = gate.wait_for_lease_closed

    def observe_joiner(*args, **kwargs):
        joiner_waiting.set()
        return original_join(*args, **kwargs)

    monkeypatch.setattr(lease.scheduler, "_start_counted_completions", fail_start)
    monkeypatch.setattr(provider, "_retain_transition_resources", fail_retention)
    monkeypatch.setattr(gate, "wait_for_lease_closed", observe_joiner)
    owner, _, owner_errors, owner_done = _start(lease.close)
    assert start_entered.wait(_TIMEOUT_S)
    joiner, _, joiner_errors, joiner_done = _start(lease.close)
    assert joiner_waiting.wait(_TIMEOUT_S)
    phase_before_recovery = None

    try:
        release_start.set()
        _join(owner, owner_done)
        _join(joiner, joiner_done)
        phase_before_recovery = gate.phase
    finally:
        release_start.set()
        if owner.is_alive():
            _join(owner, owner_done)
        if joiner.is_alive():
            _join(joiner, joiner_done)
        if gate.phase is not _TerminalPhase.RUNTIME_CLOSED:
            try:
                runtime._finish_failed_fatal_runtime_close(
                    gate._fatal_transition
                )
            except BaseException:
                pass
        _close_case(runtime, store)

    assert phase_before_recovery is _TerminalPhase.RUNTIME_CLOSED
    assert owner_errors == [primary]
    assert joiner_errors == [primary]
    assert gate._fatal_transition.primary is primary
    assert gate._fatal_publication_failure is primary
    assert gate._fatal_snapshot is _MISSING
    assert secondary in runtime._terminal_secondary_errors


def test_elected_lease_close_post_publication_retention_failure_keeps_primary(
    monkeypatch,
):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-lease-post-publication-retention-failure"
    )
    gate = runtime._terminal_gate
    start_entered = threading.Event()
    release_start = threading.Event()
    joiner_waiting = threading.Event()
    owner_ident = []
    retention_calls = []

    class LeaseClosePrimary(BaseException):
        pass

    class RetentionSecondary(BaseException):
        pass

    primary = LeaseClosePrimary("lease close scheduler primary")
    secondary = RetentionSecondary("post-publication retention secondary")

    def fail_start():
        owner_ident.append(threading.get_ident())
        start_entered.set()
        assert release_start.wait(_TIMEOUT_S)
        raise primary

    original_retention = provider._retain_terminal_lease

    def fail_owner_retention(retained_lease, error):
        retention_calls.append(threading.get_ident())
        if threading.get_ident() == owner_ident[0]:
            raise secondary
        return original_retention(retained_lease, error)

    original_join = gate.wait_for_lease_closed

    def observe_joiner(*args, **kwargs):
        joiner_waiting.set()
        return original_join(*args, **kwargs)

    monkeypatch.setattr(lease.scheduler, "_start_counted_completions", fail_start)
    monkeypatch.setattr(provider, "_retain_terminal_lease", fail_owner_retention)
    monkeypatch.setattr(gate, "wait_for_lease_closed", observe_joiner)
    owner, _, owner_errors, owner_done = _start(lease.close)
    assert start_entered.wait(_TIMEOUT_S)
    joiner, _, joiner_errors, joiner_done = _start(lease.close)
    assert joiner_waiting.wait(_TIMEOUT_S)
    phase_before_cleanup = None

    try:
        release_start.set()
        _join(owner, owner_done)
        _join(joiner, joiner_done)
        phase_before_cleanup = gate.phase
    finally:
        release_start.set()
        if owner.is_alive():
            _join(owner, owner_done)
        if joiner.is_alive():
            _join(joiner, joiner_done)
        _close_case(runtime, store)

    assert owner_errors == [primary]
    assert joiner_errors == [primary]
    assert gate._fatal_transition.primary is primary
    assert phase_before_cleanup is _TerminalPhase.FATAL_PUBLISHED
    assert secondary in runtime._terminal_secondary_errors
    assert retention_calls.count(owner_ident[0]) == 1
    with gate._condition:
        assert not any(
            state.status == "active" for state in gate._tokens.values()
        )


def test_fatal_preempts_lease_close_owner_and_joiner_with_same_primary(monkeypatch):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-fatal-close-joiner"
    )
    close_step_entered = threading.Event()
    release_close_step = threading.Event()
    joiner_waiting = threading.Event()
    original_observe = lease._observe_peaks
    original_wait = runtime._terminal_gate.wait_for_lease_closed

    def observe(**kwargs):
        close_step_entered.set()
        assert release_close_step.wait(_TIMEOUT_S)
        return original_observe(**kwargs)

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


@pytest.mark.parametrize("lease_count", (1, 2))
def test_fatal_between_healthy_leases_retains_exact_cache_snapshot(
    monkeypatch,
    lease_count,
):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-between-leases-cache-{}".format(lease_count),
    )
    provider = _provider(runtime, request)
    pointers = []

    try:
        for _ in range(lease_count):
            receipt = runtime.preflight_residency(request, plan)
            with provider.open_working_set(
                request,
                plan,
                store,
                receipt,
            ) as lease:
                with lease.acquire(_block_request(lease)) as child:
                    pointers.append(
                        child.bindings.arrays["input_0"].__array_interface__["data"][
                            0
                        ]
                    )
            assert provider._active_lease is None

        if lease_count == 2:
            assert pointers[1] == pointers[0]
            assert provider.metrics.cache_hits == 1

        cache = provider.cache
        allocations = cache.allocation_records
        assert sum(record.capacity_bytes for record in allocations) == 128
        expected = {record.identity: record for record in allocations}
        assert len(expected) == len(allocations)

        queries = []

        def allocation_records_query(_cache):
            queries.append(True)
            raise AssertionError("fatal retention queried the live cache")

        monkeypatch.setattr(
            DeviceTensorCache,
            "allocation_records",
            property(allocation_records_query),
        )
        primary = RuntimeError("fatal between healthy leases")
        assert runtime._enter_communicator_fatal(primary) is primary
        assert queries == []

        retained = runtime._terminal_quarantine._cache_allocations
        assert set(retained) == set(expected)
        assert all(retained[identity] is record for identity, record in expected.items())
        state = runtime.resource_state()
        assert state["cache_bytes"] == 128
        assert state["quarantined_bytes"] == 128

        with pytest.raises(RuntimeError) as caught:
            runtime.close()
        assert caught.value is primary
        assert queries == []
        assert runtime._terminal_quarantine._cache_allocations == retained
        assert runtime.resource_state() == state
    finally:
        _close_case(runtime, store)


def test_second_lease_eviction_reconciles_persistent_cache_identity(monkeypatch):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-between-leases-cache-eviction",
    )
    provider = _provider(runtime, request)
    allocated = []
    original_allocator = provider._cache_allocator

    def retain_allocations(spec):
        allocation = original_allocator(spec)
        allocated.append(allocation)
        return allocation

    monkeypatch.setattr(provider, "_cache_allocator", retain_allocations)
    try:
        first_receipt = runtime.preflight_residency(request, plan)
        with provider.open_working_set(
            request,
            plan,
            store,
            first_receipt,
        ) as first:
            with first.acquire(_block_request(first)):
                pass
        first_records = {
            record.identity: record for record in provider.cache.allocation_records
        }
        assert sum(record.capacity_bytes for record in first_records.values()) == 128

        refs = dict(request.host_refs)
        updated_input = store.update(
            "input_0",
            np.arange(16, dtype=np.float64).reshape(4, 4) + 100,
            expected_version=refs["input_0"].version,
        )
        second_request = replace(
            request,
            host_refs={"input_0": updated_input, "output": refs["output"]},
            store_snapshots=(store.snapshot(),) * runtime.world_size,
        )
        second_plan = ResidencyPlanner().plan(second_request)
        second_receipt = runtime.preflight_residency(second_request, second_plan)
        with provider.open_working_set(
            second_request,
            second_plan,
            store,
            second_receipt,
        ) as second:
            with second.acquire(_block_request(second)):
                pass

        current_records = {
            record.identity: record for record in provider.cache.allocation_records
        }
        assert sum(record.capacity_bytes for record in current_records.values()) == 128
        assert set(first_records).isdisjoint(current_records)
        persistent = {
            record.identity: record
            for record in provider._persistent_resources.cache_allocations
        }
        assert persistent == current_records

        def allocation_records_query(_cache):
            raise AssertionError("fatal retention queried an evicted cache")

        monkeypatch.setattr(
            DeviceTensorCache,
            "allocation_records",
            property(allocation_records_query),
        )
        primary = RuntimeError("fatal after sequential cache eviction")
        assert runtime._enter_communicator_fatal(primary) is primary
        retained = runtime._terminal_quarantine._cache_allocations
        assert retained == current_records
        assert set(retained).isdisjoint(first_records)
        assert runtime.resource_state()["cache_bytes"] == 128
    finally:
        _close_case(runtime, store)


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


def test_resource_state_retries_between_lease_to_open_scope_change(monkeypatch):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-state-between-to-open",
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    gate = runtime._terminal_gate
    original_admit = gate.admit_runtime
    original_state = provider.runtime_resource_state
    opened = []
    observed = []

    def admit(operation):
        if operation == "resource_state" and not opened:
            opened.append(
                provider.open_working_set(request, plan, store, receipt).__enter__()
            )
        return original_admit(operation)

    def resource_state(**kwargs):
        observed.append((_current_token(runtime), gate._live_epoch))
        return original_state(**kwargs)

    monkeypatch.setattr(gate, "admit_runtime", admit)
    monkeypatch.setattr(provider, "runtime_resource_state", resource_state)
    try:
        state = runtime.resource_state()
        lease = opened[0]
        assert state["active_leases"] == 1
        assert len(observed) == 1
        token, live_epoch = observed[0]
        _assert_token(token, scope="lease", epoch=lease._epoch)
        assert live_epoch == lease._epoch
    finally:
        _close_case(runtime, store)


def test_resource_state_retries_open_lease_to_between_scope_change(monkeypatch):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-state-open-to-between"
    )
    gate = runtime._terminal_gate
    original_admit = gate.admit_lease
    original_state = provider.runtime_resource_state
    closed = []
    observed = []

    def admit(epoch, operation):
        if operation == "resource_state" and not closed:
            lease.close()
            closed.append(True)
        return original_admit(epoch, operation)

    def resource_state(**kwargs):
        observed.append((_current_token(runtime), gate._live_epoch))
        return original_state(**kwargs)

    monkeypatch.setattr(gate, "admit_lease", admit)
    monkeypatch.setattr(provider, "runtime_resource_state", resource_state)
    try:
        state = runtime.resource_state()
        assert state["active_leases"] == 0
        assert len(observed) == 1
        token, live_epoch = observed[0]
        _assert_token(token, scope="runtime", epoch=None)
        assert live_epoch is None
    finally:
        _close_case(runtime, store)


@pytest.mark.parametrize("initial_phase", ("fatal_pending", "fatal_published"))
def test_fatal_runtime_close_retains_intact_resources_without_callbacks(
    monkeypatch,
    initial_phase,
):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-fatal-runtime-intact-{}".format(initial_phase)
    )
    gate = runtime._terminal_gate
    reservation = lease._store_reservation
    cache = provider.cache
    pool = lease.pool
    scheduler = lease.scheduler
    status_workspace = lease._status_workspace
    collective = runtime.collective
    primary = RuntimeError("fatal runtime close retains intact resources")
    held = None
    fatal = None
    fatal_results = []
    fatal_errors = []
    fatal_done = None
    if initial_phase == "fatal_pending":
        held = gate.admit_lease(lease._epoch, "hold_fatal_pending_close")
        fatal, fatal_results, fatal_errors, fatal_done = _start(
            lambda: runtime._enter_communicator_fatal(primary)
        )
        _wait_for_phase(runtime, _TerminalPhase.FATAL_PENDING)
    else:
        assert runtime._enter_communicator_fatal(primary) is primary
        assert gate.phase is _TerminalPhase.FATAL_PUBLISHED

    callbacks = []
    close_entered = threading.Event()

    def guard(context, obj, attribute, name):
        original = getattr(obj, attribute)

        def call(*args, **kwargs):
            callbacks.append((name, gate.phase))
            return original(*args, **kwargs)

        context.setattr(obj, attribute, call)

    try:
        with monkeypatch.context() as sentinels:
            guard(
                sentinels,
                provider,
                "_close_for_runtime",
                "provider_close",
            )
            guard(sentinels, lease, "close", "lease_close")
            guard(sentinels, reservation, "close", "reservation_close")
            guard(sentinels, cache, "close", "cache_close")
            guard(sentinels, pool, "close", "pool_close")
            guard(sentinels, scheduler, "close", "scheduler_close")
            guard(sentinels, status_workspace, "close", "status_close")
            guard(sentinels, collective, "close", "collective_close")
            original_commit = runtime._commit_fatal_runtime_close

            def commit(*args, **kwargs):
                close_entered.set()
                return original_commit(*args, **kwargs)

            sentinels.setattr(runtime, "_commit_fatal_runtime_close", commit)
            closer, close_results, close_errors, close_done = _start(runtime.close)
            assert close_entered.wait(_TIMEOUT_S)
            if held is not None:
                gate.release(held)
                held = None
            _join(closer, close_done)
            if fatal is not None:
                _join(fatal, fatal_done)

            assert close_results == []
            assert close_errors == [primary]
            assert callbacks == []
            assert runtime.collective is collective

        assert reservation._closed is False
        assert lease._store_reservation is reservation
        assert provider._active_lease is lease
        assert lease._closed is False
        assert lease._provider is provider
        assert provider.cache is cache
        assert lease.pool is pool
        assert lease.scheduler is scheduler
        assert lease._status_workspace is status_workspace
        assert cache._closed is False
        assert pool._closed is False
        assert scheduler._closed is False
        assert status_workspace._closed is False
        if fatal is not None:
            assert fatal_results == [primary]
            assert fatal_errors == []
    finally:
        if held is not None:
            gate.release(held)
        if fatal is not None and fatal.is_alive():
            _join(fatal, fatal_done)
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

        def _close_for_runtime(self, *args, **kwargs):
            observed.append(("provider_close", _current_token(runtime)))

    provider = Provider()
    runtime._active_provider = provider
    original_close = runtime.collective.close
    original_freeze = runtime._terminal_gate._freeze_runtime_close
    original_commit = runtime._terminal_gate.commit_runtime_close

    def collective_close():
        observed.append(("collective_close", _current_token(runtime)))
        return original_close()

    def freeze(token):
        transition, elected = original_freeze(token)
        transitions.append(transition)
        return transition, elected

    def commit(transition, finalizer):
        assert _current_token(runtime) is None
        return original_commit(transition, finalizer)

    monkeypatch.setattr(runtime.collective, "close", collective_close)
    monkeypatch.setattr(runtime._terminal_gate, "_freeze_runtime_close", freeze)
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
    class Provider:
        _active_lease = None
        _terminal_error = None

        def _retain_transition_resources(self, lease):
            assert lease is None

        def _close_for_runtime(self, **kwargs):
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
    assert runtime._closed is True
    assert gate.phase is _TerminalPhase.RUNTIME_CLOSED


def test_fatal_preempts_runtime_close_begin_and_joins_commit(monkeypatch):
    runtime = _runtime(collective=_LoopbackCollective(1))
    begin_entered = threading.Event()
    release_begin = threading.Event()
    gate = runtime._terminal_gate
    original_freeze = gate._freeze_runtime_close

    def freeze(token):
        begin_entered.set()
        assert release_begin.wait(_TIMEOUT_S)
        return original_freeze(token)

    monkeypatch.setattr(gate, "_freeze_runtime_close", freeze)
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
    assert collective_close_calls == []
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


def _repair_test_construction_state(gate):
    """Bound RED probes without relying on the production repair under test."""
    with gate._condition:
        active = [
            state.token
            for state in gate._tokens.values()
            if state.status == "active"
            and state.token.thread_id == threading.get_ident()
        ]
    for token in active:
        gate.release(token)
    with gate._condition:
        epoch = gate._live_epoch
        phase = None if epoch is None else gate._leases[epoch].phase
    if epoch is None or phase in {"closed", "fatal_retained"}:
        return
    transition, elected = gate.begin_lease_close(epoch)
    if elected:
        gate.wait_for_lease_admissions(transition, _TIMEOUT_S)
        gate.commit_lease_close(transition, lambda: None)


@pytest.mark.parametrize(
    "failure_boundary",
    (
        "begin_before",
        "token_before",
        "token_after",
        "begin_after",
        "validator",
    ),
)
def test_construction_transaction_owns_pre_resource_failure(
    monkeypatch,
    failure_boundary,
):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-total-construction-{}".format(failure_boundary),
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    gate = runtime._terminal_gate
    transactions = []
    original_prepare = gate._prepare_lease_construction

    def capture_transaction(*args, **kwargs):
        transaction = original_prepare(*args, **kwargs)
        transactions.append(transaction)
        return transaction

    monkeypatch.setattr(
        gate, "_prepare_lease_construction", capture_transaction
    )

    class PreResourceFailure(RuntimeError):
        pass

    failure = PreResourceFailure(failure_boundary)
    if failure_boundary in {"begin_before", "begin_after"}:
        original_begin = gate.begin_lease

        def fail_at_begin(*args, **kwargs):
            if failure_boundary == "begin_after":
                original_begin(*args, **kwargs)
            raise failure

        monkeypatch.setattr(gate, "begin_lease", fail_at_begin)
    elif failure_boundary in {"token_before", "token_after"}:
        original_new_token = gate._new_token

        def fail_at_token(*args, **kwargs):
            if failure_boundary == "token_after":
                original_new_token(*args, **kwargs)
            raise failure

        monkeypatch.setattr(gate, "_new_token", fail_at_token)
    else:

        def fail_validator(*_args, **_kwargs):
            assert gate._live_epoch is not None
            raise failure

        monkeypatch.setattr(runtime, "_exact_admission_validator", fail_validator)

    try:
        with pytest.raises(PreResourceFailure) as caught:
            provider.open_working_set(request, plan, store, receipt)
        assert caught.value is failure
        assert len(transactions) == 1
        assert transactions[0].state == "aborted"
        with gate._condition:
            assert gate._live_epoch is None
            assert not any(
                state.status == "active" for state in gate._tokens.values()
            )
            assert all(lease.phase == "closed" for lease in gate._leases.values())
        assert provider._active_lease is None
        assert provider._provisional_resources is None
    finally:
        _repair_test_construction_state(gate)
        _close_case(runtime, store)


def test_construction_failure_before_real_begin_uses_fatal_winner(
    monkeypatch,
):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime, store_id="terminal-construction-fatal-before-begin"
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    gate = runtime._terminal_gate
    fatal_primary = RuntimeError("fatal immediately before real begin")
    transactions = []
    original_prepare = gate._prepare_lease_construction
    original_begin = gate.begin_lease

    def capture_transaction(*args, **kwargs):
        transaction = original_prepare(*args, **kwargs)
        transactions.append(transaction)
        return transaction

    def publish_before_begin(*args, **kwargs):
        assert runtime._enter_communicator_fatal(fatal_primary) is fatal_primary
        return original_begin(*args, **kwargs)

    monkeypatch.setattr(
        gate, "_prepare_lease_construction", capture_transaction
    )
    monkeypatch.setattr(gate, "begin_lease", publish_before_begin)

    try:
        with pytest.raises(RuntimeError) as caught:
            provider.open_working_set(request, plan, store, receipt)
        assert caught.value is fatal_primary
        assert len(transactions) == 1
        assert transactions[0].state == "fatal_retained"
        assert gate._fatal_transition.primary is fatal_primary
        with gate._condition:
            assert gate._live_epoch is None
            assert not gate._has_active_tokens()
            assert gate._leases == {}
        assert provider._active_lease is None
        assert provider._provisional_resources is None
    finally:
        _close_case(runtime, store)


def test_construction_abort_after_real_resolution_preserves_primary(monkeypatch):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime, store_id="terminal-total-construction-abort-after"
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    gate = runtime._terminal_gate

    class ConstructionPrimary(RuntimeError):
        pass

    class AbortSecondary(BaseException):
        pass

    primary = ConstructionPrimary("construction body failed")
    secondary = AbortSecondary("abort raised after real resolution")
    monkeypatch.setattr(
        provider,
        "_provision_status_workspace",
        lambda **_kwargs: (_ for _ in ()).throw(primary),
    )
    original_abort = gate._abort_lease_construction

    def fail_after_abort(*args, **kwargs):
        original_abort(*args, **kwargs)
        raise secondary

    monkeypatch.setattr(gate, "_abort_lease_construction", fail_after_abort)
    try:
        with pytest.raises(ConstructionPrimary) as caught:
            provider.open_working_set(request, plan, store, receipt)
        assert caught.value is primary
        with gate._condition:
            assert gate._live_epoch is None
            assert not gate._has_active_tokens()
            assert next(iter(gate._leases.values())).phase == "closed"
        assert provider._active_lease is None
        assert provider._provisional_resources is None
    finally:
        _repair_test_construction_state(gate)
        _close_case(runtime, store)


def test_construction_commit_after_real_resolution_returns_committed_lease(
    monkeypatch,
):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime, store_id="terminal-total-construction-commit-after"
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    gate = runtime._terminal_gate

    class CommitSecondary(BaseException):
        pass

    secondary = CommitSecondary("commit raised after real resolution")
    original_commit = gate._commit_lease_construction

    def fail_after_commit(*args, **kwargs):
        original_commit(*args, **kwargs)
        raise secondary

    monkeypatch.setattr(gate, "_commit_lease_construction", fail_after_commit)
    lease = None
    try:
        lease = provider.open_working_set(request, plan, store, receipt)
        assert lease is provider._active_lease
        assert provider._provisional_resources is None
        with gate._condition:
            assert gate._live_epoch == lease._epoch
            assert gate._leases[lease._epoch].phase == "open"
            assert not gate._has_active_tokens()
    finally:
        if lease is not None:
            lease.close()
        _close_case(runtime, store)


@pytest.mark.parametrize("selection", ("abort", "commit"))
def test_construction_final_selection_uses_concurrent_fatal_primary(
    monkeypatch,
    selection,
):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-construction-fatal-selection-{}".format(selection),
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    gate = runtime._terminal_gate
    construction_primary = RuntimeError("local construction failure")
    fatal_primary = RuntimeError("concurrent fatal winner")
    method_name = "_{}_lease_construction".format(selection)
    original_selection = getattr(gate, method_name)
    selected = []

    def publish_before_selection(*args, **kwargs):
        if not selected:
            selected.append(True)
            assert runtime._enter_communicator_fatal(fatal_primary) is fatal_primary
        return original_selection(*args, **kwargs)

    monkeypatch.setattr(gate, method_name, publish_before_selection)
    if selection == "abort":
        monkeypatch.setattr(
            provider,
            "_provision_status_workspace",
            lambda **_kwargs: (_ for _ in ()).throw(construction_primary),
        )
    try:
        with pytest.raises(RuntimeError) as caught:
            provider.open_working_set(request, plan, store, receipt)
        assert caught.value is fatal_primary
        assert selected == [True]
        assert gate._fatal_transition.primary is fatal_primary
        if selection == "abort":
            assert construction_primary in runtime._terminal_secondary_errors
        with gate._condition:
            assert not gate._has_active_tokens()
            assert gate._leases[gate._live_epoch].phase == "fatal_retained"
        assert provider._active_lease is None
        assert provider._provisional_resources is not None
    finally:
        _close_case(runtime, store)


def test_public_provider_close_delegates_to_runtime_without_mutation(monkeypatch):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-public-provider-close-delegates"
    )
    sentinel = object()
    calls = []
    runtime._active_provider = provider
    before = (
        provider.runtime,
        provider._active_lease,
        provider._cache,
        runtime._active_provider,
        lease._store_reservation,
    )
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(
                runtime,
                "close",
                lambda: calls.append("runtime_close") or sentinel,
            )
            assert provider.close() is sentinel
            assert calls == ["runtime_close"]
            assert (
                provider.runtime,
                provider._active_lease,
                provider._cache,
                runtime._active_provider,
                lease._store_reservation,
            ) == before
    finally:
        _close_case(runtime, store)


def test_public_provider_close_rejects_private_partial_admission():
    runtime = _runtime()
    request, _, store, _, _ = _active_case(
        runtime, store_id="terminal-public-provider-close-partial"
    )
    provider = _provider(runtime, request)
    try:
        with pytest.raises(TypeError):
            provider.close(_admission_token=object())
    finally:
        _close_case(runtime, store)


@pytest.mark.parametrize("candidate", ("wrong", "stale"))
def test_private_provider_close_requires_exact_runtime_close_admission(
    monkeypatch,
    candidate,
):
    runtime = _runtime()
    request, _, store, _, _ = _active_case(
        runtime,
        store_id="terminal-private-provider-close-{}".format(candidate),
    )
    provider = _provider(runtime, request)
    runtime._active_provider = provider
    gate = runtime._terminal_gate
    request_token = gate.admit_runtime("private_provider_close_fixture")
    transition, elected = gate._freeze_runtime_close(request_token)
    assert elected is True
    token = gate.admit_runtime_close(transition, "provider_close")
    validator = runtime._exact_admission_validator(
        token,
        scope="runtime_close",
        epoch=None,
        transition_sequence=transition.sequence,
    )
    supplied = token
    if candidate == "wrong":
        supplied = replace(token, scope="runtime")
    else:
        gate.release(token)
    before = (
        provider.runtime,
        provider._cache_factory,
        provider._pool_factory,
        provider._scheduler_factory,
        runtime._active_provider,
    )
    sentinel = []
    monkeypatch.setattr(
        provider,
        "_retain_terminal_resources",
        lambda *_args: sentinel.append(True),
    )
    try:
        with pytest.raises((TypeError, RuntimeError)):
            provider._close_for_runtime(
                _admission_token=supplied,
                _admission_validator=validator,
            )
        assert sentinel == []
        assert (
            provider.runtime,
            provider._cache_factory,
            provider._pool_factory,
            provider._scheduler_factory,
            runtime._active_provider,
        ) == before
    finally:
        if candidate == "wrong":
            gate.release(token)
        gate.commit_runtime_close(transition, lambda: None)
        runtime._closed = True
        _close_case(runtime, store)


@pytest.mark.parametrize(
    ("token", "validator"),
    (
        (None, None),
        (object(), None),
        (None, lambda _token: None),
    ),
)
def test_private_provider_close_rejects_missing_admission_before_read(
    token,
    validator,
):
    runtime = _runtime()
    request, _, store, _, _ = _active_case(
        runtime, store_id="terminal-private-provider-close-missing"
    )
    provider = _provider(runtime, request)
    runtime._active_provider = provider
    before = (
        provider.runtime,
        provider._cache_factory,
        provider._pool_factory,
        provider._scheduler_factory,
        runtime._active_provider,
        provider._closed,
    )
    try:
        with pytest.raises(TypeError):
            provider._close_for_runtime(
                _admission_token=token,
                _admission_validator=validator,
            )
        assert (
            provider.runtime,
            provider._cache_factory,
            provider._pool_factory,
            provider._scheduler_factory,
            runtime._active_provider,
            provider._closed,
        ) == before
    finally:
        _close_case(runtime, store)


def test_private_provider_close_rejects_wrong_operation_before_mutation():
    runtime = _runtime()
    request, _, store, _, _ = _active_case(
        runtime, store_id="terminal-private-provider-close-operation"
    )
    provider = _provider(runtime, request)
    runtime._active_provider = provider
    gate = runtime._terminal_gate
    request_token = gate.admit_runtime("private_provider_operation_fixture")
    transition, elected = gate._freeze_runtime_close(request_token)
    assert elected is True
    token = gate.admit_runtime_close(transition, "collective_close")
    validator = runtime._exact_admission_validator(
        token,
        scope="runtime_close",
        epoch=None,
        transition_sequence=transition.sequence,
    )
    before = (
        provider.runtime,
        provider._cache_factory,
        provider._pool_factory,
        provider._scheduler_factory,
        runtime._active_provider,
        provider._closed,
    )
    try:
        with pytest.raises(RuntimeError):
            provider._close_for_runtime(
                _admission_token=token,
                _admission_validator=validator,
            )
        assert (
            provider.runtime,
            provider._cache_factory,
            provider._pool_factory,
            provider._scheduler_factory,
            runtime._active_provider,
            provider._closed,
        ) == before
    finally:
        gate.release(token)
        gate.commit_runtime_close(transition, lambda: None)
        runtime._closed = True
        _close_case(runtime, store)


def test_private_provider_close_accepts_exact_elected_admission():
    runtime = _runtime()
    request, _, store, _, _ = _active_case(
        runtime, store_id="terminal-private-provider-close-exact"
    )
    provider = _provider(runtime, request)
    runtime._active_provider = provider
    gate = runtime._terminal_gate
    request_token = gate.admit_runtime("private_provider_exact_fixture")
    transition, elected = gate._freeze_runtime_close(request_token)
    assert elected is True
    token = gate.admit_runtime_close(transition, "provider_close")
    validator = runtime._exact_admission_validator(
        token,
        scope="runtime_close",
        epoch=None,
        transition_sequence=transition.sequence,
    )
    try:
        provider._close_for_runtime(
            _admission_token=token,
            _admission_validator=validator,
        )
        assert provider._closed is True
        assert provider.runtime is None
        assert runtime._active_provider is None
    finally:
        gate.release(token)
        gate.commit_runtime_close(transition, lambda: None)
        runtime._closed = True
        _close_case(runtime, store)


@pytest.mark.parametrize("phase", ("pending", "published", "closed"))
def test_public_provider_close_preserves_fatal_retained_ownership(
    monkeypatch,
    phase,
):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-public-provider-close-fatal-{}".format(phase)
    )
    gate = runtime._terminal_gate
    runtime._active_provider = provider
    primary = RuntimeError("provider close fatal primary")
    transition = gate.begin_fatal(primary)
    if phase in {"published", "closed"}:
        gate.publish_fatal(transition, primary)
    if phase == "closed":
        gate.commit_runtime_close(transition, lambda: None)
    destructive = []
    reservation = lease._store_reservation
    cache = provider._cache
    before = (
        provider.runtime,
        provider._active_lease,
        provider._cache,
        runtime._active_provider,
        lease._store_reservation,
        provider._cache_factory,
        provider._pool_factory,
        provider._scheduler_factory,
    )
    monkeypatch.setattr(
        reservation,
        "close",
        lambda: destructive.append("reservation"),
    )
    monkeypatch.setattr(
        cache,
        "close",
        lambda **_kwargs: destructive.append("cache"),
    )
    monkeypatch.setattr(
        lease,
        "close",
        lambda: destructive.append("lease"),
    )
    try:
        with pytest.raises(RuntimeError) as caught:
            provider.close()
        assert caught.value is primary
        assert destructive == []
        assert (
            provider.runtime,
            provider._active_lease,
            provider._cache,
            runtime._active_provider,
            lease._store_reservation,
            provider._cache_factory,
            provider._pool_factory,
            provider._scheduler_factory,
        ) == before
    finally:
        if phase == "pending":
            gate._fail_fatal_publication(transition, primary)
        _close_case(runtime, store)


@pytest.mark.parametrize("row", CONSTRUCTION_MATRIX_ROWS)
def test_after_real_construction_row_is_owned_by_healthy_rollback(
    monkeypatch,
    row,
):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-after-real-healthy-{}".format(row),
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    gate = runtime._terminal_gate
    failure = RuntimeError("{} failed after its real operation".format(row))
    captured = []
    close_calls = []

    def capture_resource(resource):
        captured.append(resource)
        original_close = resource.close

        def close(*args, **kwargs):
            close_calls.append(resource)
            return original_close(*args, **kwargs)

        monkeypatch.setattr(resource, "close", close)
        return resource

    if row == "receipt_consume":
        original = runtime.consume_residency_receipt

        def fail_after_real(*args, **kwargs):
            original(*args, **kwargs)
            raise failure

        monkeypatch.setattr(runtime, "consume_residency_receipt", fail_after_real)
    elif row == "construct_status":
        original = provider._provision_status_workspace

        def fail_after_real(*args, **kwargs):
            capture_resource(original(*args, **kwargs))
            raise failure

        monkeypatch.setattr(provider, "_provision_status_workspace", fail_after_real)
    elif row == "construct_store_reservation":
        original = provider._reserve_store

        def fail_after_real(*args, **kwargs):
            capture_resource(original(*args, **kwargs))
            raise failure

        monkeypatch.setattr(provider, "_reserve_store", fail_after_real)
    elif row == "construct_cache_reservation":
        original_factory = provider._cache_factory

        def cache_factory(*args, **kwargs):
            cache = original_factory(*args, **kwargs)
            original_reserve = cache.reserve

            def fail_after_real(*reserve_args, **reserve_kwargs):
                capture_resource(
                    original_reserve(*reserve_args, **reserve_kwargs)
                )
                raise failure

            monkeypatch.setattr(cache, "reserve", fail_after_real)
            return cache

        monkeypatch.setattr(provider, "_cache_factory", cache_factory)
    elif row == "construct_pool":
        original = provider._pool_factory

        def fail_after_real(*args, **kwargs):
            capture_resource(original(*args, **kwargs))
            raise failure

        monkeypatch.setattr(provider, "_pool_factory", fail_after_real)
    elif row == "construct_scheduler":
        original = provider._scheduler_factory

        def fail_after_real(*args, **kwargs):
            capture_resource(original(*args, **kwargs))
            raise failure

        monkeypatch.setattr(provider, "_scheduler_factory", fail_after_real)
    else:
        original = provider._activate_lease

        def fail_after_real(*args, **kwargs):
            original(*args, **kwargs)
            raise failure

        monkeypatch.setattr(provider, "_activate_lease", fail_after_real)

    try:
        with pytest.raises(RuntimeError) as caught:
            provider.open_working_set(request, plan, store, receipt)
        assert caught.value is failure
        with gate._condition:
            assert gate._phase is _TerminalPhase.HEALTHY
            assert gate._live_epoch is None
            assert not gate._has_active_tokens()
            assert all(lease.phase == "closed" for lease in gate._leases.values())
        assert provider._active_lease is None
        assert provider._provisional_resources is None
        assert id(receipt) in runtime._issued_receipts
        assert store._reservations == {}
        if provider._cache is not None:
            assert provider._cache._reservation is None
        if captured:
            assert close_calls == captured
    finally:
        for resource in captured:
            if not getattr(resource, "_closed", True):
                try:
                    resource.close()
                except BaseException:
                    pass
        _repair_test_construction_state(gate)
        _close_case(runtime, store)


@pytest.mark.parametrize(
    "row", ("construct_store_reservation", "construct_cache_reservation")
)
def test_after_real_reservation_is_exactly_fatal_retained(monkeypatch, row):
    class ConstructionFatal(BaseException):
        pass

    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-after-real-fatal-{}".format(row),
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    gate = runtime._terminal_gate
    primary = ConstructionFatal("{} fatal after real".format(row))
    captured = []

    if row == "construct_store_reservation":
        original = provider._reserve_store

        def fail_after_real(*args, **kwargs):
            captured.append(original(*args, **kwargs))
            raise primary

        monkeypatch.setattr(provider, "_reserve_store", fail_after_real)
    else:
        original_factory = provider._cache_factory

        def cache_factory(*args, **kwargs):
            cache = original_factory(*args, **kwargs)
            original_reserve = cache.reserve

            def fail_after_real(*reserve_args, **reserve_kwargs):
                captured.append(
                    original_reserve(*reserve_args, **reserve_kwargs)
                )
                raise primary

            monkeypatch.setattr(cache, "reserve", fail_after_real)
            return cache

        monkeypatch.setattr(provider, "_cache_factory", cache_factory)

    try:
        with pytest.raises(ConstructionFatal) as caught:
            provider.open_working_set(request, plan, store, receipt)
        assert caught.value is primary
        assert len(captured) == 1
        assert provider._provisional_resources is not None
        assert any(
            resource is captured[0]
            for resource in provider._provisional_resources.resources
        )
        assert captured[0]._closed is False
        with gate._condition:
            assert gate._fatal_transition.primary is primary
            assert gate._leases[gate._live_epoch].phase == "fatal_retained"
            assert not gate._has_active_tokens()
    finally:
        for resource in captured:
            if not resource._closed:
                resource.close()
        _close_case(runtime, store)


def _repair_test_abandoned_close_entry(gate):
    """Bound RED probes without adopting a departed production owner."""
    with gate._condition:
        for state in gate._tokens.values():
            if state.status == "active":
                state.status = "released"
                gate._thread_tokens.pop(state.token.thread_id, None)
        if gate._phase is _TerminalPhase.RUNTIME_CLOSING:
            gate._phase = _TerminalPhase.RUNTIME_CLOSED
            gate._runtime_close_result = None
        epoch = gate._live_epoch
        if epoch is not None:
            lease_state = gate._leases[epoch]
            if lease_state.phase == "closing":
                lease_state.phase = "closed"
                lease_state.result = None
                lease_state.has_result = True
                gate._live_epoch = None
        gate._condition.notify_all()


@pytest.mark.parametrize("concurrent_fatal", (False, True))
def test_after_real_lease_close_election_has_total_owner(
    monkeypatch,
    concurrent_fatal,
):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-after-real-lease-election-{}".format(
            "fatal" if concurrent_fatal else "healthy"
        )
    )
    gate = runtime._terminal_gate
    entry_installed = threading.Event()
    release_entry = threading.Event()

    class CloseEntryFailure(BaseException):
        pass

    entry_failure = CloseEntryFailure("lease close election wrapper failed")
    fatal_primary = RuntimeError("lease close concurrent fatal winner")
    original_begin = gate.begin_lease_close

    def fail_after_real(*args, **kwargs):
        original_begin(*args, **kwargs)
        entry_installed.set()
        assert release_entry.wait(_TIMEOUT_S)
        raise entry_failure

    monkeypatch.setattr(gate, "begin_lease_close", fail_after_real)
    closer, close_results, close_errors, close_done = _start(lease.close)
    fatal = None
    fatal_results = []
    fatal_errors = []
    fatal_done = None
    try:
        assert entry_installed.wait(_TIMEOUT_S)
        if concurrent_fatal:
            fatal, fatal_results, fatal_errors, fatal_done = _start(
                lambda: runtime._enter_communicator_fatal(fatal_primary)
            )
            _wait_for_phase(runtime, _TerminalPhase.FATAL_PUBLISHED)
        release_entry.set()
        _join(closer, close_done)
        if fatal is not None:
            _join(fatal, fatal_done)

        expected = fatal_primary if concurrent_fatal else entry_failure
        assert close_results == []
        assert close_errors == [expected]
        if fatal is not None:
            assert fatal_results == [fatal_primary]
            assert fatal_errors == []
        with gate._condition:
            assert gate._fatal_transition.primary is expected
            assert gate._leases[lease._epoch].phase == "fatal_retained"
            assert not gate._has_active_tokens()
        if concurrent_fatal:
            assert entry_failure in runtime._terminal_secondary_errors
    finally:
        release_entry.set()
        _repair_test_abandoned_close_entry(gate)
        for thread, done in ((closer, close_done), (fatal, fatal_done)):
            if thread is not None and thread.is_alive():
                _join(thread, done)
        reservation = lease._store_reservation
        if reservation is not None and not reservation._closed:
            reservation.close()
        _close_case(runtime, store)


@pytest.mark.parametrize(
    "boundary",
    ("admit_runtime", "freeze_runtime_close", "runtime_close_admission"),
)
@pytest.mark.parametrize("concurrent_fatal", (False, True))
def test_after_real_runtime_close_entry_has_total_owner(
    monkeypatch,
    boundary,
    concurrent_fatal,
):
    runtime = _runtime()
    gate = runtime._terminal_gate
    entry_installed = threading.Event()
    release_entry = threading.Event()

    class CloseEntryFailure(BaseException):
        pass

    entry_failure = CloseEntryFailure("{} wrapper failed".format(boundary))
    fatal_primary = RuntimeError("{} concurrent fatal winner".format(boundary))

    class Provider:
        _active_lease = None
        _terminal_error = None

        def _retain_transition_resources(self, lease):
            assert lease is None

        def _close_for_runtime(self, **_kwargs):
            return None

    runtime._active_provider = Provider()

    def after_real(original, *args, **kwargs):
        original(*args, **kwargs)
        entry_installed.set()
        assert release_entry.wait(_TIMEOUT_S)
        raise entry_failure

    if boundary == "admit_runtime":
        original = gate.admit_runtime

        def fail_after_real(operation):
            if operation != "begin_runtime_close":
                return original(operation)
            return after_real(original, operation)

        monkeypatch.setattr(gate, "admit_runtime", fail_after_real)
    elif boundary == "freeze_runtime_close":
        original = gate._freeze_runtime_close

        def fail_after_real(token):
            return after_real(original, token)

        monkeypatch.setattr(gate, "_freeze_runtime_close", fail_after_real)
    else:
        original = gate.admit_runtime_close

        def fail_after_real(transition, operation):
            if operation != "provider_close":
                return original(transition, operation)
            return after_real(original, transition, operation)

        monkeypatch.setattr(gate, "admit_runtime_close", fail_after_real)

    closer, close_results, close_errors, close_done = _start(runtime.close)
    fatal = None
    fatal_results = []
    fatal_errors = []
    fatal_done = None
    try:
        assert entry_installed.wait(_TIMEOUT_S)
        if concurrent_fatal:
            fatal, fatal_results, fatal_errors, fatal_done = _start(
                lambda: runtime._enter_communicator_fatal(fatal_primary)
            )
            with gate._condition:
                assert gate._condition.wait_for(
                    lambda: gate._fatal_transition is not None,
                    timeout=_TIMEOUT_S,
                )
        release_entry.set()
        assert close_done.wait(_TIMEOUT_S)
        if fatal_done is not None:
            assert fatal_done.wait(_TIMEOUT_S)
        _join(closer, close_done)
        if fatal is not None:
            _join(fatal, fatal_done)

        expected = fatal_primary if concurrent_fatal else entry_failure
        assert close_results == []
        assert close_errors == [expected]
        if fatal is not None:
            assert fatal_results == [fatal_primary]
            assert fatal_errors == []
        assert runtime._closed is True
        assert gate.phase is _TerminalPhase.RUNTIME_CLOSED
        with gate._condition:
            assert not gate._has_active_tokens()
            assert gate._fatal_transition.primary is expected
        if concurrent_fatal:
            assert entry_failure in runtime._terminal_secondary_errors
    finally:
        release_entry.set()
        _repair_test_abandoned_close_entry(gate)
        for thread, done in ((closer, close_done), (fatal, fatal_done)):
            if thread is not None and thread.is_alive():
                _join(thread, done)
        runtime._closed = True
