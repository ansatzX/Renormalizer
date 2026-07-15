import ast
from contextlib import contextmanager
from dataclasses import replace
import inspect
import threading
import traceback
from types import SimpleNamespace

import numpy as np
import pytest

from renormalizer.backend import distributed_runtime as distributed_runtime_module
from renormalizer.backend._distributed.async_owner import (
    AsyncAllocationRecord,
    AsyncResourceOwner,
    _CountedAsyncAdmission,
    allocation_record,
)
from renormalizer.backend._distributed.cache import (
    CacheEntryLease,
    CacheReservation,
    DeviceTensorCache,
)
from renormalizer.backend._distributed.local_operator import DistributedLocalOperator
from renormalizer.backend._distributed.pinned import PinnedBufferPool, StagingSlot
from renormalizer.backend._distributed import pinned as pinned_module
from renormalizer.backend._distributed.residency import (
    ResidencyPlanner,
    _HostTensorReservation,
)
from renormalizer.backend._distributed.providers import (
    ActiveWorkingSetProvider,
    WorkingSetLease,
    _ActiveOperandLease,
    _LeaseResourceRecord,
    _LeaseStatusWorkspace,
)
from renormalizer.backend._distributed import providers as providers_module
from renormalizer.backend._distributed import terminal as terminal_module
from renormalizer.backend._distributed.terminal import (
    _FatalTransition,
    _LeaseConstructionResourceSlot,
    _MISSING,
    _TerminalPhase,
)
from renormalizer.backend._distributed.transfer import (
    AsyncCompletionHandle,
    TransferScheduler,
    TransferTicket,
)
from renormalizer.backend._execution.model import ExecutionBindings
from renormalizer.backend.distributed_runtime import CupyDistributedRuntime
from renormalizer.backend.tests.residency.test_active_provider import (
    _LoopbackCollective,
    _ManualEvent,
    _active_case,
    _block_request,
    _explicit_budget,
    _FakeCupyBackend,
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


def _open_managed_active_case(*, store_id):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(runtime, store_id=store_id)
    provider = _install_managed_provider(runtime, request)
    receipt = runtime.preflight_residency(request, plan)
    lease = provider.open_working_set(request, plan, store, receipt).__enter__()
    return runtime, request, plan, store, provider, lease


def _install_managed_provider(runtime, request):
    construct_token = runtime._terminal_gate.admit_runtime_setup(
        "provider_construct"
    )
    construct_validator = runtime._exact_admission_validator(
        construct_token,
        scope="runtime_setup",
        epoch=None,
        operation="provider_construct",
    )
    provider = ActiveWorkingSetProvider(
        runtime,
        device_budget_resolution=request.device_budget,
        host_budget_resolution=request.host_budget,
        prefetch_depth=request.prefetch_depth,
        _admission_token=construct_token,
        _admission_validator=construct_validator,
        _standalone=False,
    )
    runtime._release_admission(construct_token)
    install_token = runtime._terminal_gate.admit_runtime_setup("provider_install")
    install_validator = runtime._exact_admission_validator(
        install_token,
        scope="runtime_setup",
        epoch=None,
        operation="provider_install",
    )
    runtime._install_active_provider(
        provider,
        _admission_token=install_token,
        _admission_validator=install_validator,
    )
    runtime._release_admission(install_token)
    return provider


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
                loaded.close(
                    _admission_token=token,
                    _admission_validator=lease._exact_admission_validator(token),
                )

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
    for managed_resource in (
        cache,
        pool,
        scheduler,
        status,
        cache_reservation,
    ):
        managed_resource._managed_epoch = token.epoch
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
        operation=row,
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
            self, *, _admission_token, _admission_validator, _deadline
        ):
            assert isinstance(_deadline, float)
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

    def allocate_device(**kwargs):
        array = original_device(**kwargs)
        allocated.append(array)
        return array

    def allocate_host(**kwargs):
        if failure_point == "host_allocation":
            entered.set()
            assert release.wait(_TIMEOUT_S)
            raise failure
        array = original_host(**kwargs)
        allocated.append(array)
        return array

    def bootstrap(local_code, **kwargs):
        if failure_point == "bootstrap":
            entered.set()
            assert release.wait(_TIMEOUT_S)
            raise failure
        return original_bootstrap(local_code, **kwargs)

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
        with lease._lease_admission("load_close") as token:
            loaded.close(
                _admission_token=token,
                _admission_validator=lease._exact_admission_validator(token),
            )

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

    def start_counted_completions(*args, **kwargs):
        start_calls.append(threading.get_ident())
        result = original_start(*args, **kwargs)
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
    assert descendant.operation == "compute_completion"
    with gate._condition:
        parent = gate._tokens[descendant.parent_sequence].token
    assert parent.operation == "child_close"
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
    with lease._lease_admission("prefetch") as token:
        admitted = {
            "_admission_token": token,
            "_admission_validator": lease._exact_admission_validator(token),
        }
        with lease.pool.checkout(destination.nbytes, **admitted) as slot:
            ticket = lease.scheduler.stage_h2d(
                output_ref,
                destination,
                slot,
                **admitted,
            )
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

    def begin(epoch, **kwargs):
        result = original_begin(epoch, **kwargs)
        assert isinstance(result[0].deadline, float)
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

    def begin(epoch, **kwargs):
        result = original_begin(epoch, **kwargs)
        assert isinstance(result[0].deadline, float)
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

    def fail_start(*_args, **_kwargs):
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

    def fail_start(*_args, **_kwargs):
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

    def fail(*_args, **_kwargs):
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
def test_publication_failure_terminalizer_marks_without_diagnostics(
    monkeypatch,
    catch_layer,
):
    runtime = _runtime()
    gate = runtime._terminal_gate
    primary = RuntimeError("{} publication primary".format(catch_layer))

    class PublishFailure(BaseException):
        pass

    publication_failure = PublishFailure(
        "{} publication failure".format(catch_layer)
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
        raise AssertionError("publication failure dispatched diagnostics")

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
    assert marker_observations == []
    assert diagnostic_calls == []
    assert publication_failure in runtime._terminal_secondary_errors


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
    assert calls == []
    assert sum(
        retained is secondary for retained in runtime._terminal_secondary_errors
    ) == 1
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

    def fail_start(*_args, **_kwargs):
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

    def fail_start(*_args, **_kwargs):
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

    def fail_start(*_args, **_kwargs):
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

    def retain_allocations(spec, **kwargs):
        allocation = original_allocator(spec, **kwargs)
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
        operation=token.operation,
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
        operation=token.operation,
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
        operation=token.operation,
        transition_sequence=transition.sequence,
    )
    try:
        action = provider._close_for_runtime(
            _admission_token=token,
            _admission_validator=validator,
        )
        assert provider._closed is False
        assert provider.runtime is runtime
        assert runtime._active_provider is provider
        action.finalize()
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
                resource._managed = False
                resource.close()
        _close_case(runtime, store)


_CONSTRUCTION_RESOURCE_ROWS = (
    "status_workspace",
    "store_reservation",
    "cache",
    "cache_reservation",
    "pool",
    "scheduler",
    "lease",
)


def _install_construction_recorder_failure(
    monkeypatch,
    provider,
    *,
    row,
    when,
    failure,
    captured,
    close_calls,
):
    tripped = []

    def wrap_recorder(recorder):
        def fail(**recorded):
            resource = recorded.get("resource")
            if resource is None or tripped:
                return recorder(**recorded)
            tripped.append(True)
            captured.append(resource)
            close = getattr(resource, "close", None)
            if callable(close):

                def observe_close(*args, **kwargs):
                    close_calls.append(resource)
                    return close(*args, **kwargs)

                monkeypatch.setattr(resource, "close", observe_close)
            if when == "before":
                raise failure
            recorder(**recorded)
            raise failure

        return fail

    if row == "cache":
        original = provider._lease_cache_recorder

        def cache_recorder(record):
            return wrap_recorder(original(record))

        monkeypatch.setattr(provider, "_lease_cache_recorder", cache_recorder)
        return tripped

    original = provider._construction_resource_recorder

    def construction_recorder(record, name):
        recorder = original(record, name)
        if name == row:
            return wrap_recorder(recorder)
        return recorder

    monkeypatch.setattr(
        provider,
        "_construction_resource_recorder",
        construction_recorder,
    )
    return tripped


@pytest.mark.parametrize("row", _CONSTRUCTION_RESOURCE_ROWS)
@pytest.mark.parametrize("when", ("before", "after"))
@pytest.mark.parametrize("fatal", (False, True))
def test_construction_resource_is_owned_before_recorder_returns(
    monkeypatch,
    row,
    when,
    fatal,
):
    class RecorderFatal(BaseException):
        pass

    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-construction-recorder-{}-{}-{}".format(
            row,
            when,
            "fatal" if fatal else "healthy",
        ),
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    failure = (
        RecorderFatal("{} recorder {}-real".format(row, when))
        if fatal
        else RuntimeError("{} recorder {}-real".format(row, when))
    )
    captured = []
    close_calls = []
    tripped = _install_construction_recorder_failure(
        monkeypatch,
        provider,
        row=row,
        when=when,
        failure=failure,
        captured=captured,
        close_calls=close_calls,
    )
    try:
        with pytest.raises(BaseException) as caught:
            provider.open_working_set(request, plan, store, receipt)
        assert caught.value is failure
        assert tripped == [True]
        assert len(captured) == 1
        resource = captured[0]
        if fatal:
            assert close_calls == []
            assert provider._provisional_resources is not None
            assert any(
                retained is resource
                for retained in provider._provisional_resources.resources
            )
            assert any(
                retained is resource
                for retained in runtime._terminal_quarantine._resources
            )
        elif row == "lease":
            assert close_calls == []
            assert resource._closed is True
            assert provider._provisional_resources is None
        else:
            assert close_calls == [resource]
            assert provider._provisional_resources is None
    finally:
        _repair_test_construction_state(runtime._terminal_gate)
        _close_case(runtime, store)


@pytest.mark.parametrize("row", ("cache", "lease"))
@pytest.mark.parametrize("fatal", (False, True))
def test_direct_constructor_after_real_wrapper_preserves_ownership(
    monkeypatch,
    row,
    fatal,
):
    class ConstructorFatal(BaseException):
        pass

    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-direct-constructor-{}-{}".format(
            row,
            "fatal" if fatal else "healthy",
        ),
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    failure = (
        ConstructorFatal("{} factory failed after real".format(row))
        if fatal
        else RuntimeError("{} factory failed after real".format(row))
    )
    captured = []
    close_calls = []

    def capture(resource):
        captured.append(resource)
        close = resource.close

        def observe_close(*args, **kwargs):
            close_calls.append(resource)
            return close(*args, **kwargs)

        monkeypatch.setattr(resource, "close", observe_close)
        return resource

    if row == "cache":
        original = provider._cache_factory

        def fail_after_real(*args, **kwargs):
            capture(original(*args, **kwargs))
            raise failure

        monkeypatch.setattr(provider, "_cache_factory", fail_after_real)
    else:
        original = providers_module.WorkingSetLease

        def fail_after_real(*args, **kwargs):
            capture(original(*args, **kwargs))
            raise failure

        monkeypatch.setattr(providers_module, "WorkingSetLease", fail_after_real)

    try:
        with pytest.raises(BaseException) as caught:
            provider.open_working_set(request, plan, store, receipt)
        assert caught.value is failure
        assert len(captured) == 1
        resource = captured[0]
        if fatal:
            assert close_calls == []
            assert provider._provisional_resources is not None
            assert any(
                retained is resource
                for retained in provider._provisional_resources.resources
            )
        elif row == "lease":
            assert close_calls == []
            assert resource._closed is True
            assert provider._provisional_resources is None
        else:
            assert close_calls == [resource]
            assert provider._provisional_resources is None
    finally:
        _repair_test_construction_state(runtime._terminal_gate)
        _close_case(runtime, store)


@pytest.mark.parametrize(
    "boundary",
    ("slot_before_real", "slot_after_real", "reserve_after_real"),
)
@pytest.mark.parametrize("fatal", (False, True))
def test_host_store_reservation_is_owned_inside_real_callee(
    monkeypatch,
    boundary,
    fatal,
):
    class ReservationFatal(BaseException):
        pass

    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-store-callee-{}-{}".format(
            boundary,
            "fatal" if fatal else "healthy",
        ),
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    failure = (
        ReservationFatal("{} reservation failure".format(boundary))
        if fatal
        else RuntimeError("{} reservation failure".format(boundary))
    )
    captured = []
    close_calls = []

    def capture(reservation):
        if not captured:
            captured.append(reservation)
            original_close = reservation.close

            def observe_close(*args, **kwargs):
                close_calls.append(reservation)
                return original_close(*args, **kwargs)

            monkeypatch.setattr(reservation, "close", observe_close)
        return reservation

    if boundary.startswith("slot_"):
        original_publish = _LeaseConstructionResourceSlot.publish
        tripped = []

        def fail_slot_publish(slot, resource=None, **kwargs):
            if slot.name != "store_reservation" or tripped:
                return original_publish(slot, resource, **kwargs)
            tripped.append(True)
            capture(resource)
            if boundary == "slot_after_real":
                original_publish(slot, resource, **kwargs)
            raise failure

        monkeypatch.setattr(
            _LeaseConstructionResourceSlot,
            "publish",
            fail_slot_publish,
        )
    else:
        original_reserve = store.reserve

        def fail_after_reserve(*args, **kwargs):
            reservation = capture(original_reserve(*args, **kwargs))
            raise failure

        monkeypatch.setattr(store, "reserve", fail_after_reserve)

    try:
        with pytest.raises(BaseException) as caught:
            provider.open_working_set(request, plan, store, receipt)
        assert caught.value is failure
        assert len(captured) == 1
        reservation = captured[0]
        if fatal:
            assert close_calls == []
            assert store._reservations[id(reservation)] is reservation
            assert provider._provisional_resources is not None
            assert any(
                retained is reservation
                for retained in provider._provisional_resources.resources
            )
            assert any(
                retained is reservation
                for retained in runtime._terminal_quarantine._resources
            )
        else:
            assert close_calls == [reservation]
            assert store._reservations == {}
            assert provider._provisional_resources is None
            assert id(receipt) in runtime._issued_receipts
    finally:
        if captured and not captured[0]._closed:
            captured[0].close()
        _repair_test_construction_state(runtime._terminal_gate)
        _close_case(runtime, store)


_LEASE_CLOSE_CLEANUP_ROWS = (
    "child_close",
    "schedule_writeback",
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


@pytest.mark.parametrize("row", _LEASE_CLOSE_CLEANUP_ROWS)
@pytest.mark.parametrize("boundary", ("before_real", "after_real"))
@pytest.mark.parametrize("fatal", (False, True))
def test_unproven_lease_cleanup_fail_stops_without_close_commit(
    monkeypatch,
    row,
    boundary,
    fatal,
):
    class CleanupFatal(BaseException):
        pass

    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-cleanup-{}-{}-{}".format(
            row,
            boundary,
            "fatal" if fatal else "exception",
        )
    )
    gate = runtime._terminal_gate
    child = lease.acquire(_block_request(lease))
    output = runtime.backend.zeros((4,), dtype=np.float64)
    lease.mark_dirty("output", output)
    failure = (
        CleanupFatal("{} {} cleanup fatal".format(row, boundary))
        if fatal
        else RuntimeError("{} {} cleanup error".format(row, boundary))
    )
    observed = []
    failure_positions = []
    targets = {
        "child_close": (child, "close"),
        "schedule_writeback": (lease, "_schedule_writeback"),
        "observe_peaks": (lease, "_observe_peaks"),
        "scheduler_complete": (lease.scheduler, "complete_all"),
        "cache_wait": (provider.cache, "wait_for_pending"),
        "pool_reap": (lease.pool, "reap_completed"),
        "cache_invalidate": (provider.cache, "invalidate_ref"),
        "emit_profile": (lease, "_emit_profile"),
        "status_close": (lease._status_workspace, "close"),
        "scheduler_close": (lease.scheduler, "close"),
        "pool_close": (lease.pool, "close"),
        "cache_reservation_close": (lease._cache_reservation, "close"),
        "cache_lifetime_reconcile": (provider, "_reconcile_lease_cache"),
        "store_reservation_close": (lease._store_reservation, "close"),
    }

    for operation, (target, attribute) in targets.items():
        original = getattr(target, attribute)

        def observe(*args, _operation=operation, _original=original, **kwargs):
            observed.append(_operation)
            if _operation == row and boundary == "before_real":
                failure_positions.append(len(observed))
                observed.append("failure")
                raise failure
            result = _original(*args, **kwargs)
            if _operation == row:
                failure_positions.append(len(observed))
                observed.append("failure")
                raise failure
            return result

        monkeypatch.setattr(target, attribute, observe)

    retained = (
        provider._active_lease,
        lease._resource_record,
        lease._status_workspace,
        lease.scheduler,
        lease.pool,
        lease._cache_reservation,
        lease._store_reservation,
        provider._cache,
    )
    try:
        with pytest.raises(BaseException) as caught:
            lease.close()
        assert caught.value is failure
        assert len(failure_positions) == 1
        assert observed[failure_positions[0] :] == ["failure"]
        assert (
            provider._active_lease,
            lease._resource_record,
            lease._status_workspace,
            lease.scheduler,
            lease.pool,
            lease._cache_reservation,
            lease._store_reservation,
            provider._cache,
        ) == retained
        with gate._condition:
            assert gate._fatal_transition.primary is failure
            assert gate._phase is _TerminalPhase.FATAL_PUBLISHED
            assert gate._leases[lease._epoch].phase == "fatal_retained"
            assert not gate._has_active_tokens()
        assert runtime._terminal_error is failure
        assert provider._terminal_error is failure
        assert lease._poisoned_error is failure
    finally:
        reservation = retained[6]
        if reservation is not None and not reservation._closed:
            try:
                reservation.close()
            except BaseException:
                pass
        _close_case(runtime, store)


@pytest.mark.parametrize("when", ("before", "after"))
def test_status_allocation_slot_precedes_fallible_recorder(
    monkeypatch,
    when,
):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-status-allocation-recorder-{}".format(when),
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    gate = runtime._terminal_gate
    recorder_entered = threading.Event()
    release_recorder = threading.Event()
    captured_records = []
    recorder_failure = RuntimeError("status allocation recorder failed")
    fatal_primary = RuntimeError("fatal during status allocation recorder")
    original_factory = provider._construction_resource_recorder

    def recorder_factory(record, name):
        recorder = original_factory(record, name)
        if name != "status_workspace":
            return recorder
        tripped = []

        def fail(**captured):
            records = tuple(captured.get("records", ()))
            if captured.get("resource") is not None or not records or tripped:
                return recorder(**captured)
            tripped.append(True)
            captured_records.extend(records)
            if when == "after":
                recorder(**captured)
            recorder_entered.set()
            assert release_recorder.wait(_TIMEOUT_S)
            raise recorder_failure

        return fail

    monkeypatch.setattr(
        provider,
        "_construction_resource_recorder",
        recorder_factory,
    )
    opener, open_results, open_errors, open_done = _start(
        lambda: provider.open_working_set(request, plan, store, receipt)
    )
    fatal = None
    try:
        assert recorder_entered.wait(_TIMEOUT_S)
        record = provider._provisional_resources
        fatal, fatal_results, fatal_errors, fatal_done = _start(
            lambda: runtime._enter_communicator_fatal(fatal_primary)
        )
        _wait_for_phase(runtime, _TerminalPhase.FATAL_PENDING)
        release_recorder.set()
        _join(opener, open_done)
        _join(fatal, fatal_done)

        assert open_results == []
        assert open_errors == [fatal_primary]
        assert fatal_results == [fatal_primary]
        assert fatal_errors == []
        identities = {retained.identity for retained in captured_records}
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
    finally:
        release_recorder.set()
        if opener.is_alive():
            _join(opener, open_done)
        if fatal is not None and fatal.is_alive():
            _join(fatal, fatal_done)
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


@pytest.mark.parametrize("concurrent_fatal", (False, True))
def test_before_real_runtime_close_admission_error_is_never_discarded(
    monkeypatch,
    concurrent_fatal,
):
    runtime = _runtime()
    gate = runtime._terminal_gate
    entry_error = RuntimeError("runtime close admission failed before real")
    fatal_primary = RuntimeError("runtime close concurrent fatal primary")
    admission_entered = threading.Event()
    release_admission = threading.Event()
    original_admit = gate.admit_runtime

    def fail_before_real(operation):
        if operation != "begin_runtime_close":
            return original_admit(operation)
        admission_entered.set()
        assert release_admission.wait(_TIMEOUT_S)
        raise entry_error

    monkeypatch.setattr(gate, "admit_runtime", fail_before_real)
    closer, close_results, close_errors, close_done = _start(runtime.close)
    fatal = None
    fatal_results = []
    fatal_errors = []
    fatal_done = None
    try:
        assert admission_entered.wait(_TIMEOUT_S)
        if concurrent_fatal:
            fatal, fatal_results, fatal_errors, fatal_done = _start(
                lambda: runtime._enter_communicator_fatal(fatal_primary)
            )
            _wait_for_phase(runtime, _TerminalPhase.FATAL_PUBLISHED)
        release_admission.set()
        _join(closer, close_done)
        if fatal is not None:
            _join(fatal, fatal_done)

        expected = fatal_primary if concurrent_fatal else entry_error
        assert close_results == []
        assert close_errors == [expected]
        if fatal is not None:
            assert fatal_results == [fatal_primary]
            assert fatal_errors == []
            assert entry_error in runtime._terminal_secondary_errors
        assert runtime._closed is True
        with gate._condition:
            assert gate._phase is _TerminalPhase.RUNTIME_CLOSED
            assert gate._fatal_transition.primary is expected
            assert not gate._has_active_tokens()
    finally:
        release_admission.set()
        for thread, done in ((closer, close_done), (fatal, fatal_done)):
            if thread is not None and thread.is_alive():
                _join(thread, done)
        _repair_test_abandoned_close_entry(gate)
        runtime._closed = True


def test_runtime_close_active_token_drain_honors_terminal_deadline(monkeypatch):
    runtime = _runtime()
    gate = runtime._terminal_gate
    token_ready = threading.Event()
    release_token = threading.Event()
    token_released = threading.Event()

    def hold_token():
        token = gate.admit_runtime("bounded_runtime_close_token")
        token_ready.set()
        assert release_token.wait(_TIMEOUT_S)
        gate.release(token)
        token_released.set()

    holder = threading.Thread(
        target=hold_token,
        name="task-18.3-bounded-runtime-close-token",
        daemon=True,
    )
    holder.start()
    assert token_ready.wait(_TIMEOUT_S)
    request = gate.admit_runtime("bounded_runtime_close_request")
    transition, elected = gate._freeze_runtime_close(request)
    assert elected is True
    monkeypatch.setattr(
        terminal_module.time,
        "monotonic",
        lambda: transition.deadline + 1.0,
    )
    try:
        with pytest.raises(TimeoutError, match="admission drain"):
            gate._drain_runtime_close(transition)
    finally:
        release_token.set()
        assert token_released.wait(_TIMEOUT_S)
        holder.join(_TIMEOUT_S)
        assert not holder.is_alive()
        _repair_test_abandoned_close_entry(gate)
        runtime._closed = True


@pytest.mark.parametrize("owner_boundary", ("lease_close", "runtime_close"))
def test_runtime_close_departed_owner_converges_without_adoption(
    owner_boundary,
):
    if owner_boundary == "lease_close":
        runtime, _, _, store, provider, lease = _open_active_case(
            store_id="terminal-departed-lease-close-owner"
        )
    else:
        runtime = _runtime()
        store = None
        provider = None
        lease = None
    gate = runtime._terminal_gate
    owner_ready = threading.Event()
    transitions = []

    def abandon_owner():
        if owner_boundary == "lease_close":
            transition, elected = gate.begin_lease_close(lease._epoch)
        else:
            request = gate.admit_runtime("departed_runtime_close_owner")
            transition, elected = gate._freeze_runtime_close(request)
        assert elected is True
        transitions.append(transition)
        owner_ready.set()

    owner = threading.Thread(
        target=abandon_owner,
        name="task-18.3-departed-{}".format(owner_boundary),
        daemon=True,
    )
    owner.start()
    assert owner_ready.wait(_TIMEOUT_S)
    owner.join(_TIMEOUT_S)
    assert not owner.is_alive()
    closer, close_results, close_errors, close_done = _start(runtime.close)
    try:
        assert close_done.wait(_TIMEOUT_S)
        _join(closer, close_done)
        assert close_results == []
        assert len(close_errors) == 1
        primary = close_errors[0]
        assert "owner exited" in str(primary)
        with gate._condition:
            assert gate._phase is _TerminalPhase.RUNTIME_CLOSED
            assert gate._fatal_transition.primary is primary
            assert not gate._has_active_tokens()
            assert transitions[0].owner_thread_id != threading.get_ident()
    finally:
        if closer.is_alive():
            _repair_test_abandoned_close_entry(gate)
            _join(closer, close_done)
        if lease is not None:
            reservation = lease._store_reservation
            if reservation is not None and not reservation._closed:
                reservation.close()
        _repair_test_abandoned_close_entry(gate)
        runtime._closed = True
        if store is not None and not store.closed:
            store.close()


def test_direct_lease_close_joiner_fail_stops_departed_owner():
    runtime, _, _, store, _, lease = _open_active_case(
        store_id="terminal-direct-departed-lease-close-owner"
    )
    gate = runtime._terminal_gate
    owner_ready = threading.Event()
    transitions = []

    def abandon_owner():
        transition, elected = gate.begin_lease_close(lease._epoch)
        assert elected is True
        transitions.append(transition)
        owner_ready.set()

    owner = threading.Thread(
        target=abandon_owner,
        name="task-18.3-direct-departed-lease-close",
        daemon=True,
    )
    owner.start()
    assert owner_ready.wait(_TIMEOUT_S)
    owner.join(_TIMEOUT_S)
    assert not owner.is_alive()

    try:
        with pytest.raises(RuntimeError, match="owner exited") as caught:
            lease.close()
        primary = caught.value
        with gate._condition:
            assert gate._fatal_transition.primary is primary
            assert gate._phase is _TerminalPhase.FATAL_PUBLISHED
            assert gate._leases[lease._epoch].phase == "fatal_retained"
            assert not gate._has_active_tokens()
            assert transitions[0].owner_thread is owner
    finally:
        reservation = lease._store_reservation
        if reservation is not None and not reservation._closed:
            reservation.close()
        runtime._closed = True
        _repair_test_abandoned_close_entry(gate)
        if not store.closed:
            store.close()


@pytest.mark.parametrize("owner_boundary", ("lease_close", "runtime_close"))
def test_resource_state_fail_stops_departed_close_owner_without_waiting(
    monkeypatch,
    owner_boundary,
):
    if owner_boundary == "lease_close":
        runtime, _, _, store, _, lease = _open_active_case(
            store_id="terminal-resource-state-departed-lease-owner"
        )
    else:
        runtime = _runtime()
        store = None
        lease = None
    gate = runtime._terminal_gate
    ready = threading.Event()
    transitions = []

    def abandon_close():
        if lease is None:
            request = gate.admit_runtime("resource_state_departed_owner")
            transition, elected = gate._freeze_runtime_close(request)
        else:
            transition, elected = gate.begin_lease_close(lease._epoch)
        assert elected is True
        transitions.append(transition)
        ready.set()

    owner = threading.Thread(
        target=abandon_close,
        name="task-18.3-resource-state-departed-{}".format(owner_boundary),
        daemon=True,
    )
    owner.start()
    assert ready.wait(_TIMEOUT_S)
    owner.join(_TIMEOUT_S)
    assert not owner.is_alive()
    waits = []

    def reject_wait(_timeout=None):
        waits.append(True)
        raise AssertionError("resource_state waited for a departed owner")

    monkeypatch.setattr(gate._condition, "wait", reject_wait)
    try:
        with pytest.raises(RuntimeError, match="owner exited") as caught:
            runtime.resource_state()
        primary = caught.value
        assert waits == []
        with gate._condition:
            assert gate._fatal_transition.primary is primary
            assert gate._phase is _TerminalPhase.FATAL_PUBLISHED
            assert not gate._has_active_tokens()
            assert transitions[0].owner_thread is owner
    finally:
        if lease is not None:
            reservation = lease._store_reservation
            if reservation is not None and not reservation._closed:
                reservation.close()
        _repair_test_abandoned_close_entry(gate)
        runtime._closed = True
        if store is not None and not store.closed:
            store.close()


def test_resource_state_spurious_wakes_share_one_terminal_deadline(monkeypatch):
    runtime = _runtime()
    gate = runtime._terminal_gate
    owner_ready = threading.Event()
    release_owner = threading.Event()
    transitions = []

    def hold_runtime_close():
        request = gate.admit_runtime("resource_state_deadline_owner")
        transition, elected = gate._freeze_runtime_close(request)
        assert elected is True
        transitions.append(transition)
        owner_ready.set()
        assert release_owner.wait(_TIMEOUT_S)

    owner = threading.Thread(
        target=hold_runtime_close,
        name="task-18.3-resource-state-deadline-owner",
        daemon=True,
    )
    owner.start()
    assert owner_ready.wait(_TIMEOUT_S)
    ticks = iter((10.0, 13.0, 15.0))

    class Clock:
        @staticmethod
        def monotonic():
            return next(ticks)

    waits = []

    def spurious_wait(timeout=None):
        waits.append(timeout)
        if len(waits) > 1:
            raise AssertionError("resource_state restarted its terminal timeout")
        return False

    monkeypatch.setattr(distributed_runtime_module, "time", Clock, raising=False)
    monkeypatch.setattr(gate._condition, "wait", spurious_wait)
    try:
        with pytest.raises(TimeoutError, match="resource_state terminal wait") as caught:
            runtime.resource_state()
        primary = caught.value
        assert waits == [2.0]
        with gate._condition:
            assert gate._fatal_transition.primary is primary
            assert gate._phase is _TerminalPhase.FATAL_PUBLISHED
            assert not gate._has_active_tokens()
    finally:
        release_owner.set()
        owner.join(_TIMEOUT_S)
        assert not owner.is_alive()
        _repair_test_abandoned_close_entry(gate)
        runtime._closed = True


@pytest.mark.parametrize("resource_name", ("pool", "cache", "scheduler"))
def test_managed_resource_close_requires_admission_before_state_read(
    resource_name,
):
    runtime, _, _, store, provider, lease = _open_managed_active_case(
        store_id="terminal-managed-close-{}".format(resource_name)
    )
    resource = {
        "pool": lease.pool,
        "cache": provider._cache,
        "scheduler": lease.scheduler,
    }[resource_name]
    before = dict(resource.__dict__)
    try:
        with pytest.raises(TypeError, match="admission"):
            resource.close()
        assert resource.__dict__ == before
    finally:
        _close_case(runtime, store)


def test_managed_runtime_resource_state_requires_admission_before_read():
    runtime, _, _, store, provider, _ = _open_managed_active_case(
        store_id="terminal-managed-runtime-state"
    )
    before = (
        provider._active_lease,
        provider._cache,
        provider._closed,
    )
    try:
        with pytest.raises(TypeError, match="admission"):
            provider.runtime_resource_state()
        assert (
            provider._active_lease,
            provider._cache,
            provider._closed,
        ) == before
    finally:
        _close_case(runtime, store)


@pytest.mark.parametrize("row", _CONSTRUCTION_RESOURCE_ROWS)
def test_construction_slot_contains_owner_before_overridable_publish(
    monkeypatch,
    row,
):
    class SlotFailure(BaseException):
        pass

    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-direct-slot-owner-{}".format(row),
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    failure = SlotFailure("{} slot wrapper failed before real".format(row))
    snapshots = []
    original_publish = _LeaseConstructionResourceSlot.publish

    def fail_before_publish(slot, resource=None, **kwargs):
        if slot.name != row or snapshots:
            return original_publish(slot, resource, **kwargs)
        snapshots.append(slot.snapshot())
        raise failure

    monkeypatch.setattr(
        _LeaseConstructionResourceSlot,
        "publish",
        fail_before_publish,
    )
    try:
        with pytest.raises(SlotFailure) as caught:
            provider.open_working_set(request, plan, store, receipt)
        assert caught.value is failure
        assert len(snapshots) == 1
        snapshot = snapshots[0]
        if row == "status_workspace":
            assert snapshot.records
        else:
            assert snapshot.resource is not None
        if row == "pool":
            assert snapshot.resource._slot._array is not None
            assert snapshot.records == snapshot.resource.allocation_records
    finally:
        _repair_test_construction_state(runtime._terminal_gate)
        _close_case(runtime, store)


def test_scheduler_slot_contains_partial_owner_and_stream_before_wrapper(
    monkeypatch,
):
    runtime = _runtime()
    gate = runtime._terminal_gate
    transaction = gate._prepare_lease_construction(
        "scheduler_slot_probe",
        resource_names=("scheduler",),
    )
    epoch, token = gate.begin_lease(
        "scheduler_slot_probe",
        _transaction=transaction,
    )
    validator = runtime._exact_admission_validator(
        token,
        scope="construction",
        epoch=epoch,
        operation=token.operation,
    )
    failure = RuntimeError("scheduler slot wrapper failed before real")
    snapshots = []
    original_publish = _LeaseConstructionResourceSlot.publish

    def fail_before_publish(slot, resource=None, **kwargs):
        if slot.name != "scheduler" or snapshots:
            return original_publish(slot, resource, **kwargs)
        snapshots.append(slot.snapshot())
        raise failure

    monkeypatch.setattr(
        _LeaseConstructionResourceSlot,
        "publish",
        fail_before_publish,
    )
    backend = _FakeCupyBackend()
    store = SimpleNamespace(copy_into=lambda *_args, **_kwargs: None)
    pool = SimpleNamespace(checkout=lambda *_args, **_kwargs: None)
    try:
        with pytest.raises(RuntimeError) as caught:
            TransferScheduler(
                store,
                backend,
                pool,
                _admission_token=token,
                _admission_validator=validator,
                _construction_slot=transaction.resource_slots["scheduler"],
            )
        assert caught.value is failure
        snapshot = snapshots[0]
        assert snapshot.resource is not None
        assert snapshot.resource._stream is backend._cupy.cuda.stream
        assert snapshot.streams == (backend._cupy.cuda.stream,)
        snapshot.resource.close(
            _admission_token=token,
            _admission_validator=validator,
        )
        gate._abort_lease_construction(
            transaction,
            failure,
            lambda: None,
            lambda _primary: None,
        )
    finally:
        _repair_test_construction_state(gate)
        runtime._closed = True


def test_construction_rollback_stops_after_first_uncertain_cleanup(
    monkeypatch,
):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-rollback-first-uncertainty",
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    primary = RuntimeError("construction failed after all resources")
    uncertainty = RuntimeError("scheduler close completion is uncertain")
    observed = []
    original_scheduler_factory = provider._scheduler_factory
    original_pool_factory = provider._pool_factory
    original_cache_factory = provider._cache_factory
    original_store_reserve = store.reserve

    def scheduler_factory(*args, **kwargs):
        scheduler = original_scheduler_factory(*args, **kwargs)
        close = scheduler.close

        def fail_after_close(*close_args, **close_kwargs):
            observed.append("scheduler")
            close(*close_args, **close_kwargs)
            raise uncertainty

        monkeypatch.setattr(scheduler, "close", fail_after_close)
        return scheduler

    def pool_factory(*args, **kwargs):
        pool = original_pool_factory(*args, **kwargs)
        close = pool.close

        def observe_close(*close_args, **close_kwargs):
            observed.append("pool")
            return close(*close_args, **close_kwargs)

        monkeypatch.setattr(pool, "close", observe_close)
        return pool

    def cache_factory(*args, **kwargs):
        cache = original_cache_factory(*args, **kwargs)
        close = cache.close

        def observe_close(*close_args, **close_kwargs):
            observed.append("cache")
            return close(*close_args, **close_kwargs)

        monkeypatch.setattr(cache, "close", observe_close)
        return cache

    def reserve(*args, **kwargs):
        reservation = original_store_reserve(*args, **kwargs)
        close = reservation.close

        def observe_close(*close_args, **close_kwargs):
            observed.append("store")
            return close(*close_args, **close_kwargs)

        monkeypatch.setattr(reservation, "close", observe_close)
        return reservation

    monkeypatch.setattr(provider, "_scheduler_factory", scheduler_factory)
    monkeypatch.setattr(provider, "_pool_factory", pool_factory)
    monkeypatch.setattr(provider, "_cache_factory", cache_factory)
    monkeypatch.setattr(store, "reserve", reserve)
    monkeypatch.setattr(
        provider,
        "_activate_lease",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )
    try:
        with pytest.raises(RuntimeError) as caught:
            provider.open_working_set(request, plan, store, receipt)
        assert caught.value is primary
        assert observed == ["scheduler"]
        assert uncertainty in runtime._terminal_secondary_errors
        assert provider._provisional_resources is not None
    finally:
        _repair_test_construction_state(runtime._terminal_gate)
        _close_case(runtime, store)


@pytest.mark.parametrize("resource_name", ("pool", "cache", "scheduler"))
def test_managed_resource_close_stops_on_first_internal_uncertainty(
    monkeypatch,
    resource_name,
):
    runtime, _, _, store, provider, lease = _open_managed_active_case(
        store_id="terminal-managed-close-uncertainty-{}".format(resource_name)
    )
    gate = runtime._terminal_gate
    transition = None
    if resource_name == "cache":
        with gate._condition:
            gate._leases[lease._epoch].phase = "closed"
            gate._live_epoch = None
        request = gate.admit_runtime("begin_runtime_close")
        transition, elected = gate._freeze_runtime_close(request)
        assert elected is True
        transition = gate._drain_runtime_close(transition)
        token = gate.admit_runtime_close(transition, "provider_close")
        scope = "runtime_close"
        epoch = None
    else:
        transition, elected = gate.begin_lease_close(lease._epoch)
        assert elected is True
        operation = "pool_close" if resource_name == "pool" else "scheduler_close"
        token = gate.admit_lease_close(transition, operation)
        scope = "lease_close"
        epoch = lease._epoch
    validator = runtime._exact_admission_validator(
        token,
        scope=scope,
        epoch=epoch,
        operation=token.operation,
        transition_sequence=transition.sequence,
    )
    failure = RuntimeError("{} close uncertainty".format(resource_name))
    if resource_name == "pool":
        resource = lease.pool
        monkeypatch.setattr(
            resource,
            "wait_for_slot",
            lambda **_kwargs: (_ for _ in ()).throw(failure),
        )
    elif resource_name == "cache":
        resource = provider._cache
        monkeypatch.setattr(
            resource._reservation,
            "close",
            lambda **_kwargs: (_ for _ in ()).throw(failure),
        )
    else:
        resource = lease.scheduler
        monkeypatch.setattr(
            resource,
            "complete_all",
            lambda **_kwargs: (_ for _ in ()).throw(failure),
        )
    before = dict(resource.__dict__)
    try:
        with pytest.raises(RuntimeError) as caught:
            resource.close(
                _admission_token=token,
                _admission_validator=validator,
            )
        assert caught.value is failure
        assert resource.__dict__ == before
    finally:
        gate.release(token)
        _repair_test_abandoned_close_entry(gate)
        _close_case(runtime, store)


def test_private_provider_close_retains_graph_on_cleanup_uncertainty(
    monkeypatch,
):
    runtime = _runtime()
    request, _, store, _, _ = _active_case(
        runtime,
        store_id="terminal-provider-close-transaction",
    )
    provider = _provider(runtime, request)
    failure = RuntimeError("lease cleanup is uncertain")
    cache_calls = []
    lease = SimpleNamespace(
        close=lambda **_kwargs: (_ for _ in ()).throw(failure),
    )
    cache = SimpleNamespace(
        close=lambda **_kwargs: cache_calls.append(True),
        poisoned=False,
    )
    provider._active_lease = lease
    provider._cache = cache
    gate = runtime._terminal_gate
    request_token = gate.admit_runtime("begin_runtime_close")
    transition, elected = gate._freeze_runtime_close(request_token)
    assert elected is True
    transition = gate._drain_runtime_close(transition)
    token = gate.admit_runtime_close(transition, "provider_close")
    validator = runtime._exact_admission_validator(
        token,
        scope="runtime_close",
        epoch=None,
        operation=token.operation,
        transition_sequence=transition.sequence,
    )
    before = (
        provider.runtime,
        provider._active_lease,
        provider._cache,
        provider._cache_factory,
        provider._pool_factory,
        provider._scheduler_factory,
    )
    try:
        with pytest.raises(RuntimeError) as caught:
            provider._close_for_runtime(
                _admission_token=token,
                _admission_validator=validator,
                _deadline=transition.deadline,
            )
        assert caught.value is failure
        assert cache_calls == []
        assert (
            provider.runtime,
            provider._active_lease,
            provider._cache,
            provider._cache_factory,
            provider._pool_factory,
            provider._scheduler_factory,
        ) == before
    finally:
        gate.release(token)
        _repair_test_abandoned_close_entry(gate)
        runtime._closed = True
        if not store.closed:
            store.close()


def test_runtime_close_stops_before_collective_after_provider_uncertainty(
    monkeypatch,
):
    runtime = _runtime()
    request, _, store, _, _ = _active_case(
        runtime,
        store_id="terminal-runtime-close-provider-transaction",
    )
    provider = _provider(runtime, request)
    failure = RuntimeError("provider cleanup completion is uncertain")
    scheduler = SimpleNamespace(
        _start_counted_completions=lambda **_kwargs: None
    )
    lease = SimpleNamespace(
        scheduler=scheduler,
        close=lambda **_kwargs: (_ for _ in ()).throw(failure),
        _resource_record=_LeaseResourceRecord(None),
        _poisoned_error=None,
    )
    provider._active_lease = lease
    provider._provisional_resources = lease._resource_record
    collective = runtime.collective
    collective_calls = []
    monkeypatch.setattr(
        collective,
        "_close_for_runtime",
        lambda *_args, **_kwargs: collective_calls.append(True),
        raising=False,
    )
    before = (
        runtime._active_provider,
        runtime.collective,
        provider.runtime,
        provider._active_lease,
        provider._cache_factory,
        provider._pool_factory,
        provider._scheduler_factory,
    )
    try:
        with pytest.raises(RuntimeError) as caught:
            runtime.close()
        assert caught.value is failure
        assert collective_calls == []
        assert (
            runtime._active_provider,
            runtime.collective,
            provider.runtime,
            provider._active_lease,
            provider._cache_factory,
            provider._pool_factory,
            provider._scheduler_factory,
        ) == before
        with runtime._terminal_gate._condition:
            assert runtime._terminal_gate._fatal_transition.primary is failure
            assert runtime._terminal_gate._fatal_publication_failure in {
                _MISSING,
                failure,
            }
    finally:
        _repair_test_abandoned_close_entry(runtime._terminal_gate)
        runtime._closed = True
        if not store.closed:
            store.close()


@pytest.mark.parametrize("kind", ("lease", "runtime"))
def test_close_stages_reuse_one_transition_deadline(monkeypatch, kind):
    runtime, _, _, store, _, lease = _open_active_case(
        store_id="terminal-shared-{}-deadline".format(kind)
    )
    gate = runtime._terminal_gate
    if kind == "lease":
        transition, elected = gate.begin_lease_close(lease._epoch)
    else:
        gate._leases[lease._epoch].phase = "closed"
        gate._live_epoch = None
        request = gate.admit_runtime("begin_runtime_close")
        transition, elected = gate._freeze_runtime_close(request)
    assert elected is True
    captured = []

    def capture(_predicate, deadline, _message, **_kwargs):
        captured.append(deadline)

    monkeypatch.setattr(gate, "_wait_until_deadline", capture)
    monkeypatch.setattr(
        gate,
        "_deadline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("close stage created a fresh deadline")
        ),
    )
    try:
        assert transition.deadline > 0
        if kind == "lease":
            gate.wait_for_lease_admissions(transition, 999)
            gate.wait_for_lease_admissions(transition, 999)
        else:
            gate._drain_runtime_close(transition, timeout_s=999)
            gate._drain_runtime_close(transition, timeout_s=999)
        assert captured == [transition.deadline, transition.deadline]
    finally:
        monkeypatch.undo()
        reservation = lease._store_reservation
        if reservation is not None and not reservation._closed:
            reservation.close()
        _repair_test_abandoned_close_entry(gate)
        runtime._closed = True
        if not store.closed:
            store.close()


def test_failed_fatal_runtime_close_join_reuses_transition_deadline(
    monkeypatch,
):
    runtime = _runtime()
    gate = runtime._terminal_gate
    primary = RuntimeError("fatal runtime close publication failed")
    transition = gate.begin_fatal(primary)
    gate._fail_fatal_publication(transition, primary)
    captured = []
    runtime._pending_fatal_collective = SimpleNamespace(
        _wait_for_joined_fatal_publication=(
            lambda *, _deadline: captured.append(_deadline)
        ),
    )

    def fail_commit(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr(gate, "commit_runtime_close", fail_commit)
    with pytest.raises(RuntimeError) as caught:
        runtime._commit_fatal_runtime_close(transition, primary)

    assert caught.value is primary
    assert captured == [transition.deadline]


_MANAGED_RESOURCE_METHOD_ROWS = (
    (PinnedBufferPool, "checkout", (1,)),
    (PinnedBufferPool, "_checkout", (1,)),
    (PinnedBufferPool, "_release", (object(),)),
    (PinnedBufferPool, "reap_completed", ()),
    (PinnedBufferPool, "wait_for_slot", ()),
    (PinnedBufferPool, "close", ()),
    (DeviceTensorCache, "_set_resource_recorder", (None,)),
    (DeviceTensorCache, "reserve", ({}, 0)),
    (DeviceTensorCache, "acquire", (None, None)),
    (DeviceTensorCache, "reap_completed", ()),
    (DeviceTensorCache, "wait_for_pending", ()),
    (DeviceTensorCache, "contains", (None,)),
    (DeviceTensorCache, "refcount", (None,)),
    (DeviceTensorCache, "invalidate", (None,)),
    (DeviceTensorCache, "invalidate_ref", (None,)),
    (DeviceTensorCache, "close", ()),
    (TransferScheduler, "stage_h2d", (None, None, None)),
    (TransferScheduler, "wait_for_h2d", (None,)),
    (TransferScheduler, "begin_compute", ()),
    (TransferScheduler, "record_compute_completion", ()),
    (TransferScheduler, "writeback_d2h", (None, None, None)),
    (TransferScheduler, "reap_completed", ()),
    (TransferScheduler, "complete_all", ()),
    (TransferScheduler, "_runtime_event_count", ()),
    (TransferScheduler, "close", ()),
)


@pytest.mark.parametrize(
    "resource_type,method_name,args",
    _MANAGED_RESOURCE_METHOD_ROWS,
    ids=lambda value: value if isinstance(value, str) else None,
)
@pytest.mark.parametrize(
    "candidate_kind",
    ("missing", "token_only", "validator_only", "wrong", "stale"),
)
def test_every_managed_resource_method_rejects_before_object_state_read(
    resource_type,
    method_name,
    args,
    candidate_kind,
):
    runtime = _runtime()
    gate = runtime._terminal_gate
    resource = object.__new__(resource_type)
    resource._managed = True
    token = gate.admit_runtime("managed_resource_probe")
    resource._managed_epoch = token.epoch
    validator = runtime._exact_admission_validator(
        token,
        scope="runtime",
        epoch=None,
        operation=token.operation,
    )
    if candidate_kind == "missing":
        candidate = None
        candidate_validator = None
    elif candidate_kind == "token_only":
        candidate = token
        candidate_validator = None
    elif candidate_kind == "validator_only":
        candidate = None
        candidate_validator = validator
    elif candidate_kind == "wrong":
        candidate = replace(token, sequence=token.sequence + 1000)
        candidate_validator = validator
    else:
        candidate = token
        candidate_validator = validator
        gate.release(token)
        token = None
    try:
        with pytest.raises((TypeError, RuntimeError)):
            getattr(resource_type, method_name)(
                resource,
                *args,
                _admission_token=candidate,
                _admission_validator=candidate_validator,
            )
    finally:
        if token is not None:
            gate.release(token)
        runtime._closed = True


def test_lease_close_passes_one_deadline_to_every_blocking_resource_row(
    monkeypatch,
):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-resource-row-deadline"
    )
    observed = []
    transitions = []
    gate = runtime._terminal_gate
    original_begin = gate.begin_lease_close

    def begin(*args, **kwargs):
        transition, elected = original_begin(*args, **kwargs)
        transitions.append(transition)
        return transition, elected

    monkeypatch.setattr(gate, "begin_lease_close", begin)
    for resource, method_name in (
        (lease.scheduler, "complete_all"),
        (provider.cache, "wait_for_pending"),
        (lease.pool, "reap_completed"),
        (lease.scheduler, "close"),
        (lease.pool, "close"),
    ):
        original = getattr(resource, method_name)

        def wrapper(*args, _original=original, _name=method_name, **kwargs):
            observed.append((_name, kwargs.get("_deadline")))
            return _original(*args, **kwargs)

        monkeypatch.setattr(resource, method_name, wrapper)
    try:
        lease.close()
        assert len(transitions) == 1
        deadline = transitions[0].deadline
        assert observed
        assert all(row_deadline == deadline for _, row_deadline in observed)
    finally:
        _close_case(runtime, store)


def test_construction_rollback_passes_transaction_deadline_to_cleanup(
    monkeypatch,
):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-construction-cleanup-deadline",
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    transactions = []
    observed = []
    original_prepare = runtime._terminal_gate._prepare_lease_construction
    original_scheduler_factory = provider._scheduler_factory
    primary = RuntimeError("construction activation failed")

    def prepare(*args, **kwargs):
        transaction = original_prepare(*args, **kwargs)
        transactions.append(transaction)
        return transaction

    def scheduler_factory(*args, **kwargs):
        scheduler = original_scheduler_factory(*args, **kwargs)
        close = scheduler.close

        def record_close(*close_args, **close_kwargs):
            observed.append(close_kwargs.get("_deadline"))
            return close(*close_args, **close_kwargs)

        monkeypatch.setattr(scheduler, "close", record_close)
        return scheduler

    monkeypatch.setattr(
        runtime._terminal_gate,
        "_prepare_lease_construction",
        prepare,
    )
    monkeypatch.setattr(provider, "_scheduler_factory", scheduler_factory)
    monkeypatch.setattr(
        provider,
        "_activate_lease",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )
    try:
        with pytest.raises(RuntimeError) as caught:
            provider.open_working_set(request, plan, store, receipt)
        assert caught.value is primary
        assert len(transactions) == 1
        assert observed == [transactions[0].deadline]
    finally:
        _repair_test_construction_state(runtime._terminal_gate)
        _close_case(runtime, store)


def test_real_managed_descendants_reject_unadmitted_terminal_access():
    runtime, _, _, store, provider, lease = _open_managed_active_case(
        store_id="terminal-real-managed-descendants"
    )
    gate = runtime._terminal_gate
    identity = lease.cache_identities[0]
    with lease._lease_admission("acquire") as token:
        validator = lease._exact_admission_validator(token)
        cache_lease = provider._cache.acquire(
            identity,
            _admission_token=token,
            _admission_validator=validator,
        )
        checkout = lease.pool.checkout(
            8,
            _admission_token=token,
            _admission_validator=validator,
        )
        slot = checkout.__enter__()
        owner = lease.scheduler._register_owner("compute")
        handle = AsyncCompletionHandle(owner)
        ticket = TransferTicket(owner)

    primary = RuntimeError("managed descendant terminal winner")
    gate.begin_fatal(primary)
    status_workspace = lease._status_workspace
    status_device = status_workspace.__dict__.get(
        "_device_status", status_workspace.__dict__.get("device_status")
    )
    before = (
        cache_lease._entry,
        cache_lease._closed,
        slot._array,
        slot._owner,
        owner.state,
        tuple(owner._events),
        status_device,
    )
    operations = (
        lambda: cache_lease.identity,
        lambda: cache_lease.close(),
        lambda: lease._cache_reservation.allowlist,
        lambda: slot.array,
        lambda: slot.view((1,), np.dtype(np.float64)),
        lambda: owner.events,
        lambda: owner.reap(),
        lambda: handle.event,
        lambda: handle.query(),
        lambda: ticket.nbytes,
        lambda: ticket.take_result(),
        lambda: status_workspace.device_status,
        lambda: status_workspace.host_status,
        lambda: status_workspace.borrower,
        lambda: lease.scheduler.stream_count,
        lambda: provider.closed,
        lambda: provider.terminal_poisoned,
        lambda: provider.matches_config(
            provider.device_budget_resolution,
            provider.host_budget_resolution,
            provider.prefetch_depth,
        ),
        lambda: provider.resource_state(),
    )
    try:
        for operation in operations:
            with pytest.raises((TypeError, RuntimeError), match="admission"):
                operation()
        assert (
            cache_lease._entry,
            cache_lease._closed,
            slot._array,
            slot._owner,
            owner.state,
            tuple(owner._events),
            status_workspace.__dict__.get(
                "_device_status", status_workspace.__dict__.get("device_status")
            ),
        ) == before
    finally:
        _repair_test_construction_state(gate)
        runtime._closed = True
        reservation = lease._store_reservation
        if reservation is not None and not reservation._closed:
            store._release_reservation(reservation)
        if not store.closed:
            store.close()


def test_real_managed_descendants_reject_wrong_and_stale_admissions():
    runtime, _, _, store, provider, lease = _open_managed_active_case(
        store_id="terminal-real-managed-candidates"
    )
    gate = runtime._terminal_gate
    identity = lease.cache_identities[0]
    token = gate.admit_lease(lease._epoch, "acquire")
    validator = lease._exact_admission_validator(token)
    cache_lease = provider._cache.acquire(
        identity,
        _admission_token=token,
        _admission_validator=validator,
    )
    checkout = lease.pool.checkout(
        8,
        _admission_token=token,
        _admission_validator=validator,
    )
    slot = checkout.__enter__()
    owner = lease.scheduler._register_owner("compute")
    event = lease.scheduler._new_event(owner, completion=True)
    lease.scheduler._record(event)
    handle = AsyncCompletionHandle(owner)
    ticket = TransferTicket(owner)
    status_workspace = lease._status_workspace

    def operations(candidate, candidate_validator):
        return (
            lambda: cache_lease.close(
                None,
                _admission_token=candidate,
                _admission_validator=candidate_validator,
            ),
            lambda: slot.view(
                (1,),
                np.dtype(np.float64),
                _admission_token=candidate,
                _admission_validator=candidate_validator,
            ),
            lambda: owner.reap(
                _admission_token=candidate,
                _admission_validator=candidate_validator,
            ),
            lambda: handle.reap(
                _admission_token=candidate,
                _admission_validator=candidate_validator,
            ),
            lambda: ticket.take_result(
                _admission_token=candidate,
                _admission_validator=candidate_validator,
            ),
            lambda: status_workspace.close(
                _admission_token=candidate,
                _admission_validator=candidate_validator,
            ),
            lambda: lease.scheduler._runtime_event_count(
                _admission_token=candidate,
                _admission_validator=candidate_validator,
            ),
            lambda: provider.runtime_resource_state(
                _admission_token=candidate,
                _admission_validator=candidate_validator,
            ),
        )

    before = (
        cache_lease._entry,
        cache_lease._closed,
        slot._array,
        slot._owner,
        owner.state,
        tuple(owner._events),
        status_workspace._closed,
        tuple(lease.scheduler._events),
    )
    wrong = replace(token, sequence=token.sequence + 1000)
    try:
        for operation in operations(wrong, validator):
            with pytest.raises(RuntimeError):
                operation()
        assert (
            cache_lease._entry,
            cache_lease._closed,
            slot._array,
            slot._owner,
            owner.state,
            tuple(owner._events),
            status_workspace._closed,
            tuple(lease.scheduler._events),
        ) == before

        checkout.__exit__(None, None, None)
        gate.release(token)
        token = None
        stale = replace(wrong, sequence=wrong.sequence - 1000)
        for operation in operations(stale, validator):
            with pytest.raises(RuntimeError):
                operation()
        assert (
            cache_lease._entry,
            cache_lease._closed,
            slot._array,
            slot._owner,
            owner.state,
            tuple(owner._events),
            status_workspace._closed,
            tuple(lease.scheduler._events),
        ) == before

        with lease._lease_admission("child_close") as cleanup_token:
            cleanup_validator = lease._exact_admission_validator(cleanup_token)
            cache_lease.close(
                None,
                _admission_token=cleanup_token,
                _admission_validator=cleanup_validator,
            )
            owner.reap(
                _admission_token=cleanup_token,
                _admission_validator=cleanup_validator,
            )
    finally:
        if token is not None:
            if slot._checked_out:
                checkout.__exit__(None, None, None)
            gate.release(token)
        _close_case(runtime, store)


@pytest.mark.parametrize(
    "allocator_name",
    ("_allocate_status_device", "_allocate_status_host"),
)
def test_status_allocator_publishes_physical_owner_before_after_real_wrapper(
    monkeypatch,
    allocator_name,
):
    class AllocationFailure(BaseException):
        pass

    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-status-real-allocator-{}".format(allocator_name),
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    failure = AllocationFailure("status allocator wrapper failed after real")
    observed = []
    original = getattr(provider, allocator_name)

    def fail_after_real(*args, **kwargs):
        allocated = original(*args, **kwargs)
        slot = kwargs.get("_construction_slot")
        observed.append(
            (allocated, None if slot is None else slot.snapshot())
        )
        raise failure

    monkeypatch.setattr(provider, allocator_name, fail_after_real)
    try:
        with pytest.raises(AllocationFailure) as caught:
            provider.open_working_set(request, plan, store, receipt)
        assert caught.value is failure
        assert len(observed) == 1
        allocated, snapshot = observed[0]
        record = allocation_record(allocated)
        assert snapshot is not None
        assert any(
            retained.identity == record.identity
            and retained.owner is record.owner
            for retained in snapshot.records
        )
    finally:
        _repair_test_construction_state(runtime._terminal_gate)
        _close_case(runtime, store)


def test_fake_cupy_status_memory_and_array_share_canonical_device_identity():
    runtime = _runtime()
    gate = runtime._terminal_gate
    transaction = gate._prepare_lease_construction(
        "status_device_identity",
        resource_names=("status_workspace",),
    )
    epoch, token = gate.begin_lease(
        "status_device_identity",
        _transaction=transaction,
    )
    backend = _FakeCupyBackend()
    pointer = 0x12340000

    class Memory:
        def __init__(self, size):
            self.ptr = pointer
            self.size = size

    class MemoryPointer:
        def __init__(self, memory, offset):
            self.mem = memory
            self.ptr = memory.ptr + offset

    class Array:
        shape = (1,)
        dtype = np.dtype(np.int32)
        nbytes = np.dtype(np.int32).itemsize

        def __init__(self, memory_pointer):
            self.data = memory_pointer

    backend._cupy.cuda.Memory = Memory
    backend._cupy.cuda.MemoryPointer = MemoryPointer
    backend._cupy.ndarray = (
        lambda shape, dtype, memptr, order: Array(memptr)
    )
    runtime.backend = backend
    provider = ActiveWorkingSetProvider(
        runtime,
        device_budget_resolution=_explicit_budget(1024, "device"),
        host_budget_resolution=_explicit_budget(1024, "host"),
        _standalone=True,
    )
    slot = transaction.resource_slots["status_workspace"]
    failure = RuntimeError("status identity probe cleanup")
    try:
        array = provider._allocate_status_device(_construction_slot=slot)
        record = allocation_record(array)
        snapshot = slot.snapshot()
        assert record.identity == ("device", pointer)
        assert tuple(retained.identity for retained in snapshot.records) == (
            record.identity,
        )
        assert snapshot.records[0].owner is array.data.mem
    finally:
        gate.release(token)
        gate._abort_lease_construction(
            transaction,
            failure,
            lambda: None,
            lambda _primary: None,
        )
        _repair_test_construction_state(gate)
        runtime._closed = True


def test_scheduler_stream_factory_publishes_before_after_real_wrapper():
    class StreamFailure(BaseException):
        pass

    runtime = _runtime()
    gate = runtime._terminal_gate
    transaction = gate._prepare_lease_construction(
        "stream_factory_publication",
        resource_names=("scheduler",),
    )
    epoch, token = gate.begin_lease(
        "stream_factory_publication",
        _transaction=transaction,
    )
    validator = runtime._exact_admission_validator(
        token,
        scope="construction",
        epoch=epoch,
        operation=token.operation,
    )
    backend = _FakeCupyBackend()
    slot = transaction.resource_slots["scheduler"]
    failure = StreamFailure("stream factory wrapper failed after real")
    observed = []
    real_factory = TransferScheduler._create_stream_owned

    def fail_after_real(**kwargs):
        stream = real_factory(**kwargs)
        observed.append((stream, slot.snapshot()))
        raise failure

    try:
        with pytest.raises(StreamFailure) as caught:
            TransferScheduler(
                SimpleNamespace(copy_into=lambda *_args, **_kwargs: None),
                backend,
                SimpleNamespace(checkout=lambda *_args, **_kwargs: None),
                _admission_token=token,
                _admission_validator=validator,
                _construction_slot=slot,
                _stream_factory=fail_after_real,
            )
        assert caught.value is failure
        assert len(observed) == 1
        stream, snapshot = observed[0]
        assert snapshot.resource is not None
        assert snapshot.streams == (stream,)
    finally:
        gate._abort_lease_construction(
            transaction,
            failure,
            lambda: None,
            lambda _primary: None,
        )
        _repair_test_construction_state(gate)
        runtime._closed = True


def test_scheduler_event_factory_publishes_before_after_real_wrapper(
    monkeypatch,
):
    runtime = _runtime()
    scheduler = TransferScheduler(
        SimpleNamespace(copy_into=lambda *_args, **_kwargs: None),
        runtime.backend,
        SimpleNamespace(checkout=lambda *_args, **_kwargs: None),
    )
    owner = scheduler._register_owner("compute")
    failure = RuntimeError("event factory wrapper failed after real")
    observed = []
    real_factory = scheduler._create_event_owned

    def fail_after_real(**kwargs):
        event = real_factory(**kwargs)
        observed.append((event, owner._terminal_resource_snapshot()))
        raise failure

    monkeypatch.setattr(scheduler, "_create_event_owned", fail_after_real)
    with pytest.raises(RuntimeError) as caught:
        scheduler._new_event(owner, completion=True)

    assert caught.value is failure
    assert len(observed) == 1
    event, snapshot = observed[0]
    assert snapshot["completion_event"] is event
    assert snapshot["events"] == (event,)


@pytest.mark.parametrize("failure_type", (RuntimeError, KeyboardInterrupt))
def test_runtime_collective_failure_retains_prepared_provider_graph(
    monkeypatch,
    failure_type,
):
    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-runtime-two-phase-{}".format(failure_type.__name__)
    )
    collective = runtime.collective
    failure = failure_type("collective close failed after provider prepare")
    before = (
        runtime._active_provider,
        runtime.collective,
        provider.runtime,
        provider._active_lease,
        provider._cache,
        provider._cache_factory,
        provider._pool_factory,
        provider._scheduler_factory,
        lease._provider,
        lease._resource_record,
        lease.pool,
        lease.scheduler,
        lease._cache_reservation,
        lease._store_reservation,
    )

    def fail_collective(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(
        collective,
        "_close_for_runtime",
        fail_collective,
        raising=False,
    )
    try:
        with pytest.raises(failure_type) as caught:
            runtime.close()
        assert caught.value is failure
        assert (
            runtime._active_provider,
            runtime.collective,
            provider.runtime,
            provider._active_lease,
            provider._cache,
            provider._cache_factory,
            provider._pool_factory,
            provider._scheduler_factory,
            lease._provider,
            lease._resource_record,
            lease.pool,
            lease.scheduler,
            lease._cache_reservation,
            lease._store_reservation,
        ) == before
        with runtime._terminal_gate._condition:
            assert runtime._terminal_gate._fatal_transition.primary is failure
    finally:
        _repair_test_abandoned_close_entry(runtime._terminal_gate)
        runtime._closed = True
        if not store.closed:
            store.close()


@pytest.mark.parametrize("failure_boundary", ("before_real", "after_real"))
def test_runtime_provider_finalize_failure_restores_attached_graph(
    monkeypatch,
    failure_boundary,
):
    class FinalizeFailure(BaseException):
        pass

    runtime, _, _, store, provider, lease = _open_active_case(
        store_id="terminal-provider-finalize-{}".format(failure_boundary)
    )
    failure = FinalizeFailure("provider finalize {} failed".format(failure_boundary))
    before = (
        runtime._active_provider,
        runtime.collective,
        provider.runtime,
        provider._active_lease,
        provider._cache,
        provider._cache_factory,
        provider._pool_factory,
        provider._scheduler_factory,
        lease._provider,
        lease._resource_record,
        lease.pool,
        lease.scheduler,
        lease._cache_reservation,
        lease._store_reservation,
    )
    original_close = provider._close_for_runtime
    record = lease._resource_record

    def record_graph():
        slots = []
        for name, slot in sorted(record._construction_slots.items()):
            snapshot = slot.snapshot()
            slots.append(
                (
                    name,
                    id(slot),
                    id(snapshot.resource),
                    snapshot.kind,
                    tuple(
                        (
                            retained.identity,
                            id(retained.owner),
                            retained.capacity_bytes,
                        )
                        for retained in snapshot.records
                    ),
                    tuple(id(event) for event in snapshot.events),
                    tuple(id(stream) for stream in snapshot.streams),
                )
            )
        return (
            tuple(id(resource) for resource in record._resources),
            tuple(
                (name, id(resource))
                for name, resource in sorted(
                    record._construction_resources.items()
                )
            ),
            tuple(slots),
            tuple(
                (identity, id(receipt))
                for identity, receipt in sorted(
                    record._consumed_receipts.items()
                )
            ),
            tuple(
                (identity, id(retained.owner), retained.capacity_bytes)
                for identity, retained in sorted(
                    record._allocations.items(), key=lambda item: repr(item[0])
                )
            ),
            tuple(
                (identity, id(retained.owner), retained.capacity_bytes)
                for identity, retained in sorted(
                    record._cache_allocations.items(),
                    key=lambda item: repr(item[0]),
                )
            ),
            tuple(
                (identity, id(retained.owner), retained.capacity_bytes)
                for identity, retained in sorted(
                    record._pinned_allocations.items(),
                    key=lambda item: repr(item[0]),
                )
            ),
            tuple(
                (identity, id(event))
                for identity, event in sorted(record._events.items())
            ),
            tuple(
                (identity, id(stream))
                for identity, stream in sorted(record._streams.items())
            ),
        )

    record_before = record_graph()

    def close_with_failing_finalize(**kwargs):
        action = original_close(**kwargs)
        real_finalize = action.finalize

        def fail_finalize():
            if failure_boundary == "after_real":
                real_finalize()
            raise failure

        action.finalize = fail_finalize
        return action

    monkeypatch.setattr(provider, "_close_for_runtime", close_with_failing_finalize)
    try:
        with pytest.raises(FinalizeFailure) as caught:
            runtime.close()
        assert caught.value is failure
        assert (
            runtime._active_provider,
            runtime.collective,
            provider.runtime,
            provider._active_lease,
            provider._cache,
            provider._cache_factory,
            provider._pool_factory,
            provider._scheduler_factory,
            lease._provider,
            lease._resource_record,
            lease.pool,
            lease.scheduler,
            lease._cache_reservation,
            lease._store_reservation,
        ) == before
        assert lease._resource_record is record
        assert record_graph() == record_before
        with runtime._terminal_gate._condition:
            assert runtime._terminal_gate._runtime_close_result is failure
            assert runtime._terminal_gate.phase is _TerminalPhase.RUNTIME_CLOSED
        assert runtime._terminal_error is failure
    finally:
        _repair_test_abandoned_close_entry(runtime._terminal_gate)
        runtime._closed = True
        if not store.closed:
            store.close()


@pytest.mark.parametrize("candidate", ("runtime_scope", "wrong_operation"))
def test_real_managed_guard_rejects_active_noncanonical_admission(candidate):
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-managed-canonical-{}".format(candidate)
    )
    gate = runtime._terminal_gate
    if candidate == "runtime_scope":
        token = gate.admit_runtime("resource_state")
        validator = runtime._exact_admission_validator(
            token,
            scope="runtime",
            epoch=None,
            operation=token.operation,
        )
    else:
        token = gate.admit_lease(lease._epoch, "unrelated_managed_probe")
        validator = runtime._exact_admission_validator(
            token,
            scope="lease",
            epoch=lease._epoch,
            operation=token.operation,
        )
    before = (
        lease._status_workspace._closed,
        lease.scheduler._closed,
        lease.pool._closed,
        lease._store_reservation._closed,
    )
    try:
        with pytest.raises(RuntimeError, match="managed resource admission"):
            lease._status_workspace.close(
                _admission_token=token,
                _admission_validator=validator,
            )
        with pytest.raises(RuntimeError, match="managed resource admission"):
            lease._store_reservation.close(
                _admission_token=token,
                _admission_validator=validator,
            )
        assert (
            lease._status_workspace._closed,
            lease.scheduler._closed,
            lease.pool._closed,
            lease._store_reservation._closed,
        ) == before
    finally:
        gate.release(token)
        _close_case(runtime, store)


def test_real_managed_descendant_guard_binds_operation_and_lease_epoch():
    runtime, request, plan, store, provider, first = _open_managed_active_case(
        store_id="terminal-managed-descendant-epoch"
    )
    gate = runtime._terminal_gate
    identity = first.cache_identities[0]
    token = gate.admit_lease(first._epoch, "acquire")
    validator = first._exact_admission_validator(token)
    cache_lease = provider._cache.acquire(
        identity,
        _admission_token=token,
        _admission_validator=validator,
    )
    gate.release(token)
    status = first._status_workspace
    reservation = first._cache_reservation
    entry = cache_lease._entry

    wrong = gate.admit_lease(first._epoch, "mark_dirty")
    wrong_validator = first._exact_admission_validator(wrong)
    try:
        with pytest.raises(RuntimeError, match="managed resource admission"):
            cache_lease.close(
                _admission_token=wrong,
                _admission_validator=wrong_validator,
            )
        assert cache_lease._entry is entry
        assert cache_lease._closed is False
        assert entry.refcount == 1
    finally:
        gate.release(wrong)

    with first._lease_admission("child_close") as close_token:
        cache_lease.close(
            _admission_token=close_token,
            _admission_validator=first._exact_admission_validator(close_token),
        )
    first.close()

    runtime_token = gate.admit_runtime("resource_state")
    runtime_validator = runtime._exact_admission_validator(
        runtime_token,
        scope="runtime",
        epoch=None,
        operation=runtime_token.operation,
    )
    try:
        state = provider.runtime_resource_state(
            _admission_token=runtime_token,
            _admission_validator=runtime_validator,
        )
        assert state["active_leases"] == 0
        with pytest.raises(RuntimeError, match="scope is not authorized"):
            _ = provider.cache
    finally:
        gate.release(runtime_token)

    second_receipt = runtime.preflight_residency(request, plan)
    second = provider.open_working_set(
        request,
        plan,
        store,
        second_receipt,
    ).__enter__()
    assert second._epoch != first._epoch
    try:
        with second._lease_admission("resource_state"):
            with pytest.raises(RuntimeError, match="lease epoch"):
                _ = status.device_status
            with pytest.raises(RuntimeError, match="lease epoch"):
                _ = reservation.allowlist
    finally:
        _close_case(runtime, store)


def test_mark_dirty_rejects_canonical_resource_state_before_mutation():
    runtime, request, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-mark-dirty-exact-operation"
    )
    gate = runtime._terminal_gate
    output = runtime.backend.zeros((4,), dtype=np.float64)
    before = (
        lease._dirty,
        lease._dirty_allocation_records,
        dict(lease._resource_record._allocations),
    )
    token = gate.admit_lease(lease._epoch, "resource_state")
    try:
        with pytest.raises(RuntimeError, match="required operation"):
            lease.mark_dirty(
                "output",
                output,
                _admission_token=token,
            )
        assert (
            lease._dirty,
            lease._dirty_allocation_records,
            dict(lease._resource_record._allocations),
        ) == before
    finally:
        gate.release(token)

    try:
        lease.mark_dirty("output", output)
        assert lease._dirty == (dict(request.host_refs)["output"], output)
        assert lease._dirty_allocation_records == (allocation_record(output),)
    finally:
        _close_case(runtime, store)


def test_real_managed_local_operator_accepts_exact_operator_call_context():
    runtime, request, plan, store, _, lease = _open_managed_active_case(
        store_id="terminal-managed-local-operator"
    )
    distributed = request.distributed_plan
    operator = DistributedLocalOperator(
        plan=distributed,
        provider=lease,
        collective=runtime.collective,
        counters={},
        backend=runtime.backend,
        context=runtime.context,
        source_bindings=ExecutionBindings(
            {"input_0": dict(request.host_refs)["input_0"]}
        ),
        residency_request=request,
        residency_plan=plan,
        residency_receipt=lease.receipt,
        mesh=runtime.mesh,
    )
    local_vector = runtime.backend.ones(
        distributed.input_sharding.local_shape(runtime.rank),
        dtype=np.float64,
    )
    try:
        result = operator(local_vector)
        assert result.shape == distributed.output_sharding.local_shape(
            runtime.rank
        )
        assert lease._poisoned_error is None
        assert runtime._terminal_gate._fatal_transition is None
    finally:
        _close_case(runtime, store)


def test_operator_nested_acquire_requires_exact_token_and_owning_thread():
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-operator-token-thread-local"
    )
    request = _block_request(lease)
    operator_entered = threading.Event()
    run_nested = threading.Event()
    nested_ready = threading.Event()
    release_operator = threading.Event()
    ordinary_ready = threading.Event()
    release_ordinary = threading.Event()
    operator_observed = []
    ordinary_observed = []
    nested_observed = []

    def run_operator():
        with lease._operator_call(np.ones(4, dtype=np.float64)) as call:
            token = _current_token(runtime)
            operator_observed.append((threading.get_ident(), token, call.owner))
            operator_entered.set()
            assert run_nested.wait(_TIMEOUT_S)
            child = lease.acquire(request)
            nested_observed.append(
                (
                    threading.get_ident(),
                    child._admission_token,
                    child._shared_call,
                    child._compute_handle,
                    lease._active_operator_owner,
                )
            )
            child.close()
            nested_ready.set()
            assert release_operator.wait(_TIMEOUT_S)

    def run_ordinary():
        try:
            child = lease.acquire(request)
            handle = child._compute_handle
            ordinary_observed.append(
                (
                    threading.get_ident(),
                    child._admission_token,
                    child._shared_call,
                    None if handle is None else handle._owner,
                    lease._active_operator_owner,
                )
            )
            ordinary_ready.set()
            assert release_ordinary.wait(_TIMEOUT_S)
            child.close()
        finally:
            ordinary_ready.set()

    operator, _, operator_errors, operator_done = _start(run_operator)
    ordinary = None
    ordinary_errors = []
    ordinary_done = threading.Event()
    try:
        assert operator_entered.wait(_TIMEOUT_S)
        ordinary, _, ordinary_errors, ordinary_done = _start(run_ordinary)
        assert ordinary_ready.wait(_TIMEOUT_S)
        release_ordinary.set()
        _join(ordinary, ordinary_done)
        run_nested.set()
        assert nested_ready.wait(_TIMEOUT_S)
        release_operator.set()
        _join(operator, operator_done)
    finally:
        release_ordinary.set()
        run_nested.set()
        release_operator.set()
        if ordinary is not None and ordinary.is_alive():
            _join(ordinary, ordinary_done)
        if operator.is_alive():
            _join(operator, operator_done)
        _close_case(runtime, store)

    assert operator_errors == []
    assert ordinary_errors == []
    assert len(operator_observed) == 1
    assert len(ordinary_observed) == 1
    assert len(nested_observed) == 1
    operator_thread, operator_token, operator_owner = operator_observed[0]
    ordinary_thread, ordinary_token, shared, compute_owner, active_owner = (
        ordinary_observed[0]
    )
    nested_thread, nested_token, nested_shared, nested_handle, nested_owner = (
        nested_observed[0]
    )
    assert ordinary_thread != operator_thread
    assert ordinary_token.operation == "acquire"
    assert ordinary_token is not operator_token
    assert shared is False
    assert compute_owner is not None
    assert compute_owner is not operator_owner
    assert active_owner is operator_owner
    assert nested_thread == operator_thread
    assert nested_token is operator_token
    assert nested_shared is True
    assert nested_handle is None
    assert nested_owner is operator_owner


def test_child_close_rejects_schedule_writeback_before_state_mutation():
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-child-close-exact-operation"
    )
    gate = runtime._terminal_gate
    child = lease.acquire(_block_request(lease))
    cache_leases = child._cache_leases
    before = (
        child._working_set,
        child.bindings,
        child._cache_leases,
        child._compute_handle,
        child._admission_token,
        child._shared_call,
        child._closed,
        child in lease._children,
        tuple(cache_lease._entry.refcount for cache_lease in cache_leases),
    )
    transition, elected = gate.begin_lease_close(lease._epoch)
    assert elected is True
    sibling = gate.admit_lease_close(transition, "schedule_writeback")
    try:
        with pytest.raises(RuntimeError, match="required operation"):
            child.close(_admission_token=sibling)
        assert (
            child._working_set,
            child.bindings,
            child._cache_leases,
            child._compute_handle,
            child._admission_token,
            child._shared_call,
            child._closed,
            child in lease._children,
            tuple(cache_lease._entry.refcount for cache_lease in cache_leases),
        ) == before
    finally:
        gate.release(sibling)
        if not child._closed:
            cleanup = gate.admit_lease_close(transition, "child_close")
            try:
                child.close(_admission_token=cleanup)
            finally:
                gate.release(cleanup)
        _repair_test_abandoned_close_entry(gate)
        runtime._closed = True
        reservation = lease._store_reservation
        if reservation is not None and not reservation._closed:
            reservation._store._release_reservation(reservation)
            reservation._closed = True
        if not store.closed:
            store.close()


def test_preflight_admission_cannot_install_provider_before_mutation():
    runtime = _runtime()
    request, _, store, _, _ = _active_case(
        runtime,
        store_id="terminal-provider-install-exact-operation",
    )
    gate = runtime._terminal_gate
    construct = gate.admit_runtime_setup("provider_construct")
    construct_validator = runtime._exact_admission_validator(
        construct,
        scope="runtime_setup",
        epoch=None,
        operation="provider_construct",
    )
    provider = ActiveWorkingSetProvider(
        runtime,
        device_budget_resolution=request.device_budget,
        host_budget_resolution=request.host_budget,
        prefetch_depth=request.prefetch_depth,
        _admission_token=construct,
        _admission_validator=construct_validator,
        _standalone=False,
    )
    gate.release(construct)
    preflight = gate.admit_runtime_setup("preflight_residency")
    preflight_validator = runtime._exact_admission_validator(
        preflight,
        scope="runtime_setup",
        epoch=None,
        operation="preflight_residency",
    )
    try:
        with pytest.raises(RuntimeError, match="required operation"):
            runtime._install_active_provider(
                provider,
                _admission_token=preflight,
                _admission_validator=preflight_validator,
            )
        assert runtime._active_provider is None
    finally:
        gate.release(preflight)

    install = gate.admit_runtime_setup("provider_install")
    install_validator = runtime._exact_admission_validator(
        install,
        scope="runtime_setup",
        epoch=None,
        operation="provider_install",
    )
    runtime._install_active_provider(
        provider,
        _admission_token=install,
        _admission_validator=install_validator,
    )
    gate.release(install)
    execution = gate.admit_runtime_setup("execution_config")
    execution_validator = runtime._exact_admission_validator(
        execution,
        scope="runtime_setup",
        epoch=None,
        operation="execution_config",
    )
    try:
        assert runtime._install_active_provider(
            provider,
            _admission_token=execution,
            _admission_validator=execution_validator,
        ) is provider
        assert provider.matches_config(
            request.device_budget,
            request.host_budget,
            request.prefetch_depth,
            _admission_token=execution,
            _admission_validator=execution_validator,
        )
    finally:
        gate.release(execution)
    budget = gate.admit_runtime_setup("budget_probe")
    budget_validator = runtime._exact_admission_validator(
        budget,
        scope="runtime_setup",
        epoch=None,
        operation="budget_probe",
    )
    try:
        resolution = runtime._resolve_budget(
            "device",
            1024,
            _admission_token=budget,
            _admission_validator=budget_validator,
        )
        assert resolution.resolved_bytes == 1024
    finally:
        gate.release(budget)
        _close_case(runtime, store)


def test_real_managed_nonblocking_close_accepts_exact_close_progress_context():
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-managed-close-progress"
    )
    try:
        lease.close(wait=False)
        assert lease._poisoned_error is None
        assert lease._closed is False
        assert runtime._terminal_gate._fatal_transition is None
        assert runtime._terminal_gate._lease_state(lease._epoch).phase == "open"
    finally:
        _close_case(runtime, store)


def test_managed_host_reservation_accepts_exact_lease_close_admission():
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-managed-host-reservation-close"
    )
    gate = runtime._terminal_gate
    reservation = lease._store_reservation
    transition, elected = gate.begin_lease_close(lease._epoch)
    assert elected is True
    wrong = gate.admit_lease_close(transition, "status_close")
    wrong_validator = runtime._exact_admission_validator(
        wrong,
        scope="lease_close",
        epoch=lease._epoch,
        operation=wrong.operation,
        transition_sequence=transition.sequence,
    )
    try:
        with pytest.raises(RuntimeError, match="managed resource admission"):
            reservation.close(
                _admission_token=wrong,
                _admission_validator=wrong_validator,
            )
        assert reservation._closed is False
        assert id(reservation) in store._reservations
    finally:
        gate.release(wrong)

    token = gate.admit_lease_close(transition, "store_reservation_close")
    validator = runtime._exact_admission_validator(
        token,
        scope="lease_close",
        epoch=lease._epoch,
        operation=token.operation,
        transition_sequence=transition.sequence,
    )
    try:
        reservation.close(
            _admission_token=token,
            _admission_validator=validator,
        )
        assert reservation._closed is True
        assert id(reservation) not in store._reservations
    finally:
        gate.release(token)
        _repair_test_abandoned_close_entry(gate)
        runtime._closed = True
        if not store.closed:
            store.close()


def test_managed_counted_prestart_requires_exact_elected_close_transition():
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-managed-counted-prestart"
    )
    gate = runtime._terminal_gate
    with lease._lease_admission("operator_call"):
        owner = lease.scheduler._register_owner(
            "compute",
            defer_counted_completion=True,
        )
        owner.mark_enqueued()
    assert owner._async_start_pending is False
    try:
        with pytest.raises(TypeError, match="close transition"):
            lease.scheduler._start_counted_completions()
        assert owner._async_start_pending is False
        with pytest.raises(TypeError, match="managed resource requires admission"):
            owner._prepare_counted_completion(wait=False)
        assert owner._async_start_pending is False

        transition, elected = gate.begin_lease_close(lease._epoch)
        assert elected is True
        wrong_transition = replace(
            transition,
            sequence=transition.sequence + 1,
        )
        with pytest.raises(RuntimeError, match="close transition"):
            lease.scheduler._start_counted_completions(
                _close_transition=wrong_transition,
            )
        assert owner._async_start_pending is False
        lease.scheduler._start_counted_completions(
            _close_transition=transition,
        )
        assert owner._async_start_pending is True
    finally:
        admission = owner._async_admission
        if admission is not None:
            admission.cancel()
        _repair_test_abandoned_close_entry(gate)
        runtime._closed = True
        reservation = lease._store_reservation
        if reservation is not None and not reservation._closed:
            reservation._store._release_reservation(reservation)
            reservation._closed = True
        if not store.closed:
            store.close()


@pytest.mark.parametrize("producer", ("pinned", "cache"))
def test_provider_allocation_producer_publishes_before_after_real_wrapper(
    monkeypatch,
    producer,
):
    class ProducerFailure(BaseException):
        pass

    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-real-{}-producer".format(producer),
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    failure = ProducerFailure("{} producer wrapper failed".format(producer))
    observed = []

    if producer == "pinned":
        original = provider._pinned_allocator

        def fail_after_real(*args, **kwargs):
            allocated = original(*args, **kwargs)
            slot = kwargs.get("_construction_slot")
            observed.append((allocated, None if slot is None else slot.snapshot()))
            raise failure

        monkeypatch.setattr(provider, "_pinned_allocator", fail_after_real)
    try:
        if producer == "pinned":
            with pytest.raises(ProducerFailure) as caught:
                provider.open_working_set(request, plan, store, receipt)
            assert caught.value is failure
        else:
            lease = provider.open_working_set(
                request, plan, store, receipt
            ).__enter__()
            original = provider._cache._allocator

            def fail_after_real(*args, **kwargs):
                allocated = original(*args, **kwargs)
                slot = kwargs.get("_construction_slot")
                observed.append(
                    (allocated, None if slot is None else slot.snapshot())
                )
                raise failure

            provider._cache._allocator = fail_after_real
            with pytest.raises(ProducerFailure) as caught:
                lease._load_identity(lease.cache_identities[0])
            assert caught.value is failure

        assert len(observed) == 1
        allocated, snapshot = observed[0]
        assert snapshot is not None
        physical = (
            (allocated.array, allocated.transfer_array)
            if hasattr(allocated, "transfer_array")
            else (allocated,)
        )
        expected = {allocation_record(array).identity for array in physical}
        assert expected
        assert expected.issubset(
            {retained.identity for retained in snapshot.records}
        )
    finally:
        _repair_test_construction_state(runtime._terminal_gate)
        _close_case(runtime, store)


def test_pageable_fallback_producer_publishes_before_after_real_failure(
    monkeypatch,
):
    class PageableProducerFailure(BaseException):
        pass

    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-pageable-fallback-producer",
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    failure = PageableProducerFailure("pageable producer wrapper failed")
    observed = []
    original_pageable = pinned_module._pageable_array

    def fail_pinned(*_args, **_kwargs):
        raise MemoryError("force pageable fallback")

    def fail_after_real(nbytes, **kwargs):
        allocated = original_pageable(nbytes, **kwargs)
        observed.append(
            (
                allocated,
                kwargs.get("_construction_slot"),
                provider._provisional_resources,
            )
        )
        raise failure

    monkeypatch.setattr(provider, "_pinned_allocator", fail_pinned)
    monkeypatch.setattr(pinned_module, "_pageable_array", fail_after_real)
    try:
        with pytest.raises(PageableProducerFailure) as caught:
            provider.open_working_set(request, plan, store, receipt)
        assert caught.value is failure
        assert len(observed) == 1
        allocated, slot, record = observed[0]
        assert slot is not None
        assert record is not None
        expected = allocation_record(allocated)
        retained = {
            candidate.identity: candidate
            for candidate in record.pinned_allocations
        }
        assert expected.identity in retained
        assert retained[expected.identity].owner is expected.owner
        assert retained[expected.identity].capacity_bytes == expected.capacity_bytes
    finally:
        _repair_test_construction_state(runtime._terminal_gate)
        _close_case(runtime, store)


def test_pinned_after_real_provider_failure_does_not_fallback_or_lose_owner(
    monkeypatch,
):
    class AfterRealPinnedFailure(Exception):
        pass

    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-pinned-after-real-no-fallback",
    )
    receipt = runtime.preflight_residency(request, plan)
    provider = _provider(runtime, request)
    failure = AfterRealPinnedFailure("pinned producer wrapper failed after return")
    original_pinned = provider._pinned_allocator
    original_pageable = pinned_module._pageable_array
    observed = []
    pageable_allocations = []
    opened = []

    def fail_after_real(*args, **kwargs):
        allocated = original_pinned(*args, **kwargs)
        slot = kwargs["_construction_slot"]
        observed.append((allocation_record(allocated), slot.snapshot()))
        raise failure

    def allocate_pageable(*args, **kwargs):
        allocated = original_pageable(*args, **kwargs)
        pageable_allocations.append(allocation_record(allocated))
        return allocated

    monkeypatch.setattr(provider, "_pinned_allocator", fail_after_real)
    monkeypatch.setattr(pinned_module, "_pageable_array", allocate_pageable)
    try:
        with pytest.raises(AfterRealPinnedFailure) as caught:
            opened.append(
                provider.open_working_set(request, plan, store, receipt).__enter__()
            )
        assert caught.value is failure
        assert pageable_allocations == []
        assert len(observed) == 1
        physical, snapshot = observed[0]
        retained = {
            candidate.identity: candidate for candidate in snapshot.records
        }
        assert tuple(retained) == (physical.identity,)
        assert retained[physical.identity].owner is physical.owner
        assert retained[physical.identity].capacity_bytes == physical.capacity_bytes
        assert physical.capacity_bytes == request.transfer_profile.rank_staging_bytes[0]
    finally:
        for lease in opened:
            try:
                lease.close()
            except BaseException:
                pass
        _repair_test_construction_state(runtime._terminal_gate)
        _close_case(runtime, store)


@pytest.mark.parametrize(
    ("producer", "identity_kind"),
    (("pinned", "host"), ("cache", "device")),
)
def test_fake_cupy_producer_publishes_raw_owner_before_wrapper_failure(
    monkeypatch,
    producer,
    identity_kind,
):
    runtime = _runtime()
    gate = runtime._terminal_gate
    transaction = gate._prepare_lease_construction(
        "fake_cupy_{}_producer".format(producer),
        resource_names=(producer,),
    )
    _, token = gate.begin_lease(
        "fake_cupy_{}_producer".format(producer),
        _transaction=transaction,
    )
    backend = _FakeCupyBackend()
    runtime.backend = backend
    pointer = 0x45670000 if producer == "cache" else 0x76540000
    size = 64

    class RawMemory:
        def __init__(self, nbytes):
            self.ptr = pointer
            self.size = nbytes

    class MemoryPointer:
        def __init__(self, memory, offset):
            self.mem = memory
            self.ptr = memory.ptr + offset

    class DeviceArray:
        def __init__(self, memory_pointer, shape, dtype):
            self.data = memory_pointer
            self.shape = shape
            self.dtype = np.dtype(dtype)
            self.nbytes = size

    class HostArray:
        def __init__(self, memory_pointer, count):
            self.base = memory_pointer
            self.shape = (count,)
            self.dtype = np.dtype(np.uint8)
            self.nbytes = count

    if producer == "cache":
        backend._cupy.cuda.Memory = RawMemory
        backend._cupy.cuda.MemoryPointer = MemoryPointer
        backend._cupy.ndarray = (
            lambda shape, dtype, memptr, order: DeviceArray(
                memptr,
                shape,
                dtype,
            )
        )
    else:
        backend._cupy.cuda.PinnedMemory = RawMemory
        backend._cupy.cuda.PinnedMemoryPointer = MemoryPointer
        monkeypatch.setattr(
            providers_module.np,
            "frombuffer",
            lambda memory_pointer, dtype, count: HostArray(
                memory_pointer,
                count,
            ),
        )

    provider = ActiveWorkingSetProvider(
        runtime,
        device_budget_resolution=_explicit_budget(1024, "device"),
        host_budget_resolution=_explicit_budget(1024, "host"),
        _standalone=True,
    )
    slot = transaction.resource_slots[producer]
    failure = RuntimeError("fake CuPy producer wrapper failed")
    observed = []

    def fail_after_real():
        if producer == "cache":
            allocated = provider._empty_cache_array(
                SimpleNamespace(
                    shape=(8,),
                    dtype=np.dtype(np.float64),
                    nbytes=size,
                ),
                order="C",
                _construction_slot=slot,
            )
        else:
            allocated = provider._pinned_allocator(
                size,
                _construction_slot=slot,
            )
        observed.append((allocated, slot.snapshot()))
        raise failure

    try:
        with pytest.raises(RuntimeError) as caught:
            fail_after_real()
        assert caught.value is failure
        allocated, snapshot = observed[0]
        record = allocation_record(allocated)
        assert record.identity == (identity_kind, pointer)
        assert tuple(retained.identity for retained in snapshot.records) == (
            record.identity,
        )
        assert snapshot.records[0].owner is record.owner
        assert snapshot.records[0].capacity_bytes == size
    finally:
        gate.release(token)
        gate._abort_lease_construction(
            transaction,
            failure,
            lambda: None,
            lambda _primary: None,
        )
        runtime._active_provider = None
        runtime._closed = True


@pytest.mark.parametrize(
    "operation",
    ("wait_for_admissions", "publish_fatal", "wait_for_published"),
)
def test_fatal_gate_waits_reuse_transition_deadline(monkeypatch, operation):
    runtime = _runtime()
    gate = runtime._terminal_gate
    primary = RuntimeError("fatal deadline owner")
    transition = gate.begin_fatal(primary)
    captured = []

    def capture(_predicate, deadline, _message, **_kwargs):
        captured.append(deadline)

    monkeypatch.setattr(gate, "_wait_until_deadline", capture)
    monkeypatch.setattr(
        gate,
        "_deadline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fatal stage created a fresh deadline")
        ),
    )
    try:
        if operation == "wait_for_admissions":
            gate.wait_for_admissions(transition, 999)
        elif operation == "publish_fatal":
            gate.publish_fatal(transition, object())
        else:
            gate.wait_for_published(999)
        assert captured == [transition.deadline]
    finally:
        runtime._closed = True


@pytest.mark.parametrize(
    "family,parent_operation",
    (
        ("h2d_completion", "load"),
        ("d2h_completion", "close_progress"),
        ("compute_completion", "acquire"),
    ),
)
@pytest.mark.parametrize(
    "allowed_family",
    ("h2d_completion", "d2h_completion", "compute_completion"),
)
def test_provider_counted_async_semantic_family_matrix(
    family,
    parent_operation,
    allowed_family,
):
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-semantic-async-{}".format(family)
    )
    gate = runtime._terminal_gate
    observed = []
    admission = None
    try:
        with lease._lease_admission(parent_operation):
            admission = lease._spawn_async_admission(object(), family)
            sequence = admission.token.sequence

        def inspect(*, _admission_token, _admission_validator):
            lease.scheduler._require_admission(
                _admission_token,
                _admission_validator,
                allowed_operations=(allowed_family,),
            )
            observed.append(_admission_token)

        if family == allowed_family:
            admission.run(family, inspect)
            assert len(observed) == 1
            claimed = observed[0]
            assert claimed.operation == family
            assert claimed.sequence == sequence
        else:
            with pytest.raises(RuntimeError, match="operation is not authorized"):
                admission.run(family, inspect)
            assert observed == []
        with gate._condition:
            assert gate._tokens[sequence].status == "released"
    finally:
        if admission is not None:
            admission.cancel()
        _close_case(runtime, store)


@pytest.mark.parametrize(
    "family,parent_operation,commits",
    (
        ("h2d_completion", "load", False),
        ("d2h_completion", "close_progress", True),
        ("compute_completion", "acquire", False),
    ),
)
def test_only_d2h_counted_async_can_commit_host_reservation(
    family,
    parent_operation,
    commits,
):
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-semantic-reservation-{}".format(family)
    )
    reservation = lease._store_reservation
    dirty_ref = reservation._dirty_ref
    value = np.zeros(dirty_ref.shape, dtype=np.dtype(dirty_ref.dtype))
    before = store.snapshot()
    admission = None
    try:
        with lease._lease_admission(parent_operation):
            admission = lease._spawn_async_admission(object(), family)

        def commit(*, _admission_token, _admission_validator):
            return reservation.commit(
                dirty_ref,
                value,
                _admission_token=_admission_token,
                _admission_validator=_admission_validator,
            )

        if commits:
            updated = admission.run(family, commit)
            assert updated.version == dirty_ref.version + 1
            assert store.snapshot() != before
            assert reservation._committed is True
        else:
            with pytest.raises(RuntimeError, match="operation is not authorized"):
                admission.run(family, commit)
            assert store.snapshot() == before
            assert reservation._committed is False
    finally:
        if admission is not None:
            admission.cancel()
        _close_case(runtime, store)


def test_direct_child_close_rejects_sibling_before_any_state_mutation():
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-direct-child-close-operation"
    )
    gate = runtime._terminal_gate
    child = lease.acquire(_block_request(lease))
    cache_leases = child._cache_leases
    transition, elected = gate.begin_lease_close(lease._epoch)
    assert elected is True
    sibling = gate.admit_lease_close(transition, "schedule_writeback")
    validator = runtime._exact_admission_validator(
        sibling,
        scope="lease_close",
        epoch=lease._epoch,
        operation="schedule_writeback",
        transition_sequence=transition.sequence,
    )

    def snapshot():
        entries = tuple(cache_lease._entry for cache_lease in cache_leases)
        return (
            child._working_set,
            child.bindings,
            child._cache_leases,
            child._compute_handle,
            child._admission_token,
            child._shared_call,
            child._closed,
            child in lease._children,
            entries,
            tuple(None if entry is None else entry.refcount for entry in entries),
        )

    before = snapshot()
    try:
        with pytest.raises(RuntimeError, match="operation"):
            child._close_admitted(sibling, validator)
        assert snapshot() == before
    finally:
        gate.release(sibling)
        if not child._closed:
            cleanup = gate.admit_lease_close(transition, "child_close")
            try:
                child.close(_admission_token=cleanup)
            finally:
                gate.release(cleanup)
        _repair_test_abandoned_close_entry(gate)
        runtime._closed = True
        reservation = lease._store_reservation
        if reservation is not None and not reservation._closed:
            reservation._store._release_reservation(reservation)
            reservation._closed = True
        if not store.closed:
            store.close()


def test_shared_operator_child_rejects_foreign_thread_close_before_mutation():
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-shared-child-foreign-close"
    )
    operator_ready = threading.Event()
    foreign_done = threading.Event()
    release_operator = threading.Event()
    shared = []
    before = []

    def run_operator():
        with lease._operator_call(np.ones(4, dtype=np.float64)):
            child = lease.acquire(_block_request(lease))
            shared.append(child)
            before.append(
                (
                    child._working_set,
                    child.bindings,
                    child._cache_leases,
                    child._admission_token,
                    child._shared_call,
                    child._closed,
                    child in lease._children,
                )
            )
            operator_ready.set()
            assert foreign_done.wait(_TIMEOUT_S)
            assert (
                child._working_set,
                child.bindings,
                child._cache_leases,
                child._admission_token,
                child._shared_call,
                child._closed,
                child in lease._children,
            ) == before[0]
            child.close()
            assert release_operator.wait(_TIMEOUT_S)

    operator, _, operator_errors, operator_done = _start(run_operator)
    foreign = None
    foreign_errors = []
    foreign_thread_done = threading.Event()
    try:
        assert operator_ready.wait(_TIMEOUT_S)
        foreign, _, foreign_errors, foreign_thread_done = _start(shared[0].close)
        _join(foreign, foreign_thread_done)
        foreign_done.set()
        release_operator.set()
        _join(operator, operator_done)
        assert len(foreign_errors) == 1
        assert isinstance(foreign_errors[0], RuntimeError)
        assert operator_errors == []
    finally:
        foreign_done.set()
        release_operator.set()
        if foreign is not None and foreign.is_alive():
            _join(foreign, foreign_thread_done)
        if operator.is_alive():
            _join(operator, operator_done)
        _close_case(runtime, store)


@pytest.mark.parametrize(
    "helper,legitimate_operation",
    (
        ("pool_checkout", "acquire"),
        ("cache_contains", "prefetch"),
        ("scheduler_compute", "operator_call"),
        ("owner_capture", "acquire"),
    ),
)
def test_managed_lower_helper_admission_matrix(
    helper,
    legitimate_operation,
):
    runtime, _, _, store, provider, lease = _open_managed_active_case(
        store_id="terminal-fail-closed-lower-{}".format(helper)
    )
    gate = runtime._terminal_gate
    owner = None
    if helper == "owner_capture":
        with lease._lease_admission("acquire"):
            owner = lease.scheduler._register_owner(
                "compute",
                defer_counted_admission=True,
            )
    sibling = gate.admit_lease(lease._epoch, "mark_dirty")
    sibling_validator = lease._exact_admission_validator(sibling)
    marker = object()
    if helper == "pool_checkout":
        before = (lease.pool._slot._checked_out, lease.pool._checked_out_bytes)

        def invoke(token, validator):
            return lease.pool.checkout(
                1,
                _admission_token=token,
                _admission_validator=validator,
            )

        def snapshot():
            return (lease.pool._slot._checked_out, lease.pool._checked_out_bytes)

    elif helper == "cache_contains":
        before = dict(provider._cache._entries)

        def invoke(token, validator):
            return provider._cache.contains(
                lease.cache_identities[0],
                _admission_token=token,
                _admission_validator=validator,
            )

        def snapshot():
            return dict(provider._cache._entries)

    elif helper == "scheduler_compute":
        before = dict(lease.scheduler._owners)

        def invoke(token, validator):
            return lease.scheduler.begin_compute(
                _admission_token=token,
                _admission_validator=validator,
            )

        def snapshot():
            return dict(lease.scheduler._owners)

    else:
        before = owner._resources

        def invoke(token, validator):
            return owner.capture_resources(
                marker,
                _admission_token=token,
                _admission_validator=validator,
            )

        def snapshot():
            return owner._resources

    try:
        with pytest.raises((TypeError, RuntimeError), match="operation"):
            invoke(sibling, sibling_validator)
        assert snapshot() == before
        gate.release(sibling)
        sibling = None
        with lease._lease_admission(legitimate_operation) as exact:
            exact_validator = lease._exact_admission_validator(exact)
            result = invoke(exact, exact_validator)
            if helper == "pool_checkout":
                slot = result.__enter__()
                assert snapshot() != before
                assert slot is lease.pool._slot
                result.__exit__(None, None, None)
            elif helper == "cache_contains":
                assert type(result) is bool
            elif helper == "scheduler_compute":
                assert result.owner is lease.scheduler._owners[id(result.owner)]
            else:
                assert marker in snapshot()
    finally:
        if sibling is not None:
            gate.release(sibling)
        _close_case(runtime, store)


def test_every_managed_lower_helper_declares_operations_explicitly():
    managed_classes = (
        AsyncCompletionHandle,
        AsyncResourceOwner,
        CacheEntryLease,
        DeviceTensorCache,
        PinnedBufferPool,
        StagingSlot,
        TransferScheduler,
        TransferTicket,
        _HostTensorReservation,
        _LeaseStatusWorkspace,
    )
    modules = {
        inspect.getmodule(managed_class).__file__
        for managed_class in managed_classes
    }
    parsed = {}
    for path in modules:
        with open(path, encoding="utf-8") as handle:
            parsed[path] = ast.parse(handle.read())

    missing = []
    for managed_class in managed_classes:
        path = inspect.getmodule(managed_class).__file__
        class_node = next(
            node
            for node in parsed[path].body
            if isinstance(node, ast.ClassDef) and node.name == managed_class.__name__
        )
        for method in class_node.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for call in ast.walk(method):
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "_require_admission"
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "self"
                ):
                    continue
                keywords = {keyword.arg for keyword in call.keywords}
                if "allowed_operations" not in keywords:
                    missing.append(
                        "{}.{}:{}".format(
                            managed_class.__name__,
                            method.name,
                            call.lineno,
                        )
                    )

    direct_modules = modules | {
        providers_module.__file__,
        terminal_module.__file__,
    }
    for path in direct_modules:
        tree = parsed.get(path)
        if tree is None:
            with open(path, encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
        for call in ast.walk(tree):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_require_managed_resource_admission"
            ):
                continue
            if "allowed_operations" not in {
                keyword.arg for keyword in call.keywords
            }:
                missing.append("{}:{}".format(path, call.lineno))

    assert missing == []


def test_managed_activation_exception_rolls_back_without_terminal_publication(
    monkeypatch,
):
    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-managed-activation-rollback",
    )
    provider = _install_managed_provider(runtime, request)
    receipt = runtime.preflight_residency(request, plan)
    primary = RuntimeError("managed activation failed")
    created = {}

    def capture_factory(attribute, name):
        original = getattr(provider, attribute)

        def factory(*args, **kwargs):
            resource = original(*args, **kwargs)
            created[name] = resource
            return resource

        monkeypatch.setattr(provider, attribute, factory)

    capture_factory("_provision_status_workspace", "status")
    capture_factory("_cache_factory", "cache")
    capture_factory("_pool_factory", "pool")
    capture_factory("_scheduler_factory", "scheduler")
    original_reserve = store.reserve

    def reserve(*args, **kwargs):
        reservation = original_reserve(*args, **kwargs)
        created["store"] = reservation
        return reservation

    monkeypatch.setattr(store, "reserve", reserve)
    monkeypatch.setattr(
        provider,
        "_activate_lease",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )
    try:
        with pytest.raises(RuntimeError) as caught:
            provider.open_working_set(request, plan, store, receipt)
        assert caught.value is primary
        assert set(created) == {"status", "cache", "pool", "scheduler", "store"}
        assert all(resource._closed for resource in created.values())
        assert store._reservations == {}
        gate = runtime._terminal_gate
        with gate._condition:
            assert gate._phase is _TerminalPhase.HEALTHY
            assert gate._fatal_transition is None
            assert gate._live_epoch is None
            assert not gate._has_active_tokens()
        assert runtime._terminal_secondary_errors == ()
        assert provider._active_lease is None
        assert provider._provisional_resources is None
    finally:
        _repair_test_construction_state(runtime._terminal_gate)
        _close_case(runtime, store)


@pytest.mark.parametrize("candidate", ("missing", "sibling"))
def test_async_detach_proves_admission_before_owner_or_scheduler_mutation(
    candidate,
):
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-detach-before-mutation-{}".format(candidate)
    )
    gate = runtime._terminal_gate
    with lease._lease_admission("acquire"):
        owner = lease.scheduler._register_owner(
            "compute",
        )
        owner.mark_enqueued()

    def snapshot():
        return (
            dict(owner.__dict__),
            dict(lease.scheduler._owners),
            tuple(lease.scheduler._events),
            dict(lease.scheduler._event_owners),
            tuple(lease._resource_record.resources),
            tuple(lease._resource_record.allocations),
        )

    before = snapshot()
    sibling = None
    try:
        if candidate == "missing":
            with pytest.raises(TypeError, match="admission"):
                owner._detach("drained")
        else:
            sibling = gate.admit_lease(lease._epoch, "mark_dirty")
            sibling_validator = lease._exact_admission_validator(sibling)
            with pytest.raises(RuntimeError, match="operation"):
                owner._detach(
                    "drained",
                    _admission_token=sibling,
                    _admission_validator=sibling_validator,
                )
        assert snapshot() == before
    finally:
        if sibling is not None:
            gate.release(sibling)
        if owner.state != "detached":
            admission = owner._async_admission

            def detach(*, _admission_token, _admission_validator):
                owner._detach(
                    "drained",
                    _admission_token=_admission_token,
                    _admission_validator=_admission_validator,
                )

            admission.run("compute_completion", detach)
        _close_case(runtime, store)


@pytest.mark.parametrize("foreign_operation", ("child_close", "acquire"))
def test_pinned_release_requires_exact_checkout_admission_identity(
    foreign_operation,
):
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-pinned-release-owner-{}".format(foreign_operation)
    )
    gate = runtime._terminal_gate
    checkout_token = gate.admit_lease(lease._epoch, "acquire")
    checkout_validator = lease._exact_admission_validator(checkout_token)
    checkout = lease.pool.checkout(
        8,
        _admission_token=checkout_token,
        _admission_validator=checkout_validator,
    )
    slot = checkout.__enter__()

    def snapshot():
        return (
            slot._checked_out,
            slot._nbytes,
            slot._owner,
            lease.pool._checked_out_bytes,
            lease.pool._pending_bytes,
        )

    before = snapshot()

    def foreign_release():
        with lease._lease_admission(foreign_operation) as token:
            lease.pool._release(
                slot,
                _checkout_capability=checkout._checkout_capability,
                _admission_token=token,
                _admission_validator=lease._exact_admission_validator(token),
            )

    worker, _, errors, done = _start(foreign_release)
    try:
        _join(worker, done)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert snapshot() == before
    finally:
        if slot._checked_out:
            checkout.__exit__(None, None, None)
        gate.release(checkout_token)
        _close_case(runtime, store)


@pytest.mark.parametrize("family", ("compute_completion", "h2d_completion"))
def test_semantic_async_cannot_directly_close_unrelated_cache_lease(family):
    runtime, _, _, store, provider, lease = _open_managed_active_case(
        store_id="terminal-cache-direct-close-{}".format(family)
    )
    identity = lease.cache_identities[0]
    admission = None
    cache_lease = None
    try:
        with lease._lease_admission("acquire") as parent:
            validator = lease._exact_admission_validator(parent)
            cache_lease = provider._cache.acquire(
                identity,
                _admission_token=parent,
                _admission_validator=validator,
            )
            admission = lease._spawn_async_admission(object(), family)
        entry = cache_lease._entry
        before = (cache_lease._entry, cache_lease._closed, entry.refcount)

        def close(*, _admission_token, _admission_validator):
            cache_lease.close(
                _admission_token=_admission_token,
                _admission_validator=_admission_validator,
            )

        with pytest.raises(RuntimeError, match="operation"):
            admission.run(family, close)
        assert (cache_lease._entry, cache_lease._closed, entry.refcount) == before
    finally:
        if admission is not None:
            admission.cancel()
        if cache_lease is not None and not cache_lease._closed:
            with lease._lease_admission("acquire") as token:
                cache_lease.close(
                    _admission_token=token,
                    _admission_validator=lease._exact_admission_validator(token),
                )
        _close_case(runtime, store)


@pytest.mark.parametrize(
    "operation",
    ("acquire", "child_close", "load", "operator_call", "prefetch"),
)
def test_direct_cache_lease_close_accepts_exact_lease_operation(operation):
    runtime, _, _, store, provider, lease = _open_managed_active_case(
        store_id="terminal-cache-direct-close-positive-{}".format(operation)
    )
    identity = lease.cache_identities[0]
    cache_lease = None
    try:
        with lease._lease_admission("acquire") as token:
            cache_lease = provider._cache.acquire(
                identity,
                _admission_token=token,
                _admission_validator=lease._exact_admission_validator(token),
            )
        entry = cache_lease._entry
        with lease._lease_admission(operation) as token:
            cache_lease.close(
                _admission_token=token,
                _admission_validator=lease._exact_admission_validator(token),
            )
        assert cache_lease._closed is True
        assert cache_lease._entry is None
        assert entry.refcount == 0
    finally:
        _close_case(runtime, store)


def test_runtime_resource_state_reaps_scheduler_on_healthy_open_lease():
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-resource-state-scheduler-reap"
    )
    try:
        state = runtime.resource_state()
        assert state["active_leases"] == 1
        assert state["cache_bytes"] >= 0
        assert state["pinned_bytes"] == lease.pool._capacity_bytes
        assert state["stream_count"] == 0
        assert state["event_count"] == 0
    finally:
        _close_case(runtime, store)


def test_profiling_managed_operation_flows_authorize_peak_reads(monkeypatch):
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log

    runtime = _runtime()
    request, plan, store, _, _ = _active_case(
        runtime,
        store_id="terminal-profiled-managed-contracts",
    )
    provider = _install_managed_provider(runtime, request)
    receipt = runtime.preflight_residency(request, plan)
    observed = []
    lease = None
    init_log(PROFILING)
    monkeypatch.setattr(profiling, "record", lambda *_args, **_kwargs: None)
    try:
        lease = provider.open_working_set(request, plan, store, receipt).__enter__()
        original_observe = lease._observe_peaks

        def observe(*args, **kwargs):
            observed.append(_current_token(runtime).operation)
            return original_observe(*args, **kwargs)

        monkeypatch.setattr(lease, "_observe_peaks", observe)
        with lease._operator_call(np.ones(4, dtype=np.float64)):
            with lease.acquire(_block_request(lease)):
                pass
        with lease.acquire(_block_request(lease)):
            pass
        lease.mark_dirty("output", np.arange(4.0))
        lease.close()
        assert {"operator_call", "acquire", "child_close", "mark_dirty"}.issubset(
            observed
        )
        assert "observe_peaks" in observed
        assert lease._poisoned_error is None
        assert runtime._terminal_gate._fatal_transition is None
    finally:
        init_log(DEBUG)
        _close_case(runtime, store)


@pytest.mark.parametrize(
    "operation",
    ("acquire", "load", "operator_call", "prefetch"),
)
def test_pending_h2d_staging_wait_accepts_exact_transfer_operation(operation):
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-pending-h2d-wait-{}".format(operation)
    )
    wait_entered = threading.Event()
    release_wait = threading.Event()

    class BlockingEvent:
        @staticmethod
        def query():
            return False

        @staticmethod
        def synchronize():
            wait_entered.set()
            assert release_wait.wait(_TIMEOUT_S)

    owner = None
    with lease._lease_admission("load") as token:
        validator = lease._exact_admission_validator(token)
        checkout = lease.pool.checkout(
            8,
            _admission_token=token,
            _admission_validator=validator,
        )
        with checkout as slot:
            owner = lease.scheduler._register_owner(
                "h2d",
            )
            owner.mark_enqueued()
            owner.add_event(BlockingEvent(), completion=True)
            slot.retain_until(
                owner,
                _checkout_capability=checkout._checkout_capability,
                _admission_token=token,
                _admission_validator=validator,
            )
            owner.arm_completion()
    assert lease.pool._pending_bytes == 8
    before = (
        lease.pool._slot._owner,
        lease.pool._slot._nbytes,
        lease.pool._pending_bytes,
        dict(lease.scheduler._owners),
        owner.state,
    )
    with lease._lease_admission("mark_dirty") as sibling:
        with pytest.raises(RuntimeError, match="operation"):
            lease.pool.wait_for_slot(
                _admission_token=sibling,
                _admission_validator=lease._exact_admission_validator(sibling),
            )
    assert wait_entered.is_set() is False
    assert (
        lease.pool._slot._owner,
        lease.pool._slot._nbytes,
        lease.pool._pending_bytes,
        dict(lease.scheduler._owners),
        owner.state,
    ) == before

    def wait_for_staging():
        with lease._lease_admission(operation) as token:
            lease._wait_for_staging(
                _admission_token=token,
                _admission_validator=lease._exact_admission_validator(token),
            )

    worker, _, errors, done = _start(wait_for_staging)
    try:
        assert wait_entered.wait(_TIMEOUT_S)
        assert done.is_set() is False
        release_wait.set()
        _join(worker, done)
        assert errors == []
        assert owner.state == "detached"
        assert lease.pool._pending_bytes == 0
        assert lease.pool._slot._owner is None
        assert id(owner) not in lease.scheduler._owners
    finally:
        release_wait.set()
        if worker.is_alive():
            _join(worker, done)
        _close_case(runtime, store)


class _BlockingManualEvent(_ManualEvent):
    def __init__(self):
        super().__init__(done=False)
        self.wait_entered = threading.Event()
        self.release_wait = threading.Event()

    def synchronize(self):
        self.wait_entered.set()
        assert self.release_wait.wait(_TIMEOUT_S)
        self.done = True

    def complete(self):
        self.done = True
        self.release_wait.set()


@pytest.mark.parametrize("candidate", ("missing", "sibling"))
def test_scheduler_owner_detach_proves_admission_before_membership_mutation(
    candidate,
):
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-scheduler-detach-{}".format(candidate)
    )
    scheduler = lease.scheduler
    original_detached = scheduler._owner_detached
    callback_admissions = []

    def observe_detached(owner, **kwargs):
        callback_admissions.append(kwargs)
        return original_detached(owner, **kwargs)

    scheduler._owner_detached = observe_detached
    with lease._lease_admission("acquire"):
        owner = scheduler._register_owner(
            "compute",
            defer_counted_admission=True,
        )
        owner.mark_enqueued()
        event = scheduler._new_event(owner, completion=True)

    def snapshot():
        return (
            dict(scheduler._owners),
            tuple(scheduler._events),
            dict(scheduler._event_owners),
            scheduler.last_compute_event,
            owner.state,
            tuple(owner._events),
            owner._completion_event,
        )

    before = snapshot()
    try:
        if candidate == "missing":
            with pytest.raises(TypeError, match="admission"):
                original_detached(owner)
        else:
            with lease._lease_admission("mark_dirty"):
                with pytest.raises(RuntimeError, match="operation"):
                    original_detached(owner)
        assert snapshot() == before

        with lease._lease_admission("resource_state") as token:
            validator = lease._exact_admission_validator(token)
            assert owner._detach(
                "drained",
                _admission_token=token,
                _admission_validator=validator,
            )
        assert len(callback_admissions) == 1
        assert callback_admissions[0]["_admission_token"] is token
        assert callback_admissions[0]["_admission_validator"] is validator
        assert callback_admissions[0]["_detach_capability"] is not None
        assert id(owner) not in scheduler._owners
        assert event not in scheduler._events
        assert id(event) not in scheduler._event_owners
    finally:
        if owner.state != "detached":
            scheduler._owners.clear()
            scheduler._owners.update(before[0])
            scheduler._events[:] = before[1]
            scheduler._event_owners.clear()
            scheduler._event_owners.update(before[2])
            with lease._lease_admission("resource_state") as token:
                owner._detach(
                    "drained",
                    _admission_token=token,
                    _admission_validator=lease._exact_admission_validator(token),
                )
        _close_case(runtime, store)


@pytest.mark.parametrize(
    "operation",
    ("acquire", "load", "operator_call", "prefetch"),
)
def test_load_error_cleanup_waits_under_exact_creating_operation(
    monkeypatch,
    operation,
):
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-load-cleanup-{}".format(operation)
    )
    event = _BlockingManualEvent()
    lease.scheduler._event_factory = lambda: event
    primary = RuntimeError("{} readiness publication failed".format(operation))
    captured = {}
    cleanup_admitted = threading.Event()
    original_stage = lease.scheduler.stage_h2d
    original_install = CacheEntryLease.install_readiness
    original_owner_wait = AsyncResourceOwner.wait

    def capture_stage(*args, **kwargs):
        ticket = original_stage(*args, **kwargs)
        captured["ticket"] = ticket
        captured["owner"] = ticket._owner
        return ticket

    def fail_after_install(cache_lease, ticket, *args, **kwargs):
        original_install(cache_lease, ticket, *args, **kwargs)
        raise primary

    def observe_owner_wait(owner, *args, **kwargs):
        if owner is captured.get("owner"):
            cleanup_admitted.set()
        return original_owner_wait(owner, *args, **kwargs)

    monkeypatch.setattr(lease.scheduler, "stage_h2d", capture_stage)
    monkeypatch.setattr(CacheEntryLease, "install_readiness", fail_after_install)
    monkeypatch.setattr(AsyncResourceOwner, "wait", observe_owner_wait)
    identity = lease.cache_identities[0]

    def load():
        with lease._lease_admission(operation) as token:
            return lease._load_identity_admitted(
                identity,
                prefetch=operation == "prefetch",
                _admission_token=token,
                _admission_validator=lease._exact_admission_validator(token),
            )

    worker, _, errors, done = _start(load)
    try:
        assert cleanup_admitted.wait(_TIMEOUT_S)
        assert event.wait_entered.wait(_TIMEOUT_S)
        owner = captured["owner"]
        ticket = captured["ticket"]
        before = (
            owner.state,
            dict(lease.scheduler._owners),
            lease.pool._slot._owner,
            lease.pool._pending_bytes,
            owner._async_done.is_set(),
        )
        with lease._lease_admission("mark_dirty") as sibling:
            with pytest.raises(RuntimeError, match="operation"):
                ticket.wait(
                    _admission_token=sibling,
                    _admission_validator=lease._exact_admission_validator(sibling),
                )
        assert (
            owner.state,
            dict(lease.scheduler._owners),
            lease.pool._slot._owner,
            lease.pool._pending_bytes,
            owner._async_done.is_set(),
        ) == before

        event.complete()
        _join(worker, done)
        assert errors == [primary]
        assert owner.state == "detached"
        assert id(owner) not in lease.scheduler._owners
        assert lease.pool._slot._owner is None
        assert lease.pool._pending_bytes == 0
        assert lease._poisoned_error is primary
    finally:
        event.complete()
        if worker.is_alive():
            _join(worker, done)
        _close_case(runtime, store)


def test_pending_prefetch_nonblocking_close_progresses_without_poison():
    runtime, _, _, store, provider, lease = _open_managed_active_case(
        store_id="terminal-close-progress-prefetch"
    )
    event = _BlockingManualEvent()
    lease.scheduler._event_factory = lambda: event
    identity = lease.cache_identities[0]
    prefetch, _, prefetch_errors, prefetch_done = _start(
        lambda: lease._load_identity(identity, prefetch=True)
    )
    progress = None
    progress_done = threading.Event()
    progress_errors = []
    try:
        _join(prefetch, prefetch_done)
        assert prefetch_errors == []
        assert event.done is False
        assert len(lease._prefetch_tickets) == 1
        ticket = lease._prefetch_tickets[0]
        owner = ticket._owner
        pending_bytes = lease.pool._pending_bytes
        assert pending_bytes > 0
        assert lease.pool._slot._owner is owner
        assert lease.scheduler._owners[id(owner)] is owner

        progress, _, progress_errors, progress_done = _start(
            lambda: lease.close(wait=False)
        )
        _join(progress, progress_done)
        assert progress_errors == []
        assert lease._poisoned_error is None
        assert owner.state == "enqueued"
        assert lease.pool._pending_bytes == pending_bytes
        assert lease.pool._slot._owner is owner
        assert lease.scheduler._owners[id(owner)] is owner
        assert provider._cache._entries[identity].state == "loading"

        event.complete()
        lease.close(wait=False)
        assert owner._async_done.wait(_TIMEOUT_S)
        owner._async_worker.join(_TIMEOUT_S)
        assert not owner._async_worker.is_alive()
        assert lease._poisoned_error is None
        assert owner.state == "detached"
        assert id(owner) not in lease.scheduler._owners
        assert lease.pool._pending_bytes == 0
        assert lease.pool._slot._owner is None
        lease.close(wait=False)
        assert provider._cache._entries[identity].state == "ready"
        assert runtime._terminal_gate._fatal_transition is None
    finally:
        event.complete()
        if prefetch.is_alive():
            _join(prefetch, prefetch_done)
        if progress is not None and progress.is_alive():
            _join(progress, progress_done)
        _close_case(runtime, store)


def test_dirty_writeback_nonblocking_close_progresses_without_poison():
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-close-progress-writeback"
    )
    lease.mark_dirty("output", np.arange(4.0))
    completion = _BlockingManualEvent()
    events = iter((_ManualEvent(done=True), completion))
    lease.scheduler._event_factory = lambda: next(events)
    dirty = lease._dirty
    reservation = lease._store_reservation
    progress, _, progress_errors, progress_done = _start(
        lambda: lease.close(wait=False)
    )
    try:
        _join(progress, progress_done)
        assert progress_errors == []
        assert lease._poisoned_error is None
        ticket = lease._writeback_ticket
        assert ticket is not None
        owner = ticket._owner
        pending_bytes = lease.pool._pending_bytes
        assert pending_bytes == dirty[1].nbytes
        assert lease._dirty == dirty
        assert reservation._committed is False
        assert completion.done is False
        assert lease.pool._slot._owner is owner
        assert lease.scheduler._owners[id(owner)] is owner

        before = (
            lease.pool._slot._owner,
            lease.pool._slot._nbytes,
            lease.pool._pending_bytes,
            dict(lease.scheduler._owners),
            owner.state,
        )
        with lease._lease_admission("mark_dirty") as sibling:
            with pytest.raises(RuntimeError, match="operation"):
                lease.pool.wait_for_slot(
                    _admission_token=sibling,
                    _admission_validator=lease._exact_admission_validator(sibling),
                )
        assert (
            lease.pool._slot._owner,
            lease.pool._slot._nbytes,
            lease.pool._pending_bytes,
            dict(lease.scheduler._owners),
            owner.state,
        ) == before

        completion.complete()
        lease.close(wait=False)
        assert owner._async_done.wait(_TIMEOUT_S)
        owner._async_worker.join(_TIMEOUT_S)
        assert not owner._async_worker.is_alive()
        assert lease._poisoned_error is None
        assert reservation._committed is True
        assert owner.state == "detached"
        assert id(owner) not in lease.scheduler._owners
        assert lease.pool._pending_bytes == 0
        assert lease.pool._slot._owner is None
        assert runtime._terminal_gate._fatal_transition is None
    finally:
        completion.complete()
        if progress.is_alive():
            _join(progress, progress_done)
        _close_case(runtime, store)


def test_mixed_prefetch_dirty_nonblocking_progress_never_waits_for_staging(
    monkeypatch,
):
    runtime, _, _, store, provider, lease = _open_managed_active_case(
        store_id="terminal-close-progress-mixed-prefetch-dirty"
    )
    prefetch_event = _BlockingManualEvent()
    writeback_event = _BlockingManualEvent()
    events = iter(
        (prefetch_event, _ManualEvent(done=True), writeback_event)
    )
    lease.scheduler._event_factory = lambda: next(events)
    identity = lease.cache_identities[0]
    prefetch, _, prefetch_errors, prefetch_done = _start(
        lambda: lease._load_identity(identity, prefetch=True)
    )
    progress = None
    progress_returned = threading.Event()
    first_boundary = threading.Event()
    staging_wait_entered = threading.Event()
    release_staging_wait = threading.Event()
    original_wait = lease.pool.wait_for_slot

    def block_staging_wait(*args, **kwargs):
        staging_wait_entered.set()
        first_boundary.set()
        assert release_staging_wait.wait(_TIMEOUT_S)
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(lease.pool, "wait_for_slot", block_staging_wait)
    try:
        _join(prefetch, prefetch_done)
        assert prefetch_errors == []
        prefetch_ticket = lease._prefetch_tickets[0]
        prefetch_owner = prefetch_ticket._owner
        pending_bytes = lease.pool._pending_bytes
        lease.mark_dirty("output", np.arange(4.0))

        def progress_once():
            lease.close(wait=False)
            progress_returned.set()
            first_boundary.set()

        progress, _, progress_errors, progress_done = _start(progress_once)
        assert first_boundary.wait(_TIMEOUT_S)
        assert progress_returned.is_set()
        assert staging_wait_entered.is_set() is False
        _join(progress, progress_done)
        assert progress_errors == []
        assert lease._poisoned_error is None
        assert lease._writeback_ticket is None
        assert lease.pool._pending_bytes == pending_bytes
        assert lease.pool._slot._owner is prefetch_owner
        assert provider._cache._entries[identity].state == "loading"

        prefetch_event.complete()
        lease.close(wait=False)
        assert prefetch_owner._async_done.wait(_TIMEOUT_S)
        prefetch_owner._async_worker.join(_TIMEOUT_S)
        assert not prefetch_owner._async_worker.is_alive()
        lease.close(wait=False)
        writeback_ticket = lease._writeback_ticket
        assert writeback_ticket is not None
        writeback_owner = writeback_ticket._owner
        assert lease.pool._slot._owner is writeback_owner

        writeback_event.complete()
        lease.close(wait=False)
        assert writeback_owner._async_done.wait(_TIMEOUT_S)
        writeback_owner._async_worker.join(_TIMEOUT_S)
        assert not writeback_owner._async_worker.is_alive()
        lease.close(wait=False)
        assert lease._store_reservation._committed is True
        assert lease.pool._pending_bytes == 0
        assert lease.pool._slot._owner is None
        assert id(prefetch_owner) not in lease.scheduler._owners
        assert id(writeback_owner) not in lease.scheduler._owners
        assert runtime._terminal_gate._fatal_transition is None
    finally:
        release_staging_wait.set()
        prefetch_event.complete()
        writeback_event.complete()
        if prefetch.is_alive():
            _join(prefetch, prefetch_done)
        if progress is not None and progress.is_alive():
            _join(progress, progress_done)
        _close_case(runtime, store)


def test_nonblocking_progress_does_not_join_counted_completion(monkeypatch):
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-close-progress-no-worker-join"
    )
    event = _BlockingManualEvent()
    lease.scheduler._event_factory = lambda: event
    identity = lease.cache_identities[0]
    prefetch, _, prefetch_errors, prefetch_done = _start(
        lambda: lease._load_identity(identity, prefetch=True)
    )
    progress = None
    release_completion = threading.Event()
    completion_entered = threading.Event()
    progress_returned = threading.Event()
    try:
        _join(prefetch, prefetch_done)
        assert prefetch_errors == []
        owner = lease._prefetch_tickets[0]._owner
        original_completion = owner._run_completion

        def block_completion(*args, **kwargs):
            completion_entered.set()
            assert release_completion.wait(_TIMEOUT_S)
            return original_completion(*args, **kwargs)

        monkeypatch.setattr(owner, "_run_completion", block_completion)
        event.complete()

        def progress_once():
            lease.close(wait=False)
            progress_returned.set()

        progress, _, progress_errors, progress_done = _start(progress_once)
        assert completion_entered.wait(_TIMEOUT_S)
        assert progress_returned.wait(_TIMEOUT_S)
        _join(progress, progress_done)
        assert progress_errors == []
        assert owner._async_done.is_set() is False
        assert lease._poisoned_error is None
    finally:
        release_completion.set()
        event.complete()
        if prefetch.is_alive():
            _join(prefetch, prefetch_done)
        if progress is not None and progress.is_alive():
            _join(progress, progress_done)
        _close_case(runtime, store)


def test_concurrent_nonblocking_progress_has_one_writeback_publisher(
    monkeypatch,
):
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-close-progress-single-publisher"
    )
    lease.mark_dirty("output", np.arange(4.0))
    completion = _BlockingManualEvent()
    events = iter((_ManualEvent(done=True), completion))
    lease.scheduler._event_factory = lambda: next(events)
    publication_entered = threading.Event()
    release_publication = threading.Event()
    original_writeback = lease.scheduler.writeback_d2h
    published = []

    def delay_publication(*args, **kwargs):
        ticket = original_writeback(*args, **kwargs)
        published.append(ticket)
        publication_entered.set()
        assert release_publication.wait(_TIMEOUT_S)
        return ticket

    monkeypatch.setattr(lease.scheduler, "writeback_d2h", delay_publication)
    first, _, first_errors, first_done = _start(
        lambda: lease.close(wait=False)
    )
    second = None
    try:
        assert publication_entered.wait(_TIMEOUT_S)
        assert lease._writeback_ticket is None
        second, _, second_errors, second_done = _start(
            lambda: lease.close(wait=False)
        )
        _join(second, second_done)
        assert second_errors == []
        assert lease._poisoned_error is None
        assert len(published) == 1
        assert lease._writeback_ticket is None

        release_publication.set()
        _join(first, first_done)
        assert first_errors == []
        assert lease._writeback_ticket is published[0]
        assert lease.pool._slot._owner is published[0]._owner
    finally:
        release_publication.set()
        completion.complete()
        if first.is_alive():
            _join(first, first_done)
        if second is not None and second.is_alive():
            _join(second, second_done)
        _close_case(runtime, store)


def test_nonblocking_progress_yields_to_elected_close_owner(monkeypatch):
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-close-progress-elected-owner"
    )
    elected_entered = threading.Event()
    release_elected = threading.Event()
    original_start = lease.scheduler._start_counted_completions

    def block_elected_start(*args, **kwargs):
        elected_entered.set()
        assert release_elected.wait(_TIMEOUT_S)
        return original_start(*args, **kwargs)

    monkeypatch.setattr(
        lease.scheduler,
        "_start_counted_completions",
        block_elected_start,
    )
    elected, _, elected_errors, elected_done = _start(lease.close)
    progress = None
    try:
        assert elected_entered.wait(_TIMEOUT_S)
        progress, _, progress_errors, progress_done = _start(
            lambda: lease.close(wait=False)
        )
        _join(progress, progress_done)
        assert progress_errors == []
        assert lease._poisoned_error is None
        assert elected_done.is_set() is False
        assert runtime._terminal_gate._fatal_transition is None

        release_elected.set()
        _join(elected, elected_done)
        assert elected_errors == []
        assert lease._closed is True
    finally:
        release_elected.set()
        if elected.is_alive():
            _join(elected, elected_done)
        if progress is not None and progress.is_alive():
            _join(progress, progress_done)
        _close_case(runtime, store)


def test_nonblocking_progress_yields_after_close_election_before_owner(
    monkeypatch,
):
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-close-progress-election-gap"
    )
    gate = runtime._terminal_gate
    election_installed = threading.Event()
    release_election = threading.Event()
    original_begin = gate.begin_lease_close

    def begin_then_pause(*args, **kwargs):
        result = original_begin(*args, **kwargs)
        election_installed.set()
        assert release_election.wait(_TIMEOUT_S)
        return result

    monkeypatch.setattr(gate, "begin_lease_close", begin_then_pause)
    elected, _, elected_errors, elected_done = _start(lease.close)
    progress = None
    try:
        assert election_installed.wait(_TIMEOUT_S)
        assert lease._closing is False
        progress, _, progress_errors, progress_done = _start(
            lambda: lease.close(wait=False)
        )
        _join(progress, progress_done)
        assert progress_errors == []
        assert lease._poisoned_error is None
        assert elected_done.is_set() is False

        release_election.set()
        _join(elected, elected_done)
        assert elected_errors == []
        assert lease._closed is True
    finally:
        release_election.set()
        if elected.is_alive():
            _join(elected, elected_done)
        if progress is not None and progress.is_alive():
            _join(progress, progress_done)
        _close_case(runtime, store)


def test_scheduler_detach_requires_exact_owner_capability():
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-scheduler-detach-owner-capability"
    )
    scheduler = lease.scheduler
    with lease._lease_admission("acquire"):
        owner = scheduler._register_owner(
            "compute",
            defer_counted_admission=True,
        )
        owner.mark_enqueued()
        event = scheduler._new_event(owner, completion=True)

    def snapshot():
        return (
            dict(scheduler._owners),
            tuple(scheduler._events),
            dict(scheduler._event_owners),
            owner.state,
            tuple(owner._events),
        )

    before = snapshot()
    try:
        with lease._lease_admission("resource_state") as token:
            validator = lease._exact_admission_validator(token)
            for capability in (None, object()):
                with pytest.raises(RuntimeError, match="capability"):
                    scheduler._owner_detached(
                        owner,
                        _detach_capability=capability,
                        _admission_token=token,
                        _admission_validator=validator,
                    )
                assert snapshot() == before

        with lease._lease_admission("resource_state") as token:
            assert owner._detach(
                "drained",
                _admission_token=token,
                _admission_validator=lease._exact_admission_validator(token),
            )
        assert id(owner) not in scheduler._owners
        assert event not in scheduler._events
        assert id(event) not in scheduler._event_owners
    finally:
        if owner.state != "detached":
            scheduler._owners.clear()
            scheduler._owners.update(before[0])
            scheduler._events[:] = before[1]
            scheduler._event_owners.clear()
            scheduler._event_owners.update(before[2])
            with lease._lease_admission("resource_state") as token:
                owner._detach(
                    "drained",
                    _admission_token=token,
                    _admission_validator=lease._exact_admission_validator(token),
                )
        _close_case(runtime, store)


@pytest.mark.parametrize("action", ("retain", "release", "view"))
def test_staging_slot_requires_exact_checkout_capability(action):
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-staging-checkout-capability-{}".format(action)
    )
    gate = runtime._terminal_gate
    checkout_token = gate.admit_lease(lease._epoch, "acquire")
    checkout_validator = lease._exact_admission_validator(checkout_token)
    checkout = lease.pool.checkout(
        8,
        _admission_token=checkout_token,
        _admission_validator=checkout_validator,
    )
    slot = checkout.__enter__()
    owner = AsyncResourceOwner("staging")

    def snapshot():
        return (
            slot._checked_out,
            slot._nbytes,
            slot._owner,
            slot._checkout_admission,
            lease.pool._checked_out_bytes,
            lease.pool._pending_bytes,
            owner._resources,
            owner._allocations,
            tuple(owner._release_callbacks),
        )

    before = snapshot()
    foreign = None
    try:
        if action == "retain":
            foreign_errors = []

            def invoke_foreign():
                with lease._lease_admission("acquire") as token:
                    try:
                        slot.retain_until(
                            owner,
                            _checkout_capability=(
                                checkout._checkout_capability
                            ),
                            _admission_token=token,
                            _admission_validator=(
                                lease._exact_admission_validator(token)
                            ),
                        )
                    except BaseException as error:
                        foreign_errors.append(error)

            foreign, _, thread_errors, foreign_done = _start(invoke_foreign)
            _join(foreign, foreign_done)
            assert thread_errors == []
            assert len(foreign_errors) == 1
            assert isinstance(foreign_errors[0], RuntimeError)
            assert "capability" in str(foreign_errors[0])
            invoke = None
        elif action == "release":
            invoke = lambda: lease.pool._release(
                slot,
                _admission_token=checkout_token,
                _admission_validator=checkout_validator,
            )
        else:
            invoke = lambda: slot.view(
                (1,),
                np.dtype(np.float64),
                _admission_token=checkout_token,
                _admission_validator=checkout_validator,
            )
        if invoke is not None:
            with pytest.raises(RuntimeError, match="capability"):
                invoke()
        assert snapshot() == before
    finally:
        if foreign is not None and foreign.is_alive():
            _join(foreign, foreign_done)
        if slot._owner is owner:
            slot._owner = None
            owner._resources = ()
            owner._allocations = ()
            owner._release_callbacks = []
        if slot._checked_out:
            try:
                checkout.__exit__(None, None, None)
            except BaseException:
                slot._checkout_admission = None
                slot._checked_out = False
                lease.pool._checked_out_bytes = 0
                lease.pool._finalize_slot()
        gate.release(checkout_token)
        _close_case(runtime, store)


def test_staging_checkout_capability_is_invalid_after_generation_advance():
    runtime, _, _, store, _, lease = _open_managed_active_case(
        store_id="terminal-staging-checkout-stale-generation"
    )
    with lease._lease_admission("acquire") as token:
        validator = lease._exact_admission_validator(token)
        first = lease.pool.checkout(
            8,
            _admission_token=token,
            _admission_validator=validator,
        )
        first_slot = first.__enter__()
        stale = first._checkout_capability
        first.__exit__(None, None, None)
        second = lease.pool.checkout(
            8,
            _admission_token=token,
            _admission_validator=validator,
        )
        slot = second.__enter__()
        before = (
            slot._checked_out,
            slot._owner,
            slot._checkout_capability,
            lease.pool._checked_out_bytes,
        )
        try:
            with pytest.raises(RuntimeError, match="capability"):
                slot.view(
                    (1,),
                    np.dtype(np.float64),
                    _checkout_capability=stale,
                    _admission_token=token,
                    _admission_validator=validator,
                )
            assert (
                slot._checked_out,
                slot._owner,
                slot._checkout_capability,
                lease.pool._checked_out_bytes,
            ) == before
            assert first_slot is slot
            view = slot.view(
                (1,),
                np.dtype(np.float64),
                _checkout_capability=second._checkout_capability,
                _admission_token=token,
                _admission_validator=validator,
            )
            assert view.nbytes == 8
        finally:
            second.__exit__(None, None, None)
    _close_case(runtime, store)
