import gc
import threading
import weakref

import pytest

from renormalizer.backend._distributed.terminal import (
    _TerminalLifecycleGate,
    _TerminalPhase,
)


_TIMEOUT_S = 3.0


class _AsyncOwner:
    pass


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


def _call_in_thread(call):
    thread, results, errors, done = _start(call)
    _join(thread, done)
    return results, errors


def _open_lease(gate, operation="construct"):
    epoch, construction = gate.begin_lease(operation)
    gate.activate_lease(epoch, construction)
    gate.release(construction)
    return epoch


def _close_lease(gate, epoch, result=None):
    transition, owns_close = gate.begin_lease_close(epoch)
    assert owns_close is True
    gate.wait_for_lease_admissions(transition, _TIMEOUT_S)
    return gate.commit_lease_close(transition, lambda: result)


def test_sequential_lease_epochs_leave_runtime_healthy():
    gate = _TerminalLifecycleGate()

    first_epoch = _open_lease(gate, "construct_first")
    first_result = object()
    assert _close_lease(gate, first_epoch, first_result) is first_result
    assert gate.phase is _TerminalPhase.HEALTHY

    second_epoch = _open_lease(gate, "construct_second")
    assert second_epoch > first_epoch
    second_result = object()
    assert _close_lease(gate, second_epoch, second_result) is second_result
    assert gate.phase is _TerminalPhase.HEALTHY


def test_runtime_admission_is_valid_between_healthy_leases():
    gate = _TerminalLifecycleGate()
    first_epoch = _open_lease(gate)
    _close_lease(gate, first_epoch)

    token = gate.admit_runtime("inspect_cache_between_leases")

    assert token.scope == "runtime"
    assert token.epoch is None
    gate.release(token)
    second_epoch = _open_lease(gate)
    assert second_epoch > first_epoch
    _close_lease(gate, second_epoch)


def test_runtime_setup_admission_requires_no_live_lease():
    exclusive_gate = _TerminalLifecycleGate()
    active_setup = exclusive_gate.admit_runtime_setup("active_setup")
    _, lease_errors = _call_in_thread(
        lambda: exclusive_gate.begin_lease("lease_during_setup")
    )
    assert len(lease_errors) == 1
    assert isinstance(lease_errors[0], RuntimeError)
    exclusive_gate.release(active_setup)

    gate = _TerminalLifecycleGate()
    setup = gate.admit_runtime_setup("initial_setup")
    gate.release(setup)

    epoch, construction = gate.begin_lease("construct")
    _, constructing_errors = _call_in_thread(
        lambda: gate.admit_runtime_setup("setup_while_constructing")
    )
    assert len(constructing_errors) == 1
    assert isinstance(constructing_errors[0], RuntimeError)

    gate.activate_lease(epoch, construction)
    gate.release(construction)
    _, open_errors = _call_in_thread(
        lambda: gate.admit_runtime_setup("setup_while_open")
    )
    assert len(open_errors) == 1
    assert isinstance(open_errors[0], RuntimeError)

    transition, owns_close = gate.begin_lease_close(epoch)
    assert owns_close is True
    _, closing_errors = _call_in_thread(
        lambda: gate.admit_runtime_setup("setup_while_closing")
    )
    assert len(closing_errors) == 1
    assert isinstance(closing_errors[0], RuntimeError)
    gate.wait_for_lease_admissions(transition, _TIMEOUT_S)
    gate.commit_lease_close(transition, lambda: None)

    setup = gate.admit_runtime_setup("setup_after_close")
    gate.release(setup)


def test_provisional_epoch_precedes_first_resource_operation():
    gate = _TerminalLifecycleGate()

    epoch, construction = gate.begin_lease("allocate_first_resource")

    assert epoch > 0
    assert construction.scope == "construction"
    assert construction.epoch == epoch
    assert construction.operation == "allocate_first_resource"
    gate.activate_lease(epoch, construction)
    gate.release(construction)
    _close_lease(gate, epoch)


def test_construction_activation_and_rollback_share_provisional_epoch():
    gate = _TerminalLifecycleGate()
    activated_epoch, construction = gate.begin_lease("construct_for_activation")

    gate.activate_lease(activated_epoch, construction)
    assert construction.epoch == activated_epoch
    gate.release(construction)
    _close_lease(gate, activated_epoch)

    rollback_epoch, rollback = gate.begin_lease("construct_for_rollback")
    assert rollback.epoch == rollback_epoch
    gate.release(rollback)
    transition, owns_close = gate.begin_lease_close(rollback_epoch)
    assert owns_close is True
    assert transition.epoch == rollback_epoch
    gate.wait_for_lease_admissions(transition, _TIMEOUT_S)
    gate.commit_lease_close(transition, lambda: None)


def test_nested_wrong_thread_stale_and_double_release_are_rejected():
    gate = _TerminalLifecycleGate()
    runtime_token = gate.admit_runtime("outer")

    with pytest.raises(RuntimeError, match="nested"):
        gate.admit_runtime("nested")

    _, wrong_thread_errors = _call_in_thread(lambda: gate.release(runtime_token))
    assert len(wrong_thread_errors) == 1
    assert isinstance(wrong_thread_errors[0], RuntimeError)
    gate.release(runtime_token)
    with pytest.raises(RuntimeError, match="released"):
        gate.release(runtime_token)

    first_epoch = _open_lease(gate)
    _close_lease(gate, first_epoch)
    second_epoch = _open_lease(gate)
    with pytest.raises(RuntimeError, match="stale"):
        gate.admit_lease(first_epoch, "stale_epoch")
    _close_lease(gate, second_epoch)


def test_local_fatal_atomically_converts_discovering_token():
    gate = _TerminalLifecycleGate()
    token = gate.admit_runtime("discover_local_fatal")
    primary = RuntimeError("local fatal")

    transition = gate.begin_fatal(primary, token)

    assert transition.primary is primary
    assert gate.phase is _TerminalPhase.FATAL_PENDING
    with pytest.raises(RuntimeError, match="converted"):
        gate.release(token)
    gate.wait_for_admissions(transition, _TIMEOUT_S)
    snapshot = object()
    gate.publish_fatal(transition, snapshot)
    assert gate.wait_for_published(_TIMEOUT_S) is snapshot


def test_remote_fatal_waits_for_existing_admission():
    gate = _TerminalLifecycleGate()
    admitted = threading.Event()
    release_operation = threading.Event()

    def operation():
        token = gate.admit_runtime("remote_operation")
        admitted.set()
        assert release_operation.wait(_TIMEOUT_S)
        gate.release(token)

    operation_thread, _, operation_errors, operation_done = _start(operation)
    assert admitted.wait(_TIMEOUT_S)
    transition = gate.begin_fatal(RuntimeError("remote fatal"))
    wait_thread, _, wait_errors, wait_done = _start(
        lambda: gate.wait_for_admissions(transition, _TIMEOUT_S)
    )

    assert not wait_done.wait(0.05)
    release_operation.set()
    _join(operation_thread, operation_done)
    _join(wait_thread, wait_done)
    assert operation_errors == []
    assert wait_errors == []


def test_pending_wait_returns_only_published_snapshot():
    gate = _TerminalLifecycleGate()
    transition = gate.begin_fatal(RuntimeError("fatal"))
    wait_started = threading.Event()

    def wait_for_snapshot():
        wait_started.set()
        return gate.wait_for_published(_TIMEOUT_S)

    waiter, results, errors, done = _start(wait_for_snapshot)
    assert wait_started.wait(_TIMEOUT_S)
    assert not done.wait(0.05)
    snapshot = object()
    gate.publish_fatal(transition, snapshot)
    _join(waiter, done)
    assert errors == []
    assert results == [snapshot]


def test_begin_lease_close_rejects_new_operator_and_acquire_admissions():
    gate = _TerminalLifecycleGate()
    epoch = _open_lease(gate)
    prior = gate.admit_lease(epoch, "prior_operator")

    transition, owns_close = gate.begin_lease_close(epoch)

    assert owns_close is True
    gate.release(prior)
    for operation in ("operator", "acquire"):
        with pytest.raises(RuntimeError, match="closing"):
            gate.admit_lease(epoch, operation)
    gate.wait_for_lease_admissions(transition, _TIMEOUT_S)
    gate.commit_lease_close(transition, lambda: None)


def test_lease_close_drains_prior_ordinary_and_multiple_async_descendants():
    gate = _TerminalLifecycleGate()
    epoch = _open_lease(gate)
    parent = gate.admit_lease(epoch, "schedule_callbacks")
    release_callbacks = [threading.Event(), threading.Event()]
    claimed_callbacks = [threading.Event(), threading.Event()]
    callback_errors = [[], []]
    descendants = [None, None]
    owners = [_AsyncOwner(), _AsyncOwner()]
    owner_refs = [weakref.ref(owner) for owner in owners]

    def callback(index):
        try:
            claimed = gate.claim_async(descendants[index], "callback_{}".format(index))
            assert claimed.parent_sequence == parent.sequence
            claimed_callbacks[index].set()
            assert release_callbacks[index].wait(_TIMEOUT_S)
            gate.release(claimed)
        except BaseException as error:
            callback_errors[index].append(error)
            claimed_callbacks[index].set()

    callbacks = [
        threading.Thread(target=callback, args=(index,), daemon=True)
        for index in range(2)
    ]
    for index, owner in enumerate(owners):
        descendants[index] = gate.spawn_async(parent, owner)
    del owner
    owners.clear()
    gc.collect()
    assert [owner_ref() for owner_ref in owner_refs] == [None, None]
    gate.release(parent)
    transition, owns_close = gate.begin_lease_close(epoch)
    assert owns_close is True
    for callback_thread in callbacks:
        callback_thread.start()
    for claimed in claimed_callbacks:
        assert claimed.wait(_TIMEOUT_S)
    assert callback_errors == [[], []]

    waiter, _, wait_errors, wait_done = _start(
        lambda: gate.wait_for_lease_admissions(transition, _TIMEOUT_S)
    )
    assert not wait_done.wait(0.05)
    release_callbacks[0].set()
    callbacks[0].join(_TIMEOUT_S)
    assert not callbacks[0].is_alive()
    assert not wait_done.wait(0.05)
    release_callbacks[1].set()
    callbacks[1].join(_TIMEOUT_S)
    assert not callbacks[1].is_alive()
    _join(waiter, wait_done)
    assert callback_errors == [[], []]
    assert wait_errors == []
    gate.commit_lease_close(transition, lambda: None)


def test_only_elected_close_owner_can_admit_close_steps():
    gate = _TerminalLifecycleGate()
    epoch = _open_lease(gate)
    transition, owns_close = gate.begin_lease_close(epoch)
    assert owns_close is True
    gate.wait_for_lease_admissions(transition, _TIMEOUT_S)

    _, errors = _call_in_thread(
        lambda: gate.admit_lease_close(transition, "foreign_close_step")
    )
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    step = gate.admit_lease_close(transition, "owner_close_step")
    assert step.scope == "lease_close"
    assert step.transition_sequence == transition.sequence
    gate.release(step)
    gate.commit_lease_close(transition, lambda: None)


def test_second_close_caller_joins_without_running_steps():
    gate = _TerminalLifecycleGate()
    epoch = _open_lease(gate)
    transition, owns_close = gate.begin_lease_close(epoch)
    assert owns_close is True
    joined_transition = threading.Event()

    def join_close():
        joined, owns_joined_close = gate.begin_lease_close(epoch)
        assert joined == transition
        assert owns_joined_close is False
        joined_transition.set()
        return gate.wait_for_lease_closed(joined, _TIMEOUT_S)

    joiner, results, errors, done = _start(join_close)
    assert joined_transition.wait(_TIMEOUT_S)
    assert not done.wait(0.05)
    result = object()
    gate.wait_for_lease_admissions(transition, _TIMEOUT_S)
    assert gate.commit_lease_close(transition, lambda: result) is result
    _join(joiner, done)
    assert errors == []
    assert results == [result]


def test_lease_close_commit_loses_to_pending_without_running_finalizer():
    gate = _TerminalLifecycleGate()
    epoch = _open_lease(gate)
    close_transition, owns_close = gate.begin_lease_close(epoch)
    assert owns_close is True
    gate.wait_for_lease_admissions(close_transition, _TIMEOUT_S)
    fatal_transition = gate.begin_fatal(RuntimeError("fatal before lease commit"))
    finalized = []

    assert gate._leases[epoch].phase == "fatal_retained"
    with pytest.raises(RuntimeError, match="fatal"):
        gate.commit_lease_close(close_transition, lambda: finalized.append(True))

    assert finalized == []
    snapshot = object()
    gate.publish_fatal(fatal_transition, snapshot)
    assert gate.wait_for_published(_TIMEOUT_S) is snapshot


def test_runtime_close_joins_in_progress_lease_close():
    gate = _TerminalLifecycleGate()
    epoch = _open_lease(gate)
    lease_transition, owns_close = gate.begin_lease_close(epoch)
    assert owns_close is True
    runtime_calling = threading.Event()
    allow_runtime_commit = threading.Event()

    def close_runtime():
        runtime_calling.set()
        transition = gate.begin_runtime_close(None)
        assert allow_runtime_commit.wait(_TIMEOUT_S)
        return gate.commit_runtime_close(transition, lambda: "runtime_closed")

    closer, results, errors, done = _start(close_runtime)
    assert runtime_calling.wait(_TIMEOUT_S)
    assert not done.wait(0.05)
    gate.wait_for_lease_admissions(lease_transition, _TIMEOUT_S)
    gate.commit_lease_close(lease_transition, lambda: "lease_closed")
    allow_runtime_commit.set()
    _join(closer, done)
    assert errors == []
    assert results == ["runtime_closed"]
    assert gate.phase is _TerminalPhase.RUNTIME_CLOSED


def test_runtime_close_rejects_new_admissions_and_commits_query_free():
    gate = _TerminalLifecycleGate()
    request = gate.admit_runtime("begin_runtime_close")
    transition = gate.begin_runtime_close(request)

    with pytest.raises(RuntimeError, match="converted"):
        gate.release(request)
    with pytest.raises(RuntimeError, match="closing"):
        gate.admit_runtime("late_runtime_operation")
    with pytest.raises(RuntimeError, match="closing"):
        gate.admit_runtime_setup("late_setup")
    with pytest.raises(RuntimeError, match="closing"):
        gate.begin_lease("late_lease")

    step = gate.admit_runtime_close(transition, "close_provider")
    assert step.scope == "runtime_close"
    assert step.transition_sequence == transition.sequence
    gate.release(step)
    finalized = []

    def finalizer():
        finalized.append(True)
        return "closed"

    assert gate.commit_runtime_close(transition, finalizer) == "closed"
    assert finalized == [True]
    assert gate.phase is _TerminalPhase.RUNTIME_CLOSED


def test_runtime_close_converts_initiating_token_before_drain():
    gate = _TerminalLifecycleGate()
    request = gate.admit_runtime("begin_runtime_close")

    transition = gate.begin_runtime_close(request)

    assert transition.sequence > request.sequence
    with pytest.raises(RuntimeError, match="converted"):
        gate.release(request)
    assert gate.commit_runtime_close(transition, lambda: None) is None


def test_close_commits_and_join_waits_hold_no_admission():
    gate = _TerminalLifecycleGate()
    epoch = _open_lease(gate)
    lease_transition, owns_close = gate.begin_lease_close(epoch)
    assert owns_close is True
    join_started = threading.Event()

    def join_lease_close():
        joined, owns_joined_close = gate.begin_lease_close(epoch)
        assert joined == lease_transition
        assert owns_joined_close is False
        join_started.set()
        return gate.wait_for_lease_closed(joined, _TIMEOUT_S)

    joiner, join_results, join_errors, join_done = _start(join_lease_close)
    assert join_started.wait(_TIMEOUT_S)
    gate.wait_for_lease_admissions(lease_transition, _TIMEOUT_S)
    assert gate.commit_lease_close(lease_transition, lambda: "lease") == "lease"
    _join(joiner, join_done)
    assert join_errors == []
    assert join_results == ["lease"]

    request = gate.admit_runtime("begin_runtime_close")
    runtime_transition = gate.begin_runtime_close(request)
    assert gate.commit_runtime_close(runtime_transition, lambda: "runtime") == "runtime"


def test_fatal_preempts_runtime_close_only_before_commit():
    preempted = _TerminalLifecycleGate()
    close_transition = preempted.begin_runtime_close(None)
    primary = RuntimeError("fatal before runtime commit")
    fatal_transition = preempted.begin_fatal(primary)
    with pytest.raises(RuntimeError, match="fatal"):
        preempted.admit_runtime_close(close_transition, "late_close_step")
    snapshot = object()
    preempted.publish_fatal(fatal_transition, snapshot)
    finalized = []
    result = preempted.commit_runtime_close(
        fatal_transition, lambda: finalized.append("cleared")
    )
    assert result is snapshot
    assert finalized == ["cleared"]
    assert preempted.phase is _TerminalPhase.RUNTIME_CLOSED

    committed = _TerminalLifecycleGate()
    close_transition = committed.begin_runtime_close(None)
    assert committed.commit_runtime_close(close_transition, lambda: "closed") == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        committed.begin_fatal(RuntimeError("fatal after runtime commit"))


def test_runtime_close_joins_pending_and_published_fatal():
    pending = _TerminalLifecycleGate()
    primary = RuntimeError("pending fatal")
    fatal_transition = pending.begin_fatal(primary)
    close_started = threading.Event()
    cleared = []

    def join_pending():
        transition = pending.begin_runtime_close(None)
        assert transition == fatal_transition
        close_started.set()
        return pending.commit_runtime_close(
            transition, lambda: cleared.append("pending")
        )

    closer, results, errors, done = _start(join_pending)
    assert close_started.wait(_TIMEOUT_S)
    assert not done.wait(0.05)
    pending_snapshot = object()
    pending.publish_fatal(fatal_transition, pending_snapshot)
    _join(closer, done)
    assert errors == []
    assert results == [pending_snapshot]
    assert cleared == ["pending"]

    published = _TerminalLifecycleGate()
    published_transition = published.begin_fatal(RuntimeError("published fatal"))
    published_snapshot = object()
    published.publish_fatal(published_transition, published_snapshot)
    joined = published.begin_runtime_close(None)
    assert joined == published_transition
    assert published.commit_runtime_close(joined, lambda: "cleared") is published_snapshot


def test_first_pending_primary_is_immutable():
    gate = _TerminalLifecycleGate()
    first = RuntimeError("first")
    later = RuntimeError("later")

    first_transition = gate.begin_fatal(first)
    later_transition = gate.begin_fatal(later)

    assert later_transition == first_transition
    assert later_transition.primary is first
    snapshot = object()
    gate.publish_fatal(later_transition, snapshot)
    assert gate.wait_for_published(_TIMEOUT_S) is snapshot
