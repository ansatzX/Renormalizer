import gc
from pathlib import Path
import threading
from types import SimpleNamespace
import weakref

import pytest

from renormalizer.backend._distributed.terminal import (
    _FatalMonitorHandoff,
    _TerminalLifecycleGate,
    _TerminalPhase,
)


_TIMEOUT_S = 3.0


class _AsyncClaimCapability:
    pass


def test_concurrency_tests_use_exact_wait_entry_handoffs():
    source = Path(__file__).read_text(encoding="utf-8")

    assert "wait(" + "0." + "05)" not in source
    assert "allow_runtime_" + "commit" not in source


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


def _observe_condition_waits(gate, count):
    entered = [threading.Event() for _ in range(count)]
    counter_lock = threading.Lock()
    original_wait = gate._condition.wait
    next_index = 0

    def observe_wait(timeout=None):
        nonlocal next_index
        with counter_lock:
            index = next_index
            next_index += 1
        if index < len(entered):
            entered[index].set()
        return original_wait(timeout)

    gate._condition.wait = observe_wait
    return entered


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


def test_rejected_inputs_do_not_mutate_epochs_phases_or_sequences():
    lease_gate = _TerminalLifecycleGate()
    next_epoch = lease_gate._next_epoch
    next_sequence = lease_gate._next_sequence

    with pytest.raises(ValueError, match="non-empty"):
        lease_gate.begin_lease("")

    assert lease_gate.phase is _TerminalPhase.HEALTHY
    assert lease_gate._live_epoch is None
    assert lease_gate._leases == {}
    assert lease_gate._next_epoch == next_epoch
    assert lease_gate._next_sequence == next_sequence
    epoch, construction = lease_gate.begin_lease("valid_construction")
    assert epoch == next_epoch
    assert construction.sequence == next_sequence
    lease_gate.activate_lease(epoch, construction)
    lease_gate.release(construction)
    _close_lease(lease_gate, epoch)

    fatal_gate = _TerminalLifecycleGate()
    fatal_parent = fatal_gate.admit_runtime("fatal_parent")
    fatal_capability = _AsyncClaimCapability()
    fatal_descendant = fatal_gate.spawn_async(fatal_parent, fatal_capability)
    fatal_sequence = fatal_gate._next_sequence

    with pytest.raises(RuntimeError, match="unclaimed"):
        fatal_gate.begin_fatal(RuntimeError("rejected fatal"), fatal_descendant)

    assert fatal_gate.phase is _TerminalPhase.HEALTHY
    assert fatal_gate._fatal_transition is None
    assert fatal_gate._next_sequence == fatal_sequence
    fatal_gate.release(fatal_descendant)
    fatal_gate.release(fatal_parent)

    close_gate = _TerminalLifecycleGate()
    close_parent = close_gate.admit_runtime("close_parent")
    close_capability = _AsyncClaimCapability()
    close_descendant = close_gate.spawn_async(close_parent, close_capability)
    close_sequence = close_gate._next_sequence

    with pytest.raises(RuntimeError, match="unclaimed"):
        close_gate.begin_runtime_close(close_descendant)

    assert close_gate.phase is _TerminalPhase.HEALTHY
    assert close_gate._runtime_close_transition is None
    assert close_gate._next_sequence == close_sequence
    close_gate.release(close_descendant)
    close_gate.release(close_parent)


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
    wait_entries = _observe_condition_waits(gate, 1)
    wait_thread, _, wait_errors, wait_done = _start(
        lambda: gate.wait_for_admissions(transition, _TIMEOUT_S)
    )

    assert wait_entries[0].wait(_TIMEOUT_S)
    release_operation.set()
    _join(operation_thread, operation_done)
    _join(wait_thread, wait_done)
    assert operation_errors == []
    assert wait_errors == []


def test_pending_wait_returns_only_published_snapshot():
    gate = _TerminalLifecycleGate()
    transition = gate.begin_fatal(RuntimeError("fatal"))
    wait_entries = _observe_condition_waits(gate, 1)
    waiter, results, errors, done = _start(
        lambda: gate.wait_for_published(_TIMEOUT_S)
    )
    assert wait_entries[0].wait(_TIMEOUT_S)
    snapshot = object()
    gate.publish_fatal(transition, snapshot)
    _join(waiter, done)
    assert errors == []
    assert results == [snapshot]


@pytest.mark.parametrize(
    "timeout_s", [float("nan"), float("inf"), float("-inf")]
)
def test_non_finite_terminal_wait_deadlines_are_rejected(timeout_s):
    gate = _TerminalLifecycleGate()

    with pytest.raises(ValueError, match="finite"):
        gate.wait_for_published(timeout_s)


def test_token_holding_callers_are_rejected_before_terminal_waits():
    fatal_gate = _TerminalLifecycleGate()
    fatal_token = fatal_gate.admit_runtime("held_during_fatal")
    fatal_transition = fatal_gate.begin_fatal(RuntimeError("fatal"))

    def fail_if_waited(timeout=None):
        raise AssertionError("condition wait entered with a live token")

    fatal_gate._condition.wait = fail_if_waited
    with pytest.raises(RuntimeError, match="holding an admission"):
        fatal_gate.wait_for_published(_TIMEOUT_S)
    with pytest.raises(RuntimeError, match="holding an admission"):
        fatal_gate.commit_runtime_close(fatal_transition, lambda: None)
    with pytest.raises(RuntimeError, match="holding an admission"):
        fatal_gate.publish_fatal(fatal_transition, object())

    join_gate = _TerminalLifecycleGate()
    join_token = join_gate.admit_runtime("held_during_runtime_join")
    owner_waiting = threading.Event()
    owner_can_start = threading.Event()
    original_wait = join_gate._condition.wait
    owner_thread = None

    def observe_owner_wait(timeout=None):
        if threading.current_thread() is owner_thread:
            owner_waiting.set()
            return original_wait(timeout)
        raise AssertionError("runtime join waited with a live token")

    join_gate._condition.wait = observe_owner_wait
    owner_thread, _, owner_errors, owner_done = _start(
        lambda: (
            owner_can_start.wait(_TIMEOUT_S),
            join_gate.begin_runtime_close(None),
        )[1]
    )
    owner_can_start.set()
    assert owner_waiting.wait(_TIMEOUT_S)
    transition = join_gate._runtime_close_transition
    with pytest.raises(RuntimeError, match="holding an admission"):
        join_gate.commit_runtime_close(transition, lambda: None)
    join_gate.release(join_token)
    _join(owner_thread, owner_done)
    assert owner_errors == []


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
    capabilities = [_AsyncClaimCapability(), _AsyncClaimCapability()]

    def callback(index):
        try:
            claimed = gate.claim_async(
                descendants[index],
                capabilities[index],
                "callback_{}".format(index),
            )
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
    for index, capability in enumerate(capabilities):
        descendants[index] = gate.spawn_async(parent, capability)
    gate.release(parent)
    transition, owns_close = gate.begin_lease_close(epoch)
    assert owns_close is True
    for callback_thread in callbacks:
        callback_thread.start()
    for claimed in claimed_callbacks:
        assert claimed.wait(_TIMEOUT_S)
    assert callback_errors == [[], []]

    wait_entries = _observe_condition_waits(gate, 2)
    waiter, _, wait_errors, wait_done = _start(
        lambda: gate.wait_for_lease_admissions(transition, _TIMEOUT_S)
    )
    assert wait_entries[0].wait(_TIMEOUT_S)
    release_callbacks[0].set()
    callbacks[0].join(_TIMEOUT_S)
    assert not callbacks[0].is_alive()
    assert wait_entries[1].wait(_TIMEOUT_S)
    release_callbacks[1].set()
    callbacks[1].join(_TIMEOUT_S)
    assert not callbacks[1].is_alive()
    _join(waiter, wait_done)
    assert callback_errors == [[], []]
    assert wait_errors == []
    gate.commit_lease_close(transition, lambda: None)


def test_async_claim_requires_identical_capability_and_is_exact_once():
    gate = _TerminalLifecycleGate()
    epoch = _open_lease(gate)
    parent = gate.admit_lease(epoch, "schedule_callback")
    capability = _AsyncClaimCapability()
    capability_ref = weakref.ref(capability)
    descendant = gate.spawn_async(parent, capability)
    gate.release(parent)

    with pytest.raises(RuntimeError, match="capability"):
        gate.claim_async(
            descendant, _AsyncClaimCapability(), "foreign_callback"
        )

    claimed = gate.claim_async(descendant, capability, "recorded_callback")
    with pytest.raises(RuntimeError, match="already claimed"):
        gate.claim_async(claimed, capability, "duplicate_callback")

    del capability
    gc.collect()
    assert capability_ref() is not None
    gate.release(claimed)
    gc.collect()
    assert capability_ref() is None
    _close_lease(gate, epoch)


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

    wait_entries = _observe_condition_waits(gate, 1)
    joiner, results, errors, done = _start(join_close)
    assert joined_transition.wait(_TIMEOUT_S)
    assert wait_entries[0].wait(_TIMEOUT_S)
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

    def close_runtime():
        transition = gate.begin_runtime_close(None)
        return gate.commit_runtime_close(transition, lambda: "runtime_closed")

    wait_entries = _observe_condition_waits(gate, 1)
    closer, results, errors, done = _start(close_runtime)
    assert wait_entries[0].wait(_TIMEOUT_S)
    gate.wait_for_lease_admissions(lease_transition, _TIMEOUT_S)
    gate.commit_lease_close(lease_transition, lambda: "lease_closed")
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


def test_runtime_close_drain_returns_fatal_transition_after_publication():
    gate = _TerminalLifecycleGate()
    blocker = gate.admit_runtime("block_runtime_close")
    wait_entered = threading.Event()
    original_wait = gate._condition.wait

    def observe_wait(timeout=None):
        wait_entered.set()
        return original_wait(timeout)

    gate._condition.wait = observe_wait
    closer, results, errors, done = _start(
        lambda: gate.begin_runtime_close(None)
    )
    assert wait_entered.wait(_TIMEOUT_S)
    snapshot = object()
    with gate._condition:
        fatal_transition = gate.begin_fatal(
            RuntimeError("fatal during runtime drain"), blocker
        )
        gate.publish_fatal(fatal_transition, snapshot)
    _join(closer, done)

    assert errors == []
    assert results == [fatal_transition]
    assert results[0] is gate._fatal_transition
    assert gate.phase is _TerminalPhase.FATAL_PUBLISHED


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
    assert finalized == []
    assert preempted.phase is _TerminalPhase.RUNTIME_CLOSED

    committed = _TerminalLifecycleGate()
    close_transition = committed.begin_runtime_close(None)
    assert committed.commit_runtime_close(close_transition, lambda: "closed") == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        committed.begin_fatal(RuntimeError("fatal after runtime commit"))


def test_runtime_close_commit_selection_is_fatal_linearization_point():
    gate = _TerminalLifecycleGate()
    transition = gate.begin_runtime_close(None)

    assert gate.select_runtime_close_commit(transition) is transition
    assert gate._runtime_close_commit_selected is True

    with pytest.raises(RuntimeError, match="committed"):
        gate.begin_fatal(RuntimeError("fatal after close selection"))
    assert gate._fatal_transition is None

    close_step = gate.admit_runtime_close(transition, "selected_collective_close")
    gate.release(close_step)
    finalized = []
    assert gate.commit_runtime_close(
        transition, lambda: finalized.append("closed")
    ) is None
    assert finalized == ["closed"]
    assert gate.phase is _TerminalPhase.RUNTIME_CLOSED


def test_pending_runtime_finalizer_precedes_publication_and_published_join_skips_it():
    pending = _TerminalLifecycleGate()
    pending_transition = pending.begin_fatal(RuntimeError("pending fatal"))
    finalizer_called = threading.Event()
    finalizer_observations = []

    def finalize_pending():
        finalizer_observations.append(
            (pending.phase, pending._has_active_tokens())
        )
        finalizer_called.set()

    def join_pending():
        transition = pending.begin_runtime_close(None)
        return pending.commit_runtime_close(transition, finalize_pending)

    closer, results, errors, done = _start(join_pending)
    assert finalizer_called.wait(_TIMEOUT_S)
    assert finalizer_observations == [(_TerminalPhase.FATAL_PENDING, False)]
    snapshot = object()
    pending.publish_fatal(pending_transition, snapshot)
    _join(closer, done)
    assert errors == []
    assert results == [snapshot]

    published = _TerminalLifecycleGate()
    published_transition = published.begin_fatal(RuntimeError("published fatal"))
    published_snapshot = object()
    published.publish_fatal(published_transition, published_snapshot)
    finalized_after_publication = []
    joined = published.begin_runtime_close(None)
    result = published.commit_runtime_close(
        joined, lambda: finalized_after_publication.append(True)
    )
    assert result is published_snapshot
    assert finalized_after_publication == []


def test_fatal_publication_waits_for_elected_runtime_finalizer():
    gate = _TerminalLifecycleGate()
    blocker = gate.admit_runtime("block_pending_finalizer")
    fatal_transition = gate.begin_fatal(RuntimeError("fatal"))
    finalizer_observations = []
    wait_entries = _observe_condition_waits(gate, 3)

    def close_runtime():
        transition = gate.begin_runtime_close(None)

        def finalizer():
            finalizer_observations.append(
                (gate.phase, gate._has_active_tokens())
            )

        return gate.commit_runtime_close(transition, finalizer)

    closer, results, errors, done = _start(close_runtime)
    assert wait_entries[0].wait(_TIMEOUT_S)
    snapshot = object()
    with gate._condition:
        gate.release(blocker)
        gate.publish_fatal(fatal_transition, snapshot)

    assert wait_entries[1].is_set()
    _join(closer, done)
    assert errors == []
    assert results == [snapshot]
    assert finalizer_observations == [(_TerminalPhase.FATAL_PENDING, False)]


def test_waiting_fatal_publishers_cannot_overwrite_snapshot():
    gate = _TerminalLifecycleGate()
    transition = gate.begin_fatal(RuntimeError("fatal"))
    gate._fatal_runtime_finalizer_state = "pending"
    wait_entries = _observe_condition_waits(gate, 2)
    first_snapshot = object()
    second_snapshot = object()

    def publish(snapshot):
        gate.publish_fatal(transition, snapshot)
        return snapshot

    first, first_results, first_errors, first_done = _start(
        lambda: publish(first_snapshot)
    )
    second, second_results, second_errors, second_done = _start(
        lambda: publish(second_snapshot)
    )
    assert wait_entries[0].wait(_TIMEOUT_S)
    assert wait_entries[1].wait(_TIMEOUT_S)
    with gate._condition:
        gate._fatal_runtime_finalizer_state = "complete"
        gate._condition.notify_all()
    _join(first, first_done)
    _join(second, second_done)

    results = first_results + second_results
    errors = first_errors + second_errors
    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert gate.wait_for_published(_TIMEOUT_S) is results[0]


def test_runtime_close_joins_pending_and_published_fatal():
    pending = _TerminalLifecycleGate()
    primary = RuntimeError("pending fatal")
    fatal_transition = pending.begin_fatal(primary)
    finalized = threading.Event()
    cleared = []

    def join_pending():
        transition = pending.begin_runtime_close(None)
        assert transition == fatal_transition

        def finalizer():
            cleared.append("pending")
            finalized.set()

        return pending.commit_runtime_close(transition, finalizer)

    wait_entries = _observe_condition_waits(pending, 1)
    closer, results, errors, done = _start(join_pending)
    assert finalized.wait(_TIMEOUT_S)
    assert wait_entries[0].wait(_TIMEOUT_S)
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


def test_fatal_preempts_runtime_close_before_commit():
    gate = _TerminalLifecycleGate()
    close_transition = gate.begin_runtime_close(None)
    primary = RuntimeError("fatal before runtime close commit")
    fatal_transition = gate.begin_fatal(primary)
    finalizer_called = threading.Event()
    finalizer_observations = []

    def close_pending_runtime():
        def finalizer():
            finalizer_observations.append(
                (gate.phase, gate._has_active_tokens())
            )
            finalizer_called.set()

        return gate.commit_runtime_close(fatal_transition, finalizer)

    wait_entries = _observe_condition_waits(gate, 1)
    closer, results, errors, done = _start(close_pending_runtime)
    assert finalizer_called.wait(_TIMEOUT_S)
    assert wait_entries[0].wait(_TIMEOUT_S)
    assert finalizer_observations == [(_TerminalPhase.FATAL_PENDING, False)]
    with pytest.raises(RuntimeError, match="preempted"):
        gate.admit_runtime_close(close_transition, "late_collective_close")

    snapshot = object()
    gate.publish_fatal(fatal_transition, snapshot)
    _join(closer, done)

    assert errors == []
    assert results == [snapshot]
    assert gate.phase is _TerminalPhase.RUNTIME_CLOSED


def test_fatal_after_runtime_close_commit_touches_no_collective(monkeypatch):
    from renormalizer.backend.distributed_runtime import CupyDistributedRuntime

    calls = []

    class Collective:
        def __init__(self):
            self._fatal_publication_local = threading.local()

        def _close_for_runtime(self, gate, transition):
            calls.append("close")

        def _begin_fatal_publication(self, *args, **kwargs):
            raise AssertionError("fake publication must stay behind runtime entry")

        def _publish_communicator_fatal(self, primary, **kwargs):
            calls.append(("fatal", primary, kwargs))
            hook = kwargs["handler_override"]
            transition = hook.begin(
                primary,
                owner=kwargs.get("fatal_owner"),
                discovering_token=kwargs.get("discovering_token"),
            )
            return transition.primary

    backend = SimpleNamespace(_execution_terminal_error=None)
    collective = Collective()
    runtime = CupyDistributedRuntime(
        backend=backend,
        context=SimpleNamespace(rank=0, local_rank=0, world_size=1),
        rendezvous=object(),
        mesh=object(),
        collective=collective,
    )
    fatal_entered = threading.Event()
    release_fatal_entry = threading.Event()
    primary = RuntimeError("late fatal")
    original_begin = runtime._begin_communicator_fatal

    def delay_fatal_entry(*args, **kwargs):
        fatal_entered.set()
        assert release_fatal_entry.wait(_TIMEOUT_S)
        return original_begin(*args, **kwargs)

    monkeypatch.setattr(runtime, "_begin_communicator_fatal", delay_fatal_entry)

    def publish_late_fatal():
        return runtime._enter_communicator_fatal(primary)

    late, results, errors, done = _start(publish_late_fatal)
    assert fatal_entered.wait(_TIMEOUT_S)
    closer, close_results, close_errors, close_done = _start(runtime.close)
    try:
        _join(closer, close_done)
        assert close_errors == []
        assert close_results == [None]
        assert runtime._terminal_gate.phase is _TerminalPhase.RUNTIME_CLOSED
        assert runtime._closed is True
        closed_state = (
            runtime._closed,
            runtime._terminal_gate.phase,
            runtime._terminal_error,
            backend._execution_terminal_error,
        )
        release_fatal_entry.set()
        _join(late, done)
    finally:
        release_fatal_entry.set()
        if closer.is_alive():
            _join(closer, close_done)
        if late.is_alive():
            _join(late, done)

    assert results == []
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "closed" in str(errors[0])
    assert calls == ["close"]
    assert (
        runtime._closed,
        runtime._terminal_gate.phase,
        runtime._terminal_error,
        backend._execution_terminal_error,
    ) == closed_state


def test_fatal_monitor_handoff_requires_distinct_exit_acknowledgment():
    handoff = _FatalMonitorHandoff()
    store_reads = []

    observed, value = handoff.read_store_if_unselected(
        lambda: store_reads.append("read") or 7
    )
    assert observed is None
    assert value == 7
    outcome = handoff.select_fatal(RuntimeError("fatal monitor outcome"))

    observed, value = handoff.read_store_if_unselected(
        lambda: store_reads.append("late read")
    )
    assert observed is outcome
    assert value is None
    assert store_reads == ["read"]

    with pytest.raises(TimeoutError, match="exit acknowledgment"):
        handoff.wait_for_exit(outcome, 0.0)

    handoff.acknowledge_exit(outcome)

    assert handoff.wait_for_exit(outcome, _TIMEOUT_S) is outcome


def test_fatal_monitor_handoff_hides_selection_until_reservation_is_joinable():
    handoff = _FatalMonitorHandoff()
    primary = RuntimeError("fatal selection waits for reservation")
    prepare_entered = threading.Event()
    release_prepare = threading.Event()
    prepared = []

    def prepare():
        assert handoff._outcome is None
        prepared.append(primary)
        prepare_entered.set()
        assert release_prepare.wait(_TIMEOUT_S)

    selector, results, errors, done = _start(
        lambda: handoff.select_fatal(primary, before_select=prepare)
    )
    try:
        assert prepare_entered.wait(_TIMEOUT_S)
        assert handoff._outcome is None
        assert prepared == [primary]
    finally:
        release_prepare.set()
        _join(selector, done)

    assert errors == []
    assert len(results) == 1
    assert results[0].kind == "fatal_elected"
    assert results[0].primary is primary
    assert handoff.observe_outcome() is results[0]


def test_close_during_pending_joins_publication():
    gate = _TerminalLifecycleGate()
    primary = RuntimeError("pending fatal")
    fatal_transition = gate.begin_fatal(primary)
    finalizer_called = threading.Event()
    finalized = []

    def close_runtime():
        joined = gate.begin_runtime_close(None)
        assert joined is fatal_transition

        def finalizer():
            finalized.append(primary)
            finalizer_called.set()

        return gate.commit_runtime_close(joined, finalizer)

    wait_entries = _observe_condition_waits(gate, 1)
    closer, results, errors, done = _start(close_runtime)
    assert finalizer_called.wait(_TIMEOUT_S)
    assert wait_entries[0].wait(_TIMEOUT_S)
    assert gate.phase is _TerminalPhase.FATAL_PENDING

    snapshot = object()
    gate.publish_fatal(fatal_transition, snapshot)
    _join(closer, done)

    assert errors == []
    assert results == [snapshot]
    assert finalized == [primary]
    assert gate._fatal_transition.primary is primary
