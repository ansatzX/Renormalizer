import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace

from renormalizer import BasisHalfSpin, set_backend
from renormalizer.backend._distributed.collectives import SingleProcessCollective
from renormalizer.backend._distributed.context import DistributedContext
from renormalizer.backend._distributed.mesh import DeviceMesh
from renormalizer.backend._distributed.providers import DeviceResidentProvider
from renormalizer.backend.config import DistributedExecutionConfig
from renormalizer.lib import expm_krylov
from renormalizer.model.model import heisenberg_ops
from renormalizer.tn import BasisTree, TTNO, TTNS
from renormalizer.tn.distributed import run_ttns_davidson, run_ttns_krylov
from renormalizer.tn.hop_expr import hop_expr1, hop_expr2
from renormalizer.tn.tree import TTNEnviron
from renormalizer.utils.configs import EvolveConfig, EvolveMethod


@pytest.fixture(autouse=True)
def _restore_backend():
    try:
        yield
    finally:
        set_backend("numpy", precision=64)


def _execution_config():
    context = DistributedContext(0, 0, 1, 1)
    return DistributedExecutionConfig(
        context=context,
        mesh=DeviceMesh((1,), ("rank",), 0),
        collective=SingleProcessCollective(),
        provider=DeviceResidentProvider(),
    )


class _StateTraceCollective(SingleProcessCollective):
    def __init__(self):
        self.trace = []

    def allreduce(self, value, *, op="sum"):
        self.trace.append((op, np.asarray(value).dtype.str))
        return np.array(value, copy=True)

    def allreduce_inplace(self, value, *, op="sum"):
        self.trace.append((op, np.asarray(value).dtype.str))
        return value


class _SetupTraceCollective(_StateTraceCollective):
    def broadcast(self, *_args, **_kwargs):
        pytest.fail("adapter entered a data collective after setup failure")

    def reduce_scatter(self, *_args, **_kwargs):
        pytest.fail("adapter entered a data collective after setup failure")

    def allgather(self, *_args, **_kwargs):
        pytest.fail("adapter entered a data collective after setup failure")


def _execution_with_collective(collective):
    context = DistributedContext(0, 0, 1, 1)
    return DistributedExecutionConfig(
        context=context,
        mesh=DeviceMesh((1,), ("rank",), 0),
        collective=collective,
        provider=DeviceResidentProvider(),
    )


class _NumericBroadcastReplay:
    def __init__(self, rank, values):
        self.rank = rank
        self.size = 2
        self.values = values
        self.read_index = 0
        self.broadcast_trace = []

    def allreduce(self, value, *, op="sum"):
        return np.array(value, copy=True)

    def broadcast(self, value, *, root):
        assert root == 0
        host = np.asarray(value)
        assert host.dtype != np.dtype(object)
        self.broadcast_trace.append((host.dtype.str, host.shape))
        if self.rank == 0:
            self.values.append(np.array(host, copy=True))
            return value
        source = self.values[self.read_index]
        self.read_index += 1
        assert source.dtype == host.dtype
        assert source.shape == host.shape
        value[...] = source
        return value

    def allgather(self, *_args, **_kwargs):
        pytest.fail("public fallback must not allgather complete state vectors")


def _ranked_execution(rank, collective, *, memory_budget=2**30):
    context = DistributedContext(rank, rank, 2, 2)
    return DistributedExecutionConfig(
        context=context,
        mesh=DeviceMesh((2,), ("rank",), rank),
        collective=collective,
        provider=DeviceResidentProvider(),
        device_memory_budget_bytes=memory_budget,
        host_memory_budget_bytes=memory_budget,
    )


def _tree_case():
    np.random.seed(1610)
    basis = BasisTree.binary([BasisHalfSpin(index) for index in range(7)])
    state = TTNS.random(basis, qntot=0, m_max=3)
    operator = TTNO(basis, heisenberg_ops(7))
    return state, operator, TTNEnviron(state, operator)


def _active_child(state):
    return next(
        node
        for node in state.node_list
        if not node.children and node.parent is not state.root
    )


def _descriptor(state, node, center_kind):
    node_index = state.node_idx[node]
    parent_index = -1 if node.parent is None else state.node_idx[node.parent]
    child_index = -1 if node.parent is None else node.parent.children.index(node)
    return {
        "node_index": node_index,
        "parent_index": parent_index,
        "child_index": child_index,
        "degree": len(node.children) + int(node.parent is not None),
        "center_kind": center_kind,
    }


def test_ttns_mapped_one_site_krylov_matches_local_expression():
    state, operator, environment = _tree_case()
    node = _active_child(state)
    center = node.tensor.copy()
    set_backend("numpy", execution_policy="execution_ir")
    hop = hop_expr1(node, state, operator, environment)
    expected, expected_iterations = expm_krylov(
        lambda value: hop(value.reshape(center.shape)).ravel(),
        -0.02j,
        center.ravel(),
    )
    counters = {}

    actual, iterations = run_ttns_krylov(
        hop,
        distributed_execution=_execution_config(),
        center=center.ravel(),
        center_shape=center.shape,
        coefficient=-0.02j,
        counters=counters,
        **_descriptor(state, node, "one_site"),
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert iterations == expected_iterations
    assert counters["allgather_calls"] == 0


def test_ttns_solve_telemetry_includes_post_solve_state_collectives(monkeypatch):
    from renormalizer.utils import profiling

    state, operator, environment = _tree_case()
    node = _active_child(state)
    center = node.tensor.copy()
    collective = _StateTraceCollective()
    set_backend("numpy", execution_policy="execution_ir")
    hop = hop_expr1(node, state, operator, environment)
    events = []
    monkeypatch.setattr(profiling, "enabled", lambda: True)
    monkeypatch.setattr(
        profiling,
        "record",
        lambda event, **payload: events.append({"event": event, **payload}),
    )

    run_ttns_krylov(
        hop,
        distributed_execution=_execution_with_collective(collective),
        center=center.ravel(),
        center_shape=center.shape,
        coefficient=-0.02j,
        **_descriptor(state, node, "one_site"),
    )

    summary = next(
        event for event in events if event["event"] == "distributed_solve_summary"
    )
    assert summary["allreduce_calls"] == len(collective.trace) - 3
    assert summary["allgather_calls"] == 0


def test_ttns_adapter_decision_includes_structural_topology(monkeypatch):
    from renormalizer.backend._distributed import center as center_boundary

    state, operator, environment = _tree_case()
    node = _active_child(state)
    center = node.tensor.copy()
    descriptor = _descriptor(state, node, "one_site")
    set_backend("numpy", execution_policy="execution_ir")
    hop = hop_expr1(node, state, operator, environment)
    captured = []
    original = center_boundary._adapter_decision_digest

    def recording(**kwargs):
        captured.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(center_boundary, "_adapter_decision_digest", recording)
    run_ttns_krylov(
        hop,
        distributed_execution=_execution_config(),
        center=center.ravel(),
        center_shape=center.shape,
        coefficient=-0.02j,
        solver_config={"block_size": 11},
        **descriptor,
    )

    decision = captured[0]
    assert decision["network"] == "ttns"
    assert decision["operation"] == "krylov"
    assert decision["center_kind"] == "one_site"
    assert decision["topology"] == {
        "node_index": descriptor["node_index"],
        "parent_index": descriptor["parent_index"],
        "child_index": descriptor["child_index"],
        "degree": descriptor["degree"],
    }
    assert decision["solver_controls"] == {
        "block_size": 11,
        "coefficient": -0.02j,
    }


def test_ttns_mapped_two_site_davidson_matches_local_energy():
    state, operator, environment = _tree_case()
    node = _active_child(state)
    center = state.merge_with_parent(node)
    mask = state.get_qnmask(node, include_parent=True)
    set_backend("numpy", execution_policy="execution_ir")
    hop, diagonal = hop_expr2(node, state, operator, environment)

    energy, full_center, info = run_ttns_davidson(
        hop,
        distributed_execution=_execution_config(),
        qn_mask=mask,
        initial_guess=center[mask],
        diagonal=diagonal[mask],
        solver_config={
            "tol": 1e-7,
            "max_cycle": 200,
            "max_space": 24,
            "lindep": 1e-14,
            "require_convergence": True,
        },
        **_descriptor(state, node, "two_site"),
    )

    residual = hop(full_center)[mask] - energy * full_center[mask]
    assert info.converged
    assert np.linalg.norm(residual) < 1e-7


@pytest.mark.parametrize(
    "failure_site",
    (
        "center_conversion",
        "center_allocation",
        "artifact_resolution",
        "operator_allocation",
        "guess_extraction",
        "diagonal_extraction",
    ),
)
def test_ttns_davidson_setup_failures_are_synchronized_before_data_collectives(
    failure_site, monkeypatch
):
    from renormalizer.backend._distributed import center as center_boundary
    from renormalizer.backend._distributed import local_operator

    class InjectedSetupFailure(BaseException):
        pass

    class FailingMask:
        def __array__(self, *_args, **_kwargs):
            raise InjectedSetupFailure("injected center conversion failure")

    set_backend("numpy", execution_policy="execution_ir")
    state, operator, environment = _tree_case()
    node = _active_child(state)
    center = state.merge_with_parent(node)
    mask = state.get_qnmask(node, include_parent=True)
    hop, diagonal = hop_expr2(node, state, operator, environment)
    collective = _SetupTraceCollective()
    execution = _execution_with_collective(collective)
    qn_mask = FailingMask() if failure_site == "center_conversion" else mask
    original_zeros = np.zeros
    original_extract = center_boundary.CenterVectorMap.extract_local
    extraction_calls = 0

    if failure_site == "center_allocation":
        def fail_center_zeros(shape, *args, **kwargs):
            if tuple(shape) == tuple(mask.shape):
                raise InjectedSetupFailure("injected center allocation failure")
            return original_zeros(shape, *args, **kwargs)

        monkeypatch.setattr(np, "zeros", fail_center_zeros)
    elif failure_site == "artifact_resolution":
        monkeypatch.setattr(
            hop,
            "resolve_execution_artifact",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                InjectedSetupFailure("injected artifact resolution failure")
            ),
        )
    elif failure_site == "operator_allocation":
        monkeypatch.setattr(
            local_operator,
            "DistributedLocalOperator",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                InjectedSetupFailure("injected operator allocation failure")
            ),
        )
    elif failure_site in {"guess_extraction", "diagonal_extraction"}:
        target = 1 if failure_site == "guess_extraction" else 2

        def fail_extraction(self, *args, **kwargs):
            nonlocal extraction_calls
            extraction_calls += 1
            if extraction_calls == target:
                raise InjectedSetupFailure("injected local extraction failure")
            return original_extract(self, *args, **kwargs)

        monkeypatch.setattr(
            center_boundary.CenterVectorMap, "extract_local", fail_extraction
        )

    with pytest.raises(RuntimeError, match="setup failed") as raised:
        run_ttns_davidson(
            hop,
            distributed_execution=execution,
            qn_mask=qn_mask,
            initial_guess=center[mask],
            diagonal=diagonal[mask],
            **_descriptor(state, node, "two_site"),
        )

    assert isinstance(raised.value.__cause__, InjectedSetupFailure)
    status = [("max", np.dtype(np.int32).str)]
    backend_validation = [
        ("max", np.dtype(np.int32).str),
        ("min", np.dtype(np.uint64).str),
        ("max", np.dtype(np.uint64).str),
    ]
    decision = backend_validation
    expected = {
        "center_conversion": backend_validation + status,
        "center_allocation": backend_validation + status * 2,
        "artifact_resolution": backend_validation + status * 3,
        "operator_allocation": (
            backend_validation + status * 3 + decision + status
        ),
        "guess_extraction": (
            backend_validation + status * 3 + decision + status * 2
        ),
        "diagonal_extraction": (
            backend_validation + status * 3 + decision + status * 2
        ),
    }[failure_site]
    assert collective.trace == expected


@pytest.mark.parametrize("failure_site", ("center_conversion", "local_extraction"))
def test_ttns_krylov_center_setup_failures_are_synchronized_before_solver_data(
    failure_site, monkeypatch
):
    from renormalizer.backend._distributed import center as center_boundary

    class InjectedSetupFailure(BaseException):
        pass

    class FailingCenter:
        def __array__(self, *_args, **_kwargs):
            raise InjectedSetupFailure("injected center conversion failure")

    set_backend("numpy", execution_policy="execution_ir")
    state, operator, environment = _tree_case()
    node = _active_child(state)
    center = node.tensor.copy()
    hop = hop_expr1(node, state, operator, environment)
    collective = _SetupTraceCollective()
    if failure_site == "local_extraction":
        monkeypatch.setattr(
            center_boundary.CenterVectorMap,
            "extract_local",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                InjectedSetupFailure("injected local extraction failure")
            ),
        )

    with pytest.raises(RuntimeError, match="setup failed") as raised:
        run_ttns_krylov(
            hop,
            distributed_execution=_execution_with_collective(collective),
            center=FailingCenter() if failure_site == "center_conversion" else center,
            center_shape=center.shape,
            coefficient=-0.01j,
            **_descriptor(state, node, "one_site"),
        )

    assert isinstance(raised.value.__cause__, InjectedSetupFailure)
    status = [("max", np.dtype(np.int32).str)]
    backend_validation = [
        ("max", np.dtype(np.int32).str),
        ("min", np.dtype(np.uint64).str),
        ("max", np.dtype(np.uint64).str),
    ]
    if failure_site == "center_conversion":
        expected = backend_validation + status
    else:
        expected = backend_validation + status * 2 + backend_validation + status * 2
    assert collective.trace == expected


@pytest.mark.parametrize(
    ("entry", "reason"),
    [
        ("davidson", "unsupported_ttns_davidson_center_kind"),
        ("ground_state", "unsupported_ground_state_mode"),
    ],
)
def test_ttns_invalid_center_kind_reaches_coordinated_adapter_decision(
    entry, reason, monkeypatch
):
    from renormalizer.tn import distributed as adapter

    class StopAtDecision(BaseException):
        pass

    captured = []

    def coordinate(*_args, **kwargs):
        captured.append(kwargs)
        raise StopAtDecision

    monkeypatch.setattr(adapter, "coordinate_adapter_decision", coordinate)
    set_backend(
        "numpy", execution_policy="execution_ir", fallback_policy="legacy_oe"
    )
    common = {
        "distributed_execution": _execution_config(),
        "qn_mask": np.array([True, True]),
        "initial_guess": np.ones(2),
        "node_index": 1,
        "parent_index": 0,
        "child_index": 0,
        "degree": 1,
        "center_kind": "one_site",
    }

    with pytest.raises(StopAtDecision):
        if entry == "davidson":
            adapter.run_ttns_davidson(
                lambda value: value,
                diagonal=np.ones(2),
                solver_config={"tol": 1e-8},
                **common,
            )
        else:
            adapter.run_ttns_ground_state_fallback(
                lambda: pytest.fail("legacy closure ran before center decision"),
                solver_controls={"algo": "davidson"},
                selectors={"method": "2site"},
                **common,
            )

    assert captured[0]["supported"] is False
    assert captured[0]["center_kind"] == "one_site"
    assert captured[0]["fallback_reason_code"] == reason


def test_ttns_tdvp_workflow_routes_one_and_zero_site_centers(monkeypatch):
    from renormalizer.tn import distributed as adapter

    state, operator, _ = _tree_case()
    state.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    state.evolve_config.distributed_execution = _execution_config()
    set_backend("numpy", execution_policy="execution_ir")
    calls = []
    synchronized = []
    original = adapter.run_ttns_krylov

    def recording(*args, **kwargs):
        calls.append(kwargs["center_kind"])
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "run_ttns_krylov", recording)
    def synchronize(*args, **kwargs):
        synchronized.append((args, kwargs))
        return kwargs["operation"]()

    monkeypatch.setattr(adapter, "_synchronize_ttns_state", synchronize)

    state.evolve(operator, 0.001, normalize=False)

    assert "one_site" in calls
    assert "zero_site" in calls
    assert len(synchronized) >= len(calls)


@pytest.mark.parametrize(
    ("method", "adaptive"),
    [
        (EvolveMethod.prop_and_compress_tdrk4, False),
        (EvolveMethod.tdvp_ps, True),
    ],
)
def test_ttns_public_evolve_coordinates_method_and_adaptive_before_conversion(
    method, adaptive, monkeypatch
):
    from renormalizer.tn import EVOLVE_METHODS
    from renormalizer.tn import distributed as adapter

    class StopAtDecision(BaseException):
        pass

    state, operator, _ = _tree_case()
    state.evolve_config = EvolveConfig(method, adaptive=adaptive)
    state.evolve_config.distributed_execution = _execution_config()
    captured = []
    trace = []

    def coordinate(*args, **kwargs):
        trace.append("decision")
        captured.append(kwargs)
        raise StopAtDecision

    def convert(*args, **kwargs):
        trace.append("to_complex")
        pytest.fail("TTNS converted before public workflow coordination")

    def dispatch(*args, **kwargs):
        trace.append("dispatch")
        pytest.fail("TTNS dispatched before public workflow coordination")

    monkeypatch.setattr(adapter, "coordinate_ttns_workflow_entry", coordinate)
    monkeypatch.setattr(type(state), "to_complex", convert)
    monkeypatch.setitem(EVOLVE_METHODS, method, dispatch)

    with pytest.raises(StopAtDecision):
        state.evolve(operator, 0.0125, normalize=False)

    assert trace == ["decision"]
    assert captured[0]["operation"] == "public_evolve"
    assert captured[0]["supported"] is False
    assert captured[0]["selectors"] == {
        "method": method.name,
        "normalize": False,
    }
    controls = captured[0]["solver_controls"]
    assert controls["adaptive"] is adaptive
    assert controls["guess_dt"] == state.evolve_config.guess_dt
    assert controls["adaptive_rtol"] == state.evolve_config.adaptive_rtol
    assert controls["ivp_solver"] == state.evolve_config.ivp_solver
    assert controls["ivp_rtol"] == state.evolve_config.ivp_rtol
    assert controls["ivp_atol"] == state.evolve_config.ivp_atol
    assert controls["tau"] == 0.0125
    assert controls["compression"]["criteria"] == "threshold"


def test_ttns_public_unsupported_evolve_rejects_before_legacy_workflow(monkeypatch):
    from renormalizer.tn import EVOLVE_METHODS
    from renormalizer.tn import distributed as adapter

    state, operator, _ = _tree_case()
    collective = _StateTraceCollective()
    execution = _execution_with_collective(collective)
    method = EvolveMethod.prop_and_compress_tdrk4
    state.evolve_config = EvolveConfig(method)
    state.evolve_config.distributed_execution = execution
    set_backend(
        "numpy", execution_policy="execution_ir", fallback_policy="legacy_oe"
    )
    calls = {"copy": 0, "dispatch": 0, "normalize": 0}
    monkeypatch.setattr(
        type(state),
        "copy",
        lambda *_args, **_kwargs: calls.__setitem__("copy", calls["copy"] + 1),
    )
    monkeypatch.setitem(
        EVOLVE_METHODS,
        method,
        lambda *_args, **_kwargs: calls.__setitem__(
            "dispatch", calls["dispatch"] + 1
        ),
    )
    monkeypatch.setattr(
        type(state), "normalize",
        lambda *_args, **_kwargs: calls.__setitem__(
            "normalize", calls["normalize"] + 1
        ),
    )

    with pytest.raises(
        NotImplementedError, match="complete workflow capacity cannot be proven"
    ):
        state.evolve(operator, 0.01j, normalize=True)

    assert calls == {"copy": 0, "dispatch": 0, "normalize": 0}
    assert collective.trace == [
        ("max", np.dtype(np.int32).str),
        ("min", np.dtype(np.uint64).str),
        ("max", np.dtype(np.uint64).str),
    ] * 2
    assert not hasattr(adapter, "run_ttns_public_evolve_fallback")


@pytest.mark.parametrize("fail_conversion", [False, True])
def test_ttns_public_to_complex_is_a_synchronized_state_phase(
    fail_conversion, monkeypatch
):
    from renormalizer.tn import EVOLVE_METHODS
    from renormalizer.tn import distributed as adapter

    class InjectedConversionFailure(BaseException):
        pass

    state, operator, _ = _tree_case()
    collective = _StateTraceCollective()
    state.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    state.evolve_config.distributed_execution = _execution_with_collective(collective)
    monkeypatch.setattr(
        adapter, "coordinate_ttns_workflow_entry", lambda *args, **kwargs: "supported"
    )
    monkeypatch.setitem(
        EVOLVE_METHODS, EvolveMethod.tdvp_ps, lambda current, *_args: current
    )
    if fail_conversion:
        monkeypatch.setattr(
            type(state),
            "to_complex",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                InjectedConversionFailure("injected conversion failure")
            ),
        )
        with pytest.raises(RuntimeError, match="distributed state update failed"):
            state.evolve(operator, 0.01, normalize=False)
        assert collective.trace == [("max", np.dtype(np.int32).str)]
    else:
        state.evolve(operator, 0.01, normalize=False)
        assert collective.trace == [
            ("max", np.dtype(np.int32).str),
            ("min", np.dtype(np.uint64).str),
            ("max", np.dtype(np.uint64).str),
        ]


@pytest.mark.parametrize("fail_normalization", [False, True])
def test_ttns_public_normalization_is_a_synchronized_state_phase(
    fail_normalization, monkeypatch
):
    from renormalizer.tn import EVOLVE_METHODS
    from renormalizer.tn import distributed as adapter

    class InjectedNormalizationFailure(BaseException):
        pass

    state, operator, _ = _tree_case()
    collective = _StateTraceCollective()
    state.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    state.evolve_config.distributed_execution = _execution_with_collective(collective)
    monkeypatch.setattr(
        adapter, "coordinate_ttns_workflow_entry", lambda *args, **kwargs: "supported"
    )
    monkeypatch.setitem(
        EVOLVE_METHODS, EvolveMethod.tdvp_ps, lambda current, *_args: current.copy()
    )
    if fail_normalization:
        monkeypatch.setattr(
            type(state),
            "normalize",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                InjectedNormalizationFailure("injected normalization failure")
            ),
        )
        with pytest.raises(RuntimeError, match="distributed state update failed"):
            state.evolve(operator, 0.01j, normalize=True)
        assert collective.trace == [("max", np.dtype(np.int32).str)]
    else:
        state.evolve(operator, 0.01j, normalize=True)
        assert collective.trace == [
            ("max", np.dtype(np.int32).str),
            ("min", np.dtype(np.uint64).str),
            ("max", np.dtype(np.uint64).str),
        ]


def test_ttns_workflow_entry_decision_precedes_canonical_environment_and_expression(
    monkeypatch,
):
    from renormalizer.tn import distributed as adapter
    from renormalizer.tn import time_evolution

    state, operator, _ = _tree_case()
    state.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    state.evolve_config.distributed_execution = _execution_config()
    set_backend("numpy", execution_policy="execution_ir")
    trace = []
    original_decision = adapter.coordinate_adapter_decision
    original_check = type(state).check_canonical
    original_environment = time_evolution.TTNEnviron
    original_expression = time_evolution.hop_expr1

    def decision(*args, **kwargs):
        trace.append("decision")
        return original_decision(*args, **kwargs)

    def check(current, *args, **kwargs):
        trace.append("canonical")
        return original_check(current, *args, **kwargs)

    def environment(*args, **kwargs):
        trace.append("environment")
        return original_environment(*args, **kwargs)

    def expression(*args, **kwargs):
        trace.append("expression")
        return original_expression(*args, **kwargs)

    monkeypatch.setattr(adapter, "coordinate_adapter_decision", decision)
    monkeypatch.setattr(type(state), "check_canonical", check)
    monkeypatch.setattr(time_evolution, "TTNEnviron", environment)
    monkeypatch.setattr(time_evolution, "hop_expr1", expression)

    state.evolve(operator, 0.001, normalize=False)

    assert trace[0] == "decision"
    assert trace.index("decision") < trace.index("canonical")
    assert trace.index("decision") < trace.index("environment")
    assert trace.index("decision") < trace.index("expression")


def test_ttns_ground_state_routes_existing_two_site_davidson(monkeypatch):
    from renormalizer.tn import distributed as adapter
    from renormalizer.tn.gs import optimize_ttns

    state, operator, _ = _tree_case()
    state.optimize_config.algo = "davidson"
    state.optimize_config.nroots = 1
    state.optimize_config.distributed_execution = _execution_config()
    set_backend(
        "numpy", execution_policy="execution_ir", fallback_policy="legacy_oe"
    )
    calls = []
    synchronized = []
    original = adapter.run_ttns_davidson

    def recording(*args, **kwargs):
        calls.append(
            (kwargs["center_kind"], kwargs["decision_selectors"])
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "run_ttns_davidson", recording)
    def synchronize(*args, **kwargs):
        synchronized.append((args, kwargs))
        return kwargs["operation"]()

    monkeypatch.setattr(adapter, "_synchronize_ttns_state", synchronize)

    energies = optimize_ttns(state, operator, [[3, 0]])

    assert np.all(np.isfinite(energies))
    assert calls
    assert {kind for kind, _ in calls} == {"two_site"}
    assert all(
        selectors
        == {
            "algo": "davidson",
            "nroots": 1,
            "ground_state_topology": "existing_two_site",
        }
        for _, selectors in calls
    )
    assert len(synchronized) >= len(calls)


def test_ttns_local_ground_algorithm_routes_through_synchronized_fallback(monkeypatch):
    from renormalizer.tn import distributed as adapter
    from renormalizer.tn.gs import optimize_ttns

    state, operator, _ = _tree_case()
    state.optimize_config.algo = "direct"
    state.optimize_config.nroots = 1
    state.optimize_config.distributed_execution = _execution_config()
    set_backend(
        "numpy", execution_policy="execution_ir", fallback_policy="legacy_oe"
    )
    calls = []
    original = adapter.run_ttns_ground_state_fallback

    def recording(*args, **kwargs):
        calls.append((kwargs["solver_controls"], kwargs["center_kind"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "run_ttns_ground_state_fallback", recording)
    energies = optimize_ttns(state, operator, [[3, 0]])

    assert np.all(np.isfinite(energies))
    assert calls
    assert all(controls["algo"] == "direct" for controls, _ in calls)
    assert {kind for _, kind in calls} == {"two_site"}


def test_ttns_unsupported_ground_error_precedes_hamiltonian_construction(monkeypatch):
    from renormalizer.tn import gs
    from renormalizer.tn.gs import optimize_ttns

    state, operator, _ = _tree_case()
    state.optimize_config.algo = "direct"
    state.optimize_config.nroots = 1
    state.optimize_config.distributed_execution = _execution_config()
    set_backend("numpy", execution_policy="execution_ir", fallback_policy="error")
    calls = []
    original = gs.hop_expr2

    def recording(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(gs, "hop_expr2", recording)
    with pytest.raises(NotImplementedError, match="distributed adapter decision failed"):
        optimize_ttns(state, operator, [[3, 0]])

    assert calls == []


def test_ttns_ground_entry_decision_precedes_environment_and_center_setup(monkeypatch):
    from renormalizer.tn import distributed as adapter
    from renormalizer.tn import gs
    from renormalizer.tn.gs import optimize_ttns

    state, operator, _ = _tree_case()
    state.optimize_config.algo = "direct"
    state.optimize_config.nroots = 1
    state.optimize_config.distributed_execution = _execution_config()
    set_backend("numpy", execution_policy="execution_ir", fallback_policy="error")
    trace = []
    original_decision = adapter.coordinate_adapter_decision
    original_environment = gs.TTNEnviron
    original_merge = type(state).merge_with_parent

    def decision(*args, **kwargs):
        trace.append("decision")
        return original_decision(*args, **kwargs)

    def environment(*args, **kwargs):
        trace.append("environment")
        return original_environment(*args, **kwargs)

    def merge(current, *args, **kwargs):
        trace.append("center_setup")
        return original_merge(current, *args, **kwargs)

    monkeypatch.setattr(adapter, "coordinate_adapter_decision", decision)
    monkeypatch.setattr(gs, "TTNEnviron", environment)
    monkeypatch.setattr(type(state), "merge_with_parent", merge)

    with pytest.raises(NotImplementedError, match="distributed adapter decision failed"):
        optimize_ttns(state, operator, [[3, 0]])

    assert trace == ["decision"]


def test_ttns_ground_entry_coordinates_effective_procedure_and_convergence(
    monkeypatch,
):
    from renormalizer.tn import distributed as adapter
    from renormalizer.tn.gs import optimize_ttns

    class StopAtDecision(BaseException):
        pass

    state, operator, _ = _tree_case()
    state.optimize_config.e_rtol = 7e-9
    state.optimize_config.e_atol = 8e-11
    state.optimize_config.distributed_execution = _execution_config()
    procedure = [[[2, 3, 4, 5, 4, 3, 2], 0.2], [6, 0.0]]
    captured = []

    def coordinate(*args, **kwargs):
        captured.append(kwargs)
        raise StopAtDecision

    monkeypatch.setattr(adapter, "coordinate_ttns_workflow_entry", coordinate)

    with pytest.raises(StopAtDecision):
        optimize_ttns(state, operator, procedure)

    controls = captured[0]["solver_controls"]
    assert controls["convergence"] == {"rtol": 7e-9, "atol": 8e-11}
    assert controls["effective_procedure"] == [
        {"compression": [2, 3, 4, 5, 4, 3, 2], "percent": 0.2},
        {"compression": 6, "percent": 0.0},
    ]


def test_ttns_distributed_ground_workflow_final_normalization_is_synchronized(
    monkeypatch,
):
    from renormalizer.tn import distributed as adapter
    from renormalizer.tn import gs

    state, operator, _ = _tree_case()
    collective = _StateTraceCollective()
    state.optimize_config.algo = "davidson"
    state.optimize_config.nroots = 1
    state.optimize_config.distributed_execution = _execution_with_collective(
        collective
    )
    phases = []
    original = adapter._synchronize_ttns_state

    def synchronize(*args, **kwargs):
        phases.append(tuple(kwargs["metadata"]))
        return original(*args, **kwargs)

    def unnormalized_sweep(*_args, **_kwargs):
        state.root.tensor *= 2.0
        return [-1.0]

    monkeypatch.setattr(adapter, "_synchronize_ttns_state", synchronize)
    monkeypatch.setattr(gs, "optimize_recursion", unnormalized_sweep)
    set_backend("numpy", execution_policy="execution_ir")

    energies = gs.optimize_ttns(state, operator, [[3, 0.0]])

    assert energies == [-1.0]
    assert state.ttns_norm == pytest.approx(1.0, abs=1e-12)
    assert ("davidson", state.node_idx[state.root], "final_normalization") in phases


@pytest.mark.parametrize("failure_site", ["decomposition", "environment"])
def test_ttns_complete_center_transition_is_one_baseexception_state_phase(
    failure_site, monkeypatch
):
    from renormalizer.tn import time_evolution

    class InjectedTransitionFailure(BaseException):
        pass

    state, operator, _ = _tree_case()
    state.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    state.evolve_config.distributed_execution = _execution_config()
    set_backend("numpy", execution_policy="execution_ir")

    if failure_site == "decomposition":
        monkeypatch.setattr(
            type(state),
            "decompose_to_parent",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                InjectedTransitionFailure("injected decomposition failure")
            ),
        )
    else:
        monkeypatch.setattr(
            time_evolution.TTNEnviron,
            "build_children_environ_node",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                InjectedTransitionFailure("injected environment update failure")
            ),
        )

    with pytest.raises(RuntimeError, match="distributed state update failed"):
        state.evolve(operator, 0.001, normalize=False)


def test_ttns_unsupported_hop_uses_synchronized_root_fallback():
    from scipy.linalg import expm

    rng = np.random.default_rng(1611)
    matrix = rng.normal(size=(5, 5))
    matrix = matrix + matrix.T
    center = rng.normal(size=5)
    set_backend(
        "numpy", execution_policy="execution_ir", fallback_policy="legacy_oe"
    )
    counters = {}

    actual, iterations = run_ttns_krylov(
        lambda value: matrix @ value,
        distributed_execution=_execution_config(),
        center=center,
        center_shape=center.shape,
        node_index=2,
        parent_index=1,
        child_index=0,
        degree=1,
        center_kind="one_site",
        coefficient=-0.02,
        counters=counters,
    )

    np.testing.assert_allclose(actual, expm(-0.02 * matrix) @ center, atol=1e-13)
    assert iterations > 0
    assert counters["fallback_count"] == 1
    assert counters["root_operation_count"] == 1


def test_ttns_unsupported_davidson_uses_synchronized_root_fallback():
    rng = np.random.default_rng(1612)
    matrix = rng.normal(size=(6, 6))
    matrix = matrix + matrix.T
    mask = np.array([True, True, False, True, False, True])
    guess = rng.normal(size=mask.sum())
    set_backend(
        "numpy", execution_policy="execution_ir", fallback_policy="legacy_oe"
    )
    counters = {}

    energy, center, info = run_ttns_davidson(
        lambda value: matrix @ value,
        distributed_execution=_execution_config(),
        qn_mask=mask,
        initial_guess=guess,
        diagonal=np.diag(matrix)[mask],
        node_index=2,
        parent_index=1,
        child_index=0,
        degree=1,
        center_kind="two_site",
        solver_config={"tol": 1e-10, "max_cycle": 100, "max_space": 8},
        counters=counters,
    )

    allowed = matrix[np.ix_(mask, mask)]
    np.testing.assert_allclose(energy, np.linalg.eigvalsh(allowed)[0], atol=1e-10)
    np.testing.assert_allclose(allowed @ center[mask], energy * center[mask], atol=1e-8)
    assert info.h_v_count > 0
    assert counters["root_operation_count"] == 1


def test_ttns_job_parser_keeps_davidson_fixed_to_existing_two_site():
    from renormalizer.tn.distributed_job import build_parser

    args = build_parser().parse_args(
        [
            "--mode", "both",
            "--output-dir", "/tmp/ttns-stage4",
            "--expected-world-size", "2",
            "--model-size", "7",
            "--bond-dimension", "8",
            "--shard-solver-vectors",
            "--tdvp-sites", "two",
        ]
    )

    assert args.mode == "both"
    assert args.tdvp_sites == "two"
    assert not hasattr(args, "davidson_sites")


def test_ttns_davidson_dtype_resolution_does_not_coerce_device_arrays():
    from renormalizer.tn.distributed import _result_dtype

    class DeviceArray:
        dtype = np.dtype("complex128")

        def __array__(self):
            raise AssertionError("device array was coerced to NumPy")

    assert _result_dtype(np.ones(2, dtype=np.float64), DeviceArray()) == np.dtype(
        "complex128"
    )


def test_ttns_adapter_has_no_private_mps_adapter_imports():
    source = (
        Path(__file__).parents[1] / "distributed.py"
    ).read_text(encoding="utf-8")

    assert "from renormalizer.mps.distributed import" not in source
    assert "renormalizer.mps.distributed" not in source


def test_ttns_summary_uses_real_spread_and_logical_fallback_count(tmp_path):
    from renormalizer.backend._distributed.job import RANK_SCHEMA, _write_json
    from renormalizer.tn.distributed_job import _root_summary, build_parser

    args = build_parser().parse_args(
        [
            "--mode", "both",
            "--output-dir", str(tmp_path),
            "--expected-world-size", "2",
            "--shard-solver-vectors",
            "--rank-atol", "0.2",
        ]
    )
    base = {
        "schema": RANK_SCHEMA,
        "run_id": "run",
        "world_size": 2,
        "git_commit": "abc",
        "state_hash": "same",
        "solver": {"tol": 1e-6},
        "distribution": {
            "h_v_count": 4,
            "broadcast_calls": 5,
            "allreduce_calls": 6,
            "allgather_calls": 0,
            "boundary_materialization_broadcasts": 1,
            "allgather_trap_armed": True,
            "strictly_sharded": True,
            "supported_event_count": 1,
            "supported_plan_hashes": ["plan"],
            "supported_placement_hashes": ["placement"],
            "plan_hashes": ["plan"],
            "placement_hashes": ["placement"],
            "max_solver_residual_norm": 0.0,
        },
        "fallback": {"count": 1, "root_operation_count": 0},
    }
    records = [
        dict(
            base,
            rank=0,
            local_rank=0,
            numerical={
                "passed": True,
                "max_abs_error": 0.0,
                "energy": -2.0,
                "energy_error": 0.0,
                "tdvp_norm_error": 0.0,
                "davidson_norm_error": 0.0,
                "davidson_reference_norm": 1.0,
                "norm_error": 0.0,
                "residual_norm": 0.0,
            },
            fallback={"count": 1, "root_operation_count": 1},
        ),
        dict(
            base,
            rank=1,
            local_rank=1,
            numerical={
                "passed": True,
                "max_abs_error": 0.0,
                "energy": -1.9,
                "energy_error": 0.0,
                "tdvp_norm_error": 0.0,
                "davidson_norm_error": 0.0,
                "davidson_reference_norm": 1.0,
                "norm_error": 0.0,
                "residual_norm": 0.0,
            },
        ),
    ]
    for record in records:
        _write_json(tmp_path / "rank-{:05d}.json".format(record["rank"]), record)

    summary = _root_summary(
        args, SimpleNamespace(world_size=2), tmp_path, records[0]
    )

    assert summary["rank_consistency"]["max_scalar_spread"] == pytest.approx(0.1)
    assert summary["fallback"]["count"] == 1
    assert summary["fallback"]["root_operation_count"] == 1
    assert summary["distribution"]["supported_event_count"] == 1
    assert summary["distribution"]["supported_plan_hashes"] == ["plan"]
    assert summary["distribution"]["supported_placement_hashes"] == ["placement"]
    assert summary["status"] == "pass"


def test_ttns_job_copy_phase_synchronizes_baseexception_failure(monkeypatch):
    from renormalizer.tn.distributed_job import _run_tdvp

    class InjectedCopyFailure(BaseException):
        pass

    class Collective:
        rank = 0
        size = 2

        def __init__(self):
            self.trace = []

        def allreduce(self, value, *, op="sum"):
            self.trace.append((op, np.asarray(value).dtype.str))
            return np.array(value, copy=True)

    state, operator, _ = _tree_case()
    monkeypatch.setattr(
        type(state),
        "copy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            InjectedCopyFailure("injected TTNS copy failure")
        ),
    )
    args = SimpleNamespace(tdvp_sites="one")
    execution = SimpleNamespace(
        context=SimpleNamespace(rank=0), collective=Collective()
    )

    with pytest.raises(RuntimeError, match="TTNS TDVP workflow setup failed"):
        _run_tdvp(args, state, operator, execution)

    assert execution.collective.trace == [("max", np.dtype(np.int32).str)]


def test_ttns_launcher_has_explicit_gpu_safety_and_owned_cleanup():
    script = Path(__file__).parents[3] / "scripts" / "run_tree_multigpu_job.sh"
    text = script.read_text()

    assert "set -euo pipefail" in text
    assert "CUDA_VISIBLE_DEVICES" in text
    assert "--query-compute-apps" in text
    assert "nvidia-smi dmon -s pucvmte -d 1 -o T" in text
    assert "flock" in text
    assert "torch.distributed.run" in text
    assert "$1 ~ uuids" not in text
    assert "index(uuids" in text
    assert "pkill" not in text
    assert "killall" not in text
    probe_loop = text.index("for sample in range(2):")
    compute_probe = text.index("--query-compute-apps", probe_loop)
    probe_sleep = text.index("time.sleep(2.0)", probe_loop)
    assert probe_loop < compute_probe < probe_sleep
    assert "torchrun_pid=$!" in text
    assert 'signal_owned_process "$torchrun_pid" "torchrun"' in text
    assert 'wait "$torchrun_pid"' in text
    assert "ss -ltnp" in text
    assert "surviving_owned_listeners" in text
