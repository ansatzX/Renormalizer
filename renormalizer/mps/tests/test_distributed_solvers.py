import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from renormalizer import set_backend
from renormalizer.backend._distributed.collectives import SingleProcessCollective
from renormalizer.backend._distributed.context import DistributedContext
from renormalizer.backend._distributed.mesh import DeviceMesh
from renormalizer.backend._distributed.providers import DeviceResidentProvider
from renormalizer.backend.config import DistributedExecutionConfig
from renormalizer.lib import expm_krylov
from renormalizer.mps.distributed import (
    run_mps_davidson,
    run_mps_ground_state_fallback,
    run_mps_ivp_fallback,
    run_mps_krylov,
)
from renormalizer.utils.configs import EvolveConfig, OptimizeConfig


mps_hop = importlib.import_module("renormalizer.mps.hop_expr")


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


def _mps_case():
    from renormalizer import BasisHalfSpin, Model, Mpo, Mps
    from renormalizer.model.model import heisenberg_ops

    np.random.seed(1603)
    model = Model([BasisHalfSpin(index) for index in range(4)], heisenberg_ops(4))
    return Mps.random(model, 0, 4, 1), Mpo(model)


def test_optional_distributed_config_preserves_legacy_text_and_shallow_copy():
    evolve = EvolveConfig()
    optimize = OptimizeConfig()
    legacy_text = "".join(
        "\n{}: {}".format(attr, getattr(evolve, attr))
        for attr in evolve.__dict__
        if attr != "distributed_execution"
    )

    assert evolve.distributed_execution is None
    assert optimize.distributed_execution is None
    assert str(evolve) == legacy_text

    handle = object()
    evolve.distributed_execution = handle
    optimize.distributed_execution = handle
    assert evolve.copy().distributed_execution is handle
    assert optimize.copy().distributed_execution is handle


def test_non_none_evolve_config_text_is_stable_bounded_and_opaque():
    class OpaqueCollective(SingleProcessCollective):
        def __repr__(self):
            pytest.fail("collective repr must not be used by EvolveConfig.__str__")

    class OpaqueProvider(DeviceResidentProvider):
        def __repr__(self):
            pytest.fail("provider repr must not be used by EvolveConfig.__str__")

    def config_text():
        context = DistributedContext(0, 0, 1, 1)
        execution = DistributedExecutionConfig(
            context=context,
            mesh=DeviceMesh((1,), ("rank",), 0),
            collective=OpaqueCollective(),
            provider=OpaqueProvider(),
            device_memory_budget_bytes=2**30,
            host_memory_budget_bytes=2**31,
            prefetch_depth=2,
            backend_name="numpy",
            backend_device="cpu",
            backend_precision=64,
        )
        config = EvolveConfig()
        config.distributed_execution = execution
        return str(config)

    first = config_text()
    second = config_text()

    assert first == second
    assert len(first) < 1024
    assert "distributed_execution: DistributedExecutionConfig(" in first
    assert "rank=0" in first
    assert "mesh_shape=(1,)" in first
    assert "backend_name='numpy'" in first
    assert "collective" not in first
    assert "provider" not in first
    assert "object at" not in first
    assert "0x" not in first


def test_mps_mapped_krylov_matches_local_expression_without_allgather():
    rng = np.random.default_rng(1601)
    matrix = rng.normal(size=(6, 6))
    matrix = matrix + matrix.T
    center = rng.normal(size=6)
    set_backend("numpy", execution_policy="execution_ir")
    hop = mps_hop._build_hop("ab,b->a", (matrix,), center.shape, "one_site")
    expected, expected_iterations = expm_krylov(
        hop, -0.03, center.copy(), block_size=6
    )
    counters = {}

    actual, iterations = run_mps_krylov(
        hop,
        distributed_execution=_execution_config(),
        center=center,
        center_shape=center.shape,
        site_indices=(2,),
        center_kind="one_site",
        coefficient=-0.03,
        solver_config={"block_size": 6},
        counters=counters,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert iterations == expected_iterations
    assert counters["allgather_calls"] == 0
    assert counters["boundary_materialization_broadcasts"] == 1


def test_mps_adapter_decision_includes_topology_controls_and_selectors(monkeypatch):
    from renormalizer.backend._distributed import center as center_boundary

    matrix = np.eye(4)
    center = np.arange(4, dtype=np.float64)
    set_backend("numpy", execution_policy="execution_ir")
    hop = mps_hop._build_hop("ab,b->a", (matrix,), center.shape, "one_site")
    captured = []
    original = center_boundary._adapter_decision_digest

    def recording(**kwargs):
        captured.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(center_boundary, "_adapter_decision_digest", recording)
    run_mps_krylov(
        hop,
        distributed_execution=_execution_config(),
        center=center,
        center_shape=center.shape,
        site_indices=(2,),
        center_kind="one_site",
        coefficient=-0.125j,
        solver_config={"block_size": 9},
    )

    decision = captured[0]
    assert decision["network"] == "mps"
    assert decision["operation"] == "krylov"
    assert decision["center_kind"] == "one_site"
    assert decision["topology"] == {"site_indices": (2,)}
    assert decision["solver_controls"] == {
        "block_size": 9,
        "coefficient": -0.125j,
    }
    assert decision["selectors"] == {"solver": "krylov"}


def test_mps_adapter_synchronizes_runtime_backend_metadata_mismatch_before_lowering():
    class Collective:
        rank = 0
        size = 2

        def __init__(self):
            self.trace = []

        def allreduce(self, value, *, op="sum"):
            self.trace.append(("allreduce", op, np.asarray(value).dtype.str))
            return np.array(value, copy=True)

        def broadcast(self, *_args, **_kwargs):
            pytest.fail("backend mismatch must precede boundary broadcasts")

    collective = Collective()
    context = DistributedContext(0, 0, 2, 2)
    execution = DistributedExecutionConfig(
        context=context,
        mesh=DeviceMesh((2,), ("rank",), 0),
        collective=collective,
        provider=DeviceResidentProvider(),
        backend_name="cupy",
        backend_device="cuda:0",
        backend_precision=64,
    )
    set_backend("numpy", execution_policy="execution_ir")

    with pytest.raises(RuntimeError, match="distributed backend/runtime validation failed"):
        run_mps_krylov(
            lambda value: value,
            distributed_execution=execution,
            center=np.ones(4),
            center_shape=(4,),
            site_indices=(1,),
            center_kind="one_site",
            coefficient=-0.1,
        )

    assert collective.trace == [
        ("allreduce", "max", np.dtype(np.int32).str),
        ("allreduce", "min", np.dtype(np.uint64).str),
        ("allreduce", "max", np.dtype(np.uint64).str),
    ]


def test_mps_qn_mapped_davidson_matches_dense_allowed_subspace():
    rng = np.random.default_rng(1602)
    matrix = rng.normal(size=(7, 7))
    matrix = matrix + matrix.T
    mask = np.array([True, False, True, True, False, True, True])
    guess = rng.normal(size=np.count_nonzero(mask))
    diagonal = np.diag(matrix)[mask]
    set_backend("numpy", execution_policy="execution_ir")
    hop = mps_hop._build_hop("ab,b->a", (matrix,), mask.shape, "two_site")

    energy, vector, info = run_mps_davidson(
        hop,
        distributed_execution=_execution_config(),
        qn_mask=mask,
        initial_guess=guess,
        diagonal=diagonal,
        site_indices=(2, 3),
        center_kind="two_site",
        solver_config={
            "tol": 1e-12,
            "max_cycle": 100,
            "max_space": 7,
            "lindep": 1e-14,
            "require_convergence": True,
        },
    )

    allowed = matrix[np.ix_(mask, mask)]
    expected = np.linalg.eigvalsh(allowed)[0]
    np.testing.assert_allclose(energy, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(allowed @ vector, energy * vector, atol=1e-10)
    assert info.converged


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
def test_mps_davidson_setup_failures_are_synchronized_before_data_collectives(
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
    mask = np.array([True, False, True, True])
    matrix = np.diag(np.arange(1, 5, dtype=np.float64))
    hop = mps_hop._build_hop("ab,b->a", (matrix,), mask.shape, "two_site")
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
        run_mps_davidson(
            hop,
            distributed_execution=execution,
            qn_mask=qn_mask,
            initial_guess=np.ones(np.count_nonzero(mask)),
            diagonal=np.diag(matrix)[mask],
            site_indices=(1, 2),
            center_kind="two_site",
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
        "artifact_resolution": (
            backend_validation + status * 2 + backend_validation + status
        ),
        "operator_allocation": (
            backend_validation
            + status * 2
            + backend_validation
            + status
            + decision
            + status
        ),
        "guess_extraction": (
            backend_validation
            + status * 2
            + backend_validation
            + status
            + decision
            + status * 2
        ),
        "diagonal_extraction": (
            backend_validation
            + status * 2
            + backend_validation
            + status
            + decision
            + status * 2
        ),
    }[failure_site]
    assert collective.trace == expected


@pytest.mark.parametrize("failure_site", ("center_conversion", "local_extraction"))
def test_mps_krylov_center_setup_failures_are_synchronized_before_solver_data(
    failure_site, monkeypatch
):
    from renormalizer.backend._distributed import center as center_boundary

    class InjectedSetupFailure(BaseException):
        pass

    class FailingCenter:
        def __array__(self, *_args, **_kwargs):
            raise InjectedSetupFailure("injected center conversion failure")

    set_backend("numpy", execution_policy="execution_ir")
    matrix = np.diag(np.arange(1, 5, dtype=np.float64))
    center = np.ones(4)
    hop = mps_hop._build_hop("ab,b->a", (matrix,), center.shape, "one_site")
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
        run_mps_krylov(
            hop,
            distributed_execution=_execution_with_collective(collective),
            center=FailingCenter() if failure_site == "center_conversion" else center,
            center_shape=center.shape,
            site_indices=(1,),
            center_kind="one_site",
            coefficient=-0.01,
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
        expected = (
            backend_validation
            + status
            + backend_validation
            + status
            + backend_validation
            + status * 2
        )
    assert collective.trace == expected


def test_mps_tdvp_workflow_routes_one_and_zero_site_centers(monkeypatch):
    from renormalizer.mps import distributed as adapter
    from renormalizer.utils.configs import EvolveMethod

    state, mpo = _mps_case()
    state.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    state.evolve_config.distributed_execution = _execution_config()
    set_backend("numpy", execution_policy="execution_ir")
    calls = []
    synchronized = []
    original = adapter.run_mps_krylov

    def recording(*args, **kwargs):
        calls.append((kwargs["center_kind"], kwargs["site_indices"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "run_mps_krylov", recording)
    def synchronize(*args, **kwargs):
        synchronized.append((args, kwargs))
        return kwargs["operation"]()

    monkeypatch.setattr(adapter, "_synchronize_mps_state", synchronize)

    state.evolve(mpo, 0.001, normalize=False)

    assert any(kind == "one_site" for kind, _ in calls)
    assert any(kind == "zero_site" for kind, _ in calls)
    assert len(synchronized) >= len(calls)


@pytest.mark.parametrize(
    ("method_name", "adaptive"),
    [("prop_and_compress", False), ("tdvp_ps", True)],
)
def test_mps_public_evolve_coordinates_method_and_adaptive_before_dispatch(
    method_name, adaptive, monkeypatch
):
    from renormalizer.mps import distributed as adapter
    from renormalizer.utils.configs import EvolveMethod

    class StopAtDecision(BaseException):
        pass

    state, mpo = _mps_case()
    method = getattr(EvolveMethod, method_name)
    state.evolve_config = EvolveConfig(method, adaptive=adaptive)
    state.evolve_config.distributed_execution = _execution_config()
    captured = []
    dispatched = []

    def coordinate(*args, **kwargs):
        captured.append(kwargs)
        raise StopAtDecision

    def dispatch(*args, **kwargs):
        dispatched.append((args, kwargs))
        pytest.fail("public evolution dispatched before coordination")

    monkeypatch.setattr(adapter, "coordinate_mps_workflow_entry", coordinate)
    monkeypatch.setattr(
        type(state),
        {
            EvolveMethod.prop_and_compress: "_evolve_prop_and_compress",
            EvolveMethod.tdvp_ps: "_evolve_tdvp_ps",
        }[method],
        dispatch,
    )

    with pytest.raises(StopAtDecision):
        state.evolve(mpo, 0.0125, normalize=False)

    assert dispatched == []
    assert captured[0]["operation"] == "public_evolve"
    assert captured[0]["supported"] is False
    assert captured[0]["selectors"] == {
        "method": method_name,
        "normalize": False,
    }
    controls = captured[0]["solver_controls"]
    assert controls["adaptive"] is adaptive
    assert controls["guess_dt"] == state.evolve_config.guess_dt
    assert controls["adaptive_rtol"] == state.evolve_config.adaptive_rtol
    assert controls["ivp_solver"] == state.evolve_config.ivp_solver
    assert controls["ivp_rtol"] == state.evolve_config.ivp_rtol
    assert controls["ivp_atol"] == state.evolve_config.ivp_atol
    assert controls["evolve_dt"] == 0.0125
    assert controls["compression"]["criteria"] == "threshold"


def test_mps_public_unsupported_evolve_rejects_before_legacy_workflow(monkeypatch):
    from renormalizer.mps import distributed as adapter
    from renormalizer.utils.configs import EvolveMethod

    state, mpo = _mps_case()
    collective = _StateTraceCollective()
    execution = _execution_with_collective(collective)
    state.evolve_config = EvolveConfig(EvolveMethod.prop_and_compress)
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
    monkeypatch.setattr(
        type(state), "_evolve_prop_and_compress",
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
        state.evolve(mpo, 0.01, normalize=True)

    assert calls == {"copy": 0, "dispatch": 0, "normalize": 0}
    assert collective.trace == [
        ("max", np.dtype(np.int32).str),
        ("min", np.dtype(np.uint64).str),
        ("max", np.dtype(np.uint64).str),
    ] * 2
    assert not hasattr(adapter, "run_mps_public_evolve_fallback")


@pytest.mark.parametrize("fail_normalization", [False, True])
def test_mps_public_normalization_is_a_synchronized_state_phase(
    fail_normalization, monkeypatch
):
    from renormalizer.mps import distributed as adapter
    from renormalizer.utils.configs import EvolveMethod

    class InjectedNormalizationFailure(BaseException):
        pass

    state, mpo = _mps_case()
    collective = _StateTraceCollective()
    state.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    state.evolve_config.distributed_execution = _execution_with_collective(collective)
    monkeypatch.setattr(
        adapter, "coordinate_mps_workflow_entry", lambda *args, **kwargs: "supported"
    )
    monkeypatch.setattr(
        type(state), "_evolve_tdvp_ps", lambda current, *_args: current.copy()
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
            state.evolve(mpo, 0.01, normalize=True)
        assert collective.trace == [("max", np.dtype(np.int32).str)]
    else:
        state.evolve(mpo, 0.01, normalize=True)
        assert collective.trace == [
            ("max", np.dtype(np.int32).str),
            ("min", np.dtype(np.uint64).str),
            ("max", np.dtype(np.uint64).str),
        ]


def test_mps_non_krylov_tdvp_routes_every_center_to_synchronized_fallback(monkeypatch):
    from renormalizer.mps import distributed as adapter
    from renormalizer.utils.configs import EvolveMethod

    state, mpo = _mps_case()
    state.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, ivp_solver="RK23")
    state.evolve_config.distributed_execution = _execution_config()
    set_backend(
        "numpy", execution_policy="execution_ir", fallback_policy="legacy_oe"
    )
    calls = []

    def fallback(_operation, **kwargs):
        calls.append((kwargs["center_kind"], kwargs["solver"]))
        return np.asarray(kwargs["center"]).copy(), 1

    monkeypatch.setattr(adapter, "run_mps_ivp_fallback", fallback)
    state.evolve(mpo, 0.001, normalize=False)

    assert calls
    assert {kind for kind, _ in calls} == {"one_site", "zero_site"}
    assert {solver for _, solver in calls} == {"RK23"}


def test_mps_ivp_entry_decision_precedes_copy_environment_and_expression(monkeypatch):
    import renormalizer.mps.mps as mps_module
    from renormalizer.mps import distributed as adapter
    from renormalizer.utils.configs import EvolveMethod

    state, mpo = _mps_case()
    state.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps, ivp_solver="RK23")
    state.evolve_config.distributed_execution = _execution_config()
    set_backend("numpy", execution_policy="execution_ir", fallback_policy="error")
    trace = []
    original_decision = adapter.coordinate_adapter_decision
    original_copy = type(state).copy
    original_environment = mps_module.Environ
    original_expression = mps_module.hop_expr

    def decision(*args, **kwargs):
        trace.append("decision")
        return original_decision(*args, **kwargs)

    def copy(current, *args, **kwargs):
        trace.append("copy")
        return original_copy(current, *args, **kwargs)

    def environment(*args, **kwargs):
        trace.append("environment")
        return original_environment(*args, **kwargs)

    def expression(*args, **kwargs):
        trace.append("expression")
        return original_expression(*args, **kwargs)

    monkeypatch.setattr(adapter, "coordinate_adapter_decision", decision)
    monkeypatch.setattr(type(state), "copy", copy)
    monkeypatch.setattr(mps_module, "Environ", environment)
    monkeypatch.setattr(mps_module, "hop_expr", expression)

    with pytest.raises(NotImplementedError, match="distributed adapter decision failed"):
        state.evolve(mpo, 0.001, normalize=False)

    assert trace == ["decision", "decision"]


@pytest.mark.parametrize("method", ["1site", "2site"])
def test_mps_ground_state_routes_small_centers_to_distributed_davidson(
    method, monkeypatch
):
    from renormalizer import optimize_mps
    from renormalizer.mps import distributed as adapter

    state, mpo = _mps_case()
    state.optimize_config.procedure = [[4, 0], [4, 0]]
    state.optimize_config.method = method
    state.optimize_config.algo = "davidson"
    state.optimize_config.distributed_execution = _execution_config()
    set_backend("numpy", execution_policy="execution_ir")
    calls = []
    synchronized = []
    original = adapter.run_mps_davidson

    def recording(*args, **kwargs):
        calls.append(
            (
                kwargs["center_kind"],
                kwargs["site_indices"],
                kwargs["decision_selectors"],
            )
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "run_mps_davidson", recording)
    def synchronize(*args, **kwargs):
        synchronized.append((args, kwargs))
        return kwargs["operation"]()

    monkeypatch.setattr(adapter, "_synchronize_mps_state", synchronize)

    energies, _ = optimize_mps(state, mpo)

    assert energies
    assert calls
    assert {kind for kind, _, _ in calls} == {
        "one_site" if method == "1site" else "two_site"
    }
    assert all(
        selectors
        == {
            "algo": "davidson",
            "nroots": 1,
            "omega": None,
            "stacked_mpo": False,
            "method": method,
        }
        for _, _, selectors in calls
    )
    assert len(synchronized) >= len(calls)


def test_mps_ground_mode_support_classifies_every_local_only_selector():
    from renormalizer.mps.distributed import _mps_ground_mode_supported

    assert _mps_ground_mode_supported(
        algo="davidson", nroots=1, omega=None, stacked_mpo=False
    )
    assert not _mps_ground_mode_supported(
        algo="direct", nroots=1, omega=None, stacked_mpo=False
    )
    assert not _mps_ground_mode_supported(
        algo="primme", nroots=1, omega=None, stacked_mpo=False
    )
    assert not _mps_ground_mode_supported(
        algo="davidson", nroots=2, omega=None, stacked_mpo=False
    )
    assert not _mps_ground_mode_supported(
        algo="davidson", nroots=1, omega=0.5, stacked_mpo=False
    )
    assert not _mps_ground_mode_supported(
        algo="davidson", nroots=1, omega=None, stacked_mpo=True
    )


def test_mps_direct_ground_workflow_routes_through_synchronized_fallback(monkeypatch):
    from renormalizer import optimize_mps
    from renormalizer.mps import distributed as adapter

    state, mpo = _mps_case()
    state.optimize_config.procedure = [[4, 0], [4, 0]]
    state.optimize_config.method = "1site"
    state.optimize_config.algo = "direct"
    state.optimize_config.nroots = 1
    state.optimize_config.distributed_execution = _execution_config()
    set_backend(
        "numpy", execution_policy="execution_ir", fallback_policy="legacy_oe"
    )
    calls = []
    original = adapter.run_mps_ground_state_fallback

    def recording(*args, **kwargs):
        calls.append((kwargs["solver_controls"], kwargs["selectors"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "run_mps_ground_state_fallback", recording)
    energies, _ = optimize_mps(state, mpo)

    assert energies
    assert calls
    assert all(controls["algo"] == "direct" for controls, _ in calls)
    assert all(selectors["stacked_mpo"] is False for _, selectors in calls)


def test_mps_unsupported_iterative_error_precedes_hamiltonian_construction(monkeypatch):
    from renormalizer import optimize_mps
    from renormalizer.mps import gs

    state, mpo = _mps_case()
    state.optimize_config.procedure = [[4, 0], [4, 0]]
    state.optimize_config.method = "1site"
    state.optimize_config.algo = "primme"
    state.optimize_config.nroots = 1
    state.optimize_config.distributed_execution = _execution_config()
    set_backend("numpy", execution_policy="execution_ir", fallback_policy="error")
    calls = []
    original = gs.get_ham_iterative

    def recording(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(gs, "get_ham_iterative", recording)
    with pytest.raises(NotImplementedError, match="distributed adapter decision failed"):
        optimize_mps(state, mpo)

    assert calls == []


def test_mps_ground_entry_decision_precedes_canonicalization_and_environment(monkeypatch):
    from renormalizer import optimize_mps
    from renormalizer.mps import distributed as adapter
    from renormalizer.mps import gs

    state, mpo = _mps_case()
    state.optimize_config.procedure = [[4, 0], [4, 0]]
    state.optimize_config.method = "1site"
    state.optimize_config.algo = "direct"
    state.optimize_config.nroots = 1
    state.optimize_config.distributed_execution = _execution_config()
    set_backend("numpy", execution_policy="execution_ir", fallback_policy="error")
    trace = []
    original_decision = adapter.coordinate_adapter_decision
    original_right = type(state).ensure_right_canonical
    original_left = type(state).ensure_left_canonical
    original_environment = gs.Environ

    def decision(*args, **kwargs):
        trace.append("decision")
        return original_decision(*args, **kwargs)

    def right(current, *args, **kwargs):
        trace.append("canonicalization")
        return original_right(current, *args, **kwargs)

    def left(current, *args, **kwargs):
        trace.append("canonicalization")
        return original_left(current, *args, **kwargs)

    def environment(*args, **kwargs):
        trace.append("environment")
        return original_environment(*args, **kwargs)

    monkeypatch.setattr(adapter, "coordinate_adapter_decision", decision)
    monkeypatch.setattr(type(state), "ensure_right_canonical", right)
    monkeypatch.setattr(type(state), "ensure_left_canonical", left)
    monkeypatch.setattr(gs, "Environ", environment)

    with pytest.raises(NotImplementedError, match="distributed adapter decision failed"):
        optimize_mps(state, mpo)

    assert trace == ["decision"]


def test_mps_ground_entry_coordinates_effective_procedure_and_convergence(
    monkeypatch,
):
    from renormalizer import optimize_mps
    from renormalizer.mps import distributed as adapter
    from renormalizer.utils import CompressConfig, CompressCriteria

    class StopAtDecision(BaseException):
        pass

    state, mpo = _mps_case()
    compression = CompressConfig(
        criteria=CompressCriteria.both,
        threshold=2e-4,
        max_bonddim=7,
    )
    compression.max_dims = np.array([1, 4, 6, 7, 1])
    state.optimize_config.procedure = [[compression, 0.25], [5, 0.0]]
    state.optimize_config.e_rtol = 3e-9
    state.optimize_config.e_atol = 4e-11
    state.optimize_config.distributed_execution = _execution_config()
    captured = []

    def coordinate(*args, **kwargs):
        captured.append(kwargs)
        raise StopAtDecision

    monkeypatch.setattr(adapter, "coordinate_mps_workflow_entry", coordinate)

    with pytest.raises(StopAtDecision):
        optimize_mps(state, mpo)

    controls = captured[0]["solver_controls"]
    assert controls["convergence"] == {"rtol": 3e-9, "atol": 4e-11}
    assert controls["effective_procedure"] == [
        {
            "compression": {
                "criteria": "both",
                "threshold": 2e-4,
                "bond_dim_max_value": 7,
                "max_dims": [1, 4, 6, 7, 1],
                "ofs": None,
                "ofs_swap_jw": False,
            },
            "percent": 0.25,
        },
        {
            "compression": {
                "criteria": "fixed",
                "threshold": 1e-3,
                "bond_dim_max_value": 5,
                "max_dims": None,
                "ofs": None,
                "ofs_swap_jw": False,
            },
            "percent": 0.0,
        },
    ]


@pytest.mark.parametrize("failure_site", ["svd", "environment"])
def test_mps_complete_center_transition_is_one_baseexception_state_phase(
    failure_site, monkeypatch
):
    import renormalizer.mps.mps as mps_module
    from renormalizer.utils.configs import EvolveMethod

    class InjectedTransitionFailure(BaseException):
        pass

    state, mpo = _mps_case()
    state.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    state.evolve_config.distributed_execution = _execution_config()
    set_backend("numpy", execution_policy="execution_ir")

    if failure_site == "svd":
        monkeypatch.setattr(
            mps_module.svd_qn,
            "svd_qn",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                InjectedTransitionFailure("injected SVD failure")
            ),
        )
    else:
        original = mps_module.Environ.GetLR

        def fail_environment(self, *args, **kwargs):
            if kwargs.get("itensor") is not None:
                raise InjectedTransitionFailure("injected environment update failure")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(mps_module.Environ, "GetLR", fail_environment)

    with pytest.raises(RuntimeError, match="distributed state update failed"):
        state.evolve(mpo, 0.001, normalize=False)


def test_mps_unsupported_hop_uses_one_root_fallback_and_numeric_metadata():
    from scipy.linalg import expm

    rng = np.random.default_rng(1604)
    matrix = rng.normal(size=(5, 5))
    matrix = matrix + matrix.T
    center = rng.normal(size=5)
    set_backend(
        "numpy", execution_policy="execution_ir", fallback_policy="legacy_oe"
    )
    counters = {}

    actual, iterations = run_mps_krylov(
        lambda value: matrix @ value,
        distributed_execution=_execution_config(),
        center=center,
        center_shape=center.shape,
        site_indices=(1,),
        center_kind="one_site",
        coefficient=-0.02,
        counters=counters,
    )

    np.testing.assert_allclose(actual, expm(-0.02 * matrix) @ center, atol=1e-13)
    assert iterations > 0
    assert counters["fallback_count"] == 1
    assert counters["root_operation_count"] == 1
    assert counters["allgather_calls"] == 0


def test_mps_unsupported_ivp_mode_uses_one_synchronized_root_operation():
    set_backend(
        "numpy", execution_policy="execution_ir", fallback_policy="legacy_oe"
    )
    calls = []
    counters = {}

    result, evaluations = run_mps_ivp_fallback(
        lambda: (calls.append("root") or np.arange(6, dtype=np.complex128), 17),
        distributed_execution=_execution_config(),
        center=np.ones(6, dtype=np.complex128),
        center_shape=(2, 3),
        site_indices=(1, 2),
        center_kind="two_site",
        solver="RK45",
        solver_controls={"rtol": 1e-6, "atol": 1e-9},
        counters=counters,
    )

    np.testing.assert_array_equal(result, np.arange(6, dtype=np.complex128))
    assert evaluations == 17
    assert calls == ["root"]
    assert counters["fallback_count"] == 1
    assert counters["root_operation_count"] == 1


def test_mps_unsupported_ivp_error_policy_is_synchronized_before_root_work():
    set_backend("numpy", execution_policy="execution_ir", fallback_policy="error")
    calls = []

    with pytest.raises(NotImplementedError, match="distributed adapter decision failed"):
        run_mps_ivp_fallback(
            lambda: (calls.append("root") or np.ones(4), 3),
            distributed_execution=_execution_config(),
            center=np.ones(4),
            center_shape=(4,),
            site_indices=(1,),
            center_kind="one_site",
            solver="RK23",
            solver_controls={"rtol": 1e-5, "atol": 1e-8},
        )

    assert calls == []


def test_mps_unsupported_ground_mode_fallback_preserves_roots_and_vectors():
    set_backend(
        "numpy", execution_policy="execution_ir", fallback_policy="legacy_oe"
    )
    calls = []
    guesses = [np.arange(3, dtype=np.float64), np.arange(3, dtype=np.float64) + 1]

    energies, vectors = run_mps_ground_state_fallback(
        lambda: (
            calls.append("root") or np.array([-2.0, -1.0]),
            [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])],
        ),
        distributed_execution=_execution_config(),
        qn_mask=np.array([True, False, True, True]),
        initial_guesses=guesses,
        site_indices=(1, 2),
        center_kind="two_site",
        solver_controls={"algo": "primme", "nroots": 2},
        selectors={"stacked_mpo": False, "omega": None},
    )

    np.testing.assert_array_equal(energies, [-2.0, -1.0])
    np.testing.assert_array_equal(vectors[0], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(vectors[1], [0.0, 1.0, 0.0])
    assert calls == ["root"]


def test_mps_root_fallback_emits_synchronized_telemetry(monkeypatch):
    from renormalizer.utils import profiling

    matrix = np.diag(np.arange(1, 5, dtype=np.float64))
    center = np.arange(1, 5, dtype=np.float64)
    set_backend(
        "numpy", execution_policy="execution_ir", fallback_policy="legacy_oe"
    )
    events = []
    counters = {}
    monkeypatch.setattr(profiling, "enabled", lambda: True)
    monkeypatch.setattr(
        profiling,
        "record",
        lambda event, **payload: events.append({"event": event, **payload}),
    )

    run_mps_krylov(
        lambda value: matrix @ value,
        distributed_execution=_execution_config(),
        center=center,
        center_shape=center.shape,
        site_indices=(1,),
        center_kind="one_site",
        coefficient=-0.02,
        counters=counters,
    )

    summary = next(
        event for event in events if event["event"] == "distributed_solve_summary"
    )
    assert summary["fallback_count"] == 1
    assert summary["root_operation_count"] == 1
    assert summary["allgather_calls"] == 0
    assert summary["broadcast_calls"] == 2
    assert summary["allreduce_calls"] == 18
    assert summary["capacity_proven"] is False
    assert summary["capacity_reason"] == "unbounded_device_resident_root_fallback"
    assert summary["compute_s"] > 0.0
    fallback_phase = next(
        event
        for event in events
        if event.get("event") == "phase_summary"
        and event.get("phase") == "root_fallback"
    )
    assert fallback_phase["network"] == "mps"
    assert fallback_phase["operation"] == "root_solver_and_broadcast"
    assert fallback_phase["wall_s"] > 0.0


def test_mps_krylov_root_fallback_enforces_maximum_vector_count():
    matrix = np.diag(np.arange(1.0, 7.0))
    matrix += np.diag(np.full(5, -0.3), 1)
    matrix += np.diag(np.full(5, -0.3), -1)
    center = np.arange(1.0, 7.0)
    set_backend(
        "numpy", execution_policy="execution_ir", fallback_policy="legacy_oe"
    )

    actual, iterations = run_mps_krylov(
        lambda value: matrix @ value,
        distributed_execution=_execution_config(),
        center=center,
        center_shape=center.shape,
        site_indices=(1,),
        center_kind="one_site",
        coefficient=-0.02,
        solver_config={"block_size": 2, "max_krylov_vectors": 2},
    )
    expected, expected_iterations = expm_krylov(
        matrix.dot,
        -0.02,
        center,
        block_size=2,
        max_krylov_vectors=2,
    )

    assert iterations == expected_iterations == 2
    np.testing.assert_allclose(actual, expected)


def test_mps_root_fallback_preserves_promoted_krylov_and_davidson_results():
    matrix = np.diag(np.asarray([1.0, 2.0], dtype=np.float32))
    center = np.asarray([1.0, 0.0], dtype=np.float32)
    mask = np.ones(2, dtype=bool)
    set_backend(
        "numpy", execution_policy="execution_ir", fallback_policy="legacy_oe"
    )

    krylov, _ = run_mps_krylov(
        lambda value: matrix @ value,
        distributed_execution=_execution_config(),
        center=center,
        center_shape=center.shape,
        site_indices=(1,),
        center_kind="one_site",
        coefficient=-0.125j,
        solver_config={"block_size": 2, "max_krylov_vectors": 2},
    )
    _, davidson, _ = run_mps_davidson(
        lambda value: matrix @ value,
        distributed_execution=_execution_config(),
        qn_mask=mask,
        initial_guess=center,
        diagonal=np.diag(matrix).copy(),
        site_indices=(1, 2),
        center_kind="two_site",
        solver_config={"max_cycle": 2, "max_space": 2},
    )

    assert krylov.dtype == np.dtype("complex128")
    assert davidson.dtype == np.dtype("float64")


def test_mps_error_policy_rejects_unsupported_hop_before_local_hv():
    set_backend("numpy", execution_policy="execution_ir", fallback_policy="error")
    calls = []

    with pytest.raises(NotImplementedError, match="mapped adapter decision failed"):
        run_mps_krylov(
            lambda value: calls.append(1) or value,
            distributed_execution=_execution_config(),
            center=np.ones(4),
            center_shape=(4,),
            site_indices=(1,),
            center_kind="one_site",
            coefficient=-0.02,
        )

    assert calls == []


def test_mps_adapter_synchronizes_build_failure_before_data_collectives():
    class RemoteBuildFailure(SingleProcessCollective):
        def __init__(self):
            self.max_calls = 0

        def allreduce(self, array, *, op="sum"):
            result = np.array(array, copy=True)
            if op == "max":
                self.max_calls += 1
                # Artifact resolution follows both backend checks and center conversion.
                if self.max_calls == 6:
                    result[...] = 1
            return result

        def broadcast(self, array, *, root):
            raise AssertionError("data collective entered after build failure")

    context = DistributedContext(0, 0, 1, 1)
    execution = DistributedExecutionConfig(
        context=context,
        mesh=DeviceMesh((1,), ("rank",), 0),
        collective=RemoteBuildFailure(),
        provider=DeviceResidentProvider(),
    )
    matrix = np.eye(4)
    center = np.ones(4)
    set_backend("numpy", execution_policy="execution_ir")
    hop = mps_hop._build_hop("ab,b->a", (matrix,), center.shape, "one_site")

    with pytest.raises(RuntimeError, match="artifact resolution setup failed"):
        run_mps_krylov(
            hop,
            distributed_execution=execution,
            center=center,
            center_shape=center.shape,
            site_indices=(1,),
            center_kind="one_site",
            coefficient=-0.02,
        )


def test_mps_unsupported_davidson_uses_one_root_fallback():
    rng = np.random.default_rng(1605)
    matrix = rng.normal(size=(6, 6))
    matrix = matrix + matrix.T
    mask = np.array([True, False, True, True, False, True])
    guess = rng.normal(size=mask.sum())
    set_backend(
        "numpy", execution_policy="execution_ir", fallback_policy="legacy_oe"
    )
    counters = {}

    energy, vector, info = run_mps_davidson(
        lambda value: matrix @ value,
        distributed_execution=_execution_config(),
        qn_mask=mask,
        initial_guess=guess,
        diagonal=np.diag(matrix)[mask],
        site_indices=(1, 2),
        center_kind="two_site",
        solver_config={"tol": 1e-10, "max_cycle": 100, "max_space": 8},
        counters=counters,
    )

    allowed = matrix[np.ix_(mask, mask)]
    np.testing.assert_allclose(energy, np.linalg.eigvalsh(allowed)[0], atol=1e-10)
    np.testing.assert_allclose(allowed @ vector, energy * vector, atol=1e-8)
    assert info.h_v_count > 0
    assert counters["root_operation_count"] == 1


def test_fallback_metadata_is_copied_through_backend_array_boundary():
    from renormalizer.mps.distributed import _copy_metadata

    class Selected:
        def __init__(self):
            self.calls = []

        def asarray(self, values, dtype=None):
            self.calls.append((values, dtype))
            return np.asarray(values, dtype=dtype)

    selected = Selected()
    target = np.zeros(2, dtype=np.float64)

    _copy_metadata(target, (1.5, 2.5), selected)

    np.testing.assert_array_equal(target, [1.5, 2.5])
    assert selected.calls == [((1.5, 2.5), np.dtype("float64"))]


def test_mps_update_digest_detects_rank_disagreement():
    from renormalizer.mps.distributed import synchronize_mps_update

    class DisagreeingCollective:
        rank = 0
        size = 2

        def __init__(self):
            self.calls = 0

        def allreduce(self, value, *, op="sum"):
            self.calls += 1
            result = np.array(value, copy=True)
            if op == "max" and result.dtype == np.uint64:
                result[0] += 1
            return result

    context = DistributedContext(0, 0, 2, 2)
    config = DistributedExecutionConfig(
        context=context,
        mesh=DeviceMesh((2,), ("rank",), 0),
        collective=DisagreeingCollective(),
        provider=DeviceResidentProvider(),
    )

    with pytest.raises(RuntimeError, match="state digest disagreement"):
        synchronize_mps_update(
            config,
            [np.arange(4, dtype=np.float64)],
            qn_arrays=[np.array([[0], [1]])],
            metadata=("right", 2, 1e-7),
        )


def test_distributed_mps_update_uses_stable_auxiliary_rng_without_side_effects():
    from renormalizer.mps.distributed import _deterministic_mps_update

    class State:
        def _update_mps(self, *args, **kwargs):
            return np.random.random(4)

    np.random.seed(11)
    before = np.random.get_state()
    first = _deterministic_mps_update(State(), metadata=(0, True, (3, 4)))
    after = np.random.get_state()
    np.random.seed(97)
    second = _deterministic_mps_update(State(), metadata=(0, True, (3, 4)))

    np.testing.assert_array_equal(first, second)
    np.random.set_state(before)
    expected_next = np.random.random()
    np.random.set_state(after)
    assert np.random.random() == expected_next


def test_distributed_result_snapshot_uses_deterministic_update_wrapper(monkeypatch):
    from renormalizer import optimize_mps
    from renormalizer.mps import distributed as adapter

    state, mpo = _mps_case()
    state.optimize_config.procedure = [[4, 0], [4, 0]]
    state.optimize_config.method = "1site"
    state.optimize_config.algo = "davidson"
    state.optimize_config.distributed_execution = _execution_config()
    set_backend("numpy", execution_policy="execution_ir")
    metadata = []
    original = adapter._deterministic_mps_update

    def recording(*args, **kwargs):
        metadata.append(tuple(kwargs["metadata"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "_deterministic_mps_update", recording)
    optimize_mps(state, mpo)

    assert any(values[-1] == "result_snapshot" for values in metadata)
    assert any(values[-1] == "active_state" for values in metadata)


def test_mps_supported_solve_emits_bounded_distributed_telemetry(monkeypatch):
    from renormalizer.utils import profiling

    matrix = np.diag(np.arange(1, 5, dtype=np.float64))
    center = np.arange(1, 5, dtype=np.float64)
    collective = _StateTraceCollective()
    set_backend("numpy", execution_policy="execution_ir")
    hop = mps_hop._build_hop("ab,b->a", (matrix,), center.shape, "one_site")
    events = []
    monkeypatch.setattr(profiling, "enabled", lambda: True)
    monkeypatch.setattr(
        profiling,
        "record",
        lambda event, **payload: events.append({"event": event, **payload}),
    )

    run_mps_krylov(
        hop,
        distributed_execution=_execution_with_collective(collective),
        center=center,
        center_shape=center.shape,
        site_indices=(1,),
        center_kind="one_site",
        coefficient=-0.02,
        solver_config={"block_size": 4},
    )

    summary = next(
        event for event in events if event["event"] == "distributed_solve_summary"
    )
    assert summary["network"] == "mps"
    assert summary["center_kind"] == "one_site"
    assert summary["solver"] == "krylov"
    assert summary["allgather_calls"] == 0
    assert summary["allreduce_calls"] == len(collective.trace) - 3
    assert summary["boundary_materialization_broadcasts"] == 1
    assert summary["packed_count"] == 4
    assert summary["solver_residual_norm"] == 0.0
    assert summary["compute_s"] > 0.0
    assert summary["synchronization_s"] > 0.0
    phases = {
        event["phase"]: event
        for event in events
        if event.get("event") == "phase_summary"
    }
    assert phases["boundary_materialization"]["operation"] == "ordered_broadcast"
    assert phases["boundary_materialization"]["network"] == "mps"
    assert phases["boundary_materialization"]["wall_s"] > 0.0
    assert phases["state_update"]["operation"] == "synchronized_state_transition"
    assert phases["state_update"]["wall_s"] > 0.0


def test_mps_job_parser_exposes_formal_stage4_controls():
    from renormalizer.mps.distributed_job import _site_method, build_parser

    args = build_parser().parse_args(
        [
            "--mode", "both",
            "--output-dir", "/tmp/mps-stage4",
            "--expected-world-size", "4",
            "--model-size", "8",
            "--bond-dimension", "16",
            "--seed", "2019",
            "--precision", "64",
            "--shard-solver-vectors",
            "--fallback-policy", "error",
            "--tdvp-sites", "one",
            "--davidson-sites", "two",
        ]
    )

    assert args.mode == "both"
    assert args.expected_world_size == 4
    assert args.shard_solver_vectors is True
    assert args.tdvp_sites == "one"
    assert args.davidson_sites == "two"
    assert _site_method("one") == "1site"
    assert _site_method("two") == "2site"


def test_formal_job_allgather_trap_delegates_only_allowed_collectives():
    from renormalizer.mps.distributed_job import _arm_allgather_trap

    base = SingleProcessCollective()
    trapped = _arm_allgather_trap(base)
    value = np.arange(4, dtype=np.float64)

    np.testing.assert_array_equal(trapped.allreduce(value), value)
    assert trapped.rank == base.rank
    assert trapped.size == base.size
    with pytest.raises(RuntimeError, match="complete-vector allgather is forbidden"):
        trapped.allgather(value, axis=0)


def test_formal_job_solver_controls_override_workflow_defaults():
    from renormalizer.mps.distributed import _resolve_solver_config
    from renormalizer.mps.distributed_job import _trapped_execution_config

    execution = _trapped_execution_config(
        _execution_config(),
        solver_options={
            "krylov": {"block_size": 17},
            "davidson": {"tol": 2e-8, "max_space": 19},
        },
    )

    assert _resolve_solver_config(
        execution, "krylov", {"block_size": 50}
    ) == {"block_size": 17}
    assert _resolve_solver_config(
        execution, "davidson", {"tol": 1e-12, "max_space": 12}
    ) == {"tol": 2e-8, "max_space": 19}


@pytest.mark.parametrize(
    "field",
    ("atol", "rtol", "rank_atol", "davidson_tol", "davidson_lindep"),
)
@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf"), -1.0))
def test_formal_job_tolerance_setup_rejects_invalid_values_synchronously(
    field, value
):
    from renormalizer.backend._distributed.job import _validate_job_tolerances

    class Collective:
        rank = 0
        size = 2

        def __init__(self):
            self.trace = []

        def allreduce(self, array, *, op="sum"):
            self.trace.append((op, np.asarray(array).dtype.str))
            return np.array(array, copy=True)

    values = {
        "atol": 1e-6,
        "rtol": 1e-6,
        "rank_atol": 0.0,
        "davidson_tol": 1e-6,
        "davidson_lindep": 1e-14,
    }
    values[field] = value
    runtime = SimpleNamespace(rank=0, collective=Collective())

    with pytest.raises(RuntimeError, match="formal tolerance setup failed") as raised:
        _validate_job_tolerances(runtime, SimpleNamespace(**values))

    assert isinstance(raised.value.__cause__, ValueError)
    assert field in str(raised.value.__cause__)
    assert runtime.collective.trace == [("max", np.dtype(np.int32).str)]


def test_formal_job_tolerance_setup_accepts_finite_nonnegative_values():
    from renormalizer.backend._distributed.job import _validate_job_tolerances

    class Collective:
        rank = 0
        size = 2

        def __init__(self):
            self.trace = []

        def allreduce(self, array, *, op="sum"):
            self.trace.append((op, np.asarray(array).dtype.str))
            return np.array(array, copy=True)

    runtime = SimpleNamespace(rank=0, collective=Collective())
    args = SimpleNamespace(
        atol=0.0,
        rtol=0.0,
        rank_atol=0.0,
        davidson_tol=0.0,
        davidson_lindep=0.0,
    )

    assert _validate_job_tolerances(runtime, args) is None
    assert runtime.collective.trace == [("max", np.dtype(np.int32).str)]


def test_both_formal_jobs_validate_tolerances_before_backend_configuration():
    from renormalizer.mps import distributed_job as mps_job
    from renormalizer.tn import distributed_job as ttns_job

    for module in (mps_job, ttns_job):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert source.index("_validate_job_tolerances(runtime, args)") < source.index(
            '"backend configuration"'
        )


def test_job_cli_helpers_have_one_unsigned_topology_neutral_backend_owner():
    from renormalizer.backend._distributed import center
    from renormalizer.backend._distributed import job as shared_job
    from renormalizer.mps import distributed as mps_adapter
    from renormalizer.mps import distributed_job as mps_job
    from renormalizer.tn import distributed as ttns_adapter
    from renormalizer.tn import distributed_job as ttns_job

    shared_names = [
        "RANK_SCHEMA",
        "_add_common_arguments",
        "_aggregate_norm_error",
        "_arm_allgather_trap",
        "_finalize_profile",
        "_numerical_passed",
        "_prepare_output",
        "_profile_distribution",
        "_run_root_reference",
        "_summary_distribution",
        "_summary_numerical",
        "_trapped_execution_config",
        "_validate_job_tolerances",
        "_validate_rank_records",
        "_write_json",
    ]
    for name in shared_names:
        assert getattr(mps_job, name) is getattr(shared_job, name)
        assert getattr(ttns_job, name) is getattr(shared_job, name)

    shared_source = Path(shared_job.__file__).read_text(encoding="utf-8")
    ttns_source = Path(ttns_job.__file__).read_text(encoding="utf-8")
    assert "# Author:" not in shared_source
    assert "renormalizer.mps" not in shared_source
    assert "renormalizer.tn" not in shared_source
    assert "renormalizer.mps.distributed_job" not in ttns_source
    assert not hasattr(center, "run_adapter_root_state_fallback")
    assert not hasattr(mps_adapter, "run_mps_public_evolve_fallback")
    assert not hasattr(ttns_adapter, "run_ttns_public_evolve_fallback")


def test_root_gate_rejects_malformed_or_inconsistent_rank_evidence():
    from renormalizer.mps.distributed_job import RANK_SCHEMA, _validate_rank_records

    base = {
        "schema": RANK_SCHEMA,
        "run_id": "run",
        "world_size": 2,
        "git_commit": "abc",
        "state_hash": "same",
        "solver": {"tol": 1e-6},
        "distribution": {
            "h_v_count": 10,
            "broadcast_calls": 8,
            "allreduce_calls": 12,
            "allgather_calls": 0,
            "boundary_materialization_broadcasts": 2,
            "allgather_trap_armed": True,
            "strictly_sharded": True,
            "supported_event_count": 1,
            "supported_plan_hashes": ["plan"],
            "supported_placement_hashes": ["placement"],
            "plan_hashes": ["plan"],
            "placement_hashes": ["placement"],
            "max_solver_residual_norm": 0.0,
        },
        "fallback": {"count": 0, "root_operation_count": 0},
    }
    records = [dict(base, rank=0, local_rank=0), dict(base, rank=1, local_rank=1)]

    assert _validate_rank_records(records, records[0], 2) == []
    records[1] = dict(records[1], schema="wrong")
    assert "rank record schema mismatch" in _validate_rank_records(
        records, records[0], 2
    )


def test_root_gate_validates_boundary_materialization_count_agreement():
    from renormalizer.mps.distributed_job import _validate_rank_records

    records = [
        _job_rank_record(0, scalar=-2.0, fallback_count=0),
        _job_rank_record(1, scalar=-2.0, fallback_count=0),
    ]
    records[1]["distribution"]["boundary_materialization_broadcasts"] = 3

    reasons = _validate_rank_records(records, records[0], 2)

    assert "distributed counter or plan mismatch" in reasons


@pytest.mark.parametrize(
    ("supported_key", "all_key", "expected_reason"),
    [
        (
            "supported_plan_hashes",
            "plan_hashes",
            "supported distributed plan hash evidence missing",
        ),
        (
            "supported_placement_hashes",
            "placement_hashes",
            "supported distributed placement hash evidence missing",
        ),
    ],
)
@pytest.mark.parametrize("proof_kind", ("missing_supported", "fallback_hash"))
def test_root_gate_requires_genuine_supported_plan_and_placement_hashes(
    supported_key, all_key, expected_reason, proof_kind
):
    from renormalizer.mps.distributed_job import _validate_rank_records

    records = [
        _job_rank_record(0, scalar=-2.0, fallback_count=0),
        _job_rank_record(1, scalar=-2.0, fallback_count=0),
    ]
    for record in records:
        if proof_kind == "missing_supported":
            record["distribution"][supported_key] = []
        else:
            record["distribution"][all_key] = ["fallback"]

    reasons = _validate_rank_records(records, records[0], 2)

    assert expected_reason in reasons


def test_root_gate_requires_rank_zero_to_own_every_root_operation():
    from renormalizer.mps.distributed_job import _validate_rank_records

    records = [
        _job_rank_record(0, scalar=-2.0, root_operations=0),
        _job_rank_record(1, scalar=-2.0, root_operations=2),
    ]

    reasons = _validate_rank_records(records, records[0], 2)

    assert "root fallback operation attribution mismatch" in reasons


def _job_rank_record(rank, *, scalar, fallback_count=2, root_operations=0):
    from renormalizer.mps.distributed_job import RANK_SCHEMA

    return {
        "schema": RANK_SCHEMA,
        "run_id": "run",
        "rank": rank,
        "local_rank": rank,
        "world_size": 2,
        "device": "cuda:{}".format(rank),
        "git_commit": "abc",
        "state_hash": "same",
        "solver": {"tol": 1e-6},
        "numerical": {
            "passed": True,
            "energy": scalar,
            "energy_error": 0.0,
            "max_abs_error": 0.0,
            "tdvp_norm_error": 0.0,
            "davidson_norm_error": 0.0,
            "davidson_reference_norm": 1.0,
            "norm_error": 0.0,
            "residual_norm": 0.0,
        },
        "distribution": {
            "h_v_count": 10,
            "broadcast_calls": 8,
            "allreduce_calls": 12,
            "allgather_calls": 0,
            "boundary_materialization_broadcasts": 2,
            "allgather_trap_armed": True,
            "strictly_sharded": True,
            "supported_event_count": 1,
            "supported_plan_hashes": ["plan"],
            "supported_placement_hashes": ["placement"],
            "plan_hashes": ["plan"],
            "placement_hashes": ["placement"],
            "max_solver_residual_norm": 0.0,
        },
        "fallback": {
            "count": fallback_count,
            "root_operation_count": root_operations,
        },
    }


@pytest.mark.parametrize(
    ("mode", "required"),
    [
        ("tdvp", {"max_abs_error", "tdvp_norm_error", "norm_error"}),
        (
            "davidson",
            {
                "energy",
                "energy_error",
                "davidson_norm_error",
                "davidson_reference_norm",
                "norm_error",
                "residual_norm",
            },
        ),
        (
            "both",
            {
                "max_abs_error",
                "energy",
                "energy_error",
                "tdvp_norm_error",
                "davidson_norm_error",
                "davidson_reference_norm",
                "norm_error",
                "residual_norm",
            },
        ),
    ],
)
def test_formal_numerical_predicate_requires_mode_fields_and_finite_values(
    mode, required
):
    from renormalizer.backend._distributed import job as shared_job
    from renormalizer.mps import distributed_job as mps_job
    from renormalizer.tn import distributed_job as ttns_job

    args = SimpleNamespace(
        mode=mode, atol=1e-6, rtol=1e-6, davidson_tol=1e-5
    )
    numerical = {
        "max_abs_error": 0.0,
        "energy": -1.0,
        "energy_error": 0.0,
        "tdvp_norm_error": 1e-7,
        "davidson_norm_error": 2e-7,
        "davidson_reference_norm": 1.0,
        "norm_error": 0.0,
        "residual_norm": 0.0,
    }
    numerical["norm_error"] = {
        "tdvp": numerical["tdvp_norm_error"],
        "davidson": numerical["davidson_norm_error"],
        "both": max(
            numerical["tdvp_norm_error"], numerical["davidson_norm_error"]
        ),
    }[mode]

    assert mps_job._numerical_passed is shared_job._numerical_passed
    assert ttns_job._numerical_passed is shared_job._numerical_passed
    assert shared_job._numerical_passed(numerical, args) is True
    for field in required:
        missing = dict(numerical)
        missing.pop(field)
        assert shared_job._numerical_passed(missing, args) is False
        nonfinite = dict(numerical, **{field: float("nan")})
        assert shared_job._numerical_passed(nonfinite, args) is False

    if mode in {"tdvp", "both"}:
        outside_tolerance = dict(numerical, max_abs_error=3e-6)
        assert shared_job._numerical_passed(outside_tolerance, args) is False
    if mode in {"davidson", "both"}:
        outside_tolerance = dict(numerical, residual_norm=2e-5)
        assert shared_job._numerical_passed(outside_tolerance, args) is False


@pytest.mark.parametrize(
    ("mode", "expected"),
    (("tdvp", 7e-7), ("davidson", 2e-7), ("both", 7e-7)),
)
def test_formal_norm_aggregation_uses_every_requested_mode(mode, expected):
    from renormalizer.backend._distributed import job as shared_job
    from renormalizer.mps import distributed_job as mps_job
    from renormalizer.tn import distributed_job as ttns_job

    numerical = {
        "tdvp_norm_error": 7e-7,
        "davidson_norm_error": 2e-7,
    }

    assert mps_job._aggregate_norm_error is shared_job._aggregate_norm_error
    assert ttns_job._aggregate_norm_error is shared_job._aggregate_norm_error
    assert shared_job._aggregate_norm_error(numerical, mode) == pytest.approx(expected)

    required_field = {
        "tdvp": "tdvp_norm_error",
        "davidson": "davidson_norm_error",
        "both": "tdvp_norm_error",
    }[mode]
    missing = dict(numerical)
    missing.pop(required_field)
    with pytest.raises(ValueError, match="mode-specific norm evidence"):
        shared_job._aggregate_norm_error(missing, mode)

    nonfinite = dict(numerical, **{required_field: float("nan")})
    assert np.isnan(shared_job._aggregate_norm_error(nonfinite, mode))


def test_formal_both_norm_aggregation_never_hides_later_nonfinite_evidence():
    from renormalizer.backend._distributed.job import _aggregate_norm_error

    aggregate = _aggregate_norm_error(
        {"tdvp_norm_error": 0.0, "davidson_norm_error": float("nan")},
        "both",
    )

    assert np.isnan(aggregate)


@pytest.mark.parametrize(
    ("mode", "field"),
    [
        ("tdvp", "max_abs_error"),
        ("tdvp", "tdvp_norm_error"),
        ("tdvp", "norm_error"),
        ("davidson", "energy_error"),
        ("davidson", "davidson_norm_error"),
        ("davidson", "norm_error"),
        ("davidson", "residual_norm"),
    ],
)
def test_formal_numerical_predicate_rejects_negative_error_evidence(mode, field):
    from renormalizer.backend._distributed.job import _numerical_passed

    args = SimpleNamespace(
        mode=mode, atol=1e-6, rtol=1e-6, davidson_tol=1e-5
    )
    numerical = {
        "max_abs_error": 0.0,
        "energy": -1.0,
        "energy_error": 0.0,
        "tdvp_norm_error": 0.0,
        "davidson_norm_error": 0.0,
        "norm_error": 0.0,
        "residual_norm": 0.0,
    }
    numerical[field] = -1e-12
    if field == "tdvp_norm_error":
        numerical["norm_error"] = numerical[field]
    elif field == "davidson_norm_error":
        numerical["norm_error"] = numerical[field]

    assert _numerical_passed(numerical, args) is False


def test_formal_profile_and_rank_aggregates_propagate_nonfinite_metrics(tmp_path):
    from renormalizer.backend._distributed.job import (
        _max_scalar_spread,
        _profile_distribution,
        _summary_numerical,
    )

    profile = tmp_path / "profile.jsonl"
    profile.write_text(
        "\n".join(
            json.dumps(
                {
                    "event": "distributed_solve_summary",
                    "hv_count": 1,
                    "solver_residual_norm": residual,
                    "plan_hash": "plan",
                    "placement_hash": "placement",
                    "local_shard_extent": 1,
                    "global_shard_extent": 2,
                }
            )
            for residual in (0.0, float("nan"))
        )
        + "\n",
        encoding="utf-8",
    )
    records = [
        {"numerical": {"max_abs_error": 0.0, "tdvp_norm_error": 0.0, "norm_error": 0.0, "passed": True}},
        {"numerical": {"max_abs_error": float("nan"), "tdvp_norm_error": 0.0, "norm_error": 0.0, "passed": True}},
    ]
    args = SimpleNamespace(
        mode="tdvp", atol=1e-6, rtol=1e-6, davidson_tol=1e-5
    )

    assert np.isnan(_profile_distribution(profile)["max_solver_residual_norm"])
    summary = _summary_numerical(records, args)
    assert np.isnan(summary["max_abs_error"])
    assert summary["passed"] is False
    assert np.isnan(_max_scalar_spread(records))


def test_formal_both_mode_rejects_hidden_or_inconsistent_mode_norms():
    from renormalizer.backend._distributed.job import _numerical_passed

    args = SimpleNamespace(
        mode="both", atol=1e-6, rtol=1e-6, davidson_tol=1e-5
    )
    numerical = {
        "max_abs_error": 0.0,
        "energy": -1.0,
        "energy_error": 0.0,
        "tdvp_norm_error": 3e-6,
        "davidson_norm_error": 0.0,
        "davidson_reference_norm": 1.0,
        "norm_error": 0.0,
        "residual_norm": 0.0,
    }

    assert _numerical_passed(numerical, args) is False
    numerical["norm_error"] = numerical["tdvp_norm_error"]
    assert _numerical_passed(numerical, args) is False
    numerical["tdvp_norm_error"] = 1e-7
    assert _numerical_passed(numerical, args) is False
    numerical["norm_error"] = max(
        numerical["tdvp_norm_error"], numerical["davidson_norm_error"]
    )
    assert _numerical_passed(numerical, args) is True


@pytest.mark.parametrize(
    "missing_field",
    ("tdvp_norm_error", "davidson_norm_error", "norm_error"),
)
def test_formal_summary_rejects_claimed_pass_with_missing_numerical_field(
    missing_field, tmp_path
):
    from renormalizer.mps.distributed_job import _root_summary, _write_json, build_parser

    args = build_parser().parse_args(
        [
            "--mode", "both",
            "--output-dir", str(tmp_path),
            "--expected-world-size", "2",
            "--shard-solver-vectors",
        ]
    )
    records = [
        _job_rank_record(0, scalar=-2.0, fallback_count=0),
        _job_rank_record(1, scalar=-2.0, fallback_count=0),
    ]
    records[1]["numerical"].pop(missing_field)
    for record in records:
        _write_json(tmp_path / "rank-{:05d}.json".format(record["rank"]), record)

    summary = _root_summary(
        args, SimpleNamespace(world_size=2), tmp_path, records[0]
    )

    assert summary["status"] == "fail"
    assert summary["numerical"]["passed"] is False
    assert summary["numerical"][missing_field] is None


@pytest.mark.parametrize(
    "module_name",
    ("renormalizer.mps.distributed_job", "renormalizer.tn.distributed_job"),
)
def test_formal_summary_rejects_hidden_requested_mode_norm_error(
    module_name, tmp_path
):
    job = importlib.import_module(module_name)
    args = job.build_parser().parse_args(
        [
            "--mode", "both",
            "--output-dir", str(tmp_path),
            "--expected-world-size", "2",
            "--shard-solver-vectors",
            "--atol", "1e-6",
            "--rtol", "1e-6",
        ]
    )
    records = [
        _job_rank_record(0, scalar=-2.0, fallback_count=0),
        _job_rank_record(1, scalar=-2.0, fallback_count=0),
    ]
    for record in records:
        record["numerical"].update(
            tdvp_norm_error=3e-6,
            davidson_norm_error=0.0,
            norm_error=0.0,
        )
        job._write_json(
            tmp_path / "rank-{:05d}.json".format(record["rank"]), record
        )

    summary = job._root_summary(
        args, SimpleNamespace(world_size=2), tmp_path, records[0]
    )

    assert summary["status"] == "fail"
    assert summary["numerical"]["passed"] is False
    assert summary["numerical"]["tdvp_norm_error"] == pytest.approx(3e-6)
    assert summary["numerical"]["davidson_norm_error"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "module_name",
    ("renormalizer.mps.distributed_job", "renormalizer.tn.distributed_job"),
)
def test_formal_summary_status_explicitly_requires_rank_consistency(
    module_name, tmp_path, monkeypatch
):
    job = importlib.import_module(module_name)
    args = job.build_parser().parse_args(
        [
            "--mode", "both",
            "--output-dir", str(tmp_path),
            "--expected-world-size", "2",
            "--shard-solver-vectors",
        ]
    )
    records = [
        _job_rank_record(0, scalar=-2.0, fallback_count=0),
        _job_rank_record(1, scalar=-2.0, fallback_count=0),
    ]
    records[1]["state_hash"] = "different"
    for record in records:
        job._write_json(
            tmp_path / "rank-{:05d}.json".format(record["rank"]), record
        )
    monkeypatch.setattr(job, "_validate_rank_records", lambda *_args: [])

    summary = job._root_summary(
        args, SimpleNamespace(world_size=2), tmp_path, records[0]
    )

    assert summary["reasons"] == []
    assert summary["numerical"]["passed"] is True
    assert summary["rank_consistency"]["passed"] is False
    assert summary["status"] == "fail"


def test_both_formal_jobs_aggregate_norm_after_all_requested_workflows():
    from renormalizer.mps import distributed_job as mps_job
    from renormalizer.tn import distributed_job as ttns_job

    for module in (mps_job, ttns_job):
        source = Path(module.__file__).read_text(encoding="utf-8")
        aggregate = source.index("_aggregate_norm_error(numerical, args.mode)")
        assert aggregate > source.rindex("numerical.update(result)")


def test_mps_tdvp_job_preserves_mode_specific_norm_error(monkeypatch):
    from renormalizer.mps import distributed_job as job

    initial, mpo = _mps_case()
    monkeypatch.setattr(
        job, "_run_root_reference", lambda _execution, operation, _receive: operation()
    )
    set_backend("numpy", execution_policy="execution_ir")
    args = SimpleNamespace(
        tdvp_sites="one", bond_dimension=4, model_size=4, tau=1e-3
    )

    _final_state, numerical = job._run_tdvp(
        args, initial, mpo, _execution_config()
    )

    assert np.isfinite(numerical["tdvp_norm_error"])
    assert numerical["tdvp_norm_error"] >= 0.0


def test_ttns_tdvp_job_preserves_mode_specific_norm_error(monkeypatch):
    from renormalizer import BasisHalfSpin
    from renormalizer.model.model import heisenberg_ops
    from renormalizer.tn import BasisTree, TTNO, TTNS
    from renormalizer.tn import distributed_job as job

    np.random.seed(1611)
    basis = BasisTree.binary([BasisHalfSpin(index) for index in range(4)])
    initial = TTNS.random(basis, qntot=0, m_max=3)
    operator = TTNO(basis, heisenberg_ops(4))
    monkeypatch.setattr(
        job, "_run_root_reference", lambda _execution, operation, _receive: operation()
    )
    set_backend(
        "numpy", execution_policy="execution_ir", fallback_policy="legacy_oe"
    )
    args = SimpleNamespace(tdvp_sites="one", tau=1e-3)

    _final_state, numerical = job._run_tdvp(
        args, initial, operator, _execution_config()
    )

    assert np.isfinite(numerical["tdvp_norm_error"])
    assert numerical["tdvp_norm_error"] >= 0.0


def test_mps_davidson_job_measures_final_normalization_against_one(
    monkeypatch,
):
    import renormalizer
    from renormalizer.mps import distributed_job as job

    initial, mpo = _mps_case()
    observed_norms = []

    def optimize(state, _operator):
        state[0].array[...] *= 2 + len(observed_norms)
        observed_norms.append(state.mp_norm)
        return [-1.0], state

    monkeypatch.setattr(renormalizer, "optimize_mps", optimize)
    monkeypatch.setattr(
        job, "_run_root_reference", lambda _execution, operation, _receive: operation()
    )
    set_backend("numpy", execution_policy="execution_ir")
    args = SimpleNamespace(
        bond_dimension=4, davidson_sweeps=1, davidson_sites="one"
    )

    final_state, numerical = job._run_davidson(
        args, initial, mpo, _execution_config()
    )

    assert final_state.mp_norm == pytest.approx(observed_norms[1])
    assert numerical["davidson_norm_error"] == pytest.approx(
        abs(observed_norms[1] - 1.0)
    )
    assert numerical["davidson_reference_norm"] == pytest.approx(observed_norms[0])


def test_ttns_davidson_job_measures_final_normalization_against_one(
    monkeypatch,
):
    from renormalizer import BasisHalfSpin
    from renormalizer.model.model import heisenberg_ops
    from renormalizer.tn import BasisTree, TTNO, TTNS
    from renormalizer.tn import distributed_job as job
    from renormalizer.tn import gs

    np.random.seed(1610)
    basis = BasisTree.binary([BasisHalfSpin(index) for index in range(4)])
    initial = TTNS.random(basis, qntot=0, m_max=3)
    operator = TTNO(basis, heisenberg_ops(4))
    observed_norms = []

    def optimize(state, _operator, _procedure):
        state.node_list[0].tensor[...] *= 2 + len(observed_norms)
        observed_norms.append(state.ttns_norm)
        return [-1.0]

    monkeypatch.setattr(gs, "optimize_ttns", optimize)
    monkeypatch.setattr(
        job, "_run_root_reference", lambda _execution, operation, _receive: operation()
    )
    set_backend("numpy", execution_policy="execution_ir")
    args = SimpleNamespace(bond_dimension=3, davidson_sweeps=1)

    final_state, numerical = job._run_davidson(
        args, initial, operator, _execution_config()
    )

    assert final_state.ttns_norm == pytest.approx(observed_norms[1])
    assert numerical["davidson_norm_error"] == pytest.approx(
        abs(observed_norms[1] - 1.0)
    )
    assert numerical["davidson_reference_norm"] == pytest.approx(observed_norms[0])


def test_formal_multirank_rejects_fallback_only_solve_and_empty_plan_proof(tmp_path):
    from renormalizer.backend._distributed.job import _profile_distribution
    from renormalizer.mps.distributed_job import _validate_rank_records

    profile = tmp_path / "profile.jsonl"
    profile.write_text(
        json.dumps(
            {
                "event": "distributed_solve_summary",
                "hv_count": 1,
                "broadcast_calls": 1,
                "allreduce_calls": 1,
                "allgather_calls": 0,
                "fallback_count": 1,
                "root_operation_count": 1,
                "local_shard_extent": 8,
                "global_shard_extent": 8,
                "plan_hash": "plan",
                "placement_hash": "fallback",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    distribution = _profile_distribution(profile)
    assert distribution["strictly_sharded"] is False
    assert distribution["supported_event_count"] == 0
    assert distribution["supported_plan_hashes"] == []
    assert distribution["supported_placement_hashes"] == []

    records = [
        _job_rank_record(0, scalar=-2.0, fallback_count=1, root_operations=1),
        _job_rank_record(1, scalar=-2.0, fallback_count=1),
    ]
    for record in records:
        record["distribution"].update(
            strictly_sharded=True,
            supported_event_count=0,
            supported_plan_hashes=[],
            supported_placement_hashes=[],
            plan_hashes=["fallback"],
            placement_hashes=["fallback"],
        )
    reasons = _validate_rank_records(records, records[0], 2)

    assert "supported distributed solve evidence missing" in reasons
    assert "supported distributed plan hash evidence missing" in reasons
    assert "supported distributed placement hash evidence missing" in reasons

    for record in records:
        record["distribution"]["supported_event_count"] = 1
        record["distribution"]["supported_plan_hashes"] = ["fallback"]
        record["distribution"]["supported_placement_hashes"] = ["fallback"]
    assert "supported distributed plan hash evidence missing" in _validate_rank_records(
        records, records[0], 2
    )
    assert (
        "supported distributed placement hash evidence missing"
        in _validate_rank_records(records, records[0], 2)
    )


def test_profile_supported_solve_requires_genuine_plan_and_placement_hashes(tmp_path):
    from renormalizer.backend._distributed.job import _profile_distribution

    profile = tmp_path / "profile.jsonl"
    events = [
        {
            "event": "distributed_solve_summary",
            "hv_count": 1,
            "local_shard_extent": 2,
            "global_shard_extent": 4,
            "plan_hash": "plan-a",
            "placement_hash": "fallback",
        },
        {
            "event": "distributed_solve_summary",
            "hv_count": 1,
            "local_shard_extent": 2,
            "global_shard_extent": 4,
            "plan_hash": "plan-b",
            "placement_hash": "placement-b",
        },
    ]
    profile.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    distribution = _profile_distribution(profile)

    assert distribution["supported_event_count"] == 1
    assert distribution["supported_plan_hashes"] == ["plan-b"]
    assert distribution["supported_placement_hashes"] == ["placement-b"]


def test_job_summary_reports_real_scalar_spread_and_logical_fallback_count(tmp_path):
    from renormalizer.mps.distributed_job import _root_summary, _write_json, build_parser

    args = build_parser().parse_args(
        [
            "--mode", "both",
            "--output-dir", str(tmp_path),
            "--expected-world-size", "2",
            "--shard-solver-vectors",
            "--rank-atol", "0.3",
        ]
    )
    records = [
        _job_rank_record(0, scalar=-2.0, root_operations=2),
        _job_rank_record(1, scalar=-1.75),
    ]
    for record in records:
        _write_json(tmp_path / "rank-{:05d}.json".format(record["rank"]), record)

    summary = _root_summary(
        args,
        SimpleNamespace(world_size=2),
        tmp_path,
        records[0],
    )

    assert summary["rank_consistency"]["max_scalar_spread"] == pytest.approx(0.25)
    assert summary["rank_consistency"]["passed"] is True
    assert summary["fallback"] == {
        "count": 2,
        "root_operation_count": 2,
        "synchronized": True,
    }
    assert summary["distribution"]["supported_event_count"] == 1
    assert summary["distribution"]["supported_plan_hashes"] == ["plan"]
    assert summary["distribution"]["supported_placement_hashes"] == ["placement"]


def test_job_summary_turns_malformed_rank_file_into_fail_evidence(tmp_path):
    from renormalizer.mps.distributed_job import _root_summary, _write_json, build_parser

    args = build_parser().parse_args(
        [
            "--mode", "tdvp",
            "--output-dir", str(tmp_path),
            "--expected-world-size", "2",
            "--shard-solver-vectors",
        ]
    )
    record = _job_rank_record(0, scalar=-2.0, fallback_count=0)
    _write_json(tmp_path / "rank-00000.json", record)
    (tmp_path / "rank-00001.json").write_text("{malformed", encoding="utf-8")

    summary = _root_summary(
        args,
        SimpleNamespace(world_size=2),
        tmp_path,
        record,
    )

    assert summary["status"] == "fail"
    assert any("malformed rank record" in reason for reason in summary["reasons"])


def test_job_output_setup_failure_is_synchronized_without_barrier(tmp_path):
    from renormalizer.mps.distributed_job import _prepare_output

    output = tmp_path / "occupied"
    output.mkdir()
    (output / "prior.json").write_text("{}", encoding="utf-8")

    class Collective:
        rank = 0
        size = 2

        def __init__(self):
            self.trace = []

        def allreduce(self, value, *, op="sum"):
            self.trace.append((op, np.asarray(value).dtype.str))
            return np.array(value, copy=True)

    collective = Collective()
    runtime = SimpleNamespace(
        rank=0,
        world_size=2,
        collective=collective,
        barrier=lambda: pytest.fail("setup failure must not enter a barrier"),
    )

    with pytest.raises(RuntimeError, match="output setup failed"):
        _prepare_output(output, runtime)

    assert collective.trace == [("max", np.dtype(np.int32).str)]


@pytest.mark.parametrize("failure_site", ["copy", "config", "allocation"])
def test_mps_job_tdvp_setup_phases_synchronize_baseexception_failures(
    failure_site, monkeypatch
):
    from renormalizer.cons import backend
    from renormalizer.mps.distributed_job import _run_tdvp

    class InjectedSetupFailure(BaseException):
        pass

    class Collective:
        rank = 0
        size = 2

        def __init__(self):
            self.trace = []

        def allreduce(self, value, *, op="sum"):
            self.trace.append((op, np.asarray(value).dtype.str))
            return np.array(value, copy=True)

    args = SimpleNamespace(tdvp_sites="one", bond_dimension=4, model_size=4)
    execution = SimpleNamespace(
        context=SimpleNamespace(rank=0), collective=Collective()
    )
    state, mpo = _mps_case()
    set_backend("numpy", execution_policy="execution_ir")
    if failure_site == "copy":
        monkeypatch.setattr(
            type(state),
            "copy",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                InjectedSetupFailure("injected copy failure")
            ),
        )
    elif failure_site == "config":
        monkeypatch.setattr(
            "renormalizer.utils.EvolveConfig",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                InjectedSetupFailure("injected config failure")
            ),
        )
    else:
        monkeypatch.setattr(
            backend.current.array_namespace,
            "empty",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                InjectedSetupFailure("injected allocation failure")
            ),
        )

    with pytest.raises(RuntimeError, match="TDVP workflow setup failed"):
        _run_tdvp(args, state, mpo, execution)

    assert execution.collective.trace == [("max", np.dtype(np.int32).str)]


@pytest.mark.parametrize("failure_site", ["flush", "parse"])
def test_job_profile_finalize_phases_are_synchronized(
    failure_site, tmp_path, monkeypatch
):
    from renormalizer.mps.distributed_job import _finalize_profile
    from renormalizer.utils import profiling

    class InjectedProfileFailure(BaseException):
        pass

    class Collective:
        rank = 0
        size = 2

        def __init__(self):
            self.trace = []

        def allreduce(self, value, *, op="sum"):
            self.trace.append((op, np.asarray(value).dtype.str))
            return np.array(value, copy=True)

    runtime = SimpleNamespace(rank=0, collective=Collective())
    profile_path = tmp_path / "profile.jsonl"
    monkeypatch.setattr(profiling, "record_run_summary", lambda: None)
    if failure_site == "flush":
        monkeypatch.setattr(
            profiling,
            "flush_event_output",
            lambda: (_ for _ in ()).throw(
                InjectedProfileFailure("injected profile flush failure")
            ),
        )
        expected = "profile flush failed"
    else:
        monkeypatch.setattr(profiling, "flush_event_output", lambda: None)
        profile_path.write_text("{malformed\n", encoding="utf-8")
        expected = "profile parse failed"

    with pytest.raises(RuntimeError, match=expected):
        _finalize_profile(runtime, profile_path)

    expected_calls = 1 if failure_site == "flush" else 2
    assert runtime.collective.trace == [
        ("max", np.dtype(np.int32).str)
    ] * expected_calls


def test_job_reference_operation_executes_only_on_root_and_broadcasts_numeric_result():
    from renormalizer.mps.distributed_job import _run_root_reference

    root_calls = []
    root_execution = SimpleNamespace(collective=SingleProcessCollective())
    root_receive = np.empty(2, dtype=np.float64)
    root_result = _run_root_reference(
        root_execution,
        lambda: root_calls.append(0) or np.array([1.0, 2.0]),
        root_receive,
    )
    np.testing.assert_array_equal(root_result, [1.0, 2.0])
    assert root_calls == [0]

    class NonRootCollective:
        rank = 1
        size = 2

        def allreduce(self, value, *, op="sum"):
            return np.array(value, copy=True)

        def broadcast(self, value, *, root):
            value[...] = [3.0, 4.0]
            return value

    nonroot_execution = SimpleNamespace(collective=NonRootCollective())
    result = _run_root_reference(
        nonroot_execution,
        lambda: pytest.fail("reference ran on a non-root rank"),
        np.empty(2, dtype=np.float64),
    )
    np.testing.assert_array_equal(result, [3.0, 4.0])


def test_mps_launcher_has_explicit_gpu_safety_and_owned_cleanup():
    script = Path(__file__).parents[3] / "scripts" / "run_mps_multigpu_job.sh"
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


@pytest.mark.parametrize(
    "script_name",
    ["run_mps_multigpu_job.sh", "run_tree_multigpu_job.sh"],
)
def test_launcher_records_and_revalidates_process_start_identity_before_signals(
    script_name,
):
    script = Path(__file__).parents[3] / "scripts" / script_name
    text = script.read_text(encoding="utf-8")

    assert "declare -A owned_start_identities" in text
    assert "process_start_identity()" in text
    signal_function = text.index("signal_owned_process()")
    revalidation = text.index("process_start_identity", signal_function)
    signal = text.index('kill -TERM "$pid"', revalidation)
    assert signal_function < revalidation < signal
    assert '"owned_process_start_identities"' in text
    assert '"owned_identity_revalidation_succeeded"' in text
    assert '"owned_identity_errors"' in text


@pytest.mark.parametrize(
    "script_name",
    ["run_mps_multigpu_job.sh", "run_tree_multigpu_job.sh"],
)
def test_launcher_revalidates_each_parent_before_descendant_traversal(script_name):
    script = Path(__file__).parents[3] / "scripts" / script_name
    text = script.read_text(encoding="utf-8")
    refresh_start = text.index("refresh_owned_pids()")
    refresh_end = text.index("cleanup()", refresh_start)
    refresh = text[refresh_start:refresh_end]

    parent_loop = refresh.index("for parent in $owned_pids")
    identity_check = refresh.index('owned_process_is_current "$parent"', parent_loop)
    snapshot = refresh.index("process_snapshot=$(ps", identity_check)
    traversal = refresh.index("awk -v parents=", snapshot)

    assert parent_loop < identity_check < snapshot < traversal


@pytest.mark.parametrize(
    "script_name",
    ["run_mps_multigpu_job.sh", "run_tree_multigpu_job.sh"],
)
def test_launcher_compute_application_query_failure_fails_closed(
    script_name, tmp_path
):
    script = Path(__file__).parents[3] / "scripts" / script_name
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    fake_nvidia_smi = binary_dir / "nvidia-smi"
    fake_nvidia_smi.write_text(
        """#!/usr/bin/env bash
case "$*" in
  *"--query-gpu=index,uuid,name"*)
    echo "4, GPU-test-id, NVIDIA H100 80GB HBM3" ;;
  *"--query-compute-apps="*)
    echo "injected compute query failure" >&2
    exit 42 ;;
  *"--query-gpu=uuid,memory.used,utilization.gpu"*)
    echo "GPU-test-id, 1, 0" ;;
  *) exit 0 ;;
esac
""",
        encoding="ascii",
    )
    fake_nvidia_smi.chmod(0o755)
    fake_flock = binary_dir / "flock"
    fake_flock.write_text(
        "#!/usr/bin/env bash\necho 'GPU lock busy' >&2\nexit 75\n",
        encoding="ascii",
    )
    fake_flock.chmod(0o755)
    environment = dict(os.environ)
    environment.update(
        {
            "PATH": str(binary_dir) + os.pathsep + environment["PATH"],
            "CUDA_VISIBLE_DEVICES": "4",
            "RENO_MULTIGPU_NPROC": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(script), "both", str(tmp_path / "output")],
        text=True,
        capture_output=True,
        env=environment,
        timeout=15,
    )

    assert result.returncode != 0
    assert "injected compute query failure" in result.stderr
    assert "GPU lock busy" not in result.stderr


@pytest.mark.parametrize(
    "script_name",
    ["run_mps_multigpu_job.sh", "run_tree_multigpu_job.sh"],
)
def test_launcher_cleanup_compute_query_failure_fails_closed(
    script_name, tmp_path
):
    script = Path(__file__).parents[3] / "scripts" / script_name
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    query_count = tmp_path / "compute-query-count"
    fake_nvidia_smi = binary_dir / "nvidia-smi"
    fake_nvidia_smi.write_text(
        """#!/usr/bin/env bash
if [[ "${1:-}" == "dmon" ]]; then
  trap 'exit 0' TERM INT
  while true; do sleep 1; done
fi
case "$*" in
  *"--query-gpu=index,uuid,name"*)
    echo "4, GPU-test-id, NVIDIA H100 80GB HBM3" ;;
  *"--query-compute-apps="*)
    count=0
    [[ ! -f "$QUERY_COUNT" ]] || count=$(cat "$QUERY_COUNT")
    count=$((count + 1))
    printf '%s\n' "$count" >"$QUERY_COUNT"
    if [[ $count -gt 2 ]]; then
      echo "injected cleanup compute query failure" >&2
      exit 42
    fi ;;
  *"--query-gpu=uuid,memory.used,utilization.gpu"*)
    echo "GPU-test-id, 1, 0" ;;
  *) exit 0 ;;
esac
""",
        encoding="ascii",
    )
    fake_nvidia_smi.chmod(0o755)
    fake_python = binary_dir / "python"
    fake_python.write_text(
        """#!/usr/bin/env bash
if [[ "${1:-}" == "-m" && "${2:-}" == "torch.distributed.run" ]]; then
  exit 0
fi
exec "$REAL_PYTHON" "$@"
""",
        encoding="ascii",
    )
    fake_python.chmod(0o755)
    fake_flock = binary_dir / "flock"
    fake_flock.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="ascii")
    fake_flock.chmod(0o755)
    output_dir = tmp_path / "output"
    environment = dict(os.environ)
    environment.update(
        {
            "PATH": str(binary_dir) + os.pathsep + environment["PATH"],
            "CUDA_VISIBLE_DEVICES": "4",
            "RENO_MULTIGPU_NPROC": "1",
            "QUERY_COUNT": str(query_count),
            "REAL_PYTHON": sys.executable,
        }
    )

    result = subprocess.run(
        ["bash", str(script), "both", str(output_dir)],
        text=True,
        capture_output=True,
        env=environment,
        timeout=15,
    )

    assert result.returncode != 0
    summary = json.loads(
        (output_dir / "launcher-summary.json").read_text(encoding="utf-8")
    )
    assert summary["exit_code"] == 1
    assert summary["surviving_selected_gpu_contexts"] == []
    assert (
        summary["selected_gpu_context_query_error"]
        == "injected cleanup compute query failure"
    )


def _run_launcher_with_cleanup_failure(script_name, failure, tmp_path):
    script = Path(__file__).parents[3] / "scripts" / script_name
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    fake_nvidia_smi = binary_dir / "nvidia-smi"
    fake_nvidia_smi.write_text(
        """#!/usr/bin/env bash
if [[ "${1:-}" == "dmon" ]]; then
  if [[ "${INJECT_CLEANUP_FAILURE:-}" =~ ^(signal|identity)$ ]]; then
    sleep 0.3
    exit 0
  fi
  trap 'exit 0' TERM INT
  while true; do sleep 0.1; done
fi
case "$*" in
  *"--query-gpu=index,uuid,name"*)
    echo "4, GPU-test-id, NVIDIA H100 80GB HBM3" ;;
  *"--query-compute-apps="*) ;;
  *"--query-gpu=uuid,memory.used,utilization.gpu"*)
    echo "GPU-test-id, 1, 0" ;;
  *) exit 0 ;;
esac
""",
        encoding="ascii",
    )
    fake_nvidia_smi.chmod(0o755)
    fake_python = binary_dir / "python"
    fake_python.write_text(
        """#!/usr/bin/env bash
if [[ "${1:-}" == "-m" && "${2:-}" == "torch.distributed.run" ]]; then
  if [[ "${INJECT_CLEANUP_FAILURE:-}" =~ ^(parent_identity|ps)$ ]]; then
    /bin/sleep 2
  fi
  exit 0
fi
exec "$REAL_PYTHON" "$@"
""",
        encoding="ascii",
    )
    fake_python.chmod(0o755)
    fake_flock = binary_dir / "flock"
    fake_flock.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="ascii")
    fake_flock.chmod(0o755)
    bash_env = tmp_path / "bash-env"
    bash_env.write_text(
        """ps() {
  printf 'called\n' >>"$PS_LOG"
  if [[ "${INJECT_CLEANUP_FAILURE:-}" == "ps" ]]; then
    echo "injected ps failure" >&2
    return 41
  fi
  command ps "$@"
}
ss() {
  if [[ "${INJECT_CLEANUP_FAILURE:-}" == "ss" ]]; then
    echo "injected ss failure" >&2
    return 42
  fi
  return 0
}
kill() {
  if [[ "${INJECT_CLEANUP_FAILURE:-}" == "parent_identity" && "${1:-}" == "-0" && "${2:-}" == "$torchrun_pid" && ! -f "$PARENT_IDENTITY_INJECTED" ]]; then
    : >"$PARENT_IDENTITY_INJECTED"
    owned_start_identities["$torchrun_pid"]="injected-reused-parent-start"
  fi
  if [[ "${INJECT_CLEANUP_FAILURE:-}" == "identity" && "${1:-}" == "-TERM" ]]; then
    printf '%s\n' "${2:-}" >>"$SIGNAL_LOG"
  fi
  if [[ "${INJECT_CLEANUP_FAILURE:-}" == "signal" && "${1:-}" == "-TERM" ]]; then
    echo "injected signal failure" >&2
    return 43
  fi
  builtin kill "$@"
}
wait() {
  builtin wait "$@"
  observed=$?
  if [[ "${INJECT_CLEANUP_FAILURE:-}" == "identity" && ! -f "$IDENTITY_INJECTED" ]]; then
    : >"$IDENTITY_INJECTED"
    owned_start_identities["$dmon_pid"]="injected-reused-start"
  fi
  if [[ "${INJECT_CLEANUP_FAILURE:-}" == "wait" && ! -f "$WAIT_INJECTED" ]]; then
    : >"$WAIT_INJECTED"
    echo "injected wait failure" >&2
    return 47
  fi
  return "$observed"
}
""",
        encoding="ascii",
    )
    output_dir = tmp_path / "output"
    environment = dict(os.environ)
    environment.update(
        {
            "PATH": str(binary_dir) + os.pathsep + environment["PATH"],
            "BASH_ENV": str(bash_env),
            "CUDA_VISIBLE_DEVICES": "4",
            "INJECT_CLEANUP_FAILURE": failure,
            "IDENTITY_INJECTED": str(tmp_path / "identity-injected"),
            "PARENT_IDENTITY_INJECTED": str(tmp_path / "parent-identity-injected"),
            "PS_LOG": str(tmp_path / "ps-log"),
            "RENO_MULTIGPU_NPROC": "1",
            "REAL_PYTHON": sys.executable,
            "SIGNAL_LOG": str(tmp_path / "signal-log"),
            "WAIT_INJECTED": str(tmp_path / "wait-injected"),
        }
    )
    result = subprocess.run(
        ["bash", str(script), "both", str(output_dir)],
        text=True,
        capture_output=True,
        env=environment,
        timeout=15,
    )
    summary = json.loads(
        (output_dir / "launcher-summary.json").read_text(encoding="utf-8")
    )
    return result, summary


@pytest.mark.parametrize(
    "script_name",
    ["run_mps_multigpu_job.sh", "run_tree_multigpu_job.sh"],
)
def test_launcher_refuses_signal_after_process_start_identity_changes(
    script_name, tmp_path
):
    result, summary = _run_launcher_with_cleanup_failure(
        script_name, "identity", tmp_path
    )

    assert result.returncode != 0
    assert summary["exit_code"] != 0
    assert summary["owned_identity_revalidation_succeeded"] is False
    assert any("dmon" in error for error in summary["owned_identity_errors"])
    assert summary["owned_dmon_term_attempted"] is False
    assert summary["owned_dmon_term_succeeded"] is False
    assert summary["owned_process_start_identities"]
    signal_log = tmp_path / "signal-log"
    assert not signal_log.exists() or not signal_log.read_text().strip()


@pytest.mark.parametrize(
    "script_name",
    ["run_mps_multigpu_job.sh", "run_tree_multigpu_job.sh"],
)
def test_launcher_refuses_descendant_traversal_after_parent_identity_changes(
    script_name, tmp_path
):
    result, summary = _run_launcher_with_cleanup_failure(
        script_name, "parent_identity", tmp_path
    )

    assert result.returncode != 0
    assert summary["exit_code"] != 0
    assert summary["owned_identity_revalidation_succeeded"] is False
    assert any("parent" in error for error in summary["owned_identity_errors"])
    ps_log = tmp_path / "ps-log"
    assert not ps_log.exists() or not ps_log.read_text().strip()


@pytest.mark.parametrize(
    "script_name",
    ["run_mps_multigpu_job.sh", "run_tree_multigpu_job.sh"],
)
@pytest.mark.parametrize("failure", ["ps", "ss", "signal", "wait"])
def test_launcher_cleanup_command_failures_fail_closed_and_report_observations(
    script_name, failure, tmp_path
):
    result, summary = _run_launcher_with_cleanup_failure(
        script_name, failure, tmp_path
    )

    assert result.returncode != 0
    assert summary["exit_code"] != 0
    if failure == "ps":
        assert summary["owned_pid_discovery_succeeded"] is False
        assert "injected ps failure" in summary["owned_pid_discovery_error"]
    elif failure == "ss":
        assert summary["owned_listener_query_succeeded"] is False
        assert "injected ss failure" in summary["owned_listener_query_error"]
    elif failure == "signal":
        assert summary["owned_dmon_term_attempted"] is True
        assert summary["owned_dmon_term_succeeded"] is False
        assert any(
            "injected signal failure" in error
            for error in summary["owned_signal_errors"]
        )
    else:
        assert summary["owned_torchrun_waited"] is True
        assert summary["owned_torchrun_wait_status"] == 47
