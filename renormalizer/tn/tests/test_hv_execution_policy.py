import importlib

import numpy as np
import pytest

from renormalizer import BasisHalfSpin, set_backend
from renormalizer.model.model import heisenberg_ops
from renormalizer.mps.matrix import asnumpy
from renormalizer.tn.tree import TTNO, TTNS, TTNEnviron
from renormalizer.tn.treebase import BasisTree
from renormalizer.utils import profiling


ttns_hop_module = importlib.import_module("renormalizer.tn.hop_expr")
oe_wrap_module = importlib.import_module("renormalizer.mps.oe_contract_wrap")


@pytest.fixture(autouse=True)
def restore_numpy_backend():
    try:
        yield
    finally:
        set_backend("numpy", precision=64)


def _interleaved_case():
    rng = np.random.default_rng(31)
    left = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 6, 7))
    right = rng.normal(size=(8, 7, 9))
    center = rng.normal(size=(4, 6, 9))
    args = [
        left,
        ("a", "b", "c"),
        mpo,
        ("b", "d", "e", "f"),
        right,
        ("l", "f", "k"),
    ]
    return args, center, ("c", "e", "k"), ("a", "d", "l")


def _build_interleaved(policy):
    args, center, input_indices, output_indices = _interleaved_case()
    set_backend("numpy", execution_policy=policy)
    expr = ttns_hop_module._contract_expression(
        args, center.shape, input_indices, output_indices, "one_site"
    )
    return expr, args, center


def test_ttns_execution_ir_matches_interleaved_legacy_equation():
    expr, args, center = _build_interleaved("execution_ir")

    actual = asnumpy(expr(center))
    expected = np.einsum("abc,bdef,lfk,cek->adl", args[0], args[2], args[4], center)

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_ttns_default_policy_does_not_select_ir(monkeypatch):
    monkeypatch.setattr(
        ttns_hop_module,
        "build_ttns_ir_hop",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("IR selected")),
        raising=False,
    )

    expr, args, center = _build_interleaved("legacy_oe")

    np.testing.assert_allclose(
        expr(center), np.einsum("abc,bdef,lfk,cek->adl", args[0], args[2], args[4], center)
    )


def test_ttns_strict_policy_propagates_lowering_error(monkeypatch):
    args, center, input_indices, output_indices = _interleaved_case()
    set_backend("numpy", execution_policy="execution_ir", fallback_policy="error")
    monkeypatch.setattr(
        ttns_hop_module,
        "build_ttns_ir_hop",
        lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError("forced")),
        raising=False,
    )

    with pytest.raises(NotImplementedError, match="forced"):
        ttns_hop_module._contract_expression(
            args, center.shape, input_indices, output_indices, "one_site"
        )


def test_ttns_explicit_fallback_records_requested_and_actual_policy(monkeypatch):
    args, center, input_indices, output_indices = _interleaved_case()
    events = []
    set_backend(
        "numpy", execution_policy="execution_ir", fallback_policy="legacy_oe"
    )
    monkeypatch.setattr(
        ttns_hop_module,
        "build_ttns_ir_hop",
        lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError("forced")),
        raising=False,
    )
    monkeypatch.setattr(profiling, "enabled", lambda: True)
    monkeypatch.setattr(
        profiling,
        "record",
        lambda event, **payload: events.append({"event": event, **payload}),
    )

    expr = ttns_hop_module._contract_expression(
        args, center.shape, input_indices, output_indices, "one_site"
    )
    actual = expr(center)

    np.testing.assert_allclose(
        actual, np.einsum("abc,bdef,lfk,cek->adl", args[0], args[2], args[4], center)
    )
    assert len(events) == 1
    event = events[0]
    assert event["event"] == "local_hv_execute"
    assert event["requested_policy"] == "execution_ir"
    assert event["actual_policy"] == "legacy_oe"
    assert event["planner_source"] == "opt_einsum"
    assert event["oe_path_hash"]
    assert event["actual_steps"]
    assert event["fallback_reason"] == "NotImplementedError: forced"


def test_ttns_plan_is_reused_and_ir_telemetry_is_truthful(monkeypatch):
    lowerer = importlib.import_module("renormalizer.backend._gemm.ttns_lowering")
    calls = []
    events = []
    original = lowerer.lower_einsum_path

    def recording_lower(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(lowerer, "lower_einsum_path", recording_lower)
    monkeypatch.setattr(profiling, "enabled", lambda: True)
    monkeypatch.setattr(
        profiling,
        "record",
        lambda event, **payload: events.append({"event": event, **payload}),
    )

    expr, args, center = _build_interleaved("execution_ir")
    assert calls == [True] * 6
    first = expr(center)
    second = expr(center * 2)

    assert calls == [True] * 6
    assert len(events) == 2
    assert all(event["requested_policy"] == "execution_ir" for event in events)
    assert all(event["actual_policy"] == "execution_ir" for event in events)
    assert all(event["planner_source"] == "opt_einsum" for event in events)
    assert all(event["oe_path_hash"] for event in events)
    assert all(event["actual_steps"] for event in events)
    expected = np.einsum("abc,bdef,lfk,cek->adl", args[0], args[2], args[4], center)
    np.testing.assert_allclose(first, expected)
    np.testing.assert_allclose(second, expected * 2)


def test_ttns_runtime_dtype_fallback_reason_is_exact_and_reuses_oe(monkeypatch):
    args, center, input_indices, output_indices = _interleaved_case()
    center = center.astype(np.longdouble)
    lowerer = importlib.import_module("renormalizer.backend._gemm.ttns_lowering")
    planner_calls = []
    legacy_builds = []
    path_calls = []
    events = []
    original_lower = lowerer.lower_einsum_path
    original_contract_expression = oe_wrap_module.oe.contract_expression
    planner = importlib.import_module("renormalizer.backend._execution.planner")
    oe_contract_module = importlib.import_module("opt_einsum.contract")
    original_contract_path = planner.oe.contract_path

    def recording_lower(*args, **kwargs):
        planner_calls.append((kwargs["dtype"], kwargs["layouts"][-1]))
        return original_lower(*args, **kwargs)

    def recording_contract_expression(*args, **kwargs):
        legacy_builds.append(True)
        return original_contract_expression(*args, **kwargs)

    def recording_contract_path(*args, **kwargs):
        path_calls.append((args, dict(kwargs)))
        return original_contract_path(*args, **kwargs)

    monkeypatch.setattr(lowerer, "lower_einsum_path", recording_lower)
    monkeypatch.setattr(
        oe_wrap_module.oe, "contract_expression", recording_contract_expression
    )
    monkeypatch.setattr(planner.oe, "contract_path", recording_contract_path)
    monkeypatch.setattr(
        oe_contract_module, "contract_path", recording_contract_path
    )
    set_backend(
        "numpy", execution_policy="execution_ir", fallback_policy="legacy_oe"
    )
    monkeypatch.setattr(profiling, "enabled", lambda: True)
    monkeypatch.setattr(
        profiling,
        "record",
        lambda event, **payload: events.append({"event": event, **payload}),
    )

    expression = ttns_hop_module._contract_expression(
        args, center.shape, input_indices, output_indices, "one_site"
    )
    planned = tuple(planner_calls)
    assert len(planned) == 6
    assert legacy_builds == [True]
    assert sum(call[1].get("optimize") == "optimal" for call in path_calls) == 1
    assert len(path_calls) == 2
    assert path_calls[1][1]["optimize"] == expression.resolved_oe_path
    constructed_path_calls = tuple(path_calls)
    first = expression(center)
    second = expression(center * 2)

    expected = np.einsum(
        "abc,bdef,lfk,cek->adl", args[0], args[2], args[4], center
    )
    reason = (
        "UnsupportedRuntimeDtypeError: execution IR runtime result dtype 'float128' "
        "with layouts ('C',) has no configured variant; available variants: "
        "complex128/C, complex128/F, complex128/strided, float64/C, float64/F, "
        "float64/strided"
    )
    np.testing.assert_allclose(first, expected)
    np.testing.assert_allclose(second, expected * 2)
    assert tuple(planner_calls) == planned
    assert legacy_builds == [True]
    assert tuple(path_calls) == constructed_path_calls
    assert len(events) == 2
    assert all(event["requested_policy"] == "execution_ir" for event in events)
    assert all(event["actual_policy"] == "legacy_oe" for event in events)
    assert all(event["fallback_reason"] == reason for event in events)
    from renormalizer.backend._execution.profiling import oe_path_identity

    assert all(
        event["oe_path_hash"] == oe_path_identity(expression.resolved_oe_path)
        for event in events
    )


def test_ttns_hdiag_data_path_is_unchanged_by_execution_policy(monkeypatch):
    sentinel_expr = object()
    sentinel_diag = np.arange(4.0)
    monkeypatch.setattr(ttns_hop_module, "_contract_expression", lambda *args: sentinel_expr)
    monkeypatch.setattr(ttns_hop_module, "_get_hdiag", lambda *args: sentinel_diag)

    class Node:
        shape = (2, 2)

    class OperatorNode:
        tensor = np.ones((2, 2))

    class EnvironmentNode:
        environ_children = []
        environ_parent = np.ones((2, 2))

    node = Node()
    ttns = type(
        "State",
        (),
        {
            "node_idx": {node: 0},
            "get_node_indices": lambda self, *args, **kwargs: ("down", "up"),
        },
    )()
    ttno = type(
        "Operator",
        (),
        {
            "node_list": [OperatorNode()],
            "get_node_indices": lambda self, *args, **kwargs: ("down", "up"),
        },
    )()
    ttne = type(
        "Environment",
        (),
        {
            "node_list": [EnvironmentNode()],
            "get_parent_indices": lambda self, *args, **kwargs: ("down", "up"),
        },
    )()

    for policy in ("legacy_oe", "execution_ir"):
        set_backend("numpy", execution_policy=policy)
        expr, hdiag = ttns_hop_module.hop_expr1(
            node, ttns, ttno, ttne, return_hdiag=True
        )
        assert expr is sentinel_expr
        assert hdiag is sentinel_diag


def test_ttns_zero_one_two_site_and_hdiag_match_legacy_numerically():
    basis = BasisTree.binary([BasisHalfSpin(index) for index in range(7)])
    ttns = TTNS.random(basis, qntot=0, m_max=3)
    ttno = TTNO(basis, heisenberg_ops(7))
    ttne = TTNEnviron(ttns, ttno)
    child = next(
        node
        for node in ttns.node_list
        if not node.children and node.parent is not ttns.root
    )
    enode = ttne.node_list[ttns.node_idx[child]]
    zero_shape = (
        enode.parent.environ_children[enode.idx_as_child].shape[0],
        enode.environ_parent.shape[0],
    )
    zero_center = np.random.default_rng(71).normal(size=zero_shape)
    parent_axis = child.parent.children.index(child)
    two_center = np.tensordot(
        child.tensor, child.parent.tensor, axes=(-1, parent_axis)
    )

    def evaluate(policy):
        set_backend("numpy", execution_policy=policy)
        zero = ttns_hop_module.hop_expr0(child, ttns, ttno, ttne)(zero_center)
        one_expr, one_hdiag = ttns_hop_module.hop_expr1(
            child, ttns, ttno, ttne, return_hdiag=True
        )
        one = one_expr(child.tensor)
        two_expr, two_hdiag = ttns_hop_module.hop_expr2(
            child, ttns, ttno, ttne
        )
        two = two_expr(two_center)
        return tuple(
            asnumpy(value)
            for value in (zero, one, one_hdiag, two, two_hdiag)
        )

    legacy = evaluate("legacy_oe")
    execution_ir = evaluate("execution_ir")

    for actual, expected in zip(execution_ir, legacy):
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
