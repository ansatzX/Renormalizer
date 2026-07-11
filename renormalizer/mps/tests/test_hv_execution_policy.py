import importlib
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from renormalizer import set_backend
from renormalizer.backend._execution.model import TransformStep
from renormalizer.mps.matrix import asnumpy
from renormalizer.utils import profiling


mps_hop_module = importlib.import_module("renormalizer.mps.hop_expr")
oe_wrap_module = importlib.import_module("renormalizer.mps.oe_contract_wrap")


@pytest.fixture(autouse=True)
def restore_numpy_backend():
    try:
        yield
    finally:
        set_backend("numpy", precision=64)


def _values(shape, rng):
    return rng.normal(size=shape)


def _typed_values(shape, dtype, rng):
    dtype = np.dtype(dtype)
    value = rng.normal(size=shape)
    if dtype.kind == "c":
        value = value + 1j * rng.normal(size=shape)
    return value.astype(dtype)


def _mps_cases():
    rng = np.random.default_rng(19)
    return (
        (
            "zero_site",
            "abc,lbk,ck->al",
            _values((2, 2, 2), rng),
            _values((2, 2, 2), rng),
            [],
            _values((2, 2), rng),
            False,
        ),
        (
            "one_site",
            "abc,bdef,lfk,cek->adl",
            _values((2, 2, 2), rng),
            _values((2, 2, 2), rng),
            [_values((2, 2, 2, 2), rng)],
            _values((2, 2, 2), rng),
            False,
        ),
        (
            "one_site_ancilla",
            "abc,bdef,lfk,cegk->adgl",
            _values((2, 2, 2), rng),
            _values((2, 2, 2), rng),
            [_values((2, 2, 2, 2), rng)],
            _values((2, 2, 2, 2), rng),
            False,
        ),
        (
            "two_site",
            "abc,bdef,fghj,ljk,cehk->adgl",
            _values((2, 2, 2), rng),
            _values((2, 2, 2), rng),
            [_values((2, 2, 2, 2), rng), _values((2, 2, 2, 2), rng)],
            _values((2, 2, 2, 2), rng),
            False,
        ),
        (
            "two_site_ancilla",
            "abc,bdef,fghj,ljk,cemhnk->admgnl",
            _values((2, 2, 2), rng),
            _values((2, 2, 2), rng),
            [_values((2, 2, 2, 2), rng), _values((2, 2, 2, 2), rng)],
            _values((2, 2, 2, 2, 2, 2), rng),
            False,
        ),
        (
            "one_site_two_layer",
            "abcd,befg,cfhi,jgik,aej->dhk",
            _values((2, 2, 2, 2), rng),
            _values((2, 2, 2, 2), rng),
            [_values((2, 2, 2, 2), rng)],
            _values((2, 2, 2), rng),
            True,
        ),
        (
            "two_site_two_layer",
            "abcd,befg,cfhi,gjkl,ikmn,olnp,aejo->dhmp",
            _values((2, 2, 2, 2), rng),
            _values((2, 2, 2, 2), rng),
            [_values((2, 2, 2, 2), rng), _values((2, 2, 2, 2), rng)],
            _values((2, 2, 2, 2), rng),
            True,
        ),
    )


def _run_case(case, policy):
    _, equation, left, right, mpos, center, two_layer = case
    set_backend("numpy", execution_policy=policy)
    hop = mps_hop_module.hop_expr(
        left.copy(), right.copy(), [mpo.copy() for mpo in mpos], center.shape,
        twolayer=two_layer,
    )
    actual = asnumpy(hop(center.copy()))
    if two_layer:
        doubled_mpos = [mpo for mpo in mpos for _ in range(2)]
        expected = np.einsum(equation, left, *doubled_mpos, right, center)
    else:
        expected = np.einsum(equation, left, *mpos, right, center)
    return actual, expected


@pytest.mark.parametrize("case", _mps_cases(), ids=lambda case: case[0])
def test_execution_ir_preserves_all_mps_hv_equations(case):
    actual, expected = _run_case(case, "execution_ir")

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_default_policy_stays_legacy(monkeypatch):
    case = _mps_cases()[1]
    set_backend("numpy")
    monkeypatch.setattr(
        mps_hop_module,
        "build_mps_ir_hop",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("IR selected")),
        raising=False,
    )

    actual, expected = _run_case(case, "legacy_oe")

    np.testing.assert_allclose(actual, expected)


def test_execution_ir_strict_policy_propagates_lowering_error(monkeypatch):
    case = _mps_cases()[1]
    _, _, left, right, mpos, center, _ = case
    set_backend("numpy", execution_policy="execution_ir", fallback_policy="error")
    monkeypatch.setattr(
        mps_hop_module,
        "build_mps_ir_hop",
        lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError("forced")),
        raising=False,
    )

    with pytest.raises(NotImplementedError, match="forced"):
        mps_hop_module.hop_expr(left, right, list(mpos), center.shape)


def test_ir_fallback_is_explicit_and_profiled(monkeypatch):
    case = _mps_cases()[1]
    _, equation, left, right, mpos, center, _ = case
    events = []
    set_backend(
        "numpy",
        execution_policy="execution_ir",
        fallback_policy="legacy_oe",
        experimental_oe_ir=True,
    )
    monkeypatch.setattr(
        mps_hop_module,
        "build_mps_ir_hop",
        lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError("forced")),
        raising=False,
    )
    monkeypatch.setattr(
        oe_wrap_module,
        "build_experimental_einsum",
        lambda *args, **kwargs: pytest.fail(
            "specialized fallback re-entered experimental translation"
        ),
    )
    monkeypatch.setattr(profiling, "enabled", lambda: True)
    monkeypatch.setattr(
        profiling,
        "record",
        lambda event, **payload: events.append({"event": event, **payload}),
    )

    result = mps_hop_module.hop_expr(left, right, list(mpos), center.shape)(center)

    np.testing.assert_allclose(result, np.einsum(equation, left, *mpos, right, center))
    assert len(events) == 1
    fallback = events[0]
    assert fallback["event"] == "local_hv_execute"
    assert fallback["requested_policy"] == "execution_ir"
    assert fallback["actual_policy"] == "legacy_oe"
    assert fallback["planner_source"] == "opt_einsum"
    assert fallback["oe_path_hash"]
    assert fallback["actual_steps"]
    assert fallback["fallback_reason"] == "NotImplementedError: forced"


def test_execution_plan_is_built_once_for_repeated_hv_calls(monkeypatch):
    lowerer = importlib.import_module("renormalizer.backend._gemm.mps_lowering")
    calls = []
    original = lowerer.lower_einsum_path

    def recording_lower(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(lowerer, "lower_einsum_path", recording_lower)
    case = _mps_cases()[1]
    _, equation, left, right, mpos, center, _ = case
    set_backend("numpy", execution_policy="execution_ir")

    hop = mps_hop_module.hop_expr(left, right, list(mpos), center.shape)
    assert calls == [True] * 6
    first = hop(center)
    second = hop(center * 2)

    assert calls == [True] * 6
    np.testing.assert_allclose(first, np.einsum(equation, left, *mpos, right, center))
    np.testing.assert_allclose(
        second, np.einsum(equation, left, *mpos, right, center * 2)
    )


@pytest.mark.parametrize(
    ("constant_dtype", "variant_count"),
    [("float64", 6), ("float32", 12)],
)
def test_execution_variants_share_one_oe_path_search_and_never_replan(
    constant_dtype, variant_count, monkeypatch
):
    planner = importlib.import_module("renormalizer.backend._execution.planner")
    lowerer = importlib.import_module("renormalizer.backend._gemm.mps_lowering")
    rng = np.random.default_rng(65)
    left = _typed_values((2, 3, 4), constant_dtype, rng)
    mpo = _typed_values((3, 5, 4, 6), constant_dtype, rng)
    right = _typed_values((7, 6, 8), constant_dtype, rng)
    center = _typed_values((4, 4, 8), constant_dtype, rng)
    expected = np.einsum("abc,bdef,lfk,cek->adl", left, mpo, right, center)
    set_backend("numpy", execution_policy="execution_ir")
    searches = []
    lowerings = []
    original_contract_path = planner.oe.contract_path
    original_lower = lowerer.lower_einsum_path

    def recording_contract_path(*args, **kwargs):
        searches.append((args, dict(kwargs)))
        return original_contract_path(*args, **kwargs)

    def recording_lower(*args, **kwargs):
        lowerings.append((args, dict(kwargs)))
        return original_lower(*args, **kwargs)

    monkeypatch.setattr(planner.oe, "contract_path", recording_contract_path)
    monkeypatch.setattr(lowerer, "lower_einsum_path", recording_lower)

    hop = mps_hop_module.hop_expr(left, right, [mpo], center.shape)
    assert len(searches) == 1
    assert searches[0][1]["shapes"] is True
    assert searches[0][1]["optimize"] == "optimal"
    assert len(lowerings) == variant_count
    plans = tuple(hop.execution_plans.values())
    assert len(plans) == variant_count
    assert len({plan.oe_path for plan in plans}) == 1
    from renormalizer.backend._execution.profiling import oe_path_identity

    assert len({oe_path_identity(plan.oe_path) for plan in plans}) == 1
    assert len({plan.plan_hash for plan in plans}) == variant_count

    first = hop(center)
    second = hop(center * 2)

    assert len(searches) == 1
    assert len(lowerings) == variant_count
    np.testing.assert_allclose(first, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(second, expected * 2, rtol=2e-5, atol=2e-5)


def test_execution_ir_promotes_real_mpo_with_complex_environment_and_center():
    rng = np.random.default_rng(57)
    left = rng.normal(size=(2, 3, 4)) + 1j * rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 4, 6))
    right = rng.normal(size=(7, 6, 8)) + 1j * rng.normal(size=(7, 6, 8))
    center = rng.normal(size=(4, 4, 8)) + 1j * rng.normal(size=(4, 4, 8))
    set_backend("numpy", execution_policy="execution_ir")

    actual = mps_hop_module.hop_expr(left, right, [mpo], center.shape)(center)

    expected = np.einsum("abc,bdef,lfk,cek->adl", left, mpo, right, center)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_execution_ir_resolves_dtype_correct_artifact_without_executing(layout, monkeypatch):
    rng = np.random.default_rng(571)
    left = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 4, 6))
    right = rng.normal(size=(7, 6, 8))
    base = rng.normal(size=(4, 4, 16)) + 1j * rng.normal(size=(4, 4, 16))
    if layout == "C":
        center = np.array(base[..., :8], order="C", copy=True)
    elif layout == "F":
        center = np.array(base[..., :8], order="F", copy=True)
    else:
        center = base[..., ::2]
    selected = set_backend(
        "numpy", execution_policy="execution_ir", fallback_policy="legacy_oe"
    )
    hop = mps_hop_module.hop_expr(left, right, [mpo], center.shape)
    execute_calls = []
    monkeypatch.setattr(
        selected,
        "execute_plan",
        lambda *_args, **_kwargs: execute_calls.append(1),
    )

    artifact = hop.resolve_execution_artifact(center)

    assert execute_calls == []
    assert artifact.variable_index == 3
    assert artifact.variable_key not in artifact.source_bindings.arrays
    assert artifact.variable_array is center
    assert artifact.execution_plan.inputs[artifact.variable_index].key == artifact.variable_key
    assert all(
        np.dtype(array.dtype) == np.dtype("complex128")
        for array in artifact.source_bindings.arrays.values()
    )
    assert hop.legacy_fallback_expression is not None


@pytest.mark.parametrize(
    ("constant_dtype", "center_dtype", "result_dtype"),
    [
        ("float32", "float32", "float32"),
        ("float32", "float64", "float64"),
        ("float32", "complex64", "complex64"),
        ("float32", "complex128", "complex128"),
    ],
)
def test_numpy_execution_ir_matches_result_type_for_supported_dtypes(
    constant_dtype, center_dtype, result_dtype
):
    rng = np.random.default_rng(62)
    left = _typed_values((2, 3, 4), constant_dtype, rng)
    mpo = _typed_values((3, 5, 4, 6), constant_dtype, rng)
    right = _typed_values((7, 6, 8), constant_dtype, rng)
    center = _typed_values((4, 4, 8), center_dtype, rng)
    set_backend("numpy", precision=64, execution_policy="execution_ir")

    actual = mps_hop_module.hop_expr(left, right, [mpo], center.shape)(center)

    expected = oe_wrap_module.oe.contract(
        "abc,bdef,lfk,cek->adl", left, mpo, right, center, optimize="optimal"
    )
    assert np.result_type(left.dtype, mpo.dtype, right.dtype, center.dtype) == np.dtype(
        result_dtype
    )
    assert actual.dtype == expected.dtype == np.dtype(result_dtype)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_execution_ir_prebuilds_real_and_complex_plans_for_real_constants(
    monkeypatch,
):
    lowerer = importlib.import_module("renormalizer.backend._gemm.mps_lowering")
    planned_dtypes = []
    original_lower = lowerer.lower_einsum_path

    def recording_lower(*args, **kwargs):
        planned_dtypes.append(np.dtype(kwargs["dtype"]).name)
        return original_lower(*args, **kwargs)

    monkeypatch.setattr(lowerer, "lower_einsum_path", recording_lower)
    selected = set_backend("numpy", execution_policy="execution_ir")
    conversions = []
    original_array = selected.array
    rng = np.random.default_rng(58)
    left = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 4, 6))
    right = rng.normal(size=(7, 6, 8))
    constant_ids = {id(left), id(mpo), id(right)}

    def recording_array(value, *args, **kwargs):
        if id(value) in constant_ids:
            conversions.append(
                (value, kwargs.get("dtype"), kwargs.get("order"), kwargs.get("copy"))
            )
        return original_array(value, *args, **kwargs)

    monkeypatch.setattr(selected, "array", recording_array)
    real_center = rng.normal(size=(4, 4, 8))
    complex_center = real_center + 1j * rng.normal(size=real_center.shape)

    hop = mps_hop_module.hop_expr(left, right, [mpo], real_center.shape)
    assert planned_dtypes.count("float64") == 3
    assert planned_dtypes.count("complex128") == 3
    assert conversions == []
    assert not hasattr(hop, "execution_plan")
    assert hop.execution_plan_selector == (
        "key=(numpy_result_dtype_name, variable_layout_tuple)"
    )
    assert set(hop.execution_plans) == {
        (dtype, (layout,))
        for dtype in ("float64", "complex128")
        for layout in ("C", "F", "strided")
    }

    real_actual = hop(real_center)
    real_second = hop(real_center * 2)
    assert conversions == []
    complex_actual = hop(complex_center)
    assert len(conversions) == 3
    assert all(
        np.dtype(dtype).name == "complex128" and order == "C" and copy is True
        for _, dtype, order, copy in conversions
    )
    complex_second = hop(complex_center * (1 + 0.5j))

    assert len(conversions) == 3
    assert real_actual.dtype == np.dtype("float64")
    assert complex_actual.dtype == np.dtype("complex128")
    np.testing.assert_allclose(
        real_actual,
        np.einsum("abc,bdef,lfk,cek->adl", left, mpo, right, real_center),
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        real_second,
        np.einsum("abc,bdef,lfk,cek->adl", left, mpo, right, real_center * 2),
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        complex_actual,
        np.einsum("abc,bdef,lfk,cek->adl", left, mpo, right, complex_center),
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        complex_second,
        np.einsum(
            "abc,bdef,lfk,cek->adl",
            left,
            mpo,
            right,
            complex_center * (1 + 0.5j),
        ),
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_execution_ir_runtime_center_layout_is_explicit_plan_metadata(
    layout, monkeypatch
):
    rng = np.random.default_rng(63)
    left = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 4, 6))
    right = rng.normal(size=(7, 6, 8))
    base = rng.normal(size=(4, 4, 16))
    if layout == "C":
        center = np.array(base[..., :8], order="C", copy=True)
    elif layout == "F":
        center = np.array(base[..., :8], order="F", copy=True)
    else:
        center = base[..., ::2]
        assert not center.flags.c_contiguous and not center.flags.f_contiguous
    selected = set_backend("numpy", execution_policy="execution_ir")
    hop = mps_hop_module.hop_expr(left, right, [mpo], center.shape)
    plan = hop.execution_plans[("float64", (layout,))]
    assert plan.inputs[-1].spec.layout == layout
    if layout != "C":
        assert any(
            isinstance(step, TransformStep) and step.input == plan.inputs[-1]
            for step in plan.steps
        )
    array_calls = []
    original_array = selected.array

    def recording_array(*args, **kwargs):
        array_calls.append((args, kwargs))
        return original_array(*args, **kwargs)

    monkeypatch.setattr(selected, "array", recording_array)

    actual = hop(center)

    explicit_copies = sum(
        isinstance(step, TransformStep) and step.copy for step in plan.steps
    )
    assert len(array_calls) == explicit_copies
    np.testing.assert_allclose(
        actual,
        np.einsum("abc,bdef,lfk,cek->adl", left, mpo, right, center),
        rtol=1e-11,
        atol=1e-11,
    )


def test_unsupported_runtime_dtype_uses_explicit_legacy_fallback(monkeypatch):
    rng = np.random.default_rng(59)
    left = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 4, 6))
    right = rng.normal(size=(7, 6, 8))
    center = rng.normal(size=(4, 4, 8)).astype(np.longdouble)
    events = []
    planner_calls = []
    legacy_builds = []
    path_calls = []
    lowerer = importlib.import_module("renormalizer.backend._gemm.mps_lowering")
    planner = importlib.import_module("renormalizer.backend._execution.planner")
    oe_contract_module = importlib.import_module("opt_einsum.contract")
    original_lower = lowerer.lower_einsum_path
    original_contract_expression = oe_wrap_module.oe.contract_expression
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
    hop = mps_hop_module.hop_expr(left, right, [mpo], center.shape)
    planned = tuple(planner_calls)
    assert len(planned) == 6
    assert legacy_builds == [True]
    assert sum(call[1].get("optimize") == "optimal" for call in path_calls) == 1
    assert len(path_calls) == 2
    assert path_calls[1][1]["optimize"] == hop.resolved_oe_path
    constructed_path_calls = tuple(path_calls)

    actual = hop(center)
    second = hop(center * 2)

    expected = np.einsum("abc,bdef,lfk,cek->adl", left, mpo, right, center)
    reason = (
        "UnsupportedRuntimeDtypeError: execution IR runtime result dtype 'float128' "
        "with layouts ('C',) has no configured variant; available variants: "
        "complex128/C, complex128/F, complex128/strided, float64/C, float64/F, "
        "float64/strided"
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(second, expected * 2, rtol=1e-11, atol=1e-11)
    assert actual.dtype == expected.dtype
    assert tuple(planner_calls) == planned
    assert legacy_builds == [True]
    assert tuple(path_calls) == constructed_path_calls
    assert len(events) == 2
    assert all(event["requested_policy"] == "execution_ir" for event in events)
    assert all(event["actual_policy"] == "legacy_oe" for event in events)
    assert all(event["fallback_reason"] == reason for event in events)
    from renormalizer.backend._execution.profiling import oe_path_identity

    assert all(
        event["oe_path_hash"] == oe_path_identity(hop.resolved_oe_path)
        for event in events
    )


def test_unsupported_runtime_dtype_is_clear_under_strict_policy():
    rng = np.random.default_rng(60)
    left = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 4, 6))
    right = rng.normal(size=(7, 6, 8))
    center = rng.normal(size=(4, 4, 8)).astype(np.longdouble)
    set_backend("numpy", execution_policy="execution_ir", fallback_policy="error")
    hop = mps_hop_module.hop_expr(left, right, [mpo], center.shape)

    with pytest.raises(NotImplementedError) as raised:
        hop(center)
    assert str(raised.value).startswith(
        "execution IR runtime result dtype 'float128' with layouts ('C',) "
        "has no configured variant; available variants:"
    )


def test_experimental_variant_bound_precedes_cartesian_and_path_search(monkeypatch):
    experimental = importlib.import_module(
        "renormalizer.backend._gemm.experimental_einsum"
    )
    planner = importlib.import_module("renormalizer.backend._execution.planner")
    constant = np.ones((2, 2), dtype=np.float32)
    shapes = ((2, 2),) * 3
    set_backend(
        "numpy",
        execution_policy="execution_ir",
        fallback_policy="error",
        experimental_oe_ir=True,
    )
    monkeypatch.setattr(
        experimental.itertools,
        "product",
        lambda *args, **kwargs: pytest.fail(
            "cartesian layouts materialized before variant bound"
        ),
    )
    monkeypatch.setattr(
        planner.oe,
        "contract_path",
        lambda *args, **kwargs: pytest.fail("OE path searched before variant bound"),
    )
    monkeypatch.setattr(
        experimental,
        "lower_einsum_path",
        lambda *args, **kwargs: pytest.fail("lowering ran before variant bound"),
    )

    with pytest.raises(
        NotImplementedError, match="108 execution variants exceeds bounded limit 48"
    ):
        oe_wrap_module.oe_contract_expression(
            "ab,bc,cd,de->ae", constant, *shapes, constants=[0]
        )


def test_experimental_layout_cartesian_is_bounded_at_construction():
    constant = np.ones((2, 2), dtype=np.float32)
    shapes = ((2, 2),) * 3
    set_backend(
        "numpy",
        execution_policy="execution_ir",
        fallback_policy="error",
        experimental_oe_ir=True,
    )

    with pytest.raises(
        NotImplementedError, match="108 execution variants exceeds bounded limit 48"
    ):
        oe_wrap_module.oe_contract_expression(
            "ab,bc,cd,de->ae", constant, *shapes, constants=[0]
        )

    set_backend(
        "numpy",
        execution_policy="execution_ir",
        fallback_policy="legacy_oe",
        experimental_oe_ir=True,
    )
    expression = oe_wrap_module.oe_contract_expression(
        "ab,bc,cd,de->ae", constant, *shapes, constants=[0]
    )
    variables = tuple(np.eye(2, dtype=np.float32) for _ in shapes)
    np.testing.assert_allclose(
        expression(*variables), np.einsum("ab,bc,cd,de->ae", constant, *variables)
    )


def test_generic_runtime_dtype_fallback_reason_is_exact_and_reuses_oe(monkeypatch):
    constant = np.arange(6.0).reshape(2, 3)
    variable = np.arange(12.0, dtype=np.longdouble).reshape(3, 4)
    lowerer = importlib.import_module(
        "renormalizer.backend._gemm.experimental_einsum"
    )
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
        "numpy",
        execution_policy="execution_ir",
        fallback_policy="legacy_oe",
        experimental_oe_ir=True,
    )
    monkeypatch.setattr(profiling, "enabled", lambda: True)
    monkeypatch.setattr(
        profiling,
        "record",
        lambda event, **payload: events.append({"event": event, **payload}),
    )

    expression = oe_wrap_module.oe_contract_expression(
        "ab,bc->ac",
        constant,
        variable.shape,
        constants=[0],
        _profile_network="mps",
        _profile_center_kind="one_site",
    )
    planned = tuple(planner_calls)
    assert len(planned) == 6
    assert legacy_builds == [True]
    assert sum(call[1].get("optimize") == "optimal" for call in path_calls) == 1
    assert len(path_calls) == 2
    assert path_calls[1][1]["optimize"] == expression.resolved_oe_path
    constructed_path_calls = tuple(path_calls)
    first = expression(variable)
    second = expression(variable * 2)

    reason = (
        "UnsupportedRuntimeDtypeError: execution IR runtime result dtype 'float128' "
        "with layouts ('C',) has no configured variant; available variants: "
        "complex128/C, complex128/F, complex128/strided, float64/C, float64/F, "
        "float64/strided"
    )
    np.testing.assert_allclose(first, constant @ variable)
    np.testing.assert_allclose(second, (constant @ variable) * 2)
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


def test_generic_two_variable_fallback_profiles_all_operand_shapes(monkeypatch):
    left = np.arange(6.0, dtype=np.longdouble).reshape(2, 3)
    constant = np.arange(12.0).reshape(3, 4)
    right = np.arange(20.0, dtype=np.longdouble).reshape(4, 5)
    lowerer = importlib.import_module(
        "renormalizer.backend._gemm.experimental_einsum"
    )
    planner = importlib.import_module("renormalizer.backend._execution.planner")
    oe_contract_module = importlib.import_module("opt_einsum.contract")
    planner_calls = []
    path_calls = []
    legacy_builds = []
    events = []
    original_lower = lowerer.lower_einsum_path
    original_contract_path = planner.oe.contract_path
    original_contract_expression = oe_wrap_module.oe.contract_expression

    def recording_lower(*args, **kwargs):
        planner_calls.append((kwargs["dtype"], kwargs["layouts"]))
        return original_lower(*args, **kwargs)

    def recording_contract_path(*args, **kwargs):
        path_calls.append((args, dict(kwargs)))
        return original_contract_path(*args, **kwargs)

    def recording_contract_expression(*args, **kwargs):
        legacy_builds.append(True)
        return original_contract_expression(*args, **kwargs)

    monkeypatch.setattr(lowerer, "lower_einsum_path", recording_lower)
    monkeypatch.setattr(planner.oe, "contract_path", recording_contract_path)
    monkeypatch.setattr(
        oe_contract_module, "contract_path", recording_contract_path
    )
    monkeypatch.setattr(
        oe_wrap_module.oe, "contract_expression", recording_contract_expression
    )
    set_backend(
        "numpy",
        execution_policy="execution_ir",
        fallback_policy="legacy_oe",
        experimental_oe_ir=True,
    )
    monkeypatch.setattr(profiling, "enabled", lambda: True)
    monkeypatch.setattr(
        profiling,
        "record",
        lambda event, **payload: events.append({"event": event, **payload}),
    )

    expression = oe_wrap_module.oe_contract_expression(
        "ab,bc,cd->ad",
        left.shape,
        constant,
        right.shape,
        constants=[1],
        _profile_network="mps",
        _profile_center_kind="one_site",
    )
    planned = tuple(planner_calls)
    constructed_paths = tuple(path_calls)
    assert len(planned) == 18
    assert legacy_builds == [True]
    assert sum(call[1].get("optimize") == "optimal" for call in path_calls) == 1
    assert len(path_calls) == 2

    first = expression(left, right)
    second = expression(left * 2, right)

    expected = np.einsum("ab,bc,cd->ad", left, constant, right)
    layouts = ("C", "F", "strided")
    available = ", ".join(
        "{}/{}+{}".format(dtype, left_layout, right_layout)
        for dtype in ("complex128", "float64")
        for left_layout in layouts
        for right_layout in layouts
    )
    reason = (
        "UnsupportedRuntimeDtypeError: execution IR runtime result dtype 'float128' "
        "with layouts ('C', 'C') has no configured variant; available variants: "
        + available
    )
    np.testing.assert_allclose(first, expected)
    np.testing.assert_allclose(second, expected * 2)
    assert tuple(planner_calls) == planned
    assert tuple(path_calls) == constructed_paths
    assert legacy_builds == [True]
    assert len(events) == 2
    assert all(event["requested_policy"] == "execution_ir" for event in events)
    assert all(event["actual_policy"] == "legacy_oe" for event in events)
    assert all(event["fallback_reason"] == reason for event in events)
    assert all(
        event["input_shapes"]
        == [list(left.shape), list(constant.shape), list(right.shape)]
        for event in events
    )
    assert all(event["input_shape_count"] == 3 for event in events)
    assert all(event["oe_path_hash"] for event in events)
    assert all(event["actual_steps"] for event in events)


def test_profile_off_builds_no_timing_or_payload_metadata(monkeypatch):
    helper_module = "renormalizer.backend._execution.profiling"
    sys.modules.pop(helper_module, None)
    monkeypatch.setattr(profiling, "enabled", lambda: False)
    monkeypatch.setattr(
        profiling,
        "record",
        lambda *args, **kwargs: pytest.fail("disabled profiling recorded an event"),
    )
    case = _mps_cases()[1]

    actual, expected = _run_case(case, "execution_ir")

    np.testing.assert_allclose(actual, expected)
    assert helper_module not in sys.modules


def test_execution_ir_telemetry_reports_actual_plan(monkeypatch):
    events = []
    monkeypatch.setattr(profiling, "enabled", lambda: True)
    monkeypatch.setattr(
        profiling,
        "record",
        lambda event, **payload: events.append({"event": event, **payload}),
    )

    _run_case(_mps_cases()[1], "execution_ir")

    assert len(events) == 1
    event = events[0]
    assert event["event"] == "local_hv_execute"
    assert event["requested_policy"] == "execution_ir"
    assert event["actual_policy"] == "execution_ir"
    assert event["planner_source"] == "opt_einsum"
    assert event["oe_path_hash"]
    assert event["actual_steps"]
    assert event["path_override_reason"] is None
    assert event["fallback_reason"] is None


def test_experimental_oe_translation_requires_explicit_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(
        oe_wrap_module,
        "build_experimental_einsum",
        lambda *args, **kwargs: calls.append(True),
        raising=False,
    )
    left = np.arange(6.0).reshape(2, 3)
    right = np.arange(12.0).reshape(3, 4)
    set_backend("numpy", execution_policy="execution_ir", experimental_oe_ir=False)

    expr = oe_wrap_module.oe_contract_expression(
        "ab,bc->ac", left, right.shape, constants=[0]
    )
    actual = expr(right)

    assert calls == []
    np.testing.assert_allclose(actual, left @ right)


def test_experimental_oe_translation_is_numerical_and_reuses_plan(monkeypatch):
    lowerer = importlib.import_module("renormalizer.backend._gemm.experimental_einsum")
    calls = []
    original = lowerer.lower_einsum_path

    def recording_lower(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(lowerer, "lower_einsum_path", recording_lower)
    left = np.arange(6.0).reshape(2, 3)
    right = np.arange(12.0).reshape(3, 4)
    set_backend("numpy", execution_policy="execution_ir", experimental_oe_ir=True)

    expr = oe_wrap_module.oe_contract_expression(
        "ab,bc->ac", left, right.shape, constants=[0]
    )
    assert calls == [True] * 6

    np.testing.assert_allclose(expr(right), left @ right)
    np.testing.assert_allclose(expr(right * 2), left @ (right * 2))
    assert calls == [True] * 6


@pytest.mark.skipif(
    importlib.util.find_spec("cupy") is None, reason="CuPy is not installed"
)
def test_cupy_execution_ir_hv_is_device_resident_and_numerical():
    try:
        selected = set_backend(
            "cupy", device="cuda:0", precision=64, execution_policy="execution_ir"
        )
    except Exception as error:
        pytest.skip("CuPy/CUDA is unavailable: {}".format(error))
    case = _mps_cases()[1]
    _, equation, left, right, mpos, center, _ = case
    hop = mps_hop_module.hop_expr(left, right, list(mpos), center.shape)

    actual = hop(selected.asarray(center))

    assert isinstance(actual, selected.ndarray)
    assert actual.device.id == 0
    np.testing.assert_allclose(
        selected.to_numpy(actual), np.einsum(equation, left, *mpos, right, center)
    )


@pytest.mark.skipif(
    importlib.util.find_spec("cupy") is None, reason="CuPy is not installed"
)
@pytest.mark.parametrize(
    ("constant_dtype", "center_dtype", "result_dtype"),
    [
        ("float32", "float32", "float32"),
        ("float32", "float64", "float64"),
        ("float32", "complex64", "complex64"),
        ("float32", "complex128", "complex128"),
    ],
)
def test_cupy_execution_ir_matches_result_type_without_host_transfer(
    constant_dtype, center_dtype, result_dtype, monkeypatch
):
    try:
        selected = set_backend(
            "cupy", device="cuda:0", precision=64, execution_policy="execution_ir"
        )
    except Exception as error:
        pytest.skip("CuPy/CUDA is unavailable: {}".format(error))
    cp = selected._cupy
    rng = np.random.default_rng(61)
    left = _typed_values((2, 3, 4), constant_dtype, rng)
    mpo = _typed_values((3, 5, 4, 6), constant_dtype, rng)
    right = _typed_values((7, 6, 8), constant_dtype, rng)
    center = _typed_values((4, 4, 8), center_dtype, rng)
    expected = np.einsum("abc,bdef,lfk,cek->adl", left, mpo, right, center)
    monkeypatch.setattr(
        selected,
        "to_numpy",
        lambda value: pytest.fail("mixed-dtype CuPy H-v transferred to host"),
    )

    hop = mps_hop_module.hop_expr(left, right, [mpo], center.shape)
    actual = hop(selected.asarray(center))

    assert isinstance(actual, cp.ndarray)
    assert actual.device.id == 0
    assert actual.dtype == expected.dtype == cp.dtype(result_dtype)
    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(
    importlib.util.find_spec("cupy") is None, reason="CuPy is not installed"
)
def test_cupy_constant_dtype_copies_are_lazy_cached_and_device_resident(monkeypatch):
    try:
        selected = set_backend(
            "cupy", device="cuda:0", precision=64, execution_policy="execution_ir"
        )
    except Exception as error:
        pytest.skip("CuPy/CUDA is unavailable: {}".format(error))
    cp = selected._cupy
    rng = np.random.default_rng(64)
    left = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 4, 6))
    right = rng.normal(size=(7, 6, 8))
    real_center = selected.asarray(rng.normal(size=(4, 4, 8)))
    complex_center = real_center.astype(cp.complex128) * (1 + 0.25j)
    complex_copies = []
    original_array = selected.array

    def recording_array(value, *args, **kwargs):
        if kwargs.get("dtype") is not None and np.dtype(kwargs["dtype"]).name == "complex128":
            complex_copies.append((value, kwargs.get("order"), kwargs.get("copy")))
        return original_array(value, *args, **kwargs)

    monkeypatch.setattr(selected, "array", recording_array)
    monkeypatch.setattr(
        selected,
        "to_numpy",
        lambda value: pytest.fail("mixed-dtype CuPy H-v transferred to host"),
    )

    hop = mps_hop_module.hop_expr(left, right, [mpo], real_center.shape)
    assert complex_copies == []
    hop(real_center)
    assert complex_copies == []
    first = hop(complex_center)
    assert len(complex_copies) == 3
    assert all(order == "C" and copy is True for _, order, copy in complex_copies)
    second = hop(complex_center * 2)

    assert len(complex_copies) == 3
    assert isinstance(first, cp.ndarray)
    assert isinstance(second, cp.ndarray)
    cp.testing.assert_allclose(second, first * 2, rtol=1e-11, atol=1e-11)


def test_sbm_one_step_benchmark_emits_comparable_compact_json():
    payloads = []
    for policy in ("legacy_oe", "execution_ir"):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "renormalizer.backend.example_benchmark",
                "--case",
                "sbm",
                "--execution-policy",
                policy,
                "--steps",
                "1",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        assert len(lines) == 1
        assert " " not in lines[0]
        assert not any(
            marker in completed.stderr
            for marker in ("Warning", "WARNING", "ERROR", "Traceback")
        )
        payload = json.loads(lines[0])
        assert payload["case"] == "sbm"
        assert payload["execution_policy"] == policy
        assert payload["evolve_dt"] == pytest.approx(0.05)
        assert payload["evolve_time"] == pytest.approx(0.05)
        assert payload["steps_requested"] == payload["steps_executed"] == 1
        assert payload["profiling_enabled"] is False
        assert payload["wall_s"] >= 0
        payloads.append(payload)

    assert payloads[0]["observable"] == pytest.approx(payloads[1]["observable"])
    assert payloads[0]["state_norm"] == pytest.approx(payloads[1]["state_norm"])


def test_sbm_benchmark_rejects_enabled_profiling_before_model_work():
    environment = dict(os.environ)
    environment["RENO_LOG_LEVEL"] = "5"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "renormalizer.backend.example_benchmark",
            "--case",
            "sbm",
            "--execution-policy",
            "execution_ir",
            "--steps",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "profiling must be disabled" in completed.stderr
    assert "Creating TDMPS job" not in completed.stderr
    assert "# of operator terms" not in completed.stderr
