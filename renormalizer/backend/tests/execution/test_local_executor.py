import numpy as np
import pytest

from renormalizer import set_backend
from renormalizer.backend._execution.executor import execute_plan
from renormalizer.backend._execution.model import (
    BufferRef,
    ExecutionBindings,
    ExecutionPlan,
    GroupedMatmulStep,
    MatmulStep,
    ReductionStep,
    TensorSpec,
    TransformStep,
    _plan_hash,
)
from renormalizer.backend._execution.planner import plan_einsum
from renormalizer.backend._execution.workspace import workspace_bytes_for_steps


@pytest.fixture(autouse=True)
def restore_numpy_backend():
    try:
        yield
    finally:
        set_backend("numpy", precision=64)


def _numpy_backend():
    return set_backend("numpy", precision=64)


def _specialized_plan(inputs, output, steps):
    workspace_bytes = workspace_bytes_for_steps(steps, output.key)
    values = dict(
        operation="einsum",
        inputs=tuple(inputs),
        output=output,
        steps=tuple(steps),
        workspace_bytes=workspace_bytes,
        planner_source="specialized",
        oe_path=tuple((0, 1) for _ in range(len(inputs) - 1)),
        override_reason="executor test fixture",
    )
    values["plan_hash"] = _plan_hash(**values)
    return ExecutionPlan(**values)


def _ref(key, shape, modes, dtype="float64", layout="C"):
    return BufferRef(key, TensorSpec(shape, dtype, layout, modes))


@pytest.mark.parametrize("shape", [(), (4,), (1, 4, 1), (0, 4)])
def test_dual_contiguous_f_binding_and_copied_transform_output(shape):
    modes = tuple("abc"[:len(shape)])
    source = np.empty(shape, dtype=np.float64, order="F")
    source[...] = 2.5
    assert source.flags.c_contiguous
    assert source.flags.f_contiguous
    input_ref = _ref("input_0", shape, modes, layout="F")
    output_ref = _ref("output", shape, modes, layout="F")
    plan = _specialized_plan(
        (input_ref,),
        output_ref,
        (TransformStep(input_ref, output_ref, tuple(range(len(shape))), True),),
    )

    actual = execute_plan(
        _numpy_backend(), plan, ExecutionBindings({"input_0": source})
    )

    np.testing.assert_array_equal(actual, source)
    assert actual is not source
    assert actual.flags.f_contiguous


@pytest.mark.parametrize(
    ("source", "input_modes", "output_shape", "output_modes", "reduced_modes"),
    [
        (np.arange(3.0), ("a",), (), (), ("a",)),
        (np.empty((0, 3)), ("a", "b"), (0,), ("a",), ("b",)),
    ],
)
def test_dual_contiguous_f_planned_reduction_output(
    source, input_modes, output_shape, output_modes, reduced_modes
):
    input_ref = _ref("input_0", source.shape, input_modes)
    output_ref = _ref("output", output_shape, output_modes, layout="F")
    plan = _specialized_plan(
        (input_ref,),
        output_ref,
        (ReductionStep(input_ref, output_ref, reduced_modes),),
    )

    actual = execute_plan(
        _numpy_backend(), plan, ExecutionBindings({"input_0": source})
    )

    axes = tuple(input_modes.index(mode) for mode in reduced_modes)
    np.testing.assert_array_equal(actual, source.sum(axis=axes))
    assert actual.flags.f_contiguous


@pytest.mark.parametrize(
    ("expected_layout", "actual_layout"),
    [("F", "C"), ("C", "F"), ("strided", "C")],
)
def test_layout_predicate_rejects_genuine_binding_mismatch(
    expected_layout, actual_layout
):
    if actual_layout == "F":
        source = np.asfortranarray(np.arange(6.0).reshape(2, 3))
        assert source.flags.f_contiguous and not source.flags.c_contiguous
    else:
        source = np.arange(6.0).reshape(2, 3)
        assert source.flags.c_contiguous and not source.flags.f_contiguous
    input_ref = _ref(
        "input_0", source.shape, ("a", "b"), layout=expected_layout
    )
    output_ref = _ref("output", source.shape, ("a", "b"))
    plan = _specialized_plan(
        (input_ref,),
        output_ref,
        (TransformStep(input_ref, output_ref, (0, 1), True),),
    )

    with pytest.raises(ValueError, match="planned layout"):
        execute_plan(
            _numpy_backend(), plan, ExecutionBindings({"input_0": source})
        )


def test_plan_einsum_keeps_c_first_dual_contiguous_canonicalization():
    left = np.arange(4.0)
    right = np.arange(4.0)
    assert left.flags.c_contiguous and left.flags.f_contiguous

    plan, _ = plan_einsum("a,a->", {"left": left, "right": right})

    assert tuple(ref.spec.layout for ref in plan.inputs) == ("C", "C")


def test_numpy_executor_matches_required_equation():
    rng = np.random.default_rng(11)
    arrays = {
        "left": rng.normal(size=(2, 3, 4)),
        "mpo": rng.normal(size=(3, 5, 4, 6)),
        "right": rng.normal(size=(7, 6, 8)),
        "center": rng.normal(size=(4, 4, 8)),
    }
    plan, bindings = plan_einsum("abc,bdef,lfk,cek->adl", arrays)

    actual = execute_plan(_numpy_backend(), plan, bindings)

    expected = np.einsum("abc,bdef,lfk,cek->adl", *arrays.values())
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_executor_handles_complex_noncontiguous_inputs():
    rng = np.random.default_rng(12)
    a0 = rng.normal(size=(4, 3, 2)) + 1j * rng.normal(size=(4, 3, 2))
    b0 = rng.normal(size=(5, 4)) + 1j * rng.normal(size=(5, 4))
    a = a0.transpose(2, 1, 0)
    b = b0.T
    plan, bindings = plan_einsum("abc,cd->abd", {"a": a, "b": b})

    actual = _numpy_backend().execute_plan(plan, bindings)

    np.testing.assert_allclose(actual, np.einsum("abc,cd->abd", a, b))
    assert actual.dtype == np.dtype("complex128")
    assert actual.flags.c_contiguous


def test_executor_runs_explicit_pack_and_output_reorder():
    left = np.arange(24.0).reshape(2, 3, 4)
    right = np.arange(90.0).reshape(3, 5, 6)
    plan, bindings = plan_einsum("abc,bde->adce", {"left": left, "right": right})

    assert [type(step).__name__ for step in plan.steps] == [
        "TransformStep",
        "MatmulStep",
        "TransformStep",
    ]
    actual = execute_plan(_numpy_backend(), plan, bindings)

    np.testing.assert_allclose(actual, np.einsum("abc,bde->adce", left, right))
    assert actual.flags.c_contiguous


def test_executor_reduces_operand_only_modes_with_planned_dtype():
    left = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    right = np.arange(20, dtype=np.float32).reshape(4, 5)
    plan, bindings = plan_einsum("abc,cd->ad", {"left": left, "right": right})

    actual = execute_plan(_numpy_backend(), plan, bindings)

    np.testing.assert_allclose(actual, np.einsum("abc,cd->ad", left, right))
    assert actual.dtype == np.dtype("float32")


@pytest.mark.parametrize("layout", ["C", "F"])
def test_reduction_allocates_once_in_exact_planned_layout(monkeypatch, layout):
    source = np.asfortranarray(np.arange(24.0).reshape(2, 3, 4))
    input_ref = _ref(
        "input_0", source.shape, ("a", "b", "c"), layout="F"
    )
    output_ref = _ref("output", (2, 3), ("a", "b"), layout=layout)
    plan = _specialized_plan(
        (input_ref,), output_ref, (ReductionStep(input_ref, output_ref, ("c",)),)
    )
    backend = _numpy_backend()
    original_empty = backend.empty
    original_sum = backend.sum
    allocations = []
    sum_outputs = []

    def recording_empty(shape, dtype, order):
        output = original_empty(shape, dtype=dtype, order=order)
        allocations.append((shape, np.dtype(dtype).name, order, output))
        return output

    def recording_sum(value, *, out=None, axis=None, dtype=None):
        sum_outputs.append(out)
        return original_sum(value, out=out, axis=axis, dtype=dtype)

    monkeypatch.setattr(backend, "empty", recording_empty)
    monkeypatch.setattr(backend, "sum", recording_sum)

    actual = execute_plan(
        backend, plan, ExecutionBindings({"input_0": source})
    )

    np.testing.assert_array_equal(actual, source.sum(axis=2))
    assert allocations == [((2, 3), "float64", layout, actual)]
    assert sum_outputs == [actual]
    assert actual.flags.c_contiguous if layout == "C" else actual.flags.f_contiguous


@pytest.mark.parametrize(
    ("source", "input_modes", "output_shape", "output_modes", "reduced_modes"),
    [
        (np.empty((0, 3)), ("a", "b"), (3,), ("b",), ("a",)),
        (np.arange(3.0), ("a",), (), (), ("a",)),
    ],
)
def test_reduction_handles_zero_extent_and_scalar_output(
    source, input_modes, output_shape, output_modes, reduced_modes
):
    input_ref = _ref("input_0", source.shape, input_modes)
    output_ref = _ref("output", output_shape, output_modes)
    plan = _specialized_plan(
        (input_ref,),
        output_ref,
        (ReductionStep(input_ref, output_ref, reduced_modes),),
    )

    actual = execute_plan(
        _numpy_backend(), plan, ExecutionBindings({"input_0": source})
    )

    axes = tuple(input_modes.index(mode) for mode in reduced_modes)
    np.testing.assert_array_equal(actual, source.sum(axis=axes))
    assert actual.shape == output_shape


def test_reduction_allocation_memory_error_propagates_before_sum(monkeypatch):
    source = np.arange(6.0).reshape(2, 3)
    input_ref = _ref("input_0", source.shape, ("a", "b"))
    output_ref = _ref("output", (2,), ("a",))
    plan = _specialized_plan(
        (input_ref,), output_ref, (ReductionStep(input_ref, output_ref, ("b",)),)
    )
    backend = _numpy_backend()

    def fail_empty(*args, **kwargs):
        raise MemoryError("planned reduction allocation failed")

    monkeypatch.setattr(backend, "empty", fail_empty)
    monkeypatch.setattr(
        backend,
        "sum",
        lambda *args, **kwargs: pytest.fail("sum ran after allocation failure"),
    )

    with pytest.raises(MemoryError, match="planned reduction allocation failed"):
        execute_plan(
            backend, plan, ExecutionBindings({"input_0": source})
        )


def test_executor_collapses_multiple_batch_free_and_contracted_modes():
    rng = np.random.default_rng(13)
    left = rng.normal(size=(2, 3, 4, 5, 6))
    right = rng.normal(size=(2, 3, 5, 6, 7))
    plan, bindings = plan_einsum("abcde,abdef->abcf", {"left": left, "right": right})

    actual = execute_plan(_numpy_backend(), plan, bindings)

    np.testing.assert_allclose(actual, np.einsum("abcde,abdef->abcf", left, right))


@pytest.mark.parametrize(
    ("equation", "arrays"),
    [
        ("ab,bc->ac", {"a": np.empty((2, 0)), "b": np.empty((0, 3))}),
        ("a,a->", {"a": np.arange(4.0), "b": np.arange(4.0)}),
    ],
)
def test_executor_handles_zero_extent_and_scalar_output(equation, arrays):
    plan, bindings = plan_einsum(equation, arrays)

    actual = execute_plan(_numpy_backend(), plan, bindings)

    np.testing.assert_allclose(actual, np.einsum(equation, *arrays.values()))
    assert actual.shape == plan.output.spec.shape


@pytest.mark.parametrize(
    ("equation", "arrays"),
    [
        (
            "ab,bc->ac",
            {
                "left": np.arange(6.0).reshape(2, 3),
                "right": np.arange(12.0).reshape(3, 4),
            },
        ),
        ("a,a->", {"left": np.arange(4.0), "right": np.arange(4.0)}),
    ],
)
def test_exact_reshape_identity_accepts_ordinary_and_scalar_views(
    equation, arrays
):
    plan, bindings = plan_einsum(equation, arrays)

    actual = execute_plan(_numpy_backend(), plan, bindings)

    np.testing.assert_allclose(actual, np.einsum(equation, *arrays.values()))


def test_exact_reshape_identity_accepts_empty_view(monkeypatch):
    left = np.empty((2, 0))
    right = np.empty((0, 3))
    plan, bindings = plan_einsum(
        "ab,bc->ac", {"left": left, "right": right}
    )
    backend = _numpy_backend()
    original_matmul = backend.matmul
    matmul_calls = []

    def recording_matmul(a, b, *, stream=None, workspace=None):
        matmul_calls.append((a, b))
        return original_matmul(a, b, stream=stream, workspace=workspace)

    monkeypatch.setattr(backend, "matmul", recording_matmul)

    actual = execute_plan(backend, plan, bindings)

    np.testing.assert_array_equal(actual, np.zeros((2, 3)))
    assert len(matmul_calls) == 1


def _assert_numpy_reshape_rejected_before_matmul(
    monkeypatch, left, right, replacement, equation="ab,bc->ac"
):
    plan, bindings = plan_einsum(
        equation, {"left": left, "right": right}
    )
    backend = _numpy_backend()
    original_reshape = backend.reshape
    matmul_calls = []

    def injected_reshape(value, shape):
        if value is left:
            assert replacement.shape == tuple(shape)
            return replacement
        return original_reshape(value, shape)

    def fail_matmul(*args, **kwargs):
        matmul_calls.append(True)
        raise AssertionError("matmul ran after invalid reshape alias")

    monkeypatch.setattr(backend, "reshape", injected_reshape)
    monkeypatch.setattr(backend, "matmul", fail_matmul)

    with pytest.raises(ValueError, match="reshape unexpectedly copied"):
        execute_plan(backend, plan, bindings)
    assert matmul_calls == []


def test_exact_reshape_identity_rejects_zero_stride_overlap(monkeypatch):
    left = np.arange(6.0).reshape(2, 1, 3)
    right = np.arange(12.0).reshape(3, 4)
    overlapping = np.lib.stride_tricks.as_strided(
        left, shape=(2, 3), strides=(0, 0)
    )
    assert np.shares_memory(left, overlapping)
    assert not overlapping.flags.c_contiguous

    _assert_numpy_reshape_rejected_before_matmul(
        monkeypatch, left, right, overlapping, equation="abc,cd->abd"
    )


@pytest.mark.parametrize("target", [(1, 1), (1, 0)])
def test_degenerate_zero_stride_reshape_rejected_before_matmul(
    monkeypatch, target
):
    if target == (1, 1):
        left = np.empty((1,), dtype=np.float64)
        right = np.empty((1,), dtype=np.float64)
        equation = "a,a->"
    else:
        left = np.empty((1, 0, 1), dtype=np.float64)
        right = np.empty((0, 1, 1), dtype=np.float64)
        equation = "abc,bcd->ad"
    zero_stride = np.lib.stride_tricks.as_strided(
        left, shape=target, strides=(0, 0)
    )
    assert zero_stride.strides == (0, 0)
    assert zero_stride.flags.c_contiguous
    assert np.shares_memory(left, zero_stride) or left.size == 0

    _assert_numpy_reshape_rejected_before_matmul(
        monkeypatch, left, right, zero_stride, equation=equation
    )


def test_exact_reshape_identity_rejects_partial_offset_overlap(monkeypatch):
    backing = np.arange(7.0)
    left = backing[:6].reshape(2, 1, 3)
    right = np.arange(12.0).reshape(3, 4)
    offset = backing[1:].reshape(2, 3)
    assert np.shares_memory(left, offset)
    assert left.flags.c_contiguous and offset.flags.c_contiguous

    _assert_numpy_reshape_rejected_before_matmul(
        monkeypatch, left, right, offset, equation="abc,cd->abd"
    )


def test_exact_reshape_identity_rejects_unrelated_empty_view(monkeypatch):
    left = np.empty((1, 0, 1))
    right = np.empty((0, 1, 1))
    unrelated = np.empty((1, 0), dtype=left.dtype).view()

    _assert_numpy_reshape_rejected_before_matmul(
        monkeypatch, left, right, unrelated, equation="abc,bcd->ad"
    )


@pytest.mark.parametrize("layout", ["C", "F"])
def test_copying_transform_materializes_exact_layout_and_distinct_array(layout):
    source = np.arange(24.0).reshape(2, 3, 4)
    input_ref = _ref("input_0", source.shape, ("a", "b", "c"))
    output_shape = (4, 3, 2)
    output_ref = _ref("output", output_shape, ("c", "b", "a"), layout=layout)
    plan = _specialized_plan(
        (input_ref,), output_ref, (TransformStep(input_ref, output_ref, (2, 1, 0), True),)
    )

    actual = execute_plan(
        _numpy_backend(), plan, ExecutionBindings({"input_0": source})
    )

    np.testing.assert_array_equal(actual, source.transpose(2, 1, 0))
    assert not np.shares_memory(actual, source)
    assert actual.flags.c_contiguous if layout == "C" else actual.flags.f_contiguous


def test_noncopying_transform_preserves_view():
    source = np.arange(24.0).reshape(2, 3, 4)
    input_ref = _ref("input_0", source.shape, ("a", "b", "c"))
    output_ref = _ref("output", (4, 3, 2), ("c", "b", "a"), layout="F")
    plan = _specialized_plan(
        (input_ref,), output_ref, (TransformStep(input_ref, output_ref, (2, 1, 0), False),)
    )

    actual = execute_plan(
        _numpy_backend(), plan, ExecutionBindings({"input_0": source})
    )

    assert np.shares_memory(actual, source)
    actual[0, 0, 0] = -1
    assert source[0, 0, 0] == -1


def test_workspace_capacity_is_preflighted_before_any_primitive(monkeypatch):
    left = np.arange(24.0).reshape(2, 3, 4)
    right = np.arange(90.0).reshape(3, 5, 6)
    plan, bindings = plan_einsum("abc,bde->adce", {"left": left, "right": right})
    backend = _numpy_backend()
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise AssertionError("primitive must not run")

    monkeypatch.setattr(backend, "transpose", fail)
    monkeypatch.setattr(backend, "reshape", fail)
    monkeypatch.setattr(backend, "matmul", fail)

    with pytest.raises(ValueError, match="workspace capacity"):
        execute_plan(backend, plan, bindings, workspace=plan.workspace_bytes - 1)
    assert calls == []


def test_workspace_accepts_exact_integer_or_nbytes_capacity_bound():
    left = np.arange(24.0).reshape(2, 3, 4)
    right = np.arange(90.0).reshape(3, 5, 6)
    plan, bindings = plan_einsum("abc,bde->adce", {"left": left, "right": right})
    backend = _numpy_backend()

    integer_result = execute_plan(
        backend, plan, bindings, workspace=plan.workspace_bytes
    )
    token = np.empty(plan.workspace_bytes, dtype=np.uint8)
    token_result = execute_plan(backend, plan, bindings, workspace=token)

    expected = np.einsum("abc,bde->adce", left, right)
    np.testing.assert_allclose(integer_result, expected)
    np.testing.assert_allclose(token_result, expected)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (np.zeros((3, 2)), "shape"),
        (np.zeros((2, 3), dtype=np.float32), "dtype"),
        (np.asfortranarray(np.zeros((2, 3))), "layout"),
        ([[0.0] * 3] * 2, "NumPy host array"),
    ],
)
def test_invalid_binding_metadata_fails_before_primitive(monkeypatch, replacement, message):
    left = np.zeros((2, 3), dtype=np.float64)
    right = np.zeros((3, 4), dtype=np.float64)
    plan, bindings = plan_einsum("ab,bc->ac", {"left": left, "right": right})
    backend = _numpy_backend()
    calls = []
    monkeypatch.setattr(backend, "matmul", lambda *args, **kwargs: calls.append(True))
    bad = ExecutionBindings({**bindings.arrays, "input_0": replacement})

    with pytest.raises((TypeError, ValueError), match=message):
        execute_plan(backend, plan, bad)
    assert calls == []


def test_binding_keys_and_stream_are_validated_before_execution(monkeypatch):
    left = np.zeros((2, 3))
    right = np.zeros((3, 4))
    plan, bindings = plan_einsum("ab,bc->ac", {"left": left, "right": right})
    backend = _numpy_backend()
    calls = []
    monkeypatch.setattr(backend, "matmul", lambda *args, **kwargs: calls.append(True))

    with pytest.raises(ValueError, match="missing execution binding"):
        execute_plan(backend, plan, ExecutionBindings({"input_0": left}))
    with pytest.raises(ValueError, match="stream"):
        execute_plan(backend, plan, bindings, stream=object())
    assert calls == []


def _grouped_plan(shapes):
    refs = []
    outputs = []
    operations = []
    arrays = {}
    rng = np.random.default_rng(71)
    for index, (m, n, k) in enumerate(shapes):
        left_mode, contracted_mode, right_mode = "abcdefghi"[3 * index:3 * index + 3]
        left = _ref(f"input_{2 * index}", (m, k), (left_mode, contracted_mode))
        right = _ref(
            f"input_{2 * index + 1}", (k, n), (contracted_mode, right_mode)
        )
        output_key = "output" if index == 0 else f"other_{index}"
        output = _ref(output_key, (m, n), (left_mode, right_mode))
        refs.extend((left, right))
        outputs.append(output)
        operations.append(MatmulStep(left, right, output, (contracted_mode,)))
        arrays[left.key] = rng.normal(size=(m, k))
        arrays[right.key] = rng.normal(size=(k, n))
    grouped = GroupedMatmulStep(tuple(operations))
    return (
        _specialized_plan(tuple(refs), outputs[0], (grouped,)),
        ExecutionBindings(arrays),
        tuple(operations),
    )


def test_grouped_matmul_same_shape_executes_one_batched_primitive(monkeypatch):
    plan, bindings, operations = _grouped_plan(((2, 4, 3), (2, 4, 3)))
    backend = _numpy_backend()
    calls = {"scalar": 0, "batched": 0, "stack": 0}
    original_batched = backend.batched_matmul
    original_stack = backend.stack

    monkeypatch.setattr(
        backend,
        "matmul",
        lambda *args, **kwargs: pytest.fail("same-shape group used scalar matmul"),
    )

    def batched(*args, **kwargs):
        calls["batched"] += 1
        return original_batched(*args, **kwargs)

    def stack(*args, **kwargs):
        calls["stack"] += 1
        return original_stack(*args, **kwargs)

    monkeypatch.setattr(backend, "batched_matmul", batched)
    monkeypatch.setattr(backend, "stack", stack)

    actual = execute_plan(backend, plan, bindings)

    first = operations[0]
    expected = bindings.arrays[first.left.key] @ bindings.arrays[first.right.key]
    np.testing.assert_allclose(actual, expected)
    assert calls == {"scalar": 0, "batched": 1, "stack": 2}


def test_grouped_matmul_ragged_buckets_use_one_scalar_and_one_batched(monkeypatch):
    plan, bindings, operations = _grouped_plan(
        ((2, 4, 3), (5, 2, 3), (2, 4, 3))
    )
    backend = _numpy_backend()
    calls = {"scalar": 0, "batched": 0}
    original_scalar = backend.matmul
    original_batched = backend.batched_matmul

    def scalar(*args, **kwargs):
        calls["scalar"] += 1
        return original_scalar(*args, **kwargs)

    def batched(*args, **kwargs):
        calls["batched"] += 1
        return original_batched(*args, **kwargs)

    monkeypatch.setattr(backend, "matmul", scalar)
    monkeypatch.setattr(backend, "batched_matmul", batched)

    actual = execute_plan(backend, plan, bindings)

    first = operations[0]
    np.testing.assert_allclose(
        actual, bindings.arrays[first.left.key] @ bindings.arrays[first.right.key]
    )
    assert calls == {"scalar": 1, "batched": 1}


def test_grouped_workspace_counts_outputs_and_only_nonsingleton_packs():
    plan, _, operations = _grouped_plan(((2, 4, 3), (5, 2, 3), (2, 4, 3)))
    output_bytes = sum(operation.output.spec.nbytes for operation in operations)
    pair_pack_bytes = 2 * (2 * 3 + 3 * 4) * np.dtype("float64").itemsize

    assert plan.workspace_bytes == output_bytes + pair_pack_bytes


def test_grouped_workspace_preflight_runs_before_any_primitive(monkeypatch):
    plan, bindings, _ = _grouped_plan(((2, 4, 3), (2, 4, 3)))
    backend = _numpy_backend()
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise AssertionError("primitive ran before grouped workspace preflight")

    for name in ("reshape", "stack", "matmul", "batched_matmul"):
        monkeypatch.setattr(backend, name, fail)

    with pytest.raises(ValueError, match="workspace capacity"):
        execute_plan(backend, plan, bindings, workspace=plan.workspace_bytes - 1)
    assert calls == []


def test_grouped_matmul_validates_all_views_before_compute(monkeypatch):
    refs = (
        _ref("input_0", (2, 2, 3), ("a", "b", "c")),
        _ref("input_1", (3, 4), ("c", "d")),
        _ref("input_2", (2, 2, 3), ("e", "f", "g")),
        _ref("input_3", (3, 4), ("g", "h")),
    )
    outputs = (
        _ref("output", (2, 2, 4), ("a", "b", "d")),
        _ref("other", (2, 2, 4), ("e", "f", "h")),
    )
    grouped = GroupedMatmulStep(
        (
            MatmulStep(refs[0], refs[1], outputs[0], ("c",)),
            MatmulStep(refs[2], refs[3], outputs[1], ("g",)),
        )
    )
    plan = _specialized_plan(refs, outputs[0], (grouped,))
    bindings = ExecutionBindings(
        {ref.key: np.zeros(ref.spec.shape, dtype=np.float64) for ref in refs}
    )
    backend = _numpy_backend()
    original_reshape = backend.reshape
    reshape_calls = []
    compute_calls = []

    def bad_second_reshape(value, shape):
        reshape_calls.append(value)
        if len(reshape_calls) == 2:
            return np.array(original_reshape(value, shape), copy=True)
        return original_reshape(value, shape)

    monkeypatch.setattr(backend, "reshape", bad_second_reshape)
    monkeypatch.setattr(backend, "matmul", lambda *args, **kwargs: compute_calls.append("scalar"))
    monkeypatch.setattr(
        backend, "batched_matmul", lambda *args, **kwargs: compute_calls.append("batched")
    )

    with pytest.raises(ValueError, match="reshape unexpectedly copied"):
        execute_plan(backend, plan, bindings)
    assert compute_calls == []


def test_grouped_matmul_shape_mismatch_is_atomic(monkeypatch):
    refs = (
        _ref("input_0", (2, 3), ("a", "b")),
        _ref("input_1", (3, 4), ("b", "c")),
        _ref("input_2", (2, 3), ("d", "e")),
        _ref("input_3", (3, 4), ("e", "f")),
    )
    outputs = (_ref("output", (2, 4), ("a", "c")), _ref("other", (2, 4), ("d", "f")))
    grouped = GroupedMatmulStep(
        (
            MatmulStep(refs[0], refs[1], outputs[0], ("b",)),
            MatmulStep(refs[2], refs[3], outputs[1], ("e",)),
        )
    )
    plan = _specialized_plan(refs, outputs[0], (grouped,))
    backend = _numpy_backend()
    calls = []
    original_batched = backend.batched_matmul

    def bad_batched(*args, **kwargs):
        calls.append(True)
        result = original_batched(*args, **kwargs)
        return result[:, :, :-1]

    monkeypatch.setattr(backend, "batched_matmul", bad_batched)
    bindings = ExecutionBindings(
        {ref.key: np.zeros(ref.spec.shape, dtype=ref.spec.dtype) for ref in refs}
    )

    with pytest.raises(ValueError, match="shape"):
        execute_plan(backend, plan, bindings)
    assert calls == [True]


def test_backend_api_shape_and_unsupported_backends():
    from renormalizer.backend.jax_backend import JaxBackend
    from renormalizer.backend.torch_backend import TorchBackend

    backend = _numpy_backend()
    left = np.arange(6.0).reshape(2, 3)
    right = np.arange(12.0).reshape(3, 4)
    plan, bindings = plan_einsum("ab,bc->ac", {"left": left, "right": right})

    assert backend.supports_execution_ir is True
    np.testing.assert_allclose(backend.reshape(left, (3, 2)), left.reshape(3, 2))
    np.testing.assert_allclose(
        backend.matmul(left, right, stream=None, workspace=0), left @ right
    )
    np.testing.assert_allclose(backend.execute_plan(plan, bindings), left @ right)
    for backend_type in (JaxBackend, TorchBackend):
        assert backend_type.supports_execution_ir is False
        unsupported = object.__new__(backend_type)
        with pytest.raises(NotImplementedError, match="execution IR"):
            unsupported.execute_plan(plan, bindings)


def test_execute_plan_rejects_untrusted_argument_types():
    backend = _numpy_backend()
    with pytest.raises(TypeError, match="ExecutionPlan"):
        execute_plan(backend, object(), ExecutionBindings({}))
    left = np.zeros((2, 3))
    right = np.zeros((3, 4))
    plan, bindings = plan_einsum("ab,bc->ac", {"left": left, "right": right})
    with pytest.raises(TypeError, match="ExecutionBindings"):
        execute_plan(backend, plan, {})
    with pytest.raises(TypeError, match="workspace"):
        execute_plan(backend, plan, bindings, workspace=True)
