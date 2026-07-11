from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from renormalizer.backend._execution.model import (
    BatchedMatmulStep,
    BufferRef,
    ExecutionBindings,
    ExecutionPlan,
    GroupedMatmulStep,
    MatmulStep,
    ReductionStep,
    TensorSpec,
    TransformStep,
)
from renormalizer.backend._execution.planner import _array_layout, _plan_hash


def _ref(key, shape=(2, 3), modes=("a", "b"), dtype="float64", layout="C"):
    return BufferRef(key, TensorSpec(shape, dtype, layout, modes))


def _matmul_refs():
    return (
        _ref("input_0"),
        _ref("input_1", (3, 4), ("b", "c")),
        _ref("output", (2, 4), ("a", "c")),
    )


def _with_plan_hash(values):
    values = dict(values)
    values["plan_hash"] = _plan_hash(
        values["operation"],
        values["inputs"],
        values["output"],
        values["steps"],
        values["workspace_bytes"],
        values["planner_source"],
        values["oe_path"],
        values["override_reason"],
    )
    return values


def test_tensor_spec_is_immutable():
    spec = TensorSpec(shape=(2, 3), dtype="float64", layout="C", modes=("a", "b"))

    with pytest.raises(FrozenInstanceError):
        spec.shape = (6,)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"shape": (2, -1)}, "shape"),
        ({"shape": (2,), "modes": ("a", "b")}, "rank"),
        ({"modes": ("a", "a")}, "unique"),
        ({"dtype": "object"}, "dtype"),
        ({"layout": "unknown"}, "layout"),
    ],
)
def test_tensor_spec_rejects_invalid_metadata(kwargs, message):
    values = {"shape": (2, 3), "dtype": "float64", "layout": "C", "modes": ("a", "b")}
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError), match=message):
        TensorSpec(**values)


def test_step_types_are_frozen_metadata():
    left = _ref("left")
    right = _ref("right", (3, 4), ("b", "c"))
    output = _ref("output", (2, 4), ("a", "c"))
    transformed = _ref("transformed", (3, 2), ("b", "a"), layout="F")
    matmul = MatmulStep(left, right, output, ("b",))
    batch_left = _ref("batch_left", (5, 2, 3), ("x", "a", "b"))
    batch_right = _ref("batch_right", (5, 3, 4), ("x", "b", "c"))
    batch_output = _ref("batch_output", (5, 2, 4), ("x", "a", "c"))
    batched = BatchedMatmulStep(batch_left, batch_right, batch_output, ("x",), ("b",))
    second_matmul = MatmulStep(
        left, right, _ref("output_2", (2, 4), ("a", "c")), ("b",)
    )
    grouped = GroupedMatmulStep((matmul, second_matmul))
    reduction = ReductionStep(left, _ref("reduced", (2,), ("a",)), ("b",))
    transform = TransformStep(left, transformed, (1, 0), False)

    for step in (matmul, batched, grouped, reduction, transform):
        with pytest.raises(FrozenInstanceError):
            step.output = output


@pytest.mark.parametrize(
    "output, message",
    [
        (_ref("transformed", (3, 2), ("b", "a"), "float32"), "dtype"),
        (_ref("transformed", (3, 7), ("b", "a")), "shape"),
        (_ref("transformed", (3, 2), ("a", "b")), "modes"),
    ],
)
def test_transform_step_rejects_untrusted_output_metadata(output, message):
    with pytest.raises(ValueError, match=message):
        TransformStep(_ref("input"), output, (1, 0), False)


@pytest.mark.parametrize(
    "shape, order, axes",
    [
        ((), "C", ()),
        ((), "F", ()),
        ((7,), "C", (0,)),
        ((7,), "F", (0,)),
        ((2, 3), "C", (0, 1)),
        ((2, 3), "C", (1, 0)),
        ((2, 3), "F", (0, 1)),
        ((2, 3), "F", (1, 0)),
        ((2, 3, 4), "C", (1, 0, 2)),
        ((2, 3, 4), "C", (2, 1, 0)),
        ((2, 3, 4), "F", (2, 0, 1)),
        ((2, 3, 4), "F", (2, 1, 0)),
        ((2, 1, 3), "C", (1, 0, 2)),
        ((2, 1, 3), "C", (1, 2, 0)),
        ((2, 1, 3), "F", (0, 2, 1)),
        ((2, 1, 3), "F", (1, 2, 0)),
        ((1, 3), "C", (1, 0)),
        ((1, 3), "F", (1, 0)),
        ((0, 3), "C", (1, 0)),
        ((0, 3), "F", (0, 1)),
        ((2, 0, 3), "C", (1, 2, 0)),
        ((2, 0, 3), "F", (2, 0, 1)),
    ],
)
def test_transform_layout_matches_real_numpy_transpose(shape, order, axes):
    array = np.empty(shape, order=order)
    modes = tuple("abcdef"[:len(shape)])
    transposed = array.transpose(axes)
    source = _ref("source", shape, modes, layout=_array_layout(array))
    output = _ref(
        "output",
        tuple(shape[axis] for axis in axes),
        tuple(modes[axis] for axis in axes),
        layout=_array_layout(transposed),
    )

    assert TransformStep(source, output, axes, False).output.spec.layout == _array_layout(
        transposed
    )


@pytest.mark.parametrize(
    "shape, input_layout, axes, expected_layout",
    [
        ((), "strided", (), "C"),
        ((0,), "strided", (0,), "C"),
        ((2, 0, 3), "strided", (1, 2, 0), "C"),
        ((7,), "strided", (0,), "strided"),
        ((2, 3), "strided", (1, 0), "strided"),
        ((2, 1, 3), "strided", (1, 0, 2), "strided"),
    ],
)
def test_transform_layout_is_conservative_without_input_strides(
    shape, input_layout, axes, expected_layout
):
    modes = tuple("abcdef"[:len(shape)])
    source = _ref("source", shape, modes, layout=input_layout)
    output = _ref(
        "output",
        tuple(shape[axis] for axis in axes),
        tuple(modes[axis] for axis in axes),
        layout=expected_layout,
    )

    assert TransformStep(source, output, axes, False).output.spec.layout == expected_layout


@pytest.mark.parametrize(
    "shape, layout, axes, wrong_layout",
    [
        ((2, 1, 3), "C", (1, 0, 2), "strided"),
        ((0, 3), "C", (1, 0), "F"),
        ((), "strided", (), "strided"),
        ((2, 0, 3), "strided", (1, 2, 0), "strided"),
    ],
)
def test_transform_layout_rejects_noncanonical_classification(
    shape, layout, axes, wrong_layout
):
    modes = tuple("abcdef"[:len(shape)])
    source = _ref("source", shape, modes, layout=layout)
    output = _ref(
        "output",
        tuple(shape[axis] for axis in axes),
        tuple(modes[axis] for axis in axes),
        layout=wrong_layout,
    )

    with pytest.raises(ValueError, match="layout"):
        TransformStep(source, output, axes, False)


@pytest.mark.parametrize("input_layout", ["C", "F", "strided"])
@pytest.mark.parametrize("output_layout", ["C", "F"])
def test_transform_layout_accepts_materialized_contiguous_copy(
    input_layout, output_layout
):
    source = _ref("source", (2, 3), ("a", "b"), layout=input_layout)
    output = _ref("output", (3, 2), ("b", "a"), layout=output_layout)

    assert TransformStep(source, output, (1, 0), True).output.spec.layout == output_layout


def test_transform_layout_rejects_materialized_strided_copy():
    source = _ref("source")
    output = _ref("output", layout="strided")

    with pytest.raises(ValueError, match="layout"):
        TransformStep(source, output, (0, 1), True)


@pytest.mark.parametrize(
    "output, message",
    [
        (_ref("reduced", (2,), ("a",), "float32"), "dtype"),
        (_ref("reduced", (7,), ("a",)), "shape"),
        (_ref("reduced", (3,), ("b",)), "modes"),
    ],
)
def test_reduction_step_rejects_untrusted_output_metadata(output, message):
    with pytest.raises(ValueError, match=message):
        ReductionStep(_ref("input"), output, ("b",))


@pytest.mark.parametrize(
    "left, right, output, message",
    [
        (
            _ref("left"),
            _ref("right", (7, 4), ("b", "c")),
            _ref("output", (2, 4), ("a", "c")),
            "dimension.*b",
        ),
        (
            _ref("left"),
            _ref("right", (3, 4), ("b", "c")),
            _ref("output", (7, 4), ("a", "c")),
            "output shape",
        ),
        (
            _ref("left"),
            _ref("right", (3, 4), ("b", "c")),
            _ref("output", (2, 4), ("c", "a")),
            "output shape",
        ),
        (
            _ref("left"),
            _ref("right", (3, 4), ("b", "c"), "float32"),
            _ref("output", (2, 4), ("a", "c")),
            "dtype",
        ),
        (
            _ref("left"),
            _ref("right", (3, 4), ("b", "c")),
            _ref("output", (2, 4), ("a", "c"), "float32"),
            "dtype",
        ),
    ],
)
def test_matmul_step_rejects_untrusted_shape_and_dtype(left, right, output, message):
    with pytest.raises(ValueError, match=message):
        MatmulStep(left, right, output, ("b",))


@pytest.mark.parametrize("contracted_modes", [("b",), ("c", "b")])
def test_matmul_step_requires_complete_left_ordered_contracted_modes(contracted_modes):
    left = _ref("left", (2, 3, 4), ("a", "b", "c"))
    right = _ref("right", (3, 4, 5), ("b", "c", "d"))
    output = _ref("output", (2, 5), ("a", "d"))

    with pytest.raises(ValueError, match="contracted modes"):
        MatmulStep(left, right, output, contracted_modes)


def test_batched_matmul_requires_complete_ordered_mode_metadata():
    left = _ref("left", (5, 2, 3, 4), ("x", "a", "b", "c"))
    right = _ref("right", (5, 3, 4, 6), ("x", "b", "c", "d"))
    output = _ref("output", (5, 2, 6), ("x", "a", "d"))

    with pytest.raises(ValueError, match="contracted modes"):
        BatchedMatmulStep(left, right, output, ("x",), ("c", "b"))
    with pytest.raises(ValueError, match="batch modes"):
        BatchedMatmulStep(left, right, output, (), ("b", "c"))


def test_step_metadata_keeps_valid_zero_sized_and_scalar_shapes():
    left = _ref("left", (0,), ("a",))
    right = _ref("right", (0,), ("a",))
    scalar = _ref("scalar", (), ())

    assert MatmulStep(left, right, scalar, ("a",)).output.spec.shape == ()
    assert ReductionStep(left, scalar, ("a",)).output.spec.shape == ()
    assert TransformStep(scalar, _ref("scalar_copy", (), ()), (), False).axes == ()


@pytest.mark.parametrize("operations", [(), (_matmul_refs(),)])
def test_grouped_matmul_requires_at_least_two_operations(operations):
    if operations:
        left, right, output = operations[0]
        operations = (MatmulStep(left, right, output, ("b",)),)

    with pytest.raises(ValueError, match="at least two"):
        GroupedMatmulStep(operations)


def test_grouped_matmul_rejects_dependent_operations():
    left, right, temporary = _matmul_refs()
    temporary = BufferRef("temporary", temporary.spec)
    tail = _ref("tail", (4, 5), ("c", "d"))
    output = _ref("output", (2, 5), ("a", "d"))
    first = MatmulStep(left, right, temporary, ("b",))
    second = MatmulStep(temporary, tail, output, ("c",))

    with pytest.raises(ValueError, match="independent"):
        GroupedMatmulStep((first, second))


def test_grouped_matmul_rejects_duplicate_output_keys():
    left, right, output = _matmul_refs()
    first = MatmulStep(left, right, output, ("b",))
    second = MatmulStep(left, right, output, ("b",))

    with pytest.raises(ValueError, match="unique output"):
        GroupedMatmulStep((first, second))


def test_grouped_matmul_allows_shared_read_only_inputs():
    left, right, output = _matmul_refs()
    second_output = BufferRef("output_2", output.spec)
    operations = (
        MatmulStep(left, right, output, ("b",)),
        MatmulStep(left, right, second_output, ("b",)),
    )

    assert GroupedMatmulStep(operations).operations == operations


def test_execution_bindings_copy_and_validate_keys():
    source = {"input_0": object(), "input_1": object()}
    bindings = ExecutionBindings(source)
    left, right, output = _matmul_refs()
    plan = ExecutionPlan(**_with_plan_hash(dict(
        operation="einsum",
        inputs=(left, right),
        output=output,
        steps=(MatmulStep(left, right, output, ("b",)),),
        workspace_bytes=0,
        planner_source="opt_einsum",
        oe_path=((0, 1),),
        override_reason=None,
    )))

    source.clear()
    assert tuple(bindings.arrays) == ("input_0", "input_1")
    bindings.validate_for(plan)

    with pytest.raises(TypeError):
        bindings.arrays["input_2"] = object()
    with pytest.raises(ValueError, match="missing.*input_1"):
        ExecutionBindings({"input_0": object()}).validate_for(plan)
    with pytest.raises(ValueError, match="unexpected.*input_2"):
        ExecutionBindings(
            {"input_0": object(), "input_1": object(), "input_2": object()}
        ).validate_for(plan)


def test_execution_plan_rejects_invalid_trust_metadata():
    left, right, output = _matmul_refs()
    values = _with_plan_hash(dict(
        operation="einsum",
        inputs=(left, right),
        output=output,
        steps=(MatmulStep(left, right, output, ("b",)),),
        workspace_bytes=0,
        planner_source="opt_einsum",
        oe_path=((0, 1),),
        override_reason=None,
    ))

    with pytest.raises(ValueError, match="workspace"):
        ExecutionPlan(**{**values, "workspace_bytes": -1})
    with pytest.raises(ValueError, match="hash"):
        ExecutionPlan(**{**values, "plan_hash": "not-a-hash"})
    with pytest.raises(ValueError, match="duplicate"):
        ExecutionPlan(**{**values, "inputs": (_ref("input_0"), _ref("input_0"))})
    with pytest.raises(TypeError, match="step"):
        ExecutionPlan(**{**values, "steps": (object(),)})
    missing = BufferRef("missing", right.spec)
    with pytest.raises(ValueError, match="unknown buffer.*missing"):
        ExecutionPlan(
            **{**values, "steps": (MatmulStep(left, missing, output, ("b",)),)}
        )
