import dataclasses
from types import SimpleNamespace

import numpy as np
import opt_einsum as oe
import pytest

from renormalizer.backend._execution import planner as planner_module
from renormalizer.backend._execution.model import (
    BatchedMatmulStep,
    BufferRef,
    ExecutionPlan,
    MatmulStep,
    ReductionStep,
    TensorSpec,
    TransformStep,
)
from renormalizer.backend._execution.planner import _plan_hash, lower_einsum_path, plan_einsum


SINGLE_SITE_EQUATION = "abc,bdef,lfk,cek->adl"
SINGLE_SITE_SHAPES = ((2, 3, 4), (3, 5, 4, 6), (7, 6, 8), (4, 4, 8))


def _plan_values(plan, **changes):
    values = {
        "operation": plan.operation,
        "inputs": plan.inputs,
        "output": plan.output,
        "steps": plan.steps,
        "workspace_bytes": plan.workspace_bytes,
        "planner_source": plan.planner_source,
        "oe_path": plan.oe_path,
        "override_reason": plan.override_reason,
    }
    values.update(changes)
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


def _transform_workspace_plan(copy, *, final=False):
    source = BufferRef("input_0", TensorSpec((2, 3), "float64", "C", ("a", "b")))
    transformed = BufferRef(
        "output" if final else "temporary_0",
        TensorSpec((2, 3), "float64", "C", ("a", "b")),
    )
    transform = TransformStep(source, transformed, (0, 1), copy)
    if final:
        prototype = SimpleNamespace(
            operation="einsum",
            inputs=(source,),
            output=transformed,
            steps=(transform,),
            workspace_bytes=0,
            planner_source="opt_einsum",
            oe_path=(),
            override_reason=None,
        )
    else:
        right = BufferRef(
            "input_1", TensorSpec((3, 4), "float64", "C", ("b", "c"))
        )
        output = BufferRef(
            "output", TensorSpec((2, 4), "float64", "C", ("a", "c"))
        )
        steps = (transform, MatmulStep(transformed, right, output, ("b",)))
        prototype = SimpleNamespace(
            operation="einsum",
            inputs=(source, right),
            output=output,
            steps=steps,
            workspace_bytes=transformed.spec.nbytes if copy else 0,
            planner_source="opt_einsum",
            oe_path=((0, 1),),
            override_reason=None,
        )
    return ExecutionPlan(**_plan_values(prototype)), transformed


def test_single_site_path_uses_opt_einsum_or_records_override():
    plan = lower_einsum_path(
        SINGLE_SITE_EQUATION,
        SINGLE_SITE_SHAPES,
        dtype="float64",
        optimize="optimal",
    )

    expected_path, _ = oe.contract_path(
        SINGLE_SITE_EQUATION, *SINGLE_SITE_SHAPES, shapes=True, optimize="optimal"
    )
    assert plan.planner_source in ("opt_einsum", "specialized")
    assert plan.oe_path == tuple(tuple(indices) for indices in expected_path)
    assert plan.steps
    assert len(
        [step for step in plan.steps if isinstance(step, (MatmulStep, BatchedMatmulStep))]
    ) == len(plan.oe_path)
    assert any(isinstance(step, TransformStep) for step in plan.steps)
    assert plan.override_reason is None


def test_resolved_path_metadata_reuses_oracle_and_rejects_stale_input(monkeypatch):
    equation = "ab,bc->ac"
    shapes = ((2, 3), (3, 4))
    metadata = planner_module._resolve_einsum_path(
        equation, shapes, optimize="optimal"
    )
    assert metadata.equation == equation
    assert metadata.shapes == shapes
    assert metadata.optimize == "optimal"
    assert len(metadata.oe_path) == len(metadata.contraction_list)
    monkeypatch.setattr(
        planner_module.oe,
        "contract_path",
        lambda *args, **kwargs: pytest.fail("reused metadata searched OE path"),
    )

    plan = lower_einsum_path(
        equation,
        shapes,
        dtype="float64",
        layouts=("F", "strided"),
        _path_metadata=metadata,
    )

    assert plan.oe_path == metadata.oe_path
    assert tuple(ref.spec.layout for ref in plan.inputs) == ("F", "strided")
    assert plan.override_reason is None
    with pytest.raises(ValueError, match="equation"):
        lower_einsum_path(
            "ab,bc->ca", shapes, _path_metadata=metadata
        )
    with pytest.raises(ValueError, match="shapes"):
        lower_einsum_path(
            equation, ((5, 3), (3, 4)), _path_metadata=metadata
        )
    with pytest.raises(ValueError, match="optimizer"):
        lower_einsum_path(
            equation, shapes, optimize="greedy", _path_metadata=metadata
        )
    with pytest.raises(TypeError, match="path metadata"):
        lower_einsum_path(equation, shapes, _path_metadata=object())
    malformed = dataclasses.replace(metadata, oe_path=((0, 0),))
    with pytest.raises(ValueError, match="path"):
        lower_einsum_path(equation, shapes, _path_metadata=malformed)


def test_resolved_path_metadata_fingerprint_rejects_replaced_optimizer():
    metadata = planner_module._resolve_einsum_path(
        SINGLE_SITE_EQUATION, SINGLE_SITE_SHAPES, optimize="greedy"
    )
    replaced = dataclasses.replace(metadata, optimize="optimal")

    with pytest.raises(ValueError, match="integrity fingerprint"):
        lower_einsum_path(
            SINGLE_SITE_EQUATION,
            SINGLE_SITE_SHAPES,
            optimize="optimal",
            _path_metadata=replaced,
        )


def test_reusable_path_metadata_snapshots_explicit_path_and_rejects_mutation():
    equation = "ab,bc->ac"
    shapes = ((2, 3), (3, 4))
    mutable_path = [(0, 1)]
    metadata = planner_module._resolve_einsum_path(
        equation, shapes, optimize=mutable_path
    )
    assert metadata.optimize == ((0, 1),)

    mutable_path[0] = (1, 0)
    with pytest.raises(ValueError, match="optimizer"):
        lower_einsum_path(
            equation,
            shapes,
            optimize=mutable_path,
            _path_metadata=metadata,
        )


def test_custom_mutable_optimizer_remains_ordinary_only():
    class MutableOptimizer(oe.paths.PathOptimizer):
        def __init__(self):
            self.calls = 0

        def __call__(self, inputs, output, size_dict, memory_limit=None):
            self.calls += 1
            return [(0, 1), (0, 1)]

    optimizer = MutableOptimizer()
    plan = lower_einsum_path(
        "ab,bc,cd->ad", ((2, 3), (3, 4), (4, 5)), optimize=optimizer
    )
    assert plan.oe_path == ((0, 1), (0, 1))
    assert optimizer.calls == 1

    with pytest.raises(TypeError, match="reusable path metadata optimizer"):
        planner_module._resolve_einsum_path(
            "ab,bc,cd->ad", ((2, 3), (3, 4), (4, 5)), optimize=optimizer
        )
    assert optimizer.calls == 1


def test_explicit_packing_simple_matrix_product_is_direct():
    plan = lower_einsum_path("ab,bc->ac", ((2, 3), (3, 4)), dtype="float64")

    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert isinstance(step, MatmulStep)
    assert step.left == plan.inputs[0]
    assert step.right == plan.inputs[1]
    assert step.output == plan.output
    assert step.left.spec.modes == ("a", "b")
    assert step.right.spec.modes == ("b", "c")
    assert step.output.spec.modes == ("a", "c")
    assert plan.workspace_bytes == 0


def test_explicit_packing_records_input_pack_and_output_reorder():
    plan = lower_einsum_path(
        "abc,bde->adce", ((2, 3, 4), (3, 5, 6)), dtype="float64"
    )

    assert [type(step) for step in plan.steps] == [
        TransformStep,
        MatmulStep,
        TransformStep,
    ]
    pack, pair, reorder = plan.steps
    assert pack.input == plan.inputs[0]
    assert pack.output.key == "temporary_0"
    assert pack.output.spec.modes == ("a", "c", "b")
    assert pack.output.spec.layout == "C"
    assert pack.axes == (0, 2, 1)
    assert pack.copy is True
    assert pair.left == pack.output
    assert pair.right == plan.inputs[1]
    assert pair.output.key == "temporary_1"
    assert pair.output.spec.modes == ("a", "c", "d", "e")
    assert reorder.input == pair.output
    assert reorder.output == plan.output
    assert reorder.axes == (0, 2, 1, 3)
    assert reorder.copy is True
    assert plan.workspace_bytes == pack.output.spec.nbytes + pair.output.spec.nbytes


def test_explicit_packing_uses_canonical_multi_batch_and_contracted_groups():
    plan = lower_einsum_path(
        "axykl,yxlkc->xyac",
        ((4, 2, 3, 5, 6), (3, 2, 6, 5, 7)),
        dtype="float64",
    )

    pair = next(step for step in plan.steps if isinstance(step, BatchedMatmulStep))
    assert pair.batch_modes == ("x", "y")
    assert pair.contracted_modes == ("k", "l")
    assert pair.left.spec.modes == ("x", "y", "a", "k", "l")
    assert pair.right.spec.modes == ("x", "y", "k", "l", "c")
    assert pair.output.spec.modes == ("x", "y", "a", "c")
    assert pair.left.spec.layout == pair.right.spec.layout == "C"


def test_explicit_packing_plan_einsum_packs_f_and_strided_inputs():
    left = np.asfortranarray(np.arange(6.0).reshape(2, 3))
    right = np.arange(24.0).reshape(3, 8)[:, ::2]

    plan, bindings = plan_einsum("ab,bc->ac", {"left": left, "right": right})

    assert tuple(ref.spec.layout for ref in plan.inputs) == ("F", "strided")
    assert tuple(bindings.arrays) == ("input_0", "input_1")
    assert bindings.arrays["input_0"] is left
    assert bindings.arrays["input_1"] is right
    packs = [step for step in plan.steps if isinstance(step, TransformStep)]
    assert len(packs) == 2
    assert all(step.copy and step.axes == tuple(range(2)) for step in packs)
    assert all(step.output.spec.layout == "C" for step in packs)
    pair = next(step for step in plan.steps if isinstance(step, MatmulStep))
    assert pair.left == packs[0].output
    assert pair.right == packs[1].output
    assert plan.workspace_bytes == sum(step.output.spec.nbytes for step in packs)


@pytest.mark.parametrize(
    "equation, shapes",
    [
        ("ab,bc->ac", ((2, 3), (3, 4))),
        ("abc,bde->adce", ((2, 3, 4), (3, 5, 6))),
        ("axykl,yxlkc->xyac", ((4, 2, 3, 5, 6), (3, 2, 6, 5, 7))),
        (SINGLE_SITE_EQUATION, SINGLE_SITE_SHAPES),
    ],
)
def test_explicit_packing_all_pair_steps_have_canonical_physical_modes(
    equation, shapes
):
    plan = lower_einsum_path(equation, shapes, dtype="float64")

    for step in plan.steps:
        if not isinstance(step, (MatmulStep, BatchedMatmulStep)):
            continue
        shared = set(step.left.spec.modes) & set(step.right.spec.modes)
        batch = tuple(mode for mode in step.output.spec.modes if mode in shared)
        left_free = tuple(mode for mode in step.left.spec.modes if mode not in shared)
        right_free = tuple(mode for mode in step.right.spec.modes if mode not in shared)
        assert step.left.spec.modes == batch + left_free + step.contracted_modes
        assert step.right.spec.modes == batch + step.contracted_modes + right_free
        assert step.output.spec.modes == batch + left_free + right_free
        assert step.left.spec.layout == step.right.spec.layout == "C"
        assert step.output.spec.layout == "C"


def test_explicit_packing_layout_argument_is_validated():
    with pytest.raises(ValueError, match="layout count"):
        lower_einsum_path("ab,bc->ac", ((2, 3), (3, 4)), layouts=("C",))
    with pytest.raises(ValueError, match="layout"):
        lower_einsum_path(
            "ab,bc->ac", ((2, 3), (3, 4)), layouts=("C", "unknown")
        )


def test_explicit_packing_keys_and_hash_are_deterministic():
    first = lower_einsum_path("abc,bde->adce", ((2, 3, 4), (3, 5, 6)))
    second = lower_einsum_path("abc,bde->adce", ((2, 3, 4), (3, 5, 6)))

    assert first == second
    assert first.plan_hash == second.plan_hash
    assert [step.output.key for step in first.steps] == [
        "temporary_0",
        "temporary_1",
        "output",
    ]


def test_plan_hash_and_buffer_keys_are_deterministic():
    first = lower_einsum_path("ab,bc,cd->ad", ((2, 3), (3, 4), (4, 5)), dtype="float64")
    second = lower_einsum_path("ab,bc,cd->ad", ((2, 3), (3, 4), (4, 5)), dtype=np.float64)

    assert first == second
    assert len(first.plan_hash) == 64
    assert tuple(ref.key for ref in first.inputs) == ("input_0", "input_1", "input_2")
    assert first.output.key == "output"
    assert first.workspace_bytes == sum(
        step.output.spec.nbytes
        for step in first.steps
        if hasattr(step, "output") and step.output.key != "output"
    )


def test_hash_changes_when_plan_metadata_changes():
    first = lower_einsum_path("ab,bc->ac", ((2, 3), (3, 4)), dtype="float64")
    second = lower_einsum_path("ab,bc->ac", ((5, 3), (3, 4)), dtype="float64")

    assert first.plan_hash != second.plan_hash


def test_execution_plan_rejects_stale_hash():
    plan = lower_einsum_path("ab,bc->ac", ((2, 3), (3, 4)), dtype="float64")

    with pytest.raises(ValueError, match="hash.*metadata"):
        dataclasses.replace(plan, plan_hash="0" * 64)


@pytest.mark.parametrize("adjustment", [-1, 1])
def test_execution_plan_rejects_inexact_workspace_claim(adjustment):
    plan = lower_einsum_path(
        "ab,bc,cd->ad", ((2, 3), (3, 4), (4, 5)), dtype="float64"
    )
    assert plan.workspace_bytes > 0

    with pytest.raises(ValueError, match="workspace.*exact"):
        dataclasses.replace(
            plan, workspace_bytes=plan.workspace_bytes + adjustment
        )


def test_execution_plan_rejects_workspace_overflow_from_steps():
    huge = 1 << 60
    left = BufferRef("input_0", TensorSpec((huge, 1), "float64", "C", ("a", "b")))
    right = BufferRef("input_1", TensorSpec((1, 1), "float64", "C", ("b", "c")))
    tail = BufferRef("input_2", TensorSpec((1, 1), "float64", "C", ("c", "d")))
    temporary = BufferRef(
        "temporary_0", TensorSpec((huge, 1), "float64", "C", ("a", "c"))
    )
    output = BufferRef("output", TensorSpec((huge, 1), "float64", "C", ("a", "d")))
    steps = (
        MatmulStep(left, right, temporary, ("b",)),
        MatmulStep(temporary, tail, output, ("c",)),
    )
    prototype = SimpleNamespace(
        operation="einsum",
        inputs=(left, right, tail),
        output=output,
        steps=steps,
        workspace_bytes=0,
        planner_source="opt_einsum",
        oe_path=((0, 1), (0, 1)),
        override_reason=None,
    )

    with pytest.raises(OverflowError, match="workspace"):
        ExecutionPlan(**_plan_values(prototype))


def test_transform_workspace_no_copy_alias_allocates_zero():
    plan, _ = _transform_workspace_plan(False)

    assert plan.workspace_bytes == 0


def test_transform_workspace_copy_counts_once_and_final_output_is_excluded():
    copied, temporary = _transform_workspace_plan(True)
    final, _ = _transform_workspace_plan(True, final=True)

    assert copied.workspace_bytes == temporary.spec.nbytes
    assert final.workspace_bytes == 0


def test_transform_workspace_and_hash_recompute_from_copy_flag():
    alias, temporary = _transform_workspace_plan(False)
    copied, _ = _transform_workspace_plan(True)

    assert alias.plan_hash != copied.plan_hash
    with pytest.raises(ValueError, match="workspace.*exact"):
        dataclasses.replace(copied, workspace_bytes=0)
    with pytest.raises(ValueError, match="hash.*metadata"):
        dataclasses.replace(copied, plan_hash=alias.plan_hash)
    assert copied.workspace_bytes == temporary.spec.nbytes


def test_opt_einsum_plan_rejects_unrelated_range_valid_path():
    plan = lower_einsum_path(
        "ab,bc,cd->ad", ((2, 3), (3, 4), (4, 5)), dtype="float64"
    )
    alternatives = ((0, 1), (0, 2), (1, 2))
    wrong_first = next(pair for pair in alternatives if set(pair) != set(plan.oe_path[0]))
    wrong_path = (wrong_first,) + plan.oe_path[1:]

    with pytest.raises(ValueError, match="OE path.*dependencies"):
        dataclasses.replace(plan, oe_path=wrong_path)


def test_opt_einsum_path_correlation_allows_transform_preprocessing():
    left = BufferRef("input_0", TensorSpec((2, 3), "float64", "F", ("b", "a")))
    right = BufferRef("input_1", TensorSpec((2, 4), "float64", "C", ("b", "c")))
    transformed = BufferRef(
        "temporary_0", TensorSpec((3, 2), "float64", "C", ("a", "b"))
    )
    output = BufferRef("output", TensorSpec((3, 4), "float64", "C", ("a", "c")))
    steps = (
        TransformStep(left, transformed, (1, 0), False),
        MatmulStep(transformed, right, output, ("b",)),
    )
    prototype = SimpleNamespace(
        operation="einsum",
        inputs=(left, right),
        output=output,
        steps=steps,
        workspace_bytes=0,
        planner_source="opt_einsum",
        oe_path=((0, 1),),
        override_reason=None,
    )

    plan = ExecutionPlan(**_plan_values(prototype))

    assert plan.steps == steps


def test_specialized_plan_may_override_oe_pair_selection():
    plan = lower_einsum_path(
        "ab,bc,cd->ad", ((2, 3), (3, 4), (4, 5)), dtype="float64"
    )
    alternatives = ((0, 1), (0, 2), (1, 2))
    wrong_first = next(pair for pair in alternatives if set(pair) != set(plan.oe_path[0]))
    wrong_path = (wrong_first,) + plan.oe_path[1:]

    specialized = ExecutionPlan(**_plan_values(
        plan,
        planner_source="specialized",
        oe_path=wrong_path,
        override_reason="specialized pair ordering",
    ))

    assert specialized.oe_path == wrong_path


@pytest.mark.parametrize(
    "override_reason",
    [None, "", "   ", 1, ["mutable reason"], {"mutable": "reason"}],
)
def test_override_reason_rejects_non_string_or_empty_metadata(override_reason):
    plan = lower_einsum_path("ab,bc->ac", ((2, 3), (3, 4)))

    with pytest.raises(ValueError, match="override reason"):
        ExecutionPlan(**_plan_values(
            plan,
            planner_source="specialized",
            override_reason=override_reason,
        ))


def test_override_reason_accepts_nonempty_string_for_specialized_plan():
    plan = lower_einsum_path("ab,bc->ac", ((2, 3), (3, 4)))

    specialized = ExecutionPlan(**_plan_values(
        plan,
        planner_source="specialized",
        override_reason="manual pair ordering",
    ))

    assert specialized.override_reason == "manual pair ordering"


def test_override_reason_for_opt_einsum_must_remain_none():
    plan = lower_einsum_path("ab,bc->ac", ((2, 3), (3, 4)))

    with pytest.raises(ValueError, match="override reason"):
        ExecutionPlan(**_plan_values(plan, override_reason="not allowed"))


def test_plan_einsum_keeps_arrays_only_in_deterministic_bindings():
    a = np.arange(6.0).reshape(2, 3)
    b = np.arange(12.0).reshape(3, 4)[:, ::-1]
    plan, bindings = plan_einsum("ab,bc->ac", {"left": a, "right": b})

    assert tuple(bindings.arrays) == ("input_0", "input_1")
    assert bindings.arrays["input_0"] is a
    assert bindings.arrays["input_1"] is b
    assert plan.inputs[0].spec.layout == "C"
    assert plan.inputs[1].spec.layout == "strided"
    assert not any(value is a or value is b for value in _walk_values(plan))


def _walk_values(value):
    yield value
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from _walk_values(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_values(item)


def test_reductions_and_batched_contractions_are_truthful_metadata():
    reduction_plan = lower_einsum_path("ab,bc->a", ((2, 3), (3, 4)), dtype="float64")
    batched_plan = lower_einsum_path("bij,bjk->bik", ((2, 3, 4), (2, 4, 5)), dtype="float64")

    assert any(isinstance(step, ReductionStep) for step in reduction_plan.steps)
    assert isinstance(reduction_plan.steps[-1], MatmulStep)
    assert isinstance(batched_plan.steps[-1], BatchedMatmulStep)
    assert batched_plan.steps[-1].batch_modes == ("b",)


@pytest.mark.parametrize(
    "equation, shapes, message",
    [
        ("ab,bc", ((2, 3), (3, 4)), "explicit output"),
        ("ab,bc->ac", ((2, 3, 1), (3, 4)), "rank"),
        ("ab,bc->ac", ((2, 3), (5, 4)), "mode.*b"),
        ("aa,ab->b", ((2, 2), (2, 3)), "repeated"),
        ("ab,cd->abcd", ((2, 3), (4, 5)), "unsupported"),
    ],
)
def test_lowering_rejects_invalid_or_non_gemm_equations(equation, shapes, message):
    with pytest.raises(ValueError, match=message):
        lower_einsum_path(equation, shapes, dtype="float64")


def test_plan_einsum_validates_operand_collection_and_metadata():
    a = np.ones((2, 3), dtype=np.float64)
    b = np.ones((3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="operand count"):
        plan_einsum("ab,bc->ac", {"only": a})
    with pytest.raises(ValueError, match="dtype"):
        plan_einsum("ab,bc->ac", {"left": a, "right": b})
    with pytest.raises(ValueError, match="binding key"):
        plan_einsum("ab,bc->ac", {1: a, "right": a.T})


def test_public_compatibility_module_has_explicit_stage_three_exports():
    import renormalizer.backend.execution as execution

    assert execution.__all__ == [
        "BatchedMatmulStep",
        "BufferRef",
        "ExecutionBindings",
        "ExecutionPlan",
        "GroupedMatmulStep",
        "MatmulStep",
        "ReductionStep",
        "TensorSpec",
        "TransformStep",
        "lower_einsum_path",
        "plan_einsum",
    ]
