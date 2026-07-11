import importlib.util

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


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cupy") is None, reason="CuPy is not installed"
)


@pytest.fixture(autouse=True)
def restore_numpy_backend():
    try:
        yield
    finally:
        set_backend("numpy", precision=64)


@pytest.fixture
def cupy_backend():
    try:
        return set_backend("cupy", device="cuda:0", precision=64)
    except Exception as error:
        pytest.skip("CuPy/CUDA is unavailable: {}".format(error))


def _device_plan(backend, equation, arrays):
    device_arrays = {key: backend.asarray(value) for key, value in arrays.items()}
    return (*plan_einsum(equation, device_arrays), device_arrays)


def _ref(key, shape, modes, layout="C"):
    return BufferRef(key, TensorSpec(shape, "float64", layout, modes))


def _specialized_plan(inputs, output, steps):
    values = dict(
        operation="einsum",
        inputs=tuple(inputs),
        output=output,
        steps=tuple(steps),
        workspace_bytes=workspace_bytes_for_steps(steps, output.key),
        planner_source="specialized",
        oe_path=tuple((0, 1) for _ in range(len(inputs) - 1)),
        override_reason="CuPy executor test fixture",
    )
    values["plan_hash"] = _plan_hash(**values)
    return ExecutionPlan(**values)


def _reduction_plan(shape, input_layout, output_layout):
    input_ref = BufferRef(
        "input_0", TensorSpec(shape, "float64", input_layout, ("a", "b", "c"))
    )
    output_ref = BufferRef(
        "output", TensorSpec(shape[:2], "float64", output_layout, ("a", "b"))
    )
    steps = (ReductionStep(input_ref, output_ref, ("c",)),)
    values = dict(
        operation="einsum",
        inputs=(input_ref,),
        output=output_ref,
        steps=steps,
        workspace_bytes=workspace_bytes_for_steps(steps, output_ref.key),
        planner_source="specialized",
        oe_path=(),
        override_reason="CuPy reduction executor test",
    )
    values["plan_hash"] = _plan_hash(**values)
    return ExecutionPlan(**values)


@pytest.mark.parametrize("shape", [(), (4,), (1, 4, 1), (0, 4)])
def test_cupy_dual_contiguous_f_binding_and_copied_transform_output(
    cupy_backend, shape
):
    cp = cupy_backend._cupy
    modes = tuple("abc"[:len(shape)])
    source = cp.empty(shape, dtype=cp.float64, order="F")
    source.fill(2.5)
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
        cupy_backend, plan, ExecutionBindings({"input_0": source})
    )

    cp.testing.assert_array_equal(actual, source)
    assert actual.data.mem is not source.data.mem
    assert actual.flags.f_contiguous


@pytest.mark.parametrize("case", ["scalar", "empty"])
def test_cupy_dual_contiguous_f_planned_reduction_output(cupy_backend, case):
    cp = cupy_backend._cupy
    if case == "scalar":
        source = cp.arange(3.0)
        input_modes = ("a",)
        output_shape = ()
        output_modes = ()
        reduced_modes = ("a",)
    else:
        source = cp.empty((0, 3), dtype=cp.float64)
        input_modes = ("a", "b")
        output_shape = (0,)
        output_modes = ("a",)
        reduced_modes = ("b",)
    input_ref = _ref("input_0", source.shape, input_modes)
    output_ref = _ref("output", output_shape, output_modes, layout="F")
    plan = _specialized_plan(
        (input_ref,),
        output_ref,
        (ReductionStep(input_ref, output_ref, reduced_modes),),
    )

    actual = execute_plan(
        cupy_backend, plan, ExecutionBindings({"input_0": source})
    )

    axes = tuple(input_modes.index(mode) for mode in reduced_modes)
    cp.testing.assert_array_equal(actual, source.sum(axis=axes))
    assert actual.flags.f_contiguous


@pytest.mark.parametrize(
    ("expected_layout", "actual_layout"),
    [("F", "C"), ("C", "F"), ("strided", "C")],
)
def test_cupy_layout_predicate_rejects_genuine_binding_mismatch(
    cupy_backend, expected_layout, actual_layout
):
    cp = cupy_backend._cupy
    if actual_layout == "F":
        source = cp.asfortranarray(cp.arange(6.0).reshape(2, 3))
        assert source.flags.f_contiguous and not source.flags.c_contiguous
    else:
        source = cp.arange(6.0).reshape(2, 3)
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
            cupy_backend, plan, ExecutionBindings({"input_0": source})
        )


@pytest.mark.parametrize(
    ("equation", "arrays"),
    [
        (
            "abc,bdef,lfk,cek->adl",
            {
                "left": np.random.default_rng(21).normal(size=(2, 3, 4)),
                "mpo": np.random.default_rng(22).normal(size=(3, 5, 4, 6)),
                "right": np.random.default_rng(23).normal(size=(7, 6, 8)),
                "center": np.random.default_rng(24).normal(size=(4, 4, 8)),
            },
        ),
        (
            "abcde,abdef->abcf",
            {
                "left": np.random.default_rng(25).normal(size=(2, 3, 4, 5, 6)),
                "right": np.random.default_rng(26).normal(size=(2, 3, 5, 6, 7)),
            },
        ),
        (
            "abc,cd->ad",
            {
                "left": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                "right": np.arange(20, dtype=np.float32).reshape(4, 5),
            },
        ),
        (
            "ab,bc->ac",
            {"left": np.empty((2, 0)), "right": np.empty((0, 3))},
        ),
        ("a,a->", {"left": np.arange(4.0), "right": np.arange(4.0)}),
    ],
)
def test_cupy_executor_matches_numpy_on_selected_device(cupy_backend, equation, arrays):
    plan, bindings, _ = _device_plan(cupy_backend, equation, arrays)
    cp = cupy_backend._cupy

    actual = execute_plan(cupy_backend, plan, bindings)

    assert cupy_backend.supports_execution_ir is True
    assert isinstance(actual, cp.ndarray)
    assert actual.device.id == 0
    np.testing.assert_allclose(
        cp.asnumpy(actual), np.einsum(equation, *arrays.values()), rtol=1e-11, atol=1e-11
    )


def test_cupy_executor_handles_complex_noncontiguous_inputs(cupy_backend):
    cp = cupy_backend._cupy
    rng = np.random.default_rng(27)
    a0 = rng.normal(size=(4, 3, 2)) + 1j * rng.normal(size=(4, 3, 2))
    b0 = rng.normal(size=(5, 4)) + 1j * rng.normal(size=(5, 4))
    a = cupy_backend.asarray(a0).transpose(2, 1, 0)
    b = cupy_backend.asarray(b0).T
    plan, bindings = plan_einsum("abc,cd->abd", {"a": a, "b": b})

    actual = cupy_backend.execute_plan(plan, bindings)

    assert isinstance(actual, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(actual), np.einsum("abc,cd->abd", a0.T, b0.T))


@pytest.mark.parametrize("layout", ["C", "F"])
def test_cupy_reduction_uses_exact_planned_output_layout(cupy_backend, layout):
    cp = cupy_backend._cupy
    source = cp.asfortranarray(cp.arange(24.0).reshape(2, 3, 4))
    plan = _reduction_plan(source.shape, "F", layout)

    actual = execute_plan(
        cupy_backend, plan, ExecutionBindings({"input_0": source})
    )

    cp.testing.assert_array_equal(actual, source.sum(axis=2))
    assert actual.flags.c_contiguous if layout == "C" else actual.flags.f_contiguous


def test_cupy_executor_uses_explicit_stream_without_synchronizing(cupy_backend, monkeypatch):
    cp = cupy_backend._cupy
    arrays = {
        "left": np.arange(24.0).reshape(2, 3, 4),
        "right": np.arange(90.0).reshape(3, 5, 6),
    }
    plan, bindings, _ = _device_plan(cupy_backend, "abc,bde->adce", arrays)
    stream = cp.cuda.Stream(non_blocking=True)
    synchronize_calls = []
    monkeypatch.setattr(cupy_backend, "sync", lambda: synchronize_calls.append(True))

    actual = execute_plan(cupy_backend, plan, bindings, stream=stream)

    assert isinstance(actual, cp.ndarray)
    assert actual.device.id == 0
    assert synchronize_calls == []
    stream.synchronize()
    np.testing.assert_allclose(cp.asnumpy(actual), np.einsum("abc,bde->adce", *arrays.values()))


def test_cupy_executor_accepts_null_stream_on_selected_device(cupy_backend):
    cp = cupy_backend._cupy
    arrays = {"left": np.arange(6.0).reshape(2, 3), "right": np.arange(12.0).reshape(3, 4)}
    plan, bindings, _ = _device_plan(cupy_backend, "ab,bc->ac", arrays)

    actual = execute_plan(cupy_backend, plan, bindings, stream=cp.cuda.Stream.null)

    cp.cuda.Stream.null.synchronize()
    np.testing.assert_allclose(cp.asnumpy(actual), arrays["left"] @ arrays["right"])


def test_cupy_backend_matmul_directly_honors_explicit_stream(
    cupy_backend, monkeypatch
):
    cp = cupy_backend._cupy
    left = cupy_backend.asarray(np.arange(6.0).reshape(2, 3))
    right = cupy_backend.asarray(np.arange(12.0).reshape(3, 4))
    stream = cp.cuda.Stream(non_blocking=True)
    original_matmul = cp.matmul
    stream_ptrs = []
    synchronize_calls = []

    def recording_matmul(a, b):
        stream_ptrs.append(cp.cuda.get_current_stream().ptr)
        return original_matmul(a, b)

    monkeypatch.setattr(cp, "matmul", recording_matmul)
    monkeypatch.setattr(cupy_backend, "sync", lambda: synchronize_calls.append(True))

    actual = cupy_backend.matmul(left, right, stream=stream)

    assert stream_ptrs == [stream.ptr]
    assert synchronize_calls == []
    stream.synchronize()
    np.testing.assert_allclose(cp.asnumpy(actual), cp.asnumpy(left) @ cp.asnumpy(right))


def test_cupy_rejects_host_or_wrong_device_bindings_before_matmul(cupy_backend, monkeypatch):
    cp = cupy_backend._cupy
    host = {"left": np.zeros((2, 3)), "right": np.zeros((3, 4))}
    plan, _ = plan_einsum("ab,bc->ac", host)
    calls = []
    monkeypatch.setattr(cupy_backend, "matmul", lambda *args, **kwargs: calls.append(True))

    with pytest.raises(TypeError, match="CuPy device array"):
        execute_plan(
            cupy_backend,
            plan,
            ExecutionBindings(
                {"input_0": host["left"], "input_1": host["right"]}
            ),
        )
    assert calls == []
    assert all(not isinstance(value, cp.ndarray) for value in host.values())

    if cp.cuda.runtime.getDeviceCount() > 1:
        with cp.cuda.Device(1):
            wrong = cp.zeros((2, 3))
        right = cupy_backend.asarray(host["right"])
        with pytest.raises(ValueError, match="selected CUDA device"):
            execute_plan(
                cupy_backend,
                plan,
                ExecutionBindings({"input_0": wrong, "input_1": right}),
            )
        assert calls == []


def test_cupy_workspace_preflight_happens_before_primitives(cupy_backend, monkeypatch):
    arrays = {
        "left": np.arange(24.0).reshape(2, 3, 4),
        "right": np.arange(90.0).reshape(3, 5, 6),
    }
    plan, bindings, _ = _device_plan(cupy_backend, "abc,bde->adce", arrays)
    calls = []
    monkeypatch.setattr(cupy_backend, "transpose", lambda *args, **kwargs: calls.append(True))
    monkeypatch.setattr(cupy_backend, "reshape", lambda *args, **kwargs: calls.append(True))
    monkeypatch.setattr(cupy_backend, "matmul", lambda *args, **kwargs: calls.append(True))

    with pytest.raises(ValueError, match="workspace capacity"):
        execute_plan(cupy_backend, plan, bindings, workspace=plan.workspace_bytes - 1)
    assert calls == []


def test_cupy_executor_never_calls_numpy_computation(cupy_backend, monkeypatch):
    arrays = {"left": np.arange(6.0).reshape(2, 3), "right": np.arange(12.0).reshape(3, 4)}
    plan, bindings, _ = _device_plan(cupy_backend, "ab,bc->ac", arrays)
    monkeypatch.setattr(np, "matmul", lambda *args, **kwargs: pytest.fail("NumPy fallback"))
    monkeypatch.setattr(np, "einsum", lambda *args, **kwargs: pytest.fail("NumPy fallback"))
    monkeypatch.setattr(np, "tensordot", lambda *args, **kwargs: pytest.fail("NumPy fallback"))

    actual = execute_plan(cupy_backend, plan, bindings)

    assert isinstance(actual, cupy_backend._cupy.ndarray)


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
def test_cupy_exact_reshape_identity_accepts_ordinary_and_scalar_views(
    cupy_backend, equation, arrays
):
    cp = cupy_backend._cupy
    plan, bindings, _ = _device_plan(cupy_backend, equation, arrays)

    actual = execute_plan(cupy_backend, plan, bindings)

    np.testing.assert_allclose(
        cp.asnumpy(actual), np.einsum(equation, *arrays.values())
    )


def test_cupy_exact_reshape_identity_accepts_empty_view(
    cupy_backend, monkeypatch
):
    cp = cupy_backend._cupy
    arrays = {"left": np.empty((2, 0)), "right": np.empty((0, 3))}
    plan, bindings, _ = _device_plan(cupy_backend, "ab,bc->ac", arrays)
    original_matmul = cupy_backend.matmul
    matmul_calls = []

    def recording_matmul(a, b, *, stream=None, workspace=None):
        matmul_calls.append((a, b))
        return original_matmul(a, b, stream=stream, workspace=workspace)

    monkeypatch.setattr(cupy_backend, "matmul", recording_matmul)

    actual = execute_plan(cupy_backend, plan, bindings)

    cp.testing.assert_array_equal(actual, cp.zeros((2, 3)))
    assert len(matmul_calls) == 1


def _assert_cupy_reshape_rejected_before_matmul(
    cupy_backend, monkeypatch, left, right, replacement, equation="ab,bc->ac"
):
    plan, bindings = plan_einsum(
        equation, {"left": left, "right": right}
    )
    original_reshape = cupy_backend.reshape
    matmul_calls = []

    def injected_reshape(value, shape):
        if value is left:
            assert replacement.shape == tuple(shape)
            return replacement
        return original_reshape(value, shape)

    def fail_matmul(*args, **kwargs):
        matmul_calls.append(True)
        raise AssertionError("matmul ran after invalid reshape alias")

    monkeypatch.setattr(cupy_backend, "reshape", injected_reshape)
    monkeypatch.setattr(cupy_backend, "matmul", fail_matmul)

    with pytest.raises(ValueError, match="reshape unexpectedly copied"):
        execute_plan(cupy_backend, plan, bindings)
    assert matmul_calls == []


def test_cupy_exact_reshape_identity_rejects_zero_stride_overlap(
    cupy_backend, monkeypatch
):
    cp = cupy_backend._cupy
    left = cp.arange(6.0).reshape(2, 1, 3)
    right = cp.arange(12.0).reshape(3, 4)
    overlapping = cp.lib.stride_tricks.as_strided(
        left, shape=(2, 3), strides=(0, 0)
    )
    assert cp.shares_memory(left, overlapping)
    assert not overlapping.flags.c_contiguous

    _assert_cupy_reshape_rejected_before_matmul(
        cupy_backend,
        monkeypatch,
        left,
        right,
        overlapping,
        equation="abc,cd->abd",
    )


@pytest.mark.parametrize("target", [(1, 1), (1, 0)])
def test_cupy_degenerate_zero_stride_reshape_rejected_before_matmul(
    cupy_backend, monkeypatch, target
):
    cp = cupy_backend._cupy
    if target == (1, 1):
        left = cp.empty((1,), dtype=cp.float64)
        right = cp.empty((1,), dtype=cp.float64)
        equation = "a,a->"
    else:
        left = cp.empty((1, 0, 1), dtype=cp.float64)
        right = cp.empty((0, 1, 1), dtype=cp.float64)
        equation = "abc,bcd->ad"
    zero_stride = cp.lib.stride_tricks.as_strided(
        left, shape=target, strides=(0, 0)
    )
    assert zero_stride.strides == (0, 0)
    assert zero_stride.flags.c_contiguous
    assert cp.shares_memory(left, zero_stride) or left.size == 0

    _assert_cupy_reshape_rejected_before_matmul(
        cupy_backend,
        monkeypatch,
        left,
        right,
        zero_stride,
        equation=equation,
    )


def test_cupy_exact_reshape_identity_rejects_partial_offset_overlap(
    cupy_backend, monkeypatch
):
    cp = cupy_backend._cupy
    backing = cp.arange(7.0)
    left = backing[:6].reshape(2, 1, 3)
    right = cp.arange(12.0).reshape(3, 4)
    offset = backing[1:].reshape(2, 3)
    assert cp.shares_memory(left, offset)
    assert left.flags.c_contiguous and offset.flags.c_contiguous

    _assert_cupy_reshape_rejected_before_matmul(
        cupy_backend,
        monkeypatch,
        left,
        right,
        offset,
        equation="abc,cd->abd",
    )


def test_cupy_exact_reshape_identity_rejects_unrelated_empty_view(
    cupy_backend, monkeypatch
):
    cp = cupy_backend._cupy
    left = cp.empty((1, 0, 1), dtype=cp.float64)
    right = cp.empty((0, 1, 1), dtype=cp.float64)
    unrelated = cp.empty((1, 0), dtype=left.dtype).view()

    _assert_cupy_reshape_rejected_before_matmul(
        cupy_backend,
        monkeypatch,
        left,
        right,
        unrelated,
        equation="abc,bcd->ad",
    )


def test_cupy_stream_none_uses_selected_device_and_preserves_ambient_context(
    cupy_backend, monkeypatch
):
    cp = cupy_backend._cupy
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("requires two visible CUDA devices")
    arrays = {"left": np.arange(6.0).reshape(2, 3), "right": np.arange(12.0).reshape(3, 4)}
    plan, bindings, _ = _device_plan(cupy_backend, "ab,bc->ac", arrays)
    original_device = cp.cuda.runtime.getDevice()
    synchronize_calls = []
    monkeypatch.setattr(cupy_backend, "sync", lambda: synchronize_calls.append(True))
    try:
        with cp.cuda.Device(1):
            ambient_stream = cp.cuda.Stream(non_blocking=True)
            with ambient_stream:
                ambient_device = cp.cuda.runtime.getDevice()
                ambient_stream_ptr = cp.cuda.get_current_stream().ptr

                actual = execute_plan(cupy_backend, plan, bindings, stream=None)

                assert actual.device.id == 0
                assert cp.cuda.runtime.getDevice() == ambient_device == 1
                assert cp.cuda.get_current_stream().ptr == ambient_stream_ptr
        assert synchronize_calls == []
        with cp.cuda.Device(0):
            cp.cuda.Device(0).synchronize()
        np.testing.assert_allclose(cp.asnumpy(actual), arrays["left"] @ arrays["right"])
    finally:
        cp.cuda.Device(original_device).use()


def test_cupy_grouped_ir_executes_on_selected_stream_without_sync(
    cupy_backend, monkeypatch
):
    cp = cupy_backend._cupy
    refs = (
        _ref("input_0", (2, 3), ("a", "b")),
        _ref("input_1", (3, 4), ("b", "c")),
        _ref("input_2", (2, 3), ("d", "e")),
        _ref("input_3", (3, 4), ("e", "f")),
    )
    outputs = (
        _ref("output", (2, 4), ("a", "c")),
        _ref("other", (2, 4), ("d", "f")),
    )
    grouped = GroupedMatmulStep(
        (
            MatmulStep(refs[0], refs[1], outputs[0], ("b",)),
            MatmulStep(refs[2], refs[3], outputs[1], ("e",)),
        )
    )
    plan = _specialized_plan(refs, outputs[0], (grouped,))
    host = {
        ref.key: np.arange(np.prod(ref.spec.shape), dtype=np.float64).reshape(ref.spec.shape)
        for ref in refs
    }
    bindings = ExecutionBindings(
        {key: cupy_backend.asarray(value) for key, value in host.items()}
    )
    stream = cp.cuda.Stream(non_blocking=True)
    observed = []
    original_batched = cupy_backend.batched_matmul

    def batched(a, b, *, stream=None, workspace=None):
        observed.append((cp.cuda.runtime.getDevice(), cp.cuda.get_current_stream().ptr))
        return original_batched(a, b, stream=stream, workspace=workspace)

    monkeypatch.setattr(cupy_backend, "batched_matmul", batched)
    monkeypatch.setattr(cupy_backend, "sync", lambda: pytest.fail("IR synchronized"))

    actual = execute_plan(cupy_backend, plan, bindings, stream=stream)

    assert actual.device.id == 0
    assert observed == [(0, stream.ptr)]
