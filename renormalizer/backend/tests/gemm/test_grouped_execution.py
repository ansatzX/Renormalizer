import importlib
import importlib.util
import sys

import numpy as np
import pytest

from renormalizer import set_backend
from renormalizer.backend._gemm.descriptors import MatmulDesc
from renormalizer.utils import profiling


def _numpy_backend():
    return set_backend("numpy")


def _raw_shapes(m, n, k, trans_a="N", trans_b="N"):
    a_shape = (m, k) if trans_a == "N" else (k, m)
    b_shape = (k, n) if trans_b == "N" else (n, k)
    return a_shape, b_shape


def _op(value, flag):
    if flag == "N":
        return value
    if flag == "T":
        return value.T
    return value.conj().T


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("trans_a", ["N", "T", "C"])
@pytest.mark.parametrize("trans_b", ["N", "T", "C"])
def test_numpy_grouped_gemm_real_complex_transpose_alpha_beta(
    dtype, trans_a, trans_b
):
    rng = np.random.default_rng(41)
    m, n, k = 2, 4, 3
    a_shape, b_shape = _raw_shapes(m, n, k, trans_a, trans_b)

    def values(shape):
        value = rng.normal(size=shape)
        if np.issubdtype(dtype, np.complexfloating):
            value = value + 1j * rng.normal(size=shape)
        return value.astype(dtype)

    tensors = {
        "a0": values(a_shape),
        "b0": values(b_shape),
        "c0": values((m, n)),
        "a1": values(a_shape),
        "b1": values(b_shape),
        "c1": values((m, n)),
    }
    original_c = (tensors["c0"], tensors["c1"])
    old_c = (tensors["c0"].copy(), tensors["c1"].copy())
    alpha = 0.5 + (0.25j if np.issubdtype(dtype, np.complexfloating) else 0)
    beta = -0.75
    descriptors = tuple(
        MatmulDesc(
            "a{}".format(index),
            "b{}".format(index),
            "c{}".format(index),
            m,
            n,
            k,
            trans_a,
            trans_b,
            alpha,
            beta,
        )
        for index in range(2)
    )

    outputs = _numpy_backend().grouped_gemm(descriptors, tensors)

    for index, output in enumerate(outputs):
        expected = alpha * (_op(tensors[f"a{index}"], trans_a) @ _op(tensors[f"b{index}"], trans_b))
        expected += beta * old_c[index]
        np.testing.assert_allclose(output, expected)
        assert tensors[f"c{index}"] is output
        np.testing.assert_array_equal(original_c[index], old_c[index])


def test_complex_c_pair_conjugates_packs_in_place_at_exact_capacity(monkeypatch):
    backend = _numpy_backend()
    rng = np.random.default_rng(91)
    m, n, k = 2, 4, 3
    descriptors = tuple(
        MatmulDesc(
            f"a{index}", f"b{index}", f"c{index}", m, n, k, "C", "C"
        )
        for index in range(2)
    )
    tensors = {}
    for index in range(2):
        tensors[f"a{index}"] = (
            rng.normal(size=(k, m)) + 1j * rng.normal(size=(k, m))
        )
        tensors[f"b{index}"] = (
            rng.normal(size=(n, k)) + 1j * rng.normal(size=(n, k))
        )
    inputs = {key: value.copy() for key, value in tensors.items()}
    required = 2 * (m * k + k * n) * np.dtype("complex128").itemsize
    original_conj = backend._execution_conjugate_into
    conjugations = []

    def recording_conj(value, destination):
        conjugations.append((value, destination))
        return original_conj(value, destination)

    monkeypatch.setattr(backend, "_execution_conjugate_into", recording_conj)

    outputs = backend.grouped_gemm(
        descriptors, tensors, workspace=required
    )

    assert len(conjugations) == 2
    assert all(value is out for value, out in conjugations)
    assert {value.shape for value, _ in conjugations} == {
        (2, m, k),
        (2, k, n),
    }
    for index, output in enumerate(outputs):
        expected = inputs[f"a{index}"].conj().T @ inputs[f"b{index}"].conj().T
        np.testing.assert_allclose(output, expected)
    for key, value in inputs.items():
        np.testing.assert_array_equal(tensors[key], value)


def test_complex_c_singleton_capacity_counts_both_conjugate_temporaries(
    monkeypatch,
):
    backend = _numpy_backend()
    m, n, k = 2, 4, 3
    descriptor = MatmulDesc("a", "b", "c", m, n, k, "C", "C")
    tensors = {
        "a": np.ones((k, m), dtype=np.complex128),
        "b": np.ones((n, k), dtype=np.complex128),
    }
    required = (m * k + k * n) * np.dtype("complex128").itemsize
    calls = []
    original_conj = backend._execution_conjugate_into
    original_matmul = backend._execution_matmul_into
    monkeypatch.setattr(
        backend,
        "_execution_conjugate_into",
        lambda *args, **kwargs: calls.append("conj"),
    )
    monkeypatch.setattr(
        backend,
        "_execution_matmul_into", lambda *args, **kwargs: calls.append("matmul"),
    )

    with pytest.raises(ValueError, match="workspace capacity"):
        backend.grouped_gemm(
            (descriptor,), tensors, workspace=required - 1
        )
    assert calls == []

    monkeypatch.setattr(backend, "_execution_conjugate_into", original_conj)
    monkeypatch.setattr(backend, "_execution_matmul_into", original_matmul)
    (output,) = backend.grouped_gemm(
        (descriptor,), tensors, workspace=required
    )
    np.testing.assert_allclose(output, np.full((m, n), k, dtype=np.complex128))


def test_ragged_singleton_and_pair_use_exact_scalar_and_batched_dispatch(monkeypatch):
    backend = _numpy_backend()
    rng = np.random.default_rng(42)
    descriptors = (
        MatmulDesc("a0", "b0", "c0", 2, 4, 3),
        MatmulDesc("a1", "b1", "c1", 5, 2, 3),
        MatmulDesc("a2", "b2", "c2", 2, 4, 3),
    )
    tensors = {
        "a0": rng.normal(size=(2, 3)),
        "b0": rng.normal(size=(3, 4)),
        "a1": rng.normal(size=(5, 3)),
        "b1": rng.normal(size=(3, 2)),
        "a2": rng.normal(size=(2, 3)),
        "b2": rng.normal(size=(3, 4)),
    }
    original_matmul = backend._execution_matmul_into
    original_batched = backend._execution_batched_matmul_into
    original_copy = backend._execution_copy_into
    calls = {"matmul": 0, "batched": 0, "copy": 0}
    packs = []

    def matmul(a, b, destination, *, workspace=None):
        calls["matmul"] += 1
        return original_matmul(a, b, destination, workspace=workspace)

    def batched(a, b, destination, *, workspace=None):
        calls["batched"] += 1
        result = original_batched(a, b, destination, workspace=workspace)
        packs.append(destination)
        return result

    def copy(source, destination):
        calls["copy"] += 1
        return original_copy(source, destination)

    monkeypatch.setattr(backend, "_execution_matmul_into", matmul)
    monkeypatch.setattr(backend, "_execution_batched_matmul_into", batched)
    monkeypatch.setattr(backend, "_execution_copy_into", copy)

    outputs = backend.grouped_gemm(descriptors, tensors)

    assert calls == {"matmul": 1, "batched": 1, "copy": 4}
    assert np.shares_memory(outputs[0], packs[0])
    assert np.shares_memory(outputs[2], packs[0])
    for descriptor, output in zip(descriptors, outputs):
        np.testing.assert_allclose(output, tensors[descriptor.a_key] @ tensors[descriptor.b_key])


def test_same_shape_pair_never_uses_scalar_matmul(monkeypatch):
    backend = _numpy_backend()
    tensors = {
        "a": np.arange(6.0).reshape(2, 3),
        "b0": np.arange(12.0).reshape(3, 4),
        "b1": np.arange(12.0, 24.0).reshape(3, 4),
    }
    descriptors = (
        MatmulDesc("a", "b0", "c0", 2, 4, 3),
        MatmulDesc("a", "b1", "c1", 2, 4, 3),
    )
    calls = []
    original_batched = backend._execution_batched_matmul_into
    monkeypatch.setattr(
        backend,
        "_execution_matmul_into",
        lambda *args, **kwargs: pytest.fail("same-shape pair used scalar matmul"),
    )

    def batched(*args, **kwargs):
        calls.append(True)
        return original_batched(*args, **kwargs)

    monkeypatch.setattr(backend, "_execution_batched_matmul_into", batched)

    outputs = backend.grouped_gemm(descriptors, tensors)

    assert calls == [True]
    np.testing.assert_allclose(outputs[0], tensors["a"] @ tensors["b0"])
    np.testing.assert_allclose(outputs[1], tensors["a"] @ tensors["b1"])


def test_integer_group_accepts_identity_alpha_beta_defaults():
    backend = _numpy_backend()
    descriptors = (
        MatmulDesc("a0", "b0", "c0", 2, 2, 2),
        MatmulDesc("a1", "b1", "c1", 2, 2, 2),
    )
    tensors = {
        "a0": np.arange(4, dtype=np.int64).reshape(2, 2),
        "b0": np.arange(4, dtype=np.int64).reshape(2, 2),
        "a1": np.arange(4, 8, dtype=np.int64).reshape(2, 2),
        "b1": np.arange(8, 12, dtype=np.int64).reshape(2, 2),
    }

    outputs = backend.grouped_gemm(descriptors, tensors)

    np.testing.assert_array_equal(outputs[0], tensors["a0"] @ tensors["b0"])
    np.testing.assert_array_equal(outputs[1], tensors["a1"] @ tensors["b1"])


def test_same_shape_different_dtypes_remain_separate_singleton_buckets(monkeypatch):
    backend = _numpy_backend()
    descriptors = (
        MatmulDesc("a0", "b0", "c0", 2, 2, 2),
        MatmulDesc("a1", "b1", "c1", 2, 2, 2),
    )
    tensors = {
        "a0": np.eye(2, dtype=np.float32),
        "b0": np.eye(2, dtype=np.float32),
        "a1": np.eye(2, dtype=np.float64),
        "b1": np.eye(2, dtype=np.float64),
    }
    scalar_calls = []
    original_scalar = backend._execution_matmul_into

    def scalar(*args, **kwargs):
        scalar_calls.append(True)
        return original_scalar(*args, **kwargs)

    monkeypatch.setattr(backend, "_execution_matmul_into", scalar)
    monkeypatch.setattr(
        backend,
        "_execution_batched_matmul_into",
        lambda *args, **kwargs: pytest.fail("different dtypes were batched together"),
    )

    outputs = backend.grouped_gemm(descriptors, tensors)

    assert scalar_calls == [True, True]
    assert tuple(output.dtype for output in outputs) == (
        np.dtype("float32"),
        np.dtype("float64"),
    )


def test_malformed_batched_result_is_not_published(monkeypatch):
    backend = _numpy_backend()
    descriptors = (
        MatmulDesc("a0", "b0", "c0", 2, 4, 3),
        MatmulDesc("a1", "b1", "c1", 2, 4, 3),
    )
    tensors = {
        "a0": np.zeros((2, 3)),
        "b0": np.zeros((3, 4)),
        "a1": np.zeros((2, 3)),
        "b1": np.zeros((3, 4)),
    }
    monkeypatch.setattr(
        backend,
        "_execution_batched_matmul_into",
        lambda *args, **kwargs: np.zeros((2, 2, 3)),
    )

    with pytest.raises(ValueError, match="published destination"):
        backend.grouped_gemm(descriptors, tensors)
    assert "c0" not in tensors and "c1" not in tensors


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda tensors: tensors.__setitem__("a1", np.zeros((3, 2))), "shape"),
        (lambda tensors: tensors.__setitem__("b1", np.zeros((3, 4), dtype=np.float32)), "dtype"),
        (lambda tensors: tensors.pop("b1"), "missing"),
    ],
)
def test_invalid_group_fails_before_any_primitive(monkeypatch, mutation, message):
    backend = _numpy_backend()
    descriptors = (
        MatmulDesc("a0", "b0", "c0", 2, 4, 3),
        MatmulDesc("a1", "b1", "c1", 2, 4, 3),
    )
    tensors = {
        "a0": np.zeros((2, 3)),
        "b0": np.zeros((3, 4)),
        "a1": np.zeros((2, 3)),
        "b1": np.zeros((3, 4)),
    }
    mutation(tensors)
    calls = []
    monkeypatch.setattr(backend, "stack", lambda *args, **kwargs: calls.append("stack"))
    monkeypatch.setattr(backend, "matmul", lambda *args, **kwargs: calls.append("matmul"))
    monkeypatch.setattr(
        backend, "batched_matmul", lambda *args, **kwargs: calls.append("batched")
    )

    with pytest.raises((TypeError, ValueError), match=message):
        backend.grouped_gemm(descriptors, tensors)
    assert calls == []
    assert "c0" not in tensors and "c1" not in tensors


def test_beta_requires_existing_output_before_any_primitive(monkeypatch):
    backend = _numpy_backend()
    descriptor = MatmulDesc("a", "b", "c", 2, 4, 3, beta=1)
    tensors = {"a": np.zeros((2, 3)), "b": np.zeros((3, 4))}
    monkeypatch.setattr(
        backend, "matmul", lambda *args, **kwargs: pytest.fail("matmul ran")
    )

    with pytest.raises(ValueError, match="beta.*c"):
        backend.grouped_gemm((descriptor,), tensors)


def test_output_keys_and_arrays_must_be_unique_and_nonoverlapping(monkeypatch):
    backend = _numpy_backend()
    duplicate_keys = (
        MatmulDesc("a0", "b0", "c", 2, 2, 2),
        MatmulDesc("a1", "b1", "c", 2, 2, 2),
    )
    shared = np.zeros((2, 2))
    tensors = {
        "a0": np.zeros((2, 2)),
        "b0": np.zeros((2, 2)),
        "a1": shared,
        "b1": np.zeros((2, 2)),
        "c0": shared,
        "c1": np.zeros((2, 2)),
    }
    monkeypatch.setattr(
        backend, "batched_matmul", lambda *args, **kwargs: pytest.fail("batched ran")
    )

    with pytest.raises(ValueError, match="unique output"):
        backend.grouped_gemm(duplicate_keys, tensors)

    overlapping = (
        MatmulDesc("a0", "b0", "c0", 2, 2, 2),
        MatmulDesc("a1", "b1", "c1", 2, 2, 2),
    )
    with pytest.raises(ValueError, match="overlap"):
        backend.grouped_gemm(overlapping, tensors)

    colliding_key = (MatmulDesc("a0", "b0", "a0", 2, 2, 2),)
    with pytest.raises(ValueError, match="output key.*input"):
        backend.grouped_gemm(colliding_key, tensors)


def test_compute_failure_does_not_publish_any_outputs(monkeypatch):
    backend = _numpy_backend()
    descriptors = (
        MatmulDesc("a0", "b0", "c0", 5, 2, 3),
        MatmulDesc("a1", "b1", "c1", 2, 4, 3),
        MatmulDesc("a2", "b2", "c2", 2, 4, 3),
    )
    tensors = {
        "a0": np.zeros((5, 3)),
        "b0": np.zeros((3, 2)),
        "a1": np.zeros((2, 3)),
        "b1": np.zeros((3, 4)),
        "a2": np.zeros((2, 3)),
        "b2": np.zeros((3, 4)),
    }
    scalar_calls = []
    original_matmul = backend._execution_matmul_into

    def scalar(*args, **kwargs):
        scalar_calls.append(True)
        return original_matmul(*args, **kwargs)

    monkeypatch.setattr(backend, "_execution_matmul_into", scalar)
    monkeypatch.setattr(
        backend,
        "_execution_batched_matmul_into",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("forced failure")),
    )

    with pytest.raises(RuntimeError, match="forced failure"):
        backend.grouped_gemm(descriptors, tensors)
    assert scalar_calls == [True]
    assert not ({"c0", "c1", "c2"} & set(tensors))


def test_scalar_output_overlap_is_rejected_before_destination_launch(monkeypatch):
    backend = _numpy_backend()
    descriptor = MatmulDesc("a", "b", "c", 2, 2, 2)
    tensors = {
        "a": np.arange(4.0).reshape(2, 2),
        "b": np.arange(4.0, 8.0).reshape(2, 2),
    }
    snapshot = tensors["a"].copy()
    launches = []
    monkeypatch.setattr(backend, "empty", lambda *args, **kwargs: tensors["a"])
    monkeypatch.setattr(
        backend,
        "_execution_matmul_into",
        lambda *args, **kwargs: launches.append(True),
    )

    with pytest.raises(ValueError, match="overlap"):
        backend.grouped_gemm((descriptor,), tensors)
    assert launches == []
    np.testing.assert_array_equal(tensors["a"], snapshot)


@pytest.mark.parametrize("alias_key", ["a0", "c0"])
def test_scalar_alias_is_rejected_without_external_mutation(
    monkeypatch, alias_key
):
    backend = _numpy_backend()
    descriptors = (
        MatmulDesc("a0", "b0", "c0", 2, 2, 2, alpha=2, beta=1),
        MatmulDesc("a1", "b1", "c1", 3, 2, 2),
        MatmulDesc("a2", "b2", "c2", 3, 2, 2),
    )
    tensors = {
        "a0": np.arange(4.0).reshape(2, 2),
        "b0": np.arange(4.0, 8.0).reshape(2, 2),
        "c0": np.full((2, 2), 7.0),
        "a1": np.arange(6.0).reshape(3, 2),
        "b1": np.arange(4.0).reshape(2, 2),
        "a2": np.arange(6.0, 12.0).reshape(3, 2),
        "b2": np.arange(4.0, 8.0).reshape(2, 2),
    }
    original_entries = dict(tensors)
    snapshots = {key: value.copy() for key, value in tensors.items()}
    later_calls = []

    monkeypatch.setattr(
        backend,
        "_execution_matmul_into",
        lambda left, right, destination, **kwargs: tensors[alias_key],
    )

    def fail_later(*args, **kwargs):
        later_calls.append(True)
        raise RuntimeError("later bucket failed")

    monkeypatch.setattr(backend, "_execution_batched_matmul_into", fail_later)

    with pytest.raises(ValueError, match="published destination"):
        backend.grouped_gemm(descriptors, tensors)
    assert later_calls == []
    assert tensors.keys() == original_entries.keys()
    for key, original in original_entries.items():
        assert tensors[key] is original
        np.testing.assert_array_equal(tensors[key], snapshots[key])


def test_batched_input_view_alias_is_rejected_without_external_mutation(
    monkeypatch,
):
    backend = _numpy_backend()
    descriptors = (
        MatmulDesc("a0", "b0", "c0", 2, 2, 4, alpha=2, beta=1),
        MatmulDesc("a1", "b1", "c1", 2, 2, 4, alpha=3, beta=-1),
        MatmulDesc("a2", "b2", "c2", 3, 2, 2),
    )
    tensors = {
        "a0": np.arange(8.0).reshape(2, 4),
        "b0": np.arange(8.0, 16.0).reshape(4, 2),
        "c0": np.full((2, 2), 5.0),
        "a1": np.arange(16.0, 24.0).reshape(2, 4),
        "b1": np.arange(24.0, 32.0).reshape(4, 2),
        "c1": np.full((2, 2), 7.0),
        "a2": np.arange(6.0).reshape(3, 2),
        "b2": np.arange(4.0).reshape(2, 2),
    }
    original_entries = dict(tensors)
    snapshots = {key: value.copy() for key, value in tensors.items()}
    later_calls = []

    monkeypatch.setattr(
        backend,
        "_execution_batched_matmul_into",
        lambda *args, **kwargs: tensors["a0"].reshape(2, 2, 2),
    )

    def fail_later(*args, **kwargs):
        later_calls.append(True)
        raise RuntimeError("later bucket failed")

    monkeypatch.setattr(backend, "_execution_matmul_into", fail_later)

    with pytest.raises(ValueError, match="published destination"):
        backend.grouped_gemm(descriptors, tensors)
    assert later_calls == []
    assert tensors.keys() == original_entries.keys()
    for key, original in original_entries.items():
        assert tensors[key] is original
        np.testing.assert_array_equal(tensors[key], snapshots[key])


def test_grouped_packing_workspace_is_preflighted_before_stack_or_matmul(monkeypatch):
    backend = _numpy_backend()
    descriptors = (
        MatmulDesc("a0", "b0", "c0", 2, 4, 3),
        MatmulDesc("a1", "b1", "c1", 2, 4, 3),
    )
    tensors = {
        "a0": np.zeros((2, 3)),
        "b0": np.zeros((3, 4)),
        "a1": np.zeros((2, 3)),
        "b1": np.zeros((3, 4)),
    }
    required = 2 * (2 * 3 + 3 * 4) * np.dtype("float64").itemsize
    calls = []
    monkeypatch.setattr(backend, "stack", lambda *args, **kwargs: calls.append("stack"))
    monkeypatch.setattr(
        backend, "batched_matmul", lambda *args, **kwargs: calls.append("batched")
    )

    with pytest.raises(ValueError, match="workspace capacity"):
        backend.grouped_gemm(descriptors, tensors, workspace=required - 1)
    assert calls == []


def test_disabled_profiling_imports_no_gemm_payload_or_timing(monkeypatch):
    backend = _numpy_backend()
    tensors = {"a": np.eye(2), "b": np.eye(2)}
    descriptor = MatmulDesc("a", "b", "c", 2, 2, 2)
    helper_module = "renormalizer.backend._gemm.profiling"
    sys.modules.pop(helper_module, None)
    monkeypatch.setattr(profiling, "enabled", lambda: False)
    monkeypatch.setattr(
        profiling,
        "record",
        lambda *args, **kwargs: pytest.fail("disabled profiling recorded an event"),
    )

    backend.grouped_gemm((descriptor,), tensors)

    assert helper_module not in sys.modules


def test_enabled_profiling_records_bounded_truthful_execution(monkeypatch):
    backend = _numpy_backend()
    descriptors = []
    tensors = {}
    for index in range(35):
        count = 2 if index == 0 else 1
        for task in range(count):
            suffix = f"{index}_{task}"
            descriptors.append(
                MatmulDesc(
                    f"a{suffix}", f"b{suffix}", f"c{suffix}", index + 1, 1, 1
                )
            )
            tensors[f"a{suffix}"] = np.ones((index + 1, 1))
            tensors[f"b{suffix}"] = np.ones((1, 1))
    events = []
    monkeypatch.setattr(profiling, "enabled", lambda: True)
    monkeypatch.setattr(
        profiling, "record", lambda event, **payload: events.append({"event": event, **payload})
    )

    backend.grouped_gemm(tuple(descriptors), tensors, policy="unit_test")

    assert len(events) == 1
    event = events[0]
    assert event["event"] == "grouped_gemm_execute"
    assert event["grouped_execution"] is True
    assert event["task_count"] == 36
    assert event["shape_bucket_count"] == 35
    assert len(event["shape_buckets"]) == 32
    assert event["shape_buckets_truncated"] is True
    assert event["backend"] == "numpy"
    assert event["policy"] == "unit_test"
    assert event["timing_semantics"] == "host_elapsed_synchronous"
    assert event["device_synchronized"] is True
    assert all(event[field] >= 0 for field in ("pack_wall_s", "compute_wall_s", "scatter_wall_s"))


def test_enabled_profiling_does_not_claim_all_singleton_buckets(monkeypatch):
    backend = _numpy_backend()
    descriptors = (
        MatmulDesc("a0", "b0", "c0", 2, 2, 2),
        MatmulDesc("a1", "b1", "c1", 3, 2, 2),
    )
    tensors = {
        "a0": np.ones((2, 2)),
        "b0": np.ones((2, 2)),
        "a1": np.ones((3, 2)),
        "b1": np.ones((2, 2)),
    }
    events = []
    monkeypatch.setattr(profiling, "enabled", lambda: True)
    monkeypatch.setattr(
        profiling, "record", lambda event, **payload: events.append(payload)
    )

    backend.grouped_gemm(descriptors, tensors)

    assert len(events) == 1
    assert events[0]["grouped_execution"] is False
    assert events[0]["task_count"] == 2


@pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="CuPy is not installed")
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("trans_a", ["N", "T", "C"])
@pytest.mark.parametrize("trans_b", ["N", "T", "C"])
def test_cupy_grouped_gemm_real_complex_transpose_alpha_beta(
    dtype, trans_a, trans_b
):
    cp = importlib.import_module("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no visible CUDA device")
    backend = set_backend("cupy", device="cuda:0", precision=64)
    rng = np.random.default_rng(81)
    m, n, k = 2, 4, 3
    a_shape, b_shape = _raw_shapes(m, n, k, trans_a, trans_b)

    def values(shape):
        value = rng.normal(size=shape)
        if np.issubdtype(dtype, np.complexfloating):
            value = value + 1j * rng.normal(size=shape)
        return backend.asarray(value.astype(dtype))

    tensors = {
        "a0": values(a_shape),
        "b0": values(b_shape),
        "c0": values((m, n)),
        "a1": values(a_shape),
        "b1": values(b_shape),
        "c1": values((m, n)),
    }
    old_c = (tensors["c0"].copy(), tensors["c1"].copy())
    alpha = 0.5 + (0.25j if np.issubdtype(dtype, np.complexfloating) else 0)
    beta = -0.75
    descriptors = tuple(
        MatmulDesc(
            f"a{index}",
            f"b{index}",
            f"c{index}",
            m,
            n,
            k,
            trans_a,
            trans_b,
            alpha,
            beta,
        )
        for index in range(2)
    )

    outputs = backend.grouped_gemm(descriptors, tensors)

    for index, output in enumerate(outputs):
        expected = alpha * (
            _op(tensors[f"a{index}"], trans_a)
            @ _op(tensors[f"b{index}"], trans_b)
        )
        expected += beta * old_c[index]
        cp.testing.assert_allclose(output, expected)
        assert output.device.id == 0


@pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="CuPy is not installed")
def test_cupy_complex_c_pair_conjugates_packs_in_place_at_exact_capacity(
    monkeypatch,
):
    cp = importlib.import_module("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no visible CUDA device")
    backend = set_backend("cupy", device="cuda:0", precision=64)
    m, n, k = 2, 4, 3
    descriptors = tuple(
        MatmulDesc(
            f"a{index}", f"b{index}", f"c{index}", m, n, k, "C", "C"
        )
        for index in range(2)
    )
    tensors = {
        "a0": backend.asarray(
            np.arange(k * m).reshape(k, m) + 1j * np.ones((k, m))
        ),
        "b0": backend.asarray(
            np.arange(n * k).reshape(n, k) + 2j * np.ones((n, k))
        ),
        "a1": backend.asarray(
            np.arange(k * m, 2 * k * m).reshape(k, m) + 3j * np.ones((k, m))
        ),
        "b1": backend.asarray(
            np.arange(n * k, 2 * n * k).reshape(n, k) + 4j * np.ones((n, k))
        ),
    }
    inputs = {key: value.copy() for key, value in tensors.items()}
    required = 2 * (m * k + k * n) * np.dtype("complex128").itemsize
    original_conj = backend._execution_conjugate_into
    conjugations = []

    def recording_conj(value, destination):
        conjugations.append((value, destination))
        return original_conj(value, destination)

    monkeypatch.setattr(backend, "_execution_conjugate_into", recording_conj)

    outputs = backend.grouped_gemm(
        descriptors, tensors, workspace=required
    )

    assert len(conjugations) == 2
    assert all(value is out for value, out in conjugations)
    for index, output in enumerate(outputs):
        expected = inputs[f"a{index}"].conj().T @ inputs[f"b{index}"].conj().T
        cp.testing.assert_allclose(output, expected)
    for key, value in inputs.items():
        cp.testing.assert_array_equal(tensors[key], value)


@pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="CuPy is not installed")
def test_cupy_complex_c_singleton_exact_capacity_and_numerics(monkeypatch):
    cp = importlib.import_module("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no visible CUDA device")
    backend = set_backend("cupy", device="cuda:0", precision=64)
    m, n, k = 2, 4, 3
    descriptor = MatmulDesc("a", "b", "c", m, n, k, "C", "C")
    tensors = {
        "a": backend.asarray(np.ones((k, m), dtype=np.complex128)),
        "b": backend.asarray(np.ones((n, k), dtype=np.complex128)),
    }
    required = (m * k + k * n) * np.dtype("complex128").itemsize
    calls = []
    original_matmul = backend._execution_matmul_into

    def recording_matmul(*args, **kwargs):
        calls.append(True)
        return original_matmul(*args, **kwargs)

    monkeypatch.setattr(backend, "_execution_matmul_into", recording_matmul)

    with pytest.raises(ValueError, match="workspace capacity"):
        backend.grouped_gemm(
            (descriptor,), tensors, workspace=required - 1
        )
    assert calls == []

    (output,) = backend.grouped_gemm(
        (descriptor,), tensors, workspace=required
    )
    cp.testing.assert_allclose(
        output, cp.full((m, n), k, dtype=cp.complex128)
    )
    assert calls == [True]


@pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="CuPy is not installed")
def test_cupy_grouped_gemm_uses_selected_device_stream_without_sync_or_host_transfer(monkeypatch):
    cp = importlib.import_module("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no visible CUDA device")
    backend = set_backend("cupy", device="cuda:0", precision=64)
    stream = cp.cuda.Stream(non_blocking=True)
    tensors = {
        "a0": backend.asarray(np.arange(6.0).reshape(2, 3)),
        "b0": backend.asarray(np.arange(12.0).reshape(3, 4)),
        "a1": backend.asarray(np.arange(6.0, 12.0).reshape(2, 3)),
        "b1": backend.asarray(np.arange(12.0, 24.0).reshape(3, 4)),
    }
    descriptors = (
        MatmulDesc("a0", "b0", "c0", 2, 4, 3),
        MatmulDesc("a1", "b1", "c1", 2, 4, 3),
    )
    monkeypatch.setattr(backend, "sync", lambda: pytest.fail("grouped GEMM synchronized"))
    monkeypatch.setattr(backend, "to_numpy", lambda value: pytest.fail("host transfer"))
    observed = []
    events = []
    original_batched = backend._execution_batched_matmul_into

    def batched(a, b, destination, *, workspace=None):
        observed.append((cp.cuda.runtime.getDevice(), cp.cuda.get_current_stream().ptr))
        return original_batched(a, b, destination, workspace=workspace)

    monkeypatch.setattr(backend, "_execution_batched_matmul_into", batched)
    monkeypatch.setattr(profiling, "enabled", lambda: True)
    monkeypatch.setattr(
        profiling, "record", lambda event, **payload: events.append(payload)
    )

    outputs = backend.grouped_gemm(descriptors, tensors, stream=stream)

    assert observed == [(0, stream.ptr)]
    assert all(output.device.id == 0 for output in outputs)
    assert events[0]["timing_semantics"] == "host_elapsed_unsynchronized"
    assert events[0]["device_synchronized"] is False


@pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="CuPy is not installed")
def test_cupy_grouped_gemm_preserves_ambient_second_device(monkeypatch):
    cp = importlib.import_module("cupy")
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("two visible CUDA devices required")
    backend = set_backend("cupy", device="cuda:0", precision=64)
    stream = cp.cuda.Stream(non_blocking=True)
    tensors = {
        "a0": backend.asarray(np.eye(2)),
        "b0": backend.asarray(np.eye(2)),
        "a1": backend.asarray(np.eye(2)),
        "b1": backend.asarray(np.eye(2)),
    }
    descriptors = (
        MatmulDesc("a0", "b0", "c0", 2, 2, 2),
        MatmulDesc("a1", "b1", "c1", 2, 2, 2),
    )
    monkeypatch.setattr(backend, "sync", lambda: pytest.fail("grouped GEMM synchronized"))

    with cp.cuda.Device(1):
        ambient = cp.cuda.Stream(non_blocking=True)
        with ambient:
            backend.grouped_gemm(descriptors, tensors, stream=stream)
            assert cp.cuda.runtime.getDevice() == 1
            assert cp.cuda.get_current_stream().ptr == ambient.ptr
