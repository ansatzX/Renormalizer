import dataclasses

import numpy as np
import pytest

from renormalizer import set_backend
from renormalizer.backend._gemm.descriptors import MatmulDesc
from renormalizer.backend._gemm.grouping import GemmGroupKey, group_descriptors


def test_descriptor_is_frozen_and_normalizes_transpose_flags():
    descriptor = MatmulDesc("a", "b", "c", 2, 3, 4, "t", "c", 2, 0)

    assert descriptor.trans_a == "T"
    assert descriptor.trans_b == "C"
    assert descriptor.alpha == 2
    assert descriptor.beta == 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.m = 7


@pytest.mark.parametrize("field", ["a_key", "b_key", "c_key"])
@pytest.mark.parametrize("value", [None, "", "   ", 1])
def test_descriptor_rejects_invalid_keys(field, value):
    values = {"a_key": "a", "b_key": "b", "c_key": "c", "m": 2, "n": 3, "k": 4}
    values[field] = value

    with pytest.raises((TypeError, ValueError), match="key"):
        MatmulDesc(**values)


@pytest.mark.parametrize("field", ["m", "n", "k"])
@pytest.mark.parametrize("value", [-1, True, 1.5, np.int64(2)])
def test_descriptor_requires_nonnegative_python_dimensions(field, value):
    values = {"a_key": "a", "b_key": "b", "c_key": "c", "m": 2, "n": 3, "k": 4}
    values[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        MatmulDesc(**values)


@pytest.mark.parametrize("field", ["trans_a", "trans_b"])
@pytest.mark.parametrize("value", ["", "H", None, 1])
def test_descriptor_rejects_invalid_transpose_flags(field, value):
    values = {"a_key": "a", "b_key": "b", "c_key": "c", "m": 2, "n": 3, "k": 4}
    values[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        MatmulDesc(**values)


@pytest.mark.parametrize("field", ["alpha", "beta"])
@pytest.mark.parametrize(
    "value",
    [True, np.float64(1), float("inf"), complex(1, float("nan")), "1"],
)
def test_descriptor_requires_finite_python_numeric_scalars(field, value):
    values = {"a_key": "a", "b_key": "b", "c_key": "c", "m": 2, "n": 3, "k": 4}
    values[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        MatmulDesc(**values)


def test_ragged_tasks_are_stably_bucketed_by_effective_gemm_key():
    tasks = (
        MatmulDesc("a0", "b0", "c0", 16, 16, 16),
        MatmulDesc("a1", "b1", "c1", 32, 16, 8),
        MatmulDesc("a2", "b2", "c2", 16, 16, 16),
        MatmulDesc("a3", "b3", "c3", 16, 16, 16, trans_a="T"),
    )

    groups = group_descriptors(tasks, dtype=np.dtype("<f8"))

    assert tuple(groups) == (
        GemmGroupKey("float64", 16, 16, 16, "N", "N"),
        GemmGroupKey("float64", 32, 16, 8, "N", "N"),
        GemmGroupKey("float64", 16, 16, 16, "T", "N"),
    )
    assert groups[next(iter(groups))] == (tasks[0], tasks[2])
    assert sorted(len(group) for group in groups.values()) == [1, 1, 2]


def test_grouping_requires_descriptors_and_supported_canonical_dtype():
    descriptor = MatmulDesc("a", "b", "c", 2, 3, 4)

    with pytest.raises(TypeError, match="MatmulDesc"):
        group_descriptors((object(),), dtype=np.float64)
    with pytest.raises(ValueError, match="numeric"):
        group_descriptors((descriptor,), dtype=np.dtype("O"))


def test_numpy_batched_matmul_matches_stacked_reference():
    rng = np.random.default_rng(13)
    a = rng.normal(size=(5, 8, 16))
    b = rng.normal(size=(5, 16, 7))
    backend = set_backend("numpy")

    assert backend.supports_batched_matmul is True
    np.testing.assert_allclose(backend.batched_matmul(a, b), np.matmul(a, b))


def test_unsupported_backends_expose_honest_gemm_api():
    from renormalizer.backend.jax_backend import JaxBackend
    from renormalizer.backend.torch_backend import TorchBackend

    for backend_type in (JaxBackend, TorchBackend):
        backend = object.__new__(backend_type)
        assert backend.supports_batched_matmul is False
        assert backend.supports_grouped_gemm is False
        with pytest.raises(NotImplementedError, match="batched matmul"):
            backend.batched_matmul(None, None)
        with pytest.raises(NotImplementedError, match="grouped GEMM"):
            backend.grouped_gemm((), {})
