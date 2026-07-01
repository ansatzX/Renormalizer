# -*- coding: utf-8 -*-

import numpy as np
import pytest


def test_device_spec_parses_cpu_and_indexed_cuda_aliases():
    from renormalizer.backend.execution import DeviceSpec, parse_device_spec

    assert parse_device_spec("cpu") == DeviceSpec(kind="cpu")
    assert parse_device_spec("gpu") == DeviceSpec(kind="cuda")
    assert parse_device_spec("cuda:1") == DeviceSpec(kind="cuda", index=1, visible_id="1")
    assert parse_device_spec("gpu:2") == DeviceSpec(kind="cuda", index=2, visible_id="2")

    with pytest.raises(ValueError, match="Unknown backend device"):
        parse_device_spec("quantum:0")


def test_backend_config_keeps_legacy_device_kind_and_exposes_device_spec():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.execution import DeviceSpec, FallbackPolicy

    config = BackendConfig(device="cuda:1", fallback_policy="record")

    assert config.device == "gpu"
    assert config.device_spec == DeviceSpec(kind="cuda", index=1, visible_id="1")
    assert config.fallback_policy is FallbackPolicy.RECORD


def test_numpy_backend_capabilities_and_array_info_are_explicit():
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.backend.execution import DeviceSpec

    backend = NumpyBackend()
    x = np.arange(6, dtype=np.float64).reshape(2, 3)
    info = backend.array_info(x)

    assert backend.capabilities.matmul is True
    assert backend.capabilities.batched_matmul is False
    assert backend.capabilities.grouped_gemm is False
    assert backend.current_device() == DeviceSpec(kind="cpu")
    assert backend.device_count() == 1

    assert info.shape == (2, 3)
    assert info.dtype == np.dtype("float64")
    assert info.itemsize == 8
    assert info.nbytes == 48
    assert info.device == DeviceSpec(kind="cpu")
    assert info.is_host is True
    assert info.is_device is False
    assert info.order == "C"
    assert info.contiguous is True
    assert info.backend_name == "numpy"


def test_numpy_copy_policy_rejects_required_copy_and_allows_explicit_copy():
    from renormalizer.backend.execution import BackendCopyError, CopyPolicy
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    x = np.array([1.0, 2.0])

    y = backend.to_backend(x, copy=CopyPolicy.NEVER)
    assert y is x

    z = backend.to_backend(x, copy=CopyPolicy.ALWAYS)
    assert np.array_equal(z, x)
    assert not np.shares_memory(z, x)

    with pytest.raises(BackendCopyError):
        backend.to_backend([1.0, 2.0], copy=CopyPolicy.NEVER)


def test_pair_contraction_lowering_reports_gemm_shape_and_costs():
    from renormalizer.backend.execution import PairContractionSpec, TensorOperand
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    a = np.ones((2, 3))
    b = np.ones((3, 4))
    spec = PairContractionSpec.from_operands(
        TensorOperand(a, ("i", "k"), name="A"),
        TensorOperand(b, ("k", "j"), name="B"),
        output_modes=("i", "j"),
    )

    plan = backend.lower_pair_contraction_to_matmul(spec)

    assert plan.kind == "gemm"
    assert plan.output_shape == (2, 4)
    assert len(plan.descs) == 1
    desc = plan.descs[0]
    assert (desc.m, desc.n, desc.k) == (2, 4, 3)
    assert desc.batch_shape == ()
    assert plan.estimated_flops == 2 * 2 * 4 * 3
    assert plan.copy_bytes == 0
    assert plan.fallback_reason is None


def test_pair_contraction_lowering_records_batched_fallback_reason():
    from renormalizer.backend.execution import PairContractionSpec, TensorOperand
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    a = np.ones((5, 2, 3))
    b = np.ones((5, 3, 4))
    spec = PairContractionSpec.from_operands(
        TensorOperand(a, ("batch", "i", "k")),
        TensorOperand(b, ("batch", "k", "j")),
        output_modes=("batch", "i", "j"),
    )

    plan = backend.lower_pair_contraction_to_matmul(spec)

    assert plan.kind == "fallback_tensordot"
    assert plan.descs[0].batch_shape == (5,)
    assert "batched_matmul" in plan.fallback_reason
