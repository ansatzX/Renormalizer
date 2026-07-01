# -*- coding: utf-8 -*-

import json

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


def test_execute_matmul_plan_runs_gemm_and_records_execute_event(tmp_path):
    from renormalizer.backend.execution import PairContractionSpec, TensorOperand
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    spec = PairContractionSpec.from_operands(
        TensorOperand(left, ("i", "k"), name="left"),
        TensorOperand(right, ("k", "j"), name="right"),
        output_modes=("i", "j"),
    )
    plan = backend.lower_pair_contraction_to_matmul(spec)
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = backend.execute_matmul_plan(plan)
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(result, left @ right)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    executes = [payload for payload in payloads if payload["event"] == "contraction_execute"]

    assert len(executes) == 1
    event = executes[0]
    assert event["backend"] == "numpy"
    assert event["lowering"] == "gemm"
    assert event["input_shapes"] == [[2, 3], [3, 4]]
    assert event["output_shape"] == [2, 4]
    assert event["dtype"] == "float64"
    assert event["flops"] == 48
    assert event["read_bytes"] == 144
    assert event["write_bytes"] == 64
    assert event["num_gemm"] == 1
    assert event["num_batched_gemm"] == 0
    assert event["fallback_reason"] is None
    assert event["wall_s"] >= 0.0


def test_execute_matmul_plan_preserves_generic_output_mode_order_for_fallback():
    from renormalizer.backend.execution import PairContractionSpec, TensorOperand
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    right = np.arange(2 * 4 * 5, dtype=np.float64).reshape(2, 4, 5)
    spec = PairContractionSpec.from_operands(
        TensorOperand(left, ("batch", "i", "k"), name="left"),
        TensorOperand(right, ("batch", "k", "j"), name="right"),
        output_modes=("j", "batch", "i"),
    )

    plan = backend.lower_pair_contraction_to_matmul(spec)
    result = backend.execute_matmul_plan(plan)
    expected = np.einsum("bik,bkj->bij", left, right).transpose(2, 0, 1)

    assert plan.kind == "fallback_tensordot"
    assert result.shape == (5, 2, 3)
    assert np.allclose(result, expected)


def test_backend_contraction_plan_event_records_generic_operands(tmp_path):
    from renormalizer.backend.execution import PairContractionSpec, TensorOperand
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    event_path = tmp_path / "events.jsonl"
    left = np.ones((2, 3, 4), dtype=np.float32)
    right = np.ones((2, 4, 5), dtype=np.float32)
    spec = PairContractionSpec.from_operands(
        TensorOperand(left, ("batch", ("left", "site"), "bond"), name="left_tensor"),
        TensorOperand(right, ("batch", "bond", ("right", "site")), name="right_tensor"),
        output_modes=("batch", ("left", "site"), ("right", "site")),
    )
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        backend.lower_pair_contraction_to_matmul(spec)
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    plan = next(payload for payload in payloads if payload["event"] == "contraction_plan")

    assert plan["lowering"] == "fallback_tensordot"
    assert plan["batch_modes"] == ["batch"]
    assert plan["output_modes"] == ["batch", "('left', 'site')", "('right', 'site')"]
    assert plan["operands"] == [
        {
            "name": "left_tensor",
            "modes": ["batch", "('left', 'site')", "bond"],
            "shape": [2, 3, 4],
            "dtype": "float32",
            "nbytes": 96,
            "ndim": 3,
            "strides": [48, 16, 4],
            "order": "C",
            "contiguous": True,
            "backend": "numpy",
            "device": "DeviceSpec(kind='cpu', index=None, local_rank=None, global_rank=None, visible_id=None)",
            "device_kind": "cpu",
            "device_index": None,
            "is_host": True,
            "is_device": False,
        },
        {
            "name": "right_tensor",
            "modes": ["batch", "bond", "('right', 'site')"],
            "shape": [2, 4, 5],
            "dtype": "float32",
            "nbytes": 160,
            "ndim": 3,
            "strides": [80, 20, 4],
            "order": "C",
            "contiguous": True,
            "backend": "numpy",
            "device": "DeviceSpec(kind='cpu', index=None, local_rank=None, global_rank=None, visible_id=None)",
            "device_kind": "cpu",
            "device_index": None,
            "is_host": True,
            "is_device": False,
        },
    ]


def test_pair_tensor_contract_records_generic_contraction_plan(tmp_path):
    from renormalizer.mps.matrix import pair_tensor_contract
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    old_level = package_logger.level
    event_path = tmp_path / "events.jsonl"
    left = np.ones((2, 3))
    right = np.ones((3, 4))
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        result = pair_tensor_contract(left, "ik", right, "kj", {"k"})
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(result, np.tensordot(left, right, axes=((1,), (0,))))

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    plans = [payload for payload in payloads if payload["event"] == "contraction_plan"]
    executes = [payload for payload in payloads if payload["event"] == "contraction_execute"]

    assert len(plans) == 1
    assert len(executes) == 1
    plan = plans[0]
    assert plan["backend"] == "numpy"
    assert plan["lowering"] == "gemm"
    assert plan["left_modes"] == ["i", "k"]
    assert plan["right_modes"] == ["k", "j"]
    assert plan["output_modes"] == ["i", "j"]
    assert plan["contracted_modes"] == ["k"]
    assert plan["input_shapes"] == [[2, 3], [3, 4]]
    assert plan["output_shape"] == [2, 4]
    assert plan["flops"] == 48
    assert plan["num_gemm"] == 1
    assert plan["fallback_reason"] is None
    execute = executes[0]
    assert execute["backend"] == "numpy"
    assert execute["lowering"] == "gemm"
    assert execute["input_shapes"] == [[2, 3], [3, 4]]
    assert execute["output_shape"] == [2, 4]
    assert execute["flops"] == 48
    assert execute["num_gemm"] == 1
    assert execute["fallback_reason"] is None
    assert execute["wall_s"] >= 0.0
