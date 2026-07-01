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
    assert backend.capabilities.batched_matmul is True
    assert backend.capabilities.grouped_gemm is False
    assert backend.supports_batched_matmul is True
    assert backend.supports_grouped_gemm is False
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


def test_numpy_layout_transform_api_tracks_view_and_contiguous_copy():
    from renormalizer.backend.execution import BackendCopyError
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    x = np.arange(12, dtype=np.float64).reshape(3, 4)

    transposed = backend.permute(x, (1, 0))
    assert np.array_equal(transposed, x.T)
    assert np.shares_memory(transposed, x)

    assert backend.can_reshape_view(x, (4, 3)) is True
    reshaped = backend.reshape_view(x, (4, 3))
    assert reshaped.shape == (4, 3)
    assert np.shares_memory(reshaped, x)

    strided = x[::2, ::2]
    assert backend.can_reshape_view(strided, (4,)) is False
    with pytest.raises(BackendCopyError, match="reshape would require a copy"):
        backend.reshape_view(strided, (4,))

    contiguous = backend.make_contiguous(transposed)
    assert contiguous.flags["C_CONTIGUOUS"]
    assert np.array_equal(contiguous, transposed)
    assert not np.shares_memory(contiguous, transposed)


def test_backend_synchronize_alias_delegates_to_sync():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    calls = []

    def fake_sync():
        calls.append("sync")
        return "synced"

    backend.sync = fake_sync

    assert backend.synchronize(device=None, stream=None) == "synced"
    assert calls == ["sync"]


def test_jax_layout_transform_api_allows_same_size_device_reshape_when_available():
    try:
        __import__("jax")
    except (ImportError, OSError) as exc:
        pytest.skip("jax unavailable: {0}".format(exc))

    from renormalizer.backend import BackendConfig
    from renormalizer.backend.jax_backend import JaxBackend

    try:
        backend = JaxBackend(config=BackendConfig(device="cpu"))
    except (ImportError, ValueError, RuntimeError) as exc:
        pytest.skip("jax backend unavailable: {0}".format(exc))
    x = backend.to_backend(np.arange(12, dtype=np.float64).reshape(3, 4))

    assert backend.can_reshape_view(x, (4, 3)) is True
    reshaped = backend.reshape_view(x, (4, 3))
    assert tuple(reshaped.shape) == (4, 3)


def test_packed_vector_spec_roundtrips_single_and_batched_rhs():
    from renormalizer.backend.execution import PackedVectorSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mask = np.array([[True, False, True], [False, True, False]])

    single_spec = PackedVectorSpec(qn_mask=mask, center_shape=mask.shape, packed_dim=3, nrhs=1)
    single = np.array([1.0, 2.0, 3.0])
    single_struct = backend.unpack_masked_vectors(single, single_spec)
    assert single_struct.shape == mask.shape
    assert np.array_equal(single_struct[mask], single)
    assert np.array_equal(single_struct[~mask], np.zeros(np.count_nonzero(~mask)))
    assert np.array_equal(backend.pack_masked_vectors(single_struct, single_spec), single)

    batched_spec = PackedVectorSpec(qn_mask=mask, center_shape=mask.shape, packed_dim=3, nrhs=2)
    batched = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    batched_struct = backend.unpack_masked_vectors(batched, batched_spec)
    assert batched_struct.shape == mask.shape + (2,)
    assert np.array_equal(batched_struct[mask], batched)
    assert np.array_equal(backend.pack_masked_vectors(batched_struct, batched_spec), batched)


def test_packed_vector_spec_supports_non_last_batch_axis():
    from renormalizer.backend.execution import PackedVectorSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mask = np.array([[True, False, True], [False, True, False]])
    spec = PackedVectorSpec(qn_mask=mask, center_shape=mask.shape, batch_axis=0, packed_dim=3, nrhs=2)
    packed = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    struct = backend.unpack_masked_vectors(packed, spec)

    assert struct.shape == (2,) + mask.shape
    assert np.array_equal(np.moveaxis(struct, 0, -1)[mask], packed)
    assert np.array_equal(backend.pack_masked_vectors(struct, spec), packed)


def test_packed_vector_spec_rejects_wrong_packed_shapes():
    from renormalizer.backend.execution import PackedVectorSpec
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    mask = np.array([[True, False, True], [False, True, False]])
    spec = PackedVectorSpec(qn_mask=mask, center_shape=mask.shape, packed_dim=3, nrhs=2)

    with pytest.raises(ValueError, match="packed vector shape"):
        backend.unpack_masked_vectors(np.ones((4, 2)), spec)

    with pytest.raises(ValueError, match="center tensor shape"):
        backend.pack_masked_vectors(np.ones(mask.shape + (3,)), spec)


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


def test_pair_contraction_lowering_reports_batched_gemm_for_same_shape_batch():
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

    assert plan.kind == "batched_gemm"
    assert plan.descs[0].batch_shape == (5,)
    assert plan.fallback_reason is None


def test_backend_parse_einsum_builds_explicit_ir_operands():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    left = np.ones((2, 3), dtype=np.float64)
    right = np.ones((3, 4), dtype=np.float64)

    spec = backend.parse_einsum("ab,bc->ac", left, right, optimize="greedy", constants=(1,))

    assert spec.output_modes == ("a", "c")
    assert spec.optimize == "greedy"
    assert spec.constants == (1,)
    assert [operand.modes for operand in spec.operands] == [("a", "b"), ("b", "c")]
    assert [operand.name for operand in spec.operands] == ["operand0", "operand1"]
    assert spec.operands[0].array is left
    assert spec.operands[1].array is right
    assert spec.operands[0].layout.logical_modes == ("a", "b")
    assert spec.operands[1].layout.logical_modes == ("b", "c")


def test_backend_parse_einsum_rejects_implicit_output():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()

    with pytest.raises(ValueError, match="explicit output"):
        backend.parse_einsum("ab,bc", np.ones((2, 3)), np.ones((3, 4)))


def test_backend_parse_einsum_rejects_operand_rank_mismatch():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()

    with pytest.raises(ValueError, match="rank"):
        backend.parse_einsum("ab,bc->ac", np.ones((2, 3, 1)), np.ones((3, 4)))


def test_backend_raw_matmul_and_stacked_batched_matmul():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    b = np.arange(12, dtype=np.float64).reshape(3, 4)
    batch_a = np.arange(4 * 2 * 3, dtype=np.float64).reshape(4, 2, 3)
    batch_b = np.arange(4 * 3 * 5, dtype=np.float64).reshape(4, 3, 5)

    assert np.allclose(backend.matmul(a, b), a @ b)
    assert np.allclose(backend.batched_matmul(batch_a, batch_b), np.matmul(batch_a, batch_b))


def test_grouped_gemm_buckets_same_shape_tasks_and_preserves_order():
    from renormalizer.backend.gemm import GemmTask, group_tasks_by_shape
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    tasks = [
        GemmTask(
            np.arange(6, dtype=np.float64).reshape(2, 3),
            np.arange(12, dtype=np.float64).reshape(3, 4),
            tag="same-0",
        ),
        GemmTask(
            np.arange(6, 12, dtype=np.float64).reshape(2, 3),
            np.arange(12, 24, dtype=np.float64).reshape(3, 4),
            tag="same-1",
        ),
        GemmTask(
            np.arange(12, dtype=np.float64).reshape(3, 4),
            np.arange(8, dtype=np.float64).reshape(4, 2),
            tag="ragged",
        ),
    ]

    buckets = group_tasks_by_shape(tasks, xp=np)
    results = backend.grouped_gemm(tasks, pack_threshold=2)

    assert sorted(len(bucket_tasks) for bucket_tasks in buckets.values()) == [1, 2]
    assert len(results) == len(tasks)
    assert np.allclose(results[0], tasks[0].A @ tasks[0].B)
    assert np.allclose(results[1], tasks[1].A @ tasks[1].B)
    assert np.allclose(results[2], tasks[2].A @ tasks[2].B)


def test_grouped_gemm_applies_flags_alpha_beta_and_updates_c():
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    a = np.arange(6, dtype=np.float64).reshape(3, 2)
    b = np.arange(12, dtype=np.float64).reshape(3, 4)
    c = np.ones((2, 4), dtype=np.float64)
    task = GemmTask(a, b, C=c, trans_a=True, alpha=2.0, beta=3.0, tag="accumulate")

    results = backend.grouped_gemm([task], pack_threshold=2)

    expected = 2.0 * (a.T @ b) + 3.0 * np.ones((2, 4), dtype=np.float64)
    assert results == [c]
    assert np.allclose(c, expected)


def test_grouped_gemm_forbid_policy_rejects_bucketed_fallback():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.execution import BackendFeatureError
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend(config=BackendConfig(fallback_policy="forbid"))
    tasks = [
        GemmTask(
            np.arange(6, dtype=np.float64).reshape(2, 3),
            np.arange(12, dtype=np.float64).reshape(3, 4),
        )
    ]

    with pytest.raises(BackendFeatureError, match="native grouped_gemm unavailable"):
        backend.grouped_gemm(tasks)


def test_grouped_gemm_warn_policy_emits_warning_and_returns_result():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend(config=BackendConfig(fallback_policy="warn"))
    task = GemmTask(
        np.arange(6, dtype=np.float64).reshape(2, 3),
        np.arange(12, dtype=np.float64).reshape(3, 4),
    )

    with pytest.warns(RuntimeWarning, match="native grouped_gemm unavailable"):
        results = backend.grouped_gemm([task])

    assert len(results) == 1
    assert np.allclose(results[0], task.A @ task.B)


def test_should_batch_uses_copy_to_flop_heuristic():
    from renormalizer.backend.gemm import GemmTask, should_batch

    tiny_tasks = [
        GemmTask(np.ones((1, 1), dtype=np.float64), np.ones((1, 1), dtype=np.float64))
        for _ in range(8)
    ]
    large_tasks = [
        GemmTask(np.ones((128, 128), dtype=np.float64), np.ones((128, 128), dtype=np.float64))
        for _ in range(8)
    ]

    assert should_batch(tiny_tasks, xp=np, pack_threshold=4) is False
    assert should_batch(large_tasks, xp=np, pack_threshold=4) is True


def test_grouped_gemm_records_bucketed_fallback_profile(tmp_path):
    from renormalizer.backend.gemm import GemmTask
    from renormalizer.backend.numpy_backend import NumpyBackend
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = NumpyBackend()
    tasks = [
        GemmTask(np.ones((128, 128), dtype=np.float64), np.ones((128, 128), dtype=np.float64), tag="batch-0"),
        GemmTask(np.full((128, 128), 2.0, dtype=np.float64), np.ones((128, 128), dtype=np.float64), tag="batch-1"),
        GemmTask(np.ones((3, 4), dtype=np.float64), np.ones((4, 5), dtype=np.float64), tag="loop"),
    ]
    event_path = tmp_path / "events.jsonl"
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)

        results = backend.grouped_gemm(tasks, pack_threshold=2)
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    assert np.allclose(results[0], tasks[0].A @ tasks[0].B)
    assert np.allclose(results[1], tasks[1].A @ tasks[1].B)
    assert np.allclose(results[2], tasks[2].A @ tasks[2].B)

    payloads = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    events = [payload for payload in payloads if payload["event"] == "contraction_execute"]

    assert len(events) == 1
    event = events[0]
    assert event["backend"] == "numpy"
    assert event["lowering"] == "grouped_gemm"
    assert event["num_grouped_tasks"] == 3
    assert event["num_shape_buckets"] == 2
    assert event["num_batched_gemm"] == 1
    assert event["num_gemm"] == 1
    assert event["num_blocks"] == 3
    assert event["flops"] == 8388728
    assert event["read_bytes"] == 524544
    assert event["write_bytes"] == 262264
    assert event["copy_bytes"] == 524288
    assert event["fallback_reason"] == "native grouped_gemm unavailable; used bucketed fallback"
    assert event["bucket_task_counts"] == [1, 2]
    assert event["batched_bucket_count"] == 1
    assert event["loop_bucket_count"] == 1
    assert event["wall_s"] >= 0.0


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


def test_execute_batched_matmul_plan_preserves_generic_output_mode_order():
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

    assert plan.kind == "batched_gemm"
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

    assert plan["lowering"] == "batched_gemm"
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
