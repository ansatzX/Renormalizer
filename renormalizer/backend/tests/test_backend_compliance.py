# -*- coding: utf-8 -*-

import json

import numpy as np
import pytest


COMPLIANCE_CASES = (
    ("numpy", "cpu"),
    ("torch", "cpu"),
    ("jax", "cpu"),
    ("cupy", "gpu"),
    ("torch", "gpu"),
    ("jax", "gpu"),
)


def _backend_or_skip(name, device="cpu", *, fallback_policy=None):
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.factory import create_backend, is_backend_available

    if not is_backend_available(name):
        pytest.skip("{0} backend unavailable".format(name))
    try:
        return create_backend(
            name,
            config=BackendConfig(device=device, precision=64, fallback_policy=fallback_policy),
        )
    except Exception as exc:
        pytest.skip("{0}/{1} backend unavailable: {2}".format(name, device, exc))


def _assert_allclose(backend, actual, expected, **kwargs):
    np.testing.assert_allclose(backend.to_numpy(actual), expected, **kwargs)


@pytest.mark.parametrize("backend_name,device", COMPLIANCE_CASES)
def test_backend_compliance_roundtrip_layout_and_functional_update(backend_name, device):
    backend = _backend_or_skip(backend_name, device)
    x_np = np.arange(6, dtype=np.float64).reshape(2, 3)

    x = backend.to_backend(x_np)
    info = backend.array_info(x)
    y = backend.at_set(x, (0, 1), 99.0)

    _assert_allclose(backend, x, x_np)
    updated = x_np.copy()
    updated[0, 1] = 99.0
    _assert_allclose(backend, y, updated)
    _assert_allclose(backend, x, x_np)

    assert info.shape == x_np.shape
    assert info.ndim == x_np.ndim
    assert info.size == x_np.size
    assert info.itemsize == x_np.itemsize
    assert info.nbytes == x_np.nbytes
    assert info.backend_name == backend.name
    assert backend.capabilities.matmul is True
    assert backend.capabilities.batched_matmul is True
    assert backend.capabilities.functional_update is True


@pytest.mark.parametrize("backend_name,device", COMPLIANCE_CASES)
def test_backend_compliance_matmul_batched_and_pair_execute(backend_name, device):
    backend = _backend_or_skip(backend_name, device)
    left_np = np.arange(6, dtype=np.float64).reshape(2, 3)
    right_np = np.arange(12, dtype=np.float64).reshape(3, 4)
    batch_left_np = np.arange(2 * 2 * 3, dtype=np.float64).reshape(2, 2, 3)
    batch_right_np = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)

    left = backend.to_backend(left_np)
    right = backend.to_backend(right_np)
    batch_left = backend.to_backend(batch_left_np)
    batch_right = backend.to_backend(batch_right_np)
    plan = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))

    _assert_allclose(backend, backend.matmul(left, right), left_np @ right_np)
    _assert_allclose(backend, backend.batched_matmul(batch_left, batch_right), np.matmul(batch_left_np, batch_right_np))
    _assert_allclose(backend, backend.execute(plan), left_np @ right_np)
    assert plan.steps[0].kind == "gemm"
    assert plan.plan_hash


@pytest.mark.parametrize("backend_name,device", COMPLIANCE_CASES)
def test_backend_compliance_tensor_ops_and_linalg(backend_name, device):
    backend = _backend_or_skip(backend_name, device)
    left_np = np.arange(6, dtype=np.float64).reshape(2, 3)
    right_np = np.arange(12, dtype=np.float64).reshape(3, 4)
    sym_np = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=np.float64)
    rect_np = np.arange(6, dtype=np.float64).reshape(3, 2)

    left = backend.to_backend(left_np)
    right = backend.to_backend(right_np)
    sym = backend.to_backend(sym_np)
    rect = backend.to_backend(rect_np)

    _assert_allclose(backend, backend.tensordot(left, right, axes=((1,), (0,))), np.tensordot(left_np, right_np, axes=((1,), (0,))))
    _assert_allclose(backend, backend.einsum("ik,kj->ij", left, right), left_np @ right_np)
    q, r = backend.linalg.qr(rect)
    _assert_allclose(backend, q @ r, rect_np, rtol=1e-10, atol=1e-10)
    evals, evecs = backend.linalg.eigh(sym)
    _assert_allclose(backend, evecs @ backend.diag(evals) @ evecs.T, sym_np, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("backend_name,device", COMPLIANCE_CASES)
def test_backend_compliance_explicit_contract_uses_planner_and_executor(monkeypatch, backend_name, device):
    backend = _backend_or_skip(backend_name, device)
    left_np = np.arange(6, dtype=np.float64).reshape(2, 3)
    right_np = np.arange(12, dtype=np.float64).reshape(3, 4)
    left = backend.to_backend(left_np)
    right = backend.to_backend(right_np)
    plan_calls = []
    execute_calls = []
    original_plan_contraction = backend.plan_contraction
    original_execute = backend.execute

    def counting_plan_contraction(spec, **kwargs):
        plan_calls.append(spec)
        return original_plan_contraction(spec, **kwargs)

    def counting_execute(plan, **kwargs):
        execute_calls.append(plan)
        return original_execute(plan, **kwargs)

    monkeypatch.setattr(backend, "plan_contraction", counting_plan_contraction)
    monkeypatch.setattr(backend, "execute", counting_execute)

    result = backend.contract("ik,kj->ij", left, right)

    assert len(plan_calls) == 1
    assert len(execute_calls) == 1
    assert execute_calls[0].steps[0].kind == "gemm"
    _assert_allclose(backend, result, left_np @ right_np)


@pytest.mark.parametrize("backend_name,device", COMPLIANCE_CASES)
def test_backend_compliance_tensordot_uses_planner_and_executor(monkeypatch, backend_name, device):
    backend = _backend_or_skip(backend_name, device)
    left_np = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    right_np = np.arange(4 * 3 * 5, dtype=np.float64).reshape(4, 3, 5)
    left = backend.to_backend(left_np)
    right = backend.to_backend(right_np)
    plan_calls = []
    execute_calls = []
    original_plan_contraction = backend.plan_contraction
    original_execute = backend.execute

    def counting_plan_contraction(spec, **kwargs):
        plan_calls.append(spec)
        return original_plan_contraction(spec, **kwargs)

    def counting_execute(plan, **kwargs):
        execute_calls.append(plan)
        return original_execute(plan, **kwargs)

    monkeypatch.setattr(backend, "plan_contraction", counting_plan_contraction)
    monkeypatch.setattr(backend, "execute", counting_execute)

    result = backend.tensordot(left, right, axes=([2, 1], [0, 1]))

    assert len(plan_calls) == 1
    assert len(execute_calls) == 1
    assert tuple(plan_calls[0].operands[0].modes) == (0, 1, 2)
    assert tuple(plan_calls[0].operands[1].modes) == (2, 1, 3)
    assert tuple(plan_calls[0].output_modes) == (0, 3)
    assert execute_calls[0].steps[0].kind == "gemm"
    _assert_allclose(backend, result, np.tensordot(left_np, right_np, axes=([2, 1], [0, 1])))


@pytest.mark.parametrize("backend_name,device", COMPLIANCE_CASES)
def test_backend_compliance_contract_uses_backend_execution(backend_name, device):
    backend = _backend_or_skip(backend_name, device)
    left_np = np.arange(6, dtype=np.float64).reshape(2, 3)
    right_np = np.arange(12, dtype=np.float64).reshape(3, 4)
    left = backend.to_backend(left_np)
    right = backend.to_backend(right_np)

    result = backend.contract("ik,kj->ij", left, right, optimize="greedy")

    assert backend.is_array(result)
    _assert_allclose(backend, result, left_np @ right_np)


@pytest.mark.parametrize("backend_name,device", COMPLIANCE_CASES)
def test_backend_compliance_contract_expression_uses_backend_execution(backend_name, device):
    backend = _backend_or_skip(backend_name, device)
    left_np = np.arange(6, dtype=np.float64).reshape(2, 3)
    right_np = np.arange(12, dtype=np.float64).reshape(3, 4)
    left = backend.to_backend(left_np)
    right = backend.to_backend(right_np)

    expr = backend.contract_expression("ik,kj->ij", left_np.shape, right_np.shape, optimize="greedy")
    result = expr(left, right)

    assert backend.capabilities.contract_expression is True
    assert backend.is_array(result)
    _assert_allclose(backend, result, left_np @ right_np)


@pytest.mark.parametrize("backend_name,device", COMPLIANCE_CASES)
def test_backend_compliance_contract_expression_uses_planner_and_executor(monkeypatch, backend_name, device):
    backend = _backend_or_skip(backend_name, device)
    left_np = np.arange(6, dtype=np.float64).reshape(2, 3)
    right_np = np.arange(12, dtype=np.float64).reshape(3, 4)
    left = backend.to_backend(left_np)
    right = backend.to_backend(right_np)
    plan_calls = []
    execute_calls = []
    original_plan_contraction = backend.plan_contraction
    original_execute = backend.execute

    def counting_plan_contraction(spec, **kwargs):
        plan_calls.append(spec)
        return original_plan_contraction(spec, **kwargs)

    def counting_execute(plan, **kwargs):
        execute_calls.append(plan)
        return original_execute(plan, **kwargs)

    monkeypatch.setattr(backend, "plan_contraction", counting_plan_contraction)
    monkeypatch.setattr(backend, "execute", counting_execute)

    expr = backend.contract_expression("ik,kj->ij", left_np.shape, right_np.shape, optimize="greedy")
    result = expr(left, right)

    assert len(plan_calls) == 1
    assert len(execute_calls) == 1
    assert execute_calls[0].steps[0].kind == "gemm"
    _assert_allclose(backend, result, left_np @ right_np)


@pytest.mark.parametrize("backend_name,device", COMPLIANCE_CASES)
def test_backend_compliance_constant_contract_expression_uses_planner_and_executor(monkeypatch, backend_name, device):
    backend = _backend_or_skip(backend_name, device)
    left_np = np.arange(6, dtype=np.float64).reshape(2, 3)
    right_np = np.arange(12, dtype=np.float64).reshape(3, 4)
    left = backend.to_backend(left_np)
    right = backend.to_backend(right_np)
    plan_calls = []
    execute_calls = []
    original_plan_contraction = backend.plan_contraction
    original_execute = backend.execute

    def counting_plan_contraction(spec, **kwargs):
        plan_calls.append(spec)
        return original_plan_contraction(spec, **kwargs)

    def counting_execute(plan, **kwargs):
        execute_calls.append(plan)
        return original_execute(plan, **kwargs)

    monkeypatch.setattr(backend, "plan_contraction", counting_plan_contraction)
    monkeypatch.setattr(backend, "execute", counting_execute)

    expr = backend.contract_expression("ik,kj->ij", left, right_np.shape, constants=[0], optimize="greedy")
    result = expr(right)

    assert len(plan_calls) == 1
    assert len(execute_calls) == 1
    assert execute_calls[0].steps[0].kind == "gemm"
    _assert_allclose(backend, result, left_np @ right_np)


@pytest.mark.parametrize("backend_name,device", COMPLIANCE_CASES)
def test_backend_compliance_contract_path_uses_backend_surface(backend_name, device):
    backend = _backend_or_skip(backend_name, device)

    path, info = backend.contract_path("ik,kj->ij", (2, 3), (3, 4), shapes=True, optimize="greedy")

    assert backend.capabilities.contraction_path is True
    assert path == [(0, 1)]
    assert int(info.largest_intermediate) == 8
    assert int(info.opt_cost) == 48


@pytest.mark.parametrize("backend_name,device", COMPLIANCE_CASES)
def test_backend_compliance_astype_copy_policy(backend_name, device):
    from renormalizer.backend import BackendCopyError, CopyPolicy

    backend = _backend_or_skip(backend_name, device)
    x_np = np.array([1.0, 2.0], dtype=np.float64)
    x = backend.to_backend(x_np)

    same = backend.astype(x, backend.real_dtype, copy=CopyPolicy.NEVER)
    copied = backend.astype(x, backend.real_dtype, copy=CopyPolicy.ALWAYS)

    assert same is x
    assert copied is not x
    _assert_allclose(backend, copied, x_np)

    with pytest.raises(BackendCopyError, match="astype would require"):
        backend.astype(x, backend.complex_dtype, copy=CopyPolicy.NEVER)

    converted = backend.astype(x, backend.complex_dtype)
    assert np.issubdtype(backend.to_numpy(converted).dtype, np.complexfloating)
    _assert_allclose(backend, converted, x_np.astype(np.complex128))


@pytest.mark.parametrize("backend_name", ("cupy", "torch"))
def test_gpu_backend_stream_events_and_synchronize_when_available(backend_name):
    from renormalizer.backend import BackendConfig, StreamEvent
    from renormalizer.backend.factory import create_backend, is_backend_available

    if not is_backend_available(backend_name):
        pytest.skip("{0} unavailable".format(backend_name))
    try:
        backend = create_backend(backend_name, config=BackendConfig(device="gpu", precision=64))
    except Exception as exc:
        pytest.skip("{0}/gpu unavailable: {1}".format(backend_name, exc))

    assert backend.capabilities.streams is True
    assert backend.capabilities.events is True

    stream = backend.new_stream()
    default_stream = backend.default_stream()
    event = backend.record_event(stream=stream)

    assert stream is not None
    assert default_stream is not None
    assert isinstance(event, StreamEvent)
    assert event.device == backend.current_device()
    assert event.stream is stream
    assert event.token is not None

    backend.wait_event(event, stream=default_stream)

    left_np = np.arange(6, dtype=np.float64).reshape(2, 3)
    right_np = np.arange(12, dtype=np.float64).reshape(3, 4)
    left = backend.to_backend(left_np)
    right = backend.to_backend(right_np)
    plan = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))
    result = backend.execute(plan, stream=stream)
    backend.synchronize(stream=stream)

    _assert_allclose(backend, result, left_np @ right_np)


@pytest.mark.parametrize("backend_name,device", COMPLIANCE_CASES)
def test_backend_compliance_profiling_execute_event(tmp_path, backend_name, device):
    from renormalizer.utils import profiling
    from renormalizer.utils.log import DEBUG, PROFILING, init_log, package_logger

    backend = _backend_or_skip(backend_name, device)
    left_np = np.arange(6, dtype=np.float64).reshape(2, 3)
    right_np = np.arange(12, dtype=np.float64).reshape(3, 4)
    left = backend.to_backend(left_np)
    right = backend.to_backend(right_np)
    plan = backend.plan_contraction(backend.parse_einsum("ik,kj->ij", left, right))
    event_path = tmp_path / "{0}-{1}.jsonl".format(backend_name, device)
    old_level = package_logger.level
    try:
        init_log(PROFILING)
        profiling.register_event_output(event_path)
        result = backend.execute(plan)
    finally:
        profiling.close_event_output()
        profiling.flush_summaries()
        init_log(old_level or DEBUG)

    _assert_allclose(backend, result, left_np @ right_np)
    events = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if line.strip()
    ]
    execute = next(event for event in events if event["event"] == "contraction_execute")

    assert execute["backend"] == backend.name
    assert execute["lowering"] == "gemm"
    assert execute["plan_hash"] == plan.plan_hash
    assert execute["flops"] == 48
    assert execute["fallback_reason"] is None


@pytest.mark.parametrize("backend_name,device", COMPLIANCE_CASES)
def test_backend_compliance_grouped_gemm_fallback_policy(backend_name, device):
    from renormalizer.backend import BackendConfig, BackendFeatureError, GemmTask
    from renormalizer.backend.factory import create_backend, is_backend_available

    if not is_backend_available(backend_name):
        pytest.skip("{0} backend unavailable".format(backend_name))
    try:
        backend = create_backend(
            backend_name,
            config=BackendConfig(device=device, precision=64, fallback_policy="forbid"),
        )
    except Exception as exc:
        pytest.skip("{0}/{1} backend unavailable: {2}".format(backend_name, device, exc))

    left_np = np.arange(4, dtype=np.float64).reshape(2, 2)
    right_np = np.arange(4, dtype=np.float64).reshape(2, 2)
    task = GemmTask(backend.to_backend(left_np), backend.to_backend(right_np))

    if backend.capabilities.grouped_gemm:
        result = backend.grouped_gemm([task])[0]
        _assert_allclose(backend, result, left_np @ right_np)
    else:
        with pytest.raises(BackendFeatureError, match="native grouped_gemm unavailable"):
            backend.grouped_gemm([task])


@pytest.mark.parametrize("backend_name,device", COMPLIANCE_CASES)
def test_backend_grouped_gemm_capability_requires_native_override(backend_name, device):
    from renormalizer.backend.abstract import AbstractBackend

    backend = _backend_or_skip(backend_name, device)

    if type(backend).grouped_gemm is AbstractBackend.grouped_gemm:
        assert backend.capabilities.grouped_gemm is False
        assert backend.supports_grouped_gemm is False


@pytest.mark.parametrize("backend_name,device", COMPLIANCE_CASES)
def test_backend_memory_pool_capability_requires_native_hooks(backend_name, device):
    from renormalizer.backend.abstract import AbstractBackend

    backend = _backend_or_skip(backend_name, device)

    if (
        type(backend).free_all_blocks is AbstractBackend.free_all_blocks
        and type(backend).log_memory_usage is AbstractBackend.log_memory_usage
    ):
        assert backend.capabilities.memory_pool is False
        assert backend.supports_memory_pool is False


def test_torch_backend_honors_indexed_cuda_device_when_available():
    try:
        import torch
    except (ImportError, OSError) as exc:
        pytest.skip("torch unavailable: {0}".format(exc))
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("torch needs at least two CUDA devices for indexed-device compliance")

    from renormalizer.backend import BackendConfig
    from renormalizer.backend.execution import DeviceSpec
    from renormalizer.backend.factory import create_backend

    backend = create_backend("torch", config=BackendConfig(device="cuda:1", precision=64))
    x = backend.to_backend(np.ones((2,), dtype=np.float64))

    assert backend.current_device() == DeviceSpec(kind="cuda", index=1, visible_id="1")
    assert x.device.type == "cuda"
    assert x.device.index == 1


def test_cupy_backend_honors_indexed_cuda_device_when_available():
    try:
        import cupy
    except (ImportError, OSError) as exc:
        pytest.skip("cupy unavailable: {0}".format(exc))
    if cupy.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("cupy needs at least two CUDA devices for indexed-device compliance")

    from renormalizer.backend import BackendConfig
    from renormalizer.backend.execution import DeviceSpec
    from renormalizer.backend.factory import create_backend

    backend = create_backend("cupy", config=BackendConfig(device="cuda:1", precision=64))
    x = backend.to_backend(np.ones((2,), dtype=np.float64))

    assert backend.current_device() == DeviceSpec(kind="cuda", index=1, visible_id="1")
    assert x.device.id == 1


def test_jax_backend_honors_indexed_cuda_device_when_available():
    try:
        import jax
    except (ImportError, OSError) as exc:
        pytest.skip("jax unavailable: {0}".format(exc))
    try:
        devices = jax.devices("gpu")
    except RuntimeError as exc:
        pytest.skip("jax gpu unavailable: {0}".format(exc))
    if len(devices) < 2:
        pytest.skip("jax needs at least two CUDA devices for indexed-device compliance")

    from renormalizer.backend import BackendConfig
    from renormalizer.backend.execution import DeviceSpec
    from renormalizer.backend.factory import create_backend

    backend = create_backend("jax", config=BackendConfig(device="cuda:1", precision=64))
    x = backend.to_backend(np.ones((2,), dtype=np.float64))
    device = next(iter(x.devices()))

    assert backend.current_device() == DeviceSpec(kind="cuda", index=1, visible_id="1")
    assert device.platform == "gpu"
    assert device.id == 1


def test_torch_array_info_reports_actual_indexed_cuda_device_when_available():
    try:
        import torch
    except (ImportError, OSError) as exc:
        pytest.skip("torch unavailable: {0}".format(exc))
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("torch needs at least two CUDA devices for array_info device compliance")

    from renormalizer.backend import BackendConfig
    from renormalizer.backend.execution import DeviceSpec
    from renormalizer.backend.factory import create_backend

    backend = create_backend("torch", config=BackendConfig(device="cuda:1", precision=64))
    x = backend.to_backend(np.ones((2,), dtype=np.float64))
    backend.set_device("cuda:0")
    info = backend.array_info(x)

    assert backend.current_device() == DeviceSpec(kind="cuda", index=0, visible_id="0")
    assert info.device == DeviceSpec(kind="cuda", index=1, visible_id="1")
    assert info.is_device is True


def test_cupy_array_info_reports_actual_indexed_cuda_device_when_available():
    try:
        import cupy
    except (ImportError, OSError) as exc:
        pytest.skip("cupy unavailable: {0}".format(exc))
    if cupy.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("cupy needs at least two CUDA devices for array_info device compliance")

    from renormalizer.backend import BackendConfig
    from renormalizer.backend.execution import DeviceSpec
    from renormalizer.backend.factory import create_backend

    backend = create_backend("cupy", config=BackendConfig(device="cuda:1", precision=64))
    x = backend.to_backend(np.ones((2,), dtype=np.float64))
    backend.set_device("cuda:0")
    info = backend.array_info(x)

    assert backend.current_device() == DeviceSpec(kind="cuda", index=0, visible_id="0")
    assert info.device == DeviceSpec(kind="cuda", index=1, visible_id="1")
    assert info.is_device is True


def test_jax_array_info_reports_actual_indexed_cuda_device_when_available():
    try:
        import jax
    except (ImportError, OSError) as exc:
        pytest.skip("jax unavailable: {0}".format(exc))
    try:
        devices = jax.devices("gpu")
    except RuntimeError as exc:
        pytest.skip("jax gpu unavailable: {0}".format(exc))
    if len(devices) < 2:
        pytest.skip("jax needs at least two CUDA devices for array_info device compliance")

    from renormalizer.backend import BackendConfig
    from renormalizer.backend.execution import DeviceSpec
    from renormalizer.backend.factory import create_backend

    backend = create_backend("jax", config=BackendConfig(device="cuda:1", precision=64))
    x = backend.to_backend(np.ones((2,), dtype=np.float64))
    backend.set_device("cuda:0")
    info = backend.array_info(x)

    assert backend.current_device() == DeviceSpec(kind="cuda", index=0, visible_id="0")
    assert info.device == DeviceSpec(kind="cuda", index=1, visible_id="1")
    assert info.is_device is True


@pytest.mark.parametrize("backend_name", ("torch", "cupy", "jax"))
def test_gpu_to_backend_copy_policy_rejects_host_and_dtype_copy(backend_name):
    from renormalizer.backend import BackendConfig, BackendCopyError, CopyPolicy
    from renormalizer.backend.factory import create_backend, is_backend_available

    if not is_backend_available(backend_name):
        pytest.skip("{0} unavailable".format(backend_name))
    try:
        backend = create_backend(backend_name, config=BackendConfig(device="gpu", precision=64))
    except Exception as exc:
        pytest.skip("{0}/gpu unavailable: {1}".format(backend_name, exc))

    host = np.ones((2,), dtype=np.float64)
    with pytest.raises(BackendCopyError, match="to_backend would require"):
        backend.to_backend(host, copy=CopyPolicy.NEVER)

    device = backend.to_backend(host)
    same = backend.to_backend(device, copy=CopyPolicy.NEVER)
    assert same is device

    with pytest.raises(BackendCopyError, match="to_backend would require"):
        backend.to_backend(device, dtype=np.float32, copy=CopyPolicy.NEVER)


@pytest.mark.parametrize("backend_name", ("torch", "cupy", "jax"))
def test_gpu_to_host_copy_policy_rejects_device_to_host_copy(backend_name):
    from renormalizer.backend import BackendConfig, BackendCopyError, CopyPolicy
    from renormalizer.backend.factory import create_backend, is_backend_available

    if not is_backend_available(backend_name):
        pytest.skip("{0} unavailable".format(backend_name))
    try:
        backend = create_backend(backend_name, config=BackendConfig(device="gpu", precision=64))
    except Exception as exc:
        pytest.skip("{0}/gpu unavailable: {1}".format(backend_name, exc))

    device = backend.to_backend(np.ones((2,), dtype=np.float64))
    with pytest.raises(BackendCopyError, match="to_host would require"):
        backend.to_host(device, copy=CopyPolicy.NEVER)

    host = backend.to_host(device)
    same = backend.to_host(host, copy=CopyPolicy.NEVER)

    assert isinstance(host, np.ndarray)
    assert same is host
