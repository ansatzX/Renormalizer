# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from renormalizer import get_backend, set_backend
from renormalizer.backend.boundary import scalar_to_python
from renormalizer.backend.factory import SUPPORTED_BACKENDS, is_backend_available
from renormalizer.cons import xp
from renormalizer.mps.hop_expr import hop_expr
from renormalizer.mps.matrix import Matrix, asnumpy, asxp


@pytest.fixture(autouse=True)
def restore_numpy_backend():
    try:
        yield
    finally:
        set_backend("numpy", precision=64)


def _select_or_skip(name, **options):
    if not is_backend_available(name):
        pytest.skip("{} is not installed".format(name))
    if name == "cupy":
        try:
            cupy = importlib.import_module("cupy")
        except ImportError:
            pytest.skip("CuPy is not installed")
        try:
            device_count = cupy.cuda.runtime.getDeviceCount()
            if device_count < 1:
                pytest.skip("CuPy CUDA device 0 is unavailable: no visible CUDA devices")
            with cupy.cuda.Device(0):
                cupy.cuda.runtime.getDevice()
        except cupy.cuda.runtime.CUDARuntimeError as error:
            pytest.skip("CuPy CUDA device 0 is unavailable: {}".format(error))
        options.setdefault("device", "cuda:0")
    elif name in {"jax", "torch"}:
        options.setdefault("device", "cpu")
    return set_backend(name, **options)


def test_select_or_skip_propagates_backend_runtime_errors(monkeypatch):
    def failing_set_backend(*args, **kwargs):
        raise RuntimeError("adapter construction regression")

    def fail_if_skipped(reason):
        raise AssertionError("backend RuntimeError was converted to skip: {}".format(reason))

    monkeypatch.setattr(sys.modules[__name__], "set_backend", failing_set_backend)
    monkeypatch.setattr(pytest, "skip", fail_if_skipped)

    with pytest.raises(RuntimeError, match="adapter construction regression"):
        _select_or_skip("numpy")


def test_factory_exposes_exact_stage_one_backends():
    assert SUPPORTED_BACKENDS == ("numpy", "cupy", "jax", "torch")


def test_default_import_does_not_load_optional_backend_packages():
    script = """
import sys
import renormalizer
from renormalizer.backend.factory import available_backends
available_backends()
assert all(name not in sys.modules for name in ('cupy', 'jax', 'torch'))
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("name", SUPPORTED_BACKENDS)
def test_backend_roundtrip_and_tensordot(name):
    selected = _select_or_skip(name, precision=64)
    source = np.arange(24.0).reshape(2, 3, 4)[:, :, ::-1]

    value = selected.to_backend(source)
    result = selected.tensordot(value, value, axes=([2], [2]))

    assert selected.to_backend(None) is None
    assert selected.to_numpy(None) is None
    assert selected.to_host(None) is None
    assert selected.current_device() == ("cuda:0" if name == "cupy" else "cpu")
    assert selected.supports_execution_ir is False
    np.testing.assert_array_equal(selected.to_numpy(value), source)
    np.testing.assert_allclose(
        selected.to_numpy(result),
        np.tensordot(source, source, axes=([2], [2])),
    )
    assert scalar_to_python(selected.asarray(2.5), selected) == 2.5


@pytest.mark.parametrize("name", SUPPORTED_BACKENDS)
def test_backend_runs_tiny_matrix_hop_expr_path(name):
    selected = _select_or_skip(name, precision=64)
    left = np.array([[[2.0]]])
    mpo = np.arange(4.0).reshape(1, 2, 2, 1)
    right = np.array([[[3.0]]])
    center = np.arange(2.0).reshape(1, 2, 1)
    center_matrix = Matrix(center, dtype=selected.real_dtype)

    assert center_matrix.dtype == np.dtype("float64")
    assert Matrix(np.eye(2), dtype=selected.real_dtype).check_lortho()

    expression = hop_expr(left, right, [mpo], center.shape)
    actual = asnumpy(expression(asxp(center_matrix)))
    expected = np.einsum("abc,bdef,lfk,cek->adl", left, mpo, right, center)

    np.testing.assert_allclose(actual, expected)


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX is not installed")
def test_jax_honors_precision_and_deterministic_seed():
    first = _select_or_skip("jax", precision=64, seed=7)
    first_sample = first.to_numpy(first.random.random(4))
    second = _select_or_skip("jax", precision=64, seed=7)

    assert first.real_dtype == first.array_namespace.float64
    assert first.complex_dtype == first.array_namespace.complex128
    assert first.asarray([1.0]).dtype == first.array_namespace.float64
    np.testing.assert_array_equal(second.to_numpy(second.random.random(4)), first_sample)


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX is not installed")
@pytest.mark.parametrize(
    "caller_x64, precision, active_x64",
    [(False, 64, True), (True, 32, False)],
)
def test_jax_x64_lifecycle_restores_caller_after_switch_and_failed_replacement(
    monkeypatch, caller_x64, precision, active_x64
):
    import jax

    proxy_module = importlib.import_module("renormalizer.backend.proxy")
    original_x64 = bool(jax.config.jax_enable_x64)
    try:
        jax.config.update("jax_enable_x64", caller_x64)
        selected = set_backend("jax", device="cpu", precision=precision, seed=7)

        assert bool(jax.config.jax_enable_x64) is active_x64

        observed_during_replacement = []

        def failing_create_backend(*args, **kwargs):
            observed_during_replacement.append(bool(jax.config.jax_enable_x64))
            raise RuntimeError("replacement construction failed")

        with monkeypatch.context() as patch:
            patch.setattr(proxy_module, "create_backend", failing_create_backend)
            with pytest.raises(RuntimeError, match="construction failed"):
                set_backend("numpy")

        assert observed_during_replacement == [caller_x64]
        assert get_backend() is selected
        assert bool(jax.config.jax_enable_x64) is active_x64

        set_backend("numpy", precision=64)
        assert bool(jax.config.jax_enable_x64) is caller_x64
    finally:
        set_backend("numpy", precision=64)
        jax.config.update("jax_enable_x64", original_x64)


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX is not installed")
def test_jax_precision_mutators_update_active_state_and_restore_caller():
    import jax

    original_x64 = bool(jax.config.jax_enable_x64)
    try:
        jax.config.update("jax_enable_x64", False)
        selected = set_backend("jax", device="cpu", precision=64, seed=7)
        assert bool(jax.config.jax_enable_x64) is True

        selected.use_32bits()
        assert bool(jax.config.jax_enable_x64) is False
        selected.use_64bits()
        assert bool(jax.config.jax_enable_x64) is True

        set_backend("numpy", precision=64)
        assert bool(jax.config.jax_enable_x64) is False
    finally:
        set_backend("numpy", precision=64)
        jax.config.update("jax_enable_x64", original_x64)


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX is not installed")
@pytest.mark.parametrize("failure_stage", ["activate", "seed"])
def test_backend_manager_cleans_failed_candidate_and_reactivates_jax(
    monkeypatch, failure_stage
):
    import jax

    proxy_module = importlib.import_module("renormalizer.backend.proxy")
    original_x64 = bool(jax.config.jax_enable_x64)
    events = []
    observed_x64 = []

    class CandidateConfig:
        seed = 7

    class FailingCandidate:
        config = CandidateConfig()

        def __init__(self):
            self.active = False

        @property
        def random(self):
            return self

        def activate(self):
            events.append("candidate.activate")
            observed_x64.append(bool(jax.config.jax_enable_x64))
            self.active = True
            if failure_stage == "activate":
                raise RuntimeError("candidate activate failed")

        def seed(self, seed):
            events.append("candidate.seed")
            observed_x64.append(bool(jax.config.jax_enable_x64))
            if failure_stage == "seed":
                raise RuntimeError("candidate seed failed")

        def deactivate(self):
            events.append("candidate.deactivate")
            self.active = False

    candidate = FailingCandidate()
    try:
        jax.config.update("jax_enable_x64", False)
        previous = set_backend("jax", device="cpu", precision=64, seed=7)

        with monkeypatch.context() as patch:
            patch.setattr(proxy_module, "create_backend", lambda *args, **kwargs: candidate)
            with pytest.raises(RuntimeError, match="candidate {} failed".format(failure_stage)):
                set_backend("torch")

        expected_events = ["candidate.activate"]
        if failure_stage == "seed":
            expected_events.append("candidate.seed")
        expected_events.append("candidate.deactivate")
        assert events == expected_events
        assert observed_x64 and not any(observed_x64)
        assert candidate.active is False
        assert get_backend() is previous
        assert previous._active is True
        assert bool(jax.config.jax_enable_x64) is True
    finally:
        set_backend("numpy", precision=64)
        jax.config.update("jax_enable_x64", original_x64)


@pytest.mark.parametrize(
    "name, configured_device",
    [
        pytest.param("jax", "cpu", id="jax-cpu"),
        pytest.param("jax", "cuda:0", id="jax-gpu"),
        pytest.param("torch", "cpu", id="torch-cpu"),
        pytest.param("torch", "cuda:0", id="torch-gpu"),
    ],
)
def test_jax_and_torch_reject_per_call_device_overrides(name, configured_device):
    if configured_device.startswith("cuda:"):
        if name == "jax":
            if importlib.util.find_spec("jax") is None:
                pytest.skip("JAX is not installed")
            jax = importlib.import_module("jax")
            try:
                gpu_devices = jax.devices("gpu")
            except RuntimeError:
                gpu_devices = []
            if not gpu_devices:
                pytest.skip("JAX GPU is unavailable")
        else:
            if importlib.util.find_spec("torch") is None:
                pytest.skip("Torch is not installed")
            torch = importlib.import_module("torch")
            if not torch.cuda.is_available():
                pytest.skip("Torch CUDA is unavailable")

    selected = _select_or_skip(
        name, device=configured_device, precision=64, seed=7
    )
    if name == "jax":
        jax = importlib.import_module("jax")
        configured = selected._jax_device
        if configured.platform == "cpu":
            try:
                other = jax.devices("gpu")[0]
            except (IndexError, RuntimeError):
                other = configured
        else:
            other = jax.devices("cpu")[0]
    else:
        torch = importlib.import_module("torch")
        configured = torch.device(configured_device)
        if configured.type == "cpu" and not torch.cuda.is_available():
            other = configured
        else:
            other = torch.device("cuda:0" if configured.type == "cpu" else "cpu")

    calls = [
        lambda: selected.array([1.0], device=configured),
        lambda: selected.asarray([1.0], device=configured),
        lambda: selected.ones(1, device=other),
        lambda: selected.array_namespace.asarray([1.0], device=other),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="device is fixed by BackendConfig"):
            call()


@pytest.mark.parametrize("name", SUPPORTED_BACKENDS)
def test_to_backend_accepts_explicit_dtype_on_configured_device(name):
    selected = _select_or_skip(name, precision=64)

    value = selected.to_backend([1.0, 2.0], dtype=np.float32)

    assert selected.to_backend(None, dtype=np.float32) is None
    assert selected.to_numpy(value).dtype == np.dtype("float32")
    if name == "cupy":
        assert value.device.id == 0
    elif name == "jax":
        assert value.device.platform == "cpu"
    elif name == "torch":
        assert value.device.type == "cpu"
    else:
        assert isinstance(value, np.ndarray)


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX is not installed")
def test_jax_to_backend_moves_foreign_gpu_array_to_configured_cpu():
    import jax
    import jax.numpy as jnp

    try:
        gpu = jax.devices("gpu")[0]
    except (IndexError, RuntimeError):
        pytest.skip("JAX GPU is unavailable")
    with jax.default_device(gpu):
        foreign = jnp.asarray([1.0, 2.0], dtype=jnp.float32)

    selected = set_backend("jax", device="cpu", precision=64, seed=7)
    value = selected.to_backend(foreign, dtype=np.float64)
    converted_values = [
        selected.array(foreign),
        selected.asarray(foreign),
        selected.array_namespace.array(foreign),
        selected.array_namespace.asarray(foreign),
    ]

    assert foreign.device.platform == "gpu"
    assert value.device.platform == "cpu"
    assert value.dtype == jnp.float64
    for converted in converted_values:
        assert converted.device.platform == "cpu"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="Torch is not installed")
def test_torch_public_conversions_move_foreign_cpu_tensor_to_configured_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA is unavailable")
    caller_default = torch.get_default_device()
    caller_allocation = torch.ones(1).device
    foreign = torch.asarray([1.0])

    selected = set_backend("torch", device="cuda:0", precision=64, seed=7)
    with pytest.warns(UserWarning, match="copy construct from a tensor"):
        namespace_tensor = selected.array_namespace.tensor(foreign)
    converted_values = [
        selected.array(foreign),
        selected.asarray(foreign),
        selected.array_namespace.asarray(foreign),
        selected.array_namespace.as_tensor(foreign),
        namespace_tensor,
    ]

    assert foreign.device == torch.device("cpu")
    for converted in converted_values:
        assert converted.device == torch.device("cuda:0")
    assert torch.get_default_device() == caller_default
    assert torch.ones(1).device == caller_allocation


@pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX is not installed")
def test_jax_cpu_configuration_binds_public_namespace_and_random_devices(monkeypatch):
    import jax

    if jax.default_backend() != "gpu":
        pytest.skip("requires JAX GPU default backend")

    selected = set_backend("jax", device="cpu", precision=64, seed=7)
    construction_devices = []
    original_array = selected._jnp.array
    original_asarray = selected._jnp.asarray

    def recording_array(*args, **kwargs):
        construction_devices.append(jax.config.jax_default_device)
        return original_array(*args, **kwargs)

    def recording_asarray(*args, **kwargs):
        construction_devices.append(jax.config.jax_default_device)
        return original_asarray(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(selected._jnp, "array", recording_array)
        array_value = selected.array([1.0, 2.0])
    with monkeypatch.context() as patch:
        patch.setattr(selected._jnp, "asarray", recording_asarray)
        asarray_value = selected.asarray([1.0, 2.0])
    public_value = selected.ones(2)
    nested_value = selected.linalg.norm(selected.ones(2))
    random_value = selected.random.random(2)

    assert selected.current_device() == "cpu"
    assert construction_devices == [selected._jax_device, selected._jax_device]
    assert array_value.device.platform == "cpu"
    assert asarray_value.device.platform == "cpu"
    assert selected._rng_key.device.platform == "cpu"
    assert public_value.device.platform == "cpu"
    assert nested_value.device.platform == "cpu"
    assert random_value.device.platform == "cpu"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="Torch is not installed")
def test_torch_normalizes_dtype_and_tensordot_inside_adapter():
    import torch

    original_tensordot = torch.tensordot
    selected = _select_or_skip("torch", precision=64)
    real = selected.asarray(np.arange(4.0).reshape(2, 2))
    complex_value = selected.asarray(
        (np.arange(4.0) + 1j).reshape(2, 2)
    )

    result = selected.tensordot(real, complex_value, axes=(1, 0))

    assert torch.tensordot is original_tensordot
    assert real.dtype == torch.float64
    assert complex_value.dtype == torch.complex128
    assert result.dtype == torch.complex128
    assert result.device.type == "cpu"
    np.testing.assert_allclose(
        selected.to_numpy(result),
        np.tensordot(
            np.arange(4.0).reshape(2, 2),
            (np.arange(4.0) + 1j).reshape(2, 2),
            axes=(1, 0),
        ),
    )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="Torch is not installed")
def test_torch_adapter_rng_is_deterministic_without_mutating_raw_cpu_rng():
    import torch

    original_state = torch.random.get_rng_state().clone()
    try:
        torch.manual_seed(123)
        caller_state = torch.random.get_rng_state().clone()

        def samples(selected):
            return (
                selected.to_numpy(selected.random.random(4)),
                selected.to_numpy(selected.random.rand(3)),
                selected.to_numpy(selected.random.randn(2)),
                selected.to_numpy(selected.random.randint(0, 10, size=5)),
            )

        first = set_backend("torch", device="cpu", precision=64, seed=7)
        first_samples = samples(first)
        assert torch.equal(torch.random.get_rng_state(), caller_state)

        second = set_backend("torch", device="cpu", precision=64, seed=7)
        second_samples = samples(second)
        assert torch.equal(torch.random.get_rng_state(), caller_state)

        for first_value, second_value in zip(first_samples, second_samples):
            np.testing.assert_array_equal(first_value, second_value)
    finally:
        set_backend("numpy", precision=64)
        torch.random.set_rng_state(original_state)

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="Torch is not installed")
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize(
    "dtype_arg, expected_name",
    [
        pytest.param(np.float64, "float64", id="numpy-scalar-type"),
        pytest.param(np.dtype("complex128"), "complex128", id="numpy-dtype"),
        pytest.param(float, "float64", id="python-scalar-type"),
        pytest.param("torch.float32", "float32", id="native-torch-dtype"),
    ],
)
def test_torch_normalizes_explicit_dtype_for_numpy_facing_conversions(
    device, dtype_arg, expected_name
):
    import torch

    if device.startswith("cuda:") and not torch.cuda.is_available():
        pytest.skip("Torch CUDA is unavailable")
    if dtype_arg == "torch.float32":
        dtype_arg = torch.float32

    selected = set_backend("torch", device=device, precision=64, seed=7)
    expected_dtype = getattr(torch, expected_name)
    expected_device = torch.device(device)
    values = [
        selected.array([1.0], dtype=dtype_arg),
        selected.asarray([1.0], dtype=dtype_arg),
        selected.to_backend([1.0], dtype=dtype_arg),
    ]

    for value in values:
        assert value.dtype == expected_dtype
        assert value.device == expected_device


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="Torch is not installed")
@pytest.mark.parametrize("method_name", ["array", "asarray", "to_backend"])
def test_torch_rejects_unsupported_explicit_numpy_dtype(method_name):
    selected = _select_or_skip("torch", precision=64)

    with pytest.raises(TypeError, match="unsupported Torch dtype"):
        getattr(selected, method_name)([1.0], dtype=np.dtype("datetime64[ns]"))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="Torch is not installed")
@pytest.mark.parametrize("method_name", ["array", "asarray"])
@pytest.mark.parametrize(
    "dtype_arg, expected_name",
    [
        pytest.param(np.float64, "float64", id="numpy-scalar-type"),
        pytest.param(np.dtype("complex128"), "complex128", id="numpy-dtype"),
        pytest.param("torch.float32", "float32", id="native-torch-dtype"),
    ],
)
def test_torch_normalizes_numpy_style_positional_dtype(
    method_name, dtype_arg, expected_name
):
    import torch

    if dtype_arg == "torch.float32":
        dtype_arg = torch.float32
    selected = _select_or_skip("torch", precision=64)

    value = getattr(selected, method_name)([1.0], dtype_arg)

    assert value.dtype == getattr(torch, expected_name)
    assert value.device == torch.device("cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="Torch is not installed")
@pytest.mark.parametrize("method_name", ["array", "asarray"])
def test_torch_rejects_unsupported_positional_numpy_dtype(method_name):
    selected = _select_or_skip("torch", precision=64)

    with pytest.raises(TypeError, match="unsupported Torch dtype"):
        getattr(selected, method_name)([1.0], np.dtype("datetime64[ns]"))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="Torch is not installed")
@pytest.mark.parametrize("method_name", ["array", "asarray"])
def test_torch_rejects_duplicate_positional_and_keyword_dtype(method_name):
    selected = _select_or_skip("torch", precision=64)

    with pytest.raises(TypeError, match="multiple values for dtype"):
        getattr(selected, method_name)([1.0], np.float64, dtype=np.float32)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="Torch is not installed")
@pytest.mark.parametrize("method_name", ["array", "asarray"])
def test_torch_rejects_extra_positional_conversion_arguments(method_name):
    selected = _select_or_skip("torch", precision=64)

    with pytest.raises(TypeError, match="only dtype may be supplied positionally"):
        getattr(selected, method_name)([1.0], np.float64, "unsupported")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="Torch is not installed")
def test_torch_positional_dtype_uses_configured_cuda_device():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA is unavailable")
    selected = set_backend("torch", device="cuda:0", precision=64, seed=7)

    value = selected.asarray([1.0], np.float64)

    assert value.dtype == torch.float64
    assert value.device == torch.device("cuda:0")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="Torch is not installed")
def test_torch_cuda_configuration_binds_namespace_without_global_default():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA is unavailable")

    original_tensordot = torch.tensordot
    caller_default = torch.get_default_device()
    caller_allocation = torch.ones(1).device

    selected = set_backend("torch", device="cuda:0", precision=64, seed=7)
    public_value = selected.ones(2)
    proxy_value = xp.ones(2)
    nested_value = selected.linalg.norm(selected.ones(2))

    assert selected.current_device() == "cuda:0"
    assert public_value.device == torch.device("cuda:0")
    assert proxy_value.device == torch.device("cuda:0")
    assert nested_value.device == torch.device("cuda:0")
    assert torch.get_default_device() == caller_default
    assert torch.ones(1).device == caller_allocation
    assert torch.tensordot is original_tensordot
