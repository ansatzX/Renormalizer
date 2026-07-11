import numpy as np
import pytest

from renormalizer import set_backend
from renormalizer.backend.config import BackendConfig
from renormalizer.backend.factory import create_backend
from renormalizer.backend.proxy import BackendManager
from renormalizer.cons import backend, get_backend, xp


def test_backend_proxy_keeps_stable_identity_when_selected_backend_changes():
    proxy_id = id(backend)
    selected = set_backend("numpy", precision=64, seed=2019)

    assert id(backend) == proxy_id
    assert xp is backend
    assert get_backend() is selected
    assert backend.name == "numpy"


def test_unknown_backend_fails_explicitly():
    with pytest.raises(ValueError, match="unsupported backend"):
        set_backend("cupynumeric")


@pytest.mark.parametrize(
    "option, value",
    [
        ("execution_policy", "legacy_oe"),
        ("execution_policy", "execution_ir"),
        ("fallback_policy", "error"),
        ("fallback_policy", "legacy_oe"),
        ("experimental_oe_ir", False),
        ("experimental_oe_ir", True),
    ],
)
def test_stage_three_options_are_accepted(option, value):
    selected = set_backend("numpy", **{option: value})

    assert getattr(selected.config, option) == value


@pytest.mark.parametrize(
    "option, value",
    [
        ("execution_policy", "eager"),
        ("execution_policy", None),
        ("fallback_policy", "numpy"),
        ("fallback_policy", None),
    ],
)
def test_stage_three_policy_values_are_validated(option, value):
    with pytest.raises(ValueError, match=option):
        set_backend("numpy", **{option: value})


def test_backend_config_validates_device_and_precision():
    with pytest.raises(ValueError, match="device"):
        BackendConfig(device="accelerator")

    with pytest.raises(ValueError, match="precision"):
        BackendConfig(precision=16)


@pytest.mark.parametrize(
    "device, normalized",
    [
        ("cpu", "cpu"),
        ("host", "cpu"),
        ("gpu", "gpu"),
        ("cuda", "gpu"),
        ("cuda:0", "cuda:0"),
        (" CUDA:12 ", "cuda:12"),
    ],
)
def test_backend_config_accepts_device_aliases_and_indexed_cuda(device, normalized):
    assert BackendConfig(device=device).device == normalized


def test_numpy_factory_rejects_non_cpu_device():
    with pytest.raises(ValueError, match="only supports device='cpu'"):
        create_backend("numpy", device="cuda:0")


def test_legacy_fp32_environment_only_applies_when_precision_is_omitted(monkeypatch):
    monkeypatch.setenv("RENO_FP32", "1")

    assert create_backend("numpy").is_32bits
    assert not create_backend("numpy", precision=64).is_32bits
    assert not create_backend("numpy", config=BackendConfig()).is_32bits


def test_none_is_preserved_across_shared_backend_conversions():
    selected = create_backend("numpy")

    assert selected.to_numpy(None) is None
    assert selected.from_numpy(None) is None
    assert selected.to_host(None) is None
    assert selected.to_backend(None) is None


def test_backend_manager_applies_configured_and_legacy_seeds(monkeypatch):
    monkeypatch.delenv("RENO_FP32", raising=False)
    expected_legacy = np.random.RandomState(2019).random_sample()
    expected_configured = np.random.RandomState(7).random_sample()

    np.random.seed(12345)
    manager = BackendManager()
    assert manager.get_backend().random.random_sample() == expected_legacy

    selected = manager.set_backend("numpy", seed=7)
    assert selected.random.random_sample() == expected_configured

    np.random.seed(12345)
    configured_manager = BackendManager(config=BackendConfig(seed=7))
    assert configured_manager.get_backend().random.random_sample() == expected_configured


@pytest.mark.parametrize("value", [None, 0, 1, ""])
def test_experimental_oe_ir_requires_a_boolean(value):
    with pytest.raises(TypeError, match="boolean"):
        BackendConfig(experimental_oe_ir=value)


def test_backend_config_is_exported_from_all_public_facades():
    from renormalizer import BackendConfig as TopLevelConfig
    from renormalizer.backend import BackendConfig as BackendPackageConfig
    from renormalizer.cons import BackendConfig as ConsConfig

    assert BackendConfig is TopLevelConfig
    assert BackendConfig is BackendPackageConfig
    assert BackendConfig is ConsConfig
