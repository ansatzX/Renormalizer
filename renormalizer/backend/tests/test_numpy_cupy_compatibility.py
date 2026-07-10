import importlib
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from renormalizer import set_backend
from renormalizer.backend.boundary import scalar_to_python
from renormalizer.backend.factory import SUPPORTED_BACKENDS, create_backend
from renormalizer.cons import backend, xp


@pytest.fixture(autouse=True)
def restore_numpy_backend():
    try:
        yield
    finally:
        set_backend("numpy", precision=64)


def test_numpy_conversion_preserves_noncontiguous_values():
    selected = set_backend("numpy", precision=64)
    source = np.arange(24).reshape(4, 6)[:, ::2]

    result = selected.to_backend(source)

    np.testing.assert_array_equal(result, source)
    assert result.dtype == source.dtype


def test_numpy_tensordot_matches_numpy():
    selected = set_backend("numpy", precision=64)
    a = np.arange(24.0).reshape(2, 3, 4)
    b = np.arange(20.0).reshape(4, 5)

    result = selected.tensordot(a, b, axes=([2], [0]))

    np.testing.assert_allclose(result, np.tensordot(a, b, axes=([2], [0])))


def test_scalar_to_python_rejects_nonscalar_input():
    selected = set_backend("numpy", precision=64)

    with pytest.raises(ValueError, match="scalar"):
        scalar_to_python(np.array([1.0]), selected)


def test_factory_keeps_numpy_adapter_after_optional_backends_are_registered():
    from renormalizer.backend.numpy_backend import NumpyBackend

    assert SUPPORTED_BACKENDS == ("numpy", "cupy", "jax", "torch")
    assert isinstance(create_backend("numpy"), NumpyBackend)


@pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="CuPy is not installed")
def test_post_import_cupy_switch_updates_metadata_and_owned_consumers():
    backend_module = importlib.import_module("renormalizer.mps.backend")
    gs = importlib.import_module("renormalizer.mps.gs")
    tda = importlib.import_module("renormalizer.mps.tda")
    zerot = importlib.import_module("renormalizer.cv.zerot")
    vscf = importlib.import_module("renormalizer.vibration.vscf")
    matrix = importlib.import_module("renormalizer.mps.matrix")
    oe_wrap = importlib.import_module("renormalizer.mps.oe_contract_wrap")

    selected = set_backend("cupy", device="cuda:0", precision=64)
    cp = selected.array_namespace
    device_array = matrix.asxp(np.arange(6.0).reshape(2, 3))

    assert backend.name == "cupy"
    assert selected.current_device() == "cuda:0"
    assert cp.cuda.runtime.getDevice() == 0
    assert backend_module.USE_GPU is True
    assert backend_module.GPU_ID == 0
    assert backend_module.OE_BACKEND == "cupy"
    assert backend_module.ARRAY_TYPES == (np.ndarray, cp.ndarray)
    assert cp.cuda.memory.OutOfMemoryError in backend_module.MEMORY_ERRORS
    assert isinstance(device_array, cp.ndarray)
    np.testing.assert_array_equal(matrix.asnumpy(device_array), np.arange(6.0).reshape(2, 3))
    assert oe_wrap.active_array_types() == (np.ndarray, cp.ndarray)
    assert cp.cuda.memory.OutOfMemoryError in oe_wrap.active_memory_errors()

    for consumer in (gs, tda, zerot, vscf):
        assert consumer.backend is backend_module.backend

    numpy_backend = set_backend("numpy", precision=64)
    host_array = np.arange(4.0)

    assert backend_module.USE_GPU is False
    assert backend_module.GPU_ID is None
    assert backend_module.OE_BACKEND == "numpy"
    assert backend_module.ARRAY_TYPES == (np.ndarray,)
    assert backend_module.MEMORY_ERRORS == (MemoryError,)
    assert matrix.asxp(host_array) is host_array
    assert matrix.asnumpy(host_array) is host_array
    assert numpy_backend.current_device() == "cpu"


@pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="CuPy is not installed")
def test_default_cupy_device_is_valid_and_indexed_device_is_authoritative():
    selected = set_backend("cupy", precision=64)
    cp = selected.array_namespace
    default_index = int(selected.current_device().split(":", 1)[1])

    assert selected.current_device().startswith("cuda:")
    assert 0 <= default_index < cp.cuda.runtime.getDeviceCount()
    assert selected.ones(1).device.id == default_index

    indexed = set_backend("cupy", device="cuda:0", precision=64)

    assert indexed.current_device() == "cuda:0"
    assert indexed.ones(1).device.id == 0


@pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="CuPy is not installed")
def test_cupy_array_operations_stay_on_selected_device():
    from renormalizer.model import Op
    from renormalizer.utils import Quantity

    selected = set_backend("cupy", device="cuda:0", precision=64)
    cp = selected.array_namespace
    source = np.arange(6.0).reshape(2, 3)

    x = selected.asarray(source)
    result = selected.matmul(x, selected.transpose(x))

    assert isinstance(result, cp.ndarray)
    assert result.device.id == 0
    np.testing.assert_allclose(
        selected.to_numpy(result), np.array([[5.0, 14.0], [14.0, 50.0]])
    )
    assert (Op("X", 0) * cp.asarray(2.5)).factor == 2.5
    tiny_complex = complex(2.5, 1e-300)
    assert (Op("X", 0) * cp.asarray(tiny_complex)).factor == tiny_complex
    assert Quantity(cp.asarray(3.5)).value == 3.5


@pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="CuPy is not installed")
def test_cupy_namespace_stays_bound_to_indexed_device():
    import cupy as raw_cupy

    device_count = raw_cupy.cuda.runtime.getDeviceCount()
    if device_count < 2:
        pytest.skip("requires at least two visible CUDA devices")

    original_device = raw_cupy.cuda.runtime.getDevice()
    try:
        selected = set_backend("cupy", device="cuda:1", precision=64)
        namespace = selected.array_namespace
        raw_cupy.cuda.Device(0).use()
        source = raw_cupy.arange(4.0)

        xp_result = xp.ones(4)
        namespace_result = namespace.einsum("i->i", namespace.ones(4))
        random_result = namespace.random.random(4)
        q, r = namespace.linalg.qr(namespace.eye(2))
        converted = selected.asarray(source)
        namespace_converted = namespace.asarray(source)

        assert namespace.__name__ == raw_cupy.__name__
        assert namespace.random.__name__ == raw_cupy.random.__name__
        assert namespace.ndarray is raw_cupy.ndarray
        assert namespace.dtype is raw_cupy.dtype
        assert namespace.float64 is raw_cupy.float64
        assert source.device.id == 0
        assert xp_result.device.id == 1
        assert namespace_result.device.id == 1
        assert random_result.device.id == 1
        assert q.device.id == 1
        assert r.device.id == 1
        assert converted.device.id == 1
        assert namespace_converted.device.id == 1
        assert raw_cupy.cuda.runtime.getDevice() == 0
    finally:
        raw_cupy.cuda.Device(original_device).use()


@pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="CuPy is not installed")
def test_post_import_gs_direct_contraction_uses_cupy(monkeypatch):
    gs = importlib.import_module("renormalizer.mps.gs")
    selected = set_backend("cupy", device="cuda:0", precision=64)
    cp = selected.array_namespace
    mps = SimpleNamespace(optimize_config=SimpleNamespace(method="1site"))
    left = np.array([[[2.0]]])
    center = np.arange(4.0).reshape(1, 2, 2, 1)
    right = np.array([[[3.0]]])
    qn_mask = np.ones((1, 2, 1), dtype=bool)
    contraction_backends = []
    original_contract = gs.oe_contract

    def recording_contract(*args, **kwargs):
        contraction_backends.append(kwargs.get("backend"))
        return original_contract(*args, **kwargs)

    monkeypatch.setattr(gs, "oe_contract", recording_contract)

    result = gs.get_ham_direct(
        mps,
        qn_mask,
        selected.asarray(left),
        selected.asarray(right),
        [selected.asarray(center)],
        omega=None,
    )

    expected = np.einsum("abc,bdef,lfk->adlcek", left, center, right).reshape(2, 2)
    assert isinstance(result, cp.ndarray)
    assert result.device.id == 0
    assert contraction_backends == ["cupy"]
    np.testing.assert_allclose(selected.to_numpy(result), expected)
