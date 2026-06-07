# -*- coding: utf-8 -*-

import os

import numpy as np
import pytest


def test_numpy_backend_gradient_capabilities_are_explicit():
    import renormalizer as r

    assert r.backend.name == "numpy"
    assert r.backend.supports_autodiff is False
    assert r.backend.supports_jit is False
    assert r.backend.supports_functional_update is True

    with pytest.raises(NotImplementedError, match="does not provide autodiff transform 'grad'"):
        r.backend.transforms.grad(lambda x: x)


def test_numpy_backend_distributed_noops():
    import renormalizer as r

    x = np.array([1.0, 2.0])
    assert r.backend.rank == 0
    assert r.backend.size == 1
    assert r.backend.is_distributed is False
    assert r.backend.allreduce(x) is x
    assert r.backend.broadcast(x) is x
    assert r.backend.gather(x) == [x]
    assert r.backend.allgather(x) == [x]


def test_backend_proxy_identity_and_stale_xp_dispatch():
    import renormalizer as r
    from renormalizer.mps.backend import backend as legacy_backend
    from renormalizer.mps.backend import xp

    assert legacy_backend is r.backend
    old_xp = xp
    r.set_backend("numpy")
    assert old_xp is xp

    a = xp.ones((2, 2))
    b = xp.eye(2) + (1 - xp.eye(2))
    assert xp.allclose(a, b)
    assert r.backend.name == "numpy"


def test_numpy_backend_functional_updates_return_updated_array():
    import renormalizer as r

    x = r.backend.zeros((3,))
    y = r.backend.at_set(x, 1, 2.0)
    z = r.backend.at_add(y, 1, 3.0)

    assert r.backend.numpy(y).tolist() == [0.0, 2.0, 0.0]
    assert r.backend.numpy(z).tolist() == [0.0, 5.0, 0.0]


def test_matrix_stays_host_numpy_and_asxp_uses_backend_boundary():
    from renormalizer.mps.matrix import Matrix, asnumpy, asxp
    import renormalizer as r

    mat = Matrix([[1.0, 2.0], [3.0, 4.0]])

    assert isinstance(mat.array, np.ndarray)
    assert isinstance(asnumpy(mat), np.ndarray)

    xp_array = asxp(mat)
    assert r.backend.is_array(xp_array)
    assert r.backend.numpy(xp_array).tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_asnumpy_handles_backend_array_and_list():
    from renormalizer.mps.matrix import asnumpy, asxp

    backend_array = asxp(np.array([1.0, 2.0]))
    assert asnumpy(backend_array).tolist() == [1.0, 2.0]
    assert asnumpy([1.0, 2.0]).tolist() == [1.0, 2.0]


def test_scalar_to_python_handles_real_and_complex_numpy_scalars():
    from renormalizer.mps.matrix import scalar_to_python

    assert scalar_to_python(np.array(1.5)) == 1.5
    assert scalar_to_python(np.array(1.5 + 0j)) == 1.5
    assert scalar_to_python(np.array(1.5 + 2j)) == complex(1.5 + 2j)


def test_scalar_to_python_handles_torch_real_scalar_when_available():
    try:
        import torch
    except ImportError as exc:
        pytest.skip("could not import 'torch': {0}".format(exc))
    except OSError as exc:
        pytest.skip("torch is installed but failed to load: {0}".format(exc))

    from renormalizer.mps.matrix import scalar_to_python

    scalar = torch.tensor(1.25)
    assert scalar_to_python(scalar) == 1.25


def test_matrix_orthogonality_checks_use_tensor_dtype_for_identity():
    from renormalizer.mps.matrix import Matrix
    import renormalizer as r

    try:
        r.set_backend("torch", device="cpu", precision=64)
    except Exception as exc:
        pytest.skip("torch backend unavailable: {0}".format(exc))

    try:
        mat = Matrix(np.eye(2))
        assert mat.check_lortho()
        assert mat.check_rortho()
    finally:
        r.set_backend("numpy", precision=64)


def test_legacy_backend_constants_are_not_used_in_core_call_sites():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    checked = [
        repo / "renormalizer" / "mps" / "gs.py",
        repo / "renormalizer" / "mps" / "tda.py",
        repo / "renormalizer" / "cv" / "zerot.py",
        repo / "renormalizer" / "vibration" / "vscf.py",
    ]

    for path in checked:
        text = path.read_text()
        assert "OE_BACKEND" not in text


def test_backend_protocol_and_conversion_surface():
    import renormalizer as r
    from renormalizer.backend import BackendProtocol

    assert isinstance(r.backend, BackendProtocol)
    assert r.backend.supports_cpu is True
    assert r.backend.supports_gpu is False
    assert r.backend.supports_sparse is False
    assert r.backend.device == "cpu"
    assert r.backend.supported_device_kinds == ("cpu",)
    assert r.backend.available_device_kinds == ("cpu",)
    assert r.backend.host_array_types
    assert r.backend.device_array_types == ()

    x = np.array([1.0, 2.0])
    y = r.backend.to_backend(x)
    z = r.backend.to_host(y)

    assert r.backend.is_host_array(x)
    assert r.backend.is_array(y)
    assert r.backend.is_host_array(z)
    assert z.tolist() == [1.0, 2.0]
    assert r.backend.to_numpy(y).tolist() == [1.0, 2.0]
    assert r.backend.numpy(y).tolist() == [1.0, 2.0]


def test_numpy_backend_explicit_conversion_methods():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    x = np.array([[1.0, 2.0]])
    y = backend.to_backend(x)

    assert isinstance(y, np.ndarray)
    assert backend.is_host_array(y)
    assert backend.is_device_array(y) is False
    assert backend.to_host(y).tolist() == [[1.0, 2.0]]
    assert backend.to_numpy(y).tolist() == [[1.0, 2.0]]


def test_numpy_backend_array_preserves_copy_false_semantics():
    from renormalizer.backend.numpy_backend import NumpyBackend

    backend = NumpyBackend()
    x = np.array([1.0, 2.0])
    y = backend.array(x, copy=False)
    z = backend.array(x, copy=True)

    assert np.shares_memory(x, y)
    assert not np.shares_memory(x, z)


def test_backend_factory_accepts_explicit_backend_config():
    from renormalizer.backend import BackendConfig
    from renormalizer.backend.factory import create_backend

    backend = create_backend("numpy", config=BackendConfig(device="cpu", precision=32))

    assert backend.device == "cpu"
    assert backend.config.device == "cpu"
    assert backend.real_dtype == np.float32

    backend = create_backend("numpy", device="cpu", precision="64")
    assert backend.real_dtype == np.float64

    with pytest.raises(ValueError, match="numpy backend does not support device 'gpu'"):
        create_backend("numpy", device="gpu")


def test_public_set_backend_accepts_explicit_backend_config():
    import renormalizer as r

    try:
        selected = r.set_backend("numpy", device="cpu", precision=32)

        assert selected is r.backend.current
        assert r.backend.device == "cpu"
        assert r.backend.real_dtype == np.float32
    finally:
        r.set_backend("numpy", precision=64)


def test_cupy_backend_explicit_conversion_methods_when_available():
    try:
        import cupy
    except ImportError as exc:
        pytest.skip("could not import 'cupy': {0}".format(exc))
    except OSError as exc:
        pytest.skip("cupy is installed but failed to load: {0}".format(exc))
    from renormalizer.backend.cupy_backend import CupyBackend

    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip("CuPy is installed but CUDA is not available: {0}".format(exc))
    if device_count == 0:
        pytest.skip("CuPy is installed but no CUDA device is available")

    backend = CupyBackend()
    x = np.array([1.0, 2.0])
    y = backend.to_backend(x)

    assert backend.is_device_array(y)
    assert backend.is_array(y)
    assert backend.to_host(y).tolist() == [1.0, 2.0]
    assert backend.to_numpy(y).tolist() == [1.0, 2.0]


def test_jax_backend_explicit_conversion_methods_when_available():
    try:
        __import__("jax")
    except ImportError as exc:
        pytest.skip("could not import 'jax': {0}".format(exc))
    except OSError as exc:
        pytest.skip("jax is installed but failed to load: {0}".format(exc))
    from renormalizer.backend.jax_backend import JaxBackend

    backend = JaxBackend()
    x = np.array([1.0, 2.0])
    y = backend.to_backend(x)

    assert backend.is_array(y)
    assert backend.is_device_array(y)
    assert backend.to_host(y).tolist() == [1.0, 2.0]
    assert backend.to_numpy(y).tolist() == [1.0, 2.0]


def test_task4_backend_modules_are_import_safe_without_optional_deps():
    import importlib

    importlib.import_module("renormalizer.backend.cupynumeric_backend")
    importlib.import_module("renormalizer.backend.torch_backend")
    importlib.import_module("renormalizer.backend.jax_backend")


def test_cupynumeric_backend_module_import_is_lazy(monkeypatch):
    import builtins
    import importlib.util
    from pathlib import Path

    backend_path = Path(__file__).resolve().parents[2] / "backend" / "cupynumeric_backend.py"
    real_import = builtins.__import__
    cupynumeric_imports = []

    def blocked_cupynumeric_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cupynumeric":
            cupynumeric_imports.append(name)
            raise AssertionError("cupynumeric should not be imported at module import time")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_cupynumeric_import)

    spec = importlib.util.spec_from_file_location(
        "renormalizer.backend._cupynumeric_backend_lazy_import_test",
        str(backend_path),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert cupynumeric_imports == []
    assert module.cnp is None
    assert module._IMPORT_ERROR is None


def test_task4_optional_backend_modules_are_import_safe_with_broken_binary_deps(monkeypatch):
    import builtins
    import importlib.util
    from pathlib import Path

    backend_dir = Path(__file__).resolve().parents[2] / "backend"
    real_import = builtins.__import__

    def blocked_optional_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"cupynumeric", "torch"}:
            raise OSError("blocked binary dependency: {0}".format(name))
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_optional_import)

    modules = {}
    for module_name in ("cupynumeric_backend", "torch_backend"):
        spec = importlib.util.spec_from_file_location(
            "renormalizer.backend._task4_{0}_broken_import_test".format(module_name),
            str(backend_dir / "{0}.py".format(module_name)),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[module_name] = module

    assert modules["cupynumeric_backend"].cnp is None
    assert modules["cupynumeric_backend"]._IMPORT_ERROR is None
    with pytest.raises(ImportError, match="cupynumeric is not installed") as cupynumeric_exc:
        modules["cupynumeric_backend"].CupynumericBackend()
    assert isinstance(modules["cupynumeric_backend"]._IMPORT_ERROR, OSError)
    assert cupynumeric_exc.value.__cause__ is modules["cupynumeric_backend"]._IMPORT_ERROR

    assert modules["torch_backend"].torch is None
    assert isinstance(modules["torch_backend"]._IMPORT_ERROR, OSError)
    with pytest.raises(ImportError, match="torch is not installed") as torch_exc:
        modules["torch_backend"].TorchBackend()
    assert torch_exc.value.__cause__ is modules["torch_backend"]._IMPORT_ERROR


def test_cupynumeric_backend_defaults_legate_auto_config_and_catches_runtime_error(monkeypatch):
    import builtins
    import importlib.util
    from pathlib import Path

    backend_path = Path(__file__).resolve().parents[2] / "backend" / "cupynumeric_backend.py"
    real_import = builtins.__import__
    observed_env = {}

    def blocked_cupynumeric_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cupynumeric":
            observed_env["LEGATE_AUTO_CONFIG"] = os.environ.get("LEGATE_AUTO_CONFIG")
            raise RuntimeError("Legate auto-configuration failed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delenv("LEGATE_AUTO_CONFIG", raising=False)
    monkeypatch.delenv("LEGATE_CONFIG", raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked_cupynumeric_import)

    spec = importlib.util.spec_from_file_location(
        "renormalizer.backend._cupynumeric_backend_legate_runtime_error_test",
        str(backend_path),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(ImportError, match="cupynumeric is not installed") as excinfo:
        module.CupynumericBackend()

    assert observed_env["LEGATE_AUTO_CONFIG"] == "0"
    assert module.cnp is None
    assert isinstance(module._IMPORT_ERROR, RuntimeError)
    assert excinfo.value.__cause__ is module._IMPORT_ERROR


def test_cupynumeric_backend_constructor_reports_missing_dependency(monkeypatch):
    from renormalizer.backend import cupynumeric_backend

    import_error = ImportError("blocked cupynumeric")
    monkeypatch.setattr(cupynumeric_backend, "cnp", None)
    monkeypatch.setattr(cupynumeric_backend, "_IMPORT_ERROR", import_error)

    with pytest.raises(ImportError, match="cupynumeric is not installed") as excinfo:
        cupynumeric_backend.CupynumericBackend()
    assert excinfo.value.__cause__ is import_error


def test_torch_backend_constructor_reports_missing_dependency(monkeypatch):
    from renormalizer.backend import torch_backend

    import_error = ImportError("blocked torch")
    monkeypatch.setattr(torch_backend, "torch", None)
    monkeypatch.setattr(torch_backend, "_IMPORT_ERROR", import_error)

    with pytest.raises(ImportError, match="torch is not installed") as excinfo:
        torch_backend.TorchBackend()
    assert excinfo.value.__cause__ is import_error


def test_jax_backend_module_is_import_safe_with_missing_dependency(monkeypatch):
    import builtins
    import importlib.util
    from pathlib import Path

    backend_path = Path(__file__).resolve().parents[2] / "backend" / "jax_backend.py"
    real_import = builtins.__import__

    def blocked_jax_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jax" or name.startswith("jax."):
            raise OSError("blocked jax binary dependency")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_jax_import)

    spec = importlib.util.spec_from_file_location(
        "renormalizer.backend._jax_backend_broken_import_test",
        str(backend_path),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.jax is None
    assert module.jnp is None
    assert module.jr is None
    assert isinstance(module._IMPORT_ERROR, OSError)
    with pytest.raises(ImportError, match="jax is not installed") as excinfo:
        module.JaxBackend()
    assert excinfo.value.__cause__ is module._IMPORT_ERROR


def test_cupynumeric_backend_explicit_conversion_methods_when_available():
    try:
        from renormalizer.backend.cupynumeric_backend import CupynumericBackend
        backend = CupynumericBackend()
    except ImportError as exc:
        pytest.skip("could not initialize 'cupynumeric': {0}".format(exc))

    x = np.array([1.0, 2.0])
    y = backend.to_backend(x)

    assert backend.is_array(y)
    assert backend.to_host(y).tolist() == [1.0, 2.0]
    assert backend.to_numpy(y).tolist() == [1.0, 2.0]


def test_cupynumeric_backend_copies_view_when_backend_rejects_non_affine_attach(monkeypatch):
    from renormalizer.backend import cupynumeric_backend

    class FakeCupynumericArray:
        def __init__(self, value):
            self.value = np.asarray(value)

    class FakeCupynumeric:
        ndarray = FakeCupynumericArray
        linalg = np.linalg
        random = np.random

        @staticmethod
        def asarray(x):
            arr = np.asarray(x)
            if not arr.flags["C_CONTIGUOUS"]:
                raise NotImplementedError(
                    "cuPyNumeric does not currently know how to attach to array views "
                    "that are not affine transforms of their parent array."
                )
            return FakeCupynumericArray(arr)

        @staticmethod
        def asnumpy(x):
            return x.value

    monkeypatch.setattr(cupynumeric_backend, "cnp", FakeCupynumeric)
    monkeypatch.setattr(cupynumeric_backend, "_IMPORT_ERROR", None)

    backend = cupynumeric_backend.CupynumericBackend()
    view = np.arange(12).reshape(3, 4)[:, ::2]
    converted = backend.to_backend(view)

    assert backend.is_array(converted)
    assert backend.to_numpy(converted).tolist() == view.tolist()


def test_cupynumeric_backend_fails_fast_on_rank_gt_4_tensordot(monkeypatch):
    import renormalizer as r
    from renormalizer.mps.matrix import tensordot

    try:
        r.set_backend("cupynumeric")
    except Exception as exc:
        pytest.skip("cupynumeric backend unavailable: {0}".format(exc))

    try:
        a = np.zeros((2, 2, 2))
        b = np.zeros((2, 2, 2, 2))
        with pytest.raises(NotImplementedError, match="rank > 4"):
            tensordot(a, b, axes=([1], [1]))
    finally:
        r.set_backend("numpy", precision=64)


def test_torch_backend_explicit_conversion_methods_when_available():
    try:
        __import__("torch")
    except ImportError as exc:
        pytest.skip("could not import 'torch': {0}".format(exc))
    except OSError as exc:
        pytest.skip("torch is installed but failed to load: {0}".format(exc))
    from renormalizer.backend.torch_backend import TorchBackend

    backend = TorchBackend()
    x = np.array([1.0, 2.0])
    y = backend.to_backend(x)

    assert backend.is_array(y)
    assert backend.to_host(y).tolist() == [1.0, 2.0]
    assert backend.to_numpy(y).tolist() == [1.0, 2.0]


def test_torch_backend_honors_explicit_device_config_with_fake_module(monkeypatch):
    from renormalizer.backend import BackendConfig, torch_backend

    class FakeTorchTensor:
        def __init__(self, value, dtype=None, device=None):
            self.value = value
            self.dtype = dtype
            self.device = device

        def clone(self):
            value = self.value.copy() if hasattr(self.value, "copy") else self.value
            return FakeTorchTensor(value, dtype=self.dtype, device=self.device)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self.value)

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

    class FakeTorch:
        Tensor = FakeTorchTensor
        float32 = object()
        float64 = object()
        complex64 = object()
        complex128 = object()
        cuda = FakeCuda()
        linalg = object()
        random = object()
        device_calls = []

        @staticmethod
        def device(name):
            FakeTorch.device_calls.append(name)
            return ("device", name)

        @staticmethod
        def tensor(*args, **kwargs):
            dtype = kwargs.pop("dtype", None)
            device = kwargs.pop("device", None)
            return FakeTorchTensor(np.array(*args, **kwargs), dtype=dtype, device=device)

        @staticmethod
        def as_tensor(*args, **kwargs):
            dtype = kwargs.pop("dtype", None)
            device = kwargs.pop("device", None)
            return FakeTorchTensor(np.asarray(*args, **kwargs), dtype=dtype, device=device)

        @staticmethod
        def rand(*shape, dtype=None, device=None):
            return FakeTorchTensor(("rand", shape), dtype=dtype, device=device)

        @staticmethod
        def randn(*shape, dtype=None, device=None):
            return FakeTorchTensor(("randn", shape), dtype=dtype, device=device)

        @staticmethod
        def randint(low, high, size, dtype=None, device=None):
            return FakeTorchTensor(("randint", low, high, size), dtype=dtype, device=device)

    monkeypatch.setattr(torch_backend, "torch", FakeTorch)

    backend = torch_backend.TorchBackend(config=BackendConfig(device="gpu", precision=32))

    assert backend.device == "gpu"
    assert backend.available_device_kinds == ("cpu", "gpu")
    assert FakeTorch.device_calls == ["cuda"]
    assert backend.array([1.0]).device == ("device", "cuda")
    assert backend.asarray([1.0]).device == ("device", "cuda")
    assert backend.from_numpy(np.array([1.0])).device == ("device", "cuda")
    assert backend.to_backend(np.array([1.0])).device == ("device", "cuda")
    assert backend.random.random([2]).device == ("device", "cuda")
    assert backend.random.randn(2).device == ("device", "cuda")
    assert backend.random.randint(10).device == ("device", "cuda")


def test_torch_backend_reports_unavailable_explicit_gpu(monkeypatch):
    from renormalizer.backend import BackendConfig, torch_backend

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        Tensor = object
        float32 = object()
        float64 = object()
        complex64 = object()
        complex128 = object()
        cuda = FakeCuda()
        linalg = object()
        random = object()

    monkeypatch.setattr(torch_backend, "torch", FakeTorch)

    with pytest.raises(ValueError, match="torch backend device 'gpu' was requested"):
        torch_backend.TorchBackend(config=BackendConfig(device="gpu"))


def test_task4_backend_conversion_methods_with_fake_optional_modules(monkeypatch):
    from renormalizer.backend import cupynumeric_backend, torch_backend

    class FakeCupynumericArray:
        def __init__(self, value):
            self.value = np.asarray(value)

    class FakeCupynumeric:
        ndarray = FakeCupynumericArray
        linalg = np.linalg
        random = np.random

        @staticmethod
        def array(*args, **kwargs):
            return FakeCupynumericArray(np.array(*args, **kwargs))

        @staticmethod
        def asarray(*args, **kwargs):
            return FakeCupynumericArray(np.asarray(*args, **kwargs))

        @staticmethod
        def asnumpy(x):
            return x.value

    class FakeTorchTensor:
        def __init__(self, value, dtype=None):
            self.value = np.asarray(value)
            self.dtype = dtype

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class FakeTorch:
        Tensor = FakeTorchTensor
        float32 = object()
        float64 = object()
        complex64 = object()
        complex128 = object()
        linalg = object()
        random = object()

        @staticmethod
        def tensor(*args, **kwargs):
            dtype = kwargs.pop("dtype", None)
            return FakeTorchTensor(np.array(*args, **kwargs), dtype=dtype)

        @staticmethod
        def as_tensor(*args, **kwargs):
            dtype = kwargs.pop("dtype", None)
            return FakeTorchTensor(np.asarray(*args, **kwargs), dtype=dtype)

        @staticmethod
        def ones(*args, **kwargs):
            dtype = kwargs.pop("dtype", None)
            return FakeTorchTensor(np.ones(*args, **kwargs), dtype=dtype)

    monkeypatch.setattr(cupynumeric_backend, "cnp", FakeCupynumeric)
    monkeypatch.setattr(torch_backend, "torch", FakeTorch)

    cnp_backend = cupynumeric_backend.CupynumericBackend()
    cnp_array = cnp_backend.to_backend(np.array([1.0, 2.0]))
    assert cnp_backend.is_array(cnp_array)
    assert cnp_backend.to_numpy(cnp_array).tolist() == [1.0, 2.0]

    torch_backend_obj = torch_backend.TorchBackend()
    assert torch_backend_obj.real_dtype is FakeTorch.float64
    assert torch_backend_obj.complex_dtype is FakeTorch.complex128
    assert torch_backend_obj.memory_errors == (MemoryError,)
    assert torch_backend_obj.ones((2,), dtype=torch_backend_obj.real_dtype).dtype is FakeTorch.float64

    torch_backend_obj.use_32bits()
    assert torch_backend_obj.real_dtype is FakeTorch.float32
    assert torch_backend_obj.complex_dtype is FakeTorch.complex64

    torch_array = torch_backend_obj.to_backend(np.array([1.0, 2.0]))
    assert torch_backend_obj.is_array(torch_array)
    assert torch_backend_obj.to_numpy(torch_array).tolist() == [1.0, 2.0]


def test_torch_backend_random_proxy_supports_numpy_like_methods(monkeypatch):
    from renormalizer.backend import torch_backend

    class FakeTorchTensor:
        def __init__(self, value):
            self.value = value

    class FakeTorchRandom:
        seed_calls = []

        @staticmethod
        def seed(seed=None):
            FakeTorchRandom.seed_calls.append(seed)
            return seed

    class FakeTorch:
        Tensor = FakeTorchTensor
        float32 = object()
        float64 = object()
        complex64 = object()
        complex128 = object()
        linalg = object()
        random = FakeTorchRandom()
        manual_seed_calls = []
        rand_calls = []
        randn_calls = []
        randint_calls = []

        @staticmethod
        def manual_seed(seed):
            FakeTorch.manual_seed_calls.append(seed)
            return seed

        @staticmethod
        def rand(*shape, dtype=None):
            FakeTorch.rand_calls.append((shape, dtype))
            return FakeTorchTensor(("rand", shape, dtype))

        @staticmethod
        def randn(*shape, dtype=None):
            FakeTorch.randn_calls.append((shape, dtype))
            return FakeTorchTensor(("randn", shape, dtype))

        @staticmethod
        def randint(low, high, size, dtype=None):
            FakeTorch.randint_calls.append((low, high, size, dtype))
            return FakeTorchTensor(("randint", low, high, size, dtype))

        @staticmethod
        def tensor(*args, **kwargs):
            return FakeTorchTensor(args)

        @staticmethod
        def as_tensor(*args, **kwargs):
            return FakeTorchTensor(args)

    monkeypatch.setattr(torch_backend, "torch", FakeTorch)

    backend = torch_backend.TorchBackend()
    backend.random.seed(7)
    backend.random.random()
    backend.random.random([2, 3])
    backend.random.rand()
    backend.random.rand(4, 5)
    backend.random.randn()
    backend.random.randn(6)
    backend.random.randint(10)
    backend.random.random(np.int64(7))
    backend.random.randint(2, 9, size=(3,))

    assert FakeTorch.manual_seed_calls == [7]
    assert FakeTorchRandom.seed_calls == []
    assert FakeTorch.rand_calls == [
        (((),), FakeTorch.float64),
        (((2, 3),), FakeTorch.float64),
        (((),), FakeTorch.float64),
        ((4, 5), FakeTorch.float64),
        (((np.int64(7),),), FakeTorch.float64),
    ]
    assert FakeTorch.randn_calls == [(((),), FakeTorch.float64), ((6,), FakeTorch.float64)]
    assert FakeTorch.randint_calls == [(0, 10, (), None), (2, 9, (3,), None)]


def test_torch_backend_random_scalar_methods_when_available():
    try:
        __import__("torch")
    except ImportError as exc:
        pytest.skip("could not import 'torch': {0}".format(exc))
    except OSError as exc:
        pytest.skip("torch is installed but failed to load: {0}".format(exc))
    from renormalizer.backend.torch_backend import TorchBackend

    backend = TorchBackend()

    assert backend.random.random().shape == ()
    assert backend.random.rand().shape == ()
    assert backend.random.randn().shape == ()
    assert backend.random.randint(10).shape == ()


def test_jax_backend_configures_x64_and_random_proxy_with_fake_modules(monkeypatch):
    from renormalizer.backend import jax_backend

    class FakeJaxArray:
        def __mul__(self, other):
            return self

        def __add__(self, other):
            return self

    class FakeJaxConfig:
        updates = []

        @staticmethod
        def update(name, value):
            FakeJaxConfig.updates.append((name, value))

    class FakeLax:
        @staticmethod
        def stop_gradient(x):
            return x

    class FakeJax:
        config = FakeJaxConfig
        lax = FakeLax

        @staticmethod
        def grad(*args, **kwargs):
            return ("grad", args, kwargs)

        @staticmethod
        def value_and_grad(*args, **kwargs):
            return ("value_and_grad", args, kwargs)

        @staticmethod
        def jit(*args, **kwargs):
            return ("jit", args, kwargs)

        @staticmethod
        def vmap(*args, **kwargs):
            return ("vmap", args, kwargs)

    class FakeJnp:
        ndarray = FakeJaxArray
        linalg = object()
        float32 = object()
        float64 = object()
        complex64 = object()
        complex128 = object()

        @staticmethod
        def asarray(x, dtype=None):
            return ("asarray", x, dtype)

    class FakeJr:
        calls = []

        @staticmethod
        def PRNGKey(seed):
            return ("key", seed)

        @staticmethod
        def split(key):
            return (("next", key), ("sub", key))

        @staticmethod
        def uniform(key, shape=(), minval=0.0, maxval=1.0, dtype=None):
            FakeJr.calls.append(("uniform", key, shape, minval, maxval, dtype))
            return FakeJaxArray()

        @staticmethod
        def normal(key, shape=(), dtype=None):
            FakeJr.calls.append(("normal", key, shape, dtype))
            return FakeJaxArray()

        @staticmethod
        def randint(key, shape, minval, maxval, dtype=int):
            FakeJr.calls.append(("randint", key, shape, minval, maxval, dtype))
            return FakeJaxArray()

    monkeypatch.setattr(jax_backend, "jax", FakeJax)
    monkeypatch.setattr(jax_backend, "jnp", FakeJnp)
    monkeypatch.setattr(jax_backend, "jr", FakeJr)

    backend = jax_backend.JaxBackend()
    assert ("jax_enable_x64", True) in FakeJaxConfig.updates
    assert backend.real_dtype is FakeJnp.float64

    backend.use_32bits()
    assert FakeJaxConfig.updates[-1] == ("jax_enable_x64", False)
    assert backend.real_dtype is FakeJnp.float32

    backend.use_64bits()
    assert FakeJaxConfig.updates[-1] == ("jax_enable_x64", True)
    assert backend.real_dtype is FakeJnp.float64

    backend.random.random([2, 3])
    backend.random.rand(4, 5)
    backend.random.randn(6)
    backend.random.randint(2, 9, size=(3,))

    assert FakeJr.calls[0][0] == "uniform"
    assert FakeJr.calls[0][2] == (2, 3)
    assert FakeJr.calls[1][2] == (4, 5)
    assert FakeJr.calls[2][0] == "normal"
    assert FakeJr.calls[2][2] == (6,)
    assert FakeJr.calls[3] == ("randint", FakeJr.calls[3][1], (3,), 2, 9, int)


def test_jax_backend_can_be_explicitly_configured_for_cpu_or_gpu(monkeypatch):
    from renormalizer.backend import BackendConfig, jax_backend

    class FakeJaxArray:
        pass

    class FakeDevice:
        def __init__(self, platform, ident):
            self.platform = platform
            self.id = ident

    class FakeJaxConfig:
        updates = []

        @staticmethod
        def update(name, value):
            FakeJaxConfig.updates.append((name, value))

    class FakeJax:
        config = FakeJaxConfig
        lax = object()
        devices_calls = []
        device_put_calls = []
        _devices = [FakeDevice("cpu", 0), FakeDevice("gpu", 0)]

        @staticmethod
        def devices(kind=None):
            FakeJax.devices_calls.append(kind)
            if kind is None:
                return list(FakeJax._devices)
            return [device for device in FakeJax._devices if device.platform == kind]

        @staticmethod
        def default_backend():
            return "cpu"

        @staticmethod
        def device_put(x, device=None):
            FakeJax.device_put_calls.append((x, device))
            return x

    class FakeJnp:
        ndarray = FakeJaxArray
        linalg = object()
        float32 = object()
        float64 = object()
        complex64 = object()
        complex128 = object()

        @staticmethod
        def asarray(x, dtype=None):
            return FakeJaxArray()

        @staticmethod
        def array(x, dtype=None):
            return FakeJaxArray()

    class FakeJr:
        @staticmethod
        def PRNGKey(seed):
            return ("key", seed)

    monkeypatch.setattr(jax_backend, "jax", FakeJax)
    monkeypatch.setattr(jax_backend, "jnp", FakeJnp)
    monkeypatch.setattr(jax_backend, "jr", FakeJr)

    gpu_backend = jax_backend.JaxBackend(config=BackendConfig(device="gpu", precision=32))
    gpu_backend.to_backend([1.0])

    assert gpu_backend.supports_cpu is True
    assert gpu_backend.supports_gpu is True
    assert gpu_backend.supported_device_kinds == ("cpu", "gpu")
    assert gpu_backend.available_device_kinds == ("cpu", "gpu")
    assert gpu_backend.device == "gpu"
    assert gpu_backend.real_dtype is FakeJnp.float32
    assert FakeJax.device_put_calls[-1][1].platform == "gpu"

    cpu_backend = jax_backend.JaxBackend(config=BackendConfig(device="cpu"))
    cpu_backend.to_backend([1.0])

    assert cpu_backend.device == "cpu"
    assert FakeJax.device_put_calls[-1][1].platform == "cpu"


def test_jax_backend_reports_unavailable_explicit_gpu(monkeypatch):
    from renormalizer.backend import BackendConfig, jax_backend

    class FakeDevice:
        platform = "cpu"

    class FakeJaxConfig:
        @staticmethod
        def update(name, value):
            return None

    class FakeJax:
        config = FakeJaxConfig

        @staticmethod
        def devices(kind=None):
            if kind in (None, "cpu"):
                return [FakeDevice()]
            return []

        @staticmethod
        def default_backend():
            return "cpu"

    class FakeJnp:
        ndarray = object
        linalg = object()
        float32 = object()
        float64 = object()
        complex64 = object()
        complex128 = object()

    class FakeJr:
        @staticmethod
        def PRNGKey(seed):
            return ("key", seed)

    monkeypatch.setattr(jax_backend, "jax", FakeJax)
    monkeypatch.setattr(jax_backend, "jnp", FakeJnp)
    monkeypatch.setattr(jax_backend, "jr", FakeJr)

    with pytest.raises(ValueError, match="JAX GPU device was requested"):
        jax_backend.JaxBackend(config=BackendConfig(device="gpu"))


def test_torch_backend_public_selection_seeds_and_uses_default_float_dtype(monkeypatch):
    import renormalizer as r
    from renormalizer.backend import factory, torch_backend

    class FakeTorchRandom:
        @staticmethod
        def seed(*args):
            raise AssertionError("TorchBackend should seed with torch.manual_seed()")

    class FakeTorchTensor:
        def __init__(self, value, dtype=None):
            self.value = np.asarray(value)
            self.dtype = dtype

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class FakeTorch:
        Tensor = FakeTorchTensor
        float32 = object()
        float64 = object()
        complex64 = object()
        complex128 = object()
        linalg = object()
        random = FakeTorchRandom()
        OutOfMemoryError = type("FakeTorchOutOfMemoryError", (RuntimeError,), {})
        manual_seed_calls = []

        @staticmethod
        def manual_seed(seed):
            FakeTorch.manual_seed_calls.append(seed)
            return seed

        @staticmethod
        def tensor(*args, **kwargs):
            dtype = kwargs.pop("dtype", None)
            return FakeTorchTensor(np.array(*args, **kwargs), dtype=dtype)

        @staticmethod
        def as_tensor(*args, **kwargs):
            dtype = kwargs.pop("dtype", None)
            return FakeTorchTensor(np.asarray(*args, **kwargs), dtype=dtype)

        @staticmethod
        def asarray(*args, **kwargs):
            return FakeTorch.as_tensor(*args, **kwargs)

    monkeypatch.setattr(factory.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(torch_backend, "torch", FakeTorch)

    try:
        selected = r.set_backend("torch")

        assert FakeTorch.manual_seed_calls == [2019]
        assert selected.supports_autodiff is False
        assert selected.memory_errors == (MemoryError, FakeTorch.OutOfMemoryError)
        assert selected.array([1.0]).dtype is FakeTorch.float64
        assert selected.asarray([1.0]).dtype is FakeTorch.float64
        assert selected.to_backend([1.0]).dtype is FakeTorch.float64

        selected.use_32bits()
        assert selected.asarray([1.0]).dtype is FakeTorch.float32
        assert selected.to_backend(np.array([1.0])).dtype is FakeTorch.float32
        assert selected.asarray([1]).dtype is None
    finally:
        r.set_backend("numpy")


def test_backend_protocol_source_stays_python36_import_compatible():
    import importlib
    import inspect

    protocol = importlib.import_module("renormalizer.backend.protocol")
    abstract = importlib.import_module("renormalizer.backend.abstract")

    protocol_source = inspect.getsource(protocol)
    abstract_source = inspect.getsource(abstract)

    assert "from __future__ import annotations" not in protocol_source
    assert "from __future__ import annotations" not in abstract_source
    assert "typing_extensions" in protocol_source
    assert "Protocol = object" not in protocol_source
    assert "_RuntimeBackendProtocolMeta" in protocol_source


def test_backend_protocol_fallback_remains_structural_without_typing_extensions(monkeypatch):
    import builtins
    import importlib.util
    from pathlib import Path

    protocol_path = Path(__file__).resolve().parents[2] / "backend" / "protocol.py"
    real_import = builtins.__import__

    def blocked_protocol_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "typing" and {"Protocol", "runtime_checkable"} & set(fromlist or ()):
            raise ImportError("blocked typing.Protocol")
        if name == "typing_extensions":
            raise ImportError("blocked typing_extensions")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_protocol_import)

    spec = importlib.util.spec_from_file_location("renormalizer.backend._protocol_fallback_test", str(protocol_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    attrs = {name: None for name in module._BACKEND_PROTOCOL_RUNTIME_ATTRS}
    attrs.update({name: (lambda *args, **kwargs: None) for name in module._BACKEND_PROTOCOL_RUNTIME_METHODS})
    fallback_backend = type("FallbackBackend", (), attrs)()
    fallback_missing_method = type("FallbackMissingMethod", (), {
        name: value for name, value in attrs.items() if name != "to_backend"
    })()
    fallback_noncallable_method = type("FallbackNoncallableMethod", (), dict(attrs, to_backend=None))()

    assert isinstance(fallback_backend, module.BackendProtocol)
    assert not isinstance(fallback_missing_method, module.BackendProtocol)
    assert not isinstance(fallback_noncallable_method, module.BackendProtocol)
    assert not isinstance(object(), module.BackendProtocol)


def test_backend_protocol_typing_extensions_runtime_check_sees_proxy_members(monkeypatch):
    import builtins
    import importlib.util
    from pathlib import Path

    from renormalizer.backend.proxy import BackendManager, BackendProxy

    protocol_path = Path(__file__).resolve().parents[2] / "backend" / "protocol.py"
    real_import = builtins.__import__

    def blocked_typing_protocol_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "typing" and {"Protocol", "runtime_checkable"} & set(fromlist or ()):
            raise ImportError("blocked typing.Protocol")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_typing_protocol_import)

    spec = importlib.util.spec_from_file_location(
        "renormalizer.backend._protocol_typing_extensions_test",
        str(protocol_path),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    proxy = BackendProxy(BackendManager())

    assert module.Protocol.__module__ == "typing_extensions"
    assert isinstance(proxy, module.BackendProtocol)
    assert "random" in module.BackendProtocol.__protocol_attrs__
    assert "linalg" in module.BackendProtocol.__protocol_attrs__


def test_abstract_backend_to_backend_dispatches_array_like_inputs_to_asarray():
    from renormalizer.backend.abstract import AbstractBackend

    class StrictBackend(AbstractBackend):
        ndarray = np.ndarray

        def array(self, *args, **kwargs):
            return np.array(*args, **kwargs)

        def asarray(self, *args, **kwargs):
            return np.asarray(*args, **kwargs)

        def from_numpy(self, x):
            if not isinstance(x, np.ndarray):
                raise TypeError("from_numpy requires a NumPy array")
            return np.asarray(x)

        def numpy(self, x):
            return np.asarray(x)

    backend = StrictBackend()

    assert backend.to_backend(np.array([1.0, 2.0])).tolist() == [1.0, 2.0]
    assert backend.to_backend([1.0, 2.0]).tolist() == [1.0, 2.0]


def test_cupy_backend_declares_gpu_and_array_type_metadata():
    try:
        import cupy
    except ImportError as exc:
        pytest.skip("could not import 'cupy': {0}".format(exc))
    except OSError as exc:
        pytest.skip("cupy is installed but failed to load: {0}".format(exc))
    from renormalizer.backend.cupy_backend import CupyBackend

    backend = CupyBackend()

    assert backend.supports_gpu is True
    assert backend.host_array_types == (np.ndarray,)
    assert backend.device_array_types
    assert np.ndarray in backend.ndarray
    assert cupy.ndarray in backend.ndarray


def test_backend_factory_aliases_and_discovery():
    from renormalizer.backend.factory import (
        available_backends,
        create_backend,
        is_backend_available,
        normalize_backend_name,
    )

    assert normalize_backend_name(None) == "numpy"
    assert normalize_backend_name("np") == "numpy"
    assert normalize_backend_name("cp") == "cupy"
    assert normalize_backend_name("jnp") == "jax"
    assert normalize_backend_name("cupynumeric") == "cupynumeric"
    assert normalize_backend_name("cunumeric") == "cupynumeric"
    assert normalize_backend_name("torch") == "torch"
    assert normalize_backend_name("pytorch") == "torch"

    discovered = available_backends()
    assert discovered["numpy"] is True
    assert is_backend_available("numpy") is True
    assert create_backend("numpy").name == "numpy"


def test_create_backend_reports_missing_optional_backend(monkeypatch):
    from renormalizer.backend import factory

    def fake_find_spec(name):
        if name == "cupynumeric":
            return None
        return object()

    monkeypatch.setattr(factory.importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(ImportError, match="cupynumeric is not installed"):
        factory.create_backend("cupynumeric")


def test_backend_discovery_requires_local_adapter_for_task4_backends(monkeypatch):
    from renormalizer.backend import factory

    def fake_find_spec(name):
        if name == "torch":
            return object()
        if name == "renormalizer.backend.torch_backend":
            return None
        return object()

    monkeypatch.setattr(factory.importlib.util, "find_spec", fake_find_spec)

    assert factory.is_backend_available("torch") is False
    assert factory.available_backends()["torch"] is False
    with pytest.raises(ImportError, match="torch backend adapter is not available"):
        factory.create_backend("torch")


def test_backend_discovery_uses_specs_without_importing_optional_modules(monkeypatch):
    import sys

    from renormalizer.backend import factory

    optional_modules = ("cupy", "jax", "torch", "cupynumeric")
    for module_name in optional_modules:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    monkeypatch.setattr(factory.importlib.util, "find_spec", lambda name: object())

    factory.available_backends()

    for module_name in optional_modules:
        assert module_name not in sys.modules


def test_backend_discovery_helpers_are_exported_from_backend_and_legacy_facade():
    from renormalizer.backend import SUPPORTED_BACKENDS, available_backends, is_backend_available
    from renormalizer.mps import backend as legacy

    assert SUPPORTED_BACKENDS == ("numpy", "cupy", "jax", "cupynumeric", "torch")
    assert legacy.SUPPORTED_BACKENDS == SUPPORTED_BACKENDS
    assert available_backends()["numpy"] is True
    assert is_backend_available("numpy") is True
    assert legacy.available_backends()["numpy"] is True
    assert legacy.is_backend_available("numpy") is True


def test_backend_discovery_helpers_are_available_from_public_facades():
    import renormalizer as r
    from renormalizer.mps import backend as legacy

    assert r.SUPPORTED_BACKENDS == ("numpy", "cupy", "jax", "cupynumeric", "torch")
    assert r.backend.SUPPORTED_BACKENDS == r.SUPPORTED_BACKENDS
    assert r.available_backends()["numpy"] is True
    assert r.is_backend_available("numpy") is True
    assert legacy.available_backends()["numpy"] is True
    assert legacy.is_backend_available("numpy") is True
    assert r.backend.__array_namespace__() is r.backend.array_namespace


def test_matrix_contract_helpers_follow_backend_conversion_boundary():
    from renormalizer.mps.matrix import Matrix, asnumpy, asxp, multi_tensor_contract, tensordot

    left = Matrix(np.arange(6.0).reshape(1, 2, 3))
    right = Matrix(np.arange(12.0).reshape(3, 2, 2))

    left_xp = asxp(left)
    right_xp = asxp(right)
    td = tensordot(left, right, axes=([-1], [0]))
    path = [([0, 1], "abc, cde -> abde")]
    contracted = multi_tensor_contract(path, left, right)

    assert isinstance(asnumpy(left), np.ndarray)
    assert asnumpy(left_xp).shape == left.shape
    assert asnumpy(right_xp).shape == right.shape
    assert np.allclose(asnumpy(td), np.tensordot(left.array, right.array, axes=([-1], [0])))
    assert np.allclose(asnumpy(contracted), np.einsum("abc,cde->abde", left.array, right.array))


def test_cupy_backend_compute_boundaries_when_available():
    try:
        import cupy
    except ImportError as exc:
        pytest.skip("could not import 'cupy': {0}".format(exc))
    except OSError as exc:
        pytest.skip("cupy is installed but failed to load: {0}".format(exc))

    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip("CuPy is installed but CUDA is not available: {0}".format(exc))
    if device_count == 0:
        pytest.skip("CuPy is installed but no CUDA device is available")

    import renormalizer as r
    from renormalizer.mps.matrix import Matrix, asnumpy, multi_tensor_contract, tensordot

    old_backend = r.backend.name
    try:
        r.set_backend("cupy")
        left = Matrix(np.arange(6.0).reshape(1, 2, 3))
        right = Matrix(np.arange(12.0).reshape(3, 2, 2))
        td = tensordot(left, right, axes=([-1], [0]))
        path = [([0, 1], "abc, cde -> abde")]
        contracted = multi_tensor_contract(path, left, right)

        assert r.backend.name == "cupy"
        assert isinstance(left.array, np.ndarray)
        assert isinstance(right.array, np.ndarray)
        assert r.backend.is_array(td)
        assert r.backend.is_device_array(td)
        assert r.backend.is_array(contracted)
        assert r.backend.is_device_array(contracted)
        assert np.allclose(asnumpy(td), np.tensordot(left.array, right.array, axes=([-1], [0])))
        assert np.allclose(asnumpy(contracted), np.einsum("abc,cde->abde", left.array, right.array))
    finally:
        r.set_backend(old_backend)
