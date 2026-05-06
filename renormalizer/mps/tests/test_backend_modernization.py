# -*- coding: utf-8 -*-

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
