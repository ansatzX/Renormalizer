import importlib

import numpy as np

from renormalizer.mps.hop_expr import hop_expr
from renormalizer.mps.matrix import asnumpy


backend_module = importlib.import_module("renormalizer.mps.backend")


def test_legacy_backend_exports_array_namespace_and_precision():
    assert hasattr(backend_module.xp, "asarray")
    assert backend_module.backend.real_dtype in (np.float32, np.float64)
    assert backend_module.backend.complex_dtype in (np.complex64, np.complex128)


def test_single_site_hop_matches_numpy_einsum():
    rng = np.random.default_rng(7)
    left = rng.normal(size=(2, 3, 4))
    mpo = rng.normal(size=(3, 5, 4, 6))
    right = rng.normal(size=(7, 6, 8))
    center = rng.normal(size=(4, 4, 8))
    hop = hop_expr(left, right, [mpo], center.shape)
    actual = asnumpy(hop(backend_module.xp.asarray(center)))
    expected = np.einsum("abc,bdef,lfk,cek->adl", left, mpo, right, center)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
