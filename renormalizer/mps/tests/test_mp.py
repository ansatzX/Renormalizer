# -*- coding: utf-8 -*-

import os

import numpy as np
import pytest

import renormalizer.mps.mp as mp_module
import renormalizer.mps.mps as mps_module
from renormalizer.mps import Mps, Mpo, MpDm
from renormalizer.mps.matrix import Matrix, tensordot, asnumpy
from renormalizer.mps.lib import Environ, compressed_sum
from renormalizer.sbm import param2mollist
from renormalizer.tests.parameter import custom_model, holstein_model
from renormalizer.utils import CompressCriteria, Quantity

def test_save_load():
    model = holstein_model
    mps = Mpo.onsite(model, r"a^\dagger", dof_set={0}) @ Mps.ground_state(model, False)
    mpo = Mpo(model)
    mps1 = mps.copy()
    for i in range(2):
        mps1 = mps1.evolve(mpo, 10)
    mps2 = mps.evolve(mpo, 10)
    fname = "test.npz"
    mps2.dump(fname)
    mps2 = Mps.load(model, fname)
    mps2 = mps2.evolve(mpo, 10)
    assert np.allclose(mps1.e_occupations, mps2.e_occupations)
    os.remove(fname)


def check_distance(a: Mps, b: Mps):
    d1 = (a - b).mp_norm
    d2 = a.distance(b)
    a_array = a.todense()
    b_array = b.todense()
    d3 = np.linalg.norm(a_array - b_array)
    assert d1 == pytest.approx(d2) == pytest.approx(d3)


def test_distance():
    model = custom_model(n_phys_dim=(2, 2))
    a = Mps.random(model, 1, 10)
    b = Mps.random(model, 1, 10)
    check_distance(a, b)
    h = Mpo(model)
    for i in range(100):
        a = a.evolve(h, 10)
        b = b.evolve(h, 10)
        check_distance(a, b)


def test_mps_dot_uses_fast_path_without_generic_tensordot(monkeypatch):
    model = custom_model(n_phys_dim=(2, 2))
    a = Mps.random(model, 1, 10)
    b = Mps.random(model, 1, 10)
    expected = np.vdot(a.todense(), b.todense())

    def forbidden_tensordot(*args, **kwargs):
        raise AssertionError("MPS dot should not call generic tensordot")

    monkeypatch.setattr(mp_module, "tensordot", forbidden_tensordot)
    assert a.conj().dot(b) == pytest.approx(expected)


def test_mps_canonicalise_absorbs_center_without_generic_tensordot(monkeypatch):
    model = custom_model(n_phys_dim=(2, 2))
    mps = Mps.random(model, 1, 10)
    expected = mps.todense()

    def forbidden_tensordot(*args, **kwargs):
        raise AssertionError("MPS canonicalise should not call generic tensordot")

    monkeypatch.setattr(mp_module, "tensordot", forbidden_tensordot)
    mps.canonicalise()
    assert np.allclose(mps.todense(), expected)


def test_matrix_pdim_prod_uses_shape_tuple_product(monkeypatch):
    matrix = Matrix(np.zeros((2, 3, 5, 7)))

    def forbidden_numpy_prod(*args, **kwargs):
        raise AssertionError("Matrix.pdim_prod should avoid NumPy reducer overhead")

    monkeypatch.setattr(mps_module.np, "prod", forbidden_numpy_prod)

    assert matrix.pdim_prod == 15


def _small_no_qn_sbm_mps():
    model = param2mollist(0.05, Quantity(1), Quantity(20), 1, 4)
    return Mpo(model).apply(Mps.ground_state(model, max_entangled=False))


def test_no_qn_mps_canonicalise_uses_dense_decomposition(monkeypatch):
    mps = _small_no_qn_sbm_mps()
    expected = mps.todense()

    def forbidden_get_big_qn(*args, **kwargs):
        raise AssertionError("no-QN canonicalise should not construct QN blocks")

    def forbidden_svd_qn(*args, **kwargs):
        raise AssertionError("no-QN canonicalise should call dense QR directly")

    monkeypatch.setattr(mp_module.MatrixProduct, "_get_big_qn", forbidden_get_big_qn)
    monkeypatch.setattr(mp_module.svd_qn, "svd_qn", forbidden_svd_qn)

    mps.canonicalise()

    assert np.allclose(mps.todense(), expected)


def test_no_qn_mps_compress_uses_dense_decomposition(monkeypatch):
    mps = _small_no_qn_sbm_mps()
    mps.canonicalise()
    mps.compress_config.threshold = 1e-12
    expected = mps.todense()

    def forbidden_get_big_qn(*args, **kwargs):
        raise AssertionError("no-QN compress should not construct QN blocks")

    def forbidden_svd_qn(*args, **kwargs):
        raise AssertionError("no-QN compress should call dense SVD directly")

    monkeypatch.setattr(mp_module.MatrixProduct, "_get_big_qn", forbidden_get_big_qn)
    monkeypatch.setattr(mp_module.svd_qn, "svd_qn", forbidden_svd_qn)

    mps.compress()

    assert np.allclose(mps.todense(), expected, atol=1e-10)


def test_no_qn_dense_svd_uses_numpy_for_small_matrices(monkeypatch):
    mps = _small_no_qn_sbm_mps()
    mps.canonicalise()
    mps.compress_config.threshold = 1e-12
    expected = mps.todense()

    def forbidden_optimized_svd(*args, **kwargs):
        raise AssertionError("small no-QN dense SVD should use numpy.linalg.svd")

    monkeypatch.setattr(mp_module.svd_qn, "optimized_svd", forbidden_optimized_svd)

    mps.compress()

    assert np.allclose(mps.todense(), expected, atol=1e-10)


def test_no_qn_dense_qr_uses_numpy_for_to_right_qr(monkeypatch):
    mps = _small_no_qn_sbm_mps()
    zero_qn = mps._single_zero_qn_center([0])
    mps.to_right = True

    def forbidden_scipy_qr(*args, **kwargs):
        raise AssertionError("small no-QN QR should use numpy.linalg.qr")

    monkeypatch.setattr(mp_module.svd_qn.scipy.linalg, "qr", forbidden_scipy_qr)

    u, vt, qnlset, qnrset = mps._dense_qr_no_qn(0, zero_qn)

    assert u.shape[1] == vt.shape[0] == len(qnlset) == len(qnrset)


def test_no_qn_dense_rq_uses_numpy_transposed_qr(monkeypatch):
    mps = _small_no_qn_sbm_mps()
    zero_qn = mps._single_zero_qn_center([1])
    mps.to_right = False
    mt = mps[1]
    coef_matrix = mt.array.reshape((mt.shape[0], mt.pdim_prod * mt.shape[-1]))

    def forbidden_scipy_rq(*args, **kwargs):
        raise AssertionError("small no-QN RQ should use transposed numpy.linalg.qr")

    monkeypatch.setattr(mp_module.svd_qn.scipy.linalg, "rq", forbidden_scipy_rq)

    u, vt, qnlset, qnrset = mps._dense_qr_no_qn(1, zero_qn)

    assert np.allclose(u @ vt, coef_matrix)
    assert np.allclose(vt @ vt.T.conj(), np.eye(vt.shape[0]))
    assert u.shape[1] == vt.shape[0] == len(qnlset) == len(qnrset)


def test_no_qn_fast_path_reuses_zero_qn_metadata(monkeypatch):
    mps = _small_no_qn_sbm_mps()
    zero_qn = mps._single_zero_qn_center([0])
    assert zero_qn is not None

    def forbidden_get_sigmaqn(*args, **kwargs):
        raise AssertionError("cached no-QN fast path should not re-read sigma qn")

    monkeypatch.setattr(mps.__class__, "_get_sigmaqn", forbidden_get_sigmaqn)
    assert mps._single_zero_qn_center([1]) == zero_qn


def test_no_qn_array_assignment_uses_fast_matrix_wrapper(monkeypatch):
    mps = _small_no_qn_sbm_mps()
    assert mps._single_zero_qn_center([0]) is not None
    array = mps[0].array.copy()

    def forbidden_matrix_init(*args, **kwargs):
        raise AssertionError("no-QN ndarray assignment should bypass Matrix.__init__")

    def forbidden_get_sigmaqn(*args, **kwargs):
        raise AssertionError("fast no-QN assignment should reuse cached sigma qn")

    monkeypatch.setattr(mp_module.Matrix, "__init__", forbidden_matrix_init)
    monkeypatch.setattr(mps.__class__, "_get_sigmaqn", forbidden_get_sigmaqn)
    mps[0] = array

    assert np.allclose(mps[0].array, array)


def test_no_qn_array_assignment_bypasses_generic_array2mt(monkeypatch):
    mps = _small_no_qn_sbm_mps()
    assert mps._single_zero_qn_center([0]) is not None
    array = mps[0].array.copy()

    def forbidden_array2mt(*args, **kwargs):
        raise AssertionError("no-QN ndarray setitem should bypass generic _array2mt")

    monkeypatch.setattr(mp_module.MatrixProduct, "_array2mt", forbidden_array2mt)

    mps[0] = array

    assert np.allclose(mps[0].array, array)


def test_no_qn_canonicalise_updates_tensors_without_setitem(monkeypatch):
    mps = _small_no_qn_sbm_mps()
    expected = mps.todense()

    def forbidden_setitem(*args, **kwargs):
        raise AssertionError("no-QN canonicalise should update tensors through the direct fast path")

    monkeypatch.setattr(mp_module.MatrixProduct, "__setitem__", forbidden_setitem)

    mps.canonicalise()

    assert np.allclose(mps.todense(), expected)


def test_calc_1site_rdm_single_idx_uses_transfer_without_environment(monkeypatch):
    model = custom_model(n_phys_dim=(2, 2, 2))
    mps = Mps.random(model, 1, 10)
    dense = mps.todense().reshape(mps.pbond_list)
    trace_axes = tuple(axis for axis in range(dense.ndim) if axis != 1)
    expected = np.tensordot(
        dense.conj(),
        dense,
        axes=(trace_axes, trace_axes),
    )

    def forbidden_environ(*args, **kwargs):
        raise AssertionError("single-site RDM should not build a full environment")

    monkeypatch.setattr(mps_module.Environ, "__init__", forbidden_environ)
    rdm = mps.calc_1site_rdm(idx=1)

    assert set(rdm) == {1}
    assert np.allclose(rdm[1], expected)


def test_calc_1site_rdm_transfer_avoids_planned_backend_einsum(monkeypatch):
    model = custom_model(n_phys_dim=(2, 2, 2))
    mps = Mps.random(model, 1, 10)

    def forbidden_einsum(self, *args, **kwargs):
        raise AssertionError("single-site transfer RDM should use native fixed contractions")

    monkeypatch.setattr(type(mps_module.backend.current), "einsum", forbidden_einsum)

    rdm = mps.calc_1site_rdm(idx=1)

    assert set(rdm) == {1}


def test_calc_1site_rdm_boundary_uses_canonical_local_tensor(monkeypatch):
    mps = _small_no_qn_sbm_mps().ensure_right_canonical()
    dense = mps.todense().reshape(mps.pbond_list)
    expected = np.tensordot(
        dense.conj(),
        dense,
        axes=(tuple(range(1, dense.ndim)), tuple(range(1, dense.ndim))),
    )

    def forbidden_environ(*args, **kwargs):
        raise AssertionError("boundary canonical RDM should not build a full environment")

    monkeypatch.setattr(mps_module.Environ, "__init__", forbidden_environ)
    rdm = mps.calc_1site_rdm(idx=0)

    assert set(rdm) == {0}
    assert np.allclose(rdm[0], expected)


def test_scaled_mps_for_sum_avoids_full_copy(monkeypatch):
    mps = _small_no_qn_sbm_mps()
    original = mps.todense()

    def forbidden_copy(*args, **kwargs):
        raise AssertionError("temporary Taylor scaling should not deep-copy the full MPS")

    monkeypatch.setattr(mp_module.MatrixProduct, "copy", forbidden_copy)

    scaled = mps_module._scaled_mps_for_sum(mps, 2.0)

    assert np.allclose(mps.todense(), original)
    assert np.allclose(scaled.todense(), 2.0 * original)
    assert scaled[scaled.qnidx] is not mps[mps.qnidx]


def test_compressed_sum_skips_redundant_canonical_check(monkeypatch):
    left = _small_no_qn_sbm_mps()
    right = _small_no_qn_sbm_mps()
    left.compress_config.threshold = 1e-14
    right.compress_config.threshold = 1e-14
    expected = left.todense() + right.todense()

    def forbidden_check(*args, **kwargs):
        raise AssertionError("compressed_sum should not re-check immediately after canonicalise")

    monkeypatch.setattr(mp_module.MatrixProduct, "check_right_canonical", forbidden_check)

    result = compressed_sum([left, right])

    assert np.allclose(result.todense(), expected, atol=1e-10)


def test_environ():
    mps = Mps.random(holstein_model, 1, 10)
    mpo = Mpo(holstein_model)
    mps = mps.evolve(mpo, 10)
    environ = Environ(mps, mpo)
    for i in range(len(mps)-1):
        l = environ.read("L", i)
        r = environ.read("R", i+1)
        e = complex(tensordot(l, r, axes=((0, 1, 2), (0, 1, 2)))).real
        assert pytest.approx(e) == mps.expectation(mpo)

# multi_mpo routine for single mpo calculation
@pytest.mark.parametrize("mpdm", (True, False))
def test_environ_multi_mpo(mpdm):
    mps = Mps.random(holstein_model, 1, 10)
    if mpdm:
        mps = MpDm.from_mps(mps)
    mpo = Mpo(holstein_model)
    mps = mps.evolve(mpo, 10)
    environ = Environ(mps, mpo)
    environ_multi_mpo = Environ(mps, [mpo])
    for i in range(len(mps)-1):
        l = environ.read("L", i)
        r = environ.read("R", i+1)
        l2 = environ_multi_mpo.read("L", i)
        r2 = environ_multi_mpo.read("R", i+1) 
        assert np.allclose(asnumpy(l), asnumpy(l2))
        assert np.allclose(asnumpy(r), asnumpy(r2))

@pytest.mark.parametrize("comp", (True, False))
@pytest.mark.parametrize("mp", (
        "mps",
        "mpdm",
        "mpo",
))
def test_svd_compress(comp, mp):
    
    if mp == "mpo":
        mps = Mpo(holstein_model)
        M = 22
    else:
        mps = Mps.random(holstein_model, 1, 10)
        if mp == "mpdm":
            mps = MpDm.from_mps(mps)
        mps.canonicalise().normalize("mps_only")
        M = 36
    if comp:
        mps = mps.to_complex(inplace=True)
    print(f"{mps}")
    
    mpo = Mpo(holstein_model)
    if comp:
        mpo = mpo.scale(-1.0j)
    print(f"{mpo.bond_dims}")
    
    std_mps = mpo.apply(mps, canonicalise=True).canonicalise()
    print(f"std_mps: {std_mps}")
    mps.compress_config.bond_dim_max_value = M
    mps.compress_config.criteria = CompressCriteria.fixed
    svd_mps = mpo.contract(mps)
    dis = svd_mps.distance(std_mps)/std_mps.mp_norm
    print(f"svd_mps: {svd_mps}, dis: {dis}")
    assert np.allclose(dis, 0.0, atol=1e-3)
    assert np.allclose(svd_mps.mp_norm, std_mps.mp_norm, atol=1e-4)
    
    
@pytest.mark.parametrize("comp", (True, False))
@pytest.mark.parametrize("mp", ("mps", "mpdm", "mpo" ))
def test_variational_compress(comp, mp):
    
    if mp == "mpo":
        mps = Mpo(holstein_model)
        M = 20
    else:
        mps = Mps.random(holstein_model, 1, 10)
        if mp == "mpdm":
            mps = MpDm.from_mps(mps)
        mps.canonicalise().normalize("mps_only")
        M = 36
    if comp:
        mps = mps.to_complex(inplace=True)
    print(f"{mps}")
    
    mpo = Mpo(holstein_model)
    if comp:
        mpo = mpo.scale(-1.0j)
    print(f"{mpo.bond_dims}")
    
    std_mps = mpo.apply(mps, canonicalise=True).canonicalise()
    print(f"std_mps: {std_mps}")
    
    # 2site algorithm
    mps.compress_config.vprocedure = [[M,1.0],[M,0.2],[M,0.1]]+[[M,0],]*10
    mps.compress_config.vmethod = "2site"
    mps.compress_config.bond_dim_max_value = M
    mps.compress_config.criteria = CompressCriteria.fixed
    var_mps = mps.variational_compress(mpo, guess=None)
    dis = var_mps.distance(std_mps)/std_mps.mp_norm
    print(f"var2_mps: {var_mps}, dis: {dis}")
    assert np.allclose(dis, 0.0, atol=1e-4)
    assert np.allclose(var_mps.mp_norm, std_mps.mp_norm, atol=1e-4)
    
    # 1site algorithm with 2site result as a guess
    # 1site algorithm is easy to be trapped in a local minimum
    var_mps.compress_config.vprocedure = [[M,0],]*10
    var_mps.compress_config.vmethod = "1site"
    var_mps.compress_config.bond_dim_max_value = M
    var_mps.compress_config.criteria = CompressCriteria.fixed
    var_mps = mps.variational_compress(mpo, guess=var_mps)
    dis = var_mps.distance(std_mps)/std_mps.mp_norm
    print(f"var1_mps: {var_mps}, dis: {dis}")
    assert np.allclose(dis, 0.0, atol=1e-4)
    assert np.allclose(var_mps.mp_norm, std_mps.mp_norm, atol=1e-4)
