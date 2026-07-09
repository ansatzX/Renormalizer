# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import os
import random
import pickle
import json

import numpy as np
import pytest

from renormalizer.model import Mol, Phonon, HolsteinModel, Model, Op
from renormalizer.model.basis import BasisHalfSpin
from renormalizer.mps import Mpo, Mps
from renormalizer.mps.tests import cur_dir
from renormalizer.tests.parameter import custom_model, holstein_model
from renormalizer.utils import Quantity
from renormalizer.utils.qutip_utils import get_spin_hamiltonian


def _jsonl_payloads(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize("nsites", [5, 10])
# More sites make MPO representation not efficient
# Not good for testing
@pytest.mark.parametrize("nterms", [100, 1000])
@pytest.mark.parametrize("algo", ["Hopcroft-Karp", "qr"])
def test_symbolic_mpo(nsites, nterms, algo):

    possible_operators = [
        "sigma_+",
        "sigma_-",
        "sigma_z"
    ]
    ham_terms = []
    for i in range(nterms):
        op_list = [Op(random.choice(possible_operators), j) for j in range(nsites)]
        ham_terms.append(Op.product(op_list) * random.random())
    basis = [BasisHalfSpin(i) for i in range(nsites)]
    model = Model(basis, ham_terms)
    mpo = Mpo(model, algo=algo)
    dense_mpo = mpo.todense()
    qutip_ham = get_spin_hamiltonian(ham_terms)
    assert np.allclose(dense_mpo, qutip_ham.full())


@pytest.mark.parametrize("algo", ["qr", "Hopcroft-Karp"])
def test_swap_symbolic_mpo(algo):
    if algo == "qr":
        # not efficient due to more terms in the table
        # so use smaller system
        nsites = 5
        nterms = 100
    else:
        nsites = 10
        nterms = 1000

    possible_operators = [
        "sigma_+",
        "sigma_-",
        "sigma_z"
    ]
    ham_terms = []
    for i in range(nterms):
        op_list = [Op(random.choice(possible_operators), j) for j in range(nsites)]
        ham_terms.append(Op.product(op_list) * random.random())
    basis = [BasisHalfSpin(i) for i in range(nsites)]
    model = Model(basis, ham_terms)
    mpo = Mpo(model, algo=algo)
    for i in range(20):
        isite1 = max(int(random.random() * nsites) - 1, 0)
        isite2 = isite1 + 1
        basis = basis.copy()
        basis[isite1], basis[isite2] = basis[isite2], basis[isite1]
        new_model = Model(basis, ham_terms)
        mpo.try_swap_site(new_model, False, algo=algo)
        ref_mpo = Mpo(new_model, algo=algo)
        mpo_dense = mpo.todense()
        ref_dense = ref_mpo.todense()
        assert np.allclose(mpo_dense, ref_dense)


@pytest.mark.parametrize("dt, space, shift", ([30, "GS", 0.0], [30, "EX", 0.0]))
def test_exact_propagator(dt, space, shift):
    prop_mpo = Mpo.exact_propagator(holstein_model, -1.0j * dt, space, shift)
    with open(os.path.join(cur_dir, "test_exact_propagator.pickle"), "rb") as fin:
        std_dict = pickle.load(fin)
    std_mpo = std_dict[space]
    assert prop_mpo == std_mpo


@pytest.mark.parametrize("scheme", (1, 4))
def test_offset(scheme):
    ph = Phonon.simple_phonon(Quantity(3.33), Quantity(1), 2)
    m = Mol(Quantity(0), [ph] * 2)
    mlist = HolsteinModel([m] * 2, Quantity(17), )
    mpo1 = Mpo(mlist)
    assert mpo1.is_hermitian()
    f1 = mpo1.todense()
    evals1, _ = np.linalg.eigh(f1)
    offset = Quantity(0.123)
    mpo2 = Mpo(mlist, offset=offset)
    f2 = mpo2.todense()
    evals2, _ = np.linalg.eigh(f2)
    assert np.allclose(evals1 - offset.as_au(), evals2)


def test_identity():
    identity = Mpo.identity(holstein_model)
    mps = Mps.random(holstein_model, qntot=1, m_max=5)
    assert mps.expectation(identity) == pytest.approx(mps.mp_norm) == pytest.approx(1)


def test_mpo_apply_mps_profiles_grouped_gemm_for_same_shape_sites(caplog, tmp_path):
    from renormalizer.utils import profiling
    from renormalizer.utils.log import PROFILING

    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    try:
        model = custom_model(n_phys_dim=(2, 2))
        identity = Mpo.identity(model)
        mps = Mps.ground_state(model, max_entangled=False)

        applied = identity.apply(mps)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    assert applied.distance(mps) == pytest.approx(0.0, abs=1e-12)
    grouped_events = [
        payload
        for payload in _jsonl_payloads(event_path)
        if payload.get("event") == "grouped_gemm_execute"
    ]
    assert grouped_events
    assert max(event["num_tasks"] for event in grouped_events) >= model.nsite


def test_mpo_apply_mps_uses_grouped_loop_for_repeated_small_site_gemv(monkeypatch):
    import renormalizer.mps.mpo as mpo_module

    model = custom_model(n_phys_dim=(2, 2))
    identity = Mpo.identity(model)
    mps = Mps.ground_state(model, max_entangled=False)
    calls = []
    backend_impl = mpo_module.backend.current
    original_grouped_gemm = backend_impl.grouped_gemm

    def counting_grouped_gemm(tasks, *args, **kwargs):
        tasks = list(tasks)
        calls.append(len(tasks))
        return original_grouped_gemm(tasks, *args, **kwargs)

    monkeypatch.setattr(mpo_module.profiling, "should_record_op", lambda: True)
    monkeypatch.setattr(backend_impl, "grouped_gemm", counting_grouped_gemm)

    applied = identity.apply(mps)

    assert applied.distance(mps) == pytest.approx(0.0, abs=1e-12)
    assert calls == [model.nsite]


def test_mpo_apply_mps_skips_grouped_for_tiny_non_profiled_workload(monkeypatch):
    import renormalizer.mps.mpo as mpo_module

    model = custom_model(n_phys_dim=(2, 2))
    identity = Mpo.identity(model)
    mps = Mps.ground_state(model, max_entangled=False)
    backend_impl = mpo_module.backend.current

    def forbidden_grouped_gemm(*args, **kwargs):
        raise AssertionError("tiny non-profiled MPO.apply should avoid grouped_gemm setup")

    monkeypatch.setattr(mpo_module.profiling, "should_record_op", lambda: False)
    monkeypatch.setattr(backend_impl, "grouped_gemm", forbidden_grouped_gemm)

    applied = identity.apply(mps)

    assert applied.distance(mps) == pytest.approx(0.0, abs=1e-12)


def test_mpo_apply_mps_does_not_precompute_grouped_stats_before_dispatch(monkeypatch):
    import renormalizer.mps.mpo as mpo_module

    model = custom_model(n_phys_dim=(2, 2))
    identity = Mpo.identity(model)
    mps = Mps.ground_state(model, max_entangled=False)
    calls = []
    backend_impl = mpo_module.backend.current
    original_grouped_gemm = backend_impl.grouped_gemm

    def counting_grouped_gemm(tasks, *args, **kwargs):
        tasks = list(tasks)
        calls.append(len(tasks))
        return original_grouped_gemm(tasks, *args, **kwargs)

    def stats_must_not_run(*args, **kwargs):
        raise AssertionError("Mpo.apply should not run grouped_gemm_stats before backend dispatch")

    monkeypatch.setattr(backend_impl, "grouped_gemm", counting_grouped_gemm)
    monkeypatch.setattr(mpo_module, "grouped_gemm_stats", stats_must_not_run, raising=False)
    monkeypatch.setattr(mpo_module.profiling, "should_record_op", lambda: True)

    applied = identity.apply(mps)

    assert applied.distance(mps) == pytest.approx(0.0, abs=1e-12)
    assert calls == [model.nsite]


def test_mpo_apply_mps_does_not_copy_overwritten_input_tensors(monkeypatch):
    model = custom_model(n_phys_dim=(2, 2))
    identity = Mpo.identity(model)
    mps = Mps.ground_state(model, max_entangled=False)

    def copy_must_not_run(self):
        raise AssertionError("Mpo.apply should metadata-copy before overwriting tensors")

    monkeypatch.setattr(Mps, "copy", copy_must_not_run)

    applied = identity.apply(mps)

    assert applied.distance(mps) == pytest.approx(0.0, abs=1e-12)


def test_mpo_contract_skips_redundant_canonical_check(monkeypatch):
    import renormalizer.mps.mp as mp_module

    model = custom_model(n_phys_dim=(2, 2))
    identity = Mpo.identity(model)
    mps = Mps.ground_state(model, max_entangled=False)

    def forbidden_check(*args, **kwargs):
        raise AssertionError("Mpo.contract should not re-check immediately after canonicalise")

    monkeypatch.setattr(mp_module.MatrixProduct, "check_right_canonical", forbidden_check)

    contracted = identity.contract(mps)

    assert contracted.distance(mps) == pytest.approx(0.0, abs=1e-12)


def test_scheme4():
    ph = Phonon.simple_phonon(Quantity(3.33), Quantity(1), 2)
    m1 = Mol(Quantity(0), [ph])
    m2 = Mol(Quantity(0), [ph]*2)
    model4 = HolsteinModel([m1, m2], Quantity(17), 4)
    model3 = HolsteinModel([m1, m2], Quantity(17), 3)
    mpo4 = Mpo(model4)
    assert mpo4.is_hermitian()
    # for debugging
    f = mpo4.todense()
    mpo3 = Mpo(model3)
    assert mpo3.is_hermitian()
    # makeup two states
    mps4 = Mps()
    mps4.model = model4
    mps4.append(np.array([1, 0]).reshape((1,2,1)))
    mps4.append(np.array([0, 0, 1]).reshape((1,-1,1)))
    mps4.append(np.array([0.707, 0.707]).reshape((1,2,1)))
    mps4.append(np.array([1, 0]).reshape((1,2,1)))
    mps4.build_empty_qn()
    e4 = mps4.expectation(mpo4)
    mps3 = Mps()
    mps3.model = model3
    mps3.append(np.array([1, 0]).reshape((1,2,1)))
    mps3.append(np.array([1, 0]).reshape((1,2,1)))
    mps3.append(np.array([0, 1]).reshape((1,2,1)))
    mps3.append(np.array([0.707, 0.707]).reshape((1,2,1)))
    mps3.append(np.array([1, 0]).reshape((1,2,1)))
    mps3.build_empty_qn()
    e3 = mps3.expectation(mpo3)
    assert pytest.approx(e4) == e3


@pytest.mark.parametrize("scheme", (1, 4))
def test_intersite(scheme):

    local_mlist = holstein_model.switch_scheme(scheme)

    mpo1 = Mpo.intersite(local_mlist, {0:r"a^\dagger"}, {}, Quantity(1.0))
    mpo2 = Mpo.onsite(local_mlist, r"a^\dagger", dof_set=[0])
    assert mpo1.distance(mpo2) == pytest.approx(0, abs=1e-5)

    mpo3 = Mpo.intersite(local_mlist, {2:r"a^\dagger a"}, {}, Quantity(1.0))
    mpo4 = Mpo.onsite(local_mlist, r"a^\dagger a", dof_set=[2])
    assert mpo3.distance(mpo4) == pytest.approx(0, abs=1e-5)

    mpo5 = Mpo.intersite(local_mlist, {2:r"a^\dagger a"}, {}, Quantity(0.5))
    assert mpo5.add(mpo5).distance(mpo4) == pytest.approx(0, abs=1e-5)

    mpo6 = Mpo.intersite(local_mlist, {0:r"a^\dagger",2:"a"}, {}, Quantity(1.0))
    mpo7 = Mpo.onsite(local_mlist, "a", dof_set=[2])
    assert mpo2.apply(mpo7).distance(mpo6) == pytest.approx(0, abs=1e-5)

    mpo8 = Mpo.intersite(local_mlist, {0: r"a^\dagger", 2: "a"}, {},
                         Quantity(local_mlist.j_matrix[0,2]))
    mpo9 = Mpo.intersite(local_mlist, {2:r"a^\dagger",0:"a"}, {},
            Quantity(local_mlist.j_matrix[0,2]))

    assert mpo9.conj_trans().distance(mpo8) == pytest.approx(0, abs=1e-6)

    ph_mpo1 = Mpo.ph_onsite(local_mlist, "b", 1, 1)
    ph_mpo2 = Mpo.intersite(local_mlist, {}, {(1,1):"b"})
    assert ph_mpo1.distance(ph_mpo2) == pytest.approx(0, abs=1e-6)


def test_phonon_onsite():
    gs = Mps.ground_state(holstein_model, max_entangled=False)
    assert not gs.ph_occupations.any()
    b2 = Mpo.ph_onsite(holstein_model, r"b^\dagger", 0, 0)
    p1 = b2.apply(gs).normalize("mps_only")
    assert np.allclose(p1.ph_occupations, [1, 0, 0, 0, 0, 0])
    p2 = b2.apply(p1).normalize("mps_only")
    assert np.allclose(p2.ph_occupations, [2, 0, 0, 0, 0, 0])
    b = b2.conj_trans()
    assert b.distance(Mpo.ph_onsite(holstein_model, r"b", 0, 0)) == 0
    assert b.apply(p2).normalize("mps_only").distance(p1) == pytest.approx(0, abs=1e-5)
