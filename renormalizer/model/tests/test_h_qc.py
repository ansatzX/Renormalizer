import numpy as np

from renormalizer import Model, Mpo
from renormalizer.model import h_qc
from renormalizer.model.op import Op


def test_direct_jw_terms_match_legacy_simplification():
    for conserve_qn in [True, False]:
        for norbs in range(1, 6):
            a_ops, a_dag_ops = h_qc.generate_ladder_operator(norbs)
            for p in range(norbs):
                for q in range(norbs):
                    old = h_qc.simplify_op(a_dag_ops[p] * a_ops[q], norbs, conserve_qn)
                    new = h_qc._one_electron_op(p, q, conserve_qn)
                    assert old == new

            for p in range(norbs):
                for q in range(norbs):
                    for r in range(norbs):
                        for s in range(norbs):
                            old = h_qc.simplify_op(
                                Op.product([a_dag_ops[p], a_dag_ops[q], a_ops[r], a_ops[s]]),
                                norbs,
                                conserve_qn,
                            )
                            new = h_qc._two_electron_op(p, q, r, s, conserve_qn)
                            assert old == new


def test_qc_model_matches_legacy_generation():
    norbs = 6
    h1e = np.zeros((norbs, norbs))
    h2e = np.zeros((norbs, norbs, norbs, norbs))
    h1e[0, 0] = 1.1
    h1e[1, 4] = -0.3
    h1e[4, 1] = -0.3
    h2e[0, 1, 2, 3] = 0.7
    h2e[1, 4, 0, 5] = -0.2
    h2e[2, 5, 1, 4] = 0.4

    a_ops, a_dag_ops = h_qc.generate_ladder_operator(norbs)
    old_terms = []
    for p, q in np.argwhere(h1e != 0):
        op = h_qc.simplify_op(a_dag_ops[p] * a_ops[q], norbs)
        old_terms.append(op * h1e[p, q])
    for p, q, r, s in np.argwhere(h2e != 0):
        op = h_qc.simplify_op(Op.product([a_dag_ops[p], a_dag_ops[q], a_ops[r], a_ops[s]]), norbs)
        old_terms.append(op * h2e[p, q, r, s])

    _, new_terms = h_qc.qc_model(h1e, h2e)
    assert old_terms == new_terms


def test_spatial_orbital_basis_matches_spin_orbital_dense_hamiltonian():
    norbs = 4
    h1e = np.zeros((norbs, norbs))
    h2e = np.zeros((norbs, norbs, norbs, norbs))
    h1e[0, 0] = 0.8
    h1e[1, 1] = 0.7
    h1e[2, 2] = -0.2
    h1e[3, 3] = -0.1
    h1e[0, 2] = h1e[2, 0] = 0.05
    h1e[1, 3] = h1e[3, 1] = -0.04
    h2e[0, 1, 2, 3] = 0.11
    h2e[1, 2, 0, 3] = -0.07

    spin_basis, spin_terms = h_qc.qc_model(h1e, h2e)
    spatial_basis, spatial_terms = h_qc.qc_model(h1e, h2e, spatial_orbital=True)

    spin_dense = Mpo(Model(spin_basis, spin_terms)).todense()
    spatial_dense = Mpo(Model(spatial_basis, spatial_terms)).todense()

    n_spatial = norbs // 2
    spin_dims = [2] * norbs
    spatial_dims = [4] * n_spatial
    perm = np.empty(2**norbs, dtype=int)
    for spin_idx, occ in enumerate(np.ndindex(*spin_dims)):
        spatial_digits = [occ[2 * i] + 2 * occ[2 * i + 1] for i in range(n_spatial)]
        perm[spin_idx] = np.ravel_multi_index(tuple(spatial_digits), spatial_dims)

    reordered = spin_dense[np.ix_(np.argsort(perm), np.argsort(perm))]
    np.testing.assert_allclose(reordered, spatial_dense, atol=1e-12, rtol=1e-12)
