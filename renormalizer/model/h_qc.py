# -*- coding: utf-8 -*-

import itertools
import logging

import numpy as np

from renormalizer.model.op import Op
from renormalizer.model.basis import BasisHalfSpin, BasisSpatialOrbital

logger = logging.getLogger(__name__)

def read_fcidump(fname, norb):
    """
    from fcidump format electron integral to h_pq g_pqrs in arXiv:2006.02056 eq 18
    norb: number of spatial orbitals
    return sh spin-orbital 1-e integral
           aseri: 2-e integral after considering symmetry
           nuc: nuclear repulsion energy
    """
    eri = np.zeros((norb, norb, norb, norb))
    h = np.zeros((norb,norb))

    with open(fname, "r") as f:
        a = f.readlines()
        for line, info in enumerate(a):
            if line < 4:
                continue
            s  = info.split()
            integral, p, q, r, s = float(s[0]),int(s[1]),int(s[2]),int(s[3]),int(s[4])
            if r != 0:
                eri[p-1,q-1,r-1,s-1] = integral
                eri[q-1,p-1,r-1,s-1] = integral
                eri[p-1,q-1,s-1,r-1] = integral
                eri[q-1,p-1,s-1,r-1] = integral
            elif p != 0:
                h[p-1,q-1] = integral
                h[q-1,p-1] = integral
            else:
                nuc = integral

    sh, aseri = int_to_h(h, eri)

    logger.info(f"nuclear repulsion: {nuc}")

    return sh, aseri, nuc


def int_to_h(h, eri):
    nsorb = len(h) * 2
    seri = np.zeros((nsorb, nsorb, nsorb, nsorb))
    sh = np.zeros((nsorb, nsorb))
    for p, q, r, s in itertools.product(range(nsorb), repeat=4):
        # a_p^\dagger a_q^\dagger a_r a_s
        if p % 2 == s % 2 and q % 2 == r % 2:
            seri[p, q, r, s] = eri[p // 2, s // 2, q // 2, r // 2]

    for q, s in itertools.product(range(nsorb), repeat=2):
        if q % 2 == s % 2:
            sh[q, s] = h[q // 2, s // 2]

    aseri = np.zeros((nsorb, nsorb, nsorb, nsorb))
    for q, s in itertools.product(range(nsorb), repeat=2):
        for p, r in itertools.product(range(q), range(s)):
            # aseri[p,q,r,s] = seri[p,q,r,s] - seri[q,p,r,s]
            aseri[p, q, r, s] = seri[p, q, r, s] - seri[p, q, s, r]

    return sh, aseri


def generate_ladder_operator(norbs):
    # construct electronic creation/annihilation operators by Jordan-Wigner transformation
    a_ops = []
    a_dag_ops = []
    for j in range(norbs):
        # qn for each op will be processed in `process_op`
        sigma_z_list = [Op("Z", l) for l in range(j)]
        a_ops.append(Op.product(sigma_z_list + [Op("+", j)]))
        a_dag_ops.append(Op.product(sigma_z_list + [Op("-", j)]))

    return a_ops, a_dag_ops


def simplify_op(old_op: Op, norbs: int, conserve_qn:bool=True):
    # helper function to process operators.
    # Remove "sigma_z sigma_z". Use {sigma_z, sigma_+} = 0
    # and {sigma_z, sigma_-} = 0 to simplify operators,
    # and set quantum number
    dof_to_siteidx = dict(zip(range(norbs), range(norbs)))
    if conserve_qn:
        qn_dict0 = {"+": [-1, 0], "-": [1, 0], "Z": [0, 0]}
        qn_dict1 = {"+": [0, -1], "-": [0, 1], "Z": [0, 0]}
    else:
        qn_dict0 = {"+": 0, "-": 0, "Z": 0}

    old_ops, _ = old_op.split_elementary(dof_to_siteidx)
    new_ops = []
    for elem_op in old_ops:
        # move all sigma_z to the start of the operator
        # and cancel as many as possible
        n_sigma_z = elem_op.split_symbol.count("Z")
        n_non_sigma_z = 0
        n_permute = 0
        for simple_elem_op in elem_op.split_symbol:
            if simple_elem_op != "Z":
                n_non_sigma_z += 1
            else:
                n_permute += n_non_sigma_z
        # remove as many "sigma_z" as possible
        new_symbol = [s for s in elem_op.split_symbol if s != "Z"]
        if n_sigma_z % 2 == 1:
            new_symbol.insert(0, "Z")
        # this op is identity, discard it
        if not new_symbol:
            continue
        new_dof_name = elem_op.dofs[0]
        if conserve_qn and new_dof_name % 2 == 1:
            qn_dict = qn_dict1
        else:
            qn_dict = qn_dict0
        new_qn = [qn_dict[s] for s in new_symbol]
        new_ops.append(Op(" ".join(new_symbol), new_dof_name, (-1) ** n_permute, new_qn))
    return Op.product(new_ops)


def _site_qn_dict(site_idx: int, conserve_qn: bool):
    if conserve_qn:
        if site_idx % 2 == 1:
            return {"+": [0, -1], "-": [0, 1], "Z": [0, 0]}
        else:
            return {"+": [-1, 0], "-": [1, 0], "Z": [0, 0]}
    else:
        return {"+": 0, "-": 0, "Z": 0}


def _jw_term_to_op(ladder_ops, conserve_qn: bool = True):
    """
    Construct the simplified spin operator for a product of fermionic ladder
    operators without first materialising the full Jordan-Wigner string.

    ``ladder_ops`` is a list of ``(orbital_index, symbol)`` pairs, where
    ``symbol`` is ``"-"`` for creation and ``"+"`` for annihilation, matching
    :func:`generate_ladder_operator`.
    """
    grouped = {}
    for iorb, symbol in ladder_ops:
        for iz in range(iorb):
            grouped.setdefault(iz, []).append("Z")
        grouped.setdefault(iorb, []).append(symbol)

    new_symbols = []
    new_dofs = []
    new_qn = []
    factor = 1

    for site_idx in sorted(grouped):
        symbols = grouped[site_idx]

        n_sigma_z = symbols.count("Z")
        n_non_sigma_z = 0
        n_permute = 0
        for simple_symbol in symbols:
            if simple_symbol != "Z":
                n_non_sigma_z += 1
            else:
                n_permute += n_non_sigma_z
        factor *= (-1) ** n_permute

        site_symbols = [s for s in symbols if s != "Z"]
        if n_sigma_z % 2 == 1:
            site_symbols.insert(0, "Z")
        if not site_symbols:
            continue

        qn_dict = _site_qn_dict(site_idx, conserve_qn)
        new_symbols.extend(site_symbols)
        new_dofs.extend([site_idx] * len(site_symbols))
        new_qn.extend(qn_dict[s] for s in site_symbols)

    return Op(" ".join(new_symbols), new_dofs, factor, new_qn)


def _one_electron_op(p: int, q: int, conserve_qn: bool = True):
    return _jw_term_to_op([(p, "-"), (q, "+")], conserve_qn)


def _two_electron_op(p: int, q: int, r: int, s: int, conserve_qn: bool = True):
    return _jw_term_to_op([(p, "-"), (q, "-"), (r, "+"), (s, "+")], conserve_qn)


def qc_model(h1e, h2e, stacked=False, conserve_qn=True, spatial_orbital=False):
    """
    Ab initio electronic Hamiltonian in spin-orbitals
    h1e: sh above
    h2e: aseri above
    spatial_orbital: if True, use one four-state local basis for each
        adjacent alpha/beta spin-orbital pair. The Hamiltonian terms are
        still represented by spin-orbital DoF labels.
    return model: "e_0", "e_1"... is according to the orbital index in sh and
    aseri
    """
    #------------------------------------------------------------------------
    # Jordan-Wigner transformation maps fermion problem into spin problem
    #
    # |0> => |alpha> and |1> => |beta >:
    #
    #    a_j   => Prod_{l=0}^{j-1}(sigma_z[l]) * sigma_+[j]
    #    a_j^+ => Prod_{l=0}^{j-1}(sigma_z[l]) * sigma_-[j]
    # j starts from 0 as in computer science convention to be consistent
    # with the following code.
    #------------------------------------------------------------------------

    norbs = h1e.shape[0]
    logger.info(f"spin norbs: {norbs}")
    assert np.all(np.array(h1e.shape) == norbs)
    assert np.all(np.array(h2e.shape) == norbs)

    ham_terms = []
    pairs1 = np.argwhere(h1e!=0)
    pairs2 = np.argwhere(h2e!=0)
    if stacked is False:
        # 1-e terms
        for p, q in pairs1:
            op = _one_electron_op(p, q, conserve_qn)
            ham_terms.append(op * h1e[p, q])

        # 2-e terms.
        for p, q, r, s in pairs2:
            op = _two_electron_op(p, q, r, s, conserve_qn)
            ham_terms.append(op * h2e[p, q, r, s])
    else:
        p_1e = np.unique(pairs1[:, 0])
        p_2e = np.unique(pairs2[:, 0])
        ps = set(p_1e).union(p_2e)
        for p in ps:
            local_ham_terms = []
            q_values = pairs1[pairs1[:, 0] == p][:, 1]
            qrs_values = pairs2[pairs2[:, 0] == p][:, 1:]
            if q_values.size > 0:
                for q in q_values:
                    op = _one_electron_op(p, q, conserve_qn)
                    local_ham_terms.append(op * h1e[p, q])
            if qrs_values.size > 0:
                for q, r, s in qrs_values:
                    op = _two_electron_op(p, q, r, s, conserve_qn)
                    local_ham_terms.append(op * h2e[p, q, r, s])
            ham_terms.append(local_ham_terms)

    if spatial_orbital:
        if norbs % 2 != 0:
            raise ValueError("spatial_orbital=True requires an even number of spin orbitals")
        if conserve_qn:
            basis = [BasisSpatialOrbital((2 * iorb, 2 * iorb + 1)) for iorb in range(norbs // 2)]
        else:
            basis = [
                BasisSpatialOrbital((2 * iorb, 2 * iorb + 1), sigmaqn=[0, 0, 0, 0])
                for iorb in range(norbs // 2)
            ]
    else:
        basis = []
        for iorb in range(norbs):
            if conserve_qn:
                if iorb % 2 == 0:
                    sigmaqn = np.array([[0, 0], [1, 0]])
                else:
                    sigmaqn = np.array([[0, 0], [0, 1]])
            else:
                sigmaqn = [0, 0]
            b = BasisHalfSpin(iorb, sigmaqn=sigmaqn)
            basis.append(b)
    return basis, ham_terms
