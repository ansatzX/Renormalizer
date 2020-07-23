# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

from collections import OrderedDict, defaultdict
from typing import List, Union, Dict

import numpy as np

from renormalizer.model.mol import Mol, Phonon
from renormalizer.utils import Quantity, Op
from renormalizer.utils import basis as ba


class Model:
    r"""
    User-defined model

    Args:
        order (dict or list): order of degrees of freedom.
        basis (dict or list): local basis (:class:`~renormalizer.utils.basis`) of each DoF
        model (dict): model of the system or any operator of the system,
            two formats are supported: 'vibronic type' or 'general type'. All terms
            must be included, without assuming hermitian or something else.
        dipole (dict): contains the transition dipole matrix element

    Note:
        the key of order starts from "v" or "e" for vibrational DoF or electronic DoF respectively.
        after the linker '_' combined with an index. The rule is that
        the index of both 'e' and 'v' starts from 0,1,2... and the
        properties such as :meth:`~renormalizer.mps.Mps.e_occupations` are reported with such order.
        the value of order is the position of the specific DoF, starting from 0,2,...,nsite-1
        For cases that all electronic DoFs gathered in a single site,
        the value of each DoF should be the same.
        for example: MolList scheme1/2/3 order = {"e_0":0, "v_0":1, "v_1":2, "e_1":3, "v_2":4, "v_3":5}
        MolList scheme4 order ={"e_0":0, "v_0":1, "v_1":2, "e_1":0, "v_2":3, "v_3":4}

        The order of basis corresponds to the site, each element is a Basis
        class, refer to :class:`~renormalizer.utils.basis.BasisSet`
        for example: basis = [b0,b1,b2,b3]

        in ``model``, each key is a tuple of DoFs,
        each value is list. Inside the list, each element is a tuple, the
        last one is the factor of each operator term, the others are local
        operator of the operator term.
        The model_translator is ModelTranslator.general_model
        for example:
        {("e_i","e_j") : [(Op1, Op2, factor)], ("e_i", "v_0",):[(Op1, Op2,
        factor), (Op1, Op2, factor)], ("v_1","v_2"):[(Op1, Op2, factor), (Op1, Op2, factor)]...}

        dipole contains transtion dipole matrix elements between the
        electronic DoFs. For simple electron case, dipole = {("e_0",):tdm1,
        ("e_1",):tdm2}, the key has 1 e_dof. For multi electron case, dipole =
        {("e_0","e_1"):tdm1, ("e_1","e_0"):tdm1}, the key has 2 e_dofs
        represents the transition.
    """
    def __init__(self, order: Union[Dict, List], basis: Union[Dict, List], model: Dict, dipole: Dict = None):

        if isinstance(order, dict):
            self.order = order
        else:
            assert isinstance(order, list)
            self.order = dict()
            for i, o in enumerate(order):
                self.order[o] = i

        if isinstance(basis, list):
            self.basis: List[ba.BasisSet] = basis
        else:
            assert isinstance(basis, dict)
            order_value_set = set(self.order.values())
            if min(order_value_set) != 0 or max(order_value_set) != len(order_value_set) - 1:
                raise ValueError("order is not continuous integers from 0")
            self.basis: List[ba.BasisSet] = [None] * (max(self.order.values()) + 1)
            for dof_name, dof_idx in self.order.items():
                self.basis[dof_idx] = basis[dof_name]

        for b in self.basis:
            if b.multi_dof:
                self.multi_dof_basis = True
                break
        else:
            self.multi_dof_basis = False

        self.model = model
        # array(n_e, n_e)
        self.dipole = dipole
        # reusable mpos for the system
        self.mpos = dict()
        # physical bond dimension. Read only.
        self._pbond_list = [bas.nbas for bas in self.basis]

    @property
    def pbond_list(self):
        return self._pbond_list

    @property
    def dofs(self):
        # If items(), keys(), values(),  iteritems(), iterkeys(), and
        # itervalues() are called with no intervening modifications to the
        # dictionary, the lists will directly correspond.
        
        return list(self.order.keys())
    
    @property
    def nsite(self):
        return len(self.basis)

    @property
    def e_dofs(self):
        dofs = []
        for key in self.order.keys():
            if key.split("_")[0] == "e":
                dofs.append(int(key.split("_")[1]))
        assert sorted(dofs) == list(range(len(dofs)))
        return [f"e_{i}" for i in range(len(dofs))]
    
    @property
    def v_dofs(self):
        dofs = []
        for key in self.order.keys():
            if key.split("_")[0] == "v":
                dofs.append(int(key.split("_")[1]))
        assert sorted(dofs) == list(range(len(dofs)))
        return [f"v_{i}" for i in range(len(dofs))]

    @property
    def n_edofs(self):
        return len(self.e_dofs)
    
    @property
    def n_vdofs(self):
        return len(self.v_dofs)

    def is_electron(self, idx):
        return self.basis[idx].is_electron

    @property
    def pure_dmrg(self):
        return True

    def get_mpos(self, key, fun):
        if key not in self.mpos:
            mpos = fun(self)
            self.mpos[key] = mpos
        else:
            mpos = self.mpos[key]
        return mpos

    def to_dict(self):
        info_dict = OrderedDict()
        info_dict["order"] = self.order
        info_dict["model"] = self.model
        return info_dict


class HolsteinModel(Model):

    def __init__(self,  mol_list: List[Mol], j_matrix: Union[Quantity, np.ndarray, None], scheme: int = 2, periodic: bool = False):
        # construct the electronic coupling matrix

        mol_num = len(mol_list)
        self.mol_list = mol_list

        if j_matrix is None:
            # spin-boson model
            assert len(mol_list) == 1
            j_matrix = Quantity(0)

        if isinstance(j_matrix, Quantity):
            j_matrix = construct_j_matrix(mol_num, j_matrix, periodic)
        else:
            if periodic:
                assert j_matrix[0][-1] != 0 and j_matrix[-1][0] != 0
            assert j_matrix.shape[0] == mol_num

        self.j_matrix = j_matrix
        self.scheme = scheme
        self.periodic = periodic

        order = {}
        basis = []
        model = {}
        mapping = {}

        if scheme < 4:
            idx = 0
            nv = 0
            for imol, mol in enumerate(mol_list):
                order[f"e_{imol}"] = idx
                basis.append(ba.BasisSimpleElectron())
                idx += 1
                for iph, ph in enumerate(mol.ph_list):
                    order[f"v_{nv}"] = idx
                    mapping[(imol, iph)] = f"v_{nv}"
                    basis.append(ba.BasisSHO(ph.omega[0], ph.n_phys_dim))
                    idx += 1
                    nv += 1

        elif scheme == 4:

            n_left_mol = mol_num // 2

            idx = 0
            n_left_ph = 0
            nv = 0
            for imol, mol in enumerate(mol_list):
                for iph, ph in enumerate(mol.ph_list):
                    if imol < n_left_mol:
                        order[f"v_{nv}"] = idx
                        n_left_ph += 1
                    else:
                        order[f"v_{nv}"] = idx + 1

                    basis.append(ba.BasisSHO(ph.omega[0], ph.n_phys_dim))
                    mapping[(imol, iph)] = f"v_{nv}"

                    nv += 1
                    idx += 1

            for imol in range(mol_num):
                order[f"e_{imol}"] = n_left_ph
            basis.insert(n_left_ph, ba.BasisMultiElectronVac(mol_num))

        else:
            raise ValueError(f"invalid model.scheme: {scheme}")

        # model

        # electronic term
        for imol in range(mol_num):
            for jmol in range(mol_num):
                if imol == jmol:
                    model[(f"e_{imol}",)] = \
                        [(Op(r"a^\dagger a", 0), mol_list[imol].elocalex + mol_list[imol].e0)]
                else:
                    model[(f"e_{imol}", f"e_{jmol}")] = \
                        [(Op(r"a^\dagger", 1), Op("a", -1), j_matrix[imol, jmol])]
        # vibration part
        for imol, mol in enumerate(mol_list):
            for iph, ph in enumerate(mol.ph_list):
                assert np.allclose(np.array(ph.force3rd), [0.0, 0.0])

                model[(mapping[(imol, iph)],)] = [
                    (Op("p^2", 0), 0.5),
                    (Op("x^2", 0), 0.5 * ph.omega[0] ** 2)
                ]

        # vibration potential part
        for imol, mol in enumerate(mol_list):
            for iph, ph in enumerate(mol.ph_list):
                if np.allclose(ph.omega[0], ph.omega[1]):
                    model[(f"e_{imol}", f"{mapping[(imol,iph)]}")] = [
                        (Op(r"a^\dagger a", 0), Op("x", 0), -ph.omega[1] ** 2 * ph.dis[1]),
                    ]
                else:
                    model[(f"e_{imol}", f"{mapping[(imol,iph)]}")] = [
                        (Op(r"a^\dagger a", 0), Op("x^2", 0), 0.5 * (ph.omega[1] ** 2 - ph.omega[0] ** 2)),
                        (Op(r"a^\dagger a", 0), Op("x", 0), -ph.omega[1] ** 2 * ph.dis[1]),
                    ]


        dipole = {}
        for imol, mol in enumerate(mol_list):
            dipole[(f"e_{imol}",)] = mol.dipole

        super().__init__(order, basis, model, dipole=dipole)
        # map: to be compatible with MolList {(imol, iph):"v_n"}
        self.map = mapping
        self.mol_num = self.n_edofs

    def switch_scheme(self, scheme):
        return HolsteinModel(self.mol_list, self.j_matrix, scheme)

    @property
    def gs_zpe(self):
        return sum([mol.gs_zpe for mol in self.mol_list])

    @property
    def j_constant(self):
        """Extract electronic coupling constant from ``self.j_matrix``.
        Useful in transport model.
        If J is actually not a constant, a value error will be raised.

        Returns
        -------
        j constant: float
            J constant extracted from ``self.j_matrix``.
        """
        j_set = set(self.j_matrix.ravel())
        if len(j_set) == 0:
            return j_set.pop()
        elif len(j_set) == 2 and 0 in j_set:
            j_set.remove(0)
            return j_set.pop()
        else:
            raise ValueError("J is not constant")

    def __getitem__(self, item):
        return self.mol_list[item]

    def __iter__(self):
        return iter(self.mol_list)

    def __len__(self):
        return len(self.mol_list)


class VibronicModel(Model):
    """The same with :class:`MolList2`. But the defination of ``model`` is different.
        each key is a tuple of electronic DoFs represents
        a^\dagger_i a_j or the key is "I" represents the pure vibrational
        terms, the value is a dict.
        The sub-key of the dict has two types, one is 'J' with value (float or complex) for pure electronic coupling,
        one is tuple of vibrational DoF with a list as value. Inside the
        list is sevaral tuples, each tuple is a operator term. the last one
        of the tuple is factor of the term, the others represents a local
        operator (refer to :class:`~renormalizer.utils.elementop.Op`) on each Dof in the
        sub-key (one-to-one map between the sub-key and tuple).
        The model_translator is ModelTranslator.vibronic_model
        for example:

        ::

            {
              "I"           : {("v_1"):[(Op,factor),]},
              ("e_i","e_j") : {
                "J":factor,
                ("v_0",):[(Op, factor), (Op, factor)],
                ("v_1","v_2"):[(Op1, Op2, factor), (Op1, Op2, factor)]
                ...
                }
              ...
            }
    """
    def __init__(self, order: Union[Dict, List], basis: Union[Dict, List], model: Dict, dipole: Dict = None):
        new_model = defaultdict(list)
        for e_k, e_v in model.items():
            for kk, vv in e_v.items():
                # it's difficult to rename `kk` because sometimes it's related to
                # phonons sometimes it's `"J"`
                if e_k == "I":
                    # operators without electronic dof, simple phonon
                    new_model[kk] = vv
                else:
                    # operators with electronic dof
                    assert isinstance(e_k, tuple) and len(e_k) == 2
                    if e_k[0] == e_k[1]:
                        # diagonal
                        new_e_k = (e_k[0],)
                        e_op = (Op(r"a^\dagger a", 0),)
                    else:
                        # off-diagonal
                        new_e_k = e_k
                        e_op = (Op(r"a^\dagger", 1), Op("a", -1))
                    if kk == "J":
                        new_model[new_e_k] = [e_op + (vv,)]
                    else:
                        for term in vv:
                            new_key = new_e_k + kk
                            new_value = e_op + term
                            new_model[new_key].append(new_value)

        super().__init__(order, basis, new_model, dipole)


class SpinBosonModel(Model):
    r"""
    Spin-Boson model

        .. math::
            \hat{H} = \epsilon \sigma_z + \Delta \sigma_x
                + \frac{1}{2} \sum_i(p_i^2+\omega^2_i q_i^2)
                + \sigma_z \sum_i c_i q_i

    """
    def __init__(self, epsilon: Quantity, delta: Quantity, ph_list: List[Phonon], dipole: float=None):

        self.epsilon = epsilon.as_au()
        self.delta = delta.as_au()
        self.ph_list = ph_list

        order = {}
        basis = []
        model = {}

        order[f"e_0"] = 0
        basis.append(ba.BasisHalfSpin())
        for iph, ph in enumerate(ph_list):
            order[f"v_{iph}"] = iph+1
            basis.append(ba.BasisSHO(ph.omega[0], ph.n_phys_dim))

        # spin
        model[(f"e_0",)] = [(Op(r"sigma_z", 0), self.epsilon),(Op("sigma_x", 0), self.delta)]
        # vibration energy and potential
        for iph, ph in enumerate(ph_list):
            assert ph.is_simple
            model[(f"v_{iph}",)] = [
                (Op("p^2", 0), 0.5),
                (Op("x^2", 0), 0.5 * ph.omega[0] ** 2)
            ]
            model[(f"e_0", f"v_{iph}")] = [
                (Op("sigma_z", 0), Op("x", 0), -ph.omega[1] ** 2 * ph.dis[1]),
            ]
        if dipole is None:
            dipole = 0
        super().__init__(order, basis, model, dipole={"e_0": dipole})


def construct_j_matrix(mol_num, j_constant, periodic):
    # nearest neighbour interaction
    j_constant_au = j_constant.as_au()
    j_list = np.ones(mol_num - 1) * j_constant_au
    j_matrix = np.diag(j_list, k=-1) + np.diag(j_list, k=1)
    if periodic:
        j_matrix[-1, 0] = j_matrix[0, -1] = j_constant_au
    return j_matrix


def load_from_dict(param, scheme, lam: bool):
    temperature = Quantity(*param["temperature"])
    ph_list = [
        Phonon.simplest_phonon(
            Quantity(*omega), Quantity(*displacement), temperature=temperature, lam=lam
        )
        for omega, displacement in param["ph modes"]
    ]
    j_constant = Quantity(*param["j constant"])
    model = HolsteinModel([Mol(Quantity(0), ph_list)] * param["mol num"], j_constant, scheme)
    return model, temperature