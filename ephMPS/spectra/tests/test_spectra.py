# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
from __future__ import absolute_import, print_function, division

import os
import unittest

import numpy as np
from ddt import ddt, data, unpack

from ephMPS.spectra import SpectraExact, SpectraOneWayPropZeroT, SpectraTwoWayPropZeroT, \
    SpectraEmiFiniteT, SpectraAbsFiniteT, prepare_init_mps
from ephMPS.spectra.tests import cur_dir
from ephMPS.tests import parameter
from ephMPS.utils import constant
from ephMPS.utils.quantity import Quantity


@ddt
class TestZeroExactEmi(unittest.TestCase):
    @data([[[4, 4]], 1e-3])
    @unpack
    def test_zero_exact_emi(self, ph_info, rtol):
        # print "data", value
        nexciton = 1
        procedure = [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]
        mol_list = parameter.custom_mol_list(*ph_info)
        i_mps, h_mpo = prepare_init_mps(mol_list, parameter.j_matrix, procedure, nexciton, 2, offset=2.28614053,
                                        optimize=True)
        nsteps = 3000
        dt = 30.0
        temperature = Quantity(0, 'K')
        exact_emi = SpectraExact(i_mps, h_mpo, spectratype='emi', temperature=temperature)
        exact_emi.evolve(dt, nsteps)
        with open(os.path.join(cur_dir, 'ZeroExactEmi.npy'), 'rb') as fin:
            std = np.load(fin)

        self.assertTrue(np.allclose(exact_emi.autocorr, std, rtol=rtol))


@ddt
class TestZeroTCorr(unittest.TestCase):
    @data(
        [1, "svd", [[4, 4]], 1e-3],
        [2, "svd", [[4, 4]], 1e-3],
        [1, "svd", [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2],
        [2, "svd", [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2],
    )
    @unpack
    def test_zero_t_corr(self, algorithm, compress_method, ph_info, rtol):
        np.random.seed(0)
        # print "data", value
        nexciton = 0
        procedure = [[1, 0], [1, 0], [1, 0]]
        nsteps = 100
        dt = 30.0
        mol_list = parameter.custom_mol_list(*ph_info)
        i_mps, h_mpo = prepare_init_mps(mol_list, parameter.j_matrix, procedure, nexciton, 2, compress_method,
                                        2.28614053, optimize=True)
        if algorithm == 1:
            zero_t_corr = SpectraOneWayPropZeroT(i_mps, h_mpo)
        else:
            zero_t_corr = SpectraTwoWayPropZeroT(i_mps, h_mpo)
        zero_t_corr.evolve(dt, nsteps)
        with open(os.path.join(cur_dir, 'ZeroTabs_' + str(algorithm) + str(compress_method) + '.npy'), 'rb') as f:
            std = np.load(f)
        self.assertTrue(np.allclose(zero_t_corr.autocorr, std, rtol=rtol))

    @data([1, "svd", [[4, 4]], 1e-3],
          [2, "svd", [[4, 4]], 1e-3],
          [1, "svd", [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2],
          [2, "svd", [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2],
          )
    @unpack
    def test_zero_t_corr_mposcheme3(self, algorithm, compress_method, ph_info, rtol):
        np.random.seed(0)
        # print "data", value
        j_matrix = np.array([[0.0, -0.1, 0.0], [-0.1, 0.0, -0.3], [0.0, -0.3, 0.0]]) / constant.au2ev
        nexciton = 0
        procedure = [[1, 0], [1, 0], [1, 0]]
        mol_list = parameter.custom_mol_list(*ph_info)
        nsteps = 50
        dt = 30.0
        i_mps, h_mpo = prepare_init_mps(mol_list, j_matrix, procedure, nexciton, 2, compress_method,
                                        2.28614053, optimize=True)
        if algorithm == 1:
            SpectraClass = SpectraOneWayPropZeroT
        else:
            SpectraClass = SpectraTwoWayPropZeroT
        zero_t_corr2 = SpectraClass(i_mps, h_mpo).evolve(dt, nsteps)
        zero_t_corr3 = SpectraClass(i_mps, h_mpo).evolve(dt, nsteps)
        self.assertTrue(np.allclose(zero_t_corr2.autocorr, zero_t_corr3.autocorr, rtol=rtol))


@ddt
class TestFiniteTSpectraEmi(unittest.TestCase):
    @data([2, "svd", [[4, 4]], 1e-3],
          [2, "svd", [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2]
          )
    @unpack
    def test_finite_t_spectra_emi(self, algorithm, compress_method, ph_info, rtol):
        np.random.seed(0)
        # print "data", value
        nexciton = 1
        procedure = [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]
        mol_list = parameter.custom_mol_list(*ph_info)
        nsteps = 30
        dt = 30.0
        insteps = 50
        i_mps, h_mpo = prepare_init_mps(mol_list, parameter.j_matrix, procedure, nexciton, 2, compress_method,
                                        2.28614053)
        finite_t_emi = SpectraEmiFiniteT(i_mps, h_mpo, temperature=Quantity(298, 'K'), insteps=insteps)
        finite_t_emi.evolve(dt, nsteps)
        with open(os.path.join(cur_dir, 'TTemi_' + str(algorithm) + str(compress_method) + ".npy"), 'rb') as fin:
            std = np.load(fin)
        self.assertTrue(np.allclose(finite_t_emi.autocorr, std[0:nsteps], rtol=rtol))


@ddt
class TestFiniteTSpectraAbs(unittest.TestCase):
    @data([2, "svd", [[4, 4]], 1e-3],
          [2, "svd", [[4, 4], [2, 2], [1.e-7, 1.e-7]], 1e-2]
          )
    @unpack
    def test_finite_t_spectra_abs(self, algorithm, compress_method, ph_info, rtol):
        # print "data", value
        nexciton = 0
        procedure = [[1, 0], [1, 0], [1, 0]]
        mol_list = parameter.custom_mol_list(*ph_info)
        nsteps = 30
        dt = 30.0
        insteps = 50
        i_mps, h_mpo = prepare_init_mps(mol_list, parameter.j_matrix, procedure, nexciton, 2, compress_method,
                                        2.28614053)
        finite_t_abs = SpectraAbsFiniteT(i_mps, h_mpo, temperature=Quantity(298, 'K'), insteps=insteps)
        finite_t_abs.evolve(dt, nsteps)
        with open(os.path.join(cur_dir, "TTabs_" + str(compress_method) + ".npy"), 'rb') as fin:
            std = np.load(fin)
        self.assertTrue(np.allclose(finite_t_abs.autocorr, std[0:nsteps], rtol=rtol))


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFiniteTSpectraAbs)
    unittest.TextTestRunner().run(suite)
    #unittest.main()
