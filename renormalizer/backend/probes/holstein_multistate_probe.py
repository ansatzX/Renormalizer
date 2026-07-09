# -*- coding: utf-8 -*-

"""Short multi-state Holstein DMRG probe for batched RHS HMM coverage."""

from renormalizer.mps.gs import construct_mps_mpo, optimize_mps
from renormalizer.tests.parameter import holstein_model


if __name__ == "__main__":
    procedure = [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]

    mps, mpo = construct_mps_mpo(holstein_model, procedure[0][0], 1)
    mps.optimize_config.procedure = procedure
    mps.optimize_config.nroots = 4
    mps.optimize_config.method = "1site"
    mps.optimize_config.algo = "davidson"
    mps.optimize_config.e_atol = 1e-6
    mps.optimize_config.e_rtol = 1e-6

    energies, _ = optimize_mps(mps, mpo)
    print("energies_last", [float(value) for value in energies[-1]])
