import numpy as np

import psi4

from pymolresponse.interfaces.psi4 import integrals, molecules
from pymolresponse.interfaces.psi4.utils import mocoeffs_from_psi4wfn, occupations_from_psi4wfn


def test_integrals_psi4():
    mol = molecules.molecule_water_sto3g()
    mol.reset_point_group("c1")
    mol.update_geometry()
    psi4.core.set_active_molecule(mol)

    options = {"BASIS": "STO-3G", "SCF_TYPE": "PK", "E_CONVERGENCE": 1e-10, "D_CONVERGENCE": 1e-10}

    psi4.set_options(options)

    _, wfn = psi4.energy("hf", return_wfn=True)

    mints = psi4.core.MintsHelper(wfn)

    integral_generator = integrals.IntegralsPsi4(wfn)

    np.testing.assert_equal(
        np.stack([np.asarray(Mc) for Mc in mints.ao_dipole()]),
        integral_generator.integrals(integrals.DIPOLE),
    )


def test_jk_psi4():
    mol = molecules.molecule_water_sto3g()
    mol.reset_point_group("c1")
    mol.update_geometry()
    psi4.core.set_active_molecule(mol)

    options = {"BASIS": "STO-3G", "SCF_TYPE": "PK", "E_CONVERGENCE": 1e-10, "D_CONVERGENCE": 1e-10}

    psi4.set_options(options)

    _, wfn = psi4.energy("hf", return_wfn=True)
    C = mocoeffs_from_psi4wfn(wfn)
    jk_generator = integrals.JKPsi4(wfn)
    res_J, res_K = jk_generator.compute_from_mocoeffs(C[0])

    # TODO comparison against reference
    occupations = occupations_from_psi4wfn(wfn)
    # TODO this assumes that the total number of orbitals is equal to the
    # total number of basis functions
    nbasis = occupations[0] + occupations[1]
    assert res_J.shape == (nbasis, nbasis)
    assert res_K.shape == (nbasis, nbasis)
