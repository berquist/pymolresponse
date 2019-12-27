import numpy as np

import psi4

from pyresponse.psi4 import integrals, molecules


def test_integrals_psi4():
    mol = molecules.molecule_water_sto3g()
    mol.reset_point_group("c1")
    mol.update_geometry()
    psi4.core.set_active_molecule(mol)

    options = {
        "BASIS": "STO-3G",
        "SCF_TYPE": "PK",
        "E_CONVERGENCE": 1e-10,
        "D_CONVERGENCE": 1e-10,
    }

    psi4.set_options(options)

    _, wfn = psi4.energy("hf", return_wfn=True)

    mints = psi4.core.MintsHelper(wfn)

    integral_generator = integrals.IntegralsPsi4(wfn)

    np.testing.assert_equal(
        np.stack([np.asarray(Mc) for Mc in mints.ao_dipole()]),
        integral_generator.integrals("dipole"),
    )
