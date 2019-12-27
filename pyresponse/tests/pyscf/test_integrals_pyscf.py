import numpy as np

from pyresponse.pyscf import integrals, molecules


def test_integrals_pyscf():
    mol = molecules.molecule_water_sto3g()
    mol.build()
    integral_generator = integrals.IntegralsPyscf(mol)
    np.testing.assert_equal(
        mol.intor("cint1e_r_sph", comp=3),
        integral_generator.integrals(integral_generator.DIPOLE),
    )
    np.testing.assert_equal(
        mol.intor("cint1e_cg_irxp_sph", comp=3),
        integral_generator.integrals(integral_generator.ANGMOM_COMMON_GAUGE),
    )
