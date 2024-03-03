import numpy as np

from pymolresponse.interfaces.pyscf import integrals, molecules


def test_integrals_pyscf():
    mol = molecules.molecule_water_sto3g()
    mol.build()
    integral_generator = integrals.IntegralsPyscf(mol)
    np.testing.assert_equal(
        mol.intor("cint1e_r_sph", comp=3), integral_generator.integrals(integrals.DIPOLE)
    )
    np.testing.assert_equal(
        mol.intor("cint1e_cg_irxp_sph", comp=3),
        integral_generator.integrals(integrals.ANGMOM_COMMON_GAUGE),
    )


def test_jk_pyscf():
    mol = molecules.molecule_water_sto3g()
    mol.build()
    jk_generator = integrals.JKPyscf(mol)  # noqa: F841

    # print(jk_generator.compute_from_mocoeffs())
