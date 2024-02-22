import numpy as np

import pyscf

from pymolresponse.ao2mo import AO2MO
from pymolresponse.interfaces.pyscf import molecules as molecules_pyscf
from pymolresponse.interfaces.pyscf.ao2mo import AO2MOpyscf
from pymolresponse.interfaces.pyscf.utils import occupations_from_pyscf_mol
from pymolresponse.utils import fix_mocoeffs_shape


def test_ao2mo_hand_against_pyscf_rhf_full() -> None:
    mol = molecules_pyscf.molecule_physicists_water_sto3g()
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    C = fix_mocoeffs_shape(mf.mo_coeff)
    occupations = occupations_from_pyscf_mol(mol, C)
    nocc, nvirt, _, _ = occupations
    nmo = nocc + nvirt
    ntransforms = 1

    ao2mo = AO2MOpyscf(C, verbose=mol.verbose, pyscfmol=mol)
    ao2mo.perform_rhf_full()
    assert len(ao2mo.tei_mo) == ntransforms
    tei_mo_pyscf = ao2mo.tei_mo[0]

    tei_ao = mol.intor("int2e_sph", aosym="s1")

    print("1. Use the class method explicitly.")

    tei_mo_hand = AO2MO.transform(tei_ao, C[0], C[0], C[0], C[0])

    assert tei_mo_pyscf.shape == tei_mo_hand.shape == (nmo, nmo, nmo, nmo)
    np.testing.assert_allclose(tei_mo_hand, tei_mo_pyscf, rtol=0, atol=1.0e-15)

    print("2. Use the class method normally.")

    ao2mo_method = AO2MO(C, occupations, verbose=mol.verbose, I=tei_ao)
    ao2mo_method.perform_rhf_full()
    assert len(ao2mo_method.tei_mo) == ntransforms
    tei_mo_method = ao2mo_method.tei_mo[0]

    assert tei_mo_pyscf.shape == tei_mo_method.shape == (nmo, nmo, nmo, nmo)
    np.testing.assert_allclose(tei_mo_method, tei_mo_pyscf, rtol=0, atol=1.0e-15)


def test_ao2mo_hand_against_pyscf_rhf_partial() -> None:
    mol = molecules_pyscf.molecule_physicists_water_sto3g()
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    C = fix_mocoeffs_shape(mf.mo_coeff)
    occupations = occupations_from_pyscf_mol(mol, C)
    nocc, nvirt, _, _ = occupations
    nmo = nocc + nvirt
    ntransforms = 2

    o = slice(0, nocc)
    v = slice(nocc, nmo)

    ao2mo = AO2MOpyscf(C, verbose=mol.verbose, pyscfmol=mol)
    ao2mo.perform_rhf_partial()
    assert len(ao2mo.tei_mo) == ntransforms
    tei_mo_ovov_pyscf = ao2mo.tei_mo[0]
    tei_mo_oovv_pyscf = ao2mo.tei_mo[1]

    tei_ao = mol.intor("int2e_sph", aosym="s1")

    print("1. Use the class method explicitly.")

    tei_mo_ovov_hand = AO2MO.transform(tei_ao, C[0, :, o], C[0, :, v], C[0, :, o], C[0, :, v])
    tei_mo_oovv_hand = AO2MO.transform(tei_ao, C[0, :, o], C[0, :, o], C[0, :, v], C[0, :, v])

    assert tei_mo_ovov_pyscf.shape == tei_mo_ovov_hand.shape == (nocc, nvirt, nocc, nvirt)
    assert tei_mo_oovv_pyscf.shape == tei_mo_oovv_hand.shape == (nocc, nocc, nvirt, nvirt)
    np.testing.assert_allclose(tei_mo_ovov_hand, tei_mo_ovov_pyscf, rtol=0, atol=1.0e-15)
    np.testing.assert_allclose(tei_mo_oovv_hand, tei_mo_oovv_pyscf, rtol=0, atol=1.0e-15)

    print("2. Use the class method normally.")

    ao2mo_method = AO2MO(C, occupations, verbose=mol.verbose, I=tei_ao)
    ao2mo_method.perform_rhf_partial()
    assert len(ao2mo_method.tei_mo) == ntransforms
    tei_mo_ovov_method = ao2mo_method.tei_mo[0]
    tei_mo_oovv_method = ao2mo_method.tei_mo[1]

    assert tei_mo_ovov_pyscf.shape == tei_mo_ovov_method.shape == (nocc, nvirt, nocc, nvirt)
    assert tei_mo_oovv_pyscf.shape == tei_mo_oovv_method.shape == (nocc, nocc, nvirt, nvirt)
    np.testing.assert_allclose(tei_mo_ovov_method, tei_mo_ovov_pyscf, rtol=0, atol=1.0e-15)
    np.testing.assert_allclose(tei_mo_oovv_method, tei_mo_oovv_pyscf, rtol=0, atol=1.0e-15)


if __name__ == "__main__":
    # test_ao2mo_hand_against_psi4()
    test_ao2mo_hand_against_pyscf_rhf_full()
    test_ao2mo_hand_against_pyscf_rhf_partial()
