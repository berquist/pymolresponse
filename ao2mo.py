import pyscf

from utils import fix_mocoeffs_shape, occupations_from_pyscf_mol


def perform_tei_ao2mo_rhf_partial(pyscfmol, C, verbose=1):
    C = fix_mocoeffs_shape(C)
    occupations = occupations_from_pyscf_mol(pyscfmol, C)
    nocc_a, nvirt_a, _, _ = occupations
    C_occ = C[0, :, :nocc_a]
    C_virt = C[0, :, nocc_a:]
    C_ovov = (C_occ, C_virt, C_occ, C_virt)
    C_oovv = (C_occ, C_occ, C_virt, C_virt)
    tei_mo_ovov = pyscf.ao2mo.general(pyscfmol, C_ovov, aosym='s1', compact=False, verbose=verbose).reshape(nocc_a, nvirt_a, nocc_a, nvirt_a)
    tei_mo_oovv = pyscf.ao2mo.general(pyscfmol, C_oovv, aosym='s1', compact=False, verbose=verbose).reshape(nocc_a, nocc_a, nvirt_a, nvirt_a)
    tei_mo = (tei_mo_ovov, tei_mo_oovv)
    return tei_mo


def perform_tei_ao2mo_uhf_partial(pyscfmol, C, verbose=1):
    C = fix_mocoeffs_shape(C)
    occupations = occupations_from_pyscf_mol(pyscfmol, C)
    nocc_a, nvirt_a, nocc_b, nvirt_b = occupations
    C_occ_alph = C[0, :, :nocc_a]
    C_virt_alph = C[0, :, nocc_a:]
    C_occ_beta = C[1, :, :nocc_b]
    C_virt_beta = C[1, :, nocc_b:]
    C_ovov_aaaa = (C_occ_alph, C_virt_alph, C_occ_alph, C_virt_alph)
    C_ovov_aabb = (C_occ_alph, C_virt_alph, C_occ_beta, C_virt_beta)
    C_ovov_bbaa = (C_occ_beta, C_virt_beta, C_occ_alph, C_virt_alph)
    C_ovov_bbbb = (C_occ_beta, C_virt_beta, C_occ_beta, C_virt_beta)
    C_oovv_aaaa = (C_occ_alph, C_occ_alph, C_virt_alph, C_virt_alph)
    C_oovv_bbbb = (C_occ_beta, C_occ_beta, C_virt_beta, C_virt_beta)
    tei_mo_ovov_aaaa = pyscf.ao2mo.general(pyscfmol, C_ovov_aaaa, aosym='s1', compact=False, verbose=verbose).reshape(nocc_a, nvirt_a, nocc_a, nvirt_a)
    tei_mo_ovov_aabb = pyscf.ao2mo.general(pyscfmol, C_ovov_aabb, aosym='s1', compact=False, verbose=verbose).reshape(nocc_a, nvirt_a, nocc_b, nvirt_b)
    tei_mo_ovov_bbaa = pyscf.ao2mo.general(pyscfmol, C_ovov_bbaa, aosym='s1', compact=False, verbose=verbose).reshape(nocc_b, nvirt_b, nocc_a, nvirt_a)
    tei_mo_ovov_bbbb = pyscf.ao2mo.general(pyscfmol, C_ovov_bbbb, aosym='s1', compact=False, verbose=verbose).reshape(nocc_b, nvirt_b, nocc_b, nvirt_b)
    tei_mo_oovv_aaaa = pyscf.ao2mo.general(pyscfmol, C_oovv_aaaa, aosym='s1', compact=False, verbose=verbose).reshape(nocc_a, nocc_a, nvirt_a, nvirt_a)
    tei_mo_oovv_bbbb = pyscf.ao2mo.general(pyscfmol, C_oovv_bbbb, aosym='s1', compact=False, verbose=verbose).reshape(nocc_b, nocc_b, nvirt_b, nvirt_b)
    tei_mo = (tei_mo_ovov_aaaa, tei_mo_ovov_aabb, tei_mo_ovov_bbaa, tei_mo_ovov_bbbb, tei_mo_oovv_aaaa, tei_mo_oovv_bbbb)
    return tei_mo
