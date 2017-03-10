import numpy as np

from utils import np_load
from utils import (form_rpa_a_matrix_mo_singlet,
                   form_rpa_b_matrix_mo_singlet,
                   form_rpa_a_matrix_mo_triplet,
                   form_rpa_b_matrix_mo_triplet)


def form_rpa_a_matrix_mo_singlet_ss(E_MO, TEI_MO, nocc):

    norb = E_MO.shape[0]
    nvirt = norb - nocc
    nov = nocc * nvirt

    A = np.empty(shape=(nov, nov))

    for i in range(nocc):
        for a in range(nvirt):
            ia = i*nvirt + a
            for j in range(nocc):
                for b in range(nvirt):
                    jb = j*nvirt + b
                    A[ia, jb] = TEI_MO[a + nocc, i, j, b + nocc] - TEI_MO[a + nocc, b + nocc, j, i]
                    if (ia == jb):
                        A[ia, jb] += (E_MO[a + nocc, b + nocc] - E_MO[i, j])

    return A


def form_rpa_a_matrix_mo_singlet_os(TEI_MO_xxyy, nocc_x, nocc_y):

    nvirt_x = TEI_MO_xxyy.shape[0] - nocc_x
    nvirt_y = TEI_MO_xxyy.shape[2] - nocc_y
    nov_x = nocc_x * nvirt_x
    nov_y = nocc_y * nvirt_y

    A = np.empty(shape=(nov_x, nov_y))

    for i in range(nocc_x):
        for a in range(nvirt_x):
            ia = i*nvirt_x + a
            for j in range(nocc_y):
                for b in range(nvirt_y):
                    jb = j*nvirt_y + b
                    A[ia, jb] = TEI_MO_xxyy[a + nocc_x, i, j, b + nocc_y]

    return A


def form_rpa_b_matrix_mo_singlet_ss(TEI_MO, nocc):

    norb = TEI_MO.shape[0]
    nvirt = norb - nocc
    nov = nocc * nvirt

    B = np.empty(shape=(nov, nov))

    for i in range(nocc):
        for a in range(nvirt):
            ia = i*nvirt + a
            for j in range(nocc):
                for b in range(nvirt):
                    jb = j*nvirt + b
                    B[ia, jb] = TEI_MO[a + nocc, i, b + nocc, j] - TEI_MO[a + nocc, j, b + nocc, i]

    return B


def form_rpa_b_matrix_mo_singlet_os(TEI_MO_xxyy, nocc_x, nocc_y):

    nvirt_x = TEI_MO_xxyy.shape[0] - nocc_x
    nvirt_y = TEI_MO_xxyy.shape[2] - nocc_y
    nov_x = nocc_x * nvirt_x
    nov_y = nocc_y * nvirt_y

    B = np.empty(shape=(nov_x, nov_y))

    for i in range(nocc_x):
        for a in range(nvirt_x):
            ia = i*nvirt_x + a
            for j in range(nocc_y):
                for b in range(nvirt_y):
                    jb = j*nvirt_y + b
                    B[ia, jb] = TEI_MO_xxyy[a + nocc_x, i, b + nocc_y, j]

    return B

def test_direct_uhf_from_rhf():
    from pyscf import ao2mo, gto, scf

    mol = gto.Mole()
    mol.verbose = 5
    with open('water.xyz') as fh:
        mol.atom = fh.read()
    mol.unit = 'Bohr'
    mol.basis = 'sto-3g'
    mol.symmetry = False
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    mocoeffs = mf.mo_coeff
    moenergies = mf.mo_energy
    norb = mocoeffs.shape[1]
    tei_mo = ao2mo.full(mol, mocoeffs, aosym='s1', compact=False).reshape(norb, norb, norb, norb)

    C_a = mocoeffs
    C_b = C_a.copy()
    E_a = np.diag(moenergies)
    E_b = E_a.copy()
    nocc_a, nocc_b = mol.nelec
    nvirt_a, nvirt_b = norb - nocc_a, norb - nocc_b
    occupations = [nocc_a, nvirt_a, nocc_b, nvirt_b]

    # Same-spin and opposite-spin contributions should add together
    # properly for restricted wavefunction.
    A_s = form_rpa_a_matrix_mo_singlet(E_a, tei_mo, nocc_a)
    A_s_ss = form_rpa_a_matrix_mo_singlet_ss(E_a, tei_mo, nocc_a)
    A_s_os = form_rpa_a_matrix_mo_singlet_os(tei_mo, nocc_a, nocc_b)
    np.testing.assert_allclose(A_s, A_s_ss + A_s_os)
    B_s = form_rpa_b_matrix_mo_singlet(tei_mo, nocc_a)
    B_s_ss = form_rpa_b_matrix_mo_singlet_ss(tei_mo, nocc_a)
    B_s_os = form_rpa_b_matrix_mo_singlet_os(tei_mo, nocc_a, nocc_b)
    np.testing.assert_allclose(B_s, B_s_ss + B_s_os)
    # Since the "triplet" part contains no Coulomb contribution, and
    # (xx|yy) is only in the Coulomb part, there is no ss/os
    # separation for the triplet part.

    G_r = np.bmat([[A_s, B_s],
                   [B_s, A_s]])
    G_aa = np.bmat([[A_s_ss, B_s_ss],
                    [B_s_ss, A_s_ss]])
    G_bb = G_aa.copy()
    G_ab = np.bmat([[A_s_os, B_s_os],
                    [B_s_os, A_s_os]])
    G_ba = G_ab.copy()

    np.testing.assert_allclose(G_r, (G_aa + G_ab))

    G_r_inv = np.linalg.inv(G_r)
    G_aa_inv = np.linalg.inv(G_aa)
    G_ab_inv = np.linalg.inv(G_ab)
    G_ba_inv = np.linalg.inv(G_ba)
    G_bb_inv = np.linalg.inv(G_bb)

    assert G_r_inv.shape == (2*nocc_a*nvirt_a, 2*nocc_a*nvirt_a)
    assert G_aa_inv.shape == (2*nocc_a*nvirt_a, 2*nocc_a*nvirt_a)
    assert G_ab_inv.shape == (2*nocc_a*nvirt_b, 2*nocc_a*nvirt_b)
    assert G_ba_inv.shape == (2*nocc_b*nvirt_a, 2*nocc_b*nvirt_a)
    assert G_bb_inv.shape == (2*nocc_b*nvirt_b, 2*nocc_b*nvirt_b)

    # Form the operator_dependent part of the response vectors.
    left_alph = np.linalg.inv(G_aa - np.dot(G_ab, np.dot(G_bb_inv, G_ba)))
    left_beta = np.linalg.inv(G_bb - np.dot(G_ba, np.dot(G_aa_inv, G_ab)))

    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 5
    with open('water.xyz') as fh:
        mol.atom = fh.read()
    mol.unit = 'Bohr'
    mol.basis = 'sto-3g'
    mol.symmetry = False
    mol.build()

    integrals_dipole_ao = mol.intor('cint1e_r_sph', comp=3)

    integrals_dipole_mo_ai_r = []
    integrals_dipole_mo_ai_a = []
    integrals_dipole_mo_ai_b = []

    for comp in range(3):

        integrals_dipole_mo_ai_comp_r = np.dot(C_a[:, nocc_a:].T, np.dot(integrals_dipole_ao[comp, ...], C_a[:, :nocc_a]))
        integrals_dipole_mo_ai_comp_a = np.dot(C_a[:, nocc_a:].T, np.dot(integrals_dipole_ao[comp, ...], C_a[:, :nocc_a]))
        integrals_dipole_mo_ai_comp_b = np.dot(C_b[:, nocc_b:].T, np.dot(integrals_dipole_ao[comp, ...], C_b[:, :nocc_b]))

        integrals_dipole_mo_ai_comp_r = np.reshape(integrals_dipole_mo_ai_comp_r, -1, order='F')
        integrals_dipole_mo_ai_comp_a = np.reshape(integrals_dipole_mo_ai_comp_a, -1, order='F')
        integrals_dipole_mo_ai_comp_b = np.reshape(integrals_dipole_mo_ai_comp_b, -1, order='F')

        integrals_dipole_mo_ai_r.append(integrals_dipole_mo_ai_comp_r)
        integrals_dipole_mo_ai_a.append(integrals_dipole_mo_ai_comp_a)
        integrals_dipole_mo_ai_b.append(integrals_dipole_mo_ai_comp_b)

    integrals_dipole_mo_ai_r = np.stack(integrals_dipole_mo_ai_r, axis=0).T
    integrals_dipole_mo_ai_a = np.stack(integrals_dipole_mo_ai_a, axis=0).T
    integrals_dipole_mo_ai_b = np.stack(integrals_dipole_mo_ai_b, axis=0).T

    integrals_dipole_mo_ai_r_super = np.concatenate((integrals_dipole_mo_ai_r, integrals_dipole_mo_ai_r), axis=0)
    integrals_dipole_mo_ai_a_super = np.concatenate((integrals_dipole_mo_ai_a, integrals_dipole_mo_ai_a), axis=0)
    integrals_dipole_mo_ai_b_super = np.concatenate((integrals_dipole_mo_ai_b, integrals_dipole_mo_ai_b), axis=0)

    # Form the operator-dependent part of the response vectors.
    right_alph = integrals_dipole_mo_ai_a_super - (np.dot(G_ab, np.dot(G_bb_inv, integrals_dipole_mo_ai_b_super)))
    right_beta = integrals_dipole_mo_ai_b_super - (np.dot(G_ba, np.dot(G_aa_inv, integrals_dipole_mo_ai_a_super)))

    rspvec_r = np.dot(G_r_inv, integrals_dipole_mo_ai_r_super)

    # The total response vector for each spin is the product of the
    # operator-independent (left) and operator-dependent (right)
    # parts.
    rspvec_a = np.dot(left_alph, right_alph)
    rspvec_b = np.dot(left_beta, right_beta)

    res_r = 4 * np.dot(integrals_dipole_mo_ai_r_super.T, rspvec_r) / 2
    res_a = np.dot(integrals_dipole_mo_ai_a_super.T, rspvec_a) / 2
    res_b = np.dot(integrals_dipole_mo_ai_b_super.T, rspvec_b) / 2
    res_u = 2 * (res_a + res_b)

    atol = 1.0e-8
    rtol = 0.0
    np.testing.assert_allclose(res_u, res_r, rtol=rtol, atol=atol)

    print(res_r)
    print(res_u)


def test_direct_uhf():
    from pyscf import ao2mo, gto, scf

    mol = gto.Mole()
    mol.verbose = 5
    with open('water.xyz') as fh:
        mol.atom = fh.read()
    mol.unit = 'Bohr'
    mol.basis = 'sto-3g'
    mol.symmetry = False

    mol.charge = 1
    mol.spin = 1

    mol.build()

    mf = scf.UHF(mol)
    mf.kernel()
    C_a = mf.mo_coeff[0, ...]
    C_b = mf.mo_coeff[1, ...]
    E_a = np.diag(mf.mo_energy[0, ...])
    E_b = np.diag(mf.mo_energy[1, ...])
    assert C_a.shape == C_b.shape
    assert E_a.shape == E_b.shape
    norb = C_a.shape[1]
    C_aaaa = (C_a, C_a, C_a, C_a)
    C_aabb = (C_a, C_a, C_b, C_b)
    C_bbaa = (C_b, C_b, C_a, C_a)
    C_bbbb = (C_b, C_b, C_b, C_b)
    tei_mo_aaaa = ao2mo.general(mol, C_aaaa, aosym='s1', compact=False, verbose=5).reshape(norb, norb, norb, norb)
    tei_mo_aabb = ao2mo.general(mol, C_aabb, aosym='s1', compact=False, verbose=5).reshape(norb, norb, norb, norb)
    tei_mo_bbaa = ao2mo.general(mol, C_bbaa, aosym='s1', compact=False, verbose=5).reshape(norb, norb, norb, norb)
    tei_mo_bbbb = ao2mo.general(mol, C_bbbb, aosym='s1', compact=False, verbose=5).reshape(norb, norb, norb, norb)

    nocc_a, nocc_b = mol.nelec
    nvirt_a, nvirt_b = norb - nocc_a, norb - nocc_b
    occupations = [nocc_a, nvirt_a, nocc_b, nvirt_b]

    A_s_ss_a = form_rpa_a_matrix_mo_singlet_ss(E_a, tei_mo_aaaa, nocc_a)
    A_s_os_a = form_rpa_a_matrix_mo_singlet_os(tei_mo_aabb, nocc_a, nocc_b)
    B_s_ss_a = form_rpa_b_matrix_mo_singlet_ss(tei_mo_aaaa, nocc_a)
    B_s_os_a = form_rpa_b_matrix_mo_singlet_os(tei_mo_aabb, nocc_a, nocc_b)
    A_s_ss_b = form_rpa_a_matrix_mo_singlet_ss(E_b, tei_mo_bbbb, nocc_b)
    A_s_os_b = form_rpa_a_matrix_mo_singlet_os(tei_mo_bbaa, nocc_b, nocc_a)
    B_s_ss_b = form_rpa_b_matrix_mo_singlet_ss(tei_mo_bbbb, nocc_b)
    B_s_os_b = form_rpa_b_matrix_mo_singlet_os(tei_mo_bbaa, nocc_b, nocc_a)
    # Since the "triplet" part contains no Coulomb contribution, and
    # (xx|yy) is only in the Coulomb part, there is no ss/os
    # separation for the triplet part.

    G_aa = np.bmat([[A_s_ss_a, B_s_ss_a],
                    [B_s_ss_a, A_s_ss_a]])
    G_bb = np.bmat([[A_s_ss_b, B_s_ss_b],
                    [B_s_ss_b, A_s_ss_b]])
    G_ab = np.bmat([[A_s_os_a, B_s_os_a],
                    [B_s_os_a, A_s_os_a]])
    G_ba = np.bmat([[A_s_os_b, B_s_os_b],
                    [B_s_os_b, A_s_os_b]])

    G_aa_inv = np.linalg.inv(G_aa)
    G_ab_inv = np.linalg.pinv(G_ab)
    G_ba_inv = np.linalg.pinv(G_ba)
    G_bb_inv = np.linalg.inv(G_bb)

    nov_aa = nocc_a * nvirt_a
    nov_bb = nocc_b * nvirt_b

    assert G_aa_inv.shape == (2 * nov_aa, 2 * nov_aa)
    assert G_ab_inv.shape == (2 * nov_bb, 2 * nov_aa)
    assert G_ba_inv.shape == (2 * nov_aa, 2 * nov_bb)
    assert G_bb_inv.shape == (2 * nov_bb, 2 * nov_bb)

    # Form the operator_dependent part of the response vectors.
    left_a = np.linalg.inv(G_aa - np.dot(G_ab, np.dot(G_bb_inv, G_ba)))
    left_b = np.linalg.inv(G_bb - np.dot(G_ba, np.dot(G_aa_inv, G_ab)))

    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 5
    with open('water.xyz') as fh:
        mol.atom = fh.read()
    mol.unit = 'Bohr'
    mol.basis = 'sto-3g'
    mol.symmetry = False
    mol.build()

    integrals_dipole_ao = mol.intor('cint1e_r_sph', comp=3)

    integrals_dipole_mo_ai_a = []
    integrals_dipole_mo_ai_b = []

    for comp in range(3):

        integrals_dipole_mo_ai_comp_a = np.dot(C_a[:, nocc_a:].T, np.dot(integrals_dipole_ao[comp, ...], C_a[:, :nocc_a]))
        integrals_dipole_mo_ai_comp_b = np.dot(C_b[:, nocc_b:].T, np.dot(integrals_dipole_ao[comp, ...], C_b[:, :nocc_b]))

        integrals_dipole_mo_ai_comp_a = np.reshape(integrals_dipole_mo_ai_comp_a, -1, order='F')
        integrals_dipole_mo_ai_comp_b = np.reshape(integrals_dipole_mo_ai_comp_b, -1, order='F')

        integrals_dipole_mo_ai_a.append(integrals_dipole_mo_ai_comp_a)
        integrals_dipole_mo_ai_b.append(integrals_dipole_mo_ai_comp_b)

    integrals_dipole_mo_ai_a = np.stack(integrals_dipole_mo_ai_a, axis=0).T
    integrals_dipole_mo_ai_b = np.stack(integrals_dipole_mo_ai_b, axis=0).T

    integrals_dipole_mo_ai_a_super = np.concatenate((integrals_dipole_mo_ai_a, integrals_dipole_mo_ai_a), axis=0)
    integrals_dipole_mo_ai_b_super = np.concatenate((integrals_dipole_mo_ai_b, integrals_dipole_mo_ai_b), axis=0)

    # Form the operator-dependent part of the response vectors.
    right_a = integrals_dipole_mo_ai_a_super - (np.dot(G_ab, np.dot(G_bb_inv, integrals_dipole_mo_ai_b_super)))
    right_b = integrals_dipole_mo_ai_b_super - (np.dot(G_ba, np.dot(G_aa_inv, integrals_dipole_mo_ai_a_super)))

    # The total response vector for each spin is the product of the
    # operator-independent (left) and operator-dependent (right)
    # parts.
    rspvec_a = np.dot(left_a, right_a)
    rspvec_b = np.dot(left_b, right_b)

    res_a = np.dot(integrals_dipole_mo_ai_a_super.T, rspvec_a) / 2
    res_b = np.dot(integrals_dipole_mo_ai_b_super.T, rspvec_b) / 2
    res_u = 2 * (res_a + res_b)

    atol = 1.0e-8
    rtol = 0.0
    # np.testing.assert_allclose(res_u, res_r, rtol=rtol, atol=atol)

    # print(res_r)
    print(res_u)


if __name__ == '__main__':

    print(__name__)
    test_direct_uhf_from_rhf()
    test_direct_uhf()
