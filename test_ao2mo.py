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
                    B[ia, jb] = TEI_MO[a + nocc_x, i, b + nocc_y, j]

    return B


if __name__ == '__main__':

    C_a = np_load('C.npz')
    C_b = C_a
    E_a = np_load('F_MO.npz')
    E_b = E_a
    TEI_MO = np_load('TEI_MO.npz')
    nocc_a = 5
    nvirt_a = C_a.shape[1] - nocc_a
    nocc_b = nocc_a
    nvirt_b = nvirt_a

    # Same-spin and opposite-spin contributions should add together
    # properly for restricted wavefunction.
    A_s = form_rpa_a_matrix_mo_singlet(E_a, TEI_MO, nocc_a)
    A_s_ss = form_rpa_a_matrix_mo_singlet_ss(E_a, TEI_MO, nocc_a)
    A_s_os = form_rpa_a_matrix_mo_singlet_os(TEI_MO, nocc_a, nocc_b)
    np.testing.assert_allclose(A_s, A_s_ss + A_s_os)
    B_s = form_rpa_b_matrix_mo_singlet(TEI_MO, nocc_a)
    B_s_ss = form_rpa_b_matrix_mo_singlet_ss(TEI_MO, nocc_a)
    B_s_os = form_rpa_b_matrix_mo_singlet_os(TEI_MO, nocc_a, nocc_b)
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
    integrals_dipole_mo_ai_aa = []
    integrals_dipole_mo_ai_ab = []
    integrals_dipole_mo_ai_ba = []
    integrals_dipole_mo_ai_bb = []

    for comp in range(3):

        integrals_dipole_mo_ai_comp_r = np.dot(C_a[:, nocc_a:].T, np.dot(integrals_dipole_ao[comp, ...], C_a[:, :nocc_a]))
        integrals_dipole_mo_ai_comp_aa = np.dot(C_a[:, nocc_a:].T, np.dot(integrals_dipole_ao[comp, ...], C_a[:, :nocc_a]))
        integrals_dipole_mo_ai_comp_ab = np.dot(C_a[:, nocc_a:].T, np.dot(integrals_dipole_ao[comp, ...], C_b[:, :nocc_b]))
        integrals_dipole_mo_ai_comp_ba = np.dot(C_b[:, nocc_b:].T, np.dot(integrals_dipole_ao[comp, ...], C_a[:, :nocc_a]))
        integrals_dipole_mo_ai_comp_bb = np.dot(C_b[:, nocc_b:].T, np.dot(integrals_dipole_ao[comp, ...], C_b[:, :nocc_b]))

        integrals_dipole_mo_ai_comp_r = np.reshape(integrals_dipole_mo_ai_comp_r, -1, order='F')
        integrals_dipole_mo_ai_comp_aa = np.reshape(integrals_dipole_mo_ai_comp_aa, -1, order='F')
        integrals_dipole_mo_ai_comp_ab = np.reshape(integrals_dipole_mo_ai_comp_ab, -1, order='F')
        integrals_dipole_mo_ai_comp_ba = np.reshape(integrals_dipole_mo_ai_comp_ba, -1, order='F')
        integrals_dipole_mo_ai_comp_bb = np.reshape(integrals_dipole_mo_ai_comp_bb, -1, order='F')

        integrals_dipole_mo_ai_r.append(integrals_dipole_mo_ai_comp_r)
        integrals_dipole_mo_ai_aa.append(integrals_dipole_mo_ai_comp_aa)
        integrals_dipole_mo_ai_ab.append(integrals_dipole_mo_ai_comp_ab)
        integrals_dipole_mo_ai_ba.append(integrals_dipole_mo_ai_comp_ba)
        integrals_dipole_mo_ai_bb.append(integrals_dipole_mo_ai_comp_bb)

    integrals_dipole_mo_ai_r = np.stack(integrals_dipole_mo_ai_r, axis=0).T
    integrals_dipole_mo_ai_aa = np.stack(integrals_dipole_mo_ai_aa, axis=0).T
    integrals_dipole_mo_ai_ab = np.stack(integrals_dipole_mo_ai_ab, axis=0).T
    integrals_dipole_mo_ai_ba = np.stack(integrals_dipole_mo_ai_ba, axis=0).T
    integrals_dipole_mo_ai_bb = np.stack(integrals_dipole_mo_ai_bb, axis=0).T

    integrals_dipole_mo_ai_r_super = np.concatenate((integrals_dipole_mo_ai_r, integrals_dipole_mo_ai_r), axis=0)
    integrals_dipole_mo_ai_aa_super = np.concatenate((integrals_dipole_mo_ai_aa, integrals_dipole_mo_ai_aa), axis=0)
    integrals_dipole_mo_ai_ab_super = np.concatenate((integrals_dipole_mo_ai_ab, integrals_dipole_mo_ai_ab), axis=0)
    integrals_dipole_mo_ai_ba_super = np.concatenate((integrals_dipole_mo_ai_ba, integrals_dipole_mo_ai_ba), axis=0)
    integrals_dipole_mo_ai_bb_super = np.concatenate((integrals_dipole_mo_ai_bb, integrals_dipole_mo_ai_bb), axis=0)

    rspvec_r = np.dot(G_r_inv, integrals_dipole_mo_ai_r_super)
    rspvec_aa = np.dot(G_aa_inv, integrals_dipole_mo_ai_aa_super)
    rspvec_ab = np.dot(G_ab_inv, integrals_dipole_mo_ai_ab_super)
    rspvec_ba = np.dot(G_ba_inv, integrals_dipole_mo_ai_ba_super)
    rspvec_bb = np.dot(G_bb_inv, integrals_dipole_mo_ai_bb_super)
    res_r = 4 * np.dot(integrals_dipole_mo_ai_r_super.T, rspvec_r) / 2
    res_aa = np.dot(integrals_dipole_mo_ai_aa_super.T, rspvec_aa) / 2
    res_ab = np.dot(integrals_dipole_mo_ai_ab_super.T, rspvec_ab) / 2
    res_ba = np.dot(integrals_dipole_mo_ai_ba_super.T, rspvec_ba) / 2
    res_bb = np.dot(integrals_dipole_mo_ai_bb_super.T, rspvec_bb) / 2
    print(res_r)
    print(res_aa)
    print(res_ab)
    print(res_ba)
    print(res_bb)
