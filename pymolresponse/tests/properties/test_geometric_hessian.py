import os.path

import numpy as np

import psi4

from pymolresponse.ao2mo import AO2MO
from pymolresponse.core import AO2MOTransformationType, Hamiltonian, Program, Spin
from pymolresponse.cphf import CPHF
from pymolresponse.data import REFDIR
from pymolresponse.interfaces.psi4 import molecules
from pymolresponse.interfaces.psi4.utils import (
    mocoeffs_from_psi4wfn,
    moenergies_from_psi4wfn,
    occupations_from_psi4wfn,
)
from pymolresponse.operators import Operator
from pymolresponse.solvers import ExactInv

np.set_printoptions(precision=8, linewidth=200, suppress=True)


datadir = REFDIR / "psi4numpy" / "water"


def test_geometric_hessian_rhf_outside_solver_psi4numpy():
    psi4.core.set_output_file("output.dat", False)

    mol = psi4.geometry(
        """
    O
    H 1 1.1
    H 1 1.1 2 104
    symmetry c1
    """
    )

    psi4.core.set_active_molecule(mol)

    options = {"BASIS": "STO-3G", "SCF_TYPE": "PK", "E_CONVERGENCE": 1e-10, "D_CONVERGENCE": 1e-10}

    psi4.set_options(options)

    _, wfn = psi4.energy("SCF", return_wfn=True)

    # Assuming C1 symmetry
    occ = wfn.doccpi()[0]
    nmo = wfn.nmo()
    vir = nmo - occ

    C = wfn.Ca_subset("AO", "ALL")
    npC = np.asarray(C)

    mints = psi4.core.MintsHelper(wfn.basisset())
    H_ao = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())

    # Update H, transform to MO basis
    H = np.einsum("uj,vi,uv", npC, npC, H_ao)

    # Integral generation from Psi4's MintsHelper
    MO = np.asarray(mints.mo_eri(C, C, C, C))
    # Physicist notation
    MO = MO.swapaxes(1, 2)

    F = H + 2.0 * np.einsum("pmqm->pq", MO[:, :occ, :, :occ])
    F -= np.einsum("pmmq->pq", MO[:, :occ, :occ, :])
    # Uncomment every `np.save` call to regenerate reference data.
    # np.save(os.path.join(datadir, 'F.npy'), F)
    F_ref = np.load(os.path.join(datadir, "F.npy"))
    np.testing.assert_allclose(F, F_ref, rtol=0, atol=1.0e-10)
    natoms = mol.natom()
    cart = ["_X", "_Y", "_Z"]
    oei_dict = {"S": "OVERLAP", "T": "KINETIC", "V": "POTENTIAL"}

    deriv1_mat = {}
    deriv1 = {}

    # 1st Derivative of OEIs

    for atom in range(natoms):
        for key in oei_dict:
            deriv1_mat[key + str(atom)] = mints.mo_oei_deriv1(oei_dict[key], atom, C, C)
            for p in range(3):
                map_key = key + str(atom) + cart[p]
                deriv1[map_key] = np.asarray(deriv1_mat[key + str(atom)][p])
                # np.save(os.path.join(datadir, f'{map_key}.npy'), deriv1[map_key])
                deriv1_ref = np.load(os.path.join(datadir, f"{map_key}.npy"))
                np.testing.assert_allclose(deriv1[map_key], deriv1_ref, rtol=0, atol=1.0e-10)

    # 1st Derivative of TEIs

    for atom in range(natoms):
        string = "TEI" + str(atom)
        deriv1_mat[string] = mints.mo_tei_deriv1(atom, C, C, C, C)
        for p in range(3):
            map_key = string + cart[p]
            deriv1[map_key] = np.asarray(deriv1_mat[string][p])
            # np.save(os.path.join(datadir, f'{map_key}.npy'), deriv1[map_key])
            deriv1_ref = np.load(os.path.join(datadir, f"{map_key}.npy"))
            np.testing.assert_allclose(deriv1[map_key], deriv1_ref, rtol=0, atol=1.0e-10)

    Hes = {}
    deriv2_mat = {}
    deriv2 = {}

    Hes["S"] = np.zeros((3 * natoms, 3 * natoms))
    Hes["V"] = np.zeros((3 * natoms, 3 * natoms))
    Hes["T"] = np.zeros((3 * natoms, 3 * natoms))
    Hes["N"] = np.zeros((3 * natoms, 3 * natoms))
    Hes["J"] = np.zeros((3 * natoms, 3 * natoms))
    Hes["K"] = np.zeros((3 * natoms, 3 * natoms))
    Hes["R"] = np.zeros((3 * natoms, 3 * natoms))
    Hessian = np.zeros((3 * natoms, 3 * natoms))

    Hes["N"] = np.asarray(mol.nuclear_repulsion_energy_deriv2())

    psi4.core.print_out("\n\n")
    Mat = psi4.core.Matrix.from_array(Hes["N"])
    Mat.name = "NUCLEAR HESSIAN"
    Mat.print_out()

    # 2nd Derivative of OEIs

    for atom1 in range(natoms):
        for atom2 in range(atom1 + 1):
            for key in oei_dict:
                string = key + str(atom1) + str(atom2)
                deriv2_mat[string] = mints.mo_oei_deriv2(oei_dict[key], atom1, atom2, C, C)
                pq = 0
                for p in range(3):
                    for q in range(3):
                        map_key = string + cart[p] + cart[q]
                        deriv2[map_key] = np.asarray(deriv2_mat[string][pq])
                        # np.save(os.path.join(datadir, f'{map_key}.npy'), deriv2[map_key])
                        deriv2_ref = np.load(os.path.join(datadir, f"{map_key}.npy"))
                        np.testing.assert_allclose(
                            deriv2[map_key], deriv2_ref, rtol=0, atol=1.0e-10
                        )
                        pq = pq + 1
                        row = 3 * atom1 + p
                        col = 3 * atom2 + q
                        if key == "S":
                            Hes[key][row][col] = -2.0 * np.einsum(
                                "ii,ii->", F[:occ, :occ], deriv2[map_key][:occ, :occ]
                            )
                        else:
                            Hes[key][row][col] = 2.0 * np.einsum(
                                "ii->", deriv2[map_key][:occ, :occ]
                            )
                        Hes[key][col][row] = Hes[key][row][col]
                        Hes[key][col][row] = Hes[key][row][col]
                        # np.save(os.path.join(datadir, f'Hes_{map_key}.npy'), Hes[key])
                        Hes_ref = np.load(os.path.join(datadir, f"Hes_{map_key}.npy"))
                        np.testing.assert_allclose(Hes[key], Hes_ref, rtol=0, atol=1.0e-10)

    for key in Hes:
        Mat = psi4.core.Matrix.from_array(Hes[key])
        if key in oei_dict:
            Mat.name = oei_dict[key] + " HESSIAN"
            Mat.print_out()
            psi4.core.print_out("\n")

    # 2nd Derivative of TEIs

    for atom1 in range(natoms):
        for atom2 in range(atom1 + 1):
            string = "TEI" + str(atom1) + str(atom2)
            deriv2_mat[string] = mints.mo_tei_deriv2(atom1, atom2, C, C, C, C)
            pq = 0
            for p in range(3):
                for q in range(3):
                    map_key = string + cart[p] + cart[q]
                    deriv2[map_key] = np.asarray(deriv2_mat[string][pq])
                    # np.save(os.path.join(datadir, f'{map_key}.npy'), deriv2[map_key])
                    deriv2_ref = np.load(os.path.join(datadir, f"{map_key}.npy"))
                    np.testing.assert_allclose(deriv2[map_key], deriv2_ref, rtol=0, atol=1.0e-10)
                    pq = pq + 1
                    row = 3 * atom1 + p
                    col = 3 * atom2 + q
                    Hes["J"][row][col] = 2.0 * np.einsum(
                        "iijj->", deriv2[map_key][:occ, :occ, :occ, :occ]
                    )
                    Hes["K"][row][col] = -1.0 * np.einsum(
                        "ijij->", deriv2[map_key][:occ, :occ, :occ, :occ]
                    )

                    Hes["J"][col][row] = Hes["J"][row][col]
                    Hes["K"][col][row] = Hes["K"][row][col]
    for map_key in ("J", "K"):
        # np.save(os.path.join(datadir, f'Hes_{map_key}.npy'), Hes[map_key])
        Hes_ref = np.load(os.path.join(datadir, f"Hes_{map_key}.npy"))
        np.testing.assert_allclose(Hes[map_key], Hes_ref, rtol=0, atol=1.0e-10)

    JMat = psi4.core.Matrix.from_array(Hes["J"])
    KMat = psi4.core.Matrix.from_array(Hes["K"])
    JMat.name = " COULOMB  HESSIAN"
    KMat.name = " EXCHANGE HESSIAN"
    JMat.print_out()
    KMat.print_out()

    # Solve the CPHF equations here,  G_aibj Ubj^x = Bai^x (Einstein summation),
    # where G is the electronic hessian,
    # G_aibj = delta_ij * delta_ab * epsilon_ij * epsilon_ab + 4 <ij|ab> - <ij|ba> - <ia|jb>,
    # where epsilon_ij = epsilon_i - epsilon_j, (epsilon -> orbital energies),
    # x refers to the perturbation, Ubj^x are the corresponsing CPHF coefficients
    # and Bai^x = Sai^x * epsilon_ii - Fai^x + Smn^x  * (2<am|in> - <am|ni>),
    # where, S^x =  del(S)/del(x), F^x =  del(F)/del(x).

    I_occ = np.diag(np.ones(occ))
    I_vir = np.diag(np.ones(vir))
    epsilon = np.asarray(wfn.epsilon_a())
    eps_diag = epsilon[occ:].reshape(-1, 1) - epsilon[:occ]

    # Build the electronic hessian G

    G = 4 * MO[:occ, :occ, occ:, occ:]
    G -= MO[:occ, :occ, occ:, occ:].swapaxes(2, 3)
    G -= MO[:occ, occ:, :occ, occ:].swapaxes(1, 2)
    G = G.swapaxes(1, 2)
    G += np.einsum("ai,ij,ab->iajb", eps_diag, I_occ, I_vir)
    # np.save(os.path.join(datadir, 'G.npy'), G)
    G_ref = np.load(os.path.join(datadir, "G.npy"))
    np.testing.assert_allclose(G, G_ref, rtol=0, atol=1.0e-10)

    # Inverse of G
    Ginv = np.linalg.inv(G.reshape(occ * vir, -1))
    Ginv = Ginv.reshape(occ, vir, occ, vir)

    B = {}
    F_grad = {}
    U = {}

    # Build Fpq^x now

    for atom in range(natoms):
        for p in range(3):
            key = str(atom) + cart[p]
            F_grad[key] = deriv1["T" + key]
            F_grad[key] += deriv1["V" + key]
            F_grad[key] += 2.0 * np.einsum("pqmm->pq", deriv1["TEI" + key][:, :, :occ, :occ])
            F_grad[key] -= 1.0 * np.einsum("pmmq->pq", deriv1["TEI" + key][:, :occ, :occ, :])
            # np.save(os.path.join(datadir, f'F_grad_{key}.npy'), F_grad[key])
            F_grad_ref = np.load(os.path.join(datadir, f"F_grad_{key}.npy"))
            np.testing.assert_allclose(F_grad[key], F_grad_ref, rtol=0, atol=1.0e-10)

    psi4.core.print_out("\n\n CPHF Coefficients:\n")

    # Build Bai^x now

    for atom in range(natoms):
        for p in range(3):
            key = str(atom) + cart[p]
            B[key] = np.einsum("ai,ii->ai", deriv1["S" + key][occ:, :occ], F[:occ, :occ])
            B[key] -= F_grad[key][occ:, :occ]
            B[key] += 2.0 * np.einsum(
                "amin,mn->ai", MO[occ:, :occ, :occ, :occ], deriv1["S" + key][:occ, :occ]
            )
            B[key] += -1.0 * np.einsum(
                "amni,mn->ai", MO[occ:, :occ, :occ, :occ], deriv1["S" + key][:occ, :occ]
            )

            print(f"B[{key}]")
            print(B[key])
            # np.save(os.path.join(datadir, f'B_{key}.npy'), B[key])
            B_ref = np.load(os.path.join(datadir, f"B_{key}.npy"))
            np.testing.assert_allclose(B[key], B_ref, rtol=0, atol=1.0e-10)

            # Compute U^x now: U_ai^x = G^(-1)_aibj * B_bj^x

            U[key] = np.einsum("iajb,bj->ai", Ginv, B[key])
            psi4.core.print_out("\n")
            UMat = psi4.core.Matrix.from_array(U[key])
            UMat.name = key
            UMat.print_out()

            # np.save(os.path.join(datadir, f'U_{key}.npy'), U[key])
            U_ref = np.load(os.path.join(datadir, f"U_{key}.npy"))
            np.testing.assert_allclose(U[key], U_ref, rtol=0, atol=1.0e-10)

    # Build the response Hessian

    for atom1 in range(natoms):
        for atom2 in range(atom1 + 1):
            for p in range(3):
                for q in range(3):
                    key1 = str(atom1) + cart[p]
                    key2 = str(atom2) + cart[q]
                    key1S = "S" + key1
                    key2S = "S" + key2
                    r = 3 * atom1 + p
                    c = 3 * atom2 + q

                    Hes["R"][r][c] = -2.0 * np.einsum(
                        "ij,ij->", deriv1[key1S][:occ, :occ], F_grad[key2][:occ, :occ]
                    )
                    Hes["R"][r][c] -= 2.0 * np.einsum(
                        "ij,ij->", deriv1[key2S][:occ, :occ], F_grad[key1][:occ, :occ]
                    )
                    Hes["R"][r][c] += 4.0 * np.einsum(
                        "ii,mi,mi->",
                        F[:occ, :occ],
                        deriv1[key2S][:occ, :occ],
                        deriv1[key1S][:occ, :occ],
                    )

                    Hes["R"][r][c] += 4.0 * np.einsum(
                        "ij,mn,imjn->",
                        deriv1[key1S][:occ, :occ],
                        deriv1[key2S][:occ, :occ],
                        MO[:occ, :occ, :occ, :occ],
                    )
                    Hes["R"][r][c] -= 2.0 * np.einsum(
                        "ij,mn,imnj->",
                        deriv1[key1S][:occ, :occ],
                        deriv1[key2S][:occ, :occ],
                        MO[:occ, :occ, :occ, :occ],
                    )

                    Hes["R"][r][c] -= 4.0 * np.einsum("ai,ai->", U[key2], B[key1])
                    Hes["R"][c][r] = Hes["R"][r][c]

    # np.save(os.path.join(datadir, 'Hes_R.npy'), Hes["R"])
    Hes_ref = np.load(os.path.join(datadir, "Hes_R.npy"))
    np.testing.assert_allclose(Hes["R"], Hes_ref, rtol=0, atol=1.0e-10)

    Mat = psi4.core.Matrix.from_array(Hes["R"])
    Mat.name = " RESPONSE HESSIAN"
    Mat.print_out()

    for key in Hes:
        Hessian += Hes[key]

    # print('deriv1_mat')
    # print(deriv1_mat.keys())
    # print('deriv1')
    # print(deriv1.keys())
    # print('deriv2_mat')
    # print(deriv2_mat.keys())
    # print('deriv2')
    # print(deriv2.keys())
    # print('B')
    # print(B.keys())
    # print('F_grad')
    # print(F_grad.keys())
    # print('U')
    # print(U.keys())
    # print('Hes')
    # print(Hes.keys())

    Mat = psi4.core.Matrix.from_array(Hessian)
    Mat.name = " TOTAL HESSIAN"
    Mat.print_out()

    # pylint: disable=bad-whitespace
    H_psi4 = psi4.core.Matrix.from_list(
        [
            [
                7.613952269164418751e-02,
                0.000000000000000000e00,
                0.000000000000000000e00,
                -3.806976134297335168e-02,
                0.000000000000000000e00,
                0.000000000000000000e00,
                -3.806976134297410108e-02,
                0.000000000000000000e00,
                0.000000000000000000e00,
            ],
            [
                0.000000000000000000e00,
                4.829053723748517601e-01,
                0.000000000000000000e00,
                0.000000000000000000e00,
                -2.414526861845633920e-01,
                1.589001558536450587e-01,
                0.000000000000000000e00,
                -2.414526861845646133e-01,
                -1.589001558536444758e-01,
            ],
            [
                0.000000000000000000e00,
                0.000000000000000000e00,
                4.373449597848400039e-01,
                0.000000000000000000e00,
                7.344233774055003439e-02,
                -2.186724798895446076e-01,
                0.000000000000000000e00,
                -7.344233774054893804e-02,
                -2.186724798895475219e-01,
            ],
            [
                -3.806976134297335168e-02,
                0.000000000000000000e00,
                0.000000000000000000e00,
                4.537741758645107149e-02,
                0.000000000000000000e00,
                0.000000000000000000e00,
                -7.307656243475501093e-03,
                0.000000000000000000e00,
                0.000000000000000000e00,
            ],
            [
                0.000000000000000000e00,
                -2.414526861845633920e-01,
                7.344233774055003439e-02,
                0.000000000000000000e00,
                2.578650065921952450e-01,
                -1.161712467970963669e-01,
                0.000000000000000000e00,
                -1.641232040762596878e-02,
                4.272890905654690846e-02,
            ],
            [
                0.000000000000000000e00,
                1.589001558536450587e-01,
                -2.186724798895446076e-01,
                0.000000000000000000e00,
                -1.161712467970963669e-01,
                1.977519807685419462e-01,
                0.000000000000000000e00,
                -4.272890905654720684e-02,
                2.092049912100946499e-02,
            ],
            [
                -3.806976134297410108e-02,
                0.000000000000000000e00,
                0.000000000000000000e00,
                -7.307656243475501093e-03,
                0.000000000000000000e00,
                0.000000000000000000e00,
                4.537741758645268131e-02,
                0.000000000000000000e00,
                0.000000000000000000e00,
            ],
            [
                0.000000000000000000e00,
                -2.414526861845646133e-01,
                -7.344233774054893804e-02,
                0.000000000000000000e00,
                -1.641232040762596878e-02,
                -4.272890905654720684e-02,
                0.000000000000000000e00,
                2.578650065921969103e-01,
                1.161712467970957008e-01,
            ],
            [
                0.000000000000000000e00,
                -1.589001558536444758e-01,
                -2.186724798895475219e-01,
                0.000000000000000000e00,
                4.272890905654690846e-02,
                2.092049912100946499e-02,
                0.000000000000000000e00,
                1.161712467970957008e-01,
                1.977519807685442221e-01,
            ],
        ]
    )
    H_python_mat = psi4.core.Matrix.from_array(Hessian)
    psi4.compare_matrices(H_psi4, H_python_mat, 10, "RHF-HESSIAN-TEST")  # TEST


def test_geometric_hessian_rhf_outside_solver_chemists():
    psi4.core.set_output_file("output2.dat", False)

    mol = molecules.molecule_physicists_water_sto3g()
    mol.reset_point_group("c1")
    mol.update_geometry()
    psi4.core.set_active_molecule(mol)

    options = {"BASIS": "STO-3G", "SCF_TYPE": "PK", "E_CONVERGENCE": 1e-10, "D_CONVERGENCE": 1e-10}

    psi4.set_options(options)

    _, wfn = psi4.energy("hf", return_wfn=True)

    norb = wfn.nmo()
    nocc = wfn.nalpha()
    nvir = norb - nocc

    o = slice(0, nocc)
    v = slice(nocc, norb)

    C = wfn.Ca_subset("AO", "ALL")
    npC = np.asarray(C)

    mints = psi4.core.MintsHelper(wfn)
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    H_ao = T + V

    H = np.einsum("up,vq,uv->pq", npC, npC, H_ao)

    MO = np.asarray(mints.mo_eri(C, C, C, C))

    F = H + 2.0 * np.einsum("pqii->pq", MO[:, :, o, o])
    F -= np.einsum("piqi->pq", MO[:, o, :, o])
    F_ref = np.load(os.path.join(datadir, "F.npy"))
    np.testing.assert_allclose(F, F_ref, rtol=0, atol=1.0e-10)
    natoms = mol.natom()
    cart = ["_X", "_Y", "_Z"]
    oei_dict = {"S": "OVERLAP", "T": "KINETIC", "V": "POTENTIAL"}

    deriv1_mat = {}
    deriv1 = {}

    # 1st Derivative of OEIs

    for atom in range(natoms):
        for key in oei_dict:
            deriv1_mat[key + str(atom)] = mints.mo_oei_deriv1(oei_dict[key], atom, C, C)
            for p in range(3):
                map_key = key + str(atom) + cart[p]
                deriv1[map_key] = np.asarray(deriv1_mat[key + str(atom)][p])
                deriv1_ref = np.load(os.path.join(datadir, f"{map_key}.npy"))
                np.testing.assert_allclose(deriv1[map_key], deriv1_ref, rtol=0, atol=1.0e-10)

    # 1st Derivative of TEIs

    for atom in range(natoms):
        string = "TEI" + str(atom)
        deriv1_mat[string] = mints.mo_tei_deriv1(atom, C, C, C, C)
        for p in range(3):
            map_key = string + cart[p]
            deriv1[map_key] = np.asarray(deriv1_mat[string][p])
            deriv1_ref = np.load(os.path.join(datadir, f"{map_key}.npy"))
            np.testing.assert_allclose(deriv1[map_key], deriv1_ref, rtol=0, atol=1.0e-10)

    Hes = {}
    deriv2_mat = {}
    deriv2 = {}

    Hes["S"] = np.zeros((3 * natoms, 3 * natoms))
    Hes["V"] = np.zeros((3 * natoms, 3 * natoms))
    Hes["T"] = np.zeros((3 * natoms, 3 * natoms))
    Hes["N"] = np.zeros((3 * natoms, 3 * natoms))
    Hes["J"] = np.zeros((3 * natoms, 3 * natoms))
    Hes["K"] = np.zeros((3 * natoms, 3 * natoms))
    Hes["R"] = np.zeros((3 * natoms, 3 * natoms))
    Hessian = np.zeros((3 * natoms, 3 * natoms))

    Hes["N"] = np.asarray(mol.nuclear_repulsion_energy_deriv2())

    psi4.core.print_out("\n\n")
    Mat = psi4.core.Matrix.from_array(Hes["N"])
    Mat.name = "NUCLEAR HESSIAN"
    Mat.print_out()

    # 2nd Derivative of OEIs

    for atom1 in range(natoms):
        for atom2 in range(atom1 + 1):
            for key in oei_dict:
                string = key + str(atom1) + str(atom2)
                deriv2_mat[string] = mints.mo_oei_deriv2(oei_dict[key], atom1, atom2, C, C)
                pq = 0
                for p in range(3):
                    for q in range(3):
                        map_key = string + cart[p] + cart[q]
                        deriv2[map_key] = np.asarray(deriv2_mat[string][pq])
                        deriv2_ref = np.load(os.path.join(datadir, f"{map_key}.npy"))
                        np.testing.assert_allclose(
                            deriv2[map_key], deriv2_ref, rtol=0, atol=1.0e-10
                        )
                        pq = pq + 1
                        row = 3 * atom1 + p
                        col = 3 * atom2 + q
                        if key == "S":
                            Hes[key][row][col] = -2.0 * np.einsum(
                                "ii,ii->", F[o, o], deriv2[map_key][o, o]
                            )
                        else:
                            Hes[key][row][col] = 2.0 * np.einsum("ii->", deriv2[map_key][o, o])
                        Hes[key][col][row] = Hes[key][row][col]
                        Hes[key][col][row] = Hes[key][row][col]
                        Hes_ref = np.load(os.path.join(datadir, f"Hes_{map_key}.npy"))
                        np.testing.assert_allclose(Hes[key], Hes_ref, rtol=0, atol=1.0e-10)

    for key in Hes:
        Mat = psi4.core.Matrix.from_array(Hes[key])
        if key in oei_dict:
            Mat.name = oei_dict[key] + " HESSIAN"
            Mat.print_out()
            psi4.core.print_out("\n")

    # 2nd Derivative of TEIs

    for atom1 in range(natoms):
        for atom2 in range(atom1 + 1):
            string = "TEI" + str(atom1) + str(atom2)
            deriv2_mat[string] = mints.mo_tei_deriv2(atom1, atom2, C, C, C, C)
            pq = 0
            for p in range(3):
                for q in range(3):
                    map_key = string + cart[p] + cart[q]
                    deriv2[map_key] = np.asarray(deriv2_mat[string][pq])
                    deriv2_ref = np.load(os.path.join(datadir, f"{map_key}.npy"))
                    np.testing.assert_allclose(deriv2[map_key], deriv2_ref, rtol=0, atol=1.0e-10)
                    pq = pq + 1
                    row = 3 * atom1 + p
                    col = 3 * atom2 + q
                    Hes["J"][row][col] = 2.0 * np.einsum("iijj->", deriv2[map_key][o, o, o, o])
                    Hes["K"][row][col] = -1.0 * np.einsum("ijij->", deriv2[map_key][o, o, o, o])

                    Hes["J"][col][row] = Hes["J"][row][col]
                    Hes["K"][col][row] = Hes["K"][row][col]
    for map_key in ("J", "K"):
        Hes_ref = np.load(os.path.join(datadir, f"Hes_{map_key}.npy"))
        np.testing.assert_allclose(Hes[map_key], Hes_ref, rtol=0, atol=1.0e-10)

    JMat = psi4.core.Matrix.from_array(Hes["J"])
    KMat = psi4.core.Matrix.from_array(Hes["K"])
    JMat.name = " COULOMB  HESSIAN"
    KMat.name = " EXCHANGE HESSIAN"
    JMat.print_out()
    KMat.print_out()

    # Solve the CPHF equations here,  G_iajb U_jb^x = B_ia^x (Einstein summation),
    # where G is the electronic hessian,
    # G_iajb = delta_ij * delta_ab * epsilon_ij * epsilon_ab + 4(ia|jb) - (ij|ab) - (ib|ja),
    # where epsilon_ij = epsilon_i - epsilon_j, (epsilon -> orbital energies),
    # x refers to the perturbation, U_jb^x are the corresponsing CPHF coefficients
    # and B_ia^x = S_ia^x * epsilon_ii - F_ia^x + S_mn^x * [2(ia|mn) - (in|ma)],
    # where S^x = del(S)/del(x), F^x =  del(F)/del(x).

    I_occ = np.diag(np.ones(nocc))
    I_vir = np.diag(np.ones(nvir))
    epsilon = np.asarray(wfn.epsilon_a())
    eps_diag = epsilon[v].reshape(-1, 1) - epsilon[o]

    #  Build the electronic hessian G

    G = 4 * MO[o, v, o, v]
    G -= MO[o, o, v, v].swapaxes(1, 2)
    G -= MO[o, v, o, v].swapaxes(1, 3)
    G += np.einsum("ai,ij,ab->iajb", eps_diag, I_occ, I_vir)
    G_ref = np.load(os.path.join(datadir, "G.npy"))
    np.testing.assert_allclose(G, G_ref, rtol=0, atol=1.0e-10)

    # Inverse of G
    Ginv = np.linalg.inv(G.reshape(nocc * nvir, -1))
    Ginv = Ginv.reshape(nocc, nvir, nocc, nvir)

    B = {}
    F_grad = {}
    U = {}

    # Build F_pq^x now

    for atom in range(natoms):
        for p in range(3):
            key = str(atom) + cart[p]
            F_grad[key] = deriv1["T" + key]
            F_grad[key] += deriv1["V" + key]
            F_grad[key] += 2.0 * np.einsum("pqmm->pq", deriv1["TEI" + key][:, :, o, o])
            F_grad[key] -= 1.0 * np.einsum("pmmq->pq", deriv1["TEI" + key][:, o, o, :])
            F_grad_ref = np.load(os.path.join(datadir, f"F_grad_{key}.npy"))
            np.testing.assert_allclose(F_grad[key], F_grad_ref, rtol=0, atol=1.0e-10)

    psi4.core.print_out("\n\n CPHF Coefficients:\n")

    # Build B_ia^x now

    for atom in range(natoms):
        for p in range(3):
            key = str(atom) + cart[p]
            B[key] = np.einsum("ia,ii->ia", deriv1["S" + key][o, v], F[o, o])
            B[key] -= F_grad[key][o, v]
            B[key] += 2.0 * np.einsum("iamn,mn->ia", MO[o, v, o, o], deriv1["S" + key][o, o])
            B[key] += -1.0 * np.einsum("inma,mn->ia", MO[o, o, o, v], deriv1["S" + key][o, o])

            print(f"B[{key}]")
            print(B[key])
            B_ref = np.load(os.path.join(datadir, f"B_{key}.npy"))
            np.testing.assert_allclose(B[key], B_ref.T, rtol=0, atol=1.0e-10)

            # Compute U^x now: U_ia^x = G^(-1)_iajb * B_jb^x

            U[key] = np.einsum("iajb,jb->ia", Ginv, B[key])
            psi4.core.print_out("\n")
            UMat = psi4.core.Matrix.from_array(U[key])
            UMat.name = key
            UMat.print_out()

            U_ref = np.load(os.path.join(datadir, f"U_{key}.npy"))
            np.testing.assert_allclose(U[key], U_ref.T, rtol=0, atol=1.0e-10)

    # Build the response Hessian

    for atom1 in range(natoms):
        for atom2 in range(atom1 + 1):
            for p in range(3):
                for q in range(3):
                    key1 = str(atom1) + cart[p]
                    key2 = str(atom2) + cart[q]
                    key1S = "S" + key1
                    key2S = "S" + key2
                    r = 3 * atom1 + p
                    c = 3 * atom2 + q

                    Hes["R"][r][c] = -2.0 * np.einsum(
                        "ij,ij->", deriv1[key1S][o, o], F_grad[key2][o, o]
                    )
                    Hes["R"][r][c] -= 2.0 * np.einsum(
                        "ij,ij->", deriv1[key2S][o, o], F_grad[key1][o, o]
                    )
                    Hes["R"][r][c] += 4.0 * np.einsum(
                        "ii,mi,mi->", F[o, o], deriv1[key2S][o, o], deriv1[key1S][o, o]
                    )

                    Hes["R"][r][c] += 4.0 * np.einsum(
                        "ij,mn,ijmn->", deriv1[key1S][o, o], deriv1[key2S][o, o], MO[o, o, o, o]
                    )
                    Hes["R"][r][c] -= 2.0 * np.einsum(
                        "ij,mn,inmj->", deriv1[key1S][o, o], deriv1[key2S][o, o], MO[o, o, o, o]
                    )

                    Hes["R"][r][c] -= 4.0 * np.einsum("ia,ia->", U[key2], B[key1])
                    Hes["R"][c][r] = Hes["R"][r][c]

    Hes_ref = np.load(os.path.join(datadir, "Hes_R.npy"))
    np.testing.assert_allclose(Hes["R"], Hes_ref, rtol=0, atol=1.0e-10)

    Mat = psi4.core.Matrix.from_array(Hes["R"])
    Mat.name = " RESPONSE HESSIAN"
    Mat.print_out()

    for key in Hes:
        Hessian += Hes[key]

    # print('deriv1_mat')
    # print(deriv1_mat.keys())
    # print('deriv1')
    # print(deriv1.keys())
    # print('deriv2_mat')
    # print(deriv2_mat.keys())
    # print('deriv2')
    # print(deriv2.keys())
    # print('B')
    # print(B.keys())
    # print('F_grad')
    # print(F_grad.keys())
    # print('U')
    # print(U.keys())
    # print('Hes')
    # print(Hes.keys())

    Mat = psi4.core.Matrix.from_array(Hessian)
    Mat.name = " TOTAL HESSIAN"
    Mat.print_out()

    # pylint: disable=bad-whitespace
    H_psi4 = psi4.core.Matrix.from_list(
        [
            [
                7.613952269164418751e-02,
                0.000000000000000000e00,
                0.000000000000000000e00,
                -3.806976134297335168e-02,
                0.000000000000000000e00,
                0.000000000000000000e00,
                -3.806976134297410108e-02,
                0.000000000000000000e00,
                0.000000000000000000e00,
            ],
            [
                0.000000000000000000e00,
                4.829053723748517601e-01,
                0.000000000000000000e00,
                0.000000000000000000e00,
                -2.414526861845633920e-01,
                1.589001558536450587e-01,
                0.000000000000000000e00,
                -2.414526861845646133e-01,
                -1.589001558536444758e-01,
            ],
            [
                0.000000000000000000e00,
                0.000000000000000000e00,
                4.373449597848400039e-01,
                0.000000000000000000e00,
                7.344233774055003439e-02,
                -2.186724798895446076e-01,
                0.000000000000000000e00,
                -7.344233774054893804e-02,
                -2.186724798895475219e-01,
            ],
            [
                -3.806976134297335168e-02,
                0.000000000000000000e00,
                0.000000000000000000e00,
                4.537741758645107149e-02,
                0.000000000000000000e00,
                0.000000000000000000e00,
                -7.307656243475501093e-03,
                0.000000000000000000e00,
                0.000000000000000000e00,
            ],
            [
                0.000000000000000000e00,
                -2.414526861845633920e-01,
                7.344233774055003439e-02,
                0.000000000000000000e00,
                2.578650065921952450e-01,
                -1.161712467970963669e-01,
                0.000000000000000000e00,
                -1.641232040762596878e-02,
                4.272890905654690846e-02,
            ],
            [
                0.000000000000000000e00,
                1.589001558536450587e-01,
                -2.186724798895446076e-01,
                0.000000000000000000e00,
                -1.161712467970963669e-01,
                1.977519807685419462e-01,
                0.000000000000000000e00,
                -4.272890905654720684e-02,
                2.092049912100946499e-02,
            ],
            [
                -3.806976134297410108e-02,
                0.000000000000000000e00,
                0.000000000000000000e00,
                -7.307656243475501093e-03,
                0.000000000000000000e00,
                0.000000000000000000e00,
                4.537741758645268131e-02,
                0.000000000000000000e00,
                0.000000000000000000e00,
            ],
            [
                0.000000000000000000e00,
                -2.414526861845646133e-01,
                -7.344233774054893804e-02,
                0.000000000000000000e00,
                -1.641232040762596878e-02,
                -4.272890905654720684e-02,
                0.000000000000000000e00,
                2.578650065921969103e-01,
                1.161712467970957008e-01,
            ],
            [
                0.000000000000000000e00,
                -1.589001558536444758e-01,
                -2.186724798895475219e-01,
                0.000000000000000000e00,
                4.272890905654690846e-02,
                2.092049912100946499e-02,
                0.000000000000000000e00,
                1.161712467970957008e-01,
                1.977519807685442221e-01,
            ],
        ]
    )

    H_python_mat = psi4.core.Matrix.from_array(Hessian)
    psi4.compare_matrices(H_psi4, H_python_mat, 10, "RHF-HESSIAN-TEST")  # TEST


def test_geometric_hessian_rhf_right_hand_side():
    mol = molecules.molecule_physicists_water_sto3g()
    mol.reset_point_group("c1")
    mol.update_geometry()
    psi4.core.set_active_molecule(mol)

    options = {"BASIS": "STO-3G", "SCF_TYPE": "PK", "E_CONVERGENCE": 1e-10, "D_CONVERGENCE": 1e-10}

    psi4.set_options(options)

    _, wfn = psi4.energy("hf", return_wfn=True)

    occupations = occupations_from_psi4wfn(wfn)
    nocc, nvir, _, _ = occupations
    norb = nocc + nvir

    o = slice(0, nocc)
    v = slice(nocc, norb)

    C = wfn.Ca_subset("AO", "ALL")
    npC = np.asarray(C)

    mints = psi4.core.MintsHelper(wfn)
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    H_ao = T + V

    H = np.einsum("up,vq,uv->pq", npC, npC, H_ao)

    MO = np.asarray(mints.mo_eri(C, C, C, C))

    F = H + 2.0 * np.einsum("pqii->pq", MO[:, :, o, o])
    F -= np.einsum("piqi->pq", MO[:, o, :, o])
    F_ref = np.load(os.path.join(datadir, "F.npy"))
    np.testing.assert_allclose(F, F_ref, rtol=0, atol=1.0e-10)
    natoms = mol.natom()
    cart = ["_X", "_Y", "_Z"]
    oei_dict = {"S": "OVERLAP", "T": "KINETIC", "V": "POTENTIAL"}

    deriv1 = dict()

    # 1st Derivative of OEIs

    for atom in range(natoms):
        for key in oei_dict:
            deriv1_mat = mints.mo_oei_deriv1(oei_dict[key], atom, C, C)
            for p in range(3):
                map_key = key + str(atom) + cart[p]
                deriv1[map_key] = np.asarray(deriv1_mat[p])
                deriv1_ref = np.load(os.path.join(datadir, f"{map_key}.npy"))
                np.testing.assert_allclose(deriv1[map_key], deriv1_ref, rtol=0, atol=1.0e-10)

    # 1st Derivative of TEIs

    for atom in range(natoms):
        string = "TEI" + str(atom)
        deriv1_mat = mints.mo_tei_deriv1(atom, C, C, C, C)
        for p in range(3):
            map_key = string + cart[p]
            deriv1[map_key] = np.asarray(deriv1_mat[p])
            deriv1_ref = np.load(os.path.join(datadir, f"{map_key}.npy"))
            np.testing.assert_allclose(deriv1[map_key], deriv1_ref, rtol=0, atol=1.0e-10)

    # B_ia^x = S_ia^x * epsilon_ii - F_ia^x + S_mn^x * [2(ia|mn) - (in|ma)]

    F_grad = dict()
    B = dict()

    # Build F_pq^x now

    for atom in range(natoms):
        for p in range(3):
            key = str(atom) + cart[p]
            F_grad[key] = deriv1["T" + key]
            F_grad[key] += deriv1["V" + key]
            F_grad[key] += 2.0 * np.einsum("pqmm->pq", deriv1["TEI" + key][:, :, o, o])
            F_grad[key] -= 1.0 * np.einsum("pmmq->pq", deriv1["TEI" + key][:, o, o, :])
            F_grad_ref = np.load(os.path.join(datadir, f"F_grad_{key}.npy"))
            np.testing.assert_allclose(F_grad[key], F_grad_ref, rtol=0, atol=1.0e-10)

    # Build B_ia^x now

    for atom in range(natoms):
        for p in range(3):
            key = str(atom) + cart[p]
            B[key] = np.einsum("ia,ii->ia", deriv1["S" + key][o, v], F[o, o])
            B[key] -= F_grad[key][o, v]
            B[key] += 2.0 * np.einsum("iamn,mn->ia", MO[o, v, o, o], deriv1["S" + key][o, o])
            B[key] += -1.0 * np.einsum("inma,mn->ia", MO[o, o, o, v], deriv1["S" + key][o, o])

            B_ref = np.load(os.path.join(datadir, f"B_{key}.npy"))
            np.testing.assert_allclose(B[key], B_ref.T, rtol=0, atol=1.0e-10)

    from pymolresponse.integrals import _form_rhs_geometric

    B_func = _form_rhs_geometric(npC, occupations, natoms, MO, mints)
    assert B_func.keys() == B.keys()
    for k in B_func:
        np.testing.assert_allclose(B_func[k], B[k], rtol=0, atol=1.0e-12)

    return B_func


def test_atomic_polar_tensor_rhf():
    mol = molecules.molecule_physicists_water_sto3g()
    mol.reset_point_group("c1")
    mol.update_geometry()
    psi4.core.set_active_molecule(mol)

    options = {"BASIS": "STO-3G", "SCF_TYPE": "PK", "E_CONVERGENCE": 1e-10, "D_CONVERGENCE": 1e-10}

    psi4.set_options(options)

    _, wfn = psi4.energy("hf", return_wfn=True)
    mints = psi4.core.MintsHelper(wfn)

    C = mocoeffs_from_psi4wfn(wfn)
    E = moenergies_from_psi4wfn(wfn)
    occupations = occupations_from_psi4wfn(wfn)
    nocc, nvir, _, _ = occupations
    norb = nocc + nvir

    # electric perturbation part
    ao2mo = AO2MO(C, occupations, I=np.asarray(mints.ao_eri()))
    ao2mo.perform_rhf_full()
    solver = ExactInv(C, E, occupations)
    solver.tei_mo = ao2mo.tei_mo
    solver.tei_mo_type = AO2MOTransformationType.full
    driver = CPHF(solver)
    operator_diplen = Operator(
        label="dipole", is_imaginary=False, is_spin_dependent=False, triplet=False
    )
    # integrals_diplen_ao = self.pyscfmol.intor('cint1e_r_sph', comp=3)
    M = np.stack([np.asarray(Mc) for Mc in mints.ao_dipole()])
    operator_diplen.ao_integrals = M
    driver.add_operator(operator_diplen)

    # geometric perturbation part
    operator_geometric = Operator(
        label="nuclear", is_imaginary=False, is_spin_dependent=False, triplet=False
    )
    operator_geometric.form_rhs_geometric(C, occupations, mol.natom(), solver.tei_mo[0], mints)
    print(operator_geometric.label)
    print(operator_geometric.mo_integrals_ai_alph)
    print(operator_diplen.label)
    print(operator_diplen.mo_integrals_ai_alph)
    # hack for dim check in solver
    operator_geometric.ao_integrals = np.zeros((3 * mol.natom(), M.shape[1], M.shape[2]))
    # bypass driver's call to form_rhs
    driver.solver.operators.append(operator_geometric)

    driver.run(
        hamiltonian=Hamiltonian.RPA, spin=Spin.singlet, program=Program.Psi4, program_obj=wfn
    )
    print(driver.results[0])
    print(driver.results[0].T)
    print(driver.results[0] - driver.results[0].T)
    print(operator_geometric.rspvecs_alph[0])
    # Nuclear contribution to dipole gradient
    # Electronic contributions to static part of dipole gradient
    # Reorthonormalization part of dipole gradient
    # Static contribution to dipole gradient
    # Relaxation part of dipole gradient
    # Total dipole gradient - TRAROT

    print("Nuclear contribution to dipole gradient")
    natom = mol.natom()
    Z = np.asarray([mol.Z(i) for i in range(natom)])
    nuclear_contrib = np.concatenate([np.diag(Z.take(3 * [i])) for i in range(natom)])
    print(nuclear_contrib)

    return locals()


if __name__ == "__main__":
    test_geometric_hessian_rhf_outside_solver_psi4numpy()
    test_geometric_hessian_rhf_outside_solver_chemists()
    test_geometric_hessian_rhf_right_hand_side()
    var = test_atomic_polar_tensor_rhf()
