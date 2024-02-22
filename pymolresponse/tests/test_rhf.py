"""Hard-coded response equations for restricted wavefunctions."""

import numpy as np

import pyscf

from pymolresponse import explicit_equations_full as eqns
from pymolresponse.interfaces.pyscf import molecules, utils
from pymolresponse.interfaces.pyscf.ao2mo import AO2MOpyscf


def test_explicit_rhf_outside_solver_off_diagonal_blocks():
    mol = molecules.molecule_water_sto3g()
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    mocoeffs = mf.mo_coeff
    moenergies = mf.mo_energy
    ao2mo = AO2MOpyscf(mocoeffs, mol.verbose, mol)
    ao2mo.perform_rhf_full()
    tei_mo = ao2mo.tei_mo[0]

    C = mocoeffs
    E = np.diag(moenergies)
    occupations = utils.occupations_from_pyscf_mol(mol, mocoeffs)
    nocc, nvirt, _, _ = occupations

    A = eqns.form_rpa_a_matrix_mo_singlet_full(E, tei_mo, nocc)
    B = eqns.form_rpa_b_matrix_mo_singlet_full(tei_mo, nocc)

    G = np.block([[A, B], [B, A]])
    assert G.shape == (2 * nocc * nvirt, 2 * nocc * nvirt)

    G_inv = np.linalg.inv(G)

    components = 3

    integrals_dipole_ao = mol.intor("cint1e_r_sph", comp=components)

    integrals_dipole_mo_ai = []

    for component in range(components):
        integrals_dipole_mo_ai_component = np.dot(
            C[:, nocc:].T, np.dot(integrals_dipole_ao[component, ...], C[:, :nocc])
        ).reshape(-1, order="F")
        integrals_dipole_mo_ai.append(integrals_dipole_mo_ai_component)

    integrals_dipole_mo_ai = np.stack(integrals_dipole_mo_ai, axis=0).T
    integrals_dipole_mo_ai_super = np.concatenate(
        (integrals_dipole_mo_ai, -integrals_dipole_mo_ai), axis=0
    )
    rhsvecs = integrals_dipole_mo_ai_super
    rspvecs = np.dot(G_inv, rhsvecs)

    polarizability = 4 * np.dot(rhsvecs.T, rspvecs) / 2

    # pylint: disable=bad-whitespace
    result__0_00 = np.array(
        [[7.93556221, 0.0, 0.0], [0.0, 3.06821077, 0.0], [0.0, 0.0, 0.05038621]]
    )

    # TODO originally 1.0e-8
    atol = 1.5e-7
    rtol = 0.0
    np.testing.assert_allclose(polarizability, result__0_00, rtol=rtol, atol=atol)


# def test_explicit_rhf_outside_solver_all_blocks():
#     mol = molecules.molecule_water_sto3g()
#     mol.build()

#     mf = pyscf.scf.RHF(mol)
#     mf.kernel()
#     mocoeffs = mf.mo_coeff
#     moenergies = mf.mo_energy
#     ao2mo = AO2MOpyscf(mocoeffs, mol.verbose, mol)
#     ao2mo.perform_rhf_full()
#     tei_mo = ao2mo.tei_mo[0]

#     C = mocoeffs
#     E = np.diag(moenergies)
#     occupations = utils.occupations_from_pyscf_mol(mol, mocoeffs)
#     nocc, nvirt, _, _ = occupations

#     A = eqns.form_rpa_a_matrix_mo_singlet_full(E, tei_mo, nocc)
#     B = eqns.form_rpa_b_matrix_mo_singlet_full(tei_mo, nocc)

#     # G = np.block([[A, B],
#     #               [B, A]])
#     # assert G.shape == (2*nocc*nvirt, 2*nocc*nvirt)
#     G = A + B
#     assert G.shape == (nocc*nvirt, nocc*nvirt)

#     G_inv = np.linalg.inv(G)

#     components = 3

#     integrals_dipole_ao = mol.intor('cint1e_r_sph', comp=components)

#     integrals_dipole_mo_full = []
#     rspvecs = []

#     for component in range(components):
#         integrals_dipole_mo_full_component = np.dot(C.T, np.dot(integrals_dipole_ao[component, ...], C))
#         integrals_dipole_mo_full.append(integrals_dipole_mo_full_component)

#         rspvec_component = np.dot(G_inv, integrals_dipole_mo_full_component)
#         rspvecs.append(rspvec_component)

#     for a in range(components):
#         for b in range(components):
#             print(np.dot(integrals_dipole_mo_full[a].T, rspvecs[b]))

#     # integrals_dipole_mo_full = np.stack(integrals_dipole_mo_full, axis=0)
#     # print(integrals_dipole_mo_full.shape)
#     # integrals_dipole_mo_ai_super = np.concatenate((integrals_dipole_mo_ai,
#     # #                                                -integrals_dipole_mo_ai), axis=0)
#     # rhsvecs = integrals_dipole_mo_ai_super
#     # rspvecs = np.dot(G_inv, rhsvecs)

#     # polarizability = 4 * np.dot(rhsvecs.T, rspvecs) / 2

#     # # pylint: disable=bad-whitespace
#     # result__0_00 = np.array([[ 7.93556221,  0.,          0.        ],
#     #                          [ 0.,          3.06821077,  0.        ],
#     #                          [ 0.,          0.,          0.05038621]])

#     # atol = 1.0e-8
#     # rtol = 0.0
#     # np.testing.assert_allclose(polarizability, result__0_00, rtol=rtol, atol=atol)
