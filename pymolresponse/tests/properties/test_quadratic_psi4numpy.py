from itertools import permutations, product

import numpy as np

import pyscf

from pymolresponse import cphf, solvers, utils
from pymolresponse.core import Hamiltonian, Program, Spin
from pymolresponse.interfaces.pyscf.molecules import (
    molecule_physicists_water_augccpvdz,
    molecule_physicists_water_sto3g,
)
from pymolresponse.interfaces.pyscf.utils import occupations_from_pyscf_mol
from pymolresponse.properties import electric


def test_first_hyperpolarizability_shg_rhf_wigner_explicit_psi4numpy_pyscf_small():
    mol = molecule_physicists_water_sto3g()
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = occupations_from_pyscf_mol(mol, C)
    nocc_alph, nvirt_alph, _, _ = occupations
    nov_alph = nocc_alph * nvirt_alph
    norb = nocc_alph + nvirt_alph

    # calculate linear response vectors for electric dipole operator
    f1 = 0.0773178
    f2 = 2 * f1
    frequencies = [f1, f2]
    calculator = electric.Polarizability(
        Program.PySCF,
        mol,
        cphf.CPHF(solvers.ExactInv(C, E, occupations)),
        C,
        E,
        occupations,
        frequencies=frequencies,
    )
    calculator.form_operators()
    calculator.run(hamiltonian=Hamiltonian.RPA, spin=Spin.singlet)
    calculator.form_results()

    polarizability_1 = calculator.polarizabilities[0]
    polarizability_2 = calculator.polarizabilities[1]
    print("polarizability: {} a.u.".format(f1))
    print(polarizability_1)
    print("polarizability: {} a.u. (frequency doubled)".format(f2))
    print(polarizability_2)

    # each operator contains multiple sets of response vectors, one
    # set of components for each frequency
    assert isinstance(calculator.driver.solver.operators, list)
    assert len(calculator.driver.solver.operators) == 1
    operator = calculator.driver.solver.operators[0]
    rhsvecs = operator.mo_integrals_ai_supervector_alph
    assert isinstance(operator.rspvecs_alph, list)
    assert len(operator.rspvecs_alph) == 2
    rspvecs_1 = operator.rspvecs_alph[0]
    rspvecs_2 = operator.rspvecs_alph[1]

    ## Form the full [norb, norb] representation of everything.
    # Response vectors: transform X_{ia} and Y_{ia} -> U_{p,q}
    # 0. 'a' is fast index, 'i' slow
    # 1. rspvec == [X Y]
    # 2. U_{p, q} -> zero
    # 3. place X_{ia} into U_{i, a}
    # 4. place Y_{ia} into U_{a, i}

    ncomp = rhsvecs.shape[0]

    rspmats_1 = np.zeros(shape=(ncomp, norb, norb))
    rspmats_2 = np.zeros(shape=(ncomp, norb, norb))
    for i in range(ncomp):
        rspvec_1 = rspvecs_1[i, :, 0]
        rspvec_2 = rspvecs_2[i, :, 0]
        x_1 = rspvec_1[:nov_alph]
        y_1 = rspvec_1[nov_alph:]
        x_2 = rspvec_2[:nov_alph]
        y_2 = rspvec_2[nov_alph:]
        x_full_1 = utils.repack_vector_to_matrix(x_1, (nvirt_alph, nocc_alph))
        y_full_1 = utils.repack_vector_to_matrix(y_1, (nvirt_alph, nocc_alph))
        x_full_2 = utils.repack_vector_to_matrix(x_2, (nvirt_alph, nocc_alph))
        y_full_2 = utils.repack_vector_to_matrix(y_2, (nvirt_alph, nocc_alph))
        rspmats_1[i, :nocc_alph, nocc_alph:] = y_full_1.T
        rspmats_1[i, nocc_alph:, :nocc_alph] = x_full_1
        rspmats_2[i, :nocc_alph, nocc_alph:] = y_full_2.T
        rspmats_2[i, nocc_alph:, :nocc_alph] = x_full_2

    rhsmats = np.zeros(shape=(ncomp, norb, norb))
    for i in range(ncomp):
        rhsvec = rhsvecs[i, :, 0]
        rhsvec_top = rhsvec[:nov_alph]
        rhsvec_bot = rhsvec[nov_alph:]
        rhsvec_top_mat = utils.repack_vector_to_matrix(rhsvec_top, (nvirt_alph, nocc_alph))
        rhsvec_bot_mat = utils.repack_vector_to_matrix(rhsvec_bot, (nvirt_alph, nocc_alph))
        rhsmats[i, :nocc_alph, nocc_alph:] = rhsvec_top_mat.T
        rhsmats[i, nocc_alph:, :nocc_alph] = rhsvec_bot_mat

    polarizability_full_1 = np.empty_like(polarizability_1)
    polarizability_full_2 = np.empty_like(polarizability_2)
    for a in (0, 1, 2):
        for b in (0, 1, 2):
            polarizability_full_1[a, b] = 2 * np.trace(np.dot(rhsmats[a].T, rspmats_1[b]))
            polarizability_full_2[a, b] = 2 * np.trace(np.dot(rhsmats[a].T, rspmats_2[b]))

    np.testing.assert_almost_equal(polarizability_1, -polarizability_full_1)
    np.testing.assert_almost_equal(polarizability_2, -polarizability_full_2)

    # V_{p,q} <- full MO transformation of right hand side
    integrals_ao = operator.ao_integrals
    integrals_mo = np.empty_like(integrals_ao)
    for i in range(ncomp):
        integrals_mo[i] = (C[0].T).dot(integrals_ao[i]).dot(C[0])

    G_1 = np.empty_like(rspmats_1)
    G_2 = np.empty_like(rspmats_2)
    C = mf.mo_coeff
    Co = C[:, :nocc_alph]
    # TODO I feel as though if I have all the MO-basis two-electron
    # integrals, I shouldn't need another JK build.
    for i in range(ncomp):
        V = integrals_mo[i]
        x1 = rspmats_1[i, :nocc_alph, :]
        y1 = rspmats_1[i, :, :nocc_alph]
        x2 = rspmats_2[i, :nocc_alph, :]
        y2 = rspmats_2[i, :, :nocc_alph]
        Dl_1 = (Co).dot(x1).dot(C.T)
        Dr_1 = (-C).dot(y1).dot(Co.T)
        D_1 = Dl_1 + Dr_1
        Dl_2 = (Co).dot(x2).dot(C.T)
        Dr_2 = (-C).dot(y2).dot(Co.T)
        D_2 = Dl_2 + Dr_2
        J_1, K_1 = mf.get_jk(mol, D_1, hermi=0)
        J_2, K_2 = mf.get_jk(mol, D_2, hermi=0)
        F_AO_1 = 2 * J_1 - K_1
        F_AO_2 = 2 * J_2 - K_2
        F_MO_1 = (C.T).dot(F_AO_1).dot(C)
        F_MO_2 = (C.T).dot(F_AO_2).dot(C)
        G_1[i] = V + F_MO_1
        G_2[i] = V + F_MO_2

    E_diag = np.diag(E[0])
    epsilon_1 = G_1.copy()
    epsilon_2 = G_2.copy()
    for i in range(ncomp):
        eoU_1 = (E_diag[..., np.newaxis] + f1) * rspmats_1[i]
        Ue_1 = rspmats_1[i] * E_diag[np.newaxis]
        epsilon_1[i] += eoU_1 - Ue_1
        eoU_2 = (E_diag[..., np.newaxis] + f2) * rspmats_2[i]
        Ue_2 = rspmats_2[i] * E_diag[np.newaxis]
        epsilon_2[i] += eoU_2 - Ue_2

    # Assume some symmetry and calculate only part of the tensor.

    hyperpolarizability = np.zeros(shape=(6, 3))
    off1 = [0, 1, 2, 0, 0, 1]
    off2 = [0, 1, 2, 1, 2, 2]
    for r in range(6):
        b = off1[r]
        c = off2[r]
        for a in range(3):
            tl1 = np.trace(rspmats_2[a].T.dot(G_1[b]).dot(rspmats_1[c])[:nocc_alph, :nocc_alph])
            tl2 = np.trace(rspmats_1[c].dot(G_1[b]).dot(rspmats_2[a].T)[:nocc_alph, :nocc_alph])
            tl3 = np.trace(rspmats_2[a].T.dot(G_1[c]).dot(rspmats_1[b])[:nocc_alph, :nocc_alph])
            tl4 = np.trace(rspmats_1[b].dot(G_1[c]).dot(rspmats_2[a].T)[:nocc_alph, :nocc_alph])
            tl5 = np.trace(rspmats_1[c].dot(-G_2[a].T).dot(rspmats_1[b])[:nocc_alph, :nocc_alph])
            tl6 = np.trace(rspmats_1[b].dot(-G_2[a].T).dot(rspmats_1[c])[:nocc_alph, :nocc_alph])
            tr1 = np.trace(
                rspmats_1[c].dot(rspmats_1[b]).dot(-epsilon_2[a].T)[:nocc_alph, :nocc_alph]
            )
            tr2 = np.trace(
                rspmats_1[b].dot(rspmats_1[c]).dot(-epsilon_2[a].T)[:nocc_alph, :nocc_alph]
            )
            tr3 = np.trace(
                rspmats_1[c].dot(rspmats_2[a].T).dot(epsilon_1[b])[:nocc_alph, :nocc_alph]
            )
            tr4 = np.trace(
                rspmats_2[a].T.dot(rspmats_1[c]).dot(epsilon_1[b])[:nocc_alph, :nocc_alph]
            )
            tr5 = np.trace(
                rspmats_1[b].dot(rspmats_2[a].T).dot(epsilon_1[c])[:nocc_alph, :nocc_alph]
            )
            tr6 = np.trace(
                rspmats_2[a].T.dot(rspmats_1[b]).dot(epsilon_1[c])[:nocc_alph, :nocc_alph]
            )
            tl = tl1 + tl2 + tl3 + tl4 + tl5 + tl6
            tr = tr1 + tr2 + tr3 + tr4 + tr5 + tr6
            hyperpolarizability[r, a] = -2 * (tl - tr)

    # pylint: disable=C0326
    ref = np.array(
        [
            [0.00000000, 0.00000000, 0.18268574],
            [0.00000000, 0.00000000, -9.93851928],
            [0.00000000, 0.00000000, -5.70568308],
            [0.00000000, 0.00000000, 0.00000000],
            [0.14326805, 0.00000000, 0.00000000],
            [0.00000000, -10.05824143, 0.00000000],
        ]
    )
    ref_avgs = np.array([0.00000000, 0.00000000, 15.56760984])
    ref_avg = 15.56760984
    diff = np.abs(ref - hyperpolarizability)
    print("abs diff")
    print(diff)
    thresh = 2.5e-04
    assert np.all(diff < thresh)

    print("hyperpolarizability: SHG, (-{}; {}, {}), symmetry-unique components".format(f2, f1, f1))
    print(hyperpolarizability)
    print("ref")
    print(ref)

    # Transpose all frequency-doubled quantities (+2w) to get -2w.

    for i in range(ncomp):
        rspmats_2[i] = rspmats_2[i].T
        G_2[i] = -G_2[i].T
        epsilon_2[i] = -epsilon_2[i].T

    # Assume some symmetry and calculate only part of the tensor. This
    # time, work with the in-place manipulated quantities (this tests
    # their correctness).

    mU = (rspmats_2, rspmats_1)
    mG = (G_2, G_1)
    me = (epsilon_2, epsilon_1)

    hyperpolarizability = np.zeros(shape=(6, 3))
    off1 = [0, 1, 2, 0, 0, 1]
    off2 = [0, 1, 2, 1, 2, 2]
    for r in range(6):
        b = off1[r]
        c = off2[r]
        for a in range(3):
            tl1 = np.trace(mU[0][a].dot(mG[1][b]).dot(mU[1][c])[:nocc_alph, :nocc_alph])
            tl2 = np.trace(mU[1][c].dot(mG[1][b]).dot(mU[0][a])[:nocc_alph, :nocc_alph])
            tl3 = np.trace(mU[0][a].dot(mG[1][c]).dot(mU[1][b])[:nocc_alph, :nocc_alph])
            tl4 = np.trace(mU[1][b].dot(mG[1][c]).dot(mU[0][a])[:nocc_alph, :nocc_alph])
            tl5 = np.trace(mU[1][c].dot(mG[0][a]).dot(mU[1][b])[:nocc_alph, :nocc_alph])
            tl6 = np.trace(mU[1][b].dot(mG[0][a]).dot(mU[1][c])[:nocc_alph, :nocc_alph])
            tr1 = np.trace(mU[1][c].dot(mU[1][b]).dot(me[0][a])[:nocc_alph, :nocc_alph])
            tr2 = np.trace(mU[1][b].dot(mU[1][c]).dot(me[0][a])[:nocc_alph, :nocc_alph])
            tr3 = np.trace(mU[1][c].dot(mU[0][a]).dot(me[1][b])[:nocc_alph, :nocc_alph])
            tr4 = np.trace(mU[0][a].dot(mU[1][c]).dot(me[1][b])[:nocc_alph, :nocc_alph])
            tr5 = np.trace(mU[1][b].dot(mU[0][a]).dot(me[1][c])[:nocc_alph, :nocc_alph])
            tr6 = np.trace(mU[0][a].dot(mU[1][b]).dot(me[1][c])[:nocc_alph, :nocc_alph])
            tl = [tl1, tl2, tl3, tl4, tl5, tl6]
            tr = [tr1, tr2, tr3, tr4, tr5, tr6]
            hyperpolarizability[r, a] = -2 * (sum(tl) - sum(tr))

    assert np.all(np.abs(ref - hyperpolarizability) < thresh)

    # Assume no symmetry and calculate the full tensor.

    hyperpolarizability_full = np.zeros(shape=(3, 3, 3))

    # components x, y, z
    for ip, p in enumerate(list(product(range(3), range(3), range(3)))):
        a, b, c = p
        tl, tr = [], []
        # 1st tuple -> index a, b, c (*not* x, y, z!)
        # 2nd tuple -> index frequency (0 -> -2w, 1 -> +w)
        for iq, q in enumerate(list(permutations(zip(p, (0, 1, 1)), 3))):
            d, e, f = q
            tlp = (mU[d[1]][d[0]]).dot(mG[e[1]][e[0]]).dot(mU[f[1]][f[0]])
            tle = np.trace(tlp[:nocc_alph, :nocc_alph])
            tl.append(tle)
            trp = (mU[d[1]][d[0]]).dot(mU[e[1]][e[0]]).dot(me[f[1]][f[0]])
            tre = np.trace(trp[:nocc_alph, :nocc_alph])
            tr.append(tre)
        hyperpolarizability_full[a, b, c] = -2 * (sum(tl) - sum(tr))
    print("hyperpolarizability: SHG, (-{}; {}, {}), full tensor".format(f2, f1, f1))
    print(hyperpolarizability_full)

    # Check that the elements of the reduced and full tensors are
    # equivalent.

    for r in range(6):
        b = off1[r]
        c = off2[r]
        for a in range(3):
            diff = hyperpolarizability[r, a] - hyperpolarizability_full[a, b, c]
            # TODO why not 14?
            assert abs(diff) < 1.0e-13

    # Compute averages and compare to reference.

    avgs, avg = utils.form_first_hyperpolarizability_averages(hyperpolarizability_full)
    assert np.allclose(ref_avgs, avgs, rtol=0, atol=1.0e-3)
    assert np.allclose([ref_avg], [avg], rtol=0, atol=1.0e-3)
    print(avgs)
    print(avg)

    return


def test_first_hyperpolarizability_shg_rhf_wigner_explicit_psi4numpy_pyscf_large():
    mol = molecule_physicists_water_augccpvdz()
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = occupations_from_pyscf_mol(mol, C)
    nocc_alph, nvirt_alph, _, _ = occupations
    nov_alph = nocc_alph * nvirt_alph
    norb = nocc_alph + nvirt_alph

    # calculate linear response vectors for electric dipole operator
    f1 = 0.0773178
    f2 = 2 * f1
    frequencies = [f1, f2]
    calculator = electric.Polarizability(
        Program.PySCF,
        mol,
        cphf.CPHF(solvers.ExactInv(C, E, occupations)),
        C,
        E,
        occupations,
        frequencies=frequencies,
    )
    calculator.form_operators()
    calculator.run(hamiltonian=Hamiltonian.RPA, spin=Spin.singlet)
    calculator.form_results()

    polarizability_1 = calculator.polarizabilities[0]
    polarizability_2 = calculator.polarizabilities[1]
    print("polarizability: {} a.u.".format(f1))
    print(polarizability_1)
    print("polarizability: {} a.u. (frequency doubled)".format(f2))
    print(polarizability_2)

    # each operator contains multiple sets of response vectors, one
    # set of components for each frequency
    assert isinstance(calculator.driver.solver.operators, list)
    assert len(calculator.driver.solver.operators) == 1
    operator = calculator.driver.solver.operators[0]
    rhsvecs = operator.mo_integrals_ai_supervector_alph
    assert isinstance(operator.rspvecs_alph, list)
    assert len(operator.rspvecs_alph) == 2
    rspvecs_1 = operator.rspvecs_alph[0]
    rspvecs_2 = operator.rspvecs_alph[1]

    ## Form the full [norb, norb] representation of everything.
    # Response vectors: transform X_{ia} and Y_{ia} -> U_{p,q}
    # 0. 'a' is fast index, 'i' slow
    # 1. rspvec == [X Y]
    # 2. U_{p, q} -> zero
    # 3. place X_{ia} into U_{i, a}
    # 4. place Y_{ia} into U_{a, i}

    ncomp = rhsvecs.shape[0]

    rspmats_1 = np.zeros(shape=(ncomp, norb, norb))
    rspmats_2 = np.zeros(shape=(ncomp, norb, norb))
    for i in range(ncomp):
        rspvec_1 = rspvecs_1[i, :, 0]
        rspvec_2 = rspvecs_2[i, :, 0]
        x_1 = rspvec_1[:nov_alph]
        y_1 = rspvec_1[nov_alph:]
        x_2 = rspvec_2[:nov_alph]
        y_2 = rspvec_2[nov_alph:]
        x_full_1 = utils.repack_vector_to_matrix(x_1, (nvirt_alph, nocc_alph))
        y_full_1 = utils.repack_vector_to_matrix(y_1, (nvirt_alph, nocc_alph))
        x_full_2 = utils.repack_vector_to_matrix(x_2, (nvirt_alph, nocc_alph))
        y_full_2 = utils.repack_vector_to_matrix(y_2, (nvirt_alph, nocc_alph))
        rspmats_1[i, :nocc_alph, nocc_alph:] = y_full_1.T
        rspmats_1[i, nocc_alph:, :nocc_alph] = x_full_1
        rspmats_2[i, :nocc_alph, nocc_alph:] = y_full_2.T
        rspmats_2[i, nocc_alph:, :nocc_alph] = x_full_2

    rhsmats = np.zeros(shape=(ncomp, norb, norb))
    for i in range(ncomp):
        rhsvec = rhsvecs[i, :, 0]
        rhsvec_top = rhsvec[:nov_alph]
        rhsvec_bot = rhsvec[nov_alph:]
        rhsvec_top_mat = utils.repack_vector_to_matrix(rhsvec_top, (nvirt_alph, nocc_alph))
        rhsvec_bot_mat = utils.repack_vector_to_matrix(rhsvec_bot, (nvirt_alph, nocc_alph))
        rhsmats[i, :nocc_alph, nocc_alph:] = rhsvec_top_mat.T
        rhsmats[i, nocc_alph:, :nocc_alph] = rhsvec_bot_mat

    polarizability_full_1 = np.empty_like(polarizability_1)
    polarizability_full_2 = np.empty_like(polarizability_2)
    for a in (0, 1, 2):
        for b in (0, 1, 2):
            polarizability_full_1[a, b] = 2 * np.trace(np.dot(rhsmats[a].T, rspmats_1[b]))
            polarizability_full_2[a, b] = 2 * np.trace(np.dot(rhsmats[a].T, rspmats_2[b]))

    np.testing.assert_almost_equal(polarizability_1, -polarizability_full_1)
    np.testing.assert_almost_equal(polarizability_2, -polarizability_full_2)

    # V_{p,q} <- full MO transformation of right hand side
    integrals_ao = operator.ao_integrals
    integrals_mo = np.empty_like(integrals_ao)
    for i in range(ncomp):
        integrals_mo[i] = (C[0].T).dot(integrals_ao[i]).dot(C[0])

    G_1 = np.empty_like(rspmats_1)
    G_2 = np.empty_like(rspmats_2)
    C = mf.mo_coeff
    # TODO I feel as though if I have all the MO-basis two-electron
    # integrals, I shouldn't need another JK build.
    for i in range(ncomp):
        V = integrals_mo[i]
        Dl_1 = (C[:, :nocc_alph]).dot(rspmats_1[i, :nocc_alph, :]).dot(C.T)
        Dr_1 = (-C).dot(rspmats_1[i, :, :nocc_alph]).dot(C[:, :nocc_alph].T)
        D_1 = Dl_1 + Dr_1
        Dl_2 = (C[:, :nocc_alph]).dot(rspmats_2[i, :nocc_alph, :]).dot(C.T)
        Dr_2 = (-C).dot(rspmats_2[i, :, :nocc_alph]).dot(C[:, :nocc_alph].T)
        D_2 = Dl_2 + Dr_2
        J_1, K_1 = mf.get_jk(mol, D_1, hermi=0)
        J_2, K_2 = mf.get_jk(mol, D_2, hermi=0)
        F_AO_1 = 2 * J_1 - K_1
        F_AO_2 = 2 * J_2 - K_2
        F_MO_1 = (C.T).dot(F_AO_1).dot(C)
        F_MO_2 = (C.T).dot(F_AO_2).dot(C)
        G_1[i] = V + F_MO_1
        G_2[i] = V + F_MO_2

    E_diag = np.diag(E[0])
    epsilon_1 = G_1.copy()
    epsilon_2 = G_2.copy()
    for i in range(ncomp):
        eoU_1 = (E_diag[..., np.newaxis] + f1) * rspmats_1[i]
        Ue_1 = rspmats_1[i] * E_diag[np.newaxis]
        epsilon_1[i] += eoU_1 - Ue_1
        eoU_2 = (E_diag[..., np.newaxis] + f2) * rspmats_2[i]
        Ue_2 = rspmats_2[i] * E_diag[np.newaxis]
        epsilon_2[i] += eoU_2 - Ue_2

    # Assume some symmetry and calculate only part of the tensor.

    hyperpolarizability = np.zeros(shape=(6, 3))
    off1 = [0, 1, 2, 0, 0, 1]
    off2 = [0, 1, 2, 1, 2, 2]
    for r in range(6):
        b = off1[r]
        c = off2[r]
        for a in range(3):
            tl1 = np.trace(rspmats_2[a].T.dot(G_1[b]).dot(rspmats_1[c])[:nocc_alph, :nocc_alph])
            tl2 = np.trace(rspmats_1[c].dot(G_1[b]).dot(rspmats_2[a].T)[:nocc_alph, :nocc_alph])
            tl3 = np.trace(rspmats_2[a].T.dot(G_1[c]).dot(rspmats_1[b])[:nocc_alph, :nocc_alph])
            tl4 = np.trace(rspmats_1[b].dot(G_1[c]).dot(rspmats_2[a].T)[:nocc_alph, :nocc_alph])
            tl5 = np.trace(rspmats_1[c].dot(-G_2[a].T).dot(rspmats_1[b])[:nocc_alph, :nocc_alph])
            tl6 = np.trace(rspmats_1[b].dot(-G_2[a].T).dot(rspmats_1[c])[:nocc_alph, :nocc_alph])
            tr1 = np.trace(
                rspmats_1[c].dot(rspmats_1[b]).dot(-epsilon_2[a].T)[:nocc_alph, :nocc_alph]
            )
            tr2 = np.trace(
                rspmats_1[b].dot(rspmats_1[c]).dot(-epsilon_2[a].T)[:nocc_alph, :nocc_alph]
            )
            tr3 = np.trace(
                rspmats_1[c].dot(rspmats_2[a].T).dot(epsilon_1[b])[:nocc_alph, :nocc_alph]
            )
            tr4 = np.trace(
                rspmats_2[a].T.dot(rspmats_1[c]).dot(epsilon_1[b])[:nocc_alph, :nocc_alph]
            )
            tr5 = np.trace(
                rspmats_1[b].dot(rspmats_2[a].T).dot(epsilon_1[c])[:nocc_alph, :nocc_alph]
            )
            tr6 = np.trace(
                rspmats_2[a].T.dot(rspmats_1[b]).dot(epsilon_1[c])[:nocc_alph, :nocc_alph]
            )
            tl = tl1 + tl2 + tl3 + tl4 + tl5 + tl6
            tr = tr1 + tr2 + tr3 + tr4 + tr5 + tr6
            hyperpolarizability[r, a] = -2 * (tl - tr)

    # pylint: disable=C0326
    ref = np.array(
        [
            [0.00000000, 0.00000000, 1.92505358],
            [0.00000000, 0.00000000, -31.33652886],
            [0.00000000, 0.00000000, -13.92830863],
            [0.00000000, 0.00000000, 0.00000000],
            [-1.80626084, 0.00000000, 0.00000000],
            [0.00000000, -31.13504192, 0.00000000],
        ]
    )
    ref_avgs = np.array([0.00000000, 0.00000000, 45.69300223])
    ref_avg = 45.69300223
    diff = np.abs(ref - hyperpolarizability)
    print("abs diff")
    print(diff)
    thresh = 2.5e-04
    assert np.all(diff < thresh)

    print("hyperpolarizability: SHG, (-{}; {}, {}), symmetry-unique components".format(f2, f1, f1))
    print(hyperpolarizability)
    print("ref")
    print(ref)

    # Transpose all frequency-doubled quantities (+2w) to get -2w.

    for i in range(ncomp):
        rspmats_2[i] = rspmats_2[i].T
        G_2[i] = -G_2[i].T
        epsilon_2[i] = -epsilon_2[i].T

    # Assume some symmetry and calculate only part of the tensor. This
    # time, work with the in-place manipulated quantities (this tests
    # their correctness).

    mU = (rspmats_2, rspmats_1)
    mG = (G_2, G_1)
    me = (epsilon_2, epsilon_1)

    hyperpolarizability = np.zeros(shape=(6, 3))
    off1 = [0, 1, 2, 0, 0, 1]
    off2 = [0, 1, 2, 1, 2, 2]
    for r in range(6):
        b = off1[r]
        c = off2[r]
        for a in range(3):
            tl1 = np.trace(mU[0][a].dot(mG[1][b]).dot(mU[1][c])[:nocc_alph, :nocc_alph])
            tl2 = np.trace(mU[1][c].dot(mG[1][b]).dot(mU[0][a])[:nocc_alph, :nocc_alph])
            tl3 = np.trace(mU[0][a].dot(mG[1][c]).dot(mU[1][b])[:nocc_alph, :nocc_alph])
            tl4 = np.trace(mU[1][b].dot(mG[1][c]).dot(mU[0][a])[:nocc_alph, :nocc_alph])
            tl5 = np.trace(mU[1][c].dot(mG[0][a]).dot(mU[1][b])[:nocc_alph, :nocc_alph])
            tl6 = np.trace(mU[1][b].dot(mG[0][a]).dot(mU[1][c])[:nocc_alph, :nocc_alph])
            tr1 = np.trace(mU[1][c].dot(mU[1][b]).dot(me[0][a])[:nocc_alph, :nocc_alph])
            tr2 = np.trace(mU[1][b].dot(mU[1][c]).dot(me[0][a])[:nocc_alph, :nocc_alph])
            tr3 = np.trace(mU[1][c].dot(mU[0][a]).dot(me[1][b])[:nocc_alph, :nocc_alph])
            tr4 = np.trace(mU[0][a].dot(mU[1][c]).dot(me[1][b])[:nocc_alph, :nocc_alph])
            tr5 = np.trace(mU[1][b].dot(mU[0][a]).dot(me[1][c])[:nocc_alph, :nocc_alph])
            tr6 = np.trace(mU[0][a].dot(mU[1][b]).dot(me[1][c])[:nocc_alph, :nocc_alph])
            tl = [tl1, tl2, tl3, tl4, tl5, tl6]
            tr = [tr1, tr2, tr3, tr4, tr5, tr6]
            hyperpolarizability[r, a] = -2 * (sum(tl) - sum(tr))

    assert np.all(np.abs(ref - hyperpolarizability) < thresh)

    # Assume no symmetry and calculate the full tensor.

    hyperpolarizability_full = np.zeros(shape=(3, 3, 3))

    # components x, y, z
    for ip, p in enumerate(list(product(range(3), range(3), range(3)))):
        a, b, c = p
        tl, tr = [], []
        # 1st tuple -> index a, b, c (*not* x, y, z!)
        # 2nd tuple -> index frequency (0 -> -2w, 1 -> +w)
        for iq, q in enumerate(list(permutations(zip(p, (0, 1, 1)), 3))):
            d, e, f = q
            tlp = (mU[d[1]][d[0]]).dot(mG[e[1]][e[0]]).dot(mU[f[1]][f[0]])
            tle = np.trace(tlp[:nocc_alph, :nocc_alph])
            tl.append(tle)
            trp = (mU[d[1]][d[0]]).dot(mU[e[1]][e[0]]).dot(me[f[1]][f[0]])
            tre = np.trace(trp[:nocc_alph, :nocc_alph])
            tr.append(tre)
        hyperpolarizability_full[a, b, c] = -2 * (sum(tl) - sum(tr))
    print("hyperpolarizability: SHG, (-{}; {}, {}), full tensor".format(f2, f1, f1))
    print(hyperpolarizability_full)

    # Check that the elements of the reduced and full tensors are
    # equivalent.

    for r in range(6):
        b = off1[r]
        c = off2[r]
        for a in range(3):
            diff = hyperpolarizability[r, a] - hyperpolarizability_full[a, b, c]
            # TODO why not 14?
            assert abs(diff) < 1.0e-13

    # Compute averages and compare to reference.

    avgs, avg = utils.form_first_hyperpolarizability_averages(hyperpolarizability_full)
    assert np.allclose(ref_avgs, avgs, rtol=0, atol=1.0e-3)
    assert np.allclose([ref_avg], [avg], rtol=0, atol=1.0e-3)
    print(avgs)
    print(avg)

    return


# def test_first_hyperpolarizability_shg_rhf_wigner_explicit_psi4numpy_psi4_small():
#     mol = molecule_physicists_water_sto3g_psi4()

#     _, wfn = psi4.energy('hf', return_wfn=True)
#     C = utils.mocoeffs_from_psi4wfn(wfn)
#     E = utils.moenergies_from_psi4wfn(wfn)
#     occupations = occupations_from_psi4wfn(wfn)
#     nocc_alph, nvirt_alph, _, _ = occupations
#     nov_alph = nocc_alph * nvirt_alph
#     norb = nocc_alph + nvirt_alph

#     # calculate linear response vectors for electric dipole operator
#     f1 = 0.0773178
#     f2 = 2 * f1
#     frequencies = [f1, f2]
#     calculator = electric.Polarizability(Program.PySCF, mol, C, E, occupations, frequencies=frequencies)
#     calculator.form_operators()
#     calculator.run()
#     calculator.form_results()

#     polarizability_1 = calculator.polarizabilities[0]
#     polarizability_2 = calculator.polarizabilities[1]
#     print('polarizability: {} a.u.'.format(f1))
#     print(polarizability_1)
#     print('polarizability: {} a.u. (frequency doubled)'.format(f2))
#     print(polarizability_2)

#     # each operator contains multiple sets of response vectors, one
#     # set of components for each frequency
#     assert isinstance(calculator.driver.solver.operators, list)
#     assert len(calculator.driver.solver.operators) == 1
#     operator = calculator.driver.solver.operators[0]
#     rhsvecs = operator.mo_integrals_ai_supervector_alph
#     assert isinstance(operator.rspvecs_alph, list)
#     assert len(operator.rspvecs_alph) == 2
#     rspvecs_1 = operator.rspvecs_alph[0]
#     rspvecs_2 = operator.rspvecs_alph[1]

#     ## Form the full [norb, norb] representation of everything.
#     # Response vectors: transform X_{ia} and Y_{ia} -> U_{p,q}
#     # 0. 'a' is fast index, 'i' slow
#     # 1. rspvec == [X Y]
#     # 2. U_{p, q} -> zero
#     # 3. place X_{ia} into U_{i, a}
#     # 4. place Y_{ia} into U_{a, i}

#     ncomp = rhsvecs.shape[0]

#     rspmats_1 = np.zeros(shape=(ncomp, norb, norb))
#     rspmats_2 = np.zeros(shape=(ncomp, norb, norb))
#     for i in range(ncomp):
#         rspvec_1 = rspvecs_1[i, :, 0]
#         rspvec_2 = rspvecs_2[i, :, 0]
#         x_1 = rspvec_1[:nov_alph]
#         y_1 = rspvec_1[nov_alph:]
#         x_2 = rspvec_2[:nov_alph]
#         y_2 = rspvec_2[nov_alph:]
#         x_full_1 = utils.repack_vector_to_matrix(x_1, (nvirt_alph, nocc_alph))
#         y_full_1 = utils.repack_vector_to_matrix(y_1, (nvirt_alph, nocc_alph))
#         x_full_2 = utils.repack_vector_to_matrix(x_2, (nvirt_alph, nocc_alph))
#         y_full_2 = utils.repack_vector_to_matrix(y_2, (nvirt_alph, nocc_alph))
#         rspmats_1[i, :nocc_alph, nocc_alph:] = y_full_1.T
#         rspmats_1[i, nocc_alph:, :nocc_alph] = x_full_1
#         rspmats_2[i, :nocc_alph, nocc_alph:] = y_full_2.T
#         rspmats_2[i, nocc_alph:, :nocc_alph] = x_full_2

#     rhsmats = np.zeros(shape=(ncomp, norb, norb))
#     for i in range(ncomp):
#         rhsvec = rhsvecs[i, :, 0]
#         rhsvec_top = rhsvec[:nov_alph]
#         rhsvec_bot = rhsvec[nov_alph:]
#         rhsvec_top_mat = utils.repack_vector_to_matrix(rhsvec_top, (nvirt_alph, nocc_alph))
#         rhsvec_bot_mat = utils.repack_vector_to_matrix(rhsvec_bot, (nvirt_alph, nocc_alph))
#         rhsmats[i, :nocc_alph, nocc_alph:] = rhsvec_top_mat.T
#         rhsmats[i, nocc_alph:, :nocc_alph] = rhsvec_bot_mat

#     polarizability_full_1 = np.empty_like(polarizability_1)
#     polarizability_full_2 = np.empty_like(polarizability_2)
#     for a in (0, 1, 2):
#         for b in (0, 1, 2):
#             polarizability_full_1[a, b] = 2 * np.trace(np.dot(rhsmats[a].T,
#                                                               rspmats_1[b]))
#             polarizability_full_2[a, b] = 2 * np.trace(np.dot(rhsmats[a].T,
#                                                               rspmats_2[b]))

#     np.testing.assert_almost_equal(polarizability_1, -polarizability_full_1)
#     np.testing.assert_almost_equal(polarizability_2, -polarizability_full_2)

#     # V_{p,q} <- full MO transformation of right hand side
#     integrals_ao = operator.ao_integrals
#     integrals_mo = np.empty_like(integrals_ao)
#     for i in range(ncomp):
#         integrals_mo[i] = (C[0].T).dot(integrals_ao[i]).dot(C[0])

#     G_1 = np.empty_like(rspmats_1)
#     G_2 = np.empty_like(rspmats_2)
#     C = mf.mo_coeff
#     # TODO I feel as though if I have all the MO-basis two-electron
#     # integrals, I shouldn't need another JK build.
#     for i in range(ncomp):
#         V = integrals_mo[i]
#         Dl_1 = (C[:, :nocc_alph]).dot(rspmats_1[i, :nocc_alph, :]).dot(C.T)
#         Dr_1 = (-C).dot(rspmats_1[i, :, :nocc_alph]).dot(C[:, :nocc_alph].T)
#         D_1 = Dl_1 + Dr_1
#         Dl_2 = (C[:, :nocc_alph]).dot(rspmats_2[i, :nocc_alph, :]).dot(C.T)
#         Dr_2 = (-C).dot(rspmats_2[i, :, :nocc_alph]).dot(C[:, :nocc_alph].T)
#         D_2 = Dl_2 + Dr_2
#         J_1, K_1 = mf.get_jk(mol, D_1, hermi=0)
#         J_2, K_2 = mf.get_jk(mol, D_2, hermi=0)
#         F_AO_1 = 2*J_1 - K_1
#         F_AO_2 = 2*J_2 - K_2
#         F_MO_1 = (C.T).dot(F_AO_1).dot(C)
#         F_MO_2 = (C.T).dot(F_AO_2).dot(C)
#         G_1[i] = V + F_MO_1
#         G_2[i] = V + F_MO_2

#     E_diag = np.diag(E[0])
#     epsilon_1 = G_1.copy()
#     epsilon_2 = G_2.copy()
#     for i in range(ncomp):
#         eoU_1 = (E_diag[..., np.newaxis] + f1) * rspmats_1[i]
#         Ue_1 = rspmats_1[i] * E_diag[np.newaxis]
#         epsilon_1[i] += (eoU_1 - Ue_1)
#         eoU_2 = (E_diag[..., np.newaxis] + f2) * rspmats_2[i]
#         Ue_2 = rspmats_2[i] * E_diag[np.newaxis]
#         epsilon_2[i] += (eoU_2 - Ue_2)

#     # Assume some symmetry and calculate only part of the tensor.

#     hyperpolarizability = np.zeros(shape=(6, 3))
#     off1 = [0, 1, 2, 0, 0, 1]
#     off2 = [0, 1, 2, 1, 2, 2]
#     for r in range(6):
#         b = off1[r]
#         c = off2[r]
#         for a in range(3):
#             tl1 = np.trace(rspmats_2[a].T.dot(G_1[b]).dot(rspmats_1[c])[:nocc_alph, :nocc_alph])
#             tl2 = np.trace(rspmats_1[c].dot(G_1[b]).dot(rspmats_2[a].T)[:nocc_alph, :nocc_alph])
#             tl3 = np.trace(rspmats_2[a].T.dot(G_1[c]).dot(rspmats_1[b])[:nocc_alph, :nocc_alph])
#             tl4 = np.trace(rspmats_1[b].dot(G_1[c]).dot(rspmats_2[a].T)[:nocc_alph, :nocc_alph])
#             tl5 = np.trace(rspmats_1[c].dot(-G_2[a].T).dot(rspmats_1[b])[:nocc_alph, :nocc_alph])
#             tl6 = np.trace(rspmats_1[b].dot(-G_2[a].T).dot(rspmats_1[c])[:nocc_alph, :nocc_alph])
#             tr1 = np.trace(rspmats_1[c].dot(rspmats_1[b]).dot(-epsilon_2[a].T)[:nocc_alph, :nocc_alph])
#             tr2 = np.trace(rspmats_1[b].dot(rspmats_1[c]).dot(-epsilon_2[a].T)[:nocc_alph, :nocc_alph])
#             tr3 = np.trace(rspmats_1[c].dot(rspmats_2[a].T).dot(epsilon_1[b])[:nocc_alph, :nocc_alph])
#             tr4 = np.trace(rspmats_2[a].T.dot(rspmats_1[c]).dot(epsilon_1[b])[:nocc_alph, :nocc_alph])
#             tr5 = np.trace(rspmats_1[b].dot(rspmats_2[a].T).dot(epsilon_1[c])[:nocc_alph, :nocc_alph])
#             tr6 = np.trace(rspmats_2[a].T.dot(rspmats_1[b]).dot(epsilon_1[c])[:nocc_alph, :nocc_alph])
#             tl = tl1 + tl2 + tl3 + tl4 + tl5 + tl6
#             tr = tr1 + tr2 + tr3 + tr4 + tr5 + tr6
#             hyperpolarizability[r, a] = -2 * (tl - tr)

#     # pylint: disable=C0326
#     ref = np.array([
#         [0.00000000,   0.00000000,  0.18268574],
#         [0.00000000,   0.00000000, -9.93851928],
#         [0.00000000,   0.00000000, -5.70568308],
#         [0.00000000,   0.00000000,  0.00000000],
#         [0.14326805,   0.00000000,  0.00000000],
#         [0.00000000, -10.05824143,  0.00000000]
#     ])
#     ref_avgs = np.array([0.00000000, 0.00000000, 15.56760984])
#     ref_avg = 15.56760984
#     diff = np.abs(ref - hyperpolarizability)
#     print('abs diff')
#     print(diff)
#     thresh = 2.5e-04
#     assert np.all(diff < thresh)

#     print('hyperpolarizability: SHG, (-{}; {}, {}), symmetry-unique components'.format(f2, f1, f1))
#     print(hyperpolarizability)
#     print('ref')
#     print(ref)

#     # Transpose all frequency-doubled quantities (+2w) to get -2w.

#     for i in range(ncomp):
#         rspmats_2[i] = rspmats_2[i].T
#         G_2[i] = -G_2[i].T
#         epsilon_2[i] = -epsilon_2[i].T

#     # Assume some symmetry and calculate only part of the tensor. This
#     # time, work with the in-place manipulated quantities (this tests
#     # their correctness).

#     mU = (rspmats_2, rspmats_1)
#     mG = (G_2, G_1)
#     me = (epsilon_2, epsilon_1)

#     hyperpolarizability = np.zeros(shape=(6, 3))
#     off1 = [0, 1, 2, 0, 0, 1]
#     off2 = [0, 1, 2, 1, 2, 2]
#     for r in range(6):
#         b = off1[r]
#         c = off2[r]
#         for a in range(3):
#             tl1 = np.trace(mU[0][a].dot(mG[1][b]).dot(mU[1][c])[:nocc_alph, :nocc_alph])
#             tl2 = np.trace(mU[1][c].dot(mG[1][b]).dot(mU[0][a])[:nocc_alph, :nocc_alph])
#             tl3 = np.trace(mU[0][a].dot(mG[1][c]).dot(mU[1][b])[:nocc_alph, :nocc_alph])
#             tl4 = np.trace(mU[1][b].dot(mG[1][c]).dot(mU[0][a])[:nocc_alph, :nocc_alph])
#             tl5 = np.trace(mU[1][c].dot(mG[0][a]).dot(mU[1][b])[:nocc_alph, :nocc_alph])
#             tl6 = np.trace(mU[1][b].dot(mG[0][a]).dot(mU[1][c])[:nocc_alph, :nocc_alph])
#             tr1 = np.trace(mU[1][c].dot(mU[1][b]).dot(me[0][a])[:nocc_alph, :nocc_alph])
#             tr2 = np.trace(mU[1][b].dot(mU[1][c]).dot(me[0][a])[:nocc_alph, :nocc_alph])
#             tr3 = np.trace(mU[1][c].dot(mU[0][a]).dot(me[1][b])[:nocc_alph, :nocc_alph])
#             tr4 = np.trace(mU[0][a].dot(mU[1][c]).dot(me[1][b])[:nocc_alph, :nocc_alph])
#             tr5 = np.trace(mU[1][b].dot(mU[0][a]).dot(me[1][c])[:nocc_alph, :nocc_alph])
#             tr6 = np.trace(mU[0][a].dot(mU[1][b]).dot(me[1][c])[:nocc_alph, :nocc_alph])
#             tl = [tl1, tl2, tl3, tl4, tl5, tl6]
#             tr = [tr1, tr2, tr3, tr4, tr5, tr6]
#             hyperpolarizability[r, a] = -2 * (sum(tl) - sum(tr))

#     assert np.all(np.abs(ref - hyperpolarizability) < thresh)

#     # Assume no symmetry and calculate the full tensor.

#     hyperpolarizability_full = np.zeros(shape=(3, 3, 3))

#     # components x, y, z
#     for ip, p in enumerate(list(product(range(3), range(3), range(3)))):
#         a, b, c = p
#         tl, tr = [], []
#         # 1st tuple -> index a, b, c (*not* x, y, z!)
#         # 2nd tuple -> index frequency (0 -> -2w, 1 -> +w)
#         for iq, q in enumerate(list(permutations(zip(p, (0, 1, 1)), 3))):
#             d, e, f = q
#             tlp = (mU[d[1]][d[0]]).dot(mG[e[1]][e[0]]).dot(mU[f[1]][f[0]])
#             tle = np.trace(tlp[:nocc_alph, :nocc_alph])
#             tl.append(tle)
#             trp = (mU[d[1]][d[0]]).dot(mU[e[1]][e[0]]).dot(me[f[1]][f[0]])
#             tre = np.trace(trp[:nocc_alph, :nocc_alph])
#             tr.append(tre)
#         hyperpolarizability_full[a, b, c] = -2 * (sum(tl) - sum(tr))
#     print('hyperpolarizability: SHG, (-{}; {}, {}), full tensor'.format(f2, f1, f1))
#     print(hyperpolarizability_full)

#     # Check that the elements of the reduced and full tensors are
#     # equivalent.

#     for r in range(6):
#         b = off1[r]
#         c = off2[r]
#         for a in range(3):
#             diff = hyperpolarizability[r, a] - hyperpolarizability_full[a, b, c]
#             # TODO why not 14?
#             assert abs(diff) < 1.0e-13

#     # Compute averages and compare to reference.

#     avgs, avg = utils.form_first_hyperpolarizability_averages(hyperpolarizability_full)
#     assert np.allclose(ref_avgs, avgs, rtol=0, atol=1.0e-3)
#     assert np.allclose([ref_avg], [avg],rtol=0, atol=1.0e-3)
#     print(avgs)
#     print(avg)

#     return


if __name__ == "__main__":
    np.set_printoptions(precision=5, linewidth=200, suppress=True)
    # TODO automatically execute every test function found in the file when run
    # as main
    test_first_hyperpolarizability_shg_rhf_wigner_explicit_psi4numpy_pyscf_small()
    test_first_hyperpolarizability_shg_rhf_wigner_explicit_psi4numpy_pyscf_large()
    # test_first_hyperpolarizability_shg_rhf_wigner_explicit_psi4numpy_psi4_small()
