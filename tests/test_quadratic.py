from itertools import permutations, product

import numpy as np

import pyscf

from pyresponse import utils, electric
from .molecules_pyscf import molecule_water_sto3g_angstrom


def test_first_hyperpolarizability_static_rhf_wigner_explicit():
    mol = molecule_water_sto3g_angstrom()
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = utils.occupations_from_pyscf_mol(mol, C)
    nocc_alph, nvirt_alph, _, _ = occupations
    nov_alph = nocc_alph * nvirt_alph
    norb = nocc_alph + nvirt_alph

    # calculate linear response vectors for electric dipole operator
    calculator = electric.Polarizability(mol, C, E, occupations, frequencies=[0.0])
    calculator.form_operators()
    calculator.run()
    calculator.form_results()

    polarizability = calculator.polarizabilities[0]
    print('polarizability (static)')
    print(polarizability)

    operator = calculator.driver.solver.operators[0]
    rhsvecs = operator.mo_integrals_ai_supervector_alph
    rspvecs = operator.rspvecs_alph[0]

    ## Form the full [norb, norb] representation of everything.
    # Response vectors: transform X_{ia} and Y_{ia} -> U_{p,q}
    # 0. 'a' is fast index, 'i' slow
    # 1. rspvec == [X Y]
    # 2. U_{p, q} -> zero
    # 3. place X_{ia} into U_{i, a}
    # 4. place Y_{ia} into U_{a, i}

    ncomp = rhsvecs.shape[0]

    rspmats = np.zeros(shape=(ncomp, norb, norb))
    for icomp in range(ncomp):
        rspvec = rspvecs[icomp, :, 0]
        x = rspvec[:nov_alph]
        y = rspvec[nov_alph:]
        x_full = utils.repack_vector_to_matrix(x, (nvirt_alph, nocc_alph))
        y_full = utils.repack_vector_to_matrix(y, (nvirt_alph, nocc_alph))
        rspmats[icomp, :nocc_alph, nocc_alph:] = x_full.T
        rspmats[icomp, nocc_alph:, :nocc_alph] = y_full

    rhsmats = np.zeros(shape=(ncomp, norb, norb))
    for icomp in range(ncomp):
        rhsvec = rhsvecs[icomp, :, 0]
        rhsvec_top = rhsvec[:nov_alph]
        rhsvec_bot = rhsvec[nov_alph:]
        rhsvec_top_mat = utils.repack_vector_to_matrix(rhsvec_top, (nvirt_alph, nocc_alph))
        rhsvec_bot_mat = utils.repack_vector_to_matrix(rhsvec_bot, (nvirt_alph, nocc_alph))
        rhsmats[icomp, :nocc_alph, nocc_alph:] = rhsvec_top_mat.T
        rhsmats[icomp, nocc_alph:, :nocc_alph] = rhsvec_bot_mat

    polarizability_full = np.empty_like(polarizability)
    for a in (0, 1, 2):
        for b in (0, 1, 2):
            polarizability_full[a, b] = 2 * np.trace(np.dot(rhsmats[a, ...].T,
                                                            rspmats[b, ...]))

    np.testing.assert_almost_equal(polarizability, polarizability_full)

    # V_{p,q} <- full MO transformation of right hand side
    integrals_ao = operator.ao_integrals
    integrals_mo = np.empty_like(integrals_ao)
    for icomp in range(ncomp):
        integrals_mo[icomp, ...] = np.dot(C[0, ...].T,
                                          np.dot(integrals_ao[icomp, ...],
                                                 C[0, ...]))

    G = np.empty_like(rspmats)
    C = mf.mo_coeff
    # TODO I feel as though if I have all the MO-basis two-electron
    # integrals, I shouldn't need another JK build.
    for icomp in range(ncomp):
        V = integrals_mo[icomp, ...]
        Dl = np.dot(C[:, nocc_alph:], np.dot(utils.repack_vector_to_matrix(rspvecs[icomp, :nov_alph, 0], (nvirt_alph, nocc_alph)), C[:, :nocc_alph].T))
        J, K = mf.get_jk(mol, Dl, hermi=0)
        F_AO = -(4*J - K - K.T)
        F_MO = np.dot(C.T, np.dot(F_AO, C))
        G[icomp, ...] = V + F_MO

    E_diag = np.diag(E[0, ...])
    epsilon = G.copy()
    omega = 0
    for icomp in range(ncomp):
        eoU = (E_diag[..., np.newaxis] + omega) * rspmats[icomp, ...]
        Ue = rspmats[icomp, ...] * E_diag[np.newaxis, ...]
        epsilon[icomp, ...] += (eoU - Ue)

    # Assume some symmetry and calculate only part of the tensor.

    hyperpolarizability = np.zeros(shape=(6, 3))
    off1 = [0, 1, 2, 0, 0, 1]
    off2 = [0, 1, 2, 1, 2, 2]
    for r in range(6):
        b = off1[r]
        c = off2[r]
        for a in range(3):
            tl1 = 2 * np.trace(np.dot(rspmats[a, ...], np.dot(G[b, ...], rspmats[c, ...]))[:nocc_alph, :nocc_alph])
            tl2 = 2 * np.trace(np.dot(rspmats[a, ...], np.dot(G[c, ...], rspmats[b, ...]))[:nocc_alph, :nocc_alph])
            tl3 = 2 * np.trace(np.dot(rspmats[c, ...], np.dot(G[a, ...], rspmats[b, ...]))[:nocc_alph, :nocc_alph])
            tr1 = np.trace(np.dot(rspmats[c, ...], np.dot(rspmats[b, ...], epsilon[a, ...]))[:nocc_alph, :nocc_alph])
            tr2 = np.trace(np.dot(rspmats[b, ...], np.dot(rspmats[c, ...], epsilon[a, ...]))[:nocc_alph, :nocc_alph])
            tr3 = np.trace(np.dot(rspmats[c, ...], np.dot(rspmats[a, ...], epsilon[b, ...]))[:nocc_alph, :nocc_alph])
            tr4 = np.trace(np.dot(rspmats[a, ...], np.dot(rspmats[c, ...], epsilon[b, ...]))[:nocc_alph, :nocc_alph])
            tr5 = np.trace(np.dot(rspmats[b, ...], np.dot(rspmats[a, ...], epsilon[c, ...]))[:nocc_alph, :nocc_alph])
            tr6 = np.trace(np.dot(rspmats[a, ...], np.dot(rspmats[b, ...], epsilon[c, ...]))[:nocc_alph, :nocc_alph])
            tl = tl1 + tl2 + tl3
            tr = tr1 + tr2 + tr3 + tr4 + tr5 + tr6
            hyperpolarizability[r, a] = 2 * (tl - tr)

    # pylint: disable=C0326
    ref = np.array([
        [-8.86822254,  0.90192130, -0.50796586],
        [ 1.98744058,  5.13635628, -2.95319400],
        [ 0.66008119,  1.62699646, -0.85632412],
        [ 0.90192130,  1.98744058, -1.09505123],
        [-0.50796586, -1.09505123,  0.66008119],
        [-1.09505123, -2.95319400,  1.62699646]
    ])
    ref_avgs = np.array([6.22070078, -7.66527404, 4.31748398])
    ref_avg = 10.77470242

    thresh = 1.5e-4
    assert np.all(np.abs(ref - hyperpolarizability) < thresh)

    print('hyperpolarizability (static), symmetry-unique components')
    print(hyperpolarizability)

    # Assume no symmetry and calculate the full tensor.

    hyperpolarizability_full = np.zeros(shape=(3, 3, 3))
    for p in product(range(3), range(3), range(3)):
        a, b, c = p
        tl, tr = 0, 0
        for q in permutations(p, 3):
            d, e, f = q
            tl += np.trace(np.dot(rspmats[d, ...], np.dot(G[e, ...], rspmats[f, ...]))[:nocc_alph, :nocc_alph])
            tr += np.trace(np.dot(rspmats[d, ...], np.dot(rspmats[e, ...], epsilon[f, ...]))[:nocc_alph, :nocc_alph])
        hyperpolarizability_full[a, b, c] = 2 * (tl - tr)
    print('hyperpolarizability (static), full tensor')
    print(hyperpolarizability_full)

    for r in range(6):
        b = off1[r]
        c = off2[r]
        for a in range(3):
            diff = hyperpolarizability[r, a] - hyperpolarizability_full[a, b, c]
            assert abs(diff) < 1.0e-14

    return


def test_first_hyperpolarizability_shg_rhf_wigner_explicit():
    mol = molecule_water_sto3g_angstrom()
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = utils.occupations_from_pyscf_mol(mol, C)
    nocc_alph, nvirt_alph, _, _ = occupations
    nov_alph = nocc_alph * nvirt_alph
    norb = nocc_alph + nvirt_alph

    # calculate linear response vectors for electric dipole operator
    f1 = 0.0773178
    f2 = 2 * f1
    frequencies = [f1, f2]
    calculator = electric.Polarizability(mol, C, E, occupations, frequencies=frequencies)
    calculator.form_operators()
    calculator.run()
    calculator.form_results()

    polarizability_1 = calculator.polarizabilities[0]
    polarizability_2 = calculator.polarizabilities[1]
    print('polarizability: {} a.u.'.format(f1))
    print(polarizability_1)
    print('polarizability: {} a.u. (frequency doubled)'.format(f2))
    print(polarizability_2)

    # each operator contains multiple sets of response vectors, one
    # set of components for each frequency
    assert type(calculator.driver.solver.operators) == list
    assert len(calculator.driver.solver.operators) == 1
    operator = calculator.driver.solver.operators[0]
    rhsvecs = operator.mo_integrals_ai_supervector_alph
    assert type(operator.rspvecs_alph) == list
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
    for icomp in range(ncomp):
        rspvec_1 = rspvecs_1[icomp, :, 0]
        rspvec_2 = rspvecs_2[icomp, :, 0]
        x_1 = rspvec_1[:nov_alph]
        y_1 = rspvec_1[nov_alph:]
        x_2 = rspvec_2[:nov_alph]
        y_2 = rspvec_2[nov_alph:]
        x_full_1 = utils.repack_vector_to_matrix(x_1, (nvirt_alph, nocc_alph))
        y_full_1 = utils.repack_vector_to_matrix(y_1, (nvirt_alph, nocc_alph))
        x_full_2 = utils.repack_vector_to_matrix(x_2, (nvirt_alph, nocc_alph))
        y_full_2 = utils.repack_vector_to_matrix(y_2, (nvirt_alph, nocc_alph))
        rspmats_1[icomp, :nocc_alph, nocc_alph:] = y_full_1.T
        rspmats_1[icomp, nocc_alph:, :nocc_alph] = x_full_1
        rspmats_2[icomp, :nocc_alph, nocc_alph:] = y_full_2.T
        rspmats_2[icomp, nocc_alph:, :nocc_alph] = x_full_2

    rhsmats = np.zeros(shape=(ncomp, norb, norb))
    for icomp in range(ncomp):
        rhsvec = rhsvecs[icomp, :, 0]
        rhsvec_top = rhsvec[:nov_alph]
        rhsvec_bot = rhsvec[nov_alph:]
        rhsvec_top_mat = utils.repack_vector_to_matrix(rhsvec_top, (nvirt_alph, nocc_alph))
        rhsvec_bot_mat = utils.repack_vector_to_matrix(rhsvec_bot, (nvirt_alph, nocc_alph))
        rhsmats[icomp, :nocc_alph, nocc_alph:] = rhsvec_top_mat.T
        rhsmats[icomp, nocc_alph:, :nocc_alph] = rhsvec_bot_mat

    polarizability_full_1 = np.empty_like(polarizability_1)
    polarizability_full_2 = np.empty_like(polarizability_2)
    for a in (0, 1, 2):
        for b in (0, 1, 2):
            polarizability_full_1[a, b] = 2 * np.trace(np.dot(rhsmats[a, ...].T,
                                                              rspmats_1[b, ...]))
            polarizability_full_2[a, b] = 2 * np.trace(np.dot(rhsmats[a, ...].T,
                                                              rspmats_2[b, ...]))

    np.testing.assert_almost_equal(polarizability_1, -polarizability_full_1)
    np.testing.assert_almost_equal(polarizability_2, -polarizability_full_2)

    # V_{p,q} <- full MO transformation of right hand side
    integrals_ao = operator.ao_integrals
    integrals_mo = np.empty_like(integrals_ao)
    for icomp in range(ncomp):
        integrals_mo[icomp, ...] = np.dot(C[0, ...].T,
                                          np.dot(integrals_ao[icomp, ...],
                                                 C[0, ...]))

    # from pyresponse.ao2mo import AO2MOpyscf
    # ao2mo = AO2MOpyscf(C, pyscfmol=mol)
    # ao2mo.perform_rhf_full()
    # tei_mo = ao2mo.tei_mo[0]

    G_1 = np.empty_like(rspmats_1)
    G_2 = np.empty_like(rspmats_2)
    C = mf.mo_coeff
    # TODO I feel as though if I have all the MO-basis two-electron
    # integrals, I shouldn't need another JK build.
    for icomp in range(ncomp):
        V = integrals_mo[icomp, ...]
        Dl_1 = np.dot(C[:, :nocc_alph], np.dot(rspmats_1[icomp, :nocc_alph, :], C.T))
        Dr_1 = -np.dot(C, np.dot(rspmats_1[icomp, :, :nocc_alph], C[:, :nocc_alph].T))
        D_1 = Dl_1 + Dr_1
        Dl_2 = np.dot(C[:, :nocc_alph], np.dot(rspmats_2[icomp, :nocc_alph, :], C.T))
        Dr_2 = -np.dot(C, np.dot(rspmats_2[icomp, :, :nocc_alph], C[:, :nocc_alph].T))
        D_2 = Dl_2 + Dr_2
        J_1, K_1 = mf.get_jk(mol, D_1, hermi=0)
        J_2, K_2 = mf.get_jk(mol, D_2, hermi=0)
        F_AO_1 = 2*J_1 - K_1
        F_AO_2 = 2*J_2 - K_2
        F_MO_1 = np.dot(C.T, np.dot(F_AO_1, C))
        F_MO_2 = np.dot(C.T, np.dot(F_AO_2, C))
        G_1[icomp, ...] = V + F_MO_1
        G_2[icomp, ...] = V + F_MO_2

    E_diag = np.diag(E[0, ...])
    epsilon_1 = G_1.copy()
    epsilon_2 = G_2.copy()
    for icomp in range(ncomp):
        eoU_1 = (E_diag[..., np.newaxis] + f1) * rspmats_1[icomp, ...]
        Ue_1 = rspmats_1[icomp, ...] * E_diag[np.newaxis, ...]
        epsilon_1[icomp, ...] += (eoU_1 - Ue_1)
        eoU_2 = (E_diag[..., np.newaxis] + f2) * rspmats_2[icomp, ...]
        Ue_2 = rspmats_2[icomp, ...] * E_diag[np.newaxis, ...]
        epsilon_2[icomp, ...] += (eoU_2 - Ue_2)

    # Assume some symmetry and calculate only part of the tensor.

    hyperpolarizability = np.zeros(shape=(6, 3))
    off1 = [0, 1, 2, 0, 0, 1]
    off2 = [0, 1, 2, 1, 2, 2]
    for r in range(6):
        b = off1[r]
        c = off2[r]
        for a in range(3):
            tl1 = np.trace(np.dot(rspmats_2[a, ...].T, np.dot(G_1[b, ...], rspmats_1[c, ...]))[:nocc_alph, :nocc_alph])
            tl2 = np.trace(np.dot(rspmats_1[c, ...], np.dot(G_1[b, ...], rspmats_2[a, ...].T))[:nocc_alph, :nocc_alph])
            tl3 = np.trace(np.dot(rspmats_2[a, ...].T, np.dot(G_1[c, ...], rspmats_1[b, ...]))[:nocc_alph, :nocc_alph])
            tl4 = np.trace(np.dot(rspmats_1[b, ...], np.dot(G_1[c, ...], rspmats_2[a, ...].T))[:nocc_alph, :nocc_alph])
            tl5 = np.trace(np.dot(rspmats_1[c, ...], np.dot(-G_2[a, ...].T, rspmats_1[b, ...]))[:nocc_alph, :nocc_alph])
            tl6 = np.trace(np.dot(rspmats_1[b, ...], np.dot(-G_2[a, ...].T, rspmats_1[c, ...]))[:nocc_alph, :nocc_alph])
            tr1 = np.trace(np.dot(rspmats_1[c, ...], np.dot(rspmats_1[b, ...], -epsilon_2[a, ...].T))[:nocc_alph, :nocc_alph])
            tr2 = np.trace(np.dot(rspmats_1[b, ...], np.dot(rspmats_1[c, ...], -epsilon_2[a, ...].T))[:nocc_alph, :nocc_alph])
            tr3 = np.trace(np.dot(rspmats_1[c, ...], np.dot(rspmats_2[a, ...].T, epsilon_1[b, ...]))[:nocc_alph, :nocc_alph])
            tr4 = np.trace(np.dot(rspmats_2[a, ...].T, np.dot(rspmats_1[c, ...], epsilon_1[b, ...]))[:nocc_alph, :nocc_alph])
            tr5 = np.trace(np.dot(rspmats_1[b, ...], np.dot(rspmats_2[a, ...].T, epsilon_1[c, ...]))[:nocc_alph, :nocc_alph])
            tr6 = np.trace(np.dot(rspmats_2[a, ...].T, np.dot(rspmats_1[b, ...], epsilon_1[c, ...]))[:nocc_alph, :nocc_alph])
            tl = tl1 + tl2 + tl3 + tl4 + tl5 + tl6
            tr = tr1 + tr2 + tr3 + tr4 + tr5 + tr6
            hyperpolarizability[r, a] = -2 * (tl - tr)

    # pylint: disable=C0326
    ref = np.array([
        [-9.36569537,  0.95027236, -0.53517712],
        [ 2.08678513,  5.44243317, -3.12580855],
        [ 0.69963062,  1.71431536, -0.90527592],
        [ 0.97920892,  2.06615040, -1.13929344],
        [-0.55147065, -1.13929389,  0.68517252],
        [-1.14439025, -3.13384693,  1.72859812]
    ])
    ref_avgs = np.array([6.60267484, -8.13583377, 4.58248287])
    ref_avg = 11.43618185

    thresh = 2.5e-5
    assert np.all(np.abs(ref - hyperpolarizability) < thresh)

    print('hyperpolarizability: SHG, (-{}; {}, {}), symmetry-unique components'.format(f2, f1, f1))
    print(hyperpolarizability)

    # Transpose all frequency-doubled quantities (+2w) to get -2w.

    for icomp in range(ncomp):
        rspmats_2[icomp, ...] = rspmats_2[icomp, ...].T
        G_2[icomp, ...] = -G_2[icomp, ...].T
        epsilon_2[icomp, ...] = -epsilon_2[icomp, ...].T

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
            tl1 = np.trace(np.dot(mU[0][a, ...], np.dot(mG[1][b, ...], mU[1][c, ...]))[:nocc_alph, :nocc_alph])
            tl2 = np.trace(np.dot(mU[1][c, ...], np.dot(mG[1][b, ...], mU[0][a, ...]))[:nocc_alph, :nocc_alph])
            tl3 = np.trace(np.dot(mU[0][a, ...], np.dot(mG[1][c, ...], mU[1][b, ...]))[:nocc_alph, :nocc_alph])
            tl4 = np.trace(np.dot(mU[1][b, ...], np.dot(mG[1][c, ...], mU[0][a, ...]))[:nocc_alph, :nocc_alph])
            tl5 = np.trace(np.dot(mU[1][c, ...], np.dot(mG[0][a, ...], mU[1][b, ...]))[:nocc_alph, :nocc_alph])
            tl6 = np.trace(np.dot(mU[1][b, ...], np.dot(mG[0][a, ...], mU[1][c, ...]))[:nocc_alph, :nocc_alph])
            tr1 = np.trace(np.dot(mU[1][c, ...], np.dot(mU[1][b, ...], me[0][a, ...]))[:nocc_alph, :nocc_alph])
            tr2 = np.trace(np.dot(mU[1][b, ...], np.dot(mU[1][c, ...], me[0][a, ...]))[:nocc_alph, :nocc_alph])
            tr3 = np.trace(np.dot(mU[1][c, ...], np.dot(mU[0][a, ...], me[1][b, ...]))[:nocc_alph, :nocc_alph])
            tr4 = np.trace(np.dot(mU[0][a, ...], np.dot(mU[1][c, ...], me[1][b, ...]))[:nocc_alph, :nocc_alph])
            tr5 = np.trace(np.dot(mU[1][b, ...], np.dot(mU[0][a, ...], me[1][c, ...]))[:nocc_alph, :nocc_alph])
            tr6 = np.trace(np.dot(mU[0][a, ...], np.dot(mU[1][b, ...], me[1][c, ...]))[:nocc_alph, :nocc_alph])
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
            tlp = np.dot(mU[d[1]][d[0], ...], np.dot(mG[e[1]][e[0], ...], mU[f[1]][f[0], ...]))
            tle = np.trace(tlp[:nocc_alph, :nocc_alph])
            tl.append(tle)
            trp = np.dot(mU[d[1]][d[0], ...], np.dot(mU[e[1]][e[0], ...], me[f[1]][f[0], ...]))
            tre = np.trace(trp[:nocc_alph, :nocc_alph])
            tr.append(tre)
        hyperpolarizability_full[a, b, c] = -2 * (sum(tl) - sum(tr))
    print('hyperpolarizability: SHG, (-{}; {}, {}), full tensor'.format(f2, f1, f1))
    print(hyperpolarizability_full)

    for r in range(6):
        b = off1[r]
        c = off2[r]
        for a in range(3):
            diff = hyperpolarizability[r, a] - hyperpolarizability_full[a, b, c]
            assert abs(diff) < 1.0e-14

    return


def test_first_hyperpolarizability_eope_rhf_wigner_explicit():
    mol = molecule_water_sto3g_angstrom()
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = utils.occupations_from_pyscf_mol(mol, C)
    nocc_alph, nvirt_alph, _, _ = occupations
    nov_alph = nocc_alph * nvirt_alph
    norb = nocc_alph + nvirt_alph

    # calculate linear response vectors for electric dipole operator
    f1 = 0.0
    f2 = 0.0773178
    frequencies = [f1, f2]
    calculator = electric.Polarizability(mol, C, E, occupations, frequencies=frequencies)
    calculator.form_operators()
    calculator.run()
    calculator.form_results()

    polarizability_1 = calculator.polarizabilities[0]
    polarizability_2 = calculator.polarizabilities[1]
    print('polarizability: {} a.u.'.format(f1))
    print(polarizability_1)
    print('polarizability: {} a.u. (frequency doubled)'.format(f2))
    print(polarizability_2)

    # each operator contains multiple sets of response vectors, one
    # set of components for each frequency
    assert type(calculator.driver.solver.operators) == list
    assert len(calculator.driver.solver.operators) == 1
    operator = calculator.driver.solver.operators[0]
    rhsvecs = operator.mo_integrals_ai_supervector_alph
    assert type(operator.rspvecs_alph) == list
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
    for icomp in range(ncomp):
        rspvec_1 = rspvecs_1[icomp, :, 0]
        rspvec_2 = rspvecs_2[icomp, :, 0]
        x_1 = rspvec_1[:nov_alph]
        y_1 = rspvec_1[nov_alph:]
        x_2 = rspvec_2[:nov_alph]
        y_2 = rspvec_2[nov_alph:]
        x_full_1 = utils.repack_vector_to_matrix(x_1, (nvirt_alph, nocc_alph))
        y_full_1 = utils.repack_vector_to_matrix(y_1, (nvirt_alph, nocc_alph))
        x_full_2 = utils.repack_vector_to_matrix(x_2, (nvirt_alph, nocc_alph))
        y_full_2 = utils.repack_vector_to_matrix(y_2, (nvirt_alph, nocc_alph))
        rspmats_1[icomp, :nocc_alph, nocc_alph:] = y_full_1.T
        rspmats_1[icomp, nocc_alph:, :nocc_alph] = x_full_1
        rspmats_2[icomp, :nocc_alph, nocc_alph:] = y_full_2.T
        rspmats_2[icomp, nocc_alph:, :nocc_alph] = x_full_2

    rhsmats = np.zeros(shape=(ncomp, norb, norb))
    for icomp in range(ncomp):
        rhsvec = rhsvecs[icomp, :, 0]
        rhsvec_top = rhsvec[:nov_alph]
        rhsvec_bot = rhsvec[nov_alph:]
        rhsvec_top_mat = utils.repack_vector_to_matrix(rhsvec_top, (nvirt_alph, nocc_alph))
        rhsvec_bot_mat = utils.repack_vector_to_matrix(rhsvec_bot, (nvirt_alph, nocc_alph))
        rhsmats[icomp, :nocc_alph, nocc_alph:] = rhsvec_top_mat.T
        rhsmats[icomp, nocc_alph:, :nocc_alph] = rhsvec_bot_mat

    polarizability_full_1 = np.empty_like(polarizability_1)
    polarizability_full_2 = np.empty_like(polarizability_2)
    for a in (0, 1, 2):
        for b in (0, 1, 2):
            polarizability_full_1[a, b] = 2 * np.trace(np.dot(rhsmats[a, ...].T,
                                                              rspmats_1[b, ...]))
            polarizability_full_2[a, b] = 2 * np.trace(np.dot(rhsmats[a, ...].T,
                                                              rspmats_2[b, ...]))

    np.testing.assert_almost_equal(polarizability_1, -polarizability_full_1)
    np.testing.assert_almost_equal(polarizability_2, -polarizability_full_2)

    # V_{p,q} <- full MO transformation of right hand side
    integrals_ao = operator.ao_integrals
    integrals_mo = np.empty_like(integrals_ao)
    for icomp in range(ncomp):
        integrals_mo[icomp, ...] = np.dot(C[0, ...].T,
                                          np.dot(integrals_ao[icomp, ...],
                                                 C[0, ...]))

    # from pyresponse.ao2mo import AO2MOpyscf
    # ao2mo = AO2MOpyscf(C, pyscfmol=mol)
    # ao2mo.perform_rhf_full()
    # tei_mo = ao2mo.tei_mo[0]

    G_1 = np.empty_like(rspmats_1)
    G_2 = np.empty_like(rspmats_2)
    C = mf.mo_coeff
    # TODO I feel as though if I have all the MO-basis two-electron
    # integrals, I shouldn't need another JK build.
    for icomp in range(ncomp):
        V = integrals_mo[icomp, ...]
        Dl_1 = np.dot(C[:, :nocc_alph], np.dot(rspmats_1[icomp, :nocc_alph, :], C.T))
        Dr_1 = -np.dot(C, np.dot(rspmats_1[icomp, :, :nocc_alph], C[:, :nocc_alph].T))
        D_1 = Dl_1 + Dr_1
        Dl_2 = np.dot(C[:, :nocc_alph], np.dot(rspmats_2[icomp, :nocc_alph, :], C.T))
        Dr_2 = -np.dot(C, np.dot(rspmats_2[icomp, :, :nocc_alph], C[:, :nocc_alph].T))
        D_2 = Dl_2 + Dr_2
        J_1, K_1 = mf.get_jk(mol, D_1, hermi=0)
        J_2, K_2 = mf.get_jk(mol, D_2, hermi=0)
        F_AO_1 = 2*J_1 - K_1
        F_AO_2 = 2*J_2 - K_2
        F_MO_1 = np.dot(C.T, np.dot(F_AO_1, C))
        F_MO_2 = np.dot(C.T, np.dot(F_AO_2, C))
        G_1[icomp, ...] = V + F_MO_1
        G_2[icomp, ...] = V + F_MO_2

    E_diag = np.diag(E[0, ...])
    epsilon_1 = G_1.copy()
    epsilon_2 = G_2.copy()
    for icomp in range(ncomp):
        eoU_1 = (E_diag[..., np.newaxis] + f1) * rspmats_1[icomp, ...]
        Ue_1 = rspmats_1[icomp, ...] * E_diag[np.newaxis, ...]
        epsilon_1[icomp, ...] += (eoU_1 - Ue_1)
        eoU_2 = (E_diag[..., np.newaxis] + f2) * rspmats_2[icomp, ...]
        Ue_2 = rspmats_2[icomp, ...] * E_diag[np.newaxis, ...]
        epsilon_2[icomp, ...] += (eoU_2 - Ue_2)

    # Assume some symmetry and calculate only part of the tensor.

    hyperpolarizability = np.zeros(shape=(6, 3))
    off1 = [0, 1, 2, 0, 0, 1]
    off2 = [0, 1, 2, 1, 2, 2]
    for r in range(6):
        b = off1[r]
        c = off2[r]
        for a in range(3):
            # _1 -> static (w = 0)
            # _2 -> perturbation (dynamic)
            # b is _1, c is _2, a is _2 transposed/negated
            tl1 = np.trace(np.dot(rspmats_2[a, ...].T, np.dot(G_1[b, ...], rspmats_2[c, ...]))[:nocc_alph, :nocc_alph])
            tl2 = np.trace(np.dot(rspmats_2[c, ...], np.dot(G_1[b, ...], rspmats_2[a, ...].T))[:nocc_alph, :nocc_alph])
            tl3 = np.trace(np.dot(rspmats_2[a, ...].T, np.dot(G_2[c, ...], rspmats_1[b, ...]))[:nocc_alph, :nocc_alph])
            tl4 = np.trace(np.dot(rspmats_1[b, ...], np.dot(G_2[c, ...], rspmats_2[a, ...].T))[:nocc_alph, :nocc_alph])
            tl5 = np.trace(np.dot(rspmats_2[c, ...], np.dot(-G_2[a, ...].T, rspmats_1[b, ...]))[:nocc_alph, :nocc_alph])
            tl6 = np.trace(np.dot(rspmats_1[b, ...], np.dot(-G_2[a, ...].T, rspmats_2[c, ...]))[:nocc_alph, :nocc_alph])
            tr1 = np.trace(np.dot(rspmats_2[c, ...], np.dot(rspmats_1[b, ...], -epsilon_2[a, ...].T))[:nocc_alph, :nocc_alph])
            tr2 = np.trace(np.dot(rspmats_1[b, ...], np.dot(rspmats_2[c, ...], -epsilon_2[a, ...].T))[:nocc_alph, :nocc_alph])
            tr3 = np.trace(np.dot(rspmats_2[c, ...], np.dot(rspmats_2[a, ...].T, epsilon_1[b, ...]))[:nocc_alph, :nocc_alph])
            tr4 = np.trace(np.dot(rspmats_2[a, ...].T, np.dot(rspmats_2[c, ...], epsilon_1[b, ...]))[:nocc_alph, :nocc_alph])
            tr5 = np.trace(np.dot(rspmats_1[b, ...], np.dot(rspmats_2[a, ...].T, epsilon_2[c, ...]))[:nocc_alph, :nocc_alph])
            tr6 = np.trace(np.dot(rspmats_2[a, ...].T, np.dot(rspmats_1[b, ...], epsilon_2[c, ...]))[:nocc_alph, :nocc_alph])
            tl = tl1 + tl2 + tl3 + tl4 + tl5 + tl6
            tr = tr1 + tr2 + tr3 + tr4 + tr5 + tr6
            hyperpolarizability[r, a] = -2 * (tl - tr)

    # pylint: disable=C0326
    ref = np.array([
        [-9.02854579,  0.92021130, -0.51824440],
        [ 2.01769267,  5.23470702, -3.00950655],
        [ 0.67140102,  1.65659586, -0.87205853],
        [ 0.92021130,  2.01080066, -1.10887175],
        [-0.51824440, -1.10887175,  0.66669794],
        [-1.11067586, -3.00950655,  1.66112712]
    ])
    ref_avgs = np.array([6.34718216, -7.81453502, 4.39980948])
    ref_avg = 10.98690140

    thresh = 4.0e-5
    assert np.all(np.abs(ref - hyperpolarizability) < thresh)

    print('hyperpolarizability: EOPE, (-{}; {}, {}), symmetry-unique components'.format(f2, f1, f2))
    print(hyperpolarizability)

    # # Transpose all frequency-doubled quantities (+2w) to get -2w.

    # for icomp in range(ncomp):
    #     rspmats_2[icomp, ...] = rspmats_2[icomp, ...].T
    #     G_2[icomp, ...] = -G_2[icomp, ...].T
    #     epsilon_2[icomp, ...] = -epsilon_2[icomp, ...].T

    # # Assume some symmetry and calculate only part of the tensor. This
    # # time, work with the in-place manipulated quantities (this tests
    # # their correctness).

    # mU = (rspmats_2, rspmats_1)
    # mG = (G_2, G_1)
    # me = (epsilon_2, epsilon_1)

    # hyperpolarizability = np.zeros(shape=(6, 3))
    # off1 = [0, 1, 2, 0, 0, 1]
    # off2 = [0, 1, 2, 1, 2, 2]
    # for r in range(6):
    #     b = off1[r]
    #     c = off2[r]
    #     for a in range(3):
    #         tl1 = np.trace(np.dot(mU[0][a, ...], np.dot(mG[1][b, ...], mU[1][c, ...]))[:nocc_alph, :nocc_alph])
    #         tl2 = np.trace(np.dot(mU[1][c, ...], np.dot(mG[1][b, ...], mU[0][a, ...]))[:nocc_alph, :nocc_alph])
    #         tl3 = np.trace(np.dot(mU[0][a, ...], np.dot(mG[1][c, ...], mU[1][b, ...]))[:nocc_alph, :nocc_alph])
    #         tl4 = np.trace(np.dot(mU[1][b, ...], np.dot(mG[1][c, ...], mU[0][a, ...]))[:nocc_alph, :nocc_alph])
    #         tl5 = np.trace(np.dot(mU[1][c, ...], np.dot(mG[0][a, ...], mU[1][b, ...]))[:nocc_alph, :nocc_alph])
    #         tl6 = np.trace(np.dot(mU[1][b, ...], np.dot(mG[0][a, ...], mU[1][c, ...]))[:nocc_alph, :nocc_alph])
    #         tr1 = np.trace(np.dot(mU[1][c, ...], np.dot(mU[1][b, ...], me[0][a, ...]))[:nocc_alph, :nocc_alph])
    #         tr2 = np.trace(np.dot(mU[1][b, ...], np.dot(mU[1][c, ...], me[0][a, ...]))[:nocc_alph, :nocc_alph])
    #         tr3 = np.trace(np.dot(mU[1][c, ...], np.dot(mU[0][a, ...], me[1][b, ...]))[:nocc_alph, :nocc_alph])
    #         tr4 = np.trace(np.dot(mU[0][a, ...], np.dot(mU[1][c, ...], me[1][b, ...]))[:nocc_alph, :nocc_alph])
    #         tr5 = np.trace(np.dot(mU[1][b, ...], np.dot(mU[0][a, ...], me[1][c, ...]))[:nocc_alph, :nocc_alph])
    #         tr6 = np.trace(np.dot(mU[0][a, ...], np.dot(mU[1][b, ...], me[1][c, ...]))[:nocc_alph, :nocc_alph])
    #         tl = [tl1, tl2, tl3, tl4, tl5, tl6]
    #         tr = [tr1, tr2, tr3, tr4, tr5, tr6]
    #         hyperpolarizability[r, a] = -2 * (sum(tl) - sum(tr))

    # assert np.all(np.abs(ref - hyperpolarizability) < thresh)

    # # Assume no symmetry and calculate the full tensor.

    # hyperpolarizability_full = np.zeros(shape=(3, 3, 3))

    # # components x, y, z
    # for ip, p in enumerate(list(product(range(3), range(3), range(3)))):
    #     a, b, c = p
    #     tl, tr = [], []
    #     # 1st tuple -> index a, b, c (*not* x, y, z!)
    #     # 2nd tuple -> index frequency (0 -> -2w, 1 -> +w)
    #     for iq, q in enumerate(list(permutations(zip(p, (0, 1, 1)), 3))):
    #         d, e, f = q
    #         tlp = np.dot(mU[d[1]][d[0], ...], np.dot(mG[e[1]][e[0], ...], mU[f[1]][f[0], ...]))
    #         tle = np.trace(tlp[:nocc_alph, :nocc_alph])
    #         tl.append(tle)
    #         trp = np.dot(mU[d[1]][d[0], ...], np.dot(mU[e[1]][e[0], ...], me[f[1]][f[0], ...]))
    #         tre = np.trace(trp[:nocc_alph, :nocc_alph])
    #         tr.append(tre)
    #     hyperpolarizability_full[a, b, c] = -2 * (sum(tl) - sum(tr))
    # print('hyperpolarizability: SHG, (-{}; {}, {}), full tensor'.format(f2, f1, f1))
    # print(hyperpolarizability_full)

    # for r in range(6):
    #     b = off1[r]
    #     c = off2[r]
    #     for a in range(3):
    #         diff = hyperpolarizability[r, a] - hyperpolarizability_full[a, b, c]
    #         assert abs(diff) < 1.0e-14

    return


if __name__ == '__main__':
    test_first_hyperpolarizability_static_rhf_wigner_explicit()
    test_first_hyperpolarizability_shg_rhf_wigner_explicit()
    test_first_hyperpolarizability_eope_rhf_wigner_explicit()
