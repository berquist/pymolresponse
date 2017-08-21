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
    print('polarizability')
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

    polarizability = calculator.polarizabilities[0]
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
        eoU = (E_diag[..., np.newaxis] - omega) * rspmats[icomp, ...]
        Ue = rspmats[icomp, ...] * E_diag[np.newaxis, ...]
        epsilon[icomp, ...] += (eoU - Ue)

    # Assume some symmetry and calculate only part of the tensor.

    hyperpolarizability = np.zeros(shape=(6, 3))
    off1 = [0, 1, 2, 0, 0, 1]
    off2 = [0, 1, 2, 1, 2, 2]
    for r in range(6):
        b = off1[r]
        c = off2[r]
        for i in range(3):
            a = i
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
            hyperpolarizability[r, i] = 2 * (tl - tr)

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
    print('hyperpolarizability')
    print(hyperpolarizability_full)
