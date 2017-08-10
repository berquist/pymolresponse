#!/usr/bin/env python

import os.path
import numpy as np
import pyscf
from pyresponse import utils, electric

__filedir__ = os.path.realpath(os.path.dirname(__file__))
refdir = os.path.join(__filedir__, 'reference_data')

def molecule_water_sto3g(verbose=0):

    mol = pyscf.gto.Mole()
    mol.verbose = verbose
    mol.output = None

    with open(os.path.join(refdir, 'water.xyz')) as fh:
        mol.atom = fh.read()
    mol.basis = 'sto-3g'
    mol.charge = 0
    mol.spin = 0

    mol.unit = 'Bohr'

    return mol

mol = molecule_water_sto3g()
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

operator = calculator.driver.solver.operators[0]
# print(operator)
# print('\n'.join(dir(operator)))
rhsvecs = operator.mo_integrals_ai_supervector_alph
rspvecs = operator.rspvecs_alph[0]

# transform X_{ia} and Y_{ia} -> U_{p,q}
# 0. 'a' is fast index, 'i' slow
# 1. rspvec == [X Y]
# 2. U_{p, q} -> zero
# 3. place X_{ia} into U_{i, a}
# 4. place Y_{ia} into U_{a, i}

np.set_printoptions(linewidth=200)

ncomp = rhsvecs.shape[0]
rhsvecs_full = np.zeros(shape=(ncomp, norb, norb))
for icomp in range(ncomp):
    rhsvec = rhsvecs[icomp, :, 0]
    x = rhsvec[:nov_alph]
    y = rhsvec[nov_alph:]
    x_full = utils.repack_vector_to_matrix(x, (nvirt_alph, nocc_alph))
    y_full = utils.repack_vector_to_matrix(y, (nvirt_alph, nocc_alph))
    rhsvecs_full[icomp, :nocc_alph, nocc_alph:] = x_full.T
    rhsvecs_full[icomp, nocc_alph:, :nocc_alph] = y_full
rspvecs_full = np.zeros(shape=(ncomp, norb, norb))
for icomp in range(ncomp):
    rspvec = rspvecs[icomp, :, 0]
    x = rspvec[:nov_alph]
    y = rspvec[nov_alph:]
    x_full = utils.repack_vector_to_matrix(x, (nvirt_alph, nocc_alph))
    y_full = utils.repack_vector_to_matrix(y, (nvirt_alph, nocc_alph))
    rspvecs_full[icomp, :nocc_alph, nocc_alph:] = x_full.T
    rspvecs_full[icomp, nocc_alph:, :nocc_alph] = y_full

polarizability = calculator.polarizabilities[0]
polarizability_full = np.empty_like(polarizability)
for a in (0, 1, 2):
    for b in (0, 1, 2):
        polarizability_full[a, b] = 2 * np.trace(np.dot(rhsvecs_full[a, ...].T,
                                                        rspvecs_full[b, ...]))

np.testing.assert_almost_equal(polarizability, polarizability_full)

# Turns out: won't need 2nd-order quantities: see eqn. (VII-4)
# for a in (0, 1, 2):
#     rspvec_a = rspvecs_full[a, ...]
#     for b in (0, 1, 2):
#         rspvec_b = rspvecs_full[b, ...]
#         diag_ab = 0.5 * (np.dot(rspvec_a, rspvec_b) + np.dot(rspvec_b, rspvec_a))
#         # print(a, b)
#         # print(diag_ab)

# TODO: transform G_{ia,jb} -> ?
# U^{ab} <-
# C^{ab} <- C^{0} U^{ab}
#  <= 2nd-order perturbed MO coefficients from ground-state MO
#     coefficients and 2nd-order orbital rotation matrix
# D^{ab} <- C^{ab}nC^{0,T} + C^{a}nC^{b,T} + C^{b}nC^{a,T} + C^{0}nC^{ab,T}
#  <= 2nd-order perturbed density
# F^{ab} <- D^{ab}[2J^{0} - K^{0}]
#  <= bra AO, ket MO perturbed Fock matrix {munu}
# D_{\lambda\sigma}^{ab} [2(\mu\nu|\lambda\sigma) - (\mu\sigma|\lambda\nu)]
# G^{ab} <- C^{0,T} F^{ab} C^{0}
#  <= bra MO, ket MO perturbed Fock matrix {pq}
