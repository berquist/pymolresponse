import numpy as np
import scipy as sp

from cphf import CPHF
from operators import Operator
from utils import form_results

from explicit_equations_full import \
    (form_rpa_a_matrix_mo_singlet_full,
     form_rpa_a_matrix_mo_singlet_ss_full,
     form_rpa_a_matrix_mo_singlet_os_full,
     form_rpa_a_matrix_mo_triplet_full,
     form_rpa_b_matrix_mo_singlet_full,
     form_rpa_b_matrix_mo_singlet_ss_full,
     form_rpa_b_matrix_mo_singlet_os_full,
     form_rpa_b_matrix_mo_triplet_full)
from explicit_equations_partial import \
    (form_rpa_a_matrix_mo_singlet_partial,
     form_rpa_a_matrix_mo_singlet_ss_partial,
     form_rpa_a_matrix_mo_singlet_os_partial,
     form_rpa_a_matrix_mo_triplet_partial,
     form_rpa_b_matrix_mo_singlet_partial,
     form_rpa_b_matrix_mo_singlet_ss_partial,
     form_rpa_b_matrix_mo_singlet_os_partial,
     form_rpa_b_matrix_mo_triplet_partial)


class TDHF(CPHF):

    def __init__(self, mocoeffs, moenergies, occupations, *args, **kwargs):
        # don't call superclass __init__
        assert len(mocoeffs.shape) == 3
        assert (mocoeffs.shape[0] == 1) or (mocoeffs.shape[0] == 2)
        self.is_uhf = (mocoeffs.shape[0] == 2)
        assert len(moenergies.shape) == 3
        assert (moenergies.shape[0] == 1) or (moenergies.shape[0] == 2)
        if self.is_uhf:
            assert moenergies.shape[0] == 2
        else:
            assert moenergies.shape[0] == 1
        assert moenergies.shape[1] == moenergies.shape[2]
        assert len(occupations) == 4

        self.mocoeffs = mocoeffs
        self.moenergies = moenergies
        self.occupations = occupations
        self.form_ranges_from_occupations()

        self.solver = 'explicit'
        self.hamiltonian = 'rpa'
        self.spin = 'singlet'

        self.operators = []
        self.frequencies = []
        self.results = []

    def run(self, solver=None, hamiltonian=None, spin=None, **kwargs):

        if not solver:
            solver = self.solver
        if not hamiltonian:
            hamiltonian = self.hamiltonian
        if not spin:
            spin = self.spin

        # Set the current state.
        self.solver = solver
        self.hamiltonian = hamiltonian
        self.spin = spin

        if solver == 'explicit':
            self.form_explicit_hessian(hamiltonian, spin, None)
            self.diagonalize_explicit_hessian()
        # Nothing else implemented yet.
        else:
            pass

        self.form_results()

    def form_explicit_hessian(self, hamiltonian=None, spin=None, frequency=None):

        assert hasattr(self, 'tei_mo')
        assert self.tei_mo is not None
        assert len(self.tei_mo) in (1, 2, 4, 6)
        assert self.tei_mo_type in ('full', 'partial')

        if not hamiltonian:
            hamiltonian = self.hamiltonian
        if not spin:
            spin = self.spin

        assert hamiltonian in ('rpa', 'tda')
        assert spin in ('singlet', 'triplet')

        self.hamiltonian = hamiltonian
        self.spin = spin

        nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = self.occupations
        nov_alph = nocc_alph * nvirt_alph
        nov_beta = nocc_beta * nvirt_beta

        if not self.is_uhf:

            # Set up "function pointers".
            if self.tei_mo_type == 'full':
                assert len(self.tei_mo) == 1
                tei_mo = self.tei_mo[0]
            elif self.tei_mo_type == 'partial':
                assert len(self.tei_mo) == 2
                tei_mo_ovov = self.tei_mo[0]
                tei_mo_oovv = self.tei_mo[1]

            if self.tei_mo_type == 'full':
                if hamiltonian == 'rpa' and spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_full(self.moenergies[0, ...], tei_mo, nocc_alph)
                    B = form_rpa_b_matrix_mo_singlet_full(tei_mo, nocc_alph)
                elif hamiltonian == 'rpa' and spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_full(self.moenergies[0, ...], tei_mo, nocc_alph)
                    B = form_rpa_b_matrix_mo_triplet_full(tei_mo, nocc_alph)
                elif hamiltonian == 'tda' and spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_full(self.moenergies[0, ...], tei_mo, nocc_alph)
                    B = np.zeros(shape=(nov_alph, nov_alph))
                elif hamiltonian == 'tda' and spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_full(self.moenergies[0, ...], tei_mo, nocc_alph)
                    B = np.zeros(shape=(nov_alph, nov_alph))
            elif self.tei_mo_type == 'partial':
                if hamiltonian == 'rpa' and spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_partial(self.moenergies[0, ...], tei_mo_ovov, tei_mo_oovv)
                    B = form_rpa_b_matrix_mo_singlet_partial(tei_mo_ovov)
                elif hamiltonian == 'rpa' and spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_partial(self.moenergies[0, ...], tei_mo_oovv)
                    B = form_rpa_b_matrix_mo_triplet_partial(tei_mo_ovov)
                elif hamiltonian == 'tda' and spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_partial(self.moenergies[0, ...], tei_mo_ovov, tei_mo_oovv)
                    B = np.zeros(shape=(nov_alph, nov_alph))
                elif hamiltonian == 'tda' and spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_partial(self.moenergies[0, ...], tei_mo_oovv)
                    B = np.zeros(shape=(nov_alph, nov_alph))

            # pylint: disable=bad-whitespace
            G = np.asarray(np.bmat([[ A,  B],
                                    [-B, -A]]))
            self.explicit_hessian = G

        else:

            # Set up "function pointers".
            if self.tei_mo_type == 'full':
                assert len(self.tei_mo) == 4
                tei_mo_aaaa = self.tei_mo[0]
                tei_mo_aabb = self.tei_mo[1]
                tei_mo_bbaa = self.tei_mo[2]
                tei_mo_bbbb = self.tei_mo[3]
            elif self.tei_mo_type == 'partial':
                assert len(self.tei_mo) == 6
                tei_mo_ovov_aaaa = self.tei_mo[0]
                tei_mo_ovov_aabb = self.tei_mo[1]
                tei_mo_ovov_bbaa = self.tei_mo[2]
                tei_mo_ovov_bbbb = self.tei_mo[3]
                tei_mo_oovv_aaaa = self.tei_mo[4]
                tei_mo_oovv_bbbb = self.tei_mo[5]
            else:
                pass

            E_a = self.moenergies[0, ...]
            E_b = self.moenergies[1, ...]

            # TODO clean this up...
            if self.tei_mo_type == 'full':
                if hamiltonian == 'rpa' and spin == 'singlet':
                    A_ss_a = form_rpa_a_matrix_mo_singlet_ss_full(E_a, tei_mo_aaaa, nocc_alph)
                    A_os_a = form_rpa_a_matrix_mo_singlet_os_full(tei_mo_aabb, nocc_alph, nocc_beta)
                    B_ss_a = form_rpa_b_matrix_mo_singlet_ss_full(tei_mo_aaaa, nocc_alph)
                    B_os_a = form_rpa_b_matrix_mo_singlet_os_full(tei_mo_aabb, nocc_alph, nocc_beta)
                    A_ss_b = form_rpa_a_matrix_mo_singlet_ss_full(E_b, tei_mo_bbbb, nocc_beta)
                    A_os_b = form_rpa_a_matrix_mo_singlet_os_full(tei_mo_bbaa, nocc_beta, nocc_alph)
                    B_ss_b = form_rpa_b_matrix_mo_singlet_ss_full(tei_mo_bbbb, nocc_beta)
                    B_os_b = form_rpa_b_matrix_mo_singlet_os_full(tei_mo_bbaa, nocc_beta, nocc_alph)
                elif hamiltonian == 'rpa' and spin == 'triplet':
                    # Since the "triplet" part contains no Coulomb contribution, and
                    # (xx|yy) is only in the Coulomb part, there is no ss/os
                    # separation for the triplet part.
                    zeros_ab = np.zeros(shape=(nov_alph, nov_beta))
                    zeros_ba = zeros_ab.T
                    A_ss_a = form_rpa_a_matrix_mo_triplet_full(E_a, tei_mo_aaaa, nocc_alph)
                    A_os_a = zeros_ab
                    B_ss_a = form_rpa_b_matrix_mo_triplet_full(tei_mo_aaaa, nocc_alph)
                    B_os_a = zeros_ab
                    A_ss_b = form_rpa_a_matrix_mo_triplet_full(E_b, tei_mo_bbbb, nocc_beta)
                    A_os_b = zeros_ba
                    B_ss_b = form_rpa_b_matrix_mo_triplet_full(tei_mo_bbbb, nocc_beta)
                    B_os_b = zeros_ba
                elif hamiltonian == 'tda' and spin == 'singlet':
                    zeros_aa = np.zeros(shape=(nov_alph, nov_alph))
                    zeros_ab = np.zeros(shape=(nov_alph, nov_beta))
                    zeros_ba = zeros_ab.T
                    zeros_bb = np.zeros(shape=(nov_beta, nov_beta))
                    A_ss_a = form_rpa_a_matrix_mo_singlet_ss_full(E_a, tei_mo_aaaa, nocc_alph)
                    A_os_a = form_rpa_a_matrix_mo_singlet_os_full(tei_mo_aabb, nocc_alph, nocc_beta)
                    B_ss_a = zeros_aa
                    B_os_a = zeros_ab
                    A_ss_b = form_rpa_a_matrix_mo_singlet_ss_full(E_b, tei_mo_bbbb, nocc_beta)
                    A_os_b = form_rpa_a_matrix_mo_singlet_os_full(tei_mo_bbaa, nocc_beta, nocc_alph)
                    B_ss_b = zeros_bb
                    B_os_b = zeros_ba
                elif hamiltonian == 'tda' and spin == 'triplet':
                    zeros_aa = np.zeros(shape=(nov_alph, nov_alph))
                    zeros_ab = np.zeros(shape=(nov_alph, nov_beta))
                    zeros_ba = zeros_ab.T
                    zeros_bb = np.zeros(shape=(nov_beta, nov_beta))
                    A_ss_a = form_rpa_a_matrix_mo_triplet_full(E_a, tei_mo_aaaa, nocc_alph)
                    A_os_a = zeros_ab
                    B_ss_a = zeros_aa
                    B_os_a = zeros_ab
                    A_ss_b = form_rpa_a_matrix_mo_triplet_full(E_b, tei_mo_bbbb, nocc_beta)
                    A_os_b = zeros_ba
                    B_ss_b = zeros_bb
                    B_os_b = zeros_ba

            elif self.tei_mo_type == 'partial':
                if hamiltonian == 'rpa' and spin == 'singlet':
                    A_ss_a = form_rpa_a_matrix_mo_singlet_ss_partial(E_a, tei_mo_ovov_aaaa, tei_mo_oovv_aaaa)
                    A_os_a = form_rpa_a_matrix_mo_singlet_os_partial(tei_mo_ovov_aabb)
                    B_ss_a = form_rpa_b_matrix_mo_singlet_ss_partial(tei_mo_ovov_aaaa)
                    B_os_a = form_rpa_b_matrix_mo_singlet_os_partial(tei_mo_ovov_aabb)
                    A_ss_b = form_rpa_a_matrix_mo_singlet_ss_partial(E_b, tei_mo_ovov_bbbb, tei_mo_oovv_bbbb)
                    A_os_b = form_rpa_a_matrix_mo_singlet_os_partial(tei_mo_ovov_bbaa)
                    B_ss_b = form_rpa_b_matrix_mo_singlet_ss_partial(tei_mo_ovov_bbbb)
                    B_os_b = form_rpa_b_matrix_mo_singlet_os_partial(tei_mo_ovov_bbaa)
                elif hamiltonian == 'rpa' and spin == 'triplet':
                    # Since the "triplet" part contains no Coulomb contribution, and
                    # (xx|yy) is only in the Coulomb part, there is no ss/os
                    # separation for the triplet part.
                    zeros_ab = np.zeros(shape=(nov_alph, nov_beta))
                    zeros_ba = zeros_ab.T
                    A_ss_a = form_rpa_a_matrix_mo_triplet_partial(E_a, tei_mo_oovv_aaaa)
                    A_os_a = zeros_ab
                    B_ss_a = form_rpa_b_matrix_mo_triplet_partial(tei_mo_ovov_aaaa)
                    B_os_a = zeros_ab
                    A_ss_b = form_rpa_a_matrix_mo_triplet_partial(E_b, tei_mo_oovv_bbbb)
                    A_os_b = zeros_ba
                    B_ss_b = form_rpa_b_matrix_mo_triplet_partial(tei_mo_ovov_bbbb)
                    B_os_b = zeros_ba
                elif hamiltonian == 'tda' and spin == 'singlet':
                    zeros_aa = np.zeros(shape=(nov_alph, nov_alph))
                    zeros_ab = np.zeros(shape=(nov_alph, nov_beta))
                    zeros_ba = zeros_ab.T
                    zeros_bb = np.zeros(shape=(nov_beta, nov_beta))
                    A_ss_a = form_rpa_a_matrix_mo_singlet_ss_partial(E_a, tei_mo_ovov_aaaa, tei_mo_oovv_aaaa)
                    A_os_a = form_rpa_a_matrix_mo_singlet_os_partial(tei_mo_ovov_aabb)
                    B_ss_a = zeros_aa
                    B_os_a = zeros_ab
                    A_ss_b = form_rpa_a_matrix_mo_singlet_ss_partial(E_b, tei_mo_ovov_bbbb, tei_mo_oovv_bbbb)
                    A_os_b = form_rpa_a_matrix_mo_singlet_os_partial(tei_mo_ovov_bbaa)
                    B_ss_b = zeros_bb
                    B_os_b = zeros_ba
                elif hamiltonian == 'tda' and spin == 'triplet':
                    zeros_aa = np.zeros(shape=(nov_alph, nov_alph))
                    zeros_ab = np.zeros(shape=(nov_alph, nov_beta))
                    zeros_ba = zeros_ab.T
                    zeros_bb = np.zeros(shape=(nov_beta, nov_beta))
                    A_ss_a = form_rpa_a_matrix_mo_triplet_partial(E_a, tei_mo_oovv_aaaa)
                    A_os_a = zeros_ab
                    B_ss_a = zeros_aa
                    B_os_a = zeros_ab
                    A_ss_b = form_rpa_a_matrix_mo_triplet_partial(E_b, tei_mo_oovv_bbbb)
                    A_os_b = zeros_ba
                    B_ss_b = zeros_bb
                    B_os_b = zeros_ba

            G_aa = np.asarray(np.bmat([[A_ss_a, B_ss_a],
                                       [-B_ss_a, -A_ss_a]]))
            G_ab = np.asarray(np.bmat([[A_os_a, B_os_a],
                                       [-B_os_a, -A_os_a]]))
            G_ba = np.asarray(np.bmat([[A_os_b, B_os_b],
                                       [-B_os_b, -A_os_b]]))
            G_bb = np.asarray(np.bmat([[A_ss_b, B_ss_b],
                                       [-B_ss_b, -A_ss_b]]))

            self.explicit_hessian = (
                G_aa,
                G_ab,
                G_ba,
                G_bb,
            )

    def diagonalize_explicit_hessian(self):
        nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = self.occupations
        nov_alph = nocc_alph * nvirt_alph
        nov_beta = nocc_beta * nvirt_beta
        if not self.is_uhf:
            eigvals, eigvecs = sp.linalg.eig(self.explicit_hessian)
            # Sort from lowest to highest eigenvalue (excitation
            # energy).
            idx = eigvals.argsort()
            self.eigvals = eigvals[idx]
            # Each eigenvector is a column vector.
            self.eigvecs = eigvecs[:, idx]
            # Fix the ordering of everything. The first eigenvectors
            # are those with negative excitation energies.
            self.eigvals = self.eigvals[nov_alph:]
            self.eigvecs = self.eigvecs[:, nov_alph:]
        else:
            assert len(self.explicit_hessian) == 4
            G_aa, G_ab, G_ba, G_bb = self.explicit_hessian
            G_aa_inv = np.linalg.inv(G_aa)
            G_bb_inv = np.linalg.inv(G_bb)
            lhs_alph = G_aa - np.dot(G_ab, np.dot(G_bb_inv, G_ba))
            lhs_beta = G_bb - np.dot(G_ba, np.dot(G_aa_inv, G_ab))
            eigvals_alph, eigvecs_alph = sp.linalg.eig(lhs_alph)
            eigvals_beta, eigvecs_beta = sp.linalg.eig(lhs_beta)
            idx_alph = eigvals_alph.argsort()
            idx_beta = eigvals_beta.argsort()
            self.eigvals_alph = eigvals_alph[idx_alph]
            self.eigvals_beta = eigvals_beta[idx_beta]
            self.eigvecs_alph = eigvecs_alph[:, idx_alph]
            self.eigvecs_beta = eigvecs_beta[:, idx_beta]
            # Fix the ordering of everything. The first eigenvectors
            # are those with negative excitation energies.
            self.eigvals_alph = self.eigvals_alph[nov_alph:]
            self.eigvals_beta = self.eigvals_beta[nov_beta:]
            self.eigvecs_alph = self.eigvecs_alph[:, nov_alph:]
            self.eigvecs_beta = self.eigvecs_beta[:, nov_beta:]
            np.set_printoptions(linewidth=200)
            print(self.eigvals_alph.real)
            print(self.eigvals_beta.real)
            print(self.eigvecs_alph.real)
            print(self.eigvecs_beta.real)

    @staticmethod
    def norm_xy(z, nocc, nvirt):
        x, y = z.reshape(2, nvirt, nocc)
        norm = 2 * (np.linalg.norm(x)**2 - np.linalg.norm(y)**2)
        norm = 1 / np.sqrt(norm)
        return (x*norm).flatten(), (y*norm).flatten()

    def form_results(self):

        nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = self.occupations
        nov_alph = nocc_alph * nvirt_alph
        nov_beta = nocc_beta * nvirt_beta

        if not self.is_uhf:
            self.eigvecs_normed = self.eigvecs.copy()
            for idx in range(nov_alph):
                print('=' * 78)
                eigvec = self.eigvecs[:, idx]
                x_normed, y_normed = TDHF.norm_xy(self.eigvecs[:, idx], nocc_alph, nvirt_alph)
                eigvec_normed = np.concatenate((x_normed.flatten(), y_normed.flatten()), axis=0)
                self.eigvecs_normed[:, idx] = eigvec_normed
                eigval = self.eigvals[idx].real
                print(' State: {}'.format(idx + 1))
                print(' Excitation energy [a.u.]: {}'.format(eigval))
                print(' Excitation energy [eV]  : {}'.format(eigval * 27.2114))
                # contract the components of every operator with every
                # eigenvector
                for operator in self.operators:
                    print('-' * 78)
                    print(' Operator: {}'.format(operator.label))
                    integrals = operator.mo_integrals_ai_supervector_alph[:, :, 0]
                    # TODO why the 2?
                    res_normed = 2 * np.dot(integrals, eigvec_normed)
                    transition_moment = res_normed
                    oscillator_strength = (2 / 3) * eigval * transition_moment ** 2
                    total_oscillator_strength = (2 / 3) * eigval * np.dot(transition_moment, transition_moment)
                    print(' Transition moment: {}'.format(transition_moment))
                    print(' Oscillator strength: {}'.format(oscillator_strength))
                    print(' Oscillator strength (total): {}'.format(total_oscillator_strength))
                    if not hasattr(operator, 'transition_moments'):
                        operator.transition_moments = []
                    if not hasattr(operator, 'oscillator_strengths'):
                        operator.oscillator_strengths = []
                    if not hasattr(operator, 'total_oscillator_strengths'):
                        operator.total_oscillator_strengths = []
                    operator.transition_moments.append(transition_moment)
                    operator.oscillator_strengths.append(oscillator_strength)
                    operator.total_oscillator_strengths.append(total_oscillator_strength)
            for operator in self.operators:
                operator.transition_moments = np.array(operator.transition_moments)
                operator.oscillator_strengths = np.array(operator.oscillator_strengths)
                operator.total_oscillator_strengths = np.array(operator.total_oscillator_strengths)

        else:
            pass


class TDA(TDHF):

    def __init__(self, mocoeffs, moenergies, occupations, *args, **kwargs):
        super().__init__(mocoeffs, moenergies, occupations, *args, **kwargs)

    def form_explicit_hessian(self, hamiltonian=None, spin=None, frequency=None):

        assert hasattr(self, 'tei_mo')
        assert self.tei_mo is not None
        assert len(self.tei_mo) in (1, 2, 4, 6)
        assert self.tei_mo_type in ('full', 'partial')

        if not hamiltonian:
            hamiltonian = self.hamiltonian
        if not spin:
            spin = self.spin

        assert hamiltonian == 'tda'
        assert spin in ('singlet', 'triplet')

        self.hamiltonian = hamiltonian
        self.spin = spin

        nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = self.occupations
        nov_alph = nocc_alph * nvirt_alph
        nov_beta = nocc_beta * nvirt_beta

        if not self.is_uhf:

            # Set up "function pointers".
            if self.tei_mo_type == 'full':
                assert len(self.tei_mo) == 1
                tei_mo = self.tei_mo[0]
            elif self.tei_mo_type == 'partial':
                assert len(self.tei_mo) == 2
                tei_mo_ovov = self.tei_mo[0]
                tei_mo_oovv = self.tei_mo[1]

            if self.tei_mo_type == 'full':
                if spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_full(self.moenergies[0, ...], tei_mo, nocc_alph)
                elif spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_full(self.moenergies[0, ...], tei_mo, nocc_alph)
            elif self.tei_mo_type == 'partial':
                if spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_partial(self.moenergies[0, ...], tei_mo_ovov, tei_mo_oovv)
                elif spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_partial(self.moenergies[0, ...], tei_mo_oovv)

            self.explicit_hessian = A

        else:

            # Set up "function pointers".
            if self.tei_mo_type == 'full':
                assert len(self.tei_mo) == 4
                tei_mo_aaaa = self.tei_mo[0]
                tei_mo_aabb = self.tei_mo[1]
                tei_mo_bbaa = self.tei_mo[2]
                tei_mo_bbbb = self.tei_mo[3]
            elif self.tei_mo_type == 'partial':
                assert len(self.tei_mo) == 6
                tei_mo_ovov_aaaa = self.tei_mo[0]
                tei_mo_ovov_aabb = self.tei_mo[1]
                tei_mo_ovov_bbaa = self.tei_mo[2]
                tei_mo_ovov_bbbb = self.tei_mo[3]
                tei_mo_oovv_aaaa = self.tei_mo[4]
                tei_mo_oovv_bbbb = self.tei_mo[5]
            else:
                pass

            E_a = self.moenergies[0, ...]
            E_b = self.moenergies[1, ...]

            # TODO clean this up...
            if self.tei_mo_type == 'full':
                if spin == 'singlet':
                    A_ss_a = form_rpa_a_matrix_mo_singlet_ss_full(E_a, tei_mo_aaaa, nocc_alph)
                    A_os_a = form_rpa_a_matrix_mo_singlet_os_full(tei_mo_aabb, nocc_alph, nocc_beta)
                    A_ss_b = form_rpa_a_matrix_mo_singlet_ss_full(E_b, tei_mo_bbbb, nocc_beta)
                    A_os_b = form_rpa_a_matrix_mo_singlet_os_full(tei_mo_bbaa, nocc_beta, nocc_alph)
                elif spin == 'triplet':
                    zeros_ab = np.zeros(shape=(nov_alph, nov_beta))
                    zeros_ba = zeros_ab.T
                    A_ss_a = form_rpa_a_matrix_mo_triplet_full(E_a, tei_mo_aaaa, nocc_alph)
                    A_os_a = zeros_ab
                    A_ss_b = form_rpa_a_matrix_mo_triplet_full(E_b, tei_mo_bbbb, nocc_beta)
                    A_os_b = zeros_ba

            elif self.tei_mo_type == 'partial':
                if spin == 'singlet':
                    A_ss_a = form_rpa_a_matrix_mo_singlet_ss_partial(E_a, tei_mo_ovov_aaaa, tei_mo_oovv_aaaa)
                    A_os_a = form_rpa_a_matrix_mo_singlet_os_partial(tei_mo_ovov_aabb)
                    A_ss_b = form_rpa_a_matrix_mo_singlet_ss_partial(E_b, tei_mo_ovov_bbbb, tei_mo_oovv_bbbb)
                    A_os_b = form_rpa_a_matrix_mo_singlet_os_partial(tei_mo_ovov_bbaa)
                elif spin == 'triplet':
                    zeros_ab = np.zeros(shape=(nov_alph, nov_beta))
                    zeros_ba = zeros_ab.T
                    A_ss_a = form_rpa_a_matrix_mo_triplet_partial(E_a, tei_mo_oovv_aaaa)
                    A_os_a = zeros_ab
                    A_ss_b = form_rpa_a_matrix_mo_triplet_partial(E_b, tei_mo_oovv_bbbb)
                    A_os_b = zeros_ba

            self.explicit_hessian = (
                A_ss_a,
                A_os_a,
                A_os_b,
                A_ss_b,
            )

    def diagonalize_explicit_hessian(self):
        nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = self.occupations
        nov_alph = nocc_alph * nvirt_alph
        nov_beta = nocc_beta * nvirt_beta
        if not self.is_uhf:
            eigvals, eigvecs = sp.linalg.eig(self.explicit_hessian)
            # Sort from lowest to highest eigenvalue (excitation
            # energy).
            idx = eigvals.argsort()
            self.eigvals = eigvals[idx]
            # Each eigenvector is a column vector.
            self.eigvecs = eigvecs[:, idx]
        else:
            assert len(self.explicit_hessian) == 4
            G_aa, G_ab, G_ba, G_bb = self.explicit_hessian
            # TODO don't need this if "triplet"
            G_aa_inv = np.linalg.inv(G_aa)
            G_bb_inv = np.linalg.inv(G_bb)
            lhs_alph = G_aa - np.dot(G_ab, np.dot(G_bb_inv, G_ba))
            lhs_beta = G_bb - np.dot(G_ba, np.dot(G_aa_inv, G_ab))
            eigvals_alph, eigvecs_alph = sp.linalg.eig(lhs_alph)
            eigvals_beta, eigvecs_beta = sp.linalg.eig(lhs_beta)
            idx_alph = eigvals_alph.argsort()
            idx_beta = eigvals_beta.argsort()
            self.eigvals_alph = eigvals_alph[idx_alph]
            self.eigvals_beta = eigvals_beta[idx_beta]
            self.eigvecs_alph = eigvecs_alph[:, idx_alph]
            self.eigvecs_beta = eigvecs_beta[:, idx_beta]
            np.set_printoptions(linewidth=200)
            print(self.eigvals_alph.real)
            print(self.eigvals_beta.real)
            print(self.eigvecs_alph.real)
            print(self.eigvecs_beta.real)

    def form_results(self):

        nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = self.occupations
        nov_alph = nocc_alph * nvirt_alph
        nov_beta = nocc_beta * nvirt_beta

        if not self.is_uhf:
            self.eigvecs_normed = self.eigvecs.copy()
            for idx in range(nov_alph):
                print('=' * 78)
                norm = (1 / np.sqrt(2))
                eigvec = self.eigvecs[:, idx]
                eigvec_normed = self.eigvecs[:, idx] * norm
                self.eigvecs_normed[:, idx] = eigvec_normed
                eigval = self.eigvals[idx].real
                print(' State: {}'.format(idx + 1))
                print(' Excitation energy [a.u.]: {}'.format(eigval))
                print(' Excitation energy [eV]  : {}'.format(eigval * 27.2114))
                # contract the components of every operator with every
                # eigenvector
                for operator in self.operators:
                    print('-' * 78)
                    print(' Operator: {}'.format(operator.label))
                    integrals = operator.mo_integrals_ai_alph[:, :, 0]
                    # TODO why the 2?
                    res_normed = 2 * np.dot(integrals, eigvec_normed)
                    transition_moment = res_normed
                    oscillator_strength = (2 / 3) * eigval * transition_moment ** 2
                    total_oscillator_strength = (2 / 3) * eigval * np.dot(transition_moment, transition_moment)
                    print(' Transition moment: {}'.format(transition_moment))
                    print(' Oscillator strength: {}'.format(oscillator_strength))
                    print(' Oscillator strength (total): {}'.format(total_oscillator_strength))
                    if not hasattr(operator, 'transition_moments'):
                        operator.transition_moments = []
                    if not hasattr(operator, 'oscillator_strengths'):
                        operator.oscillator_strengths = []
                    if not hasattr(operator, 'total_oscillator_strengths'):
                        operator.total_oscillator_strengths = []
                    operator.transition_moments.append(transition_moment)
                    operator.oscillator_strengths.append(oscillator_strength)
                    operator.total_oscillator_strengths.append(total_oscillator_strength)
            for operator in self.operators:
                operator.transition_moments = np.array(operator.transition_moments)
                operator.oscillator_strengths = np.array(operator.oscillator_strengths)
                operator.total_oscillator_strengths = np.array(operator.total_oscillator_strengths)

        else:
            pass
