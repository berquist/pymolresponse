import numpy as np
import scipy as sp

from cphf import Operator, CPHF
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
            # TODO UHF
            pass

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
            # TODO UHF
            pass

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


class TDA(TDHF):

    def __init__(self, mocoeffs, moenergies, occupations, *args, **kwargs):
        super(TDA, self).__init__(mocoeffs, moenergies, occupations, *args, **kwargs)

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
            # TODO UHF
            pass

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
            # TODO UHF
            pass


    def form_results(self):
        nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = self.occupations
        nov_alph = nocc_alph * nvirt_alph
        nov_beta = nocc_beta * nvirt_beta
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
