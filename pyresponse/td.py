"""Drivers for solving the time-dependent Hartree-Fock (TDHF)
equations."""

import numpy as np
import scipy.constants as spc

from .cphf import CPHF
from .iterators import EigSolver, ExactDiagonalizationSolver
from .operators import Operator
from .utils import form_results

HARTREE_TO_EV = spc.physical_constants["Hartree energy in eV"][0]


class TDHF(CPHF):
    """Driver for solving the time-dependent Hartree-Fock (TDHF)
    equations, also called the random phase approximation (RPA)
    equations.
    """

    def __init__(self, solver, *args, **kwargs):
        super().__init__(solver, *args, **kwargs)

    def run(self, solver_type=None, hamiltonian=None, spin=None, **kwargs):

        assert self.solver is not None
        assert isinstance(self.solver, (EigSolver,))

        if not solver_type:
            solver_type = self.solver_type
        if not hamiltonian:
            hamiltonian = self.hamiltonian
        if not spin:
            spin = self.spin

        assert isinstance(solver_type, str)
        assert isinstance(hamiltonian, str)
        assert isinstance(spin, str)

        # Set the current state.
        self.solver_type = solver_type.lower()
        self.hamiltonian = hamiltonian.lower()
        self.spin = spin.lower()

        if 'exact' in solver_type:
            assert isinstance(self.solver, (ExactDiagonalizationSolver,))
            self.solver.form_explicit_hessian(hamiltonian, spin, None)
            self.solver.diagonalize_explicit_hessian()
        else:
            raise NotImplementedError

        # TODO Is there an equivalent to the uncoupled result? Just
        # orbital energy differences?
        self.form_results()

    def form_results(self):
        nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = self.solver.occupations
        nov_alph = nocc_alph * nvirt_alph
        nov_beta = nocc_beta * nvirt_beta
        self.solver.eigvecs_normed = self.solver.eigvecs.copy()
        # This is because we've calculated all possible roots.
        for idx in range(nov_alph):
            print('=' * 78)
            eigvec = self.solver.eigvecs[:, idx]
            x_normed, y_normed = self.solver.norm_xy(self.solver.eigvecs[:, idx], nocc_alph, nvirt_alph)
            eigvec_normed = np.concatenate((x_normed.flatten(), y_normed.flatten()), axis=0)
            self.solver.eigvecs_normed[:, idx] = eigvec_normed
            eigval = self.solver.eigvals[idx].real
            print(f' State: {idx + 1}')
            print(f' Excitation energy [a.u.]: {eigval}')
            print(f' Excitation energy [eV]  : {eigval * HARTREE_TO_EV}')
            # contract the components of every operator with every
            # eigenvector
            for operator in self.solver.operators:
                print('-' * 78)
                print(f' Operator: {operator.label}')
                integrals = operator.mo_integrals_ai_supervector_alph[:, :, 0]
                # TODO why the 2?
                res_normed = 2 * np.dot(integrals, eigvec_normed)
                transition_moment = res_normed
                oscillator_strength = (2 / 3) * eigval * transition_moment ** 2
                total_oscillator_strength = (2 / 3) * eigval * np.dot(transition_moment,
                                                                      transition_moment)
                print(f' Transition moment: {transition_moment}')
                print(f' Oscillator strength: {oscillator_strength}')
                print(f' Oscillator strength (total): {total_oscillator_strength}')
                if not hasattr(operator, 'transition_moments'):
                    operator.transition_moments = []
                if not hasattr(operator, 'oscillator_strengths'):
                    operator.oscillator_strengths = []
                if not hasattr(operator, 'total_oscillator_strengths'):
                    operator.total_oscillator_strengths = []
                operator.transition_moments.append(transition_moment)
                operator.oscillator_strengths.append(oscillator_strength)
                operator.total_oscillator_strengths.append(total_oscillator_strength)
        for operator in self.solver.operators:
            operator.transition_moments = np.array(operator.transition_moments)
            operator.oscillator_strengths = np.array(operator.oscillator_strengths)
            operator.total_oscillator_strengths = np.array(operator.total_oscillator_strengths)


class TDA(TDHF):
    """Driver for solving the time-dependent Hartree-Fock equations with
    the Tamm-Dancoff approximation (TDA), also called the configuration
    interaction with single excitation (CIS) equations."""

    def __init__(self, solver, *args, **kwargs):
        super().__init__(solver, *args, **kwargs)

    def form_results(self):
        nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = self.solver.occupations
        nov_alph = nocc_alph * nvirt_alph
        nov_beta = nocc_beta * nvirt_beta
        self.solver.eigvecs_normed = self.solver.eigvecs.copy()
        # This is because we've calculated all possible roots.
        for idx in range(nov_alph):
            print('=' * 78)
            norm = (1 / np.sqrt(2))
            eigvec = self.solver.eigvecs[:, idx]
            eigvec_normed = self.solver.eigvecs[:, idx] * norm
            self.solver.eigvecs_normed[:, idx] = eigvec_normed
            eigval = self.solver.eigvals[idx].real
            print(f' State: {idx + 1}')
            print(f' Excitation energy [a.u.]: {eigval}')
            print(f' Excitation energy [eV]  : {eigval * HARTREE_TO_EV}')
            # contract the components of every operator with every
            # eigenvector
            for operator in self.solver.operators:
                print('-' * 78)
                print(f' Operator: {operator.label}')
                integrals = operator.mo_integrals_ai_alph[:, :, 0]
                # TODO why the 2?
                res_normed = 2 * np.dot(integrals, eigvec_normed)
                transition_moment = res_normed
                oscillator_strength = (2 / 3) * eigval * transition_moment ** 2
                total_oscillator_strength = (2 / 3) * eigval * np.dot(transition_moment,
                                                                      transition_moment)
                print(f' Transition moment: {transition_moment}')
                print(f' Oscillator strength: {oscillator_strength}')
                print(f' Oscillator strength (total): {total_oscillator_strength}')
                if not hasattr(operator, 'transition_moments'):
                    operator.transition_moments = []
                if not hasattr(operator, 'oscillator_strengths'):
                    operator.oscillator_strengths = []
                if not hasattr(operator, 'total_oscillator_strengths'):
                    operator.total_oscillator_strengths = []
                operator.transition_moments.append(transition_moment)
                operator.oscillator_strengths.append(oscillator_strength)
                operator.total_oscillator_strengths.append(total_oscillator_strength)
        for operator in self.solver.operators:
            operator.transition_moments = np.array(operator.transition_moments)
            operator.oscillator_strengths = np.array(operator.oscillator_strengths)
            operator.total_oscillator_strengths = np.array(operator.total_oscillator_strengths)
