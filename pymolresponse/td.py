"""Drivers for solving the time-dependent Hartree-Fock (TDHF)
equations."""

import numpy as np

from pymolresponse.constants import HARTREE_TO_EV, HARTREE_TO_INVCM
from pymolresponse.core import Hamiltonian, Program, Spin
from pymolresponse.cphf import CPHF
from pymolresponse.solvers import EigSolver, EigSolverTDA, Solver
from pymolresponse.utils import form_indices_orbwin, form_vec_energy_differences


class TDHF(CPHF):
    """Driver for solving the time-dependent Hartree-Fock (TDHF)
    equations, also called the random phase approximation (RPA)
    equations.
    """

    def __init__(self, solver: Solver) -> None:
        assert isinstance(solver, EigSolver)
        super().__init__(solver)

    def run(self, hamiltonian: Hamiltonian, spin: Spin, program: Program, program_obj) -> None:
        assert isinstance(hamiltonian, Hamiltonian)
        assert isinstance(spin, Spin)
        assert isinstance(program, (Program, type(None)))
        # TODO program_obj

        self.solver.run(hamiltonian, spin, program, program_obj)
        # TODO Is there an equivalent to the uncoupled result? Just
        # orbital energy differences?
        self.form_results()

    def form_results(self) -> None:
        nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = self.solver.occupations
        nov_alph = nocc_alph * nvirt_alph
        # nov_beta = nocc_beta * nvirt_beta
        self.solver.eigvecs_normed = self.solver.eigvecs.copy()
        # This is because we've calculated all possible roots.
        for idx in range(nov_alph):
            eigvec = self.solver.eigvecs[:, idx]
            x_normed, y_normed = self.solver.norm_xy(eigvec, nocc_alph, nvirt_alph)
            eigvec_normed = np.concatenate((x_normed.flatten(), y_normed.flatten()), axis=0)
            self.solver.eigvecs_normed[:, idx] = eigvec_normed
            eigval = self.solver.eigvals[idx].real
            # contract the components of every operator with every
            # eigenvector
            for operator in self.solver.operators:
                integrals = operator.mo_integrals_ai_supervector_alph[:, :, 0]
                # TODO why the 2?
                res_normed = 2 * np.dot(integrals, eigvec_normed)
                transition_moment = res_normed
                oscillator_strength = (2 / 3) * eigval * transition_moment**2
                total_oscillator_strength = (
                    (2 / 3) * eigval * np.dot(transition_moment, transition_moment)
                )
                if not hasattr(operator, "transition_moments"):
                    operator.transition_moments = []
                if not hasattr(operator, "oscillator_strengths"):
                    operator.oscillator_strengths = []
                if not hasattr(operator, "total_oscillator_strengths"):
                    operator.total_oscillator_strengths = []
                operator.transition_moments.append(transition_moment)
                operator.oscillator_strengths.append(oscillator_strength)
                operator.total_oscillator_strengths.append(total_oscillator_strength)
        for operator in self.solver.operators:
            operator.transition_moments = np.array(operator.transition_moments)
            operator.oscillator_strengths = np.array(operator.oscillator_strengths)
            operator.total_oscillator_strengths = np.array(operator.total_oscillator_strengths)

    def print_results(self) -> None:
        energies = self.solver.eigvals.real
        for idx in range(len(energies)):
            print("=" * 78)
            print(f" State: {idx + 1}")
            print(f" Excitation energy [a.u.]: {energies[idx]}")
            print(f" Excitation energy [eV]  : {energies[idx] * HARTREE_TO_EV}")
            for operator in self.solver.operators:
                transition_moment = operator.transition_moments[idx]
                oscillator_strength = operator.oscillator_strengths[idx]
                total_oscillator_strength = operator.total_oscillator_strengths[idx]
                print("-" * 78)
                print(f" Operator: {operator.label}")
                print(f" Transition moment: {transition_moment}")
                print(f" Oscillator strength: {oscillator_strength}")
                print(f" Oscillator strength (total): {total_oscillator_strength}")

    def print_results_nwchem(self) -> str:
        # TODO UHF
        nocc_tot, nvirt_tot, _, _ = self.solver.occupations
        moene = np.diag(self.solver.moenergies[0])
        moene_occ = moene[:nocc_tot]
        moene_virt = moene[nocc_tot:]
        ediff = form_vec_energy_differences(moene_occ, moene_virt) * HARTREE_TO_EV
        idxsort = np.argsort(ediff)
        ediff_sorted = ediff[idxsort]
        indices_unrestricted_orbwin = form_indices_orbwin(nocc_tot, nvirt_tot)
        indices_sorted = [indices_unrestricted_orbwin[i] for i in idxsort]
        ndiff = 10
        lines = []
        lines.append(f"   {ndiff:>2d} smallest eigenvalue differences (eV) ")
        lines.append("--------------------------------------------------------")
        lines.append("  No. Spin  Occ  Vir  Irrep   E(Occ)    E(Vir)   E(Diff)")
        lines.append("--------------------------------------------------------")
        for idx in range(ndiff):
            iocc, ivirt = indices_sorted[idx]
            lines.append(
                f"{idx + 1:>5d}{1:>5d}{iocc + 1:>5d}{ivirt + 1:>5d} "
                f"{'X':<5}{moene[iocc]:>10.3f}{moene[ivirt]:>10.3f}{ediff_sorted[idx]:>10.3f}"
            )
        lines.append("--------------------------------------------------------")
        return "\n".join(lines)

    _HAMILTONIAN_MAP_ORCA = {Hamiltonian.TDA: "CIS-", Hamiltonian.RPA: "RPA "}

    _SPIN_MAP_ORCA = {Spin.singlet: "SINGLETS", Spin.triplet: "TRIPLETS"}

    def print_results_orca(self, cutoff=0.01):
        energies = self.solver.eigvals.real
        energies_ev = energies * HARTREE_TO_EV
        energies_invcm = energies * HARTREE_TO_INVCM
        nocc_tot, nvirt_tot, _, _ = self.solver.occupations
        indices = form_indices_orbwin(nocc_tot, nvirt_tot)
        eigvecs = self.solver.eigvecs
        # eigvecs_normed = self.solver.eigvecs_normed
        square_eigvecs = np.power(eigvecs, 2)
        # square_eigvecs_normed = np.power(eigvecs_normed, 2)
        # mask = square_eigvecs > cutoff
        # print(square_eigvecs.shape)
        # print(mask.shape)
        # mask_normed = square_eigvecs_normed > cutoff
        # print(eigvecs[mask].shape)
        # print(eigvecs_normed[mask_normed])
        # print(square_eigvecs[mask].shape)
        # print(square_eigvecs_normed[mask_normed])
        lines = [
            "-----------------------------",
            f"{self._HAMILTONIAN_MAP_ORCA[self.hamiltonian]}EXCITED STATES ({self._SPIN_MAP_ORCA[self.spin]})",
            "-----------------------------",
            "",
            f"the weight of the individual excitations are printed if larger than {cutoff:>4.2f}",
            "",
        ]
        nstates = len(energies)
        for state in range(nstates):
            lines.append(
                f"STATE{state + 1:>3d}:  E={energies[state]:>11.6f} "
                f"au{energies_ev[state]:>11.3f} eV{energies_invcm[state]:>11.1f} cm**-1"
            )
            eigvec_state = eigvecs[:, state]
            square_eigvec_state = square_eigvecs[:, state]
            mask = square_eigvec_state > cutoff
            coeffs_cutoff = eigvec_state[mask]
            weight_cutoff = square_eigvec_state[mask]
            mask_indices = np.array([p for (p, b) in enumerate(mask) if b])
            for i in range(len(coeffs_cutoff)):
                iocc, ivirt = indices[mask_indices[i]]
                lines.append(
                    f"{iocc:>6d}a ->{ivirt:>4d}a  :{weight_cutoff[i]:>13.6f} (c={coeffs_cutoff[i]:>12.8f})"
                )
            lines.append("")
        return "\n".join(lines)

    _SPIN_MAP_QCHEM = {Spin.singlet: "Singlet", Spin.triplet: "Triplet"}


class TDA(TDHF):
    """Driver for solving the time-dependent Hartree-Fock equations with the
    Tamm-Dancoff approximation (TDA), also called the configuration
    interaction with single excitation (CIS) equations when no XC contribution
    is present.
    """

    def __init__(self, solver: EigSolverTDA) -> None:
        assert isinstance(solver, EigSolverTDA)
        super().__init__(solver)

    def form_results(self) -> None:
        nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = self.solver.occupations
        nov_alph = nocc_alph * nvirt_alph
        # nov_beta = nocc_beta * nvirt_beta
        self.solver.eigvecs_normed = self.solver.eigvecs.copy()
        # This is because we've calculated all possible roots.

        for idx in range(nov_alph):
            norm = 1 / np.sqrt(2)
            eigvec = self.solver.eigvecs[:, idx]
            eigvec_normed = eigvec * norm
            self.solver.eigvecs_normed[:, idx] = eigvec_normed
            eigval = self.solver.eigvals[idx].real
            # contract the components of every operator with every
            # eigenvector
            for operator in self.solver.operators:
                integrals = operator.mo_integrals_ai_alph[:, :, 0]
                # TODO why the 2?
                res_normed = 2 * np.dot(integrals, eigvec_normed)
                transition_moment = res_normed
                oscillator_strength = (2 / 3) * eigval * transition_moment**2
                total_oscillator_strength = (
                    (2 / 3) * eigval * np.dot(transition_moment, transition_moment)
                )
                if not hasattr(operator, "transition_moments"):
                    operator.transition_moments = []
                if not hasattr(operator, "oscillator_strengths"):
                    operator.oscillator_strengths = []
                if not hasattr(operator, "total_oscillator_strengths"):
                    operator.total_oscillator_strengths = []
                operator.transition_moments.append(transition_moment)
                operator.oscillator_strengths.append(oscillator_strength)
                operator.total_oscillator_strengths.append(total_oscillator_strength)
        for operator in self.solver.operators:
            operator.transition_moments = np.array(operator.transition_moments)
            operator.oscillator_strengths = np.array(operator.oscillator_strengths)
            operator.total_oscillator_strengths = np.array(operator.total_oscillator_strengths)
