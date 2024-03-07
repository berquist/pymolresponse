"""Wrapper for performing an electronic circular dichroism (ECD)
calculation."""

import numpy as np

from pymolresponse.constants import HARTREE_TO_EV, HARTREE_TO_INVCM, alpha, esuecd
from pymolresponse.core import Program
from pymolresponse.molecular_property import TransitionProperty
from pymolresponse.operators import Operator
from pymolresponse.td import TDHF
from pymolresponse.utils import form_indices_zero


class ECD(TransitionProperty):
    """Wrapper for performing an electronic circular dichroism (ECD)
    calculation.
    """

    def __init__(
        self,
        program: Program,
        program_obj,
        driver: TDHF,
        mocoeffs: np.ndarray,
        moenergies: np.ndarray,
        occupations: np.ndarray,
        *,
        do_tda: bool = False,
        do_dipvel: bool = False,
    ) -> None:
        super().__init__(
            program, program_obj, driver, mocoeffs, moenergies, occupations, do_tda=do_tda
        )
        self.do_dipvel = do_dipvel

    def form_operators(self) -> None:
        if self.program == Program.PySCF:
            from pymolresponse.interfaces.pyscf import integrals

            integral_generator = integrals.IntegralsPyscf(self.program_obj)
        elif self.program == Program.Psi4:
            from pymolresponse.interfaces.psi4 import integrals

            integral_generator = integrals.IntegralsPsi4(self.program_obj)
        else:
            raise RuntimeError

        operator_angmom = Operator(
            label="angmom", is_imaginary=True, is_spin_dependent=False, triplet=False
        )
        operator_angmom.ao_integrals = integral_generator.integrals(integrals.ANGMOM_COMMON_GAUGE)
        self.driver.add_operator(operator_angmom)

        operator_diplen = Operator(
            label="dipole", is_imaginary=False, is_spin_dependent=False, triplet=False
        )
        operator_diplen.ao_integrals = integral_generator.integrals(integrals.DIPOLE)
        self.driver.add_operator(operator_diplen)

        if self.do_dipvel:
            operator_dipvel = Operator(
                label="dipvel", is_imaginary=True, is_spin_dependent=False, triplet=False
            )
            operator_dipvel.ao_integrals = integral_generator.integrals(integrals.DIPVEL)
            self.driver.add_operator(operator_dipvel)

    def form_results(self) -> None:
        operator_angmom = self.driver.solver.operators[0]
        operator_diplen = self.driver.solver.operators[1]
        assert len(operator_angmom.transition_moments) == len(operator_diplen.transition_moments)
        nstates = len(operator_diplen.transition_moments)
        rotational_strengths_diplen = []
        rotational_strengths_dipvel = []
        if self.do_dipvel:
            assert len(self.driver.solver.operators) == 3
            operator_dipvel = self.driver.solver.operators[2]
            assert len(operator_dipvel.transition_moments) == nstates
        for stateidx in range(nstates):
            print("-" * 78)
            eigval = self.driver.solver.eigvals[stateidx].real
            rotstr_diplen = (
                esuecd
                * (-1 / 2)
                * np.dot(
                    operator_diplen.transition_moments[stateidx],
                    operator_angmom.transition_moments[stateidx],
                )
            )
            print("length  ", rotstr_diplen)
            rotational_strengths_diplen.append(rotstr_diplen)
            if self.do_dipvel:
                rotstr_dipvel = (
                    esuecd
                    * (-1 / 2)
                    * np.dot(
                        operator_dipvel.transition_moments[stateidx] / eigval,
                        operator_angmom.transition_moments[stateidx],
                    )
                )
                print("velocity", rotstr_dipvel)
                rotational_strengths_dipvel.append(rotstr_dipvel)
        self.rotational_strengths_diplen = np.array(rotational_strengths_diplen)
        if self.do_dipvel:
            self.rotational_strengths_dipvel = np.array(rotational_strengths_dipvel)

    def print_results_nwchem(self) -> str:
        excitation_block = self.driver.print_results_nwchem()
        lines = [excitation_block]
        energies = self.driver.solver.eigvals.real
        energies_ev = energies * HARTREE_TO_EV
        op_diplen = self.driver.solver.operators[1]
        tmom_diplen = op_diplen.transition_moments
        etoscslen = op_diplen.total_oscillator_strengths
        op_angmom = self.driver.solver.operators[0]
        tmom_angmom = op_angmom.transition_moments
        rotstrlen = self.rotational_strengths_diplen
        if self.do_dipvel:
            op_dipvel = self.driver.solver.operators[2]
            rotstrvel = self.rotational_strengths_dipvel
            tmom_dipvel = op_dipvel.transition_moments
            etoscsvel = op_dipvel.total_oscillator_strengths
        nstates = len(energies)
        for state in range(nstates):
            lines.append(
                "  ----------------------------------------------------------------------------"
            )
            lines.append(
                f"  Root {state + 1:>3d} singlet a{energies[state]:>25.9f} a.u.{energies_ev[state]:>22.4f} eV"
            )
            lines.append(
                "  ----------------------------------------------------------------------------"
            )
            lines.append(
                f"     Transition Moments    X{tmom_diplen[state, 0]:>9.5f}   Y{tmom_diplen[state, 1]:>9.5f}   Z{tmom_diplen[state, 2]:>9.5f}"
            )
            ## TODO these require second moment (length) integrals
            ## lines.append(f'     Transition Moments   XX -0.28379  XY  0.08824  XZ -0.17416')
            ## lines.append(f'     Transition Moments   YY -0.40247  YZ -0.45981  ZZ  0.59211')
            lines.append(f"     Dipole Oscillator Strength {etoscslen[state]:>31.5f}")
            lines.append("")
            lines.append("     Electric Transition Dipole:")
            lines.append(
                f"            X{tmom_diplen[state, 0]:>13.7f}   Y{tmom_diplen[state, 1]:>13.7f}   Z{tmom_diplen[state, 2]:>13.7f}"
            )
            lines.append("     Magnetic Transition Dipole (Length):")
            lines.append(
                f"            X{tmom_angmom[state, 0]:>13.7f}   Y{tmom_angmom[state, 1]:>13.7f}   Z{tmom_angmom[state, 2]:>13.7f}"
            )
            lines.append("     Magnetic Transition Dipole * 1/c :")
            lines.append(
                f"            X{tmom_angmom[state, 0] * alpha:>13.7f}   Y{tmom_angmom[state, 1] * alpha:>13.7f}   Z{tmom_angmom[state, 2] * alpha:>13.7f}"
            )
            lines.append(f"     Rotatory Strength (1E-40 esu**2cm**2):{rotstrlen[state]:>21.7f}")
            lines.append("")
            if self.do_dipvel:
                lines.append("     Electric Transition Dipole (velocity representation):")
                lines.append(
                    f"            X{tmom_dipvel[state, 0]:>13.7f}   Y{tmom_dipvel[state, 1]:>13.7f}   Z{tmom_dipvel[state, 2]:>13.7f}"
                )
                # lines.append(f'     Oscillator Strength (velocity repr.) :            0.0069989')
                lines.append(
                    f"     Oscillator Strength (velocity repr.) :{etoscsvel[state]:>21.7f}"
                )
                # lines.append(f'     Oscillator Strength (mixed repr.   ) :            0.0074981')
                # lines.append(f'     Oscillator Strength (mixed repr.   ) :{>21.7f}')
                lines.append(
                    f"     Rotatory Strength   (velocity repr.) :{rotstrvel[state]:>21.7f}"
                )
                lines.append("")
            # lines.append(str(self.driver.solver.eigvecs[:, state]))
            # lines.append(str(self.driver.solver.eigvecs_normed[:, state]))

        return "\n".join(lines)

    def print_results_orca(self) -> str:
        excitation_block = self.driver.print_results_orca()
        lines = [excitation_block]
        energies = self.driver.solver.eigvals.real
        energies_to_invcm = energies * HARTREE_TO_INVCM
        energies_to_nm = 10000000 / energies_to_invcm
        op_diplen = self.driver.solver.operators[1]
        etoscslen = op_diplen.total_oscillator_strengths
        tmom_diplen = op_diplen.transition_moments
        t2_diplen = np.asarray(
            [np.dot(tmom_diplen[x], tmom_diplen[x]) for x in range(len(tmom_diplen))]
        )
        rotstrlen = self.rotational_strengths_diplen
        op_angmom = self.driver.solver.operators[0]
        tmom_angmom = op_angmom.transition_moments
        if self.do_dipvel:
            op_dipvel = self.driver.solver.operators[2]
            rotstrvel = self.rotational_strengths_dipvel  # noqa: F841
            etoscsvel = op_dipvel.total_oscillator_strengths
            tmom_dipvel = op_dipvel.transition_moments
            t2_dipvel = np.asarray(
                [np.dot(tmom_dipvel[x], tmom_dipvel[x]) for x in range(len(tmom_dipvel))]
            )
        nstates = len(energies)
        lines.append(
            "-----------------------------------------------------------------------------"
        )
        lines.append("         ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS")
        lines.append(
            "-----------------------------------------------------------------------------"
        )
        lines.append("State   Energy  Wavelength   fosc         T2         TX        TY        TZ")
        lines.append("        (cm-1)    (nm)                  (au**2)     (au)      (au)      (au)")
        lines.append(
            "-----------------------------------------------------------------------------"
        )
        for state in range(nstates):
            lines.append(
                f"{state + 1:>4d}{energies_to_invcm[state]:>10.1f}{energies_to_nm[state]:>9.1f}{etoscslen[state]:>14.9f}{t2_diplen[state]:>10.5f}{tmom_diplen[state, 0]:>10.5f}{tmom_diplen[state, 1]:>10.5f}{tmom_diplen[state, 2]:>10.5f}"
            )
        lines.append("")
        lines.append(
            "-----------------------------------------------------------------------------"
        )
        lines.append("         ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS")
        lines.append(
            "-----------------------------------------------------------------------------"
        )
        lines.append("State   Energy  Wavelength   fosc         P2         PX        PY        PZ")
        lines.append("        (cm-1)    (nm)                  (au**2)     (au)      (au)      (au)")
        lines.append(
            "-----------------------------------------------------------------------------"
        )
        for state in range(nstates):
            lines.append(
                f"{state + 1:>4d}{energies_to_invcm[state]:>10.1f}{energies_to_nm[state]:>9.1f}{etoscsvel[state]:>14.9f}{t2_dipvel[state]:>10.5f}{tmom_dipvel[state, 0]:>10.5f}{tmom_dipvel[state, 1]:>10.5f}{tmom_dipvel[state, 2]:>10.5f}"
            )
        lines.append("")
        lines.append("-------------------------------------------------------------------")
        lines.append("                             CD SPECTRUM")
        lines.append("-------------------------------------------------------------------")
        lines.append("State   Energy Wavelength       R         MX        MY        MZ")
        lines.append("        (cm-1)   (nm)       (1e40*cgs)   (au)      (au)      (au)")
        lines.append("-------------------------------------------------------------------")
        for state in range(nstates):
            lines.append(
                f"{state + 1:>4d}{energies_to_invcm[state]:>10.1f}{energies_to_nm[state]:>9.1f}{rotstrlen[state]:>13.5f}{tmom_angmom[state, 0]:>10.5f}{tmom_angmom[state, 1]:>10.5f}{tmom_angmom[state, 2]:>10.5f}"
            )
        lines.append("")
        return "\n".join(lines)

    _HAMILTONIAN_PREFIX_QCHEM = {"tda": "", "rpa": "X: "}

    # TODO cutoff taken from ORCA, check the source code to see the
    # real criterion
    def print_results_qchem(self, cutoff: float = 0.01) -> str:
        energies = self.driver.solver.eigvals.real
        energies_ev = energies * HARTREE_TO_EV
        op_diplen = self.driver.solver.operators[1]
        tmom_diplen = op_diplen.transition_moments
        etoscslen = op_diplen.total_oscillator_strengths
        nocc_tot, nvirt_tot, _, _ = self.driver.solver.occupations
        indices = form_indices_zero(nocc_tot, nvirt_tot)
        eigvecs = self.driver.solver.eigvecs
        square_eigvecs = np.power(eigvecs, 2)
        lines = []
        lines.append(" ---------------------------------------------------")
        lines.append(
            f"               {self.driver._HAMILTONIAN_MAP_ORCA[self.driver.hamiltonian]} Excitation Energies              "
        )
        lines.append(" ---------------------------------------------------")
        lines.append("")
        nstates = len(energies)
        for state in range(nstates):
            lines.append(
                f" Excited state{state + 1:>4d}: excitation energy (eV) ={energies_ev[state]:>10.4f}"
            )
            lines.append(f" Total energy for state{state + 1:>3d}:{0:>31.8f} au")
            lines.append(f"    Multiplicity: {self.driver._SPIN_MAP_QCHEM[self.driver.spin]}")
            lines.append(
                f"    Trans. Mom.:{tmom_diplen[state, 0]:>8.4f} X{tmom_diplen[state, 1]:>9.4f} Y{tmom_diplen[state, 2]:>9.4f} Z"
            )
            lines.append(f"    Strength   :{etoscslen[state]:>17.10f}")
            eigvec_state = eigvecs[:, state]
            square_eigvec_state = square_eigvecs[:, state]
            mask = square_eigvec_state > cutoff
            coeffs_cutoff = eigvec_state[mask]
            mask_indices = np.array([p for (p, b) in enumerate(mask) if b])
            for i in range(len(coeffs_cutoff)):
                iocc, ivirt = indices[mask_indices[i]]
                lines.append(
                    f"    {self._HAMILTONIAN_PREFIX_QCHEM[self.driver.hamiltonian]}D({iocc + 1:>3d}) --> V({ivirt + 1:>3d}) amplitude ={coeffs_cutoff[i]:>8.4f}"
                )
            lines.append("")
        return "\n".join(lines)
