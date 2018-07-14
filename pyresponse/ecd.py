"""Wrapper for performing an electronic circular dichroism (ECD)
calculation."""

import numpy as np

from .constants import esuecd, HARTREE_TO_EV, HARTREE_TO_INVCM
from .operators import Operator
from .molecular_property import TransitionProperty


class ECD(TransitionProperty):
    """Wrapper for performing an electronic circular dichroism (ECD)
    calculation.
    """

    def __init__(self, pyscfmol, mocoeffs, moenergies, occupations,
                 do_dipvel=False, *args, **kwargs):
        super().__init__(pyscfmol, mocoeffs, moenergies, occupations,
                         do_dipvel, *args, **kwargs)
        self.do_dipvel = do_dipvel

    def form_operators(self):

        operator_angmom = Operator(label='angmom',
                                   is_imaginary=True,
                                   is_spin_dependent=False,
                                   triplet=False)
        integrals_angmom_ao = self.pyscfmol.intor('cint1e_cg_irxp_sph',
                                                  comp=3)
        operator_angmom.ao_integrals = integrals_angmom_ao
        self.driver.add_operator(operator_angmom)

        operator_diplen = Operator(label='dipole',
                                   is_imaginary=False,
                                   is_spin_dependent=False,
                                   triplet=False)
        integrals_diplen_ao = self.pyscfmol.intor('cint1e_r_sph',
                                                  comp=3)
        operator_diplen.ao_integrals = integrals_diplen_ao
        self.driver.add_operator(operator_diplen)

        if self.do_dipvel:

            operator_dipvel = Operator(label='dipvel',
                                       is_imaginary=True,
                                       is_spin_dependent=False,
                                       triplet=False)
            integrals_dipvel_ao = self.pyscfmol.intor('cint1e_ipovlp_sph',
                                                      comp=3)
            operator_dipvel.ao_integrals = integrals_dipvel_ao
            self.driver.add_operator(operator_dipvel)

    def form_results(self):

        operator_angmom = self.driver.solver.operators[0]
        operator_diplen = self.driver.solver.operators[1]
        assert len(operator_angmom.transition_moments) == \
            len(operator_diplen.transition_moments)
        nstates = len(operator_diplen.transition_moments)
        rotational_strengths_diplen = []
        rotational_strengths_dipvel = []
        if self.do_dipvel:
            assert len(self.driver.solver.operators) == 3
            operator_dipvel = self.driver.solver.operators[2]
            assert len(operator_dipvel.transition_moments) == nstates
        for stateidx in range(nstates):
            print('-' * 78)
            eigval = self.driver.solver.eigvals[stateidx].real
            rotstr_diplen = esuecd * (-1/2) * np.dot(operator_diplen.transition_moments[stateidx],
                                                     operator_angmom.transition_moments[stateidx])
            print('length  ', rotstr_diplen)
            rotational_strengths_diplen.append(rotstr_diplen)
            if self.do_dipvel:
                rotstr_dipvel = esuecd * (-1/2) * np.dot(operator_dipvel.transition_moments[stateidx] / eigval,
                                                         operator_angmom.transition_moments[stateidx])
                print('velocity', rotstr_dipvel)
                rotational_strengths_dipvel.append(rotstr_dipvel)
        self.rotational_strengths_diplen = np.array(rotational_strengths_diplen)
        if self.do_dipvel:
            self.rotational_strengths_dipvel = np.array(rotational_strengths_dipvel)

    def make_results_nwchem(self):
        lines = []
        energies = self.driver.solver.eigvals.real
        energies_ev = energies * HARTREE_TO_EV
        op_diplen = self.driver.solver.operators[1]
        op_angmom = self.driver.solver.operators[0]
        if self.do_dipvel:
            op_dipvel = self.driver.solver.operators[2]
        rotstrlen = self.rotational_strengths_diplen
        if self.do_dipvel:
            rotstrvel = self.rotational_strengths_dipvel
        nstates = len(op_diplen.transition_moments)
        for state in range(nstates):
            lines.append('  ----------------------------------------------------------------------------')
            lines.append(f'  Root {state + 1:>3d} singlet a{energies[state]:>25.9f} a.u.{energies_ev[state]:>22.4f} eV')
            lines.append('  ----------------------------------------------------------------------------')
            lines.append(f'     Transition Moments    X{op_diplen.transition_moments[state, 0]:>9.5f}   Y{op_diplen.transition_moments[state, 1]:>9.5f}   Z{op_diplen.transition_moments[state, 2]:>9.5f}')
            ## TODO these require second moment (length) integrals
            ## lines.append(f'     Transition Moments   XX -0.28379  XY  0.08824  XZ -0.17416')
            ## lines.append(f'     Transition Moments   YY -0.40247  YZ -0.45981  ZZ  0.59211')
            lines.append(f'     Dipole Oscillator Strength {op_diplen.total_oscillator_strengths[state]:>31.5f}')
            lines.append('')
            lines.append('     Electric Transition Dipole:')
            lines.append(f'            X{op_diplen.transition_moments[state, 0]:>13.7f}   Y{op_diplen.transition_moments[state, 1]:>13.7f}   Z{op_diplen.transition_moments[state, 2]:>13.7f}')
            lines.append('     Magnetic Transition Dipole (Length):')
            lines.append(f'            X{op_angmom.transition_moments[state, 0]:>13.7f}   Y{op_angmom.transition_moments[state, 1]:>13.7f}   Z{op_angmom.transition_moments[state, 2]:>13.7f}')
            # TODO conversion?
            # lines.append('     Magnetic Transition Dipole * 1/c :')
            # lines.append(f'            X   -0.0032796   Y  -0.0006302   Z  -0.0002774')
            lines.append(f'     Rotatory Strength (1E-40 esu**2cm**2):{rotstrlen[state]:>21.7f}')
            lines.append('')
            if self.do_dipvel:
                lines.append('     Electric Transition Dipole (velocity representation):')
                lines.append(f'            X{op_dipvel.transition_moments[state, 0]:>13.7f}   Y{op_dipvel.transition_moments[state, 1]:>13.7f}   Z{op_dipvel.transition_moments[state, 2]:>13.7f}')
                # lines.append(f'     Oscillator Strength (velocity repr.) :            0.0069989')
                lines.append(f'     Oscillator Strength (velocity repr.) :{op_dipvel.total_oscillator_strengths[state]:>21.7f}')
                # lines.append(f'     Oscillator Strength (mixed repr.   ) :            0.0074981')
                # lines.append(f'     Oscillator Strength (mixed repr.   ) :{>21.7f}')
                lines.append(f'     Rotatory Strength   (velocity repr.) :{rotstrvel[state]:>21.7f}')
                lines.append('')
            # lines.append(str(self.driver.solver.eigvecs[:, state]))
            # lines.append(str(self.driver.solver.eigvecs_normed[:, state]))

        return '\n'.join(lines)

    def make_results_orca(self):
        lines = []
        energies = self.driver.solver.eigvals.real
        energies_to_invcm = energies * HARTREE_TO_INVCM
        energies_to_nm = 10000000 / energies_to_invcm
        op_diplen = self.driver.solver.operators[1]
        tmom_diplen = op_diplen.transition_moments
        t2_diplen = np.asarray([np.dot(tmom_diplen[x], tmom_diplen[x])
                                for x in range(len(tmom_diplen))])
        rotstrlen = self.rotational_strengths_diplen
        op_angmom = self.driver.solver.operators[0]
        tmom_angmom = op_angmom.transition_moments
        if self.do_dipvel:
            op_dipvel = self.driver.solver.operators[2]
            rotstrvel = self.rotational_strengths_dipvel
            tmom_dipvel = op_dipvel.transition_moments
            t2_dipvel = np.asarray([np.dot(tmom_dipvel[x], tmom_dipvel[x])
                                    for x in range(len(tmom_dipvel))])
        nstates = len(op_diplen.transition_moments)
        lines.append('-----------------------------------------------------------------------------')
        lines.append('         ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS')
        lines.append('-----------------------------------------------------------------------------')
        lines.append('State   Energy  Wavelength   fosc         T2         TX        TY        TZ  ')
        lines.append('        (cm-1)    (nm)                  (au**2)     (au)      (au)      (au) ')
        lines.append('-----------------------------------------------------------------------------')
        for state in range(nstates):
            lines.append(f'{state + 1:>4d}{energies_to_invcm[state]:>10.1f}{energies_to_nm[state]:>9.1f}{op_diplen.total_oscillator_strengths[state]:>14.9f}{t2_diplen[state]:>10.5f}{op_diplen.transition_moments[state, 0]:>10.5f}{op_diplen.transition_moments[state, 1]:>10.5f}{op_diplen.transition_moments[state, 2]:>10.5f}')
        lines.append('')
        lines.append('-----------------------------------------------------------------------------')
        lines.append('         ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS')
        lines.append('-----------------------------------------------------------------------------')
        lines.append('State   Energy  Wavelength   fosc         P2         PX        PY        PZ  ')
        lines.append('        (cm-1)    (nm)                  (au**2)     (au)      (au)      (au) ')
        lines.append('-----------------------------------------------------------------------------')
        for state in range(nstates):
            lines.append(f'{state + 1:>4d}{energies_to_invcm[state]:>10.1f}{energies_to_nm[state]:>9.1f}{op_dipvel.total_oscillator_strengths[state]:>14.9f}{t2_dipvel[state]:>10.5f}{op_dipvel.transition_moments[state, 0]:>10.5f}{op_dipvel.transition_moments[state, 1]:>10.5f}{op_dipvel.transition_moments[state, 2]:>10.5f}')
        lines.append('')
        lines.append('-------------------------------------------------------------------')
        lines.append('                             CD SPECTRUM')
        lines.append('-------------------------------------------------------------------')
        lines.append('State   Energy Wavelength       R         MX        MY        MZ   ')
        lines.append('        (cm-1)   (nm)       (1e40*cgs)   (au)      (au)      (au)  ')
        lines.append('-------------------------------------------------------------------')
        for state in range(nstates):
            lines.append(f'{state + 1:>4d}{energies_to_invcm[state]:>10.1f}{energies_to_nm[state]:>9.1f}{rotstrlen[state]:>13.5f}{op_angmom.transition_moments[state, 0]:>10.5f}{op_angmom.transition_moments[state, 1]:>10.5f}{op_angmom.transition_moments[state, 2]:>10.5f}')
        lines.append('')
        return '\n'.join(lines)
