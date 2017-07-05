from __future__ import print_function
from __future__ import division

import numpy as np

from .operators import Operator
from .molecular_property import TransitionProperty


class ECD(TransitionProperty):

    def __init__(self, pyscfmol, mocoeffs, moenergies, occupations, do_dipvel=False, *args, **kwargs):
        super().__init__(pyscfmol, mocoeffs, moenergies, occupations, do_dipvel, *args, **kwargs)
        self.do_dipvel = do_dipvel

    def form_operators(self):

        operator_angmom = Operator(label='angmom', is_imaginary=True, is_spin_dependent=False, triplet=False)
        integrals_angmom_ao = self.pyscfmol.intor('cint1e_cg_irxp_sph', comp=3)
        operator_angmom.ao_integrals = integrals_angmom_ao
        self.driver.add_operator(operator_angmom)
        operator_diplen = Operator(label='dipole', is_imaginary=False, is_spin_dependent=False, triplet=False)
        integrals_diplen_ao = self.pyscfmol.intor('cint1e_r_sph', comp=3)
        operator_diplen.ao_integrals = integrals_diplen_ao
        self.driver.add_operator(operator_diplen)
        if self.do_dipvel:
            operator_dipvel = Operator(label='dipvel', is_imaginary=True, is_spin_dependent=False, triplet=False)
            integrals_dipvel_ao = self.pyscfmol.intor('cint1e_ipovlp_sph', comp=3)
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
        from constants import esuecd
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
