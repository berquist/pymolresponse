import numpy as np

import ao2mo
from cphf import Operator
from td import TDHF, TDA


class ECD(object):

    def __init__(self, pyscfmol, mocoeffs, moenergies, occupations, do_dipvel=False, do_tda=False, solver=None, *args, **kwargs):
        self.pyscfmol = pyscfmol
        self.mocoeffs = mocoeffs
        self.moenergies = moenergies
        self.occupations = occupations
        self.do_dipvel = do_dipvel
        self.do_tda = do_tda

        if solver:
            self.solver = solver
            # Clear out any operators that may already be present.
            self.solver.operators = []
        elif do_tda:
            self.solver = TDA(self.mocoeffs, self.moenergies, self.occupations)
        else:
            self.solver = TDHF(self.mocoeffs, self.moenergies, self.occupations)

        nden = self.mocoeffs.shape[0]
        if not hasattr(self.solver, 'tei_mo'):
            if nden == 2:
                tei_mo_func = ao2mo.perform_tei_ao2mo_uhf_partial
            else:
                tei_mo_func = ao2mo.perform_tei_ao2mo_rhf_partial
            self.solver.tei_mo = tei_mo_func(self.pyscfmol, self.mocoeffs, self.pyscfmol.verbose)
            self.solver.tei_mo_type = 'partial'

        operator_angmom = Operator(label='angmom', is_imaginary=True, is_spin_dependent=False, triplet=False)
        integrals_angmom_ao = self.pyscfmol.intor('cint1e_cg_irxp_sph', comp=3)
        operator_angmom.ao_integrals = integrals_angmom_ao
        self.solver.add_operator(operator_angmom)
        operator_diplen = Operator(label='dipole', is_imaginary=False, is_spin_dependent=False, triplet=False)
        integrals_diplen_ao = self.pyscfmol.intor('cint1e_r_sph', comp=3)
        operator_diplen.ao_integrals = integrals_diplen_ao
        self.solver.add_operator(operator_diplen)
        if do_dipvel:
            operator_dipvel = Operator(label='dipvel', is_imaginary=True, is_spin_dependent=False, triplet=False)
            integrals_dipvel_ao = self.pyscfmol.intor('cint1e_ipovlp_sph', comp=3)
            operator_dipvel.ao_integrals = integrals_dipvel_ao
            self.solver.add_operator(operator_dipvel)

    def run(self):
        hamiltonian = 'rpa'
        if self.do_tda:
            hamiltonian = 'tda'
        # TODO triplet?
        self.solver.run(solver='explicit', hamiltonian=hamiltonian, spin='singlet')

    def form_results(self):

        operator_angmom = self.solver.operators[0]
        operator_diplen = self.solver.operators[1]
        assert len(operator_angmom.transition_moments) == \
            len(operator_diplen.transition_moments)
        nstates = len(operator_diplen.transition_moments)
        rotational_strengths_diplen = []
        rotational_strengths_dipvel = []
        if self.do_dipvel:
            assert len(self.solver.operators) == 3
            operator_dipvel = self.solver.operators[2]
            assert len(operator_dipvel.transition_moments) == nstates
        from constants import esuecd
        for stateidx in range(nstates):
            print('-' * 78)
            eigval = self.solver.eigvals[stateidx].real
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
        self.rotational_strengths_dipvel = np.array(rotational_strengths_dipvel)
