from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.constants as spc

import ao2mo
from cphf import CPHF, Operator
from utils import tensor_printer


class ORD(object):

    def __init__(self, pyscfmol, mocoeffs, moenergies, occupations, hamiltonian='rpa', spin='singlet', frequencies=[0.0], do_dipvel=False, *args, **kwargs):
        self.pyscfmol = pyscfmol
        self.mocoeffs = mocoeffs
        self.moenergies = moenergies
        self.occupations = occupations
        self.hamiltonian = hamiltonian
        self.spin = spin
        # Don't allow a single number.
        assert isinstance(frequencies, (list, tuple, np.ndarray))
        self.frequencies = frequencies
        self.do_dipvel = do_dipvel

        # nden = mocoeffs.shape[0]

        self.solver = CPHF(mocoeffs, moenergies, occupations)
        self.solver.set_frequencies(frequencies)

    def form_operators(self):

        nden = self.mocoeffs.shape[0]
        if not self.solver.tei_mo:
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
        if self.do_dipvel:
            operator_dipvel = Operator(label='dipvel', is_imaginary=True, is_spin_dependent=False, triplet=False)
            integrals_dipvel_ao = self.pyscfmol.intor('cint1e_ipovlp_sph', comp=3)
            operator_dipvel.ao_integrals = integrals_dipvel_ao
            self.solver.add_operator(operator_dipvel)

    def run(self):
        self.solver.run(solver='explicit', hamiltonian=self.hamiltonian, spin=self.spin)

    def form_results(self):
        assert len(self.solver.results) == len(self.frequencies)
        self.polarizabilities = []
        for idxf, frequency in enumerate(self.frequencies):
            # print('=' * 78)
            results = self.solver.results[idxf]
            if self.do_dipvel:
                assert results.shape == (9, 9)
            else:
                assert results.shape == (6, 6)
            # Ordering of the operators:
            # 1. angular momentum
            # 2. dipole (length gauge)
            # 3. dipole (velocity gauge) [if calculated]
            # print('frequency')
            # print(frequency)
            polarizability = results[3:6, 3:6]
            self.polarizabilities.append(polarizability)
            # print('symmetry check')
            abs_diff = results[0:3, 3:6] - results[3:6, 0:3].T
            # print('electric dipole-magnetic dipole')
            # eigvals, iso, _ = tensor_printer(results[0:3, 3:6])
            # eigvals, iso, _ = tensor_printer(results[3:6, 0:3])
