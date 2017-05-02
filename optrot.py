from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.constants as spc

from .operators import Operator
from .molecular_property import ResponseProperty
from .utils import tensor_printer


class ORD(ResponseProperty):

    def __init__(self, pyscfmol, mocoeffs, moenergies, occupations, frequencies, do_dipvel=False, *args, **kwargs):
        super().__init__(pyscfmol, mocoeffs, moenergies, occupations, frequencies, *args, **kwargs)
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

        assert len(self.driver.results) == len(self.frequencies)
        self.polarizabilities = []
        for idxf, frequency in enumerate(self.frequencies):
            # print('=' * 78)
            results = self.driver.results[idxf]
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
