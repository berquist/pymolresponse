from .molecular_property import ResponseProperty
from .operators import Operator


class Polarizability(ResponseProperty):

    def __init__(self, pyscfmol, mocoeffs, moenergies, occupations, frequencies, *args, **kwargs):
        super().__init__(pyscfmol, mocoeffs, moenergies, occupations, frequencies, *args, **kwargs)

    def form_operators(self):

        operator_diplen = Operator(label='dipole', is_imaginary=False, is_spin_dependent=False, triplet=False)
        integrals_diplen_ao = self.pyscfmol.intor('cint1e_r_sph', comp=3)
        operator_diplen.ao_integrals = integrals_diplen_ao
        self.driver.add_operator(operator_diplen)

    def form_results(self):

        assert len(self.driver.results) == len(self.frequencies)
        self.polarizabilities = []
        for idxf, frequency in enumerate(self.frequencies):
            # print('=' * 78)
            results = self.driver.results[idxf]
            assert results.shape == (3, 3)
            # print('frequency')
            # print(frequency)
            self.polarizabilities.append(results)
