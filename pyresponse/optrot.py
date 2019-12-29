from pyresponse.interfaces import Program
from pyresponse.molecular_property import ResponseProperty
from pyresponse.operators import Operator


class ORD(ResponseProperty):
    def __init__(
        self,
        program,
        program_obj,
        mocoeffs,
        moenergies,
        occupations,
        frequencies,
        do_dipvel=False,
        *args,
        **kwargs
    ):
        super().__init__(
            program,
            program_obj,
            mocoeffs,
            moenergies,
            occupations,
            frequencies,
            *args,
            **kwargs
        )
        self.do_dipvel = do_dipvel

    def form_operators(self):

        if self.program == Program.PySCF:
            from pyresponse.pyscf import integrals

            integral_generator = integrals.IntegralsPyscf(self.program_obj)
        elif self.program == Program.Psi4:
            from pyresponse.psi4 import integrals

            integral_generator = integrals.IntegralsPsi4(self.program_obj)
        else:
            raise RuntimeError

        operator_angmom = Operator(
            label="angmom", is_imaginary=True, is_spin_dependent=False, triplet=False
        )
        operator_angmom.ao_integrals = integral_generator.integrals(
            integrals.ANGMOM_COMMON_GAUGE
        )
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

    def form_results(self):

        assert len(self.driver.results) == len(self.frequencies)
        self.polarizabilities = []
        self.polarizabilities_lenmag = []

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
            # abs_diff = results[0:3, 3:6] - results[3:6, 0:3].T
            # print(abs_diff)
            # print('electric dipole-magnetic dipole polarizabilities')
            # eigvals, iso, _ = tensor_printer(results[0:3, 3:6])
            # eigvals, iso, _ = tensor_printer(results[3:6, 0:3])
            polarizability_lenmag = results[0:3, 3:6]
            self.polarizabilities_lenmag.append(polarizability_lenmag)
