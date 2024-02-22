"""Wrapper for performing a dipole polarizability calculation."""

from typing import Sequence

import numpy as np

from pymolresponse.core import Program
from pymolresponse.cphf import CPHF
from pymolresponse.molecular_property import ResponseProperty
from pymolresponse.operators import Operator


class Polarizability(ResponseProperty):
    """Wrapper for performing a dipole polarizability calculation."""

    def __init__(
        self,
        program: Program,
        program_obj,
        driver: CPHF,
        mocoeffs: np.ndarray,
        moenergies: np.ndarray,
        occupations: np.ndarray,
        *,
        frequencies: Sequence[float] = [0.0],
    ) -> None:
        super().__init__(
            program, program_obj, driver, mocoeffs, moenergies, occupations, frequencies=frequencies
        )
        self.polarizabilities = []

    def form_operators(self) -> None:
        if self.program == Program.PySCF:
            from pymolresponse.interfaces.pyscf import integrals

            integral_generator = integrals.IntegralsPyscf(self.program_obj)
        elif self.program == Program.Psi4:
            from pymolresponse.interfaces.psi4 import integrals

            integral_generator = integrals.IntegralsPsi4(self.program_obj)
        else:
            raise RuntimeError

        operator_diplen = Operator(
            label="dipole", is_imaginary=False, is_spin_dependent=False, triplet=False
        )
        operator_diplen.ao_integrals = integral_generator.integrals(integrals.DIPOLE)
        self.driver.add_operator(operator_diplen)

    def form_results(self) -> None:
        assert len(self.driver.results) == len(self.frequencies)

        for idxf, frequency in enumerate(self.frequencies):
            results = self.driver.results[idxf]
            assert results.shape == (3, 3)
            self.polarizabilities.append(results)
