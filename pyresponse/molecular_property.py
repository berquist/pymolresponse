from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from pyresponse.core import Hamiltonian, Program, Spin
from pyresponse.cphf import CPHF, Driver
from pyresponse.td import TDHF


class MolecularProperty(ABC):
    """A molecular property that is calculated from one or more operators.

    The
    """

    def __init__(
        self,
        program: Program,
        program_obj,
        driver: Driver,
        mocoeffs: np.ndarray,
        moenergies: np.ndarray,
        occupations: np.ndarray,
    ) -> None:
        assert isinstance(program, Program)
        # TODO isinstance(program_obj, ...)
        assert isinstance(mocoeffs, np.ndarray)
        assert isinstance(moenergies, np.ndarray)
        assert isinstance(occupations, np.ndarray)
        assert isinstance(driver, Driver)
        self.program = program
        self.program_obj = program_obj
        self.driver = driver
        self.mocoeffs = mocoeffs
        self.moenergies = moenergies
        self.occupations = occupations

    def run(self, hamiltonian: Hamiltonian, spin: Spin) -> None:
        assert self.driver.solver is not None
        assert isinstance(hamiltonian, Hamiltonian)
        assert isinstance(spin, Spin)
        self.driver.run(
            hamiltonian=hamiltonian,
            spin=spin,
            program=self.program,
            program_obj=self.program_obj,
        )

    @abstractmethod
    def form_operators(self):
        pass

    @abstractmethod
    def form_results(self):
        pass


class ResponseProperty(MolecularProperty, ABC):
    """A molecular property that is calculated as the response to one or more
    perturbations, each represented by an operator.
    """

    def __init__(
        self,
        program: Program,
        program_obj,
        driver: CPHF,
        mocoeffs: np.ndarray,
        moenergies: np.ndarray,
        occupations: np.ndarray,
        *,
        frequencies: Sequence[float],
    ):
        driver.set_frequencies(frequencies)
        super().__init__(program, program_obj, driver, mocoeffs, moenergies, occupations)
        self.frequencies = self.driver.frequencies

    @abstractmethod
    def form_operators(self):
        pass

    @abstractmethod
    def form_results(self):
        pass


class TransitionProperty(MolecularProperty, ABC):
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
    ):

        # if driver is None:
        #     driver_cls = TDA if do_tda else TDHF
        #     driver = driver_cls(
        #         solvers.ExactDiagonalizationSolver(mocoeffs, moenergies, occupations)
        #     )
        assert isinstance(do_tda, bool)
        self.do_tda = do_tda

        super().__init__(program, program_obj, driver, mocoeffs, moenergies, occupations)

    @abstractmethod
    def form_operators(self) -> None:
        pass

    @abstractmethod
    def form_results(self) -> None:
        pass
