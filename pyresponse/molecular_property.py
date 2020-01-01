from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np

from pyresponse import iterators
from pyresponse.core import Hamiltonian, Program, Spin
from pyresponse.cphf import CPHF, Driver
from pyresponse.iterators import Solver
from pyresponse.td import TDA, TDHF


class MolecularProperty(ABC):
    """A molecular property that is calculated from one or more operators.

    The
    """

    def __init__(
        self,
        program: Program,
        program_obj,
        mocoeffs: np.ndarray,
        moenergies: np.ndarray,
        occupations: np.ndarray,
        *,
        driver: Optional[Driver] = None,
    ) -> None:
        assert isinstance(program, Program)
        # TODO isinstance(program_obj, ...)
        assert isinstance(mocoeffs, np.ndarray)
        assert isinstance(moenergies, np.ndarray)
        assert isinstance(occupations, np.ndarray)
        assert isinstance(driver, (type(None), Driver))
        self.program = program
        self.program_obj = program_obj
        self.mocoeffs = mocoeffs
        self.moenergies = moenergies
        self.occupations = occupations
        self.driver = driver

    def run(self, hamiltonian: Hamiltonian, spin: Spin) -> None:
        assert self.driver is not None
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
        mocoeffs: np.ndarray,
        moenergies: np.ndarray,
        occupations: np.ndarray,
        *,
        driver: Optional[Driver] = None,
        frequencies: Sequence[float] = [0.0],
    ):
        if driver is None:
            driver = CPHF(iterators.ExactInv(mocoeffs, moenergies, occupations))
        driver.set_frequencies(frequencies)
        super().__init__(
            program, program_obj, mocoeffs, moenergies, occupations, driver=driver
        )
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
        mocoeffs: np.ndarray,
        moenergies: np.ndarray,
        occupations: np.ndarray,
        *,
        driver: Optional[Driver] = None,
        do_tda: bool = False,
    ):

        if driver is None:
            driver_cls = TDA if do_tda else TDHF
            driver = driver_cls(
                iterators.ExactDiagonalizationSolver(mocoeffs, moenergies, occupations)
            )
        self.do_tda = do_tda

        super().__init__(
            program, program_obj, mocoeffs, moenergies, occupations, driver=driver
        )

    @abstractmethod
    def form_operators(self) -> None:
        pass

    @abstractmethod
    def form_results(self) -> None:
        pass
