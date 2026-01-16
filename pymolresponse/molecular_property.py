from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from pymolresponse.core import Hamiltonian, Program, Spin
from pymolresponse.cphf import CPHF, Driver
from pymolresponse.td import TDHF

if TYPE_CHECKING:
    from pymolresponse.indices import Occupations


class MolecularProperty(ABC):
    """A molecular property that is calculated from one or more operators.

    The TODO
    """

    def __init__(
        self,
        program: Program,
        program_obj: Any,
        driver: Driver,
        mocoeffs: np.ndarray,
        moenergies: np.ndarray,
        occupations: "Occupations",
    ) -> None:
        assert isinstance(program, Program)
        # TODO isinstance(program_obj, ...)
        assert isinstance(mocoeffs, np.ndarray)
        assert isinstance(moenergies, np.ndarray)
        assert isinstance(occupations, tuple)
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
            hamiltonian=hamiltonian, spin=spin, program=self.program, program_obj=self.program_obj
        )

    @abstractmethod
    def form_operators(self) -> None:
        pass

    @abstractmethod
    def form_results(self) -> None:
        pass


class ResponseProperty(MolecularProperty, ABC):
    """A molecular property that is calculated as the response to one or more
    perturbations, each represented by an operator.
    """

    def __init__(
        self,
        program: Program,
        program_obj: Any,
        driver: CPHF,
        mocoeffs: np.ndarray,
        moenergies: np.ndarray,
        occupations: "Occupations",
        *,
        frequencies: Sequence[float],
    ) -> None:
        driver.set_frequencies(frequencies)
        super().__init__(program, program_obj, driver, mocoeffs, moenergies, occupations)
        self.frequencies = self.driver.frequencies

    @abstractmethod
    def form_operators(self) -> None:
        pass

    @abstractmethod
    def form_results(self) -> None:
        pass


class TransitionProperty(MolecularProperty, ABC):
    def __init__(
        self,
        program: Program,
        program_obj: Any,
        driver: TDHF,
        mocoeffs: np.ndarray,
        moenergies: np.ndarray,
        occupations: "Occupations",
    ):
        super().__init__(program, program_obj, driver, mocoeffs, moenergies, occupations)

    @abstractmethod
    def form_operators(self) -> None:
        pass

    @abstractmethod
    def form_results(self) -> None:
        pass
