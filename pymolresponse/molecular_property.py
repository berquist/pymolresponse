from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pymolresponse.cphf import CPHF
from pymolresponse.td import TDHF


if TYPE_CHECKING:
    from pymolresponse.core import Hamiltonian, Program, Spin
    from pymolresponse.cphf import Driver


class MolecularProperty(ABC):
    """A molecular property that is calculated from one or more operators.

    The TODO
    """

    def __init__(self, program: "Program", program_obj: Any, driver: "Driver") -> None:
        # TODO isinstance(program_obj, ...)
        self.program = program
        self.program_obj = program_obj
        self.driver = driver

    def run(self, hamiltonian: "Hamiltonian", spin: "Spin") -> None:
        assert self.driver.solver is not None
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
        self, program: "Program", program_obj: Any, driver: "CPHF", *, frequencies: Sequence[float]
    ) -> None:
        driver.set_frequencies(frequencies)
        super().__init__(program, program_obj, driver)
        assert isinstance(self.driver, CPHF)  # for ty
        self.frequencies = self.driver.frequencies


class TransitionProperty(MolecularProperty, ABC):
    def __init__(self, program: "Program", program_obj: Any, driver: "TDHF"):
        super().__init__(program, program_obj, driver)
        assert isinstance(self.driver, TDHF)  # for ty
