from abc import ABC, abstractmethod

import numpy as np

from pyresponse import iterators
from pyresponse.cphf import CPHF
from pyresponse.interfaces import Program
from pyresponse.td import TDA, TDHF


class MolecularProperty(ABC):
    def __init__(
        self, program, program_obj, mocoeffs, moenergies, occupations, *args, **kwargs
    ):
        assert isinstance(mocoeffs, np.ndarray)
        assert isinstance(moenergies, np.ndarray)
        assert isinstance(occupations, (np.ndarray, tuple, list))
        self.program = program
        self.program_obj = program_obj
        self.mocoeffs = mocoeffs
        self.moenergies = moenergies
        self.occupations = np.asarray(occupations)

        self.solver = None
        self.driver = None

    def run(self, hamiltonian=None, spin=None):
        assert hasattr(self, "driver")
        assert self.driver is not None
        if hamiltonian is None:
            hamiltonian = "rpa"
        if spin is None:
            spin = "singlet"
        assert isinstance(hamiltonian, str)
        assert isinstance(spin, str)
        assert self.driver.solver is not None
        self.driver.run(
            solver_type="exact",
            hamiltonian=hamiltonian.lower(),
            spin=spin.lower(),
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
    def __init__(
        self,
        program,
        program_obj,
        mocoeffs,
        moenergies,
        occupations,
        frequencies=[0.0],
        *args,
        **kwargs
    ):
        super().__init__(
            program, program_obj, mocoeffs, moenergies, occupations, *args, **kwargs
        )

        # Don't allow a single number; force one of the basic
        # iterables.
        assert isinstance(frequencies, (list, tuple, np.ndarray))
        self.frequencies = frequencies

        if "solver" in kwargs:
            solver = kwargs["solver"]
        elif self.solver is not None:
            solver = self.solver
        else:
            solver = iterators.ExactInv(mocoeffs, moenergies, occupations)

        if "driver" in kwargs:
            driver = kwargs["driver"]
        else:
            driver = CPHF(solver)
        self.driver = driver

        self.driver.set_frequencies(frequencies)

    @abstractmethod
    def form_operators(self):
        pass

    @abstractmethod
    def form_results(self):
        pass


class TransitionProperty(MolecularProperty, ABC):
    def __init__(
        self, program, program_obj, mocoeffs, moenergies, occupations, *args, **kwargs
    ):
        super().__init__(
            program, program_obj, mocoeffs, moenergies, occupations, *args, **kwargs
        )

        if "solver" in kwargs:
            solver = kwargs["solver"]
        elif self.solver is not None:
            solver = self.solver
        else:
            solver = iterators.ExactDiagonalizationSolver(
                mocoeffs, moenergies, occupations
            )

        if kwargs.get("driver", None):
            driver = kwargs["driver"]
        elif kwargs.get("do_tda", None):
            driver = TDA
        else:
            driver = TDHF
        self.driver = driver(solver)
        assert isinstance(self.driver, (TDHF,))

    @abstractmethod
    def form_operators(self):
        pass

    @abstractmethod
    def form_results(self):
        pass
