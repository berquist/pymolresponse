import numpy as np

from . import iterators
from .cphf import CPHF
from .td import TDHF, TDA


class MolecularProperty:

    def __init__(self, pyscfmol, mocoeffs, moenergies, occupations, *args, **kwargs):
        # TODO add more type assertions (pyscfmol)
        assert isinstance(mocoeffs, np.ndarray)
        assert isinstance(moenergies, np.ndarray)
        assert isinstance(occupations, (np.ndarray, tuple, list))
        self.pyscfmol = pyscfmol
        self.mocoeffs = mocoeffs
        self.moenergies = moenergies
        self.occupations = np.asarray(occupations)

        self.solver = None
        self.driver = None

    def run(self, hamiltonian=None, spin=None):
        assert hasattr(self, 'driver')
        assert self.driver is not None
        if hamiltonian is None:
            hamiltonian = 'rpa'
        if spin is None:
            spin = 'singlet'
        assert isinstance(hamiltonian, str)
        assert isinstance(spin, str)
        assert self.driver.solver is not None
        self.driver.run(solver_type='exact', hamiltonian=hamiltonian.lower(), spin=spin.lower())

    def form_operators(self):
        raise NotImplementedError("This must be implemented in a grandchild class.")

    def form_results(self):
        raise NotImplementedError("This must be implemented in a grandchild class.")


class ResponseProperty(MolecularProperty):

    def __init__(self, pyscfmol, mocoeffs, moenergies, occupations, frequencies=[0.0], *args, **kwargs):
        super().__init__(pyscfmol, mocoeffs, moenergies, occupations, *args, **kwargs)

        # Don't allow a single number; force one of the basic
        # iterables.
        assert isinstance(frequencies, (list, tuple, np.ndarray))
        self.frequencies = frequencies

        if 'solver' in kwargs:
            solver = kwargs['solver']
        elif self.solver is not None:
            solver = self.solver
        else:
            solver = iterators.ExactInv(mocoeffs, moenergies, occupations)

        # TODO this doesn't belong here.
        if solver.tei_mo is None:
            solver.form_tei_mo(pyscfmol)

        if 'driver' in kwargs:
            driver = kwargs['driver']
        else:
            driver = CPHF(solver)
        self.driver = driver

        self.driver.set_frequencies(frequencies)

    def form_operators(self):
        raise NotImplementedError("This must be implemented in a child class.")

    def form_results(self):
        raise NotImplementedError("This must be implemented in a child class.")


class TransitionProperty(MolecularProperty):

    def __init__(self, pyscfmol, mocoeffs, moenergies, occupations, *args, **kwargs):
        super().__init__(pyscfmol, mocoeffs, moenergies, occupations, *args, **kwargs)

        if 'solver' in kwargs:
            solver = kwargs['solver']
        elif self.solver is not None:
            solver = self.solver
        else:
            solver = iterators.ExactDiagonalizationSolver(mocoeffs, moenergies, occupations)

        # TODO this doesn't belong here.
        if solver.tei_mo is None:
            solver.form_tei_mo(pyscfmol)

        if kwargs.get('driver', None):
            driver = kwargs['driver']
        elif kwargs.get('do_tda', None):
            driver = TDA
        else:
            driver = TDHF
        self.driver = driver(solver)
        assert isinstance(self.driver, (TDHF,))

    def form_operators(self):
        raise NotImplementedError("This must be implemented in a child class.")

    def form_results(self):
        raise NotImplementedError("This must be implemented in a child class.")
