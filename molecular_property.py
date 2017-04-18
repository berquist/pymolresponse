import numpy as np

from . import ao2mo
from .cphf import CPHF
from .td import TDHF, TDA


class MolecularProperty(object):

    def __init__(self, pyscfmol, mocoeffs, moenergies, occupations, hamiltonian, spin, *args, **kwargs):
        # TODO add more type assertions (pyscfmol)
        assert isinstance(mocoeffs, np.ndarray)
        assert isinstance(moenergies, np.ndarray)
        assert isinstance(occupations, (np.ndarray, tuple, list))
        assert isinstance(hamiltonian, str)
        assert isinstance(spin, str)
        self.pyscfmol = pyscfmol
        self.mocoeffs = mocoeffs
        self.moenergies = moenergies
        self.occupations = np.asarray(occupations)
        self.hamiltonian = hamiltonian.lower()
        self.spin = spin.lower()

        self.solver = None

    def form_tei_mo(self):
        assert hasattr(self, 'solver')
        assert self.solver is not None
        nden = self.mocoeffs.shape[0]
        if not hasattr(self.solver, 'tei_mo') or self.solver.tei_mo is None:
            if nden == 2:
                tei_mo_func = ao2mo.perform_tei_ao2mo_uhf_partial
            else:
                tei_mo_func = ao2mo.perform_tei_ao2mo_rhf_partial
            self.solver.tei_mo = tei_mo_func(self.pyscfmol, self.mocoeffs, self.pyscfmol.verbose)
            self.solver.tei_mo_type = 'partial'

    def run(self, hamiltonian=None, spin=None):
        assert hasattr(self, 'solver')
        assert self.solver is not None
        if not hamiltonian:
            hamiltonian = self.hamiltonian
        if not spin:
            spin = self.spin
        self.solver.run(solver='explicit', hamiltonian=hamiltonian, spin=spin)


class ResponseProperty(MolecularProperty):

    def __init__(self, pyscfmol, mocoeffs, moenergies, occupations, hamiltonian, spin, frequencies=[0.0], *args, **kwargs):
        super().__init__(pyscfmol, mocoeffs, moenergies, occupations, hamiltonian, spin, *args, **kwargs)

        # Don't allow a single number; force one of the basic
        # iterables.
        assert isinstance(frequencies, (list, tuple, np.ndarray))
        self.frequencies = frequencies

        if 'solver' in kwargs:
            solver = kwargs['solver']
        else:
            solver = CPHF(mocoeffs, moenergies, occupations)
        self.solver = solver

        self.solver.set_frequencies(frequencies)

        self.form_tei_mo()

class TransitionProperty(MolecularProperty):

    def __init__(self, pyscfmol, mocoeffs, moenergies, occupations, hamiltonian, spin, *args, **kwargs):
        super().__init__(pyscfmol, mocoeffs, moenergies, occupations, hamiltonian, spin, *args, **kwargs)

        if 'solver' in kwargs:
            solver = kwargs['solver']
        elif hamiltonian == 'tda':
            solver = TDA
        else:
            solver = TDHF
        self.solver = solver(mocoeffs, moenergies, occupations)

        self.form_tei_mo()
