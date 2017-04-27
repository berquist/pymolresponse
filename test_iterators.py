from __future__ import print_function
from __future__ import division

import numpy as np

import pyscf

from . import iterators
from . import molecules
from . import utils
from .magnetic import Magnetizability


def test_iterators():

    mol = molecules.molecule_glycine_HF_STO3G()
    mol.charge = 1
    mol.spin = 1
    mol.build()

    mf = pyscf.scf.uhf.UHF(mol)
    mf.scf()

    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = utils.occupations_from_pyscf_mol(mol, C)

    solver_inv = iterators.ExactInv(C, E, occupations)
    solver_pinv = iterators.ExactPinv(C, E, occupations)

    calculator_inv = Magnetizability(mol, C, E, occupations, solver=solver_inv)
    calculator_inv.form_operators()
    calculator_inv.run(hamiltonian='rpa', spin='singlet')
    calculator_inv.form_results()
    print(calculator_inv.magnetizability)
    calculator_pinv = Magnetizability(mol, C, E, occupations, solver=solver_pinv)
    calculator_pinv.form_operators()
    calculator_pinv.run(hamiltonian='rpa', spin='singlet')
    calculator_pinv.form_results()
    print(calculator_pinv.magnetizability)

    assert np.all(np.equal(np.sign(calculator_inv.magnetizability),
                           np.sign(calculator_pinv.magnetizability)))
    thresh = 1.0e-14
    assert np.all(np.abs(calculator_inv.magnetizability - calculator_inv.magnetizability) < thresh)

    return


if __name__ == '__main__':
    test_iterators()
