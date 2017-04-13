from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.constants as spc

import pyscf

from . import utils

from .magnetic import Magnetizability, ElectronicGTensor
from molecules import molecule_BC2H4_neutral_radical_HF_STO3G

ref_magnetizability_rhf = np.array([[9.491770490066, -0.000297478459, -2.237615614426],
                                    [-0.000297478459, 51.486607258336, 0.010790985557],
                                    [-2.237615614426, 0.010790985557, 11.943759880337]])

ref_magnetizability_rohf = np.array([[10.730460126094, -0.031625404934, -2.412066190731],
                                     [-0.031625404934, 50.874378111410, -0.064676119234],
                                     [-2.412066190731, -0.064676119234, 13.770943967357]])


def test_magnetizability_rhf():
    mol = pyscf.gto.Mole()
    mol.verbose = 0
    mol.output = None
    with open('glycine.xyz') as fh:
        next(fh)
        next(fh)
        mol.atom = fh.read()
    mol.basis = 'sto-3g'
    mol.charge = 0
    mol.spin = 0
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.scf()

    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = utils.occupations_from_pyscf_mol(mol, C)

    calculator_common = Magnetizability(mol, C, E, occupations, hamiltonian='rpa', spin='singlet')
    calculator_common.form_operators()
    calculator_common.run()
    calculator_common.form_results()
    ref_eigvals, ref_iso, _ = utils.tensor_printer(ref_magnetizability_rhf)
    res_eigvals, res_iso, _ = utils.tensor_printer(calculator_common.magnetizability)
    thresh_eigval = 1.0e-3
    for i in range(3):
        assert abs(ref_eigvals[i] - res_eigvals[i]) < thresh_eigval

    # TODO it isn't so simple; there are actually many more terms
    # present when using London orbitals.
    # calculator_giao = Magnetizability(mol, C, E, occupations, hamiltonian='rpa', spin='singlet', use_giao=True)
    # calculator_giao.form_operators()
    # calculator_giao.run()
    # calculator_giao.form_results()

    return


def test_magnetizability_uhf():
    mol = pyscf.gto.Mole()
    mol.verbose = 5
    mol.output = None
    with open('glycine.xyz') as fh:
        next(fh)
        next(fh)
        mol.atom = fh.read()
    mol.basis = 'sto-3g'
    mol.charge = 1
    mol.spin = 1
    mol.build()

    mf = pyscf.scf.uhf.UHF(mol)
    mf.scf()

    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = utils.occupations_from_pyscf_mol(mol, C)

    calculator_common = Magnetizability(mol, C, E, occupations, hamiltonian='rpa', spin='singlet')
    calculator_common.form_operators()
    calculator_common.run()
    calculator_common.form_results()
    ref_eigvals, ref_iso, _ = utils.tensor_printer(ref_magnetizability_rohf)
    res_eigvals, res_iso, _ = utils.tensor_printer(calculator_common.magnetizability)
    thresh_eigval = 1.0e-1
    for i in range(3):
        assert abs(ref_eigvals[i] - res_eigvals[i]) < thresh_eigval

    return

# def test_electronicgtensor():

#     mol = pyscf.gto.Mole()
#     mol.verbose = 5
#     mol.output = None
#     with open('0w4a.xyz') as fh:
#         next(fh)
#         next(fh)
#         mol.atom = fh.read()
#     mol.basis = '3-21g'
#     mol.charge = 2
#     mol.spin = 1

#     # mol = molecule_BC2H4_neutral_radical_HF_STO3G(5)
#     mol.build()

#     mf = pyscf.scf.uhf.UHF(mol)
#     mf.scf()

#     C = utils.fix_mocoeffs_shape(mf.mo_coeff)
#     E = utils.fix_moenergies_shape(mf.mo_energy)
#     occupations = utils.occupations_from_pyscf_mol(mol, C)

#     # mol.set_common_orig([0.00000000, 0.00000000, 0.94703972])
#     # mol.set_common_orig([-0.00838852, 0.73729752, -1.38870875])
#     mol.set_common_orig([-1.44316614, -1.93460731, -0.04099788])
#     gtensor_calculator = ElectronicGTensor(mol, C, E, occupations, hamiltonian='rpa', spin='singlet')
#     gtensor_calculator.form_operators()
#     gtensor_calculator.run()
#     gtensor_calculator.form_results()

#     return

if __name__ == '__main__':
    # test_magnetizability_rhf()
    # test_magnetizability_uhf()
    # test_electronicgtensor()
    pass
