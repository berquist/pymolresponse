from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.constants as spc

import pyscf

from . import molecules
from . import utils

from .magnetic import Magnetizability, ElectronicGTensor


# These were generated using DALTON.
ref_magnetizability_rhf = np.array([[9.491770490066, -0.000297478459, -2.237615614426],
                                    [-0.000297478459, 51.486607258336, 0.010790985557],
                                    [-2.237615614426, 0.010790985557, 11.943759880337]])

ref_magnetizability_rohf = np.array([[10.730460126094, -0.031625404934, -2.412066190731],
                                     [-0.031625404934, 50.874378111410, -0.064676119234],
                                     [-2.412066190731, -0.064676119234, 13.770943967357]])


def test_magnetizability_rhf():
    mol = molecules.molecule_glycine_HF_STO3G()
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.scf()

    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = utils.occupations_from_pyscf_mol(mol, C)

    calculator_common = Magnetizability(mol, C, E, occupations)
    calculator_common.form_operators()
    calculator_common.run(hamiltonian='rpa', spin='singlet')
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
    mol = molecules.molecule_glycine_HF_STO3G()
    mol.charge = 1
    mol.spin = 1
    mol.build()

    mf = pyscf.scf.uhf.UHF(mol)
    mf.scf()

    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = utils.occupations_from_pyscf_mol(mol, C)

    calculator_common = Magnetizability(mol, C, E, occupations)
    calculator_common.form_operators()
    calculator_common.run(hamiltonian='rpa', spin='singlet')
    calculator_common.form_results()
    ref_eigvals, ref_iso, _ = utils.tensor_printer(ref_magnetizability_rohf)
    res_eigvals, res_iso, _ = utils.tensor_printer(calculator_common.magnetizability)
    thresh_eigval = 1.0e-1
    for i in range(3):
        assert abs(ref_eigvals[i] - res_eigvals[i]) < thresh_eigval

    return


# These were generated with this program.
ref_electronicgtensor_tiny = np.array([[1.15894158e-05, 0.0, 0.0],
                                       [0.0, 1.15894158e-05, 0.0],
                                       [0.0, 0.0, 0.0]])
ref_electronicgtensor_small = np.array([[2.68449088e-03, -7.88488679e-04, -1.71222252e-04],
                                        [-6.68698723e-04, 2.61494786e-03, -4.68601905e-05],
                                        [-1.67939714e-04, 1.02402038e-05, 4.19142287e-03]])
ref_electronicgtensor_large = np.array([[0.30798458, -0.07215188, 0.07146523],
                                        [-0.07215626, 0.82904402, -0.52638538],
                                        [0.07145804, -0.5263783, 0.82010589]])


def test_electronicgtensor_tiny():

    mol = molecules.molecule_LiH_cation_HF_STO3G(0)
    mol.build()

    mf = pyscf.scf.uhf.UHF(mol)
    mf.scf()

    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = utils.occupations_from_pyscf_mol(mol, C)

    gtensor_calculator = ElectronicGTensor(mol, C, E, occupations)
    gtensor_calculator.form_operators()
    gtensor_calculator.run(hamiltonian='rpa', spin='singlet')
    gtensor_calculator.form_results()

    print(ref_electronicgtensor_tiny)
    print(gtensor_calculator.g_oz_soc_1)

    assert np.all(np.equal(np.sign(ref_electronicgtensor_tiny),
                           np.sign(utils.screen(gtensor_calculator.g_oz_soc_1))))

    return


def test_electronicgtensor_small():

    mol = molecules.molecule_BC2H4_neutral_radical_HF_STO3G(0)
    mol.build()

    mf = pyscf.scf.uhf.UHF(mol)
    mf.scf()

    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = utils.occupations_from_pyscf_mol(mol, C)

    gtensor_calculator = ElectronicGTensor(mol, C, E, occupations)
    gtensor_calculator.form_operators()
    gtensor_calculator.run(hamiltonian='rpa', spin='singlet')
    gtensor_calculator.form_results()

    print(ref_electronicgtensor_small)
    print(gtensor_calculator.g_oz_soc_1)

    assert np.all(np.equal(np.sign(ref_electronicgtensor_small),
                           np.sign(gtensor_calculator.g_oz_soc_1)))

    return


# def test_electronicgtensor_large():

#     mol = molecules.molecule_0w4a_dication_HF_321G(0)
#     mol.build()

#     mf = pyscf.scf.uhf.UHF(mol)
#     mf.scf()

#     C = utils.fix_mocoeffs_shape(mf.mo_coeff)
#     E = utils.fix_moenergies_shape(mf.mo_energy)
#     occupations = utils.occupations_from_pyscf_mol(mol, C)

#     gtensor_calculator = ElectronicGTensor(mol, C, E, occupations, hamiltonian='rpa', spin='singlet')
#     gtensor_calculator.form_operators()
#     gtensor_calculator.run()
#     gtensor_calculator.form_results()

#     print(ref_electronicgtensor_large)
#     print(gtensor_calculator.g_oz_soc_1)

#     assert np.all(np.equal(np.sign(ref_electronicgtensor_large),
#                            np.sign(gtensor_calculator.g_oz_soc_1)))

#     return

if __name__ == '__main__':
    test_magnetizability_rhf()
    test_magnetizability_uhf()
    test_electronicgtensor_tiny()
    test_electronicgtensor_small()
    # test_electronicgtensor_large()
