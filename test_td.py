from __future__ import print_function

from copy import deepcopy

import numpy as np

import pyscf

import ao2mo
import utils

from cphf import Operator
from td import TDA, TDHF
from ecd import ECD


def molecule_BC2H4_cation_HF_STO3G():
    verbose = 0

    mol = pyscf.gto.Mole()
    mol.verbose = verbose
    mol.output = None

    with open('BC2H4.xyz') as fh:
        next(fh)
        next(fh)
        mol.atom = fh.read()
    mol.basis = 'sto-3g'
    mol.charge = 1
    return mol


def molecule_BC2H4_neutral_radical_HF_STO3G():
    mol = molecule_BC2H4_cation_HF_STO3G()
    mol.charge = 0
    mol.spin = 1
    return mol


def test_ECD_RPA_singlet_BC2H4_cation_HF_STO3G():
    mol = molecule_BC2H4_cation_HF_STO3G()
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.scf()

    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = np.diag(mf.mo_energy)[np.newaxis, ...]
    occupations = utils.occupations_from_pyscf_mol(mol, C)

    ecd_dipvel_rpa = ECD(mol, C, E, occupations, do_dipvel=True, do_tda=False)
    ecd_dipvel_rpa.run()
    ecd_dipvel_rpa.form_results()

    ref = BC2H4_cation_STO3G_RPA_nwchem
    nroots = ref['nroots']

    # excitation energies
    ref_etenergies = np.array(BC2H4_cation_STO3G_RPA_nwchem['etenergies'])
    res_etenergies = ecd_dipvel_rpa.solver.eigvals.real[:nroots]
    thresh = 1.0e-7
    for i in range(nroots):
        abs_diff = abs(ref_etenergies[i] - res_etenergies[i])
        assert abs_diff < thresh

    # dipole (length) oscillator strengths
    ref_etoscslen = np.array(BC2H4_cation_STO3G_RPA_nwchem['etoscslen'])
    res_etoscslen = ecd_dipvel_rpa.solver.operators[1].total_oscillator_strengths[:nroots]
    thresh = 1.0e-5
    for i in range(nroots):
        abs_diff = abs(ref_etoscslen[i] - res_etoscslen[i])
        assert abs_diff < thresh

    # TODO dipole (mixed length/velocity) oscillator strengths

    # TODO dipole (velocity) oscillator strengths
    # print(np.array(BC2H4_cation_STO3G_RPA_nwchem['etoscsvel']))
    # print(ecd_dipvel_rpa.solver.operators[2].total_oscillator_strengths[:nroots])

    # rotatory strengths (length)
    ref_etrotstrlen = np.array(BC2H4_cation_STO3G_RPA_nwchem['etrotstrlen'])
    res_etrotstrlen = ecd_dipvel_rpa.rotational_strengths_diplen[:nroots]
    # TODO unlike other quantities, the error isn't uniformly
    # distributed among the roots; how should this be handled?
    thresh = 1.5e+1
    for i in range(nroots):
        abs_diff = abs(ref_etrotstrlen[i] - res_etrotstrlen[i])
        assert abs_diff < thresh

    # rotatory strengths (velocity)
    ref_etrotstrvel = np.array(BC2H4_cation_STO3G_RPA_nwchem['etrotstrvel'])
    res_etrotstrvel = ecd_dipvel_rpa.rotational_strengths_dipvel[:nroots]
    thresh = 1.0e-2
    for i in range(nroots):
        abs_diff = abs(ref_etrotstrvel[i] - res_etrotstrvel[i])
        assert abs_diff < thresh

    return


def test_ECD_RPA_singlet_BC2H4_neutral_radical_HF_STO3G():
    mol = molecule_BC2H4_neutral_radical_HF_STO3G()
    mol.build()

    mf = pyscf.scf.UHF(mol)
    mf.scf()

    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    return


BC2H4_cation_STO3G_RPA_nwchem = {
    'etenergies': [
        0.116938283,
        0.153688860,
        0.302306677,
        0.327380785,
        0.340637548,
        0.391151295,
        0.427233992,
        0.521916988,
        0.534473141,
        0.567055549,
    ],
    'etoscslen': [
        0.01116,
        0.00473,
        0.00458,
        0.09761,
        0.09190,
        0.03514,
        0.03804,
        0.28900,
        0.19604,
        0.43408,
    ],
    'etoscsmix': [
        0.0074981,
        0.0054776,
        0.0052432,
        0.0748931,
        0.0547656,
        0.0207675,
        0.0205230,
        0.1853662,
        0.1077085,
        0.2874928,
    ],
    'etoscsvel': [
        0.0069989,
        0.0076728,
        0.0092702,
        0.0575288,
        0.0327921,
        0.0132039,
        0.0113311,
        0.1195340,
        0.0602382,
        0.1923483,
    ],
    'etrotstrlen': [
        -77.6721763,
        -11.6203780,
        13.6253032,
        203.2296044,
        -2.7209904,
        14.1994514,
        -16.5542125,
        -101.6752655,
        76.2221837,
        -106.0751407,
    ],
    'etrotstrvel': [
        -50.5415342,
        -38.3324307,
        -13.0716770,
        153.6307152,
        -0.2283890,
        14.8708870,
        -9.8364102,
        -73.4642390,
        40.6989398,
        -70.9771590,
    ],
    'nroots': 10,
}

BC2H4_neutral_radical_HF_STO3G_RPA_nwchem = {
    'etenergies': [
    ],
    'etoscslen': [
    ],
    'etoscsmix': [
    ],
    'etoscsvel': [
    ],
    'etrotstrlen': [
    ],
    'etrotstrvel': [
    ],
    'nroots': 10,
}


def test_HF_both_singlet_HF_STO3G():
    mol = pyscf.gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom = [
        ['H' , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)],
    ]
    mol.basis = 'sto-3g'
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.scf()

    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = np.diag(mf.mo_energy)[np.newaxis, ...]
    occupations = utils.occupations_from_pyscf_mol(mol, C)
    tda = TDA(C, E, occupations)
    tdhf = TDHF(C, E, occupations)
    tei_mo = ao2mo.perform_tei_ao2mo_rhf_partial(mol, C, mol.verbose)
    tda.tei_mo = tei_mo
    tda.tei_mo_type = 'partial'
    tdhf.tei_mo = tei_mo
    tdhf.tei_mo_type = 'partial'

    nroots = 5

    print('TDA using TDA()')
    tda.run(solver='explicit', hamiltonian='tda', spin='singlet')
    excitation_energies_tda_using_tda = tda.eigvals[:nroots].real
    print('TDA using TDHF()')
    tdhf.run(solver='explicit', hamiltonian='tda', spin='singlet')
    excitation_energies_tda_using_tdhf = tdhf.eigvals[:nroots].real
    print('RPA using TDHF()')
    tdhf.run(solver='explicit', hamiltonian='rpa', spin='singlet')
    excitation_energies_rpa = tdhf.eigvals[:nroots].real

    assert excitation_energies_tda_using_tda.shape == excitation_energies_tda_using_tdhf.shape
    assert excitation_energies_tda_using_tdhf.shape == excitation_energies_rpa.shape

    # There should be no difference in the TDA results regardless of
    # which implementation used.
    assert (excitation_energies_tda_using_tda - excitation_energies_tda_using_tdhf).all() == 0

    # Now compare against reference_data
    ref_tda = HF_neutral_singlet_HF_STO3G_CIS_qchem
    ref_rpa = HF_neutral_singlet_HF_STO3G_RPA_qchem

    thresh = 1.0e-7
    for i in range(nroots):
        abs_diff = abs(ref_tda['etenergies'][i] - excitation_energies_tda_using_tda[i])
        assert abs_diff < thresh

    thresh = 1.0e-7
    for i in range(nroots):
        abs_diff = abs(ref_rpa['etenergies'][i] - excitation_energies_rpa[i])
        assert abs_diff < thresh

    return


HF_neutral_singlet_HF_STO3G_CIS_qchem = {
    'etenergies': [
        -98.067181246814 - -98.5707799863,
        -98.067181246768 - -98.5707799863,
        -97.686596454655 - -98.5707799863,
        -96.958042326818 - -98.5707799863,
        -72.879307887356 - -98.5707799863,
    ],
    'etoscslen': [
        0.0002003489,
        0.0002003489,
        0.9621809426,
        0.0531137481,
        0.0691994928,
    ],
}


HF_neutral_singlet_HF_STO3G_RPA_qchem = {
    'etenergies': [
        -98.068185050585 - -98.5707799863,
        -98.068185050538 - -98.5707799863,
        -97.703584999956 - -98.5707799863,
        -96.962988495302 - -98.5707799863,
        -72.879331844690 - -98.5707799863,
    ],
    'etoscslen': [
        0.0001877054,
        0.0001877054,
        0.7777380206,
        0.0322221420,
        0.0686085799,
    ],
}
