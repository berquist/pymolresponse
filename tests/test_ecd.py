import numpy as np

import pyscf

from pyresponse import utils, ecd
from . import molecules_pyscf as molecules


BC2H4_cation_HF_STO3G_RPA_singlet_nwchem = {
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


BC2H4_neutral_radical_HF_STO3G_RPA_singlet_nwchem = {
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


def test_ECD_RPA_singlet_BC2H4_cation_HF_STO3G():

    ref = BC2H4_cation_HF_STO3G_RPA_singlet_nwchem
    nroots = ref['nroots']

    mol = molecules.molecule_bc2h4_cation_sto3g()
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.scf()

    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = utils.occupations_from_pyscf_mol(mol, C)

    ecd_dipvel_rpa = ecd.ECD(mol, C, E, occupations, do_dipvel=True)
    ecd_dipvel_rpa.form_operators()
    ecd_dipvel_rpa.run(hamiltonian='rpa', spin='singlet')
    ecd_dipvel_rpa.form_results()

    print('excitation energies')
    ref_etenergies = np.array(ref['etenergies'])
    res_etenergies = ecd_dipvel_rpa.driver.solver.eigvals.real[:nroots]
    print('ref, res')
    for refval, resval in zip(ref_etenergies, res_etenergies):
        print(refval, resval)
    thresh = 1.0e-7
    for i in range(nroots):
        abs_diff = abs(ref_etenergies[i] - res_etenergies[i])
        assert abs_diff < thresh

    print('dipole (length) oscillator strengths')
    ref_etoscslen = np.array(ref['etoscslen'])
    res_etoscslen = ecd_dipvel_rpa.driver.solver.operators[1].total_oscillator_strengths[:nroots]
    print('ref, res')
    for refval, resval in zip(ref_etoscslen, res_etoscslen):
        print(refval, resval)
    thresh = 1.0e-5
    for i in range(nroots):
        abs_diff = abs(ref_etoscslen[i] - res_etoscslen[i])
        assert abs_diff < thresh

    # TODO
    print('TODO dipole (mixed length/velocity) oscillator strengths')

    # TODO
    print('TODO dipole (velocity) oscillator strengths')
    # print(np.array(BC2H4_cation_HF_STO3G_RPA_singlet_nwchem['etoscsvel']))
    # print(ecd_dipvel_rpa.solver.operators[2].total_oscillator_strengths[:nroots])

    print('rotatory strengths (length)')
    ref_etrotstrlen = np.array(ref['etrotstrlen'])
    res_etrotstrlen = ecd_dipvel_rpa.rotational_strengths_diplen[:nroots]
    print('ref, res')
    for refval, resval in zip(ref_etrotstrlen, res_etrotstrlen):
        print(refval, resval)
    # TODO unlike other quantities, the error isn't uniformly
    # distributed among the roots; how should this be handled?
    thresh = 1.5e+1
    for i in range(nroots):
        abs_diff = abs(ref_etrotstrlen[i] - res_etrotstrlen[i])
        assert abs_diff < thresh

    print('rotatory strengths (velocity)')
    ref_etrotstrvel = np.array(ref['etrotstrvel'])
    res_etrotstrvel = ecd_dipvel_rpa.rotational_strengths_dipvel[:nroots]
    print('ref, res')
    for refval, resval in zip(ref_etrotstrvel, res_etrotstrvel):
        print(refval, resval)
    thresh = 1.0e-2
    for i in range(nroots):
        abs_diff = abs(ref_etrotstrvel[i] - res_etrotstrvel[i])
        assert abs_diff < thresh

    return


# TODO once UHF is done
# def test_ECD_RPA_singlet_BC2H4_neutral_radical_HF_STO3G():
#     mol = molecules.molecule_bc2h4_neutral_radical_sto3g()
#     mol.build()

#     mf = pyscf.scf.UHF(mol)
#     mf.scf()

#     C = utils.fix_mocoeffs_shape(mf.mo_coeff)
#     return


if __name__ == '__main__':
    test_ECD_RPA_singlet_BC2H4_cation_HF_STO3G()
