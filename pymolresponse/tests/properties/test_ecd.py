import numpy as np

import pyscf

from pymolresponse import solvers, td, utils
from pymolresponse.core import Hamiltonian, Program, Spin
from pymolresponse.interfaces.pyscf import molecules
from pymolresponse.interfaces.pyscf.utils import occupations_from_pyscf_mol
from pymolresponse.properties import ecd

BC2H4_cation_HF_STO3G_RPA_singlet_nwchem = {
    "etenergies": [
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
    "etoscslen": [
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
    "etoscsmix": [
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
    "etoscsvel": [
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
    "etrotstrlen": [
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
    "etrotstrvel": [
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
    "nroots": 10,
}


BC2H4_neutral_radical_HF_STO3G_RPA_singlet_nwchem = {
    "etenergies": [],
    "etoscslen": [],
    "etoscsmix": [],
    "etoscsvel": [],
    "etrotstrlen": [],
    "etrotstrvel": [],
    "nroots": 10,
}


BC2H4_cation_HF_STO3G_TDA_singlet_orca = {
    "etenergies": [
        0.125219,
        0.160132,
        0.304206,
        0.330029,
        0.346364,
        0.394249,
        0.428824,
        0.525677,
        0.540485,
        0.573778,
    ],
    "etoscslen": [
        0.017454719,
        0.007513348,
        0.004928672,
        0.111374395,
        0.117626606,
        0.048377204,
        0.044244183,
        0.180619359,
        0.349435586,
        0.492350077,
    ],
    "etoscsmix": [],
    "etoscsvel": [
        0.016456475,
        0.000843190,
        0.010314035,
        0.022709524,
        0.005049133,
        0.004114132,
        0.006144748,
        0.043759216,
        0.051361944,
        0.124348134,
    ],
    "etrotstrlen": [
        -88.56094,
        -12.35844,
        14.45593,
        209.05890,
        0.42486,
        23.08029,
        -17.22645,
        -96.09431,
        76.23472,
        -135.00060,
    ],
    "etrotstrvel": [],
    "nroots": 10,
}


def test_ECD_TDA_singlet_BC2H4_cation_HF_STO3G() -> None:
    ref = BC2H4_cation_HF_STO3G_TDA_singlet_orca
    nroots = ref["nroots"]

    mol = molecules.molecule_bc2h4_cation_sto3g()
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.scf()

    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = occupations_from_pyscf_mol(mol, C)

    ecd_dipvel_tda = ecd.ECD(
        Program.PySCF,
        mol,
        td.TDHF(solvers.ExactDiagonalizationSolver(C, E, occupations)),
        C,
        E,
        occupations,
        do_dipvel=True,
    )
    ecd_dipvel_tda.form_operators()
    ecd_dipvel_tda.run(hamiltonian=Hamiltonian.TDA, spin=Spin.singlet)
    ecd_dipvel_tda.form_results()

    print("excitation energies")
    ref_etenergies = np.array(ref["etenergies"])
    res_etenergies = ecd_dipvel_tda.driver.solver.eigvals.real[:nroots]
    print("ref, res")
    for refval, resval in zip(ref_etenergies, res_etenergies):
        print(refval, resval)
    # TODO this might be from ORCA, should use NWChem instead
    thresh = 1.0e-3
    for i in range(nroots):
        abs_diff = abs(ref_etenergies[i] - res_etenergies[i])
        assert abs_diff < thresh

    print("dipole (length) oscillator strengths")
    ref_etoscslen = np.array(ref["etoscslen"])
    res_etoscslen = ecd_dipvel_tda.driver.solver.operators[1].total_oscillator_strengths[:nroots]
    print("ref, res")
    for refval, resval in zip(ref_etoscslen, res_etoscslen):
        print(refval, resval)
    thresh = 1.0e-3
    # np.testing.assert_allclose(ref_etoscslen, res_etoscslen)
    for i in range(nroots):
        abs_diff = abs(ref_etoscslen[i] - res_etoscslen[i])
        assert abs_diff < thresh

    # TODO
    print("TODO dipole (mixed length/velocity) oscillator strengths")

    # TODO
    print("TODO dipole (velocity) oscillator strengths")

    print("rotatory strengths (length)")
    ref_etrotstrlen = np.array(ref["etrotstrlen"])
    res_etrotstrlen = ecd_dipvel_tda.rotational_strengths_diplen[:nroots]
    print("ref, res")
    for refval, resval in zip(ref_etrotstrlen, res_etrotstrlen):
        print(refval, resval)
    # TODO unlike other quantities, the error isn't uniformly
    # distributed among the roots; how should this be handled?
    thresh = 1.0e2
    for i in range(nroots):
        abs_diff = abs(ref_etrotstrlen[i] - res_etrotstrlen[i])
        assert abs_diff < thresh

    # print('rotatory strengths (velocity)')
    # ref_etrotstrvel = np.array(ref['etrotstrvel'])
    # res_etrotstrvel = ecd_dipvel_tda.rotational_strengths_dipvel[:nroots]
    # print('ref, res')
    # for refval, resval in zip(ref_etrotstrvel, res_etrotstrvel):
    #     print(refval, resval)
    # thresh = 1.0e-2
    # for i in range(nroots):
    #     abs_diff = abs(ref_etrotstrvel[i] - res_etrotstrvel[i])
    #     assert abs_diff < thresh

    # print(ecd_dipvel_tda.print_results_nwchem())
    # print(ecd_dipvel_tda.print_results_orca())
    # print(ecd_dipvel_tda.print_results_qchem())


def test_ECD_RPA_singlet_BC2H4_cation_HF_STO3G() -> None:
    ref = BC2H4_cation_HF_STO3G_RPA_singlet_nwchem
    nroots = ref["nroots"]

    mol = molecules.molecule_bc2h4_cation_sto3g()
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.scf()

    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = occupations_from_pyscf_mol(mol, C)

    ecd_dipvel_rpa = ecd.ECD(
        Program.PySCF,
        mol,
        td.TDHF(solvers.ExactDiagonalizationSolver(C, E, occupations)),
        C,
        E,
        occupations,
        do_dipvel=True,
    )
    ecd_dipvel_rpa.form_operators()
    ecd_dipvel_rpa.run(hamiltonian=Hamiltonian.RPA, spin=Spin.singlet)
    ecd_dipvel_rpa.form_results()

    print("excitation energies")
    ref_etenergies = np.array(ref["etenergies"])
    res_etenergies = ecd_dipvel_rpa.driver.solver.eigvals.real[:nroots]
    print("ref, res")
    for refval, resval in zip(ref_etenergies, res_etenergies):
        print(refval, resval)
    thresh = 2.5e-7
    for i in range(nroots):
        abs_diff = abs(ref_etenergies[i] - res_etenergies[i])
        assert abs_diff < thresh

    print("dipole (length) oscillator strengths")
    ref_etoscslen = np.array(ref["etoscslen"])
    res_etoscslen = ecd_dipvel_rpa.driver.solver.operators[1].total_oscillator_strengths[:nroots]
    print("ref, res")
    for refval, resval in zip(ref_etoscslen, res_etoscslen):
        print(refval, resval)
    thresh = 1.0e-5
    for i in range(nroots):
        abs_diff = abs(ref_etoscslen[i] - res_etoscslen[i])
        assert abs_diff < thresh

    # TODO
    print("TODO dipole (mixed length/velocity) oscillator strengths")

    # TODO
    print("TODO dipole (velocity) oscillator strengths")
    ref_etoscsvel = np.array(ref["etoscsvel"])  # noqa: F841
    res_etoscsvel = ecd_dipvel_rpa.driver.solver.operators[2].total_oscillator_strengths[:nroots]  # noqa: F841
    # print('ref, res')
    # for refval, resval in zip(ref_etoscsvel, res_etoscsvel):
    #     print(refval, resval)
    # print(ref_etoscsvel / res_etoscsvel)
    # print(res_etoscsvel / ref_etoscsvel)

    print("rotatory strengths (length)")
    ref_etrotstrlen = np.array(ref["etrotstrlen"])
    res_etrotstrlen = ecd_dipvel_rpa.rotational_strengths_diplen[:nroots]
    print("ref, res")
    for refval, resval in zip(ref_etrotstrlen, res_etrotstrlen):
        print(refval, resval)
    # TODO unlike other quantities, the error isn't uniformly
    # distributed among the roots; how should this be handled?
    thresh = 1.5e1
    for i in range(nroots):
        abs_diff = abs(ref_etrotstrlen[i] - res_etrotstrlen[i])
        assert abs_diff < thresh

    print("rotatory strengths (velocity)")
    ref_etrotstrvel = np.array(ref["etrotstrvel"])
    res_etrotstrvel = ecd_dipvel_rpa.rotational_strengths_dipvel[:nroots]
    print("ref, res")
    for refval, resval in zip(ref_etrotstrvel, res_etrotstrvel):
        print(refval, resval)
    thresh = 1.0e-2
    for i in range(nroots):
        abs_diff = abs(ref_etrotstrvel[i] - res_etrotstrvel[i])
        assert abs_diff < thresh

    # with open(os.path.join(refdir, 'BC2H4_cation', 'nwchem_singlet_rpa_velocity_root.str')) as fh:
    #     ref_str = fh.read()
    # res_str = ecd_dipvel_rpa.print_results_nwchem()
    # assert res_str == ref_str
    # print(ecd_dipvel_rpa.print_results_nwchem())
    # print(ecd_dipvel_rpa.print_results_orca())
    # print(ecd_dipvel_rpa.print_results_qchem())

    # tmom_diplen = ecd_dipvel_rpa.driver.solver.operators[1].transition_moments
    # tmom_dipvel = ecd_dipvel_rpa.driver.solver.operators[2].transition_moments
    # print(tmom_diplen)
    # print('dipole')
    # for i in range(nroots):
    #     print((2 / 3) * res_etenergies[i] * np.dot(tmom_diplen[i], tmom_diplen[i]))
    # print('mixed')
    # for i in range(nroots):
    #     print((2 / 3) * res_etenergies[i] * np.dot(tmom_diplen[i], tmom_dipvel[i]))

    # print('sum rule')
    # print('ref_etoscslen:', sum(ref_etoscslen))
    # print('res_etoscslen:', sum(res_etoscslen))
    # # print('ref_etoscsmix:', sum(ref_etoscsmix))
    # # print('res_etoscsmix:', sum(res_etoscsmix))
    # print('ref_etoscsvel:', sum(ref_etoscsvel))
    # print('res_etoscsvel:', sum(res_etoscsvel))


# TODO once UHF is done
# def test_ECD_RPA_singlet_BC2H4_neutral_radical_HF_STO3G():
#     mol = molecules.molecule_bc2h4_neutral_radical_sto3g()
#     mol.build()

#     mf = pyscf.scf.UHF(mol)
#     mf.scf()

#     C = utils.fix_mocoeffs_shape(mf.mo_coeff)


if __name__ == "__main__":
    test_ECD_TDA_singlet_BC2H4_cation_HF_STO3G()
    test_ECD_RPA_singlet_BC2H4_cation_HF_STO3G()
