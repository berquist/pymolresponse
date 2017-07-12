import pyscf

from pyresponse import utils, iterators, td
from pyresponse.ao2mo import AO2MOpyscf


def test_HF_both_singlet_HF_STO3G():
    mol = pyscf.gto.Mole()
    mol.verbose = 0
    mol.output = None

    # pylint: disable=bad-whitespace
    mol.atom = [
        ['H' , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)],
    ]
    mol.basis = 'sto-3g'
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.scf()

    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = utils.occupations_from_pyscf_mol(mol, C)
    solver_tda = iterators.ExactDiagonalizationSolverTDA(C, E, occupations)
    solver_tdhf = iterators.ExactDiagonalizationSolver(C, E, occupations)
    ao2mo = AO2MOpyscf(C, mol.verbose, mol)
    ao2mo.perform_rhf_partial()
    tei_mo = ao2mo.tei_mo
    solver_tda.tei_mo = tei_mo
    solver_tda.tei_mo_type = 'partial'
    solver_tdhf.tei_mo = tei_mo
    solver_tdhf.tei_mo_type = 'partial'
    driver_tda = td.TDA(solver_tda)
    driver_tdhf = td.TDHF(solver_tdhf)

    nroots = 5

    print('TDA using TDA()')
    driver_tda.run(solver_type='exact', hamiltonian='tda', spin='singlet')
    excitation_energies_tda_using_tda = driver_tda.solver.eigvals[:nroots].real
    print('TDA using TDHF()')
    driver_tdhf.run(solver_type='exact', hamiltonian='tda', spin='singlet')
    excitation_energies_tda_using_tdhf = driver_tdhf.solver.eigvals[:nroots].real
    print('RPA using TDHF()')
    driver_tdhf.run(solver_type='exact', hamiltonian='rpa', spin='singlet')
    excitation_energies_rpa = driver_tdhf.solver.eigvals[:nroots].real

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

if __name__ == '__main__':
    test_HF_both_singlet_HF_STO3G()
