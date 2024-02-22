import pyscf

from pymolresponse import solvers, td, utils
from pymolresponse.core import AO2MOTransformationType, Hamiltonian, Program, Spin
from pymolresponse.interfaces.pyscf.ao2mo import AO2MOpyscf
from pymolresponse.interfaces.pyscf.utils import occupations_from_pyscf_mol


def test_HF_both_singlet_HF_STO3G():
    mol = pyscf.gto.Mole()
    mol.verbose = 0
    mol.output = None

    # pylint: disable=bad-whitespace
    mol.atom = [["H", (0.0, 0.0, 0.917)], ["F", (0.0, 0.0, 0.0)]]
    mol.basis = "sto-3g"
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.scf()

    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = occupations_from_pyscf_mol(mol, C)
    solver_tda = solvers.ExactDiagonalizationSolverTDA(C, E, occupations)
    solver_tdhf = solvers.ExactDiagonalizationSolver(C, E, occupations)
    ao2mo = AO2MOpyscf(C, mol.verbose, mol)
    ao2mo.perform_rhf_partial()
    tei_mo = ao2mo.tei_mo
    solver_tda.tei_mo = tei_mo
    solver_tda.tei_mo_type = AO2MOTransformationType.partial
    solver_tdhf.tei_mo = tei_mo
    solver_tdhf.tei_mo_type = AO2MOTransformationType.partial
    driver_tda = td.TDA(solver_tda)
    driver_tdhf = td.TDHF(solver_tdhf)

    nroots = 5

    print("TDA using TDA()")
    driver_tda.run(
        hamiltonian=Hamiltonian.TDA, spin=Spin.singlet, program=Program.PySCF, program_obj=mol
    )
    excitation_energies_tda_using_tda = driver_tda.solver.eigvals[:nroots].real
    print("TDA using TDHF()")
    driver_tdhf.run(
        hamiltonian=Hamiltonian.TDA, spin=Spin.singlet, program=Program.PySCF, program_obj=mol
    )
    excitation_energies_tda_using_tdhf = driver_tdhf.solver.eigvals[:nroots].real
    print("RPA using TDHF()")
    driver_tdhf.run(
        hamiltonian=Hamiltonian.RPA, spin=Spin.singlet, program=Program.PySCF, program_obj=mol
    )
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
        abs_diff = abs(ref_tda["etenergies"][i] - excitation_energies_tda_using_tda[i])
        assert abs_diff < thresh

    thresh = 1.0e-7
    for i in range(nroots):
        abs_diff = abs(ref_rpa["etenergies"][i] - excitation_energies_rpa[i])
        assert abs_diff < thresh


HF_neutral_singlet_HF_STO3G_qchem = -98.5707799863

HF_neutral_singlet_HF_STO3G_CIS_qchem = {
    "etenergies": [
        -98.067181246814 - HF_neutral_singlet_HF_STO3G_qchem,
        -98.067181246768 - HF_neutral_singlet_HF_STO3G_qchem,
        -97.686596454655 - HF_neutral_singlet_HF_STO3G_qchem,
        -96.958042326818 - HF_neutral_singlet_HF_STO3G_qchem,
        -72.879307887356 - HF_neutral_singlet_HF_STO3G_qchem,
    ],
    "etoscslen": [0.0002003489, 0.0002003489, 0.9621809426, 0.0531137481, 0.0691994928],
}


HF_neutral_singlet_HF_STO3G_RPA_qchem = {
    "etenergies": [
        -98.068185050585 - HF_neutral_singlet_HF_STO3G_qchem,
        -98.068185050538 - HF_neutral_singlet_HF_STO3G_qchem,
        -97.703584999956 - HF_neutral_singlet_HF_STO3G_qchem,
        -96.962988495302 - HF_neutral_singlet_HF_STO3G_qchem,
        -72.879331844690 - HF_neutral_singlet_HF_STO3G_qchem,
    ],
    "etoscslen": [0.0001877054, 0.0001877054, 0.7777380206, 0.0322221420, 0.0686085799],
}

# TODO
#
# def test_LiH_cation_TDA_singlet_HF_STO3G() -> None:
#     from pymolresponse.interfaces.pyscf.molecules import molecule_lih_cation_sto3g

#     mol = molecule_lih_cation_sto3g()
#     mf = pyscf.scf.UHF(mol)
#     mf.scf()
#     C = utils.fix_mocoeffs_shape(mf.mo_coeff)
#     E = utils.fix_moenergies_shape(mf.mo_energy)
#     occupations = occupations_from_pyscf_mol(mol, C)
#     solver_tda = solvers.ExactDiagonalizationSolverTDA(C, E, occupations)
#     # TODO i thought this was part of the solver interface
#     ao2mo = AO2MOpyscf(C, mol.verbose, mol)
#     ao2mo.perform_uhf_partial()
#     tei_mo = ao2mo.tei_mo
#     solver_tda.tei_mo = tei_mo
#     driver_tda = td.TDA(solver_tda)
#     print("TDA using TDA()")
#     driver_tda.run(
#         hamiltonian=Hamiltonian.TDA, spin=Spin.singlet, program=Program.PySCF, program_obj=mol
#     )
#     print(driver_tda.solver.eigvals.real)


if __name__ == "__main__":
    test_HF_both_singlet_HF_STO3G()
