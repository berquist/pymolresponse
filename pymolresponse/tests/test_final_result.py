import numpy as np

from pymolresponse import cphf, operators, solvers, utils
from pymolresponse.core import AO2MOTransformationType, Hamiltonian, Program, Spin
from pymolresponse.data import REFDIR
from pymolresponse.interfaces.pyscf import molecules
from pymolresponse.properties import electric
from pymolresponse.tests.test_runners import (
    run_as_many_tests_as_possible_rhf_disk,
    run_as_many_tests_as_possible_uhf_disk,
)


def test_final_result_rhf_h2o_sto3g_rpa_singlet() -> None:
    """Test correctness of the final result for water/STO-3G with full RPA for
    singlet response induced by the dipole length operator (the electric
    polarizability) computed with quantities from disk.
    """
    hamiltonian = Hamiltonian.RPA
    spin = Spin.singlet

    C = utils.fix_mocoeffs_shape(utils.np_load(REFDIR / "C.npz"))
    E = utils.fix_moenergies_shape(utils.np_load(REFDIR / "F_MO.npz"))
    TEI_MO = utils.np_load(REFDIR / "TEI_MO.npz")
    # nocc_alph, nvirt_alph, nocc_beta, nvirt_beta
    occupations = np.asarray([5, 2, 5, 2], dtype=int)
    stub = "h2o_sto3g_"
    dim = occupations[0] + occupations[1]
    mat_dipole_x = utils.parse_int_file_2(REFDIR / f"{stub}mux.dat", dim)
    mat_dipole_y = utils.parse_int_file_2(REFDIR / f"{stub}muy.dat", dim)
    mat_dipole_z = utils.parse_int_file_2(REFDIR / f"{stub}muz.dat", dim)

    solver = solvers.ExactInv(C, E, occupations)
    solver.tei_mo = (TEI_MO,)
    solver.tei_mo_type = AO2MOTransformationType.full
    driver = cphf.CPHF(solver)
    ao_integrals_dipole = np.stack((mat_dipole_x, mat_dipole_y, mat_dipole_z), axis=0)
    operator_dipole = operators.Operator(
        label="dipole", is_imaginary=False, is_spin_dependent=False
    )
    operator_dipole.ao_integrals = ao_integrals_dipole
    driver.add_operator(operator_dipole)
    frequencies = (0.0, 0.02, 0.06, 0.1)
    driver.set_frequencies(frequencies)

    driver.run(hamiltonian=hamiltonian, spin=spin, program=None, program_obj=None)

    assert len(driver.results) == len(frequencies)

    result__0_00 = np.array(
        [[7.93556221, 0.0, 0.0], [0.0, 3.06821077, 0.0], [0.0, 0.0, 0.05038621]]
    )
    result__0_02 = np.array(
        [[7.94312371, 0.0, 0.0], [0.0, 3.07051688, 0.0], [0.0, 0.0, 0.05054685]]
    )
    result__0_06 = np.array(
        [[8.00414009, 0.0, 0.0], [0.0, 3.08913608, 0.0], [0.0, 0.0, 0.05186977]]
    )
    result__0_10 = np.array([[8.1290378, 0.0, 0.0], [0.0, 3.12731363, 0.0], [0.0, 0.0, 0.05473482]])

    atol = 1.0e-8
    rtol = 0.0
    np.testing.assert_allclose(driver.results[0], result__0_00, rtol=rtol, atol=atol)
    np.testing.assert_allclose(driver.results[1], result__0_02, rtol=rtol, atol=atol)
    np.testing.assert_allclose(driver.results[2], result__0_06, rtol=rtol, atol=atol)
    np.testing.assert_allclose(driver.results[3], result__0_10, rtol=rtol, atol=atol)

    # Reminder: there's no call to do SCF here because we already have
    # the MO coefficients.
    mol = molecules.molecule_water_sto3g()
    mol.build()
    polarizability = electric.Polarizability(
        Program.PySCF,
        mol,
        cphf.CPHF(solvers.ExactInv(C, E, occupations)),
        C,
        E,
        occupations,
        frequencies=frequencies,
    )
    polarizability.form_operators()
    polarizability.run(hamiltonian=hamiltonian, spin=spin)
    polarizability.form_results()

    np.testing.assert_allclose(
        polarizability.polarizabilities[0], result__0_00, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        polarizability.polarizabilities[1], result__0_02, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        polarizability.polarizabilities[2], result__0_06, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        polarizability.polarizabilities[3], result__0_10, rtol=rtol, atol=atol
    )


def test_final_result_rhf_h2o_sto3g_rpa_triplet() -> None:
    """Test correctness of the final result for water/STO-3G with full RPA for
    triplet response induced by the dipole length operator computed with
    quantities from disk.
    """
    hamiltonian = Hamiltonian.RPA
    spin = Spin.triplet

    C = utils.fix_mocoeffs_shape(utils.np_load(REFDIR / "C.npz"))
    E = utils.fix_moenergies_shape(utils.np_load(REFDIR / "F_MO.npz"))
    TEI_MO = utils.np_load(REFDIR / "TEI_MO.npz")
    # nocc_alph, nvirt_alph, nocc_beta, nvirt_beta
    occupations = np.asarray([5, 2, 5, 2], dtype=int)
    stub = "h2o_sto3g_"
    dim = occupations[0] + occupations[1]
    mat_dipole_x = utils.parse_int_file_2(REFDIR / f"{stub}mux.dat", dim)
    mat_dipole_y = utils.parse_int_file_2(REFDIR / f"{stub}muy.dat", dim)
    mat_dipole_z = utils.parse_int_file_2(REFDIR / f"{stub}muz.dat", dim)

    solver = solvers.ExactInv(C, E, occupations)
    solver.tei_mo = (TEI_MO,)
    solver.tei_mo_type = AO2MOTransformationType.full
    driver = cphf.CPHF(solver)
    ao_integrals_dipole = np.stack((mat_dipole_x, mat_dipole_y, mat_dipole_z), axis=0)
    operator_dipole = operators.Operator(
        label="dipole", is_imaginary=False, is_spin_dependent=False
    )
    operator_dipole.ao_integrals = ao_integrals_dipole
    driver.add_operator(operator_dipole)
    frequencies = (0.0, 0.02, 0.06, 0.1)
    driver.set_frequencies(frequencies)

    driver.run(hamiltonian=hamiltonian, spin=spin, program=None, program_obj=None)

    assert len(driver.results) == len(frequencies)

    result__0_00 = np.array(
        [[26.59744305, 0.0, 0.0], [0.0, 18.11879557, 0.0], [0.0, 0.0, 0.07798969]]
    )
    result__0_02 = np.array(
        [[26.68282287, 0.0, 0.0], [0.0, 18.19390051, 0.0], [0.0, 0.0, 0.07837521]]
    )
    result__0_06 = np.array(
        [[27.38617401, 0.0, 0.0], [0.0, 18.81922578, 0.0], [0.0, 0.0, 0.08160226]]
    )
    result__0_10 = np.array(
        [[28.91067234, 0.0, 0.0], [0.0, 20.21670386, 0.0], [0.0, 0.0, 0.08892512]]
    )

    atol = 1.0e-8
    rtol = 0.0
    np.testing.assert_allclose(driver.results[0], result__0_00, rtol=rtol, atol=atol)
    np.testing.assert_allclose(driver.results[1], result__0_02, rtol=rtol, atol=atol)
    np.testing.assert_allclose(driver.results[2], result__0_06, rtol=rtol, atol=atol)
    np.testing.assert_allclose(driver.results[3], result__0_10, rtol=rtol, atol=atol)

    mol = molecules.molecule_water_sto3g()
    mol.build()
    polarizability = electric.Polarizability(
        Program.PySCF,
        mol,
        cphf.CPHF(solvers.ExactInv(C, E, occupations)),
        C,
        E,
        occupations,
        frequencies=frequencies,
    )
    polarizability.form_operators()
    polarizability.run(hamiltonian=hamiltonian, spin=spin)
    polarizability.form_results()

    np.testing.assert_allclose(
        polarizability.polarizabilities[0], result__0_00, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        polarizability.polarizabilities[1], result__0_02, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        polarizability.polarizabilities[2], result__0_06, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        polarizability.polarizabilities[3], result__0_10, rtol=rtol, atol=atol
    )


def test_final_result_rhf_h2o_sto3g_tda_singlet() -> None:
    """Test correctness of the final result for water/STO-3G with the TDA
    approximation/CIS for singlet response induced by the dipole length
    operator computed with quantities from disk.
    """
    hamiltonian = Hamiltonian.TDA
    spin = Spin.singlet

    C = utils.fix_mocoeffs_shape(utils.np_load(REFDIR / "C.npz"))
    E = utils.fix_moenergies_shape(utils.np_load(REFDIR / "F_MO.npz"))
    TEI_MO = utils.np_load(REFDIR / "TEI_MO.npz")
    # nocc_alph, nvirt_alph, nocc_beta, nvirt_beta
    occupations = np.asarray([5, 2, 5, 2], dtype=int)
    stub = "h2o_sto3g_"
    dim = occupations[0] + occupations[1]
    mat_dipole_x = utils.parse_int_file_2(REFDIR / f"{stub}mux.dat", dim)
    mat_dipole_y = utils.parse_int_file_2(REFDIR / f"{stub}muy.dat", dim)
    mat_dipole_z = utils.parse_int_file_2(REFDIR / f"{stub}muz.dat", dim)

    solver = solvers.ExactInv(C, E, occupations)
    solver.tei_mo = (TEI_MO,)
    solver.tei_mo_type = AO2MOTransformationType.full
    driver = cphf.CPHF(solver)
    ao_integrals_dipole = np.stack((mat_dipole_x, mat_dipole_y, mat_dipole_z), axis=0)
    operator_dipole = operators.Operator(
        label="dipole", is_imaginary=False, is_spin_dependent=False
    )
    operator_dipole.ao_integrals = ao_integrals_dipole
    driver.add_operator(operator_dipole)
    frequencies = (0.0, 0.02, 0.06, 0.1)
    driver.set_frequencies(frequencies)

    driver.run(hamiltonian=hamiltonian, spin=spin, program=None, program_obj=None)

    assert len(driver.results) == len(frequencies)

    result__0_00 = np.array([[8.89855952, 0.0, 0.0], [0.0, 4.00026556, 0.0], [0.0, 0.0, 0.0552774]])
    result__0_02 = np.array(
        [[8.90690928, 0.0, 0.0], [0.0, 4.00298342, 0.0], [0.0, 0.0, 0.05545196]]
    )
    result__0_06 = np.array(
        [[8.97427725, 0.0, 0.0], [0.0, 4.02491517, 0.0], [0.0, 0.0, 0.05688918]]
    )
    result__0_10 = np.array(
        [[9.11212633, 0.0, 0.0], [0.0, 4.06981937, 0.0], [0.0, 0.0, 0.05999934]]
    )

    atol = 1.0e-8
    rtol = 0.0
    np.testing.assert_allclose(driver.results[0], result__0_00, rtol=rtol, atol=atol)
    np.testing.assert_allclose(driver.results[1], result__0_02, rtol=rtol, atol=atol)
    np.testing.assert_allclose(driver.results[2], result__0_06, rtol=rtol, atol=atol)
    np.testing.assert_allclose(driver.results[3], result__0_10, rtol=rtol, atol=atol)

    mol = molecules.molecule_water_sto3g()
    mol.build()
    polarizability = electric.Polarizability(
        Program.PySCF,
        mol,
        cphf.CPHF(solvers.ExactInv(C, E, occupations)),
        C,
        E,
        occupations,
        frequencies=frequencies,
    )
    polarizability.form_operators()
    polarizability.run(hamiltonian=hamiltonian, spin=spin)
    polarizability.form_results()

    np.testing.assert_allclose(
        polarizability.polarizabilities[0], result__0_00, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        polarizability.polarizabilities[1], result__0_02, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        polarizability.polarizabilities[2], result__0_06, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        polarizability.polarizabilities[3], result__0_10, rtol=rtol, atol=atol
    )


def test_final_result_rhf_h2o_sto3g_tda_triplet() -> None:
    """Test correctness of the final result for water/STO-3G with the TDA
    approximation/CIS for triplet response induced by the dipole length
    operator computed with quantities from disk.
    """
    hamiltonian = Hamiltonian.TDA
    spin = Spin.triplet

    C = utils.fix_mocoeffs_shape(utils.np_load(REFDIR / "C.npz"))
    E = utils.fix_moenergies_shape(utils.np_load(REFDIR / "F_MO.npz"))
    TEI_MO = utils.np_load(REFDIR / "TEI_MO.npz")
    # nocc_alph, nvirt_alph, nocc_beta, nvirt_beta
    occupations = np.asarray([5, 2, 5, 2], dtype=int)
    stub = "h2o_sto3g_"
    dim = occupations[0] + occupations[1]
    mat_dipole_x = utils.parse_int_file_2(REFDIR / f"{stub}mux.dat", dim)
    mat_dipole_y = utils.parse_int_file_2(REFDIR / f"{stub}muy.dat", dim)
    mat_dipole_z = utils.parse_int_file_2(REFDIR / f"{stub}muz.dat", dim)

    solver = solvers.ExactInv(C, E, occupations)
    solver.tei_mo = (TEI_MO,)
    solver.tei_mo_type = AO2MOTransformationType.full
    driver = cphf.CPHF(solver)
    ao_integrals_dipole = np.stack((mat_dipole_x, mat_dipole_y, mat_dipole_z), axis=0)
    operator_dipole = operators.Operator(
        label="dipole", is_imaginary=False, is_spin_dependent=False
    )
    operator_dipole.ao_integrals = ao_integrals_dipole
    driver.add_operator(operator_dipole)
    frequencies = (0.0, 0.02, 0.06, 0.1)
    driver.set_frequencies(frequencies)

    driver.run(hamiltonian=hamiltonian, spin=spin, program=None, program_obj=None)

    assert len(driver.results) == len(frequencies)

    result__0_00 = np.array(
        [[14.64430714, 0.0, 0.0], [0.0, 8.80921432, 0.0], [0.0, 0.0, 0.06859496]]
    )
    result__0_02 = np.array(
        [[14.68168443, 0.0, 0.0], [0.0, 8.83562647, 0.0], [0.0, 0.0, 0.0689291]]
    )
    result__0_06 = np.array(
        [[14.98774296, 0.0, 0.0], [0.0, 9.0532224, 0.0], [0.0, 0.0, 0.07172414]]
    )
    result__0_10 = np.array(
        [[15.63997724, 0.0, 0.0], [0.0, 9.52504267, 0.0], [0.0, 0.0, 0.07805428]]
    )

    atol = 1.0e-8
    rtol = 0.0
    np.testing.assert_allclose(driver.results[0], result__0_00, rtol=rtol, atol=atol)
    np.testing.assert_allclose(driver.results[1], result__0_02, rtol=rtol, atol=atol)
    np.testing.assert_allclose(driver.results[2], result__0_06, rtol=rtol, atol=atol)
    np.testing.assert_allclose(driver.results[3], result__0_10, rtol=rtol, atol=atol)

    mol = molecules.molecule_water_sto3g()
    mol.build()
    polarizability = electric.Polarizability(
        Program.PySCF,
        mol,
        cphf.CPHF(solvers.ExactInv(C, E, occupations)),
        C,
        E,
        occupations,
        frequencies=frequencies,
    )
    polarizability.form_operators()
    polarizability.run(hamiltonian=hamiltonian, spin=spin)
    polarizability.form_results()

    np.testing.assert_allclose(
        polarizability.polarizabilities[0], result__0_00, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        polarizability.polarizabilities[1], result__0_02, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        polarizability.polarizabilities[2], result__0_06, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        polarizability.polarizabilities[3], result__0_10, rtol=rtol, atol=atol
    )


def test_as_many_as_possible_rhf_disk() -> None:
    """Test correctness of the final result for closed-shell molecules against
    DALTON references.
    """
    run_as_many_tests_as_possible_rhf_disk("r_lih_hf_sto-3g")


def test_as_many_as_possible_uhf_disk() -> None:
    """Test correctness of the final result for open-shell (UHF) molecules
    against DALTON (ROHF) references.
    """
    run_as_many_tests_as_possible_uhf_disk("u_lih_cation_hf_sto-3g")


if __name__ == "__main__":
    test_final_result_rhf_h2o_sto3g_rpa_singlet()
    test_final_result_rhf_h2o_sto3g_rpa_triplet()
    test_final_result_rhf_h2o_sto3g_tda_singlet()
    test_final_result_rhf_h2o_sto3g_tda_triplet()
    test_as_many_as_possible_rhf_disk()
    test_as_many_as_possible_uhf_disk()
