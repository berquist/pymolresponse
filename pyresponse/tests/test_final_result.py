import numpy as np

from pyresponse import cphf, electric, iterators, operators, utils
from pyresponse.data import REFDIR
from pyresponse.pyscf import molecules
from pyresponse.tests.test_runners import (
    run_as_many_tests_as_possible_rhf_disk,
    run_as_many_tests_as_possible_uhf_disk
)


def test_final_result_rhf_h2o_sto3g_rpa_singlet():
    hamiltonian = 'rpa'
    spin = 'singlet'

    C = utils.fix_mocoeffs_shape(utils.np_load(REFDIR / "C.npz"))
    E = utils.fix_moenergies_shape(utils.np_load(REFDIR / "F_MO.npz"))
    TEI_MO = utils.np_load(REFDIR / "TEI_MO.npz")
    # nocc_alph, nvirt_alph, nocc_beta, nvirt_beta
    occupations = [5, 2, 5, 2]
    stub = 'h2o_sto3g_'
    dim = occupations[0] + occupations[1]
    mat_dipole_x = utils.parse_int_file_2(REFDIR / f"{stub}mux.dat", dim)
    mat_dipole_y = utils.parse_int_file_2(REFDIR / f"{stub}muy.dat", dim)
    mat_dipole_z = utils.parse_int_file_2(REFDIR / f"{stub}muz.dat", dim)

    solver = iterators.ExactInv(C, E, occupations)
    solver.tei_mo = (TEI_MO, )
    solver.tei_mo_type = 'full'
    driver = cphf.CPHF(solver)
    ao_integrals_dipole = np.stack((mat_dipole_x, mat_dipole_y, mat_dipole_z), axis=0)
    operator_dipole = operators.Operator(label='dipole',
                                         is_imaginary=False, is_spin_dependent=False)
    operator_dipole.ao_integrals = ao_integrals_dipole
    driver.add_operator(operator_dipole)
    frequencies = (0.0, 0.02, 0.06, 0.1)
    driver.set_frequencies(frequencies)

    driver.run(solver_type='exact', hamiltonian=hamiltonian, spin=spin)

    assert len(driver.results) == len(frequencies)

    # pylint: disable=bad-whitespace
    result__0_00 = np.array([[ 7.93556221,  0.,          0.        ],
                             [ 0.,          3.06821077,  0.        ],
                             [ 0.,          0.,          0.05038621]])

    # pylint: disable=bad-whitespace
    result__0_02 = np.array([[ 7.94312371,  0.,          0.        ],
                             [ 0.,          3.07051688,  0.        ],
                             [ 0.,          0.,          0.05054685]])

    # pylint: disable=bad-whitespace
    result__0_06 = np.array([[ 8.00414009,  0.,          0.        ],
                             [ 0.,          3.08913608,  0.        ],
                             [ 0.,          0.,          0.05186977]])

    # pylint: disable=bad-whitespace
    result__0_10 = np.array([[ 8.1290378,   0.,          0.        ],
                             [ 0.,          3.12731363,  0.        ],
                             [ 0.,          0.,          0.05473482]])

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
    polarizability = electric.Polarizability(mol, C, E, occupations, frequencies)
    polarizability.form_operators()
    polarizability.run(hamiltonian=hamiltonian, spin=spin)
    polarizability.form_results()

    np.testing.assert_allclose(polarizability.polarizabilities[0],
                               result__0_00, rtol=rtol, atol=atol)
    np.testing.assert_allclose(polarizability.polarizabilities[1],
                               result__0_02, rtol=rtol, atol=atol)
    np.testing.assert_allclose(polarizability.polarizabilities[2],
                               result__0_06, rtol=rtol, atol=atol)
    np.testing.assert_allclose(polarizability.polarizabilities[3],
                               result__0_10, rtol=rtol, atol=atol)

    return


def test_final_result_rhf_h2o_sto3g_rpa_triplet():
    hamiltonian = 'rpa'
    spin = 'triplet'

    C = utils.fix_mocoeffs_shape(utils.np_load(REFDIR / "C.npz"))
    E = utils.fix_moenergies_shape(utils.np_load(REFDIR / "F_MO.npz"))
    TEI_MO = utils.np_load(REFDIR / "TEI_MO.npz")
    # nocc_alph, nvirt_alph, nocc_beta, nvirt_beta
    occupations = [5, 2, 5, 2]
    stub = 'h2o_sto3g_'
    dim = occupations[0] + occupations[1]
    mat_dipole_x = utils.parse_int_file_2(REFDIR / f"{stub}mux.dat", dim)
    mat_dipole_y = utils.parse_int_file_2(REFDIR / f"{stub}muy.dat", dim)
    mat_dipole_z = utils.parse_int_file_2(REFDIR / f"{stub}muz.dat", dim)

    solver = iterators.ExactInv(C, E, occupations)
    solver.tei_mo = (TEI_MO, )
    solver.tei_mo_type = 'full'
    driver = cphf.CPHF(solver)
    ao_integrals_dipole = np.stack((mat_dipole_x, mat_dipole_y, mat_dipole_z), axis=0)
    operator_dipole = operators.Operator(label='dipole',
                                         is_imaginary=False, is_spin_dependent=False)
    operator_dipole.ao_integrals = ao_integrals_dipole
    driver.add_operator(operator_dipole)
    frequencies = (0.0, 0.02, 0.06, 0.1)
    driver.set_frequencies(frequencies)

    driver.run(solver_type='exact', hamiltonian=hamiltonian, spin=spin)

    assert len(driver.results) == len(frequencies)

    # pylint: disable=bad-whitespace
    result__0_00 = np.array([[ 26.59744305,   0.,           0.        ],
                             [  0.,          18.11879557,   0.        ],
                             [  0.,           0.,           0.07798969]])

    # pylint: disable=bad-whitespace
    result__0_02 = np.array([[ 26.68282287,   0.,           0.        ],
                             [  0.,          18.19390051,   0.        ],
                             [  0.,           0.,           0.07837521]])

    # pylint: disable=bad-whitespace
    result__0_06 = np.array([[ 27.38617401,   0.,           0.        ],
                             [  0.,          18.81922578,   0.        ],
                             [  0.,           0.,           0.08160226]])

    # pylint: disable=bad-whitespace
    result__0_10 = np.array([[ 28.91067234,   0.,           0.        ],
                             [  0.,          20.21670386,   0.        ],
                             [  0.,           0.,           0.08892512]])

    atol = 1.0e-8
    rtol = 0.0
    np.testing.assert_allclose(driver.results[0], result__0_00, rtol=rtol, atol=atol)
    np.testing.assert_allclose(driver.results[1], result__0_02, rtol=rtol, atol=atol)
    np.testing.assert_allclose(driver.results[2], result__0_06, rtol=rtol, atol=atol)
    np.testing.assert_allclose(driver.results[3], result__0_10, rtol=rtol, atol=atol)

    mol = molecules.molecule_water_sto3g()
    mol.build()
    polarizability = electric.Polarizability(mol, C, E, occupations, frequencies)
    polarizability.form_operators()
    polarizability.run(hamiltonian=hamiltonian, spin=spin)
    polarizability.form_results()

    np.testing.assert_allclose(polarizability.polarizabilities[0],
                               result__0_00, rtol=rtol, atol=atol)
    np.testing.assert_allclose(polarizability.polarizabilities[1],
                               result__0_02, rtol=rtol, atol=atol)
    np.testing.assert_allclose(polarizability.polarizabilities[2],
                               result__0_06, rtol=rtol, atol=atol)
    np.testing.assert_allclose(polarizability.polarizabilities[3],
                               result__0_10, rtol=rtol, atol=atol)

    return


def test_final_result_rhf_h2o_sto3g_tda_singlet():
    hamiltonian = 'tda'
    spin = 'singlet'

    C = utils.fix_mocoeffs_shape(utils.np_load(REFDIR / "C.npz"))
    E = utils.fix_moenergies_shape(utils.np_load(REFDIR / "F_MO.npz"))
    TEI_MO = utils.np_load(REFDIR / "TEI_MO.npz")
    # nocc_alph, nvirt_alph, nocc_beta, nvirt_beta
    occupations = [5, 2, 5, 2]
    stub = 'h2o_sto3g_'
    dim = occupations[0] + occupations[1]
    mat_dipole_x = utils.parse_int_file_2(REFDIR / f"{stub}mux.dat", dim)
    mat_dipole_y = utils.parse_int_file_2(REFDIR / f"{stub}muy.dat", dim)
    mat_dipole_z = utils.parse_int_file_2(REFDIR / f"{stub}muz.dat", dim)

    solver = iterators.ExactInv(C, E, occupations)
    solver.tei_mo = (TEI_MO, )
    solver.tei_mo_type = 'full'
    driver = cphf.CPHF(solver)
    ao_integrals_dipole = np.stack((mat_dipole_x, mat_dipole_y, mat_dipole_z), axis=0)
    operator_dipole = operators.Operator(label='dipole',
                                         is_imaginary=False, is_spin_dependent=False)
    operator_dipole.ao_integrals = ao_integrals_dipole
    driver.add_operator(operator_dipole)
    frequencies = (0.0, 0.02, 0.06, 0.1)
    driver.set_frequencies(frequencies)

    driver.run(solver_type='exact', hamiltonian=hamiltonian, spin=spin)

    assert len(driver.results) == len(frequencies)

    # pylint: disable=bad-whitespace
    result__0_00 = np.array([[ 8.89855952,  0.,          0.        ],
                             [ 0.,          4.00026556,  0.        ],
                             [ 0.,          0.,          0.0552774 ]])

    # pylint: disable=bad-whitespace
    result__0_02 = np.array([[ 8.90690928,  0.,          0.        ],
                             [ 0.,          4.00298342,  0.        ],
                             [ 0.,          0.,          0.05545196]])

    # pylint: disable=bad-whitespace
    result__0_06 = np.array([[ 8.97427725,  0.,          0.        ],
                             [ 0.,          4.02491517,  0.        ],
                             [ 0.,          0.,          0.05688918]])

    # pylint: disable=bad-whitespace
    result__0_10 = np.array([[ 9.11212633,  0.,          0.        ],
                             [ 0.,          4.06981937,  0.        ],
                             [ 0.,          0.,          0.05999934]])

    atol = 1.0e-8
    rtol = 0.0
    np.testing.assert_allclose(driver.results[0], result__0_00, rtol=rtol, atol=atol)
    np.testing.assert_allclose(driver.results[1], result__0_02, rtol=rtol, atol=atol)
    np.testing.assert_allclose(driver.results[2], result__0_06, rtol=rtol, atol=atol)
    np.testing.assert_allclose(driver.results[3], result__0_10, rtol=rtol, atol=atol)

    mol = molecules.molecule_water_sto3g()
    mol.build()
    polarizability = electric.Polarizability(mol, C, E, occupations, frequencies)
    polarizability.form_operators()
    polarizability.run(hamiltonian=hamiltonian, spin=spin)
    polarizability.form_results()

    np.testing.assert_allclose(polarizability.polarizabilities[0],
                               result__0_00, rtol=rtol, atol=atol)
    np.testing.assert_allclose(polarizability.polarizabilities[1],
                               result__0_02, rtol=rtol, atol=atol)
    np.testing.assert_allclose(polarizability.polarizabilities[2],
                               result__0_06, rtol=rtol, atol=atol)
    np.testing.assert_allclose(polarizability.polarizabilities[3],
                               result__0_10, rtol=rtol, atol=atol)

    return


def test_final_result_rhf_h2o_sto3g_tda_triplet():
    hamiltonian = 'tda'
    spin = 'triplet'

    C = utils.fix_mocoeffs_shape(utils.np_load(REFDIR / "C.npz"))
    E = utils.fix_moenergies_shape(utils.np_load(REFDIR / "F_MO.npz"))
    TEI_MO = utils.np_load(REFDIR / "TEI_MO.npz")
    # nocc_alph, nvirt_alph, nocc_beta, nvirt_beta
    occupations = [5, 2, 5, 2]
    stub = 'h2o_sto3g_'
    dim = occupations[0] + occupations[1]
    mat_dipole_x = utils.parse_int_file_2(REFDIR / f"{stub}mux.dat", dim)
    mat_dipole_y = utils.parse_int_file_2(REFDIR / f"{stub}muy.dat", dim)
    mat_dipole_z = utils.parse_int_file_2(REFDIR / f"{stub}muz.dat", dim)

    solver = iterators.ExactInv(C, E, occupations)
    solver.tei_mo = (TEI_MO, )
    solver.tei_mo_type = 'full'
    driver = cphf.CPHF(solver)
    ao_integrals_dipole = np.stack((mat_dipole_x, mat_dipole_y, mat_dipole_z), axis=0)
    operator_dipole = operators.Operator(label='dipole',
                                         is_imaginary=False, is_spin_dependent=False)
    operator_dipole.ao_integrals = ao_integrals_dipole
    driver.add_operator(operator_dipole)
    frequencies = (0.0, 0.02, 0.06, 0.1)
    driver.set_frequencies(frequencies)

    driver.run(solver_type='exact', hamiltonian=hamiltonian, spin=spin)

    assert len(driver.results) == len(frequencies)

    # pylint: disable=bad-whitespace
    result__0_00 = np.array([[ 14.64430714,   0.,           0.        ],
                             [  0.,           8.80921432,   0.        ],
                             [  0.,           0.,           0.06859496]])

    # pylint: disable=bad-whitespace
    result__0_02 = np.array([[ 14.68168443,   0.,           0.        ],
                             [  0.,           8.83562647,   0.        ],
                             [  0.,           0.,           0.0689291 ]])

    # pylint: disable=bad-whitespace
    result__0_06 = np.array([[ 14.98774296,   0.,           0.        ],
                             [  0.,           9.0532224,    0.        ],
                             [  0.,           0.,           0.07172414]])

    # pylint: disable=bad-whitespace
    result__0_10 = np.array([[ 15.63997724,   0.,           0.        ],
                             [  0.,           9.52504267,   0.        ],
                             [  0.,           0.,           0.07805428]])

    atol = 1.0e-8
    rtol = 0.0
    np.testing.assert_allclose(driver.results[0], result__0_00, rtol=rtol, atol=atol)
    np.testing.assert_allclose(driver.results[1], result__0_02, rtol=rtol, atol=atol)
    np.testing.assert_allclose(driver.results[2], result__0_06, rtol=rtol, atol=atol)
    np.testing.assert_allclose(driver.results[3], result__0_10, rtol=rtol, atol=atol)

    mol = molecules.molecule_water_sto3g()
    mol.build()
    polarizability = electric.Polarizability(mol, C, E, occupations, frequencies)
    polarizability.form_operators()
    polarizability.run(hamiltonian=hamiltonian, spin=spin)
    polarizability.form_results()

    np.testing.assert_allclose(polarizability.polarizabilities[0],
                               result__0_00, rtol=rtol, atol=atol)
    np.testing.assert_allclose(polarizability.polarizabilities[1],
                               result__0_02, rtol=rtol, atol=atol)
    np.testing.assert_allclose(polarizability.polarizabilities[2],
                               result__0_06, rtol=rtol, atol=atol)
    np.testing.assert_allclose(polarizability.polarizabilities[3],
                               result__0_10, rtol=rtol, atol=atol)

    return


def test_as_many_as_possible_rhf_disk():

    run_as_many_tests_as_possible_rhf_disk('r_lih_hf_sto-3g')

    return


def test_as_many_as_possible_uhf_disk():

    run_as_many_tests_as_possible_uhf_disk('u_lih_cation_hf_sto-3g')

    return

# TODO what is this?
# if __name__ == '__main__':

#     from pyscf import ao2mo, gto, scf

#     import utils

#     mol = gto.Mole()
#     mol.verbose = 5
#     with open('water.xyz') as fh:
#         mol.atom = fh.read()
#     mol.unit = 'Bohr'
#     mol.basis = 'sto-3g'
#     mol.symmetry = False
#     mol.build()

#     mf = scf.RHF(mol)
#     mf.kernel()

#     # In the future, we want the full Fock matrices in the MO basis.
#     fock = utils.fix_moenergies_shape(mf.mo_energy)
#     mocoeffs = utils.fix_mocoeffs_shape(mf.mo_coeff)
#     norb = mocoeffs.shape[-1]
#     tei_mo = ao2mo.full(mol, mocoeffs, aosym='s4', compact=False).reshape(norb, norb, norb, norb)

#     ao_integrals_dipole = mol.intor('cint1e_r_sph', comp=3)
#     # 'cg' stands for common gauge
#     ao_integrals_angmom = mol.intor('cint1e_cg_irxp_sph', comp=3)
#     # ao_integrals_spnorb = mol.intor('cint1e_ia01p_sph', comp=3)
#     ao_integrals_spnorb = 0
#     for atm_id in range(mol.natm):
#         mol.set_rinv_orig(mol.atom_coord(atm_id))
#         chg = mol.atom_charge(atm_id)
#         ao_integrals_spnorb += chg * mol.intor('cint1e_prinvxp_sph', comp=3)

#     operator_dipole = operators.Operator(label='dipole', is_imaginary=False, is_spin_dependent=False)
#     operator_fermi = operators.Operator(label='fermi', is_imaginary=False, is_spin_dependent=True)
#     operator_angmom = operators.Operator(label='angmom', is_imaginary=True, is_spin_dependent=False)
#     operator_spnorb = operators.Operator(label='spinorb', is_imaginary=True, is_spin_dependent=True)
#     operator_dipole.ao_integrals = ao_integrals_dipole
#     from daltools import prop
#     ao_integrals_fermi1 = prop.read('FC O  01', tmpdir='/home/eric/development/pyresponse/dalton_fermi/DALTON_scratch_eric/dalton.h2o_sto3g.response_static_rpa_singlet_5672')
#     ao_integrals_fermi2 = prop.read('FC H  02', tmpdir='/home/eric/development/pyresponse/dalton_fermi/DALTON_scratch_eric/dalton.h2o_sto3g.response_static_rpa_singlet_5672')
#     ao_integrals_fermi3 = prop.read('FC H  03', tmpdir='/home/eric/development/pyresponse/dalton_fermi/DALTON_scratch_eric/dalton.h2o_sto3g.response_static_rpa_singlet_5672')
#     operator_fermi.ao_integrals = np.concatenate((ao_integrals_fermi1, ao_integrals_fermi2, ao_integrals_fermi3), axis=0)
#     # print(operator_fermi.ao_integrals.shape)
#     # import sys; sys.exit()
#     operator_angmom.ao_integrals = ao_integrals_angmom
#     operator_spnorb.ao_integrals = ao_integrals_spnorb

#     nocc_a, nocc_b = mol.nelec
#     nvirt_a, nvirt_b = norb - nocc_a, norb - nocc_b
#     occupations = [nocc_a, nvirt_a, nocc_b, nvirt_b]
#     cphf = CPHF(mocoeffs, fock, occupations)
#     cphf.tei_mo = (tei_mo, )
#     cphf.tei_mo_type = 'full'

#     # cphf.add_operator(operator_dipole)
#     cphf.add_operator(operator_fermi)
#     # cphf.add_operator(operator_angmom)
#     cphf.add_operator(operator_spnorb)

#     cphf.set_frequencies()

#     for hamiltonian in ('rpa', 'tda'):
#         for spin in ('singlet', 'triplet'):
#             print('hamiltonian: {}, spin: {}'.format(hamiltonian, spin))
#             cphf.run(solver_type='exact', hamiltonian=hamiltonian, spin=spin)
#             thresh = 1.0e-10
#             cphf.results[0][cphf.results[0] < thresh] = 0.0
#             print(cphf.results[0])

if __name__ == '__main__':
    test_final_result_rhf_h2o_sto3g_rpa_singlet()
    test_final_result_rhf_h2o_sto3g_rpa_triplet()
    test_final_result_rhf_h2o_sto3g_tda_singlet()
    test_final_result_rhf_h2o_sto3g_tda_triplet()
    test_as_many_as_possible_rhf_disk()
    test_as_many_as_possible_uhf_disk()
