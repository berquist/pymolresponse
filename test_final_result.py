import numpy as np

from utils import (np_load, parse_int_file_2)
from cphf import (CPHF, Operator)


def test_final_result_rhf_h2o_sto3g_rpa_singlet():
    hamiltonian = 'rpa'
    spin = 'singlet'

    C = np_load('C.npz')
    C = C[np.newaxis, ...]
    E = np_load('F_MO.npz')
    E = E[np.newaxis, ...]
    TEI_MO = np_load('TEI_MO.npz')
    # nocc_alph, nvirt_alph, nocc_beta, nvirt_beta
    occupations = [5, 2, 5, 2]
    stub = 'h2o_sto3g_'
    dim = occupations[0] + occupations[1]
    mat_dipole_x = parse_int_file_2(stub + "mux.dat", dim)
    mat_dipole_y = parse_int_file_2(stub + "muy.dat", dim)
    mat_dipole_z = parse_int_file_2(stub + "muz.dat", dim)

    cphf = CPHF(C, E, occupations)
    frequencies = (0.0, 0.02, 0.06, 0.1)
    cphf.frequencies = frequencies
    ao_integrals_dipole = np.empty(shape=(3, dim, dim))
    ao_integrals_dipole[0, :, :] = mat_dipole_x
    ao_integrals_dipole[1, :, :] = mat_dipole_y
    ao_integrals_dipole[2, :, :] = mat_dipole_z
    operator_dipole = Operator(label='dipole', is_imaginary=False, is_spin_dependent=False)
    operator_dipole.ao_integrals = ao_integrals_dipole
    cphf.add_operator(operator_dipole)
    cphf.tei_mo = TEI_MO

    cphf.run(solver='explicit', hamiltonian=hamiltonian, spin=spin)

    assert len(cphf.results) == len(frequencies)

    result__0_00 = np.array([[ 7.93556221,  0.,          0.        ],
                             [ 0.,          3.06821077,  0.        ],
                             [ 0.,          0.,          0.05038621]])

    result__0_02 = np.array([[ 7.94312371,  0.,          0.        ],
                             [ 0.,          3.07051688,  0.        ],
                             [ 0.,          0.,          0.05054685]])

    result__0_06 = np.array([[ 8.00414009,  0.,          0.        ],
                             [ 0.,          3.08913608,  0.        ],
                             [ 0.,          0.,          0.05186977]])

    result__0_10 = np.array([[ 8.1290378,   0.,          0.        ],
                             [ 0.,          3.12731363,  0.        ],
                             [ 0.,          0.,          0.05473482]])

    atol = 1.0e-8
    rtol = 0.0
    np.testing.assert_allclose(cphf.results[0], result__0_00, rtol=rtol, atol=atol)
    np.testing.assert_allclose(cphf.results[1], result__0_02, rtol=rtol, atol=atol)
    np.testing.assert_allclose(cphf.results[2], result__0_06, rtol=rtol, atol=atol)
    np.testing.assert_allclose(cphf.results[3], result__0_10, rtol=rtol, atol=atol)

    return


def test_final_result_rhf_h2o_sto3g_rpa_triplet():
    hamiltonian = 'rpa'
    spin = 'triplet'

    C = np_load('C.npz')
    C = C[np.newaxis, ...]
    E = np_load('F_MO.npz')
    E = E[np.newaxis, ...]
    TEI_MO = np_load('TEI_MO.npz')
    # nocc_alph, nvirt_alph, nocc_beta, nvirt_beta
    occupations = [5, 2, 5, 2]
    stub = 'h2o_sto3g_'
    dim = occupations[0] + occupations[1]
    mat_dipole_x = parse_int_file_2(stub + "mux.dat", dim)
    mat_dipole_y = parse_int_file_2(stub + "muy.dat", dim)
    mat_dipole_z = parse_int_file_2(stub + "muz.dat", dim)

    cphf = CPHF(C, E, occupations)
    frequencies = (0.0, 0.02, 0.06, 0.1)
    cphf.frequencies = frequencies
    ao_integrals_dipole = np.empty(shape=(3, dim, dim))
    ao_integrals_dipole[0, :, :] = mat_dipole_x
    ao_integrals_dipole[1, :, :] = mat_dipole_y
    ao_integrals_dipole[2, :, :] = mat_dipole_z
    operator_dipole = Operator(label='dipole', is_imaginary=False, is_spin_dependent=False)
    operator_dipole.ao_integrals = ao_integrals_dipole
    cphf.add_operator(operator_dipole)
    cphf.tei_mo = TEI_MO

    cphf.run(solver='explicit', hamiltonian=hamiltonian, spin=spin)

    assert len(cphf.results) == len(frequencies)

    result__0_00 = np.array([[ 26.59744305,   0.,           0.        ],
                             [  0.,          18.11879557,   0.        ],
                             [  0.,           0.,           0.07798969]])

    result__0_02 = np.array([[ 26.68282287,   0.,           0.        ],
                             [  0.,          18.19390051,   0.        ],
                             [  0.,           0.,           0.07837521]])

    result__0_06 = np.array([[ 27.38617401,   0.,           0.        ],
                             [  0.,          18.81922578,   0.        ],
                             [  0.,           0.,           0.08160226]])

    result__0_10 = np.array([[ 28.91067234,   0.,           0.        ],
                             [  0.,          20.21670386,   0.        ],
                             [  0.,           0.,           0.08892512]])

    atol = 1.0e-8
    rtol = 0.0
    np.testing.assert_allclose(cphf.results[0], result__0_00, rtol=rtol, atol=atol)
    np.testing.assert_allclose(cphf.results[1], result__0_02, rtol=rtol, atol=atol)
    np.testing.assert_allclose(cphf.results[2], result__0_06, rtol=rtol, atol=atol)
    np.testing.assert_allclose(cphf.results[3], result__0_10, rtol=rtol, atol=atol)

    return


def test_final_result_rhf_h2o_sto3g_tda_singlet():
    hamiltonian = 'tda'
    spin = 'singlet'

    C = np_load('C.npz')
    C = C[np.newaxis, ...]
    E = np_load('F_MO.npz')
    E = E[np.newaxis, ...]
    TEI_MO = np_load('TEI_MO.npz')
    # nocc_alph, nvirt_alph, nocc_beta, nvirt_beta
    occupations = [5, 2, 5, 2]
    stub = 'h2o_sto3g_'
    dim = occupations[0] + occupations[1]
    mat_dipole_x = parse_int_file_2(stub + "mux.dat", dim)
    mat_dipole_y = parse_int_file_2(stub + "muy.dat", dim)
    mat_dipole_z = parse_int_file_2(stub + "muz.dat", dim)

    cphf = CPHF(C, E, occupations)
    frequencies = (0.0, 0.02, 0.06, 0.1)
    cphf.frequencies = frequencies
    ao_integrals_dipole = np.empty(shape=(3, dim, dim))
    ao_integrals_dipole[0, :, :] = mat_dipole_x
    ao_integrals_dipole[1, :, :] = mat_dipole_y
    ao_integrals_dipole[2, :, :] = mat_dipole_z
    operator_dipole = Operator(label='dipole', is_imaginary=False, is_spin_dependent=False)
    operator_dipole.ao_integrals = ao_integrals_dipole
    cphf.add_operator(operator_dipole)
    cphf.tei_mo = TEI_MO

    cphf.run(solver='explicit', hamiltonian=hamiltonian, spin=spin)

    assert len(cphf.results) == len(frequencies)

    result__0_00 = np.array([[ 8.89855952,  0.,          0.        ],
                             [ 0.,          4.00026556,  0.        ],
                             [ 0.,          0.,          0.0552774 ]])

    result__0_02 = np.array([[ 8.90690928,  0.,          0.        ],
                             [ 0.,          4.00298342,  0.        ],
                             [ 0.,          0.,          0.05545196]])

    result__0_06 = np.array([[ 8.97427725,  0.,          0.        ],
                             [ 0.,          4.02491517,  0.        ],
                             [ 0.,          0.,          0.05688918]])

    result__0_10 = np.array([[ 9.11212633,  0.,          0.        ],
                             [ 0.,          4.06981937,  0.        ],
                             [ 0.,          0.,          0.05999934]])

    atol = 1.0e-8
    rtol = 0.0
    np.testing.assert_allclose(cphf.results[0], result__0_00, rtol=rtol, atol=atol)
    np.testing.assert_allclose(cphf.results[1], result__0_02, rtol=rtol, atol=atol)
    np.testing.assert_allclose(cphf.results[2], result__0_06, rtol=rtol, atol=atol)
    np.testing.assert_allclose(cphf.results[3], result__0_10, rtol=rtol, atol=atol)

    return


def test_final_result_rhf_h2o_sto3g_tda_triplet():
    hamiltonian = 'tda'
    spin = 'triplet'

    C = np_load('C.npz')
    C = C[np.newaxis, ...]
    E = np_load('F_MO.npz')
    E = E[np.newaxis, ...]
    TEI_MO = np_load('TEI_MO.npz')
    # nocc_alph, nvirt_alph, nocc_beta, nvirt_beta
    occupations = [5, 2, 5, 2]
    stub = 'h2o_sto3g_'
    dim = occupations[0] + occupations[1]
    mat_dipole_x = parse_int_file_2(stub + "mux.dat", dim)
    mat_dipole_y = parse_int_file_2(stub + "muy.dat", dim)
    mat_dipole_z = parse_int_file_2(stub + "muz.dat", dim)

    cphf = CPHF(C, E, occupations)
    frequencies = (0.0, 0.02, 0.06, 0.1)
    cphf.frequencies = frequencies
    ao_integrals_dipole = np.empty(shape=(3, dim, dim))
    ao_integrals_dipole[0, :, :] = mat_dipole_x
    ao_integrals_dipole[1, :, :] = mat_dipole_y
    ao_integrals_dipole[2, :, :] = mat_dipole_z
    operator_dipole = Operator(label='dipole', is_imaginary=False, is_spin_dependent=False)
    operator_dipole.ao_integrals = ao_integrals_dipole
    cphf.add_operator(operator_dipole)
    cphf.tei_mo = TEI_MO

    cphf.run(solver='explicit', hamiltonian=hamiltonian, spin=spin)

    assert len(cphf.results) == len(frequencies)

    result__0_00 = np.array([[ 14.64430714,   0.,           0.        ],
                             [  0.,           8.80921432,   0.        ],
                             [  0.,           0.,           0.06859496]])

    result__0_02 = np.array([[ 14.68168443,   0.,           0.        ],
                             [  0.,           8.83562647,   0.        ],
                             [  0.,           0.,           0.0689291 ]])

    result__0_06 = np.array([[ 14.98774296,   0.,           0.        ],
                             [  0.,           9.0532224,    0.        ],
                             [  0.,           0.,           0.07172414]])

    result__0_10 = np.array([[ 15.63997724,   0.,           0.        ],
                             [  0.,           9.52504267,   0.        ],
                             [  0.,           0.,           0.07805428]])

    atol = 1.0e-8
    rtol = 0.0
    np.testing.assert_allclose(cphf.results[0], result__0_00, rtol=rtol, atol=atol)
    np.testing.assert_allclose(cphf.results[1], result__0_02, rtol=rtol, atol=atol)
    np.testing.assert_allclose(cphf.results[2], result__0_06, rtol=rtol, atol=atol)
    np.testing.assert_allclose(cphf.results[3], result__0_10, rtol=rtol, atol=atol)

    return


def getargs():
    """Get command-line arguments."""

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('filename')
    parser.add_argument('-v', '--verbose', action='count', default=0)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    from pyscf import ao2mo, gto, scf

    mol = gto.Mole()
    mol.verbose = 5
    with open('water.xyz') as fh:
        mol.atom = fh.read()
    mol.unit = 'Bohr'
    mol.basis = 'sto-3g'
    mol.symmetry = False
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    mocoeffs = mf.mo_coeff
    moenergies = mf.mo_energy
    norb = mocoeffs.shape[1]
    tei_mo = ao2mo.full(mol, mocoeffs, aosym='s1', compact=False).reshape(norb, norb, norb, norb)

    ao_integrals_dipole = mol.intor('cint1e_r_sph', comp=3)
    # 'cg' stands for common gauge
    ao_integrals_angmom = mol.intor('cint1e_cg_irxp_sph', comp=3)
    # ao_integrals_spnorb = mol.intor('cint1e_ia01p_sph', comp=3)
    ao_integrals_spnorb = 0
    for atm_id in range(mol.natm):
        mol.set_rinv_orig(mol.atom_coord(atm_id))
        chg = mol.atom_charge(atm_id)
        ao_integrals_spnorb += chg * mol.intor('cint1e_prinvxp_sph', comp=3)

    operator_dipole = Operator(label='dipole', is_imaginary=False, is_spin_dependent=False)
    operator_fermi = Operator(label='fermi', is_imaginary=False, is_spin_dependent=True)
    operator_angmom = Operator(label='angmom', is_imaginary=True, is_spin_dependent=False)
    operator_spnorb = Operator(label='spinorb', is_imaginary=True, is_spin_dependent=True)
    operator_dipole.ao_integrals = ao_integrals_dipole
    from daltools import prop
    ao_integrals_fermi1 = prop.read('FC O  01', tmpdir='/home/eric/development/pyresponse/dalton_fermi/DALTON_scratch_eric/dalton.h2o_sto3g.response_static_rpa_singlet_5672')
    ao_integrals_fermi2 = prop.read('FC H  02', tmpdir='/home/eric/development/pyresponse/dalton_fermi/DALTON_scratch_eric/dalton.h2o_sto3g.response_static_rpa_singlet_5672')
    ao_integrals_fermi3 = prop.read('FC H  03', tmpdir='/home/eric/development/pyresponse/dalton_fermi/DALTON_scratch_eric/dalton.h2o_sto3g.response_static_rpa_singlet_5672')
    operator_fermi.ao_integrals = np.concatenate((ao_integrals_fermi1, ao_integrals_fermi2, ao_integrals_fermi3), axis=0)
    # print(operator_fermi.ao_integrals.shape)
    # import sys; sys.exit()
    operator_angmom.ao_integrals = ao_integrals_angmom
    operator_spnorb.ao_integrals = ao_integrals_spnorb

    # In the future, we want the full Fock matrices in the MO basis.
    fock = np.diag(moenergies)[np.newaxis, ...]
    mocoeffs = mocoeffs[np.newaxis, ...]
    nocc_a, nocc_b = mol.nelec
    nvirt_a, nvirt_b = norb - nocc_a, norb - nocc_b
    occupations = [nocc_a, nvirt_a, nocc_b, nvirt_b]
    cphf = CPHF(mocoeffs, fock, occupations)
    cphf.TEI_MO = tei_mo

    # cphf.add_operator(operator_dipole)
    cphf.add_operator(operator_fermi)
    # cphf.add_operator(operator_angmom)
    cphf.add_operator(operator_spnorb)

    cphf.set_frequencies()

    for hamiltonian in ('rpa', 'tda'):
        for spin in ('singlet', 'triplet'):
            print('hamiltonian: {}, spin: {}'.format(hamiltonian, spin))
            cphf.run(solver='explicit', hamiltonian=hamiltonian, spin=spin)
            thresh = 1.0e-10
            cphf.results[0][cphf.results[0] < thresh] = 0.0
            print(cphf.results[0])
