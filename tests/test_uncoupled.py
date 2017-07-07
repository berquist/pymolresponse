import numpy as np

import pyscf

from pyresponse import utils, cphf, iterators, operators, ao2mo, molecules


def mol_atom(symbol='He', charge=0, spin=0, basis='sto-3g', verbose=0):
    mol = pyscf.gto.Mole()
    mol.verbose = verbose
    mol.output = None
    mol.atom = [
        [symbol, (0.0, 0.0, 0.0)],
    ]
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()
    return mol


# pylint: disable=bad-whitespace
rhf_coupled = {
    0.0: {
        'result': np.array([[  3.47899e+01,  -1.77591e-05,   2.24416e-04],
                            [ -1.77591e-05,   4.43286e+01,  -1.91191e-01],
                            [  2.24416e-04,  -1.91191e-01,   1.68476e+01]]),
        'error_max_diag': 1.0e-4,
    },
    0.0773178: {
        'result': np.array([[  3.53972e+01,  -8.48249e-06,   2.05118e-04],
                            [ -8.48249e-06,   4.52552e+01,  -2.17003e-01],
                            [  2.05118e-04,  -2.17003e-01,   1.70848e+01]]),
        'error_max_diag': 1.0e-4,
    },
    0.128347: {
        'result': np.array([[  3.65252e+01,  -2.36748e-06,   1.76171e-04],
                            [ -2.36748e-06,   4.70314e+01,  -2.86868e-01],
                            [  1.76171e-04,  -2.86868e-01,   1.75452e+01]]),
        'error_max_diag': 1.0e-4,
    },
}


# pylint: disable=bad-whitespace
rhf_uncoupled = {
    0.0: {
        'result': np.array([[  3.22077e+01,   1.50870e-04,   5.33903e-04],
                            [  1.50870e-04,   4.04007e+01,   2.38977e+00],
                            [  5.33903e-04,   2.38977e+00,   1.49105e+01]]),
        'error_max_diag': 1.0e-4,
    },
    0.0773178: {
        'result': np.array([[  3.25145e+01,   1.51073e-04,   5.32981e-04],
                            [  1.51073e-04,   4.08495e+01,   2.43278e+00],
                            [  5.32981e-04,   2.43278e+00,   1.50160e+01]]),
        'error_max_diag': 1.0e-4,
    },
    0.128347: {
        'result': np.array([[  3.30688e+01,   1.51325e-04,   5.30935e-04],
                            [  1.51325e-04,   4.16650e+01,   2.51199e+00],
                            [  5.30935e-04,   2.51199e+00,   1.52058e+01]]),
        'error_max_diag': 1.0e-4,
    },
}


# pylint: disable=bad-whitespace
uhf_coupled = {
    0.0: {
        'result': np.array([[  3.63373e+01,  -8.63039e-04,   4.27969e-04],
                            [ -8.63039e-04,   3.50789e+01,  -2.31340e+00],
                            [  4.27969e-04,  -2.31340e+00,   1.91465e+01]]),
        'error_max_diag': 1.0e-4,
    },
    0.0773178: {
        'result': np.array([[  3.70513e+01,  -1.35934e-02,   2.24680e-03],
                            [ -1.35934e-02,   4.33419e+01,   1.02081e-01],
                            [  2.24680e-03,   1.02081e-01,   2.04381e+01]]),
        'error_max_diag': 1.0e-3,
    },
    0.128347: {
        'result': np.array([[  3.97633e+01,   2.90266e-02,  -2.85016e-03],
                            [  2.90266e-02,   4.65961e+01,   1.43676e+00],
                            [ -2.85016e-03,   1.43676e+00,   2.25248e+01]]),
        'error_max_diag': 1.0e-2,
    },
    # 100 nm!
    0.4556355: {
        'result': np.array([[  1.55494e+02,  -4.03311e-01,  -4.85361e-01],
                            [ -4.03311e-01,  -6.60084e+02,  -9.94324e+01],
                            [ -4.85361e-01,  -9.94324e+01,   2.80687e+01]]),
        'error_max_diag': 2.0e-0,
    },
}


# pylint: disable=bad-whitespace
uhf_uncoupled = {
    0.0: {
        'result': np.array([[  3.45537e+01,   8.12383e-05,   9.13783e-04],
                            [  8.12383e-05,   5.73968e+01,   8.85793e+00],
                            [  9.13783e-04,   8.85793e+00,   1.95421e+01]]),
        'error_max_diag': 1.0e-4,
    },
    0.0773178: {
        'result': np.array([[  3.49369e+01,   9.40696e-05,   9.31091e-04],
                            [  9.40696e-05,   5.89094e+01,   9.26378e+00],
                            [  9.31091e-04,   9.26378e+00,   1.98017e+01]]),
        'error_max_diag': 1.0e-4,
    },
    0.128347: {
        'result': np.array([[  3.56387e+01,   1.22202e-04,   9.68661e-04],
                            [  1.22202e-04,   6.19498e+01,   1.00967e+01],
                            [  9.68661e-04,   1.00967e+01,   2.03043e+01]]),
        'error_max_diag': 2.0e-4,
    },
    0.4556355: {
        'result': np.array([[  1.03406e+02,  -1.24375e-02,  -1.16407e-02],
                            [ -1.24375e-02,   7.97942e+01,   1.90354e+01],
                            [ -1.16407e-02,   1.90354e+01,   3.33718e+01]]),
        'error_max_diag': 2.0e-2,
    },
}


def test_uncoupled_rhf():
    mol = molecules.molecule_trithiolane_HF_STO3G(0)
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.scf()

    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = utils.occupations_from_pyscf_mol(mol, C)

    solver = iterators.ExactInv(C, E, occupations)

    solver.tei_mo = ao2mo.perform_tei_ao2mo_rhf_partial(mol, C, mol.verbose)
    solver.tei_mo_type = 'partial'

    driver = cphf.CPHF(solver)

    operator_diplen = operators.Operator(label='dipole',
                                         is_imaginary=False, is_spin_dependent=False, triplet=False)
    integrals_diplen_ao = mol.intor('cint1e_r_sph', comp=3)
    operator_diplen.ao_integrals = integrals_diplen_ao
    driver.add_operator(operator_diplen)

    frequencies = [0.0, 0.0773178, 0.128347]
    driver.set_frequencies(frequencies)

    driver.run(solver_type='exact', hamiltonian='rpa', spin='singlet')

    for idxf, frequency in enumerate(frequencies):
        print(idxf, frequency)
        print('uncoupled')
        diag_res = np.diag(driver.uncoupled_results[idxf])
        diag_ref = np.diag(rhf_uncoupled[frequency]['result'])
        diff = diag_res - diag_ref
        print(diag_res)
        print(diag_ref)
        print(diff)
        assert np.max(np.abs(diff)) < rhf_uncoupled[frequency]['error_max_diag']
        print('coupled')
        diag_res = np.diag(driver.results[idxf])
        diag_ref = np.diag(rhf_coupled[frequency]['result'])
        diff = diag_res - diag_ref
        print(diag_res)
        print(diag_ref)
        print(diff)
        assert np.max(np.abs(diff)) < rhf_coupled[frequency]['error_max_diag']

    return


def test_uncoupled_uhf():
    mol = molecules.molecule_trithiolane_HF_STO3G(0)
    mol.charge = 1
    mol.spin = 1
    mol.build()

    mf = pyscf.scf.uhf.UHF(mol)
    mf.scf()

    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = utils.occupations_from_pyscf_mol(mol, C)

    solver = iterators.ExactInv(C, E, occupations)

    solver.tei_mo = ao2mo.perform_tei_ao2mo_uhf_partial(mol, C, mol.verbose)
    solver.tei_mo_type = 'partial'

    driver = cphf.CPHF(solver)

    operator_diplen = operators.Operator(label='dipole',
                                         is_imaginary=False, is_spin_dependent=False, triplet=False)
    integrals_diplen_ao = mol.intor('cint1e_r_sph', comp=3)
    operator_diplen.ao_integrals = integrals_diplen_ao
    driver.add_operator(operator_diplen)

    frequencies = [0.0, 0.0773178, 0.128347, 0.4556355]
    driver.set_frequencies(frequencies)

    driver.run(solver_type='exact', hamiltonian='rpa', spin='singlet')

    for idxf, frequency in enumerate(frequencies):
        print(idxf, frequency)
        print('uncoupled')
        diag_res = np.diag(driver.uncoupled_results[idxf])
        diag_ref = np.diag(uhf_uncoupled[frequency]['result'])
        diff = diag_res - diag_ref
        print(diag_res)
        print(diag_ref)
        print(diff)
        assert np.max(np.abs(diff)) < uhf_uncoupled[frequency]['error_max_diag']
        print('coupled')
        diag_res = np.diag(driver.results[idxf])
        diag_ref = np.diag(uhf_coupled[frequency]['result'])
        diff = diag_res - diag_ref
        print(diag_res)
        print(diag_ref)
        print(diff)
        assert np.max(np.abs(diff)) < uhf_coupled[frequency]['error_max_diag']

    return

if __name__ == '__main__':
    test_uncoupled_rhf()
    test_uncoupled_uhf()
