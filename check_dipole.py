from utils import get_reference_value_from_file
from test_calculators import calculate_rhf, calculate_uhf

frequency = '0.000000e+00'

operator_label = 'dipole'
label_1 = 'xdiplen'
label_2 = 'xdiplen'

testcases = (
    'r_lih_hf_sto-3g',
    'u_lih_cation_hf_sto-3g',
)
hamiltonians = ('rpa', 'tda')
spins = ('singlet', 'triplet')

for testcase in testcases:
    for hamiltonian in hamiltonians:
        for spin in spins:

            if testcase[0] == 'r':
                calculator = calculate_rhf
            elif testcase[0] == 'u':
                calculator = calculate_uhf
            else:
                pass

            dalton_tmpdir = '{}/dalton_response_{}_{}'.format(testcase, hamiltonian, spin)
            ref = get_reference_value_from_file(testcase + '/ref', hamiltonian, spin, frequency, label_1, label_2)
            print(ref)
            # DALTON MO energies can't be used in UHF calculations.
            if calculator == calculate_rhf:
                print(calculator(dalton_tmpdir, hamiltonian, spin, operator_label, 'dalton', 'dalton')[0, 0])
            print(calculator(dalton_tmpdir, hamiltonian, spin, operator_label, 'pyscf', 'dalton')[0, 0])
            # DALTON MO energies can't be used in UHF calculations.
            if calculator == calculate_rhf:
                print(calculator(dalton_tmpdir, hamiltonian, spin, operator_label, 'dalton', 'pyscf')[0, 0])
            print(calculator(dalton_tmpdir, hamiltonian, spin, operator_label, 'pyscf', 'pyscf')[0, 0])
