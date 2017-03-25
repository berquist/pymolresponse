import numpy as np
from utils import get_reference_value_from_file, clean_dalton_label, dalton_label_to_operator
from test_calculators import calculate_rhf, calculate_uhf
from integrals import parse_aoproper

frequency = '0.000000e+00'

dalton_label = 'FC Li 01'
label_1 = clean_dalton_label(dalton_label)
label_2 = label_1
operator = dalton_label_to_operator(label_1)
dalton_integrals = parse_aoproper('r_lih_hf_sto-3g/dalton_response_rpa_singlet/AOPROPER')
operator.ao_integrals = dalton_integrals[dalton_label]['integrals'][np.newaxis, ...]
slice_idx = operator.slice_idx

testcases = (
    'r_lih_hf_sto-3g',
    'u_lih_cation_hf_sto-3g',
)
hamiltonians = ('rpa', 'tda')
spins = ('singlet', 'triplet')

print('val', 'moenergies', 'mocoeffs')

for testcase in testcases:
    for hamiltonian in hamiltonians:
        for spin in spins:

            print(testcase, hamiltonian, spin)

            if testcase[0] == 'r':
                calculator = calculate_rhf
            elif testcase[0] == 'u':
                calculator = calculate_uhf
            else:
                pass

            dalton_tmpdir = '{}/dalton_response_{}_{}'.format(testcase, hamiltonian, spin)
            ref = get_reference_value_from_file(testcase + '/ref', hamiltonian, spin, frequency, label_1, label_2)
            print(ref, 'ref')
            # DALTON MO energies can't be used in UHF calculations.
            if calculator == calculate_rhf:
                print(calculator(
                    dalton_tmpdir,
                    hamiltonian=hamiltonian,
                    spin=spin,
                    operator=operator,
                    source_moenergies='dalton',
                    source_mocoeffs='dalton')[slice_idx, slice_idx],
                      'dalton', 'dalton'
                )
            print(calculator(
                dalton_tmpdir,
                hamiltonian=hamiltonian,
                spin=spin,
                operator=operator,
                source_moenergies='pyscf',
                source_mocoeffs='dalton')[slice_idx, slice_idx],
                  'pyscf', 'dalton'
            )
            # DALTON MO energies can't be used in UHF calculations.
            if calculator == calculate_rhf:
                print(calculator(
                    dalton_tmpdir,
                    hamiltonian=hamiltonian,
                    spin=spin,
                    operator=operator,
                    source_moenergies='dalton',
                    source_mocoeffs='pyscf')[slice_idx, slice_idx],
                      'dalton', 'pyscf'
                )
            print(calculator(
                dalton_tmpdir,
                hamiltonian=hamiltonian,
                spin=spin,
                operator=operator,
                source_moenergies='pyscf',
                source_mocoeffs='pyscf')[slice_idx, slice_idx],
                  'pyscf', 'pyscf'
            )
