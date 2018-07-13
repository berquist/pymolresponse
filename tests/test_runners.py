import os.path

from pyresponse import utils


__filedir__ = os.path.realpath(os.path.dirname(__file__))
refdir = os.path.join(__filedir__, 'reference_data')


def run_dalton_label_to_operator(dalton_label, operator_label, slice_idx, is_imaginary, is_spin_dependent):
    operator = utils.dalton_label_to_operator(dalton_label)
    assert operator.label == operator_label
    assert operator.slice_idx == slice_idx
    assert operator.is_imaginary == is_imaginary
    assert operator.is_spin_dependent == is_spin_dependent
    return operator


# def run_reference_disk_rhf(testcase):

def run_as_many_tests_as_possible_rhf_disk(testcase):

    from .test_calculators import calculate_disk_rhf

    testcasedir = os.path.join(refdir, testcase)

    thresh = 5.0e-3

    # These are operators that we can't run for some reason.
    exclude_parts = (
        '_spnorb',
    )

    entries = []

    with open(os.path.join(testcasedir, 'ref')) as fh:
        for line in fh:
            tokens = line.split()
            assert len(tokens) == 6
            hamiltonian = tokens[0]
            spin = tokens[1]
            frequency = tokens[2]
            label_1 = tokens[3]
            label_2 = tokens[4]
            ref = float(tokens[5])

            entry = (hamiltonian, spin, frequency, label_1, label_2, ref)
            entries.append(entry)

    for entry in entries:
        assert len(entry) == 6
        hamiltonian, spin, frequency, label_1, label_2, ref = entry
        ignore_label_1 = any(exclude_part in label_1 for exclude_part in exclude_parts)
        ignore_label_2 = any(exclude_part in label_2 for exclude_part in exclude_parts)
        if not ignore_label_1 and not ignore_label_2:
            res = calculate_disk_rhf(testcasedir, hamiltonian, spin, frequency, label_1, label_2)
            diff = abs(ref) - abs(res)
            format_list = (testcase, hamiltonian, spin, label_1, label_2, ref, res, diff)
            print('{} {} {} {:10} {:10} {:+10e} {:+10e} {:+10e}'.format(*format_list))
            assert diff < thresh

    return


def run_as_many_tests_as_possible_uhf_disk(testcase):

    from .test_calculators import calculate_disk_uhf

    testcasedir = os.path.join(refdir, testcase)

    thresh = 1.0e-1

    # These are operators that we can't run for some reason.
    exclude_parts = (
        '_spnorb',
    )

    entries = []

    with open(os.path.join(testcasedir, 'ref')) as fh:
        for line in fh:
            tokens = line.split()
            assert len(tokens) == 6
            hamiltonian = tokens[0]
            spin = tokens[1]
            frequency = tokens[2]
            label_1 = tokens[3]
            label_2 = tokens[4]
            ref = float(tokens[5])

            entry = (hamiltonian, spin, frequency, label_1, label_2, ref)
            entries.append(entry)

    for entry in entries:
        assert len(entry) == 6
        hamiltonian, spin, frequency, label_1, label_2, ref = entry
        ignore_label_1 = any(exclude_part in label_1 for exclude_part in exclude_parts)
        ignore_label_2 = any(exclude_part in label_2 for exclude_part in exclude_parts)
        if not ignore_label_1 and not ignore_label_2:
            res = calculate_disk_uhf(testcasedir, hamiltonian, spin, frequency, label_1, label_2)
            diff = abs(ref) - abs(res)
            format_list = (testcase, hamiltonian, spin, label_1, label_2, ref, res, diff)
            print('{} {} {} {:10} {:10} {:+10e} {:+10e} {:+10e}'.format(*format_list))
            assert diff < thresh

    return
