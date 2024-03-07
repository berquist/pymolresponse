from pathlib import Path
from typing import Union

from pymolresponse.data import REFDIR
from pymolresponse.interfaces.dalton.utils import dalton_label_to_operator
from pymolresponse.operators import Operator
from pymolresponse.tests.test_calculators import calculate_disk_rhf


def run_dalton_label_to_operator(
    dalton_label: str,
    operator_label: str,
    slice_idx: int,
    is_imaginary: bool,
    is_spin_dependent: bool,
) -> Operator:
    operator = dalton_label_to_operator(dalton_label)
    assert operator.label == operator_label
    assert operator.slice_idx == slice_idx
    assert operator.is_imaginary == is_imaginary
    assert operator.is_spin_dependent == is_spin_dependent
    return operator


# def run_reference_disk_rhf(testcase):


def run_as_many_tests_as_possible_rhf_disk(testcase: Union[Path, str]) -> None:
    testcasedir = REFDIR / testcase

    thresh = 5.0e-3

    # These are operators that we can't run for some reason.
    exclude_parts = ("_spnorb",)

    entries = []

    with open(testcasedir / "ref") as fh:
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
            print("{} {} {} {:10} {:10} {:+10e} {:+10e} {:+10e}".format(*format_list))
            assert diff < thresh


def run_as_many_tests_as_possible_uhf_disk(testcase: Union[Path, str]) -> None:
    from .test_calculators import calculate_disk_uhf

    testcasedir = REFDIR / testcase

    thresh = 1.0e-1

    # These are operators that we can't run for some reason.
    exclude_parts = ("_spnorb",)

    entries = []

    with open(testcasedir / "ref") as fh:
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
            print("{} {} {} {:10} {:10} {:+10e} {:+10e} {:+10e}".format(*format_list))
            assert diff < thresh
