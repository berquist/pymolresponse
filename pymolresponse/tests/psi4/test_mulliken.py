from collections import defaultdict

import numpy as np

import psi4

from pymolresponse.helpers import make_density
from pymolresponse.interfaces.psi4 import integrals, molecules
from pymolresponse.interfaces.psi4.utils import (
    mocoeffs_from_psi4wfn,
    occupations_from_psi4wfn,
)


# def get_ao_function_start(atombasis: list[list[int]], atombasis_center: list[int]) -> int | None:
#     if atombasis_center:
#         return atombasis_center[-1]
#     elif atombasis:
#         return atombasis[-1][-1]
#     else:
#         return None


# def test_get_ao_function_start() -> None:
#     assert get_ao_function_start([], [0, 1, 2]) == 2
#     assert get_ao_function_start([[0, 1, 2, 3, 4]], []) == 4
#     assert get_ao_function_start([], []) is None


def test_make_atombasis() -> None:
    assert make_atombasis([0], [0]) == [[0]]
    assert make_atombasis([0, 0, 0, 1, 2], [0, 1, 2, 3, 4]) == [[0, 1, 2], [3], [4]]
    # water sto-3g
    assert make_atombasis([0, 0, 0, 1, 2], [0, 1, 2, 5, 6]) == [[0, 1, 2, 3, 4], [5], [6]]


def make_atombasis(shell_to_center: list[int], shell_to_ao_function: list[int]) -> list[list[int]]:
    assert len(shell_to_center) == len(shell_to_ao_function)
    # shell_to_center
    # [0, 0, 0, 1, 2]
    # shell_to_ao_function
    # [0, 1, 2, 5, 6]
    # shell_to_basis_function
    # [0, 1, 2, 5, 6]
    #
    # center to function
    # 0 -> 0-4 [0, 1, 2-4]
    # 1 -> 5
    # 2 -> 6
    #
    # []
    # [[0]]
    # [[0, 1]]
    # [[0, 1, 2]]
    # [[0, 1, 2, 3, 4], [5]]
    # [[0, 1, 2, 3, 4], [5], [6]]
    center_to_function = defaultdict(list)
    for center, function in zip(shell_to_center, shell_to_ao_function):
        center_to_function[center].append(function)
    for center, functions in center_to_function.items():
        if center > 0:
            prev_center = center_to_function[center - 1]
            prev_center_gapless_last_function = prev_center[-1] + 1
            curr_center_first_function = functions[0]
            if curr_center_first_function > prev_center_gapless_last_function:
                for function in range(
                    prev_center_gapless_last_function, curr_center_first_function
                ):
                    prev_center.append(function)
    return list(center_to_function.values())


def make_atombasis_from_psi4bs(bs) -> list[list[int]]:
    nshell = bs.nshell()
    shell_to_center = [bs.shell_to_center(i) for i in range(nshell)]
    # TODO what's the difference?
    shell_to_ao_function = [bs.shell_to_ao_function(i) for i in range(nshell)]
    # shell_to_basis_function = [bs.shell_to_basis_function(i) for i in range(nshell)]
    return make_atombasis(shell_to_center, shell_to_ao_function)


def test_mulliken() -> None:
    mol = molecules.molecule_water_sto3g_angstrom()
    psi4.core.set_active_molecule(mol)

    options = {"BASIS": "STO-3G", "SCF_TYPE": "PK", "E_CONVERGENCE": 1e-10, "D_CONVERGENCE": 1e-10}

    psi4.set_options(options)

    _, wfn = psi4.energy("hf", return_wfn=True)

    integral_generator = integrals.IntegralsPsi4(wfn)

    S = integral_generator.integrals(integrals.OVERLAP)
    assert S.shape == (7, 7)

    C = mocoeffs_from_psi4wfn(wfn)
    occupations = occupations_from_psi4wfn(wfn)

    D = make_density(C, occupations)

    bs = wfn.basisset()
    atombasis = make_atombasis_from_psi4bs(bs)

    gop = 0.5 * ((S @ D[0].T) + (S.T @ D[0]))
    atomic_populations = [
        np.trace(gop[np.ix_(atombasis[icen], atombasis[icen])]) for icen in range(len(atombasis))
    ]

    atomnos = [mol.fcharge(i) for i in range(len(atombasis))]

    atomic_charges = [atomno - atompop for atomno, atompop in zip(atomnos, atomic_populations)]

    # data/reference/water/qchem_103_static.out
    np.testing.assert_allclose(
        atomic_charges, [-0.336476, 0.168238, 0.168238], rtol=0.0, atol=1.0e-6
    )
