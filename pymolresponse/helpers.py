"""Utility functions that are core to calculating physical values."""

from typing import List, Sequence, Tuple

import numpy as np
import periodictable


def get_most_abundant_isotope(element: periodictable.core.Element) -> periodictable.core.Isotope:
    most_abundant_isotope = element.isotopes[0]
    abundance = 0
    for iso in element:
        if iso.abundance > abundance:
            most_abundant_isotope = iso
            abundance = iso.abundance
    return most_abundant_isotope


def get_isotopic_masses(charges: Sequence[int]) -> np.ndarray:
    masses = []
    for charge in charges:
        el = periodictable.elements[charge]
        isotope = get_most_abundant_isotope(el)
        mass = isotope.mass
        masses.append(mass)
    return np.array(masses)


def calc_center_of_mass(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    assert len(masses.shape) == 1
    denominator = np.sum(masses)
    numerator = np.sum(coords * masses[..., np.newaxis], axis=0)
    return numerator / denominator


def calc_center_of_nuclear_charge(coords: np.ndarray, charges: np.ndarray) -> np.ndarray:
    dummy = np.zeros(3)
    center = nuclear_dipole_contribution(coords, charges, dummy)
    total_charge = np.sum(charges)
    return center / total_charge


def nuclear_dipole_contribution(
    nuccoords: np.ndarray, nuccharges: np.ndarray, origin_in_bohrs: np.ndarray
) -> np.ndarray:
    assert isinstance(nuccoords, np.ndarray)
    assert isinstance(nuccharges, np.ndarray)
    assert isinstance(origin_in_bohrs, np.ndarray)
    assert len(nuccoords.shape) == 2
    assert nuccoords.shape[1] == 3
    assert nuccoords.shape[0] == nuccharges.shape[0]
    assert origin_in_bohrs.shape == (3,)
    assert len(nuccharges.shape) in (1, 2)
    if len(nuccharges.shape) == 1:
        charges = nuccharges[..., np.newaxis]
    else:
        assert nuccharges.shape[1] == 1
        charges = nuccharges

    return np.sum((nuccoords - origin_in_bohrs) * charges, axis=0)


def get_uhf_values(
    mat_uhf_a: np.ndarray,
    mat_uhf_b: np.ndarray,
    pair_rohf: Tuple[int, int],
    nocc_a: int,
    nvirt_a: int,
    nocc_b: int,
    nvirt_b: int,
) -> List[float]:
    """For a pair ROHF 1-based indices, find the corresponing alpha- and
    beta-spin UHF values.
    """

    # TODO there has to be a better way than including this here...
    range_uhf_a_closed = list(range(0, nocc_a))
    range_uhf_a_virt = list(range(nocc_a, nocc_a + nvirt_a))
    range_uhf_b_closed = list(range(0, nocc_b))
    range_uhf_b_virt = list(range(nocc_b, nocc_b + nvirt_b))
    indices_uhf_a = [(i, a) for i in range(nocc_a) for a in range(nvirt_a)]
    indices_uhf_b = [(i, a) for i in range(nocc_b) for a in range(nvirt_b)]
    # These are the indices for unique pairs considering the full
    # dimensionality of the system (correct orbital window), [norb,
    # norb], starting from 1.
    indices_display_uhf_a = [(p + 1, q + 1) for p in range_uhf_a_closed for q in range_uhf_a_virt]
    indices_display_uhf_b = [(p + 1, q + 1) for p in range_uhf_b_closed for q in range_uhf_b_virt]

    values = []
    if pair_rohf in indices_display_uhf_a:
        idx_uhf_a = indices_display_uhf_a.index(pair_rohf)
        p_a, q_a = indices_uhf_a[idx_uhf_a]
        val_uhf_a = mat_uhf_a[p_a, q_a]
        values.append(val_uhf_a)
    if pair_rohf in indices_display_uhf_b:
        idx_uhf_b = indices_display_uhf_b.index(pair_rohf)
        p_b, q_b = indices_uhf_b[idx_uhf_b]
        val_uhf_b = mat_uhf_b[p_b, q_b]
        values.append(val_uhf_b)
    return values


def mat_uhf_to_packed_rohf(
    mat_alpha: np.ndarray, mat_beta: np.ndarray, indices_display_rohf: List[Tuple[int, int]]
) -> np.ndarray:
    dim = len(indices_display_rohf)
    mat_rohf = np.zeros(dim)
    for idx, pair_rohf in enumerate(indices_display_rohf):
        mat_rohf[idx] = sum(get_uhf_values(mat_alpha, mat_beta, pair_rohf))
    return mat_rohf
