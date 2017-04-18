from __future__ import print_function
from __future__ import division

import numpy as np
import numpy.linalg as npl

import periodictable

from constants import convfac_au_to_debye


def get_most_abundant_isotope(element):
    most_abundant_isotope = element.isotopes[0]
    abundance = 0
    for iso in element:
        if iso.abundance > abundance:
            most_abundant_isotope = iso
            abundance = iso.abundance
    return most_abundant_isotope


def get_isotopic_masses(charges):
    masses = []
    for charge in charges:
        el = periodictable.elements[charge]
        isotope = get_most_abundant_isotope(el)
        mass = isotope.mass
        masses.append(mass)
    return np.array(masses)


def calc_center_of_mass_pyscf(pyscfmol):
    charges = pyscfmol.atom_charges()
    masses = get_isotopic_masses(charges)
    coords = pyscfmol.atom_coords()
    return calc_center_of_mass(coords, masses)


def calc_center_of_mass(coords, masses):
    assert len(masses.shape) == 1
    denominator = np.sum(masses)
    numerator = np.sum(coords * masses[..., np.newaxis], axis=0)
    return numerator / denominator


def calc_center_of_nuclear_charge(coords, charges):
    dummy = np.zeros(3)
    center = nuclear_dipole_contribution(coords, charges, dummy)
    total_charge = np.sum(charges)
    return center / total_charge


def calc_center_of_electronic_charge_pyscf(D, pyscfmol):
    assert len(D.shape) == 2
    # no linear dependencies!
    assert D.shape[0] == D.shape[1]
    zerovec = np.zeros(3)
    dipole_at_zerovec = electronic_dipole_contribution_pyscf(D, pyscfmol, zerovec)
    nelec = pyscfmol.tot_electrons()
    return -dipole_at_zerovec / nelec


def nuclear_dipole_contribution(nuccoords, nuccharges, origin_in_bohrs):
    assert isinstance(nuccoords, np.ndarray)
    assert isinstance(nuccharges, np.ndarray)
    assert isinstance(origin_in_bohrs, np.ndarray)
    assert len(nuccoords.shape) == 2
    assert nuccoords.shape[1] == 3
    # assert len(nuccharges.shape) == 1
    assert nuccoords.shape[0] == nuccharges.shape[0]
    assert origin_in_bohrs.shape == (3,)

    return np.sum((nuccoords - origin_in_bohrs) * nuccharges, axis=0)


def electronic_dipole_contribution_pyscf(D, pyscfmol, origin_in_bohrs):
    assert isinstance(D, np.ndarray)
    assert len(D.shape) == 2
    assert D.shape[0] == D.shape[1]
    assert isinstance(origin_in_bohrs, np.ndarray)
    assert origin_in_bohrs.shape == (3,)
    # TODO what to assert about pyscfmol? at least isinstance

    M_AO = pyscfmol.intor('cint1e_r_sph', comp=3)
    assert isinstance(M_AO, np.ndarray)
    assert len(M_AO.shape) == 3
    assert M_AO.shape[1] == M_AO.shape[2]
    assert M_AO.shape[0] == 3
    M100_AO = M_AO[0, :, :]
    M010_AO = M_AO[1, :, :]
    M001_AO = M_AO[2, :, :]

    M100_MO = D * M100_AO
    M010_MO = D * M010_AO
    M001_MO = D * M001_AO

    dipole_electronic_atomic_units = -np.asarray([np.sum(M100_MO), np.sum(M010_MO), np.sum(M001_MO)])
    return dipole_electronic_atomic_units


def calculate_dipole_pyscf(nuccoords, nuccharges, origin, D, pyscfmol, do_print=False):
    assert origin.shape == (3,)
    nuclear_components_au = nuclear_dipole_contribution(nuccoords, nuccharges, origin)
    electronic_components_au = electronic_dipole_contribution_pyscf(D, pyscfmol, origin)
    total_components_au = electronic_components_au + nuclear_components_au
    if do_print:
        nuclear_components_debye = nuclear_components_au * convfac_au_to_debye
        electronic_components_debye = electronic_components_au * convfac_au_to_debye
        total_components_debye = total_components_au * convfac_au_to_debye
        nuclear_norm_au = npl.norm(nuclear_components_au)
        electronic_norm_au = npl.norm(electronic_components_au)
        total_norm_au = npl.norm(total_components_au)
        nuclear_norm_debye = nuclear_norm_au * convfac_au_to_debye
        electronic_norm_debye = electronic_norm_au * convfac_au_to_debye
        total_norm_debye = total_norm_au * convfac_au_to_debye
        print(' origin                        [a.u.]: {} {} {}'.format(*origin))
        print(' dipole components, electronic [a.u.]: {} {} {}'.format(*electronic_components_au))
        print(' dipole components, nuclear    [a.u.]: {} {} {}'.format(*nuclear_components_au))
        print(' dipole components, total      [a.u.]: {} {} {}'.format(*total_components_au))
        print(' dipole moment, electronic     [a.u.]: {}'.format(electronic_norm_au))
        print(' dipole moment, nuclear        [a.u.]: {}'.format(nuclear_norm_au))
        print(' dipole moment, total          [a.u.]: {}'.format(total_norm_au))
        print(' dipole components, electronic [D]   : {} {} {}'.format(*electronic_components_debye))
        print(' dipole components, nuclear    [D]   : {} {} {}'.format(*nuclear_components_debye))
        print(' dipole components, total      [D]   : {} {} {}'.format(*total_components_debye))
        print(' dipole moment, electronic     [D]   : {}'.format(electronic_norm_debye))
        print(' dipole moment, nuclear        [D]   : {}'.format(nuclear_norm_debye))
        print(' dipole moment, total          [D]   : {}'.format(total_norm_debye))
    return total_components_au


def calculate_origin_pyscf(origin_string, nuccoords, nuccharges, D, pyscfmol, do_print=False):
    assert isinstance(origin_string, str)
    origin_string = origin_string.lower()
    assert origin_string in ('explicitly-set', 'zero',
                             'com', 'centerofmass',
                             'ecc', 'centerofelcharge',
                             'ncc', 'centerofnuccharge')
    zerovec = np.zeros(3)

    if origin_string == 'explicitly-set':
        if do_print:
            print(" --- Origin: explicitly-set ---")
        origin = zerovec
    elif origin_string == 'zero':
        if do_print:
            print(" --- Origin: zero ---")
        origin = zerovec
    elif origin_string in ('com', 'centerofmass'):
        if do_print:
            print(" --- Origin: center of mass ---")
        masses = get_isotopic_masses(nuccharges[:, 0])
        origin = calc_center_of_mass(nuccoords, masses)
    elif origin_string in ('ecc', 'centerofelcharge'):
        if do_print:
            print(" --- Origin: center of electronic charge ---")
        origin = calc_center_of_electronic_charge_pyscf(D, pyscfmol)
    elif origin_string in ('ncc', 'centerofnuccharge'):
        if do_print:
            print(" --- Origin: center of nuclear charge ---")
        origin = calc_center_of_nuclear_charge(nuccoords, nuccharges)
    else:
        pass

    return origin
