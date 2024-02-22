import numpy as np
import numpy.linalg as npl

import pyscf

from pymolresponse.constants import convfac_au_to_debye
from pymolresponse.helpers import (
    calc_center_of_mass,
    calc_center_of_nuclear_charge,
    get_isotopic_masses,
    nuclear_dipole_contribution,
)


def calc_center_of_mass_pyscf(pyscfmol: pyscf.gto.Mole) -> np.ndarray:
    charges = pyscfmol.atom_charges()
    masses = get_isotopic_masses(charges)
    coords = pyscfmol.atom_coords()
    return calc_center_of_mass(coords, masses)


def calc_center_of_electronic_charge_pyscf(D: np.ndarray, pyscfmol: pyscf.gto.Mole) -> np.ndarray:
    assert len(D.shape) == 2
    # no linear dependencies!
    assert D.shape[0] == D.shape[1]
    zerovec = np.zeros(3)
    dipole_at_zerovec = electronic_dipole_contribution_pyscf(D, pyscfmol, zerovec)
    nelec = pyscfmol.tot_electrons()
    return -dipole_at_zerovec / nelec


def electronic_dipole_contribution_pyscf(
    D: np.ndarray, pyscfmol: pyscf.gto.Mole, origin_in_bohrs: np.ndarray
) -> np.ndarray:
    assert isinstance(D, np.ndarray)
    assert len(D.shape) == 2
    assert D.shape[0] == D.shape[1]
    assert isinstance(origin_in_bohrs, np.ndarray)
    assert origin_in_bohrs.shape == (3,)
    # TODO what to assert about pyscfmol? at least isinstance

    M_AO = pyscfmol.intor("cint1e_r_sph", comp=3)
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

    # pylint: disable=invalid-unary-operand-type
    dipole_electronic_atomic_units = -np.asarray(
        [np.sum(M100_MO), np.sum(M010_MO), np.sum(M001_MO)]
    )
    return dipole_electronic_atomic_units


def calculate_dipole_pyscf(
    nuccoords: np.ndarray,
    nuccharges: np.ndarray,
    origin: np.ndarray,
    D: np.ndarray,
    pyscfmol: pyscf.gto.Mole,
    do_print: bool = False,
) -> np.ndarray:
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
        print(" origin                        [a.u.]: {} {} {}".format(*origin))
        print(" dipole components, electronic [a.u.]: {} {} {}".format(*electronic_components_au))
        print(" dipole components, nuclear    [a.u.]: {} {} {}".format(*nuclear_components_au))
        print(" dipole components, total      [a.u.]: {} {} {}".format(*total_components_au))
        print(" dipole moment, electronic     [a.u.]: {}".format(electronic_norm_au))
        print(" dipole moment, nuclear        [a.u.]: {}".format(nuclear_norm_au))
        print(" dipole moment, total          [a.u.]: {}".format(total_norm_au))
        print(
            " dipole components, electronic [D]   : {} {} {}".format(*electronic_components_debye)
        )
        print(" dipole components, nuclear    [D]   : {} {} {}".format(*nuclear_components_debye))
        print(" dipole components, total      [D]   : {} {} {}".format(*total_components_debye))
        print(" dipole moment, electronic     [D]   : {}".format(electronic_norm_debye))
        print(" dipole moment, nuclear        [D]   : {}".format(nuclear_norm_debye))
        print(" dipole moment, total          [D]   : {}".format(total_norm_debye))
    return total_components_au


def calculate_origin_pyscf(
    origin_string: str,
    nuccoords: np.ndarray,
    nuccharges: np.ndarray,
    D: np.ndarray,
    pyscfmol: pyscf.gto.Mole,
    do_print: bool = False,
) -> np.ndarray:
    assert isinstance(origin_string, str)
    origin_string = origin_string.lower()
    assert origin_string in (
        "explicitly-set",
        "zero",
        "com",
        "centerofmass",
        "ecc",
        "centerofelcharge",
        "ncc",
        "centerofnuccharge",
    )
    zerovec = np.zeros(3)

    if origin_string == "explicitly-set":
        if do_print:
            print(" --- Origin: explicitly-set ---")
        origin = zerovec
    elif origin_string == "zero":
        if do_print:
            print(" --- Origin: zero ---")
        origin = zerovec
    elif origin_string in ("com", "centerofmass"):
        if do_print:
            print(" --- Origin: center of mass ---")
        masses = get_isotopic_masses(nuccharges[:, 0])
        origin = calc_center_of_mass(nuccoords, masses)
    elif origin_string in ("ecc", "centerofelcharge"):
        if do_print:
            print(" --- Origin: center of electronic charge ---")
        origin = calc_center_of_electronic_charge_pyscf(D, pyscfmol)
    elif origin_string in ("ncc", "centerofnuccharge"):
        if do_print:
            print(" --- Origin: center of nuclear charge ---")
        origin = calc_center_of_nuclear_charge(nuccoords, nuccharges)
    else:
        pass

    if do_print:
        print(" Calculating the dipole at the requested origin...")
        calculate_dipole_pyscf(nuccoords, nuccharges, origin, D, pyscfmol, do_print)

    return origin
