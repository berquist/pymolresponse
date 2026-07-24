import psi4

from pymolresponse.data import COORDDIR


def molecule_water_sto3g_angstrom() -> psi4.core.Molecule:
    mol = _read("water_angstrom.xyz")

    basis = "sto-3g"
    mol.set_basis_all_atoms(basis, "BASIS")
    mol.set_molecular_charge(0)
    mol.set_multiplicity(1)

    mol.update_geometry()

    psi4.core.set_global_option("BASIS", basis)

    return mol


def molecule_water_sto3g() -> psi4.core.Molecule:
    # TODO this isn"t a proper xyz file. Why?
    mol = _read("water.xyz", 0)

    # TODO This is supposed to be in bohr.

    basis = "sto-3g"
    mol.set_basis_all_atoms(basis, "BASIS")
    mol.set_molecular_charge(0)
    mol.set_multiplicity(1)

    mol.update_geometry()

    psi4.core.set_global_option("BASIS", basis)

    return mol


def molecule_physicists_water_sto3g() -> psi4.core.Molecule:
    mol = _read("water_psi4numpy.xyz")

    basis = "sto-3g"
    mol.set_basis_all_atoms(basis, "BASIS")
    mol.set_molecular_charge(0)
    mol.set_multiplicity(1)

    mol.update_geometry()

    psi4.core.set_global_option("BASIS", basis)

    return mol


def molecule_physicists_water_augccpvdz() -> psi4.core.Molecule:
    mol = _read("water_psi4numpy.xyz")

    basis = "aug-cc-pvdz"
    mol.set_basis_all_atoms(basis, "BASIS")
    mol.set_molecular_charge(0)
    mol.set_multiplicity(1)

    mol.update_geometry()

    psi4.core.set_global_option("BASIS", basis)

    return mol


def molecule_glycine_sto3g() -> psi4.core.Molecule:
    mol = _read("glycine.xyz")

    basis = "sto-3g"
    mol.set_basis_all_atoms(basis, "BASIS")
    mol.set_molecular_charge(0)
    mol.set_multiplicity(1)

    mol.update_geometry()

    psi4.core.set_global_option("BASIS", basis)

    return mol


def molecule_trithiolane_sto3g() -> psi4.core.Molecule:
    mol = _read("trithiolane.xyz")

    basis = "sto-3g"
    mol.set_basis_all_atoms(basis, "BASIS")
    mol.set_molecular_charge(0)
    mol.set_multiplicity(1)

    mol.update_geometry()

    psi4.core.set_global_option("BASIS", basis)

    return mol


def hydrogen_atom_sto3g() -> psi4.core.Molecule:
    mol = psi4.geometry(
        """
0 2
H
"""
    )

    basis = "sto-3g"
    mol.set_basis_all_atoms(basis, "BASIS")

    mol.update_geometry()

    psi4.core.set_global_option("BASIS", basis)

    return mol


def molecule_bc2h4_cation_sto3g() -> psi4.core.Molecule:
    mol = _read("BC2H4.xyz")

    basis = "sto-3g"
    mol.set_basis_all_atoms(basis, "BASIS")
    mol.set_molecular_charge(1)
    mol.set_multiplicity(1)

    mol.update_geometry()

    psi4.core.set_global_option("BASIS", basis)

    return mol


def molecule_bc2h4_neutral_radical_sto3g() -> psi4.core.Molecule:
    mol = molecule_bc2h4_cation_sto3g()

    mol.set_molecular_charge(0)
    mol.set_multiplicity(2)

    mol.update_geometry()

    return mol


def molecule_lih_cation_sto3g() -> psi4.core.Molecule:
    mol = _read("LiH.xyz")

    basis = "sto-3g"
    mol.set_basis_all_atoms(basis, "BASIS")
    mol.set_molecular_charge(1)
    mol.set_multiplicity(2)

    mol.update_geometry()

    psi4.core.set_global_option("BASIS", basis)

    return mol


def molecule_0w4a_dication_321g() -> psi4.core.Molecule:
    mol = _read("0w4a.xyz")

    basis = "3-21g"
    mol.set_basis_all_atoms(basis, "BASIS")
    mol.set_molecular_charge(2)
    mol.set_multiplicity(2)

    mol.update_geometry()

    psi4.core.set_global_option("BASIS", basis)

    return mol


def molecule_bh_cation_def2_svp() -> psi4.core.Molecule:
    mol = psi4.geometry(
        """
1 2
B 0.0000 0.0000 0.0000
H 0.0000 0.0000 1.2340
"""
    )

    basis = "def2-svp"
    mol.set_basis_all_atoms(basis, "BASIS")

    mol.update_geometry()

    psi4.core.set_global_option("BASIS", basis)

    return mol


def _read(xyzname: str, nlineskip: int = 2) -> psi4.core.Molecule:
    with open(COORDDIR / xyzname) as fh:
        mol = psi4.core.Molecule.from_string(
            molstr="\n".join(fh.readlines()[nlineskip:]), fix_com=True, fix_orientation=True
        )
    mol.set_point_group(psi4.core.PointGroup("c1"))
    return mol
