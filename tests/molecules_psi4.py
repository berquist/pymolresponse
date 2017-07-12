import os.path

import psi4


__filedir__ = os.path.realpath(os.path.dirname(__file__))
refdir = os.path.join(__filedir__, 'reference_data')


def molecule_water():

    # TODO this isn't a proper xyz file. Why?
    with open(os.path.join(refdir, 'water.xyz')) as fh:
        mol = psi4.geometry('\n'.join(fh.readlines()))

    # TODO This is supposed to be in bohr.

    mol.set_basis_all_atoms('sto-3g', 'BASIS')
    mol.set_molecular_charge(0)
    mol.set_multiplicity(1)

    mol.update_geometry()

    return mol


def molecule_glycine_sto3g():

    with open(os.path.join(refdir, 'glycine.xyz')) as fh:
        mol = psi4.geometry('\n'.join(fh.readlines()[2:]))

    mol.set_basis_all_atoms('sto-3g', 'BASIS')
    mol.set_molecular_charge(0)
    mol.set_multiplicity(1)

    mol.update_geometry()

    return mol


def molecule_trithiolane_sto3g():

    with open(os.path.join(refdir, 'trithiolane.xyz')) as fh:
        mol = psi4.geometry('\n'.join(fh.readlines()[2:]))

    mol.set_basis_all_atoms('sto-3g', 'BASIS')
    mol.set_molecular_charge(0)
    mol.set_multiplicity(1)

    mol.update_geometry()

    return mol


def hydrogen_atom_sto3g():

    mol = psi4.geometry("""
0 2
H
""")

    mol.update_geometry()

    return mol


def molecule_bc2h4_cation_sto3g():

    with open(os.path.join(refdir, 'BC2H4.xyz')) as fh:
        mol = psi4.geometry('\n'.join(fh.readlines()[2:]))

    mol.set_basis_all_atoms('sto-3g', 'BASIS')
    mol.set_molecular_charge(1)
    mol.set_multiplicity(1)

    mol.update_geometry()

    return mol


def molecule_bc2h4_neutral_radical_hf_sto3g():

    mol = molecule_bc2h4_cation_hf_sto3g()

    mol.set_molecular_charge(0)
    mol.set_multiplicity(2)

    mol.update_geometry()

    return mol


def molecule_lih_cation_sto3g():

    with open(os.path.join(refdir, 'LiH.xyz')) as fh:
        mol = psi4.geometry('\n'.join(fh.readlines()[2:]))

    mol.set_basis_all_atoms('sto-3g', 'BASIS')
    mol.set_molecular_charge(1)
    mol.set_multiplicity(2)

    mol.update_geometry()

    return mol


def molecule_0w4a_dication_321g():

    with open(os.path.join(refdir, '0w4a.xyz')) as fh:
        mol = psi4.geometry('\n'.join(fh.readlines()[2:]))

    mol.set_basis_all_atoms('3-21g', 'BASIS')
    mol.set_molecular_charge(2)
    mol.set_multiplicity(2)

    mol.update_geometry()

    return mol


def molecule_bh_cation_def2_svp():

    mol = psi4.geometry("""
1 2
B 0.0000 0.0000 0.0000
H 0.0000 0.0000 1.2340
""")

    mol.set_basis_all_atoms('def2-svp', 'BASIS')

    mol.update_geometry()

    return mol
