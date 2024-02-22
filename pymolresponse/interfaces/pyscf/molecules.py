import pyscf

from pymolresponse.data import COORDDIR


def molecule_water_sto3g_angstrom(verbose: int = 0) -> pyscf.gto.Mole:
    mol = pyscf.gto.Mole()
    mol.verbose = verbose
    mol.output = None

    mol.basis = "sto-3g"
    mol.charge = 0
    mol.spin = 0

    mol.atom = """O         -1.81298        0.53384       -0.01233
H         -0.82365        0.49649        0.00870
H         -2.10234       -0.29131        0.45244
"""

    return mol


def molecule_water_sto3g(verbose: int = 0) -> pyscf.gto.Mole:
    mol = pyscf.gto.Mole()
    mol.verbose = verbose
    mol.output = None

    with open(COORDDIR / "water.xyz") as fh:
        mol.atom = fh.read()
    mol.basis = "sto-3g"
    mol.charge = 0
    mol.spin = 0

    mol.unit = "Bohr"

    return mol


def molecule_physicists_water_sto3g(verbose: int = 0) -> pyscf.gto.Mole:
    mol = pyscf.gto.Mole()
    mol.verbose = verbose
    mol.output = None

    with open(COORDDIR / "water_psi4numpy.xyz") as fh:
        next(fh)
        next(fh)
        mol.atom = fh.read()
    mol.basis = "sto-3g"
    mol.charge = 0
    mol.spin = 0

    return mol


def molecule_physicists_water_augccpvdz(verbose: int = 0) -> pyscf.gto.Mole:
    mol = pyscf.gto.Mole()
    mol.verbose = verbose
    mol.output = None

    with open(COORDDIR / "water_psi4numpy.xyz") as fh:
        next(fh)
        next(fh)
        mol.atom = fh.read()
    mol.basis = "aug-cc-pvdz"
    mol.charge = 0
    mol.spin = 0

    return mol


def molecule_glycine_sto3g(verbose: int = 0) -> pyscf.gto.Mole:
    mol = pyscf.gto.Mole()
    mol.verbose = verbose
    mol.output = None

    with open(COORDDIR / "glycine.xyz") as fh:
        next(fh)
        next(fh)
        mol.atom = fh.read()
    mol.basis = "sto-3g"
    mol.charge = 0
    mol.spin = 0

    return mol


def molecule_trithiolane_sto3g(verbose: int = 0) -> pyscf.gto.Mole:
    mol = pyscf.gto.Mole()
    mol.verbose = verbose
    mol.output = None

    with open(COORDDIR / "trithiolane.xyz") as fh:
        next(fh)
        next(fh)
        mol.atom = fh.read()
    mol.basis = "sto-3g"
    mol.charge = 0
    mol.spin = 0

    return mol


def hydrogen_atom_sto3g(verbose: int = 0) -> pyscf.gto.Mole:
    mol = pyscf.gto.Mole()
    mol.verbose = verbose
    mol.output = None

    mol.atom = [["H", (0.0, 0.0, 0.0)]]
    mol.basis = "sto-3g"
    mol.charge = 0
    mol.spin = 1

    return mol


def molecule_bc2h4_cation_sto3g(verbose: int = 0) -> pyscf.gto.Mole:
    mol = pyscf.gto.Mole()
    mol.verbose = verbose
    mol.output = None

    with open(COORDDIR / "BC2H4.xyz") as fh:
        next(fh)
        next(fh)
        mol.atom = fh.read()
    mol.basis = "sto-3g"
    mol.charge = 1
    mol.spin = 0

    return mol


def molecule_bc2h4_neutral_radical_sto3g(verbose: int = 0) -> pyscf.gto.Mole:
    mol = molecule_bc2h4_cation_sto3g(verbose)
    mol.charge = 0
    mol.spin = 1

    return mol


def molecule_lih_cation_sto3g(verbose: int = 0) -> pyscf.gto.Mole:
    mol = pyscf.gto.Mole()
    mol.verbose = verbose
    mol.output = None

    with open(COORDDIR / "LiH.xyz") as fh:
        next(fh)
        next(fh)
        mol.atom = fh.read()
    mol.basis = "sto-3g"
    mol.charge = 1
    mol.spin = 1

    return mol


def molecule_0w4a_dication_321g(verbose: int = 0) -> pyscf.gto.Mole:
    mol = pyscf.gto.Mole()
    mol.verbose = verbose
    mol.output = None

    with open(COORDDIR / "0w4a.xyz") as fh:
        next(fh)
        next(fh)
        mol.atom = fh.read()
    mol.basis = "3-21g"
    mol.charge = 2
    mol.spin = 1

    return mol


def molecule_bh_cation_def2_svp(verbose: int = 0) -> pyscf.gto.Mole:
    mol = pyscf.gto.Mole()
    mol.verbose = verbose
    mol.output = None

    mol.atom = [["B", (0.0000, 0.0000, 0.0000)], ["H", (0.0000, 0.0000, 1.2340)]]
    mol.basis = "def2-svp"
    mol.charge = 1
    mol.spin = 1

    return mol
