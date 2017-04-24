import pyscf


def hydrogen_atom_STO3G(verbose=0):

    mol = pyscf.gto.Mole()
    mol.verbose = verbose
    mol.output = None

    mol.atom = [
        ['H' , (0.0, 0.0, 0.0)]
    ]
    mol.basis = 'sto-3g'
    mol.charge = 0
    mol.spin = 1

    return mol


def molecule_BC2H4_cation_HF_STO3G(verbose=0):

    mol = pyscf.gto.Mole()
    mol.verbose = verbose
    mol.output = None

    with open('BC2H4.xyz') as fh:
        next(fh)
        next(fh)
        mol.atom = fh.read()
    mol.basis = 'sto-3g'
    mol.charge = 1
    mol.spin = 0

    return mol


def molecule_BC2H4_neutral_radical_HF_STO3G(verbose=0):

    mol = molecule_BC2H4_cation_HF_STO3G(verbose)
    mol.charge = 0
    mol.spin = 1

    return mol


def molecule_LiH_cation_HF_STO3G(verbose=0):

    mol = pyscf.gto.Mole()
    mol.verbose = verbose
    mol.output = None

    with open('LiH.xyz') as fh:
        next(fh)
        next(fh)
        mol.atom = fh.read()
    mol.basis = 'sto-3g'
    mol.charge = 1
    mol.spin = 1

    return mol


def molecule_0w4a_dication_HF_321G(verbose=0):

    mol = pyscf.gto.Mole()
    mol.verbose = verbose
    mol.output = None

    with open('0w4a.xyz') as fh:
        next(fh)
        next(fh)
        mol.atom = fh.read()
    mol.basis = '3-21g'
    mol.charge = 2
    mol.spin = 1

    return mol


def molecule_BH_cation_HF_def2_SVP(verbose=0):

    mol = pyscf.gto.Mole()
    mol.verbose = verbose
    mol.output = None

    mol.atom = [
        ['B', (0.0000, 0.0000, 0.0000)],
        ['H', (0.0000, 0.0000, 1.2340)],
    ]
    mol.basis = 'def2-svp'
    mol.charge = 1
    mol.spin = 1

    return mol
