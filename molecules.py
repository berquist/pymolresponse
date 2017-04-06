import pyscf


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
