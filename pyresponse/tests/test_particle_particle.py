import os.path

import pyscf


__filedir__ = os.path.realpath(os.path.dirname(__file__))
refdir = os.path.join(__filedir__, 'reference_data')


def test_particle_particle_ethene_singlet():

    mol = pyscf.gto.Mole()
    mol.verbose = 5
    mol.output = None

    with open(os.path.join(refdir, 'ethene.xyz')) as fh:
        next(fh)
        next(fh)
        mol.atom = fh.read()
    mol.basis = 'aug-cc-pvdz'
    mol.charge = 0
    mol.spin = 0
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.scf()

    return

if __name__ == '__main__':
    test_particle_particle_ethene_singlet()
