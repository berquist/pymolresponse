import numpy as np

import pyscf

from pymolresponse.utils import fix_mocoeffs_shape


def occupations_from_pyscf_mol(mol: pyscf.gto.Mole, C: np.ndarray) -> np.ndarray:
    norb = fix_mocoeffs_shape(C).shape[-1]
    nocc_a, nocc_b = mol.nelec
    nvirt_a, nvirt_b = norb - nocc_a, norb - nocc_b
    return np.asarray([nocc_a, nvirt_a, nocc_b, nvirt_b], dtype=int)
