import numpy as np

import pyscf

from pymolresponse.utils import DirtyMocoeffs, fix_mocoeffs_shape


def occupations_from_pyscf_mol(
    mol: pyscf.gto.Mole, C: DirtyMocoeffs
) -> np.ndarray[tuple[int], np.dtype[np.integer]]:
    norb = fix_mocoeffs_shape(C).shape[-1]
    nocc_a, nocc_b = mol.nelec
    nvirt_a, nvirt_b = norb - nocc_a, norb - nocc_b
    ret = np.asarray([nocc_a, nvirt_a, nocc_b, nvirt_b], dtype=int)
    return ret
