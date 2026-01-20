from typing import TYPE_CHECKING

import pyscf

from pymolresponse.utils import DirtyMocoeffs, fix_mocoeffs_shape


if TYPE_CHECKING:
    from pymolresponse.indices import Occupations


def occupations_from_pyscf_mol(mol: pyscf.gto.Mole, C: DirtyMocoeffs) -> "Occupations":
    norb = fix_mocoeffs_shape(C).shape[-1]
    nocc_a, nocc_b = mol.nelec
    nvirt_a, nvirt_b = norb - nocc_a, norb - nocc_b
    return nocc_a, nvirt_a, nocc_b, nvirt_b
