import numpy as np

import psi4

from pymolresponse.utils import fix_mocoeffs_shape, fix_moenergies_shape


def occupations_from_psi4wfn(wfn: psi4.core.Wavefunction) -> np.ndarray:
    # Not needed.
    # occupations_a = wfn.occupation_a().to_array()
    # occupations_b = wfn.occupation_b().to_brray()
    # assert occupations_a.shape == occupations_b.shape
    norb = wfn.nmo()
    nocc_a = wfn.nalpha()
    nocc_b = wfn.nbeta()
    nvirt_a = norb - nocc_a
    nvirt_b = norb - nocc_b
    return np.asarray([nocc_a, nvirt_a, nocc_b, nvirt_b], dtype=int)


def mocoeffs_from_psi4wfn(wfn: psi4.core.Wavefunction) -> np.ndarray:
    is_uhf = not wfn.same_a_b_orbs()
    Ca = wfn.Ca().to_array()
    if is_uhf:
        Cb = wfn.Cb().to_array()
        C = np.stack((Ca, Cb), axis=0)
    else:
        C = Ca
    # Clean up.
    return fix_mocoeffs_shape(C)


def moenergies_from_psi4wfn(wfn: psi4.core.Wavefunction) -> np.ndarray:
    is_uhf = not wfn.same_a_b_orbs()
    Ea = wfn.epsilon_a().to_array()
    if is_uhf:
        Eb = wfn.epsilon_b().to_array()
        E = np.stack((Ea, Eb), axis=0).T
    else:
        E = Ea
    # Clean up.
    return fix_moenergies_shape(E)
