from typing import Optional, Tuple

import numpy as np

import psi4

from pymolresponse.integrals import JK, IntegralLabel, Integrals

DIPOLE = object()
DIPVEL = object()
ANGMOM_COMMON_GAUGE = object()


class IntegralsPsi4(Integrals):
    def __init__(self, wfn_or_mol) -> None:
        super().__init__()

        if isinstance(wfn_or_mol, psi4.core.Molecule):
            wfn = psi4.core.Wavefunction.build(wfn_or_mol, psi4.core.get_global_option("BASIS"))
        elif isinstance(wfn_or_mol, psi4.core.Wavefunction):
            wfn = wfn_or_mol
        else:
            raise RuntimeError
        self._mints = psi4.core.MintsHelper(wfn)

    def _compute(self, label: IntegralLabel) -> np.ndarray:
        if label == DIPOLE:
            return np.stack([np.asarray(Mc) for Mc in self._mints.ao_dipole()])
        elif label == DIPVEL:
            return np.stack([np.asarray(Mc) for Mc in self._mints.ao_nabla()])
        elif label == ANGMOM_COMMON_GAUGE:
            return np.stack([np.asarray(Lc) for Lc in self._mints.ao_angular_momentum()])
        else:
            raise RuntimeError


# Taken from Psi4NumPy's helper_HF.py
def compute_jk(jk, C_left, C_right=None):
    """
    A python wrapper for a Psi4 JK object to consume and produce NumPy arrays.

    Computes the following matrices:
    D = C_left C_right.T
    J_pq = (pq|rs) D_rs
    K_pq = (pr|qs) D_rs

    Parameters
    ----------
    jk : psi4.core.JK
        A initialized Psi4 JK object
    C_left : list of array_like or a array_like object
        Orbitals used to compute the JK object with
    C_right : list of array_like (optional, None)
        Optional C_right orbitals, otherwise it is assumed C_right == C_left

    Returns
    -------
    JK : tuple of ndarray
        Returns the J and K objects

    Notes
    -----
    This function uses the Psi4 JK object and will compute the initialized JK type (DF, PK, OUT_OF_CORE, etc)


    Examples
    --------

    ndocc = 5
    nbf = 15

    Cocc = np.random.rand(nbf, ndocc)

    jk = psi4.core.JK.build(wfn.basisset())
    jk.set_memory(int(1.25e8))  # 1GB
    jk.initialize()
    jk.print_header()


    J, K = compute_jk(jk, Cocc)

    J_list, K_list = compute_jk(jk, [Cocc, Cocc])
    """

    # Clear out the matrices
    jk.C_clear()

    list_input = True
    if not isinstance(C_left, (list, tuple)):
        C_left = [C_left]
        list_input = False

    for c in C_left:
        mat = psi4.core.Matrix.from_array(c)
        jk.C_left_add(mat)

    # Do we have C_right?
    if C_right is not None:
        if not isinstance(C_right, (list, tuple)):
            C_right = [C_right]

        if len(C_left) != len(C_right):
            raise ValueError("JK: length of left and right matrices is not equal")

        if not isinstance(C_right, (list, tuple)):
            C_right = [C_right]

        for c in C_right:
            mat = psi4.core.Matrix.from_array(c)
            jk.C_right_add(mat)

    # Compute the JK
    jk.compute()

    # Unpack
    J = []
    K = []
    for n in range(len(C_left)):
        J.append(np.array(jk.J()[n]))
        K.append(np.array(jk.K()[n]))

    jk.C_clear()

    # Duck type the return
    if list_input:
        return (J, K)
    else:
        return (J[0], K[0])


class JKPsi4(JK):
    def __init__(self, wfn) -> None:
        super().__init__()

        self._jk = psi4.core.JK.build(wfn.basisset())
        self._jk.initialize()

    def compute_from_density(self, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def compute_from_mocoeffs(
        self, C_left: np.ndarray, C_right: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        # TODO is would be good to understand why this doesn't work.
        #
        # self._jk.finalize()
        # self._jk.initialize()
        # self._jk.C_clear()
        # self._jk.C_left_add(psi4.core.Matrix.from_array(C_left))
        # if C_right is not None:
        #     if len(C_left) != len(C_right):
        #         raise ValueError(
        #             "JK: length of left and right MO coefficient matrices is not equal"
        #         )
        #     self._jk.C_right_add(psi4.core.Matrix.from_array(C_right))
        # self._jk.compute()
        # self._jk.C_clear()
        # return self._jk.J()[0].np, self._jk.K()[0].np
        return compute_jk(self._jk, C_left, C_right)
