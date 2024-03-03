"""Tools for performing AO-to-MO transformations of two-electron integrals."""

from typing import Optional, Sequence

import numpy as np

from pymolresponse.utils import fix_mocoeffs_shape


class AO2MO:
    """Interface for performing AO-to-MO tranformations of two-electron integrals."""

    # TODO see
    # https://github.com/psi4/psi4numpy/blob/master/Tutorials/01_Psi4NumPy-Basics/1f_tensor-manipulation.ipynb

    def __init__(
        self,
        C: np.ndarray,
        occupations: Sequence[int],
        verbose: int = 1,
        I: Optional[np.ndarray] = None,  # noqa: E741
    ) -> None:
        self.C = fix_mocoeffs_shape(C)
        self.occupations = occupations
        self.verbose = verbose
        self.I = I

        self.nocc_alph, self.nvirt_alph, self.nocc_beta, self.nvirt_beta = occupations

        self.tei_mo = tuple()

    @staticmethod
    def transform(
        I: np.ndarray,  # noqa: E741
        C1: np.ndarray,
        C2: np.ndarray,
        C3: np.ndarray,
        C4: np.ndarray,  # noqa: E741
    ) -> np.ndarray:
        """
        Transforms the 4-index ERI I with the 4 transformation matrices C1 to C4.
        """
        nao = I.shape[0]
        MO = np.dot(C1.T, I.reshape(nao, -1)).reshape(C1.shape[1], nao, nao, nao)

        MO = np.einsum("qB,Aqrs->ABrs", C2, MO)
        MO = np.einsum("rC,ABrs->ABCs", C3, MO)
        MO = np.einsum("sD,ABCs->ABCD", C4, MO)
        return MO

    def perform_rhf_full(self) -> None:
        r"""Perform the transformation :math:`(\mu\nu|\lambda\sigma) \rightarrow (pq|rs)`."""
        tei_mo = self.transform(self.I, self.C[0], self.C[0], self.C[0], self.C[0])
        self.tei_mo = (tei_mo,)

    def perform_rhf_partial(self) -> None:
        r"""Perform the transformation :math:`(\mu\nu|\lambda\sigma) \rightarrow (ia|jb), (ij|ab)`."""
        norb = self.nocc_alph + self.nvirt_alph
        oa = slice(0, self.nocc_alph)
        va = slice(self.nocc_alph, norb)
        tei_mo_ovov = self.transform(
            self.I, self.C[0, :, oa], self.C[0, :, va], self.C[0, :, oa], self.C[0, :, va]
        )
        tei_mo_oovv = self.transform(
            self.I, self.C[0, :, oa], self.C[0, :, oa], self.C[0, :, va], self.C[0, :, va]
        )
        self.tei_mo = (tei_mo_ovov, tei_mo_oovv)

    def perform_uhf_full(self) -> None:
        r"""Perform the transformation :math:`(\mu\nu|\lambda\sigma) \rightarrow (p^{\alpha}q^{\alpha}|r^{\alpha}s^{\alpha}), (p^{\alpha}q^{\alpha}|r^{\beta}s^{\beta}), (p^{\beta}q^{\beta}|r^{\alpha}s^{\alpha}), (p^{\beta}q^{\beta}|r^{\beta}s^{\beta})`."""

    def perform_uhf_partial(self) -> None:
        r"""Perform the transformation :math:`(\mu\nu|\lambda\sigma) \rightarrow (i^{\alpha}a^{\alpha}|j^{\alpha}b^{\alpha}), (i^{\alpha}a^{\alpha}|j^{\beta}b^{\beta}), (i^{\beta}a^{\beta}|j^{\alpha}b^{\alpha}), (i^{\beta}a^{\beta}|j^{\beta}b^{\beta}), (i^{\alpha}j^{\alpha}|a^{\alpha}b^{\alpha}), (i^{\beta}j^{\beta}|a^{\beta}b^{\beta})`."""
