"""Tools for performing AO-to-MO transformations of two-electron integrals."""

from .utils import fix_mocoeffs_shape


class AO2MO(object):
    """Interface for performing AO-to-MO tranformations of two-electron integrals.
    """

    # TODO see
    # https://github.com/psi4/psi4numpy/blob/master/Tutorials/01_Psi4NumPy-Basics/1f_tensor-manipulation.ipynb

    def __init__(self, C, verbose=1):

        self.C = fix_mocoeffs_shape(C)
        self.verbose = verbose

        self.tei_mo = tuple()

    def perform_rhf_full(self):
        r"""Perform the transformation :math:`(\mu\nu|\lambda\sigma) \rightarrow (pq|rs)`."""
        raise NotImplementedError

    def perform_rhf_partial(self):
        r"""Perform the transformation :math:`(\mu\nu|\lambda\sigma) \rightarrow (ia|jb), (ij|ab)`."""
        raise NotImplementedError

    def perform_uhf_full(self):
        r"""Perform the transformation :math:`(\mu\nu|\lambda\sigma) \rightarrow (p^{\alpha}q^{\alpha}|r^{\alpha}s^{\alpha}), (p^{\alpha}q^{\alpha}|r^{\beta}s^{\beta}), (p^{\beta}q^{\beta}|r^{\alpha}s^{\alpha}), (p^{\beta}q^{\beta}|r^{\beta}s^{\beta})`."""
        raise NotImplementedError

    def perform_uhf_partial(self):
        r"""Perform the transformation :math:`(\mu\nu|\lambda\sigma) \rightarrow (i^{\alpha}a^{\alpha}|j^{\alpha}b^{\alpha}), (i^{\alpha}a^{\alpha}|j^{\beta}b^{\beta}), (i^{\beta}a^{\beta}|j^{\alpha}b^{\alpha}), (i^{\beta}a^{\beta}|j^{\beta}b^{\beta}), (i^{\alpha}j^{\alpha}|a^{\alpha}b^{\alpha}), (i^{\beta}j^{\beta}|a^{\beta}b^{\beta})`."""
        raise NotImplementedError


class AO2MOpyscf(AO2MO):
    """Perform AO-to-MO transformations using pyscf."""

    # TODO what does the pyscf compact kwarg do?
    def __init__(self, C, verbose=1, pyscfmol=None):
        super().__init__(C, verbose)

        self.pyscfmol = pyscfmol
        from .utils import occupations_from_pyscf_mol
        self.occupations = occupations_from_pyscf_mol(self.pyscfmol, self.C)

    def perform_rhf_full(self):
        norb = self.C.shape[-1]
        from pyscf.ao2mo import full
        tei_mo = full(self.pyscfmol, self.C[0], aosym='s4', compact=False, verbose=self.verbose).reshape(norb, norb, norb, norb)
        self.tei_mo = (tei_mo, )

    def perform_rhf_partial(self):
        nocc_a, nvirt_a, _, _ = self.occupations
        C_occ = self.C[0, :, :nocc_a]
        C_virt = self.C[0, :, nocc_a:]
        C_ovov = (C_occ, C_virt, C_occ, C_virt)
        C_oovv = (C_occ, C_occ, C_virt, C_virt)
        from pyscf.ao2mo import general
        tei_mo_ovov = general(self.pyscfmol, C_ovov, aosym='s4', compact=False, verbose=self.verbose).reshape(nocc_a, nvirt_a, nocc_a, nvirt_a)
        tei_mo_oovv = general(self.pyscfmol, C_oovv, aosym='s4', compact=False, verbose=self.verbose).reshape(nocc_a, nocc_a, nvirt_a, nvirt_a)
        self.tei_mo = (tei_mo_ovov, tei_mo_oovv)

    def perform_uhf_full(self):
        norb = self.C.shape[-1]
        C_a = self.C[0]
        C_b = self.C[1]
        C_aaaa = (C_a, C_a, C_a, C_a)
        C_aabb = (C_a, C_a, C_b, C_b)
        C_bbaa = (C_b, C_b, C_a, C_a)
        C_bbbb = (C_b, C_b, C_b, C_b)
        from pyscf.ao2mo import general
        tei_mo_aaaa = general(self.pyscfmol, C_aaaa, aosym='s4', compact=False, verbose=self.verbose).reshape(norb, norb, norb, norb)
        tei_mo_aabb = general(self.pyscfmol, C_aabb, aosym='s4', compact=False, verbose=self.verbose).reshape(norb, norb, norb, norb)
        tei_mo_bbaa = general(self.pyscfmol, C_bbaa, aosym='s4', compact=False, verbose=self.verbose).reshape(norb, norb, norb, norb)
        tei_mo_bbbb = general(self.pyscfmol, C_bbbb, aosym='s4', compact=False, verbose=self.verbose).reshape(norb, norb, norb, norb)
        self.tei_mo = (tei_mo_aaaa, tei_mo_aabb, tei_mo_bbaa, tei_mo_bbbb)

    # pylint: disable=too-many-locals
    def perform_uhf_partial(self):
        nocc_a, nvirt_a, nocc_b, nvirt_b = self.occupations
        C_occ_alph = self.C[0, :, :nocc_a]
        C_virt_alph = self.C[0, :, nocc_a:]
        C_occ_beta = self.C[1, :, :nocc_b]
        C_virt_beta = self.C[1, :, nocc_b:]
        C_ovov_aaaa = (C_occ_alph, C_virt_alph, C_occ_alph, C_virt_alph)
        C_ovov_aabb = (C_occ_alph, C_virt_alph, C_occ_beta, C_virt_beta)
        C_ovov_bbaa = (C_occ_beta, C_virt_beta, C_occ_alph, C_virt_alph)
        C_ovov_bbbb = (C_occ_beta, C_virt_beta, C_occ_beta, C_virt_beta)
        C_oovv_aaaa = (C_occ_alph, C_occ_alph, C_virt_alph, C_virt_alph)
        C_oovv_bbbb = (C_occ_beta, C_occ_beta, C_virt_beta, C_virt_beta)
        from pyscf.ao2mo import general
        tei_mo_ovov_aaaa = general(self.pyscfmol, C_ovov_aaaa, aosym='s4', compact=False, verbose=self.verbose).reshape(nocc_a, nvirt_a, nocc_a, nvirt_a)
        tei_mo_ovov_aabb = general(self.pyscfmol, C_ovov_aabb, aosym='s4', compact=False, verbose=self.verbose).reshape(nocc_a, nvirt_a, nocc_b, nvirt_b)
        tei_mo_ovov_bbaa = general(self.pyscfmol, C_ovov_bbaa, aosym='s4', compact=False, verbose=self.verbose).reshape(nocc_b, nvirt_b, nocc_a, nvirt_a)
        tei_mo_ovov_bbbb = general(self.pyscfmol, C_ovov_bbbb, aosym='s4', compact=False, verbose=self.verbose).reshape(nocc_b, nvirt_b, nocc_b, nvirt_b)
        tei_mo_oovv_aaaa = general(self.pyscfmol, C_oovv_aaaa, aosym='s4', compact=False, verbose=self.verbose).reshape(nocc_a, nocc_a, nvirt_a, nvirt_a)
        tei_mo_oovv_bbbb = general(self.pyscfmol, C_oovv_bbbb, aosym='s4', compact=False, verbose=self.verbose).reshape(nocc_b, nocc_b, nvirt_b, nvirt_b)
        self.tei_mo = (tei_mo_ovov_aaaa, tei_mo_ovov_aabb, tei_mo_ovov_bbaa, tei_mo_ovov_bbbb, tei_mo_oovv_aaaa, tei_mo_oovv_bbbb)
