from .iterators import EigSolver

def form_pp_a_matrix_mo_singlet_full(E_MO, TEI_MO, nocc):
    norb = E_MO.shape[0]
    nvirt = norb - nocc
    nvv = nvirt * nvirt

    A = np.empty(shape=(nvv, nvv))

    for a in range(nvirt):
        for c in range(nvirt):
            ac = a*nvirt + c
            for b in range(nvirt):
                for d in range(nvirt):
                    bd = b*nvirt + d
                    A[ac, bd] = 2*TEI_MO[a + nocc, c + nocc, b + nocc, d + nocc] \
                                - TEI_MO[a + nocc, d + nocc, b + nocc, c + nocc]
                    if (a == c) and (b == d):
                        A[ac, bd] += E_MO[a, a] + E_MO[b, b]

    return A


def form_pp_b_matrix_mo_singlet_full(TEI_MO, nocc):
    norb = TEI_MO.shape[0]
    nvirt = norb - nocc
    nov = nocc * nvirt

    B = np.empty(shape=(nov, nov))

    for a in range(nvirt):
        for i in range(nocc):
            ai = a*nocc + i
            for b in range(nvirt):
                for j in range(nocc):
                    bj = b*nocc + j
                    B[ai, bj] = 2*TEI_MO[a + nocc, i, b + nocc, j] \
                                - TEI_MO[a + nocc, j, b + nocc, i]

    return B


def form_pp_c_matrix_mo_singlet_full(E_MO, TEI_MO, nocc):
    norb = E_MO.shape[0]
    noo = nocc * nocc

    C = np.empty(shape=(noo, noo))

    for i in range(nocc):
        for k in range(nocc):
            ik = i*nocc + k
            for j in range(nocc):
                for l in range(nocc):
                    jl = j*nocc + l
                    C[ik, jl] = 2 * TEI_MO[i, k, j, l] - TEI_MO[i, l, k, j]
                    if (i == k) and (j == l):
                        C[ik, jl] -= E_MO[i, i] + E_MO[j, j]

    return C


class ppExactDiagonalizationSolver(EigSolver):

    def __init__(self, mocoeffs, moenergies, occupations, *args, **kwargs):
        super().__init__(mocoeffs, moenergies, occupations, *args, **kwargs)

    def form_explicit_hessian(self, hamiltonian=None, spin=None, frequency=None):

        assert hasattr(self, 'tei_mo')
        assert self.tei_mo is not None
        # for now, RHF only, and full (pq|rs) rather than (vv|vv),
        # (vo|vo), (oo|oo)
        assert len(self.tei_mo) == 1
        assert self.tei_mo_type == 'full'

        if not hamiltonian:
            hamiltonian = self.hamiltonian
        if not spin:
            spin = self.spin

        assert isinstance(hamiltonian, str)
        assert isinstance(spin, str)
        hamiltonian = hamiltonian.lower()
        spin = spin.lower()

        # assert hamiltonian in ('rpa', 'tda')
        # assert spin in ('singlet', 'triplet')
        assert hamiltonian == 'pp-rpa'
        assert spin == 'singlet'

        self.hamiltonian = hamiltonian
        self.spin = spin

        nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = self.occupations
        nov_alph = nocc_alph * nvirt_alph
        nov_beta = nocc_beta * nvirt_beta

        tei_mo = self.tei_mo[0]

        A = form_pp_a_matrix_mo_singlet_full(self.moenergies[0, ...], tei_mo, nocc_alph)
        B = form_pp_b_matrix_mo_singlet_full(tei_mo, nocc_alph)
        C = form_pp_c_matrix_mo_singlet_full(self.moenergies[0, ...], tei_mo, nocc_alph)

        # TODO ???
        G = np.asarray(np.bmat([[A, B],
                                [B.T, C]]))
        self.explicit_hessian = G

    def diagonalize_explicit_hessian(self):
        nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = self.occupations

        eigvals, eigvecs = sp.linalg.eig(self.explicit_hessian)
        # TODO sorting?
        self.eigvals = eigvals
        self.eigvecs = eigvecs
