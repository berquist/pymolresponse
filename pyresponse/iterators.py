import numpy as np
import scipy as sp

from .explicit_equations_full import \
    (form_rpa_a_matrix_mo_singlet_full,
     form_rpa_a_matrix_mo_singlet_ss_full,
     form_rpa_a_matrix_mo_singlet_os_full,
     form_rpa_a_matrix_mo_triplet_full,
     form_rpa_b_matrix_mo_singlet_full,
     form_rpa_b_matrix_mo_singlet_ss_full,
     form_rpa_b_matrix_mo_singlet_os_full,
     form_rpa_b_matrix_mo_triplet_full)
from .explicit_equations_partial import \
    (form_rpa_a_matrix_mo_singlet_partial,
     form_rpa_a_matrix_mo_singlet_ss_partial,
     form_rpa_a_matrix_mo_singlet_os_partial,
     form_rpa_a_matrix_mo_triplet_partial,
     form_rpa_b_matrix_mo_singlet_partial,
     form_rpa_b_matrix_mo_singlet_ss_partial,
     form_rpa_b_matrix_mo_singlet_os_partial,
     form_rpa_b_matrix_mo_triplet_partial)


class Solver:

    def __init__(self, mocoeffs, moenergies, occupations, *args, **kwargs):

        assert len(mocoeffs.shape) == 3
        assert (mocoeffs.shape[0] == 1) or (mocoeffs.shape[0] == 2)
        self.is_uhf = (mocoeffs.shape[0] == 2)
        assert len(moenergies.shape) == 3
        assert (moenergies.shape[0] == 1) or (moenergies.shape[0] == 2)
        if self.is_uhf:
            assert moenergies.shape[0] == 2
        else:
            assert moenergies.shape[0] == 1
        assert moenergies.shape[1] == moenergies.shape[2]
        assert len(occupations) == 4

        self.mocoeffs = mocoeffs
        self.moenergies = moenergies
        self.occupations = occupations
        self.form_ranges_from_occupations()

        self.operators = []
        self.frequencies = []

        self.hamiltonian = 'rpa'
        self.spin = 'singlet'

        # These are needed for MO-based solvers.
        self.tei_mo = None
        self.tei_mo_type = 'partial'
        self.explicit_hessian = None
        self.explicit_hessian_inv = None

    def form_tei_mo(self, pyscfmol, tei_mo_type='partial'):
        assert tei_mo_type in ('partial', 'full')
        nden = self.mocoeffs.shape[0]
        assert nden in (1, 2)
        from .ao2mo import AO2MOpyscf
        ao2mo = AO2MOpyscf(self.mocoeffs, pyscfmol.verbose, pyscfmol)
        if tei_mo_type == 'partial' and nden == 2:
            ao2mo.perform_uhf_partial()
        elif tei_mo_type == 'partial' and nden == 1:
            ao2mo.perform_rhf_partial()
        elif tei_mo_type == 'full' and nden == 2:
            ao2mo.perform_uhf_full()
        elif tei_mo_type == 'full' and nden == 1:
            ao2mo.perform_rhf_full()
        else:
            # TODO more specific exception
            raise Exception
        self.tei_mo = ao2mo.tei_mo
        self.tei_mo_type = tei_mo_type

    def form_ranges_from_occupations(self):
        assert len(self.occupations) == 4
        nocc_a, nvirt_a, nocc_b, nvirt_b = self.occupations
        assert (nocc_a + nvirt_a) == (nocc_b + nvirt_b)
        norb = nocc_a + nvirt_a
        nelec = nocc_a + nocc_b
        nact = abs(int(nocc_a - nocc_b))
        nclosed = (nelec - nact) // 2
        nsecondary = norb - (nclosed + nact)
        range_closed = list(range(0, nclosed))
        range_act = list(range(nclosed, nclosed + nact))
        range_secondary = list(range(nclosed + nact, nclosed + nact + nsecondary))
        self.indices_closed_act = [(i, t) for i in range_closed for t in range_act]
        self.indices_closed_secondary = [(i, a) for i in range_closed for a in range_secondary]
        self.indices_act_secondary = [(t, a) for t in range_act for a in range_secondary]
        self.indices_rohf = self.indices_closed_act + self.indices_closed_secondary + self.indices_act_secondary
        self.indices_display_rohf = [(p+1, q+1) for (p, q) in self.indices_rohf]

    def set_frequencies(self, frequencies=None):
        if frequencies is None:
            self.frequencies = [0.0]
        else:
            self.frequencies = frequencies
        if hasattr(self, 'operators'):
            for operator in self.operators:
                operator.frequencies = self.frequencies

    def add_operator(self, operator):
        # First dimension is the number of Cartesian components, next
        # two are the number of AOs.
        assert hasattr(operator, 'ao_integrals')
        shape = operator.ao_integrals.shape
        assert len(shape) == 3
        assert shape[0] >= 1
        assert shape[1] == shape[2]
        operator.indices_closed_act = self.indices_closed_act
        operator.indices_closed_secondary = self.indices_closed_secondary
        operator.indices_act_secondary = self.indices_act_secondary
        # Form the property gradient.
        operator.form_rhs(self.mocoeffs, self.occupations)
        self.operators.append(operator)

    def form_explicit_hessian(self, hamiltonian=None, spin=None, frequency=None):
        raise NotImplementedError


class LineqSolver(Solver):
    def __init__(self, mocoeffs, moenergies, occupations, *args, **kwargs):
        super().__init__(mocoeffs, moenergies, occupations, *args, **kwargs)


class ExactLineqSolver(LineqSolver):
    def __init__(self, mocoeffs, moenergies, occupations, *args, **kwargs):
        super().__init__(mocoeffs, moenergies, occupations, *args, **kwargs)

    def form_explicit_hessian(self, hamiltonian=None, spin=None, frequency=None):

        assert hasattr(self, 'tei_mo')
        assert self.tei_mo is not None
        assert len(self.tei_mo) in (1, 2, 4, 6)
        assert self.tei_mo_type in ('full', 'partial')

        if not hamiltonian:
            hamiltonian = self.hamiltonian
        if not spin:
            spin = self.spin
        if not frequency:
            frequency = 0.0

        assert hamiltonian in ('rpa', 'tda')
        assert spin in ('singlet', 'triplet')

        self.hamiltonian = hamiltonian
        self.spin = spin

        nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = self.occupations
        nov_alph = nocc_alph * nvirt_alph
        nov_beta = nocc_beta * nvirt_beta

        superoverlap_alph = np.block([[np.eye(nov_alph), np.zeros(shape=(nov_alph, nov_alph))],
                                      [np.zeros(shape=(nov_alph, nov_alph)), -np.eye(nov_alph)]])
        superoverlap_alph = superoverlap_alph * frequency

        if not self.is_uhf:

            # Set up "function pointers".
            if self.tei_mo_type == 'full':
                assert len(self.tei_mo) == 1
                tei_mo = self.tei_mo[0]
            elif self.tei_mo_type == 'partial':
                assert len(self.tei_mo) == 2
                tei_mo_ovov = self.tei_mo[0]
                tei_mo_oovv = self.tei_mo[1]

            if self.tei_mo_type == 'full':
                if hamiltonian == 'rpa' and spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_full(self.moenergies[0], tei_mo, nocc_alph)
                    B = form_rpa_b_matrix_mo_singlet_full(tei_mo, nocc_alph)
                elif hamiltonian == 'rpa' and spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_full(self.moenergies[0], tei_mo, nocc_alph)
                    B = form_rpa_b_matrix_mo_triplet_full(tei_mo, nocc_alph)
                elif hamiltonian == 'tda' and spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_full(self.moenergies[0], tei_mo, nocc_alph)
                    B = np.zeros(shape=(nov_alph, nov_alph))
                elif hamiltonian == 'tda' and spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_full(self.moenergies[0], tei_mo, nocc_alph)
                    B = np.zeros(shape=(nov_alph, nov_alph))
            elif self.tei_mo_type == 'partial':
                if hamiltonian == 'rpa' and spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_partial(self.moenergies[0], tei_mo_ovov, tei_mo_oovv)
                    B = form_rpa_b_matrix_mo_singlet_partial(tei_mo_ovov)
                elif hamiltonian == 'rpa' and spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_partial(self.moenergies[0], tei_mo_oovv)
                    B = form_rpa_b_matrix_mo_triplet_partial(tei_mo_ovov)
                elif hamiltonian == 'tda' and spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_partial(self.moenergies[0], tei_mo_ovov, tei_mo_oovv)
                    B = np.zeros(shape=(nov_alph, nov_alph))
                elif hamiltonian == 'tda' and spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_partial(self.moenergies[0], tei_mo_oovv)
                    B = np.zeros(shape=(nov_alph, nov_alph))

            G = np.block([[A, B],
                          [B, A]])
            self.explicit_hessian = G - superoverlap_alph

        else:
            # For UHF there are both "operator-dependent" and
            # operator-indepenent parts of the orbital Hessian because
            # the opposite-spin property gradient couples in. Here,
            # only form the 4 blocks of the "super-Hessian" (a
            # supermatrix of supermatrices); the equations will get
            # pieced together when it is time ot form the response
            # vectors.

            # Set up "function pointers".
            if self.tei_mo_type == 'full':
                assert len(self.tei_mo) == 4
                tei_mo_aaaa = self.tei_mo[0]
                tei_mo_aabb = self.tei_mo[1]
                tei_mo_bbaa = self.tei_mo[2]
                tei_mo_bbbb = self.tei_mo[3]
            elif self.tei_mo_type == 'partial':
                assert len(self.tei_mo) == 6
                tei_mo_ovov_aaaa = self.tei_mo[0]
                tei_mo_ovov_aabb = self.tei_mo[1]
                tei_mo_ovov_bbaa = self.tei_mo[2]
                tei_mo_ovov_bbbb = self.tei_mo[3]
                tei_mo_oovv_aaaa = self.tei_mo[4]
                tei_mo_oovv_bbbb = self.tei_mo[5]
            else:
                pass

            E_a = self.moenergies[0]
            E_b = self.moenergies[1]

            # TODO clean this up...
            if self.tei_mo_type == 'full':
                if hamiltonian == 'rpa' and spin == 'singlet':
                    A_ss_a = form_rpa_a_matrix_mo_singlet_ss_full(E_a, tei_mo_aaaa, nocc_alph)
                    A_os_a = form_rpa_a_matrix_mo_singlet_os_full(tei_mo_aabb, nocc_alph, nocc_beta)
                    B_ss_a = form_rpa_b_matrix_mo_singlet_ss_full(tei_mo_aaaa, nocc_alph)
                    B_os_a = form_rpa_b_matrix_mo_singlet_os_full(tei_mo_aabb, nocc_alph, nocc_beta)
                    A_ss_b = form_rpa_a_matrix_mo_singlet_ss_full(E_b, tei_mo_bbbb, nocc_beta)
                    A_os_b = form_rpa_a_matrix_mo_singlet_os_full(tei_mo_bbaa, nocc_beta, nocc_alph)
                    B_ss_b = form_rpa_b_matrix_mo_singlet_ss_full(tei_mo_bbbb, nocc_beta)
                    B_os_b = form_rpa_b_matrix_mo_singlet_os_full(tei_mo_bbaa, nocc_beta, nocc_alph)
                elif hamiltonian == 'rpa' and spin == 'triplet':
                    # Since the "triplet" part contains no Coulomb contribution, and
                    # (xx|yy) is only in the Coulomb part, there is no ss/os
                    # separation for the triplet part.
                    zeros_ab = np.zeros(shape=(nov_alph, nov_beta))
                    zeros_ba = zeros_ab.T
                    A_ss_a = form_rpa_a_matrix_mo_triplet_full(E_a, tei_mo_aaaa, nocc_alph)
                    A_os_a = zeros_ab
                    B_ss_a = form_rpa_b_matrix_mo_triplet_full(tei_mo_aaaa, nocc_alph)
                    B_os_a = zeros_ab
                    A_ss_b = form_rpa_a_matrix_mo_triplet_full(E_b, tei_mo_bbbb, nocc_beta)
                    A_os_b = zeros_ba
                    B_ss_b = form_rpa_b_matrix_mo_triplet_full(tei_mo_bbbb, nocc_beta)
                    B_os_b = zeros_ba
                elif hamiltonian == 'tda' and spin == 'singlet':
                    zeros_aa = np.zeros(shape=(nov_alph, nov_alph))
                    zeros_ab = np.zeros(shape=(nov_alph, nov_beta))
                    zeros_ba = zeros_ab.T
                    zeros_bb = np.zeros(shape=(nov_beta, nov_beta))
                    A_ss_a = form_rpa_a_matrix_mo_singlet_ss_full(E_a, tei_mo_aaaa, nocc_alph)
                    A_os_a = form_rpa_a_matrix_mo_singlet_os_full(tei_mo_aabb, nocc_alph, nocc_beta)
                    B_ss_a = zeros_aa
                    B_os_a = zeros_ab
                    A_ss_b = form_rpa_a_matrix_mo_singlet_ss_full(E_b, tei_mo_bbbb, nocc_beta)
                    A_os_b = form_rpa_a_matrix_mo_singlet_os_full(tei_mo_bbaa, nocc_beta, nocc_alph)
                    B_ss_b = zeros_bb
                    B_os_b = zeros_ba
                elif hamiltonian == 'tda' and spin == 'triplet':
                    zeros_aa = np.zeros(shape=(nov_alph, nov_alph))
                    zeros_ab = np.zeros(shape=(nov_alph, nov_beta))
                    zeros_ba = zeros_ab.T
                    zeros_bb = np.zeros(shape=(nov_beta, nov_beta))
                    A_ss_a = form_rpa_a_matrix_mo_triplet_full(E_a, tei_mo_aaaa, nocc_alph)
                    A_os_a = zeros_ab
                    B_ss_a = zeros_aa
                    B_os_a = zeros_ab
                    A_ss_b = form_rpa_a_matrix_mo_triplet_full(E_b, tei_mo_bbbb, nocc_beta)
                    A_os_b = zeros_ba
                    B_ss_b = zeros_bb
                    B_os_b = zeros_ba

            elif self.tei_mo_type == 'partial':
                if hamiltonian == 'rpa' and spin == 'singlet':
                    A_ss_a = form_rpa_a_matrix_mo_singlet_ss_partial(E_a, tei_mo_ovov_aaaa, tei_mo_oovv_aaaa)
                    A_os_a = form_rpa_a_matrix_mo_singlet_os_partial(tei_mo_ovov_aabb)
                    B_ss_a = form_rpa_b_matrix_mo_singlet_ss_partial(tei_mo_ovov_aaaa)
                    B_os_a = form_rpa_b_matrix_mo_singlet_os_partial(tei_mo_ovov_aabb)
                    A_ss_b = form_rpa_a_matrix_mo_singlet_ss_partial(E_b, tei_mo_ovov_bbbb, tei_mo_oovv_bbbb)
                    A_os_b = form_rpa_a_matrix_mo_singlet_os_partial(tei_mo_ovov_bbaa)
                    B_ss_b = form_rpa_b_matrix_mo_singlet_ss_partial(tei_mo_ovov_bbbb)
                    B_os_b = form_rpa_b_matrix_mo_singlet_os_partial(tei_mo_ovov_bbaa)
                elif hamiltonian == 'rpa' and spin == 'triplet':
                    # Since the "triplet" part contains no Coulomb contribution, and
                    # (xx|yy) is only in the Coulomb part, there is no ss/os
                    # separation for the triplet part.
                    zeros_ab = np.zeros(shape=(nov_alph, nov_beta))
                    zeros_ba = zeros_ab.T
                    A_ss_a = form_rpa_a_matrix_mo_triplet_partial(E_a, tei_mo_oovv_aaaa)
                    A_os_a = zeros_ab
                    B_ss_a = form_rpa_b_matrix_mo_triplet_partial(tei_mo_ovov_aaaa)
                    B_os_a = zeros_ab
                    A_ss_b = form_rpa_a_matrix_mo_triplet_partial(E_b, tei_mo_oovv_bbbb)
                    A_os_b = zeros_ba
                    B_ss_b = form_rpa_b_matrix_mo_triplet_partial(tei_mo_ovov_bbbb)
                    B_os_b = zeros_ba
                elif hamiltonian == 'tda' and spin == 'singlet':
                    zeros_aa = np.zeros(shape=(nov_alph, nov_alph))
                    zeros_ab = np.zeros(shape=(nov_alph, nov_beta))
                    zeros_ba = zeros_ab.T
                    zeros_bb = np.zeros(shape=(nov_beta, nov_beta))
                    A_ss_a = form_rpa_a_matrix_mo_singlet_ss_partial(E_a, tei_mo_ovov_aaaa, tei_mo_oovv_aaaa)
                    A_os_a = form_rpa_a_matrix_mo_singlet_os_partial(tei_mo_ovov_aabb)
                    B_ss_a = zeros_aa
                    B_os_a = zeros_ab
                    A_ss_b = form_rpa_a_matrix_mo_singlet_ss_partial(E_b, tei_mo_ovov_bbbb, tei_mo_oovv_bbbb)
                    A_os_b = form_rpa_a_matrix_mo_singlet_os_partial(tei_mo_ovov_bbaa)
                    B_ss_b = zeros_bb
                    B_os_b = zeros_ba
                elif hamiltonian == 'tda' and spin == 'triplet':
                    zeros_aa = np.zeros(shape=(nov_alph, nov_alph))
                    zeros_ab = np.zeros(shape=(nov_alph, nov_beta))
                    zeros_ba = zeros_ab.T
                    zeros_bb = np.zeros(shape=(nov_beta, nov_beta))
                    A_ss_a = form_rpa_a_matrix_mo_triplet_partial(E_a, tei_mo_oovv_aaaa)
                    A_os_a = zeros_ab
                    B_ss_a = zeros_aa
                    B_os_a = zeros_ab
                    A_ss_b = form_rpa_a_matrix_mo_triplet_partial(E_b, tei_mo_oovv_bbbb)
                    A_os_b = zeros_ba
                    B_ss_b = zeros_bb
                    B_os_b = zeros_ba

            superoverlap_beta = np.block([[np.eye(nov_beta), np.zeros(shape=(nov_beta, nov_beta))],
                                          [np.zeros(shape=(nov_beta, nov_beta)), -np.eye(nov_beta)]])
            superoverlap_beta = superoverlap_beta * frequency

            G_aa = np.block([[A_ss_a, B_ss_a],
                             [B_ss_a, A_ss_a]])
            G_ab = np.block([[A_os_a, B_os_a],
                             [B_os_a, A_os_a]])
            G_ba = np.block([[A_os_b, B_os_b],
                             [B_os_b, A_os_b]])
            G_bb = np.block([[A_ss_b, B_ss_b],
                             [B_ss_b, A_ss_b]])

            self.explicit_hessian = (
                G_aa - superoverlap_alph,
                G_ab,
                G_ba,
                G_bb - superoverlap_beta,
            )

    def form_response_vectors(self):
        if self.is_uhf:
            G_aa, G_ab, G_ba, G_bb = self.explicit_hessian
            G_aa_inv, G_bb_inv = self.explicit_hessian_inv
        for operator in self.operators:
            if not self.is_uhf:
                rspvecs_operator = []
                for idx_operator_component in range(operator.ao_integrals.shape[0]):
                    shape = operator.mo_integrals_ai_supervector_alph[idx_operator_component].shape
                    assert len(shape) == 2
                    assert shape[1] == 1
                    rspvec_operator_component = np.dot(self.explicit_hessian_inv,
                                                       operator.mo_integrals_ai_supervector_alph[idx_operator_component])
                    assert rspvec_operator_component.shape == shape
                    rspvecs_operator.append(rspvec_operator_component)
                # TODO this isn't working and I don't know why
                # rspvecs_operator = np.stack(rspvecs_operator, axis=0)
                # All the lines with 'tmp' could be replaced by a working
                # stack call.
                tmp = np.empty(shape=(len(rspvecs_operator), shape[0], 1),
                               dtype=rspvec_operator_component.dtype)
                for idx, rspvec_operator_component in enumerate(rspvecs_operator):
                    tmp[idx] = rspvec_operator_component
                rspvecs_operator = tmp
                operator.rspvecs_alph.append(rspvecs_operator)
            else:
                # Form the operator-dependent part of the response vectors.
                rspvecs_operator_alph = []
                rspvecs_operator_beta = []
                for idx_operator_component in range(operator.ao_integrals.shape[0]):
                    operator_component_alph = operator.mo_integrals_ai_supervector_alph[idx_operator_component]
                    operator_component_beta = operator.mo_integrals_ai_supervector_beta[idx_operator_component]
                    shape_alph = operator_component_alph.shape
                    shape_beta = operator_component_beta.shape
                    assert len(shape_alph) == len(shape_beta) == 2
                    assert shape_alph[1] == shape_beta[1] == 1
                    right_alph = operator_component_alph - (np.dot(G_ab, np.dot(G_bb_inv, operator_component_beta)))
                    right_beta = operator_component_beta - (np.dot(G_ba, np.dot(G_aa_inv, operator_component_alph)))
                    assert right_alph.shape == shape_alph
                    assert right_beta.shape == shape_beta
                    rspvecs_operator_alph.append(np.dot(self.left_alph, right_alph))
                    rspvecs_operator_beta.append(np.dot(self.left_beta, right_beta))
                tmp_alph = np.empty(shape=(len(rspvecs_operator_alph), shape_alph[0], 1),
                                    dtype=operator_component_alph.dtype)
                tmp_beta = np.empty(shape=(len(rspvecs_operator_beta), shape_beta[0], 1),
                                    dtype=operator_component_beta.dtype)
                for idx_alph, operator_component_alph in enumerate(rspvecs_operator_alph):
                    tmp_alph[idx_alph] = operator_component_alph
                for idx_beta, operator_component_beta in enumerate(rspvecs_operator_beta):
                    tmp_beta[idx_beta] = operator_component_beta
                rspvecs_operator_alph = tmp_alph
                rspvecs_operator_beta = tmp_beta
                operator.rspvecs_alph.append(rspvecs_operator_alph)
                operator.rspvecs_beta.append(rspvecs_operator_beta)


class ExactInv(ExactLineqSolver):

    def __init__(self, mocoeffs, moenergies, occupations, *args, **kwargs):
        super().__init__(mocoeffs, moenergies, occupations, *args, **kwargs)

        inv_func = kwargs.get('inv_func')
        if inv_func:
            self.inv_func = inv_func
        else:
            self.inv_func = np.linalg.inv

    def invert_explicit_hessian(self):
        assert hasattr(self, 'explicit_hessian')
        if not self.is_uhf:
            self.explicit_hessian_inv = self.inv_func(self.explicit_hessian)
        else:
            assert len(self.explicit_hessian) == 4
            self.explicit_hessian_inv = []
            G_aa, G_ab, G_ba, G_bb = self.explicit_hessian
            # The inverse of the opposite-spin blocks is not
            # necessary.
            G_aa_inv = self.inv_func(G_aa)
            G_bb_inv = self.inv_func(G_bb)
            self.explicit_hessian_inv.append(G_aa_inv)
            self.explicit_hessian_inv.append(G_bb_inv)
            # Form the operator-independent part of the response
            # vectors.
            self.left_alph = self.inv_func(G_aa - np.dot(G_ab, np.dot(G_bb_inv, G_ba)))
            self.left_beta = self.inv_func(G_bb - np.dot(G_ba, np.dot(G_aa_inv, G_ab)))


class ExactInvCholesky(ExactLineqSolver):

    def __init__(self, mocoeffs, moenergies, occupations, *args, **kwargs):
        super().__init__(mocoeffs, moenergies, occupations, *args, **kwargs)

    def invert_explicit_hessian(self):
        assert hasattr(self, 'explicit_hessian')
        if not self.is_uhf:
            fac = sp.linalg.cholesky(self.explicit_hessian)
            fac_inv = sp.linalg.inv(fac)
            inv = np.dot(fac_inv, fac_inv.T)
            self.explicit_hessian_inv = inv
        else:
            assert len(self.explicit_hessian) == 4
            self.explicit_hessian_inv = []
            G_aa, G_ab, G_ba, G_bb = self.explicit_hessian
            # The inverse of the opposite-spin blocks is not
            # necessary.
            G_aa_R_inv = sp.linalg.inv(sp.linalg.cholesky(G_aa, lower=False))
            G_bb_R_inv = sp.linalg.inv(sp.linalg.cholesky(G_bb, lower=False))
            G_aa_inv = np.dot(G_aa_R_inv, G_aa_R_inv.T)
            G_bb_inv = np.dot(G_bb_R_inv, G_bb_R_inv.T)
            self.explicit_hessian_inv.append(G_aa_inv)
            self.explicit_hessian_inv.append(G_bb_inv)
            # Form the operator-independent part of the response
            # vectors.
            left_alph_R = sp.linalg.cholesky(G_aa - np.dot(G_ab, np.dot(G_bb_inv, G_ba)), lower=False)
            left_beta_R = sp.linalg.cholesky(G_bb - np.dot(G_ba, np.dot(G_aa_inv, G_ab)), lower=False)
            left_alph_R_inv = sp.linalg.inv(left_alph_R)
            left_beta_R_inv = sp.linalg.inv(left_beta_R)
            self.left_alph = np.dot(left_alph_R_inv, left_alph_R_inv.T)
            self.left_beta = np.dot(left_beta_R_inv, left_beta_R_inv.T)


class EigSolver(Solver):

    def __init__(self, mocoeffs, moenergies, occupations, *args, **kwargs):
        super().__init__(mocoeffs, moenergies, occupations, *args, **kwargs)

    @staticmethod
    def norm_xy(z, nocc, nvirt):
        x, y = z.reshape(2, nvirt, nocc)
        norm = 2 * (np.linalg.norm(x)**2 - np.linalg.norm(y)**2)
        norm = 1 / np.sqrt(norm)
        return (x*norm).flatten(), (y*norm).flatten()

class EigSolverTDA(EigSolver):

    def __init__(self, mocoeffs, moenergies, occupations, *args, **kwargs):
        super().__init__(mocoeffs, moenergies, occupations, *args, **kwargs)


class ExactDiagonalizationSolver(EigSolver):

    def __init__(self, mocoeffs, moenergies, occupations, *args, **kwargs):
        super().__init__(mocoeffs, moenergies, occupations, *args, **kwargs)

    def form_explicit_hessian(self, hamiltonian=None, spin=None, frequency=None):

        assert hasattr(self, 'tei_mo')
        assert self.tei_mo is not None
        assert len(self.tei_mo) in (1, 2, 4, 6)
        assert self.tei_mo_type in ('full', 'partial')

        if not hamiltonian:
            hamiltonian = self.hamiltonian
        if not spin:
            spin = self.spin

        assert isinstance(hamiltonian, str)
        assert isinstance(spin, str)
        hamiltonian = hamiltonian.lower()
        spin = spin.lower()

        assert hamiltonian in ('rpa', 'tda')
        assert spin in ('singlet', 'triplet')

        self.hamiltonian = hamiltonian
        self.spin = spin

        nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = self.occupations
        nov_alph = nocc_alph * nvirt_alph
        nov_beta = nocc_beta * nvirt_beta

        if not self.is_uhf:

            # Set up "function pointers".
            if self.tei_mo_type == 'full':
                assert len(self.tei_mo) == 1
                tei_mo = self.tei_mo[0]
            elif self.tei_mo_type == 'partial':
                assert len(self.tei_mo) == 2
                tei_mo_ovov = self.tei_mo[0]
                tei_mo_oovv = self.tei_mo[1]

            if self.tei_mo_type == 'full':
                if hamiltonian == 'rpa' and spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_full(self.moenergies[0], tei_mo, nocc_alph)
                    B = form_rpa_b_matrix_mo_singlet_full(tei_mo, nocc_alph)
                elif hamiltonian == 'rpa' and spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_full(self.moenergies[0], tei_mo, nocc_alph)
                    B = form_rpa_b_matrix_mo_triplet_full(tei_mo, nocc_alph)
                elif hamiltonian == 'tda' and spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_full(self.moenergies[0], tei_mo, nocc_alph)
                    B = np.zeros(shape=(nov_alph, nov_alph))
                elif hamiltonian == 'tda' and spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_full(self.moenergies[0], tei_mo, nocc_alph)
                    B = np.zeros(shape=(nov_alph, nov_alph))
            elif self.tei_mo_type == 'partial':
                if hamiltonian == 'rpa' and spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_partial(self.moenergies[0], tei_mo_ovov, tei_mo_oovv)
                    B = form_rpa_b_matrix_mo_singlet_partial(tei_mo_ovov)
                elif hamiltonian == 'rpa' and spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_partial(self.moenergies[0], tei_mo_oovv)
                    B = form_rpa_b_matrix_mo_triplet_partial(tei_mo_ovov)
                elif hamiltonian == 'tda' and spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_partial(self.moenergies[0], tei_mo_ovov, tei_mo_oovv)
                    B = np.zeros(shape=(nov_alph, nov_alph))
                elif hamiltonian == 'tda' and spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_partial(self.moenergies[0], tei_mo_oovv)
                    B = np.zeros(shape=(nov_alph, nov_alph))

            # pylint: disable=bad-whitespace
            G = np.block([[ A,  B],
                          [-B, -A]])
            self.explicit_hessian = G

        else:
            # TODO UHF
            pass

    def diagonalize_explicit_hessian(self):
        nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = self.occupations
        nov_alph = nocc_alph * nvirt_alph
        nov_beta = nocc_beta * nvirt_beta
        if not self.is_uhf:
            eigvals, eigvecs = sp.linalg.eig(self.explicit_hessian)
            # Sort from lowest to highest eigenvalue (excitation
            # energy).
            idx = eigvals.argsort()
            self.eigvals = eigvals[idx]
            # Each eigenvector is a column vector.
            self.eigvecs = eigvecs[:, idx]
            # Fix the ordering of everything. The first eigenvectors
            # are those with negative excitation energies.
            self.eigvals = self.eigvals[nov_alph:]
            self.eigvecs = self.eigvecs[:, nov_alph:]
        else:
            # TODO UHF
            pass


class ExactDiagonalizationSolverTDA(ExactDiagonalizationSolver, EigSolverTDA):

    def __init__(self, mocoeffs, moenergies, occupations, *args, **kwargs):
        super().__init__(mocoeffs, moenergies, occupations, *args, **kwargs)

    def form_explicit_hessian(self, hamiltonian=None, spin=None, frequency=None):

        assert hasattr(self, 'tei_mo')
        assert self.tei_mo is not None
        assert len(self.tei_mo) in (1, 2, 4, 6)
        assert self.tei_mo_type in ('full', 'partial')

        if not hamiltonian:
            hamiltonian = self.hamiltonian
        if not spin:
            spin = self.spin

        assert isinstance(hamiltonian, str)
        assert isinstance(spin, str)
        hamiltonian = hamiltonian.lower()
        spin = spin.lower()

        assert hamiltonian == 'tda'
        assert spin in ('singlet', 'triplet')

        self.hamiltonian = hamiltonian
        self.spin = spin

        nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = self.occupations
        nov_alph = nocc_alph * nvirt_alph
        nov_beta = nocc_beta * nvirt_beta

        if not self.is_uhf:

            # Set up "function pointers".
            if self.tei_mo_type == 'full':
                assert len(self.tei_mo) == 1
                tei_mo = self.tei_mo[0]
            elif self.tei_mo_type == 'partial':
                assert len(self.tei_mo) == 2
                tei_mo_ovov = self.tei_mo[0]
                tei_mo_oovv = self.tei_mo[1]

            if self.tei_mo_type == 'full':
                if spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_full(self.moenergies[0], tei_mo, nocc_alph)
                elif spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_full(self.moenergies[0], tei_mo, nocc_alph)
            elif self.tei_mo_type == 'partial':
                if spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_partial(self.moenergies[0], tei_mo_ovov, tei_mo_oovv)
                elif spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_partial(self.moenergies[0], tei_mo_oovv)

            self.explicit_hessian = A

        else:
            # TODO UHF
            pass

    def diagonalize_explicit_hessian(self):
        nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = self.occupations
        nov_alph = nocc_alph * nvirt_alph
        nov_beta = nocc_beta * nvirt_beta
        if not self.is_uhf:
            eigvals, eigvecs = sp.linalg.eig(self.explicit_hessian)
            # Sort from lowest to highest eigenvalue (excitation
            # energy).
            idx = eigvals.argsort()
            self.eigvals = eigvals[idx]
            # Each eigenvector is a column vector.
            self.eigvecs = eigvecs[:, idx]
        else:
            # TODO UHF
            pass
