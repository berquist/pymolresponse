#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.constants as spc

from utils import (form_results, np_load, parse_int_file_2,
                   repack_matrix_to_vector)
from explicit_equations_full import \
    (form_rpa_a_matrix_mo_singlet_full,
     form_rpa_a_matrix_mo_singlet_ss_full,
     form_rpa_a_matrix_mo_singlet_os_full,
     form_rpa_a_matrix_mo_triplet_full,
     form_rpa_b_matrix_mo_singlet_full,
     form_rpa_b_matrix_mo_singlet_ss_full,
     form_rpa_b_matrix_mo_singlet_os_full,
     form_rpa_b_matrix_mo_triplet_full)
from explicit_equations_partial import \
    (form_rpa_a_matrix_mo_singlet_partial,
     form_rpa_a_matrix_mo_singlet_ss_partial,
     form_rpa_a_matrix_mo_singlet_os_partial,
     form_rpa_a_matrix_mo_triplet_partial,
     form_rpa_b_matrix_mo_singlet_partial,
     form_rpa_b_matrix_mo_singlet_ss_partial,
     form_rpa_b_matrix_mo_singlet_os_partial,
     form_rpa_b_matrix_mo_triplet_partial)


class Operator(object):

    def __init__(self, label='', is_imaginary=False, is_spin_dependent=False, slice_idx=-1):
        self.label = label
        # TODO In general, this is not used outside of referencing a
        # specific operator component from DALTON.
        self.slice_idx = slice_idx
        self.is_imaginary = is_imaginary
        self.is_spin_dependent = is_spin_dependent

        if 'spinorb' in label:
            self.hsofac = (spc.alpha ** 2) / 4

        self.frequencies = None
        self.rspvecs_alph = []
        self.rspvecs_beta = []

    def form_rhs(self, C, occupations):
        if len(C.shape) == 2:
            C = C[np.newaxis, ...]
        assert len(C.shape) == 3
        assert (C.shape[0] == 1) or (C.shape[0] == 2)
        is_uhf = (C.shape[0] == 2)
        C_alph = C[0, ...]
        if is_uhf:
            C_beta = C[1, ...]
        assert len(occupations) == 4
        nocc_alph, _, nocc_beta, _ = occupations
        b_prefactor = 1
        if self.is_imaginary:
            b_prefactor = -1
        operator_ai_alph = []
        operator_ai_supervector_alph = []
        operator_ai_beta = []
        operator_ai_supervector_beta = []
        # Loop over the operator components (usually multiple
        # Cartesian directions).
        for idx in range(self.ao_integrals.shape[0]):
            operator_component_ai_alph = np.dot(C_alph[:, nocc_alph:].T, np.dot(self.ao_integrals[idx, ...], C_alph[:, :nocc_alph]))
            # If the operator is a triplet operator and doing singlet
            # response, remove inactive -> secondary excitations.
            # Is this only true for spin-orbit operators?
            # if self.is_spin_dependent:
            #     for (i, a) in self.indices_closed_secondary:
            #         operator_component_ai_alph[a - nocc_alph, i] = 0.0
            operator_component_ai_alph = repack_matrix_to_vector(operator_component_ai_alph)[:, np.newaxis]
            if hasattr(self, 'hsofac'):
                operator_component_ai_alph *= self.hsofac
            operator_component_ai_supervector_alph = np.concatenate((operator_component_ai_alph,
                                                                     operator_component_ai_alph * b_prefactor), axis=0)
            operator_ai_alph.append(operator_component_ai_alph)
            operator_ai_supervector_alph.append(operator_component_ai_supervector_alph)
            if is_uhf:
                operator_component_ai_beta = np.dot(C_beta[:, nocc_beta:].T, np.dot(self.ao_integrals[idx, ...], C_beta[:, :nocc_beta]))
                # if self.is_spin_dependent:
                #     for (i, a) in self.indices_closed_secondary:
                #         operator_component_ai_beta[a - nocc_beta, i] = 0.0
                operator_component_ai_beta = repack_matrix_to_vector(operator_component_ai_beta)[:, np.newaxis]
                if hasattr(self, 'hsofac'):
                    operator_component_ai_beta *= self.hsofac
                operator_component_ai_supervector_beta = np.concatenate((operator_component_ai_beta,
                                                                         operator_component_ai_beta * b_prefactor), axis=0)
                operator_ai_beta.append(operator_component_ai_beta)
                operator_ai_supervector_beta.append(operator_component_ai_supervector_beta)
        self.mo_integrals_ai_alph = np.stack(operator_ai_alph, axis=0)
        self.mo_integrals_ai_supervector_alph = np.stack(operator_ai_supervector_alph, axis=0)
        if is_uhf:
            self.mo_integrals_ai_beta = np.stack(operator_ai_beta, axis=0)
            self.mo_integrals_ai_supervector_beta = np.stack(operator_ai_supervector_beta, axis=0)


class CPHF(object):

    def __init__(self, mocoeffs, moenergies, occupations):
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

        self.solver = 'explicit'
        self.hamiltonian = 'rpa'
        self.spin = 'singlet'

        self.operators = []
        self.frequencies = []
        self.results = []

        self.tei_mo = None
        self.tei_mo_type = 'partial'
        self.explicit_hessian = None
        self.explicit_hessian_inv = None

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

    def set_frequencies(self, frequencies=None):
        if frequencies is None:
            self.frequencies = [0.0]
        else:
            self.frequencies = frequencies
        for operator in self.operators:
            operator.frequencies = self.frequencies

    def add_operator(self, operator):
        # First dimension is the number of Cartesian components, next
        # two are the number of AOs.
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

    def run(self, solver=None, hamiltonian=None, spin=None):

        if not solver:
            solver = self.solver
        if not hamiltonian:
            hamiltonian = self.hamiltonian
        if not spin:
            spin = self.spin
        if not hasattr(self, 'frequencies'):
            self.frequencies = [0.0]

        # Set the current state.
        self.solver = solver
        self.hamiltonian = hamiltonian
        self.spin = spin

        if solver == 'explicit':
            for frequency in self.frequencies:
                self.form_explicit_hessian(hamiltonian, spin, frequency)
                self.invert_explicit_hessian()
                self.form_response_vectors()
        # Nothing else implemented yet.
        else:
            pass

        self.form_results()

        for operator in self.operators:
            operator.rspvecs_alph = []
            operator.rspvecs_beta = []

    def form_explicit_hessian(self, hamiltonian=None, spin=None, frequency=None):

        # TODO blow up
        if not hasattr(self, 'tei_mo'):
            pass
        elif not self.tei_mo:
            pass

        if not hamiltonian:
            hamiltonian = self.hamiltonian
        if not spin:
            spin = self.spin
        if not frequency:
            frequency = 0.0

        if hamiltonian not in ('rpa', 'tda'):
            pass
        if spin not in ('singlet', 'triplet'):
            pass

        self.hamiltonian = hamiltonian
        self.spin = spin

        nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = self.occupations
        nov_alph = nocc_alph * nvirt_alph
        nov_beta = nocc_beta * nvirt_beta

        superoverlap = np.asarray(np.bmat([[np.eye(nov_alph), np.zeros(shape=(nov_alph, nov_alph))],
                                           [np.zeros(shape=(nov_alph, nov_alph)), -np.eye(nov_alph)]]))
        superoverlap = superoverlap * frequency

        if not self.is_uhf:

            # Set up "function pointers".
            if self.tei_mo_type == 'full':
                assert len(self.tei_mo) == 1
                tei_mo = self.tei_mo[0]
            elif self.tei_mo_type == 'partial':
                assert len(self.tei_mo) == 2
                tei_mo_ovov = self.tei_mo[0]
                tei_mo_oovv = self.tei_mo[1]
            else:
                # TODO blow up
                pass

            if self.tei_mo_type == 'full':
                if hamiltonian == 'rpa' and spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_full(self.moenergies[0, ...], tei_mo, nocc_alph)
                    B = form_rpa_b_matrix_mo_singlet_full(tei_mo, nocc_alph)
                elif hamiltonian == 'rpa' and spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_full(self.moenergies[0, ...], tei_mo, nocc_alph)
                    B = form_rpa_b_matrix_mo_triplet_full(tei_mo, nocc_alph)
                elif hamiltonian == 'tda' and spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_full(self.moenergies[0, ...], tei_mo, nocc_alph)
                    B = np.zeros(shape=(nov_alph, nov_alph))
                elif hamiltonian == 'tda' and spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_full(self.moenergies[0, ...], tei_mo, nocc_alph)
                    B = np.zeros(shape=(nov_alph, nov_alph))
            elif self.tei_mo_type == 'partial':
                if hamiltonian == 'rpa' and spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_partial(self.moenergies[0, ...], tei_mo_ovov, tei_mo_oovv)
                    B = form_rpa_b_matrix_mo_singlet_partial(tei_mo_ovov)
                elif hamiltonian == 'rpa' and spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_partial(self.moenergies[0, ...], tei_mo_oovv)
                    B = form_rpa_b_matrix_mo_triplet_partial(tei_mo_ovov)
                elif hamiltonian == 'tda' and spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_partial(self.moenergies[0, ...], tei_mo_ovov, tei_mo_oovv)
                    B = np.zeros(shape=(nov_alph, nov_alph))
                elif hamiltonian == 'tda' and spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_partial(self.moenergies[0, ...], tei_mo_oovv)
                    B = np.zeros(shape=(nov_alph, nov_alph))

            G = np.asarray(np.bmat([[A, B],
                                    [B, A]]))
            self.explicit_hessian = G - superoverlap

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
                # TODO blow up
                pass
                
            E_a = self.moenergies[0, ...]
            E_b = self.moenergies[1, ...]

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

            # TODO blow up
            else:
                pass

            G_aa = np.asarray(np.bmat([[A_ss_a, B_ss_a],
                                       [B_ss_a, A_ss_a]]))
            G_ab = np.asarray(np.bmat([[A_os_a, B_os_a],
                                       [B_os_a, A_os_a]]))
            G_ba = np.asarray(np.bmat([[A_os_b, B_os_b],
                                       [B_os_b, A_os_b]]))
            G_bb = np.asarray(np.bmat([[A_ss_b, B_ss_b],
                                       [B_ss_b, A_ss_b]]))

            self.explicit_hessian = [G_aa, G_ab, G_ba, G_bb]

    def invert_explicit_hessian(self):
        if not self.is_uhf:
            self.explicit_hessian_inv = np.linalg.inv(self.explicit_hessian)
        else:
            assert len(self.explicit_hessian) == 4
            self.explicit_hessian_inv = []
            # The inverse of the opposite-spin blocks is not necessary.
            G_aa, _, _, G_bb = self.explicit_hessian
            self.explicit_hessian_inv.append(np.linalg.inv(G_aa))
            self.explicit_hessian_inv.append(np.linalg.inv(G_bb))

    def form_response_vectors(self):
        if self.is_uhf:
            G_aa, G_ab, G_ba, G_bb = self.explicit_hessian
            G_aa_inv, G_bb_inv = self.explicit_hessian_inv
            # Form the operator-independent part of the response vectors.
            left_alph = np.linalg.inv(G_aa - np.dot(G_ab, np.dot(G_bb_inv, G_ba)))
            left_beta = np.linalg.inv(G_bb - np.dot(G_ba, np.dot(G_aa_inv, G_ab)))
        for operator in self.operators:
            if not self.is_uhf:
                rspvecs_operator = []
                for idx_operator_component in range(operator.ao_integrals.shape[0]):
                    shape = operator.mo_integrals_ai_supervector_alph[idx_operator_component, ...].shape
                    assert len(shape) == 2
                    assert shape[1] == 1
                    rspvec_operator_component = np.dot(self.explicit_hessian_inv,
                                                       operator.mo_integrals_ai_supervector_alph[idx_operator_component, ...])
                    assert rspvec_operator_component.shape == shape
                    rspvecs_operator.append(rspvec_operator_component)
                # TODO this isn't working and I don't know why
                # rspvecs_operator = np.stack(rspvecs_operator, axis=0)
                # All the lines with 'tmp' could be replaced by a working
                # stack call.
                tmp = np.empty(shape=(len(rspvecs_operator), shape[0], 1),
                               dtype=rspvec_operator_component.dtype)
                for idx, rspvec_operator_component in enumerate(rspvecs_operator):
                    tmp[idx, ...] = rspvec_operator_component
                rspvecs_operator = tmp
                operator.rspvecs_alph.append(rspvecs_operator)
            else:
                # Form the operator-dependent part of the response vectors.
                rspvecs_operator_alph = []
                rspvecs_operator_beta = []
                for idx_operator_component in range(operator.ao_integrals.shape[0]):
                    operator_component_alph = operator.mo_integrals_ai_supervector_alph[idx_operator_component, ...]
                    operator_component_beta = operator.mo_integrals_ai_supervector_beta[idx_operator_component, ...]
                    shape_alph = operator_component_alph.shape
                    shape_beta = operator_component_beta.shape
                    assert len(shape_alph) == len(shape_beta) == 2
                    assert shape_alph[1] == shape_beta[1] == 1
                    right_alph = operator_component_alph - (np.dot(G_ab, np.dot(G_bb_inv, operator_component_beta)))
                    right_beta = operator_component_beta - (np.dot(G_ba, np.dot(G_aa_inv, operator_component_alph)))
                    assert right_alph.shape == shape_alph
                    assert right_beta.shape == shape_beta
                    rspvecs_operator_alph.append(np.dot(left_alph, right_alph))
                    rspvecs_operator_beta.append(np.dot(left_beta, right_beta))
                tmp_alph = np.empty(shape=(len(rspvecs_operator_alph), shape_alph[0], 1),
                                    dtype=operator_component_alph.dtype)
                tmp_beta = np.empty(shape=(len(rspvecs_operator_beta), shape_beta[0], 1),
                                    dtype=operator_component_beta.dtype)
                for idx_alph, operator_component_alph in enumerate(rspvecs_operator_alph):
                    tmp_alph[idx_alph, ...] = operator_component_alph
                for idx_beta, operator_component_beta in enumerate(rspvecs_operator_beta):
                    tmp_beta[idx_beta, ...] = operator_component_beta
                rspvecs_operator_alph = tmp_alph
                rspvecs_operator_beta = tmp_beta
                operator.rspvecs_alph.append(rspvecs_operator_alph)
                operator.rspvecs_beta.append(rspvecs_operator_beta)

    def form_results(self):

        self.results = []

        # TODO change now that Operators keep their own rspvecs
        # self.rspvecs structure:
        # 1. list, index is for frequencies
        # 2. list, index is for operators
        # 3. numpy.ndarray, shape is [a,b,c]
        #   a -> number of operator components
        #   b -> actual matrix elements
        #   c -> 1 (always)

        for operator in self.operators:
            assert len(self.frequencies) == len(operator.rspvecs_alph)
            if self.is_uhf:
                assert len(self.frequencies) == len(operator.rspvecs_beta)
            else:
                assert len(operator.rspvecs_beta) == 0

        for f in range(len(self.frequencies)):
            for i in range(len(self.operators)):
                assert self.operators[i].rspvecs_alph[f].shape == self.operators[i].mo_integrals_ai_supervector_alph.shape
                if self.is_uhf:
                    assert self.operators[i].rspvecs_beta[f].shape == self.operators[i].mo_integrals_ai_supervector_beta.shape

        for f in range(len(self.frequencies)):

            # dim_rows -> (number of operators) * (number of components for each operator)
            # dim_cols -> total number of response vectors
            dim_rows = sum(self.operators[i].mo_integrals_ai_supervector_alph.shape[0]
                           for i in range(len(self.operators)))
            dim_cols = sum(self.operators[i].rspvecs_alph[f].shape[0]
                           for i in range(len(self.operators)))
            assert dim_rows == dim_cols

            # FIXME
            results = np.zeros(shape=(dim_rows, dim_cols),
                               dtype=self.operators[0].rspvecs_alph[f].dtype)

            # Form the result blocks between each pair of
            # operators. Ignore any potential symmetry in the final
            # result matrix for now.
            result_blocks = []
            row_starts = []
            col_starts = []
            row_start = 0
            for iop1, op1 in enumerate(self.operators):
                col_start = 0
                for iop2, op2 in enumerate(self.operators):
                    result_block = 0.0
                    result_block += form_results(op1.mo_integrals_ai_supervector_alph, op2.rspvecs_alph[f])
                    if self.is_uhf:
                        result_block += form_results(op1.mo_integrals_ai_supervector_beta, op2.rspvecs_beta[f])
                    result_blocks.append(result_block)
                    row_starts.append(row_start)
                    col_starts.append(col_start)
                    col_start += op2.rspvecs_alph[f].shape[0]
                row_start += op1.mo_integrals_ai_supervector_alph.shape[0]

            # Put each of the result blocks back in the main results
            # matrix.
            assert len(row_starts) == len(col_starts) == len(result_blocks)
            for idx, result_block in enumerate(result_blocks):
                nr, nc = result_block.shape
                rs, cs = row_starts[idx], col_starts[idx]
                results[rs:rs+nr, cs:cs+nc] = result_block

            # The / 2 is because of the supervector part.
            if self.is_uhf:
                results = 2 * results / 2
            else:
                results = 4 * results / 2
            self.results.append(results)



if __name__ == '__main__':

    C = np_load('C.npz')
    C = C[np.newaxis, ...]
    E = np_load('F_MO.npz')
    E = E[np.newaxis, ...]
    TEI_MO = np_load('TEI_MO.npz')
    # nocc_alph, nvirt_alph, nocc_beta, nvirt_beta
    occupations = [5, 2, 5, 2]
    stub = 'h2o_sto3g_'
    dim = occupations[0] + occupations[1]
    mat_dipole_x = parse_int_file_2(stub + "mux.dat", dim)
    mat_dipole_y = parse_int_file_2(stub + "muy.dat", dim)
    mat_dipole_z = parse_int_file_2(stub + "muz.dat", dim)

    cphf = CPHF(C, E, occupations)
    ao_integrals_dipole = np.empty(shape=(3, dim, dim))
    ao_integrals_dipole[0, :, :] = mat_dipole_x
    ao_integrals_dipole[1, :, :] = mat_dipole_y
    ao_integrals_dipole[2, :, :] = mat_dipole_z
    operator_dipole = Operator(label='dipole', is_imaginary=False, is_spin_dependent=False)
    operator_dipole.ao_integrals = ao_integrals_dipole
    cphf.add_operator(operator_dipole)
    frequencies = (0.0, 0.02, 0.06, 0.1)
    cphf.set_frequencies(frequencies)
    cphf.tei_mo = TEI_MO
    for hamiltonian in ('rpa', 'tda'):
        for spin in ('singlet', 'triplet'):
            print('hamiltonian: {}, spin: {}'.format(hamiltonian, spin))
            cphf.run(solver='explicit', hamiltonian=hamiltonian, spin=spin)
            print(len(cphf.results))
            for f, frequency in enumerate(frequencies):
                print('frequency: {}'.format(frequency))
                print(cphf.results[f])
