#!/usr/bin/env python3.6

from __future__ import print_function
from __future__ import division

import numpy as np

from utils import (np_load, parse_int_file_2,
                   form_rpa_a_matrix_mo_singlet, form_rpa_a_matrix_mo_triplet,
                   form_rpa_b_matrix_mo_singlet, form_rpa_b_matrix_mo_triplet)


# TODO could probably just do this with a reshape...
def repack_matrix_to_vector(mat):
    nrows, ncols = mat.shape
    vec = np.empty((nrows * ncols))
    for a in range(nrows):
        for i in range(ncols):
            ia = (i * nrows) + a
            vec[ia] = mat[a, i]
    return vec


def form_results(vecs_property, vecs_response):
    assert vecs_property.shape[1:] == vecs_response.shape[1:]
    assert len(vecs_property.shape) == 3
    assert vecs_property.shape[2] == 1
    # nr = vecs_property.shape[0]
    # nc = vecs_response.shape[0]
    # results = np.empty(shape=(nr, nc))
    # for a in range(nr):
    #     for b in range(nc):
    #         results[a, b] = np.dot(vecs_property[a, :, 0], vecs_response[b, :, 0])
    # print(results)
    results = np.dot(vecs_property[:, :, 0], vecs_response[:, :, 0].T)
    return results


class Operator(object):

    def __init__(self, is_imaginary=False, is_spin_dependent=False):
        self.is_imaginary = is_imaginary
        self.is_spin_dependent = is_spin_dependent


class CPHF(object):

    def __init__(self, mocoeffs, moenergies, occupations):
        self.mocoeffs = mocoeffs
        self.moenergies = moenergies
        self.occupations = occupations

        self.solver = 'direct'
        self.hamiltonian = 'rpa'
        self.spin = 'singlet'

        self.operators = []
        self.rspvecs = []
        self.results = []

        self.TEI_MO = None
        self.explicit_hessian = None
        self.explicit_hessian_inv = None

    def set_frequencies(self, frequencies=None):
        if not frequencies:
            self.frequencies = [0.0]
        else:
            self.frequencies = frequencies

    def add_operator(self, operator):
        b_prefactor = 1
        if operator.is_imaginary:
            b_prefactor = -1
        # RHF only for now
        nocc = self.occupations[0]
        shape = operator.ao_integrals.shape
        # First dimension is the number of Cartesian components, next
        # two are the number of AOs.
        assert len(shape) == 3
        assert shape[0] >= 1
        assert shape[1] == shape[2]
        operator_ai = []
        operator_ai_supervector = []
        # Loop over the operator components (usually multiple
        # Cartesian directions).
        for idx in range(operator.ao_integrals.shape[0]):
            operator_component_ai = np.dot(self.mocoeffs[:, nocc:].T, np.dot(operator.ao_integrals[idx, :, :], self.mocoeffs[:, :nocc]))
            operator_component_ai = repack_matrix_to_vector(operator_component_ai)[:, np.newaxis]
            operator_component_ai_supervector = np.concatenate((operator_component_ai,
                                                                operator_component_ai * b_prefactor), axis=0)
            operator_ai.append(operator_component_ai)
            operator_ai_supervector.append(operator_component_ai_supervector)
        operator.mo_integrals_ai = np.stack(operator_ai, axis=0)
        operator.mo_integrals_ai_supervector = np.stack(operator_ai_supervector, axis=0)
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

        if solver == 'direct':
            for frequency in self.frequencies:
                self.form_explicit_hessian(hamiltonian, spin, frequency)
                self.invert_explicit_hessian()
                self.form_response_vectors()
        # Nothing else implemented yet.
        else:
            pass

        self.form_results()
        self.rspvecs = []

    def form_explicit_hessian(self, hamiltonian=None, spin=None, frequency=None):

        # TODO blow up
        # if not self.TEI_MO:
        #     pass

        if not hamiltonian:
            hamiltonian = self.hamiltonian
        if not spin:
            spin = self.spin
        if not frequency:
            frequency = 0.0

        self.hamiltonian = hamiltonian
        self.spin = spin

        nocc = self.occupations[0]
        nvirt = self.occupations[1]
        nov = nocc * nvirt

        superoverlap = np.bmat([[np.eye(nov), np.zeros(shape=(nov, nov))],
                                [np.zeros(shape=(nov, nov)), -np.eye(nov)]])
        superoverlap *= frequency

        if hamiltonian == 'rpa' and spin == 'singlet':
            A_singlet = form_rpa_a_matrix_mo_singlet(self.moenergies, self.TEI_MO, nocc)
            B_singlet = form_rpa_b_matrix_mo_singlet(self.TEI_MO, nocc)
            explicit_hessian = np.bmat([[A_singlet, B_singlet],
                                        [B_singlet, A_singlet]])
            explicit_hessian -= superoverlap
        elif hamiltonian == 'rpa' and spin == 'triplet':
            A_triplet = form_rpa_a_matrix_mo_triplet(self.moenergies, self.TEI_MO, nocc)
            B_triplet = form_rpa_b_matrix_mo_triplet(self.TEI_MO, nocc)
            explicit_hessian = np.bmat([[A_triplet, B_triplet],
                                        [B_triplet, A_triplet]])
            explicit_hessian -= superoverlap
        elif hamiltonian == 'tda' and spin == 'singlet':
            A_singlet = form_rpa_a_matrix_mo_singlet(self.moenergies, self.TEI_MO, nocc)
            B_singlet = np.zeros(shape=(nov, nov))
            explicit_hessian = np.bmat([[A_singlet, B_singlet],
                                        [B_singlet, A_singlet]])
            explicit_hessian -= superoverlap
        elif hamiltonian == 'tda' and spin == 'triplet':
            A_triplet = form_rpa_a_matrix_mo_triplet(self.moenergies, self.TEI_MO, nocc)
            B_triplet = np.zeros(shape=(nov, nov))
            explicit_hessian = np.bmat([[A_triplet, B_triplet],
                                        [B_triplet, A_triplet]])
            explicit_hessian -= superoverlap
        # TODO blow up
        else:
            pass

        self.explicit_hessian = explicit_hessian

    def invert_explicit_hessian(self):
        self.explicit_hessian_inv = np.linalg.inv(self.explicit_hessian)

    def form_response_vectors(self):
        rspvecs = []
        for operator in self.operators:
            rspvecs_operator = []
            for idx_operator_component in range(operator.ao_integrals.shape[0]):
                shape = operator.mo_integrals_ai_supervector[idx_operator_component, ...].shape
                assert len(shape) == 2
                assert shape[1] == 1
                rspvec_operator_component = np.dot(self.explicit_hessian_inv, operator.mo_integrals_ai_supervector[idx_operator_component, ...])
                assert rspvec_operator_component.shape == shape
                rspvecs_operator.append(rspvec_operator_component)
            # TODO this isn't working and I don't know why
            # rspvecs_operator = np.stack(rspvecs_operator, axis=0)
            # All the lines with 'tmp' could be replaced by a working
            # stack call.
            tmp = np.empty(shape=(len(rspvecs_operator), shape[0], 1))
            for idx, rspvec_operator_component in enumerate(rspvecs_operator):
                tmp[idx, ...] = rspvec_operator_component
            rspvecs_operator = tmp
            rspvecs.append(rspvecs_operator)
        self.rspvecs.append(rspvecs)

    def form_results(self):

        self.results = []

        # self.rspvecs structure:
        # 1. list, index is for frequencies
        # 2. list, index is for operators
        # 3. numpy.ndarray, shape is [a,b,c]
        #   a -> number of operator components
        #   b -> actual matrix elements
        #   c -> 1 (always)

        assert len(self.frequencies) == len(self.rspvecs)
        for f in range(len(self.frequencies)):
            assert len(self.rspvecs[f]) == len(self.operators)
            for i in range(len(self.operators)):
                assert self.rspvecs[f][i].shape == self.operators[i].mo_integrals_ai_supervector.shape

        for f in range(len(self.frequencies)):

            # dim_rows -> (number of operators) * (number of components for each operator)
            # dim_cols -> total number of response vectors
            dim_rows = sum(self.operators[i].mo_integrals_ai_supervector.shape[0]
                           for i in range(len(self.operators)))
            dim_cols = sum(self.rspvecs[f][i].shape[0]
                           for i in range(len(self.rspvecs[f])))
            assert dim_rows == dim_cols

            results = np.zeros(shape=(dim_rows, dim_cols))

            # Form the result blocks between each pair of
            # operators. Ignore any potential symmetry in the final
            # result matrix for now.
            result_blocks = []
            row_starts = []
            col_starts = []
            row_start = 0
            for iop1, op1 in enumerate(self.operators):
                col_start = 0
                # Why didn't I make Operators carry their own response vectors? TODO
                # for iop2, op2 in enumerate(self.operators):
                for iop2, op2rsp in enumerate(self.rspvecs[f]):
                    result_block = form_results(op1.mo_integrals_ai_supervector, op2rsp)
                    result_blocks.append(result_block)
                    row_starts.append(row_start)
                    col_starts.append(col_start)
                    col_start += op2rsp.shape[0]
                row_start += op1.mo_integrals_ai_supervector.shape[0]

            # Put each of the result blocks back in the main results
            # matrix.
            assert len(row_starts) == len(col_starts) == len(result_blocks)
            for idx, result_block in enumerate(result_blocks):
                nr, nc = result_block.shape
                rs, cs = row_starts[idx], col_starts[idx]
                results[rs:rs+nr, cs:cs+nc] = result_block

            # The 2 is because of the supervector part.
            results = 4 * results / 2
            self.results.append(results)



if __name__ == '__main__':

    C = np_load('C.npz')
    E = np_load('F_MO.npz')
    TEI_MO = np_load('TEI_MO.npz')
    # nocc_alph, nvirt_alph, nocc_beta, nvirt_beta
    occupations = [5, 2, 5, 2]
    stub = 'h2o_sto3g_'
    dim = occupations[0] + occupations[1]
    mat_dipole_x = parse_int_file_2(stub + "mux.dat", dim)
    mat_dipole_y = parse_int_file_2(stub + "muy.dat", dim)
    mat_dipole_z = parse_int_file_2(stub + "muz.dat", dim)

    cphf = CPHF(C, E, occupations)
    frequencies = (0.0, 0.02, 0.06, 0.1)
    cphf.frequencies = frequencies
    ao_integrals_dipole = np.empty(shape=(3, dim, dim))
    ao_integrals_dipole[0, :, :] = mat_dipole_x
    ao_integrals_dipole[1, :, :] = mat_dipole_y
    ao_integrals_dipole[2, :, :] = mat_dipole_z
    operator_dipole = Operator(is_imaginary=False, is_spin_dependent=False)
    operator_dipole.ao_integrals = ao_integrals_dipole
    cphf.add_operator(operator_dipole)
    cphf.TEI_MO = TEI_MO
    for hamiltonian in ('rpa', 'tda'):
        for spin in ('singlet', 'triplet'):
            print(f'hamiltonian: {hamiltonian}, spin: {spin}')
            cphf.run(solver='direct', hamiltonian=hamiltonian, spin=spin)
            print(len(cphf.results))
            for f, frequency in enumerate(frequencies):
                print(f'frequency: {frequency}')
                print(cphf.results[f])
