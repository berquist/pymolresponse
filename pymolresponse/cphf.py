"""Driver for solving the coupled perturbed Hartree-Fock (CPHF) equations."""

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import numpy as np

from pymolresponse.core import Hamiltonian, Program, Spin
from pymolresponse.operators import Operator
from pymolresponse.solvers import Solver
from pymolresponse.utils import form_results, form_vec_energy_differences


class Driver(ABC):
    def __init__(self, solver: Solver) -> None:
        assert isinstance(solver, Solver)
        self.solver = solver

        self.results: List[np.ndarray] = []

    @abstractmethod
    def run(self, hamiltonian: Hamiltonian, spin: Spin, program: Program, program_obj) -> None:
        pass


class CPHF(Driver):
    """Driver for solving the coupled perturbed Hartree-Fock (CPHF) equations."""

    def __init__(self, solver: Solver) -> None:
        assert isinstance(solver, Solver)
        super().__init__(solver)

    def set_frequencies(self, frequencies: Optional[Sequence[float]] = None) -> None:
        r"""Set the frequencies :math:`\omega_f` for which frequency-dependent
        CPHF is performed."""

        # :type frequencies float or list
        # :param frequencies one or more frequencies in atomic units;
        # if None, do the static case (0.0)
        assert self.solver is not None
        self.solver.set_frequencies(frequencies)
        self.frequencies = self.solver.frequencies

    def add_operator(self, operator: Operator) -> None:
        """Add an operator to the list of operators that will be used as the
        right-hand side perturbation."""

        assert self.solver is not None
        self.solver.add_operator(operator)

    def run(self, hamiltonian: Hamiltonian, spin: Spin, program: Program, program_obj) -> None:
        assert isinstance(hamiltonian, Hamiltonian)
        assert isinstance(spin, Spin)
        assert isinstance(program, (Program, type(None)))
        # TODO program_obj

        if not hasattr(self, "frequencies"):
            self.set_frequencies([0.0])

        self.solver.run(hamiltonian, spin, program, program_obj)
        self.form_uncoupled_results()
        self.form_results()

    def form_uncoupled_results(self) -> None:
        # We avoid the formation of the full Hessian, but the energy
        # differences on the diagonal are still needed. Form them
        # here. The dynamic frequency contribution will be handled
        # inside the main loop.
        nocc_a, _, nocc_b, _ = self.solver.occupations
        moene_alph = np.diag(self.solver.moenergies[0])
        moene_occ_alph = moene_alph[:nocc_a]
        moene_virt_alph = moene_alph[nocc_a:]
        ediff_alph = form_vec_energy_differences(moene_occ_alph, moene_virt_alph)
        ediff_supervector_alph = np.concatenate((ediff_alph, ediff_alph))
        ediff_supervector_alph_static = ediff_supervector_alph[..., np.newaxis]
        nov_alph = len(ediff_alph)
        if self.solver.is_uhf:
            moene_beta = np.diag(self.solver.moenergies[1])
            moene_occ_beta = moene_beta[:nocc_b]
            moene_virt_beta = moene_beta[nocc_b:]
            ediff_beta = form_vec_energy_differences(moene_occ_beta, moene_virt_beta)
            ediff_supervector_beta = np.concatenate((ediff_beta, ediff_beta))
            ediff_supervector_beta_static = ediff_supervector_beta[..., np.newaxis]
            nov_beta = len(ediff_beta)

        self.uncoupled_results = []

        for f in range(len(self.solver.frequencies)):
            frequency = self.solver.frequencies[f]
            ediff_supervector_alph = ediff_supervector_alph_static.copy()
            ediff_supervector_alph[:nov_alph] = ediff_supervector_alph_static[:nov_alph] - frequency
            ediff_supervector_alph[nov_alph:] = ediff_supervector_alph_static[nov_alph:] + frequency
            if self.solver.is_uhf:
                ediff_supervector_beta = ediff_supervector_beta_static.copy()
                ediff_supervector_beta[:nov_beta] = (
                    ediff_supervector_beta_static[:nov_beta] - frequency
                )
                ediff_supervector_beta[nov_beta:] = (
                    ediff_supervector_beta_static[nov_beta:] + frequency
                )

            # dim_rows -> (number of operators) * (number of components for each operator)
            # dim_cols -> total number of response vectors
            dim_rows = sum(
                self.solver.operators[i].mo_integrals_ai_supervector_alph.shape[0]
                for i in range(len(self.solver.operators))
            )
            dim_cols = dim_rows

            # FIXME, though the reason is weak: the dtype is being inferred
            # from a single operator on the solver, when in reality the dtype
            # of all results should be considered.
            results = np.zeros(
                shape=(dim_rows, dim_cols),
                dtype=self.solver.operators[0].mo_integrals_ai_supervector_alph.dtype,
            )

            # Form the result blocks between each pair of
            # operators. Ignore any potential symmetry in the final
            # result matrix for now.
            result_blocks = []
            row_starts = []
            col_starts = []
            row_start = 0
            for op1 in self.solver.operators:
                col_start = 0
                for op2 in self.solver.operators:
                    result_block = 0.0
                    result_block_alph = form_results(
                        op1.mo_integrals_ai_supervector_alph,
                        op2.mo_integrals_ai_supervector_alph / ediff_supervector_alph,
                    )
                    result_block += result_block_alph
                    if self.solver.is_uhf:
                        result_block_beta = form_results(
                            op1.mo_integrals_ai_supervector_beta,
                            op2.mo_integrals_ai_supervector_beta / ediff_supervector_beta,
                        )
                        result_block += result_block_beta
                    result_blocks.append(result_block)
                    row_starts.append(row_start)
                    col_starts.append(col_start)
                    col_start += op2.mo_integrals_ai_supervector_alph.shape[0]
                row_start += op1.mo_integrals_ai_supervector_alph.shape[0]

            # Put each of the result blocks back in the main results
            # matrix.
            assert len(row_starts) == len(col_starts) == len(result_blocks)
            for idx, result_block in enumerate(result_blocks):
                nr, nc = result_block.shape
                rs, cs = row_starts[idx], col_starts[idx]
                results[rs : rs + nr, cs : cs + nc] = result_block

            # The / 2 is because of the supervector part.
            if self.solver.is_uhf:
                results = 2 * results / 2
            else:
                results = 4 * results / 2
            self.uncoupled_results.append(results)

    def form_results(self) -> None:
        self.results = []

        # TODO change now that Operators keep their own rspvecs
        # self.rspvecs structure:
        # 1. list, index is for frequencies
        # 2. list, index is for operators
        # 3. numpy.ndarray, shape is [a,b,c]
        #   a -> number of operator components
        #   b -> actual matrix elements
        #   c -> 1 (always)

        for operator in self.solver.operators:
            assert len(self.solver.frequencies) == len(operator.rspvecs_alph)
            if self.solver.is_uhf:
                assert len(self.solver.frequencies) == len(operator.rspvecs_beta)
            else:
                assert len(operator.rspvecs_beta) == 0

        for f in range(len(self.solver.frequencies)):
            for i in range(len(self.solver.operators)):
                assert (
                    self.solver.operators[i].rspvecs_alph[f].shape
                    == self.solver.operators[i].mo_integrals_ai_supervector_alph.shape
                )
                if self.solver.is_uhf:
                    assert (
                        self.solver.operators[i].rspvecs_beta[f].shape
                        == self.solver.operators[i].mo_integrals_ai_supervector_beta.shape
                    )

        for f in range(len(self.solver.frequencies)):
            # dim_rows -> (number of operators) * (number of components for each operator)
            # dim_cols -> total number of response vectors
            dim_rows = sum(
                self.solver.operators[i].mo_integrals_ai_supervector_alph.shape[0]
                for i in range(len(self.solver.operators))
            )
            dim_cols = sum(
                self.solver.operators[i].rspvecs_alph[f].shape[0]
                for i in range(len(self.solver.operators))
            )
            assert dim_rows == dim_cols

            # FIXME same comment about the dtype as above
            results = np.zeros(
                shape=(dim_rows, dim_cols), dtype=self.solver.operators[0].rspvecs_alph[f].dtype
            )

            # Form the result blocks between each pair of
            # operators. Ignore any potential symmetry in the final
            # result matrix for now.
            result_blocks = []
            row_starts = []
            col_starts = []
            row_start = 0
            for op1 in self.solver.operators:
                col_start = 0
                for op2 in self.solver.operators:
                    result_block = 0.0
                    result_block_alph = form_results(
                        op1.mo_integrals_ai_supervector_alph, op2.rspvecs_alph[f]
                    )
                    result_block += result_block_alph
                    if self.solver.is_uhf:
                        result_block_beta = form_results(
                            op1.mo_integrals_ai_supervector_beta, op2.rspvecs_beta[f]
                        )
                        result_block += result_block_beta
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
                results[rs : rs + nr, cs : cs + nc] = result_block

            # The / 2 is because of the supervector part.
            if self.solver.is_uhf:
                results = 2 * results / 2
            else:
                results = 4 * results / 2
            self.results.append(results)
