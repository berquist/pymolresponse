from typing import Optional

import numpy as np
import scipy.constants as spc

from pymolresponse.utils import fix_mocoeffs_shape, repack_matrix_to_vector


class Operator:
    """Handle property integrals, taking them from the AO basis to a
    representation of a right-hand side perturbation for CPHF or
    transition properties."""

    def __init__(
        self,
        label: str = "",
        is_imaginary: bool = False,
        is_spin_dependent: bool = False,
        triplet: bool = False,
        slice_idx: int = -1,
        ao_integrals: Optional[np.ndarray] = None,
    ) -> None:
        self.label = label
        self.is_imaginary = is_imaginary
        self.is_spin_dependent = is_spin_dependent
        self.triplet = triplet
        # TODO In general, this is not used outside of referencing a
        # specific operator component from DALTON.
        self.slice_idx = slice_idx

        if "spinorb" in label:
            self.hsofac = (spc.alpha**2) / 4

        self.frequencies = None
        self.ao_integrals = ao_integrals
        self.rspvecs_alph = []
        self.rspvecs_beta = []

    def __str__(self) -> str:
        return (
            f'Operator(label="{self.label}", is_imaginary={self.is_imaginary}, '
            f"is_spin_dependent={self.is_spin_dependent}, triplet={self.triplet}, "
            f"slice_idx={self.slice_idx})"
        )

    def calculate_ao_integrals(self) -> None:
        pass

    def form_rhs(self, C: np.ndarray, occupations: np.ndarray) -> None:
        """Form the right-hand side for CPHF."""
        assert isinstance(self.ao_integrals, np.ndarray)
        if len(C.shape) == 2:
            C = C[np.newaxis]
        assert len(C.shape) == 3
        assert (C.shape[0] == 1) or (C.shape[0] == 2)
        is_uhf = C.shape[0] == 2
        C_alph = C[0]
        if is_uhf:
            C_beta = C[1]
        assert len(occupations) == 4
        nocc_alph, _, nocc_beta, _ = occupations
        b_prefactor = -1
        if self.is_imaginary:
            b_prefactor = +1
        operator_ai_alph = []
        operator_ai_supervector_alph = []
        operator_ai_beta = []
        operator_ai_supervector_beta = []
        # Loop over the operator components (usually multiple
        # Cartesian directions).
        # pylint: disable=no-member
        for idx in range(self.ao_integrals.shape[0]):
            operator_component_ai_alph = np.dot(
                C_alph[:, nocc_alph:].T, np.dot(self.ao_integrals[idx], C_alph[:, :nocc_alph])
            )
            # If the operator is a triplet operator and doing singlet
            # response, remove inactive -> secondary excitations.
            # Is this only true for spin-orbit operators?
            if self.triplet:
                for i, a in self.indices_closed_secondary:
                    operator_component_ai_alph[a - nocc_alph, i] = 0.0
            operator_component_ai_alph = repack_matrix_to_vector(operator_component_ai_alph)[
                :, np.newaxis
            ]
            if hasattr(self, "hsofac"):
                operator_component_ai_alph *= self.hsofac
            operator_component_ai_supervector_alph = np.concatenate(
                (operator_component_ai_alph, operator_component_ai_alph * b_prefactor), axis=0
            )
            operator_ai_alph.append(operator_component_ai_alph)
            operator_ai_supervector_alph.append(operator_component_ai_supervector_alph)
            if is_uhf:
                operator_component_ai_beta = np.dot(
                    C_beta[:, nocc_beta:].T, np.dot(self.ao_integrals[idx], C_beta[:, :nocc_beta])
                )
                if self.triplet:
                    for i, a in self.indices_closed_secondary:
                        operator_component_ai_beta[a - nocc_beta, i] = 0.0
                operator_component_ai_beta = repack_matrix_to_vector(operator_component_ai_beta)[
                    :, np.newaxis
                ]
                if hasattr(self, "hsofac"):
                    operator_component_ai_beta *= self.hsofac
                operator_component_ai_supervector_beta = np.concatenate(
                    (operator_component_ai_beta, operator_component_ai_beta * b_prefactor), axis=0
                )
                operator_ai_beta.append(operator_component_ai_beta)
                operator_ai_supervector_beta.append(operator_component_ai_supervector_beta)
        self.mo_integrals_ai_alph = np.stack(operator_ai_alph, axis=0)
        self.mo_integrals_ai_supervector_alph = np.stack(operator_ai_supervector_alph, axis=0)
        if is_uhf:
            self.mo_integrals_ai_beta = np.stack(operator_ai_beta, axis=0)
            self.mo_integrals_ai_supervector_beta = np.stack(operator_ai_supervector_beta, axis=0)

    def form_rhs_geometric(
        self,
        C: np.ndarray,
        occupations: np.ndarray,
        natoms,
        MO_full,
        mints,
        return_dict: bool = False,
    ) -> None:
        from pymolresponse.integrals import _form_rhs_geometric

        C_ = fix_mocoeffs_shape(C)
        B_dict = _form_rhs_geometric(C_[0], occupations, natoms, MO_full, mints)
        if return_dict:
            return B_dict
        # TODO I think this is wrong, because the matrices are [i, a]
        B_matrices = [B_dict[k].T for k in sorted(B_dict.keys())]
        B_vectors = [repack_matrix_to_vector(B)[:, np.newaxis] for B in B_matrices]
        mo_integrals_ai_alph = np.stack(B_vectors)
        self.mo_integrals_ai_alph = mo_integrals_ai_alph
        # pylint: disable=invalid-unary-operand-type
        self.mo_integrals_ai_supervector_alph = np.concatenate(
            (mo_integrals_ai_alph, -mo_integrals_ai_alph), axis=1
        )
