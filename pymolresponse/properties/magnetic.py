from typing import Union

import numpy as np

from pymolresponse.core import Program
from pymolresponse.cphf import CPHF
from pymolresponse.interfaces.pyscf.helpers import calculate_origin_pyscf
from pymolresponse.molecular_property import ResponseProperty
from pymolresponse.operators import Operator


class Magnetizability(ResponseProperty):
    def __init__(
        self,
        program: Program,
        program_obj,
        driver: CPHF,
        mocoeffs: np.ndarray,
        moenergies: np.ndarray,
        occupations: np.ndarray,
        use_giao: bool = False,
    ) -> None:
        super().__init__(
            program,
            program_obj,
            driver,
            mocoeffs,
            moenergies,
            occupations,
            frequencies=np.asarray([0.0]),
        )
        self.use_giao = use_giao

    def form_operators(self) -> None:
        if self.program == Program.PySCF:
            from pymolresponse.interfaces.pyscf import integrals

            integral_generator = integrals.IntegralsPyscf(self.program_obj)
        elif self.program == Program.Psi4:
            from pymolresponse.interfaces.psi4 import integrals

            integral_generator = integrals.IntegralsPsi4(self.program_obj)
        else:
            raise RuntimeError

        if self.use_giao:
            integrals_angmom_ao = integral_generator.integrals(integrals.ANGMOM_GIAO)
        else:
            integrals_angmom_ao = integral_generator.integrals(integrals.ANGMOM_COMMON_GAUGE)
        operator_angmom = Operator(
            label="angmom", is_imaginary=True, is_spin_dependent=False, triplet=False
        )
        operator_angmom.ao_integrals = integrals_angmom_ao
        self.driver.add_operator(operator_angmom)

    def form_results(self) -> None:
        assert len(self.driver.results) == 1
        operator_angmom = self.driver.solver.operators[0]  # noqa: F841
        self.magnetizability = (1 / 4) * self.driver.results[0]
        # print('paramagnetic part of magnetic susceptibility/magnetizability, no GIAO, Cartesian origin')
        # print(self.magnetizability)


class ElectronicGTensor(ResponseProperty):
    def __init__(
        self,
        program: Program,
        program_obj,
        driver: CPHF,
        mocoeffs: np.ndarray,
        moenergies: np.ndarray,
        occupations: np.ndarray,
        *,
        gauge_origin: Union[str, np.ndarray] = "ecc",
    ) -> None:
        super().__init__(
            program, program_obj, driver, mocoeffs, moenergies, occupations, frequencies=[0.0]
        )

        if program == Program.PySCF:
            assert isinstance(gauge_origin, (str, list, tuple, np.ndarray))
            if isinstance(gauge_origin, str):
                coords = program_obj.atom_coords()
                charges = program_obj.atom_charges()
                is_uhf = mocoeffs.shape[0] == 2
                if is_uhf:
                    Ca = mocoeffs[0]
                    Cb = mocoeffs[1]
                    nocc_a, _, nocc_b, _ = occupations
                    Da = np.dot(Ca[:, :nocc_a], Ca[:, :nocc_a].T)
                    Db = np.dot(Cb[:, :nocc_b], Cb[:, :nocc_b].T)
                    D = Da + Db
                else:
                    C = mocoeffs[0]
                    nocc_a, _, _, _ = occupations
                    D = 2 * np.dot(C[:, :nocc_a], C[:, :nocc_a].T)
                self.gauge_origin = calculate_origin_pyscf(
                    gauge_origin, coords, charges, D, program_obj, do_print=True
                )
            else:
                assert len(gauge_origin) == 3
                if isinstance(gauge_origin, np.ndarray):
                    assert gauge_origin.flatten().shape == (3,)
                self.gauge_origin = np.asarray(gauge_origin)
        else:
            raise RuntimeError

    def form_operators(self) -> None:
        if self.program == Program.PySCF:
            from pymolresponse.interfaces.pyscf import integrals

            integral_generator = integrals.IntegralsPyscf(self.program_obj)
        elif self.program == Program.Psi4:
            from pymolresponse.interfaces.psi4 import integrals

            integral_generator = integrals.IntegralsPsi4(self.program_obj)
        else:
            raise RuntimeError

        # angular momentum
        operator_angmom = Operator(
            label="angmom", is_imaginary=True, is_spin_dependent=False, triplet=False
        )
        self.program_obj.set_common_orig(self.gauge_origin)
        operator_angmom.ao_integrals = integral_generator.integrals(integrals.ANGMOM_COMMON_GAUGE)
        self.driver.add_operator(operator_angmom)

        # spin-orbit (1-electron, exact nuclear charges)
        operator_spinorb = Operator(
            label="spinorb", is_imaginary=True, is_spin_dependent=False, triplet=False
        )
        integrals_spinorb_ao = 0
        for atm_id in range(self.program_obj.natm):
            self.program_obj.set_rinv_orig(self.program_obj.atom_coord(atm_id))
            chg = self.program_obj.atom_charge(atm_id)
            integrals_spinorb_ao += chg * integral_generator.integrals(integrals.SO_SPHER_1e)
        operator_spinorb.ao_integrals = integrals_spinorb_ao
        self.driver.add_operator(operator_spinorb)

        # spin-orbit (1-electron, effective nuclear charges)
        operator_spinorb_eff = Operator(
            label="spinorb_eff", is_imaginary=True, is_spin_dependent=False, triplet=False
        )
        integrals_spinorb_eff_ao = 0
        for atm_id in range(self.program_obj.natm):
            self.program_obj.set_rinv_orig(self.program_obj.atom_coord(atm_id))
            # chg = self.program_obj.atom_effective_charge[atm_id]
            chg = 0
            integrals_spinorb_eff_ao += chg * integral_generator.integrals(integrals.SO_SPHER_1e)
        operator_spinorb_eff.ao_integrals = integrals_spinorb_eff_ao
        self.driver.add_operator(operator_spinorb_eff)

    def form_results(self) -> None:
        operator_angmom = self.driver.solver.operators[0]  # noqa: F841
        # angmom_grad_alph = operator_angmom.mo_integrals_ai_supervector_alph
        # print(angmom_grad_alph[0, :, 0])
        # angmom_resp_alph = operator_angmom.rspvecs_alph[0]
        # angmom_resp_beta = operator_angmom.rspvecs_beta[0]
        # print(angmom_resp_alph.shape)
        # print(np.linalg.norm(angmom_resp_alph[0, :, 0]))
        # print(angmom_resp_beta.shape)
        # print(np.linalg.norm(angmom_resp_beta[0, :, 0]))
        operator_spinorb = self.driver.solver.operators[1]  # noqa: F841
        operator_spinorb_eff = self.driver.solver.operators[2]  # noqa: F841

        np_formatter = {"float_kind": lambda x: "{:14.8f}".format(x)}  # noqa: F841
        # np.set_printoptions(linewidth=200, formatter=np_formatter)
        assert len(self.driver.results) == 1
        results = self.driver.results[0]
        assert results.shape == (9, 9)
        block_1 = results[0:3, 0:3]  # angmom/angmom  # noqa: F841
        block_2 = results[0:3, 3:6]  # angmom/spinorb
        block_3 = results[0:3, 6:9]  # angmom/spinorb_eff
        block_4 = results[3:6, 0:3]  # spinorb/angmom  # noqa: F841
        block_5 = results[3:6, 3:6]  # spinorb/spinorb  # noqa: F841
        block_6 = results[3:6, 6:9]  # spinorb/spinorb_eff  # noqa: F841
        block_7 = results[6:9, 0:3]  # spinorb_eff/angmom  # noqa: F841
        block_8 = results[6:9, 3:6]  # spinorb_eff/spinorb  # noqa: F841
        block_9 = results[6:9, 6:9]  # spinorb_eff/spinorb_eff  # noqa: F841

        nalph, nbeta = self.program_obj.nelec
        exact_spin = 0.5 * (nalph - nbeta)
        res_1 = block_2 / exact_spin
        res_2 = (block_3 - block_2) / exact_spin
        res = res_1 + res_2  # noqa: F841

        # principal values are sqrt(eigvals(g.T * g)
        prin_1 = np.sqrt(np.linalg.eigvals(np.dot(res_1.T, res_1)))

        self.g_oz_soc_1 = res_1
        self.g_oz_soc_1_eig = prin_1
