import numpy as np
import scipy as sp

import psi4
import pyscf

from pymolresponse import cphf, solvers, utils
from pymolresponse.core import Hamiltonian, Program, Spin
from pymolresponse.interfaces.psi4 import integrals
from pymolresponse.interfaces.psi4 import molecules as molecules_psi4
from pymolresponse.interfaces.psi4.utils import (
    mocoeffs_from_psi4wfn,
    moenergies_from_psi4wfn,
    occupations_from_psi4wfn,
)
from pymolresponse.interfaces.pyscf import molecules as molecules_pyscf
from pymolresponse.interfaces.pyscf.utils import occupations_from_pyscf_mol
from pymolresponse.properties import electric, magnetic


def test_inversion() -> None:
    """Test that each kind of inversion function gives identical results."""

    mol = molecules_pyscf.molecule_glycine_sto3g()
    mol.charge = 1
    mol.spin = 1
    mol.build()

    mf = pyscf.scf.uhf.UHF(mol)
    mf.scf()

    assert isinstance(mf.mo_coeff, np.ndarray)
    assert len(mf.mo_coeff) == 2
    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = occupations_from_pyscf_mol(mol, C)

    calculator_ref = magnetic.Magnetizability(
        Program.PySCF, mol, cphf.CPHF(solvers.ExactInv(C, E, occupations)), C, E, occupations
    )
    calculator_ref.form_operators()
    calculator_ref.run(hamiltonian=Hamiltonian.RPA, spin=Spin.singlet)
    calculator_ref.form_results()

    inv_funcs = (sp.linalg.inv, sp.linalg.pinv)

    thresh = 6.0e-14

    # TODO actually test the different inversion functions...
    for _ in inv_funcs:
        calculator_res = magnetic.Magnetizability(
            Program.PySCF, mol, cphf.CPHF(solvers.ExactInv(C, E, occupations)), C, E, occupations
        )
        calculator_res.form_operators()
        calculator_res.run(hamiltonian=Hamiltonian.RPA, spin=Spin.singlet)
        calculator_res.form_results()

        np.testing.assert_equal(
            np.sign(calculator_ref.magnetizability), np.sign(calculator_res.magnetizability)
        )
        diff = calculator_ref.magnetizability - calculator_res.magnetizability
        abs_diff = np.abs(diff)
        print(abs_diff)
        assert np.all(abs_diff < thresh)


def test_final_result_rhf_h2o_sto3g_rpa_singlet_iter() -> None:
    mol = molecules_psi4.molecule_physicists_water_sto3g()
    psi4.core.set_active_molecule(mol)
    psi4.set_options(
        {
            "scf_type": "direct",
            "df_scf_guess": False,
            "e_convergence": 1e-11,
            "d_convergence": 1e-11,
        }
    )
    _, wfn = psi4.energy("hf", return_wfn=True)
    C = mocoeffs_from_psi4wfn(wfn)
    E = moenergies_from_psi4wfn(wfn)
    occupations = occupations_from_psi4wfn(wfn)

    frequencies = [0.0, 0.0773178]

    ref_polarizability = electric.Polarizability(
        Program.Psi4,
        mol,
        cphf.CPHF(solvers.ExactInv(C, E, occupations)),
        C,
        E,
        occupations,
        frequencies=frequencies,
    )
    ref_polarizability.form_operators()
    ref_polarizability.run(hamiltonian=Hamiltonian.RPA, spin=Spin.singlet)
    ref_polarizability.form_results()
    ref_operator = ref_polarizability.driver.solver.operators[0]  # noqa: F841
    res_polarizability = electric.Polarizability(
        Program.Psi4,
        mol,
        cphf.CPHF(
            solvers.IterativeLinEqSolver(C, E, occupations, integrals.JKPsi4(wfn), conv=1.0e-12)
        ),
        C,
        E,
        occupations,
        frequencies=frequencies,
    )
    res_polarizability.form_operators()
    res_polarizability.run(hamiltonian=Hamiltonian.RPA, spin=Spin.singlet)
    res_polarizability.form_results()
    res_operator = res_polarizability.driver.solver.operators[0]  # noqa: F841

    np.testing.assert_allclose(
        ref_polarizability.polarizabilities,
        res_polarizability.polarizabilities,
        rtol=0.0,
        atol=1.0e-6,
    )


if __name__ == "__main__":
    test_inversion()
    test_final_result_rhf_h2o_sto3g_rpa_singlet_iter()
