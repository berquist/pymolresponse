# import findiff
import numpy as np

import pyscf
from pyscf.scf.hf import get_hcore as get_hcore_original

from pymolresponse import cphf, solvers, utils
from pymolresponse.core import Hamiltonian, Program, Spin
from pymolresponse.interfaces.pyscf import molecules
from pymolresponse.interfaces.pyscf.utils import occupations_from_pyscf_mol
from pymolresponse.properties import electric


# adapted from
# https://github.com/pyscf/pyscf/blob/477a96a6ba0d3ff43d0155384fb3336d8b19aed1/examples/scf/40-apply_electric_field.py#L41


# def apply_field(E):
#     mol.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral
#     h = (
#         mol.intor("cint1e_kin_sph")
#         + mol.intor("cint1e_nuc_sph")
#         + numpy.einsum("x,xij->ij", E, mol.intor("cint1e_r_sph", comp=3))
#     )
#     mf = scf.RHF(mol)
#     mf.get_hcore = lambda *args: h
#     mf.scf(dm_init_guess[0])
#     dm_init_guess[0] = mf.make_rdm1()
#     mo = mf.mo_coeff[:, mo_id]
#     if mo[23] < -1e-5:  # To ensure that all MOs have same phase
#         mo *= -1
#     return mo


def get_hcore_with_field(mol, dipole_comp: np.ndarray, field_strength: float) -> np.ndarray:
    return get_hcore_original(mol) + (field_strength * dipole_comp)


def test_numerical() -> None:
    mol = molecules.molecule_alanine_sto3g()
    mol.build()

    mf = pyscf.scf.RHF(mol)
    # conv_tol = 1.0e-11
    mf.kernel()
    mf.dump_scf_summary()
    _, zero_field_dip_moment = mf.analyze()
    print(zero_field_dip_moment)

    # In the derivative formalism, the (static) polarizability is the second
    # derivative of the energy with respect to (applied) electric field, so a
    # double electric perturbation along two independent (field) coordinates.
    # Each of the perturbations can be treated either analytically or
    # numerically.
    #
    # A numeric derivative for one coordinate and analytical derivative for
    # the other is done by first recognizing that the first derivative of the
    # energy with respect to electric field is the dipole moment.  Along one
    # coordinate, using a difference scheme, usually central, compute the
    # dipole moment at the added field strengths, which is done by computing
    # dipole integrals and adding them to the one-electron Hamiltonian
    # multiplied by a constant (the field strength), with the three Cartesian
    # field directions leading to three energy calculations and the dipole
    # moment is an expectation value as is usual.

    # x = findiff.coefficients(deriv=1, acc=2)["center"]
    # coefficients = x["coefficients"]
    # offsets = x["offsets"]

    mol.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral
    ncomp = 3
    integrals_dipole_ao = mol.intor("cint1e_r_sph", comp=ncomp)
    step_size = 1.0e-4
    DIPOLE_AU_TO_DEBYE = 2.54174945416929
    res2 = np.zeros(shape=(ncomp, ncomp))
    dipoles_forwards = []
    for c in range(ncomp):
        mf.get_hcore = lambda mol: get_hcore_with_field(
            mol, dipole_comp=integrals_dipole_ao[c], field_strength=step_size
        )
        mf.kernel()
        _, dip_moment = mf.analyze()
        print(dip_moment)
        res2[c] = (dip_moment - zero_field_dip_moment) / DIPOLE_AU_TO_DEBYE / step_size
        dipoles_forwards.append(dip_moment)
    # TODO why no negative sign?
    res = (np.asarray(dipoles_forwards) - zero_field_dip_moment) / DIPOLE_AU_TO_DEBYE / step_size
    print("FD polarizability 1")
    print(res)
    print("FD polarizability 2")
    print(res2)

    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = occupations_from_pyscf_mol(mol, C)
    polarizability = electric.Polarizability(
        Program.PySCF,
        mol,
        cphf.CPHF(
            solvers.ExactInv(
                C,
                E,
                occupations,
            )
        ),
        frequencies=[0.0],
    )
    polarizability.form_operators()
    polarizability.run(hamiltonian=Hamiltonian.RPA, spin=Spin.singlet)
    polarizability.form_results()

    print("Analytical polarizability")
    print(polarizability.polarizabilities[0])
