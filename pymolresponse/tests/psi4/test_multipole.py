import numpy as np

import psi4
from psi4.core import GeometryUnits

from pymolresponse.helpers import make_density
from pymolresponse.interfaces.psi4 import molecules
from pymolresponse.interfaces.psi4.helpers import calculate_dipole
from pymolresponse.interfaces.psi4.utils import (
    mocoeffs_from_psi4wfn,
    occupations_from_psi4wfn,
)


def test_dipole() -> None:
    mol = molecules.molecule_water_sto3g_angstrom()
    psi4.core.set_active_molecule(mol)

    options = {"BASIS": "STO-3G", "SCF_TYPE": "PK", "E_CONVERGENCE": 1e-10, "D_CONVERGENCE": 1e-10}

    psi4.set_options(options)

    _, wfn = psi4.energy("hf", return_wfn=True)

    C = mocoeffs_from_psi4wfn(wfn)
    occupations = occupations_from_psi4wfn(wfn)

    D = make_density(C, occupations)

    mol.set_units(GeometryUnits.Bohr)
    nuccoords = np.asarray(mol.full_geometry())
    nuccharges = np.asarray([mol.charge(i) for i in range(mol.natom())])
    origin = np.zeros(3)

    dipole = calculate_dipole(nuccoords, nuccharges, origin, D[0], wfn, True)

    # Properties will be evaluated at   0.000000,   0.000000,   0.000000 [a0]
    #
    # Properties computed using the SCF density matrix
    #
    #
    #  Multipole Moments:
    #
    #  ------------------------------------------------------------------------------------
    #      Multipole            Electronic (a.u.)      Nuclear  (a.u.)        Total (a.u.)
    #  ------------------------------------------------------------------------------------
    #
    #  L = 1.  Multiply by 2.5417464519 to convert [e a0] to [Debye]
    #  Dipole X            :          0.2835536           -0.8238802           -0.5403267
    #  Dipole Y            :          0.2005028           -0.5825737           -0.3820709
    #  Dipole Z            :         -0.0000000            0.0000000           -0.0000000
    #  Magnitude           :                                                    0.6617637
    #
    #  ------------------------------------------------------------------------------------

    # origin                        [a.u.]: 0.0 0.0 0.0
    # dipole components, electronic [a.u.]: -0.2835535500148627 -0.20050276011671642 1.685250929332574e-17
    # dipole components, nuclear    [a.u.]: -0.4359786427167471 -0.3082847307294024 0.0
    # dipole components, total      [a.u.]: -0.7195321927316098 -0.5087874908461187 1.685250929332574e-17
    # dipole moment, electronic     [a.u.]: 0.34728082662371784
    # dipole moment, nuclear        [a.u.]: 0.5339633434104228
    # dipole moment, total          [a.u.]: 0.8812441700338494
    # dipole components, electronic [D]   : -0.7207211668132235 -0.5096271346735644 4.283480196590654e-17
    # dipole components, nuclear    [D]   : -1.1081470715778003 -0.7835815521630717 0.0
    # dipole components, total      [D]   : -1.828868238391024 -1.293208686836636 4.283480196590654e-17
    # dipole moment, electronic     [D]   : 0.8826997318953946
    # dipole moment, nuclear        [D]   : 1.3571993151843038
    # dipole moment, total          [D]   : 2.239899047078958

    np.testing.assert_allclose(dipole, [-0.5403267, -0.3820709, -0.0000000])
