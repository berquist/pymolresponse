#!/usr/bin/env python

import pyscf

from pyresponse import utils
from pyresponse.molecules import molecule_LiH_cation_HF_STO3G
from pyresponse.electric import Polarizability


mol = molecule_LiH_cation_HF_STO3G(5)
mol.charge = 0
mol.spin = 0
mol.build()

mf = pyscf.scf.RHF(mol)
mf.kernel()

C = utils.fix_mocoeffs_shape(mf.mo_coeff)
E = utils.fix_moenergies_shape(mf.mo_energy)
occupations = utils.occupations_from_pyscf_mol(mol, C)

from pyresponse.iterators import ExactInvCholesky
from pyresponse.cphf import CPHF

solver = ExactInvCholesky(C, E, occupations)
solver.form_tei_mo(mol, tei_mo_type='full')
driver = CPHF(solver)
polarizability = Polarizability(mol, C, E, occupations, frequencies=[0.0], solver=solver)
polarizability.form_operators()
polarizability.run()
polarizability.form_results()

# print('\n'.join(dir(polarizability)))
# print(type(polarizability.driver))
# print(type(polarizability.solver)) # NoneType
# print(type(polarizability.driver.solver))
# print(type(polarizability.driver.solver.tei_mo))
# print(len(polarizability.driver.solver.tei_mo))

np_formatter = {
    'float_kind': lambda x: '{:12.8f}'.format(x)
}
import numpy as np
np.set_printoptions(linewidth=200, formatter=np_formatter)
print(2 * solver.explicit_hessian)
nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = occupations
nov_alph = nocc_alph * nvirt_alph
print(np.reshape(solver.operators[0].mo_integrals_ai_supervector_alph[0, ...], (nov_alph, 2), order='F'))
