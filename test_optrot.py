from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.constants as spc

import pyscf

from . import utils

from .optrot import ORD
from .test_ecd import molecule_BC2H4_cation_HF_STO3G


BC2H4_cation_HF_STO3G_RPA_singlet_nwchem = {
    # from minimal
    0.0 : {
        'polar': np.array([[16.9282686, -4.8000244, -0.7875224],
                           [-4.8000245, 15.3007190, -2.7824200],
                           [-0.7875224, -2.7824200, 19.3955636]]),
    },
    # from orbeta
    0.001 : {
        'polar': np.array([[16.9283289, -4.8000606, -0.7875053],
                           [-4.8000606, 15.3008509, -2.7825269],
                           [-0.7875053, -2.7825269, 19.3957197]]),
        'orbeta': np.array([[0.0684, -9.8683, -1.1279],
                            [8.8646, -7.1416, 6.5276],
                            [0.9846, 5.6305, -12.6550]]),
    },
    # from orbeta
    0.0773178 : {
        'polar': np.array([[17.3117129, -5.0815177, -0.6234223],
                           [-5.0815177, 16.4754295, -3.8455056],
                           [-0.6234223, -3.8455056, 20.8275781]]),
        'orbeta': np.array([[-0.0736, -9.0074, -1.3061],
                            [10.7935, -13.5900, 14.8878],
                            [-1.1477, 11.8707, -25.1217]]),
    },
    # from orbeta
    0.128347 : {
        'polar': np.array([[18.0091879, -4.7985368, -1.4998108],
                           [-4.7985368, 11.1823569, 3.6436223],
                           [-1.4998108, 3.6436223, 11.9881428]]),
        'orbeta': np.array([[1.9247, -11.4465, 19.3223],
                            [-3.6550, 18.6493, -101.8119],
                            [16.9194, -35.1753, 99.3169]]),
    },
}


def test_ORD_RPA_singlet_BC2H4_cation_HF_STO3G():

    ref = BC2H4_cation_HF_STO3G_RPA_singlet_nwchem

    pyscfmol = molecule_BC2H4_cation_HF_STO3G(0)
    pyscfmol.build()

    mf = pyscf.scf.RHF(pyscfmol)
    mf.scf()

    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = np.diag(mf.mo_energy)[np.newaxis, ...]
    occupations = utils.occupations_from_pyscf_mol(pyscfmol, C)

    frequencies = [0.0, 0.001, 0.0773178, 0.128347]
    ord_solver = ORD(pyscfmol, C, E, occupations, hamiltonian='rpa', spin='singlet', frequencies=frequencies, do_dipvel=False)
    ord_solver.form_operators()
    ord_solver.run()
    ord_solver.form_results()

    print('Polarizabilities')
    assert len(frequencies) == len(ord_solver.polarizabilities)
    thresh = 5.0e-4
    for idxf, frequency in enumerate(frequencies):
        ref_polar = ref[frequency]['polar']
        res_polar = ord_solver.polarizabilities[idxf]
        abs_diff = abs(res_polar - ref_polar)
        print(idxf, frequency)
        print(res_polar)
        print(abs_diff)
        assert (abs_diff < thresh).all()

    print(r'\beta(\omega)')
    thresh = 5.0e-2
    for idxf, frequency in enumerate(frequencies):
        if 'orbeta' in ref[frequency]:
            ref_beta = ref[frequency]['orbeta']
            # TODO why no speed of light?
            # TODO why the (1/2)?
            res_beta = -(0.5 / frequency) * ord_solver.solver.results[idxf][3:6, 0:3]
            abs_diff = abs(res_beta - ref_beta)
            print(idxf, frequency)
            print(res_beta)
            print(abs_diff)
            assert (abs_diff < thresh).all()

    # from ecd import ECD
    # ecd = ECD(pyscfmol, C, E, occupations, do_dipvel=True, do_tda=False)
    # ecd.run()
    # ecd.form_results()
    # ord_solver.form_operators()
    # ord_solver.run()
    # ord_solver.form_results()
    # from constants import esuecd
    # prefac = -(2 / 3) / esuecd
    # for idxf, frequency in enumerate(frequencies):
    #     print(sum(prefac * ecd.rotational_strengths_diplen / ((frequency ** 2) * (ecd.solver.eigvals.real ** 2))))

    return

if __name__ == '__main__':
    test_ORD_RPA_singlet_BC2H4_cation_HF_STO3G()
