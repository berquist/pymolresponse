import numpy as np

import pyscf

from pymolresponse import cphf, solvers, utils
from pymolresponse.core import Hamiltonian, Program, Spin
from pymolresponse.interfaces.pyscf import molecules
from pymolresponse.interfaces.pyscf.utils import occupations_from_pyscf_mol
from pymolresponse.properties import optrot

BC2H4_cation_HF_STO3G_RPA_singlet_nwchem = {
    # from minimal
    0.0: {
        "polar": np.array(
            [
                [16.9282686, -4.8000244, -0.7875224],
                [-4.8000245, 15.3007190, -2.7824200],
                [-0.7875224, -2.7824200, 19.3955636],
            ]
        )
    },
    # from orbeta
    0.001: {
        "polar": np.array(
            [
                [16.9283289, -4.8000606, -0.7875053],
                [-4.8000606, 15.3008509, -2.7825269],
                [-0.7875053, -2.7825269, 19.3957197],
            ]
        ),
        "orbeta": np.array(
            [[0.0684, -9.8683, -1.1279], [8.8646, -7.1416, 6.5276], [0.9846, 5.6305, -12.6550]]
        ),
    },
    # from orbeta
    0.0773178: {
        "polar": np.array(
            [
                [17.3117129, -5.0815177, -0.6234223],
                [-5.0815177, 16.4754295, -3.8455056],
                [-0.6234223, -3.8455056, 20.8275781],
            ]
        ),
        "orbeta": np.array(
            [
                [-0.0736, -9.0074, -1.3061],
                [10.7935, -13.5900, 14.8878],
                [-1.1477, 11.8707, -25.1217],
            ]
        ),
    },
    # from orbeta
    0.128347: {
        "polar": np.array(
            [
                [18.0091879, -4.7985368, -1.4998108],
                [-4.7985368, 11.1823569, 3.6436223],
                [-1.4998108, 3.6436223, 11.9881428],
            ]
        ),
        "orbeta": np.array(
            [
                [1.9247, -11.4465, 19.3223],
                [-3.6550, 18.6493, -101.8119],
                [16.9194, -35.1753, 99.3169],
            ]
        ),
    },
}


BC2H4_HF_STO3G_RPA_singlet_nwchem = {
    # from minimal
    0.0: {
        "polar": np.array(
            [
                [16.0702163, -4.9832488, -2.2293822],
                [-4.9832488, 13.1502378, 0.2844524],
                [-2.2293822, 0.2844524, 18.1993490],
            ]
        )
    },
    # from orbeta
    0.001: {
        "polar": np.array(
            [
                [16.0702789, -4.9832818, -2.2293882],
                [-4.9832818, 13.1502964, 0.2844393],
                [-2.2293882, 0.2844393, 18.1994070],
            ]
        ),
        "orbeta": np.array(
            [[-1.7067, -5.8002, -2.9465], [7.2404, 0.9278, 5.3503], [4.9573, -0.5056, -3.3741]]
        ),
    },
    # from orbeta
    0.0773178: {
        "polar": np.array(
            [
                [16.4963293, -5.2448779, -2.2476469],
                [-5.2448779, 13.6064856, 0.1637870],
                [-2.2476469, 0.1637869, 18.5823648],
            ]
        ),
        "orbeta": np.array(
            [[-1.4712, -5.3345, -4.2604], [6.4702, 0.3833, 8.8611], [5.6097, -0.4006, -5.0196]]
        ),
    },
    # from orbeta
    0.128347: {
        "polar": np.array(
            [
                [15.3394615, -2.0395326, -3.7937930],
                [-2.0395327, 7.7254042, 2.6104237],
                [-3.7937933, 2.6104240, 18.3446840],
            ]
        ),
        # Flipped the sign on the [2, 1] element.
        "orbeta": np.array(
            [[-17.0050, -8.6288, 53.2469], [33.1190, 7.7765, -90.7071], [-4.0315, -3.5893, 34.0186]]
        ),
    },
}


trithiolane_HF_STO3G_RPA_singlet = {
    0.0: {
        "polar": np.asarray(
            [
                [3.47899000e01, -1.77591000e-05, 2.24416000e-04],
                [-1.77591000e-05, 4.43286000e01, -1.91191000e-01],
                [2.24416000e-04, -1.91191000e-01, 1.68476000e01],
            ]
        ),
        "dipmag": np.asarray(
            [
                [-0.854380e01, -0.279465e-02, 0.693947e-03],
                [-0.537619e-02, 0.125637e02, -0.447617e01],
                [0.281657e-02, -0.985703e01, -0.506194e01],
            ]
        ),
    },
    0.0773178: {
        "polar": np.asarray(
            [
                [3.53972000e01, -8.48249000e-06, 2.05118000e-04],
                [-8.48249000e-06, 4.52552000e01, -2.17003000e-01],
                [2.05118000e-04, -2.17003000e-01, 1.70848000e01],
            ]
        ),
        "dipmag": np.asarray(
            [
                [-0.873446e01, -0.291464e-02, 0.749139e-03],
                [-0.546200e-02, 0.129058e02, -0.453107e01],
                [0.290515e-02, -0.105516e02, -0.537653e01],
            ]
        ),
    },
    0.128347: {
        "polar": np.asarray(
            [
                [3.65252000e01, -2.36748000e-06, 1.76171000e-04],
                [-2.36748000e-06, 4.70314000e01, -2.86868000e-01],
                [1.76171000e-04, -2.86868000e-01, 1.75452000e01],
            ]
        ),
        "dipmag": np.asarray(
            [
                [-0.908406e01, -0.315108e-02, 0.843565e-03],
                [-0.563037e-02, 0.135977e02, -0.466676e01],
                [0.306764e-02, -0.118891e02, -0.609620e01],
            ]
        ),
    },
}


def test_ORD_RPA_singlet_BC2H4_cation_HF_STO3G():
    ref = BC2H4_cation_HF_STO3G_RPA_singlet_nwchem

    pyscfmol = molecules.molecule_bc2h4_cation_sto3g()
    pyscfmol.build()

    mf = pyscf.scf.RHF(pyscfmol)
    mf.scf()

    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = occupations_from_pyscf_mol(pyscfmol, C)

    frequencies = [0.0, 0.001, 0.0773178, 0.128347]
    ord_solver = optrot.ORD(
        Program.PySCF,
        pyscfmol,
        cphf.CPHF(solvers.ExactInv(C, E, occupations)),
        C,
        E,
        occupations,
        frequencies=frequencies,
        do_dipvel=False,
    )
    ord_solver.form_operators()
    ord_solver.run(hamiltonian=Hamiltonian.RPA, spin=Spin.singlet)
    ord_solver.form_results()

    print("Polarizabilities")
    assert len(frequencies) == len(ord_solver.polarizabilities)
    thresh = 5.0e-4
    for idxf, frequency in enumerate(frequencies):
        ref_polar = ref[frequency]["polar"]
        res_polar = ord_solver.polarizabilities[idxf]
        abs_diff = abs(res_polar - ref_polar)
        print(idxf, frequency)
        print(res_polar)
        print(abs_diff)
        assert (abs_diff < thresh).all()

    print(r"\beta(\omega)")
    thresh = 5.0e-2
    for idxf, frequency in enumerate(frequencies):
        if "orbeta" in ref[frequency]:
            ref_beta = ref[frequency]["orbeta"]
            # TODO why no speed of light?
            # TODO why the (1/2)?
            res_beta = -(0.5 / frequency) * ord_solver.driver.results[idxf][3:6, 0:3]
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


def test_ORD_RPA_singlet_BC2H4_HF_STO3G():
    ref = BC2H4_HF_STO3G_RPA_singlet_nwchem

    pyscfmol = molecules.molecule_bc2h4_neutral_radical_sto3g()
    pyscfmol.build()

    mf = pyscf.scf.UHF(pyscfmol)
    mf.scf()

    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = occupations_from_pyscf_mol(pyscfmol, C)

    frequencies = [0.0, 0.001, 0.0773178, 0.128347]
    ord_solver = optrot.ORD(
        Program.PySCF,
        pyscfmol,
        cphf.CPHF(solvers.ExactInv(C, E, occupations)),
        C,
        E,
        occupations,
        frequencies=frequencies,
        do_dipvel=False,
    )
    ord_solver.form_operators()
    ord_solver.run(hamiltonian=Hamiltonian.RPA, spin=Spin.singlet)
    ord_solver.form_results()

    print("Polarizabilities")
    assert len(frequencies) == len(ord_solver.polarizabilities)
    thresh = 6.5e-4
    for idxf, frequency in enumerate(frequencies):
        ref_polar = ref[frequency]["polar"]
        res_polar = ord_solver.polarizabilities[idxf]
        abs_diff = abs(res_polar - ref_polar)
        print(idxf, frequency)
        print(res_polar)
        print(abs_diff)
        if frequency == 0.128347:
            thresh = 6.0e-3
        assert (abs_diff < thresh).all()

    print(r"\beta(\omega)")
    thresh = 0.09
    for idxf, frequency in enumerate(frequencies):
        if "orbeta" in ref[frequency]:
            ref_beta = ref[frequency]["orbeta"]
            # TODO why no speed of light?
            # TODO why the (1/2)?
            res_beta = -(0.5 / frequency) * ord_solver.driver.results[idxf][3:6, 0:3]
            abs_diff = abs(res_beta - ref_beta)
            print(idxf, frequency)
            print(res_beta)
            print(ref_beta)
            print(abs_diff)
            assert (abs_diff < thresh).all()


def test_ORD_RPA_singlet_trithiolane_HF_STO3G():
    ref = trithiolane_HF_STO3G_RPA_singlet

    pyscfmol = molecules.molecule_trithiolane_sto3g()
    pyscfmol.build()

    mf = pyscf.scf.RHF(pyscfmol)
    mf.scf()

    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = utils.fix_moenergies_shape(mf.mo_energy)
    occupations = occupations_from_pyscf_mol(pyscfmol, C)

    frequencies = [0.0, 0.0773178, 0.128347]
    ord_solver = optrot.ORD(
        Program.PySCF,
        pyscfmol,
        cphf.CPHF(solvers.ExactInv(C, E, occupations)),
        C,
        E,
        occupations,
        frequencies=frequencies,
        do_dipvel=False,
    )
    ord_solver.form_operators()
    ord_solver.run(hamiltonian=Hamiltonian.RPA, spin=Spin.singlet)
    ord_solver.form_results()

    print("Polarizabilities")
    assert len(frequencies) == len(ord_solver.polarizabilities)
    thresh = 5.0e-4
    for idxf, frequency in enumerate(frequencies):
        ref_polar = ref[frequency]["polar"]
        res_polar = ord_solver.polarizabilities[idxf]
        abs_diff = abs(res_polar - ref_polar)
        print(idxf, frequency)
        print(res_polar)
        print(abs_diff)
        assert (abs_diff < thresh).all()

    # print('Electric dipole-magnetic dipole polarizabilities')
    # assert len(frequencies) == len(ord_solver.polarizabilities_lenmag)
    # # thresh
    # for idxf, frequency in enumerate(frequencies):
    #     if 'dipmag' in ref[frequency]:
    #         ref_dipmag = ref[frequency]['dipmag']
    #         res_dipmag = ord_solver.polarizabilities_lenmag[idxf]
    #         print(idxf, frequency)
    #         # print(ref_dipmag)
    #         # print(res_dipmag)
    #         print(res_dipmag / ref_dipmag)

    # print(r'\beta(\omega)')
    # thresh = 5.0e-2
    # for idxf, frequency in enumerate(frequencies):
    #     if 'orbeta' in ref[frequency]:
    #         ref_beta = ref[frequency]['orbeta']
    #         # why no speed of light? atomic units...
    #         # TODO why the (1/2)?
    #         res_beta = -(0.5 / frequency) * ord_solver.solver.results[idxf][3:6, 0:3]
    #         abs_diff = abs(res_beta - ref_beta)
    #         print(idxf, frequency)
    #         print(res_beta)
    #         print(abs_diff)
    #         assert (abs_diff < thresh).all()


if __name__ == "__main__":
    test_ORD_RPA_singlet_BC2H4_cation_HF_STO3G()
    test_ORD_RPA_singlet_BC2H4_HF_STO3G()
    test_ORD_RPA_singlet_trithiolane_HF_STO3G()
