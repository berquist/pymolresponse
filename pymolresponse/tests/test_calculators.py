from pathlib import Path

import numpy as np
from cclib.io import ccopen
from cclib.parser.utils import convertor

import pyscf

from pymolresponse import cphf, operators, solvers, utils
from pymolresponse.core import AO2MOTransformationType, Hamiltonian, Spin
from pymolresponse.interfaces.dalton.utils import dalton_label_to_operator
from pymolresponse.interfaces.pyscf.ao2mo import AO2MOpyscf

try:
    from daltools import mol as dalmol
    from daltools import sirifc
except ImportError:
    pass


def calculate_disk_rhf(
    testcasedir: Path, hamiltonian: str, spin: str, frequency: str, label_1: str, label_2: str
) -> float:
    occupations = utils.read_file_occupations(testcasedir / "occupations")
    nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = occupations
    assert nocc_alph == nocc_beta
    assert nvirt_alph == nvirt_beta
    norb = nocc_alph + nvirt_alph
    C = utils.read_file_3(testcasedir / "C")
    assert C.shape[0] == 1
    assert C.shape[2] == norb
    nbasis = C.shape[1]
    moene = utils.read_file_2(testcasedir / "moene")
    assert moene.shape == (norb, 1)
    moints_iajb_aaaa = utils.read_file_4(testcasedir / "moints_iajb_aaaa")
    moints_ijab_aaaa = utils.read_file_4(testcasedir / "moints_ijab_aaaa")
    assert moints_iajb_aaaa.shape == (nocc_alph, nvirt_alph, nocc_alph, nvirt_alph)
    assert moints_ijab_aaaa.shape == (nocc_alph, nocc_alph, nvirt_alph, nvirt_alph)

    operator_1 = dalton_label_to_operator(label_1)
    operator_2 = dalton_label_to_operator(label_2)

    operator_1_integrals_mn = utils.read_file_3(testcasedir / f"operator_mn_{operator_1.label}")
    operator_2_integrals_mn = utils.read_file_3(testcasedir / f"operator_mn_{operator_2.label}")
    # The first dimension can"t be checked since there may be multiple
    # components.
    assert operator_1_integrals_mn.shape[1:] == (nbasis, nbasis)
    assert operator_2_integrals_mn.shape[1:] == (nbasis, nbasis)

    # Only take the component/slice from the integral as determined
    # from the DALTON operator label.
    operator_1_integrals_mn = operator_1_integrals_mn[operator_1.slice_idx]
    operator_2_integrals_mn = operator_2_integrals_mn[operator_2.slice_idx]
    # However, this eliminates an axis, which needs to be added back.
    operator_1_integrals_mn = operator_1_integrals_mn[np.newaxis, ...]
    operator_2_integrals_mn = operator_2_integrals_mn[np.newaxis, ...]

    operator_1.ao_integrals = operator_1_integrals_mn
    operator_2.ao_integrals = operator_2_integrals_mn

    moene = np.diag(moene[:, 0])[np.newaxis, ...]
    assert moene.shape == (1, norb, norb)

    solver = solvers.ExactInv(C, moene, occupations)
    solver.tei_mo = (moints_iajb_aaaa, moints_ijab_aaaa)
    solver.tei_mo_type = AO2MOTransformationType.partial

    driver = cphf.CPHF(solver)
    driver.add_operator(operator_1)
    driver.add_operator(operator_2)

    driver.set_frequencies([float(frequency)])

    driver.run(
        hamiltonian=Hamiltonian[hamiltonian.upper()],
        spin=Spin[spin],
        program=None,
        program_obj=None,
    )

    assert len(driver.frequencies) == len(driver.results) == 1
    res = driver.results[0]
    assert res.shape == (2, 2)
    bl = res[1, 0]
    tr = res[0, 1]
    diff = abs(abs(bl) - abs(tr))
    # Results should be symmetric w.r.t. interchange between operators
    # in the LR equations.
    thresh = 1.0e-13
    assert diff < thresh

    return bl


def calculate_disk_uhf(
    testcasedir: Path, hamiltonian: str, spin: str, frequency: str, label_1: str, label_2: str
) -> float:
    occupations = utils.read_file_occupations(testcasedir / "occupations")
    nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = occupations
    norb = nocc_alph + nvirt_alph
    C = utils.read_file_3(testcasedir / "C")
    assert C.shape[0] == 2
    assert C.shape[2] == norb
    nbasis = C.shape[1]
    moene = utils.read_file_2(testcasedir / "moene")
    assert moene.shape == (norb, 2)
    moints_iajb_aaaa = utils.read_file_4(testcasedir / "moints_iajb_aaaa")
    moints_iajb_aabb = utils.read_file_4(testcasedir / "moints_iajb_aabb")
    moints_iajb_bbaa = utils.read_file_4(testcasedir / "moints_iajb_bbaa")
    moints_iajb_bbbb = utils.read_file_4(testcasedir / "moints_iajb_bbbb")
    moints_ijab_aaaa = utils.read_file_4(testcasedir / "moints_ijab_aaaa")
    moints_ijab_bbbb = utils.read_file_4(testcasedir / "moints_ijab_bbbb")
    assert moints_iajb_aaaa.shape == (nocc_alph, nvirt_alph, nocc_alph, nvirt_alph)
    assert moints_iajb_aabb.shape == (nocc_alph, nvirt_alph, nocc_beta, nvirt_beta)
    assert moints_iajb_bbaa.shape == (nocc_beta, nvirt_beta, nocc_alph, nvirt_alph)
    assert moints_iajb_bbbb.shape == (nocc_beta, nvirt_beta, nocc_beta, nvirt_beta)
    assert moints_ijab_aaaa.shape == (nocc_alph, nocc_alph, nvirt_alph, nvirt_alph)
    assert moints_ijab_bbbb.shape == (nocc_beta, nocc_beta, nvirt_beta, nvirt_beta)

    operator_1 = dalton_label_to_operator(label_1)
    operator_2 = dalton_label_to_operator(label_2)

    operator_1_integrals_mn = utils.read_file_3(testcasedir / f"operator_mn_{operator_1.label}")
    operator_2_integrals_mn = utils.read_file_3(testcasedir / f"operator_mn_{operator_2.label}")
    # The first dimension can"t be checked since there may be multiple
    # components.
    assert operator_1_integrals_mn.shape[1:] == (nbasis, nbasis)
    assert operator_2_integrals_mn.shape[1:] == (nbasis, nbasis)

    # Only take the component/slice from the integral as determined
    # from the DALTON operator label.
    operator_1_integrals_mn = operator_1_integrals_mn[operator_1.slice_idx]
    operator_2_integrals_mn = operator_2_integrals_mn[operator_2.slice_idx]
    # However, this eliminates an axis, which needs to be added back.
    operator_1_integrals_mn = operator_1_integrals_mn[np.newaxis, ...]
    operator_2_integrals_mn = operator_2_integrals_mn[np.newaxis, ...]

    operator_1.ao_integrals = operator_1_integrals_mn
    operator_2.ao_integrals = operator_2_integrals_mn

    moene_alph = np.diag(moene[:, 0])
    moene_beta = np.diag(moene[:, 1])
    moene = np.stack((moene_alph, moene_beta), axis=0)
    assert moene.shape == (2, norb, norb)

    solver = solvers.ExactInv(C, moene, occupations)
    solver.tei_mo = (
        moints_iajb_aaaa,
        moints_iajb_aabb,
        moints_iajb_bbaa,
        moints_iajb_bbbb,
        moints_ijab_aaaa,
        moints_ijab_bbbb,
    )
    solver.tei_mo_type = AO2MOTransformationType.partial

    driver = cphf.CPHF(solver)
    driver.add_operator(operator_1)
    driver.add_operator(operator_2)

    driver.set_frequencies([float(frequency)])

    driver.run(
        hamiltonian=Hamiltonian[hamiltonian.upper()],
        spin=Spin[spin],
        program=None,
        program_obj=None,
    )

    assert len(driver.frequencies) == len(driver.results) == 1
    res = driver.results[0]
    assert res.shape == (2, 2)
    bl = res[1, 0]
    tr = res[0, 1]
    diff = abs(abs(bl) - abs(tr))
    # Results should be symmetric w.r.t. interchange between operators
    # in the LR equations.
    thresh = 1.0e-14
    assert diff < thresh

    return bl


def calculate_rhf(
    dalton_tmpdir: Path,
    hamiltonian: str,
    spin: str,
    operator_label: str,
    operator: str,
    source_moenergies: str,
    source_mocoeffs: str,
    source_operator: str,
):
    if operator_label:
        # TODO add dipvel
        assert operator_label in ("dipole", "angmom", "spinorb")
    assert source_moenergies in ("pyscf", "dalton")
    assert source_mocoeffs in ("pyscf", "dalton")

    dalton_molecule = dalmol.readin(dalton_tmpdir / "DALTON.BAS")
    lines = []
    for atom in dalton_molecule:
        label = atom["label"][0]
        center = atom["center"][0]
        center_str = " ".join(["{:f}".format(pos) for pos in center])
        line = "{:3} {}".format(label, center_str)
        lines.append(line)
    lines = "\n".join(lines)

    # PySCF molecule setup, needed for generating the TEIs in the MO
    # basis.
    mol = pyscf.gto.Mole()
    verbose = 1
    mol.verbose = verbose
    mol.atom = lines
    mol.unit = "Bohr"
    # TODO read basis from DALTON molecule
    mol.basis = "sto-3g"
    mol.symmetry = False
    # TODO read charge from DALTON molecule?
    mol.charge = 0
    # TODO read spin from DALTON molecule?
    mol.spin = 0

    mol.build()

    ifc = sirifc.sirifc(dalton_tmpdir / "SIRIFC")
    occupations = utils.occupations_from_sirifc(ifc)

    if source_moenergies == "pyscf" or source_mocoeffs == "pyscf":
        mf = pyscf.scf.RHF(mol)
        mf.kernel()

    if source_moenergies == "pyscf":
        E = np.diag(mf.mo_energy)[np.newaxis, ...]
    elif source_moenergies == "dalton":
        job = ccopen(dalton_tmpdir / "DALTON.OUT")
        data = job.parse()
        # pylint: disable=no-member
        E = np.diag([convertor(x, "eV", "hartree") for x in data.moenergies[0]])[np.newaxis, ...]
    else:
        pass

    if source_mocoeffs == "pyscf":
        C = mf.mo_coeff[np.newaxis, ...]
    elif source_mocoeffs == "dalton":
        C = ifc.cmo[0][np.newaxis, ...]
    else:
        pass

    solver = solvers.ExactInv(C, E, occupations)

    solver.tei_mo = AO2MOpyscf(mol, C).perform_rhf_partial()
    solver.tei_mo_type = AO2MOTransformationType.partial

    driver = cphf.CPHF(solver)

    if operator:
        driver.add_operator(operator)
    elif operator_label:
        if operator_label == "dipole":
            operator_dipole = operators.Operator(
                label="dipole", is_imaginary=False, is_spin_dependent=False, triplet=False
            )
            integrals_dipole_ao = mol.intor("cint1e_r_sph", comp=3)
            operator_dipole.ao_integrals = integrals_dipole_ao
            driver.add_operator(operator_dipole)
        elif operator_label == "angmom":
            operator_angmom = operators.Operator(
                label="angmom", is_imaginary=True, is_spin_dependent=False, triplet=False
            )
            integrals_angmom_ao = mol.intor("cint1e_cg_irxp_sph", comp=3)
            operator_angmom.ao_integrals = integrals_angmom_ao
            driver.add_operator(operator_angmom)
        elif operator_label == "spinorb":
            operator_spinorb = operators.Operator(
                label="spinorb", is_imaginary=True, is_spin_dependent=False, triplet=False
            )
            integrals_spinorb_ao = 0
            for atm_id in range(mol.natm):
                mol.set_rinv_orig(mol.atom_coord(atm_id))
                chg = mol.atom_charge(atm_id)
                integrals_spinorb_ao += chg * mol.intor("cint1e_prinvxp_sph", comp=3)
            operator_spinorb.ao_integrals = integrals_spinorb_ao
            driver.add_operator(operator_spinorb)
        else:
            pass
    else:
        pass

    driver.set_frequencies()

    driver.run(hamiltonian=Hamiltonian[hamiltonian.upper()], spin=Spin[spin])

    return driver.results[0]


def calculate_uhf(
    dalton_tmpdir: Path,
    hamiltonian: str,
    spin: str,
    operator_label: str,
    operator: str,
    source_moenergies: str,
    source_mocoeffs: str,
    source_operator: str,
):
    if operator_label:
        # TODO add dipvel
        assert operator_label in ("dipole", "angmom", "spinorb")
    assert source_moenergies in ("pyscf", "dalton")
    assert source_mocoeffs in ("pyscf", "dalton")

    dalton_molecule = dalmol.readin(dalton_tmpdir / "DALTON.BAS")
    lines = []
    for atom in dalton_molecule:
        label = atom["label"][0]
        center = atom["center"][0]
        center_str = " ".join(["{:f}".format(pos) for pos in center])
        line = "{:3} {}".format(label, center_str)
        lines.append(line)
    lines = "\n".join(lines)

    # PySCF molecule setup, needed for generating the TEIs in the MO
    # basis.
    mol = pyscf.gto.Mole()
    verbose = 1
    mol.verbose = verbose
    mol.atom = lines
    mol.unit = "Bohr"
    # TODO read basis from DALTON molecule
    mol.basis = "sto-3g"
    mol.symmetry = False
    # TODO read charge from DALTON molecule?
    mol.charge = 1
    # TODO read spin from DALTON molecule?
    mol.spin = 1

    mol.build()

    ifc = sirifc.sirifc(dalton_tmpdir / "SIRIFC")
    occupations = utils.occupations_from_sirifc(ifc)

    if source_moenergies == "pyscf" or source_mocoeffs == "pyscf":
        mf = pyscf.scf.UHF(mol)
        mf.kernel()

    if source_moenergies == "pyscf":
        E_alph = np.diag(mf.mo_energy[0])
        E_beta = np.diag(mf.mo_energy[1])
        E = np.stack((E_alph, E_beta), axis=0)
    elif source_moenergies == "dalton":
        job = ccopen(dalton_tmpdir / "DALTON.OUT")
        data = job.parse()
        # pylint: disable=no-member
        E = np.diag([convertor(x, "eV", "hartree") for x in data.moenergies[0]])[np.newaxis, ...]
        E = np.concatenate((E, E), axis=0)
    else:
        pass

    if source_mocoeffs == "pyscf":
        C = mf.mo_coeff
    elif source_mocoeffs == "dalton":
        C = ifc.cmo[0][np.newaxis, ...]
        C = np.concatenate((C, C), axis=0)
    else:
        pass

    solver = solvers.ExactInv(C, E, occupations)

    solver.tei_mo = AO2MOpyscf(mol, C).perform_uhf_partial()
    solver.tei_mo_type = AO2MOTransformationType.partial

    driver = cphf.CPHF(solver)

    if operator:
        driver.add_operator(operator)
    elif operator_label:
        if operator_label == "dipole":
            operator_dipole = operators.Operator(
                label="dipole", is_imaginary=False, is_spin_dependent=False, triplet=False
            )
            integrals_dipole_ao = mol.intor("cint1e_r_sph", comp=3)
            operator_dipole.ao_integrals = integrals_dipole_ao
            driver.add_operator(operator_dipole)
        elif operator_label == "angmom":
            operator_angmom = operators.Operator(
                label="angmom", is_imaginary=True, is_spin_dependent=False, triplet=False
            )
            integrals_angmom_ao = mol.intor("cint1e_cg_irxp_sph", comp=3)
            operator_angmom.ao_integrals = integrals_angmom_ao
            driver.add_operator(operator_angmom)
        elif operator_label == "spinorb":
            operator_spinorb = operators.Operator(
                label="spinorb", is_imaginary=True, is_spin_dependent=False, triplet=False
            )
            integrals_spinorb_ao = 0
            for atm_id in range(mol.natm):
                mol.set_rinv_orig(mol.atom_coord(atm_id))
                chg = mol.atom_charge(atm_id)
                integrals_spinorb_ao += chg * mol.intor("cint1e_prinvxp_sph", comp=3)
            operator_spinorb.ao_integrals = integrals_spinorb_ao
            driver.add_operator(operator_spinorb)
        else:
            pass
    else:
        pass

    driver.set_frequencies()

    driver.run(hamiltonian=Hamiltonian[hamiltonian.upper()], spin=Spin[spin])

    return driver.results[0]
