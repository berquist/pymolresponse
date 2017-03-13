import numpy as np

from cphf import CPHF
from utils import read_file_3, read_file_2, read_file_occupations, read_file_4, dalton_label_to_operator


def calculate_disk_rhf(testcase, hamiltonian, spin, frequency, label_1, label_2):

    occupations = read_file_occupations(testcase + '/' + 'occupations')
    nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = occupations
    assert nocc_alph == nocc_beta
    assert nvirt_alph == nvirt_beta
    norb = nocc_alph + nvirt_alph
    C = read_file_3(testcase + '/' + 'C')
    assert C.shape[0] == 1
    assert C.shape[2] == norb
    nbasis = C.shape[1]
    moene = read_file_2(testcase + '/' + 'moene')
    assert moene.shape == (norb, 1)
    moints_iajb_aaaa = read_file_4(testcase + '/' + 'moints_iajb_aaaa')
    moints_ijab_aaaa = read_file_4(testcase + '/' + 'moints_ijab_aaaa')
    assert moints_iajb_aaaa.shape == (nocc_alph, nvirt_alph, nocc_alph, nvirt_alph)
    assert moints_ijab_aaaa.shape == (nocc_alph, nocc_alph, nvirt_alph, nvirt_alph)

    operator_1 = dalton_label_to_operator(label_1)
    operator_2 = dalton_label_to_operator(label_2)

    operator_1_integrals_mn = read_file_3(testcase + '/' + 'operator_mn_' + operator_1.label)
    operator_2_integrals_mn = read_file_3(testcase + '/' + 'operator_mn_' + operator_2.label)
    # The first dimension can't be checked since there may be multiple
    # components.
    assert operator_1_integrals_mn.shape[1:] == (nbasis, nbasis)
    assert operator_2_integrals_mn.shape[1:] == (nbasis, nbasis)

    # Only take the component/slice from the integral as determined
    # from the DALTON operator label.
    operator_1_integrals_mn = operator_1_integrals_mn[operator_1.slice_idx, ...]
    operator_2_integrals_mn = operator_2_integrals_mn[operator_2.slice_idx, ...]
    # However, this eliminates an axis, which needs to be added back.
    operator_1_integrals_mn = operator_1_integrals_mn[np.newaxis, ...]
    operator_2_integrals_mn = operator_2_integrals_mn[np.newaxis, ...]

    operator_1.ao_integrals = operator_1_integrals_mn
    operator_2.ao_integrals = operator_2_integrals_mn

    moene = np.diag(moene[:, 0])[np.newaxis, ...]
    assert moene.shape == (1, norb, norb)

    cphf = CPHF(C, moene, occupations)
    cphf.add_operator(operator_1)
    cphf.add_operator(operator_2)

    cphf.tei_mo = (moints_iajb_aaaa, moints_ijab_aaaa)
    cphf.tei_mo_type = 'partial'

    cphf.set_frequencies([float(frequency)])

    cphf.run(solver='explicit', hamiltonian=hamiltonian, spin=spin)

    assert len(cphf.frequencies) == len(cphf.results) == 1
    res = cphf.results[0]
    assert res.shape == (2, 2)
    bl = res[1, 0]
    tr = res[0, 1]
    diff = abs(abs(bl) - abs(tr))
    # Results should be symmetric w.r.t. interchange between operators
    # in the LR equations.
    thresh = 1.0e-14
    assert diff < thresh

    return bl


def calculate_disk_uhf(testcase, hamiltonian, spin, frequency, label_1, label_2):

    occupations = read_file_occupations(testcase + '/' + 'occupations')
    nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = occupations
    # assert nocc_alph == nocc_beta
    # assert nvirt_alph == nvirt_beta
    norb = nocc_alph + nvirt_alph
    C = read_file_3(testcase + '/' + 'C')
    assert C.shape[0] == 2
    assert C.shape[2] == norb
    nbasis = C.shape[1]
    moene = read_file_2(testcase + '/' + 'moene')
    assert moene.shape == (norb, 2)
    moints_iajb_aaaa = read_file_4(testcase + '/' + 'moints_iajb_aaaa')
    moints_iajb_aabb = read_file_4(testcase + '/' + 'moints_iajb_aabb')
    moints_iajb_bbaa = read_file_4(testcase + '/' + 'moints_iajb_bbaa')
    moints_iajb_bbbb = read_file_4(testcase + '/' + 'moints_iajb_bbbb')
    moints_ijab_aaaa = read_file_4(testcase + '/' + 'moints_ijab_aaaa')
    moints_ijab_bbbb = read_file_4(testcase + '/' + 'moints_ijab_bbbb')
    assert moints_iajb_aaaa.shape == (nocc_alph, nvirt_alph, nocc_alph, nvirt_alph)
    assert moints_iajb_aabb.shape == (nocc_alph, nvirt_alph, nocc_beta, nvirt_beta)
    assert moints_iajb_bbaa.shape == (nocc_beta, nvirt_beta, nocc_alph, nvirt_alph)
    assert moints_iajb_bbbb.shape == (nocc_beta, nvirt_beta, nocc_beta, nvirt_beta)
    assert moints_ijab_aaaa.shape == (nocc_alph, nocc_alph, nvirt_alph, nvirt_alph)
    assert moints_ijab_bbbb.shape == (nocc_beta, nocc_beta, nvirt_beta, nvirt_beta)

    operator_1 = dalton_label_to_operator(label_1)
    operator_2 = dalton_label_to_operator(label_2)

    operator_1_integrals_mn = read_file_3(testcase + '/' + 'operator_mn_' + operator_1.label)
    operator_2_integrals_mn = read_file_3(testcase + '/' + 'operator_mn_' + operator_2.label)
    # The first dimension can't be checked since there may be multiple
    # components.
    assert operator_1_integrals_mn.shape[1:] == (nbasis, nbasis)
    assert operator_2_integrals_mn.shape[1:] == (nbasis, nbasis)

    # Only take the component/slice from the integral as determined
    # from the DALTON operator label.
    operator_1_integrals_mn = operator_1_integrals_mn[operator_1.slice_idx, ...]
    operator_2_integrals_mn = operator_2_integrals_mn[operator_2.slice_idx, ...]
    # However, this eliminates an axis, which needs to be added back.
    operator_1_integrals_mn = operator_1_integrals_mn[np.newaxis, ...]
    operator_2_integrals_mn = operator_2_integrals_mn[np.newaxis, ...]

    operator_1.ao_integrals = operator_1_integrals_mn
    operator_2.ao_integrals = operator_2_integrals_mn

    moene_alph = np.diag(moene[:, 0])
    moene_beta = np.diag(moene[:, 1])
    moene = np.stack((moene_alph, moene_beta), axis=0)
    assert moene.shape == (2, norb, norb)

    cphf = CPHF(C, moene, occupations)
    cphf.add_operator(operator_1)
    cphf.add_operator(operator_2)

    cphf.tei_mo = (moints_iajb_aaaa, moints_iajb_aabb, moints_iajb_bbaa, moints_iajb_bbbb, moints_ijab_aaaa, moints_ijab_bbbb)
    cphf.tei_mo_type = 'partial'

    cphf.set_frequencies([float(frequency)])

    cphf.run(solver='explicit', hamiltonian=hamiltonian, spin=spin)

    assert len(cphf.frequencies) == len(cphf.results) == 1
    res = cphf.results[0]
    assert res.shape == (2, 2)
    bl = res[1, 0]
    tr = res[0, 1]
    diff = abs(abs(bl) - abs(tr))
    # Results should be symmetric w.r.t. interchange between operators
    # in the LR equations.
    thresh = 1.0e-14
    assert diff < thresh

    return bl
