import numpy as np

from .utils import form_vec_energy_differences


def form_rpa_a_matrix_mo_singlet_partial(E_MO, TEI_MO_iajb, TEI_MO_ijab):
    """Form the A (CIS) matrix for RPA in the molecular orbital (MO)
    basis. [singlet]

    The equation for element {ia,jb} is <aj||ib> = <aj|ib> - <aj|bi> =
    [ai|jb] - [ab|ji] = 2(ai|jb) - (ab|ji). It also includes the
    virt-occ energy difference on the diagonal.
    """

    shape_iajb = TEI_MO_iajb.shape
    shape_ijab = TEI_MO_ijab.shape
    assert len(shape_iajb) == len(shape_ijab) == 4
    assert shape_iajb[0] == shape_iajb[2] == shape_ijab[0] == shape_ijab[1]
    assert shape_iajb[1] == shape_iajb[3] == shape_ijab[2] == shape_ijab[3]
    nocc = shape_iajb[0]
    nvirt = shape_iajb[1]
    norb = nocc + nvirt
    assert len(E_MO.shape) == 2
    assert E_MO.shape[0] == E_MO.shape[1] == norb
    nov = nocc * nvirt

    ediff = form_vec_energy_differences(np.diag(E_MO)[:nocc],
                                        np.diag(E_MO)[nocc:])

    # A = np.empty(shape=(nov, nov))

    # for i in range(nocc):
    #     for a in range(nvirt):
    #         ia = i*nvirt + a
    #         for j in range(nocc):
    #             for b in range(nvirt):
    #                 jb = j*nvirt + b
    #                 A[ia, jb] = 2*TEI_MO_iajb[i, a, j, b] - TEI_MO_ijab[i, j, a, b]
    #                 if (ia == jb):
    #                     A[ia, jb] += (E_MO[a + nocc, b + nocc] - E_MO[i, j])

    A = 2 * TEI_MO_iajb
    A -= TEI_MO_ijab.swapaxes(1, 2)
    A.shape = (nov, nov)

    A += np.diag(ediff)

    return A


def form_rpa_a_matrix_mo_triplet_partial(E_MO, TEI_MO_ijab):
    """Form the A (CIS) matrix for RPA in the molecular orbital (MO)
    basis. [triplet]

    The equation for element {ia,jb} is - <aj|bi> = - [ab|ji] = -
    (ab|ji). It also includes the virt-occ energy difference on the
    diagonal.
    """

    shape_ijab = TEI_MO_ijab.shape
    assert len(shape_ijab) == 4
    assert shape_ijab[0] == shape_ijab[1]
    assert shape_ijab[2] == shape_ijab[3]
    nocc = shape_ijab[0]
    nvirt = shape_ijab[2]
    norb = nocc + nvirt
    assert len(E_MO.shape) == 2
    assert E_MO.shape[0] == E_MO.shape[1] == norb
    nov = nocc * nvirt

    # A = np.empty(shape=(nov, nov))

    # for i in range(nocc):
    #     for a in range(nvirt):
    #         ia = i*nvirt + a
    #         for j in range(nocc):
    #             for b in range(nvirt):
    #                 jb = j*nvirt + b
    #                 A[ia, jb] = - TEI_MO_ijab[i, j, a, b]
    #                 if (ia == jb):
    #                     A[ia, jb] += (E_MO[a + nocc, b + nocc] - E_MO[i, j])

    ediff = form_vec_energy_differences(np.diag(E_MO)[:nocc],
                                        np.diag(E_MO)[nocc:])

    A = np.zeros((nocc, nvirt, nocc, nvirt))
    A -= TEI_MO_ijab.swapaxes(1, 2)
    A.shape = (nov, nov)

    A += np.diag(ediff)

    return A


def form_rpa_b_matrix_mo_singlet_partial(TEI_MO_iajb):
    """Form the B matrix for RPA in the molecular orbital (MO)
    basis. [singlet]

    The equation for element {ia,jb} is <ab||ij> = <ab|ij> - <ab|ji> =
    [ai|bj] - [aj|bi] = 2*(ai|bj) - (aj|bi).
    """

    shape_iajb = TEI_MO_iajb.shape
    assert len(shape_iajb) == 4
    assert shape_iajb[0] == shape_iajb[2]
    assert shape_iajb[1] == shape_iajb[3]
    nocc = shape_iajb[0]
    nvirt = shape_iajb[1]
    nov = nocc * nvirt

    # B = np.empty(shape=(nov, nov))

    # for i in range(nocc):
    #     for a in range(nvirt):
    #         ia = i*nvirt + a
    #         for j in range(nocc):
    #             for b in range(nvirt):
    #                 jb = j*nvirt + b
    #                 B[ia, jb] = 2*TEI_MO_iajb[i, a, j, b] - TEI_MO_iajb[i, b, j, a]

    B = 2 * TEI_MO_iajb
    B -= TEI_MO_iajb.swapaxes(1, 3)
    B.shape = (nov, nov)

    return B


def form_rpa_b_matrix_mo_triplet_partial(TEI_MO_iajb):

    shape_iajb = TEI_MO_iajb.shape
    assert len(shape_iajb) == 4
    assert shape_iajb[0] == shape_iajb[2]
    assert shape_iajb[1] == shape_iajb[3]
    nocc = shape_iajb[0]
    nvirt = shape_iajb[1]
    nov = nocc * nvirt

    # B = np.empty(shape=(nov, nov))

    # for i in range(nocc):
    #     for a in range(nvirt):
    #         ia = i*nvirt + a
    #         for j in range(nocc):
    #             for b in range(nvirt):
    #                 jb = j*nvirt + b
    #                 B[ia, jb] = - TEI_MO_iajb[i, b, j, a]

    B = np.zeros((nocc, nvirt, nocc, nvirt))
    B -= TEI_MO_iajb.swapaxes(1, 3)
    B.shape = (nov, nov)

    return B


def form_rpa_a_matrix_mo_singlet_ss_partial(E_MO, TEI_MO_iajb, TEI_MO_ijab):

    shape_iajb = TEI_MO_iajb.shape
    shape_ijab = TEI_MO_ijab.shape
    assert len(shape_iajb) == len(shape_ijab) == 4
    assert shape_iajb[0] == shape_iajb[2] == shape_ijab[0] == shape_ijab[1]
    assert shape_iajb[1] == shape_iajb[3] == shape_ijab[2] == shape_ijab[3]
    nocc = shape_iajb[0]
    nvirt = shape_iajb[1]
    norb = nocc + nvirt
    assert len(E_MO.shape) == 2
    assert E_MO.shape[0] == E_MO.shape[1] == norb
    nov = nocc * nvirt

    ediff = form_vec_energy_differences(np.diag(E_MO)[:nocc],
                                        np.diag(E_MO)[nocc:])

    # A = np.empty(shape=(nov, nov))

    # for i in range(nocc):
    #     for a in range(nvirt):
    #         ia = i*nvirt + a
    #         for j in range(nocc):
    #             for b in range(nvirt):
    #                 jb = j*nvirt + b
    #                 A[ia, jb] = TEI_MO_iajb[i, a, j, b] - TEI_MO_ijab[i, j, a, b]
    #                 if (ia == jb):
    #                     A[ia, jb] += (E_MO[a + nocc, b + nocc] - E_MO[i, j])

    A = TEI_MO_iajb.copy()
    A -= TEI_MO_ijab.swapaxes(1, 2)
    A.shape = (nov, nov)

    A += np.diag(ediff)

    return A


def form_rpa_a_matrix_mo_singlet_os_partial(TEI_MO_iajb_xxyy):

    shape = TEI_MO_iajb_xxyy.shape
    assert len(shape) == 4
    nocc_x, nvirt_x, nocc_y, nvirt_y = shape
    nov_x = nocc_x * nvirt_x
    nov_y = nocc_y * nvirt_y

    # A = np.empty(shape=(nov_x, nov_y))

    # for i in range(nocc_x):
    #     for a in range(nvirt_x):
    #         ia = i*nvirt_x + a
    #         for j in range(nocc_y):
    #             for b in range(nvirt_y):
    #                 jb = j*nvirt_y + b
    #                 A[ia, jb] = TEI_MO_iajb_xxyy[i, a, j, b]

    A = TEI_MO_iajb_xxyy.copy()
    A.shape = (nov_x, nov_y)

    return A


def form_rpa_b_matrix_mo_singlet_ss_partial(TEI_MO_iajb):

    shape_iajb = TEI_MO_iajb.shape
    assert len(shape_iajb) == 4
    assert shape_iajb[0] == shape_iajb[2]
    assert shape_iajb[1] == shape_iajb[3]
    nocc = shape_iajb[0]
    nvirt = shape_iajb[1]
    nov = nocc * nvirt

    # B = np.empty(shape=(nov, nov))

    # for i in range(nocc):
    #     for a in range(nvirt):
    #         ia = i*nvirt + a
    #         for j in range(nocc):
    #             for b in range(nvirt):
    #                 jb = j*nvirt + b
    #                 B[ia, jb] = TEI_MO_iajb[i, a, j, b] - TEI_MO_iajb[i, b, j, a]

    B = TEI_MO_iajb.copy()
    B -= TEI_MO_iajb.swapaxes(1, 3)
    B.shape = (nov, nov)

    return B


def form_rpa_b_matrix_mo_singlet_os_partial(TEI_MO_iajb_xxyy):

    shape = TEI_MO_iajb_xxyy.shape
    assert len(shape) == 4
    nocc_x, nvirt_x, nocc_y, nvirt_y = shape
    nov_x = nocc_x * nvirt_x
    nov_y = nocc_y * nvirt_y

    # B = np.empty(shape=(nov_x, nov_y))

    # for i in range(nocc_x):
    #     for a in range(nvirt_x):
    #         ia = i*nvirt_x + a
    #         for j in range(nocc_y):
    #             for b in range(nvirt_y):
    #                 jb = j*nvirt_y + b
    #                 B[ia, jb] = TEI_MO_iajb_xxyy[i, a, j, b]

    B = TEI_MO_iajb_xxyy.copy()
    B.shape = (nov_x, nov_y)

    return B
