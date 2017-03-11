import numpy as np


def form_rpa_a_matrix_mo_singlet(E_MO, TEI_MO_iajb, TEI_MO_ijab):
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

    A = np.empty(shape=(nov, nov))

    for i in range(nocc):
        for a in range(nvirt):
            ia = i*nvirt + a
            for j in range(nocc):
                for b in range(nvirt):
                    jb = j*nvirt + b
                    A[ia, jb] = 2*TEI_MO_iajb[i, a, j, b] - TEI_MO_ijab[i, j, a, b]
                    if (ia == jb):
                        A[ia, jb] += (E_MO[a + nocc, b + nocc] - E_MO[i, j])

    return A


def form_rpa_a_matrix_mo_triplet(E_MO, TEI_MO_ijab):
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

    A = np.empty(shape=(nov, nov))

    for i in range(nocc):
        for a in range(nvirt):
            ia = i*nvirt + a
            for j in range(nocc):
                for b in range(nvirt):
                    jb = j*nvirt + b
                    A[ia, jb] = - TEI_MO_ijab[i, j, a, b]
                    if (ia == jb):
                        A[ia, jb] += (E_MO[a + nocc, b + nocc] - E_MO[i, j])

    return A


def form_rpa_b_matrix_mo_singlet(TEI_MO_iajb):
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

    B = np.empty(shape=(nov, nov))

    for i in range(nocc):
        for a in range(nvirt):
            ia = i*nvirt + a
            for j in range(nocc):
                for b in range(nvirt):
                    jb = j*nvirt + b
                    B[ia, jb] = 2*TEI_MO_iajb[i, a, j, b] - TEI_MO_iajb[i, b, j, a]

    return B


def form_rpa_b_matrix_mo_triplet(TEI_MO_iajb):

    shape_iajb = TEI_MO_iajb.shape
    assert len(shape_iajb) == 4
    assert shape_iajb[0] == shape_iajb[2]
    assert shape_iajb[1] == shape_iajb[3]
    nocc = shape_iajb[0]
    nvirt = shape_iajb[1]
    nov = nocc * nvirt

    B = np.empty(shape=(nov, nov))

    for i in range(nocc):
        for a in range(nvirt):
            ia = i*nvirt + a
            for j in range(nocc):
                for b in range(nvirt):
                    jb = j*nvirt + b
                    B[ia, jb] = - TEI_MO_iajb[i, b, j, a]

    return B


def form_rpa_a_matrix_mo_singlet_ss(E_MO, TEI_MO_iajb, TEI_MO_ijab):

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

    A = np.empty(shape=(nov, nov))

    for i in range(nocc):
        for a in range(nvirt):
            ia = i*nvirt + a
            for j in range(nocc):
                for b in range(nvirt):
                    jb = j*nvirt + b
                    A[ia, jb] = TEI_MO_iajb[i, a, j, b] - TEI_MO_ijab[i, j, a, b]
                    if (ia == jb):
                        A[ia, jb] += (E_MO[a + nocc, b + nocc] - E_MO[i, j])

    return A


def form_rpa_a_matrix_mo_singlet_os(TEI_MO_iajb_xxyy):

    shape = TEI_MO_iajb_xxyy.shape
    assert len(shape) == 4
    nocc_x, nvirt_x, nocc_y, nvirt_y = shape
    nov_x = nocc_x * nvirt_x
    nov_y = nocc_y * nvirt_y

    A = np.empty(shape=(nov_x, nov_y))

    for i in range(nocc_x):
        for a in range(nvirt_x):
            ia = i*nvirt_x + a
            for j in range(nocc_y):
                for b in range(nvirt_y):
                    jb = j*nvirt_y + b
                    A[ia, jb] = TEI_MO_iajb_xxyy[i, a, j, b]

    return A


def form_rpa_b_matrix_mo_singlet_ss(TEI_MO_iajb):

    shape_iajb = TEI_MO_iajb.shape
    assert len(shape_iajb) == 4
    assert shape_iajb[0] == shape_iajb[2]
    assert shape_iajb[1] == shape_iajb[3]
    nocc = shape_iajb[0]
    nvirt = shape_iajb[1]
    nov = nocc * nvirt

    B = np.empty(shape=(nov, nov))

    for i in range(nocc):
        for a in range(nvirt):
            ia = i*nvirt + a
            for j in range(nocc):
                for b in range(nvirt):
                    jb = j*nvirt + b
                    B[ia, jb] = TEI_MO_iajb[i, a, j, b] - TEI_MO_iajb[i, b, j, a]

    return B


def form_rpa_b_matrix_mo_singlet_os(TEI_MO_iajb_xxyy):

    shape = TEI_MO_iajb_xxyy.shape
    assert len(shape) == 4
    nocc_x, nvirt_x, nocc_y, nvirt_y = shape
    nov_x = nocc_x * nvirt_x
    nov_y = nocc_y * nvirt_y

    B = np.empty(shape=(nov_x, nov_y))

    for i in range(nocc_x):
        for a in range(nvirt_x):
            ia = i*nvirt_x + a
            for j in range(nocc_y):
                for b in range(nvirt_y):
                    jb = j*nvirt_y + b
                    B[ia, jb] = TEI_MO_iajb_xxyy[i, a, j, b]

    return B
