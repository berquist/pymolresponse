r"""Explicit equations for orbital Hessian terms using partially-transformed MO-basis two-electron integrals, *e.g.*, :math:`(ia|jb), (ij|ab)`."""

import numpy as np

from pymolresponse.utils import form_vec_energy_differences


def form_rpa_a_matrix_mo_singlet_partial(
    E_MO: np.ndarray, TEI_MO_iajb: np.ndarray, TEI_MO_ijab: np.ndarray
) -> np.ndarray:
    r"""Form the A (CIS) matrix in the MO basis. [singlet]

    The equation for element :math:`\{ia,jb\}` is
    :math:`\left<aj||ib\right> = \left<aj|ib\right> -
    \left<aj|bi\right> = [ai|jb] - [ab|ji] = 2(ai|jb) - (ab|ji)`. It
    also includes the virt-occ energy difference on the diagonal.
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

    ediff = form_vec_energy_differences(np.diag(E_MO)[:nocc], np.diag(E_MO)[nocc:])

    A = 2 * TEI_MO_iajb
    A -= TEI_MO_ijab.swapaxes(1, 2)
    A.shape = (nov, nov)

    A += np.diag(ediff)

    return A


def form_rpa_a_matrix_mo_triplet_partial(E_MO: np.ndarray, TEI_MO_ijab: np.ndarray) -> np.ndarray:
    r"""Form the A (CIS) matrix in the MO basis. [triplet]

    The equation for element :math:`\{ia,jb\}` is :math:`-
    \left<aj|bi\right> = - [ab|ji] = - (ab|ji)`. It also includes the
    virt-occ energy difference on the diagonal.
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

    ediff = form_vec_energy_differences(np.diag(E_MO)[:nocc], np.diag(E_MO)[nocc:])

    A = np.zeros((nocc, nvirt, nocc, nvirt))
    A -= TEI_MO_ijab.swapaxes(1, 2)
    A.shape = (nov, nov)

    A += np.diag(ediff)

    return A


def form_rpa_b_matrix_mo_singlet_partial(TEI_MO_iajb: np.ndarray) -> np.ndarray:
    r"""Form the B matrix for RPA in the MO basis. [singlet]

    The equation for element :math:`\{ia,jb\}` is
    :math:`\left<ab||ij\right> = \left<ab|ij\right> -
    \left<ab|ji\right> = [ai|bj] - [aj|bi] = 2(ai|bj) - (aj|bi)`.
    """

    shape_iajb = TEI_MO_iajb.shape
    assert len(shape_iajb) == 4
    assert shape_iajb[0] == shape_iajb[2]
    assert shape_iajb[1] == shape_iajb[3]
    nocc = shape_iajb[0]
    nvirt = shape_iajb[1]
    nov = nocc * nvirt

    B = 2 * TEI_MO_iajb
    B -= TEI_MO_iajb.swapaxes(1, 3)
    B.shape = (nov, nov)

    return -B


def form_rpa_b_matrix_mo_triplet_partial(TEI_MO_iajb: np.ndarray) -> np.ndarray:
    r"""Form the B matrix for RPA in the MO basis. [triplet]

    The equation for element :math:`\{ia,jb\}` is :math:`????`.
    """

    shape_iajb = TEI_MO_iajb.shape
    assert len(shape_iajb) == 4
    assert shape_iajb[0] == shape_iajb[2]
    assert shape_iajb[1] == shape_iajb[3]
    nocc = shape_iajb[0]
    nvirt = shape_iajb[1]
    nov = nocc * nvirt

    B = np.zeros((nocc, nvirt, nocc, nvirt))
    B -= TEI_MO_iajb.swapaxes(1, 3)
    B.shape = (nov, nov)

    return -B


def form_rpa_a_matrix_mo_singlet_ss_partial(
    E_MO: np.ndarray, TEI_MO_iajb: np.ndarray, TEI_MO_ijab: np.ndarray
) -> np.ndarray:
    r"""Form the same-spin part of the A (CIS) matrix in the MO
    basis. [singlet]

    The equation for element :math:`\{ia,jb\}` is :math:`????`.
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

    ediff = form_vec_energy_differences(np.diag(E_MO)[:nocc], np.diag(E_MO)[nocc:])

    A = TEI_MO_iajb.copy()
    A -= TEI_MO_ijab.swapaxes(1, 2)
    A.shape = (nov, nov)

    A += np.diag(ediff)

    return A


def form_rpa_a_matrix_mo_singlet_os_partial(TEI_MO_iajb_xxyy: np.ndarray) -> np.ndarray:
    r"""Form the opposite-spin part of the A (CIS) matrix in the MO
    basis. [singlet]

    The equation for element :math:`\{ia,jb\}` is :math:`????`.
    """

    shape = TEI_MO_iajb_xxyy.shape
    assert len(shape) == 4
    nocc_x, nvirt_x, nocc_y, nvirt_y = shape
    nov_x = nocc_x * nvirt_x
    nov_y = nocc_y * nvirt_y

    A = TEI_MO_iajb_xxyy.copy()
    A.shape = (nov_x, nov_y)

    return A


def form_rpa_b_matrix_mo_singlet_ss_partial(TEI_MO_iajb: np.ndarray) -> np.ndarray:
    r"""Form the same-spin part of the RPA B matrix in the MO
    basis. [singlet]

    The equation for element :math:`\{ia,jb\}` is :math:`????`.
    """

    shape_iajb = TEI_MO_iajb.shape
    assert len(shape_iajb) == 4
    assert shape_iajb[0] == shape_iajb[2]
    assert shape_iajb[1] == shape_iajb[3]
    nocc = shape_iajb[0]
    nvirt = shape_iajb[1]
    nov = nocc * nvirt

    B = TEI_MO_iajb.copy()
    B -= TEI_MO_iajb.swapaxes(1, 3)
    B.shape = (nov, nov)

    return -B


def form_rpa_b_matrix_mo_singlet_os_partial(TEI_MO_iajb_xxyy: np.ndarray) -> np.ndarray:
    r"""Form the opposite-spin part of the RPA B matrix in the MO
    basis. [singlet]

    The equation for element :math:`\{ia,jb\}` is :math:`????`.
    """

    shape = TEI_MO_iajb_xxyy.shape
    assert len(shape) == 4
    nocc_x, nvirt_x, nocc_y, nvirt_y = shape
    nov_x = nocc_x * nvirt_x
    nov_y = nocc_y * nvirt_y

    B = TEI_MO_iajb_xxyy.copy()
    B.shape = (nov_x, nov_y)

    return -B
