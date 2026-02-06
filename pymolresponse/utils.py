"""Utility functions that are not core to calculating physical values."""

from collections.abc import Iterable
from itertools import accumulate
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from daltools.sirifc import sirifc

    from pymolresponse.indices import Occupations


def form_results(
    vecs_property: np.ndarray[tuple[int, int, int], np.dtype[np.floating]],
    vecs_response: np.ndarray[tuple[int, int, int], np.dtype[np.floating]],
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """Form all possible results by contracting response vectors with property gradients."""
    # TODO document what the assertions mean
    assert vecs_property.shape[1:] == vecs_response.shape[1:]
    assert len(vecs_property.shape) == 3
    assert vecs_property.shape[2] == 1
    results = np.dot(vecs_property[:, :, 0], vecs_response[:, :, 0].T)
    return results


def np_load(filename: str | Path) -> np.ndarray[tuple[int, ...], np.dtype[np.number]]:
    """Read a file using NumPy."""
    arr = np.load(filename)
    if isinstance(arr, np.lib.npyio.NpzFile):
        # Make the assumption that there's only a single array
        # present, even though *.npz files can hold multiple arrays.
        for _arr_name, _arr in arr.items():
            arr = _arr
            break
    return arr


def np_load_2(filename: str | Path) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """Read a file with a floating-point matrix using NumPy."""
    mat = np_load(filename)
    assert len(mat.shape) == 2
    # TODO assert dtype
    return mat  # ty: ignore[invalid-return-type]


def parse_int_file_2(
    filename: str | Path, dim: int
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """Read a 2-D array from a formatted text file.

    The first two columns are the one-based indices and the third column is
    the array element.
    """
    mat = np.zeros(shape=(dim, dim))
    with open(filename) as fh:
        contents = fh.readlines()
    for line in contents:
        mu, nu, intval = (float(x) for x in line.split())
        mu, nu = int(mu - 1), int(nu - 1)
        mat[mu, nu] = mat[nu, mu] = intval
    return mat


def repack_matrix_to_vector(mat: np.ndarray) -> np.ndarray:
    """Convert a matrix to a vector for compound indexing."""
    return np.reshape(mat, -1, order="F")


def repack_vector_to_matrix(vec: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Convert a vector with an assumed compound index into a dense matrix."""
    return vec.reshape(shape, order="F")


def get_reference_value_from_file(
    filename: Path | str,
    hamiltonian: str,
    spin: str,
    frequency: str,
    label_1: str,
    label_2: str,
) -> float:
    """Find a reference value for the linear response of a system to two operators."""
    # TODO need to pass the frequency as a string identical to the one
    # found in the file, can't pass a float due to fp error; how to
    # get around this?
    found = False
    with open(filename) as fh:
        for line in fh:
            tokens = line.split()
            # no comments allowed for now
            assert len(tokens) == 6
            l_hamiltonian, l_spin, l_frequency, l_label_1, l_label_2, l_val = tokens
            if (
                (l_hamiltonian == hamiltonian)
                and (l_spin == spin)
                and (l_frequency == frequency)
                and (l_label_1 == label_1)
                and (l_label_2 == label_2)
            ):
                ref = float(l_val)
                found = True

    if not found:
        raise ValueError("Could not find reference value")

    return ref


def read_file_occupations(filename: Path | str) -> "Occupations":
    """Read molecular orbital occupations from a file.

    The file should contain a single line of four integers
    - number of occupied alpha orbitals
    - number of unoccupied (virtual) alpha orbitals
    - number of occupied beta orbitals
    - number of unoccupied (virtual) beta orbitals
    which are parsed and returned in an array in the same order.

    The file makes no distinction about whether or not they come from a
    restricted calculation, in which case the alpha and beta values will be
    identical.
    """
    with open(filename) as fh:
        contents = fh.read().strip()
    tokens = contents.split()
    assert len(tokens) == 4
    return tuple(int(x) for x in tokens)  # ty: ignore[invalid-return-type]


def read_file_1(filename: Path | str) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    """Read a libaview-formatted 1-D array."""
    elements = []
    with open(filename) as fh:
        n_elem = int(next(fh))
        for line in fh:
            elements.append(float(line))
    assert len(elements) == n_elem
    return np.array(elements, dtype=float)


def read_file_2(filename: Path | str) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """Read a libaview-formatted 2-D array."""
    elements = []
    with open(filename) as fh:
        n_rows, n_cols = (int(x) for x in next(fh).split())
        for line in fh:
            elements.append(float(line))
    assert len(elements) == (n_rows * n_cols)
    # The last index is the fast index (cols).
    return np.reshape(np.array(elements, dtype=float), (n_rows, n_cols))


def read_file_3(
    filename: Path | str,
) -> np.ndarray[tuple[int, int, int], np.dtype[np.floating]]:
    """Read a libaview-formatted 3-D array."""
    elements = []
    with open(filename) as fh:
        n_slices, n_rows, n_cols = (int(x) for x in next(fh).split())
        for line in fh:
            elements.append(float(line))
    assert len(elements) == (n_rows * n_cols * n_slices)
    return np.reshape(np.array(elements, dtype=float), (n_slices, n_rows, n_cols))


def read_file_4(
    filename: Path | str,
) -> np.ndarray[tuple[int, int, int, int], np.dtype[np.floating]]:
    """Read a libaview-formatted 4-D array."""
    elements = []
    with open(filename) as fh:
        n_d1, n_d2, n_d3, n_d4 = (int(x) for x in next(fh).split())
        for line in fh:
            elements.append(float(line))
    assert len(elements) == (n_d1 * n_d2 * n_d3 * n_d4)
    return np.reshape(np.array(elements, dtype=float), (n_d1, n_d2, n_d3, n_d4))


def occupations_from_sirifc(ifc: "sirifc") -> "Occupations":
    """Read orbital occupations from a parsed DALTON SIRIFC object."""
    nocc_a, nocc_b = ifc.nisht + ifc.nasht, ifc.nisht
    norb = ifc.norbt
    nvirt_a, nvirt_b = norb - nocc_a, norb - nocc_b
    return nocc_a, nvirt_a, nocc_b, nvirt_b


class Splitter:
    """Split a line based on a number of field widths."""

    def __init__(self, widths: Iterable[int]) -> None:
        self.start_indices = [0] + list(accumulate(widths))[:-1]
        self.end_indices = list(accumulate(widths))

    def split(self, line: str, truncate: bool = True) -> list[str]:
        """Split a line using field widths passed on class initialization.

        Handle lines that contain fewer fields than specified in the
        widths; they are added as empty strings. If `truncate`, remove
        them.
        """
        elements = [
            line[start:end].strip() for (start, end) in zip(self.start_indices, self.end_indices)
        ]
        if truncate:
            for _ in range(1, len(elements)):
                if elements[-1] == "":
                    elements.pop()
                else:
                    break
        return elements


DirtyMocoeffs = (
    tuple[np.ndarray[tuple[int, ...], np.dtype[np.floating]], ...]
    | np.ndarray[tuple[int, int] | tuple[int, int, int], np.dtype[np.floating]],
)


def fix_mocoeffs_shape(
    mocoeffs: DirtyMocoeffs,
) -> np.ndarray[tuple[int, int, int], np.dtype[np.floating]]:
    """Clean up the dimensionality of molecular orbital coefficients.

    The result is 3-D, where the first index is spin, the second is the atomic
    orbital, and the third is the molecular orbital.  The first index will
    only ever be of length 1 or 2.
    """
    if isinstance(mocoeffs, tuple):
        # this will properly fall through to the else clause
        mocoeffs_new = fix_mocoeffs_shape(np.stack(mocoeffs, axis=0))
    # assume np.ndarray
    else:
        shape = mocoeffs.shape
        assert len(shape) in (2, 3)
        if len(shape) == 2:
            mocoeffs_new = mocoeffs[np.newaxis]
        else:
            mocoeffs_new = mocoeffs
    assert len(mocoeffs_new.shape) == 3
    return mocoeffs_new  # ty: ignore[invalid-return-type]


def fix_moenergies_shape(
    moenergies: tuple[np.ndarray[tuple[int, ...], np.dtype[np.floating]], ...]
    | np.ndarray[tuple[int] | tuple[int, int] | tuple[int, int, int], np.dtype[np.floating]],
) -> np.ndarray[tuple[int, int, int], np.dtype[np.floating]]:
    """Clean up the dimensionality of molecular orbital energies.

    The result is 3-D, where the first index is spin and the second and third
    indices are molecular orbitals.  For canonical orbitals, all off-diagonal
    elements will be zero, but it is more convenient to have the shape be a
    matrix rather than a vector.
    """
    if isinstance(moenergies, tuple):
        # this will properly fall through to the else clause
        moenergies_new = fix_moenergies_shape(np.stack(moenergies, axis=0))
    else:
        shape = moenergies.shape
        ls = len(shape)
        assert ls in (1, 2, 3)
        if ls == 1:
            # It's a vector.
            moenergies_new = np.diag(moenergies)[np.newaxis]
        elif ls == 2:
            # If it's a square matrix, assume it's already diagonal.  If it
            # isn't a square matrix, then it probably has one or two columns,
            # one for each spin case.  TODO check that all off-diagonal
            # elements are zero?  Not true for Fock matrix in non-orthogonal
            # basis.
            if shape[0] == shape[1]:  # ty: ignore[index-out-of-bounds]
                moenergies_new = moenergies[np.newaxis]
            else:
                assert shape[0] in (1, 2)
                if shape[0] == 1:
                    # (1, norb)
                    # FIXME swapped?
                    moenergies_new = np.diag(moenergies[:, 0])[np.newaxis]
                else:
                    # (2, norb)
                    moenergies_alph = np.diag(moenergies[0, :])[np.newaxis]
                    moenergies_beta = np.diag(moenergies[1, :])[np.newaxis]
                    moenergies_new = np.concatenate((moenergies_alph, moenergies_beta), axis=0)
        else:
            assert shape[0] in (1, 2)
            # You might think at first glance there's an assumption that nbsf
            # == nmo here, but the (Fock) matrix is entirely in the MO basis.
            assert shape[1] == shape[2]  # ty: ignore[index-out-of-bounds]
            moenergies_new = moenergies
    assert len(moenergies_new.shape) == 3
    assert moenergies_new.shape[0] in (1, 2)
    assert moenergies_new.shape[1] == moenergies_new.shape[2]  # ty: ignore[index-out-of-bounds]
    return moenergies_new  # ty: ignore[invalid-return-type]


def read_dalton_propfile(tmpdir: Path) -> list[str]:
    """Parse a DALTON.PROP file."""
    proplist = []
    with open(tmpdir / "DALTON.PROP") as propfile:
        proplines = propfile.readlines()
    splitter = Splitter([5, 3, 4, 11, 23, 9, 9, 9, 9, 23, 23, 23, 4, 4, 4])
    for line in proplines:
        sline = splitter.split(line)
        # print(sline)
        proplist.append(sline)
    return proplist


def tensor_printer(
    tensor: np.ndarray[tuple[int, int], np.dtype[np.floating]],
) -> tuple[np.ndarray[tuple[int], np.dtype[np.floating]], np.floating, float]:
    """Pretty-print a 2-D array."""
    print(tensor)
    eigvals = np.linalg.eigvals(tensor)
    # or should this be the trace of the matrix?
    iso = np.average(eigvals)
    # anisotropic = 0.e0;
    # for(int j1=0;j1 < 9; j1++)
    #   anisotropic += statpol[j1]*statpol[j1];
    # anisotropic = sqrt(fabs(1.5e0 * (anisotropic - 3*isotropic * isotropic)));
    # aniso = np.sum(tensor ** 2)
    aniso = 0.0
    print(eigvals)
    print(iso)
    # print(np.trace(tensor) / tensor.shape[0])
    # print(aniso)
    return eigvals, iso, aniso


def form_vec_energy_differences(
    moene_occ: np.ndarray[tuple[int], np.dtype[np.floating]],
    moene_virt: np.ndarray[tuple[int], np.dtype[np.floating]],
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    """Form a vector of virtual-occupied MO energy differences.

    In the compound-indexed result vector, the virtual index is fast and the
    occupied index is slow.
    """
    nocc = moene_occ.shape[0]
    nvirt = moene_virt.shape[0]
    nov = nocc * nvirt
    # The stupid loop is faster!
    ediff = np.zeros(nov)
    for i in range(nocc):
        for a in range(nvirt):
            ia = (i * nvirt) + a
            ediff[ia] = moene_virt[a] - moene_occ[i]
    # eo = np.einsum('ij,ab->iajb', np.diag(moene_occ), np.diag(np.ones(nvirt)))
    # ev = np.einsum('ab,ij->iajb', np.diag(moene_virt), np.diag(np.ones(nocc)))
    # ediff = np.diag((ev - eo).reshape(nov, nov))
    return ediff


def screen(
    mat: np.ndarray[tuple[int, ...], np.dtype[np.number]], thresh: float = 1.0e-16
) -> np.ndarray[tuple[int, ...], np.dtype[np.number]]:
    """Set all values smaller than the threshold to zero.

    This function makes a copy of mat and does not modify it in-place.

    Parameters
    ----------
    mat : np.ndarray
    thresh : float
        Threshold below which all elements of `mat` smaller than
        `thresh` are set to zero.

    Returns
    -------
    np.ndarray
    """
    mat_screened = mat.copy()
    mat_screened[np.abs(mat) <= thresh] = 0.0
    return mat_screened


def matsym(amat: np.ndarray[tuple[int, int], np.dtype[np.number]], thrzer: float = 1.0e-14) -> int:
    """Determine matrix symmetry.

    - Copied from ``DALTON/gp/gphjj.F/MATSYM``.
    - `thrzer` taken from ``DALTON/include/thrzer.h``.

    Parameters
    ----------
    amat : np.ndarray
    thrzer : float
        Threshold below which a (floating point) number is considered
        zero.

    Returns
    -------
    int
        - 1 if the matrix is symmetric to threshold `thrzer`
        - 2 if the matrix is antisymmetric to threshold `thrzer`
        - 3 if all elements are below `thrzer`
        - 0 otherwise (the matrix is unsymmetric about the diagonal)
    """
    assert amat.shape[0] == amat.shape[1]

    n = amat.shape[0]

    isym = 1
    iasym = 2
    for j in range(n):
        # for i in range(j+1):
        # The +1 is so the diagonal elements are checked.
        for i in range(j + 1):
            amats = abs(amat[i, j] + amat[j, i])
            amata = abs(amat[i, j] - amat[j, i])
            if amats > thrzer:
                iasym = 0
            if amata > thrzer:
                isym = 0

    return isym + iasym


def flip_triangle_sign(
    A: np.ndarray[tuple[int, int], np.dtype[np.number]], triangle: str = "lower"
) -> np.ndarray[tuple[int, int], np.dtype[np.number]]:
    """Flip the sign of either the lower or upper triangle of a square matrix.

    Assumes nothing about the input matrix symmetry.

    Parameters
    ----------
    A : np.ndarray
    triangle : {'lower', 'upper'}

    Returns
    -------
    np.ndarray
    """
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    dim = A.shape[0]
    if triangle == "lower":
        indices = np.tril_indices(dim)
    elif triangle == "upper":
        indices = np.triu_indices(dim)
    else:
        raise ValueError("argument to triangle must be 'upper' or 'lower'")
    B = A.copy()
    B[indices] *= -1.0
    return B


def form_first_hyperpolarizability_averages(
    beta: np.ndarray[tuple[int, int, int], np.dtype[np.floating]],
) -> tuple[np.ndarray[tuple[int], np.dtype[np.floating]], np.floating]:
    """Form the relevant averages from a complete (no symmetry) hyperpolarizability tensor."""
    assert beta.shape == (3, 3, 3)
    avgs = (-1 / 3) * (
        np.einsum("ijj->i", beta) + np.einsum("jij->i", beta) + np.einsum("jji->i", beta)
    )
    avg = np.sum(avgs**2) ** (1 / 2)
    return avgs, avg


def form_indices_orbwin(nocc: int, nvirt: int) -> list[tuple[int, int]]:
    """Form all occupied-virtual pairs of indices starting from their absolute position."""
    norb = nocc + nvirt
    return [(i, a) for i in range(0, nocc) for a in range(nocc, norb)]


def form_indices_zero(nocc: int, nvirt: int) -> list[tuple[int, int]]:
    """Form all occupied-virtual pairs of indices, both starting from zero."""
    return [(i, a) for i in range(nocc) for a in range(nvirt)]
