"""Utility functions that are not core to calculating physical values."""

import os.path

import numpy as np

# Define for any Python version <= 3.3,
# See https://github.com/kachayev/fn.py/commit/391824c43fb388e0eca94e568ff62cc35b543ecb
import sys
if sys.version_info.major == 2 or sys.version_info.minor <= 3:
    import operator
    def accumulate(iterable, func=operator.add):
        """Return running totals.

        >>> acc = accumulate([1, 2, 3, 4, 5])
        >>> [x for x in acc]
        [1, 3, 6, 10, 15]
        >>> acc = accumulate([1, 2, 3, 4, 5], operator.mul)
        >>> [x for x in acc]
        [1, 2, 6, 24, 120]
        """
        it = iter(iterable)
        try:
            total = next(it)
        except StopIteration:
            return
        yield total
        for element in it:
            total = func(total, element)
            yield total
else:
    from itertools import accumulate


def form_results(vecs_property, vecs_response):
    assert vecs_property.shape[1:] == vecs_response.shape[1:]
    assert len(vecs_property.shape) == 3
    assert vecs_property.shape[2] == 1
    results = np.dot(vecs_property[:, :, 0], vecs_response[:, :, 0].T)
    return results


def np_load(filename):
    """Read a file using NumPy."""
    arr = np.load(filename)
    if isinstance(arr, np.lib.npyio.NpzFile):
        # Make the assumption that there's only a single array
        # present, even though *.npz files can hold multiple arrays.
        arr = arr.items()[0][1]
    return arr


def parse_int_file_2(filename, dim):
    mat = np.zeros(shape=(dim, dim))
    with open(filename) as fh:
        contents = fh.readlines()
    for line in contents:
        mu, nu, intval = [float(x) for x in line.split()]
        mu, nu = int(mu - 1), int(nu - 1)
        mat[mu, nu] = mat[nu, mu] = intval
    return mat


def repack_matrix_to_vector(mat):
    return np.reshape(mat, -1, order='F')


def repack_vector_to_matrix(vec, shape):
    return vec.reshape(shape, order='F')


def clean_dalton_label(original_label):
    """Operator/integral labels in DALTON are in uppercase and may have
    spaces in them; replace spaces with underscores and make all
    letters lowercase.

    >>> clean_dalton_label("PSO 002")
    "pso_002"
    """
    cleaned_label = original_label.lower().replace(' ', '_')
    return cleaned_label


def dalton_label_to_operator(label):

    label = clean_dalton_label(label)

    from .operators import Operator

    coord1_to_slice = {
        'x': 0, 'y': 1, 'z': 2,
    }
    coord2_to_slice = {
        'xx': 0, 'xy': 1, 'xz': 2, 'yy': 3, 'yz': 4, 'zz': 5,
        'yx': 1, 'zx': 2, 'zy': 4,
    }
    slice_to_coord1 = {v:k for (k, v) in coord1_to_slice.items()}

    # dipole length
    if 'diplen' in label:
        operator_label = 'dipole'
        _coord = label[0]
        slice_idx = coord1_to_slice[_coord]
        is_imaginary = False
        is_spin_dependent = False
    # dipole velocity
    elif 'dipvel' in label:
        operator_label = 'dipvel'
        _coord = label[0]
        slice_idx = coord1_to_slice[_coord]
        is_imaginary = True
        is_spin_dependent = False
    # angular momentum
    elif 'angmom' in label:
        operator_label = 'angmom'
        _coord = label[0]
        slice_idx = coord1_to_slice[_coord]
        is_imaginary = True
        is_spin_dependent = False
    # spin-orbit
    elif 'spnorb' in label:
        operator_label = 'spinorb'
        _coord = label[0]
        slice_idx = coord1_to_slice[_coord]
        is_imaginary = True
        is_spin_dependent = True
        _nelec = label[1]
        if _nelec in ('1', '2'):
            operator_label += _nelec
        # combined one- and two-electron
        elif _nelec in (' ', '_'):
            operator_label += 'c'
        else:
            pass
    # Fermi contact
    elif 'fc' in label:
        operator_label = 'fermi'
        _atomid = label[6:6+2]
        slice_idx = int(_atomid) - 1
        is_imaginary = False
        is_spin_dependent = True
    # spin-dipole
    elif 'sd' in label:
        operator_label = 'sd'
        _coord_atom = label[3:3+3]
        _coord = label[7]
        _atomid = (int(_coord_atom) - 1) // 3
        _coord_1 = (int(_coord_atom) - 1) % 3
        _coord_2 = slice_to_coord1[_coord_1] + _coord
        slice_idx = (6 * _atomid) + coord2_to_slice[_coord_2]
        is_imaginary = False
        is_spin_dependent = True
    # TODO SD+FC?
    # nucleus-orbit
    elif 'pso' in label:
        operator_label = 'pso'
        # TODO coord manipulation
        is_imaginary = True
        # TODO is this correct?
        is_spin_dependent = False
        # FIXME
        slice_idx = None
    else:
        operator_label = ''
        is_imaginary = None
        is_spin_dependent = None
        slice_idx = None

    operator = Operator(
        label=operator_label,
        is_imaginary=is_imaginary,
        is_spin_dependent=is_spin_dependent,
        slice_idx=slice_idx,
    )

    return operator


def get_reference_value_from_file(filename, hamiltonian, spin, frequency, label_1, label_2):
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
            if (l_hamiltonian == hamiltonian) and \
               (l_spin == spin) and \
               (l_frequency == frequency) and \
               (l_label_1 == label_1) and \
               (l_label_2 == label_2):
                ref = float(l_val)
                found = True

    if not found:
        raise ValueError("Could not find reference value")

    return ref


def read_file_occupations(filename):
    with open(filename) as fh:
        contents = fh.read().strip()
    tokens = contents.split()
    assert len(tokens) == 4
    nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = [int(x) for x in tokens]
    return [nocc_alph, nvirt_alph, nocc_beta, nvirt_beta]


def read_file_1(filename):
    elements = []
    with open(filename) as fh:
        n_elem = int(next(fh))
        for line in fh:
            elements.append(float(line))
    assert len(elements) == n_elem
    return np.array(elements, dtype=float)


def read_file_2(filename):
    elements = []
    with open(filename) as fh:
        n_rows, n_cols = [int(x) for x in next(fh).split()]
        for line in fh:
            elements.append(float(line))
    assert len(elements) == (n_rows * n_cols)
    # The last index is the fast index (cols).
    return np.reshape(np.array(elements, dtype=float), (n_rows, n_cols))


def read_file_3(filename):
    elements = []
    with open(filename) as fh:
        n_slices, n_rows, n_cols = [int(x) for x in next(fh).split()]
        for line in fh:
            elements.append(float(line))
    assert len(elements) == (n_rows * n_cols * n_slices)
    return np.reshape(np.array(elements, dtype=float), (n_slices, n_rows, n_cols))


def read_file_4(filename):
    elements = []
    with open(filename) as fh:
        n_d1, n_d2, n_d3, n_d4 = [int(x) for x in next(fh).split()]
        for line in fh:
            elements.append(float(line))
    assert len(elements) == (n_d1 * n_d2 * n_d3 * n_d4)
    return np.reshape(np.array(elements, dtype=float), (n_d1, n_d2, n_d3, n_d4))


def occupations_from_pyscf_mol(mol, C):
    norb = fix_mocoeffs_shape(C).shape[-1]
    nocc_a, nocc_b = mol.nelec
    nvirt_a, nvirt_b = norb - nocc_a, norb - nocc_b
    occupations = (nocc_a, nvirt_a, nocc_b, nvirt_b)
    return occupations


def occupations_from_sirifc(ifc):
    nocc_a, nocc_b = ifc.nisht + ifc.nasht, ifc.nisht
    norb = ifc.norbt
    nvirt_a, nvirt_b = norb - nocc_a, norb - nocc_b
    occupations = (nocc_a, nvirt_a, nocc_b, nvirt_b)
    return occupations


def occupations_from_psi4wfn(wfn):
    # Not needed.
    # occupations_a = wfn.occupation_a().to_array()
    # occupations_b = wfn.occupation_b().to_brray()
    # assert occupations_a.shape == occupations_b.shape
    norb = wfn.nmo()
    nocc_a = wfn.nalpha()
    nocc_b = wfn.nbeta()
    nvirt_a = norb - nocc_a
    nvirt_b = norb - nocc_b
    occupations = (nocc_a, nvirt_a, nocc_b, nvirt_b)
    return occupations


def mocoeffs_from_psi4wfn(wfn):
    is_uhf = not wfn.same_a_b_orbs()
    Ca = wfn.Ca().to_array()
    if is_uhf:
        Cb = wfn.Cb().to_array()
        C = np.stack((Ca, Cb), axis=0)
    else:
        C = Ca
    # Clean up.
    return fix_mocoeffs_shape(C)


def moenergies_from_psi4wfn(wfn):
    is_uhf = not wfn.same_a_b_orbs()
    Ea = wfn.epsilon_a().to_array()
    if is_uhf:
        Eb = wfn.epsilon_b().to_array()
        E = np.stack((Ea, Eb), axis=0).T
    else:
        E = Ea
    # Clean up.
    return fix_moenergies_shape(E)


class Splitter:
    """Split a line based not on a character, but a given number of field
    widths.
    """

    def __init__(self, widths):
        self.start_indices = [0] + list(accumulate(widths))[:-1]
        self.end_indices = list(accumulate(widths))

    def split(self, line, truncate=True):
        """Split the given line using the field widths passed in on class
        initialization.

        Handle lines that contain fewer fields than specified in the
        widths; they are added as empty strings. If `truncate`, remove
        them.
        """
        elements = [line[start:end].strip()
                    for (start, end) in zip(self.start_indices, self.end_indices)]
        if truncate:
            for i in range(1, len(elements)):
                if elements[-1] == '':
                    elements.pop()
                else:
                    break
        return elements


def fix_mocoeffs_shape(mocoeffs):
    if isinstance(mocoeffs, tuple):
        # this will properly fall through to the else clause
        mocoeffs_new = fix_mocoeffs_shape(np.stack(mocoeffs, axis=0))
    # assume np.ndarray
    else:
        shape = mocoeffs.shape
        assert len(shape) in (2, 3)
        if len(shape) == 2:
            mocoeffs_new = mocoeffs[np.newaxis, ...]
        else:
            mocoeffs_new = mocoeffs
    return mocoeffs_new


def fix_moenergies_shape(moenergies):
    if isinstance(moenergies, tuple):
        # this will properly fall through to the else clause
        moenergies_new = fix_moenergies_shape(np.stack(moenergies, axis=0))
    # assume np.ndarray
    else:
        shape = moenergies.shape
        ls = len(shape)
        assert ls in (1, 2, 3)
        if ls == 1:
            # It's a vector.
            moenergies_new = np.diag(moenergies)[np.newaxis, ...]
        elif ls == 2:
            # If it's a square matrix, assume it's already diagonal. If it
            # isn't a square matrix, then it probably has one or two
            # columns, one for each spin case.
            if shape[0] == shape[1]:
                moenergies_new = moenergies[np.newaxis, ...]
            else:
                assert shape[0] in (1, 2)
                if shape[0] == 1:
                    # (1, norb)
                    moenergies_new = np.diag(moenergies[:, 0])[np.newaxis, ...]
                else:
                    # (2, norb)
                    moenergies_alph = np.diag(moenergies[0, :])[np.newaxis, ...]
                    moenergies_beta = np.diag(moenergies[1, :])[np.newaxis, ...]
                    moenergies_new = np.concatenate((moenergies_alph, moenergies_beta), axis=0)
        else:
            assert shape[0] in (1, 2)
            assert shape[1] == shape[2]
            moenergies_new = moenergies
    return moenergies_new


def read_dalton_propfile(tmpdir):
    proplist = []
    with open(os.path.join(tmpdir, 'DALTON.PROP')) as propfile:
        proplines = propfile.readlines()
    splitter = Splitter([5, 3, 4, 11, 23, 9, 9, 9, 9, 23, 23, 23, 4, 4, 4])
    for line in proplines:
        sline = splitter.split(line)
        # print(sline)
        proplist.append(sline)
    return proplist


def tensor_printer(tensor):
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
    return (eigvals, iso, aniso)


def form_vec_energy_differences(moene_occ, moene_virt):
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


def screen(mat, thresh=1.0e-16):
    """Set all values smaller than the given threshold to zero
    (considering them as numerical noise).

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


def matsym(amat, thrzer=1.0e-14):
    """
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
        for i in range(j+1):
            amats = abs(amat[i, j] + amat[j, i])
            amata = abs(amat[i, j] - amat[j, i])
            if amats > thrzer:
                iasym = 0
            if amata > thrzer:
                isym = 0

    return (isym + iasym)


def flip_triangle_sign(A, triangle='lower'):
    """Flip the sign of either the lower or upper triangle of a sqare
    matrix. Assume nothing about its symmetry.

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
    if triangle == 'lower':
        indices = np.tril_indices(dim)
    elif triangle == 'upper':
        indices = np.triu_indices(dim)
    else:
        sys.exit(1)
    B = A.copy()
    B[indices] *= -1.0
    return B


def form_first_hyperpolarizability_averages(beta):
    assert beta.shape == (3, 3, 3)
    avgs = (-1 / 3) * (np.einsum('ijj->i', beta) +
                       np.einsum('jij->i', beta) +
                       np.einsum('jji->i', beta))
    avg = np.sum(avgs ** 2) ** (1 / 2)
    return avgs, avg
