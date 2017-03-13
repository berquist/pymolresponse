import numpy as np


def form_results(vecs_property, vecs_response):
    assert vecs_property.shape[1:] == vecs_response.shape[1:]
    assert len(vecs_property.shape) == 3
    assert vecs_property.shape[2] == 1
    results = np.dot(vecs_property[:, :, 0], vecs_response[:, :, 0].T)
    return results


def np_load(filename):
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


def dalton_label_to_operator(label):

    from cphf import Operator

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
        elif _nelec == ' ':
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

    operator = Operator(
        label=operator_label,
        is_imaginary=is_imaginary,
        is_spin_dependent=is_spin_dependent,
        slice_idx=slice_idx,
    )

    return operator
