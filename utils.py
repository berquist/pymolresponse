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
