import numpy as np

STARS = '********'


def read_binary(binaryfilename):
    """Return the bytes present in the given binary file name.
    """

    with open(binaryfilename, 'rb') as binaryfile:
        readbytes = binaryfile.read()

    return readbytes


def parse_aoproper(integralfilename):
    """Parse the AOPROPER file generated by DALTON and save the integral
    matrices to disk if asked.
    """

    integral_dict = dict()

    # Specify the encoding explicitly, so there's no confusion.
    encoding = 'utf-8'

    # There are a few labels we want to ignore.
    labels_to_ignore = (
        'HUCKOVLP',
        'HUCKEL',
        'HJPOPOVL',
        'EOFLABEL',
    )

    # For now, we naively read the whole file into memory, split
    # directly on the delimiter, *then* iterate. This will eventually
    # have to be changed over to a more memory-efficient version in
    # order to deal with hundreds of basis functions (if we ever want
    # that).
    integralfile_bytes = read_binary(integralfilename)
    # Remove b' \x00\x00\x00********' -> 12 bytes
    integralfile_bytes = integralfile_bytes[12:]
    integralfile_records = integralfile_bytes.split(STARS.encode(encoding=encoding))

    for record in integralfile_records:

        full_label = record[8:24].decode(encoding=encoding)
        # The first part shows the structure of the matrix.
        #  SQUARE   -> stored as the full N^2 matrix
        #  SYMMETRI -> stored as lower triangle with N*(N+1)/2 elements,
        #              ordered by rows
        #  ANTISYMM -> stored as lower triangle with N*(N+1)/2 elements,
        #              ordered by rows, (make the top triangle negative?)
        # The second part is the actual label
        # that would appear in the output file.
        shape, label = full_label[:8].strip(), full_label[8:].strip()

        # There are a few labels we want to ignore.
        if not any(label in x for x in labels_to_ignore):
            # top: b'1 A  15 ANTISYMMXDIPVEL  \x00\x00\x00\xa8\x00\x00\x00'
            # end: b'\xa8\x00\x00\x00 \x00\x00\x00' -> (168, 32)
            integrals_as_bytes = record[32:-8]
            # pylint: disable=no-member
            integrals_tril = np.fromstring(integrals_as_bytes, dtype=np.double)
            # positive solution to x = n*(n+1)/2
            nbasis = int(0.5 * (-1 + np.sqrt(1 + (8 * len(integrals_tril)))))
            # print(shape, label, len(integrals_as_bytes), nbasis, integrals_tril.shape)

            # form the full "square" matrix representation
            integrals_square = np.zeros(shape=(nbasis, nbasis))
            tril_indices = np.tril_indices(nbasis)
            integrals_square[tril_indices] = integrals_tril
            diag = np.diag(integrals_square) * np.eye(nbasis)
            if shape == 'SYMMETRI':
                integrals_square = -diag + integrals_square + integrals_square.T
            elif shape == 'ANTISYMM':
                integrals_square = -diag + integrals_square - integrals_square.T
                # If the integrals are antisymmetrized, the whole
                # thing should sum to zero.
                asum = abs(np.sum(integrals_square))
                assert asum < 1.0e-10
            else:
                print("Shouldn't be here.")

            # Was the (anti)symmetrization done correctly?
            assert integrals_square[tril_indices].all() == integrals_tril.all()

            record_dict = {
                'label': label,
                'nbasis': nbasis,
                'shape': shape,
                'integrals': integrals_square,
            }
            integral_dict[label] = record_dict

    return integral_dict


def form_rhs_geometric(natoms, MO, wfn):
    import psi4
    C = wfn.Ca()
    npC = np.asarray(C)
    norb = wfn.nmo()
    nocc = wfn.nalpha()
    o = slice(0, nocc)
    v = slice(nocc, norb)
    cart = ['_X', '_Y', '_Z']
    oei_dict = {"S" : "OVERLAP", "T" : "KINETIC", "V" : "POTENTIAL"}
    mints = psi4.core.MintsHelper(wfn)

    # Fock matrix (MO)
    T = (npC.T).dot(np.asarray(mints.ao_kinetic())).dot(npC)
    V = (npC.T).dot(np.asarray(mints.ao_potential())).dot(npC)
    H = T + V
    J = np.einsum('pqii->pq', MO[:, :, o, o])
    K = np.einsum('piqi->pq', MO[:, o, :, o])
    F = H + (2 * J) - K

    deriv1_mat = dict()
    deriv1 = dict()
    for atom in range(natoms):
        for key in oei_dict:
            deriv1_mat[key + str(atom)] = mints.mo_oei_deriv1(oei_dict[key], atom, C, C)
            for p in range(3):
                map_key = key + str(atom) + cart[p]
                deriv1[map_key] = np.asarray(deriv1_mat[key + str(atom)][p])
    for atom in range(natoms):
        string = "TEI" + str(atom)
        deriv1_mat[string] = mints.mo_tei_deriv1(atom, C, C, C, C)
        for p in range(3):
            map_key = string + cart[p]
            deriv1[map_key] = np.asarray(deriv1_mat[string][p])
    deriv2_mat = dict()
    deriv2 = dict()
    for atom1 in range(natoms):
        for atom2 in range(atom1 + 1):
            for key in oei_dict:
                string = key + str(atom1) + str(atom2)
                deriv2_mat[string] = mints.mo_oei_deriv2(oei_dict[key], atom1, atom2, C, C)
                pq = 0
                for p in range(3):
                    for q in range(3):
                        map_key = string + cart[p] + cart[q]
                        deriv2[map_key] = np.asarray(deriv2_mat[string][pq])
                        pq += 1
    for atom1 in range(natoms):
        for atom2 in range(atom1 + 1):
            string = "TEI" + str(atom1) + str(atom2)
            deriv2_mat[string] = mints.mo_tei_deriv2(atom1, atom2, C, C, C, C)
            pq = 0
            for p in range(3):
                for q in range(3):
                    map_key = string + cart[p] + cart[q]
                    deriv2[map_key] = np.asarray(deriv2_mat[string][pq])
                    pq += 1
    F_grad = dict()
    B = dict()
    for atom in range(natoms):
        for p in range(3):
            key = str(atom) + cart[p]
            contr1 = np.einsum('pqmm->pq', deriv1["TEI" + key][:, :, o, o])
            contr2 = np.einsum('pmmq->pq', deriv1["TEI" + key][:, o, o, :])
            F_grad[key] = deriv1["T" + key] + deriv1["V" + key] + (2 * contr1) - contr2
    for atom in range(natoms):
        for p in range(3):
            key = str(atom) + cart[p]
            contr1 = np.einsum("ia,ii->ia", deriv1["S" + key][o, v], F[o, o])
            contr2 = np.einsum("iamn,mn->ia", MO[o, v, o, o], deriv1["S" + key][o, o])
            contr3 = np.einsum("inma,mn->ia", MO[o, o, o, v], deriv1["S" + key][o, o])
            B[key] = contr1 - F_grad[key][o, v] + (2 * contr2) - contr3
    return B

if __name__ == '__main__':
    dalton_integrals = parse_aoproper('r_lih_hf_sto-3g/dalton_response_rpa_singlet/AOPROPER')
    from .utils import dalton_label_to_operator
    labels = dalton_integrals.keys()
    for label in labels:
        print(dalton_label_to_operator(label))
