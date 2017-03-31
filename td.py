import numpy as np
import scipy as sp

from cphf import CPHF
from utils import form_results

from explicit_equations_full import \
    (form_rpa_a_matrix_mo_singlet_full,
     form_rpa_a_matrix_mo_singlet_ss_full,
     form_rpa_a_matrix_mo_singlet_os_full,
     form_rpa_a_matrix_mo_triplet_full,
     form_rpa_b_matrix_mo_singlet_full,
     form_rpa_b_matrix_mo_singlet_ss_full,
     form_rpa_b_matrix_mo_singlet_os_full,
     form_rpa_b_matrix_mo_triplet_full)
from explicit_equations_partial import \
    (form_rpa_a_matrix_mo_singlet_partial,
     form_rpa_a_matrix_mo_singlet_ss_partial,
     form_rpa_a_matrix_mo_singlet_os_partial,
     form_rpa_a_matrix_mo_triplet_partial,
     form_rpa_b_matrix_mo_singlet_partial,
     form_rpa_b_matrix_mo_singlet_ss_partial,
     form_rpa_b_matrix_mo_singlet_os_partial,
     form_rpa_b_matrix_mo_triplet_partial)


class TDHF(CPHF):

    def __init__(self, mocoeffs, moenergies, occupations, *args, **kwargs):
        # don't call superclass __init__
        assert len(mocoeffs.shape) == 3
        assert (mocoeffs.shape[0] == 1) or (mocoeffs.shape[0] == 2)
        self.is_uhf = (mocoeffs.shape[0] == 2)
        assert len(moenergies.shape) == 3
        assert (moenergies.shape[0] == 1) or (moenergies.shape[0] == 2)
        if self.is_uhf:
            assert moenergies.shape[0] == 2
        else:
            assert moenergies.shape[0] == 1
        assert moenergies.shape[1] == moenergies.shape[2]
        assert len(occupations) == 4

        self.mocoeffs = mocoeffs
        self.moenergies = moenergies
        self.occupations = occupations
        self.form_ranges_from_occupations()

        self.solver = 'explicit'
        self.hamiltonian = 'rpa'
        self.spin = 'singlet'

        self.operators = []
        self.frequencies = []
        self.results = []

    def set_frequencies(self, frequencies=None):
        pass

    # def add_operator(self, operator):
    #     pass

    def run(self, solver=None, hamiltonian=None, spin=None, **kwargs):

        if not solver:
            solver = self.solver
        if not hamiltonian:
            hamiltonian = self.hamiltonian
        if not spin:
            spin = self.spin

        # Set the current state.
        self.solver = solver
        self.hamiltonian = hamiltonian
        self.spin = spin

        if solver == 'explicit':
            self.form_explicit_hessian(hamiltonian, spin, None)
            # self.invert_explicit_hessian()
            self.diagonalize_explicit_hessian()
        # Nothing else implemented yet.
        else:
            pass

        self.form_results()

    def form_explicit_hessian(self, hamiltonian=None, spin=None, frequency=None):

        assert hasattr(self, 'tei_mo')
        assert len(self.tei_mo) in (1, 2, 4, 6)

        if not hamiltonian:
            hamiltonian = self.hamiltonian
        if not spin:
            spin = self.spin

        assert hamiltonian in ('rpa', 'tda')
        assert spin in ('singlet', 'triplet')

        self.hamiltonian = hamiltonian
        self.spin = spin

        nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = self.occupations
        nov_alph = nocc_alph * nvirt_alph
        nov_beta = nocc_beta * nvirt_beta

        assert self.tei_mo_type in ('full', 'partial')

        if not self.is_uhf:

            # Set up "function pointers".
            if self.tei_mo_type == 'full':
                assert len(self.tei_mo) == 1
                tei_mo = self.tei_mo[0]
            elif self.tei_mo_type == 'partial':
                assert len(self.tei_mo) == 2
                tei_mo_ovov = self.tei_mo[0]
                tei_mo_oovv = self.tei_mo[1]

            if self.tei_mo_type == 'full':
                if hamiltonian == 'rpa' and spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_full(self.moenergies[0, ...], tei_mo, nocc_alph)
                    B = form_rpa_b_matrix_mo_singlet_full(tei_mo, nocc_alph)
                elif hamiltonian == 'rpa' and spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_full(self.moenergies[0, ...], tei_mo, nocc_alph)
                    B = form_rpa_b_matrix_mo_triplet_full(tei_mo, nocc_alph)
                elif hamiltonian == 'tda' and spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_full(self.moenergies[0, ...], tei_mo, nocc_alph)
                    B = np.zeros(shape=(nov_alph, nov_alph))
                elif hamiltonian == 'tda' and spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_full(self.moenergies[0, ...], tei_mo, nocc_alph)
                    B = np.zeros(shape=(nov_alph, nov_alph))
            elif self.tei_mo_type == 'partial':
                if hamiltonian == 'rpa' and spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_partial(self.moenergies[0, ...], tei_mo_ovov, tei_mo_oovv)
                    B = form_rpa_b_matrix_mo_singlet_partial(tei_mo_ovov)
                elif hamiltonian == 'rpa' and spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_partial(self.moenergies[0, ...], tei_mo_oovv)
                    B = form_rpa_b_matrix_mo_triplet_partial(tei_mo_ovov)
                elif hamiltonian == 'tda' and spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_partial(self.moenergies[0, ...], tei_mo_ovov, tei_mo_oovv)
                    B = np.zeros(shape=(nov_alph, nov_alph))
                elif hamiltonian == 'tda' and spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_partial(self.moenergies[0, ...], tei_mo_oovv)
                    B = np.zeros(shape=(nov_alph, nov_alph))

            G = np.asarray(np.bmat([[ A,  B],
                                    [-B, -A]]))
            self.explicit_hessian = G

        else:
            # TODO UHF
            pass

    def invert_explicit_hessian(self):
        pass

    def diagonalize_explicit_hessian(self):
        if not self.is_uhf:
            eigvals, eigvecs = sp.linalg.eig(self.explicit_hessian)
            idx = eigvals.argsort()
            self.eigvals = eigvals[idx]
            # Each eigenvector is a column vector.
            self.eigvecs = eigvecs[:, idx]
            # for x in self.eigvals.real * 27.2114:
            #     print(x)
        else:
            # TODO UHF
            pass

    def form_results(self):
        # def norm_xy(z):
        #     x, y = z.reshape(2,nvir,nocc)
        #     norm = 2 * (np.linalg.norm(x)**2 - np.linalg.norm(y)**2)
        #     norm = 1 / numpy.sqrt(norm)
        #     return x*norm, y*norm
        if len(self.operators) > 0:
            nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = self.occupations
            nov_alph = nocc_alph * nvirt_alph
            nov_beta = nocc_beta * nvirt_beta
            # contract the components of every operator with every
            # eigenvector
            dim = self.eigvecs.shape[1]
            for operator in self.operators:
                for idx in range(dim):
                    norm = 2 * (np.linalg.norm(self.eigvecs[:nov_alph, idx])**2 - \
                                np.linalg.norm(self.eigvecs[nov_alph:, idx])**2)
                    norm = (1 / np.sqrt(norm))
                    print('-' * 78)
                    eigvec = self.eigvecs[:, idx]
                    eigvec_normed = self.eigvecs[:, idx] * norm
                    # print(eigvec)
                    # print(eigvec_normed)
                    integrals = operator.mo_integrals_ai_supervector_alph[:, :, 0]
                    res = np.dot(integrals, eigvec)
                    # TODO why the -2?
                    res_normed = -2 * np.dot(integrals, eigvec_normed)
                    eigval = self.eigvals[idx].real
                    print(' State: {}'.format(idx + 1))
                    print(' Excitation energy [a.u.]: {}'.format(eigval))
                    print(' Excitation energy [eV]  : {}'.format(eigval * 27.2114))
                    print(' Transition moment: {}'.format(res_normed))
                    print(' Oscillator strength: {}'.format((2 / 3) * eigval * res_normed ** 2))
                    print(' Oscillator strength (total): {}'.format((2 / 3) * eigval * np.dot(res_normed, res_normed)))


class TDA(TDHF):

    def __init__(self, mocoeffs, moenergies, occupations, *args, **kwargs):
        super(TDA, self).__init__(mocoeffs, moenergies, occupations, *args, **kwargs)

    def form_explicit_hessian(self, hamiltonian=None, spin=None, frequency=None):
        assert hasattr(self, 'tei_mo')
        assert len(self.tei_mo) in (1, 2, 4, 6)

        if not hamiltonian:
            hamiltonian = self.hamiltonian
        if not spin:
            spin = self.spin

        assert hamiltonian == 'tda'
        assert spin in ('singlet', 'triplet')

        self.hamiltonian = hamiltonian
        self.spin = spin

        nocc_alph, nvirt_alph, nocc_beta, nvirt_beta = self.occupations
        nov_alph = nocc_alph * nvirt_alph
        nov_beta = nocc_beta * nvirt_beta

        assert self.tei_mo_type in ('full', 'partial')

        if not self.is_uhf:

            # Set up "function pointers".
            if self.tei_mo_type == 'full':
                assert len(self.tei_mo) == 1
                tei_mo = self.tei_mo[0]
            elif self.tei_mo_type == 'partial':
                assert len(self.tei_mo) == 2
                tei_mo_ovov = self.tei_mo[0]
                tei_mo_oovv = self.tei_mo[1]

            if self.tei_mo_type == 'full':
                if spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_full(self.moenergies[0, ...], tei_mo, nocc_alph)
                elif spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_full(self.moenergies[0, ...], tei_mo, nocc_alph)
            elif self.tei_mo_type == 'partial':
                if spin == 'singlet':
                    A = form_rpa_a_matrix_mo_singlet_partial(self.moenergies[0, ...], tei_mo_ovov, tei_mo_oovv)
                elif spin == 'triplet':
                    A = form_rpa_a_matrix_mo_triplet_partial(self.moenergies[0, ...], tei_mo_oovv)

            self.explicit_hessian = A

        else:
            # TODO UHF
            pass

    def form_results(self):
        if len(self.operators) > 0:
            # Contract the components of every operator with every
            # eigenvector
            dim = self.eigvecs.shape[1]
            for operator in self.operators:
                for idx in range(dim):
                    print('-' * 78)
                    norm = (1 / np.sqrt(2))
                    eigvec = self.eigvecs[:, idx]
                    eigvec_normed = self.eigvecs[:, idx] * norm
                    # print(eigvec)
                    # print(eigvec_normed)
                    integrals = operator.mo_integrals_ai_alph[:, :, 0]
                    res = np.dot(integrals, eigvec)
                    # TODO why the -2?
                    res_normed = -2 * np.dot(integrals, eigvec_normed)
                    eigval = self.eigvals[idx].real
                    print(' State: {}'.format(idx + 1))
                    print(' Excitation energy [a.u.]: {}'.format(eigval))
                    print(' Excitation energy [eV]  : {}'.format(eigval * 27.2114))
                    print(' Transition moment: {}'.format(res_normed))
                    print(' Oscillator strength: {}'.format((2 / 3) * eigval * res_normed ** 2))
                    print(' Oscillator strength (total): {}'.format((2 / 3) * eigval * np.dot(res_normed, res_normed)))

if __name__ == '__main__':
    import pyscf

    mol = pyscf.gto.Mole()
    mol.verbose = 1
    mol.output = None

    mol.atom = [
        ['H' , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ]
    mol.basis = 'sto-3g'
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.scf()

    import utils
    C = utils.fix_mocoeffs_shape(mf.mo_coeff)
    E = np.diag(mf.mo_energy)[np.newaxis, ...]
    # print(E.shape)
    # import sys
    # sys.exit()
    occupations = utils.occupations_from_pyscf_mol(mol, C)
    tda = TDA(C, E, occupations)
    tdhf = TDHF(C, E, occupations)
    import ao2mo
    tei_mo = ao2mo.perform_tei_ao2mo_rhf_partial(mol, C, mol.verbose)
    tda.tei_mo = tei_mo
    tda.tei_mo_type = 'partial'
    tdhf.tei_mo = tei_mo
    tdhf.tei_mo_type = 'partial'

    from cphf import Operator
    operator_dipole = Operator(label='dipole', is_imaginary=False, is_spin_dependent=False, triplet=False)
    integrals_dipole_ao = mol.intor('cint1e_r_sph', comp=3)
    operator_dipole.ao_integrals = integrals_dipole_ao

    tda.add_operator(operator_dipole)
    tdhf.add_operator(operator_dipole)

    print('=' * 78)
    print(' dipole')
    print('-' * 78)
    print('TDA using TDA()')
    tda.run(solver='explicit', hamiltonian='tda', spin='singlet')
    print('TDA using TDHF()')
    tdhf.run(solver='explicit', hamiltonian='tda', spin='singlet')
    print('RPA using TDHF()')
    tdhf.run(solver='explicit', hamiltonian='rpa', spin='singlet')
    # print('=' * 78)
    # print(' angular momentum')
    # print('-' * 78)
    # print('RPA using TDHF()')
    # tdhf.operators = []
    # operator_angmom = Operator(label='angmom', is_imaginary=True, is_spin_dependent=False, triplet=False)
    # integrals_angmom_ao = mol.intor('cint1e_cg_irxp_sph', comp=3)
    # operator_angmom.ao_integrals = integrals_angmom_ao
    # tdhf.add_operator(operator_angmom)
    # tdhf.run(solver='explicit', hamiltonian='rpa', spin='singlet')
