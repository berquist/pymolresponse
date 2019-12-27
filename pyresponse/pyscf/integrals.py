from pyresponse.integrals import JK, Integrals


class IntegralsPyscf(Integrals):
    def __init__(self, pyscfmol, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mol = pyscfmol

    ANGMOM_COMMON_GAUGE = "cint1e_cg_irxp_sph"
    DIPOLE = "cint1e_r_sph"
    DIPVEL = "cint1e_ipovlp_sph"

    def _compute(self, pyscf_integral_label):
        return self.mol.intor(pyscf_integral_label, comp=3)


class JKPyscf(JK):
    def __init__(self, pyscfmol, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mol = pyscfmol

    def compute_from_density(self, D):
        raise NotImplementedError

    def compute_from_mocoeffs(self, C_left, C_right=None):
        raise NotImplementedError
