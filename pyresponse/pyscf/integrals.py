from enum import Enum, auto, unique
from typing import Optional

from attr import attrib, attrs
from attr.validators import instance_of, optional

from pyresponse.integrals import JK, Integrals


@unique
class IntegralSymmetry(Enum):
    SYMMETRIC = auto()
    ANTISYMMETRIC = auto()


@attrs(frozen=True, slots=True)
class LabelPyscf:
    label: str = attrib(validator=instance_of(str))
    comp: Optional[int] = attrib(validator=optional(instance_of(int)), default=None)
    symmetry: IntegralSymmetry = attrib(
        validator=instance_of(IntegralSymmetry), default=IntegralSymmetry.SYMMETRIC
    )


ANGMOM_COMMON_GAUGE = LabelPyscf("cint1e_cg_irxp_sph", 3)
ANGMOM_GIAO = LabelPyscf("cint1e_giao_irjxp_sph", 3)
DIPOLE = LabelPyscf("cint1e_r_sph", 3)
DIPVEL = LabelPyscf("cint1e_ipovlp_sph", 3)
KINETIC = LabelPyscf("int1e_kin")
A11PART_GIAO = LabelPyscf("int1e_giao_a11part", 9)
A11PART_CG = LabelPyscf("int1e_cg_a11part", 9)
A01 = LabelPyscf("int1e_a01gp", 9)
NUCLEAR = LabelPyscf("int1e_nuc")
SO_1e = LabelPyscf("int1e_prinvxp", 3, IntegralSymmetry.ANTISYMMETRIC)
NSO_1e = LabelPyscf("int1e_pnucxp", 3, IntegralSymmetry.ANTISYMMETRIC)
SO_SPHER_1e = LabelPyscf("cint1e_prinvxp_sph", 3)


class IntegralsPyscf(Integrals):
    def __init__(self, pyscfmol, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mol = pyscfmol

    def _compute(self, label: LabelPyscf):
        if label.symmetry == IntegralSymmetry.ANTISYMMETRIC:
            return self.mol.intor_asymmetric(label.label, comp=label.comp)
        return self.mol.intor(label.label, comp=label.comp)


class JKPyscf(JK):
    def __init__(self, pyscfmol, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mol = pyscfmol

    def compute_from_density(self, D):
        raise NotImplementedError

    def compute_from_mocoeffs(self, C_left, C_right=None):
        raise NotImplementedError
