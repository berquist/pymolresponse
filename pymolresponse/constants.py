r"""Storage of common physical constants. Uses `scipy.constants`: https://docs.scipy.org/doc/scipy/reference/constants.html."""

import pint
import scipy.constants as spc


ureg = pint.UnitRegistry(system="atomic")

## Fundamental

alpha = spc.alpha

## (Excitation) energies
HARTREE_TO_EV = spc.physical_constants["Hartree energy in eV"][0]
HARTREE_TO_INVCM = spc.physical_constants["hartree-inverse meter relationship"][0] * (1 / 100)

## Dipole

# Convert a dipole moment from atomic units (a.u.) to Debye.
convfac_au_to_debye = 2.541746230211
base_units_dipole = ureg.get_base_units(input_units=ureg.debye)[1]

## Electronic circular dichroism (ECD)

# ESUECD =  ECHARGE*XTANG*CCM*1D36*ECHARGE*HBAR/EMASS
echarge = spc.elementary_charge
xtang = spc.physical_constants["atomic unit of length"][0] * 1.0e10
ccm = spc.c
hbar = spc.hbar
emass = spc.electron_mass
esuecd = echarge * xtang * ccm * 1.0e36 * echarge * hbar / emass
# print(esuecd)
# # DALTON 1998
# echarge = 1.602176462e-19
# xtang = 0.5291772083e0
# ccm = 299792458.0e0
# hbar = 1.054571596e-34
# emass = 9.10938188e-31
# esuecd = echarge * xtang * ccm * 1.0e36 * echarge * hbar / emass
# print(esuecd)
# # DALTON 2002
# echarge = 1.60217653e-19
# xtang = 0.5291772108e0
# ccm = 299792458.0e0
# hbar = 1.05457168e-34
# emass = 9.1093826e-31
# esuecd = echarge * xtang * ccm * 1.0e36 * echarge * hbar / emass
# print(esuecd)

del pint
