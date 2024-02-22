"""Core types."""

from enum import Enum, unique


@unique
class AO2MOTransformationType(Enum):
    r"""For routines that perform AO-to-MO transformations, specify the kind of
    transformation.

    Starting from :math:`(\mu\nu|\lambda\sigma)`,

    - a 'partial' transformation will produce :math:`(ia|jb)` and :math:`(ij|ab)` if needed
    - a 'full' transformation will produce :math:`(pq|rs)`
    """

    partial = "partial"
    full = "full"


@unique
class Hamiltonian(Enum):
    """Specify which approximation for the orbital Hessian should be used.

    - RPA indicates the random phase approximation: the orbital Hessian is
      exact for the first-order polarization propagator.
    - TDA indicates the Tamm-Dancoff approximation: the B matrix is set to zero.

    Currently this is only capable of handling the usual first-order
    polarization propagator for CPHF and TDHF.
    """

    RPA = "rpa"
    TDA = "tda"


@unique
class Spin(Enum):
    """Specify whether the calculation should conserve spin (singlet) or flip spin
    (triplet).

    For response properties, this affects both the orbital Hessian and the
    perturbing operator. Note that these two specifications are currently
    independent!

    For excitation energies, this only affects the orbital Hessian and
    determines whether singlet or triplet excitation energies are calculated.
    """

    singlet = 0
    triplet = 1


@unique
class Program(Enum):
    """Specify which program should be used for a particular step."""

    PySCF = "pyscf"
    Psi4 = "psi4"
    DALTON = "dalton"
