"""Core types."""

from enum import Enum, unique


@unique
class AO2MOTransformationType(Enum):
    partial = "partial"
    full = "full"


@unique
class Hamiltonian(Enum):
    """
    - RPA indicates the random phase approximation
    - TDA indicates the Tamm-Dancoff approximation
    """

    RPA = "rpa"
    TDA = "tda"


@unique
class Spin(Enum):
    singlet = 0
    triplet = 1


@unique
class Program(Enum):
    PySCF = "pyscf"
    Psi4 = "psi4"
    DALTON = "dalton"
