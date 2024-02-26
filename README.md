# pymolresponse

Molecular frequency-dependent response properties for arbitrary operators.

[![build status](https://github.com/berquist/pymolresponse/actions/workflows/test.yml/badge.svg)](https://github.com/berquist/pymolresponse/blob/main/.github/workflows/test.yml)
[![codecov](https://codecov.io/gh/berquist/pymolresponse/branch/main/graph/badge.svg)](https://codecov.io/gh/berquist/pymolresponse)
[![license](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg?style=flat)](https://github.com/berquist/pymolresponse/blob/main/LICENSE)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/berquist/pymolresponse/main.svg)](https://results.pre-commit.ci/latest/github/berquist/pymolresponse/main)

For documentation, go to https://berquist.github.io/pymolresponse/.

Currently, the goal is to provide:

1. a pedagogical example of a working molecular orbital response program as an almost direct translation from equations to code,
2. an implementation that gives "exact" results for testing, and
3. an example of testing and documenting scientific code using modern software development tools.

## Installation

To set up a conda environment with all dependencies for running, testing, and building the documentation, look under `devtools`.

## Requirements

* Python >= ~~3.2 because of pyscf~~3.6 because of [f-strings](https://cito.github.io/blog/f-strings/).
* [pyscf](https://github.com/sunqm/pyscf) and its dependencies: CMake, NumPy, SciPy, HDF5 + h5py
* [Psi4](https://psicode.org/)

### Other Python dependencies

* [periodictable](https://github.com/pkienzle/periodictable) (for calculating the center of mass)
* [pytest](http://doc.pytest.org/en/latest/) (for testing)
* [daltools](https://github.com/vahtras/daltools) (for testing?)
* [cclib](https://github.com/cclib/cclib) (for testing)

## Testing

```bash
make pytest
```

## Caveats

* RHF and UHF references only; no ROHF yet.
* Hartree-Fock and DFT only; no post-HF methods yet.
* Real orbitals only; no complex or generalized orbitals yet.
* Because the dimensioning of all arrays is based around the ov/vo space, methods that have non-zero contributions from the oo space (specifically derivatives of GIAOs/London orbitals w.r.t. the B-field) are not currently possible.
* An iterative solver exists for response properties, not transition properties, where only explicit formation and then diagonalization of the orbital Hessian is available.
* Linear response and single residues only.
* _unrestricted diagonalization-based properties are not implemented/working_

## Desired features (in no specific order)

* Non-orthogonal orbitals. Requires switch from using MO energies to full Fock matrices.
* ROHF reference (compare against DALTON). Requires equations for the ROHF orbital Hessian.
* Post-HF support: MP2, CCSD, and CIS. Requires constriction of a Lagrangian.
* Support for GIAOs. Only requires re-dimensioning as long as AO matrix elements are available??
* At least one iterative method for each property type, for example DIIS for inversion and Davidson for diagonalization. Requires matrix-vector products.
* Quadratic response and associated single residues (needed for phosphorescence) and double residues (excited state expectation values and transition moments of linear operators). Requires permutation of linear response solution vectors.

### Desired features that don't fix the caveats

* Open-ended response: see [Ringholm, Jonsson, and Ruud](https://doi.org/10.1002/jcc.23533).
* Finite-difference for testing and higher-order response.
* Interface to [PyQuante](https://github.com/berquist/pyquante) and/or [pyquante2](https://github.com/rpmuller/pyquante2).
* Jupyter Notebook-based tutorials.
* Argument type-checking using [mypy](http://mypy-lang.org/).

## References

Forthcoming...
