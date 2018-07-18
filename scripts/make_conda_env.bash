#!/usr/bin/env bash

## make_conda_env.bash:

set -o errexit
set -v

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(realpath "${SCRIPTDIR}"/..)"

## before_install
if [ ! -d "${PROJECT_ROOT}/miniconda/bin" ]; then
    echo "Downloading miniconda..."
    rm -rf "${PROJECT_ROOT}"/miniconda
    curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
    bash miniconda.sh -b -p "${PROJECT_ROOT}"/miniconda
fi
PATH="${PROJECT_ROOT}/miniconda/bin:${PATH}"
hash -r
# TODO this currently modifies $HOME/.condarc!!!
conda config --set always_yes yes
conda config --set changeps1 no
conda update conda
conda info -a
# If this fails, it's probably because the environment already exists.
conda create -q -n p4env python=3.6 numpy scipy h5py psi4 psi4-rt -c psi4/label/dev -c psi4 || retval=$?
source activate p4env
conda install -c pyscf pyscf
conda install -c conda-forge pytest-cov codecov sphinx sphinx-automodapi sphinx_rtd_theme numpydoc graphviz python-graphviz
pip install travis-sphinx
conda list

## install
pip install "${PROJECT_ROOT}"

## before_script
mkdir -p "${PROJECT_ROOT}"/PSI_SCRATCH
PSI_SCRATCH="${PROJECT_ROOT}"/PSI_SCRATCH
# PYTHONPATH="$(pwd):${PYTHONPATH}"
GH_REPO_NAME=pyresponse_docs
GH_REPO_SLUG=berquist/${GH_REPO_NAME}
GH_REPO_REF=github.com/${GH_REPO_SLUG}.git
DOCS_BRANCH_NAME=gh-pages
DOCS_DIR="${PROJECT_ROOT}"/doc

## script
make test
travis-sphinx --verbose --outdir="${DOCS_DIR}"/build build --source="${DOCS_DIR}"/source --nowarn
install -dm755 "${DOCS_DIR}"/build/html
touch "${DOCS_DIR}"/build/html/.nojekyll

## after_success
# codecov
