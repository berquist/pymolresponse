#!/usr/bin/env bash
set -e
pip install scipy h5py
remote_repo="https://github.com/sunqm/pyscf.git"
pyscf_dir="$HOME/pyscf"
# if [ ! -d "$pyscf_dir" ]; then
#     git clone $remote_repo "$HOME"/pyscf
# elif [ ! -d "$pyscf_dir/.git" ]; then
#     cd "$pyscf_dir"
#     git init
#     git remote add origin $remote_repo
#     git pull
# else
#     cd "$pyscf_dir"
#     git pull
# fi
git clone $remote_repo "$HOME"/pyscf
mkdir -p "$pyscf_dir"/pyscf/lib/build
cd "$pyscf_dir"/pyscf/lib/build || exit
cmake \
    -DENABLE_XCFUN=0 \
    ..
make
cd ../../.. || exit
python setup.py install
