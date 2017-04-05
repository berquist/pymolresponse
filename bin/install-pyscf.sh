#!/usr/bin/env bash
set -e
pip install scipy h5py
if [ ! -d "$HOME/pyscf" ]; then
    git clone https://github.com/sunqm/pyscf.git "$HOME"/pyscf
else
    cd "$HOME"/pyscf
    git pull
fi
mkdir -p "$HOME"/pyscf/lib/build
cd "$HOME"/pyscf/lib/build || exit
cmake ..
make
cd ../.. || exit
python setup.py install
