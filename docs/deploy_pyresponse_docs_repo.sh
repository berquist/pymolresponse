#!/usr/bin/env bash

# deploy_pyresponse_docs_repo.sh: Run this to build the HTML
# documentation using Sphinx, then commit and push it to
# berquist/pyresponse_docs, which serves from the master branch.

set -o errexit

# git submodule add -b master git@github.com:berquist/pyresponse_docs pyresponse_docs
# git submodule add -b master https://github.com/berquist/pyresponse_docs.git pyresponse_docs
git clone git@github.com:berquist/pyresponse_docs.git
make html
cp -a build/html/* pyresponse_docs
cd pyresponse_docs
git add --all
git commit -m "Deploy documentation `date`"
git push origin master
cd ..
