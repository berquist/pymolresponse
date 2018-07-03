#!/usr/bin/env bash

# deploy_pyresponse_docs_repo.sh: Run this to build the HTML
# documentation using Sphinx, then commit and push it to
# berquist/pyresponse_docs, which serves from the master branch.

set -o errexit

GH_USER=berquist
GH_REPO_NAME=pyresponse_docs
DOCS_BRANCH_NAME=master
GH_REPO_REF=github.com:$GH_USER/$GH_REPO_NAME.git

make html

if [ ! -d $GH_REPO_NAME ]; then
    echo "Cloning $DOCS_BRANCH_NAME branch of $GH_REPO_REF..."
    git clone -b $DOCS_BRANCH_NAME git@$GH_REPO_REF
fi
cd $GH_REPO_NAME
rm -rf ./*
echo "" > .nojekyll
echo "Copying built HTML..."
cp -a ../build/html/* .

echo "Adding changes..."
git add --all
echo "Committing..."
# This will return 1 if there are no changes, which should not result
# in failure.
git commit -m "Deploy documentation `date`" || ret=$?
git push --force origin $DOCS_BRANCH_NAME

cd ..
