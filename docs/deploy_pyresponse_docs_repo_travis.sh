#!/usr/bin/env bash

# Adapted from
# https://gist.github.com/vidavidorra/548ffbcdae99d752da02

set -o errexit

git config user.name "Travis CI"
git config user.email "travis@travis-ci.org"

make html

git clone -b $DOCS_BRANCH_NAME https://git@$GH_REPO_REF
cd $GH_REPO_NAME
rm -rf ./*
echo "" > .nojekyll
echo "Copying built HTML..."
cp -a ../build/html/* .

if [ -f "index.html" ]; then
    echo "Adding changes..."
    git add --all
    echo "Committing..."
    # This will return 1 if there are no changes, which should not
    # result in failure.
    git commit -m "Deploy code docs to GitHub Pages Travis build: ${TRAVIS_BUILD_NUMBER}" -m "Commit: ${TRAVIS_COMMIT}" || ret=$?
    git push --force "https://${GH_REPO_TOKEN}@${GH_REPO_REF}" > /dev/null 2>&1
else
    echo '' >&2
    echo 'Warning: No documentation (html) files have been found!' >&2
    echo 'Warning: Not going to push the documentation to GitHub!' >&2
    exit 1
fi

cd ..
