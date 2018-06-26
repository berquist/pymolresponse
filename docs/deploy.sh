#!/usr/bin/env bash

# Something like this to publish on my personal site:

# git clone git@github.com:berquist/berquist.github.io.git
# make html
# mkdir -p berquist.github.io/docs
# mv buiid/html berquist.github.io/docs/pyresponse


# Follow the workflow found here: https://gist.github.com/cobyism/4730490#gistcomment-2375522

make html
mkdir -p html
rm -rf html/*
# git worktree add html gh-pages
cp -a build/html/* html
cd html
git add --all
git commit -m "Deploy to gh-pages"
git push -f origin gh-pages
cd ..
rm -rf html/*

# Other useful information:
# * https://blog.github.com/2009-12-29-bypassing-jekyll-on-github-pages/
# * https://stackoverflow.com/questions/35227274/how-to-port-python-sphinx-doc-to-github-pages
