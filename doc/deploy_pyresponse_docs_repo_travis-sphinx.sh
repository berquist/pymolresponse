#!/usr/bin/env bash

MESSAGE="Deploy code docs to GitHub Pages Travis build: ${TRAVIS_BUILD_NUMBER}\n\nCommit: ${TRAVIS_COMMIT}"
travis-sphinx --verbose deploy --message="${MESSAGE}"
