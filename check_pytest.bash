#!/usr/bin/env bash

# check_pytest.bash: Check if we are about to use the pytest installed in the
# active virtualenv. If not, the system pytest is probably being used, and the
# modules will be wrong, giving strange test failures.

dir_pytest=$(dirname $(dirname $(command -v pytest)))

if [[ -z ${VIRTUAL_ENV} && ${VIRTUAL_ENV} != ${dir_pytest} ]]; then
    echo "You are not using a pytest that belongs to the current virtualenv!"
    echo "If you receive seemingly dumb import errors, that may be why."
    echo "-- VIRTUAL_ENV: ${VIRTUAL_ENV}"
    echo "-- pytest dir:  ${dir_pytest}"
    exit 1
fi
