test:
    python -m pytest

test-pixi:
    pixi run -e test test

pre:
    pre-commit run -a

docs:
    sphinx-apidoc -o docs/source pymolresponse
    cd docs && make html

# make-conda-env:
#     echo "hi"
