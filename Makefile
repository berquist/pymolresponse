test: pytest-cov

pylint:
	pylint pyresponse/*.py tests/*.py

pytest:
	pytest -v --doctest-modules tests

nosetest:
	nosetests -v tests

pytest-cov:
	pytest -v --doctest-modules --cov=pyresponse tests

nosetest-cov:
	nosetests -v --with-coverage tests

.PHONY: docs

docs:
	sphinx-apidoc -o docs/source pyresponse
	cd docs && make html
