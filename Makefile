test: pytest

pylint:
	pylint *.py tests/*.py

pytest:
	pytest -v tests

nosetest:
	nosetests -v tests
