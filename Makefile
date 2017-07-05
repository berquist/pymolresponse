test: pytest

pylint:
	pylint pyresponse/*.py tests/*.py

pytest:
	pytest -v tests

nosetest:
	nosetests -v tests
