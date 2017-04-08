test: pytest

pylint:
	pylint *.py

pytest:
	pytest -v test_*.py

nosetest:
	nosetests -v test_*.py
