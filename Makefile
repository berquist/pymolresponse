test: pytest

pylint:
	pylint *.py

pytest:
	OMP_NUM_THREADS=1 pytest -v
