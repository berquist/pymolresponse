.PHONY: test
test:
	python -m pytest -v --cov=pyresponse

.PHONY: precommit
precommit:
	python -m isort .
	python -m black .

.PHONY: pylint
pylint:
	python -m pylint pyresponse

.PHONY: mypy
mypy:
	python -m mypy pyresponse

.PHONY: docs
docs:
	sphinx-apidoc -o docs/source pyresponse
	cd doc && make html
