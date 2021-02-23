.PHONY: test
test:
	bash check_pytest.bash
	python -m pytest -v --doctest-modules --cov=pyresponse pyresponse

.PHONY: precommit
precommit:
	isort -rc .
	black .

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
