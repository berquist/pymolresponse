.PHONY: test
test:
	bash check_pytest.bash
	pytest -v --doctest-modules --isort --cov=pyresponse pyresponse

.PHONY: isort
isort:
	isort -rc .

.PHONY: pylint
pylint:
	pylint pyresponse

.PHONY: mypy
mypy:
	mypy pyresponse | perl -ne 'print if !/(No library stub file for module|Cannot find module named)/'

.PHONY: docs
docs:
	# sphinx-apidoc -o docs/source pyresponse
	cd doc && make html
