.PHONY: test
test:
	bash check_pytest.bash
	pytest -v --doctest-modules --cov=pyresponse tests

.PHONY: pylint
pylint:
	pylint pyresponse tests

.PHONY: mypy
mypy:
	mypy pyresponse tests | perl -ne 'print if !/(No library stub file for module|Cannot find module named)/'

.PHONY: docs
docs:
	# sphinx-apidoc -o docs/source pyresponse
	cd doc && make html
