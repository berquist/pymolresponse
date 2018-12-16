.PHONY: test
test:
	bash check_pytest.bash
	pytest -v --doctest-modules --cov=pyresponse tests

.PHONY: pylint
pylint:
	pylint pyresponse/*.py tests/*.py

.PHONY: docs
docs:
	# sphinx-apidoc -o docs/source pyresponse
	cd doc && make html
