.PHONY: test
test:
	python -m pytest -v --cov=pyresponse

.PHONY: precommit
precommit:
	pre-commit run -a

.PHONY: docs
docs:
	sphinx-apidoc -o docs/source pyresponse
	cd docs && make html
