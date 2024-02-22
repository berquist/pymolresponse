.PHONY: test
test:
	python -m pytest -v --cov=pymolresponse

.PHONY: precommit
precommit:
	pre-commit run -a

.PHONY: docs
docs:
	sphinx-apidoc -o docs/source pymolresponse
	cd docs && make html
