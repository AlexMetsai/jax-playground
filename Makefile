PYTHON=python
PIP=pip
PYTHON_PROJECT_ROOT=.
VERSION=0.0.0

.PHONY: lint
lint:
	${PYTHON} -m flake8 .

.PHONY: install
install:
	PYTHONPATH=${PYTHON_PROJECT_ROOT} ${PIP} install -r requirements.txt
