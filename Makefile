.PHONY: venv install install-optional lint lint-fix format check

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
RUFF := $(VENV)/bin/ruff

venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -r requirements-dev.txt

install-optional:
	$(PIP) install -r requirements-optional.txt

lint:
	$(RUFF) check dataset eval

lint-fix:
	$(RUFF) check --fix dataset eval

format:
	$(RUFF) format dataset eval

check: lint
	$(PY) -c "import chess, numpy, pandas, requests, tqdm, zstandard; print('core imports OK')"
