.PHONY: venv install install-optional lint lint-fix format check test export-web

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
	$(RUFF) check dataset eval tests

lint-fix:
	$(RUFF) check --fix dataset eval tests

format:
	$(RUFF) format dataset eval tests

check: lint
	$(PY) -c "import chess, numpy, pandas, requests, tqdm, zstandard; print('core imports OK')"

test:
	$(PY) -m pytest tests/ -q

export-web:
	$(PY) eval/export_web.py --db-path results/chessqa.sqlite3 --out-dir ../chess-benchmark-showcase/public/data
