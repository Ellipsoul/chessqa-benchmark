#!/bin/bash
# =============================================================================
# ChessQA Setup Script — creates a local venv and installs dependencies.
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${ROOT}/.venv"
PYTHON="${PYTHON:-python3}"

echo "ChessQA — Setup"
echo "==============="
echo

# Python
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "Python 3.10+ is required. Install Python and re-run, or set PYTHON=/path/to/python3."
  exit 1
fi
echo "Python: $("${PYTHON}" --version)"

# Virtual environment
if [ ! -d "${VENV}" ]; then
  echo "Creating virtual environment at .venv ..."
  "${PYTHON}" -m venv "${VENV}"
else
  echo "Using existing virtual environment at .venv"
fi

PIP="${VENV}/bin/pip"
PY="${VENV}/bin/python"

echo "Upgrading pip ..."
"${PIP}" install --upgrade pip

echo "Installing core dependencies ..."
"${PIP}" install -r "${ROOT}/requirements.txt"

echo "Installing dev dependencies (ruff) ..."
"${PIP}" install -r "${ROOT}/requirements-dev.txt"

# API keys (OpenRouter)
echo
echo "API key setup (OpenRouter)"
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  if [ -f "${ROOT}/../keys/api_keys.json" ]; then
    echo "Found ../keys/api_keys.json (used by eval/run_openrouter.py)"
  elif [ -f "${ROOT}/keys/openrouter.key" ]; then
    export OPENROUTER_API_KEY="$(cat "${ROOT}/keys/openrouter.key")"
    echo "Loaded OPENROUTER_API_KEY from keys/openrouter.key"
  else
    echo "OPENROUTER_API_KEY not set. For cloud inference:"
    echo "  export OPENROUTER_API_KEY=\"your_key\""
    echo "  or save to ../keys/api_keys.json (see CLAUDE.md)"
    echo "  Get a key: https://openrouter.ai/"
  fi
else
  echo "Detected OPENROUTER_API_KEY in environment"
fi

# Data files
echo
echo "Checking for source data (data/raw/) ..."
need_help=0
for f in \
  "data/raw/lichess_db_puzzle.csv" \
  "data/raw/lichess_db_eval.jsonl.zst"
do
  if [ -f "${ROOT}/${f}" ]; then
    echo "  ok  ${f}"
  else
    echo "  missing  ${f}"
    need_help=1
  fi
done
if [ -f "${ROOT}/data/raw/lichess_db_broadcast_2025-04.pgn" ]; then
  echo "  ok  data/raw/lichess_db_broadcast_2025-04.pgn (optional)"
else
  echo "  optional  data/raw/lichess_db_broadcast_2025-04.pgn (state tracking)"
fi
if [ "$need_help" -eq 1 ]; then
  echo "  Download from https://database.lichess.org/ and place under data/raw/"
fi

# Optional extras
echo
echo "Optional extras:"
echo "  Semantic MCQ (05_semantic.py):  .venv/bin/pip install -r requirements-optional.txt"
echo "  Offline vLLM pipeline (05_2/05_3):  .venv/bin/pip install torch vllm"

# Verify core imports
echo
echo "Verifying core imports ..."
"${PY}" - <<'PY'
import chess, numpy, pandas, requests, tqdm, zstandard
print("  core imports OK")
PY

echo
echo "Next steps"
echo "----------"
echo "  source .venv/bin/activate"
echo "  make lint          # run ruff"
echo
echo "  # Run inference against checked-in benchmark:"
echo "  python eval/run_openrouter.py --dataset-root benchmark --model anthropic/claude-sonnet-4.5 \\"
echo "    --output-dir results --workers 256 --max-tasks 5"
echo
echo "Setup complete."
