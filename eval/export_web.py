"""Export the canonical smoke runs from the SQLite results DB to static JSON for the web explorer.

Produces <out-dir>/{index.json, categories/<slug>.json, traces/<task_id>.json} per
docs/superpowers/specs/2026-07-13-chessqa-explorer-design.md (rev 2); the default out-dir
is the sibling showcase repo's public/data. Read-only on the DB; raw provider payloads
(raw_message etc.) are never exported.
"""

import re
import sqlite3
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
MIN_RESULTS = 50
CATEGORY_SLUGS = {
    "Structural": "structural",
    "Motifs": "motifs",
    "Short Tactics": "short-tactics",
    "Position Judgement": "position-judgement",
    "Semantic": "semantic",
}

SELECT_RUNS_SQL = """
SELECT r.* FROM runs r
WHERE r.status = 'complete'
  AND (SELECT COUNT(*) FROM results x WHERE x.run_id = r.run_id) >= :min_results
ORDER BY r.run_key
"""


def select_runs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Canonical runs per the spec's rule — never hard-code run ids."""
    return list(conn.execute(SELECT_RUNS_SQL, {"min_results": MIN_RESULTS}))


def run_display_name(run_key: str, model: str, enable_thinking: int) -> str:
    short_model = model.split("/", 1)[-1]
    if run_key.endswith("-verbose-cot"):
        return f"{short_model} (verbose CoT)"
    if enable_thinking:
        return f"{short_model} (thinking)"
    return f"{short_model} (no thinking)"


UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")


def split_input(input_str: str | None) -> tuple[str, list[str]]:
    """Task inputs are 'FEN' or 'FEN | uci moves' (state-tracking); split them."""
    if not input_str:
        return "", []
    if "|" in input_str:
        fen, moves = input_str.split("|", 1)
        return fen.strip(), moves.split()
    return input_str.strip(), []


def move_legality(fen: str, extracted: str | None) -> str | None:
    """Classify a single-move answer against the position. None = nothing to check."""
    if not extracted or not extracted.strip():
        return None
    text = extracted.strip().lower()
    if not UCI_RE.match(text):
        return "unparseable"
    board = chess.Board(fen)  # task FENs are trusted; a bad one should crash the export
    return "legal" if chess.Move.from_uci(text) in board.legal_moves else "illegal"


def outcome_code(error_type: str | None, legality: str | None) -> str:
    """Collapse the runner's error taxonomy to the heatmap palette (spec rev 2)."""
    if error_type == "correct":
        return "correct"
    if error_type == "max_token_reached":
        return "capped"
    if error_type == "format_error":
        return "format_error"
    if legality == "illegal":
        return "illegal"
    return "wrong"
