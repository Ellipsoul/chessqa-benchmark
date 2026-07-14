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


SQUARE_RE = re.compile(r"^[a-h][1-8]$")
MOVES_TASK_TYPES = {
    "structural_check_in_1", "structural_legal_move_all", "structural_legal_move_piece",
    "motifs_discovered_check", "motifs_double_check",
}
SQUARES_TASK_TYPES = {"structural_capture_squares", "structural_control_squares", "structural_protect_squares"}


def _split_multi(text: str) -> list[str]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise ValueError(text)
    return parts


def _parse_uci_arrow(token: str) -> dict:
    move = token.strip().lower()
    if not UCI_RE.match(move):
        raise ValueError(token)
    arrow = {"from": move[0:2], "to": move[2:4]}
    if len(move) == 5:
        arrow["promotion"] = move[4]
    return arrow


def _parse_moves(text: str) -> dict:
    return {"type": "moves", "arrows": [_parse_uci_arrow(part) for part in _split_multi(text)]}


def _parse_squares(text: str) -> dict:
    squares = _split_multi(text)
    if not all(SQUARE_RE.match(square) for square in squares):
        raise ValueError(text)
    return {"type": "squares", "squares": squares}


def _parse_eval(text: str) -> dict:
    match = re.fullmatch(r"[-+]?\d+", text.strip())
    if not match:
        raise ValueError(text)
    return {"type": "eval", "value": int(match.group())}


def _parse_choice(text: str) -> dict:
    match = re.fullmatch(r"\(?([A-Da-d])\)?\.?", text.strip())
    if not match:
        raise ValueError(text)
    return {"type": "choice", "letter": match.group(1).upper()}


def parse_answer_primitives(task_type: str, answer: str | None, correct_fen: str | None = None) -> dict:
    """Map an answer string to board-render primitives; {'type':'text'} on anything unparseable.

    Used for both correct answers and model answers — model output can be arbitrary
    garbage, so every family parser raises ValueError and we fall back rather than crash.
    """
    if answer is None or not answer.strip():
        return {"type": "none"}
    text = answer.strip()
    if text.lower() == "none":
        return {"type": "none"}
    try:
        if task_type.startswith("short_tactics") or task_type in MOVES_TASK_TYPES:
            return _parse_moves(text)
        if task_type in SQUARES_TASK_TYPES:
            return _parse_squares(text)
        if task_type.startswith("position_judgement"):
            return _parse_eval(text)
        if task_type.startswith("semantic"):
            return _parse_choice(text)
        raise ValueError(task_type)  # families added in Task 4
    except ValueError:
        return {"type": "text", "text": text}
