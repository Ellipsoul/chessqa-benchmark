"""Export the canonical smoke runs from the SQLite results DB to static JSON for the web explorer.

Produces <out-dir>/{index.json, categories/<slug>.json, traces/<task_id>.json} per
docs/superpowers/specs/2026-07-13-chessqa-explorer-design.md (rev 2); the default out-dir
is the sibling showcase repo's public/data. Read-only on the DB; raw provider payloads
(raw_message etc.) are never exported.
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))
import storage  # noqa: E402
from run_benchmark import format_prompt  # noqa: E402

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


PIECE_AT_RE = re.compile(r"(White|Black)\s+(King|Queen|Rook|Bishop|Knight|Pawn)\s+at\s+([a-h][1-8])", re.IGNORECASE)
ARRANGEMENT_RE = re.compile(r"(White|Black)\s+(King|Queen|Rook|Bishop|Knight|Pawn):\s*\[([^\]]*)\]", re.IGNORECASE)


def _parse_chain(text: str) -> dict:
    arrows = []
    for part in _split_multi(text):
        squares = [square.strip() for square in part.split(">")]
        if len(squares) < 2 or not all(SQUARE_RE.match(square) for square in squares):
            raise ValueError(part)
        arrows.extend({"from": a, "to": b} for a, b in zip(squares, squares[1:], strict=False))
    return {"type": "chain", "arrows": arrows}


def _parse_fork(text: str) -> dict:
    arrows = []
    for part in _split_multi(text):
        if ">" not in part:
            raise ValueError(part)
        forker, victims_blob = part.split(">", 1)
        forker = forker.strip()
        victims = [victim.strip() for victim in victims_blob.split("-")]
        if not SQUARE_RE.match(forker) or not all(SQUARE_RE.match(victim) for victim in victims):
            raise ValueError(part)
        arrows.extend({"from": forker, "to": victim} for victim in victims)
    return {"type": "chain", "arrows": arrows}


def _parse_pieces_at(text: str) -> dict:
    items = [
        {"color": color.title(), "piece": piece.title(), "square": square.lower()}
        for color, piece, square in PIECE_AT_RE.findall(text)
    ]
    if not items:
        raise ValueError(text)
    return {"type": "pieces", "items": items}


def _parse_piece_arrangement(text: str) -> dict:
    items = []
    for color, piece, squares_blob in ARRANGEMENT_RE.findall(text):
        for square in re.findall(r"[a-h][1-8]", squares_blob):
            items.append({"color": color.title(), "piece": piece.title(), "square": square})
    if not items:
        raise ValueError(text)
    return {"type": "pieces", "items": items}


def _parse_fen_answer(text: str, correct_fen: str | None) -> dict:
    try:
        board = chess.Board(text)
    except ValueError as exc:
        raise ValueError(text) from exc
    diff_squares = []
    if correct_fen:
        reference = chess.Board(correct_fen)
        diff_squares = sorted(
            chess.square_name(square)
            for square in chess.SQUARES
            if board.piece_at(square) != reference.piece_at(square)
        )
    return {"type": "fen", "fen": board.fen(), "diff_squares": diff_squares}


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
        if task_type in ("motifs_pin", "motifs_skewer", "motifs_battery"):
            return _parse_chain(text)
        if task_type == "motifs_fork":
            return _parse_fork(text)
        if task_type == "structural_check_detection":
            return _parse_pieces_at(text)
        if task_type == "structural_piece_arrangement":
            return _parse_piece_arrangement(text)
        if task_type.startswith("structural_state_tracking"):
            return _parse_fen_answer(text, correct_fen)
        raise ValueError(task_type)  # unknown family: text fallback keeps the export alive
    except ValueError:
        return {"type": "text", "text": text}


def resolve_prompt(task: sqlite3.Row) -> str:
    """Fill CONTEXT/FORMAT_EXAMPLE placeholders exactly as the runner did (group 1, no context)."""
    task_dict = {
        "question": task["question"],
        "input": task["input"],
        "format_examples": json.loads(task["format_examples"]) if task["format_examples"] else [],
    }
    return format_prompt(task_dict, add_context=False, format_example_group=1)


def _write_json(path: Path, obj) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    path.write_text(blob, encoding="utf-8")
    return len(blob.encode("utf-8"))


def _run_summary(conn: sqlite3.Connection, run: sqlite3.Row, outcomes_by_task: dict) -> dict:
    rows = list(conn.execute("SELECT * FROM results WHERE run_id = ?", (run["run_id"],)))
    slug = run["run_key"]
    outcome_list = [outcomes_by_task[row["task_id"]][slug] for row in rows]
    thinking_sources: dict[str, int] = {}
    for row in rows:
        source = row["thinking_source"] or "none"
        thinking_sources[source] = thinking_sources.get(source, 0) + 1
    costs = [row["cost_usd"] for row in rows if row["cost_usd"] is not None]
    tokens = [row["completion_tokens"] for row in rows if row["completion_tokens"] is not None]
    return {
        "slug": slug,
        "display_name": run_display_name(slug, run["model"], run["enable_thinking"]),
        "model": run["model"],
        "backend": run["backend"],
        "enable_thinking": bool(run["enable_thinking"]),
        "reasoning_config": json.loads(run["reasoning_config"]) if run["reasoning_config"] else None,
        "git_commit": run["git_commit"],
        "started_at": run["started_at"],
        "n_results": len(rows),
        "n_correct": sum(1 for o in outcome_list if o == "correct"),
        "n_capped": sum(1 for o in outcome_list if o == "capped"),
        "n_illegal": sum(1 for o in outcome_list if o == "illegal"),
        "total_cost_usd": round(sum(costs), 4) if costs else None,
        "avg_completion_tokens": round(sum(tokens) / len(tokens)) if tokens else None,
        "thinking_sources": thinking_sources,
    }


def build_export(conn: sqlite3.Connection, out_dir: Path) -> dict:
    runs = select_runs(conn)
    if not runs:
        raise SystemExit("No canonical runs found (status='complete' with >=50 results).")
    for run in runs:  # resolved_prompt is exported once per task; that only holds if all runs share the variant flags
        if run["add_context"] or run["format_example_group"] != 1:
            raise SystemExit(f"{run['run_key']}: prompt-variant flags differ; make resolved_prompt per-run first.")
    run_ids = [run["run_id"] for run in runs]
    slug_by_id = {run["run_id"]: run["run_key"] for run in runs}

    placeholders = ",".join("?" * len(run_ids))
    task_rows = list(conn.execute(
        f"""SELECT DISTINCT t.* FROM tasks t
            JOIN results r ON r.task_id = t.task_id AND r.run_id IN ({placeholders})
            ORDER BY t.task_category, t.task_type""", run_ids))
    result_rows = list(conn.execute(
        f"SELECT * FROM results WHERE run_id IN ({placeholders})", run_ids))
    by_task: dict[str, dict[str, sqlite3.Row]] = {}
    for row in result_rows:
        by_task.setdefault(row["task_id"], {})[slug_by_id[row["run_id"]]] = row

    total_bytes = 0
    outcomes_by_task: dict[str, dict[str, str]] = {}
    categories: dict[str, dict] = {}
    for task in task_rows:
        fen, input_moves = split_input(task["input"])
        slug = CATEGORY_SLUGS[task["task_category"]]
        is_tactics_single = task["task_type"].startswith("short_tactics")
        task_results, outcomes, traces = [], {}, {}
        for run in runs:
            row = by_task[task["task_id"]][run["run_key"]]
            legality = move_legality(fen, row["extracted"]) if is_tactics_single else None
            outcome = outcome_code(row["error_type"], legality)
            outcomes[run["run_key"]] = outcome
            task_results.append({
                "run": run["run_key"],
                "extracted": row["extracted"],
                "error_type": row["error_type"],
                "outcome": outcome,
                "legality": legality,
                "primitives": parse_answer_primitives(task["task_type"], row["extracted"], correct_fen=task["correct_answer"]),
                "cost_usd": row["cost_usd"],
                "completion_tokens": row["completion_tokens"],
                "reasoning_tokens": row["reasoning_tokens"],
                "latency_ms": row["latency_ms"],
                "n_attempts": row["n_attempts"],
                "thinking_source": row["thinking_source"],
                "thinking_chars": len(row["thinking_content"] or ""),
            })
            traces[run["run_key"]] = {
                "thinking_source": row["thinking_source"],
                "content": row["thinking_content"] or "",
                "response": row["response"] or "",
            }
        outcomes_by_task[task["task_id"]] = outcomes
        categories.setdefault(slug, {"category": task["task_category"], "slug": slug, "tasks": []})
        categories[slug]["tasks"].append({
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "task_category": task["task_category"],
            "question": task["question"],
            "resolved_prompt": resolve_prompt(task),
            "input_fen": fen,
            "input_moves": input_moves,
            "metadata": json.loads(task["metadata"]) if task["metadata"] else None,
            "correct_answer": task["correct_answer"],
            "answer_type": task["answer_type"],
            "correct_primitives": parse_answer_primitives(task["task_type"], task["correct_answer"]),
            "results": task_results,
        })
        total_bytes += _write_json(out_dir / "traces" / f"{task['task_id']}.json",
                                   {"task_id": task["task_id"], "traces": traces})

    for slug, payload in categories.items():
        total_bytes += _write_json(out_dir / "categories" / f"{slug}.json", payload)

    index = {
        "generated_at": max(run["updated_at"] or "" for run in runs),
        "dataset_hash": task_rows[0]["dataset_hash"],
        "runs": [_run_summary(conn, run, outcomes_by_task) for run in runs],
        "tasks": [{
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "task_category": task["task_category"],
            "category_slug": CATEGORY_SLUGS[task["task_category"]],
            "fen": split_input(task["input"])[0],
            "answer_type": task["answer_type"],
            "outcomes": outcomes_by_task[task["task_id"]],
        } for task in task_rows],
    }
    total_bytes += _write_json(out_dir / "index.json", index)
    return {"runs": len(runs), "tasks": len(task_rows), "bytes": total_bytes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export canonical smoke runs to static JSON for the web explorer")
    parser.add_argument("--db-path", type=Path, default=REPO_ROOT / "results" / "chessqa.sqlite3")
    # Default assumes the showcase repo is checked out as a sibling of this repo.
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT.parent / "chess-benchmark-showcase" / "public" / "data")
    args = parser.parse_args()
    if not args.db_path.exists():
        raise SystemExit(f"{args.db_path} not found — rebuild with: python eval/storage.py ingest results/*.jsonl --dataset-root benchmark")
    conn = storage.connect(args.db_path)
    try:
        summary = build_export(conn, args.out_dir)
    finally:
        conn.close()
    print(f"exported {summary['runs']} runs x {summary['tasks']} tasks -> {args.out_dir} ({summary['bytes']/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
