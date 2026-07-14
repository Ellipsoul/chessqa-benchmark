# ChessQA Explorer — Exporter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `eval/export_web.py`, which reads `results/chessqa.sqlite3` and writes the static JSON consumed by the Explorer web app (`index.json`, `categories/<slug>.json`, `traces/<task_id>.json`).

**Architecture:** Single flat script in `eval/` (matching the repo's script-not-package layout), pure functions for all chess/answer parsing, one `build_export()` orchestrator, argparse `main()`. Tests build a throwaway DB from the checked-in results JSONLs + benchmark tasks (the sqlite file itself is gitignored), so everything runs on a fresh clone.

**Tech Stack:** Python stdlib + `chess` (python-chess 1.10, already in requirements.txt) + existing `eval/storage.py` and `format_prompt` from `eval/run_openrouter.py`.

**Spec:** `docs/superpowers/specs/2026-07-13-chessqa-explorer-design.md` (rev 2). This plan is subsystem 1 of 2; subsystem 2 (Next.js app) is `2026-07-13-explorer-web.md` and consumes the JSON this plan produces.

**Layout update (2026-07-14):** the web app is a separate sibling repo,
`~/Desktop/Coding_Adventures/chess-benchmark-showcase` — not `web/` in this repo. The
exporter's default `--out-dir` is therefore `../chess-benchmark-showcase/public/data`
(sibling-checkout assumption; the flag overrides it). The exported JSON gets committed in
the showcase repo, never here.

## Global Constraints

- Included runs = `status='complete'` AND ≥ 50 results. Never hard-code run_ids.
- `raw_message`, `usage_json`, `provider_meta` must never appear in any exported file.
- Output must be byte-deterministic: `json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))` + `"\n"`, lists explicitly sorted (runs by `run_key`, tasks by `(category_slug, task_type)`, results by run slug).
- Category slugs, exactly: `Structural→structural`, `Motifs→motifs`, `Short Tactics→short-tactics`, `Position Judgement→position-judgement`, `Semantic→semantic` (note British spelling "Judgement" in the DB).
- Run slug = `run_key` verbatim (already unique + URL-safe, e.g. `anthropic_claude-sonnet-5-verbose-cot`).
- Outcome codes, exactly: `correct | wrong | illegal | capped | format_error`.
- Default paths: repo root = `Path(__file__).resolve().parent.parent` (eval/'s parent — do NOT copy the `parent.parent.parent` pattern from older scripts; CLAUDE.md documents that trap).
- Lint: `make lint` (ruff) must stay clean; tests via `.venv/bin/python -m pytest tests/ -q`.
- Commits on a feature branch (e.g. `feat/explorer-exporter`); PRs to `Ellipsoul/chessqa-benchmark` only, never CSSLab.

---

### Task 1: Run selection, slugs, display names

**Files:**
- Create: `eval/export_web.py`
- Create: `tests/test_export_web.py`

**Interfaces:**
- Consumes: `storage.connect(db_path)`, `storage.ingest_jsonl(conn, path)`, `storage.upsert_tasks(conn, tasks, ...)` (existing).
- Produces: `select_runs(conn) -> list[sqlite3.Row]`; `run_display_name(run_key: str, model: str, enable_thinking: int) -> str`; module constants `CATEGORY_SLUGS: dict[str, str]`, `MIN_RESULTS = 50`. The test module's `db` fixture (session-scoped, real-data) is reused by every later task.

- [ ] **Step 1: Write the failing tests + shared fixture**

```python
# tests/test_export_web.py
"""Exporter tests: run selection, answer→primitive parsing, and full-export integration.

The DB fixture is rebuilt from the checked-in results JSONLs + benchmark tasks because
results/chessqa.sqlite3 is gitignored (derived artifact) — this mirrors `storage.py ingest`.
"""

import json
from pathlib import Path

import pytest

import export_web
import storage

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def db(tmp_path_factory):
    conn = storage.connect(tmp_path_factory.mktemp("export") / "test.sqlite3")
    for task_file in sorted((REPO_ROOT / "benchmark").glob("*.jsonl")):
        tasks = [json.loads(line) for line in task_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        storage.upsert_tasks(conn, tasks, source_file=task_file.name)
    for results_file in sorted((REPO_ROOT / "results").glob("*.jsonl")):
        storage.ingest_jsonl(conn, results_file)
    yield conn
    conn.close()


def test_select_runs_applies_min_results_rule(db):
    runs = export_web.select_runs(db)
    keys = [run["run_key"] for run in runs]
    assert len(runs) == 16
    assert keys == sorted(keys)  # deterministic order
    # the five 1-task probes are excluded by the >=50 rule
    assert not any("opus-4.6" in k or "fable-5" in k or "gpt-5.1" in k or "grok-4.5" in k or "glm-5.2" in k for k in keys)
    # both sonnet-5 variants survive as distinct runs
    assert "anthropic_claude-sonnet-5-thinking" in keys
    assert "anthropic_claude-sonnet-5-verbose-cot" in keys


def test_run_display_names():
    assert export_web.run_display_name("anthropic_claude-sonnet-5-thinking", "anthropic/claude-sonnet-5", 1) == "claude-sonnet-5 (thinking)"
    # verbose-cot beats the enable_thinking flag (the sidecar records what was *requested*)
    assert export_web.run_display_name("anthropic_claude-sonnet-5-verbose-cot", "anthropic/claude-sonnet-5", 1) == "claude-sonnet-5 (verbose CoT)"
    assert export_web.run_display_name("anthropic_claude-haiku-4.5", "anthropic/claude-haiku-4.5", 0) == "claude-haiku-4.5 (no thinking)"
    assert export_web.run_display_name("meta_llama-4-maverick", "meta/llama-4-maverick", 0) == "llama-4-maverick (no thinking)"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_export_web.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'export_web'` (conftest.py already puts `eval/` on sys.path).

- [ ] **Step 3: Write the module skeleton with selection + naming**

```python
# eval/export_web.py
"""Export the canonical smoke runs from the SQLite results DB to static JSON for web/.

Produces web/public/data/{index.json, categories/<slug>.json, traces/<task_id>.json}
per docs/superpowers/specs/2026-07-13-chessqa-explorer-design.md (rev 2). Read-only on
the DB; raw provider payloads (raw_message etc.) are never exported.
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
from run_openrouter import format_prompt  # noqa: E402

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_export_web.py -v`
Expected: 2 passed (session fixture takes ~10–20 s to ingest once).

- [ ] **Step 5: Lint and commit**

```bash
make lint
git add eval/export_web.py tests/test_export_web.py
git commit -m "feat(export): run selection rule and display names for web exporter"
```

---

### Task 2: Move legality + outcome codes

**Files:**
- Modify: `eval/export_web.py` (append)
- Modify: `tests/test_export_web.py` (append)

**Interfaces:**
- Produces: `move_legality(fen: str, extracted: str | None) -> str | None` returning `"legal" | "illegal" | "unparseable" | None`; `outcome_code(error_type: str | None, legality: str | None) -> str`; `UCI_RE` regex; `split_input(input_str: str | None) -> tuple[str, list[str]]`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_export_web.py
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_move_legality():
    assert export_web.move_legality(START_FEN, "e2e4") == "legal"
    assert export_web.move_legality(START_FEN, "e2e5") == "illegal"     # pawn can't triple-step
    assert export_web.move_legality(START_FEN, "Qh5") == "unparseable"  # SAN, not UCI
    assert export_web.move_legality(START_FEN, "e7e8q") == "illegal"    # promotion syntax accepted, move illegal here
    assert export_web.move_legality(START_FEN, None) is None
    assert export_web.move_legality(START_FEN, "  E2E4 ") == "legal"    # case/whitespace tolerant


def test_split_input():
    fen, moves = export_web.split_input("4kb1r/5ppp/p2p1q2/8 w KQk - 0 16 | d1b3 f8e7")
    assert fen == "4kb1r/5ppp/p2p1q2/8 w KQk - 0 16"
    assert moves == ["d1b3", "f8e7"]
    fen, moves = export_web.split_input("8/8/8/8/8/8/8/K6k w - - 0 1")
    assert moves == []


def test_outcome_codes():
    assert export_web.outcome_code("correct", "legal") == "correct"
    assert export_web.outcome_code("max_token_reached", None) == "capped"
    assert export_web.outcome_code("format_error", None) == "format_error"
    assert export_web.outcome_code("wrong_answer", "illegal") == "illegal"
    assert export_web.outcome_code("wrong_answer", "legal") == "wrong"
    assert export_web.outcome_code("wrong_answer", "unparseable") == "wrong"  # unparseable is a badge, not an outcome
    assert export_web.outcome_code("multi_extra_items", None) == "wrong"      # multi partials collapse; raw error_type is exported alongside
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_export_web.py -v -k "legality or split or outcome"`
Expected: FAIL with `AttributeError: module 'export_web' has no attribute 'move_legality'`.

- [ ] **Step 3: Implement**

```python
# append to eval/export_web.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_export_web.py -v`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
make lint
git add eval/export_web.py tests/test_export_web.py
git commit -m "feat(export): move-legality check and outcome codes"
```

---

### Task 3: Answer → primitive parsers (moves, squares, eval, choice)

**Files:**
- Modify: `eval/export_web.py` (append)
- Modify: `tests/test_export_web.py` (append)

**Interfaces:**
- Produces: `parse_answer_primitives(task_type: str, answer: str | None, correct_fen: str | None = None) -> dict` — always returns a dict with a `"type"` key; garbage falls back to `{"type": "text", "text": ...}`. Internal helpers `_split_multi`, `_parse_uci_arrow`. Task 4 extends the same dispatcher — do not rename it.

- [ ] **Step 1: Write the failing tests** (answer strings are verbatim from the real dataset)

```python
# append to tests/test_export_web.py
def test_primitives_moves_single_and_multi():
    p = export_web.parse_answer_primitives("short_tactics_theme_mateIn2", "h6h7")
    assert p == {"type": "moves", "arrows": [{"from": "h6", "to": "h7"}]}
    p = export_web.parse_answer_primitives("short_tactics_theme_promotion", "e7e8q")
    assert p["arrows"] == [{"from": "e7", "to": "e8", "promotion": "q"}]
    p = export_web.parse_answer_primitives("structural_check_in_1", "d2d3, d2g2")
    assert [a["to"] for a in p["arrows"]] == ["d3", "g2"]


def test_primitives_squares():
    p = export_web.parse_answer_primitives("structural_protect_squares", "f6, f8")
    assert p == {"type": "squares", "squares": ["f6", "f8"]}


def test_primitives_eval_and_choice():
    assert export_web.parse_answer_primitives("position_judgement_losing", "-400") == {"type": "eval", "value": -400}
    assert export_web.parse_answer_primitives("position_judgement_advantage", "+200") == {"type": "eval", "value": 200}
    assert export_web.parse_answer_primitives("semantic_keyword", "A") == {"type": "choice", "letter": "A"}
    assert export_web.parse_answer_primitives("semantic_keyword", "(c)") == {"type": "choice", "letter": "C"}


def test_primitives_fallbacks():
    # model wrote prose instead of a move -> text fallback, never an exception
    p = export_web.parse_answer_primitives("short_tactics_rating_expert", "The best move is Qxe6!")
    assert p == {"type": "text", "text": "The best move is Qxe6!"}
    assert export_web.parse_answer_primitives("motifs_pin", "None") == {"type": "none"}
    assert export_web.parse_answer_primitives("short_tactics_rating_expert", None) == {"type": "none"}
    assert export_web.parse_answer_primitives("short_tactics_rating_expert", "  ") == {"type": "none"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_export_web.py -v -k primitives`
Expected: FAIL with `AttributeError ... parse_answer_primitives`.

- [ ] **Step 3: Implement dispatcher + these four families**

```python
# append to eval/export_web.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_export_web.py -v`
Expected: all pass (motif/pieces/FEN tests don't exist yet).

- [ ] **Step 5: Lint and commit**

```bash
make lint
git add eval/export_web.py tests/test_export_web.py
git commit -m "feat(export): primitive parsers for moves, squares, eval, choice"
```

---

### Task 4: Answer → primitive parsers (motif chains, pieces, FEN diff)

**Files:**
- Modify: `eval/export_web.py` (extend `parse_answer_primitives` and add helpers)
- Modify: `tests/test_export_web.py` (append)

**Interfaces:**
- Produces: the remaining primitive families inside the existing `parse_answer_primitives`. Semantics verified against `dataset/02_motifs.py`: pin/skewer = `a>b>c` chains (arrows between consecutive squares), battery = `s1>s2>…` square chain (same rendering), fork = `forker>victim1-victim2` (arrows forker→each victim), discovered/double check = UCI moves (already handled in Task 3 via `MOVES_TASK_TYPES`).

- [ ] **Step 1: Write the failing tests** (answers verbatim from the real dataset)

```python
# append to tests/test_export_web.py
def test_primitives_motif_chains():
    p = export_web.parse_answer_primitives("motifs_pin", "b4>d2>e1")
    assert p == {"type": "chain", "arrows": [{"from": "b4", "to": "d2"}, {"from": "d2", "to": "e1"}]}
    p = export_web.parse_answer_primitives("motifs_skewer", "d3>e4>f5")
    assert len(p["arrows"]) == 2
    p = export_web.parse_answer_primitives("motifs_battery", "b3>e3")
    assert p == {"type": "chain", "arrows": [{"from": "b3", "to": "e3"}]}
    p = export_web.parse_answer_primitives("motifs_fork", "d3>c5-f2")
    assert p == {"type": "chain", "arrows": [{"from": "d3", "to": "c5"}, {"from": "d3", "to": "f2"}]}
    # multiple motifs, comma-separated
    p = export_web.parse_answer_primitives("motifs_pin", "b4>d2>e1, a4>c2>e2")
    assert len(p["arrows"]) == 4


def test_primitives_pieces():
    p = export_web.parse_answer_primitives("structural_check_detection", "White Knight at f6")
    assert p == {"type": "pieces", "items": [{"color": "White", "piece": "Knight", "square": "f6"}]}
    arrangement = "White King: ['c1'], White Rook: ['d1', 'h1'], Black Queen: ['f6']"
    p = export_web.parse_answer_primitives("structural_piece_arrangement", arrangement)
    assert {"color": "White", "piece": "Rook", "square": "h1"} in p["items"]
    assert len(p["items"]) == 4


def test_primitives_fen_diff():
    correct = "5rk1/4bppp/P2p4/8/1P1rp3/N7/P4PPP/1R3RK1 b - - 2 22"
    # model imagined the rook on d5 instead of d4 -> both squares differ from the reference
    wrong = "5rk1/4bppp/P2p4/3r4/1P2p3/N7/P4PPP/1R3RK1 b - - 2 22"
    p = export_web.parse_answer_primitives("structural_state_tracking_long", wrong, correct_fen=correct)
    assert p["type"] == "fen"
    assert p["diff_squares"] == ["d4", "d5"]
    # the correct answer itself: no reference passed -> no diff
    p = export_web.parse_answer_primitives("structural_state_tracking_long", correct)
    assert p["diff_squares"] == []
    # invalid FEN from a confused model -> text fallback
    p = export_web.parse_answer_primitives("structural_state_tracking_long", "not a fen at all")
    assert p["type"] == "text"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_export_web.py -v -k "motif or pieces or fen_diff"`
Expected: FAIL — these families currently fall through to `{"type": "text"}`.

- [ ] **Step 3: Implement**

```python
# append to eval/export_web.py (helpers), and add the dispatch lines inside parse_answer_primitives
PIECE_AT_RE = re.compile(r"(White|Black)\s+(King|Queen|Rook|Bishop|Knight|Pawn)\s+at\s+([a-h][1-8])", re.IGNORECASE)
ARRANGEMENT_RE = re.compile(r"(White|Black)\s+(King|Queen|Rook|Bishop|Knight|Pawn):\s*\[([^\]]*)\]", re.IGNORECASE)


def _parse_chain(text: str) -> dict:
    arrows = []
    for part in _split_multi(text):
        squares = [square.strip() for square in part.split(">")]
        if len(squares) < 2 or not all(SQUARE_RE.match(square) for square in squares):
            raise ValueError(part)
        arrows.extend({"from": a, "to": b} for a, b in zip(squares, squares[1:]))
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
```

Inside `parse_answer_primitives`, replace the `raise ValueError(task_type)` line with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_export_web.py -v`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
make lint
git add eval/export_web.py tests/test_export_web.py
git commit -m "feat(export): motif-chain, piece, and FEN-diff primitives"
```

---

### Task 5: Payload builders (index, categories, traces)

**Files:**
- Modify: `eval/export_web.py` (append)
- Modify: `tests/test_export_web.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 1–4; `format_prompt(task: dict, add_context: bool, format_example_group: int) -> str` (imported in Task 1).
- Produces: `build_export(conn, out_dir: Path) -> dict` (returns `{"runs": int, "tasks": int, "bytes": int}` summary); `resolve_prompt(task_row: sqlite3.Row) -> str`; `_write_json(path: Path, obj) -> int`. JSON schemas as below — the web plan's `lib/types.ts` mirrors these names exactly:
  - `index.json`: `{generated_at, dataset_hash, runs: [{slug, display_name, model, backend, enable_thinking, reasoning_config, git_commit, started_at, n_results, n_correct, n_capped, n_illegal, total_cost_usd, avg_completion_tokens, thinking_sources}], tasks: [{task_id, task_type, task_category, category_slug, fen, answer_type, outcomes: {run_slug: outcome}}]}`
  - `categories/<slug>.json`: `{category, slug, tasks: [{task_id, task_type, task_category, question, resolved_prompt, input_fen, input_moves, metadata, correct_answer, answer_type, correct_primitives, results: [{run, extracted, error_type, outcome, legality, primitives, cost_usd, completion_tokens, reasoning_tokens, latency_ms, n_attempts, thinking_source, thinking_chars}]}]}`
  - `traces/<task_id>.json`: `{task_id, traces: {run_slug: {thinking_source, content, response}}}` (`response` = the visible answer text, so the drawer can show the final message too).

- [ ] **Step 1: Write the failing integration test**

```python
# append to tests/test_export_web.py
@pytest.fixture(scope="session")
def export_dir(db, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("data")
    export_web.build_export(db, out_dir)
    return out_dir


def test_export_file_inventory(export_dir):
    assert (export_dir / "index.json").exists()
    assert len(list((export_dir / "categories").glob("*.json"))) == 5
    assert len(list((export_dir / "traces").glob("*.json"))) == 50


def test_index_shape(export_dir):
    index = json.loads((export_dir / "index.json").read_text())
    assert len(index["runs"]) == 16
    assert len(index["tasks"]) == 50
    for task in index["tasks"]:
        assert len(task["outcomes"]) == 16
        assert set(task["outcomes"].values()) <= {"correct", "wrong", "illegal", "capped", "format_error"}
    gemini = next(r for r in index["runs"] if r["slug"] == "google_gemini-3.1-pro-preview-thinking")
    assert gemini["n_correct"] == 45 and gemini["n_results"] == 50


def test_category_payloads_resolved_and_private(export_dir):
    shorttac = json.loads((export_dir / "categories" / "short-tactics.json").read_text())
    assert len(shorttac["tasks"]) == 24
    for task in shorttac["tasks"]:
        assert "CONTEXT_PLACEHOLDER" not in task["resolved_prompt"]
        assert "FORMAT_EXAMPLE_PLACEHOLDER" not in task["resolved_prompt"]
        assert len(task["results"]) == 16
        for result in task["results"]:
            assert "thinking_content" not in result and "raw_message" not in result
    # state tracking: input carries the uci move prefix for the animation
    structural = json.loads((export_dir / "categories" / "structural.json").read_text())
    tracking = next(t for t in structural["tasks"] if t["task_type"] == "structural_state_tracking_long")
    assert len(tracking["input_moves"]) > 0


def test_traces_have_capped_exhibit(export_dir):
    blob = json.loads((export_dir / "traces" / "short_tactics_theme_defensiveMove_0024.json").read_text())
    gem = blob["traces"]["google_gemini-3.1-pro-preview-thinking"]
    assert len(gem["content"]) > 200_000  # the 266K-char capped loop survives the pipeline
    assert gem["response"] == ""


def test_export_is_deterministic(db, export_dir, tmp_path_factory):
    second = tmp_path_factory.mktemp("data2")
    export_web.build_export(db, second)
    for path in sorted(export_dir.rglob("*.json")):
        other = second / path.relative_to(export_dir)
        assert path.read_bytes() == other.read_bytes(), f"nondeterministic: {path.name}"
```

Note on determinism: `generated_at` would break byte-identity — the implementation derives it from the max `runs.updated_at` in the DB, not the wall clock (provenance without nondeterminism).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_export_web.py -v -k "export or index or category or traces or deterministic"`
Expected: FAIL with `AttributeError ... build_export`.

- [ ] **Step 3: Implement the builders**

```python
# append to eval/export_web.py
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
```

One subtlety encoded above: for state-tracking model answers, `correct_fen=task["correct_answer"]` is what produces `diff_squares` (imagined board vs correct board); for every non-FEN family the parser ignores that argument, and `correct_primitives` is built without a reference so the correct board shows no diff.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_export_web.py -v`
Expected: all pass. If `test_index_shape`'s gemini assertion fails, print the actual run summaries — the DB may have gained runs since this plan was written; fix the *test expectation* only after confirming against `results/*_stats.json`.

- [ ] **Step 5: Lint and commit**

```bash
make lint
git add eval/export_web.py tests/test_export_web.py
git commit -m "feat(export): index/category/trace payload builders"
```

---

### Task 6: CLI, Makefile target, real export sanity check

**Files:**
- Modify: `eval/export_web.py` (append `main()`)
- Modify: `Makefile` (add `export-web` target)
- Modify: `CLAUDE.md` (one line in Commands)

**Interfaces:**
- Produces: `python eval/export_web.py [--db-path results/chessqa.sqlite3] [--out-dir ../chess-benchmark-showcase/public/data]` and `make export-web`. **Note:** generated JSON is NOT committed in this plan — it lands in the sibling showcase repo, and committing it there is Task 1 of the web plan.

- [ ] **Step 1: Write `main()`**

```python
# append to eval/export_web.py
def main() -> None:
    parser = argparse.ArgumentParser(description="Export canonical smoke runs to static JSON for web/")
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
```

- [ ] **Step 2: Add the Makefile target** (after the `test:` target)

```make
export-web:
	$(PY) eval/export_web.py --db-path results/chessqa.sqlite3 --out-dir ../chess-benchmark-showcase/public/data
```
Add `export-web` to the `.PHONY` line.

- [ ] **Step 3: Run the real export to a scratch dir and eyeball it**

Run: `.venv/bin/python eval/export_web.py --out-dir /tmp/explorer-data-check`
Expected output line: `exported 16 runs x 50 tasks -> /tmp/explorer-data-check (~17 MB)`.
Then: `python3 -c "import json; d=json.load(open('/tmp/explorer-data-check/index.json')); print([r['display_name'] for r in d['runs']])"` — confirm the sonnet-5 pair reads `(thinking)` / `(verbose CoT)`.

- [ ] **Step 4: Full test suite + lint**

Run: `.venv/bin/python -m pytest tests/ -q && make lint`
Expected: whole suite green (including pre-existing tests), lint clean.

- [ ] **Step 5: Document + commit**

Add to CLAUDE.md's Commands block (after the storage ingest line):
```bash
# Export canonical runs to static JSON for the web explorer (writes web/public/data/)
make export-web
```

```bash
git add eval/export_web.py Makefile CLAUDE.md
git commit -m "feat(export): CLI entrypoint and make export-web target"
```

Then open the PR: `gh pr create --repo Ellipsoul/chessqa-benchmark --base main ...` and verify the printed URL starts with `github.com/Ellipsoul/`.
