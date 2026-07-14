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
