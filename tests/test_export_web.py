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
