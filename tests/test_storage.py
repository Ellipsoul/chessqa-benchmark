"""Storage-layer tests: schema, idempotent ingest of the real smoke-run fixtures, live-run parity."""

import json
from pathlib import Path

import pytest

import storage

REPO_ROOT = Path(__file__).parent.parent
HAIKU_JSONL = REPO_ROOT / "results" / "anthropic_claude-haiku-4.5.jsonl"
SONNET_JSONL = REPO_ROOT / "results" / "anthropic_claude-sonnet-5-thinking.jsonl"


@pytest.fixture()
def conn(tmp_path):
    connection = storage.connect(tmp_path / "test.sqlite3")
    yield connection
    connection.close()


def test_schema_created(conn):
    tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"schema_meta", "tasks", "runs", "results", "attempts"} <= tables
    version = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()
    assert version["value"] == storage.DB_SCHEMA_VERSION


def test_run_meta_from_filename():
    meta = storage.run_meta_from_filename("results/anthropic_claude-sonnet-5-thinking.jsonl")
    assert meta["enable_thinking"] is True and meta["model"] == "anthropic_claude-sonnet-5"
    meta = storage.run_meta_from_filename("x/gpt-5-thinking-piecearr-fmt2-openrouter.jsonl")
    assert meta == {
        "backend": "openrouter", "enable_thinking": True, "add_context": True,
        "format_example_group": 2, "model": "gpt-5",
    }
    meta = storage.run_meta_from_filename("x/anthropic_claude-haiku-4.5.jsonl")
    assert meta["enable_thinking"] is False and meta["backend"] == "vercel-gateway"


def test_ingest_smoke_fixtures(conn):
    """Both real smoke files ingest; aggregates reproduce the stats sidecars from SQL."""
    _, haiku_rows = storage.ingest_jsonl(conn, HAIKU_JSONL)
    _, sonnet_rows = storage.ingest_jsonl(conn, SONNET_JSONL)
    assert haiku_rows == 50 and sonnet_rows == 50

    assert conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"] == 2
    assert conn.execute("SELECT COUNT(*) AS n FROM results").fetchone()["n"] == 100

    # Accuracy reproduced from SQL matches the stats sidecar (0.68 sonnet, 0.14 haiku)
    row = conn.execute(
        """SELECT r.run_key, AVG(res.is_correct) AS accuracy, SUM(res.cost_usd) AS cost
           FROM results res JOIN runs r ON r.run_id = res.run_id GROUP BY r.run_key"""
    ).fetchall()
    by_key = {record["run_key"]: record for record in row}
    assert by_key["anthropic_claude-haiku-4.5"]["accuracy"] == pytest.approx(0.14)
    assert by_key["anthropic_claude-sonnet-5-thinking"]["accuracy"] == pytest.approx(0.68)
    assert by_key["anthropic_claude-haiku-4.5"]["cost"] == pytest.approx(0.1972, abs=0.001)

    # Run metadata came from the stats sidecar
    sonnet_run = conn.execute(
        "SELECT * FROM runs WHERE run_key = 'anthropic_claude-sonnet-5-thinking'"
    ).fetchone()
    assert sonnet_run["enable_thinking"] == 1
    assert sonnet_run["backend"] == "vercel-gateway"
    assert sonnet_run["status"] == "complete"
    assert json.loads(sonnet_run["stats"])["accuracy"] == 0.68

    # Pre-storage-era rows: prompt_hash derived, raw_message/attempts NULL
    sample = conn.execute("SELECT prompt_hash, raw_message, n_attempts FROM results LIMIT 1").fetchone()
    assert sample["prompt_hash"] is not None and len(sample["prompt_hash"]) == 64
    assert sample["raw_message"] is None and sample["n_attempts"] is None


def test_ingest_idempotent(conn):
    storage.ingest_jsonl(conn, HAIKU_JSONL)
    first = conn.execute("SELECT COUNT(*) AS n FROM results").fetchone()["n"]
    storage.ingest_jsonl(conn, HAIKU_JSONL)
    second = conn.execute("SELECT COUNT(*) AS n FROM results").fetchone()["n"]
    assert first == second == 50
    assert conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"] == 1


def test_truncated_then_full_ingest_reconciles(conn, tmp_path):
    lines = HAIKU_JSONL.read_text().splitlines()
    truncated = tmp_path / HAIKU_JSONL.name
    truncated.write_text("\n".join(lines[:20]) + "\n")
    _, rows = storage.ingest_jsonl(conn, truncated)
    assert rows == 20
    _, rows = storage.ingest_jsonl(conn, HAIKU_JSONL)
    assert rows == 50
    assert conn.execute("SELECT COUNT(*) AS n FROM results").fetchone()["n"] == 50


def test_tasks_upsert_from_dataset(conn):
    benchmark_file = REPO_ROOT / "benchmark" / "motifs.jsonl"
    tasks = [json.loads(line) for line in benchmark_file.read_text().splitlines() if line.strip()]
    written = storage.upsert_tasks(conn, tasks, dataset_hash="testhash", source_file="motifs.jsonl")
    assert written == 600
    assert conn.execute("SELECT COUNT(*) AS n FROM tasks WHERE task_category='Motifs'").fetchone()["n"] == 600
    # Idempotent
    storage.upsert_tasks(conn, tasks, dataset_hash="testhash", source_file="motifs.jsonl")
    assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 600


def test_live_row_with_attempts_roundtrip(conn):
    """A post-PR-1 row (attempts, raw_message, latency) lands with all columns populated."""
    run_id = storage.get_or_create_run(conn, "test-run", {"model": "test/model", "backend": "vercel-gateway"})
    row = {
        "task_id": "synthetic_0001",
        "task_type": "synthetic",
        "task_category": "Synthetic",
        "input": "8/8/8/8/8/8/8/8 w - - 0 1",
        "question": "q",
        "format_examples": ["a", "b"],
        "correct_answer": "e2e4",
        "answer_type": "single",
        "metadata": None,
        "inference": {
            "prompt": "rendered prompt",
            "response": "FINAL ANSWER: e2e4",
            "thinking_content": "let me think",
            "thinking_source": "full_text",
            "extracted": "e2e4",
            "extraction_successful": True,
            "is_correct": True,
            "error_type": "correct",
            "usage": {
                "prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30, "cost": 0.002,
                "completion_tokens_details": {"reasoning_tokens": 15},
                "prompt_tokens_details": {"cached_tokens": 5},
            },
            "raw_message": {"role": "assistant", "content": "FINAL ANSWER: e2e4"},
            "provider_meta": {"id": "cmpl-9", "provider": "anthropic"},
            "attempts": [
                {"attempt_no": 1, "http_status": 429, "error_class": "http_429", "billed_risk": 0,
                 "retry_after_s": 3.0, "wait_s": 3.0, "outcome": "retried"},
                {"attempt_no": 2, "http_status": 200, "error_class": "ok", "billed_risk": 1, "outcome": "success"},
            ],
            "latency_ms": 812,
        },
    }
    storage.upsert_result(conn, run_id, row)
    conn.commit()

    stored = conn.execute("SELECT * FROM results WHERE task_id='synthetic_0001'").fetchone()
    assert stored["reasoning_tokens"] == 15 and stored["cached_tokens"] == 5
    assert stored["cost_usd"] == 0.002 and stored["n_attempts"] == 2 and stored["latency_ms"] == 812
    assert json.loads(stored["raw_message"])["content"] == "FINAL ANSWER: e2e4"

    attempts = conn.execute(
        "SELECT * FROM attempts WHERE task_id='synthetic_0001' ORDER BY attempt_no"
    ).fetchall()
    assert [attempt["error_class"] for attempt in attempts] == ["http_429", "ok"]
    assert attempts[0]["retry_after_s"] == 3.0

    # Re-upsert replaces, never duplicates
    storage.upsert_result(conn, run_id, row)
    conn.commit()
    assert conn.execute("SELECT COUNT(*) AS n FROM attempts WHERE task_id='synthetic_0001'").fetchone()["n"] == 2
