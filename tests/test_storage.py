"""Storage-layer tests: schema, idempotent ingest of the real smoke-run fixtures, live-run parity."""

import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

import storage

REPO_ROOT = Path(__file__).parent.parent
HAIKU_JSONL = REPO_ROOT / "results" / "anthropic_claude-haiku-4.5.jsonl"
# The Sonnet-5 smoke run was archived under a non-"-thinking" name: --enable-thinking was
# set but the model returned zero traces (Claude 5 adaptive thinking is redacted / not
# mappable via the gateway), so the file is a verbose visible-CoT baseline. The rename also
# prevents a future *fixed* thinking run from silently resuming into it. The stats sidecar
# still records enable_thinking=true (what was requested), which ingest treats as truth.
SONNET_JSONL = REPO_ROOT / "results" / "anthropic_claude-sonnet-5-verbose-cot.jsonl"


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
    # Archived verbose baseline: no suffix tokens -> flags come from the stats sidecar at ingest
    meta = storage.run_meta_from_filename("results/anthropic_claude-sonnet-5-verbose-cot.jsonl")
    assert meta["enable_thinking"] is False and meta["model"] == "anthropic_claude-sonnet-5-verbose-cot"
    meta = storage.run_meta_from_filename("x/gpt-5-thinking-piecearr-fmt2.jsonl")
    assert meta == {
        "backend": "vercel-gateway", "enable_thinking": True, "add_context": True,
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
    assert by_key["anthropic_claude-sonnet-5-verbose-cot"]["accuracy"] == pytest.approx(0.68)
    assert by_key["anthropic_claude-haiku-4.5"]["cost"] == pytest.approx(0.1972, abs=0.001)

    # Run metadata came from the stats sidecar
    sonnet_run = conn.execute(
        "SELECT * FROM runs WHERE run_key = 'anthropic_claude-sonnet-5-verbose-cot'"
    ).fetchone()
    assert sonnet_run["enable_thinking"] == 1
    assert sonnet_run["backend"] == "vercel-gateway"
    assert sonnet_run["status"] == "complete"
    assert json.loads(sonnet_run["stats"])["accuracy"] == 0.68

    # Pre-storage-era rows: prompt_hash derived, raw_message/attempts NULL
    sample = conn.execute("SELECT prompt_hash, raw_message, n_attempts FROM results LIMIT 1").fetchone()
    assert sample["prompt_hash"] is not None and len(sample["prompt_hash"]) == 64
    assert sample["raw_message"] is None and sample["n_attempts"] is None


class FlakyLockConn:
    """Duck-typed sqlite3.Connection whose executescript raises 'database is locked' the
    first N times. Models the real failure: on deadlock-prone lock upgrades SQLite returns
    SQLITE_BUSY *without* invoking the busy handler, so busy_timeout never gets a say."""

    def __init__(self, real: sqlite3.Connection, failures: int, message: str = "database is locked"):
        self._real = real
        self._failures_left = failures
        self._message = message
        self.executescript_calls = 0

    def executescript(self, sql):
        self.executescript_calls += 1
        if self._failures_left > 0:
            self._failures_left -= 1
            raise sqlite3.OperationalError(self._message)
        return self._real.executescript(sql)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_ensure_schema_retries_locked_ddl(tmp_path, monkeypatch):
    """Concurrent-startup race (observed 2026-07-12: 2 of 3 simultaneous resumes died in
    ensure_schema): locked DDL must be retried, not raised out of storage.connect()."""
    monkeypatch.setattr(time, "sleep", lambda seconds: None)  # keep retries instant
    real = sqlite3.connect(str(tmp_path / "flaky.sqlite3"))
    real.row_factory = sqlite3.Row
    flaky = FlakyLockConn(real, failures=2)

    storage.ensure_schema(flaky)

    assert flaky.executescript_calls == 3
    tables = {row["name"] for row in real.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"schema_meta", "tasks", "runs", "results", "attempts"} <= tables
    real.close()


def test_ensure_schema_does_not_retry_other_errors(tmp_path):
    """Only lock contention is retryable; real errors (corruption, bad SQL) surface at once."""
    real = sqlite3.connect(str(tmp_path / "broken.sqlite3"))
    flaky = FlakyLockConn(real, failures=100, message="no such table: nonsense")

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        storage.ensure_schema(flaky)

    assert flaky.executescript_calls == 1
    real.close()


def test_connect_outlasts_concurrent_writer(tmp_path, monkeypatch):
    """End-to-end startup race: a sibling holds the write lock on a fresh DB past
    busy_timeout while we create the schema. connect() must wait it out and succeed."""
    monkeypatch.setattr(storage, "BUSY_TIMEOUT_MS", 100)
    db_path = tmp_path / "race.sqlite3"
    # check_same_thread=False: the release Timer commits from another thread
    holder = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    holder.execute("BEGIN IMMEDIATE")

    releaser = threading.Timer(0.5, holder.commit)
    releaser.start()
    try:
        connection = storage.connect(db_path)
        tables = {row["name"] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"schema_meta", "tasks", "runs", "results", "attempts"} <= tables
        connection.close()
    finally:
        releaser.join()
        holder.close()


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


def _minimal_row(task_id: str) -> dict:
    return {
        "task_id": task_id, "task_type": "synthetic", "task_category": "Synthetic",
        "question": "q", "correct_answer": "a", "answer_type": "single",
        "inference": {"response": "FINAL ANSWER: a", "extracted": "a",
                      "extraction_successful": True, "is_correct": True, "error_type": "correct"},
    }


def test_connect_sets_generous_busy_timeout(conn):
    # Concurrent runs share one DB; even with per-row hook commits, ingest/final-sync
    # batches can hold the writer slot longer than sqlite3's 5s default.
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000


def test_concurrent_writers_do_not_error(tmp_path):
    """Two connections upserting into the same DB (the multi-run scenario) both succeed."""
    db_path = tmp_path / "shared.sqlite3"
    storage.connect(db_path).close()  # create schema before threads race on DDL
    errors = []

    def writer(name: str) -> None:
        connection = storage.connect(db_path)
        try:
            run_id = storage.get_or_create_run(connection, name, {"model": name})
            for row_index in range(25):
                storage.upsert_result(connection, run_id, _minimal_row(f"{name}_{row_index}"))
            connection.commit()
        except Exception as exc:  # noqa: BLE001 - collected and asserted below
            errors.append(exc)
        finally:
            connection.close()

    threads = [threading.Thread(target=writer, args=(f"run{worker}",)) for worker in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    check = storage.connect(db_path)
    assert check.execute("SELECT COUNT(*) AS n FROM results").fetchone()["n"] == 50
    check.close()


def test_record_hook_survives_locked_db(tmp_path, capsys):
    """A locked DB must never kill the (paid) inference run: the hook warns and continues."""
    import run_benchmark

    db_path = tmp_path / "locked.sqlite3"
    hook_conn = storage.connect(db_path)
    run_id = storage.get_or_create_run(hook_conn, "lock-test", {"model": "m"})
    hook_conn.execute("PRAGMA busy_timeout = 100")  # don't wait 30s in the test

    hook = run_benchmark.make_db_record_hook(hook_conn, run_id)

    blocker = storage.connect(db_path)
    blocker.execute("BEGIN IMMEDIATE")  # hold the write lock like a sibling run would
    try:
        hook(_minimal_row("blocked_task"))  # must not raise
    finally:
        blocker.rollback()
        blocker.close()
    stderr = capsys.readouterr().err
    assert "WARNING" in stderr and "blocked_task" in stderr
    # The failed upsert must not leave a dangling write transaction holding the lock.
    assert not hook_conn.in_transaction

    # Connection stays usable once the lock clears; later rows still land (and commit).
    hook(_minimal_row("later_task"))
    rows = hook_conn.execute("SELECT task_id FROM results").fetchall()
    assert [row["task_id"] for row in rows] == ["later_task"]
    hook_conn.close()


def test_record_hook_releases_writer_slot_per_row(tmp_path):
    """The WAL-starvation regression (observed 2026-07-12): the hook used to batch
    commits, leaving its write transaction open across the minutes between task
    completions and starving every sibling process. After each recorded row, another
    connection must be able to take the writer slot IMMEDIATELY (busy_timeout=0)."""
    import run_benchmark

    db_path = tmp_path / "shared.sqlite3"
    hook_conn = storage.connect(db_path)
    run_id = storage.get_or_create_run(hook_conn, "starve-test", {"model": "m"})
    hook = run_benchmark.make_db_record_hook(hook_conn, run_id)

    sibling = sqlite3.connect(str(db_path))
    sibling.execute("PRAGMA busy_timeout = 0")  # any held writer slot fails instantly
    try:
        for row_index in range(3):
            hook(_minimal_row(f"row_{row_index}"))
            assert not hook_conn.in_transaction, "hook must commit per row"
            # Would raise 'database is locked' under the old batched-commit design
            sibling.execute("BEGIN IMMEDIATE")
            sibling.rollback()
    finally:
        sibling.close()
        hook_conn.close()


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
