"""SQLite results store for the ChessQA harness.

Design contract (agreed 2026-07-06): the per-run results JSONL remains the canonical,
crash-safe, append-only record — this SQLite database is a *derived, rebuildable index*
over it. Never write anything here that is not also in a JSONL row; conversely, the whole
database can be reconstructed at any time with:

    python eval/storage.py ingest results/*.jsonl --dataset-root benchmark

Why SQLite: a single portable file (stdlib, zero deps) that makes cross-run analysis a
SQL query today (or a Datasette UI with zero code), and cleanly migrates to Postgres/Turso
when the public web explorer gets built. Schema notes:

- ``tasks``: the benchmark itself, keyed by task_id. Results reference tasks instead of
  duplicating them; the rendered prompt is NOT stored here or in results (it is
  deterministic from task + variant flags) — ``results.prompt_hash`` detects renderer drift.
- ``runs``: one row per *logical run* (= one results filename / ``run_key``), which may
  span multiple resumed invocations. Carries full provenance: model, backend, flags, the
  exact reasoning payload sent, git commit, dataset hash, CLI args, and the final stats.
- ``results``: one row per (run, task), upsert-idempotent — re-ingesting is always safe.
  First-class columns for what gets aggregated constantly (tokens, reasoning tokens,
  cost); everything else (full usage dict, raw provider message with typed
  reasoning_details + signatures, provider metadata) is retained as JSON.
- ``attempts``: per-HTTP-attempt records (from eval/throttle.py classification) — makes
  retry pressure and billing discrepancies answerable with a join instead of a dashboard
  archaeology session.
"""

import argparse
import contextlib
import glob
import hashlib
import json
import random
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

DB_SCHEMA_VERSION = "1"
DEFAULT_DB_NAME = "chessqa.sqlite3"
# Concurrent runs share one DB and the runner batches commits (--save-interval), so a
# sibling process can hold the write lock far longer than sqlite3's 5s default.
BUSY_TIMEOUT_MS = 30_000
# busy_timeout is not enough for concurrent *startup*: on deadlock-prone lock upgrades
# (the rollback->WAL journal switch on a fresh DB, WAL snapshot conflicts) SQLite returns
# SQLITE_BUSY without invoking the busy handler at all (observed 2026-07-12: 2 of 3
# simultaneous resumes died in ensure_schema). Setup is idempotent, so retry it instead.
STARTUP_LOCK_ATTEMPTS = 10

# Filename-suffix tokens appended by the runner (see _build_variant_suffix / -thinking),
# in the order they appear in a stem; parsed back off right-to-left by run_meta_from_filename.
_SUFFIX_FLAGS = [
    ("-fmt2", ("format_example_group", 2)),
    ("-piecearr", ("add_context", True)),
    ("-thinking", ("enable_thinking", True)),
]

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS tasks (
  task_id         TEXT PRIMARY KEY,
  task_type       TEXT NOT NULL,
  task_category   TEXT NOT NULL,
  input           TEXT,
  question        TEXT NOT NULL,
  format_examples TEXT,
  correct_answer  TEXT NOT NULL,
  answer_type     TEXT NOT NULL,
  metadata        TEXT,
  source_file     TEXT,
  dataset_hash    TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_type     ON tasks(task_type);
CREATE INDEX IF NOT EXISTS idx_tasks_category ON tasks(task_category);

CREATE TABLE IF NOT EXISTS runs (
  run_id               INTEGER PRIMARY KEY,
  run_key              TEXT NOT NULL UNIQUE,
  model                TEXT NOT NULL,
  backend              TEXT,
  enable_thinking      INTEGER NOT NULL DEFAULT 0,
  add_context          INTEGER NOT NULL DEFAULT 0,
  format_example_group INTEGER NOT NULL DEFAULT 1,
  max_tokens           INTEGER,
  reasoning_config     TEXT,
  dataset_hash         TEXT,
  git_commit           TEXT,
  cli_args             TEXT,
  started_at           TEXT,
  updated_at           TEXT,
  status               TEXT NOT NULL DEFAULT 'in_progress',
  stats                TEXT
);

CREATE TABLE IF NOT EXISTS results (
  result_id             INTEGER PRIMARY KEY,
  run_id                INTEGER NOT NULL REFERENCES runs(run_id),
  task_id               TEXT NOT NULL REFERENCES tasks(task_id),
  prompt_hash           TEXT,
  response              TEXT,
  thinking_content      TEXT,
  thinking_source       TEXT,
  extracted             TEXT,
  extraction_successful INTEGER,
  is_correct            INTEGER,
  error_type            TEXT,
  prompt_tokens         INTEGER,
  completion_tokens     INTEGER,
  total_tokens          INTEGER,
  reasoning_tokens      INTEGER,
  cached_tokens         INTEGER,
  cost_usd              REAL,
  usage_json            TEXT,
  raw_message           TEXT,
  provider_meta         TEXT,
  n_attempts            INTEGER,
  latency_ms            INTEGER,
  created_at            TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  UNIQUE (run_id, task_id)
);
CREATE INDEX IF NOT EXISTS idx_results_task        ON results(task_id);
CREATE INDEX IF NOT EXISTS idx_results_run_correct ON results(run_id, is_correct);
CREATE INDEX IF NOT EXISTS idx_results_run_error   ON results(run_id, error_type);

CREATE TABLE IF NOT EXISTS attempts (
  attempt_id    INTEGER PRIMARY KEY,
  run_id        INTEGER NOT NULL,
  task_id       TEXT NOT NULL,
  attempt_no    INTEGER NOT NULL,
  started_at    TEXT,
  duration_ms   INTEGER,
  http_status   INTEGER,
  error_class   TEXT,
  billed_risk   INTEGER NOT NULL DEFAULT 0,
  retry_after_s REAL,
  wait_s        REAL,
  outcome       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attempts_run_task ON attempts(run_id, task_id);
"""


def _utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _retry_while_locked(fn, conn: sqlite3.Connection, attempts: int = STARTUP_LOCK_ATTEMPTS):
    """Run fn(), retrying with jittered sleeps on 'database is locked/busy' errors.

    Only for idempotent startup work (pragmas, CREATE IF NOT EXISTS DDL). Any other
    OperationalError, or lock contention that outlasts every attempt, is re-raised.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if ("locked" not in message and "busy" not in message) or attempt == attempts - 1:
                raise
            with contextlib.suppress(sqlite3.Error):
                conn.rollback()
            time.sleep(random.uniform(0.05, 0.3) * (attempt + 1))


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open (creating if needed) the results database with WAL mode and the v1 schema."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row

    def configure() -> None:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = NORMAL")

    try:
        _retry_while_locked(configure, conn)
        ensure_schema(conn)
    except Exception:
        conn.close()
        raise
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    def apply() -> None:
        conn.executescript(SCHEMA_DDL)
        conn.execute(
            "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (DB_SCHEMA_VERSION,),
        )
        conn.commit()

    _retry_while_locked(apply, conn)


def compute_dataset_hash(dataset_root: Path | str) -> str:
    """Hash the benchmark snapshot: sha256 over sorted (filename, bytes) of all JSONL files."""
    hasher = hashlib.sha256()
    for path in sorted(Path(dataset_root).glob("*.jsonl")):
        hasher.update(path.name.encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def git_commit_or_none(repo_root: Path | str | None = None) -> str | None:
    """Best-effort `git rev-parse HEAD` for run provenance; None outside a repo."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=str(repo_root) if repo_root else None, timeout=10,
        ).stdout.strip()
    except Exception:
        return None


def upsert_tasks(
    conn: sqlite3.Connection,
    tasks: list[dict],
    dataset_hash: str | None = None,
    source_file: str | None = None,
) -> int:
    """Idempotently upsert benchmark task rows (keyed by task_id). Returns the count written."""
    written = 0
    for task in tasks:
        conn.execute(
            """INSERT INTO tasks (task_id, task_type, task_category, input, question,
                                  format_examples, correct_answer, answer_type, metadata,
                                  source_file, dataset_hash)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(task_id) DO UPDATE SET
                 task_type=excluded.task_type, task_category=excluded.task_category,
                 input=excluded.input, question=excluded.question,
                 format_examples=excluded.format_examples, correct_answer=excluded.correct_answer,
                 answer_type=excluded.answer_type, metadata=excluded.metadata,
                 source_file=COALESCE(excluded.source_file, tasks.source_file),
                 dataset_hash=COALESCE(excluded.dataset_hash, tasks.dataset_hash)""",
            (
                task["task_id"], task.get("task_type", "unknown"), task.get("task_category", "unknown"),
                task.get("input"), task.get("question", ""),
                json.dumps(task.get("format_examples"), ensure_ascii=False),
                task.get("correct_answer", ""), task.get("answer_type", "single"),
                json.dumps(task.get("metadata"), ensure_ascii=False),
                source_file, dataset_hash,
            ),
        )
        written += 1
    conn.commit()
    return written


def get_or_create_run(conn: sqlite3.Connection, run_key: str, run_meta: dict) -> int:
    """Find or create the run row for a results-file stem; refresh mutable provenance fields.

    A "run" is the logical evaluation identified by its results filename — resumed
    invocations days apart update the same row (updated_at / cli_args / status).
    """
    now = _utcnow()
    conn.execute(
        """INSERT INTO runs (run_key, model, backend, enable_thinking, add_context,
                             format_example_group, max_tokens, reasoning_config, dataset_hash,
                             git_commit, cli_args, started_at, updated_at, status)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(run_key) DO UPDATE SET
             updated_at=excluded.updated_at,
             cli_args=COALESCE(excluded.cli_args, runs.cli_args),
             git_commit=COALESCE(excluded.git_commit, runs.git_commit),
             dataset_hash=COALESCE(excluded.dataset_hash, runs.dataset_hash),
             reasoning_config=COALESCE(excluded.reasoning_config, runs.reasoning_config),
             max_tokens=COALESCE(excluded.max_tokens, runs.max_tokens),
             status='in_progress'""",
        (
            run_key,
            run_meta.get("model", run_key),
            run_meta.get("backend"),
            int(bool(run_meta.get("enable_thinking"))),
            int(bool(run_meta.get("add_context"))),
            int(run_meta.get("format_example_group") or 1),
            run_meta.get("max_tokens"),
            json.dumps(run_meta["reasoning_config"]) if run_meta.get("reasoning_config") else None,
            run_meta.get("dataset_hash"),
            run_meta.get("git_commit"),
            json.dumps(run_meta.get("cli_args"), default=str) if run_meta.get("cli_args") else None,
            now, now, "in_progress",
        ),
    )
    conn.commit()
    row = conn.execute("SELECT run_id FROM runs WHERE run_key = ?", (run_key,)).fetchone()
    return int(row["run_id"])


def upsert_result(conn: sqlite3.Connection, run_id: int, jsonl_row: dict) -> None:
    """Upsert one results-JSONL row (the single ingest contract for live runs AND backfill).

    Derives first-class columns from the row; tolerates pre-storage-era rows that lack
    raw_message/attempts/latency (NULLs). Also upserts the row's embedded task fields so
    the results FK is always satisfiable even when ingesting without a dataset root.
    """
    inference = jsonl_row.get("inference", {}) or {}
    usage = inference.get("usage", {}) or {}
    completion_details = usage.get("completion_tokens_details", {}) or {}
    prompt_details = usage.get("prompt_tokens_details", {}) or {}
    prompt_text = inference.get("prompt")
    attempts = inference.get("attempts")

    # Guarantee the task row exists (rows embed all task fields).
    upsert_tasks(conn, [jsonl_row], dataset_hash=None, source_file=None)

    conn.execute(
        """INSERT INTO results (run_id, task_id, prompt_hash, response, thinking_content,
                                thinking_source, extracted, extraction_successful, is_correct,
                                error_type, prompt_tokens, completion_tokens, total_tokens,
                                reasoning_tokens, cached_tokens, cost_usd, usage_json,
                                raw_message, provider_meta, n_attempts, latency_ms)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(run_id, task_id) DO UPDATE SET
             prompt_hash=excluded.prompt_hash, response=excluded.response,
             thinking_content=excluded.thinking_content, thinking_source=excluded.thinking_source,
             extracted=excluded.extracted, extraction_successful=excluded.extraction_successful,
             is_correct=excluded.is_correct, error_type=excluded.error_type,
             prompt_tokens=excluded.prompt_tokens, completion_tokens=excluded.completion_tokens,
             total_tokens=excluded.total_tokens, reasoning_tokens=excluded.reasoning_tokens,
             cached_tokens=excluded.cached_tokens, cost_usd=excluded.cost_usd,
             usage_json=excluded.usage_json, raw_message=excluded.raw_message,
             provider_meta=excluded.provider_meta, n_attempts=excluded.n_attempts,
             latency_ms=excluded.latency_ms""",
        (
            run_id,
            jsonl_row["task_id"],
            hashlib.sha256(prompt_text.encode()).hexdigest() if prompt_text else None,
            inference.get("response"),
            inference.get("thinking_content"),
            inference.get("thinking_source"),
            inference.get("extracted"),
            _as_int(inference.get("extraction_successful")),
            _as_int(inference.get("is_correct")),
            inference.get("error_type"),
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
            completion_details.get("reasoning_tokens"),
            prompt_details.get("cached_tokens"),
            usage.get("cost"),
            json.dumps(usage, ensure_ascii=False) if usage else None,
            json.dumps(inference.get("raw_message"), ensure_ascii=False) if inference.get("raw_message") else None,
            json.dumps(inference.get("provider_meta"), ensure_ascii=False) if inference.get("provider_meta") else None,
            len(attempts) if attempts is not None else None,
            inference.get("latency_ms"),
        ),
    )
    if attempts:
        record_attempts(conn, run_id, jsonl_row["task_id"], attempts)


def _as_int(value) -> int | None:
    return None if value is None else int(bool(value))


def record_attempts(conn: sqlite3.Connection, run_id: int, task_id: str, attempts: list[dict]) -> None:
    """Replace the attempt records for one (run, task) — idempotent under re-ingest."""
    conn.execute("DELETE FROM attempts WHERE run_id = ? AND task_id = ?", (run_id, task_id))
    conn.executemany(
        """INSERT INTO attempts (run_id, task_id, attempt_no, started_at, duration_ms,
                                 http_status, error_class, billed_risk, retry_after_s, wait_s, outcome)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                run_id, task_id, attempt.get("attempt_no", position + 1), attempt.get("started_at"),
                attempt.get("duration_ms"), attempt.get("http_status"), attempt.get("error_class"),
                int(bool(attempt.get("billed_risk"))), attempt.get("retry_after_s"),
                attempt.get("wait_s"), attempt.get("outcome", "unknown"),
            )
            for position, attempt in enumerate(attempts)
        ],
    )


def finalize_run(conn: sqlite3.Connection, run_id: int, stats: dict | None, status: str = "complete") -> None:
    conn.execute(
        "UPDATE runs SET stats = COALESCE(?, stats), status = ?, updated_at = ? WHERE run_id = ?",
        (json.dumps(stats, ensure_ascii=False) if stats else None, status, _utcnow(), run_id),
    )
    conn.commit()


def run_meta_from_filename(jsonl_path: Path | str) -> dict:
    """Reconstruct run flags from a results filename stem (fallback when no stats sidecar).

    Strips the known suffix tokens right-to-left; what remains is the model-safe name
    (slashes/colons already replaced by underscores — the original model string is not
    perfectly recoverable, so prefer the stats sidecar's metadata when present).
    """
    stem = Path(jsonl_path).stem
    meta = {"backend": "vercel-gateway", "enable_thinking": False, "add_context": False, "format_example_group": 1}
    for suffix, (key, value) in _SUFFIX_FLAGS:
        if stem.endswith(suffix):
            meta[key] = value
            stem = stem[: -len(suffix)]
    meta["model"] = stem
    return meta


def ingest_jsonl(conn: sqlite3.Connection, jsonl_path: Path | str, run_meta: dict | None = None) -> tuple[int, int]:
    """Ingest one results JSONL (and its _stats.json sidecar if present) into the database.

    Idempotent: re-ingesting the same file leaves identical row counts. Returns
    (run_id, rows_ingested).
    """
    jsonl_path = Path(jsonl_path)
    run_key = jsonl_path.stem

    meta = run_meta_from_filename(jsonl_path)
    stats = None
    stats_path = jsonl_path.with_name(f"{run_key}_stats.json")
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        sidecar_meta = stats.get("metadata", {}) or {}
        meta["model"] = stats.get("model", meta["model"])
        for key in ("backend", "add_context", "format_example_group", "enable_thinking"):
            if key in sidecar_meta:
                meta[key] = sidecar_meta[key]
    if run_meta:
        meta.update(run_meta)

    run_id = get_or_create_run(conn, run_key, meta)
    rows = 0
    with open(jsonl_path, encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if line.strip():
                upsert_result(conn, run_id, json.loads(line))
                rows += 1
    finalize_run(conn, run_id, stats, status="complete" if stats else "in_progress")
    return run_id, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="ChessQA results database maintenance")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest", help="(Re)build the database from results JSONL files")
    ingest.add_argument("jsonl", nargs="+", help="Results JSONL paths (globs ok)")
    ingest.add_argument("--db", type=Path, default=None, help="Database path (default: <first jsonl dir>/chessqa.sqlite3)")
    ingest.add_argument("--dataset-root", type=Path, default=None, help="Benchmark dir: loads all tasks + dataset hash")
    args = parser.parse_args()

    paths = []
    for pattern in args.jsonl:
        matches = glob.glob(pattern)
        paths.extend(Path(match) for match in (matches or [pattern]))
    paths = [path for path in paths if path.suffix == ".jsonl"]
    if not paths:
        raise SystemExit("No .jsonl files matched.")

    db_path = args.db or (paths[0].parent / DEFAULT_DB_NAME)
    conn = connect(db_path)

    if args.dataset_root:
        dataset_hash = compute_dataset_hash(args.dataset_root)
        total_tasks = 0
        for task_file in sorted(Path(args.dataset_root).glob("*.jsonl")):
            tasks = [json.loads(line) for line in task_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            total_tasks += upsert_tasks(conn, tasks, dataset_hash=dataset_hash, source_file=task_file.name)
        print(f"tasks: upserted {total_tasks} from {args.dataset_root} (hash {dataset_hash[:12]}...)")

    for path in paths:
        run_id, rows = ingest_jsonl(conn, path)
        print(f"ingested {path.name}: run_id={run_id}, rows={rows}")
    conn.commit()

    run_count = conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"]
    result_count = conn.execute("SELECT COUNT(*) AS n FROM results").fetchone()["n"]
    print(f"database {db_path}: {run_count} runs, {result_count} results")


if __name__ == "__main__":
    sys.exit(main())
