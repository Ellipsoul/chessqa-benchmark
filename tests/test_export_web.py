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
