"""Export the canonical smoke runs from the SQLite results DB to static JSON for the web explorer.

Produces <out-dir>/{index.json, categories/<slug>.json, traces/<task_id>.json} per
docs/superpowers/specs/2026-07-13-chessqa-explorer-design.md (rev 2); the default out-dir
is the sibling showcase repo's public/data. Read-only on the DB; raw provider payloads
(raw_message etc.) are never exported.
"""

import sqlite3
import sys
from pathlib import Path

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
