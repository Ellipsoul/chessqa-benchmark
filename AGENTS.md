# ChessQA benchmark repository guidance

This file is the provider-neutral entry point for coding agents and human contributors.
Claude-specific guidance remains in `CLAUDE.md`; do not remove it. Start with the shared
context here, then load only the task-relevant documents from `docs/README.md`.

## Start here

1. Read `docs/README.md` for the documentation map and source-of-truth rules.
2. Read `docs/STATUS.md` for the current cross-repository handoff.
3. For substantive research or evaluation work, read
   `docs/PROJECT_BRIEF_chess_llm_benchmark.md`.
4. Inspect the current code, tests, and git state before trusting a dated plan or command.

The sibling repository `../chess-benchmark-showcase` is the static public explorer. This
repository owns dataset generation, inference, scoring, canonical result JSONL, the
derived SQLite index, and the exporter. The showcase owns presentation only.

## Mission and research standards

This fork builds on CSSLab's ChessQA benchmark to:

1. reproduce selected paper results;
2. evaluate current frontier models with honest uncertainty and provenance;
3. add a reasoning-trace root-cause analysis layer; and
4. publish a reproducible harness and public behavior explorer with prominent CSSLab
   attribution.

Keep these distinctions explicit in code, documentation, and UI copy:

- A 50-task smoke sample (one item per task type) is diagnostic evidence, not a
  leaderboard-quality estimate. Prefer counts to percentages for that sample.
- `thinking_source` is a per-result fidelity label. Provider summaries, plain visible
  reasoning, and mechanically captured `full_text` are not interchangeable. A captured
  trace is evidence about model behavior, not proof of the model's internal computation.
- Answer scoring and reasoning analysis are separate layers. The current answer taxonomy
  does not by itself establish a root cause.
- Aron's chess labels are the authoritative expert ground truth for chess-specific
  qualitative claims. Automated judges should be calibrated against a hand-labeled set.
- Time-sensitive model availability, pricing, and provider behavior must be verified
  before they become load-bearing decisions.

## Data and architecture invariants

The main pipeline is:

```text
raw Lichess data
  -> dataset/ generators
  -> benchmark/*.jsonl
  -> eval/run_benchmark.py
  -> results/*.jsonl + *_stats.json
  -> results/chessqa.sqlite3
  -> eval/export_web.py
  -> ../chess-benchmark-showcase/public/data/
```

- `benchmark/*.jsonl` is the checked-in 3,500-task benchmark. Preserve the literal
  `CONTEXT_PLACEHOLDER` and `FORMAT_EXAMPLE_PLACEHOLDER` tokens.
- `results/*.jsonl` is the canonical inference record. Treat
  `results/chessqa.sqlite3` as a derived, rebuildable index, never the sole source of
  truth.
- Result filenames encode run variants (`-thinking`, `-piecearr`, `-fmt2`). Resume and
  `--eval-only` flags must match the original run.
- The exporter owns answer parsing, move-legality checks, and render primitives. The
  browser must not reimplement chess-answer interpretation.
- `public/data/` in the sibling repository is generated and deliberately committed for
  review. Do not hand-edit it. Re-export, then review the generated diff.
- Never export private provider payloads such as `raw_message`, `usage_json`, or
  `provider_meta`. Resolved prompts, answers, approved traces, and aggregate costs are
  the intended public contract.
- Changes to prompts, answer extraction, scoring, sampling, or dataset generation alter
  experimental behavior. Call them out explicitly and add regression coverage.

## Operational safety

- Do not start paid inference, reasoning probes, large dataset downloads, or deployment
  work unless the user's request explicitly authorizes it. Before a paid run, state the
  sample size, model, flags, and expected cost range.
- Never print, commit, or copy values from `.env`. The runner reads the repo-root `.env`;
  real environment variables take precedence.
- Preserve completed smoke results. `docs/model-trials/2026-07-12-smoke-campaign.md`
  states that the campaign is complete and must not be rerun casually.
- This is a fork. `origin` is `Ellipsoul/chessqa-benchmark`; `upstream` is the CSSLab
  repository. If asked to create a PR, always pass
  `--repo Ellipsoul/chessqa-benchmark` and verify the resulting URL begins with
  `https://github.com/Ellipsoul/`. Never target CSSLab without explicit instruction.
- Preserve unrelated working-tree changes. Do not commit, push, deploy, or publish unless
  the user asks for that specific side effect.

## Development workflow

Use the repository virtual environment when it exists:

```bash
make install
make test
make lint
make check
```

Focused commands:

```bash
.venv/bin/python -m pytest tests/test_export_web.py -q
.venv/bin/python eval/run_benchmark.py --help
.venv/bin/python eval/storage.py ingest results/*.jsonl --dataset-root benchmark
make export-web
```

`make export-web` writes into the sibling repository and is therefore a cross-repository
mutation. Use it only when the requested slice includes refreshing showcase data.

The top-level layout differs from upstream. Scripts live in `dataset/` and `eval/`, not
`code/dataset/` and `code/eval/`. For inference, pass explicit paths:
`--dataset-root benchmark --output-dir results`.

Prefer tests before implementation for behavior changes. Verification should be
proportional to the slice, but prompt/scoring/export-contract changes require focused
tests plus the full relevant suite.

## Documentation practice for future slices

- `docs/STATUS.md` is the concise, current handoff. Update it when a slice changes
  capabilities, open decisions, or the next safe starting point.
- `docs/README.md` is the navigation layer. Add new durable documents there and label
  their lifecycle.
- Dated files under `docs/model-trials/` are experimental evidence. Preserve their
  chronology; append a dated correction rather than silently rewriting an observation.
- Dated files under `docs/superpowers/plans/` are implementation records, not live task
  lists. They may contain obsolete paths such as `eval/run_openrouter.py` or `web/`.
- The approved explorer spec is a decision record. If the implementation deliberately
  diverges, document the new decision instead of making history look as if it always
  agreed.
- Separate verified current facts, historical facts, proposed work, and user decisions.
  Date claims that can drift.

## Working with Aron

Communicate directly and with evidence. Teaching formal evaluation practice is part of
the project: explain the non-obvious benchmark or statistical consequence of a change in
the concrete code being touched. Present genuine research trade-offs for Aron's decision
instead of silently choosing them. Do not over-explain routine engineering mechanics.
