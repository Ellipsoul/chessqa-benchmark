# Documentation map

This directory contains three different kinds of material: current operating guidance,
durable research decisions, and dated evidence from completed work. They are all useful,
but they should not be read as one undifferentiated backlog.

For a new session, read `../AGENTS.md`, then `STATUS.md`. Open the founding brief for
substantive evaluation work, and load dated reports only when the current slice needs
their evidence.

## Current orientation

| Document | Lifecycle | Use it for |
|---|---|---|
| [`../AGENTS.md`](../AGENTS.md) | active | Repository rules, architecture invariants, safety boundaries, and development workflow |
| [`STATUS.md`](STATUS.md) | active handoff | Current capabilities, unresolved decisions, and the cross-repository starting point |
| [`PROJECT_BRIEF_chess_llm_benchmark.md`](PROJECT_BRIEF_chess_llm_benchmark.md) | active strategic charter | Research motivation, four-phase programme, teaching mandate, and known methodological risks |
| [`benchmark-quality-validation.md`](benchmark-quality-validation.md) | proposed future slice | Expert review interface and durable annotation requirements for validating all 3,500 benchmark tasks before treating their generated labels as chess ground truth |
| [`model-fleet.md`](model-fleet.md) | active model reference with a pending decision | Calling quirks and measured fleet evidence; its pre-smoke tier tables are historical proposals until Aron approves a re-cut |
| [`../README.md`](../README.md) | active public/contributor guide | Installation, upstream deviations, dataset construction, and inference usage |

`CLAUDE.md` remains the Claude Code entry point. `AGENTS.md` is the shared,
provider-neutral entry point used by Codex and other agents. When shared facts change,
prefer updating the neutral docs and keep any duplicated Claude-specific fact aligned.

## Research context and primary references

| Document | Lifecycle | Notes |
|---|---|---|
| [`ChessQA_Paper.pdf`](ChessQA_Paper.pdf) | primary source | CSSLab paper. Use this, not a secondary summary, for claims about the original benchmark. |
| [`Chess_as_a_Benchmark_for_AGI.md`](Chess_as_a_Benchmark_for_AGI.md) | research thesis | Aron's original hypotheses and observations. Treat proposed failure modes as hypotheses until the benchmark validates them. |
| [`cost-baseline-2026-07-06.md`](cost-baseline-2026-07-06.md) | historical baseline | Early cost model and billing investigation. Prices and pre-smoke projections are dated. |

## Experimental evidence

Files under `model-trials/` are append-style lab records. They retain observations,
failed assumptions, and incident history because that provenance matters.

| Document | Lifecycle | Notes |
|---|---|---|
| [`model-trials/2026-07-10-fidelity-and-cost.md`](model-trials/2026-07-10-fidelity-and-cost.md) | completed experiment | Provider trace-fidelity probes and first cost trials. |
| [`model-trials/2026-07-12-smoke-campaign.md`](model-trials/2026-07-12-smoke-campaign.md) | completed campaign / canonical smoke evidence | Final 50-task smoke results plus SQLite, streaming, retry, and duration-ceiling incident history. The final-results section supersedes earlier checkpoints in the same file. |

When correcting an experimental report, append a dated correction or clearly mark the
superseding section. Do not erase the sequence of observations that led to a fix.

## Design records and completed plans

| Document | Lifecycle | Notes |
|---|---|---|
| [`superpowers/specs/2026-07-13-chessqa-explorer-design.md`](superpowers/specs/2026-07-13-chessqa-explorer-design.md) | approved design record; v1 implemented | Durable product/data-contract decisions for the explorer. The sibling repository contains the implementation. |
| [`superpowers/plans/2026-07-13-explorer-exporter.md`](superpowers/plans/2026-07-13-explorer-exporter.md) | completed implementation plan | Historical TDD sequence for `eval/export_web.py`; code snippets and paths may predate the final layout. |
| [`superpowers/plans/2026-07-13-explorer-web.md`](superpowers/plans/2026-07-13-explorer-web.md) | completed implementation plan | Historical web build plan. Its leading repo-layout update supersedes later `web/` paths in the original task text. |
| [`superpowers/plans/2026-07-06-repo-cleanup.md`](superpowers/plans/2026-07-06-repo-cleanup.md) | completed implementation plan | Repository hygiene work; references the former runner filename. |
| [`superpowers/plans/2026-07-06-vercel-gateway-backend.md`](superpowers/plans/2026-07-06-vercel-gateway-backend.md) | completed and later superseded plan | Initial dual-backend migration. The current runner is Vercel AI Gateway-only and is named `eval/run_benchmark.py`. |

Do not execute a dated plan as a live checklist without reconciling it against the
current code and `STATUS.md`.

## Cross-repository ownership

| Concern | Source of truth |
|---|---|
| Dataset construction, prompts, inference, scoring | this repository |
| Canonical result records | `results/*.jsonl` in this repository |
| Cross-run query index | derived `results/chessqa.sqlite3` in this repository |
| Export schema and chess-answer parsing | `eval/export_web.py` and its tests in this repository |
| Explorer UI and static loading behavior | `../chess-benchmark-showcase` |
| Public generated snapshot | `../chess-benchmark-showcase/public/data/` |

## Maintaining the handoff

At the end of a substantive slice:

1. update `STATUS.md` if current capabilities, decisions, or next steps changed;
2. add new durable documents to this index with a lifecycle label;
3. record verification commands and any unverified assumptions;
4. keep historical plans and experiment logs intact; and
5. make cross-repository contract changes explicit in both repositories.
