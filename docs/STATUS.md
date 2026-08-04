# ChessQA project status

Last reviewed: 2026-08-04

This is the concise handoff for work spanning `chessqa-benchmark` and its sibling
`chess-benchmark-showcase`. Verify git status and recent commits at the start of every
session; this document describes project state, not ownership of uncommitted changes.

## Current programme state

- The CSSLab benchmark fork is installed as a top-level Python project with dataset
  generators, a Vercel AI Gateway-only inference runner, deterministic subsampling,
  answer scoring, retry/rate-limit handling, canonical JSONL results, and a derived
  SQLite results index.
- The 50-task frontier-model smoke campaign is complete. Its canonical evidence and
  final results are in
  `model-trials/2026-07-12-smoke-campaign.md`. Those runs should not be repeated unless a
  new, explicitly scoped experiment requires it.
- The smoke data has been exported into a separate static Next.js explorer in
  `../chess-benchmark-showcase`. The v1 explorer is implemented and its generated data
  is committed there.
- The first mechanical analysis feature has shipped: single-move legality is checked by
  `eval/export_web.py` and surfaced as a distinct outcome in the explorer.
- The broader Phase 3 programme—calculation-line extraction and verification,
  calibrated failure-mode judging, and cross-category cascade analysis—has not yet been
  implemented in the current codebase.
- Full 3,500-task fleet runs are not documented as completed. The measured smoke costs
  invalidated the original x1..x6 planning bands, and the fleet tier re-cut still
  requires Aron's decision before paid full runs.
- Manual chess-expert validation of the 3,500 benchmark tasks is now an approved future
  slice. A known skewer example demonstrates that deterministic Motifs labels can diverge
  from the chess concept stated in the prompt. The proposed local review workflow and
  annotation requirements are recorded in `benchmark-quality-validation.md`; no review UI
  or expert-label dataset exists yet.

## Repository boundary

### `chessqa-benchmark`

Owns benchmark JSONL, model-facing prompts, inference transport, answer extraction and
scoring, canonical run artifacts, SQLite ingestion, chess-aware export logic, and
research/campaign documentation.

Key code:

- `eval/run_benchmark.py`: inference, streaming, retries, extraction, scoring, and run
  orchestration.
- `eval/storage.py`: rebuildable SQLite index and run provenance.
- `eval/export_web.py`: static export contract and answer-to-render primitives.
- `dataset/`: benchmark generation; changes here can affect comparability with upstream.

### `chess-benchmark-showcase`

Owns the static presentation layer: home heatmap, category and model pages, chess-board
overlays, trace dialogs, navigation, responsive behavior, and the committed
`public/data/` snapshot. It must consume exporter primitives rather than infer chess
semantics independently.

## Decisions that remain in force

- Build on and prominently attribute CSSLab's MIT-licensed ChessQA work.
- Use Vercel AI Gateway as the sole inference transport in the current harness.
- Preserve full provider responses privately in canonical artifacts while exporting only
  the approved public fields.
- Treat JSONL as canonical and SQLite as derived/rebuildable.
- Treat trace fidelity per result and never present a summary as raw chain of thought.
- Present the 50-task sample as a behavior browser using counts, not as a ranked model
  evaluation.
- Keep the explorer fully static; exporter-side chess parsing is the trust boundary.
- Preserve exact model-facing behavior for reproduction runs. Any intentional deviation
  needs documentation and tests.

## Open decisions, not an automatic backlog

These items need explicit prioritization or design review before implementation:

1. Re-cut the budget and funded full-run fleets using measured smoke costs.
2. Decide which original-paper model/configuration constitutes the formal reproduction
   target and what numerical tolerance counts as reproduced.
3. Design the Phase 3 trace parser and mechanical line-verification schema on a small,
   representative trace set.
4. Define Aron's hand-label calibration sample and the acceptance metric for an
   LLM-as-judge failure taxonomy.
5. Decide when the 50-task static explorer should evolve toward full-run-scale data
   sharding, search, or server-backed access.
6. Revisit time-sensitive provider claims—including Gemini trace semantics, model
   availability, and pricing—when they become relevant to a new paid campaign.
7. Design and implement the expert benchmark-quality review slice described in
   `benchmark-quality-validation.md`, beginning with Motifs and preserving the original
   upstream-compatible benchmark separately from any later corrected version.

## Safe start for the next slice

1. Confirm which repository owns the requested behavior.
2. Read `AGENTS.md`, this file, and the task-relevant source document from `README.md`.
3. Inspect the current implementation and tests; do not copy commands from a dated plan
   without reconciling renamed files and the sibling-repository split.
4. For experimental changes, state what remains comparable to upstream and what changes
   model-facing behavior.
5. For cross-repository changes, test the producer contract first, regenerate data only
   if the slice calls for it, then test the consumer.
6. Do not run paid inference, deploy, commit, push, or open a PR unless that side effect
   is explicitly part of the request.
