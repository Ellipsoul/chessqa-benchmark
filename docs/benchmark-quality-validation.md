# Proposed future slice: expert validation of benchmark quality

Status: proposed work, approved in principle on 2026-08-04; not yet designed or
implemented.

## Motivation

The checked-in 3,500-task benchmark is reproducibly generated, but reproducibility does
not guarantee that every generated answer key faithfully represents the chess concept
described by its prompt. A concrete example is `motifs_skewer_0056` (Lichess puzzle
`9idgU`): the generator labels the geometric alignment `d3>e4>f5` as a skewer because the
front knight is more valuable than the back pawn, while the prompt describes the stronger
chess concept of an attack that forces the front piece to move. The detector does not test
that forcing condition or the tactical soundness of the attack.

This creates a construct-validity risk: a model can match a deterministic answer key
without demonstrating the intended chess understanding, while a model that rejects an
invalid theme can be scored as wrong. The paper and generator code do not document an
item-by-item chess-expert audit of the Motifs labels. Other categories use stronger sources
of ground truth in many places, but their task construction and labels should still be
reviewed rather than assumed correct.

Aron can provide fast, authoritative chess-expert validation if all benchmark positions and
their task-specific claims are presented in a low-friction review interface.

## Objective

Create a simple local review tool that lets Aron inspect every checked-in benchmark task,
understand what it claims to test, and record one of at least these outcomes:

- accepted: the position and answer key validly test the stated task;
- flagged: the task is incorrect, ambiguous, or does not test the stated concept; or
- unreviewed.

The primary goal is review throughput and durable expert labels, not a polished public
product.

## Minimum review experience

For each task, show enough information to validate the task without opening source files:

- a correctly oriented chessboard rendered from the task FEN, including side to move;
- task ID, category, and task type;
- the exact task question and current gold answer;
- a visual rendering of the gold answer where meaningful, such as motif arrows, move
  arrows, or highlighted squares;
- relevant provenance and metadata, including the Lichess puzzle link when available;
- one-action Accept and Flag controls, with keyboard shortcuts;
- an optional short note and problem reason for flagged tasks;
- progress counts and filters for category, task type, review state, and flagged reason;
  and
- reliable resume behaviour so a review session never loses completed labels.

Review must be task-level, even when multiple tasks share a FEN. Correctness depends on the
question and answer key, not only on whether the underlying board position is valid. The UI
may group or de-duplicate identical positions for navigation, but it must preserve a separate
decision for every task ID.

## Durable annotation requirements

- Store expert labels separately from `benchmark/*.jsonl`; reviewing must not silently
  rewrite the upstream-compatible benchmark.
- Key every annotation by stable task ID and a benchmark-content fingerprint or version so
  labels cannot be accidentally applied to regenerated or changed tasks.
- Preserve at least reviewer, timestamp, disposition, optional reason, and optional note.
- Keep the annotations in this repository as canonical research data. A UI may live here or
  reuse components from `../chess-benchmark-showcase`, but the public showcase must not
  become the sole source of truth.
- Support a simple export suitable for auditing, analysis, and later regression tests.

## Validation scope and order

The intended scope is all 3,500 checked-in tasks across Structural, Motifs, Short Tactics,
Position Judgment, and Semantic. A sensible implementation sequence is to build and prove
the workflow on Motifs first, where the known detector mismatch makes the quality risk most
immediate, then cover the remaining categories without changing the underlying annotation
contract.

This expert review is distinct from the future hand-label sample used to calibrate an
LLM-as-judge failure taxonomy. This slice validates the benchmark questions and gold labels;
the Phase 3 calibration work labels causes in model responses.

## Decisions required before implementation

1. Choose whether the first UI is a small feature in the existing showcase, a separate local
   route/app, or a minimal tool in this repository.
2. Define a compact flag-reason taxonomy after reviewing a representative set; do not force
   all quality failures into categories chosen from a single example.
3. Decide whether a second state such as `uncertain` or `needs_engine_check` is useful beyond
   the minimum accepted/flagged workflow.
4. Decide the benchmark-version/fingerprint scheme and canonical annotation file format.
5. Decide how flagged items affect published evaluation. Corrections must be versioned and
   reported separately from the original upstream benchmark rather than silently changing
   historical results.

## Non-goals for the initial slice

- Do not regenerate benchmark data or correct answer keys while building the review tool.
- Do not run paid model inference or use an LLM as a substitute for Aron's chess labels.
- Do not make the annotation interface a public production service unless later requested.
- Do not treat a mechanically legal position or move as sufficient proof that the task's
  intended chess concept is valid.

## Expected research output

The completed review should produce a versioned expert-annotation dataset, counts of accepted
and flagged tasks by category and task type, a defensible catalogue of benchmark-quality
failure modes, and an explicit decision about whether to preserve, filter, or correct each
class of problem in a future ChessQA version.
