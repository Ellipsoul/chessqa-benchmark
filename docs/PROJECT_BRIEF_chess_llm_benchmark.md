# Project Brief: Reproducing and Extending ChessQA — A Chess Benchmark for LLM Understanding

> **Purpose of this document.** This is the founding artifact for a Claude Code session. It captures the full context of a project conceived and researched in prior conversations: the original motivation, the research landscape, the strategic pivot to building on ChessQA, immediate objectives, and — critically — a set of embedded **learning checkpoints**. The human collaborator (Aron) is using this project as a vehicle to learn LLM benchmarking, eval design, and agent harness construction from the ground up. The Claude Code agent reading this should treat *teaching* as a first-class deliverable alongside working code.

---

## 1. Who you're working with

- **Aron** — Software/Site Reliability Engineer at Google (London), ~5 years industry experience (THG, Amazon, Stripe, Google). Internal AI subject-matter expert; deeply familiar with Claude Code, multi-agent orchestration, MCP, and frontier model tooling.
- **FIDE Master and national chess champion.** His chess judgment is expert-level ground truth. When classifying model errors or validating puzzle datasets, his labels are authoritative — design workflows that leverage this (e.g., have him hand-label a validation subset that calibrates automated judges).
- **Strong engineering fundamentals** (algorithms, distributed systems, React/Next.js/TypeScript, Python) but **new to formal LLM evaluation**: benchmark construction, statistical rigor in evals, harness frameworks, LLM-as-judge methodology. That gap is the point of this project.
- **Communication preferences:** direct, technically specific, evidence-grounded. No filler encouragement. Honest assessments of what's novel vs. derivative. He explicitly welcomes scope expansion when it's justified.

---

## 2. Original motivation (Aron's thesis)

Aron has been playing text-based chess against LLMs since GPT-3, and wrote a long-form document ("Chess as a Benchmark for AGI") arguing that chess is an unusually rich, objectively-scoreable probe of general intelligence. Core claims:

**Two overarching realms of required intelligence:**

1. **Object permanence & state tracking** — maintaining the board state across moves; spatial awareness of constructions (pins, skewers, smothered mates); even understanding which direction is "forward" for each side's pawns. Frontier models (tested through GPT-5 and Gemini 2.5 Pro Thinking) still fail here.
2. **Positional reasoning** — choosing good moves given a known-correct position. The Kaggle Game Arena (Google DeepMind partnership) isolated this by supplying the exact FEN every move; models still produced illegal moves, failed checkmates, and calculation lines unbacked by chess logic.

**Key empirical observations from Aron's own testing (these are original findings, treat them as hypotheses to formalize):**

- **The theory-boundary decay effect:** a model's board-state accuracy degrades as a function of *distance (in plies) from the last position documented in opening theory* — NOT as a function of position complexity or number of exchanges. E.g., vs. GPT-5: optimal play through ~move 17 (end of theory in a mainline Nimzo-Indian), rapid quality collapse by ~move 21, rook blunder move 29, illegal move attempt move 30. This suggests retrieval/memorization masquerading as state tracking.
- **Board orientation confusion:** Gemini 2.5 Pro believed pawns could capture backwards, misattributing which pawn made a capture (c5 vs. c7 capturing on d6). Pawns are the only unidirectional pieces, making them a targeted probe for orientation understanding.
- **Impossible-geometry hallucinations:** e.g., believing two same-side bishops (necessarily on opposite colors) could both recapture on the same square.
- **Check-chasing heuristic:** models over-prioritize giving check as a proxy for progress, producing e.g. threefold repetition instead of a simple winning pawn push.
- **Self-report:** when asked, Gemini identified state tracking (not move selection) as its dominant difficulty.

**His originally proposed test categories** (kept here because they inform the extension phase):
1. Object permanence — play along theory lines of varying popularity, require full board-state output after every move, measure where tracking breaks relative to the theory boundary.
2. Spatial/directional awareness — legality judgment on move sequences seeded with illegal states; identification/exploitation of pins, skewers, smothered mates.
3. Basic reasoning — mate-in-1 consistency; multi-move calculation streams scored for legality and quality.
4. Creativity — large-scale self-play measuring deviation from the most popular opening lines (models currently follow the most-played variations with near-unanimity).
5. Adversarial/continuity extras — single-conversation games (not fresh prompts per move) so models can reflect on their own prior moves; intentionally playing illegal moves against the model to test whether it objects; attempting to continue play after checkmate.

---

## 3. Research already conducted (deep research pass, June 2026)

A comprehensive research report was produced. Key findings the agent should internalize (and verify/refresh where load-bearing, since the space moves fast):

**The landscape is crowded.** Existing work includes:
- **Kaggle Game Arena chess-text** (Google DeepMind) — model-vs-model play from supplied FENs; Elo-style ratings; tests move selection but not state tracking.
- **ChessQA** (CSSLab, University of Toronto — Ashton Anderson's lab, the Maia Chess group; arXiv:2510.23948) — the paper this project now builds on. See §4.
- **PGN2FEN** (Aidan Cooper) — PGN→FEN translation accuracy as a function of game length; the closest existing test of pure state tracking.
- **LLM Chess leaderboards** (e.g., maxim-saplin/llm_chess; dubesor) — full-game play with legality/quality stats.
- **kagisearch/llm-chess-puzzles** — Lichess puzzle solving in FEN.
- **Chess-GPT / world-model research** (Adam Karvonen; OthelloGPT lineage) — linear probes showing transformers trained on games form internal board representations.
- **DeepMind's grandmaster-level chess without search** (ChessBench) — relevant context, not a benchmark competitor.

**Genuinely novel gaps identified (Aron's differentiators — no existing benchmark covers these):**
1. State-tracking decay measured *relative to the opening-theory boundary* (requires the Lichess opening explorer API to quantify "popularity" and locate the boundary per line).
2. Adversarial illegal-move injection (does the model object?).
3. Single-conversation continuity across a full game (all existing arenas use fresh prompts per move).
4. Creativity via opening-deviation measurement under self-play.

**Framework recommendation from the research:** Inspect AI (UK AI Safety Institute) as the strongest general eval framework — solver/scorer/task architecture, native multi-turn/agentic loops, transcript logging, model-agnostic providers. Alternatives assessed: lm-evaluation-harness, OpenAI Evals, HELM, promptfoo, DeepEval, plus lightweight custom harnesses over LiteLLM / OpenRouter / Vercel AI Gateway. Chess tooling: python-chess (legality, FEN/PGN), Stockfish (centipawn loss, WDL), Lichess puzzle DB (millions of theme-tagged puzzles: `pin`, `skewer`, `smotheredMate`, `mateIn1`, ...), Lichess opening explorer (theory popularity). Known caveats flagged: training-data contamination (openings are massively memorized — Chess960 is a useful control), prompt sensitivity, variance/error bars (see Anthropic's "Adding Error Bars to Evals"), and chain-of-thought faithfulness limits (reasoning traces are evidence, not ground truth, of the model's actual computation).

---

## 4. The pivot: ChessQA as the foundation

After reading the ChessQA paper in full, Aron decided to **pivot from building a benchmark from scratch to (a) reproducing ChessQA's results, (b) updating it with frontier models, and (c) extending it with a reasoning-trace root-cause analysis layer** — the thing the paper only briefly touches on. This maximizes learning while contributing something genuinely new.

**ChessQA in one paragraph:** five task categories forming an ascending ladder of chess abstraction — **Structural** (rules/board mechanics), **Motifs** (tactical pattern recognition), **Short Tactics** (calculation), **Position Judgment** (evaluation), **Semantic** (high-level concept description, MCQ from commentary). Dynamic by design: prompts, answer keys, and construction scripts can be regenerated as source data evolves. The paper found persistent weaknesses across all five categories in contemporary LLMs and provides per-category error analyses.

**The source code exists but is obscure:** **https://github.com/CSSLab/chessqa-benchmark** (MIT license, ~3 commits, ~3 stars, at least one open issue — expect rough edges; budget a debugging session for the first run). Repo structure as verified:

- `code/dataset/` — generation scripts per category: `01_structural.py`, `02_motifs.py`, `03_short_tactics.py`, `04_position_judgement.py`, `05_semantic.py` (plus `05_2_comment_cleaning.py`, `05_3_comment_judging.py`, optional vLLM-based filtering for the Semantic pipeline).
- `code/eval/` — `run_openrouter.py`: OpenRouter-based inference runner. Already model-agnostic: `--model anthropic/claude-3.5-haiku`, `--workers 256`, resume support, token/cost tracking. Includes an `--enable-thinking` flag and an `evaluate_answer_with_error_type` function — i.e., a shallow error taxonomy already exists in code (it classifies *answers*, not *reasoning*; that's our gap to fill).
- `benchmark/` — pre-generated JSONL files per category, shipped with the repo. **Reproduction can start here without regenerating datasets.** Prompts are intentionally kept templated so downstream users can reconstruct prompting variants.
- `results/` — per-model outputs with accuracy, per-category breakdowns, and error shares.

**Raw data needed only for later regeneration:** Lichess puzzle dump (CSV), Lichess engine-evaluation database (`lichess_db_eval.jsonl.zst`), broadcast PGNs. All free downloads. Regenerating with fresh 2026 data doubles as a **contamination control** (post-cutoff broadcast games cannot be memorized).

---

## 5. Immediate objectives (in order)

### Phase 1 — Reproduce
1. Clone the repo, get it running, evaluate on 1–2 of the paper's original models, and compare against the paper's reported numbers. Reproduction within noise = validated setup. Document every rough edge fixed (this becomes contributor documentation later).
2. **Early empirical check (do this before designing anything around traces):** verify which frontier models expose *full* reasoning traces via OpenRouter vs. summaries only (OpenAI models often return summarized reasoning). Run a handful of tasks with `--enable-thinking` across providers and inspect what actually comes back. This single check determines the feasible depth of Phase 3 per provider.

### Phase 2 — Update with frontier models
3. Run the full benchmark across the current frontier set (Claude Opus 4.x / Fable-class, GPT-5.x, Gemini 3.x, Grok, DeepSeek — whatever is current at execution time). Cost is explicitly deprioritized for now; explore the limits first, dial back later.
4. Produce an updated results table/leaderboard in the paper's format, with proper uncertainty quantification (see learning checkpoint L4).

### Phase 3 — Extend: reasoning-trace root-cause analysis (the novel contribution)
5. Capture full thinking traces for all runs where available.
6. **Mechanical verification layer:** parse candidate variations out of traces and verify every calculated line for legality with python-chess. This yields hard, objective metrics: % of calculation lines containing illegal moves, depth at which lines break, phantom-piece references.
7. **Failure-mode taxonomy:** classify root causes using categories grounded in Aron's observations — board-orientation errors (backwards pawn moves), state drift, piece-value confusion, impossible-geometry hallucinations (same-color-bishop class), phantom/missing pieces, correct-plan-wrong-execution, check-chasing. Implement as LLM-as-judge, **calibrated against a validation subset hand-labeled by Aron** (FM-level ground truth is this project's unfair advantage).
8. **Cross-category correlation:** the paper's ladder hypothesis implies failures should cascade (Motif failures on positions where Structural understanding also failed). Test it: do error types in higher categories trace back to lower-category failures on the same/similar positions? This causal analysis is absent from the paper.
9. Longer-term hooks (from §2): theory-boundary decay tasks, adversarial illegal-move injection, single-conversation continuity — these can become new ChessQA-style categories once the core extension lands.

### Phase 4 — Publish
10. Public repo (fork or fresh, with attribution), reproducible harness, results, and a writeup. Target: anyone can rerun with one config change per model.

---

## 6. Refactoring policy

Aron is **fully open to replacing components — including entire frameworks** — provided the replacement (a) matches the use cases above and (b) maximizes learning. Specifically on the table:

- **Eval framework:** the repo is a bespoke script harness. Migrating to **Inspect AI** is a live option and likely a good learning vehicle (tasks/solvers/scorers, transcript viewer, built-in parallelism and provider abstraction). Decision rule: reproduce on the original harness *first* (so reproduction is apples-to-apples with the paper), then evaluate migration for Phases 2–3.
- **Model gateway:** the repo uses OpenRouter. **Vercel AI Gateway** is Aron's suggested alternative (he's a Next.js developer; it fits his stack familiarity). Evaluate on: reasoning-trace fidelity per provider, unified model naming, cost tracking, rate limits, caching behavior, and reproducibility implications of gateway routing vs. direct APIs. Trace fidelity should be the deciding criterion given Phase 3.
- **Language:** Python will likely dominate (python-chess, Stockfish bindings, the existing codebase), but TS components (dashboards, leaderboard site) are welcome and play to Aron's strengths.

When you refactor, do it *with* him, not *for* him — see the teaching mandate below.

---

## 7. Teaching mandate: learning checkpoints

Aron's stated goal: "learn, from scratch, how to set up agent harnesses, how to create benchmarks, evals." The agent should pause at these natural moments and teach — short, concrete, tied to the code in front of you. Prefer "here's the concept, here's where it bites us in *this* repo" over abstract lectures.

- **L1 — Anatomy of a benchmark (Phase 1, first repo walkthrough).** Dataset construction → prompt templating → inference runner → answer extraction → scoring → aggregation. Map each ChessQA file to this pipeline. Discuss why they ship templated prompts (prompt-variant research) and why answer extraction is often the most fragile link.
- **L2 — Reproduction discipline (Phase 1).** What counts as "reproduced"? Exact-number matching vs. within-variance matching; sources of legitimate divergence (model version drift behind an API alias, temperature, provider-side changes, answer-parser differences). Teach pinning: model snapshots, seeds where supported, logged raw responses as the ultimate artifact.
- **L3 — LLM-as-judge design (Phase 3).** Judge prompts, rubric design, position bias, self-preference bias, and why calibration against human labels (his FM labels) is non-negotiable. Cohen's kappa between judge and human as the acceptance gate.
- **L4 — Error bars on evals (Phase 2).** Anthropic's "Adding Error Bars to Evals" as the reference. Clustered standard errors when questions share source positions; how many samples per category before a model comparison is meaningful; why single-run leaderboard deltas of 1–2% are usually noise.
- **L5 — Contamination and memorization (Phases 2–3).** Chess openings are among the most memorized text on the internet. Teach the memorization-vs-reasoning distinction using his own theory-boundary observation as the motivating example; Chess960 and post-cutoff broadcast games as controls.
- **L6 — CoT faithfulness (Phase 3).** Reasoning traces are *evidence about*, not *transcripts of*, model computation. Cover the faithfulness literature at a working level, and how the mechanical legality-verification layer sidesteps this (verifying stated lines is valid regardless of whether the trace reflects the true computation).
- **L7 — Harness architecture (if/when migrating to Inspect AI).** Solvers vs. scorers vs. tasks; how multi-turn game loops are expressed; what the framework buys vs. hand-rolled scripts (logging, caching, parallelism, the transcript viewer) and what it costs (abstraction overhead, debugging through layers).
- **L8 — Benchmark publishing (Phase 4).** Versioning datasets, results submission workflows, what makes benchmarks trusted (SWE-bench/ARC-AGI as case studies), and how to write a results readme that survives scrutiny.

Beyond these, opportunistically explain any non-obvious design decision before making it, and ask Aron to make the call when a decision is a genuine trade-off — he learns most from owning the decisions.

---

## 8. Known risks and open questions

- **Trace availability** is the biggest unknown for Phase 3 — resolve empirically in Phase 1 (objective #2). Fallback: run trace-dependent analysis only on providers with raw traces; use answer-level error taxonomy elsewhere.
- **Repo rough edges:** 3 commits, minimal docs, an open issue. Expect missing pins, hardcoded paths, undocumented data prep. Fixing these is Phase 1 work and future contributor docs.
- **Cost at frontier scale:** explicitly deprioritized for now, but log per-model spend from the start (the runner tracks tokens/cost) so dialing back later is informed.
- **Model naming drift:** frontier model set should be finalized at execution time, not from this document.
- **Attribution:** building on CSSLab's MIT-licensed work — attribute prominently, and consider reaching out to the authors once reproduction succeeds; an acknowledged extension is worth more (scientifically and reputationally) than a silent fork.

---

## 9. Key resources

- ChessQA paper: https://arxiv.org/abs/2510.23948 (OpenReview: https://openreview.net/forum?id=gBz9NMbvYS)
- ChessQA code: https://github.com/CSSLab/chessqa-benchmark
- Kaggle Game Arena chess-text: https://www.kaggle.com/benchmarks/kaggle/chess-text
- PGN2FEN: https://github.com/AidanCooper/pgn2fen-benchmark
- Kagi llm-chess-puzzles: https://github.com/kagisearch/llm-chess-puzzles
- Inspect AI: https://inspect.aisi.org.uk/
- python-chess: https://python-chess.readthedocs.io/
- Lichess open data (puzzles, evals, opening explorer): https://database.lichess.org/ and https://lichess.org/api
- Anthropic, "Adding Error Bars to Evals": https://www.anthropic.com/research/statistical-approach-to-model-evals
- Aron's original document: "Chess as a Benchmark for AGI" (he can re-share the file in the Claude Code session)

---

*Document prepared July 2026 from prior Claude.ai research and planning sessions. The Claude Code agent should verify time-sensitive facts (current frontier model names, repo state, gateway capabilities) before acting on them.*
