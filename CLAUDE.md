# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

This repo starts from a clone of **CSSLab's ChessQA benchmark** (University of Toronto; arXiv:2510.23948, MIT license, upstream: https://github.com/CSSLab/chessqa-benchmark) — a 3,500-item benchmark evaluating LLM chess understanding across five categories of ascending abstraction: Structural, Motifs, Short Tactics, Position Judgment, Semantic.

The project built on top of it is defined in `docs/PROJECT_BRIEF_chess_llm_benchmark.md` — **read it before doing substantive work**. In short, four phases:

1. **Reproduce** the paper's results on 1–2 original models (original harness first, apples-to-apples). Early empirical check: which providers return *full* reasoning traces via OpenRouter vs. summaries — this gates Phase 3 depth.
2. **Update** with current frontier models, with proper uncertainty quantification.
3. **Extend (the novel contribution):** reasoning-trace root-cause analysis — mechanically verify calculated lines in thinking traces with python-chess (% illegal lines, break depth, phantom pieces), plus an LLM-as-judge failure-mode taxonomy calibrated against Aron's hand labels. Then cross-category failure cascade analysis.
4. **Publish** a reproducible harness, results, and writeup, with prominent attribution to CSSLab.

Other docs: `docs/ChessQA_Paper.pdf` (the paper) and `docs/Chess_as_a_Benchmark_for_AGI.md` (Aron's original thesis — source of the failure-mode hypotheses: theory-boundary decay, board-orientation confusion, impossible-geometry hallucinations, check-chasing).

## Working with Aron

- SRE at Google, ~5 years industry experience; deeply familiar with Claude Code/MCP/agent tooling; strong in Python and TS/Next.js. **New to formal LLM evaluation** — that gap is the point of the project.
- **FIDE Master and national champion: his chess labels are authoritative ground truth.** Design validation workflows around this (e.g., hand-labeled calibration subsets for LLM judges).
- **Teaching is a first-class deliverable.** The brief defines learning checkpoints L1–L8 (benchmark anatomy, reproduction discipline, LLM-as-judge design, error bars, contamination, CoT faithfulness, harness architecture, publishing). Pause at these moments and teach concretely, tied to the code at hand. Explain non-obvious design decisions before making them; when a decision is a genuine trade-off, present it and let Aron make the call.
- Communication: direct, technically specific, evidence-grounded, no filler encouragement. Honest about what's novel vs. derivative. Scope expansion welcome when justified.
- Refactoring policy: replacing components — even the whole harness (Inspect AI) or gateway (Vercel AI Gateway vs. OpenRouter, trace fidelity being the deciding criterion) — is on the table *after* reproduction on the original harness. Refactor *with* him, not *for* him.

## Reproduction reference points (from the paper)

15 models / 23 runs via OpenRouter, default sampling, thinking at medium effort with 32K token budget (8,192 for non-thinking). Headlines: GPT-5-thinking best overall at 79.3%; only 4 runs exceed 50%; thinking adds +14.7pp on average (pairwise); Short Tactics is the hardest category (mean 17.4%); Position Judgment is 5-way classification (random = 20%) and stays ≤40% even for top models; Structural is easiest (GPT-5* at 97%). Answer evaluation is exact match (set match for multi-answer). Adding piece-arrangement context (`--add-context`) significantly improves scores — the paper's evidence that board-state hallucination is a core bottleneck.

## Layout vs. README (important)

The README documents the upstream layout (`code/dataset`, `code/eval`, `code/plot`). In this repo the scripts live at top-level `dataset/` and `eval/`; the `code/plot` and `eval/browse_results.py` scripts referenced by README/setup.sh do not exist here. Because of the move, **default paths inside the scripts resolve via `script_dir.parent.parent` to outside the repo**, so always pass explicit paths:

- `eval/run_openrouter.py`: pass `--dataset-root benchmark --output-dir results`
- API keys: the runner loads the repo-root `.env` at startup (`cp .env.example .env`; real env vars take precedence). Default backend (Vercel AI Gateway) uses `AI_GATEWAY_API_KEY` (fallback `VERCEL_OIDC_TOKEN`). `--backend openrouter` uses `OPENROUTER_API_KEY`, falling back to the legacy `../keys/api_keys.json` (sibling of the repo root) expecting `{"openrouter_api_key": "..."}`.

## Commands

```bash
pip install -r requirements.txt          # or ./setup.sh

# Run inference against the checked-in benchmark (default backend: Vercel AI Gateway)
AI_GATEWAY_API_KEY=... python eval/run_openrouter.py --dataset-root benchmark \
  --model anthropic/claude-sonnet-4.5 --output-dir results --workers 256

# Useful flags: --backend {vercel-gateway,openrouter} (openrouter = the paper's transport,
# results get an -openrouter suffix), --max-tasks N, --N-samples-per-task N (deterministic
# uniform sample per task_type), --add-context (inject piece arrangement + legal moves),
# --enable-thinking, --use-format-example-group {1,2}, --no-resume,
# --eval-only (re-extract/re-score an existing results JSONL without API calls)

# Regenerate datasets (needs Lichess dumps under data/raw/, see README)
python dataset/01_structural.py --puzzle_path data/raw/lichess_db_puzzle.csv \
  --pgn_path data/raw/lichess_db_broadcast_2025-04.pgn --output_root data/benchmark --N_sample 100
# 02_motifs.py, 03_short_tactics.py, 04_position_judgement.py, 05_semantic.py follow the same pattern
```

There is no test suite, linter, or package build — plain Python scripts.

## Architecture

**Data flow:** raw Lichess dumps (`data/raw/`) → `dataset/0N_*.py` generators → benchmark JSONL (one file per category, checked in under `benchmark/`, 3,500 tasks total) → `eval/run_openrouter.py` → `results/<model>.jsonl` + `<model>_pretty.json` + `<model>_stats.json`.

**Task record schema** (defined as `ChessQuestionAnsweringTask` in `dataset/utils.py`): `task_id`, `task_type`, `task_category`, `input` (FEN, sometimes `"FEN | uci moves"` — strip after `|` before parsing), `question`, `format_examples` (two variants), `correct_answer`, `answer_type` (`"single"` = exact match, `"multi"` = comma-separated set comparison).

**Prompt templating:** questions contain literal `CONTEXT_PLACEHOLDER` and `FORMAT_EXAMPLE_PLACEHOLDER` strings, resolved at inference time by `format_prompt()` in `eval/run_openrouter.py`. Keep placeholders intact in the JSONL — downstream users reconstruct prompt variants from them.

**Eval runner** (`eval/run_openrouter.py`, single file; name kept from upstream): loads all `*.jsonl` from `--dataset-root`, fans out via `multiprocessing.Pool` to the selected backend's chat completions API — Vercel AI Gateway by default, OpenRouter via `--backend openrouter` (OpenRouter-only: per-call cost accounting and hardcoded provider-order pins in `call_model`; gateway runs report zero cost in stats, spend lives in the Vercel dashboard). Extracts the answer from the last `FINAL ANSWER:` line (with `\boxed{}` fallbacks) and scores with `evaluate_answer_with_error_type`. Error taxonomy: `correct`, `max_token_reached` (≥98% of max-tokens), `format_error`, `wrong_answer`, and `multi_extra_items`/`multi_missing_items`/`multi_false_items` for multi answers. Note this classifies *answers*, not *reasoning* — the Phase 3 gap. Thinking traces are saved per task as `thinking_content` plus a `thinking_source` fidelity tag (`full_text`/`summary`/`untyped`/`plain`/`encrypted_only`/`none`, from `extract_thinking`) — the gateway's typed `reasoning_details` blocks make full-trace vs. summary machine-legible, which is the Phase 3 gating signal.

**Resume behavior:** runs resume by default from the existing results JSONL, matching by `task_id`. The results filename encodes the variant — `<model with / and : replaced by _>` plus suffixes `-thinking`, `-piecearr` (from `--add-context`), `-fmt2` (from format example group 2), `-openrouter` (from `--backend openrouter`) — so flags must match the original run for resume and `--eval-only` to find the file.

**Dataset generators** (`dataset/`): numbered by category; all share `dataset/utils.py` (task dataclass, `FORMAT_EXAMPLES_*` constants, FEN/piece-arrangement helpers, seeding). `05_semantic.py` builds MCQs from commentary and needs `sentence-transformers`/`faiss`; the `05_1`–`05_3` comment filtering/cleaning/judging helpers are an offline vLLM pipeline used to produce its input (`comment_dataset.final.json`). Regenerating with post-cutoff data doubles as a contamination control.
