# ChessQA: Evaluating Large Language Models for Chess Understanding

ChessQA is a comprehensive, dynamic benchmark that evaluates LLM chess understanding across five ascending levels of abstraction — from basic rules to high‑level semantic reasoning — with objective ground truth and reproducible pipelines for dataset construction, inference, and analysis.

## Abstract
Chess provides an ideal testbed for evaluating the reasoning, modeling, and abstraction capabilities of large language models (LLMs), as it has well-defined structure and objective ground truth while admitting a wide spectrum of skill levels. However, existing evaluations of LLM ability in chess are ad hoc and narrow in scope, making it difficult to accurately measure LLM chess understanding and how it varies with scale, post-training methodologies, or architecture choices. We present ChessQA, a comprehensive benchmark that assesses LLM chess understanding across five task categories (Structural, Motifs, Short Tactics, Position Judgment, and Semantic), which approximately correspond to the ascending abstractions that players master as they accumulate chess knowledge, from understanding basic rules and learning tactical motifs to correctly calculating tactics, evaluating positions, and semantically describing high-level concepts. In this
way, ChessQA captures a more comprehensive picture of chess ability and understanding, going significantly beyond the simple move quality evaluations done previously, and offers a controlled, consistent setting for diagnosis and comparison. Furthermore, ChessQA is inherently dynamic, with prompts, answer keys, and construction scripts that can evolve as models improve. Evaluating a range of contemporary LLMs, we find persistent weaknesses across all five categories and provide results and error analyses by category. We will release the code, periodically refreshed datasets, and a public leaderboard to support further research.

## What’s Inside

- Five categories with objective answer keys and robust extraction
  - Structural: piece arrangement, legal moves (piece/all), check detection and check‑in‑1, capture/control/protect squares, and state tracking (FEN after UCI sequences)
  - Motifs: pin, fork, skewer, battery, discovered check, double check
  - Short Tactics: best‑move puzzles by rating buckets (beginner→expert) and by theme (dozens of tactical themes)
  - Position Judgment: centipawn evaluation selection across bands (neutral/advantage/winning/…)
  - Semantic: multiple‑choice commentary understanding with several distractor strategies (keyword, piece+stage, semantic embedding, easy random)
- Evaluation runner (Vercel AI Gateway) with parallelism, resume, cost/tokens tracking, per‑category and per‑task‑type stats
- Dynamic dataset builders that regenerate as the underlying sources improve

## Repository Layout

- `dataset/`: dataset generation scripts for each category
- `eval/`: inference runner (Vercel AI Gateway)
- `benchmark/`: generated benchmark JSONL files (one per category)
- `results/`: per‑model outputs (`*.jsonl`, `*_pretty.json`, `*_stats.json`)
- `docs/`: project brief, the ChessQA paper, and planning notes

## Deviations from Upstream (CSSLab/chessqa-benchmark)

This repo is a fork of [CSSLab/chessqa-benchmark](https://github.com/CSSLab/chessqa-benchmark)
(arXiv:2510.23948). It deliberately deviates from upstream in the following ways — keep this
list in mind when comparing any numbers against the paper's published results.

**Behavioral fixes (change model-facing behavior):**

1. **Pin prompt bug fixed** (`dataset/02_motifs.py`). Upstream reassigned `task_description`
   instead of appending, so every `motifs_pin` question shipped *without* its definition
   sentence ("Identify all absolute pins… would expose its own king to check."). The
   checked-in `benchmark/motifs.jsonl` still contains the upstream (truncated) prompts;
   any *regenerated* motifs data will include the full sentence and is therefore not
   prompt-identical to the paper's pin tasks.
2. **`--add-context` piece ordering** (`eval/run_benchmark.py`, `get_context`). Upstream
   injected the piece arrangement in board-scan order (a1→h8); this fork uses the same
   canonical ordering the `piece_arrangement` answers demand (White then Black,
   King/Queen/Rook/Bishop/Knight/Pawn, squares alphabetical). Any run using `--add-context`
   (`-piecearr` result files) is not byte-comparable to upstream's piecearr runs.
3. **Seeded subsampling** (`eval/run_benchmark.py`, `load_tasks`). `--N-samples-per-task`
   now shuffles with a fixed-seed RNG (upstream's comment claimed a fixed seed but used the
   unseeded global RNG). Subsampled runs are now reproducible and resume-coherent; full runs
   are unaffected.
4. **Inference transport is Vercel AI Gateway, not OpenRouter.** The paper's 23 runs all
   went through OpenRouter; this fork routes everything through the Vercel AI Gateway
   (the runner was renamed `eval/run_benchmark.py`, and the OpenRouter code path was
   removed on 2026-07-16). Two knock-on effects: cost accounting comes from the gateway
   (verified live: it returns per-call `cost`/`gateway_cost`/`market_cost` in `usage`,
   so cost columns in `*_stats.json` are populated; the Vercel dashboard adds
   request-level observability), and each result records a `thinking_source` fidelity
   tag (`full_text`/`summary`/`encrypted_only`/…) derived from the gateway's typed
   `reasoning_details` blocks.

**Unaffected:** full-benchmark runs without `--add-context` use exactly the checked-in task
prompts, so results remain prompt-identical to the paper (transport aside).

**Non-behavioral deviations:**

- Layout: scripts live at top-level `dataset/` and `eval/` (upstream: `code/dataset`,
  `code/eval`); upstream's `code/plot` and `eval/browse_results.py` are not present. Because
  of the move, in-script default paths resolve outside the repo — always pass
  `--dataset-root benchmark --output-dir results` (and explicit paths to the generators).
- API keys: `AI_GATEWAY_API_KEY` env var (fallback `VERCEL_OIDC_TOKEN`), from the shell
  or the repo-root `.env`.
- Tooling: pinned/capped `requirements.txt` (verified via fresh-venv install), ruff lint
  config in `pyproject.toml`, `Makefile`, `.venv`-based `setup.sh`, and a comprehensive
  documentation pass over all scripts. Stale CLI help texts for `--workers`/`--max-retries`/
  `--timeout` were corrected to match the actual defaults (256 / 10 / 6000).

## Install

- Python 3.8+
- `pip install -r requirements.txt`
- Optional for semantic MCQ embeddings: `pip install sentence-transformers faiss-cpu`

API keys
- Easiest: `cp .env.example .env` and fill in your key — the eval runner loads the
  repo-root `.env` at startup (gitignored; real environment variables take precedence).
- Vercel AI Gateway: `AI_GATEWAY_API_KEY` (create a key in the Vercel dashboard under
  AI Gateway).

## Data

Place the following under `data/raw/` (paths match defaults in scripts):
- `lichess_db_puzzle.csv` — Lichess puzzle dump
- `lichess_db_eval.jsonl.zst` — Engine evaluations (for Position Judgment)
- `lichess_db_broadcast_2025-04.pgn` — PGN stream (for Structural state tracking)
- Optional: `chessbase.pgn` / `filtered_chessbase.pgn` — additional PGNs

## Create Benchmark Datasets

Structural
```bash
python code/dataset/01_structural.py \
  --puzzle_path data/raw/lichess_db_puzzle.csv \
  --pgn_path data/raw/lichess_db_broadcast_2025-04.pgn \
  --output_root data/benchmark --N_sample 100
```

Motifs
```bash
python code/dataset/02_motifs.py \
  --puzzle_path data/raw/lichess_db_puzzle.csv \
  --output_root data/benchmark --N_sample 100
```

Short Tactics
```bash
python code/dataset/03_short_tactics.py \
  --puzzle_path data/raw/lichess_db_puzzle.csv \
  --all_themes_path data/info/all_themes_to_include.json \
  --output_root data/benchmark --N_sample_rating 100 --N_sample_theme 25
```

Position Judgement
```bash
python code/dataset/04_position_judgement.py \
  --data_path data/raw/lichess_db_eval.jsonl.zst \
  --output_root data/benchmark --tasks_per_category 100 --max_evaluations 10000
```

Semantic (MCQ from commentary; requires sentence-transformers)
```bash
python code/dataset/05_semantic.py \
  --input data/mid/comment_dataset.final.json \
  --output_root data/benchmark --N_sample_mcq 100
```

Note: comment cleaning and judging helpers for producing `data/mid/comment_dataset.final.json` live in `code/dataset/05_2_comment_cleaning.py` and `code/dataset/05_3_comment_judging.py` (optional, uses vLLM for offline filtering).

## Run Inference

Basic run (Vercel AI Gateway, the default backend)
```bash
AI_GATEWAY_API_KEY=... python eval/run_benchmark.py \
  --dataset-root benchmark \
  --model anthropic/claude-haiku-4.5 \
  --output-dir results --workers 256
```

Smoke test (deterministic sample of ~1 task per task type):
```bash
AI_GATEWAY_API_KEY=... python eval/run_benchmark.py \
  --dataset-root benchmark --output-dir results \
  --model anthropic/claude-haiku-4.5 --N-samples-per-task 1 --workers 16
```

Options
- Rate limiting: `--rps 2.0 --burst 4` (request starts/second shared across all worker
  threads; `--workers 24` only sets in-flight concurrency — throughput is governed by rps)
- Limit total tasks: `--max-tasks 800`
- Uniform sampling per task type: `--N-samples-per-task 50` (deterministic)
- Add auto‑generated context (piece arrangement + legal moves): `--add-context`
- Enable “thinking” for models that support it: `--enable-thinking` (the runner warns if
  the model returned zero traces — e.g. Claude 5 adaptive-thinking models, whose thinking
  is currently redacted at the API level; check `thinking_source` per result)
- Re‑evaluate an existing JSONL without calling APIs: `--eval-only`
- Results database: `--db-path` (default `results/chessqa.sqlite3`), `--no-db` to skip

Outputs
- `results/<model>.jsonl`: per‑task records — prompts/responses, thinking trace +
  `thinking_source` fidelity tag, raw provider message, per-attempt retry records, usage.
  **This file is canonical.**
- `results/<model>_stats.json`: accuracy, per‑category breakdowns, error shares, cost and tokens
- `results/chessqa.sqlite3`: derived, queryable cross-run index (tasks / runs / results /
  attempts; gitignored). Rebuild anytime:
  `python eval/storage.py ingest results/*.jsonl --dataset-root benchmark`.
  Instant browsable UI: `pipx run datasette results/chessqa.sqlite3`.

### Working with Placeholders (for Hugging Face / custom inference)

The released JSONL files intentionally keep templated prompts so that downstream users can reconstruct different prompting variants. Each task record may contain:

- `CONTEXT_PLACEHOLDER` — replaced at inference time with autogenerated context (piece arrangement + legal moves) when `--add-context` is used.
- `FORMAT_EXAMPLE_PLACEHOLDER` — replaced with a format example drawn from `format_examples`.

The helper in `eval/run_benchmark.py` demonstrates how to resolve these placeholders. Minimal example:

```python
from eval.run_benchmark import format_prompt, get_context
import json

with open("data/benchmark/motifs.jsonl") as fh:
    task = json.loads(next(iter(fh)))

prompt = format_prompt(
    task,
    add_context=True,            # inject piece arrangement / legal moves
    format_example_group=1       # choose example variant
)

# to reproduce this repo's inference flow with your own backend:
#   1. call format_prompt for each task
#   2. send the prompt to your model/backend
#   3. evaluate responses with extract_answer/evaluate_answer_with_error_type
```

When publishing on Hugging Face, retain the JSONL files as-is and reference this workflow so users can opt into context injection or alternate formatting as needed.

## Model Fleet (2026 evaluation)

The canonical list of models this fork evaluates — a budget tier for immediate full runs
and a funded tier deferred until funding — lives in **[docs/model-fleet.md](docs/model-fleet.md)**,
together with per-model calling quirks (Anthropic ≥4.7 adaptive routing, OpenAI
summary-only reasoning, provider response shapes) and the full-run cost model with its
calculations. Fidelity probes and one-task trial evidence behind those numbers:
[docs/model-trials/2026-07-10-fidelity-and-cost.md](docs/model-trials/2026-07-10-fidelity-and-cost.md).

## License

MIT License. See `LICENSE` for details.
