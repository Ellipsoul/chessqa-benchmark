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
- OpenRouter evaluation runner with parallelism, resume, cost/tokens tracking, per‑category and per‑task‑type stats
- Dynamic dataset builders that regenerate as the underlying sources improve

## Repository Layout

- `dataset/`: dataset generation scripts for each category
- `eval/`: OpenRouter inference runner
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
2. **`--add-context` piece ordering** (`eval/run_openrouter.py`, `get_context`). Upstream
   injected the piece arrangement in board-scan order (a1→h8); this fork uses the same
   canonical ordering the `piece_arrangement` answers demand (White then Black,
   King/Queen/Rook/Bishop/Knight/Pawn, squares alphabetical). Any run using `--add-context`
   (`-piecearr` result files) is not byte-comparable to upstream's piecearr runs.
3. **Seeded subsampling** (`eval/run_openrouter.py`, `load_tasks`). `--N-samples-per-task`
   now shuffles with a fixed-seed RNG (upstream's comment claimed a fixed seed but used the
   unseeded global RNG). Subsampled runs are now reproducible and resume-coherent; full runs
   are unaffected.
4. **Default inference backend is Vercel AI Gateway, not OpenRouter.** The runner grew a
   `--backend {vercel-gateway,openrouter}` flag (default `vercel-gateway`); the paper's 23
   runs all went through OpenRouter, which remains selectable for apples-to-apples
   comparisons (those result files get an `-openrouter` suffix). Two knock-on effects:
   per-response dollar-cost accounting is OpenRouter-only, so cost columns in
   `*_stats.json` are zero on gateway runs (spend lives in the Vercel dashboard), and
   each result now records a `thinking_source` fidelity tag
   (`full_text`/`summary`/`encrypted_only`/…) derived from the gateway's typed
   `reasoning_details` blocks.

**Unaffected:** full-benchmark runs without `--add-context` use exactly the checked-in task
prompts, so `--backend openrouter` runs remain apples-to-apples with the paper.

**Non-behavioral deviations:**

- Layout: scripts live at top-level `dataset/` and `eval/` (upstream: `code/dataset`,
  `code/eval`); upstream's `code/plot` and `eval/browse_results.py` are not present. Because
  of the move, in-script default paths resolve outside the repo — always pass
  `--dataset-root benchmark --output-dir results` (and explicit paths to the generators).
- API keys: `AI_GATEWAY_API_KEY` env var for the default backend. For `--backend
  openrouter`: `OPENROUTER_API_KEY` env var, falling back to upstream's legacy
  `../keys/api_keys.json` (a `keys/` directory *beside* this checkout) expecting
  `{"openrouter_api_key": "..."}`.
- Tooling: pinned/capped `requirements.txt` (verified via fresh-venv install), ruff lint
  config in `pyproject.toml`, `Makefile`, `.venv`-based `setup.sh`, and a comprehensive
  documentation pass over all scripts. Stale CLI help texts for `--workers`/`--max-retries`/
  `--timeout` were corrected to match the actual defaults (256 / 10 / 6000).

## Install

- Python 3.8+
- `pip install -r requirements.txt`
- Optional for semantic MCQ embeddings: `pip install sentence-transformers faiss-cpu`

API keys
- Easiest: `cp .env.example .env` and fill in your key(s) — the eval runner loads the
  repo-root `.env` at startup (gitignored; real environment variables take precedence).
- Default backend (Vercel AI Gateway): `AI_GATEWAY_API_KEY` (create a key in the Vercel
  dashboard under AI Gateway).
- `--backend openrouter`: `OPENROUTER_API_KEY`, or upstream's legacy
  `../keys/api_keys.json` beside the checkout (see `setup.sh`).

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
AI_GATEWAY_API_KEY=... python eval/run_openrouter.py \
  --dataset-root benchmark \
  --model anthropic/claude-haiku-4.5 \
  --output-dir results --workers 256
```

Smoke test (deterministic sample of ~1 task per task type):
```bash
AI_GATEWAY_API_KEY=... python eval/run_openrouter.py \
  --dataset-root benchmark --output-dir results \
  --model anthropic/claude-haiku-4.5 --N-samples-per-task 1 --workers 16
```

Options
- Backend: `--backend {vercel-gateway,openrouter}` (default `vercel-gateway`; use
  `openrouter` for paper-comparable runs — those results get an `-openrouter` suffix)
- Limit total tasks: `--max-tasks 800`
- Uniform sampling per task type: `--N-samples-per-task 50`
- Add auto‑generated context (piece arrangement + legal moves): `--add-context`
- Enable “thinking” for models that support it: `--enable-thinking`
- Re‑evaluate an existing JSONL without calling APIs: `--eval-only`

Outputs
- `results/<model>.jsonl`: per‑task records with prompts/responses and extraction
- `results/<model>_pretty.json`: summarized JSON for quick reading
- `results/<model>_stats.json`: accuracy, per‑category breakdowns, error shares, cost and tokens

### Working with Placeholders (for Hugging Face / custom inference)

The released JSONL files intentionally keep templated prompts so that downstream users can reconstruct different prompting variants. Each task record may contain:

- `CONTEXT_PLACEHOLDER` — replaced at inference time with autogenerated context (piece arrangement + legal moves) when `--add-context` is used.
- `FORMAT_EXAMPLE_PLACEHOLDER` — replaced with a format example drawn from `format_examples`.

The helper in `eval/run_openrouter.py` demonstrates how to resolve these placeholders. Minimal example:

```python
from code.eval.run_openrouter import format_prompt, get_context
import json

with open("data/benchmark/motifs.jsonl") as fh:
    task = json.loads(next(iter(fh)))

prompt = format_prompt(
    task,
    add_context=True,            # inject piece arrangement / legal moves
    format_example_group=1       # choose example variant
)

# to reproduce OpenRouter usage:
#   1. call format_prompt for each task
#   2. send the prompt to your model/backend
#   3. evaluate responses with extract_answer/evaluate_answer_with_error_type
```

When publishing on Hugging Face, retain the JSONL files as-is and reference this workflow so users can opt into context injection or alternate formatting as needed.

## License

MIT License. See `LICENSE` for details.
