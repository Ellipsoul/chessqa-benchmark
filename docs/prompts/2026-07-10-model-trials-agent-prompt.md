# Agent Prompt: Model Enumeration, Shortlist, and Trial Runs

> Copy everything below the line into a fresh agent session in this repo. It is
> self-contained: all context an agent needs is inline or in referenced repo files.

---

You are working in `chessqa-benchmark` (see `CLAUDE.md` for full project context). The goal
of this session: **enumerate available models on Vercel AI Gateway, propose a benchmark
fleet shortlist, measure each shortlisted model's thinking-trace fidelity, and execute a
one-task trial run per model** — producing the fidelity/cost table that decides which models
the full ChessQA benchmark runs (and the Phase 3 reasoning-trace analysis) will use.

## Established facts you must not re-derive (probe-verified 2026-07-06..10)

- The harness (`eval/run_openrouter.py`) defaults to the Vercel AI Gateway backend; the key
  is in the repo-root `.env` (`AI_GATEWAY_API_KEY`), loaded automatically.
- `--enable-thinking` sends `reasoning: {"effort": "medium"}`. This yields FULL-TEXT traces
  (`thinking_source: full_text`) for classic-thinking Anthropic models (verified:
  claude-haiku-4.5) and for models the gateway maps effort onto.
- **Claude 5-family models (e.g. `anthropic/claude-sonnet-5`) return NO thinking text via
  any transport** — proven Anthropic-side (streaming the gateway's native endpoint returns
  only signature deltas for sonnet-5, while the identical path streams full thinking text
  for haiku-4.5). Do not burn spend re-testing sonnet-5 thinking; mark it
  `encrypted_only`/unusable for trace analysis unless Anthropic changes behavior.
- Each result row records `thinking_source` (full_text / summary / untyped / plain /
  encrypted_only / none) — this per-response fidelity tag is the selection criterion for
  Phase 3 models.
- Cost model: `docs/cost-baseline-2026-07-06.md`. Full run ≈ 0.8M input tokens + [2.9M
  concise .. 17.7M verbose] output tokens; output dominates. Extrapolate per-model full-run
  cost from trial-task tokens using that doc's method (weighted by task-type counts).
- Retry/rate-limit behavior is already polite (shared limiter, Retry-After). Defaults are
  fine; do not raise `--workers` or `--rps`.
- All runs land in `results/*.jsonl` (canonical) + `results/chessqa.sqlite3` (queryable).

## Budget and conduct

- Total spend cap for this session: **$10**. The plan below should cost $2–5.
- Never run the full benchmark. Never kill an in-flight paid run without asking Aron.
- Ship work via the established branch → PR → merge workflow (`gh` is authenticated).

## Step 1 — Enumerate models

`GET https://ai-gateway.vercel.sh/v1/models` (Bearer `AI_GATEWAY_API_KEY` from `.env`).
Save the full list to `docs/model-trials/available-models-<date>.json` and summarize
counts by provider prefix.

## Step 2 — Propose the shortlist (STOP for approval)

Select ~12–18 candidates against these criteria, then **present the shortlist to Aron with
a one-line rationale each and the estimated trial cost, and wait for approval before any
paid calls**:

1. Frontier coverage (2026): the strongest available from Anthropic, OpenAI, Google, xAI,
   DeepSeek, Qwen, Meta — prefer dot-versioned latest slugs from the actual model list.
2. Paper-roster continuity: successors of the paper's 15 models (GPT-5 lineage, Claude,
   Gemini, DeepSeek R1 lineage, Qwen, Llama) so the frontier update is comparable.
3. Thinking mix: include both reasoning-capable and non-thinking models; note any model
   family known to summarize/redact CoT.
4. Price spread: at least two cheap models (Haiku-class) for methodology iterations.
5. Aron's picks: anthropic/claude-haiku-4.5 (baseline anchor, already measured) and
   anthropic/claude-sonnet-5 (verbose baseline exists at
   results/anthropic_claude-sonnet-5-verbose-cot.jsonl — skip re-running it).

## Step 3 — Fidelity probes (cents each)

For every approved reasoning-capable model:
`python eval/probe_reasoning.py --model <slug>` — record which payload shape (if any)
yields `reasoning_tokens > 0` and the resulting `thinking_source`. If a model needs a
different payload shape than `{"effort": "medium"}`, note it; `build_reasoning_payload`
in `eval/run_openrouter.py` is where a per-model mapping would go.

## Step 4 — Trial runs (1 task per model)

For each approved model (thinking flag per Step 3 outcome):

```bash
python eval/run_openrouter.py --dataset-root benchmark --output-dir results \
  --model <slug> --N-samples-per-task 1 --max-tasks 1 --workers 1 \
  [--enable-thinking --max-tokens 32768]
```

`--N-samples-per-task 1 --max-tasks 1` is deterministic — every model gets the *same*
single task, so latency/tokens/cost are comparable. Watch for the runner's zero-trace
warning on thinking runs.

## Step 5 — Deliverable

A committed report `docs/model-trials/<date>-fidelity-and-cost.md` containing one table:
model | thinking payload used | thinking_source | reasoning_tokens | completion tokens |
latency | trial cost | extrapolated full-run cost (per the cost-baseline method) |
recommend for full benchmark? | recommend for Phase 3 trace analysis? (requires
full_text). Include SQL used against `results/chessqa.sqlite3` so numbers are auditable,
plus a short "surprises" section. Present the recommended fleet + total projected cost to
Aron — he makes the final call and owns any funding decision.
