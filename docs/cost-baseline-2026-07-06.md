# ChessQA Cost Baseline (2026-07-06)

First empirical cost model for full-benchmark runs, derived from two 50-task smoke tests
against the Vercel AI Gateway. Written to inform run budgeting and an eventual funding ask.

## Smoke-test setup

Both runs used the deterministic `--N-samples-per-task 1` sample: exactly one task from each
of the benchmark's 50 task types (the full benchmark is 3,500 tasks: 30 types × 100 tasks +
20 Short Tactics theme types × 25).

| | Run 1 | Run 2 |
|---|---|---|
| Model | `anthropic/claude-haiku-4.5` | `anthropic/claude-sonnet-5` |
| Flags | defaults (`--max-tokens 8192`) | `--enable-thinking --max-tokens 32768` |
| Workers | 16 | 16 |
| Wall clock | 161 s | 1,216 s |
| Accuracy | 7/50 (14%) | 34/50 (68%) |
| Format-following | 98% | 100% |
| Output tokens (total / median / max per task) | 37.6K / 691 / 1,250 | 346K / 3,491 / 29,601 |
| Console-reported cost | $0.197 | $3.483 |
| Dashboard-billed cost | $0.197 (exact match) | **$5.69 (+63% — see Discrepancy)** |

**Critical caveat on Run 2:** despite `--enable-thinking`, extended thinking never engaged —
all 50 responses have `thinking_source: none` and `reasoning_tokens: 0`. The gateway appears
to ignore OpenRouter-style `reasoning: {"effort": ...}` for Anthropic models (its docs map a
token *budget*, `reasoning.max_tokens`, to Anthropic's thinking budget). Sonnet-5 instead
produced long *visible* chain-of-thought answers. Run 2 therefore measures **"Sonnet-5,
verbose visible CoT, no extended thinking"** — a valid verbosity/cost profile, but not a
thinking run. The results file `anthropic_claude-sonnet-5-thinking.jsonl` should be treated
as this verbose baseline (archive before any corrected rerun; resume would otherwise skip
all 50 tasks).

## Cost model

$$\text{Cost}(P_{in}, P_{out}) \approx In \cdot P_{in} + Out \cdot P_{out}$$

where $P$ are prices in $/M tokens and $In$/$Out$ are full-run token volumes in M.
**Output tokens are ~95% of the bill**; input is nearly a rounding error.

### Input side (fixed, known precisely)

Prompts are deterministic, so $In$ is computable without an API call: rendering all 3,500
prompts through `format_prompt` (default variant: no context injection, format group 1) and
calibrating characters→tokens against billed `prompt_tokens` gives:

- **$In$ ≈ 0.70M tokens** (Haiku tokenizer accounting) to **0.89M** (Sonnet accounting).
- Chess notation is token-dense: ~2.5 chars/token vs ~4 for prose.
- At $1–3/M this is **$1–3 per full run**. Negligible.
- `--add-context` (piece arrangement + legal moves injected per task) would raise input
  substantially (roughly 3–5× — not yet measured; measure before running the piecearr variant).

### Output side (the whole game; a model-behavior variable)

Per-task-type completion tokens from each smoke run, weighted by the type's true count:

| Response profile | $Out$ (full run) | Evidence |
|---|---|---|
| Concise (Haiku 4.5) | **~2.9M tokens** | measured |
| Verbose visible CoT (Sonnet-5, no thinking) | **~17.7M tokens** | measured |
| True extended thinking | **unknown; ≥ verbose is plausible** | pending reasoning-parameter fix + 10-task probe (~$1–2) |
| Hard ceiling (every task exhausts the paper's 32K budget) | 112M tokens | arithmetic bound, unrealistic |

### Full-run estimates at current Anthropic prices

| Run | Formula | Estimate |
|---|---|---|
| Haiku 4.5 ($1/$5), concise | 0.70×1 + 2.85×5 | **≈ $15** |
| Sonnet 5 ($3/$15), verbose, no thinking | 0.89×3 + 17.7×15 | **≈ $268** |
| Sonnet 5 thinking | — | **$270–$700 plausible band; measure via probe before committing** |
| Generic model | 0.8·P_in + Out·P_out | ≈ 3·P_out (concise) … 18·P_out (verbose) |

Sanity anchors: naive 70× scaling of smoke costs gives $13.8 / $244; the weighted numbers
run slightly higher because token-hungry Structural types carry 100-task weights. Rate-limit
retries (429s) are not billed.

### Paper-scale extrapolation (funding context)

The paper ran 23 model configurations. At our measured profiles, a comparable 2026 fleet:

- 23 runs × concise-model profile: ~$350 — trivial.
- 23 runs × Sonnet-class verbose/thinking profile at $15/M output: **$6,000–16,000**.
- A realistic mixed fleet (a few frontier thinking models, several mid/concise) with one
  `--add-context` variant each: **plausibly $3,000–8,000**, dominated by 2–4 expensive
  thinking models.

Implication: model selection and per-model output-profile probes (50-task smoke ≈ $0.20–$3.50
each) should precede any full-fleet commitment; a funding request needs the probe data, not
this document's band.

## Billing discrepancy (Sonnet run): console $3.48 vs dashboard $5.69

The console sums `usage.cost` over *recorded* responses — i.e., only the final successful
attempt per task. The retry loop treats **any** exception as retriable and logs nothing, so
an attempt that fails *after the provider has generated (and billed) the response* — a
connection reset mid-body, a gateway 5xx/timeout after upstream completion — is silently
retried and billed twice, visible only on the dashboard.

Evidence this is the mechanism:

- Haiku (max generation 1,250 tokens; requests last seconds): console = dashboard **exactly**.
- Sonnet (19 of 50 responses >5K tokens, up to 29.6K; multi-minute *non-streaming* HTTP
  requests): +63% dashboard premium. Long-lived idle connections are precisely what proxies
  and gateways kill.
- The $2.21 delta ≈ 145K output tokens ≈ a handful of the 18–30K-token monsters generated twice.

Verification (Vercel dashboard → AI Gateway → Logs, model = claude-sonnet-5): count
status-200 requests — if the mechanism is right there will be **more than 50** successful,
billed completions, and their summed output tokens will exceed our recorded 346K by ~145K.

Harness fixes planned (see next section): per-attempt logging so retries are visible in our
own data, and streaming to keep long connections alive.

## Client politeness: findings and plan

Observed on both runs: bursts of 429s absorbed by blind exponential backoff (162 dashboard
requests for Haiku's 50 tasks = 50 successes + ~112 throttled attempts). Current client
weaknesses: ignores `Retry-After`; 16 independent processes with no shared rate-limit state
(thundering herd on window reopen); retries permanent errors (a 400 would be retried 10×);
zero visibility into failed attempts; non-streaming long requests (the double-billing vector).

**Does parallelism make sense at all?** Yes, but throughput is capped by the provider's
tokens-per-minute limit, not by our worker count — concurrency beyond (TPM ÷ per-request
token rate) only converts capacity into 429 churn. With multi-minute generations, some
concurrency is essential for wall-clock sanity (sequential thinking runs would take hours);
the right posture is *modest concurrency + adaptive politeness*, not more workers.

Planned improvements, in priority order:

1. **Record `attempts` + per-attempt error summaries** in each result and log retries to
   stderr (also resolves discrepancy blindness).
2. **Honor `Retry-After`** on 429/503 (fall back to jittered exponential backoff).
3. **Fail fast on non-retriable status codes** (400/401/403/404/422); retry only
   408/429/5xx/connection errors.
4. **Lower default workers** for thinking-profile runs (≤8) and document TPM reasoning.
5. **Streaming responses** (`stream=True` + SSE assembly) so long generations keep the
   connection alive — directly attacks both the premature-disconnect double-billing and
   gateway visibility gaps. Larger change; do after 1–4.

## Reproduction notes

- Smoke command: `python eval/run_openrouter.py --dataset-root benchmark --output-dir results
  --model <model> --N-samples-per-task 1 --workers 16 [--enable-thinking --max-tokens 32768]`
- Input-side computation and weighted projections: render prompts via `format_prompt`,
  calibrate tokens/char on billed `prompt_tokens`, weight per-type smoke usage by true type
  counts (30×100 + 20×25).
- All figures from single samples per task type (n=1): treat per-type numbers as ±large;
  aggregate totals are steadier but still ~±20%.
