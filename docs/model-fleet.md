# ChessQA model fleet (canonical)

The models the full ChessQA benchmark runs on, split into a **budget tier** (light enough
to full-run now) and a **funded tier** (deferred until funding is secured). Derived from
the 2026-07-10 fidelity/cost trials — every number below is reproducible from
`docs/model-trials/2026-07-10-fidelity-and-cost.md` (probe outputs, trial rows, and the
SQL against `results/chessqa.sqlite3`). Update this file whenever the fleet changes;
treat it as the single source of truth for *which* models we run and *how each must be
called*.

Status: **proposed 2026-07-10, awaiting Aron's sign-off.** Nothing beyond one-task trials
has been run.

## Cost model (how every estimate below was computed)

```
full_run_cost($) = 0.8M x P_in + 3,500 x trial_task_completion_tokens x k x P_out
```

- `P_in` / `P_out`: gateway list prices in $/M tokens (2026-07-10 snapshot of
  `GET https://ai-gateway.vercel.sh/v1/models`).
- `0.8M`: full-run input tokens — fixed and precisely known, because prompts are
  deterministic (`docs/cost-baseline-2026-07-06.md`). Input is ~5% of any bill.
- `trial_task_completion_tokens`: the model's measured output on `motifs_battery_0042`,
  the deterministic first task of the seed-42 one-task sample. Same task for every model,
  so these are directly comparable.
- `k`: the honesty term. One task cannot reveal a model's *output profile*. The two
  measured 50-task anchors bracket it: for the concise haiku-4.5 profile the trial task
  is dead-representative (k = 828/858 ≈ **0.97**); for the verbose sonnet-5 profile it
  under-represents the weighted full-run mean ~6x (k = 5,057/850 ≈ **5.95**), because
  verbose models blow up on the 100-task-weight Structural types, not on Motifs. Hence
  every estimate is a **[x1 .. x6] band**. A 50-task smoke
  (`--N-samples-per-task 1`, $0.2–3.5 per model) collapses the band to ±20% — **always
  smoke an expensive model before committing to its full run.**

## Tier 1 — budget fleet (full runs now)

Selection: worst-case (x6) cost ≤ ~$350 each, while still covering 5 providers, three
`full_text` reasoners for Phase 3, one summary-class reasoner, and a non-reasoning
control. This tier alone can build and validate the entire Phase 3 pipeline and the
error-bar methodology before any frontier spend.

| model | fidelity | full-run $ (x1) | full-run $ (x6 worst case) | role |
|---|---|---|---|---|
| deepseek/deepseek-v4-pro | **full_text** | 9 | 53 | Phase 3 workhorse; best value on the board |
| minimax/minimax-m3 | **full_text** | 22 | 129 | second independent open-weight reasoner |
| alibaba/qwen3.7-max | **full_text** | 53 | 312 | open-weight frontier; paper-roster (Qwen) continuity |
| openai/gpt-5.4-mini | summary | 28 | 163 | OpenAI coverage; summary-class judge calibration |
| xai/grok-4.3 | summary | 11 | 60 | xAI coverage (stand-in while grok-4.5 is down) |
| meta/llama-4-maverick | none | 3 | 18 | non-reasoning control; paper-roster (Llama) continuity |
| **Tier 1 total** | | **~$126** | **~$735** | |

Near-free companion runs worth bundling: claude-haiku-4.5 **non-thinking** (~$15,
measured profile — completes the thinking/non-thinking contrast cheaply) and the
50-task smokes for Tier 2 (~$10–25 total across all eight).

## Tier 2 — funded fleet (deferred until funding)

The frontier-update headline claims (paper Phase 2) come from this tier.

| model | fidelity | full-run $ (x1) | full-run $ (x6 worst case) | role |
|---|---|---|---|---|
| anthropic/claude-sonnet-5 | summary | 93 | 549 | Anthropic frontier |
| anthropic/claude-opus-4.6 | **full_text** | 90 | 518 | strongest Anthropic verbatim CoT — Phase 3 anchor |
| anthropic/claude-haiku-4.5 (thinking) | **full_text** | 144 | 860 | cheap-tier thinking run (verbose when thinking: 8.2K tokens on trial) |
| openai/gpt-5.6-sol | summary | 66 | 376 | OpenAI frontier |
| google/gemini-3.5-flash | **full_text** | 101 | 601 | Google mid-tier, full_text |
| google/gemini-3.1-pro-preview | **full_text** | 331 | 1,977 | Google frontier — widest band, smoke first |
| moonshotai/kimi-k2.6 | **full_text** | 91 | 544 | open-weight frontier reasoner |
| deepseek/deepseek-r1 | **full_text** | 119 | 708 | the paper's actual model — reproduction continuity |
| **Tier 2 total** | | **~$1,035** | **~$6,133** | |

Optional add-ons, in priority order (not in either tier's total):
openai/gpt-5.1-thinking ($74–442, direct paper GPT-5-thinking lineage),
anthropic/claude-opus-4.8 ($101–583, Anthropic score ceiling),
anthropic/claude-fable-5 ($146–838, Mythos-tier ceiling),
zai/glm-5.2 ($196–1,173, third open lab; verbose, 306 s/task — wall-clock risk),
xai/grok-4.5 (unpriced — was 503 upstream throughout trials; re-probe, then replace
grok-4.3 if healthy).

**Both tiers, x1 sum ≈ $1,160; x6 sum ≈ $6,870** — matching the fleet projection in the
trials report.

## Per-model calling quirks (persistent — read before running anything)

The runner handles all of this automatically; this section documents *what* it does and
the response shapes downstream code must expect.

**Anthropic, trailing version >= 4.7 (claude-sonnet-5, claude-opus-4.7/4.8, claude-fable-5,
all Claude 5s):**
- Adaptive-thinking interface only. Classic `thinking.type: enabled` is rejected;
  every chat-completions `reasoning` shape silently no-ops (zero reasoning tokens).
- The runner auto-routes `--enable-thinking` runs to the gateway's Anthropic-native
  `/v1/messages` with `{"type": "adaptive", "display": "summarized"}`
  (`anthropic_adaptive_thinking()` in `eval/run_openrouter.py`, boundary probe-verified
  2026-07-10). `display: "full"` → 400; raw CoT is unavailable, period.
- Response shape: native content blocks (`thinking` + `text`), normalized by
  `parse_native_anthropic_message`. `thinking_source: summary`.
- **No per-call `usage.cost` on the native endpoint** — stats cost columns read $0;
  compute spend as tokens x list price. No separate `reasoning_tokens` count either
  (thinking is inside `output_tokens`).
- Adaptive models may skip thinking entirely on easy prompts (`thinking_source: none`
  on that row is expected behavior, not a bug).

**Anthropic, <= 4.6 (claude-opus-4.6, claude-sonnet-4.6, claude-haiku-4.5 and older):**
- Classic thinking on plain chat completions; `reasoning: {"effort": "medium"}` yields
  verbatim CoT (`reasoning_details` type `reasoning.text`, `thinking_source: full_text`).
- Re-probe on every new Anthropic dot release — the adaptive boundary moved from 5 to
  4.7 once already.

**OpenAI GPT-5.x:**
- `reasoning: {"effort"}` or `{"enabled"}` works; a budget-style `{"max_tokens": N}`
  payload silently disables reasoning (0 reasoning tokens) — do not use.
- Returns `reasoning.summary` + `reasoning.encrypted` details; **summary emission is
  length-dependent** — gpt-5.6-sol returned encrypted-only on short probe answers and a
  summary on the longer trial task. Treat `thinking_source` per-result, never per-model.
- Quirk: the slug `openai/gpt-5.1-thinking` produces results files named
  `openai_gpt-5.1-thinking-thinking.jsonl` (model name + `-thinking` run suffix).

**Google Gemini 3.x:**
- `full_text` via `reasoning.text` (+ `reasoning.encrypted` signatures) on every payload
  shape. Caveat: Google's docs describe returned "thoughts" as summaries of internal
  reasoning; ours read as detailed step-by-step CoT and tag `full_text` mechanically —
  verify against current Google docs before Phase 3 leans on Gemini verbatim claims.
- Verbose profile: gemini-3.1-pro spent 7.8K completion tokens on the trial task.

**xAI Grok:**
- Always-on reasoner: returns `reasoning` + `reasoning.summary`/`reasoning.encrypted`
  details even with *no* reasoning payload sent (format `xai-responses-v1`).
- Availability is flaky: grok-4.5 was 503 upstream (no gateway fallbacks) for an entire
  day. `results/xai_grok-4.5-thinking.jsonl` holds one ERROR row on purpose — resume
  retries it once the provider recovers.

**Open-weight reasoners (deepseek-r1/v4-pro, qwen3.7-max, kimi-k2.6, glm-5.2,
minimax-m3):**
- All return plain `reasoning.text` verbatim CoT (`full_text`) with the default
  `reasoning: {"effort": "medium"}` payload. No per-model handling needed.
- glm-5.2: 12.7K tokens / 306 s on one task — needs high `--workers` for wall-clock
  sanity and is the most expensive open model. kimi-k2.6 also runs heavy (4.7K reasoning
  tokens, 84 s).

**meta/llama-4-maverick (non-reasoning control):**
- No reasoning payload; run without `--enable-thinking` (default `--max-tokens 8192`).
- Weak format compliance: failed the trial with `format_error` (no `FINAL ANSWER:` line).
  Expect elevated `format_error` rates — that is signal, not harness breakage.

**Every model:** per-call cost comes from `usage.cost` on the chat-completions path only;
results filenames encode variant suffixes (`-thinking`, `-piecearr`, `-fmt2`,
`-openrouter`) and resume/`--eval-only` require identical flags to find the file.
