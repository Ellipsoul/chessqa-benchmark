# 2026-07-12 fleet smoke campaign — partial results, the 340s wall, and the resume playbook

Status: **paused by design, not failure.** 7 of 14 planned 50-task smokes completed clean,
3 completed with casualties from a newly-discovered transport defect (the "340s wall"),
7 held until a streaming fix lands. Session spend ≈ **$21** against a $38 cap (~$10.7 of
it burned by the defect). Aron approved pausing and deferring the fix to a dedicated
session (2026-07-12).

**Read this before touching the smoke campaign.** Companion docs:
`docs/model-fleet.md` (fleet + measured projections), `docs/cost-baseline-2026-07-06.md`
(projection method), `docs/model-trials/2026-07-10-fidelity-and-cost.md` (fidelity, trial
rows, old [x1..x6] bands).

## Measured smoke results (n = 50 sample, seed-42, 1 task/type)

All thinking runs: `--enable-thinking --max-tokens 32768 --N-samples-per-task 1 --workers 8`.
"ok/err": completed rows vs gave_up ERROR rows (340s wall victims — see below). Accuracy is
over completed rows only; for damaged runs it is **survivor-biased upward** (the tasks that
died are the long/hard ones) and the projection is a **lower bound** (the dead types are the
token-heavy ones).

| model | ok/err | smoke $ | acc | Struct | Motifs | ShortTac | PosJudg | Semantic | 32K-capped | thinking_source | projected full-run $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xai/grok-4.3 | 50/0 | 0.28 | 56% | 8/11 | 3/6 | 12/24 | 2/5 | 3/4 | 0 | summary ×50 | **18.9** |
| meta/llama-4-maverick | 50/0 | 0.03 | 14% | 0/11 | 0/6 | 3/24 | 2/5 | 2/4 | 0 | none ×50 | **4.0** |
| openai/gpt-5.4-mini | 50/0 | 2.47 | 50% | 6/11 | 4/6 | 9/24 | 3/5 | 3/4 | 6 | summary ×50 | **128.4** |
| minimax/minimax-m3 | 50/0 | 1.05 | 40% | 8/11 | 3/6 | 3/24 | 2/5 | 4/4 | 8 | full_text ×50 | **59.6** |
| deepseek/deepseek-v4-pro | 30/**20** | 0.26 | 63%* | 9/11 | 3/5 | 4/6 | 0/4 | 3/4 | 0 | full_text ×30 | **≥21.4*** |
| alibaba/qwen3.7-max | 40/**10** | 1.30 | 82%* | 9/11 | 6/6 | 12/16 | 2/3 | 4/4 | 0 | full_text ×40 | **≥93.1*** |
| moonshotai/kimi-k2.6 | 43/**7** | 3.86 | 70%* | 11/11 | 4/6 | 11/19 | 1/3 | 3/4 | 12 | full_text ×43 | **≥272.3*** |
| anthropic/claude-haiku-4.5 (no-think, reused) | 50/0 | 0.20 | 14% | 1/11 | 1/6 | 2/24 | 1/5 | 2/4 | 0 | none ×50 | **15.1** |
| anthropic/claude-sonnet-5 (verbose-cot, reused) | 50/0 | 3.48 | 68% | 8/11 | 5/6 | 14/24 | 3/5 | 4/4 | 0 | none ×50 | **178.3** |

\* damaged runs — re-run after the streaming fix before quoting these numbers anywhere.

Projection method (validated): weight each task_type's smoke completion tokens by its true
benchmark count (30 types ×100 + Short Tactics rating/theme mix summing to 900), add the
fixed 0.8M input tokens, price at the gateway list snapshot (2026-07-12, identical to
2026-07-10 for the fleet). **Sanity anchor reproduced: haiku-4.5 non-thinking projects
$15.05 vs the $15 hand-computed baseline.** Analysis script committed at
`eval/smoke_analysis.py` (prices inline; rerun as
`.venv/bin/python eval/smoke_analysis.py results/<file>.jsonl`).

Equivalent SQL against `results/chessqa.sqlite3` (after
`python eval/storage.py ingest results/*.jsonl --dataset-root benchmark`), e.g.
thinking-source distribution per run:

```sql
SELECT r2.run_key, res.thinking_source, COUNT(*) AS n
FROM results res JOIN runs r2 ON r2.run_id = res.run_id
GROUP BY r2.run_key, res.thinking_source ORDER BY r2.run_key;
```

## Predicted vs measured — did the [x1..x6] bands work?

| model | old band (x1..x6) | measured | measured/x1 |
|---|---|---|---|
| meta/llama-4-maverick | 3 .. 18 | 4.0 | ×1.3 |
| anthropic/claude-haiku-4.5 (no-think) | ~15 anchor | 15.1 | ×1.0 |
| xai/grok-4.3 | 11 .. 60 | 18.9 | ×1.7 |
| deepseek/deepseek-v4-pro | 9 .. 53 | ≥21.4 | ×2.4+ |
| alibaba/qwen3.7-max | 53 .. 312 | ≥93.1 | ×1.8+ |
| minimax/minimax-m3 | 22 .. 129 | 59.6 | ×2.7 |
| moonshotai/kimi-k2.6 | 91 .. 544 | ≥272.3 | ×3.0+ |
| openai/gpt-5.4-mini | 28 .. 163 | 128.4 | ×4.6 |

Verdict: every measurement landed **inside** its band (the method was honest), but x1 was
systematically optimistic — the single trial task (Motifs) under-represents the verbose
tail. True multipliers run ×1.3–×4.6, worst for models that pad medium tasks
(gpt-5.4-mini median 7.3K completion tokens; 6–12 of 50 tasks hitting the 32K cap on the
three flagged models is both a cost and a truncated-reasoning red flag). Budget-tier
implication: the measured Tier-1 sum is **≥ $325** vs the $126 x1 estimate — re-cut the
tiers only after the held smokes are measured.

## Incident 1 — SQLite lock crashes (fixed, merged)

Three concurrent runner processes → `sqlite3.OperationalError: database is locked` from the
per-result record hook, killing two paid runs mid-flight (old code held the WAL write lock
across whole save batches). Fixed same day (merged via PR #16): 30s busy_timeout +
non-fatal record hook + non-fatal DB touchpoints. Residual rule: the JSONL is canonical;
if in doubt run with `--no-db` and ingest afterwards.

## Incident 2 — the 340s wall (FIXED 2026-07-12 — Slice 1 streaming transport landed)

> **Status update (2026-07-12, Slice 1):** chat-completions requests now stream (SSE,
> `stream: true` + `stream_options.include_usage`) and are reassembled into the exact
> non-streaming message shape (`consume_chat_sse` in `eval/run_openrouter.py`), so bytes
> move continuously and the gateway's idle timer never fires. Mid-stream drops after
> tokens arrived are classified `stream_drop` (billed-risk, 2-attempt cap) with partial
> counters recorded per attempt; `ttft_ms` is recorded per result; live tokens/s and
> cumulative recorded cost show in the tqdm postfix. The Anthropic-native /v1/messages
> path (adaptive models) remains non-streaming — different SSE grammar, short summarized
> runs, never hit the wall. Kill-shot note: the spec below named a structural
> state-tracking task, but deepseek's structural tasks all finished ≤190s in the smoke —
> the actual wall victims were Motifs/Short Tactics — so the verification used
> `motifs_discovered_check_0076`, a task that died at the wall 4× on
> deepseek/deepseek-v4-pro. Slice 2 (campaign resume) remains separate.

**Mechanism.** The runner sends non-streaming requests (`stream=False`,
`eval/run_openrouter.py` ~line 790). During a multi-minute generation zero bytes move on
the wire, and an edge/idle timer in the Vercel AI Gateway path kills the socket at
**exactly ~340s** (observed `duration_ms` 340010–340020 across every failure;
`http_status: null`, `error_class: connection`; client `--timeout` is 6000s, so it is not
ours). The runner classes it as a cheap-retriable connection error and retries up to 4× —
but the retry needs the same >340s of silence, so it dies at the same wall, and after 4
attempts the task lands as a gave_up ERROR row.

**Proof.** Max completable tokens = decode speed × 340s, and observed ceilings match:

| model | decode tok/s (median) | predicted ceiling | observed max ctoks | err rows | failed attempts |
|---|---|---|---|---|---|
| deepseek/deepseek-v4-pro | ~58–60 | ~20.5K | 20,718 | 20 | 94 |
| alibaba/qwen3.7-max | ~55 | ~18.6K | 17,234 | 10 | 56 |
| moonshotai/kimi-k2.6 | ~77–101 (high variance) | ~26–34K | — | 7 | 35 |
| minimax/minimax-m3 | ~102 | ~34.6K | 33,792 | 0 | 9 |
| openai/gpt-5.4-mini | ~120 | ~40K | 32,768 (cap) | 0 | 0 |
| xai/grok-4.3 | ~128 | n/a (concise) | 4,963 | 0 | 0 |

**Billing.** Killed attempts are billed upstream (the provider finishes generating; only
our socket died). 194 failed attempts ≈ **$10.7 estimated** — the gateway credits meter
(`GET https://ai-gateway.vercel.sh/v1/credits`, the only ground truth; recorded
`usage.cost` in the JSONLs structurally undercounts) moved ~$21 this session vs ~$9.5
recorded. deepseek's smoke spent more on corpses than on answers.

**Exposure of the held models** (per doomed task ≈ 4 attempts × 340s × decode × out-price):
opus-4.6 (~49 tok/s trial decode, $25/M) ≈ $1.7/doomed task; gpt-5.6-sol (~54 tok/s, $30/M)
≈ $2.2; gemini-3.1-pro ($12/M) exposed if decode < ~100 tok/s. This is why the queue is
held: without the fix, the expensive half of the fleet pays the most for the least data.

## Session spend reconciliation

| component | $ |
|---|---|
| recorded `usage.cost`, all result files (incl. ~$4.2 pre-campaign) | 9.49 |
| estimated billed 340s-killed attempts (194 × duration × decode × list) | ~10.7 |
| gateway meter delta for the session (start ≈ $8.8 → end $29.94) | **≈ 21** |

Rule for future sessions: track spend by the credits meter, not by summing `usage.cost`.

## Campaign state and resume playbook

**Done, clean (do NOT re-run):** grok-4.3, llama-4-maverick, gpt-5.4-mini, minimax-m3,
haiku-4.5 non-thinking, sonnet-5 verbose-cot.

**Done, damaged (resume AFTER streaming fix — ERROR rows are retried automatically):**
deepseek-v4-pro (20 ERROR), qwen3.7-max (10), kimi-k2.6 (7).

**Held (run AFTER streaming fix, cheapest-first):** gpt-5.6-sol, anthropic/claude-opus-4.8
(swapped in for opus-4.6, Aron 2026-07-12 — same list price, stronger model; costs the
full_text trace: 4.8 is adaptive/summary-only with no per-call cost field),
anthropic/claude-sonnet-5 (thinking), anthropic/claude-haiku-4.5 (thinking),
google/gemini-3.5-flash, deepseek/deepseek-r1, and LAST google/gemini-3.1-pro-preview.
xai/grok-4.5: still 503 upstream (re-probed 2026-07-12); grok-4.3 already smoked as the
stand-in — only revisit 4.5 if it comes back.

Exact command (thinking models — flags MUST match for resume to find the file):

```bash
.venv/bin/python eval/run_openrouter.py --dataset-root benchmark --output-dir results \
  --model <slug> --N-samples-per-task 1 --workers 8 --enable-thinking --max-tokens 32768
```

Non-thinking (llama-only in this fleet; already done): same without
`--enable-thinking --max-tokens`. Concurrency: 2–3 processes max. The SQLite fix is
merged so `--no-db` is no longer required, but stays a safe fallback. After any batch:
`python eval/storage.py ingest results/*.jsonl --dataset-root benchmark`.

Anthropic native-routed runs (sonnet-5, and any ≥4.7 model) report **no per-call cost**
— price them as tokens × list ($/M): sonnet-5 in 2.00/out 10.00, opus-4.8 5.00/25.00
(same list price as opus-4.6, verified by Aron 2026-07-12),
haiku-4.5 1.00/5.00 (full snapshot in `eval/smoke_analysis.py`).

## Follow-up plan: two separate slices (Aron, 2026-07-12)

The remaining work is deliberately split — do NOT combine them in one session:

- **Slice 1 — streaming fix only.** Implement and verify streaming (spec below). Ends at
  a merged PR with the kill-shot test passing. No campaign runs beyond the single
  cheap verification task.
- **Slice 2 — campaign resume.** Runs the "Campaign state and resume playbook" section
  above, exactly as written, on the fixed harness: recover the 37 ERROR rows in
  deepseek-v4-pro/qwen3.7-max/kimi-k2.6, then the held seven cheapest-first, then the
  measured tier re-cut in `docs/model-fleet.md` (Aron signs off on the re-cut).

**Policy on 32K-cap exhaustion (`max_token_reached`):** this is an experimental result,
not a harness defect — record and report it as a failure mode per model/category. The
original paper treats it the same way: "Max Token Reached" is one of the six outcome
classes in its Figure 4 response-evaluation breakdown (figure-only; never quantified in
prose), with thin slices concentrated in thinking models on Short Tactics and Position
Judgement. Our 2026 rates are much higher (kimi-k2.6 24%, minimax-m3 16%, gpt-5.4-mini
12% of the smoke) — today's reasoners think longer against the same 32K budget the paper
fixed, which is itself a Phase 2 finding. Streaming does not (and should not) change it.

## Slice 2 progress checkpoint (2026-07-12 evening — resume here)

Recovery + first held smokes ran on the streamed harness. Session spend ≈ **$9 by the
credits meter** (baseline total_used 29.97 → ~38.8; balance ≈ $36). Two NEW transport
findings and fixes landed mid-campaign:

- **The ~785s duration ceiling (Incident 3).** Distinct from the fixed 340s idle wall:
  the gateway kills streams at a hard ~785s total duration, and **forges a graceful
  `[DONE]`** on the way out. First seen as 4 qwen rows cut at exactly 785.1s scoring as
  silent empty `format_error`s (PR #22 detected terminatorless EOF; insufficient), then
  6 more passing as fake successes because of the forged terminator (PR #24: completion
  now requires positive evidence — finish_reason or usage). Consequence: a task is
  unrunnable if the model cannot emit its full generation inside ~785s at current
  provider throughput (qwen was decoding at ~16 tok/s that evening vs ~55 in the smoke —
  time-of-day dependent). Retries land as honest ERROR rows; resume retries them free of
  charge on the next attempt.
- **`--retry-capped` (PR #23):** resume no longer re-runs `max_token_reached` rows by
  default (upstream re-billed every cap on every resume; a deepseek resume re-ran 18
  caps alongside 1 intended task before the flag landed, and the mid-flight kill of
  that re-run is why deepseek is 9 rows short below).

State after the checkpoint (all rows verified clean of poison; DB re-ingested):

| run | rows | acc | caps | note |
|---|---|---|---|---|
| kimi-k2.6 | 50/50 | 62% | 18 | fully recovered; caps re-ran once pre-flag (+6 new caps) |
| qwen3.7-max | 44/50 | 72%* | 0 | 6 tasks scrubbed — 785s-ceiling victims, resume re-runs them |
| deepseek-v4-pro | 41/50 | 56%* | 7 | re-run killed mid-flight at day end; resume runs 9 missing |
| gemini-3.5-flash | 50/50 | **82%** | 1 | best clean score of the campaign |
| deepseek-r1 | 50/50 | **24%** | 2 | the paper's model — strikingly low on ChessQA-2026 |

\* accuracy over present rows; final number after the missing rows complete.

**Tomorrow's queue (in order):** (1) deepseek-v4-pro resume (9 tasks), (2) qwen resume
(6 tasks — if they 785s-fail twice, accept the ERROR rows and document as
ceiling-victims), (3) held smokes cheapest-first: haiku-4.5-thinking, sonnet-5,
gpt-5.6-sol, opus-4.8 (replaces 4.6 per PR #23), LAST gemini-3.1-pro-preview; optional
grok-4.5 re-probe. Parallel launches no longer need `--no-db`: the record hook now
commits one short transaction per row instead of batching (the batched design held the
single WAL writer slot across the minutes between completions, starving concurrent
starters — the root cause behind the PR #21 symptom). `--no-db` + ingest remains a safe
fallback. Then the measured tier re-cut in `docs/model-fleet.md` (Aron signs off).

## The streaming fix (spec for Slice 1)

Change `call_api` in `eval/run_openrouter.py` transport-only; prompts, flags, scoring,
filenames unchanged, so results remain comparable and resume just works.

1. Chat-completions path: send `"stream": true` + `"stream_options": {"include_usage": true}`;
   consume SSE via `resp.iter_lines()`; accumulate `delta.content`, `delta.reasoning`, and
   typed `reasoning_details` fragments; take `usage` from the final chunk; reassemble the
   exact message dict downstream code already parses (verify `extract_thinking` and cost
   capture see no difference).
2. Anthropic-native `/v1/messages` path: separate SSE grammar (`message_start`,
   `content_block_delta` with `thinking_delta`/`text_delta`, `message_delta` carrying
   usage). Phase 2 if needed — adaptive summarized runs are short and survived the wall.
3. Progress: streaming gives per-task time-to-first-token and live token counts — surface
   tokens/s and cumulative cost in the progress bar (tqdm postfix), and record
   `ttft_ms` per result. This is the "even more accurate forward progress" ask.
4. Retry semantics: a mid-stream drop with partial tokens is a **billed-risk** failure
   (attempts cap 2), not a cheap connection retry — reclassify accordingly.
5. Tests: (a) parity — short task streamed vs stored non-streamed response ⇒ identical
   extracted answer/usage/thinking_source; (b) the kill-shot — one structural
   state-tracking task on deepseek-v4-pro (the exact profile that died at 340s) completes
   with >20.5K completion tokens or runs >340s wall-clock; costs cents.

Slice 1 acceptance: parity + kill-shot green, `ruff` + pytest green, PR merged. Campaign
runs (recovering the 37 ERROR rows, the held seven) belong to Slice 2 — playbook above.
