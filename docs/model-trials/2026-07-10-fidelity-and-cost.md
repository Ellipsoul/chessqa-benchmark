# Model fleet trials: thinking-trace fidelity and cost (2026-07-10)

Decides which models the full ChessQA benchmark runs — and which qualify for Phase 3
reasoning-trace analysis. Method: enumerate the Vercel AI Gateway catalog (306 models),
shortlist 16 with Aron's approval, probe each reasoning-capable model's trace fidelity
(`eval/probe_reasoning.py`, 4 payload shapes each), then run every model against the
**same single task** (`motifs_battery_0042`, the deterministic first task of the seed-42
`--N-samples-per-task 1 --max-tasks 1` sample) so tokens, latency, and cost are directly
comparable.

Trial command (thinking runs; drop the last two flags for non-reasoning models):

```bash
python eval/run_openrouter.py --dataset-root benchmark --output-dir results \
  --model <slug> --N-samples-per-task 1 --max-tasks 1 --workers 1 \
  --enable-thinking --max-tokens 32768
```

Session spend: ~$0.30 in probes + $0.48 in trials (summed `usage.cost`, native-routed runs
priced from tokens x list) ≈ **$0.80 total** — well under the $10 cap.

## Headline findings

1. **The Anthropic adaptive-thinking boundary moved down to 4.7.** `claude-opus-4.7` and
   `claude-opus-4.8` reject the classic `thinking.type: enabled` interface and allow
   `thinking.display` of only `omitted|summarized` — exactly like the Claude 5 family.
   The harness's `>= 5` version heuristic silently misrouted opus-4.8 (zero reasoning
   tokens on every chat-completions payload). Fixed in `anthropic_adaptive_thinking`
   (boundary now `>= 4.7`, probe-verified, with tests). Consequence: **the strongest
   Anthropic model that returns verbatim CoT is `claude-opus-4.6`** (sonnet-4.6 and
   haiku-4.5 also qualify).
2. **Gemini 3.x returns full-text traces through the gateway** (`reasoning.text` details,
   `thinking_source: full_text`) — the only closed-lab frontier traces we can line-verify.
   Caveat below.
3. **Every open-weight reasoner is full_text**: deepseek-r1, deepseek-v4-pro, qwen3.7-max,
   kimi-k2.6, glm-5.2, minimax-m3. Phase 3's verbatim line-verification has a deep bench.
4. **OpenAI is summaries at best**: gpt-5.1-thinking and gpt-5.4-mini return
   `reasoning.summary` + encrypted blobs; gpt-5.6-sol returned **encrypted-only** on all
   four probe shapes (a short summary appeared on the longer trial task). OpenAI models
   qualify only for coarse failure-mode analysis, not line verification.
5. **xai/grok-4.5 was down all session** (upstream 503, no gateway fallbacks).
   `grok-4.3` works (always-on reasoner, summary-class) and stands in for xAI.

## Fidelity + cost table

All rows are the same task, `motifs_battery_0042` (Motifs, multi-answer). "Phase 3?" means
qualifies for verbatim line-verification (requires `full_text`); every reasoning model
qualifies for coarse failure-mode analysis.

| model | payload | thinking_source | reasoning toks | completion toks | latency s | trial $ | full-run $ (x1..x6) | correct | full bench? | Phase 3? |
|---|---|---|---|---|---|---|---|---|---|---|
| anthropic/claude-haiku-4.5 | effort (chat) | **full_text** | 2,120 | 8,181 | 52 | 0.0412 (usage) | 144 .. 860 | no (multi_false_items) | yes — cheap anchor | **yes** |
| anthropic/claude-opus-4.6 | effort (chat) | **full_text** | 286 | 980 | 20 | 0.0256 (usage) | 90 .. 518 | yes | yes — Anthropic full_text anchor | **yes** |
| anthropic/claude-opus-4.8 | adaptive+summarized (native) | summary | n/a | 1,103 | 15 | 0.0290 (listx) | 101 .. 583 | yes | optional | coarse only |
| anthropic/claude-sonnet-5 | adaptive+summarized (native) | summary | n/a | 2,606 | 28 | 0.0266 (listx) | 93 .. 549 | yes — Anthropic frontier | coarse only |
| anthropic/claude-fable-5 | adaptive+summarized (native) | summary | n/a | 790 | 13 | 0.0423 (listx) | 146 .. 838 | optional (ceiling data point) | coarse only |
| openai/gpt-5.6-sol | effort (chat) | summary | 467 | 591 | 11 | 0.0187 (usage) | 66 .. 376 | yes — OpenAI frontier | coarse only |
| openai/gpt-5.1-thinking | effort (chat) | summary | 1,530 | 2,098 | 24 | 0.0212 (usage) | 74 .. 442 | optional (paper GPT-5 lineage) | coarse only |
| openai/gpt-5.4-mini | effort (chat) | summary | 1,552 | 1,722 | 15 | 0.0079 (usage) | 28 .. 163 | yes — cheap tier | coarse only |
| google/gemini-3.1-pro-preview | effort (chat) | **full_text** | 7,294 | 7,840 | 67 | 0.0945 (usage) | 331 .. 1,977 | smoke first — priciest projection | **yes** |
| google/gemini-3.5-flash | effort (chat) | **full_text** | 2,821 | 3,176 | 20 | 0.0289 (usage) | 101 .. 601 | yes | **yes** |
| xai/grok-4.5 | effort (chat) | — | — | — | — | — | — | provider 503 all session | retry later | ? |
| xai/grok-4.3 | effort (chat) | summary | 990 | 1,127 | 11 | 0.0032 (usage) | 11 .. 60 | no (multi_false_items) | yes — xAI stand-in | coarse only |
| deepseek/deepseek-v4-pro | effort (chat) | **full_text** | 2,709 | 2,904 | 48 | 0.0026 (usage) | 9 .. 53 | yes — best value on the board | **yes** |
| deepseek/deepseek-r1 | effort (chat) | **full_text** | 4,900 | 6,233 | 37 | 0.0339 (usage) | 119 .. 708 | optional (the paper's model) | **yes** |
| alibaba/qwen3.7-max | effort (chat) | **full_text** | 3,480 | 3,949 | 72 | 0.0151 (usage) | 53 .. 312 | yes | **yes** |
| moonshotai/kimi-k2.6 | effort (chat) | **full_text** | 4,692 | 6,468 | 84 | 0.0261 (usage) | 91 .. 544 | yes | **yes** |
| zai/glm-5.2 | effort (chat) | **full_text** | 11,406 | 12,684 | 306 | 0.0559 (usage) | 196 .. 1,173 | optional — verbose + 5 min/task | **yes** |
| minimax/minimax-m3 | effort (chat) | **full_text** | 3,300 | 5,125 | 57 | 0.0062 (usage) | 22 .. 129 | yes — cheapest full_text reasoner | **yes** |
| meta/llama-4-maverick | none (non-reasoning) | none | 0 | 880 | 21 | 0.0006 (usage) | 3 .. 18 | no (format_error) | yes — non-reasoning control | n/a |

The chat-completions `reasoning: {"effort": "medium"}` payload the harness already sends
worked for **every** non-Anthropic reasoning model — no per-model payload extensions were
needed beyond the Anthropic adaptive-boundary fix (finding 1).

Notes on columns:

- **thinking_source / reasoning toks** come from the trial row in
  `results/chessqa.sqlite3` (mirrored live from the canonical JSONL).
- **trial cost**: `(usage)` = the gateway's per-call `usage.cost`; `(listx)` = computed
  as tokens x list price because the Anthropic-native `/v1/messages` route returns no
  cost field (fable-5, opus-4.8, sonnet-5).
- **full-run $ (x1..x6)**: `0.8M x P_in + 3500 x trial_completion_tokens x k x P_out`
  for k = 1 and 6 — see next section for why the band is that wide.

## Full-run cost extrapolation method (and its honesty problem)

The cost baseline (`docs/cost-baseline-2026-07-06.md`) gives full-run output volumes for
two measured 50-task profiles: concise haiku-4.5 ≈ 2.9M tokens, verbose sonnet-5 ≈ 17.7M.
Dividing by 3,500 tasks gives weighted per-task means of ~830 and ~5,060 tokens. The trial
task's representativeness differs wildly by profile:

| anchor run | motifs_battery_0042 tokens | weighted mean tokens/task | correction k |
|---|---|---|---|
| haiku-4.5 (concise) | 858 | ~830 | **0.97** |
| sonnet-5 verbose CoT | 850 | ~5,060 | **5.95** |

The task is dead-representative for concise models and under-represents verbose models
~6x (verbose profiles blow up on the 100-task-weight Structural types, not on Motifs).
A single task cannot tell us which regime a new model is in, so every full-run estimate
is a **[x1 .. x6] band**. Before committing to any full run of an expensive model, spend
$1–3 on a 50-task smoke (`--N-samples-per-task 1`) to collapse the band — that was the
baseline doc's advice and this data confirms it.

Anchor-extraction SQL:

```sql
SELECT r.run_key,
       (SELECT res2.completion_tokens FROM results res2
         WHERE res2.run_id = r.run_id AND res2.task_id = 'motifs_battery_0042') AS trial_task_tokens,
       COUNT(*) AS n_tasks,
       ROUND(AVG(res.completion_tokens), 0) AS unweighted_mean_tokens
FROM runs r JOIN results res ON res.run_id = r.run_id
WHERE r.run_key IN ('anthropic_claude-haiku-4.5', 'anthropic_claude-sonnet-5-verbose-cot')
GROUP BY r.run_id;
```

Main-table SQL (every number above is reproducible from this):

```sql
SELECT r.run_key, r.model, res.thinking_source, res.reasoning_tokens,
       res.completion_tokens, res.prompt_tokens, res.latency_ms, res.cost_usd,
       res.is_correct, res.error_type
FROM runs r JOIN results res ON res.run_id = r.run_id
WHERE res.task_id = 'motifs_battery_0042'
ORDER BY r.model;
```

List prices came from `GET https://ai-gateway.vercel.sh/v1/models` (2026-07-10 snapshot).

## Recommended fleet

**Core fleet (14 runs), projected $1,160 .. $6,870** (sum of x1 and x6 bounds — the true
number is inside, and 50-task smokes on the expensive models will pin it before commitment):

- **Frontier closed (4):** claude-sonnet-5, gpt-5.6-sol, gemini-3.5-flash, grok-4.5
  (grok-4.3 until 4.5 recovers)
- **Phase 3 full_text backbone (6):** claude-opus-4.6 (only frontier-adjacent Anthropic
  verbatim CoT), gemini-3.1-pro-preview (**smoke before full run** — widest cost band),
  deepseek-v4-pro, qwen3.7-max, kimi-k2.6, minimax-m3
- **Cheap/methodology tier (3):** claude-haiku-4.5, gpt-5.4-mini, llama-4-maverick
  (non-reasoning control)
- **Paper continuity (1):** deepseek-r1 (the paper's actual model)

**Optional add-ons, in priority order:** gpt-5.1-thinking ($74–442, direct paper-lineage
GPT-5 comparison), claude-opus-4.8 ($101–583, Anthropic score ceiling next to sonnet-5),
claude-fable-5 ($146–838, Mythos-tier ceiling), glm-5.2 ($196–1,173, third open lab — but
12.7K tokens and 306 s on one task; wall-clock and cost risk).

**Phase 3 trace-analysis set (full_text, 10 models):** claude-opus-4.6, claude-haiku-4.5,
gemini-3.1-pro-preview, gemini-3.5-flash, deepseek-r1, deepseek-v4-pro, qwen3.7-max,
kimi-k2.6, glm-5.2, minimax-m3. Summary-class models (all OpenAI, Claude 5 family +
opus-4.7/4.8, grok) support only the coarse LLM-judge failure-mode taxonomy, not
python-chess line verification.

Aron makes the final call on fleet composition; nothing beyond the trial rows above has
been run.

## Surprises

- **Anthropic quietly moved Opus 4.7+ onto the adaptive (summarized-only) interface.** The
  probe that was supposed to be a formality falsified our "trailing version >= 5" model of
  the world and cost us frontier Anthropic verbatim traces: opus-4.6 is now the ceiling for
  Anthropic full_text. Worth re-probing whenever Anthropic ships a new dot version.
- **Gemini 3.x hands back full reasoning text through the gateway** (`reasoning.text`
  details) — memory said Google returned only thought summaries. Caveat: Google's API docs
  historically describe returned "thoughts" as model-generated summaries of internal
  reasoning; the gateway tags them `reasoning.text` and they read as detailed step-by-step
  CoT, but before Phase 3 leans on Gemini traces we should verify with Google's current
  docs whether these are verbatim. Tagged full_text per the harness's mechanical criterion.
- **gpt-5.6-sol returned encrypted-only reasoning on all four probe shapes but a summary
  on the (longer) trial task** — OpenAI's summary emission appears length- or
  content-dependent, so `thinking_source` for OpenAI models can vary row to row within a
  run. Phase 3 tooling should treat thinking_source as per-result, not per-model (it
  already is per-result in the DB).
- **Thinking made haiku-4.5 verbose and still wrong**: 858 tokens (concise, wrong) without
  thinking vs 8,181 tokens (wrong differently) with. Meanwhile opus-4.6 answered correctly
  in 980 total completion tokens. Token spend and quality are decoupled at this task — n=1,
  but consistent with the paper's finding that thinking helps unevenly.
- **The correctness split on this one task is stark**: every closed-lab frontier model
  (Claude 5s, opus-4.6/4.8, all OpenAI, both Geminis, qwen3.7-max, glm-5.2) got it right;
  deepseek-r1/v4-pro, kimi-k2.6, minimax-m3, grok-4.3 all failed with multi-answer set
  errors. n=1 — do not generalize — but it previews the multi-answer scoring pain the
  paper reports.
- **glm-5.2 took 306 seconds and 12.7K tokens for one task.** A 3,500-task run at that
  latency needs high concurrency to finish in reasonable wall-clock; it is also the most
  expensive open model per run. Optional for that reason, despite full_text traces.
- **grok-4.5 was 503 at xAI all session** (gateway shows `fallbacksAvailable: []`), so the
  xAI slot is measured on grok-4.3. Re-probe 4.5 before fleet launch.
- The `xai_grok-4.5-thinking` results file/DB run exists with one failed row
  (`format_error`, 0 tokens) — ignore or delete before analysis; it is an availability
  artifact, not model behavior.
