# ChessQA Explorer — design spec

Date: 2026-07-13 (rev 2, same day: positions grouped into five category pages instead of
one page per task; compression/loading strategy specified with measured numbers; task
count corrected 51 → 50)
Status: approved in session (audience, data layer, data slice, stack, and all four design
sections approved by Aron; rev 2 changes requested by Aron)
Working title: **ChessQA Explorer** (rename freely before launch)

## Goal

A public, static web showcase of the smoke-campaign results: all 50 smoked ChessQA tasks
rendered on real chess boards, grouped into the benchmark's five categories (one page per
category, scrollable through its positions), with every fleet model's answer, outcome,
cost, and — the centerpiece — the unedited thinking trace, including the runs that burned
their entire 32K token budget and never produced an answer.

It is explicitly a **behavior browser, not a leaderboard**: 50 tasks per model, one task
per task type, so per-category numbers are anecdotes. The UI shows counts ("23/24"), never
bare percentages, and the home page carries a framing box saying exactly this.

Secondary goals: serve as the seed of the "public web explorer" the project brief
anticipates for the full run, and preview Phase 3 (the exporter's move-legality check is
the first mechanical trace/answer verification shipped).

## Decisions already made

| Decision | Choice |
|---|---|
| Audience | Public showcase (polished, shareable, explains the benchmark) |
| Data layer | Static snapshot exported from SQLite; no cloud DB. Neon rejected: full-run traces (>1 GB) exceed its free tier anyway; 700 results don't need a server |
| Data slice | The sixteen 50-task runs (14 fleet configs + 2 reused baselines). 1-task probes excluded. Per-result **cost shown**. **Full resolved prompts shown**. `raw_message` never exported |
| Stack | Next.js (App Router), fully static export, deployed on Vercel. **Updated 2026-07-14:** lives in a separate sibling repo `~/Desktop/Coding_Adventures/chess-benchmark-showcase` (Next 16.2, React 19, Tailwind 4, no `src/` dir — `app/`, `lib/`, `components/`, `public/` at root), not in `web/` here. The exporter stays in this repo and writes to `../chess-benchmark-showcase/public/data` |
| Board | react-chessboard v5 + chess.js (both MIT). **Not** chessground — GPL-3, would contaminate this MIT repo |

Included run_ids as of the 2026-07-13 DB state: 1, 2, 3, 5, 6, 8, 10, 11, 12, 14, 15, 16,
17, 19, 20, 21. The exporter selects by rule, not hard-coded ids: `status='complete' AND
n_results >= 50`, excluding superseded duplicates if any appear later. Variants get
distinct display names, e.g. `claude-sonnet-5 (thinking)` vs `claude-sonnet-5 (verbose
CoT)`, `claude-haiku-4.5 (thinking)` vs `(no thinking)`.

## What exploration established (2026-07-13)

- 805 results / 21 runs in `results/chessqa.sqlite3`; the 16 canonical runs cover 50
  distinct tasks (one per task type) across all five categories: Structural 11, Motifs 6,
  Short Tactics 24, Position Judgement 5, Semantic 4.
- Display content is small: responses 1.1 MB + traces 15.2 MB (16.24 MB total measured
  across the 16 canonical runs). The 63 MB DB is mostly `raw_message` (39 MB), which
  stays private. Worst single task ≈ 0.71 MB of trace text.
- Traces: 401 `full_text` (avg 33K chars, max 266K), 213 `summary`, 36 `plain`, 152 `none`.
- 60 results are `max_token_reached`; most have an empty response but a full trace
  (kimi-k2.6 ×18, deepseek-v4-pro ×14, minimax-m3 ×8). Gemini-3.1-pro has a 266K-char
  looping trace on `short_tactics_theme_defensiveMove` with no answer.
- Of 158 wrong Short Tactics answers, 41 are **illegal moves** (python-chess verified);
  12 more aren't parseable as UCI.
- Task metadata is rich: Lichess `puzzle_id` (deep link `lichess.org/training/<id>`),
  puzzle rating, themes, full PV; Position Judgement has Stockfish depth-48 eval, best
  line, and the five answer buckets (−400, −200, 0, 200, 400).

## Answer shapes → render primitives

Computed **at export time** with python-chess. The frontend never parses chess notation;
it renders primitives.

| Task family | Answer shape | Primitive |
|---|---|---|
| short_tactics_*, structural_check_in_1, structural_legal_move_* | UCI move(s), possibly comma-separated | `{type:"moves", arrows:[{from,to,promotion?}]}` |
| motifs_* | Chains like `b4>d2>e1` (pin: attacker>through>target) or `d3>c5-f2` (fork: attacker>victim-victim) | `{type:"chain", arrows:[…]}` — parser per motif family, unit-tested against dataset generator semantics in `dataset/02_motifs.py`; fallback = highlight all referenced squares |
| structural_{capture,control,protect}_squares | Square list `f6, f8` | `{type:"squares", squares:[…]}` |
| structural_piece_arrangement, structural_check_detection | Piece placements / `White Knight at f6` | `{type:"pieces", items:[{piece, square}]}` |
| structural_state_tracking_* | A full FEN | `{type:"fen", fen, diff_squares:[…]}` — diff of the model's imagined board vs the correct FEN, rendered as a second board with divergent squares highlighted |
| position_judgement_* | One of −400…400 | `{type:"eval", value}` — marker on a scale, next to Stockfish's depth-48 value |
| semantic_* | Letter A–D | `{type:"choice", letter}` — task page shows every run's pick (consensus view) |

Every extracted single-move answer also gets `legality: legal | illegal | unparseable`
(vs the task FEN). `illegal` is surfaced as a badge and its own heatmap color — the
Phase 3 teaser.

## Export pipeline

`eval/export_web.py` (stdlib + python-chess; reuses `eval/storage.py` access patterns).
Reads the SQLite, writes the showcase repo's `public/data/` (default
`../chess-benchmark-showcase/public/data`, sibling-checkout assumption; `--out-dir` overrides):

- **`index.json`** (~100 KB): `generated_at`, `dataset_hash`, `runs[]` (slug, display
  name, model, variant, backend, exact `reasoning_config`, git_commit, started_at,
  n_correct, n_capped, n_illegal, total_cost_usd, avg completion tokens, dominant
  `thinking_source`), `tasks[]` (task_id, slug, type, category, FEN, answer_type, and
  `outcomes: {run_slug: outcome_code}` for the heatmap).
- **`categories/<slug>.json`** (~30–200 KB raw; Short Tactics largest): the category's
  tasks in display order, each with question, resolved prompt (placeholders filled with
  the same logic as `format_prompt()` in `eval/run_openrouter.py`, matching each run's
  format-example group), input FEN + UCI move list when input is `"FEN | moves"`, task
  metadata verbatim, correct answer + its primitives, and `results[]` per run: extracted
  answer, `error_type` (raw), outcome code, legality, primitives, cost_usd,
  completion/reasoning tokens, latency_ms, n_attempts, `thinking_source`,
  `thinking_chars`. **No trace text.**
- **`traces/<task_id>.json`** (≤ ~0.7 MB raw, ~150–200 KB compressed worst case): trace
  text per run slug. One file per task, fetched lazily (see Compression & loading).

Outcome codes (heatmap palette collapses raw `error_type`): `correct`, `wrong`,
`illegal` (wrong ∧ illegal move), `capped`, `format_error`. Multi-answer partials
(`multi_extra/missing/false_items`) collapse to `wrong` with the raw subtype shown on the
card.

Deterministic output (sorted keys, stable slugs). Pytest coverage in `tests/` for: run
selection rule, every primitive parser (esp. motif chains), legality tagging, placeholder
resolution, determinism. `make export-web` target.

## Site structure

The showcase repo — Next.js App Router, TypeScript, Tailwind, `output: 'export'` (guarantees pure
static, no functions, $0). Data loaded from `/data/*.json` static assets.

- **`/` Home** — hero with one-paragraph explanation + prominent CSSLab/arXiv attribution;
  the honest-framing box; model summary table (counts, cost, tokens, capped, trace
  fidelity); the **16×50 heatmap** (rows = runs, columns = tasks grouped by category,
  cells = outcome color, every cell links to the position's anchor on its category page,
  with the run preselected); **Exhibits** — 3–5 hand-picked deep links defined in
  `exhibits.ts` (showcase repo root). Initial set:
  1. Gemini-3.1-pro's 266K-char loop on the defensive-move puzzle (capped, no answer).
  2. Illegal-move gallery (the 41, grouped by model).
  3. A state-tracking diff where a model's imagined board drifts from reality.
  4. DeepSeek R1's mate-in-2: the trace confidently narrates a forced mate
     (`Nh7+ … Qg7#`) and commits to it — but the line is unsound; the correct move was
     the queen check on h7. (Blurb wording is Aron's to verify — his chess labels are
     ground truth.)
- **`/category/<slug>`** (×5: `structural`, `motifs`, `short-tactics`,
  `position-judgement`, `semantic`) — a short category explainer, a **position
  navigator** (sticky sidebar/strip listing the category's task types; Short Tactics, at
  24 positions, additionally groups by rating-band vs theme), and the positions rendered
  as scrollable sections. Each position section is the full unit from rev 1: board
  (oriented to side-to-move; dragging disabled; state-tracking inputs get step-through
  animation of the move prefix), question with "show full prompt" toggle, metadata panel
  (Lichess link, rating, themes, Stockfish eval/best line where present), correct-answer
  overlay toggle, then one **attempt card** per run: answer + outcome/legality badges +
  cost/tokens/latency; selecting a card overlays its primitives on the main board
  (distinct color vs correct answer); trace drawer per card — fidelity badge, char/token
  counts, capped banner ("hit the 32K ceiling mid-thought; no answer was ever produced"),
  unedited text in measure-limited typography.
  **Deep links:** every position has a stable hash anchor (`/category/short-tactics#short_tactics_theme_mateIn2`),
  and an optional `?run=<run_slug>` query preselects/highlights that model's card —
  this is what heatmap cells and exhibits link to. Position sections below the fold
  render lazily (virtualized) so the 24-position Short Tactics page stays light.
- **`/model/<run_slug>`** (×16) — config provenance (exact reasoning payload, backend,
  run date, git commit, total cost), per-category outcome strip with explicit n's,
  filterable 50-attempt list (all/correct/wrong/illegal/capped).
- **`/about`** — methodology, what the smoke test is and isn't, attribution (CSSLab
  upstream, arXiv:2510.23948, MIT), link to this repo, note that costs are real measured
  spend, Phase 3 roadmap teaser.

## Compression & loading (measured 2026-07-13)

The bandwidth concern is real but smaller than the 63 MB DB suggests. Measured on the
actual corpus (all responses + traces for the 16 canonical runs):

| | size |
|---|---|
| Raw text content | 16.24 MB |
| gzip −9 | 4.11 MB (3.9×) |
| brotli −11 | 3.00 MB (5.4×) |

Per-category raw content (traces dominate): Short Tactics 9.87 MB, Structural 1.99 MB,
Motifs 1.77 MB, Position Judgement 1.70 MB, Semantic 0.89 MB.

**Strategy: standard HTTP compression + lazy loading. No custom compression or
client-side decompression code.**

1. **Wire compression is the CDN's job.** All data files are plain `.json` static
   assets; Vercel's edge network negotiates `Content-Encoding` (brotli/gzip) per the
   client's `Accept-Encoding`. LLM prose compresses extremely well (measured above), so
   even the absolute worst case — a user opening every trace on the site — transfers
   ~3–4 MB total. *Implementation checklist item:* after first deploy, verify
   `Content-Encoding: br` (or at minimum `gzip`) on a `traces/*.json` response and
   record actual transfer sizes; the budget holds even if the edge only applies gzip.
2. **Nobody fetches the corpus.** The page-load path is small and staged:
   - Home: `index.json` (~100 KB raw → ~20 KB wire).
   - Category page: its `categories/<slug>.json` (30–200 KB raw → roughly 10–40 KB
     wire) — everything needed for first paint: boards, questions, answers, badges,
     stats. Renders immediately; no trace bytes on this path.
   - Traces: per-task `traces/<task_id>.json` files (median ~50–100 KB wire, worst
     ~200 KB), fetched (a) on first trace-drawer open, and (b) speculatively for the
     position currently in view, during browser idle time (IntersectionObserver +
     `requestIdleCallback`), so opening a trace usually feels instant. Fetched files
     are kept in memory; JSON parsing at these sizes is not a concern.
3. **Caching:** stable filenames + Vercel's default static-asset ETags (revalidation =
   cheap 304s between campaigns). If full-run scale ever makes revalidation chatter
   matter, switch the exporter to content-hashed filenames referenced from `index.json`
   and mark them immutable — noted, not built, in v1 (YAGNI).

## Deployment & operations

- Vercel project on the `chess-benchmark-showcase` repo (repo root; no root-directory
  config); static output; no env vars, no secrets.
- Update cycle: run campaign in this repo → `make export-web` (writes into the sibling
  checkout) → commit JSON + push in the showcase repo → auto-deploy. Exported JSON is
  checked in there (reviewable diffs of what's publicly visible).
- Full-run scale: same exporter and loading scheme; per-task trace JSON keeps any single
  fetch bounded; if trace files grow past ~2 MB raw, split per-run trace files under
  `traces/<task_id>/`. Category pages already virtualize position sections, but at 700
  positions per category the `categories/<slug>.json` files themselves will need
  pagination/sharding — revisit then, not now.
- CI: pytest (exporter) in this repo's existing test flow; `next build` + eslint in the
  showcase repo.

## Testing

- Exporter: pytest as above (the chess-parsing logic is the risk surface).
- Web: TypeScript strict + eslint + `next build`; component smoke tests only where logic
  lives (heatmap outcome mapping, primitive→board-prop translation). No E2E suite in v1.

## Out of scope (v1)

- Full-run data, server-side anything, search over traces, per-task OG images,
  user accounts/comments, Phase 3 verified-line rendering inside traces (the legality
  badge is the only Phase 3 preview), custom domain (Vercel subdomain fine at launch).

## Open items for implementation

- Final name + Vercel project name (Aron).
- Visual design direction: use the frontend-design skill at implementation time; chess
  aesthetic, light/dark, restrained palette; heatmap colors must pass contrast checks.
- Exhibit blurbs: Aron reviews wording (his chess judgment is ground truth for claims
  like "describes the mate but plays the wrong move").
