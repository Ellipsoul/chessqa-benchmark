# ChessQA Explorer — Web App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the static Next.js showcase in `web/` that renders the exported smoke-campaign JSON: home (summary + heatmap + exhibits), five category pages with scrollable positions and boards, sixteen model pages, and an about page.

**Architecture:** Next.js App Router with `output: 'export'` (pure static, no functions). Page shells are prerendered; data is fetched client-side from `/data/*.json` static assets (category core ~10–40 KB wire; traces lazy per task). All chess-answer intelligence arrived pre-parsed as "primitives" from the exporter — the frontend only translates primitives to react-chessboard props (one pure, unit-tested module).

**Tech Stack:** Next.js (App Router, TypeScript, Tailwind — from Aron's scaffold), react-chessboard v5 (MIT), chess.js, vitest for the pure logic modules.

**Spec:** `docs/superpowers/specs/2026-07-13-chessqa-explorer-design.md` (rev 2). Subsystem 2 of 2 — requires the exporter plan (`2026-07-13-explorer-exporter.md`) to be merged first.

## Repo layout update (2026-07-14) — READ FIRST

The app lives in a **separate sibling repository**, not in `web/` of the benchmark repo:
`~/Desktop/Coding_Adventures/chess-benchmark-showcase` (create-next-app scaffold verified
2026-07-14: Next 16.2.10, React 19.2, Tailwind 4, App Router, ESLint, **no `src/`
directory** — `app/` at root, tsconfig maps `@/*` → `./*`, so the `@/lib/...` imports in
this plan work unchanged). Translate every path in the tasks below:

| Plan says | Actually |
|---|---|
| `web/src/lib/...`, `web/src/components/...`, `web/src/app/...`, `web/src/exhibits.ts` | `lib/...`, `components/...`, `app/...`, `exhibits.ts` at the showcase repo root |
| `web/public/data` | `public/data` in the showcase repo (the exporter writes here from the benchmark repo via `make export-web`) |
| `cd web && ...` | run from the showcase repo root |
| `web/next.config.ts`, `web/package.json`, `web/vitest.config.ts` | same filenames at the showcase repo root |

Git: commits for this plan go to the **showcase repo** (currently `main`-only, no remote —
work on a feature branch and let Aron wire up the GitHub remote/PR flow). The benchmark
repo's PR rules (Ellipsoul fork, never CSSLab) apply only to exporter-side changes.

## Global Constraints

- **Precondition:** the sibling showcase repo exists as described in the layout-update
  block above. Task 1 verifies and configures it; if the scaffold differs materially
  (pages router, missing Tailwind), STOP and flag rather than adapt silently.
- Data contract: the JSON schemas in the exporter plan's Task 5 Interfaces block. `lib/types.ts` mirrors them exactly — field names come from the exporter, do not "improve" them.
- Run slugs contain dots (e.g. `anthropic_claude-haiku-4.5-thinking`); routes and `document.getElementById` must not assume alphanumeric ids.
- Counts, never bare percentages: every accuracy display is `n/N` (spec's honest-framing rule).
- Board library: react-chessboard v5 API — `<Chessboard options={{ position, arrows, squareStyles, boardOrientation, allowDragging }} />`, `Arrow = { startSquare, endSquare, color }`. If TypeScript rejects an option name, check current docs via context7 (`/clariity/react-chessboard`) — do not cast to `any`.
- Attribution (About page + footer): ChessQA benchmark by CSSLab, University of Toronto — arXiv:2510.23948, upstream github.com/CSSLab/chessqa-benchmark, MIT license.
- Verification gates: `npx vitest run`, `npm run lint`, `npm run build` — all from `web/`.
- Commits on a feature branch; PRs to `Ellipsoul/chessqa-benchmark` only, never CSSLab.

---

### Task 1: Configure scaffold, commit exported data, types + loaders

**Files:**
- Modify: `web/next.config.ts`, `web/package.json`
- Create: `web/src/lib/types.ts`, `web/src/lib/data.ts`, `web/src/lib/outcome.ts`, `web/vitest.config.ts`, `web/src/lib/outcome.test.ts`
- Create (generated): `web/public/data/**` via `make export-web`

**Interfaces:**
- Produces: all shared types (`IndexData`, `RunSummary`, `TaskSummary`, `CategoryData`, `TaskDetail`, `TaskResult`, `Primitives`, `Outcome`, `TraceData`); `getIndex(): Promise<IndexData>`, `getCategory(slug): Promise<CategoryData>`, `getTraces(taskId): Promise<TraceData>` (memoized); `OUTCOME_META` + `CATEGORY_ORDER`. Every later task imports from these three modules.

- [ ] **Step 1: Verify scaffold + configure static export**

Confirm `web/src/app/` exists and `web/package.json` has `next` ≥ 15. Then set:

```ts
// web/next.config.ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
};

export default nextConfig;
```

- [ ] **Step 2: Install runtime + test deps**

```bash
cd web && npm install react-chessboard chess.js && npm install -D vitest
```

Add to `web/package.json` scripts: `"test": "vitest run"`.

```ts
// web/vitest.config.ts
import path from "path";
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: { environment: "node", include: ["src/**/*.test.ts"] },
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
});
```

- [ ] **Step 3: Generate and commit the data**

```bash
cd ~/Desktop/Coding_Adventures/chessqa-benchmark && make export-web
cd ~/Desktop/Coding_Adventures/chess-benchmark-showcase && git add public/data
```
Expected: `exported 16 runs x 50 tasks -> .../chess-benchmark-showcase/public/data (~17 MB)`. The JSON is checked into the showcase repo deliberately (spec: reviewable diffs of what's publicly visible).

- [ ] **Step 4: Write the failing outcome test**

```ts
// web/src/lib/outcome.test.ts
import { describe, expect, it } from "vitest";
import { OUTCOME_META } from "@/lib/outcome";
import type { Outcome } from "@/lib/types";

describe("OUTCOME_META", () => {
  it("covers every outcome code with label and colors", () => {
    const outcomes: Outcome[] = ["correct", "wrong", "illegal", "capped", "format_error"];
    for (const outcome of outcomes) {
      expect(OUTCOME_META[outcome].label).toBeTruthy();
      expect(OUTCOME_META[outcome].cellClass).toMatch(/^bg-/);
      expect(OUTCOME_META[outcome].badgeClass).toBeTruthy();
    }
  });
});
```

Run: `cd web && npx vitest run` — expected FAIL (`Cannot find module '@/lib/outcome'`).

- [ ] **Step 5: Write types, loaders, outcome meta**

```ts
// web/src/lib/types.ts — mirrors eval/export_web.py output; do not rename fields
export type Outcome = "correct" | "wrong" | "illegal" | "capped" | "format_error";
export type Legality = "legal" | "illegal" | "unparseable" | null;
export type ThinkingSource = "full_text" | "summary" | "plain" | "untyped" | "encrypted_only" | "none" | null;

export interface ArrowSpec { from: string; to: string; promotion?: string }
export type Primitives =
  | { type: "moves"; arrows: ArrowSpec[] }
  | { type: "chain"; arrows: ArrowSpec[] }
  | { type: "squares"; squares: string[] }
  | { type: "pieces"; items: { color: string; piece: string; square: string }[] }
  | { type: "fen"; fen: string; diff_squares: string[] }
  | { type: "eval"; value: number }
  | { type: "choice"; letter: string }
  | { type: "none" }
  | { type: "text"; text: string };

export interface RunSummary {
  slug: string; display_name: string; model: string; backend: string;
  enable_thinking: boolean; reasoning_config: Record<string, unknown> | null;
  git_commit: string | null; started_at: string | null;
  n_results: number; n_correct: number; n_capped: number; n_illegal: number;
  total_cost_usd: number | null; avg_completion_tokens: number | null;
  thinking_sources: Record<string, number>;
}

export interface TaskSummary {
  task_id: string; task_type: string; task_category: string; category_slug: string;
  fen: string; answer_type: "single" | "multi"; outcomes: Record<string, Outcome>;
}

export interface IndexData { generated_at: string; dataset_hash: string | null; runs: RunSummary[]; tasks: TaskSummary[] }

export interface TaskResult {
  run: string; extracted: string | null; error_type: string; outcome: Outcome;
  legality: Legality; primitives: Primitives; cost_usd: number | null;
  completion_tokens: number | null; reasoning_tokens: number | null; latency_ms: number | null;
  n_attempts: number | null; thinking_source: ThinkingSource; thinking_chars: number;
}

export interface TaskDetail {
  task_id: string; task_type: string; task_category: string; question: string;
  resolved_prompt: string; input_fen: string; input_moves: string[];
  metadata: Record<string, unknown> | null; correct_answer: string;
  answer_type: "single" | "multi"; correct_primitives: Primitives; results: TaskResult[];
}

export interface CategoryData { category: string; slug: string; tasks: TaskDetail[] }
export interface TraceData {
  task_id: string;
  traces: Record<string, { thinking_source: ThinkingSource; content: string; response: string }>;
}
```

```ts
// web/src/lib/data.ts — memoized fetchers for the static /data assets
import type { CategoryData, IndexData, TraceData } from "./types";

const cache = new Map<string, Promise<unknown>>();

function fetchJson<T>(path: string): Promise<T> {
  if (!cache.has(path)) {
    const promise = fetch(path).then((res) => {
      if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
      return res.json();
    });
    promise.catch(() => cache.delete(path)); // don't cache failures
    cache.set(path, promise);
  }
  return cache.get(path) as Promise<T>;
}

export const getIndex = () => fetchJson<IndexData>("/data/index.json");
export const getCategory = (slug: string) => fetchJson<CategoryData>(`/data/categories/${slug}.json`);
export const getTraces = (taskId: string) => fetchJson<TraceData>(`/data/traces/${taskId}.json`);
```

```ts
// web/src/lib/outcome.ts
import type { Outcome } from "./types";

export const OUTCOME_META: Record<Outcome, { label: string; badgeClass: string; cellClass: string }> = {
  correct:      { label: "Correct",           badgeClass: "bg-emerald-100 text-emerald-800", cellClass: "bg-emerald-500" },
  wrong:        { label: "Wrong",             badgeClass: "bg-rose-100 text-rose-800",       cellClass: "bg-rose-400" },
  illegal:      { label: "Illegal move",      badgeClass: "bg-fuchsia-100 text-fuchsia-800", cellClass: "bg-fuchsia-600" },
  capped:       { label: "Ran out of tokens", badgeClass: "bg-amber-100 text-amber-800",     cellClass: "bg-amber-400" },
  format_error: { label: "Format error",      badgeClass: "bg-slate-200 text-slate-700",     cellClass: "bg-slate-400" },
};

export const CATEGORY_ORDER: { slug: string; name: string }[] = [
  { slug: "structural", name: "Structural" },
  { slug: "motifs", name: "Motifs" },
  { slug: "short-tactics", name: "Short Tactics" },
  { slug: "position-judgement", name: "Position Judgement" },
  { slug: "semantic", name: "Semantic" },
];
```

- [ ] **Step 6: Verify and commit**

Run: `cd web && npx vitest run && npm run lint && npm run build`
Expected: test passes, build succeeds (default scaffold pages still fine).

```bash
git add web
git commit -m "feat(web): static-export config, exported data, types and loaders"
```

---

### Task 2: Board logic module + Board component

**Files:**
- Create: `web/src/lib/board.ts`, `web/src/lib/board.test.ts`, `web/src/components/Board.tsx`

**Interfaces:**
- Produces: `primitivesToOverlay(p, color) -> BoardOverlay`, `mergeOverlays(...overlays)`, `diffOverlay(squares)`, `sideToMove(fen)`, `fenAfterMoves(fen, moves, count)`, constants `CORRECT_COLOR`/`MODEL_COLOR`; `<Board fen orientation overlay maxWidth />` (client component, dragging disabled). Consumed by PositionSection (Task 4) and exhibits.

- [ ] **Step 1: Write the failing tests**

```ts
// web/src/lib/board.test.ts
import { describe, expect, it } from "vitest";
import { CORRECT_COLOR, fenAfterMoves, mergeOverlays, primitivesToOverlay, sideToMove } from "@/lib/board";

const START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

describe("primitivesToOverlay", () => {
  it("turns move/chain primitives into colored arrows", () => {
    const overlay = primitivesToOverlay({ type: "moves", arrows: [{ from: "h6", to: "h7" }] }, CORRECT_COLOR);
    expect(overlay.arrows).toEqual([{ startSquare: "h6", endSquare: "h7", color: CORRECT_COLOR }]);
  });
  it("turns squares into background styles and pieces into rings", () => {
    const squares = primitivesToOverlay({ type: "squares", squares: ["f6", "f8"] }, CORRECT_COLOR);
    expect(Object.keys(squares.squareStyles)).toEqual(["f6", "f8"]);
    const pieces = primitivesToOverlay({ type: "pieces", items: [{ color: "White", piece: "Knight", square: "f6" }] }, CORRECT_COLOR);
    expect(pieces.squareStyles.f6.boxShadow).toContain(CORRECT_COLOR);
  });
  it("returns an empty overlay for non-board primitives", () => {
    for (const p of [{ type: "eval", value: 200 }, { type: "choice", letter: "A" }, { type: "none" }, { type: "text", text: "hi" }] as const) {
      expect(primitivesToOverlay(p, CORRECT_COLOR)).toEqual({ arrows: [], squareStyles: {} });
    }
  });
});

describe("board helpers", () => {
  it("sideToMove reads the FEN", () => {
    expect(sideToMove(START)).toBe("white");
    expect(sideToMove("8/8/8/8/8/8/8/K6k b - - 0 1")).toBe("black");
  });
  it("fenAfterMoves applies a uci prefix", () => {
    const after = fenAfterMoves(START, ["e2e4", "e7e5"], 1);
    expect(after.split(" ")[1]).toBe("b"); // one move applied -> black to move
    expect(after.startsWith("rnbqkbnr/pppppppp/8/8/4P3/")).toBe(true);
    expect(fenAfterMoves(START, ["e2e4", "e7e5"], 0)).toBe(START);
  });
  it("mergeOverlays concatenates arrows and merges styles", () => {
    const a = primitivesToOverlay({ type: "moves", arrows: [{ from: "e2", to: "e4" }] }, "#111111");
    const b = primitivesToOverlay({ type: "squares", squares: ["d4"] }, "#222222");
    const merged = mergeOverlays(a, b);
    expect(merged.arrows).toHaveLength(1);
    expect(merged.squareStyles.d4).toBeDefined();
  });
});
```

Run: `cd web && npx vitest run` — expected FAIL (`Cannot find module '@/lib/board'`).

- [ ] **Step 2: Implement `lib/board.ts`**

```ts
// web/src/lib/board.ts — the only place primitives meet react-chessboard's prop shapes
import { Chess } from "chess.js";
import type { CSSProperties } from "react";
import type { Primitives } from "./types";

export interface BoardArrow { startSquare: string; endSquare: string; color: string }
export interface BoardOverlay { arrows: BoardArrow[]; squareStyles: Record<string, CSSProperties> }

export const CORRECT_COLOR = "#16a34a";
export const MODEL_COLOR = "#ea580c";

function hexToRgba(hex: string, alpha: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

export function primitivesToOverlay(p: Primitives | null | undefined, color: string): BoardOverlay {
  const overlay: BoardOverlay = { arrows: [], squareStyles: {} };
  if (!p) return overlay;
  if (p.type === "moves" || p.type === "chain") {
    overlay.arrows = p.arrows.map((a) => ({ startSquare: a.from, endSquare: a.to, color }));
  } else if (p.type === "squares") {
    for (const square of p.squares) overlay.squareStyles[square] = { backgroundColor: hexToRgba(color, 0.4) };
  } else if (p.type === "pieces") {
    for (const item of p.items) overlay.squareStyles[item.square] = { boxShadow: `inset 0 0 0 3px ${color}` };
  }
  return overlay; // fen/eval/choice/none/text are rendered outside the board
}

export function mergeOverlays(...overlays: BoardOverlay[]): BoardOverlay {
  return {
    arrows: overlays.flatMap((o) => o.arrows),
    squareStyles: Object.assign({}, ...overlays.map((o) => o.squareStyles)),
  };
}

export function diffOverlay(diffSquares: string[]): BoardOverlay {
  const overlay: BoardOverlay = { arrows: [], squareStyles: {} };
  for (const square of diffSquares) overlay.squareStyles[square] = { backgroundColor: "rgba(220, 38, 38, 0.45)" };
  return overlay;
}

export function sideToMove(fen: string): "white" | "black" {
  return fen.split(" ")[1] === "b" ? "black" : "white";
}

export function fenAfterMoves(fen: string, moves: string[], count: number): string {
  if (count === 0) return fen;
  const chess = new Chess(fen);
  for (const move of moves.slice(0, count)) {
    chess.move({ from: move.slice(0, 2), to: move.slice(2, 4), promotion: move[4] as "q" | "r" | "b" | "n" | undefined });
  }
  return chess.fen();
}
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd web && npx vitest run` — expected: all pass.

- [ ] **Step 4: Implement the Board component**

```tsx
// web/src/components/Board.tsx
"use client";

import { Chessboard } from "react-chessboard";
import type { BoardOverlay } from "@/lib/board";

export function Board({ fen, orientation, overlay, maxWidth = 400 }: {
  fen: string;
  orientation: "white" | "black";
  overlay?: BoardOverlay;
  maxWidth?: number;
}) {
  return (
    <div style={{ maxWidth }} className="w-full select-none">
      <Chessboard
        options={{
          position: fen,
          boardOrientation: orientation,
          allowDragging: false,
          arrows: overlay?.arrows ?? [],
          squareStyles: overlay?.squareStyles ?? {},
        }}
      />
    </div>
  );
}
```

- [ ] **Step 5: Verify compile + commit**

Run: `cd web && npm run lint && npm run build` — expected: clean (component not yet routed; build validates types).

```bash
git add web/src
git commit -m "feat(web): board overlay logic and Board component"
```

---

### Task 3: LazyMount, AttemptCard, TraceDrawer

**Files:**
- Create: `web/src/components/LazyMount.tsx`, `web/src/components/TraceDrawer.tsx`, `web/src/components/AttemptCard.tsx`

**Interfaces:**
- Consumes: `getTraces`, `OUTCOME_META`, types.
- Produces: `<LazyMount minHeight eager>{children}</LazyMount>`; `<TraceDrawer taskId result />`; `<AttemptCard result run selected onSelect taskId />` where `run: RunSummary`, `result: TaskResult`, `onSelect: () => void`. Consumed by PositionSection (Task 4).

- [ ] **Step 1: LazyMount (IntersectionObserver, mounts once)**

```tsx
// web/src/components/LazyMount.tsx
"use client";

import { useEffect, useRef, useState } from "react";

export function LazyMount({ children, minHeight = 600, eager = false }: {
  children: React.ReactNode;
  minHeight?: number;
  eager?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(eager);

  useEffect(() => {
    if (visible || !ref.current) return;
    const observer = new IntersectionObserver(
      (entries) => entries[0].isIntersecting && setVisible(true),
      { rootMargin: "800px" },
    );
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, [visible]);

  return <div ref={ref} style={visible ? undefined : { minHeight }}>{visible ? children : null}</div>;
}
```

- [ ] **Step 2: TraceDrawer (lazy fetch on first open, fidelity badge, capped banner)**

```tsx
// web/src/components/TraceDrawer.tsx
"use client";

import { useEffect, useState } from "react";
import { getTraces } from "@/lib/data";
import type { TaskResult } from "@/lib/types";

const FIDELITY_LABELS: Record<string, string> = {
  full_text: "full unedited stream",
  summary: "provider-summarized",
  plain: "plain-text reasoning",
  untyped: "untyped reasoning",
  encrypted_only: "encrypted (not viewable)",
  none: "no trace returned",
};

export function TraceDrawer({ taskId, result }: { taskId: string; result: TaskResult }) {
  const [open, setOpen] = useState(false);
  const [trace, setTrace] = useState<{ content: string; response: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open || trace) return;
    getTraces(taskId)
      .then((data) => setTrace(data.traces[result.run] ?? { content: "", response: "" }))
      .catch((err) => setError(String(err)));
  }, [open, trace, taskId, result.run]);

  const fidelity = FIDELITY_LABELS[result.thinking_source ?? "none"] ?? result.thinking_source;

  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="text-sm underline decoration-dotted text-slate-600 hover:text-slate-900"
      >
        {open ? "Hide thoughts" : `Show thoughts (${fidelity}, ${result.thinking_chars.toLocaleString()} chars)`}
      </button>
      {open && (
        <div className="mt-2 rounded border border-slate-200 bg-slate-50 p-3 text-sm">
          {result.outcome === "capped" && (
            <p className="mb-2 rounded bg-amber-100 px-2 py-1 font-medium text-amber-900">
              Hit the 32K token ceiling mid-thought — no answer was ever produced.
            </p>
          )}
          {error && <p className="text-rose-700">Failed to load trace: {error}</p>}
          {!trace && !error && <p className="text-slate-500">Loading…</p>}
          {trace && (
            <>
              <pre className="max-h-96 overflow-y-auto whitespace-pre-wrap break-words font-mono text-xs leading-relaxed">
                {trace.content || "(no thinking trace returned)"}
              </pre>
              {trace.response && (
                <details className="mt-2">
                  <summary className="cursor-pointer text-slate-600">Final response text</summary>
                  <pre className="mt-1 max-h-60 overflow-y-auto whitespace-pre-wrap break-words font-mono text-xs">{trace.response}</pre>
                </details>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: AttemptCard**

```tsx
// web/src/components/AttemptCard.tsx
"use client";

import { OUTCOME_META } from "@/lib/outcome";
import type { RunSummary, TaskResult } from "@/lib/types";
import { TraceDrawer } from "./TraceDrawer";

function Stat({ label, value }: { label: string; value: string }) {
  return <span className="text-xs text-slate-500">{label} <span className="font-medium text-slate-700">{value}</span></span>;
}

export function AttemptCard({ taskId, result, run, selected, onSelect }: {
  taskId: string;
  result: TaskResult;
  run: RunSummary;
  selected: boolean;
  onSelect: () => void;
}) {
  const meta = OUTCOME_META[result.outcome];
  return (
    <div id={`${taskId}--${run.slug}`}
         className={`rounded-lg border p-3 ${selected ? "border-orange-400 ring-1 ring-orange-300" : "border-slate-200"}`}>
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" onClick={onSelect} className="font-medium hover:underline">{run.display_name}</button>
        <span className={`rounded px-1.5 py-0.5 text-xs font-medium ${meta.badgeClass}`}>{meta.label}</span>
        {result.legality === "illegal" && result.outcome !== "illegal" && (
          <span className="rounded bg-fuchsia-100 px-1.5 py-0.5 text-xs text-fuchsia-800">illegal move</span>
        )}
        {result.legality === "unparseable" && (
          <span className="rounded bg-slate-200 px-1.5 py-0.5 text-xs text-slate-700">not UCI</span>
        )}
      </div>
      <p className="mt-1 font-mono text-sm">
        {result.extracted ? result.extracted : <span className="italic text-slate-400">no answer</span>}
      </p>
      <div className="mt-1 flex flex-wrap gap-3">
        {result.cost_usd != null && <Stat label="cost" value={`$${result.cost_usd.toFixed(4)}`} />}
        {result.completion_tokens != null && <Stat label="tokens" value={result.completion_tokens.toLocaleString()} />}
        {result.latency_ms != null && <Stat label="time" value={`${(result.latency_ms / 1000).toFixed(1)}s`} />}
      </div>
      <TraceDrawer taskId={taskId} result={result} />
    </div>
  );
}
```

- [ ] **Step 4: Verify compile + commit**

Run: `cd web && npm run lint && npm run build` — expected: clean.

```bash
git add web/src/components
git commit -m "feat(web): lazy mount, attempt card, trace drawer"
```

---

### Task 4: PositionSection (board + question + metadata + attempts)

**Files:**
- Create: `web/src/components/PositionSection.tsx`

**Interfaces:**
- Consumes: `Board`, `AttemptCard`, `LazyMount` (used by the category page, not here), board lib, types.
- Produces: `<PositionSection task runs preselectRun />` where `task: TaskDetail`, `runs: RunSummary[]`, `preselectRun?: string`. The section's root element has `id={task.task_id}` (hash-anchor target).

- [ ] **Step 1: Implement**

```tsx
// web/src/components/PositionSection.tsx
"use client";

import { useEffect, useMemo, useState } from "react";
import { Board } from "./Board";
import { AttemptCard } from "./AttemptCard";
import { CORRECT_COLOR, MODEL_COLOR, diffOverlay, fenAfterMoves, mergeOverlays, primitivesToOverlay, sideToMove } from "@/lib/board";
import { getTraces } from "@/lib/data";
import type { RunSummary, TaskDetail } from "@/lib/types";

function ChoicePicks({ correct, picks }: { correct: string; picks: { name: string; letter: string }[] }) {
  return (
    <div className="flex gap-1 text-xs">
      {["A", "B", "C", "D"].map((letter) => {
        const names = picks.filter((p) => p.letter === letter).map((p) => p.name);
        return (
          <div key={letter}
               className={`flex-1 rounded border p-1 text-center ${letter === correct ? "border-emerald-500 bg-emerald-50" : "border-slate-200"}`}
               title={names.join(", ")}>
            <div className="font-mono">{letter}</div>
            <div className="text-slate-500">{names.length > 0 ? `${names.length} model${names.length > 1 ? "s" : ""}` : "—"}</div>
          </div>
        );
      })}
    </div>
  );
}

function EvalScale({ correct, picks }: { correct: number; picks: { name: string; value: number }[] }) {
  const buckets = [-400, -200, 0, 200, 400];
  return (
    <div className="flex gap-1 text-xs">
      {buckets.map((bucket) => {
        const names = picks.filter((p) => p.value === bucket).map((p) => p.name);
        return (
          <div key={bucket}
               className={`flex-1 rounded border p-1 text-center ${bucket === correct ? "border-emerald-500 bg-emerald-50" : "border-slate-200"}`}
               title={names.join(", ")}>
            <div className="font-mono">{bucket > 0 ? `+${bucket}` : bucket}</div>
            <div className="text-slate-500">{names.length > 0 ? `${names.length} model${names.length > 1 ? "s" : ""}` : "—"}</div>
          </div>
        );
      })}
    </div>
  );
}

export function PositionSection({ task, runs, preselectRun }: {
  task: TaskDetail;
  runs: RunSummary[];
  preselectRun?: string;
}) {
  const [selectedRun, setSelectedRun] = useState<string | null>(preselectRun ?? null);
  const [showCorrect, setShowCorrect] = useState(true);
  const [showPrompt, setShowPrompt] = useState(false);
  const [moveIndex, setMoveIndex] = useState(task.input_moves.length); // state-tracking: start at the final position

  const displayFen = task.input_moves.length > 0 ? fenAfterMoves(task.input_fen, task.input_moves, moveIndex) : task.input_fen;
  const orientation = sideToMove(task.input_fen);
  const selected = task.results.find((r) => r.run === selectedRun);

  const overlay = useMemo(() => mergeOverlays(
    showCorrect ? primitivesToOverlay(task.correct_primitives, CORRECT_COLOR) : { arrows: [], squareStyles: {} },
    selected ? primitivesToOverlay(selected.primitives, MODEL_COLOR) : { arrows: [], squareStyles: {} },
  ), [showCorrect, selected, task.correct_primitives]);

  // Speculative prefetch (spec: traces load during idle time for in-view positions).
  // LazyMount only mounts sections near the viewport, so mount ≈ in view.
  useEffect(() => {
    const prefetch = () => { getTraces(task.task_id).catch(() => undefined); };
    if ("requestIdleCallback" in window) {
      const handle = window.requestIdleCallback(prefetch);
      return () => window.cancelIdleCallback(handle);
    }
    const timeout = window.setTimeout(prefetch, 2000); // Safari has no requestIdleCallback
    return () => window.clearTimeout(timeout);
  }, [task.task_id]);

  const meta = task.metadata ?? {};
  const puzzleId = typeof meta.puzzle_id === "string" ? meta.puzzle_id : null;
  const themes = Array.isArray(meta.themes) ? (meta.themes as string[]) : [];
  const bestLine = typeof meta.best_line === "string" ? meta.best_line : null;
  const displayName = (slug: string) => runs.find((run) => run.slug === slug)?.display_name ?? slug;
  const evalPicks = task.results
    .filter((r) => r.primitives.type === "eval")
    .map((r) => ({ name: displayName(r.run), value: (r.primitives as { value: number }).value }));
  const choicePicks = task.results
    .filter((r) => r.primitives.type === "choice")
    .map((r) => ({ name: displayName(r.run), letter: (r.primitives as { letter: string }).letter }));

  return (
    <section id={task.task_id} className="scroll-mt-20 border-t border-slate-200 py-8">
      <h3 className="font-mono text-sm text-slate-500">{task.task_type}</h3>
      <div className="mt-3 grid gap-6 lg:grid-cols-[minmax(280px,420px)_1fr]">
        <div>
          <Board fen={displayFen} orientation={orientation} overlay={overlay} />
          {task.input_moves.length > 0 && (
            <div className="mt-2 flex items-center gap-2 text-sm">
              <button type="button" className="rounded border px-2" disabled={moveIndex === 0} onClick={() => setMoveIndex(moveIndex - 1)}>‹</button>
              <span className="font-mono">{moveIndex}/{task.input_moves.length} moves</span>
              <button type="button" className="rounded border px-2" disabled={moveIndex === task.input_moves.length} onClick={() => setMoveIndex(moveIndex + 1)}>›</button>
            </div>
          )}
          {selected?.primitives.type === "fen" && (
            <div className="mt-3">
              <p className="mb-1 text-xs font-medium text-slate-600">What this model imagined (differences in red):</p>
              <Board fen={selected.primitives.fen} orientation={orientation} overlay={diffOverlay(selected.primitives.diff_squares)} maxWidth={300} />
            </div>
          )}
          <label className="mt-2 flex items-center gap-1 text-xs text-slate-600">
            <input type="checkbox" checked={showCorrect} onChange={(e) => setShowCorrect(e.target.checked)} />
            show correct answer (green)
          </label>
        </div>
        <div>
          <p className="whitespace-pre-wrap text-sm">{task.question.replace("CONTEXT_PLACEHOLDER", "").split("Analyze step by step")[0].trim()}</p>
          <p className="mt-2 text-sm">Correct answer: <span className="font-mono font-semibold text-emerald-700">{task.correct_answer}</span></p>
          <button type="button" className="mt-1 text-xs underline decoration-dotted text-slate-500" onClick={() => setShowPrompt(!showPrompt)}>
            {showPrompt ? "hide full prompt" : "show full prompt"}
          </button>
          {showPrompt && <pre className="mt-1 max-h-60 overflow-y-auto whitespace-pre-wrap rounded bg-slate-50 p-2 font-mono text-xs">{task.resolved_prompt}</pre>}
          <div className="mt-2 flex flex-wrap gap-3 text-xs text-slate-500">
            {puzzleId && <a className="underline" href={`https://lichess.org/training/${puzzleId}`} target="_blank" rel="noreferrer">Lichess puzzle {puzzleId}</a>}
            {typeof meta.rating === "number" && <span>puzzle rating {meta.rating}</span>}
            {typeof meta.depth === "number" && <span>Stockfish depth {meta.depth}</span>}
            {themes.map((theme) => <span key={theme} className="rounded bg-slate-100 px-1.5">{theme}</span>)}
          </div>
          {bestLine && <p className="mt-1 font-mono text-xs text-slate-500">engine line: {bestLine}</p>}
          {evalPicks.length > 0 && (
            <div className="mt-3"><EvalScale correct={Number(task.correct_answer)} picks={evalPicks} /></div>
          )}
          {choicePicks.length > 0 && (
            <div className="mt-3"><ChoicePicks correct={task.correct_answer.trim().toUpperCase()} picks={choicePicks} /></div>
          )}
          <div className="mt-4 grid gap-2 md:grid-cols-2">
            {task.results.map((result) => (
              <AttemptCard key={result.run} taskId={task.task_id} result={result}
                           run={runs.find((run) => run.slug === result.run)!}
                           selected={selectedRun === result.run}
                           onSelect={() => setSelectedRun(selectedRun === result.run ? null : result.run)} />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
```

Note: the question-text trim (`split("Analyze step by step")[0]`) shows the human-readable ask; the full untrimmed prompt lives behind the toggle. Semantic MCQ options are part of `question` and therefore stay visible — correct, since the options ARE the task.

- [ ] **Step 2: Verify compile + commit**

Run: `cd web && npm run lint && npm run build` — expected: clean.

```bash
git add web/src/components/PositionSection.tsx
git commit -m "feat(web): position section with overlays, stepper, eval scale, fen diff"
```

---

### Task 5: Category pages

**Files:**
- Create: `web/src/app/category/[slug]/page.tsx`, `web/src/components/CategoryClient.tsx`

**Interfaces:**
- Consumes: `getCategory`, `getIndex`, `PositionSection`, `LazyMount`, `CATEGORY_ORDER`.
- Produces: routes `/category/<slug>` ×5 with hash-anchor deep links + `?run=` preselect (the exact URL contract exhibits and heatmap cells rely on: `/category/<slug>?run=<run_slug>#<task_id>`).

- [ ] **Step 1: Static route shell**

```tsx
// web/src/app/category/[slug]/page.tsx
import { Suspense } from "react";
import { CategoryClient } from "@/components/CategoryClient";
import { CATEGORY_ORDER } from "@/lib/outcome";

export function generateStaticParams() {
  return CATEGORY_ORDER.map(({ slug }) => ({ slug }));
}

export default async function CategoryPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  return (
    <Suspense>
      <CategoryClient slug={slug} />
    </Suspense>
  );
}
```

(`useSearchParams` in the client component requires the Suspense boundary under static export.)

- [ ] **Step 2: Client component**

```tsx
// web/src/components/CategoryClient.tsx
"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { getCategory, getIndex } from "@/lib/data";
import { CATEGORY_ORDER } from "@/lib/outcome";
import type { CategoryData, IndexData } from "@/lib/types";
import { LazyMount } from "./LazyMount";
import { PositionSection } from "./PositionSection";

export function CategoryClient({ slug }: { slug: string }) {
  const [data, setData] = useState<CategoryData | null>(null);
  const [index, setIndex] = useState<IndexData | null>(null);
  const [initialHash] = useState(() => (typeof window === "undefined" ? "" : window.location.hash.slice(1)));
  const preselectRun = useSearchParams().get("run") ?? undefined;

  useEffect(() => {
    Promise.all([getCategory(slug), getIndex()]).then(([category, idx]) => {
      setData(category);
      setIndex(idx);
    });
  }, [slug]);

  useEffect(() => {
    if (data && initialHash) document.getElementById(initialHash)?.scrollIntoView();
  }, [data, initialHash]);

  if (!data || !index) return <p className="p-8 text-slate-500">Loading positions…</p>;
  const categoryName = CATEGORY_ORDER.find((c) => c.slug === slug)?.name ?? data.category;

  return (
    <main className="mx-auto max-w-6xl px-4 py-8">
      <h1 className="text-2xl font-semibold">{categoryName}</h1>
      <p className="mt-1 text-sm text-slate-600">{data.tasks.length} positions · 16 model attempts each · one position per task type</p>
      <nav className="sticky top-0 z-10 -mx-4 mt-4 overflow-x-auto border-b border-slate-200 bg-white/95 px-4 py-2 backdrop-blur">
        <ul className="flex gap-3 whitespace-nowrap text-xs">
          {data.tasks.map((task) => (
            <li key={task.task_id}>
              <a className="text-slate-600 underline-offset-2 hover:underline" href={`#${task.task_id}`}>
                {task.task_type.replace(/^(structural|motifs|short_tactics|position_judgement|semantic)_/, "")}
              </a>
            </li>
          ))}
        </ul>
      </nav>
      {data.tasks.map((task) => (
        <LazyMount key={task.task_id} eager={task.task_id === initialHash || data.tasks.indexOf(task) === 0}>
          <PositionSection task={task} runs={index.runs} preselectRun={preselectRun} />
        </LazyMount>
      ))}
    </main>
  );
}
```

Known trade-off (fine for v1): with lazy mounting, hash-scroll lands near — not pixel-exact on — the target section, because unmounted sections above it use the `minHeight` placeholder. `eager` on the target section plus the generous `rootMargin` keeps this acceptable; do not add scroll-correction machinery.

- [ ] **Step 3: Verify in the browser**

Run: `cd web && npm run dev`, open `http://localhost:3000/category/short-tactics`. Check: nav lists 24 positions; boards render oriented to side-to-move; clicking a model card overlays an orange arrow next to the green correct arrow; a trace opens (network tab shows `traces/<task_id>.json` fetched only on open); `http://localhost:3000/category/structural#structural_state_tracking_long_0049` scrolls to the stepper section and ‹/› steps the board; `?run=anthropic_claude-haiku-4.5` preselects that model's card.

- [ ] **Step 4: Build + commit**

Run: `cd web && npm run lint && npm run build` — expected: `/category/[slug]` shows 5 static paths.

```bash
git add web/src/app/category web/src/components/CategoryClient.tsx
git commit -m "feat(web): category pages with position nav, deep links, lazy sections"
```

---

### Task 6: Home page (framing, summary table, heatmap, exhibits)

**Files:**
- Create: `web/src/components/Heatmap.tsx`, `web/src/components/RunSummaryTable.tsx`, `web/src/exhibits.ts`
- Modify: `web/src/app/page.tsx` (replace scaffold content)

**Interfaces:**
- Consumes: `getIndex`, `OUTCOME_META`, `CATEGORY_ORDER`.
- Produces: the `/` route; `EXHIBITS: { title, blurb, href }[]`.

- [ ] **Step 1: Exhibits config** (chess claims flagged for Aron's review — his labels are ground truth)

```ts
// web/src/exhibits.ts
export interface Exhibit { title: string; blurb: string; href: string }

// NOTE for reviewer (Aron): blurb wording makes chess claims — please verify before launch.
export const EXHIBITS: Exhibit[] = [
  {
    title: "266,000 characters of re-examining the king's desperation",
    blurb: "Gemini 3.1 Pro burns its entire 32K-token budget circling one defensive puzzle — and never produces an answer.",
    href: "/category/short-tactics?run=google_gemini-3.1-pro-preview-thinking#short_tactics_theme_defensiveMove_0024",
  },
  {
    title: "A confidently announced mate that doesn't exist",
    blurb: "DeepSeek R1 narrates a forced mate-in-two and commits to it. The line is unsound — the real answer was the queen check.",
    href: "/category/short-tactics?run=deepseek_deepseek-r1-thinking#short_tactics_theme_mateIn2_0013",
  },
  {
    title: "A rook that phases through its own pawn",
    blurb: "Haiku 4.5 answers f8f5 — through White's own pawn on f7. One of 41 illegal moves models played in this smoke set.",
    href: "/category/short-tactics?run=anthropic_claude-haiku-4.5-thinking#short_tactics_theme_advancedPawn_0001",
  },
  {
    title: "The imagined board drifts from reality",
    blurb: "Asked to track a long move sequence, Haiku's mental board ends with king and rook on the wrong squares — shown side by side with the truth.",
    href: "/category/structural?run=anthropic_claude-haiku-4.5#structural_state_tracking_long_0049",
  },
];
```

- [ ] **Step 2: Summary table + heatmap components**

```tsx
// web/src/components/RunSummaryTable.tsx
"use client";

import type { RunSummary } from "@/lib/types";

export function RunSummaryTable({ runs }: { runs: RunSummary[] }) {
  const sorted = [...runs].sort((a, b) => b.n_correct - a.n_correct);
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b text-left text-xs uppercase tracking-wide text-slate-500">
            <th className="py-2 pr-4">Model</th><th className="pr-4">Correct</th><th className="pr-4">Capped</th>
            <th className="pr-4">Illegal</th><th className="pr-4">Avg tokens</th><th className="pr-4">Cost</th><th>Traces</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((run) => (
            <tr key={run.slug} className="border-b border-slate-100">
              <td className="py-2 pr-4"><a className="font-medium hover:underline" href={`/model/${run.slug}`}>{run.display_name}</a></td>
              <td className="pr-4 font-mono">{run.n_correct}/{run.n_results}</td>
              <td className="pr-4 font-mono">{run.n_capped || "—"}</td>
              <td className="pr-4 font-mono">{run.n_illegal || "—"}</td>
              <td className="pr-4 font-mono">{run.avg_completion_tokens?.toLocaleString() ?? "—"}</td>
              <td className="pr-4 font-mono">{run.total_cost_usd != null ? `$${run.total_cost_usd.toFixed(2)}` : "n/a"}</td>
              <td className="text-xs text-slate-500">{Object.entries(run.thinking_sources).map(([k, v]) => `${k}×${v}`).join(" ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

```tsx
// web/src/components/Heatmap.tsx
"use client";

import { OUTCOME_META } from "@/lib/outcome";
import type { IndexData } from "@/lib/types";

export function Heatmap({ index }: { index: IndexData }) {
  return (
    <div className="overflow-x-auto">
      <table className="border-separate border-spacing-px">
        <thead>
          <tr>
            <th className="sticky left-0 bg-white pr-2 text-left text-xs font-normal text-slate-500">model \ task</th>
            {index.tasks.map((task) => (
              <th key={task.task_id} className="p-0 text-[9px] font-normal text-slate-400">
                <div className="h-16 w-4 [writing-mode:vertical-rl]" title={task.task_type}>{task.task_type.slice(0, 22)}</div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {index.runs.map((run) => (
            <tr key={run.slug}>
              <th className="sticky left-0 bg-white pr-2 text-left text-xs font-normal text-slate-600 whitespace-nowrap">{run.display_name}</th>
              {index.tasks.map((task) => {
                const outcome = task.outcomes[run.slug];
                return (
                  <td key={task.task_id} className="p-0">
                    <a href={`/category/${task.category_slug}?run=${run.slug}#${task.task_id}`}
                       title={`${run.display_name} · ${task.task_type}: ${OUTCOME_META[outcome].label}`}
                       className={`block h-4 w-4 ${OUTCOME_META[outcome].cellClass} hover:opacity-70`} />
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-2 flex flex-wrap gap-3 text-xs text-slate-600">
        {Object.entries(OUTCOME_META).map(([key, meta]) => (
          <span key={key} className="flex items-center gap-1"><span className={`inline-block h-3 w-3 ${meta.cellClass}`} />{meta.label}</span>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Home page**

```tsx
// web/src/app/page.tsx
"use client";

import { useEffect, useState } from "react";
import { getIndex } from "@/lib/data";
import type { IndexData } from "@/lib/types";
import { Heatmap } from "@/components/Heatmap";
import { RunSummaryTable } from "@/components/RunSummaryTable";
import { EXHIBITS } from "@/exhibits";

export default function Home() {
  const [index, setIndex] = useState<IndexData | null>(null);
  useEffect(() => { getIndex().then(setIndex); }, []);

  return (
    <main className="mx-auto max-w-6xl px-4 py-10">
      <h1 className="text-3xl font-semibold">ChessQA Explorer</h1>
      <p className="mt-3 max-w-3xl text-slate-700">
        Sixteen frontier-model configurations, fifty chess questions, every answer and every unedited thought
        stream. Built on the <a className="underline" href="https://arxiv.org/abs/2510.23948">ChessQA benchmark</a> by
        CSSLab, University of Toronto.
      </p>
      <div className="mt-4 max-w-3xl rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
        <strong>What this is (and isn't):</strong> a browser for model behavior, not a leaderboard. Each model saw a
        50-task smoke sample — one position per task type — so category-level numbers are anecdotes with wide error
        bars. Counts are shown everywhere instead of percentages for exactly that reason.
      </div>
      {index ? (
        <>
          <h2 className="mt-10 text-xl font-semibold">The fleet</h2>
          <RunSummaryTable runs={index.runs} />
          <h2 className="mt-10 text-xl font-semibold">Every attempt at a glance</h2>
          <p className="mb-2 text-sm text-slate-600">Click any cell to see that model's answer and thoughts on that position.</p>
          <Heatmap index={index} />
        </>
      ) : (
        <p className="mt-10 text-slate-500">Loading results…</p>
      )}
      <h2 className="mt-10 text-xl font-semibold">Exhibits</h2>
      <div className="mt-3 grid gap-4 md:grid-cols-2">
        {EXHIBITS.map((exhibit) => (
          <a key={exhibit.href} href={exhibit.href} className="rounded-lg border border-slate-200 p-4 hover:border-slate-400">
            <h3 className="font-medium">{exhibit.title}</h3>
            <p className="mt-1 text-sm text-slate-600">{exhibit.blurb}</p>
          </a>
        ))}
      </div>
    </main>
  );
}
```

- [ ] **Step 4: Verify in browser + build + commit**

Run: `cd web && npm run dev` → `http://localhost:3000`. Check: table sorted by correct count with `n/N` formatting, sonnet-5 shows both variants; heatmap is 16×50 with legend; each exhibit link lands on the right position with the right model preselected.
Run: `npm run lint && npm run build` — clean.

```bash
git add web/src
git commit -m "feat(web): home page with framing, summary table, heatmap, exhibits"
```

---

### Task 7: Model pages, About page, layout/nav/footer

**Files:**
- Create: `web/src/app/model/[slug]/page.tsx`, `web/src/components/ModelClient.tsx`, `web/src/app/about/page.tsx`
- Modify: `web/src/app/layout.tsx`

**Interfaces:**
- Consumes: `getIndex`, `getCategory`, `OUTCOME_META`, `CATEGORY_ORDER`.
- Produces: `/model/<run_slug>` ×16, `/about`, shared nav + attribution footer.

- [ ] **Step 1: Model route (slugs come from the exported index at build time)**

```tsx
// web/src/app/model/[slug]/page.tsx
import { promises as fs } from "fs";
import path from "path";
import { ModelClient } from "@/components/ModelClient";

export async function generateStaticParams() {
  const raw = await fs.readFile(path.join(process.cwd(), "public", "data", "index.json"), "utf-8");
  const index = JSON.parse(raw) as { runs: { slug: string }[] };
  return index.runs.map((run) => ({ slug: run.slug }));
}

export default async function ModelPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  return <ModelClient slug={slug} />;
}
```

- [ ] **Step 2: Model client (fetches index + all five category files, ~100 KB wire total)**

```tsx
// web/src/components/ModelClient.tsx
"use client";

import { useEffect, useState } from "react";
import { getCategory, getIndex } from "@/lib/data";
import { CATEGORY_ORDER, OUTCOME_META } from "@/lib/outcome";
import type { IndexData, Outcome, RunSummary } from "@/lib/types";

interface AttemptRow { task_id: string; task_type: string; category_slug: string; extracted: string | null; outcome: Outcome }

export function ModelClient({ slug }: { slug: string }) {
  const [run, setRun] = useState<RunSummary | null>(null);
  const [index, setIndex] = useState<IndexData | null>(null);
  const [attempts, setAttempts] = useState<AttemptRow[]>([]);
  const [filter, setFilter] = useState<Outcome | "all">("all");

  useEffect(() => {
    getIndex().then((idx) => {
      setIndex(idx);
      setRun(idx.runs.find((r) => r.slug === slug) ?? null);
    });
    Promise.all(CATEGORY_ORDER.map(({ slug: cat }) => getCategory(cat))).then((categories) => {
      setAttempts(categories.flatMap((category) =>
        category.tasks.map((task) => {
          const result = task.results.find((r) => r.run === slug)!;
          return { task_id: task.task_id, task_type: task.task_type, category_slug: category.slug,
                   extracted: result.extracted, outcome: result.outcome };
        })));
    });
  }, [slug]);

  if (!run || !index) return <p className="p-8 text-slate-500">Loading…</p>;
  const shown = attempts.filter((a) => filter === "all" || a.outcome === filter);
  const byCategory = CATEGORY_ORDER.map(({ slug: cat, name }) => {
    const rows = attempts.filter((a) => a.category_slug === cat);
    return { name, correct: rows.filter((a) => a.outcome === "correct").length, total: rows.length };
  });

  return (
    <main className="mx-auto max-w-4xl px-4 py-8">
      <h1 className="text-2xl font-semibold">{run.display_name}</h1>
      <dl className="mt-3 grid grid-cols-2 gap-2 text-sm md:grid-cols-3">
        <div><dt className="text-slate-500">Model id</dt><dd className="font-mono">{run.model}</dd></div>
        <div><dt className="text-slate-500">Backend</dt><dd className="font-mono">{run.backend}</dd></div>
        <div><dt className="text-slate-500">Reasoning payload</dt><dd className="font-mono">{run.reasoning_config ? JSON.stringify(run.reasoning_config) : "none"}</dd></div>
        <div><dt className="text-slate-500">Run date</dt><dd>{run.started_at?.slice(0, 10) ?? "—"}</dd></div>
        <div><dt className="text-slate-500">Harness commit</dt><dd className="font-mono">{run.git_commit?.slice(0, 8) ?? "—"}</dd></div>
        <div><dt className="text-slate-500">Total cost</dt><dd className="font-mono">{run.total_cost_usd != null ? `$${run.total_cost_usd.toFixed(2)}` : "n/a"}</dd></div>
      </dl>
      <div className="mt-4 flex flex-wrap gap-2 text-sm">
        {byCategory.map((c) => (
          <span key={c.name} className="rounded border border-slate-200 px-2 py-1">{c.name}: <span className="font-mono">{c.correct}/{c.total}</span></span>
        ))}
      </div>
      <div className="mt-6 flex gap-2 text-xs">
        {(["all", "correct", "wrong", "illegal", "capped", "format_error"] as const).map((option) => (
          <button key={option} type="button" onClick={() => setFilter(option)}
                  className={`rounded border px-2 py-1 ${filter === option ? "border-slate-700 font-medium" : "border-slate-200 text-slate-500"}`}>
            {option === "all" ? "all" : OUTCOME_META[option].label.toLowerCase()}
          </button>
        ))}
      </div>
      <ul className="mt-3 divide-y divide-slate-100 text-sm">
        {shown.map((attempt) => (
          <li key={attempt.task_id} className="flex items-center justify-between gap-2 py-2">
            <a className="hover:underline" href={`/category/${attempt.category_slug}?run=${slug}#${attempt.task_id}`}>{attempt.task_type}</a>
            <span className="flex items-center gap-2">
              <span className="font-mono text-xs text-slate-500">{attempt.extracted ?? "—"}</span>
              <span className={`rounded px-1.5 py-0.5 text-xs ${OUTCOME_META[attempt.outcome].badgeClass}`}>{OUTCOME_META[attempt.outcome].label}</span>
            </span>
          </li>
        ))}
      </ul>
    </main>
  );
}
```

- [ ] **Step 3: Layout with nav + attribution footer, and the About page**

```tsx
// web/src/app/layout.tsx — keep the scaffold's font/global.css imports, replace body content
import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "ChessQA Explorer",
  description: "How frontier LLMs answer chess questions — every answer, every unedited thought stream.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="text-slate-900 antialiased">
        <header className="border-b border-slate-200">
          <nav className="mx-auto flex max-w-6xl items-center gap-5 px-4 py-3 text-sm">
            <Link href="/" className="font-semibold">ChessQA Explorer</Link>
            <Link href="/category/structural" className="text-slate-600 hover:text-slate-900">Structural</Link>
            <Link href="/category/motifs" className="text-slate-600 hover:text-slate-900">Motifs</Link>
            <Link href="/category/short-tactics" className="text-slate-600 hover:text-slate-900">Short Tactics</Link>
            <Link href="/category/position-judgement" className="text-slate-600 hover:text-slate-900">Position Judgement</Link>
            <Link href="/category/semantic" className="text-slate-600 hover:text-slate-900">Semantic</Link>
            <Link href="/about" className="ml-auto text-slate-600 hover:text-slate-900">About</Link>
          </nav>
        </header>
        {children}
        <footer className="mt-16 border-t border-slate-200 py-6 text-center text-xs text-slate-500">
          Benchmark: <a className="underline" href="https://github.com/CSSLab/chessqa-benchmark">ChessQA</a> by CSSLab,
          University of Toronto (<a className="underline" href="https://arxiv.org/abs/2510.23948">arXiv:2510.23948</a>, MIT).
          Harness &amp; results: <a className="underline" href="https://github.com/Ellipsoul/chessqa-benchmark">Ellipsoul/chessqa-benchmark</a>.
          Costs shown are real measured API spend.
        </footer>
      </body>
    </html>
  );
}
```

About page — static prose covering (write as real copy, not lorem): what ChessQA measures (five categories of ascending abstraction), what this smoke campaign was (50 tasks/model, 16 configs, July 2026), how answers were scored (exact/set match + the outcome taxonomy incl. the export-time legality check), trace fidelity tiers (`full_text`/`summary`/`plain`/`none` and why Anthropic ≥4.7 models are summarized), the roadmap (full 3,500-task run; reasoning-trace root-cause analysis), and attribution.

- [ ] **Step 4: Verify + build + commit**

Run: `cd web && npm run dev` → check `/model/anthropic_claude-sonnet-5-thinking` (provenance card, category counts, filters work; dots in slug resolve), `/about`. Then `npm run lint && npm run build` — expected: 16 static model paths.

```bash
git add web/src
git commit -m "feat(web): model pages, about page, nav and attribution footer"
```

---

### Task 8: Visual polish, full verification, deploy

**Files:**
- Modify: whatever the polish pass touches under `web/src/` (styling only — no logic changes in this task)

- [ ] **Step 1: Polish pass** — invoke the `frontend-design` skill and restyle within these bounds: chess-adjacent restrained aesthetic, light + dark theme, heatmap colors must keep ≥3:1 contrast against the page background and stay distinguishable for color-blind users (the fuchsia/emerald/amber/rose/slate palette above was chosen to differ in lightness, not only hue — preserve that property), mobile: boards full-width, heatmap and tables scroll horizontally inside their own containers.

- [ ] **Step 2: Full local verification**

```bash
cd web && npx vitest run && npm run lint && npm run build
npx serve out  # or: python3 -m http.server -d out 8000
```
Walk every route type once (home, one category with hash+run deep link, one model, about) on the served static build — not the dev server — to catch export-only issues.

- [ ] **Step 3: Deploy to Vercel**

Vercel dashboard → New Project → import the `chess-benchmark-showcase` GitHub repo (Aron creates the remote if it doesn't exist yet) → repo root, framework Next.js, defaults otherwise (no env vars). Deploy the feature branch as a preview first.

- [ ] **Step 4: Verify the spec's compression checklist item on the preview URL**

```bash
curl -sI -H 'Accept-Encoding: br, gzip' https://<preview-url>/data/index.json | grep -i content-encoding
curl -so /dev/null -w 'wire bytes: %{size_download}\n' --compressed https://<preview-url>/data/traces/short_tactics_theme_defensiveMove_0024.json
```
Expected: `content-encoding: br` (gzip acceptable — budget holds either way); the worst-case trace file transfers ≤ ~250 KB. Record both numbers in `docs/superpowers/specs/2026-07-13-chessqa-explorer-design.md` (replace the "implementation checklist item" sentence with the measured result).

- [ ] **Step 5: PR**

```bash
git add -A && git commit -m "feat: visual polish pass"
```
Merge/PR flow in the showcase repo is Aron's call (it may not have a remote yet). If a PR is opened there, it targets the showcase repo — the `--repo Ellipsoul/chessqa-benchmark` rule is for the benchmark repo only. After merge, promote the Vercel deployment to production.
