"""Per-model smoke analysis + weighted full-run cost projection.

Usage: .venv/bin/python eval/smoke_analysis.py results/<file>.jsonl [...]

Method (docs/cost-baseline-2026-07-06.md): weight each task_type's smoke completion
tokens by its true benchmark count, add the fixed 0.8M full-run input tokens, price at
gateway list ($/M). Anthropic native-routed runs carry usage.cost == 0, so their smoke
cost is computed as tokens x list price instead. ERROR rows (no usage) are excluded from
token stats and reported separately — a nonzero error_rows count means the projection is
a lower bound (see docs/model-trials/2026-07-12-smoke-campaign.md, "the 340s wall").
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FULL_RUN_INPUT_TOKENS = 0.8e6

# Gateway list prices, $/M tokens (input, output). Snapshot 2026-07-12 via
# GET https://ai-gateway.vercel.sh/v1/models — refresh when the fleet changes.
PRICES = {
    "deepseek/deepseek-v4-pro": (0.435, 0.87),
    "xai/grok-4.3": (1.25, 2.50),
    "xai/grok-4.5": (2.00, 6.00),
    "minimax/minimax-m3": (0.30, 1.20),
    "openai/gpt-5.4-mini": (0.75, 4.50),
    "alibaba/qwen3.7-max": (1.25, 3.75),
    "moonshotai/kimi-k2.6": (0.95, 4.00),
    "openai/gpt-5.6-sol": (5.00, 30.00),
    "anthropic/claude-opus-4.6": (5.00, 25.00),
    "anthropic/claude-sonnet-5": (2.00, 10.00),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "google/gemini-3.5-flash": (1.50, 9.00),
    "deepseek/deepseek-r1": (1.35, 5.40),
    "google/gemini-3.1-pro-preview": (2.00, 12.00),
    "meta/llama-4-maverick": (0.24, 0.97),
}


def type_counts() -> Counter:
    counts = Counter()
    for f in (REPO_ROOT / "benchmark").glob("*.jsonl"):
        with open(f) as fh:
            for line in fh:
                counts[json.loads(line)["task_type"]] += 1
    return counts


def slug_from_filename(path: Path) -> str:
    stem = path.stem
    for suf in ("-openrouter", "-fmt2", "-piecearr", "-thinking", "-verbose-cot"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break  # only the outermost run suffix; gpt-5.1-thinking keeps its name
    return stem.replace("_", "/", 1)


def analyze(path: Path, counts: Counter) -> dict:
    with open(path) as fh:
        all_rows = [json.loads(line) for line in fh]
    slug = slug_from_filename(path)
    pin, pout = PRICES[slug]
    rows = [r for r in all_rows if (r["inference"].get("usage") or {}).get("completion_tokens")]
    n = len(rows)
    inf = [r["inference"] for r in rows]
    usage_cost = sum(i["usage"].get("cost") or 0 for i in inf)
    ptoks = sum(i["usage"]["prompt_tokens"] for i in inf)
    ctoks = sum(i["usage"]["completion_tokens"] for i in inf)
    native = usage_cost == 0
    smoke_cost = (ptoks * pin + ctoks * pout) / 1e6 if native else usage_cost

    proj_out = 0
    covered = set()
    for r in rows:
        proj_out += counts[r["task_type"]] * r["inference"]["usage"]["completion_tokens"]
        covered.add(r["task_type"])
    proj_cost = (FULL_RUN_INPUT_TOKENS * pin + proj_out * pout) / 1e6

    by_cat = defaultdict(lambda: [0, 0])
    for r in rows:
        cell = by_cat[r["task_category"]]
        cell[1] += 1
        cell[0] += bool(r["inference"]["is_correct"])
    src = Counter(i.get("thinking_source") or "none" for i in inf)
    err = Counter(i.get("error_type") or ("correct" if i["is_correct"] else "?") for i in inf)

    return dict(
        file=path.name, model=slug, n=n, error_rows=len(all_rows) - n,
        native_priced=native, smoke_cost=round(smoke_cost, 4),
        prompt_tokens=ptoks, completion_tokens=ctoks,
        accuracy=round(sum(bool(i["is_correct"]) for i in inf) / n, 3) if n else None,
        by_category={k: f"{v[0]}/{v[1]}" for k, v in sorted(by_cat.items())},
        thinking_source=dict(src), error_types=dict(err),
        max_token_reached=err.get("max_token_reached", 0),
        projected_full_out_tokens_M=round(proj_out / 1e6, 2),
        projected_full_run_cost=round(proj_cost, 2),
        types_missing=len(counts) - len(covered),
    )


if __name__ == "__main__":
    weights = type_counts()
    print(json.dumps([analyze(Path(p), weights) for p in sys.argv[1:]], indent=1))
