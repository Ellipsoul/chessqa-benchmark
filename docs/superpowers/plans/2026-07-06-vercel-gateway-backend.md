# Vercel AI Gateway Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Vercel AI Gateway the eval runner's default transport while keeping OpenRouter selectable via `--backend`, and upgrade thinking-trace extraction to record the gateway's typed fidelity signal (`text` / `summary` / `encrypted`).

**Architecture:** All changes live in `eval/run_openrouter.py` — the transport surface is one method (`call_model`) plus key loading. A `--backend` flag selects endpoint/auth/payload behavior; a new module-level `extract_thinking(message)` helper replaces the inline reasoning-extraction chain and returns `(thinking_content, thinking_source)`, with `thinking_source` persisted per task in the results JSONL. OpenRouter-only request fields (`usage.include`, hardcoded provider-order pins) are gated behind `backend == "openrouter"`.

**Tech Stack:** Python 3.11, `requests`, Vercel AI Gateway OpenAI-compatible endpoint (`https://ai-gateway.vercel.sh/v1/chat/completions`, verified 2026-07-06 against docs rev. 2026-05-11).

## Global Constraints

- **Scoring, extraction, prompts, resume logic untouched.** Only transport + trace capture change.
- **Backend affects results filenames:** OpenRouter runs get an `-openrouter` suffix (via `_build_variant_suffix`) so the two backends' result files never collide; gateway (default) filenames stay bare. No existing results files exist, so nothing breaks.
- **Auth:** gateway key from `AI_GATEWAY_API_KEY` (fallback `VERCEL_OIDC_TOKEN`); OpenRouter from `OPENROUTER_API_KEY` env (new, matches README's long-standing claim) falling back to the legacy `../keys/api_keys.json`.
- **Fail fast with actionable message** when no credential is found for the chosen backend.
- **`call_model` returns a 4-tuple** `(content, thinking_content, thinking_source, usage)`; both call sites and the `inference` dict updated together.
- **Docs updated in the same PR:** README (install/run/deviations) and CLAUDE.md (commands/API-key notes).

---

### Task 1: Backend plumbing in `eval/run_openrouter.py`

**Files:**
- Modify: `eval/run_openrouter.py`

**Interfaces:**
- Produces: `BACKEND_URLS: dict[str, str]`; `resolve_api_key(backend: str) -> str`; `extract_thinking(message: dict) -> tuple[str, str]`; `_build_variant_suffix(add_context, format_example_group, backend) -> str`; `OpenrouterInferencer.__init__(..., backend: str = "vercel-gateway", ...)`; `call_model -> tuple[str, str, str, dict]`; CLI flag `--backend {vercel-gateway,openrouter}` default `vercel-gateway`.

- [x] **Step 1: Add `os` import, `BACKEND_URLS`, `resolve_api_key`, `extract_thinking`**

```python
import os  # with the other stdlib imports

BACKEND_URLS = {
    "vercel-gateway": "https://ai-gateway.vercel.sh/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}

def resolve_api_key(backend: str) -> str:
    if backend == "vercel-gateway":
        api_key = os.environ.get("AI_GATEWAY_API_KEY") or os.environ.get("VERCEL_OIDC_TOKEN")
        if not api_key:
            raise SystemExit("... set AI_GATEWAY_API_KEY (or run `vercel env pull`) ...")
        return api_key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        return api_key
    keys_path = Path(__file__).parent.parent.parent / "keys" / "api_keys.json"
    try:
        with open(keys_path) as keys_file:
            api_key = json.load(keys_file).get("openrouter_api_key")
    except FileNotFoundError:
        api_key = None
    if not api_key:
        raise SystemExit("... set OPENROUTER_API_KEY or provide ../keys/api_keys.json ...")
    return api_key

def extract_thinking(message: dict) -> tuple[str, str]:
    """Returns (thinking_content, thinking_source) where source is one of
    full_text | summary | untyped | plain | encrypted_only | none."""
    details = message.get("reasoning_details") or []
    text_parts = [d.get("text") for d in details
                  if isinstance(d, dict) and d.get("type") == "reasoning.text" and d.get("text")]
    summary_parts = [d.get("summary") for d in details
                     if isinstance(d, dict) and d.get("type") == "reasoning.summary" and d.get("summary")]
    untyped_parts = [d.get("text") for d in details
                     if isinstance(d, dict) and "type" not in d and d.get("text")]
    untyped_parts += [d for d in details if isinstance(d, str)]
    if text_parts:
        return "\n".join(text_parts), "full_text"
    if summary_parts:
        return "\n".join(summary_parts), "summary"
    if untyped_parts:
        return "\n".join(untyped_parts), "untyped"
    if message.get("reasoning"):
        return str(message["reasoning"]), "plain"
    if any(isinstance(d, dict) and d.get("type") == "reasoning.encrypted" for d in details):
        return "", "encrypted_only"
    return "", "none"
```

- [x] **Step 2: Extend `_build_variant_suffix` with `backend` param** (emit `"openrouter"` part when backend == "openrouter"); update its 3 call sites in `main` and the `variant_suffix` construction.

- [x] **Step 3: Thread `backend` through `OpenrouterInferencer.__init__` (default `"vercel-gateway"`), `process_single_task` args tuple, and `run_inference`'s parallel `args_list`.** `__init__` sets `self.url = BACKEND_URLS[backend]` and `self.api_key = resolve_api_key(backend)`.

- [x] **Step 4: Rework `call_model` request/response:** gate `data["usage"] = {"include": True}` and the 3 provider-order pins behind `self.backend == "openrouter"`; keep `reasoning: {"effort": "medium"}` for both (same schema). Replace the inline reasoning-extraction chain with `extract_thinking(message)`; return `(content, thinking_content, thinking_source, usage)`.

- [x] **Step 5: Update both call sites** (`process_single_task`, sequential branch of `run_inference`) to unpack 4 values and add `"thinking_source": thinking_source` to the `inference` dict.

- [x] **Step 6: CLI + metadata:** add `--backend` (choices `["vercel-gateway", "openrouter"]`, default `"vercel-gateway"`), pass to inferencer, add `"backend": args.backend` to stats metadata. Update module + affected docstrings (API-key paragraph, `call_model`, `_build_variant_suffix`).

- [x] **Step 7: Gate** — `ruff check`, `compileall`, `--help` shows `--backend`.

### Task 2: Mock-transport verification + optional live smoke

**Files:**
- Scratch: `$SCRATCHPAD/verify_backends.py`

- [x] **Step 1: Write mock test** — monkeypatch `requests.post` to capture the payload and return canned responses; assert:
  1. gateway payload has no `usage`/`provider` keys, has `reasoning` when thinking enabled, URL is the gateway URL;
  2. openrouter payload keeps `usage.include` + provider pin for `qwen/qwen3-next-80b-a3b-thinking`, URL is OpenRouter's;
  3. typed Anthropic response (`reasoning.text` + signature) → `("…", "full_text")`;
  4. typed OpenAI response (`reasoning.summary` + `reasoning.encrypted`) → `("…", "summary")`;
  5. legacy OpenRouter shapes (plain `reasoning` string; untyped `reasoning_details[].text`) → `plain` / `untyped`;
  6. `resolve_api_key` failure paths raise SystemExit with the right hint.
- [x] **Step 2: Run it** — expected `ALL BACKEND CHECKS PASS`.
- [x] **Step 3: If `AI_GATEWAY_API_KEY` is present in env, live smoke:** `--backend vercel-gateway --model anthropic/claude-haiku-4.5 --N-samples-per-task 1 --max-tasks 3 --workers 3` against `benchmark/`, then inspect the results JSONL for `thinking_source` and non-empty responses. Skip (and say so) if no key.

### Task 3: Docs (README + CLAUDE.md) and PR

**Files:**
- Modify: `README.md` (Install/API keys, Run Inference examples, Deviations section), `CLAUDE.md` (commands + API-key bullet)

- [x] **Step 1: README** — API keys: `AI_GATEWAY_API_KEY` default; OpenRouter legacy path documented under `--backend openrouter`. Run example updated. Deviations section gains a "Backend" bullet (gateway default; per-response cost accounting now dashboard-side, cost columns zero on gateway runs; `-openrouter` filename suffix).
- [x] **Step 2: CLAUDE.md** — update the "Layout vs. README" API-key bullet and the example command.
- [x] **Step 3: Commit on `feat/vercel-ai-gateway-backend`, push, `gh pr create`, merge, sync main** (established workflow).
