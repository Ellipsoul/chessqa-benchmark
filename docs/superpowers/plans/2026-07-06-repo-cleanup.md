# Repo Cleanup (Gold-Standard Hygiene, No Logic Changes) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the chessqa-benchmark repo to gold-standard engineering hygiene — verified-installable environment, zero lint warnings under a validated config, and comprehensive documentation of every module/function — while provably changing zero runtime logic.

**Architecture:** Three layers of work: (1) environment verification — prove `requirements.txt` installs into a fresh venv and cap dependency majors at tested versions; (2) lint validation — confirm the existing ruff config's per-file ignores are justified, not masking problems; (3) a file-by-file documentation pass over all 11 Python files (~4,800 lines), gated by an AST-equivalence check that mechanically proves no logic changed.

**Tech Stack:** Python 3.11, ruff 0.15.x, python `ast` module for logic-equivalence verification.

## Global Constraints

- **No logic changes.** Only docstrings, comments, and dependency/config metadata may change. Verified mechanically by Task 6's AST comparison (docstrings stripped, HEAD vs. working tree).
- **No reformatting.** Do not run `ruff format` — formatting churn would bury the documentation diff. Lint = `ruff check` only.
- **Preserve upstream semantics for reproduction.** This repo must reproduce the paper's runs (Phase 1); prompt strings, placeholders (`CONTEXT_PLACEHOLDER`, `FORMAT_EXAMPLE_PLACEHOLDER`), scoring, and extraction logic are untouchable.
- **Documentation register:** written for a strong engineer new to LLM evaluation (Aron). Explain *why* (design intent, paper linkage) at module level; explain *what/how* at function level; inline comments only where the code is non-obvious. No noise comments restating the code.
- **Docstring convention:** Google style (`Args:` / `Returns:` / `Raises:` sections) for functions with non-trivial signatures; one-line docstrings for trivial helpers. Module docstrings state the file's role in the pipeline (raw data → generator → benchmark JSONL → eval runner).
- **No commits** until Aron reviews the diff (per session policy); the final task prepares a suggested commit message.

---

### Task 1: Fresh-venv install proof + dependency caps

**Files:**
- Modify: `requirements.txt` (add upper bound to pandas)
- Scratch: `$SCRATCHPAD/fresh-venv/` (throwaway venv, not in repo)

**Interfaces:**
- Produces: a `requirements.txt` whose bounds match versions actually verified (chess==1.10.0, numpy>=1.24,<3, pandas>=2.0,<4, tqdm, requests, zstandard).

- [x] **Step 1: Build a fresh venv in the scratchpad and install requirements**

```bash
python3 -m venv "$SCRATCHPAD/fresh-venv"
"$SCRATCHPAD/fresh-venv/bin/pip" install --upgrade pip -q
"$SCRATCHPAD/fresh-venv/bin/pip" install -r requirements.txt -r requirements-dev.txt
```

Expected: clean install, no resolver errors.

- [x] **Step 2: Verify imports and CLI smoke test in the fresh venv**

```bash
"$SCRATCHPAD/fresh-venv/bin/python" -c "import chess, numpy, pandas, requests, tqdm, zstandard; print('OK')"
"$SCRATCHPAD/fresh-venv/bin/python" eval/run_openrouter.py --help > /dev/null && echo "CLI OK"
```

Expected: `OK` and `CLI OK`.

- [x] **Step 3: Cap pandas major version in requirements.txt**

Change `pandas>=2.0.0` → `pandas>=2.0.0,<4` with a comment noting tested version. Rationale: the repo's pandas surface is two call sites (`read_csv`, `Series` field access in `dataset/utils.py`), verified working under pandas 3.0.3; an uncapped major is how reproducibility silently breaks.

- [x] **Step 4: Re-run install of modified requirements in fresh venv; delete scratch venv**

```bash
"$SCRATCHPAD/fresh-venv/bin/pip" install -r requirements.txt
rm -rf "$SCRATCHPAD/fresh-venv"
```

Expected: "Requirement already satisfied" lines, exit 0.

### Task 2: Validate lint config (no masked warnings)

**Files:**
- Modify (maybe): `pyproject.toml` per-file-ignores

**Interfaces:**
- Produces: a ruff config where every `ignore` is verified necessary; `ruff check dataset eval` exits 0.

- [x] **Step 1: Test whether the per-file ignores for 05_2/05_3 are actually needed**

```bash
.venv/bin/ruff check --no-cache dataset/05_2_comment_cleaning.py dataset/05_3_comment_judging.py \
  --config 'lint.per-file-ignores={}'
```

If it passes: remove the per-file-ignores block from pyproject.toml (dead config). If it fails with E402/F401: keep, but confirm each flagged line is genuinely intentional (vLLM import ordering), and narrow the ignore list to only the codes that fire.

- [x] **Step 2: Check the global E501 ignore isn't hiding pathological lines**

```bash
.venv/bin/ruff check --no-cache dataset eval --config 'lint.ignore=["B008"]' --select E501 2>&1 | tail -3
```

Report the count; long prompt-string lines are acceptable (they're literals), no action needed unless code lines (not strings) exceed 120.

- [x] **Step 3: Full lint gate**

```bash
.venv/bin/ruff check --no-cache dataset eval && .venv/bin/python -m compileall -q dataset eval && echo GATE-PASS
```

Expected: `GATE-PASS`.

### Task 3: Documentation pass — shared foundation (`dataset/utils.py`, `dataset/preprocess.py`)

**Files:**
- Modify: `dataset/utils.py` (223 lines, 3 docstring markers), `dataset/preprocess.py` (46 lines, zero comments)

Document per the Global Constraints register: module docstrings explaining role in the data flow; docstring for `ChessQuestionAnsweringTask` explaining every field and the `"single"`/`"multi"` answer-type contract; docstrings for FEN/piece-arrangement helpers, `make_pre_move`, format-example constants, seeding. `preprocess.py`: module docstring + per-function docstrings.

- [x] **Step 1: Read both files fully; write docstrings/comments**
- [x] **Step 2: Gate** — `ruff check` + `py_compile` on both files, expected clean.

### Task 4: Documentation pass — dataset generators (`01`–`04`, `05_semantic`, `05_1`–`05_3`)

**Files:**
- Modify: `dataset/01_structural.py`, `02_motifs.py`, `03_short_tactics.py`, `04_position_judgement.py`, `05_semantic.py`, `05_1_comment_filtering.py`, `05_2_comment_cleaning.py`, `05_3_comment_judging.py`

Each file gets: module docstring naming its benchmark category, its data source (Lichess puzzles / broadcasts / evals / commentary), the task types it emits, and the CLI entry point; function docstrings; inline comments only at chess-logic or data-munging steps that aren't self-evident. For 05_2/05_3, the module docstring must state these are offline vLLM pipeline stages not needed for eval.

- [x] **Step 1: Document 01_structural.py + 02_motifs.py; gate with ruff+compile**
- [x] **Step 2: Document 03_short_tactics.py + 04_position_judgement.py; gate**
- [x] **Step 3: Document 05_semantic.py + 05_1/05_2/05_3; gate**

### Task 5: Documentation pass — eval runner (`eval/run_openrouter.py`)

**Files:**
- Modify: `eval/run_openrouter.py` (1,065 lines)

Highest-value file: this is the harness Aron must understand for Phases 1–3. Module docstring covering the full flow (load JSONL → format prompts → multiprocessing fan-out → OpenRouter call → answer extraction → scoring → resume). Function docstrings for every function, with special care on: `format_prompt` (placeholder resolution), `call_model` (provider-order overrides, thinking extraction from `reasoning`/`reasoning_details`), answer extraction (`FINAL ANSWER:` / `\boxed{}` fallbacks), `evaluate_answer_with_error_type` (full error taxonomy), resume/filename-suffix logic.

- [x] **Step 1: Read fully, document; gate with ruff+compile+`--help` smoke test**

### Task 6: Mechanical no-logic-change proof + final verification

**Files:**
- Scratch: `$SCRATCHPAD/ast_equiv.py`

- [x] **Step 1: Write the AST-equivalence checker**

```python
"""Compare HEAD vs working tree: ASTs must be identical after stripping docstrings."""
import ast, subprocess, sys

def strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
               and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree

files = subprocess.run(["git", "ls-files", "dataset/*.py", "eval/*.py"],
                       capture_output=True, text=True, check=True).stdout.split()
failed = []
for f in files:
    head_src = subprocess.run(["git", "show", f"HEAD:{f}"], capture_output=True, text=True, check=True).stdout
    with open(f) as fh:
        work_src = fh.read()
    a = ast.dump(strip_docstrings(ast.parse(head_src)))
    b = ast.dump(strip_docstrings(ast.parse(work_src)))
    if a != b:
        failed.append(f)
print("AST-EQUIV PASS" if not failed else f"AST-EQUIV FAIL: {failed}")
sys.exit(1 if failed else 0)
```

- [x] **Step 2: Run it** — Expected: `AST-EQUIV PASS`.
- [x] **Step 3: Full final gate**

```bash
.venv/bin/ruff check --no-cache dataset eval \
  && .venv/bin/python -m compileall -q dataset eval \
  && .venv/bin/python eval/run_openrouter.py --help > /dev/null \
  && echo FINAL-GATE-PASS
```

- [x] **Step 4: Summarize diff for Aron** — `git diff --stat`, suggested commit message; do not commit.
