#!/usr/bin/env python3
"""Stage 3 (final) of the offline commentary pipeline feeding the Semantic category.

Pipeline: 05_1 (extract+filter) -> 05_2 (LLM cleaning) -> 05_3 (LLM quality judging, this
file) -> comment_dataset.final.json -> 05_semantic.py (MCQ assembly). Like stage 2 this
needs a GPU + vLLM. A cheap regex/keyword heuristic pre-drops obviously irrelevant comments
before the LLM sees them.

Final relevance filter for cleaned comments using vLLM (offline).

Goal: Keep only comments that are explicitly about the given move/position in this game.

Strict judgement output: KEEP or DROP (uppercase, no extra text).

Signals to KEEP (non-exhaustive):
- Directly references the current move, pieces, or squares (SAN/UCI like Rae1, Qxh7+, e4e5, a1h8)
- Talks about the position (e.g., "kingside attack", "weak d6 pawn", "open c-file")
- Provides evaluation or reason related to this move/position

Signals to DROP:
- Generic aphorisms/quotes or meta commentary with no tie to the current move/position
- Player biography/psychology, event logistics, audience, stream, etc.
- Comments referencing unrelated games or people (should have been removed earlier)

Heuristic pre-filter: Before calling the LLM, quickly DROP comments that clearly lack any chess-specific signal
(no piece/square notation, too short, obvious generic phrases). You can disable this via --no-heuristics.

Usage:
  python scripts/judge_comments_vllm.py \
    --input ./datasets/comment_tasks/comment_dataset.cleaned.json \
    --output ./datasets/comment_tasks/comment_dataset.final.json \
    --model Qwen/Qwen3-4B --batch-size 256 --no-flashinfer --disable-thinking
"""

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    from vllm import LLM, SamplingParams
except Exception:
    print("vLLM is required. Install with: pip install vllm")
    raise


def _setup_env(
    model_name: str,
    enable_prefix_caching: bool = True,
    attention_backend: str = None,
    use_flashinfer: bool | None = False,
) -> None:
    """Set vLLM environment knobs before engine construction (same helper as in 05_2)."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    if enable_prefix_caching:
        os.environ.setdefault("VLLM_ENABLE_PREFIX_CACHING", "1")
    else:
        os.environ["VLLM_ENABLE_PREFIX_CACHING"] = "0"
    if attention_backend and "VLLM_ATTENTION_BACKEND" not in os.environ:
        os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend
    if "gemma-3" in (model_name or "").lower() and "VLLM_ATTENTION_BACKEND" not in os.environ:
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    if use_flashinfer is not None:
        os.environ["VLLM_USE_FLASHINFER"] = "1" if use_flashinfer else "0"
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1" if use_flashinfer else "0"
        if not use_flashinfer:
            os.environ.setdefault("VLLM_FLASHINFER_CACHE_DISABLED", "1")


def load_items(path: Path, max_records: int = 0) -> list[dict[str, Any]]:
    """Load the stage-2 JSON array, optionally truncated to max_records for quick test runs."""
    with open(path, encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON array")
    if max_records and max_records > 0:
        data = data[:max_records]
    return data


def save_items(path: Path, items: list[dict[str, Any]]) -> None:
    """Write kept records as a pretty-printed JSON array, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(items, output_file, ensure_ascii=False, indent=2)


PIECE_WORDS = {
    "king",
    "queen",
    "rook",
    "bishop",
    "knight",
    "pawn",
    "kingside",
    "queenside",
    "file",
    "rank",
    "diagonal",
    "check",
    "checkmate",
    "mate",
    "castle",
    "castling",
    "gambit",
    "fork",
    "pin",
    "skewer",
    "x-ray",
    "xray",
    "tempo",
    "initiative",
    "zugzwang",
    "passed pawn",
    "outpost",
}


SAN_UCI_PATTERNS = [
    re.compile(r"\b[O0]-O(?:-O)?\b", re.I),  # O-O, O-O-O
    re.compile(r"\b[a-h][1-8][a-h][1-8]\b"),  # UCI move e2e4
    re.compile(r"\b[a-h]x?[a-h][1-8](?:[+#])?\b", re.I),  # SAN-like exd5, Qh4+, etc.
    re.compile(r"\b[QRBNK][a-h]?[1-8]?x?[a-h][1-8](?:[+#])?\b"),  # piece SAN moves
]


GENERIC_PHRASES = [
    "i will play",
    "we will draw",
    "as they say",
    "people say",
    "it is said",
    "in general",
    "generally speaking",
    "quote",
    "famous quote",
    "life",
    "luck",
]


def heuristic_is_relevant(text: str) -> bool:
    """Cheap pre-filter: True if the comment plausibly discusses concrete chess content.

    Drops: empty/very short comments and ones containing known generic phrases. Keeps:
    anything with SAN/UCI notation, chess vocabulary, or at least a square name. Only
    'False' is final here — 'True' just promotes the comment to LLM judging.
    """
    if not text:
        return False
    stripped_text = text.strip()
    # Length threshold
    if len(stripped_text) < 25:
        return False
    lowercase_text = stripped_text.lower()
    # Obvious generic signals
    for generic_phrase in GENERIC_PHRASES:
        if generic_phrase in lowercase_text:
            return False
    # SAN/UCI presence
    if any(pattern.search(stripped_text) for pattern in SAN_UCI_PATTERNS):
        return True
    # Chess word presence
    if any(chess_word in lowercase_text for chess_word in PIECE_WORDS):
        return True
    # Square mention (at least one square)
    return bool(re.search(r"\b[a-h][1-8]\b", stripped_text))


def build_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    """Build the KEEP/DROP judging prompt: strict system rules, one few-shot DROP example
    (a generic aphorism), then the comment with its full game context (FENs, move, PGN)."""
    comment = item.get("cleaned_comment") or item.get("comment", "")
    fen_before = item.get("fen_before", "")
    fen_after = item.get("fen_after", "")
    move_uci = item.get("move_uci", "")
    move_number = item.get("move_number", "")
    side_to_move = item.get("side_to_move", "")
    pgn_until = item.get("pgn_until_move", "")

    system_prompt = (
        "You are a strict chess commentary relevance judge. "
        "Given a single comment and the exact game context (FENs, PGN so far, and the move), decide if the comment is directly useful to understand or evaluate the current move/position in THIS game. "
        "Output exactly one token: KEEP or DROP."
        "\nKEEP if and only if the comment provides actionable insight about the current move or position (e.g., mentions pieces/squares, tactical/positional ideas, consequences relevant to this position)."
        "\nDROP if the comment is generic quotes/aphorisms, meta or biography, audience/event chatter, or otherwise unrelated to the concrete move/position."
    )

    user_prompt = (
        f"Comment: {comment}\n"
        f"Move UCI: {move_uci} | Move number: {move_number} | Side to move: {side_to_move}\n"
        f"FEN before: {fen_before}\n"
        f"FEN after: {fen_after}\n"
        f"PGN until move: {pgn_until}\n"
        "Answer strictly with KEEP or DROP."
    )

    # One tiny few-shot to anchor behavior
    ex_user = (
        "Comment: I will play 40 good moves. If my opponent plays 40 good moves too, we will draw.\n"
        "Move UCI: e2e4 | Move number: 1 | Side to move: white\n"
        "FEN before: startpos\n"
        "FEN after: <omitted>\n"
        "PGN until move: 1. e4\n"
        "Answer strictly with KEEP or DROP."
    )
    ex_assistant = "DROP"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ex_user},
        {"role": "assistant", "content": ex_assistant},
        {"role": "user", "content": user_prompt},
    ]


def sanitize_label(text: str) -> str:
    """Coerce raw LLM output to exactly "KEEP" or "DROP", defaulting to DROP on anything
    unparseable (strips think-blocks/backticks, then searches for either label)."""
    if not text:
        return "DROP"
    working_text = text.strip().strip("` ")
    # remove <think> blocks if any
    working_text = re.sub(r"(?is)<think>.*?(</think>|$)", "", working_text).strip()
    # collapse whitespace
    working_text = re.sub(r"\s+", " ", working_text)
    # extract a label if present in text
    label_match = re.search(r"\b(KEEP|DROP)\b", working_text, re.I)
    if label_match:
        return label_match.group(1).upper()
    # fallback: first token upper
    first_token = working_text.split()[0].upper() if working_text.split() else "DROP"
    return first_token if first_token in {"KEEP", "DROP"} else "DROP"


def main() -> int:
    """CLI entry: heuristic pre-filter, vLLM KEEP/DROP judging, write the final dataset.

    Label bookkeeping: ``heuristic_labels`` holds one slot per input item ("DROP" or
    "CANDIDATE"); after judging, each CANDIDATE slot is overwritten in order with the LLM's
    verdict, so ``final_labels`` lines up 1:1 with ``items``.
    """
    default_input = Path("../../data/mid/comment_dataset.cleaned.json")
    default_output = Path("../../data/mid/comment_dataset.final.json")
    argument_parser = argparse.ArgumentParser(description="Judge relevance of cleaned comments using vLLM")
    argument_parser.add_argument("--input", type=Path, default=default_input)
    argument_parser.add_argument("--output", type=Path, default=default_output)
    argument_parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    argument_parser.add_argument("--batch-size", type=int, default=1024)
    argument_parser.add_argument("--max-records", type=int, default=0)
    argument_parser.add_argument("--max-model-len", type=int, default=4096)
    argument_parser.add_argument("--max-tokens", type=int, default=8)
    argument_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    argument_parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    argument_parser.add_argument(
        "--dtype", type=str, default=None, choices=[None, "auto", "float16", "bfloat16", "float32"], nargs="?"
    )
    argument_parser.add_argument("--attention-backend", type=str, default=None, choices=[None, "FLASH_ATTN", "XFORMERS"], nargs="?")
    flashinfer_group = argument_parser.add_mutually_exclusive_group()
    flashinfer_group.add_argument("--use-flashinfer", dest="use_flashinfer", action="store_true")
    flashinfer_group.add_argument("--no-flashinfer", dest="use_flashinfer", action="store_false")
    argument_parser.set_defaults(use_flashinfer=False)
    thinking_group = argument_parser.add_mutually_exclusive_group()
    thinking_group.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    thinking_group.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    argument_parser.set_defaults(enable_thinking=False)
    heuristics_group = argument_parser.add_mutually_exclusive_group()
    heuristics_group.add_argument("--heuristics", dest="heuristics", action="store_true")
    heuristics_group.add_argument("--no-heuristics", dest="heuristics", action="store_false")
    argument_parser.set_defaults(heuristics=True)

    args = argument_parser.parse_args()

    items = load_items(args.input, args.max_records)
    kept: list[dict[str, Any]] = []

    # Heuristic prefilter
    heuristic_labels: list[str] = []
    to_judge: list[dict[str, Any]] = []
    if args.heuristics:
        for item in items:
            text = item.get("cleaned_comment") or item.get("comment", "")
            if heuristic_is_relevant(text):
                to_judge.append(item)
                heuristic_labels.append("CANDIDATE")
            else:
                heuristic_labels.append("DROP")
        # NOTE (upstream dead code, kept as-is): this comprehension's result is discarded —
        # the candidate->item mapping is actually reconstructed positionally further down.
        [label_index for label_index, label in enumerate(heuristic_labels) if label == "CANDIDATE"]
    else:
        to_judge = items
        # NOTE (upstream dead code, kept as-is): result discarded, same as above.
        list(range(len(items)))

    # Setup vLLM
    _setup_env(
        args.model,
        enable_prefix_caching=True,
        attention_backend=args.attention_backend,
        use_flashinfer=args.use_flashinfer,
    )
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "max_model_len": int(args.max_model_len),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_num_seqs": 1024,
    }
    if args.dtype and args.dtype != "auto":
        llm_kwargs["dtype"] = args.dtype

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(args.max_tokens))

    # Judge in batches
    labels: list[str] = [None] * len(to_judge)
    start_time = time.time()
    for batch_start in range(0, len(to_judge), args.batch_size):
        batch = to_judge[batch_start : batch_start + args.batch_size]
        prompts = [build_messages(item) for item in batch]
        chat_kwargs: dict[str, Any] = {}
        if args.enable_thinking is not None:
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": bool(args.enable_thinking)}
        try:
            batch_outputs = llm.chat(prompts, sampling_params, **chat_kwargs)
        except Exception:
            batch_outputs = llm.chat(prompts, sampling_params)
        for batch_offset, model_output in enumerate(batch_outputs):
            text = model_output.outputs[0].text if model_output.outputs and model_output.outputs[0] else ""
            labels[batch_start + batch_offset] = sanitize_label(text)

    elapsed_seconds = time.time() - start_time
    print(f"Judged {len(to_judge)} candidates in {elapsed_seconds:.1f}s")

    # Combine heuristic and LLM labels
    if args.heuristics:
        judged_cursor = 0
        for label_index, label in enumerate(heuristic_labels):
            if label == "DROP":
                continue
            heuristic_labels[label_index] = labels[judged_cursor]
            judged_cursor += 1
        final_labels = heuristic_labels
    else:
        final_labels = labels

    # Decide and keep
    keeps = 0
    for item_index, item in enumerate(items):
        label = final_labels[item_index] if item_index < len(final_labels) else "DROP"
        if label == "KEEP":
            kept.append(item)
            keeps += 1

    save_items(args.output, kept)
    print(f"Saved final dataset: {args.output}")
    print(f"Input: {len(items)}, kept: {len(kept)}, dropped: {len(items) - len(kept)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
