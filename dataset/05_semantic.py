"""Generator for the Semantic category (benchmark/semantic.jsonl).

Top rung of the abstraction ladder: connecting positions to natural-language chess
understanding. Given a FEN and a move, pick which of four commentary snippets actually
describes that position/move (MCQ, answer A-D).

Input is ``comment_dataset.final.json`` — real annotator commentary already filtered,
cleaned, and quality-judged by the offline vLLM pipeline (05_1 -> 05_2 -> 05_3). This
script only assembles MCQs from it. Regenerating from post-training-cutoff commentary
doubles as a contamination control.

Four variants, differing only in how distractor comments are chosen — a built-in probe of
whether models pick answers by superficial cues:
- easy_random: distractors drawn uniformly (baseline; superficial cues suffice).
- keyword: distractors share extracted keywords with the correct comment.
- piece_stage: distractors involve the same moving piece and game phase (opening/middle/end).
- embedding: distractors are nearest neighbors by sentence-embedding cosine similarity
  (hardest — distractors are *about* similar situations).

Each variant falls back to random fill when its strategy yields too few distractors.

Extra dependencies beyond requirements.txt: sentence-transformers (requirements-optional.txt).

Usage:
    python dataset/05_semantic.py --input <comment_dataset.final.json> \
        --output_root data/benchmark --cache_dir <embedding-cache-dir>
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import tqdm
from sentence_transformers import SentenceTransformer

from utils import (
    FORMAT_EXAMPLES_MCQ,
    ChessQuestionAnsweringTask,
    construct_prompt,
    save_tasks,
    seed_everything,
)


def normalize_comment_text(text: str) -> str:
    """Normalize commentary for display/embedding: strip zero-width chars, convert
    non-breaking spaces, and replace chess figurine glyphs (private-use codepoints used
    by annotation software) with K/Q/R/B/N/P letters."""
    if not text:
        return ""

    # Replace non-breaking space with regular space
    normalized_text = text.replace("\u00a0", " ")

    # Remove invisible Unicode characters
    normalized_text = re.sub(r"[\u200B-\u200F\u202A-\u202E]", "", normalized_text)

    # Replace chess figurines only if they exist
    figurine_replacements = {
        "\ue024": "K",
        "\ue025": "Q",
        "\ue026": "R",
        "\ue027": "B",
        "\ue028": "N",
        "\ue029": "P",
    }

    for figurine, letter in figurine_replacements.items():
        if figurine in normalized_text:
            normalized_text = normalized_text.replace(figurine, letter)

    return normalized_text.strip()


def _stage_bucket(move_number: int, opening_threshold: int = 12, middlegame_threshold: int = 30) -> str:
    """Bucket a move number into opening/middlegame/endgame (crude but only used to match
    distractors to a similar game phase); unparseable move numbers fall into 'opening'."""
    try:
        numeric_move_number = int(move_number)
    except Exception:
        numeric_move_number = 0
    if numeric_move_number <= opening_threshold:
        return "opening"
    if numeric_move_number <= middlegame_threshold:
        return "middlegame"
    return "endgame"


def _build_indices(items: list[dict[str, Any]], config):
    """Build inverted indices (keyword -> item ids, piece -> ids, (piece, stage) -> ids)
    so the keyword/piece_stage distractor strategies can look up candidates in O(1)."""
    by_keyword = {}
    by_piece = {}
    by_piece_stage = {}

    for item_index, item in enumerate(items):
        # Keywords
        keywords = [
            str(raw_keyword).strip().lower() for raw_keyword in (item.get("keywords") or []) if str(raw_keyword).strip()
        ]
        for keyword in keywords:
            by_keyword.setdefault(keyword, []).append(item_index)

        # Piece
        piece = str(item.get("move_piece") or "").strip().lower() or "unknown"
        by_piece.setdefault(piece, []).append(item_index)

        # Piece + stage
        stage = _stage_bucket(item.get("move_number", 0), config.opening_threshold, config.middlegame_threshold)
        by_piece_stage.setdefault((piece, stage), []).append(item_index)

    return by_keyword, by_piece, by_piece_stage


def _pick_random(pool: list[int], exclude: set, sample_count: int) -> list[int]:
    """Pick k random indices from pool, excluding given ones (returns fewer if pool is small)."""
    choices = [candidate_index for candidate_index in pool if candidate_index not in exclude]
    if len(choices) <= sample_count:
        random.shuffle(choices)
        return choices
    return random.sample(choices, sample_count)


def _get_neighbors(
    item_index: int, items: list[dict[str, Any]], indices: dict, neighbor_count: int, strategy: str, config
) -> list[int]:
    """Return up to ``neighbor_count`` distractor candidates related to ``item_index`` by the given strategy
    (shared keyword / same moving piece / same piece and game phase), self excluded.

    ``indices`` must be the inverted index matching the strategy (from _build_indices).
    """
    if strategy == "keyword":
        keywords = [
            str(raw_keyword).strip().lower()
            for raw_keyword in (items[item_index].get("keywords") or [])
            if str(raw_keyword).strip()
        ]
        candidate_indices = set()
        for keyword in keywords:
            candidate_indices.update(indices.get(keyword, []))
        candidate_indices.discard(item_index)
        return random.sample(list(candidate_indices), min(neighbor_count, len(candidate_indices)))

    elif strategy == "piece":
        piece = str(items[item_index].get("move_piece") or "").strip().lower() or "unknown"
        candidate_indices = [
            candidate_index for candidate_index in indices.get(piece, []) if candidate_index != item_index
        ]
        return random.sample(candidate_indices, min(neighbor_count, len(candidate_indices)))

    elif strategy == "piece_stage":
        piece = str(items[item_index].get("move_piece") or "").strip().lower() or "unknown"
        stage = _stage_bucket(
            items[item_index].get("move_number", 0),
            config.opening_threshold,
            config.middlegame_threshold,
        )
        candidate_indices = [
            candidate_index for candidate_index in indices.get((piece, stage), []) if candidate_index != item_index
        ]
        return random.sample(candidate_indices, min(neighbor_count, len(candidate_indices)))

    return []


def _ensure_unique_options(base_opts: list[str], need: int, all_items: list[dict], exclude_indices: set) -> list[str]:
    """De-duplicate the option texts and top up with random comments until ``need`` options.

    Duplicates can occur when two source games share commentary; the random top-up keeps
    every MCQ at exactly num_options choices regardless of the distractor strategy's yield.
    """
    options = list(dict.fromkeys([option_text for option_text in base_opts if option_text]))  # Remove duplicates and empty

    if len(options) < need:
        # Fill with random options
        available = [item_index for item_index in range(len(all_items)) if item_index not in exclude_indices]
        needed = need - len(options)
        random_indices = random.sample(available, min(needed, len(available)))

        for item_index in random_indices:
            text = all_items[item_index].get("_comment_text", "")
            if text and text not in options:
                options.append(text)
                if len(options) >= need:
                    break

    return options[:need]


def _compute_embeddings(items: list[dict[str, Any]], config) -> np.ndarray:
    """Compute (or load from cache) L2-normalized sentence embeddings for every comment.

    The cache under config.cache_dir is keyed by item count + model name in a sidecar JSON;
    any mismatch triggers a full recompute. Normalized embeddings mean a plain dot product
    is cosine similarity (used by _get_semantic_neighbors). Embedding all comments is
    required even for non-embedding variants because main() computes this unconditionally.
    """
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = cache_dir / "comment_embeddings.npy"
    map_path = cache_dir / "comment_embeddings.index.json"

    texts = [normalize_comment_text(item.get("cleaned_comment") or item.get("comment", "")) for item in items]

    # Try loading cached embeddings
    if emb_path.exists() and map_path.exists():
        try:
            with open(map_path, encoding="utf-8") as metadata_file:
                meta = json.load(metadata_file)
            if meta.get("count") == len(texts) and meta.get("model") == config.embed_model:
                embedding_array = np.load(str(emb_path))
                if embedding_array.shape[0] == len(texts):
                    print(f"Loaded cached embeddings from {emb_path}")
                    return embedding_array.astype(np.float32, copy=False)
        except Exception as error:
            print(f"Failed to load cached embeddings: {error}")

    # Must compute embeddings
    print(f"Computing embeddings using {config.embed_model}...")
    model = SentenceTransformer(config.embed_model)
    embedding_batches = []
    for batch_start in range(0, len(texts), max(1, config.embed_batch)):
        batch = texts[batch_start : batch_start + config.embed_batch]
        batch_vectors = model.encode(
            batch,
            batch_size=min(config.embed_batch, config.embed_max_batch),
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embedding_batches.append(np.asarray(batch_vectors, dtype=np.float32))

    embedding_array = np.vstack(embedding_batches).astype(np.float32)

    # Cache results
    print(f"Saving embeddings to {emb_path}")
    np.save(str(emb_path), embedding_array)
    with open(map_path, "w", encoding="utf-8") as metadata_file:
        json.dump({"count": len(texts), "model": config.embed_model}, metadata_file)

    return embedding_array


def _get_semantic_neighbors(item_index: int, embeddings: np.ndarray, config, neighbor_count: int = None) -> list[int]:
    """Return the ``neighbor_count`` most cosine-similar comments to ``item_index`` (self excluded), most
    similar first — argpartition keeps this O(n) rather than a full sort."""
    if embeddings is None:
        return []

    if neighbor_count is None:
        neighbor_count = config.num_distractors

    query_embedding = embeddings[item_index]
    similarities = query_embedding @ embeddings.T
    similarities[item_index] = -np.inf  # Exclude self

    if neighbor_count < len(similarities):
        neighbor_indices = np.argpartition(similarities, -neighbor_count)[-neighbor_count:]
        neighbor_indices = neighbor_indices[np.argsort(similarities[neighbor_indices])[::-1]]
    else:
        neighbor_indices = np.argsort(similarities)[::-1]

    return [int(neighbor_index) for neighbor_index in neighbor_indices[:neighbor_count].tolist()]


def generate_comment_mcq_task(
    item: dict[str, Any], variant: str, options: list[str], task_index: int
) -> ChessQuestionAnsweringTask:
    """Assemble one MCQ task: shuffle the options, letter them A-D, record the correct letter.

    CONTEXT_PLACEHOLDER is stripped — the commentary itself describes the position, so
    injecting a piece arrangement would partially give the answer away. The correct option's
    text and the full option list are kept in metadata for later analysis.
    """

    # Build context information
    fen_before = item.get("fen_before", "")
    move_uci = item.get("move_uci", "")

    prefix = f"You are given a chess position in FEN: {fen_before}\n"
    prefix += f"A player makes the move: {move_uci}\n"

    # Shuffle options for fairness and find correct letter
    correct_text = item["_comment_text"]
    shuffled_options = options[:]
    random.shuffle(shuffled_options)

    # Find the correct letter after shuffling
    correct_index = shuffled_options.index(correct_text)
    correct_letter = chr(65 + correct_index)  # A, B, C, D

    # Create multiple choice format
    option_text = "\n".join(
        [f"{chr(65 + option_index)}. {option}" for option_index, option in enumerate(shuffled_options)]
    )

    task_description = f"Select the commentary that best describes this position and move.\n\nOptions:\n{option_text}\n"
    suffix = "Example format: FORMAT_EXAMPLE_PLACEHOLDER"

    full_question = construct_prompt(prefix, task_description, suffix).replace("CONTEXT_PLACEHOLDER", "")

    return ChessQuestionAnsweringTask(
        task_id=f"semantic_{variant}_{task_index:04d}",
        task_type=f"semantic_{variant}",
        task_category="Semantic",
        input=fen_before,
        question=full_question,
        format_examples=FORMAT_EXAMPLES_MCQ,
        correct_answer=correct_letter,
        answer_type="single",
        metadata={
            "variant": variant,
            "options": shuffled_options,
            "correct_text": correct_text,
            "fen_before": item.get("fen_before"),
            "fen_after": item.get("fen_after"),
            "move_uci": item.get("move_uci"),
            "move_number": item.get("move_number"),
            "side_to_move": item.get("side_to_move"),
        },
    )


def find_comment_tasks_by_variant(
    items: list[dict[str, Any]], variant: str, config, embeddings: np.ndarray
) -> list[ChessQuestionAnsweringTask]:
    """Generate up to config.N_sample_mcq MCQ tasks for one distractor-strategy variant.

    Per candidate item: gather distractors via the variant's strategy, fall back to random
    fill when short (see module docstring), then keep the task only if exactly
    config.num_options unique options survived including the correct comment.
    """

    # Build indices for neighbor finding
    by_keyword, by_piece, by_piece_stage = _build_indices(items, config)

    found = []
    found_counter = 0
    universe = list(range(len(items)))

    # Sample indices if needed
    if config.sample_size and config.sample_size > 0:
        total = len(items)
        sample_count = min(config.sample_size, total)
        base_indices = (
            random.sample(list(range(total)), sample_count) if config.random_sample else list(range(sample_count))
        )
    else:
        base_indices = list(range(len(items)))

    for item_index in tqdm.tqdm(base_indices):
        if found_counter >= config.N_sample_mcq:
            break

        item = items[item_index]
        correct_text = item.get("_comment_text", "")
        if not correct_text:
            continue

        exclude = {item_index}

        # Build options based on variant strategy
        if variant == "easy_random":
            distractor_indices = _pick_random(universe, exclude, config.num_distractors)
        elif variant == "keyword":
            distractor_indices = _get_neighbors(item_index, items, by_keyword, config.num_distractors, "keyword", config)
            if len(distractor_indices) < config.num_distractors:
                more = _pick_random(
                    universe,
                    exclude | set(distractor_indices),
                    config.num_distractors - len(distractor_indices),
                )
                distractor_indices.extend(more)
        elif variant == "piece_stage":
            distractor_indices = _get_neighbors(
                item_index, items, by_piece_stage, config.num_distractors, "piece_stage", config
            )
            if len(distractor_indices) < config.num_distractors:
                more = _get_neighbors(
                    item_index,
                    items,
                    by_piece,
                    config.num_distractors - len(distractor_indices),
                    "piece",
                    config,
                )
                distractor_indices.extend(more)
            if len(distractor_indices) < config.num_distractors:
                more = _pick_random(
                    universe,
                    exclude | set(distractor_indices),
                    config.num_distractors - len(distractor_indices),
                )
                distractor_indices.extend(more)
        elif variant == "embedding":
            distractor_indices = _get_semantic_neighbors(item_index, embeddings, config)
            # Ensure we have enough indices
            if len(distractor_indices) < config.num_distractors:
                more = _pick_random(
                    universe,
                    exclude | set(distractor_indices),
                    config.num_distractors - len(distractor_indices),
                )
                distractor_indices.extend(more)
        else:
            # Default to random
            distractor_indices = _pick_random(universe, exclude, config.num_distractors)

        # Build options list
        options = [correct_text]
        for distractor_index in distractor_indices[: config.num_distractors]:
            distractor_text = items[distractor_index].get("_comment_text", "")
            if distractor_text and distractor_text != correct_text:
                options.append(distractor_text)

        # Ensure we have exactly num_options unique options
        options = _ensure_unique_options(options, config.num_options, items, exclude | set(distractor_indices))

        if len(options) == config.num_options and correct_text in options:
            task = generate_comment_mcq_task(item, variant, options, found_counter)
            found.append(task)
            found_counter += 1

    return found


def parse_args():
    """Parse CLI flags (self-documenting via help strings). Defaults use the upstream layout —
    pass explicit paths in this repo."""
    parser = argparse.ArgumentParser(description="Generate comment MCQ tasks with multiple distractor strategies")

    # Input/Output paths
    parser.add_argument(
        "--input",
        type=str,
        default="../../data/mid/comment_dataset.final.json",
        help="Path to input comment dataset",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="../../data/benchmark",
        help="Output directory for generated tasks",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/chess-llm-benchmark/data/mid",
        help="Directory for embedding cache",
    )

    # Task generation parameters
    parser.add_argument("--N_sample_mcq", type=int, default=100, help="Number of MCQ tasks per variant")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="Number of items to sample for task generation (0 = all)",
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        default=True,
        help="Use random sampling instead of sequential",
    )

    # Game phase thresholds
    parser.add_argument(
        "--opening_threshold",
        type=int,
        default=12,
        help="Move number threshold for opening phase",
    )
    parser.add_argument(
        "--middlegame_threshold",
        type=int,
        default=30,
        help="Move number threshold for middlegame phase",
    )

    # Embedding parameters
    parser.add_argument(
        "--embed_model",
        type=str,
        default="Qwen/Qwen3-Embedding-8B",
        help="Sentence transformer model for embeddings",
    )
    parser.add_argument(
        "--embed_batch",
        type=int,
        default=256,
        help="Batch size for embedding computation",
    )
    parser.add_argument(
        "--embed_max_batch",
        type=int,
        default=64,
        help="Maximum batch size for model encoding",
    )

    # MCQ parameters
    parser.add_argument("--num_options", type=int, default=4, help="Number of options per MCQ task")
    parser.add_argument(
        "--num_distractors",
        type=int,
        default=3,
        help="Number of distractor options per task",
    )

    # Output parameters
    parser.add_argument(
        "--output_filename",
        type=str,
        default="semantic.jsonl",
        help="Output filename for generated tasks",
    )

    # General parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()


def main():
    """Generate the Semantic benchmark file: load comments, embed, build all four MCQ variants."""
    config = parse_args()
    seed_everything(config.seed)

    # Load data
    print(f"Loading comment data from {config.input}")
    with open(config.input, encoding="utf-8") as comment_file:
        items = json.load(comment_file)

    # Normalize comment texts
    for item in items:
        comment_text = item.get("cleaned_comment") or item.get("comment", "")
        item["_comment_text"] = normalize_comment_text(comment_text)

    print(f"Loaded {len(items)} comment items")

    # Always compute/load embeddings for semantic variant
    embeddings = _compute_embeddings(items, config)

    # Generate tasks for each variant
    variants = ["easy_random", "keyword", "piece_stage", "embedding"]
    all_tasks = []

    for variant in variants:
        print(f"Generating {variant} tasks...")
        found = find_comment_tasks_by_variant(items, variant, config, embeddings)
        all_tasks.extend(found)
        print(f"Generated {len(found)} {variant} tasks")

    # Save all tasks to single file
    save_tasks(all_tasks, config.output_filename, config)
    print(f"Saved {len(all_tasks)} total tasks to {config.output_filename}")


if __name__ == "__main__":
    main()
