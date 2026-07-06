"""Generator for the Position Judgment category (benchmark/position_judgement.jsonl).

Fourth rung of the abstraction ladder: holistic evaluation. Given a FEN, estimate the
Stockfish centipawn evaluation (from White's perspective) by choosing the closest of five
fixed options: -400, -200, 0, +200, +400. That makes it effectively 5-way classification —
random guessing scores 20%, and even top models stay <= 40% in the paper.

Positions come from the Lichess evaluations dump (lichess_db_eval.jsonl.zst): community-
contributed deep Stockfish analyses. Ground truth is the deepest available eval's principal
variation score. Positions are sampled only from five narrow evaluation bands (one per
option, e.g. "winning" = +350..+450) so each task has an unambiguous nearest option;
mate scores and positions between bands are discarded.

Usage:
    python dataset/04_position_judgement.py --data_path data/raw/lichess_db_eval.jsonl.zst \
        --output_root data/benchmark --tasks_per_category 100
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from typing import Any

import tqdm
import zstandard as zstd

from utils import FORMAT_EXAMPLES_CENTIPAWN, ChessQuestionAnsweringTask, construct_prompt, save_tasks, seed_everything

# Sampling bands (centipawns, White's perspective). Each band is centered on one of the five
# answer options with a +/-50cp tolerance; gaps between bands are intentional so no position
# sits ambiguously between two options.
EVAL_CATEGORIES = {
    "neutral": {"min": -50, "max": 50, "name": "neutral"},
    "winning": {"min": 350, "max": 450, "name": "winning"},
    "advantage": {"min": 150, "max": 250, "name": "advantage"},
    "disadvantage": {"min": -250, "max": -150, "name": "disadvantage"},
    "losing": {"min": -450, "max": -350, "name": "losing"},
}


def _get_position_hash(fen: str) -> str:
    """Hash the first four FEN fields (board, turn, castling, en passant) for dedup.

    Move counters are excluded so transpositions reached at different move numbers count
    as the same position.
    """
    position_part = " ".join(fen.split()[:4])
    return hashlib.md5(position_part.encode()).hexdigest()[:16]


def _determine_evaluation_category(centipawns: int) -> str | None:
    """Map a centipawn eval to its sampling band, or None if it falls in a gap (discard)."""
    for category, bounds in EVAL_CATEGORIES.items():
        if bounds["min"] <= centipawns <= bounds["max"]:
            return category
    return None


def _find_best_option_set(target_eval: int) -> list[int]:
    """Return the answer options. Fixed for every task (the parameter is vestigial —
    upstream presumably experimented with target-dependent option sets)."""
    return [-400, -200, 0, 200, 400]


def _get_correct_answer(target_eval: int, options: list[int]) -> str:
    """Return the option closest to the true eval — unambiguous given the band sampling."""
    closest_option = min(options, key=lambda x: abs(x - target_eval))
    return str(closest_option)


def _parse_evaluation_line(line: str) -> dict[str, Any] | None:
    """Parse one JSON line of the Lichess eval dump into a flat record, or None to discard.

    Takes the deepest engine analysis available for the position and its first principal
    variation. Discards: malformed lines, lines missing FEN/evals, and mate scores (mate
    has no centipawn value, so it can't be bucketed).
    """
    try:
        data = json.loads(line)
        fen = data.get("fen", "")
        evals = data.get("evals", [])

        if not fen or not evals:
            return None

        best_eval = max(evals, key=lambda e: e.get("depth", 0))
        depth = best_eval.get("depth", 0)
        knodes = best_eval.get("knodes", 0)

        pvs = best_eval.get("pvs", [])
        if not pvs:
            return None

        best_pv = pvs[0]
        centipawns = best_pv.get("cp", 0)
        line_moves = best_pv.get("line", "")

        if "mate" in best_pv:
            return None

        position_id = _get_position_hash(fen)

        return {
            "fen": fen,
            "best_evaluation": centipawns,
            "depth": depth,
            "knodes": knodes,
            "best_line": line_moves,
            "position_id": position_id,
        }

    except (ValueError, KeyError):
        return None


def _load_evaluations(data_path: str, max_evaluations: int | None = None) -> list[dict[str, Any]]:
    """Stream parsed eval records from the zstd-compressed JSONL dump.

    The dump is far too large to decompress to disk, so this reads 1 MB compressed chunks,
    splitting on newlines and carrying the trailing partial line in ``buffer`` to the next
    chunk (flushed once at EOF). ``max_evaluations`` caps the number of *parsed* records;
    the ``remaining_needed * 2`` slice is just a throttle assuming roughly half of raw lines
    survive parsing.
    """
    evaluations = []
    evaluations_loaded = 0

    try:
        with open(data_path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            chunk_size = 1024 * 1024
            buffer = ""

            with dctx.stream_reader(f) as reader:
                while True:
                    chunk = reader.read(chunk_size)
                    if not chunk:
                        break

                    buffer += chunk.decode("utf-8")
                    lines = buffer.split("\n")
                    buffer = lines[-1]

                    lines_to_process = lines[:-1]
                    if max_evaluations:
                        remaining_needed = max_evaluations - evaluations_loaded
                        if remaining_needed <= 0:
                            break
                        lines_to_process = lines_to_process[: remaining_needed * 2]

                    for line in tqdm.tqdm(
                        lines_to_process, desc=f"Processing evaluations (loaded: {evaluations_loaded})", leave=False
                    ):
                        if max_evaluations and evaluations_loaded >= max_evaluations:
                            break

                        evaluation = _parse_evaluation_line(line)
                        if evaluation:
                            evaluations.append(evaluation)
                            evaluations_loaded += 1

                    if max_evaluations and evaluations_loaded >= max_evaluations:
                        break

                if buffer.strip() and (not max_evaluations or evaluations_loaded < max_evaluations):
                    evaluation = _parse_evaluation_line(buffer)
                    if evaluation:
                        evaluations.append(evaluation)
                        evaluations_loaded += 1

    except Exception as e:
        raise Exception(f"Error loading evaluations: {e}") from e

    return evaluations


def generate_centipawn_eval_task(
    evaluation: dict[str, Any], category: str, found_counter: dict[str, int]
) -> ChessQuestionAnsweringTask:
    """Build one 5-option centipawn-estimation task from a parsed eval record.

    The true eval, engine depth, and best line go into metadata — the model sees only the
    FEN and the five options.
    """
    task_id = f"position_judgement_{category}_{found_counter[category]:04d}"

    option_values = _find_best_option_set(evaluation["best_evaluation"])
    correct_answer = _get_correct_answer(evaluation["best_evaluation"], option_values)

    prefix = f"You are analyzing a chess position in FEN: {evaluation['fen']}.\n"
    task_description = "Estimate the Stockfish evaluation in centipawns (from White's perspective). Think deeper about this position: Don't just evaluate the current board state. Consider what the most likely moves are for both sides and how the centipawn evaluation would change as the position develops. Analyze a moves ahead - what does the future of this position look like? How would a strong engine assess this position after calculating many moves deep?Analyze step by step and explain your reasoning.\n"
    suffix = f"Choose the closest evaluation from the following options: {', '.join(map(str, option_values))}.\nExample final answers: FORMAT_EXAMPLE_PLACEHOLDER"

    return ChessQuestionAnsweringTask(
        task_id=task_id,
        task_type=f"position_judgement_{category}",
        task_category="Position Judgement",
        input=evaluation["fen"],
        question=construct_prompt(prefix, task_description, suffix),
        format_examples=FORMAT_EXAMPLES_CENTIPAWN,
        correct_answer=correct_answer,
        answer_type="single",
        metadata={
            "position_id": evaluation["position_id"],
            "actual_evaluation": evaluation["best_evaluation"],
            "depth": evaluation["depth"],
            "knodes": evaluation["knodes"],
            "best_line": evaluation["best_line"],
            "category": category,
            "options": option_values,
        },
    )


def find_centipawn_eval_tasks(cfg):
    """Load evals, dedup by position hash, bucket into the five bands, sample per band.

    Each band is shuffled then truncated to cfg.tasks_per_category, so the output is a
    uniform random sample within each evaluation band.
    """
    found_counter = {category: 0 for category in EVAL_CATEGORIES}
    found = []
    position_hashes = set()

    evaluations = _load_evaluations(cfg.data_path, cfg.max_evaluations if cfg.max_evaluations > 0 else None)

    eval_buckets = defaultdict(list)

    for evaluation in tqdm.tqdm(evaluations, desc="Categorizing evaluations"):
        if evaluation["position_id"] in position_hashes:
            continue
        position_hashes.add(evaluation["position_id"])

        category = _determine_evaluation_category(evaluation["best_evaluation"])
        if category is None:
            continue
        eval_buckets[category].append(evaluation)

    for category, evaluations in eval_buckets.items():
        if not evaluations:
            continue

        random.shuffle(evaluations)
        selected_evaluations = evaluations[: cfg.tasks_per_category]

        for evaluation in selected_evaluations:
            if found_counter[category] >= cfg.tasks_per_category:
                break
            task = generate_centipawn_eval_task(evaluation, category, found_counter)
            found_counter[category] += 1
            found.append(task)
            print(f"Found judgement task in category {category}, total found: {found_counter}")

    return found


def parse_args():
    """Parse CLI flags. Defaults use the upstream repo layout — pass explicit paths in this repo."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../../data/raw/lichess_db_eval.jsonl.zst")
    parser.add_argument("--output_root", type=str, default="../../data/benchmark")
    parser.add_argument("--max_evaluations", type=int, default=10000)
    parser.add_argument("--tasks_per_category", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    """Generate the Position Judgment benchmark file from the Lichess eval dump."""
    cfg = parse_args()
    seed_everything(cfg.seed)
    found = find_centipawn_eval_tasks(cfg)
    save_tasks(found, "position_judgement.jsonl", cfg)


if __name__ == "__main__":
    main()
