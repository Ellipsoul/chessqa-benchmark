"""Generator for the Short Tactics category (benchmark/short_tactics.jsonl).

Third rung of the abstraction ladder — and empirically the hardest category (paper mean
17.4%): find the single best move in a Lichess tactics puzzle. Unlike Structural/Motifs,
the answer requires calculation, not just reading the board.

Two samplings of the same underlying task ("best move in UCI"):
- short_tactics_rating_{beginner,intermediate,advanced,expert}: stratified by puzzle Elo
  (<=999 / <=1499 / <=1999 / 2000+), N_sample_rating each — measures difficulty scaling.
- short_tactics_theme_<theme>: stratified by tactical theme from a curated theme list
  (see preprocess.py / all_themes_to_include.json), N_sample_theme each — measures
  per-motif calculation ability.

Quality filters: low rating deviation (stable Elo estimate), high popularity (community
agreement the puzzle is sound), and short solutions (max_pv_length <= 4 plies, so "best
move" is sharply defined). The setup move is applied via make_pre_move, and only the first
solution move is asked for — Lichess guarantees it is the unique winning move (or mate).

Usage:
    python dataset/03_short_tactics.py --puzzle_path data/raw/lichess_db_puzzle.csv \
        --all_themes_path <all_themes_to_include.json> --output_root data/benchmark
"""

import argparse
import json

import tqdm

from utils import (
    FORMAT_EXAMPLES_UCI_MOVE,
    ChessQuestionAnsweringTask,
    construct_prompt,
    make_pre_move,
    read_puzzles,
    save_tasks,
    seed_everything,
)


def rating_to_level(rating):
    """Bucket a Lichess puzzle Elo into beginner/intermediate/advanced/expert difficulty bands."""
    if rating <= 999:
        return "beginner"
    elif rating <= 1499:
        return "intermediate"
    elif rating <= 1999:
        return "advanced"
    else:
        return "expert"


def find_puzzles_by_rating(unique_puzzles, data, config):
    """Collect config.N_sample_rating best-move tasks per difficulty band.

    Iterates the shuffled puzzle table, applying the quality filters (rating deviation,
    popularity, solution length), skipping already-used puzzles, and filling each band
    until full. The full solution line is kept in metadata (``pv``) even though only the
    first move is the answer — useful later for reasoning-trace analysis.

    Returns ``(tasks, unique_puzzles)`` so the theme pass can keep excluding these puzzles.
    """
    found = []
    found_counter = {"beginner": 0, "intermediate": 0, "advanced": 0, "expert": 0}

    for _, row in tqdm.tqdm(data.iterrows()):
        rating_deviation = row["RatingDeviation"]
        popularity = row["Popularity"]
        pv_length = len(row["Moves"].split(" "))

        if (
            rating_deviation > config.max_rating_deviation
            or popularity < config.min_popularity
            or pv_length > config.max_pv_length
        ):
            continue

        puzzle_id = row["PuzzleId"]
        if puzzle_id in unique_puzzles:
            continue

        fen, move = make_pre_move(row)
        task_name = rating_to_level(row["Rating"])

        if found_counter[task_name] < config.N_sample_rating:
            prefix = f"You are given a chess position in FEN: {fen}.\n"
            task_description = "Find the best move for the side to play.\n"
            suffix = "Use UCI notation (e.g., FORMAT_EXAMPLE_PLACEHOLDER) for the final answer."

            sample = ChessQuestionAnsweringTask(
                task_id=f"short_tactics_rating_{task_name}_{found_counter[task_name]:04d}",
                task_type=f"short_tactics_rating_{task_name}",
                task_category="Short Tactics",
                input=fen,
                question=construct_prompt(prefix, task_description, suffix),
                format_examples=FORMAT_EXAMPLES_UCI_MOVE,
                correct_answer=move,
                answer_type="single",
                metadata={
                    "puzzle_id": puzzle_id,
                    "rating": row["Rating"],
                    "themes": row["Themes"].split(" "),
                    "rating_deviation": rating_deviation,
                    "popularity": popularity,
                    "pv_length": pv_length,
                    "pv": row["Moves"],
                },
            )
            found_counter[task_name] = found_counter.get(task_name, 0) + 1
            found.append(sample)
            unique_puzzles.add(puzzle_id)

        if all(count >= config.N_sample_rating for count in found_counter.values()):
            break

    print(found_counter, flush=True)

    return found, unique_puzzles


def find_puzzles_by_theme(unique_puzzles, data, config):
    """Collect config.N_sample_theme best-move tasks per curated tactical theme.

    Same filtering as the rating pass. A puzzle usually carries several theme tags; it is
    assigned to the first still-unfilled tag it intersects (iteration order of the set —
    fixed by PYTHONHASHSEED via seed_everything), recorded as ``primary_theme`` in metadata.
    """
    with open(config.all_themes_path) as themes_file:
        all_themes_to_include = set(json.load(themes_file))

    all_themes_to_include = set(all_themes_to_include)
    found_counter = {theme: 0 for theme in all_themes_to_include}
    found = []

    for _, row in tqdm.tqdm(data.iterrows()):
        rating_deviation = row["RatingDeviation"]
        popularity = row["Popularity"]
        pv_length = len(row["Moves"].split(" "))

        if (
            rating_deviation > config.max_rating_deviation
            or popularity < config.min_popularity
            or pv_length > config.max_pv_length
        ):
            continue

        puzzle_id = row["PuzzleId"]
        if puzzle_id in unique_puzzles:
            continue

        fen, move = make_pre_move(row)
        puzzle_themes = set(row["Themes"].split(" "))
        intersecting_themes = puzzle_themes.intersection(all_themes_to_include)
        if not intersecting_themes:
            continue

        for primary_theme in intersecting_themes:
            if found_counter[primary_theme] < config.N_sample_theme:
                prefix = f"You are given a chess position in FEN: {fen}.\n"
                task_description = "Find the best move for the side to play.\n"
                suffix = "Use UCI notation (e.g., FORMAT_EXAMPLE_PLACEHOLDER) for the final answer."

                sample = ChessQuestionAnsweringTask(
                    task_id=f"short_tactics_theme_{primary_theme}_{found_counter[primary_theme]:04d}",
                    task_type=f"short_tactics_theme_{primary_theme}",
                    task_category="Short Tactics",
                    input=fen,
                    question=construct_prompt(prefix, task_description, suffix),
                    format_examples=FORMAT_EXAMPLES_UCI_MOVE,
                    correct_answer=move,
                    answer_type="single",
                    metadata={
                        "puzzle_id": puzzle_id,
                        "rating": row["Rating"],
                        "themes": row["Themes"].split(" "),
                        "primary_theme": primary_theme,
                        "rating_deviation": rating_deviation,
                        "popularity": popularity,
                        "pv_length": pv_length,
                        "pv": row["Moves"],
                    },
                )
                found_counter[primary_theme] = found_counter.get(primary_theme, 0) + 1
                found.append(sample)
                unique_puzzles.add(puzzle_id)
                break

        if all(count >= config.N_sample_theme for count in found_counter.values()):
            break

    print(found_counter, flush=True)

    return found


def parse_args():
    """Parse CLI flags. Defaults use the upstream repo layout — pass explicit paths in this repo."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzle_path", type=str, default="../../data/raw/lichess_db_puzzle.csv")
    parser.add_argument("--all_themes_path", type=str, default="../../data/info/all_themes_to_include.json")
    parser.add_argument("--output_root", type=str, default="../../data/benchmark")
    parser.add_argument("--N_sample_rating", type=int, default=100)
    parser.add_argument("--N_sample_theme", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_popularity", type=int, default=90)
    parser.add_argument("--max_pv_length", type=int, default=4)
    parser.add_argument("--max_rating_deviation", type=int, default=100)

    return parser.parse_args()


def main():
    """Generate the Short Tactics benchmark file: rating-stratified pass, then theme pass."""
    config = parse_args()
    seed_everything(config.seed)
    data = read_puzzles(config.puzzle_path)

    unique_puzzles = set()
    found_rating, unique_puzzles = find_puzzles_by_rating(unique_puzzles, data, config)
    found_theme = find_puzzles_by_theme(unique_puzzles, data, config)
    found = found_rating + found_theme

    save_tasks(found, "short_tactics.jsonl", config)


if __name__ == "__main__":
    main()
