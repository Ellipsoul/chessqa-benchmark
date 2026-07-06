"""One-off exploration script: count Lichess puzzle theme tags.

Scans the raw puzzle CSV and writes ``<output_root>/info/all_themes.json`` mapping each theme
tag (e.g. ``"fork"``, ``"mateIn2"``) to its puzzle count. Not part of the benchmark data flow —
it informed which themes the Motifs/Short Tactics generators (02, 03) filter on.

Usage:
    python dataset/preprocess.py --puzzle_path data/raw/lichess_db_puzzle.csv --output_root <dir>

Note: the ``info/`` subdirectory of ``--output_root`` must already exist.
"""

import argparse
import json
import os
from pathlib import Path

import tqdm

from utils import read_puzzles


def find_all_themes(data, config):
    """Tally every space-separated theme tag across the puzzle DataFrame and dump to JSON.

    Args:
        data: Puzzle DataFrame with a ``Themes`` column of space-separated tags.
        config: Parsed CLI args providing ``output_root``.

    Returns:
        Dict mapping theme tag -> occurrence count.
    """
    all_themes = {}
    for _, row in tqdm.tqdm(data.iterrows()):
        themes = row["Themes"]
        if themes:
            themes_list = themes.split(" ")
            for theme in themes_list:
                all_themes[theme] = all_themes.get(theme, 0) + 1

    with open(os.path.join(config.output_root, "info", "all_themes.json"), "w") as themes_file:
        json.dump(all_themes, themes_file, indent=4)

    return all_themes


def parse_args():
    """Parse CLI flags. Note the defaults resolve relative to the CWD (upstream layout); pass explicit paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzle_path", type=Path, default="../../data/raw/lichess_db_puzzle.csv")
    parser.add_argument("--output_root", type=Path, default="chess-llm-benchmark/data")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    """Load puzzles, count themes, report the number of unique tags found."""
    config = parse_args()
    puzzle_data = read_puzzles(config.puzzle_path)
    all_themes = find_all_themes(puzzle_data, config)
    print(f"Found {len(all_themes)} unique themes.")


if __name__ == "__main__":
    main()
