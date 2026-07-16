"""Shared utilities for the ChessQA dataset generators (dataset/0N_*.py).

This module sits at the front of the data flow:

    raw Lichess dumps -> dataset/0N_*.py generators -> benchmark JSONL -> eval/run_benchmark.py

It provides the pieces every generator needs:

- ``ChessQuestionAnsweringTask``: the task record schema serialized into the benchmark JSONL.
- ``FORMAT_EXAMPLES_*`` constants: the two answer-format examples embedded in each task
  (the eval runner picks one via ``--use-format-example-group`` to test format sensitivity).
- FEN/board helpers (``get_piece_arrangement``, ``fen_to_pieces``, ``get_piece_name``) used both
  to build questions and to inject board-state context at inference time (``--add-context``).
- ``construct_prompt``: assembles a question with the literal placeholder strings that the eval
  runner substitutes at inference time (see ``format_prompt`` in eval/run_benchmark.py).
- I/O and reproducibility helpers (``save_tasks``, ``read_puzzles``, ``seed_everything``).
"""

import json
import os
import random
import time
from dataclasses import asdict, dataclass

import chess
import numpy as np
import pandas as pd

# Each FORMAT_EXAMPLES_* list holds exactly two example answers for one answer syntax.
# They are stored per-task in `format_examples` and resolved into the prompt at inference
# time (FORMAT_EXAMPLE_PLACEHOLDER), so a single benchmark file supports two prompt variants.
# Notation used by the examples:
#   ","  separates multiple answers in a multi-answer task ("multi" answer_type, set-compared)
#   ">"  chains squares along a line/battery (e.g. "d1>d7>d8" = piece path through squares)
#   "-"  separates the two forked targets in a fork answer (e.g. "e5>e7-f6")
FORMAT_EXAMPLES_UCI_MOVE = ["e2e4, c2b1q", "g1f3, a2a1q"]
FORMAT_EXAMPLES_UCI_MOVE_SAME_START = ["e2e3, e2e4", "d7d5, d7d6"]
FORMAT_EXAMPLES_LINE = ["d1>d7>d8, a2>e2>h2", "a5>e5>h5, h1>h4>h7"]
FORMAT_EXAMPLES_FORK = ["e5>e7-f6, f3>e1-g1", "e4>b1-g2, c4>b2-d6"]
FORMAT_EXAMPLES_BATTERY = ["b2>e5>h8, h1>h7", "h1>h4, a1>a4>a8"]
FORMAT_EXAMPLES_FEN = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r2qr1k1/1b1p1ppp/p2Q1n2/1p6/8/1BN2n2/PPP2PPP/R1B1R1K1 w - - 1 15",
]
FORMAT_EXAMPLES_CENTIPAWN = ["400", "-200"]
FORMAT_EXAMPLES_SQUARES = ["e4, f5", "a1, b2, c3"]
FORMAT_EXAMPLES_PIECE = ["White Queen at e5", "Black Rook at d5"]
FORMAT_EXAMPLES_ARRANGEMENT = [
    "White King: ['e1'], White Queen: ['d1'], White Rook: ['a1', 'h1'], White Bishop: ['c1', 'f1'], White Knight: ['b1', 'g1'], White Pawn: ['a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2'], Black King: ['e8'], Black Queen: ['d8'], Black Rook: ['a8', 'h8'], Black Bishop: ['c8', 'f8'], Black Knight: ['b8', 'g8'], Black Pawn: ['a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7']",
    "White King: ['g1'], White Queen: ['d6'], White Rook: ['a1', 'e1'], White Bishop: ['b3', 'c1'], White Knight: ['c3'], White Pawn: ['a2', 'b2', 'c2', 'f2', 'g2', 'h2'], Black King: ['g8'], Black Queen: ['d8'], Black Rook: ['a8', 'e8'], Black Bishop: ['b7'], Black Knight: ['f3', 'f6'], Black Pawn: ['a6', 'b5', 'd7', 'f7', 'g7', 'h7']",
]
FORMAT_EXAMPLES_MCQ = ["A", "D"]


# Maps python-chess (piece_type, color) pairs to the human-readable names used in prompts
# and piece-arrangement strings. Kept as a module-level table so wording is identical
# everywhere it appears (prompt text is part of the benchmark contract).
PIECE_NAMES = {
    (chess.PAWN, chess.WHITE): "White Pawn",
    (chess.KNIGHT, chess.WHITE): "White Knight",
    (chess.BISHOP, chess.WHITE): "White Bishop",
    (chess.ROOK, chess.WHITE): "White Rook",
    (chess.QUEEN, chess.WHITE): "White Queen",
    (chess.KING, chess.WHITE): "White King",
    (chess.PAWN, chess.BLACK): "Black Pawn",
    (chess.KNIGHT, chess.BLACK): "Black Knight",
    (chess.BISHOP, chess.BLACK): "Black Bishop",
    (chess.ROOK, chess.BLACK): "Black Rook",
    (chess.QUEEN, chess.BLACK): "Black Queen",
    (chess.KING, chess.BLACK): "Black King",
}


def get_piece_arrangement(fen):
    """Render a FEN as a human-readable piece-arrangement string.

    This is the string injected into prompts by ``--add-context`` — the paper's key
    intervention showing that spelling out the board state significantly improves scores
    (i.e. board-state hallucination is a core bottleneck).

    Args:
        fen: Position in Forsyth-Edwards Notation.

    Returns:
        A string like ``"White King: ['e1'], White Queen: ['d1'], ..."`` listing White pieces
        first (King, Queen, Rook, Bishop, Knight, Pawn), then Black in the same order, with
        each piece type's squares sorted alphabetically. Piece types absent from the board
        are omitted entirely. See ``FORMAT_EXAMPLES_ARRANGEMENT`` for full examples.
    """
    board = chess.Board(fen)

    # Define piece order
    piece_order = ["King", "Queen", "Rook", "Bishop", "Knight", "Pawn"]
    colors = ["White", "Black"]

    pieces = {}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = "White" if piece.color == chess.WHITE else "Black"
            names = {
                1: "Pawn",
                2: "Knight",
                3: "Bishop",
                4: "Rook",
                5: "Queen",
                6: "King",
            }
            piece_key = f"{color} {names[piece.piece_type]}"
            if piece_key not in pieces:
                pieces[piece_key] = []
            pieces[piece_key].append(chess.square_name(square))

    # Sort squares alphabetically for each piece type
    for piece_key in pieces:
        pieces[piece_key].sort()

    # Build arrangement string in specified order
    arrangement_parts = []
    for color in colors:
        for piece_type in piece_order:
            piece_key = f"{color} {piece_type}"
            if piece_key in pieces:
                arrangement_parts.append(f"{piece_key}: {pieces[piece_key]}")

    return ", ".join(arrangement_parts)


@dataclass
class ChessQuestionAnsweringTask:
    """One benchmark item; serialized as one JSON line in benchmark/<category>.jsonl.

    Attributes:
        task_id: Unique id, conventionally ``"<task_type>_<index>"``.
        task_type: Fine-grained question kind (e.g. ``"piece_arrangement"``, ``"fork"``).
        task_category: One of the five paper categories (Structural, Motifs, Short Tactics,
            Position Judgment, Semantic) — the unit results are reported over.
        input: The position. Usually a bare FEN; some tasks use ``"FEN | uci moves"`` where
            the moves after ``|`` are to be applied to the FEN (consumers must strip after
            ``|`` before parsing the FEN itself).
        question: Prompt text containing the literal placeholders ``CONTEXT_PLACEHOLDER`` and
            (via the suffix) ``FORMAT_EXAMPLE_PLACEHOLDER``, resolved by the eval runner's
            ``format_prompt`` at inference time. Keep the placeholders intact in the JSONL.
        format_examples: Two example answers (one per prompt-variant group); see the
            ``FORMAT_EXAMPLES_*`` constants.
        correct_answer: Ground-truth answer string.
        answer_type: ``"single"`` = exact string match; ``"multi"`` = comma-separated,
            compared as a set (order-insensitive).
        metadata: Optional extras for analysis (e.g. source puzzle rating/themes); not
            shown to the model and not used in scoring.
    """

    task_id: str
    task_type: str
    task_category: str
    input: str
    question: str
    format_examples: list[str]
    correct_answer: str
    answer_type: str
    metadata: dict | None


def construct_prompt(prefix, task_description, suffix):
    """Assemble a task's `question` string with placeholders left unresolved.

    Layout: ``prefix`` (typically names the position/FEN) + ``CONTEXT_PLACEHOLDER`` (replaced
    at inference time with piece arrangement + legal moves under ``--add-context``, or removed
    otherwise) + ``task_description`` + fixed chain-of-thought/answer-format boilerplate +
    ``suffix`` (typically carries ``FORMAT_EXAMPLE_PLACEHOLDER``).

    The ``"FINAL ANSWER: <answer>"`` line mandated here is what the eval runner's extractor
    looks for when scoring — the two ends of the contract must stay in sync.
    """
    question = prefix
    question += "CONTEXT_PLACEHOLDER"
    question += task_description
    question += "Analyze step by step and explain your reasoning.\n"
    question += "Finish with a single line formatted EXACTLY as:\n"
    question += "FINAL ANSWER: <answer>\n"
    question += suffix

    return question


def make_pre_move(row: pd.Series) -> tuple[str, str]:
    """Advance a Lichess puzzle row by its setup move.

    Lichess puzzle CSVs store the position *before* the opponent's last move: the first entry
    in ``Moves`` is that setup move, and the puzzle actually starts after it is played.
    Generators therefore push ``moves[0]`` onto the board and treat ``moves[1]`` as the first
    solution move.

    Args:
        row: Puzzle CSV row with ``FEN`` and space-separated UCI ``Moves`` columns.

    Returns:
        ``(fen_after_setup_move, first_solution_move_uci)``.
    """
    fen_before = row["FEN"]
    moves = row["Moves"].split(" ")
    board = chess.Board(fen_before)
    pre_move = chess.Move.from_uci(moves[0])
    board.push(pre_move)
    fen_after = board.fen()

    return fen_after, moves[1]


def save_tasks(tasks, file_name, config):
    """Write tasks as JSONL (one ``ChessQuestionAnsweringTask`` dict per line) under ``config.output_root``."""
    if not os.path.exists(config.output_root):
        os.makedirs(config.output_root)
    output_path = os.path.join(config.output_root, file_name)
    tasks_data = [asdict(task) for task in tasks]

    with open(output_path, "w", encoding="utf-8") as output_file:
        for task in tasks_data:
            output_file.write(json.dumps(task, ensure_ascii=False) + "\n")


def get_piece_name(piece: chess.Piece) -> str:
    """Return the prompt-facing name of a piece, e.g. ``"White Queen"``."""
    color_name = "White" if piece.color == chess.WHITE else "Black"
    piece_names = {
        chess.PAWN: "Pawn",
        chess.KNIGHT: "Knight",
        chess.BISHOP: "Bishop",
        chess.ROOK: "Rook",
        chess.QUEEN: "Queen",
        chess.KING: "King",
    }
    piece_name = piece_names[piece.piece_type]
    return f"{color_name} {piece_name}"


def fen_to_pieces(fen):
    """Map a FEN to ``{"White Queen": ["d1"], ...}`` in board-scan (a1..h8) order.

    Structured counterpart of ``get_piece_arrangement``: same information, but returned as a
    dict for generators that need to iterate pieces rather than embed a display string.
    """
    board = chess.Board(fen)
    pieces = {}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_key = PIECE_NAMES[(piece.piece_type, piece.color)]
            square_name = chess.square_name(square)

            if piece_key not in pieces:
                pieces[piece_key] = []
            pieces[piece_key].append(square_name)

    return pieces


def seed_everything(seed):
    """Seed Python, NumPy, and (if installed) PyTorch for reproducible dataset generation."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def readable_num(number):
    """Format a count with a B/M/K suffix for progress logs, e.g. ``1_500_000 -> "1.50M"``."""
    if number >= 1e9:
        return f"{number / 1e9:.2f}B"
    elif number >= 1e6:
        return f"{number / 1e6:.2f}M"
    elif number >= 1e3:
        return f"{number / 1e3:.2f}K"
    else:
        return str(number)


def readable_time(elapsed_time):
    """Format seconds as ``"1h 2m 3.00s"`` / ``"2m 3.00s"`` / ``"3.00s"`` for progress logs."""
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes > 0:
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"


def read_puzzles(file_path):
    """Load the Lichess puzzle CSV and shuffle row order.

    The shuffle uses pandas' global random state, so ``seed_everything`` must be called first
    for reproducible sampling. The CSV is large (~5M rows), hence the timing printouts.
    """
    read_start_time = time.time()
    puzzle_dataframe = pd.read_csv(file_path)
    shuffle_start_time = time.time()
    puzzle_dataframe = puzzle_dataframe.sample(frac=1).reset_index(drop=True)
    print(f"Time to read: {readable_time(shuffle_start_time - read_start_time)}", flush=True)
    print(f"Time to shuffle: {readable_time(time.time() - shuffle_start_time)}", flush=True)

    return puzzle_dataframe
