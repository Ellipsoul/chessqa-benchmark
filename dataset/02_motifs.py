"""Generator for the Motifs category (benchmark/motifs.jsonl).

Second rung of the abstraction ladder: recognizing named tactical patterns in a static
position (no calculation required — every motif is detectable from the current board alone).

Six task types, all sourced from Lichess puzzle positions, all "multi" answers:
- pin: absolute pins (piece shielding its own king from a slider).
- fork: one piece attacking >= 2 enemy pieces at once.
- battery: >= 2 aligned same-color sliders with nothing between them.
- skewer: slider attacks a valuable piece with a cheaper one behind it on the ray.
- discovered_check / double_check: moves that would produce these (found by trying every
  legal move) — the answers are moves, unlike the other four which are square patterns.

Ground truth comes from hand-rolled geometric detectors in this file, not from an engine;
their precise definitions (documented per detector below) ARE the task semantics.

Usage:
    python dataset/02_motifs.py --puzzle_path data/raw/lichess_db_puzzle.csv \
        --output_root data/benchmark --N_sample 100
"""

import argparse

import chess

from utils import (
    FORMAT_EXAMPLES_BATTERY,
    FORMAT_EXAMPLES_FORK,
    FORMAT_EXAMPLES_LINE,
    FORMAT_EXAMPLES_UCI_MOVE,
    ChessQuestionAnsweringTask,
    construct_prompt,
    read_puzzles,
    save_tasks,
    seed_everything,
)


def is_sliding_piece(piece_type: int) -> bool:
    """True for the long-range pieces (bishop/rook/queen) that can pin, skewer, or form batteries."""
    return piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]


def detect_skewers(board):
    """Find skewers for both colors: (attacker_square, front_square, back_square) triples.

    Definition used: a slider attacks an enemy piece, and continuing along the same ray the
    first piece behind it is another enemy piece of *strictly lower* value (king counts 100,
    so king-front skewers always qualify). Both colors are scanned regardless of whose turn
    it is — the motif is positional, not move-dependent.
    """
    skewers = []
    for color in [chess.WHITE, chess.BLACK]:
        opponent_color = not color
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color and is_sliding_piece(piece.piece_type):
                attacks = board.attacks(square)
                for front_square in attacks:
                    front_piece = board.piece_at(front_square)
                    if not front_piece or front_piece.color != opponent_color:
                        continue
                    from_file, from_rank, file_step, rank_step = _get_direction_steps(square, front_square)
                    if file_step is None:
                        continue
                    # Walk the ray beyond the front piece; the first occupied square decides
                    # (enemy + cheaper = skewer, anything else = blocked, stop either way).
                    to_file, to_rank = chess.square_file(front_square), chess.square_rank(front_square)
                    current_file, current_rank = to_file + file_step, to_rank + rank_step
                    while 0 <= current_file <= 7 and 0 <= current_rank <= 7:
                        back_square = chess.square(current_file, current_rank)
                        back_piece = board.piece_at(back_square)
                        if back_piece:
                            if back_piece.color == opponent_color:
                                piece_values = {
                                    chess.PAWN: 1,
                                    chess.KNIGHT: 3,
                                    chess.BISHOP: 3,
                                    chess.ROOK: 5,
                                    chess.QUEEN: 9,
                                    chess.KING: 100,
                                }
                                if piece_values[front_piece.piece_type] > piece_values[back_piece.piece_type]:
                                    skewers.append(
                                        (
                                            chess.square_name(square),
                                            chess.square_name(front_square),
                                            chess.square_name(back_square),
                                        )
                                    )
                            break
                        current_file += file_step
                        current_rank += rank_step
    return skewers


def generate_skewer_task(board, found_counter, puzzle_id):
    """Build an 'identify all skewers' task; answers use the a>b>c square-chain notation."""
    skewers = detect_skewers(board)
    if not skewers:
        return None
    prefix = f"You are given a chess position in FEN: {board.fen()}.\n"
    task_description = "Identify all skewers in this position. A skewer occurs when a more valuable piece is attacked first and forced to move, exposing a less valuable piece behind it to be captured."
    task_description += " For each skewer, provide the key squares in the format: skewering_piece>front_piece>back_piece (e.g., FORMAT_EXAMPLE_PLACEHOLDER).\n"
    suffix = "If more than one, separate with a comma and a space."
    answer_parts = [
        f"{skewering_square}>{front_square}>{back_square}" for skewering_square, front_square, back_square in skewers
    ]
    correct_answer = ", ".join(answer_parts) if answer_parts else "None"
    return ChessQuestionAnsweringTask(
        task_id=f"motifs_skewer_{found_counter['skewer']:04d}",
        task_type="motifs_skewer",
        task_category="Motifs",
        input=board.fen(),
        question=construct_prompt(prefix, task_description, suffix),
        format_examples=FORMAT_EXAMPLES_LINE,
        correct_answer=correct_answer,
        answer_type="multi",
        metadata={"puzzle_id": puzzle_id, "n_skewers": len(skewers)},
    )


def detect_pins(board: chess.Board) -> list[tuple[str, str, str]]:
    """Find absolute pins for both colors: (pinning, pinned, king) square triples.

    For each enemy slider, walk each of its movement rays collecting the first two occupied
    squares; it's a pin exactly when both belong to the opposing side and the second is the
    king. Only absolute pins (to the king) count — pins to a queen/rook are not detected.
    """
    pins = []
    for color in [chess.WHITE, chess.BLACK]:
        opponent_color = not color
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == opponent_color and is_sliding_piece(piece.piece_type):
                directions = []
                if piece.piece_type == chess.ROOK or piece.piece_type == chess.QUEEN:
                    directions.extend([(0, 1), (0, -1), (1, 0), (-1, 0)])
                if piece.piece_type == chess.BISHOP or piece.piece_type == chess.QUEEN:
                    directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
                for file_step, rank_step in directions:
                    current_file = chess.square_file(square)
                    current_rank = chess.square_rank(square)
                    pieces_on_ray = []
                    while True:
                        current_file += file_step
                        current_rank += rank_step
                        if not (0 <= current_file <= 7 and 0 <= current_rank <= 7):
                            break
                        current_square = chess.square(current_file, current_rank)
                        current_piece = board.piece_at(current_square)
                        if current_piece:
                            pieces_on_ray.append((current_square, current_piece))
                            if len(pieces_on_ray) >= 2:
                                break
                    if len(pieces_on_ray) == 2:
                        pinned_square, pinned_piece = pieces_on_ray[0]
                        target_square, target_piece = pieces_on_ray[1]
                        if (
                            pinned_piece.color == color
                            and target_piece.color == color
                            and target_piece.piece_type == chess.KING
                        ):
                            pins.append(
                                    (
                                        chess.square_name(square),
                                        chess.square_name(pinned_square),
                                        chess.square_name(target_square),
                                    )
                                )
    return pins


def generate_pin_task(board, found_counter, puzzle_id):
    """Build an 'identify all absolute pins' task; answers use the a>b>c square-chain notation."""
    pins = detect_pins(board)
    if not pins:
        return None
    prefix = f"You are given a chess position in FEN: {board.fen()}.\n"
    # Fixed relative to upstream, which reassigned ("=") instead of appending ("+=") on the
    # second line, silently dropping the definition sentence from every pin prompt. The
    # checked-in benchmark/motifs.jsonl predates this fix, so its pin questions carry only
    # the format instruction — regenerated data will differ from the paper's on this task.
    task_description = "Identify all absolute pins in this position. An absolute pin occurs when a piece cannot move because it would expose its own king to check."
    task_description += " For each pin, provide the key squares in the format: pinning_piece>pinned_piece>target_piece (e.g., FORMAT_EXAMPLE_PLACEHOLDER).\n"
    suffix = "If more than one, separate with a comma and a space."
    answer_parts = [f"{pinning}>{pinned}>{target}" for pinning, pinned, target in pins]
    correct_answer = ", ".join(answer_parts) if answer_parts else "None"
    return ChessQuestionAnsweringTask(
        task_id=f"motifs_pin_{found_counter['pin']:04d}",
        task_type="motifs_pin",
        task_category="Motifs",
        input=board.fen(),
        question=construct_prompt(prefix, task_description, suffix),
        format_examples=FORMAT_EXAMPLES_LINE,
        correct_answer=correct_answer,
        answer_type="multi",
        metadata={"puzzle_id": puzzle_id, "n_pins": len(pins)},
    )


def detect_forks(board):
    """Find forks for both colors: (forking_square, [attacked enemy squares]) pairs.

    Any piece (pawns and kings included) attacking two or more enemy pieces counts; attack
    maps are raw ``board.attacks`` — pins, legality, and whether the fork wins material are
    all ignored. That makes this a *pattern recognition* task, not a tactics-quality task.
    """
    forks = []
    for color in [chess.WHITE, chess.BLACK]:
        opponent_color = not color
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                attacks = board.attacks(square)
                attacked_enemy_squares = []
                for attack_square in attacks:
                    attacked_piece = board.piece_at(attack_square)
                    if attacked_piece and attacked_piece.color == opponent_color:
                        attacked_enemy_squares.append(chess.square_name(attack_square))
                if len(attacked_enemy_squares) >= 2:
                    forks.append((chess.square_name(square), attacked_enemy_squares))
    return forks


def generate_fork_task(board, found_counter, puzzle_id):
    """Build an 'identify all forks' task; answers use forker>victim1-victim2 notation."""
    forks = detect_forks(board)
    if not forks:
        return None
    prefix = f"You are given a chess position in FEN: {board.fen()}.\n"
    task_description = "Identify all forks in this position. A fork occurs when one piece attacks two or more enemy pieces simultaneously."
    task_description += " For each fork, provide the key squares in the format: forking_piece>attacked_piece1-attacked_piece2(-attacked_piece3 ...) (e.g., FORMAT_EXAMPLE_PLACEHOLDER).\n"
    suffix = (
        "Order attacked pieces alphabetically (a>h, then 1>8). If more than one, separate with a comma and a space."
    )
    answer_parts = []
    for forking_square, attacked_squares in forks:
        attacked_list = "-".join(sorted(attacked_squares))
        answer_parts.append(f"{forking_square}>{attacked_list}")
    correct_answer = ", ".join(answer_parts)
    assert correct_answer, "Should have at least one fork"
    return ChessQuestionAnsweringTask(
        task_id=f"motifs_fork_{found_counter['fork']:04d}",
        task_type="motifs_fork",
        task_category="Motifs",
        input=board.fen(),
        question=construct_prompt(prefix, task_description, suffix),
        format_examples=FORMAT_EXAMPLES_FORK,
        correct_answer=correct_answer,
        answer_type="multi",
        metadata={"puzzle_id": puzzle_id, "n_forks": len(forks)},
    )


def _can_use_line(piece_type, line_type):
    """True if a piece type moves along the given line type (rook: rank/file, bishop: diagonal, queen: both)."""
    if piece_type == chess.QUEEN:
        return True
    if piece_type == chess.ROOK:
        return line_type in ("rank", "file")
    if piece_type == chess.BISHOP:
        return line_type == "diagonal"
    return False


def _yield_rank_lines():
    """Yield the 8 ranks as ("rank", [squares a-file to h-file]) for battery scanning."""
    for r in range(8):
        yield "rank", [chess.square(f, r) for f in range(8)]


def _yield_file_lines():
    """Yield the 8 files as ("file", [squares rank 1 to 8]) for battery scanning."""
    for f in range(8):
        yield "file", [chess.square(f, r) for r in range(8)]


def _yield_diag_lines():
    """Yield every diagonal of length >= 2 (both directions) for battery scanning.

    Four loops cover: NE-going diagonals starting on rank 1, NE-going starting on the a-file,
    NW-going starting on rank 1, and NW-going starting on the h-file.
    """
    for f0 in range(8):
        line = []
        f, r = f0, 0
        while 0 <= f <= 7 and 0 <= r <= 7:
            line.append(chess.square(f, r))
            f += 1
            r += 1
        if len(line) >= 2:
            yield "diagonal", line
    for r0 in range(1, 8):
        line = []
        f, r = 0, r0
        while 0 <= f <= 7 and 0 <= r <= 7:
            line.append(chess.square(f, r))
            f += 1
            r += 1
        if len(line) >= 2:
            yield "diagonal", line
    for f0 in range(7, -1, -1):
        line = []
        f, r = f0, 0
        while 0 <= f <= 7 and 0 <= r <= 7:
            line.append(chess.square(f, r))
            f -= 1
            r += 1
        if len(line) >= 2:
            yield "diagonal", line
    for r0 in range(1, 8):
        line = []
        f, r = 7, r0
        while 0 <= f <= 7 and 0 <= r <= 7:
            line.append(chess.square(f, r))
            f -= 1
            r += 1
        if len(line) >= 2:
            yield "diagonal", line


def detect_batteries(board):
    """Find batteries: maximal runs of >= 2 same-color sliders on a shared line, nothing between.

    Scans every rank/file/diagonal. A run starts at a slider that can move along that line
    type and extends through empty squares to further compatible friendly sliders; any other
    piece breaks the run. Each battery is reported as the ordered list of its pieces' square
    names, de-duplicated across lines at the end.
    """
    batteries = []
    for line_type, line in list(_yield_rank_lines()) + list(_yield_file_lines()) + list(_yield_diag_lines()):
        i = 0
        while i < len(line):
            sq = line[i]
            p = board.piece_at(sq)
            if not p or p.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
                i += 1
                continue
            color = p.color
            if not _can_use_line(p.piece_type, line_type):
                i += 1
                continue
            # Extend the run: skip empty squares; a compatible friendly slider joins the run,
            # any other piece ends it.
            run = [sq]
            j = i + 1
            while j < len(line):
                sqj = line[j]
                pj = board.piece_at(sqj)
                if pj is None:
                    j += 1
                    continue
                if (
                    pj.color == color
                    and pj.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN)
                    and _can_use_line(pj.piece_type, line_type)
                ):
                    run.append(sqj)
                    j += 1
                    while j < len(line) and board.piece_at(line[j]) is None:
                        j += 1
                    continue
                break
            if len(run) >= 2:
                batteries.append([chess.square_name(x) for x in run])
            i = j if j > i else i + 1
    uniq = []
    seen = set()
    for group in batteries:
        key = tuple(group)
        if key not in seen:
            seen.add(key)
            uniq.append(group)
    return uniq


def generate_battery_task(board, found_counter, puzzle_id):
    """Build an 'identify every battery' task; answers chain each battery's squares with '>'."""
    batteries = detect_batteries(board)
    if not batteries:
        return None
    prefix = f"You are given a chess position in FEN: {board.fen()}.\n"
    task_description = "Identify every battery (2 or more aligned long-range pieces, e.g, RR/RQ/QQ on files or ranks, BQ/BB/QQ on diagonals, of the same color with no pieces between)."
    task_description += " Report each battery as the squares of the pieces in alphabetical order (a>h, 1>8), using '>' to separate squares in a battery and ',' to separate multiple batteries, e.g., FORMAT_EXAMPLE_PLACEHOLDER.\n"
    suffix = "If more than one, separate with a comma and a space."
    answer_parts = [">".join(g) for g in batteries]
    correct_answer = ", ".join(answer_parts)
    assert correct_answer, "Should have at least one battery"
    return ChessQuestionAnsweringTask(
        task_id=f"motifs_battery_{found_counter['battery']:04d}",
        task_type="motifs_battery",
        task_category="Motifs",
        input=board.fen(),
        question=construct_prompt(prefix, task_description, suffix),
        format_examples=FORMAT_EXAMPLES_BATTERY,
        correct_answer=correct_answer,
        answer_type="multi",
        metadata={"puzzle_id": puzzle_id, "n_batteries": len(batteries)},
    )


def _get_direction_steps(from_square, to_square):
    """Return (from_file, from_rank, file_step, rank_step) unit direction between two squares.

    Steps are each -1/0/+1; returns all-None when the squares don't share a rank, file, or
    diagonal (i.e. no slider could travel between them).
    """
    from_file, from_rank = chess.square_file(from_square), chess.square_rank(from_square)
    to_file, to_rank = chess.square_file(to_square), chess.square_rank(to_square)
    file_diff, rank_diff = to_file - from_file, to_rank - from_rank
    if file_diff != 0 and rank_diff != 0 and abs(file_diff) != abs(rank_diff):
        return None, None, None, None
    file_step = 0 if file_diff == 0 else (1 if file_diff > 0 else -1)
    rank_step = 0 if rank_diff == 0 else (1 if rank_diff > 0 else -1)
    return from_file, from_rank, file_step, rank_step


def get_ray_between(from_square, to_square):
    """Return the squares strictly between two aligned squares (empty list if not aligned)."""
    if from_square == to_square:
        return []
    from_file, from_rank, file_step, rank_step = _get_direction_steps(from_square, to_square)
    if file_step is None:
        return []
    ray_squares = []
    current_file, current_rank = from_file + file_step, from_rank + rank_step
    to_file, to_rank = chess.square_file(to_square), chess.square_rank(to_square)
    while current_file != to_file or current_rank != to_rank:
        if 0 <= current_file <= 7 and 0 <= current_rank <= 7:
            ray_squares.append(chess.square(current_file, current_rank))
        current_file += file_step
        current_rank += rank_step
    return ray_squares


def detect_discovered_check_moves(board):
    """Find moves for the side to move that give discovered check: (from, to, checker) triples.

    A discovered check means: after the move, some checker is (a) not the piece that just
    moved, (b) a friendly slider, and (c) was blocked *only* by the moved piece beforehand —
    verified by checking the moved piece sat on the checker->king ray with no other blockers
    on the pre-move board. Unlike the static motif detectors, this searches moves, so only
    the side to move is considered.
    """
    results = []
    color = board.turn
    opp = not color
    king_sq_before = board.king(opp)
    for mv in list(board.legal_moves):
        from_sq = mv.from_square
        to_sq = mv.to_square
        pre = board.copy()
        board.push(mv)
        try:
            if not board.is_check():
                continue
            checkers = list(chess.SquareSet(board.checkers()))
            # Check if any checker is delivering a discovered check
            # (the one that was blocked by the piece that just moved)
            found_discovered = False
            for checker_sq in checkers:
                if checker_sq == to_sq:
                    # This is a direct check, not discovered
                    continue
                cp = board.piece_at(checker_sq)
                if cp is None or cp.color != color or cp.piece_type not in (chess.ROOK, chess.BISHOP, chess.QUEEN):
                    continue
                ray = get_ray_between(checker_sq, king_sq_before)
                if from_sq not in ray:
                    continue
                other_blockers = [s for s in ray if s != from_sq and pre.piece_at(s) is not None]
                if other_blockers:
                    continue
                # This is a discovered check
                if not found_discovered:
                    results.append(
                        (chess.square_name(from_sq), chess.square_name(to_sq), chess.square_name(checker_sq))
                    )
                    found_discovered = True
                    break  # Only record once per move
        finally:
            board.pop()
    return list(dict.fromkeys(results))


def generate_discovered_check_task(board, found_counter, puzzle_id):
    """Build an 'identify all discovered-check moves' task; answers are UCI moves."""
    moves = detect_discovered_check_moves(board)
    if not moves:
        return None
    prefix = f"You are given a chess position in FEN: {board.fen()}.\n"
    task_description = (
        "Identify all discovered-check moves (your move uncovers a check from a rook/bishop/queen on the enemy king)."
    )
    task_description += " Report each as UCI move (e.g., FORMAT_EXAMPLE_PLACEHOLDER).\n"
    suffix = "If more than one, separate with a comma and a space."
    answer_parts = [f"{f}{t}" for (f, t, _) in moves]
    correct_answer = ", ".join(answer_parts)
    assert correct_answer, "Should have at least one discovered check"
    return ChessQuestionAnsweringTask(
        task_id=f"motifs_discovered_check_{found_counter['discovered_check']:04d}",
        task_type="motifs_discovered_check",
        task_category="Motifs",
        input=board.fen(),
        question=construct_prompt(prefix, task_description, suffix),
        format_examples=FORMAT_EXAMPLES_UCI_MOVE,
        correct_answer=correct_answer,
        answer_type="multi",
        metadata={"puzzle_id": puzzle_id, "n_discovered_checks": len(moves)},
    )


def detect_double_check_moves(board):
    """Find moves that give double check (>= 2 checkers afterwards): (from, to, checkers) triples."""
    results = []
    for mv in list(board.legal_moves):
        from_sq = mv.from_square
        to_sq = mv.to_square
        board.push(mv)
        try:
            if not board.is_check():
                continue
            checkers = list(chess.SquareSet(board.checkers()))
            if len(checkers) >= 2:
                results.append(
                    (
                        chess.square_name(from_sq),
                        chess.square_name(to_sq),
                        sorted(chess.square_name(s) for s in checkers),
                    )
                )
        finally:
            board.pop()
    uniq = []
    seen = set()
    for f, t, cs in results:
        key = (f, t, tuple(cs))
        if key not in seen:
            seen.add(key)
            uniq.append((f, t, cs))
    return uniq


def generate_double_check_task(board, found_counter, puzzle_id):
    """Build an 'identify all double-check moves' task; answers are UCI moves."""
    moves = detect_double_check_moves(board)
    if not moves:
        return None
    prefix = f"You are given a chess position in FEN: {board.fen()}.\n"
    task_description = "Identify all moves that deliver a double check (two pieces give check after the move)."
    task_description += " Report each as UCI move (e.g., FORMAT_EXAMPLE_PLACEHOLDER).\n"
    suffix = "If more than one, separate with a comma and a space."
    answer_parts = [f"{f}{t}" for (f, t, cs) in moves]
    correct_answer = ", ".join(answer_parts)
    assert correct_answer, "Should have at least one double check"
    return ChessQuestionAnsweringTask(
        task_id=f"motifs_double_check_{found_counter['double_check']:04d}",
        task_type="motifs_double_check",
        task_category="Motifs",
        input=board.fen(),
        question=construct_prompt(prefix, task_description, suffix),
        format_examples=FORMAT_EXAMPLES_UCI_MOVE,
        correct_answer=correct_answer,
        answer_type="multi",
        metadata={"puzzle_id": puzzle_id, "n_double_checks": len(moves)},
    )


def find_tactical_tasks(unique_positions, data, cfg):
    """Drive the six motif generators until each has cfg.N_sample tasks.

    Same driver pattern as 01_structural: one task max per puzzle position (first generator
    that fires wins, then ``break``), with ``unique_positions`` shared across generators to
    keep positions distinct. Note: no try/except here — the generators' asserts can't fire
    because each returns None before building an empty answer.
    """
    found = []
    found_counter = {"pin": 0, "fork": 0, "battery": 0, "skewer": 0, "discovered_check": 0, "double_check": 0}
    task_generators = {
        "pin": generate_pin_task,
        "fork": generate_fork_task,
        "battery": generate_battery_task,
        "skewer": generate_skewer_task,
        "discovered_check": generate_discovered_check_task,
        "double_check": generate_double_check_task,
    }

    for _, row in data.iterrows():
        puzzle_id = row["PuzzleId"]
        if puzzle_id in unique_positions:
            continue
        board = chess.Board(row["FEN"])
        for task_type, task_generator in task_generators.items():
            if found_counter[task_type] >= cfg.N_sample:
                continue
            task = task_generator(board, found_counter, puzzle_id)
            if task:
                found_counter[task_type] += 1
                found.append(task)
                unique_positions.add(puzzle_id)
                print(f"Found {task.task_type} in puzzle {puzzle_id}, total found: {found_counter}")
                break
        if all([found_counter[t] >= cfg.N_sample for t in found_counter]):
            break
    return found


def parse_args():
    """Parse CLI flags. Defaults use the upstream repo layout — pass explicit paths in this repo."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzle_path", type=str, default="../../data/raw/lichess_db_puzzle.csv")
    parser.add_argument("--output_root", type=str, default="../../data/benchmark")
    parser.add_argument("--N_sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    """Generate the Motifs benchmark file: seed, load puzzles, run detectors, write JSONL."""
    cfg = parse_args()
    seed_everything(cfg.seed)
    data = read_puzzles(cfg.puzzle_path)
    unique_positions = set()
    found = find_tactical_tasks(unique_positions, data, cfg)
    save_tasks(found, "motifs.jsonl", cfg)


if __name__ == "__main__":
    main()
