from pathlib import Path

import pytest

from meoudoku.board import CellState, Delta, load_board, parse_board
from meoudoku.engine import Solver, UnsolvedError
from meoudoku.heuristics import EnforceCatConsequences, default_stack
from meoudoku.heuristics.backtrack import BacktrackHeuristic, pick_mrv_cells

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_engine_rewinds_to_first_heuristic():
    order: list[str] = []

    class First:
        name = "first"

        def apply(self, board):
            order.append(self.name)
            if board.get(0, 0) is CellState.EMPTY and not board.cat_positions():
                return Delta.of(cats={(0, 0)})
            return None

    class Second:
        name = "second"

        def apply(self, board):
            order.append(self.name)
            return None

    board = parse_board(
        """\
COLORS
a b c d
a b c d
a b c d
a b c d
STATE
. . . .
. . . .
. . . .
. . . .
"""
    )
    solver = Solver([First(), Second(), EnforceCatConsequences()])
    hit1 = solver.step(board)
    assert hit1 is not None and hit1[0] == "first"
    board.apply(hit1[1])
    order.clear()
    hit2 = solver.step(board)
    assert order[0] == "first"
    assert hit2 is not None
    assert hit2[0] == "enforce_cat_consequences"


def test_solve_level_tiny():
    board = load_board(FIXTURES / "level_tiny.txt")
    result = Solver().solve(board)  # deduction only; no backtrack
    assert result.board.is_solved()
    assert len(result.board.cat_positions()) == 4


def test_solve_level_sample():
    board = load_board(FIXTURES / "level_sample.txt")
    result = Solver(default_stack(include_backtrack=True)).solve(board)
    assert result.board.is_solved()
    assert len(result.board.cat_positions()) == 5


def test_backtrack_returns_full_mrv_list():
    board = load_board(FIXTURES / "level_sample.txt")
    # Strip to a mid-state where deduction stalls (same as phone would see).
    solver = Solver(default_stack(include_backtrack=False))
    work = board.copy()
    for _ in range(50):
        hit = solver.step(work)
        if hit is None:
            break
        work.apply(hit[1])
    cells = pick_mrv_cells(work)
    assert cells is not None and len(cells) >= 2
    delta = BacktrackHeuristic().apply(work)
    assert delta is not None
    assert set(delta.cats) == set(cells)
    assert len(delta.cats) == len(cells)


def test_contradiction_unsolved():
    board = parse_board(
        """\
COLORS
a b c d
a b c d
a b c d
a b c d
STATE
C C . .
. . . .
. . . .
. . . .
"""
    )
    with pytest.raises(UnsolvedError):
        Solver(default_stack(include_backtrack=True)).solve(board)


def test_enforce_then_single_progress_on_partial_tiny():
    board = load_board(FIXTURES / "level_tiny.txt")
    board.apply(Delta.of(cats={(0, 2)}))
    solver = Solver([EnforceCatConsequences()])
    hit = solver.step(board)
    assert hit is not None
    assert hit[0] == "enforce_cat_consequences"
    board.apply(hit[1])
    assert board.get(0, 0) is CellState.X or board.get(0, 1) is CellState.X
