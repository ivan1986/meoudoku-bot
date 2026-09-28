"""Game rule: after each CAT, mark row/col/color and 8-neighbors with X."""

from __future__ import annotations

from meoudoku.board import Board, CellState, Delta


class EnforceCatConsequences:
    name = "enforce_cat_consequences"

    def apply(self, board: Board) -> Delta | None:
        xs: set[tuple[int, int]] = set()
        for r, c in board.cat_positions():
            color_id = board.color(r, c)
            for cc in range(board.n):
                if board.get(r, cc) is CellState.EMPTY:
                    xs.add((r, cc))
            for rr in range(board.n):
                if board.get(rr, c) is CellState.EMPTY:
                    xs.add((rr, c))
            for rr in range(board.n):
                for cc in range(board.n):
                    if board.color(rr, cc) == color_id and board.get(rr, cc) is CellState.EMPTY:
                        xs.add((rr, cc))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if board.in_bounds(rr, cc) and board.get(rr, cc) is CellState.EMPTY:
                        xs.add((rr, cc))
        if not xs:
            return None
        return Delta.of(xs=xs)
