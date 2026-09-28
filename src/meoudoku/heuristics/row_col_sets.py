"""Row ↔ column pigeonhole heuristics (no colors)."""

from __future__ import annotations

from itertools import combinations

from meoudoku.board import Board, Delta


def _open_rows(board: Board) -> list[int]:
    return [r for r in range(board.n) if not board.row_has_cat(r)]


def _open_cols(board: Board) -> list[int]:
    return [c for c in range(board.n) if not board.col_has_cat(c)]


def _rows_confined_to_cols(board: Board, max_k: int) -> Delta | None:
    """
    If K open rows have candidates only in K columns,
    mark X on other rows' candidates inside those columns.
    """
    open_rows = _open_rows(board)
    xs: set[tuple[int, int]] = set()
    for k in range(1, min(max_k, len(open_rows)) + 1):
        for row_set in combinations(open_rows, k):
            row_set_f = frozenset(row_set)
            cells: list[tuple[int, int]] = []
            for r in row_set:
                cells.extend(board.candidates_in_row(r))
            if not cells:
                continue
            cols = {c for _, c in cells}
            if len(cols) != k:
                continue
            for c in cols:
                for r, cc in board.candidates_in_col(c):
                    if r not in row_set_f:
                        xs.add((r, cc))
    if not xs:
        return None
    return Delta.of(xs=xs)


def _cols_confined_to_rows(board: Board, max_k: int) -> Delta | None:
    """
    If K open columns have candidates only in K rows,
    mark X on other columns' candidates inside those rows.
    """
    open_cols = _open_cols(board)
    xs: set[tuple[int, int]] = set()
    for k in range(1, min(max_k, len(open_cols)) + 1):
        for col_set in combinations(open_cols, k):
            col_set_f = frozenset(col_set)
            cells: list[tuple[int, int]] = []
            for c in col_set:
                cells.extend(board.candidates_in_col(c))
            if not cells:
                continue
            rows = {r for r, _ in cells}
            if len(rows) != k:
                continue
            for r in rows:
                for rr, c in board.candidates_in_row(r):
                    if c not in col_set_f:
                        xs.add((rr, c))
    if not xs:
        return None
    return Delta.of(xs=xs)


class RowsConfinedToCols:
    name = "rows_confined_to_cols"

    def __init__(self, max_k: int = 3) -> None:
        self.max_k = max_k

    def apply(self, board: Board) -> Delta | None:
        return _rows_confined_to_cols(board, self.max_k)


class ColsConfinedToRows:
    name = "cols_confined_to_rows"

    def __init__(self, max_k: int = 3) -> None:
        self.max_k = max_k

    def apply(self, board: Board) -> Delta | None:
        return _cols_confined_to_rows(board, self.max_k)
