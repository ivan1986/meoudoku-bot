"""Shared helpers for color/row/col set confinement heuristics."""

from __future__ import annotations

from itertools import combinations

from meoudoku.board import Board, Delta


def _open_colors(board: Board) -> list[int]:
    return [cid for cid in range(board.n) if not board.color_has_cat(cid)]


def _open_rows(board: Board) -> list[int]:
    return [r for r in range(board.n) if not board.row_has_cat(r)]


def _open_cols(board: Board) -> list[int]:
    return [c for c in range(board.n) if not board.col_has_cat(c)]


def colors_confined_to_lines(
    board: Board,
    *,
    line_of: str,
    max_k: int = 3,
) -> Delta | None:
    """
    If K open colors have candidate cells only in K lines (rows or cols),
    mark X on other colors' candidates inside those lines.
    """
    open_colors = _open_colors(board)
    xs: set[tuple[int, int]] = set()

    for k in range(1, min(max_k, len(open_colors)) + 1):
        for color_set in combinations(open_colors, k):
            color_set_f = frozenset(color_set)
            cells: list[tuple[int, int]] = []
            for cid in color_set:
                cells.extend(board.candidates_in_color(cid))
            if not cells:
                continue
            if line_of == "row":
                lines = {r for r, _ in cells}
            else:
                lines = {c for _, c in cells}
            if len(lines) != k:
                continue
            for line in lines:
                if line_of == "row":
                    for r, c in board.candidates_in_row(line):
                        if board.color(r, c) not in color_set_f:
                            xs.add((r, c))
                else:
                    for r, c in board.candidates_in_col(line):
                        if board.color(r, c) not in color_set_f:
                            xs.add((r, c))
    if not xs:
        return None
    return Delta.of(xs=xs)


def lines_contain_only_colors(
    board: Board,
    *,
    line_of: str,
    max_k: int = 3,
) -> Delta | None:
    """
    If K open lines contain only K colors among candidates,
    mark X on those colors' candidates outside the K lines.
    """
    open_lines = _open_rows(board) if line_of == "row" else _open_cols(board)
    xs: set[tuple[int, int]] = set()

    for k in range(1, min(max_k, len(open_lines)) + 1):
        for line_set in combinations(open_lines, k):
            line_set_f = frozenset(line_set)
            colors_present: set[int] = set()
            for line in line_set:
                cells = board.candidates_in_row(line) if line_of == "row" else board.candidates_in_col(line)
                for r, c in cells:
                    colors_present.add(board.color(r, c))
            if len(colors_present) != k:
                continue
            for cid in colors_present:
                for r, c in board.candidates_in_color(cid):
                    line_id = r if line_of == "row" else c
                    if line_id not in line_set_f:
                        xs.add((r, c))
    if not xs:
        return None
    return Delta.of(xs=xs)
