"""Col ↔ color set confinement heuristics."""

from __future__ import annotations

from meoudoku.board import Board, Delta
from meoudoku.heuristics.sets import colors_confined_to_lines, lines_contain_only_colors


class ColorsConfinedToCols:
    name = "colors_confined_to_cols"

    def __init__(self, max_k: int = 3) -> None:
        self.max_k = max_k

    def apply(self, board: Board) -> Delta | None:
        return colors_confined_to_lines(board, line_of="col", max_k=self.max_k)


class ColsContainOnlyColors:
    name = "cols_contain_only_colors"

    def __init__(self, max_k: int = 3) -> None:
        self.max_k = max_k

    def apply(self, board: Board) -> Delta | None:
        return lines_contain_only_colors(board, line_of="col", max_k=self.max_k)
