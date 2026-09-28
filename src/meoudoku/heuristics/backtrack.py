"""Last-resort heuristic: return full MRV candidate list (search is engine-only)."""

from __future__ import annotations

from meoudoku.board import Board, Delta


class BacktrackHeuristic:
    name = "backtrack"

    def apply(self, board: Board) -> Delta | None:
        """
        Return every MRV-unit candidate as cats (ordered list via candidates()).
        Engine must branch over the list in memory — never apply the whole Delta
        and never tap trials on a physical board.
        """
        cells = self.candidates(board)
        if not cells:
            return None
        return Delta.of(cats=set(cells))

    def candidates(self, board: Board) -> list[tuple[int, int]] | None:
        return pick_mrv_cells(board)


def pick_mrv_cells(board: Board) -> list[tuple[int, int]] | None:
    """All candidates of the most constrained open unit (row/col/color)."""
    best: list[tuple[int, int]] | None = None

    def consider(cells: list[tuple[int, int]]) -> None:
        nonlocal best
        if not cells:
            return
        if best is None or len(cells) < len(best):
            best = cells

    for r in range(board.n):
        if not board.row_has_cat(r):
            consider(board.candidates_in_row(r))
    for c in range(board.n):
        if not board.col_has_cat(c):
            consider(board.candidates_in_col(c))
    for color_id in range(board.n):
        if not board.color_has_cat(color_id):
            consider(board.candidates_in_color(color_id))

    return best


# Backward-compatible alias (first cell only — prefer pick_mrv_cells / candidates()).
def pick_mrv_cell(board: Board) -> tuple[int, int] | None:
    cells = pick_mrv_cells(board)
    return None if not cells else cells[0]
