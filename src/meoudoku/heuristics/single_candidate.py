"""If a row/col/color has exactly one candidate cell, place a cat there."""

from __future__ import annotations

from meoudoku.board import Board, Delta


class SingleCandidate:
    name = "single_candidate"

    def apply(self, board: Board) -> Delta | None:
        # One cat per step so the engine can restart the stack cleanly.
        for r in range(board.n):
            if board.row_has_cat(r):
                continue
            cand = board.candidates_in_row(r)
            if len(cand) == 1:
                return Delta.of(cats={cand[0]})

        for c in range(board.n):
            if board.col_has_cat(c):
                continue
            cand = board.candidates_in_col(c)
            if len(cand) == 1:
                return Delta.of(cats={cand[0]})

        for color_id in range(board.n):
            if board.color_has_cat(color_id):
                continue
            cand = board.candidates_in_color(color_id)
            if len(cand) == 1:
                return Delta.of(cats={cand[0]})

        return None
