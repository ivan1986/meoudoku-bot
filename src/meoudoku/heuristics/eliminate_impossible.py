"""Mark X on cells where a trial CAT immediately contradicts (unit propagation)."""

from __future__ import annotations

from meoudoku.board import Board, BoardError, Delta
from meoudoku.heuristics.color_col_sets import ColorsConfinedToCols, ColsContainOnlyColors
from meoudoku.heuristics.color_row_sets import ColorsConfinedToRows, RowsContainOnlyColors
from meoudoku.heuristics.enforce_cat import EnforceCatConsequences
from meoudoku.heuristics.row_col_sets import ColsConfinedToRows, RowsConfinedToCols
from meoudoku.heuristics.single_candidate import SingleCandidate


def local_propagate(board: Board, *, max_k: int = 3) -> None:
    """Sound local propagation for trials (no search / eliminate / force)."""
    heuristics = [
        EnforceCatConsequences(),
        SingleCandidate(),
        ColorsConfinedToRows(max_k=max_k),
        RowsContainOnlyColors(max_k=max_k),
        ColorsConfinedToCols(max_k=max_k),
        ColsContainOnlyColors(max_k=max_k),
        RowsConfinedToCols(max_k=max_k),
        ColsConfinedToRows(max_k=max_k),
    ]
    while True:
        if board.has_contradiction():
            return
        progressed = False
        for heuristic in heuristics:
            delta = heuristic.apply(board)
            if delta is not None:
                board.apply(delta)
                progressed = True
                break
        if not progressed:
            return


class EliminateImpossible:
    name = "eliminate_impossible"

    def __init__(self, max_k: int = 3) -> None:
        self.max_k = max_k

    def apply(self, board: Board) -> Delta | None:
        xs: set[tuple[int, int]] = set()
        for r, c in board.candidates():
            trial = board.copy()
            try:
                trial.apply(Delta.of(cats={(r, c)}))
                local_propagate(trial, max_k=self.max_k)
            except BoardError:
                xs.add((r, c))
                continue
            if trial.has_contradiction():
                xs.add((r, c))
        if not xs:
            return None
        return Delta.of(xs=xs)
