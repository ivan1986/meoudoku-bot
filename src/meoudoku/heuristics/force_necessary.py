"""Place CAT where marking the cell X would contradict (dual of eliminate)."""

from __future__ import annotations

from meoudoku.board import Board, BoardError, Delta
from meoudoku.heuristics.eliminate_impossible import local_propagate


class ForceNecessary:
    name = "force_necessary"

    def __init__(self, max_k: int = 3) -> None:
        self.max_k = max_k

    def apply(self, board: Board) -> Delta | None:
        # One cat per step so the engine can restart the stack cleanly.
        for r, c in board.candidates():
            trial = board.copy()
            try:
                trial.apply(Delta.of(xs={(r, c)}))
                local_propagate(trial, max_k=self.max_k)
            except BoardError:
                return Delta.of(cats={(r, c)})
            if trial.has_contradiction():
                return Delta.of(cats={(r, c)})
        return None
