"""Default ordered heuristic stack."""

from __future__ import annotations

from meoudoku.heuristics.backtrack import BacktrackHeuristic
from meoudoku.heuristics.base import Heuristic
from meoudoku.heuristics.color_col_sets import ColorsConfinedToCols, ColsContainOnlyColors
from meoudoku.heuristics.color_row_sets import ColorsConfinedToRows, RowsContainOnlyColors
from meoudoku.heuristics.eliminate_impossible import EliminateImpossible
from meoudoku.heuristics.enforce_cat import EnforceCatConsequences
from meoudoku.heuristics.force_necessary import ForceNecessary
from meoudoku.heuristics.row_col_sets import ColsConfinedToRows, RowsConfinedToCols
from meoudoku.heuristics.single_candidate import SingleCandidate


def default_stack(*, include_backtrack: bool = False, max_k: int = 5) -> list[Heuristic]:
    stack: list[Heuristic] = [
        EnforceCatConsequences(),
        SingleCandidate(),
        ColorsConfinedToRows(max_k=max_k),
        RowsContainOnlyColors(max_k=max_k),
        ColorsConfinedToCols(max_k=max_k),
        ColsContainOnlyColors(max_k=max_k),
        RowsConfinedToCols(max_k=max_k),
        ColsConfinedToRows(max_k=max_k),
        EliminateImpossible(max_k=max_k),
        ForceNecessary(max_k=max_k),
    ]
    # Backtracking off by default (phone play); pass include_backtrack=True to enable.
    if include_backtrack:
        stack.append(BacktrackHeuristic())
    return stack


__all__ = [
    "Heuristic",
    "default_stack",
    "EnforceCatConsequences",
    "SingleCandidate",
    "ColorsConfinedToRows",
    "RowsContainOnlyColors",
    "ColorsConfinedToCols",
    "ColsContainOnlyColors",
    "RowsConfinedToCols",
    "ColsConfinedToRows",
    "EliminateImpossible",
    "ForceNecessary",
    "BacktrackHeuristic",
]
