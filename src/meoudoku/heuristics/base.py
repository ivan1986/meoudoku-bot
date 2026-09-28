"""Heuristic protocol."""

from __future__ import annotations

from typing import Protocol

from meoudoku.board import Board, Delta


class Heuristic(Protocol):
    name: str

    def apply(self, board: Board) -> Delta | None:
        """Return placements if this rule fires; None if no progress."""
