"""Heuristic-stack solver engine with rewind-to-start and backtracking."""

from __future__ import annotations

from dataclasses import dataclass, field

from meoudoku.board import Board, BoardError, CellState, Delta
from meoudoku.heuristics import Heuristic, default_stack
from meoudoku.heuristics.backtrack import pick_mrv_cells


class UnsolvedError(RuntimeError):
    """Board could not be solved."""


@dataclass
class _Frame:
    """In-memory branch: try each MRV cell as CAT; failed siblings become X."""

    snapshot: list[list[CellState]]
    cells: list[tuple[int, int]]
    index: int = 0


@dataclass
class SolveResult:
    board: Board
    steps: int = 0
    guesses: int = 0
    applied: list[tuple[str, Delta]] = field(default_factory=list)


class Solver:
    def __init__(self, stack: list[Heuristic] | None = None) -> None:
        self.stack = stack if stack is not None else default_stack()

    def step(self, board: Board) -> tuple[str, Delta] | None:
        """
        Run stack from the start; return (heuristic_name, delta) of first hit.
        Does not mutate board.

        For backtrack, delta.cats is the full MRV candidate set (not a single
        placement). Callers that tap a device must refuse this heuristic.
        """
        for h in self.stack:
            delta = h.apply(board)
            if delta:
                return h.name, delta
        return None

    def solve(self, board: Board, *, max_steps: int = 100_000) -> SolveResult:
        work = board.copy()
        trail: list[_Frame] = []
        result = SolveResult(board=work)

        while result.steps < max_steps:
            if work.is_solved():
                return result
            if work.has_contradiction():
                if not self._retreat(work, trail, result):
                    raise UnsolvedError("Contradiction and no branches left")
                continue

            hit = self.step(work)
            if hit is None:
                if not self._retreat(work, trail, result):
                    raise UnsolvedError("No heuristic progressed")
                continue

            name, delta = hit
            if name == "backtrack":
                cells = pick_mrv_cells(work)
                if not cells:
                    if not self._retreat(work, trail, result):
                        raise UnsolvedError("Backtrack with empty candidate list")
                    continue
                # Branch in memory only — never apply the multi-cat delta.
                trail.append(_Frame(snapshot=work.snapshot(), cells=list(cells), index=0))
                result.guesses += 1
                delta = Delta.of(cats={cells[0]})

            try:
                work.apply(delta)
            except BoardError:
                if not self._retreat(work, trail, result):
                    raise UnsolvedError("Invalid apply and no branches left") from None
                continue

            result.applied.append((name, delta))
            result.steps += 1
            # Successful apply → caller loop restarts stack from index 0 next iteration.

        raise UnsolvedError(f"Exceeded max_steps={max_steps}")

    def _retreat(self, board: Board, trail: list[_Frame], result: SolveResult) -> bool:
        while trail:
            frame = trail[-1]
            while True:
                frame.index += 1
                if frame.index >= len(frame.cells):
                    trail.pop()
                    break
                board.restore(frame.snapshot)
                failed = frame.cells[: frame.index]
                cell = frame.cells[frame.index]
                try:
                    board.apply(Delta.of(xs=set(failed), cats={cell}))
                except BoardError:
                    continue
                result.applied.append(
                    ("backtrack", Delta.of(xs=set(failed), cats={cell})),
                )
                result.steps += 1
                result.guesses += 1
                return True
        return False


def solve(board: Board) -> Board:
    return Solver().solve(board).board
