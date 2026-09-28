"""Board model, Delta, and text file parse/dump."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


class CellState(Enum):
    EMPTY = auto()
    CAT = auto()
    X = auto()


@dataclass(frozen=True)
class Delta:
    cats: frozenset[tuple[int, int]] = frozenset()
    xs: frozenset[tuple[int, int]] = frozenset()

    def __bool__(self) -> bool:
        return bool(self.cats or self.xs)

    @staticmethod
    def of(
        *,
        cats: set[tuple[int, int]] | frozenset[tuple[int, int]] | None = None,
        xs: set[tuple[int, int]] | frozenset[tuple[int, int]] | None = None,
    ) -> Delta:
        return Delta(
            cats=frozenset(cats or ()),
            xs=frozenset(xs or ()),
        )


class BoardError(ValueError):
    """Invalid board operation or parse error."""


class Board:
    def __init__(self, colors: list[list[int]], state: list[list[CellState]] | None = None):
        if not colors or not colors[0]:
            raise BoardError("Board must be non-empty")
        self.n = len(colors)
        if any(len(row) != self.n for row in colors):
            raise BoardError("Board must be square NxN")
        color_ids = {c for row in colors for c in row}
        if color_ids != set(range(self.n)):
            raise BoardError(f"Expected color ids 0..{self.n - 1}, got {sorted(color_ids)}")
        self.colors = [list(row) for row in colors]
        if state is None:
            self.state = [[CellState.EMPTY for _ in range(self.n)] for _ in range(self.n)]
        else:
            if len(state) != self.n or any(len(row) != self.n for row in state):
                raise BoardError("STATE must match COLORS size")
            self.state = [list(row) for row in state]

    def copy(self) -> Board:
        return Board(deepcopy(self.colors), deepcopy(self.state))

    def snapshot(self) -> list[list[CellState]]:
        return deepcopy(self.state)

    def restore(self, state: list[list[CellState]]) -> None:
        self.state = deepcopy(state)

    def get(self, r: int, c: int) -> CellState:
        return self.state[r][c]

    def color(self, r: int, c: int) -> int:
        return self.colors[r][c]

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.n and 0 <= c < self.n

    def cat_positions(self) -> list[tuple[int, int]]:
        return [(r, c) for r in range(self.n) for c in range(self.n) if self.state[r][c] is CellState.CAT]

    def is_solved(self) -> bool:
        cats = self.cat_positions()
        if len(cats) != self.n:
            return False
        rows = {r for r, _ in cats}
        cols = {c for _, c in cats}
        colors = {self.colors[r][c] for r, c in cats}
        if rows != set(range(self.n)) or cols != set(range(self.n)) or colors != set(range(self.n)):
            return False
        for r, c in cats:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if self.in_bounds(rr, cc) and self.state[rr][cc] is CellState.CAT:
                        return False
        return True

    def is_candidate(self, r: int, c: int) -> bool:
        return self.state[r][c] is CellState.EMPTY

    def candidates(self) -> list[tuple[int, int]]:
        return [(r, c) for r in range(self.n) for c in range(self.n) if self.is_candidate(r, c)]

    def candidates_in_row(self, r: int) -> list[tuple[int, int]]:
        return [(r, c) for c in range(self.n) if self.is_candidate(r, c)]

    def candidates_in_col(self, c: int) -> list[tuple[int, int]]:
        return [(r, c) for r in range(self.n) if self.is_candidate(r, c)]

    def candidates_in_color(self, color_id: int) -> list[tuple[int, int]]:
        return [
            (r, c)
            for r in range(self.n)
            for c in range(self.n)
            if self.colors[r][c] == color_id and self.is_candidate(r, c)
        ]

    def row_has_cat(self, r: int) -> bool:
        return any(self.state[r][c] is CellState.CAT for c in range(self.n))

    def col_has_cat(self, c: int) -> bool:
        return any(self.state[r][c] is CellState.CAT for r in range(self.n))

    def color_has_cat(self, color_id: int) -> bool:
        return any(
            self.state[r][c] is CellState.CAT and self.colors[r][c] == color_id
            for r in range(self.n)
            for c in range(self.n)
        )

    def apply(self, delta: Delta) -> None:
        """Apply delta. Raises BoardError on conflict."""
        for r, c in delta.cats:
            if not self.in_bounds(r, c):
                raise BoardError(f"Cat out of bounds: {(r, c)}")
            cur = self.state[r][c]
            if cur is CellState.X:
                raise BoardError(f"Cannot place cat on X at {(r, c)}")
            if cur is CellState.CAT:
                continue
            self.state[r][c] = CellState.CAT
        for r, c in delta.xs:
            if not self.in_bounds(r, c):
                raise BoardError(f"X out of bounds: {(r, c)}")
            cur = self.state[r][c]
            if cur is CellState.CAT:
                raise BoardError(f"Cannot place X on cat at {(r, c)}")
            if cur is CellState.X:
                continue
            self.state[r][c] = CellState.X
        self._assert_no_immediate_conflict()

    def _assert_no_immediate_conflict(self) -> None:
        cats = self.cat_positions()
        rows = [r for r, _ in cats]
        cols = [c for _, c in cats]
        colors = [self.colors[r][c] for r, c in cats]
        if len(rows) != len(set(rows)) or len(cols) != len(set(cols)) or len(colors) != len(set(colors)):
            raise BoardError("Duplicate cat in row/col/color")
        for r, c in cats:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if self.in_bounds(rr, cc) and self.state[rr][cc] is CellState.CAT:
                        raise BoardError(f"Cats touch at {(r, c)} and {(rr, cc)}")

    def has_contradiction(self) -> bool:
        """True if an open unit has zero candidates."""
        try:
            self._assert_no_immediate_conflict()
        except BoardError:
            return True
        for r in range(self.n):
            if not self.row_has_cat(r) and not self.candidates_in_row(r):
                return True
        for c in range(self.n):
            if not self.col_has_cat(c) and not self.candidates_in_col(c):
                return True
        for color_id in range(self.n):
            if not self.color_has_cat(color_id) and not self.candidates_in_color(color_id):
                return True
        return False

    def dump(self) -> str:
        id_to_letter = {i: chr(ord("a") + i) for i in range(self.n)}
        lines = ["COLORS"]
        for r in range(self.n):
            lines.append(" ".join(id_to_letter[self.colors[r][c]] for c in range(self.n)))
        lines.append("STATE")
        for r in range(self.n):
            cells = []
            for c in range(self.n):
                st = self.state[r][c]
                if st is CellState.CAT:
                    cells.append("C")
                elif st is CellState.X:
                    cells.append("X")
                else:
                    cells.append(".")
            lines.append(" ".join(cells))
        return "\n".join(lines) + "\n"

    def __str__(self) -> str:
        return self.dump()


def _parse_color_token(token: str, letter_to_id: dict[str, int]) -> int:
    if len(token) != 1 or not token.isalpha():
        raise BoardError(f"Invalid color token: {token!r}")
    key = token.lower()
    if key not in letter_to_id:
        letter_to_id[key] = len(letter_to_id)
    return letter_to_id[key]


def _parse_state_token(token: str) -> CellState:
    if token == ".":
        return CellState.EMPTY
    if token in ("C", "c"):
        return CellState.CAT
    if token in ("X", "x", "#"):
        return CellState.X
    raise BoardError(f"Invalid state token: {token!r}")


def parse_board(text: str) -> Board:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise BoardError("Empty board file")

    colors_rows: list[list[int]] = []
    state_rows: list[list[CellState]] = []
    letter_to_id: dict[str, int] = {}
    mode: str | None = None

    for ln in lines:
        upper = ln.upper()
        if upper == "COLORS":
            mode = "colors"
            continue
        if upper == "STATE":
            mode = "state"
            continue
        tokens = ln.split()
        if mode == "colors" or (mode is None and all(len(t) == 1 and t.isalpha() for t in tokens)):
            mode = "colors"
            colors_rows.append([_parse_color_token(t, letter_to_id) for t in tokens])
        elif mode == "state":
            state_rows.append([_parse_state_token(t) for t in tokens])
        else:
            raise BoardError(f"Unexpected line: {ln!r}")

    if not colors_rows:
        raise BoardError("COLORS section missing")

    # Remap letter ids to 0..N-1 sorted by first appearance already;
    # validate count equals N after Board.__init__.
    n = len(colors_rows)
    if len(letter_to_id) != n:
        # Compact remap if letters are a subset naming — require exactly N distinct.
        raise BoardError(f"Expected {n} distinct colors, got {len(letter_to_id)}")

    state = state_rows if state_rows else None
    if state is not None and len(state) != n:
        raise BoardError("STATE row count mismatch")
    return Board(colors_rows, state)


def load_board(path: str | Path) -> Board:
    return parse_board(Path(path).read_text(encoding="utf-8"))
