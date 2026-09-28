from meoudoku.board import parse_board
from meoudoku.heuristics import (
    ColorsConfinedToRows,
    EliminateImpossible,
    EnforceCatConsequences,
    ForceNecessary,
    RowsConfinedToCols,
    RowsContainOnlyColors,
    SingleCandidate,
)


def test_enforce_cat_marks_row_col_color_and_neighbors():
    board = parse_board(
        """\
COLORS
a a b c
a d b c
a d b c
a d d c
STATE
. C . .
. . . .
. . . .
. . . .
"""
    )
    delta = EnforceCatConsequences().apply(board)
    assert delta is not None
    assert (0, 0) in delta.xs
    assert (0, 2) in delta.xs
    assert (1, 1) in delta.xs
    assert (2, 1) in delta.xs
    assert not delta.cats


def test_single_candidate_row():
    board = parse_board(
        """\
COLORS
a b c d
a b c d
a b c d
a b c d
STATE
X . X X
. . . .
. . . .
. . . .
"""
    )
    delta = SingleCandidate().apply(board)
    assert delta is not None
    assert delta.cats == frozenset({(0, 1)})


def test_colors_confined_to_one_row():
    # Color a lives only in row 0 → X other colors in row 0
    board = parse_board(
        """\
COLORS
a a a b
c c d b
c c d b
c d d b
STATE
. . . .
. . . .
. . . .
. . . .
"""
    )
    delta = ColorsConfinedToRows(max_k=1).apply(board)
    assert delta is not None
    assert (0, 3) in delta.xs


def test_row_contains_only_one_color():
    # Row 3 is all color d → X color d outside row 3
    board = parse_board(
        """\
COLORS
a a b c
a d b c
a b b c
d d d d
STATE
. . . .
. . . .
. . . .
. . . .
"""
    )
    delta = RowsContainOnlyColors(max_k=1).apply(board)
    assert delta is not None
    assert (1, 1) in delta.xs


def test_rows_confined_to_cols():
    # Rows 0..1 only have candidates in cols 0..1 → X (2,0) and (2,1)
    board = parse_board(
        """\
COLORS
a a b c
a a b c
d d b c
d d d d
STATE
. . X X
. . X X
. . . .
. . . .
"""
    )
    delta = RowsConfinedToCols(max_k=2).apply(board)
    assert delta is not None
    assert (2, 0) in delta.xs
    assert (2, 1) in delta.xs


def test_eliminate_impossible_marks_dead_cell():
    # Mid-solve of level_tiny: trial CAT at (0,0) contradicts; (0,2) is safe.
    board = parse_board(
        """\
COLORS
a a b c
a a b c
d d b c
d d d d
STATE
. . . .
. . . .
X X . .
. . X X
"""
    )
    assert SingleCandidate().apply(board) is None
    delta = EliminateImpossible().apply(board)
    assert delta is not None
    assert (0, 0) in delta.xs
    assert (0, 2) not in delta.xs


def test_force_necessary_places_only_safe_cat():
    # After eliminating dead cells in row 0, only (0,2) remains → forced CAT.
    board = parse_board(
        """\
COLORS
a a b c
a a b c
d d b c
d d d d
STATE
X X . X
. . . .
X X . .
. . X X
"""
    )
    delta = ForceNecessary().apply(board)
    assert delta is not None
    assert delta.cats == frozenset({(0, 2)})
