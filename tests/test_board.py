from meoudoku.board import CellState, Delta, parse_board


def test_parse_and_dump_roundtrip():
    text = """\
COLORS
a a b b
a c c b
d c c b
d d d d
STATE
. C . .
. . . X
. . . .
. . . .
"""
    board = parse_board(text)
    assert board.n == 4
    assert board.get(0, 1) is CellState.CAT
    assert board.get(1, 3) is CellState.X
    again = parse_board(board.dump())
    assert again.colors == board.colors
    assert again.state == board.state


def test_apply_delta():
    board = parse_board(
        """\
COLORS
a b c d
a b c d
a b c d
a b c d
STATE
. . . .
. . . .
. . . .
. . . .
"""
    )
    board.apply(Delta.of(cats={(0, 0)}, xs={(0, 1), (1, 0)}))
    assert board.get(0, 0) is CellState.CAT
    assert board.get(0, 1) is CellState.X
