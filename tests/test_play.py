from unittest.mock import MagicMock

import pytest

from meoudoku.board import Board, CellState
from meoudoku.play import PlayError, read_board, read_board_stable
from meoudoku.vision.cv_parser import ParseResult
from meoudoku.vision.roi import Roi
from meoudoku.vision.cv_parser import Grid


def _board(n: int) -> Board:
    colors = [[(r + c) % n for c in range(n)] for r in range(n)]
    return Board(colors)


def _parsed(n: int) -> ParseResult:
    edges = tuple(i * 10 for i in range(n + 1))
    return ParseResult(board=_board(n), roi=Roi(0, 0, n * 10, n * 10), grid=Grid(n, edges, edges))


def test_read_board_rejects_n_mismatch():
    device = MagicMock()
    device.screenshot.return_value = b"fake"

    calls = {"i": 0}

    def fake_parse_ex(_png):
        # Always return N=9 while expected is 10.
        calls["i"] += 1
        return _parsed(9)

    with pytest.raises(PlayError, match="size mismatch"):
        # Patch parser used inside read_board
        import meoudoku.play as play_mod

        original = play_mod.CvBoardParser

        class FakeParser:
            def __init__(self, *args, **kwargs):
                pass

            def parse_ex(self, png):
                return fake_parse_ex(png)

        play_mod.CvBoardParser = FakeParser
        try:
            read_board(device, expected_n=10, retries=2, retry_delay_s=0.0)
        finally:
            play_mod.CvBoardParser = original

    assert calls["i"] == 2


def test_read_board_accepts_matching_n():
    device = MagicMock()
    device.screenshot.return_value = b"fake"
    import meoudoku.play as play_mod

    class FakeParser:
        def __init__(self, *args, **kwargs):
            pass

        def parse_ex(self, png):
            return _parsed(10)

    original = play_mod.CvBoardParser
    play_mod.CvBoardParser = FakeParser
    try:
        capture = read_board(device, expected_n=10, retries=1, retry_delay_s=0.0)
    finally:
        play_mod.CvBoardParser = original
    assert capture.parsed.board.n == 10
    assert capture.image == b"fake"
    assert capture.parsed.board.get(0, 0) is CellState.EMPTY


def test_read_board_stable_requires_matching_pair():
    device = MagicMock()
    device.screenshot.return_value = b"fake"
    import meoudoku.play as play_mod

    ns = [11, 10, 10]  # first read mismatches expected N; then a stable pair

    class FakeParser:
        def __init__(self, *args, **kwargs):
            pass

        def parse_ex(self, png):
            return _parsed(ns.pop(0))

    original = play_mod.CvBoardParser
    play_mod.CvBoardParser = FakeParser
    try:
        capture = read_board_stable(
            device,
            expected_n=10,
            n_force=None,
            retries=3,
            pause_s=0.0,
            retry_delay_s=0.0,
        )
    finally:
        play_mod.CvBoardParser = original
    assert capture.parsed.board.n == 10
    assert capture.image == b"fake"
    assert ns == []
