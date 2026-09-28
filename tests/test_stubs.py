from pathlib import Path

import pytest

from meoudoku.device import StubDevice
from meoudoku.vision import StubBoardParser

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_stub_parser_reads_txt():
    board = StubBoardParser().parse(FIXTURES / "level_tiny.txt")
    assert board.n == 4


def test_stub_parser_rejects_image():
    with pytest.raises(NotImplementedError):
        StubBoardParser().parse(FIXTURES / "nope.png")


def test_stub_device_logs_taps(tmp_path: Path):
    shot = tmp_path / "screen.bin"
    shot.write_bytes(b"fake")
    dev = StubDevice(screenshot_path=shot)
    assert dev.screenshot() == b"fake"
    dev.tap(10, 20)
    assert dev.taps == [(10, 20)]
