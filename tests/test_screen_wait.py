from pathlib import Path

import cv2
import numpy as np
import pytest

from meoudoku.vision.screen_wait import (
    ScreenWaitError,
    decode_bgr,
    frac_to_xy,
    is_dark,
    is_dark_bg,
    is_orange,
    is_orange_at,
    is_post_win_dark,
    wait_until,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "screens"


def test_frac_to_xy_corners():
    assert frac_to_xy(100, 200, 0.0, 0.0) == (0, 0)
    assert frac_to_xy(100, 200, 1.0, 1.0) == (99, 199)
    assert frac_to_xy(101, 101, 0.5, 0.5) == (50, 50)


def test_is_orange_hsv_thresholds():
    assert is_orange((16.0, 230.0, 251.0))
    assert not is_orange((9.0, 10.0, 249.0))  # pale pink menu bg
    assert not is_orange((100.0, 200.0, 200.0))  # green-ish


def test_menu_level_button_is_orange():
    path = FIXTURES / "menu_level.png"
    if not path.is_file():
        pytest.skip("menu_level.png fixture missing")
    bgr = cv2.imread(str(path))
    assert bgr is not None
    h, w = bgr.shape[:2]
    x, y = frac_to_xy(w, h, 0.5, 0.72)
    assert is_orange_at(bgr, x, y)
    # Pale area above the button should not be orange.
    x2, y2 = frac_to_xy(w, h, 0.5, 0.55)
    assert not is_orange_at(bgr, x2, y2)


def test_is_dark_bg_synthetic():
    dark = np.full((200, 100, 3), (0x1D, 0x1D, 0x2B), dtype=np.uint8)
    mid = np.full((200, 100, 3), (70, 70, 80), dtype=np.uint8)  # ~luma 72, banner-like
    light = np.full((200, 100, 3), (200, 210, 220), dtype=np.uint8)
    assert is_dark_bg(dark, fx=0.5, fy=0.48)
    assert is_dark_bg(mid, fx=0.5, fy=0.48)
    assert not is_dark_bg(light, fx=0.5, fy=0.48)


def test_board_center_not_dark_bg():
    path = FIXTURES / "board_empty_10.png"
    if not path.is_file():
        pytest.skip("board_empty_10.png fixture missing")
    bgr = cv2.imread(str(path))
    assert bgr is not None
    assert not is_dark_bg(bgr, fx=0.5, fy=0.48)


def test_banner_fixture_is_dark_bg():
    path = FIXTURES / "banner_start.png"
    if not path.is_file():
        pytest.skip("banner_start.png fixture missing")
    bgr = cv2.imread(str(path))
    assert bgr is not None
    # Crop is mostly the dark overlay; treat as banner-present.
    assert is_dark_bg(bgr, fx=0.5, fy=0.48)


def test_is_dark_full_frame():
    black = np.zeros((40, 40, 3), dtype=np.uint8)
    white = np.full((40, 40, 3), 255, dtype=np.uint8)
    assert is_dark(black, max_mean=40)
    assert not is_dark(white, max_mean=40)


def test_post_win_dark_on_leaderboard_fixture():
    path = FIXTURES / "leaderboard_postwin.png"
    if not path.is_file():
        pytest.skip("leaderboard_postwin.png fixture missing")
    bgr = cv2.imread(str(path))
    assert bgr is not None
    assert is_post_win_dark(bgr)
    # Full-frame mean stays bright because of the podium UI.
    assert not is_dark(bgr, max_mean=40)


def test_post_win_dark_false_on_board():
    path = FIXTURES / "board_empty_10.png"
    if not path.is_file():
        pytest.skip("board_empty_10.png fixture missing")
    bgr = cv2.imread(str(path))
    assert bgr is not None
    assert not is_post_win_dark(bgr)


def test_wait_until_succeeds_and_times_out():
    frames = [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.full((10, 10, 3), 255, dtype=np.uint8),
    ]

    class FakeDevice:
        def __init__(self):
            self.i = 0

        def screenshot(self):
            img = frames[min(self.i, len(frames) - 1)]
            self.i += 1
            ok, buf = cv2.imencode(".png", img)
            assert ok
            return buf.tobytes()

    device = FakeDevice()
    out = wait_until(
        device,
        lambda bgr: float(bgr.mean()) > 100,
        timeout_s=2.0,
        poll_s=0.0,
    )
    assert float(out.mean()) > 100

    device2 = FakeDevice()
    with pytest.raises(ScreenWaitError):
        wait_until(
            device2,
            lambda bgr: False,
            timeout_s=0.05,
            poll_s=0.01,
        )


def test_decode_bgr_roundtrip():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[:] = (10, 20, 30)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    decoded = decode_bgr(buf.tobytes())
    assert decoded.shape == (8, 8, 3)
