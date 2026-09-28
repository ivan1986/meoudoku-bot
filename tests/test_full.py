from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from meoudoku.board import Board
from meoudoku.full import FullError, full_loop
from meoudoku.play import PlayResult


def _png(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    assert ok
    return buf.tobytes()


def _board(n: int = 8) -> Board:
    colors = [[(r + c) % n for c in range(n)] for r in range(n)]
    return Board(colors)


def test_full_loop_happy_path_order():
    orange = np.zeros((100, 100, 3), dtype=np.uint8)
    # Bright orange BGR ≈ (25, 149, 251)
    orange[:] = (240, 240, 245)
    orange[70:85, 30:70] = (25, 149, 251)

    light = np.full((100, 100, 3), 200, dtype=np.uint8)
    dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    shots = {
        "phase": "menu",
        "calls": [],
    }

    device = MagicMock()

    def screenshot():
        shots["calls"].append("screenshot")
        if shots["phase"] == "menu":
            return _png(orange)
        if shots["phase"] == "post_tap":
            return _png(light)
        return _png(dark_frame)

    device.screenshot.side_effect = screenshot
    device.wm_size.return_value = (100, 100)
    device.force_stop.return_value = None
    device.launch_app.return_value = None
    device.tap.return_value = None

    play_result = PlayResult(board=_board(8), n=8, rounds=3, steps=[])

    def fake_play_loop(**kwargs):
        shots["calls"].append("play")
        shots["phase"] = "dark"
        assert kwargs.get("device") is device
        return play_result

    events: list[str] = []

    with (
        patch("meoudoku.full.AdbDevice", return_value=device),
        patch("meoudoku.full.time.sleep", side_effect=lambda s: shots["calls"].append(f"sleep:{s}")),
        patch("meoudoku.full.play_loop", side_effect=fake_play_loop),
    ):
        def tap_side_effect(x, y, **kwargs):
            shots["calls"].append(f"tap:{x},{y}")
            shots["phase"] = "post_tap"

        device.tap.side_effect = tap_side_effect

        result = full_loop(
            banner_delay_s=5.0,
            menu_timeout_s=2.0,
            dark_timeout_s=2.0,
            poll_s=0.0,
            on_event=events.append,
        )

    assert result.n == 8
    assert result.rounds == 3
    assert device.force_stop.call_count >= 2
    device.launch_app.assert_called_once()
    assert any(c.startswith("tap:") for c in shots["calls"])
    assert "sleep:5.0" in shots["calls"]
    assert shots["calls"].index("sleep:5.0") < shots["calls"].index("play")
    assert device.force_stop.call_args_list[0].args[0] == "com.oakever.meowdoku"
    assert device.force_stop.call_args_list[-1].args[0] == "com.oakever.meowdoku"


def test_full_loop_force_stop_on_error():
    from meoudoku.device.adb import AdbError

    device = MagicMock()
    device.wm_size.return_value = (100, 100)
    device.force_stop.return_value = None
    device.launch_app.side_effect = AdbError("boom")

    with (
        patch("meoudoku.full.AdbDevice", return_value=device),
        patch("meoudoku.full.time.sleep"),
        pytest.raises(FullError, match="launch failed"),
    ):
        full_loop(menu_timeout_s=0.1, poll_s=0.0)
    assert device.force_stop.call_count >= 2
