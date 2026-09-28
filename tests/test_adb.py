from unittest.mock import MagicMock, patch

import pytest

from meoudoku.device.adb import AdbDevice, AdbError


def test_adb_missing_binary():
    with patch("meoudoku.device.adb.shutil.which", return_value=None):
        with pytest.raises(AdbError, match="not found"):
            AdbDevice()


def test_screenshot_ok():
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
    with patch("meoudoku.device.adb.shutil.which", return_value="/usr/bin/adb"):
        device = AdbDevice(serial="emu")
    fake = MagicMock()
    fake.returncode = 0
    fake.stdout = png
    fake.stderr = b""
    with patch("meoudoku.device.adb.subprocess.run", return_value=fake) as run:
        assert device.screenshot() == png
        cmd = run.call_args.args[0]
        assert cmd[:4] == ["adb", "-s", "emu", "exec-out"]


def test_screenshot_rejects_non_png():
    with patch("meoudoku.device.adb.shutil.which", return_value="/usr/bin/adb"):
        device = AdbDevice()
    fake = MagicMock()
    fake.returncode = 0
    fake.stdout = b"not-an-image"
    fake.stderr = b""
    with patch("meoudoku.device.adb.subprocess.run", return_value=fake):
        with pytest.raises(AdbError, match="PNG"):
            device.screenshot()


def test_tap_runs_shell_input():
    with patch("meoudoku.device.adb.shutil.which", return_value="/usr/bin/adb"):
        device = AdbDevice()
    fake = MagicMock()
    fake.returncode = 0
    fake.stdout = b""
    fake.stderr = b""
    with patch("meoudoku.device.adb.subprocess.run", return_value=fake) as run:
        device.tap(10, 20)
        assert run.call_args.args[0][-4:] == ["input", "tap", "10", "20"]


def test_double_tap():
    with patch("meoudoku.device.adb.shutil.which", return_value="/usr/bin/adb"):
        device = AdbDevice()
    fake = MagicMock()
    fake.returncode = 0
    fake.stdout = b""
    fake.stderr = b""
    with (
        patch("meoudoku.device.adb.subprocess.run", return_value=fake) as run,
        patch("meoudoku.device.adb.time.sleep") as sleep,
    ):
        device.tap(5, 6, times=2, gap_s=0.01)
        assert run.call_count == 2
        sleep.assert_called_once_with(0.01)
        assert run.call_args_list[0].args[0][-4:] == ["input", "tap", "5", "6"]
        assert run.call_args_list[1].args[0][-4:] == ["input", "tap", "5", "6"]


def test_launch_and_force_stop():
    with patch("meoudoku.device.adb.shutil.which", return_value="/usr/bin/adb"):
        device = AdbDevice()
    fake = MagicMock()
    fake.returncode = 0
    fake.stdout = b""
    fake.stderr = b""
    with patch("meoudoku.device.adb.subprocess.run", return_value=fake) as run:
        device.launch_app("com.oakever.meowdoku", "com.godot.game.GodotAppLauncher")
        assert run.call_args.args[0][-4:] == [
            "am",
            "start",
            "-n",
            "com.oakever.meowdoku/com.godot.game.GodotAppLauncher",
        ]
        device.force_stop("com.oakever.meowdoku")
        assert run.call_args.args[0][-3:] == ["am", "force-stop", "com.oakever.meowdoku"]


def test_wm_size_parses_physical():
    with patch("meoudoku.device.adb.shutil.which", return_value="/usr/bin/adb"):
        device = AdbDevice()
    fake = MagicMock()
    fake.returncode = 0
    fake.stdout = b"Physical size: 1080x2400\n"
    fake.stderr = b""
    with patch("meoudoku.device.adb.subprocess.run", return_value=fake):
        assert device.wm_size() == (1080, 2400)


def test_wm_size_prefers_override():
    with patch("meoudoku.device.adb.shutil.which", return_value="/usr/bin/adb"):
        device = AdbDevice()
    fake = MagicMock()
    fake.returncode = 0
    fake.stdout = b"Physical size: 1080x2400\nOverride size: 720x1600\n"
    fake.stderr = b""
    with patch("meoudoku.device.adb.subprocess.run", return_value=fake):
        assert device.wm_size() == (720, 1600)


def test_foreground_package_from_focus():
    from meoudoku.device.adb import _parse_foreground_package

    text = (
        "  mCurrentFocus=Window{abc123 u0 com.oakever.meowdoku/com.godot.game.GodotAppLauncher}\n"
    )
    assert _parse_foreground_package(text) == "com.oakever.meowdoku"
