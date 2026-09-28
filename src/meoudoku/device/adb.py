"""ADB-backed device: screenshot and tap."""

from __future__ import annotations

import re
import shutil
import subprocess
import time


class AdbError(RuntimeError):
    """ADB command failed or device unavailable."""


class AdbDevice:
    def __init__(self, serial: str | None = None, adb_bin: str = "adb") -> None:
        self.serial = serial
        self.adb_bin = adb_bin
        if shutil.which(adb_bin) is None:
            raise AdbError(f"{adb_bin!r} not found in PATH")

    def _base_cmd(self) -> list[str]:
        cmd = [self.adb_bin]
        if self.serial:
            cmd.extend(["-s", self.serial])
        return cmd

    def _run(self, args: list[str], *, binary: bool = False) -> bytes | str:
        cmd = self._base_cmd() + args
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
            )
        except OSError as exc:
            raise AdbError(f"Failed to run {' '.join(cmd)}: {exc}") from exc
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise AdbError(f"ADB failed ({proc.returncode}): {err or ' '.join(cmd)}")
        if binary:
            return proc.stdout
        return proc.stdout.decode("utf-8", errors="replace")

    def screenshot(self) -> bytes:
        data = self._run(["exec-out", "screencap", "-p"], binary=True)
        assert isinstance(data, bytes)
        if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
            # Some devices insert CRLF; try fixing common adb mangling
            fixed = data.replace(b"\r\n", b"\n")
            if fixed[:8] == b"\x89PNG\r\n\x1a\n" or fixed[:4] == b"\x89PNG":
                data = fixed if fixed[:8] == b"\x89PNG\r\n\x1a\n" else data
            elif not data.startswith(b"\x89PNG"):
                raise AdbError("screencap did not return a PNG image")
        return data

    def tap(self, x: int, y: int, *, times: int = 1, gap_s: float = 0.07) -> None:
        """Tap screen coordinates; times=2 for cat, times=1 for cross."""
        for i in range(max(1, times)):
            if i:
                time.sleep(gap_s)
            self._run(["shell", "input", "tap", str(int(x)), str(int(y))])

    def launch_app(self, package: str, activity: str) -> None:
        """Start an activity: am start -n package/activity."""
        component = f"{package}/{activity}"
        self._run(["shell", "am", "start", "-n", component])

    def force_stop(self, package: str) -> None:
        """Force-stop a package: am force-stop."""
        self._run(["shell", "am", "force-stop", package])

    def wm_size(self) -> tuple[int, int]:
        """Return physical display size (width, height) from `wm size`."""
        out = self._run(["shell", "wm", "size"])
        assert isinstance(out, str)
        # "Physical size: 1080x2400" or "Override size: ..."
        width = height = None
        for line in out.splitlines():
            line = line.strip()
            if ":" not in line:
                continue
            _, _, rhs = line.partition(":")
            rhs = rhs.strip().lower().replace(" ", "")
            if "x" not in rhs:
                continue
            a, _, b = rhs.partition("x")
            try:
                w, h = int(a), int(b)
            except ValueError:
                continue
            if "override" in line.lower():
                return w, h
            width, height = w, h
        if width is None or height is None:
            raise AdbError(f"Could not parse wm size output: {out!r}")
        return width, height

    def foreground_package(self) -> str | None:
        """Best-effort package name of the focused/resumed activity."""
        # Prefer dumpsys window focus; fall back to resumed activity line.
        for args in (
            ["shell", "dumpsys", "window", "displays"],
            ["shell", "dumpsys", "window"],
            ["shell", "dumpsys", "activity", "activities"],
        ):
            try:
                out = self._run(args)
            except AdbError:
                continue
            assert isinstance(out, str)
            pkg = _parse_foreground_package(out)
            if pkg:
                return pkg
        return None


def _parse_foreground_package(dumpsys: str) -> str | None:
    """Extract package from dumpsys focus / resumed activity text."""
    patterns = (
        r"mCurrentFocus=Window\{[^ ]+ [^ ]+ ([a-zA-Z0-9._]+)/",
        r"mFocusedApp=.*? ([a-zA-Z0-9._]+)/",
        r"topResumedActivity=ActivityRecord\{[^ ]+ [^ ]+ ([a-zA-Z0-9._]+)/",
        r"mResumedActivity: ActivityRecord\{[^ ]+ [^ ]+ ([a-zA-Z0-9._]+)/",
    )
    for pat in patterns:
        m = re.search(pat, dumpsys)
        if m:
            return m.group(1)
    return None
