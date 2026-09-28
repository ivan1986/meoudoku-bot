"""Device control contract (screenshot + tap)."""

from __future__ import annotations

from typing import Protocol


class Device(Protocol):
    def screenshot(self) -> bytes:
        """Capture current screen PNG/JPEG bytes."""

    def tap(self, x: int, y: int) -> None:
        """Tap screen coordinates in pixels."""
