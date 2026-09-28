"""Stub device — logs taps, serves a fixture file as screenshot."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class StubDevice:
    def __init__(self, screenshot_path: Path | str | None = None) -> None:
        self.screenshot_path = Path(screenshot_path) if screenshot_path else None
        self.taps: list[tuple[int, int]] = []

    def screenshot(self) -> bytes:
        if self.screenshot_path is None:
            raise NotImplementedError("StubDevice has no screenshot_path configured")
        return self.screenshot_path.read_bytes()

    def tap(self, x: int, y: int) -> None:
        self.taps.append((x, y))
        logger.info("StubDevice tap (%s, %s)", x, y)
