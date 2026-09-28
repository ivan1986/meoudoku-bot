"""Vision board parser contract."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from meoudoku.board import Board


class BoardParser(Protocol):
    def parse(self, image: bytes | Path | str) -> Board:
        """Parse a screenshot (bytes or path) into a Board."""
