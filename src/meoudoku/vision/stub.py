"""Stub board parser — loads text fixtures, not real screenshots."""

from __future__ import annotations

from pathlib import Path

from meoudoku.board import Board, load_board


class StubBoardParser:
    """
    Development stub.

    - If `image` points to a .txt board file, parse it.
    - Otherwise raise NotImplementedError (real CV not in v1).
    """

    def __init__(self, fixture_path: Path | str | None = None) -> None:
        self.fixture_path = Path(fixture_path) if fixture_path else None

    def parse(self, image: bytes | Path | str) -> Board:
        if self.fixture_path is not None and image == b"":
            return load_board(self.fixture_path)

        if isinstance(image, (str, Path)):
            path = Path(image)
            if path.suffix.lower() == ".txt":
                return load_board(path)
            raise NotImplementedError(
                f"StubBoardParser cannot parse image file {path}; use a .txt board fixture"
            )

        raise NotImplementedError(
            "StubBoardParser does not implement screenshot CV; pass a .txt path or set fixture_path"
        )
