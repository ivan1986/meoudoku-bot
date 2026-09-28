"""Full phone play loop: read → one heuristic → tap → reread until solved."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from meoudoku.board import Board, BoardError, Delta
from meoudoku.device.adb import AdbDevice, AdbError
from meoudoku.engine import Solver, UnsolvedError
from meoudoku.vision.cv_parser import CvBoardParser, ParseResult, VisionError
from meoudoku.vision.roi import Roi, parse_roi_arg


class PlayError(RuntimeError):
    """Play loop failed (read/N mismatch/taps/stuck)."""


@dataclass
class PlayStep:
    round_no: int
    n: int
    heuristic: str
    delta: Delta
    board: Board


@dataclass
class PlayResult:
    board: Board
    n: int
    rounds: int
    steps: list[PlayStep] = field(default_factory=list)


@dataclass(frozen=True)
class BoardRead:
    """Parsed board plus the screenshot bytes used for that parse."""

    parsed: ParseResult
    image: bytes


def read_board(
    device: AdbDevice,
    *,
    expected_n: int | None = None,
    n_force: int | None = None,
    roi: Roi | None = None,
    retries: int = 3,
    retry_delay_s: float = 0.25,
) -> BoardRead:
    """
    Screenshot + parse. If expected_n is set, every successful parse must have
    board.n == expected_n (retries on mismatch).
    """
    parse_n = n_force if n_force is not None else expected_n
    last_err: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            png = device.screenshot()
            parsed = CvBoardParser(n=parse_n, roi=roi).parse_ex(png)
        except (AdbError, VisionError, ValueError) as exc:
            last_err = exc
            time.sleep(retry_delay_s)
            continue

        if expected_n is not None and parsed.board.n != expected_n:
            last_err = PlayError(
                f"board size mismatch: got N={parsed.board.n}, expected N={expected_n} "
                f"(attempt {attempt + 1}/{retries})"
            )
            time.sleep(retry_delay_s)
            continue
        return BoardRead(parsed=parsed, image=png)

    raise PlayError(str(last_err) if last_err else "failed to read board")


def read_board_stable(
    device: AdbDevice,
    *,
    expected_n: int,
    n_force: int | None = None,
    roi: Roi | None = None,
    retries: int = 3,
    pause_s: float = 0.35,
    retry_delay_s: float = 0.25,
) -> BoardRead:
    """
    Read the board twice with a pause; both parses must report the same N.

    Guards against mid-animation frames where gap/edge detection briefly drifts.
    Returns the second read when sizes match.
    """
    last_err: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            first = read_board(
                device,
                expected_n=expected_n,
                n_force=n_force,
                roi=roi,
                retries=1,
                retry_delay_s=0.0,
            )
            time.sleep(pause_s)
            second = read_board(
                device,
                expected_n=expected_n,
                n_force=n_force,
                roi=roi,
                retries=1,
                retry_delay_s=0.0,
            )
        except PlayError as exc:
            last_err = exc
            time.sleep(retry_delay_s)
            continue

        if first.parsed.board.n != second.parsed.board.n:
            last_err = PlayError(
                f"unstable board size across reads: {first.parsed.board.n} then "
                f"{second.parsed.board.n} (attempt {attempt + 1}/{retries})"
            )
            time.sleep(retry_delay_s)
            continue
        return second

    raise PlayError(str(last_err) if last_err else "failed to read stable board")


def apply_taps(device: AdbDevice, parsed: ParseResult, delta: Delta) -> None:
    """One tap = X, double tap = cat."""
    for r, c in sorted(delta.xs):
        x, y = parsed.cell_center(r, c)
        device.tap(x, y, times=1)
    for r, c in sorted(delta.cats):
        x, y = parsed.cell_center(r, c)
        device.tap(x, y, times=2)


def _redraw_debug(
    viewer,
    image: bytes,
    parsed: ParseResult,
    *,
    title: str,
    on_event=None,
) -> None:
    from meoudoku.vision.debug import build_read_debug_image

    if viewer is None or viewer.closed:
        return
    overlay, info = build_read_debug_image(image, parsed)
    if on_event is not None:
        on_event(f"debug: seed={info.get('seed')} roi={info.get('roi')} size={info.get('roi_size')}")
        est = info.get("estimate")
        if est:
            on_event(
                f"debug: estimate N={est['n']} source={est['source']} "
                f"confident={est['confident']}"
            )
        for key, counts in (info.get("side_counts") or {}).items():
            on_event(f"debug: {key} {counts}")
    viewer.update(overlay, title=title)


def play_loop(
    *,
    serial: str | None = None,
    device: AdbDevice | None = None,
    n: int | None = None,
    roi: str | Roi | None = None,
    max_rounds: int = 500,
    settle_s: float = 0.35,
    read_retries: int = 3,
    show: bool = False,
    reread: bool = True,
    on_event=None,
) -> PlayResult:
    """
    Full solve on device:
      read → step heuristic → tap → (re-read) → repeat.

    reread=True (default): re-read the screen after every tap before the next decision.
    reread=False: keep an in-memory board after taps (faster; may drift from the phone).
    show=True: live vision debug window redraw (no pause).
    """
    from meoudoku.vision.debug import DebugViewer

    if device is None:
        device = AdbDevice(serial=serial)
    roi_obj: Roi | None
    if isinstance(roi, str):
        roi_obj = parse_roi_arg(roi)
    else:
        roi_obj = roi

    def emit(msg: str) -> None:
        if on_event is not None:
            on_event(msg)

    viewer = DebugViewer(title="meoudoku play") if show else None
    try:
        capture = read_board(device, expected_n=None, n_force=n, roi=roi_obj, retries=read_retries)
        parsed = capture.parsed
        last_image = capture.image
        expected_n = parsed.board.n
        emit(f"N: {expected_n}")
        emit(f"source: adb (initial)")
        emit(f"reread: {'on' if reread else 'off'}")
        _redraw_debug(
            viewer,
            last_image,
            parsed,
            title=f"play initial — N={expected_n}",
            on_event=on_event,
        )

        solver = Solver()
        steps: list[PlayStep] = []

        for round_no in range(1, max_rounds + 1):
            board = parsed.board
            if board.is_solved():
                emit("status: solved")
                return PlayResult(board=board, n=expected_n, rounds=round_no - 1, steps=steps)

            hit = solver.step(board)
            if hit is None:
                raise PlayError("no heuristic could progress")

            name, delta = hit
            if name == "backtrack":
                raise PlayError(
                    "backtrack is in-memory only; cannot trial-tap candidates on the device"
                )

            planned = board.copy()
            try:
                planned.apply(delta)
            except BoardError as exc:
                raise PlayError(f"apply failed ({name}): {exc}") from exc

            emit(f"round: {round_no}")
            emit(f"heuristic: {name}")
            cats = ", ".join(f"({r},{c})" for r, c in sorted(delta.cats)) or "-"
            xs = ", ".join(f"({r},{c})" for r, c in sorted(delta.xs)) or "-"
            emit(f"change: cats=[{cats}] xs=[{xs}]")

            if delta.cats or delta.xs:
                try:
                    apply_taps(device, parsed, delta)
                except AdbError as exc:
                    raise PlayError(f"tap failed: {exc}") from exc
                emit(f"tapped: {len(delta.xs)} X, {len(delta.cats)} CAT")
                time.sleep(settle_s)

            steps.append(
                PlayStep(round_no=round_no, n=expected_n, heuristic=name, delta=delta, board=planned)
            )

            if reread:
                emit("read: screen")
                capture = read_board_stable(
                    device,
                    expected_n=expected_n,
                    n_force=None,
                    roi=roi_obj,
                    retries=read_retries,
                    pause_s=settle_s,
                )
                parsed = capture.parsed
                last_image = capture.image
                emit(f"N: {parsed.board.n} (stable)")
            else:
                # Keep tap geometry from the last screenshot; advance logical board only.
                parsed = ParseResult(board=planned, roi=parsed.roi, grid=parsed.grid)
                emit("read: skipped (--no-reread)")

            _redraw_debug(
                viewer,
                last_image,
                parsed,
                title=f"play round {round_no} — N={parsed.board.n}",
                on_event=on_event,
            )

            if parsed.board.is_solved():
                emit("status: solved")
                return PlayResult(
                    board=parsed.board,
                    n=expected_n,
                    rounds=round_no,
                    steps=steps,
                )

        raise PlayError(f"exceeded max_rounds={max_rounds}")
    finally:
        if viewer is not None:
            viewer.close()
