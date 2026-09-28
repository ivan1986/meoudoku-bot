"""Full app session: launch → menu tap → play → wait dark → force-stop."""

from __future__ import annotations

import time
from dataclasses import dataclass

from meoudoku.board import Board
from meoudoku.device.adb import AdbDevice, AdbError
from meoudoku.play import PlayError, PlayResult, play_loop
from meoudoku.vision.cv_parser import CvBoardParser, VisionError
from meoudoku.vision.roi import Roi, parse_roi_arg
from meoudoku.vision.screen_wait import (
    DARK_BG_MAX_LUMA,
    ScreenWaitError,
    decode_bgr,
    frac_to_xy,
    is_dark_bg,
    is_orange_at,
    is_post_win_dark,
    parse_frac_pair,
    parse_xy_pair,
    patch_mean_luma,
    post_win_band_lumas,
    sample_patch_bgr,
    wait_until,
)

DEFAULT_PACKAGE = "com.oakever.meowdoku"
DEFAULT_ACTIVITY = "com.godot.game.GodotAppLauncher"
DEFAULT_TAP_FRAC = (0.5, 0.72)
DEFAULT_BANNER_FRAC = (0.5, 0.48)
# Leaderboard / "tap to continue" after a win sits near the bottom.
DEFAULT_POST_WIN_TAP_FRAC = (0.5, 0.92)


class FullError(RuntimeError):
    """Full app session failed."""


@dataclass
class FullResult:
    board: Board
    n: int
    rounds: int
    play: PlayResult


def full_loop(
    *,
    serial: str | None = None,
    package: str = DEFAULT_PACKAGE,
    activity: str = DEFAULT_ACTIVITY,
    tap_frac: tuple[float, float] | str = DEFAULT_TAP_FRAC,
    tap_xy: tuple[int, int] | str | None = None,
    banner_frac: tuple[float, float] | str = DEFAULT_BANNER_FRAC,
    banner_delay_s: float = 5.0,
    post_win_tap_frac: tuple[float, float] | str = DEFAULT_POST_WIN_TAP_FRAC,
    menu_timeout_s: float = 60.0,
    banner_gone_timeout_s: float = 10.0,
    board_timeout_s: float = 30.0,
    dark_timeout_s: float = 5.0,
    dark_hold_s: float = 0.0,
    poll_s: float = 0.35,
    n: int | None = None,
    roi: str | Roi | None = None,
    max_rounds: int = 500,
    settle_s: float = 0.35,
    show: bool = False,
    reread: bool = True,
    on_event=None,
) -> FullResult:
    """
    Launch Meowdoku, tap level button, sleep for banner, play, wait dark exit.
    Always force-stops the package in finally.
    """
    _ = (banner_frac, banner_gone_timeout_s, board_timeout_s)  # kept for CLI compat
    device = AdbDevice(serial=serial)

    def emit(msg: str) -> None:
        if on_event is not None:
            on_event(msg)

    if isinstance(tap_frac, str):
        tap_frac = parse_frac_pair(tap_frac)
    if isinstance(post_win_tap_frac, str):
        post_win_tap_frac = parse_frac_pair(post_win_tap_frac)
    if isinstance(tap_xy, str):
        tap_xy = parse_xy_pair(tap_xy)

    roi_obj: Roi | None
    if isinstance(roi, str):
        roi_obj = parse_roi_arg(roi)
    else:
        roi_obj = roi

    try:
        emit(f"full: force-stop {package}")
        try:
            device.force_stop(package)
        except AdbError as exc:
            emit(f"full: force-stop ignored: {exc}")

        emit(f"full: launch {package}/{activity}")
        try:
            device.launch_app(package, activity)
        except AdbError as exc:
            raise FullError(f"launch failed: {exc}") from exc

        try:
            wm_w, wm_h = device.wm_size()
        except AdbError as exc:
            raise FullError(f"wm size failed: {exc}") from exc
        emit(f"full: wm_size {wm_w}x{wm_h}")

        if tap_xy is not None:
            tap_x, tap_y = tap_xy
            use_frac = False
        else:
            tap_x, tap_y = frac_to_xy(wm_w, wm_h, tap_frac[0], tap_frac[1])
            use_frac = True
        emit(f"full: level tap point ({tap_x},{tap_y})")

        emit("full: wait orange level button")

        def _orange_ready(bgr) -> bool:
            if use_frac:
                x, y = frac_to_xy(bgr.shape[1], bgr.shape[0], tap_frac[0], tap_frac[1])
            else:
                x, y = tap_x, tap_y
            return is_orange_at(bgr, x, y)

        try:
            wait_until(
                device,
                _orange_ready,
                timeout_s=menu_timeout_s,
                poll_s=poll_s,
            )
        except ScreenWaitError as exc:
            raise FullError(f"menu orange wait failed: {exc}") from exc

        emit(f"full: tap level ({tap_x},{tap_y})")
        try:
            device.tap(tap_x, tap_y)
        except AdbError as exc:
            raise FullError(f"level tap failed: {exc}") from exc

        # Start banner is optional / flaky to detect — just wait, then play.
        emit(f"full: post-tap delay {banner_delay_s:.1f}s")
        time.sleep(max(0.0, banner_delay_s))

        emit("full: play")
        try:
            play = play_loop(
                device=device,
                serial=serial,
                n=n,
                roi=roi_obj,
                max_rounds=max_rounds,
                settle_s=settle_s,
                show=show,
                reread=reread,
                on_event=on_event,
            )
        except PlayError as exc:
            raise FullError(f"play failed: {exc}") from exc

        # Post-win leaderboard: dark bands at top + bottom (not full-frame black).
        emit("full: wait dark (post-win top+bottom)")
        dark_poll = min(poll_s, 0.1) if poll_s > 0 else 0.1
        try:
            _wait_dark(
                device,
                timeout_s=dark_timeout_s,
                poll_s=dark_poll,
                hold_s=dark_hold_s,
                on_event=emit,
            )
        except ScreenWaitError as exc:
            # Solve already succeeded; overlay may be skipped on some levels.
            emit(f"full: dark wait skipped — {exc}")
        else:
            px, py = frac_to_xy(wm_w, wm_h, post_win_tap_frac[0], post_win_tap_frac[1])
            emit(f"full: post-win tap ({px},{py})")
            try:
                device.tap(px, py)
            except AdbError as exc:
                emit(f"full: post-win tap ignored: {exc}")
            time.sleep(0.35)

        emit("full: done")
        return FullResult(board=play.board, n=play.n, rounds=play.rounds, play=play)
    finally:
        emit(f"full: force-stop {package} (cleanup)")
        try:
            device.force_stop(package)
        except AdbError as exc:
            emit(f"full: cleanup force-stop ignored: {exc}")


def _wait_dark(
    device: AdbDevice,
    *,
    timeout_s: float,
    poll_s: float,
    hold_s: float,
    on_event=None,
) -> None:
    """Poll until top+bottom post-win bands are dark; log band lumas while waiting."""
    deadline = time.monotonic() + max(0.0, timeout_s)
    held_since: float | None = None
    last_log = 0.0
    top_luma = bot_luma = -1.0
    while True:
        png = device.screenshot()
        bgr = decode_bgr(png)
        top_luma, bot_luma = post_win_band_lumas(bgr)
        dark = is_post_win_dark(bgr)
        now = time.monotonic()
        if on_event is not None and now - last_log >= 1.0:
            on_event(f"full: waiting dark — top={top_luma:.0f} bottom={bot_luma:.0f}")
            last_log = now
        if dark:
            if hold_s <= 0:
                if on_event is not None:
                    on_event(f"full: dark ok top={top_luma:.0f} bottom={bot_luma:.0f}")
                return
            if held_since is None:
                held_since = now
            elif now - held_since >= hold_s:
                if on_event is not None:
                    on_event(f"full: dark ok top={top_luma:.0f} bottom={bot_luma:.0f}")
                return
        else:
            held_since = None
        if now >= deadline:
            raise ScreenWaitError(
                f"timeout after {timeout_s:.1f}s (top={top_luma:.0f} bottom={bot_luma:.0f})"
            )
        time.sleep(max(0.01, poll_s))


def _wait_board(
    device: AdbDevice,
    *,
    n_force: int | None,
    roi: Roi | None,
    timeout_s: float,
    poll_s: float,
    on_event=None,
) -> None:
    """Poll until a plausible board parse succeeds (N in 5..20 unless forced)."""
    _wait_board_clear(
        device,
        banner_fx=None,
        banner_fy=None,
        n_force=n_force,
        roi=roi,
        timeout_s=timeout_s,
        poll_s=poll_s,
        on_event=on_event,
    )


def _wait_board_clear(
    device: AdbDevice,
    *,
    banner_fx: float | None,
    banner_fy: float | None,
    n_force: int | None,
    roi: Roi | None,
    timeout_s: float,
    poll_s: float,
    on_event=None,
) -> None:
    """
    Poll until the board parses. Optional center-dark check is status-only:
    a successful parse always wins (start banner usually breaks clustering anyway).
    """
    deadline = time.monotonic() + max(0.0, timeout_s)
    last_err: Exception | None = None
    last_log = 0.0
    while time.monotonic() < deadline:
        try:
            png = device.screenshot()
            bgr = decode_bgr(png)
        except (AdbError, ValueError) as exc:
            last_err = exc
            last_log = _emit_wait(on_event, f"full: waiting — screenshot: {exc}", last_log)
            time.sleep(max(0.01, poll_s))
            continue

        luma = None
        dark = False
        if banner_fx is not None and banner_fy is not None:
            h, w = bgr.shape[:2]
            bx, by = frac_to_xy(w, h, banner_fx, banner_fy)
            luma = patch_mean_luma(sample_patch_bgr(bgr, bx, by, size=21))
            dark = is_dark_bg(bgr, fx=banner_fx, fy=banner_fy, max_luma=DARK_BG_MAX_LUMA)

        try:
            parsed = CvBoardParser(n=n_force, roi=roi).parse_ex(png)
            n = parsed.board.n
            if n_force is not None or 5 <= n <= 20:
                # Reject parse while start-banner center is still dark — overlay
                # can yield a false N/colors on a half-covered plate.
                if dark:
                    last_err = FullError(
                        f"parse N={n} but center still dark luma={luma:.0f}"
                    )
                else:
                    if on_event is not None:
                        on_event(f"full: board ready N={n}")
                    return
            else:
                last_err = FullError(f"board N={n} out of range 5..20")
        except (VisionError, ValueError) as exc:
            if dark and luma is not None:
                last_err = FullError(f"banner/dark center luma={luma:.0f}; parse: {exc}")
            else:
                last_err = FullError(
                    f"center luma={luma:.0f}; parse: {exc}"
                    if luma is not None
                    else f"parse: {exc}"
                )

        now = time.monotonic()
        if on_event is not None and now - last_log >= 0.8:
            on_event(f"full: waiting — {last_err}")
            last_log = now
        time.sleep(max(0.01, poll_s))
    raise FullError(f"board wait timeout: {last_err}")


def _emit_wait(on_event, msg: str, last_log: float) -> float:
    now = time.monotonic()
    if on_event is not None and now - last_log >= 0.8:
        on_event(msg)
        return now
    return last_log
