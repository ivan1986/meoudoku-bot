"""CLI entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from meoudoku.board import load_board
from meoudoku.engine import Solver, UnsolvedError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meoudoku", description="Cat puzzle solver")
    sub = parser.add_subparsers(dest="command", required=True)

    solve_p = sub.add_parser("solve", help="Solve a board text file")
    solve_p.add_argument("board", type=Path, help="Path to COLORS/STATE board file")
    solve_p.add_argument("-v", "--verbose", action="store_true", help="Print applied heuristics")

    step_p = sub.add_parser("step", help="Apply one solver heuristic and show what changed")
    step_p.add_argument(
        "board",
        type=Path,
        nargs="?",
        default=None,
        help="Path to COLORS/STATE board file (omit with --adb)",
    )
    step_p.add_argument("--adb", action="store_true", help="Read board from phone screenshot via adb")
    step_p.add_argument("--serial", default=None, help="ADB device serial")
    step_p.add_argument("--n", type=int, default=None, help="Force board size N (with --adb)")
    step_p.add_argument("--roi", type=str, default=None, help="Override ROI as L,T,R,B (with --adb)")
    step_p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write updated board file (with --adb: write instead of tapping)",
    )
    step_p.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write/tap; only print heuristic/delta and resulting board",
    )

    capture_p = sub.add_parser("capture", help="Capture screenshot from connected Android device")
    capture_p.add_argument("-o", "--output", type=Path, default=Path("capture.png"), help="Output PNG path")
    capture_p.add_argument("--serial", default=None, help="ADB device serial")

    read_p = sub.add_parser("read", help="Parse board from screenshot (ADB or image file)")
    read_p.add_argument("--image", type=Path, default=None, help="PNG/JPG path (default: capture via adb)")
    read_p.add_argument("--adb", action="store_true", help="Force capture via adb")
    read_p.add_argument("--serial", default=None, help="ADB device serial")
    read_p.add_argument("--n", type=int, default=None, help="Force board size N")
    read_p.add_argument("--roi", type=str, default=None, help="Override ROI as L,T,R,B pixels")
    read_p.add_argument("-o", "--output", type=Path, default=None, help="Write board text file")
    read_p.add_argument("--solve", action="store_true", help="Solve after parsing (no taps)")
    read_p.add_argument("-v", "--verbose", action="store_true", help="Verbose solve log")
    read_p.add_argument(
        "--show",
        action="store_true",
        help="Open debug window: ROI rays, N-edge scans, grid, color sample strips",
    )

    play_p = sub.add_parser("play", help="Full solve loop on phone: read → step → tap → reread")
    play_p.add_argument("--serial", default=None, help="ADB device serial")
    play_p.add_argument("--n", type=int, default=None, help="Force initial board size N")
    play_p.add_argument("--roi", type=str, default=None, help="Override ROI as L,T,R,B pixels")
    play_p.add_argument("--max-rounds", type=int, default=500, help="Safety limit for loop rounds")
    play_p.add_argument(
        "--settle",
        type=float,
        default=0.35,
        help="Seconds to wait after taps and between the two post-step re-reads",
    )
    play_p.add_argument(
        "--show",
        action="store_true",
        help="Live vision debug window (redraw after each read, no pause)",
    )
    play_p.add_argument(
        "--no-reread",
        action="store_true",
        help="Do not re-read the screen after taps; keep an in-memory board",
    )

    full_p = sub.add_parser(
        "full",
        help="Launch app, tap level, play loop, wait dark, force-stop",
    )
    full_p.add_argument("--serial", default=None, help="ADB device serial")
    full_p.add_argument(
        "--package",
        default="com.oakever.meowdoku",
        help="Android package name",
    )
    full_p.add_argument(
        "--activity",
        default="com.godot.game.GodotAppLauncher",
        help="Launcher activity",
    )
    full_p.add_argument(
        "--tap-frac",
        default="0.5,0.72",
        help="Level button as FX,FY fractions (default: 0.5,0.72)",
    )
    full_p.add_argument(
        "--tap",
        default=None,
        help="Level button as X,Y pixels (overrides --tap-frac)",
    )
    full_p.add_argument(
        "--banner-frac",
        default="0.5,0.48",
        help=argparse.SUPPRESS,  # unused; kept for CLI compat
    )
    full_p.add_argument(
        "--banner-delay",
        type=float,
        default=5.0,
        help="Seconds to sleep after level tap before play (default: 5)",
    )
    full_p.add_argument(
        "--post-win-tap-frac",
        default="0.5,0.92",
        help="Tap after solve to dismiss leaderboard / continue (FX,FY)",
    )
    full_p.add_argument("--menu-timeout", type=float, default=60.0, help="Wait for orange button")
    full_p.add_argument(
        "--banner-gone-timeout",
        type=float,
        default=10.0,
        help=argparse.SUPPRESS,  # unused; kept for CLI compat
    )
    full_p.add_argument(
        "--board-timeout",
        type=float,
        default=30.0,
        help=argparse.SUPPRESS,  # unused; kept for CLI compat
    )
    full_p.add_argument(
        "--dark-timeout",
        type=float,
        default=5.0,
        help="Wait for post-win dark fade (soft-skip on timeout; does not fail the run)",
    )
    full_p.add_argument("--poll", type=float, default=0.35, help="Screenshot poll interval seconds")
    full_p.add_argument("--n", type=int, default=None, help="Force board size N")
    full_p.add_argument("--roi", type=str, default=None, help="Override ROI as L,T,R,B pixels")
    full_p.add_argument("--max-rounds", type=int, default=500, help="Play loop safety limit")
    full_p.add_argument(
        "--settle",
        type=float,
        default=0.35,
        help="Seconds to wait after taps / between re-reads",
    )
    full_p.add_argument(
        "--show",
        action="store_true",
        help="Live vision debug window during play",
    )
    full_p.add_argument(
        "--no-reread",
        action="store_true",
        help="Do not re-read the screen after taps during play",
    )
    full_p.add_argument(
        "--repeat",
        action="store_true",
        help="Repeat full sessions until Ctrl+C (errors do not stop the loop)",
    )

    args = parser.parse_args(argv)

    if args.command == "solve":
        return _cmd_solve(args.board, verbose=args.verbose)

    if args.command == "step":
        return _cmd_step(args)

    if args.command == "capture":
        return _cmd_capture(args.output, serial=args.serial)

    if args.command == "read":
        return _cmd_read(args)

    if args.command == "play":
        return _cmd_play(args)

    if args.command == "full":
        return _cmd_full(args)

    return 2


def _cmd_solve(path: Path, *, verbose: bool) -> int:
    board = load_board(path)
    try:
        result = Solver().solve(board)
    except UnsolvedError as exc:
        print(f"unsolved: {exc}", file=sys.stderr)
        return 1
    if verbose:
        for name, delta in result.applied:
            print(f"{name}: cats={sorted(delta.cats)} xs={sorted(delta.xs)}")
        print(f"steps={result.steps} guesses={result.guesses}")
    print(result.board.dump(), end="")
    return 0


def _load_parse_from_adb(*, serial: str | None, n: int | None, roi_arg: str | None):
    from meoudoku.device import AdbDevice, AdbError
    from meoudoku.vision import CvBoardParser, VisionError, parse_roi_arg

    roi = parse_roi_arg(roi_arg) if roi_arg else None
    try:
        device = AdbDevice(serial=serial)
        png = device.screenshot()
        parsed = CvBoardParser(n=n, roi=roi).parse_ex(png)
        return device, parsed
    except (AdbError, VisionError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc


def _apply_delta_taps(device, parsed, delta) -> None:
    """One tap = X, double tap = cat. Geometry from ParseResult."""
    from meoudoku.device import AdbError

    for r, c in sorted(delta.xs):
        x, y = parsed.cell_center(r, c)
        print(f"tap: X ({r},{c}) -> ({x},{y})")
        try:
            device.tap(x, y, times=1)
        except AdbError as exc:
            raise RuntimeError(f"tap failed at ({x},{y}): {exc}") from exc
    for r, c in sorted(delta.cats):
        x, y = parsed.cell_center(r, c)
        print(f"tap: CAT ({r},{c}) -> ({x},{y}) x2")
        try:
            device.tap(x, y, times=2)
        except AdbError as exc:
            raise RuntimeError(f"tap failed at ({x},{y}): {exc}") from exc


def _cmd_step(args: argparse.Namespace) -> int:
    from meoudoku.board import BoardError

    device = None
    parsed = None

    if args.adb:
        if args.board is not None:
            print("step: do not pass board path together with --adb", file=sys.stderr)
            return 2
        try:
            device, parsed = _load_parse_from_adb(serial=args.serial, n=args.n, roi_arg=args.roi)
        except RuntimeError as exc:
            print(f"adb/read error: {exc}", file=sys.stderr)
            return 1
        board = parsed.board
        print("source: adb")
        print(f"N: {board.n}")
    else:
        if args.board is None:
            print("step: board path required (or use --adb)", file=sys.stderr)
            return 2
        board = load_board(args.board)
        print(f"source: {args.board}")

    if board.is_solved():
        print("heuristic: (none)")
        print("change: board already solved")
        print(board.dump(), end="")
        return 0

    solver = Solver()
    hit = solver.step(board)
    if hit is None:
        print("heuristic: (none)")
        print("change: no heuristic could progress")
        print(board.dump(), end="")
        return 1

    name, delta = hit
    if name == "backtrack":
        cells = ", ".join(f"({r},{c})" for r, c in sorted(delta.cats)) or "-"
        print(f"heuristic: {name}")
        print(f"change: branches=[{cells}] (in-memory only; not applied)")
        if args.adb:
            print("step: refuse to trial-tap backtrack on device", file=sys.stderr)
            return 1
        return 0

    try:
        board.apply(delta)
    except BoardError as exc:
        print(f"heuristic: {name}", file=sys.stderr)
        print(f"change: apply failed: {exc}", file=sys.stderr)
        return 1

    cats = ", ".join(f"({r},{c})" for r, c in sorted(delta.cats)) or "-"
    xs = ", ".join(f"({r},{c})" for r, c in sorted(delta.xs)) or "-"
    print(f"heuristic: {name}")
    print(f"change: cats=[{cats}] xs=[{xs}]")
    if board.is_solved():
        print("status: solved")
    elif board.has_contradiction():
        print("status: contradiction")
    else:
        print("status: in_progress")

    text = board.dump()
    tap_mode = bool(args.adb and args.output is None and not args.no_write)

    if tap_mode:
        assert device is not None and parsed is not None
        try:
            _apply_delta_taps(device, parsed, delta)
        except RuntimeError as exc:
            print(f"tap error: {exc}", file=sys.stderr)
            print(text, end="")
            return 1
        print(f"tapped: {len(delta.xs)} X, {len(delta.cats)} CAT")
    elif not args.no_write:
        out = args.output if args.output is not None else args.board
        assert out is not None
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote: {out}")

    print(text, end="")
    return 0


def _cmd_capture(output: Path, *, serial: str | None) -> int:
    from meoudoku.device import AdbDevice, AdbError

    try:
        device = AdbDevice(serial=serial)
        png = device.screenshot()
    except AdbError as exc:
        print(f"adb error: {exc}", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(png)
    print(f"wrote {output} ({len(png)} bytes)")
    return 0


def _cmd_read(args: argparse.Namespace) -> int:
    from meoudoku.device import AdbDevice, AdbError
    from meoudoku.vision import CvBoardParser, VisionError, parse_roi_arg
    from meoudoku.vision.debug import build_read_debug_image, show_debug_window

    roi = parse_roi_arg(args.roi) if args.roi else None

    try:
        if args.image is not None and not args.adb:
            image: bytes | Path = args.image
        else:
            device = AdbDevice(serial=args.serial)
            image = device.screenshot()
        parsed = CvBoardParser(n=args.n, roi=roi).parse_ex(image)
        board = parsed.board
    except (AdbError, VisionError, ValueError) as exc:
        print(f"read error: {exc}", file=sys.stderr)
        return 1

    print(f"N: {board.n}")
    text = board.dump()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")

    if args.show:
        overlay, info = build_read_debug_image(image, parsed)
        print(f"debug: seed={info.get('seed')} page_bg={info.get('page_bg_bgr')}")
        print(f"debug: roi LTRB={info.get('roi')} size={info.get('roi_size')}")
        print(f"debug: ray stops={info.get('stops')}")
        print(f"debug: edge_fracs={info.get('edge_fracs')}")
        print(f"debug: content={info.get('content')}")
        est = info.get("estimate")
        if est:
            print(
                f"debug: estimate N={est['n']} source={est['source']} "
                f"confident={est['confident']} votes={est['votes']}"
            )
        for key, counts in (info.get("side_counts") or {}).items():
            print(f"debug: {key} {counts}")
        if info.get("parsed_n") is not None:
            print(f"debug: parsed N={info['parsed_n']}")
        show_debug_window(overlay, title=f"meoudoku read — N={board.n}")

    if args.solve:
        try:
            result = Solver().solve(board)
        except UnsolvedError as exc:
            print(text, end="")
            print(f"unsolved: {exc}", file=sys.stderr)
            return 1
        if args.verbose:
            for name, delta in result.applied:
                print(f"{name}: cats={sorted(delta.cats)} xs={sorted(delta.xs)}")
            print(f"steps={result.steps} guesses={result.guesses}")
        print(result.board.dump(), end="")
        return 0

    print(text, end="")
    return 0


def _cmd_play(args: argparse.Namespace) -> int:
    from meoudoku.play import PlayError, play_loop

    try:
        result = play_loop(
            serial=args.serial,
            n=args.n,
            roi=args.roi,
            max_rounds=args.max_rounds,
            settle_s=args.settle,
            show=args.show,
            reread=not args.no_reread,
            on_event=print,
        )
    except PlayError as exc:
        print(f"play error: {exc}", file=sys.stderr)
        return 1

    print(f"rounds: {result.rounds}")
    print(f"N: {result.n}")
    print(result.board.dump(), end="")
    return 0


def _cmd_full(args: argparse.Namespace) -> int:
    import time

    from meoudoku.full import FullError, full_loop

    def _log(msg: str) -> None:
        print(msg, flush=True)

    def _run_once() -> int:
        try:
            result = full_loop(
                serial=args.serial,
                package=args.package,
                activity=args.activity,
                tap_frac=args.tap_frac,
                tap_xy=args.tap,
                banner_frac=args.banner_frac,
                banner_delay_s=args.banner_delay,
                post_win_tap_frac=args.post_win_tap_frac,
                menu_timeout_s=args.menu_timeout,
                banner_gone_timeout_s=args.banner_gone_timeout,
                board_timeout_s=args.board_timeout,
                dark_timeout_s=args.dark_timeout,
                poll_s=args.poll,
                n=args.n,
                roi=args.roi,
                max_rounds=args.max_rounds,
                settle_s=args.settle,
                show=args.show,
                reread=not args.no_reread,
                on_event=_log,
            )
        except (FullError, ValueError) as exc:
            print(f"full error: {exc}", file=sys.stderr, flush=True)
            return 1

        print(f"rounds: {result.rounds}", flush=True)
        print(f"N: {result.n}", flush=True)
        print(result.board.dump(), end="", flush=True)
        return 0

    if not args.repeat:
        return _run_once()

    session = 0
    ok = 0
    fail = 0
    _log("full: --repeat on (Ctrl+C to stop)")
    try:
        while True:
            session += 1
            _log(f"full: session {session} start")
            rc = _run_once()
            if rc == 0:
                ok += 1
            else:
                fail += 1
            _log(f"full: session {session} done rc={rc} (ok={ok} fail={fail})")
            time.sleep(2.0)
    except KeyboardInterrupt:
        _log(f"full: stopped by Ctrl+C after {session} session(s) (ok={ok} fail={fail})")
        return 0 if fail == 0 or ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
