"""Debug overlay and window for vision (read --show)."""

from __future__ import annotations

import base64
import tkinter as tk

import cv2
import numpy as np

from meoudoku.vision.cv_parser import (
    ColorRun,
    ParseResult,
    _content_bbox,
    _homogeneous_color_run_spans,
    decode_image,
    estimate_n,
)
from meoudoku.vision.roi import (
    Roi,
    _edge_onboard_frac,
    estimate_page_bg,
    find_board_roi,
    is_on_board_pixel,
    is_page_bg_pixel,
)


def _draw_hline(img: np.ndarray, y: int, x0: int, x1: int, color: tuple[int, int, int], thickness: int = 1) -> None:
    cv2.line(img, (x0, y), (x1, y), color, thickness, lineType=cv2.LINE_AA)


def _draw_vline(img: np.ndarray, x: int, y0: int, y1: int, color: tuple[int, int, int], thickness: int = 1) -> None:
    cv2.line(img, (x, y0), (x, y1), color, thickness, lineType=cv2.LINE_AA)


def _paint_run_h(
    img: np.ndarray,
    y: int,
    x_base: int,
    runs: list[ColorRun],
    color: tuple[int, int, int],
) -> None:
    for run in runs:
        for x in range(x_base + run.start, x_base + run.end):
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                img[y, x] = color
        mx = x_base + (run.start + run.end) // 2
        cv2.circle(img, (mx, y), 3, color, -1, lineType=cv2.LINE_AA)


def _paint_run_v(
    img: np.ndarray,
    x: int,
    y_base: int,
    runs: list[ColorRun],
    color: tuple[int, int, int],
) -> None:
    for run in runs:
        for y in range(y_base + run.start, y_base + run.end):
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                img[y, x] = color
        my = y_base + (run.start + run.end) // 2
        cv2.circle(img, (x, my), 3, color, -1, lineType=cv2.LINE_AA)


def _trace_ray_pixels(
    bgr: np.ndarray,
    x0: int,
    y0: int,
    dx: int,
    dy: int,
    page_bg: np.ndarray,
    *,
    streak_required: int = 4,
) -> tuple[list[tuple[int, int, bool]], tuple[int, int]]:
    h, w = bgr.shape[:2]
    x, y = x0, y0
    last_on = (x0, y0)
    streak = 0
    pixels: list[tuple[int, int, bool]] = []
    while 0 <= x < w and 0 <= y < h:
        is_bg = is_page_bg_pixel(bgr[y, x], page_bg)
        pixels.append((x, y, is_bg))
        if is_bg:
            streak += 1
            if streak >= streak_required:
                break
        else:
            streak = 0
            last_on = (x, y)
        x += dx
        y += dy
    return pixels, last_on


def _draw_roi_layer(
    overlay: np.ndarray,
    bgr: np.ndarray,
    used: Roi,
    page_bg: np.ndarray,
    info: dict,
) -> None:
    h, w = bgr.shape[:2]
    cx, cy = w // 2, h // 2
    seed = (cx, cy)
    if not is_on_board_pixel(bgr[cy, cx], page_bg):
        found = False
        for radius in range(1, min(h, w) // 4):
            for yy in range(cy - radius, cy + radius + 1):
                for xx in range(cx - radius, cx + radius + 1):
                    if 0 <= xx < w and 0 <= yy < h and is_on_board_pixel(bgr[yy, xx], page_bg):
                        seed = (xx, yy)
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            info["roi_error"] = "Could not find board plate from image center"
            return

    sx, sy = seed
    info["seed"] = seed
    cv2.circle(overlay, seed, 6, (0, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(overlay, seed, 8, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    stops: dict[str, tuple[int, int]] = {}
    for name, (dx, dy) in {
        "left": (-1, 0),
        "right": (1, 0),
        "up": (0, -1),
        "down": (0, 1),
    }.items():
        pixels, last_on = _trace_ray_pixels(bgr, sx, sy, dx, dy, page_bg)
        stops[name] = last_on
        for x, y, is_bg in pixels:
            overlay[y, x] = (0, 0, 255) if is_bg else (80, 220, 80)
        cv2.circle(overlay, last_on, 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    info["stops"] = {k: {"x": v[0], "y": v[1]} for k, v in stops.items()}
    info["roi"] = (used.left, used.top, used.right, used.bottom)
    info["roi_size"] = (used.right - used.left, used.bottom - used.top)

    cv2.rectangle(
        overlay,
        (used.left, used.top),
        (used.right - 1, used.bottom - 1),
        (255, 255, 0),
        2,
        lineType=cv2.LINE_AA,
    )

    edge_fracs: dict[str, dict[str, float]] = {}
    for inset, color in ((2, (255, 128, 0)), (3, (255, 200, 0))):
        y_top = used.top + inset
        y_bot = used.bottom - 1 - inset
        x_left = used.left + inset
        x_right = used.right - 1 - inset
        _draw_hline(overlay, y_top, used.left, used.right - 1, color)
        _draw_hline(overlay, y_bot, used.left, used.right - 1, color)
        _draw_vline(overlay, x_left, used.top, used.bottom - 1, color)
        _draw_vline(overlay, x_right, used.top, used.bottom - 1, color)
        edge_fracs[f"inset_{inset}"] = {
            side: round(_edge_onboard_frac(bgr, used, page_bg, side=side, inset=inset), 3)
            for side in ("top", "bottom", "left", "right")
        }
    info["edge_fracs"] = edge_fracs


def build_read_debug_image(
    image: bytes | np.ndarray | str,
    parsed: ParseResult | None = None,
    *,
    insets: tuple[int, ...] = (2, 3),
) -> tuple[np.ndarray, dict]:
    """
    Full vision debug overlay.

    Colors:
      yellow seed / cyan ROI / orange ROI edge-check
      green ray on-board / red ray page-bg / white ray stop
      yellow content bbox
      magenta/orange N-scan insets / lime N color-runs
      blue cell color sample strips / red grid
    """
    bgr = decode_image(image)
    overlay = bgr.copy()
    page_bg = estimate_page_bg(bgr)
    info: dict = {
        "page_bg_bgr": [float(x) for x in page_bg],
        "image_size": (int(bgr.shape[1]), int(bgr.shape[0])),
    }

    if parsed is not None:
        roi = parsed.roi
        grid = parsed.grid
        n = parsed.board.n
    else:
        roi = find_board_roi(bgr, page_bg=page_bg)
        grid = None
        n = None

    _draw_roi_layer(overlay, bgr, roi, page_bg, info)

    plate = bgr[roi.top : roi.bottom, roi.left : roi.right]
    try:
        cl, ct, cr, cb = _content_bbox(plate)
    except Exception as exc:  # noqa: BLE001 — debug path
        info["content_error"] = str(exc)
        cl = ct = 0
        cr, cb = plate.shape[1] - 1, plate.shape[0] - 1

    abs_l, abs_t = roi.left + cl, roi.top + ct
    abs_r, abs_b = roi.left + cr, roi.top + cb
    info["content"] = (abs_l, abs_t, abs_r, abs_b)
    cv2.rectangle(overlay, (abs_l, abs_t), (abs_r, abs_b), (0, 255, 255), 2)

    side_counts: dict[str, dict[str, int]] = {}
    inset_colors = {2: (255, 0, 255), 3: (0, 140, 255)}
    run_color = (0, 255, 0)
    for inset in insets:
        color = inset_colors.get(inset, (200, 200, 200))
        y_top = abs_t + inset
        y_bot = abs_b - inset
        _draw_hline(overlay, y_top, abs_l, abs_r, color)
        _draw_hline(overlay, y_bot, abs_l, abs_r, color)
        top_runs = _homogeneous_color_run_spans(plate[ct + inset, cl : cr + 1])
        bot_runs = _homogeneous_color_run_spans(plate[cb - inset, cl : cr + 1])
        _paint_run_h(overlay, y_top, abs_l, top_runs, run_color)
        _paint_run_h(overlay, y_bot, abs_l, bot_runs, run_color)

        x_left = abs_l + inset
        x_right = abs_r - inset
        _draw_vline(overlay, x_left, abs_t, abs_b, color)
        _draw_vline(overlay, x_right, abs_t, abs_b, color)
        left_runs = _homogeneous_color_run_spans(plate[ct : cb + 1, cl + inset])
        right_runs = _homogeneous_color_run_spans(plate[ct : cb + 1, cr - inset])
        _paint_run_v(overlay, x_left, abs_t, left_runs, run_color)
        _paint_run_v(overlay, x_right, abs_t, right_runs, run_color)

        side_counts[f"inset_{inset}"] = {
            "top": len(top_runs),
            "bottom": len(bot_runs),
            "left": len(left_runs),
            "right": len(right_runs),
        }

    est = estimate_n(plate)
    info["estimate"] = None if est is None else {
        "n": est.n,
        "source": est.source,
        "confident": est.confident,
        "votes": est.votes,
    }
    info["side_counts"] = side_counts

    if grid is not None and n is not None:
        for x in grid.xs:
            _draw_vline(overlay, roi.left + x, roi.top, roi.bottom - 1, (0, 0, 255))
        for y in grid.ys:
            _draw_hline(overlay, roi.top + y, roi.left, roi.right - 1, (0, 0, 255))

        edge_inset, patch = 2, 4
        for r in range(n):
            for c in range(n):
                x0, y0, x1, y1 = grid.cell_rect(r, c)
                sh, sw = y1 - y0, x1 - x0
                if sw < edge_inset + patch + 1 or sh < 8:
                    continue
                y_pad = max(2, sh // 8)
                y_a, y_b = y0 + y_pad, y1 - y_pad
                for xa, xb in (
                    (x0 + edge_inset, x0 + edge_inset + patch),
                    (x1 - edge_inset - patch, x1 - edge_inset),
                ):
                    cv2.rectangle(
                        overlay,
                        (roi.left + xa, roi.top + y_a),
                        (roi.left + xb - 1, roi.top + y_b - 1),
                        (255, 80, 0),
                        1,
                    )
        info["parsed_n"] = n

    legend = [
        "ROI: yellow seed, cyan box, orange edge-check, green/red rays",
        "N: magenta/orange scan, lime runs; yellow=content; blue=color; red=grid",
    ]
    if info.get("estimate"):
        e = info["estimate"]
        legend.append(f"estimate N={e['n']} source={e['source']} conf={e['confident']}")
    if n is not None:
        legend.append(f"parsed N={n} ROI={info.get('roi_size')}")
    for i, line in enumerate(legend):
        cv2.putText(overlay, line, (12, 28 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(overlay, line, (12, 28 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

    return overlay, info


def build_roi_debug_image(
    image: bytes | np.ndarray | str,
    *,
    roi: Roi | None = None,
    inset: int = 3,
) -> tuple[np.ndarray, dict]:
    """ROI-only overlay (subset of full debug)."""
    bgr = decode_image(image)
    overlay = bgr.copy()
    page_bg = estimate_page_bg(bgr)
    info: dict = {
        "page_bg_bgr": [float(x) for x in page_bg],
        "image_size": (int(bgr.shape[1]), int(bgr.shape[0])),
    }
    used = roi if roi is not None else find_board_roi(bgr, inset=inset, page_bg=page_bg)
    _draw_roi_layer(overlay, bgr, used, page_bg, info)
    legend = [
        "yellow=seed  green=on-board ray  red=page-bg  white=stop  cyan=ROI  orange=edge",
        f"ROI LTRB={used.left},{used.top},{used.right},{used.bottom} "
        f"size={used.right - used.left}x{used.bottom - used.top}",
    ]
    for i, line in enumerate(legend):
        cv2.putText(overlay, line, (12, 28 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(overlay, line, (12, 28 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return overlay, info


def _bgr_to_photo(bgr: np.ndarray, *, max_w: int = 1400, max_h: int = 900) -> tk.PhotoImage:
    h, w = bgr.shape[:2]
    scale = min(1.0, max_w / w, max_h / h)
    if scale < 1.0:
        disp = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        disp = bgr
    ok, buf = cv2.imencode(".png", disp)
    if not ok:
        raise RuntimeError("Failed to encode debug image")
    return tk.PhotoImage(data=base64.b64encode(buf.tobytes()))


class DebugViewer:
    """Reusable Tk window: update() redraws without blocking the caller."""

    def __init__(self, *, title: str = "meoudoku") -> None:
        self.root = tk.Tk()
        self.root.title(title)
        self.label = tk.Label(self.root)
        self.label.pack()
        self._photo: tk.PhotoImage | None = None
        self._closed = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Key>", self._on_key)

    def _on_key(self, event: tk.Event) -> None:
        if event.keysym in ("q", "Q", "Escape"):
            self.close()

    @property
    def closed(self) -> bool:
        return self._closed

    def update(self, bgr: np.ndarray, *, title: str | None = None) -> None:
        if self._closed:
            return
        if title is not None:
            self.root.title(title)
        self._photo = _bgr_to_photo(bgr)
        self.label.configure(image=self._photo)
        self.label.image = self._photo
        self.root.update_idletasks()
        self.root.update()

    def wait(self) -> None:
        """Block until the user closes the window (for one-shot read --show)."""
        if self._closed:
            return
        self.root.mainloop()
        self._closed = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def show_debug_window(bgr: np.ndarray, *, title: str = "meoudoku read --show") -> None:
    """Show BGR image and block until the window is closed (q / Esc)."""
    viewer = DebugViewer(title=title)
    viewer.update(bgr, title=title)
    viewer.wait()
