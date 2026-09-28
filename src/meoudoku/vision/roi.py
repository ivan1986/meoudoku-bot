"""Detect the white board plate ROI by rays from center until page background."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Roi:
    left: int
    top: int
    right: int
    bottom: int

    def clamp(self, width: int, height: int) -> Roi:
        return Roi(
            left=max(0, min(self.left, width - 1)),
            top=max(0, min(self.top, height - 1)),
            right=max(1, min(self.right, width)),
            bottom=max(1, min(self.bottom, height)),
        )

    def as_slice(self) -> tuple[slice, slice]:
        return slice(self.top, self.bottom), slice(self.left, self.right)

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def is_near_white(
    bgr: np.ndarray,
    *,
    min_channel: int = 245,
    max_chroma: int = 12,
) -> np.ndarray:
    """
    Board white (gaps/outline/AA around rounded cells), not pale page bg.
    Page beige/pink has lower min channel and/or higher chroma.
    """
    if bgr.ndim == 1:
        mn = int(np.min(bgr))
        mx = int(np.max(bgr))
        return mn >= min_channel and (mx - mn) <= max_chroma
    mn = np.min(bgr, axis=2)
    mx = np.max(bgr, axis=2)
    return (mn >= min_channel) & ((mx - mn) <= max_chroma)


def is_near_black(bgr: np.ndarray, *, max_channel: int = 45) -> np.ndarray:
    if bgr.ndim == 1:
        return bool(int(np.max(bgr)) <= max_channel)
    return np.max(bgr, axis=2) <= max_channel


def estimate_page_bg(bgr: np.ndarray, margin: int = 8) -> np.ndarray:
    """Sample pale page background from image corners (mean BGR)."""
    h, w = bgr.shape[:2]
    m = min(margin, h // 10, w // 10)
    patches = [
        bgr[0:m, 0:m],
        bgr[0:m, w - m : w],
        bgr[h - m : h, 0:m],
        bgr[h - m : h, w - m : w],
    ]
    samples = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0).astype(np.float32)
    return samples.mean(axis=0)


def is_page_bg_pixel(
    bgr_pixel: np.ndarray,
    page_bg: np.ndarray,
    *,
    max_dist: float = 22.0,
    min_cell_chroma: int = 20,
) -> bool:
    """
    True if pixel matches page background.
    Board white/AA, black cat marks, and chromatic cell colors are never page bg.
    """
    if is_near_white(bgr_pixel):
        return False
    if is_near_black(bgr_pixel):
        return False
    chroma = int(np.max(bgr_pixel)) - int(np.min(bgr_pixel))
    if chroma >= min_cell_chroma:
        return False
    dist = float(np.linalg.norm(bgr_pixel.astype(np.float32) - page_bg))
    return dist <= max_dist


def is_on_board_pixel(
    bgr_pixel: np.ndarray,
    page_bg: np.ndarray,
    *,
    max_dist: float = 22.0,
) -> bool:
    return not is_page_bg_pixel(bgr_pixel, page_bg, max_dist=max_dist)


def _cross_section_page_bg_frac(
    bgr: np.ndarray,
    x: int,
    y: int,
    dx: int,
    dy: int,
    page_bg: np.ndarray,
    *,
    half: int = 10,
) -> float:
    """Fraction of page-bg pixels on a short segment perpendicular to the ray."""
    h, w = bgr.shape[:2]
    # Perpendicular unit step.
    px, py = -dy, dx
    hits = 0
    total = 0
    for k in range(-half, half + 1):
        xx, yy = x + px * k, y + py * k
        if 0 <= xx < w and 0 <= yy < h:
            total += 1
            if is_page_bg_pixel(bgr[yy, xx], page_bg):
                hits += 1
    return hits / total if total else 1.0


def _ray_until_page_bg(
    bgr: np.ndarray,
    x0: int,
    y0: int,
    dx: int,
    dy: int,
    page_bg: np.ndarray,
    *,
    streak_required: int = 4,
    cross_frac: float = 0.65,
) -> tuple[int, int]:
    """
    Walk from seed until a confirmed page-bg boundary.

    Thin pale AA around a black cat on the ray alone must not stop the scan —
    a real plate edge is page-bg across a perpendicular cross-section too.
    """
    h, w = bgr.shape[:2]
    x, y = x0, y0
    last_on = (x0, y0)
    streak = 0
    while 0 <= x < w and 0 <= y < h:
        if is_page_bg_pixel(bgr[y, x], page_bg):
            streak += 1
            if streak >= streak_required:
                if _cross_section_page_bg_frac(bgr, x, y, dx, dy, page_bg) >= cross_frac:
                    return last_on
                # Cat/AA noise on the ray — keep going.
                streak = 0
        else:
            streak = 0
            last_on = (x, y)
        x += dx
        y += dy
    return last_on


def _edge_onboard_frac(
    bgr: np.ndarray,
    roi: Roi,
    page_bg: np.ndarray,
    *,
    side: str,
    inset: int,
) -> float:
    """Fraction of on-board pixels on a line inset from one ROI edge."""
    h, w = bgr.shape[:2]
    if side == "top":
        y = roi.top + inset
        if not (0 <= y < h):
            return 0.0
        xs = range(roi.left, roi.right)
        vals = [is_on_board_pixel(bgr[y, x], page_bg) for x in xs if 0 <= x < w]
    elif side == "bottom":
        y = roi.bottom - 1 - inset
        if not (0 <= y < h):
            return 0.0
        xs = range(roi.left, roi.right)
        vals = [is_on_board_pixel(bgr[y, x], page_bg) for x in xs if 0 <= x < w]
    elif side == "left":
        x = roi.left + inset
        if not (0 <= x < w):
            return 0.0
        ys = range(roi.top, roi.bottom)
        vals = [is_on_board_pixel(bgr[y, x], page_bg) for y in ys if 0 <= y < h]
    elif side == "right":
        x = roi.right - 1 - inset
        if not (0 <= x < w):
            return 0.0
        ys = range(roi.top, roi.bottom)
        vals = [is_on_board_pixel(bgr[y, x], page_bg) for y in ys if 0 <= y < h]
    else:
        raise ValueError(side)
    return float(sum(vals) / len(vals)) if vals else 0.0


def _edges_ok(
    bgr: np.ndarray,
    roi: Roi,
    page_bg: np.ndarray,
    *,
    insets: tuple[int, ...] = (2, 3),
    min_frac: float = 0.75,
) -> bool:
    """True if every side at each inset is mostly on-board (square plate check)."""
    for side in ("top", "bottom", "left", "right"):
        for inset in insets:
            if _edge_onboard_frac(bgr, roi, page_bg, side=side, inset=inset) < min_frac:
                return False
    return True


def _square_roi_validated(
    bgr: np.ndarray,
    rough: Roi,
    page_bg: np.ndarray,
    *,
    edge_insets: tuple[int, ...] = (2, 3),
) -> Roi:
    """
    Force a square ROI using the longer ray span, then verify / nudge using
    edge samples a couple of pixels inside each side.
    """
    h, w = bgr.shape[:2]
    bw, bh = rough.width, rough.height
    side = max(bw, bh)
    max_shift = max(8, side // 20)

    candidates: list[Roi] = []
    if bw >= bh:
        left, right = rough.left, rough.right
        side = right - left
        # Keep L/R from rays; expand short vertical axis (cat often breaks bottom ray).
        for top_c in (rough.top, rough.bottom - side, (rough.top + rough.bottom - side) // 2):
            for d in range(0, max_shift + 1):
                for top in ((top_c,) if d == 0 else (top_c - d, top_c + d)):
                    candidates.append(Roi(left, top, right, top + side).clamp(w, h))
    else:
        top, bottom = rough.top, rough.bottom
        side = bottom - top
        for left_c in (rough.left, rough.right - side, (rough.left + rough.right - side) // 2):
            for d in range(0, max_shift + 1):
                for left in ((left_c,) if d == 0 else (left_c - d, left_c + d)):
                    candidates.append(Roi(left, top, left + side, bottom).clamp(w, h))

    best: Roi | None = None
    best_score = -1.0
    for cand in candidates:
        if cand.width < side * 0.95 or cand.height < side * 0.95:
            continue
        score = sum(
            _edge_onboard_frac(bgr, cand, page_bg, side=s, inset=ins)
            for s in ("top", "bottom", "left", "right")
            for ins in edge_insets
        )
        ok = _edges_ok(bgr, cand, page_bg, insets=edge_insets)
        if ok and score > best_score:
            return cand
        if score > best_score:
            best_score = score
            best = cand
    if best is not None:
        return best
    # Fallback: axis-aligned square from longer side, clamped.
    if bw >= bh:
        return Roi(rough.left, rough.top, rough.right, rough.top + (rough.right - rough.left)).clamp(w, h)
    return Roi(rough.left, rough.top, rough.left + (rough.bottom - rough.top), rough.bottom).clamp(w, h)


def find_board_roi(
    bgr: np.ndarray,
    *,
    inset: int = 3,
    page_bg: np.ndarray | None = None,
) -> Roi:
    """
    Expand from image center in 4 directions while on white plate / cells;
    stop at page background (pale pink/beige). White gaps and black cats are
    still on-board. Result is refined to a square validated on edge insets.
    """
    h, w = bgr.shape[:2]
    if page_bg is None:
        page_bg = estimate_page_bg(bgr)

    cx, cy = w // 2, h // 2
    # If center is somehow page bg, search nearby for an on-board pixel.
    if not is_on_board_pixel(bgr[cy, cx], page_bg):
        found = False
        for radius in range(1, min(h, w) // 4):
            for yy in range(cy - radius, cy + radius + 1):
                for xx in range(cx - radius, cx + radius + 1):
                    if 0 <= xx < w and 0 <= yy < h and is_on_board_pixel(bgr[yy, xx], page_bg):
                        cx, cy = xx, yy
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            raise ValueError("Could not find board plate from image center")

    left_x, _ = _ray_until_page_bg(bgr, cx, cy, -1, 0, page_bg)
    right_x, _ = _ray_until_page_bg(bgr, cx, cy, 1, 0, page_bg)
    _, top_y = _ray_until_page_bg(bgr, cx, cy, 0, -1, page_bg)
    _, bottom_y = _ray_until_page_bg(bgr, cx, cy, 0, 1, page_bg)

    rough = Roi(
        left=left_x + inset,
        top=top_y + inset,
        right=right_x - inset + 1,
        bottom=bottom_y - inset + 1,
    ).clamp(w, h)

    if rough.width < 20 or rough.height < 20:
        raise ValueError(f"Detected ROI too small: {rough}")

    roi = _square_roi_validated(bgr, rough, page_bg, edge_insets=(2, 3))
    if roi.width < 20 or roi.height < 20:
        raise ValueError(f"Detected ROI too small after square fit: {roi}")
    return roi


def parse_roi_arg(text: str) -> Roi:
    parts = [int(p.strip()) for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be L,T,R,B")
    return Roi(left=parts[0], top=parts[1], right=parts[2], bottom=parts[3])
