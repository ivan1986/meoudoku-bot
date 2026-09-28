"""Screen wait helpers: sample patches, orange button, dark overlay checks."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from meoudoku.device.adb import AdbDevice


class ScreenWaitError(RuntimeError):
    """Timed out waiting for a screen condition."""


# OpenCV HSV: H in [0,180]. Meowdoku level button is bright orange.
ORANGE_H_MIN = 5
ORANGE_H_MAX = 25
ORANGE_S_MIN = 100
ORANGE_V_MIN = 150

# Start-banner / post-win darkness (0..255 gray mean).
# Banner overlay is brownish ~#2B1D1D but semi-transparent over cells → mean ~80-110.
# Plain board cells at center are typically >160. Threshold must sit between them.
DARK_BG_MAX_LUMA = 120
DARK_FRAME_MAX_LUMA = 40
# Leaderboard post-win: nearly-black bands at top + bottom (center stays colorful).
POST_WIN_TOP_FRAC = (0.5, 0.12)
POST_WIN_BOTTOM_FRAC = (0.5, 0.92)
POST_WIN_BAND_MAX_LUMA = 70


def decode_bgr(image: bytes | np.ndarray) -> np.ndarray:
    """Decode PNG/JPG bytes or pass through a BGR ndarray."""
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image array must be HxWx3 BGR")
        return image
    arr = np.frombuffer(image, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode image bytes")
    return bgr


def frac_to_xy(width: int, height: int, fx: float, fy: float) -> tuple[int, int]:
    """Map normalized fractions in [0,1] to pixel coordinates."""
    if width < 1 or height < 1:
        raise ValueError("width/height must be positive")
    x = int(round(fx * (width - 1)))
    y = int(round(fy * (height - 1)))
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def parse_frac_pair(text: str) -> tuple[float, float]:
    """Parse 'FX,FY' fractions."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected FX,FY got {text!r}")
    fx, fy = float(parts[0]), float(parts[1])
    if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
        raise ValueError(f"fractions must be in [0,1]: {text!r}")
    return fx, fy


def parse_xy_pair(text: str) -> tuple[int, int]:
    """Parse 'X,Y' pixel coordinates."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected X,Y got {text!r}")
    return int(parts[0]), int(parts[1])


def sample_patch_bgr(
    bgr: np.ndarray,
    x: int,
    y: int,
    *,
    size: int = 11,
) -> np.ndarray:
    """Return a (size x size) BGR patch clipped to image bounds, centered on (x,y)."""
    h, w = bgr.shape[:2]
    half = max(0, size // 2)
    x0 = max(0, x - half)
    y0 = max(0, y - half)
    x1 = min(w, x + half + 1)
    y1 = min(h, y + half + 1)
    patch = bgr[y0:y1, x0:x1]
    if patch.size == 0:
        raise ValueError(f"empty patch at ({x},{y})")
    return patch


def patch_mean_bgr(patch: np.ndarray) -> tuple[float, float, float]:
    mean = patch.reshape(-1, 3).mean(axis=0)
    return float(mean[0]), float(mean[1]), float(mean[2])


def patch_mean_hsv(patch: np.ndarray) -> tuple[float, float, float]:
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mean = hsv.reshape(-1, 3).mean(axis=0)
    return float(mean[0]), float(mean[1]), float(mean[2])


def patch_mean_luma(patch: np.ndarray) -> float:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def is_orange(
    hsv_mean: tuple[float, float, float] | np.ndarray,
    *,
    h_min: float = ORANGE_H_MIN,
    h_max: float = ORANGE_H_MAX,
    s_min: float = ORANGE_S_MIN,
    v_min: float = ORANGE_V_MIN,
) -> bool:
    """True if mean HSV looks like the bright orange level button."""
    h, s, v = (float(hsv_mean[0]), float(hsv_mean[1]), float(hsv_mean[2]))
    return h_min <= h <= h_max and s >= s_min and v >= v_min


def is_orange_at(
    bgr: np.ndarray,
    x: int,
    y: int,
    *,
    size: int = 11,
) -> bool:
    patch = sample_patch_bgr(bgr, x, y, size=size)
    return is_orange(patch_mean_hsv(patch))


def is_dark_bg(
    bgr: np.ndarray,
    *,
    fx: float = 0.5,
    fy: float = 0.48,
    size: int = 21,
    max_luma: float = DARK_BG_MAX_LUMA,
) -> bool:
    """True if the patch at (fx,fy) is dark (start-banner overlay)."""
    h, w = bgr.shape[:2]
    x, y = frac_to_xy(w, h, fx, fy)
    return patch_mean_luma(sample_patch_bgr(bgr, x, y, size=size)) <= max_luma


def is_dark(
    bgr: np.ndarray,
    *,
    max_mean: float = DARK_FRAME_MAX_LUMA,
) -> bool:
    """True if the whole frame mean luminance is below threshold (post-win fade)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) <= max_mean


def is_post_win_dark(
    bgr: np.ndarray,
    *,
    top_frac: tuple[float, float] = POST_WIN_TOP_FRAC,
    bottom_frac: tuple[float, float] = POST_WIN_BOTTOM_FRAC,
    max_luma: float = POST_WIN_BAND_MAX_LUMA,
    size: int = 21,
) -> bool:
    """
    True if both the top and bottom center patches are dark.

    Matches the leaderboard / post-win overlay (dimmed chrome at top and bottom
    with "Нажмите, чтобы продолжить"), not a full-frame black fade.
    """
    return is_dark_bg(
        bgr, fx=top_frac[0], fy=top_frac[1], size=size, max_luma=max_luma
    ) and is_dark_bg(
        bgr, fx=bottom_frac[0], fy=bottom_frac[1], size=size, max_luma=max_luma
    )


def post_win_band_lumas(
    bgr: np.ndarray,
    *,
    top_frac: tuple[float, float] = POST_WIN_TOP_FRAC,
    bottom_frac: tuple[float, float] = POST_WIN_BOTTOM_FRAC,
    size: int = 21,
) -> tuple[float, float]:
    """Return (top_luma, bottom_luma) for logging."""
    h, w = bgr.shape[:2]
    tx, ty = frac_to_xy(w, h, top_frac[0], top_frac[1])
    bx, by = frac_to_xy(w, h, bottom_frac[0], bottom_frac[1])
    top = patch_mean_luma(sample_patch_bgr(bgr, tx, ty, size=size))
    bot = patch_mean_luma(sample_patch_bgr(bgr, bx, by, size=size))
    return top, bot


def wait_until(
    device: AdbDevice,
    predicate: Callable[[np.ndarray], bool],
    *,
    timeout_s: float,
    poll_s: float = 0.35,
    hold_s: float = 0.0,
) -> np.ndarray:
    """
    Poll screenshots until predicate(bgr) is true.

    If hold_s > 0, require the predicate to stay true for hold_s seconds.
    Returns the last matching BGR frame. Raises ScreenWaitError on timeout.
    """
    deadline = time.monotonic() + max(0.0, timeout_s)
    held_since: float | None = None
    last: np.ndarray | None = None
    while True:
        png = device.screenshot()
        bgr = decode_bgr(png)
        last = bgr
        if predicate(bgr):
            now = time.monotonic()
            if hold_s <= 0:
                return bgr
            if held_since is None:
                held_since = now
            elif now - held_since >= hold_s:
                return bgr
        else:
            held_since = None
        if time.monotonic() >= deadline:
            raise ScreenWaitError(f"timeout after {timeout_s:.1f}s waiting for screen condition")
        time.sleep(max(0.01, poll_s))
