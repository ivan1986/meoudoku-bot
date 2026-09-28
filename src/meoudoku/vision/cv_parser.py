"""OpenCV board parser: ROI → NxN colors + CAT/X state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from meoudoku.board import Board, BoardError, CellState
from meoudoku.vision.roi import Roi, find_board_roi, is_near_black, is_near_white


class VisionError(ValueError):
    """Failed to parse board from image."""


def decode_image(image: bytes | Path | str | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise VisionError("Image array must be HxWx3 BGR")
        return image
    if isinstance(image, (str, Path)):
        data = Path(image).read_bytes()
    else:
        data = image
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise VisionError("Could not decode image")
    return bgr


@dataclass(frozen=True)
class Grid:
    """Pixel edges for an NxN board: xs/ys length N+1."""

    n: int
    xs: tuple[int, ...]
    ys: tuple[int, ...]

    def cell_rect(self, r: int, c: int) -> tuple[int, int, int, int]:
        return self.xs[c], self.ys[r], self.xs[c + 1], self.ys[r + 1]


def _gap_runs(
    white_frac_1d: np.ndarray,
    *,
    thr: float = 0.50,
    min_run: int = 1,
) -> list[tuple[int, int]]:
    on = white_frac_1d > thr
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, value in enumerate(on):
        if value and not in_run:
            in_run = True
            start = i
        elif not value and in_run:
            in_run = False
            if i - start >= min_run:
                runs.append((start, i))
    if in_run and len(on) - start >= min_run:
        runs.append((start, len(on)))
    return runs


def _gaps_for_n(
    col_frac: np.ndarray,
    row_frac: np.ndarray,
    n: int,
    *,
    min_run: int = 1,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Find a white-fraction threshold that yields exactly N+1 gap bands on both axes."""
    for thr in (0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45):
        col_gaps = _gap_runs(col_frac, thr=thr, min_run=min_run)
        row_gaps = _gap_runs(row_frac, thr=thr, min_run=min_run)
        if len(col_gaps) == n + 1 and len(row_gaps) == n + 1:
            return col_gaps, row_gaps
    return None


def build_grid(plate: np.ndarray, n: int) -> Grid:
    """Build cell bounds from white gap bands; fall back to uniform split."""
    h, w = plate.shape[:2]
    min_run = max(2, min(h, w) // 200)
    white = is_near_white(plate)
    col_frac = white.mean(axis=0)
    row_frac = white.mean(axis=1)
    found = _gaps_for_n(col_frac, row_frac, n, min_run=min_run)
    if found is not None:
        col_gaps, row_gaps = found
    else:
        col_gaps, row_gaps = [], []

    def edges_from_gaps(gaps: list[tuple[int, int]], count: int, size: int) -> tuple[int, ...]:
        if len(gaps) == count + 1:
            # Cell i lies between gap i and gap i+1.
            return tuple([gaps[i][1] for i in range(count)] + [gaps[count][0]])
        return tuple(int(round(i * size / count)) for i in range(count + 1))

    xs = edges_from_gaps(col_gaps, n, w)
    ys = edges_from_gaps(row_gaps, n, h)
    # Ensure strictly increasing edges.
    xs_l = list(xs)
    ys_l = list(ys)
    for i in range(1, n + 1):
        xs_l[i] = max(xs_l[i], xs_l[i - 1] + 2)
        ys_l[i] = max(ys_l[i], ys_l[i - 1] + 2)
    xs_l[-1] = min(xs_l[-1], w)
    ys_l[-1] = min(ys_l[-1], h)
    return Grid(n=n, xs=tuple(xs_l), ys=tuple(ys_l))


def _cell_rect(
    plate: np.ndarray,
    n: int,
    r: int,
    c: int,
    grid: Grid | None = None,
) -> tuple[int, int, int, int]:
    if grid is not None:
        return grid.cell_rect(r, c)
    h, w = plate.shape[:2]
    x0 = int(round(c * w / n))
    x1 = int(round((c + 1) * w / n))
    y0 = int(round(r * h / n))
    y1 = int(round((r + 1) * h / n))
    return x0, y0, x1, y1


def _inset_crop(
    plate: np.ndarray,
    n: int,
    r: int,
    c: int,
    inset_frac: float = 0.28,
    grid: Grid | None = None,
) -> np.ndarray:
    x0, y0, x1, y1 = _cell_rect(plate, n, r, c, grid=grid)
    cw, ch = x1 - x0, y1 - y0
    ix = max(1, int(cw * inset_frac))
    iy = max(1, int(ch * inset_frac))
    return plate[y0 + iy : y1 - iy, x0 + ix : x1 - ix]


def _median_fill_bgr(patch: np.ndarray) -> np.ndarray | None:
    """Median BGR of non-white/non-black pixels in a small patch."""
    if patch.size == 0:
        return None
    white = is_near_white(patch)
    black = is_near_black(patch)
    mask = ~(white | black)
    if not np.any(mask):
        return None
    return np.median(patch[mask], axis=0).astype(np.float32)


def _sample_cell_color(
    plate: np.ndarray,
    n: int,
    r: int,
    c: int,
    *,
    grid: Grid | None = None,
    edge_inset: int = 2,
    patch: int = 4,
) -> np.ndarray | None:
    """
    Sample region fill from the side of the colored tile (not the center).

    Cat/X marks sit in the middle; use a vertical strip a couple of pixels inside
    the left/right edge (full height minus corner padding) and take the median of
    non-white/non-black pixels so a cat touching mid-edge still leaves fill above/below.
    """
    x0, y0, x1, y1 = _cell_rect(plate, n, r, c, grid=grid)
    slot = plate[y0:y1, x0:x1]
    if slot.size == 0:
        return None

    sh, sw = slot.shape[:2]
    if sw < edge_inset + patch + 1 or sh < 8:
        return _median_fill_bgr(_inset_crop(plate, n, r, c, inset_frac=0.34, grid=grid))

    # Skip rounded-corner / gap AA near top and bottom of the tile.
    y_pad = max(2, sh // 8)
    y_a, y_b = y_pad, sh - y_pad
    if y_b <= y_a:
        y_a, y_b = 0, sh

    strips = [
        (edge_inset, edge_inset + patch),
        (sw - edge_inset - patch, sw - edge_inset),
    ]
    best: np.ndarray | None = None
    best_count = -1
    for xa, xb in strips:
        xa = max(0, xa)
        xb = min(sw, xb)
        if xb <= xa:
            continue
        crop = slot[y_a:y_b, xa:xb]
        color = _median_fill_bgr(crop)
        if color is None:
            continue
        count = int(np.count_nonzero(~(is_near_white(crop) | is_near_black(crop))))
        if count > best_count:
            best = color
            best_count = count

    if best is not None:
        return best
    fill = ~(is_near_white(slot) | is_near_black(slot))
    if np.any(fill):
        return np.median(slot[fill], axis=0).astype(np.float32)
    return None


def _detect_state(
    cell: np.ndarray,
    fill_bgr: np.ndarray | None = None,
    *,
    black_frac: float = 0.05,
    white_frac: float = 0.20,
    mark_dist: float = 50.0,
    mark_frac: float = 0.35,
) -> CellState:
    """
    CAT: near-black pixels present.
    X: near-white in the center, or a colored mark far from the side-sampled fill
    (game sometimes draws non-white crosses).
    """
    if cell.size == 0:
        return CellState.EMPTY
    total = cell.shape[0] * cell.shape[1]
    black_n = int(np.count_nonzero(is_near_black(cell)))
    white_n = int(np.count_nonzero(is_near_white(cell, min_channel=252, max_chroma=6)))
    if black_n / total >= black_frac:
        return CellState.CAT
    if white_n / total >= white_frac:
        return CellState.X
    if fill_bgr is not None:
        dist = np.linalg.norm(cell.astype(np.float32) - fill_bgr.astype(np.float32), axis=2)
        if float(np.mean(dist > mark_dist)) >= mark_frac:
            return CellState.X
    return CellState.EMPTY


def _cluster_score(samples: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    # Lower intra-cluster variance is better; require all clusters used.
    if len(centers) == 0:
        return float("inf")
    used = len(np.unique(labels))
    if used != len(centers):
        return float("inf")
    var = 0.0
    for i in range(len(centers)):
        pts = samples[labels.ravel() == i]
        if len(pts) == 0:
            return float("inf")
        var += float(np.mean(np.sum((pts - centers[i]) ** 2, axis=1)))
    # Prefer balanced cluster sizes slightly
    counts = np.bincount(labels.ravel(), minlength=len(centers)).astype(np.float32)
    balance = float(np.std(counts) / (np.mean(counts) + 1e-6))
    return var + 0.15 * balance * var


def _kmeans_labels(samples: np.ndarray, k: int, attempts: int = 5) -> tuple[np.ndarray, np.ndarray, float]:
    samples32 = samples.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    compactness, labels, centers = cv2.kmeans(
        samples32,
        k,
        None,
        criteria,
        attempts,
        cv2.KMEANS_PP_CENTERS,
    )
    score = _cluster_score(samples32, labels, centers) + float(compactness) * 1e-6
    return labels.ravel(), centers, score


@dataclass(frozen=True)
class GapNEstimate:
    n: int
    votes: dict[int, int]
    confident: bool
    source: str = "gaps"


def _content_bbox(
    plate: np.ndarray,
    *,
    nonwhite_thr: float = 0.12,
) -> tuple[int, int, int, int]:
    """Tight bbox around colored tiles inside the white plate (left, top, right, bottom inclusive)."""
    white = is_near_white(plate)
    row_nw = (~white).mean(axis=1)
    col_nw = (~white).mean(axis=0)
    h, w = plate.shape[:2]

    def first(arr: np.ndarray) -> int:
        for i, value in enumerate(arr):
            if float(value) > nonwhite_thr:
                return i
        raise VisionError("No colored tiles found inside plate")

    top = first(row_nw)
    bottom = h - 1 - first(row_nw[::-1])
    left = first(col_nw)
    right = w - 1 - first(col_nw[::-1])
    if right - left < 20 or bottom - top < 20:
        raise VisionError("Colored tile bbox too small")
    return left, top, right, bottom


@dataclass(frozen=True)
class ColorRun:
    start: int
    end: int  # exclusive
    mean_bgr: np.ndarray  # float32[3]


def _homogeneous_color_run_spans(
    line_bgr: np.ndarray,
    *,
    min_len: int | None = None,
    max_delta: float = 35.0,
) -> list[ColorRun]:
    """Split a 1-D BGR scanline into homogeneous non-white runs (cell sides)."""
    if line_bgr.ndim != 2 or line_bgr.shape[1] != 3:
        raise VisionError("Expected Nx3 BGR scanline")
    n = int(line_bgr.shape[0])
    if min_len is None:
        min_len = max(6, n // 80)
    mn = line_bgr.min(axis=1)
    mx = line_bgr.max(axis=1)
    is_w = (mn >= 245) & ((mx - mn) <= 12)
    runs: list[ColorRun] = []
    i = 0
    while i < n:
        if is_w[i]:
            i += 1
            continue
        start = i
        ref = line_bgr[i].astype(np.float32)
        i += 1
        while i < n and not is_w[i]:
            if float(np.linalg.norm(line_bgr[i].astype(np.float32) - ref)) > max_delta:
                break
            ref = 0.9 * ref + 0.1 * line_bgr[i].astype(np.float32)
            i += 1
        if i - start >= min_len:
            mean = line_bgr[start:i].mean(axis=0).astype(np.float32)
            runs.append(ColorRun(start=start, end=i, mean_bgr=mean))
    return runs


def _homogeneous_color_runs(
    line_bgr: np.ndarray,
    *,
    min_len: int | None = None,
    max_delta: float = 35.0,
) -> list[np.ndarray]:
    """Mean BGR of each homogeneous non-white run on a scanline."""
    return [
        run.mean_bgr
        for run in _homogeneous_color_run_spans(line_bgr, min_len=min_len, max_delta=max_delta)
    ]


def _colors_match(
    a: list[np.ndarray],
    b: list[np.ndarray],
    *,
    max_dist: float = 30.0,
) -> bool:
    if len(a) != len(b):
        return False
    for ca, cb in zip(a, b, strict=True):
        if float(np.linalg.norm(ca.astype(np.float32) - cb.astype(np.float32))) > max_dist:
            return False
    return True


def estimate_n_by_edges(
    plate: np.ndarray,
    *,
    n_min: int = 5,
    n_max: int = 20,
    insets: tuple[int, ...] = (2, 3),
) -> GapNEstimate | None:
    """
    Count N from homogeneous colored tile sides.

    For each of the four content edges, sample scanlines at insets (default 2 and 3 px).
    Run count and mean colors must match between insets; all four sides must agree on N.
    """
    try:
        left, top, right, bottom = _content_bbox(plate)
    except VisionError:
        return None

    def side_runs(side: str, inset: int) -> list[np.ndarray]:
        if side == "top":
            return _homogeneous_color_runs(plate[top + inset, left : right + 1])
        if side == "bottom":
            return _homogeneous_color_runs(plate[bottom - inset, left : right + 1])
        if side == "left":
            return _homogeneous_color_runs(plate[top : bottom + 1, left + inset])
        if side == "right":
            return _homogeneous_color_runs(plate[top : bottom + 1, right - inset])
        raise VisionError(f"Unknown side {side}")

    side_ns: list[int] = []
    for side in ("top", "bottom", "left", "right"):
        by_inset = [side_runs(side, inset) for inset in insets]
        if any(not (n_min <= len(runs) <= n_max) for runs in by_inset):
            return None
        # Insets must agree on count and colors (solid cell side, not AA).
        for other in by_inset[1:]:
            if not _colors_match(by_inset[0], other):
                return None
        side_ns.append(len(by_inset[0]))

    if len(set(side_ns)) != 1:
        return None
    n = side_ns[0]
    return GapNEstimate(n=n, votes={n: 4}, confident=True, source="edges")


def estimate_n_by_gaps(
    plate: np.ndarray,
    *,
    n_min: int = 5,
    n_max: int = 20,
) -> GapNEstimate | None:
    """
    Estimate N from white gap lines: N cells ⇒ N+1 gap bands on each axis.

    Real gaps are nearly fully white; X marks only partially raise a column's
    white fraction. Vote across thresholds and keep N where row/col agree —
    preferring values that stay stable at higher thresholds.
    """
    from collections import Counter

    h, w = plate.shape[:2]
    min_run = max(2, min(h, w) // 200)
    white = is_near_white(plate)
    col_frac = white.mean(axis=0)
    row_frac = white.mean(axis=1)
    votes: Counter[int] = Counter()
    # High → low: true gaps survive; false bands from X marks drop out first.
    for thr in (0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45):
        col_n = len(_gap_runs(col_frac, thr=thr, min_run=min_run)) - 1
        row_n = len(_gap_runs(row_frac, thr=thr, min_run=min_run)) - 1
        if col_n == row_n and n_min <= col_n <= n_max:
            # Weight higher thresholds more (more reliable).
            votes[col_n] += 3 if thr >= 0.70 else (2 if thr >= 0.60 else 1)
    if not votes:
        return None
    # Tie-break: more votes, then prefer smaller N (avoid overcount from AA).
    ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    best_n, best_v = ranked[0]
    second_v = ranked[1][1] if len(ranked) > 1 else 0
    confident = best_v >= max(2 * second_v, second_v + 3) if second_v else True
    return GapNEstimate(n=best_n, votes=dict(votes), confident=confident, source="gaps")


def estimate_n(
    plate: np.ndarray,
    *,
    n_min: int = 5,
    n_max: int = 20,
) -> GapNEstimate | None:
    """Prefer edge-of-tile counting; fall back to white-gap voting."""
    edged = estimate_n_by_edges(plate, n_min=n_min, n_max=n_max)
    if edged is not None:
        return edged
    return estimate_n_by_gaps(plate, n_min=n_min, n_max=n_max)


def choose_n_and_colors(
    plate: np.ndarray,
    *,
    n: int | None = None,
    n_min: int = 5,
    n_max: int = 20,
) -> tuple[int, list[list[int]]]:
    if n is None:
        guessed = estimate_n(plate, n_min=n_min, n_max=n_max)
        if guessed is not None and guessed.confident:
            # Do not let raw KMeans compactness prefer N+1 over a solid gap/edge vote.
            candidates = [guessed.n]
        elif guessed is not None:
            candidates = [guessed.n]
            for dn in (-1, 1):
                alt = guessed.n + dn
                if n_min <= alt <= n_max:
                    candidates.append(alt)
        else:
            candidates = list(range(n_min, n_max + 1))
    else:
        candidates = [n]
    best: tuple[float, int, list[list[int]]] | None = None

    for cand in candidates:
        grid = build_grid(plate, cand)
        samples: list[np.ndarray] = []
        coords: list[tuple[int, int]] = []
        ok = True
        for r in range(cand):
            for c in range(cand):
                color = _sample_cell_color(plate, cand, r, c, grid=grid)
                if color is None:
                    ok = False
                    break
                samples.append(color)
                coords.append((r, c))
            if not ok:
                break
        if not ok or len(samples) != cand * cand:
            continue

        arr = np.stack(samples, axis=0)
        try:
            labels, _centers, score = _kmeans_labels(arr, cand)
        except cv2.error:
            continue
        if not np.isfinite(score):
            continue

        # Remap labels to 0..N-1 by first appearance order (row-major).
        remap: dict[int, int] = {}
        colors = [[0] * cand for _ in range(cand)]
        next_id = 0
        for (r, c), lab in zip(coords, labels, strict=True):
            lab_i = int(lab)
            if lab_i not in remap:
                remap[lab_i] = next_id
                next_id += 1
            colors[r][c] = remap[lab_i]
        if next_id != cand:
            continue

        # Normalize: raw compactness falls with more clusters.
        norm_score = float(score) / float(cand * cand)
        if best is None or norm_score < best[0]:
            best = (norm_score, cand, colors)

    if best is None:
        raise VisionError("Failed to detect board size N / colors")
    return best[1], best[2]


def parse_states(plate: np.ndarray, n: int, grid: Grid | None = None) -> list[list[CellState]]:
    if grid is None:
        grid = build_grid(plate, n)
    state: list[list[CellState]] = []
    for r in range(n):
        row: list[CellState] = []
        for c in range(n):
            # Stay in cell center to avoid white gaps between rounded tiles.
            cell = _inset_crop(plate, n, r, c, inset_frac=0.34, grid=grid)
            fill = _sample_cell_color(plate, n, r, c, grid=grid)
            row.append(_detect_state(cell, fill))
        state.append(row)
    return state


@dataclass(frozen=True)
class ParseResult:
    """Parsed board plus screen geometry for taps."""

    board: Board
    roi: Roi
    grid: Grid

    def cell_center(self, r: int, c: int) -> tuple[int, int]:
        """Absolute screenshot/screen coordinates of cell center."""
        x0, y0, x1, y1 = self.grid.cell_rect(r, c)
        return self.roi.left + (x0 + x1) // 2, self.roi.top + (y0 + y1) // 2


class CvBoardParser:
    def __init__(
        self,
        *,
        n: int | None = None,
        roi: Roi | None = None,
        n_min: int = 5,
        n_max: int = 20,
        skip_roi: bool = False,
    ) -> None:
        self.n = n
        self.roi = roi
        self.n_min = n_min
        self.n_max = n_max
        self.skip_roi = skip_roi

    def parse(self, image: bytes | Path | str | np.ndarray) -> Board:
        return self.parse_ex(image).board

    def parse_ex(self, image: bytes | Path | str | np.ndarray) -> ParseResult:
        bgr = decode_image(image)
        if self.skip_roi:
            plate = bgr
            roi = Roi(0, 0, bgr.shape[1], bgr.shape[0])
        elif self.roi is not None:
            roi = self.roi
            ys, xs = roi.as_slice()
            plate = bgr[ys, xs]
        else:
            roi = find_board_roi(bgr)
            ys, xs = roi.as_slice()
            plate = bgr[ys, xs]

        n, colors = choose_n_and_colors(plate, n=self.n, n_min=self.n_min, n_max=self.n_max)
        grid = build_grid(plate, n)
        state = parse_states(plate, n, grid=grid)
        try:
            board = Board(colors, state)
        except BoardError as exc:
            raise VisionError(str(exc)) from exc
        return ParseResult(board=board, roi=roi, grid=grid)
