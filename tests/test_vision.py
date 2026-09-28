import cv2
import numpy as np

from meoudoku.board import CellState
from meoudoku.vision import CvBoardParser, find_board_roi
from meoudoku.vision.cv_parser import estimate_n_by_edges
from meoudoku.vision.roi import is_near_white


# Distinct non-white/non-black BGR colors
PALETTE = [
    (40, 180, 60),  # green
    (200, 100, 40),  # blue-ish
    (40, 80, 220),  # red
    (40, 200, 220),  # yellow
    (180, 80, 180),  # purple
    (30, 160, 220),  # orange
]


def _draw_rounded_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], radius: int = 8) -> None:
    cv2.rectangle(img, (x0 + radius, y0), (x1 - radius, y1), color, -1)
    cv2.rectangle(img, (x0, y0 + radius), (x1, y1 - radius), color, -1)
    cv2.circle(img, (x0 + radius, y0 + radius), radius, color, -1)
    cv2.circle(img, (x1 - radius, y0 + radius), radius, color, -1)
    cv2.circle(img, (x0 + radius, y1 - radius), radius, color, -1)
    cv2.circle(img, (x1 - radius, y1 - radius), radius, color, -1)


def make_synthetic_board(
    n: int = 5,
    *,
    cell: int = 40,
    gap: int = 4,
    pad: int = 16,
    page_margin: int = 40,
    marks: dict[tuple[int, int], str] | None = None,
) -> tuple[np.ndarray, list[list[int]]]:
    """Return full screenshot BGR and expected color id grid (by appearance order)."""
    marks = marks or {}
    plate = n * cell + (n + 1) * gap
    white = (255, 255, 255)
    page = (240, 242, 255)  # pale pink-ish in BGR
    board_w = plate + 2 * pad
    board_h = plate + 2 * pad
    img = np.full((board_h + 2 * page_margin, board_w + 2 * page_margin, 3), page, dtype=np.uint8)
    ox, oy = page_margin, page_margin
    img[oy : oy + board_h, ox : ox + board_w] = white

    # Assign each cell a color id 0..n-1 in a simple pattern: color = (r + 2*c) % n
    # Ensure all n colors appear.
    expected = [[(r + 2 * c) % n for c in range(n)] for r in range(n)]
    # Fix if somehow missing (shouldn't for n>=1)
    seen = {expected[r][c] for r in range(n) for c in range(n)}
    assert seen == set(range(n))

    for r in range(n):
        for c in range(n):
            cid = expected[r][c]
            color = PALETTE[cid % len(PALETTE)]
            x0 = ox + pad + gap + c * (cell + gap)
            y0 = oy + pad + gap + r * (cell + gap)
            x1, y1 = x0 + cell, y0 + cell
            _draw_rounded_rect(img, x0, y0, x1, y1, color)
            mark = marks.get((r, c))
            if mark == "C":
                cv2.rectangle(img, (x0 + 10, y0 + 10), (x1 - 10, y1 - 10), (0, 0, 0), -1)
            elif mark == "X":
                # Thick diagonal strokes similar to the game art.
                cv2.line(img, (x0 + 6, y0 + 6), (x1 - 6, y1 - 6), (255, 255, 255), 7)
                cv2.line(img, (x1 - 6, y0 + 6), (x0 + 6, y1 - 6), (255, 255, 255), 7)

    return img, expected


def test_find_board_roi_stops_at_page_bg_not_white_gaps():
    img, _ = make_synthetic_board(5)
    roi = find_board_roi(img, inset=2)
    crop = img[roi.top : roi.bottom, roi.left : roi.right]
    # Crop should contain white (gaps) and colored cells, not mostly page bg
    white_frac = float(np.mean(is_near_white(crop)))
    assert 0.05 < white_frac < 0.9
    assert crop.shape[0] > 100 and crop.shape[1] > 100


def test_cv_parser_colors_and_marks():
    img, expected = make_synthetic_board(5, marks={(0, 0): "C", (1, 2): "X"})
    board = CvBoardParser(n=5).parse(img)
    assert board.n == 5
    # Color ids may be remapped by first-appearance vs our formula; check partition equality
    def signature(grid: list[list[int]]) -> set[frozenset[tuple[int, int]]]:
        groups: dict[int, set[tuple[int, int]]] = {}
        for r in range(5):
            for c in range(5):
                groups.setdefault(grid[r][c], set()).add((r, c))
        return {frozenset(v) for v in groups.values()}

    assert signature(board.colors) == signature(expected)
    assert board.get(0, 0) is CellState.CAT
    assert board.get(1, 2) is CellState.X
    assert board.get(2, 2) is CellState.EMPTY


def test_cv_parser_autodetect_n():
    img, _ = make_synthetic_board(6)
    board = CvBoardParser(n_min=5, n_max=8).parse(img)
    assert board.n == 6


def test_estimate_n_by_edges_matches_on_all_sides():
    img, _ = make_synthetic_board(8)
    roi = find_board_roi(img, inset=2)
    plate = img[roi.top : roi.bottom, roi.left : roi.right]
    est = estimate_n_by_edges(plate, n_min=5, n_max=12)
    assert est is not None
    assert est.n == 8
    assert est.confident
    assert est.source == "edges"


def _wrap_plate(plate: np.ndarray, margin: int = 40) -> np.ndarray:
    page = (240, 242, 255)
    full = np.full((plate.shape[0] + 2 * margin, plate.shape[1] + 2 * margin, 3), page, dtype=np.uint8)
    full[margin : margin + plate.shape[0], margin : margin + plate.shape[1]] = plate
    return full


def test_real_board_empty_10_autodetect_n():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "fixtures" / "screens" / "board_empty_10.png"
    assert path.exists(), "missing fixtures/screens/board_empty_10.png"
    img = cv2.imread(str(path))
    assert img is not None
    board = CvBoardParser(n_min=5, n_max=20).parse(img)
    assert board.n == 10


def test_real_board_plate_fixture():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "fixtures" / "screens" / "board_plate.png"
    if not path.exists():
        return
    plate = cv2.imread(str(path))
    assert plate is not None
    board = CvBoardParser(n_min=5, n_max=12).parse(_wrap_plate(plate, 60))
    assert board.n == 10
    assert all(board.get(r, c) is CellState.EMPTY for r in range(board.n) for c in range(board.n))


def test_real_board_with_x_fixture():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "fixtures" / "screens" / "board_with_x.png"
    if not path.exists():
        return
    plate = cv2.imread(str(path))
    assert plate is not None
    board = CvBoardParser(n_min=5, n_max=12).parse(_wrap_plate(plate))
    assert board.n == 10
    expected = {(0, 0), (1, 0), (2, 0), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4), (7, 4), (8, 4), (9, 4)}
    xs = {(r, c) for r in range(10) for c in range(10) if board.get(r, c) is CellState.X}
    assert xs == expected


def test_real_board_orange_x_and_cat():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "fixtures" / "screens" / "board_orange_x.png"
    if not path.exists():
        return
    plate = cv2.imread(str(path))
    assert plate is not None
    board = CvBoardParser(n_min=5, n_max=12).parse(_wrap_plate(plate))
    assert board.n == 10
    assert board.get(3, 5) is CellState.CAT
    assert board.get(9, 9) is CellState.X  # non-white (orange) cross
    # Cat cell must share color with neighboring medium-blue cell, not green.
    assert board.color(3, 5) == board.color(3, 4)


