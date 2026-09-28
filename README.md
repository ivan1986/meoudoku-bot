# meoudoku

Solver for the Android puzzle [Meowdoku](https://play.google.com/store/apps/details?id=com.oakever.meowdoku) (1 cat per row, column, color; cats cannot touch, including diagonally).

Heuristic-stack solver + screen reading via ADB/OpenCV.

## Install

```bash
cd /home/ivan/projects/meoudoku
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires `adb` on PATH for device capture.

## Board format

```text
COLORS
a a b b
a c c b
d c c b
d d d b
STATE
. . . .
. . . .
. . . .
. . . .
```

`STATE` cells: `.` empty, `C` cat, `X` blocked. `STATE` may be omitted (all empty).

## Solve from text

```bash
python -m meoudoku.cli solve fixtures/level_tiny.txt
# one heuristic pass (prints name + cats/xs, updates board file):
python -m meoudoku.cli step fixtures/level_tiny.txt
python -m meoudoku.cli step fixtures/level_tiny.txt --no-write
# one pass from phone (taps: 1x=X, 2x=CAT):
python -m meoudoku.cli step --adb
# dry-run / save file instead of tapping:
python -m meoudoku.cli step --adb --no-write
python -m meoudoku.cli step --adb -o /tmp/board.txt
# full solve loop (re-read each round; N must stay the same):
python -m meoudoku.cli play
# launch app → wait orange level button → tap → sleep → wait not-dark → play → wait dark → force-stop:
python -m meoudoku.cli full
# repeat sessions until Ctrl+C:
python -m meoudoku.cli full --repeat
pytest
```

### Full session (`full`)

Starts Meowdoku on the phone, waits for the orange **Уровень N** button at a fixed screen fraction (`--tap-frac 0.5,0.72`), taps it, sleeps `--banner-delay` (default 5s), then runs `play`. After solve, waits for post-win dark bands (top+bottom), taps continue, and force-stops the app.

`--repeat` runs session after session until Ctrl+C (a failed session is logged and the loop continues).

Useful flags: `--tap X,Y`, `--banner-delay`, `--post-win-tap-frac`, `--menu-timeout`, `--dark-timeout`, `--repeat`, plus the same play flags (`--n`, `--roi`, `--settle`, `--show`, `--no-reread`).

## Read board from phone

Phone connected, level open:

```bash
adb devices
python -m meoudoku.cli capture -o /tmp/level.png
python -m meoudoku.cli read --image /tmp/level.png
# or directly from device:
python -m meoudoku.cli read --adb
python -m meoudoku.cli read --adb --solve
```

Optional: `--n 9` to force size, `--roi L,T,R,B` to override auto ROI.

### Vision notes

- Board sits on a **white plate** (white gaps between rounded cells + white outline).
- Outside the plate is **page background** (pale pink/beige), not white.
- ROI is found by rays from screen center until page background (white gaps stay inside).
- Region fills are solid colors (never white/black). **X** is white; **cat** always has black pixels.

## Architecture

Heuristics are classes. The engine runs them in order from the start; the first that returns a `Delta` (cats/xs) is applied, then the stack restarts. Game rules first, backtracking last.
