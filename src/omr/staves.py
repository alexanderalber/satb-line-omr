"""Staff and system detection, in plain array operations.

Deliberately written with nothing but numpy indexing and arithmetic -- no
OpenCV, no morphology library. The delivery contract requires this to ship as
dependency-free plain JS, so if it cannot be expressed in flat array ops here,
it cannot be delivered there either.

Assumes cleanly engraved digital sheet music (our stated scope): near-horizontal
staff lines, solid black on white.
"""
import numpy as np


INK_THRESHOLD = 0.6
"""Where ink stops being ink.

0.5 is the obvious choice and it is wrong at small engraving sizes. A staff line
is about a fifth of a line spacing thick, so below roughly 13 px spacing the
stroke is thinner than a pixel and antialiasing spreads it over two rows at
around 50 % grey each. At 0.5 those rows drop out and the staff disappears --
measured on chor002: at scale 70 the detector found 32 line runs where there are
40, and grouped no staff at all, while the page looks perfect to the eye. At 0.6
all eight staves are found at every scale from 60 to 120, and the real score is
unaffected (`scripts/_probe_scale.py`, `scripts/07_staff_detection.py`).
"""


def binarize(gray, threshold=INK_THRESHOLD):
    """gray: float32 (H, W) in [0,1]. Returns bool (H, W), True = ink."""
    return gray < threshold


def longest_run(ink, only_rows=None):
    """Length of the longest uninterrupted horizontal ink run, per row.

    One pass over the columns, carrying a running length -- flat array ops, so
    it ports to plain JS unchanged. `only_rows` restricts the pass to a subset
    of rows; the rest come back as 0. In JS this loop is cheap, in numpy the
    per-column step is not, and a row can be skipped for free whenever its total
    ink is already below the threshold we are testing against (a run can never
    be longer than the row sum, so the shortcut changes no result).
    """
    h, w = ink.shape
    best = np.zeros(h, dtype=np.int32)
    sub = ink if only_rows is None else ink[only_rows]
    run = np.zeros(sub.shape[0], dtype=np.int32)
    sub_best = np.zeros(sub.shape[0], dtype=np.int32)
    for x in range(w):
        col = sub[:, x]
        run = np.where(col, run + 1, 0)
        sub_best = np.maximum(sub_best, run)
    if only_rows is None:
        return sub_best
    best[only_rows] = sub_best
    return best


def find_staff_lines(ink, min_run_frac=0.10):
    """Rows that look like staff lines, as (start, end) row ranges.

    Measured as the longest *contiguous* ink run rather than the total ink in
    the row. Total ink is proportional to system width, so a short closing
    system -- one and a half bars, very common on a last page -- falls below any
    threshold tuned for full-width systems, which is exactly how chor003 lost
    its third accolade. A run length asks the question we actually mean: is
    there a long unbroken horizontal rule in this row?
    """
    width = ink.shape[1]
    threshold = min_run_frac * width
    candidates = ink.sum(axis=1) >= threshold
    is_line = longest_run(ink, only_rows=candidates) >= threshold

    runs = []
    start = None
    for y, v in enumerate(is_line):
        if v and start is None:
            start = y
        elif not v and start is not None:
            runs.append((start, y - 1))
            start = None
    if start is not None:
        runs.append((start, len(is_line) - 1))
    return runs


def _evenly_spaced(centers, tolerance):
    gaps = np.diff(centers)
    if gaps.min() <= 0:
        return False
    return (gaps.max() - gaps.min()) <= tolerance * gaps.mean()


def _fits(anchors, idx, tolerance):
    """Do these five runs sit at equal distance under *any* reading of them?

    A clean staff line is a two-pixel run and all three readings -- top edge,
    middle, bottom edge -- agree. Where a beam or a slur touches the line the
    run swells to ten pixels and the middle drifts by half of that, enough to
    fail the evenness test and lose the whole staff. Which edge stays honest
    depends on which way the extra ink grew, so all three are tried and one
    agreement is enough. Trying more readings can only ever find staves, never
    remove one that the strict reading already accepted.
    """
    for a in anchors:
        if _evenly_spaced([a[j] for j in idx], tolerance):
            return True
    return False


def _group_pass(pool, anchors, tolerance):
    """One greedy left-to-right walk over `pool`, a list of run indices.

    A long flat tie or slur can be ink-continuous over a few hundred pixels and
    therefore looks like a staff line, landing between two real ones. It cannot
    be filtered out by length alone -- a short closing system has genuinely short
    lines. So when five consecutive runs do not fit, one intruder may be skipped:
    the honest reading is tried first, the skip only as a fallback.
    """
    picked_all = []
    i = 0
    while i + 4 < len(pool):
        if _fits(anchors, [pool[j] for j in range(i, i + 5)], tolerance):
            picked_all.append([pool[j] for j in range(i, i + 5)])
            i += 5
            continue

        picked = None
        if i + 5 < len(pool):
            for skip in range(6):
                idx = [j for j in range(i, i + 6) if j != i + skip]
                if _fits(anchors, [pool[j] for j in idx], tolerance):
                    picked = idx
                    break
        if picked:
            picked_all.append([pool[j] for j in picked])
            i = picked[-1] + 1
        else:
            i += 1
    return picked_all


def group_lines_into_staves(runs, tolerance=0.45):
    """Group line runs into staves of five equally spaced lines.

    Measured on the choir's own 107 engraved scores (`26_consistency.py`): the
    strict reading, which took each run's midpoint as the line position, lost a
    staff in 148 of 2238 systems -- 42 % of the scores were affected. A lost
    staff is a whole voice missing from the MIDI with nothing to flag it, so it
    is the most expensive failure this file can produce.

    The cause is not the greedy walk but the midpoint. Where a beam or a slur
    touches a staff line the ink run swells and its middle drifts with it; see
    `_fits`. Reading each run at all three of its edges brings the figure down
    to 84 systems and 22 scores, with no change on the reference score
    (`07_staff_detection.py`, 20 of 20 systems four-staff before and after).

    Two things that did *not* work, so they are not tried again: a second
    grouping pass over the leftover runs made it worse (148 -> 195, it grafts
    staves out of slur clusters), and constraining that pass by the page's
    median line spacing made it contribute exactly nothing once `_fits` was in
    place.
    """
    if len(runs) < 5:
        return []

    tops = [float(a) for a, _ in runs]
    bottoms = [float(b) for _, b in runs]
    centers = [(a + b) / 2.0 for a, b in runs]
    anchors = (centers, tops, bottoms)

    def spacing_of(idx):
        # Read the spacing off whichever edge is most even for this staff: the
        # swollen run must not drag the spacing along with it either.
        best = None
        for a in anchors:
            gaps = [a[idx[k + 1]] - a[idx[k]] for k in range(4)]
            spread = max(gaps) - min(gaps)
            if best is None or spread < best[0]:
                best = (spread, sum(gaps) / 4.0)
        return best[1]

    def emit(idx):
        return {
            "top": int(runs[idx[0]][0]),
            "bottom": int(runs[idx[4]][1]),
            "line_spacing": float(spacing_of(idx)),
            "line_centers": [centers[j] for j in idx],
        }

    return [emit(g) for g in _group_pass(list(range(len(runs))), anchors, tolerance)]


def find_systems(ink, staves, min_span_frac=0.8):
    """Group staves into systems using the vertical bracket/barline at the left.

    Staves joined by a vertical rule running through the gap between them belong
    to the same system; a gap with no such rule is a system break.
    """
    if not staves:
        return []

    systems = [[0]]
    for i in range(1, len(staves)):
        prev, cur = staves[i - 1], staves[i]
        gap_top, gap_bot = prev["bottom"] + 1, cur["top"] - 1
        joined = False
        if gap_bot > gap_top:
            band = ink[gap_top:gap_bot + 1, :]
            # A column is "connecting" if it is ink for most of the gap height.
            col_frac = band.sum(axis=0) / band.shape[0]
            joined = bool((col_frac >= min_span_frac).any())
        if joined:
            systems[-1].append(i)
        else:
            systems.append([i])
    return systems


def grow_edge(ink, x0, x1, y_start, direction, spacing,
              min_pad_ratio=2.0, max_pad_ratio=3.0, gap_ratio=0.8, limit=None):
    """How far to extend a crop away from the staff, in pixels.

    A fixed padding factor cannot serve both masters: fermatas and low ledger
    notes need room, lyrics sit right underneath and must stay out. So the edge
    is grown through connected ink and stopped at the first clear gap of
    `gap_ratio` spacings -- which is what separates a staff from the lyric line
    below it, while a stem or ledger line hanging off the staff has no such gap.

    `min_pad_ratio` is applied unconditionally, because floating symbols
    (fermatas, slurs) sit in their own small gap above the staff.
    """
    min_pad = int(round(min_pad_ratio * spacing))
    max_pad = int(round(max_pad_ratio * spacing))
    if limit is not None:
        max_pad = min(max_pad, int(limit))
    max_pad = max(max_pad, min_pad)
    gap_need = max(1, int(round(gap_ratio * spacing)))

    pad, run_empty = min_pad, 0
    y = y_start + direction * min_pad
    while pad < max_pad:
        y += direction
        if y < 0 or y >= ink.shape[0]:
            break
        run_empty = 0 if ink[y, x0:x1].any() else run_empty + 1
        if run_empty >= gap_need:
            pad -= run_empty - 1        # do not keep the blank paper we walked
            break
        pad += 1
    return max(min_pad, min(pad, max_pad))


def staff_boxes(ink, staves, systems, pad_ratio=None):
    """Crop boxes per staff, padded by ledger-line room but not into lyrics.

    Padding is expressed in staff-line spacings so it scales with the engraving
    rather than with the page dpi. `pad_ratio`, if given, forces the old fixed
    padding -- kept so the measurement scripts can compare the two.
    """
    h, w = ink.shape

    boxes = []
    for sys_idx, staff_ids in enumerate(systems):
        # Horizontal extent is taken per system, not per page: a short closing
        # system must not be padded out to the width of the full ones, or the
        # crop is mostly blank paper.
        top = staves[staff_ids[0]]["top"]
        bottom = staves[staff_ids[-1]]["bottom"]
        cols = ink[top:bottom + 1, :].any(axis=0)
        x0 = int(np.argmax(cols)) if cols.any() else 0
        x1 = int(w - np.argmax(cols[::-1])) if cols.any() else w

        for voice_idx, sid in enumerate(staff_ids):
            s = staves[sid]
            spacing = s["line_spacing"]
            # Never grow more than halfway towards a neighbouring staff, or two
            # crops would claim the same ink.
            up_limit = ((s["top"] - staves[staff_ids[voice_idx - 1]]["bottom"]) / 2
                        if voice_idx > 0 else None)
            down_limit = ((staves[staff_ids[voice_idx + 1]]["top"] - s["bottom"]) / 2
                          if voice_idx + 1 < len(staff_ids) else None)

            if pad_ratio is not None:
                pad_up = pad_down = int(round(pad_ratio * spacing))
            else:
                pad_up = grow_edge(ink, x0, x1, s["top"], -1, spacing, limit=up_limit)
                pad_down = grow_edge(ink, x0, x1, s["bottom"], +1, spacing,
                                     limit=down_limit)

            y0 = max(0, s["top"] - pad_up)
            y1 = min(h, s["bottom"] + pad_down)
            boxes.append({
                "staff": sid,
                "system": sys_idx,
                "voice": voice_idx,
                "x": x0,
                "y": y0,
                "w": x1 - x0,
                "h": y1 - y0,
                "line_spacing": spacing,
                "pad_up": pad_up,
                "pad_down": pad_down,
            })
    return boxes


def detect(gray, threshold=INK_THRESHOLD):
    """Full pipeline: grayscale page -> staff crop boxes with voice indices."""
    ink = binarize(gray, threshold)
    runs = find_staff_lines(ink)
    staves = group_lines_into_staves(runs)
    systems = find_systems(ink, staves)
    return {
        "line_runs": len(runs),
        "staves": staves,
        "systems": systems,
        "boxes": staff_boxes(ink, staves, systems),
    }
