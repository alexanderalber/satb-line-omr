"""The rejection path: scan and closed score are recognised, not supported.

CLAUDE.md Nr. 7 and entscheid-schema-freeze-v12-2026-08-04.md §2.1. One field,
one source: this module is the only producer of `rejected`, and the JS port
(`omr-reject.js`) mirrors it statement for statement -- the frontend adds no
heuristic of its own.

Two stages, and they end differently on purpose:

  1. **Scan suspicion** from render statistics of the grey page -> a WARNING.
     Thresholds fitted on the training half of the choir's own archive and
     reported on the holdout half (befund-ablehnungspfad-2026-08-04.md,
     work/52_ablehnungspfad.json): holdout recall 0.8837 at a false-warning
     rate of 0.0000 over 114 engraved scores.
  2. **Closed score** from the modal number of staves per system -> a
     REJECTION. Two staves per system in a choral score means two voices share
     a staff, which breaks "voice = row index" (CLAUDE.md Nr. 3). That is not
     a quality problem the editor can absorb; it is the wrong structure.

The asymmetry is an owner decision, not a technical one: a scan produces bad
notes, closed score produces wrong ones. Stage 2 stays silent when stage 1
fired -- measured reason, see decide().

Flat array arithmetic only: what cannot be expressed this way is not
deliverable in ECMAScript either.
"""

# Fitted on the training half only, never touched afterwards. Moving these
# without a new measurement is how a measured rule turns into a guessed one.
SCAN_WHITE_MAX = 0.875318      # share of pixels >= 0.99: paper is not white
SCAN_RUN_MAX = 0.385546        # staff-line runs, in page widths: skew breaks them
WHITE_LEVEL = 0.99
INK_LEVEL = 0.6
RUN_ROWS = 20
RUN_MIN_INK = 0.30

CLOSED_SCORE_STAVES = 2

REASON_SCAN = "scan-suspected"
REASON_CLOSED_SCORE = "closed-score-suspected"
REASON_NO_STAVES = "no-staves-found"


def page_stats(gray):
    """The two render statistics the rule uses, for one grey page.

    gray: 2-D float32 array in [0, 1], exactly what `staves.detect` takes.
    Plain NumPy on purpose, same reason as staves.py: what does not express
    itself in flat array operations is not deliverable in ECMAScript either.

    Ink comes from `staves.binarize` rather than a second comparison written
    here. One ink definition per project: the shipped preprocessing compares
    `< 0.6`, and at the float32 boundary `<=` and `<` do not classify the same
    pixel the same way.
    """
    import numpy as np
    from . import staves

    height, width = gray.shape
    ink = staves.binarize(gray)
    ink_per_row = ink.sum(axis=1)
    candidates = np.flatnonzero(ink_per_row >= RUN_MIN_INK * width)
    if candidates.size:
        # stable on purpose: ties keep the smaller row index, so the JS port
        # (Array.sort with an explicit index tiebreak) cuts the top-20 at the
        # same row. numpy's default quicksort would not guarantee that.
        rows = candidates[np.argsort(-ink_per_row[candidates],
                                     kind="stable")][:RUN_ROWS]
        runs = staves.longest_run(ink, only_rows=rows)[rows]
        run_frac = float(np.mean(runs)) / width
    else:
        run_frac = 0.0
    return {"whiteFrac": float((gray >= WHITE_LEVEL).mean()),
            "runFrac": run_frac}


def median(values):
    """Median with the same tie handling as numpy: mean of the two middle
    values on even counts. One dark title page must not decide a score."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def scan_suspected(stats_per_page):
    """The fitted rule: white_frac <= t1 ODER run_frac <= t2, on the median
    over the measured pages -- exactly as it was measured."""
    w = median([s["whiteFrac"] for s in stats_per_page])
    r = median([s["runFrac"] for s in stats_per_page])
    return (w <= SCAN_WHITE_MAX) or (r <= SCAN_RUN_MAX), {"whiteFrac": w,
                                                          "runFrac": r}


def modal_staves(system_staff_counts):
    """Most common staves-per-system; ties go to the smaller count, so a tie
    can only ever make the closed-score rule MORE careful, never less.

    Deliberately the OPPOSITE of `ir.py::_modal`, where a tie goes to the
    larger count: there a wrong modal drops a voice from the IR (the most
    expensive error class this project knows), here it only makes a rejection
    more cautious. Two rules, two costs -- whoever changes one has to read
    the other (`durchsicht-reihenfolge-2026-08-09.md`, Nr. 1 und 4).
    """
    if not system_staff_counts:
        return 0
    best, n = 0, -1
    for v in sorted(set(system_staff_counts)):
        c = system_staff_counts.count(v)
        if c > n:
            best, n = v, c
    return best


def decide(stats_per_page, system_staff_counts):
    """-> {rejected, reason, scanSuspected, detail, warnings}.

    Owner decision of 2026-08-04 (befund-ablehnungspfad-2026-08-04.md, §9):
    **a suspected scan is a warning, not a rejection.** The page says "no
    scans" and the users are adults -- they get told and decide. Measured
    consequence, accepted: the tool then produces notes from a scan that are
    not worth keeping, and the user throws them away herself.

    Closed score stays a rejection: there the output is not merely poor, it is
    structurally wrong (two voices on one staff breaks "voice = row index",
    CLAUDE.md Nr. 3), and no amount of editing in the frontend repairs it.

    The closed-score question is not asked on a suspected scan. This one line
    is the single most valuable thing in this module, and the number says so:
    **24 of the 80 warned scans have a modal staff count of exactly 2**
    (work/53_ablehnung_gegenprobe.json). Without the guard, 30 % of every scan
    the tool recognises would be rejected as closed score -- a rejection whose
    stated reason is false, and the reason is what the user is shown. On a
    scan the detector returns systems that mean nothing; two staves per system
    is what noise looks like, not what the page says.
    """
    warnings = []
    is_scan, medians = scan_suspected(stats_per_page)
    if is_scan:
        warnings.append({
            "code": REASON_SCAN,
            "message": "Die Seite sieht nach einem Scan aus (Weissanteil "
                       f"{round(medians['whiteFrac'], 4)}, Notenlinienlauf "
                       f"{round(medians['runFrac'], 4)}). Dafuer ist das "
                       "Werkzeug nicht gebaut -- die Erkennung laeuft "
                       "trotzdem, das Ergebnis ist vermutlich unbrauchbar."})

    if not system_staff_counts:
        warnings.append({
            "code": REASON_NO_STAVES,
            "message": "Keine Notensysteme gefunden."})
        return {"rejected": True, "reason": REASON_NO_STAVES,
                "scanSuspected": is_scan, "detail": medians,
                "warnings": warnings}

    modal = modal_staves(system_staff_counts)
    detail = {**medians, "modalStaves": modal}
    if modal == CLOSED_SCORE_STAVES and not is_scan:
        warnings.append({
            "code": REASON_CLOSED_SCORE,
            "message": "Zwei Systeme je Akkolade: zwei Stimmen teilen sich "
                       "eine Zeile (closed score). Die Zuordnung Note -> "
                       "Stimme ist so nicht moeglich."})
        return {"rejected": True, "reason": REASON_CLOSED_SCORE,
                "scanSuspected": is_scan, "detail": detail,
                "warnings": warnings}

    return {"rejected": False, "reason": None, "scanSuspected": is_scan,
            "detail": detail, "warnings": warnings}
