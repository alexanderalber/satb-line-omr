"""Rule checks that need neither ground truth nor a model.

Part of the delivered path. A staff that the detector drops is the most
expensive error the pipeline can make -- a whole voice missing from the MIDI,
with no warning -- and unlike a wrong duration it produces no bar-sum error to
flag it: the three remaining voices can be perfectly consistent among
themselves. It is a silent failure, and silent is dearer than frequent.

The cheap net: within one score the number of staves per system is almost
always constant. A system that deviates from the score's own majority is
suspect. That is arithmetic over the detector's output -- no numpy, no model,
and it ports to JS as-is.

This is a *net*, not a fix. It says "something is off here", not what.
"""

# A score whose systems disagree this much is not inconsistent, it genuinely
# changes layout (a solo section, a divisi). Then the rule has no majority to
# compare against and must stay silent rather than flag everything.
MIN_MAJORITY = 0.6

# Four staves per system is the norm, but five to seven are common in a
# cappella arrangements and two means closed score -- four voices on two
# staves, which breaks "voice = row index" and is a rejection, not a warning.
CLOSED_SCORE_SIZE = 2


def modal_size(sizes):
    """The staff count that the score itself votes for, and its share."""
    if not sizes:
        return None, 0.0
    counts = {}
    for s in sizes:
        counts[s] = counts.get(s, 0) + 1
    # Ties go to the larger count: a dropped staff makes systems smaller, never
    # bigger, so the larger candidate is the more likely truth.
    best = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return best[0], best[1] / len(sizes)


def check_piece(sizes):
    """Flag the systems that disagree with the rest of the score.

    `sizes` is the staff count of each system, in reading order. Returns a dict
    with the verdict for the score and one entry per suspect system.
    """
    modal, share = modal_size(sizes)
    if modal is None:
        return {"verdict": "leer", "modal": None, "share": 0.0, "suspect": []}

    if modal == CLOSED_SCORE_SIZE and share >= MIN_MAJORITY:
        # Not a detector problem. Two staves carrying four voices is a layout
        # the line-wise approach cannot represent at all.
        return {"verdict": "closed score", "modal": modal, "share": share,
                "suspect": []}

    if share < MIN_MAJORITY:
        # No majority to measure against. Saying nothing is the honest answer;
        # flagging every system would train the user to ignore the colour.
        return {"verdict": "wechselndes Satzbild", "modal": modal,
                "share": share, "suspect": []}

    suspect = [{"system": i, "staves": s, "expected": modal,
                "missing": modal - s}
               for i, s in enumerate(sizes) if s < modal]
    return {
        "verdict": "verdaechtig" if suspect else "konsistent",
        "modal": modal,
        "share": share,
        "suspect": suspect,
    }
