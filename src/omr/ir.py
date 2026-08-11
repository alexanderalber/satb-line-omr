"""Score-IR assembly: CTC logits -> the intermediate format of contract 3.

Reference implementation for the JS port (`omr-decode.js`), therefore flat:
no classes with state, no numpy tricks that do not translate to typed arrays.
Schema: handoffs/design-score-ir-2026-08-03.md.

Two layers, mirroring the delivery contract:
  decode_line(logits, i2w)      -> token events with frame spans + confidences
  assemble_ir(line_results)     -> the IR dict for one score

The token grammar is the corpus tokenisation: records separated by `<b>`,
a record is a barline (`=...`), an interpretation (`*...`) or an event
(duration token, then pitch or `r`, then accidental/tie/slur/beam/fermata
in kern order). `<b>` itself carries no ink and gets no coordinates.
"""
from fractions import Fraction

# 1.2: `rejected` gets a real producer (reject.py), `scan-suspected` changed
# from a rejection reason to a warning, warnings carry `page`, and
# `line-truncated` moved here from the site repo's worker. The fields are
# additive, but the MEANING of scan-suspected moved -- that is what a version
# bump is for (schema freeze §1).
IR_VERSION = "1.2"
WIDTH_REDUCTION = 4          # CTC frame -> normalised pixel, model constant
DIV_WHOLE = 192              # divisions of a whole note (48 per quarter)

BLANK = 0
SEP = "<b>"

DURATIONS = {"0", "1", "2", "3", "4", "6", "8", "12", "16", "24", "32"}
ACCIDENTALS = {"#": 1, "##": 2, "-": -1, "--": -2, "n": 0}
BEAMS = {"L", "LL", "J", "JJ", "JJJ", "JJJJ", "k"}

BARLINE_TYPES = {
    "=": "regular", "==": "final", "=||": "double", "=-": "invisible",
    "=:|!": "repeat-end", "=!|:": "repeat-start", "=:|!|:": "repeat-both",
    "==:|!": "final-repeat-end",
}

# v1.1 null-bar collapse. The net double-emits at printed double bars
# (`=||` AND `=` for one object -> zero-duration measure, +1 bar count;
# befund-nulltakt-kollaps-2026-08-04.md). Adjacent barline records merge --
# EXCEPT the two pairings the ground truth itself contains (245 238 lines
# counted, work/_probe_gt_leertakt.json): empty printed measures. Those
# stay, everything else is provably not in the corpus and collapses.
LEGITIMATE_ADJACENT = {("=", "="), ("=", "==")}


def softmax_rows(logits):
    """(T, C) float logits -> per-frame probabilities, plain loops avoided
    but nothing beyond exp/sum so the JS port stays line-for-line.

    float64 on purpose: the JS port computes in doubles (JS has nothing
    else), and parity of the rounded confidences requires the reference
    to do the same -- float32 exp/sum diverges in the 7th digit."""
    import numpy as np
    z = logits.astype(np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def decode_line(logits, i2w):
    """Greedy CTC decode with frame spans and confidences.

    logits: (T, C) for one line. Returns a list of
    {token, f0, f1, confidence}: f0..f1 inclusive frame span of the emission,
    confidence = mean probability of the emitted class over its span.
    """
    probs = softmax_rows(logits)
    ids = logits.argmax(-1)
    events = []
    prev = -1
    for f, k in enumerate(ids.tolist()):
        if k != BLANK and k != prev:
            events.append({"token": i2w[k], "f0": f, "f1": f,
                           "_psum": float(probs[f, k]), "_n": 1})
        elif k != BLANK and k == prev:
            ev = events[-1]
            ev["f1"] = f
            ev["_psum"] += float(probs[f, k])
            ev["_n"] += 1
        prev = k
    for ev in events:
        ev["confidence"] = round(ev.pop("_psum") / ev.pop("_n"), 4)
    return events


def _records(token_events):
    """Split the token event stream into records at `<b>`."""
    recs, cur = [], []
    for ev in token_events:
        if ev["token"] == SEP:
            if cur:
                recs.append(cur)
            cur = []
        else:
            cur.append(ev)
    if cur:
        recs.append(cur)
    return recs


def _merge_barlines(a, b):
    """Two adjacent barline records -> one. The more specific token survives
    (if exactly one of the pair is a plain `=`, the other one); the span
    widens to the union, because both emissions point at the same ink and
    the bbox should cover it. Confidence stays the survivor's."""
    keep = b if (a[0]["token"] == "=" and b[0]["token"] != "=") else a
    f0 = min(a[0]["f0"], b[0]["f0"])
    f1 = max(a[-1]["f1"], b[-1]["f1"])
    first = dict(keep[0])
    first["f0"], first["f1"] = f0, f1
    return [first] + keep[1:]


def _collapse_barlines(recs):
    """v1.1: merge adjacent barline records unless the pairing is legitimate
    ground truth (empty printed measures, LEGITIMATE_ADJACENT). Runs while
    appending, so a chain of three emissions merges into one."""
    out = []
    for rec in recs:
        out.append(rec)
        while len(out) >= 2:
            a, b = out[-2], out[-1]
            if not (a[0]["token"].startswith("=")
                    and b[0]["token"].startswith("=")):
                break
            if (a[0]["token"], b[0]["token"]) in LEGITIMATE_ADJACENT:
                break
            merged = _merge_barlines(a, b)
            del out[-2:]
            out.append(merged)
    return out


def _dur(token):
    """Duration token -> (divisions, base, dots); None if not a duration."""
    base = token.rstrip(".")
    if base not in DURATIONS:
        return None
    dots = len(token) - len(base)
    d = DIV_WHOLE * 2 if base == "0" else DIV_WHOLE // int(base)
    total, add = d, d
    for _ in range(dots):
        add //= 2
        total += add
    return total, base, dots


def _pitch(token):
    """kern pitch letters -> (step, octave); None if not a pitch."""
    if not token or not token[0].isalpha():
        return None
    ch = token[0]
    if token != ch * len(token) or ch.upper() not in "ABCDEFG":
        return None
    if ch.islower():
        return ch.upper(), 3 + len(token)      # c=C4, cc=C5, ...
    return ch, 4 - len(token)                  # C=C3, CC=C2, ...


def _classify(record):
    """One record (list of token events) -> an IR element dict, or None."""
    toks = [ev["token"] for ev in record]
    conf = min(ev["confidence"] for ev in record)
    first = toks[0]

    if first.startswith("="):
        return {"kind": "barline",
                "type": BARLINE_TYPES.get(first, "other"),
                "confidence": conf}

    if first.startswith("*"):
        el = {"kind": "attribute", "confidence": conf}
        if first.startswith("*clef"):
            body = first[len("*clef"):]           # G2, F4, Gv2
            octave = -1 if "v" in body else (1 if "^" in body else 0)
            body = body.replace("v", "").replace("^", "")
            el["clef"] = {"sign": body[0], "line": int(body[1:] or 0),
                          "octaveChange": octave}
        elif first.startswith("*k["):
            inner = first[3:-1]
            n = inner.count("#") or -inner.count("-")
            el["keyFifths"] = n
        elif first.startswith("*M"):
            num, den = first[2:].split("/")
            el["time"] = {"num": int(num), "den": int(den)}
        else:
            return None
        return el

    dur = next((d for t in toks if (d := _dur(t))), None)
    if dur is None:
        return {"kind": "unparseable", "tokens": toks, "confidence": conf}
    divisions, base, dots = dur
    el = {"kind": "rest" if "r" in toks else "note",
          "duration": {"divisions": divisions, "base": base, "dots": dots},
          "confidence": conf}
    if el["kind"] == "note":
        pitch = next((p for t in toks if (p := _pitch(t))), None)
        if pitch is None:
            return {"kind": "unparseable", "tokens": toks, "confidence": conf}
        step, octave = pitch
        alter = next((ACCIDENTALS[t] for t in toks if t in ACCIDENTALS), 0)
        el["pitch"] = {"step": step, "alter": alter, "octave": octave}
        if "[" in toks:
            el["tie"] = "start"
        elif "]" in toks:
            el["tie"] = "stop"
        elif "_" in toks:
            el["tie"] = "continue"
        else:
            el["tie"] = None
        if "(" in toks:
            el["slur"] = "start"
        elif ")" in toks:
            el["slur"] = "stop"
        el["fermata"] = ";" in toks
    return el


def _bbox(record, staff):
    """Union frame span of a record -> page-pixel bbox (vertical stripe)."""
    f0 = min(ev["f0"] for ev in record)
    f1 = max(ev["f1"] for ev in record)
    scale = staff["lineSpacingPx"] / staff["normSpacing"]
    x0 = staff["bbox"][0] + WIDTH_REDUCTION * f0 * scale
    x1 = staff["bbox"][0] + WIDTH_REDUCTION * (f1 + 1) * scale
    return [round(x0, 1), staff["bbox"][1],
            round(x1 - x0, 1), staff["bbox"][3]]


def line_fragment(token_events, staff):
    """Decoded line -> list of elements with tokens, confidence, bbox.

    staff: {page, system, staffIndex, bbox, lineSpacingPx, normSpacing}.
    """
    out = []
    for record in _collapse_barlines(_records(token_events)):
        el = _classify(record)
        if el is None:
            continue
        el["tokens"] = [ev["token"] for ev in record]
        el["src"] = {"page": staff["page"], "system": staff["system"],
                     "staff": staff["staffIndex"],
                     "bbox": _bbox(record, staff)}
        out.append(el)
    return out


def assemble_ir(lines, pages, generator="ir.py reference", rejection=None):
    """lines: [{staff: {...}, elements: [...]}] in reading order.

    Groups by part (staffIndex within system), walks measures across
    systems, collects structure and warnings. Returns the IR dict.

    rejection: the verdict of `reject.decide()`, or None when the caller did
    not run the rejection path. It is passed in rather than computed here
    because it needs the grey page, which the IR layer never sees -- but it
    stays the ONLY source of `rejected` (entscheid-schema-freeze-v12 §2.1).
    A rejected score still carries whatever was decoded: the frontend shows
    the reason, and the user decides. Nothing is silently withheld.
    """
    # v1.2: keyed by (page, system). System indices restart at 0 on every
    # page, so the v1.1 counting folded system n of page 1 together with
    # system n of page 2 -- on a one-page score invisible, on a multi-page one
    # simply wrong. The site repo found the same thing from the other side.
    counts = {}
    for ln in lines:
        key = (ln["staff"]["page"], ln["staff"]["system"])
        counts[key] = max(counts.get(key, 0), ln["staff"]["staffIndex"] + 1)
    modal = _modal(list(counts.values())) if counts else 0

    warnings = []
    for (page, sysno), n in sorted(counts.items()):
        if n != modal:
            warnings.append({"code": "staff-count-mismatch",
                             "page": page, "system": sysno,
                             "message": f"Seite {page}, System {sysno}: "
                                        f"{n} Zeilen erkannt, "
                                        f"Struktur sagt {modal}"})

    parts = [{"index": i, "label": None, "measures": []} for i in range(modal)]
    open_measures = [None] * modal
    structure = {"stavesPerSystem": modal, "clefs": [None] * modal,
                 "keyFifths": None, "time": None}

    for ln in lines:
        st = ln["staff"]
        p = st["staffIndex"]
        # v1.2: the preprocessing reports `truncated` per line (MAX_WIDTH);
        # until now the site repo's worker appended this code after the fact.
        # The namespace keeps one source, so it is emitted here.
        if st.get("truncated"):
            warnings.append({"code": "line-truncated",
                             "page": st["page"], "system": st["system"],
                             "staff": p,
                             "message": f"Seite {st['page']}, System "
                                        f"{st['system']}, Zeile {p}: rechts "
                                        "abgeschnitten (MAX_WIDTH) -- was "
                                        "dahinter steht, wurde nicht gelesen"})
        if p >= modal:
            continue
        part = parts[p]
        for el in ln["elements"]:
            if el["kind"] == "unparseable":
                # This message stays RAW: the tokens joined by single spaces,
                # no prose frame, unlike every other warning above. The site
                # repo's insert suggestion parses it with `message.split(" ")`
                # and would keep splitting happily on a prose sentence, just
                # into the wrong words -- a silent break, not an exception.
                # Formatting here is therefore load-bearing for a parser over
                # there. v1.3 removes the coupling by shipping `tokens`
                # additively, and `message` stays unchanged even then, because
                # it is their fallback for older imported score JSONs.
                # See entscheid-v13-zuschnitt-2026-08-04.md §2.
                warnings.append({"code": "unparseable-tokens",
                                 "page": st["page"], "system": st["system"],
                                 "staff": p,
                                 "message": " ".join(el["tokens"])})
                continue
            if el["kind"] == "attribute":
                m = _open(part, open_measures, p, st["system"])
                m.setdefault("attributes", {}).update(
                    {k: v for k, v in el.items()
                     if k in ("clef", "keyFifths", "time")})
                if "clef" in el and structure["clefs"][p] is None:
                    c = el["clef"]
                    structure["clefs"][p] = (c["sign"] + str(c["line"])
                                             + ("v8" if c["octaveChange"] < 0 else ""))
                if "keyFifths" in el and structure["keyFifths"] is None:
                    structure["keyFifths"] = el["keyFifths"]
                if "time" in el and structure["time"] is None:
                    structure["time"] = el["time"]
                continue
            if el["kind"] == "barline":
                m = open_measures[p]
                if m is not None:
                    m["barline"] = {k: el[k] for k in
                                    ("type", "confidence", "tokens", "src")}
                    open_measures[p] = None
                continue
            m = _open(part, open_measures, p, st["system"])
            m["events"].append(el)

    if rejection is not None:
        # the rejection reasons come first: they explain the page as a whole,
        # while the decode warnings explain single spots on it
        warnings = list(rejection.get("warnings", [])) + warnings

    ir = {"irVersion": IR_VERSION,
          "generator": {"model": "omr-2026-08", "decoder": generator},
          "rejected": bool(rejection["rejected"]) if rejection else False,
          "rejectionReason": rejection["reason"] if rejection else None,
          "source": {"pages": pages},
          "structure": {"recognized": structure, "confirmed": None},
          "systems": _systems(lines),
          "parts": parts,
          "warnings": warnings}
    return ir


def _open(part, open_measures, p, system):
    if open_measures[p] is None:
        m = {"index": len(part["measures"]), "system": system, "events": []}
        part["measures"].append(m)
        open_measures[p] = m
    return open_measures[p]


def _modal(values):
    """Most common value; on a tie the LARGER one wins.

    v1.2. Two reasons, and the second is the real one:

    1. Parity. The old loop ran over `set(values)`, whose order is an
       implementation detail; the JS port sorted ascending. They agreed by
       accident for small integers, which is not a guarantee.
    2. Correctness. `modal` decides how many parts exist, and a line with
       `staffIndex >= modal` is dropped from the IR entirely. Picking the
       smaller count on a tie therefore deletes a voice in silence, while
       picking the larger one merely raises `staff-count-mismatch` on the
       short systems. A lost voice is the most expensive error this project
       knows (befund-chorrepertoire.md §3); a warning is the cheapest.

    Deliberately the OPPOSITE of `reject.py::modal_staves`, where a tie goes
    to the smaller count -- there a tie only makes the closed-score rejection
    more cautious. Two rules, two costs; whoever changes one has to read the
    other (`durchsicht-reihenfolge-2026-08-09.md`, Nr. 1 und 4).
    """
    best, n = 0, 0
    for v in sorted(set(values), reverse=True):
        c = values.count(v)
        if c > n:
            best, n = v, c
    return best


def _systems(lines):
    by_sys = {}
    for ln in lines:
        st = ln["staff"]
        by_sys.setdefault((st["page"], st["system"]), []).append(st)
    out = []
    for (page, sysno), staves in sorted(by_sys.items()):
        xs = [s["bbox"][0] for s in staves]
        ys = [s["bbox"][1] for s in staves]
        x2 = [s["bbox"][0] + s["bbox"][2] for s in staves]
        y2 = [s["bbox"][1] + s["bbox"][3] for s in staves]
        out.append({"index": sysno, "page": page,
                    "bbox": [min(xs), min(ys),
                             max(x2) - min(xs), max(y2) - min(ys)],
                    "staves": [{"part": s["staffIndex"], "bbox": s["bbox"],
                                "lineSpacingPx": s["lineSpacingPx"],
                                "normScale": round(
                                    s["lineSpacingPx"] / s["normSpacing"], 4)}
                               for s in sorted(staves,
                                               key=lambda s: s["bbox"][1])]})
    return out


def bar_sums(part):
    """Duration sum per measure in quarters, for validation and tests."""
    out = []
    for m in part["measures"]:
        q = sum(Fraction(e["duration"]["divisions"], 48) for e in m["events"])
        out.append(q)
    return out
