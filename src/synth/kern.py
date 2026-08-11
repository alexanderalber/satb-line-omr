"""Humdrum-kern reading, reduced to what the line-wise ground truth needs.

Not a general Humdrum implementation. It splits a score into records, tracks the
per-spine notation state (clef, key, meter) so that a fragment cut out of the
middle of a piece can be given the same header verovio actually drew at the start
of that staff, and emits single-spine fragments for `parse_kern`.

Training data, not delivered code -- this never ships to the browser.
"""
from dataclasses import dataclass, field
from typing import Literal

RecordKind = Literal["global", "interp", "barline", "data"]

# Labels are dropped: verovio draws "Soprano"/"Alto"/... into the left margin of
# the first system, which is ink our ground truth does not describe and which
# would drag the detector's left crop edge out into the margin. Both the literal
# label (*I"Bass) and the instrument code (*Ibass) produce one, so both go.
# Real choral scores do carry voice names -- reinstating them belongs in the
# phase-2 augmentation, together with lyrics.
_DROP_PREFIXES = ('*I"', "*I'", "*I")


@dataclass
class Record:
    line_no: int          # 1-based line number in the source file, as verovio counts it
    kind: RecordKind
    fields: list[str] = field(default_factory=list)
    raw: str = ""


@dataclass
class Score:
    records: list[Record]
    n_spines: int
    spine_types: list[str]


def parse(text: str) -> Score:
    """Split a kern file into records. Comments are kept for line numbering."""
    records: list[Record] = []
    n_spines = 0
    spine_types: list[str] = []

    for i, raw in enumerate(text.split("\n"), start=1):
        line = raw.rstrip("\r")
        if not line:
            continue
        if line.startswith("!"):
            kind: RecordKind = "global"
            fields: list[str] = []
        else:
            fields = line.split("\t")
            if line.startswith("**"):
                kind = "interp"
                spine_types = fields
                n_spines = len(fields)
            elif line.startswith("*"):
                kind = "interp"
            elif line.startswith("="):
                kind = "barline"
            else:
                kind = "data"
        records.append(Record(line_no=i, kind=kind, fields=fields, raw=line))

    return Score(records=records, n_spines=n_spines, spine_types=spine_types)


def drop_labels(text: str) -> str:
    """Remove instrument-label records (see _DROP_PREFIXES)."""
    keep = [ln for ln in text.split("\n")
            if not any(ln.startswith(p) for p in _DROP_PREFIXES)]
    return "\n".join(keep)


def state_before(score: Score, line_no: int, spine: int) -> dict:
    """Clef, key signature and meter in force for `spine` just before `line_no`.

    `spine` is a 1-based kern field index, matching verovio's `F<n>` element ids.
    """
    st = {"clef": None, "key": None, "meter": None, "met": None}
    for rec in score.records:
        if rec.line_no >= line_no:
            break
        if rec.kind != "interp" or len(rec.fields) < spine:
            continue
        tok = rec.fields[spine - 1]
        if tok.startswith("*clef"):
            st["clef"] = tok
        elif tok.startswith("*k["):
            st["key"] = tok
        elif tok.startswith("*M") and not tok.startswith("*MM"):
            st["meter"] = tok
        elif tok.startswith("*met("):
            st["met"] = tok
    return st


def spine_fragment(score: Score, spine: int, start: int, end: int,
                   header: dict, closing_barline: str | None = None) -> str:
    """A standalone one-spine kern fragment for source lines [start, end].

    `header` selects which of the state tokens to emit, mirroring what verovio
    actually drew on that staff -- a staff that repeats the clef but not the
    meter gets a clef token and no meter token, so image and tokens agree.
    """
    out = ["**kern"]
    for key in ("clef", "key", "meter"):
        tok = header.get(key)
        if tok:
            out.append(tok)

    for rec in score.records:
        if rec.line_no < start or rec.line_no > end:
            continue
        if rec.kind in ("global", "interp") or len(rec.fields) < spine:
            continue
        tok = rec.fields[spine - 1]
        # Null tokens are dropped. A "." means some *other* voice had an event
        # here; nothing is drawn on this staff for it. Keeping them would put
        # symbols in the target that no pixel corresponds to, and worse, two
        # visually identical staves would carry different token sequences
        # depending on what the neighbouring voices do -- not learnable.
        if tok == ".":
            continue
        out.append(tok)

    if closing_barline is not None:
        out.append(closing_barline)
    out.append("*-")
    return "\n".join(out)
