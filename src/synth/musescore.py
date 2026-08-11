"""The second renderer: kern -> music21 -> MusicXML -> MuseScore -> PNG.

Why this exists: the measured gap of the line model is rendered-vs-printed, and
31 of the 202 real PDFs are themselves MuseScore output. Rendering the corpus a
second time with MuseScore yields foreign engraving with exact ground truth --
no hand labelling, no optimistic selection.

The two decisions that shape this module:

- **Ground truth comes from the MusicXML, never from the original kern.**
  music21 completes the kern (it pads voices with rests -- measured in
  `_probe_musicxml.py`: 60x "rests added", 20x identical, 0x "notes changed"
  over 20 chorales), and those rests have ink once MuseScore draws them. By
  CLAUDE.md rule 6 the target must contain exactly what is drawn, so the token
  sequence is derived from the very file MuseScore renders.
- **The measure -> system mapping comes out of MuseScore itself** (`-o *.mpos`,
  one element per measure with page and y), the same way the verovio pipeline
  reads systems out of SVG ids instead of reconstructing them from geometry.

Beams are safe by construction: music21 carries the kern beam marks into
<beam> elements (measured: 48 in = 48 out on chor001), and MuseScore draws the
beams the file specifies rather than re-deriving its own.

MuseScore Studio 4 is GPL-3.0 and music21 BSD-3; both are build-time tools in
the same category as verovio -- they render training/evaluation data and are
never shipped (see MANIFEST.md).

Training/evaluation rig only. Never shipped.
"""
import re
import subprocess
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

MUSESCORE = Path(r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe")

# Applied to *every* invocation: mpos and page render must see one layout, or
# the measure->system mapping would describe a different score than the image.
# The values and their measurement live in the file itself.
STYLE = Path(__file__).with_name("musescore_style.mss")

# 82 px staff-line spacing at 1200 dpi -> 14.0 px at 205 dpi, measured with the
# shipping detector on chor001. That is the band the training corpus lives in.
DPI = 205


class Unconvertible(Exception):
    """This piece cannot carry exact ground truth through the chain."""


# ------------------------------------------------------------------ rendering
def _add_syl(note, text: str) -> None:
    """music21 spells continuation differently from Humdrum: the syllabic
    kind is derived from the hyphens. Repeated addLyric on the same note
    numbers the verses 1..n (measured, `_probe_verses.py`)."""
    if text.startswith("-") and text.endswith("-"):
        note.addLyric(text.strip("-"))
        note.lyrics[-1].syllabic = "middle"
    elif text.endswith("-"):
        note.addLyric(text[:-1])
        note.lyrics[-1].syllabic = "begin"
    elif text.startswith("-"):
        note.addLyric(text[1:])
        note.lyrics[-1].syllabic = "end"
    else:
        note.addLyric(text)
        note.lyrics[-1].syllabic = "single"


def add_lyrics(score, hymn: int, verses: int = 1) -> int:
    """Attach hymn syllables to every sung note, as the realism layer does.

    Ruling in antwort-zwischenbericht-2026-07-30-abend.md: a text-free
    MuseScore share would reopen exactly the crop-geometry gap that was
    measured and closed for verovio -- lyrics change the picture inside the
    crop (spacing) and below it (neighbouring text in the padding), and the
    real MuseScore choral PDFs all carry text. Same word source as
    `realism.syllables`, so the width distribution matches; the syllables are
    ink without a target token either way. `verses` stacks that many hymn
    lines under each staff (pop-package decision, 02.08.), offset from `hymn`
    like the realism layer does. Returns the syllable count.
    """
    from .realism import HYMN_LINES, syllables

    syls = [syllables((hymn + v) % len(HYMN_LINES)) for v in range(verses)]
    n = 0
    for part in score.parts:
        k = 0
        for note in part.recurse().notes:
            if note.tie is not None and note.tie.type in ("continue", "stop"):
                continue                      # the word is held, not sung again
            for syl in syls:
                _add_syl(note, syl[k % len(syl)])
                n += 1
            k += 1
    return n


def to_musicxml(kern_path: Path, xml_path: Path, hymn: int | None = None,
                verses: int = 1) -> None:
    """kern file -> MusicXML on disk, via music21; `hymn` adds lyrics."""
    from music21 import converter

    score = converter.parse(str(kern_path))
    if hymn is not None:
        add_lyrics(score, hymn, verses=verses)
    score.write("musicxml", fp=str(xml_path))


def run_musescore(args: list[str]) -> None:
    r = subprocess.run([str(MUSESCORE), *args], capture_output=True, text=True,
                       timeout=600)
    if r.returncode != 0:
        raise Unconvertible(f"MuseScore exit {r.returncode}: {r.stderr[-300:]}")


def load_gray(path: Path) -> np.ndarray:
    """PNG -> float32 grayscale in [0,1]; MuseScore exports transparent paper."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        flat = Image.new("RGB", img.size, (255, 255, 255))
        flat.paste(img, mask=img.split()[3])
        img = flat
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def render(xml_path: Path, out_dir: Path, dpi: int = DPI):
    """MusicXML -> (page images, systems as lists of measure indices).

    Two MuseScore invocations: one for the page PNGs, one for the measure
    positions. The mpos y coordinate is only used to group measures into
    systems and order them; pixel geometry comes from the staff detector.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / xml_path.stem
    mpos = stem.with_suffix(".mpos")
    run_musescore(["-S", str(STYLE), "-o", str(mpos), str(xml_path)])
    run_musescore(["-S", str(STYLE), "-r", str(dpi), "-o",
                   str(stem.with_suffix(".png")), str(xml_path)])

    pages = [load_gray(p) for p in sorted(
        out_dir.glob(f"{xml_path.stem}-*.png"),
        key=lambda p: int(p.stem.rsplit("-", 1)[1]))]
    if not pages:
        raise Unconvertible("MuseScore produced no PNG pages")
    return pages, mpos_systems(mpos)


def mpos_systems(mpos_path: Path) -> list[list[int]]:
    """The .mpos file -> measure indices grouped into systems, reading order.

    One <element> per measure, carrying page and y. Measures sharing both sit
    on the same system; the ids are the measure order, and a gap in them would
    mean the file does not describe the score we think it does.
    """
    root = ET.fromstring(mpos_path.read_text(encoding="utf-8"))
    rows = []
    for el in root.iter("element"):
        rows.append((int(el.get("id")), int(el.get("page")),
                     round(float(el.get("y")), 1)))
    rows.sort()
    if [r[0] for r in rows] != list(range(len(rows))):
        raise Unconvertible("mpos measure ids are not contiguous")

    systems, key = [], None
    for mid, page, y in rows:
        if (page, y) != key:
            systems.append([])
            key = (page, y)
        systems[-1].append(mid)
    return systems


# --------------------------------------------------------------- ground truth
_LETTERS = "CDEFGAB"

# Notated duration -> kern reciprocal. Tuplet ratios multiply on top.
_TYPE_RECIP = {"breve": 0, "whole": 1, "half": 2, "quarter": 4, "eighth": 8,
               "16th": 16, "32nd": 32, "64th": 64}

_SHARPS = ["f#", "c#", "g#", "d#", "a#", "e#", "b#"]
_FLATS = ["b-", "e-", "a-", "d-", "g-", "c-", "f-"]


def key_token(fifths: int) -> str:
    if fifths > 0:
        return "*k[" + "".join(_SHARPS[:min(fifths, 7)]) + "]"
    if fifths < 0:
        return "*k[" + "".join(_FLATS[:min(-fifths, 7)]) + "]"
    return "*k[]"


def clef_token(clef) -> str:
    sign = getattr(clef, "sign", None)
    line = getattr(clef, "line", None)
    octave = getattr(clef, "octaveChange", 0) or 0
    if sign == "G" and line == 2:
        return "*clefGv2" if octave == -1 else "*clefG2"
    if sign == "F" and line == 4:
        return "*clefF4"
    if sign == "C" and line in (3, 4):
        return f"*clefC{line}"
    raise Unconvertible(f"clef {sign}{line} oct {octave}")


def kern_pitch(p) -> str:
    """music21 pitch -> kern spelling: letter run for the octave, then alter.

    The kern convention the whole corpus uses: the accidental suffix is the
    *sounding* alteration (`f#` in A major carries no printed sharp but always
    the `#`), and `n` appears only where an explicit natural is drawn.
    """
    letter = p.step
    if letter not in _LETTERS:
        raise Unconvertible(f"pitch step {letter!r}")
    octave = p.octave
    if octave is None:
        raise Unconvertible("pitch without octave")
    name = letter.lower() * (octave - 3) if octave >= 4 \
        else letter.upper() * (4 - octave)
    alter = int(p.alter or 0)
    if p.alter != alter:
        raise Unconvertible(f"microtonal alter {p.alter}")
    if alter > 0:
        return name + "#" * alter
    if alter < 0:
        return name + "-" * (-alter)
    if p.accidental is not None and p.accidental.name == "natural" \
            and p.accidental.displayStatus is not False:
        return name + "n"
    return name


def duration_code(d) -> str:
    """music21 duration -> kern code, from the *notated* type, not the length.

    Going through quarterLength would lose the distinction between an eighth
    and a dotted triplet quarter; the notated type is what is drawn.
    """
    if d.isGrace:
        raise Unconvertible("grace note")
    if d.type not in _TYPE_RECIP:
        raise Unconvertible(f"duration type {d.type!r}")
    recip = Fraction(_TYPE_RECIP[d.type])
    for tup in d.tuplets:
        recip = recip * Fraction(tup.numberNotesActual, tup.numberNotesNormal)
    if recip.denominator != 1:
        raise Unconvertible(f"tuplet gives non-integer reciprocal {recip}")
    return f"{recip.numerator}{'.' * d.dots}"


def note_token(n) -> str:
    """One music21 note or rest -> the whole kern token the corpus would carry.

    Sub-token order follows the Bach corpus: opening tie bracket, duration,
    pitch, accidental, beams, closing tie mark, fermata.
    """
    code = duration_code(n.duration)
    if n.isRest:
        tok = code + "r"
    elif n.isChord:
        raise Unconvertible("chord in a monophonic part")
    else:
        tok = code + kern_pitch(n.pitch)
        beams = ""
        for b in n.beams.beamsList:
            if b.type == "start":
                beams += "L"
            elif b.type == "stop":
                beams += "J"
            elif b.type == "partial":
                beams += "k"
        # kern writes multi-beam marks as one run per kind: LL, not LJ mixes.
        beams = "".join(sorted(beams, key=lambda c: c != "L"))
        tok += beams
        if n.tie is not None:
            if n.tie.type == "start":
                tok = "[" + tok
            elif n.tie.type == "stop":
                tok = tok + "]"
            elif n.tie.type == "continue":
                tok = tok + "_"
    from music21 import expressions
    if any(isinstance(e, expressions.Fermata) for e in n.expressions):
        tok += ";"
    return tok


def _is_repeat(bar, direction: str) -> bool:
    from music21 import bar as m21bar

    return isinstance(bar, m21bar.Repeat) and bar.direction == direction


def barline_token(measure, next_left_repeat: bool = False) -> str:
    """Right barline of one measure -> kern token, repeat-aware.

    A music21 Repeat carries type 'final' by default, so the type alone would
    silently label a drawn repeat sign as a final barline (the chor004 finding
    in uebergabe-2026-07-30-nacht.md). The class decides first, the style
    only for plain barlines. A start repeat on the *next* measure is drawn at
    this very barline position when both sit on one system, so it folds into
    this token, as kern spells it: `=:|!|:`.
    """
    bar = measure.rightBarline
    if _is_repeat(bar, "start"):
        raise Unconvertible("start repeat as a right barline")
    if _is_repeat(bar, "end"):
        core = ":|!"
    else:
        style = getattr(bar, "type", None)
        if style in (None, "regular", "short", "tick"):
            core = ""
        elif style == "final":
            core = "="
        elif style in ("double", "light-light"):
            core = "||"
        else:
            raise Unconvertible(f"barline {style!r}")
    if next_left_repeat:
        if core not in ("", ":|!"):
            raise Unconvertible(f"start repeat after {core!r} barline")
        core += "|:" if core else "!|:"
    return "=" + core


def part_measures(part) -> list[dict]:
    """One part -> per-measure records: tokens, barlines, header changes."""
    from music21 import stream

    out = []
    for m in part.getElementsByClass(stream.Measure):
        if m.voices:
            raise Unconvertible("measure with multiple voices in one part")
        notes = []
        last_end = Fraction(0)
        for n in m.notesAndRests:
            off = Fraction(n.offset)
            if off < last_end:
                raise Unconvertible("two notes sound at once in one part")
            last_end = off + Fraction(n.duration.quarterLength)
            notes.append(note_token(n))
        out.append({
            "notes": notes,
            "measure": m,                         # barline needs its neighbour
            "left_repeat": _is_repeat(m.leftBarline, "start"),
            "clef": m.clef,                       # None on most measures
            "key": m.keySignature,                # set where a signature prints
            "meter": m.timeSignature,
        })
    return out


def system_tokens(measures: list[dict], systems: list[list[int]]
                  ) -> list[list[str]]:
    """Per-measure records of one part -> whole-token sequence per system.

    Header logic mirrors what MuseScore draws: clef and key signature reprint
    at every system start (an empty signature prints nothing and gets no
    token), the meter only where the file states one. The barline closing a
    measure belongs to the line it ends on, exactly as in the verovio builder.
    A start repeat merges into the barline before it when both share a system
    (`=:|!|:`); across a system break MuseScore draws it again at the new
    line's start, so there it becomes that line's own `=!|:` token.
    """
    if len(measures) != sum(len(s) for s in systems):
        raise Unconvertible(f"{len(measures)} measures in the part, "
                            f"{sum(len(s) for s in systems)} in the mpos")
    clef = key = None
    fifths = 0
    lines = []
    for system in systems:
        toks: list[str] = []
        for pos, mid in enumerate(system):
            rec = measures[mid]
            if rec["clef"] is not None:
                clef = rec["clef"]
            if rec["key"] is not None:
                key = rec["key"]
                fifths = key.sharps or 0
            if pos == 0:
                if clef is None:
                    raise Unconvertible("system starts with no known clef")
                toks.append(clef_token(clef))
                if fifths:
                    toks.append(key_token(fifths))
                if rec["meter"] is not None:
                    toks.append(f"*M{rec['meter'].ratioString}")
                if rec["left_repeat"]:
                    toks.append("=!|:")
            else:
                if rec["key"] is not None and fifths:
                    toks.append(key_token(fifths))
                if rec["meter"] is not None:
                    toks.append(f"*M{rec['meter'].ratioString}")
            toks.extend(rec["notes"])
            same_system = pos + 1 < len(system)
            toks.append(barline_token(
                rec["measure"],
                same_system and measures[system[pos + 1]]["left_repeat"]))
        lines.append(toks)
    return lines


def with_separators(tokens: list[str]) -> list[str]:
    """Interleave the `<b>` record separator, as the training corpus stores it."""
    out: list[str] = []
    for t in tokens:
        if out:
            out.append("<b>")
        out.append(t)
    return out


def ground_truth(xml_path: Path, systems: list[list[int]]) -> list[list[list[str]]]:
    """The rendered MusicXML -> tokens[system][voice], voices top to bottom.

    Parsed from the very file MuseScore drew, so image and target can only
    disagree if MuseScore deviates from its input -- the case the probes are
    there to catch, not this function.
    """
    from music21 import converter

    score = converter.parse(str(xml_path))
    parts = list(score.parts)
    if not parts:
        raise Unconvertible("no parts")
    per_part = [system_tokens(part_measures(p), systems) for p in parts]
    return [[with_separators(per_part[v][s]) for v in range(len(parts))]
            for s in range(len(systems))]
