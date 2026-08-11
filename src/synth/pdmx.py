"""PDMX score JSON -> 2-8-spine Humdrum kern.

The phase-2 handoff proposed widening the corpus through PDMX by loading its
MusicXML into verovio and taking Humdrum back out. That route does not exist:
**PDMX ships no MusicXML.** Its scores are MusPy-style event JSON -- notes with
a MIDI number, an onset and a duration in ticks, plus lyrics, chords, key and
time signatures and barlines. There is nothing to round-trip.

What is there is enough to engrave from, and that is what this module does.
Everything the representation does not carry -- rests, ties across barlines,
beams, clefs -- is derived here. That sounds like a lot of invention, and it
would be a problem if the ground truth came from anywhere else. It does not:
the kern produced here is what verovio renders *and* what the target tokens are
read from, so the two agree by construction, exactly as for the Bach corpus.
`15_count_check.py` is the same acceptance test either way.

What the conversion refuses, it refuses loudly (`Unconvertible`), and
`19_pdmx_to_kern.py` counts the reasons.

Training data only. Never shipped.
"""
import bisect
import json
from fractions import Fraction
from pathlib import Path

# kern duration codes. A code r means 4/r quarter notes; one dot adds a half,
# two dots three quarters. 3, 6, 12, 24 cover the triplet family.
#
# What may be *written* is narrower than what is arithmetically expressible, and
# the difference is not cosmetic. A dotted triplet is never needed -- `12.` is
# exactly an eighth -- so it can only ever arise from source ticks that fit no
# printable note, and then it prints something no engraver would print. Worse,
# verovio dies on it: `6..B_` beamed to `48B]` and `8G` (pdmx01200, bar 22)
# raises 0xC0000005 and takes the process down, uncatchable, killing a build
# that has been running for hours. Restricting the vocabulary means such a
# length becomes unrepresentable, `duration_tokens` raises, and the piece is
# rejected and counted like any other -- which is what should have happened.
# 256 of 3734 pieces (6.9 %) fall out this way.
_PLAIN_RECIPS = [1, 2, 4, 8, 16, 32]      # dots allowed: 0, 1, 2
_TUPLET_RECIPS = [3, 6, 12, 24]           # dots allowed: 0
_DURATIONS: list[tuple[Fraction, str]] = []
for _r, _max_dots in ([(r, 2) for r in _PLAIN_RECIPS]
                      + [(r, 0) for r in _TUPLET_RECIPS]):
    for _dots in range(_max_dots + 1):
        _q = Fraction(4, _r) * (2 - Fraction(1, 2 ** _dots))
        _DURATIONS.append((_q, f"{_r}{'.' * _dots}"))

# Several codes can mean the same length -- an eighth is `8`, and also `12.`,
# a dotted triplet quarter. Only the first is what an engraver writes; the
# second makes verovio draw a tuplet bracket over every note on the page
# (seen, and it is unmistakable). So ties are broken towards the fewest dots
# and then the plainest code, both in the lookup and in the greedy split.
_DURATIONS.sort(key=lambda t: (-t[0], t[1].count("."), int(t[1].rstrip("."))))
_BY_QUARTERS: dict[Fraction, str] = {}
for _q, _code in sorted(_DURATIONS,
                        key=lambda t: (t[1].count("."), int(t[1].rstrip(".")))):
    _BY_QUARTERS.setdefault(_q, _code)

_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Sharps and flats in the order a key signature prints them.
_SHARPS = ["f#", "c#", "g#", "d#", "a#", "e#", "b#"]
_FLATS = ["b-", "e-", "a-", "d-", "g-", "c-", "f-"]


class Unconvertible(Exception):
    """This piece is not 2-8-part monophonic vocal music we can engrave."""


# The real repertoire (befund-chorrepertoire.md): 39 % of the engraved scores
# are five- to seven-part. The synthesis has to know those layouts, or the
# training corpus misses 39 % of the target page shapes -- decision 8 in
# CLAUDE.md and §3 of design-korpus-neuaufbau.md.
MIN_VOICES, MAX_VOICES = 2, 8


def kern_pitch(midi: int, spelled: str) -> str:
    """MIDI number plus a spelled name ("Bb", "F#") -> a kern pitch token.

    The octave comes from the spelling, not from the MIDI number alone: B#3 and
    C4 are the same key and different letters, and writing the wrong one would
    put the notehead on the wrong staff line.
    """
    letter = spelled[0].upper()
    if letter not in _LETTER_PC:
        raise Unconvertible(f"unspelled pitch {spelled!r}")
    alter = spelled.count("#") - spelled.count("b") + spelled.count("-")
    if spelled[1:].startswith("b"):                 # "Bb" -- the b is a flat
        alter = -spelled[1:].count("b")
    octave, rest = divmod(midi - (_LETTER_PC[letter] + alter), 12)
    if rest:
        raise Unconvertible(f"spelling {spelled!r} does not match midi {midi}")
    octave -= 1
    acc = "#" * alter if alter > 0 else "-" * (-alter)

    if octave >= 4:
        name = letter.lower() * (octave - 3)
    else:
        name = letter.upper() * (4 - octave)
    return name + acc


def duration_tokens(quarters: Fraction) -> list[str]:
    """A length in quarter notes -> the kern duration codes it needs.

    One code if the length is writable as a single note, otherwise the greedy
    decomposition into tied notes that an engraver would also write.
    """
    if quarters <= 0:
        raise Unconvertible("non-positive duration")
    if quarters in _BY_QUARTERS:
        return [_BY_QUARTERS[quarters]]

    out, left = [], quarters
    for q, code in _DURATIONS:
        while left >= q:
            out.append(code)
            left -= q
        if left == 0:
            break
    if left != 0:
        raise Unconvertible(f"duration {left} not representable")
    if len(out) > 4:
        raise Unconvertible(f"duration {quarters} needs {len(out)} tied notes")
    return out


def key_token(fifths: int | None) -> str:
    if not fifths:
        return "*k[]"
    if fifths > 0:
        return "*k[" + "".join(_SHARPS[:min(fifths, 7)]) + "]"
    return "*k[" + "".join(_FLATS[:min(-fifths, 7)]) + "]"


def clef_token(pitches: list[int], voice_from_top: int, n_voices: int) -> str:
    """Bass clef low down, the octave-transposing G clef for the tenor.

    The tenor line is the one the page-wise model got wrong, so it is not left
    to a pitch threshold alone: in a four-part setting the second voice from the
    bottom is the tenor and gets `*clefGv2` whenever it sits in tenor range.
    In other layouts (2-8 voices) only the pitch thresholds decide -- there is
    no fixed slot the tenor lives in.
    """
    median = sorted(pitches)[len(pitches) // 2]
    if median < 55:
        return "*clefF4"
    if n_voices == 4 and voice_from_top == 2 and median < 65:
        return "*clefGv2"
    if median < 60:
        return "*clefGv2"
    return "*clefG2"


def _events(track, resolution: int):
    """Notes of one track as (start, end, midi, spelled), monophony enforced."""
    seen = set()
    out = []
    for n in track["notes"]:
        if n.get("is_grace"):
            raise Unconvertible("grace notes")
        key = (n["time"], n["pitch"], n["duration"])
        if key in seen:
            continue
        seen.add(key)
        out.append((n["time"], n["time"] + n["duration"], n["pitch"],
                    n.get("pitch_str") or ""))
    out.sort(key=lambda e: (e[0], e[2]))
    for a, b in zip(out, out[1:]):
        if b[0] < a[1]:
            raise Unconvertible("two notes sound at once in one part")
    return out


def _measure_starts(doc) -> list[int]:
    starts = sorted({b["time"] for b in doc.get("barlines", [])})
    if len(starts) < 4:
        raise Unconvertible("fewer than four barlines")
    return starts


def _lyrics_by_time(track) -> dict[int, str]:
    out = {}
    for ly in track.get("lyrics", []):
        text = (ly.get("lyric") or "").strip()
        if text:
            out[ly["time"]] = text
    return out


def beam_group_ticks(numerator: int, denominator: int, resolution: int) -> int:
    """How wide a beam group is, in ticks.

    Compound meters beam in dotted beats (three eighths in 6/8), simple meters
    in beats. Getting this wrong is visible at a glance, which is the point:
    PDMX carries no beams at all, and unbeamed eighths look nothing like either
    the Bach corpus or the target material.
    """
    if denominator == 8 and numerator % 3 == 0 and numerator > 3:
        return resolution * 3 // 2
    if denominator == 2:
        return resolution * 2
    return resolution


def _apply_beams(records, starts, group_ticks, resolution):
    """Add kern's `L`/`J` beam markers to runs of short notes within a beat.

    Groups are counted from the start of the *bar*, not from the start of the
    piece. Counting from the piece only agrees with the barlines while every bar
    is a whole number of groups long; after a meter change it drifts, and a run
    then continues across the barline. Verovio does not survive that: the beam
    opened at the end of bar 32 of pdmx01200 and closed in bar 33 killed the
    renderer with an access violation (0xC0000005), which no `except` can catch
    because it takes the process down. 258 of 3734 converted pieces (6.9 %) had
    at least one such beam. Bar-relative grouping also puts the groups back on
    the beat after a meter change, which is where an engraver puts them.
    """
    if group_ticks <= 0:
        return records

    def with_mark(tok: str, mark: str) -> str:
        # Beam markers sit before a closing tie bracket: `8f#L]`, as the Bach
        # corpus writes it.
        for suffix in ("]", "_"):
            if tok.endswith(suffix):
                return tok[:-1] + mark + suffix
        return tok + mark

    bounds = sorted(starts)
    out = list(records)
    run: list[int] = []

    def group_of(pos: int) -> tuple[int, int]:
        """(bar index, group index inside that bar) -- a beam may span neither."""
        bar = bisect.bisect_right(bounds, pos) - 1
        if bar < 0:
            return (-1, (pos - bounds[0]) // group_ticks)
        return (bar, (pos - bounds[bar]) // group_ticks)

    def flush():
        if len(run) >= 2:
            out[run[0]] = (out[run[0]][0], out[run[0]][1],
                           with_mark(out[run[0]][2], "L"), out[run[0]][3])
            out[run[-1]] = (out[run[-1]][0], out[run[-1]][1],
                            with_mark(out[run[-1]][2], "J"), out[run[-1]][3])
        run.clear()

    prev_group = None
    for i, (pos, ticks, tok, is_rest) in enumerate(out):
        group = group_of(pos)
        # Eighths and shorter get beamed, whatever the meter. Tying this to the
        # group width instead would beam quarter notes in 2/2.
        beamable = (not is_rest) and ticks * 2 <= resolution
        if not beamable or group != prev_group:
            flush()
        if beamable:
            run.append(i)
            prev_group = group
        else:
            prev_group = None
    flush()
    return out


def _spine_records(events, starts, resolution, end_time, group_ticks):
    """One kern spine as a list of (time, token) records, rests filled in.

    Rests are not in the source at all -- PDMX only stores notes -- so every gap
    between the end of one note and the start of the next becomes a rest, and a
    note that runs over a barline is split into tied parts. Both are decisions
    an engraver makes too; what matters is that the same decision produces the
    image and the target.
    """
    bounds = list(starts) + [end_time]
    records: list[tuple[int, int, str, bool]] = []   # pos, ticks, token, is_rest

    def ticks_of(code: str) -> int:
        dots = code.count(".")
        recip = int(code.rstrip("."))
        return int(Fraction(4 * resolution, recip) * (2 - Fraction(1, 2 ** dots)))

    def emit(t0: int, t1: int, pitch_token: str | None):
        """Fill [t0, t1) with notes or rests, broken at every barline."""
        pieces, t = [], t0
        for b in bounds:
            if t0 < b < t1:
                pieces.append((t, b))
                t = b
        pieces.append((t, t1))
        for k, (a, b) in enumerate(pieces):
            codes = duration_tokens(Fraction(b - a, resolution))
            pos = a
            for j, code in enumerate(codes):
                ticks = ticks_of(code)
                if pitch_token is None:
                    records.append((pos, ticks, f"{code}r", True))
                else:
                    first = (k == 0 and j == 0)
                    last = (k == len(pieces) - 1 and j == len(codes) - 1)
                    tok = f"{code}{pitch_token}"
                    if first and not last:
                        tok = "[" + tok
                    elif last and not first:
                        tok = tok + "]"
                    elif not first and not last:
                        tok = tok + "_"
                    records.append((pos, ticks, tok, False))
                pos += ticks

    cursor = starts[0]
    for start, end, midi, spelled in events:
        if start < cursor:
            continue
        if start > cursor:
            emit(cursor, start, None)
        emit(start, end, kern_pitch(midi, spelled))
        cursor = end
    if cursor < end_time:
        emit(cursor, end_time, None)
    return _apply_beams(records, starts, group_ticks, resolution)


def convert(path: Path) -> tuple[str, dict]:
    """PDMX json file -> (kern text, stats). Raises Unconvertible."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    resolution = doc.get("resolution")
    if resolution != 480:
        raise Unconvertible(f"resolution {resolution}")

    tracks = [t for t in doc.get("tracks", []) if not t.get("is_drum")]
    n = len(tracks)
    if not MIN_VOICES <= n <= MAX_VOICES:
        raise Unconvertible(f"{n} pitched tracks, want {MIN_VOICES}-{MAX_VOICES}")

    metas = sorted(doc.get("time_signatures") or [], key=lambda m: m["time"])
    if not metas:
        raise Unconvertible("no time signature")
    # Meter changes are kept rather than rejected: they were a quarter of all
    # rejections in the first run, and the Bach corpus contains none at all, so
    # this is the only source of that case in the whole training set.
    meter_at = {m["time"]: f"*M{m['numerator']}/{m['denominator']}" for m in metas}
    meter = meter_at[metas[0]["time"]]

    keys = doc.get("key_signatures") or []
    key = key_token(keys[0].get("fifths") if keys else 0)

    starts = _measure_starts(doc)
    per_track = [_events(t, resolution) for t in tracks]
    if any(len(e) < 8 for e in per_track):
        raise Unconvertible("a part has fewer than eight notes")

    end_time = max(e[-1][1] for e in per_track)
    if end_time <= starts[0]:
        raise Unconvertible("no music after the first barline")

    # Order the parts from lowest to highest, which is the order Humdrum spines
    # are written in, and check they really are separate voices. The register
    # check keeps instrumental textures out: a choral setting of three or more
    # voices has *some* part below G4. Duets (SA and the like) legitimately may
    # not, so for two voices the check is skipped.
    order = sorted(range(n), key=lambda i: sorted(
        p for _, _, p, _ in per_track[i])[len(per_track[i]) // 2])
    medians = [sorted(p for _, _, p, _ in per_track[i])[len(per_track[i]) // 2]
               for i in order]
    if n >= 3 and medians[0] > 67:
        raise Unconvertible("no part below G4 -- not a choral layout")

    group_ticks = beam_group_ticks(metas[0]["numerator"],
                                   metas[0]["denominator"], resolution)
    spines = []
    lyrics = []
    for slot, i in enumerate(order):
        spines.append(_spine_records(per_track[i], starts, resolution, end_time,
                                     group_ticks))
        lyrics.append(_lyrics_by_time(tracks[i]))
    clefs = [clef_token([p for _, _, p, _ in per_track[i]], n - 1 - slot, n)
             for slot, i in enumerate(order)]

    times = sorted({t for sp in spines for t, _, _, _ in sp} | set(starts)
                   | set(meter_at))
    times = [t for t in times if t >= starts[0]]
    bar_at = {t: k + 1 for k, t in enumerate(starts)}

    rows = ["\t".join(["**kern"] * n),
            "\t".join(clefs),
            "\t".join([key] * n),
            "\t".join([meter] * n)]

    by_time = [{t: tok for t, _, tok, _ in sp} for sp in spines]
    for t in times:
        if t in bar_at and t != starts[0]:
            rows.append("\t".join([f"={bar_at[t]}"] * n))
        if t in meter_at and t != metas[0]["time"]:
            rows.append("\t".join([meter_at[t]] * n))
        row = [by_time[s].get(t, ".") for s in range(n)]
        if all(c == "." for c in row):
            continue
        rows.append("\t".join(row))
    rows.append("\t".join(["=="] * n))
    rows.append("\t".join(["*-"] * n))

    stats = {
        "title": (doc.get("metadata") or {}).get("title"),
        "creators": (doc.get("metadata") or {}).get("creators"),
        "voices": n,
        "measures": len(starts),
        "notes": sum(len(e) for e in per_track),
        "meter": meter, "meter_changes": len(meter_at) - 1,
        "key": key, "clefs": clefs,
        "lyric_syllables": [len(x) for x in lyrics],
    }
    return "\n".join(rows), stats
