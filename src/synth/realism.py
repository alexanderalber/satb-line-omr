"""Make the rendered training pages carry the same kinds of ink as real scores.

The rule this module exists for, from the phase-2 handoff:

    The training images must contain the same categories of ink that the crop
    lets in at run time.

The detector's adaptive padding was measured against the real SATB score and
found to let dynamics and hairpins into the crop. Bach chorales have none, so
the model would meet a class of symbol at run time it had never seen -- not by
accident but by our own cropping decision. Lyrics matter even more, and not
because they sit under the staff: verovio pulls the notes apart so the syllables
fit, so the text changes the picture *inside* the crop.

Everything here is added as extra Humdrum spines. That keeps the ground truth
untouched by construction: what appears is ink with no target token, exactly
like a barline glyph already is. The kern spines keep their own field indices,
which is what `render.parse_structure` addresses them by -- the indices shift,
but the shift is known because we build the file.

Spelling of the added spines is not guessed. `scripts/_probe_realism3.py`
renders each candidate and counts the elements verovio drew:

    **dynam  p / mf / f            -> <dynam>
    **dynam  `<` ... `[`           -> <hairpin>   (crescendo, open and close)
    **dynam  `>` ... `]`           -> <hairpin>   (decrescendo)
    **mxhm   C / G7 / Am           -> <harm>
    !!!OMD                         -> <tempo>
    *I"Soprano                     -> <label>

Training data only; nothing here ships to the browser.
"""
import random
import re
from dataclasses import dataclass, field

from .tokens import token_pitch

# German hymn verses, hyphenated at the syllable boundaries an engraver would
# use. Real text rather than nonsense because the point of the lyrics is their
# *width*: syllable length and hyphenation decide how far verovio pulls the
# notes apart, and generated gibberish would get that distribution wrong.
# All of these are 16th-18th century and long out of copyright.
HYMN_LINES = [
    "Aus mei-nes Her-zens Grun-de sag ich dir Lob und Dank",
    "Nun ru-hen al-le Wäl-der Vieh Men-schen Städt und Fel-der",
    "Wach-et auf ruft uns die Stim-me der Wäch-ter sehr hoch auf der Zin-ne",
    "O Haupt voll Blut und Wun-den voll Schmerz und vol-ler Hohn",
    "Lo-be den Her-ren den mäch-ti-gen Kö-nig der Eh-ren",
    "Ein fes-te Burg ist un-ser Gott ein gu-te Wehr und Waf-fen",
    "Je-su mei-ne Freu-de mei-nes Her-zens Wei-de",
    "Wie schön leuch-tet der Mor-gen-stern voll Gnad und Wahr-heit von dem Herrn",
    "Nun danket al-le Gott mit Her-zen Mund und Hän-den",
    "Christ lag in To-des-ban-den für un-sre Sünd ge-ge-ben",
    "Vom Him-mel hoch da komm ich her ich bring euch gu-te neu-e Mär",
    "Herz-lich tut mich ver-lan-gen nach ei-nem sel-gen End",
    "Gott des Him-mels und der Er-den Va-ter Sohn und Hei-lger Geist",
    "Al-lein Gott in der Höh sei Ehr und Dank für sei-ne Gna-de",
    "Der Mond ist auf-ge-gan-gen die gold-nen Stern-lein pran-gen",
    "Wer nur den lie-ben Gott lässt wal-ten und hof-fet auf ihn al-le-zeit",
]

# Voice names bottom staff first, one layout per part count, following the
# splits the real repertoire uses (SA duet, SAB, SATB, SSATB, SSATBB,
# SSAATBB, SSAATTBB). The names are ink without a target token -- what matters
# is that a plausible word sits in the left margin, not which one.
VOICE_NAMES = {
    2: ["Alt", "Sopran"],
    3: ["Bass", "Alt", "Sopran"],
    4: ["Bass", "Tenor", "Alto", "Soprano"],
    5: ["Bass", "Tenor", "Alt", "Sopran 2", "Sopran 1"],
    6: ["Bass 2", "Bass 1", "Tenor", "Alt", "Sopran 2", "Sopran 1"],
    7: ["Bass 2", "Bass 1", "Tenor", "Alt 2", "Alt 1", "Sopran 2", "Sopran 1"],
    8: ["Bass 2", "Bass 1", "Tenor 2", "Tenor 1", "Alt 2", "Alt 1",
        "Sopran 2", "Sopran 1"],
}
VOICE_ABBR = {
    2: ["A.", "S."],
    3: ["B.", "A.", "S."],
    4: ["B.", "T.", "A.", "S."],
    5: ["B.", "T.", "A.", "S. 2", "S. 1"],
    6: ["B. 2", "B. 1", "T.", "A.", "S. 2", "S. 1"],
    7: ["B. 2", "B. 1", "T.", "A. 2", "A. 1", "S. 2", "S. 1"],
    8: ["B. 2", "B. 1", "T. 2", "T. 1", "A. 2", "A. 1", "S. 2", "S. 1"],
}

DYNAMICS = ["pp", "p", "mp", "mf", "f", "ff"]

# Chord symbols as MusicXML harmony spells them, which is what `**mxhm` takes
# and what a pop choral score prints above the top staff.
CHORDS = ["C", "Dm", "Em", "F", "G", "G7", "Am", "B-", "D", "A7", "E7",
          "Bm", "F#m", "Csus4", "Gsus4", "Fmaj7", "Cm", "E-", "A-", "D7"]

TEMPI = ["Andante", "Moderato", "Allegro", "Adagio", "Largo",
         "Andante con moto", "Allegretto", "Ruhig", "Fließend", "Getragen"]

# Verovio ships several SMuFL fonts. The model must not learn one engraver's
# glyph shapes when it will meet Sibelius, Finale or MuseScore at run time.
FONTS = ["Leipzig", "Bravura", "Gootville", "Petaluma", "Leland"]

_TIE_CONT = re.compile(r"[\]_]")


@dataclass
class RealismConfig:
    """Which extras this piece gets. Not everything on every page: the target
    material mixes staves with and without lyrics, so the model has to keep
    handling the plain case too."""
    lyrics: bool = False
    labels: bool = False
    tempo: bool = False
    harmony: bool = False
    dynamic_voices: tuple[int, ...] = ()      # indices into the kern spines
    font: str = "Leipzig"
    hymn: int = 0
    spacing_staff: int = 16
    scale: int = 100
    verses: int = 1

    def as_dict(self) -> dict:
        return {"lyrics": self.lyrics, "labels": self.labels,
                "tempo": self.tempo, "harmony": self.harmony,
                "dynamic_voices": list(self.dynamic_voices),
                "font": self.font, "hymn": self.hymn,
                "spacing_staff": self.spacing_staff, "scale": self.scale,
                "verses": self.verses}

    def render_options(self) -> dict:
        return {"font": self.font, "spacingStaff": self.spacing_staff,
                "lyricTopMinMargin": 4, "scale": self.scale}


def choose(rng: random.Random, n_spines: int = 4) -> RealismConfig:
    """A random combination, weighted towards what the target material shows."""
    return RealismConfig(
        lyrics=rng.random() < 0.75,
        labels=rng.random() < 0.55,
        tempo=rng.random() < 0.45,
        harmony=rng.random() < 0.35,
        dynamic_voices=tuple(i for i in range(n_spines) if rng.random() < 0.25),
        font=rng.choice(FONTS),
        hymn=rng.randrange(len(HYMN_LINES)),
        # Verovio's default packs the staves 10.0 line spacings apart. The real
        # SATB score measures 12.5 to 15.8 (`_probe_geometry.py`), and the
        # difference is not cosmetic: the detector's padding stops at the first
        # free gap of 0.8 spacings, so on the tighter page the lyrics of the
        # neighbouring voice end up *inside* the crop -- ink the crop at run
        # time does not contain. spacingStaff n gives a pitch of 4 + n/2.
        spacing_staff=rng.randrange(16, 24, 2),
        # Mild scale variation only. The scope is clean digital engraving, so
        # noise and warping stay out; the staff size does vary between
        # publishers and the normalisation has to cope with it.
        scale=rng.choice([70, 80, 90, 100, 110]),
    )
    # `verses` is NOT drawn here: any extra draw on this rng would shift the
    # hairpin and chord placements in `augment` and silently change every
    # rendered image. The verse count comes from its own stream, set by
    # `corpus.prepare` -- and only when the caller opts in (pop package).


def syllables(hymn_index: int) -> list[str]:
    """One hymn line as Humdrum `**text` syllables.

    Humdrum marks word continuation with hyphens on both sides of the break:
    `mei-` then `-nes`. Verovio turns that into two `<syl>` elements joined by a
    printed hyphen, which is what makes the word occupy the width it does.
    """
    out: list[str] = []
    for word in HYMN_LINES[hymn_index].split():
        parts = word.split("-")
        if len(parts) == 1:
            out.append(word)
            continue
        for i, part in enumerate(parts):
            s = part
            if i > 0:
                s = "-" + s
            if i < len(parts) - 1:
                s = s + "-"
            out.append(s)
    return out


def _is_sung(tok: str) -> bool:
    """Does this kern token get a syllable of its own?

    Notes do, rests and null tokens do not, and neither does the continuation
    or the end of a tie -- the word is held over, it is not sung again.
    """
    pitch = token_pitch(tok)
    if pitch is None or pitch == "r":
        return False
    return not _TIE_CONT.search(tok)


def _hairpin_plan(rng: random.Random, n_events: int) -> dict[int, str]:
    """Where the dynamics and hairpins of one voice go.

    `<` opens a crescendo and `[` closes it, `>` opens a decrescendo and `]`
    closes it -- the asymmetry is verovio's, measured, not assumed.
    """
    plan: dict[int, str] = {}
    if n_events < 6:
        return plan
    pos = rng.randrange(0, max(1, n_events // 4))
    while pos < n_events - 2:
        kind = rng.random()
        if kind < 0.45:
            plan[pos] = rng.choice(DYNAMICS)
            pos += rng.randrange(4, 12)
        else:
            span = rng.randrange(2, 6)
            if pos + span >= n_events:
                break
            open_, close = ("<", "[") if rng.random() < 0.5 else (">", "]")
            plan[pos] = open_
            plan[pos + span] = close
            pos += span + rng.randrange(2, 8)
    return plan


@dataclass
class Augmented:
    text: str
    kern_fields: list[int]        # 1-based field index of each original spine
    config: RealismConfig
    n_fields: int
    stats: dict = field(default_factory=dict)


def augment(text: str, cfg: RealismConfig, rng: random.Random) -> Augmented:
    """Insert the extra spines into a 4-spine kern score.

    The original spines keep their order, so `kern_fields[i]` is where spine i
    (0-based, left to right, i.e. bass to soprano) now lives.
    """
    lines = [ln for ln in text.split("\n") if ln]

    # Field layout. Humdrum orders spines left to right = bottom staff to top,
    # so the last kern spine is the soprano; harmony sits with it.
    exclusive = next(ln for ln in lines if ln.startswith("**"))
    n_kern = len(exclusive.split("\t"))

    # (role, kern spine it belongs to, verse). Verse is 0 except for the text
    # spines: adjacent `**text` spines render as stacked verses, top to bottom
    # in spine order (measured, `_probe_verses.py`).
    layout: list[tuple[str, int, int]] = []
    for i in range(n_kern):
        layout.append(("kern", i, 0))
        if i in cfg.dynamic_voices:
            layout.append(("dynam", i, 0))
        if cfg.lyrics:
            for v in range(cfg.verses):
                layout.append(("text", i, v))
        if cfg.harmony and i == n_kern - 1:
            layout.append(("harm", i, 0))

    kern_fields = [pos + 1 for pos, (role, _, _) in enumerate(layout)
                   if role == "kern"]

    # Pass one: collect, per kern spine, the data records that carry a sung note.
    # The syllables and dynamics are laid out over those, so they land on
    # noteheads rather than on thin air.
    sung: dict[int, list[int]] = {i: [] for i in range(n_kern)}
    events: dict[int, list[int]] = {i: [] for i in range(n_kern)}
    for idx, ln in enumerate(lines):
        if ln.startswith(("!", "*", "=")):
            continue
        for i, tok in enumerate(ln.split("\t")[:n_kern]):
            if tok == ".":
                continue
            events[i].append(idx)
            if _is_sung(tok):
                sung[i].append(idx)

    # Each verse gets its own hymn line -- offset from cfg.hymn rather than
    # freshly drawn, so the rng stream stays independent of the verse count.
    syl_at: dict[tuple[int, int], dict[int, str]] = {}
    for v in range(cfg.verses):
        syl = syllables((cfg.hymn + v) % len(HYMN_LINES))
        for i in range(n_kern):
            syl_at[(i, v)] = {idx: syl[k % len(syl)]
                              for k, idx in enumerate(sung[i])}

    dyn_at: dict[int, dict[int, str]] = {}
    for i in cfg.dynamic_voices:
        plan = _hairpin_plan(rng, len(events[i]))
        dyn_at[i] = {events[i][k]: v for k, v in plan.items()}

    # Chord symbols: on the first note of most bars, above the top staff.
    harm_at: dict[int, str] = {}
    if cfg.harmony:
        top = n_kern - 1
        bar_open = True
        for idx, ln in enumerate(lines):
            if ln.startswith("="):
                bar_open = True
                continue
            if ln.startswith(("!", "*")):
                continue
            if bar_open and idx in set(events[top]):
                if rng.random() < 0.85:
                    harm_at[idx] = rng.choice(CHORDS)
                bar_open = False

    out: list[str] = []
    for idx, ln in enumerate(lines):
        if ln.startswith("!"):
            out.append(ln)
            continue
        fields = ln.split("\t")
        if len(fields) != n_kern:
            out.append(ln)
            continue

        row: list[str] = []
        for role, spine, verse in layout:
            tok = fields[spine]
            if role == "kern":
                row.append(tok)
            elif ln.startswith("**"):
                row.append({"text": "**text", "dynam": "**dynam",
                            "harm": "**mxhm"}[role])
            elif tok == "*-":
                row.append("*-")
            elif ln.startswith("="):
                row.append(tok)
            elif ln.startswith("*"):
                # Section markers have to agree across all spines; everything
                # else on the added spines is a null interpretation.
                row.append(tok if tok.startswith("*>") else "*")
            elif role == "text":
                row.append(syl_at[(spine, verse)].get(idx, "."))
            elif role == "dynam":
                row.append(dyn_at.get(spine, {}).get(idx, "."))
            else:
                row.append(harm_at.get(idx, "."))
        out.append("\t".join(row))

    if cfg.harmony:
        out = _insert_placement(out, layout)
    if cfg.labels:
        out = _insert_labels(out, layout, n_kern)
    if cfg.tempo:
        out = _insert_tempo(out, rng)

    return Augmented(
        text="\n".join(out),
        kern_fields=kern_fields,
        config=cfg,
        n_fields=len(layout),
        stats={"syllables": {i: sum(len(syl_at[(i, v)])
                                    for v in range(cfg.verses)
                                    if (i, v) in syl_at)
                             for i in range(n_kern)},
               "dynamics": {i: len(v) for i, v in dyn_at.items()},
               "chords": len(harm_at)},
    )


def _insert_placement(rows: list[str], layout) -> list[str]:
    """Chord symbols belong above the top staff.

    Verovio's default for `**mxhm` is *below*, where they collide with the
    lyrics -- measured, see `_probe_harm_place.py`. Humdrum's `*above`
    interpretation moves them, and only that spelling works: `*Xabove` and
    `*place:above` leave them where they were.
    """
    marks = ["*above" if role == "harm" else "*" for role, _, _ in layout]
    at = next(i for i, ln in enumerate(rows) if ln.startswith("**")) + 1
    return rows[:at] + ["\t".join(marks)] + rows[at:]


def _insert_labels(rows: list[str], layout, n_kern: int) -> list[str]:
    """Voice names in the left margin, as real choral scores carry them."""
    names, abbrs = [], []
    voice_names = VOICE_NAMES[n_kern]
    voice_abbr = VOICE_ABBR[n_kern]
    for role, spine, _ in layout:
        if role != "kern":
            names.append("*")
            abbrs.append("*")
            continue
        # Spine order is bottom staff first, and so is VOICE_NAMES.
        names.append(f'*I"{voice_names[spine]}')
        abbrs.append(f"*I'{voice_abbr[spine]}")

    at = next(i for i, ln in enumerate(rows) if ln.startswith("**")) + 1
    return rows[:at] + ["\t".join(names), "\t".join(abbrs)] + rows[at:]


def _insert_tempo(rows: list[str], rng: random.Random) -> list[str]:
    """A tempo mark above the first system.

    Inserted directly in front of the exclusive interpretation, never at the top
    of the file: put before the `!!!!SEGMENT` reference record, verovio's
    Humdrum importer crashes (measured -- exit 0xC0000005).

    Text only, no metronome glyph. `[quarter]` produces a SMuFL character that
    resvg has no font for and renders as two missing-glyph boxes -- ink, but ink
    no real score contains, which is worse than leaving it out.
    """
    mark = rng.choice(TEMPI)
    if rng.random() < 0.5:
        mark = f"{mark} (M.M. {rng.choice([56, 63, 72, 84, 92, 96, 108, 120])})"
    at = next(i for i, ln in enumerate(rows) if ln.startswith("**"))
    return rows[:at] + [f"!!!OMD: {mark}"] + rows[at:]
