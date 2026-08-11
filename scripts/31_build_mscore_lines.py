"""The paired MuseScore third of the training corpus.

Design §3 with the review's ruling: one third of the pieces, **paired** --
drawn from pieces whose verovio rendering is also in the corpus, over both
sources. The same piece in two hands is the strongest invariance signal
available without touching the architecture; a disjoint third would just be
more data in another font, the quantity logic that already failed.

Acceptance per piece (the 15_count_check equivalent the review demanded),
every failure a counted rejection, never a silent skip:

1. music21 round trip must not change a note: the kern parsed directly and
   the MusicXML parsed back must agree per part on (pitch, duration, tie),
   rests aside -- the `_probe_musicxml.py` criterion, now enforced per piece.
2. MuseScore's mpos systems must match the shipping detector's systems, and
   every system must show one staff per part.
3. The bar-sum sequence of the derived ground truth must equal, per voice,
   the bar-sum sequence of the accepted verovio ground truth of the same
   piece. This replaces the old interior-bars-sum-to-the-meter rule, which
   was stricter than the corpus it pairs with: 92 % of its 383 rejections
   were pickups, voltas and shortened final bars that the verovio side
   contains as legitimate music (work/_probe_barsums.json). Content parity
   instead of metrical normality -- both acceptance bars at the same
   height; a genuine derivation defect still diverges from the verovio
   sums and is caught (the 32 verovio-clean rejections of the old rule).

Reserve pieces (work/reserve_testset.json) are never built here: they are
the render-axis measuring instrument and must not exist in a training
directory at all.

Writes work/lines_mscore/<piece>/ (same schema as work/lines/) and
work/31_mscore_lines.json.
"""
import json
import random
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from _common import kern_path                                       # noqa: E402
from omr.staves import detect                                       # noqa: E402
from synth import musescore as ms                                   # noqa: E402
from synth.realism import HYMN_LINES                                # noqa: E402

WORK = REPO / "work"
LINES = WORK / "lines"
OUT = WORK / "lines_mscore"
XMLS = WORK / "mscore_xml_corpus"

SEED = 20260730
FRACTION = 1 / 3

_RECIP = re.compile(r"^(\d+)(\.*)")


class Rejected(Exception):
    pass


def picked(piece: str) -> bool:
    """Deterministic per-piece draw, a function of the name alone."""
    return random.Random(f"{SEED}:{piece}").random() < FRACTION


def hymn_of(piece: str) -> int:
    return random.Random(f"{SEED}:hymn:{piece}").randrange(len(HYMN_LINES))


# Pop package (Entscheid 02.08.): verse blocks are opt-in (`--verses`) until
# the s12 decision is through, so a default rebuild stays byte-identical.
MULTI_VERSE = False


def verses_of(piece: str) -> int:
    """Same distribution as `corpus.verses_of` (pop-package decision, 02.08.):
    a third single-verse, the rest 2-4."""
    if not MULTI_VERSE:
        return 1
    return random.Random(f"{SEED}:verses:{piece}").choice([1, 1, 2, 2, 3, 4])


def spine(part):
    """(pitch, quarterLength, tie) per note, rests skipped -- see probe."""
    out = []
    for n in part.recurse().notesAndRests:
        if n.isRest:
            continue
        if n.isChord:
            raise Rejected("chord in a monophonic part")
        out.append((n.pitch.nameWithOctave, float(n.quarterLength),
                    n.tie.type if n.tie else None))
    return out


def check_roundtrip(krn: Path, xml: Path) -> None:
    """Acceptance 1: no note may change between kern and rendered MusicXML."""
    from music21 import converter

    a = converter.parse(str(krn))
    b = converter.parse(str(xml))
    if len(a.parts) != len(b.parts):
        raise Rejected(f"roundtrip changed part count "
                       f"{len(a.parts)} -> {len(b.parts)}")
    for x, y in zip(a.parts, b.parts):
        if spine(x) != spine(y):
            raise Rejected("roundtrip changed notes")


def token_quarters(tok: str) -> Fraction | None:
    m = _RECIP.match(tok)
    if not m:
        return None
    recip = int(m.group(1))
    if recip == 0:
        return Fraction(8)
    return Fraction(4, recip) * (2 - Fraction(1, 2 ** len(m.group(2))))


def bar_sums(tokens: list[str]) -> list[tuple[Fraction, Fraction]]:
    """Token stream -> (total, notes-only) quarter sum per bar.

    The two entries separate music from padding: music21 completes short
    bars with *drawn* rests on the MuseScore side (chor006 final bar
    3 -> 4), so rest durations may legitimately differ between the two
    ground truths while note durations never may.
    """
    sums: list[tuple[Fraction, Fraction]] = []
    cur = notes = Fraction(0)
    for tok in tokens:
        if tok == "<b>" or tok.startswith("*"):
            continue
        if tok.startswith("="):
            sums.append((cur, notes))
            cur = notes = Fraction(0)
            continue
        q = token_quarters(tok)
        if q is not None:
            cur += q
            if "r" not in tok:     # kern spells rests `4r`; no note sub-token uses r
                notes += q
    if cur:
        sums.append((cur, notes))  # a piece may end without a final barline
    # Zero bars are barline spelling, not music: a double bar or a repeat
    # pair (`=:|!` `=!|:`) yields two consecutive `=` tokens on one side
    # and one merged token on the other.
    return [s for s in sums if s[0]]


def verovio_bar_sums(piece: str) -> dict[int, list[Fraction]]:
    """Bar-sum sequence per voice of the accepted verovio corpus lines.

    Numeric sort (the s99/s100 lesson from the 15er): the concatenation
    must follow reading order, not lexicographic file order.
    """
    metas = sorted((LINES / piece).glob("*.json"),
                   key=lambda p: [int(x) for x in re.findall(r"\d+", p.stem)])
    by_voice: dict[int, list[str]] = {}
    for m in metas:
        d = json.loads(m.read_text(encoding="utf-8"))
        by_voice.setdefault(d["voice"], []).extend(d["tokens"])
    return {v: bar_sums(toks) for v, toks in by_voice.items()}


def check_bar_parity(gt, piece: str) -> None:
    """Acceptance 3: bar sums must match the verovio GT of the same piece."""
    ref = verovio_bar_sums(piece)
    n_voices = len(gt[0])
    if sorted(ref) != list(range(n_voices)):
        raise Rejected(f"verovio GT has voices {sorted(ref)}, "
                       f"musescore {n_voices}")
    for voice in range(n_voices):
        mine = bar_sums([t for system in gt for t in system[voice]])
        want = ref[voice]
        if len(mine) != len(want):
            raise Rejected(
                f"bar count diverges from verovio GT: voice {voice}, "
                f"{len(mine)} vs {len(want)} bars")
        cap = max(w for w, _ in want)
        for k, ((a, a_notes), (b, b_notes)) in enumerate(zip(mine, want)):
            if a_notes != b_notes:
                raise Rejected(
                    f"note sums diverge from verovio GT: voice {voice}, "
                    f"bar {k}: {a_notes} vs {b_notes}")
            # Rest padding may only grow a bar, and never past the longest
            # bar the verovio side knows (the meter proxy): a duration
            # defect that overshoots stays a rejection.
            if a != b and not (b < a <= cap):
                raise Rejected(
                    f"bar sums diverge from verovio GT: voice {voice}, "
                    f"bar {k}: {a} vs {b}")


def build_piece(piece: str) -> tuple[list[dict], list[np.ndarray], int]:
    hymn = hymn_of(piece)
    xml = XMLS / f"{piece}.musicxml"
    ms.to_musicxml(kern_path(piece), xml, hymn=hymn, verses=verses_of(piece))
    check_roundtrip(kern_path(piece), xml)

    pages, systems = ms.render(xml, XMLS)
    gt = ms.ground_truth(xml, systems)
    n_voices = len(gt[0])

    detected = []
    for page_idx, gray in enumerate(pages):
        res = detect(gray)
        for sys_local in range(len(res["systems"])):
            boxes = [b for b in res["boxes"] if b["system"] == sys_local]
            detected.append((page_idx, boxes))
    if len(detected) != len(systems):
        raise Rejected(f"detector {len(detected)} systems, mpos {len(systems)}")
    for i, (_, boxes) in enumerate(detected):
        if len(boxes) != n_voices:
            raise Rejected(f"system {i}: {len(boxes)} staves, "
                           f"{n_voices} parts")

    check_bar_parity(gt, piece)
    lines = []
    for sysi, (page_idx, boxes) in enumerate(detected):
        for voice, box in enumerate(boxes):
            lines.append({"page": page_idx, "system": sysi, "voice": voice,
                          "box": box, "tokens": gt[sysi][voice], "hymn": hymn})
    return lines, pages, n_voices


def main():
    global MULTI_VERSE
    MULTI_VERSE = "--verses" in sys.argv
    limit = int(sys.argv[1]) if len(sys.argv) > 1 \
        and not sys.argv[1].startswith("--") else None
    # --shard=K/N as in 10_build_lines.py: deterministic partition of the
    # chosen list, one worker per shard. MuseScore runs as an external process
    # and its crashes surface as rejections, so no quarantine machinery here.
    shard = next((a.split("=", 1)[1] for a in sys.argv
                  if a.startswith("--shard=")), None)
    shard_k, shard_n = (int(x) for x in shard.split("/")) if shard else (0, 1)
    summary_path = (WORK / f"31_mscore_lines_s{shard_k}.json") if shard \
        else (WORK / "31_mscore_lines.json")
    OUT.mkdir(exist_ok=True)
    XMLS.mkdir(exist_ok=True)

    reserve = json.loads((WORK / "reserve_testset.json")
                         .read_text(encoding="utf-8"))
    blocked = set(reserve["bach"]) | set(reserve["pdmx"])

    pool = sorted(p.name for p in LINES.iterdir() if p.is_dir())
    chosen = [p for p in pool if p not in blocked and picked(p)]
    if limit:
        chosen = chosen[:limit]
    chosen = chosen[shard_k::shard_n]
    if "--skip-existing" in sys.argv:
        before = len(chosen)
        chosen = [p for p in chosen if not (OUT / p).is_dir()]
        print(f"{before - len(chosen)} already built, {len(chosen)} to go",
              flush=True)

    rejected, spacings, tok_lens = [], [], []
    n_lines = 0
    voices_hist = Counter()
    for k, piece in enumerate(chosen):
        try:
            lines, pages, n_voices = build_piece(piece)
        except (Rejected, ms.Unconvertible) as e:
            rejected.append({"piece": piece, "reason": str(e)})
            continue
        except Exception as e:                                  # noqa: BLE001
            rejected.append({"piece": piece,
                             "reason": f"{type(e).__name__}: {e}"})
            continue

        pdir = OUT / piece
        pdir.mkdir(exist_ok=True)
        for ln in lines:
            gray = pages[ln["page"]]
            box = ln["box"]
            crop = gray[box["y"]:box["y"] + box["h"],
                        box["x"]:box["x"] + box["w"]]
            # Three digits: two-digit padding overflowed at system 100 and
            # rotated every checker that sorted file names (15er finding).
            name = f"s{ln['system']:03d}v{ln['voice']}"
            Image.fromarray((crop * 255).astype(np.uint8)).save(
                pdir / f"{name}.png")
            (pdir / f"{name}.json").write_text(json.dumps({
                "piece": piece, "renderer": "musescore",
                **{k: v for k, v in ln.items()},
            }, indent=1), encoding="utf-8")
            spacings.append(box["line_spacing"])
            tok_lens.append(sum(1 for t in ln["tokens"] if t != "<b>"))
            n_lines += 1
        voices_hist[n_voices] += 1
        if (k + 1) % 25 == 0:
            print(f"{k + 1}/{len(chosen)} pieces, {n_lines} lines, "
                  f"{len(rejected)} rejected", flush=True)

    summary = {
        "fraction": FRACTION, "seed": SEED,
        "pool": len(pool), "reserve_blocked": len(blocked & set(pool)),
        "chosen": len(chosen),
        "accepted": len(chosen) - len(rejected),
        "lines": n_lines,
        "voices_per_piece": {str(k): v for k, v in sorted(voices_hist.items())},
        "line_spacing_px": {
            "min": min(spacings, default=None),
            "p50": float(np.percentile(spacings, 50)) if spacings else None,
            "max": max(spacings, default=None)},
        "tokens_per_line_p50":
            float(np.percentile(tok_lens, 50)) if tok_lens else None,
        "rejection_reasons": Counter(
            r["reason"].split(" -- ")[0][:60] for r in rejected).most_common(10),
        "rejected": rejected[:50],
    }
    summary["shard"] = shard
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rejected"},
                     indent=2))


if __name__ == "__main__":
    main()
