"""Convert the 2-8-part vocal slice of PDMX into kern the pipeline can render.

The handoff asked to verify the route "MusicXML into verovio, Humdrum out of it"
before believing it. Verified, and it does not exist: PDMX has no MusicXML in it
(see `synth/pdmx.py`). What it does have is enough to engrave from, so the
conversion is done here instead, and everything downstream stays as it was.

Piece names are **stable**: `pdmx_` plus a hash of the source path, never a
running number. The first version numbered by acceptance order, which meant
that widening the track filter (4 -> 2-8 voices, design §3) would have made
every existing name -- including the ones on the append-only reserve list --
silently point at a different piece. A name must survive every future change
of the filter, or the reserve rule is unenforceable.

Rejections are counted by reason, not swallowed.

The pieces on the reserve list (work/reserve_testset.json, append-only) are
converted **in addition to** `want`, whether or not the shuffled draw reaches
them: they are the measuring instrument of the render axis, and a filter
change reshuffles the candidate order, so no `want` can guarantee them.
Measured on the first 2-8 rebuild: 16 of 40 reserve pieces fell outside a
4500-piece draw that way. A reserve piece that cannot be converted any more
is reported loudly in the JSON, never dropped.

Writes work/pdmx_kern/*.krn and work/19_pdmx.json.
"""
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from synth.pdmx import MAX_VOICES, MIN_VOICES, Unconvertible, convert  # noqa: E402

WORK = REPO / "work"
ROOT = WORK / "PDMX"
OUT = WORK / "pdmx_kern"
SEED = 20260729
csv.field_size_limit(10_000_000)


def piece_name(source_rel: str) -> str:
    """Stable name of a PDMX piece, a function of its source path alone."""
    return "pdmx_" + hashlib.sha1(source_rel.encode("utf-8")).hexdigest()[:10]


# Pop family without folk, as Vorstufe B measured it (`_probe_popsatz.py`):
# the target is the choir's pop repertoire, and folk dominates the family
# without resembling it. Every candidate of these genres is converted in
# addition to `want` (Entscheid Pop-Paket 02.08., Auflage 2) -- no weights,
# no oversampling, just all of the ~70 convertible pieces in the corpus.
POP = {"pop", "rock", "rnb", "hiphop", "rap", "punk", "metal", "disco",
       "funk", "soul", "reggae", "blues", "country"}


def candidates():
    """2-8-track, lyric-bearing, deduplicated, public-domain rows."""
    out = []
    with (ROOT / "PDMX.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                n_tracks = int(row["n_tracks"])
            except ValueError:
                continue
            if (MIN_VOICES <= n_tracks <= MAX_VOICES
                    and row["has_lyrics"] == "True"
                    and row["subset:deduplicated"] == "True"
                    and row["license"] in ("publicdomain", "cc-zero")):
                out.append((row["path"], row["license"], row["title"],
                            row["song_length.bars"],
                            (row["genres"] or "NA").strip().lower()))
    return out


def main():
    # Default matches the corpus stand (accepted 4516 = 4500 + reserve in the
    # committed 19_pdmx.json). The old default of 600 was stale against that
    # and silently shrank work/pdmx_kern on a bare rerun -- measured the hard
    # way on 02.08.; this script deletes before it converts.
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 4500
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.krn"):
        old.unlink()

    rows = candidates()
    random.Random(SEED).shuffle(rows)

    reasons = Counter()
    licences = Counter()
    voices = Counter()
    accepted = []
    for rel, licence, title, bars, genre in rows:
        if len(accepted) >= want:
            break
        path = ROOT / rel.lstrip("./")
        try:
            text, stats = convert(path)
        except Unconvertible as e:
            reasons[str(e).split(" -- ")[0]] += 1
            continue
        except Exception as e:                       # malformed file
            reasons[f"{type(e).__name__}"] += 1
            continue

        name = piece_name(rel)
        (OUT / f"{name}.krn").write_text(text, encoding="utf-8")
        licences[licence] += 1
        voices[stats["voices"]] += 1
        accepted.append({"piece": name, "source": rel, "license": licence,
                         "title": title, "genre": genre, **stats})

    # The reserve pieces, beyond want. Same acceptance path, extra bookkeeping.
    reserve_file = WORK / "reserve_testset.json"
    reserve_names = set()
    reserve_extra, reserve_lost = [], []
    if reserve_file.exists():
        reserve = json.loads(reserve_file.read_text(encoding="utf-8"))["pdmx"]
        reserve_names = set(reserve)
        have = {a["piece"] for a in accepted}
        by_name = {piece_name(rel): (rel, licence, title, bars, genre)
                   for rel, licence, title, bars, genre in rows}
        for name in reserve:
            if name in have:
                continue
            if name not in by_name:
                reserve_lost.append({"piece": name,
                                     "reason": "no longer a candidate"})
                continue
            rel, licence, title, bars, genre = by_name[name]
            try:
                text, stats = convert(ROOT / rel.lstrip("./"))
            except Exception as e:                   # noqa: BLE001
                reserve_lost.append({"piece": name,
                                     "reason": f"{type(e).__name__}: {e}"})
                continue
            (OUT / f"{name}.krn").write_text(text, encoding="utf-8")
            licences[licence] += 1
            voices[stats["voices"]] += 1
            accepted.append({"piece": name, "source": rel, "license": licence,
                             "title": title, "genre": genre,
                             "reserve_only": True, **stats})
            reserve_extra.append(name)

    # The genre pull (Entscheid Pop-Paket 02.08., Auflage 2): every pop-family
    # candidate beyond want, complete rather than sampled -- the funnel caps
    # the yield at ~70 pieces anyway (Vorstufe B). Reserve pieces stay test
    # material and are never trained on.
    genre_pull, genre_reasons = [], Counter()
    have = {a["piece"] for a in accepted}
    for rel, licence, title, bars, genre in rows:
        if genre not in POP:
            continue
        name = piece_name(rel)
        if name in have or name in reserve_names:
            continue
        try:
            text, stats = convert(ROOT / rel.lstrip("./"))
        except Unconvertible as e:
            genre_reasons[str(e).split(" -- ")[0]] += 1
            continue
        except Exception as e:                       # noqa: BLE001
            genre_reasons[f"{type(e).__name__}"] += 1
            continue
        (OUT / f"{name}.krn").write_text(text, encoding="utf-8")
        licences[licence] += 1
        voices[stats["voices"]] += 1
        accepted.append({"piece": name, "source": rel, "license": licence,
                         "title": title, "genre": genre,
                         "genre_pull": True, **stats})
        genre_pull.append(name)

    out = {
        "candidates_in_csv": len(rows),
        "reserve_converted_extra": reserve_extra,
        "reserve_unconvertible": reserve_lost,
        "genre_pull_converted": genre_pull,
        "genre_pull_rejections": genre_reasons.most_common(10),
        "examined": sum(reasons.values()) + len(accepted),
        "accepted": len(accepted),
        "licences": dict(licences),
        "voices": {str(k): v for k, v in sorted(voices.items())},
        "rejection_reasons": reasons.most_common(20),
        "pieces": accepted,
    }
    (WORK / "19_pdmx.json").write_text(json.dumps(out, indent=2),
                                       encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "pieces"}, indent=2))
    for p in accepted[:8]:
        print(f"  {p['piece']}  {p['meter']} {p['key']} {p['clefs']} "
              f"{p['measures']} bars, {p['notes']} notes  "
              f"{(p['title'] or '')[:50]}")


if __name__ == "__main__":
    main()
