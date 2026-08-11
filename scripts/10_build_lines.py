"""Build (staff image, token sequence) pairs from kern chorales.

Path 2 of the phase-1 handoff: render the whole page, then cut it with the very
detector that will later run in the browser, so training and inference see the
same crops. The staff <-> spine assignment is not assumed; it is taken from
verovio's element ids and cross-checked against the detector's geometric order.

Phase 2 adds the realism layer (`synth.realism`): lyrics, voice labels,
dynamics, chord symbols, tempo marks and a varying music font. All of it enters
as extra Humdrum spines, so the ground truth is untouched by construction --
the kern spines keep their tokens, they merely move to different field indices,
and `synth.corpus.prepare` is the single place that knows where.

Writes work/lines/<piece>/ with one png + one json per staff, plus a corpus-wide
work/10_lines.json holding the statistics the report needs.
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third_party" / "SMT"))

from _common import KERN_SOURCES                   # noqa: E402
from omr.staves import detect                      # noqa: E402
from synth import corpus as corpus_mod             # noqa: E402
from synth import kern as kern_mod                 # noqa: E402
from synth import render as render_mod             # noqa: E402
from utils import parse_kern                       # noqa: E402  (SMT, MIT)

WORK = REPO / "work"
KERN_DIR = WORK / "bach-370-chorales-master" / "kern"
OUT = WORK / "lines"
CURRENT = WORK / "10_current.txt"        # piece being rendered right now
QUARANTINE = WORK / "10_quarantine.json"  # pieces known to kill the renderer
NS = "{http://www.w3.org/2000/svg}"

# A rest carrying Humdrum's "yy" is explicitly not engraved.
INVISIBLE_REST = re.compile(r"^\d+\.*r.*yy")


class Rejected(Exception):
    """A piece we refuse to emit -- counted and reported, never silently fixed."""


def first_data_line(score) -> int:
    for rec in score.records:
        if rec.kind in ("data", "barline"):
            return rec.line_no
    raise Rejected("no data records")


def last_content_line(score) -> int:
    last = None
    for rec in score.records:
        if rec.kind in ("data", "barline"):
            last = rec.line_no
    if last is None:
        raise Rejected("no data records")
    return last


def barline_at(score, line_no: int) -> str | None:
    for rec in score.records:
        if rec.line_no == line_no and rec.kind == "barline":
            return rec.fields[0].split("\t")[0] if rec.fields else None
    return None


def system_ranges(score, systems) -> list[tuple[int, int, str | None]]:
    """Source line range and closing barline for each system, in reading order.

    A system owns the records from the start of its first measure up to the
    record before the next system's first measure. The barline that opens the
    next system is drawn at the *end* of this one, so it is handed over as the
    closing symbol.
    """
    starts = []
    for sysm in systems:
        if not sysm.measure_starts:
            raise Rejected("system without measures")
        starts.append(max(min(sysm.measure_starts), first_data_line(score)))

    ranges = []
    for i, start in enumerate(starts):
        if i + 1 < len(starts):
            nxt = starts[i + 1]
            ranges.append((start, nxt - 1, barline_at(score, nxt)))
        else:
            ranges.append((start, last_content_line(score), None))
    return ranges


def multirest_count(svgs) -> int:
    """Bars verovio collapsed into a single multi-rest glyph.

    Such a staff shows one symbol where the token sequence has one rest per bar,
    so image and target disagree by construction. Phase 1 never met the case;
    it is counted here rather than assumed away.
    """
    n = 0
    for svg in svgs:
        for g in ET.fromstring(svg).iter(NS + "g"):
            if "multiRest" in (g.get("class") or "").split():
                n += 1
    return n


def build_piece(path: Path, toolkits, realism: bool, multi_verse: bool = False):
    raw = path.read_text(encoding="utf-8", errors="replace")
    if "*^" in raw or "*v" in raw:
        raise Rejected("spine split/join (*^ / *v) -- field indices would shift")

    probe = kern_mod.parse(kern_mod.drop_labels(raw))
    # 2-8 systems per accolade: 39 % of the real engraved repertoire is five-
    # to seven-part (decision 8 in CLAUDE.md); "voice = staff index" scales,
    # the synthesis just has to show the layouts.
    if not 2 <= probe.n_spines <= 8:
        raise Rejected(f"{probe.n_spines} spines, expected 2-8")
    if any(t != "**kern" for t in probe.spine_types):
        raise Rejected(f"non-kern spines: {probe.spine_types}")

    prep = corpus_mod.prepare(path.stem, raw, realism=realism,
                              n_spines=probe.n_spines, multi_verse=multi_verse)
    text = prep.text
    score = kern_mod.parse(text)
    tk = toolkits(prep)

    svgs = render_mod.render_svg(tk, text)
    n_multirest = multirest_count(svgs)
    if n_multirest:
        raise Rejected(f"{n_multirest} multi-rest glyphs -- one symbol for "
                       f"several bars, ground truth would claim several rests")
    systems = render_mod.parse_structure(svgs)
    if not systems:
        raise Rejected("no systems rendered")

    pages = [render_mod.rasterize(s) for s in svgs]
    ranges = system_ranges(score, systems)

    # The detector runs on exactly the image the browser would see.
    detected = []
    for page_idx, gray in enumerate(pages):
        res = detect(gray)
        for sys_local, staff_ids in enumerate(res["systems"]):
            boxes = [b for b in res["boxes"]
                     if b["system"] == sys_local]
            detected.append((page_idx, boxes))

    if len(detected) != len(systems):
        raise Rejected(f"detector found {len(detected)} systems, "
                       f"verovio rendered {len(systems)}")

    out = []
    for idx, (sysm, (start, end, closing)) in enumerate(zip(systems, ranges)):
        page_idx, boxes = detected[idx]
        if page_idx != sysm.page:
            raise Rejected(f"system {idx} on page {page_idx} vs {sysm.page}")
        if len(boxes) != len(sysm.staves):
            raise Rejected(f"system {idx}: detector {len(boxes)} staves, "
                           f"verovio {len(sysm.staves)}")
        # The realism spines shift the field indices. prepare() claims to know
        # where the kern spines ended up; check it against what verovio drew
        # instead of trusting the arithmetic.
        drawn_fields = sorted(s.field_no for s in sysm.staves)
        if drawn_fields != sorted(prep.kern_fields):
            raise Rejected(f"system {idx}: staves at fields {drawn_fields}, "
                           f"prepare() says {sorted(prep.kern_fields)}")

        for voice, (box, staff) in enumerate(zip(boxes, sysm.staves)):
            state = kern_mod.state_before(score, start, staff.field_no)
            header = {
                "clef": state["clef"] if staff.has_clef else None,
                "key": state["key"] if staff.has_keysig else None,
                "meter": state["meter"] if staff.has_metersig else None,
            }
            frag = kern_mod.spine_fragment(score, staff.field_no, start, end,
                                           header, closing)
            tokens = [t for t in parse_kern(frag, "bekern") if t]
            tokens = strip_spine_markers(tokens)
            out.append({
                "page": sysm.page, "system": idx, "voice": voice,
                "field": staff.field_no, "box": box,
                "start_line": start, "end_line": end,
                "kern": frag, "tokens": tokens,
            })
    return out, pages, prep


def strip_spine_markers(tokens: list[str]) -> list[str]:
    """Reduce the token sequence to what is actually drawn on the staff.

    Two kinds of token carry no ink and must go:

    - `**kern` / `*-`, the spine delimiters. They are structural. CTC assigns
      every emitted label a frame, i.e. an x position, so a label with nothing
      to point at is both wasted budget and a broken coordinate contract.
    - invisible rests (`4ryy`, Humdrum's `yy` = do not engrave). Verovio does
      not draw them and neither did the 1875 print. Consequence to be aware of:
      the token sequence of such a staff no longer accounts for the full bar
      duration, so a MIDI export has to re-derive the gap from the bar length.

    The stored kern fragment keeps both, so it stays renderable and rhythmically
    complete.
    """
    out = [t for t in tokens if not INVISIBLE_REST.match(t)]
    while out and out[0] in ("**kern", "<b>"):
        out.pop(0)
    while out and out[-1] in ("*-", "<b>"):
        out.pop()
    # Collapse separators left doubled by a removal.
    dedup = []
    for tok in out:
        if tok == "<b>" and dedup and dedup[-1] == "<b>":
            continue
        dedup.append(tok)
    return dedup


def crop(gray: np.ndarray, box: dict) -> np.ndarray:
    y0, y1 = box["y"], box["y"] + box["h"]
    x0, x1 = box["x"], box["x"] + box["w"]
    return gray[y0:y1, x0:x1]


def toolkit_cache():
    """One verovio toolkit per option set -- reconfiguring a live toolkit is not
    worth the risk of state left over from the previous piece."""
    cache = {}

    def get(prep):
        if prep.key not in cache:
            cache[prep.key] = render_mod.make_toolkit(prep.options)
        return cache[prep.key]
    return get


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = int(args[0]) if args else 40
    realism = "--plain" not in sys.argv
    # Pop package (Entscheid 02.08.): verse blocks are opt-in until the s12
    # decision is through, so every default build stays byte-identical.
    multi_verse = "--verses" in sys.argv
    source = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--source=")),
                  "chor")
    # --shard=K/N: every Nth piece of the *full* sorted list, offset K. The
    # partition is a function of the list alone, so N workers cover the corpus
    # exactly once regardless of what is already on disk. Rendering is
    # single-core (verovio), the pieces are independent -- this is where the
    # other 15 cores come from.
    shard = next((a.split("=", 1)[1] for a in sys.argv
                  if a.startswith("--shard=")), None)
    shard_k, shard_n = (int(x) for x in shard.split("/")) if shard else (0, 1)
    current_path = (WORK / f"10_current_s{shard_k}.txt") if shard else CURRENT
    summary_path = (WORK / f"10_lines_s{shard_k}.json") if shard \
        else (WORK / "10_lines.json")
    OUT.mkdir(parents=True, exist_ok=True)
    toolkits = toolkit_cache()

    files = sorted(KERN_SOURCES[source].glob("*.krn"))[:limit][shard_k::shard_n]
    if "--skip-existing" in sys.argv:
        # prepare() is deterministic in the piece name, so a piece already on
        # disk would be re-rendered to exactly the same images. Skipping it lets
        # the corpus grow without redoing the hours already spent.
        before = len(files)
        files = [f for f in files if not (OUT / f.stem).is_dir()]
        print(f"{before - len(files)} pieces already built, {len(files)} to go",
              flush=True)
    # A piece that kills the renderer outright cannot be caught below: an access
    # violation inside verovio ends the process, so the loop never resumes. The
    # name of the piece being rendered is therefore written to disk *before* the
    # attempt, and the driver (24_build_all.py) turns the last name into a
    # quarantine entry after a fall. Quarantined pieces are counted as
    # rejections like any other, not silently dropped.
    quarantine = []
    if QUARANTINE.exists():
        quarantine = json.loads(QUARANTINE.read_text(encoding="utf-8"))
    q_set = {q["piece"] if isinstance(q, dict) else q for q in quarantine}

    rejected, lines_meta = [], []
    for name in sorted(q_set):
        rejected.append({"piece": name,
                         "reason": "quarantined -- crashed the renderer"})
    files = [f for f in files if f.stem not in q_set]
    tok_lens, widths, spacings, voices_per_field = [], [], [], Counter()
    tok_lens_nob = []          # same, with the <b> record separators removed
    fonts, features = Counter(), Counter()

    for path in files:
        current_path.write_text(path.stem, encoding="utf-8")
        try:
            lines, pages, prep = build_piece(path, toolkits, realism,
                                             multi_verse=multi_verse)
        except Rejected as e:
            rejected.append({"piece": path.stem, "reason": str(e)})
            continue
        except Exception as e:                      # renderer or parser blew up
            rejected.append({"piece": path.stem,
                             "reason": f"{type(e).__name__}: {e}"})
            continue

        cfg = prep.config.as_dict() if prep.config else {}
        fonts[prep.font] += 1
        for key in ("lyrics", "labels", "tempo", "harmony"):
            features[key] += bool(cfg.get(key))
        features["dynamics"] += bool(cfg.get("dynamic_voices"))

        # No extra files in the piece directory: every *.json in there is a
        # staff, and three other scripts rely on that.
        pdir = OUT / path.stem
        pdir.mkdir(exist_ok=True)
        for ln in lines:
            img = crop(pages[ln["page"]], ln["box"])
            # Three digits: two-digit padding overflowed at system 100 and
            # rotated every checker that sorted file names (15er finding).
            name = f"s{ln['system']:03d}v{ln['voice']}"
            Image.fromarray((img * 255).astype(np.uint8)).save(pdir / f"{name}.png")
            meta = {k: v for k, v in ln.items() if k != "kern"}
            meta["piece"] = path.stem
            meta["height"] = int(img.shape[0])
            meta["width"] = int(img.shape[1])
            meta["font"] = prep.font
            meta["realism"] = cfg
            (pdir / f"{name}.json").write_text(
                json.dumps({**meta, "kern": ln["kern"]}, indent=1), encoding="utf-8")
            lines_meta.append(meta)
            tok_lens.append(len(ln["tokens"]))
            tok_lens_nob.append(sum(1 for t in ln["tokens"] if t != "<b>"))
            widths.append(img.shape[1])
            spacings.append(ln["box"]["line_spacing"])
            voices_per_field[(ln["voice"], ln["field"])] += 1

    def pct(xs, p):
        return float(np.percentile(xs, p)) if xs else None

    summary = {
        "realism": realism,
        "source": source,
        "pieces_attempted": len(files),
        "pieces_accepted": len({m["piece"] for m in lines_meta}),
        "rejected": rejected,
        "rejection_reasons": Counter(
            r["reason"].split(" -- ")[0] for r in rejected).most_common(10),
        "fonts": dict(fonts),
        "pieces_with_feature": dict(features),
        "lines": len(lines_meta),
        "tokens_per_line": {
            "min": min(tok_lens, default=None), "p50": pct(tok_lens, 50),
            "p90": pct(tok_lens, 90), "p99": pct(tok_lens, 99),
            "max": max(tok_lens, default=None),
        },
        "tokens_per_line_without_b": {
            "p50": pct(tok_lens_nob, 50), "p99": pct(tok_lens_nob, 99),
            "max": max(tok_lens_nob, default=None),
        },
        "crop_width_px": {"min": min(widths, default=None),
                          "p50": pct(widths, 50), "max": max(widths, default=None)},
        "line_spacing_px": {"min": min(spacings, default=None),
                            "p50": pct(spacings, 50),
                            "max": max(spacings, default=None)},
        "voice_to_field": {f"v{v}->F{f}": n
                           for (v, f), n in sorted(voices_per_field.items())},
    }
    summary["shard"] = shard
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)[:2600])


if __name__ == "__main__":
    main()
