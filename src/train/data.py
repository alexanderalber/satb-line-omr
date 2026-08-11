"""Loading the built corpus into (normalised image, sub-token) pairs.

The split between train and test is by *piece*, never by staff: two staves of
the same chorale share engraver, key, meter and often whole phrases, so a
line-wise split would measure memorisation and call it generalisation.
"""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from omr.normalize import TARGET_HEIGHT, normalize_staff
from synth.tokens import split_tokens

MAX_WIDTH = 1400          # normalised px; wider lines are reported, not silently cut
WIDTH_REDUCTION = 4


def geometry(spacing: float = 10.0) -> tuple[int, int]:
    """(target_height, max_width) for a staff-line spacing.

    The resolution runs (status-phase2.md (c)) train at spacing 12/14
    instead of 10. Height is the next multiple of 16 (the conv stack
    collapses height // 16) *upwards* of the proportional 12.8 spacings,
    so no run has less air above and below the staff than the 10 px
    baseline; width budget scales linearly. 10 -> (128, 1400) exactly.
    """
    height = int(-(-12.8 * spacing // 16) * 16)
    return height, int(round(MAX_WIDTH * spacing / 10.0))


def to_store(norm: np.ndarray) -> np.ndarray:
    """The normalised staff as it is *kept*, which is uint8 and not float32.

    Measured on the corpus after the big build: 144 808 lines at 128 px height
    and 1078 px mean width are **79.9 GB** as float32 and **20.0 GB** as uint8.
    The machine has 64 GB, so float32 is not a tuning question but the
    difference between a run and no run. The precision costs nothing that is
    there to begin with -- the source is an 8-bit PNG, and the only step in
    between is a resize. `collate` divides by 255 per batch.
    """
    return (np.clip(norm, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def ctc_min_frames(tokens) -> int:
    """Frames a CTC path needs: one per label plus a blank between repeats."""
    if not tokens:
        return 0
    repeats = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    return len(tokens) + repeats


def iter_lines(lines_dir: Path, pieces=None, spacing=10.0):
    """One `(kind, record)` per staff json, in sorted path order.

    The single walk both `load_samples` and the 40er pack build on -- one code
    path, so the pack can never filter differently from the live loader.
    `kind` is "ok" (record is a sample dict), "too_wide" or "too_long"
    (record is the drop bookkeeping).
    """
    target_height, max_width = geometry(spacing)
    metas = sorted(lines_dir.glob("*/*.json"))
    for meta in metas:
        piece = meta.parent.name
        if pieces is not None and piece not in pieces:
            continue
        d = json.loads(meta.read_text(encoding="utf-8"))
        img = np.asarray(Image.open(meta.with_suffix(".png")).convert("L"),
                         dtype=np.float32) / 255.0
        norm = normalize_staff(img, d["box"]["line_spacing"],
                               staff_offset=d["box"].get("pad_up"),
                               target_spacing=spacing,
                               target_height=target_height)
        if norm.shape[1] > max_width:
            yield "too_wide", {"piece": piece, "file": meta.name,
                               "width": int(norm.shape[1])}
            continue
        tokens = split_tokens(d["tokens"])
        frames = norm.shape[1] // WIDTH_REDUCTION
        if frames < ctc_min_frames(tokens):
            yield "too_long", {"piece": piece, "file": meta.name,
                               "frames": int(frames),
                               "labels_needed": ctc_min_frames(tokens)}
            continue
        yield "ok", {"x": to_store(norm), "tokens": tokens, "meta": d,
                     "piece": piece}


def load_samples(lines_dir: Path, pieces=None, keep_image=True, spacing=10.0):
    """Every staff under `lines_dir`, optionally restricted to `pieces`.

    Two kinds of line are dropped, both counted rather than silently skipped:
    lines wider than `MAX_WIDTH` after normalisation, and lines whose target is
    longer than the frames the convolutional stack leaves. The second kind did
    not exist in phase 1; the wider corpus contains a handful of very dense
    staves for which CTC has no valid alignment at all, so they can only ever
    contribute a zeroed loss.
    """
    out, skipped, too_long = [], [], []
    for kind, rec in iter_lines(lines_dir, pieces=pieces, spacing=spacing):
        if kind == "too_wide":
            skipped.append(rec)
        elif kind == "too_long":
            too_long.append(rec)
        else:
            if not keep_image:
                rec = {**rec, "x": None}
            out.append(rec)
    return out, {"too_wide": skipped, "target_longer_than_frames": too_long}


def load_packed(pack_base: Path, pieces=None, allow_partial=False):
    """Samples from a `40_pack_lines.py` pack -- same dicts as `load_samples`
    except `meta`, which training never reads and the pack does not carry
    (evaluation scripts keep using `load_samples`).

    Images stay on disk: `x` is a view into a read-only uint8 memmap, the OS
    page cache does the rest. That replaces hours of PNG decoding and a ~30 GB
    RAM spike with seconds of index parsing.

    A pack is a *derived* artefact; if the corpus was rebuilt since, its
    contents are silently wrong. Cheap guard: the count of meta jsons under
    the packed directory must still match. Deliberately a hard error, not a
    fallback -- a training run on a stale pack must not start.
    """
    index = json.loads(pack_base.with_suffix(".json").read_text(encoding="utf-8"))
    if index.get("limit") is not None and not allow_partial:
        raise RuntimeError(
            f"pack {pack_base.name} ist ein --limit-Probe-Pack "
            f"({index['limit']} Stuecke) -- nicht zum Training gedacht")
    lines_dir = Path(index["lines_dir"])
    n_now = len(list(lines_dir.glob("*/*.json")))
    if n_now != index["n_meta_files"]:
        raise RuntimeError(
            f"pack {pack_base.name} ist stale: {n_now} meta files unter "
            f"{lines_dir.name}, {index['n_meta_files']} beim Packen -- "
            f"40_pack_lines.py neu laufen lassen")
    mm = np.memmap(pack_base.with_suffix(".u8"), dtype=np.uint8, mode="r")
    h = index["height"]
    out = []
    for rec in index["samples"]:
        if pieces is not None and rec["piece"] not in pieces:
            continue
        w = rec["width"]
        x = mm[rec["offset"]:rec["offset"] + h * w].reshape(h, w)
        out.append({"x": x, "tokens": rec["tokens"], "meta": None,
                    "piece": rec["piece"]})
    drops = index["drops"]
    if pieces is not None:
        drops = {k: [r for r in v if r["piece"] in pieces]
                 for k, v in drops.items()}
    return out, drops


def piece_split(pieces, test_fraction=0.1, seed=20260729):
    """Deterministic split by piece name."""
    names = sorted(set(pieces))
    rng = random.Random(seed)
    rng.shuffle(names)
    n_test = max(1, int(round(len(names) * test_fraction)))
    return set(names[n_test:]), set(names[:n_test])


def build_vocab(samples):
    vocab = sorted({t for s in samples for t in s["tokens"]})
    w2i = {t: i + 1 for i, t in enumerate(vocab)}      # 0 = CTC blank
    i2w = {i + 1: t for i, t in enumerate(vocab)}
    return vocab, w2i, i2w


__all__ = ["load_samples", "load_packed", "iter_lines", "piece_split",
           "build_vocab", "geometry", "MAX_WIDTH", "TARGET_HEIGHT"]
