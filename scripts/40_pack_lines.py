"""Pack the built corpus into a memmap, so training starts in seconds.

Three runs in a row spent hours in the loading phase: 145k PNGs decoded and
normalised one by one, single-threaded, with a ~30 GB RAM spike (uint8 already,
see `data.to_store` -- the spike is unavoidable as long as everything must sit
in RAM before the first epoch). The pack does that work once, per lines
directory and per spacing, and streams it to disk: RAM stays flat while
packing, and `data.load_packed` afterwards serves the images as memmap views.

The walk and the filters are `data.iter_lines` -- the same generator
`load_samples` is built on, so pack and live loader cannot disagree. Byte
equivalence is measured, not assumed: `_probe_pack_equiv.py`.

A pack is derived data and goes stale when the corpus is rebuilt;
`load_packed` guards that with a meta-file count and refuses to serve. Rerun
this script after every 10er/31er build.

Writes work/pack_<dir>_s<spacing>.u8 (images) and .json (index).
Usage: 40_pack_lines.py [--spacing=10] [--dir=lines] [--limit=N]
       --dir may repeat (--dir=lines --dir=lines_mscore); --limit packs only
       the first N pieces (probe use, the index still records the full count).
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from train.data import geometry, iter_lines                         # noqa: E402

WORK = REPO / "work"


def pack_dir(lines_dir: Path, spacing: float, limit: int | None):
    height, _ = geometry(spacing)
    base = WORK / f"pack_{lines_dir.name}_s{spacing:g}"
    n_meta = len(list(lines_dir.glob("*/*.json")))

    pieces = None
    if limit is not None:
        pieces = set(sorted(p.name for p in lines_dir.iterdir()
                            if p.is_dir())[:limit])

    samples, drops = [], {"too_wide": [], "target_longer_than_frames": []}
    t0, offset = time.time(), 0
    with open(base.with_suffix(".u8"), "wb") as fh:
        for kind, rec in iter_lines(lines_dir, pieces=pieces, spacing=spacing):
            if kind == "too_wide":
                drops["too_wide"].append(rec)
                continue
            if kind == "too_long":
                drops["target_longer_than_frames"].append(rec)
                continue
            x = rec["x"]
            assert x.dtype.name == "uint8" and x.shape[0] == height
            fh.write(x.tobytes())
            samples.append({"piece": rec["piece"], "width": int(x.shape[1]),
                            "offset": offset, "tokens": rec["tokens"]})
            offset += x.size
            if len(samples) % 10000 == 0:
                print(f"  {lines_dir.name}: {len(samples)} lines, "
                      f"{offset/1e9:.1f} GB, {(time.time()-t0)/60:.1f} min",
                      flush=True)

    index = {"lines_dir": str(lines_dir), "spacing": spacing,
             "height": height, "n_meta_files": n_meta,
             "limit": limit, "bytes": offset,
             "samples": samples, "drops": drops}
    base.with_suffix(".json").write_text(json.dumps(index), encoding="utf-8")
    print(f"{lines_dir.name}: {len(samples)} lines packed, "
          f"{len(drops['too_wide'])} too wide, "
          f"{len(drops['target_longer_than_frames'])} too long, "
          f"{offset/1e9:.1f} GB, {(time.time()-t0)/60:.1f} min "
          f"-> {base.name}.u8/.json", flush=True)


def main():
    spacing = next((float(a.split("=", 1)[1]) for a in sys.argv
                    if a.startswith("--spacing=")), 10.0)
    dirs = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--dir=")] \
        or ["lines"]
    limit = next((int(a.split("=", 1)[1]) for a in sys.argv
                  if a.startswith("--limit=")), None)
    for d in dirs:
        pack_dir(WORK / d, spacing, limit)


if __name__ == "__main__":
    main()
