"""minimal end to end call: one staff line image in, kern tokens out.

    python examples/run_line.py                     # the example line shipped here
    python examples/run_line.py my-line.png 18.0    # your own crop, staff spacing in px

the second argument is the distance between two staff lines in the input image.
the real pipeline measures it during staff detection (`omr-preprocess.js`, or
`src/omr/staves.py`); this script asks for it because it starts one step later.
scaling is driven by that spacing and not by dpi, so a page scanned at any
resolution lands on the same geometry.

needs: onnxruntime, numpy, pillow.
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from omr.normalize import normalize_staff          # noqa: E402

MODEL = REPO / "model-public-domain" / "omr-line.fp32.onnx"
VOCAB = REPO / "model-public-domain" / "vocab.json"
EXAMPLE = REPO / "examples" / "line.png"
EXAMPLE_META = REPO / "examples" / "line.json"


def prepare(path: Path, line_spacing: float) -> np.ndarray:
    """png -> float32 [1, 1, 128, width], ink near 1.0, paper 0.0."""
    grey = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    # normalize_staff scales by staff spacing and pads to the fixed height of
    # 128. it is the same function the browser runs, in numpy on purpose: what
    # cannot be written in flat array operations cannot be shipped as js.
    norm = normalize_staff(grey, line_spacing)
    # the model was trained on inverted input: ink = 1, paper = 0. feeding it
    # the other way round produces garbage that looks like a broken export.
    return (1.0 - norm)[None, None].astype(np.float32)


def greedy_ctc(logits: np.ndarray, i2w: list) -> list:
    """argmax per frame, collapse repeats, drop the blank (class 0)."""
    best = logits.argmax(axis=-1)
    out, prev = [], -1
    for cls in best:
        if cls != prev and cls != 0:
            out.append(i2w[cls])
        prev = int(cls)
    return out


def main() -> None:
    if len(sys.argv) > 1:
        path, spacing = Path(sys.argv[1]), float(sys.argv[2])
        expected = None
    else:
        meta = json.loads(EXAMPLE_META.read_text(encoding="utf-8"))
        path, spacing = EXAMPLE, meta["line_spacing_px"]
        expected = meta["ground_truth_kern"]

    i2w = [None] + json.loads(VOCAB.read_text(encoding="utf-8"))["tokens"]

    x = prepare(path, spacing)
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    logits = session.run(None, {name: x})[0][0]     # [frames, vocab + 1]

    # the model emits sub tokens: "4cc" is the two classes "4" and "cc". that
    # keeps the vocabulary at 141 entries instead of thousands. "<b>" is the
    # record separator between kern tokens, so joining is just concatenating
    # between separators. do it this way and not with a heuristic: the
    # separator is what the model was trained to place.
    sub = greedy_ctc(logits, i2w)
    kern, cur = [], []
    for t in sub:
        if t == "<b>":
            if cur:
                kern.append("".join(cur))
                cur = []
        else:
            cur.append(t)
    if cur:
        kern.append("".join(cur))

    print(f"{path.name}: {x.shape[3]} px wide -> {logits.shape[0]} frames")
    print(" ".join(kern))

    if expected is not None:
        hits = sum(a == b for a, b in zip(kern, expected))
        print(f"\nground truth: {' '.join(expected)}")
        print(f"{hits} of {len(expected)} tokens match "
              f"({len(kern)} predicted)")


if __name__ == "__main__":
    main()
