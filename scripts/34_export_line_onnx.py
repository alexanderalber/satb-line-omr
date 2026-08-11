"""Phase 3, step 1: ONNX export of the line model and parity against PyTorch.

Exports the CRNN checkpoint (`--checkpoint`, default the lauf-B best) with a
dynamic width axis, then runs the *full* synthetic test split of that run
through both PyTorch-CPU and onnxruntime-CPU and compares the greedy token
sequences line by line. Acceptance is identical sequences (target: 0
mismatching lines); small logit differences are irrelevant as long as argmax
is stable, so mismatches are counted and dumped for inspection rather than
averaged away.

Side products, both consumed by the later phase-3 steps:
- `work/onnx_fixtures/`: ~two dozen unpadded input tensors (float32, already
  inverted -- exactly what the model eats) plus the ORT-CPU reference tokens.
  The Node/onnxruntime-web smoke test (35) and the WASM benchmark (37) run
  against these, so the browser check never depends on Python being around.
- `work/34_export_line_onnx.json`: the measurement.

Runs in `venv` (CPU), no GPU involved.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from train.data import load_samples                               # noqa: E402
from train.model import CRNN, collate, greedy                     # noqa: E402

WORK = REPO / "work"
LINES = WORK / "lines"
FIXTURES = WORK / "onnx_fixtures"

OPSET = 17
N_FIXTURES = 24


def export(ckpt_path: Path, out_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab = ckpt["vocab"]
    model = CRNN(len(vocab) + 1)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()                        # freeze BatchNorm statistics

    dummy = torch.zeros(1, 1, 128, 256, dtype=torch.float32)
    kwargs = dict(
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch", 3: "width"},
                      "logits": {0: "batch", 1: "frames"}},
        opset_version=OPSET,
    )
    try:                                # torch >= 2.x grew a dynamo exporter;
        torch.onnx.export(model, dummy, str(out_path), dynamo=False, **kwargs)
    except TypeError:                   # older signature has no such flag
        torch.onnx.export(model, dummy, str(out_path), **kwargs)

    m = onnx.load(str(out_path))
    onnx.checker.check_model(m)
    ops = sorted({n.op_type for n in m.graph.node})
    return model, vocab, ckpt.get("epoch"), ops


def resolve_split(train_json: Path):
    d = json.loads(train_json.read_text(encoding="utf-8"))
    names = d.get("test_piece_names")
    if not names:
        sys.exit(f"{train_json} records no test_piece_names")
    missing = [n for n in names if not (LINES / n).is_dir()]
    if missing:
        sys.exit(f"{len(missing)} test pieces missing on disk, e.g. {missing[:3]}")
    return set(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(WORK / "model_best_lauf_b.pt"))
    ap.add_argument("--train-json", default=str(WORK / "20_train_lauf_b.json"),
                    help="run record whose test_piece_names define the split")
    ap.add_argument("--out", default=None,
                    help="output .onnx (default: work/<checkpoint stem>.onnx)")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out) if args.out else WORK / (ckpt_path.stem + ".onnx")

    model, vocab, epoch, ops = export(ckpt_path, out_path)
    i2w = {i + 1: t for i, t in enumerate(vocab)}
    w2i = {t: i + 1 for i, t in enumerate(vocab)}
    size = out_path.stat().st_size
    print(f"exported {out_path.name}: {size/1e6:.1f} MB, opset {OPSET}, "
          f"ops: {', '.join(ops)}", flush=True)

    sess = ort.InferenceSession(str(out_path),
                                providers=["CPUExecutionProvider"])

    pieces = resolve_split(Path(args.train_json))
    samples, dropped = load_samples(LINES, pieces=pieces)
    # Parity ignores the targets, but collate() builds them; GT tokens outside
    # the model vocabulary are mapped to vocab[0] like in 22/28.
    unseen = set()
    for s in samples:
        unseen.update(t for t in s["tokens"] if t not in w2i)
        s["tokens"] = [t if t in w2i else vocab[0] for t in s["tokens"]]
    print(f"{len(samples)} test lines "
          f"({sum(len(v) for v in dropped.values())} dropped by loader, "
          f"{len(unseen)} GT tokens outside vocab)", flush=True)

    ordered = sorted(samples, key=lambda s: s["x"].shape[1])
    mismatches = []
    n_done = 0
    t_torch = t_ort = 0.0
    torch_greedy, ort_greedy = [], []

    with torch.no_grad():
        for i in range(0, len(ordered), args.batch):
            chunk = ordered[i:i + args.batch]
            x, _, in_lens, _ = collate(chunk, w2i, "cpu")
            t0 = time.perf_counter()
            lt = model(x)
            t1 = time.perf_counter()
            lo = sess.run(["logits"], {"input": x.numpy()})[0]
            t2 = time.perf_counter()
            t_torch += t1 - t0
            t_ort += t2 - t1
            for k, s in enumerate(chunk):
                w = min(int(in_lens[k]), lt.shape[1])
                ht = greedy(lt[k, :w], i2w)
                ho = greedy(lo[k, :w], i2w)
                torch_greedy.append(ht)
                ort_greedy.append(ho)
                if ht != ho:
                    mismatches.append({
                        "piece": s["piece"], "meta": s["meta"].get("system"),
                        "width": int(s["x"].shape[1]),
                        "torch": " ".join(ht), "ort": " ".join(ho)})
            n_done += len(chunk)
            if (i // args.batch) % 200 == 0:
                print(f"  {n_done}/{len(ordered)} lines, "
                      f"{len(mismatches)} mismatches", flush=True)

    # Fixtures for the Node/WASM steps: spread across the width range so the
    # benchmark sees short and long lines, reference tokens from ORT-CPU.
    FIXTURES.mkdir(exist_ok=True)
    idx = np.linspace(0, len(ordered) - 1, N_FIXTURES).astype(int)
    index = []
    for n, j in enumerate(idx):
        s = ordered[j]
        a = s["x"].astype(np.float32) / 255.0
        inp = (1.0 - a).astype(np.float32)          # ink 1, paper 0
        fn = f"line_{n:02d}.bin"
        (FIXTURES / fn).write_bytes(inp.tobytes())
        index.append({"file": fn, "height": int(inp.shape[0]),
                      "width": int(inp.shape[1]), "piece": s["piece"],
                      "tokens_ort_cpu": ort_greedy[j],
                      "token_ids_ort_cpu":
                          [w2i[t] for t in ort_greedy[j]]})
    (FIXTURES / "index.json").write_text(json.dumps({
        "note": "input .bin is float32 row-major (height,width), ALREADY "
                "inverted (ink~1, paper 0); feed as (1,1,H,W)",
        "vocab_file": "../vocab_model_lauf_b.json",
        "blank": 0,
        "lines": index}, indent=1), encoding="utf-8")

    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    out = {
        "checkpoint": ckpt_path.name, "epoch": epoch,
        "onnx_file": out_path.name, "onnx_bytes": size,
        "onnx_sha256": sha, "opset": OPSET,
        "graph_ops": ops,
        "onnxruntime": ort.__version__, "torch": torch.__version__,
        "test_split": {"source": Path(args.train_json).name,
                       "pieces": len(pieces), "lines": len(ordered)},
        "parity": {
            "criterion": "identical greedy token sequences per line",
            "lines_compared": len(ordered),
            "lines_mismatching": len(mismatches),
            "mismatches": mismatches[:50],
        },
        "seconds_forward_total": {"torch_cpu": round(t_torch, 1),
                                  "ort_cpu": round(t_ort, 1)},
        "fixtures": {"dir": "onnx_fixtures", "count": N_FIXTURES},
    }
    (WORK / "34_export_line_onnx.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in
                      ("onnx_bytes", "opset", "graph_ops", "parity",
                       "seconds_forward_total")}, indent=2))


if __name__ == "__main__":
    main()
