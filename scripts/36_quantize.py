"""Phase 3, step 3 -- Go/No-Go 2: quantisation, measured instead of guessed.

Takes the fp32 ONNX export from 34 and produces an int8 (dynamic) and an fp16
variant, then measures SER of all three under onnxruntime-CPU on the two fixed
sets the contract names:

1. the synthetic test split of the run (test_piece_names from the train json),
2. the 28er reserve set, both renderers: `work/mscore_lines/` and the verovio
   lines of the same pieces from `work/lines/`.

The contract is taken literally: the numbers (SER fp32 vs int8 on the same
test set), not the impression. File sizes are reported against the budget
(<= 40 MB total, <= 25 MB per file). IO stays float32 in all variants
(fp16 conversion runs with keep_io_types) so the frontend contract does not
change with the variant.

Writes work/36_quantize.json. Runs in `venv`, CPU only.
"""
import argparse
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
from train.model import collate, greedy, ser                      # noqa: E402

WORK = REPO / "work"
LINES = WORK / "lines"
MSCORE = WORK / "mscore_lines"


def make_int8(src: Path, dst: Path):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return sorted({n.op_type for n in onnx.load(str(dst)).graph.node})


def make_fp16(src: Path, dst: Path):
    from onnxruntime.transformers.float16 import convert_float_to_float16
    m = onnx.load(str(src))
    m16 = convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m16, str(dst))
    return sorted({n.op_type for n in m16.graph.node})


def load_sets(train_json: Path):
    d = json.loads(train_json.read_text(encoding="utf-8"))
    names = d.get("test_piece_names")
    if not names:
        sys.exit(f"{train_json} records no test_piece_names")
    synth, synth_drop = load_samples(LINES, pieces=set(names))

    if not MSCORE.is_dir():
        sys.exit("work/mscore_lines missing -- run 28_musescore_eval.py first")
    ms, _ = load_samples(MSCORE)
    ms_pieces = sorted({s["piece"] for s in ms})
    vv, _ = load_samples(LINES, pieces=set(ms_pieces))
    return {"synthetic_testsplit": synth,
            "reserve28_musescore": ms,
            "reserve28_verovio": vv}


def evaluate(sess, samples, w2i, i2w, vocab, batch=8):
    """SER of one ONNX session over one line set, ORT-CPU."""
    prepared = []
    unseen = set()
    for s in samples:
        toks = s["tokens"]
        unseen.update(t for t in toks if t not in w2i)
        prepared.append({**s, "tokens": [t if t in w2i else vocab[0]
                                         for t in toks]})
    ordered = sorted(prepared, key=lambda s: s["x"].shape[1])
    sers = []
    t0 = time.perf_counter()
    for i in range(0, len(ordered), batch):
        chunk = ordered[i:i + batch]
        x, _, in_lens, _ = collate(chunk, w2i, "cpu")
        logits = sess.run(["logits"], {"input": x.numpy()})[0]
        for k, s in enumerate(chunk):
            w = min(int(in_lens[k]), logits.shape[1])
            hyp = greedy(logits[k, :w], i2w)
            sers.append(ser(s["tokens"], hyp))
    dt = time.perf_counter() - t0
    return {
        "lines": len(sers),
        "ser_mean": round(float(np.mean(sers)), 5) if sers else None,
        "ser_median": round(float(np.median(sers)), 5) if sers else None,
        "lines_exact": int(sum(1 for e in sers if e == 0)),
        "exact_percent": round(100 * sum(1 for e in sers if e == 0)
                               / len(sers), 1) if sers else None,
        "gt_tokens_not_in_vocab": sorted(unseen),
        "seconds": round(dt, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(WORK / "model_best_lauf_b.onnx"))
    ap.add_argument("--checkpoint", default=str(WORK / "model_best_lauf_b.pt"))
    ap.add_argument("--train-json", default=str(WORK / "20_train_lauf_b.json"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--only", choices=["fp32", "int8", "fp16"], default=None,
                    help="measure a single variant, merge into existing JSON")
    args = ap.parse_args()

    src = Path(args.onnx)
    p_int8 = src.with_name(src.stem + "_int8.onnx")
    p_fp16 = src.with_name(src.stem + "_fp16.onnx")

    ops_int8 = make_int8(src, p_int8)
    print(f"int8: {p_int8.stat().st_size/1e6:.1f} MB, ops: "
          f"{', '.join(ops_int8)}", flush=True)
    ops_fp16 = make_fp16(src, p_fp16)
    print(f"fp16: {p_fp16.stat().st_size/1e6:.1f} MB, ops: "
          f"{', '.join(ops_fp16)}", flush=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vocab = ckpt["vocab"]
    w2i = {t: i + 1 for i, t in enumerate(vocab)}
    i2w = {i + 1: t for i, t in enumerate(vocab)}

    sets = load_sets(Path(args.train_json))
    for name, samples in sets.items():
        print(f"{name}: {len(samples)} lines", flush=True)

    variants = {"fp32": src, "int8": p_int8, "fp16": p_fp16}
    if args.only:
        variants = {args.only: variants[args.only]}

    # The result file is written after every variant and merged with whatever
    # is already there: the first run of this script lost fp32 and int8 to a
    # native crash (0xC0000005) at fp16 session *creation* -- ORT 1.28.0's
    # graph optimizer dies fusing the fp16 LSTM. fp16 therefore runs with
    # optimizations disabled, and that workaround is part of the measurement.
    out_path = WORK / "36_quantize.json"
    out = (json.loads(out_path.read_text(encoding="utf-8"))
           if out_path.exists() else {})
    out.update({"source_onnx": src.name,
                "budget": "<=40 MB total, <=25 MB per file",
                "int8_graph_ops": ops_int8,
                "onnxruntime": ort.__version__})
    out.setdefault("variants", {})

    for vname, vpath in variants.items():
        so = ort.SessionOptions()
        note = None
        if vname == "fp16":
            so.graph_optimization_level = \
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            note = ("graph optimizations disabled: default level crashes "
                    "ORT 1.28.0 natively (0xC0000005) when fusing the fp16 "
                    "LSTM at session creation")
        sess = ort.InferenceSession(str(vpath), so,
                                    providers=["CPUExecutionProvider"])
        entry = {"file": vpath.name,
                 "megabytes": round(vpath.stat().st_size / 1e6, 2),
                 "sets": {}}
        if note:
            entry["note"] = note
        for sname, samples in sets.items():
            entry["sets"][sname] = evaluate(sess, samples, w2i, i2w, vocab,
                                            batch=args.batch)
            print(f"{vname} / {sname}: SER "
                  f"{entry['sets'][sname]['ser_mean']} "
                  f"({entry['sets'][sname]['seconds']} s)", flush=True)
        out["variants"][vname] = entry
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        del sess

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
