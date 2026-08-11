"""The real training run: the whole corpus, split by piece, on the 3090.

The split is by piece and never by staff. Two staves of the same piece share
engraver, key, meter, font, the realism draw and often whole phrases, so a
staff-wise split would report memorisation as generalisation.

The architecture is the one the overfit gate cleared (`17_overfit_gate.json`):
CRNN, four conv blocks, BiLSTM 2x256, CTC, width reduction 4.

Usage:  20_train.py [epochs] [batch] [--source=chor|pdmx|alle] [--tag=NAME]
                    [--mix=B:P|--epoch-lines=N] [--mscore] [--spacing=10|12|14]
                    [--pack] [--dry]

`--source` restricts the corpus, `--tag` suffixes every output file. Both exist
for controlled runs: the big corpus arrived together with a new recipe (batch
8 -> 32, 40 -> 15 epochs), which leaves corpus and recipe confounded. Isolating
the recipe means holding the corpus fixed and changing nothing else, and that
needs a run that neither overwrites `model_best.pt` nor silently trains on
everything.

`--mix=1:2` holds the bach:pdmx *line* ratio at training time by resampling
the larger pool every epoch -- the mixture is a sampling parameter, not a
build parameter (review ruling of 30.07.): the measured optimum sits near
1:2 and drifted to 1:25 as a side effect of piece counts, which cost real-
material accuracy. `--mscore` adds the paired MuseScore renderings from
work/lines_mscore (run B of the design); they join their piece's source pool
for the mix and are train-only -- the synthetic test stays verovio, the
render axis is measured on the reserve set.

Pieces on work/reserve_testset.json never enter this script's corpus at all,
in either renderer -- that list is append-only and the exclusion is the rule
that makes it a measuring instrument.

`--source` filters the *training* set, after the split (Auflage 04.08., see
the comment at the split): two runs that differ only in `--source` then
differ in exactly the pieces that source excludes, and in nothing else.

Writes work/model_best{tag}.pt, work/vocab_model{tag}.json and
work/20_train{tag}.json.
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from train.data import (build_vocab, geometry, load_packed,       # noqa: E402
                        load_samples, piece_split)
from train.model import CRNN, collate, greedy, ser                 # noqa: E402

WORK = REPO / "work"
LINES = WORK / "lines"

BATCH = 8
SPACING = 10.0
HEIGHT = 128
"""--spacing=12|14 runs the resolution experiments of status-phase2.md (c):
same corpus, same recipe, larger normalised staff. Height follows via
data.geometry (next multiple of 16 upwards), so the conv stack fits."""
"""Overridable as the second argument.

8 came from the smoke test on 16 060 lines and stayed by inertia. With 3.19 M
parameters and eight images per step the 3090 spends most of its time on kernel
launch overhead rather than arithmetic, and the corpus is now nine times larger.
The learning rate is deliberately *not* scaled along with it: the baseline ran
at 1e-3 and never showed instability, and changing two things at once would make
the comparison between the two runs unreadable.
"""


def make_batches(samples, rng):
    """Width-sorted buckets, shuffled between epochs.

    Sorting by width keeps the padding small; shuffling the *order of the
    buckets* keeps the gradient from following the width."""
    ordered = sorted(samples, key=lambda s: s["x"].shape[1])
    batches = [ordered[i:i + BATCH] for i in range(0, len(ordered), BATCH)]
    rng.shuffle(batches)
    return batches


def evaluate(model, samples, w2i, i2w, device, limit=None):
    model.eval()
    picked = samples if limit is None else samples[:limit]
    ordered = sorted(picked, key=lambda s: s["x"].shape[1])
    sers, pairs = [], []
    with torch.no_grad():
        for i in range(0, len(ordered), BATCH):
            batch = ordered[i:i + BATCH]
            x, _, in_lens, _ = collate(batch, w2i, device, height=HEIGHT)
            logits = model(x)
            for k, s in enumerate(batch):
                w = min(int(in_lens[k]), logits.shape[1])
                hyp = greedy(logits[k, :w], i2w)
                sers.append(ser(s["tokens"], hyp))
                pairs.append((s, hyp))
    return float(np.mean(sers)) if sers else 1.0, pairs


PREFIX = {"chor": "chor", "pdmx": "pdmx", "alle": ""}


def main():
    global BATCH, SPACING, HEIGHT
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    epochs = int(pos[0]) if pos else 40
    if len(pos) > 1:
        BATCH = int(pos[1])
    opts = dict(f[2:].split("=", 1) if "=" in f else (f[2:], "1")
                for f in flags)
    SPACING = float(opts.get("spacing", SPACING))
    HEIGHT, _ = geometry(SPACING)
    source = opts.get("source", "alle")
    tag = opts.get("tag", "")
    mix = opts.get("mix")
    use_mscore = "mscore" in opts
    use_pack = "pack" in opts
    if source not in PREFIX:
        sys.exit(f"--source muss eines von {sorted(PREFIX)} sein")
    mix_ratio = None
    if mix:
        b, p = mix.split(":")
        mix_ratio = float(p) / float(b)
    epoch_lines = int(opts["epoch-lines"]) if "epoch-lines" in opts else None
    if epoch_lines is not None and mix_ratio is not None:
        sys.exit("--mix und --epoch-lines schliessen sich aus: das eine setzt "
                 "die Zusammensetzung der Epoche, das andere ihre Groesse")

    torch.manual_seed(0)
    rng = random.Random(7)

    reserve_file = WORK / "reserve_testset.json"
    reserved: set[str] = set()
    if reserve_file.exists():
        r = json.loads(reserve_file.read_text(encoding="utf-8"))
        reserved = set(r["bach"]) | set(r["pdmx"])

    # Split *before* filter (Supervisor-Auflage 04.08.,
    # `lesung-mscore-pool-2026-08-04.md`). `piece_split` is deterministic in
    # the *list* it is given, so filtering first reshuffles every remaining
    # piece between train and test. Measured for `--source=pdmx`: 783 PDMX
    # pieces change sides, 20 825 lines leave the training set and 19 580
    # different ones enter -- four times the effect the Bach comparison is
    # after (4 912 Bach lines). So the split runs once on the full list and
    # the source filter applies to the training set afterwards; the test set
    # stays identical across arms and both runs are read on one material.
    pieces = sorted(p.name for p in LINES.iterdir() if p.is_dir()
                    and p.name not in reserved)
    n_blocked = sum(1 for p in LINES.iterdir()
                    if p.is_dir() and p.name in reserved)
    if not pieces:
        sys.exit("keine Stuecke -- erst 10_build_lines.py laufen lassen")
    train_pieces, test_pieces = piece_split(pieces, test_fraction=0.1)
    n_train_unfiltered = len(train_pieces)
    if PREFIX[source]:
        train_pieces = {p for p in train_pieces
                        if p.startswith(PREFIX[source])}
        if not train_pieces:
            sys.exit(f"keine Trainingsstuecke fuer --source={source}")
    if "dry" in opts:
        # The corpus assignment without loading a byte of it: seconds, no GPU.
        # It exists so the shipped trainer can be *shown* to produce the
        # registered split (57_split_vor_filter.py compares its digests),
        # instead of a replica of this logic being trusted to match.
        import hashlib
        dig = (lambda names: hashlib.sha256(
            "\n".join(sorted(names)).encode("utf-8")).hexdigest())
        info = {
            "source": source, "pieces_total": len(pieces),
            "pieces_train_before_source_filter": n_train_unfiltered,
            "pieces_train": len(train_pieces),
            "pieces_test": len(test_pieces),
            "train_sha256": dig(train_pieces),
            "test_sha256": dig(test_pieces)}
        # The epoch size the mixture would produce, from the pack indexes --
        # so a registration can name it before a GPU is touched. Counted the
        # way the run counts it: mscore lines belong to their piece's source.
        pools = {"bach": 0, "pdmx": 0}
        for d in ("lines", "lines_mscore") if use_mscore else ("lines",):
            idx_path = WORK / f"pack_{d}_s{SPACING:g}.json"
            if not idx_path.exists():
                pools = None
                break
            for rec in json.loads(idx_path.read_text(encoding="utf-8"))["samples"]:
                if rec["piece"] in train_pieces:
                    pools["bach" if rec["piece"].startswith("chor")
                          else "pdmx"] += 1
        if pools is not None:
            info["pool_bach_lines"], info["pool_pdmx_lines"] = \
                pools["bach"], pools["pdmx"]
            if mix_ratio is not None:
                want = int(round(pools["bach"] * mix_ratio))
                info["epoch_lines_mix"] = (pools["bach"] + pools["pdmx"]
                                           if want >= pools["pdmx"]
                                           else pools["bach"] + want)
            info["epoch_lines_flag"] = epoch_lines
        print(json.dumps(info, indent=2))
        return
    print(f"{len(pieces)} pieces ({n_blocked} reserviert ausgeschlossen) -> "
          f"{n_train_unfiltered} train / {len(test_pieces)} test, "
          f"--source={source} laesst {len(train_pieces)} Trainingsstuecke",
          flush=True)

    # --pack serves the images out of the 40er memmap instead of decoding
    # 145k PNGs (hours, ~30 GB spike). Byte-equivalent by construction (same
    # iter_lines walk) and by measurement (_probe_pack_equiv.py); load_packed
    # hard-fails on a pack older than the corpus rather than training on it.
    if use_pack:
        train, drop_a = load_packed(WORK / f"pack_lines_s{SPACING:g}",
                                    pieces=train_pieces)
        test, drop_b = load_packed(WORK / f"pack_lines_s{SPACING:g}",
                                   pieces=test_pieces)
    else:
        train, drop_a = load_samples(LINES, pieces=train_pieces, spacing=SPACING)
        test, drop_b = load_samples(LINES, pieces=test_pieces, spacing=SPACING)
    if not train:
        sys.exit("no samples -- run 10_build_lines.py first")

    n_mscore = 0
    if use_mscore:
        # Train-only by construction: restricted to the train split. The
        # synthetic test set stays pure verovio so its numbers remain
        # comparable across runs A and B.
        if use_pack:
            ms_train, _ = load_packed(WORK / f"pack_lines_mscore_s{SPACING:g}",
                                      pieces=train_pieces)
        else:
            ms_train, _ = load_samples(WORK / "lines_mscore",
                                       pieces=train_pieces, spacing=SPACING)
        n_mscore = len(ms_train)
        train += ms_train
        print(f"+ {n_mscore} musescore lines (train only)", flush=True)

    # The vocabulary comes from the training split alone. A token only the test
    # split contains cannot be predicted, and pretending otherwise by fitting
    # the vocabulary on everything would quietly leak.
    vocab, w2i, i2w = build_vocab(train)
    unseen = sorted({t for s in test for t in s["tokens"]} - set(vocab))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CRNN(len(vocab) + 1, height=HEIGHT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=1e-5)
    lossf = nn.CTCLoss(blank=0, zero_infinity=True)
    n_params = sum(p.numel() for p in model.parameters())

    # Unseen test tokens are mapped to a spare id so the batch can be built;
    # they can never be predicted, and they are counted as errors, which is the
    # honest treatment.
    for s in test:
        s["tokens"] = [t if t in w2i else vocab[0] for t in s["tokens"]]

    print(f"{len(train)} train lines / {len(test)} test lines, "
          f"vocab {len(vocab)}, {n_params/1e6:.2f}M params, device {device}, "
          f"unseen test tokens {unseen}", flush=True)

    # The two pools of the mixture. MuseScore renderings count towards the
    # pool of their piece's source -- the mix is about musical material, the
    # renderer axis is handled by the pairing.
    pool_bach = [s for s in train if s["piece"].startswith("chor")]
    pool_pdmx = [s for s in train if s["piece"].startswith("pdmx")]
    if mix_ratio is not None and (not pool_bach or not pool_pdmx):
        sys.exit("--mix braucht beide Korpora im Trainingssatz")
    if epoch_lines is not None and epoch_lines > len(train):
        sys.exit(f"--epoch-lines={epoch_lines} > {len(train)} Trainingszeilen")

    if epoch_lines is not None:
        n_epoch_lines = epoch_lines
    elif mix_ratio is None:
        n_epoch_lines = len(train)
    else:
        _want = int(round(len(pool_bach) * mix_ratio))
        n_epoch_lines = (len(train) if _want >= len(pool_pdmx)
                         else len(pool_bach) + _want)

    def epoch_samples():
        # `--epoch-lines` is the mixture's counterpart for a run that has no
        # bach pool to hold a ratio against (`--source=pdmx`): it fixes the
        # epoch *size* instead of the epoch *composition*, so control and
        # treatment do the same number of updates on the same amount of
        # material. See the registration of the bach nights for why that is
        # the honest analogue -- a treatment epoch that simply drops the bach
        # share would train 33 % fewer steps, and b80 measured that step
        # count is not free.
        if epoch_lines is not None:
            return rng.sample(train, epoch_lines)
        if mix_ratio is None:
            return train
        want = int(round(len(pool_bach) * mix_ratio))
        if want >= len(pool_pdmx):
            return train
        # A fresh draw every epoch: over 40 epochs the model still meets the
        # whole pdmx pool, but every single epoch holds the measured ratio.
        return pool_bach + rng.sample(pool_pdmx, want)

    # Die Teilmenge, auf der die Modellauswahl laeuft. Frueher `test[:600]` --
    # und weil `test` nach Stuecknamen sortiert ist (`chor*` vor `pdmx*`),
    # bestand dieses Kriterium aus 508 Bach- und 92 PDMX-Zeilen: **84,7 %
    # Bach** bei 2,29 % Bach im Testsatz. Fuer Laeufe mit Bach war das ein
    # Schoenheitsfehler, fuer den bachfreien Arm der Bach-Naechte ein
    # Ausschlussgrund -- er wurde nach Material ausgewaehlt, das er nie
    # gesehen hat (befund-bach-naechte-2026-08-09.md §6).
    # Jetzt: seed-feste Zufallsstichprobe gleicher Groesse (Entscheid E1 des
    # Supervisors, 09.08.). Gleiche Laufzeit, repraesentative Zusammensetzung,
    # beide Straten je Epoche berichtet. Der volle Testsatz waere 37-fach und
    # zahlt sich nicht aus. **Bruch:** `test_ser_best_subset` aelterer Laeufe
    # ist eine andere Groesse als die neuer -- `test_ser_full` bleibt
    # vergleichbar. Das Kriterium steht als Feld im Ergebnis-JSON.
    SELECT_SEED, SELECT_N = 20260809, 600
    select = random.Random(SELECT_SEED).sample(test, min(SELECT_N, len(test)))
    sel_by_corpus: dict[str, int] = {}
    for s in select:
        k = "bach" if s["piece"].startswith("chor") else "pdmx"
        sel_by_corpus[k] = sel_by_corpus.get(k, 0) + 1
    print(f"Auswahl-Teilmenge: {len(select)} Zeilen, "
          f"{dict(sorted(sel_by_corpus.items()))}, seed {SELECT_SEED}",
          flush=True)

    ckpt_path = WORK / f"model_best{tag}.pt"
    history, best = [], {"test_ser": 1.0, "epoch": -1}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        total = 0.0
        ep_train = epoch_samples()
        for batch in make_batches(ep_train, rng):
            x, tg, in_lens, tg_lens = collate(batch, w2i, device,
                                              height=HEIGHT)
            logits = model(x)
            logp = logits.log_softmax(-1).permute(1, 0, 2)
            in_lens = torch.clamp(in_lens, max=logits.shape[1])
            loss = lossf(logp, tg, in_lens, tg_lens)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(batch)
        sched.step()
        mean = total / len(ep_train)

        test_ser, sel_pairs = evaluate(model, select, w2i, i2w, device)
        strata: dict[str, list[float]] = {}
        for s, hyp in sel_pairs:
            strata.setdefault("bach" if s["piece"].startswith("chor")
                              else "pdmx", []).append(ser(s["tokens"], hyp))
        by_corpus = {k: round(float(np.mean(v)), 5)
                     for k, v in sorted(strata.items())}
        history.append({"epoch": ep, "loss": round(mean, 4),
                        "test_ser": round(test_ser, 5),
                        "test_ser_by_corpus": by_corpus,
                        "minutes": round((time.time() - t0) / 60, 1)})
        print(f"  epoch {ep:3d}  loss {mean:.4f}  test SER {test_ser:.5f}  "
              f"{by_corpus}  ({(time.time()-t0)/60:.1f} min)", flush=True)

        if test_ser < best["test_ser"]:
            best = {"test_ser": test_ser, "epoch": ep}
            torch.save({"state_dict": model.state_dict(), "vocab": vocab,
                        "epoch": ep, "test_ser": test_ser,
                        "spacing": SPACING, "height": HEIGHT}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    full_ser, pairs = evaluate(model, test, w2i, i2w, device)

    (WORK / f"vocab_model{tag}.json").write_text(
        json.dumps({"tokens": vocab}, indent=1), encoding="utf-8")

    by_corpus = {}
    for s, hyp in pairs:
        key = "bach" if s["piece"].startswith("chor") else "pdmx"
        by_corpus.setdefault(key, []).append(ser(s["tokens"], hyp))

    out = {
        "source": source, "tag": tag,
        "spacing": SPACING, "height": HEIGHT,
        "mix": mix, "mscore_lines_train": n_mscore, "pack": use_pack,
        # What one epoch actually draws -- with `--mix` far less than the
        # training set (the ratio is held by subsampling the larger pool).
        # Registered runs compare this number across arms, not `lines_train`.
        "epoch_lines": n_epoch_lines, "epoch_lines_flag": epoch_lines,
        "pool_bach_lines": len(pool_bach), "pool_pdmx_lines": len(pool_pdmx),
        "reserved_pieces_excluded": n_blocked,
        # `pieces_total` and `pieces_train_before_source_filter` are the
        # unfiltered list: with split-before-filter they are identical in both
        # arms, and only `pieces_train` differs. That difference is the
        # treatment, and nothing else -- the point of the whole rearrangement.
        "pieces_total": len(pieces),
        "pieces_train_before_source_filter": n_train_unfiltered,
        "pieces_train": len(train_pieces), "pieces_test": len(test_pieces),
        # The names, not just the count. `piece_split` is deterministic in the
        # *list* it is given, so a corpus that has grown since produces a
        # different split -- and an evaluation that re-derives it would score the
        # model on pieces it was trained on. Writing the names down is what makes
        # the evaluation reproducible; the first run of this script did not, and
        # its split could not be recovered afterwards.
        "test_piece_names": sorted(test_pieces),
        "lines_train": len(train), "lines_test": len(test),
        "dropped_too_wide": len(drop_a["too_wide"]) + len(drop_b["too_wide"]),
        "dropped_target_longer_than_frames":
            len(drop_a["target_longer_than_frames"])
            + len(drop_b["target_longer_than_frames"]),
        "dropped_examples": (drop_a["target_longer_than_frames"]
                             + drop_b["target_longer_than_frames"])[:5],
        # Womit der Checkpoint ausgewaehlt wurde. Aeltere Laeufe tragen dieses
        # Feld nicht -- dort war es `test[:600]`, also 84,7 % Bach.
        "auswahl_kriterium": {
            "art": "zufallsstichprobe_des_vollen_testsatzes",
            "n": len(select), "seed": SELECT_SEED,
            "lines_by_corpus": dict(sorted(sel_by_corpus.items())),
            "geaendert_am": "2026-08-09 (Entscheid E1, "
                            "befund-bach-naechte-2026-08-09.md §6)"},
        "vocab_size": len(vocab), "vocabulary": vocab,
        "unseen_test_tokens": unseen,
        "params": n_params, "params_millions": round(n_params / 1e6, 2),
        "fp32_megabytes": round(n_params * 4 / 1e6, 1),
        "device": device, "torch": torch.__version__,
        "epochs": epochs, "batch_size": BATCH,
        "best_epoch": best["epoch"],
        "test_ser_best_subset": round(best["test_ser"], 5),
        "test_ser_full": round(full_ser, 5),
        "test_ser_by_corpus": {k: round(float(np.mean(v)), 5)
                               for k, v in by_corpus.items()},
        "lines_by_corpus": {k: len(v) for k, v in by_corpus.items()},
        "minutes": round((time.time() - t0) / 60, 1),
        "history": history,
    }
    (WORK / f"20_train{tag}.json").write_text(json.dumps(out, indent=2),
                                              encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("history", "vocabulary")}, indent=2))


if __name__ == "__main__":
    main()
    # Skip the CUDA teardown: after the result JSON is written there is
    # nothing left to save, and the cu130 teardown hung as a multi-GB zombie
    # on 2 of 2 runs (uebergabe-2026-07-31.md, Nacht-Vorfall).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
