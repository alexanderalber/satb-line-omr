# satb-line-omr

optical music recognition for four-part choral scores, one staff line at a time.
a small CRNN with CTC loss, 3.2 M parameters, 12.8 MB as fp32 ONNX. runs in a
browser worker via ONNX Runtime Web, no server, no Python at inference time.

the point is not a benchmark rank. the point is that there is no freely usable
choral OMR model: what exists and works well is GPL or AGPL, which blocks reuse.
this repo is the same work done on public domain training material, released under
MIT.

## two models, two licenses

| directory | license | training data | use it when |
|---|---|---|---|
| `model-public-domain/` | MIT | PDMX only (public domain / CC0) | default. no obligations attached. |
| `model-cc-by-nc-sa/` | CC BY-NC-SA 4.0 | PDMX plus the Bach 370 chorales | you work on chorale or hymn style material, non commercially |

the licenses live in the directory names on purpose. the weights in
`model-cc-by-nc-sa/` are derived from Craig Stuart Sapp's Bach chorale corpus,
which is CC BY-NC-SA 4.0. that obligation travels with them. everything else in
this repo, including all code, is MIT.

the difference between the two models is one material class and nothing else.
measured on a held out test set:

| stratum | MIT model | NC model | ratio |
|---|---|---|---|
| pop, musical, general choral (21 679 lines) | 0.00396 | 0.00395 | 1.00x |
| Bach chorales (508 lines) | 0.06618 | 0.00350 | 18.9x worse |

so: on chorale style material the NC model is far better. on everything else the
two are indistinguishable. numbers are symbol error rate, lower is better. the
full picture, including two independent renderers and the limits of these
numbers, is in the model cards.

## what it does

input is one staff line, cropped, 128 pixels high, any width. output is a token
sequence in a Humdrum kern derived vocabulary: pitches, durations, rests, clefs,
key and time signatures, barlines. voice assignment is not learned, it is the line
index inside the system, which is why a wrong note is cheap and a wrong structure
does not happen.

lyrics, dynamics and chord symbols are deliberately not in the vocabulary.

## why it looks like this

every design decision here follows from one constraint: it has to run in a
browser tab, on the machine of whoever uploaded the PDF, with no server and no
request to anyone. that is a privacy promise, not a preference, and it is also
the cheapest way to host a tool nobody wants to pay for.

what follows from it:

- **no autoregressive decoder.** the first attempt used a seq2seq model with
  cross attention over about 24 000 memory positions. measured, that is 155 ms
  per decoded token, which is roughly 20 hours per page. a staff line is
  monotonic left to right, so it does not need attention that can look anywhere:
  CTC reads it in one forward pass, with no KV cache and no prefix recompute.
- **small enough to ship.** the whole model is 3.2 M parameters, 6.4 MB as fp16.
  the budget it was built against is 40 MB for all weights together and 2 MB per
  JS module, because they sit in the repo that serves the page.
- **preprocessing in plain array operations.** no OpenCV, no WASM of our own,
  nothing that would need a build step. `src/omr/staves.py` is deliberately
  written in flat NumPy: what cannot be expressed that way cannot be shipped as
  JS either, so the constraint is visible in the reference implementation
  instead of being discovered during the port.
- **line by line, not page at once.** the voice of a note is the index of its
  staff line, which is geometry rather than something learned. that is what
  keeps the failure mode cheap: single wrong notes, which a user sees and fixes,
  instead of wrong voice assignment, which a user does not see at all.

it is fast enough that the constraint stopped mattering. measured under ONNX
Runtime Web on an AMD Ryzen 7 5800X3D, fp16, twelve staff lines per page:

| threads | per line | per page | model |
|---|---|---|---|
| 1 | 176 ms | 2.2 s | MIT |
| 1 | 168 ms | 2.1 s | CC BY-NC-SA |
| 4 | 49 ms | 0.6 s | MIT |
| 4 | 52 ms | 0.6 s | CC BY-NC-SA |

both weight files were measured, not one of them extrapolated to the other.

one thread is the number that counts: browser multithreading needs
`SharedArrayBuffer`, which needs COOP and COEP response headers, which GitHub
Pages cannot set. thread counts above one are upside, not the default. in the
real tool a page takes about two and a half seconds end to end, model plus
preprocessing.

## repo layout

```
js/                 dependency free ES modules, no DOM, no WASM of our own
                    omr-preprocess.js  binarise, deskew, find staff lines, crop
                    omr-decode.js      greedy CTC decode, tokens to score IR
                    omr-reject.js      scan and closed score detection
src/omr/            python reference for the same three stages, plus staves.py
src/train/          model definition and dataset
src/synth/          training data generation from kern and PDMX JSON
scripts/            the recipe, numbered in the order it runs
examples/           runnable example, with one public domain line and its ground truth
docs/               what the output tokens mean
third_party/SMT/    one file (utils.py, parse_kern) from PRAIG/smt-fp-grandstaff, MIT
model-*/            weights, vocabulary, model card, license
```

the JS modules and the Python modules are kept in parity by test harnesses in the
originating repo. they are not a port maintained by hand: identical output on the
same input is a release condition.

## using the weights

```js
import * as ort from 'onnxruntime-web'
const session = await ort.InferenceSession.create('model-public-domain/omr-line.fp16.onnx')
// input: float32, shape [batch, 1, 128, width]
// values are inverted: ink is near 1.0, paper is 0.0. feeding un-inverted
// images produces garbage that looks like an export bug.
const out = await session.run({input})
// out.logits: [batch, floor(width/4), vocab+1]
// class 0 is the CTC blank, class i>0 is vocabulary token i-1
```

the same thing in Python, runnable, about sixty lines including comments:

```
python examples/run_line.py                     # the example line shipped in examples/
python examples/run_line.py my-line.png 18.0    # your own crop, staff spacing in px
```

`omr-decode.js` does the decode and the mapping back to token strings. the
vocabulary is per model, it ships next to the weights, and it must not be shared
between the two models: the index to token mapping differs. what the tokens mean
and how to turn them back into kern is in `docs/vocabulary.md`.

three variants per model:

| variant | size | when |
|---|---|---|
| `.fp32.onnx` | 12.8 MB | reference |
| `.fp16.onnx` | 6.4 MB | browser default |
| `.int8.onnx` | 3.3 MB | half the size of fp16 and the same accuracy, but five times slower under WASM (880 ms per line single threaded). only worth it if download size beats latency for you |

for training or surgery in PyTorch there is `weights.safetensors` next to them,
the raw state_dict for the `CRNN` class in `src/train/model.py`. loading it and
loading the original checkpoint produce bit identical output, that was checked
rather than assumed. there is deliberately no `.pt`: a torch checkpoint is a
Python pickle, and unpickling runs code, which is not something a stranger's
weights should ask of you.

## reproducing the training

the recipe is here, the data is not. rendering it takes a few hours of CPU and
about 40 GB of disk.

```
scripts/19_pdmx_to_kern.py      PDMX event JSON to kern
scripts/10_build_lines.py       render with verovio, cut into lines, build ground truth
scripts/31_build_mscore_lines.py  second renderer, musescore
scripts/40_pack_lines.py        memmap pack for training
scripts/20_train.py 40 8 --mix=1:2 --mscore --pack        the NC model
scripts/20_train.py 40 8 --epoch-lines=19596 --mscore --pack --source=pdmx   the MIT model
scripts/34_export_line_onnx.py  ONNX export plus parity against pytorch
scripts/36_quantize.py          fp16 and int8, each measured, not assumed
```

`--source=pdmx` is the switch that keeps Bach out. it is the reason the MIT model
exists.

training took about two hours per model on one RTX 3090. it does not need one.
measured peak memory, batch 8, one full step including CTC backward and the
optimizer:

| case | width | VRAM peak |
|---|---|---|
| training, batch 8 | 1400 px, the training cap | 1.2 GB allocated, 1.7 GB reserved |
| training, batch 8 | 1078 px, the corpus median | 0.9 GB allocated, 1.3 GB reserved |
| inference, batch 1 | 1400 px | 80 MB allocated, 230 MB reserved |

so a 4 GB card trains this, a 2 GB card runs it, and the 3090 it was built on is
idling. host RAM during training is about 0.7 GB working set: the corpus is
served from a memmap, so the process holds batches and an index, not the data.
disk is the real requirement, roughly 43 GB for both rendered packs.

training is not reproducible bit for bit: `ctc_loss_backward_gpu` has no
deterministic CUDA implementation, so two runs of the same recipe on the same
data differ by up to 20 percent in test symbol error rate. do not read small
differences between two runs as an effect.

## what this model is not

- not a full page system. it reads one line. staff detection is in
  `omr-preprocess.js`, everything above that is your job.
- not trained on scans. it expects engraved output, printed or rendered. scanned
  material is detected and flagged by `omr-reject.js` rather than silently read
  badly.
- not a piano or orchestral model. one voice per staff is the assumption. the
  training material is four staves per system, so five to eight voice writing is
  within the architecture but outside what the model has seen. closed score, two
  staves carrying four voices, is detected and rejected rather than read wrong.
- not benchmarked against Audiveris or homr. those are GPL or AGPL and were
  excluded on license grounds, not on measured weakness.

## credits

the NC model is trained in part on the [Bach 370 chorales](https://github.com/craigsapp/bach-370-chorales)
by Craig Stuart Sapp, CC BY-NC-SA 4.0.

`third_party/SMT/utils.py` is from [PRAIG/smt-fp-grandstaff](https://github.com/PRAIG/smt-fp-grandstaff),
MIT, copyright 2023 Antonio Ríos-Vila. only the kern parser is used.

the public domain training material is PDMX (Hugging Face mirror `openmusic/pdmx`).

see `MANIFEST.md` for the full provenance chain with sources.

built with Claude Code. 6 million tokens died for this.
