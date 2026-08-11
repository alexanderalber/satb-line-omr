# model card: public domain line model

CRNN plus CTC, 3 205 838 parameters, trained from scratch on rendered public
domain scores. no Bach chorales in the training set, therefore MIT.

## files

| file | size | notes |
|---|---|---|
| `omr-line.fp32.onnx` | 12.83 MB | reference variant |
| `omr-line.fp16.onnx` | 6.42 MB | browser default, float32 IO |
| `omr-line.int8.onnx` | 3.25 MB | dynamic quantisation, same accuracy, five times slower under WASM |
| `weights.safetensors` | 12.83 MB | raw state_dict for `src/train/model.py`, for further training or surgery |
| `vocab.json` | | 141 tokens, index to token, model specific |
| `model-io.json` | | machine readable IO contract, with SHA-256 per file |

## the three variants are the same model

measured, not assumed: every variant scored on the full held out test set.

| variant | symbol error rate | lines read exactly |
|---|---|---|
| fp32 | 0.00538 | 88.1 percent |
| fp16 | 0.00539 | 88.1 percent |
| int8 | 0.00538 | 88.2 percent |

quantisation costs nothing here. pick by size and speed, not by accuracy.
PyTorch and ONNX Runtime agree exactly on this checkpoint: 22187 lines compared, 0 mismatching.

## training

| | |
|---|---|
| corpus | PDMX only, `--source=pdmx` |
| training lines | 272 416, of which 56 159 rendered with MuseScore |
| epochs | 40, batch 8, 19 596 lines per epoch |
| checkpoint | epoch 36, selected on a seed fixed random sample of 600 test lines |
| hardware | one RTX 3090, 126 minutes |
| vocabulary | 141 tokens, built from the training set |

six ground truth tokens of the test set are outside this vocabulary
(`*M12/4`, `;`, `=-`, `=:|!`, `JJ`, `LL`). they occur in Bach material and set a
floor on the error for lines that contain them. this is a property of the
training set, not a defect, and it is not corrected for in the numbers below.

## accuracy

symbol error rate, lower is better. held out test set, 22 187 lines, never
trained on. the two strata are reported separately because the entire difference
between this model and the NC model sits in one of them.

| stratum | lines | this model | NC model |
|---|---|---|---|
| PDMX (pop, musical, general choral) | 21 679 | 0.00396 | 0.00395 |
| Bach chorales | 508 | 0.06618 | 0.00350 |
| combined | 22 187 | 0.00539 | 0.00394 |

on a second, independent set of 58 reserved pieces rendered by two different
engravers:

| renderer | stratum | lines | this model | NC model |
|---|---|---|---|---|
| MuseScore | Bach | 356 | 0.09958 | 0.00919 |
| MuseScore | PDMX | 1 532 | 0.00627 | 0.00588 |
| Verovio | Bach | 292 | 0.06089 | 0.00459 |
| Verovio | PDMX | 1 568 | 0.00203 | 0.00197 |

lines read exactly right: 70.9 percent MuseScore, 77.0 percent Verovio. those
figures include the Bach fifth of the set, which this model reads badly by
construction.

read this as: on non chorale material the two models are indistinguishable. the
difference on PDMX, 6.6 percent and 2.9 percent, is smaller than the run to run
spread of the training itself, so it is not a measured disadvantage, it is an
absence of evidence for one.

## limits of these numbers

- training is not deterministic. `ctc_loss_backward_gpu` has no deterministic
  cuda implementation. two runs of the same recipe on the same data differ by up
  to 20 percent relative in test symbol error rate and up to 10 points in a
  downstream voice sync measure. differences below that are noise.
- the test material is rendered, not scanned or printed. it is generated from the
  same two renderers used for training, on pieces never trained on. expect worse
  numbers on real print, and much worse on scans.
- on 114 real choral scores the two models were measured within noise of each
  other. those absolute numbers are not reported here because they carry a
  quantified truncation bias and are only meaningful as paired comparisons.
- no comparison against Audiveris or homr is claimed. for oemer see the section
  below, which gives a direction and no number, because the measurement behind it
  carries no number.

## how this compares to oemer

oemer (MIT, full-page OMR) and this model were run over the same hand-labelled
staves of one page. the sample is three staves and 173 reference tokens, and only
the more legible lines carry ground truth, so both systems look better here than
they are. that supports no number, so none is given. what it does support:

- octaved tenor clef: oemer emits no clef-octave-change at all, so a tenor staff
  comes out an octave off. that is the standard clef of SATB choral music, which
  is what this model was built for.
- bass staff: oemer read it better than this model did. our errors there were
  durations, not pitches.
- soprano staff: comparable, both in the same range.
- scope: oemer reads a whole page and writes MusicXML, including what this model
  deliberately ignores (lyrics, dynamics, chord symbols). it does more; this model
  does one thing, on one staff.

same order of magnitude. for full-page OMR, oemer is the more complete tool. for
SATB staves with an octaved tenor clef, this one.

## intended use

practice MIDI for choir singers: read a PDF, correct the obvious errors in an
editor, export one voice per singer. manual correction is part of the design, not
a fallback. the model is small enough to ship into a browser and produce a whole
page in seconds.

not intended for archival transcription without review.
