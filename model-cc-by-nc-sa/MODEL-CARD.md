# model card: CC BY-NC-SA line model

CRNN plus CTC, 3 211 994 parameters, trained from scratch. same architecture and
same recipe as the MIT model in `model-public-domain/`, with one difference: the
Bach 370 chorales are in the training set. that is why this model is CC BY-NC-SA
4.0 and the other one is not.

**attribution:** trained in part on the Bach 370 chorales by Craig Stuart Sapp,
CC BY-NC-SA 4.0. sharealike applies to these weights.

## files

| file | size | notes |
|---|---|---|
| `omr-line.fp32.onnx` | 12.85 MB | reference variant |
| `omr-line.fp16.onnx` | 6.43 MB | browser default, float32 IO |
| `omr-line.int8.onnx` | 3.26 MB | dynamic quantisation, same accuracy, five times slower under WASM |
| `weights.safetensors` | 12.85 MB | raw state_dict for `src/train/model.py`, for further training or surgery |
| `vocab.json` | | 153 tokens, index to token, model specific |
| `model-io.json` | | machine readable IO contract, with SHA-256 per file |

the vocabulary is not interchangeable with the MIT model. it has 153 entries
instead of 141 and a different index mapping.

## the three variants are the same model

measured, not assumed: every variant scored on the full held out test set.

| variant | symbol error rate | lines read exactly |
|---|---|---|
| fp32 | 0.00394 | 90.1 percent |
| fp16 | 0.00394 | 90.1 percent |
| int8 | 0.00394 | 90.1 percent |

quantisation costs nothing here. pick by size and speed, not by accuracy.
PyTorch and ONNX Runtime agree exactly on this checkpoint: 22187 lines compared, 0 mismatching.

## training

| | |
|---|---|
| corpus | PDMX plus Bach 370 chorales, `--mix=1:2` |
| training lines | 278 948, of which 57 779 rendered with MuseScore |
| epochs | 40, batch 8, 19 596 lines per epoch |
| checkpoint | epoch 35, selected on a seed fixed random sample of 600 test lines |
| hardware | one RTX 3090, 119 minutes |
| vocabulary | 153 tokens, built from the training set |

note on the mixture: Bach is 2.3 percent of the corpus but 33 percent of every
epoch, because `--mix` subsamples the larger pool to hold the ratio. the model
sees chorale style writing far more often than the corpus proportion suggests.
that is the intended behaviour here and it is the reason the two models differ as
much as they do.

## accuracy

symbol error rate, lower is better. held out test set, 22 187 lines, identical to
the one used for the MIT model.

| stratum | lines | this model | MIT model |
|---|---|---|---|
| Bach chorales | 508 | 0.00350 | 0.06618 |
| PDMX (pop, musical, general choral) | 21 679 | 0.00395 | 0.00396 |
| combined | 22 187 | 0.00394 | 0.00539 |

second, independent set of 58 reserved pieces, two engravers:

| renderer | stratum | lines | this model | MIT model |
|---|---|---|---|---|
| MuseScore | Bach | 356 | 0.00919 | 0.09958 |
| MuseScore | PDMX | 1 532 | 0.00588 | 0.00627 |
| Verovio | Bach | 292 | 0.00459 | 0.06089 |
| Verovio | PDMX | 1 568 | 0.00197 | 0.00203 |

lines read exactly right: 87.8 percent MuseScore, 90.9 percent Verovio.

read this as: this model is worth its license obligation only for chorale and
hymn style writing, where it is roughly ten to nineteen times more accurate. on
everything else the MIT model is the same. if your material is pop, musical or
general choral, take the MIT model and avoid the obligation.

## limits of these numbers

identical to the MIT model card, and they matter here too:

- training is not deterministic, `ctc_loss_backward_gpu` has no deterministic cuda
  implementation. two runs of the same recipe differ by up to 20 percent relative
  in test symbol error rate. differences below that are noise.
- the test material is rendered, not printed or scanned.
- the advantage on Bach is measured on rendered Bach specifically. it is
  plausible but not measured that it carries over to hymnals and other chorale
  style engraving.
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

the measured page was pop material, not chorale. it says nothing about this
model's advantage on Bach, in either direction.

## intended use

chorale, hymn and early music material, non commercially. for musicology and
church music the nc clause is usually not an obstacle, which is why this model is
published at all rather than kept private.

for anything commercial, use `model-public-domain/`.
