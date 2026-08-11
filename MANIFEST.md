# provenance

every line here was checked against the license file of the source, not against
its readme or its paper. the distinction is not academic: transcoda claims cc by
4.0 in its paper and ships an AGPL license file.

## what is in this repo

| component | origin | license |
|---|---|---|
| model architecture, training, decode, preprocessing code | this project | MIT |
| `js/*.js` | this project | MIT |
| `model-public-domain/` weights and vocabulary | this project, trained on PDMX only | MIT |
| `model-cc-by-nc-sa/` weights and vocabulary | this project, trained on PDMX plus Bach chorales | CC BY-NC-SA 4.0 |
| `third_party/SMT/utils.py` | [PRAIG/smt-fp-grandstaff](https://github.com/PRAIG/smt-fp-grandstaff) | MIT, copyright 2023 Antonio Ríos-Vila |

only `parse_kern` from that file is used, by `scripts/10_build_lines.py`. the
license file of the source is included next to it unchanged.

## training data

| corpus | license | evidence |
|---|---|---|
| PDMX (Hugging Face mirror `openmusic/pdmx`) | `publicdomain` (public domain mark 1.0) and `cc-zero` (CC0 1.0) | per piece rows in `PDMX.csv`. across the whole dataset 210 364 `publicdomain` and 43 713 `cc-zero`, nothing else. archive MD5 `660944735e4545d1e3594f42ba933e42`. |
| [craigsapp/bach-370-chorales](https://github.com/craigsapp/bach-370-chorales) | CC BY-NC-SA 4.0 | `LICENSE.txt`: "copyright (c) 2009 Craig Stuart Sapp ... licensed with attribution-noncommercial-sharealike 4.0 international" |

the Bach corpus is used for `model-cc-by-nc-sa/` only. `model-public-domain/` is
trained with `--source=pdmx`, which removes all 370 chorales from the training
set. the split between train and test is computed before that filter runs, so
both models are tested on exactly the same held out material.

PDMX does not contain musicxml. it is MusPy style event JSON: pitch as MIDI
number, onset and duration in ticks, plus lyrics, time signature, key signature
and barlines. the conversion is in `src/synth/pdmx.py`.

## tools

these produce the training data. none of them contributes code to the released
artifacts.

| tool | license | role |
|---|---|---|
| Verovio 6.2.1 | LGPL-2.1 | renders kern to SVG |
| resvg 0.3.3 (`resvg-py`) | MPL-2.0 | SVG to PNG |
| MuseScore Studio 4 | GPL-3.0 | second renderer, for engraving variety |
| music21 10.5.0 | BSD-3 | kern to musicxml, because MuseScore does not read Humdrum |
| PyTorch, ONNX, onnxruntime | BSD-3, apache-2.0, MIT | training and export |

rendered bitmaps are not derivative works of the renderer in the copyleft sense,
and neither Verovio nor MuseScore code ends up in this repo.

## the sharealike question

whether trained weights are a derivative work of the training data is legally
unsettled. `model-cc-by-nc-sa/` does not try to settle it: it is released under
the same license as the Bach corpus, with attribution, which is what sharealike
would ask for if the answer turned out to be yes.

`model-public-domain/` avoids the question entirely. its training material
carries no obligations.

## what is deliberately absent

no rendered training data, no scores, no test fixtures built from purchased
material. reproducible here means the recipe, not the dataset. the scripts in
`scripts/` rebuild the corpus from the two public sources above.
