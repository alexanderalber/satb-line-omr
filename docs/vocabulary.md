# the vocabulary

the model does not emit kern tokens. it emits sub tokens, and a separator
between them. this page says what you get back and how to turn it into kern.

## why it is split

a kern note token carries duration, pitch, accidental and modifiers in one
string: `16A#JJ`. taken whole, that is 1442 classes over 4548 staves in the
original corpus, most of them seen once or twice. a CTC classifier cannot learn
a class it has seen once, and it does not have to: duration, pitch and modifiers
are independent and visually separable. splitting them brings the vocabulary
down to 141 classes for the MIT model and 153 for the CC BY-NC-SA one.

## what comes back

| kind | examples | notes |
|---|---|---|
| duration | `4`, `8`, `2.`, `16` | digits, optional augmentation dots |
| pitch | `c`, `cc`, `C`, `CC`, `r` | repetition encodes the octave: `CC` < `C` < `c` < `cc`. `r` is a rest |
| accidental | `#`, `##`, `-`, `n` | sharp, double sharp, flat, natural. these change the sounding pitch and are kept |
| modifier | `L`, `J`, `LL`, `JJ`, `[`, `]`, `(`, `)` | beam start and end, doubled for two beams, tie and slur brackets |
| whole token | `*clefG2`, `*k[f#]`, `*M4/4`, `=`, `=\|\|` | clefs, key signatures, meters, barlines. anything starting with `*`, `=`, `!` or `<` passes through unsplit |
| separator | `<b>` | marks the boundary between two kern tokens |

## joining

concatenate everything between two `<b>` separators:

```python
kern, cur = [], []
for t in subtokens:
    if t == "<b>":
        if cur:
            kern.append("".join(cur))
            cur = []
    else:
        cur.append(t)
if cur:
    kern.append("".join(cur))
```

`examples/run_line.py` does exactly this. do not reconstruct the boundaries with
a heuristic instead: `join_tokens` in `src/synth/tokens.py` tries and gets ties
wrong, for instance `4f# [2a` comes back as `4f#[ 2a`. the separator is what the
model was trained to place, so use it.

## what is deliberately not in there

- lyrics, dynamics, chord symbols, tempo marks, fingerings. the target is a
  practice MIDI: pitch, rhythm and which voice a note belongs to.
- `X` and `y`, the engraving instructions "print this accidental anyway" and
  "this rest is editorial". both were counted over the whole corpus first: 1890
  and 354 occurrences, together 0.57 percent of all sub tokens. they are
  instructions to an engraver, not sound, and in the case of `y` the image often
  does not show them at all.
- null tokens `.`, spine brackets, invisible rests `ryy`. ground truth may only
  contain what has ink on the page. otherwise two identical images carry
  different targets and the model is punished for a distinction it cannot see.

## voice assignment

there is none in the vocabulary, on purpose. the voice is the index of the staff
line inside its system, which the preprocessing hands you. a misread note costs
one correction in an editor. a misassigned voice is not visible to the user at
all, so it is not something a model should be allowed to guess.
