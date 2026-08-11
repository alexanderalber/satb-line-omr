"""Split kern tokens into sub-tokens -- the step `bekern` assumes has happened.

`parse_kern(krn, "bekern")` splits on the `·` and `@` markers that FP-GrandStaff
carries in its source files. The Bach chorales have no such markers, so every
note stays monolithic and the vocabulary explodes: 1442 classes over 4548
staves, a long tail of them seen exactly once (`16A#JJ`, `16AnXLL`, ...). A CTC
classifier cannot learn a class it sees once, and does not have to: duration,
pitch and modifiers are independent, visually separable properties.

So the same decomposition is done here explicitly. Everything that is not a note
or rest -- clefs, keys, meters, barlines, structural markers -- is left whole.
"""
import re

# duration: digits with optional augmentation dots; pitch: repeated letter,
# where the repetition encodes the octave (GG < G < g < gg); then accidental;
# whatever remains are single-character modifiers (beams, ties, slurs, marks).
_NOTE = re.compile(
    r"^(?P<pre>[\[\(]*)"
    r"(?P<dur>\d+\.*)"
    r"(?P<pitch>[a-gA-G]+|r)"
    r"(?P<acc>(?:#+|-+|n)?)"
    r"(?P<rest>.*)$"
)

PASS_THROUGH_PREFIXES = ("*", "=", "!", "<")

# Engraving instructions, not sound. `X` forces an accidental to be printed
# (`4gnX` sounds like `4gn`), `y` marks a rest editorially (`4ry` sounds like
# `4r`). Both were measured over the whole corpus before being dropped: X 1890
# occurrences, y 354, together 0.57 % of all sub-tokens, and X sits after an
# accidental in 1881 of its 1890 cases (`_probe_xy_context.py`). Keeping them
# would ask the model to reproduce a distinction a practice MIDI cannot hear --
# and possibly one the image does not even show. Accidentals themselves
# (`n`, `#`, `-`) stay: a natural changes the sounding pitch.
DROPPED_MODIFIERS = frozenset("Xy")


def split_token(tok: str) -> list[str]:
    """One kern token -> its sub-tokens, in reading order."""
    if not tok or tok.startswith(PASS_THROUGH_PREFIXES):
        return [tok]

    m = _NOTE.match(tok)
    if not m:
        return [tok]

    out: list[str] = []
    out += list(m.group("pre"))
    out.append(m.group("dur"))
    out.append(m.group("pitch"))
    if m.group("acc"):
        out.append(m.group("acc"))
    # Modifiers are single characters, except the doubled beam marks LL/JJ which
    # mean "two beams" and are a different symbol from L/J.
    tail = m.group("rest")
    for run in re.findall(r"L+|J+|.", tail):
        if run in DROPPED_MODIFIERS:
            continue
        out.append(run)
    return out


def canonical_token(tok: str) -> str:
    """The token as the ground truth carries it, i.e. without the dropped hints.

    The split is a pure decomposition -- concatenating its output reproduces the
    input -- so joining the sub-tokens of a single token yields exactly the
    canonical spelling. This is the reference `join_tokens` has to reach.
    """
    return "".join(split_token(tok))


def canonical_tokens(tokens: list[str]) -> list[str]:
    return [canonical_token(t) for t in tokens]


def token_pitch(tok: str) -> str | None:
    """The pitch of a note token, `r` for a rest, None if it is neither.

    Reads the regex groups rather than indexing into split_token's output: a
    token opening a tie or slur (`[4g`) puts brackets in front, so positional
    indexing silently returns the duration instead.
    """
    if not tok or tok.startswith(PASS_THROUGH_PREFIXES):
        return None
    m = _NOTE.match(tok)
    return m.group("pitch") if m else None


def split_tokens(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        out.extend(split_token(tok))
    return out


def join_tokens(subtokens: list[str]) -> list[str]:
    """Inverse of split_tokens, so a prediction can be turned back into kern.

    A new token starts at anything that is a pass-through, an opening bracket or
    a duration; everything after it attaches to it.
    """
    out: list[str] = []
    for sub in subtokens:
        starts_new = (
            not out
            or sub.startswith(PASS_THROUGH_PREFIXES)
            or out[-1].startswith(PASS_THROUGH_PREFIXES)
            or re.match(r"^\d+\.*$", sub) and not re.match(r"^[\[\(]+$", out[-1])
        )
        if starts_new:
            out.append(sub)
        else:
            out[-1] += sub
    return out
