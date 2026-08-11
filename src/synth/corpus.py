"""One place that decides what a piece looks like, so every script agrees.

The ground-truth check (`15_count_check.py`) re-renders a piece from the source
file and compares against what the corpus builder stored. That only works if
both produce byte-identical kern, so the random realism decisions must be a
function of the piece name and a seed, never of call order.
"""
import random
from dataclasses import dataclass

from . import kern as kern_mod
from . import realism as realism_mod

SEED = 20260729


@dataclass
class Prepared:
    text: str                 # the kern actually rendered
    kern_fields: list[int]    # 1-based field index per original spine
    config: realism_mod.RealismConfig | None
    options: dict             # verovio options this piece is rendered with

    @property
    def font(self) -> str:
        return self.options.get("font", "Leipzig")

    @property
    def key(self) -> tuple:
        """Hashable identity of the render options, for toolkit caching."""
        return tuple(sorted(self.options.items()))


def verses_of(piece: str, seed: int = SEED) -> int:
    """Verse count from its own stream, so enabling it does not move any other
    realism decision. Same distribution as `31_build_mscore_lines.verses_of`
    (pop-package decision, 02.08.): a third single-verse, the rest 2-4."""
    return random.Random(f"{seed}:verses:{piece}").choice([1, 1, 2, 2, 3, 4])


def prepare(piece: str, raw_text: str, seed: int = SEED,
            realism: bool = True, n_spines: int = 4,
            multi_verse: bool = False) -> Prepared:
    """Source kern -> the augmented kern this piece is rendered from.

    `multi_verse` is opt-in: every render is byte-identical to the pre-package
    state until a builder or evaluation explicitly flips it, so comparisons
    against earlier checkpoints stay single-variable (Auflage 4, 02.08.).
    """
    base = kern_mod.drop_labels(raw_text)
    if not realism:
        return Prepared(text=base, kern_fields=list(range(1, n_spines + 1)),
                        config=None, options={})

    rng = random.Random(f"{seed}:{piece}")
    cfg = realism_mod.choose(rng, n_spines=n_spines)
    if multi_verse:
        cfg.verses = verses_of(piece, seed)
    aug = realism_mod.augment(base, cfg, rng)
    return Prepared(text=aug.text, kern_fields=aug.kern_fields,
                    config=cfg, options=cfg.render_options())
