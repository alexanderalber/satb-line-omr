"""Verovio rendering plus read-out of the score structure it produced.

The whole point: verovio stamps the source coordinates of every element into its
svg id -- `note-L21F4` is kern line 21, field 4; `staff-L9F3` is field 3;
`measure-L22` is the measure that starts at line 22. So the mapping from a
rendered staff back to its kern tokens is *read out of the renderer*, not
reconstructed from barline geometry. That removes the segmentation guesswork the
phase-1 handoff flagged as its hardest part.

Verovio is LGPL and a build-time tool only: it renders training data and is never
shipped. Rendered bitmaps are not derivative works of the renderer.
"""
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import verovio

NS = "{http://www.w3.org/2000/svg}"
_ID_LF = re.compile(r"-L(\d+)(?:F(\d+))?")
_PATH_Y = re.compile(r"[ML]\s*-?\d+\s+(-?\d+)")

PAGE_W, PAGE_H = 2100, 2970

# Page header and footer are off: they are ink far away from any staff and the
# crop would never see them anyway.
#
# `mnumInterval: 0` does *not* suppress measure numbers -- verovio still prints
# one at the start of each system (visible in work/probe_chor002_p0.png). That
# was stated the other way round in phase 1. It is left as it is on purpose:
# real scores number their systems too, so this is ink without a target token,
# in the same category as a barline glyph, and removing it would make the
# training pages less like the target material rather than more.
OPTIONS = {
    "pageWidth": PAGE_W,
    "pageHeight": PAGE_H,
    "scale": 100,
    "adjustPageHeight": False,
    "footer": "none",
    "header": "none",
    "mnumInterval": 0,
    "svgViewBox": True,
    "breaks": "auto",
}


@dataclass
class Staff:
    field_no: int             # 1-based kern field, from the svg id
    y_top: float              # topmost staff line, in svg user units
    has_clef: bool = False
    has_keysig: bool = False
    has_metersig: bool = False


@dataclass
class System:
    page: int
    y_top: float
    measure_starts: list[int] = field(default_factory=list)
    staves: list[Staff] = field(default_factory=list)


def _cls(el) -> list[str]:
    return (el.get("class") or "").split()


def _first(el, name: str):
    return next((g for g in el.iter(NS + "g") if name in _cls(g)), None)


def _staff_y(staff_el) -> float:
    ys = [float(m) for p in staff_el.iter(NS + "path")
          for m in _PATH_Y.findall(p.get("d") or "")]
    return min(ys) if ys else 0.0


def make_toolkit(extra_options: dict | None = None):
    verovio.enableLog(verovio.LOG_OFF)
    tk = verovio.toolkit()
    opts = dict(OPTIONS)
    if extra_options:
        opts.update(extra_options)
    tk.setOptions(opts)
    return tk


def render_svg(tk, krn_text: str) -> list[str]:
    """kern -> one svg string per page."""
    if not tk.loadData(krn_text):
        raise ValueError("verovio could not load this kern data")
    return [tk.renderToSVG(p) for p in range(1, tk.getPageCount() + 1)]


def parse_structure(svgs: list[str]) -> list[System]:
    """Read systems, staves and measure starts out of the rendered svg."""
    systems: list[System] = []

    for page, svg in enumerate(svgs):
        root = ET.fromstring(svg)
        for sys_el in (g for g in root.iter(NS + "g") if "system" in _cls(g)):
            measures = [m for m in sys_el.iter(NS + "g") if "measure" in _cls(m)]
            starts = []
            for m in measures:
                hit = _ID_LF.search(m.get("id") or "")
                if hit:
                    starts.append(int(hit.group(1)))

            staves: list[Staff] = []
            # A staff appears once per measure; the first measure is the one that
            # carries the clef/key/meter that verovio decided to draw here.
            if measures:
                for st_el in (g for g in measures[0].iter(NS + "g")
                              if "staff" in _cls(g)):
                    hit = _ID_LF.search(st_el.get("id") or "")
                    if not hit or not hit.group(2):
                        continue
                    staves.append(Staff(
                        field_no=int(hit.group(2)),
                        y_top=_staff_y(st_el),
                        has_clef=_first(st_el, "clef") is not None,
                        has_keysig=_first(st_el, "keySig") is not None,
                        has_metersig=_first(st_el, "meterSig") is not None,
                    ))

            if not staves:
                continue
            staves.sort(key=lambda s: s.y_top)
            systems.append(System(page=page, y_top=min(s.y_top for s in staves),
                                  measure_starts=sorted(set(starts)),
                                  staves=staves))

    systems.sort(key=lambda s: (s.page, s.y_top))
    return systems


def rasterize(svg: str) -> np.ndarray:
    """svg string -> float32 (H, W) grayscale in [0,1], white background.

    Same convention as everywhere else in this repo: grayscale, /255, nothing
    else. resvg is used rather than cairosvg because it needs no system cairo
    on Windows and renders verovio's <use>/<symbol> glyph output correctly.
    """
    import io

    import resvg_py
    from PIL import Image

    png = bytes(resvg_py.svg_to_bytes(svg_string=svg))
    img = Image.open(io.BytesIO(png))
    if img.mode == "RGBA":                       # resvg gives transparent paper
        flat = Image.new("RGB", img.size, (255, 255, 255))
        flat.paste(img, mask=img.split()[3])
        img = flat
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0
