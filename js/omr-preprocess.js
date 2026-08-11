/* omr-preprocess.js -- grayscale page -> normalised model-input line tensors.
 *
 * Port of src/omr/staves.py + src/omr/normalize.py in the model repo; the
 * Python files are the reference, this file must reproduce their output
 * (verified by scripts/48_preprocess_parity.mjs against dist/fixtures/:
 * boxes exact, tensors within max abs diff 1e-4).
 *
 * Plain ES module: no imports, no DOM, no globals, runs under Node and in a
 * Web Worker. All inputs are typed arrays / plain objects.
 *
 * Float semantics: numpy computes the bilinear sampling grid in float32.
 * Wherever a float32 rounding decides an integer (floor of a sample
 * coordinate, the ink threshold), this port forces float32 via Math.fround so
 * the decision cannot differ from the reference. Interpolation weights ride
 * in doubles -- the difference is orders of magnitude below the contract
 * tolerance -- and results are stored into Float32Arrays, which rounds again.
 *
 * API (mirrors the Python call chain):
 *   grayFromBytes(u8, n)                  -> Float32Array in [0,1] (v/255)
 *   detect(gray, width, height)           -> { lineRuns, staves, systems, boxes }
 *   normalizeStaff(crop, cw, ch, lineSpacing, staffOffset, opts)
 *                                         -> { data, width, height }
 *   preprocessPage(gray, width, height, opts)
 *                                         -> { boxes, inputs } -- inputs are the
 *                                            inverted float32 model tensors,
 *                                            one per staff line, plus geometry
 */

export const INK_THRESHOLD = 0.6;
export const TARGET_SPACING = 10.0;
export const TARGET_HEIGHT = 128;
export const MAX_WIDTH = 1400;      /* at spacing 10; scales linearly */

const fr = Math.fround;

/* Python round(): round-half-even, to an integer. int(round(x)) in the
 * reference; Math.round rounds .5 away from zero and would drift paddings. */
function pyRoundInt(x) {
  const f = Math.floor(x);
  const diff = x - f;
  if (diff > 0.5) return f + 1;
  if (diff < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

export function grayFromBytes(bytes, n) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = fr(bytes[i] / 255);
  return out;
}

/* gray: Float32Array (height*width, row-major), values in [0,1], white = 1.
 * Returns Uint8Array, 1 = ink. Threshold compared in float64; every stored
 * gray value is a float32, and no float32 lies between f64(0.6) and f32(0.6),
 * so the comparison matches numpy's float32 compare bit for bit. */
export function binarize(gray, width, height, threshold = INK_THRESHOLD) {
  const ink = new Uint8Array(width * height);
  for (let i = 0; i < gray.length; i++) ink[i] = gray[i] < threshold ? 1 : 0;
  return ink;
}

/* Longest uninterrupted horizontal ink run per row (staves.py::longest_run).
 * Only rows whose total ink already clears `threshold` are walked -- a run can
 * never beat the row sum, so the shortcut changes no result. */
function longestRuns(ink, width, height, threshold) {
  const best = new Int32Array(height);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let sum = 0;
    for (let x = 0; x < width; x++) sum += ink[row + x];
    if (sum < threshold) continue;
    let run = 0, b = 0;
    for (let x = 0; x < width; x++) {
      run = ink[row + x] ? run + 1 : 0;
      if (run > b) b = run;
    }
    best[y] = b;
  }
  return best;
}

/* Rows that look like staff lines, as [start, end] row ranges. */
export function findStaffLines(ink, width, height, minRunFrac = 0.10) {
  const threshold = minRunFrac * width;
  const best = longestRuns(ink, width, height, threshold);
  const runs = [];
  let start = null;
  for (let y = 0; y < height; y++) {
    const v = best[y] >= threshold;
    if (v && start === null) start = y;
    else if (!v && start !== null) { runs.push([start, y - 1]); start = null; }
  }
  if (start !== null) runs.push([start, height - 1]);
  return runs;
}

function evenlySpaced(vals, tolerance) {
  let mn = Infinity, mx = -Infinity, sum = 0;
  for (let k = 0; k + 1 < vals.length; k++) {
    const gap = vals[k + 1] - vals[k];
    if (gap < mn) mn = gap;
    if (gap > mx) mx = gap;
    sum += gap;
  }
  if (mn <= 0) return false;
  return (mx - mn) <= tolerance * (sum / (vals.length - 1));
}

/* Do these five runs sit at equal distance under *any* reading (middle, top
 * edge, bottom edge)? One agreement is enough -- see staves.py::_fits. */
function fits(anchors, idx, tolerance) {
  for (const a of anchors) {
    const vals = idx.map((j) => a[j]);
    if (evenlySpaced(vals, tolerance)) return true;
  }
  return false;
}

/* One greedy left-to-right walk; may skip one intruder run (tie/slur) when
 * five consecutive runs do not fit. */
function groupPass(count, anchors, tolerance) {
  const picked = [];
  let i = 0;
  while (i + 4 < count) {
    const straight = [i, i + 1, i + 2, i + 3, i + 4];
    if (fits(anchors, straight, tolerance)) {
      picked.push(straight);
      i += 5;
      continue;
    }
    let found = null;
    if (i + 5 < count) {
      for (let skip = 0; skip < 6; skip++) {
        const idx = [];
        for (let j = i; j < i + 6; j++) if (j !== i + skip) idx.push(j);
        if (fits(anchors, idx, tolerance)) { found = idx; break; }
      }
    }
    if (found) {
      picked.push(found);
      i = found[found.length - 1] + 1;
    } else {
      i += 1;
    }
  }
  return picked;
}

export function groupLinesIntoStaves(runs, tolerance = 0.45) {
  if (runs.length < 5) return [];
  const tops = runs.map((r) => r[0]);
  const bottoms = runs.map((r) => r[1]);
  const centers = runs.map((r) => (r[0] + r[1]) / 2.0);
  const anchors = [centers, tops, bottoms];

  /* Read the spacing off whichever edge is most even for this staff. */
  function spacingOf(idx) {
    let bestSpread = Infinity, bestMean = 0;
    for (const a of anchors) {
      let mn = Infinity, mx = -Infinity, sum = 0;
      for (let k = 0; k < 4; k++) {
        const gap = a[idx[k + 1]] - a[idx[k]];
        if (gap < mn) mn = gap;
        if (gap > mx) mx = gap;
        sum += gap;
      }
      const spread = mx - mn;
      if (spread < bestSpread) { bestSpread = spread; bestMean = sum / 4.0; }
    }
    return bestMean;
  }

  return groupPass(runs.length, anchors, tolerance).map((idx) => ({
    top: runs[idx[0]][0],
    bottom: runs[idx[4]][1],
    lineSpacing: spacingOf(idx),
    lineCenters: idx.map((j) => centers[j]),
  }));
}

/* Staves joined by a vertical rule through the gap between them belong to the
 * same system. Returns arrays of staff indices. */
export function findSystems(ink, width, staves, minSpanFrac = 0.8) {
  if (staves.length === 0) return [];
  const systems = [[0]];
  for (let i = 1; i < staves.length; i++) {
    const gapTop = staves[i - 1].bottom + 1;
    const gapBot = staves[i].top - 1;
    let joined = false;
    if (gapBot > gapTop) {
      const rows = gapBot - gapTop + 1;
      for (let x = 0; x < width && !joined; x++) {
        let colSum = 0;
        for (let y = gapTop; y <= gapBot; y++) colSum += ink[y * width + x];
        if (colSum / rows >= minSpanFrac) joined = true;
      }
    }
    if (joined) systems[systems.length - 1].push(i);
    else systems.push([i]);
  }
  return systems;
}

/* How far to extend a crop away from the staff: grow through connected ink,
 * stop at the first clear gap of `gapRatio` spacings (staves.py::grow_edge). */
export function growEdge(ink, width, height, x0, x1, yStart, direction, spacing,
                         limit = null,
                         minPadRatio = 2.0, maxPadRatio = 3.0, gapRatio = 0.8) {
  const minPad = pyRoundInt(minPadRatio * spacing);
  let maxPad = pyRoundInt(maxPadRatio * spacing);
  if (limit !== null) maxPad = Math.min(maxPad, Math.trunc(limit));
  maxPad = Math.max(maxPad, minPad);
  const gapNeed = Math.max(1, pyRoundInt(gapRatio * spacing));

  let pad = minPad, runEmpty = 0;
  let y = yStart + direction * minPad;
  while (pad < maxPad) {
    y += direction;
    if (y < 0 || y >= height) break;
    let any = false;
    const row = y * width;
    for (let x = x0; x < x1; x++) if (ink[row + x]) { any = true; break; }
    runEmpty = any ? 0 : runEmpty + 1;
    if (runEmpty >= gapNeed) {
      pad -= runEmpty - 1;          /* do not keep the blank paper we walked */
      break;
    }
    pad += 1;
  }
  return Math.max(minPad, Math.min(pad, maxPad));
}

export function staffBoxes(ink, width, height, staves, systems) {
  const boxes = [];
  for (let sysIdx = 0; sysIdx < systems.length; sysIdx++) {
    const staffIds = systems[sysIdx];
    /* Horizontal extent per system, not per page. */
    const top = staves[staffIds[0]].top;
    const bottom = staves[staffIds[staffIds.length - 1]].bottom;
    let x0 = 0, x1 = width, anyCol = false;
    let first = -1, last = -1;
    for (let x = 0; x < width; x++) {
      let colAny = false;
      for (let y = top; y <= bottom; y++) {
        if (ink[y * width + x]) { colAny = true; break; }
      }
      if (colAny) {
        if (first < 0) first = x;
        last = x;
        anyCol = true;
      }
    }
    if (anyCol) { x0 = first; x1 = last + 1; }

    for (let voiceIdx = 0; voiceIdx < staffIds.length; voiceIdx++) {
      const s = staves[staffIds[voiceIdx]];
      const spacing = s.lineSpacing;
      /* Never grow more than halfway towards a neighbouring staff. */
      const upLimit = voiceIdx > 0
        ? (s.top - staves[staffIds[voiceIdx - 1]].bottom) / 2 : null;
      const downLimit = voiceIdx + 1 < staffIds.length
        ? (staves[staffIds[voiceIdx + 1]].top - s.bottom) / 2 : null;

      const padUp = growEdge(ink, width, height, x0, x1, s.top, -1, spacing, upLimit);
      const padDown = growEdge(ink, width, height, x0, x1, s.bottom, +1, spacing, downLimit);

      const y0 = Math.max(0, s.top - padUp);
      const y1 = Math.min(height, s.bottom + padDown);
      boxes.push({
        staff: staffIds[voiceIdx],
        system: sysIdx,
        voice: voiceIdx,
        x: x0, y: y0, w: x1 - x0, h: y1 - y0,
        lineSpacing: spacing,
        padUp, padDown,
      });
    }
  }
  return boxes;
}

/* Full detection pipeline: grayscale page -> staff crop boxes. */
export function detect(gray, width, height, threshold = INK_THRESHOLD) {
  const ink = binarize(gray, width, height, threshold);
  const runs = findStaffLines(ink, width, height);
  const staves = groupLinesIntoStaves(runs);
  const systems = findSystems(ink, width, staves);
  return {
    lineRuns: runs.length,
    staves,
    systems,
    boxes: staffBoxes(ink, width, height, staves, systems),
  };
}

/* Bilinear resample of a float32 (inH, inW) image to (outH, outW).
 * The sampling grid is computed in float32 exactly as numpy does -- fround on
 * every step -- so floor() lands on the same source pixel as the reference. */
export function resizeBilinear(img, inH, inW, outH, outW) {
  if (inH === outH && inW === outW) return Float32Array.from(img);
  const out = new Float32Array(outH * outW);

  const scaleY = fr(inH / outH);
  const scaleX = fr(inW / outW);
  const y0 = new Int32Array(outH), y1 = new Int32Array(outH);
  const wy = new Float64Array(outH);
  for (let i = 0; i < outH; i++) {
    let v = fr(fr(fr(fr(i) + 0.5) * scaleY) - 0.5);
    if (v < 0) v = 0;
    if (v > inH - 1) v = fr(inH - 1);
    const f = Math.floor(v);
    y0[i] = f;
    y1[i] = Math.min(f + 1, inH - 1);
    wy[i] = fr(v - f);
  }
  const x0 = new Int32Array(outW), x1 = new Int32Array(outW);
  const wx = new Float64Array(outW);
  for (let j = 0; j < outW; j++) {
    let v = fr(fr(fr(fr(j) + 0.5) * scaleX) - 0.5);
    if (v < 0) v = 0;
    if (v > inW - 1) v = fr(inW - 1);
    const f = Math.floor(v);
    x0[j] = f;
    x1[j] = Math.min(f + 1, inW - 1);
    wx[j] = fr(v - f);
  }

  for (let i = 0; i < outH; i++) {
    const rTop = y0[i] * inW, rBot = y1[i] * inW, w = wy[i], o = i * outW;
    for (let j = 0; j < outW; j++) {
      const top = img[rTop + x0[j]] * (1 - wx[j]) + img[rTop + x1[j]] * wx[j];
      const bot = img[rBot + x0[j]] * (1 - wx[j]) + img[rBot + x1[j]] * wx[j];
      out[o + j] = top * (1 - w) + bot * w;
    }
  }
  return out;
}

/* Scale a staff crop to the canonical spacing, pad to fixed height.
 * `staffOffset` is the distance from the crop top to the topmost staff line
 * (pass the box's padUp), so the five lines land at the same place for every
 * sample. Returns { data: Float32Array, width, height }, white = 1. */
export function normalizeStaff(crop, cropWidth, cropHeight, lineSpacing,
                               staffOffset = null,
                               targetSpacing = TARGET_SPACING,
                               targetHeight = TARGET_HEIGHT) {
  const scale = targetSpacing / lineSpacing;
  const outH = Math.max(1, pyRoundInt(cropHeight * scale));
  const outW = Math.max(1, pyRoundInt(cropWidth * scale));
  const scaled = resizeBilinear(crop, cropHeight, cropWidth, outH, outW);

  const canvas = new Float32Array(targetHeight * outW).fill(1);

  let top;
  if (staffOffset === null) {
    top = Math.floor((targetHeight - outH) / 2);
  } else {
    /* Staff at a fixed position: four spacings of staff, equal air around. */
    const staffTopScaled = staffOffset * scale;
    const want = (targetHeight - 4 * targetSpacing) / 2.0;
    top = pyRoundInt(want - staffTopScaled);
  }

  const srcY0 = Math.max(0, -top);
  const srcY1 = Math.min(outH, targetHeight - top);
  const dstY0 = Math.max(0, top);
  for (let y = srcY0; y < srcY1; y++) {
    const src = y * outW, dst = (dstY0 + (y - srcY0)) * outW;
    for (let x = 0; x < outW; x++) canvas[dst + x] = scaled[src + x];
  }
  return { data: canvas, width: outW, height: targetHeight };
}

/* Full page pipeline, mirroring the reference run (scripts/42_emit_ir.py):
 * detect -> crop -> normalize (staffOffset = padUp) -> truncate at maxWidth
 * -> invert (ink = 1). Returns { boxes, inputs }; each input is
 * { box, data, width, height, truncated } where data is the float32 tensor
 * the ONNX model expects as (1, 1, height, width). */
export function preprocessPage(gray, width, height, opts = {}) {
  const targetSpacing = opts.targetSpacing ?? TARGET_SPACING;
  const targetHeight = opts.targetHeight ?? TARGET_HEIGHT;
  const maxWidth = opts.maxWidth ??
    Math.round(MAX_WIDTH * targetSpacing / 10.0);
  const threshold = opts.threshold ?? INK_THRESHOLD;

  const det = detect(gray, width, height, threshold);
  const inputs = [];
  for (const box of det.boxes) {
    const crop = new Float32Array(box.w * box.h);
    for (let y = 0; y < box.h; y++) {
      const src = (box.y + y) * width + box.x, dst = y * box.w;
      for (let x = 0; x < box.w; x++) crop[dst + x] = gray[src + x];
    }
    const norm = normalizeStaff(crop, box.w, box.h, box.lineSpacing,
                                box.padUp, targetSpacing, targetHeight);
    let w = norm.width, truncated = false;
    if (w > maxWidth) { w = maxWidth; truncated = true; }
    const data = new Float32Array(norm.height * w);
    for (let y = 0; y < norm.height; y++) {
      const src = y * norm.width, dst = y * w;
      for (let x = 0; x < w; x++) data[dst + x] = 1.0 - norm.data[src + x];
    }
    inputs.push({ box, data, width: w, height: norm.height, truncated });
  }
  return { boxes: det.boxes, inputs };
}
