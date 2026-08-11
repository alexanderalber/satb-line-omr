/* omr-reject.js -- is this page a scan, or closed score?
 *
 * Port of src/omr/reject.py in the model repo; the Python file is the
 * reference, this file must reproduce its verdict (verified by
 * scripts/54_reject_parity.mjs against work/53_ablehnung_gegenprobe.json:
 * verdict and reason exact over 199 real scores, statistics within 1e-9).
 *
 * One field, one source (entscheid-schema-freeze-v12-2026-08-04.md §2.1):
 * this module is the only producer of `rejected`. Please do not add a second
 * heuristic in the frontend -- if this module is wrong, it gets fixed and
 * remeasured here, where the choir's archive is.
 *
 * Plain ES module: no imports, no DOM, no globals, no state between calls.
 *
 * Float semantics: the ink comparison is `< 0.6` against float32 gray values,
 * identical to omr-preprocess.js::binarize -- no float32 lies between f64(0.6)
 * and f32(0.6), so the comparison matches the numpy reference bit for bit.
 * whiteFrac and runFrac are integer counts divided in doubles, so they are
 * exact on both sides.
 *
 * API:
 *   pageStats(gray, width, height)      -> { whiteFrac, runFrac }
 *   decide(statsPerPage, staffCounts)
 *       -> { rejected, reason, scanSuspected, detail, warnings }
 *          rejected      : stop, the structure is wrong (closed score, no staves)
 *          scanSuspected : keep going, but say so -- see decide()
 *
 * Usage: one pageStats() per rendered page, then one decide() per document,
 * with staffCounts = the number of staves of every system found on the pages
 * (detect().systems.map(s => s.length) from omr-preprocess.js), flattened.
 */

/* Fitted on the training half of the choir's archive and reported on the
 * holdout half. Moving these without a new measurement turns a measured rule
 * into a guessed one. */
export const SCAN_WHITE_MAX = 0.875318;
export const SCAN_RUN_MAX = 0.385546;
export const WHITE_LEVEL = 0.99;
export const INK_THRESHOLD = 0.6;
export const RUN_ROWS = 20;
export const RUN_MIN_INK = 0.30;
export const CLOSED_SCORE_STAVES = 2;

export const REASON_SCAN = 'scan-suspected';
export const REASON_CLOSED_SCORE = 'closed-score-suspected';
export const REASON_NO_STAVES = 'no-staves-found';

/* gray: Float32Array (height*width, row-major), values in [0,1], white = 1. */
export function pageStats(gray, width, height) {
  let white = 0;
  const inkPerRow = new Int32Array(height);
  const bestRun = new Int32Array(height);

  for (let y = 0; y < height; y++) {
    const row = y * width;
    let n = 0, run = 0, best = 0;
    for (let x = 0; x < width; x++) {
      const g = gray[row + x];
      if (g >= WHITE_LEVEL) white++;
      if (g < INK_THRESHOLD) {
        n++;
        run++;
        if (run > best) best = run;
      } else {
        run = 0;
      }
    }
    inkPerRow[y] = n;
    bestRun[y] = best;
  }

  /* The rows with the most ink are the staff lines. Ties keep the smaller row
   * index, matching numpy's stable argsort on the candidate list. */
  const minInk = RUN_MIN_INK * width;
  const cand = [];
  for (let y = 0; y < height; y++) if (inkPerRow[y] >= minInk) cand.push(y);
  cand.sort((a, b) => (inkPerRow[b] - inkPerRow[a]) || (a - b));
  const rows = cand.slice(0, RUN_ROWS);

  let runFrac = 0;
  if (rows.length) {
    let sum = 0;
    for (const y of rows) sum += bestRun[y];
    runFrac = sum / rows.length / width;
  }
  return { whiteFrac: white / (width * height), runFrac };
}

/* Median with numpy's tie handling: mean of the two middle values on even
 * counts. One dark title page must not decide a score. */
export function median(values) {
  if (!values.length) return 0;
  const s = values.slice().sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

export function scanSuspected(statsPerPage) {
  const w = median(statsPerPage.map((s) => s.whiteFrac));
  const r = median(statsPerPage.map((s) => s.runFrac));
  return { isScan: w <= SCAN_WHITE_MAX || r <= SCAN_RUN_MAX,
           medians: { whiteFrac: w, runFrac: r } };
}

/* Most common staves-per-system; ties go to the smaller count, so a tie can
 * only ever make the closed-score rule more careful, never less. */
export function modalStaves(counts) {
  if (!counts.length) return 0;
  let best = 0, n = -1;
  for (const v of Array.from(new Set(counts)).sort((a, b) => a - b)) {
    const c = counts.filter((x) => x === v).length;
    if (c > n) { best = v; n = c; }
  }
  return best;
}

/* Owner decision of 2026-08-04: a suspected scan is a WARNING, not a
 * rejection. The page says "no scans" and the users are adults -- they get
 * told and decide. Closed score stays a rejection: there the output is not
 * merely poor, it is structurally wrong, and no editing repairs it.
 *
 * The closed-score question is not asked on a suspected scan. This one line is
 * the single most valuable thing in this module, and the number says so: 24 of
 * the 80 warned scans have a modal staff count of exactly 2. Without the
 * guard, 30 % of every scan the tool recognises would be rejected as closed
 * score -- a rejection whose stated reason is false, and the reason is what
 * the user is shown. On a scan the detector returns systems that mean nothing;
 * two staves per system is what noise looks like, not what the page says. */
export function decide(statsPerPage, staffCounts) {
  const warnings = [];
  const { isScan, medians } = scanSuspected(statsPerPage);
  const r4 = (x) => Math.round(x * 1e4) / 1e4;

  if (isScan) {
    warnings.push({ code: REASON_SCAN,
      message: `Die Seite sieht nach einem Scan aus (Weissanteil `
        + `${r4(medians.whiteFrac)}, Notenlinienlauf ${r4(medians.runFrac)}). `
        + `Dafuer ist das Werkzeug nicht gebaut -- die Erkennung laeuft `
        + `trotzdem, das Ergebnis ist vermutlich unbrauchbar.` });
  }

  if (!staffCounts.length) {
    warnings.push({ code: REASON_NO_STAVES,
      message: 'Keine Notensysteme gefunden.' });
    return { rejected: true, reason: REASON_NO_STAVES, scanSuspected: isScan,
             detail: medians, warnings };
  }

  const modal = modalStaves(staffCounts);
  const detail = { ...medians, modalStaves: modal };
  if (modal === CLOSED_SCORE_STAVES && !isScan) {
    warnings.push({ code: REASON_CLOSED_SCORE,
      message: 'Zwei Systeme je Akkolade: zwei Stimmen teilen sich eine '
        + 'Zeile (closed score). Die Zuordnung Note -> Stimme ist so nicht '
        + 'moeglich.' });
    return { rejected: true, reason: REASON_CLOSED_SCORE,
             scanSuspected: isScan, detail, warnings };
  }

  return { rejected: false, reason: null, scanSuspected: isScan, detail,
           warnings };
}
