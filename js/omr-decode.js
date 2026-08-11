/* omr-decode.js -- CTC logits -> Score-IR (contract 3 + 5).
 *
 * Line-for-line port of src/omr/ir.py in the model repo; the Python file is
 * the reference, this file must stay token-identical to it in behaviour
 * (verified by scripts/44_decode_parity.mjs against dist/fixtures/).
 *
 * Plain ES module: no imports, no DOM, no globals, runs under Node and in a
 * Web Worker. All inputs are typed arrays / plain objects.
 *
 * API (two layers, contract 5):
 *   decodeLine(logits, T, C, i2w)         -> token events with spans + confs
 *   lineFragment(events, staff)           -> IR elements for one line
 *   assembleIR(lines, pages, generator, rejection)
 *                                         -> the IR object for one score
 *   i2wFromVocab(vocabTokens)             -> index map (class 0 = CTC blank)
 *
 * `rejection` is the verdict of omr-reject.js::decide(), passed in because it
 * needs the grey page, which this layer never sees. It stays the only source
 * of `rejected` -- please do not compute a second opinion in the frontend.
 * Omitting it keeps the v1.1 behaviour (`rejected: false`).
 */

/* 1.2: `rejected` gets a real producer (omr-reject.js), `scan-suspected`
 * changed from a rejection reason to a warning, warnings carry `page`, and
 * `line-truncated` moved here from the site repo's worker. Additive fields,
 * but the MEANING of scan-suspected moved -- hence the bump. */
export const IR_VERSION = "1.2";
export const WIDTH_REDUCTION = 4;
export const DIV_WHOLE = 192;

const BLANK = 0;
const SEP = "<b>";

const DURATIONS = new Set(["0", "1", "2", "3", "4", "6", "8", "12", "16", "24", "32"]);
const ACCIDENTALS = { "#": 1, "##": 2, "-": -1, "--": -2, "n": 0 };

const BARLINE_TYPES = {
  "=": "regular", "==": "final", "=||": "double", "=-": "invisible",
  "=:|!": "repeat-end", "=!|:": "repeat-start", "=:|!|:": "repeat-both",
  "==:|!": "final-repeat-end",
};

/* v1.1 null-bar collapse. The net double-emits at printed double bars
 * (`=||` AND `=` for one object -> zero-duration measure, +1 bar count).
 * Adjacent barline records merge -- EXCEPT the two pairings the ground
 * truth itself contains (empty printed measures): `=`+`=` and `=`+`==`. */
const LEGITIMATE_ADJACENT = new Set(["= =", "= =="]);

/* Python round(): decimal round-half-even. Math.round / toFixed both round
 * ties differently; parity against the Python-emitted golden IR needs this. */
function roundPy(x, digits) {
  const p = Math.pow(10, digits);
  const y = x * p;
  const f = Math.floor(y);
  const diff = y - f;
  let r;
  if (diff > 0.5) r = f + 1;
  else if (diff < 0.5) r = f;
  else r = f % 2 === 0 ? f : f + 1;
  return r / p;
}

export function i2wFromVocab(vocabTokens) {
  const i2w = {};
  for (let i = 0; i < vocabTokens.length; i++) i2w[i + 1] = vocabTokens[i];
  return i2w;
}

export function decodeLine(logits, T, C, i2w) {
  const events = [];
  let prev = -1;
  for (let f = 0; f < T; f++) {
    const row = f * C;
    let k = 0, best = logits[row];
    for (let c = 1; c < C; c++) {
      if (logits[row + c] > best) { best = logits[row + c]; k = c; }
    }
    if (k !== BLANK) {
      let sum = 0;
      for (let c = 0; c < C; c++) sum += Math.exp(logits[row + c] - best);
      const p = 1 / sum;                       // softmax of the argmax class
      if (k !== prev) {
        events.push({ token: i2w[k], f0: f, f1: f, _psum: p, _n: 1 });
      } else {
        const ev = events[events.length - 1];
        ev.f1 = f; ev._psum += p; ev._n += 1;
      }
    }
    prev = k;
  }
  for (const ev of events) {
    ev.confidence = roundPy(ev._psum / ev._n, 4);
    delete ev._psum; delete ev._n;
  }
  return events;
}

function records(tokenEvents) {
  const recs = []; let cur = [];
  for (const ev of tokenEvents) {
    if (ev.token === SEP) { if (cur.length) recs.push(cur); cur = []; }
    else cur.push(ev);
  }
  if (cur.length) recs.push(cur);
  return recs;
}

/* Two adjacent barline records -> one. The more specific token survives
 * (if exactly one of the pair is a plain `=`, the other one); the span
 * widens to the union, because both emissions point at the same ink and
 * the bbox should cover it. Confidence stays the survivor's. */
function mergeBarlines(a, b) {
  const keep = (a[0].token === "=" && b[0].token !== "=") ? b : a;
  const f0 = Math.min(a[0].f0, b[0].f0);
  const f1 = Math.max(a[a.length - 1].f1, b[b.length - 1].f1);
  const first = { ...keep[0], f0, f1 };
  return [first, ...keep.slice(1)];
}

/* v1.1: merge adjacent barline records unless the pairing is legitimate
 * ground truth (empty printed measures, LEGITIMATE_ADJACENT). Runs while
 * appending, so a chain of three emissions merges into one. */
function collapseBarlines(recs) {
  const out = [];
  for (const rec of recs) {
    out.push(rec);
    while (out.length >= 2) {
      const a = out[out.length - 2], b = out[out.length - 1];
      if (!(a[0].token.startsWith("=") && b[0].token.startsWith("="))) break;
      if (LEGITIMATE_ADJACENT.has(a[0].token + " " + b[0].token)) break;
      const merged = mergeBarlines(a, b);
      out.length -= 2;
      out.push(merged);
    }
  }
  return out;
}

function dur(token) {
  const base = token.replace(/\.+$/, "");
  if (!DURATIONS.has(base)) return null;
  const dots = token.length - base.length;
  const d = base === "0" ? DIV_WHOLE * 2 : Math.floor(DIV_WHOLE / parseInt(base, 10));
  let total = d, add = d;
  for (let i = 0; i < dots; i++) { add = Math.floor(add / 2); total += add; }
  return { divisions: total, base, dots };
}

function pitch(token) {
  if (!token || !/^[a-gA-G]+$/.test(token)) return null;
  const ch = token[0];
  if (token !== ch.repeat(token.length)) return null;
  if (ch === ch.toLowerCase()) return { step: ch.toUpperCase(), octave: 3 + token.length };
  return { step: ch, octave: 4 - token.length };
}

function classify(record) {
  const toks = record.map((ev) => ev.token);
  let conf = Infinity;
  for (const ev of record) if (ev.confidence < conf) conf = ev.confidence;
  const first = toks[0];

  if (first.startsWith("=")) {
    return { kind: "barline",
             type: BARLINE_TYPES[first] ?? "other", confidence: conf };
  }

  if (first.startsWith("*")) {
    const el = { kind: "attribute", confidence: conf };
    if (first.startsWith("*clef")) {
      let body = first.slice(5);
      const octave = body.includes("v") ? -1 : body.includes("^") ? 1 : 0;
      body = body.replace("v", "").replace("^", "");
      el.clef = { sign: body[0], line: parseInt(body.slice(1) || "0", 10),
                  octaveChange: octave };
    } else if (first.startsWith("*k[")) {
      const inner = first.slice(3, -1);
      const sharps = (inner.match(/#/g) || []).length;
      /* v1.2: `sharps || -flats` yields -0 for C major, which only becomes
       * 0 on a JSON roundtrip. Normalised here so the field is 0 in the
       * object too, not just in the serialisation. */
      el.keyFifths = (sharps || -(inner.match(/-/g) || []).length) + 0;
    } else if (first.startsWith("*M")) {
      const [num, den] = first.slice(2).split("/");
      el.time = { num: parseInt(num, 10), den: parseInt(den, 10) };
    } else {
      return null;
    }
    return el;
  }

  let d = null;
  for (const t of toks) { d = dur(t); if (d) break; }
  if (!d) return { kind: "unparseable", tokens: toks, confidence: conf };
  const el = { kind: toks.includes("r") ? "rest" : "note",
               duration: d, confidence: conf };
  if (el.kind === "note") {
    let p = null;
    for (const t of toks) { p = pitch(t); if (p) break; }
    if (!p) return { kind: "unparseable", tokens: toks, confidence: conf };
    let alter = 0;
    for (const t of toks) {
      if (t in ACCIDENTALS) { alter = ACCIDENTALS[t]; break; }
    }
    el.pitch = { step: p.step, alter, octave: p.octave };
    el.tie = toks.includes("[") ? "start"
      : toks.includes("]") ? "stop"
      : toks.includes("_") ? "continue" : null;
    if (toks.includes("(")) el.slur = "start";
    else if (toks.includes(")")) el.slur = "stop";
    el.fermata = toks.includes(";");
  }
  return el;
}

function bboxOf(record, staff) {
  let f0 = Infinity, f1 = -Infinity;
  for (const ev of record) {
    if (ev.f0 < f0) f0 = ev.f0;
    if (ev.f1 > f1) f1 = ev.f1;
  }
  const scale = staff.lineSpacingPx / staff.normSpacing;
  const x0 = staff.bbox[0] + WIDTH_REDUCTION * f0 * scale;
  const x1 = staff.bbox[0] + WIDTH_REDUCTION * (f1 + 1) * scale;
  return [roundPy(x0, 1), staff.bbox[1], roundPy(x1 - x0, 1), staff.bbox[3]];
}

export function lineFragment(tokenEvents, staff) {
  const out = [];
  for (const record of collapseBarlines(records(tokenEvents))) {
    const el = classify(record);
    if (el === null) continue;
    el.tokens = record.map((ev) => ev.token);
    el.src = { page: staff.page, system: staff.system,
               staff: staff.staffIndex, bbox: bboxOf(record, staff) };
    out.push(el);
  }
  return out;
}

/* Most common value; on a tie the LARGER one wins (v1.2). `modal` decides how
 * many parts exist, and a line with staffIndex >= modal is dropped from the IR
 * entirely -- so the smaller count deletes a voice in silence, while the
 * larger one merely raises staff-count-mismatch on the short systems. A lost
 * voice is the most expensive error this project knows; a warning is the
 * cheapest. */
function modal(values) {
  const uniq = [...new Set(values)].sort((a, b) => b - a);
  let best = 0, n = 0;
  for (const v of uniq) {
    const c = values.filter((x) => x === v).length;
    if (c > n) { best = v; n = c; }
  }
  return best;
}

function systemsOf(lines) {
  const bySys = new Map();
  for (const ln of lines) {
    const st = ln.staff;
    const key = st.page * 1e6 + st.system;
    if (!bySys.has(key)) bySys.set(key, []);
    bySys.get(key).push(st);
  }
  const out = [];
  for (const key of [...bySys.keys()].sort((a, b) => a - b)) {
    const staves = bySys.get(key);
    const xs = staves.map((s) => s.bbox[0]);
    const ys = staves.map((s) => s.bbox[1]);
    const x2 = staves.map((s) => s.bbox[0] + s.bbox[2]);
    const y2 = staves.map((s) => s.bbox[1] + s.bbox[3]);
    out.push({
      index: staves[0].system, page: staves[0].page,
      bbox: [Math.min(...xs), Math.min(...ys),
             Math.max(...x2) - Math.min(...xs),
             Math.max(...y2) - Math.min(...ys)],
      staves: [...staves].sort((a, b) => a.bbox[1] - b.bbox[1]).map((s) => ({
        part: s.staffIndex, bbox: s.bbox, lineSpacingPx: s.lineSpacingPx,
        normScale: roundPy(s.lineSpacingPx / s.normSpacing, 4),
      })),
    });
  }
  return out;
}

export function assembleIR(lines, pages, generator = "omr-decode.js",
                           rejection = null) {
  /* v1.2: keyed by page AND system. System indices restart at 0 on every
   * page, so the v1.1 counting folded system n of page 1 together with system
   * n of page 2 -- invisible on a one-page score, simply wrong on a
   * multi-page one. */
  const counts = new Map();
  for (const ln of lines) {
    const s = ln.staff;
    const key = `${s.page}/${s.system}`;
    counts.set(key, Math.max(counts.get(key) ?? 0, s.staffIndex + 1));
  }
  const m = counts.size ? modal([...counts.values()]) : 0;

  const warnings = [];
  const keys = [...counts.keys()].sort((a, b) => {
    const [pa, sa] = a.split("/").map(Number);
    const [pb, sb] = b.split("/").map(Number);
    return pa - pb || sa - sb;
  });
  for (const key of keys) {
    const [page, sysno] = key.split("/").map(Number);
    const n = counts.get(key);
    if (n !== m) {
      warnings.push({ code: "staff-count-mismatch", page, system: sysno,
                      message: `Seite ${page}, System ${sysno}: ` +
                               `${n} Zeilen erkannt, Struktur sagt ${m}` });
    }
  }

  const parts = [];
  for (let i = 0; i < m; i++) parts.push({ index: i, label: null, measures: [] });
  const openMeasures = new Array(m).fill(null);
  const structure = { stavesPerSystem: m, clefs: new Array(m).fill(null),
                      keyFifths: null, time: null };

  const open = (p, system) => {
    if (openMeasures[p] === null) {
      const mm = { index: parts[p].measures.length, system, events: [] };
      parts[p].measures.push(mm);
      openMeasures[p] = mm;
    }
    return openMeasures[p];
  };

  for (const ln of lines) {
    const st = ln.staff;
    const p = st.staffIndex;
    /* v1.2: the preprocessing reports `truncated` per line (MAX_WIDTH); until
     * now the site repo's worker appended this code after the fact. The
     * namespace keeps one source, so it is emitted here. */
    if (st.truncated) {
      warnings.push({ code: "line-truncated", page: st.page,
                      system: st.system, staff: p,
                      message: `Seite ${st.page}, System ${st.system}, `
                        + `Zeile ${p}: rechts abgeschnitten (MAX_WIDTH) -- `
                        + `was dahinter steht, wurde nicht gelesen` });
    }
    if (p >= m) continue;
    for (const el of ln.elements) {
      if (el.kind === "unparseable") {
        // RAW on purpose: tokens joined by single spaces, no prose frame,
        // unlike every other warning here. Your insert suggestion parses this
        // with message.split(" "); a prose sentence would still split, just
        // into the wrong words. Silent break, no exception. v1.3 ships
        // `tokens` additively and `message` stays exactly as it is, because
        // it is the fallback for older imported score JSONs.
        // See entscheid-v13-zuschnitt-2026-08-04.md §2.
        warnings.push({ code: "unparseable-tokens", page: st.page,
                        system: st.system, staff: p,
                        message: el.tokens.join(" ") });
        continue;
      }
      if (el.kind === "attribute") {
        const mm = open(p, st.system);
        if (!("attributes" in mm)) mm.attributes = {};
        for (const k of ["clef", "keyFifths", "time"]) {
          if (k in el) mm.attributes[k] = el[k];
        }
        if ("clef" in el && structure.clefs[p] === null) {
          const c = el.clef;
          structure.clefs[p] = c.sign + String(c.line) +
            (c.octaveChange < 0 ? "v8" : "");
        }
        if ("keyFifths" in el && structure.keyFifths === null) {
          structure.keyFifths = el.keyFifths;
        }
        if ("time" in el && structure.time === null) structure.time = el.time;
        continue;
      }
      if (el.kind === "barline") {
        const mm = openMeasures[p];
        if (mm !== null) {
          mm.barline = { type: el.type, confidence: el.confidence,
                         tokens: el.tokens, src: el.src };
          openMeasures[p] = null;
        }
        continue;
      }
      open(p, st.system).events.push(el);
    }
  }

  /* The rejection reasons come first: they explain the page as a whole,
   * while the decode warnings explain single spots on it. */
  const allWarnings = rejection ? [...(rejection.warnings ?? []), ...warnings]
                                : warnings;

  return {
    irVersion: IR_VERSION,
    generator: { model: "omr-2026-08", decoder: generator },
    rejected: rejection ? Boolean(rejection.rejected) : false,
    rejectionReason: rejection ? rejection.reason : null,
    source: { pages },
    structure: { recognized: structure, confirmed: null },
    systems: systemsOf(lines),
    parts,
    warnings: allWarnings,
  };
}
