// smoke test for the js modules: they must import under plain node, without a
// dom and without any dependency, and they must hold no state between calls.
// run: node test/smoke.mjs
import assert from 'node:assert/strict'
import * as decode from '../js/omr-decode.js'
import * as pre from '../js/omr-preprocess.js'
import * as reject from '../js/omr-reject.js'

// vocabulary mapping: class 0 is the ctc blank, class i>0 is token i-1
const i2w = decode.i2wFromVocab(['4c', '4d'])
assert.equal(i2w[0], undefined)
assert.equal(i2w[1], '4c')
assert.equal(i2w[2], '4d')

// greedy decode: repeats collapse, blanks drop
const blank = 0
const frames = [blank, 1, 1, blank, 2, 2, blank]
const logits = new Float32Array(frames.length * 3)
frames.forEach((cls, t) => { logits[t * 3 + cls] = 1 })
// each entry carries the token, its frame range and a confidence, so a caller
// can map a symbol back to a pixel range in the original image
const out = decode.decodeLine(logits, frames.length, 3, i2w)
assert.deepEqual(out.map(o => o.token), ['4c', '4d'])
assert.equal(out[0].f0, 1)
assert.equal(out[0].f1, 2)
assert.ok(out[0].confidence > 0 && out[0].confidence <= 1)

// a blank page has no staves and is rejected rather than read
const white = new Uint8Array(200 * 200).fill(255)
const stats = reject.pageStats(white, 200, 200)
assert.equal(reject.modalStaves([]), 0)
assert.ok(stats)

// preprocessing constants are the contract with the model
assert.equal(pre.TARGET_HEIGHT, 128)

console.log('ok')
