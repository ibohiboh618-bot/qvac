'use strict'

// Generates benchmarks/performance/inputs.json: for each swept batch size, a set
// of synthetic filler sentences each padded to roughly that batch size in tokens
// (this is a SPEED benchmark, so only the token count matters, not the content).
// MAX_SEQUENCES strings are emitted per batch size so the desktop sweep can slice
// the array-mode sequence-count axis (5 / 10 / 20) from a single source. Kept in
// sync with PARAMETER_SWEEP.batchSize in _sweep-grid.js.
//
// Run: node benchmarks/performance/generate-inputs.js   (or npm run generate:inputs)

const fs = require('fs')
const path = require('path')
const { PARAMETER_SWEEP, MAX_ARRAY_SEQUENCES } = require('./_sweep-grid')

// Established for this filler against the embedding tokenizer: the existing
// inputs.json sized 256/512/1024/2048-token sequences at 1081/2165/4331/8663
// chars, i.e. ~4.23 chars per token. Padding to that ratio reproduces the same
// per-batch token lengths and extrapolates 4096 cleanly.
const CHARS_PER_TOKEN = 4.23
const HEAD = 'Some input. '
const UNIT = 'Some more input. '

function fillerForTokens (tokens) {
  const targetChars = Math.round(tokens * CHARS_PER_TOKEN)
  let s = HEAD
  while (s.length < targetChars) s += UNIT
  return s.slice(0, targetChars)
}

function main () {
  const out = {}
  for (const batchSize of PARAMETER_SWEEP.batchSize) {
    const sentence = fillerForTokens(batchSize)
    out[String(batchSize)] = Array.from({ length: MAX_ARRAY_SEQUENCES }, () => sentence)
  }
  const dest = path.resolve(__dirname, 'inputs.json')
  fs.writeFileSync(dest, JSON.stringify(out, null, 2) + '\n')
  const summary = Object.keys(out)
    .map((bs) => `bs=${bs}: ${out[bs].length} x ${out[bs][0].length} chars`)
    .join(', ')
  console.log(`Wrote ${dest}\n  ${summary}`)
}

main()
