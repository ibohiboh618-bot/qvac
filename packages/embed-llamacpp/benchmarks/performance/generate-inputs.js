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

// This filler tokenizes at ~4.2 chars/token against the embedding tokenizer. Each
// sequence must stay strictly UNDER its batch size — the addon's
// validateBatchLimitsOrThrow rejects an input line whose token count (including
// the tokenizer's BOS/EOS) reaches the batch size. So generate to a target a
// margin below the batch size, with a slightly conservative chars/token, so no
// sequence can overflow.
const CHARS_PER_TOKEN = 4.1
const TOKEN_SAFETY_MARGIN = 16
const HEAD = 'Some input. '
const UNIT = 'Some more input. '

function fillerForBatch (batchSize) {
  const targetTokens = Math.max(1, batchSize - TOKEN_SAFETY_MARGIN)
  const targetChars = Math.round(targetTokens * CHARS_PER_TOKEN)
  let s = HEAD
  while (s.length < targetChars) s += UNIT
  return s.slice(0, targetChars)
}

function main () {
  const out = {}
  for (const batchSize of PARAMETER_SWEEP.batchSize) {
    const sentence = fillerForBatch(batchSize)
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
