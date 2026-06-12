'use strict'
// QVAC-19371 (A1 scaffold): desktop run driver — the future home of the
// process-level orchestration the in-process harness cannot do:
//   • swap addon prebuild dirs between candidate/baseline blocks   TODO(A2)
//   • schedule interleaved warmup/measured blocks + stability guard TODO(A3)
//   • sample peak RSS around the bare process (macOS/Windows)       TODO(A4)
//
// NOT WIRED YET: the workflow's "Run VLM matrix" step still runs the harness
// directly (npx brittle … && bare …). A2/A3 replace that step body with
//   node benchmarks/vlm-benchmark/run-desktop.cjs
// keeping all future run-logic changes inside this folder (no YAML churn).
//
// OWNERSHIP: runner workstream (Dev A).
//
// --selfcheck validates this folder's contract wiring without running any
// model: config loads, scenarios resolve, the models grammar parses, and the
// sample markers file matches the v2 schema. CI and devs run it cheaply.

const fs = require('fs')
const path = require('path')

function selfcheck () {
  const config = require('./config.cjs')
  const { parseModels } = require('./models.cjs')
  const { parseSources } = require('./sources.cjs')
  const { planBlocks } = require('./methodology.cjs')
  const problems = []

  // config/scenario invariants
  if (!config.scenarios || !config.scenarios[config.defaultScenario]) problems.push('defaultScenario missing from scenarios')
  for (const name of config.defaultModels || []) {
    if (!config.catalog || !config.catalog[name]) problems.push(`defaultModels entry '${name}' not in catalog`)
  }

  // grammar round-trips
  parseModels('qwen3.5-f16,qwen3.5-q8', config.catalog, config.models)
  parseModels('t=https://huggingface.co/org/repo/resolve/0123456789012345678901234567890123456789/m.gguf|https://example.com/p.gguf@ctx=8192', config.catalog, [])
  parseSources('addon,addon@candidate,fabric@v8189.0.2,upstream@b8189')
  if (planBlocks(['a', 'b'], config.methodology).length < 2) problems.push('planBlocks produced no plan')

  // sample markers conform to the v2 contract
  const sample = fs.readFileSync(path.join(__dirname, 'markers-v2.sample.log'), 'utf8')
  const need = ['v', 'scenario', 'source_id', 'source_ref', 'block', 'cell', 'model', 'device', 'task', 'metric']
  let rows = 0
  for (const line of sample.split(/\r?\n/)) {
    const m = line.match(/\[VLMROW\](.*?)\[\/VLMROW\]/)
    if (!m) continue
    rows++
    const r = JSON.parse(m[1])
    if (r.v === undefined) continue // a deliberate legacy (v1) row in the sample
    for (const k of need) if (!(k in r)) problems.push(`sample row ${rows} missing '${k}'`)
  }
  if (rows < 10) problems.push(`sample markers file has only ${rows} rows`)

  if (problems.length) {
    console.error('selfcheck FAILED:\n  - ' + problems.join('\n  - '))
    process.exit(1)
  }
  console.log(`selfcheck OK (${rows} sample rows, ${Object.keys(config.catalog).length} catalog models, ${Object.keys(config.scenarios).length} scenarios)`)
}

if (require.main === module) {
  if (process.argv.includes('--selfcheck')) selfcheck()
  else {
    console.error('run-desktop.cjs: block-scheduling driver not implemented yet (A2/A3); only --selfcheck is available')
    process.exit(2)
  }
}

module.exports = { selfcheck }
