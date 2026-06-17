'use strict'

// Single source of truth for the mobile perf benchmark matrix. The per-shard
// test files (benchmark-perf-<model>-<quant>.test.js) and the Benchmark
// Performance workflow's test_groups override are both generated from this list
// by scripts/generate-benchmark-shards.js, so the (model x quant) matrix is
// defined in exactly one place. Underscore prefix keeps it out of the test
// globs (it is not a *.test.js file).
//
// Embeddings are a single prefill-only forward pass, so one shard per
// (model x quant) cell sweeps the rest of the axes (device x batchSize x
// flashAttn x inputMode) INTERNALLY on the device. The cells come from the
// benchmark model manifest, not a hardcoded list, so the shards always track
// what the manifest ships.

// The (model, quant) cells are read from the benchmark performance manifest
// (the same file the desktop sweep and the renderer's coverage read), so the
// mobile shards never drift from the models the addon actually benchmarks. A
// relative JSON require resolves under both Node (generator + renderer) and the
// Bare shard runner, avoiding a runtime-specific path module.
const manifest = require('../../benchmarks/performance/models.manifest.json')

// Flatten the manifest into one cell per (modelId, quant), preserving manifest
// order so the shard list and workflow groups are stable.
function matrix () {
  const out = []
  for (const model of (manifest.models || [])) {
    const quants = model.gguf && Array.isArray(model.gguf.quantizations) ? model.gguf.quantizations : []
    for (const quant of quants) {
      out.push({ model: model.id, quant, repo: model.gguf.repo, revision: model.gguf.revision })
    }
  }
  return out
}

// Filename slug: lowercase, drop dots, underscores -> dashes.
// 'embeddingGemma' -> 'embeddinggemma', 'Qwen3-embedding-0.6B' ->
// 'qwen3-embedding-06b', 'Q4_K_M' -> 'q4-k-m', 'F16' -> 'f16'.
function slug (value) {
  return String(value).toLowerCase().replace(/\./g, '').replace(/_/g, '-')
}

function shardFileName (cell) {
  return `benchmark-perf-${slug(cell.model)}-${slug(cell.quant)}.test.js`
}

// Manifest model id for a cell. Single source for the id used by the on-device
// benchmark (modelSpec) and the report renderer's coverage check, so both agree
// on shard identity.
function modelId (cell) {
  return cell.model
}

// Stable per-shard key matching the renderer's "[<modelId> q=<quant>] ..." row
// label, so coverage can be reconciled against the matrix.
function mobileShardKey (cell) {
  return `${modelId(cell)}|${cell.quant}`
}

// Mirrors toFunctionName in scripts/generate-mobile-integration-tests.js:
// split the base name on non-alphanumerics, capitalize each part, prefix run.
// The shard file ends in `.test.js`, so the `test` token survives and the name
// carries a `Test` suffix, exactly matching the generated mobile runner.
function runFunctionName (cell) {
  const base = shardFileName(cell).replace(/\.js$/, '')
  const parts = base.split(/[^a-zA-Z0-9]+/).filter(Boolean)
  const suffix = parts.map((p) => p.charAt(0).toUpperCase() + p.slice(1)).join('')
  return `run${suffix}`
}

// The exact lines + trailing newline each generated shard file holds.
function shardContents (cell) {
  return [
    "'use strict'",
    "const { benchmarkModel } = require('./_benchmark-perf.js')",
    `benchmarkModel('${cell.model}', '${cell.quant}')`,
    ''
  ].join('\n')
}

// One workflow matrix entry per model, each carrying that model's quant groups
// in manifest order, matching the mobile-benchmark job's test_groups. Grouping
// by model keeps each Device Farm batch to a single set of weights so its
// quants download once.
function workflowBatches () {
  const byModel = new Map()
  for (const cell of matrix()) {
    if (!byModel.has(cell.model)) byModel.set(cell.model, [])
    const grep = runFunctionName(cell)
    byModel.get(cell.model).push({ name: grep.slice(3).replace(/Test$/, ''), grep })
  }
  return [...byModel.entries()].map(([model, groups]) => ({ model: slug(model), groups }))
}

module.exports = {
  matrix,
  slug,
  shardFileName,
  modelId,
  mobileShardKey,
  runFunctionName,
  shardContents,
  workflowBatches
}
