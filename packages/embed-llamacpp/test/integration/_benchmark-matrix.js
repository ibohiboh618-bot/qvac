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
// flashAttn x inputMode) INTERNALLY on the device.
//
// The cells are hardcoded here rather than read from the manifest because the
// mobile Device Farm bundler only bundles test/integration, so this module
// (reachable from the on-device shards) must not require anything outside it.
// The desktop sweep + renderer read benchmarks/performance/models.manifest.json
// directly; these cells mirror it. scripts/generate-benchmark-shards.js --check
// asserts they stay in sync with the manifest (read under Node), so they cannot
// silently drift.
// `file` is the exact GGUF filename in the HF repo. Repos are inconsistent about
// case per quant (embeddinggemma-300M-Q8_0 vs -300m-Q4_0; Qwen ...-f16 lowercase),
// so the filename is pinned here rather than reconstructed: a guessed name 404s,
// and a 404 is not retried. Verified against each repo's HF file listing.
const CELLS = [
  { model: 'embeddingGemma', quant: 'Q8_0', repo: 'unsloth/embeddinggemma-300m-GGUF', revision: 'main', file: 'embeddinggemma-300M-Q8_0.gguf' },
  { model: 'embeddingGemma', quant: 'Q4_0', repo: 'unsloth/embeddinggemma-300m-GGUF', revision: 'main', file: 'embeddinggemma-300m-Q4_0.gguf' },
  { model: 'Qwen3-embedding-0.6B', quant: 'Q8_0', repo: 'Qwen/Qwen3-Embedding-0.6B-GGUF', revision: 'main', file: 'Qwen3-Embedding-0.6B-Q8_0.gguf' },
  { model: 'Qwen3-embedding-0.6B', quant: 'F16', repo: 'Qwen/Qwen3-Embedding-0.6B-GGUF', revision: 'main', file: 'Qwen3-Embedding-0.6B-f16.gguf' },
  { model: 'Qwen3-embedding-4B-gguf', quant: 'Q8_0', repo: 'Qwen/Qwen3-Embedding-4B-GGUF', revision: 'main', file: 'Qwen3-Embedding-4B-Q8_0.gguf' },
  { model: 'Qwen3-embedding-4B-gguf', quant: 'Q4_K_M', repo: 'Qwen/Qwen3-Embedding-4B-GGUF', revision: 'main', file: 'Qwen3-Embedding-4B-Q4_K_M.gguf' },
  { model: 'Qwen3-embedding-4B-gguf', quant: 'F16', repo: 'Qwen/Qwen3-Embedding-4B-GGUF', revision: 'main', file: 'Qwen3-Embedding-4B-f16.gguf' }
]

// Sweep axes + input modes for the mobile sweep. The desktop copy of these
// lives in benchmarks/performance/_sweep-grid.js; the small axis literals are
// duplicated here so this module stays self-contained for the mobile bundler.
const PARAMETER_SWEEP = {
  quantization: ['Q4_0', 'Q4_K_M', 'Q8_0', 'F16'],
  device: ['cpu', 'gpu'],
  batchSize: [256, 512, 1024, 2048],
  flashAttn: ['off', 'on']
}
const INPUT_MODES = ['single', 'array']

// One cell per (modelId, quant), preserving order so the shard list and
// workflow groups are stable.
function matrix () {
  return CELLS.map((cell) => ({ model: cell.model, quant: cell.quant, repo: cell.repo, revision: cell.revision, file: cell.file }))
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
// in matrix order, matching the mobile-benchmark job's test_groups. Grouping
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
  workflowBatches,
  PARAMETER_SWEEP,
  INPUT_MODES
}
