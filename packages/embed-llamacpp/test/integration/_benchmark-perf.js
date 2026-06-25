'use strict'

// Shared runner for the mobile embed perf benchmark. Sharded into one test file
// per (model x quant x batchSize x flashAttn)
// (benchmark-perf-<model>-<quant>-bs<N>-fa<on|off>.test.js); this module holds
// the logic they all share. Underscore prefix keeps it out of the mobile test
// generator (it is not a *.test.js file).
//
// batchSize and flashAttn are the reload-heavy axes (each needs a fresh
// GGMLBert()+load()), so they are the shard key: one (batchSize, flashAttn) per
// shard. Each shard sweeps only device(cpu,gpu) x inputMode(single,array)
// INTERNALLY — device requires a fresh load (2 loads/session), inputMode is a
// runtime-only re-run of the same loaded model (no reload). Embedding is a
// single prefill-only forward pass, so each config records prefill throughput
// (ppTPS), prefill latency (ms), embeddings/sec, and cosine similarity vs an
// in-run baseline (the first successful config for the same input mode). The
// axes and input modes come from the benchmark sweep grid, so the mobile sweep
// never drifts from the desktop one.

const fs = require('bare-fs')
const path = require('bare-path')
const os = require('bare-os')
const GGMLBert = require('../../index.js')
const { safeTest, downloadFile } = require('./utils')
const { attachSpecLogger } = require('./spec-logger')
const { recordPerformance, isMobile } = require('./_perf-helper.js')
const { matrix, PARAMETER_SWEEP, INPUT_MODES } = require('./_benchmark-matrix.js')

// Inlined from benchmarks/performance/math.js + case-runner.js so this runner
// stays self-contained for the mobile Device Farm bundler (which only bundles
// test/integration). Kept byte-identical in behavior to the desktop copies.
function cosineSimilarity (a, b) {
  if (a.length !== b.length) {
    throw new Error(`Vector length mismatch: ${a.length} vs ${b.length}`)
  }
  let dotProduct = 0
  let normA = 0
  let normB = 0
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i]
    normA += a[i] * a[i]
    normB += b[i] * b[i]
  }
  const denominator = Math.sqrt(normA) * Math.sqrt(normB)
  if (denominator < 1e-12) return 1.0 * Math.sign(dotProduct)
  return dotProduct / denominator
}

function average (values) {
  if (!values.length) return null
  let sum = 0
  for (const value of values) sum += value
  return sum / values.length
}

function stddev (values) {
  if (!values.length) return null
  if (values.length === 1) return 0
  const avg = average(values)
  let varianceSum = 0
  for (const value of values) {
    const diff = value - avg
    varianceSum += diff * diff
  }
  return Math.sqrt(varianceSum / values.length)
}

// The addon's prefill timer (t_p_eval_ms) has ~millisecond resolution. A single
// short input prefills faster than it can measure, so the addon reports a
// sub-millisecond time and a tokens_per_second inflated to ~1e8. Treat prefill
// timing below this floor as unmeasured so ppTPS / latency / embeddings-per-sec
// report null for those configs instead of a fabricated value. Mirrors the
// desktop case-runner.
const MIN_RELIABLE_PREFILL_MS = 1

function reliablePrefillMs (totalTimeMs) {
  return totalTimeMs != null && totalTimeMs >= MIN_RELIABLE_PREFILL_MS ? totalTimeMs : null
}

// Prefill throughput (ppTPS) as measured by the addon; only meaningful when the
// prefill time is reliable, which the caller enforces.
function prefillTokensPerSecond (runtimeStats) {
  return runtimeStats.tokens_per_second != null ? runtimeStats.tokens_per_second : null
}

function _envInt (key, fallback) {
  let raw = ''
  if (typeof os.getEnv === 'function') raw = os.getEnv(key) || ''
  if (!raw && typeof process !== 'undefined' && process.env) raw = process.env[key] || ''
  const v = parseInt(raw, 10)
  return Number.isFinite(v) && v > 0 ? v : fallback
}

// Measured repetitions per config, reported as mean +/- stddev (matching the
// desktop sweep, which repeats 5x). Repeating on-device guards against a single
// shot skewed by mobile thermal throttling. Overridable via QVAC_PERF_RUNS /
// QVAC_PERF_WARMUP_RUNS, which the Benchmark Performance workflow pushes to the
// device (qvacPerfConfig.txt -> os.setEnv in integration-runtime.cjs), matching
// the LLM benchmark runner.
const PERF_RUNS = _envInt('QVAC_PERF_RUNS', 3)
const PERF_WARMUP_RUNS = _envInt('QVAC_PERF_WARMUP_RUNS', 1)

function meanOf (values) {
  return values.length ? average(values) : null
}

function stdOf (values) {
  return values.length > 1 ? stddev(values) : null
}

const platform = os.platform()
const isDarwinX64 = platform === 'darwin' && os.arch() === 'x64'
const isLinuxArm64 = platform === 'linux' && os.arch() === 'arm64'

// darwin-x64 and linux-arm64 sweep CPU only, matching the integration suite's
// device list; everything else sweeps both.
const DEVICES = (isDarwinX64 || isLinuxArm64) ? ['cpu'] : PARAMETER_SWEEP.device

// A small, fixed sentence set, deliberately NOT the 81KB desktop inputs.json:
// mobile memory is tight and the perf signal does not need hundreds of
// sequences. `single` embeds the first sentence; `array` embeds all of them.
const SENTENCES = [
  'That is a happy person enjoying a sunny afternoon in the park.',
  'The quarterly report shows steady growth across every region.',
  'Quantization reduces model memory footprint at a small accuracy cost.',
  'She carefully folded the letter and placed it inside the drawer.',
  'On-device inference keeps private data from ever leaving the phone.'
]

function inputsFor (inputMode) {
  return inputMode === 'single' ? SENTENCES[0] : SENTENCES
}

function modelSpec (modelName, quant) {
  const cell = matrix().find((c) => c.model === modelName && c.quant === quant)
  if (!cell) throw new Error(`No benchmark matrix cell for model "${modelName}" quant "${quant}"`)
  // One exact URL per cell (cell.file is the pinned HF filename). A wrong guess
  // would 404, and downloadFile does not retry a 404, so do not guess.
  const url = `https://huggingface.co/${cell.repo}/resolve/${cell.revision}/${cell.file}`
  return { id: cell.model, quant, name: cell.file, urls: [url] }
}

// Mirrors test/integration/utils.js ensureModel, but takes an ordered URL list
// (HF filename case varies per repo) and downloads into the shared model cache.
// The mobile framework patches utils.js's model dir to global.testDir (the
// app's writable Documents/files dir); __dirname here is the read-only bundle,
// so resolve the same writable location the regular tests use instead.
async function ensureBenchmarkModel (spec) {
  const modelDir = path.join(global.testDir || os.tmpdir(), 'test', 'model')
  const modelPath = path.join(modelDir, spec.name)
  if (fs.existsSync(modelPath)) {
    const stat = fs.statSync(modelPath)
    if (stat.size > 0) return modelPath
    fs.unlinkSync(modelPath)
  }
  fs.mkdirSync(modelDir, { recursive: true })
  console.log(`[download] Downloading benchmark model: ${spec.name}...`)
  await downloadFile(spec.urls, modelPath, { minBytes: 1024 })
  const stat = fs.statSync(modelPath)
  console.log(`[download] Model ready: ${(stat.size / 1024 / 1024).toFixed(1)}MB`)
  return modelPath
}

function buildConfig (device, batchSize, flashAttn, modelDir) {
  const config = {
    gpu_layers: device === 'cpu' ? '0' : '999',
    batch_size: String(batchSize),
    flash_attn: flashAttn,
    verbosity: '0',
    openclCacheDir: modelDir
  }
  if (device === 'cpu' || device === 'gpu') config.device = device
  return config
}

function normalizeEmbeddings (raw) {
  if (!Array.isArray(raw) || !Array.isArray(raw[0])) throw new Error('Invalid embedding response structure')
  return raw[0].map((vector) => Array.from(vector))
}

// avg cosine similarity of each sequence's embedding vs the baseline's, matching
// the desktop similarityStats.avg. Baseline is the first successful config for
// the same input mode, so it reads ~1.0 by construction.
function avgCosine (baseline, candidate) {
  if (!baseline || !candidate || baseline.length !== candidate.length || baseline.length === 0) return null
  let sum = 0
  for (let i = 0; i < baseline.length; i++) sum += cosineSimilarity(baseline[i], candidate[i])
  return sum / baseline.length
}

function labelFor (spec, device, batchSize, flashAttn, inputMode) {
  return `[${spec.id} q=${spec.quant}] [${device}] [bs=${batchSize}] [fa=${flashAttn}] [input=${inputMode}]`
}

// Records a placeholder row with no metrics for a single config. The renderer
// shows any row without ppTPS/latency as Crashed, and the reporter flushes each
// record to console immediately, so a config that crashes the Device Farm
// session after its placeholder is written still leaves a Crashed row. A
// successful run records the real metrics afterwards, superseding the
// placeholder.
function recordCrashedPlaceholder (label, device, model) {
  recordPerformance(label, { deviceId: device, status: 'crashed', model })
}

// Registers the benchmark test for one (model x quant x batchSize x flashAttn),
// sweeping device x inputMode internally. One Device Farm session per call.
// batchSize and flashAttn are fixed per shard (the reload-heavy axes live in the
// shard key), so this session does at most 2 model loads (one per device) — not
// one per batchSize/flashAttn — which is what keeps it inside the phone's memory
// budget. inputMode re-runs the already-loaded model with a different input, no
// reload. A config that fails to load or run is caught and reported as Crashed
// rather than aborting the sweep.
function benchmarkModel (modelName, quant, batchSize, flashAttn) {
  const spec = modelSpec(modelName, quant)
  safeTest(`Mobile perf benchmark: ${spec.id} q=${quant} bs=${batchSize} fa=${flashAttn} (ppTPS / latency / embeddings-per-sec / cosine)`, {
    timeout: 1_800_000,
    skip: !isMobile
  }, async (t) => {
    const specLogger = attachSpecLogger({ forwardToConsole: true })
    try {
      const modelPath = await ensureBenchmarkModel(spec)
      const modelDir = path.dirname(modelPath)

      // Once the model is downloaded, write a Crashed placeholder for EVERY
      // config before any load/run, so a hard native crash mid-sweep still
      // leaves rows for the rest. Real metrics supersede these. A download
      // failure throws above this loop and leaves no rows for this shard.
      for (const device of DEVICES) {
        for (const inputMode of INPUT_MODES) {
          recordCrashedPlaceholder(labelFor(spec, device, batchSize, flashAttn, inputMode), device, `${spec.id}-${quant}`)
        }
      }

      // Cosine baseline per input mode: the first successful config's embeddings
      // for that mode (reads ~1.0 against itself).
      const baselineByInputMode = new Map()

      for (const device of DEVICES) {
        let addon = null
        try {
          addon = new GGMLBert({
            files: { model: [modelPath] },
            config: buildConfig(device, batchSize, flashAttn, modelDir),
            logger: { error: () => {}, warn: () => {}, info: () => {}, debug: () => {} },
            opts: { stats: true }
          })
          await addon.load()
        } catch (loadErr) {
          t.comment(`[${spec.id} q=${quant}] [${device}] [bs=${batchSize}] [fa=${flashAttn}] load failed (reported as Crashed): ${loadErr && loadErr.message ? loadErr.message : loadErr}`)
          await (addon && addon.unload && addon.unload().catch(() => {}))
          continue
        }

        try {
          // Warm up per loaded backend (discarded, never a measured rep) to
          // prime GPU kernels/caches so rep 1 isn't a cold-start outlier.
          try {
            for (let warm = 1; warm <= PERF_WARMUP_RUNS; warm++) {
              const w = await addon.run(inputsFor(INPUT_MODES[0]))
              await w.await()
              t.comment(`[${spec.id} q=${quant}] [${device}] warmup ${warm}/${PERF_WARMUP_RUNS} - perf NOT recorded`)
            }
          } catch (warmErr) {
            t.comment(`[${spec.id} q=${quant}] [${device}] warmup failed: ${warmErr && warmErr.message ? warmErr.message : warmErr}`)
          }
          for (const inputMode of INPUT_MODES) {
            const label = labelFor(spec, device, batchSize, flashAttn, inputMode)
            try {
              const ppTpsValues = []
              const latencyValues = []
              const embPerSecValues = []
              let firstEmbeddings = null
              let inputTokens = null
              for (let rep = 1; rep <= PERF_RUNS; rep++) {
                const response = await addon.run(inputsFor(inputMode))
                const raw = await response.await()
                const stats = response.stats || {}
                const embeddings = normalizeEmbeddings(raw)
                if (!firstEmbeddings) firstEmbeddings = embeddings
                if (inputTokens == null && stats.total_tokens != null) inputTokens = stats.total_tokens
                const latencyMs = reliablePrefillMs(stats.total_time_ms)
                const ppTps = latencyMs != null ? prefillTokensPerSecond(stats) : null
                if (ppTps != null) ppTpsValues.push(ppTps)
                if (latencyMs != null) latencyValues.push(latencyMs)
                if (latencyMs != null && latencyMs > 0) embPerSecValues.push(embeddings.length / (latencyMs / 1000))
              }

              // Cosine baseline per input mode is the first successful config's
              // first-rep embeddings; reps of the same config are numerically
              // identical, so one rep's embeddings suffice for the comparison.
              let cosine = null
              if (!baselineByInputMode.has(inputMode)) {
                baselineByInputMode.set(inputMode, firstEmbeddings)
                cosine = 1
              } else {
                cosine = avgCosine(baselineByInputMode.get(inputMode), firstEmbeddings)
              }

              t.comment(recordPerformance(label, {
                deviceId: device,
                ppTps: meanOf(ppTpsValues),
                ppTpsStd: stdOf(ppTpsValues),
                latencyMs: meanOf(latencyValues),
                latencyMsStd: stdOf(latencyValues),
                embPerSec: meanOf(embPerSecValues),
                embPerSecStd: stdOf(embPerSecValues),
                cosine,
                inputTokens,
                // Richest series: ppTPS can be null on a zero-prefill-time
                // edge while latency is still valid, so don't let it under-
                // report the sample count.
                sampleCount: Math.max(ppTpsValues.length, latencyValues.length, embPerSecValues.length),
                status: 'ok',
                model: `${spec.id}-${quant}`
              }))
              t.ok(firstEmbeddings.length > 0, `${label} produced embeddings`)
            } catch (runErr) {
              t.comment(`${label} run failed (reported as Crashed): ${runErr && runErr.message ? runErr.message : runErr}`)
            }
          }
        } finally {
          await addon.unload().catch(() => {})
        }
      }
    } finally {
      specLogger.release()
    }
  })
}

module.exports = { benchmarkModel, modelSpec }
