'use strict'

// Shared runner for the mobile embed perf benchmark. Sharded into one test file
// per (model x quant) (benchmark-perf-<model>-<quant>.test.js); this module
// holds the logic they all share. Underscore prefix keeps it out of the mobile
// test generator (it is not a *.test.js file).
//
// Each shard loads its (model, quant) once and sweeps the rest of the axes
// INTERNALLY on the device: device(cpu,gpu) x batchSize(256,512,1024,2048) x
// flashAttn(off,on) x inputMode(single,array). Embedding is a single
// prefill-only forward pass, so each config records prefill throughput (ppTPS),
// prefill latency (ms), embeddings/sec, and cosine similarity vs an in-run
// baseline (the first successful config for the same input mode). The axes and
// input modes come from the benchmark sweep grid, so the mobile sweep never
// drifts from the desktop one.

const fs = require('bare-fs')
const path = require('bare-path')
const os = require('bare-os')
const GGMLBert = require('../../index.js')
const { safeTest, downloadFile } = require('./utils')
const { attachSpecLogger } = require('./spec-logger')
const { recordPerformance, isMobile } = require('./_perf-helper.js')
const { matrix } = require('./_benchmark-matrix.js')
const { PARAMETER_SWEEP, INPUT_MODES } = require('../../benchmarks/performance/_sweep-grid.js')
const { prefillTokensPerSecond } = require('../../benchmarks/performance/case-runner.js')
const { cosineSimilarity, average, stddev } = require('../../benchmarks/performance/math.js')

// Measured repetitions per config, reported as mean +/- stddev (matching the
// desktop sweep). Repeating on-device guards against a single shot skewed by
// mobile thermal throttling.
const MOBILE_REPEATS = 3

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

// Candidate GGUF filenames for a (repo, quant). HF repos are inconsistent about
// case (e.g. 300M vs 300m in the stem, F16 vs f16 in the quant), so we try each
// variant; downloadFile cycles through the array across retries.
function downloadUrls (repo, revision, quant) {
  const name = repo.split('/').pop()
  const stem = name.toUpperCase().endsWith('-GGUF') ? name.slice(0, -5) : name
  const stems = [...new Set([stem, stem.replace(/300m/i, '300M'), stem.replace(/300m/i, '300m')])]
  const u = quant.toUpperCase()
  const quants = [...new Set([u, u.toLowerCase(), ...(u === 'F16' ? ['f16', 'fp16'] : [])])]
  const urls = []
  for (const s of stems) for (const q of quants) urls.push(`https://huggingface.co/${repo}/resolve/${revision}/${s}-${q}.gguf`)
  return [...new Set(urls)]
}

function modelSpec (modelName, quant) {
  const cell = matrix().find((c) => c.model === modelName && c.quant === quant)
  if (!cell) throw new Error(`No benchmark matrix cell for model "${modelName}" quant "${quant}"`)
  const urls = downloadUrls(cell.repo, cell.revision, quant)
  // Local file name: stem-quant.gguf, slug-free of slashes; the actual file
  // downloaded may use a different case, so name it canonically for the cache.
  const stem = cell.repo.split('/').pop().replace(/-GGUF$/i, '')
  return { id: cell.model, quant, name: `${stem}-${quant}.gguf`, urls }
}

// Mirrors test/integration/utils.js ensureModel, but takes an ordered URL list
// (HF filename case varies per repo) and downloads into the shared model cache.
async function ensureBenchmarkModel (spec) {
  const modelDir = path.resolve(__dirname, '../model')
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

// Registers the benchmark test for one (model x quant), sweeping
// device x batchSize x flashAttn x inputMode internally. One Device Farm
// session per call. A config that fails to load or run is caught and reported
// as Crashed rather than aborting the sweep.
function benchmarkModel (modelName, quant) {
  const spec = modelSpec(modelName, quant)
  safeTest(`Mobile perf benchmark: ${spec.id} q=${quant} (ppTPS / latency / embeddings-per-sec / cosine)`, {
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
        for (const batchSize of PARAMETER_SWEEP.batchSize) {
          for (const flashAttn of PARAMETER_SWEEP.flashAttn) {
            for (const inputMode of INPUT_MODES) {
              recordCrashedPlaceholder(labelFor(spec, device, batchSize, flashAttn, inputMode), device, `${spec.id}-${quant}`)
            }
          }
        }
      }

      // Cosine baseline per input mode: the first successful config's embeddings
      // for that mode (reads ~1.0 against itself).
      const baselineByInputMode = new Map()

      for (const device of DEVICES) {
        for (const batchSize of PARAMETER_SWEEP.batchSize) {
          for (const flashAttn of PARAMETER_SWEEP.flashAttn) {
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
              for (const inputMode of INPUT_MODES) {
                const label = labelFor(spec, device, batchSize, flashAttn, inputMode)
                try {
                  const ppTpsValues = []
                  const latencyValues = []
                  const embPerSecValues = []
                  let firstEmbeddings = null
                  let inputTokens = null
                  for (let rep = 1; rep <= MOBILE_REPEATS; rep++) {
                    const response = await addon.run(inputsFor(inputMode))
                    const raw = await response.await()
                    const stats = response.stats || {}
                    const embeddings = normalizeEmbeddings(raw)
                    if (!firstEmbeddings) firstEmbeddings = embeddings
                    if (inputTokens == null && stats.total_tokens != null) inputTokens = stats.total_tokens
                    const ppTps = prefillTokensPerSecond(stats)
                    const latencyMs = stats.total_time_ms != null ? stats.total_time_ms : null
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
                    sampleCount: ppTpsValues.length,
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
        }
      }
    } finally {
      specLogger.release()
    }
  })
}

module.exports = { benchmarkModel, modelSpec }
