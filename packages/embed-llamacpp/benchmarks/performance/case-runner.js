'use strict'

const fs = require('bare-fs')
const path = require('bare-path')
const process = require('bare-process')
const {
  elapsedMs,
  round,
  similarityStats,
  cartesianProduct,
  average,
  stddev
} = require('./math')
const { INPUT_MODES, maxBatchForModel } = require('./_sweep-grid')

function createAddonRuntimeLogger (debugEnabled) {
  if (!debugEnabled) {
    return {
      error: () => {},
      warn: () => {},
      info: () => {},
      debug: () => {}
    }
  }

  return {
    error: (...msgs) => console.error(...msgs),
    warn: (...msgs) => console.warn(...msgs),
    info: (...msgs) => console.log(...msgs),
    debug: (...msgs) => console.debug(...msgs)
  }
}

function normalizeEmbeddings (rawEmbeddings) {
  if (!Array.isArray(rawEmbeddings) || !Array.isArray(rawEmbeddings[0])) {
    throw new Error('Invalid embedding response structure')
  }
  return rawEmbeddings[0].map((vector) => Array.from(vector))
}

// The addon's prefill timer (t_p_eval_ms) has ~millisecond resolution. A single
// short input prefills faster than it can measure, so the addon reports a
// sub-millisecond prefill time and a tokens_per_second inflated to ~1e8. Treat
// prefill timing below this floor as unmeasured so the timing-derived metrics
// (ppTPS, latency, embeddings/sec) report null for those configs instead of a
// fabricated value. Array inputs spend real milliseconds in prefill and are
// unaffected.
const MIN_RELIABLE_PREFILL_MS = 1

function reliablePrefillMs (totalTimeMs) {
  return totalTimeMs != null && totalTimeMs >= MIN_RELIABLE_PREFILL_MS ? totalTimeMs : null
}

// Prefill throughput (ppTPS) as measured by the addon; only meaningful when the
// prefill time is reliable, which the caller enforces.
function prefillTokensPerSecond (runtimeStats) {
  return runtimeStats.tokens_per_second != null ? runtimeStats.tokens_per_second : null
}

function buildAddonConfig (runtimeConfig, options = {}) {
  const debugEnabled = !!options.debugEnabled
  const config = { verbosity: debugEnabled ? '2' : '0' }
  if (runtimeConfig.device != null) config.device = String(runtimeConfig.device)
  if (runtimeConfig.batchSize != null) config.batch_size = String(runtimeConfig.batchSize)
  if (runtimeConfig.flashAttn != null) config.flash_attn = String(runtimeConfig.flashAttn)
  if (runtimeConfig.ngl != null) config.gpu_layers = String(runtimeConfig.ngl)
  if (runtimeConfig.noMmap) config['no-mmap'] = ''
  return config
}

function resolveModelName (modelDef, quantization) {
  return modelDef.quantizationFiles[quantization] || null
}

function checkModelExists (modelDir, modelName) {
  return fs.existsSync(path.join(modelDir, modelName))
}

// Highest-fidelity available build, used as the cosine-similarity reference:
// F16 where the model ships it, else the best available quant.
const FIDELITY_ORDER = ['F16', 'Q8_0', 'Q6_K', 'Q4_K_M', 'Q4_1', 'Q4_0']

function buildCases (modelDef, sweep) {
  const defaults = modelDef.defaults
  const baseQuant = FIDELITY_ORDER.find((quant) => !!resolveModelName(modelDef, quant)) ||
    modelDef.quantizations[0]
  if (baseQuant == null) {
    throw new Error(`No baseline quantization configured for model "${modelDef.id}"`)
  }
  const supportedQuants = sweep.quantization
    .filter((quant) => !!resolveModelName(modelDef, quant))

  if (supportedQuants.length === 0) {
    throw new Error(`No supported quantizations found for model "${modelDef.id}"`)
  }

  const cases = []
  for (const inputMode of INPUT_MODES) {
    cases.push({
      caseId: `${modelDef.id}__q=${baseQuant}__baseline-defaults__input=${inputMode}`,
      parameter: 'baseline',
      quantization: baseQuant,
      modelName: resolveModelName(modelDef, baseQuant),
      runtimeConfig: { ...defaults },
      inputMode,
      isBaseline: true
    })
  }

  // Skip batch sizes the model can't hold (e.g. embeddingGemma's 2048 context),
  // which would otherwise overflow and crash every such config.
  const maxBatch = maxBatchForModel(modelDef.id)
  const combos = cartesianProduct([
    supportedQuants,
    sweep.device,
    sweep.batchSize,
    sweep.flashAttn
  ]).filter(([, , batchSize]) => batchSize <= maxBatch)

  for (const [quantization, device, batchSize, flashAttn] of combos) {
    for (const inputMode of INPUT_MODES) {
      cases.push({
        caseId: `${modelDef.id}__q=${quantization}__dev=${device}__bs=${batchSize}__fa=${flashAttn}__input=${inputMode}`,
        parameter: 'full-grid',
        quantization,
        modelName: resolveModelName(modelDef, quantization),
        runtimeConfig: {
          ...defaults,
          device,
          batchSize,
          flashAttn
        },
        inputMode,
        isBaseline: false
      })
    }
  }

  cases.sort((a, b) => Number(b.isBaseline) - Number(a.isBaseline))
  return cases
}

// Embedding is a single forward pass (prefill only), so every metric here is a
// prefill or end-to-end quantity — there is no decode phase. The renderer reads:
//   ppTpsMean/ppTpsStd      prefill tokens/sec (addon tokens_per_second)
//   latencyMsMean/latencyMsStd  prefill time in ms (addon total_time_ms)
//   embPerSecMean/embPerSecStd  embeddings/sec = sequences / (prefill_ms / 1000)
//   inputTokens             tokens fed to the model for this case
// loadMs/runMs/unloadMs are kept for the legacy .jsonl/.md reporters.
function aggregateRunMetrics (runMetrics) {
  const repeatsSucceeded = runMetrics.length
  if (repeatsSucceeded === 0) {
    return {
      repeats: 0,
      loadMs: null,
      runMs: null,
      unloadMs: null,
      tps: null,
      ppTpsMean: null,
      ppTpsStd: null,
      latencyMsMean: null,
      latencyMsStd: null,
      embPerSecMean: null,
      embPerSecStd: null,
      inputTokens: null
    }
  }

  const wallMsValues = runMetrics.map((x) => x.wallMs).filter((x) => x != null)
  const prefillMsValues = runMetrics.map((x) => x.prefillMs).filter((x) => x != null)
  const ppTpsValues = runMetrics.map((x) => x.ppTps).filter((x) => x != null)
  // One embeddings/sec sample per repeat, from that repeat's own prefill time.
  const embPerSecValues = runMetrics
    .map((x) => (x.prefillMs != null && x.prefillMs > 0 && x.embeddingCount != null)
      ? x.embeddingCount / (x.prefillMs / 1000)
      : null)
    .filter((x) => x != null)
  const inputTokens = runMetrics.find((x) => x.totalTokens != null)?.totalTokens ?? null

  return {
    repeats: repeatsSucceeded,
    loadMs: round(runMetrics[0].loadMs, 3),
    runMs: wallMsValues.length ? round(average(wallMsValues), 3) : null,
    unloadMs: round(runMetrics[0].unloadMs, 3),
    tps: ppTpsValues.length ? round(average(ppTpsValues), 3) : null,
    ppTpsMean: ppTpsValues.length ? round(average(ppTpsValues), 3) : null,
    ppTpsStd: ppTpsValues.length ? round(stddev(ppTpsValues), 3) : null,
    latencyMsMean: prefillMsValues.length ? round(average(prefillMsValues), 3) : null,
    latencyMsStd: prefillMsValues.length ? round(stddev(prefillMsValues), 3) : null,
    embPerSecMean: embPerSecValues.length ? round(average(embPerSecValues), 3) : null,
    embPerSecStd: embPerSecValues.length ? round(stddev(embPerSecValues), 3) : null,
    inputTokens
  }
}

async function runCaseWithRepeats ({ AddonCtor, modelDir, modelName, runtimeConfig, inputs, repeats, onRepeatComplete, debugEnabled }) {
  const addonConfig = buildAddonConfig(runtimeConfig, { debugEnabled })
  const addonRuntimeLogger = createAddonRuntimeLogger(debugEnabled)

  let model = null
  let loadMs = null
  let unloadMs = null
  let firstEmbeddings = null
  const runMetrics = []
  const errors = []
  let primaryError = null
  const cleanupErrors = []

  try {
    model = new AddonCtor({
      files: { model: [path.join(modelDir, modelName)] },
      config: addonConfig,
      logger: addonRuntimeLogger,
      opts: { stats: true }
    })

    const loadStart = process.hrtime()
    await model.load()
    loadMs = elapsedMs(loadStart)

    // Warmup run (discarded) so the first measured repeat isn't skewed by
    // cold-start graph build / GPU kernel warmup. Without it the first run is a
    // large outlier that makes the ppTPS / latency mean ± stddev meaningless.
    // Mirrors the LLM desktop sweep and the mobile runner's warmup.
    try {
      const warmup = await model.run(inputs)
      await warmup.await()
    } catch (_) { /* measured runs below surface any real error */ }

    for (let repeat = 1; repeat <= repeats; repeat++) {
      try {
        const runStart = process.hrtime()
        const response = await model.run(inputs)
        const rawEmbeddings = await response.await()
        const wallMs = elapsedMs(runStart)
        const runtimeStats = response.stats
        const embeddings = normalizeEmbeddings(rawEmbeddings)
        if (!firstEmbeddings) {
          firstEmbeddings = embeddings
        }
        // ppTPS = prefill tokens/sec; prefillMs = prefill time; wallMs = end-to-end
        // run latency; embeddingCount = sequences embedded in this run (used to
        // derive embeddings/sec). Embedding is a single forward pass, so there is
        // no decode phase and no generated-token metric.
        const prefillMs = reliablePrefillMs(runtimeStats.total_time_ms)
        runMetrics.push({
          loadMs,
          prefillMs,
          wallMs,
          ppTps: prefillMs != null ? prefillTokensPerSecond(runtimeStats) : null,
          totalTokens: runtimeStats.total_tokens,
          embeddingCount: embeddings.length,
          unloadMs: null
        })
      } catch (error) {
        const message = error.message || String(error)
        errors.push({
          repeat,
          message
        })
      } finally {
        if (typeof onRepeatComplete === 'function') {
          onRepeatComplete({ repeat, repeats })
        }
      }

      // Small settle delay between repeats (model stays loaded) so consecutive
      // measured runs don't contend and skew the mean ± stddev. Mirrors the LLM
      // sweep's inter-repeat delay.
      if (repeat < repeats) {
        await new Promise((resolve) => setTimeout(resolve, 50))
      }
    }
  } catch (err) {
    primaryError = err
  } finally {
    try {
      if (model) {
        const unloadStart = process.hrtime()
        await model.unload()
        unloadMs = elapsedMs(unloadStart)
      }
    } catch (unloadError) {
      cleanupErrors.push(`unload_error=${unloadError && unloadError.message ? unloadError.message : String(unloadError)}`)
    }
  }

  if (primaryError) {
    const primary = primaryError.message || String(primaryError)
    throw new Error(`Case failed: ${primary}`)
  }

  if (cleanupErrors.length > 0) {
    errors.push({
      repeat: null,
      message: `Cleanup failed: ${cleanupErrors.join('; ')}`
    })
  }

  for (const metric of runMetrics) {
    metric.unloadMs = unloadMs
  }

  return {
    metrics: aggregateRunMetrics(runMetrics),
    embeddings: firstEmbeddings,
    errors,
    repeatsAttempted: repeats,
    repeatsSucceeded: runMetrics.length
  }
}

function buildCaseResult ({
  testCase,
  executionResult,
  baselineEmbeddingsByInputMode,
  repeats,
  failureMessage
}) {
  if (failureMessage) {
    return {
      ...testCase,
      metrics: null,
      similarity: null,
      status: 'failed',
      repeatsAttempted: repeats,
      repeatsSucceeded: 0,
      error: {
        message: failureMessage
      }
    }
  }

  if (testCase.parameter === 'baseline' && executionResult.embeddings) {
    baselineEmbeddingsByInputMode.set(testCase.inputMode, executionResult.embeddings)
  }

  const similarity = testCase.parameter === 'baseline'
    ? (
        executionResult.embeddings
          ? { avg: 1, min: 1, max: 1, count: executionResult.embeddings.length }
          : null
      )
    : similarityStats(
      baselineEmbeddingsByInputMode.get(testCase.inputMode),
      executionResult.embeddings
    )

  const hasRepeatErrors = Array.isArray(executionResult.errors) && executionResult.errors.length > 0
  const status = hasRepeatErrors
    ? (executionResult.repeatsSucceeded > 0 ? 'partial-failure' : 'failed')
    : 'ok'
  const error = hasRepeatErrors
    ? (() => {
        const uniqueMessages = [...new Set(executionResult.errors.map((entry) => entry.message))]
        const detail = uniqueMessages.length === 1
          ? uniqueMessages[0]
          : `${uniqueMessages.length} distinct errors (first: ${uniqueMessages[0]})`
        return {
          message: `${executionResult.errors.length}/${executionResult.repeatsAttempted} repeats failed: ${detail}`,
          repeats: executionResult.errors
        }
      })()
    : null

  return {
    ...testCase,
    metrics: executionResult.metrics,
    similarity,
    status,
    repeatsAttempted: executionResult.repeatsAttempted,
    repeatsSucceeded: executionResult.repeatsSucceeded,
    error
  }
}

async function runModelCases ({
  AddonCtor,
  repeats,
  debugEnabled,
  debugLogger,
  modelDef,
  cases,
  inputsByBatchSize,
  progress
}) {
  debugLogger.log(`\n=== ${modelDef.id} ===`)
  debugLogger.log(`Cases to run: ${cases.length}`)
  const baselineEmbeddingsByInputMode = new Map()
  const caseResults = []

  for (let caseIndex = 0; caseIndex < cases.length; caseIndex++) {
    const testCase = cases[caseIndex]
    let executionResult = null
    let failureMessage = null

    try {
      if (!testCase.modelName) {
        throw new Error(
          `Quantization "${testCase.quantization}" is not configured for model "${modelDef.id}" (case ${testCase.caseId})`
        )
      }
      if (!checkModelExists(modelDef.modelDir, testCase.modelName)) {
        throw new Error(
          `Missing model file for case ${testCase.caseId}: ${path.join(modelDef.modelDir, testCase.modelName)}. ` +
          'Run model preparation first (npm run performance:prepare-models).'
        )
      }

      debugLogger.log(`Running: ${testCase.caseId}`)
      const inputsRaw = inputsByBatchSize[testCase.runtimeConfig.batchSize]
      if (!Array.isArray(inputsRaw) || inputsRaw.length === 0) {
        const configuredBatchSizes = Object.keys(inputsByBatchSize || {}).sort()
        throw new Error(
          `Invalid inputs.json for case ${testCase.caseId}: missing or empty inputs for batch size ` +
          `${testCase.runtimeConfig.batchSize}. Configured batch sizes: ` +
          `${configuredBatchSizes.length ? configuredBatchSizes.join(', ') : '(none)'}`
        )
      }
      // 'single' embeds one sequence; 'array-N' embeds the first N sequences in
      // one call (batched-throughput sweep). inputs.json provides enough
      // sequences per batch size (see MAX_ARRAY_SEQUENCES).
      let inputs
      if (testCase.inputMode === 'single') {
        inputs = inputsRaw[0]
      } else {
        const n = Number(testCase.inputMode.slice('array-'.length))
        if (!Number.isInteger(n) || n <= 0) {
          throw new Error(`Unrecognised input mode "${testCase.inputMode}" for case ${testCase.caseId}`)
        }
        if (inputsRaw.length < n) {
          throw new Error(
            `Invalid inputs.json for case ${testCase.caseId}: input mode ${testCase.inputMode} needs ` +
            `${n} sequences at batch size ${testCase.runtimeConfig.batchSize}, but only ${inputsRaw.length} ` +
            'are provided. Regenerate with `npm run generate:inputs`.'
          )
        }
        inputs = inputsRaw.slice(0, n)
      }
      executionResult = await runCaseWithRepeats({
        AddonCtor,
        modelDir: modelDef.modelDir,
        modelName: testCase.modelName,
        runtimeConfig: testCase.runtimeConfig,
        inputs,
        repeats,
        debugEnabled,
        onRepeatComplete: ({ repeat, repeats: repeatsForCase }) => {
          progress.tick({
            modelId: modelDef.id,
            caseIndex: caseIndex + 1,
            caseCount: cases.length,
            repeat,
            repeats: repeatsForCase
          })
        }
      })
    } catch (error) {
      failureMessage = error.message || String(error)
      debugLogger.warn(`Case failed: ${testCase.caseId}: ${failureMessage}`)
      for (let repeat = 1; repeat <= repeats; repeat++) {
        progress.tick({
          modelId: modelDef.id,
          caseIndex: caseIndex + 1,
          caseCount: cases.length,
          repeat,
          repeats
        })
      }
    }

    const caseResult = buildCaseResult({
      testCase,
      executionResult,
      baselineEmbeddingsByInputMode,
      repeats,
      failureMessage
    })
    caseResults.push(caseResult)

    // Fail fast when the baseline case cannot initialize the model. Continuing
    // the full grid in this state only floods logs with the same fatal error.
    if (failureMessage && testCase.isBaseline &&
        /UnableToLoadModel|Failed to initialize model|failed to load model|failed to create context/i.test(failureMessage)) {
      throw new Error(
        `Baseline case failed to initialize model "${testCase.modelName}". ` +
        'Please re-prepare models and verify disk/free space before running the sweep again. ' +
        `Underlying error: ${failureMessage}`
      )
    }
  }

  return {
    modelId: modelDef.id,
    source: modelDef.source,
    modelDir: modelDef.modelDir,
    cases: caseResults
  }
}

module.exports = {
  buildCases,
  runModelCases
}
