'use strict'

// Supertonic GPU-verify runner (QVAC-20557, DO-NOT-MERGE diag).
//
// Sibling of run.js (which is Chatterbox-only). Standalone, env-driven,
// SUPERTONIC-ONLY. One synth per process; run it once per variant (v2 / v3).
// Reuses the EXISTING ../test/utils/runSupertonicTTS helpers — no synth logic
// is duplicated here. Supertonic already passes allow_arm_mali=true in master,
// so it runs on the ARM Mali Vulkan GPU with the SAME overlay build as the
// Chatterbox verify; this just drives it and captures audio + backend info.
//
// CPU is the known-good reference; GPU (default) is what we are verifying.
//
// Env:
//   ST_VARIANT    v2 | v3                  (default v2)
//   ST_BACKEND    gpu | cpu                (default gpu)
//   ST3_QUANT     q4_0 | q8_0 | f16 | f32  (default q4_0; only used for v3)
//   MODEL_DIR     dir holding the GGUFs    (default ../models)
//   ST_MODEL      explicit gguf filename   (overrides ST_VARIANT/ST3_QUANT)
//   OUT_WAV       output wav path          (default ./out/supertonic-<auto>.wav)
//   RESULT_OUT    result json path         (default <OUT_WAV>.result.json)
//   ST_TEXT       text to synthesize       (default fixed sentence)
//   ST_VOICE      voice id                 (default F1)
//   ST_LANG       language code            (default en)
//   SEED          rng seed                 (default 42)

const proc = require('bare-process')
const path = require('bare-path')
const fs = require('bare-fs')
const { loadSupertonicTTS, runSupertonicTTS } = require('../test/utils/runSupertonicTTS')

function envOr (key, fallback) {
  const v = proc.env ? proc.env[key] : undefined
  return (v != null && v !== '') ? v : fallback
}

function fail (msg) {
  console.error(`[st-verify] ERROR: ${msg}`)
  if (typeof proc.exit === 'function') proc.exit(1)
}

function modelFileFor (variant, quant) {
  if (variant === 'v3') return `supertonic3-${quant}.gguf`
  if (variant === 'v2') return 'supertonic2.gguf'
  return null
}

async function main () {
  const variant = String(envOr('ST_VARIANT', 'v2')).toLowerCase()
  const backend = String(envOr('ST_BACKEND', 'gpu')).toLowerCase()
  const quant = String(envOr('ST3_QUANT', 'q4_0')).toLowerCase()

  if (variant !== 'v2' && variant !== 'v3') {
    return fail(`ST_VARIANT must be v2|v3, got "${variant}"`)
  }
  if (backend !== 'cpu' && backend !== 'gpu') {
    return fail(`ST_BACKEND must be cpu|gpu, got "${backend}"`)
  }

  const modelDir = envOr('MODEL_DIR', path.join(__dirname, '..', 'models'))
  const modelFile = envOr('ST_MODEL', modelFileFor(variant, quant))
  const supertonicModelPath = path.join(modelDir, modelFile)

  const useGPU = (backend === 'gpu')
  const seed = Number(envOr('SEED', '42'))
  const text = envOr('ST_TEXT', 'GPU versus CPU correctness check. One, two, three, four, five.')
  const voice = envOr('ST_VOICE', 'F1')
  const language = envOr('ST_LANG', 'en')
  const defaultOut = path.join(__dirname, 'out', `supertonic-${variant}-${backend}.wav`)
  const outWav = envOr('OUT_WAV', defaultOut)

  console.log(`[st-verify] config: variant=${variant} backend=${backend} useGPU=${useGPU} model=${modelFile} seed=${seed}`)
  console.log(`[st-verify] model=${supertonicModelPath}`)
  console.log(`[st-verify] out=${outWav}`)

  if (!fs.existsSync(supertonicModelPath)) {
    return fail(`model not found: ${supertonicModelPath} (download group supertonic${variant === 'v3' ? '3' : '2'})`)
  }

  let model
  try {
    model = await loadSupertonicTTS({ supertonicModelPath, useGPU, voice, language, seed })
  } catch (err) {
    return fail(`model load failed: ${err && err.message ? err.message : err}`)
  }

  const res = await runSupertonicTTS(model, {
    text,
    saveWav: true,
    wavOutputPath: outWav
  })

  const stats = (res.data && res.data.stats) ? res.data.stats : {}
  const durMs = (res.data && typeof res.data.durationMs === 'number')
    ? res.data.durationMs.toFixed(0)
    : 'n/a'
  console.log(`[st-verify] ${res.output}`)
  console.log(`[st-verify] backendDevice=${stats.backendDevice} backendId=${stats.backendId} gpuUnsupported=${stats.gpuUnsupported} rtf=${stats.realTimeFactor}`)
  console.log(`[st-verify] samples=${res.data ? res.data.sampleCount : 'n/a'} durationMs=${durMs}`)

  // Surface a silent GPU->CPU fallback: GPU requested but engine reports CPU
  // (backendDevice 0 = CPU) -> prebuilds/backends not found, or Mali declined.
  if (useGPU && stats.backendDevice === 0) {
    console.log('[st-verify] WARNING: GPU requested but backendDevice=0 (CPU) -- this WAV is a CPU run, not a GPU run.')
  }

  const result = {
    engine: 'supertonic',
    variant,
    backend,
    useGPU,
    modelFile,
    backendDevice: stats.backendDevice,
    backendId: stats.backendId,
    gpuUnsupported: stats.gpuUnsupported,
    realTimeFactor: stats.realTimeFactor,
    sampleCount: res.data ? res.data.sampleCount : null,
    durationMs: res.data ? res.data.durationMs : null,
    passed: res.passed,
    wav: outWav
  }
  const resultOut = envOr('RESULT_OUT', `${outWav}.result.json`)
  try {
    fs.writeFileSync(resultOut, JSON.stringify(result, null, 2))
    console.log(`[st-verify] result file: ${resultOut}`)
  } catch (err) {
    console.error(`[st-verify] could not write result file: ${err && err.message ? err.message : err}`)
  }

  if (!res.passed) {
    return fail(`synthesis reported not-passed: ${res.output}`)
  }
  console.log('[st-verify] DONE')
}

main()
  .then(() => {
    // bare can hang on exit with the GPU context still open; artifacts are
    // written synchronously above, so force a clean exit.
    if (typeof proc.exit === 'function') proc.exit(0)
  })
  .catch(err => {
    fail(err && err.stack ? err.stack : String(err))
  })
