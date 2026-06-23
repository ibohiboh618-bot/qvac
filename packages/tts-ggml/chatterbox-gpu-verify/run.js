'use strict'

// Chatterbox GPU-vs-CPU verification runner (QVAC-20557, DO-NOT-MERGE diag).
//
// Standalone, env-driven, CHATTERBOX-ONLY (never loads Supertonic). One synth
// per process; run it repeatedly with different env to produce the matrix
//   2 (CPU/GPU) x 2 (S3Gen/T3) x 2 (turbo/MTL).
// Reuses the EXISTING end-to-end synth helpers in ../test/utils. See INSTRUCTIONS.md.
//
// CPU is the known-good reference; GPU is what we are verifying.
//
// Env:
//   CHBX_VARIANT   turbo | mtl              (default turbo)
//   T3_BACKEND     cpu | gpu                (default gpu)
//   S3GEN_BACKEND  cpu | gpu                (default gpu)
//   MODEL_DIR      dir holding the GGUFs    (default ../models)
//   OUT_WAV        output wav path          (default ./out/<auto>.wav)
//   REF_WAV        reference audio wav      (default test/reference-audio/jfk.wav)
//   CORR_TEXT      text to synthesize       (default fixed sentence)
//   SEED           rng seed                 (default 42)
//   BACKENDS_DIR   prebuilds dir override   (optional; dodges the backendsDir lookup)
//
// Backend placement:
//   useGPU              = (S3GEN_BACKEND === 'gpu')   -> both models attempt GPU
//   TTS_CPP_T3_FORCE_CPU=1  when T3_BACKEND === 'cpu' -> pins T3 to CPU only
// The (T3=gpu, S3Gen=cpu) cell is UNREACHABLE (no S3Gen-force-CPU lever) and is
// skipped -- CPU is the reference, only GPU needs verifying.

const proc = require('bare-process')
const path = require('bare-path')
const fs = require('bare-fs')
const { loadChatterboxTTS, runChatterboxTTS } = require('../test/utils/runChatterboxTTS')

function envOr (key, fallback) {
  const v = proc.env ? proc.env[key] : undefined
  return (v != null && v !== '') ? v : fallback
}

function fail (msg) {
  console.error(`[chbx-verify] ERROR: ${msg}`)
  if (typeof proc.exit === 'function') proc.exit(1)
}

async function main () {
  const variant = String(envOr('CHBX_VARIANT', 'turbo')).toLowerCase()
  const t3Backend = String(envOr('T3_BACKEND', 'gpu')).toLowerCase()
  const s3genBackend = String(envOr('S3GEN_BACKEND', 'gpu')).toLowerCase()

  if (variant !== 'turbo' && variant !== 'mtl') {
    return fail(`CHBX_VARIANT must be turbo|mtl, got "${variant}"`)
  }
  if (t3Backend !== 'cpu' && t3Backend !== 'gpu') {
    return fail(`T3_BACKEND must be cpu|gpu, got "${t3Backend}"`)
  }
  if (s3genBackend !== 'cpu' && s3genBackend !== 'gpu') {
    return fail(`S3GEN_BACKEND must be cpu|gpu, got "${s3genBackend}"`)
  }

  // The only unreachable placement: there is no S3Gen-force-CPU lever, so we
  // cannot put S3Gen on CPU while T3 runs on GPU. CPU is the reference anyway.
  if (t3Backend === 'gpu' && s3genBackend === 'cpu') {
    console.log('[chbx-verify] SKIP: (T3=gpu, S3Gen=cpu) is unreachable on this branch.')
    console.log('[chbx-verify] CPU is the known-good reference; only GPU needs verifying. Nothing to run.')
    return
  }

  const modelDir = envOr('MODEL_DIR', path.join(__dirname, '..', 'models'))
  const t3File = variant === 'mtl' ? 'chatterbox-t3-mtl.gguf' : 'chatterbox-t3-turbo.gguf'
  const s3File = variant === 'mtl' ? 'chatterbox-s3gen-mtl.gguf' : 'chatterbox-s3gen.gguf'
  const t3ModelPath = path.join(modelDir, t3File)
  const s3genModelPath = path.join(modelDir, s3File)

  const useGPU = (s3genBackend === 'gpu')
  // Set ONCE before load: bare proc.env mutation mid-process does not reliably
  // reach native std::getenv; set-once does (proven for FORCE_CPU on device).
  if (t3Backend === 'cpu') proc.env.TTS_CPP_T3_FORCE_CPU = '1'
  proc.env.TTS_CPP_GPU_TRACE = '1' // emit per-stage [gpu-diag] to logcat / stderr

  const seed = Number(envOr('SEED', '42'))
  const text = envOr('CORR_TEXT', 'GPU versus CPU correctness check. One, two, three, four, five.')
  const defaultOut = path.join(__dirname, 'out', `chatterbox-${variant}-t3-${t3Backend}-s3gen-${s3genBackend}.wav`)
  const outWav = envOr('OUT_WAV', defaultOut)
  const refWav = envOr('REF_WAV', '') // empty -> helper default (jfk.wav)
  const backendsDir = envOr('BACKENDS_DIR', '')

  console.log(`[chbx-verify] config: variant=${variant} T3=${t3Backend} S3Gen=${s3genBackend} useGPU=${useGPU} seed=${seed}`)
  console.log(`[chbx-verify] t3=${t3ModelPath}`)
  console.log(`[chbx-verify] s3gen=${s3genModelPath}`)
  console.log(`[chbx-verify] out=${outWav}`)

  const loadParams = { modelDir, t3ModelPath, s3genModelPath, useGPU, seed }
  if (refWav) loadParams.refWavPath = refWav
  if (backendsDir) loadParams.backendsDir = backendsDir

  let model
  try {
    model = await loadChatterboxTTS(loadParams)
  } catch (err) {
    return fail(`model load failed: ${err && err.message ? err.message : err}`)
  }

  const res = await runChatterboxTTS(model, {
    text,
    saveWav: true,
    wavOutputPath: outWav
  })

  const stats = (res.data && res.data.stats) ? res.data.stats : {}
  const durMs = (res.data && typeof res.data.durationMs === 'number')
    ? res.data.durationMs.toFixed(0)
    : 'n/a'
  console.log(`[chbx-verify] ${res.output}`)
  console.log(`[chbx-verify] backendDevice=${stats.backendDevice} backendId=${stats.backendId} gpuUnsupported=${stats.gpuUnsupported} rtf=${stats.realTimeFactor}`)
  console.log(`[chbx-verify] samples=${res.data ? res.data.sampleCount : 'n/a'} durationMs=${durMs}`)

  // Surface a silent GPU->CPU fallback: if GPU was requested but the engine
  // reports CPU (backendDevice 0 = CPU, non-zero = GPU), the prebuilds/backends
  // were not found -> this WAV is NOT a GPU verification.
  if (useGPU && stats.backendDevice === 0) {
    console.log('[chbx-verify] WARNING: GPU requested but backendDevice=0 (CPU) -- GPU backend .so not found?')
    console.log('[chbx-verify] WARNING: this WAV is a CPU run, not a GPU run. Fix BACKENDS_DIR / prebuilds placement.')
  }

  // Write a result file too: bare stdout is swallowed over `adb shell`, so this
  // is the reliable channel for backendDevice/stats off-device.
  const result = {
    variant,
    t3Backend,
    s3genBackend,
    useGPU,
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
    console.log(`[chbx-verify] result file: ${resultOut}`)
  } catch (err) {
    console.error(`[chbx-verify] could not write result file: ${err && err.message ? err.message : err}`)
  }

  if (!res.passed) {
    return fail(`synthesis reported not-passed: ${res.output}`)
  }
  console.log('[chbx-verify] DONE')
}

main()
  .then(() => {
    // bare can hang on exit with the GPU context still open; all artifacts are
    // written synchronously above, so force a clean exit.
    if (typeof proc.exit === 'function') proc.exit(0)
  })
  .catch(err => {
    fail(err && err.stack ? err.stack : String(err))
  })
