'use strict'

// ─────────────────────────────────────────────────────────────────────────────
// QVAC-20557 — remote Supertonic GPU verification (Mali-Vulkan bring-up).
//
// Runs the @qvac/tts-ggml Supertonic engine once (GPU or CPU per env), reports
// which ggml backend actually engaged, the audio RMS amplitude, and whether the
// PCM contains NaN/Inf. Writes a single JSON line to TTS_OUT so the result
// survives even if bare's stdout is swallowed over `adb shell` / a teardown crash.
//
// The decisive per-op numbers (the `[gpu-diag] dprobe_*` trisection of the
// duration predictor) are emitted by the NATIVE probe build straight to the
// Android log (logcat tag `qvac-supertonic`) — they are NOT in this JS output.
// run-on-device.sh captures them with `adb logcat`.
//
// Env knobs:
//   TTS_USE_GPU=1|0   engage the GPU backend (default 1)
//   TTS_OUT=<path>    JSON result file   (default ./tts-result.out next to this script)
//   TTS_MODEL_DIR=<d> model dir          (default <dirname>/models)
//   TTS_SUPERTONIC=<gguf>                (default <model dir>/supertonic2.gguf)
//   TTS_VOICE=<id>    voice id           (default F1)
//   TTS_TEXT=<text>   sentence to synthesize
//
// Backend id mapping: 0=CPU 1=Metal 2=CUDA 3=Vulkan 4=OpenCL 99=other.
// On a Pixel-9 / Mali-G715 a correct GPU run reports backendId=3 (Vulkan).
// ─────────────────────────────────────────────────────────────────────────────

const fs = require('bare-fs')
const path = require('bare-path')
const os = require('bare-os')
const TTSGgml = require('./index')

function env (name, dflt) { return os.hasEnv(name) ? os.getEnv(name) : dflt }

const HERE = __dirname
const USE_GPU = env('TTS_USE_GPU', '1') === '1'
const MODEL_DIR = env('TTS_MODEL_DIR', path.join(HERE, 'models'))
const SUPERTONIC = env('TTS_SUPERTONIC', path.join(MODEL_DIR, 'supertonic2.gguf'))
const VOICE = env('TTS_VOICE', 'F1')
const OUT = env('TTS_OUT', path.join(HERE, 'tts-result.out'))
const TEXT = env('TTS_TEXT', 'The quick brown fox jumps over the lazy dog.')

let buf = ''
function log (...a) {
  const line = a.map(x => (typeof x === 'string' ? x : JSON.stringify(x))).join(' ')
  buf += line + '\n'
  try { fs.writeFileSync(OUT, buf) } catch (e) {}
  console.log(line)
}

function backendName (id) {
  const m = { 0: 'CPU', 1: 'Metal', 2: 'CUDA', 3: 'Vulkan', 4: 'OpenCL', 99: 'other' }
  return Object.prototype.hasOwnProperty.call(m, id) ? m[id] : ('unknown(' + id + ')')
}

// RMS amplitude + NaN/Inf scan (int16 PCM samples).
function analyse (samples) {
  if (!samples || samples.length === 0) return { rms: 0, bad: 0, n: 0 }
  let sumSq = 0
  let bad = 0
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i]
    if (!Number.isFinite(v)) { bad++; continue }
    const f = v / 32768
    sumSq += f * f
  }
  const n = samples.length - bad
  return { rms: n > 0 ? Math.sqrt(sumSq / n) : 0, bad, n: samples.length }
}

// Load + synthesize once on the requested backend. Returns the stats + PCM +
// analysis so the caller can compare a GPU run against a CPU reference.
async function synth (useGpu) {
  const model = new TTSGgml({
    engine: TTSGgml.ENGINE_SUPERTONIC,
    files: { supertonicModel: SUPERTONIC },
    voice: VOICE,
    config: { language: 'en', useGPU: useGpu },
    opts: { stats: true }
  })
  const tLoad = Date.now()
  await model.load()
  const loadMs = Date.now() - tLoad

  const samples = []
  const tInf = Date.now()
  const response = await model.run({ input: TEXT, type: 'text' })
  await response
    .onUpdate(d => {
      if (d && d.outputArray) {
        for (let i = 0; i < d.outputArray.length; i++) samples.push(d.outputArray[i])
      }
    })
    .await()
  const inferMs = Date.now() - tInf
  const st = response.stats || {}
  const a = analyse(samples)
  try { await model.unload() } catch (_e) {}
  return { st, samples, loadMs, inferMs, rms: a.rms, bad: a.bad, n: a.n }
}

// Per-sample GPU-vs-CPU comparison. Supertonic's CFM is deterministic given the
// same text/voice, so a CORRECT GPU run matches the CPU reference within fp
// tolerance. The Mali-Vulkan miscompute shows up as a large rms-delta even when
// the GPU audio is non-silent (rms>0.01) — which the old audible-only check
// missed. Diffs are in normalised (/32768) units.
function compare (gpu, cpu) {
  const n = Math.min(gpu.length, cpu.length)
  if (n === 0) return { rmsDelta: 1, maxAbsDiff: 1, lenGpu: gpu.length, lenCpu: cpu.length }
  let sumSq = 0
  let maxAbs = 0
  for (let i = 0; i < n; i++) {
    const d = (gpu[i] - cpu[i]) / 32768
    sumSq += d * d
    const ad = Math.abs(d)
    if (ad > maxAbs) maxAbs = ad
  }
  return { rmsDelta: Math.sqrt(sumSq / n), maxAbsDiff: maxAbs, lenGpu: gpu.length, lenCpu: cpu.length }
}

async function main () {
  log('==== QVAC-20557 SUPERTONIC GPU VERIFY ====')
  log('platform   :', os.platform(), os.arch())
  log('useGPU     :', String(USE_GPU))
  log('model dir  :', MODEL_DIR)
  log('supertonic :', SUPERTONIC)
  log('voice      :', VOICE)
  log('text       :', JSON.stringify(TEXT))
  log('-----------------------------')

  if (!fs.existsSync(SUPERTONIC)) {
    log('FATAL: supertonic model not found at', SUPERTONIC)
    log('---- RESULT_JSON ' + JSON.stringify({ ok: false, error: 'model_missing', path: SUPERTONIC }))
    throw new Error('model_missing')
  }

  const primary = await synth(USE_GPU)
  const st = primary.st
  log('loaded_ms  :', String(primary.loadMs))
  log('---- STATS ----')
  log(JSON.stringify(st))
  log('backendDevice :', String(st.backendDevice))
  log('backendId     :', String(st.backendId), '(' + backendName(st.backendId) + ')')
  log('gpuUnsupported:', String(st.gpuUnsupported))
  log('---- RESULT ----')
  log('infer_ms      :', String(primary.inferMs))
  log('samples       :', String(primary.n))
  log('nonfinite     :', String(primary.bad))
  log('rms           :', primary.rms.toFixed(6))

  // When this is a GPU run, ALSO synthesize on CPU and compare per-sample, so
  // the verdict reflects CORRECTNESS (matches CPU), not just audibility.
  let cmp = null
  let cpuRms = null
  if (USE_GPU) {
    log('-- running CPU reference for GPU-vs-CPU comparison --')
    const ref = await synth(false)
    cpuRms = ref.rms
    cmp = compare(primary.samples, ref.samples)
    log('cpu_rms       :', cpuRms.toFixed(6))
    log('rms_delta     :', cmp.rmsDelta.toFixed(6))
    log('max_abs_diff  :', cmp.maxAbsDiff.toFixed(6))
    log('len gpu/cpu   :', String(cmp.lenGpu) + '/' + String(cmp.lenCpu))
  }

  const audible = primary.rms > 0.01 && primary.bad === 0
  // GPU is CORRECT only if audible AND it matches the CPU reference. A correct
  // run has rms_delta ~0; a miscompute (even audible) blows past the tolerance.
  const MATCH_RMS_TOL = 0.02
  const MATCH_MAXABS_TOL = 0.15
  const correct = USE_GPU
    ? (audible && cmp !== null && cmp.rmsDelta < MATCH_RMS_TOL && cmp.maxAbsDiff < MATCH_MAXABS_TOL)
    : audible
  log('verdict       :',
    correct ? 'CORRECT' : (audible ? 'AUDIBLE_BUT_WRONG (GPU != CPU)' : 'AUDIO_BAD (silent/NaN)'))

  log('---- RESULT_JSON ' + JSON.stringify({
    ok: true,
    useGpu: USE_GPU,
    backendDevice: st.backendDevice,
    backendId: st.backendId,
    backend: backendName(st.backendId),
    gpuUnsupported: st.gpuUnsupported,
    infer_ms: primary.inferMs,
    samples: primary.n,
    nonfinite: primary.bad,
    rms: Number(primary.rms.toFixed(6)),
    cpu_rms: cpuRms !== null ? Number(cpuRms.toFixed(6)) : undefined,
    rms_delta: cmp !== null ? Number(cmp.rmsDelta.toFixed(6)) : undefined,
    max_abs_diff: cmp !== null ? Number(cmp.maxAbsDiff.toFixed(6)) : undefined,
    healthy: audible,
    correct
  }))
  log('----------------')
  log('==== DONE ====')
}

main().then(() => log('EXIT_OK')).catch((e) => {
  log('==== FATAL ====')
  log(e && e.stack ? e.stack : String(e))
  throw e
})
