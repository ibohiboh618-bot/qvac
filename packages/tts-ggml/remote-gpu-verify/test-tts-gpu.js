'use strict'

// ─────────────────────────────────────────────────────────────────────────────
// QVAC-20557 — remote TTS GPU verification (Mali-Vulkan bring-up), BOTH engines.
//
// Runs ONE @qvac/tts-ggml engine (Supertonic OR Chatterbox) once on the requested
// backend, reports which ggml backend actually engaged, the audio RMS amplitude
// and a NaN/Inf scan, and — the point of this round — writes the synthesized
// audio to a WAV file (TTS_WAV_OUT) so the colleague can LISTEN to it by ear.
// A single JSON line (RESULT_JSON) is written to TTS_OUT so the result survives
// even if bare's stdout is swallowed over `adb shell` / a teardown crash.
//
// The decisive per-stage numbers (the `[gpu-diag]` rms/min/max/nan trace of each
// engine stage) are emitted by the NATIVE measurement build straight to the
// Android log (logcat tag `qvac-tts`) when TTS_CPP_GPU_TRACE=1 — they are NOT in
// this JS output. run-on-device.sh captures them with `adb logcat`.
//
// One invocation = one (engine, backend, pad-mode). run-on-device.sh drives the
// full matrix (Supertonic GPU/CPU; Chatterbox raw-GPU / padded-GPU / CPU).
//
// Env knobs:
//   TTS_ENGINE=supertonic|chatterbox   which engine (default supertonic)
//   TTS_USE_GPU=1|0    engage the GPU backend (default 1)
//   TTS_OUT=<path>     JSON result file (default ./tts-result.out next to this script)
//   TTS_WAV_OUT=<path> WAV output file  (default ./tts-out.wav)
//   TTS_MODEL_DIR=<d>  model dir         (default <dirname>/models)
//   TTS_TEXT=<text>    sentence to synthesize
//   TTS_VOICE=<id>     Supertonic voice id (default F1)
//   TTS_REF_WAV=<path> Chatterbox speaker-reference wav (default <dirname>/jfk.wav)
//   TTS_SUPERTONIC / TTS_T3 / TTS_S3GEN override individual model paths.
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
const ENGINE = (env('TTS_ENGINE', 'supertonic') || 'supertonic').toLowerCase()
const USE_GPU = env('TTS_USE_GPU', '1') === '1'
const MODEL_DIR = env('TTS_MODEL_DIR', path.join(HERE, 'models'))
const OUT = env('TTS_OUT', path.join(HERE, 'tts-result.out'))
const WAV_OUT = env('TTS_WAV_OUT', path.join(HERE, 'tts-out.wav'))
const VOICE = env('TTS_VOICE', 'F1')
const REF_WAV = env('TTS_REF_WAV', path.join(HERE, 'jfk.wav'))
const TEXT = env('TTS_TEXT', 'The quick brown fox jumps over the lazy dog.')

const SUPERTONIC = env('TTS_SUPERTONIC', path.join(MODEL_DIR, 'supertonic2.gguf'))
const T3 = env('TTS_T3', path.join(MODEL_DIR, 'chatterbox-t3-turbo.gguf'))
const S3GEN = env('TTS_S3GEN', path.join(MODEL_DIR, 'chatterbox-s3gen.gguf'))

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

// Minimal 16-bit-PCM mono WAV encoder (RIFF). Uses only Uint8Array + DataView
// (standard everywhere — Bare does NOT guarantee a global Buffer), and bare-fs
// writes a Uint8Array directly. Returns the bytes the colleague can play.
function createWavBuffer (samples, sampleRate) {
  const numChannels = 1
  const bytesPerSample = 2
  const blockAlign = numChannels * bytesPerSample
  const byteRate = sampleRate * blockAlign
  const dataSize = samples.length * bytesPerSample
  const out = new Uint8Array(44 + dataSize)
  const dv = new DataView(out.buffer)
  const wstr = (off, s) => { for (let i = 0; i < s.length; i++) out[off + i] = s.charCodeAt(i) }
  wstr(0, 'RIFF')
  dv.setUint32(4, 36 + dataSize, true)
  wstr(8, 'WAVE')
  wstr(12, 'fmt ')
  dv.setUint32(16, 16, true) // fmt chunk size
  dv.setUint16(20, 1, true) // PCM (format)
  dv.setUint16(22, numChannels, true)
  dv.setUint32(24, sampleRate, true)
  dv.setUint32(28, byteRate, true)
  dv.setUint16(32, blockAlign, true)
  dv.setUint16(34, 8 * bytesPerSample, true)
  wstr(36, 'data')
  dv.setUint32(40, dataSize, true)
  let off = 44
  for (let i = 0; i < samples.length; i++) {
    let s = samples[i]
    if (!Number.isFinite(s)) s = 0
    s = Math.max(-32768, Math.min(32767, Math.round(s)))
    dv.setInt16(off, s, true)
    off += 2
  }
  return out
}

function buildModel (useGpu) {
  if (ENGINE === 'chatterbox') {
    return new TTSGgml({
      engine: TTSGgml.ENGINE_CHATTERBOX,
      files: { t3Model: T3, s3genModel: S3GEN },
      referenceAudio: REF_WAV,
      config: { language: 'en', useGPU: useGpu },
      opts: { stats: true }
    })
  }
  return new TTSGgml({
    engine: TTSGgml.ENGINE_SUPERTONIC,
    files: { supertonicModel: SUPERTONIC },
    voice: VOICE,
    config: { language: 'en', useGPU: useGpu },
    opts: { stats: true }
  })
}

async function synth (useGpu) {
  const model = buildModel(useGpu)
  const tLoad = Date.now()
  await model.load()
  const loadMs = Date.now() - tLoad

  const samples = []
  let sampleRate = ENGINE === 'chatterbox' ? 24000 : 44100
  const tInf = Date.now()
  const response = await model.run({ input: TEXT, type: 'text' })
  await response
    .onUpdate(d => {
      if (d && d.outputArray) {
        if (d.sampleRate) sampleRate = d.sampleRate
        for (let i = 0; i < d.outputArray.length; i++) samples.push(d.outputArray[i])
      }
    })
    .await()
  const inferMs = Date.now() - tInf
  const st = response.stats || {}
  const a = analyse(samples)
  try { await model.unload() } catch (_e) {}
  return { st, samples, sampleRate, loadMs, inferMs, rms: a.rms, bad: a.bad, n: a.n }
}

function checkModels () {
  if (ENGINE === 'chatterbox') {
    if (!fs.existsSync(T3)) return { ok: false, error: 'model_missing', path: T3 }
    if (!fs.existsSync(S3GEN)) return { ok: false, error: 'model_missing', path: S3GEN }
    if (!fs.existsSync(REF_WAV)) return { ok: false, error: 'ref_wav_missing', path: REF_WAV }
    return { ok: true }
  }
  if (!fs.existsSync(SUPERTONIC)) return { ok: false, error: 'model_missing', path: SUPERTONIC }
  return { ok: true }
}

async function main () {
  log('==== QVAC-20557 TTS GPU VERIFY ====')
  log('engine     :', ENGINE)
  log('platform   :', os.platform(), os.arch())
  log('useGPU     :', String(USE_GPU))
  log('mali_pad   :', env('TTS_CPP_MALI_PAD', '(default/device-identity)'))
  log('model dir  :', MODEL_DIR)
  if (ENGINE === 'chatterbox') {
    log('t3         :', T3)
    log('s3gen      :', S3GEN)
    log('ref wav    :', REF_WAV)
  } else {
    log('supertonic :', SUPERTONIC)
    log('voice      :', VOICE)
  }
  log('text       :', JSON.stringify(TEXT))
  log('wav out    :', WAV_OUT)
  log('-----------------------------')

  const chk = checkModels()
  if (!chk.ok) {
    log('FATAL:', chk.error, 'at', chk.path)
    log('---- RESULT_JSON ' + JSON.stringify({ ok: false, engine: ENGINE, error: chk.error, path: chk.path }))
    throw new Error(chk.error)
  }

  const r = await synth(USE_GPU)
  const st = r.st
  log('loaded_ms  :', String(r.loadMs))
  log('---- STATS ----')
  log(JSON.stringify(st))
  log('backendDevice :', String(st.backendDevice))
  log('backendId     :', String(st.backendId), '(' + backendName(st.backendId) + ')')
  log('gpuUnsupported:', String(st.gpuUnsupported))
  log('---- RESULT ----')
  log('infer_ms      :', String(r.inferMs))
  log('sample_rate   :', String(r.sampleRate))
  log('samples       :', String(r.n))
  log('nonfinite     :', String(r.bad))
  log('rms           :', r.rms.toFixed(6))

  // Write the audio out for by-ear verification (the goal of this round).
  let wavBytes = 0
  try {
    const wav = createWavBuffer(r.samples, r.sampleRate)
    fs.writeFileSync(WAV_OUT, wav)
    wavBytes = wav.length
    log('wav written   :', WAV_OUT, '(' + String(wavBytes) + ' bytes)')
  } catch (e) {
    log('wav write FAILED:', String(e && e.message ? e.message : e))
  }

  const audible = r.rms > 0.01 && r.bad === 0
  log('verdict       :', audible ? 'AUDIBLE' : 'AUDIO_BAD (silent/NaN)')

  log('---- RESULT_JSON ' + JSON.stringify({
    ok: true,
    engine: ENGINE,
    useGpu: USE_GPU,
    maliPad: env('TTS_CPP_MALI_PAD', null),
    backendDevice: st.backendDevice,
    backendId: st.backendId,
    backend: backendName(st.backendId),
    gpuUnsupported: st.gpuUnsupported,
    infer_ms: r.inferMs,
    sample_rate: r.sampleRate,
    samples: r.n,
    nonfinite: r.bad,
    rms: Number(r.rms.toFixed(6)),
    wav_bytes: wavBytes,
    healthy: audible
  }))
  log('----------------')
  log('==== DONE ====')
}

main().then(() => log('EXIT_OK')).catch((e) => {
  log('==== FATAL ====')
  log(e && e.stack ? e.stack : String(e))
  throw e
})
