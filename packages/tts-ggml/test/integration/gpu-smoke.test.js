'use strict'

// GPU smoke tests for both tts-ggml engines (chatterbox + supertonic).
//
// Mirrors transcription-parakeet/test/integration/gpu-smoke.test.js's
// strict-on-CPU policy: a useGPU=true request that resolves to the CPU
// backend on a GPU-capable platform is treated as a regression because
// it usually means a build / linkage / kernel-init drift that CI must
// catch.  Set QVAC_TTS_GPU_SMOKE_RELAX=1 to downgrade the gate to a
// warning (e.g. for a Linux host without Vulkan SDK, an emulator
// without Metal, or an Adreno-tier device that ggml-opencl rejects by
// design).
//
// CI runners without a real GPU (or hosted macOS where the
// Paravirtual Metal device crashes ggml's encoder) export NO_GPU=true
// to skip every smoke entry.  Real GPU runners and local dev leave
// NO_GPU unset so the strict assertions still fire there.
//
// The strict gate uses `response.stats.backendDevice` (0=CPU, 1=GPU)
// and `response.stats.backendId` (0=CPU, 1=Metal, 2=CUDA, 3=Vulkan,
// 4=OpenCL, 99=other), both surfaced by ChatterboxModel +
// SupertonicModel after Engine::backend_device() / backend_name() were
// added in tts-cpp.

const fs = require('bare-fs')
const os = require('bare-os')
const path = require('bare-path')
const proc = require('bare-process')
const test = require('brittle')

const { loadChatterboxTTS, runChatterboxTTS, resolveRefWavPath } = require('../utils/runChatterboxTTS')
const { loadSupertonicTTS, runSupertonicTTS } = require('../utils/runSupertonicTTS')
const { ensureChatterboxModels, ensureSupertonicModel } = require('../utils/downloadModel')
const { recordTtsStats } = require('../utils/perf-helper')
const { assertSampleCorrelation, pearson } = require('../utils/correlation-helper')
const { createWavBuffer } = require('../utils/wav-helper')

const platform = os.platform()
const isMobile = platform === 'ios' || platform === 'android'
const RELAX = proc.env && proc.env.QVAC_TTS_GPU_SMOKE_RELAX === '1'
const NO_GPU = proc.env && proc.env.NO_GPU === 'true'

// DEBUG (QVAC-20557, DO-NOT-MERGE): enable the native per-stage [gpu-diag] trace
// so a device-farm round carries per-stage GPU-vs-CPU stats (rms/nan/inf/min/max)
// in logcat_full.txt, localizing the first stage a GPU backend miscomputes. Set
// before any Engine construction (native getenv runs at load()).
if (proc.env) proc.env.TTS_CPP_GPU_TRACE = '1'

// GPU-vs-CPU correlation threshold. GPU and CPU load the SAME gguf — only the
// compute backend differs — so a correct backend correlates ~0.999+.
const CORR_THRESHOLD = 0.99

function getBaseDir () {
  return isMobile && global.testDir ? global.testDir : '.'
}

function backendIdToName (id) {
  switch (id) {
    case 0: return 'CPU'
    case 1: return 'Metal'
    case 2: return 'CUDA'
    case 3: return 'Vulkan'
    case 4: return 'OpenCL'
    case 99: return 'other-GPU'
    default: return `unknown(${id})`
  }
}

// Which platforms wire up a GPU backend in tts-cpp's vcpkg port
// today (default-features in qvac-registry-vcpkg/ports/tts-cpp/vcpkg.json):
//   - darwin / ios:        metal
//   - linux / win32:       vulkan
//   - android:             vulkan + opencl
function expectsGpu () {
  return (
    platform === 'darwin' ||
    platform === 'ios' ||
    platform === 'linux' ||
    platform === 'win32' ||
    platform === 'android'
  )
}

function assertGpuBackend (t, engineTag, stats) {
  if (!stats) {
    t.fail(`${engineTag}/GPU: no response.stats returned (cannot verify backend)`)
    return
  }
  const dev = stats.backendDevice
  const id = stats.backendId
  const name = backendIdToName(id)
  console.log(`[${engineTag}/GPU] backendDevice=${dev} backendId=${id} (${name})`)

  if (!expectsGpu()) {
    t.is(dev, 0, `${engineTag}/${platform}: backendDevice must be 0 (CPU) on platforms with no GPU wired in`)
    return
  }

  if (dev !== 1) {
    const msg = `${engineTag}/${platform}: expected GPU backend, got ${name} (backendDevice=${dev}, backendId=${id}). ` +
                'useGPU=true was requested but the engine fell back to CPU. ' +
                'Inspect addon native logs for the load-time backend init message.'
    if (RELAX) {
      t.comment(`WARNING (relaxed): ${msg}`)
      t.pass(`${engineTag}/GPU smoke completed (relaxed)`)
    } else {
      t.fail(msg)
    }
    return
  }

  if (platform === 'darwin' || platform === 'ios') {
    t.is(id, 1, `${engineTag}/${platform}: expected Metal backendId=1, got ${name}`)
  } else if (platform === 'linux' || platform === 'win32') {
    t.is(id, 3, `${engineTag}/${platform}: expected Vulkan backendId=3, got ${name}`)
  } else if (platform === 'android') {
    t.ok(id === 3 || id === 4, `${engineTag}/${platform}: expected Vulkan(3) or OpenCL(4) backendId, got ${name}`)
  }
}

// Companion to assertGpuBackend: when the caller passes useGPU=false we
// expect the engine to actually pick the CPU backend.  This is the gate
// that prevents `useGPU=false` from silently still running on GPU when
// the underlying tts-cpp library default is non-zero n_gpu_layers.
function assertCpuBackend (t, engineTag, stats) {
  if (!stats) {
    t.fail(`${engineTag}/CPU: no response.stats returned (cannot verify backend)`)
    return
  }
  const dev = stats.backendDevice
  const id = stats.backendId
  const name = backendIdToName(id)
  console.log(`[${engineTag}/CPU] backendDevice=${dev} backendId=${id} (${name})`)
  t.is(dev, 0, `${engineTag}: useGPU:false must resolve to backendDevice=0 (CPU), got ${name}`)
  t.is(id, 0, `${engineTag}: useGPU:false must resolve to backendId=0 (CPU), got ${name}`)
}

// Records a perf row for a smoke run. Passes stats.backendDevice so
// recordTtsStats tags the row CPU/GPU from the backend the engine actually
// resolved to (0=CPU, 1=GPU) rather than what was requested — so a relaxed
// GPU→CPU fallback is reported honestly. recordTtsStats also derives RTF from
// wall time + audio duration when the addon doesn't report a positive
// realTimeFactor.
function recordSmoke (t, label, result, wallMs) {
  const st = (result && result.data && result.data.stats) || {}
  t.comment(recordTtsStats(
    label,
    { realTimeFactor: st.realTimeFactor, audioDurationMs: st.audioDurationMs || (result && result.data && result.data.durationMs), totalSamples: st.totalSamples, backendDevice: st.backendDevice },
    { wallMs, sampleCount: result && result.data && result.data.sampleCount, model: label }
  ))
}

test('Chatterbox GPU smoke - useGPU=true must engage the GPU backend on GPU-capable platforms', { timeout: 600000, skip: NO_GPU }, async (t) => {
  // DEBUG (QVAC-20557, DO-NOT-MERGE): Android skip REMOVED so the device-farm
  // round runs Chatterbox on Mali/Adreno GPU and measures correctness.
  const baseDir = getBaseDir()
  const modelsDir = path.join(baseDir, 'models')

  const download = await ensureChatterboxModels({ targetDir: modelsDir })
  if (!download.success) {
    t.fail('Chatterbox GGUFs not available - registry fetch failed. Run `npm run download-models:registry` or stage models locally.')
    return
  }

  // Mobile-aware resolution: see multiple-runs.test.js for rationale.
  const refWavPath = resolveRefWavPath({})
  if (!fs.existsSync(refWavPath)) {
    t.pass('Skipped: reference audio missing')
    return
  }

  const model = await loadChatterboxTTS({
    modelDir: download.targetDir,
    refWavPath,
    language: 'en',
    useGPU: true
  })
  try {
    const t0 = Date.now()
    const result = await runChatterboxTTS(
      model,
      { text: 'GPU smoke check.' },
      { minSamples: 5000 }
    )
    const wallMs = Date.now() - t0
    console.log(result.output)
    t.ok(result.passed, 'Chatterbox/GPU produced expected sample count')
    t.ok(result.data.sampleCount > 0, 'Chatterbox/GPU produced audio')
    assertGpuBackend(t, 'Chatterbox', result.data.stats)
    recordSmoke(t, 'chatterbox gpu-smoke', result, wallMs)
  } finally {
    try { await model.unload() } catch (_e) {}
  }
})

test('Supertonic GPU smoke - useGPU=true must engage the GPU backend on GPU-capable platforms', { timeout: 600000, skip: NO_GPU }, async (t) => {
  // DEBUG (QVAC-20557, DO-NOT-MERGE): Android skip REMOVED so the device-farm
  // round runs Supertonic on Mali/Adreno GPU and measures correctness.
  const baseDir = getBaseDir()
  const modelsDir = path.join(baseDir, 'models')

  const download = await ensureSupertonicModel({ targetDir: modelsDir })
  if (!download || !download.success) {
    t.fail('Supertonic GGUF not available - registry fetch failed. Run `npm run download-models:registry` or stage models locally.')
    return
  }

  const supertonicPath = download.path ||
    path.join(modelsDir, 'supertonic.gguf')

  const model = await loadSupertonicTTS({
    supertonicModelPath: supertonicPath,
    language: 'en',
    voice: 'F1',
    useGPU: true
  })
  try {
    const t0 = Date.now()
    const result = await runSupertonicTTS(
      model,
      { text: 'GPU smoke check.' },
      { minSamples: 5000 }
    )
    const wallMs = Date.now() - t0
    console.log(result.output)
    t.ok(result.passed, 'Supertonic/GPU produced expected sample count')
    t.ok(result.data.sampleCount > 0, 'Supertonic/GPU produced audio')
    assertGpuBackend(t, 'Supertonic', result.data.stats)
    recordSmoke(t, 'supertonic gpu-smoke', result, wallMs)
  } finally {
    try { await model.unload() } catch (_e) {}
  }
})

// CPU smoke: useGPU:false must actually pin the engine to CPU on every
// platform (no NO_GPU skip — CPU is expected to work everywhere).  This
// is the counterpart to the GPU smoke above and exists because the
// previous tts-ggml behaviour left n_gpu_layers at the tts-cpp library
// default when useGPU:false was passed without an explicit nGpuLayers,
// which could silently fall back to GPU.  Now that ChatterboxModel /
// SupertonicModel translate explicit useGPU=false → n_gpu_layers=0,
// these tests lock that contract in.
test('Chatterbox CPU smoke - useGPU=false must run on the CPU backend', { timeout: 600000 }, async (t) => {
  const baseDir = getBaseDir()
  const modelsDir = path.join(baseDir, 'models')

  const download = await ensureChatterboxModels({ targetDir: modelsDir })
  if (!download.success) {
    t.fail('Chatterbox GGUFs not available - registry fetch failed. Run `npm run download-models:registry` or stage models locally.')
    return
  }

  // Mobile-aware resolution: see multiple-runs.test.js for rationale.
  const refWavPath = resolveRefWavPath({})
  if (!fs.existsSync(refWavPath)) {
    t.pass('Skipped: reference audio missing')
    return
  }

  const model = await loadChatterboxTTS({
    modelDir: download.targetDir,
    refWavPath,
    language: 'en',
    useGPU: false
  })
  try {
    const t0 = Date.now()
    const result = await runChatterboxTTS(
      model,
      { text: 'CPU smoke check.' },
      { minSamples: 5000 }
    )
    const wallMs = Date.now() - t0
    console.log(result.output)
    t.ok(result.passed, 'Chatterbox/CPU produced expected sample count')
    t.ok(result.data.sampleCount > 0, 'Chatterbox/CPU produced audio')
    assertCpuBackend(t, 'Chatterbox', result.data.stats)
    recordSmoke(t, 'chatterbox cpu-smoke', result, wallMs)
  } finally {
    try { await model.unload() } catch (_e) {}
  }
})

test('Supertonic CPU smoke - useGPU=false must run on the CPU backend', { timeout: 600000 }, async (t) => {
  const baseDir = getBaseDir()
  const modelsDir = path.join(baseDir, 'models')

  const download = await ensureSupertonicModel({ targetDir: modelsDir })
  if (!download || !download.success) {
    t.fail('Supertonic GGUF not available - registry fetch failed. Run `npm run download-models:registry` or stage models locally.')
    return
  }

  const supertonicPath = download.path ||
    path.join(modelsDir, 'supertonic.gguf')

  const model = await loadSupertonicTTS({
    supertonicModelPath: supertonicPath,
    language: 'en',
    voice: 'F1',
    useGPU: false
  })
  try {
    const t0 = Date.now()
    const result = await runSupertonicTTS(
      model,
      { text: 'CPU smoke check.' },
      { minSamples: 5000 }
    )
    const wallMs = Date.now() - t0
    console.log(result.output)
    t.ok(result.passed, 'Supertonic/CPU produced expected sample count')
    t.ok(result.data.sampleCount > 0, 'Supertonic/CPU produced audio')
    assertCpuBackend(t, 'Supertonic', result.data.stats)
    recordSmoke(t, 'supertonic cpu-smoke', result, wallMs)
  } finally {
    try { await model.unload() } catch (_e) {}
  }
})

// ---------------------------------------------------------------------------
// GPU-vs-CPU correctness gate (QVAC-20557, DO-NOT-MERGE measurement).
//
// The smoke tests above only prove the GPU is *engaged*. These prove its OUTPUT
// matches CPU — the signal missing when a GPU could mis-compute yet still emit
// non-silent audio. Supertonic is deterministic for a fixed seed → HARD corr
// gate. Chatterbox's T3 is autoregressive + stochastic → identical seeds do NOT
// give identical tokens across backends, so its end-to-end corr is informational
// (still HARD-fails on a silent/NaN GPU); its real per-stage correctness signal
// is the native [gpu-diag] trace in logcat_full.txt.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// DEBUG (QVAC-20557, DO-NOT-MERGE): per-stage GPU-vs-CPU correlation + artifacts.
// The native engine dumps each stage's raw f32 to $TTS_CPP_GPU_DUMP_DIR as
// <model>_<backend>_<stage>.f32 (set by ensureDumpDir before any model loads);
// we read GPU vs CPU back here and Pearson-correlate per stage to localize the
// FIRST diverging stage, and write the generated GPU/CPU WAVs so the device farm
// can pull them for offline inspection. Stage lists mirror the native dg() order.
// ---------------------------------------------------------------------------
const SUPERTONIC_STAGES = ['latent_in', 'text_emb', 'cfm_latent', 'wav_full']
const CHATTERBOX_STAGES = ['input_embed', 'encoder_mu', 'cfm_mel', 'f0', 'stft', 'hift_wav']

function ensureDumpDir () {
  const dir = path.join(getBaseDir(), 'gpu-diag')
  try { fs.mkdirSync(dir, { recursive: true }) } catch (_e) {
    try { fs.mkdirSync(dir) } catch (_e2) {}
  }
  if (proc.env) proc.env.TTS_CPP_GPU_DUMP_DIR = dir
  // The pinned-tokens file gates the Chatterbox T3 bypass; clear any stale one so
  // the GPU/first run decodes real (stochastic) tokens.
  try { fs.unlinkSync(path.join(dir, 'chatterbox_pinned_tokens.i32')) } catch (_e) {}
  return dir
}

function readF32 (filePath) {
  const buf = fs.readFileSync(filePath)
  const n = buf.length >>> 2
  const out = new Float32Array(n)
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.length)
  for (let i = 0; i < n; i++) out[i] = dv.getFloat32(i * 4, true)
  return out
}

// Read GPU+CPU stage dumps and Pearson-correlate each stage. Logs every number
// (so it lands in the device-farm console log) and, when hard, gates each stage.
function assertPerStageCorrelation (t, model, gpuBtag, stages, dir, opts = {}) {
  const hard = opts.hard === true
  const threshold = opts.threshold !== undefined ? opts.threshold : CORR_THRESHOLD
  for (const stage of stages) {
    const gpuPath = path.join(dir, `${model}_${gpuBtag}_${stage}.f32`)
    const cpuPath = path.join(dir, `${model}_CPU_${stage}.f32`)
    const haveG = fs.existsSync(gpuPath)
    const haveC = fs.existsSync(cpuPath)
    if (!haveG || !haveC) {
      const msg = `[${model}/${stage}/corr] MISSING dump (gpu=${haveG} cpu=${haveC})`
      console.log(msg)
      if (hard) t.fail(msg + ' — native per-stage f32 dump did not land; observability broken')
      continue
    }
    const g = readF32(gpuPath)
    const c = readF32(cpuPath)
    const { corr, n, reason } = pearson(g, c)
    const corrStr = Number.isFinite(corr) ? corr.toFixed(6) : String(corr)
    console.log(`[${model}/${stage}/corr] ${gpuBtag}-vs-CPU gpu_n=${g.length} cpu_n=${c.length} aligned_n=${n} corr=${corrStr}${reason ? ' reason=' + reason : ''}`)
    if (hard) {
      t.ok(Number.isFinite(corr) && corr >= threshold,
        `${model}/${stage}: per-stage GPU-vs-CPU corr ${corrStr} must be >= ${threshold}${reason ? ' (' + reason + ')' : ''}`)
    }
  }
}

function writeDumpWav (dir, name, samples, sampleRate) {
  try {
    const buf = createWavBuffer(samples, sampleRate)
    fs.writeFileSync(path.join(dir, name), buf)
    console.log(`[wav] wrote ${name} (${samples.length} samples @ ${sampleRate} Hz)`)
  } catch (e) {
    console.log(`[wav] failed to write ${name}: ${e && e.message}`)
  }
}

function rms16 (samples) {
  let s = 0
  for (let i = 0; i < samples.length; i++) { const v = samples[i] / 32768; s += v * v }
  return samples.length ? Math.sqrt(s / samples.length) : 0
}

async function runSupertonicOn (useGPU, supertonicPath) {
  const model = await loadSupertonicTTS({
    supertonicModelPath: supertonicPath, language: 'en', voice: 'F1', useGPU, seed: 42
  })
  try {
    const r = await runSupertonicTTS(model, { text: 'GPU versus CPU correctness check.' }, { minSamples: 5000 })
    return { samples: r.data.samples, stats: r.data.stats }
  } finally {
    try { await model.unload() } catch (_e) {}
  }
}

test('Supertonic GPU-vs-CPU correctness - output must correlate with CPU', { timeout: 600000, skip: NO_GPU }, async (t) => {
  if (!expectsGpu()) { t.pass('Supertonic corr: no GPU wired on this platform'); return }
  const dumpDir = ensureDumpDir()
  const baseDir = getBaseDir()
  const modelsDir = path.join(baseDir, 'models')
  const download = await ensureSupertonicModel({ targetDir: modelsDir })
  if (!download || !download.success) {
    t.fail('Supertonic GGUF not available - registry fetch failed.')
    return
  }
  const supertonicPath = download.path || path.join(modelsDir, 'supertonic.gguf')

  const gpu = await runSupertonicOn(true, supertonicPath)
  if (!gpu.stats || gpu.stats.backendDevice !== 1) {
    const msg = `Supertonic corr: GPU did not engage (backendDevice=${gpu.stats && gpu.stats.backendDevice}); engagement is asserted by the smoke test above`
    if (RELAX) { t.pass(msg + ' (relaxed)') } else { t.comment(msg) }
    return
  }
  const cpu = await runSupertonicOn(false, supertonicPath)
  const gpuBtag = backendIdToName(gpu.stats.backendId)
  // Supertonic is deterministic (fixed seed) → both end-to-end and per-stage
  // GPU-vs-CPU correlation are HARD gates.
  assertSampleCorrelation(t, `Supertonic/${gpuBtag}`, gpu.samples, cpu.samples, {
    threshold: CORR_THRESHOLD, minSamples: 1000, minLenRatio: 0.90
  })
  assertPerStageCorrelation(t, 'supertonic', gpuBtag, SUPERTONIC_STAGES, dumpDir, { hard: true })
  // Audio artifacts (pulled by the device farm) + an audible-RMS floor that
  // catches a silent/NaN GPU independently of the correlation maths.
  writeDumpWav(dumpDir, `supertonic_${gpuBtag}.wav`, gpu.samples, 44100)
  writeDumpWav(dumpDir, 'supertonic_CPU.wav', cpu.samples, 44100)
  const r = rms16(gpu.samples)
  console.log(`[Supertonic/${gpuBtag}/rms] ${r.toFixed(5)}`)
  t.ok(r > 0.001, `Supertonic/${gpuBtag}: GPU audio must be audible (rms ${r.toFixed(5)} > 0.001)`)
})

async function runChatterboxOn (useGPU, modelDir, refWavPath) {
  const model = await loadChatterboxTTS({ modelDir, refWavPath, language: 'en', useGPU, seed: 42 })
  try {
    const r = await runChatterboxTTS(model, { text: 'GPU versus CPU correctness check.' }, { minSamples: 5000 })
    return { samples: r.data.samples, stats: r.data.stats }
  } finally {
    try { await model.unload() } catch (_e) {}
  }
}

test('Chatterbox GPU-vs-CPU correctness - token-pinned per-stage + end-to-end', { timeout: 600000, skip: NO_GPU }, async (t) => {
  if (!expectsGpu()) { t.pass('Chatterbox corr: no GPU wired on this platform'); return }
  const dumpDir = ensureDumpDir()
  const baseDir = getBaseDir()
  const modelsDir = path.join(baseDir, 'models')
  const download = await ensureChatterboxModels({ targetDir: modelsDir })
  if (!download.success) {
    t.fail('Chatterbox GGUFs not available - registry fetch failed.')
    return
  }
  const refWavPath = resolveRefWavPath({})
  if (!fs.existsSync(refWavPath)) { t.pass('Skipped: reference audio missing'); return }

  // GPU run: real (stochastic) T3 → S3Gen. The native side writes the tokens it
  // used to chatterbox_speech_tokens.i32 and dumps chatterbox_<gpu>_*.f32 stages.
  const gpu = await runChatterboxOn(true, download.targetDir, refWavPath)
  if (!gpu.stats || gpu.stats.backendDevice !== 1) {
    t.comment(`Chatterbox corr: GPU did not engage (backendDevice=${gpu.stats && gpu.stats.backendDevice}); per-stage [gpu-diag] trace still emitted`)
    return
  }
  const gpuBtag = backendIdToName(gpu.stats.backendId)

  // Pin the GPU run's tokens for the CPU run so both S3Gen runs decode IDENTICAL
  // tokens — removes T3 stochasticity from the comparison and makes it a HARD
  // gate. If the capture file is missing (observability gap), fall back to soft.
  const tokSrc = path.join(dumpDir, 'chatterbox_speech_tokens.i32')
  const tokPin = path.join(dumpDir, 'chatterbox_pinned_tokens.i32')
  let pinned = false
  if (fs.existsSync(tokSrc)) {
    try { fs.writeFileSync(tokPin, fs.readFileSync(tokSrc)); pinned = true } catch (_e) {}
  }
  if (!pinned) {
    t.comment('Chatterbox corr: token capture file missing — comparison is informational (T3 not pinned)')
  }

  const cpu = await runChatterboxOn(false, download.targetDir, refWavPath)
  // Remove the pin so it can't leak into a later test run on the same device.
  try { fs.unlinkSync(tokPin) } catch (_e) {}

  // With pinned tokens + fixed seed the S3Gen comparison is deterministic → HARD
  // (end-to-end AND per-stage). Without pinning, end-to-end stays informational.
  assertSampleCorrelation(t, `Chatterbox/${gpuBtag}`, gpu.samples, cpu.samples, {
    threshold: CORR_THRESHOLD, soft: !pinned, minSamples: 1000, minLenRatio: pinned ? 0.90 : 0.0
  })
  assertPerStageCorrelation(t, 'chatterbox', gpuBtag, CHATTERBOX_STAGES, dumpDir, { hard: pinned })
  writeDumpWav(dumpDir, `chatterbox_${gpuBtag}.wav`, gpu.samples, 24000)
  writeDumpWav(dumpDir, 'chatterbox_CPU.wav', cpu.samples, 24000)
  const r = rms16(gpu.samples)
  console.log(`[Chatterbox/${gpuBtag}/rms] ${r.toFixed(5)}`)
  t.ok(r > 0.001, `Chatterbox/${gpuBtag}: GPU audio must be audible (rms ${r.toFixed(5)} > 0.001)`)
})
