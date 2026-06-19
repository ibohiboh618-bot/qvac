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
const { assertSampleCorrelation } = require('../utils/correlation-helper')

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
  assertSampleCorrelation(t, `Supertonic/${backendIdToName(gpu.stats.backendId)}`, gpu.samples, cpu.samples, {
    threshold: CORR_THRESHOLD, minSamples: 1000, minLenRatio: 0.90
  })
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

test('Chatterbox GPU-vs-CPU correctness - GPU output must be finite (corr informational)', { timeout: 600000, skip: NO_GPU }, async (t) => {
  if (!expectsGpu()) { t.pass('Chatterbox corr: no GPU wired on this platform'); return }
  const baseDir = getBaseDir()
  const modelsDir = path.join(baseDir, 'models')
  const download = await ensureChatterboxModels({ targetDir: modelsDir })
  if (!download.success) {
    t.fail('Chatterbox GGUFs not available - registry fetch failed.')
    return
  }
  const refWavPath = resolveRefWavPath({})
  if (!fs.existsSync(refWavPath)) { t.pass('Skipped: reference audio missing'); return }

  const gpu = await runChatterboxOn(true, download.targetDir, refWavPath)
  if (!gpu.stats || gpu.stats.backendDevice !== 1) {
    t.comment(`Chatterbox corr: GPU did not engage (backendDevice=${gpu.stats && gpu.stats.backendDevice}); per-stage [gpu-diag] trace still emitted`)
    return
  }
  const cpu = await runChatterboxOn(false, download.targetDir, refWavPath)
  // soft: T3 stochasticity makes the magnitude informational; the hard signal
  // here is finite-ness (silent/NaN GPU fails) + the per-stage [gpu-diag] trace.
  assertSampleCorrelation(t, `Chatterbox/${backendIdToName(gpu.stats.backendId)}`, gpu.samples, cpu.samples, {
    threshold: CORR_THRESHOLD, soft: true, minSamples: 1000, minLenRatio: 0.0
  })
})
