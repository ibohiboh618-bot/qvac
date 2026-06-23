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
// DEBUG (QVAC-20557 Mali GPU correctness diagnostic, DO-NOT-MERGE).
const { assertSampleCorrelation, compareSamples } = require('../utils/correlation-helper')

const platform = os.platform()
const isMobile = platform === 'ios' || platform === 'android'
const RELAX = proc.env && proc.env.QVAC_TTS_GPU_SMOKE_RELAX === '1'
const NO_GPU = proc.env && proc.env.NO_GPU === 'true'

// DEBUG (QVAC-20557 Mali GPU correctness diagnostic, DO-NOT-MERGE): enable the
// native per-stage [gpu-diag] trace (rms/nan/inf/min/max, on BOTH GPU and CPU,
// tagged by backend) so a device-farm round can diff per stage in logcat_full.txt
// and localize the first stage a GPU backend miscomputes. The addon also injects
// a __android_log_print diag_sink unconditionally; this env enables the desktop
// stderr fallback for local pre-flight. Must be set before any Engine construction
// (native getenv runs at load()).
if (proc.env) proc.env.TTS_CPP_GPU_TRACE = '1'

// GPU-vs-CPU correlation threshold. GPU and CPU load the SAME gguf — only the
// compute backend differs — so a correct backend correlates ~0.999+.
const CORR_THRESHOLD = 0.99

// RMS of an int16 PCM array in [-1,1] units. A NaN/garbage GPU collapses audio to
// silence (rms ~0) once int16-clamped, so an audible-RMS floor is a backend-
// independent correctness gate for the stochastic Chatterbox engine (where a
// GPU-vs-CPU sample correlation is not valid — T3 samples different tokens).
function rms16 (samples) {
  let s = 0
  for (let i = 0; i < samples.length; i++) { const v = samples[i] / 32768; s += v * v }
  return samples.length ? Math.sqrt(s / samples.length) : 0
}

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

function assertGpuBackend (t, engineTag, stats, allowPolicyCpu = false) {
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

  // Engines tts-cpp declines on a given vendor (e.g. Chatterbox on Mali,
  // allow_arm_mali=false) legitimately fall back to CPU and flag it via
  // stats.gpuUnsupported. That is the correct result there, not a GPU regression.
  if (allowPolicyCpu && dev === 0 && stats.gpuUnsupported) {
    t.pass(`${engineTag}/${platform}: GPU present but declined by policy (gpuUnsupported=1); correctly using CPU`)
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
    assertGpuBackend(t, 'Chatterbox', result.data.stats, /* allowPolicyCpu */ true)
    recordSmoke(t, 'chatterbox gpu-smoke', result, wallMs)
  } finally {
    try { await model.unload() } catch (_e) {}
  }
})

test('Supertonic GPU smoke - useGPU=true must engage the GPU backend on GPU-capable platforms', { timeout: 600000, skip: NO_GPU }, async (t) => {
  // Supertonic GPU: Metal on Apple, Vulkan/CUDA on desktop, Vulkan/OpenCL on
  // Android (Adreno/Xclipse/Mali, validated under QVAC-20557 / tts-cpp 2026-06-18).
  // The strict assertion runs on every GPU-capable platform including Android.
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
// GPU-vs-CPU CORRECTNESS gate (QVAC-20557 Mali diagnostic, DO-NOT-MERGE).
//
// The smoke tests above only prove the GPU is *engaged*. These prove its OUTPUT
// is correct — the signal that was missing when a GPU could mis-compute yet still
// emit non-silent audio (the Mali Valhall mul_mat bug). No token-pinning, no f32
// dumps: per-stage localization comes from the native [gpu-diag] trace in
// logcat_full.txt (diff GPU vs CPU rms/nan/min/max per stage); these tests carry
// the hard PASS/FAIL signal.
//
//   - Supertonic is DETERMINISTIC (fixed seed) → GPU-vs-CPU end-to-end audio
//     Pearson is a HARD gate (corr >= 0.99).
//   - Chatterbox's T3 is STOCHASTIC → GPU and CPU sample different tokens, so a
//     GPU-vs-CPU correlation is NOT valid. Its hard gate is audio SANITY (GPU
//     output finite + audible); a NaN/garbage GPU collapses to silence and fails
//     it. The CPU run + corr are logged as INFORMATIONAL only.
//
// ADRENO (S25) is the validity control: both engines MUST pass here (Adreno is a
// validated vendor). If Adreno fails, the harness is wrong, not the device.
// ---------------------------------------------------------------------------

const CORR_TEXT = 'GPU versus CPU correctness check.'

async function runSupertonicOn (useGPU, supertonicPath) {
  const model = await loadSupertonicTTS({
    supertonicModelPath: supertonicPath, language: 'en', voice: 'F1', useGPU, seed: 42
  })
  try {
    const r = await runSupertonicTTS(model, { text: CORR_TEXT }, { minSamples: 5000 })
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
  const gpuBtag = backendIdToName(gpu.stats.backendId)
  // Deterministic → HARD gate.
  assertSampleCorrelation(t, `Supertonic/${gpuBtag}`, gpu.samples, cpu.samples, {
    threshold: CORR_THRESHOLD, minSamples: 1000, minLenRatio: 0.90
  })
  const rg = rms16(gpu.samples)
  console.log(`[Supertonic/${gpuBtag}/rms] gpu=${rg.toFixed(5)} cpu=${rms16(cpu.samples).toFixed(5)}`)
  t.ok(rg > 0.001, `Supertonic/${gpuBtag}: GPU audio must be audible (rms ${rg.toFixed(5)} > 0.001)`)
})

async function runChatterboxOn (useGPU, modelDir, refWavPath) {
  const model = await loadChatterboxTTS({ modelDir, refWavPath, language: 'en', useGPU, seed: 42 })
  try {
    const r = await runChatterboxTTS(model, { text: CORR_TEXT }, { minSamples: 5000 })
    return { samples: r.data.samples, stats: r.data.stats }
  } finally {
    try { await model.unload() } catch (_e) {}
  }
}

test('Chatterbox GPU correctness - GPU audio must be finite + audible (per-stage in [gpu-diag])', { timeout: 600000, skip: NO_GPU }, async (t) => {
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
  const gpuBtag = backendIdToName(gpu.stats.backendId)

  // HARD gate: GPU audio finite + audible. A Mali NaN cascade (the f0/stft NaN
  // seen on Mali) collapses the int16 audio to silence and fails this.
  const rg = rms16(gpu.samples)
  // INFORMATIONAL: a CPU reference run (T3 stochastic → corr is NOT a valid gate;
  // logged only so the by-ear RMS magnitudes can be compared).
  const cpu = await runChatterboxOn(false, download.targetDir, refWavPath)
  const rc = rms16(cpu.samples)
  console.log(`[Chatterbox/${gpuBtag}/rms] gpu=${rg.toFixed(5)} cpu=${rc.toFixed(5)} (gpu samples=${gpu.samples.length} cpu samples=${cpu.samples.length})`)
  console.log(`[Chatterbox/${gpuBtag}/note] T3 is stochastic; GPU-vs-CPU sample correlation is not a valid gate — localize via the per-stage [gpu-diag] trace in logcat_full.txt`)
  t.ok(rg > 0.001, `Chatterbox/${gpuBtag}: GPU audio must be finite + audible (rms ${rg.toFixed(5)} > 0.001); silence/NaN = Mali miscompute`)
})

// DEBUG (QVAC-20557 round-C verify, DO-NOT-MERGE): a DUPLICATE Supertonic
// GPU-vs-CPU correctness test placed LAST in the file. On the device farm the
// Supertonic GPU load triggers the native vk_mulmat_selftest oracle + the
// GGML_VK_DISABLE_COOPMAT env line + the mulmat_needs_pad line, and the
// correlation helper logs the corr value. In round C those landed mid-run and
// were EVICTED from the logcat ring buffer by the long Chatterbox correctness
// run that follows the original (mid-bundle) copy. This copy runs LAST so all of
// that coopmat-off proof lands in the TAIL of the buffer and survives. Reuses
// runSupertonicOn / assertSampleCorrelation; identical assertion to the copy above.
test('Supertonic GPU-vs-CPU correctness (CAPTURE-LAST) - coopmat-off verify', { timeout: 600000, skip: NO_GPU }, async (t) => {
  if (!expectsGpu()) { t.pass('Supertonic capture-last: no GPU wired on this platform'); return }
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
    const msg = `[MALI-VERIFY] Supertonic capture-last: GPU did not engage (backendDevice=${gpu.stats && gpu.stats.backendDevice})`
    if (RELAX) { t.pass(msg + ' (relaxed)') } else { t.comment(msg) }
    return
  }
  const cpu = await runSupertonicOn(false, supertonicPath)
  const gpuBtag = backendIdToName(gpu.stats.backendId)
  // Deterministic → HARD gate (same as the mid-bundle copy).
  assertSampleCorrelation(t, `Supertonic/${gpuBtag}/capture-last`, gpu.samples, cpu.samples, {
    threshold: CORR_THRESHOLD, minSamples: 1000, minLenRatio: 0.90
  })
  const rg = rms16(gpu.samples)
  console.log(`[MALI-VERIFY] Supertonic/${gpuBtag} gpu_rms=${rg.toFixed(5)} cpu_rms=${rms16(cpu.samples).toFixed(5)} backendId=${gpu.stats.backendId} gpuUnsupported=${gpu.stats.gpuUnsupported}`)
  t.ok(rg > 0.001, `Supertonic/${gpuBtag} capture-last: GPU audio audible (rms ${rg.toFixed(5)} > 0.001)`)
})

// DEBUG (QVAC-20557 round-L, DO-NOT-MERGE): DETERMINISTIC Chatterbox S3Gen
// GPU-vs-CPU correctness. Chatterbox's T3 is stochastic so an end-to-end GPU-vs-CPU
// audio correlation is invalid (GPU and CPU sample different tokens). This test pins
// T3 to CPU (TTS_CPP_T3_FORCE_CPU=1 -> native init_backend returns CPU for T3 only;
// S3Gen still follows useGPU via its own backend) so BOTH runs feed S3Gen the SAME
// deterministic tokens (seed=42, mt19937). The S3Gen Vulkan-vs-CPU output is then a
// valid signal, and the native per-stage `[gpu-diag] xcorr.s3gen.*` trace in
// logcat_full.txt localizes the first S3Gen stage that decorrelates on Mali (round K:
// hift_wav ~5x quiet). Informational corr here (the authoritative GPU-engage proof +
// per-stage localization is the `Vulkan.s3gen.*` btag + xcorr lines in logcat); the
// hard gate + Mali-GPU ship decision follow once localization is in hand.
test('Chatterbox S3Gen GPU-vs-CPU correctness (T3->CPU, deterministic tokens)', { timeout: 600000, skip: NO_GPU }, async (t) => {
  if (!expectsGpu()) { t.pass('Chatterbox S3Gen corr: no GPU wired on this platform'); return }
  const baseDir = getBaseDir()
  const modelsDir = path.join(baseDir, 'models')
  const download = await ensureChatterboxModels({ targetDir: modelsDir })
  if (!download.success) {
    t.fail('Chatterbox GGUFs not available - registry fetch failed.')
    return
  }
  const refWavPath = resolveRefWavPath({})
  if (!fs.existsSync(refWavPath)) { t.pass('Skipped: reference audio missing'); return }

  const hadEnv = proc.env.TTS_CPP_T3_FORCE_CPU
  if (proc.env) proc.env.TTS_CPP_T3_FORCE_CPU = '1'
  try {
    const gpu = await runChatterboxOn(true, download.targetDir, refWavPath)
    const cpu = await runChatterboxOn(false, download.targetDir, refWavPath)
    const gpuBtag = backendIdToName(gpu.stats && gpu.stats.backendId)
    const rg = rms16(gpu.samples)
    const rc = rms16(cpu.samples)
    // backendDevice here reflects T3 (pinned to CPU); S3Gen's backend is confirmed by
    // the `Vulkan.s3gen.*` vs `CPU.s3gen.*` btag in the per-stage logcat trace.
    const res = compareSamples(gpu.samples, cpu.samples, { minSamples: 1000, minLenRatio: 0.90 })
    const corrStr = Number.isFinite(res.corr) ? res.corr.toFixed(6) : String(res.corr)
    console.log(`[Chatterbox-S3Gen/corr] gpu_n=${gpu.samples.length} cpu_n=${cpu.samples.length} ` +
      `aligned_n=${res.n} lenRatio=${(res.lenRatio || 0).toFixed(3)} corr=${corrStr} ` +
      `gpu_rms=${rg.toFixed(5)} cpu_rms=${rc.toFixed(5)} t3BackendId=${gpu.stats && gpu.stats.backendId}` +
      `${res.reason ? ' reason=' + res.reason : ''}`)
    console.log('[Chatterbox-S3Gen/note] T3 pinned to CPU; S3Gen backend = the Vulkan.s3gen.*/CPU.s3gen.* btag in ' +
      'logcat. Read xcorr.s3gen.* to localize the first diverging S3Gen stage (informational corr; hard gate follows).')
    t.ok(rg > 0.001, `Chatterbox S3Gen: GPU audio finite + audible (rms ${rg.toFixed(5)} > 0.001)`)
  } finally {
    if (proc.env) {
      if (hadEnv === undefined) delete proc.env.TTS_CPP_T3_FORCE_CPU
      else proc.env.TTS_CPP_T3_FORCE_CPU = hadEnv
    }
  }
})

// DEBUG (QVAC-20557 T3-on-Mali, DO-NOT-MERGE): TEACHER-FORCED T3 logits validation.
// T3 is stochastic (samples speech tokens) so GPU-vs-CPU token sequences diverge and a
// per-token correlation is impossible. The native engine supports a teacher-force mode
// (TTS_CPP_T3_TEACHER, read fresh per run): a CPU "record" run captures the fed token
// sequence + each step's conditional logits; a GPU "replay" run feeds the SAME tokens and
// Pearson-correlates its per-step logits against the recorded CPU logits, isolating T3's
// matmul correctness from the sampling cascade. The native emits ONE eviction-safe summary
// line `[gpu-diag] t3 logits xcorr: steps=.. min_corr=.. first_below_0.99=..` (replay run)
// plus `[gpu-diag] t3_audio.full rms/zcr/samples` for BOTH runs (end-to-end audio shape).
// Read those from logcat_full.txt: min_corr≈1.0 => T3 computes correctly on Mali (ship full
// Chatterbox on GPU); a drop / first_below_0.99>=0 => T3 miscomputes (keep T3->CPU hybrid).
// NOT TTS_CPP_T3_FORCE_CPU here: T3 runs on the ACTUAL backend (CPU on the cpu run, Mali GPU
// on the gpu run) so this is full Chatterbox (T3+S3Gen) on the Mali GPU. Single-segment text
// (CORR_TEXT, one sentence) is required: the record/replay buffer is per-run_t3.
test('Chatterbox T3 GPU-vs-CPU logits (teacher-forced, full Chatterbox on GPU)', { timeout: 600000, skip: NO_GPU }, async (t) => {
  if (!expectsGpu()) { t.pass('Chatterbox T3 teacher-force: no GPU wired on this platform'); return }
  const baseDir = getBaseDir()
  const modelsDir = path.join(baseDir, 'models')
  const download = await ensureChatterboxModels({ targetDir: modelsDir })
  if (!download.success) {
    t.fail('Chatterbox GGUFs not available - registry fetch failed.')
    return
  }
  const refWavPath = resolveRefWavPath({})
  if (!fs.existsSync(refWavPath)) { t.pass('Skipped: reference audio missing'); return }

  const hadEnv = proc.env.TTS_CPP_T3_TEACHER
  try {
    // Run 1 (CPU, record): captures the fed token sequence + per-step logits (process-global).
    if (proc.env) proc.env.TTS_CPP_T3_TEACHER = 'record'
    const cpu = await runChatterboxOn(false, download.targetDir, refWavPath)
    // Run 2 (GPU, replay): teacher-forces the recorded tokens, xcorr per-step logits, emits
    // the summary + audio characteristics. Full Chatterbox (T3+S3Gen) on the Mali GPU.
    if (proc.env) proc.env.TTS_CPP_T3_TEACHER = 'replay'
    const gpu = await runChatterboxOn(true, download.targetDir, refWavPath)

    const gpuBtag = backendIdToName(gpu.stats && gpu.stats.backendId)
    const rg = rms16(gpu.samples)
    const rc = rms16(cpu.samples)
    console.log(`[Chatterbox-T3/teacher] gpu_btag=${gpuBtag} gpu_backendDevice=${gpu.stats && gpu.stats.backendDevice} ` +
      `gpu_rms=${rg.toFixed(5)} cpu_rms=${rc.toFixed(5)} gpu_n=${gpu.samples.length} cpu_n=${cpu.samples.length}`)
    console.log('[Chatterbox-T3/note] read `[gpu-diag] t3 logits xcorr: ... min_corr=..` (GPU replay run) in ' +
      'logcat_full.txt: min_corr≈1.0 => T3 matmuls correct on Mali; a drop => T3 miscomputes. ' +
      '`[gpu-diag] t3_audio.full` (both runs) = end-to-end audio shape (rms/zcr).')
    // HARD gate: full-Chatterbox GPU audio finite + audible (a T3 garbage cascade collapses
    // to silence/NaN). The numeric T3-correctness gate is the logcat min_corr.
    t.ok(rg > 0.001, `Chatterbox T3 teacher-force: full-GPU audio finite + audible (rms ${rg.toFixed(5)} > 0.001)`)
  } finally {
    if (proc.env) {
      if (hadEnv === undefined) delete proc.env.TTS_CPP_T3_TEACHER
      else proc.env.TTS_CPP_T3_TEACHER = hadEnv
    }
  }
})
