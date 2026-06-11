'use strict'

// DO-NOT-MERGE device-farm GPU validation for @qvac/bci-whispercpp.
//
// On the AWS Device Farm dual-flagship run this asserts bci engages a real GPU
// backend and transcribes correctly on each device — naturally OpenCL on the
// Samsung S25 Ultra (Adreno) and Vulkan on the Pixel 9 (Mali-G715) — and it
// force-loads OpenCL even on the non-Adreno Mali GPU so the run REPORTS whether
// OpenCL actually engages there. Expected: on Mali, OpenCL is rejected at
// ggml_cl2_init (not an Adreno/Intel/AMD-64-wide device) and bci falls back to
// Vulkan; the rejection reason lands in logcat. Modelled on
// packages/tts-ggml/test/integration/gpu-smoke.test.js + bci's verify-gpu.js.
//
// backendDevice: 0=CPU, 1=GPU. backendId: 0=CPU,1=Metal,2=CUDA,3=Vulkan,
// 4=OpenCL,99=other (BCIModel.cpp backendIdFromRegName). Both surface via
// opts:{stats:true} -> response.stats.
//
// Env: QVAC_BCI_GPU_SMOKE_RELAX=1 downgrades the strict GPU gate to a warning
// (host without a real GPU); NO_GPU=true skips the GPU case entirely.

const fs = require('bare-fs')
const os = require('bare-os')

// OpenCL-on-Mali force-load attempt. ggml registers backends once per process
// inside the first bci.load(), reading these via getenv — so set them at module
// top, before any model loads. No-op on Adreno (OpenCL loads there anyway); on
// Mali this makes ggml attempt (and log) the OpenCL device init it would
// otherwise skip. Guarded in case the bare-os build predates setEnv.
const HAS_SET_ENV = typeof os.setEnv === 'function'
if (HAS_SET_ENV) {
  os.setEnv('GGML_OPENCL_FORCE_LOAD', '1')
  os.setEnv('GGML_OPENCL_ALLOW_UNKNOWN_GPU', '1')
}

const test = require('brittle')
const BCIWhispercpp = require('../../index')
const { getTestPaths, getModelPath, computeWER, detectPlatform } = require('./helpers')
const { flattenSegments } = require('@qvac/bci-whispercpp/util')

const PLAT = detectPlatform()
const platform = PLAT.platform
const { manifest, getSamplePath } = getTestPaths()

const MODEL_PATH = (os.hasEnv('WHISPER_MODEL_PATH') ? os.getEnv('WHISPER_MODEL_PATH') : null) ||
  getModelPath('ggml-bci-windowed.bin')
const hasModel = fs.existsSync(MODEL_PATH)
const hasSamples = manifest.samples && manifest.samples.length > 0

const RELAX = os.hasEnv('QVAC_BCI_GPU_SMOKE_RELAX') && os.getEnv('QVAC_BCI_GPU_SMOKE_RELAX') === '1'
const NO_GPU = os.hasEnv('NO_GPU') && os.getEnv('NO_GPU') === 'true'

// DO-NOT-MERGE validation: this exercises the AWS Device Farm Android GPUs
// (Adreno + Mali). Skip on desktop/iOS CI (GPU-less runners would fail the
// strict GPU assertion). Android device-farm devices always have a real GPU.
const NOT_ON_DEVICE = platform !== 'android'

function backendIdToName (id) {
  switch (id) {
    case 0: return 'CPU'
    case 1: return 'Metal'
    case 2: return 'CUDA'
    case 3: return 'Vulkan'
    case 4: return 'OpenCL'
    case 99: return 'other-GPU'
    default: return 'unknown(' + id + ')'
  }
}

function envVal (name) {
  return os.hasEnv(name) ? os.getEnv(name) : '(unset)'
}

// Platforms that wire up a GPU backend for bci today: android (opencl+vulkan),
// linux/win (vulkan), darwin/ios (metal).
function expectsGpu () {
  return platform === 'darwin' || platform === 'ios' ||
         platform === 'linux' || platform === 'win32' || platform === 'android'
}

function bciConfigFor (sample) {
  return typeof sample?.day_idx === 'number' ? { day_idx: sample.day_idx } : undefined
}

function makeBci (useGpu, sample, gpuDevice) {
  return new BCIWhispercpp({
    files: { model: MODEL_PATH },
    opts: { stats: true }
  }, {
    contextParams: { use_gpu: useGpu, gpu_device: typeof gpuDevice === 'number' ? gpuDevice : 0 },
    whisperConfig: { language: 'en', temperature: 0.0 },
    miscConfig: { caption_enabled: false },
    bciConfig: bciConfigFor(sample)
  })
}

function logEnv (t) {
  t.comment('[BCI/setup] platform=' + PLAT.label + ' os.setEnv=' + HAS_SET_ENV)
  t.comment('[BCI/setup] GGML_OPENCL_FORCE_LOAD=' + envVal('GGML_OPENCL_FORCE_LOAD') +
    ' GGML_OPENCL_ALLOW_UNKNOWN_GPU=' + envVal('GGML_OPENCL_ALLOW_UNKNOWN_GPU'))
}

function logBackend (t, tag, stats, text, wer) {
  t.comment(tag + ' backendDevice=' + stats.backendDevice + ' backendId=' + stats.backendId +
    ' (' + backendIdToName(stats.backendId) + ')')
  t.comment(tag + ' gpuMemTotalMb=' + stats.gpuMemTotalMb + ' gpuMemFreeMb=' + stats.gpuMemFreeMb)
  t.comment(tag + ' WER=' + (wer * 100).toFixed(1) + '%  text="' + text + '"')
  t.comment(tag + ' stats=' + JSON.stringify(stats))
}

test('[BCI] GPU smoke - use_gpu=true must engage a GPU backend and transcribe', {
  timeout: 600000,
  skip: !hasModel || !hasSamples || NO_GPU || NOT_ON_DEVICE
}, async (t) => {
  logEnv(t)
  const sample = manifest.samples[0]
  const samplePath = getSamplePath(sample.file)
  t.ok(fs.existsSync(samplePath), 'Fixture ' + sample.file + ' must exist')

  const bci = makeBci(true, sample)
  try {
    await bci.load()
    const response = await bci.transcribeFile(samplePath)
    const output = await response.await()
    const text = flattenSegments(output).map(s => s.text).join('').trim()
    const stats = response.stats || {}
    const dev = stats.backendDevice
    const id = stats.backendId
    const name = backendIdToName(id)
    const wer = computeWER(text, sample.expected_text)

    logBackend(t, '[BCI/GPU]', stats, text, wer)
    // The OpenCL-on-Mali answer: OpenCL(4) = engaged (expected on Adreno);
    // anything else on android = OpenCL not engaged (expected on Mali, see logcat).
    if (platform === 'android') {
      t.comment(id === 4
        ? '[BCI/OpenCL] OpenCL ENGAGED on this device (backendId=4)'
        : '[BCI/OpenCL] OpenCL NOT engaged on this device -> ' + name + ' (force-load attempted; see logcat for the ggml_cl2_init reason)')
    }

    t.ok(typeof text === 'string' && text.length > 0, '[BCI/GPU] produced a transcription')
    t.ok(wer < 0.5, '[BCI/GPU] WER below 50% (got ' + (wer * 100).toFixed(1) + '%)')

    if (!expectsGpu()) {
      t.is(dev, 0, '[BCI/' + platform + '] no GPU wired in -> backendDevice must be 0 (CPU)')
      return
    }
    if (dev !== 1) {
      const msg = '[BCI/' + platform + '] expected a GPU backend, got ' + name +
        ' (backendDevice=' + dev + '). use_gpu=true was requested but it fell back to CPU.'
      if (RELAX) {
        t.comment('WARNING (relaxed): ' + msg)
        t.pass('[BCI/GPU] smoke completed (relaxed)')
      } else {
        t.fail(msg)
      }
      return
    }
    if (platform === 'android') {
      t.ok(id === 3 || id === 4, '[BCI/android] expected Vulkan(3) or OpenCL(4), got ' + name)
    } else if (platform === 'darwin' || platform === 'ios') {
      t.is(id, 1, '[BCI/' + platform + '] expected Metal(1), got ' + name)
    } else {
      t.is(id, 3, '[BCI/' + platform + '] expected Vulkan(3), got ' + name)
    }
  } finally {
    try { await bci.destroy() } catch (_e) {}
  }
})

// Report-only: load with each GPU device index and log the backend that
// resolves. Reveals which GPU backends are actually REGISTERED on this device
// (the JS-visible signal for "is OpenCL even loaded on Mali"). On Mali with
// OpenCL rejected at init, only the Vulkan device exists, so higher indices
// fall back to Vulkan/CPU. Never fails — diagnostics only.
test('[BCI] GPU device enumeration (report-only) - which GPU backends are registered', {
  timeout: 900000,
  skip: !hasModel || !hasSamples || NO_GPU || NOT_ON_DEVICE
}, async (t) => {
  logEnv(t)
  const sample = manifest.samples[0]
  const samplePath = getSamplePath(sample.file)
  for (let idx = 0; idx <= 2; idx++) {
    const bci = makeBci(true, sample, idx)
    try {
      await bci.load()
      const response = await bci.transcribeFile(samplePath)
      const output = await response.await()
      const text = flattenSegments(output).map(s => s.text).join('').trim()
      const stats = response.stats || {}
      t.comment('[BCI/enum] gpu_device=' + idx + ' -> backendDevice=' + stats.backendDevice +
        ' backendId=' + stats.backendId + ' (' + backendIdToName(stats.backendId) + ')' +
        ' gpuMemTotalMb=' + stats.gpuMemTotalMb + ' textLen=' + text.length)
    } catch (e) {
      t.comment('[BCI/enum] gpu_device=' + idx + ' -> error: ' + (e && e.message ? e.message : e))
    } finally {
      try { await bci.destroy() } catch (_e) {}
    }
  }
  t.pass('[BCI/enum] enumeration complete (per-index backendId in the comments above)')
})

test('[BCI] CPU smoke - use_gpu=false must run on the CPU backend', {
  timeout: 600000,
  skip: !hasModel || !hasSamples || NOT_ON_DEVICE
}, async (t) => {
  const sample = manifest.samples[0]
  const samplePath = getSamplePath(sample.file)

  const bci = makeBci(false, sample)
  try {
    await bci.load()
    const response = await bci.transcribeFile(samplePath)
    const output = await response.await()
    const text = flattenSegments(output).map(s => s.text).join('').trim()
    const stats = response.stats || {}
    t.comment('[BCI/CPU] backendDevice=' + stats.backendDevice + ' backendId=' + stats.backendId)
    t.ok(typeof text === 'string' && text.length > 0, '[BCI/CPU] produced a transcription')
    t.is(stats.backendDevice, 0, '[BCI/CPU] use_gpu=false must resolve to backendDevice=0 (CPU)')
    t.is(stats.backendId, 0, '[BCI/CPU] use_gpu=false must resolve to backendId=0 (CPU)')
  } finally {
    try { await bci.destroy() } catch (_e) {}
  }
})
