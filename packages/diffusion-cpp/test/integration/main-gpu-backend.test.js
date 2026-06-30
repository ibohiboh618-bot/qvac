'use strict'

const path = require('bare-path')
const os = require('bare-os')
const proc = require('bare-process')
const test = require('brittle')
const binding = require('../../binding')
const ImgStableDiffusion = require('../../index')
const { ensureModel, releaseJsLogger } = require('./utils')

const BACKENDS_DIR = path.resolve(__dirname, '../../prebuilds')
const MIN_TARGET_VRAM_BYTES = 1536 * 1024 * 1024

const MODEL = {
  name: 'stable-diffusion-v2-1-Q4_0.gguf',
  url: 'https://huggingface.co/gpustack/stable-diffusion-v2-1-GGUF/resolve/main/stable-diffusion-v2-1-Q4_0.gguf'
}

function backendDevices () {
  if (typeof binding.getBackendDevicesJson !== 'function') {
    return []
  }
  return JSON.parse(binding.getBackendDevicesJson(BACKENDS_DIR))
}

function pickNonDefaultVulkanGpu (devices) {
  return devices
    .filter((dev) =>
      (dev.type === 'GPU' || dev.type === 'IGPU') &&
      typeof dev.name === 'string' &&
      /^Vulkan\d+$/.test(dev.name) &&
      dev.gpuIndex > 0
    )
    .find((dev) => !dev.totalBytes || dev.totalBytes >= MIN_TARGET_VRAM_BYTES)
}

test('main-gpu selects a non-default Vulkan GPU when multiple GPUs are visible', { timeout: 600000 }, async (t) => {
  if (os.platform() !== 'linux' || os.arch() !== 'x64') {
    t.pass('main-gpu multi-GPU integration is linux-x64 only')
    return
  }
  if (proc.env && proc.env.NO_GPU === 'true') {
    t.pass('NO_GPU=true; skipping main-gpu multi-GPU integration')
    return
  }

  const logs = []
  binding.setLogger((priority, message) => {
    logs.push(String(message))
    console.log(`[C++ ${priority}] ${message}`)
  })

  let model = null
  try {
    const devices = backendDevices()
    console.log('GGML backend devices:', JSON.stringify(devices))

    const gpuDevices = devices.filter((dev) => dev.type === 'GPU' || dev.type === 'IGPU')
    if (gpuDevices.length < 2) {
      t.pass(`Only ${gpuDevices.length} GPU device(s) visible; skipping main-gpu selection assertion`)
      return
    }

    const target = pickNonDefaultVulkanGpu(devices)
    if (!target) {
      t.pass('No non-default Vulkan GPU with enough reported memory; skipping main-gpu selection assertion')
      return
    }

    const [modelName, modelDir] = await ensureModel({
      modelName: MODEL.name,
      downloadUrl: MODEL.url
    })

    model = new ImgStableDiffusion({
      files: {
        model: path.join(modelDir, modelName)
      },
      config: {
        device: 'gpu',
        'main-gpu': target.gpuIndex,
        threads: 4,
        prediction: 'v',
        diffusion_fa: true,
        verbosity: 2,
        backendsDir: BACKENDS_DIR
      },
      logger: console,
      opts: { stats: true }
    })

    await model.load()

    const resolvedLog = logs.find((line) =>
      line.includes(`main-gpu resolved to backend '${target.name}'`)
    )
    const directOrLegacyPinLog = logs.find((line) =>
      line.includes(`main-gpu pinning stable-diffusion backend '${target.name}'`) ||
      line.includes(`main-gpu using legacy SD_VK_DEVICE fallback for backend '${target.name}'`) ||
      line.includes(`Selecting ${target.name} as main device by env var SD_VK_DEVICE`)
    )

    t.ok(resolvedLog, `main-gpu resolved to ${target.name}`)
    t.ok(directOrLegacyPinLog, `stable-diffusion.cpp was asked to use ${target.name}`)
  } finally {
    if (model) await model.unload().catch(() => {})
    releaseJsLogger(binding)
  }
})
