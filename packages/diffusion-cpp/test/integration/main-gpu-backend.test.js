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

function isVulkanGpu (dev) {
  return (
    (dev.type === 'GPU' || dev.type === 'IGPU') &&
    typeof dev.name === 'string' &&
    /^Vulkan\d+$/.test(dev.name)
  )
}

function looksIntegrated (dev) {
  const label = `${dev.description || ''} ${dev.name || ''}`.toLowerCase()
  return (
    label.includes('intel') ||
    label.includes('uhd') ||
    label.includes('iris') ||
    label.includes('integrated')
  )
}

function hasEnoughMemory (dev) {
  return !dev.totalBytes || dev.totalBytes >= MIN_TARGET_VRAM_BYTES
}

function pickMainGpuTarget (devices) {
  const vulkanGpus = devices.filter(isVulkanGpu)
  if (os.platform() === 'win32') {
    const integrated = vulkanGpus.find((dev) =>
      dev.gpuIndex >= 0 && looksIntegrated(dev) && hasEnoughMemory(dev)
    )
    if (integrated) return integrated
  }

  return vulkanGpus.find((dev) => dev.gpuIndex > 0 && hasEnoughMemory(dev))
}

test('main-gpu pins an explicit Vulkan GPU when multiple GPUs are visible', { timeout: 600000 }, async (t) => {
  const isSupportedDesktop = (os.platform() === 'linux' || os.platform() === 'win32') && os.arch() === 'x64'
  if (!isSupportedDesktop) {
    t.pass('main-gpu multi-GPU integration is desktop x64 only')
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

    const target = pickMainGpuTarget(devices)
    if (!target) {
      t.pass('No suitable Vulkan GPU target with enough reported memory; skipping main-gpu selection assertion')
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
