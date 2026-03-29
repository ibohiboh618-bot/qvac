'use strict'

const test = require('brittle')
const os = require('bare-os')
const proc = require('bare-process')
const binding = require('../../binding')
const ImgStableDiffusion = require('../../index')
const {
  ensureModel,
  setupJsLogger,
  isPng
} = require('./utils')

const isDarwinX64 = os.platform() === 'darwin' && os.arch() === 'x64'
const isLinuxArm64 = os.platform() === 'linux' && os.arch() === 'arm64'
const isMobile = os.platform() === 'ios' || os.platform() === 'android'
const noGpu = proc.env && proc.env.NO_GPU === 'true'
const useCpu = isDarwinX64 || isLinuxArm64 || noGpu
const skip = isMobile || noGpu

const DEFAULT_MODEL = {
  name: 'stable-diffusion-v2-1-Q8_0.gguf',
  url: 'https://huggingface.co/gpustack/stable-diffusion-v2-1-GGUF/resolve/main/stable-diffusion-v2-1-Q8_0.gguf'
}

const TEST_PARAMS = {
  prompt: 'a bright lighthouse on a stormy coast, cinematic, detailed',
  negative_prompt: 'blurry, low quality, watermark',
  steps: 12,
  width: 512,
  height: 512,
  cfg_scale: 7.5,
  seed: 42
}

function createModel (modelDir, modelName, logger) {
  return new ImgStableDiffusion(
    {
      logger,
      diskPath: modelDir,
      modelName
    },
    {
      threads: 4,
      device: useCpu ? 'cpu' : 'gpu',
      prediction: 'v'
    }
  )
}

function withTimeout (promise, ms, label) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`${label} timed out after ${ms}ms`))
    }, ms)

    promise.then(
      (value) => {
        clearTimeout(timer)
        resolve(value)
      },
      (error) => {
        clearTimeout(timer)
        reject(error)
      }
    )
  })
}

async function unloadModel (model, label, { suppressError = false } = {}) {
  try {
    console.log(`[${label}] unloading model`)
    await model.unload()
    console.log(`[${label}] unload complete`)
  } catch (error) {
    console.error(`[${label}] unload failed:`, error)
    if (!suppressError) throw error
  }
}

async function setupTwoModels (t, { loadSecond = true } = {}) {
  setupJsLogger(binding)

  const [downloadedModelName, modelDir] = await ensureModel({
    modelName: DEFAULT_MODEL.name,
    downloadUrl: DEFAULT_MODEL.url
  })

  console.log('\n=== Preparing two diffusion instances ===')
  console.log(`Model: ${downloadedModelName}`)
  console.log(`Dir  : ${modelDir}`)

  const model1 = createModel(modelDir, downloadedModelName, console)
  const model2 = createModel(modelDir, downloadedModelName, console)

  t.teardown(async () => {
    await unloadModel(model1, 'instance-1 teardown', { suppressError: true })
    await unloadModel(model2, 'instance-2 teardown', { suppressError: true })
    try {
      binding.releaseLogger()
    } catch (_) {}
  })

  if (loadSecond) {
    console.log('[setup] loading both model instances in parallel')
    await Promise.all([
      model1.load(),
      model2.load()
    ])
    console.log('[setup] both model instances loaded')
  } else {
    console.log('[setup] loading instance-1 only')
    await model1.load()
    console.log('[setup] instance-1 loaded')
  }

  return { model1, model2 }
}

async function startTrackedRun (model, label, params = TEST_PARAMS) {
  console.log(`[${label}] starting inference`)

  const response = await model.run(params)
  const images = []
  const progressTicks = []

  let resolveFirstProgress
  let rejectFirstProgress
  let sawFirstProgress = false

  const firstProgress = new Promise((resolve, reject) => {
    resolveFirstProgress = resolve
    rejectFirstProgress = reject
  })

  const completion = response
    .onUpdate((data) => {
      if (data instanceof Uint8Array) {
        images.push(data)
        console.log(`[${label}] received image chunk (${data.length} bytes)`)
        return
      }

      if (typeof data !== 'string') return

      try {
        const tick = JSON.parse(data)
        progressTicks.push(tick)

        if (!sawFirstProgress && typeof tick.step === 'number') {
          sawFirstProgress = true
          console.log(`[${label}] first progress tick: step ${tick.step}/${tick.total}`)
          resolveFirstProgress(tick)
        }
      } catch (error) {
        console.error(`[${label}] failed to parse progress update:`, error)
      }
    })
    .await()
    .then(() => {
      if (!sawFirstProgress) resolveFirstProgress(null)
      console.log(`[${label}] inference completed`)
      return { images, progressTicks }
    })
    .catch((error) => {
      rejectFirstProgress(error)
      console.error(`[${label}] inference failed:`, error)
      throw error
    })

  return { firstProgress, completion }
}

async function assertSuccessfulRun (t, runResult, label) {
  const { images, progressTicks } = await runResult.completion

  t.ok(progressTicks.length > 0, `${label} produced progress updates`)
  t.ok(images.length > 0, `${label} produced at least one image`)
  t.ok(isPng(images[images.length - 1]), `${label} produced a valid PNG`)
}

test('diffusion multi-instance - unload instance 2 while instance 1 is running', { timeout: 900000, skip }, async t => {
  const { model1, model2 } = await setupTwoModels(t)

  console.log('\n=== Scenario 1: unload instance 2 while instance 1 is running ===')

  try {
    const run1 = await startTrackedRun(model1, 'scenario-1 instance-1', { ...TEST_PARAMS, seed: 303 })
    await withTimeout(run1.firstProgress, 120000, 'scenario-1 first progress')

    console.log('[scenario-1] instance-1 is active, unloading instance-2')
    await unloadModel(model2, 'scenario-1 instance-2')

    await assertSuccessfulRun(t, run1, 'scenario-1 instance-1')
    t.pass('scenario-1 completed without interrupting the active run')
  } catch (error) {
    console.error('[scenario-1] error:', error)
    throw error
  }
})

test('diffusion multi-instance - load and run instance 2 while instance 1 is running', { timeout: 900000, skip }, async t => {
  const { model1, model2 } = await setupTwoModels(t, { loadSecond: false })

  console.log('\n=== Scenario 2: load and run instance 2 while instance 1 is running ===')

  try {
    const run1 = await startTrackedRun(model1, 'scenario-2 instance-1', { ...TEST_PARAMS, seed: 404 })
    await withTimeout(run1.firstProgress, 120000, 'scenario-2 first progress')

    console.log('[scenario-2] loading instance-2 while instance-1 is still running')
    await model2.load()

    console.log('[scenario-2] starting instance-2 while instance-1 is still running')
    const run2 = await startTrackedRun(model2, 'scenario-2 instance-2', { ...TEST_PARAMS, seed: 505 })

    await Promise.all([
      assertSuccessfulRun(t, run1, 'scenario-2 instance-1'),
      assertSuccessfulRun(t, run2, 'scenario-2 instance-2')
    ])
  } catch (error) {
    console.error('[scenario-2] error:', error)
    throw error
  }
})

setImmediate(() => {
  setTimeout(() => {}, 500)
})
