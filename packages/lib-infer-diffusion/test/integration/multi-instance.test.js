'use strict'

const test = require('brittle')
const os = require('bare-os')
const proc = require('bare-process')
const ImgStableDiffusion = require('../../index')
const {
  ensureModel,
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

const unloadedModels = new WeakSet()

function createModel (modelDir, modelName) {
  return new ImgStableDiffusion(
    {
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
  if (unloadedModels.has(model)) {
    console.log(`[${label}] unload skipped (already unloaded)`)
    return
  }

  try {
    console.log(`[${label}] unloading model`)
    await model.unload()
    unloadedModels.add(model)
    console.log(`[${label}] unload complete`)
  } catch (error) {
    console.error(`[${label}] unload failed:`, error)
    if (!suppressError) throw error
  }
}

async function setupModelEnvironment (t) {
  const [downloadedModelName, modelDir] = await ensureModel({
    modelName: DEFAULT_MODEL.name,
    downloadUrl: DEFAULT_MODEL.url
  })

  console.log('\n=== Preparing two diffusion instances ===')
  console.log(`Model: ${downloadedModelName}`)
  console.log(`Dir  : ${modelDir}`)

  return { downloadedModelName, modelDir }
}

async function setupTwoModels (t, { loadSecond = true } = {}) {
  const { downloadedModelName, modelDir } = await setupModelEnvironment(t)

  const model1 = createModel(modelDir, downloadedModelName)
  const model2 = createModel(modelDir, downloadedModelName)

  t.teardown(async () => {
    await unloadModel(model1, 'instance-1 teardown', { suppressError: true })
    await unloadModel(model2, 'instance-2 teardown', { suppressError: true })
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

test('diffusion multi-instance - two instances can run inference simultaneously', { timeout: 900000, skip }, async t => {
  const { model1, model2 } = await setupTwoModels(t)

  console.log('\n=== Scenario 1: two instances can run inference simultaneously ===')

  const run1 = await startTrackedRun(model1, 'scenario-1 instance-1', { ...TEST_PARAMS, seed: 101 })
  const run2 = await startTrackedRun(model2, 'scenario-1 instance-2', { ...TEST_PARAMS, seed: 202 })

  await Promise.all([
    assertSuccessfulRun(t, run1, 'scenario-1 instance-1'),
    assertSuccessfulRun(t, run2, 'scenario-1 instance-2')
  ])
})

test('diffusion multi-instance - repeated load/unload cycles should remain stable', { timeout: 900000, skip }, async t => {
  const { downloadedModelName, modelDir } = await setupModelEnvironment(t)
  const NUM_CYCLES = 3

  console.log('\n=== Scenario 2: repeated load/unload cycles should remain stable ===')

  for (let i = 0; i < NUM_CYCLES; i++) {
    const model = createModel(modelDir, downloadedModelName)
    await model.load()

    const run = await startTrackedRun(model, `scenario-2 cycle-${i + 1}`, { ...TEST_PARAMS, seed: 300 + i })
    await assertSuccessfulRun(t, run, `scenario-2 cycle-${i + 1}`)

    await unloadModel(model, `scenario-2 cycle-${i + 1}`)
    t.pass(`scenario-2 cycle ${i + 1}: load/unload completed`)
  }
})

test('diffusion multi-instance - unload instance 2 while instance 1 is running', { timeout: 900000, skip }, async t => {
  const { model1, model2 } = await setupTwoModels(t)

  console.log('\n=== Scenario 3: unload instance 2 while instance 1 is running ===')

  try {
    const run1 = await startTrackedRun(model1, 'scenario-3 instance-1', { ...TEST_PARAMS, seed: 303 })
    await withTimeout(run1.firstProgress, 120000, 'scenario-3 first progress')

    console.log('[scenario-3] instance-1 is active, unloading instance-2')
    await unloadModel(model2, 'scenario-3 instance-2')

    await assertSuccessfulRun(t, run1, 'scenario-3 instance-1')
    t.pass('scenario-3 completed without interrupting the active run')
  } catch (error) {
    console.error('[scenario-3] error:', error)
    throw error
  }
})

test('diffusion multi-instance - multiple load/unload cycles on one instance while another generates', { timeout: 900000, skip }, async t => {
  const { downloadedModelName, modelDir } = await setupModelEnvironment(t)
  const model1 = createModel(modelDir, downloadedModelName)
  const NUM_CYCLES = 3

  t.teardown(async () => {
    await unloadModel(model1, 'instance-1 teardown', { suppressError: true })
  })

  await model1.load()

  console.log('\n=== Scenario 4: multiple load/unload cycles on one instance while another generates ===')

  try {
    const run1 = await startTrackedRun(model1, 'scenario-4 instance-1', { ...TEST_PARAMS, steps: 50, seed: 404 })
    await withTimeout(run1.firstProgress, 120000, 'scenario-4 first progress')

    for (let i = 0; i < NUM_CYCLES; i++) {
      const model2 = createModel(modelDir, downloadedModelName)
      await model2.load()
      await unloadModel(model2, `scenario-4 cycle-${i + 1}`)
      t.pass(`scenario-4 cycle ${i + 1}: load/unload completed while instance-1 generates`)
    }

    await assertSuccessfulRun(t, run1, 'scenario-4 instance-1')
  } catch (error) {
    console.error('[scenario-4] error:', error)
    throw error
  }
})

setImmediate(() => {
  setTimeout(() => {}, 500)
})
