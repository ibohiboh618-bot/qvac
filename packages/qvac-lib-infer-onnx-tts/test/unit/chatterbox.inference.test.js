'use strict'

const test = require('brittle')
const ONNXTTS = require('../../index.js')
const { TTSInterface } = require('../../tts.js')
const MockedBinding = require('../mock/MockedBinding.js')
const process = require('process')

global.process = process
const sinon = require('sinon')

function createMockedChatterboxModel ({ onOutput = () => { }, binding = undefined, exclusiveRun = false } = {}) {
  const args = {
    opts: { stats: true },
    exclusiveRun,
    tokenizerPath: './models/chatterbox/tokenizer.json',
    speechEncoderPath: './models/chatterbox/speech_encoder.onnx',
    embedTokensPath: './models/chatterbox/embed_tokens.onnx',
    conditionalDecoderPath: './models/chatterbox/conditional_decoder.onnx',
    languageModelPath: './models/chatterbox/language_model.onnx'
    // No loader - _downloadWeights will skip
  }
  const config = {
    language: 'en',
    useGPU: false
  }
  const model = new ONNXTTS(args, config)

  sinon.stub(model, '_createAddon').callsFake((configurationParams, outputCb) => {
    const _binding = binding || new MockedBinding()
    const addon = new TTSInterface(_binding, configurationParams, outputCb)

    if (_binding.setBaseInferenceCallback) {
      _binding.setBaseInferenceCallback(onOutput)
    }

    return addon
  })
  return model
}

async function waitWithTimeout (promise, timeoutMs, message) {
  let timeoutId
  const timeoutPromise = new Promise((resolve, reject) => {
    timeoutId = setTimeout(() => reject(new Error(message)), timeoutMs)
  })
  try {
    return await Promise.race([promise, timeoutPromise])
  } finally {
    clearTimeout(timeoutId)
  }
}

test('Chatterbox: run returns audio output and stats', async (t) => {
  const events = []
  const callbackArity = []
  const model = createMockedChatterboxModel({
    onOutput: function (addon, event, data, error) {
      callbackArity.push(arguments.length)
      events.push({ event, data, error })
    }
  })
  await model.load()

  const response = await model.run({ type: 'text', input: 'Hello world' })
  const outputs = []
  await response.onUpdate(data => outputs.push(data)).await()

  t.ok(outputs.length > 0, 'Response should emit at least one update')
  t.ok(outputs.some(d => d.outputArray), 'Response should contain outputArray payload')
  t.ok(response.stats.totalSamples > 0, 'Response stats should include total samples')
  t.ok(events.length > 0, 'Raw addon callback should have been called')
  t.ok(callbackArity.every(length => length === 4), 'Native callbacks should not include a native jobId argument')
  await model.unload()
})

test('Chatterbox: exclusiveRun does not deadlock run()', async (t) => {
  const model = createMockedChatterboxModel({ exclusiveRun: true })
  await model.load()

  const response = await waitWithTimeout(
    model.run({ type: 'text', input: 'Hello with exclusive run' }),
    1000,
    'run() timed out under exclusiveRun'
  )

  await waitWithTimeout(
    response.await(),
    1000,
    'response.await() timed out under exclusiveRun'
  )

  t.ok(response.stats.totalSamples > 0, 'Exclusive run should still produce runtime stats')
  await model.unload()
})

test('Chatterbox: Reload reloads configuration', async (t) => {
  const model = createMockedChatterboxModel()
  await model.load()

  const before = await model.run({ type: 'text', input: 'hello' })
  await before.await()

  await model.reload({ language: 'es' })
  const after = await model.run({ type: 'text', input: 'hola' })
  await after.await()

  t.ok(after.stats.audioDurationMs > 0, 'Reloaded model should still produce stats')
  await model.unload()
})

test('Chatterbox: exclusiveRun does not deadlock reload() or unload()', async (t) => {
  const model = createMockedChatterboxModel({ exclusiveRun: true })
  await model.load()

  await waitWithTimeout(
    model.reload({ language: 'es' }),
    1000,
    'reload() timed out under exclusiveRun'
  )

  const response = await waitWithTimeout(
    model.run({ type: 'text', input: 'after reload' }),
    1000,
    'run() after reload timed out under exclusiveRun'
  )
  await waitWithTimeout(
    response.await(),
    1000,
    'response.await() after reload timed out under exclusiveRun'
  )

  await waitWithTimeout(
    model.unload(),
    1000,
    'unload() timed out under exclusiveRun'
  )
  t.pass('exclusiveRun operations complete without deadlock')
})

test('Chatterbox: reload during in-flight job does not stay busy', async (t) => {
  const binding = new MockedBinding({ jobDelayMs: 100 })
  const model = createMockedChatterboxModel({ binding })
  await model.load()

  const inFlight = await model.run({ type: 'text', input: 'hello before reload' })
  await model.reload({ language: 'es' })

  let rejected = false
  try {
    await inFlight.await()
  } catch (error) {
    rejected = true
    t.ok(String(error.message).includes('reloaded'), 'In-flight job should fail on reload')
  }
  t.ok(rejected, 'Reload should reject the in-flight response')
  t.is(model._hasActiveResponse, false, 'Reload should clear active response busy flag')

  // Let stale callbacks from the destroyed addon drain before submitting a new job.
  await new Promise(resolve => setTimeout(resolve, 150))

  const afterReload = await model.run({ type: 'text', input: 'hello after reload' })
  await afterReload.await()
  t.ok(afterReload.stats.totalSamples > 0, 'Model should accept and complete jobs after reload')

  await model.unload()
})

test('Chatterbox: long text is split into chunks and all audio is collected', async (t) => {
  const chunkInputs = []
  const binding = new MockedBinding()
  const origRunJob = binding.runJob.bind(binding)
  binding.runJob = function (handle, data) {
    chunkInputs.push(data.input)
    return origRunJob(handle, data)
  }

  const model = createMockedChatterboxModel({ binding })
  await model.load()

  const longText = 'This is the first sentence. This is the second sentence. And here is a third one.'
  const response = await model.run({ type: 'text', input: longText })
  const outputs = []
  await response.onUpdate(data => outputs.push(data)).await()

  t.ok(chunkInputs.length >= 2, 'Long text should be split into multiple addon runJob calls (got ' + chunkInputs.length + ')')
  t.ok(outputs.length >= 2, 'Should receive output from each chunk')
  t.ok(outputs.every(d => d.outputArray), 'Every output should contain audio data')
  t.is(model._hasActiveResponse, false, 'Active response flag should be cleared after chunked run')
  await model.unload()
})

test('Chatterbox: short text is not chunked', async (t) => {
  const chunkInputs = []
  const binding = new MockedBinding()
  const origRunJob = binding.runJob.bind(binding)
  binding.runJob = function (handle, data) {
    chunkInputs.push(data.input)
    return origRunJob(handle, data)
  }

  const model = createMockedChatterboxModel({ binding })
  await model.load()

  const response = await model.run({ type: 'text', input: 'Hello world' })
  await response.onUpdate(() => {}).await()

  t.is(chunkInputs.length, 1, 'Short text should result in exactly one runJob call')
  t.is(chunkInputs[0], 'Hello world', 'Input should be passed unchanged')
  await model.unload()
})

test('Chatterbox: cancel during chunked processing stops remaining chunks', async (t) => {
  const binding = new MockedBinding({ jobDelayMs: 50 })
  const chunkInputs = []
  const origRunJob = binding.runJob.bind(binding)
  binding.runJob = function (handle, data) {
    chunkInputs.push(data.input)
    return origRunJob(handle, data)
  }

  const model = createMockedChatterboxModel({ binding })
  await model.load()

  const longText = 'First sentence here. Second sentence here. Third sentence here. Fourth sentence here.'
  const response = await model.run({ type: 'text', input: longText })

  await new Promise(resolve => setTimeout(resolve, 30))
  await response.cancel()

  let failed = false
  try {
    await response.await()
  } catch (error) {
    failed = true
    t.ok(String(error.message).includes('cancel'), 'Cancelled chunked response should reject with cancel message')
  }

  t.ok(failed, 'Cancelled chunked response should fail')
  await new Promise(resolve => setTimeout(resolve, 100))
  t.is(model._hasActiveResponse, false, 'Active response flag should be cleared after cancel')
  await model.unload()
})

test('Chatterbox: reload during chunked processing does not leave model busy', async (t) => {
  const binding = new MockedBinding({ jobDelayMs: 100 })
  const model = createMockedChatterboxModel({ binding })
  await model.load()

  const longText = 'First sentence here. Second sentence here. Third sentence here.'
  const inFlight = await model.run({ type: 'text', input: longText })

  await model.reload({ language: 'es' })

  let rejected = false
  try {
    await inFlight.await()
  } catch (error) {
    rejected = true
    t.ok(String(error.message).includes('reloaded'), 'In-flight chunked job should fail on reload')
  }
  t.ok(rejected, 'Reload should reject the in-flight chunked response')
  t.is(model._hasActiveResponse, false, 'Reload should clear active response busy flag')

  await new Promise(resolve => setTimeout(resolve, 150))

  const afterReload = await model.run({ type: 'text', input: 'hello after reload' })
  await afterReload.await()
  t.ok(afterReload.stats.totalSamples > 0, 'Model should accept and complete jobs after reload during chunked processing')

  await model.unload()
})

test('Chatterbox: Static methods return expected values', async (t) => {
  const modelKey = ONNXTTS.getModelKey({})
  t.is(modelKey, 'onnx-tts', 'getModelKey should return "onnx-tts"')
  t.ok(ONNXTTS.inferenceManagerConfig, 'inferenceManagerConfig should exist')
  t.is(ONNXTTS.inferenceManagerConfig.noAdditionalDownload, true, 'noAdditionalDownload should be true')
})

test('Chatterbox: Engine type is detected correctly', async (t) => {
  const chatterboxArgs = {
    tokenizerPath: './tokenizer.json',
    speechEncoderPath: './speech_encoder.onnx',
    embedTokensPath: './embed_tokens.onnx',
    conditionalDecoderPath: './conditional_decoder.onnx',
    languageModelPath: './language_model.onnx'
  }
  const chatterboxModel = new ONNXTTS(chatterboxArgs, {})
  t.is(chatterboxModel._engineType, 'chatterbox', 'Should detect Chatterbox engine when Chatterbox paths are provided')
})

test('Chatterbox: cancel propagates as job failure', async (t) => {
  const model = createMockedChatterboxModel()
  await model.load()

  const response = await model.run({ type: 'text', input: 'cancel me' })
  await response.cancel()

  let failed = false
  try {
    await response.await()
  } catch (error) {
    failed = true
    t.ok(String(error.message).includes('cancel'), 'Cancelled response should reject')
  }

  t.ok(failed, 'Cancelled response should fail')
  await model.unload()
})
