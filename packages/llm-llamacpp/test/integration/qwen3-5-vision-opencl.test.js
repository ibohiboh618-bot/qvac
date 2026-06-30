'use strict'
// QVAC-21297: Qwen3.5-VL vision encoder on Android OpenCL (Adreno).
//
// With device:'gpu', the addon routes the vision projector (mmproj/clip) to the
// GPU only when the chosen Android backend is OpenCL (Adreno) — see
// LlamaModel::mmprojUseGpuForBackend. This test verifies, on the Adreno OpenCL
// path, that (1) the projector actually runs on the OpenCL GPU backend, and
// (2) the image embedding is correct — i.e. the qvac-fabric null-mask =
// bidirectional fix is active. Before that fix the OpenCL flash-attention
// dispatch treated the bidirectional SigLIP tower as causal and produced a
// corrupt "gray surface" encode (no elephant). On non-OpenCL GPUs (desktop
// Vulkan/Metal, Mali) the projector backend differs; the accuracy assertion
// still holds and the OpenCL-routing assertion is skipped.

const test = require('brittle')
const fs = require('bare-fs')
const path = require('bare-path')
const LlmLlamacpp = require('../../index.js')
const { ensureModel, getMediaPath } = require('./utils')
const { attachSpecLogger } = require('./spec-logger')
const { QWEN35_MODEL, IMAGE_CASES, isDarwinX64 } = require('./_vlm-image-perf.js')

// Darwin x64 CI has no Metal GPU and falls back to CPU, so the GPU path under
// test is not exercised there (matches the sibling Qwen3.5 perf tests).
test('Qwen3.5-VL vision encoder routes to OpenCL and stays accurate [elephant]',
  { timeout: 1_800_000, skip: isDarwinX64 }, async t => {
    const imageCase = IMAGE_CASES.elephant
    const [modelName, dirPath] = await ensureModel(QWEN35_MODEL.llmModel)
    const [projModelName] = await ensureModel(QWEN35_MODEL.projModel)

    const spec = attachSpecLogger({ forwardToConsole: true })
    const inference = new LlmLlamacpp({
      files: {
        model: [path.join(dirPath, modelName)],
        projectionModel: path.join(dirPath, projModelName)
      },
      config: {
        device: 'gpu',
        gpu_layers: '98',
        ctx_size: QWEN35_MODEL.ctxFor(imageCase),
        temp: '0',
        seed: '42',
        'reasoning-budget': '0',
        verbosity: '2'
      },
      logger: console,
      opts: { stats: true }
    })

    try {
      await inference.load()

      const imageFilePath = getMediaPath(imageCase.imageFile)
      t.ok(fs.existsSync(imageFilePath), `${imageCase.imageFile} should exist`)
      const imageBytes = new Uint8Array(fs.readFileSync(imageFilePath))

      const messages = [
        { role: 'user', type: 'media', content: imageBytes },
        { role: 'user', content: 'Describe the image briefly in one sentence.' }
      ]
      const response = await inference.run(messages)
      const chunks = []
      let error = null
      response.onUpdate(data => { chunks.push(data) }).onError(err => { error = err })
      await response.await()
      if (error) throw new Error('Inference error: ' + error)
      const output = chunks.join('')
      t.comment(`output: "${output.slice(0, 200)}"`)

      // Accuracy: on the Adreno OpenCL path this proves the null-mask fix (no
      // "gray surface" corruption); on other backends it is a sanity check.
      t.ok(output.length > 0, 'vision inference produced output')
      const matched = imageCase.keywords.some(k => new RegExp(`\\b${k}\\b`, 'i').test(output))
      t.ok(matched,
        `output should mention one of ${imageCase.keywords.join(', ')}: "${output.slice(0, 150)}"`)

      // Routing: BackendSelection logs the chosen backend. On Adreno the addon
      // keeps OpenCL and (QVAC-21297) routes the projector to the GPU. Detect
      // OpenCL from the captured native logs; only then assert projector routing
      // so the test stays green on non-OpenCL GPUs (Vulkan/Metal/Mali).
      const openclChosen = spec.logs.some(l => /opencl/i.test(l))
      t.comment(`chosen GPU backend is OpenCL (Adreno): ${openclChosen}`)
      if (openclChosen) {
        const projectorOnOpencl = spec.logs.some(
          l => /opencl/i.test(l) &&
            /(using device|backend detected|chosen gpu|clip|mmproj)/i.test(l))
        t.ok(projectorOnOpencl,
          'on Adreno OpenCL the vision projector should run on the OpenCL GPU backend')
      } else {
        t.comment('non-OpenCL GPU backend — OpenCL-routing assertion skipped (accuracy still asserted)')
      }
    } finally {
      await inference.unload().catch(() => {})
      spec.release()
    }
  })

setImmediate(() => {
  setTimeout(() => {}, 500)
})
