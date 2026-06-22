'use strict'

const { OcrGgml } = require('../..')
const test = require('brittle')
const os = require('bare-os')
const process = require('bare-process')
const { platform, getImagePath, ensureModelPath, safeUnload } = require('./utils')

const DESKTOP_TIMEOUT = 240 * 1000 // 4 minutes: loads + runs the pipeline twice

// Guards the opt-in CRAFT F16-activation path (QVAC-20908). By default CRAFT
// activations are F32 (only the conv kernels may be F16); OCR_GGML_CRAFT_F16_ACT=1
// runs the detection U-net's intermediate activations in F16 (input cast to
// F16, conv outputs kept F16, final cast back to F32). That must be accuracy-
// equivalent, so this runs the SAME image on the SAME backend twice — F32
// activations (default) then F16 activations — and asserts identical recognized
// output, plus that each pass is correct.
//
// Why Metal-only: F16 activations require every CRAFT op (conv/pool/relu/concat/
// interpolate) to support F16 on the backend, and the flag only takes effect
// when the conv kernels are F16. Apple Metal ships F16 kernels by default and
// is the desktop backend we can exercise here without risking a GGML_ASSERT on
// an unsupported F16 op; the real fast-F16 targets (NVIDIA Vulkan, Adreno) are
// validated via the #2544 quality/perf benchmark. We deliberately skip linux
// CPU/Vulkan and Windows (F16-activation support there is unverified) and mobile
// (env toggles don't propagate to the device farm).
const ENV_KEY = 'OCR_GGML_CRAFT_F16_ACT'
const shouldSkip = platform !== 'darwin'

test('EasyOCR CRAFT F16 activations match F32 (Metal)', { timeout: DESKTOP_TIMEOUT, skip: shouldSkip }, async function (t) {
  const hasGetEnv = typeof os.getEnv === 'function'
  const hasSetEnv = typeof os.setEnv === 'function'
  const prevValue = (hasGetEnv ? os.getEnv(ENV_KEY) : process.env[ENV_KEY]) || ''

  function setEnv (val) {
    if (hasSetEnv) os.setEnv(ENV_KEY, val)
    process.env[ENV_KEY] = val
  }

  function restoreEnv () {
    if (prevValue) { setEnv(prevValue); return }
    if (typeof os.unsetEnv === 'function') os.unsetEnv(ENV_KEY)
    else if (hasSetEnv) os.setEnv(ENV_KEY, '')
    // bare-process's env proxy rejects `delete` (TypeError under strict mode); '' is sufficient since the addon reads via std::getenv.
    process.env[ENV_KEY] = ''
  }

  // Run the EasyOCR pipeline once on Metal and return the recognized regions.
  async function runMetalPass (tag) {
    const detectorPath = await ensureModelPath('detector_craft')
    const recognizerPath = await ensureModelPath('recognizer_latin')
    const imagePath = getImagePath('/test/images/basic_test.bmp')
    const ocrGgml = new OcrGgml({
      params: { pathDetector: detectorPath, pathRecognizer: recognizerPath, langList: ['en'], backendDevice: 'metal' },
      opts: { stats: true }
    })
    await ocrGgml.load()
    const info = typeof ocrGgml.getBackendInfo === 'function' ? ocrGgml.getBackendInfo() : null
    const isGpu = !!info && (info.backendDevice === 'GPU' || info.backendDevice === 'IGPU')
    if (!isGpu) {
      // Metal didn't resolve to a GPU: bail rather than force F16 activations
      // onto a backend whose conv kernels may be F32 (the flag would no-op) or
      // that may not support F16 vision ops.
      await safeUnload(ocrGgml)
      return { skipped: true, output: [] }
    }
    let output = []
    try {
      const response = await ocrGgml.run({ path: imagePath, options: { paragraph: false } })
      await response
        .onUpdate(o => { output = o })
        .onError(error => { t.fail(tag + ': unexpected error: ' + JSON.stringify(error)) })
        .await()
      return { skipped: false, output }
    } finally {
      await safeUnload(ocrGgml)
      await new Promise(resolve => setTimeout(resolve, 1000))
    }
  }

  function assertCorrect (output, tag) {
    t.ok(Array.isArray(output), tag + ': output should be an array')
    t.is(output.length, 3, tag + `: output length should be 3, got ${output.length}`)
    const texts = output.map(o => o[1])
    t.ok(texts.includes('tilted'), tag + ': should contain "tilted"')
    t.ok(texts.includes('normal'), tag + ': should contain "normal"')
    t.ok(texts.includes('vertical'), tag + ': should contain "vertical"')
  }

  try {
    t.comment('Pass A: F32 activations (default); platform: ' + platform)
    setEnv('')
    const resF32 = await runMetalPass('f32-act')
    if (resF32.skipped) {
      t.pass('skipped: Metal GPU not available on this host')
      return
    }
    assertCorrect(resF32.output, 'f32-act')

    t.comment('Pass B: F16 activations (OCR_GGML_CRAFT_F16_ACT=1)')
    setEnv('1')
    const resF16 = await runMetalPass('f16-act')
    if (resF16.skipped) {
      t.pass('skipped: Metal GPU not available for the F16 pass')
      return
    }
    assertCorrect(resF16.output, 'f16-act')

    // Equivalence: F16 activations must match the F32 path region-for-region.
    t.is(resF16.output.length, resF32.output.length, 'F16 and F32 activations produce the same number of regions')
    const n = Math.min(resF16.output.length, resF32.output.length)
    for (let i = 0; i < n; i++) {
      t.is(resF16.output[i][1], resF32.output[i][1], `region ${i}: F16-act text matches F32 ("${resF32.output[i][1]}")`)
    }

    t.pass('CRAFT F16-activation path matches F32 output on Metal')
  } finally {
    restoreEnv()
  }
})
