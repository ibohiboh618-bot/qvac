'use strict'

const test = require('brittle')
const { getImagePath, runDoctrWarmProfile } = require('./utils')

const DOCTR_TEST_TIMEOUT = 180 * 1000

// Resolve models the same way as images: on desktop -> ./models/<name>; on
// mobile -> the bundled testAssets/<name> URI (models are bundled into the app
// via the addon media/ folder so no network/presigned-URL download is needed).
function modelPath (name) {
  return getImagePath('/models/' + name)
}

test('DocTR warm profile [VULKAN] - cold vs warm runs', { timeout: DOCTR_TEST_TIMEOUT }, async function (t) {
  let detector, recognizer
  try {
    detector = modelPath('db_mobilenet_v3_large.gguf')
    recognizer = modelPath('crnn_mobilenet_v3_small.gguf')
  } catch (e) {
    t.comment('DocTR models not bundled — warm test skipped: ' + e.message)
    return
  }
  const imagePath = getImagePath('/test/images/clinical_chemistry.png')
  const os = require('bare-os')
  // NOTE: use bare-os setEnv/unsetEnv directly — `delete process.env.X` throws
  // in strict mode (bare-env's Proxy deleteProperty trap returns undefined).
  // Confirmation profile: everything at defaults (auto-hybrid + CPU-assist
  // recognition + LSTM split), with the fabric per-op profiler in logcat.
  os.setEnv('OCR_CPU_PROF', '1')
  try {
    await runDoctrWarmProfile(t, {
      label: ':auto',
      params: { pathDetector: detector, pathRecognizer: recognizer },
      imagePath,
      runs: 3
    })
    // Q8_0-quantised 1x1 detector convs (int8 GEMM via sdot/i8mm). Output
    // verified token-identical to F16 on x64 except one already-unstable
    // low-confidence token; the kw= guard re-checks on device.
    os.setEnv('OCR_DOCTR_DET_Q8', '1')
    await runDoctrWarmProfile(t, {
      label: ':q8',
      params: { pathDetector: detector, pathRecognizer: recognizer },
      imagePath,
      runs: 3
    })
  } finally {
    os.unsetEnv('OCR_CPU_PROF')
    os.unsetEnv('OCR_DOCTR_DET_Q8')
  }
})
