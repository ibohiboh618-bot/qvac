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
  // Confirmation profile: everything at defaults (auto-hybrid + CPU-assist
  // recognition + LSTM split). Logs per-run detection/recognition timings,
  // box count, and the clinical-chemistry keyword guard for each warm run.
  await runDoctrWarmProfile(t, {
    label: ':auto',
    params: { pathDetector: detector, pathRecognizer: recognizer },
    imagePath,
    runs: 3
  })
})
