'use strict'

const test = require('brittle')
const { getImagePath, runDoctrWarmProfile, ensureDoctrModels } = require('./utils')

const DOCTR_TEST_TIMEOUT = 180 * 1000

let DB_MOBILENET
let CRNN_MOBILENET
let modelsAvailable = false

test('DocTR warm profile - download models', { timeout: DOCTR_TEST_TIMEOUT }, async function (t) {
  const models = await ensureDoctrModels()
  if (!models) {
    t.comment('DocTR models unavailable (download failed) — warm test skipped')
    return
  }
  DB_MOBILENET = models.db_mobilenet_v3_large
  CRNN_MOBILENET = models.crnn_mobilenet_v3_small
  modelsAvailable = true
  t.ok(DB_MOBILENET, 'db_mobilenet model available')
  t.ok(CRNN_MOBILENET, 'crnn_mobilenet model available')
})

test('DocTR warm profile [VULKAN] - cold vs warm runs', { timeout: DOCTR_TEST_TIMEOUT }, async function (t) {
  if (!modelsAvailable) { t.comment('Skipped — models unavailable'); return }
  const imagePath = getImagePath('/test/images/clinical_chemistry.png')
  await runDoctrWarmProfile(t, {
    params: { pathDetector: DB_MOBILENET, pathRecognizer: CRNN_MOBILENET },
    imagePath,
    runs: 5
  })
})
