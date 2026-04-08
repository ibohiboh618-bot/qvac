'use strict'

const fs = require('bare-fs')
const path = require('bare-path')

const DEFAULT_DISK_PATH = './models/ocr'

const HF_BASE_URL = 'https://huggingface.co/olyas/easyocr-onnx/resolve/main/rec_dyn'

const OCR_MODELS = {
  detector: {
    url: `${HF_BASE_URL}/detector_craft.onnx`,
    filename: 'detector_craft.onnx'
  },
  recognizer_latin: {
    url: `${HF_BASE_URL}/recognizer_latin.onnx`,
    filename: 'recognizer_latin.onnx'
  }
}

async function ensureModels (diskPath) {
  diskPath = diskPath || DEFAULT_DISK_PATH

  const detectorPath = path.join(diskPath, OCR_MODELS.detector.filename)
  const recognizerPath = path.join(diskPath, OCR_MODELS.recognizer_latin.filename)

  // Check if models already exist
  if (fs.existsSync(detectorPath) && fs.existsSync(recognizerPath)) {
    console.log('Models already cached locally.')
    return { detectorPath, recognizerPath }
  }

  fs.mkdirSync(diskPath, { recursive: true })

  console.log('Downloading OCR models from HuggingFace...')
  const fetch = require('bare-fetch')

  for (const [key, model] of [['detector', OCR_MODELS.detector], ['recognizer_latin', OCR_MODELS.recognizer_latin]]) {
    const outPath = key === 'detector' ? detectorPath : recognizerPath
    if (fs.existsSync(outPath)) {
      console.log(`  ${model.filename} already exists.`)
      continue
    }
    console.log(`  Downloading ${model.filename}...`)
    const response = await fetch(model.url)
    if (!response.ok) throw new Error(`HTTP ${response.status} downloading ${model.filename}`)
    const buffer = await response.arrayBuffer()
    fs.writeFileSync(outPath, Buffer.from(buffer))
    console.log(`  Downloaded: ${model.filename}`)
  }

  console.log('Models ready.')

  return { detectorPath, recognizerPath }
}

module.exports = { ensureModels, OCR_MODELS, DEFAULT_DISK_PATH }
