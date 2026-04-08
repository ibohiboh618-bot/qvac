#!/usr/bin/env node
'use strict'

/**
 * Download OCR models from HuggingFace into models/ocr/rec_dyn/.
 *
 * Usage:
 *   node scripts/download-ocr-models.js              # detector + latin only
 *   node scripts/download-ocr-models.js korean       # also download recognizer_korean
 *   node scripts/download-ocr-models.js korean arabic # multiple recognizers
 *   node scripts/download-ocr-models.js all           # all recognizers used by full-ocr-suite
 */

const fs = require('fs')
const path = require('path')
const https = require('https')

const HF_BASE_URL = 'https://huggingface.co/olyas/easyocr-onnx/resolve/main/rec_dyn'
const OUT_DIR = path.resolve(__dirname, '..', 'models', 'ocr', 'rec_dyn')

const FULL_OCR_SUITE_RECOGNIZERS = ['latin', 'korean', 'arabic', 'cyrillic', 'devanagari', 'bengali', 'thai', 'zh_sim', 'zh_tra', 'japanese', 'tamil', 'telugu', 'kannada']

const MODELS = {
  detector_craft: { filename: 'detector_craft.onnx' }
}
for (const name of FULL_OCR_SUITE_RECOGNIZERS) {
  MODELS[`recognizer_${name}`] = { filename: `recognizer_${name}.onnx` }
}

function downloadFile (url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest)
    https.get(url, (response) => {
      if (response.statusCode === 301 || response.statusCode === 302) {
        file.close()
        fs.unlinkSync(dest)
        return downloadFile(response.headers.location, dest).then(resolve, reject)
      }
      if (response.statusCode !== 200) {
        file.close()
        fs.unlinkSync(dest)
        return reject(new Error(`HTTP ${response.statusCode} for ${url}`))
      }
      response.pipe(file)
      file.on('finish', () => file.close(resolve))
    }).on('error', (err) => {
      fs.unlinkSync(dest)
      reject(err)
    })
  })
}

async function main () {
  const args = process.argv.slice(2).map(a => a.toLowerCase())
  const wantAll = args.includes('all')
  const requested = args.filter(a => a !== 'all')

  const toDownload = ['detector_craft', 'recognizer_latin']
  if (wantAll) {
    for (const name of FULL_OCR_SUITE_RECOGNIZERS) {
      toDownload.push(`recognizer_${name}`)
    }
  } else {
    for (const name of requested) {
      const key = name.startsWith('recognizer_') ? name : `recognizer_${name}`
      if (MODELS[key]) toDownload.push(key)
      else console.warn(`Unknown model: ${name} (skipping)`)
    }
  }
  const unique = [...new Set(toDownload)]

  fs.mkdirSync(OUT_DIR, { recursive: true })

  console.log('Downloading OCR models from HuggingFace...')
  console.log('  Output dir:', OUT_DIR)
  console.log('  Models:', unique.join(', '))

  for (const key of unique) {
    const model = MODELS[key]
    if (!model) continue
    const outPath = path.join(OUT_DIR, model.filename)
    if (fs.existsSync(outPath)) {
      console.log(`  ✓ ${model.filename} (already exists)`)
      continue
    }
    try {
      console.log(`  Downloading ${model.filename}...`)
      await downloadFile(`${HF_BASE_URL}/${model.filename}`, outPath)
      console.log(`  ✓ ${model.filename}`)
    } catch (err) {
      console.error(`  ✗ ${model.filename}: ${err.message}`)
    }
  }
  console.log('Done.')
}

main().catch(err => {
  console.error(err)
  process.exit(1)
})
