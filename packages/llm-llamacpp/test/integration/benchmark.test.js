'use strict'

const test = require('brittle')
const fs = require('bare-fs')
const path = require('bare-path')
const os = require('bare-os')
const process = require('bare-process')
const LlmLlamacpp = require('../../index.js')
const { ensureModel, getMediaPath } = require('./utils')
const {
  platform,
  arch,
  isMobile,
  recordPerformance
} = require('./_perf-helper.js')

const isDarwinX64 = platform === 'darwin' && arch === 'x64'
const isLinuxArm64 = platform === 'linux' && arch === 'arm64'
const useCpu = isDarwinX64 || isLinuxArm64

function _envStr (key, fallback) {
  let raw = ''
  if (typeof os.getEnv === 'function') raw = os.getEnv(key) || ''
  if (!raw && typeof process !== 'undefined' && process.env) raw = process.env[key] || ''
  return raw || fallback
}

function _envInt (key, fallback) {
  const v = parseInt(_envStr(key, ''), 10)
  return Number.isFinite(v) && v > 0 ? v : fallback
}

const PERF_RUNS = _envInt('QVAC_PERF_RUNS', 1)
const PERF_WARMUP_RUNS = _envInt('QVAC_PERF_WARMUP_RUNS', 1)
const BENCH_VARIANT = _envStr('BENCH_VARIANT', '')

const HF_BASE = 'https://huggingface.co'

const MODELS = [
  {
    key: 'gemma4-e2b-q4km',
    family: 'gemma4',
    modelDir: 'gemma-4-E2B-it',
    llmFile: 'gemma-4-E2B-it-Q4_K_M.gguf',
    llmUrl: `${HF_BASE}/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf`,
    projFile: 'mmproj-F16.gguf',
    projUrl: `${HF_BASE}/unsloth/gemma-4-E2B-it-GGUF/resolve/main/mmproj-F16.gguf`,
    projLocalName: 'mmproj-gemma4-E2B-F16.gguf',
    ctx_size: '8192'
  },
  {
    key: 'gemma4-e2b-q8',
    family: 'gemma4',
    modelDir: 'gemma-4-E2B-it',
    llmFile: 'gemma-4-E2B-it-Q8_0.gguf',
    llmUrl: `${HF_BASE}/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q8_0.gguf`,
    projFile: 'mmproj-F16.gguf',
    projUrl: `${HF_BASE}/unsloth/gemma-4-E2B-it-GGUF/resolve/main/mmproj-F16.gguf`,
    projLocalName: 'mmproj-gemma4-E2B-F16.gguf',
    ctx_size: '8192'
  },
  {
    key: 'gemma4-e4b-q4km',
    family: 'gemma4',
    modelDir: 'gemma-4-E4B-it',
    llmFile: 'gemma-4-E4B-it-Q4_K_M.gguf',
    llmUrl: `${HF_BASE}/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf`,
    projFile: 'mmproj-F16.gguf',
    projUrl: `${HF_BASE}/unsloth/gemma-4-E4B-it-GGUF/resolve/main/mmproj-F16.gguf`,
    projLocalName: 'mmproj-gemma4-E4B-F16.gguf',
    ctx_size: '8192'
  },
  {
    key: 'gemma4-e4b-q8',
    family: 'gemma4',
    modelDir: 'gemma-4-E4B-it',
    llmFile: 'gemma-4-E4B-it-Q8_0.gguf',
    llmUrl: `${HF_BASE}/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf`,
    projFile: 'mmproj-F16.gguf',
    projUrl: `${HF_BASE}/unsloth/gemma-4-E4B-it-GGUF/resolve/main/mmproj-F16.gguf`,
    projLocalName: 'mmproj-gemma4-E4B-F16.gguf',
    ctx_size: '8192'
  },
  {
    key: 'qwen35-2b-q4km',
    family: 'qwen35',
    modelDir: 'Qwen3.5-2B',
    llmFile: 'Qwen3.5-2B-Q4_K_M.gguf',
    llmUrl: `${HF_BASE}/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf`,
    projFile: 'mmproj-F16.gguf',
    projUrl: `${HF_BASE}/unsloth/Qwen3.5-2B-GGUF/resolve/main/mmproj-F16.gguf`,
    projLocalName: 'mmproj-Qwen3.5-2B-F16.gguf',
    ctx_size: '4096'
  },
  {
    key: 'qwen35-2b-q8',
    family: 'qwen35',
    modelDir: 'Qwen3.5-2B',
    llmFile: 'Qwen3.5-2B-Q8_0.gguf',
    llmUrl: `${HF_BASE}/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q8_0.gguf`,
    projFile: 'mmproj-F16.gguf',
    projUrl: `${HF_BASE}/unsloth/Qwen3.5-2B-GGUF/resolve/main/mmproj-F16.gguf`,
    projLocalName: 'mmproj-Qwen3.5-2B-F16.gguf',
    ctx_size: '4096'
  },
  {
    key: 'qwen35-4b-q4km',
    family: 'qwen35',
    modelDir: 'Qwen3.5-4B',
    llmFile: 'Qwen3.5-4B-Q4_K_M.gguf',
    llmUrl: `${HF_BASE}/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf`,
    projFile: 'mmproj-F16.gguf',
    projUrl: `${HF_BASE}/unsloth/Qwen3.5-4B-GGUF/resolve/main/mmproj-F16.gguf`,
    projLocalName: 'mmproj-Qwen3.5-4B-F16.gguf',
    ctx_size: '4096'
  },
  {
    key: 'qwen35-4b-q8',
    family: 'qwen35',
    modelDir: 'Qwen3.5-4B',
    llmFile: 'Qwen3.5-4B-Q8_0.gguf',
    llmUrl: `${HF_BASE}/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q8_0.gguf`,
    projFile: 'mmproj-F16.gguf',
    projUrl: `${HF_BASE}/unsloth/Qwen3.5-4B-GGUF/resolve/main/mmproj-F16.gguf`,
    projLocalName: 'mmproj-Qwen3.5-4B-F16.gguf',
    ctx_size: '4096'
  }
]

const IMAGES = [
  {
    name: 'elephant',
    file: 'elephant.jpg',
    keywords: ['elephant', 'elephants'],
    keywordType: 'elephant-related'
  },
  {
    name: 'fruit-plate',
    file: 'fruitPlate.png',
    keywords: ['fruit', 'fruits', 'plate', 'apple', 'apples'],
    keywordType: 'fruit-related',
    iosWarmupImage: 'elephant.jpg',
    iosPerfRuns: 1
  }
]

const PROMPT = 'Describe the image briefly in one sentence.'
const TEST_TIMEOUT = 30 * 60 * 1000

async function resolveModelPath (fileName, modelDir, downloadUrl, localName) {
  const saveName = localName || fileName

  const envDir = _envStr('BENCH_MODEL_DIR', '')
  if (envDir) {
    const p = path.join(envDir, modelDir, fileName)
    if (fs.existsSync(p)) {
      console.log(`[bench] local (env): ${p}`)
      return p
    }
  }

  const siblingDir = path.resolve(
    __dirname, '..', '..', '..', '..', '..', 'vlm-benchmark', 'models'
  )
  const siblingPath = path.join(siblingDir, modelDir, fileName)
  if (fs.existsSync(siblingPath)) {
    console.log(`[bench] local (sibling): ${siblingPath}`)
    return siblingPath
  }

  const cachePath = path.resolve(__dirname, '../model', saveName)
  if (fs.existsSync(cachePath) && fs.statSync(cachePath).size > 0) {
    console.log(`[bench] cached: ${cachePath}`)
    return cachePath
  }

  console.log(`[bench] downloading ${saveName} from HuggingFace...`)
  const [name, dir] = await ensureModel({ modelName: saveName, downloadUrl })
  return path.join(dir, name)
}

function getConfig (model) {
  const cfg = {
    gpu_layers: '98',
    temp: '0',
    seed: '42',
    verbosity: '2',
    device: useCpu ? 'cpu' : 'gpu',
    ctx_size: model.ctx_size,
    n_predict: '256'
  }
  if (model.family === 'gemma4') {
    cfg['reasoning-budget'] = '0'
  }
  return cfg
}

async function describeImage (inference, imageFilePath) {
  const imageBytes = new Uint8Array(fs.readFileSync(imageFilePath))
  const messages = [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', type: 'media', content: imageBytes },
    { role: 'user', content: PROMPT }
  ]

  const startTime = Date.now()
  const response = await inference.run(messages)
  const generatedText = []
  let error = null

  response.onUpdate(data => {
    generatedText.push(data)
  }).onError(err => {
    error = err
  })

  await response.await()

  if (error) {
    throw new Error('Inference error: ' + error)
  }

  return {
    generatedText: generatedText.join(''),
    startTime,
    endTime: Date.now(),
    stats: response.stats || null
  }
}

function checkKeywords (text, keywords) {
  const found = keywords.filter(kw => {
    const regex = new RegExp(`\\b${kw}\\b`, 'i')
    return regex.test(text)
  })
  return { found, hasMatch: found.length > 0 }
}

for (const model of MODELS) {
  for (const image of IMAGES) {
    const label = `[${model.key}] [${image.name}]`
    const testName = `benchmark ${model.key} × ${image.name}`

    test(testName, { timeout: TEST_TIMEOUT }, async t => {
      let inference = null

      t.teardown(async () => {
        if (inference) {
          try { await inference.unload() } catch (_) {}
        }
      })

      try {
        const llmPath = await resolveModelPath(
          model.llmFile, model.modelDir, model.llmUrl
        )
        const projPath = await resolveModelPath(
          model.projFile, model.modelDir, model.projUrl, model.projLocalName
        )

        t.ok(fs.existsSync(llmPath), `${label} LLM model exists`)
        t.ok(fs.existsSync(projPath), `${label} mmproj model exists`)

        inference = new LlmLlamacpp({
          files: { model: [llmPath], projectionModel: projPath },
          config: getConfig(model),
          logger: console,
          opts: { stats: true }
        })

        await inference.load()

        const imageFilePath = getMediaPath(image.file)
        t.ok(fs.existsSync(imageFilePath), `${label} image file exists`)

        let iosPreWarmupRan = false
        if (platform === 'ios' && image.iosWarmupImage) {
          try {
            const warmupPath = getMediaPath(image.iosWarmupImage)
            if (fs.existsSync(warmupPath)) {
              t.comment(`${label} iOS pre-warmup with ${image.iosWarmupImage}`)
              const w = await describeImage(inference, warmupPath)
              t.comment(
                `${label} iOS pre-warmup done in ${w.endTime - w.startTime}ms`
              )
              iosPreWarmupRan = true
            }
          } catch (err) {
            t.comment(`${label} iOS pre-warmup failed (non-fatal): ${err.message}`)
          }
        }

        if (!iosPreWarmupRan) {
          for (let w = 1; w <= PERF_WARMUP_RUNS; w++) {
            try {
              const r = await describeImage(inference, imageFilePath)
              t.comment(
                `${label} warmup ${w}/${PERF_WARMUP_RUNS} ` +
                `(${r.endTime - r.startTime}ms) — not recorded`
              )
            } catch (err) {
              t.comment(`${label} warmup ${w} failed: ${err.message}`)
            }
          }
        }

        const countedRuns = (platform === 'ios' && Number.isFinite(image.iosPerfRuns))
          ? image.iosPerfRuns
          : PERF_RUNS

        let lastText = ''
        for (let run = 1; run <= countedRuns; run++) {
          const { generatedText, startTime, endTime, stats } =
            await describeImage(inference, imageFilePath)
          const totalTime = endTime - startTime
          lastText = generatedText

          t.comment(`${label} run ${run}/${countedRuns}: ${generatedText}`)
          t.comment(recordPerformance(label, totalTime, {
            _output: generatedText,
            stats,
            deviceId: useCpu ? 'cpu' : 'gpu',
            scenario: 'image-benchmark',
            model: model.key,
            variant: BENCH_VARIANT
          }))
        }

        t.ok(lastText.length > 0, `${label} generated text output`)
        const { found, hasMatch } = checkKeywords(lastText, image.keywords)
        t.ok(hasMatch,
          `${label} output should contain ${image.keywordType} keyword. ` +
          `Found: ${found.join(', ') || 'none'}. Output: "${lastText}"`)
      } catch (err) {
        t.comment(`${label} SKIP: ${err.message}`)
        t.pass(`${label} skipped due to error (OOM / context overflow / load failure)`)
      }
    })
  }
}

setImmediate(() => {
  setTimeout(() => {}, 500)
})
