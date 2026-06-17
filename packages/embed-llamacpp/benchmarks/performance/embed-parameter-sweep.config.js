'use strict'

const fs = require('bare-fs')
const path = require('bare-path')
const { PARAMETER_SWEEP } = require('./_sweep-grid')

const DEFAULT_RESULTS_DIR = path.resolve(__dirname, 'results', 'parameter-sweep')
const DEFAULT_MODELS_DIR = path.resolve(__dirname, 'models')
const MANIFEST_PATH = path.resolve(__dirname, 'models.manifest.json')
const RESOLVED_MODELS_PATH = path.resolve(__dirname, 'resolved-models.json')
const DEFAULT_INPUTS_FILE = path.resolve(__dirname, 'inputs.json')
const DEFAULT_REPEATS = 5

// Baseline reference config (cosine-similarity is measured against its
// embeddings): CPU, flash-attn off, batch-size 256. The reference quantization
// is the highest-fidelity build per model (F16 where available, else the best
// quant) — chosen in buildCases, not here.
const BENCH_DEFAULT_RUNTIME = {
  device: 'cpu',
  batchSize: 256,
  noMmap: false,
  flashAttn: 'off',
  ngl: 0
}

// Optional per-model runtime overrides. Only add entries when a model needs
// non-global defaults (for example because of VRAM limitations).
const MODEL_RUNTIME_OVERRIDES = {
}

function toModelRelativePath (localPath) {
  const normalized = String(localPath || '')
    .trim()
    .replace(/\\/g, '/')
    .replace(/^\.\//, '')
  if (!normalized) return null

  // resolved-models.json stores paths relative to benchmark root (for example
  // "models/<repo>/<revision>/<file>.gguf"). The runtime loader uses modelDir
  // as root, so strip the leading "models/" segment when present.
  return normalized.startsWith('models/')
    ? normalized.slice('models/'.length)
    : normalized
}

function buildQuantizationFiles (manifestModel, resolvedModelEntry) {
  const manifestQuants = Array.isArray(manifestModel.gguf && manifestModel.gguf.quantizations)
    ? manifestModel.gguf.quantizations
    : []

  if (resolvedModelEntry && resolvedModelEntry.gguf && resolvedModelEntry.gguf.files) {
    const normalized = {}
    for (const [quantization, localPath] of Object.entries(resolvedModelEntry.gguf.files)) {
      normalized[quantization] = toModelRelativePath(localPath)
    }
    return normalized
  }

  // Without resolved-models.json, keep supported quantization keys but mark
  // filenames as null so checks can fail with a clear "missing model file" error.
  const fallback = {}
  for (const quantization of manifestQuants) {
    fallback[quantization] = null
  }
  return fallback
}

function loadModelsFromManifest () {
  const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf8'))
  let resolved = null
  if (fs.existsSync(RESOLVED_MODELS_PATH)) {
    resolved = JSON.parse(fs.readFileSync(RESOLVED_MODELS_PATH, 'utf8'))
  }

  const manifestModels = manifest.models || []
  return manifestModels.map((model) => {
    const resolvedEntry = resolved && resolved.models ? resolved.models[model.id] : null
    const quantizationFiles = buildQuantizationFiles(model, resolvedEntry)
    const defaults = {
      ...BENCH_DEFAULT_RUNTIME,
      ...(MODEL_RUNTIME_OVERRIDES[model.id] || {})
    }
    return {
      id: model.id,
      source: `https://huggingface.co/${model.gguf.repo}`,
      modelDir: DEFAULT_MODELS_DIR,
      quantizations: Array.isArray(model.gguf.quantizations) ? model.gguf.quantizations : [],
      quantizationFiles,
      defaults
    }
  })
}

const MODELS = loadModelsFromManifest()

// Sweep axes (PARAMETER_SWEEP) live in ./_sweep-grid so the Node renderer can
// share them as its coverage denominator without loading this bare-fs module.
// ubatch-size and mmap are not swept (mmap is held at its default).

module.exports = {
  DEFAULT_RESULTS_DIR,
  DEFAULT_MODELS_DIR,
  MANIFEST_PATH,
  RESOLVED_MODELS_PATH,
  DEFAULT_REPEATS,
  DEFAULT_INPUTS_FILE,
  BENCH_DEFAULT_RUNTIME,
  MODEL_RUNTIME_OVERRIDES,
  MODELS,
  PARAMETER_SWEEP
}
