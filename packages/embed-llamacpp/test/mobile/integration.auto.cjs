'use strict'
require('./integration-runtime.cjs')

// AUTO-GENERATED FILE. Run `npm run test:mobile:generate` to update.
// Each function mirrors a single file under test/integration/.

/* global runIntegrationModule */

async function runAddonTest (options = {}) { // eslint-disable-line no-unused-vars
  return runIntegrationModule('../integration/addon.test.js', options)
}

async function runBenchmarkPerfEmbeddinggemmaQ40Test (options = {}) { // eslint-disable-line no-unused-vars
  return runIntegrationModule('../integration/benchmark-perf-embeddinggemma-q4-0.test.js', options)
}

async function runBenchmarkPerfEmbeddinggemmaQ80Test (options = {}) { // eslint-disable-line no-unused-vars
  return runIntegrationModule('../integration/benchmark-perf-embeddinggemma-q8-0.test.js', options)
}

async function runBenchmarkPerfQwen3Embedding06bF16Test (options = {}) { // eslint-disable-line no-unused-vars
  return runIntegrationModule('../integration/benchmark-perf-qwen3-embedding-06b-f16.test.js', options)
}

async function runBenchmarkPerfQwen3Embedding06bQ80Test (options = {}) { // eslint-disable-line no-unused-vars
  return runIntegrationModule('../integration/benchmark-perf-qwen3-embedding-06b-q8-0.test.js', options)
}

async function runBenchmarkPerfQwen3Embedding4bGgufF16Test (options = {}) { // eslint-disable-line no-unused-vars
  return runIntegrationModule('../integration/benchmark-perf-qwen3-embedding-4b-gguf-f16.test.js', options)
}

async function runBenchmarkPerfQwen3Embedding4bGgufQ4KMTest (options = {}) { // eslint-disable-line no-unused-vars
  return runIntegrationModule('../integration/benchmark-perf-qwen3-embedding-4b-gguf-q4-k-m.test.js', options)
}

async function runBenchmarkPerfQwen3Embedding4bGgufQ80Test (options = {}) { // eslint-disable-line no-unused-vars
  return runIntegrationModule('../integration/benchmark-perf-qwen3-embedding-4b-gguf-q8-0.test.js', options)
}

async function runModelLoadingTest (options = {}) { // eslint-disable-line no-unused-vars
  return runIntegrationModule('../integration/model-loading.test.js', options)
}

async function runMultiGpuTest (options = {}) { // eslint-disable-line no-unused-vars
  return runIntegrationModule('../integration/multi-gpu.test.js', options)
}

async function runMultiInstanceTest (options = {}) { // eslint-disable-line no-unused-vars
  return runIntegrationModule('../integration/multi-instance.test.js', options)
}
