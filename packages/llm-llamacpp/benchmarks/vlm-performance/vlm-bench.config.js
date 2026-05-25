'use strict'

// Single source of truth for benchmark defaults. Every field is
// overridable via CLI flag on run-vlm-bench.js or via workflow_dispatch
// input on the CI workflow. See README §Config Overrides.

module.exports = {
  // ── Sources ──────────────────────────────────────────────────────
  // The benchmark compares THREE inference engines on the SAME model:
  //   addon    — @qvac/llm-llamacpp JS addon (via Bare runtime)
  //   fabric   — llama-mtmd-cli built from qvac-fabric fork
  //   upstream — llama-mtmd-cli built from upstream ggml-org/llama.cpp
  //
  // This measures JS binding overhead (addon vs fabric) and fork
  // divergence (fabric vs upstream). Each source can be individually
  // enabled/disabled via --sources=addon,fabric,upstream.
  sources: {
    addon:    { type: 'addon', enabled: true },
    fabric:   { type: 'cli',  enabled: true, configKey: 'fabric' },
    upstream: { type: 'cli',  enabled: true, configKey: 'upstream' }
  },

  comparisonMode: 'source-engines',

  // ── Model ────────────────────────────────────────────────────────
  // Single model used across all three sources. All sources receive
  // the same GGUF files so performance differences reflect the
  // inference engine, not the model.
  model: {
    id: 'qwen3.5-0.8b',
    ctxSize: 4096,
    nPredict: 512,
    quant: 'Q4_K_M',
    label: 'unsloth-Q4_K_M',
    hfRepo: 'unsloth/Qwen3.5-0.8B-GGUF',
    hfRevision: '6ab461498e2023f6e3c1baea90a8f0fe38ab64d0',
    llmFile: 'Qwen3.5-0.8B-Q4_K_M.gguf',
    mmprojFile: 'mmproj-Qwen3.5-0.8B-F16.gguf',
    url: {
      llm: 'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/Qwen3.5-0.8B-Q4_K_M.gguf',
      mmproj: 'https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/mmproj-F16.gguf'
    }
  },

  // ── Case (image + prompt + ground truth) ─────────────────────────
  case: {
    image: 'assets/seven_objects.jpg',
    prompt: 'List only short English object names (comma-separated)',
    groundTruth: [
      { canonical: 'Elephant', accepts: ['elephant', 'elephants'] },
      { canonical: 'Umbrella', accepts: ['umbrella', 'umbrellas'] },
      { canonical: 'Submarine', accepts: ['submarine', 'submarines'] },
      { canonical: 'Backpack', accepts: ['backpack', 'backpacks', 'back pack', 'rucksack'] },
      { canonical: 'Windmill', accepts: ['windmill', 'windmills', 'wind mill'] },
      { canonical: 'Cactus', accepts: ['cactus', 'cacti', 'cactuses'] },
      { canonical: 'Helicopter', accepts: ['helicopter', 'helicopters', 'chopper'] }
    ]
  },

  // ── Sampling ─────────────────────────────────────────────────────
  sampling: { temperature: 0, seed: 42 },

  // ── Reasoning mode ───────────────────────────────────────────────
  // Thinking ON by default: the model emits <think>...</think> before
  // the answer, generating 200-400 tokens total. This gives stable TPS
  // measurements (16 tokens is too noisy). The accuracy scorer strips
  // think blocks automatically so recall scoring still works.
  thinking: { enabled: true },

  // ── Methodology ──────────────────────────────────────────────────
  run: {
    warmupRuns: 1,
    measuredRuns: 3,
    cooldownMs: 5000,
    perRunTimeoutMs: 5 * 60 * 1000
  },

  // ── Platforms / backends ─────────────────────────────────────────
  platforms: {
    'macos-arm64': { backends: ['gpu'] },
    'windows-x64': { backends: ['cpu', 'gpu'] },
    'linux-x64': { backends: ['cpu', 'gpu'] },
    android: { backends: ['auto'], device: 'Samsung Galaxy S25 Ultra' },
    ios: { backends: ['gpu'], device: 'iPhone 17' }
  },

  // ── Reporting ────────────────────────────────────────────────────
  reporting: {
    resultsDir: 'results',
    surfaceFullAnswer: true,
    answerTruncChars: 8000
  }
}
