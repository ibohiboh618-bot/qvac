'use strict'

// Single source of truth for the sweep axes + input modes, shared by the Bare
// sweep (config + case-runner) and the Node renderer (coverage denominator).
// Plain literals only — no bare-fs/fs imports — so it loads in both runtimes.
const PARAMETER_SWEEP = {
  quantization: ['Q4_0', 'Q4_K_M', 'Q8_0', 'F16'],
  // Desktop is GPU-only, matching the LLM benchmark (its getDefaultSweepDevices
  // returns ['gpu'] off Android). CPU embedding of the large batch/array configs
  // is impractical (a 4096 x array-20 prefill is ~82k tokens at a few hundred
  // tok/s) and isn't a real desktop use case; CPU is covered on the mobile path.
  device: ['gpu'],
  batchSize: [256, 512, 1024, 2048, 4096],
  flashAttn: ['off', 'on']
}

// Array-mode sequence counts (desktop sweep): embed a batch of N sequences in one
// call to measure batched throughput at N = 5 / 10 / 20.
const ARRAY_SEQUENCE_COUNTS = [5, 10, 20]
const MAX_ARRAY_SEQUENCES = Math.max(...ARRAY_SEQUENCE_COUNTS)

// Input modes per swept config: 'single' embeds one sequence; 'array-N' embeds N
// sequences in one call. inputs.json provides MAX_ARRAY_SEQUENCES sequences per
// batch size, sliced to N for each array mode.
const INPUT_MODES = ['single', ...ARRAY_SEQUENCE_COUNTS.map((n) => `array-${n}`)]

// Effective context (max sequence length) per model. A batch size above this
// overflows the model, so those configs are skipped from the sweep and the
// coverage denominator. embeddingGemma-300m is 2048; Qwen3 embedding models are
// 32k, so they take every swept batch size (default = no cap).
const MODEL_MAX_CONTEXT = { embeddingGemma: 2048 }

function maxBatchForModel (modelId) {
  return MODEL_MAX_CONTEXT[modelId] || Infinity
}

module.exports = { PARAMETER_SWEEP, INPUT_MODES, ARRAY_SEQUENCE_COUNTS, MAX_ARRAY_SEQUENCES, MODEL_MAX_CONTEXT, maxBatchForModel }
