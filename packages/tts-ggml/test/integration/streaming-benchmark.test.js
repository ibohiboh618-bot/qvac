'use strict'

// Mobile entry point for the streaming-latency benchmark. See the long header
// in `./rtf-benchmark.test.js` for the full rationale; the only difference here
// is that this shim re-exports `test/benchmark/streaming-benchmark.test.js`
// instead, which measures Time-to-First-Audio (TTFA) and inter-chunk gap for
// `run({ streamOutput: true })`.

const os = require('bare-os')

const flag = typeof os.getEnv === 'function' ? (os.getEnv('QVAC_TTS_GGML_RUN_BENCHMARK_ON_MOBILE') || '') : ''
const enabled = flag === '1' || flag.toLowerCase() === 'true' || flag.toLowerCase() === 'yes'

if (enabled) {
  require('../benchmark/streaming-benchmark.test.js')
} else {
  // Declare an INTENTIONAL skip by registering a real brittle skip (→ total > 0
  // → reported skipped, never a silent green pass, and a genuine 0/0 still
  // FAILs). See ./rtf-benchmark.test.js for the full rationale.
  console.log('[streaming-benchmark mobile shim] QVAC_TTS_GGML_RUN_BENCHMARK_ON_MOBILE not set; skipping benchmark.')
  require('brittle').skip('Streaming benchmark — QVAC_TTS_GGML_RUN_BENCHMARK_ON_MOBILE not set', () => {})
}
