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
  // Declare an INTENTIONAL skip via the harness-provided global.skipMobileTest
  // (registers a real brittle skip → total > 0 → reported skipped, never a
  // silent green pass). See ./rtf-benchmark.test.js for the full rationale.
  if (typeof globalThis !== 'undefined' && typeof globalThis.skipMobileTest === 'function') {
    globalThis.skipMobileTest('Streaming benchmark', 'QVAC_TTS_GGML_RUN_BENCHMARK_ON_MOBILE not set')
  } else {
    console.log('[streaming-benchmark mobile shim] QVAC_TTS_GGML_RUN_BENCHMARK_ON_MOBILE not set; skipping benchmark.')
  }
}
