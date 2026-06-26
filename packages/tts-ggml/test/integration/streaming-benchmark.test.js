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
  console.log('[streaming-benchmark mobile shim] QVAC_TTS_GGML_RUN_BENCHMARK_ON_MOBILE not set; skipping benchmark.')
  // bare-pack isolates globalThis/global per module, but console is shared.
  console.__QVAC_SKIP_FLAG = true
  if (typeof globalThis !== 'undefined') globalThis.__QVAC_TEST_SKIPPED = true
  exports.__QVAC_SKIPPED = true
  module.exports = { __QVAC_SKIPPED: true }
}
