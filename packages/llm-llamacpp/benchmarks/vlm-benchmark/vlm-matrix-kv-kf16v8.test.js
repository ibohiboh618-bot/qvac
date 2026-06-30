'use strict'
// QVAC-21318: per-cell VLM kv-sweep entry — mixed K=f16 / V=q8_0 ONLY. Filename →
// mobile generator function `runVlmMatrixKvKf16v8Test` (own test-group in stage.cjs),
// so this cell runs in its OWN Device Farm run/process on mobile. K stays f16, so the
// cache shift never requantizes K — this is the crash-free way to get V-cache savings
// on Adreno OpenCL (the hypothesis under test).
const { runAll } = require('./harness.cjs')
runAll({ kvLabels: ['kf16v8'] })
