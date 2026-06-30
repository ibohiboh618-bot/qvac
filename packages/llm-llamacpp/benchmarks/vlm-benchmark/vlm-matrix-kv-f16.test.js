'use strict'
// QVAC-21318: per-cell VLM kv-sweep entry — f16 baseline ONLY. Filename → mobile
// generator function `runVlmMatrixKvF16Test` (registered as its own test-group in
// stage.cjs), so on mobile this cell runs in its OWN Device Farm run/process. That
// isolates a native abort in another cell (e.g. k8vf16 on Adreno OpenCL) from this one.
// Desktop uses vlm-matrix.test.js (full sweep, one process); these per-cell files are
// only staged onto mobile by stage.cjs.
const { runAll } = require('./harness.cjs')
runAll({ kvLabels: ['f16'] })
