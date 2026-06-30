'use strict'
// QVAC-21318: per-cell VLM kv-sweep entry — mixed K=q8_0 / V=f16 ONLY. Filename →
// mobile generator function `runVlmMatrixKvK8vf16Test` (own test-group in stage.cjs),
// so this cell runs in its OWN Device Farm run/process on mobile. K=q8_0 routes the K
// cache through the quantized RoPE cache-shift (dequant→rope→requant), which aborts
// natively on Adreno OpenCL — process isolation keeps that abort from affecting the f16
// / kf16v8 runs.
const { runAll } = require('./harness.cjs')
runAll({ kvLabels: ['k8vf16'] })
