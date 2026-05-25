'use strict'

const test = require('brittle')
const { runMobilePerfCase } = require('./mobile-perf-runner.js')

test('Mobile perf tiny GPU', { timeout: 600000 }, async (t) => {
  // QVAC-19213: un-quarantined on Android to verify the Adreno 740 Vulkan
  // fix (mul_mat_vec subgroup->shmem on Qualcomm) on the AWS Device Farm.
  // Ordered LAST in the suite (see integration.auto.cjs) so that if the
  // separate GPU-teardown crash (whisper.cpp#2373) still fires on the
  // Samsung S25 Ultra, it cannot abort the other integration cases — the
  // bare app shares one process per device (wdio.template.js checkAppCrash).
  await runMobilePerfCase(t, {
    modelFile: 'ggml-tiny.bin',
    useGPU: true
  })
})
