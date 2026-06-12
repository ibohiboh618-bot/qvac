'use strict'
// QVAC-19371 (A1 contract): SCENARIOS — the workload axis of the VLM benchmark.
// A scenario names a kind of work (which fixture tasks run, how answers are
// scored, how large the images are) independent of the model, the source build
// and the platform. Selected per run via QVAC_VLM_SCENARIOS (workflow input
// `matrix_scenarios`); the first CSV token is the active scenario (multi-
// scenario runs are reserved in CONTRACT.md, not implemented yet).
//
// OWNERSHIP: this file belongs to the scenarios/reporting workstream (Dev B).
// The runner side (harness.cjs) only reads it. Fields:
//   tasks       fixture task ids that make up the scenario (the task universe;
//               presets/env can narrow it, never widen it)
//   metricSet   'lmms' (existing per-task lmms-eval scorers) | 'category'
//               (VLM_General_Benchmark category rubric — scorer lands in B2)
//   tolerance   max allowed accuracy drop of addon@candidate vs addon@baseline
//               before the gate FAILs (consumed by the B4 gate)
//   maxSide     fixture image size cap in px (build-fixture.cjs --max-side)
//   fixturePending  true until the scenario's fixture exists — selecting such
//               a scenario fails fast with a clear error instead of running 0 items

module.exports = {
  // The proven default: the 5 open-licensed lmms-eval tasks shipped today.
  'vqa-suite': {
    title: 'VQA suite (low-MP)',
    tasks: ['textvqa', 'vizwiz', 'gqa', 'docvqa', 'ai2d'],
    metricSet: 'lmms',
    tolerance: 0.02,
    maxSide: 1024
  },
  // V1 target scenario — fixture + category scorer land in B1/B2.
  'image-description': {
    title: 'Image description (low-MP)',
    tasks: ['imgdesc'],
    metricSet: 'category',
    tolerance: 0.02,
    maxSide: 1024,
    fixturePending: true
  },
  // V3 target scenario — exercises the MMPROJ-bound (image-encoder) code path.
  'ocr-highmp': {
    title: 'OCR (high-MP)',
    tasks: ['ocrdoc'],
    metricSet: 'category',
    tolerance: 0.02,
    maxSide: 4096,
    fixturePending: true
  }
}
