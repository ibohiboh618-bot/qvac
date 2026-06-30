'use strict'
// QVAC-19178: stage the VLM benchmark into the locations the mobile test framework
// scans and bundles. The source of truth is THIS directory; the copies written here
// are git-ignored. Run before the mobile build (the reusable mobile workflow's
// `pre_build_script` input points at this file). Desktop needs NO staging — its CI
// leg points brittle straight at benchmarks/vlm-benchmark/vlm-matrix.test.js.
//
// The mobile generator scans test/integration/*.test.js (→ runVlmMatrixKv*Test) and the
// app bundles test/mobile/testAssets/. Because both test/integration and this dir are
// 2 levels under packages/llm-llamacpp, the staged harness's ../../-relative requires
// resolve to the same files; ./ requires resolve against the staged siblings.
const fs = require('fs')
const path = require('path')

const HERE = __dirname
const INTEG = path.resolve(HERE, '..', '..', 'test', 'integration')
const ASSETS = path.resolve(HERE, '..', '..', 'test', 'mobile', 'testAssets')
const IMAGES = path.join(HERE, 'images')

// The entries + the modules they require must sit together in test/integration so each
// entry's `./harness.cjs` / harness's `./config.cjs` `./fixture.data.cjs`
// `./stdout-parser` (QVAC-21318: KV-cache-size parse) resolve for the mobile bundler.
// QVAC-21318: mobile stages the PER-CELL entries (one KV cell each) instead of the
// combined vlm-matrix.test.js, so each cell becomes its own mobile function → test-group
// → Device Farm run/process. That isolates a native abort in one cell (k8vf16 on Adreno
// OpenCL) from the others. Desktop is untouched — it runs vlm-matrix.test.js in place.
const CODE = [
  'vlm-matrix-kv-f16.test.js', 'vlm-matrix-kv-kf16v8.test.js', 'vlm-matrix-kv-k8vf16.test.js',
  'harness.cjs', 'config.cjs', 'fixture.data.cjs', 'stdout-parser.js'
]

fs.mkdirSync(INTEG, { recursive: true })
fs.mkdirSync(ASSETS, { recursive: true })

for (const f of CODE) {
  fs.copyFileSync(path.join(HERE, f), path.join(INTEG, f))
  console.log(`staged -> test/integration/${f}`)
}
// Images aren't in git — CI syncs them from the fixture object store (URI configured
// in the benchmark workflow) into images/ before this runs. Fail loudly if skipped.
const imgs = fs.existsSync(IMAGES) ? fs.readdirSync(IMAGES).filter(f => /\.(png|jpe?g|webp|gif)$/i.test(f)) : []
if (!imgs.length) throw new Error(`No images in ${IMAGES} — sync the fixture image store into it first`)
for (const f of imgs) fs.copyFileSync(path.join(IMAGES, f), path.join(ASSETS, f))
console.log(`staged ${imgs.length} images -> test/mobile/testAssets`)

// Register each per-cell test in test-groups.json so the mobile generator's per-platform
// grouping validation passes (it requires EVERY test to be in a group, on every
// platform). QVAC-21318: ONE group per KV cell — the workflow's `test_groups` override
// lists all three, so upload-to-devicefarm schedules one Device Farm run (process) per
// cell, isolating a native abort. This edit is ephemeral (the file is pristine on a
// fresh checkout).
const CELL_GROUPS = {
  vlmMatrixF16: ['runVlmMatrixKvF16Test'],
  vlmMatrixKf16v8: ['runVlmMatrixKvKf16v8Test'],
  vlmMatrixK8vf16: ['runVlmMatrixKvK8vf16Test']
}
const GROUPS = path.resolve(HERE, '..', '..', 'test', 'mobile', 'test-groups.json')
if (fs.existsSync(GROUPS)) {
  const groups = JSON.parse(fs.readFileSync(GROUPS, 'utf8'))
  for (const platform of Object.keys(groups)) Object.assign(groups[platform], CELL_GROUPS)
  fs.writeFileSync(GROUPS, JSON.stringify(groups, null, 2) + '\n')
  console.log(`registered ${Object.keys(CELL_GROUPS).join(', ')} in test-groups.json (${Object.keys(groups).join(', ')})`)
}
