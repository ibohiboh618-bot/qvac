'use strict'
// Desktop run driver — the process-level orchestration the in-process harness
// cannot do. The harness loads ONE build in ONE process, so warmup/measured round
// scheduling (and candidate-vs-baseline build comparison) has to happen out here
// by spawning one harness process per { source × model × block }.
//
// What this does:
//   • planBlocks() lays out 1 warmup + N measured blocks per source, INTERLEAVED
//     across sources (B,C,B,C…) so neither build sits on a hotter machine.
//   • stabilityGuard() waits for a steady thermal state between blocks; we emit a
//     [VLMBLOCK] marker recording what it did.
//   • each block is a fresh `bare` process pinned to one source/model/block via
//     env (QVAC_VLM_SOURCE_ID/REF, QVAC_VLM_MODEL_INDEX, QVAC_VLM_BLOCK,
//     QVAC_VLM_BACKENDS_DIR) so its markers carry the right identity and exactly
//     one build is loaded per process.
//
// The workflow's "Run VLM matrix" step now just runs this file, so all run-logic
// changes stay in this folder (no YAML churn). The report side takes the median
// over measured blocks (block >= 1) and drops warmup (block 0).
//
// addon@candidate / addon@baseline load builds staged by CI into
// builds/<id>/prebuilds; stagePrebuild() symlinks the live prebuilds dir at the
// right one before each block so the whole native build (binding + backends) is
// swapped. When both candidate and baseline point at the same staged build it is
// a genuine A/A test (expect ~0% deltas).
//
// --selfcheck validates the contract wiring without running any model.

const fs = require('fs')
const path = require('path')
const { spawn, spawnSync } = require('child_process')

function env (key, dflt) {
  const v = process.env[key]
  return v == null || v === '' ? dflt : v
}

function note (...args) {
  console.error('[run-desktop]', ...args)
}

// The harness loads from packages/llm-llamacpp; CI invokes us with that as cwd.
const WORKDIR = process.cwd()

// Resolve the models under test EXACTLY as the harness does, so the scheduler and
// the child agree on the list — we then pick one per child by index.
function resolveModels (config, parseModels) {
  const mode = env('QVAC_VLM_MODE', config.mode || 'two-models')
  if (mode === 'several-sources') return [config.sourcesModel]
  return parseModels(env('QVAC_VLM_MODELS', ''), config.catalog, config.models)
}

// Source identity → the version string stamped into markers (source_ref): the
// candidate's git sha, the baseline's pinned npm version, or the published addon.
function sourceRef (config, src) {
  if (src.type !== 'addon') return src.ref
  if (src.ref === 'baseline') return 'npm:' + ((config.defaultBaseline && config.defaultBaseline.npm) || 'baseline')
  if (src.ref === 'candidate') return env('QVAC_VLM_CANDIDATE_REF', 'git:candidate')
  return env('QVAC_VLM_SOURCE_REF', 'npm:' + env('ADDON_VERSION', 'published'))
}

function stabilityOptsFrom (config) {
  const want = (config.methodology && config.methodology.stability) || 'auto'
  // 'temp' (Mac mini sensor) is not wired yet → probe everywhere; 'off' disables.
  if (want === 'off') return { mode: 'off' }
  return { mode: 'probe' }
}

function emitBlock (config, entry, src, model, device, stability) {
  const obj = {
    v: 2,
    scenario: env('QVAC_VLM_SCENARIOS', config.defaultScenario || 'vqa-suite').split(',')[0].trim(),
    source_id: src.id,
    source_ref: sourceRef(config, src),
    model: model.label,
    device,
    block: entry.block,
    stability
  }
  process.stdout.write('[VLMBLOCK]' + JSON.stringify(obj) + '[/VLMBLOCK]\n')
}

// brittle -r generates test/integration/all.js referencing the matrix test; we
// then run that under bare once per block.
function generateRunner () {
  const r = spawnSync('npx', ['brittle', '-r', 'test/integration/all.js', 'benchmarks/vlm-benchmark/vlm-matrix.test.js'], { cwd: WORKDIR, stdio: 'inherit' })
  if (r.status !== 0) throw new Error('brittle runner generation failed (status ' + r.status + ')')
}

// Point the live prebuilds dir at the build this block must load. The harness
// loads BOTH its binding (require.addon from ./prebuilds) and the compute
// backends (backendsDir) from here, so swapping the dir swaps the whole build —
// candidate vs baseline end to end. A symlink keeps it cheap (no copying .bare
// files once per block). No-op for the published 'addon' source (loads in place).
function stagePrebuild (targetDir, workdir) {
  const live = path.join(workdir, 'prebuilds')
  if (path.resolve(live) === path.resolve(targetDir)) return
  if (!fs.existsSync(targetDir)) {
    throw new Error('prebuild not staged: ' + targetDir + ' — expected the candidate/baseline build on disk (check the prebuild-candidate job / baseline npm pack)')
  }
  try { fs.rmSync(live, { recursive: true, force: true }) } catch (_) {}
  fs.symlinkSync(path.resolve(targetDir), live, 'junction')
}

function runBlock (config, src, model, modelIndex, device, block, prebuildDir) {
  const childEnv = Object.assign({}, process.env, {
    QVAC_VLM_MATRIX: '1',
    QVAC_VLM_DEVICES: device,
    QVAC_VLM_REPEATS: '1', // one block = one pass over the fixture
    QVAC_VLM_BLOCK: String(block),
    QVAC_VLM_MODEL_INDEX: String(modelIndex),
    QVAC_VLM_SOURCE_ID: src.id,
    QVAC_VLM_SOURCE_REF: sourceRef(config, src),
    QVAC_VLM_BACKENDS_DIR: prebuildDir
  })
  return new Promise(resolve => {
    const child = spawn('bare', ['test/integration/all.js', '--exit'], { cwd: WORKDIR, stdio: 'inherit', env: childEnv })
    child.on('exit', code => resolve(code == null ? 1 : code))
    child.on('error', err => { note('failed to spawn bare:', (err && err.message) || err); resolve(1) })
  })
}

async function run () {
  const config = require('./config.cjs')
  const { parseModels } = require('./models.cjs')
  const { parseSources, addonPrebuildDir } = require('./sources.cjs')
  const { planBlocks, stabilityGuard } = require('./methodology.cjs')

  // The comparison axis depends on the mode:
  //   several-sources — ONE model across several sources. Candidate-vs-baseline
  //     lives here: addon@candidate vs addon@baseline are two builds of the same
  //     model. The addon-type sources are scheduled here (one prebuild each);
  //     fabric/upstream CLIs are built and run by the separate native-CLI step.
  //   two-models — two models compared on ONE source (the published addon). No
  //     build comparison, so the source axis is just the published addon.
  const mode = env('QVAC_VLM_MODE', config.mode || 'two-models')
  let sources
  if (mode === 'several-sources') {
    sources = parseSources(env('QVAC_VLM_SOURCES', 'addon')).filter(s => {
      if (s.type === 'addon') return true
      note(`skipping non-addon source '${s.id}' — CLI sources run via the several-sources native-CLI step`)
      return false
    })
    if (!sources.length) sources = parseSources('addon') // at least the published addon
  } else {
    sources = parseSources('addon')
  }

  const byId = new Map(sources.map(s => [s.id, s]))
  const sourceIds = sources.map(s => s.id)
  const devices = env('QVAC_VLM_DEVICES', 'cpu').split(',').map(s => s.trim()).filter(Boolean)
  const models = resolveModels(config, parseModels)
  const plan = planBlocks(sourceIds, config.methodology)
  const stabOpts = stabilityOptsFrom(config)

  note(`scheduling ${models.length} model(s) × ${devices.join(',')} × ${plan.length} block-run(s); sources: ${sourceIds.join(', ')}`)
  generateRunner()

  let failed = 0
  for (let mi = 0; mi < models.length; mi++) {
    for (const device of devices) {
      for (const entry of plan) {
        const src = byId.get(entry.source)
        const prebuildDir = addonPrebuildDir(src, WORKDIR)
        stagePrebuild(prebuildDir, WORKDIR)
        const stability = await stabilityGuard(stabOpts)
        emitBlock(config, entry, src, models[mi], device, stability)
        const code = await runBlock(config, src, models[mi], mi, device, entry.block, prebuildDir)
        if (code !== 0) {
          failed++
          note(`block FAILED (source=${src.id} model=${models[mi].label} device=${device} block=${entry.block} exit=${code})`)
        }
      }
    }
  }
  if (failed) {
    note(`${failed} block run(s) failed`)
    process.exit(1)
  }
  note('all block runs completed')
}

function selfcheck () {
  const config = require('./config.cjs')
  const { parseModels } = require('./models.cjs')
  const { parseSources } = require('./sources.cjs')
  const { planBlocks } = require('./methodology.cjs')
  const problems = []

  // config/scenario invariants
  if (!config.scenarios || !config.scenarios[config.defaultScenario]) problems.push('defaultScenario missing from scenarios')
  for (const name of config.defaultModels || []) {
    if (!config.catalog || !config.catalog[name]) problems.push(`defaultModels entry '${name}' not in catalog`)
  }

  // grammar round-trips
  parseModels('qwen3.5-f16,qwen3.5-q8', config.catalog, config.models)
  parseModels('t=https://huggingface.co/org/repo/resolve/0123456789012345678901234567890123456789/m.gguf|https://example.com/p.gguf@ctx=8192', config.catalog, [])
  parseSources('addon,addon@candidate,fabric@v8189.0.2,upstream@b8189')
  if (planBlocks(['a', 'b'], config.methodology).length < 2) problems.push('planBlocks produced no plan')

  // sample markers conform to the v2 contract
  // .txt, not .log — the package-level .gitignore swallows *.log
  const sample = fs.readFileSync(path.join(__dirname, 'markers-v2.sample.txt'), 'utf8')
  const need = ['v', 'scenario', 'source_id', 'source_ref', 'block', 'cell', 'model', 'device', 'task', 'metric']
  let rows = 0
  for (const line of sample.split(/\r?\n/)) {
    const m = line.match(/\[VLMROW\](.*?)\[\/VLMROW\]/)
    if (!m) continue
    rows++
    const r = JSON.parse(m[1])
    if (r.v === undefined) continue // a deliberate legacy (v1) row in the sample
    for (const k of need) if (!(k in r)) problems.push(`sample row ${rows} missing '${k}'`)
  }
  if (rows < 10) problems.push(`sample markers file has only ${rows} rows`)

  if (problems.length) {
    console.error('selfcheck FAILED:\n  - ' + problems.join('\n  - '))
    process.exit(1)
  }
  console.log(`selfcheck OK (${rows} sample rows, ${Object.keys(config.catalog).length} catalog models, ${Object.keys(config.scenarios).length} scenarios)`)
}

if (require.main === module) {
  if (process.argv.includes('--selfcheck')) selfcheck()
  else run().catch(err => { console.error('run-desktop failed:', (err && err.stack) || err); process.exit(1) })
}

module.exports = { selfcheck, run, stagePrebuild }
