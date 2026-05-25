#!/usr/bin/env node
'use strict'

// Orchestrator for the 3-source VLM benchmark.
//
// Sequence:
//   1. Parse CLI args + load config.
//   2. Resolve model (single model shared by all sources).
//   3. Resolve sources (addon, fabric-cli, upstream-cli).
//   4. For each (source, backend):
//        addon  → spawn `bare case-runner.js`
//        cli    → spawn `node cli-case-runner.js`
//   5. Write reports (full matrix + delta MD, plus raw JSON).

const fs = require('fs')
const path = require('path')
const os = require('os')
const { spawnSync } = require('child_process')

const config = require('./vlm-bench.config')
const { parseArgs, csvOrArray } = require('./utils')
const { resolveSources } = require('./source-resolver')
const { buildSummary, writeReports } = require('./reporters')
const { parseStdoutMetrics } = require('./stdout-parser')
const { detectAll, hasUsableGpu } = require('./hardware')

function safeExecStr (cmd, args) {
  try { return require('child_process').execFileSync(cmd, args, { encoding: 'utf8', timeout: 5000, stdio: ['ignore', 'pipe', 'ignore'] }).trim() } catch { return null }
}

function safeReadJson (filePath) {
  try { return JSON.parse(fs.readFileSync(filePath, 'utf8')) } catch { return null }
}

function detectAddonProvenance () {
  const pkgPath = path.join(SCRIPT_DIR, 'node_modules', '@qvac', 'llm-llamacpp', 'package.json')
  const pkg = safeReadJson(pkgPath)
  if (!pkg) return null
  const addonRoot = path.dirname(pkgPath)
  const prebuildDir = path.join(addonRoot, 'prebuilds', `${process.platform}-${process.arch}`)
  let prebuildFile = null
  let prebuildSize = null
  if (fs.existsSync(prebuildDir)) {
    for (const f of fs.readdirSync(prebuildDir)) {
      if (f.endsWith('.bare') || f.endsWith('.node')) {
        const full = path.join(prebuildDir, f)
        try {
          const s = fs.statSync(full)
          if (!prebuildSize || s.size > prebuildSize) {
            prebuildFile = full
            prebuildSize = s.size
          }
        } catch {}
      }
    }
  }
  return {
    name: pkg.name,
    version: pkg.version,
    installedAt: addonRoot,
    prebuildFile,
    prebuildSizeMb: prebuildSize ? Math.round((prebuildSize / (1024 * 1024)) * 100) / 100 : null
  }
}

function detectBareProvenance () {
  const localBareBin = path.join(SCRIPT_DIR, 'node_modules', 'bare', 'bin', 'bare')
  if (fs.existsSync(localBareBin)) {
    const ver = safeExecStr(process.execPath, [localBareBin, '--version'])
    return { source: 'local', binary: localBareBin, version: ver }
  }
  const ver = safeExecStr('bare', ['--version'])
  return { source: 'global', binary: 'bare', version: ver }
}

function detectGitProvenance () {
  const sha = safeExecStr('git', ['-C', SCRIPT_DIR, 'rev-parse', 'HEAD'])
  if (!sha) return null
  return {
    sha,
    shortSha: sha.slice(0, 8),
    branch: safeExecStr('git', ['-C', SCRIPT_DIR, 'rev-parse', '--abbrev-ref', 'HEAD']),
    title: safeExecStr('git', ['-C', SCRIPT_DIR, 'log', '-1', '--pretty=%s']),
    date: safeExecStr('git', ['-C', SCRIPT_DIR, 'log', '-1', '--pretty=%cI'])
  }
}

const SCRIPT_DIR = __dirname
const RESOLVED_MODELS_PATH = path.join(SCRIPT_DIR, 'resolved-models.json')

function log (...args) { console.log('[run-vlm-bench]', ...args) }

function parseBooleanFlag (raw) {
  if (raw === true) return true
  if (raw === false) return false
  if (raw == null) return null
  const s = String(raw).trim().toLowerCase()
  if (['on', 'true', 'yes', '1', 'enabled', 'enable'].includes(s)) return true
  if (['off', 'false', 'no', '0', 'disabled', 'disable'].includes(s)) return false
  return null
}

function resolveThinkingFlag (args, cfg) {
  if ('enable-thinking' in args && args['enable-thinking'] !== false) return true
  if ('disable-thinking' in args && args['disable-thinking'] !== false) return false
  if ('thinking' in args) {
    const parsed = parseBooleanFlag(args.thinking)
    if (parsed != null) return parsed
    throw new Error(`Invalid --thinking value: ${args.thinking}. Use on/off.`)
  }
  return Boolean(cfg.thinking && cfg.thinking.enabled)
}

function detectPlatformKey () {
  const p = os.platform()
  const a = os.arch()
  if (p === 'darwin' && a === 'arm64') return 'macos-arm64'
  if (p === 'win32' && a === 'x64') return 'windows-x64'
  if (p === 'linux' && a === 'x64') return 'linux-x64'
  return `${p}-${a}`
}

function pickBackends (args, platformKey, hardwareInfo) {
  const fromCli = csvOrArray(args.backends || args.backend)
  let backends
  if (fromCli.length) {
    backends = fromCli
  } else {
    const platform = config.platforms[platformKey]
    backends = platform && Array.isArray(platform.backends) ? platform.backends.slice() : ['gpu']
  }
  if (!args['force-gpu-row'] && !hasUsableGpu(hardwareInfo)) {
    backends = backends.filter((b) => b !== 'gpu')
  }
  return backends
}

function ensureModelsResolved (args) {
  let cached = null
  if (fs.existsSync(RESOLVED_MODELS_PATH)) {
    try { cached = JSON.parse(fs.readFileSync(RESOLVED_MODELS_PATH, 'utf8')) } catch {}
  }
  if (cached && !args['force-prepare']) return cached

  log('running prepare-models.js')
  const passThrough = []
  if (args['local-model']) passThrough.push('--local-model', args['local-model'])
  if (args['local-mmproj']) passThrough.push('--local-mmproj', args['local-mmproj'])
  const r = spawnSync(process.execPath, [path.join(SCRIPT_DIR, 'prepare-models.js'), ...passThrough], {
    cwd: SCRIPT_DIR,
    stdio: 'inherit',
    env: process.env
  })
  if (r.status !== 0) {
    throw new Error(`prepare-models.js failed with status ${r.status}`)
  }
  return JSON.parse(fs.readFileSync(RESOLVED_MODELS_PATH, 'utf8'))
}

function buildCaseSpec ({ source, backend, model, imagePath, runArgs }) {
  const base = {
    sourceKey: source.key,
    sourceLabel: source.label,
    sourceType: source.type,
    backend,
    llmPath: model.llmPath,
    mmprojPath: model.mmprojPath,
    imagePath,
    prompt: config.case.prompt,
    ctxSize: config.model.ctxSize,
    nPredict: config.model.nPredict,
    temperature: config.sampling.temperature,
    seed: config.sampling.seed,
    thinkingEnabled: runArgs.thinkingEnabled,
    warmupRuns: runArgs.warmupRuns,
    measuredRuns: runArgs.measuredRuns,
    cooldownMs: runArgs.cooldownMs,
    perRunTimeoutMs: config.run.perRunTimeoutMs,
    groundTruth: config.case.groundTruth,
    answerTruncChars: config.reporting.answerTruncChars
  }
  if (source.type === 'addon') {
    base.addonRequirePath = source.addonPath || 'local'
  } else if (source.type === 'cli') {
    base.cliBinaryPath = source.binaryPath
  }
  return base
}

function resolveLocalBare () {
  const localBin = path.join(SCRIPT_DIR, 'node_modules', 'bare', 'bin', 'bare')
  if (fs.existsSync(localBin)) return localBin
  return null
}

function runAddonCell (spec, resultsDir, cellIdx) {
  const specPath = path.join(resultsDir, `cell-${cellIdx}-spec.json`)
  const resultPath = path.join(resultsDir, `cell-${cellIdx}-result.json`)
  fs.writeFileSync(specPath, JSON.stringify(spec, null, 2))

  const env = {
    ...process.env,
    VLM_CASE_SPEC_PATH: specPath,
    VLM_RESULT_PATH: resultPath
  }

  const localBare = resolveLocalBare()
  log(`spawning bare case-runner: ${spec.sourceLabel} / ${spec.backend}`)
  const spawnArgs = localBare
    ? [process.execPath, [localBare, path.join(SCRIPT_DIR, 'case-runner.js')]]
    : ['bare', [path.join(SCRIPT_DIR, 'case-runner.js')]]
  const r = spawnSync(spawnArgs[0], spawnArgs[1], {
    cwd: SCRIPT_DIR,
    env,
    shell: !localBare && process.platform === 'win32',
    encoding: 'utf8'
  })
  const stdoutBuf = r.stdout || ''
  const stderrBuf = r.stderr || ''
  if (stdoutBuf) process.stdout.write(stdoutBuf)
  if (stderrBuf) process.stderr.write(stderrBuf)

  const fullLog = stderrBuf + '\n--- stdout ---\n' + stdoutBuf
  const logPath = path.join(resultsDir, `cell-${cellIdx}-stderr.log`)
  fs.writeFileSync(logPath, fullLog)

  if (r.error) throw new Error(`spawn failed: ${r.error.message}`)
  if (r.status !== 0) {
    const allLines = fullLog.trim().split('\n')
    const errorLines = allLines
      .filter((ln) => /error|fatal|missing|not found|cannot|enoent|module_not_found|throw|throw new|prebuilds?/i.test(ln))
      .slice(-10)
    const tail = errorLines.length ? errorLines.join('\n  ') : allLines.slice(-10).join('\n  ')
    throw new Error(`case-runner exited with status ${r.status} for ${spec.sourceLabel} / ${spec.backend}. Likely error lines:\n  ${tail}\nFull log: ${logPath}`)
  }

  const cell = JSON.parse(fs.readFileSync(resultPath, 'utf8'))
  mergeHostStdoutMetrics(cell, fullLog)
  return cell
}

function runCliCell (spec, resultsDir, cellIdx) {
  const specPath = path.join(resultsDir, `cell-${cellIdx}-spec.json`)
  const resultPath = path.join(resultsDir, `cell-${cellIdx}-result.json`)
  fs.writeFileSync(specPath, JSON.stringify(spec, null, 2))

  const env = {
    ...process.env,
    VLM_CASE_SPEC_PATH: specPath,
    VLM_RESULT_PATH: resultPath
  }

  log(`spawning cli-case-runner: ${spec.sourceLabel} / ${spec.backend}`)
  const r = spawnSync(process.execPath, [path.join(SCRIPT_DIR, 'cli-case-runner.js')], {
    cwd: SCRIPT_DIR,
    env,
    encoding: 'utf8'
  })
  const stdoutBuf = r.stdout || ''
  const stderrBuf = r.stderr || ''
  if (stdoutBuf) process.stdout.write(stdoutBuf)
  if (stderrBuf) process.stderr.write(stderrBuf)

  const fullLog = stderrBuf + '\n--- stdout ---\n' + stdoutBuf
  const logPath = path.join(resultsDir, `cell-${cellIdx}-stderr.log`)
  fs.writeFileSync(logPath, fullLog)

  if (r.error) throw new Error(`spawn failed: ${r.error.message}`)
  if (r.status !== 0) {
    const allLines = fullLog.trim().split('\n')
    const tail = allLines.slice(-10).join('\n  ')
    throw new Error(`cli-case-runner exited with status ${r.status} for ${spec.sourceLabel} / ${spec.backend}.\n  ${tail}\nFull log: ${logPath}`)
  }

  const cell = JSON.parse(fs.readFileSync(resultPath, 'utf8'))
  mergeHostStdoutMetrics(cell, fullLog)
  return cell
}

function runOneCell (spec, resultsDir, cellIdx) {
  if (spec.sourceType === 'cli') {
    return runCliCell(spec, resultsDir, cellIdx)
  }
  return runAddonCell(spec, resultsDir, cellIdx)
}

function mergeHostStdoutMetrics (cell, fullLog) {
  const sessionMetrics = parseStdoutMetrics(fullLog)

  const segmentRegex = /\[BENCH_RUN_BEGIN (warmup|measured) (\d+)\]([\s\S]*?)\[BENCH_RUN_END \1 \2\]/g
  const perRunMetrics = new Map()
  let m
  while ((m = segmentRegex.exec(fullLog)) !== null) {
    const phase = m[1]
    if (phase !== 'measured') continue
    perRunMetrics.set(Number(m[2]), parseStdoutMetrics(m[3]))
  }

  if (!cell.runs) return
  for (const run of cell.runs) {
    if (!run.ok) continue
    const segmentMetrics = perRunMetrics.get(run.index) || {}
    run.stdoutMetrics = {
      ...(run.stdoutMetrics || {}),
      ...(sessionMetrics.visionEncodeMs != null ? { visionEncodeMs: sessionMetrics.visionEncodeMs } : {}),
      ...segmentMetrics
    }
  }
}

async function main () {
  const args = parseArgs(process.argv.slice(2))

  const runArgs = {
    warmupRuns: Number(args['warmup-runs'] ?? config.run.warmupRuns),
    measuredRuns: Number(args['measured-runs'] ?? config.run.measuredRuns),
    cooldownMs: Number(args['cooldown-ms'] ?? config.run.cooldownMs),
    thinkingEnabled: resolveThinkingFlag(args, config)
  }

  const resolved = ensureModelsResolved(args)
  const platformKey = detectPlatformKey()
  const hardwareInfo = detectAll()
  log(`hardware: cpu="${hardwareInfo.cpu.model}" cores=${hardwareInfo.cpu.cores} ram=${hardwareInfo.ram.totalGb}GB gpus=${hardwareInfo.gpus.length}`)
  for (const g of hardwareInfo.gpus) log(`  GPU: ${g.vendor || ''} ${g.model || '?'} ${g.memoryMb ? `(${g.memoryMb}MB)` : ''}`)
  const backends = pickBackends(args, platformKey, hardwareInfo)
  if (backends.length === 0) {
    throw new Error('No backends resolved for this host (--force-gpu-row may help).')
  }
  log(`backends to run: ${backends.join(', ')}`)

  // Resolve the 3 inference sources
  const sources = resolveSources(config, args)
  const runnableSources = sources.filter((s) => {
    if (s.type === 'cli' && s.requiresBuild) {
      log(`skipping ${s.key}: CLI binary not built (run scripts/build-cli-sources.js first)`)
      return false
    }
    if (s.type === 'cli' && !s.binaryPath) {
      log(`skipping ${s.key}: no binary path resolved`)
      return false
    }
    return true
  })

  if (runnableSources.length === 0) {
    throw new Error('No runnable sources. Ensure at least one source is available.')
  }
  log(`sources: ${runnableSources.map((s) => `${s.key}(${s.type})`).join(', ')}`)

  const imagePath = path.resolve(SCRIPT_DIR, config.case.image)
  if (!fs.existsSync(imagePath)) {
    throw new Error(`Image not found: ${imagePath}. Copy seven_objects.jpg into assets/`)
  }

  const resultsDir = path.resolve(SCRIPT_DIR, args['results-dir'] || config.reporting.resultsDir)
  fs.mkdirSync(resultsDir, { recursive: true })

  // Interleaved methodology: cycle through all (source, backend) pairs
  // once per measured iteration, with cooldown between rounds. This
  // reduces thermal/frequency bias compared to running all iterations
  // for one source before moving to the next.
  //
  // Round-robin order per iteration:
  //   iteration 1: addon/cpu, fabric/cpu, upstream/cpu, [cooldown]
  //   iteration 2: addon/cpu, fabric/cpu, upstream/cpu, [cooldown]
  //   ...
  //
  // Each spawn gets warmupRuns on the first iteration only, measuredRuns=1.
  // Results are merged into one cell per (source, backend) at the end.

  const cellPairs = []
  for (const source of runnableSources) {
    for (const backend of backends) {
      cellPairs.push({ source, backend })
    }
  }

  // Accumulate per-iteration single-run results keyed by "sourceKey|backend"
  const accumulated = new Map()
  for (const { source, backend } of cellPairs) {
    accumulated.set(`${source.key}|${backend}`, {
      cell: {
        sourceKey: source.key,
        sourceLabel: source.label,
        backend,
        platform: os.platform(),
        arch: os.arch()
      },
      runs: [],
      errors: [],
      spec: null
    })
  }

  let cellIdx = 0
  const totalIterations = runArgs.warmupRuns + runArgs.measuredRuns

  for (let iter = 0; iter < totalIterations; iter++) {
    const isWarmup = iter < runArgs.warmupRuns
    const phase = isWarmup ? 'warmup' : 'measured'
    const iterNum = isWarmup ? iter : iter - runArgs.warmupRuns
    log(`--- ${phase} iteration ${iterNum} (${cellPairs.length} cells) ---`)

    for (const { source, backend } of cellPairs) {
      const spec = buildCaseSpec({
        source, backend, model: resolved, imagePath,
        runArgs: { ...runArgs, warmupRuns: 0, measuredRuns: 1 }
      })
      const accKey = `${source.key}|${backend}`
      const acc = accumulated.get(accKey)
      if (!acc.spec) acc.spec = spec

      try {
        const cell = runOneCell(spec, resultsDir, cellIdx++)
        if (!isWarmup && cell.runs) {
          for (const run of cell.runs) {
            if (run.ok) run.index = acc.runs.filter((r) => r.ok).length
            acc.runs.push(run)
          }
        }
        if (cell.errors) acc.errors.push(...cell.errors)
      } catch (e) {
        log(`cell failed (${phase} ${iterNum}): ${e.message}`)
        if (!isWarmup) {
          acc.errors.push({ phase: 'spawn', message: e.message })
        }
      }
    }

    if (runArgs.cooldownMs > 0) {
      log(`cooldown ${runArgs.cooldownMs}ms`)
      const end = Date.now() + runArgs.cooldownMs
      while (Date.now() < end) { /* busy-wait */ }
    }
  }

  const cells = Array.from(accumulated.values())

  const summary = buildSummary(cells)
  const meta = {
    modelId: config.model.id,
    modelLabel: config.model.label,
    modelQuant: config.model.quant,
    image: config.case.image,
    prompt: config.case.prompt,
    groundTruth: config.case.groundTruth.map((g) => g.canonical),
    groundTruthCount: config.case.groundTruth.length,
    warmupRuns: runArgs.warmupRuns,
    measuredRuns: runArgs.measuredRuns,
    thinkingEnabled: runArgs.thinkingEnabled,
    generatedAt: new Date().toISOString(),
    platformKey,
    backends,
    hardware: hardwareInfo,
    methodology: 'interleaved',
    comparisonMode: config.comparisonMode || (runnableSources.length > 1 ? 'source-engines' : 'none'),
    sources: runnableSources.map((s) => ({
      key: s.key,
      type: s.type,
      label: s.label,
      commitSha: s.commitSha || null,
      provenance: s.provenance || null
    })),
    model: {
      label: resolved.label,
      quant: resolved.quant,
      hfRepo: resolved.hfRepo,
      hfRevision: resolved.hfRevision,
      provenance: resolved.provenance || null
    },
    software: {
      addon: detectAddonProvenance(),
      bare: detectBareProvenance(),
      git: detectGitProvenance(),
      node: process.version
    }
  }
  const written = writeReports({ outputDir: resultsDir, summary, meta })

  log('done.')
  log(`  full matrix: ${written.matrixPath}`)
  log(`  delta:       ${written.deltaPath}`)
  log(`  raw JSON:    ${written.jsonPath}`)
}

main().catch((err) => {
  console.error(`[run-vlm-bench] failed: ${err && err.message ? err.message : String(err)}`)
  process.exit(1)
})
