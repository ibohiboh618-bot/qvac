#!/usr/bin/env node
'use strict'

// CLI-based case runner (plain Node.js — NOT Bare).
//
// Same contract as case-runner.js but drives inference through a native
// llama-mtmd-cli binary instead of the JS addon. Used for the fabric
// and upstream legs of the 3-source comparison.
//
// Inputs via env (matching case-runner.js):
//   VLM_CASE_SPEC_PATH  — JSON with CaseSpec (see below)
//   VLM_RESULT_PATH     — where to write per-cell JSON
//
// CaseSpec additions for CLI mode:
//   cliBinaryPath       — absolute path to llama-mtmd-cli

const fs = require('fs')
const path = require('path')
const os = require('os')
const { spawnSync } = require('child_process')

const { scoreAnswer } = require('./accuracy')
const { parseStdoutMetrics } = require('./stdout-parser')
const { truncate } = require('./utils')

function readSpec (specPath) {
  return JSON.parse(fs.readFileSync(specPath, 'utf8'))
}

function sleep (ms) {
  const end = Date.now() + ms
  while (Date.now() < end) { /* busy-wait — no async in this runner */ }
}

function buildCliArgs (spec) {
  // Qwen3.5 /no_think convention: when thinking is disabled, prepend
  // /no_think to the prompt so the jinja template skips the <think>
  // block. This works on both fabric and upstream CLIs (unlike
  // --reasoning-budget which is addon-only).
  const prompt = spec.thinkingEnabled
    ? spec.prompt
    : `/no_think\n${spec.prompt}`

  const args = [
    '--model', spec.llmPath,
    '--mmproj', spec.mmprojPath,
    '--image', spec.imagePath,
    '--ctx-size', String(spec.ctxSize),
    '--predict', String(spec.nPredict),
    '--gpu-layers', spec.backend === 'cpu' ? '0' : '99',
    '--threads', String(os.cpus().length),
    '--temp', String(spec.temperature ?? 0),
    '--seed', String(spec.seed ?? 42),
    '--jinja',
    '-p', prompt
  ]

  return args
}

function extractGeneratedText (stdout) {
  if (!stdout) return ''
  // llama-mtmd-cli prints the generated text to stdout. Strip any
  // leading/trailing whitespace and control sequences.
  return stdout
    .replace(/\x1b\[[0-9;]*m/g, '')
    .trim()
}

function parseMaxRssFromTimeV (stderr) {
  // GNU /usr/bin/time -v outputs "Maximum resident set size (kbytes): NNN"
  const match = stderr.match(/Maximum resident set size \(kbytes\):\s*(\d+)/)
  if (match) return Number(match[1]) / 1024
  return null
}

function runOnceCli (spec) {
  const args = buildCliArgs(spec)
  const timeout = spec.perRunTimeoutMs || 5 * 60 * 1000

  // On Linux, wrap with /usr/bin/time -v to capture peak RSS.
  // On other platforms, skip RSS collection (no portable equivalent).
  const useTimeWrapper = os.platform() === 'linux' && fs.existsSync('/usr/bin/time')
  const spawnCmd = useTimeWrapper ? '/usr/bin/time' : spec.cliBinaryPath
  const spawnArgs = useTimeWrapper
    ? ['-v', spec.cliBinaryPath, ...args]
    : args

  const t0 = Date.now()
  const result = spawnSync(spawnCmd, spawnArgs, {
    encoding: 'utf8',
    timeout,
    env: { ...process.env },
    cwd: path.dirname(spec.cliBinaryPath)
  })
  const wallMs = Date.now() - t0

  if (result.error) {
    throw new Error(`CLI spawn failed: ${result.error.message}`)
  }
  if (result.status !== 0) {
    const tail = (result.stderr || '').trim().split('\n').slice(-10).join('\n')
    throw new Error(`CLI exited with status ${result.status}:\n${tail}`)
  }

  const text = extractGeneratedText(result.stdout)
  const stderr = result.stderr || ''
  const peakRssMb = useTimeWrapper ? parseMaxRssFromTimeV(stderr) : null

  return { text, wallMs, stderr, peakRssMb }
}

function main () {
  const specPath = process.env.VLM_CASE_SPEC_PATH
  const resultPath = process.env.VLM_RESULT_PATH
  if (!specPath || !resultPath) {
    console.error('VLM_CASE_SPEC_PATH and VLM_RESULT_PATH env vars are required')
    process.exit(2)
  }

  const spec = readSpec(specPath)

  if (!spec.cliBinaryPath || !fs.existsSync(spec.cliBinaryPath)) {
    console.error(`CLI binary not found: ${spec.cliBinaryPath}`)
    process.exit(2)
  }

  const cellStartedAt = new Date().toISOString()
  const errors = []
  const runs = []

  // Warmup runs
  for (let i = 0; i < (spec.warmupRuns || 0); i++) {
    console.log(`[BENCH_RUN_BEGIN warmup ${i}]`)
    try {
      runOnceCli(spec)
    } catch (e) {
      errors.push({ phase: 'warmup', index: i, message: String(e && e.message || e) })
    }
    console.log(`[BENCH_RUN_END warmup ${i}]`)
    if (spec.cooldownMs) sleep(spec.cooldownMs)
  }

  // Measured runs
  for (let i = 0; i < (spec.measuredRuns || 0); i++) {
    console.log(`[BENCH_RUN_BEGIN measured ${i}]`)
    try {
      const r = runOnceCli(spec)
      const stdoutMetrics = parseStdoutMetrics(r.stderr)
      const accuracy = scoreAnswer(r.text, spec.groundTruth)

      runs.push({
        index: i,
        ok: true,
        wallMs: r.wallMs,
        peakRssMb: r.peakRssMb,
        stats: null,
        stdoutMetrics,
        accuracy,
        fullAnswer: truncate(r.text, spec.answerTruncChars || 8000)
      })
    } catch (e) {
      runs.push({ index: i, ok: false, error: String(e && e.message || e) })
    }
    console.log(`[BENCH_RUN_END measured ${i}]`)
    if (spec.cooldownMs) sleep(spec.cooldownMs)
  }

  const out = {
    cell: {
      sourceKey: spec.sourceKey,
      sourceLabel: spec.sourceLabel,
      backend: spec.backend,
      platform: os.platform(),
      arch: os.arch()
    },
    startedAt: cellStartedAt,
    finishedAt: new Date().toISOString(),
    runs,
    errors,
    spec
  }

  fs.writeFileSync(resultPath, JSON.stringify(out, null, 2))
  console.log(`[cli-case-runner] wrote ${resultPath}`)
}

main()
