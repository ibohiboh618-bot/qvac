#!/usr/bin/env node
'use strict'

const { spawnSync } = require('node:child_process')
const path = require('node:path')

const DEFAULT_ARGS = [
  '--models', 'qwen3.5-0.8b,qwen3.5-2b,qwen3.5-4b',
  '--quantization', 'Q4_K_M,Q6_K,Q8_0',
  '--sampling-preset', 'qvac-current,qwen-thinking-general,qwen-thinking-conservative,qwen-thinking-low-penalty,qwen-nonthinking-general',
  '--prompt-case', 'direct-qa,reasoning-heavy,structured-output,loop-stress,tool-run-cli,tool-web-search,tool-google-calendar,tool-google-drive,tool-github,tool-slack,tool-asana,tool-gmail,tool-google-sheets',
  '--prompts-file', './qwen-preset-prompts.json',
  '--ctx-size', '4096',
  '--batch-size', '512',
  '--ubatch-size', '128',
  '--flash-attn', 'off',
  '--cache-type-k', 'f16',
  '--cache-type-v', 'f16',
  '--threads', '4',
  '--repeats', '5',
  '--addon-source', 'npm',
  '--results-dir', './results/qwen3.5-scaled'
]

const result = spawnSync(process.execPath, [
  path.resolve(__dirname, 'run-param-sweep.js'),
  ...DEFAULT_ARGS,
  ...process.argv.slice(2)
], {
  cwd: __dirname,
  stdio: 'inherit',
  env: process.env
})

if (result.error) {
  console.error(result.error.message || String(result.error))
  process.exit(1)
}

if (typeof result.status === 'number' && result.status !== 0) process.exit(result.status)
if (result.signal) process.kill(process.pid, result.signal)
