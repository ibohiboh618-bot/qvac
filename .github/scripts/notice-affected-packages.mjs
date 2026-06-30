#!/usr/bin/env node
'use strict'

// Resolve which Tier-1 in-scope packages a pull request actually affects, and
// emit them as a JSON array for the notice-drift matrix (QVAC-21558).
//
// Why: regenerating every in-scope NOTICE on every PR is slow and would fail a
// PR on pre-existing drift it did not introduce. Instead we only re-check the
// packages whose dependency manifests changed — falling back to the FULL set
// when a change can affect every package's generated NOTICE (the generator
// itself, this workflow/script, or the shared model registry).
//
// Inputs (environment):
//   GITHUB_EVENT_NAME  - 'pull_request' | 'workflow_dispatch' (auto-set)
//   GITHUB_REPOSITORY  - 'owner/repo' (auto-set)
//   PR_NUMBER          - pull request number (set by the workflow)
//   GH_TOKEN           - token for the PR files API (github.token is enough)
//   GITHUB_OUTPUT      - step output file (auto-set)
//
// Output: writes `packages=<json-array>` to $GITHUB_OUTPUT (and logs to stderr).

import fs from 'node:fs'

// PROVISIONAL Tier-1 shipping scope (C1 "ships to end users" / C4
// "signs/publishes releases"). Authoritative list: docs/devops/TIER-1-SCOPE.md
// (QVAC-19052), still DRAFT. Keep this in sync with that doc before the gate is
// promoted to a required check.
const IN_SCOPE = [
  'sdk',
  'bare-sdk',
  'cli',
  'rag',
  'ai-sdk-provider',
  'registry-server/client',
  'llm-llamacpp',
  'embed-llamacpp',
  'transcription-whispercpp',
  'transcription-parakeet',
  'tts-onnx',
  'tts-ggml',
  'ocr-onnx',
  'ocr-ggml',
  'diffusion-cpp',
  'translation-nmtcpp',
  'classification-ggml',
  'vla-ggml',
  'bci-whispercpp',
  'decoder-audio'
]

// A change to any of these can alter every package's generated NOTICE, so it
// forces a full re-check.
const FULL_TRIGGERS = [
  '.cursor/skills/qv-notice-generate/',
  '.github/workflows/pr-validation-notice.yml',
  '.github/scripts/notice-affected-packages.mjs',
  'packages/registry-server/data/models.prod.json'
]

// Per-package manifest files whose changes can change a NOTICE.
const MANIFEST_RE =
  /\/(package\.json|package-lock\.json|vcpkg\.json|requirements[^/]*\.txt|pyproject\.toml|NOTICE)$/

function packageOf (file) {
  if (!file.startsWith('packages/')) return null
  const seg = file.slice('packages/'.length).split('/')
  // registry-server has sub-packages (client, shared); only some are in scope.
  if (seg[0] === 'registry-server') return seg[1] ? `registry-server/${seg[1]}` : null
  return seg[0] || null
}

async function fetchChangedFiles () {
  const repo = process.env.GITHUB_REPOSITORY
  const pr = process.env.PR_NUMBER
  const token = process.env.GH_TOKEN
  if (!repo || !pr) {
    throw new Error('PR_NUMBER / GITHUB_REPOSITORY not set; cannot list changed files')
  }

  const headers = {
    Accept: 'application/vnd.github+json',
    'User-Agent': 'qvac-notice-drift'
  }
  if (token) headers.Authorization = `Bearer ${token}`

  const files = []
  // Cap pagination defensively (3000 files is far beyond any real PR).
  for (let page = 1; page <= 30; page++) {
    const url = `https://api.github.com/repos/${repo}/pulls/${pr}/files?per_page=100&page=${page}`
    const res = await fetch(url, { headers })
    if (!res.ok) throw new Error(`GitHub API ${res.status} listing PR files`)
    const batch = await res.json()
    for (const f of batch) files.push(f.filename)
    if (batch.length < 100) break
  }
  return files
}

function selectPackages (files) {
  // files === null => full check (e.g. manual dispatch).
  if (files === null) return [...IN_SCOPE]
  if (files.some(f => FULL_TRIGGERS.some(t => f === t || f.startsWith(t)))) {
    return [...IN_SCOPE]
  }
  const affected = new Set()
  for (const f of files) {
    if (!MANIFEST_RE.test(f)) continue
    const pkg = packageOf(f)
    if (pkg && IN_SCOPE.includes(pkg)) affected.add(pkg)
  }
  return [...affected].sort()
}

async function main () {
  const event = process.env.GITHUB_EVENT_NAME
  const files = event === 'workflow_dispatch' ? null : await fetchChangedFiles()
  const packages = selectPackages(files)

  const json = JSON.stringify(packages)
  console.error(
    `event=${event || 'unknown'} affected in-scope packages (${packages.length}): ${json}`
  )

  const out = process.env.GITHUB_OUTPUT
  if (out) fs.appendFileSync(out, `packages=${json}\n`)
  else console.log(json)
}

main().catch(err => {
  console.error(err)
  process.exit(1)
})
