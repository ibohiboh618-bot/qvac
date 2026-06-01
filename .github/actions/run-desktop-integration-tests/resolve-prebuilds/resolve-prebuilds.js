#!/usr/bin/env node
'use strict'

/**
 * Cross-platform prebuild resolver for the desktop integration test pipeline.
 *
 * Replaces the previous per-shell (bash + PowerShell) duplication in
 * action.yml with one Node implementation, so the move / npm-pack / rename
 * logic lives in a single place instead of two languages kept in sync.
 *
 * Modes (PREBUILD_MODE):
 *   - "artifact": move an already-downloaded build artifact from
 *     <staging-dir> into <workdir>/prebuilds
 *   - "package": `npm pack` PREBUILD_PACKAGE, extract it, and copy its
 *     prebuilds into <workdir>/prebuilds
 *
 * After either mode, applies an optional filename/dirname rename
 * (PREBUILD_RENAME = none | single-underscore | double-underscore).
 *
 * All inputs arrive via env so the action.yml call site stays trivial.
 */

const fs = require('fs')
const path = require('path')
const os = require('os')
const { execFileSync } = require('child_process')

const {
  PREBUILD_MODE,
  PREBUILD_WORKDIR,
  PREBUILD_PACKAGE = '',
  PREBUILD_RENAME = 'none',
  PREBUILD_STAGING_DIR = '',
  RUNNER_TEMP = os.tmpdir()
} = process.env

function log (msg) {
  console.log(`[resolve-prebuilds] ${msg}`)
}

function fail (msg) {
  console.error(`[resolve-prebuilds] ERROR: ${msg}`)
  process.exit(1)
}

if (!PREBUILD_WORKDIR) fail('PREBUILD_WORKDIR is required')

const workdir = path.resolve(PREBUILD_WORKDIR)
const prebuildsDir = path.join(workdir, 'prebuilds')

function ensureDir (dir) {
  fs.mkdirSync(dir, { recursive: true })
}

// Recursively copy contents of src into dest (merging into existing dest).
function copyContentsInto (src, dest) {
  ensureDir(dest)
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const from = path.join(src, entry.name)
    const to = path.join(dest, entry.name)
    if (entry.isDirectory()) {
      copyContentsInto(from, to)
    } else {
      fs.copyFileSync(from, to)
    }
  }
}

function resolveFromArtifact () {
  if (!PREBUILD_STAGING_DIR) fail('PREBUILD_STAGING_DIR is required in artifact mode')
  ensureDir(prebuildsDir)
  if (!fs.existsSync(PREBUILD_STAGING_DIR)) {
    log(`staging dir ${PREBUILD_STAGING_DIR} not found — nothing to move`)
    return
  }
  copyContentsInto(PREBUILD_STAGING_DIR, prebuildsDir)
  log('prebuilds moved from staging area')
}

function resolveFromPackage () {
  if (!/^@?[A-Za-z0-9._/~^>=<-]+$/.test(PREBUILD_PACKAGE)) {
    fail(`invalid prebuild-package spec: ${PREBUILD_PACKAGE}`)
  }
  const tmp = fs.mkdtempSync(path.join(RUNNER_TEMP, 'prebuild-'))
  log(`npm pack ${PREBUILD_PACKAGE} -> ${tmp}`)
  const packed = execFileSync(
    'npm',
    ['pack', PREBUILD_PACKAGE, '--pack-destination', tmp, '--json'],
    { encoding: 'utf8', shell: false }
  )
  const filename = JSON.parse(packed)[0].filename
  execFileSync('tar', ['-xzf', path.join(tmp, filename), '-C', tmp], { shell: false })
  const packagePrebuilds = path.join(tmp, 'package', 'prebuilds')
  if (!fs.existsSync(packagePrebuilds)) {
    fail('no prebuilds directory found in package')
  }
  copyContentsInto(packagePrebuilds, prebuildsDir)
  log('prebuilds extracted from package')
}

// Depth-first rename so both files and directories carrying the prefix are
// handled without invalidating parent paths mid-walk.
function renamePrebuilds (from, to) {
  if (!fs.existsSync(prebuildsDir)) return
  const walk = (dir) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name)
      if (entry.isDirectory()) walk(full)
      if (entry.name.startsWith(from)) {
        const renamed = path.join(dir, to + entry.name.slice(from.length))
        fs.renameSync(full, renamed)
      }
    }
  }
  walk(prebuildsDir)
  log(`renamed prebuilds (${from} -> ${to})`)
}

function main () {
  if (PREBUILD_MODE === 'package') {
    resolveFromPackage()
  } else if (PREBUILD_MODE === 'artifact') {
    resolveFromArtifact()
  } else {
    fail(`unknown PREBUILD_MODE: ${PREBUILD_MODE}`)
  }

  if (PREBUILD_RENAME === 'double-underscore') {
    renamePrebuilds('tetherto__', 'qvac__')
  } else if (PREBUILD_RENAME === 'single-underscore') {
    renamePrebuilds('tetherto_', 'qvac_')
  } else if (PREBUILD_RENAME !== 'none') {
    fail(`unknown PREBUILD_RENAME: ${PREBUILD_RENAME}`)
  }

  if (fs.existsSync(prebuildsDir)) {
    log(`final prebuilds contents under ${prebuildsDir}:`)
    for (const e of fs.readdirSync(prebuildsDir)) log(`  ${e}`)
  }
}

main()
