#!/usr/bin/env node
'use strict'

// Run with `node`, not `bare`: this script is a build-time helper that uses
// node's built-in `fs` / `path` (same convention as the sibling
// validate-mobile-tests.js).  Everything inside the addon itself runs under
// `bare` and uses bare-fs / bare-path instead.

const fs = require('fs')
const path = require('path')

const repoRoot = path.resolve(__dirname, '..')
const integrationDir = path.join(repoRoot, 'test', 'integration')
const mobileDir = path.join(repoRoot, 'test', 'mobile')
const outputFile = path.join(mobileDir, 'integration.auto.cjs')

// DEBUG (QVAC-20557 Mali GPU correctness diagnostic, DO-NOT-MERGE): restrict the
// ON-DEVICE mobile bundle to the GPU correctness test so BOTH device-farm
// flagships (Pixel 9 / Mali + S25 / Adreno) finish within the farm timeout.
// Round-2 (#2723) timed out on the S25/Adreno leg before reaching gpu-smoke,
// leaving NO Adreno control. Source tests under test/integration/ are UNTOUCHED
// and desktop integration CI still runs them all; this only narrows the mobile
// bundle. validate-mobile-tests.js carries the SAME constant so the two agree.
// Set to null to restore the full mobile bundle (this branch is never merged).
const MOBILE_DIAG_SUBSET = ['gpu-smoke.test.js']

function getIntegrationFiles () {
  if (!fs.existsSync(integrationDir)) {
    throw new Error(`Integration directory not found: ${integrationDir}`)
  }

  let files = fs.readdirSync(integrationDir)
    .filter(entry => entry.endsWith('.test.js'))
    .sort()
  if (MOBILE_DIAG_SUBSET) {
    const allow = new Set(MOBILE_DIAG_SUBSET)
    files = files.filter(entry => allow.has(entry))
  }
  return files
}

function toFunctionName (fileName) {
  const base = fileName.replace(/\.js$/, '')
  const parts = base.split(/[^a-zA-Z0-9]+/).filter(Boolean)
  const suffix = parts.map(part => part.charAt(0).toUpperCase() + part.slice(1)).join('')
  return `run${suffix}`
}

function buildFileContents (files) {
  const lines = []
  const functionNames = files.map(toFunctionName)
  lines.push("'use strict'")
  lines.push("require('./integration-runtime.cjs')")
  lines.push('')
  lines.push('// AUTO-GENERATED FILE. Run `npm run test:mobile:generate` to update.')
  lines.push('// Each function mirrors a single file under test/integration/.')
  lines.push('')
  lines.push('/* global runIntegrationModule */')
  lines.push('')

  for (let i = 0; i < files.length; i++) {
    const file = files[i]
    const fnName = functionNames[i]
    const relativePath = `../integration/${file}`
    lines.push(`async function ${fnName} (options = {}) { // eslint-disable-line no-unused-vars`)
    lines.push(`  return runIntegrationModule('${relativePath}', options)`)
    lines.push('}')
    if (i < files.length - 1) {
      lines.push('')
    }
  }

  lines.push('')
  lines.push('module.exports = {')
  for (let i = 0; i < functionNames.length; i++) {
    const suffix = i < functionNames.length - 1 ? ',' : ''
    lines.push(`  ${functionNames[i]}${suffix}`)
  }
  lines.push('}')

  return `${lines.join('\n')}\n`
}

function main () {
  const files = getIntegrationFiles()
  if (files.length === 0) {
    throw new Error(`No integration test files found inside ${integrationDir}`)
  }

  const content = buildFileContents(files)
  fs.writeFileSync(outputFile, content, 'utf8')
  console.log(`Generated ${outputFile} with ${files.length} integration runners.`)
}

if (require.main === module) {
  main()
}
