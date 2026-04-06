#!/usr/bin/env node
'use strict'

/**
 * TAP-to-JSON converter for brittle test output.
 *
 * Parses TAP output and writes a structured JSON report — the portable
 * intermediate format consumed by render-report.js and upload-to-qase.js.
 *
 * Usage:
 *   node tap-to-json.js [options]
 *
 * Options:
 *   --input, -i   <path>   TAP input file  (default: stdin)
 *   --output, -o  <path>   JSON output file (default: stdout)
 *   --suite, -s   <name>   Suite name       (default: "brittle")
 *
 * JSON schema (same format mobile can produce directly):
 *   {
 *     "suite": "llamacpp-llm",
 *     "timestamp": "2026-04-06T12:00:00.000Z",
 *     "summary": { "total": 5, "passed": 4, "failed": 1, "skipped": 0 },
 *     "results": [
 *       {
 *         "name": "test case name",
 *         "status": "passed",
 *         "assertions": 4,
 *         "failures": [],
 *         "comments": []
 *       }
 *     ]
 *   }
 */

const fs = require('fs')
const path = require('path')

function parseArgs (argv) {
  const args = { suite: 'brittle', input: null, output: null }
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i]
    if ((arg === '--input' || arg === '-i') && argv[i + 1]) args.input = argv[++i]
    else if ((arg === '--output' || arg === '-o') && argv[i + 1]) args.output = argv[++i]
    else if ((arg === '--suite' || arg === '-s') && argv[i + 1]) args.suite = argv[++i]
  }
  return args
}

/**
 * Parse brittle TAP output into structured test results.
 *
 * Brittle emits:
 *   # <top-level test name>
 *       ok N - <assertion>
 *       not ok N - <assertion>
 *       # <comment>
 *
 * Each `# <name>` block (non-indented) becomes a test case.
 */
function parseTap (tap) {
  const lines = tap.split('\n')
  const results = []
  let current = null

  for (const raw of lines) {
    const line = raw.trimEnd()

    const topLevelComment = line.match(/^# (.+)/)
    if (topLevelComment) {
      if (current) results.push(current)
      current = {
        name: topLevelComment[1].trim(),
        status: 'passed',
        assertions: 0,
        failures: [],
        comments: []
      }
      continue
    }

    if (line.match(/^\s+# (.+)/) && current) {
      current.comments.push(line.replace(/^\s+# /, '').trim())
      continue
    }

    const okMatch = line.match(/^\s*(ok|not ok)\s+\d+\s*(?:-\s*(.*))?/)
    if (okMatch) {
      if (!current) {
        current = { name: 'unknown', status: 'passed', assertions: 0, failures: [], comments: [] }
      }
      current.assertions++
      const ok = okMatch[1]
      const desc = (okMatch[2] || '').trim()

      if (ok === 'not ok') {
        current.status = 'failed'
        current.failures.push(desc || `assertion ${current.assertions} failed`)
      }

      if (desc.includes('# SKIP') || desc.includes('# skip')) {
        current.status = 'skipped'
      }
      continue
    }

    const bailOut = line.match(/^Bail out!\s*(.*)/)
    if (bailOut) {
      if (current) {
        current.status = 'failed'
        current.failures.push(`Bail out: ${bailOut[1]}`)
      }
      break
    }
  }

  if (current) results.push(current)
  return results
}

function main () {
  const args = parseArgs(process.argv)

  let tap
  if (args.input) {
    tap = fs.readFileSync(path.resolve(args.input), 'utf8')
  } else if (!process.stdin.isTTY) {
    tap = fs.readFileSync('/dev/stdin', 'utf8')
  } else {
    console.error('Error: no TAP input. Provide --input <file> or pipe TAP to stdin.')
    process.exit(1)
  }

  const results = parseTap(tap)

  if (results.length === 0) {
    console.error('Warning: no test cases found in TAP input.')
  }

  const report = {
    suite: args.suite,
    timestamp: new Date().toISOString(),
    summary: {
      total: results.length,
      passed: results.filter(r => r.status === 'passed').length,
      failed: results.filter(r => r.status === 'failed').length,
      skipped: results.filter(r => r.status === 'skipped').length
    },
    results
  }

  const json = JSON.stringify(report, null, 2) + '\n'

  if (args.output) {
    const outPath = path.resolve(args.output)
    fs.mkdirSync(path.dirname(outPath), { recursive: true })
    fs.writeFileSync(outPath, json)
    console.error(`Wrote ${results.length} test results to ${outPath}`)
  } else {
    process.stdout.write(json)
  }
}

main()
