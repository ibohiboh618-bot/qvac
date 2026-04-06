#!/usr/bin/env node
'use strict'

/**
 * Convert AWS Device Farm logs into the standard JSON test report format.
 *
 * Downloads and parses device logs from a completed Device Farm run to extract
 * brittle TAP output. Falls back to Device Farm job-level results when TAP
 * output is not found in the logs.
 *
 * Expected directory layout (created by the CI download step):
 *
 *   <logs-dir>/
 *     <Device_Name>/
 *       DEVICE_NAME        # plain text: actual device name
 *       JOB_RESULT         # plain text: PASSED | FAILED | ERRORED
 *       TESTSPEC_OUTPUT.log # test spec stdout (may contain TAP)
 *       LOGCAT.log          # Android logcat (may contain TAP)
 *       DEVICE_LOG.log      # iOS device console (may contain TAP)
 *       CUSTOMER_ARTIFACT.log
 *
 * Usage:
 *   node devicefarm-to-json.js [options]
 *
 * Options:
 *   --logs-dir, -l  <path>   Directory containing per-device subdirectories
 *   --output, -o    <path>   JSON output file (default: stdout)
 *   --suite, -s     <name>   Suite name (default: "mobile-test")
 *   --platform      <name>   Platform label (e.g. "android", "ios")
 *   --job-result    <result> Fallback result when no device dirs found
 */

const fs = require('fs')
const path = require('path')

function parseArgs (argv) {
  const args = {
    logsDir: null,
    output: null,
    suite: 'mobile-test',
    platform: '',
    jobResult: null
  }
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i]
    if ((arg === '--logs-dir' || arg === '-l') && argv[i + 1]) args.logsDir = argv[++i]
    else if ((arg === '--output' || arg === '-o') && argv[i + 1]) args.output = argv[++i]
    else if ((arg === '--suite' || arg === '-s') && argv[i + 1]) args.suite = argv[++i]
    else if (arg === '--platform' && argv[i + 1]) args.platform = argv[++i]
    else if (arg === '--job-result' && argv[i + 1]) args.jobResult = argv[++i]
  }
  return args
}

function readTextFile (filePath) {
  try { return fs.readFileSync(filePath, 'utf8').trim() } catch { return null }
}

/**
 * Extract brittle TAP results from arbitrary log text.
 * Tolerates non-TAP lines mixed in (logcat timestamps, console prefixes, etc.).
 */
function parseTapFromText (text) {
  const lines = text.split('\n')
  const results = []
  let current = null

  for (const raw of lines) {
    const line = raw.replace(/^\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d+\s+\S+\s+\S+\s+\S+\s*:\s*/, '')
      .replace(/^\[[\d:. ]+\]\s*/, '')
      .trimEnd()

    const topLevelComment = line.match(/^#\s+(.+)/)
    if (topLevelComment) {
      const name = topLevelComment[1].trim()
      if (/^(ok|not ok|tests|pass|fail|Subtest)/.test(name)) continue
      if (/^\d+$/.test(name)) continue

      if (current) results.push(current)
      current = {
        name,
        status: 'passed',
        assertions: 0,
        failures: [],
        comments: []
      }
      continue
    }

    if (current && line.match(/^\s+#\s+(.+)/)) {
      current.comments.push(line.replace(/^\s+#\s+/, '').trim())
      continue
    }

    const okMatch = line.match(/^\s*(ok|not ok)\s+\d+\s*(?:-\s*(.*))?/)
    if (okMatch) {
      if (!current) {
        current = { name: 'unknown', status: 'passed', assertions: 0, failures: [], comments: [] }
      }
      current.assertions++
      if (okMatch[1] === 'not ok') {
        current.status = 'failed'
        current.failures.push((okMatch[2] || '').trim() || `assertion ${current.assertions} failed`)
      }
      if ((okMatch[2] || '').includes('# SKIP') || (okMatch[2] || '').includes('# skip')) {
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

/**
 * Recursively read all text files in a directory and try to extract TAP.
 * Returns { results, sourceFile } or null.
 */
function scanDirForTap (dir) {
  if (!dir || !fs.existsSync(dir)) return []

  const files = []
  function collect (d) {
    for (const entry of fs.readdirSync(d, { withFileTypes: true })) {
      const full = path.join(d, entry.name)
      if (entry.isDirectory()) collect(full)
      else if (entry.isFile()) files.push(full)
    }
  }
  collect(dir)

  const priorityOrder = ['TESTSPEC_OUTPUT', 'LOGCAT', 'DEVICE_LOG', 'CUSTOMER_ARTIFACT']
  files.sort((a, b) => {
    const aIdx = priorityOrder.findIndex(p => path.basename(a).includes(p))
    const bIdx = priorityOrder.findIndex(p => path.basename(b).includes(p))
    return (aIdx === -1 ? 99 : aIdx) - (bIdx === -1 ? 99 : bIdx)
  })

  for (const file of files) {
    try {
      const content = fs.readFileSync(file, 'utf8')
      if (content.includes('ok ') && content.includes('# ')) {
        const results = parseTapFromText(content)
        if (results.length > 0) {
          console.error(`  Found ${results.length} test results in ${path.basename(file)}`)
          return results
        }
      }
    } catch {}
  }

  return []
}

/**
 * Process a single device subdirectory.
 * Returns an array of test result objects, each tagged with device name.
 */
function processDeviceDir (deviceDir) {
  const deviceName = readTextFile(path.join(deviceDir, 'DEVICE_NAME')) ||
    path.basename(deviceDir).replace(/_/g, ' ')
  const jobResult = readTextFile(path.join(deviceDir, 'JOB_RESULT')) || 'UNKNOWN'

  console.error(`Processing device: ${deviceName} (job result: ${jobResult})`)

  const tapResults = scanDirForTap(deviceDir)

  if (tapResults.length > 0) {
    return tapResults.map(r => ({ ...r, device: deviceName }))
  }

  const status = jobResult === 'PASSED' ? 'passed'
    : jobResult === 'SKIPPED' ? 'skipped'
      : 'failed'

  return [{
    name: `${deviceName} — integration tests`,
    status,
    assertions: 0,
    failures: status === 'failed' ? [`Device Farm result: ${jobResult}`] : [],
    comments: [`Device: ${deviceName}`, `Source: Device Farm job-level result`],
    device: deviceName
  }]
}

function main () {
  const args = parseArgs(process.argv)
  let allResults = []
  let source = 'device-farm'

  if (args.logsDir && fs.existsSync(args.logsDir)) {
    const entries = fs.readdirSync(args.logsDir, { withFileTypes: true })
    const deviceDirs = entries.filter(e => e.isDirectory())

    if (deviceDirs.length > 0) {
      for (const dir of deviceDirs) {
        const results = processDeviceDir(path.join(args.logsDir, dir.name))
        allResults.push(...results)
      }
      const hasTap = allResults.some(r => !r.comments?.some(c => c.includes('job-level')))
      source = hasTap ? 'tap-logs' : 'device-farm-job'
    } else {
      allResults = scanDirForTap(args.logsDir)
      if (allResults.length > 0) source = 'tap-logs'
    }
  }

  if (allResults.length === 0 && args.jobResult) {
    const status = args.jobResult === 'PASSED' ? 'passed'
      : args.jobResult === 'SKIPPED' ? 'skipped'
        : 'failed'

    allResults = [{
      name: `${args.platform || 'device'} — integration tests`,
      status,
      assertions: 0,
      failures: status === 'failed' ? [`Device Farm result: ${args.jobResult}`] : [],
      comments: ['Source: Device Farm run-level result']
    }]
    source = 'device-farm-run'
    console.error(`No device logs found, using run-level result: ${args.jobResult}`)
  }

  const suiteName = args.platform
    ? `${args.suite}-${args.platform}`
    : args.suite

  const devices = [...new Set(allResults.map(r => r.device).filter(Boolean))]

  const report = {
    suite: suiteName,
    timestamp: new Date().toISOString(),
    source,
    platform: args.platform || null,
    devices: devices.length > 0 ? devices : null,
    summary: {
      total: allResults.length,
      passed: allResults.filter(r => r.status === 'passed').length,
      failed: allResults.filter(r => r.status === 'failed').length,
      skipped: allResults.filter(r => r.status === 'skipped').length
    },
    results: allResults
  }

  const json = JSON.stringify(report, null, 2) + '\n'

  if (args.output) {
    const outPath = path.resolve(args.output)
    fs.mkdirSync(path.dirname(outPath), { recursive: true })
    fs.writeFileSync(outPath, json)
    console.error(`Wrote report (${allResults.length} results) to ${outPath}`)
  } else {
    process.stdout.write(json)
  }
}

main()
