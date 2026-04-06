#!/usr/bin/env node
'use strict'

/**
 * Render a test report from the JSON intermediate format.
 *
 * Reads the JSON produced by tap-to-json.js (or mobile directly) and
 * outputs HTML or JUnit XML.
 *
 * Usage:
 *   node render-report.js [options]
 *
 * Options:
 *   --input, -i   <path>     JSON report file (required)
 *   --output, -o  <path>     Output file      (default: stdout)
 *   --format, -f  <fmt>      "html" or "junit" (default: "html")
 */

const fs = require('fs')
const path = require('path')

function parseArgs (argv) {
  const args = { input: null, output: null, format: 'html' }
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i]
    if ((arg === '--input' || arg === '-i') && argv[i + 1]) args.input = argv[++i]
    else if ((arg === '--output' || arg === '-o') && argv[i + 1]) args.output = argv[++i]
    else if ((arg === '--format' || arg === '-f') && argv[i + 1]) args.format = argv[++i]
  }
  return args
}

// ── HTML renderer ──────────────────────────────────────────────────────────

function escapeHtml (str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
}

function renderHtml (report) {
  const { suite, timestamp, summary, results } = report
  const passRate = summary.total - summary.skipped > 0
    ? ((summary.passed / (summary.total - summary.skipped)) * 100).toFixed(1)
    : '0.0'
  const ts = timestamp
    ? new Date(timestamp).toISOString().replace('T', ' ').slice(0, 19) + ' UTC'
    : new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC'

  const metaParts = [ts]
  if (report.platform) metaParts.push(`Platform: ${report.platform}`)
  if (report.source) metaParts.push(`Source: ${report.source}`)
  if (report.devices && report.devices.length > 0) metaParts.push(`Devices: ${report.devices.join(', ')}`)
  const metaLine = metaParts.join(' &bull; ')

  const hasDevices = results.some(tc => tc.device)

  const rows = results.map((tc, idx) => {
    let statusClass, statusLabel
    if (tc.status === 'skipped') { statusClass = 'skip'; statusLabel = 'SKIP' }
    else if (tc.status === 'passed') { statusClass = 'pass'; statusLabel = 'PASS' }
    else { statusClass = 'fail'; statusLabel = 'FAIL' }

    let details = ''
    if (tc.failures && tc.failures.length > 0) {
      details += `<div class="failure-detail">${tc.failures.map(f => escapeHtml(f)).join('<br>')}</div>`
    }
    if (tc.comments && tc.comments.length > 0) {
      details += `<div class="comment-detail">${tc.comments.map(c => escapeHtml(c)).join('<br>')}</div>`
    }

    const deviceCell = hasDevices
      ? `<td class="col-device">${escapeHtml(tc.device || '')}</td>`
      : ''

    return `
      <tr class="row-${statusClass}">
        <td class="col-num">${idx + 1}</td>
        <td class="col-status"><span class="badge badge-${statusClass}">${statusLabel}</span></td>
        <td class="col-name">
          ${escapeHtml(tc.name)}
          ${details}
        </td>
        ${deviceCell}
        <td class="col-assert">${tc.assertions || 0}</td>
      </tr>`
  }).join('\n')

  const deviceHeader = hasDevices ? '<th>Device</th>' : ''

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${escapeHtml(suite)}</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; line-height: 1.5; padding: 24px; }
  .container { max-width: 960px; margin: 0 auto; }
  h1 { font-size: 1.5rem; font-weight: 600; margin-bottom: 4px; color: #f0f6fc; }
  .meta { color: #8b949e; font-size: 0.85rem; margin-bottom: 20px; }
  .summary { display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }
  .summary-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px 20px; min-width: 120px; text-align: center; }
  .summary-card .num { font-size: 1.8rem; font-weight: 700; }
  .summary-card .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; color: #8b949e; }
  .num-total { color: #f0f6fc; }
  .num-pass { color: #3fb950; }
  .num-fail { color: #f85149; }
  .num-skip { color: #d29922; }
  .num-rate { color: #58a6ff; }
  .progress-bar { height: 8px; background: #21262d; border-radius: 4px; overflow: hidden; margin-bottom: 24px; display: flex; }
  .progress-pass { background: #3fb950; }
  .progress-fail { background: #f85149; }
  .progress-skip { background: #d29922; }
  table { width: 100%; border-collapse: collapse; background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }
  th { background: #21262d; text-align: left; padding: 10px 14px; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; color: #8b949e; border-bottom: 1px solid #30363d; }
  td { padding: 10px 14px; border-bottom: 1px solid #21262d; vertical-align: top; }
  tr:last-child td { border-bottom: none; }
  .col-num { width: 40px; text-align: center; color: #484f58; font-size: 0.8rem; }
  .col-status { width: 70px; text-align: center; }
  .col-assert { width: 60px; text-align: center; color: #8b949e; }
  .col-device { width: 160px; color: #8b949e; font-size: 0.82rem; }
  .badge { padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.3px; }
  .badge-pass { background: #0d3321; color: #3fb950; }
  .badge-fail { background: #3d1318; color: #f85149; }
  .badge-skip { background: #3d2e00; color: #d29922; }
  .failure-detail { margin-top: 6px; padding: 8px 12px; background: #1c1014; border-left: 3px solid #f85149; border-radius: 4px; font-size: 0.82rem; color: #f0a0a0; font-family: 'SF Mono', Menlo, monospace; }
  .comment-detail { margin-top: 6px; padding: 6px 12px; color: #8b949e; font-size: 0.82rem; font-family: 'SF Mono', Menlo, monospace; }
  .row-fail { background: rgba(248, 81, 73, 0.04); }
  .filter-bar { display: flex; gap: 8px; margin-bottom: 16px; }
  .filter-btn { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 6px 14px; font-size: 0.8rem; cursor: pointer; transition: all 0.15s; }
  .filter-btn:hover { background: #30363d; }
  .filter-btn.active { background: #388bfd26; border-color: #388bfd; color: #58a6ff; }
</style>
</head>
<body>
<div class="container">
  <h1>${escapeHtml(suite)}</h1>
  <p class="meta">${metaLine}</p>

  <div class="summary">
    <div class="summary-card"><div class="num num-total">${summary.total}</div><div class="label">Total</div></div>
    <div class="summary-card"><div class="num num-pass">${summary.passed}</div><div class="label">Passed</div></div>
    <div class="summary-card"><div class="num num-fail">${summary.failed}</div><div class="label">Failed</div></div>
    <div class="summary-card"><div class="num num-skip">${summary.skipped}</div><div class="label">Skipped</div></div>
    <div class="summary-card"><div class="num num-rate">${passRate}%</div><div class="label">Pass Rate</div></div>
  </div>

  <div class="progress-bar">
    <div class="progress-pass" style="width:${summary.total > 0 ? (summary.passed / summary.total * 100) : 0}%"></div>
    <div class="progress-fail" style="width:${summary.total > 0 ? (summary.failed / summary.total * 100) : 0}%"></div>
    <div class="progress-skip" style="width:${summary.total > 0 ? (summary.skipped / summary.total * 100) : 0}%"></div>
  </div>

  <div class="filter-bar">
    <button class="filter-btn active" data-filter="all">All</button>
    <button class="filter-btn" data-filter="pass">Passed</button>
    <button class="filter-btn" data-filter="fail">Failed</button>
    <button class="filter-btn" data-filter="skip">Skipped</button>
  </div>

  <table>
    <thead><tr><th>#</th><th>Status</th><th>Test</th>${deviceHeader}<th>Asserts</th></tr></thead>
    <tbody>
${rows}
    </tbody>
  </table>
</div>
<script>
document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const filter = btn.dataset.filter;
    document.querySelectorAll('tbody tr').forEach(row => {
      if (filter === 'all') { row.style.display = ''; return; }
      row.style.display = row.classList.contains('row-' + filter) ? '' : 'none';
    });
  });
});
</script>
</body>
</html>
`
}

// ── JUnit XML renderer ─────────────────────────────────────────────────────

function escapeXml (str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&apos;')
}

function renderJunit (report) {
  const { suite, summary, results } = report

  const lines = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    `<testsuites tests="${summary.total}" failures="${summary.failed}" errors="0" skipped="${summary.skipped}">`,
    `  <testsuite name="${escapeXml(suite)}" tests="${summary.total}" failures="${summary.failed}" errors="0" skipped="${summary.skipped}">`
  ]

  for (const tc of results) {
    const className = tc.device ? `${escapeXml(suite)}.${escapeXml(tc.device)}` : escapeXml(suite)
    lines.push(`    <testcase name="${escapeXml(tc.name)}" classname="${className}">`)

    if (tc.status === 'skipped') {
      lines.push('      <skipped/>')
    } else if (tc.status === 'failed' && tc.failures) {
      const msg = tc.failures.join('; ')
      lines.push(`      <failure message="${escapeXml(msg)}">${escapeXml(msg)}</failure>`)
    }

    lines.push('    </testcase>')
  }

  lines.push('  </testsuite>')
  lines.push('</testsuites>')
  lines.push('')

  return lines.join('\n')
}

// ── Main ───────────────────────────────────────────────────────────────────

function main () {
  const args = parseArgs(process.argv)

  if (!args.input) {
    console.error('Error: --input <path> is required (JSON report file).')
    process.exit(1)
  }

  const inputPath = path.resolve(args.input)
  if (!fs.existsSync(inputPath)) {
    console.error(`Error: file not found: ${inputPath}`)
    process.exit(1)
  }

  const report = JSON.parse(fs.readFileSync(inputPath, 'utf8'))

  let output
  if (args.format === 'junit' || args.format === 'xml') {
    output = renderJunit(report)
  } else if (args.format === 'html') {
    output = renderHtml(report)
  } else {
    console.error(`Error: unknown format "${args.format}". Use "html" or "junit".`)
    process.exit(1)
  }

  if (args.output) {
    const outPath = path.resolve(args.output)
    fs.mkdirSync(path.dirname(outPath), { recursive: true })
    fs.writeFileSync(outPath, output)
    console.error(`Wrote ${args.format} report (${report.results.length} tests) to ${outPath}`)
  } else {
    process.stdout.write(output)
  }
}

main()
