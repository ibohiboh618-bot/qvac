#!/usr/bin/env node
/**
 * Parses lychee's JSON report, counts broken pages (HTTP 404), writes a
 * human-readable breakdown to the GitHub Actions job summary, and exposes
 * counts as step outputs.
 *
 * Why a 404 gate instead of lychee's own `fail`: the requirement is to alert
 * (Slack) only on broken pages. Transient timeouts / DNS / 5xx land in the job
 * summary for inspection but must NOT ping Slack, so the workflow gates the
 * Slack + job-failure steps on `count_404 > 0`.
 *
 * lychee's `status` field is not a stable shape (see
 * https://github.com/lycheeverse/lychee/pull/1367): it is either
 *   - an object `{ text, code?, details? }`, or
 *   - a string `"Failed: HTTP status client error (404 Not Found) for url (...)"`.
 * The code is extracted defensively from both.
 *
 * Usage: node parse-lychee.mjs [report.json]   (default ./lychee-report.json)
 */

import { readFile, appendFile } from 'node:fs/promises';
import process from 'node:process';

const REPORT_PATH = process.argv[2] || './lychee-report.json';

/**
 * Extracts the HTTP status code from a lychee `status` value, or null when
 * there is no HTTP code (network / DNS / TLS errors). The string branch only
 * matches a 3-digit code inside parentheses (e.g. `(404 Not Found)`) so a URL
 * that merely contains `404` (e.g. `/404.html`) never produces a false match.
 */
function statusToCode(status) {
  if (typeof status === 'number') return status;
  if (status && typeof status === 'object') {
    return typeof status.code === 'number' ? status.code : null;
  }
  if (typeof status === 'string') {
    const inParens = status.match(/\((\d{3})\b/);
    if (inParens) return Number(inParens[1]);
    const labelled = status.match(/\bstatus(?:\s*code)?[:\s]+(\d{3})\b/i);
    if (labelled) return Number(labelled[1]);
    return null;
  }
  return null;
}

function statusToText(status) {
  if (typeof status === 'string') return status;
  if (status && typeof status === 'object') {
    const parts = [];
    if (status.code != null) parts.push(String(status.code));
    if (status.text) parts.push(status.text);
    if (status.details) parts.push(status.details);
    return parts.length > 0 ? parts.join(' — ') : JSON.stringify(status);
  }
  return String(status);
}

function escapeCell(value) {
  return String(value).replace(/\|/g, '\\|').replace(/\r?\n/g, ' ');
}

function renderTable(rows) {
  const lines = ['| URL | Status |', '| --- | --- |'];
  for (const row of rows) {
    lines.push(`| ${escapeCell(row.url)} | ${escapeCell(row.text)} |`);
  }
  return lines.join('\n');
}

async function main() {
  let data;
  try {
    data = JSON.parse(await readFile(REPORT_PATH, 'utf-8'));
  } catch (err) {
    console.error(`ERROR: could not read lychee report at ${REPORT_PATH}: ${err.message}`);
    process.exit(1);
  }

  const failMap = data.fail_map || data.error_map || {};
  const failures = [];
  for (const entries of Object.values(failMap)) {
    if (!Array.isArray(entries)) continue;
    for (const entry of entries) {
      const code = statusToCode(entry.status);
      failures.push({
        url: entry.url,
        code,
        text: statusToText(entry.status),
      });
    }
  }

  const broken = failures.filter((f) => f.code === 404);
  const other = failures.filter((f) => f.code !== 404);
  const total = typeof data.total === 'number' ? data.total : failures.length;

  const summary = [];
  summary.push('## Docs Site Health Check');
  summary.push('');
  summary.push(`- URLs scanned: **${total}**`);
  summary.push(`- Broken pages (404): **${broken.length}**`);
  summary.push(`- Other errors (not 404): **${other.length}**`);
  summary.push('');

  if (broken.length > 0) {
    summary.push('### Broken pages (404)');
    summary.push('');
    summary.push(renderTable(broken));
    summary.push('');
  }

  if (other.length > 0) {
    summary.push('### Other errors (not 404 — not alerted on)');
    summary.push('');
    summary.push(renderTable(other));
    summary.push('');
  }

  if (failures.length === 0) {
    summary.push('All scanned URLs are healthy. No broken pages detected.');
    summary.push('');
  }

  const summaryText = summary.join('\n');
  if (process.env.GITHUB_STEP_SUMMARY) {
    await appendFile(process.env.GITHUB_STEP_SUMMARY, summaryText + '\n', 'utf-8');
  }
  console.log(summaryText);

  if (process.env.GITHUB_OUTPUT) {
    await appendFile(
      process.env.GITHUB_OUTPUT,
      `count_404=${broken.length}\ncount_other=${other.length}\ntotal=${total}\n`,
      'utf-8',
    );
  }
}

main().catch((err) => {
  console.error(`ERROR: ${err.stack || err.message}`);
  process.exit(1);
});
