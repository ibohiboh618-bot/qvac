#!/usr/bin/env node
/**
 * Builds the URL list scanned by the daily docs health check.
 *
 * Output: a plaintext file (one URL per line) that `lycheeverse/lychee-action`
 * consumes — lychee extracts every URL from a plaintext input and checks it.
 *
 * Sources (all against a single origin, default production):
 *   1. Every `<loc>` in the live `/sitemap.xml`.
 *   2. The `.md` variant of every sitemap entry. The mapping mirrors
 *      `docs/website/scripts/generate-llm-md-files.ts` exactly:
 *        `/`            -> `/index.md`
 *        `/quickstart/` -> `/quickstart.md`
 *        `/a/b/`        -> `/a/b.md`
 *   3. Every permanent (`301`) redirect SOURCE path from
 *      `docs/website/public/_redirects`, excluding pattern rules (sources
 *      containing `:` placeholders or `*` splats). lychee follows redirects,
 *      so checking the source verifies the old bookmarked path still lands on
 *      a live page.
 *   4. `/sitemap.xml`, `/llms.txt`, `/llms-full.txt`.
 *
 * If the sitemap fetch fails the script logs a warning and still emits the
 * base URLs (so a broken `/sitemap.xml` is itself caught by lychee as a 404).
 *
 * Env:
 *   SITE_ORIGIN    Origin to scan (default https://docs.qvac.tether.io).
 *   REDIRECTS_FILE Path to the `_redirects` file
 *                  (default docs/website/public/_redirects).
 *   OUTPUT_FILE    Output path (default urls.txt).
 */

import { readFile, writeFile } from 'node:fs/promises';
import process from 'node:process';

const SITE_ORIGIN = (process.env.SITE_ORIGIN || 'https://docs.qvac.tether.io').replace(/\/+$/, '');
const REDIRECTS_FILE = process.env.REDIRECTS_FILE || 'docs/website/public/_redirects';
const OUTPUT_FILE = process.env.OUTPUT_FILE || 'urls.txt';

const SITEMAP_URL = `${SITE_ORIGIN}/sitemap.xml`;

/**
 * Mirrors `urlToMarkdownRelativePath` from generate-llm-md-files.ts:
 * strip leading/trailing slashes; empty path -> `index.md`; else `<path>.md`.
 * Derived from the loc URL's own origin so the result stays internally
 * consistent with whatever the sitemap advertises.
 */
function toMarkdownUrl(locUrl) {
  const u = new URL(locUrl);
  const trimmed = u.pathname.replace(/^\/+/, '').replace(/\/+$/, '');
  const rel = trimmed.length === 0 ? 'index.md' : `${trimmed}.md`;
  return `${u.origin}/${rel}`;
}

async function fetchSitemapUrls() {
  try {
    const res = await fetch(SITEMAP_URL, {
      headers: { 'User-Agent': 'qvac-docs-health-check' },
    });
    if (!res.ok) {
      console.warn(`WARN: ${SITEMAP_URL} returned HTTP ${res.status}; continuing with base URLs only.`);
      return [];
    }
    const xml = await res.text();
    const locs = [...xml.matchAll(/<loc>\s*([^<\s]+)\s*<\/loc>/g)].map((m) => m[1].trim());
    if (locs.length === 0) {
      console.warn('WARN: sitemap.xml contained no <loc> entries; continuing with base URLs only.');
    }
    return locs;
  } catch (err) {
    console.warn(`WARN: failed to fetch ${SITEMAP_URL}: ${err.message}; continuing with base URLs only.`);
    return [];
  }
}

async function readRedirectSources() {
  let raw;
  try {
    raw = await readFile(REDIRECTS_FILE, 'utf-8');
  } catch (err) {
    console.warn(`WARN: could not read ${REDIRECTS_FILE}: ${err.message}; skipping redirect checks.`);
    return [];
  }

  const sources = [];
  for (const line of raw.split('\n')) {
    const trimmed = line.trim();
    if (trimmed.length === 0 || trimmed.startsWith('#')) continue;

    const tokens = trimmed.split(/\s+/);
    // Permanent redirects only: a 301 status token must be present.
    const status = tokens.find((t) => /^\d{3}$/.test(t));
    if (status !== '301') continue;

    const source = tokens[0];
    // Skip pattern rules — `:placeholder` and `*` splat sources are not
    // literal URLs and cannot be fetched directly.
    if (!source.startsWith('/') || source.includes(':') || source.includes('*')) continue;

    sources.push(`${SITE_ORIGIN}${source}`);
  }
  return sources;
}

async function main() {
  const sitemapLocs = await fetchSitemapUrls();
  const redirectSources = await readRedirectSources();

  const urls = new Set();

  for (const loc of sitemapLocs) {
    urls.add(loc);
    try {
      urls.add(toMarkdownUrl(loc));
    } catch (err) {
      console.warn(`WARN: could not derive .md variant for ${loc}: ${err.message}`);
    }
  }

  for (const source of redirectSources) urls.add(source);

  urls.add(SITEMAP_URL);
  urls.add(`${SITE_ORIGIN}/llms.txt`);
  urls.add(`${SITE_ORIGIN}/llms-full.txt`);

  const sorted = [...urls].sort();
  await writeFile(OUTPUT_FILE, sorted.join('\n') + '\n', 'utf-8');

  console.log(`Origin:           ${SITE_ORIGIN}`);
  console.log(`Sitemap entries:  ${sitemapLocs.length}`);
  console.log(`Redirect sources: ${redirectSources.length}`);
  console.log(`Total URLs:       ${sorted.length} written to ${OUTPUT_FILE}`);
}

main().catch((err) => {
  console.error(`ERROR: ${err.stack || err.message}`);
  process.exit(1);
});
