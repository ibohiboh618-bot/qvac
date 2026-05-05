'use strict'

/**
 * Bergamot Model Fetcher
 *
 * Downloads Bergamot (Firefox Translations) model files from the
 * Firefox Remote Settings CDN — the same source Firefox browser uses.
 *
 * This module does NOT touch OPUS or IndicTrans models.
 */

const fs = require('bare-fs')
const path = require('bare-path')

// ============================================================================
// Firefox Remote Settings CDN
// ============================================================================

const FIREFOX_RECORDS_URL =
  'https://firefox.settings.services.mozilla.com/v1/buckets/main/collections/translations-models/records'
const FIREFOX_ATTACHMENT_BASE =
  'https://firefox-settings-attachments.cdn.mozilla.net'

// ============================================================================
// Helpers
// ============================================================================

/**
 * Returns expected Bergamot model filenames for a language pair.
 * CJK target languages (zh, ja, ko) use separate src/trg vocabs.
 */
function getBergamotFileNames (srcLang, dstLang) {
  const pair = `${srcLang}${dstLang}`
  const cjk = ['zh', 'ja', 'ko']
  const separateVocab = cjk.includes(dstLang) || (cjk.includes(srcLang) && dstLang === 'en' && srcLang !== 'en')

  return {
    modelName: `model.${pair}.intgemm.alphas.bin`,
    srcVocabName: separateVocab ? `srcvocab.${pair}.spm` : `vocab.${pair}.spm`,
    dstVocabName: separateVocab ? `trgvocab.${pair}.spm` : `vocab.${pair}.spm`,
    lexName: `lex.50.50.${pair}.s2t.bin`
  }
}

/**
 * Checks whether a directory already contains a valid Bergamot model
 * (at minimum an .intgemm model file and a .spm vocab file).
 */
function hasBergamotModelFiles (dir) {
  try {
    const files = fs.readdirSync(dir)
    return files.some(f => f.includes('.intgemm')) && files.some(f => f.endsWith('.spm'))
  } catch {
    return false
  }
}

// ============================================================================
// Download via Firefox Remote Settings CDN
// ============================================================================

// Minimum plausible size (bytes) for a completed Bergamot artifact.
// Real files are 800KB+ (vocab) to 30MB+ (intgemm). Anything under 1KB is
// either a stub or a truncated/failed download and should be re-fetched.
const MIN_VALID_FILE_BYTES = 1024

/**
 * Returns true if `destPath` already exists as a non-trivially-sized file.
 * Used to skip re-downloads across invocations (per pivot sub-test) and
 * across duplicate records within a single invocation (Firefox's records
 * collection has production + dev variants sharing the same filename).
 */
function _isDownloadedFile (destPath) {
  try {
    const stat = fs.statSync(destPath)
    return stat.isFile() && stat.size >= MIN_VALID_FILE_BYTES
  } catch {
    return false
  }
}

/**
 * Downloads a single file from a URL to a local path.
 * Follows redirects via bare-fetch.
 * Skips the fetch entirely if the file already exists with non-trivial size.
 */
async function downloadFile (url, destPath) {
  if (_isDownloadedFile(destPath)) {
    return fs.statSync(destPath).size
  }

  const fetch = require('bare-fetch')

  const response = await fetch(url, { redirect: 'follow', follow: 5 })
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} downloading ${url}`)
  }
  const buffer = await response.arrayBuffer()
  fs.writeFileSync(destPath, Buffer.from(buffer))
  return buffer.byteLength
}

/**
 * Parses a dotted version string (e.g. "2.1") into a numeric tuple suitable
 * for lexicographic comparison. Non-numeric parts collapse to 0 so we never
 * throw on unexpected metadata.
 */
function _parseVersion (v) {
  if (typeof v !== 'string') return [0]
  return v.split('.').map(p => Number.parseInt(p, 10) || 0)
}

/** Returns positive when `a` is newer than `b`, negative when older, 0 when equal. */
function _compareVersion (a, b) {
  const av = _parseVersion(a)
  const bv = _parseVersion(b)
  const len = Math.max(av.length, bv.length)
  for (let i = 0; i < len; i++) {
    const ai = av[i] || 0
    const bi = bv[i] || 0
    if (ai !== bi) return ai - bi
  }
  return 0
}

/**
 * Groups Firefox `translations-models` records by their canonical filename
 * (the same `model.<pair>.intgemm.alphas.bin` / `vocab.<pair>.spm` /
 * `lex.<pair>.s2t.bin` shows up once per published version), then keeps the
 * highest-version record for each filename.
 *
 * This is the variant-pinning logic that protects callers from accidentally
 * landing on the legacy v1.0 (a.k.a. "tiny") models — which are still served
 * from the Firefox CDN for backward compatibility but produce detokenised
 * output with a stray space before sentence-final punctuation
 * ("Ciao mondo !" instead of "Ciao mondo!"). The v2.x records correspond to
 * the "base-memory" variant Mozilla currently recommends for Firefox itself.
 */
function _selectNewestPerFilename (records) {
  const byFilename = new Map()
  for (const record of records) {
    const att = record.attachment
    if (!att || !att.location) continue
    const filename = record.name || att.filename || path.basename(att.location)
    const existing = byFilename.get(filename)
    if (!existing || _compareVersion(record.version, existing.version) > 0) {
      byFilename.set(filename, record)
    }
  }
  return Array.from(byFilename.values())
}

/**
 * Downloads Bergamot model files from Mozilla's Firefox Remote Settings CDN.
 * This is the same source Firefox itself uses for translation models.
 *
 * @param {string} srcLang Source language code (e.g. 'en')
 * @param {string} dstLang Target language code (e.g. 'it')
 * @param {string} destDir Directory to write files into
 * @param {Object} [options]
 * @param {string} [options.minVersion]
 *   Lower bound on the Firefox record `version` field. Records older than
 *   this are dropped. Defaults to '2.0' to exclude the legacy v1.x ("tiny")
 *   variant whose detokenisation regresses sentence-final punctuation. Pass
 *   '0' or null to disable the filter (e.g. for explicit legacy regression
 *   testing).
 */
async function downloadBergamotFromFirefox (srcLang, dstLang, destDir, options = {}) {
  const fetch = require('bare-fetch')

  const minVersion = options.minVersion === undefined ? '2.0' : options.minVersion

  console.log(`[bergamot-fetcher] Downloading ${srcLang}-${dstLang} from Firefox Remote Settings CDN...`)

  const res = await fetch(FIREFOX_RECORDS_URL)
  if (!res.ok) throw new Error(`Failed to fetch Firefox model records: HTTP ${res.status}`)
  const body = await res.json()
  const records = body.data || []

  let pairRecords = records.filter(
    r => r.fromLang === srcLang && r.toLang === dstLang && r.attachment
  )

  if (pairRecords.length === 0) {
    throw new Error(
      `No Firefox Translations model found for ${srcLang}-${dstLang}. ` +
      'Check https://github.com/mozilla/firefox-translations-models for supported pairs.'
    )
  }

  if (minVersion) {
    const filtered = pairRecords.filter(r => _compareVersion(r.version, minVersion) >= 0)
    if (filtered.length === 0) {
      throw new Error(
        `No Firefox Translations model for ${srcLang}-${dstLang} satisfies minVersion=${minVersion}. ` +
        'Pass `{ minVersion: null }` to opt into legacy variants explicitly.'
      )
    }
    pairRecords = filtered
  }

  const selected = _selectNewestPerFilename(pairRecords)
  console.log(
    `[bergamot-fetcher] Selected ${selected.length} files (` +
    selected.map(r => `${r.name || r.attachment?.filename}@v${r.version}`).join(', ') +
    ')'
  )

  fs.mkdirSync(destDir, { recursive: true })

  // Records have already been narrowed to one entry per filename (the
  // newest version satisfying `minVersion`), so a simple iteration is
  // enough — no need for a runtime `seenFilenames` guard.
  for (const record of selected) {
    const att = record.attachment
    const filename = record.name || att.filename || path.basename(att.location)
    const url = `${FIREFOX_ATTACHMENT_BASE}/${att.location}`
    const dest = path.join(destDir, filename)

    if (_isDownloadedFile(dest)) {
      const existingMB = fs.statSync(dest).size / 1024 / 1024
      console.log(`[bergamot-fetcher]   ✓ ${filename} (${existingMB.toFixed(1)}MB, cached)`)
      continue
    }

    console.log(`[bergamot-fetcher]   Downloading ${filename} (v${record.version})...`)
    const bytes = await downloadFile(url, dest)
    console.log(`[bergamot-fetcher]   ✓ ${filename} (${(bytes / 1024 / 1024).toFixed(1)}MB)`)
  }

  if (!hasBergamotModelFiles(destDir)) {
    throw new Error('Firefox CDN download incomplete — missing model or vocab files')
  }

  console.log(`[bergamot-fetcher] Firefox CDN download complete → ${destDir}`)
  return destDir
}

// ============================================================================
// Public API
// ============================================================================

/**
 * Ensures Bergamot model files are present in destDir for a given language pair.
 *
 *   1. If model files already exist in destDir → returns immediately
 *   2. Downloads from Firefox Remote Settings CDN
 *
 * @param {string} srcLang  Source language code (e.g. 'en')
 * @param {string} dstLang  Target language code (e.g. 'it')
 * @param {string} destDir  Directory to store model files
 * @param {Object} [options]
 * @param {string|null} [options.minVersion]
 *   Forwarded to `downloadBergamotFromFirefox`. Defaults to '2.0' (excludes
 *   the legacy v1.x "tiny" Bergamot variant whose detokenisation regresses
 *   sentence-final punctuation). Pass `null` to opt back into legacy
 *   variants explicitly.
 * @returns {Promise<string>} Resolved path to the model directory
 */
async function ensureBergamotModelFiles (srcLang, dstLang, destDir, options = {}) {
  if (hasBergamotModelFiles(destDir)) {
    console.log(`[bergamot-fetcher] Model already available at ${destDir}`)
    return destDir
  }

  return await downloadBergamotFromFirefox(srcLang, dstLang, destDir, options)
}

// ============================================================================
// Exports
// ============================================================================

module.exports = {
  getBergamotFileNames,
  hasBergamotModelFiles,
  ensureBergamotModelFiles,
  downloadBergamotFromFirefox
}
