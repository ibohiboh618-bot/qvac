#!/usr/bin/env node
'use strict'

// Resolves model files needed by the benchmark and writes their paths
// to resolved-models.json. A single model is downloaded and shared
// across all three inference sources (addon, fabric-cli, upstream-cli).
//
// Output (resolved-models.json):
//   {
//     label, quant, hfRepo, hfRevision,
//     llmPath, mmprojPath, provenance
//   }

const fs = require('fs')
const path = require('path')
const https = require('https')
const crypto = require('crypto')

const config = require('./vlm-bench.config')
const { parseArgs } = require('./utils')

const SCRIPT_DIR = __dirname
const DEFAULT_MODELS_DIR = path.join(SCRIPT_DIR, 'models')

function log (...args) { console.log('[prepare-models]', ...args) }

function ensureDir (dir) { fs.mkdirSync(dir, { recursive: true }) }

function downloadFile (url, destination, headers, redirects = 5) {
  return new Promise((resolve, reject) => {
    ensureDir(path.dirname(destination))
    const tmpPath = `${destination}.partial`
    const out = fs.createWriteStream(tmpPath)
    let settled = false
    const cleanup = () => { try { if (fs.existsSync(tmpPath)) fs.rmSync(tmpPath) } catch {} }
    const fail = (error) => {
      if (settled) return
      settled = true
      try { out.destroy() } catch {}
      cleanup()
      reject(error)
    }
    const succeed = () => {
      if (settled) return
      settled = true
      try { fs.renameSync(tmpPath, destination); resolve() } catch (e) { reject(e) }
    }
    https.get(url, { headers }, (res) => {
      const status = res.statusCode || 0
      if (status >= 300 && status < 400 && res.headers.location) {
        res.resume()
        if (redirects <= 0) return fail(new Error(`Too many redirects for ${url}`))
        const next = new URL(res.headers.location, url).toString()
        settled = true
        try { out.destroy() } catch {}
        cleanup()
        return resolve(downloadFile(next, destination, headers, redirects - 1))
      }
      if (status < 200 || status >= 300) {
        res.resume()
        const err = new Error(`HTTP ${status}`)
        err.code = status
        err.url = url
        return fail(err)
      }
      res.pipe(out)
      out.on('finish', () => out.close(succeed))
      res.on('error', fail)
    }).on('error', fail)
    out.on('error', (e) => fail(e))
  })
}

function sha256OfFile (filePath) {
  return new Promise((resolve, reject) => {
    const h = crypto.createHash('sha256')
    const s = fs.createReadStream(filePath)
    s.on('error', reject)
    s.on('data', (c) => h.update(c))
    s.on('end', () => resolve(h.digest('hex')))
  })
}

async function provenanceFor (filePath, urlUsed) {
  if (!filePath || !fs.existsSync(filePath)) return null
  const stat = fs.statSync(filePath)
  const sha256 = await sha256OfFile(filePath)
  return {
    path: filePath,
    sizeBytes: stat.size,
    sizeMb: Math.round((stat.size / (1024 * 1024)) * 100) / 100,
    sha256,
    url: urlUsed || null
  }
}

async function resolvePart ({ label, kind, url, destination, localOverride, headers }) {
  if (localOverride) {
    if (!fs.existsSync(localOverride)) {
      throw new Error(`--local-${kind} override file not found: ${localOverride}`)
    }
    log(`${label}/${kind}: using local file ${localOverride}`)
    return path.resolve(localOverride)
  }
  if (fs.existsSync(destination)) {
    log(`${label}/${kind}: already present -> ${destination}`)
    return destination
  }
  if (!url) {
    throw new Error(`No URL configured for ${label}/${kind}`)
  }
  log(`${label}/${kind}: downloading from ${url}`)
  await downloadFile(url, destination, headers)
  log(`${label}/${kind}: downloaded -> ${destination}`)
  return destination
}

async function main () {
  const args = parseArgs(process.argv.slice(2))
  const modelsDir = args['models-dir'] ? path.resolve(args['models-dir']) : DEFAULT_MODELS_DIR
  ensureDir(modelsDir)

  const headers = { 'User-Agent': 'qvac-vlm-benchmark-prep/1.0' }
  if (process.env.HF_TOKEN) headers.Authorization = `Bearer ${process.env.HF_TOKEN}`

  const m = config.model

  const llmDest = path.join(modelsDir, m.llmFile)
  const mmprojDest = path.join(modelsDir, m.mmprojFile)

  const llmPath = await resolvePart({
    label: m.label,
    kind: 'llm',
    url: m.url.llm,
    destination: llmDest,
    localOverride: args['local-model'],
    headers
  })

  const mmprojPath = await resolvePart({
    label: m.label,
    kind: 'mmproj',
    url: m.url.mmproj,
    destination: mmprojDest,
    localOverride: args['local-mmproj'],
    headers
  })

  const llmProvenance = await provenanceFor(llmPath, m.url.llm)
  const mmprojProvenance = await provenanceFor(mmprojPath, m.url.mmproj)

  const resolved = {
    generatedAt: new Date().toISOString(),
    modelsDir,
    label: m.label,
    quant: m.quant,
    hfRepo: m.hfRepo,
    hfRevision: m.hfRevision,
    llmPath,
    mmprojPath,
    provenance: { llm: llmProvenance, mmproj: mmprojProvenance },
    model: { id: m.id, ctxSize: m.ctxSize, nPredict: m.nPredict }
  }

  const outPath = path.join(SCRIPT_DIR, 'resolved-models.json')
  fs.writeFileSync(outPath, JSON.stringify(resolved, null, 2) + '\n', 'utf8')
  log(`wrote ${outPath}`)
}

main().catch((err) => {
  console.error(`[prepare-models] failed: ${err && err.message ? err.message : String(err)}`)
  process.exit(1)
})
