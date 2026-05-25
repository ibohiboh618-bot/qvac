'use strict'

// Resolves benchmark sources to concrete entrypoints the runners use.
//
// Source types:
//   addon — JS addon via require('@qvac/llm-llamacpp'), driven by case-runner.js (Bare)
//   cli   — native llama-mtmd-cli binary, driven by cli-case-runner.js (Node.js)
//
// The 3-source comparison (addon vs fabric-cli vs upstream-cli) runs
// the same model through three inference engines to measure JS binding
// overhead and fork divergence.

const fs = require('fs')
const path = require('path')

const SCRIPT_DIR = __dirname
const CLI_RESOLVED_PATH = path.join(SCRIPT_DIR, 'cli-sources-resolved.json')

function loadCliResolved () {
  if (!fs.existsSync(CLI_RESOLVED_PATH)) return {}
  try { return JSON.parse(fs.readFileSync(CLI_RESOLVED_PATH, 'utf8')) } catch { return {} }
}

function resolveAddonSource (key) {
  return {
    key,
    type: 'addon',
    label: 'addon@npm',
    addonPath: null
  }
}

function resolveCliSource (key, spec, cliOverrides) {
  const cliResolved = loadCliResolved()
  const configKey = spec.configKey || key

  // Direct binary override via --fabric-binary=<path> or --upstream-binary=<path>
  const overrideKey = `${key}-binary`
  if (cliOverrides && cliOverrides[overrideKey]) {
    const binaryPath = cliOverrides[overrideKey]
    if (!fs.existsSync(binaryPath)) {
      throw new Error(`--${overrideKey}=${binaryPath}: not found`)
    }
    return {
      key,
      type: 'cli',
      label: `${configKey}-cli@custom`,
      binaryPath,
      commitSha: null
    }
  }

  // Resolve from cli-sources-resolved.json (written by build-cli-sources.js)
  const resolved = cliResolved[configKey]
  if (resolved && resolved.binaryPath && fs.existsSync(resolved.binaryPath)) {
    return {
      key,
      type: 'cli',
      label: resolved.label || `${configKey}-cli`,
      binaryPath: resolved.binaryPath,
      commitSha: resolved.commitSha || null,
      provenance: resolved.provenance || null
    }
  }

  return {
    key,
    type: 'cli',
    label: `${configKey}-cli`,
    binaryPath: null,
    requiresBuild: true
  }
}

function resolveSources (config, args) {
  const enabledKeys = args.sources
    ? String(args.sources).split(',').map((s) => s.trim()).filter(Boolean)
    : Object.keys(config.sources).filter((k) => config.sources[k].enabled !== false)

  const cliOverrides = {}
  if (args['fabric-binary']) cliOverrides['fabric-binary'] = args['fabric-binary']
  if (args['upstream-binary']) cliOverrides['upstream-binary'] = args['upstream-binary']

  const out = []
  for (const key of enabledKeys) {
    const spec = config.sources[key]
    if (!spec) {
      console.warn(`[source-resolver] unknown source '${key}', skipping`)
      continue
    }
    if (spec.type === 'addon') {
      out.push(resolveAddonSource(key))
    } else if (spec.type === 'cli') {
      out.push(resolveCliSource(key, spec, cliOverrides))
    } else {
      throw new Error(`Unknown source type '${spec.type}' for source '${key}'`)
    }
  }
  return out
}

module.exports = { resolveSources }
