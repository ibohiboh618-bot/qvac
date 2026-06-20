#!/usr/bin/env node
'use strict'

// Desktop benchmark report renderer for the embed parameter sweep.
//
// Reads embed sweep JSON from --dir (recursively) and renders ONE markdown
// report:
//   - header with addon version, input token size, repeats-per-config, device
//   - one table per model: Config | ppTPS | latency (ms) | embeddings/sec |
//     cosine-similarity vs baseline
//   - a Coverage section comparing measured configs against the expected grid
//   - a "## Charts" mermaid section (ppTPS per config at a representative point)
//     plus a self-contained HTML chart artifact (--html) with ppTPS, similarity
//     and latency charts
//   - run-meta stamping (--addon-version is overridden by a stamped run-meta)
//
// Embedding is a single forward pass (prefill only): there is no decode phase,
// so there is NO TTFT, NO decode TPS and NO generated-tokens column. The input
// token size is stated in the report header, not per row.
//
// Input schema (desktop embed sweep):
//   { models:[{ modelId, cases:[{ quantization, runtimeConfig:{device,batchSize,
//     flashAttn}, metrics:{ppTpsMean,ppTpsStd,latencyMsMean,latencyMsStd,
//     embPerSecMean,embPerSecStd,inputTokens}, similarity:{avg,min,max},
//     status, isBaseline }]}], repeats, ... }

const fs = require('fs')
const path = require('path')
// Sweep axes (coverage denominator) + input modes are the single source of
// truth shared with the bare sweep. _sweep-grid is plain literals (no bare-fs),
// so it loads here under Node too — keeping the renderer's coverage grid from
// drifting out of step with what the sweep actually runs.
const { PARAMETER_SWEEP, INPUT_MODES } = require('./_sweep-grid')
// Mobile shard matrix (model x quant x batchSize x flashAttn cells) for the
// mobile coverage check, so
// the renderer scores a mobile run against the same source of truth the shard
// generator and the workflow test_groups derive from.
const { matrix, mobileShardKey } = require('../../test/integration/_benchmark-matrix')

function parseArgs (argv) {
  const a = {
    dir: null,
    output: null,
    html: null,
    chartsUrl: null,
    device: 'Desktop (linux-x64 GPU)',
    addonVersion: null
  }
  for (let i = 2; i < argv.length; i++) {
    const t = argv[i]
    if (t === '--dir') a.dir = argv[++i]
    else if (t === '--output') a.output = argv[++i]
    else if (t === '--html') a.html = argv[++i]
    else if (t === '--charts-url') a.chartsUrl = argv[++i]
    else if (t === '--device') a.device = argv[++i]
    else if (t === '--addon-version') a.addonVersion = argv[++i]
  }
  if (!a.dir) {
    throw new Error(
      'usage: render-report.js --dir <path> [--output <md>] [--html <html>] ' +
      '[--device <name>] [--addon-version <ver>] [--charts-url <url>]'
    )
  }
  return a
}

function walkJson (dir) {
  const out = []
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name)
    if (entry.isDirectory()) out.push(...walkJson(p))
    else if (entry.name.endsWith('.json')) out.push(p)
  }
  return out
}

function num (v) {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

function int (v) {
  const n = num(v)
  return n !== null ? Math.round(n) : null
}

// Collect rows + run metadata from every JSON file under a directory.
// Returns { rows, meta } where meta = { addonVersion, repeats, device }.
function loadDir (dir, deviceArg) {
  const files = walkJson(dir)
  // The desktop device name (incl. the detected GPU) is stamped into
  // run-meta.json at run time, so re-renders show the real device even though
  // the renderer ran elsewhere. Falls back to the passed/default name.
  let resolvedDevice = deviceArg
  for (const f of files) {
    try {
      const d = JSON.parse(fs.readFileSync(f, 'utf8'))
      if (d && typeof d.device === 'string' && d.device) { resolvedDevice = d.device; break }
    } catch {}
  }
  const meta = { addonVersion: null, repeats: null, expectedShards: null, device: resolvedDevice }
  const rows = []
  for (const f of files) rows.push(...rowsFromFile(f, resolvedDevice, meta))
  return { rows: collapseMobileRows(rows), meta }
}

// Mobile rows arrive one per config plus a Crashed placeholder emitted before
// the config ran; collapse each (device, config) to a single row, preferring a
// non-crashed real row over the placeholder. Desktop rows are left untouched.
function collapseMobileRows (rows) {
  const out = []
  const byKey = new Map()
  for (const r of rows) {
    if (!r.mobile) { out.push(r); continue }
    const k = `${r.device}@@${r.config}`
    const prev = byKey.get(k)
    if (!prev || (prev.crashed && !r.crashed)) byKey.set(k, r)
  }
  for (const r of byKey.values()) out.push(r)
  return out
}

// Normalise a report file into rows:
//   { model, config, ppTps, ppTpsStd, latency, latencyStd, embPerSec,
//     embPerSecStd, similarity, isBaseline, crashed, sampleCount }
// Also fills meta fields when found. The baseline row is kept (it anchors the
// per-model table and shows similarity 1.0).
function rowsFromFile (file, device, meta) {
  let doc
  try { doc = JSON.parse(fs.readFileSync(file, 'utf8')) } catch { return [] }
  const rows = []

  // run-meta.json — the addon version + mobile shard list stamped into the
  // run's artifacts at benchmark time, so a re-render reflects the version and
  // matrix THAT run targeted rather than whatever the code says now. Carries no
  // benchmark rows.
  if (doc && typeof doc.addonVersion === 'string') {
    if (meta.addonVersion === null) meta.addonVersion = doc.addonVersion
    if (meta.expectedShards === null && Array.isArray(doc.expectedShards)) {
      meta.expectedShards = doc.expectedShards
    }
    return rows
  }

  // run-meta.json variant that only stamps the device name — resolved in
  // loadDir's first pass; nothing to render from it here.
  if (doc && typeof doc.device === 'string' && !Array.isArray(doc.models)) return rows

  // Mobile perf-report schema: { device:{name}, results:[{test, status,
  // metrics:{pp_tps, pp_tps_std, latency_ms, latency_ms_std, emb_per_sec,
  // emb_per_sec_std, cosine_similarity, input_tokens, sample_count}}] }. One row
  // per config per device; the on-device runner emits a Crashed placeholder
  // before each config and a real row after, so a (device, config) may appear
  // twice and the real row supersedes the placeholder in collapseMobileRows.
  if (doc && doc.device && typeof doc.device === 'object' && Array.isArray(doc.results)) {
    const mobileDevice = (doc.device.name || 'unknown').trim()
    for (const r of doc.results) {
      const m = r.metrics || {}
      const crashed = (r.status && String(r.status).toLowerCase() === 'crashed') ||
        (num(m.pp_tps) === null && num(m.latency_ms) === null && num(m.emb_per_sec) === null)
      rows.push({
        device: mobileDevice,
        mobile: true,
        model: null,
        config: r.test || '(unknown)',
        quant: null,
        backend: null,
        batchSize: null,
        flashAttn: null,
        inputMode: null,
        isBaseline: false,
        ppTps: num(m.pp_tps),
        ppTpsStd: num(m.pp_tps_std),
        latency: num(m.latency_ms),
        latencyStd: num(m.latency_ms_std),
        embPerSec: num(m.emb_per_sec),
        embPerSecStd: num(m.emb_per_sec_std),
        similarity: num(m.cosine_similarity),
        crashed: !!crashed,
        partial: r.status === 'partial-failure',
        inputTokens: int(m.input_tokens),
        sampleCount: int(m.sample_count)
      })
    }
    return rows
  }

  if (!Array.isArray(doc.models) || !doc.models.length || !Array.isArray(doc.models[0].cases)) {
    return rows
  }

  if (num(doc.repeats) !== null && meta.repeats === null) meta.repeats = doc.repeats

  for (const model of doc.models) {
    for (const c of model.cases) {
      const rc = c.runtimeConfig || {}
      const m = c.metrics || {}
      // A config with status 'failed' (or any non-ok/non-partial) crashed and
      // produced no metrics. 'partial-failure' DID produce data but only from
      // the repeats that succeeded — kept as a real row, flagged so its smaller
      // sample is never read as a clean full run.
      const crashed = c.status && c.status !== 'ok' && c.status !== 'partial-failure'
      rows.push({
        device,
        model: model.modelId,
        config: configLabel({
          model: model.modelId,
          quant: c.quantization,
          device: rc.device,
          batchSize: rc.batchSize,
          flashAttn: rc.flashAttn,
          inputMode: c.inputMode,
          isBaseline: c.isBaseline
        }),
        quant: c.quantization,
        backend: rc.device || null,
        batchSize: rc.batchSize != null ? String(rc.batchSize) : null,
        flashAttn: rc.flashAttn != null ? String(rc.flashAttn) : null,
        inputMode: c.inputMode || null,
        isBaseline: !!c.isBaseline,
        ppTps: num(m.ppTpsMean),
        ppTpsStd: num(m.ppTpsStd),
        latency: num(m.latencyMsMean),
        latencyStd: num(m.latencyMsStd),
        embPerSec: num(m.embPerSecMean),
        embPerSecStd: num(m.embPerSecStd),
        similarity: c.similarity && num(c.similarity.avg) !== null ? num(c.similarity.avg) : null,
        crashed: !!crashed,
        partial: c.status === 'partial-failure',
        repeatsAttempted: int(c.repeatsAttempted),
        repeatsSucceeded: int(c.repeatsSucceeded),
        inputTokens: int(m.inputTokens),
        sampleCount: int(m.repeats)
      })
    }
  }
  return rows
}

// "[model q=Q8_0] [gpu] [bs=512] [fa=on] [input=array]". The baseline row's
// device/bs/fa are the reference config the similarity column is measured
// against, so they are shown rather than collapsed to "default".
function configLabel ({ model, quant, device, batchSize, flashAttn, inputMode, isBaseline }) {
  const parts = [`[${model} q=${quant}]`]
  if (device) parts.push(`[${device}]`)
  if (batchSize != null) parts.push(`[bs=${batchSize}]`)
  if (flashAttn != null) parts.push(`[fa=${flashAttn}]`)
  if (inputMode) parts.push(`[input=${inputMode}]`)
  if (isBaseline) parts.push('[baseline]')
  return parts.join(' ')
}

function fmt (v, decimals = 2) {
  if (v === null || v === undefined) return '-'
  return (Math.round(v * Math.pow(10, decimals)) / Math.pow(10, decimals)).toFixed(decimals)
}

// "mean ± std" when there is more than one sample; bare mean otherwise.
function fmtMS (meanV, stdV, sampleCount, decimals = 2) {
  if (meanV === null || meanV === undefined) return '-'
  if (stdV !== null && stdV !== undefined && sampleCount && sampleCount > 1) {
    return `${fmt(meanV, decimals)} ± ${fmt(stdV, decimals)}`
  }
  return fmt(meanV, decimals)
}

// Cosine similarity is dimensionless and reads best at 4 decimals; the baseline
// row is exactly 1.0 by construction.
function fmtSim (v) {
  if (v === null || v === undefined) return '-'
  return (Math.round(v * 1e4) / 1e4).toFixed(4)
}

// A model only runs the quants it actually ships, so the per-model coverage
// denominator is the sweep quant axis intersected with that model's manifest
// builds — mirroring buildCases in case-runner.js. The manifest is plain JSON
// (Node-loadable). Returns null if it cannot be read, so coverage can say so
// rather than silently treating every model as supporting every quant.
function manifestQuantsById () {
  let manifest
  try {
    manifest = JSON.parse(fs.readFileSync(path.join(__dirname, 'models.manifest.json'), 'utf8'))
  } catch {
    return null
  }
  const map = new Map()
  for (const m of (manifest.models || [])) {
    const quants = m.gguf && Array.isArray(m.gguf.quantizations) ? m.gguf.quantizations : []
    map.set(m.id, quants)
  }
  return map
}

function modelQuants (model, manifestQuants) {
  const supported = manifestQuants && manifestQuants.get(model)
  if (!supported || !supported.length) return PARAMETER_SWEEP.quantization
  return PARAMETER_SWEEP.quantization.filter((q) => supported.includes(q))
}

function expectedConfigKeys (model, quants) {
  const keys = []
  for (const quant of quants) {
    for (const device of PARAMETER_SWEEP.device) {
      for (const batchSize of PARAMETER_SWEEP.batchSize) {
        for (const flashAttn of PARAMETER_SWEEP.flashAttn) {
          for (const inputMode of INPUT_MODES) {
            keys.push(`${model}|${quant}|${device}|${batchSize}|${flashAttn}|${inputMode}`)
          }
        }
      }
    }
  }
  return keys
}

function rowConfigKey (r) {
  return `${r.model}|${r.quant}|${r.backend}|${r.batchSize}|${r.flashAttn}|${r.inputMode}`
}

// Per-model coverage of the expected sweep grid. A config that ran but crashed
// still counts as reported (it produced a placeholder row); a config with no
// row at all never ran, which keeps a partial sweep from rendering as complete.
function coverageLines (rows, models) {
  const mq = manifestQuantsById()
  const lines = ['## Coverage', '']
  lines.push(
    'Expected grid per model: (supported quants) x ' +
    `${PARAMETER_SWEEP.device.length} devices x ${PARAMETER_SWEEP.batchSize.length} batch sizes x ` +
    `${PARAMETER_SWEEP.flashAttn.length} flash-attn x ${INPUT_MODES.length} input modes, plus 1 baseline ` +
    `reference. Supported quants are the sweep set (${PARAMETER_SWEEP.quantization.join(', ')}) ` +
    "intersected with each model's manifest builds, so the denominator differs per model."
  )
  if (!mq) {
    lines.push('')
    lines.push(
      '> Note: the model manifest could not be read, so the denominator below falls back to the full ' +
      'quant axis and may overstate the expected count for models that ship fewer quants.'
    )
  }
  lines.push('')
  lines.push('| Model | Grid configs reported | Baseline |')
  lines.push('| --- | ---: | :---: |')

  const seenByModel = new Map(models.map((m) => [m, new Set()]))
  const baselineByModel = new Map(models.map((m) => [m, false]))
  for (const r of rows) {
    if (r.isBaseline) { baselineByModel.set(r.model, true); continue }
    const set = seenByModel.get(r.model)
    if (set) set.add(rowConfigKey(r))
  }

  for (const model of models) {
    const expected = new Set(expectedConfigKeys(model, modelQuants(model, mq)))
    const seen = [...seenByModel.get(model)].filter((k) => expected.has(k))
    lines.push(`| ${model} | ${seen.length} / ${expected.size} | ${baselineByModel.get(model) ? 'yes' : 'MISSING'} |`)
  }
  lines.push('')

  for (const model of models) {
    const expected = expectedConfigKeys(model, modelQuants(model, mq))
    const seen = seenByModel.get(model)
    const missing = expected.filter((k) => !seen.has(k))
    if (missing.length) {
      lines.push(`**${model}** is missing ${missing.length} grid config(s):`)
      for (const k of missing) lines.push(`- ${shardLabel(k)}`)
      lines.push('')
    }
  }
  return lines
}

function shardLabel (key) {
  const [, quant, device, batchSize, flashAttn, inputMode] = key.split('|')
  return `q=${quant} [${device}] [bs=${batchSize}] [fa=${flashAttn}] [input=${inputMode}]`
}

// Held axes for the headline / HTML charts. Each bar is one measured
// configuration (no averaging across quants, batch sizes or flash-attn). ppTPS
// is the headline metric; similarity and latency are additional HTML charts.
const CHART_BACKEND = 'gpu'
const CHART_BATCH = '512'
const CHART_FLASH = 'off'
const CHART_INPUT = 'array'
const CHART_QUANT_HELD = 'Q8_0'

function atConfig (rows, { backend, batchSize, flashAttn, inputMode, quant }) {
  return rows.filter((r) =>
    !r.isBaseline && !r.crashed &&
    (backend == null || r.backend === backend) &&
    (batchSize == null || r.batchSize === batchSize) &&
    (flashAttn == null || r.flashAttn === flashAttn) &&
    (inputMode == null || r.inputMode === inputMode) &&
    (quant == null || r.quant === quant)
  )
}

function mermaidBar (title, ylabel, labels, values) {
  const max = Math.ceil(Math.max(...values, 1) * 1.15)
  return [
    '```mermaid',
    'xychart-beta',
    `    title "${title}"`,
    `    x-axis [${labels.map((l) => `"${l}"`).join(', ')}]`,
    `    y-axis "${ylabel}" 0 --> ${max}`,
    `    bar [${values.map((v) => Math.round(v * 10) / 10).join(', ')}]`,
    '```'
  ]
}

// Headline at-a-glance chart: ppTPS per model at ONE fixed config. xychart-beta
// is single-series and cannot draw error bars, so the per-model breakdowns by
// quantization / batch size with stddev whiskers live in the HTML artifact.
function mermaidSection (rows, chartsUrl) {
  const held = { backend: CHART_BACKEND, batchSize: CHART_BATCH, flashAttn: CHART_FLASH, inputMode: CHART_INPUT, quant: CHART_QUANT_HELD }
  const pts = atConfig(rows, held).filter((r) => r.ppTps !== null)
  if (pts.length < 1) return []
  const byModel = new Map()
  for (const r of pts) if (!byModel.has(r.model)) byModel.set(r.model, r.ppTps)
  const models = [...byModel.keys()].sort((a, b) => byModel.get(b) - byModel.get(a))
  const cfg = `${CHART_QUANT_HELD}, ${CHART_BACKEND.toUpperCase()}, bs=${CHART_BATCH}, fa=${CHART_FLASH}, ${CHART_INPUT} input`
  // The download URL only exists after the artifact is uploaded, so the workflow
  // passes it in post-upload; a local render leaves the artifact name as plain text.
  const artifact = chartsUrl
    ? `[**embed-benchmark-charts** artifact](${chartsUrl})`
    : '**embed-benchmark-charts** artifact'
  return [
    '## Charts',
    '',
    `> At-a-glance prefill throughput (ppTPS) by model at one fixed config: **${cfg}**. ` +
    'Per-model charts broken down by quantization and batch size, plus the cosine-similarity ' +
    `and latency charts, are in the ${artifact} — download and open \`embed-benchmark-charts.html\` inside. ` +
    'The full grid is in the tables below.',
    '',
    ...mermaidBar(`ppTPS by model (${cfg})`, 'ppTPS', models, models.map((m) => byModel.get(m))),
    ''
  ]
}

// One token count is shown only if every measured config used the same input
// length. Single vs array input modes (and batch sizes) generally differ, so
// otherwise it is omitted rather than picking one case's value misleadingly.
function uniformInputTokens (rows) {
  const vals = [...new Set(rows.filter((r) => !r.crashed && r.inputTokens != null).map((r) => r.inputTokens))]
  return vals.length === 1 ? vals[0] : null
}

// Shard key for a mobile row, parsed from its
// "[<model> q=<quant>] [<device>] [bs=<N>] [fa=<on|off>] [input=...]" label, to
// match _benchmark-matrix.js mobileShardKey (<model>|<quant>|bs<N>|fa<on|off>).
// batchSize and flashAttn are the shard key; device and inputMode are swept
// within a shard, so they are excluded from the key.
function shardKeyOf (config) {
  const m = /^\[([^\]]+?)\s+q=([^\]]+)\]/.exec(config)
  if (!m) return null
  const bs = /\[bs=([^\]]+)\]/.exec(config)
  const fa = /\[fa=([^\]]+)\]/.exec(config)
  if (!bs || !fa) return null
  return `${m[1]}|${m[2]}|bs${bs[1]}|fa${fa[1]}`
}

function shardKeyLabel (key) {
  const [model, quant, bs, fa] = key.split('|')
  return `${model} q=${quant} ${bs} ${fa}`
}

// Per-device coverage of the mobile shard matrix (model x quant x batchSize x
// flashAttn cells). Every
// shard that runs emits at least a Crashed placeholder for each config, so a
// shard with no row at all never ran or its data was lost (e.g. a dropped batch
// artifact). Surfacing this keeps a partial run from rendering as complete.
function mobileCoverageLines (rows, devices, expectedShards) {
  // Prefer the shard list stamped into the run's run-meta (so a re-render scores
  // against the matrix THAT run targeted); fall back to the live matrix.
  const expected = expectedShards && expectedShards.length
    ? expectedShards
    : matrix().map(mobileShardKey)
  const expectedSet = new Set(expected)
  const lines = ['## Coverage', '']
  if (!devices.length) {
    lines.push(
      `**Warning: 0 mobile devices reported.** ${expected.length} shards expected per device. ` +
      'If mobile was enabled for this run, its data was lost (failed job or dropped artifacts).',
      ''
    )
    return lines
  }
  lines.push(
    `Mobile matrix: ${expected.length} shards (model x quant x batch size x flash-attn) expected per device. ` +
    `${devices.length} device(s) reported.`
  )
  lines.push('')
  lines.push('| Device | Shards reported |')
  lines.push('| --- | ---: |')
  const seenByDevice = new Map(devices.map((d) => [d, new Set()]))
  const seenAll = new Set()
  for (const r of rows) {
    const k = shardKeyOf(r.config)
    if (!k || !expectedSet.has(k) || !seenByDevice.has(r.device)) continue
    seenByDevice.get(r.device).add(k)
    seenAll.add(k)
  }
  for (const d of devices) lines.push(`| ${d} | ${seenByDevice.get(d).size} / ${expected.length} |`)
  lines.push('')
  const missingEverywhere = expected.filter((k) => !seenAll.has(k))
  if (missingEverywhere.length) {
    lines.push(`**${missingEverywhere.length} shard(s) produced no data on any device** (likely a dropped batch):`)
    for (const k of missingEverywhere) lines.push(`- ${shardKeyLabel(k)}`)
    lines.push('')
  }
  for (const d of devices) {
    const miss = expected.filter((k) => seenAll.has(k) && !seenByDevice.get(d).has(k))
    if (miss.length) {
      lines.push(`**${d}** is missing ${miss.length} shard(s) other devices reported:`)
      for (const k of miss) lines.push(`- ${shardKeyLabel(k)}`)
      lines.push('')
    }
  }
  return lines
}

// Mobile report: one table per device with the same columns as the desktop
// per-model tables (ppTPS | latency (ms) | embeddings/sec | cosine-similarity),
// plus mobile coverage scored against the (model x quant x batchSize x
// flashAttn) shard matrix.
function renderMobile (rows, meta, addonVersionArg, heading = '# Embed Benchmark Results') {
  const addonVersion = meta.addonVersion || addonVersionArg || null
  const byDevice = new Map()
  for (const r of rows) {
    if (!byDevice.has(r.device)) byDevice.set(r.device, [])
    byDevice.get(r.device).push(r)
  }
  const devices = [...byDevice.keys()].sort()

  const lines = []
  lines.push(heading)
  lines.push('')
  const metaParts = []
  if (addonVersion) metaParts.push(`**Addon:** \`${addonVersion}\``)
  if (devices.length) metaParts.push(`**Devices:** ${devices.join(', ')}`)
  if (metaParts.length) {
    lines.push(metaParts.join(' · '))
    lines.push('')
  }
  lines.push(
    'Metrics are addon `runtimeStats` (embedding = single prefill pass, no decode): ' +
    'ppTPS = prefill tokens/sec, latency = prefill time (ms), embeddings/sec = sequences ' +
    'embedded per second, cosine-similarity = avg cosine similarity of each config\'s ' +
    'embeddings vs the per-input-mode baseline (the first successful config), which reads ' +
    '1.0 by construction. `Crashed` = configuration crashed or produced no output.'
  )
  lines.push('')
  lines.push(
    'Config labels read `[model q=<quant>] [gpu|cpu] [bs=<batch>] [fa=<on|off>] [input=<single|array>]`. ' +
    'Each mobile shard is one (model, quant, batch size, flash-attn) cell and sweeps device x input mode.'
  )
  lines.push('')

  for (const l of mobileCoverageLines(rows, devices, meta.expectedShards)) lines.push(l)

  for (const device of devices) {
    const items = byDevice.get(device).slice().sort((a, b) => a.config.localeCompare(b.config))
    lines.push(`## ${device}`)
    lines.push('')
    lines.push('| Config | ppTPS | latency (ms) | embeddings/sec | cosine-similarity |')
    lines.push('| --- | ---: | ---: | ---: | ---: |')
    for (const r of items) {
      if (r.crashed) {
        lines.push(`| ${r.config} | Crashed | Crashed | Crashed | - |`)
      } else {
        lines.push(
          `| ${r.config} | ${fmtMS(r.ppTps, r.ppTpsStd, r.sampleCount)} | ` +
          `${fmtMS(r.latency, r.latencyStd, r.sampleCount)} | ` +
          `${fmtMS(r.embPerSec, r.embPerSecStd, r.sampleCount)} | ${fmtSim(r.similarity)} |`
        )
      }
    }
    lines.push('')
  }

  lines.push('## Best configuration per device')
  lines.push('')
  lines.push('| Device | Highest ppTPS | Highest embeddings/sec |')
  lines.push('| --- | --- | --- |')
  for (const device of devices) {
    const ok = byDevice.get(device).filter((r) => !r.crashed)
    const bestPp = ok.filter((r) => r.ppTps !== null).sort((a, b) => b.ppTps - a.ppTps)[0]
    const bestEmb = ok.filter((r) => r.embPerSec !== null).sort((a, b) => b.embPerSec - a.embPerSec)[0]
    const ppCell = bestPp ? `${bestPp.config} — ${fmt(bestPp.ppTps)}` : '-'
    const embCell = bestEmb ? `${bestEmb.config} — ${fmt(bestEmb.embPerSec)}` : '-'
    lines.push(`| ${device} | ${ppCell} | ${embCell} |`)
  }
  lines.push('')
  return lines.join('\n') + '\n'
}

function render (rows, meta, addonVersionArg, chartsUrl) {
  // Mobile perf-report rows are device-keyed; the desktop sweep is model-keyed
  // with repeats. Render each from its OWN rows so a combined run (both present)
  // shows a "— Desktop" and a "— Mobile" section, and a single-kind run shows
  // just that one. (A desktop-only run keeps the bare "# Embed Benchmark Results".)
  const desktopRows = rows.filter((r) => !r.mobile)
  const mobileRows = rows.filter((r) => r.mobile)
  if (!desktopRows.length) return renderMobile(mobileRows, meta, addonVersionArg)

  const models = [...new Set(desktopRows.map((r) => r.model))].sort()
  // Stamped version (from run-meta) wins over any manually-passed value.
  const addonVersion = meta.addonVersion || addonVersionArg || null
  const inputTokens = uniformInputTokens(desktopRows)

  const lines = []
  lines.push(mobileRows.length ? '# Embed Benchmark Results — Desktop' : '# Embed Benchmark Results')
  lines.push('')

  const metaParts = []
  if (addonVersion) metaParts.push(`**Addon:** \`${addonVersion}\``)
  metaParts.push(`**Device:** ${meta.device}`)
  if (inputTokens !== null) metaParts.push(`**Input:** ${inputTokens} tokens`)
  if (meta.repeats !== null) metaParts.push(`**Repeats:** ${meta.repeats}`)
  if (metaParts.length) {
    lines.push(metaParts.join(' · '))
    lines.push('')
  }

  lines.push(
    'Metrics are addon `runtimeStats` (embedding = single prefill pass, no decode): ' +
    'ppTPS = prefill tokens/sec, latency = prefill time (ms), embeddings/sec = sequences ' +
    'embedded per second, cosine-similarity = avg cosine similarity of each config\'s ' +
    'embeddings vs the baseline\'s. The baseline (highest-fidelity quant, cpu, bs256, ' +
    'flash-off) reads 1.0 by construction. `Crashed` = configuration crashed or produced no output.'
  )
  lines.push('')
  lines.push(
    'Config labels read `[model q=<quant>] [gpu|cpu] [bs=<batch>] [fa=<on|off>] [input=<single|array>]`. ' +
    'A `(partial: N/M repeats)` note means only N of M repeats succeeded, so that row\'s stats are over ' +
    'fewer samples. Where input length is uniform across configs it is shown in the header above.'
  )
  lines.push('')

  for (const l of coverageLines(desktopRows, models)) lines.push(l)

  for (const l of mermaidSection(desktopRows, chartsUrl)) lines.push(l)

  const byModel = new Map()
  for (const r of desktopRows) {
    if (!byModel.has(r.model)) byModel.set(r.model, [])
    byModel.get(r.model).push(r)
  }

  for (const model of models) {
    const items = byModel.get(model).slice().sort((a, b) => {
      if (a.isBaseline !== b.isBaseline) return a.isBaseline ? -1 : 1
      return a.config.localeCompare(b.config)
    })
    lines.push(`## ${model}`)
    lines.push('')
    lines.push('| Config | ppTPS | latency (ms) | embeddings/sec | cosine-similarity |')
    lines.push('| --- | ---: | ---: | ---: | ---: |')
    for (const r of items) {
      if (r.crashed) {
        lines.push(`| ${r.config} | Crashed | Crashed | Crashed | - |`)
      } else {
        const note = r.partial && r.repeatsSucceeded !== null && r.repeatsAttempted !== null
          ? ` _(partial: ${r.repeatsSucceeded}/${r.repeatsAttempted} repeats)_`
          : ''
        lines.push(
          `| ${r.config}${note} | ${fmtMS(r.ppTps, r.ppTpsStd, r.sampleCount)} | ` +
          `${fmtMS(r.latency, r.latencyStd, r.sampleCount)} | ` +
          `${fmtMS(r.embPerSec, r.embPerSecStd, r.sampleCount)} | ${fmtSim(r.similarity)} |`
        )
      }
    }
    lines.push('')
  }

  lines.push('## Best configuration per model')
  lines.push('')
  lines.push('| Model | Highest ppTPS | Highest embeddings/sec |')
  lines.push('| --- | --- | --- |')
  for (const model of models) {
    const ok = byModel.get(model).filter((r) => !r.crashed && !r.isBaseline)
    const bestPp = ok.filter((r) => r.ppTps !== null).sort((a, b) => b.ppTps - a.ppTps)[0]
    const bestEmb = ok.filter((r) => r.embPerSec !== null).sort((a, b) => b.embPerSec - a.embPerSec)[0]
    const ppCell = bestPp ? `${bestPp.config} — ${fmt(bestPp.ppTps)}` : '-'
    const embCell = bestEmb ? `${bestEmb.config} — ${fmt(bestEmb.embPerSec)}` : '-'
    lines.push(`| ${model} | ${ppCell} | ${embCell} |`)
  }
  lines.push('')

  // Combined run: append the mobile per-device section under its own heading.
  if (mobileRows.length) {
    lines.push(renderMobile(mobileRows, meta, addonVersionArg, '# Embed Benchmark Results — Mobile'))
  }
  return lines.join('\n') + '\n'
}

// ── Visual HTML report: self-contained inline SVG bar charts, no deps or CDN ──
const CHART_COLORS = ['#2563eb', '#dc2626', '#16a34a', '#d97706', '#7c3aed', '#0891b2', '#db2777']

// One bar per model for each category of the varied axis. The caller passes rows
// already pinned to a single point on every OTHER axis (via atConfig), so each
// (model, category) is one measured config: the bar is that row's mean and the
// whisker its own stddev.
function chartSeries (rows, byKey, order, metric, stdKey) {
  const pts = rows.filter((r) => r[metric] !== null && byKey(r) !== null)
  const present = new Set(pts.map((r) => byKey(r)))
  const cats = order.filter((c) => present.has(c))
  const models = [...new Set(pts.map((r) => r.model))].sort()
  const series = models.map((model, i) => ({
    name: model,
    color: CHART_COLORS[i % CHART_COLORS.length],
    cells: cats.map((cat) => {
      const row = pts.find((r) => r.model === model && byKey(r) === cat)
      return row ? { mean: row[metric], std: row[stdKey] != null ? row[stdKey] : null } : null
    })
  }))
  return { cats, series }
}

function svgBarChart (title, unit, cats, series, maxOverride) {
  const W = 860; const H = 360
  const m = { l: 64, r: 16, t: 16, b: 70 }
  const pw = W - m.l - m.r; const ph = H - m.t - m.b
  let max = maxOverride || 0
  if (!maxOverride) for (const s of series) for (const c of s.cells) if (c) max = Math.max(max, c.mean + (c.std || 0))
  const niceMax = (max > 0 ? max : 1) * 1.1
  const y = (v) => m.t + ph - (v / niceMax) * ph
  const out = ['<svg viewBox="0 0 ' + W + ' ' + H + '" width="100%" preserveAspectRatio="xMidYMid meet" font-family="system-ui,Arial" font-size="12">']
  for (let g = 0; g <= 4; g++) {
    const v = (niceMax / 4) * g; const yy = y(v)
    out.push(`<line x1="${m.l}" y1="${yy.toFixed(1)}" x2="${W - m.r}" y2="${yy.toFixed(1)}" stroke="#e5e7eb"/>`)
    out.push(`<text x="${m.l - 6}" y="${(yy + 4).toFixed(1)}" text-anchor="end" fill="#6b7280">${fmt(v, v < 10 ? 1 : 0)}</text>`)
  }
  const groupW = pw / cats.length
  const barW = (groupW * 0.74) / series.length
  cats.forEach((cat, ci) => {
    const gx = m.l + ci * groupW + groupW * 0.13
    series.forEach((s, si) => {
      const cell = s.cells[ci]
      if (!cell) return
      const bx = gx + si * barW
      const by = y(cell.mean)
      out.push(`<rect x="${bx.toFixed(1)}" y="${by.toFixed(1)}" width="${(barW * 0.92).toFixed(1)}" height="${Math.max(0, m.t + ph - by).toFixed(1)}" fill="${s.color}"><title>${s.name} | ${cat}: ${fmt(cell.mean)} ${unit}</title></rect>`)
      if (cell.std) {
        const cx = (bx + barW * 0.46).toFixed(1)
        out.push(`<line x1="${cx}" y1="${y(cell.mean + cell.std).toFixed(1)}" x2="${cx}" y2="${y(Math.max(0, cell.mean - cell.std)).toFixed(1)}" stroke="#1f2937"/>`)
      }
    })
    out.push(`<text x="${(gx + groupW * 0.37).toFixed(1)}" y="${m.t + ph + 18}" text-anchor="middle" fill="#111827">${cat}</text>`)
  })
  out.push('</svg>')
  return `<figure style="margin:0 0 26px"><figcaption style="font-weight:600;margin:0 0 4px">${title} <span style="font-weight:400;color:#6b7280">(${unit})</span></figcaption>${out.join('')}</figure>`
}

function renderHtml (rows, meta, addonVersionArg) {
  const addonVersion = meta.addonVersion || addonVersionArg || ''
  // Charts are desktop-only (the sweep axes); mobile rows carry no chart axes.
  const measured = rows.filter((r) => !r.mobile && !r.isBaseline && !r.crashed)
  const models = [...new Set(measured.map((r) => r.model))].sort()
  const legend = models.map((mName, i) => `<span style="display:inline-flex;align-items:center;margin:0 14px 6px 0"><span style="width:12px;height:12px;background:${CHART_COLORS[i % CHART_COLORS.length]};display:inline-block;margin-right:5px;border-radius:2px"></span>${mName}</span>`).join('')
  const QO = PARAMETER_SWEEP.quantization.slice()
  const BO = PARAMETER_SWEEP.batchSize.map((b) => String(b))
  const byQuant = (r) => r.quant
  const byBatch = (r) => r.batchSize
  // [title, unit, byKey, order, metric, stdKey, held]. Each chart varies one axis
  // and holds the rest at a fixed point, so each bar is one measured config.
  const defs = [
    ['ppTPS by quantization', 'ppTPS, tokens/sec', byQuant, QO, 'ppTps', 'ppTpsStd', { batchSize: CHART_BATCH }],
    ['ppTPS by batch size', 'ppTPS, tokens/sec', byBatch, BO, 'ppTps', 'ppTpsStd', { quant: CHART_QUANT_HELD }],
    ['cosine-similarity by quantization', 'cosine similarity', byQuant, QO, 'similarity', 'noStd', { batchSize: CHART_BATCH }],
    ['latency by quantization', 'latency, ms (lower is better)', byQuant, QO, 'latency', 'latencyStd', { batchSize: CHART_BATCH }]
  ]
  const desktopRows = rows.filter((r) => !r.mobile)
  let charts = ''
  for (const [title, unit, byKey, order, metric, stdKey, extra] of defs) {
    const subset = atConfig(desktopRows, { backend: CHART_BACKEND, flashAttn: CHART_FLASH, inputMode: CHART_INPUT, ...extra })
    const { cats, series } = chartSeries(subset, byKey, order, metric, stdKey)
    if (!cats.length) continue
    charts += svgBarChart(title, unit, cats, series)
  }
  const inputTokens = uniformInputTokens(desktopRows)
  const metaBits = [addonVersion && `Addon <code>${addonVersion}</code>`, `Device ${meta.device}`, inputTokens && `Input ${inputTokens} tok`].filter(Boolean).join(' &middot; ')
  const caption = `Each bar is one measured configuration on <b>${CHART_BACKEND.toUpperCase()}, flash-attn ${CHART_FLASH}, ${CHART_INPUT} input</b>. The quantization charts hold batch size at ${CHART_BATCH}; the batch-size chart holds the quant at ${CHART_QUANT_HELD}. Configs are never averaged together. Whiskers are &plusmn;1 stddev over the repeats. cosine-similarity is each config's embeddings vs the baseline (cpu, highest-fidelity quant). A missing bar means that configuration crashed, was not run, or had no measurable metric for this point. The full grid is in the report tables.`
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Embed Benchmark Charts</title></head><body style="max-width:920px;margin:24px auto;padding:0 16px;font-family:system-ui,Arial;color:#111827"><h1 style="font-size:20px;margin-bottom:2px">Embed Benchmark Charts</h1><p style="color:#6b7280;margin-top:0">${metaBits}</p><p style="color:#374151">${caption}</p><div style="margin:8px 0 20px">${legend}</div>${charts || '<p>No data to chart.</p>'}</body></html>`
}

function main () {
  const args = parseArgs(process.argv)

  const { rows, meta } = loadDir(args.dir, args.device)

  if (rows.length === 0) {
    // Metadata-only artifacts (run-meta) are valid JSON but carry no rows; a
    // "any JSON present" precheck cannot tell them apart from real results. Fail
    // so a run that produced no benchmark data never renders as a green,
    // complete-looking report.
    const msg = 'No benchmark results found.\n'
    if (args.output) fs.writeFileSync(args.output, msg)
    else process.stdout.write(msg)
    process.exitCode = 1
    return
  }

  const md = render(rows, meta, args.addonVersion, args.chartsUrl)
  if (args.output) fs.writeFileSync(args.output, md)
  else process.stdout.write(md)

  // HTML charts are a desktop-sweep artifact (per-quant / per-batch SVG bars).
  // Mobile rows carry no per-axis chart fields, so the mobile report is the
  // per-device markdown tables only — skip the empty HTML rather than emit it.
  const isMobileRun = rows.every((r) => r.mobile)
  if (args.html && !isMobileRun) fs.writeFileSync(args.html, renderHtml(rows, meta, args.addonVersion))
}

main()
