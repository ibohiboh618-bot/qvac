#!/usr/bin/env node
'use strict'

// Desktop benchmark report renderer for the embed parameter sweep.
//
// Reads embed sweep JSON from --dir (recursively) and renders ONE markdown
// report:
//   - header with addon version, input token size, repeats-per-config, device
//   - one `## <device>` section, each with a `### <model>` table: Config |
//     ppTPS | latency (ms) | embeddings/sec | cosine-similarity vs baseline
//   - a Coverage section comparing measured configs against the expected grid
//   - a "## Charts" mermaid section (base-case ppTPS per model: each model's
//     baseline-config row) plus a self-contained HTML chart artifact (--html)
//     with the base-case-per-model headline, prefill-throughput-vs-batch-size and
//     batched-throughput-vs-sequence-count scaling charts (one series per model),
//     and cosine-similarity-vs-F16 and latency charts
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
      const config = r.test || '(unknown)'
      rows.push({
        device: mobileDevice,
        mobile: true,
        model: modelOf(config),
        config,
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

// Bare model name from a "[<model> q=<quant>] ..." config label, used to group
// mobile rows (which carry no modelId field) under a model sub-heading. Returns
// '(unknown)' when the label is not in the expected shape.
function modelOf (config) {
  const m = /^\[([^\]]+?)\s+q=/.exec(config)
  return m ? m[1] : '(unknown)'
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

// Representative held point for the HTML scaling charts: the rows are pinned to
// a single point on every axis the chart is NOT varying, so each bar is one
// measured config (no averaging). ppTPS is the headline metric (the per-model
// base case); the scaling charts vary batch size or sequence count.
const CHART_BACKEND = 'gpu'
const CHART_BATCH = '512'
const CHART_FLASH = 'on'
const CHART_INPUT = 'single'

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

// The sequence count a swept input mode embeds in one call: 'single' is 1,
// 'array-N' is N. Drives the batched-throughput (embeddings/sec) chart's X axis.
function sequenceCount (inputMode) {
  if (inputMode === 'single') return 1
  const m = /^array-(\d+)$/.exec(inputMode || '')
  return m ? Number(m[1]) : null
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

// The per-model base case: the baseline row (default config: cpu / fa=off /
// bs=256 / single / highest-fidelity quant) carries the reference ppTPS each
// other config's similarity is measured against. The baseline runs one row per
// input mode; the headline uses the 'single' one (the default-config base case),
// falling back to any baseline row for the model if 'single' is absent.
function baseCaseByModel (rows) {
  const byModel = new Map()
  for (const r of rows) {
    if (!r.isBaseline || r.crashed || r.ppTps === null) continue
    const prev = byModel.get(r.model)
    if (!prev || (prev.inputMode !== 'single' && r.inputMode === 'single')) byModel.set(r.model, r)
  }
  return byModel
}

// Headline at-a-glance chart: the base case (baseline-config ppTPS) per model,
// one bar per model. xychart-beta is single-series, so the batch-size and
// sequence-count scaling charts (one series per model) live in the HTML artifact.
function mermaidSection (rows, chartsUrl) {
  const byModel = baseCaseByModel(rows)
  if (byModel.size < 1) return []
  const models = [...byModel.keys()].sort((a, b) => byModel.get(b).ppTps - byModel.get(a).ppTps)
  const cfg = 'cpu / fa=off / bs=256 / single'
  const title = `Base-case prefill throughput per model (ppTPS, baseline config: ${cfg})`
  // The download URL only exists after the artifact is uploaded, so the workflow
  // passes it in post-upload; a local render leaves the artifact name as plain text.
  const artifact = chartsUrl
    ? `[**embed-benchmark-charts** artifact](${chartsUrl})`
    : '**embed-benchmark-charts** artifact'
  return [
    '## Charts',
    '',
    `> Base-case prefill throughput (ppTPS) per model at the baseline config **${cfg}**. ` +
    'The prefill-throughput-vs-batch-size and batched-throughput-vs-sequence-count scaling charts ' +
    `(one series per model), plus the cosine-similarity and latency charts, are in the ${artifact} ` +
    '— download and open `embed-benchmark-charts.html` inside. The full grid is in the tables below.',
    '',
    ...mermaidBar(title, 'ppTPS', models, models.map((m) => byModel.get(m).ppTps)),
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

// A device's rows rendered as one `### <model>` sub-section per model, each with
// the metric table (ppTPS | latency | embeddings/sec | cosine-similarity).
// Rows within a model are baseline-first then config-sorted. Used by both the
// desktop and mobile sections so every device groups model-first.
function deviceModelTables (rows) {
  const byModel = new Map()
  for (const r of rows) {
    if (!byModel.has(r.model)) byModel.set(r.model, [])
    byModel.get(r.model).push(r)
  }
  const models = [...byModel.keys()].sort()
  const lines = []
  for (const model of models) {
    const items = byModel.get(model).slice().sort((a, b) => {
      if (a.isBaseline !== b.isBaseline) return a.isBaseline ? -1 : 1
      return a.config.localeCompare(b.config)
    })
    lines.push(`### ${model}`)
    lines.push('')
    lines.push('| Config | ppTPS | latency (ms) | embeddings/sec | cosine-similarity |')
    lines.push('| --- | ---: | ---: | ---: | ---: |')
    for (const r of items) {
      if (r.crashed) {
        lines.push(`| ${r.config} | Crashed | Crashed | Crashed | - |`)
      } else {
        const note = r.partial && r.repeatsSucceeded != null && r.repeatsAttempted != null
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
  return lines
}

// Mobile report: per device, one `### <model>` table with the same columns as
// the desktop tables (ppTPS | latency (ms) | embeddings/sec | cosine-similarity),
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
    'Config labels read `[model q=<quant>] [gpu|cpu] [bs=<batch>] [fa=<on|off>] [input=<single|array-N>]`, ' +
    'where `single` embeds one sequence and `array-N` embeds N sequences in one call. ' +
    'Each mobile shard is one (model, quant, batch size, flash-attn) cell and sweeps device x input mode.'
  )
  lines.push('')

  for (const l of mobileCoverageLines(rows, devices, meta.expectedShards)) lines.push(l)

  for (const device of devices) {
    lines.push(`## ${device}`)
    lines.push('')
    for (const l of deviceModelTables(byDevice.get(device))) lines.push(l)
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
  // Both desktop and mobile group device-first then model; the desktop sweep
  // carries repeats while mobile is one perf-report per device. Render each from
  // its OWN rows so a combined run (both present) shows a "— Desktop" and a
  // "— Mobile" section, and a single-kind run shows just that one. (A
  // desktop-only run keeps the bare "# Embed Benchmark Results".)
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
    'Config labels read `[model q=<quant>] [gpu|cpu] [bs=<batch>] [fa=<on|off>] [input=<single|array-N>]`, ' +
    'where `single` embeds one sequence and `array-N` embeds N sequences in one call. ' +
    'A `(partial: N/M repeats)` note means only N of M repeats succeeded, so that row\'s stats are over ' +
    'fewer samples. Where input length is uniform across configs it is shown in the header above.'
  )
  lines.push('')

  for (const l of coverageLines(desktopRows, models)) lines.push(l)

  for (const l of mermaidSection(desktopRows, chartsUrl)) lines.push(l)

  // Desktop is a single device (meta.device); group device-first then model so
  // the section nests `## <device>` › `### <model>` like the mobile section.
  const byDevice = new Map()
  for (const r of desktopRows) {
    if (!byDevice.has(r.device)) byDevice.set(r.device, [])
    byDevice.get(r.device).push(r)
  }
  const devices = [...byDevice.keys()].sort()

  for (const device of devices) {
    lines.push(`## ${device}`)
    lines.push('')
    for (const l of deviceModelTables(byDevice.get(device))) lines.push(l)
  }

  lines.push('## Best configuration per device')
  lines.push('')
  lines.push('| Device | Highest ppTPS | Highest embeddings/sec |')
  lines.push('| --- | --- | --- |')
  for (const device of devices) {
    const ok = byDevice.get(device).filter((r) => !r.crashed && !r.isBaseline)
    const bestPp = ok.filter((r) => r.ppTps !== null).sort((a, b) => b.ppTps - a.ppTps)[0]
    const bestEmb = ok.filter((r) => r.embPerSec !== null).sort((a, b) => b.embPerSec - a.embPerSec)[0]
    const ppCell = bestPp ? `${bestPp.config} — ${fmt(bestPp.ppTps)}` : '-'
    const embCell = bestEmb ? `${bestEmb.config} — ${fmt(bestEmb.embPerSec)}` : '-'
    lines.push(`| ${device} | ${ppCell} | ${embCell} |`)
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

// One bar per model on a single shared category: the base case per model. Built
// as one series whose cells are each model's baseline ppTPS, so svgBarChart
// renders one bar per model side by side.
function baseCaseSeries (rows) {
  const byModel = baseCaseByModel(rows)
  const models = [...byModel.keys()].sort((a, b) => byModel.get(b).ppTps - byModel.get(a).ppTps)
  return {
    cats: models,
    series: [{
      name: 'base case',
      color: CHART_COLORS[0],
      cells: models.map((m) => ({ mean: byModel.get(m).ppTps, std: null }))
    }]
  }
}

function renderHtml (rows, meta, addonVersionArg) {
  const addonVersion = meta.addonVersion || addonVersionArg || ''
  // Charts are desktop-only (the sweep axes); mobile rows carry no chart axes.
  const desktopRows = rows.filter((r) => !r.mobile)
  const measured = desktopRows.filter((r) => !r.isBaseline && !r.crashed)
  const models = [...new Set(measured.map((r) => r.model))].sort()
  const legend = models.map((mName, i) => `<span style="display:inline-flex;align-items:center;margin:0 14px 6px 0"><span style="width:12px;height:12px;background:${CHART_COLORS[i % CHART_COLORS.length]};display:inline-block;margin-right:5px;border-radius:2px"></span>${mName}</span>`).join('')
  const BO = PARAMETER_SWEEP.batchSize.map((b) => String(b))
  const SO = INPUT_MODES.map((m) => String(sequenceCount(m))).filter((s) => s !== 'null')
  const byBatch = (r) => r.batchSize
  const bySeq = (r) => { const n = sequenceCount(r.inputMode); return n === null ? null : String(n) }

  let charts = ''

  // 1. Headline: base case (baseline-config ppTPS) per model, one bar per model.
  const base = baseCaseSeries(desktopRows)
  if (base.cats.length) {
    charts += svgBarChart(
      'Base-case prefill throughput per model (baseline config: cpu / fa=off / bs=256 / single)',
      'ppTPS, prefill tokens/sec', base.cats, base.series
    )
  }

  // F16 is the reference quant; pick the held quant per model: F16 if any F16 row
  // exists for the model in the subset, else the model's base-case quant, so a
  // model lacking an F16 row at the held point still gets a series.
  function pinQuant (subset) {
    const baseQuant = baseCaseByModel(desktopRows)
    return subset.filter((r) => {
      const hasF16 = subset.some((x) => x.model === r.model && x.quant === 'F16')
      if (hasF16) return r.quant === 'F16'
      const bc = baseQuant.get(r.model)
      return bc ? r.quant === bc.quant : false
    })
  }

  // 2. Prefill throughput vs batch size: X = batch size, one series per model,
  // Y = ppTPS. Held gpu / fa=on / single, quant F16 (fallback baseline quant).
  {
    const subset = pinQuant(atConfig(desktopRows, { backend: CHART_BACKEND, flashAttn: CHART_FLASH, inputMode: CHART_INPUT }))
    const { cats, series } = chartSeries(subset, byBatch, BO, 'ppTps', 'ppTpsStd')
    if (cats.length) charts += svgBarChart('Prefill throughput vs batch size', 'ppTPS, prefill tokens/sec', cats, series)
  }

  // 3. Batched throughput vs sequence count: X = sequence count (1 / 5 / 10 / 20),
  // one series per model, Y = embeddings/sec. Held gpu / fa=on / bs=512, quant F16.
  {
    const subset = pinQuant(atConfig(desktopRows, { backend: CHART_BACKEND, flashAttn: CHART_FLASH, batchSize: CHART_BATCH }))
    const { cats, series } = chartSeries(subset, bySeq, SO, 'embPerSec', 'embPerSecStd')
    if (cats.length) charts += svgBarChart('Batched throughput vs sequence count', 'embeddings/sec', cats, series)
  }

  // 4. Cosine-similarity vs F16 and latency, per batch size, one series per model.
  {
    const subset = pinQuant(atConfig(desktopRows, { backend: CHART_BACKEND, flashAttn: CHART_FLASH, inputMode: CHART_INPUT }))
    const sim = chartSeries(subset, byBatch, BO, 'similarity', 'noStd')
    if (sim.cats.length) charts += svgBarChart('Cosine-similarity vs F16 baseline by batch size', 'cosine similarity', sim.cats, sim.series, 1)
    const lat = chartSeries(subset, byBatch, BO, 'latency', 'latencyStd')
    if (lat.cats.length) charts += svgBarChart('Prefill latency by batch size', 'latency (ms), lower is better', lat.cats, lat.series)
  }

  const inputTokens = uniformInputTokens(desktopRows)
  const metaBits = [addonVersion && `Addon <code>${addonVersion}</code>`, `Device ${meta.device}`, inputTokens && `Input ${inputTokens} tok`].filter(Boolean).join(' &middot; ')
  const caption = `Embedding is a single prefill pass: <b>ppTPS</b> = prefill tokens/sec, <b>embeddings/sec</b> = sequences embedded per second, <b>latency (ms)</b> = prefill time, <b>cosine-similarity</b> = each config's embeddings vs the F16 baseline. The headline is the base case (baseline config cpu / fa=off / bs=256 / single) per model. The scaling charts hold the other axes at <b>${CHART_BACKEND.toUpperCase()}, flash-attn ${CHART_FLASH}</b> (batch-size charts at single input; the sequence-count chart at batch size ${CHART_BATCH}) and quant <b>F16</b> (falling back to a model's base-case quant if its F16 row is absent at the held point). A missing bar means that configuration crashed, was not run, or had no measurable metric for this point. The full grid is in the report tables.`
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
