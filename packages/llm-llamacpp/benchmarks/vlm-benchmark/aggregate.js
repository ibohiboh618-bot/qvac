'use strict'
// QVAC-19178: aggregate [VLMROW] markers from one or more run logs into quality +
// speed matrices (markdown). Quality metrics mirror the local lmms-eval harness:
// VQA accuracy, ANLS, ChartQA relaxed accuracy, multiple-choice accuracy.
//
// Usage: node aggregate.js --title "Linux CPU" --out summary.md <log1> [log2 ...]
//   (no deps; reads [VLMROW]{json}[/VLMROW] lines, prints markdown to stdout + --out)

const fs = require('fs')

// Single source of truth for the two-models column labels (base/candidate); CLI flags
// still override. Loaded best-effort so aggregate works even outside the package.
let CONFIG = {}
try { CONFIG = require('./config.cjs') } catch (_) {}

const ARTICLES = new Set(['a', 'an', 'the'])
const PUNCT = /[;/[\]"{}()=+\\_\-><@`,?!.]/g

function norm (s) {
  s = String(s == null ? '' : s).toLowerCase().trim().replace(/[\t\n]/g, ' ').replace(PUNCT, ' ')
  return s.split(/\s+/).filter(w => w && !ARTICLES.has(w)).join(' ').trim()
}
function lev (a, b) {
  if (a === b) return 0
  if (!a) return b.length
  if (!b) return a.length
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i)
  for (let i = 1; i <= a.length; i++) {
    const cur = [i]
    for (let j = 1; j <= b.length; j++) {
      cur.push(Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] !== b[j - 1] ? 1 : 0)))
    }
    prev = cur
  }
  return prev[b.length]
}
function toF (s) {
  const f = parseFloat(String(s).replace(/,/g, '').replace(/%/g, '').trim())
  return isNaN(f) ? null : f
}
// Strip the boilerplate small models wrap answers in ("the answer is …", "the title
// is …", surrounding quotes) so a correct answer buried in a sentence still matches.
function extractAnswer (s) {
  let p = String(s == null ? '' : s).trim()
  p = p.replace(/^["'`\s]*(?:the\s+)?(?:answer|title|name|result|value)(?:\s+is)?\s*[:=]?\s*/i, '')
  p = p.replace(/^["'([]+/, '').replace(/["')\].\s]+$/, '')
  return p
}
// Word-set F1 — graded partial credit for near-misses / verbose answers that contain
// the gold words (e.g. "the title is Every Now and Then" vs "every now then").
function tokenF1 (a, b) {
  const A = a.split(/\s+/).filter(Boolean); const B = b.split(/\s+/).filter(Boolean)
  if (!A.length || !B.length) return 0
  const setB = new Set(B); let hit = 0
  for (const w of new Set(A)) if (setB.has(w)) hit++
  if (!hit) return 0
  const prec = hit / new Set(A).size; const rec = hit / new Set(B).size
  return (2 * prec * rec) / (prec + rec)
}
// Graded text similarity used by vqa + anls: exact normalized match = 1; otherwise the
// best of word-overlap F1 and (only when genuinely close) char-level similarity. Wholly
// unrelated answers stay ~0 — partial credit is earned, not floored in.
function textSim (pred, gold) {
  const p = norm(extractAnswer(pred)); const g = norm(gold)
  if (!p && !g) return 1
  if (!p || !g) return 0
  if (p === g) return 1
  const f1 = tokenF1(p, g)
  const cs = 1 - lev(p, g) / Math.max(p.length, g.length, 1)
  return Math.max(f1, cs >= 0.5 ? cs : 0)
}
const SCORERS = {
  // Accuracy with graded partial credit. An exact (normalized) match against ANY gold
  // is 1.0 — we use exact-match accuracy rather than the VQAv2 min(1, hits/3) agreement
  // formula, which is calibrated for 10-annotator sets and unfairly caps few-annotator
  // golds (a correct single-annotator GQA answer would otherwise max out at 0.33). A
  // near-miss earns its word-overlap / similarity score; an unrelated answer stays ~0.
  vqa: (pred, golds) => Math.max(0, ...golds.map(g => textSim(pred, g))),
  anls: (pred, golds) => Math.max(0, ...golds.map(g => textSim(pred, g))),
  relaxed: (pred, golds) => {
    let pf = toF(pred)
    if (pf === null) { const m = String(pred).replace(/,/g, '').match(/-?\d+(?:\.\d+)?/); if (m) pf = parseFloat(m[0]) }
    for (const g of golds) {
      const gf = toF(g)
      if (pf !== null && gf !== null) {
        if (gf === 0) { if (Math.abs(pf) < 1e-6) return 1 } else if (Math.abs(pf - gf) / Math.abs(gf) <= 0.05) return 1
      }
      if (norm(pred) === norm(g) || norm(pred).includes(norm(g))) return 1
    }
    return 0
  },
  ocr_contains: (pred, golds) => golds.some(g => String(pred).toLowerCase().includes(String(g).toLowerCase().trim())) ? 1 : 0,
  // Multiple-choice: only credit an answer the model actually states as its choice —
  // an explicit "answer/option <L>" phrase, or a short letter-led reply ("C", "C. pupa").
  // A verbose reasoning paragraph with no stated choice scores 0 rather than grabbing a
  // quasi-random interior letter (which made MC scores noise; see investigation).
  mc: (pred, golds) => {
    const gold = String(golds[0]).trim().toUpperCase()
    const p = String(pred).trim()
    const ans = p.match(/(?:final\s+answer|correct\s+answer|answer(?:\s+is)?|option)\b[^A-Za-z0-9]*([A-Ha-h])\b/i)
    if (ans) return ans[1].toUpperCase() === gold ? 1 : 0
    const lead = p.match(/^[^A-Za-z0-9]*([A-Ha-h])\b/)
    if (lead && p.replace(/[^A-Za-z]/g, '').length <= 30) return lead[1].toUpperCase() === gold ? 1 : 0
    return 0
  }
}
function score (metric, pred, golds) { return (SCORERS[metric] || (() => 0))(pred, golds) }

// ── OCR metrics ───────────────────────────────────────────────────────────
// Classical-OCR scoring per VLM_General_Benchmark.md: CER/WER are error rates
// (↓ lower better, 0 = perfect, >1 = worse-than-empty), BLEU is overlap (↑).
// These live on their OWN path and feed a SEPARATE table — they are never mixed
// into the higher-better "Overall %" quality table (mixing ↑ and ↓ is meaningless).
const OCR_METRICS = new Set(['ocr', 'cer'])
const ocrNorm = s => String(s == null ? '' : s).toLowerCase().replace(/\s+/g, ' ').trim()
const cerOne = (p, g) => g.length ? lev(p, g) / g.length : (p.length ? 1 : 0)
// lev() compares by index + !==, so it works on word arrays too (word-level edits).
const werOne = (p, g) => { const pw = p ? p.split(' ') : []; const gw = g ? g.split(' ') : []; return gw.length ? lev(pw, gw) / gw.length : (pw.length ? 1 : 0) }
function ngrams (a, n) { const o = []; for (let i = 0; i + n <= a.length; i++) o.push(a.slice(i, i + n).join('')); return o }
function bleuOne (p, g) {
  const pw = p ? p.split(' ') : []; const gw = g ? g.split(' ') : []
  if (!pw.length || !gw.length) return 0
  let logsum = 0
  for (let n = 1; n <= 4; n++) {
    const pn = ngrams(pw, n); const gc = {}
    for (const x of ngrams(gw, n)) gc[x] = (gc[x] || 0) + 1
    let hit = 0; const used = {}
    for (const x of pn) if ((gc[x] || 0) > (used[x] || 0)) { hit++; used[x] = (used[x] || 0) + 1 }
    logsum += Math.log((hit + 1) / (pn.length + 1)) // add-1 smoothing (short strings)
  }
  const bp = pw.length >= gw.length ? 1 : Math.exp(1 - gw.length / pw.length) // brevity penalty
  return bp * Math.exp(logsum / 4)
}
// Best (most charitable) over the gold alternatives: min error, max overlap.
function ocrScore (pred, golds) {
  const p = ocrNorm(pred); const gs = (golds || []).map(ocrNorm).filter(Boolean)
  if (!gs.length) return { cer: null, wer: null, bleu: null }
  return {
    cer: Math.min(...gs.map(g => cerOne(p, g))),
    wer: Math.min(...gs.map(g => werOne(p, g))),
    bleu: Math.max(...gs.map(g => bleuOne(p, g)))
  }
}

// Human-readable task names for the report (fallback to the raw id).
const TASK_LABELS = {
  textvqa: 'TextVQA — read text in natural photos',
  vizwiz: 'VizWiz — photo questions',
  gqa: 'GQA — compositional scene reasoning',
  docvqa: 'DocVQA — document understanding (ANLS)',
  ai2d: 'AI2D — science-diagram multiple choice',
  'ocr-small': 'OCR · small — read a short phrase/sentence',
  'ocr-page': 'OCR · page — full-page text recognition'
}
const taskLabel = t => TASK_LABELS[t] || t

function parseArgs (argv) {
  const out = { files: [], inputs: [], title: 'VLM Matrix', outFile: null, prov: [], mode: '', engine: '', base: CONFIG.base || 'model_1', candidate: CONFIG.candidate || 'model_2' }
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === '--title') out.title = argv[++i]
    else if (argv[i] === '--out') out.outFile = argv[++i]
    else if (argv[i] === '--mode') out.mode = argv[++i]
    else if (argv[i] === '--engine') out.engine = argv[++i]
    else if (argv[i] === '--base') out.base = argv[++i]
    else if (argv[i] === '--candidate') out.candidate = argv[++i]
    else if (argv[i] === '--provenance') out.prov.push(argv[++i])
    // --in <host-label> <file>: tag every marker in <file> with a platform
    // label (e.g. linux / s25). [VLMROW].device only carries cpu/gpu, so the
    // caller — which knows which artifact is which platform — supplies the
    // host here; otherwise S25 rows would collapse onto the Linux rows.
    else if (argv[i] === '--in') {
      const label = argv[++i]
      const file = argv[++i]
      out.inputs.push({ label, file })
    } else out.files.push(argv[i])
  }
  return out
}
// llama.cpp/mtmd print these on native stderr (captured by `2>&1 | tee`), NOT via the
// JS logger — so we attribute the timing lines that precede each [VLMROW] to that row.
const VISION_RE = /image (?:slice )?encoded in\s+(\d+(?:\.\d+)?)\s*ms/i
const ROW_RE = /\[VLMROW\](.*?)\[\/VLMROW\]/
const SEG_RE = /\[VLMSEG\](.*?)\[\/VLMSEG\]/
const META_RE = /\[VLMMETA\](.*?)\[\/VLMMETA\]/

// Per-cell vision-encode comes from [VLMSEG] segments (stderr, same stream as the
// `image slice encoded` lines) — alignment-proof. Per-row quality/TTFT/TPS come from
// the [VLMROW] markers (stdout). They're joined on the cell|device key, not position.
function parseLog (inputs) {
  const rows = []
  const vision = {} // host|cell|device -> { segMs: [perSegmentSummedMs], segTiles: [perSegmentEncodeCount] }
  const meta = {} // cell -> { main_origin, mmproj_origin, ... }
  // Android logcat captures each printed line twice (live + flushed bare buffer),
  // so the same [VLMROW] appears more than once. Dedup exact rows per host.
  const seenRows = new Set()
  for (const { label, file } of inputs) {
    const host = label || ''
    let txt = ''
    try { txt = fs.readFileSync(file, 'utf-8') } catch (_) { continue }
    let cur = null
    for (const line of txt.split(/\r?\n/)) {
      const mm = line.match(META_RE)
      if (mm) { try { const m = JSON.parse(mm[1]); meta[m.cell] = m } catch (_) {} continue }
      const sm = line.match(SEG_RE)
      if (sm) {
        try {
          const s = JSON.parse(sm[1])
          cur = `${host}|${s.cell}|${s.device}`
          if (!vision[cur]) vision[cur] = { segMs: [], segTiles: [] }
          vision[cur].segMs.push(0); vision[cur].segTiles.push(0)
        } catch (_) {}
        continue
      }
      const vm = line.match(VISION_RE)
      if (vm && cur && vision[cur] && vision[cur].segMs.length) {
        const v = vision[cur]
        v.segMs[v.segMs.length - 1] += Number(vm[1]); v.segTiles[v.segTiles.length - 1]++
        continue
      }
      const rm = line.match(ROW_RE)
      if (rm) {
        const sig = host + ' ' + rm[1]
        if (seenRows.has(sig)) continue
        seenRows.add(sig)
        try { const r = JSON.parse(rm[1]); r.__host = host; rows.push(r) } catch (_) {}
      }
    }
  }
  return { rows, vision, meta }
}
const mean = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null
const fmtPct = x => x == null ? '—' : (100 * x).toFixed(1)
const fmtNum = (x, d = 1) => x == null ? '—' : Number(x).toFixed(d)

function build (rows, vision, meta, provText, title, opts = {}) {
  const base = opts.base || 'model_1'
  const candidate = opts.candidate || 'model_2'
  // Warmup blocks (block 0) are excluded from every stat per CONTRACT.md — the first pass
  // after a model load carries the JIT/shader spike. Legacy v1 rows (no block field) stay.
  rows = rows.filter(r => r.block !== 0)
  // Drop the first segment per cell as warmup (Vulkan shader-compile / JIT spike on the
  // first encode after each model load) so the mean reflects steady-state encode cost.
  function visStats (key) {
    const v = vision[key]
    if (!v || !v.segMs.length) return { mean: null, tiles: null }
    const drop = v.segMs.length > 3 ? 1 : 0
    const ms = v.segMs.slice(drop)
    const tiles = v.segTiles.slice(drop)
    const tilesMean = mean(tiles)
    // No `image ... encoded in N ms` lines were seen for this group (e.g. on
    // Android the addon's native vision-encode log never reaches logcat) — report
    // it as unavailable rather than a misleading 0.
    if (!tilesMean) return { mean: null, tiles: null }
    return { mean: mean(ms), tiles: tilesMean }
  }
  const visMean = key => visStats(key).mean
  const visSlices = key => visStats(key).tiles
  // group key = host|cell|device (host distinguishes linux vs s25, etc.)
  const keys = []
  const byKey = {}
  for (const r of rows) {
    const k = `${r.__host || ''}|${r.cell}|${r.device}`
    if (!byKey[k]) { byKey[k] = []; keys.push(k) }
    byKey[k].push(r)
  }
  keys.sort()
  const hosts = [...new Set(rows.map(r => r.__host || ''))].sort()
  const devs = [...new Set(rows.map(r => r.device).filter(Boolean))].sort()
  const cells = [...new Set(rows.map(r => r.cell).filter(Boolean))] // models (two-models) or sources
  // Split the task universe by metric family: VQA-family tasks feed the
  // higher-better "%" tables; OCR tasks (CER/WER/BLEU) get their own table.
  const allTasks = [...new Set(rows.map(r => r.task).filter(Boolean))]
  const ocrTasks = allTasks.filter(t => rows.some(r => r.task === t && OCR_METRICS.has(r.metric)))
  const vqaTasks = allTasks.filter(t => !ocrTasks.includes(t))

  // One pass of stats per host|cell|device group, reused across all sections.
  function groupStats (key) {
    const rs = byKey[key]
    if (!rs) return null
    const okRows = rs.filter(r => !r.error)
    const perTask = vqaTasks.map(t => {
      const sc = rs.filter(r => r.task === t && !r.error).map(r => score(r.metric, r.pred, r.gold))
      return mean(sc)
    })
    return {
      total: rs.length,
      n: okRows.length,
      errs: rs.length - okRows.length,
      perTask,
      overall: mean(perTask.filter(v => v != null)),
      ve: visMean(key),
      sl: visSlices(key),
      ttft: mean(okRows.map(r => r.ttft_ms).filter(v => v != null)),
      tps: mean(okRows.map(r => r.decode_tps).filter(v => v != null)),
      wall: mean(okRows.map(r => r.ms).filter(v => v != null))
    }
  }
  // Avg OCR metrics (CER/WER ↓, BLEU ↑) for a host|cell|device group, over its OCR rows.
  function ocrGroup (key) {
    const rs = (byKey[key] || []).filter(r => OCR_METRICS.has(r.metric) && !r.error)
    if (!rs.length) return null
    const cs = []; const ws = []; const bs = []
    for (const r of rs) { const o = ocrScore(r.pred, r.gold); if (o.cer != null) { cs.push(o.cer); ws.push(o.wer); bs.push(o.bleu) } }
    return { cer: mean(cs), wer: mean(ws), bleu: mean(bs), n: rs.length }
  }
  const OCR_ROWS = [{ k: 'cer', label: 'CER ↓' }, { k: 'wer', label: 'WER ↓' }, { k: 'bleu', label: 'BLEU ↑' }]

  const L = []
  L.push(`## ${title}\n`)
  const modeLabel = opts.mode === 'several-sources'
    ? 'several sources (engine varies; model fixed)'
    : `two models (${base} vs ${candidate}; engine fixed)`
  L.push(`**Mode:** ${modeLabel}  ·  **Engine:** ${opts.engine || 'addon'}\n`)
  const severalSources = opts.mode === 'several-sources'
  L.push(severalSources
    ? '_one fixed model across inference engines · quality = lmms-eval ' +
      '(VQA / ANLS / relaxed / MC), equal-weight mean across tasks._\n'
    : `_comparing two models — base = **${base}**, candidate = **${candidate}** · quality = lmms-eval ` +
      '(VQA / ANLS / relaxed / MC), equal-weight mean across tasks._\n')

  if (!keys.length) { L.push('> ⚠️ No [VLMROW] markers found in the provided logs.\n'); return L.join('\n') }

  // ── 1 · Highlights — Quality + Speed tables ───────────────────────────────
  L.push('# 1 · Highlights\n')
  if (severalSources) {
    // Comparison axis is the engine; one column per source.
    const sources = [...new Set(rows.map(r => r.cell))].sort()
    L.push(`Inference engines on the same model: **${sources.join(', ')}**.\n`)
    L.push('### Quality — overall % per source\n')
    L.push('| Platform · device | ' + sources.join(' | ') + ' |')
    L.push('|' + '---|'.repeat(sources.length + 1))
    for (const host of hosts) {
      for (const dv of devs) {
        const vals = sources.map(s => { const g = groupStats(`${host}|${s}|${dv}`); return g ? fmtPct(g.overall) : '—' })
        if (vals.every(v => v === '—')) continue
        L.push(`| ${host || '—'} · ${dv.toUpperCase()} | ` + vals.join(' | ') + ' |')
      }
    }
    L.push('')
    L.push('### Speed — mmproj-encode ms per source (lower = faster)\n')
    L.push('| Platform · device | ' + sources.join(' | ') + ' |')
    L.push('|' + '---|'.repeat(sources.length + 1))
    for (const host of hosts) {
      for (const dv of devs) {
        const vals = sources.map(s => { const g = groupStats(`${host}|${s}|${dv}`); return g ? fmtNum(g.ve != null ? g.ve : g.ttft, g.ve != null ? 1 : 0) : '—' })
        if (vals.every(v => v === '—')) continue
        L.push(`| ${host || '—'} · ${dv.toUpperCase()} | ` + vals.join(' | ') + ' |')
      }
    }
    L.push('')
    if (ocrTasks.length) {
      L.push('### OCR — avg CER/WER/BLEU per source (CER/WER lower better · BLEU higher better)\n')
      L.push('| Platform · device | metric | ' + sources.join(' | ') + ' |')
      L.push('|' + '---|'.repeat(sources.length + 2))
      for (const host of hosts) {
        for (const dv of devs) {
          const gs = sources.map(s => ocrGroup(`${host}|${s}|${dv}`))
          if (gs.every(g => !g)) continue
          for (const { k, label } of OCR_ROWS) {
            L.push(`| ${host || '—'} · ${dv.toUpperCase()} | ${label} | ` + gs.map(g => fmtNum(g && g[k], 3)).join(' | ') + ' |')
          }
        }
      }
      L.push('')
    }
  } else {
    // Comparison axis is the model: base cell vs candidate cell, per platform · device.
    L.push(`Two models — **${base}** (base) vs **${candidate}** (candidate), per platform · device.\n`)
    L.push(`### Quality — overall %: ${base} vs ${candidate}\n`)
    L.push(`| Platform · device | ${base} % | ${candidate} % | Δ (pp, cand−base) |`)
    L.push('|---|---|---|---|')
    for (const host of hosts) {
      for (const dv of devs) {
        const b = groupStats(`${host}|${base}|${dv}`)
        const c = groupStats(`${host}|${candidate}|${dv}`)
        if (!b && !c) continue
        const dpp = (b && b.overall != null && c && c.overall != null) ? (c.overall - b.overall) * 100 : null
        L.push(`| ${host || '—'} · ${dv.toUpperCase()} | ${fmtPct(b && b.overall)} | ${fmtPct(c && c.overall)} | ${dpp == null ? '—' : (dpp >= 0 ? '+' : '') + dpp.toFixed(1)} |`)
      }
    }
    L.push('')
    L.push(`### Speed: ${base} vs ${candidate} (lower = faster; metric is mmproj-encode on desktop, TTFT on mobile)\n`)
    L.push(`| Platform · device | metric | ${base} | ${candidate} | Δ ms (cand−base, −=faster) |`)
    L.push('|---|---|---|---|---|')
    for (const host of hosts) {
      for (const dv of devs) {
        const b = groupStats(`${host}|${base}|${dv}`)
        const c = groupStats(`${host}|${candidate}|${dv}`)
        if (!b && !c) continue
        const useEnc = (b && b.ve != null) || (c && c.ve != null)
        const metric = useEnc ? 'mmproj-enc ms' : 'TTFT ms (incl. enc)'
        const bv = useEnc ? (b && b.ve) : (b && b.ttft)
        const cv = useEnc ? (c && c.ve) : (c && c.ttft)
        const d = useEnc ? 1 : 0
        // Absolute ms delta — robust near zero, unlike a "% faster" that explodes when
        // the baseline is a few ms (e.g. GPU mmproj-encode).
        const dms = (bv != null && cv != null) ? cv - bv : null
        L.push(`| ${host || '—'} · ${dv.toUpperCase()} | ${metric} | ${fmtNum(bv, d)} | ${fmtNum(cv, d)} | ${dms == null ? '—' : (dms >= 0 ? '+' : '') + dms.toFixed(d)} |`)
      }
    }
    L.push('')
    // OCR comparison (only when OCR tasks ran): avg of each metric per model + Δ.
    if (ocrTasks.length) {
      L.push(`### OCR — avg CER/WER/BLEU: ${base} vs ${candidate} (CER/WER lower better · BLEU higher better)\n`)
      L.push(`| Platform · device | metric | ${base} | ${candidate} | Δ (cand−base) |`)
      L.push('|---|---|---|---|---|')
      for (const host of hosts) {
        for (const dv of devs) {
          const b = ocrGroup(`${host}|${base}|${dv}`)
          const c = ocrGroup(`${host}|${candidate}|${dv}`)
          if (!b && !c) continue
          for (const { k, label } of OCR_ROWS) {
            const bv = b && b[k]; const cv = c && c[k]
            const delta = (bv != null && cv != null) ? (cv - bv) : null
            L.push(`| ${host || '—'} · ${dv.toUpperCase()} | ${label} | ${fmtNum(bv, 3)} | ${fmtNum(cv, 3)} | ${delta == null ? '—' : (delta >= 0 ? '+' : '') + delta.toFixed(3)} |`)
          }
        }
      }
      L.push('')
    }
  }

  // ── 2 · Details ───────────────────────────────────────────────────────────
  L.push('# 2 · Details\n')
  if (Object.keys(meta).length) {
    L.push('### Models & origins (Source = Registry / HF / S3 / URL · pinned commits)\n')
    L.push('| Cell | main model | mmproj |')
    L.push('|---|---|---|')
    for (const cell of Object.keys(meta).sort()) {
      const m = meta[cell]
      const main = `**${m.main_source || '—'}** · ${m.main_origin || '—'}`
      const proj = `**${m.mmproj_source || '—'}** · ${m.mmproj_origin || '—'}`
      L.push(`| \`${cell}\` | ${main} | ${proj} |`)
    }
    L.push('')
  }
  if (provText && provText.trim()) {
    L.push('### Provenance — hardware & software\n')
    L.push(provText.trim() + '\n')
  }
  L.push('### Quality (%)\n')
  L.push('| Config | host | ' + vqaTasks.join(' | ') + ' | **Overall %** |')
  L.push('|' + '---|'.repeat(vqaTasks.length + 3))
  for (const k of keys) {
    const [host, cell, dev] = k.split('|')
    const g = groupStats(k)
    L.push(`| \`${cell}\` · ${dev.toUpperCase()} | ${host || '—'} | ` + g.perTask.map(fmtPct).join(' | ') + ` | **${fmtPct(g.overall)}** |`)
  }
  L.push('')

  // Per-task quality, the way a reader picks a model for a task: human-readable
  // task name × each model (or each source) as columns, % mean across platforms.
  if (vqaTasks.length && cells.length) {
    const pctCellTask = (cell, t) => mean(rows.filter(r => r.cell === cell && r.task === t && !r.error)
      .map(r => score(r.metric, r.pred, r.gold)))
    const axis = severalSources ? 'source' : 'model'
    L.push(`### Quality by task (% — higher better, mean across platforms; one column per ${axis})\n`)
    L.push('| Task | ' + cells.join(' | ') + ' |')
    L.push('|' + '---|'.repeat(cells.length + 1))
    for (const t of vqaTasks) {
      L.push(`| ${taskLabel(t)} | ` + cells.map(c => fmtPct(pctCellTask(c, t))).join(' | ') + ' |')
    }
    L.push('')
  }

  // OCR tasks get their own table — CER/WER are error rates (↓), BLEU overlap (↑);
  // never blended into the % above. One row per task × config that produced OCR rows.
  if (ocrTasks.length) {
    const ocrAgg = (cell, t, host, dev) => {
      const rs = rows.filter(r => r.cell === cell && r.task === t && (r.__host || '') === host && r.device === dev && !r.error)
      const cs = []; const ws = []; const bs = []
      for (const r of rs) { const o = ocrScore(r.pred, r.gold); if (o.cer != null) { cs.push(o.cer); ws.push(o.wer); bs.push(o.bleu) } }
      return { cer: mean(cs), wer: mean(ws), bleu: mean(bs), n: rs.length }
    }
    L.push('### OCR — text recognition (CER ↓ / WER ↓ lower better · BLEU ↑ higher better)\n')
    L.push('| Task | Config | host | CER ↓ | WER ↓ | BLEU ↑ |')
    L.push('|---|---|---|---|---|---|')
    for (const t of ocrTasks) {
      for (const k of keys) {
        const [host, cell, dev] = k.split('|')
        const o = ocrAgg(cell, t, host, dev)
        if (!o.n) continue
        L.push(`| ${taskLabel(t)} | \`${cell}\` · ${dev.toUpperCase()} | ${host || '—'} | ${fmtNum(o.cer, 3)} | ${fmtNum(o.wer, 3)} | ${fmtNum(o.bleu, 3)} |`)
      }
    }
    L.push('')
  }
  L.push('### Speed\n')
  L.push('| Config | host | n | err | **mmproj enc (ms)** | tiles | TTFT (ms) | decode TPS | wall (ms) |')
  L.push('|---|---|---|---|---|---|---|---|---|')
  for (const k of keys) {
    const [host, cell, dev] = k.split('|')
    const g = groupStats(k)
    L.push(`| \`${cell}\` · ${dev.toUpperCase()} | ${host || '—'} | ${g.n} | ${g.errs} | ${fmtNum(g.ve, 1)} | ${fmtNum(g.sl, 1)} | ${fmtNum(g.ttft, 0)} | ${fmtNum(g.tps, 1)} | ${fmtNum(g.wall, 0)} |`)
  }
  L.push('')
  L.push('> **mmproj enc** is parsed from llama.cpp\'s native stderr. On mobile (Device Farm) that ' +
    'stream is not captured (Android logcat / iOS console), so it shows `—` there; TTFT on mobile ' +
    'already includes the vision-encode + prompt-eval time and is the cross-platform proxy.\n')
  // ── 3 · Test results (Device-Farm-style Metric | Count, per platform) ──────
  L.push('# 3 · Test Results (per platform)\n')
  L.push('| Platform | Metric | Count |')
  L.push('|---|---|---|')
  for (const host of hosts) {
    const hk = keys.filter(k => k.split('|')[0] === (host || ''))
    let total = 0; let passed = 0; let failed = 0
    for (const k of hk) { const g = groupStats(k); total += g.total; passed += g.n; failed += g.errs }
    L.push(`| ${host || '—'} | samples run | ${total} |`)
    L.push(`| ${host || '—'} | passed (inference ok) | ${passed} |`)
    L.push(`| ${host || '—'} | failed | ${failed} |`)
  }
  L.push('')

  // ── 4 · Image samples (per task, with resolution) ─────────────────────────
  const imgRows = []
  const seenImg = new Set()
  for (const r of rows) {
    if (!r.img) continue
    const k = `${r.task}|${r.img}`
    if (seenImg.has(k)) continue
    seenImg.add(k)
    imgRows.push({ task: r.task || '—', img: r.img, wh: (r.img_w && r.img_h) ? `${r.img_w}×${r.img_h}` : '—' })
  }
  imgRows.sort((a, b) => a.task.localeCompare(b.task) || a.img.localeCompare(b.img))
  if (imgRows.length) {
    L.push('# 4 · Image samples\n')
    L.push('| Task | Image | Resolution (W×H) |')
    L.push('|---|---|---|')
    for (const ir of imgRows) L.push(`| ${ir.task} | \`${ir.img}\` | ${ir.wh} |`)
    L.push('')
  }
  return L.join('\n')
}

// Exported for unit/scoring tests (score-check.cjs). The CLI only runs when invoked
// directly, so `require('./aggregate.js')` is side-effect-free.
module.exports = { SCORERS, score, ocrScore, OCR_METRICS, textSim, tokenF1, norm, extractAnswer, build, parseLog }

if (require.main === module) {
  const args = parseArgs(process.argv)
  // Labelled inputs (--in host file) plus any bare positional files (host '').
  const allInputs = args.inputs.concat(args.files.map(f => ({ label: '', file: f })))
  const { rows, vision, meta } = parseLog(allInputs)
  const provText = args.prov.map(p => { try { return fs.readFileSync(p, 'utf-8') } catch (_) { return '' } }).join('\n')
  const md = build(rows, vision, meta, provText, args.title, { mode: args.mode, engine: args.engine, base: args.base, candidate: args.candidate })
  process.stdout.write(md + '\n')
  if (args.outFile) fs.writeFileSync(args.outFile, md + '\n')
}
