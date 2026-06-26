'use strict'
// QVAC-21257: CPU-fingerprint device matching for the shipping-vs-optimized VLM
// comparison on the SHARED Device Farm pool (no device pinning available).
//
// Each run is one mmproj=both dispatch on a random Pixel 9 unit, producing a
// `mmproj-cpu` and a `mmproj-gpu` cell. The 3 optimizations all live in ggml-vulkan,
// so the CPU-projector cell's mmproj-encode runs on the ggml-CPU backend and is
// BYTE-IDENTICAL across builds — a clean per-unit performance fingerprint. We dispatch
// the shipping fabric N times and the optimized fabric M times, then pair the
// shipping↔optimized runs whose CPU fingerprint is closest and compare that pair's GPU
// cells as a same-unit-EQUIVALENT measurement.
//
// IMPORTANT: match ONLY on CPU mmproj-encode. TTFT/decode are NOT build-invariant — the
// main LLM runs on GPU (Vulkan) in every cell and the warptile change is vendor-wide.
//
// Usage:
//   node match-devices.js --shipping s1.log s2.log --optimized o1.log o2.log [--out matched.md]
//   (optional: --label <file>=<name> to name a run; default name = basename)

const fs = require('fs')
const path = require('path')
const { parseLog, score } = require('./aggregate.js')

const TASKS = ['textvqa', 'vizwiz', 'gqa', 'docvqa', 'ai2d']
const mean = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null
const fmtNum = (x, d = 1) => x == null ? '—' : Number(x).toFixed(d)
const fmtPct = x => x == null ? '—' : (100 * x).toFixed(1)
// "faster" delta: GPU vs CPU style (lower = faster), negative = slower.
const pctFaster = (ref, val) => (ref != null && val != null && ref !== 0) ? ((ref - val) / ref * 100) : null
const fmtDelta = p => p == null ? '—' : (p >= 0 ? '+' : '') + p.toFixed(1) + '%'

function parseArgs (argv) {
  const out = { shipping: [], optimized: [], outFile: null, names: {} }
  let bucket = null
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--shipping') bucket = 'shipping'
    else if (a === '--optimized') bucket = 'optimized'
    else if (a === '--out') { out.outFile = argv[++i]; bucket = null }
    else if (a === '--label') { const [f, n] = String(argv[++i]).split('='); out.names[f] = n }
    else if (bucket) out[bucket].push(a)
  }
  return out
}

// Per-run cell stats. device is always 'gpu' for both mmproj cells (mmproj-compare runs
// the model backend on GPU; only the projector backend varies), so memory keys are
// `|mmproj-cpu|gpu` / `|mmproj-gpu|gpu`.
function runStats (file, build, name) {
  const { rows, memory } = parseLog([{ label: '', file }])
  const byCell = {}
  for (const r of rows) (byCell[r.cell] = byCell[r.cell] || []).push(r)
  function cellStat (cell) {
    const rs = byCell[cell] || []
    const ok = rs.filter(r => !r.error)
    const perTask = TASKS.map(t => mean(rs.filter(r => r.task === t && !r.error).map(r => score(r.metric, r.pred, r.gold))))
    const memRec = memory[`|${cell}|gpu`] || memory[`|${cell}|cpu`] || null
    return {
      n: ok.length,
      ve: mean(ok.map(r => r.vision_ms).filter(v => v != null)),
      ttft: mean(ok.map(r => r.ttft_ms).filter(v => v != null)),
      wall: mean(ok.map(r => r.ms).filter(v => v != null)),
      tps: mean(ok.map(r => r.decode_tps).filter(v => v != null)),
      overall: mean(perTask.filter(v => v != null)),
      rssMb: (memRec && memRec.vmhwm_kb != null) ? memRec.vmhwm_kb / 1024 : null
    }
  }
  return { name: name || path.basename(file), build, file, cpu: cellStat('mmproj-cpu'), gpu: cellStat('mmproj-gpu') }
}

function build (shipFiles, optFiles, names) {
  const ship = shipFiles.map(f => runStats(f, 'shipping', names[f]))
  const opt = optFiles.map(f => runStats(f, 'optimized', names[f]))
  const L = []
  L.push('# QVAC-21257 — CPU-matched shipping-vs-optimized (Pixel 9, shared pool)\n')

  // ── Per-run summary ───────────────────────────────────────────────────────
  L.push('## Per-run summary (each run = one random Pixel 9 unit)\n')
  L.push('| Run | Build | CPU enc (ms) *fingerprint* | GPU enc (ms) | GPU TTFT (ms) | GPU/CPU enc | Quality % (cpu/gpu) | GPU peak RSS (MB) |')
  L.push('|---|---|--:|--:|--:|--:|--:|--:|')
  for (const r of [...ship, ...opt]) {
    const ratio = (r.cpu.ve != null && r.gpu.ve != null && r.cpu.ve !== 0) ? (r.gpu.ve / r.cpu.ve) : null
    L.push(`| ${r.name} | ${r.build} | ${fmtNum(r.cpu.ve, 1)} | ${fmtNum(r.gpu.ve, 1)} | ${fmtNum(r.gpu.ttft, 0)} | ${ratio == null ? '—' : ratio.toFixed(3) + '×'} | ${fmtPct(r.cpu.overall)} / ${fmtPct(r.gpu.overall)} | ${fmtNum(r.gpu.rssMb, 0)} |`)
  }
  L.push('')

  // ── Pairings by CPU-fingerprint distance ──────────────────────────────────
  const pairs = []
  for (const s of ship) for (const o of opt) {
    if (s.cpu.ve == null || o.cpu.ve == null) continue
    const base = (s.cpu.ve + o.cpu.ve) / 2
    pairs.push({ s, o, dAbs: Math.abs(s.cpu.ve - o.cpu.ve), dPct: base ? Math.abs(s.cpu.ve - o.cpu.ve) / base * 100 : null })
  }
  pairs.sort((a, b) => a.dAbs - b.dAbs)
  L.push('## CPU-fingerprint pairings (closer = better same-unit proxy)\n')
  L.push('| Shipping run | Optimized run | CPU enc ship (ms) | CPU enc opt (ms) | Δ CPU % |')
  L.push('|---|---|--:|--:|--:|')
  for (const p of pairs) {
    L.push(`| ${p.s.name} | ${p.o.name} | ${fmtNum(p.s.cpu.ve, 1)} | ${fmtNum(p.o.cpu.ve, 1)} | ${fmtNum(p.dPct, 1)}% |`)
  }
  L.push('')
  if (!pairs.length) { L.push('> ⚠️ No pairable runs (missing CPU mmproj-encode).\n'); return L.join('\n') }

  // ── Best-matched same-unit-equivalent comparison ──────────────────────────
  const best = pairs[0]
  const s = best.s; const o = best.o
  const closeNote = best.dPct != null && best.dPct <= 5
    ? `**${fmtNum(best.dPct, 1)}%** — a close match; the comparison below is a sound same-unit proxy.`
    : `**${fmtNum(best.dPct, 1)}%** — ⚠️ not tight (>5%); dispatch 1–2 more runs per build and re-match before trusting absolutes.`
  L.push('## Best-matched comparison (same-unit-equivalent)\n')
  L.push(`Matched **${s.name}** (shipping) ↔ **${o.name}** (optimized) on CPU fingerprint; CPU-match closeness ${closeNote}\n`)
  L.push('| Metric | Shipping-CPU | Shipping-GPU | Optimized-CPU | Optimized-GPU | GPU: opt vs ship |')
  L.push('|---|--:|--:|--:|--:|--:|')
  const row = (label, sc, sg, oc, og, faster) => L.push(`| ${label} | ${sc} | ${sg} | ${oc} | ${og} | ${faster} |`)
  row('mmproj-encode (ms)', fmtNum(s.cpu.ve, 1), fmtNum(s.gpu.ve, 1), fmtNum(o.cpu.ve, 1), fmtNum(o.gpu.ve, 1), fmtDelta(pctFaster(s.gpu.ve, o.gpu.ve)))
  row('TTFT (ms)', fmtNum(s.cpu.ttft, 0), fmtNum(s.gpu.ttft, 0), fmtNum(o.cpu.ttft, 0), fmtNum(o.gpu.ttft, 0), fmtDelta(pctFaster(s.gpu.ttft, o.gpu.ttft)))
  row('wall (ms)', fmtNum(s.cpu.wall, 0), fmtNum(s.gpu.wall, 0), fmtNum(o.cpu.wall, 0), fmtNum(o.gpu.wall, 0), fmtDelta(pctFaster(s.gpu.wall, o.gpu.wall)))
  row('decode (tok/s)', fmtNum(s.cpu.tps, 1), fmtNum(s.gpu.tps, 1), fmtNum(o.cpu.tps, 1), fmtNum(o.gpu.tps, 1), '—')
  row('quality (%)', fmtPct(s.cpu.overall), fmtPct(s.gpu.overall), fmtPct(o.cpu.overall), fmtPct(o.gpu.overall), '—')
  row('peak RSS (MB)', '—', fmtNum(s.gpu.rssMb, 0), '—', fmtNum(o.gpu.rssMb, 0), '—')
  L.push('')

  // Within-run ratios (fully device-normalized — the most robust metric).
  const sRatio = (s.gpu.ve != null && s.cpu.ve) ? s.gpu.ve / s.cpu.ve : null
  const oRatio = (o.gpu.ve != null && o.cpu.ve) ? o.gpu.ve / o.cpu.ve : null
  L.push('### Within-run GPU/CPU encode ratio (device-normalized — primary robust metric)\n')
  L.push('| Build | GPU/CPU mmproj-encode ratio |')
  L.push('|---|--:|')
  L.push(`| Shipping | ${sRatio == null ? '—' : sRatio.toFixed(3) + '×'} |`)
  L.push(`| Optimized | ${oRatio == null ? '—' : oRatio.toFixed(3) + '×'} |`)
  L.push('')

  // Cross comparison the doc caveat was about — now on CPU-matched units.
  L.push('### Cross-comparison — shipping-CPU vs optimized-GPU (CPU-matched units)\n')
  L.push('| Metric | Shipping-CPU | Optimized-GPU | Opt-GPU vs Ship-CPU |')
  L.push('|---|--:|--:|--:|')
  L.push(`| mmproj-encode (ms) | ${fmtNum(s.cpu.ve, 1)} | ${fmtNum(o.gpu.ve, 1)} | ${fmtDelta(pctFaster(s.cpu.ve, o.gpu.ve))} |`)
  L.push(`| TTFT (ms) | ${fmtNum(s.cpu.ttft, 0)} | ${fmtNum(o.gpu.ttft, 0)} | ${fmtDelta(pctFaster(s.cpu.ttft, o.gpu.ttft))} |`)
  L.push('')
  L.push('> Method: shared-pool runs matched by build-invariant CPU mmproj-encode (ggml-CPU backend, ' +
    'identical across builds). A close CPU match controls the dominant device-variance axis but does not ' +
    'prove identical GPU baselines; the within-run GPU/CPU ratio above is the most robust figure. Decode/TTFT ' +
    'are not build-invariant (the LLM runs on the optimized Vulkan backend too).\n')
  return L.join('\n')
}

if (require.main === module) {
  const args = parseArgs(process.argv)
  if (!args.shipping.length || !args.optimized.length) {
    process.stderr.write('usage: node match-devices.js --shipping s1.log [s2.log ...] --optimized o1.log [o2.log ...] [--out matched.md]\n')
    process.exit(2)
  }
  const md = build(args.shipping, args.optimized, args.names)
  process.stdout.write(md + '\n')
  if (args.outFile) fs.writeFileSync(args.outFile, md + '\n')
}

module.exports = { runStats, build }
