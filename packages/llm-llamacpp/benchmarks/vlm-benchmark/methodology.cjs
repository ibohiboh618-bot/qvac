'use strict'
// QVAC-19371 (A1 contract): MEASUREMENT METHODOLOGY — how rounds are scheduled
// so deltas are trustworthy (config.cjs `methodology` block is the knobs).
//
// Target behaviour (per docs/perf/metal-baseline.md findings):
//   • 1 warmup block + N measured blocks per source; the report takes the
//     MEDIAN per metric (a "block" = one full pass over the fixture for one
//     source × model × backend; marker field `block`: 0 = warmup, 1.. = measured)
//   • blocks INTERLEAVED across sources (B,C,B,C…) so neither build
//     systematically runs on a hotter machine (per-inference interleaving is
//     impossible — two addon builds can't share one process)
//   • STABILITY GUARD between blocks: real temperature sensor where available
//     (self-hosted Mac mini, pending sudo/powermetrics confirmation), else a
//     fixed micro-workload whose timing must stabilise (sensor-free proxy)
//
// OWNERSHIP: runner workstream (Dev A). TODO(A3) implements the scheduler in
// run-desktop.cjs using these helpers; until then the harness emits block = rep+1
// (no warmup blocks) and the report side must already handle any block layout.

// Block plan for one (model × backend): the order sources run their blocks in.
// interleave=true → [s1.warmup, s2.warmup, s1.b1, s2.b1, s1.b2, s2.b2, …]
function planBlocks (sourceIds, m) {
  const warmup = m && m.warmupBlocks != null ? m.warmupBlocks : 1
  const measured = m && m.measuredBlocks != null ? m.measuredBlocks : 3
  const interleave = !m || m.interleave !== false
  const plan = []
  const rounds = []
  for (let b = -warmup + 1; b <= measured; b++) rounds.push(Math.max(0, b)) // 0,1..N (0 repeated warmup times)
  if (interleave) {
    for (const block of rounds) for (const id of sourceIds) plan.push({ source: id, block })
  } else {
    for (const id of sourceIds) for (const block of rounds) plan.push({ source: id, block })
  }
  return plan
}

// Median — the reported statistic (robust against one-off hiccups).
function median (xs) {
  const a = xs.filter(v => v != null).slice().sort((x, y) => x - y)
  if (!a.length) return null
  const mid = a.length >> 1
  return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2
}

// TODO(A3): stabilityGuard(kind) — powermetrics polling on macmini ('temp'),
// fixed micro-probe timing elsewhere ('probe'), no-op on Device Farm.
async function stabilityGuard () { /* TODO(A3) */ }

module.exports = { planBlocks, median, stabilityGuard }
