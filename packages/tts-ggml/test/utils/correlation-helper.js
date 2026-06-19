'use strict'

// GPU-vs-CPU output correlation gate (QVAC-20557).
//
// Reuses the Pearson metric from tts-cpp/scripts/validate-precision-parity.sh
// (corr = corrcoef(gpu, cpu) on min-length-aligned PCM): a correct GPU backend
// matches CPU at corr ~0.999+, while the ARM Mali Valhall mul_mat miscompute
// produces garbage/NaN/silence at corr ~0.003 (or undefined). This is the
// correctness signal the shipped gpu-smoke test lacked — it only proved the GPU
// was *engaged* (backendDevice/backendId), never that its output matched CPU.
//
// Always logs the number so every model×device correlation is visible in
// bare_console.log on the device farm (which does NOT export the WAV).
//
// NOTE on determinism: deterministic engines (Supertonic) can be HARD-gated.
// Chatterbox's T3 is autoregressive + stochastic (temp>0, sampled per token),
// so identical seeds do NOT yield identical tokens across backends — its
// end-to-end corr is informational only (soft); its real per-stage correctness
// signal is the native `[gpu-diag]` trace in logcat_full.txt.

function toFloat (samples) {
  const n = samples.length
  const out = new Float64Array(n)
  // int16 PCM → [-1, 1]. Correlation is scale-invariant so the exact divisor
  // does not matter, but normalising keeps the intermediate sums well-scaled.
  for (let i = 0; i < n; i++) out[i] = samples[i] / 32768
  return out
}

// Pearson correlation over the min-length prefix of two PCM arrays. Returns
// corr=NaN with a reason when one side is constant (silence / NaN-collapsed
// output) — that is itself a hard failure, never a pass.
function pearson (a, b) {
  const n = Math.min(a ? a.length : 0, b ? b.length : 0)
  if (n === 0) return { corr: NaN, n: 0, reason: 'empty' }
  const fa = toFloat(a)
  const fb = toFloat(b)
  let ma = 0
  let mb = 0
  for (let i = 0; i < n; i++) { ma += fa[i]; mb += fb[i] }
  ma /= n
  mb /= n
  let cov = 0
  let va = 0
  let vb = 0
  for (let i = 0; i < n; i++) {
    const da = fa[i] - ma
    const db = fb[i] - mb
    cov += da * db
    va += da * da
    vb += db * db
  }
  if (va === 0 || vb === 0) {
    return { corr: NaN, n, reason: va === 0 ? 'gpu-constant(silent/NaN)' : 'cpu-constant(silent/NaN)' }
  }
  return { corr: cov / Math.sqrt(va * vb), n, reason: null }
}

// Length + sample-count sanity around pearson(). minLenRatio guards against a
// mid-synthesis crash (one side much shorter); minSamples mirrors the existing
// gpu-smoke floor.
function compareSamples (gpu, cpu, opts = {}) {
  const minSamples = opts.minSamples || 1000
  const minLenRatio = opts.minLenRatio || 0.90
  const lg = gpu ? gpu.length : 0
  const lc = cpu ? cpu.length : 0
  if (lg < minSamples || lc < minSamples) {
    return { ok: false, corr: NaN, n: Math.min(lg, lc), lenRatio: 0, reason: `too-few-samples (gpu=${lg} cpu=${lc} min=${minSamples})` }
  }
  const lenRatio = Math.min(lg, lc) / Math.max(lg, lc)
  const { corr, n, reason } = pearson(gpu, cpu)
  if (reason) return { ok: false, corr, n, lenRatio, reason }
  const lenOk = lenRatio >= minLenRatio
  return { ok: lenOk, corr, n, lenRatio, reason: lenOk ? null : `len-ratio ${lenRatio.toFixed(3)} < ${minLenRatio}` }
}

// Assert GPU output correlates with CPU.
//   opts.threshold : minimum acceptable corr (default 0.99)
//   opts.soft      : informational mode (Chatterbox/stochastic) — still HARD-fails
//                    on undefined corr (silent/NaN GPU), but only WARNS on a low
//                    magnitude (expected when T3 sampled different tokens).
// Always logs the number. Returns the compareSamples result.
function assertSampleCorrelation (t, engineTag, gpu, cpu, opts = {}) {
  const threshold = opts.threshold !== undefined ? opts.threshold : 0.99
  const soft = opts.soft === true
  const res = compareSamples(gpu, cpu, opts)
  const corrStr = Number.isFinite(res.corr) ? res.corr.toFixed(6) : String(res.corr)
  console.log(
    `[${engineTag}/corr] gpu_n=${gpu ? gpu.length : 0} cpu_n=${cpu ? cpu.length : 0} ` +
    `aligned_n=${res.n} lenRatio=${(res.lenRatio || 0).toFixed(3)} corr=${corrStr} ` +
    `threshold=${threshold} mode=${soft ? 'soft(informational)' : 'hard'}` +
    `${res.reason ? ' reason=' + res.reason : ''}`
  )

  // Undefined correlation (silent / NaN / collapsed GPU output) is a hard
  // failure in BOTH modes — it is the Valhall garbage signature, token-independent.
  if (!Number.isFinite(res.corr)) {
    t.fail(`${engineTag}: GPU-vs-CPU correlation undefined (${res.reason}) — GPU output is silent/NaN/garbage`)
    return res
  }

  if (soft) {
    t.pass(`${engineTag}: GPU output finite; corr=${corrStr} (informational — T3 stochastic, see [gpu-diag] per-stage trace)`)
    if (res.corr < threshold) {
      t.comment(`${engineTag}: corr ${corrStr} < ${threshold} — expected for stochastic T3; verify via [gpu-diag] per-stage trace, not this number`)
    }
    return res
  }

  t.ok(res.corr >= threshold && res.ok,
    `${engineTag}: GPU-vs-CPU correlation ${corrStr} must be >= ${threshold} (length/quality ok=${res.ok}${res.reason ? ', ' + res.reason : ''})`)
  return res
}

module.exports = { pearson, compareSamples, assertSampleCorrelation }
