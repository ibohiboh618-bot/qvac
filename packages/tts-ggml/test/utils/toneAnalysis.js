'use strict'

// QVAC-20557 Bug 1 (SVE/armv9 CPU Nyquist tone) — tone metrics for the on-device
// CPU A/B verify. Run under `bare` (reuses wav-helper's bare-fs reader).
//
// The bug is a constant ~12 kHz line = the Nyquist frequency of the 24 kHz output
// (SR/2). The cleanest discriminator is therefore the fraction of signal energy
// sitting exactly at Nyquist; clean speech rolls off long before 12 kHz.

const { readWavAsFloat32 } = require('./wav-helper')

// Exact Nyquist-bin (f = SR/2) energy fraction, FFT-free via Parseval on the
// alternating sum: X[N/2] = Σ s[n]·(-1)^n, energy_Nyq = (1/N)|X[N/2]|^2,
// total = Σ s[n]^2  =>  frac = (Σ s[n]·(-1)^n)^2 / (N · Σ s[n]^2).
// 1.0 for a pure ±A·(-1)^n tone, ~0 for DC / band-limited speech.
function nyquistEnergyFraction (samples) {
  if (!samples || samples.length === 0) return 0
  let alt = 0
  let energy = 0
  let sign = 1
  for (let i = 0; i < samples.length; i++) {
    const x = samples[i]
    alt += sign * x
    energy += x * x
    sign = -sign
  }
  if (energy <= 0) return 0
  return (alt * alt) / (samples.length * energy)
}

// In-place iterative radix-2 Cooley–Tukey FFT (re/im, length must be a power of 2).
function fftRadix2 (re, im) {
  const n = re.length
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      const tr = re[i]; re[i] = re[j]; re[j] = tr
      const ti = im[i]; im[i] = im[j]; im[j] = ti
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len
    const wr = Math.cos(ang)
    const wi = Math.sin(ang)
    for (let i = 0; i < n; i += len) {
      let cr = 1
      let ci = 0
      for (let k = 0; k < len / 2; k++) {
        const a = i + k
        const b = i + k + len / 2
        const vr = re[b] * cr - im[b] * ci
        const vi = re[b] * ci + im[b] * cr
        re[b] = re[a] - vr; im[b] = im[a] - vi
        re[a] = re[a] + vr; im[a] = im[a] + vi
        const ncr = cr * wr - ci * wi
        ci = cr * wi + ci * wr
        cr = ncr
      }
    }
  }
}

// Fraction of spectral energy above cutoffHz, summed over non-overlapping 8192-pt
// frames. Cross-check for the alternating-sum measure (catches a comb whose energy
// sits a little below exact Nyquist).
function highBandEnergyFraction (samples, sampleRate, cutoffHz = 10000) {
  const FRAME = 8192
  if (!samples || samples.length < FRAME) return nyquistEnergyFraction(samples)
  const binHz = sampleRate / FRAME
  const cutBin = Math.max(1, Math.floor(cutoffHz / binHz))
  let hi = 0
  let tot = 0
  for (let off = 0; off + FRAME <= samples.length; off += FRAME) {
    const re = new Float64Array(FRAME)
    const im = new Float64Array(FRAME)
    for (let i = 0; i < FRAME; i++) re[i] = samples[off + i]
    fftRadix2(re, im)
    for (let k = 1; k <= FRAME / 2; k++) {
      const p = re[k] * re[k] + im[k] * im[k]
      tot += p
      if (k >= cutBin) hi += p
    }
  }
  return tot > 0 ? hi / tot : 0
}

function analyzeWav (wavPath) {
  const { samples, sampleRate } = readWavAsFloat32(wavPath)
  return {
    wav: wavPath,
    sampleRate,
    samples: samples.length,
    nyquistEnergyFraction: nyquistEnergyFraction(samples),
    highBandEnergyFraction: highBandEnergyFraction(samples, sampleRate, 10000)
  }
}

module.exports = { nyquistEnergyFraction, highBandEnergyFraction, analyzeWav }
