'use strict'

// QVAC-20557 Bug 1 — host-side tone gate for the on-device CPU A/B WAVs.
// Usage (host, after pulling WAVs):  bare test/utils/analyze-tone.js <wav> [<wav> ...]
// PASS gate: a CLEAN WAV has nyquistEnergyFraction < 0.1 (threshold). The decisive
// signal is the SEPARATION: the post-fix SVE run and the NEON control are CLEAN,
// the pre-fix (TTS_SVE_DOT_UNFIXED=1) run is TONE.

const proc = require('bare-process')
const { analyzeWav } = require('./toneAnalysis')

const THRESHOLD = 0.1
const args = proc.argv.slice(2)
if (args.length === 0) {
  console.error('usage: bare test/utils/analyze-tone.js <wav> [<wav> ...]')
  proc.exit(2)
}

for (const w of args) {
  try {
    const r = analyzeWav(w)
    const verdict = r.nyquistEnergyFraction < THRESHOLD ? 'CLEAN' : 'TONE '
    console.log(`${verdict}  nyqFrac=${r.nyquistEnergyFraction.toExponential(3)}  highBand(>10k)=${r.highBandEnergyFraction.toExponential(3)}  sr=${r.sampleRate}  n=${r.samples}  ${r.wav}`)
  } catch (e) {
    console.log(`ERROR   ${w}: ${e && e.message ? e.message : e}`)
  }
}
