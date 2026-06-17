'use strict'

// Single source of truth for the sweep axes + input modes, shared by the Bare
// sweep (config + case-runner) and the Node renderer (coverage denominator).
// Plain literals only — no bare-fs/fs imports — so it loads in both runtimes.
const PARAMETER_SWEEP = {
  quantization: ['Q4_0', 'Q4_K_M', 'Q8_0', 'F16'],
  device: ['cpu', 'gpu'],
  batchSize: [256, 512, 1024, 2048],
  flashAttn: ['off', 'on']
}

// Each swept config is run once per input mode (one sentence vs the full array).
const INPUT_MODES = ['single', 'array']

module.exports = { PARAMETER_SWEEP, INPUT_MODES }
