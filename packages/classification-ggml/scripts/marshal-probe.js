'use strict'

// =============================================================================
// === TEMPORARY DIAGNOSTIC SCAFFOLDING -- DO NOT KEEP IN MAIN ===============
//
// Standalone driver for the marshalling-probe suite defined in
// `addon/src/js-interface/marshal_probe.hpp`. See that file's header for the
// full removal checklist (this script is one of the 5 things to delete).
//
// Each probe is a controlled experiment that exercises ONE distinct candidate
// trigger of the win32-x64 first-`js_create_double`-returns-0 bug. The driver
// runs every probe, compares the JS-observed value to the C++-side trace
// (printed on stderr by the C++ probe), and produces a structured pass/fail
// summary at the end.
//
// Designed to be run on a Win32 CI runner (where the bug reproduces) AND on
// developer Win32 boxes (where it does not). The pattern of which probes
// pass/fail across the two environments tells us which axis of the
// hypothesis space is responsible.
//
// USAGE
//   bare scripts/marshal-probe.js
//
// EXIT CODE
//   Always 0. The probes are diagnostic; they should never gate a build.
// =============================================================================

const binding = require('../binding')
const process = require('bare-process')

const SEP = '─'.repeat(72)

function approxEq (a, b) {
  if (typeof a !== 'number' || typeof b !== 'number') return false
  if (Number.isNaN(a) && Number.isNaN(b)) return true
  if (!Number.isFinite(a) || !Number.isFinite(b)) return a === b
  return Math.abs(a - b) < 1e-9
}

const summary = []

function check (name, observed, expected, notes) {
  let ok = true
  let detail = ''
  if (Array.isArray(expected)) {
    for (let i = 0; i < expected.length; i++) {
      const e = expected[i]
      const o = Array.isArray(observed) ? observed[i] : undefined
      const m = approxEq(o, e)
      if (!m) {
        ok = false
        detail += `\n    [${i}] expected=${String(e)} observed=${String(o)}`
      }
    }
  } else {
    ok = approxEq(observed, expected)
    if (!ok) detail = `\n    expected=${expected} observed=${observed}`
  }
  summary.push({ name, ok, detail, notes })
  console.log(`  result: ${ok ? 'OK    ' : 'FAIL'} ${notes ? '(' + notes + ')' : ''}${detail}`)
}

console.log(SEP)
console.log(' QVAC marshalling probe suite (diagnostic)')
console.log(' Platform :', process.platform, process.arch)
console.log(' Node/Bare:', typeof process.versions === 'object' ? JSON.stringify(process.versions) : 'n/a')
console.log(SEP)

// 1. probeSyncDoubles -- baseline.
console.log('\n[1] probeSyncDoubles -- baseline 10 sequential doubles')
{
  const expected = [0.708, 0.224, 0.068, 0.500, 0.999, 0.111, 0.314, 0.272, 0.866, 0.577]
  const got = binding.probeSyncDoubles()
  check('1.syncDoubles', got, expected, 'tests `js_create_double` in the simplest possible context')
}

// 2. probeSyncBitPatterns -- value-dependent corruption?
console.log('\n[2] probeSyncBitPatterns -- special doubles (NaN, ±Inf, denormals, ...)')
{
  const expected = [
    0.0, -0.0, 0.5, 1.0, 0.708, Math.PI, Math.E,
    1e-308, 4.9406564584124654e-324, 1.7976931348623157e308,
    NaN, Infinity, -Infinity
  ]
  const got = binding.probeSyncBitPatterns()
  check('2.syncBitPatterns', got, expected, 'tests value-dependent corruption (FZ/DAZ flags, NaN handling, denormals)')
}

// 3. probeSyncMixedFirst -- is the issue specific to `js_create_double`?
console.log('\n[3] probeSyncMixedFirst -- int32+int64 BEFORE doubles')
{
  const expected = [0.708, 0.224, 0.068]
  const got = binding.probeSyncMixedFirst()
  check('3.syncMixedFirst', got, expected, 'if doubles[0] still corrupt, bug is double-specific not "first call"')
}

// 4. probeSyncStringFirst -- mirror Whisper's object-then-string-then-number sequence
console.log('\n[4] probeSyncStringFirst -- object+string BEFORE first double (Whisper pattern)')
{
  const got = binding.probeSyncStringFirst()
  // Returns an object: { label: "food", confidence: 0.708 }
  const ok = got && got.label === 'food' && approxEq(got.confidence, 0.708)
  summary.push({ name: '4.syncStringFirst', ok, detail: ok ? '' : '\n    got=' + JSON.stringify(got) })
  console.log(`  result: ${ok ? 'OK   ' : 'FAIL'} (mirrors whisper handler structure)${ok ? '' : '\n    got=' + JSON.stringify(got)}`)
}

// 5. probeSyncRepeated -- is corruption per-invocation or once-only?
console.log('\n[5] probeSyncRepeated -- 5 sequential invocations of doubles probe')
{
  const expected = [0.708, 0.224, 0.068]
  const got = binding.probeSyncRepeated()
  // got is array of arrays
  let ok = Array.isArray(got) && got.length === 5
  let detail = ''
  if (ok) {
    for (let r = 0; r < got.length; r++) {
      for (let i = 0; i < expected.length; i++) {
        if (!approxEq(got[r][i], expected[i])) {
          ok = false
          detail += `\n    invocation ${r} index ${i}: expected=${expected[i]} got=${got[r][i]}`
        }
      }
    }
  }
  summary.push({ name: '5.syncRepeated', ok, detail })
  console.log(`  result: ${ok ? 'OK   ' : 'FAIL'} (tests "first invocation only" vs persistent)${detail}`)
}

// 6. probeSyncNestedScopes -- does scope nesting alone trigger it?
console.log('\n[6] probeSyncNestedScopes -- doubles inside an inner js_handle_scope')
{
  // The probe returns an empty array (we don't escape values across scope
  // boundaries; only the stderr trace matters here).
  const got = binding.probeSyncNestedScopes()
  console.log('  (note) values created in inner scope intentionally not returned;')
  console.log('  inspect stderr trace for create_rc / cpp_readback per element')
  summary.push({ name: '6.syncNestedScopes', ok: true, detail: '(see stderr only)' })
  console.log(`  result: SEEN  (Array.isArray(got)=${Array.isArray(got)})`)
}

// 7. probeSyncStorageElement -- does js_set_element / js_get_element corrupt?
console.log('\n[7] probeSyncStorageElement -- create + set_element + get_element + readback')
{
  const expected = [0.708, 0.224, 0.068]
  const got = binding.probeSyncStorageElement()
  check('7.syncStorageElement', got, expected, 'isolates element-write/read corruption from create-time corruption')
}

// 8. probeSyncStorageProperty -- does js_set_named_property corrupt?
console.log('\n[8] probeSyncStorageProperty -- create + set_named_property + get_named_property')
{
  const got = binding.probeSyncStorageProperty()
  const ok = got && approxEq(got.p0, 0.708) && approxEq(got.p1, 0.224) && approxEq(got.p2, 0.068)
  summary.push({ name: '8.syncStorageProperty', ok, detail: ok ? '' : '\n    got=' + JSON.stringify(got) })
  console.log(`  result: ${ok ? 'OK   ' : 'FAIL'} (isolates named-property write/read corruption)${ok ? '' : '\n    got=' + JSON.stringify(got)}`)
}

// 9. probeSyncSequenceMimic -- exact mirror of our handler structure
console.log('\n[9] probeSyncSequenceMimic -- exact JsClassifyOutputHandler call sequence (no burn-one)')
{
  const got = binding.probeSyncSequenceMimic()
  const ok = Array.isArray(got) && got.length === 3 &&
    got[0].label === 'food' && approxEq(got[0].confidence, 0.708) &&
    got[1].label === 'other' && approxEq(got[1].confidence, 0.224) &&
    got[2].label === 'report' && approxEq(got[2].confidence, 0.068)
  summary.push({ name: '9.syncSequenceMimic', ok, detail: ok ? '' : '\n    got=' + JSON.stringify(got) })
  console.log(`  result: ${ok ? 'OK   ' : 'FAIL'} (THE direct sync repro of the failing path)${ok ? '' : '\n    got=' + JSON.stringify(got)}`)
}

// 10. probeSyncFpState -- detects MXCSR / x87 mutation around js_create_double
console.log('\n[10] probeSyncFpState -- MXCSR + x87 control word probe')
{
  const got = binding.probeSyncFpState()
  const ok = approxEq(got, 0.708)
  summary.push({ name: '10.syncFpState', ok, detail: ok ? '' : '\n    got=' + got })
  console.log(`  result: ${ok ? 'OK   ' : 'FAIL'} (see stderr for mxcsr/x87 trace)${ok ? '' : '\n    got=' + got}`)
}

// 11. probeAsyncCallback -- closest mirror of OutputCallBackJs::jsOutputCallback
console.log('\n[11] probeAsyncCallback -- doubles inside uv_async callback + nested scopes')
;(async () => {
  try {
    const expected = [0.708, 0.224, 0.068, 0.500, 0.999]
    const got = await binding.probeAsyncCallback()
    check('11.asyncCallback', got, expected, 'CLOSEST repro of the failing OutputCallBackJs context')
  } catch (e) {
    summary.push({ name: '11.asyncCallback', ok: false, detail: '\n    threw: ' + e.message })
    console.log(`  result: FAIL (threw: ${e.message})`)
  }

  // Final summary
  console.log('\n' + SEP)
  console.log(' SUMMARY')
  console.log(SEP)
  let pass = 0
  let fail = 0
  for (const r of summary) {
    const tag = r.ok ? 'OK   ' : 'FAIL '
    console.log(` ${tag} ${r.name}`)
    if (!r.ok && r.detail) console.log('       ' + r.detail.trim())
    if (r.ok) pass++; else fail++
  }
  console.log(SEP)
  console.log(` ${pass} passed, ${fail} failed of ${summary.length}`)
  console.log(SEP)
  console.log('\nDIAGNOSTIC INTERPRETATION:')
  console.log('  - If [9] syncSequenceMimic FAILS: bug reproduces in the synchronous')
  console.log('    handler structure with no async / OutputCallBackJs involvement.')
  console.log('    Burn-one is masking a bug in the synchronous JS-API path.')
  console.log('  - If [9] passes but [11] asyncCallback FAILS: bug needs the async')
  console.log('    libuv-callback context. Investigation should focus on')
  console.log('    OutputCallBackJs::jsOutputCallback handle-scope setup.')
  console.log('  - If only [1..3] FAIL: bug is in `js_create_double` itself.')
  console.log('  - If [2] FAILS for specific values: value-dependent corruption.')
  console.log('  - If [5] shows corruption in 1st invocation only: stateful init bug.')
  console.log('  - If FP state diff in [10] != 0: a JS-API call mutates MXCSR/x87.')
  console.log('  - If everything passes: bug NOT reproduced on this host.')
  console.log(SEP)
})()
