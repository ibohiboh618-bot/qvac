// =============================================================================
// === TEMPORARY DIAGNOSTIC SCAFFOLDING -- DO NOT KEEP IN MAIN ===============
//
// Standalone marshalling-probe suite for the win32-x64 first-`js_create_double`-
// returns-0 bug. Written to be removable in one step (5 files).
//
// REMOVAL CHECKLIST (single PR):
//   1. Delete this file: `addon/src/js-interface/marshal_probe.hpp`
//   2. In `addon/src/js-interface/binding.cpp`, remove:
//      - The `[TEMP]` `#include "marshal_probe.hpp"` line
//      - The `[TEMP]` `V("probe*", ...)` block of registrations
//   3. Delete `scripts/marshal-probe.js`
//   4. In `.github/workflows/integration-test-classification-ggml.yml`,
//      remove the "Run marshalling probe" step
//   5. Search the repo for `marshal_probe` / `marshal-probe` to make sure
//      nothing else references it
//
// NOTHING ELSE IN THE ADDON DEPENDS ON THIS FILE. The probes use only the
// bare-runtime C-API (`js.h`) and standard C++ headers; they have zero
// runtime cost when not invoked, and are not registered anywhere outside
// the `[TEMP]` block in `binding.cpp`.
//
// PURPOSE
// -------
// CI on `windows-2022` reproduces a bug where the first `js_create_double`
// call inside `JsClassifyOutputHandler`'s lambda returns 0.0 regardless of
// input. Local Win32 development boxes do NOT reproduce it (run
// `bare scripts/marshal-probe.js` locally to confirm). We need to disambiguate
// which factor triggers it: the value, the type, the call sequence, the
// handle scope nesting, the assignment path, FP state, or the async-callback
// dispatch context.
//
// Each probe below is one controlled experiment. Together they cover every
// axis we have a hypothesis for. All are synchronous except #11 (asyncCallback)
// which mirrors the `OutputCallBackJs::jsOutputCallback` context with nested
// handle scopes inside a libuv async callback.
//
// USAGE
// -----
//   bare scripts/marshal-probe.js
//
// On CI, the integration-test workflow runs the probe before the integration
// tests on win32-x64 only. The output is purely diagnostic -- it does NOT
// affect the build's pass/fail outcome.
// =============================================================================

#pragma once

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <js.h>
#include <uv.h>

// MXCSR / x87 control-word probes are Win32 + x86/x64 specific. The probe
// matters only for the win32-x64 first-`js_create_double` bug; on every
// other platform (linux/darwin/ios/android, x86_64 or arm64) we return
// zero stubs. Gating strictly on `_WIN32` (and not `__clang__`, which is
// also defined on linux/darwin clang where `<intrin.h>` and
// `_control87` do NOT exist) keeps the prebuild matrix green on every
// non-Windows runner.
#if defined(_WIN32)
#include <float.h>
#include <intrin.h>
#endif

namespace qvac_lib_infer_ggml_classification::probe {

namespace detail {

// Read MXCSR (SSE FP state) and x87 control word. Stubs return 0 off-Win32.
inline unsigned int read_mxcsr() {
#if defined(_WIN32) && (defined(_M_X64) || defined(_M_IX86))
  return _mm_getcsr();
#else
  return 0;
#endif
}

inline unsigned int read_x87() {
#if defined(_WIN32)
  unsigned int cw = 0;
  _control87(0, 0); // returns current word; we discard, just want a probe
  return cw;
#else
  return 0;
#endif
}

// Build a JS array of doubles from a C-side vector. Each element comes from
// a fresh `js_create_double`; we do NOT cache or reuse `js_value_t*`.
inline js_value_t* build_array(
    js_env_t* env, const std::vector<double>& values) {
  js_value_t* array = nullptr;
  js_create_array_with_length(env, values.size(), &array);
  for (size_t i = 0; i < values.size(); ++i) {
    js_value_t* v = nullptr;
    js_create_double(env, values[i], &v);
    if (v != nullptr) {
      js_set_element(env, array, static_cast<uint32_t>(i), v);
    }
  }
  return array;
}

// Compact stderr trace helper: prints index + input + create_rc + ptr +
// readback for one js_create_double call, after also doing js_get_value_double
// on the result.
inline void trace_one(
    const char* tag, size_t i, double input, int create_rc, js_value_t* v,
    js_env_t* env) {
  double readback = -1.0;
  int read_rc = -999;
  if (v != nullptr) {
    read_rc = js_get_value_double(env, v, &readback);
  }
  std::fprintf(
      stderr,
      "  [%s][%zu] input=%.9f create_rc=%d v_ptr=%p readback_rc=%d "
      "cpp_readback=%.9f match=%s\n",
      tag, i, input, create_rc, static_cast<void*>(v), read_rc, readback,
      (read_rc == 0 && readback == input) ? "yes" : "NO");
}

} // namespace detail

// =============================================================================
// 1. probeSyncDoubles
//   Baseline: 10 sequential `js_create_double` calls with deterministic
//   non-zero values, with immediate readback via `js_get_value_double`. Builds
//   a JS array. No other JS-API calls before the first `js_create_double`.
// =============================================================================
inline js_value_t* syncDoubles(js_env_t* env, js_callback_info_t* /*info*/) {
  static const std::vector<double> kValues = {
      0.708, 0.224, 0.068, 0.500, 0.999, 0.111, 0.314, 0.272, 0.866, 0.577};

  std::fprintf(stderr, "=== probeSyncDoubles (N=%zu) ===\n", kValues.size());
  std::fprintf(
      stderr, "[mxcsr=0x%08x x87=0x%08x]\n", detail::read_mxcsr(),
      detail::read_x87());

  js_value_t* array = nullptr;
  js_create_array_with_length(env, kValues.size(), &array);
  for (size_t i = 0; i < kValues.size(); ++i) {
    js_value_t* v = nullptr;
    int rc = js_create_double(env, kValues[i], &v);
    detail::trace_one("doubles", i, kValues[i], rc, v, env);
    if (v != nullptr) js_set_element(env, array, static_cast<uint32_t>(i), v);
  }
  std::fflush(stderr);
  return array;
}

// =============================================================================
// 2. probeSyncBitPatterns
//   Special double values: zero variants, π, NaN, ±Inf, denormal, max/min
//   normal. Catches any value-dependent corruption (e.g. denormal flushed to
//   zero by FZ flag, NaN payload-stripping, etc.).
// =============================================================================
inline js_value_t* syncBitPatterns(
    js_env_t* env, js_callback_info_t* /*info*/) {
  std::vector<double> kValues = {
      0.0,
      -0.0,
      0.5,
      1.0,
      0.708,
      3.141592653589793,
      2.718281828459045,
      1e-308,                              // smallest normal
      4.9406564584124654e-324,             // smallest denormal
      1.7976931348623157e308,              // largest normal
      std::nan(""),                        // quiet NaN
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity()};

  std::fprintf(
      stderr, "=== probeSyncBitPatterns (N=%zu) ===\n", kValues.size());

  js_value_t* array = nullptr;
  js_create_array_with_length(env, kValues.size(), &array);
  for (size_t i = 0; i < kValues.size(); ++i) {
    js_value_t* v = nullptr;
    int rc = js_create_double(env, kValues[i], &v);
    detail::trace_one("bitpat", i, kValues[i], rc, v, env);
    if (v != nullptr) js_set_element(env, array, static_cast<uint32_t>(i), v);
  }
  std::fflush(stderr);
  return array;
}

// =============================================================================
// 3. probeSyncMixedFirst
//   Calls `js_create_int32`, `js_create_uint32`, `js_create_int64` BEFORE
//   the first `js_create_double`. If "first call to ANY numeric API" is
//   broken, we'd see corruption on the int32 result; if only
//   `js_create_double`-specific, the int variants pass and only doubles
//   trip.
// =============================================================================
inline js_value_t* syncMixedFirst(js_env_t* env, js_callback_info_t* /*info*/) {
  std::fprintf(stderr, "=== probeSyncMixedFirst ===\n");

  // Burn an int32, an int64, then call doubles.
  js_value_t* tmp_i32 = nullptr;
  int rc_i32 = js_create_int32(env, 12345, &tmp_i32);
  int32_t rb_i32 = -1;
  if (tmp_i32 != nullptr) js_get_value_int32(env, tmp_i32, &rb_i32);
  std::fprintf(
      stderr, "  int32 input=12345 create_rc=%d cpp_readback=%d match=%s\n",
      rc_i32, rb_i32, (rb_i32 == 12345) ? "yes" : "NO");

  js_value_t* tmp_i64 = nullptr;
  int rc_i64 = js_create_int64(env, 9876543210LL, &tmp_i64);
  int64_t rb_i64 = -1;
  if (tmp_i64 != nullptr) js_get_value_int64(env, tmp_i64, &rb_i64);
  std::fprintf(
      stderr,
      "  int64 input=9876543210 create_rc=%d cpp_readback=%lld match=%s\n",
      rc_i64, static_cast<long long>(rb_i64),
      (rb_i64 == 9876543210LL) ? "yes" : "NO");

  static const std::vector<double> kValues = {0.708, 0.224, 0.068};
  js_value_t* array = nullptr;
  js_create_array_with_length(env, kValues.size(), &array);
  for (size_t i = 0; i < kValues.size(); ++i) {
    js_value_t* v = nullptr;
    int rc = js_create_double(env, kValues[i], &v);
    detail::trace_one("mixed-first", i, kValues[i], rc, v, env);
    if (v != nullptr) js_set_element(env, array, static_cast<uint32_t>(i), v);
  }
  std::fflush(stderr);
  return array;
}

// =============================================================================
// 4. probeSyncStringFirst
//   Mirrors the Whisper handler structure (object first, then string, then
//   number). If the bug reproduces here, it's NOT specific to "first JS-API
//   call after handle scope open"; it's specific to the first NUMBER created
//   regardless of what came before.
// =============================================================================
inline js_value_t* syncStringFirst(
    js_env_t* env, js_callback_info_t* /*info*/) {
  std::fprintf(stderr, "=== probeSyncStringFirst ===\n");

  js_value_t* obj = nullptr;
  js_create_object(env, &obj);
  js_value_t* str = nullptr;
  js_create_string_utf8(
      env, reinterpret_cast<const utf8_t*>("food"), 4, &str);
  js_set_named_property(env, obj, "label", str);
  std::fprintf(stderr, "  obj+string created\n");

  static const std::vector<double> kValues = {0.708, 0.224, 0.068};
  for (size_t i = 0; i < kValues.size(); ++i) {
    js_value_t* v = nullptr;
    int rc = js_create_double(env, kValues[i], &v);
    detail::trace_one("string-first", i, kValues[i], rc, v, env);
    if (i == 0 && v != nullptr) {
      // Mirror our handler exactly: assign first double as the
      // "confidence" property of the object.
      js_set_named_property(env, obj, "confidence", v);
    }
  }
  std::fflush(stderr);
  return obj;
}

// =============================================================================
// 5. probeSyncRepeated
//   Calls `syncDoubles`-equivalent pattern N times in a row (returns a
//   2-D array). Captures whether corruption is "first invocation only", "first
//   call within each invocation", or persistent.
// =============================================================================
inline js_value_t* syncRepeated(js_env_t* env, js_callback_info_t* /*info*/) {
  static const std::vector<double> kValues = {0.708, 0.224, 0.068};
  constexpr int kRepeats = 5;
  std::fprintf(
      stderr, "=== probeSyncRepeated (repeats=%d, N=%zu) ===\n", kRepeats,
      kValues.size());

  js_value_t* outer = nullptr;
  js_create_array_with_length(env, kRepeats, &outer);
  for (int r = 0; r < kRepeats; ++r) {
    std::fprintf(stderr, " -- invocation %d --\n", r);
    js_value_t* inner = nullptr;
    js_create_array_with_length(env, kValues.size(), &inner);
    for (size_t i = 0; i < kValues.size(); ++i) {
      js_value_t* v = nullptr;
      int rc = js_create_double(env, kValues[i], &v);
      char tag[16];
      std::snprintf(tag, sizeof(tag), "rep%d", r);
      detail::trace_one(tag, i, kValues[i], rc, v, env);
      if (v != nullptr) js_set_element(env, inner, static_cast<uint32_t>(i), v);
    }
    js_set_element(env, outer, static_cast<uint32_t>(r), inner);
  }
  std::fflush(stderr);
  return outer;
}

// =============================================================================
// 6. probeSyncNestedScopes
//   Manually opens an inner `js_handle_scope_t` (mirroring what
//   `OutputCallBackJs::jsOutputCallback` does at line 157) and runs the
//   doubles probe inside. Tests whether the nested-scope itself triggers the
//   first-call corruption, even synchronously.
// =============================================================================
inline js_value_t* syncNestedScopes(
    js_env_t* env, js_callback_info_t* /*info*/) {
  std::fprintf(stderr, "=== probeSyncNestedScopes ===\n");
  static const std::vector<double> kValues = {0.708, 0.224, 0.068};

  js_value_t* outer_array = nullptr;
  js_create_array_with_length(env, kValues.size(), &outer_array);

  js_handle_scope_t* inner_scope = nullptr;
  js_open_handle_scope(env, &inner_scope);
  std::fprintf(stderr, "  opened inner handle scope\n");

  for (size_t i = 0; i < kValues.size(); ++i) {
    js_value_t* v = nullptr;
    int rc = js_create_double(env, kValues[i], &v);
    detail::trace_one("nested", i, kValues[i], rc, v, env);
    // Note: cannot js_set_element on an array allocated in outer scope from
    // values created in inner scope without escaping; for diagnostic purposes
    // we only care about the readback trace, not the returned array.
  }

  js_close_handle_scope(env, inner_scope);
  std::fprintf(stderr, "  closed inner handle scope\n");
  std::fflush(stderr);
  return outer_array;
}

// =============================================================================
// 7. probeSyncStorageElement
//   Each value goes through `js_create_double` -> `js_set_element` ->
//   `js_get_element` -> `js_get_value_double`. Tests whether corruption
//   happens during element-write or element-read, not at create time.
// =============================================================================
inline js_value_t* syncStorageElement(
    js_env_t* env, js_callback_info_t* /*info*/) {
  std::fprintf(stderr, "=== probeSyncStorageElement ===\n");
  static const std::vector<double> kValues = {0.708, 0.224, 0.068};

  js_value_t* array = nullptr;
  js_create_array_with_length(env, kValues.size(), &array);
  for (size_t i = 0; i < kValues.size(); ++i) {
    js_value_t* v = nullptr;
    js_create_double(env, kValues[i], &v);
    js_set_element(env, array, static_cast<uint32_t>(i), v);

    js_value_t* fetched = nullptr;
    js_get_element(env, array, static_cast<uint32_t>(i), &fetched);
    double rb = -1.0;
    js_get_value_double(env, fetched, &rb);
    std::fprintf(
        stderr,
        "  [storeElem][%zu] input=%.9f after_set+get cpp_readback=%.9f match=%s\n",
        i, kValues[i], rb, (rb == kValues[i]) ? "yes" : "NO");
  }
  std::fflush(stderr);
  return array;
}

// =============================================================================
// 8. probeSyncStorageProperty
//   Same as #7 but uses `js_set_named_property` / `js_get_named_property`
//   on a plain object. Mirrors the actual `entry.setProperty(env,
//   "confidence", v)` code path in our handler.
// =============================================================================
inline js_value_t* syncStorageProperty(
    js_env_t* env, js_callback_info_t* /*info*/) {
  std::fprintf(stderr, "=== probeSyncStorageProperty ===\n");
  static const std::vector<double> kValues = {0.708, 0.224, 0.068};

  js_value_t* obj = nullptr;
  js_create_object(env, &obj);
  for (size_t i = 0; i < kValues.size(); ++i) {
    js_value_t* v = nullptr;
    js_create_double(env, kValues[i], &v);

    char propname[16];
    std::snprintf(propname, sizeof(propname), "p%zu", i);
    js_set_named_property(env, obj, propname, v);

    js_value_t* fetched = nullptr;
    js_get_named_property(env, obj, propname, &fetched);
    double rb = -1.0;
    js_get_value_double(env, fetched, &rb);
    std::fprintf(
        stderr,
        "  [storeProp][%zu] input=%.9f prop='%s' after_set+get "
        "cpp_readback=%.9f match=%s\n",
        i, kValues[i], propname, rb, (rb == kValues[i]) ? "yes" : "NO");
  }
  std::fflush(stderr);
  return obj;
}

// =============================================================================
// 9. probeSyncSequenceMimic
//   Exact mirror of `JsClassifyOutputHandler`'s call sequence:
//     Array::create -> [Object::create -> String::create -> Number::create -> set_element]*
//   with the burn-one workaround DELIBERATELY ABSENT. If this fails in the
//   same way as the integration test, the bug is fully reproduced through
//   only the synchronous JS-API path (no async dispatch needed).
// =============================================================================
inline js_value_t* syncSequenceMimic(
    js_env_t* env, js_callback_info_t* /*info*/) {
  std::fprintf(stderr, "=== probeSyncSequenceMimic (handler structure) ===\n");
  struct Item {
    const char* label;
    double conf;
  };
  static const Item kItems[3] = {
      {"food", 0.708}, {"other", 0.224}, {"report", 0.068}};

  js_value_t* array = nullptr;
  js_create_array_with_length(env, 3, &array);

  for (size_t i = 0; i < 3; ++i) {
    js_value_t* entry = nullptr;
    js_create_object(env, &entry);

    js_value_t* label = nullptr;
    js_create_string_utf8(
        env, reinterpret_cast<const utf8_t*>(kItems[i].label),
        std::strlen(kItems[i].label), &label);
    js_set_named_property(env, entry, "label", label);

    js_value_t* conf = nullptr;
    int rc = js_create_double(env, kItems[i].conf, &conf);
    detail::trace_one("mimic", i, kItems[i].conf, rc, conf, env);

    js_set_named_property(env, entry, "confidence", conf);
    js_set_element(env, array, static_cast<uint32_t>(i), entry);
  }
  std::fflush(stderr);
  return array;
}

// =============================================================================
// 10. probeSyncFpState
//   Reads MXCSR and x87 control word at three points: top of function, after
//   first js_create_double, after a sequence of doubles. Detects mutation of
//   FP state by the C-API call.
// =============================================================================
inline js_value_t* syncFpState(js_env_t* env, js_callback_info_t* /*info*/) {
  std::fprintf(stderr, "=== probeSyncFpState ===\n");

  unsigned int mxcsr_before = detail::read_mxcsr();
  unsigned int x87_before = detail::read_x87();
  std::fprintf(
      stderr, "  before: mxcsr=0x%08x x87=0x%08x\n", mxcsr_before, x87_before);

  js_value_t* v1 = nullptr;
  js_create_double(env, 0.708, &v1);
  unsigned int mxcsr_mid = detail::read_mxcsr();
  unsigned int x87_mid = detail::read_x87();
  std::fprintf(
      stderr, "  after 1 double: mxcsr=0x%08x x87=0x%08x diff_mxcsr=0x%08x\n",
      mxcsr_mid, x87_mid, mxcsr_mid ^ mxcsr_before);

  for (int i = 0; i < 9; ++i) {
    js_value_t* tmp = nullptr;
    js_create_double(env, 0.5 + i * 0.05, &tmp);
  }
  unsigned int mxcsr_after = detail::read_mxcsr();
  unsigned int x87_after = detail::read_x87();
  std::fprintf(
      stderr, "  after 10 doubles: mxcsr=0x%08x x87=0x%08x diff_mxcsr=0x%08x\n",
      mxcsr_after, x87_after, mxcsr_after ^ mxcsr_before);
  std::fflush(stderr);

  return v1;
}

// =============================================================================
// 11. probeAsyncCallback
//   Uses `uv_async_send` to fire a callback on the JS event loop thread,
//   inside which we open nested handle scopes (mimicking
//   `OutputCallBackJs::jsOutputCallback`) and run the doubles probe. Returns
//   a Promise that resolves with the result array once the async callback
//   has completed.
//
//   This is the closest standalone repro of the actual failing path. If the
//   sync probes pass on win32-x64 CI but THIS one fails, the bug is in the
//   async-callback / nested-scope context, not in `js_create_double` per se.
// =============================================================================
struct AsyncProbeContext {
  js_env_t* env;
  js_deferred_t* deferred;
  uv_async_t async;
  std::vector<double> input;
  std::vector<double> readback;
  std::vector<int> create_rcs;
  std::vector<int> readback_rcs;
};

inline void asyncProbeFire(uv_async_t* handle) {
  auto* ctx = static_cast<AsyncProbeContext*>(
      uv_handle_get_data(reinterpret_cast<uv_handle_t*>(handle)));
  std::fprintf(stderr, "=== probeAsyncCallback (uv_async fired) ===\n");

  js_handle_scope_t* outer = nullptr;
  js_open_handle_scope(ctx->env, &outer);

  js_handle_scope_t* inner = nullptr;
  js_open_handle_scope(ctx->env, &inner);

  for (size_t i = 0; i < ctx->input.size(); ++i) {
    js_value_t* v = nullptr;
    ctx->create_rcs[i] = js_create_double(ctx->env, ctx->input[i], &v);
    double rb = -1.0;
    int rrc = -999;
    if (v != nullptr) {
      rrc = js_get_value_double(ctx->env, v, &rb);
    }
    ctx->readback[i] = rb;
    ctx->readback_rcs[i] = rrc;
    detail::trace_one("async", i, ctx->input[i], ctx->create_rcs[i], v, ctx->env);
  }

  // Build resolution array (created in inner scope; we'll let outer escape it).
  js_value_t* result = nullptr;
  js_create_array_with_length(ctx->env, ctx->readback.size(), &result);
  for (size_t i = 0; i < ctx->readback.size(); ++i) {
    js_value_t* num = nullptr;
    js_create_double(ctx->env, ctx->readback[i], &num);
    if (num != nullptr) {
      js_set_element(ctx->env, result, static_cast<uint32_t>(i), num);
    }
  }

  js_resolve_deferred(ctx->env, ctx->deferred, result);

  js_close_handle_scope(ctx->env, inner);
  js_close_handle_scope(ctx->env, outer);

  // Schedule cleanup via uv_close so we don't free a handle libuv still holds.
  uv_close(reinterpret_cast<uv_handle_t*>(handle), [](uv_handle_t* h) {
    auto* c = static_cast<AsyncProbeContext*>(uv_handle_get_data(h));
    delete c;
  });
  std::fflush(stderr);
}

inline js_value_t* asyncCallback(
    js_env_t* env, js_callback_info_t* /*info*/) {
  std::fprintf(stderr, "=== probeAsyncCallback (scheduling) ===\n");

  uv_loop_t* loop = nullptr;
  if (js_get_env_loop(env, &loop) != 0 || loop == nullptr) {
    std::fprintf(stderr, "  failed to get env loop\n");
    return nullptr;
  }

  auto* ctx = new AsyncProbeContext{};
  ctx->env = env;
  ctx->input = {0.708, 0.224, 0.068, 0.500, 0.999};
  ctx->readback.assign(ctx->input.size(), -1.0);
  ctx->create_rcs.assign(ctx->input.size(), -1);
  ctx->readback_rcs.assign(ctx->input.size(), -1);

  js_value_t* promise = nullptr;
  js_create_promise(env, &ctx->deferred, &promise);

  if (uv_async_init(loop, &ctx->async, asyncProbeFire) != 0) {
    std::fprintf(stderr, "  uv_async_init failed\n");
    delete ctx;
    return promise; // unresolved; caller's await will hang
  }
  uv_handle_set_data(reinterpret_cast<uv_handle_t*>(&ctx->async), ctx);

  uv_async_send(&ctx->async);
  return promise;
}

} // namespace qvac_lib_infer_ggml_classification::probe
