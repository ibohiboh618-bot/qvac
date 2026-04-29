#include <bare.h>

#include "addon/AddonJs.hpp"
// === [TEMP] Marshalling probe -- DIAGNOSTIC SCAFFOLDING, DO NOT KEEP ===
// Adds a set of `binding.probe*` functions used by
// `scripts/marshal-probe.js` to localise the win32-x64 first-double
// marshalling bug. Removal: delete `marshal_probe.hpp`, this `#include`,
// the matching `V("probe*", ...)` block in this file, the
// `scripts/marshal-probe.js` driver, and the "Run marshalling probe"
// step in `.github/workflows/integration-test-classification-ggml.yml`.
// See top of `marshal_probe.hpp` for the full removal checklist.
#include "marshal_probe.hpp"
// === [/TEMP] ===

js_value_t* qvac_lib_infer_ggml_classification_exports(
    js_env_t* env,
    js_value_t* exports) { // NOLINT(readability-identifier-naming)

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define V(name, fn)                                                            \
  {                                                                            \
    js_value_t* val;                                                           \
    if (js_create_function(env, name, -1, fn, nullptr, &val) != 0) {           \
      return nullptr;                                                          \
    }                                                                          \
    if (js_set_named_property(env, exports, name, val) != 0) {                 \
      return nullptr;                                                          \
    }                                                                          \
  }

  V("createInstance",
    qvac_lib_infer_ggml_classification::bindings::createInstance)
  V("runJob", qvac_lib_infer_ggml_classification::bindings::runJob)

  V("loadWeights", qvac_lib_inference_addon_cpp::JsInterface::loadWeights)
  V("activate", qvac_lib_inference_addon_cpp::JsInterface::activate)
  V("cancel", qvac_lib_inference_addon_cpp::JsInterface::cancel)
  V("destroyInstance",
    qvac_lib_inference_addon_cpp::JsInterface::destroyInstance)
  V("setLogger", qvac_lib_inference_addon_cpp::JsInterface::setLogger)
  V("releaseLogger",
    qvac_lib_inference_addon_cpp::JsInterface::releaseLogger)

  // === [TEMP] Marshalling probe registrations -- remove with the rest. ===
  V("probeSyncDoubles",
    qvac_lib_infer_ggml_classification::probe::syncDoubles)
  V("probeSyncBitPatterns",
    qvac_lib_infer_ggml_classification::probe::syncBitPatterns)
  V("probeSyncMixedFirst",
    qvac_lib_infer_ggml_classification::probe::syncMixedFirst)
  V("probeSyncStringFirst",
    qvac_lib_infer_ggml_classification::probe::syncStringFirst)
  V("probeSyncRepeated",
    qvac_lib_infer_ggml_classification::probe::syncRepeated)
  V("probeSyncNestedScopes",
    qvac_lib_infer_ggml_classification::probe::syncNestedScopes)
  V("probeSyncStorageElement",
    qvac_lib_infer_ggml_classification::probe::syncStorageElement)
  V("probeSyncStorageProperty",
    qvac_lib_infer_ggml_classification::probe::syncStorageProperty)
  V("probeSyncSequenceMimic",
    qvac_lib_infer_ggml_classification::probe::syncSequenceMimic)
  V("probeSyncFpState",
    qvac_lib_infer_ggml_classification::probe::syncFpState)
  V("probeAsyncCallback",
    qvac_lib_infer_ggml_classification::probe::asyncCallback)
  // === [/TEMP] ===

#undef V
  // NOLINTEND(cppcoreguidelines-macro-usage)

  return exports;
}

BARE_MODULE(
    qvac - lib - infer - ggml - classification,
    qvac_lib_infer_ggml_classification_exports)
