#pragma once

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <js.h>
#include <qvac-lib-inference-addon-cpp/Errors.hpp>
#include <qvac-lib-inference-addon-cpp/JsInterface.hpp>
#include <qvac-lib-inference-addon-cpp/JsUtils.hpp>
#include <qvac-lib-inference-addon-cpp/ModelInterfaces.hpp>
#include <qvac-lib-inference-addon-cpp/addon/AddonJs.hpp>
#include <qvac-lib-inference-addon-cpp/handlers/JsOutputHandlerImplementations.hpp>
#include <qvac-lib-inference-addon-cpp/handlers/OutputHandler.hpp>
#include <qvac-lib-inference-addon-cpp/queue/OutputCallbackJs.hpp>
#include <qvac-lib-inference-addon-cpp/queue/OutputQueue.hpp>

#include "model-interface/ClassificationModel.hpp"

namespace qvac_lib_infer_ggml_classification::bindings {

namespace addon_cpp = qvac_lib_inference_addon_cpp;
namespace jsu = qvac_lib_inference_addon_cpp::js;

using qvac_errors::StatusError;
using qvac_errors::general_error::InvalidArgument;

// NOTE on two upstream issues this binding has had to work around,
// pending a single upstream fix in qvac-lib-inference-addon-cpp /
// bare-runtime. Each was independently surfaced by CI; their fixes
// are deliberately asymmetric because their nature is different.
//
//   1. `OutputCallBackJs` use-after-free race (worked around in
//      tests, NOT in this file).
//      Race is in `~OutputCallBackJs` (queue/OutputCallbackJs.hpp,
//      lines 48-58): JS refs are deleted synchronously while the
//      async handle's `uv_close` is still pending, so a queued
//      `jsOutputCallback` can fire afterwards against dead refs.
//      Reproduced in CI as SIGSEGV on linux-x64 / darwin-arm64 /
//      android / ios across rapid ImageClassifier create/destroy
//      cycles (CI runs 24891210942, 24892803959, 25062157099,
//      25070800838). Local C++ workaround
//      (`DeferredOutputCallBackJs` wrapper) was removed for
//      consistency with how other addons in this monorepo cope
//      with the same upstream bug -- they paper over it by
//      sleeping after every `unload()` in their integration tests
//      (see ocr-onnx/test/integration/lifecycle.test.js:56,85,
//      115; full-ocr-suite.test.js:107,115,123;
//      qvac-lib-infer-llamacpp-llm/test/integration/sliding-context.test.js:
//      163,355). We adopted that pattern in
//      `test/integration/utils.js::cleanupClassifier()`
//      (two-phase: 500-1000ms pre-unload yield to drain in-flight
//      worker events, plus 2000-3000ms post-unload sleep). CI run
//      25074595106 confirmed the test-side workaround is
//      sufficient on every platform that uses `OutputCallBackJs`.
//
//   2. Win32 first-`js_create_double` returns 0.0 (worked around
//      below in `JsClassifyOutputHandler`).
//      On win32-x64 (clang-cl + bare-runtime + V8) the very first
//      `js_create_double` call after entering an
//      `OutputCallBackJs` callback returns 0.0 regardless of the
//      input value. Subsequent calls in the same handle scope are
//      correct. Reproduced as test failure on `meal_1.jpg`
//      ("sorted desc [0]>=[1]" with confidence[0] = 0.0) in CI
//      runs 24851301107, 24891210942, 24897445066, 24900278513,
//      25002820522, 25062157099, 25070800838, 25074595106.
//      No test-side workaround is possible: the bug corrupts a
//      semantically-critical value (the highest-confidence class
//      after sort). Other addons accidentally dodge it only
//      because (a) their first emitted number happens to be 0
//      (whisper/parakeet segment.start), (b) they assert only
//      `typeof === 'number'` / `!isNaN` (llamacpp-llm stats),
//      (c) they never assert the value (ocr-onnx bounding-box
//      coords), or (d) they do not emit numbers at all
//      (lib-infer-diffusion). The local C++ "burn one" workaround
//      in the lambda below consumes the broken first slot so the
//      per-element marshalling that follows is correct on every
//      platform; cost is one ephemeral js_number per classify()
//      call. To be removed once the upstream marshalling layer is
//      patched.

/// Handler mapping ClassifyOutput → JS array of { label, confidence }.
///
/// Implementation notes:
///   - The label string and the confidence float are read into named
///     local variables before being handed to the JS-side helpers.
///     This is defensive against compiler code-gen quirks (notably
///     observed on win32-x64 / clang-cl, where an inline
///     `static_cast<double>(cppOut.results[i].confidence)` fed
///     directly into `Number::create(...)` lost the value at index
///     0 of the result array, while indices 1..N marshalled
///     correctly). Reading into named locals forces the compiler
///     to materialise the values before the call sequence and
///     gives us a stable point to instrument when diagnosing
///     marshalling issues.
///   - When `QVAC_CLASSIFICATION_TRACE=1` is set in the
///     environment, every entry is printed to stderr with both
///     the C++ float view and the C++ double view of the value.
///     Combined with the per-inference trace in
///     `ClassificationModel::process()`, this lets us pinpoint
///     exactly which step (sort / marshal / JS-side conversion)
///     loses a value when one ever does.
struct JsClassifyOutputHandler
    : addon_cpp::out_handl::JsBaseOutputHandler<ClassifyOutput> {
  JsClassifyOutputHandler()
      : JsBaseOutputHandler<ClassifyOutput>(
            [this](const ClassifyOutput& cppOut) -> js_value_t* {
              auto array = jsu::Array::create(this->env_);
              const bool trace = []() {
                const char* v = std::getenv("QVAC_CLASSIFICATION_TRACE");
                return v != nullptr && v[0] == '1';
              }();

              // ----- WIN32 MARSHALLING WORKAROUND -----
              // See note 2 in the file-level comment above. The very
              // first `js_create_double` call after entering this
              // lambda on win32-x64 (clang-cl + bare-runtime + V8)
              // returns 0.0 regardless of the input value. We burn
              // that broken slot with a throwaway call so the
              // per-element `Number::create` calls below produce the
              // correct value at index 0. The throwaway value is
              // intentionally never wired into the array; cost is
              // one ephemeral js_number per classify() call. To be
              // removed once the upstream bare-runtime marshalling
              // layer is patched.
              {
                js_value_t* dummy = nullptr;
                (void)js_create_double(this->env_, 0.0, &dummy);
                (void)dummy;
              }
              // ----------------------------------------

              for (size_t i = 0; i < cppOut.results.size(); ++i) {
                // Read into named locals BEFORE creating any JS values.
                const std::string& labelString = cppOut.results[i].label;
                const float confidenceFloat = cppOut.results[i].confidence;
                const double confidenceDouble =
                    static_cast<double>(confidenceFloat);

                if (trace) {
                  std::fprintf(
                      stderr,
                      "[qvac-classify-marshal] i=%zu label='%s' "
                      "confidence_float=%.9f confidence_double=%.9f\n",
                      i,
                      labelString.c_str(),
                      static_cast<double>(confidenceFloat),
                      confidenceDouble);
                  std::fflush(stderr);
                }

                auto entry = jsu::Object::create(this->env_);
                entry.setProperty(
                    this->env_,
                    "label",
                    jsu::String::create(this->env_, labelString));
                entry.setProperty(
                    this->env_,
                    "confidence",
                    jsu::Number::create(this->env_, confidenceDouble));
                array.set(this->env_, i, entry);
              }
              return array;
            }) {}
};

inline js_value_t* createInstance(
    js_env_t* env, js_callback_info_t* info) try {
  addon_cpp::JsArgsParser args(env, info);

  // Single-place parsing & validation of the JS-side configuration
  // object. The JS wrapper passes config straight through; this is
  // where every "is it a string / positive integer / present" rule
  // lives, so there is one source of truth for what counts as a
  // valid construction argument.
  auto configObj = args.getJsObject(1, "config");
  auto modelPath =
      configObj.getProperty<jsu::String>(env, "path").as<std::string>(env);
  if (modelPath.empty()) {
    throw StatusError(
        InvalidArgument,
        "Configuration 'path' is required and must be a non-empty string "
        "pointing at the FP16 GGUF weights file");
  }

  auto model = std::make_unique<ClassificationModel>(modelPath);

  // Optional threads hint nested under `config`. When provided it
  // must be a positive integer; bare-runtime's `as<int32_t>` truncates
  // floats and accepts negatives without complaint, so we range-check
  // explicitly.
  auto innerConfig =
      configObj.getOptionalProperty<jsu::Object>(env, "config");
  if (innerConfig.has_value()) {
    auto threadsOpt =
        innerConfig->getOptionalProperty<jsu::Number>(env, "threads");
    if (threadsOpt.has_value()) {
      const int32_t threads = threadsOpt->as<int32_t>(env);
      if (threads < 1) {
        throw StatusError(
            InvalidArgument,
            "Configuration 'config.threads' must be a positive integer "
            "when provided; got " + std::to_string(threads));
      }
      model->setNumThreads(threads);
    }
  }

  model->load();

  addon_cpp::out_handl::OutputHandlers<addon_cpp::out_handl::JsOutputHandlerInterface>
      outHandlers;
  outHandlers.add(std::make_shared<JsClassifyOutputHandler>());

  auto callback = std::make_unique<addon_cpp::OutputCallBackJs>(
      env, args.get(0, "jsHandle"), args.getFunction(2, "outputCallback"),
      std::move(outHandlers));

  auto addon = std::make_unique<addon_cpp::AddonJs>(
      env, std::move(callback),
      std::unique_ptr<addon_cpp::model::IModel>(std::move(model)));

  return addon_cpp::JsInterface::createInstance(env, std::move(addon));
}
JSCATCH

inline js_value_t* runJob(js_env_t* env, js_callback_info_t* info) try {
  addon_cpp::JsArgsParser args(env, info);
  addon_cpp::AddonJs& instance =
      addon_cpp::JsInterface::getInstance(env, args.get(0, "instance"));

  // Single-place parsing & validation of the JS-side per-call payload.
  // Every "is the argument the right type / range / shape" rule for
  // a classify() call lives here, so the JS wrapper can be a thin
  // pass-through and the model + preprocessor only ever operate on
  // an already-validated `ClassifyInput`.
  auto inputObj = args.getJsObject(1, "inputObj");
  auto type =
      inputObj.getProperty<jsu::String>(env, "type").as<std::string>(env);
  if (type != "image") {
    throw StatusError(
        InvalidArgument,
        "Classification addon accepts only 'image' input type, got '" + type +
            "'");
  }

  ClassifyInput cppInput;

  // Image buffer: accept Uint8Array / Buffer (stored as TypedArray on the
  // JS side). The "is required" wording in the message is the contract
  // surface; existing JS-side tests assert against the substring
  // "required" / "null" / "undefined" for the null-input path.
  auto bufferVal = inputObj.getProperty(env, "content");
  if (!jsu::is<jsu::TypedArray<uint8_t>>(env, bufferVal)) {
    throw StatusError(
        InvalidArgument,
        "Image 'content' is required and must be a Uint8Array / Buffer of "
        "encoded JPEG/PNG bytes or raw RGB bytes (got null, undefined, or "
        "wrong type)");
  }
  auto ta = jsu::TypedArray<uint8_t>(env, bufferVal);
  auto span = ta.as<std::span<const uint8_t>>(env);
  if (span.empty()) {
    throw StatusError(InvalidArgument, "Image 'content' buffer is empty");
  }
  cppInput.data.assign(span.begin(), span.end());

  // Optional dimension metadata for the raw-RGB path. Either:
  //   - none of {width, height, channels} are provided  -> encoded JPEG/PNG
  //   - all three are provided and validated            -> raw RGB
  // Anything in between is a programming error in the caller; reject
  // explicitly rather than letting validateRawRgb produce a confusing
  // size-mismatch error downstream.
  auto widthOpt = inputObj.getOptionalProperty<jsu::Number>(env, "width");
  auto heightOpt = inputObj.getOptionalProperty<jsu::Number>(env, "height");
  auto channelsOpt =
      inputObj.getOptionalProperty<jsu::Number>(env, "channels");
  const int provided = (widthOpt.has_value() ? 1 : 0) +
                       (heightOpt.has_value() ? 1 : 0) +
                       (channelsOpt.has_value() ? 1 : 0);
  if (provided != 0 && provided != 3) {
    throw StatusError(
        InvalidArgument,
        "Raw RGB input requires all of 'width', 'height', and 'channels' "
        "to be provided together; received " + std::to_string(provided) +
        " of 3");
  }
  if (provided == 3) {
    // bare-runtime's `as<uint32_t>` does a static_cast on a possibly-
    // negative or fractional JS Number; range-check the int32_t view
    // first so a negative width does not silently wrap to ~4 billion
    // and tunnel into validateRawRgb as a buffer-size mismatch.
    const int32_t w = widthOpt->as<int32_t>(env);
    const int32_t h = heightOpt->as<int32_t>(env);
    const int32_t c = channelsOpt->as<int32_t>(env);
    if (w <= 0) {
      throw StatusError(
          InvalidArgument,
          "Image 'width' must be a positive integer when passing raw RGB "
          "bytes; got " + std::to_string(w));
    }
    if (h <= 0) {
      throw StatusError(
          InvalidArgument,
          "Image 'height' must be a positive integer when passing raw RGB "
          "bytes; got " + std::to_string(h));
    }
    if (c != 3) {
      throw StatusError(
          InvalidArgument,
          "Image 'channels' must be exactly 3 (RGB) when passing raw RGB "
          "bytes; got " + std::to_string(c));
    }
    cppInput.rawRgb = RawRgbDims{
        static_cast<uint32_t>(w), static_cast<uint32_t>(h),
        static_cast<uint32_t>(c)};
  }

  auto topKOpt = inputObj.getOptionalProperty<jsu::Number>(env, "topK");
  if (topKOpt.has_value()) {
    const int32_t topK = topKOpt->as<int32_t>(env);
    if (topK <= 0) {
      throw StatusError(
          InvalidArgument,
          "Image 'topK' must be a positive integer when provided; got " +
              std::to_string(topK));
    }
    cppInput.topK = static_cast<uint32_t>(topK);
  }

  return instance.runJob(std::any(std::move(cppInput)));
}
JSCATCH

} // namespace qvac_lib_infer_ggml_classification::bindings
