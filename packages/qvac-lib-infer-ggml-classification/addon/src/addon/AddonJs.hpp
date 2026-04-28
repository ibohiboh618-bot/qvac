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

// NOTE on previously-applied workarounds (intentionally removed for
// consistency with other addons; pending a single upstream fix in
// qvac-lib-inference-addon-cpp / bare-runtime):
//
//   1. `DeferredOutputCallBackJs` wrapper -- deferred destruction of
//      the upstream `OutputCallBackJs` to a `uv_check_t` to avoid a
//      use-after-free race against pending `uv_async_send` callbacks.
//      The race is in `~OutputCallBackJs` (queue/OutputCallbackJs.hpp,
//      lines 48-58): JS refs are deleted synchronously while the
//      async handle's `uv_close` is still pending, so a queued
//      `jsOutputCallback` can fire afterwards against dead refs.
//      Reproduced in CI as SIGSEGV on linux-x64 / darwin-arm64 /
//      android / ios across rapid ImageClassifier create/destroy
//      cycles. Other addons in this repo work around the same race
//      empirically by sleeping 1-20s after `unload()` in their
//      tests (see ocr-onnx/test/integration/lifecycle.test.js:56,85,
//      115; full-ocr-suite.test.js:107,115,123;
//      qvac-lib-infer-llamacpp-llm/test/integration/sliding-context.test.js:
//      163,355). We adopt the same approach in
//      `test/integration/error-cases.test.js` until upstream lands.
//
//   2. Win32 "burn-one" `js_create_double(env, 0.0, &dummy)` at the
//      top of `JsClassifyOutputHandler`'s lambda -- the very first
//      `js_create_double` call after entering an `OutputCallBackJs`
//      callback returned 0.0 on win32-x64 (clang-cl + bare-runtime +
//      V8) regardless of the input value. Other addons accidentally
//      dodge this either because (a) their first emitted number is
//      naturally 0 (whisper/parakeet segment.start), (b) their tests
//      assert only `typeof === 'number'` / `!isNaN` (llamacpp-llm
//      stats checks), (c) the first emitted number is never asserted
//      on (ocr-onnx bounding-box coords), or (d) they do not emit
//      numbers at all (lib-infer-diffusion). We are dropping the
//      workaround here to keep the addon consistent; win32-x64 CI is
//      expected to start failing on the first per-image confidence
//      until the upstream marshalling layer is patched.

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

  auto configObj = args.getJsObject(1, "config");
  auto modelPath =
      configObj.getProperty<jsu::String>(env, "path").as<std::string>(env);

  auto model = std::make_unique<ClassificationModel>(modelPath);

  // Optional threads hint nested under `config`.
  auto innerConfig =
      configObj.getOptionalProperty<jsu::Object>(env, "config");
  if (innerConfig.has_value()) {
    auto threadsOpt =
        innerConfig->getOptionalProperty<jsu::Number>(env, "threads");
    if (threadsOpt.has_value()) {
      model->setNumThreads(threadsOpt->as<int32_t>(env));
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

  // The JS layer delivers a single message of type "image" carrying the
  // image buffer and, optionally, raw-RGB dimension metadata.
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

  // Image buffer: accept Uint8Array / Buffer (stored as TypedArray on the JS
  // side).
  auto bufferVal = inputObj.getProperty(env, "content");
  if (!jsu::is<jsu::TypedArray<uint8_t>>(env, bufferVal)) {
    throw StatusError(
        InvalidArgument,
        "Image 'content' must be a Uint8Array / Buffer of encoded JPEG/PNG "
        "bytes or raw RGB bytes.");
  }
  auto ta = jsu::TypedArray<uint8_t>(env, bufferVal);
  auto span = ta.as<std::span<const uint8_t>>(env);
  if (span.empty()) {
    throw StatusError(InvalidArgument, "Image 'content' buffer is empty");
  }
  cppInput.data.assign(span.begin(), span.end());

  // Optional dimension metadata for the raw-RGB path.
  auto widthOpt = inputObj.getOptionalProperty<jsu::Number>(env, "width");
  auto heightOpt = inputObj.getOptionalProperty<jsu::Number>(env, "height");
  auto channelsOpt =
      inputObj.getOptionalProperty<jsu::Number>(env, "channels");
  if (widthOpt.has_value()) {
    cppInput.width = widthOpt->as<uint32_t>(env);
  }
  if (heightOpt.has_value()) {
    cppInput.height = heightOpt->as<uint32_t>(env);
  }
  if (channelsOpt.has_value()) {
    cppInput.channels = channelsOpt->as<uint32_t>(env);
  }

  auto topKOpt = inputObj.getOptionalProperty<jsu::Number>(env, "topK");
  if (topKOpt.has_value()) {
    cppInput.topK = topKOpt->as<uint32_t>(env);
  }

  return instance.runJob(std::any(std::move(cppInput)));
}
JSCATCH

} // namespace qvac_lib_infer_ggml_classification::bindings
