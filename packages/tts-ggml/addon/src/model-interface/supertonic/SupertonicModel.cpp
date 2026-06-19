#include "model-interface/supertonic/SupertonicModel.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <tts-cpp/supertonic/engine.h>
// DEBUG (QVAC-20557 GPU correctness bring-up, DO-NOT-MERGE): tts_cpp_log_set +
// ggml_log_callback/ggml_log_level (pulls in ggml.h) — see ggmlLogTrampoline.
#include <tts-cpp/log.h>

#if defined(__ANDROID__)
// DEBUG (Mali/Adreno bring-up): __android_log_print — see emitDeviceDiag.
#include <android/log.h>
#endif

#include "addon/TTSErrors.hpp"
#include "model-interface/BackendUtils.hpp"
#include "inference-addon-cpp/Errors.hpp"
#include "inference-addon-cpp/Logger.hpp"

namespace qvac::ttsggml::supertonic {

namespace {

using qvac_errors::createTTSError;
using qvac_errors::StatusError;
using qvac_errors::tts_error::TTSErrorCode;
namespace general_error = qvac_errors::general_error;
namespace logger = qvac_lib_inference_addon_cpp::logger;

// DEBUG (QVAC-20557 GPU correctness bring-up, DO-NOT-MERGE) — native log bridge,
// ported from QVAC PR #2610/#2601. On-device, QLOG/JsLogger rides a uv_async
// callback that never reaches a captured sink in the embedded Bare-in-app
// (React-Native host) runtime, and native stderr is swallowed — so native
// diagnostics vanish. Emit STRAIGHT to the platform log: __android_log_print
// lands synchronously in the full device logcat artifact (logcat_full.txt).
// Off-device, stderr keeps local pre-flight working.
void emitDeviceDiag(const std::string& line) {
#if defined(__ANDROID__)
  __android_log_print(ANDROID_LOG_INFO, "qvac-supertonic", "%s", line.c_str());
#else
  std::fputs(line.c_str(), stderr);
  std::fputc('\n', stderr);
  std::fflush(stderr);
#endif
}

// ggml emits log text in fragments that are not necessarily newline-terminated;
// buffer and flush complete lines so backend-init banners ("ggml_vulkan: …
// Mali-G715 … fp16…"), op-support warnings, and unsupported-op fallbacks each
// reach the device log as one clean line. Installed via tts_cpp_log_set, which
// forwards to ggml_log_set.
void ggmlLogTrampoline(ggml_log_level /*level*/, const char* text,
                       void* /*user_data*/) {
  if (!text) return;
  static std::mutex mu;
  static std::string buf;
  std::lock_guard<std::mutex> lk(mu);
  buf += text;
  std::size_t nl;
  while ((nl = buf.find('\n')) != std::string::npos) {
    emitDeviceDiag(buf.substr(0, nl));
    buf.erase(0, nl + 1);
  }
}

void installGgmlLogTrampolineOnce() {
  static std::once_flag once;
  std::call_once(once, [] { tts_cpp_log_set(&ggmlLogTrampoline, nullptr); });
}

// DEBUG (QVAC-20557, DO-NOT-MERGE): per-stage f32 dumps key off one dir,
// $TTS_CPP_GPU_DUMP_DIR (set by the gpu-smoke test to a device path the
// device-farm pulls). Empty/unset = no dump.
std::string gpuDumpDir() {
  const char* d = std::getenv("TTS_CPP_GPU_DUMP_DIR");
  return (d && *d) ? std::string(d) : std::string();
}

tts_cpp::supertonic::EngineOptions toEngineOptions(const SupertonicConfig& cfg) {
  tts_cpp::supertonic::EngineOptions opts;
  opts.model_gguf_path = cfg.modelGgufPath;
  opts.voice           = cfg.voice;
  if (!cfg.language.empty()) opts.language = cfg.language;
  if (cfg.steps.has_value())   opts.steps = *cfg.steps;
  if (cfg.speed.has_value())   opts.speed = *cfg.speed;
  if (cfg.seed.has_value())    opts.seed  = *cfg.seed;
  if (cfg.threads.has_value()) opts.n_threads = *cfg.threads;
  if (cfg.nGpuLayers.has_value()) {
    opts.n_gpu_layers = *cfg.nGpuLayers;
  } else if (cfg.useGpu.has_value()) {
    opts.n_gpu_layers = *cfg.useGpu ? 99 : 0;
  }
  opts.noise_npy_path = cfg.noiseNpyPath;

  // Mirrors ChatterboxModel::toEngineOptions; see that file for the
  // detailed rationale. Compose `cfg.backendsDir / BACKENDS_SUBDIR`
  // before forwarding so a host that already passes
  // `path.join(__dirname, 'prebuilds')` (the qvac
  // llm-llamacpp / transcription-parakeet convention) gets the
  // expected `<bare-target>/qvac__tts-ggml/` scan dir without
  // knowing the per-arch shape.
  if (!cfg.backendsDir.empty()) {
    std::filesystem::path backendsDirPath(cfg.backendsDir);
#ifdef BACKENDS_SUBDIR
    backendsDirPath =
        (backendsDirPath / std::filesystem::path(BACKENDS_SUBDIR)).lexically_normal();
#endif
    opts.backends_dir = backendsDirPath.string();
  }
  opts.opencl_cache_dir = cfg.openclCacheDir;

  // DEBUG (QVAC-20557, DO-NOT-MERGE): route the engine's per-stage [gpu-diag]
  // lines (latent_in/text_emb/cfm_latent/wav_full) to the device log bridge so
  // they reach logcat_full.txt; when $TTS_CPP_GPU_DUMP_DIR is set, also dump
  // each stage's raw f32 there for GPU-vs-CPU correlation. Supertonic is
  // deterministic so the CPU run loads as a separate model (no token pinning).
  opts.diag_sink     = &emitDeviceDiag;
  opts.diag_dump_dir = gpuDumpDir();
  return opts;
}

std::vector<int16_t> pcmFloatToInt16(const float* pcm, size_t samples) {
  std::vector<int16_t> out;
  out.resize(samples);
  for (size_t i = 0; i < samples; ++i) {
    float s = std::clamp(pcm[i], -1.0f, 1.0f);
    out[i] = static_cast<int16_t>(std::lround(s * 32767.0f));
  }
  return out;
}

}

SupertonicModel::SupertonicModel(SupertonicConfig config)
    : cfg_(std::move(config)) {
  validateConfig(cfg_);
  // See ChatterboxModel ctor: load() is deferred to
  // waitForLoadInitialization() so the GGUF parse runs off the JS event
  // loop via JsAsyncTask::run-driven addon.activate().
}

SupertonicModel::~SupertonicModel() noexcept = default;

void SupertonicModel::validateConfig(const SupertonicConfig& cfg) {
  if (cfg.modelGgufPath.empty()) {
    throw StatusError(general_error::InvalidArgument,
                      "supertonicModelPath is required");
  }
  if (!std::filesystem::exists(cfg.modelGgufPath)) {
    throw createTTSError(TTSErrorCode::ModelFileNotFound,
                         "supertonic model not found: " + cfg.modelGgufPath);
  }
  if (cfg.steps.has_value() && *cfg.steps < 0) {
    throw StatusError(general_error::InvalidArgument,
                      "steps must be >= 0");
  }
  if (cfg.speed.has_value() && *cfg.speed < 0.0f) {
    throw StatusError(general_error::InvalidArgument,
                      "speed must be >= 0");
  }
  if (!cfg.noiseNpyPath.empty() &&
      !std::filesystem::exists(cfg.noiseNpyPath)) {
    throw createTTSError(TTSErrorCode::ModelFileNotFound,
                         "noise npy not found: " + cfg.noiseNpyPath);
  }
  // Defense-in-depth: the JS layer (index.js::_validateConfig) runs the
  // same conflict check before this method is reached, so direct C++
  // callers are the only ones who can actually trip this branch.
  // Mirror the Chatterbox suffix verbatim so users see an identical
  // hint regardless of which engine they instantiated.  `layers != 0`
  // matches llama.cpp's "-1 = offload all" sentinel convention.
  if (cfg.useGpu.has_value() && cfg.nGpuLayers.has_value()) {
    const bool wantsGpuFlag   = *cfg.useGpu;
    const int  layers         = *cfg.nGpuLayers;
    const bool layersWantGpu  = layers != 0;
    if (wantsGpuFlag != layersWantGpu) {
      throw StatusError(
          general_error::InvalidArgument,
          std::string("SupertonicModel: useGPU=") +
              (wantsGpuFlag ? "true" : "false") +
              " conflicts with nGpuLayers=" + std::to_string(layers) +
              ". Either drop one of the two, or make them agree "
              "(useGPU:true + nGpuLayers!=0, or useGPU:false + nGpuLayers=0).");
    }
  }
  // GPU execution is honored for Supertonic on GPU-capable hosts (Metal on
  // Apple, Vulkan/CUDA on desktop). On Android it is still forced to CPU in
  // loadLocked() below (Adreno OpenCL/Vulkan ggml graph compute is unstable);
  // the cross-field conflict check above is the only hard rejection here.
}

void SupertonicModel::load() {
  std::lock_guard lk(engineMu_);
  loadLocked();
}

void SupertonicModel::unload() {
  std::lock_guard lk(engineMu_);
  unloadLocked();
}

void SupertonicModel::reload() {
  std::lock_guard lk(engineMu_);
  unloadLocked();
  loadLocked();
}

void SupertonicModel::loadLocked() {
  if (engine_) return;

  // DEBUG (QVAC-20557, DO-NOT-MERGE): the Android GPU force-to-CPU was REMOVED
  // here so a device-farm round can MEASURE Supertonic GPU-vs-CPU correctness on
  // Mali/Adreno (the original block set cfg_.useGpu=false; cfg_.nGpuLayers=0).
  // Measurement scaffolding, NOT a ship change.
#ifdef __ANDROID__
  {
    const bool wantsGpu =
        cfg_.useGpu.value_or(false) ||
        (cfg_.nGpuLayers.has_value() && *cfg_.nGpuLayers != 0);
    if (wantsGpu) {
      QLOG(logger::Priority::WARNING,
           "Supertonic: [QVAC-20557 DO-NOT-MERGE] Android GPU force-to-CPU "
           "disabled — admitting GPU to measure GPU-vs-CPU correctness.");
    }
  }
#endif

  // DEBUG (QVAC-20557, DO-NOT-MERGE): install the ggml log trampoline BEFORE the
  // Engine ctor so the backend-init banner (which backend Mali/Adreno selected +
  // its fp16/coopmat caps) is captured; emit a canary so the device-farm logcat
  // confirms the native log pipe reaches host before trusting the rest.
  installGgmlLogTrampolineOnce();
  emitDeviceDiag("[gpu-diag] canary: native log reaches host (supertonic)");

  try {
    engine_ = std::make_shared<tts_cpp::supertonic::Engine>(toEngineOptions(cfg_));
  } catch (const std::exception& e) {
    engine_.reset();
    throw createTTSError(
        TTSErrorCode::InitializationFailed,
        std::string("SupertonicModel::load: ") + e.what());
  }

  backendName_   = engine_->backend_name();
  backendDevice_ = backendDeviceCode(engine_->backend_device());
  backendId_     = backendIdFromName(backendName_);
}

void SupertonicModel::unloadLocked() {
  engine_.reset();
}

void SupertonicModel::cancel() const {
  cancelRequested_.store(true, std::memory_order_relaxed);
  std::shared_ptr<tts_cpp::supertonic::Engine> e;
  {
    std::lock_guard lk(engineMu_);
    e = engine_;
  }
  if (e) e->cancel();
}

SupertonicModel::Output SupertonicModel::synthesize(const std::string& text) {
  std::shared_ptr<tts_cpp::supertonic::Engine> engine;
  {
    std::lock_guard lk(engineMu_);
    engine = engine_;
  }
  if (!engine) {
    throw createTTSError(TTSErrorCode::InitializationFailed,
                         "SupertonicModel::synthesize: engine not loaded");
  }
  if (cancelRequested_.load(std::memory_order_relaxed)) {
    throw createTTSError(TTSErrorCode::SynthesisFailed,
                         "synthesis cancelled before it started");
  }

  textLength_ = text.size();

  const auto t0 = std::chrono::steady_clock::now();
  tts_cpp::supertonic::SynthesisResult result;
  try {
    result = engine->synthesize(text);
  } catch (const std::exception& e) {
    throw createTTSError(TTSErrorCode::SynthesisFailed,
                         std::string("supertonic.synthesize: ") + e.what());
  }
  const auto t1 = std::chrono::steady_clock::now();

  sampleRate_ = result.sample_rate;
  totalSamples_ = static_cast<int64_t>(result.pcm.size());
  audioDurationMs_ = result.duration_s > 0.0f
      ? result.duration_s * 1000.0
      : (sampleRate_ > 0 ? (static_cast<double>(totalSamples_) * 1000.0 /
                            static_cast<double>(sampleRate_))
                         : 0.0);
  totalTime_ = std::chrono::duration<double>(t1 - t0).count();
  realTimeFactor_ = audioDurationMs_ > 0.0
      ? (totalTime_ * 1000.0) / audioDurationMs_
      : 0.0;
  tokensPerSecond_ = totalTime_ > 0.0
      ? static_cast<double>(textLength_) / totalTime_
      : 0.0;

  return pcmFloatToInt16(result.pcm.data(), result.pcm.size());
}

std::any SupertonicModel::process(const std::any& input) {
  const auto* anyInput = std::any_cast<AnyInput>(&input);
  if (!anyInput) {
    throw StatusError(general_error::InvalidArgument,
                      "SupertonicModel::process: input must be AnyInput");
  }

  bool expected = false;
  if (!jobInProgress_.compare_exchange_strong(expected, true,
                                              std::memory_order_acq_rel)) {
    throw StatusError(general_error::InternalError,
                      "SupertonicModel::process: job already in progress");
  }
  struct InProgressGuard {
    std::atomic_bool& flag;
    ~InProgressGuard() { flag.store(false, std::memory_order_release); }
  } guard{jobInProgress_};

  cancelRequested_.store(false, std::memory_order_relaxed);
  return std::any(synthesize(anyInput->text));
}

qvac_lib_inference_addon_cpp::RuntimeStats SupertonicModel::runtimeStats() const {
  qvac_lib_inference_addon_cpp::RuntimeStats stats;
  stats.emplace_back("totalTime", totalTime_);
  stats.emplace_back("tokensPerSecond", tokensPerSecond_);
  stats.emplace_back("realTimeFactor", realTimeFactor_);
  stats.emplace_back("audioDurationMs", audioDurationMs_);
  stats.emplace_back("totalSamples", totalSamples_);
  stats.emplace_back("backendDevice", static_cast<int64_t>(backendDevice_));
  stats.emplace_back("backendId",     static_cast<int64_t>(backendId_));
  return stats;
}

}
