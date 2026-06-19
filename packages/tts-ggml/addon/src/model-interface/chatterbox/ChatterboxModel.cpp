#include "model-interface/chatterbox/ChatterboxModel.hpp"

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

#include <tts-cpp/chatterbox/engine.h>
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

namespace qvac::ttsggml::chatterbox {

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
// diagnostics vanish (why earlier rounds were blind). Emit STRAIGHT to the
// platform log: __android_log_print lands synchronously in the full device
// logcat artifact (logcat_full.txt). Off-device, stderr keeps local pre-flight
// working.
void emitDeviceDiag(const std::string& line) {
#if defined(__ANDROID__)
  __android_log_print(ANDROID_LOG_INFO, "qvac-chatterbox", "%s", line.c_str());
#else
  std::fputs(line.c_str(), stderr);
  std::fputc('\n', stderr);
  std::fflush(stderr);
#endif
}

// ggml emits log text in fragments that are not necessarily newline-terminated;
// buffer and flush complete lines so backend-init banners, op-support warnings,
// and unsupported-op fallbacks each reach the device log as one clean line.
// Installed via tts_cpp_log_set, which forwards to ggml_log_set.
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

// DEBUG (QVAC-20557, DO-NOT-MERGE): per-stage f32 dumps + token-pinning are
// keyed off one dir, $TTS_CPP_GPU_DUMP_DIR (set by the gpu-smoke test to a
// device path the device-farm pulls). Empty/unset = no dump, no pin.
std::string gpuDumpDir() {
  const char* d = std::getenv("TTS_CPP_GPU_DUMP_DIR");
  return (d && *d) ? std::string(d) : std::string();
}

// DEBUG (QVAC-20557, DO-NOT-MERGE): read pre-captured T3 tokens for pinning. The
// gpu-smoke test runs the GPU model first (which writes chatterbox_speech_tokens.i32
// — see synthesize()), copies it to chatterbox_pinned_tokens.i32, then loads the
// CPU model; this reads that file so CPU S3Gen decodes the SAME tokens the GPU run
// used. Absent file = normal stochastic T3 (the GPU/first run). int32 little-endian.
std::vector<int32_t> readPinnedTokens(const std::string& dir) {
  std::vector<int32_t> toks;
  if (dir.empty()) return toks;
  const std::string path = dir + "/chatterbox_pinned_tokens.i32";
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return toks;
  std::fseek(f, 0, SEEK_END);
  long bytes = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (bytes > 0) {
    toks.resize(static_cast<size_t>(bytes) / sizeof(int32_t));
    if (std::fread(toks.data(), sizeof(int32_t), toks.size(), f) != toks.size()) toks.clear();
  }
  std::fclose(f);
  return toks;
}

// DEBUG (QVAC-20557, DO-NOT-MERGE): persist the tokens this synth used so the
// follow-up CPU run can pin them (see readPinnedTokens).
void writeSpeechTokens(const std::string& dir, const std::vector<int32_t>& toks) {
  if (dir.empty() || toks.empty()) return;
  const std::string path = dir + "/chatterbox_speech_tokens.i32";
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return;
  std::fwrite(toks.data(), sizeof(int32_t), toks.size(), f);
  std::fclose(f);
}

tts_cpp::chatterbox::EngineOptions toEngineOptions(const ChatterboxConfig& cfg) {
  tts_cpp::chatterbox::EngineOptions opts;
  opts.t3_gguf_path    = cfg.t3ModelPath;
  opts.s3gen_gguf_path = cfg.s3genModelPath;
  opts.reference_audio = cfg.referenceAudio;
  opts.voice_dir       = cfg.voiceDir;
  if (!cfg.language.empty()) opts.language = cfg.language;
  if (cfg.seed.has_value())    opts.seed         = *cfg.seed;
  if (cfg.threads.has_value()) opts.n_threads    = *cfg.threads;
  if (cfg.nGpuLayers.has_value()) {
    opts.n_gpu_layers = *cfg.nGpuLayers;
  } else if (cfg.useGpu.has_value()) {
    // Explicit useGpu must produce an explicit n_gpu_layers so we don't
    // depend on the tts-cpp library default flipping out from under us
    // (see also: gpu-smoke.test.js asserts backendDevice from this).
    opts.n_gpu_layers = *cfg.useGpu ? 99 : 0;
  }
  if (cfg.streamChunkTokens.has_value())      opts.stream_chunk_tokens       = *cfg.streamChunkTokens;
  if (cfg.streamFirstChunkTokens.has_value()) opts.stream_first_chunk_tokens = *cfg.streamFirstChunkTokens;
  if (cfg.streamCfmSteps.has_value())         opts.stream_cfm_steps          = *cfg.streamCfmSteps;

  // Compose the actual backends-scan directory from the host-provided
  // prebuilds root plus the cmake-bare per-target subdir
  // (BACKENDS_SUBDIR, e.g. `android-arm64/qvac__tts-ggml`). Mirrors
  // the exact shape qvac/packages/transcription-parakeet uses in
  // ParakeetModel.cpp + qvac/packages/llm-llamacpp uses in
  // LlamaLazyInitializeBackend.cpp so a host that already passes
  // `path.join(__dirname, 'prebuilds')` gets identical resolution
  // semantics across the three addons. Empty `backendsDir` -> leave
  // `opts.backends_dir` empty so tts-cpp falls back to ggml's
  // compile-time default search path (`ggml_backend_load_all()`
  // rather than `..._from_path()`).
  if (!cfg.backendsDir.empty()) {
    std::filesystem::path backendsDirPath(cfg.backendsDir);
#ifdef BACKENDS_SUBDIR
    backendsDirPath =
        (backendsDirPath / std::filesystem::path(BACKENDS_SUBDIR)).lexically_normal();
#endif
    opts.backends_dir = backendsDirPath.string();
  }
  // Forwarded as-is. Empty -> leave $GGML_OPENCL_CACHE_DIR alone
  // (the env-set-by-host path still wins). Only consumed on Android
  // by `tts_cpp::detail::set_opencl_cache_dir()`; other platforms
  // ignore it. Process-singleton scoped: a second Engine ctor with
  // a different value is silently ignored on the tts-cpp side
  // because ggml-opencl only reads the env var once at first init.
  opts.opencl_cache_dir = cfg.openclCacheDir;

  // Multilingual text preprocessing dictionaries.  Only consumed by the
  // multilingual T3 variant inside tts-cpp (Turbo ignores them); leave
  // the EngineOptions fields empty when the host didn't resolve a path
  // so tts-cpp keeps its character-level fallback.
  if (!cfg.mecabDictPath.empty())  opts.mecab_dict_path  = cfg.mecabDictPath;
  if (!cfg.cangjieTsvPath.empty()) opts.cangjie_tsv_path = cfg.cangjieTsvPath;

  // DEBUG (QVAC-20557, DO-NOT-MERGE): route the engine's per-stage [gpu-diag]
  // lines (input_embed/encoder_mu/cfm_mel/f0/stft/hift_wav) to the device log
  // bridge so they reach logcat_full.txt; when $TTS_CPP_GPU_DUMP_DIR is set,
  // also dump each S3Gen stage's raw f32 there for GPU-vs-CPU correlation; and
  // if a pinned-tokens file is present, decode those tokens instead of running
  // stochastic T3 (so a CPU run matches the GPU run's tokens).
  opts.diag_sink     = &emitDeviceDiag;
  opts.diag_dump_dir = gpuDumpDir();
  opts.test_pinned_tokens = readPinnedTokens(opts.diag_dump_dir);
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

std::vector<int16_t> pcmFloatToInt16(const std::vector<float>& pcm) {
  return pcmFloatToInt16(pcm.data(), pcm.size());
}

} // namespace

ChatterboxModel::ChatterboxModel(ChatterboxConfig config)
    : cfg_(std::move(config)) {
  validateConfig(cfg_);
  // Constructor deliberately does NOT call load(): GGUF parsing is the
  // multi-hundred-MB step (ggml_backend_alloc_ctx_tensors + voice-
  // conditioning bake) and used to stall the Bare event loop because
  // qvac_lib_inference_addon_cpp::JsInterface::createInstance is
  // synchronous.  AddonCpp::activate() (driven by the JsAsyncTask::run
  // wrapper in addon_js::activate) now calls
  // waitForLoadInitialization() on a worker thread, which delegates to
  // load() lazily.  Direct C++ callers (and the unit-test suite in
  // addon/tests/) can still invoke load() explicitly when they want
  // synchronous semantics.
}

ChatterboxModel::~ChatterboxModel() noexcept = default;

void ChatterboxModel::validateConfig(const ChatterboxConfig& cfg) {
  if (cfg.useGpu.has_value() && cfg.nGpuLayers.has_value()) {
    const bool wantsGpu = *cfg.useGpu;
    const int  layers   = *cfg.nGpuLayers;
    // `layers != 0` (rather than `layers > 0`) so a llama.cpp-style
    // sentinel like nGpuLayers=-1 ("offload all layers") is treated as
    // "wants GPU" and doesn't falsely pass through against useGPU:true.
    const bool layersWantGpu = layers != 0;
    if (wantsGpu != layersWantGpu) {
      throw StatusError(
          general_error::InvalidArgument,
          std::string("ChatterboxModel: useGPU=") +
              (wantsGpu ? "true" : "false") +
              " conflicts with nGpuLayers=" + std::to_string(layers) +
              ". Either drop one of the two, or make them agree "
              "(useGPU:true + nGpuLayers!=0, or useGPU:false + nGpuLayers=0).");
    }
  }
  if (cfg.t3ModelPath.empty()) {
    throw StatusError(general_error::InvalidArgument, "t3ModelPath is required");
  }
  if (cfg.s3genModelPath.empty()) {
    throw StatusError(general_error::InvalidArgument, "s3genModelPath is required");
  }
  if (!std::filesystem::exists(cfg.t3ModelPath)) {
    throw createTTSError(TTSErrorCode::ModelFileNotFound, "t3 model not found: " + cfg.t3ModelPath);
  }
  if (!std::filesystem::exists(cfg.s3genModelPath)) {
    throw createTTSError(TTSErrorCode::ModelFileNotFound, "s3gen model not found: " + cfg.s3genModelPath);
  }
  if (!cfg.referenceAudio.empty() &&
      !std::filesystem::exists(cfg.referenceAudio)) {
    throw createTTSError(TTSErrorCode::ModelFileNotFound, "reference audio not found: " + cfg.referenceAudio);
  }
  if (!cfg.voiceDir.empty()) {
    if (!std::filesystem::exists(cfg.voiceDir)) {
      throw createTTSError(TTSErrorCode::ModelFileNotFound, "voice dir not found: " + cfg.voiceDir);
    }
    if (!std::filesystem::is_directory(cfg.voiceDir)) {
      throw StatusError(
          general_error::InvalidArgument,
          "voiceDir path exists but is not a directory: " + cfg.voiceDir);
    }
  }
  // No JS-side allow-list of language codes: the active GGUF variant
  // (turbo English vs multilingual) determines what's supported, and
  // tts_cpp::chatterbox::Engine throws a clear runtime error when the
  // requested language doesn't match the loaded variant.  Forcing a
  // hard-coded "en"-only check here would leak the turbo-variant
  // assumption into the addon and silently reject the multilingual
  // GGUFs (chatterbox-t3-mtl + chatterbox-s3gen-mtl) the converter
  // pipeline already produces.
}

void ChatterboxModel::load() {
  std::lock_guard lk(engineMu_);
  loadLocked();
}

void ChatterboxModel::unload() {
  std::lock_guard lk(engineMu_);
  unloadLocked();
}

void ChatterboxModel::reload() {
  std::lock_guard lk(engineMu_);
  unloadLocked();
  loadLocked();
}

void ChatterboxModel::loadLocked() {
  if (engine_) return;

  // DEBUG (QVAC-20557, DO-NOT-MERGE): the Android GPU force-to-CPU was REMOVED
  // here so a device-farm round can MEASURE Chatterbox GPU-vs-CPU correctness on
  // Mali/Adreno (the original block set cfg_.useGpu=false; cfg_.nGpuLayers=0).
  // Measurement scaffolding, NOT a ship change.
#ifdef __ANDROID__
  {
    const bool wantsGpu =
        cfg_.useGpu.value_or(false) ||
        (cfg_.nGpuLayers.has_value() && *cfg_.nGpuLayers != 0);
    if (wantsGpu) {
      QLOG(logger::Priority::WARNING,
           "Chatterbox: [QVAC-20557 DO-NOT-MERGE] Android GPU force-to-CPU "
           "disabled — admitting GPU to measure GPU-vs-CPU correctness.");
    }
  }
#endif

  // DEBUG (QVAC-20557, DO-NOT-MERGE): install the ggml log trampoline BEFORE the
  // Engine ctor so the backend-init banner (which backend Mali/Adreno selected +
  // its fp16/coopmat caps) is captured; emit a canary so the device-farm logcat
  // confirms the native log pipe reaches host before trusting the rest.
  installGgmlLogTrampolineOnce();
  emitDeviceDiag("[gpu-diag] canary: native log reaches host (chatterbox)");

  try {
    engine_ = std::make_shared<tts_cpp::chatterbox::Engine>(toEngineOptions(cfg_));
  } catch (const std::exception& e) {
    engine_.reset();
    throw createTTSError(
        TTSErrorCode::InitializationFailed,
        std::string("ChatterboxModel::load: ") + e.what());
  }

  backendName_   = engine_->backend_name();
  backendDevice_ = backendDeviceCode(engine_->backend_device());
  backendId_     = backendIdFromName(backendName_);
}

void ChatterboxModel::unloadLocked() {
  engine_.reset();
}

void ChatterboxModel::cancel() const {
  cancelRequested_.store(true, std::memory_order_relaxed);
  // Grab a local copy of engine_ under the lock so we can invoke
  // cancel() safely even if another thread calls unload()/reload() in
  // parallel.  The Engine itself is responsible for making cancel()
  // thread-safe against its in-flight synthesize().
  std::shared_ptr<tts_cpp::chatterbox::Engine> e;
  {
    std::lock_guard lk(engineMu_);
    e = engine_;
  }
  if (e) e->cancel();
}

ChatterboxModel::SynthesizeResult ChatterboxModel::synthesize(
    const std::string& text, const ChunkCallback& chunkCallback) {
  // Capture the engine under the lock; keep it alive for the duration
  // of synthesize() via the local `engine` shared_ptr even if reload()
  // concurrently swaps a new one in.  Reload's new engine takes effect
  // on the NEXT synthesize call.
  std::shared_ptr<tts_cpp::chatterbox::Engine> engine;
  {
    std::lock_guard lk(engineMu_);
    engine = engine_;
  }
  if (!engine) {
    throw createTTSError(TTSErrorCode::ModelNotLoaded,
                         "ChatterboxModel::synthesize: engine not loaded");
  }
  if (cancelRequested_.load(std::memory_order_relaxed)) {
    throw createTTSError(TTSErrorCode::SynthesisFailed,
                         "synthesis cancelled before it started");
  }

  // Snapshot the streaming decision against the engine we're actually
  // about to call, BEFORE process() needs it.  Reading engine_ /
  // engine->options() outside the lock from process() would race with
  // reload() swapping a new engine in; pinning the decision here keeps
  // the read tied to the local `engine` shared_ptr for the call's
  // lifetime.
  const bool wasStreaming =
      static_cast<bool>(chunkCallback) &&
      engine->options().stream_chunk_tokens > 0;

  const auto tStart = std::chrono::steady_clock::now();

  tts_cpp::chatterbox::SynthesisResult result;
  try {
    if (wasStreaming) {
      result = engine->synthesize(
          text,
          [&chunkCallback](const float* pcm, std::size_t samples,
                           int chunkIndex, bool isLast) {
            chunkCallback(pcmFloatToInt16(pcm, samples), chunkIndex, isLast);
          });
    } else {
      result = engine->synthesize(text);
    }
  } catch (const std::exception& e) {
    throw createTTSError(TTSErrorCode::SynthesisFailed,
                         std::string("engine.synthesize: ") + e.what());
  }

  // DEBUG (QVAC-20557, DO-NOT-MERGE): persist the T3 tokens this synth used so a
  // follow-up CPU run can pin them (readPinnedTokens) for a token-matched
  // GPU-vs-CPU per-stage comparison. No-op when $TTS_CPP_GPU_DUMP_DIR is unset.
  writeSpeechTokens(gpuDumpDir(), result.speech_tokens);

  std::vector<int16_t> pcm = pcmFloatToInt16(result.pcm);

  const auto tEnd = std::chrono::steady_clock::now();
  const double elapsedSec =
      std::chrono::duration<double>(tEnd - tStart).count();

  totalTime_ = elapsedSec;
  totalSamples_ = static_cast<int64_t>(pcm.size());
  audioDurationMs_ = result.sample_rate > 0
      ? (static_cast<double>(pcm.size()) * 1000.0 /
         static_cast<double>(result.sample_rate))
      : 0.0;
  realTimeFactor_ =
      audioDurationMs_ > 0 ? (elapsedSec * 1000.0) / audioDurationMs_ : 0.0;
  textLength_ = text.size();
  tokensPerSecond_ =
      elapsedSec > 0 ? static_cast<double>(textLength_) / elapsedSec : 0.0;

  return {std::move(pcm), wasStreaming};
}

std::any ChatterboxModel::process(const std::any& input) {
  const auto* anyInput = std::any_cast<AnyInput>(&input);
  if (anyInput == nullptr) {
    throw StatusError(
        general_error::InvalidArgument,
        "ChatterboxModel::process: expected AnyInput (text + chunkCallback)");
  }
  if (anyInput->text.empty()) {
    throw StatusError(
        general_error::InvalidArgument, "ChatterboxModel::process: empty text");
  }

  // Serialize concurrent process() calls.  The outer JobRunner already
  // queues jobs sequentially, but a direct C++ caller (or a future
  // pipeline that bypasses JobRunner) could still overlap — fail fast
  // with a clear error instead of data-racing on engine_ state.
  bool expected = false;
  if (!jobInProgress_.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    throw StatusError(
        general_error::InvalidArgument,
        "ChatterboxModel::process: another synthesis job is already in progress");
  }
  struct InProgressGuard {
    std::atomic_bool& flag;
    ~InProgressGuard() { flag.store(false, std::memory_order_release); }
  } guard{jobInProgress_};

  cancelRequested_.store(false, std::memory_order_relaxed);
  auto result = synthesize(anyInput->text, anyInput->chunkCallback);
  // Streaming mode: chunks have already been published via chunkCallback
  // → OutputQueue.  Returning the concatenated PCM here would cause a
  // duplicate final `outputArray` event after all the chunks.  Return an
  // empty std::any so no output handler matches — JobRunner still emits
  // JobEnded with runtimeStats on its own.  We trust the wasStreaming
  // bit captured under the engine lock inside synthesize() rather than
  // re-reading engine_ here (which would race with a concurrent
  // reload()).
  if (result.wasStreaming) return {};
  return std::any(std::move(result.pcm));
}

qvac_lib_inference_addon_cpp::RuntimeStats ChatterboxModel::runtimeStats() const {
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

} // namespace qvac::ttsggml::chatterbox
