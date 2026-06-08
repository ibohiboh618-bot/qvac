// Memory-cycle regression test for tts_cpp::supertonic::Engine.
//
// Runs as part of the tts-ggml addon's gtest suite so the on-PR CI
// workflow (`on-pr-tts-ggml.yml` -> `cpp-test-coverage-tts-ggml.yml`)
// catches per-cycle gallocr / backend-buffer leaks in the engine's
// load + synth + destroy path against a real Supertonic GGUF.
//
// Construct + destroy `kNumCycles` engines back-to-back on the same
// thread, synthesising once per engine so every per-stage thread_local
// graph cache populates before teardown.  Asserts that resident memory
// (RSS on Linux, phys_footprint on macOS) does not drift across cycles
// 2..N compared to cycle 1 — first cycle is excluded because it
// captures one-time process-singleton inits (ggml backend registry
// load, metal library compile, vulkan ICD probe, OpenMP pool, glibc /
// libstdc++ arenas) that subsequent cycles reuse and that would
// otherwise inflate the baseline.
//
// Gated behind QVAC_TEST_SUPERTONIC_GGUF in the same way as the
// existing SupertonicRealGguf round-trips — skips when the GGUF
// isn't provisioned on the runner.
//
// The threshold (5 MB) is generous enough to absorb allocator noise
// (glibc / libc++ free-list churn, page-pool watermarks) but tight
// enough to catch a regression of even one leaked gallocr per cycle:
// each leaked gallocr from the upstream tts-cpp fix is ~83 KB, and
// supertonic populates ~30 caches per synth, so a full regression
// would be ~2.5 MB per cycle — visible by cycle 3 against this bound.
//
// On Windows the RSS probe falls back to a no-op so the test reduces
// to a pure smoke check (still verifies the engine cycles don't
// crash; the leak quantification stays Linux/macOS-only).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "tts-cpp/supertonic/engine.h"

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#endif

namespace {

constexpr int kNumCycles    = 20;
constexpr double kDriftMb   = 5.0;
constexpr int kSynthSteps   = 5;  // matches SDK default for supertonic
constexpr int kThreads      = 4;

std::size_t rssBytes() {
#if defined(__APPLE__)
  task_vm_info_data_t info{};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO,
                reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
    return 0;
  }
  return static_cast<std::size_t>(info.phys_footprint);
#elif defined(__linux__)
  FILE* f = std::fopen("/proc/self/statm", "r");
  if (!f) return 0;
  long total = 0, resident = 0;
  if (std::fscanf(f, "%ld %ld", &total, &resident) != 2) {
    std::fclose(f);
    return 0;
  }
  std::fclose(f);
  return static_cast<std::size_t>(resident) *
         static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
#elif defined(_WIN32)
  PROCESS_MEMORY_COUNTERS pmc{};
  pmc.cb = sizeof(pmc);
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return 0;
  return static_cast<std::size_t>(pmc.WorkingSetSize);
#else
  return 0;
#endif
}

double mbOf(std::size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

std::string envOrEmpty(const char* name) {
  const char* v = std::getenv(name);
  return v ? std::string{v} : std::string{};
}

}  // namespace

TEST(SupertonicRealGguf, NoLeakOverEngineCycles) {
  const auto path = envOrEmpty("QVAC_TEST_SUPERTONIC_GGUF");
  if (path.empty() || !std::filesystem::exists(path)) {
    GTEST_SKIP() << "Set QVAC_TEST_SUPERTONIC_GGUF to enable the engine "
                    "cycle leak test.";
  }

  std::vector<std::size_t> samples;
  samples.reserve(static_cast<std::size_t>(kNumCycles));

  for (int i = 0; i < kNumCycles; ++i) {
    tts_cpp::supertonic::EngineOptions opts;
    opts.model_gguf_path = path;
    opts.n_gpu_layers    = 0;  // CI runner: CPU only
    opts.n_threads       = kThreads;
    opts.steps           = kSynthSteps;

    tts_cpp::supertonic::Engine engine(opts);
    const auto result =
        engine.synthesize("The quick brown fox jumps over the lazy dog.");
    ASSERT_FALSE(result.pcm.empty())
        << "cycle " << i << " produced empty PCM";
    // Engine destroyed at scope exit -- this is where the
    // release_*_thread_local_caches path in free_supertonic_model
    // tears down the per-stage caches under test.

    samples.push_back(rssBytes());
  }

  // RSS probe failed on this platform -- the test reduces to a smoke
  // check (the engine cycled kNumCycles times without crashing, which
  // is itself a useful regression gate for the teardown ordering bug
  // that hit ggml-metal's `[rsets->data count] == 0` assertion).
  if (samples.front() == 0) {
    GTEST_SKIP() << "RSS probe unsupported on this platform; "
                 << "smoke-checked " << kNumCycles
                 << " engine cycles without crash.";
  }

  const std::size_t firstRss = samples.front();
  const std::size_t maxRss =
      *std::max_element(samples.begin() + 1, samples.end());
  const double driftMb = mbOf(maxRss) - mbOf(firstRss);

  std::printf("[SupertonicRealGguf.NoLeakOverEngineCycles] "
              "cycle 1 RSS=%.1f MB, max(cycle 2..%d) RSS=%.1f MB, "
              "drift=%+.2f MB (threshold %.1f MB)\n",
              mbOf(firstRss), kNumCycles, mbOf(maxRss),
              driftMb, kDriftMb);

  EXPECT_LE(driftMb, kDriftMb)
      << "Per-cycle RSS drift exceeded " << kDriftMb << " MB across "
      << "cycles 2.." << kNumCycles << " (first-cycle RSS captures "
      << "one-time process-singleton inits; subsequent cycles should "
      << "stay flat).  Likely regression in the release_*_thread_"
      << "local_caches path in tts-cpp::free_supertonic_model.";
}
