#pragma once

// Lightweight per-stage wall-clock profiling for the DocTR pipeline steps.
// One log line per inference stage; on Android it always goes to logcat
// (tag "ocr-prof") so device-lab runs can be analysed from the bundled
// logcat artifact, on desktop it is silent unless OCR_DOCTR_PROF is set.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#ifdef __ANDROID__
#include <android/log.h>
#endif

namespace doctr::ggml::pipeline::prof {

using Clock = std::chrono::steady_clock;

inline double msSince(Clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
             Clock::now() - start)
      .count();
}

inline void log(const std::string& msg) {
#ifdef __ANDROID__
  __android_log_print(ANDROID_LOG_INFO, "ocr-prof", "%s", msg.c_str());
#else
  static const bool enabled = std::getenv("OCR_DOCTR_PROF") != nullptr;
  if (enabled) {
    std::fprintf(stderr, "%s\n", msg.c_str());
  }
#endif
}

} // namespace doctr::ggml::pipeline::prof
