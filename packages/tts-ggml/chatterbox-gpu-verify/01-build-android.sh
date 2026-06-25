#!/usr/bin/env bash
# Build the arm64-android @qvac/tts-ggml prebuild.
# QVAC-20557 Bug-1 CPU-verify branch: picks up ../ports/ggml-speech (the SVE
# vec_dot fix + TTS_SVE_DOT_UNFIXED toggle + [sve-diag], pinned to the verify
# branch) via vcpkg-configuration.json "overlay-ports". tts-cpp is consumed
# UNCHANGED from the registry (CPU-only run; no Mali/GPU changes needed).
# Requires: bare, bare-make, Android NDK, vcpkg toolchain (auto-detected).
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build-android}"
cd "$PKG"
echo "== bare-make generate ($BUILD_DIR, android/arm64, ANDROID_STL=c++_shared)"
bare-make generate -b "$BUILD_DIR" --platform android --arch arm64 -D ANDROID_STL=c++_shared
echo "== bare-make build"
bare-make build -b "$BUILD_DIR"
echo "== bare-make install"
bare-make install -b "$BUILD_DIR"
echo "OK -> $PKG/prebuilds/android-arm64 (.bare + libqvac-speech-ggml-{opencl,vulkan,cpu*}.so)"
