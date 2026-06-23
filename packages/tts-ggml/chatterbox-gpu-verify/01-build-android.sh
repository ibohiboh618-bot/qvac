#!/usr/bin/env bash
# Build the arm64-android @qvac/tts-ggml prebuild from the local vcpkg overlay
# (../ports/tts-cpp @ 63f25312 + ../ports/ggml-speech @ 25655933).
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
