#!/usr/bin/env bash
# Stage the prebuilt addon + bare CLI + libc++ + models onto a USB-attached device.
# Required env:
#   BARE_CLI   path to the android-arm64 bare binary
#              (npm pack bare-runtime-android-arm64@<host bare --version> -> package/bin/bare)
#   LIBCXX_SO  path to the NDK aarch64 libc++_shared.so
# Optional env:
#   DEVICE_DIR (default /data/local/tmp/chbx)  MODELS_DIR (default ../models)  SERIAL (adb serial)
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/chbx}"
MODELS_DIR="${MODELS_DIR:-$PKG/models}"
ADB=(adb); [ -n "${SERIAL:-}" ] && ADB=(adb -s "$SERIAL")
: "${BARE_CLI:?set BARE_CLI to the android-arm64 bare binary}"
: "${LIBCXX_SO:?set LIBCXX_SO to the NDK aarch64 libc++_shared.so}"
[ -d "$PKG/prebuilds/android-arm64" ] || { echo "no prebuilds/android-arm64 — run 01-build-android.sh first"; exit 1; }

BUNDLE=/tmp/chbx-bundle.tgz
echo "== bundling $PKG (excludes build dirs / models / host prebuild)"
tar -czf "$BUNDLE" -C "$PKG" \
  --exclude=build-android --exclude=build --exclude=models --exclude=.cache --exclude=.git \
  --exclude=prebuilds/darwin-arm64 --exclude=examples --exclude=benchmarks --exclude=addon --exclude=ports .

echo "== pushing + extracting bundle to $DEVICE_DIR"
"${ADB[@]}" shell "mkdir -p $DEVICE_DIR/out $DEVICE_DIR/models"
"${ADB[@]}" push "$BUNDLE" "$DEVICE_DIR/bundle.tgz"
"${ADB[@]}" shell "cd $DEVICE_DIR && tar xzf bundle.tgz"

echo "== pushing bare CLI + libc++_shared.so"
"${ADB[@]}" push "$BARE_CLI" "$DEVICE_DIR/bare"
"${ADB[@]}" shell "chmod 755 $DEVICE_DIR/bare"
"${ADB[@]}" push "$LIBCXX_SO" "$DEVICE_DIR/libc++_shared.so"

echo "== pushing models from $MODELS_DIR"
# Chatterbox (turbo = first two, mtl = next two) + Supertonic 2 / 3 (q4_0) for the
# batched Mali-GPU verify. Missing files are skipped with a warning, so you only
# need the models for the configs you actually run.
for f in chatterbox-t3-turbo.gguf chatterbox-s3gen.gguf chatterbox-t3-mtl.gguf chatterbox-s3gen-mtl.gguf supertonic2.gguf supertonic3-q4_0.gguf; do
  if [ -f "$MODELS_DIR/$f" ]; then
    "${ADB[@]}" push "$MODELS_DIR/$f" "$DEVICE_DIR/models/"
  else
    echo "   WARN missing $MODELS_DIR/$f (download groups: chatterbox / chatterbox-mtl / supertonic2 / supertonic3)"
  fi
done
echo "OK deployed to $DEVICE_DIR"
