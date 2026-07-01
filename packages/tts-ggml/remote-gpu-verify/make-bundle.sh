#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# QVAC-20557 — bundle builder (REQUESTER side; run on the build machine).
#
# Builds a THIN bundle.tgz containing only the pieces a remote tester CANNOT
# fetch publicly:
#   * prebuilds/      — the custom build under test (.bare + backend .so)
#   * libc++_shared.so — must match the NDK the addon was built with
#   * the addon JS     — index.js / lib / binding.js / tts.js / addonLogging.js
#   * package.json     — a MINIMAL dep manifest (from _bundle-template/)
#   * download-tts-ggml-models.js — the public model fetcher
#
# It deliberately does NOT ship node_modules or the model: setup.sh on the
# tester's side gets those from PUBLIC sources (npm install — @qvac packages are
# public on npm, token is push-only; and the gguf via @qvac/registry-client S3).
# The `bare` android CLI is also fetched by setup.sh (public npm package).
#
# Usage:
#   bash make-bundle.sh                 # thin bundle.tgz
#   bash make-bundle.sh --prebuild-only # small prebuild-update.tgz (per fix round)
#
# Env: PKG_DIR (default: parent of this folder), STAGE_DIR (on-device staging to
#      pull libc++ from, default /data/local/tmp/ttsg), NDK_LIBCXX (a local
#      libc++_shared.so path to use instead of pulling), ADB_SERIAL.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="${PKG_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
STAGE_DIR="${STAGE_DIR:-/data/local/tmp/ttsg}"
OUT_BUNDLE="$SCRIPT_DIR/bundle"
ADB=(adb)
[ -n "${ADB_SERIAL:-}" ] && ADB=(adb -s "$ADB_SERIAL")

PREBUILD_ONLY=0
[ "${1:-}" = "--prebuild-only" ] && PREBUILD_ONLY=1

say() { printf '\033[1;36m== %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

[ -d "$PKG_DIR/prebuilds/android-arm64" ] || die "no prebuilds at $PKG_DIR/prebuilds/android-arm64 — run bare-make generate/build/install first"

mkdir -p "$OUT_BUNDLE"

# ── prebuild (always — this is what changes each round) ──────────────────────
say "copying freshly-built prebuilds/"
rm -rf "$OUT_BUNDLE/prebuilds"
mkdir -p "$OUT_BUNDLE/prebuilds"
cp -R "$PKG_DIR/prebuilds/android-arm64" "$OUT_BUNDLE/prebuilds/"

# ── per-op tester (test-backend-ops; custom build, not publicly fetchable) ───
TBO_BIN="${TBO_BIN:-/Users/pratiknarola/workstuff/qvac-ext-ggml/build-android-tbo/bin/test-backend-ops}"
if [ -f "$TBO_BIN" ]; then
  say "copying test-backend-ops (per-op Vulkan-vs-CPU tester)"
  mkdir -p "$OUT_BUNDLE/tools"; cp "$TBO_BIN" "$OUT_BUNDLE/tools/test-backend-ops"
else
  printf '\033[1;33m   !! no test-backend-ops at %s — bundle will ship without the per-op tester\033[0m\n' "$TBO_BIN"
fi

if [ "$PREBUILD_ONLY" = "1" ]; then
  say "packing prebuild-update.tgz (prebuild + test-backend-ops)"
  tar czf "$SCRIPT_DIR/prebuild-update.tgz" -C "$OUT_BUNDLE" prebuilds $( [ -d "$OUT_BUNDLE/tools" ] && echo tools )
  echo "wrote $SCRIPT_DIR/prebuild-update.tgz — tester applies it with: tar xzf prebuild-update.tgz -C bundle/  (or just run setup.sh)"
  exit 0
fi

# ── addon JS wrapper (small; matches the shipped prebuild) ───────────────────
say "copying addon JS"
for f in index.js binding.js tts.js addonLogging.js; do
  [ -f "$PKG_DIR/$f" ] && cp "$PKG_DIR/$f" "$OUT_BUNDLE/$f"
done
[ -d "$PKG_DIR/lib" ] && { rm -rf "$OUT_BUNDLE/lib"; cp -R "$PKG_DIR/lib" "$OUT_BUNDLE/lib"; }

# ── Chatterbox speaker reference (jfk.wav) — small, ship it ───────────────────
# Chatterbox clones the voice from this reference wav; the tester can't fetch it
# from the public registry, so it travels in the (still-thin) bundle.
say "copying jfk.wav (Chatterbox speaker reference)"
JFK_SRC="${JFK_WAV:-$PKG_DIR/test/reference-audio/jfk.wav}"
[ -f "$JFK_SRC" ] && cp "$JFK_SRC" "$OUT_BUNDLE/jfk.wav" \
  || die "missing reference audio at $JFK_SRC — set JFK_WAV to a mono wav"

# ── minimal dep manifest + the public model fetcher ──────────────────────────
say "adding minimal package.json + model download script"
cp "$SCRIPT_DIR/_bundle-template/package.json" "$OUT_BUNDLE/package.json"
[ -f "$PKG_DIR/scripts/download-tts-ggml-models.js" ] \
  && cp "$PKG_DIR/scripts/download-tts-ggml-models.js" "$OUT_BUNDLE/download-tts-ggml-models.js" \
  || die "missing $PKG_DIR/scripts/download-tts-ggml-models.js"

# ── libc++_shared.so (NDK-matched; not publicly fetchable → ship it) ─────────
say "providing libc++_shared.so"
if [ -n "${NDK_LIBCXX:-}" ]; then
  cp "$NDK_LIBCXX" "$OUT_BUNDLE/libc++_shared.so"
else
  "${ADB[@]}" pull "$STAGE_DIR/libc++_shared.so" "$OUT_BUNDLE/libc++_shared.so" >/dev/null 2>&1 \
    || die "could not pull libc++_shared.so from $STAGE_DIR — set NDK_LIBCXX to the NDK aarch64 libc++_shared.so"
fi

# ── pack ─────────────────────────────────────────────────────────────────────
# Exclude the publicly-fetchable pieces even if a prior setup.sh left them in
# bundle/ — the shipped bundle.tgz must stay THIN (the tester re-fetches these).
say "packing bundle.tgz (thin — excludes node_modules/model/bare; the tester fetches those)"
tar czf "$SCRIPT_DIR/bundle.tgz" -C "$OUT_BUNDLE" \
  --exclude=node_modules --exclude=node_modules.tgz --exclude=models --exclude=bare .
SZ="$(du -h "$SCRIPT_DIR/bundle.tgz" | cut -f1)"
echo "wrote $SCRIPT_DIR/bundle.tgz ($SZ)"
echo "Send these to the tester:"
echo "  bundle.tgz  INSTRUCTIONS.md  REFERENCE-what-to-test.md"
echo "  setup.sh  run-on-device.sh  run-backend-ops.sh  package-results.sh  test-tts-gpu.js"
