#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# QVAC-20557 — host-side setup (run BEFORE run-on-device.sh).
#
# The colleague ships a THIN bundle (custom prebuild + addon JS + libc++ + a
# minimal package.json + the model-download script). This script fetches the
# PUBLIC pieces so the test can run:
#   1. the `bare` android CLI            — public npm: bare-runtime-android-arm64
#   2. node_modules                      — `npm install` (the @qvac deps are PUBLIC
#                                          on npm; a token is needed only to PUBLISH,
#                                          not to read. bare deps ship all-platform
#                                          prebuilds, so a host install works on-device)
#   3. models/supertonic2.gguf           — `node download-tts-ggml-models.js` pulls it
#                                          from the public QVAC model registry (S3)
# It then tars node_modules for a fast on-device push and prints READY / NOT READY.
#
# Requires: node + npm + internet. (No adb here — that's run-on-device.sh.)
#
# Usage:  bash setup.sh            (re-runnable; skips pieces already present)
#         bash setup.sh --force    (re-fetch node_modules + model)
# Env:    BARE_PKG (default bare-runtime-android-arm64), MODEL_GROUP (default supertonic2)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$SCRIPT_DIR/bundle"
BARE_PKG="${BARE_PKG:-bare-runtime-android-arm64}"
MODEL_GROUP="${MODEL_GROUP:-supertonic2}"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

say()  { printf '\033[1;36m== %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m   OK   %s\033[0m\n' "$*"; }
miss() { printf '\033[1;31m   MISS %s\033[0m\n' "$*"; }
note() { printf '\033[1;33m   ..   %s\033[0m\n' "$*"; }

# ── 0. extract the bundle the colleague sent ────────────────────────────────
if [ ! -d "$BUNDLE_DIR" ] && [ -f "$SCRIPT_DIR/bundle.tgz" ]; then
  say "extracting bundle.tgz"; mkdir -p "$BUNDLE_DIR"; tar xzf "$SCRIPT_DIR/bundle.tgz" -C "$BUNDLE_DIR"
fi
if [ -f "$SCRIPT_DIR/prebuild-update.tgz" ]; then
  say "applying prebuild-update.tgz (newer build under test)"; tar xzf "$SCRIPT_DIR/prebuild-update.tgz" -C "$BUNDLE_DIR"
fi
[ -d "$BUNDLE_DIR" ] || { miss "no ./bundle/ and no bundle.tgz — ask the colleague for bundle.tgz"; exit 1; }

command -v node >/dev/null 2>&1 || { miss "node not found — install Node.js (https://nodejs.org)"; exit 1; }
command -v npm  >/dev/null 2>&1 || { miss "npm not found — install Node.js"; exit 1; }

# ── 1. bare android CLI (public npm) ─────────────────────────────────────────
if [ -f "$BUNDLE_DIR/bare" ]; then
  ok "bare CLI present (shipped)"
else
  say "fetching bare CLI ($BARE_PKG) from public npm"
  TMP="$(mktemp -d)"
  if ( cd "$TMP" && npm pack "$BARE_PKG" >/dev/null 2>&1 ); then
    TGZ="$(ls "$TMP"/*.tgz 2>/dev/null | head -1)"; [ -n "$TGZ" ] && tar xzf "$TGZ" -C "$TMP"
    if [ -f "$TMP/package/bin/bare" ]; then cp "$TMP/package/bin/bare" "$BUNDLE_DIR/bare"; chmod +x "$BUNDLE_DIR/bare"; ok "fetched bare"; fi
  fi
  rm -rf "$TMP"
  [ -f "$BUNDLE_DIR/bare" ] || note "could not fetch bare automatically — ask the colleague for the android-arm64 'bare' CLI"
fi

# ── 2. node_modules via public npm ───────────────────────────────────────────
if [ "$FORCE" = "1" ] || [ ! -d "$BUNDLE_DIR/node_modules" ]; then
  say "npm install (public @qvac deps — no token needed for read)"
  ( cd "$BUNDLE_DIR" && npm install --ignore-scripts --no-audit --no-fund ) || { miss "npm install failed — check internet / npm"; exit 1; }
  ok "node_modules installed"
else
  ok "node_modules present"
fi

# ── 3. model from the public registry ────────────────────────────────────────
if [ "$FORCE" = "1" ] || ! ls "$BUNDLE_DIR"/models/*.gguf >/dev/null 2>&1; then
  say "downloading model group '$MODEL_GROUP' from the public QVAC registry"
  mkdir -p "$BUNDLE_DIR/models"
  ( cd "$BUNDLE_DIR" && node download-tts-ggml-models.js --group "$MODEL_GROUP" --output "$BUNDLE_DIR/models" ) \
    || { miss "model download failed — needs internet; or ask the colleague for supertonic2.gguf"; exit 1; }
  ok "model downloaded"
else
  ok "model present"
fi

# ── 4. tar node_modules for a fast, reliable on-device push ──────────────────
if [ -d "$BUNDLE_DIR/node_modules" ]; then
  if [ "$FORCE" = "1" ] || [ ! -f "$BUNDLE_DIR/node_modules.tgz" ] || [ "$BUNDLE_DIR/node_modules" -nt "$BUNDLE_DIR/node_modules.tgz" ]; then
    say "taring node_modules for device push"
    ( cd "$BUNDLE_DIR" && tar czf node_modules.tgz node_modules )
    ok "node_modules.tgz ready"
  fi
fi

# ── 5. readiness check ───────────────────────────────────────────────────────
say "readiness"
READY=1
chk_f() { if [ -f "$1" ]; then ok "$2"; else miss "$2 — $3"; READY=0; fi; }
chk_g() { if ls $1 >/dev/null 2>&1; then ok "$2"; else miss "$2 — $3"; READY=0; fi; }
chk_f "$BUNDLE_DIR/bare"             "bare CLI"             "public: npm pack $BARE_PKG, or ask the colleague"
chk_f "$BUNDLE_DIR/libc++_shared.so" "libc++_shared.so"    "shipped in the bundle — ask the colleague if absent"
chk_f "$BUNDLE_DIR/index.js"         "addon JS"            "shipped in the bundle — ask the colleague if absent"
chk_f "$BUNDLE_DIR/node_modules.tgz" "node_modules.tgz"    "re-run setup.sh (npm install + tar)"
chk_g "$BUNDLE_DIR/models/*.gguf"    "model (.gguf)"       "re-run setup.sh, or ask the colleague for the .gguf"
chk_g "$BUNDLE_DIR/prebuilds/android-arm64/*.bare" "prebuild (.bare)" "shipped in the bundle — ask the colleague"

echo
if [ "$READY" = "1" ]; then
  ok "READY — connect the phone and run:  bash run-on-device.sh --cpu-baseline"
else
  miss "NOT READY — resolve the MISS items above, then re-run setup.sh"; exit 1
fi
