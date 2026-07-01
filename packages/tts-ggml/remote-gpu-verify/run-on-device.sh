#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# QVAC-20557 — remote Supertonic GPU verification harness (Mali-Vulkan).
#
# Stages a prebuilt @qvac/tts-ggml addon onto a USB-attached Android device, runs
# the Supertonic GPU test, and collects the decisive logs into a per-iteration
# folder under results/:
#   results/<iter>/gpu-result.out    — JSON: backend, rms, NaN check
#   results/<iter>/gpu-gpudiag.log   — the native [gpu-diag] dprobe_* trisection
#   results/<iter>/full-logcat.txt   — full device logcat for the run (crash triage)
#   results/<iter>/device-info.txt   — model + GPU renderer (must be Mali-G715)
#   results/<iter>/SUMMARY.txt       — one-screen verdict
# Optionally also runs a CPU pass for a known-good rms baseline.
#
# Everything on the device lives under /data/local/tmp — no personal data is read
# or modified, nothing is installed as an app.
#
# Usage:
#   bash run-on-device.sh                          # auto-numbered iteration, GPU pass
#   bash run-on-device.sh --cpu-baseline           # also run a CPU pass for comparison
#   bash run-on-device.sh --text "Some sentence."  # vary the synthesized input
#   bash run-on-device.sh --label probe-longtext   # name this iteration's folder
#   bash run-on-device.sh --full-stage             # force re-push the whole base bundle
#   bash run-on-device.sh --serial XXXX            # pick a device when several attached
#
# Requires: adb on PATH, a populated ./bundle/ (provided by the requester).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$SCRIPT_DIR/bundle"
RESULTS_ROOT="$SCRIPT_DIR/results"
DEVICE_DIR="/data/local/tmp/qvac-ttsg-verify"
TEST_JS="test-supertonic-gpu.js"

CPU_BASELINE=0
FULL_STAGE=0
ADB_SERIAL=""
LABEL=""
TEXT_OVERRIDE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --cpu-baseline) CPU_BASELINE=1 ;;
    --full-stage)   FULL_STAGE=1 ;;
    --serial)       ADB_SERIAL="$2"; shift ;;
    --label)        LABEL="$2"; shift ;;
    --text)         TEXT_OVERRIDE="$2"; shift ;;
    -h|--help)      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

ADB=(adb)
[ -n "$ADB_SERIAL" ] && ADB=(adb -s "$ADB_SERIAL")

say()  { printf '\033[1;36m== %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m   OK %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m   !! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

# ── 0. preflight ─────────────────────────────────────────────────────────────
command -v adb >/dev/null 2>&1 || die "adb not found on PATH. Install Android platform-tools."
[ -d "$BUNDLE_DIR" ] || die "missing ./bundle/ — extract the bundle.tgz the requester provided into $BUNDLE_DIR"

say "checking device"
STATE="$("${ADB[@]}" get-state 2>/dev/null || true)"
# IMPORTANT for the operating agent: a non-'device' state usually means a CABLE /
# authorization / USB-debugging problem, NOT a code problem. STOP and ask the
# human to check the cable + the 'Allow USB debugging' prompt before retrying.
[ "$STATE" = "device" ] || die "no authorized device (adb get-state='$STATE'). Likely a cable / USB-debugging / 'Allow' prompt issue — ask the human, do not retry blindly."

# pick / create this iteration's output folder
if [ -z "$LABEL" ]; then
  n=1; while [ -d "$RESULTS_ROOT/iter$n" ]; do n=$((n+1)); done
  LABEL="iter$n"
fi
OUTDIR="$RESULTS_ROOT/$LABEL"
mkdir -p "$OUTDIR"
say "iteration: $LABEL  ->  $OUTDIR"

MODEL="$("${ADB[@]}" shell getprop ro.product.model | tr -d '\r')"
ANDROID="$("${ADB[@]}" shell getprop ro.build.version.release | tr -d '\r')"
RENDERER="$("${ADB[@]}" shell dumpsys SurfaceFlinger 2>/dev/null | grep -iE 'GLES:|renderer' | head -1 | tr -d '\r' || true)"
{
  echo "model           : $MODEL"
  echo "android         : $ANDROID"
  echo "gpu (SurfaceFlinger): $RENDERER"
} | tee "$OUTDIR/device-info.txt"

if echo "$RENDERER $MODEL" | grep -iq 'mali'; then
  ok "Mali GPU detected — this is the target."
else
  warn "Could not confirm a Mali GPU. This test targets a Pixel 9 / Mali-G715."
  warn "Run anyway and report the GPU — a non-Mali result is still informative."
fi

# ── 1. stage ─────────────────────────────────────────────────────────────────
# The base bundle (bare CLI, libc++, models, node_modules, addon JS) is pushed
# once. The prebuild (.bare + backend .so) is re-pushed every run because that is
# the part the requester changes between fix rounds.
need_full_stage() {
  [ "$FULL_STAGE" = "1" ] && return 0
  "${ADB[@]}" shell "[ -f $DEVICE_DIR/bare ] && [ -d $DEVICE_DIR/node_modules ] && [ -f $DEVICE_DIR/index.js ]" >/dev/null 2>&1 && return 1
  return 0
}

"${ADB[@]}" shell "mkdir -p $DEVICE_DIR" >/dev/null

if need_full_stage; then
  say "staging base bundle (one-time; ~minutes)"
  [ -f "$BUNDLE_DIR/bare" ]             || die "bundle missing 'bare' (android-arm64 bare CLI)"
  [ -f "$BUNDLE_DIR/libc++_shared.so" ] || die "bundle missing 'libc++_shared.so'"
  [ -f "$BUNDLE_DIR/index.js" ]         || die "bundle missing the addon JS (index.js etc.)"

  "${ADB[@]}" push "$BUNDLE_DIR/bare" "$DEVICE_DIR/bare" >/dev/null
  "${ADB[@]}" shell "chmod 755 $DEVICE_DIR/bare"
  "${ADB[@]}" push "$BUNDLE_DIR/libc++_shared.so" "$DEVICE_DIR/libc++_shared.so" >/dev/null
  for f in index.js binding.js package.json tts.js addonLogging.js; do
    [ -f "$BUNDLE_DIR/$f" ] && "${ADB[@]}" push "$BUNDLE_DIR/$f" "$DEVICE_DIR/$f" >/dev/null
  done
  [ -d "$BUNDLE_DIR/lib" ] && "${ADB[@]}" push "$BUNDLE_DIR/lib" "$DEVICE_DIR/" >/dev/null
  [ -d "$BUNDLE_DIR/models" ] || die "bundle missing models/ (supertonic2.gguf)"
  "${ADB[@]}" push "$BUNDLE_DIR/models" "$DEVICE_DIR/" >/dev/null

  # node_modules: ~30k tiny files drop the USB link mid-push, so ship a tarball
  # and extract on-device (toybox tar handles -xzf).
  if [ -f "$BUNDLE_DIR/node_modules.tgz" ]; then
    "${ADB[@]}" push "$BUNDLE_DIR/node_modules.tgz" "$DEVICE_DIR/node_modules.tgz" >/dev/null
    "${ADB[@]}" shell "cd $DEVICE_DIR && tar xzf node_modules.tgz && rm -f node_modules.tgz"
  elif [ -d "$BUNDLE_DIR/node_modules" ]; then
    warn "pushing node_modules as a directory (slow / may drop the USB link). Prefer node_modules.tgz."
    "${ADB[@]}" push "$BUNDLE_DIR/node_modules" "$DEVICE_DIR/" >/dev/null
  else
    die "bundle missing node_modules.tgz (or node_modules/)"
  fi
  ok "base staged"
else
  ok "base already staged (use --full-stage to re-push)"
fi

say "staging prebuild (.bare + backend .so) + test"
[ -d "$BUNDLE_DIR/prebuilds" ] || die "bundle missing prebuilds/ (the addon .bare + .so to test)"
"${ADB[@]}" push "$BUNDLE_DIR/prebuilds" "$DEVICE_DIR/" >/dev/null
"${ADB[@]}" push "$SCRIPT_DIR/$TEST_JS" "$DEVICE_DIR/$TEST_JS" >/dev/null
ok "prebuild + test staged"

# ── 2. run ───────────────────────────────────────────────────────────────────
run_pass() {
  local label="$1" use_gpu="$2"
  local out_dev="$DEVICE_DIR/result-$label.out"
  local text_env=""
  [ -n "$TEXT_OVERRIDE" ] && text_env="TTS_TEXT=$(printf '%q' "$TEXT_OVERRIDE")"
  say "running $label pass (useGPU=$use_gpu)"
  "${ADB[@]}" logcat -c || true
  set +e
  "${ADB[@]}" shell "LD_LIBRARY_PATH=$DEVICE_DIR TTS_USE_GPU=$use_gpu TTS_OUT=$out_dev $text_env $DEVICE_DIR/bare $DEVICE_DIR/$TEST_JS"
  local rc=$?
  set -e
  "${ADB[@]}" pull "$out_dev" "$OUTDIR/$label-result.out" >/dev/null 2>&1 || warn "no result file (the run may have crashed before writing)"
  # tag-filtered diag (the decisive lines) AND the full logcat (for crash triage)
  "${ADB[@]}" logcat -d -s qvac-supertonic:* > "$OUTDIR/$label-gpudiag.log" 2>/dev/null || true
  "${ADB[@]}" logcat -d > "$OUTDIR/$label-full-logcat.txt" 2>/dev/null || true
  ok "$label pass done (exit $rc)"
  return 0
}

run_pass gpu 1
[ "$CPU_BASELINE" = "1" ] && run_pass cpu 0

# ── 3. summarise ─────────────────────────────────────────────────────────────
say "summary"
{
  echo "QVAC-20557 Supertonic GPU verify — iteration $LABEL — $(date)"
  echo "device: $MODEL  |  $RENDERER"
  [ -n "$TEXT_OVERRIDE" ] && echo "text override: $TEXT_OVERRIDE"
  echo
  echo "=== GPU pass — RESULT_JSON ==="
  grep -h 'RESULT_JSON' "$OUTDIR/gpu-result.out" 2>/dev/null || echo "(none — see gpu-full-logcat.txt for a crash)"
  echo
  echo "=== GPU pass — duration-predictor trisection (the decisive numbers) ==="
  grep -hE 'dprobe_pw1_im2col|dprobe_pw1_mulmat|dprobe_pwconv1|duration raw|wav_full|text_emb|cfm_latent' \
       "$OUTDIR/gpu-gpudiag.log" 2>/dev/null || echo "(no [gpu-diag] lines — crash before compute, or GPU fell back to CPU)"
  if [ "$CPU_BASELINE" = "1" ]; then
    echo
    echo "=== CPU baseline — RESULT_JSON ==="
    grep -h 'RESULT_JSON' "$OUTDIR/cpu-result.out" 2>/dev/null || echo "(none)"
  fi
} | tee "$OUTDIR/SUMMARY.txt"

echo
ok "iteration $LABEL collected into $OUTDIR/"
echo "   When the analysis is complete, run:  bash package-results.sh"
