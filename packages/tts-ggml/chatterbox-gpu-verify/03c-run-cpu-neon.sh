#!/usr/bin/env bash
# Run Chatterbox ALL-CPU with the NEON (armv8.6) CPU variant FORCED, to isolate the
# SVE(armv9) vs NEON(armv8.6) CPU-variant axis on the SAME Tensor device.
#
# WHY: the only thing differing between toney-Tensor-CPU and clean-Adreno-CPU is which
# aarch64 ggml-cpu variant runs (GGML_CPU_ALL_VARIANTS=ON; the loader scores a hardcoded
# candidate list and picks the highest). Tensor (SVE) lands on an armv9 SVE variant;
# Qualcomm/Adreno (SVE fused off) lands on NEON armv8.6. There is NO env override, so we
# force NEON by temporarily hiding the armv9.* variant .so (rename .so -> .so.off) so the
# loader can only pick armv8.6, then restore them.
#
# Pair this with the normal all-CPU run (03-run-config.sh turbo cpu cpu turbo-s3gen-cpu),
# which keeps all variants present and therefore loads the armv9 SVE variant (the control).
#
# Usage: 03c-run-cpu-neon.sh <turbo|mtl> <label>
# Optional env: DEVICE_DIR (default /data/local/tmp/chbx)  OUT_DIR (default ./chbx-results)
#               SERIAL (adb serial)  TIMEOUT (default 300)
set -uo pipefail
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/chbx}"
OUT_DIR="${OUT_DIR:-./chbx-results}"
TIMEOUT="${TIMEOUT:-300}"
ADB=(adb); [ -n "${SERIAL:-}" ] && ADB=(adb -s "$SERIAL")
VARIANT="$1"; LABEL="$2"
mkdir -p "$OUT_DIR"
echo "== RUN $LABEL : Chatterbox all-CPU, FORCED NEON (armv8.6) — variant=$VARIANT"

# Hide armv9.* CPU variant .so so only the NEON (armv8.x) variants remain loadable.
"${ADB[@]}" shell "for f in \$(find $DEVICE_DIR/prebuilds -name 'libqvac-speech-ggml-cpu-android_armv9*.so'); do mv \"\$f\" \"\$f.off\"; done"
echo "   remaining cpu variant .so on device (armv9 should be gone):"
"${ADB[@]}" shell "find $DEVICE_DIR/prebuilds -name 'libqvac-speech-ggml-cpu-android_armv*.so'" || true

# Run all-CPU (T3 cpu, S3Gen cpu). With armv9 hidden the loader scores onto armv8.6 (NEON).
"${ADB[@]}" shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=$DEVICE_DIR MODEL_DIR=$DEVICE_DIR/models BACKENDS_DIR=$DEVICE_DIR/prebuilds REF_WAV=$DEVICE_DIR/test/reference-audio/jfk.wav CHBX_VARIANT=$VARIANT T3_BACKEND=cpu S3GEN_BACKEND=cpu OUT_WAV=$DEVICE_DIR/out/$LABEL.wav RESULT_OUT=$DEVICE_DIR/out/$LABEL.result.json timeout $TIMEOUT ./bare chatterbox-gpu-verify/run.js" > "$OUT_DIR/$LABEL.console.txt" 2>&1 || echo "   (run non-zero / timed out -- see $LABEL.console.txt)"

# Restore the armv9.* variants no matter what (so later runs see all variants again).
"${ADB[@]}" shell "for f in \$(find $DEVICE_DIR/prebuilds -name 'libqvac-speech-ggml-cpu-android_armv9*.so.off'); do mv \"\$f\" \"\${f%.off}\"; done"
echo "   restored armv9 variants:"
"${ADB[@]}" shell "find $DEVICE_DIR/prebuilds -name 'libqvac-speech-ggml-cpu-android_armv9*.so'" || true

# Pull artifacts
"${ADB[@]}" pull "$DEVICE_DIR/out/$LABEL.wav" "$OUT_DIR/$LABEL.wav" || echo "   NO WAV"
"${ADB[@]}" pull "$DEVICE_DIR/out/$LABEL.result.json" "$OUT_DIR/$LABEL.result.json" || echo "   NO RESULT"
"${ADB[@]}" shell "pkill -9 bare" >/dev/null 2>&1 || true
echo "OK $LABEL -> $OUT_DIR/$LABEL.{wav,result.json,console.txt}"
