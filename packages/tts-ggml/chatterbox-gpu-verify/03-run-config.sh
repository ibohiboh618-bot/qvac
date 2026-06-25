#!/usr/bin/env bash
# Run ONE Chatterbox config on the device; pull AUDIO first, then result.json, then logcat.
# Usage: 03-run-config.sh <turbo|mtl> <cpu|gpu (T3)> <cpu|gpu (S3Gen)> <label>
# Optional env: DEVICE_DIR (default /data/local/tmp/chbx)  OUT_DIR (default ./chbx-results)
#               SERIAL (adb serial)  TIMEOUT (default 300)
set -uo pipefail
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/chbx}"
OUT_DIR="${OUT_DIR:-./chbx-results}"
TIMEOUT="${TIMEOUT:-300}"
ADB=(adb); [ -n "${SERIAL:-}" ] && ADB=(adb -s "$SERIAL")
VARIANT="$1"; T3="$2"; S3="$3"; LABEL="$4"
mkdir -p "$OUT_DIR"
echo "== RUN $LABEL : variant=$VARIANT T3=$T3 S3Gen=$S3 (hard timeout ${TIMEOUT}s)"
"${ADB[@]}" logcat -c || true
# bare JS console.log is swallowed over adb shell -> we rely on the result.json file;
# native ggml/BENCH stdout + the [s3gen-diag] per-stage trace (fprintf stderr) are
# captured into console.txt. 5-min hard timeout = no hang.
# S3GEN_DIAG / S3GEN_FIX are passed through from the caller's env (05-run-bug2-round1.sh)
# in the launch-env prefix so native std::getenv sees them (set-once-before-load).
"${ADB[@]}" shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=$DEVICE_DIR MODEL_DIR=$DEVICE_DIR/models BACKENDS_DIR=$DEVICE_DIR/prebuilds REF_WAV=$DEVICE_DIR/test/reference-audio/jfk.wav CHBX_VARIANT=$VARIANT T3_BACKEND=$T3 S3GEN_BACKEND=$S3 S3GEN_DIAG=${S3GEN_DIAG:-} S3GEN_FIX=${S3GEN_FIX:-} TTS_CPP_CHBX_CFM_FA=${TTS_CPP_CHBX_CFM_FA:-} CORR_TEXT='${CORR_TEXT:-}' OUT_WAV=$DEVICE_DIR/out/$LABEL.wav RESULT_OUT=$DEVICE_DIR/out/$LABEL.result.json timeout $TIMEOUT ./bare chatterbox-gpu-verify/run.js" > "$OUT_DIR/$LABEL.console.txt" 2>&1 || echo "   (run non-zero / timed out -- see $LABEL.console.txt)"
# 1) AUDIO FIRST
"${ADB[@]}" pull "$DEVICE_DIR/out/$LABEL.wav" "$OUT_DIR/$LABEL.wav" || echo "   NO WAV"
# 2) result.json (backendDevice / RTF / passed)
"${ADB[@]}" pull "$DEVICE_DIR/out/$LABEL.result.json" "$OUT_DIR/$LABEL.result.json" || echo "   NO RESULT"
# 3) per-stage [gpu-diag] (tag-filtered so driver spam doesn't roll it out of the buffer)
"${ADB[@]}" logcat -d -s 'qvac-chatterbox:*' > "$OUT_DIR/$LABEL.gpudiag.txt" 2>/dev/null || true
"${ADB[@]}" shell "pkill -9 bare" >/dev/null 2>&1 || true
echo "OK $LABEL -> $OUT_DIR/$LABEL.{wav,result.json,console.txt,gpudiag.txt}"
