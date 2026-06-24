#!/usr/bin/env bash
# Run ONE Supertonic config on the device; pull AUDIO first, then result.json, then console.
# Usage: 03b-run-supertonic.sh <v2|v3> <gpu|cpu> <label>
# Optional env: DEVICE_DIR (default /data/local/tmp/chbx)  OUT_DIR (default ./chbx-results)
#               SERIAL (adb serial)  TIMEOUT (default 300)  ST3_QUANT (default q4_0)
set -uo pipefail
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/chbx}"
OUT_DIR="${OUT_DIR:-./chbx-results}"
TIMEOUT="${TIMEOUT:-300}"
ST3_QUANT="${ST3_QUANT:-q4_0}"
ADB=(adb); [ -n "${SERIAL:-}" ] && ADB=(adb -s "$SERIAL")
VARIANT="$1"; BACKEND="$2"; LABEL="$3"
mkdir -p "$OUT_DIR"
echo "== RUN $LABEL : supertonic variant=$VARIANT backend=$BACKEND (hard timeout ${TIMEOUT}s)"
# bare JS console.log is swallowed over adb shell -> we rely on the result.json file;
# native ggml/[st-verify] stdout is captured into console.txt. 5-min hard timeout = no hang.
"${ADB[@]}" shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=$DEVICE_DIR MODEL_DIR=$DEVICE_DIR/models BACKENDS_DIR=$DEVICE_DIR/prebuilds ST_VARIANT=$VARIANT ST_BACKEND=$BACKEND ST3_QUANT=$ST3_QUANT OUT_WAV=$DEVICE_DIR/out/$LABEL.wav RESULT_OUT=$DEVICE_DIR/out/$LABEL.result.json timeout $TIMEOUT ./bare chatterbox-gpu-verify/supertonic-run.js" > "$OUT_DIR/$LABEL.console.txt" 2>&1 || echo "   (run non-zero / timed out -- see $LABEL.console.txt)"
# 1) AUDIO FIRST
"${ADB[@]}" pull "$DEVICE_DIR/out/$LABEL.wav" "$OUT_DIR/$LABEL.wav" || echo "   NO WAV"
# 2) result.json (backendDevice / backendId / RTF / passed)
"${ADB[@]}" pull "$DEVICE_DIR/out/$LABEL.result.json" "$OUT_DIR/$LABEL.result.json" || echo "   NO RESULT"
"${ADB[@]}" shell "pkill -9 bare" >/dev/null 2>&1 || true
echo "OK $LABEL -> $OUT_DIR/$LABEL.{wav,result.json,console.txt}"
