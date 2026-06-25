#!/usr/bin/env bash
# QVAC-20557 Bug 1 (SVE/armv9 CPU Nyquist tone) — on-device CPU A/B verify.
# Chatterbox ALL-CPU, THREE runs on ONE build. The ggml-speech overlay pins the
# verify branch = the svmad->svmla leftover-tail fix + a TTS_SVE_DOT_UNFIXED env
# toggle + a once-per-process [sve-diag] print of svcntb/epr/tail.
#
#   1) bug1-sve-postfix : all CPU variants present -> loader scores onto armv9 SVE;
#                         default env (svmla FIX).  PRIMARY BAR.
#                         EXPECT console "[sve-diag] ... svcntb=16 epr=4 tail=svmla(FIXED)"
#                                + nyqFrac < 0.1 (CLEAN).
#   2) bug1-sve-prefix  : same, but TTS_SVE_DOT_UNFIXED=1 -> old svmad tail.  A/B CONTROL.
#                         EXPECT "[sve-diag] ... tail=svmad(UNFIXED)" + nyqFrac >> 0.1 (TONE).
#   3) bug1-neon-control: hide armv9*.so so the loader picks armv8.6 NEON; default env.
#                         EXPECT NO "[sve-diag]" line (SVE dot path never runs) + nyqFrac < 0.1 (CLEAN).
#
# After pulling, gate host-side:
#   bare packages/tts-ggml/test/utils/analyze-tone.js <OUT_DIR>/bug1-*.wav
#
# Usage: 07-run-bug1-sve.sh [turbo]      (variant; default turbo)
# Env:   DEVICE_DIR (default /data/local/tmp/chbx)  OUT_DIR (default ./chbx-results)
#        SERIAL (adb serial)  TIMEOUT (default 300)
set -uo pipefail
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/chbx}"
OUT_DIR="${OUT_DIR:-./chbx-results}"
TIMEOUT="${TIMEOUT:-300}"
ADB=(adb); [ -n "${SERIAL:-}" ] && ADB=(adb -s "$SERIAL")
VARIANT="${1:-turbo}"
mkdir -p "$OUT_DIR"

run_cpu () {  # $1=label  $2=extra-env (e.g. "TTS_SVE_DOT_UNFIXED=1 " — keep trailing space)
  local LABEL="$1" EXTRA="$2"
  echo "== RUN $LABEL : Chatterbox all-CPU (variant=$VARIANT) env=[${EXTRA:-<default>}]"
  "${ADB[@]}" shell "cd $DEVICE_DIR && LD_LIBRARY_PATH=$DEVICE_DIR MODEL_DIR=$DEVICE_DIR/models BACKENDS_DIR=$DEVICE_DIR/prebuilds REF_WAV=$DEVICE_DIR/test/reference-audio/jfk.wav CHBX_VARIANT=$VARIANT T3_BACKEND=cpu S3GEN_BACKEND=cpu ${EXTRA}OUT_WAV=$DEVICE_DIR/out/$LABEL.wav RESULT_OUT=$DEVICE_DIR/out/$LABEL.result.json timeout $TIMEOUT ./bare chatterbox-gpu-verify/run.js" > "$OUT_DIR/$LABEL.console.txt" 2>&1 || echo "   (non-zero/timeout -- see $LABEL.console.txt)"
  "${ADB[@]}" pull "$DEVICE_DIR/out/$LABEL.wav" "$OUT_DIR/$LABEL.wav" || echo "   NO WAV"
  "${ADB[@]}" pull "$DEVICE_DIR/out/$LABEL.result.json" "$OUT_DIR/$LABEL.result.json" || echo "   NO RESULT"
  "${ADB[@]}" shell "pkill -9 bare" >/dev/null 2>&1 || true
  echo "   [sve-diag] captured:"
  grep -a "\[sve-diag\]" "$OUT_DIR/$LABEL.console.txt" || echo "   (none — SVE dot path did not run)"
  echo "OK $LABEL -> $OUT_DIR/$LABEL.{wav,result.json,console.txt}"
}

# 1) SVE post-fix (default = svmla) — PRIMARY BAR
run_cpu "bug1-sve-postfix" ""

# 2) SVE pre-fix (force old svmad) — A/B control, reproduces the tone
run_cpu "bug1-sve-prefix" "TTS_SVE_DOT_UNFIXED=1 "

# 3) NEON control — hide armv9*.so so the loader scores onto armv8.6 (NEON), then restore
"${ADB[@]}" shell "for f in \$(find $DEVICE_DIR/prebuilds -name 'libqvac-speech-ggml-cpu-android_armv9*.so'); do mv \"\$f\" \"\$f.off\"; done"
echo "   remaining cpu variant .so (armv9 should be gone):"
"${ADB[@]}" shell "find $DEVICE_DIR/prebuilds -name 'libqvac-speech-ggml-cpu-android_armv*.so'" || true
run_cpu "bug1-neon-control" ""
"${ADB[@]}" shell "for f in \$(find $DEVICE_DIR/prebuilds -name 'libqvac-speech-ggml-cpu-android_armv9*.so.off'); do mv \"\$f\" \"\${f%.off}\"; done"
echo "   restored armv9 variants."

echo
echo "== HOST GATE: bare packages/tts-ggml/test/utils/analyze-tone.js $OUT_DIR/bug1-sve-postfix.wav $OUT_DIR/bug1-sve-prefix.wav $OUT_DIR/bug1-neon-control.wav"
echo "== PASS: postfix CLEAN (nyqFrac<0.1) + [sve-diag] tail=svmla(FIXED); prefix TONE (nyqFrac>>0.1) + tail=svmad(UNFIXED); neon CLEAN + no [sve-diag]."
