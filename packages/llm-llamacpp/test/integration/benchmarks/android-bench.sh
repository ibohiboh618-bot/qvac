#!/system/bin/sh
# ──────────────────────────────────────────────────────────────────────────────
# android-bench.sh — Run all integration-test models on Android
#
# Text-only models use llama-completion (single-shot, no chat loop).
# VLM (vision) models use llama-mtmd-cli (single-shot with --image).
#
# Usage:
#   chmod +x android-bench.sh
#   BUILD_TAG=openmp    ./android-bench.sh   # run with OpenMP build
#   BUILD_TAG=no-openmp ./android-bench.sh   # run with non-OpenMP build
#
# Tunables (env vars):
#   BUILD_TAG  — label for this run's log directory     (default: "default")
#   RUNS       — iterations per test                    (default: 3)
#   NGL        — GPU layers (-ngl)                      (default: 99)
#   THREADS    — CPU threads (-t)                       (default: 4)
#   PREDICT    — max tokens to generate (-n)            (default: 256)
#
# All logs land in ./logs/<BUILD_TAG>/<model>-<test>.log
#
# Required directory layout:
#
#   /data/local/tmp/llm-bench/
#   ├── android-bench.sh
#   ├── llama-completion              (text-only binary)
#   ├── llama-mtmd-cli                (multimodal binary)
#   ├── libllama.so  libmtmd.so  libggml.so  libggml-base.so
#   ├── libggml-cpu*.so  libggml-vulkan.so  libggml-opencl.so
#   │
#   ├── models/
#   │   ├── Qwen3-0.6B-Q8_0.gguf
#   │   ├── Qwen3-1.7B-Q4_0.gguf
#   │   ├── Qwen3.5-0.8B-Q8_0.gguf
#   │   ├── mmproj-Qwen3.5-0.8B-F16.gguf
#   │   ├── google_gemma-4-E2B-it-Q4_K_M.gguf
#   │   ├── mmproj-google_gemma-4-E2B-it-bf16.gguf
#   │   ├── SmolVLM2-500M-Video-Instruct-Q8_0.gguf
#   │   ├── mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf
#   │   ├── Llama-3.2-1B-Instruct-Q4_0.gguf
#   │   ├── bitnet_b1_58-large-TQ2_0.gguf
#   │   ├── PaddleOCR-VL-1.5.gguf
#   │   ├── PaddleOCR-VL-1.5-mmproj.gguf
#   │   ├── Qwen3VL-2B-Instruct-Q4_K_M.gguf
#   │   └── mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf
#   │
#   └── images/
#       ├── elephant.jpg
#       ├── fruitPlate.png
#       ├── highRes3000x4000.jpg
#       └── news-paper.jpg
# ──────────────────────────────────────────────────────────────────────────────

set -e

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
CLI_TEXT="$BENCH_DIR/llama-completion"
CLI_VLM="$BENCH_DIR/llama-mtmd-cli"
MODEL_DIR="$BENCH_DIR/models"
IMAGE_DIR="$BENCH_DIR/images"

BUILD_TAG="${BUILD_TAG:-default}"
RUNS="${RUNS:-3}"
NGL="${NGL:-99}"
THREADS="${THREADS:-4}"
PREDICT="${PREDICT:-256}"

LOG_DIR="$BENCH_DIR/logs/$BUILD_TAG"
mkdir -p "$LOG_DIR"

export LD_LIBRARY_PATH="$BENCH_DIR"

# ── helpers ──────────────────────────────────────────────────────────────────

ts() { date '+%Y-%m-%d %H:%M:%S'; }

log_header() {
  local logfile="$1"
  local binary="$2"
  echo "================================================================" >> "$logfile"
  echo "BUILD_TAG : $BUILD_TAG"   >> "$logfile"
  echo "DATE      : $(ts)"       >> "$logfile"
  echo "DEVICE    : $(getprop ro.product.model 2>/dev/null || echo unknown)" >> "$logfile"
  echo "SOC       : $(getprop ro.hardware.chipname 2>/dev/null || getprop ro.board 2>/dev/null || echo unknown)" >> "$logfile"
  echo "ANDROID   : $(getprop ro.build.version.release 2>/dev/null || echo unknown)" >> "$logfile"
  echo "BINARY    : $(basename "$binary")" >> "$logfile"
  echo "NGL       : $NGL"        >> "$logfile"
  echo "THREADS   : $THREADS"    >> "$logfile"
  echo "PREDICT   : $PREDICT"    >> "$logfile"
  echo "RUNS      : $RUNS"       >> "$logfile"
  echo "================================================================" >> "$logfile"
  echo "" >> "$logfile"
}

run_text() {
  local label="$1"
  local logfile="$2"
  shift 2

  echo ""
  echo "──── $label (llama-completion) ────"
  echo "[$(ts)] START: $label"

  log_header "$logfile" "$CLI_TEXT"
  echo "COMMAND: $CLI_TEXT $*" >> "$logfile"
  echo "" >> "$logfile"

  local run=1
  while [ "$run" -le "$RUNS" ]; do
    echo "--- run $run/$RUNS ---" >> "$logfile"
    echo "[$(ts)] run $run/$RUNS ..."

    "$CLI_TEXT" -no-cnv "$@" >> "$logfile" 2>&1 || {
      echo "[$(ts)] FAILED (exit $?)" | tee -a "$logfile"
    }

    echo "" >> "$logfile"
    run=$((run + 1))
  done

  echo "[$(ts)] DONE:  $label"
}

run_vlm() {
  local label="$1"
  local logfile="$2"
  shift 2

  echo ""
  echo "──── $label (llama-mtmd-cli) ────"
  echo "[$(ts)] START: $label"

  log_header "$logfile" "$CLI_VLM"
  echo "COMMAND: $CLI_VLM $*" >> "$logfile"
  echo "" >> "$logfile"

  local run=1
  while [ "$run" -le "$RUNS" ]; do
    echo "--- run $run/$RUNS ---" >> "$logfile"
    echo "[$(ts)] run $run/$RUNS ..."

    "$CLI_VLM" "$@" >> "$logfile" 2>&1 || {
      echo "[$(ts)] FAILED (exit $?)" | tee -a "$logfile"
    }

    echo "" >> "$logfile"
    run=$((run + 1))
  done

  echo "[$(ts)] DONE:  $label"
}

# ── preflight ────────────────────────────────────────────────────────────────

HAVE_TEXT=true
HAVE_VLM=true

if [ ! -x "$CLI_TEXT" ]; then
  echo "WARNING: $CLI_TEXT not found — text-only tests will be skipped"
  HAVE_TEXT=false
fi
if [ ! -x "$CLI_VLM" ]; then
  echo "WARNING: $CLI_VLM not found — VLM tests will be skipped"
  HAVE_VLM=false
fi
if [ "$HAVE_TEXT" = false ] && [ "$HAVE_VLM" = false ]; then
  echo "ERROR: neither llama-completion nor llama-mtmd-cli found"
  exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  android-bench.sh                                              ║"
echo "║  BUILD_TAG=$BUILD_TAG   RUNS=$RUNS   NGL=$NGL   THREADS=$THREADS          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
#  1. Qwen3-0.6B  (text-only)
# ──────────────────────────────────────────────────────────────────────────────

M_QWEN3_06="$MODEL_DIR/Qwen3-0.6B-Q8_0.gguf"
if [ -f "$M_QWEN3_06" ] && [ "$HAVE_TEXT" = true ]; then
  run_text "qwen3-0.6b / text" "$LOG_DIR/qwen3-0.6b-text.log" \
    -m "$M_QWEN3_06" \
    -p "What is 2+2? Answer in one word." \
    -ngl "$NGL" -c 2048 -n "$PREDICT" -t "$THREADS" \
    --temp 0 --seed 42

  run_text "qwen3-0.6b / long-prompt" "$LOG_DIR/qwen3-0.6b-long.log" \
    -m "$M_QWEN3_06" \
    -p "What is the capital of France? Answer in one word." \
    -ngl "$NGL" -c 2048 -n "$PREDICT" -t "$THREADS" \
    --temp 0 --seed 42
else
  echo "SKIP: Qwen3-0.6B (model missing or llama-completion unavailable)"
fi

# ──────────────────────────────────────────────────────────────────────────────
#  2. Qwen3-1.7B  (text-only)
# ──────────────────────────────────────────────────────────────────────────────

M_QWEN3_17="$MODEL_DIR/Qwen3-1.7B-Q4_0.gguf"
if [ -f "$M_QWEN3_17" ] && [ "$HAVE_TEXT" = true ]; then
  run_text "qwen3-1.7b / text" "$LOG_DIR/qwen3-1.7b-text.log" \
    -m "$M_QWEN3_17" \
    -p "What is the capital of France? Answer in one word." \
    -ngl "$NGL" -c 4096 -n "$PREDICT" -t "$THREADS" \
    --temp 0 --seed 42
else
  echo "SKIP: Qwen3-1.7B (model missing or llama-completion unavailable)"
fi

# ──────────────────────────────────────────────────────────────────────────────
#  3. Llama-3.2-1B  (text-only)
# ──────────────────────────────────────────────────────────────────────────────

M_LLAMA32="$MODEL_DIR/Llama-3.2-1B-Instruct-Q4_0.gguf"
if [ -f "$M_LLAMA32" ] && [ "$HAVE_TEXT" = true ]; then
  run_text "llama-3.2-1b / text" "$LOG_DIR/llama-3.2-1b-text.log" \
    -m "$M_LLAMA32" \
    -p "Write a short poem about the ocean." \
    -ngl "$NGL" -c 2048 -n "$PREDICT" -t "$THREADS" \
    --temp 0 --seed 42
else
  echo "SKIP: Llama-3.2-1B (model missing or llama-completion unavailable)"
fi

# ──────────────────────────────────────────────────────────────────────────────
#  4. BitNet  (text-only, Android-only test)
# ──────────────────────────────────────────────────────────────────────────────

M_BITNET="$MODEL_DIR/bitnet_b1_58-large-TQ2_0.gguf"
if [ -f "$M_BITNET" ] && [ "$HAVE_TEXT" = true ]; then
  run_text "bitnet / text" "$LOG_DIR/bitnet-text.log" \
    -m "$M_BITNET" \
    -p "What is 2 + 2?" \
    -ngl "$NGL" -c 2048 -n "$PREDICT" -t "$THREADS" \
    --temp 0 --seed 42
else
  echo "SKIP: BitNet (model missing or llama-completion unavailable)"
fi

# ──────────────────────────────────────────────────────────────────────────────
#  5. Gemma 4 E2B VLM  (elephant.jpg)
# ──────────────────────────────────────────────────────────────────────────────

M_GEMMA4="$MODEL_DIR/google_gemma-4-E2B-it-Q4_K_M.gguf"
P_GEMMA4="$MODEL_DIR/mmproj-google_gemma-4-E2B-it-bf16.gguf"
if [ -f "$M_GEMMA4" ] && [ -f "$P_GEMMA4" ] && [ "$HAVE_VLM" = true ] && [ -f "$IMAGE_DIR/elephant.jpg" ]; then

  run_vlm "gemma4 / elephant" "$LOG_DIR/gemma4-elephant.log" \
    -m "$M_GEMMA4" \
    --mmproj "$P_GEMMA4" \
    --image "$IMAGE_DIR/elephant.jpg" \
    -p "Describe the image briefly in one sentence." \
    -ngl "$NGL" -c 4096 -n "$PREDICT" -t "$THREADS" \
    --temp 0 --seed 42 --jinja -fit off \
    --ubatch-size 320

else
  echo "SKIP: Gemma 4 (model/mmproj/image missing or llama-mtmd-cli unavailable)"
fi

# ──────────────────────────────────────────────────────────────────────────────
#  6. Qwen3.5-VL 0.8B  (text + elephant.jpg)
# ──────────────────────────────────────────────────────────────────────────────

M_QWEN35="$MODEL_DIR/Qwen3.5-0.8B-Q8_0.gguf"
P_QWEN35="$MODEL_DIR/mmproj-Qwen3.5-0.8B-F16.gguf"

# text-only via llama-completion
if [ -f "$M_QWEN35" ] && [ "$HAVE_TEXT" = true ]; then
  run_text "qwen3.5-vl / text" "$LOG_DIR/qwen35-text.log" \
    -m "$M_QWEN35" \
    -p "What is 2+2? Answer in one word." \
    -ngl "$NGL" -c 2048 -n "$PREDICT" -t "$THREADS" \
    --temp 0 --seed 42
fi

# VLM via llama-mtmd-cli
if [ -f "$M_QWEN35" ] && [ -f "$P_QWEN35" ] && [ "$HAVE_VLM" = true ] && [ -f "$IMAGE_DIR/elephant.jpg" ]; then
  run_vlm "qwen3.5-vl / elephant" "$LOG_DIR/qwen35-elephant.log" \
    -m "$M_QWEN35" \
    --mmproj "$P_QWEN35" \
    --image "$IMAGE_DIR/elephant.jpg" \
    -p "Describe the image briefly in one sentence." \
    -ngl "$NGL" -c 4096 -n "$PREDICT" -t "$THREADS" \
    --temp 0 --seed 42 --jinja -fit off \
    --no-mmproj-offload
else
  echo "SKIP: Qwen3.5-VL VLM (mmproj/image missing or llama-mtmd-cli unavailable)"
fi

# ──────────────────────────────────────────────────────────────────────────────
#  7. SmolVLM2-500M  (elephant.jpg)
# ──────────────────────────────────────────────────────────────────────────────

M_SMOL="$MODEL_DIR/SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
P_SMOL="$MODEL_DIR/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
if [ -f "$M_SMOL" ] && [ -f "$P_SMOL" ] && [ "$HAVE_VLM" = true ] && [ -f "$IMAGE_DIR/elephant.jpg" ]; then

  run_vlm "smolvlm2 / elephant" "$LOG_DIR/smolvlm2-elephant.log" \
    -m "$M_SMOL" \
    --mmproj "$P_SMOL" \
    --image "$IMAGE_DIR/elephant.jpg" \
    -p "Describe the image briefly in one sentence." \
    -ngl "$NGL" -c 2048 -n "$PREDICT" -t "$THREADS" \
    --temp 0 --seed 42 --jinja -fit off

else
  echo "SKIP: SmolVLM2 (model/mmproj/image missing or llama-mtmd-cli unavailable)"
fi

# ──────────────────────────────────────────────────────────────────────────────
#  8. Qwen3-VL 2B  (elephant.jpg)
# ──────────────────────────────────────────────────────────────────────────────

M_QWEN3VL="$MODEL_DIR/Qwen3VL-2B-Instruct-Q4_K_M.gguf"
P_QWEN3VL="$MODEL_DIR/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
if [ -f "$M_QWEN3VL" ] && [ -f "$P_QWEN3VL" ] && [ "$HAVE_VLM" = true ] && [ -f "$IMAGE_DIR/elephant.jpg" ]; then

  run_vlm "qwen3vl-2b / elephant" "$LOG_DIR/qwen3vl-2b-elephant.log" \
    -m "$M_QWEN3VL" \
    --mmproj "$P_QWEN3VL" \
    --image "$IMAGE_DIR/elephant.jpg" \
    -p "Describe the image briefly in one sentence." \
    -ngl "$NGL" -c 7046 -n "$PREDICT" -t "$THREADS" \
    --temp 0 --seed 42 --jinja -fit off

else
  echo "SKIP: Qwen3VL-2B (model/mmproj/image missing or llama-mtmd-cli unavailable)"
fi

# ──────────────────────────────────────────────────────────────────────────────
#  9. PaddleOCR-VL  (news-paper.jpg, CPU on mobile)
# ──────────────────────────────────────────────────────────────────────────────

M_PADDLE="$MODEL_DIR/PaddleOCR-VL-1.5.gguf"
P_PADDLE="$MODEL_DIR/PaddleOCR-VL-1.5-mmproj.gguf"
if [ -f "$M_PADDLE" ] && [ -f "$P_PADDLE" ] && [ "$HAVE_VLM" = true ] && [ -f "$IMAGE_DIR/news-paper.jpg" ]; then

  run_vlm "paddle-ocr / news-paper (CPU)" "$LOG_DIR/paddle-ocr-newspaper.log" \
    -m "$M_PADDLE" \
    --mmproj "$P_PADDLE" \
    --image "$IMAGE_DIR/news-paper.jpg" \
    -p "Extract all text from this image." \
    -ngl 0 -c 4096 -n 768 -t "$THREADS" \
    --temp 0.1 --seed 42 --jinja -fit off

else
  echo "SKIP: PaddleOCR (model/mmproj/image missing or llama-mtmd-cli unavailable)"
fi

# ──────────────────────────────────────────────────────────────────────────────
#  Summary
# ──────────────────────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  ALL DONE — logs in: $LOG_DIR"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "To compare OpenMP vs non-OpenMP builds, run:"
echo "  adb pull $LOG_DIR /tmp/android-bench-$BUILD_TAG"
echo ""
echo "Key perf lines to grep from logs:"
echo "  grep -h 'prompt eval time\|eval time\|total time\|image.*encoded' $LOG_DIR/*.log"
echo ""
