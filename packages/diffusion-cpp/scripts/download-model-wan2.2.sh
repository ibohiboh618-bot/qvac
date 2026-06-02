#!/usr/bin/env bash
set -euo pipefail

# Wan 2.2 TI2V-5B Video Generation Model (GGUF Quantized)
#
# Source: QuantStack/Wan2.2-TI2V-5B-GGUF
#         Community quantized GGUF conversion for consumer hardware.
#
# Available quantization levels:
#   Q2_K   (~1.85 GB) - Smallest, fastest, lowest quality
#   Q3_K_S (~2.29 GB)
#   Q3_K_M (~2.55 GB)
#   Q4_0   (~3.03 GB)
#   Q4_K_S (~3.12 GB) - Good balance
#   Q4_1   (~3.25 GB)
#   Q4_K_M (~3.43 GB)
#   Q5_0   (~3.64 GB)
#   Q5_K_S (~3.56 GB)
#   Q5_1   (~3.87 GB)
#   Q5_K_M (~3.81 GB)
#   Q6_K   (~4.21 GB)
#   Q8_0   (~5.4 GB)  - Best quality, DEFAULT
#
# Usage:
#   ./download-model-wan2.2.sh              # Downloads Q8_0 (default)
#   ./download-model-wan2.2.sh --q4ks       # Downloads Q4_K_S
#   ./download-model-wan2.2.sh --q2k --q4ks # Downloads multiple quantizations
#   ./download-model-wan2.2.sh --all        # Downloads all quantizations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(cd "$SCRIPT_DIR/.." && pwd)/models"
HF="https://huggingface.co"
REPO="QuantStack/Wan2.2-TI2V-5B-GGUF"

mkdir -p "$OUT"

source "$SCRIPT_DIR/dl-functions.sh"

# Parse command line arguments
QUANTIZATIONS=()
DOWNLOAD_ALL=false

for arg in "$@"; do
  case "$arg" in
    --all) DOWNLOAD_ALL=true ;;
    --q2k) QUANTIZATIONS+=("Q2_K") ;;
    --q3ks) QUANTIZATIONS+=("Q3_K_S") ;;
    --q3km) QUANTIZATIONS+=("Q3_K_M") ;;
    --q40) QUANTIZATIONS+=("Q4_0") ;;
    --q4ks) QUANTIZATIONS+=("Q4_K_S") ;;
    --q41) QUANTIZATIONS+=("Q4_1") ;;
    --q4km) QUANTIZATIONS+=("Q4_K_M") ;;
    --q50) QUANTIZATIONS+=("Q5_0") ;;
    --q5ks) QUANTIZATIONS+=("Q5_K_S") ;;
    --q51) QUANTIZATIONS+=("Q5_1") ;;
    --q5km) QUANTIZATIONS+=("Q5_K_M") ;;
    --q6k) QUANTIZATIONS+=("Q6_K") ;;
    --q80) QUANTIZATIONS+=("Q8_0") ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

# If --all, download all quantizations
if [ "$DOWNLOAD_ALL" = true ]; then
  QUANTIZATIONS=("Q2_K" "Q3_K_S" "Q3_K_M" "Q4_0" "Q4_K_S" "Q4_1" "Q4_K_M" "Q5_0" "Q5_K_S" "Q5_1" "Q5_K_M" "Q6_K" "Q8_0")
fi

# If no quantizations specified, default to Q8_0
if [ ${#QUANTIZATIONS[@]} -eq 0 ]; then
  QUANTIZATIONS=("Q8_0")
fi

echo "Wan2.2 TI2V-5B GGUF Model Download"
echo "===================================="
echo ""
echo "Downloading quantizations: ${QUANTIZATIONS[@]}"
echo "Output directory: $OUT"
echo ""

# Download each quantization
for quant in "${QUANTIZATIONS[@]}"; do
  filename="Wan2.2-TI2V-5B-${quant}.gguf"
  url="$HF/$REPO/resolve/main/$filename"
  output_path="$OUT/$filename"
  
  dl "$url" "$output_path"
done

echo ""
echo "✓ Done → $OUT"
echo ""
echo "Models downloaded:"
ls -lh "$OUT"/Wan2.2-TI2V-5B-*.gguf 2>/dev/null || echo "No models found"
