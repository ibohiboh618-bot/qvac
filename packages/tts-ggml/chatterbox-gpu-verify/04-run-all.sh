#!/usr/bin/env bash
# Run the full 8-sample matrix (6 distinct backend placements; the two *-t3-cpu
# labels are identical to *-s3gen-gpu and are produced by copy).
# Optional env: OUT_DIR (default ./chbx-results) + the env honoured by 03-run-config.sh.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-./chbx-results}"
export OUT_DIR
run() { bash "$HERE/03-run-config.sh" "$@"; }

run turbo cpu cpu turbo-s3gen-cpu   # all-CPU reference
run turbo cpu gpu turbo-s3gen-gpu   # S3Gen on GPU (T3 fixed CPU)
run turbo gpu gpu turbo-t3-gpu      # full GPU
run mtl   cpu cpu mtl-s3gen-cpu
run mtl   cpu gpu mtl-s3gen-gpu
run mtl   gpu gpu mtl-t3-gpu

# *-t3-cpu == *-s3gen-gpu (T3 CPU + S3Gen GPU); copy so the 8-file set is complete.
cp "$OUT_DIR/turbo-s3gen-gpu.wav" "$OUT_DIR/turbo-t3-cpu.wav" 2>/dev/null || true
cp "$OUT_DIR/mtl-s3gen-gpu.wav"   "$OUT_DIR/mtl-t3-cpu.wav"   2>/dev/null || true
echo "ALL DONE -> $OUT_DIR (8 labelled wavs; *-t3-cpu are copies of *-s3gen-gpu)"
