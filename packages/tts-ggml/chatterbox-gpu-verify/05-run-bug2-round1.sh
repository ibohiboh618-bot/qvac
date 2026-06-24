#!/usr/bin/env bash
# Bug-2 (Mali Vulkan S3Gen "token-32 collapse") — ROUND 1.  ONE build, THREE runs.
# All turbo, T3 pinned to CPU (deterministic speech tokens), seed 42.
#
#   r1-base        S3Gen on GPU, S3GEN_DIAG=1            -> localize the first
#                  diverging stage (encoder mu_T vs CFM mel vs f0) + baseline WAV.
#   r1-cfmunfused  S3Gen on GPU, S3GEN_DIAG=1,           -> FA-on-Mali fix-swing:
#                  S3GEN_FIX=cfm_unfused                    does the collapse vanish?
#   r1-cpuref      S3Gen on CPU, S3GEN_DIAG=1            -> known-good per-stage
#                                                           trace to diff A against.
#
# Each run writes <label>.{wav,result.json,console.txt,gpudiag.txt} to OUT_DIR.
# The [s3gen-diag] per-stage/per-block trace is in <label>.console.txt.
#
# Optional env: OUT_DIR (default ./chbx-results)  SERIAL (adb serial)  TIMEOUT (default 300)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-./chbx-results}"
export OUT_DIR
run() { bash "$HERE/03-run-config.sh" "$@"; }

export S3GEN_DIAG=1

export S3GEN_FIX=""
run turbo cpu gpu r1-base          # baseline GPU + diag (localize)

export S3GEN_FIX="cfm_unfused"
run turbo cpu gpu r1-cfmunfused    # GPU + FA->soft_max fix-swing

export S3GEN_FIX=""
run turbo cpu cpu r1-cpuref        # CPU reference + diag

echo "ALL DONE -> $OUT_DIR"
echo "Send back, per label r1-base / r1-cfmunfused / r1-cpuref:"
echo "  <label>.wav  <label>.result.json  <label>.console.txt  <label>.gpudiag.txt"
echo "Most important: the three *.console.txt (they hold the [s3gen-diag] trace)."
