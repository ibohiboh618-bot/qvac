#!/usr/bin/env bash
# Bug-2 ROUND 2 — confirm the is_arm_mali GATE auto-fixes Chatterbox on Mali with
# NO env flags (the shipping code path = PR #67 fix). ONE build. turbo T3 pinned
# to CPU, seed 42.
#
#   r2-gated    turbo GPU, S3GEN_DIAG=1                       -> GATE auto-fix.
#               Expect: config is_mali=1 cfm_unfused=1, f0 bad=0, clean audio.
#   r2-forcefa  turbo GPU, S3GEN_DIAG=1 TTS_CPP_CHBX_CFM_FA=1 -> A/B control: force
#               the fused path -> reproduces the break (cfm_unfused=0, f0 explodes).
#   r2-text2    turbo GPU, S3GEN_DIAG=1, longer text          -> shape robustness.
#   r2-mtl      mtl   GPU, S3GEN_DIAG=1                        -> B=2 path (basic_tfm_b);
#               needs chatterbox-t3-mtl.gguf + chatterbox-s3gen-mtl.gguf on device.
#
# Each writes ./chbx-results/<label>.{wav,result.json,console.txt,gpudiag.txt}.
# Optional env: OUT_DIR (default ./chbx-results)  SERIAL  TIMEOUT (default 300)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-./chbx-results}"
export OUT_DIR
run() { bash "$HERE/03-run-config.sh" "$@"; }

export S3GEN_DIAG=1

export TTS_CPP_CHBX_CFM_FA="" CORR_TEXT=""
run turbo cpu gpu r2-gated          # GATE auto-fix (no env) -- the shipping path

export TTS_CPP_CHBX_CFM_FA=1 CORR_TEXT=""
run turbo cpu gpu r2-forcefa        # A/B control: force FA -> reproduce the break

export TTS_CPP_CHBX_CFM_FA="" CORR_TEXT="The quick brown fox jumps over the lazy dog, while seven bright stars shimmer above the quiet harbor at midnight."
run turbo cpu gpu r2-text2          # robustness on a longer sentence

export TTS_CPP_CHBX_CFM_FA="" CORR_TEXT=""
run mtl cpu gpu r2-mtl             # B=2 path (skips with a warning if mtl models absent)

echo "ALL DONE -> $OUT_DIR"
echo "Send back per label r2-gated / r2-forcefa / r2-text2 / r2-mtl:"
echo "  <label>.wav  <label>.result.json  <label>.console.txt"
echo "Key: r2-gated -> is_mali=1 cfm_unfused=1, f0 bad=0, clean; r2-forcefa -> cfm_unfused=0, f0 explodes."
