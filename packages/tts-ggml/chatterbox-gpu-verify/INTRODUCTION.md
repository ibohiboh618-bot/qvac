# Chatterbox Mali-GPU fix — ROUND 2 (confirm the auto-gate) — 2026-06-25

**Read this first.** Round 1 found it: Chatterbox's CFM `flash_attn_ext` miscomputes on Mali Vulkan
(subtly-wrong mel → the f0 predictor blows up to NaN → broken audio), and swapping that attention to an
unfused soft_max+matmul fixed it. Round 2 confirms the **shipping fix**: an `is_arm_mali` gate that
applies that swap **automatically on Mali, with no env flags** (PR #67).

## What this build carries (overlay `ports/tts-cpp`, DO-NOT-MERGE)
The **PR #67 fix** (master + the `is_arm_mali`-gated unfused CFM attention, B=1 and B=2) **plus** the
`S3GEN_DIAG` per-stage trace (verify-only; not in the PR). On Mali the gate fires automatically; off
Mali nothing changes. `TTS_CPP_CHBX_CFM_FA=1` forces the old (broken-on-Mali) fused path for an A/B.

## The round — 4 runs, ONE build (turbo T3 pinned to CPU, seed 42)
| label        | what                                   | expected                                                |
|--------------|----------------------------------------|---------------------------------------------------------|
| `r2-gated`   | Mali GPU, no env (the shipping path)   | **clean audio**; config `is_mali=1 cfm_unfused=1`, `f0 bad=0` |
| `r2-forcefa` | Mali GPU, `TTS_CPP_CHBX_CFM_FA=1`      | **broken** (A/B control): `cfm_unfused=0`, f0 explodes   |
| `r2-text2`   | Mali GPU, no env, a longer sentence    | clean (robustness on a different shape)                  |
| `r2-mtl`     | Mali GPU, **mtl** variant, no env      | clean (exercises the B=2 attention path)                 |

`06-run-bug2-round2.sh` runs all four and sets the env for you.

## Steps (run from `packages/tts-ggml`)
```bash
# 1. models — turbo for r2-gated/forcefa/text2; mtl too if you want r2-mtl
node scripts/download-tts-ggml-models.js --group chatterbox,chatterbox-mtl

# 2. build (picks up the ports/ overlay automatically)
bash chatterbox-gpu-verify/01-build-android.sh

# 3. deploy
BARE_CLI=/path/to/android-arm64/bare \
LIBCXX_SO=$ANDROID_NDK/toolchains/llvm/prebuilt/<host>/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so \
bash chatterbox-gpu-verify/02-deploy.sh

# 4. run the round (writes ./chbx-results/r2-*.{wav,result.json,console.txt})
OUT_DIR=./chbx-results bash chatterbox-gpu-verify/06-run-bug2-round2.sh
```
(If you don't have the mtl GGUFs, `r2-mtl` will just warn-and-skip — the other three are the core.)

## What to send back
From `./chbx-results/`, for each of `r2-gated`, `r2-forcefa`, `r2-text2`, `r2-mtl`:
- **`<label>.console.txt`** (holds the `[s3gen-diag] config ...` line + the `f0` trace)
- `<label>.wav`
- `<label>.result.json`

Plus a one-line by-ear verdict per WAV (**clean** / **broken** / **crashed**).

## The single result that matters
**`r2-gated` clean with `is_mali=1 cfm_unfused=1` and `f0 bad=0`** = the auto-gate works on the real
device → PR #67 ships. `r2-forcefa` broken on the same binary is the proof that the gate (not something
else) is what fixes it.
