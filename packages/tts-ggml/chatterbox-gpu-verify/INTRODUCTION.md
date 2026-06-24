# Chatterbox Mali-GPU verify — Bug 2, ROUND 1 (2026-06-24)

**Read this first. It supersedes `INSTRUCTIONS.md` and the previous round's notes.**
The build/deploy *scripts* still apply; the run step is now a single script (`05-run-bug2-round1.sh`).

## What we're chasing (Bug 2 — different from the old 12 kHz CPU tone)
With Chatterbox forced onto the **Mali (Vulkan) GPU**, the audio is clean for the first ~1.3 s and then
**collapses at a fixed point** into a buzzy/“morse” comb for the rest of the clip. It is **clean on
Adreno (OpenCL), on Qualcomm CPU, and on desktop** — so it's a **Mali-Vulkan-specific S3Gen miscompute**.
The garbage starts in the **mel** (produced by the encoder + CFM on the GPU); HiFT is downstream.

**Prime suspect:** the CFM estimator's `flash_attn_ext` (the only flash-attn in the graph) miscomputing
on the Mali f32 FA kernel. This round localizes it AND tries a fix in one device session.

## What this build carries (overlay `ports/tts-cpp`, DO-NOT-MERGE)
Three env-gated changes vs source-of-truth master (all default-off = stock behaviour):
1. `allow_arm_mali=true` — admits Chatterbox onto the Mali Vulkan GPU.
2. `S3GEN_DIAG=1` — prints a per-stage (`mu_T` → `mel` → `f0` → `wav`) per-block rms/min/max +
   NaN/Inf trace to **`<label>.console.txt`**. This is how we localize the first diverging stage.
3. `S3GEN_FIX=cfm_unfused` — swaps the CFM flash-attn for the encoder's soft_max+matmul (Mali-correct)
   formulation. The **fix-swing**: if the FA kernel is the bug, this run should be clean.

## The round — 3 runs, ONE build (turbo, T3 pinned to CPU, seed 42)
| label           | S3Gen backend | extra env              | purpose                                  |
|-----------------|---------------|------------------------|------------------------------------------|
| `r1-base`       | **Mali GPU**  | `S3GEN_DIAG=1`         | reproduce the break + localize the stage |
| `r1-cfmunfused` | **Mali GPU**  | `+ S3GEN_FIX=cfm_unfused` | **fix-swing** — does the break vanish? |
| `r1-cpuref`     | CPU           | `S3GEN_DIAG=1`         | known-good per-stage trace to compare    |

`05-run-bug2-round1.sh` runs all three and sets the env for you — no need to set anything by hand.

## Prerequisites (host that builds)
- Android NDK, `bare`, `bare-make`, vcpkg toolchain (same as any tts-ggml arm64-android build).
- vcpkg must be able to fetch a private GitHub repo (the overlay pulls `tetherto/qvac-ext-lib-whisper.cpp`
  by commit — same `GH_TOKEN` / GitHub auth as registry builds).
- An authorized Mali/Pixel device on `adb`. `npm install` already run in `packages/tts-ggml`.

## Steps (run from `packages/tts-ggml`)
```bash
# 1. models (turbo Chatterbox only this round)
node scripts/download-tts-ggml-models.js --group chatterbox

# 2. build the arm64-android prebuild (picks up ports/ overlay automatically)
bash chatterbox-gpu-verify/01-build-android.sh

# 3. deploy to the device
BARE_CLI=/path/to/android-arm64/bare \
LIBCXX_SO=$ANDROID_NDK/toolchains/llvm/prebuilt/<host>/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so \
bash chatterbox-gpu-verify/02-deploy.sh

# 4. run the 3-run round (writes ./chbx-results/r1-*.{wav,result.json,console.txt,gpudiag.txt})
OUT_DIR=./chbx-results bash chatterbox-gpu-verify/05-run-bug2-round1.sh
```

## What to send back
From `./chbx-results/`, for **each** of `r1-base`, `r1-cfmunfused`, `r1-cpuref`:
- **`<label>.console.txt`** ← most important (holds the `[s3gen-diag]` per-stage/per-block trace)
- `<label>.wav`
- `<label>.result.json`

…plus a one-line by-ear verdict per WAV (**clean** / **breaks partway** / **crashed**). We'll do the
numeric analysis from the `.console.txt` traces. (No need to interpret them yourself.)

## Quick sanity (optional, helps us trust the run)
- In each `r1-base` / `r1-cpuref` `console.txt` there should be a line `[s3gen-diag] config ... backend_cpu=…`.
  `r1-base` should show `backend_cpu=0` (ran on GPU); `r1-cpuref` `backend_cpu=1`. If `r1-base` shows
  `backend_cpu=1`, the GPU backend `.so` wasn't found — flag it (the BACKENDS_DIR/prebuilds path), that run
  is not a GPU test.
