# TTS — ARM Mali/Tensor verify round (2026-06-24)

**Read this first. It supersedes `INSTRUCTIONS.md`** (that file is an older conv_transpose probe and
expects `[gpu-diag]` logcat lines — this build has NO `[gpu-diag]`, so ignore those parts). The
build/deploy/run *scripts* still apply.

## Background

Chatterbox emits a constant high-pitched **~12 kHz tone** over the speech, **only on Mali/Pixel
(Google Tensor) devices**. Qualcomm/Adreno devices are clean. On shipped `main`, Chatterbox on a
Mali device runs **entirely on the CPU**. The single thing that differs between toney-Tensor-CPU and
clean-Adreno-CPU is **which aarch64 CPU variant runs**: `GGML_CPU_ALL_VARIANTS=ON` ships per-arch
`.so` variants and a loader picks the highest the device supports — **Tensor → an armv9 SVE variant**,
**Qualcomm/Adreno → the NEON armv8.6 variant**. So the prime suspect is an **SVE/armv9 CPU-kernel bug**.

This round gets, in ONE device session, **4 diagnostics**:

1. **Chatterbox CPU, default variant (SVE/armv9)** — the toney control.
2. **Chatterbox CPU, forced NEON (armv8.6)** — **THE DECISIVE TEST.** Same device, same model, only
   the CPU variant changes. If the tone disappears under NEON, it's confirmed an SVE/armv9 kernel bug.
3. **Chatterbox on the Mali GPU** — sanity-check the GPU path (see the caveat below).
4. **Supertonic 2 + Supertonic 3 on the Mali GPU** — fresh audio to confirm they're correct on Mali.

For each, just report: **clean**, **~12 kHz tone**, or **crash / no audio**.

### Caveat on the GPU runs (#3)
Vulkan only accepts `CONV_TRANSPOSE_1D` with **F32** operands, but the default S3Gen GGUF's HiFT
upsample is **F16**. So even on the "GPU" run the vocoder conv_transpose will **stay on CPU (or the
op is skipped)** — the GPU run is a useful sanity/abort check, but it is **not** a clean test of the
vocoder on the GPU. The decisive signal this round is the **CPU SVE-vs-NEON** pair (#1 vs #2).

## What this build is

A throwaway overlay (`ports/tts-cpp`, DO-NOT-MERGE) whose only change vs source-of-truth master is
`allow_arm_mali=true` for Chatterbox (so it *can* run on the Mali GPU). Supertonic was already true.
ggml-speech is unchanged from the registry.

## Prerequisites (host that builds)

- Android NDK, `bare`, `bare-make`, vcpkg toolchain (same as any tts-ggml arm64-android build).
- **vcpkg must be able to fetch a private GitHub repo** — the overlay pulls
  `tetherto/qvac-ext-lib-whisper.cpp` by commit (same vcpkg GitHub auth / `GH_TOKEN` as registry builds).
- An authorized Mali/Pixel device on `adb`.
- `npm install` already run in `packages/tts-ggml`.

## Steps (run from `packages/tts-ggml`)

```bash
# 1. models
node scripts/download-tts-ggml-models.js --group chatterbox,supertonic2,supertonic3

# 2. build the arm64-android prebuild — picks up the ports/ overlay automatically
bash chatterbox-gpu-verify/01-build-android.sh

# 3. deploy to the device
BARE_CLI=/path/to/android-arm64/bare \
LIBCXX_SO=$ANDROID_NDK/toolchains/llvm/prebuilt/<host>/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so \
bash chatterbox-gpu-verify/02-deploy.sh

# 4. CPU variant pair — THE DECISIVE TEST (same device, only the CPU variant differs)
OUT_DIR=./chbx-results bash chatterbox-gpu-verify/03-run-config.sh turbo cpu cpu turbo-cpu-sve   # default = armv9 SVE (control)
OUT_DIR=./chbx-results bash chatterbox-gpu-verify/03c-run-cpu-neon.sh turbo turbo-cpu-neon       # forced armv8.6 NEON

# 5. Chatterbox on the Mali GPU (sanity / abort check — see caveat)
OUT_DIR=./chbx-results bash chatterbox-gpu-verify/03-run-config.sh turbo cpu gpu turbo-s3gen-gpu
OUT_DIR=./chbx-results bash chatterbox-gpu-verify/03-run-config.sh turbo gpu gpu turbo-t3-gpu

# 6. Supertonic 2 and 3 on the Mali GPU
OUT_DIR=./chbx-results bash chatterbox-gpu-verify/03b-run-supertonic.sh v2 gpu supertonic2-gpu
OUT_DIR=./chbx-results bash chatterbox-gpu-verify/03b-run-supertonic.sh v3 gpu supertonic3-gpu
```

`03c-run-cpu-neon.sh` hides the `armv9.*` variant `.so` on the device (rename `.so`→`.so.off`) so the
loader can only pick `armv8.6` (NEON), runs, then restores them. It prints the remaining cpu `.so`
before the run — confirm no `armv9` is listed there.

## Config map

| label             | engine       | backend / variant                  | role                                          |
|-------------------|--------------|------------------------------------|-----------------------------------------------|
| `turbo-cpu-sve`   | Chatterbox   | all CPU, **armv9 SVE** (default)   | **CONTROL** — expect the 12 kHz tone          |
| `turbo-cpu-neon`  | Chatterbox   | all CPU, **armv8.6 NEON** (forced) | **DECISIVE** — clean ⇒ SVE/armv9 kernel bug   |
| `turbo-s3gen-gpu` | Chatterbox   | S3Gen Mali GPU, T3 CPU             | GPU sanity (vocoder conv_transpose stays CPU) |
| `turbo-t3-gpu`    | Chatterbox   | full Mali GPU                      | GPU sanity / abort check                      |
| `supertonic2-gpu` | Supertonic 2 | Mali GPU                           | correct on Mali GPU?                          |
| `supertonic3-gpu` | Supertonic 3 | Mali GPU                           | correct on Mali GPU?                          |

## How to read it

- **`turbo-cpu-sve` toney + `turbo-cpu-neon` clean** → confirmed: an SVE/armv9 CPU-kernel bug. That's
  the win we're hoping for (fix the kernel, or force NEON on Tensor).
- **Both toney** → not a CPU-variant issue → model/build-side; deeper probe needed.
- GPU configs: `result.json` `backendId` should be `"vulkan"`; if a GPU run crashes / yields no WAV,
  that's the Mali Vulkan abort outcome — report it (it's informative).

## What to send back

From `./chbx-results/`, each config's **`.wav` + `.result.json`** (plus `.console.txt` for any crash),
and a one-line verdict per config from listening: **clean**, **~12 kHz tone**, or **crashed**.
The `turbo-cpu-sve` vs `turbo-cpu-neon` pair is the one that decides our next move.
