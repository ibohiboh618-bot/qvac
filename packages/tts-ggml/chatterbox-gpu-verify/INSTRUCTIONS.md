# Chatterbox GPU-vs-CPU on-device verification (QVAC-20557)

Run **Chatterbox** end-to-end on a physical Android device and capture audio + logs, so we can verify the GPU path
**by ear** and from the per-stage `[gpu-diag]` trace. **CPU is the known-good reference; the GPU run is what we verify.**

This is **Chatterbox-only** (the runner never loads Supertonic). The runner is a thin wrapper over the package's existing
synth helpers (`test/utils/runChatterboxTTS.js`); one synthesis per process.

> **Verified end-to-end on 2026-06-23** on an Adreno 740 (SM8550) device: build → deploy → on-device GPU synth →
> pullable WAV + `result.json` + tag-filtered `[gpu-diag]` all work. The branch already pins the Mali fix
> (`ports/ggml-speech` → qvac-ext-ggml `25655933`; `ports/tts-cpp` → `63f25312`, both Chatterbox stages `allow_arm_mali=true`).

---

## How Chatterbox works (so the matrix makes sense)

Chatterbox is **one model made of two stages run in sequence** (never in parallel):

1. **T3** — text → speech tokens (autoregressive).
2. **S3Gen** — speech tokens → waveform (vocoder).

The model is *stored* as **two `.gguf` files** (T3 weights + S3Gen weights); **both are required** to make audio. You
select the model with a **single** input — `CHBX_VARIANT=turbo|mtl` — and the runner derives both filenames. The
`S3Gen/T3` axis only chooses **which stage runs on GPU vs CPU** within that one synthesis.

GGUFs per variant (must be present in `MODEL_DIR` on the device):
- **turbo**: `chatterbox-t3-turbo.gguf`, `chatterbox-s3gen.gguf`
- **mtl**:   `chatterbox-t3-mtl.gguf`,   `chatterbox-s3gen-mtl.gguf`

---

## The matrix — 8 samples = 2 (CPU/GPU) × 2 (S3Gen/T3) × 2 (turbo/MTL)

Backend placement: `useGPU = (S3GEN_BACKEND==gpu)` (both stages attempt GPU); `T3_BACKEND=cpu` adds
`TTS_CPP_T3_FORCE_CPU=1` (pins T3 to CPU). There is **no** S3Gen-force-CPU lever, so `(T3=gpu, S3Gen=cpu)` is
unreachable — and unneeded, since CPU is the reference.

| label              | CHBX_VARIANT | T3_BACKEND | S3GEN_BACKEND | placement (T3,S3Gen) | what it checks |
|--------------------|--------------|------------|---------------|----------------------|----------------|
| turbo-s3gen-cpu    | turbo        | cpu        | cpu           | (CPU,CPU)            | all-CPU reference |
| turbo-s3gen-gpu    | turbo        | cpu        | gpu           | (CPU,GPU)            | **S3Gen on GPU** (T3 fixed CPU → deterministic). By ear should **match** turbo-s3gen-cpu. |
| turbo-t3-cpu       | turbo        | cpu        | gpu           | (CPU,GPU)            | same placement as turbo-s3gen-gpu (identical audio). |
| turbo-t3-gpu       | turbo        | gpu        | gpu           | (GPU,GPU)            | **T3 on GPU**, end-to-end. T3 is stochastic → judge coherence vs CPU, not a match. |
| mtl-s3gen-cpu / -gpu / t3-cpu / t3-gpu | mtl | (same four) | | | same four for MTL |

`*-t3-cpu` ≡ `*-s3gen-gpu` (same placement → identical audio); `04-run-all.sh` copies the file so you still get 8 named WAVs.

---

## Prerequisites
- The four GGUFs above (you have them).
- Android NDK + `bare` + `bare-make` + vcpkg toolchain (to build the addon). The vcpkg overlay pins are in `../ports`.
- The Android `bare` CLI: `npm pack bare-runtime-android-arm64@<host-bare-version>` → `package/bin/bare` (must match your
  host `bare --version`).
- The NDK aarch64 `libc++_shared.so` (the addon is built with `c++_shared`).
- `adb` with the device authorized (`adb get-state` = `device`).

## Steps (helper scripts in this folder)

```sh
# 1. build the arm64-android prebuild from the overlay
bash chatterbox-gpu-verify/01-build-android.sh

# 2. deploy package + bare CLI + libc++ + models to the device
BARE_CLI=/path/to/package/bin/bare \
LIBCXX_SO="$ANDROID_NDK/toolchains/llvm/prebuilt/<host>/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" \
MODELS_DIR=./models \
bash chatterbox-gpu-verify/02-deploy.sh

# 3. run the whole 8-sample matrix (pulls AUDIO first, then result.json, then logcat, per run)
OUT_DIR=./chbx-results bash chatterbox-gpu-verify/04-run-all.sh
# ...or a single config:
OUT_DIR=./chbx-results bash chatterbox-gpu-verify/03-run-config.sh turbo gpu gpu turbo-t3-gpu
```

Each run produces, under `OUT_DIR` (default `./chbx-results`):
- `<label>.wav` — the audio (LISTEN to it).
- `<label>.result.json` — `backendDevice / backendId / gpuUnsupported / realTimeFactor / sampleCount / durationMs / passed`.
- `<label>.gpudiag.txt` — the per-stage `[gpu-diag]` trace (tag-filtered `qvac-chatterbox`).
- `<label>.console.txt` — native stdout (`ggml` init, `BENCH: S3GEN_INFER_MS/AUDIO_MS`, backend selection).

Zip `<label>.wav` + `<label>.gpudiag.txt` (+ console/result) per sample and send back.

## How to read the results
- **GPU actually engaged?** `result.json` → `backendDevice` must be **non-zero** for any `*-gpu` run (`0` = CPU; the
  runner also prints a WARNING if GPU was requested but the backend `.so` wasn't found). The `[gpu-diag]` backend prefix
  also shows the path (`OpenCL`/`Vulkan` vs `CPU`).
- **By ear:** S3Gen-GPU vs the matching CPU sample should sound the **same** (identical T3-CPU tokens). T3-GPU is
  stochastic ⇒ judge naturalness/coherence (no silence/garble), not a sample-for-sample match.
- **Per-stage `[gpu-diag]`:** every stage should read `nan=0`. (Observed-benign exception: `hift.y_postdiv` may show
  `nan=1` from a window-edge divide; it is cleared by `hift_wav` which must be `nan=0`.)

## Notes / gotchas (all handled by the scripts)
- The runner writes a **result file** because bare's JS `console.log` is swallowed over `adb shell` (native `ggml`
  stdout does come through). Native `[gpu-diag]` goes to **logcat** (`__android_log`, tag `qvac-chatterbox`) — capture it
  **tag-filtered** (`adb logcat -d -s 'qvac-chatterbox:*'`) or the driver spam rolls it out.
- The runner calls `proc.exit(0)` after writing artifacts (bare can otherwise hang on exit with the GPU context open);
  `03-run-config.sh` also wraps the run in a `timeout` (default 300 s) as a hard safety net.
- Push `node_modules` as the bundled tarball (done by `02-deploy.sh`) — a raw `adb push` of ~30k files drops the USB link.

_DO-NOT-MERGE diagnostic harness._
