# QVAC TTS — Mali/Vulkan GPU verification kit

Thanks for running this! It checks whether two text-to-speech engines compute correctly on your phone's
**ARM Mali GPU (Vulkan)**. **The primary target this round is Chatterbox.** Everything runs in a developer
scratch dir over `adb`; nothing is installed as an app and no personal data is touched.

## Prereqs
- **node + npm** and **internet** (to fetch public pieces + the models)
- **adb** (Android platform-tools) and a **USB-attached phone** with USB debugging enabled
- macOS/Linux shell

## Quickstart (Chatterbox — the main target)

```bash
# 1) one-time host setup: fetches the bare CLI, npm deps, and DOWNLOADS THE MODELS
#    (supertonic2.gguf + chatterbox-t3-turbo.gguf + chatterbox-s3gen.gguf, ~1.3 GB
#    chatterbox; from the PUBLIC QVAC registry — no login/token needed)
bash setup.sh

# 2) plug in the phone, confirm it's visible
adb devices            # must show exactly one "device"

# 3) run the Chatterbox passes (raw GPU, padded GPU, CPU) — saves a .wav for each
bash run-on-device.sh --chatterbox-only

# 4) package everything (logs + the .wav audio) to send back
bash package-results.sh     # writes qvac-mali-verify-<timestamp>.zip
```

Then send the `qvac-mali-verify-*.zip` back. (To also test Supertonic, run `bash run-on-device.sh` with no
flags for the full matrix.)

## What you get back per run (`results/iterN/`)
- `chatterbox-gpu-raw.wav`, `chatterbox-gpu-padded.wav`, `chatterbox-cpu.wav` — **listen to these**
- `*-gpudiag.log` — the per-stage `[gpu-diag]` numeric trace (where the math first goes wrong, if it does)
- `*-result.out`, `*-full-logcat.txt`, `SUMMARY.txt`, `device-info.txt`

## The question we're answering
Does **`chatterbox-gpu-raw`** sound broken (silent / noise / garbled) while **`chatterbox-gpu-padded`**
sounds correct? The `cpu` one is the known-good reference. If raw is broken and padded matches CPU, then
Chatterbox needs the Mali "pad" fix (which Supertonic already ships).

## How to download the models manually (if `setup.sh` step 1 fails)
`setup.sh` runs this for you, but you can run it directly (after `npm install` inside `bundle/`):
```bash
cd bundle
node download-tts-ggml-models.js --group chatterbox --output ./models   # t3-turbo + s3gen
node download-tts-ggml-models.js --group supertonic2 --output ./models  # supertonic (optional)
```

## More detail
- `INSTRUCTIONS.md` — the full operating brief (security review, iteration, packaging).
- `REFERENCE-what-to-test.md` — the bug background + what to probe per-op.

If anything is unclear or a step fails, capture the error and send it back rather than guessing.
