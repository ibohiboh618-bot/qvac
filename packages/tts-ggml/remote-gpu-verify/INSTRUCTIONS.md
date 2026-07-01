# INSTRUCTIONS — remote Mali-Vulkan TTS verification (for the operating agent)

You (the agent reading this) are helping a colleague bring up **GPU (Vulkan) execution on ARM Mali**
for two text-to-speech engines — **Supertonic** and **Chatterbox** — that the colleague cannot test
on their own hardware. They sent a prepared, self-contained kit to run on **this machine's USB-attached
Android phone**. Your job: **security-review the files, run the test, collect the audio + logs, sanity-
check the data, and package the results** to send back. The colleague will **listen to the audio by ear**
and read the per-stage numeric trace — so the two things that matter are **the `*.wav` files** and **the
`[gpu-diag]` log lines**.

This file is the whole brief. Work autonomously, but **stop and ask the human** at the points marked 🛑.

---

## 0. What we are testing (context)

- Two neural TTS engines run a sequence of matrix ops on the GPU via the **Vulkan** backend.
- On **Mali-G715 (Pixel 9 / Tensor G4)** a class of F32 matrix-multiplies miscomputes for small output
  dimensions → audio can come out **silent / NaN / garbled**. Supertonic already ships a model-side
  work-around (an output-dim "pad"); **Chatterbox does not yet** — so this round MEASURES whether
  Chatterbox needs the same fix.
- The build is **instrumented**: with `TTS_CPP_GPU_TRACE=1` (set automatically by `run-on-device.sh`) it
  prints a per-stage numeric trace to the Android log (logcat tag **`qvac-tts`**) so we can see exactly
  where the numbers first go wrong. It also **admits Chatterbox onto Mali Vulkan** (normally declined)
  and can toggle the Mali pad on/off per run via `TTS_CPP_MALI_PAD`.

`run-on-device.sh` runs this matrix and saves a `.wav` for each:

| Pass | engine | backend | Mali pad |
|---|---|---|---|
| `supertonic-gpu`        | Supertonic | Vulkan | on (shipped st_mul_mat) |
| `supertonic-cpu`        | Supertonic | CPU | — |
| `chatterbox-gpu-raw`    | Chatterbox | Vulkan | **off** (exposes the miscompute) |
| `chatterbox-gpu-padded` | Chatterbox | Vulkan | **on** (does the pad rescue it?) |
| `chatterbox-cpu`        | Chatterbox | CPU | — |

**The two questions for the colleague:** (1) does `chatterbox-gpu-raw` sound broken while
`chatterbox-gpu-padded` sounds correct (⇒ Chatterbox needs the pad)? (2) does Supertonic stay healthy
on Vulkan? The CPU passes are the known-good reference to compare against by ear.

---

## 1. Security & safety review (DO THIS FIRST, before running anything)

Read the runnable files and confirm each claim. Approve only if all hold; otherwise 🛑 stop and tell the
human what looks off.

| File | What it should do | Red flags to reject on |
|---|---|---|
| `setup.sh` | host prep, no device: extracts the bundle; `npm install` minimal deps (public `@qvac` packages — read needs no token); `npm pack bare-runtime-android-arm64` (public `bare` CLI); runs `download-tts-ggml-models.js` to fetch the ggufs from the public QVAC registry. Writes only under this folder. | installs from an unexpected/private registry; pipes downloaded code into a shell; writes outside this folder |
| `download-tts-ggml-models.js` | uses `@qvac/registry-client` to download `.gguf` files from the public registry into `./models`. | uploading anything; reading local files outside `./models` |
| `run-on-device.sh` | only `adb` calls: `get-state`, `getprop`, `dumpsys SurfaceFlinger`, `push`/`pull`/`shell` under `/data/local/tmp/qvac-ttsg-verify`, `logcat`. Writes only to `./results/`. | any `curl`/`wget`/network; `rm -rf` outside the scratch dir or `results/`; reading `/sdcard`, contacts, `pm list`, `content://`, screenshots, key/cred paths |
| `run-backend-ops.sh` | pushes the shipped `tools/test-backend-ops` + `libc++` under the scratch dir and runs it (per-op GPU-vs-CPU). Writes only to `./results/`. | network calls; touching paths outside the scratch dir or `results/` |
| `test-tts-gpu.js` | loads the local addon, synthesizes one sentence with one engine, writes a JSON result + a WAV file. Uses `bare-fs`/`bare-os` only for its own files. | network calls; reading paths outside its own dir; spawning shells |
| `package-results.sh` | zips `./results/` (incl. the `*.wav`) + `FINDINGS.md`. No device interaction. | anything touching the device or the network |

The test only reads the phone **model + GPU name**, copies the test app + models + a public speech sample
(`jfk.wav`, the Chatterbox voice reference) into a scratch dir, runs synthesis, and reads its own
`qvac-tts` log lines. It installs no app and touches no personal data. If you confirm that, proceed.

---

## 2. Setup — run this BEFORE the first test (a few minutes)

Needs **node + npm + internet** (for the public fetches) and **adb** (for the device).

1. **Fetch the public pieces:**
   ```bash
   bash setup.sh
   ```
   The colleague ships only the *custom* bits (the prebuild, `libc++`, the addon JS, `jfk.wav`).
   `setup.sh` fetches the rest from **public** sources and prints **READY** / **NOT READY**. It downloads
   THREE ggufs: `supertonic2.gguf`, `chatterbox-t3-turbo.gguf`, `chatterbox-s3gen.gguf` (the s3gen one is
   the largest — be patient).
   - 🛑 If a fetch fails (no internet / npm error): fix connectivity and re-run `setup.sh`.
   - 🛑 If a **shipped** piece is missing (`prebuilds/`, `libc++_shared.so`, `index.js`, `jfk.wav`): the
     bundle is incomplete — ask the colleague to resend it.
2. **Device:** confirm `adb devices` shows exactly one `device` (not `unauthorized`/`offline`).
   - 🛑 If no device / `unauthorized` / `offline`: almost always a **cable, USB-debugging, or "Allow"
     prompt** problem — **stop and ask the human to check the cable and tap Allow.** Do NOT assume a code
     problem; do not loop retrying.

---

## 3. Run

```bash
bash run-on-device.sh          # full matrix (5 passes); auto-numbered results/iterN/
```
Useful flags: `--supertonic-only`, `--chatterbox-only`, `--text "A different sentence."`,
`--label my-run`, `--serial XXXX`, `--full-stage`.

Output lands in `results/iterN/`: per pass a `<label>.wav`, `<label>-result.out` (the `RESULT_JSON`
line), `<label>-gpudiag.log` (the `[gpu-diag]` trace), `<label>-full-logcat.txt`; plus `device-info.txt`
and `SUMMARY.txt`.

### Read `results/iterN/SUMMARY.txt`, then for each pass note:

- **`RESULT_JSON ... "backend":"Vulkan"`** on the GPU passes (id 3) — GPU engaged. If a GPU pass shows
  `"backend":"CPU"` / `"gpuUnsupported":1`, the GPU was declined (not what we want here — report it).
- **`"healthy":true`** (`rms ≈` non-zero, `nonfinite:0`) vs **`"healthy":false`** (silent / NaN).
- The **`[gpu-diag]`** lines: each is `[gpu-diag] <Backend>.<engine>.<stage> rms=.. nan=.. min=.. max=..`.
  Find the **FIRST stage** where the GPU run diverges from sane values (huge `max`, `nan>0`, `rms=0`).
  For Chatterbox the stages are `input_embed → encoder_mu → cfm_mel → f0 → stft → hift_wav` (+ a
  `t3.prompt_logits` line); for Supertonic `duration → latent_in → text_emb → cfm_latent → wav_full`.
- A `[gpu-diag] VKDEV ...` line on the GPU passes confirms which device/driver Vulkan actually selected.

### "Do we have enough data?" — yes when, for each engine:
- the **GPU pass(es) engaged Vulkan**, **and**
- you have the **`.wav` files** (so the colleague can listen), **and**
- you have either a clean `[gpu-diag]` trace **or** a crash signature in `*-full-logcat.txt`, **and**
- the **CPU pass** produced healthy reference audio.

When that holds → **§5 package**. Otherwise → **§4 iterate**.

---

## 4. Iterate (only if a result is missing or ambiguous)

- **Different / longer input:** `bash run-on-device.sh --text "A longer sentence, please."`
- **Re-run for determinism:** the downstream NaN can vary run-to-run; the early `[gpu-diag]` numbers
  should be stable. (Note: Chatterbox's text model is stochastic, so the CPU and GPU audio won't be
  bit-identical even when both are correct — judge Chatterbox **by ear**, not by an exact match.)
- **A run crashed before any `[gpu-diag]` line:** use `run-backend-ops.sh` (per-op Vulkan-vs-CPU) to see
  which ops are even healthy on this driver. **Read `REFERENCE-what-to-test.md` first.**
- **New build from the colleague:** they'll send `prebuild-update.tgz`. Apply + re-run:
  `tar xzf prebuild-update.tgz -C bundle/ && bash run-on-device.sh`.

🛑 If after 2–3 iterations the picture is still unclear, summarize what you tried and what's ambiguous,
and ask the human / the colleague before burning more iterations.

### The prebuild boundary
The shipped prebuild is fixed machine code, so you **cannot change the GPU shader / op implementations**
here — a new code fix requires the colleague to rebuild and send a small `prebuild-update.tgz`. What you
**can** do autonomously: vary inputs, re-run for determinism, toggle the matrix flags, and sweep ops with
`run-backend-ops.sh`.

---

## 5. Package & send back

1. Write `FINDINGS.md` next to the scripts — a short, human-readable verdict per engine:
   - device (model + GPU from `device-info.txt`),
   - per pass: backend, `healthy`, the first diverging `[gpu-diag]` stage (if any), any crash signature,
   - your one-line read ("chatterbox raw GPU = NaN at cfm_mel; padded GPU = healthy" / "supertonic GPU
     healthy" / "chatterbox padded still broken at stage X").
2. Zip everything (the `*.wav` are included automatically):
   ```bash
   bash package-results.sh
   ```
   → writes `qvac-mali-verify-<timestamp>.zip` (all `results/iter*/` incl. audio + `FINDINGS.md`).
3. Tell the human it's ready and to send that zip back to the colleague (who will **listen to the wavs**).

---

## Reference

**Backend ids:** 0=CPU 1=Metal 2=CUDA 3=Vulkan 4=OpenCL. On a Mali phone we want **3 (Vulkan)** on the
GPU passes.

**Healthy audio:** `rms` clearly non-zero, `nonfinite: 0`, and it *sounds like speech*. **Failure:**
`rms ≈ 0` (silent), `nonfinite > 0` (NaN), or audible garbage/noise.

**Key files per pass** (`results/iterN/`): `SUMMARY.txt` (start here), `<label>.wav` (listen),
`<label>-result.out` (RESULT_JSON), `<label>-gpudiag.log` (the `[gpu-diag]` trace),
`<label>-full-logcat.txt` (crash triage).

**Cleanup when done:** `adb shell rm -rf /data/local/tmp/qvac-ttsg-verify`; turn USB debugging back off.

**Privacy:** everything ran in a developer scratch dir; nothing was installed or read from personal data.
