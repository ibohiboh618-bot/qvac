# INSTRUCTIONS — remote Mali-Vulkan GPU verification (for the operating agent)

You (the agent reading this) are helping a colleague debug a **GPU miscompute** in a text-to-speech
engine that only reproduces on **ARM Mali** GPUs. The colleague cannot reproduce it on their own
hardware, so they sent a prepared, self-contained test to run on **this machine's USB-attached Android
phone**. Your job: **security-review the files, run the test, read the logs, decide whether the data is
conclusive, iterate if not, and package the results** to send back.

This file is the whole brief. Work autonomously, but **stop and ask the human** at the points marked
🛑 below.

---

## 0. What we are testing (context)

- A neural TTS model ("Supertonic") runs a sequence of matrix ops on the GPU via the Vulkan backend.
- On Mali-G715 (Pixel 9 / Tensor G4) one specific matrix-multiply miscomputes → the audio comes out
  **silent or NaN**. On other GPUs it's correct.
- The colleague is trying **fix attempts** in the GPU shader code. Each attempt is a fresh prebuilt
  binary (already in `bundle/prebuilds/`). **Your run tells them whether that attempt worked.**
- The build is instrumented: it prints a per-operation "trisection" to the Android log so we can see
  exactly where the numbers go wrong. You will capture and read those lines.

**The single most important number:** in the GPU log, the line
`[gpu-diag] dprobe_pw1_mulmat ... max=<N>`.
- `max ≈ 2.5` → the fix **worked** (matches a correct GPU).
- `max ≈ 10`  → the bug is **still there** (this attempt didn't fix it).

---

## 1. Security & safety review (DO THIS FIRST, before running anything)

Read the three runnable files and confirm each claim. Approve only if all hold; otherwise 🛑 stop and
tell the human what looks off.

| File | What it should do | Red flags to reject on |
|---|---|---|
| `setup.sh` | host prep, no device: extracts the bundle; `npm install` the minimal deps (public `@qvac` packages — read needs no token); `npm pack bare-runtime-android-arm64` (public `bare` CLI); runs `download-tts-ggml-models.js` to fetch the model from the public QVAC registry. Writes only under this folder. | installs from an unexpected/private registry; pipes downloaded code into a shell; writes outside this folder |
| `download-tts-ggml-models.js` | uses `@qvac/registry-client` to download a `.gguf` from the public registry into `./models`. | uploading anything; reading local files outside `./models` |
| `run-on-device.sh` | only `adb` calls: `get-state`, `getprop`, `dumpsys SurfaceFlinger`, `push`/`pull`/`shell` under `/data/local/tmp/qvac-ttsg-verify`, `logcat`. Writes only to `./results/`. | any `curl`/`wget`/network; any `rm -rf` outside `/data/local/tmp/qvac-ttsg-verify` or the local `results/`; reading `/sdcard`, contacts, `pm list`, `content://`, screenshots, key/cred paths |
| `run-backend-ops.sh` | pushes the shipped `tools/test-backend-ops` + `libc++` under `/data/local/tmp/qvac-ttsg-verify` and runs it (per-op GPU-vs-CPU). Writes only to `./results/`. | network calls; touching paths outside the scratch dir or `results/` |
| `test-supertonic-gpu.js` | loads the local addon, synthesizes one fixed sentence, writes a JSON result file. Uses `bare-fs`/`bare-os` only for its own files. | network calls; reading paths outside its own dir; spawning shells |
| `package-results.sh` | zips `./results/` + `FINDINGS.md`. No device interaction. | anything touching the device or the network |

(`bundle/tools/test-backend-ops` is a standard ggml correctness tester — a native binary; it only does GPU/CPU math and prints pass/fail. `bundle/prebuilds/*` are the addon's native libraries. These are binaries from the colleague's build; trust them at the same level as the rest of the bundle.)

The test only: reads the phone **model + GPU name**, copies the test app + model into a scratch dir,
runs one speech synthesis, and reads its own log lines (tag `qvac-supertonic`). It installs no app and
touches no personal data. If you confirm that, proceed.

---

## 2. Setup — run this BEFORE the first test (a couple of minutes)

Needs **node + npm + internet** (for the public fetches) and **adb** (for the device).

1. **Fetch the public pieces & get ready:**
   ```bash
   bash setup.sh
   ```
   The colleague ships only the *custom* bits (the prebuild, `libc++`, the addon JS). `setup.sh` fetches
   the rest from **public** sources: `npm install` for the deps (the `@qvac` packages are public on npm —
   a token is needed only to *publish*, not to read), the `bare` CLI via `npm pack`, and the model via
   `download-tts-ggml-models.js` (public QVAC registry). It tars `node_modules` for a fast device push and
   prints **READY** / **NOT READY**.
   - 🛑 If a fetch fails (no internet / npm error): fix connectivity and re-run `setup.sh`.
   - 🛑 If a **shipped** piece is missing (`prebuilds/`, `libc++_shared.so`, `index.js`): the bundle is
     incomplete — ask the colleague to resend it.
2. **Device:** confirm `adb devices` shows exactly one `device` (not `unauthorized`/`offline`).
   - 🛑 If no device / `unauthorized` / `offline`: almost always a **cable, USB-debugging, or "Allow"
     prompt** problem — **stop and ask the human to check the cable and tap Allow.** Do NOT assume a
     code problem; do not loop retrying.
3. You don't push anything by hand — `run-on-device.sh` stages everything to the phone (and re-stages
   only the changed prebuild on later runs).

---

## 3. The loop: run → read → decide

### Run an iteration
```bash
bash run-on-device.sh --cpu-baseline        # first iteration: GPU + a CPU reference
```
Output lands in `results/iter1/` (auto-numbered each run): `SUMMARY.txt`, `gpu-result.out`,
`gpu-gpudiag.log`, `gpu-full-logcat.txt`, `cpu-result.out`, `device-info.txt`.

### Read `results/iterN/SUMMARY.txt`, then classify the outcome:

| Observation | Meaning | Do this |
|---|---|---|
| `backend":"Vulkan"` **and** `dprobe_pw1_mulmat max≈2.5` **and** `healthy":true` | **Fix worked.** | Record it, write `FINDINGS.md`, package, done. |
| `backend":"Vulkan"` **and** `dprobe_pw1_mulmat max≈10` (and/or `healthy":false`) | **Fix did NOT work** — bug reproduced. | This is still a clean, conclusive result. Record the full trisection, package, done. |
| `backend":"CPU"` / `gpuUnsupported":1` | GPU was declined → not testing Vulkan. | Note it. The fix build is supposed to run Mali on Vulkan; if it won't, that's a finding — report to the human. Confirm the device really is Mali (`device-info.txt`). |
| No `RESULT_JSON`, or `==== FATAL ====`, or a SIGABRT/abort in `gpu-full-logcat.txt` | The run **crashed** (often the bug itself). | Find the crash signature in `gpu-full-logcat.txt` (the abort line + the op/shader name). That *is* decisive data — record it. |
| `dprobe_*` lines missing but audio produced | Probe didn't fire / different build. | Re-run once; if still missing, report the build may not be the instrumented one. |

### "Do we have enough data?" — yes when:
- The **GPU backend engaged** (Vulkan), **and**
- you have either the **trisection numbers** (a clear fixed/not-fixed verdict) **or** a **crash signature**
  pinning where it dies, **and**
- a **CPU baseline** for comparison (`cpu-result.out`, the known-good rms).

When all three hold → go to **§5 package**. Otherwise → **§4 iterate**.

---

## 4. Iterate (only if a result is missing or ambiguous)

Pick the smallest change that resolves the ambiguity, re-run, and keep each iteration in its own folder.

- **Different / longer input** (more data, different tensor shapes):
  `bash run-on-device.sh --text "A longer sentence with more words to synthesize, please."`
- **Re-run to check determinism** (the bug's downstream effect can vary run-to-run): just run again;
  compare `dprobe_pw1_mulmat max` across iterations (it should be stable; the audio NaN may not be).
- **Inspect the model** if you suspect a bad/missing file: confirm `bundle/models/supertonic2.gguf`
  exists and is non-trivial in size; the colleague can confirm the expected checksum on request.
- **New fix attempt from the colleague:** they'll send `prebuild-update.tgz`. Apply + re-run:
  `tar xzf prebuild-update.tgz -C bundle/ && bash run-on-device.sh`  (base already staged → fast).

🛑 If after 2-3 iterations the picture is still unclear, summarize what you tried and what's ambiguous,
and ask the human / the colleague how to proceed before burning more iterations.

### Per-op verification (test-backend-ops) — available any time, not just a last resort
The bundle ships `test-backend-ops` (in `bundle/tools/`), a standalone tester that compares individual GPU
ops to the CPU reference on-device. `run-backend-ops.sh` drives it — use it to localize WHICH op/shape
diverges, autonomously, **without a new build from the colleague**:
```bash
bash run-backend-ops.sh                # MUL_MAT on Vulkan0 (the prime suspect)
bash run-backend-ops.sh --all          # sweep all ops (slower)
```
**Read `REFERENCE-what-to-test.md` first** — it says what to probe (F32 `MUL_MAT`, contraction K>32) and what
is already cleared (don't re-test im2col / depthwise / coopmat / f16).
- **Use it especially when** the whole-model run crashes before any `[gpu-diag]` line (per-op then tells us
  which ops are even healthy on this driver), or to independently corroborate the mul_mat finding.
- **Hard caveat:** `test-backend-ops` uses RANDOM inputs + a tolerance, and its built-in shapes skew small-N.
  For this exact bug it may **PASS while the real model still miscomputes** (a sibling bug did exactly that).
  So a **FAIL is a decisive localization; a PASS does NOT clear the op** — the real oracle stays the
  whole-model `dprobe_pw1_mulmat` from `run-on-device.sh`.

### The prebuild boundary (what the agent CAN and CAN'T change without the colleague)
The shipped prebuild is fixed machine code, so you **cannot change the GPU shader / op implementations** here
— a new *code* fix (e.g. a different shader) requires the colleague to rebuild and send a small
`prebuild-update.tgz` (cheap: `tar xzf prebuild-update.tgz -C bundle/ && bash run-on-device.sh`). What you
**can** do autonomously, no round-trip: vary inputs (`--text`), re-run for determinism, and sweep ops with
`run-backend-ops.sh`. Localize as much as possible with those, so each rebuild the colleague does is
maximally targeted.

---

## 5. Package & send back

1. Write `FINDINGS.md` next to the scripts — a short, human-readable verdict:
   - device (model + GPU from `device-info.txt`),
   - per-iteration: backend, `dprobe_pw1_mulmat max`, `healthy`, and any crash signature,
   - your one-line conclusion ("fix worked" / "bug still present, mulmat max=10.4" / "crashes at op X").
2. Zip everything:
   ```bash
   bash package-results.sh
   ```
   → writes `qvac-mali-verify-<timestamp>.zip` (all `results/iter*/` + `FINDINGS.md`).
3. Tell the human it's ready and to send that zip back to the colleague.

---

## Reference

**Backend ids:** 0=CPU 1=Metal 2=CUDA 3=Vulkan 4=OpenCL. On a Mali phone we want **3 (Vulkan)**.

**Healthy audio:** `rms ≈ 0.037`, `nonfinite: 0`. **Failure:** `rms ≈ 0`, and/or `nonfinite > 0`.

**Key files per iteration** (`results/iterN/`):
- `SUMMARY.txt` — start here.
- `gpu-result.out` — the JSON line (`RESULT_JSON {...}`): backend, rms, nonfinite, healthy.
- `gpu-gpudiag.log` — the `[gpu-diag]` lines: the `dprobe_*` trisection, `duration raw`, `text_emb`,
  `cfm_latent`, `wav_full`.
- `gpu-full-logcat.txt` — full device log for crash triage.

**Cleanup when done:** `adb shell rm -rf /data/local/tmp/qvac-ttsg-verify`; turn USB debugging back off.

**Privacy:** everything ran in a developer scratch dir; nothing was installed or read from personal data.
