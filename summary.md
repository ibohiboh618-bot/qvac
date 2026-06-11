# DocTR on Pixel 9 Pro (Mali-G715) — Vulkan Performance Investigation

**Goal:** Reach ~2 s inference for DocTR (`db_mobilenet_v3_large` detector + `crnn_mobilenet_v3_small` recognizer) on the `clinical_chemistry` test image on a Pixel 9 Pro, preferably on Vulkan.

**Status:** Root cause fully isolated and a fix landed. The pipeline went from **10.3 s → 2.6 s warm** on real Mali (**~4×**) via two committed changes: (1) Vulkan op-level optimizations, and (2) an **auto-hybrid** path that runs detection on CPU and recognition on Vulkan, with CPU detection accelerated by **NHWC/KleidiAI conv2d prepacking**. The residual is a **Mali-specific GPU cost on the detector's conv2d dispatches** that resisted every full-Vulkan backend/app-level fix tried — so full-Vulkan 2 s is **not reachable** with available levers. The hybrid is auto-enabled on Mali/Immortalis GPUs only; other GPUs stay full-Vulkan. Warm result **2.6 s** (det 1.35 s CPU+KleidiAI, rec 1.22 s Vulkan), boxes=197 (detections unchanged).

Work branch: `perf/doctr-vulkan-mali` (off `ocr-ggml-doctr-bn-fold`).

---

## 1. Baseline (the problem)

Pixel 9 Pro, Mali-G715, DocTR `clinical_chemistry` — CI run `27284290269`:

| Backend | total | detection | recognition |
|---|---|---|---|
| CPU | 3418 ms | 1289 | 2107 |
| **Vulkan** | **10345 ms** | **5448** | **4871** |

Vulkan was **3× slower than CPU** on Mali. For contrast, on desktop NVIDIA (Vulkan) and Intel iGPU (Vulkan), the same graph is **~2.4–4× *faster* than CPU**. So the graph is fine; the slowness is **Mali-specific**.

---

## 2. What was committed (10.3 s → 6.0 s on Mali)

All changes are Vulkan-only (Metal/CPU paths untouched) and preserve correctness (12/12 expected keywords detected):

1. **`b798cf11` backend-aware recognizer batch** — `recognizerBatchSize` default made backend-aware (Vulkan → 32, Metal/CPU → 4). The recognizer feature extractor was running batches of 4 (a Metal-tuned value), causing many tiny per-crop dispatches. Recognition **4871 → 2227 ms**.
2. **`a4b2c9e9` fused `GGML_OP_CONV_2D` for the detector** — replaced `im2col + mul_mat` (`ggml_conv_2d`) with the fused direct conv (`ggml_conv_2d_direct`) on Vulkan. Detection **5448 → ~4378 ms**.
3. **`5640183b` fused `GGML_OP_CONV_2D` for the recognizer** — same treatment for the CRNN feature extractor.

Measured on real Mali (CI):
| Config | total | detection | recognition |
|---|---|---|---|
| baseline | 10345 | 5448 | 4871 |
| + batching | 7431 | 5160 | 2227 |
| **+ fused conv (both)** | **6001** | **4378** | **1599** |

---

## 3. Root-cause investigation (what's left)

After the committed wins, **detection is the wall**. Key facts established by measurement:

- **Warm vs cold** (load once, run 5×): cold run includes one-time pipeline compilation (~850 ms); warm detection is still **~3.6 s**, warm recognition **~1.2 s**. So the problem is **per-inference**, not cold-start.
- **Per-op GPU profiler** (`GGML_VK_PERF_LOGGER`): detection's conv kernels sum to only **236 ms** of GPU compute (convs run at a dismal 2–14 GFLOP/s on Mali).
- **In-step timing** (`DETPROF`): detection = preprocess 18 ms + **`graph_compute` 3.6 s** + postprocess 5 ms. (`DETPROF2`: input upload 1 ms, so it's all in graph execution.)
- **Inside ggml-vulkan** (`GCPROF`): the detector graph is **369 nodes**; `ggml_backend_vk_graph_compute` itself returns in **2–8 ms** (async: it only records + submits). The 3.6 s is the GPU wait in the subsequent `ggml_backend_synchronize`.

So: **the GPU genuinely spends ~3.6 s executing the detector command buffer, but only 236 ms is "compute" — ~3.4 s is Mali GPU-side overhead the per-op timestamps don't capture.** It scales with op-count × tensor-size (recognition has more ops but on tiny tensors and is fast).

### Hypotheses tested and ruled out (each by a direct experiment on real Mali)

| Hypothesis | Experiment | Result |
|---|---|---|
| Cold-start / pipeline compile | warm runs (run 1–4) | only ~850 ms, one-time |
| Pipeline barriers (L2 flush) | no-op `ggml_vk_sync_buffers` on ARM | **no change** (~150 ms) |
| Submit count (chunked submits) | force single submit (`nodes_per_submit` huge + disable `almost_ready`) | **no change** |
| Pipeline recompilation per call | code review | cached by conv config; not recompiled |
| CPU command recording | `GCPROF` | 2 ms |
| Op-fusion (conv + bias-add + activation) | `FUSION_PROBE`: skip bias-add + activation | **no change** (~50 ms) → **fusion would NOT help** |
| CPU-detection: more threads | hybrid nThreads 4 vs 8 | 8 is *worse* (det 2.9s vs 1.5s; A520 efficiency cores) |
| CPU-detection: KleidiAI (feature only) | enabled `kleidiai` feature, no weight prepack | **no change** (det ~1.47s) — KleidiAI only claims convs whose weights are NHWC-prepacked into its "extra" buffer type **and** that use the fused `GGML_OP_CONV_2D`; the im2col+mul_mat path bypasses it entirely |
| CPU-detection: KleidiAI **+ NHWC prepack** | route regular conv weights through the prepack buft + switch those convs to `GGML_OP_CONV_2D` | **landed** — det **1.48s → 1.35s**, boxes=197 (committed `perf[mod]: NHWC-prepack`). Modest: KleidiAI accelerates the regular convs, but depthwise + FPN/head ops remain the CPU floor |

**Conclusion:** the residual ~3.4 s is intrinsic to Mali executing the detector's conv2d dispatches and is not addressable via barriers, submission strategy, op fusion, or pipeline caching. `shader_core_count` is `0` on Mali (ARM falls through to the placeholder), so conv tile selection is also untuned — but since measured conv *compute* is only 236 ms, tile tuning is a long shot for a 3.4 s gap.

### Cross-device confirmation that it's Mali-specific
Same graph, same `.so` family:
| GPU | warm detection |
|---|---|
| Intel Iris Xe (Vulkan) | 0.26 s |
| Pixel 10 Pro — PowerVR DXT-48 (Vulkan) | **0.78 s** |
| **Pixel 9 Pro — Mali-G715 (Vulkan)** | **3.6 s** |

---

## 4. Tools & infrastructure used for validation

This is the part most reusable for follow-up work.

### 4.1 Real-device test loops (fastest → slowest)
1. **Firebase Test Lab — the fast real-Mali loop (~12 min/round).** Project `qvac-test`, device model **`caiman` = Pixel 9 Pro** (also `tokay` = Pixel 9, `komodo` = 9 Pro XL).
   - Built the test APK from the Expo app `tetherto/qvac-test-addon-mobile` (`npm run build <addon> <addon>/test/mobile` → `expo prebuild` → `./gradlew :app:assembleRelease`, debug-signed, JS bundled).
   - Bundled the two `.gguf` models **as app assets** (dropped them in `<addon>/media/`; the build copies `media/ → testAssets/`, registers them in `assetManifest.js`, metro bundles them — `gguf` is in `assetExts`). This avoids the presigned-S3 model download the CI uses (which needs AWS creds).
   - Patched `app/index.js` to **auto-run** a profile test on launch (`AUTO_PROFILE_BEGIN/END` markers) so a headless `--type robo` run needs no appium UI driver.
   - Ran: `gcloud firebase test android run --type robo --app app-release.apk --device model=caiman,version=35,... --timeout 280s`.
   - Pulled results: `gsutil cp gs://<results-bucket>/<run>/caiman-35-en-portrait/logcat`, then grepped the custom markers (`[WARM]`, `[DETPROF]`, `[GCPROF]`).
   - Auth: `gcloud auth login` with a **`@tether.to` work account** (the personal account had no project access; org tokens expire ~hourly → periodic re-auth).
2. **Pixel 10 Pro over local adb (~minutes/round, but PowerVR not Mali).** Cross-compiled `android-arm64` locally (NDK 27.2), pushed the standalone **`bare-runtime-android-arm64`** binary + a curated bundle (models, image, `node_modules`, `prebuilds/android-arm64`, `libc++_shared.so`) to `/data/local/tmp`, ran `LD_LIBRARY_PATH=… ./bare profile_doctr.js vulkan 0`. Used purely as a *diagnostic* (a second mobile GPU) to prove the slowness is Mali-specific.
3. **Local Intel Iris Xe (Vulkan) proxy (~seconds/round).** Built `ocr-ggml` `linux-x64` with Vulkan and ran `profile_doctr.js`. Bandwidth-bound integrated GPU, good for graph-level correctness + relative wins; **does not reproduce the Mali pathology** (Intel runs the graph fast).
4. **CI — `gh workflow run on-pr-ocr-ggml.yml --ref <branch>` (~1.5 h/round).** Real Pixel 9 Pro via AWS Device Farm. Authoritative but slow (full prebuild matrix + Device Farm queue). Perf data: artifact `perf-report-ocr-Android-<run#>` → `Google_Pixel_9_Pro/performance-report.json`. (The lean `benchmark-performance-ocr-ggml.yml` has a `startup_failure` and was unusable.)

### 4.2 Profiling / instrumentation (added for this work)
- **`GGML_VK_PERF_LOGGER`** (built-in ggml-vulkan per-op GPU timing). Enabled on-device via a compile flag `OCR_VK_PROFILE` that (a) `setenv`s the flag before backend init and (b) routes `GGML_LOG_LEVEL_DEBUG` to logcat in `OcrLazyInitializeBackend.cpp`. Gave the per-op breakdown (conv shapes + GFLOP/s).
- **`DETPROF` / `DETPROF2`** — `std::chrono` timing in `StepDoctrDetectionGGML::process` / `runInference`, logged to logcat (`__android_log_print`), splitting preprocess / graph_compute / postprocess and upload / graph_compute.
- **`GCPROF`** — timing inside `ggml_backend_vk_graph_compute` (record+submit vs `ggml_vk_synchronize`), logged via `GGML_LOG_WARN` (which reaches logcat through the addon's WARN routing). This is what revealed `async=1` + node count 369 + the wait being deferred to `synchronize`.
- **`[WARM]` profiler** — `runDoctrWarmProfile` (new): loads one `OcrGgml` and runs the image N× on Vulkan, logging per-run detection/recognition. Separates cold-start from steady-state. Wired as mobile test `runDoctrWarmTest` (`test/integration/doctr-warm.test.js` + `test/mobile/integration.auto.cjs` + `test-groups.json`).
- **`FUSION_PROBE`** — compile flag that skips bias-add + activation in `convBnAct`/`dwConvBnAct` to estimate the op-fusion ceiling (showed ~0 benefit → fusion ruled out).
- **`profile_doctr.js`** — standalone `bare` harness (CPU/Vulkan, `RUNS`/`BATCH`/`SHOWTEXT` env) used on Intel and on the Pixel 10 over adb.
- **Python aggregators** — parse the perf-logger logcat into per-op-category and top-op tables.

### 4.3 Fast fabric (ggml-vulkan) iteration trick
Rebuilding the ggml fork through vcpkg is ~20 min. Instead: edit `ggml-vulkan.cpp` directly in the vcpkg **buildtree** (`/usr/local/share/vcpkg/buildtrees/qvac-fabric/src/<hash>.clean/...`), incrementally `ninja bin/libqvac-ggml-vulkan.so` in `arm64-android-rel` (~2 min), then hand-copy the `.so` into `prebuilds/android-arm64/qvac__ocr-ggml/` before re-packing the APK. This made the barrier / submit / `GCPROF` experiments feasible.

### 4.4 Build toolchain notes
- `clang-22` required (libc++-22 needs `__builtin_ctzg`); symlinked `clang`/`clang++` → `clang-22` on `PATH` since the toolchains use unversioned `clang`.
- Android: `bare-make generate --platform android --arch arm64 -D ANDROID_STL=c++_shared -D ANDROID_PLATFORM=android-29` (OpenCV needs API ≥ 29; bare-make defaults to 28).
- GH token via `gh auth`; vcpkg binary cache made fabric/opencv reuse fast after first build.

---

## 5. Final numbers (real Mali — Pixel 9 Pro / Firebase `caiman`, warm)

| | detection | recognition | total | boxes |
|---|---|---|---|---|
| baseline Vulkan (run 27284290269) | 5448 | 4871 | 10345 | — |
| Vulkan (committed: batch + fused conv) | ~3.6 s | ~1.2 s | ~4.8 s warm / 6.0 s cold | 197 ✓ |
| CPU | ~1.3 s | ~2.1 s | ~3.4 s | 197 ✓ |
| Hybrid (CPU det + Vulkan rec) — pre-KleidiAI | ~1.48 s | ~1.25 s | ~2.76 s warm | 197 ✓ |
| **Hybrid + NHWC/KleidiAI — validated** | **~1.35 s** | **~1.22 s** | **~2.6 s warm** (3.5 s cold) | **197 ✓** |

Hybrid is implemented (`detectionBackendDevice` param, auto-on-Mali) and the
NHWC/KleidiAI conv2d prepack for CPU detection is committed; both measured on
real Mali: `[WARM:auto] total 2582–2619 ms, det 1343–1370 (CPU+KleidiAI),
rec 1219–1233 (Vulkan), boxes=197`.
**~4× faster than the Vulkan baseline; recognition stays on Vulkan.** The
remaining gap to 2.0 s: warm hybrid is 2.6 s (CPU detection is now the larger
half); cold is ~3.5 s due to the Vulkan recognizer's one-time pipeline
compilation (~860 ms) — a persistent `VkPipelineCache` / load-time warm-up
would bring cold ≈ warm.

---

## 6. Recommendation / next steps

1. **Hybrid + NHWC/KleidiAI landed, auto-on-Mali, validated (~2.6 s warm on real Mali).**
   - `detectionBackendDevice` param + **auto-policy**: a plain `backendDevice:'vulkan'` request on a Mali/Immortalis GPU auto-routes DocTR detection to CPU and keeps recognition on Vulkan; other GPUs stay full-Vulkan; explicit override wins. Validated: `[WARM:auto] det ~1.35s (CPU+KleidiAI) + rec ~1.22s (Vulkan) = ~2.6s, boxes=197`; `[WARM:fullvk] = ~5.1s`. The normal benchmark now reflects this automatically on Mali.
   - **NHWC/KleidiAI conv2d prepack (committed):** regular conv weights are NHWC-packed into the CPU device's KleidiAI prepack buffer type at load and switched to the fused `GGML_OP_CONV_2D`, dropping CPU detection 1.48s → 1.35s with no accuracy change (boxes=197).
   - **nThreads:** tested 4 vs 8 for CPU detection — 8 is *worse* (det 2.9s vs 1.5s; the extra cores are slow A520 efficiency cores). 4 (the X4 + A720 cluster) is optimal.
   - To reach 2.0 s from here the CPU-detection half (~1.35 s) must drop further. KleidiAI now engaged for the regular convs; the remaining floor is the depthwise convs + FPN/head ops (not KleidiAI-accelerated). Bigger levers: **int8-quantize the detector** (KleidiAI i8mm + half the memory traffic — needs a quantized `.gguf` + medical-OCR accuracy check) and/or **further recognizer batching**. Persistent `VkPipelineCache` / load-time warm-up would remove the ~860 ms recognizer cold-start (helps first inference, not warm).
2. **Full-Vulkan 2 s is blocked** on a Mali GPU characteristic (per-conv-dispatch overhead, ~9 ms × the heavy convs, invisible to the per-op profiler). Realistic only via deep Mali-driver-level work or upstream Mali ggml-vulkan support — low odds, large effort.
3. To shave the hybrid toward 2.0 s: int8-quantized detector (KleidiAI i8mm) and/or further recognizer batching; verify clinical-chemistry keyword accuracy on any quantization.

## 7. Open question
The per-op GPU profiler (236 ms) vs the real GPU wait (~3.6 s) discrepancy is the crux. Confirming *physically* what the GPU does in that 3.4 s (e.g., with a Mali profiler such as Arm Streamline / `perfetto` GPU counters) would settle whether it's per-dispatch job-manager overhead, a hidden tile flush, or DVFS, and whether anything can be done at the backend level.
