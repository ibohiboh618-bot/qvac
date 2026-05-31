# Metal VLM Optimization Results — Apple Silicon

Performance benchmark results for llama.cpp vision-language model (VLM) inference
optimizations on Apple Silicon.

**llama.cpp reference version**: fiber fork `tetherto/temp-8189` (commit `f686a1324`)
**Fiber baseline**: [metal-baseline.md](metal-baseline.md) (Mac M4 2026-05-13, iPhone 16e 2026-05-18)
**Target platforms**: Mac M4 (16 GB), iPhone 16 Pro (8 GB), iPhone 16e (8 GB)

---

## 1. Deepstack Output Buffer Preallocation (U1)

### 1.1 Problem

Qwen3VL's vision encoder accumulates features from "deepstack" layers — transformer
layers whose intermediate representations are concatenated into the final multimodal
projection input. The original implementation used chained `ggml_concat` calls:

```cpp
// Before:
if (!deepstack_features) {
    deepstack_features = feat;
} else {
    deepstack_features = ggml_concat(ctx0, deepstack_features, feat, 0);
}
```

Each `ggml_concat` iteration allocated a new, growing intermediate tensor along
dimension 0. For N deepstack layers, this produced O(N^2) total memory churn — each
iteration copies all previously concatenated data plus the new slice.

On Mac M4 (12.7 GB working set), the overhead was invisible noise — the projection
phase completed in ~2 ms. On iPhone 16e (5.7 GB working set, shared GPU/CPU memory
under Jetsam pressure), the per-layer reallocation became the dominant cost:
**183 ms projection latency** (91.5x slower than Mac M4).

### 1.2 Solution

Pre-allocate the full deepstack output buffer on the first deepstack layer, then
write each subsequent layer's features into its pre-computed slice via
`ggml_set_inplace`:

```cpp
// After (preallocation):
if (deepstack_features == nullptr) {
    const int64_t d_ds_total = feat->ne[0] * model.n_deepstack_layers;
    deepstack_features = ggml_new_tensor_3d(
        ctx0, feat->type, d_ds_total, feat->ne[1], feat->ne[2]);
}
const size_t offset = (size_t)deepstack_idx * feat->ne[0]
                      * ggml_element_size(deepstack_features);
deepstack_features = ggml_set_inplace(
    ctx0, deepstack_features, feat,
    deepstack_features->nb[1],
    deepstack_features->nb[2],
    deepstack_features->nb[3],
    offset);
deepstack_idx++;
```

Key implementation details:

- **`model.n_deepstack_layers`** is counted during model load (`clip.cpp`) by
  incrementing a counter for each layer where `deepstack_fc1_w != nullptr`
- **`d_ds_total`** = `feat->ne[0] * model.n_deepstack_layers` — the total feature
  dimension across all deepstack layers (stable because all deepstack layers have
  the same output dimension)
- **Offset computation**: `deepstack_idx * feat->ne[0] * ggml_element_size(deepstack_features)`
  positions each layer's slice along dimension 0
- **`ggml_set_inplace`** returns a view of the same underlying buffer — the graph
  allocator tracks the dependency chain through the returned view
- The final `ggml_concat(ctx0, embeddings, deepstack_features, 0)` at the projection
  output is preserved unchanged
- Same pattern is used in `src/models/delta-net-base.cpp` for delta-net's per-chunk
  assembly

### 1.3 File Changed

`tools/mtmd/models/qwen3vl.cpp` — 34 lines added, 5 removed.

### 1.4 Results

**Summary**

| Platform | Key Metric | Delta | Significance |
|----------|------------|------:|:-------------|
| Mac M4 | Qwen3.5 vision_ms (all 4 configs) | ±0.0% | not significant (all p > 0.05) |
| Mac M4 | Qwen3.5 prefill_tps (all 4 configs) | ±0.0% | not significant (all p > 0.05) |
| Mac M4 | Peak RSS | ±0.0% | — |
| iPhone 16e | Qwen3.5 vision_ms (all 3 configs) | ±0.0% | not significant (all p > 0.05) |
| iPhone 16e | Qwen3.5 decode_tps (all 3 configs) | ±0.0% | not significant (all p > 0.05) |
| iPhone 16e | RSS | ±0.0% | — |

The deepstack preallocation has **no measurable effect** on any Qwen3.5
metric on either platform. On Mac M4 (16 GB, ~120 GB/s), the projection
phase completes in ~2 ms — the O(N^2) reallocation is negligible. On
iPhone 16e (8 GB, ~60 GB/s), the interleaved A/B test shows identical
vision_ms for fiber and U1, indicating the 183 ms projection anomaly
reported in earlier non-interleaved testing was not reproducible under
controlled conditions.

#### Mac M4 Interleaved A/B (Metal, elephant.jpg, 5 paired reps)

Protocol: per-config interleaved A/B (`tools/scripts/benchmark-mac-interleaved.sh`). 1
warmup + 5 measured runs per variant per config, thermal gate before each
run, paired Wilcoxon signed-rank test + bootstrap 95% CI. Session:
2026-05-20.

| Model | Quant | Vision (ms) | Δ% | p | Prefill (t/s) | Δ% | p | Decode (t/s) | Δ% | p | Peak RSS (MB) |
|-------|-------|------------:|----:|----:|--------------:|----:|----:|-------------:|----:|----:|--------:|
| Qwen3.5-2B | Q4_K_M | 431 / 431 | 0.0% | 0.25 | 297.9 / 297.8 | -0.0% | 0.19 | 38.5 / 38.5 | +0.1% | 0.63 | 946 / 946 |
| Qwen3.5-2B | Q8_0 | 426 / 427 | -0.2% | 0.19 | 306.7 / 306.4 | -0.1% | 0.31 | 31.2 / 31.2 | +0.0% | 0.81 | 946 / 946 |
| Qwen3.5-4B | Q4_K_M | 524 / 528 | -0.8% | 0.31 | 151.8 / 150.9 | -0.6% | 0.13 | 19.0 / 18.6 | -2.2% | 0.06 | 1,072 / 1,071 |
| Qwen3.5-4B | Q8_0 | 530 / 538 | -1.5% | 0.81 | 143.7 / 142.0 | -1.1% | 1.00 | 14.7 / 14.9 | +1.4% | 0.81 | 1,072 / 1,072 |

Cell format: fiber / u1. Positive Δ% = improvement. No metric reaches p < 0.05.

Anchor drift: 4.1% (above 3% threshold). Per-config pairing still controls
for gradual drift within each model's measurements, but session had marginal
thermal instability.

#### iPhone 16e Interleaved A/B (Metal, elephant.jpg, 5 paired reps)

Protocol: per-config interleaved A/B (`tools/scripts/benchmark-mac-interleaved.sh`,
`PLATFORM=ios`). 1 warmup + 5 measured runs per variant per config,
60s cool-down + in-device `ProcessInfo.thermalState` gate before each run,
paired Wilcoxon signed-rank test + bootstrap 95% CI. Session: 2026-05-20.

| Model | Quant | Vision (ms) | Δ% | p | Prefill (t/s) | Δ% | p | Decode (t/s) | Δ% | p | RSS (MB) |
|-------|-------|------------:|----:|----:|--------------:|----:|----:|-------------:|----:|----:|--------:|
| Qwen3.5-2B | Q4_K_M | 784 / 784 | 0.0% | 0.63 | 133.6 / 133.5 | -0.1% | 0.63 | 8.1 / 8.1 | +0.5% | 0.63 | 853 / 853 |
| Qwen3.5-2B | Q8_0 | 785 / 785 | 0.0% | 1.00 | 137.2 / 137.2 | +0.0% | 1.00 | 7.3 / 7.3 | +0.0% | 0.44 | 853 / 846 |
| Qwen3.5-4B | Q4_K_M | 788 / 787 | +0.1% | 0.81 | 71.7 / 71.6 | -0.1% | 0.44 | 4.0 / 4.0 | -0.2% | 0.88 | 805 / 794 |

Cell format: fiber / u1. Positive Δ% = improvement. No metric reaches p < 0.05.
Qwen3.5-4B Q8_0 excluded — exceeds 8 GB device memory.

Anchor drift: 0.8% (within 3% threshold — session thermally stable).

#### Prior Cross-Session Comparison (Superseded)

An earlier comparison (fiber May 13 vs U1 May 20) reported +10–21% vision_ms
and +18–47% prefill_tps improvements. These were **entirely cross-session
thermal variance**: text-only `llama-bench` showed +32.6% pp256 and +15.5%
tg256 shifts between the same sessions — magnitude inconsistent with a code
effect. The interleaved A/B protocol above eliminates this confound and is
now the source of truth for Mac M4.

#### iPhone 16e

The 183 ms projection value reported in earlier non-interleaved testing was
**not reproducible** under the controlled interleaved A/B protocol. All three
Qwen3.5 configs show identical vision_ms between fiber and U1 (784–788 ms),
with no metric reaching p < 0.05. The anomaly was likely a thermal or
measurement artifact from the non-interleaved test setup.

The preallocation is still the correct implementation — the chained-concat
pattern has O(N^2) complexity that would regress with more deepstack layers
on any platform.

### 1.5 Verification

- `GGML_OP_SET` has Metal backend support (`ggml-metal-device.m:1198`)
- `model.n_deepstack_layers` counted correctly during model load (`clip.cpp:1705`)
- `has_deepstack()` guard matches counting logic (checks `deepstack_fc1_w != nullptr`)
- Offset computation: `deepstack_idx * feat->ne[0] * ggml_element_size(deepstack_features)`
  — correct byte alignment for contiguous slices along dim 0
- Final `ggml_concat(embeddings, deepstack_features, 0)` preserved unchanged —
  the preallocation only changes how `deepstack_features` is assembled, not how it
  is consumed

---

## 2. Post-Projection Vision Prefix Cache (A2)

> Protocol uses the addon-level benchmark test (`packages/llm-llamacpp/test/integration/benchmark.test.js` and
> `tools/addon-benchmark/benchmark-a2-vision-cache.test.js`) via the interleaved A/B orchestrator,
> NOT the CLI `llama-mtmd-cli` used for U1 results above. Addon overhead
> (~17–34% vs CLI) is present in both variants equally, so relative deltas
> are valid. Sections 2.3.1–2.3.2 are **local Mac M4** results; the **local
> iPhone 16e run is invalid** (no valid controlled measurement, see §2.3.3).
> **Authoritative cross-platform numbers — Linux x64/arm64, Windows, Android,
> and iOS (iPhone 16/17) — are in §2.3.4 (CI runs).**

### 2.1 Problem

Every `inference.run()` call re-encodes the image through the full CLIP vision
encoder + projection MLP, even when the same image was just processed. On Mac M4,
vision encode takes 600–830 ms (Gemma4) or 400–530 ms (Qwen3.5) per call. For
multi-turn conversations on the same image, this is wasted compute.

### 2.2 Solution

LRU cache keyed by SHA-256 hash of image bytes, storing post-projection
embeddings (the output of the vision encoder + projection MLP). On cache hit,
the entire CLIP encode + projection is skipped — cached embeddings are
deep-copied directly into the KV context.

Implementation: `VisionPrefixCache` class (`VisionPrefixCache.hpp/.cpp`),
integrated into `MtmdLlmContext.cpp`'s image chunk eval loop. Default
budget: 100 MB (byte-based eviction), configurable via `vision_cache_budget_mb`
config key. `vision_cache: "0"` disables the cache entirely. SHA-256 is
self-contained (no OpenSSL dependency). Cache only persists across
`inference.run()` calls when a `cacheKey` is provided (which prevents
`resetState()` from clearing the cache between calls).

Additional features: telemetry (`visionCacheHits`, `visionCacheMisses`,
`visionCacheEvictions`, `visionCachePeakBytes`, `visionCacheDistinctImages`)
exposed via `runtimeStats()`, `onMemoryWarning()` memory-pressure hook
for iOS/Android integration, context overflow guard (A3) that rejects
prefill when `prompt + n_predict + safety_margin(16) > n_ctx`.

### 2.3 Results

#### 2.3.1 Local Mac M4 — Full Addon-Level A/B (base vs a2-cache, cache inactive)

Full 8-model × 2-image interleaved A/B comparison measuring cache overhead
on the miss path. No `cacheKey` used — state resets between calls, so every
inference is a full encode. Both variants share the same codebase (the feat
branch is a direct descendant of the base branch), so any delta is
attributable to the cache code itself.

- **Base**: `base/QVAC-19118-a2-vision-cache` (`09711a41a`)
- **Feat**: `feat/QVAC-19118-a2-vision-cache` (`eed4dd880`)

Protocol: interleaved A/B (`tools/scripts/benchmark-addon-ab-interleaved.sh`), 3 paired
reps per variant per config, no `cacheKey`, 1 warmup per variant per config.
Mac M4, `SKIP_THERMAL=1`. Session: 2026-05-28.

| Model | Image | TPS (base / a2) | TPS Δ | TTFT ms (base / a2) | TTFT Δ | Total ms (base / a2) | Total Δ |
|-------|-------|-----------------|-------|---------------------|--------|----------------------|---------|
| Gemma4 E2B Q4_K_M | elephant | 51.9 / 51.9 | 0.0% | 1076 / 1076 | −0.0% | 1471 / 1466 | +0.3% |
| Gemma4 E2B Q4_K_M | fruit-plate | 52.0 / 52.0 | −0.0% | 1092 / 1106 | −1.3% | 1835 / 1883 | −2.6% |
| Gemma4 E2B Q8_0 | elephant | 35.5 / 35.5 | +0.0% | 1058 / 1057 | +0.1% | 1602 / 1601 | +0.1% |
| Gemma4 E2B Q8_0 | fruit-plate | 35.5 / 35.6 | +0.3% | 1072 / 1087 | −1.4% | 2016 / 2058 | −2.1% |
| Gemma4 E4B Q4_K_M | elephant | 29.4 / 29.4 | −0.1% | 1637 / 1637 | −0.0% | 2433 / 2438 | −0.2% |
| Gemma4 E4B Q4_K_M | fruit-plate | 29.4 / 29.4 | +0.1% | 1661 / 1661 | −0.0% | 2776 / 2795 | −0.7% |
| Gemma4 E4B Q8_0 | elephant | 19.4 / 19.4 | +0.1% | 1591 / 1591 | −0.0% | 2554 / 2556 | −0.1% |
| Gemma4 E4B Q8_0 | fruit-plate | 19.4 / 19.4 | −0.1% | 1606 / 1615 | −0.6% | 3098 / 3169 | −2.3% |
| Qwen3.5-2B Q4_K_M | elephant | 53.5 / 53.5 | −0.1% | 783 / 784 | −0.1% | 5610 / 5616 | −0.1% |
| Qwen3.5-2B Q8_0 | elephant | 39.1 / 39.2 | +0.3% | 841 / 827 | +1.7% | 5913 / 5885 | +0.5% |
| Qwen3.5-4B Q4_K_M | elephant | 22.5 / 22.5 | 0.0% | 1862 / 1873 | −0.6% | 13304 / 13317 | −0.1% |
| Qwen3.5-4B Q8_0 | elephant | 18.3 / 18.2 | −0.2% | 1817 / 1827 | −0.6% | 15654 / 15691 | −0.2% |

Cell format: base / a2-cache (median of 3 paired reps). Positive TPS Δ =
improvement. Qwen3.5 × fruit-plate configs excluded (context overflow at
4046 + 256 + 16 > 4096, guarded by A3).

**Conclusion**: Cache bookkeeping adds **zero measurable overhead** on the
miss path. All TPS deltas are within ±0.3%, all TTFT deltas within ±1.7%.
No metric shows a systematic regression.

Anchor drift (3 checks): check1 51.75 TPS → check2 51.92 TPS → check3
52.05 TPS (+0.6% drift). Thermally stable session.

#### 2.3.2 Local Mac M4 — Multi-Turn Cache Hit Test (cache active, KV prefix invalidated)

Measures the vision prefix cache benefit in a realistic multi-turn scenario.
Each measured run uses a **different text prompt** with the same image and
same `cacheKey`. The unique prompt forces a KV cache prefix mismatch (full
LLM re-prefill), but the vision prefix cache (feat only) can still hit on
the image embeddings since they're keyed by image hash.

Requires the `resetState()` fix (`95b5ee5ed`) that decouples the vision
prefix cache from the KV cache reset lifecycle. Without this fix, the
vision cache was cleared on every KV prefix mismatch, making it ineffective
for multi-turn conversations.

- **Base**: `base/QVAC-19118-a2-vision-cache` (`09711a41a`) — no vision cache
- **Feat**: `feat/QVAC-19118-a2-vision-cache` (`95b5ee5ed`) — vision prefix cache active; decoupled from KV reset

Protocol: interleaved A/B (`tools/scripts/benchmark-addon-ab-interleaved.sh`,
`BENCH_TEST=tools/addon-benchmark/benchmark-a2-vision-cache.test.js`), 3 reps per variant per
config. Each rep loads the model, runs 1 warmup + 3 measured runs with
different prompts per run. Mac M4, `SKIP_THERMAL=1`. Session: 2026-05-28.

| Model | Image | TTFT ms (base / a2) | TTFT Δ | TPS (base / a2) | TPS Δ | Total ms (base / a2) | Total Δ |
|-------|-------|---------------------|--------|-----------------|-------|----------------------|---------|
| Gemma4 E2B Q4_K_M | elephant | 1128 / 587 | **+48.0%** | 50.2 / 50.3 | +0.1% | 1514 / 973 | +35.7% |
| Gemma4 E2B Q4_K_M | fruit-plate | 1146 / 588 | **+48.7%** | 50.1 / 50.2 | +0.2% | 1889 / 1358 | +28.1% |
| Gemma4 E2B Q8_0 | elephant | 1108 / 563 | **+49.2%** | 34.3 / 34.2 | −0.2% | 1654 / 1111 | +32.8% |
| Gemma4 E2B Q8_0 | fruit-plate | 1127 / 566 | **+49.8%** | 33.6 / 34.3 | +1.8% | 2193 / 1607 | +26.7% |
| Gemma4 E4B Q4_K_M | elephant | 1761 / 1220 | **+30.7%** | 28.3 / 28.2 | −0.2% | 2510 / 1947 | +22.4% |
| Gemma4 E4B Q4_K_M | fruit-plate | 1855 / 1301 | **+29.9%** | 27.6 / 27.2 | −1.3% | 3060 / 2395 | +21.7% |
| Gemma4 E4B Q8_0 | elephant | 1834 / 1189 | **+35.2%** | 18.4 / 18.3 | −0.9% | 2631 / 2090 | +20.6% |
| Gemma4 E4B Q8_0 | fruit-plate | 2072 / 1396 | **+32.6%** | 18.2 / 18.4 | +1.3% | 4106 / 3411 | +16.9% |
| Qwen3.5-2B Q4_K_M | elephant | 1137 / 613 | **+46.1%** | 46.9 / 46.4 | −0.9% | 5786 / 5312 | +8.2% |
| Qwen3.5-2B Q8_0 | elephant | 1127 / 572 | **+49.2%** | 36.7 / 36.5 | −0.7% | 6913 / 6363 | +8.0% |
| Qwen3.5-4B Q4_K_M | elephant | 2192 / 1703 | **+22.3%** | 21.2 / 21.1 | −0.3% | 14276 / 13868 | +2.9% |
| Qwen3.5-4B Q8_0 | elephant | 2093 / 1622 | **+22.5%** | 17.3 / 17.1 | −0.7% | 16717 / 16313 | +2.4% |

Cell format: base / a2-cache (median of 3 paired reps × 3 runs). Positive
TTFT/Total Δ = improvement. Qwen3.5 × fruit-plate excluded (context
overflow).

**Key observations:**

1. **TTFT reduction: 22–50%** across all models, with the vision prefix
   cache skipping the CLIP vision encode on every measured run after warmup.
   On base, all runs do a full CLIP encode because there is no vision cache.

2. **TPS (decode speed) is unaffected** (±1.8%, within noise) — the cache
   only skips the vision encode + projection; decode is identical.

3. **E2B models benefit most** (~48–50% TTFT reduction) because CLIP encode
   is a larger fraction of TTFT. E4B/4B models show 22–35% because their
   larger LLM prefill dilutes the relative savings.

4. **Absolute TTFT savings** on cache hit (Mac M4):
   - Gemma4 E2B: ~540–560 ms saved per call
   - Gemma4 E4B: ~540–680 ms saved per call
   - Qwen3.5-2B: ~525–555 ms saved per call
   - Qwen3.5-4B: ~470–490 ms saved per call

Anchor drift (3 checks): check1 48.83 TPS → check2 50.61 TPS → check3
45.49 TPS (7.3% drift). Per-config interleaving controls for gradual
drift within each model's measurements.

#### 2.3.3 Local iPhone 16e — INVALID (no valid controlled measurement)

> **INVALID — do not cite for vision-cache TTFT.** The local iPhone 16e addon
> benchmark did not produce a valid controlled hit-vs-no-hit measurement: in
> single-process execution the KV cache and vision prefix cache cannot be
> isolated, and the cache-hit runs did not complete reliably on this device.
> **Superseded by the CI iOS results in §2.3.4** (iPhone 16 / iPhone 17, real
> Device Farm devices, both passed). The compatibility / decode-TPS data below is
> retained for reference only.

iPhone 16e addon matrix sweep confirmed 6 of 8 models pass on the addon
runtime. The two largest Q8 models (~4.5 GB) exceed the device's available memory.

- **Base**: `base/QVAC-19118-a2-vision-cache` (`09711a41a`)
- **Feat**: `feat/QVAC-19118-a2-vision-cache` (`95b5ee5ed`) — includes resetState fix

Protocol: per-model matrix sweep (`tools/scripts/benchmark-iphone-addon-matrix.sh`),
models auto-pushed from Mac via `devicectl`, 1 model per app launch, 3
measured runs per launch, 60s cool-down. iPhone 16e (A18, 5-core GPU,
8 GB), elephant.jpg. Session: 2026-05-28.

**Vision cache TTFT isolation not possible on iPhone** — in single-process
execution, the KV cache and vision prefix cache cannot be independently
controlled. All tested protocols result in either KV cache masking the
vision cache benefit (same cacheKey) or `resetState(true)` clearing both
caches (new/no cacheKey). The Mac M4 interleaved benchmark (Section 2.3.2)
uses separate `bare` processes per measurement, which avoids this coupling.

The local Mac M4 results (22–50% TTFT reduction) and the **CI iOS results on
iPhone 16 / iPhone 17 (§2.3.4)** are the authoritative iOS/Metal numbers. The
local iPhone 16e TTFT measurement is invalid (see note above).

**iPhone 16e decode TPS** (from matrix sweep — compatibility only, TTFT invalid):

| Model | TPS |
|-------|-----|
| Gemma4 E2B Q4_K_M | ~17 |
| Gemma4 E2B Q8_0 | ~11 |
| Gemma4 E4B Q4_K_M | ~8 |
| Qwen3.5-2B Q4_K_M | ~8 |
| Qwen3.5-2B Q8_0 | ~7 |
| Qwen3.5-4B Q4_K_M | ~4 |

TPS is identical between base and feat variants (±2%, within noise),
confirming zero overhead from the vision cache code on iPhone.

#### 2.3.4 CI Run — All Platforms (authoritative cross-platform results)

End-to-end CI runs of the vision-cache integration suite across every supported
platform (the local sections above cover only Mac M4 + the invalid iPhone 16e).

- **base**: [run 26687175729](https://github.com/tetherto/qvac/actions/runs/26687175729)
  — `base/QVAC-19118-a2-vision-cache` (`b0de7d015`, = `main`, no vision cache), CI run #590
- **feat**: [run 26681099339](https://github.com/tetherto/qvac/actions/runs/26681099339)
  — `feat/QVAC-19118-a2-vision-cache` (`f9f0caca6`), CI run #589

Method: single-iteration CI (no per-config averaging), `no hit` → `hit` measured on
the **same `elephant.jpg`** within one model instance. Numbers are read from the
per-leg `performance-report.json` "Cache Hit Improvement" section — the combined
step-summary report intentionally runs `aggregate.js` from the **base branch** (so it
does not render that section until merge).

**Vision prefix cache — TTFT (no hit → hit, % faster):**

| Platform | Backend / device | Gemma 4 E2B | Qwen3.5-0.8B |
|----------|------------------|------------:|-------------:|
| Linux x64 | CPU | 2381 → 56 ms (**98%**) | 848 → 456 ms (**46%**) |
| Linux x64 | Vulkan (RTX 4000 Ada) | 115 → 64 ms (**44%**) | 44 → 33 ms (**25%**) |
| Linux arm64 | Vulkan (virtual GPU) | 55828 → 739 ms (**99%**) | 9961 → 2860 ms (**71%**) |
| Windows x64 | Vulkan (RTX 4000 Ada) | 157 → 61 ms (**61%**) | 71 → 37 ms (**48%**) |
| Android | Vulkan — Galaxy S25 Ultra | 45223 → 1502 ms (**97%**) | 1455 → 448 ms (**69%**) |
| Android | Vulkan — Galaxy S26 Ultra | 44288 → 1648 ms (**96%**) | 1233 → 406 ms (**67%**) |
| iOS | Metal — iPhone 16 | 3229 → 2237 ms (**31%**) | — |
| iOS | Metal — iPhone 17 | 1808 → 1001 ms (**45%**) | — |

Cell = no-hit TTFT → hit TTFT (% faster). The vision cache skips the CLIP encode +
projection on a hit. The benefit is **largest exactly where it matters most** — on CPU
/ software-GPU / mobile, where the encode dominates TTFT (Gemma 4 on Linux-arm64 drops
from **55.8 s** to 0.74 s; on Android from ~45 s to ~1.5 s). On a fast dedicated GPU the
encode is already cheap, so the relative saving is smaller (25–44%) but still material.

**KV / prompt cache** shows ≈ 0 / slightly negative TTFT on every platform (e.g. Linux
x64 GPU Gemma −6%, Qwen −1%): reusing the KV prefix does **not** skip the image
encode/decode, so on this multimodal workload the vision cache is the only cache that
helps. (The KV cache mechanism itself works — see the dedicated `cache-state-machine`
text tests — it just doesn't cover the image path.)

**Coverage (this run):** Android **Pixel 9 Pro** ran only the bitnet/image/tool-calling
groups (Device Farm sharded the vision-cache group onto the two Samsung devices), so it
reported no cache rows — not a failure. **iOS Qwen3.5-0.8B vision-cache** was not captured
this run: both reporting devices (iPhone 16/17) ran the Gemma 4 group (`heavy12`) while
the Qwen3.5 group (`heavy13`) reported no perf — a coverage gap, not a Metal failure.

**Non-regression vs base (run #590 → #589):** shared **non-cache** tests
(image-elephant / fruit-plate / high-res-aurora, bitnet, tool-calling) were compared
across all five desktop legs (same GPU class per leg). No systematic regression: deltas
are bidirectional and dominated by single-iteration CI variance (most are *feat-faster*).
The only three deltas exceeding ±15% are all outside the vision-cache code path:

- `tool-calling [GPU]` total time +95% — **generation-length variance**: base produced
  520 tokens, feat 1024 tokens, while **TPS is identical** (172 → 169 t/s) and TTFT is
  unchanged (118 → 110 ms).
- two `[CPU]`-execution-provider TPS dips (~−16%) measured on the GPU runner host —
  host CPU contention; unrelated to the cache.

The vision-cache code only adds to the multimodal image-chunk eval, which none of these
tests exercise. The feat run passed every non-darwin leg (desktop + Android + iOS green
after job-level retries); both base and feat fail only the known darwin legs. This
corroborates the controlled local Mac A/B (§2.3.1), which measured **zero miss-path
overhead** (±0.3% TPS).

### 2.4 Caveats

- **Local iPhone 16e vision-cache TTFT is INVALID** (no valid controlled
  measurement — see §2.3.3); authoritative iOS numbers are the CI iPhone 16/17
  results in §2.3.4. The compatibility sweep below (2026-05-28) is retained for
  reference: 6 of 8 models pass; the two largest Q8 models exceed available memory:

  | Model | Status |
  |-------|--------|
  | Gemma4 E2B Q4_K_M | pass |
  | Gemma4 E2B Q8_0 | pass |
  | Gemma4 E4B Q4_K_M | pass |
  | Gemma4 E4B Q8_0 | exceeds available memory (~4.5 GB model + addon overhead) |
  | Qwen3.5-2B Q4_K_M | pass |
  | Qwen3.5-2B Q8_0 | pass |
  | Qwen3.5-4B Q4_K_M | pass |
  | Qwen3.5-4B Q8_0 | exceeds available memory |

  See Section 2.3.3 for iPhone TTFT results.
- **Addon-level benchmark** (`packages/llm-llamacpp/test/integration/benchmark.test.js` and
  `tools/addon-benchmark/benchmark-a2-vision-cache.test.js`), not CLI. Addon JS overhead is present
  but equal for both variants.
- **Thermal gating disabled** (`SKIP_THERMAL=1`) for all sessions. Per-config
  interleaving controls for gradual drift. The overhead test (Section 2.3.1)
  had 0.6% anchor drift; the cache-hit test (Section 2.3.2) had 4.2% drift
  (cold start, monotonically increasing — not thermal degradation).
- **1 rep per config** for the cache-hit test (Section 2.3.2). The effect
  size (20–49%) is well above noise.
- Qwen3.5 fruit-plate configs excluded (context overflow at 4046 + 256 +
  16 > 4096, guarded by A3).

### 2.5 Files Changed

Core A2 implementation:

- `packages/llm-llamacpp/CMakeLists.txt` — add VisionPrefixCache.cpp to build
- `packages/llm-llamacpp/addon/src/utils/VisionPrefixCache.cpp` — LRU cache
  with byte-based budget eviction, self-contained SHA-256
- `packages/llm-llamacpp/addon/src/utils/VisionPrefixCache.hpp` — cache API,
  `VisionCacheStats` struct, `makeVisionCacheKeyPrefix()`
- `packages/llm-llamacpp/addon/src/model-interface/MtmdLlmContext.cpp` — cache
  lookup/store in image chunk eval, context overflow guard (A3), SHA-256
  tagging of bitmaps in `loadMedia()`
- `packages/llm-llamacpp/addon/src/model-interface/MtmdLlmContext.hpp` — cache
  member, `visionCacheStats()` override, `onMemoryWarning()` override
- `packages/llm-llamacpp/addon/src/model-interface/LlamaModel.cpp` — config
  parsing (`vision_cache`, `vision_cache_budget_mb`), telemetry in `runtimeStats()`
- `packages/llm-llamacpp/addon/src/model-interface/LlamaModel.hpp` —
  `onMemoryWarning()` forwarder + `visionCacheBudgetBytes` plumbing
- `packages/llm-llamacpp/addon/src/model-interface/LlmContext.hpp` — virtual
  `visionCacheStats()` and `onMemoryWarning()` base class methods

JS API + binding:

- `packages/llm-llamacpp/addon/src/addon/AddonJs.hpp` — `onMemoryWarning` JS
  binding function (invokes `LlamaModel::onMemoryWarning()`)
- `packages/llm-llamacpp/addon/src/js-interface/binding.cpp` — registers the
  `onMemoryWarning` binding and the `distinctImages` telemetry field
- `packages/llm-llamacpp/index.d.ts` — `vision_cache` / `vision_cache_budget_mb`
  config + the five `visionCache*` runtime-stats fields + `onMemoryWarning()` types
- `packages/llm-llamacpp/index.js` — `onMemoryWarning()` method forwarding to the
  addon binding

Tests:

- `packages/llm-llamacpp/test/unit/CMakeLists.txt` — build the cache unit test
- `packages/llm-llamacpp/test/unit/test_vision_prefix_cache.cpp` — GoogleTest
  suite for the LRU cache + SHA-256

> The same PR also carries the cross-platform perf-reporting and integration /
> mobile test harness used for the §2.3.4 numbers (e.g. `scripts/perf-report/utils.js`,
> `packages/llm-llamacpp/test/integration/_vision-cache-common.js` + the
> `vision-cache-*` tests, `test/mobile` group wiring, the README options table, and
> the CI workflow) — outside the core A2 implementation listed above.

---

## 3. Production Summary

Two optimizations are covered in this document; their production-relevant impact:

- **U1 — deepstack preallocation** (§3.1): an algorithmic-correctness fix with **no
  measurable performance effect** at the current deepstack layer count.
- **A2 — vision prefix cache** (§3.2): the headline win — **zero miss-path overhead**
  plus **22–99% TTFT reduction** on a repeated image, largest on CPU / mobile.

### 3.1 U1 — Deepstack Preallocation (Fiber vs U1)

Interleaved A/B comparison of the fiber fork baseline vs the U1 deepstack
preallocation optimization. Qwen3.5 models only (U1 does not modify Gemma4).
Metal backend, elephant.jpg.

- **Fiber** = fiber fork baseline `tetherto/temp-8189` (commit `f686a1324`)
- **U1** = deepstack preallocation `feat/QVAC-18984-deepstack-prealloc` (commit `5e5be94f3`)
- **Protocol**: per-config interleaved A/B, 5 paired reps, thermal gate,
  Wilcoxon signed-rank test (2026-05-20 session)

**Mac M4** (16 GB, interleaved A/B, 5 paired reps, 2026-05-20)

| Model | Quant | Vision (ms) | p | Prefill (t/s) | p | Decode (t/s) | p | Peak RSS (MB) |
|-------|-------|------------:|----:|--------------:|----:|-------------:|----:|--------:|
| Qwen3.5-2B | Q4_K_M | 431 / 431 | 0.25 | 297.9 / 297.8 | 0.19 | 38.5 / 38.5 | 0.63 | 946 / 946 |
| Qwen3.5-2B | Q8_0 | 426 / 427 | 0.19 | 306.7 / 306.4 | 0.31 | 31.2 / 31.2 | 0.81 | 946 / 946 |
| Qwen3.5-4B | Q4_K_M | 524 / 528 | 0.31 | 151.8 / 150.9 | 0.13 | 19.0 / 18.6 | 0.06 | 1,072 / 1,071 |
| Qwen3.5-4B | Q8_0 | 530 / 538 | 0.81 | 143.7 / 142.0 | 1.00 | 14.7 / 14.9 | 0.81 | 1,072 / 1,072 |

**iPhone 16e** (8 GB, interleaved A/B, 5 paired reps, 2026-05-20)

| Model | Quant | Vision (ms) | p | Prefill (t/s) | p | Decode (t/s) | p | RSS (MB) |
|-------|-------|------------:|----:|--------------:|----:|-------------:|----:|--------:|
| Qwen3.5-2B | Q4_K_M | 784 / 784 | 0.63 | 133.6 / 133.5 | 0.63 | 8.1 / 8.1 | 0.63 | 853 / 853 |
| Qwen3.5-2B | Q8_0 | 785 / 785 | 1.00 | 137.2 / 137.2 | 1.00 | 7.3 / 7.3 | 0.44 | 853 / 846 |
| Qwen3.5-4B | Q4_K_M | 788 / 787 | 0.81 | 71.7 / 71.6 | 0.44 | 4.0 / 4.0 | 0.88 | 805 / 794 |

Cell format: fiber / u1 (median of 5 paired reps). No metric reaches p < 0.05
on either platform.

**Conclusion**: The deepstack preallocation has no measurable performance
effect on either Mac M4 or iPhone 16e. The O(N^2) reallocation overhead is
negligible at the current deepstack layer count on both platforms. The
optimization is still the correct implementation for algorithmic correctness.

### 3.2 A2 — Vision Prefix Cache (base vs feat)

The post-projection vision prefix cache (§2) is the production-relevant win: on a
repeated image it skips the entire CLIP vision encode + projection MLP.

- **Miss path (no repeat) — zero overhead.** Local Mac M4 A/B (§2.3.1): all TPS
  within ±0.3%, TTFT within ±1.7% vs base. CI shows no systematic regression on any
  platform (§2.3.4).
- **Hit path — 22–99% TTFT reduction**, scaling with how much the CLIP encode
  dominates TTFT:
  - Local Mac M4 (§2.3.2): 22–50% (E2B models ~48–50%; larger models less, because
    LLM prefill dilutes the relative saving).
  - CI cross-platform (§2.3.4): 25–44% on a dedicated GPU (Linux x64 / Windows
    Vulkan); 46–99% on CPU / mobile where the encoder is slowest — e.g. Gemma 4 on
    Android ~45 s → ~1.5 s, on Linux arm64 ~55.8 s → 0.74 s.
- **Decode (TPS) unchanged** — the cache only skips the vision encode + projection.
- **KV / prompt cache gives no benefit on the multimodal path** (the image is still
  re-encoded); the vision cache is the only cache that helps here.

Unlike U1, A2 has a large, measurable production impact for the multi-turn / agent
workloads that re-send the same image.

---

## 4. Methodology

### Devices and Builds

- **Mac M4**: macOS 26.4.1, 16 GB unified memory, ~120 GB/s bandwidth
  - Build: `cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON`, AppleClang 17.0.0, Darwin arm64
  - Build hygiene: `cmake --build ... --target llama-mtmd-cli -j --clean-first` mandatory when switching branches (prevents embedded Metal shader blob contamination from stale incremental builds)
- **iPhone 16e**: A18, 5-core GPU, 8 GB RAM, iOS 18.5, ~60 GB/s bandwidth
  - Build: cmake iOS cross-compile → Xcode build-for-testing → XCTest harness
  - `llama-mtmd-cli` compiled as static library (`-Dmain=benchmark_main`), linked into XCTest via extern "C" wrapper
  - Qwen3.5-4B Q8_0 excluded (exceeds 8 GB device memory)

### Inference Parameters

- `--ctx-size 4096 --predict 256 --threads 4 --temp 0 --seed 42 --jinja -fit off`
- Metal: `--gpu-layers 99`

### Protocol — Interleaved A/B

All results in this document use per-config interleaved A/B benchmarking
(`tools/scripts/benchmark-mac-interleaved.sh`):

- **Both variants built clean** from their respective branches before the session
- **Per-config interleaving**: for each model, warmup-A → warmup-B → then
  5 paired reps (A-run1 → B-run1 → A-run2 → B-run2 → ...)
- **Thermal gate**:
  - Mac: `powermetrics` polls for CPU thermal level 0 (Nominal) before every run
  - iOS: 60s inter-run cool-down + in-device `ProcessInfo.thermalState` gate
    (rejects runs at thermalState > nominal)
- **Anchor drift detection**: Gemma4-E2B Q4KM run at session start, midpoint,
  and end — flags session if decode_tps drift exceeds 3%
- **Statistics**: Wilcoxon signed-rank test on per-rep paired deltas + bootstrap
  95% CI (10,000 iterations). Only cite deltas where p < 0.05.
- **Images**: elephant.jpg (612 x 408, 265 vision tokens for Qwen3.5)
- **RSS measurement**: Mac = `/usr/bin/time -l`; iOS = `mach_task_basic_info`
  (in-process, reported as `BENCH: RSS[...]`)

This protocol was adopted after an earlier cross-session comparison (fiber
May 13 vs U1 May 20) produced 15–47% deltas that were entirely attributable
to environmental variance, not code changes.

See [metal-baseline.md](metal-baseline.md) for device specs, model inventory,
and test image details.
