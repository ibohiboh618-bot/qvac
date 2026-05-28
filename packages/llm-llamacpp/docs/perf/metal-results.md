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

Protocol: per-config interleaved A/B (`benchmark-mac-interleaved.sh`). 1
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

Protocol: per-config interleaved A/B (`benchmark-mac-interleaved.sh`,
`PLATFORM=ios`). 1 warmup + 5 measured runs per variant per config,
60s cool-down + in-device `ProcessInfo.thermalState` gate before each run,
paired Wilcoxon signed-rank test + bootstrap 95% CI. Session: 2026-05-20.

| Model | Quant | Vision (ms) | Δ% | p | Prefill (t/s) | Δ% | p | Decode (t/s) | Δ% | p | RSS (MB) |
|-------|-------|------------:|----:|----:|--------------:|----:|----:|-------------:|----:|----:|--------:|
| Qwen3.5-2B | Q4_K_M | 784 / 784 | 0.0% | 0.63 | 133.6 / 133.5 | -0.1% | 0.63 | 8.1 / 8.1 | +0.5% | 0.63 | 853 / 853 |
| Qwen3.5-2B | Q8_0 | 785 / 785 | 0.0% | 1.00 | 137.2 / 137.2 | +0.0% | 1.00 | 7.3 / 7.3 | +0.0% | 0.44 | 853 / 846 |
| Qwen3.5-4B | Q4_K_M | 788 / 787 | +0.1% | 0.81 | 71.7 / 71.6 | -0.1% | 0.44 | 4.0 / 4.0 | -0.2% | 0.88 | 805 / 794 |

Cell format: fiber / u1. Positive Δ% = improvement. No metric reaches p < 0.05.
Qwen3.5-4B Q8_0 excluded — OOM on 8 GB device.

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

> Protocol uses the addon-level benchmark test (`benchmark.test.js` and
> `benchmark-a2-vision-cache.test.js`) via the interleaved A/B orchestrator,
> NOT the CLI `llama-mtmd-cli` used for U1 results above. Addon overhead
> (~17–34% vs CLI) is present in both variants equally, so relative deltas
> are valid. iPhone addon benchmarks blocked by Metal crash in the iOS
> prebuild (see Section 2.4).

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

#### 2.3.1 Full Addon-Level A/B — base vs a2-cache (cache inactive)

Full 8-model × 2-image interleaved A/B comparison measuring cache overhead
on the miss path. No `cacheKey` used — state resets between calls, so every
inference is a full encode. Both variants share the same codebase (the feat
branch is a direct descendant of the base branch), so any delta is
attributable to the cache code itself.

- **Base**: `base/QVAC-19118-a2-vision-cache` (`09711a41a`)
- **Feat**: `feat/QVAC-19118-a2-vision-cache` (`eed4dd880`)

Protocol: interleaved A/B (`benchmark-addon-ab-interleaved.sh`), 3 paired
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

#### 2.3.2 Cache Hit Test — With cacheKey (cache active)

Measures the actual vision cache benefit. Model loaded once, 1 warmup run
(no cacheKey), then 3 measured runs with `cacheKey`. Run 1 = cache miss
(full CLIP encode + projection, result stored). Runs 2–3 = cache hit
(skip encode, reuse cached embeddings).

- **Base**: `base/QVAC-19118-a2-vision-cache` (`09711a41a`) — no vision cache; `cacheKey` only preserves the KV cache
- **Feat**: `feat/QVAC-19118-a2-vision-cache` (`eed4dd880`) — vision prefix cache active; runs 2–3 skip CLIP encode

Protocol: interleaved A/B (`benchmark-addon-ab-interleaved.sh`,
`BENCH_TEST=benchmark-a2-vision-cache.test.js`), 1 rep × 3 sequential
inferences per model load, `cacheKey='bench-vision-cache-session'`.
Mac M4, `SKIP_THERMAL=1`. Session: 2026-05-28.

**Gemma4 — Per-Run TTFT Breakdown (elephant.jpg)**

| Model | Variant | Run 1 (miss) | Run 2 (hit) | Run 3 (hit) | Cache Δ (r1→r2) |
|-------|---------|-------------|-------------|-------------|-----------------|
| E2B Q4_K_M | base | 1125 ms | 1137 ms | 1148 ms | — |
| E2B Q4_K_M | a2-cache | 1122 ms | **583 ms** | **597 ms** | **−48%** |
| E2B Q8_0 | base | 1100 ms | 1131 ms | 1151 ms | — |
| E2B Q8_0 | a2-cache | 1105 ms | **573 ms** | **588 ms** | **−48%** |
| E4B Q4_K_M | base | 1792 ms | 1851 ms | 1885 ms | — |
| E4B Q4_K_M | a2-cache | 1815 ms | **1285 ms** | **1318 ms** | **−29%** |
| E4B Q8_0 | base | 2332 ms | 2412 ms | 2338 ms | — |
| E4B Q8_0 | a2-cache | 2335 ms | **1570 ms** | **1580 ms** | **−33%** |

**Gemma4 — Per-Run TTFT Breakdown (fruitPlate.png)**

| Model | Variant | Run 1 (miss) | Run 2 (hit) | Run 3 (hit) | Cache Δ (r1→r2) |
|-------|---------|-------------|-------------|-------------|-----------------|
| E2B Q4_K_M | base | 1139 ms | 1169 ms | 1187 ms | — |
| E2B Q4_K_M | a2-cache | 1144 ms | **592 ms** | **607 ms** | **−48%** |
| E2B Q8_0 | base | 1125 ms | 1159 ms | 1182 ms | — |
| E2B Q8_0 | a2-cache | 1129 ms | **575 ms** | **591 ms** | **−49%** |
| E4B Q4_K_M | base | 2587 ms | 2631 ms | 2652 ms | — |
| E4B Q4_K_M | a2-cache | 2684 ms | **1966 ms** | **1898 ms** | **−27%** |
| E4B Q8_0 | base | 2187 ms | 2482 ms | 2302 ms | — |
| E4B Q8_0 | a2-cache | 2213 ms | **1636 ms** | **1609 ms** | **−26%** |

**Qwen3.5 — Per-Run TTFT Breakdown (elephant.jpg)**

| Model | Variant | Run 1 (miss) | Run 2 (hit) | Run 3 (hit) | Cache Δ (r1→r2) |
|-------|---------|-------------|-------------|-------------|-----------------|
| 2B Q4_K_M | base | 1264 ms | 1279 ms | 1271 ms | — |
| 2B Q4_K_M | a2-cache | 1268 ms | **687 ms** | **657 ms** | **−46%** |
| 2B Q8_0 | base | 1210 ms | 1205 ms | 1215 ms | — |
| 2B Q8_0 | a2-cache | 1210 ms | **621 ms** | **625 ms** | **−49%** |
| 4B Q4_K_M | base | 2243 ms | 2306 ms | 2333 ms | — |
| 4B Q4_K_M | a2-cache | 2280 ms | **1824 ms** | **1830 ms** | **−20%** |
| 4B Q8_0 | base | 2212 ms | 2188 ms | 2231 ms | — |
| 4B Q8_0 | a2-cache | 2208 ms | **1718 ms** | **1724 ms** | **−22%** |

**Key observations:**

1. **Run 1 TTFT is identical** between base and a2-cache (cache miss — full
   CLIP encode on both). This confirms the cache miss path adds no overhead.

2. **Runs 2–3 on a2-cache skip the vision encode**, cutting TTFT by 20–49%
   depending on model size. On base, all 3 runs do a full encode regardless
   of `cacheKey` (no `VisionPrefixCache` on base — `cacheKey` only preserves
   the KV cache there).

3. **TPS (decode speed) is unaffected** — the cache only skips the vision
   encode + projection; decode is identical.

4. **E2B models benefit most** (~48–49% TTFT reduction) because the CLIP
   encode is a larger fraction of TTFT. E4B and 4B models show 20–33%
   because their larger LLM prefill dilutes the cache savings.

5. **Absolute TTFT savings** on cache hit (Mac M4):
   - Gemma4 E2B: ~530–560 ms saved per call
   - Gemma4 E4B: ~530–850 ms saved per call
   - Qwen3.5-2B: ~580–590 ms saved per call
   - Qwen3.5-4B: ~470–490 ms saved per call

Anchor drift (3 checks): check1 48.16 TPS → check2 49.84 TPS → check3
50.17 TPS (+4.2% drift). Session started cold; TPS increased as system
warmed. Per-config pairing still controls for gradual drift within each
model's measurements.

### 2.4 Caveats

- **Mac M4 only** — iPhone 16e addon benchmarks blocked by Metal
  `ggml_metal_synchronize` SIGABRT in the addon's iOS prebuild (see
  Section 14.5 of metal-benchmarking-plan.md). CLI-based iPhone benchmarks
  (U1, Section 1) passed on the same device. Expected cache savings on
  iPhone are larger in absolute terms (vision encode takes 927–1285 ms on
  iPhone 16e vs 400–830 ms on Mac M4).
- **Addon-level benchmark** (`benchmark.test.js` and
  `benchmark-a2-vision-cache.test.js`), not CLI. Addon JS overhead is present
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

- `packages/llm-llamacpp/CMakeLists.txt` — add VisionPrefixCache.cpp to build
- `packages/llm-llamacpp/addon/src/model-interface/LlamaModel.cpp` — config
  parsing (`vision_cache`, `vision_cache_budget_mb`), telemetry in runtimeStats
- `packages/llm-llamacpp/addon/src/model-interface/LlmContext.hpp` — virtual
  `visionCacheStats()` and `onMemoryWarning()` base class methods
- `packages/llm-llamacpp/addon/src/model-interface/MtmdLlmContext.cpp` — cache
  lookup/store in image chunk eval, context overflow guard (A3), SHA-256
  tagging of bitmaps in `loadMedia()`
- `packages/llm-llamacpp/addon/src/model-interface/MtmdLlmContext.hpp` — cache
  member, `visionCacheStats()` override, `onMemoryWarning()` override
- `packages/llm-llamacpp/addon/src/utils/VisionPrefixCache.cpp` — LRU cache
  with byte-based budget eviction, self-contained SHA-256
- `packages/llm-llamacpp/addon/src/utils/VisionPrefixCache.hpp` — cache API,
  `VisionCacheStats` struct, `makeVisionCacheKeyPrefix()`
- `packages/llm-llamacpp/addon/src/js-interface/binding.cpp` — `onMemoryWarning`
  JS binding, `distinctImages` telemetry field

---

## 3. Production Summary — Fiber vs U1

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

---

## 4. Methodology

### Devices and Builds

- **Mac M4**: macOS 26.4.1, 16 GB unified memory, ~120 GB/s bandwidth
  - Build: `cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON`, AppleClang 17.0.0, Darwin arm64
  - Build hygiene: `cmake --build ... --target llama-mtmd-cli -j --clean-first` mandatory when switching branches (prevents embedded Metal shader blob contamination from stale incremental builds)
- **iPhone 16e**: A18, 5-core GPU, 8 GB RAM, iOS 18.5, ~60 GB/s bandwidth
  - Build: cmake iOS cross-compile → Xcode build-for-testing → XCTest harness
  - `llama-mtmd-cli` compiled as static library (`-Dmain=benchmark_main`), linked into XCTest via extern "C" wrapper
  - Qwen3.5-4B Q8_0 excluded (OOM on 8 GB device)

### Inference Parameters

- `--ctx-size 4096 --predict 256 --threads 4 --temp 0 --seed 42 --jinja -fit off`
- Metal: `--gpu-layers 99`

### Protocol — Interleaved A/B

All results in this document use per-config interleaved A/B benchmarking
(`benchmark-mac-interleaved.sh`):

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
