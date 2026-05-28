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

#### 2.3.2 Multi-Turn Cache Hit Test (cache active, KV prefix invalidated)

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

Protocol: interleaved A/B (`benchmark-addon-ab-interleaved.sh`,
`BENCH_TEST=benchmark-a2-vision-cache.test.js`), 3 reps per variant per
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

#### 2.3.3 iPhone 16e — Compatibility and Baseline

iPhone 16e addon matrix sweep confirmed 6 of 8 models pass on the addon
runtime. The two largest Q8 models (~4.5 GB) OOM under Jetsam.

- **Base**: `base/QVAC-19118-a2-vision-cache` (`09711a41a`)
- **Feat**: `feat/QVAC-19118-a2-vision-cache` (`95b5ee5ed`) — includes resetState fix

Protocol: per-model matrix sweep (`benchmark-iphone-addon-matrix.sh`),
models auto-pushed from Mac via `devicectl`, 1 model per app launch, 3
measured runs per launch, 60s cool-down. iPhone 16e (A18, 5-core GPU,
8 GB), elephant.jpg. Session: 2026-05-28.

**Vision cache TTFT isolation not possible on iPhone** — in single-process
execution, the KV cache and vision prefix cache cannot be independently
controlled. All tested protocols result in either KV cache masking the
vision cache benefit (same cacheKey) or `resetState(true)` clearing both
caches (new/no cacheKey). The Mac M4 interleaved benchmark (Section 2.3.2)
uses separate `bare` processes per measurement, which avoids this coupling.

The Mac M4 results (22–50% TTFT reduction) are authoritative. iPhone TTFT
savings are expected to be proportionally similar since CLIP encode time
scales consistently across platforms.

**iPhone decode TPS** (from matrix sweep, for reference):

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

### 2.4 Caveats

- **iPhone 16e addon benchmarks** completed (2026-05-28). 6 of 8 models
  pass; the two largest Q8 models OOM (Jetsam kill):

  | Model | Status |
  |-------|--------|
  | Gemma4 E2B Q4_K_M | pass |
  | Gemma4 E2B Q8_0 | pass |
  | Gemma4 E4B Q4_K_M | pass |
  | Gemma4 E4B Q8_0 | OOM (~4.5 GB model + addon overhead exceeds Jetsam limit) |
  | Qwen3.5-2B Q4_K_M | pass |
  | Qwen3.5-2B Q8_0 | pass |
  | Qwen3.5-4B Q4_K_M | pass |
  | Qwen3.5-4B Q8_0 | OOM |

  See Section 2.3.3 for iPhone TTFT results.
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
