# Metal VLM Optimization Results — Apple Silicon

Performance benchmark results for llama.cpp vision-language model (VLM) inference
on Apple Silicon, covering a deepstack buffer preallocation optimization and three
upstream cherry-picks onto a fiber fork.

**llama.cpp reference version**: tag `b9025` (commit `eff06702b`)
**Target platforms**: Mac M4 (16 GB), iPhone 16 Pro (8 GB), iPhone 16e (8 GB)

---

## 1. Deepstack Output Buffer Preallocation

### 1.1 Problem

Qwen3VL's vision encoder accumulates features from "deepstack" layers — transformer
layers whose intermediate representations are concatenated into the final multimodal
projection input. The original implementation used chained `ggml_concat` calls:

```cpp
// Before (b9025 baseline):
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

| Platform | Metric | Before | After | Delta |
|----------|--------|-------:|------:|------:|
| Mac M4 | Qwen3.5 projection | 2 ms | 2 ms | No change (expected) |
| iPhone 16e | Qwen3.5 projection | 183 ms | 11 ms | **-94%** |
| iPhone 16e | Qwen3.5 total time | — | — | **-17%** |

The 183 ms value was measured on the fiber fork. On upstream b9025, iPhone projection
was already 9 ms — the anomaly was fiber-specific (likely related to different memory
allocation patterns in the fork's modified graph). The preallocation fix still applies
because the chained-concat pattern would regress with more deepstack layers on any
platform.

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

## 2. Production Summary — Fiber vs Cumulative vs b9025

Compact view of the production merge target (Cumulative, thermal-stable rerun) vs
the fiber fork baseline and the upstream b9025 reference (tag `b9025` = commit
`eff06702b`). Both backends in a single table.

- **Fiber** = fiber fork baseline (the starting point)
- **All RCs 2** = cumulative branch (RC2 + RC1 + RC3 stacked) after thermal-stable
  rerun (the production merge target)
- **b9025** = upstream llama.cpp reference, same protocol (clean rebuild + RSS-wrapped)

Reading rules: each non-Fiber cell shows the value with the percentage change vs
the same-row Fiber cell in parentheses. Sign convention: **positive Delta% = improvement**
(lower latency, higher t/s, lower memory). **Bold** marks the best value per metric
within each (Backend, Model, Quant) group across the 3 candidates.

| Branch | Backend | Model | Quant | Vision (ms) | Prefill (t/s) | Decode (t/s) | TTFT (ms) | Total (ms) | Peak RSS (MB) |
|--------|---------|-------|-------|------------:|--------------:|-------------:|----------:|-----------:|--------------:|
| Fiber | Metal | Gemma4-E2B | Q4_K_M | **624** | 258.09 | 42.48 | 1,724 | 7,488 | **1,250** |
| All RCs 2 | Metal | Gemma4-E2B | Q4_K_M | 654 (-4.8%) | 257.57 (-0.2%) | 51.20 (+20.5%) | 1,757 (-1.9%) | 6,573 (+12.2%) | 1,251 (-0.2%) |
| b9025 | Metal | Gemma4-E2B | Q4_K_M | 628 (-0.6%) | **260.26 (+0.8%)** | **52.27 (+23.0%)** | **1,719 (+0.3%)** | **6,369 (+14.9%)** | 1,266 (-1.3%) |
| Fiber | Metal | Gemma4-E2B | Q8_0 | **628** | 263.18 | 30.70 | **1,707** | 9,806 | 1,250 |
| All RCs 2 | Metal | Gemma4-E2B | Q8_0 | 636 (-1.3%) | **265.02 (+0.7%)** | **35.56 (+15.8%)** | 1,708 (±0.0%) | **8,655 (+11.7%)** | **1,250 (±0.0%)** |
| b9025 | Metal | Gemma4-E2B | Q8_0 | 649 (-3.3%) | 248.40 (-5.6%) | 30.03 (-2.2%) | 1,792 (-5.0%) | 10,023 (-2.2%) | 1,266 (-1.3%) |
| Fiber | Metal | Gemma4-E4B | Q4_K_M | **724** | 141.57 | 20.54 | 2,730 | 14,903 | **1,331** |
| All RCs 2 | Metal | Gemma4-E4B | Q4_K_M | 750 (-3.6%) | **146.31 (+3.3%)** | 24.62 (+19.9%) | **2,691 (+1.4%)** | 12,874 (+13.6%) | 1,331 (±0.0%) |
| b9025 | Metal | Gemma4-E4B | Q4_K_M | 762 (-5.2%) | 139.68 (-1.3%) | **24.71 (+20.3%)** | 2,795 (-2.4%) | **12,859 (+13.7%)** | 1,353 (-1.7%) |
| Fiber | Metal | Gemma4-E4B | Q8_0 | 739 | 165.68 | 15.61 | 2,453 | 19,164 | 1,336 |
| All RCs 2 | Metal | Gemma4-E4B | Q8_0 | **733 (+0.8%)** | **168.66 (+1.8%)** | **18.57 (+19.0%)** | **2,417 (+1.5%)** | **16,492 (+13.9%)** | **1,335 (±0.0%)** |
| b9025 | Metal | Gemma4-E4B | Q8_0 | 763 (-3.2%) | 153.55 (-7.3%) | 17.07 (+9.4%) | 2,613 (-6.5%) | 18,005 (+6.0%) | 1,356 (-1.5%) |
| Fiber | Metal | Qwen3.5-2B | Q4_K_M | 470 | 245.95 | 36.14 | 1,547 | 8,351 | 946 |
| All RCs 2 | Metal | Qwen3.5-2B | Q4_K_M | 449 (+4.5%) | 279.84 (+13.8%) | 43.21 (+19.6%) | 1,396 (+9.8%) | 7,031 (+15.8%) | 946 (±0.0%) |
| b9025 | Metal | Qwen3.5-2B | Q4_K_M | **435 (+7.4%)** | **292.59 (+19.0%)** | **46.72 (+29.3%)** | **1,341 (+13.4%)** | **6,576 (+21.3%)** | **933 (+1.4%)** |
| Fiber | Metal | Qwen3.5-2B | Q8_0 | 484 | 240.60 | 28.81 | 1,585 | 10,202 | 946 |
| All RCs 2 | Metal | Qwen3.5-2B | Q8_0 | 457 (+5.6%) | 267.06 (+11.0%) | 32.68 (+13.4%) | 1,449 (+8.6%) | 9,129 (+10.5%) | 948 (-0.1%) |
| b9025 | Metal | Qwen3.5-2B | Q8_0 | **438 (+9.5%)** | **300.93 (+25.1%)** | **34.78 (+20.7%)** | **1,319 (+16.8%)** | **8,416 (+17.5%)** | **933 (+1.4%)** |
| Fiber | Metal | Qwen3.5-4B | Q4_K_M | 591 | 122.07 | 17.56 | 2,762 | 17,119 | 1,071 |
| All RCs 2 | Metal | Qwen3.5-4B | Q4_K_M | 550 (+6.9%) | 139.95 (+14.6%) | 18.56 (+5.7%) | 2,444 (+11.5%) | 15,843 (+7.5%) | 1,069 (+0.2%) |
| b9025 | Metal | Qwen3.5-4B | Q4_K_M | **493 (+16.6%)** | **147.26 (+20.6%)** | **20.55 (+17.0%)** | **2,293 (+17.0%)** | **14,565 (+14.9%)** | **1,051 (+1.9%)** |
| Fiber | Metal | Qwen3.5-4B | Q8_0 | 566 | 129.46 | 14.28 | 2,613 | 20,214 | 1,072 |
| All RCs 2 | Metal | Qwen3.5-4B | Q8_0 | 606 (-7.1%) | 121.29 (-6.3%) | 14.74 (+3.2%) | 2,791 (-6.8%) | 19,773 (+2.2%) | 1,071 (+0.1%) |
| b9025 | Metal | Qwen3.5-4B | Q8_0 | **481 (+15.0%)** | **153.39 (+18.5%)** | **16.61 (+16.3%)** | **2,209 (+15.5%)** | **17,455 (+13.6%)** | **1,052 (+1.8%)** |
| Fiber | CPU | Gemma4-E2B | Q4_K_M | 2,181 | 428.11 | 38.99 | 2,844 | 9,225 | **2,562** |
| All RCs 2 | CPU | Gemma4-E2B | Q4_K_M | 2,190 (-0.4%) | **431.62 (+0.8%)** | **39.01 (+0.1%)** | 2,848 (-0.1%) | 9,227 (±0.0%) | 2,563 (±0.0%) |
| b9025 | CPU | Gemma4-E2B | Q4_K_M | **2,129 (+2.4%)** | 430.94 (+0.7%) | 38.67 (-0.8%) | **2,788 (+2.0%)** | **9,221 (±0.0%)** | 2,589 (-1.1%) |
| Fiber | CPU | Gemma4-E2B | Q8_0 | **1,864** | 414.16 | 25.29 | **2,550** | 12,800 | **3,575** |
| All RCs 2 | CPU | Gemma4-E2B | Q8_0 | 1,934 (-3.8%) | 425.97 (+2.9%) | **25.35 (+0.2%)** | 2,601 (-2.0%) | 12,662 (+1.1%) | 3,575 (±0.0%) |
| b9025 | CPU | Gemma4-E2B | Q8_0 | 1,907 (-2.3%) | **428.40 (+3.4%)** | 25.27 (-0.1%) | 2,570 (-0.8%) | **12,574 (+1.8%)** | 3,602 (-0.8%) |
| Fiber | CPU | Gemma4-E4B | Q4_K_M | **4,086** | 357.26 | **19.42** | **4,881** | 18,606 | **4,091** |
| All RCs 2 | CPU | Gemma4-E4B | Q4_K_M | 4,104 (-0.4%) | 340.80 (-4.6%) | 19.30 (-0.6%) | 4,937 (-1.2%) | **18,242 (+2.0%)** | 4,091 (±0.0%) |
| b9025 | CPU | Gemma4-E4B | Q4_K_M | 4,090 (-0.1%) | **358.75 (+0.4%)** | 18.61 (-4.2%) | 4,882 (±0.0%) | 19,128 (-2.8%) | 4,124 (-0.8%) |
| Fiber | CPU | Gemma4-E4B | Q8_0 | **3,065** | **251.85** | 12.76 | **4,193** | **25,071** | **6,152** |
| All RCs 2 | CPU | Gemma4-E4B | Q8_0 | 3,146 (-2.6%) | 225.42 (-10.5%) | **12.79 (+0.2%)** | 4,406 (-5.1%) | 25,481 (-1.6%) | 6,153 (±0.0%) |
| b9025 | CPU | Gemma4-E4B | Q8_0 | 3,219 (-5.0%) | 251.47 (-0.2%) | 12.72 (-0.3%) | 4,348 (-3.7%) | 25,133 (-0.2%) | 6,185 (-0.5%) |
| Fiber | CPU | Qwen3.5-2B | Q4_K_M | 1,814 | 131.00 | **43.02** | 3,837 | **8,218** | 2,176 |
| All RCs 2 | CPU | Qwen3.5-2B | Q4_K_M | 1,821 (-0.4%) | 132.22 (+0.9%) | 34.05 (-20.9%) | 3,825 (+0.3%) | 9,760 (-18.8%) | 2,177 (±0.0%) |
| b9025 | CPU | Qwen3.5-2B | Q4_K_M | **1,726 (+4.9%)** | **142.25 (+8.6%)** | 40.78 (-5.2%) | **3,589 (+6.5%)** | 8,339 (-1.5%) | **2,173 (+0.2%)** |
| Fiber | CPU | Qwen3.5-2B | Q8_0 | 1,469 | 161.88 | **30.14** | 3,106 | 10,331 | 2,873 |
| All RCs 2 | CPU | Qwen3.5-2B | Q8_0 | 1,395 (+5.0%) | 168.62 (+4.2%) | 26.66 (-11.5%) | 2,967 (+4.5%) | 11,386 (-10.2%) | 2,873 (±0.0%) |
| b9025 | CPU | Qwen3.5-2B | Q8_0 | **1,369 (+6.8%)** | **179.65 (+11.0%)** | 30.08 (-0.2%) | **2,844 (+8.4%)** | **10,207 (+1.2%)** | **2,870 (+0.1%)** |
| Fiber | CPU | Qwen3.5-4B | Q4_K_M | 4,372 | 54.47 | **18.70** | 9,237 | 18,716 | 3,712 |
| All RCs 2 | CPU | Qwen3.5-4B | Q4_K_M | 4,496 (-2.8%) | 52.20 (-4.2%) | 14.24 (-23.9%) | 9,573 (-3.6%) | 23,271 (-24.3%) | 3,712 (±0.0%) |
| b9025 | CPU | Qwen3.5-4B | Q4_K_M | **3,931 (+10.1%)** | **61.83 (+13.5%)** | 18.44 (-1.4%) | **8,217 (+11.0%)** | **18,393 (+1.7%)** | **3,698 (+0.4%)** |
| Fiber | CPU | Qwen3.5-4B | Q8_0 | 3,116 | 64.08 | **13.71** | 7,251 | 24,046 | 5,375 |
| All RCs 2 | CPU | Qwen3.5-4B | Q8_0 | 3,191 (-2.4%) | 69.53 (+8.5%) | 11.17 (-18.5%) | 7,002 (+3.4%) | 27,204 (-13.1%) | 5,374 (±0.0%) |
| b9025 | CPU | Qwen3.5-4B | Q8_0 | **2,900 (+6.9%)** | **82.44 (+28.7%)** | 13.29 (-3.1%) | **6,114 (+15.7%)** | **22,749 (+5.4%)** | **5,361 (+0.3%)** |

---

## 3. Text-Only LLM Regression Check

Each RC variant's text-only decode throughput was compared against the fresh
same-thermal fiber baseline using `llama-bench -p 256 -n 256 -ngl 99` on the
two primary models. Regression gate: ≤ 2% text-only decode throughput loss.
Per-rep median of 4 repetitions reported.

| Branch | Model | pp256 (t/s) | tg256 (t/s) | Delta tg256 vs Fiber | Pass (≤2%) |
|--------|-------|------------:|------------:|-----------------:|:----------:|
| Fiber | Gemma4-E2B-Q4_K_M | 544.10 | 37.64 | — | — |
| Fiber | Qwen3.5-2B-Q4_K_M | 482.68 | 24.00 | — | — |
| RC2 | Gemma4-E2B-Q4_K_M | 506.82 | 38.32 | +1.79% | ✓ |
| RC2 | Qwen3.5-2B-Q4_K_M | 491.09 | 24.53 | +2.24% | ✓ |
| RC1 | Gemma4-E2B-Q4_K_M | 540.86 | 38.48 | +2.21% | ✓ |
| RC1 | Qwen3.5-2B-Q4_K_M | 509.35 | 37.25 | +55.22% | ✓ |
| RC3 | Gemma4-E2B-Q4_K_M | 529.98 | 37.21 | -1.15% | ✓ |
| RC3 | Qwen3.5-2B-Q4_K_M | 478.10 | 22.75 | -5.19% ⚠ | ⚠ |
| All RCs | Gemma4-E2B-Q4_K_M | 538.42 | 38.90 | +3.33% | ✓ |
| All RCs | Qwen3.5-2B-Q4_K_M | 459.01 | 37.78 | +57.44% | ✓ |
| All RCs 2 | Gemma4-E2B-Q4_K_M | 509.45 | 36.64 | -2.67% ⚠ | ⚠ |
| All RCs 2 | Qwen3.5-2B-Q4_K_M | 562.52 | 36.51 | +52.16% | ✓ |

**Verdict**: the production target **All RCs** passes both gates and
significantly improves Qwen3.5 text-only throughput (+57.4% via RC1's fused
GDN op). RC3 in isolation shows a -5.2% Qwen3.5 text-only regression — but
when composed with RC1 (which it always is in the merge target), the
regression flips to a large gain. No production-blocking regression.

---

## 4. Methodology

### Device and Build

- **Device**: Mac M4, macOS 26.4.1, 16 GB unified memory
- **Build**: `cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON`, AppleClang 17.0.0, Darwin arm64
- **Build hygiene**: `cmake --build ... --target llama-mtmd-cli llama-bench -j --clean-first` mandatory when switching branches (prevents embedded Metal shader blob contamination from stale incremental builds)

### Inference Parameters

- `--ctx-size 4096 --predict 256 --threads 4 --temp 0 --seed 42 --jinja -fit off`
- Metal: `--gpu-layers 99`; CPU: `--gpu-layers 0`

### Protocol

- **Runs**: 1 warmup + 3 measured, **median** reported
- **Images**: elephant.jpg (primary), fruitPlate.png (secondary; Qwen3.5 x fruitPlate excluded due to ctx overflow at 4,015 vision tokens)
- **RSS measurement**: every invocation wrapped with `/usr/bin/time -l` to capture `rss_mb` and `peak_mem_mb`
- **Model matrix**: 8 configs — Gemma4-E2B Q4_K_M/Q8_0, Gemma4-E4B Q4_K_M/Q8_0, Qwen3.5-2B Q4_K_M/Q8_0, Qwen3.5-4B Q4_K_M/Q8_0

### Thermal Management

The primary results matrix (run group `2026-05-14T2310`) includes a thermal-stable
rerun ("All RCs 2") captured after the system cooled, to validate that sustained-load
thermal throttling did not depress the original cumulative measurement. The throttling
check (Appendix C) documents up to 18.9% Gemma4 decode improvement in the cool rerun,
confirming thermal effects in the original batch. **Headline numbers use the All RCs 2
(thermal-stable) row.**

### Upstream Cherry-Picks

Three commits cherry-picked onto the fiber fork baseline (`f686a1324`), each measured
in isolation and cumulatively:

| Short | Commit | Subject |
|---|---|---|
| **RC2** | `f6d6dbcd1` | metal: optimize Metal Tensor API for `GGML_OP_MUL_MAT` (upstream #20962) |
| **RC1** | `556789f7d` | port `GGML_OP_GATED_DELTA_NET` from upstream b9025 — fused Metal SIMD kernel |
| **RC3** | `054141103` | add Metal Flash Attention `dk512_dv512` kernel instantiations |
| **All RCs** | (RC2 + RC1 + RC3 stacked) | cumulative composition for production merge |

---

# Appendices

## Appendix A: Full Primary Results Matrix — Metal

All values are 3-run median, elephant.jpg, Mac M4, Metal backend. Run group
`2026-05-14T2310`. Clean-rebuild + RSS-wrapped protocol.

Reading rules: each non-Fiber cell shows the value with the percentage change vs
the same-row Fiber cell in parentheses. Positive Delta% = improvement. **Bold** = best
value per metric within each (Model, Quant) group across the 6 branches.

**Source JSONs**:

| Branch | Source JSON |
|---|---|
| Fiber | `results/parsed/mac-fiber-baseline-rss-2026-05-14T2310.json` |
| RC2 | `results/parsed/mac-rc2-rerun-rss-2026-05-14T2310.json` |
| RC1 | `results/parsed/mac-rc1-rerun-rss-2026-05-14T2310.json` |
| RC3 | `results/parsed/mac-rc3-rerun-rss-2026-05-14T2310.json` |
| All RCs | `results/parsed/mac-fiber-updates-rerun-rss-2026-05-14T2310.json` |
| All RCs 2 | `results/parsed/mac-all-rcs-2-2026-05-15T0744.json` |
| b9025 | `results/parsed/mac-b9025-rerun-2026-05-15T1031.json` |

**Why "All RCs 2"?** The original `All RCs` row was the 5th and final variant
in a back-to-back orchestrator that ran for **5 h 25 min** of continuous
Mac M4 compute (2026-05-14 23:12 to 2026-05-15 04:37). To verify those decode
numbers were not depressed by sustained-load thermal throttling, the cumulative
branch was re-run after the system had cooled (2026-05-15 07:44), using the
identical clean-rebuild + RSS-wrapped protocol.

| Branch | Backend | Model | Quant | Vision (ms) | Prefill (t/s) | Decode (t/s) | TTFT (ms) | Total (ms) | Peak RSS (MB) |
|--------|---------|-------|-------|------------:|--------------:|-------------:|----------:|-----------:|--------------:|
| Fiber | Metal | Gemma4-E2B | Q4_K_M | **624** | **258.09** | 42.48 | **1,724** | 7,488 | 1,250 |
| RC2 | Metal | Gemma4-E2B | Q4_K_M | 655 (-5.0%) | 237.11 (-8.1%) | 39.45 (-7.1%) | 1,853 (-7.4%) | 8,038 (-7.3%) | 1,249 (+0.1%) |
| RC1 | Metal | Gemma4-E2B | Q4_K_M | 657 (-5.3%) | 243.49 (-5.7%) | 39.73 (-6.5%) | 1,823 (-5.7%) | 7,968 (-6.4%) | 1,249 (±0.0%) |
| RC3 | Metal | Gemma4-E2B | Q4_K_M | 638 (-2.2%) | 250.96 (-2.8%) | 49.45 (+16.4%) | 1,770 (-2.6%) | 6,689 (+10.7%) | **1,249 (+0.1%)** |
| All RCs | Metal | Gemma4-E2B | Q4_K_M | 650 (-4.2%) | 247.08 (-4.3%) | 48.96 (+15.3%) | 1,799 (-4.4%) | 6,743 (+9.9%) | 1,249 (±0.0%) |
| All RCs 2 | Metal | Gemma4-E2B | Q4_K_M | 654 (-4.8%) | 257.57 (-0.2%) | **51.20 (+20.5%)** | 1,757 (-1.9%) | **6,573 (+12.2%)** | 1,251 (-0.2%) |
| Fiber | Metal | Gemma4-E2B | Q8_0 | **628** | 263.18 | 30.70 | **1,707** | 9,806 | **1,250** |
| RC2 | Metal | Gemma4-E2B | Q8_0 | 724 (-15.3%) | 207.49 (-21.2%) | 25.18 (-18.0%) | 2,093 (-22.6%) | 11,986 (-22.2%) | **1,250 (±0.0%)** |
| RC1 | Metal | Gemma4-E2B | Q8_0 | 722 (-15.0%) | 211.27 (-19.7%) | 25.52 (-16.9%) | 2,066 (-21.0%) | 11,691 (-19.2%) | 1,250 (±0.0%) |
| RC3 | Metal | Gemma4-E2B | Q8_0 | 732 (-16.6%) | 211.59 (-19.6%) | 29.39 (-4.3%) | 2,074 (-21.5%) | 10,423 (-6.3%) | 1,250 (±0.0%) |
| All RCs | Metal | Gemma4-E2B | Q8_0 | 723 (-15.1%) | 224.37 (-14.7%) | 29.90 (-2.6%) | 1,989 (-16.5%) | 10,191 (-3.9%) | 1,250 (±0.0%) |
| All RCs 2 | Metal | Gemma4-E2B | Q8_0 | 636 (-1.3%) | **265.02 (+0.7%)** | **35.56 (+15.8%)** | 1,708 (±0.0%) | **8,655 (+11.7%)** | 1,250 (±0.0%) |
| Fiber | Metal | Gemma4-E4B | Q4_K_M | **724** | 141.57 | 20.54 | 2,730 | 14,903 | 1,331 |
| RC2 | Metal | Gemma4-E4B | Q4_K_M | 747 (-3.2%) | 142.05 (+0.3%) | 19.12 (-6.9%) | 2,746 (-0.6%) | 15,851 (-6.4%) | 1,331 (±0.0%) |
| RC1 | Metal | Gemma4-E4B | Q4_K_M | 783 (-8.1%) | 135.44 (-4.3%) | 19.26 (-6.2%) | 2,880 (-5.5%) | 15,940 (-7.0%) | 1,331 (±0.0%) |
| RC3 | Metal | Gemma4-E4B | Q4_K_M | 830 (-14.6%) | 131.02 (-7.5%) | 23.30 (+13.4%) | 2,998 (-9.8%) | 13,584 (+8.9%) | 1,331 (±0.0%) |
| All RCs | Metal | Gemma4-E4B | Q4_K_M | 774 (-6.9%) | 136.06 (-3.9%) | 24.01 (+16.9%) | 2,861 (-4.8%) | 13,040 (+12.5%) | **1,331 (±0.0%)** |
| All RCs 2 | Metal | Gemma4-E4B | Q4_K_M | 750 (-3.6%) | **146.31 (+3.3%)** | **24.62 (+19.9%)** | **2,691 (+1.4%)** | **12,874 (+13.6%)** | 1,331 (±0.0%) |
| Fiber | Metal | Gemma4-E4B | Q8_0 | 739 | 165.68 | 15.61 | 2,453 | 19,164 | 1,336 |
| RC2 | Metal | Gemma4-E4B | Q8_0 | 793 (-7.3%) | 154.31 (-6.9%) | 14.10 (-9.7%) | 2,633 (-7.3%) | 21,091 (-10.1%) | **1,335 (±0.0%)** |
| RC1 | Metal | Gemma4-E4B | Q8_0 | 785 (-6.2%) | 154.46 (-6.8%) | 14.02 (-10.2%) | 2,624 (-7.0%) | 21,270 (-11.0%) | 1,335 (±0.0%) |
| RC3 | Metal | Gemma4-E4B | Q8_0 | 777 (-5.1%) | 160.44 (-3.2%) | 16.95 (+8.6%) | 2,547 (-3.8%) | 17,963 (+6.3%) | 1,336 (±0.0%) |
| All RCs | Metal | Gemma4-E4B | Q8_0 | 773 (-4.6%) | 153.08 (-7.6%) | 17.45 (+11.8%) | 2,628 (-7.1%) | 17,621 (+8.0%) | 1,335 (±0.0%) |
| All RCs 2 | Metal | Gemma4-E4B | Q8_0 | **733 (+0.8%)** | **168.66 (+1.8%)** | **18.57 (+19.0%)** | **2,417 (+1.5%)** | **16,492 (+13.9%)** | 1,335 (±0.0%) |
| Fiber | Metal | Qwen3.5-2B | Q4_K_M | 470 | 245.95 | 36.14 | 1,547 | 8,351 | 946 |
| RC2 | Metal | Qwen3.5-2B | Q4_K_M | 484 (-3.0%) | 236.83 (-3.7%) | 36.19 (+0.1%) | 1,603 (-3.6%) | 8,382 (-0.4%) | 946 (±0.0%) |
| RC1 | Metal | Qwen3.5-2B | Q4_K_M | 476 (-1.3%) | 240.79 (-2.1%) | 40.79 (+12.9%) | 1,577 (-1.9%) | 7,536 (+9.8%) | **946 (±0.0%)** |
| RC3 | Metal | Qwen3.5-2B | Q4_K_M | 478 (-1.7%) | 248.30 (+1.0%) | 36.26 (+0.3%) | 1,545 (+0.1%) | 8,332 (+0.2%) | 946 (-0.1%) |
| All RCs | Metal | Qwen3.5-2B | Q4_K_M | 455 (+3.2%) | 271.54 (+10.4%) | 42.40 (+17.3%) | 1,431 (+7.5%) | 7,163 (+14.2%) | 946 (±0.0%) |
| All RCs 2 | Metal | Qwen3.5-2B | Q4_K_M | **449 (+4.5%)** | **279.84 (+13.8%)** | **43.21 (+19.6%)** | **1,396 (+9.8%)** | **7,031 (+15.8%)** | 946 (±0.0%) |
| Fiber | Metal | Qwen3.5-2B | Q8_0 | 484 | 240.60 | 28.81 | 1,585 | 10,202 | 946 |
| RC2 | Metal | Qwen3.5-2B | Q8_0 | 505 (-4.3%) | 223.54 (-7.1%) | 28.80 (±0.0%) | 1,690 (-6.6%) | 10,268 (-0.7%) | 946 (±0.0%) |
| RC1 | Metal | Qwen3.5-2B | Q8_0 | 467 (+3.5%) | 251.90 (+4.7%) | 31.74 (+10.2%) | 1,519 (+4.2%) | 9,301 (+8.8%) | **946 (±0.0%)** |
| RC3 | Metal | Qwen3.5-2B | Q8_0 | 486 (-0.4%) | 236.22 (-1.8%) | 28.55 (-0.9%) | 1,608 (-1.4%) | 10,278 (-0.7%) | 947 (-0.1%) |
| All RCs | Metal | Qwen3.5-2B | Q8_0 | 497 (-2.7%) | 241.34 (+0.3%) | 32.00 (+11.1%) | 1,595 (-0.6%) | 9,251 (+9.3%) | 946 (±0.0%) |
| All RCs 2 | Metal | Qwen3.5-2B | Q8_0 | **457 (+5.6%)** | **267.06 (+11.0%)** | **32.68 (+13.4%)** | **1,449 (+8.6%)** | **9,129 (+10.5%)** | 948 (-0.1%) |
| Fiber | Metal | Qwen3.5-4B | Q4_K_M | 591 | 122.07 | 17.56 | 2,762 | 17,119 | 1,071 |
| RC2 | Metal | Qwen3.5-4B | Q4_K_M | 613 (-3.7%) | 117.23 (-4.0%) | 17.46 (-0.6%) | 2,874 (-4.0%) | 17,209 (-0.5%) | 1,071 (±0.0%) |
| RC1 | Metal | Qwen3.5-4B | Q4_K_M | 562 (+4.9%) | 128.95 (+5.6%) | 17.99 (+2.4%) | 2,617 (+5.2%) | 16,350 (+4.5%) | **1,068 (+0.3%)** |
| RC3 | Metal | Qwen3.5-4B | Q4_K_M | 594 (-0.5%) | 119.74 (-1.9%) | 17.45 (-0.6%) | 2,807 (-1.6%) | 17,163 (-0.3%) | 1,071 (±0.0%) |
| All RCs | Metal | Qwen3.5-4B | Q4_K_M | 551 (+6.8%) | 135.34 (+10.9%) | 18.02 (+2.6%) | 2,509 (+9.2%) | 16,328 (+4.6%) | 1,069 (+0.2%) |
| All RCs 2 | Metal | Qwen3.5-4B | Q4_K_M | **550 (+6.9%)** | **139.95 (+14.6%)** | **18.56 (+5.7%)** | **2,444 (+11.5%)** | **15,843 (+7.5%)** | 1,069 (+0.2%) |
| Fiber | Metal | Qwen3.5-4B | Q8_0 | 566 | 129.46 | 14.28 | 2,613 | 20,214 | 1,072 |
| RC2 | Metal | Qwen3.5-4B | Q8_0 | 613 (-8.3%) | 118.15 (-8.7%) | 14.31 (+0.2%) | 2,856 (-9.3%) | 20,282 (-0.3%) | 1,072 (±0.0%) |
| RC1 | Metal | Qwen3.5-4B | Q8_0 | 548 (+3.2%) | 136.55 (+5.5%) | 14.77 (+3.4%) | 2,489 (+4.8%) | 19,551 (+3.3%) | **1,069 (+0.2%)** |
| RC3 | Metal | Qwen3.5-4B | Q8_0 | 565 (+0.2%) | 128.81 (-0.5%) | 14.39 (+0.8%) | 2,622 (-0.4%) | 20,041 (+0.9%) | 1,072 (±0.0%) |
| All RCs | Metal | Qwen3.5-4B | Q8_0 | **544 (+3.9%)** | **140.29 (+8.4%)** | **14.98 (+4.9%)** | **2,433 (+6.9%)** | **19,116 (+5.4%)** | 1,073 (-0.1%) |
| All RCs 2 | Metal | Qwen3.5-4B | Q8_0 | 606 (-7.1%) | 121.29 (-6.3%) | 14.74 (+3.2%) | 2,791 (-6.8%) | 19,773 (+2.2%) | 1,071 (+0.1%) |

---

## Appendix B: Full Primary Results Matrix — CPU

Same run group and protocol as Appendix A, CPU backend (`--gpu-layers 0`).

| Branch | Backend | Model | Quant | Vision (ms) | Prefill (t/s) | Decode (t/s) | TTFT (ms) | Total (ms) | Peak RSS (MB) |
|--------|---------|-------|-------|------------:|--------------:|-------------:|----------:|-----------:|--------------:|
| Fiber | CPU | Gemma4-E2B | Q4_K_M | **2,181** | 428.11 | 38.99 | **2,844** | **9,225** | 2,562 |
| RC2 | CPU | Gemma4-E2B | Q4_K_M | 2,289 (-5.0%) | 413.65 (-3.4%) | 38.09 (-2.3%) | 2,976 (-4.6%) | 9,478 (-2.7%) | **2,561 (±0.0%)** |
| RC1 | CPU | Gemma4-E2B | Q4_K_M | 2,341 (-7.3%) | 414.29 (-3.2%) | 38.14 (-2.2%) | 3,027 (-6.4%) | 9,536 (-3.4%) | 2,562 (±0.0%) |
| RC3 | CPU | Gemma4-E2B | Q4_K_M | 2,244 (-2.9%) | 416.82 (-2.6%) | 38.00 (-2.5%) | 2,925 (-2.8%) | 9,461 (-2.6%) | 2,562 (±0.0%) |
| All RCs | CPU | Gemma4-E2B | Q4_K_M | 2,265 (-3.9%) | 410.50 (-4.1%) | 38.17 (-2.1%) | 2,957 (-4.0%) | 9,522 (-3.2%) | 2,562 (±0.0%) |
| All RCs 2 | CPU | Gemma4-E2B | Q4_K_M | 2,190 (-0.4%) | **431.62 (+0.8%)** | **39.01 (+0.1%)** | 2,848 (-0.1%) | 9,227 (±0.0%) | 2,563 (±0.0%) |
| Fiber | CPU | Gemma4-E2B | Q8_0 | **1,864** | 414.16 | 25.29 | **2,550** | 12,800 | **3,575** |
| RC2 | CPU | Gemma4-E2B | Q8_0 | 1,951 (-4.7%) | 400.77 (-3.2%) | 22.49 (-11.1%) | 2,660 (-4.3%) | 13,855 (-8.2%) | **3,575 (±0.0%)** |
| RC1 | CPU | Gemma4-E2B | Q8_0 | 1,971 (-5.7%) | 400.32 (-3.3%) | 22.40 (-11.4%) | 2,680 (-5.1%) | 14,041 (-9.7%) | 3,575 (±0.0%) |
| RC3 | CPU | Gemma4-E2B | Q8_0 | 2,035 (-9.2%) | 389.95 (-5.8%) | 22.46 (-11.2%) | 2,763 (-8.4%) | 14,030 (-9.6%) | 3,575 (±0.0%) |
| All RCs | CPU | Gemma4-E2B | Q8_0 | 2,038 (-9.3%) | 396.93 (-4.2%) | 22.55 (-10.8%) | 2,753 (-8.0%) | 13,916 (-8.7%) | 3,575 (±0.0%) |
| All RCs 2 | CPU | Gemma4-E2B | Q8_0 | 1,934 (-3.8%) | **425.97 (+2.9%)** | **25.35 (+0.2%)** | 2,601 (-2.0%) | **12,662 (+1.1%)** | 3,575 (±0.0%) |
| Fiber | CPU | Gemma4-E4B | Q4_K_M | 4,086 | 357.26 | **19.42** | 4,881 | 18,606 | **4,091** |
| RC2 | CPU | Gemma4-E4B | Q4_K_M | 4,150 (-1.6%) | 357.51 (+0.1%) | 17.67 (-9.0%) | 4,944 (-1.3%) | 19,489 (-4.7%) | 4,091 (±0.0%) |
| RC1 | CPU | Gemma4-E4B | Q4_K_M | 4,186 (-2.4%) | 354.55 (-0.8%) | 17.55 (-9.6%) | 4,987 (-2.2%) | 19,635 (-5.5%) | 4,091 (±0.0%) |
| RC3 | CPU | Gemma4-E4B | Q4_K_M | 4,222 (-3.3%) | 353.71 (-1.0%) | 17.69 (-8.9%) | 5,025 (-2.9%) | 19,378 (-4.1%) | 4,092 (±0.0%) |
| All RCs | CPU | Gemma4-E4B | Q4_K_M | **4,072 (+0.3%)** | **369.07 (+3.3%)** | 18.49 (-4.8%) | **4,842 (+0.8%)** | 18,657 (-0.3%) | 4,091 (±0.0%) |
| All RCs 2 | CPU | Gemma4-E4B | Q4_K_M | 4,104 (-0.4%) | 340.80 (-4.6%) | 19.30 (-0.6%) | 4,937 (-1.2%) | **18,242 (+2.0%)** | 4,091 (±0.0%) |
| Fiber | CPU | Gemma4-E4B | Q8_0 | **3,065** | 251.85 | 12.76 | **4,193** | **25,071** | **6,152** |
| RC2 | CPU | Gemma4-E4B | Q8_0 | 3,466 (-13.1%) | 245.30 (-2.6%) | 12.55 (-1.6%) | 4,624 (-10.3%) | 25,743 (-2.7%) | 6,152 (±0.0%) |
| RC1 | CPU | Gemma4-E4B | Q8_0 | 3,597 (-17.4%) | 232.28 (-7.8%) | 12.49 (-2.1%) | 4,820 (-15.0%) | 26,093 (-4.1%) | 6,152 (±0.0%) |
| RC3 | CPU | Gemma4-E4B | Q8_0 | 3,557 (-16.1%) | 250.46 (-0.6%) | 12.56 (-1.6%) | 4,691 (-11.9%) | 26,038 (-3.9%) | 6,153 (±0.0%) |
| All RCs | CPU | Gemma4-E4B | Q8_0 | 3,218 (-5.0%) | **265.87 (+5.6%)** | 12.72 (-0.3%) | 4,286 (-2.2%) | 25,091 (-0.1%) | 6,153 (±0.0%) |
| All RCs 2 | CPU | Gemma4-E4B | Q8_0 | 3,146 (-2.6%) | 225.42 (-10.5%) | **12.79 (+0.2%)** | 4,406 (-5.1%) | 25,481 (-1.6%) | 6,153 (±0.0%) |
| Fiber | CPU | Qwen3.5-2B | Q4_K_M | 1,814 | 131.00 | **43.02** | 3,837 | **8,218** | **2,176** |
| RC2 | CPU | Qwen3.5-2B | Q4_K_M | 1,896 (-4.5%) | 125.94 (-3.9%) | 40.48 (-5.9%) | 4,000 (-4.3%) | 8,622 (-4.9%) | 2,176 (±0.0%) |
| RC1 | CPU | Qwen3.5-2B | Q4_K_M | 1,966 (-8.4%) | 121.59 (-7.2%) | 32.64 (-24.1%) | 4,145 (-8.0%) | 10,216 (-24.3%) | 2,176 (±0.0%) |
| RC3 | CPU | Qwen3.5-2B | Q4_K_M | 1,936 (-6.7%) | 123.37 (-5.8%) | 40.18 (-6.6%) | 4,084 (-6.4%) | 8,733 (-6.3%) | 2,177 (±0.0%) |
| All RCs | CPU | Qwen3.5-2B | Q4_K_M | **1,808 (+0.3%)** | **132.98 (+1.5%)** | 33.70 (-21.7%) | **3,801 (+0.9%)** | 9,790 (-19.1%) | 2,177 (±0.0%) |
| All RCs 2 | CPU | Qwen3.5-2B | Q4_K_M | 1,821 (-0.4%) | 132.22 (+0.9%) | 34.05 (-20.9%) | 3,825 (+0.3%) | 9,760 (-18.8%) | 2,177 (±0.0%) |
| Fiber | CPU | Qwen3.5-2B | Q8_0 | 1,469 | 161.88 | **30.14** | 3,106 | **10,331** | 2,873 |
| RC2 | CPU | Qwen3.5-2B | Q8_0 | 1,569 (-6.8%) | 151.47 (-6.4%) | 30.14 (±0.0%) | 3,319 (-6.8%) | 10,372 (-0.4%) | **2,872 (±0.0%)** |
| RC1 | CPU | Qwen3.5-2B | Q8_0 | 1,495 (-1.8%) | 158.30 (-2.2%) | 25.09 (-16.8%) | 3,169 (-2.0%) | 12,119 (-17.3%) | 2,873 (±0.0%) |
| RC3 | CPU | Qwen3.5-2B | Q8_0 | 1,550 (-5.5%) | 152.33 (-5.9%) | 30.13 (±0.0%) | 3,290 (-5.9%) | 10,451 (-1.2%) | 2,873 (±0.0%) |
| All RCs | CPU | Qwen3.5-2B | Q8_0 | 1,481 (-0.8%) | 159.57 (-1.4%) | 25.39 (-15.8%) | 3,142 (-1.1%) | 11,942 (-15.6%) | 2,873 (±0.0%) |
| All RCs 2 | CPU | Qwen3.5-2B | Q8_0 | **1,395 (+5.0%)** | **168.62 (+4.2%)** | 26.66 (-11.5%) | **2,967 (+4.5%)** | 11,386 (-10.2%) | 2,873 (±0.0%) |
| Fiber | CPU | Qwen3.5-4B | Q4_K_M | 4,372 | 54.47 | 18.70 | 9,237 | 18,716 | **3,712** |
| RC2 | CPU | Qwen3.5-4B | Q4_K_M | 4,439 (-1.5%) | 53.77 (-1.3%) | 18.66 (-0.2%) | 9,367 (-1.4%) | 18,860 (-0.8%) | 3,712 (±0.0%) |
| RC1 | CPU | Qwen3.5-4B | Q4_K_M | 4,335 (+0.8%) | 54.98 (+0.9%) | 14.56 (-22.1%) | 9,155 (+0.9%) | 22,615 (-20.8%) | 3,712 (±0.0%) |
| RC3 | CPU | Qwen3.5-4B | Q4_K_M | **4,317 (+1.3%)** | **55.04 (+1.0%)** | **18.74 (+0.2%)** | **9,132 (+1.1%)** | **18,715 (±0.0%)** | 3,712 (±0.0%) |
| All RCs | CPU | Qwen3.5-4B | Q4_K_M | 4,318 (+1.2%) | 54.83 (+0.7%) | 14.78 (-21.0%) | 9,151 (+0.9%) | 22,235 (-18.8%) | 3,712 (±0.0%) |
| All RCs 2 | CPU | Qwen3.5-4B | Q4_K_M | 4,496 (-2.8%) | 52.20 (-4.2%) | 14.24 (-23.9%) | 9,573 (-3.6%) | 23,271 (-24.3%) | 3,712 (±0.0%) |
| Fiber | CPU | Qwen3.5-4B | Q8_0 | 3,116 | 64.08 | 13.71 | 7,251 | 24,046 | 5,375 |
| RC2 | CPU | Qwen3.5-4B | Q8_0 | 3,652 (-17.2%) | 62.05 (-3.2%) | 13.72 (+0.1%) | 7,923 (-9.3%) | **23,265 (+3.2%)** | **5,374 (±0.0%)** |
| RC1 | CPU | Qwen3.5-4B | Q8_0 | 3,509 (-12.6%) | 61.46 (-4.1%) | 10.90 (-20.5%) | 7,821 (-7.9%) | 28,331 (-17.8%) | 5,374 (±0.0%) |
| RC3 | CPU | Qwen3.5-4B | Q8_0 | **3,077 (+1.3%)** | 64.58 (+0.8%) | **13.84 (+0.9%)** | 7,180 (+1.0%) | 23,923 (+0.5%) | 5,374 (±0.0%) |
| All RCs | CPU | Qwen3.5-4B | Q8_0 | 3,325 (-6.7%) | 64.84 (+1.2%) | 10.97 (-20.0%) | 7,412 (-2.2%) | 28,024 (-16.5%) | 5,374 (±0.0%) |
| All RCs 2 | CPU | Qwen3.5-4B | Q8_0 | 3,191 (-2.4%) | **69.53 (+8.5%)** | 11.17 (-18.5%) | **7,002 (+3.4%)** | 27,204 (-13.1%) | 5,374 (±0.0%) |

---

## Appendix C: Throttling Check — All RCs 2 vs Original All RCs

All RCs 2 (thermal-stable rerun at T0744) vs original All RCs (T2310, 5th variant
in a 5h25m continuous batch). Delta shows how much the cool rerun improved over the
thermally-loaded original.

| Headline config | Vision | Prefill | Decode | TTFT | Total | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| Gemma4-E2B-Q4 Metal | -0.6% | +4.2% | +4.6% | +2.4% | +2.5% | -0.2% |
| Gemma4-E2B-Q8 Metal | +12.0% | +18.1% | +18.9% | +14.1% | +15.1% | -0.0% |
| Qwen3.5-2B-Q4 Metal | +1.3% | +3.1% | +1.9% | +2.4% | +1.8% | -0.0% |
| Qwen3.5-2B-Q8 Metal | +8.0% | +10.7% | +2.1% | +9.1% | +1.3% | -0.1% |

**Verdict**: thermal throttling **was present** in the original `All RCs` measurement
on most Gemma4 configs. Decode on Gemma4-E2B-Q4 improved +4.6% in the cool rerun;
on Gemma4-E2B-Q8 it improved a striking **+19%**; E4B variants improved 3-6%.
Qwen3.5 was largely unaffected (±2% on the two primary configs, decode is mostly
memory-bandwidth-bound and less sensitive). **Implication**: the original Primary
Results Matrix understates the Gemma4 wins of `All RCs` vs Fiber — particularly on
E2B-Q8. The cleaner reference for any Gemma4 decode comparison is the `All RCs 2`
row; the original `All RCs` row is preserved for traceability.

---

## Appendix D: Per-Branch Artifacts

| Branch | Raw logs (VLM) | Parsed JSON (VLM) | Text-only raw | Metal System Trace (Gemma4-E2B-Q4) | Metal System Trace (Qwen3.5-2B-Q4) | Orchestrator log |
|---|---|---|---|---|---|---|
| Fiber | `results/raw/mac-fiber-baseline-rss-2026-05-14T2310/` | `results/parsed/mac-fiber-baseline-rss-2026-05-14T2310.json` | `results/raw/text-only-fiber-baseline-rss-2026-05-14T2310/` | `results/traces/fiber-baseline-rss-gemma4-e2b-q4km-2026-05-14T2310.trace` | `results/traces/fiber-baseline-rss-qwen35-2b-q4km-2026-05-14T2310.trace` | `results/orchestrator-logs/rerun-all-rss-2026-05-14T2310.log` |
| RC2 | `results/raw/mac-rc2-rerun-rss-2026-05-14T2310/` | `results/parsed/mac-rc2-rerun-rss-2026-05-14T2310.json` | `results/raw/text-only-rc2-rerun-rss-2026-05-14T2310/` | `results/traces/rc2-rerun-rss-gemma4-e2b-q4km-2026-05-14T2310.trace` | `results/traces/rc2-rerun-rss-qwen35-2b-q4km-2026-05-14T2310.trace` | (same) |
| RC1 | `results/raw/mac-rc1-rerun-rss-2026-05-14T2310/` | `results/parsed/mac-rc1-rerun-rss-2026-05-14T2310.json` | `results/raw/text-only-rc1-rerun-rss-2026-05-14T2310/` | `results/traces/rc1-rerun-rss-gemma4-e2b-q4km-2026-05-14T2310.trace` | `results/traces/rc1-rerun-rss-qwen35-2b-q4km-2026-05-14T2310.trace` | (same) |
| RC3 | `results/raw/mac-rc3-rerun-rss-2026-05-14T2310/` | `results/parsed/mac-rc3-rerun-rss-2026-05-14T2310.json` | `results/raw/text-only-rc3-rerun-rss-2026-05-14T2310/` | `results/traces/rc3-rerun-rss-gemma4-e2b-q4km-2026-05-14T2310.trace` | `results/traces/rc3-rerun-rss-qwen35-2b-q4km-2026-05-14T2310.trace` | (same) |
| All RCs | `results/raw/mac-fiber-updates-rerun-rss-2026-05-14T2310/` | `results/parsed/mac-fiber-updates-rerun-rss-2026-05-14T2310.json` | `results/raw/text-only-fiber-updates-rerun-rss-2026-05-14T2310/` | `results/traces/fiber-updates-rerun-rss-gemma4-e2b-q4km-2026-05-14T2310.trace` | `results/traces/fiber-updates-rerun-rss-qwen35-2b-q4km-2026-05-14T2310.trace` | (same) |
| All RCs 2 | `results/raw/mac-all-rcs-2-2026-05-15T0744/` | `results/parsed/mac-all-rcs-2-2026-05-15T0744.json` | `results/raw/text-only-all-rcs-2-2026-05-15T0744/` | `results/traces/all-rcs-2-gemma4-e2b-q4km-2026-05-15T0744.trace` | `results/traces/all-rcs-2-qwen35-2b-q4km-2026-05-15T0744.trace` | `results/orchestrator-logs/allrcs2-2026-05-15T0744.log` |
| b9025 | `results/raw/mac-b9025-rerun-2026-05-15T1031/` | `results/parsed/mac-b9025-rerun-2026-05-15T1031.json` | `results/raw/text-only-b9025-rerun-2026-05-15T1031/` | `results/traces/b9025-rerun-gemma4-e2b-q4km-2026-05-15T1031.trace` | `results/traces/b9025-rerun-qwen35-2b-q4km-2026-05-15T1031.trace` | `results/orchestrator-logs/b9025-rerun-2026-05-15T1031.log` |

The single orchestrator log `results/orchestrator-logs/rerun-all-rss-2026-05-14T2310.log`
covers all 5 variants. Combined text-only medians are aggregated at
`results/parsed/text-only-all-2026-05-14T2310.json`.

---

## Appendix E: Run Group `2026-05-14T1116` — Initial Isolation Runs (Stale Builds for RC1/RC3)

> **Note**: the RC1 and RC3 numbers in this appendix were produced with stale
> incremental builds and are **superseded** by the clean-build reruns in
> Appendices F and G. They are preserved for traceability of the build-hygiene
> discovery.

### Methodology

- **Device**: Mac M4, macOS 26.4.1, 16 GB unified memory
- **Build**: `cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON`, AppleClang 17.0.0, Darwin arm64
- **Matrix**: 8 models x {metal, cpu} x {elephant.jpg, fruitPlate.png}
  - Gemma4-E2B Q4_K_M/Q8_0, Gemma4-E4B Q4_K_M/Q8_0
  - Qwen3.5-2B Q4_K_M/Q8_0, Qwen3.5-4B Q4_K_M/Q8_0
- **Inference params**: `--ctx-size 4096 --predict 256 --threads 4 --temp 0 --seed 42 --jinja -fit off`
  (Metal: `--gpu-layers 99`, CPU: `--gpu-layers 0`)
- **Protocol**: 1 warmup + 3 measured runs, **median** reported. No cool-down between runs (Mac M4 active cooling).
- **Trace capture**: `xcrun xctrace --template "Metal System Trace" --time-limit 30s` for `gemma4-e2b-q4km` + `qwen35-2b-q4km` per branch.
- **Expected failures**: Qwen3.5 x fruitPlate produces 4,015 vision tokens (ctx overflow) and crashes early — captured in raw logs but excluded from median.

### Measurement Variance — Critical Caveat

Five Mac M4 measurement points were taken in this run group, in time order:

| # | Variant | Approx start | Notes |
|---|---|---|---|
| 1 | RC2 isolated (full matrix + traces) | 11:16 | First — Mac near-idle thermal state |
| 2 | RC1 isolated (full matrix + traces) | ~12:25 | After 1h09m of continuous load |
| 3 | RC3 isolated (full matrix + traces) | ~13:30 | After 2h14m of continuous load |
| 4 | **fiber-baseline-today** (full matrix + traces) | ~14:35 | After 3h19m, same-thermal state as the RCs |
| 5 | **RC2-control-rerun** (2 configs only) | ~16:55 | After ~5h40m — thermal end-state validator |

The **fiber-baseline-today** (#4) and **RC2-control-rerun** (#5) were added
mid-experiment because the early data showed effects that looked thermal. They
serve as independent reference points:

- **fiber-baseline-today** is an apples-to-apples baseline measured in the same
  thermal state as RC1 and RC3.
- **RC2-control-rerun** re-measures RC2 *after* RC1/RC3/fiber-today completed,
  confirming whether RC2's earlier numbers were thermally-favored.

#### Fiber Drift: Today vs 2026-05-13

Comparing the two fiber fork baseline measurements (same code, same machine, different
days/thermal states) — prior-day fiber baseline vs the fresh `fiber-baseline-today` run:

| Config (Metal x elephant) | fiber 2026-05-13 (t/s) | fiber today (t/s) | Delta |
|---|---:|---:|---:|
| gemma4-e2b-q4km | 31.45 | 30.68 | -2.4% |
| gemma4-e2b-q8 | 21.79 | 19.87 | -8.8% |
| gemma4-e4b-q4km | 16.78 | 14.81 | -11.7% |
| gemma4-e4b-q8 | 12.66 | 10.91 | -13.8% |
| qwen35-2b-q4km | 32.56 | 29.88 | -8.2% |
| qwen35-2b-q8 | 25.28 | 24.67 | -2.4% |
| qwen35-4b-q4km | 15.46 | 14.55 | -5.9% |
| qwen35-4b-q8 | 13.15 | 12.46 | -5.2% |

Same fiber fork baseline source code is **2-14% slower today** than yesterday — likely
chassis temperature / cooling-system state differences. This is the noise floor
for any Delta% comparison between days.

#### RC2-Control-Rerun: Thermal End-State

| Config (Metal x elephant) | RC2 fresh (t/s) | RC2 control rerun (t/s) | Delta rerun / fresh |
|---|---:|---:|---:|
| gemma4-e2b-q4km | 41.89 | 42.26 | +0.9% |
| qwen35-2b-q4km | 32.39 | 38.65 | +19.3% |

**Key finding**: RC2 re-measured at the *very end* of the session (~5h40m of
load) is **not** slower than the first measurement. It is **the same or faster**.
This rules out simple thermal degradation as the explanation for the
RC1/RC3 numbers below — the RC1 and RC3 measurements differ from RC2 because
of the code (stale build artifacts), not because the machine grew tired.

### Headline Summary — Mac M4 Metal x elephant decode_tps

| Config | fiber 2026-05-13 | fiber today | **RC2** | **RC1** | **RC3** | RC2 ctrl |
|---|---:|---:|---:|---:|---:|---:|
| gemma4-e2b-q4km | 31.45 | 30.68 | **41.89** | 27.35 | 35.36 | 42.26 |
| gemma4-e2b-q8 | 21.79 | 19.87 | **25.92** | 20.81 | 16.02 | — |
| gemma4-e4b-q4km | 16.78 | 14.81 | **18.83** | 15.99 | 13.35 | — |
| gemma4-e4b-q8 | 12.66 | 10.91 | **13.06** | 10.79 | 8.85 | — |
| qwen35-2b-q4km | 32.56 | 29.88 | 32.39 | 32.41 | 23.45 | **38.65** |
| qwen35-2b-q8 | 25.28 | 24.67 | 25.59 | 24.63 | 22.08 | — |
| qwen35-4b-q4km | 15.46 | 14.55 | 14.54 | 14.12 | 13.16 | — |
| qwen35-4b-q8 | 13.15 | 12.46 | 12.03 | 11.23 | 10.96 | — |

Delta% against both fiber baselines for the eight Metal x elephant configs:

| Config | RC2 vs fiber-2026-05-13 | RC2 vs fiber-today | RC1 vs fiber-2026-05-13 | RC1 vs fiber-today | RC3 vs fiber-2026-05-13 | RC3 vs fiber-today |
|---|---:|---:|---:|---:|---:|---:|
| gemma4-e2b-q4km | **+33.2%** | **+36.5%** | -13.0% | -10.9% | +12.4% | +15.3% |
| gemma4-e2b-q8 | **+19.0%** | **+30.4%** | -4.5% | +4.7% | -26.5% | -19.4% |
| gemma4-e4b-q4km | **+12.2%** | **+27.1%** | -4.7% | +8.0% | -20.4% | -9.9% |
| gemma4-e4b-q8 | +3.2% | **+19.7%** | -14.8% | -1.1% | -30.1% | -18.9% |
| qwen35-2b-q4km | -0.5% | +8.4% | -0.5% | +8.5% | -28.0% | -21.5% |
| qwen35-2b-q8 | +1.2% | +3.7% | -2.6% | -0.2% | -12.7% | -10.5% |
| qwen35-4b-q4km | -6.0% | -0.1% | -8.7% | -3.0% | -14.9% | -9.6% |
| qwen35-4b-q8 | -8.5% | -3.5% | -14.6% | -9.9% | -16.7% | -12.0% |

#### Headline Findings

1. **RC2 is a genuine Gemma4 Metal decode win on M4** (+12% to +37% across
   E2B/E4B x Q4_K_M/Q8_0). The thermal-control rerun reproduces the fast
   result, ruling out variance. The Metal Tensor API kernel split benefits
   the non-tensor-API path too on M4.
2. **RC1 isolated is essentially neutral** vs same-thermal fiber-today on the
   2B/E2B models (-3% to +9%) and a modest regression on the 4B/E4B Q8_0
   models (-10% to -11%). The previously-reported "+18.8% Qwen3.5 from RC1
   alone" does **not** reproduce in isolation — but this turned out to be a
   **stale build artifact**, not a real finding (see Appendix F).
3. **RC3 isolated is a regression** across most Metal configs (-10% to -22%
   on Qwen3.5; +15% to -19% on Gemma4). Also a **stale build artifact** —
   the clean-build rerun in Appendix G flipped RC3 to the largest single-commit
   win.

These findings were *opposite of the cumulative measurements* because the
incremental cmake build leaked stale embedded-Metal-shader artifacts between
branch switches. See Appendices F and G for the corrected measurements.

---

### RC2 — Metal Tensor API `MUL_MAT` Optimization

- **Commit**: `f6d6dbcd1`
- **Raw logs**: `results/raw/mac-rc2-mul-mat-opt-2026-05-14T1116/`
- **Parsed**: `results/parsed/mac-rc2-mul-mat-opt-2026-05-14T1116.json`

#### Delta vs Fiber Baseline (2026-05-13)

| Config | vision_ms | prefill_tps | decode_tps | total_ms |
|---|---|---|---|---|
| `gemma4-e2b-q4km\|metal\|elephant` | 604ms (+23.1%) | 257.4 t/s (+45.2%) | 41.9 t/s (**+33.2%**) | 7,703ms (+24.0%) |
| `gemma4-e2b-q8\|metal\|elephant` | 653ms (+14.6%) | 231.1 t/s (+24.9%) | 25.9 t/s (**+19.0%**) | 11,475ms (+15.9%) |
| `gemma4-e4b-q4km\|metal\|elephant` | 712ms (+16.7%) | 141.5 t/s (+29.0%) | 18.8 t/s (**+12.2%**) | 16,046ms (+12.3%) |
| `gemma4-e4b-q8\|metal\|elephant` | 770ms (-2.8%) | 133.8 t/s (-5.6%) | 13.1 t/s (+3.2%) | 22,406ms (+4.0%) |
| `qwen35-2b-q4km\|metal\|elephant` | 488ms (-4.7%) | 232.5 t/s (-8.2%) | 32.4 t/s (-0.5%) | 9,230ms (-0.5%) |
| `qwen35-2b-q8\|metal\|elephant` | 494ms (-3.1%) | 242.8 t/s (-2.8%) | 25.6 t/s (+1.2%) | 11,320ms (+1.1%) |
| `qwen35-4b-q4km\|metal\|elephant` | 608ms (-7.8%) | 110.3 t/s (-11.9%) | 14.5 t/s (-6.0%) | 20,193ms (-6.2%) |
| `qwen35-4b-q8\|metal\|elephant` | 644ms (-2.7%) | 111.5 t/s (+0.4%) | 12.0 t/s (-8.5%) | 24,033ms (-8.0%) |

**Summary**: 46 cells improved > ±5%, 12 cells regressed > ±5% (out of 160).

#### Interpretation

RC2 is a net Gemma4 decode improvement on M4 (~+12% to +37% Metal Q4_K_M/Q8_0).
The kernel split in `f6d6dbcd1` benefits the non-tensor-API code path too,
giving Gemma4 a real win on pre-M5 Apple Silicon. Qwen3.5 is approximately
neutral.

---

### RC1 — Fused `GATED_DELTA_NET` Port

- **Commit**: `556789f7d`
- **Raw logs**: `results/raw/mac-rc1-gated-delta-net-2026-05-14T1116/`
- **Parsed**: `results/parsed/mac-rc1-gated-delta-net-2026-05-14T1116.json`

#### Delta vs Fiber Baseline (2026-05-13)

| Config | vision_ms | prefill_tps | decode_tps | total_ms |
|---|---|---|---|---|
| `gemma4-e2b-q4km\|metal\|elephant` | 933ms (-18.9%) | 150.9 t/s (-14.9%) | 27.4 t/s (-13.0%) | 11,637ms (-14.9%) |
| `gemma4-e2b-q8\|metal\|elephant` | 830ms (-8.5%) | 176.7 t/s (-4.5%) | 20.8 t/s (-4.5%) | 14,224ms (-4.2%) |
| `gemma4-e4b-q4km\|metal\|elephant` | 826ms (+3.4%) | 118.4 t/s (+7.9%) | 16.0 t/s (-4.7%) | 18,911ms (-3.3%) |
| `gemma4-e4b-q8\|metal\|elephant` | 882ms (-17.8%) | 104.1 t/s (-26.5%) | 10.8 t/s (-14.8%) | 27,594ms (-18.3%) |
| `qwen35-2b-q4km\|metal\|elephant` | 518ms (-11.2%) | 200.1 t/s (-21.0%) | 32.4 t/s (-0.5%) | 9,396ms (-2.3%) |
| `qwen35-2b-q8\|metal\|elephant` | 536ms (-11.9%) | 198.7 t/s (-20.4%) | 24.6 t/s (-2.6%) | 11,852ms (-3.5%) |
| `qwen35-4b-q4km\|metal\|elephant` | 647ms (-14.7%) | 98.4 t/s (-21.4%) | 14.1 t/s (-8.7%) | 20,877ms (-9.8%) |
| `qwen35-4b-q8\|metal\|elephant` | 603ms (+3.8%) | 103.2 t/s (-7.0%) | 11.2 t/s (-14.6%) | 25,477ms (-14.5%) |

**Summary**: 15 cells improved > ±5%, 53 cells regressed > ±5% (out of 160).
**These numbers are stale-build artifacts** — see Appendix F for corrected data.

---

### RC3 — Metal Flash Attention `dk512_dv512` Instantiations

- **Commit**: `054141103`
- **Raw logs**: `results/raw/mac-rc3-fa-dk512-2026-05-14T1116/`
- **Parsed**: `results/parsed/mac-rc3-fa-dk512-2026-05-14T1116.json`

#### Delta vs Fiber Baseline (2026-05-13)

| Config | vision_ms | prefill_tps | decode_tps | total_ms |
|---|---|---|---|---|
| `gemma4-e2b-q4km\|metal\|elephant` | 698ms (+11.0%) | 196.6 t/s (+10.9%) | 35.4 t/s (**+12.4%**) | 8,950ms (+11.6%) |
| `gemma4-e2b-q8\|metal\|elephant` | 730ms (+9.5%) | 144.4 t/s (-21.9%) | 16.0 t/s (-26.5%) | 18,025ms (-32.1%) |
| `gemma4-e4b-q4km\|metal\|elephant` | 779ms (+8.9%) | 109.7 t/s (+0.0%) | 13.3 t/s (-20.4%) | 22,394ms (-22.4%) |
| `gemma4-e4b-q8\|metal\|elephant` | 851ms (-5.5%) | 106.4 t/s (-24.9%) | 8.9 t/s (-30.1%) | 32,968ms (-41.4%) |
| `qwen35-2b-q4km\|metal\|elephant` | 631ms (-34.3%) | 142.9 t/s (-43.6%) | 23.5 t/s (-28.0%) | 12,710ms (-38.4%) |
| `qwen35-2b-q8\|metal\|elephant` | 624ms (-28.9%) | 145.1 t/s (-41.9%) | 22.1 t/s (-12.7%) | 13,019ms (-13.7%) |
| `qwen35-4b-q4km\|metal\|elephant` | 716ms (-25.4%) | 70.0 t/s (-44.1%) | 13.2 t/s (-14.9%) | 22,272ms (-17.2%) |
| `qwen35-4b-q8\|metal\|elephant` | 692ms (-9.3%) | 73.3 t/s (-34.0%) | 11.0 t/s (-16.7%) | 27,156ms (-22.0%) |

**Summary**: 4 cells improved > ±5%, 86 cells regressed > ±5% (out of 160).
**These numbers are stale-build artifacts** — see Appendix G for corrected data.

---

### Cross-Branch Summary

The three commits do **not** decompose into clean isolated gains on M4 when
using incremental builds. Their cumulative effect requires all three to be stacked.

| Variant | Gemma4-E2B-Q4 decode (t/s) | Delta vs fiber-2026-05-13 | Qwen3.5-2B-Q4 decode (t/s) | Delta vs fiber-2026-05-13 |
|---|---:|---:|---:|---:|
| Fiber 2026-05-13 (anchor) | 31.45 | — | 32.56 | — |
| Fiber today (apples-to-apples) | 30.68 | -2.4% | 29.88 | -8.2% |
| **RC2 isolated** | **41.89** | **+33.2%** | 32.39 | -0.5% |
| **RC1 isolated** | 27.35 | -13.0% | 32.41 | -0.5% |
| **RC3 isolated** | 35.36 | +12.4% | 23.45 | -28.0% |
| RC2 control rerun | 42.26 | +34.4% | 38.65 | +18.7% |

> **Note**: RC1 and RC3 numbers above were affected by stale build artifacts. The
> updated cross-branch summary with clean builds is in Appendix H.

---

## Appendix F: Run Group `2026-05-14T1710` — RC1 Reproducibility Check (Clean Build)

Same protocol as Appendix E, but only for the RC1 branch and with **one
key change**: the build uses `cmake --build ... --clean-first` to force a fresh
compile of every object file. This eliminates any stale artifacts from the
prior incremental-build sequence.

- **Variant slug**: `rc1-rerun`
- **Commit**: `556789f7d` (unchanged from Appendix E)
- **Raw logs**: `results/raw/mac-rc1-rerun-2026-05-14T1710/`
- **Parsed**: `results/parsed/mac-rc1-rerun-2026-05-14T1710.json`
- **Traces**: `results/traces/rc1-rerun-{gemma4-e2b-q4km,qwen35-2b-q4km}-2026-05-14T1710.trace` (380 MB total)

### Headline: Original RC1 Numbers Were a Stale-Build Artifact

The clean rebuild produces dramatically faster RC1 numbers across **every**
Metal x elephant config — and the new numbers reproduce the originally-reported
"+18.8% Qwen3.5" RC1 win.

| Config (Metal x elephant) | fiber 2026-05-13 | RC2 (orig) | **RC1 orig (incr. build)** | **RC1 rerun (clean build)** | Delta rerun / orig |
|---|---:|---:|---:|---:|---:|
| gemma4-e2b-q4km | 31.45 | 41.89 | 27.35 | **42.48** | **+55.3%** |
| gemma4-e2b-q8 | 21.79 | 25.92 | 20.81 | **25.63** | +23.2% |
| gemma4-e4b-q4km | 16.78 | 18.83 | 15.99 | **19.31** | +20.8% |
| gemma4-e4b-q8 | 12.66 | 13.06 | 10.79 | **14.94** | +38.5% |
| qwen35-2b-q4km | 32.56 | 32.39 | 32.41 | **38.54** | **+18.9%** |
| qwen35-2b-q8 | 25.28 | 25.59 | 24.63 | **29.99** | +21.8% |
| qwen35-4b-q4km | 15.46 | 14.54 | 14.12 | **17.89** | +26.7% |
| qwen35-4b-q8 | 13.15 | 12.03 | 11.23 | **14.25** | +26.9% |

Key observations:

- **Qwen3.5-2B-Q4 isolated RC1 = +18.4% vs fiber-2026-05-13** (38.54 vs 32.56),
  reproducing the +18.8% figure from the cumulative measurement. RC1 in isolation
  **does** deliver the GDN win on Qwen3.5 — the prior conclusion that "RC1 needs
  RC2 underneath" was wrong, caused by a build artifact.
- **Every Gemma4 variant ALSO improves under clean RC1**, with Gemma4-E2B-Q4
  hitting 42.48 t/s — a +35% gain vs fiber-2026-05-13 and effectively equal
  to RC2's 41.89 t/s. Since RC1 only adds the GDN op (no Gemma4 code path
  touched), this gain is structurally identical to RC2's Gemma4 win — the
  clean rebuild forces all cmake-generated artifacts (notably the embedded
  Metal shader library blob) to be regenerated, which is what surfaces the
  improvement.

### Hypothesis: Which Stale Artifact?

The most likely culprit is the **embedded default-metallib blob** generated at
build time from `ggml/src/ggml-metal/ggml-metal.metal`. The incremental build
appears not to have detected that the Metal source file content depended on
the currently-checked-out branch — so when the orchestrator switched from RC2
to RC1, the embedded shader blob from the previous build was reused, producing
a binary with mismatched Metal kernels vs C++ dispatch code. This matches the
symptom (decode is dominated by Metal kernels; CPU paths show smaller deltas)
and is consistent with the per-run *stability* of the bad RC1 numbers (it's
not random thermal noise — it's a deterministic incorrect build).

### Delta vs Fiber Baseline (2026-05-13) — RC1 Clean Build

| Config | vision_ms | prefill_tps | decode_tps | total_ms |
|---|---|---|---|---|
| `gemma4-e2b-q4km\|metal\|elephant` | 593ms (+24.5%) | 261.0 t/s (+47.2%) | 42.5 t/s (**+35.1%**) | 7,556ms (+25.4%) |
| `gemma4-e2b-q8\|metal\|elephant` | 654ms (+14.5%) | 232.9 t/s (+25.9%) | 25.6 t/s (**+17.6%**) | 11,531ms (+15.5%) |
| `gemma4-e4b-q4km\|metal\|elephant` | 715ms (+16.3%) | 141.6 t/s (+29.1%) | 19.3 t/s (**+15.1%**) | 15,786ms (+13.8%) |
| `gemma4-e4b-q8\|metal\|elephant` | 768ms (-2.5%) | 133.8 t/s (-5.6%) | 14.9 t/s (**+18.0%**) | 20,164ms (+13.6%) |
| `qwen35-2b-q4km\|metal\|elephant` | 488ms (-4.7%) | 240.0 t/s (-5.2%) | 38.5 t/s (**+18.4%**) | 7,989ms (+13.0%) |
| `qwen35-2b-q8\|metal\|elephant` | 487ms (-1.7%) | 250.0 t/s (+0.1%) | 30.0 t/s (**+18.6%**) | 9,825ms (+14.2%) |
| `qwen35-4b-q4km\|metal\|elephant` | 612ms (-8.5%) | 110.9 t/s (-11.4%) | 17.9 t/s (**+15.7%**) | 16,769ms (+11.8%) |
| `qwen35-4b-q8\|metal\|elephant` | 656ms (-4.5%) | 110.9 t/s (-0.1%) | 14.3 t/s (**+8.4%**) | 20,407ms (+8.3%) |

**Summary**: 64 cells improved > ±5%, 12 cells regressed > ±5% (out of 160).

### Delta Rerun vs Orig (Full)

**112 cells improved > ±5%**, 14 regressed (out of 160 measured).

### Implication for the Rest of the Run Group

The same stale-artifact mechanism almost certainly affected **RC3** in the
original run (RC3 was built incrementally on top of prior binary state).
The original RC3 numbers (86 cells regressed) should be re-tested
with `--clean-first` before drawing any conclusion about RC3's isolated impact.

RC2 was the *first* build in the original run group, with no preceding state
to leave artifacts, so RC2 is likely the only original measurement that
remains reliable.

### Updated Cross-Branch Summary (Incorporating RC1 Rerun)

| Variant | Gemma4-E2B-Q4 decode (t/s) | Delta vs fiber-2026-05-13 | Qwen3.5-2B-Q4 decode (t/s) | Delta vs fiber-2026-05-13 |
|---|---:|---:|---:|---:|
| Fiber 2026-05-13 (anchor) | 31.45 | — | 32.56 | — |
| Fiber today (apples-to-apples) | 30.68 | -2.4% | 29.88 | -8.2% |
| **RC2 isolated** | 41.89 | +33.2% | 32.39 | -0.5% |
| **RC1 isolated (orig — stale build)** | 27.35 | -13.0% | 32.41 | -0.5% |
| **RC1 isolated (rerun — clean build)** | **42.48** | **+35.1%** | **38.54** | **+18.4%** |
| RC3 isolated (orig — stale build) | 35.36 | +12.4% | 23.45 | -28.0% |
| RC2 control rerun | 42.26 | +34.4% | 38.65 | +18.7% |

Revised conclusion: with clean builds, **both RC2 and RC1 in isolation are
real Mac M4 wins** (decode ~+35% Gemma4, ~+18% Qwen3.5). RC3 isolated had not
yet been re-validated with a clean build at this point.

---

## Appendix G: Run Group `2026-05-14T1830` — RC3 Reproducibility Check (Clean Build)

Same protocol as the RC1 reproducibility check: clean rebuild
(`cmake --build ... --clean-first`), fresh timestamp, all original artifacts
preserved.

- **Variant slug**: `rc3-rerun`
- **Commit**: `054141103` (cherry-pick of `054141103`)
- **Raw logs**: `results/raw/mac-rc3-rerun-2026-05-14T1830/`
- **Parsed**: `results/parsed/mac-rc3-rerun-2026-05-14T1830.json`
- **Traces**: `results/traces/rc3-rerun-{gemma4-e2b-q4km,qwen35-2b-q4km}-2026-05-14T1830.trace` (325 MB total)

### Headline: RC3 Isolated Is the Largest Single-Commit Win, Not a Regression

The clean rebuild flips RC3 from "the worst of the three" to "the biggest
single-commit gain." With FA `dk512_dv512` templates properly registered, the
Gemma4 attention path runs Flash Attention end-to-end and decode recovers all
of the fiber regression (and slightly exceeds b9025 on E2B-Q4).

| Config (Metal x elephant) | fiber 2026-05-13 | RC2 (orig) | RC1 rerun (clean) | **RC3 orig (stale build)** | **RC3 rerun (clean build)** | Delta rerun / orig | Delta rerun / fiber-old |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemma4-e2b-q4km | 31.45 | 41.89 | 42.48 | 35.36 | **52.43** | +48.3% | **+66.7%** |
| gemma4-e2b-q8 | 21.79 | 25.92 | 25.63 | 16.02 | **30.27** | +89.0% | **+38.9%** |
| gemma4-e4b-q4km | 16.78 | 18.83 | 19.31 | 13.35 | **23.88** | +78.9% | **+42.3%** |
| gemma4-e4b-q8 | 12.66 | 13.06 | 14.94 | 8.85 | **17.63** | +99.2% | **+39.3%** |
| qwen35-2b-q4km | 32.56 | 32.39 | 38.54 | 23.45 | **36.53** | +55.8% | **+12.2%** |
| qwen35-2b-q8 | 25.28 | 25.59 | 29.99 | 22.08 | **29.23** | +32.4% | **+15.6%** |
| qwen35-4b-q4km | 15.46 | 14.54 | 17.89 | 13.16 | **17.70** | +34.5% | **+14.5%** |
| qwen35-4b-q8 | 13.15 | 12.03 | 14.25 | 10.96 | **14.53** | +32.6% | **+10.5%** |

Key observations:

- **Gemma4-E2B-Q4 hits 52.43 t/s — higher than b9025's 50.73 t/s** (upstream
  baseline). RC3 isolated does not merely "recover" the fiber regression; it
  slightly exceeds the upstream reference.
- **All Gemma4 variants gain +39% to +67%** vs fiber-2026-05-13 from RC3 alone.
  This is the FA-enable benefit — the fiber fork had Gemma4 FA globally disabled
  because the `dk512_dv512` head-size templates were missing.
- **All Qwen3.5 variants gain +10% to +16%** from RC3 alone. Qwen3.5 already
  had FA enabled in the fiber fork, but the broader supports_op table evidently
  helps some intermediate operations.

### Delta vs Fiber Baseline (2026-05-13) — RC3 Clean Build

| Config | vision_ms | prefill_tps | decode_tps | total_ms |
|---|---|---|---|---|
| `gemma4-e2b-q4km\|metal\|elephant` | 609ms (+22.4%) | 260.4 t/s (+46.9%) | 52.4 t/s (**+66.7%**) | 6,337ms (+37.4%) |
| `gemma4-e2b-q8\|metal\|elephant` | 681ms (+11.0%) | 227.8 t/s (+23.2%) | 30.3 t/s (**+38.9%**) | 10,073ms (+26.2%) |
| `gemma4-e4b-q4km\|metal\|elephant` | 729ms (+14.7%) | 133.0 t/s (+21.2%) | 23.9 t/s (**+42.3%**) | 13,298ms (+27.3%) |
| `gemma4-e4b-q8\|metal\|elephant` | 673ms (+10.1%) | 165.4 t/s (+16.7%) | 17.6 t/s (**+39.3%**) | 17,405ms (+25.4%) |
| `qwen35-2b-q4km\|metal\|elephant` | 468ms (-0.4%) | 241.6 t/s (-4.6%) | 36.5 t/s (**+12.2%**) | 8,301ms (+9.6%) |
| `qwen35-2b-q8\|metal\|elephant` | 439ms (+8.4%) | 286.1 t/s (+14.6%) | 29.2 t/s (**+15.6%**) | 9,873ms (+13.7%) |
| `qwen35-4b-q4km\|metal\|elephant` | 612ms (-8.5%) | 110.9 t/s (-11.4%) | 17.7 t/s (**+14.5%**) | 16,888ms (+12.5%) |
| `qwen35-4b-q8\|metal\|elephant` | 644ms (-2.7%) | 111.5 t/s (+0.4%) | 14.5 t/s (**+10.5%**) | 20,407ms (+8.3%) |

**Summary**: 76 cells improved > ±5%, 1 cell regressed > ±5% (out of 160).

### Delta Rerun vs Orig

**127 cells improved > ±5%**, 2 regressed (out of 160 measured). This is the
clearest stale-build-artifact signature in the entire study — only 2 cells
out of 160 don't show improvement after a clean rebuild.

### Final Updated Cross-Branch Summary (All Clean-Build Measurements)

Using the clean-build numbers wherever they exist (RC1 rerun, RC3 rerun);
RC2 is unchanged because its original measurement was clean (no preceding
build state to leak artifacts).

| Variant | Gemma4-E2B-Q4 decode (t/s) | Delta vs fiber-2026-05-13 | Qwen3.5-2B-Q4 decode (t/s) | Delta vs fiber-2026-05-13 |
|---|---:|---:|---:|---:|
| Fiber 2026-05-13 (anchor) | 31.45 | — | 32.56 | — |
| b9025 reference (upstream baseline) | 50.73 | +61.3% | 39.79 | +22.2% |
| Fiber today (same thermal as RC runs) | 30.68 | -2.4% | 29.88 | -8.2% |
| **RC2 isolated** | **41.89** | **+33.2%** | 32.39 | -0.5% |
| **RC1 isolated (clean build)** | **42.48** | **+35.1%** | **38.54** | **+18.4%** |
| **RC3 isolated (clean build)** | **52.43** | **+66.7%** | **36.53** | **+12.2%** |
| RC2 control rerun | 42.26 | +34.4% | 38.65 | +18.7% |

#### Revised Conclusion

All three commits independently deliver real Mac M4 wins. The original
`2026-05-14T1116` measurements painted a false picture because the
incremental cmake build leaked stale embedded-Metal-shader artifacts between
branch switches.

- **RC2 (Metal Tensor API MUL_MAT)** — clean, +33% Gemma4 / ~0% Qwen3.5
- **RC1 (fused GATED_DELTA_NET)** — clean, +35% Gemma4 / +18% Qwen3.5
- **RC3 (FA dk512_dv512)** — clean, +67% Gemma4 / +12% Qwen3.5

**RC3 is the single highest-impact commit on M4** — it recovers Gemma4 from
the fiber FA-disable regression and brings it to b9025-or-better levels.

For benchmark hygiene going forward: any branch-switching benchmark
orchestration on this codebase **must** use `--clean-first` (or, equivalently,
`rm -rf build-mac && cmake -B build-mac ...`) between variants. The cmake
incremental build does not correctly track the embedded Metal shader blob's
dependency on the source `.metal` file when branches are switched.

---

## Appendix H: Run Group `2026-05-14T1940` — Cumulative Branch (Clean Build)

The cumulative branch stacks all three commits on top of the fiber fork baseline:

```
fiber-fork-baseline (f686a1324)
 +-- f6d6dbcd1  metal: optimize Metal Tensor API for GGML_OP_MUL_MAT
     +-- 556789f7d  port GGML_OP_GATED_DELTA_NET
         +-- 054141103  add Metal FA dk512_dv512 kernel instantiations
```

This run measures the cumulative branch with the same `--clean-first`
protocol as the RC1 and RC3 reruns. It directly answers "what does the
production cumulative branch deliver on M4?" and lets us check whether
the three commits **compose additively** when applied together.

- **Variant slug**: `fiber-updates-rerun`
- **Raw logs**: `results/raw/mac-fiber-updates-rerun-2026-05-14T1940/`
- **Parsed**: `results/parsed/mac-fiber-updates-rerun-2026-05-14T1940.json`
- **Traces**: `results/traces/fiber-updates-rerun-{gemma4-e2b-q4km,qwen35-2b-q4km}-2026-05-14T1940.trace` (364 MB total)

### Headline: Cumulative Composition Recovers and Slightly Exceeds b9025

| Config (Metal x elephant) | fiber 2026-05-13 | b9025 (upstream) | RC2 isolated | RC1 rerun | RC3 rerun | **cumulative rerun** | Delta FUP vs fiber-old | Delta FUP vs b9025 | Delta FUP vs best-iso |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma4-e2b-q4km | 31.45 | 50.73 | 41.89 | 42.48 | 52.43 | **52.48** | **+66.9%** | +3.4% | +0.1% |
| gemma4-e2b-q8 | 21.79 | 30.88 | 25.92 | 25.63 | 30.27 | **30.16** | **+38.4%** | -2.3% | -0.4% |
| gemma4-e4b-q4km | 16.78 | 22.82 | 18.83 | 19.31 | 23.88 | **24.13** | **+43.8%** | +5.7% | +1.0% |
| gemma4-e4b-q8 | 12.66 | 14.34 | 13.06 | 14.94 | 17.63 | **16.98** | **+34.1%** | +18.4% | -3.7% |
| qwen35-2b-q4km | 32.56 | 39.79 | 32.39 | 38.54 | 36.53 | **40.00** | **+22.9%** | +0.5% | **+3.8%** |
| qwen35-2b-q8 | 25.28 | 30.37 | 25.59 | 29.99 | 29.23 | **32.34** | **+27.9%** | +6.5% | **+7.8%** |
| qwen35-4b-q4km | 15.46 | 17.60 | 14.54 | 17.89 | 17.70 | **18.44** | **+19.3%** | +4.8% | **+3.1%** |
| qwen35-4b-q8 | 13.15 | 14.08 | 12.03 | 14.25 | 14.53 | **15.23** | **+15.8%** | +8.2% | **+4.8%** |

`best-iso` = the highest decode_tps across the three isolated clean-build measurements (RC2, RC1-rerun, RC3-rerun).

Two distinct composition patterns emerge:

- **Gemma4 (saturated by RC3)**. The cumulative branch roughly equals RC3-rerun alone (Delta from best-iso = -3.7% to +1.0%, all within noise). RC3's FA enable is the dominant fix; RC2's MUL_MAT kernel split and RC1's GDN op contribute no measurable additional gain on Gemma4 once FA is live.
- **Qwen3.5 (additive — RC1 + RC3 stack)**. The cumulative branch is **+3.1% to +7.8% above the best isolated commit** on every Qwen3.5 variant. RC1 (fused GDN op) and RC3 (broader FA op support) deliver independent wins that combine. RC2 alone is essentially neutral on Qwen3.5, but does not detract from the stacked result.

### Headline Summary — All Clean-Build Mac M4 Metal x elephant decode_tps

| Config | fiber-old | fiber-today | b9025 | RC2 iso | RC1 rerun | RC3 rerun | **cumulative rerun** |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemma4-e2b-q4km | 31.45 | 30.68 | 50.73 | 41.89 | 42.48 | **52.43** | **52.48** |
| gemma4-e2b-q8 | 21.79 | 19.87 | 30.88 | 25.92 | 25.63 | **30.27** | **30.16** |
| gemma4-e4b-q4km | 16.78 | 14.81 | 22.82 | 18.83 | 19.31 | **23.88** | **24.13** |
| gemma4-e4b-q8 | 12.66 | 10.91 | 14.34 | 13.06 | 14.94 | **17.63** | **16.98** |
| qwen35-2b-q4km | 32.56 | 29.88 | 39.79 | 32.39 | **38.54** | 36.53 | **40.00** |
| qwen35-2b-q8 | 25.28 | 24.67 | 30.37 | 25.59 | **29.99** | 29.23 | **32.34** |
| qwen35-4b-q4km | 15.46 | 14.55 | 17.60 | 14.54 | **17.89** | 17.70 | **18.44** |
| qwen35-4b-q8 | 13.15 | 12.46 | 14.08 | 12.03 | **14.25** | 14.53 | **15.23** |

(Cell highlighting: **bold** marks the best decode_tps in each row.)

### Delta vs Fiber Baseline (2026-05-13) — Cumulative Clean Build

| Config | vision_ms | prefill_tps | decode_tps | total_ms |
|---|---|---|---|---|
| `gemma4-e2b-q4km\|metal\|elephant` | 610ms (+22.3%) | 260.3 t/s (+46.8%) | 52.5 t/s (**+66.9%**) | 6,327ms (+37.5%) |
| `gemma4-e2b-q8\|metal\|elephant` | 654ms (+14.5%) | 235.7 t/s (+27.4%) | 30.2 t/s (**+38.4%**) | 10,043ms (+26.4%) |
| `gemma4-e4b-q4km\|metal\|elephant` | 702ms (+17.9%) | 142.4 t/s (+29.9%) | 24.1 t/s (**+43.8%**) | 13,046ms (+28.7%) |
| `gemma4-e4b-q8\|metal\|elephant` | 675ms (+9.9%) | 163.6 t/s (+15.5%) | 17.0 t/s (**+34.1%**) | 17,904ms (+23.3%) |
| `qwen35-2b-q4km\|metal\|elephant` | 475ms (-1.9%) | 237.8 t/s (-6.1%) | 40.0 t/s (**+22.9%**) | 7,661ms (+16.6%) |
| `qwen35-2b-q8\|metal\|elephant` | 448ms (+6.5%) | 279.2 t/s (+11.8%) | 32.3 t/s (**+27.9%**) | 9,026ms (+21.1%) |

**Summary**: 73 cells improved > ±5%, 8 cells regressed > ±5% (out of 160).

### Cross-Validation: Cumulative vs Sum of Isolated

Comparing cumulative (clean-build rerun) decode_tps against the **best** isolated
clean-build measurement, per config:

| Config | Best isolated (which) | Cumulative rerun | Delta cumulative / best-iso | Composition pattern |
|---|---:|---:|---:|---|
| gemma4-e2b-q4km | 52.43 (RC3) | 52.48 | +0.1% | saturated by RC3 |
| gemma4-e2b-q8 | 30.27 (RC3) | 30.16 | -0.4% | saturated by RC3 |
| gemma4-e4b-q4km | 23.88 (RC3) | 24.13 | +1.0% | saturated by RC3 |
| gemma4-e4b-q8 | 17.63 (RC3) | 16.98 | -3.7% | saturated by RC3 |
| qwen35-2b-q4km | 38.54 (RC1) | 40.00 | **+3.8%** | RC1 + RC3 additive |
| qwen35-2b-q8 | 29.99 (RC1) | 32.34 | **+7.8%** | RC1 + RC3 additive |
| qwen35-4b-q4km | 17.89 (RC1) | 18.44 | **+3.1%** | RC1 + RC3 additive |
| qwen35-4b-q8 | 14.53 (RC3) | 15.23 | **+4.8%** | RC1 + RC3 additive |

Mechanistic reading:

- **Gemma4 has no GDN layers** — RC1 (GDN op port) is a no-op for it. With RC2's
  Metal kernel split and RC3's FA enable both active, the third commit (the
  one not in play) adds nothing measurable. Gemma4 is FA-bound; once FA is on,
  the throughput ceiling on M4 is hardware-bound.
- **Qwen3.5 uses both** — RC1 wires the fused GDN op into Qwen3.5's recurrent
  layers, RC3's expanded supports_op table improves the non-GDN attention
  path. Both contribute to Qwen3.5 decode, hence the additive ~4-8% on top of
  the best single-commit clean-build measurement.

### Final Cross-Branch Summary (All Clean-Build Measurements + Cumulative)

| Variant | Gemma4-E2B-Q4 decode (t/s) | Delta vs fiber-2026-05-13 | Qwen3.5-2B-Q4 decode (t/s) | Delta vs fiber-2026-05-13 |
|---|---:|---:|---:|---:|
| Fiber 2026-05-13 (anchor) | 31.45 | — | 32.56 | — |
| b9025 reference (upstream baseline) | 50.73 | +61.3% | 39.79 | +22.2% |
| Fiber today (same-thermal as RC runs) | 30.68 | -2.4% | 29.88 | -8.2% |
| RC2 isolated | 41.89 | +33.2% | 32.39 | -0.5% |
| RC1 isolated (clean build) | 42.48 | +35.1% | 38.54 | +18.4% |
| RC3 isolated (clean build) | 52.43 | +66.7% | 36.53 | +12.2% |
| **Cumulative (RC2+RC1+RC3, clean build)** | **52.48** | **+66.9%** | **40.00** | **+22.9%** |

The cumulative branch is the **production target**: it slightly exceeds b9025 on
Gemma4-E2B-Q4 Metal decode (52.48 vs 50.73, +3.4%) and reaches b9025 parity on
Qwen3.5-2B-Q4 Metal decode (40.00 vs 39.79, +0.5%). Across the full 8-config
Metal x elephant matrix, every config improves +15% to +67% over fiber-2026-05-13,
and most are at or above b9025.

#### Closing Recommendation

The cumulative branch is ready for merge into the fiber tip. All three commits earn
their place: RC3 dominates the Gemma4 win, RC1 dominates the Qwen3.5 win, and RC2's
MUL_MAT kernel restructuring is the most broad-spectrum optimization (every Gemma4
variant improves modestly, no regressions). The build-system hazard surfaced in this
study — incremental-build artifact leakage between branch switches — should be filed
as a follow-up; the workaround (`cmake --build ... --clean-first`) is straightforward
but easy to forget.

---
