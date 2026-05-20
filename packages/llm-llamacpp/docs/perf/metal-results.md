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
| iPhone 16e | Qwen3.5 projection | **-94%** | 183 ms → 11 ms |
| iPhone 16e | Qwen3.5 total time | **-17%** | — |

On Mac M4, the deepstack preallocation has **no measurable effect** on any
Qwen3.5 metric. This is consistent with Section 1.1: the projection phase
completes in ~2 ms on Mac M4 (12.7 GB working set), so eliminating the
O(N^2) reallocation saves negligible time. The optimization targets iPhone
16e, where the same projection takes 183 ms under Jetsam pressure.

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

#### Prior Cross-Session Comparison (Superseded)

An earlier comparison (fiber May 13 vs U1 May 20) reported +10–21% vision_ms
and +18–47% prefill_tps improvements. These were **entirely cross-session
thermal variance**: text-only `llama-bench` showed +32.6% pp256 and +15.5%
tg256 shifts between the same sessions — magnitude inconsistent with a code
effect. The interleaved A/B protocol above eliminates this confound and is
now the source of truth for Mac M4.

#### iPhone 16e

The 183 ms iPhone 16e projection value was measured on the fiber fork. On
upstream llama.cpp, iPhone projection was already 9 ms — the anomaly was
fiber-specific (likely related to different memory allocation patterns in
the fork's modified graph). The preallocation fix still applies because the
chained-concat pattern would regress with more deepstack layers on any
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

## 2. Production Summary — Fiber vs U1

Interleaved A/B comparison of the fiber fork baseline vs the U1 deepstack
preallocation optimization. Qwen3.5 models only (U1 does not modify Gemma4).
Metal backend, elephant.jpg.

- **Fiber** = fiber fork baseline `tetherto/temp-8189` (commit `f686a1324`)
- **U1** = deepstack preallocation `feat/QVAC-18984-deepstack-prealloc` (commit `5e5be94f3`)
- **Protocol**: per-config interleaved A/B, 5 paired reps, thermal gate,
  Wilcoxon signed-rank test (2026-05-20 session)

| Model | Quant | Vision (ms) | p | Prefill (t/s) | p | Decode (t/s) | p | Peak RSS (MB) |
|-------|-------|------------:|----:|--------------:|----:|-------------:|----:|--------:|
| Qwen3.5-2B | Q4_K_M | 431 / 431 | 0.25 | 297.9 / 297.8 | 0.19 | 38.5 / 38.5 | 0.63 | 946 / 946 |
| Qwen3.5-2B | Q8_0 | 426 / 427 | 0.19 | 306.7 / 306.4 | 0.31 | 31.2 / 31.2 | 0.81 | 946 / 946 |
| Qwen3.5-4B | Q4_K_M | 524 / 528 | 0.31 | 151.8 / 150.9 | 0.13 | 19.0 / 18.6 | 0.06 | 1,072 / 1,071 |
| Qwen3.5-4B | Q8_0 | 530 / 538 | 0.81 | 143.7 / 142.0 | 1.00 | 14.7 / 14.9 | 0.81 | 1,072 / 1,072 |

Cell format: fiber / u1 (median of 5 paired reps). No metric reaches p < 0.05.

**Mac M4**: No measurable effect. The deepstack reallocation overhead is
negligible on Mac M4's 16 GB unified memory with ~120 GB/s bandwidth.

**iPhone 16e**: Projection latency reduced from 183 ms to 11 ms (−94%).
This is the target platform where the O(N^2) reallocation dominates under
Jetsam memory pressure on 8 GB devices. iPhone interleaved A/B testing will
follow.

---

## 3. Methodology

### Device and Build

- **Device**: Mac M4, macOS 26.4.1, 16 GB unified memory
- **Build**: `cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON`, AppleClang 17.0.0, Darwin arm64
- **Build hygiene**: `cmake --build ... --target llama-mtmd-cli -j --clean-first` mandatory when switching branches (prevents embedded Metal shader blob contamination from stale incremental builds)

### Inference Parameters

- `--ctx-size 4096 --predict 256 --threads 4 --temp 0 --seed 42 --jinja -fit off`
- Metal: `--gpu-layers 99`

### Protocol — Interleaved A/B

Mac M4 results in this document use per-config interleaved A/B benchmarking
(`benchmark-mac-interleaved.sh`):

- **Both binaries built clean** from their respective branches before the session
- **Per-config interleaving**: for each model, warmup-A → warmup-B → then
  5 paired reps (A-run1 → B-run1 → A-run2 → B-run2 → ...)
- **Thermal gate**: `powermetrics` polls for CPU thermal level 0 (Nominal)
  before every measured run — replaces fixed-time cooldowns
- **Anchor drift detection**: Gemma4-E2B Q4KM run at session start, midpoint,
  and end — flags session if decode_tps drift exceeds 3%
- **Statistics**: Wilcoxon signed-rank test on per-rep paired deltas + bootstrap
  95% CI (10,000 iterations). Only cite deltas where p < 0.05.
- **Images**: elephant.jpg (612 x 408, 265 vision tokens for Qwen3.5)
- **RSS measurement**: every invocation wrapped with `/usr/bin/time -l`

This protocol was adopted after an earlier cross-session comparison (fiber
May 13 vs U1 May 20) produced 15–47% deltas that were entirely attributable
to environmental variance, not code changes.

See [metal-baseline.md](metal-baseline.md) for device specs, model inventory,
and test image details.
