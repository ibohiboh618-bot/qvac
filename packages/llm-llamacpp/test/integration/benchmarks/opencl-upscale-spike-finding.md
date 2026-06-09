# Spike: UPSCALE Op on OpenCL Backend

**Status:** Finding complete — ready to unblock implementation task  
**Device tested:** Samsung Galaxy S25 Ultra (Adreno, OpenCL) via Firebase Test Lab  
**llama.cpp base:** b8828 (`tetherto/qvac-fabric-llm.cpp` v8828.0.2)  
**Date:** 2026-06-08

---

## 1. What the Graph Requests

The CLIP vision encoder for **Qwen3VL** (and Qwen2VL/Qwen2.5VL) calls `resize_position_embeddings()` which emits an `GGML_OP_UPSCALE` (via `ggml_interpolate`) with:

| Field | Value |
|-------|-------|
| Mode | `GGML_SCALE_MODE_BILINEAR \| GGML_SCALE_FLAG_ANTIALIAS` (combined value: **0x201 = 513**) |
| Type | `f32` |
| Output shape (from warmup) | `[92, 92, 768, 1]` — i.e. `[width, height, n_embd, 1]` |
| Input shape | The learned position embedding grid resized to match `img_size / patch_size` |

**Source:** `tools/mtmd/clip-graph.h` line:

```cpp
#define DEFAULT_INTERPOLATION_MODE (GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS)
```

Called from `clip_graph::resize_position_embeddings()` → `ggml_interpolate(ctx0, pos_embd, width, height, n_embd, 1, mode)`.

The Qwen3VL graph builder (`tools/mtmd/models/qwen3vl.cpp`) calls `resize_position_embeddings()` with the default mode (no explicit override), so it always requests **bilinear + antialias**.

The representative tensor dimensions (92×92) come from warmup at 1472×1472 image / 16 patch_size = 92 patches per side.

---

## 2. What OpenCL Implements Today

**File:** `ggml/src/ggml-opencl/ggml-opencl.cpp`

### supports_op check (line ~4073):

```cpp
case GGML_OP_UPSCALE: {
    ggml_scale_mode mode = (ggml_scale_mode)(ggml_get_op_params_i32(op, 0) & 0xFF);
    const bool antialias = (ggml_scale_mode)(ggml_get_op_params_i32(op, 0) & GGML_SCALE_FLAG_ANTIALIAS);
    return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
        (mode == GGML_SCALE_MODE_NEAREST || mode == GGML_SCALE_MODE_BILINEAR) && !antialias;
}
```

### Implemented kernels (in `kernels/upscale.cl`):

| Kernel | Mode | Description |
|--------|------|-------------|
| `kernel_upscale` | NEAREST | Simple index-mapped copy |
| `kernel_upscale_bilinear` | BILINEAR (no antialias) | 4-point bilinear interpolation |

### Dispatch (line ~9020):

```cpp
if (mode == GGML_SCALE_MODE_NEAREST) {
    kernel = backend_ctx->kernel_upscale;
} else if (mode == GGML_SCALE_MODE_BILINEAR) {
    kernel = backend_ctx->kernel_upscale_bilinear;
} else {
    GGML_LOG_WARN("unsupported upscale mode %d, skipping.\n", mode);
    return;
}
```

### Fallback path:

When `supports_op` returns `false` (because `antialias == true`), the graph scheduler assigns the UPSCALE node to **CPU**. This causes a graph split mid-CLIP, the data is transferred GPU→CPU→GPU, and the resulting embeddings are **corrupt** — producing hallucinated captions.

---

## 3. The Exact Gap

> **OpenCL supports `GGML_SCALE_MODE_BILINEAR` without antialias; the CLIP projector requires `GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS`.**

The `!antialias` condition in `supports_op` causes the op to fall back to CPU. The graph split (GPU→CPU→GPU) for this single op in the middle of the CLIP encoder appears to corrupt the intermediate tensor data, producing completely wrong image embeddings.

---

## 4. Reference to Port From

### Recommended: CUDA (`ggml/src/ggml-cuda/upscale.cu`)

The `upscale_f32_bilinear_antialias` kernel is **~40 lines**, self-contained, and uses the same programming model as OpenCL (flat global index, per-element computation).

**Algorithm (triangle filter with area-based weighting):**
1. Compute support radius = `max(1/scale_factor, 1.0)` per axis
2. For each output pixel, iterate over source pixels within the support window
3. Weight each source pixel by `triangle_filter(distance * invscale)`
4. Normalize by total weight

### Alternative: Vulkan (`ggml/src/ggml-vulkan/vulkan-shaders/upscale.comp`)

The `interpolate_bilinear_antialias` function is **~30 lines** and identical logic. Uses `constant_id = 0` specialization with `BILINEAR_ANTIALIAS = 513` as a mode value.

### Comparison:

| Backend | Lines | Loop structure | Notes |
|---------|-------|---------------|-------|
| CUDA | ~40 | Double for-loop over support window | Direct translation to OpenCL |
| Vulkan | ~30 | Same double for-loop | Slightly more compact (GLSL builtins) |
| Metal | ~30 | Same algorithm | Also a good reference |

**Recommendation: Use the CUDA kernel as primary reference** — its syntax is closest to OpenCL C (same `max`/`min`/`floor` builtins, same pointer arithmetic, same thread-per-element model). The Vulkan shader is a good cross-check.

---

## 5. Fix Location

**Fix in:** `tetherto/qvac-fabric-llm.cpp` (our fork, currently at v8828.0.2)

**Rationale:** Upstream llama.cpp has the same gap (issue [#21268](https://github.com/ggml-org/llama.cpp/issues/21268), [#21941](https://github.com/ggml-org/llama.cpp/issues/21941)), and the OpenCL backend in upstream is community-contributed without active maintainership for this op. Fixing in our fork unblocks Qwen3VL on Adreno immediately. The fix is small and clean enough to upstream later as a PR.

**Files to modify (3):**

| File | Change |
|------|--------|
| `ggml/src/ggml-opencl/kernels/upscale.cl` | Add `kernel_upscale_bilinear_antialias` kernel |
| `ggml/src/ggml-opencl/ggml-opencl.cpp` (supports_op) | Remove `&& !antialias` from the BILINEAR case (or add a separate check for `kernel_upscale_bilinear_antialias != nullptr`) |
| `ggml/src/ggml-opencl/ggml-opencl.cpp` (dispatch) | Add `else if (mode == GGML_SCALE_MODE_BILINEAR && antialias)` branch using the new kernel |

---

## 6. Rough Effort Estimate

### Code writing (pure implementation)

| Task | Estimate |
|------|----------|
| Write `kernel_upscale_bilinear_antialias` in `upscale.cl` | 1–2 hours |
| Wire kernel creation + dispatch in `ggml-opencl.cpp` | 1–2 hours |
| Update `supports_op` to accept antialias | 15 min |
| **Code subtotal** | **~3–4 hours** |

### Build & test iterations (Firebase / Android Studio)

Each iteration = rebuild ggml OpenCL .so → push to device via Firebase Test Lab → run llama-mtmd-cli → inspect output. This is the dominant time cost.

| Task | Estimate |
|------|----------|
| Initial Android NDK build of modified ggml-opencl | 30–60 min (first build + fix compile errors) |
| Deploy to Firebase Test Lab + verify kernel loads | 30–45 min per attempt |
| First on-device run (check no crash, warning gone) | 1 iteration |
| Debug kernel arg mismatch / CL build failures | 1–3 iterations |
| Validate correct output vs CPU baseline | 1–2 iterations |
| Test across quants (Q8_0, Q4_K_M, Q4_0) + both models (0.8B, 2B) | 2–3 iterations |
| **Iteration subtotal** (~30–45 min per round-trip) | **4–8 iterations = 3–6 hours** |

### Total realistic estimate

| | Optimistic | Pessimistic |
|--|-----------|-------------|
| Code writing | 3 hours | 4 hours |
| Build/test iterations | 3 hours (4 iterations) | 6 hours (8 iterations) |
| **Total** | **~1 day** | **~2 days** |

The algorithm is straightforward (triangle filter, same as existing bilinear but with variable-radius support window). No new data structures, no new kernel arg patterns beyond what `kernel_upscale_bilinear` already uses. The bottleneck is the Firebase deploy round-trip, not code complexity.

---

## Runtime Confirmation (diff of CPU vs OpenCL projector)

Device: Samsung S25 Ultra (Adreno 830), Qwen3.5-0.8B-Q8_0, elephant.jpg, `-n 256 --temp 0 --seed 42`

```diff
-clip_ctx: CLIP using CPU backend
+clip_ctx: CLIP using OpenCL backend

-alloc_compute_meta:        CPU compute buffer size =   161.31 MiB
-alloc_compute_meta: graph splits = 1, nodes = 388
+alloc_compute_meta:     OpenCL compute buffer size =   161.31 MiB
+alloc_compute_meta:        CPU compute buffer size =    49.72 MiB
+alloc_compute_meta: graph splits = 3, nodes = 388
 warmup: flash attention is enabled
+warmup: *****************************************************************
+warmup: WARNING: the CLIP graph uses unsupported operators by the backend
+warmup:          list of unsupported ops (backend=OpenCL):
+warmup:          UPSCALE: type = f32, ne = [92 92 768 1]
+warmup: *****************************************************************

-image slice encoded in 1270 ms
+image slice encoded in 511 ms

-An elephant stands on a white background.
+ A close-up of a textured, gray, possibly metallic or stone-like surface with a subtle, repeating pattern.

-llama_perf_context_print: prompt eval time =    1964.53 ms /   265 tokens (    7.41 ms per token,   134.89 tokens per second)
+llama_perf_context_print: prompt eval time =    1217.26 ms /   265 tokens (    4.59 ms per token,   217.70 tokens per second)
```

**Key structural difference:** Same 388 graph nodes. CPU path = 1 split (no transitions). OpenCL path = 3 splits (`OpenCL → CPU [UPSCALE] → OpenCL`). The 49.72 MiB CPU buffer is allocated solely for the fallback UPSCALE node + transfer scaffolding.

---

## Risks & Mitigations

### Risk 1: Fix lands but accuracy is still wrong

**Scenario:** The antialias kernel is implemented correctly, `graph splits = 1` on OpenCL, no warnings — but the output still hallucinates.

**What it would mean:** The corruption is not caused by the graph split / data transfer. Something else in the OpenCL CLIP graph is numerically divergent (e.g., fp16 accumulation in attention, conv2d precision, or a different unsupported op that doesn't trigger the warning for non-warmup image sizes).

**Mitigation:**
- Only investigate after the fix is implemented and tested. If output is still wrong post-fix, then compare raw CLIP embeddings between CPU and OpenCL paths.
- Do not attempt embedding dumps or layer-by-layer bisection before the fix — requires code changes and the fix itself is the most likely resolution.

**Likelihood:** Low. The CPU projector is correct in 6/6 configurations, and UPSCALE is the only flagged unsupported op. But not zero — the CLIP graph also runs conv2d, attention, GELU, layernorm on OpenCL; any of these could have subtle numerical differences that are masked when position embeddings are correct.

**Time impact if this risk materialises:** High. Debugging a numerical divergence across 388 graph nodes on a remote Firebase device with 30-45 min round-trips could add **3–5 additional days** (embedding dumps, layer-by-layer bisection, each requiring code changes and redeploys). This would need to be scoped as a separate follow-up task.

### Risk 2: Antialias kernel produces different results than CPU reference

**Scenario:** The kernel is implemented but uses slightly different floating-point order of operations, producing a numerically different (but not necessarily wrong) result compared to the CPU backend.

**What it would mean:** The CLIP position embeddings will be slightly different from the CPU path. For UPSCALE of a 48×48 → 92×92 grid with 768/1024 channels, even small differences propagate through 12-24 transformer layers.

**Mitigation:**
- Validate the kernel against the CPU reference by computing max absolute error and cosine similarity on the UPSCALE output tensor.
- Acceptable threshold: max abs error < 1e-5 (f32 precision), cosine similarity > 0.9999.
- The CUDA and Vulkan kernels use the same algorithm and produce correct results on those backends, so a faithful port should match.

### Risk 3: Adreno-specific performance issue with dynamic loop bounds

**Scenario:** The antialias kernel has a variable-radius support window (loop bound depends on scale factor). On Adreno GPUs, divergent loop counts across workgroup threads can cause severe performance degradation.

**What it would mean:** The kernel is correct but slower than CPU, defeating the purpose.

**Mitigation:**
- For the Qwen3VL case, the scale factor is fixed per image (48→92, sf ≈ 1.9×). At sf > 1, `support = max(1/sf, 1) = 1`, so the loop iterates over at most a 2×2 window — same as plain bilinear. Performance should be nearly identical to the existing `kernel_upscale_bilinear`.
- If downscaling cases arise (sf < 1), the loop radius grows. Profile on device and consider capping the support radius or tiling.

### Risk 4: The graph split itself is not the corruption mechanism

**Scenario:** The CPU fallback actually computes UPSCALE correctly, and the corruption comes from a buffer reuse / lifetime issue in the scheduler when transitioning between backends mid-graph.

**What it would mean:** The fix still resolves it (by eliminating the split entirely), but the root cause is a scheduler bug rather than a wrong UPSCALE result. This matters for other models that might hit different unsupported ops.

**Mitigation:**
- After the fix, if we encounter other unsupported ops on OpenCL in the future, test whether the CPU fallback path produces correct results in isolation. If not, file upstream against `ggml_backend_sched`.

---

## Upstream References

| Issue | Context |
|-------|---------|
| [llama.cpp #16842](https://github.com/ggml-org/llama.cpp/issues/16842) | Qwen-VL requires ≥1024 image tokens for grounding |
| [llama.cpp #16837 (comment)](https://github.com/ggml-org/llama.cpp/pull/16837#issuecomment-3461676118) | CLIP unsupported ops warning mechanism |
| [llama.cpp #19543](https://github.com/ggml-org/llama.cpp/issues/19543) | Same class of issue on SYCL |
| [llama.cpp #20011](https://github.com/ggml-org/llama.cpp/issues/20011) | Same class of issue on Metal |
| [llama.cpp #21268](https://github.com/ggml-org/llama.cpp/issues/21268) | CLIP unsupported ops (generic) |
| [llama.cpp #21941](https://github.com/ggml-org/llama.cpp/issues/21941) | CLIP unsupported ops (OpenCL specific) |

---

## Appendix: Kernel Algorithm (pseudocode)

```c
// For each output element dst[x_dst, y_dst, c, b]:
float y = (y_dst + 0.5) / sf1;
float x = (x_dst + 0.5) / sf0;

float support_y = max(1.0 / sf1, 1.0);
float support_x = max(1.0 / sf0, 1.0);

float val = 0, total_weight = 0;
for (sy = max(0, y - support_y + 0.5); sy < min(src_h, y + support_y + 0.5); sy++) {
    float wy = triangle(|sy - y + 0.5| / support_y);
    for (sx = max(0, x - support_x + 0.5); sx < min(src_w, x + support_x + 0.5); sx++) {
        float wx = triangle(|sx - x + 0.5| / support_x);
        val += src[sx, sy, c, b] * wx * wy;
        total_weight += wx * wy;
    }
}
dst[x_dst, y_dst, c, b] = val / total_weight;

// where triangle(t) = max(1 - |t|, 0)
```
