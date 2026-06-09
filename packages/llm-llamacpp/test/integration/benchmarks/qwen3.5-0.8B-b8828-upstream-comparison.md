# Qwen3.5 VLM Benchmark — Upstream llama.cpp (b8828 / fcc7508)

## Test Environment

| Parameter | Value |
|-----------|-------|
| Device | Samsung Galaxy S25 Ultra (Snapdragon 8 Elite / Adreno 830, OpenCL 3.0) |
| Build | Firebase Test Lab (console build) |
| Models | Qwen3.5-0.8B, Qwen3.5-2B |
| Quantizations | Q8_0, Q4_K_M, Q4_0 |
| Projector | mmproj-Qwen3.5-{0.8B,2B}-F16.gguf |
| Image | elephant.jpg |
| GPU layers | 98 |
| Context | 4096 |
| Temp | 0 |
| Seed | 42 |
| Prompt | "Describe the image briefly in one sentence." |
| CLI | llama-mtmd-cli |
| Backend | OpenCL (Adreno) |
| Commit | fcc7508 (b8828 base) |

---

## Why `--image-min-tokens`?

From the llama.cpp logs:

> **Qwen-VL models require at minimum 1024 image tokens to function correctly on grounding tasks.**
> If you encounter problems with accuracy, try adding `--image-min-tokens 1024`.

When `--image-min-tokens 1024` is set, `image_min_pixels` is overridden from the default `8192` to `1048576` (custom value), which forces the image to be encoded at higher resolution with more tokens (1080 vs 247).

- **Reference issue:** [ggml-org/llama.cpp#16842](https://github.com/ggml-org/llama.cpp/issues/16842)
- **Related fix (UPSCALE op):** [ggml-org/llama.cpp#16837 (comment)](https://github.com/ggml-org/llama.cpp/pull/16837#issuecomment-3461676118)

---

## Full Accuracy Comparison (All Models & Quants)

### Qwen3.5-0.8B (n_embd=768, UPSCALE shape: [92 92 768 1])

| Quant | Projector | CLIP Backend | Graph Splits | Image Encode | Prompt Eval (t/s) | Eval (t/s) | Total | Output | Correct? |
|-------|-----------|-------------|-------------|-------------|-------------------|-----------|-------|--------|----------|
| Q8_0 | OpenCL | OpenCL | 3 | 512 ms | 215.17 | 33.82 | 2333 ms | "A close-up of a textured, gray, possibly metallic or stone-like surface with a subtle, repeating pattern." | **No** |
| Q8_0 | CPU | CPU | 1 | 1065 ms | 151.19 | 35.23 | 2160 ms | "An elephant stands on a white background." | **Yes** |
| Q4_K_M | OpenCL | OpenCL | 3 | 528 ms | 194.14 | 35.61 | 2739 ms | "The image shows a close-up of a textured, light-colored surface, possibly a fabric or paper, with a subtle pattern of darker lines or folds running diagonally across it." | **No** |
| Q4_K_M | CPU | CPU | 1 | 1227 ms | 131.01 | 39.22 | 2665 ms | "An elephant stands against a white background, showcasing its wrinkled gray skin and large ears." | **Yes** |
| Q4_0 | OpenCL | OpenCL | 3 | 523 ms | 211.85 | 38.31 | 2194 ms | "The image displays a close-up of a textured, light-colored surface with intricate patterns and a slightly reflective quality." | **No** |
| Q4_0 | CPU | CPU | 1 | 1338 ms | 130.88 | 41.08 | 2535 ms | "The image shows a large elephant standing against a white background." | **Yes** |

### Qwen3.5-2B (n_embd=1024, UPSCALE shape: [92 92 1024 1])

| Quant | Projector | CLIP Backend | Graph Splits | Image Encode | Prompt Eval (t/s) | Eval (t/s) | Total | Output | Correct? |
|-------|-----------|-------------|-------------|-------------|-------------------|-----------|-------|--------|----------|
| Q8_0 | OpenCL | OpenCL | 3 | 1470 ms | 109.01 | 20.79 | 4505 ms | "A 3D-rendered elephant with a repeating, fractal-like pattern on its trunk and body." | **Partial** |
| Q8_0 | CPU | CPU | 1 | 3518 ms | 58.88 | 20.30 | 6120 ms | "A large African elephant stands facing forward with its trunk hanging down and tusks visible against a plain white background." | **Yes** |
| Q4_K_M | OpenCL | OpenCL | 3 | 1489 ms | 98.23 | 17.63 | 4998 ms | "A 3D rendered elephant with a repeating, fractal-like pattern on its trunk and body." | **Partial** |
| Q4_K_M | CPU | CPU | 1 | 3241 ms | 59.47 | 18.03 | 6350 ms | "A large African elephant stands against a white background, facing forward with its trunk hanging down and tusks visible." | **Yes** |
| Q4_0 | OpenCL | OpenCL | 3 | 1475 ms | 109.56 | 18.50 | 5030 ms | "A surreal, layered image of an elephant's trunk and head, with each successive layer slightly offset and fading into the background, creating a sense of depth and repetition." | **Partial** |
| Q4_0 | CPU | CPU | 1 | 3413 ms | 60.95 | 19.09 | 6135 ms | "A large African elephant stands prominently against a stark white background, showcasing its wrinkled gray skin, massive ears, and curved trunk." | **Yes** |

### Accuracy Summary

| | OpenCL Projector (graph splits = 3) | CPU Projector (graph splits = 1) |
|---|---|---|
| **0.8B** (all quants) | Complete hallucination — cannot identify elephant at all | Correct — identifies elephant every time |
| **2B** (all quants) | Partial — recognizes elephant but hallucinates "fractal patterns", "3D rendered", "surreal layered" artifacts | Correct — accurate detailed descriptions |

**Conclusion:** The 2B model's larger capacity allows it to partially recover from the corrupted embeddings (it can still identify the subject), but it still hallucinates visual artifacts that don't exist. The 0.8B model fails completely. Both are fully correct with CPU projector.

---

## Detailed Results (0.8B, --image-min-tokens variants)

### Summary Table (0.8B Q8_0 only, with image-min-tokens variants)

| # | Config | Image Encode | Image Decode | Prompt Eval (t/s) | Eval (t/s) | Total Time | Tokens | Output Correct? |
|---|--------|-------------|-------------|-------------------|-----------|-----------|--------|-----------------|
| 1 | CPU projector, default tokens | 1063 ms | 295 ms | 149.82 | 33.27 | 2273 ms | 277 | Yes |
| 2 | CPU projector, min-tokens 1024 | 7798 ms | 4747 ms | 83.74 | 27.77 | 13684 ms | 1110 | Yes |
| 3 | OpenCL projector, default tokens | 506 ms | 292 ms | 219.24 | 33.50 | 2255 ms | 291 | **No** (hallucination) |
| 4 | OpenCL projector, min-tokens 1024 | 3337 ms | 4690 ms | 127.93 | 28.15 | 9968 ms | 1129 | Partial |

---

## Detailed Results

### Test 1 — CPU Projector, Default Tokens (no --image-min-tokens)

```
--no-mmproj-offload
```

| Metric | Value |
|--------|-------|
| Image tokens | 247 |
| Image encode | 1063 ms |
| Image decode | 295 ms |
| Prompt eval | 1768.79 ms / 265 tokens (6.67 ms/token, **149.82 t/s**) |
| Eval | 360.69 ms / 12 runs (30.06 ms/token, **33.27 t/s**) |
| Total | 2272.82 ms / 277 tokens |
| CLIP backend | CPU |

**Output:** "An elephant stands on a white background."

---

### Test 2 — CPU Projector, --image-min-tokens 1024

```
--no-mmproj-offload --image-min-tokens 1024
```

| Metric | Value |
|--------|-------|
| Image tokens | 1080 |
| Image encode | 7798 ms |
| Image decode | 4747 ms |
| Prompt eval | 13112.05 ms / 1098 tokens (11.94 ms/token, **83.74 t/s**) |
| Eval | 432.20 ms / 12 runs (36.02 ms/token, **27.77 t/s**) |
| Total | 13684.12 ms / 1110 tokens |
| CLIP backend | CPU |
| image_min_pixels | 1048576 (custom value) |

**Output:** "An elephant stands against a white background."

---

### Test 3 — OpenCL Projector, Default Tokens (no --image-min-tokens)

```
(default, no --no-mmproj-offload)
```

| Metric | Value |
|--------|-------|
| Image tokens | 247 |
| Image encode | 506 ms |
| Image decode | 292 ms |
| Prompt eval | 1208.70 ms / 265 tokens (4.56 ms/token, **219.24 t/s**) |
| Eval | 776.12 ms / 26 runs (29.85 ms/token, **33.50 t/s**) |
| Total | 2255.37 ms / 291 tokens |
| CLIP backend | OpenCL |

**Output:** "A close-up of a textured, gray, possibly metallic or stone-like surface with a subtle, repeating pattern."

> **WARNING:** CLIP graph uses unsupported operator `UPSCALE` (f32, [92 92 768 1]) on OpenCL backend. Output is a **hallucination** — the image was not correctly processed.

---

### Test 4 — OpenCL Projector, --image-min-tokens 1024

```
--image-min-tokens 1024
```

| Metric | Value |
|--------|-------|
| Image tokens | 1080 |
| Image encode | 3337 ms |
| Image decode | 4690 ms |
| Prompt eval | 8582.99 ms / 1098 tokens (7.82 ms/token, **127.93 t/s**) |
| Eval | 1101.29 ms / 31 runs (35.53 ms/token, **28.15 t/s**) |
| Total | 9968.49 ms / 1129 tokens |
| CLIP backend | OpenCL |
| image_min_pixels | 1048576 (custom value) |

**Output:** "An elephant's trunk is shown in a close-up view, with its wrinkled skin and textured surface clearly visible against a white background."

> **WARNING:** Same unsupported `UPSCALE` op on OpenCL. Output partially correct (mentions elephant) but still inaccurate in description.

---

## Known Issues & Fixes

### UPSCALE Op Not Supported on OpenCL (Tests 3 & 4)

The CLIP vision encoder graph for Qwen3.5 uses an `UPSCALE` operator with `GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS`. On the OpenCL backend this mode is **unsupported** (antialias flag rejected) and falls back to CPU, breaking the image encoding pipeline:

- 0.8B: `UPSCALE: type = f32, ne = [92 92 768 1]`
- 2B: `UPSCALE: type = f32, ne = [92 92 1024 1]`

```
warmup: WARNING: the CLIP graph uses unsupported operators by the backend
warmup:          the performance will be suboptimal
warmup:          list of unsupported ops (backend=OpenCL):
warmup:          UPSCALE: type = f32, ne = [92 92 768 1]
```

- **Issue/Fix:** [ggml-org/llama.cpp#16837 (comment)](https://github.com/ggml-org/llama.cpp/pull/16837#issuecomment-3461676118)
- **Impact:** Hallucinated output when projector runs on OpenCL without the fix

### image-min-tokens Requirement for Qwen-VL (Issue #16842)

Qwen-VL architecture needs a minimum number of image tokens for correct visual grounding. Without `--image-min-tokens 1024`, the model receives only 247 tokens which is insufficient for complex scenes.

- **Issue:** [ggml-org/llama.cpp#16842](https://github.com/ggml-org/llama.cpp/issues/16842)
- **Default** `image_min_pixels`: 8192 → produces 247 tokens
- **With** `--image-min-tokens 1024`: `image_min_pixels` = 1048576 → produces 1080 tokens
- **Trade-off:** Accuracy improves significantly but total inference time increases ~5-7x

---

## Key Observations

1. **100% accuracy failure rate with OpenCL projector** — Across 6 configurations (2 models × 3 quants), every single OpenCL projector run produces incorrect output. Every CPU projector run is correct. The UPSCALE graph split is the sole differentiator.

2. **0.8B completely hallucinates, 2B partially hallucinates** — The 0.8B model cannot even identify the elephant (describes "textured surfaces"). The 2B model recognizes the elephant but hallucinates "fractal patterns" and "3D rendering" artifacts that don't exist. This suggests the corrupted position embeddings provide enough spatial signal for larger models to partially recover.

3. **CPU projector is ~2.4x slower on image encoding** — 0.8B: 512 ms (OpenCL) vs 1210 ms (CPU avg). 2B: 1478 ms (OpenCL) vs 3390 ms (CPU avg). This is the speed gain we'd unlock by fixing UPSCALE on OpenCL.

4. **Prompt eval throughput** — OpenCL projector gives ~2x better prompt eval (215 t/s vs 131 t/s for 0.8B; 109 t/s vs 59 t/s for 2B) but the results are wrong.

5. **Quantization doesn't affect the accuracy bug** — Q8_0, Q4_K_M, and Q4_0 all show the same hallucination pattern with OpenCL projector. The bug is in the CLIP encoder graph split, independent of LLM quantization.

6. **Graph splits confirm the issue** — OpenCL projector: `graph splits = 3` (GPU→CPU→GPU at UPSCALE). CPU projector: `graph splits = 1` (no device transitions).

7. **Eval speed is unaffected** — Generation speed is consistent (~20 t/s for 2B, ~35 t/s for 0.8B) regardless of projector choice, since the LLM always runs on GPU.

8. **Recommended config for accuracy** — Use `--no-mmproj-offload` (CPU projector) until the OpenCL `UPSCALE` fix lands. Use `--image-min-tokens 1024` for grounding tasks that need spatial accuracy.
