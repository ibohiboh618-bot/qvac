# FLUX.2 Klein Vulkan profiling notes

## Purpose

Internal profiling notes for QVAC-18818. They capture baseline observations used to guide Vulkan optimization work on `diffusion-cpp`. This is **not** user-facing SDK documentation.

## Benchmark configuration

| Field | Value |
|-------|-------|
| Diffusion model | `flux-2-klein-4b-Q8_0.gguf` |
| Text encoder | `Qwen3-4B-Q4_K_M.gguf` |
| VAE | `flux2-vae.safetensors` |
| Config | `threads=4`, `diffusion_fa=true`, `verbosity=2` |
| Prompt | a majestic red fox standing in a snowy forest at dusk, soft golden light through the pine trees, photorealistic, 8k, detailed fur |
| Seed | 42 |
| Resolution | 512×512 |
| Steps | 20 |
| Guidance | 3.5 |

Runs used the FLUX.2 klein example configuration (`examples/generate-image.js` parameters) with native stage timers enabled via `verbosity: 2` and `addonLogging.setLogger`.

## Hardware

| Machine | GPU | Driver / backend | Notes |
|---------|-----|------------------|-------|
| Strix Halo (integrated) | AMD Radeon 8060S (RADV GFX1151) | Vulkan, KHR_coopmat, UMA | Primary optimization target |
| RTX5090 workstation | NVIDIA GeForce RTX 5090 | Vulkan, NV_coopmat2 | Reference dGPU baseline; two GPUs visible, **only GPU0 used** |

Stock ggml from vcpkg pin `2026-01-30#8` unless noted otherwise.

## Stage timings

| Hardware | Text encoder | Sampling | VAE decode | Total generation |
|----------|--------------|----------|------------|------------------|
| Strix (stock) | 0.56 s | 38.23 s | 0.62 s | 39.5 s (`generate_image` 39.42 s) |
| Strix (patched) | — | 25.47 s | — | — |
| RTX5090 (stock) | 3.95 s | 5.16 s | 0.34 s | 9.5 s (`generate_image` 9.45 s) |

Strix generation times were stable across repeated runs (within ~2–3%). GPU utilization on Strix was 90–100% during sampling.

## Strix Vulkan bottleneck

- Diffusion **sampling is ~97% of `generate_image`**; text encoding (~1.4%) and VAE decode (~1.6%) are not the limiter.
- Vulkan perf logger (`GGML_VK_PERF_LOGGER=1`) on the denoise path showed:

| Op family | Share of logged GPU op time |
|-----------|----------------------------|
| **Matmul** (`MUL_MAT*`, mostly Q8_0) | **70.6%** |
| Copy / layout (`CONT`, `CONCAT`, `REPEAT`) | 12.1% |
| Elementwise (`MUL`, `ADD`, `SILU`, …) | 10.9% |
| **Flash attention** | **6.0%** |
| Norm | 0.4% |

- Largest single kernels: `MUL_MAT q8_0 m=27648 n=1536 k=3072` and `m=18432 n=1024 k=3072` (FLUX joint-attention / MLP shapes).
- Disabling cooperative matrices (`GGML_VK_DISABLE_COOPMAT=1`) caused ~2.3× slower sampling; graph-optimization and host-memory env toggles were neutral.

## Optimization finding

An AMD UMA Q8_0 MMQ workgroup-denom change in ggml-vulkan improved Strix FLUX.2 klein sampling:

| | Sampling |
|---|----------|
| Strix stock | ~38.23 s |
| Strix patched (avg) | ~25.47 s |
| Improvement | **~33% faster** |

The code change is isolated in [qvac-ext-ggml PR #16](https://github.com/tetherto/qvac-ext-ggml/pull/16). It is gated on **AMD + cooperative matrices + non-proprietary driver + UMA**, so dedicated AMD dGPU behavior is intentionally excluded until tested separately.

Follow-up rollout (not in this document): vcpkg registry bump → QVAC pin update.

## RTX5090 reference baseline

Reference baseline only — **no optimization patch applied**.

| Metric | Value |
|--------|-------|
| Sampling | 5.16 s |
| `generate_image` | 9.45 s |
| Max VRAM | 8317 MiB |
| GPU used | GPU0 / Vulkan0 only |

RTX5090 stock sampling is ~7.4× faster than Strix stock and ~4.9× faster than Strix patched. This confirms the Strix bottleneck class is UMA / bandwidth-bound rather than a general Vulkan regression on fast dGPU.

## Conclusions

- **VAE is not the bottleneck** on either platform tested.
- **Flash attention is not the main bottleneck on Strix** (~6% of logged op time vs ~71% matmul).
- **Q8_0 Vulkan matmul dominates** the denoise path on Strix.
- Further work should stay in **small isolated PRs**:
  1. [qvac-ext-ggml PR #16](https://github.com/tetherto/qvac-ext-ggml/pull/16) — AMD UMA MMQ denom patch
  2. vcpkg registry bump (after merge)
  3. QVAC pin update
  4. Separate MoE / LLM follow-up if needed (Strix LLM benchmarks showed dense-model gains but MoE pp512 regression)
