# FLUX.2 Klein — Vulkan optimization log (QVAC-18818)

Investigation notes and experiment tracking. Fill one row per change attempt.

---

## Experiment log

| task | hardware | git commit | build config | benchmark command | model/config | prompt/seed/resolution/steps | total runtime | per-stage runtime | peak RSS | peak VRAM | GPU utilization | attempted change | hypothesis | result | keep/revert decision | notes |
|------|----------|------------|--------------|-------------------|--------------|------------------------------|---------------|-------------------|----------|-----------|-------------------|------------------|------------|--------|----------------------|-------|
| | | | | | | | | | | | | | | | | |

---

## Strix baseline results (qvac-dev-strix-1)

Recorded on branch `feat/QVAC-18818-vulkan-flux2-optimization`. Raw run logs live under `packages/diffusion-cpp/logs/` (gitignored).

### Environment

| Field | Value |
|-------|-------|
| Machine | qvac-dev-strix-1 |
| Hardware | AMD Ryzen AI MAX+ 395 with Radeon 8060S Graphics / RADV GFX1151 |
| Git commit | `5b6ac029124dfde32cb69e893cda5e82b57fd266` |
| `VCPKG_ROOT` | `/home/dev/vcpkg` |
| `MESA_VK_DEVICE_SELECT` | `1002:1586` |
| `MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE` | `1` |

### Build

```bash
cd packages/diffusion-cpp
npm install
bare-make generate
bare-make build
bare-make install
```

### Model / run config

| Field | Value |
|-------|-------|
| Diffusion | `flux-2-klein-4b-Q8_0.gguf` |
| Text encoder | `Qwen3-4B-Q4_K_M.gguf` |
| VAE | `flux2-vae.safetensors` |
| Config | `threads=4`, `diffusion_fa=true` |
| Benchmark | `npm run generate` |
| Prompt | a majestic red fox standing in a snowy forest at dusk, soft golden light through the pine trees, photorealistic, 8k, detailed fur |
| Seed | 42 |
| Resolution | 512x512 |
| Steps | 20 |

### Runs

| Run | Generation | Elapsed | Peak RSS | GPU busy (max / avg) | Max VRAM | Max GTT | Exit |
|-----|------------|---------|----------|----------------------|----------|---------|------|
| Baseline 1 | 39.8 s | 42.365 s | 2230.2 MB | not captured | — | — | 0 |
| Baseline 2 | 38.9 s | 41.666 s | 1737.1 MB | 100% / 89.9% | 2037 MiB | 8153 MiB | 0 |
| Baseline 3 | 38.9 s | 41.367 s | 1697.0 MB | 100% / 90.1% | 2037 MiB | 8153 MiB | 0 |

### Conclusion

- Vulkan is definitely active.
- Logs show: `ggml_vulkan: Found 1 Vulkan devices`.
- Device used: Radeon 8060S Graphics (RADV GFX1151).
- Timing is stable: generation within about 2.3%, elapsed within about 2.4%.
- Main current bottleneck hypothesis: diffusion steps dominate. GPU is 90–100% busy, so this is likely Vulkan/ggml kernel or memory bandwidth related, not backend selection or CPU fallback.
- Per-stage timing is still missing, so next step should be instrumentation/profiling to split text encoder / diffusion / VAE decode before optimization.

---

## Strix verbosity profiling

Recorded on `qvac-dev-strix-1`, branch `feat/QVAC-18818-vulkan-flux2-optimization`, commit `f95317b5`. Raw artifacts: `packages/diffusion-cpp/logs/flux2_strix_verbosity2_profile.log`, `flux2_strix_verbosity2_profile_sysfs.csv` (gitignored). Temporary harness: `packages/diffusion-cpp/logs/generate-image-verbosity2.js` (same prompt/seed/512²/20 steps/guidance 3.5 as baseline; `verbosity: 2`; routes native logs via `addonLogging.setLogger`).

### Command

```bash
cd ~/Projects/qvac/packages/diffusion-cpp
export VCPKG_ROOT="$HOME/vcpkg"
export MESA_VK_DEVICE_SELECT=1002:1586
export MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1
# sysfs sampler (1 Hz) + run (see logs/run_verbosity_profile.sh)
bare logs/generate-image-verbosity2.js 2>&1 | tee logs/flux2_strix_verbosity2_profile.log
```

### Verbosity level

- **`verbosity: 2` is sufficient** for upstream `LOG_INFO` stage timers once `addonLogging.setLogger` is configured.
- Without `setLogger`, neither verbosity 2 nor 3 emitted upstream timing lines (only JS wrapper + `ggml_vulkan` device enumeration). Verbosity 3 was not required after wiring the logger.

### Timings (upstream + JS)

| Stage | Upstream log | Time |
|-------|----------------|------|
| Model load (JS) | `Loaded in` | 2.0 s |
| Text encoder | `get_learned_condition completed, taking 559 ms` | 0.56 s (single line; guidance 3.5, one conditioner pass) |
| Diffusion sampling | `sampling completed, taking 38.23s` | 38.23 s |
| Latent generation window | `generating 1 latent images completed, taking 38.23s` | 38.23 s (matches sampling; denoise loop) |
| VAE decode | `latent 1 decoded, taking 0.62s` / `decode_first_stage completed, taking 0.62s` | 0.62 s |
| Upstream total | `generate_image completed in 39.42s` | 39.42 s |
| JS generation | `Generated in` | 39.5 s |
| PNG / addon overhead (approx.) | JS gen − upstream `generate_image` | ~0.08 s |

Vulkan device (unchanged): `ggml_vulkan: 0 = Radeon 8060S Graphics (RADV GFX1151) (radv)`.

### sysfs (41 samples @ 1 Hz during run)

| Metric | Value |
|--------|-------|
| GPU busy max / avg | 100% / 90.5% |
| Max VRAM | 2037 MiB |
| Max GTT | 8150 MiB |

### Conclusion

- **Bottleneck is diffusion sampling (~97% of `generate_image`)**, not text encoding (~1.4%) or VAE decode (~1.6%).
- GPU remains saturated (90–100% busy) with the same VRAM/GTT footprint as baseline 2–3, consistent with Vulkan/ggml kernel or memory-bandwidth limits on the denoise path.
- PNG encode and JS/addon wrapper overhead are negligible on this run (~80 ms).

---

## Strix sampling bottleneck investigation

Recorded on `qvac-dev-strix-1`, commit `e9cde7d3`. Same benchmark as verbosity profiling (`bare logs/generate-image-verbosity2.js`, `threads=4`, `diffusion_fa=true`, seed 42, 512², 20 steps). No tracked source changes; experiment artifacts under `packages/diffusion-cpp/logs/` (gitignored).

### Files inspected (pinned vcpkg buildtrees on Strix)

| Component | Path |
|-----------|------|
| Upstream txt2img / sampling | `/home/dev/vcpkg/buildtrees/stable-diffusion-cpp/src/f028dfb9fe-865f9e2efc.clean/src/stable-diffusion.cpp` |
| Diffusion API | `.../src/diffusion_model.hpp`, `.../src/flux.hpp` |
| Attention / FA graph | `.../src/ggml_extend.hpp` |
| ggml Vulkan backend | `/home/dev/vcpkg/buildtrees/ggml/src/45bc58f723-cbfedfc90c.clean/src/ggml-vulkan/ggml-vulkan.cpp` |
| ggml scheduler / node debug | `.../src/ggml-backend.cpp` (`GGML_SCHED_DEBUG`) |
| Addon `diffusion_fa` → upstream | `packages/diffusion-cpp/addon/src/model-interface/SdModel.cpp` (`params.diffusion_flash_attn`), `addon/src/handlers/SdCtxHandlers.cpp` |

### Sampling call path (confirmed)

1. `generate_image_internal()` → `sd_ctx->sd->sample(...)` (`stable-diffusion.cpp` ~3555).
2. Per-step denoise loop → `work_diffusion_model->compute(n_threads, diffusion_params, &out_cond)` (~2179).
3. FLUX.2 klein → `FluxModel::compute()` → `flux.compute(...)` (`diffusion_model.hpp` / `flux.hpp` ~1464).
4. Graph build uses `ggml_extend` attention; with `diffusion_flash_attn` → `ggml_flash_attn_ext` when constraints pass (`ggml_extend.hpp` ~1360–1372).
5. Execution → ggml Vulkan (`ggml_vulkan.cpp`): matmul (`ggml_vk_mul_mat*`) + `ggml_vk_flash_attn()` for `GGML_OP_FLASH_ATTN_EXT`.

### `diffusion_fa` and flash attention (confirmed active)

- JS `diffusion_fa: true` → addon `SdCtxConfig::diffusionFlashAttn` → `sd_ctx_params_t.diffusion_flash_attn` (`SdModel.cpp`).
- Upstream enables diffusion-only FA when `flash_attn || diffusion_flash_attn` (`stable-diffusion.cpp` ~862–864).
- Run logs always include: `Using flash attention in the diffusion model`.
- Device line: `fp16: 1`, `bf16: 0`, `matrix cores: KHR_coopmat` (baseline / coopmat-on runs).
- ggml FA path selects `FA_COOPMAT1` when `coopmat1_fa_support` + shmem/shape checks pass, else `FA_SCALAR` (`ggml-vulkan.cpp` ~8467–8523).

### ggml Vulkan env vars in pinned source (`ggml-vulkan.cpp`)

Present in this tree (do not use names not listed here):

`GGML_VK_PREFER_HOST_MEMORY`, `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM`, `GGML_VK_ALLOW_SYSMEM_FALLBACK`, `GGML_VK_DISABLE_GRAPH_OPTIMIZE`, `GGML_VK_DISABLE_COOPMAT`, `GGML_VK_DISABLE_COOPMAT2`, `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT`, `GGML_VK_DISABLE_BFLOAT16`, `GGML_VK_ENABLE_MEMORY_PRIORITY`, `GGML_VK_DISABLE_ASYNC`, `GGML_VK_DISABLE_F16`, `GGML_VK_DISABLE_FUSION`, `GGML_VK_DISABLE_MMVQ`, `GGML_VK_FORCE_MMVQ`, `GGML_VK_DISABLE_MULTI_ADD`, `GGML_VK_FORCE_MAX_ALLOCATION_SIZE`, `GGML_VK_FORCE_MAX_BUFFER_SIZE`, `GGML_VK_SUBALLOCATION_BLOCK_SIZE`, `GGML_VK_DEBUG_MARKERS`, `GGML_VK_PERF_LOGGER`, `GGML_VK_PERF_LOGGER_CONCURRENT`, `GGML_VK_PERF_LOGGER_FREQUENCY`, `GGML_VK_SYNC_LOGGER`, `GGML_VK_MEMORY_LOGGER`, `GGML_VK_VISIBLE_DEVICES`, `GGML_VULKAN_SKIP_CHECKS`, `GGML_VULKAN_OUTPUT_TENSOR`.

Related (other ggml files): `GGML_DISABLE_VULKAN` (`ggml-backend-reg.cpp`), `GGML_SCHED_DEBUG` / `GGML_SCHED_DEBUG_REALLOC` (`ggml-backend.cpp`), `GGML_OP_OFFLOAD_MIN_BATCH`.

**Profiling-only (not run as timing experiments):** `GGML_VK_PERF_LOGGER=1` logs per-op Vulkan timings with tensor `name` / fusion labels (`vk_perf_logger` in `ggml-vulkan.cpp`).

### Env experiments (max 3; same benchmark)

| Run | Env override | Gen (s) | `generate_image` (s) | Sampling (s) | Text (s) | VAE (s) | GPU busy max/avg | VRAM max | GTT max | Coopmat | vs baseline |
|-----|----------------|---------|----------------------|--------------|----------|---------|------------------|----------|---------|---------|-------------|
| Baseline (verbosity profiling) | — | 39.5 | 39.42 | 38.23 | 0.56 | 0.62 | 100% / 90.5% | 2037 MiB | 8150 MiB | KHR_coopmat | — |
| 1 `disable_coopmat` | `GGML_VK_DISABLE_COOPMAT=1` | 89.2 | 89.12 | 87.48 | 0.99 | 0.65 | 100% / 95.5% | 2035 MiB | 7790 MiB | none | **~2.3× slower** |
| 2 `disable_graph_optimize` | `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1` | 39.4 | 39.34 | 38.17 | 0.56 | 0.61 | 100% / 90.5% | 2037 MiB | 8150 MiB | KHR_coopmat | neutral (~−0.2%) |
| 3 `prefer_host_memory` | `GGML_VK_PREFER_HOST_MEMORY=1` | 39.8 | 39.76 | 38.38 | 0.68 | 0.69 | 100% / 89.2% | 149 MiB* | 8551 MiB | KHR_coopmat | neutral (~+0.4%) |

\*UMA accounting shift: sysfs VRAM counter dropped while GTT rose; sampling time unchanged.

Logs: `flux2_strix_sampling_<name>.log`, sysfs: `flux2_strix_sampling_<name>_sysfs.csv`. Runner: `logs/run_sampling_experiment.sh` + `logs/experiment_env/<name>.env`.

### Conclusion

- **No env-only win** on Strix for this workload; baseline Vulkan path already uses cooperative matrices and graph optimization.
- **`GGML_VK_DISABLE_COOPMAT=1` is strongly harmful** (+129% sampling time): matmul + flash-attn fall back to non-coopmat paths on RADV GFX1151.
- Bottleneck remains **20× FLUX diffusion `compute()` per step** (flash-attn + Q8_0/BF16 matmul on Vulkan), not host memory preference or graph fusion toggles.
- AMD RDNA-class matmul warptile tuning already applies when `vendor_id==AMD && coopmat_support` (`ggml-vulkan.cpp` ~3029).

### Recommended next steps

1. **Diagnostic (no code):** one run with `GGML_VK_PERF_LOGGER=1` (and optional `GGML_VK_PERF_LOGGER_FREQUENCY=20`) to rank hot ops inside the diffusion step graph.
2. **First code-level candidate:** `ggml_vk_flash_attn()` in pinned `ggml-vulkan.cpp` (~8408–8664) — tune `FA_COOPMAT1` vs `FA_SCALAR` selection and AMD RDNA3 + RADV shmem/warptile paths for FLUX joint attention (24 heads, 512² latent). **Risk: medium** (shader path changes; must not regress other GPUs; coopmat disable proved critical).
3. **Second candidate:** AMD `l_warptile` / `l_warptile_mmq` block for `vendor_id==AMD && coopmat_support` (~3029–3034) after perf logger identifies matmul-bound ops. **Risk: medium-high** (affects all Q8_0 diffusion matmuls).

**Not recommended without measurement:** disabling `diffusion_fa` (FLUX2 VRAM hazard), `GGML_VK_DISABLE_F16`, or CPU fallback.

---

## Reference (from repo inspection)

### Primary package

`packages/diffusion-cpp` — Bare addon over `qvac-ext-stable-diffusion.cpp` (vcpkg: `stable-diffusion-cpp`, `ggml`).

### FLUX.2 Klein touchpoints

| Area | Path |
|------|------|
| txt2img example | `packages/diffusion-cpp/examples/generate-image.js` |
| img2img examples | `packages/diffusion-cpp/examples/img2img-flux2.js`, `img2img-flux2-f16.js` |
| Model download | `packages/diffusion-cpp/scripts/download-model.sh` |
| Integration tests | `packages/diffusion-cpp/test/integration/generate-image-flux2*.test.js` |
| C++ load / run | `packages/diffusion-cpp/addon/src/model-interface/SdModel.cpp` (`generate_image`) |
| SDK example | `packages/sdk/examples/diffusion-flux2-klein.ts` |
| SDK plugin | `packages/sdk/server/bare/plugins/sdcpp-generation/` |

### Vulkan backend selection (addon)

`packages/diffusion-cpp/addon/src/utils/BackendSelection.cpp` → `preferred_gpu_backend` passed in `SdModel::load()` (`SdModel.cpp`). Non-Adreno Linux/Windows defaults to GPU (Vulkan via stable-diffusion `init_backend`). Adreno 800+ may prefer OpenCL.

### VAE

Executed inside upstream `generate_image()` (not split in addon). Addon configures `vae_path`, `vae_decode_only=false`, `keep_vae_on_cpu`, `vae_tiling` / `vae_conv_direct` via `SdCtxHandlers` / `SdGenHandlers`.

### Vulkan shaders / kernels

Not present in this monorepo. Built with ggml via vcpkg registry ports; Android ships `libqvac-diffusion-ggml-vulkan.so` (see `packages/diffusion-cpp/CMakeLists.txt`).

### Build (Linux, from source)

```bash
cd packages/diffusion-cpp
npm install
export VCPKG_ROOT=...   # see packages/diffusion-cpp/build.md
export VULKAN_SDK=...   # see packages/diffusion-cpp/build.md
npm run build
```

(`npm run build:vulkan` runs the same pipeline as `npm run build`; Vulkan is enabled via default vcpkg features on ggml / stable-diffusion-cpp per `CMakeLists.txt`.)

### Run / measure (no dedicated flux2 Vulkan benchmark in repo)

```bash
cd packages/diffusion-cpp
./scripts/download-model.sh
npm run generate
# or: bare examples/img2img-flux2.js
# integration: npm run test:integration (includes generate-image-flux2.test.js)
```

SDK:

```bash
cd packages/sdk
bun run build
bun run bare:example dist/examples/diffusion-flux2-klein.ts
```

### Profiling / stats

- `opts: { stats: true }` → `RuntimeStats` (`modelLoadMs`, `generationMs`, …) — no per-stage VAE/diffusion split (`packages/diffusion-cpp/index.d.ts`).
- `config.verbosity` 0–3 + `addonLogging.setLogger` for native logs; backend enumeration in `BackendLoader.cpp`.
- Ad-hoc timing: `scripts/compare-flux-models.sh` (`time bare examples/...`).
- C++ coverage: `npm run coverage:cpp` (not flux2-specific).

### Not in repo

- QVAC-18818 ticket text
- Dedicated flux2-klein Vulkan benchmark script
- Peak RSS/VRAM/GPU utilization APIs in diffusion-cpp `RuntimeStats`
- Vulkan `.comp` / shader source paths (upstream ggml vcpkg build tree only)
