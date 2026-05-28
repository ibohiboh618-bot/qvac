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
