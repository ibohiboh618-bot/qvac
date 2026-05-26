# FLUX.2 Klein — Vulkan optimization log (QVAC-18818)

Investigation notes and experiment tracking. Fill one row per change attempt.

---

## Experiment log

| task | hardware | git commit | build config | benchmark command | model/config | prompt/seed/resolution/steps | total runtime | per-stage runtime | peak RSS | peak VRAM | GPU utilization | attempted change | hypothesis | result | keep/revert decision | notes |
|------|----------|------------|--------------|-------------------|--------------|------------------------------|---------------|-------------------|----------|-----------|-------------------|------------------|------------|--------|----------------------|-------|
| | | | | | | | | | | | | | | | | |

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
