# LTX-2.3 on NVIDIA (CUDA) — build + reproduce the Metal/Vulkan diagnostics

Runbook to build `diffusion-cpp` with the **CUDA** backend on an NVIDIA box (e.g. RTX 5090)
on branch `feature/ltx-diag`, capture the same diagnostics we gathered on Metal + Vulkan, and
settle which "basin" CUDA lands in for the cat-vs-human bistable prompt.

Branch must include commit `5ab5f1dc` (or later) and these 5 vcpkg patches:
`0001-attn-diag` `[ATTN-DIAG]`, `0002-latent-diag` `[LATENT-DIAG]`, `0003-vae-decode-diag`
`[DECODE-DIAG]`, `0004-decode-only-latent-dump-load` `[DECODE-ONLY-DIAG]`,
`0005-cond-diag` `[COND-DIAG]` (+ `LTX_DUMP_COND`/`LTX_LOAD_COND`). Overlay port is
`stable-diffusion-cpp@2026-06-04#6`.

---

## 0. TL;DR — what we already know (the thing to confirm on CUDA)

- Same seed/params produce a **different but coherent** video per backend; it's GPU fp
  non-determinism, not a bug.
- At **512×320** the prompt subject is robust → **cat** on both Metal and Vulkan.
- At **768×512 / CFG=1 / 8 steps** the subject is **bistable** (cat vs human). Metal lands on
  **cat**; Vulkan (RADV) lands on **human**.
- We isolated the tipping factor to the **DiT flash-attention kernel**: VAE ruled out
  (identical-latent decode matches to 0.07%), Gemma ruled out (loading Metal's exact cond on
  Vulkan still gave a human). On Vulkan, **`DIFFUSION_FA=false` (F32 attention) recovers the cat.**

**Question for CUDA:** at 768×512 with flash-attn ON, does the 5090 land on **cat** (like Metal)
or **human** (like Vulkan)? And does `DIFFUSION_FA=false` change it?

---

## 1. Prerequisites (NVIDIA / Linux)

- **NVIDIA driver + CUDA Toolkit**. For **RTX 5090 (Blackwell, sm_120) you need CUDA Toolkit ≥ 12.8**
  (older toolkits cannot generate sm_120 code). Verify: `nvidia-smi` and `nvcc --version`.
- **LLVM/Clang 22 with libc++**, **CMake 3.25+** (Linux toolchain this repo uses):
  ```bash
  wget -q https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && sudo ./llvm.sh 22 all
  ```
- **Vulkan SDK** — still required, because on Linux the `vulkan` ggml feature is a *default* and is
  compiled alongside CUDA (CUDA is added on top). glslc/shaderc come from the SDK:
  ```bash
  sudo apt install -y xz-utils
  wget -q -O /tmp/vulkansdk.tar.xz https://sdk.lunarg.com/sdk/download/latest/linux/vulkan_sdk.tar.xz
  mkdir -p ~/vulkan && cd ~/vulkan && tar xf /tmp/vulkansdk.tar.xz --strip-components=1
  export VULKAN_SDK=~/vulkan/x86_64
  sudo apt-get install -y libxi-dev libxtst-dev libxrandr-dev
  ```
- **bare toolchain + vcpkg pinned to 2025.12.12**:
  ```bash
  npm install -g bare bare-make
  cd ~ && git clone --branch 2025.12.12 --single-branch https://github.com/microsoft/vcpkg.git
  cd vcpkg && ./bootstrap-vcpkg.sh -disableMetrics && export VCPKG_ROOT=$(pwd)
  ```

> Keep `VCPKG_ROOT` (and `VULKAN_SDK`) exported in the shell you build in.

---

## 2. Get the branch + confirm patches

```bash
cd <repo>/qvac && git fetch origin && git checkout feature/ltx-diag && git pull
git log -1 --format='%H %s'            # expect 5ab5f1dc or later
cd packages/diffusion-cpp
ls vcpkg/ports/stable-diffusion-cpp/000*-*.patch
# expect: 0001-attn-diag-logging  0002-latent-diag-logging  0003-vae-decode-diag-logging
#         0004-decode-only-latent-dump-load  0005-cond-diag-logging
```

---

## 3. Build with the CUDA backend

```bash
export VCPKG_ROOT=/path/to/vcpkg          # must be set in THIS shell
npm install
npm run build:cuda                        # = bare-make generate -D SD_CUDA=ON && build && install
```

If the build can't auto-detect the 5090's arch, force it:
```bash
rm -rf build
bare-make generate -D SD_CUDA=ON -D CMAKE_CUDA_ARCHITECTURES=120
bare-make build && bare-make install
```

### Verify it actually rebuilt with the patches (guard against a stale binary-cache hit)
In the build output confirm ALL of:
```
-- Applying patch 0001-attn-diag-logging.patch
-- Applying patch 0002-latent-diag-logging.patch
-- Applying patch 0003-vae-decode-diag-logging.patch
-- Applying patch 0004-decode-only-latent-dump-load.patch
-- Applying patch 0005-cond-diag-logging.patch
Building stable-diffusion-cpp[...,cuda,...]:x64-linux@2026-06-04#6...
```
It must say **Building ... #6** (with `cuda` in the feature list), not only "Restored …". If it's a
stale cache, `touch vcpkg/ports/stable-diffusion-cpp/portfile.cmake` and rebuild.

---

## 4. Download weights (~34 GB, skips any already present)

```bash
./scripts/download-model-ltx.sh
```

---

## 5. Run A — canonical diagnostic capture (512×320 / 9f)

This is the control: the per-step noise is CPU Philox so **step 1 must match byte-for-byte**.

```bash
SEED=42 FRAMES=9 WIDTH=512 HEIGHT=320 STEPS=8 CFG_SCALE=1.0 \
PROMPT="a claymation cat playing jazz on a piano" \
npm run generate:ltx 2>&1 | tee /tmp/ltx_cuda_diag.log

grep -aE 'ATTN-DIAG|LATENT-DIAG|COND-DIAG|DECODE-DIAG' /tmp/ltx_cuda_diag.log
grep -aiE 'ggml_cuda|CUDA|device|backend' /tmp/ltx_cuda_diag.log | head
```

**Confirm CUDA is the active backend** (look for `ggml_cuda`/`CUDA0`/the 5090's name in the load
logs). With both CUDA+Vulkan compiled, stable-diffusion.cpp initializes CUDA first, so it should be
selected — but verify.

### Expected / reference values (compare CUDA against these)
```
[LATENT-DIAG] step=1  mean=-0.00654257 l2=206.174273 min=-4.395776 max=3.931298   <- MUST match
[ATTN-DIAG]   flash_attn ACTIVE (fused F16 flash_attn_ext) d_head=128 n_head=32 L_q=1024 L_k=1024
[COND-DIAG]   stage=cond numel=61440 ...   Metal l2=1429.004819 | Vulkan-GPU l2=1431.167567
[DECODE-DIAG] stage=video ...              Metal canonical-latent decode l2=644.982275
```
- If `[LATENT-DIAG] step=1` differs → stop; seeding/noise isn't comparable.
- Record CUDA's `[COND-DIAG]` l2 (where does it fall vs Metal 1429.00 / Vulkan 1431.17?).
- Later `[LATENT-DIAG]` steps will drift from both Metal and Vulkan — that's expected.

---

## 6. Run B — the bistable subject test (768×512), THE key experiment

Baseline (flash-attn ON, the default):
```bash
SEED=42 FRAMES=25 WIDTH=768 HEIGHT=512 STEPS=8 CFG_SCALE=1.0 NEG_PROMPT="" TEMPORAL_TILING=true \
PROMPT="a claymation cat playing jazz on a piano" \
npm run generate:ltx 2>&1 | tee /tmp/ltx_cuda_768_faON.log
cp output/ltx_t2v_seed42.avi output/cuda_768_faON.avi
```

Flash-attn OFF (F32 attention — the Vulkan cat-recovery lever):
```bash
DIFFUSION_FA=false SEED=42 FRAMES=25 WIDTH=768 HEIGHT=512 STEPS=8 CFG_SCALE=1.0 NEG_PROMPT="" TEMPORAL_TILING=true \
PROMPT="a claymation cat playing jazz on a piano" \
npm run generate:ltx 2>&1 | tee /tmp/ltx_cuda_768_faOFF.log
cp output/ltx_t2v_seed42.avi output/cuda_768_faOFF.avi
```

Then **look at the frames** (no ffmpeg needed — the AVI is MJPG; see §8) and report the subject:
**cat / human / hybrid** for each.

Interpretation:
- faON → **cat**: CUDA agrees with Metal (Vulkan is the odd one out at 768).
- faON → **human**, faOFF → **cat**: CUDA behaves like Vulkan; flash-attn is the tipping kernel on
  NVIDIA too.
- both human: subject is just unstable at 768/CFG=1 on this backend → use 512×320, more steps, or a
  cat-emphatic prompt.

---

## 7. (Optional) Cross-backend isolation tests

These need two binary files from the Strix diagnostic bundle (they live under `output/`, which is
**git-ignored**, so they are NOT in the repo — copy them onto the 5090, e.g. via scp):
`cond_metal.bin` (61440 floats) and `ltx_final_latent.bin` (40960 floats).

VAE-only (decode an identical Metal latent — isolates VAE kernels):
```bash
SEED=42 FRAMES=9 WIDTH=512 HEIGHT=320 STEPS=8 CFG_SCALE=1.0 \
LTX_LOAD_LATENT=/path/to/ltx_final_latent.bin \
PROMPT="a claymation cat playing jazz on a piano" npm run generate:ltx 2>&1 | tee /tmp/ltx_cuda_decodeonly.log
grep -aE 'DECODE-ONLY-DIAG|DECODE-DIAG' /tmp/ltx_cuda_decodeonly.log
# compare [DECODE-DIAG] stage=video to Metal l2=644.982275 (≈identical -> VAE kernels agree)
```

Conditioning-only (drive CUDA diffusion with Metal's exact cond — isolates Gemma):
```bash
SEED=42 FRAMES=25 WIDTH=768 HEIGHT=512 STEPS=8 CFG_SCALE=1.0 NEG_PROMPT="" TEMPORAL_TILING=true \
LTX_LOAD_COND=/path/to/cond_metal.bin \
PROMPT="a claymation cat playing jazz on a piano" npm run generate:ltx 2>&1 | tee /tmp/ltx_cuda_loadcond.log
# look at frames: if still NOT a cat, cond/Gemma is not the tipping factor on CUDA either
```

Gemma-on-CPU (does CUDA's GPU text-encode differ from CPU like Vulkan's did?):
```bash
CLIP_ON_CPU=true SEED=42 FRAMES=9 WIDTH=512 HEIGHT=320 STEPS=8 CFG_SCALE=1.0 \
PROMPT="a claymation cat playing jazz on a piano" npm run generate:ltx 2>&1 | tee /tmp/ltx_cuda_clipcpu.log
grep -a 'COND-DIAG' /tmp/ltx_cuda_clipcpu.log   # CPU encode ≈ Metal 1429.00 on Vulkan
```

> Skip `GGML_VK_*` env vars on CUDA — those are Vulkan-only (and full-F32 via `GGML_VK_DISABLE_F16`
> produced NaNs on RADV anyway).

---

## 8. Frame extraction helper (no ffmpeg required)

```python
# python3 - <<'PY'   (run from packages/diffusion-cpp/output)
import os, sys
avi = sys.argv[1] if len(sys.argv)>1 else 'ltx_t2v_seed42.avi'
data = open(avi,'rb').read()
frames=[]; i=0
while True:
    s=data.find(b'\xff\xd8\xff', i)
    if s<0: break
    e=data.find(b'\xff\xd9', s)
    if e<0: break
    e+=2
    if e-s>2000: frames.append((s,e))   # skip tiny non-frame matches
    i=e
out=os.path.splitext(os.path.basename(avi))[0]+'_frames'; os.makedirs(out,exist_ok=True)
for fi in [0, len(frames)//2, len(frames)-1]:
    s,e=frames[fi]; open(f'{out}/f{fi:02d}.jpg','wb').write(data[s:e])
print(avi, '->', len(frames), 'frames; wrote first/mid/last to', out)
# PY
```
Open the resulting `.jpg`s to judge the subject. (Play the `.avi` in VLC for the muxed audio.)

---

## 9. What to report back

1. Active backend line(s) (confirm `ggml_cuda` / 5090 name) + driver/CUDA version.
2. Run A: `[LATENT-DIAG] step=1` (must match), the `[ATTN-DIAG]` line, CUDA's `[COND-DIAG]` l2, and
   `[DECODE-DIAG]` video/audio.
3. Run B: subject (cat/human/hybrid) for **faON** and **faOFF**, with one representative frame each.
4. (If run) §7 isolation results.

### Reference scorecard (Metal vs Vulkan/RADV, 768×512 / SEED=42 / STEPS=8 / CFG=1)
| lever | Vulkan subject | note |
|------|----------------|------|
| baseline (flash ON) | human | Metal = cat here |
| VAE_CONV_DIRECT=false | n/a | decode-only l2=644.515 vs Metal 644.982 (0.07%) — VAE not it |
| load Metal cond | human | cond byte-identical to Metal — Gemma not it |
| CLIP_ON_CPU=true | human | cond l2=1429.06 ≈ Metal 1429.00 (fixes cond, not subject) |
| GGML_VK_DISABLE_COOPMAT=1 | human + cat figure | denoiser numerics do matter |
| GGML_VK_DISABLE_F16=1 | NaN/blank | full-F32 broken on RADV |
| **DIFFUSION_FA=false** | **cat** | flash-attn kernel is the tipping factor |
