# LTX-2.3 on NVIDIA via Vulkan — build + reproduce the Metal/RADV diagnostics

Runbook to build `diffusion-cpp` with the **Vulkan** backend on an NVIDIA box (e.g. RTX 5090, using
NVIDIA's Vulkan driver — **not** CUDA) on branch `feature/ltx-diag`, capture the same diagnostics we
gathered on Metal + AMD/RADV, and see which "basin" NVIDIA's Vulkan driver lands in.

This is a **third Vulkan datapoint**: identical Vulkan code path, different ICD/driver
(NVIDIA proprietary vs AMD RADV vs Apple Metal). The interesting question is whether NVIDIA's Vulkan
flash-attention numerics behave like RADV (human at 768) or like Metal (cat at 768).

Branch must include commit `5ab5f1dc` (or later) and these 5 vcpkg patches:
`0001-attn-diag` `[ATTN-DIAG]`, `0002-latent-diag` `[LATENT-DIAG]`, `0003-vae-decode-diag`
`[DECODE-DIAG]`, `0004-decode-only-latent-dump-load` `[DECODE-ONLY-DIAG]`,
`0005-cond-diag` `[COND-DIAG]` (+ `LTX_DUMP_COND`/`LTX_LOAD_COND`). Overlay port is
`stable-diffusion-cpp@2026-06-04#6`.

---

## 0. TL;DR — what we already know (the thing to confirm on NVIDIA/Vulkan)

- Same seed/params produce a **different but coherent** video per backend; it's GPU fp
  non-determinism, not a bug.
- At **512×320** the prompt subject is robust → **cat** on both Metal and RADV.
- At **768×512 / CFG=1 / 8 steps** the subject is **bistable** (cat vs human). Metal lands on
  **cat**; AMD/RADV Vulkan lands on **human**.
- Tipping factor isolated to the **DiT flash-attention kernel**: VAE ruled out (identical-latent
  decode matches to 0.07%), Gemma ruled out (loading Metal's exact cond on RADV still gave a human).
  On RADV, **`DIFFUSION_FA=false` (F32 attention) recovers the cat.**

**Questions for NVIDIA/Vulkan (5090):**
1. At 768×512 flash-attn ON, does it land on **cat** (Metal) or **human** (RADV)?
2. Does `DIFFUSION_FA=false` change the subject like it did on RADV?
3. Does `GGML_VK_DISABLE_COOPMAT=1` change it? (NVIDIA's coopmat impl differs from RADV's.)
4. Does `GGML_VK_DISABLE_F16=1` run cleanly here? (It produced **NaNs** on RADV.)

---

## 1. Prerequisites (NVIDIA / Linux, Vulkan backend)

- **NVIDIA driver with Vulkan support.** Verify the GPU shows up as a Vulkan device:
  ```bash
  vulkaninfo --summary | grep -iE 'deviceName|driverName|driverInfo'
  # expect: NVIDIA GeForce RTX 5090 ... driverName = NVIDIA
  ```
  No CUDA toolkit needed — this is a pure Vulkan build.
- **LLVM/Clang 22 with libc++**, **CMake 3.25+**:
  ```bash
  wget -q https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && sudo ./llvm.sh 22 all
  ```
- **Vulkan SDK** (provides glslc/shaderc to compile ggml's Vulkan shaders):
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

> Keep `VCPKG_ROOT` and `VULKAN_SDK` exported in the shell you build in.

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

## 3. Build (Vulkan is the Linux default)

```bash
export VCPKG_ROOT=/path/to/vcpkg          # must be set in THIS shell
npm install
npm run build                             # vulkan feature is default on linux
```

### Verify it actually rebuilt with the patches (guard against a stale binary-cache hit)
In the build output confirm ALL of:
```
-- Applying patch 0001-attn-diag-logging.patch
-- Applying patch 0002-latent-diag-logging.patch
-- Applying patch 0003-vae-decode-diag-logging.patch
-- Applying patch 0004-decode-only-latent-dump-load.patch
-- Applying patch 0005-cond-diag-logging.patch
Building stable-diffusion-cpp[core,vulkan]:x64-linux@2026-06-04#6...
```
It must say **Building ... #6**, not only "Restored …". If it's a stale cache,
`touch vcpkg/ports/stable-diffusion-cpp/portfile.cmake` and rebuild.

---

## 4. Download weights (~34 GB, skips any already present)

```bash
./scripts/download-model-ltx.sh
```

---

## 5. Run A — canonical diagnostic capture (512×320 / 9f)

Control run: per-step noise is CPU Philox so **step 1 must match byte-for-byte** across all backends.

```bash
SEED=42 FRAMES=9 WIDTH=512 HEIGHT=320 STEPS=8 CFG_SCALE=1.0 \
PROMPT="a claymation cat playing jazz on a piano" \
npm run generate:ltx 2>&1 | tee /tmp/ltx_nv_diag.log

grep -aE 'ATTN-DIAG|LATENT-DIAG|COND-DIAG|DECODE-DIAG' /tmp/ltx_nv_diag.log
grep -aiE 'ggml_vulkan|Vulkan|device|driver' /tmp/ltx_nv_diag.log | head
```

**Confirm the active Vulkan device is the NVIDIA GPU** (`ggml_vulkan: ... NVIDIA GeForce RTX 5090`).

### Reference values (compare NVIDIA/Vulkan against these)
```
[LATENT-DIAG] step=1  mean=-0.00654257 l2=206.174273 min=-4.395776 max=3.931298   <- MUST match
[ATTN-DIAG]   flash_attn ACTIVE (fused F16 flash_attn_ext) d_head=128 n_head=32 L_q=1024 L_k=1024
[COND-DIAG]   stage=cond numel=61440 ...   Metal l2=1429.004819 | RADV-GPU l2=1431.167567 | CPU 1429.060712
[DECODE-DIAG] stage=video ...              Metal canonical-latent decode l2=644.982275
```
- If `[LATENT-DIAG] step=1` differs → stop; seeding/noise isn't comparable.
- Record where NVIDIA's `[COND-DIAG]` l2 falls (vs Metal 1429.00 / RADV 1431.17 — i.e. is NVIDIA's
  GPU Gemma closer to Metal or to RADV?).
- Expect `[ATTN-DIAG]` to read `flash_attn ACTIVE`. If it says `FALLBACK to F32`, flag it.

---

## 6. Run B — the bistable subject test (768×512), THE key experiment

Baseline (flash-attn ON, default):
```bash
SEED=42 FRAMES=25 WIDTH=768 HEIGHT=512 STEPS=8 CFG_SCALE=1.0 NEG_PROMPT="" TEMPORAL_TILING=true \
PROMPT="a claymation cat playing jazz on a piano" \
npm run generate:ltx 2>&1 | tee /tmp/ltx_nv_768_faON.log
cp output/ltx_t2v_seed42.avi output/nv_768_faON.avi
```

Flash-attn OFF (F32 attention — the RADV cat-recovery lever):
```bash
DIFFUSION_FA=false SEED=42 FRAMES=25 WIDTH=768 HEIGHT=512 STEPS=8 CFG_SCALE=1.0 NEG_PROMPT="" TEMPORAL_TILING=true \
PROMPT="a claymation cat playing jazz on a piano" \
npm run generate:ltx 2>&1 | tee /tmp/ltx_nv_768_faOFF.log
cp output/ltx_t2v_seed42.avi output/nv_768_faOFF.avi
```

Then **look at the frames** (no ffmpeg needed — MJPG, see §8) and report the subject
(**cat / human / hybrid**) for each.

Interpretation:
- faON → **cat**: NVIDIA's Vulkan agrees with Metal; RADV is the outlier.
- faON → **human**, faOFF → **cat**: NVIDIA's Vulkan behaves like RADV; the flash-attn kernel is the
  tipping factor across Vulkan drivers.
- both human / both cat: subject just sits on a different side of the bifurcation for this driver.

---

## 7. Vulkan precision levers (these now apply — it's a Vulkan build)

Re-run the 768×512 baseline with each, then check the subject + `[LATENT-DIAG] step=8`:

```bash
# disable cooperative-matrix (NVIDIA's coopmat impl differs from RADV's)
GGML_VK_DISABLE_COOPMAT=1 SEED=42 FRAMES=25 WIDTH=768 HEIGHT=512 STEPS=8 CFG_SCALE=1.0 NEG_PROMPT="" TEMPORAL_TILING=true \
PROMPT="a claymation cat playing jazz on a piano" npm run generate:ltx 2>&1 | tee /tmp/ltx_nv_coopmat.log

# force full F32 (storage+compute). On RADV this produced NaNs — check if NVIDIA survives it.
GGML_VK_DISABLE_F16=1 SEED=42 FRAMES=25 WIDTH=768 HEIGHT=512 STEPS=8 CFG_SCALE=1.0 NEG_PROMPT="" TEMPORAL_TILING=true \
PROMPT="a claymation cat playing jazz on a piano" npm run generate:ltx 2>&1 | tee /tmp/ltx_nv_f32.log
grep -aE 'LATENT-DIAG] step=8' /tmp/ltx_nv_f32.log   # watch for nan
```
Other useful Vulkan env vars: `GGML_VK_DISABLE_COOPMAT2`, `GGML_VK_DISABLE_BFLOAT16`,
`GGML_VK_DISABLE_INTEGER_DOT_PRODUCT`.

---

## 8. (Optional) Cross-backend isolation tests

Need two binaries from the Strix diagnostic bundle (they live under `output/`, which is
**git-ignored**, so NOT in the repo — scp them onto the NVIDIA box):
`cond_metal.bin` (61440 floats) and `ltx_final_latent.bin` (40960 floats).

VAE-only (decode an identical Metal latent — isolates VAE kernels):
```bash
SEED=42 FRAMES=9 WIDTH=512 HEIGHT=320 STEPS=8 CFG_SCALE=1.0 \
LTX_LOAD_LATENT=/path/to/ltx_final_latent.bin \
PROMPT="a claymation cat playing jazz on a piano" npm run generate:ltx 2>&1 | tee /tmp/ltx_nv_decodeonly.log
grep -aE 'DECODE-ONLY-DIAG|DECODE-DIAG' /tmp/ltx_nv_decodeonly.log
# compare [DECODE-DIAG] stage=video to Metal l2=644.982275 (≈identical -> VAE kernels agree)
```

Conditioning-only (drive NVIDIA diffusion with Metal's exact cond — isolates Gemma):
```bash
SEED=42 FRAMES=25 WIDTH=768 HEIGHT=512 STEPS=8 CFG_SCALE=1.0 NEG_PROMPT="" TEMPORAL_TILING=true \
LTX_LOAD_COND=/path/to/cond_metal.bin \
PROMPT="a claymation cat playing jazz on a piano" npm run generate:ltx 2>&1 | tee /tmp/ltx_nv_loadcond.log
# look at frames: if still NOT a cat, cond/Gemma is not the tipping factor on this driver either
```

Gemma-on-CPU (does NVIDIA's GPU text-encode differ from CPU, like RADV's did?):
```bash
CLIP_ON_CPU=true SEED=42 FRAMES=9 WIDTH=512 HEIGHT=320 STEPS=8 CFG_SCALE=1.0 \
PROMPT="a claymation cat playing jazz on a piano" npm run generate:ltx 2>&1 | tee /tmp/ltx_nv_clipcpu.log
grep -a 'COND-DIAG' /tmp/ltx_nv_clipcpu.log   # CPU encode ≈ Metal 1429.00
```

---

## 9. Frame extraction helper (no ffmpeg required)

```python
# python3 frames.py ltx_t2v_seed42.avi    (run from packages/diffusion-cpp/output)
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
```
Open the resulting `.jpg`s to judge the subject. (Play the `.avi` in VLC for the muxed audio.)

---

## 10. What to report back

1. Active Vulkan device + driver line (`ggml_vulkan: ... NVIDIA GeForce RTX 5090`, driverInfo).
2. Run A: `[LATENT-DIAG] step=1` (must match), the `[ATTN-DIAG]` line, NVIDIA's `[COND-DIAG]` l2
   (closer to Metal 1429.00 or RADV 1431.17?), and `[DECODE-DIAG]` video/audio.
3. Run B: subject (cat/human/hybrid) for **faON** and **faOFF**, one representative frame each.
4. §7 levers: subject + step-8 latent for coopmat-off and F16-off (did F32 NaN like RADV?).
5. (If run) §8 isolation results.

### Reference scorecard — AMD RADV/Vulkan (768×512 / SEED=42 / STEPS=8 / CFG=1), Metal = cat
| lever | RADV subject | note |
|------|--------------|------|
| baseline (flash ON) | human | Metal = cat here |
| VAE_CONV_DIRECT=false | n/a | decode-only l2=644.515 vs Metal 644.982 (0.07%) — VAE not it |
| load Metal cond | human | cond byte-identical to Metal — Gemma not it |
| CLIP_ON_CPU=true | human | cond l2=1429.06 ≈ Metal 1429.00 (fixes cond, not subject) |
| GGML_VK_DISABLE_COOPMAT=1 | human + cat figure | denoiser numerics do matter |
| GGML_VK_DISABLE_F16=1 | NaN/blank | full-F32 broken on RADV (check if NVIDIA differs) |
| **DIFFUSION_FA=false** | **cat** | flash-attn kernel is the tipping factor |
