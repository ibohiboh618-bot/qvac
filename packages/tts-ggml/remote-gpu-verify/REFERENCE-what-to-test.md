# REFERENCE — what to test, what to skip (Mali-Vulkan GPU miscompute)

Condensed findings so the operating agent can focus its per-op probing. (Full campaign ledger lives with
the colleague; this is the actionable summary.)

## The bug, in one paragraph
On **ARM Mali-G715 / Valhall, Vulkan backend**, a **plain (non-broadcast) F32 `MUL_MAT`** miscomputes for
some shapes: a few output elements come out **~2-4× too large** while the overall rms stays normal (so it's
not a global scale error — it's specific output positions). It is **deterministic** in magnitude. Downstream
this compounds to NaN / silent audio in the TTS model. Confirmed on-device for the duration predictor's
pointwise conv: operands `src0=[K=64, M=17]` × `src1=[K=64, N=256]`, F32 → Mali `max≈10.4` vs a correct
device `max≈2.5` for a **bit-identical input**.

## Mechanism suspect (where to look per-op)
- The miscompute is in ggml-vulkan's **non-coopmat F32 GEMM** (`mul_mm.comp`), specifically the **aligned,
  no-bounds-check vec4 load** path (`LOAD_VEC_A=4`) used when K is a multiple of the load width and M,N>8.
- `K=64` is the smallest case that spans **2 `BK=32` K-tiles** — so suspect **K > 32** (K spanning ≥2 tiles:
  64, 96, 128, 256, …). `K ≤ 32` (single tile) is **clean**.
- Android builds disable coopmat for ALL Vulkan devices, so Mali (and Xclipse) use this non-coopmat path;
  the bug is **Mali-driver-specific** (Xclipse + Adreno compute it correctly).

## What to TEST per-op (with test-backend-ops, Vulkan vs CPU)
- **`MUL_MAT`, F32**, sweeping the contraction dim **K = {16, 32, 48, 64, 96, 128, 256}** and a few M,N
  (small M like 16-32, N like 64-256). Look for K>32 cases that **diverge from CPU**.
- Note whether divergence tracks **K>32** (2-tile) and/or specific M/N alignments.

## What is CLEAN — do NOT chase (already refuted on-device)
- **`K ≤ 32` MUL_MAT** (single tile) — clean.
- **IM2COL** — for the pointwise case it's a pure transpose; bit-exact Mali==CPU. Not the culprit.
- **Depthwise conv, layernorm** — bit-exact.
- **The broadcast bias-add** — exonerated (post-bias ≈ pre-bias).
- **Env toggles**: `GGML_VK_DISABLE_COOPMAT` (made it WORSE), `GGML_VK_DISABLE_F16` (no effect). Don't bother.

## IMPORTANT caveat about test-backend-ops
It uses **random inputs + a tolerance**. A sibling bug on another driver **passed test-backend-ops yet
miscomputed on real data**. So:
- A **FAIL** on a MUL_MAT shape = a solid, decisive localization. Report it.
- A **PASS** does **not** fully clear an op — the real-data oracle is the whole-model run
  (`run-on-device.sh` + its `dprobe_pw1_mulmat` trisection). If test-backend-ops MUL_MAT all pass but the
  whole-model trisection still shows the outlier, that itself is the key finding (data-specific bug).

## Two complementary tools in this folder
- **`run-on-device.sh`** — the real model on the GPU; the `dprobe_pw1_mulmat` max is the decisive number
  (≈2.5 good / ≈10 bug). This is the oracle.
- **`run-backend-ops.sh`** — `test-backend-ops` per-op Vulkan-vs-CPU sweep; localizes which op shapes
  diverge, autonomously, without a new build from the colleague.
