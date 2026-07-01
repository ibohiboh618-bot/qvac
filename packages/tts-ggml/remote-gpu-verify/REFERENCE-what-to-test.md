# REFERENCE — what to test, what to skip (Mali-Vulkan TTS GPU bring-up)

Condensed findings so the operating agent can focus its probing. (Full campaign ledger lives with the
colleague; this is the actionable summary.)

## The bug, in one paragraph
On **ARM Mali-G715 / Valhall, Vulkan backend**, a **plain (non-broadcast) F32 `MUL_MAT`** miscomputes for
some shapes: a few output elements come out **~2-4× too large** while the overall rms stays normal — so
it's not a global scale error, it's specific output positions. It is **deterministic** in magnitude;
downstream it compounds to **NaN / silent / garbled** audio. The trigger is a **small output dimension**
(M = src0->ne1, the seq-len / row count) **< ~48**; padding that dim up to 64, computing, and slicing the
real block back is exact and works around it. Other GPUs (Adreno, Xclipse) compute it correctly.

## How the two engines relate to it
- **Supertonic** already ships the model-side pad (`st_mul_mat`), always-on for Mali. So `supertonic-gpu`
  is expected **healthy** — this round just confirms it (and gives an audio reference).
- **Chatterbox (T3 + S3Gen)** does **NOT** ship the pad. This kit can toggle a generic Mali pad via
  `TTS_CPP_MALI_PAD` (`run-on-device.sh` runs `chatterbox-gpu-raw` = pad **off** and
  `chatterbox-gpu-padded` = pad **on**). The whole point: does **raw** break and **padded** fix it?

## The oracle is the whole-model run + your ears
- The decisive signal is the per-stage **`[gpu-diag]`** trace in `<label>-gpudiag.log` (find the first
  stage that goes huge/NaN on the raw GPU run) **plus listening to the `.wav`** (raw vs padded vs CPU).
- Chatterbox's text model (T3) is **stochastic**, so a GPU and a CPU Chatterbox run won't be
  bit-identical even when both are correct — **judge Chatterbox by ear + by the per-stage trace**, not by
  an exact GPU-vs-CPU sample match. Supertonic is deterministic (GPU should closely match CPU).

## What to TEST per-op with `run-backend-ops.sh` (Vulkan vs CPU), if a run crashes early
- **`MUL_MAT`, F32**, sweeping a **small output dim M = {8,16,17,24,32,48,64}** and a few K,N. Look for
  M < ~48 cases that **diverge from CPU**. A **FAIL is a decisive localization**.

## What is CLEAN — do NOT chase (already refuted on-device for Supertonic)
- **Large output-dim MUL_MAT** (M ≥ ~48) — clean.
- **IM2COL** (pointwise = pure transpose), **depthwise conv**, **layernorm** — bit-exact.
- The **broadcast bias-add** — exonerated.
- **Env toggles** `GGML_VK_DISABLE_COOPMAT` (made it worse) / `GGML_VK_DISABLE_F16` (no effect).

## IMPORTANT caveat about `test-backend-ops`
It uses **random inputs + a tolerance**. A sibling bug **passed `test-backend-ops` yet miscomputed on real
data**. So a **FAIL** = solid localization; a **PASS does NOT clear an op** — the real-data oracle stays
the whole-model run (`run-on-device.sh`'s `[gpu-diag]` trace + the audio).

## The two tools in this folder
- **`run-on-device.sh`** — the real engines on the GPU; the `[gpu-diag]` per-stage trace + the `.wav`
  files are the decisive output. This is the oracle.
- **`run-backend-ops.sh`** — `test-backend-ops` per-op Vulkan-vs-CPU sweep; localizes which op shapes
  diverge, autonomously, without a new build from the colleague.
