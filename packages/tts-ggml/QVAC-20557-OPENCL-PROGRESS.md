# QVAC-20557 — Supertonic Android GPU (OpenCL) — Progress & Hypothesis Log

**Goal:** all tts-cpp models (Supertonic, Chatterbox T3+S3Gen) produce correct audio on each
Android device's auto-selected GPU backend (OpenCL on Adreno, Vulkan on Mali/Xclipse), verified
on Adreno 740 (local) + S25 Ultra/Adreno 830 + Pixel 9/Mali (device farm) + Xclipse 920 (local),
with trustworthy green CI. Proper, generic, production-quality fixes only — no hacks.

**Device under active test:** vivo I2212 = Snapdragon 8 Gen 2 / **Adreno 740**, adb-attached.
Harness: `/data/local/tmp/ttsg` (bare CLI + libc++ + models + `verify-gpu.js`). Stage RMS via
`SUPERTONIC_STAGE_RMS=1` (native stderr, captured over adb).

---

## Verified device matrix (Adreno 740, rms vs CPU gold)

| Model | CPU | Adreno OpenCL | Adreno Vulkan |
|---|---|---|---|
| Chatterbox (T3+S3Gen) | ✓ 0.0457 | ✓ 0.0450 (slow ~143s) | ✓ 0.0517 (~24s) |
| Supertonic | ✓ 0.0374 (~1.5s) | ✓ **FIXED** 0.0374 (~14s) — was NaN→silent | ✓ 0.0372 (~4s) |

**Perf note (Adreno 740, Supertonic):** CPU 1.5s (RTF 0.48) < Vulkan ~4s < OpenCL ~14s (RTF 4.49).
The Adreno>700 auto-policy picks OpenCL, which is now CORRECT but ~9× slower than CPU and ~3.5×
slower than Vulkan for Supertonic. Correctness (the task) is met; the backend-selection/perf
trade-off (should Supertonic prefer Vulkan or CPU on Adreno?) is a separate product decision —
flag for discussion, do not silently special-case.

CPU gold per-stage: duration_raw=3.261, noise_in rms≈0.998, text_emb rms=0.359,
**cfm_latent rms=0.2585**, wav rms=0.0372.

---

## ROOT CAUSE (CONFIRMED 2026-06-15)

On Adreno **OpenCL**, ggml-opencl force-routes `GGML_OP_FLASH_ATTN_EXT` to **CPU**
(`qvac-ext-ggml@44fd4817 src/ggml-opencl/ggml-opencl.cpp:4770-4776` — "OpenCL flash-attention
kernels miscompute on real inputs"). Supertonic's vector-field text-attention therefore runs as a
**CPU↔OpenCL cross-backend split** under the ggml scheduler. When the per-step attention graph is
**reused across CFM steps** (step 1+), the **OpenCL-side copy of the bridged attention context is
stale** (not refreshed) → the attention **out-projection** (`dense_matmul_time_pretransposed_ggml`,
first GPU op after the CPU flash-attn) reads stale data → wrong output → error accumulates over the
4 CFM steps. Manifests as **NaN** in the fused one-graph path and **~3.4× too-large (non-deterministic)**
in the decomposed path.

Why the other paths are fine:
- **GPU-resident ops (convnext) reuse correctly** — only the cross-backend-bridged attention is stale.
- **Text-encoder** has the same bridge but runs **once** (no reuse) → correct.
- **Vulkan** supports `FLASH_ATTN_EXT` on-GPU (`ggml-vulkan.cpp:13704`) → **no bridge** → correct.
- **CPU** → all on CPU → correct.

First-divergence evidence (decomposed path, per-op CPU-vs-GPU trace, step 1):
`ve_attn0_ctx` (flash-attn output) MATCHES (maxdiff 0.000); the very next op `ve_attn0_out`
(out-projection) DIVERGES (gpu_rms 0.093 vs cpu_rms 0.366, maxdiff 2.73). Step 0 matches; steps
1→4 diverge increasingly (maxdiff 1.06→1.42→1.61→1.84).

---

## Hypothesis log

### H0 — "Latest ggml-speech/tts-cpp already fix Android GPU; just remove guards + overlay"
- **Test:** removed the two `loadLocked __ANDROID__` guards, pinned ggml-speech@44fd4817 + tts-cpp
  (master + defect-A reroute), built, ran full matrix on Adreno 740.
- **Result:** Chatterbox OpenCL/Vulkan ✓. Supertonic Vulkan ✓. **Supertonic OpenCL ✗.**
- **Verdict:** PARTIALLY WRONG. Chatterbox done; Supertonic-OpenCL is a real defect-B. → root-cause it.

### H1 — "Supertonic-OpenCL crashes are an F16 issue (F32×F16 mat-vec)"
- **Issue:** with f16 on, abort `GGML_ASSERT(src1t==F32)` in ggml-opencl mat-vec.
- **Test:** disabled `use_f16_attn`/`use_f16_weights` on OpenCL (kv_attn_type resolves to f32).
- **Result:** no more crash, but output **silent (rms=0)**.
- **Verdict:** PARTIALLY CORRECT (fixed the crash) but NOT the core bug (silent remained). f16 ruled
  out as the silent-output cause (kv_attn_type=f32 confirmed). → keep f16-off as a separate concern;
  hunt the silent output.

### H2 — "It's a `supports_op`-routed GPU op miscomputing; bisect by forcing ops to CPU"
- **Test:** added `GGML_OPENCL_FORCE_CPU_OPS` debug hook; forced matmul/conv/norm/elementwise/all to CPU.
- **Result:** still NaN even with ALL compute ops forced to CPU.
- **Verdict:** WRONG (and the experiment was INVALID). `supertonic_graph_compute` uses **direct**
  `ggml_backend_graph_compute(model.backend, …)` with **no scheduler** — a `supports_op→false` hook
  can't move ops off the single OpenCL backend. Lesson: the hook only works on sched paths (Chatterbox).

### H3 — "Loop-fusion (step-to-step GPU latent) is the bug"
- **Test:** `SUPERTONIC_DISABLE_LOOP_GRAPH=1`.
- **Result:** still NaN. **Verdict:** WRONG. The per-step path also NaNs.

### H4 — "Uninitialized GPU memory (parakeet-class), unifying NaN + 3.4×"
- **Test:** decomposed path (`DISABLE_LOOP_GRAPH=1`+`DISABLE_ONE_GRAPH=1`); ran 4× identical input.
- **Result:** non-silent but **3.4× too large AND non-deterministic** (cfm max 3.53→3.83→4.17).
- **Verdict:** CORRECT (root-cause CLASS = uninitialized/stale GPU buffer). Narrowed to the CFM stage
  (duration/text_emb correct; cfm_latent NaN/garbage; wav follows). → localize the exact buffer.

### H5 — "In-place RoPE (NEOX) hazard on Adreno OpenCL" (from background workflow)
- **Test:** read the NEOX kernel (`rope.cl:203` — each work-item owns disjoint pair, reads before
  writes → in-place-safe). Then restored the defensive `ggml_cont` on the RoPE input, rebuilt, ran.
- **Result:** still NaN. **Verdict:** WRONG. The proposed cross-work-item hazard doesn't hold, and
  forcing RoPE out-of-place changed nothing. Lesson: don't act on a hypothesis with an unverified
  mechanism — tested it, refuted it.

### H6 — "Localize via per-op CPU-vs-GPU trace diff in the decomposed path" (METHOD)
- **Test:** added `SUPERTONIC_TRACE_DIFF` — forces both `scalar_trace` (CPU) and `ggml_trace` (GPU)
  in `supertonic_vector_trace_proj_ggml`, prints per-named-tensor gpu_rms/cpu_rms/maxdiff in graph order.
- **Result:** step 0 ALL match (~1e-5). Step 1: first divergence at **`ve_attn0_out`** (out-projection),
  while `ve_attn0_ctx` (flash-attn output) matches. Divergence grows over steps.
- **Verdict:** CORRECT + decisive. Localized to the attention out-projection on **reuse** (step 1+).

### H7 — "Missing `ggml_backend_sched_reset` between reused computes"
- **Test:** read `supertonic_sched_alloc` (`supertonic_gguf.cpp:1769`).
- **Result:** it DOES call `ggml_backend_sched_reset` before each alloc.
- **Verdict:** WRONG. Not a missing reset. → the staleness is in the cross-backend copy/buffer itself.

### H8 — "Cross-backend (CPU↔OpenCL) bridge of the attention context goes stale on reuse" (ROOT CAUSE)
- **Evidence:** GPU-resident convnext ops reuse fine; only the op downstream of the CPU-routed
  flash-attn (the out-projection) is stale on reuse. flash-attn is CPU-routed only on OpenCL
  (`ggml-opencl.cpp:4770`); Vulkan runs it on-GPU (no bridge) and works; text-encoder has the bridge
  but runs once and is correct.
- **Verdict:** CONFIRMED root cause (see ROOT CAUSE section).

---

### H9 — "WHY does the FA→CPU gate exist? (must understand before routing around it)"
- **Concern (user):** removing/bypassing the FA gate could re-open the bug it was added for.
- **Investigated:** memory `project_bci_android_gpu_fix` (on-device A/B result) + ggml-opencl.cpp:4770
  comment. The gate exists because the **fused flash-attention KERNEL miscomputes on real Adreno inputs**
  (passes test-backend-ops, fails on the model). bci EMPIRICALLY REFUTED softmax + fast-math as the cause:
  "FA→CPU (softmax still on GPU) gives 6% WER → softmax-on-Adreno-OpenCL is fine."
- **Result:** the gate is **FA-kernel-specific**. `mul_mat` (verified this session — convnext matmuls
  match CPU ~1e-5) and `soft_max` (verified by bci, GPU softmax correct) are FINE on Adreno OpenCL.
- **Verdict:** CORRECT/decisive. Option A (explicit attention = mul_mat+soft_max+mul_mat) uses only
  verified-good ops, does NOT remove the gate, does NOT use the broken FA kernel → avoids both the FA
  bug and the cross-backend bridge. → proceed with A, verify on-device via trace-diff.

### H10 — "Will explicit attention's SOFT_MAX stay on GPU (no new bridge) on Adreno OpenCL?"
- **Concern:** if ggml-opencl routes SOFT_MAX to CPU, explicit attention re-creates a CPU bridge.
- **Test:** read ggml-opencl@44fd4817 `supports_op`: `case GGML_OP_SOFT_MAX: return true;` (line 4646-4648,
  unconditional). bci empirically confirmed GPU softmax correct on Adreno (6% WER).
- **Result/Verdict:** CORRECT — SOFT_MAX runs on the Adreno OpenCL GPU. Explicit attention
  (mul_mat + soft_max_ext + mul_mat) stays fully GPU-resident → no bridge. Path A cleared.
- **Remaining risk to verify on-device:** batched/3D `mul_mat` over n_heads for the QKᵀ and ·V
  steps on strided views — confirm via the trace-diff (ve_attn0_ctx must match CPU at step 0 AND not
  diverge at step 1+). If a strided-view mul_mat misbehaves, `ggml_cont` the q/k/v views first.

### H11 — "Implement Option A: in-situ supports_op gate + explicit GPU attention" (IMPLEMENTED — verify pending)
- **Refinement of the plan:** the original A plan used a load-time capability probe + a
  `model.backend_supports_flash_attn` field. Replaced with an **in-situ
  `ggml_backend_supports_op(model.backend, fa_node)` gate** inside a shared helper. Rationale
  (all verified, not assumed):
  - The OpenCL FA gate is **dtype-independent** (`ggml-opencl.cpp:4787` `case FLASH_ATTN_EXT: if
    (gpu_family==ADRENO || generic_subgroup_64) return false;`) → one in-situ check is exactly as
    correct as a probe, with no probe/live dtype-mismatch risk.
  - Mirrors the code's existing direct-vs-sched check (`run_text_attention_cache:1202`) and the
    plan's stated preference for the `supports_op` gate over backend-name special-casing.
  - Node budgets are generous (decomposed cache NODES=256; one-graph MAX_NODES=8192) → the
    throwaway FA node built only to query `supports_op` costs nothing.
  - Far less surface than the probe variant (no new probe fn / caps field / model field / load
    wiring).
- **Verified all explicit-path ops are GPU-supported AND correct-class on Adreno OpenCL**
  (`ggml-opencl.cpp` supports_op): CONT F32→F32 (4561), MUL_MAT F32×F32 (4716), SOFT_MAX true
  (4663), PERMUTE/TRANSPOSE/RESHAPE true (4738-4742). So explicit attention stays fully
  GPU-resident — no new CPU bridge. (`mul_mat` correctness on convnext verified ~1e-5 this
  session; soft_max correctness verified by bci 6% WER.)
- **Change (3 edits in `supertonic_vector_estimator.cpp`, tts-cpp worktree):**
  1. Added `supertonic_attention_ctx_ggml(ctx, model, q_in, k_in, v_in, q_len, n_heads, head_dim,
     scale)` helper (before `build_text_attention_cache`). Builds the FA node; if
     `!supports_op(model.backend, attn)` → explicit `cont(q/k/v)` → `mul_mat(K,Q)` →
     `soft_max_ext(·,scale)` → `cont(permute(V))` → `mul_mat(V_t,KQ)` → `cont(permute→[hd,nh,q]))`;
     else FA. Then shared `reshape_2d` + `cont(transpose)`. Output layout identical to FA.
  2. `build_text_attention_cache:1153` (decomposed path) → calls helper (scale 1/16).
  3. `append_text_attention_subgraph:4373` (one-graph / shipping path) → calls helper (scale param).
  - Also reverted the **refuted H5 rope-input `ggml_cont`** debug (`apply_supertonic_rope_ggml`).
- **Math:** explicit path is mathematically identical to FA (kqv layout matches FA's
  `[head_dim, n_heads, q_len]` so the shared downstream reshape/transpose/cont is unchanged).
- **Scope:** only the two CFM/vector-estimator FA sites. Text-encoder FA
  (`supertonic_text_encoder.cpp:818,928`) is a SEPARATE stage fed in as `text_in`, runs ONCE (no
  reuse), and was already correct on OpenCL (stage-RMS: text_emb=0.359 matches CPU) → left
  untouched (surgical). Will re-confirm via full-synth.
- **Result: CORRECT / VERIFIED on Adreno 740 (commit `479fb9bb`).** Both paths now correct on OpenCL:
  - **Default/shipping (one-graph):** `backendId=4 (OpenCL)`, EXIT_OK, cfm_latent rms=0.258538
    (gold 0.2585), wav_full rms=0.037232, RESULT rms=0.037403 — **identical to CPU's 0.037403**.
    Was NaN→silent (rms=0) before.
  - **Decomposed trace-diff:** `ve_attn0_ctx`/`ve_attn0_out` match CPU at **every** step s0–s4
    (maxdiff=0.00000); previously step1+ diverged (maxdiff 2.73→1.84). **Every** traced op across
    all steps is within ≤1e-4 of CPU → full-graph parity.
  - **H10 batched-mul_mat risk CLEARED:** the 3D `mul_mat` over n_heads is numerically exact on
    Adreno OpenCL (ve_attn0_ctx maxdiff=0.00000) — no per-head fallback needed.
  - **Audio A/B (user-confirmed):** OpenCL vs CPU WAVs (`qvac-20557-audio/sup_{opencl,cpu}.wav`,
    44.1 kHz) are clear and indistinguishable.
- **Verdict:** Option A is the correct, generic, verified fix on the local Adreno 740. Proceed to
  the multi-model/multi-backend matrix, then device farm + Xclipse, then gpu-smoke + PRs.
- **No-regression confirmed (current build `479fb9bb`, Adreno 740):**
  - Supertonic **CPU** (backendId=0): rms 0.037403, ~1.5s — unchanged ✓.
  - Supertonic **Vulkan** (backendId=3; OpenCL `.so` temporarily hidden to force it): rms 0.037256,
    cfm_latent 0.258657, ~4.1s — unchanged ✓. (FA branch of the helper emits the identical graph,
    so CPU/Vulkan/Metal/CUDA paths can't regress — confirmed empirically for CPU+Vulkan.)
  - Chatterbox code is untouched (change is isolated to `supertonic_vector_estimator.cpp`); prior
    matrix had Chatterbox ✓ on CPU/OpenCL/Vulkan. Re-confirm in the device-farm/CI phase.

## USER DECISIONS (2026-06-15, post-fix)
1. **Adreno backend = ship OpenCL as-is** for Supertonic (correct, meets the bar). NO per-model
   Vulkan-preference scope. (Perf on newer Adreno/S25 may differ; device farm will show.)
2. **Next = prepare the device-farm DO-NOT-MERGE validation PR** (S25/Adreno OpenCL + Pixel9/Mali
   Vulkan). Steps: strengthen gpu-smoke → cleanup debug scaffolding → clean tts-cpp commit → switch
   overlay to `vcpkg_from_github` → raise overlay PR. **Pushes to remotes require user confirmation.**

### Cleanup checklist for the clean PR (no fix depends on any ggml-speech change — pure tts-cpp)
- tts-cpp clean branch from `f88ad73c` (f16-off gate, on defect-A `63b21818`, on master `ed749556`):
  apply ONLY the explicit-attention helper + 2 call sites. DROP debug commits `7404bfba`
  (stage-rms), `6df847e9` (rope-cont, refuted), `6b29c6bb` (trace-diff). On the clean base the rope
  is already un-cont'd, so no revert needed.
- ggml-speech overlay → pin published `speech@44fd4817` via `vcpkg_from_github` (drop the local
  `GGML_OPENCL_FORCE_CPU_OPS` debug pin `1f0c0192`).
- f16-OpenCL-off gate (`f88ad73c`): KEEP (real fix for the F32×F16 mat-vec crash).

## DEVICE-FARM PR PREP (in progress, 2026-06-15)
- **Clean tts-cpp PR branch** `QVAC-20557-supertonic-opencl-attn` = master `ed749556` + `63b21818`
  (defect-A) + `f88ad73c` (f16-off) + **`58b8b61b`** (explicit attention, NO debug). Diff vs base =
  only `supertonic_vector_estimator.cpp` (+54/-8). **Re-verified on Adreno 740 OpenCL: backendId 4,
  rms 0.037403, 136972 samples** — identical to debug build → clean commit is good.
- **gpu-smoke.test.js strengthened** (lint clean): un-skip Android on both GPU tests; add
  `assertAudibleRms` floor (0.01) to all 4 tests (catches the silent/garbage regression the old
  sample-count assertion missed); device-correct backend already asserted (android→3|4); stale
  comments refreshed.
- **Overlays:** ggml-speech → reverted to `vcpkg_from_github` `44fd4817` (published; my fix needs no
  ggml-speech change). tts-cpp → currently local `vcpkg_from_git` `58b8b61b` for on-device verify;
  **must switch to `vcpkg_from_github` after pushing `58b8b61b`** (REPO + REF + SHA512-from-tarball).
- **Clean commit amended to `6090814538b246f3eb71b0a6cdc44bbf7d3fc229`** (trimmed the helper
  comment per user: minimal comments, no Asana tickets in code comments). Shipped C++/JS source is
  ticket-free; only the throwaway overlay portfile headers still cite tickets (build context;
  deleted before the real merge).
- **FINAL PR-config build verified on Adreno 740:** github ggml-speech `44fd4817` (CI-fetchable,
  restored from binary cache) + tts-cpp `6090814` → OpenCL backendId 4, rms 0.037403, 136972
  samples. The exact config the device-farm CI will build.
- **PENDING USER CONFIRMATION (outward-facing pushes):** push target for `6090814`
  (origin=tetherto vs personalupstream=fork) + branch name; then qvac DO-NOT-MERGE overlay PR.
- **Exclude from PRs (throwaway):** `verify-gpu.js`, `QVAC-20557-OPENCL-PROGRESS.md`,
  `qvac-20557-audio/`. ggml-speech debug worktree (`1f0c0192` force-cpu hook) is NOT shipped.

## PUSHED + PR OPENED (2026-06-15)
- **tts-cpp source branch pushed:** `QVAC-20557-supertonic-opencl-attn` → `origin`
  (tetherto/qvac-ext-lib-whisper.cpp), tip `6090814538b246f3eb71b0a6cdc44bbf7d3fc229`.
  SHA512 of the github tarball = `87885addf5da412e1b7ae9259b1f6ec665bd22dd81ddd55a6c6ec2723b9eb19a328fcf725a05ee83d3b4b6b352640bb1db3175c06df74acdbac5391785373209`.
- **qvac branch pushed:** `QVAC-20557-tts-ggml-android-gpu` → `origin` (tetherto/qvac), commit
  `0458b091` (guards removed + overlays + gpu-smoke).
- **DO-NOT-MERGE device-farm validation PR: #2605** → https://github.com/tetherto/qvac/pull/2605
- **Full CI-form overlay build verified locally** (both github pins fetch + build + link + on-device
  OpenCL rms 0.037403).
- **NEXT (user/CI):** apply the **`verified`** label on #2605 to trigger device farm (S25/Adreno
  OpenCL + Pixel9/Mali Vulkan). Then: local Xclipse 920; companion tts-cpp source PR to
  qvac-ext-lib-whisper.cpp (coordinate with PR #43 defect-A overlap — do NOT rebase onto it);
  registry publish + clean merge PR that drops ports/.

## ⚠️ COURSE-CORRECTION (2026-06-15, Xclipse 920 / SM-S711B testing)
**Finding (verified on-device):** tts-cpp `init_gpu_backend` (backend_selection.cpp:422-443) has an
**Android GPU allowlist = Qualcomm Adreno ONLY** (`is_qualcomm_adreno`). Non-Adreno Android GPUs
(Mali, Samsung Xclipse) are deliberately skipped → **CPU fallback** (comment: Mali aborts the host
process via GGML_ASSERT, so it's intentionally not validated). Pre-existing tts-cpp behavior, NOT
from my change.

Verified on Xclipse 920 (SM-S711B): loader logs `no Adreno GPU detected (-1); skipping OpenCL,
relying on Vulkan/CPU`; Vulkan backend loads but `init_gpu_backend` skips it (non-Adreno) →
**Supertonic runs on CPU (backendId 0), rms 0.037555 (correct)**. So the addon's auto-selected
backend on Xclipse = **CPU**, and it works. User's "auto-selected backend works" = MET (CPU).

**Implications:**
1. **Device matrix was wrong.** Android GPU is Adreno-only: S25/Adreno→OpenCL(GPU); Pixel9/Mali and
   Xclipse→CPU (by design). The plan's "Pixel9/Mali → Vulkan" premise is INCORRECT.
2. **gpu-smoke + PR #2605 risk:** my strengthened `assertGpuBackend` requires `backendDevice==1`, so
   it will FAIL on Pixel9/Mali (which legitimately runs CPU). Need to accept non-Adreno CPU-fallback.
3. **Parakeet's fix** = `stats.gpuUnsupported` (set by parakeet engine when a GPU is present but
   allowlist-skipped); its gpu-smoke accepts `android && dev==0 && gpuUnsupported`. **tts-ggml does
   NOT expose gpuUnsupported** (verified: not in supertonic_engine.cpp/backend_selection.cpp/engine.h;
   addon stats only have backendDevice+backendId). The addon alone CAN'T distinguish legit
   Mali/Xclipse CPU-fallback from an Adreno GPU-init regression (would mask the regression).
4. **Xclipse Vulkan/OpenCL can't be tested via the addon** (allowlist forces CPU). Forcing them needs
   a debug allowlist-bypass + rebuild, OR the tts-cli harness (tts-vk dir already has prior
   QVAC-19253 Supertonic-Vulkan wavs on Xclipse). Product-irrelevant (allowlist won't use them).

**OPEN DECISIONS (ask user):** (a) gpu-smoke for non-Adreno Android — add proper engine-level
`gpuUnsupported` (tts-cpp + addon, re-push/re-pin) vs a lighter test; (b) whether to force-test
Xclipse Vulkan/OpenCL with my fix (debug bypass + rebuild; informational only).

## XCLIPSE 920 FORCE-TEST RESULTS (2026-06-15, debug allowlist bypass)
Debug: added env `TTS_CPP_ALLOW_NON_ADRENO_GPU=1` to init_gpu_backend (debug branch
`QVAC-20557-xclipse-debug` `04038ebd`, NOT in the PR). Results on SM-S711B / Xclipse 920:
- **Auto-select (no bypass):** CPU (Adreno allowlist), rms 0.037555 ✓ correct.
- **Vulkan (bypass on):** backendId 3, backendDevice 1, **rms 0.037577 ✓ correct, NO abort, ~5s.**
  Xclipse Vulkan WORKS for Supertonic with the fix. (Mali-abort concern does NOT apply to Xclipse.)
- **OpenCL (bypass + GGML_OPENCL_FORCE_LOAD=1 + GGML_DISABLE_VULKAN=1 + GGML_OPENCL_ALLOW_UNKNOWN_GPU=1):**
  ggml-opencl accepts the device via the bci `generic_subgroup_64`→ADRENO path (ggml-opencl.cpp:3656-3670),
  loads kernels, then **SIGABRTs in the text encoder: `map::at: key not found`** (tts-cpp logic error on
  the Xclipse OpenCL path). NOT viable. Also product-irrelevant: the backend loader does NOT load OpenCL
  for non-Adreno, so Xclipse only ever has Vulkan in a normal run.

**CONCLUSION:** Worth adding **Samsung Xclipse → Vulkan** as a 2nd validated Android GPU vendor (works).
Keep OpenCL off for non-Adreno (not loaded; aborts when forced). Keep **Mali excluded** (aborts via
GGML_ASSERT per the existing comment). This also motivates the `gpu_unsupported` flag for the genuinely-
unsupported GPUs (Mali) so gpu-smoke accepts their CPU fallback (Pixel9 device-farm leg).

### Proposed proper change (pending user confirm on design)
1. tts-cpp `backend_selection.cpp`: replace the debug env with a real allowlist = Adreno OR Samsung
   Xclipse (identify by "Xclipse" in name/desc). Xclipse → Vulkan (other_gpu bucket). Mali etc. still skip.
2. tts-cpp: track "a GPU device was present but unused" in init_gpu_backend; expose `gpu_unsupported()`
   on Supertonic + Chatterbox engines (mirror parakeet).
3. addon SupertonicModel/ChatterboxModel: surface `gpuUnsupported` in stats.
4. gpu-smoke.test.js: accept `android && dev==0 && stats.gpuUnsupported` (Mali CPU fallback OK);
   Adreno/Xclipse must engage GPU.
5. New tts-cpp commit → re-pin overlay (github) → rebuild → re-verify Xclipse(Vulkan auto) + Adreno(OpenCL)
   → amend PR #2605.

## XCLIPSE SUPPORT IMPLEMENTED + VERIFIED (2026-06-15)
Added proper Samsung Xclipse → Vulkan support + a `gpu_unsupported` policy-decline flag.
**tts-cpp commit `aa2c9056`** (on the clean branch, on top of `6090814`):
- `backend_selection.cpp`: allowlist = Adreno OR Xclipse (`is_samsung_xclipse`; extracted `contains_ci`);
  `init_gpu_backend` out-param `out_gpu_present_but_unused` set ONLY at policy-skips (Mali allowlist /
  Adreno-6xx), NOT on init-failure (so an Adreno GPU-init regression still surfaces).
- Supertonic + Chatterbox engines expose `gpu_unsupported()` (threaded via init_supertonic_backend /
  init_backend → model field).
**addon (qvac):** SupertonicModel + ChatterboxModel surface `gpuUnsupported` stat.
**test:** gpu-smoke accepts `android && dev==0 && stats.gpuUnsupported` (Mali/Pixel9 CPU OK).

**Verified on Xclipse 920 (SM-S711B), auto-select, NO bypass env:**
- Supertonic → backendId 3 (Vulkan), backendDevice 1, gpuUnsupported 0, rms 0.037577 ✓
- Chatterbox → backendId 3 (Vulkan), backendDevice 1, gpuUnsupported 0, rms 0.045463 ✓ (~0.045 range)
- Both EXIT_OK, no abort. WAVs: qvac-20557-audio/xclipse_{supertonic,chatterbox}_vulkan.wav.
Adreno path unchanged by construction (still matches is_qualcomm_adreno); Mali → CPU+gpuUnsupported=1
(device farm Pixel9 will exercise). CPU/Vulkan emit identical graphs for the explicit-attn fix.

**DONE:** tts-cpp `aa2c9056` pushed to origin (branch QVAC-20557-supertonic-opencl-attn); overlay
re-pinned to github `aa2c9056` SHA512 `2fc32d81…`; qvac PR #2605 amended (commit `1d6fa830`) + body
updated. Full CI-form build verified. xclipse-debug branch `04038ebd` is local-only (not shipped).
**NEXT (user):** apply `verified` label on #2605 → device farm (S25/Adreno OpenCL + Pixel9/Mali CPU+
gpuUnsupported). Then companion tts-cpp source PR + registry publish + clean merge dropping ports/.

## MALI-VULKAN PROBE SET UP (2026-06-15) — awaiting device-farm run
No local Mali; probing on the device-farm Pixel 9 / Mali-G715. Throwaway, SEPARATE from #2605
(Mali→CPU and Mali→Vulkan-probe are mutually exclusive in one build).
- tts-cpp probe branch `QVAC-20557-mali-probe` @ tetherto, tip **`0aa594a6`** = aa2c9056 + one commit
  that adds `is_arm_mali` to the init_gpu_backend allowlist (so Pixel9 attempts Mali Vulkan via the
  existing gpu-smoke). SHA512 `9df14ca4…`.
- qvac probe branch `QVAC-20557-mali-probe`; **DO-NOT-MERGE PR #2610** (overlay pins 0aa594a6).
- **NEXT (user):** apply `verified` on #2610 → read Pixel9/Mali logs. Failure-mode table in the PR:
  ABORT (GGML_ASSERT/ggml_abort → note op; needs supports_op/decompose or ggml-vulkan fix) /
  GARBAGE (rms≪0.01, RMS floor fails → trace-diff root-cause) / WORKS (backendId 3, rms≈CPU → add
  is_arm_mali to the real allowlist). If not cleanly fixable → drop probe, Mali stays CPU (#2605).
- Probe build compiled clean; provably identical to aa2c9056 on Xclipse (is_arm_mali only adds Mali).
  Xclipse device disconnected mid-session so no local re-run (not needed).

### MALI-VULKAN PROBE RESULT (run 27567320883, job 81496752185, 2026-06-16) — GARBAGE/SILENT
Pixel 9 / Mali-G715, Vulkan (backendId=3, gpuUnsupported=0):
- **Chatterbox → WORKS:** rms **0.021163** ≈ Adreno-OpenCL 0.022146 for the same "GPU smoke check."
  utterance (S25 same run) → correct, not degraded.
- **Supertonic → BROKEN:** `[Supertonic] Synthesized 0 samples (duration: 0ms, RTF: 0.0000)` → rms
  0.000000, `not ok 1 produced expected sample count` + `not ok 2 produced audio`. **No exception
  thrown** (success path logged 0 samples) → silent miscompute / empty output, NOT an abort.
- S25/Adreno leg unaffected (Chatterbox 0.0221, Supertonic 0.0385, both OpenCL). Failure isolated to
  **Supertonic on Mali-Vulkan only** (Supertonic is correct on Adreno OpenCL+Vulkan, Xclipse Vulkan, CPU).
**Classification:** GARBAGE (silent), not ABORT. Mali-G715-driver-specific Vulkan miscompute (matches
the parakeet Mali-G715 precedent). "0 samples / 0ms duration" → degenerate early-stage output (duration
predictor / text encoder). **Not cleanly fixable in scope:** no local Mali → root-cause needs multi-round
device-farm per-stage trace-diff; driver-specific. **Per pre-agreed fallback → Mali stays CPU (#2605, green).**
Drop the probe (#2610 + tts-cpp `QVAC-20557-mali-probe` + the is_arm_mali commit). Optional future
enhancement (NOT now): per-model Mali allowlist (Chatterbox→Vulkan works, Supertonic→CPU) — adds
model-aware backend-selection complexity for half the models; needs more Chatterbox-Mali confidence (WER).

### ROOT-CAUSE INVESTIGATION — Supertonic-Mali-Vulkan (user chose to root-cause, 2026-06-16)
Enablers found (no rounds burned): (a) **native ggml/stderr logs do NOT reach the device-farm logcat**
(grep of the Pixel logcat for ggml-vulkan init = only appium/react noise) → diagnostics must go via
**JS console.log** (the `I bare :` lines) i.e. addon stats / sampleCount / rms. (b) **No committed
per-stage tracing** in tts-cpp (`SUPERTONIC_STAGE_RMS` was a throwaway local build, not in aa2c9056).
(c) **Rich env toggles exist** in `supertonic_vector_estimator.cpp`, read at synthesis time (NOT
backend-registration) so a test's `os.setEnv` before `run()` flips them with **no tts-cpp rebuild**:
`SUPERTONIC_DISABLE_LOOP_GRAPH` (5307; drops unrolled all-steps graph → per-step loop),
`SUPERTONIC_DISABLE_ONE_GRAPH` (5355; per-step → multi-cache/decomposed), `DISABLE_FUSED_{EDGE_PAD,
DEPTHWISE,LAYER_NORM,BIAS_GELU,PW}`, `DISABLE_CT_CONVNEXT`, `FORCE_EXPLICIT_REPEAT`,
`DISABLE_WEIGHT_PRETRANSPOSE`, `DISABLE_CPU_CUSTOM_OPS`.

Hypotheses (ranked):
- H-M1: fused/unrolled CFM graph aliases/miscomputes on Mali-Vulkan (analogue of the Adreno graph-reuse
  class). Test: `DISABLE_LOOP_GRAPH=1 DISABLE_ONE_GRAPH=1` → if audio returns, it's the fused CFM path.
- H-M2: a specific fused op (depthwise/layernorm/bias-gelu/pw/edge-pad/convnext) miscomputes on Mali.
  Test: bisect the `DISABLE_FUSED_*` toggles after H-M1 narrows to "fused-op, not structure".
- H-M3: failure is UPSTREAM of CFM (length/duration/text-encoder → latent_len or step count = 0 → 0
  samples). Indicated if H-M1 (CFM toggles) does NOT restore output → then need a tts-cpp probe build
  that surfaces latent_len/total_steps/stage-rms via JS stats (rebuild round).

Harness: env-only bisection on the `QVAC-20557-mali-probe` branch — the Supertonic gpu-smoke test sets
the toggle env(s) at module top via `bare-os.setEnv` before load; observe sampleCount/rms via existing
JS stats in the Pixel logcat. Trim to that one test with `fast-mobile-ci` (short rounds). Each round =
edit env set + push + (user re-applies `verified`) + ~10–15min + `fast-mobile-ci result`; NO rebuild.
Free local pre-check (Adreno 740, if re-attached): confirm the E1 toggle config still yields correct
Supertonic audio on a working GPU, so a null Mali result means "toggle didn't help", not "toggle broke it".
**E1 = `DISABLE_LOOP_GRAPH=1 DISABLE_ONE_GRAPH=1` on Mali.** Doubles as the first real `fast-mobile-ci` run.

**Pre-check (Adreno 740, local, 2026-06-16): PASS.** Default GPU(OpenCL) Supertonic = rms 0.037403;
with `DISABLE_LOOP_GRAPH=1 DISABLE_ONE_GRAPH=1` = rms **0.037403** (correct) → the per-step decomposed
toggle config does NOT break a working GPU. (Could not force Adreno→Vulkan locally: the 740 selector is
hardcoded to OpenCL; hiding the opencl .so just SIGBUS'd — artifact, not signal. ggml-vulkan path
correctness for Supertonic already established on Adreno/Xclipse Vulkan in the device matrix.)
Confirmed native ggml logs reach LOCAL adb stdout but NOT the device-farm logcat.

**E1 PUSHED (2026-06-16):** `QVAC-20557-mali-probe` commit **`6c8c6481`** = fast-ci trim to
`gpu-smoke.test.js` only (bundle regenerated → 1 runner `runGpuSmokeTest`, validate clean) + Android-gated
`os.setEnv('SUPERTONIC_DISABLE_LOOP_GRAPH','1')`/`...ONE_GRAPH` at the gpu-smoke module top. PR #2610.
**NEXT (user):** re-apply `verified` on #2610 → device farm; then read Pixel/Mali Supertonic rms via
`fast-mobile-ci result`. Recovers (~0.037) ⇒ fused CFM graph = culprit (narrow LOOP vs ONE, then fused
ops). Still 0 ⇒ op-level or upstream (H-M3) → need a tts-cpp probe build surfacing latent_len/stage-rms
via JS stats. First real `fast-mobile-ci` run (1 on-device test vs 9).

### E1 RESULT (run 27573655102, commit 6c8c6481, 2026-06-16) — INCONCLUSIVE (HANG)
- **fast-ci + os.setEnv both worked**: only `runGpuSmokeTest` ran; behavior changed vs default ⇒ toggles
  engaged on-device. (Validates the `fast-mobile-ci` pipeline + the env-probe mechanism.)
- **Adreno S25 + E1 toggles → Supertonic GPU passed** (20.7s) — toggles are sound (matches local pre-check).
- **Mali (landed on Pixel 9a) + E1 toggles → Supertonic synthesis HUNG** (model downloaded, then no result;
  runner stuck "running"; job timed out). The decomposed CFM path is pathological on Mali → NO clean
  audio-vs-0 signal. ⇒ H-M1 NOT confirmable via this toggle. Device variance: `MODEL CONTAINS "Pixel 9"`
  also matches **Pixel 9a** (slower Tensor-G4; Chatterbox alone took 472s) — not pinned to Pixel 9 Pro XL.

### ✅ ROOT CAUSE LOCALIZED (from existing data, no probe build, 2026-06-16) — DURATION PREDICTOR
`supertonic_engine.cpp`: `wav_len=(int)(duration_s*sample_rate)`; `latent_len=max(1,…)` (≥1, so CFM/vocoder
always run); **`result.pcm = wav_full[0 : min(wav_len, wav_full.size())]`** → empty iff `wav_len==0` iff
`duration_s≈0`. `SupertonicModel.cpp:209`: `audioDurationMs = duration_s>0 ? duration_s*1000 : samples*1000/sr`.
- Mali (#2610 Pixel 9 Pro XL): `Synthesized 0 samples (duration: 0ms …)` ⇒ `audioDurationMs≈0`. CPU/Adreno: `3105.96`.
- Discriminator: a vocoder/CFM break (duration normal, 0 samples) would log **3106ms**; Mali logs **0ms** ⇒
  `duration_s ≤ ~0` ⇒ **the Supertonic duration predictor (`supertonic_duration_forward_ggml`,
  supertonic_engine.cpp:380) miscomputes to ≈0 on Mali-Vulkan**, collapsing output length to zero.
  (CPU gold `duration_raw=3.261`.) OPEN: is duration the SOLE Mali break, or are downstream stages also
  broken? (E1 CFM-hang hints at more.) — decides the fix.

### INSTRUMENTATION PLAN (heavy debug logging, port PR #2601 pattern — APPROVED, in progress 2026-06-16)
Goal: one Mali device-farm run emits rich `[gpu-diag]` per-stage diagnostics into `logcat_full.txt` to
confirm the duration break + reveal downstream stage health. Full design in the plan file
`~/.claude/plans/mutable-kindling-book.md`. KEY mechanism (from PR #2601 parakeet): native stderr AND
QLOG/JsLogger (uv_async) are DROPPED on-device — only **`__android_log_print`** reaches `logcat_full.txt`.
Layers: (1) addon bridge in `SupertonicModel.{cpp,hpp}` — `emitDeviceDiag`(__android_log_print) +
`ggml_log_set` trampoline → emitDeviceDiag + canary + set `opts.diag_sink`; Android `liblog` link in
`CMakeLists.txt`. (2) tts-cpp `EngineOptions.diag_sink` (engine.h) + per-stage `[gpu-diag]` rms/nan/min/max
on the HOST vectors in `run_single_chunk` (duration_raw/wav_len/latent_len, text_emb, latent, final_latent,
wav_full). Run the DEFAULT path (E1 toggles removed — they hang). Pre-flight on Adreno 740 (validate
instrument + reference values), then push (re-pin overlay to the diag commit), device round, parse Pixel
`logcat_full`. Decision: duration-only-broken ⇒ try `duration→CPU on Mali`; downstream-also-broken ⇒ Mali→CPU.
STATUS: implementing Layer 2 first (engine.h diag_sink + supertonic_engine.cpp emissions); nothing edited yet.

### INSTRUMENTATION IMPLEMENTED + COMPILE-VALIDATED + ROUND LIVE (2026-06-16)
Both layers built exactly to plan; all five stage emissions wired:
- **tts-cpp** (`_wt-tts-cpp-gpu` @ **d7292fbb**, pushed to origin): `EngineOptions::diag_sink`
  (`std::function<void(const std::string&)>`, engine.h, null-default = zero behaviour change) +
  `stage_diag()` helper + emissions in `run_single_chunk`: `duration` (raw/s/wav_len/latent_len),
  `latent_in`, `text_emb`, `cfm_latent`, `wav_full`. Each line:
  `[gpu-diag] <stage> rms=.. nan=.. inf=.. min=.. max=.. n=..`. CPU/Adreno gold baked into comments
  (duration raw≈3.261, text_emb rms≈0.359, cfm_latent rms≈0.2585, wav_full rms≈0.0372).
- **addon** (`_wt-tts-ggml-gpu` @ **3468e2c0**, PR #2610 head): `emitDeviceDiag` →
  `__android_log_print(ANDROID_LOG_INFO, "qvac-supertonic", …)` (#else stderr); `ggmlLogTrampoline`
  (line-buffered) installed via `tts_cpp_log_set` (captures the backend-init banner + op-support
  warnings); load-time canary `[gpu-diag] canary: native log reaches host (supertonic)`;
  `opts.diag_sink = &emitDeviceDiag`. `CMakeLists.txt` links `liblog` PRIVATE on Android. Overlay
  portfile re-pinned to d7292fbb (SHA512 17936ad2…). E1 toggle block removed (default one-graph path;
  decomposed path hangs). Mobile bundle re-validated (E1 was top-level, not inlined → auto.cjs unchanged).
- **Pre-flight = LOCAL ANDROID COMPILE-ONLY** (user choice; Adreno device not attached): full
  `arm64-android` build via `bare-make generate/build/install` PASSED — tts-cpp + `SupertonicModel.cpp`
  compile AND link incl. the android-only `__android_log_print`/`liblog` path + `ggml.h` resolution via
  `tts-cpp/log.h`. Runtime [gpu-diag] reference comes free from the device-farm S25/Adreno leg.
- **DEVICE ROUND LIVE**: push triggered run **27593875419** ("On PR Trigger (TTS GGML)",
  `pull_request_target`, sha 3468e2c0) — `verified` label persisted (no re-apply needed). Expect
  gpu-smoke to FAIL on Mali (intended) while emitting the full `[gpu-diag]` stage table.
- NEXT: when the run finishes, pull `console-logs-qvac-tts-ggml-Android`, grep the **Pixel**
  `logcat_full.txt` for `qvac-supertonic` / `[gpu-diag]` — read `duration raw` (expect ≈0) + each
  downstream stage's rms/nan to decide `duration→CPU on Mali` (downstream normal) vs `Mali→CPU`
  (downstream also broken).

### ✅ DIAG ROUND RESULT (run 27593875419, PR #2610 @ 3468e2c0, 2026-06-16) — DEFINITIVE
Instrumentation worked perfectly (canary + 12 `[gpu-diag]` hits per device). Pixel 9a (Mali-G715)
`logcat_full.txt`, **Supertonic GPU run (engages Vulkan(3), gpuUnsupported=0 — NOT a CPU fallback)**:
```
[gpu-diag] duration   raw=nan s=nan wav_len=0 latent_len=1
[gpu-diag] latent_in  rms=0.946547 min=-2.6197 max=2.4632 n=144        <- host RNG, healthy
[gpu-diag] text_emb   rms=nan min=nan max=nan n=4096                    <- NaN
[gpu-diag] cfm_latent rms=nan min=nan max=nan n=144                     <- NaN
[gpu-diag] wav_full   rms=nan min=0 max=0 n=3072                        <- NaN -> rms=0.000000
```
Pixel **Supertonic CPU run** (same device): `duration raw=2.015 … wav_full rms=0.009694` — HEALTHY.
S25/Adreno (OpenCL): healthy reference (`text_emb rms≈0.39`, `wav_full rms≈0.038`).
Also: `[Chatterbox/GPU] backendId=3 (Vulkan)` works on the SAME Mali-Vulkan.

**Findings (this revises the earlier "duration predictor is the SOLE break" guess):**
1. The break is **NOT duration-only** — EVERY GPU compute stage is NaN, starting at the FIRST GPU op
   (duration predictor). `duration→CPU on Mali` would NOT fix it. So per the pre-agreed tree this is the
   "downstream also broken ⇒ Mali→CPU" branch — BUT see the localize decision below.
2. **Chatterbox works on Mali-Vulkan; Supertonic doesn't** ⇒ a **Supertonic-specific op** the Mali driver
   miscomputes. Supertonic uses **ConvNeXt depthwise convs**; Chatterbox uses none. ggml-vulkan.cpp ~16225
   already has a Mali guard for *"Valhall miscomputes depthwise im2col"* but only the F16-conv2d case —
   Supertonic's F32 conv1d depthwise isn't covered. → leading suspect.
3. **Instrument caveat:** `nan=0`/`inf=0` despite `min/max=nan` ⇒ `std::isnan`/`std::isinf` compile to
   no-ops under **fast-math**; the nan/inf COUNTS are unreliable (min/max/rms still expose NaN). Round-2
   switches `stage_diag` to a bit-pattern NaN check.

**USER DECISION (AskUserQuestion 2026-06-16): "Localize, then decide"** — name the exact culprit op in one
more cheap round, then decide fix-now vs defer. (#2605 already ships Mali→CPU correctly + green; native
Mali-Vulkan is the stretch goal. The adb/verify-gpu.js workflow fallback the user raised is NOT needed —
the in-test `[gpu-diag]` approach delivers.)

### ROUND-2 PLAN (op-level NaN localization) — approved, implementing 2026-06-16
Two methods ruled OUT by exploration: (a) a `SUPERTONIC_DISABLE_*` toggle matrix in one process — the
toggles are `static const = getenv()!=nullptr`, read ONCE per process (frozen after first synth); (b) a
`ggml_backend_sched_set_eval_callback` op tracer — the duration predictor runs on the DIRECT path
(`ggml_backend_graph_compute`, no sched) and allocates via `ggml_gallocr` (buffer reuse ⇒ post-compute
intermediates are clobbered; only `ggml_set_output` tensors survive).
APPROACH: mark the first ConvNeXt block's per-op intermediates (`duration_convnext_ggml`,
supertonic_duration.cpp ~187) as graph OUTPUTS (gallocr preserves them; no math/placement change), read +
NaN-check post-compute, emit via the existing `diag_sink` → `emitDeviceDiag`. Probes: block input,
depthwise-conv out, layer-norm out, pw1 out, gelu out, pw2 out, each block out. First NaN probe = culprit
op (expected: the depthwise conv). tts-cpp-only change (`supertonic_internal.h` diag_sink on model +
`supertonic_gguf.cpp` set it + `supertonic_duration.cpp` probes + `supertonic_engine.cpp` bit-pattern fix);
addon/test unchanged. Verify = local arm64-android compile gate → push tts-cpp → re-pin → push qvac → round
→ grep Pixel logcat. Plan file: `~/.claude/plans/mutable-kindling-book.md`.

### ROUND-2 IMPLEMENTED + COMPILE-VALIDATED + ROUND LIVE (2026-06-16)
- **tts-cpp** (`_wt-tts-cpp-gpu` @ **72a71099**, pushed to origin): `supertonic_internal.h` — shared inline
  `supertonic_diag_stats()` (BIT-PATTERN NaN/Inf — fixes the fast-math `std::isnan` no-op) + `diag_sink`
  field on `supertonic_model`. `supertonic_engine.cpp` — set `model.diag_sink = opts.diag_sink` after load;
  `stage_diag` now delegates to the shared helper. `supertonic_duration.cpp` — `duration_convnext_ggml`
  gains a `probe` flag that marks block-0's intra-op intermediates (`dprobe_dwconv/layernorm/pwconv1/gelu/
  pwconv2`) as ggml outputs; after `supertonic_graph_compute`, gated on `diag_sink && !backend_is_cpu`,
  reads those + the 6 `duration_convnext` block outputs + `duration_embed` and emits per-op `[gpu-diag]`.
- **Method rationale** (see ledger above): the duration predictor uses `ggml_gallocr` (reuse) on the DIRECT
  backend path → no eval-callback + post-compute intermediates clobbered, so we mark intermediates as
  OUTPUTS (gallocr preserves them). Caveat: marking can disable cross-op fusion at that point — if the NaN
  *moves/disappears*, that's itself a fusion clue.
- **Pre-flight = local arm64-android compile-only**: `bare-make generate/build` PASSED (tts-cpp 72a71099 +
  addon compile+link). addon/test UNCHANGED this round (probe auto-fires during the normal Supertonic GPU
  synth).
- **DEVICE ROUND LIVE**: tts-cpp pushed (origin @ 72a71099) → portfile re-pinned (SHA512 2818bd50…) → qvac
  pushed (PR #2610 @ **deb4fd2a**) → run **27596800814** queued (`verified` persisted). Expect gpu-smoke to
  FAIL on Mali (intended) while emitting the per-op probe lines.
- NEXT: pull `console-logs-qvac-tts-ggml-Android`, grep Pixel `logcat_full.txt` for `dprobe_` /
  `duration_convnext` → the first probe whose data is NaN (expected `dprobe_dwconv`) is the culprit op.

### ✅ ROUND-2 RESULT (run 27596800814, Pixel 9 / Mali-G715, 2026-06-16) — DEPTHWISE REFUTED, mul_mat IS THE BUG
Bit-pattern NaN counts now reliable. Adreno (S25) computed all 6 blocks clean (sound probe). Mali per-op:
```
dprobe_dwconv     Mali rms=0.114032 min=-0.5759 max=0.7057   |  Adreno rms=0.114032 min=-0.5759 max=0.7057   BIT-EXACT
dprobe_layernorm  Mali rms=0.744192 min=-4.2810 max=2.7766   |  Adreno rms=0.744193 min=-4.2810 max=2.7766   identical
dprobe_pwconv1    Mali rms=1.341813 min=-9.1171 max=10.3491  |  Adreno rms=1.364835 min=-6.0801 max=2.4826   DIVERGES <-
duration_convnext0 Mali 0.209  conv1 0.249  conv2..5 = NaN(1088)   |  Adreno conv0..5 0.201->0.232 all clean
duration raw=nan (Mali)   |   duration raw=1.734 (Adreno)
```
**Findings — the depthwise hypothesis is REFUTED:**
1. Depthwise conv + layernorm are **bit-exact** Mali vs Adreno. The **pointwise conv (`conv1d_f32` =
   `ggml_mul_mat`, `dprobe_pwconv1`) DIVERGES** — same input (depthwise/layernorm bit-exact) + same weights,
   but Mali's matmul has 4× outliers (max 10.35 vs 2.48). ⇒ **Mali-Vulkan `ggml_mul_mat` miscomputes.**
2. The matmul error is hidden in block 0 by the ConvNeXt layer-scale γ (output ≈ input), but compounds in
   the residual stream (Mali block outputs 0.209 → 0.249 → **NaN at block 2** vs Adreno stable 0.20→0.23),
   NaN-ing every downstream stage ⇒ `duration raw=nan` ⇒ 0 pcm. (CPU run healthy: `duration raw=1.727`.)
3. Analogous to the known **Xclipse `mul_mv`** miscompute (memory `bci_xclipse_gpu`) — but on **Vulkan**,
   where the backend has NO Mali matmul guard. Chatterbox works on the same Mali-Vulkan (different matmul
   shapes / no 6-deep residual compounding).
4. The probe-marking confound (block 0 intra-ops marked as outputs) did NOT move the result — block 1 (NOT
   intra-marked) is also clean, and the always-marked block outputs are the unperturbed signal.

**USER DECISION (AskUserQuestion 2026-06-16): "Cheap coopmat-disable round"** — test whether the
matrix-core (cooperative-matrix) matmul path is the Mali trigger before committing to a fix vs deferring.

### ROUND-3 PLAN (cheap coopmat-disable, pure-JS) — approved, implementing 2026-06-16
No tts-cpp/portfile change. Set `GGML_VK_DISABLE_COOPMAT=1` + `GGML_VK_DISABLE_COOPMAT2=1` at
gpu-smoke.test.js module-top (Android-only, read once at Vulkan device init); keep probe pin 72a71099 so
the per-op map still logs. Read Pixel `dprobe_pwconv1` (max ~2.48 ⇒ coopmat was wrong) + `duration_convnext*`
+ `duration raw`. FIXED ⇒ coopmat is the trigger ⇒ gate coopmat off for Mali (ggml-vulkan device guard) →
lean fix-now. NOT fixed ⇒ next cheap levers (F16, async, host-mem) or Xclipse-style Vulkan GEMM reroute /
defer. Plan file: `~/.claude/plans/mutable-kindling-book.md`.
- PUSHED + LIVE: qvac PR #2610 @ **fe3e6b08** (gpu-smoke.test.js module-top coopmat-disable; bundle
  byte-identical, validate green; no tts-cpp/portfile change). Device run **27598540133** in_progress.
  NEXT: pull Pixel logcat → `dprobe_pwconv1` max (~2.48 Adreno-like ⇒ coopmat was the bug) + `duration_*`.

### ❌ ROUND-3 RESULT (run 27598540133, Pixel 9 Pro XL / Mali-G715, 2026-06-16) — COOPMAT REFUTED
With `GGML_VK_DISABLE_COOPMAT=1` + `GGML_VK_DISABLE_COOPMAT2=1` (confirmed in logcat: `[mali-probe] …`):
```
dprobe_dwconv     Mali rms=0.114032 min=-0.5759 max=0.7057   == Adreno   (depthwise STILL bit-exact)
dprobe_layernorm  Mali rms=0.744192 min=-4.2810 max=2.7766   == Adreno
dprobe_pwconv1    Mali rms=1.539037 min=-12.6584 max=13.1657  (coopmat-ON was 10.35; Adreno 2.48)  WORSE
duration_convnext0 clean; convnext1..5 = NaN(1088)           (NaN now at block 1, was block 2)
duration raw=nan  ⇒ text_emb/cfm_latent/wav_full all NaN
```
**Disabling coopmat did NOT fix the matmul — it made it WORSE** (more outliers, NaN one block earlier). ⇒
the cooperative-matrix path is NOT the trigger; the **base (scalar) Mali-Vulkan `ggml_mul_mat` is also
broken**. Coopmat hypothesis REFUTED.

**Adreno (S25) "failure" this round = INFRASTRUCTURE FLAKE, not the change.** Logcat shows
`Chatterbox GGUFs not found and registry fetch failed` + `Supertonic GGUF not available - registry fetch
failed` (model-download/network flake on the device-farm runner). `GGML_VK_*` is Vulkan-only; Adreno runs
Supertonic on **OpenCL**, which ignores it. When the model WAS present, S25 Supertonic computed healthy
(`duration raw=1.727`, `wav_full rms=0.037`). So the coopmat env change could not and did not affect Adreno.

### ✅ ROOT CAUSE — FINAL (Supertonic Mali-Vulkan)
`conv1d_f32` (supertonic_duration.cpp:104) = `ggml_im2col` + **`ggml_mul_mat`**. The pointwise conv
(`pwconv1`) is a **dense K=64 reduction** `[L,64]·[64,256]`. The **depthwise** conv uses a *grouped/small*
mul_mat and is **bit-exact** on Mali; the **dense pointwise mul_mat miscomputes a FEW output elements**
(rms ~normal, but max/min 4–5× too large) on **both coopmat AND scalar** Vulkan paths. The error is hidden
in block 0 by the ConvNeXt layer-scale γ, then compounds in the residual stream to NaN by block 1–2,
NaN-ing every downstream stage ⇒ `duration raw=nan` ⇒ 0 pcm. This is the **Xclipse `mul_mv` K-reduction
miscompute class** (memory [[project_bci_xclipse_gpu]]) — but on **Mali-Vulkan**, where ggml-vulkan has no
guard (the Xclipse K-tail-GEMM fix is OpenCL-only). Adreno-OpenCL + CPU correct; Chatterbox survives Mali
(no dense pointwise-conv stacked 6-deep with γ-compounding).

### 🚫 DEAD ENDS — DO NOT REPEAT (refuted with on-device evidence)
- **Depthwise / im2col is the culprit** — NO. `dprobe_dwconv` is bit-exact Mali==Adreno across runs 2 & 3.
- **Duration predictor is the SOLE break / `duration→CPU` would fix it** — NO. Every GPU stage is NaN; the
  cause is upstream (the matmul), so duration-only routing is insufficient.
- **A single op TYPE Supertonic uses that Chatterbox doesn't** — NO. Both use mul_mat; it's the dense
  K-reduction SHAPE + γ-compounding, not an exotic op.
- **coopmat (matrix cores) is the trigger** — NO. `GGML_VK_DISABLE_COOPMAT(+2)` made it WORSE (run 3).
- **`SUPERTONIC_DISABLE_*` toggle MATRIX in one device round** — IMPOSSIBLE. They're
  `static const = getenv()!=nullptr` (frozen once per process); one config per round only.
- **`ggml_backend_sched_set_eval_callback` op tracer for the duration predictor** — WON'T FIRE. It runs on
  the DIRECT `ggml_backend_graph_compute` path (no sched); intermediates are gallocr-reused → must mark
  `ggml_set_output` to read them.
- **`std::isnan`/`std::isinf` for NaN detection in diag** — UNRELIABLE (no-op under fast-math). Use the
  bit-pattern check in `detail::supertonic_diag_stats`.
- **Treating an Adreno/S25 red as a code regression without reading the logcat** — the device farm flakes
  on model **registry fetch / GGUF download** (seen run 3); always confirm it's not `registry fetch failed`
  before attributing a failure to a code change. OpenCL (Adreno) ignores all `GGML_VK_*`.
- **Forcing Adreno onto Vulkan by hiding `opencl.so`** — SIGBUS (selector hardcoded to keep OpenCL); see
  earlier note.

### ⏭️ REMAINING LEVERS (if pursuing native Mali-Vulkan) vs DEFER
Cheap (pure-JS, one config/round, low-probability given it's a shape/K-reduction shader bug, not precision):
`GGML_VK_DISABLE_F16` (fp32-only), then `GGML_VK_DISABLE_ASYNC` / `GGML_VK_PREFER_HOST_MEMORY`. Real fix:
an **Xclipse-style Mali-Vulkan `mul_mat` reroute/guard in ggml-vulkan** (ggml-speech change; significant,
possibly upstream-ggml territory). Pragmatic: **DEFER → Mali→CPU** (already shipped & green in #2605:
Adreno OpenCL + Xclipse Vulkan + Mali CPU) + a follow-up ticket carrying this root cause. ← decision pending
with user.

### 🔗 PARAKEET (QVAC-20556) CROSS-CHECK (`/tmp/qvac-parakeet-mali-progress.md`, last hyp R3g run 27598096451)
Parakeet independently localized its Mali-Vulkan break to a **`ggml_mul_mat` miscompute** — the **broadcast**
mul_mat inside its subsampler's DEPTHWISE conv2d decomposition (F32 im2col finite/clean → broadcast mul_mat
= deterministic **inf**); regular **non-broadcast** mul_mat (`sub_conv0`) is CLEAN.
- **MATCH (class):** both = Mali-G715/Valhall Vulkan `ggml_mul_mat` miscompute; im2col exonerated;
  Adreno-OpenCL+Metal+CPU clean; both cite the Xclipse `mul_mv` precedent.
- **DIFFER (specifics):** parakeet = **broadcast** mul_mat → `inf`; ours = **plain 2D** mul_mat (K=64) →
  **finite ~4-5× outliers** compounding to NaN. TENSION: parakeet found plain mul_mat CLEAN, OUR plain
  pointwise mul_mat is subtly broken ⇒ Valhall mul_mat fragility is **broader / shape-dependent** than
  parakeet's "broadcast-only". (Our depthwise is clean only because it uses a fused CUSTOM op, not the
  im2col+broadcast-mul_mat path — so no contradiction.)
- **🚫 CRITICAL SHARED TRAP — `supports_op` Mali-gates are DEAD on the direct `ggml_backend_graph_compute`
  path (parakeet FACT 2), and Supertonic's duration predictor uses exactly that path (no sched).** Parakeet
  wasted 3 device rounds (3c/3d/3f) on dead supports_op gates. ANY Mali fix for our duration path MUST be a
  LIVE graph change (route the pointwise mul_mat→CPU / reformulate / add a `ggml_backend_sched`), NOT
  supports_op. Parakeet has **no working fix yet** (R3g localized the op; fix still iterating) → nothing to
  copy, but a proper ggml-vulkan Valhall mul_mat fix would serve parakeet + Supertonic + bci(Xclipse).

### ROUND-4 (GGML_VK_DISABLE_F16 — last cheap lever) — LIVE 2026-06-16
USER DECISION: try F16 first (cheap); if it fails, commit to the ggml-vulkan fix. Adreno now on adb (local
gate for the FIX phase — it's OpenCL so it IGNORES GGML_VK_* and can't test F16). Pure-JS: gpu-smoke.test.js
module-top `GGML_VK_DISABLE_F16=1` (swapped from coopmat); probe pin 72a71099 kept. qvac PR #2610 @
**9111c102**; device run **27601409020** queued. EXPECT (low-prob, finite-outlier ≠ precision): Pixel
`dprobe_pwconv1` max ~2.48 + no NaN ⇒ F16 was it (small Mali fp16-off fix); still ~10-13/NaN ⇒ F16 REFUTED ⇒
commit to the live-graph ggml-vulkan mul_mat fix (Mali-only, Adreno-regression-gated locally, coord w/ parakeet).

### ❌ ROUND-4 RESULT (run 27601718214, Pixel 9 Pro XL / Mali-G715, 2026-06-16) — F16 REFUTED → ALL CHEAP LEVERS EXHAUSTED
With `GGML_VK_DISABLE_F16=1` (confirmed: `[mali-probe] GGML_VK_DISABLE_F16=1` in logcat):
```
dprobe_dwconv     rms=0.114032 min=-0.5759 max=0.7057   (bit-exact, still fine)
dprobe_pwconv1    rms=1.392400 min=-9.1171 max=10.3491  <- BIT-IDENTICAL to round-2 default (coopmat-on)
duration_convnext0/1 clean; convnext2..5 = NaN(1088); duration raw=nan ⇒ text_emb/cfm/wav_full all NaN
```
**Disabling fp16 changed NOTHING — `dprobe_pwconv1` is byte-for-byte the round-2 default (max 10.3491,
min -9.1171).** So fp16 is not even on this matmul path; **F16 REFUTED.** Summary of the cheap-lever sweep on
the SAME failing op (`dprobe_pwconv1`): default/coopmat-on max **10.35**; coopmat-off max **13.17 (worse)**;
f16-off max **10.35 (identical to default)**. ⇒ the miscompute is in the **base Valhall `mul_mat`
computation itself** (integer/indexing/reduction logic), independent of fp16 and coopmat. **No cheap env
lever fixes it.**

**DECISION (per user): COMMIT to the ggml-vulkan Valhall `mul_mat` fix.** Implement + test properly; in the
new session, plan via a Workflow + decide the best approach. Constraints unchanged: LIVE graph change (NOT
supports_op — dead on the duration predictor's direct `ggml_backend_graph_compute` path), Mali-only,
Adreno-regression-gated on the attached adb device, coordinate with parakeet (shared Valhall mul_mat bug).
**CHECKPOINT/restore point committed:** qvac **`30dcb446`** (this ledger) + tts-cpp **`72a71099`** (probes).
Handoff: `/tmp/qvac-20557-ggml-vulkan-matmul-fix-handoff.md`. Plan: `~/.claude/plans/mutable-kindling-book.md`.

### ⚠️ ROUND-5 PREP (2026-06-16) — design workflow REOPENED the localization; the round-4 "commit to the mul_mat fix" is CORRECTED to "split-capture probe FIRST"
A multi-agent design workflow (5 source investigators → synthesis → adversarial review; verdict **SOUND**)
re-examined the "K=64 plain `mul_mat` is broken" root cause against source and the parakeet ledger. It does
**NOT** hold up as a confirmed isolation — three findings:
1. **`dprobe_pwconv1` folds THREE ops, never split.** It is marked AFTER `conv1d_f32` **+ a broadcasting
   bias-add** (`supertonic_duration.cpp:212-214`): im2col + the plain K=64 `ggml_mul_mat` + `ggml_add` of a
   `repeat_like` view `[1,256]` broadcast over `[L,256]`. The bare mul_mat output was **never read in
   isolation**. "the mul_mat is broken" is an INFERENCE — exactly the R3g/R11 situation that FLIPPED parakeet.
2. **im2col is doubly refuted** (so the round won't waste a slot on it): (a) source — `im2col.comp` for the
   pointwise case (KW=KH=1,p=0,s=1) is a 1:1 `[L,64]→[64,L]` transpose, bounds-branch always true, cannot 4×
   a value; (b) empirical — ggml-vulkan does NOT implement `GGML_OP_SUPERTONIC_DEPTHWISE_1D` (grep=0), so the
   depthwise takes the **im2col+mul_mat fallback** and `dprobe_dwconv` is bit-exact Mali==Adreno ⇒ Vulkan
   im2col runs clean on this graph/driver.
3. **A new, parakeet-grade suspect: the broadcasting bias-add.** Parakeet's SETTLED fact (`/tmp/qvac-parakeet-
   mali-progress.md` R3g): the **BROADCAST op-class** is what breaks on this exact Valhall driver, and **plain
   non-broadcast `mul_mat` stays CLEAN** (their `sub_conv0`, K≈9; reinforced by our `dwconv` mul_mat K=5 clean).
   Our pwconv1 mul_mat is plain/non-broadcast → by parakeet it is the *less*-suspected op; the broadcast bias-add
   is in the *confirmed-broken* class. So "plain mul_mat clean" CANNOT dismiss the bias-add. Reconciliation of
   the contradiction toward **K-dependence** (K=9/5 single-tile clean vs K=64 two-`BK=32`-tile) remains OPEN —
   the K=64 mul_mat was never read alone.
**CORRECTION carried (R15):** parakeet's `ggml_mul`+`ggml_sum_rows` reformulation (commit `4ab3519d`) was
DO-NOT-MERGE/**Metal-preflighted-only, NEVER Mali-confirmed**; their SHIPPED fix was Mali→CPU (`bb585eb1`). So
it is a CANDIDATE pattern, not a "proven" transfer.

**ROUND-5 HYPOTHESIS / TEST:** trisect block-0 pwconv1 into three named outputs — `dprobe_pw1_im2col` (im2col),
`dprobe_pw1_mulmat` (raw `ggml_mul_mat`, pre-bias), `dprobe_pwconv1` (post broadcast bias-add) — + emit `ne[]`
for every probe (confirm real K & L). Pure tts-cpp probe (no ggml change). **Pre-registered interpretation:**
(a) im2col carries the outlier → deeper im2col bug (contradicts source; re-open); (b) im2col clean + bare mul_mat
carries it → K-dependent non-broadcast `mul_mat` bug CONFIRMED (a NEW failure mode parakeet did not see) → fix
the K=64 mul_mat; (c) im2col+mul_mat clean + only post-bias bad → the **broadcasting bias-add** is the culprit
(parakeet-class broadcast bug) → model-side explicit-`repeat`/add-on-CPU. **All fixes via Approach B** (LIVE
Mali-gated model-side change — NOT a bare `supports_op` gate [dead on the direct `graph_compute` path], NOT the
universal-GEMM shader edit of Approach A). Probe HANGS / outlier MOVES when marks added ⇒ DID-NOT-RUN /
fusion-sensitivity (R8/R13), not a clean negative.

**State:** tts-cpp probe **`8bba2619`** (split-capture) pushed to `tetherto/qvac-ext-lib-whisper.cpp`
@ `QVAC-20557-mali-probe`; addon portfile re-pinned (SHA512 `d179a824…`). Workflow run `wxwefu5hf`.

**LOCAL ADRENO PRE-GATE — GREEN (2026-06-16, device `10BD1C1LEF0001R`, OpenCL):** trisection emits cleanly,
no crash, **K=64 CONFIRMED** (`dprobe_pw1_im2col ne=[64,54]`; pwconv1 expands 64→256, `dprobe_pw1_mulmat
ne=[54,256]`), Adreno rms=0.037403 (no regression). **`dprobe_pw1_im2col` rms/min/max == `dprobe_layernorm`
EXACTLY** (0.720668 / -3.2892 / 3.2276) ⇒ the pointwise im2col is a pure transpose (cannot create the outlier).
Adreno GOLD for the Mali comparison: `pw1_mulmat` rms=1.403 min=-5.878 max=2.334; `pwconv1` rms=1.507
min=-6.305 max=2.306 (Mali round-4 `pwconv1` max was **10.35** — the Mali round shows if that outlier is in
`pw1_mulmat` [the bare K=64 mul_mat] or only post-bias).

**ROUND-5 LIVE:** qvac #2610 @ **`66518cff`** pushed; device-farm run **`27608930766`** queued (default
Mali-Vulkan path, F16 env dropped). Read Pixel/Mali `dprobe_pw1_im2col` / `dprobe_pw1_mulmat` / `dprobe_pwconv1`
per the pre-registered three-way interpretation above.

### ✅ ROUND-5 RESULT (run 27608930766, Pixel 9 / Mali-G715 + S25/Adreno control, 2026-06-16) — INTERPRETATION (b) CONFIRMED: the bare K=64 mul_mat is the broken op
Apples-to-apples, SAME text (L=17), trisection on BOTH devices:
```
                  Adreno S25 (OpenCL, correct)        Mali Pixel 9 (Vulkan, broken)
dprobe_layernorm  rms 0.744 min -4.281 max 2.777      rms 0.744 min -4.281 max 2.777   (BIT-IDENTICAL)
dprobe_pw1_im2col rms 0.744 min -4.281 max 2.777      rms 0.744 min -4.281 max 2.777   (== layernorm; pure transpose)
dprobe_pw1_mulmat rms 1.282 min -5.682 max  2.547     rms 1.255 min -9.104 max 10.408  <- OUTLIER FIRST APPEARS HERE
dprobe_pwconv1    (post-bias) max 2.483               (post-bias) max 10.349           (≈ pre-bias; bias-add adds nothing)
```
**VERDICT:** the **bare non-broadcast K=64 `ggml_mul_mat` miscomputes on Mali** — the input (`pw1_im2col`) is
BIT-IDENTICAL to Adreno yet the output diverges (a few elements ~2-4× too large; rms ~normal). **im2col is
EXONERATED** (== layernorm, identical both devices). **The broadcast bias-add is EXONERATED** (`pwconv1` ≈
`pw1_mulmat` on both). K-dependence holds (K=5/9 clean, K=64 broken). The outlier is **DETERMINISTIC** (pwconv1
max=10.3491 identical in round-4 f16-off AND round-5 default); only the downstream NaN cascade is
non-deterministic (Mali run-1: duration 1.40/latent 20 wrong + `text_emb` NaN → silent; Mali run-2: clean,
wav rms 0.037). The same broken mul_mat hits the text encoder (K=64/128/256), CFM attention (K=64, ~20×/synth),
vocoder (K≈256) → PERVASIVE → model-side reroute not viable. **PLAN APPROVED (`~/.claude/plans/mutable-kindling-book.md`):**
fix the ggml-vulkan **non-coopmat F32 `mul_mm` reduction** (`mul_mm.comp` 305-349; auto Mali-scoped — coopmat
devices use the other branch). H1 = Mali-deviceName-gated force-unaligned-pipeline (cheapest, ships-if-works).

### ROUND-6 / H1 — Mali-gated force-unaligned F32 mul_mm pipeline (ggml-vulkan) — LIVE
**Correction (verified in `ports/ggml-speech/portfile.cmake`):** the addon builds Android Vulkan with
`GGML_VULKAN_DISABLE_COOPMAT=ON` → **Xclipse ALSO uses the non-coopmat `mul_mm` path** (and is correct there),
so a fix in that path is NOT auto-Mali-scoped; it MUST be `deviceName("Mali")`-gated. The `[m=17,n=256,k=64]`
case selects the **aligned small** pipeline `a_s` (`aligned`=true at ggml-vulkan.cpp:7924 since K%align==0, m>8,
n>8) → the aligned vec4 no-bounds-check load (`mul_mm_funcs.glsl` LOAD_VEC_A=4) is exercised.
**H1 (qvac-ext-ggml `47d7351d`, branch `QVAC-20557-mali-mulmat` off clean `44fd4817`):** in
`ggml_vk_guess_matmul_pipeline` (~7491), Mali+F32 → force `aligned=false` → the bounds-checked UNALIGNED
pipeline (`mmp->s`, LOAD_VEC_BATCH_A=2). Tests whether the aligned load is the culprit (same reduction either
way, so it isolates the LOAD path). Addon `ports/ggml-speech` re-pinned `47d7351d` (SHA512 `e07c26b8…`);
tts-cpp split-capture probe `8bba2619` + default Mali-Vulkan gpu-smoke kept.
**Compile gate GREEN** (ggml-speech arm64-android-vulkan built). **Local Adreno pre-gate GREEN** (the rebuilt
vulkan .so + .bare load+run; backendId=4 OpenCL, rms=0.037403 == baseline → no regression; gate inert on
non-Mali, as designed). **DECISIVE READ on the Mali round:** `dprobe_pw1_mulmat` max **10.4 → ~2.5** (== Adreno
control) ⇒ aligned load was the bug → ship the Mali gate; still ~10.4 ⇒ H2 (strengthen non-coopmat K-tile
barriers `mul_mm.comp` 287/349) → H3 (WARP=32 vs HW subgroup=16 geometry).

**H1 round 1 (run 27611952432) was CANCELLED (hung)** — no verdict. Re-triggering with a gate-fired diagnostic.
**Gate-fired log added (ggml `27cbd6f4`):** a one-shot `GGML_LOG_WARN("[gpu-diag] QVAC-20557 H1: Mali F32
mul_mat -> UNALIGNED (gate FIRED, dev=%s)")` in the H1 gate — anti-dead-gate guard (the addon's
`tts_cpp_log_set`→`ggml_log_set` shares the sink, so ggml logs reach logcat here, unlike parakeet). If that
line is ABSENT but the bug persists, the gate (not the unaligned path) is at fault. ggml-speech re-pinned
`27cbd6f4` (SHA512 `4eb608b1…`).

**TWO PARALLEL MALI-TEST PATHS now:**
1. **CI device-farm** (Pixel/Mali) — re-trigger via push (`verified` persists); ~40 min; whole-model gpu-smoke
   + the `[gpu-diag]` trisection. The official path.
2. **Remote friend's harness** (`packages/tts-ggml/remote-gpu-verify/`, zipped to
   `~/workstuff/qvac-mali-verify-kit.zip`) — agent-driven on a friend's physical Pixel 9 via raw `adb`. Runs
   the SAME H1 prebuild whole-model (`run-on-device.sh` → the trisection) AND `test-backend-ops` per-op
   (`run-backend-ops.sh`). Setup fetches public pieces (npm install — @qvac is PUBLIC on npm, no token; model
   via `download-tts-ggml-models.js`; bare CLI via `npm pack`). Faster iteration than CI; per-fix-round we send
   a small `prebuild-update.tgz` (`make-bundle.sh --prebuild-only`). **`test-backend-ops` CANNOT run in the CI
   device-farm** (Expo/Appium APK sandbox can't exec a raw ELF + `pull_request_target` runs the base workflow)
   — the friend's raw-adb harness IS the per-op path. Caveat: test-backend-ops uses random inputs + small-N
   shapes → may PASS despite the real-data bug (bci precedent); a FAIL localizes, a PASS doesn't clear — the
   whole-model `dprobe_pw1_mulmat` stays the oracle.

## ✅ RESOLVED — PR #2605 CI failure (diagnosed + fixed 2026-06-15)
**Issue:** #2605 CI (run 27562752301) failed despite #2605 NOT enabling Mali. Two relevant red
checks: `cpp-lint` (clang-format — out of scope per user) and `run-mobile-integration-tests /
Build Android` (job 81480422920); `merge-guard` failed downstream.

**Diagnosis (device-farm artifact `console-logs-qvac-tts-ggml-Android`, ID 7646333895):**
- Device split: **S25/Adreno PASSED 3/3**; **Pixel 9/Mali FAILED 1 of 3**. Failing runner =
  `runGpuSmokeTest`, only on Mali. Pixel `logcat_full`:
  - `[Chatterbox/GPU] backendDevice=0 backendId=0 (CPU) gpuUnsupported=undefined` → `not ok - Chatterbox/android: expected GPU backend, got CPU …`
  - `[Supertonic/GPU] backendDevice=0 backendId=0 (CPU) gpuUnsupported=1` → `ok - Supertonic/android: GPU present but unsupported vendor … correctly using CPU`
- Both correctly fall back to CPU on Mali (intended). gpu-smoke accept-branch
  (`gpu-smoke.test.js:108` `android && dev===0 && stats.gpuUnsupported`) fired for Supertonic
  but NOT Chatterbox because Chatterbox's `stats.gpuUnsupported` was `undefined` (key absent, ≠ 0).

**H12 — "Chatterbox `gpuUnsupported` undefined = C++/addon not surfacing it":** WRONG.
Verified `1d6fa830` (the commit CI built) has `ChatterboxModel.cpp:203,345` computing + emitting
the stat; `chatterbox_engine.cpp:801` returns `pimpl_->model.gpu_unsupported` (same model whose
`.backend` returned CPU on Mali → flag is set via the identical shared `init_gpu_backend` decline
path Supertonic uses). C++ produces `1` on Mali. So C++ is correct.

**H13 — "Test harness drops the field for Chatterbox":** CORRECT (root cause). The Chatterbox
gpu-smoke runner goes `runChatterboxTTS` → shared `runTTS()` (`test/utils/runTTS.js:223-233`),
which rebuilds stats into a whitelist `roundedStats` copying `backendDevice`/`backendId` but
**omitting `gpuUnsupported`**. Supertonic's runner (`runSupertonicTTS.js:68,99`) passes
`response.stats` wholesale, so its `gpuUnsupported=1` survived. → JS-only bug, not C++/Mali/lint.

**Fix (applied + pushed):** added `gpuUnsupported: stats.gpuUnsupported` to `roundedStats` in
`test/utils/runTTS.js`. Additive; `npx standard` clean. No C++/overlay/pin change.
- #2605 (`QVAC-20557-tts-ggml-android-gpu`): amended into the feature commit → **`600fc71f`**
  (was `1d6fa830`), force-pushed.
- #2610 (`QVAC-20557-mali-probe`): rebased onto the new #2605 tip → base `600fc71f` + probe pin
  **`dc0feb01`** (still pins tts-cpp@0aa594a6), force-pushed.

**Verify-next:** re-run #2605 device farm (apply `verified` label) → Pixel 9 Chatterbox gpu-smoke
now passes via the accept-branch; S25/Adreno stays green. (Can't exercise the Mali accept-branch
locally — no Mali; local Adreno engages the GPU so `dev===1`.) `cpp-lint` stays red (ignored per
user; trivially clang-format-fixable later if a fully-green run is wanted).

## DECISION: Option A — explicit GPU attention on OpenCL (capability-gated). Proceeding.

### Implementation plan for A (next steps — not yet done)
1. ggml-speech debug worktree already pins via local `vcpkg_from_git` (no change needed for A).
2. tts-cpp (`/Users/pratiknarola/workstuff/_wt-tts-cpp-gpu`):
   a. Add capability probe `backend_supports_f32_flash_attn_uncached(backend)` in `supertonic_gguf.cpp`
      (mirror `backend_supports_f16_kv_flash_attn_uncached` @424, but F32 K/V). On Adreno OpenCL →
      false (FA gated to CPU); Vulkan/Metal/CPU → true.
   b. Add `bool f32_flash_attn;` to `backend_capabilities` struct (@~704) + set in
      `cached_backend_capabilities` (@~780).
   c. Add `bool backend_supports_flash_attn = true;` to `supertonic_model` (`supertonic_internal.h`,
      near `backend_is_cpu`); set at load from `cached_backend_capabilities(model.backend).f32_flash_attn`
      (`supertonic_gguf.cpp` ~1785, where `use_native_leaky_relu` is set).
   d. Add helper `explicit_text_attention_ggml(ctx, q_in, k_in, v_in, scale, q_len, kv_len, n_heads,
      head_dim)` returning `[width=n_heads*head_dim, q_len]` to match FA's reshaped output:
      `kq = mul_mat(k_in, q_in)` → `soft_max_ext(kq, NULL, scale, 0)` → `v_t = cont(permute(v_in,1,0,2,3))`
      → `kqv = mul_mat(v_t, kq)` → `cont_2d(permute(kqv,0,2,1,3), width, q_len)`. (cont the q/k/v views
      first if strided-view mul_mat misbehaves — verify on device.)
   e. Gate both FA sites on `model.backend_supports_flash_attn`:
      - `append_text_attention_subgraph` @4373 (ONE-GRAPH / shipping path).
      - `build_text_attention_cache` @1153 (decomposed path).
3. Rebuild tts-cpp (~20s) + addon; push `.bare`; verify on Adreno 740:
   - Trace-diff (`SUPERTONIC_DISABLE_LOOP_GRAPH=1 SUPERTONIC_DISABLE_ONE_GRAPH=1 SUPERTONIC_TRACE_DIFF=1`):
     `ve_attn0_ctx`/`ve_attn0_out` must match CPU at step 0 AND step 1+ (no divergence).
   - Default path (`SUPERTONIC_STAGE_RMS=1`): cfm_latent ≈ 0.2585, wav ≈ 0.0374 (== CPU gold), not NaN.
4. If A verified: run full matrix {Supertonic,Chatterbox}×{CPU,OpenCL,Vulkan} on 740; then strengthen
   gpu-smoke; then device-farm (S25/Pixel9) + Xclipse; then PRs. If A's explicit attention miscomputes
   on Adreno (ctx fails at step 0) → fall back to option D (route Supertonic→Vulkan on Adreno).
5. CLEANUP before PR: remove all DEBUG scaffolding (GGML_OPENCL_FORCE_CPU_OPS hook, SUPERTONIC_STAGE_RMS,
   SUPERTONIC_TRACE_DIFF, the refuted rope-input cont @4330); switch overlays from local `vcpkg_from_git`
   back to `vcpkg_from_github` REF+SHA512 on pushed commits; decide f16-OpenCL-off gate's fate.

## Open decision — fix approach (RESOLVED → A)

- **A. Remove the OpenCL CPU-bridge: compute Supertonic text-attention with explicit GPU ops**
  (mul_mat QKᵀ · scale → soft_max → mul_mat V) instead of `ggml_flash_attn_ext`, on OpenCL only.
  All ops are OpenCL-supported and proven-correct here → GPU-resident, no bridge, no staleness.
  Mathematically identical to FA. tts-cpp change, generic. **(recommended)**
- **B. Fix the ggml-opencl cross-backend-copy-under-reuse staleness** (most generic; hardest;
  qvac-ext-ggml; coordinate with ggml-speech owner).
- **C. Make Adreno-OpenCL flash-attn correct + drop its guard** (big ggml task; the guard exists
  because FA miscomputes on large inputs).
- **D. Route Supertonic→Vulkan on Adreno** (Vulkan is correct + faster; "auto-selected backend works"
  fallback; not an OpenCL fix).

## Debug scaffolding currently in the build (REMOVE before PR)
- ggml-opencl: `GGML_OPENCL_FORCE_CPU_OPS` hook (commit on `QVAC-20557-ggml-opencl-dbg`).
- tts-cpp: `SUPERTONIC_STAGE_RMS` engine logging, `SUPERTONIC_TRACE_DIFF` per-op diff, rope-input
  `ggml_cont` (refuted — revert), the f16-OpenCL-off gate (keep or fold into final fix).
- tts-ggml overlay temporarily pins local `vcpkg_from_git` refs (revert to `vcpkg_from_github` for PR).
