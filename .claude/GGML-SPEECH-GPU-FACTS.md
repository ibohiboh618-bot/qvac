# ggml-speech / qvac GPU — structural facts & traps

Read before debugging tts-cpp / parakeet-cpp / whisper-cpp on a GPU backend. Confirmed from
QVAC-20556/20557. Check each against your target FIRST. Companion to `~/.claude/CLAUDE.local.md` (R9).

## Execution model
- Supertonic CFM stage & parakeet encoder run on a SINGLE backend via `ggml_backend_graph_compute(backend, gf)` — NO `ggml_backend_sched`. So `supports_op` / `ggml_backend_supports_op` gates are DEAD CODE on these paths (never consulted, can't move an op to CPU, can't be force-CPU-bisected). Chatterbox DOES use a scheduler — supports_op works there. grep `ggml_backend_sched` before any routing edit. ⟨20557 H2, 20556 FACT 2 / R3c-d-f⟩
- Single-backend graph_compute ABORTS on an unsupported op instead of CPU-falling-back — abort-not-fallback is the fingerprint of "no scheduler here." ⟨20556 R3e Metal abort⟩

## ggml op lowering (function name ≠ emitted op)
- `ggml_conv_2d_dw` (depthwise) decomposes to `im2col`(F16) + reshape + broadcast `mul_mat` — emits NO `GGML_OP_CONV_2D_DW`. Only `ggml_conv_2d_dw_direct` emits that op. ⟨20556 R3a/FACT 1⟩
- `ggml_conv_2d` lowers to `im2col + mul_mat`.
- `GGML_OP_CONV_2D_DW` has NO ggml-metal encoder → swapping to `_direct` regresses iOS/Metal. ⟨20556 R3e⟩

## Backend correctness (per-driver)
- Adreno OpenCL: fused `FLASH_ATTN_EXT` MISCOMPUTES on real inputs (passes test-backend-ops) → force-routed to CPU (ggml-opencl.cpp:4770). `mul_mat`/`soft_max`/`im2col` are correct. Fix for the FA-reuse staleness = explicit `mul_mat+soft_max+mul_mat` on GPU (no FA, no CPU bridge). F32×F16 mat-vec aborts → keep the f16-off gate. ⟨20557 H8–H11⟩
- Mali-G715 (Valhall) Vulkan: miscomputes the depthwise im2col(F16)+broadcast-mul_mat decomposition (non-deterministic inf/nan). Regular conv (sub_conv0 im2col+mul_mat) is clean. Chatterbox works on Mali-Vulkan; Supertonic/parakeet don't (per-model, per-op). ⟨20556 R2/R3a, 20557 Mali diag⟩
- Samsung Xclipse 920 Vulkan: correct for Supertonic + Chatterbox.
- `test-backend-ops` can PASS an op that fails on the real model (fresh buffers + random inputs hide uninitialized-memory / non-det bugs). Don't let green test-backend-ops exonerate an op. ⟨bci⟩

## Backend selection (tts-cpp)
- `init_gpu_backend` (backend_selection.cpp) allowlist = Qualcomm Adreno (+ Samsung Xclipse) ONLY; Mali/others → CPU by design (Mali aborts via GGML_ASSERT). "Android GPU → Vulkan" is FALSE — verify in source. `gpu_unsupported()` flags a policy-declined GPU so gpu-smoke accepts CPU fallback. ⟨20557⟩

## On-device diagnostics (Android device farm)
- Native `stderr`, `std::cout`, and the async QLOG/JsLogger (uv_async) bridge are DROPPED on-device. Only `__android_log_print` reaches `logcat_full.txt` (captured unconditionally); synchronous JS `console.log` also reaches host. Route diag via `ggml_log_set` → `__android_log_print` + link `liblog`. ⟨20556 L1/L2, 20557 diag round⟩
- `std::isnan`/`std::isinf` → no-ops under fast-math; use bit-pattern NaN checks in diag.
- Test-harness stat whitelists can drop fields (`runTTS.js` roundedStats omitted gpuUnsupported) — verify the JS path surfaces the field, not just C++. ⟨20557 H13⟩
