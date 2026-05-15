# QVAC-18297: b9025 Cherry-Picked Changes

**Date**: 2026-05-15
**Fiber base**: `tetherto/temp-8189` (`f686a1324`)
**Upstream**:   tag `b9025` = ggml-org/llama.cpp `eff06702b`
**Merge base**: `4d828bd1a` (836 commits behind b9025)

---

## RC1 — Fused `GATED_DELTA_NET` port

- **Branch**: `feat/QVAC-18297-rc1-gated-delta-net`
- **Commit**: `98b08344f`
- **Type**: **Adapted port** (not a direct cherry-pick)
- **Upstream sources** (4 commits squashed into one fiber-side commit):

| SHA | PR | Title |
|---|---|---|
| `c5a778891` | #19504 | ggml: add GATED_DELTA_NET op |
| `d28961d81` | #20340 | llama: enable chunked fused GDN path (+ Metal kernel #20361) |
| `e30f1fdf7` | #20443 | graph: remove redundant GDN state transposes |
| `f17b3be63` | #20468 | llama: fix pooling assertion in chunked GDN detection |

### Why a port, not a cherry-pick

The four upstream commits cumulatively touch ~42 files across ggml core,
ggml-cpu, ggml-metal, ggml-cuda, ggml-vulkan, `src/models/`, and
`src/llama-context.cpp`. Fiber uses `llm_build_*` builder classes while
upstream b9025 has refactored these to `llama_model_*::graph`, so the
model-dispatch portions cannot apply cleanly. The fiber-side port squashes
the four upstream commits into one and (a) drops CUDA/Vulkan/HIP backends
fiber doesn't build, (b) ports the Metal kernel verbatim, (c) re-wires
model dispatch under fiber's class hierarchy.

---

## RC3 — Metal Flash Attention `dk512_dv512` instantiations

- **Branch**: `feat/QVAC-18297-rc3-fa-dk512`
- **Commit**: `460207e83`
- **Type**: **Direct cherry-pick**
- **Upstream source**:

| SHA | PR | Title |
|---|---|---|
| `342d6125b` | #20902 | metal: add FA instantiations for HSK=512, HSV=512 |

---