'use strict'
// QVAC-19178: single source of truth for the VLM benchmark (models + presets).
//
// ─ What the benchmark compares ─
//   two-models      MODEL_1 vs MODEL_2 — two complete VLMs, one inference engine.
//                   They can be two BLOBS/VARIANTS of the same model (the default:
//                   Qwen3.5-0.8B with the mmproj at F16 vs Q8 — same LLM, different
//                   projector) or two DIFFERENT models (point the two `llm` blobs at
//                   different models). Runs on every target (desktop + mobile, CPU + GPU).
//   several-sources SOURCES_MODEL across several inference engines (addon / fabric-cli
//                   / upstream-cli). Desktop-only — the CLIs are native binaries.
//
// ─ Targets ─ A "target" is a (platform × backend) pair. Platform-agnostic:
//   desktop (Linux by default) and mobile (Samsung Galaxy S25 by default), each on
//   CPU and GPU where applicable. Adding an OS/phone is a workflow/runner change.
//   • desktop reads the active preset from QVAC_VLM_PRESET (per-field QVAC_VLM_*
//     overrides); on mobile there is NO env passthrough, so it always uses
//     `defaultPreset` below.
//
// ─ A "model" ─ a complete VLM: a main LLM blob + a vision-projector (mmproj) blob.
//   Each blob carries a `source` descriptor (how to fetch the bytes) and an optional
//   `registry` annotation (a published QVAC-registry entry; reported as Source =
//   "Registry"). See resolveBlob() in harness.cjs.
//     source.type 'hf'  : { type:'hf', repo, sha, file } -> pinned HuggingFace commit
//     source.type 'url' : { type:'url', url }             -> arbitrary direct link
//     source.type 's3'  : { type:'s3', url }              -> S3 (presigned URL)

// QVAC-21372: the A1 model pair — Gemma4-E2B (prefill prefers CPU on M4) vs Qwen3.5-2B
// (prefill prefers GPU). Sourced from the unsloth GGUF repos at revision `main`, matching
// benchmarks/performance/models.manifest.json and the gemma4/qwen integration tests. We use
// a `url` source (the same `resolve/main/...` link those tests use): the repos are public,
// so the mobile (Device Farm) legs — which forward no HF token — can fetch them too.
const HF = 'https://huggingface.co'
const GEMMA_REPO = 'unsloth/gemma-4-E2B-it-GGUF'
const QWEN_REPO = 'unsloth/Qwen3.5-2B-GGUF'

// url-source blob helper: { modelName (local cache file), origin (human label),
// source: { type:'url', url } }.
function url (modelName, origin, repo, file) {
  return { modelName, origin, source: { type: 'url', url: `${HF}/${repo}/resolve/main/${file}` } }
}

// ════════════════════ THE TWO MODELS UNDER TEST (two-models mode) ════════════════════
// QVAC-21372 uses two DIFFERENT models (not two mmproj variants): the A1 pair. Each is run
// on CPU and GPU (premise: CPU-vs-GPU prefill) on every target — the mobile legs measure the
// per-SoC premise; the desktop legs add the hybrid hook (several-sources, below). Q4_K_M is
// the cross-leg variant (fits every phone + desktop); mmproj at F16.
const MODEL_1 = {
  label: 'gemma4-e2b-q4km', // short id — report column + marker key (keep filesystem-safe)
  name: 'Gemma4-E2B-it · Q4_K_M · mmproj-F16',
  ctx_size: '4096',
  llm: url('gemma-4-E2B-it-Q4_K_M.gguf', 'unsloth/gemma-4-E2B-it-GGUF · Q4_K_M',
    GEMMA_REPO, 'gemma-4-E2B-it-Q4_K_M.gguf'),
  mmproj: url('gemma-4-E2B-it-mmproj-F16.gguf', 'unsloth/gemma-4-E2B-it-GGUF · mmproj-F16',
    GEMMA_REPO, 'mmproj-F16.gguf')
}

const MODEL_2 = {
  label: 'qwen35-2b-q4km', //  short id
  name: 'Qwen3.5-2B · Q4_K_M · mmproj-F16',
  ctx_size: '4096',
  llm: url('Qwen3.5-2B-Q4_K_M.gguf', 'unsloth/Qwen3.5-2B-GGUF · Q4_K_M',
    QWEN_REPO, 'Qwen3.5-2B-Q4_K_M.gguf'),
  mmproj: url('Qwen3.5-2B-mmproj-F16.gguf', 'unsloth/Qwen3.5-2B-GGUF · mmproj-F16',
    QWEN_REPO, 'mmproj-F16.gguf')
}

// ════════════════════ THE MODEL FOR SOURCE COMPARISON (several-sources mode) ════════════════════
// The desktop hybrid leg runs this one VLM through fabric-cli with QVAC_PREFILL_CPU off vs on.
// We pick Gemma4-E2B: on M4 its prefill prefers CPU, so the hybrid hook is the meaningful test.
// Its blob filenames must match the names the workflow's CLI step feeds to fabric-cli.
const SOURCES_MODEL = {
  label: 'gemma4-e2b-q4km',
  name: 'Gemma4-E2B-it · Q4_K_M (mmproj-F16)',
  ctx_size: '4096',
  llm: url('gemma-4-E2B-it-Q4_K_M.gguf', 'unsloth/gemma-4-E2B-it-GGUF · Q4_K_M',
    GEMMA_REPO, 'gemma-4-E2B-it-Q4_K_M.gguf'),
  mmproj: url('gemma-4-E2B-it-mmproj-F16.gguf', 'unsloth/gemma-4-E2B-it-GGUF · mmproj-F16',
    GEMMA_REPO, 'mmproj-F16.gguf')
}

// Open-licensed fixture tasks (regenerate/curate via build-fixture.cjs;
// per-image attribution in fixture.NOTICE.md).
const TASKS = ['textvqa', 'vizwiz', 'gqa', 'docvqa', 'ai2d']

module.exports = {
  // ════════════════════════ MODE — what is compared ════════════════════════
  // 'two-models' | 'several-sources'. The workflow's matrix_mode input sets it on
  // desktop (QVAC_VLM_MODE); on mobile this default is used.
  mode: 'two-models',

  // two-models compares these two complete VLMs:
  models: [MODEL_1, MODEL_2],
  // several-sources runs this one VLM across the engines below:
  sourcesModel: SOURCES_MODEL,
  engines: ['addon', 'fabric-cli', 'upstream-cli'],
  engine: 'addon', //         the fixed engine for two-models

  // Report column labels for two-models (derived from the two models above).
  base: MODEL_1.label,
  candidate: MODEL_2.label,

  // ════════════════════════ mmproj projector backend ════════════════════════
  // QVAC-21257: which backend runs the multimodal projector (vision encoder).
  //   'auto' — leave the addon's per-platform default (Android CPU, desktop/iOS GPU)
  //   'cpu' / 'gpu' — force the projector backend via the addon's mmproj-use-gpu key
  //   'both' — mmproj-compare: run ONE model (mmprojModel) on the GPU model-backend
  //            with the projector on CPU vs GPU as the two report columns.
  // Desktop overrides this with QVAC_VLM_MMPROJ_GPU; on mobile (no env passthrough)
  // this default governs the on-device run.
  // QVAC-21372: 'auto' (NOT 'both') so the mobile run stays in two-models mode and runs
  // BOTH models with the LLM on CPU and GPU (the A1 prefill premise) rather than the
  // QVAC-21257 mmproj-placement comparison.
  mmprojGpu: 'auto',
  // Single VLM used by mmproj-compare (mmprojGpu='both'); reuses MODEL_2's blobs.
  mmprojModel: MODEL_2,

  // QVAC-21372: add a third "<model>-hybrid" leg per model on the GPU model-backend with
  // the engine's single-load hybrid-prefill hook on (QVAC_PREFILL_CPU=1, set per-leg in
  // harness.cjs). Compares GPU-baseline vs prefill-routed-to-CPU. Requires a prebuild
  // built from the fork branch carrying the hook (vcpkg overlay port).
  hybridPrefill: true,

  // QVAC-21372: small decode budget for the mobile premise+hybrid run. The A1 headline is
  // TTFT/prefill (pre-decode), so a short decode keeps a decode-TPS estimate while letting
  // all 6 legs (2 models × cpu/gpu/gpu-hybrid) finish within the Device Farm session.
  nPredict: 16,

  // ════════════════════════ PRESET — how much is run ════════════════════════
  // A preset is purely the run size (tasks × samples × repeats); it is independent of
  // the mode. Used verbatim on mobile, and the desktop default when QVAC_VLM_PRESET is
  // unset. Per-field desktop env overrides:
  //   QVAC_VLM_SAMPLES→samplesPerTask · QVAC_VLM_REPEATS→repeats
  //   QVAC_VLM_DEVICES→devices (csv) · QVAC_VLM_TASKS→tasks (csv)
  // `devices: null` = CPU + GPU where applicable; `tasks: null` = all fixture tasks.
  // QVAC-21372: mobile runs the A1 premise+hybrid (6 legs/device). Gemma4-E2B on a phone
  // CPU is very slow, so the full `base` set times out the Device Farm session before all
  // legs finish. Use the lighter `mobilea1` preset on mobile so every leg completes.
  defaultPreset: 'mobilea1',

  presets: {
    // smoke — 1 task, 1 image, 1 repeat: a single inference per config (wiring check).
    smoke: { tasks: ['textvqa'], samplesPerTask: 1, repeats: 1, devices: null },
    // base — DEFAULT eval: 5 tasks × 3 samples × 1 repeat.
    base: { tasks: TASKS, samplesPerTask: 3, repeats: 1, devices: null },
    // full — 5 tasks × 5 samples × 1 repeat (the complete fixture).
    full: { tasks: TASKS, samplesPerTask: 5, repeats: 1, devices: null },
    // mobilea1 (QVAC-21372) — 3 tasks × 1 sample so all 6 legs (2 models × cpu/gpu/
    // gpu-hybrid) finish within the Device Farm session on slow phone CPU legs.
    mobilea1: { tasks: ['textvqa', 'docvqa', 'ai2d'], samplesPerTask: 1, repeats: 1, devices: null }
  }
}
