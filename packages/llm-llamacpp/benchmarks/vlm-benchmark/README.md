# VLM Benchmark

A **universal quality + speed benchmark** for vision-language inference with
`@qvac/llm-llamacpp`. It runs one frozen image fixture through a
chosen configuration and renders a single consolidated report, so the same numbers
are produced — and directly comparable — across platforms and backends.

It is built to be **flexible first, with sensible defaults**: out of the box it
compares Qwen3.5-0.8B with its vision projector at F16 vs Q8 across a desktop and a
mobile platform, but every axis (model, engine, platform, backend, tasks, samples) is
configurable.

---

## How it works

The harness loads a model configuration, runs every fixture sample, and prints
machine-readable markers to the log; a host-side script collects those logs and turns
them into one report. The **same harness runs on every target**, which is what makes
results comparable.

Pipeline:

1. **Harness** — `vlm-matrix.test.js` (entry) → `harness.cjs` loads the model(s), runs
   each `(task, sample, repeat)`, and emits `[VLMROW]` / `[VLMSEG]` / `[VLMMETA]`
   markers to stdout.
2. **Collect** — CI gathers the per-target logs as artifacts.
3. **Aggregate** — `aggregate.js` parses the markers, scores quality, and writes a
   Markdown report to the workflow **step summary**, a **PR comment**, and an artifact.

Everything ships from **this one directory**
(`packages/llm-llamacpp/benchmarks/vlm-benchmark/`). Desktop runs the files in place;
the mobile build first runs `stage.cjs`, which copies the entry + harness + config +
fixture into `test/integration/` and the images into `test/mobile/testAssets/` (both
git-ignored) so the mobile test generator and app bundler pick them up. The
`../../`-relative requires resolve identically from either location.

---

## Platforms & backends

A run targets one or more **(platform × backend)** pairs. The benchmark is
platform-agnostic — adding an OS or a device is a runner/workflow change, not a config
change.

Every leg is a dispatch token — pick any combination per run, e.g.
*"linux-cpu, linux-gpu, iphone17-cpu, s25-cpu, s25-gpu"*.

**Desktop** — `matrix_desktop`, tokens `<os>-<backend>`:

| token | runner | GPU backend |
|---|---|---|
| `linux-cpu` / `linux-gpu` | `qvac-ubuntu2204-x64` / `qvac-ubuntu2404-x64-gpu` | Vulkan |
| `macos-cpu` / `macos-gpu` | `macos-15-xlarge` (GitHub-hosted Apple Silicon VM) | Metal |
| `macmini-cpu` / `macmini-gpu` | `mac-mini-m4-gpu` (self-hosted bare-metal M4) | Metal |
| `windows-cpu` / `windows-gpu` | `qvac-win25-x64` / `qvac-win25-x64-gpu` | Vulkan |

**Mobile (AWS Device Farm)** — `matrix_mobile`, tokens `<device>[-<backend>]`;
a bare device token runs CPU **and** GPU in one on-device session, a `-cpu`/`-gpu`
suffix pins one backend:

| device token | Device Farm filter |
|---|---|
| `s25` | Samsung · "S25 Ultra" (Android) |
| `pixel9` | Google · "Pixel 9" (Android) |
| `iphone16` | Apple · "iPhone 16" (iOS) |
| `iphone17` | Apple · "iPhone 17" (iOS; CONTAINS — may pick a 17-family variant) |
| `iphone17pro` | Apple · "iPhone 17 Pro" (iOS) |

- Each mobile leg schedules **exactly one phone** (model filter, maxDevices 1)
  through the `integration-mobile-test-llm-llamacpp.yml` workflow; the backend
  selection and the dispatched mode/preset are forwarded to the device via the
  `qvacPerfConfig.txt` push channel (`device_env`).
- **Adding a platform is one map entry**: a desktop OS = one case in the workflow
  `context` job's `dmatrix` step; a phone (e.g. a future S26) = one case in the
  `mmatrix` step — provided the device exists in the Device Farm fleet.
- The config never names a platform — it only selects backends via `devices`
  (`cpu` / `gpu` / both).

---

## Modes

The benchmark runs in exactly one **comparison mode** per run. Each mode fixes one axis
and varies another.

| | **two-models** | **several-sources** |
|---|---|---|
| Varies | the **model** | the **inference engine** |
| Holds fixed | the engine (default `addon`) | the model |
| Compares | `MODEL_1` vs `MODEL_2` | `addon` vs `fabric-cli` vs `upstream-cli` |
| Default example | Qwen3.5 mmproj-F16 vs mmproj-Q8 | Qwen3.5 + q8 mmproj across all three engines |
| Targets | desktop + mobile, CPU + GPU | **Linux only** (the CLIs are native binaries built there) |
| Headline metric | per-model quality + vision-encode time | per-engine quality + encode/TTFT |

**two-models** compares the two complete VLMs configured as `MODEL_1` and `MODEL_2`.
Each is a main LLM blob + an mmproj blob. They can be **two blobs/variants of the same
model** (the default — Qwen3.5 with the projector at F16 vs Q8: same `llm`, different
`mmproj`) or **two different models** (point the two `llm` blobs at different models).
The report labels the two columns from each model's `label`.

---

## Configuration

All behavior lives in **`config.cjs`** — the single source of truth, staged to the
device so it configures every target. Two independent axes:

- **mode** (what's compared) — `mode` field / `matrix_mode` input.
- **preset** (how much is run) — `defaultPreset` field / `matrix_preset` input.

**The models** are explicit at the top of the config:

| constant | used by | meaning |
|---|---|---|
| `MODEL_1`, `MODEL_2` | two-models | the two complete VLMs to compare |
| `SOURCES_MODEL` | several-sources | the one VLM run through every engine |

Each is a full spec — `label`, `name`, `ctx_size`, an `llm` blob and an `mmproj` blob
(each blob has a `source` descriptor + optional `registry` annotation). Edit those
constants to change what runs; nothing else needs to change.

**Presets** are pure run-size bundles (independent of mode):

| preset | tasks × samples × repeats | use |
|---|---|---|
| `smoke` | 1 task × 1 × 1 | a single inference per config — wiring check |
| `base` | 5 tasks × 3 × 1 | **default** evaluation |
| `full` | 5 tasks × 5 × 1 | the complete fixture |

**Run knobs** (preset fields). Each is overridable by env on every target — desktop
gets env directly from the workflow; mobile gets it via the `qvacPerfConfig.txt`
file the workflow pushes to the device (`device_env`):

| field | env override | meaning |
|---|---|---|
| `samplesPerTask` | `QVAC_VLM_SAMPLES` | images per task |
| `repeats` | `QVAC_VLM_REPEATS` | runs per sample, mean reported (default 3 desktop / 1 mobile) |
| `devices` | `QVAC_VLM_DEVICES`, `NO_GPU` | backends; `null` = CPU + GPU where applicable |
| `tasks` | `QVAC_VLM_TASKS` | task subset; `null` = all fixture tasks |

**Which preset runs where.** Every leg uses `QVAC_VLM_PRESET` (set from the
workflow's `matrix_preset` input — forwarded to phones as device env), falling
back to the committed `defaultPreset` when run outside the workflow.

**Model sources.** Each model blob carries a `source` descriptor — `hf` (pinned
HuggingFace commit), `url` (direct link), `s3` (presigned URL) — plus an optional
`registry` annotation (a published QVAC-registry entry; reported as Source = "Registry",
bytes fetched from its canonical pinned URL). See `resolveBlob()` in `harness.cjs`.

---

## Running it

The benchmark is driven by the **Benchmark VLM (model comparison)** workflow
(`.github/workflows/benchmark-vlm-model-comparison.yml`). *Run workflow* (or `gh workflow run`)
→ set `run_matrix = true`, then pick the axes via the dispatch inputs below.

### Launch configuration checklist

There are **two ways to configure a launch**, and each item below shows both:

- **Config (committed):** edit `config.cjs` and push. Required for the model choice
  (models are config-only); also the fallback for mode/preset when run outside the
  workflow.
- **Dispatch (`-f`):** pass to `gh workflow run` (or the *Run workflow* UI). Overrides
  the config on **every leg** — desktop via env, phones via the pushed device env —
  no commit needed.

Walk it top-to-bottom. Steps 1–2 (model + source versions) decide *what* is measured;
3–9 decide *how* it runs.

**1. Set the model(s).** *(dispatch or config — see `CONTRACT.md` §3)*
   - **Dispatch — ANY model, zero code changes:** `-f matrix_models=…`, comma-separated:
     catalog names (`qwen3.5-f16,qwen3.5-q8`) and/or ad-hoc pairs
     `[label=]<llm-gguf-url>|<mmproj-gguf-url>[@ctx=N]` (two https URLs are all a model
     needs; HF resolve-URLs are reported with repo+ref provenance), or `json:[…]` for
     exotic cases (registry sources — desktop-only). Empty = config `defaultModels`.
   - **Config:** edit the `catalog` / `MODEL_1`/`MODEL_2` specs in `config.cjs` (two
     blobs of one model → same `llm`, different `mmproj`; distinct `label` each).
   - several-sources mode always runs the committed `SOURCES_MODEL`.

**2. Update the source versions.**
   - **Builds under comparison:** `-f matrix_sources=addon,fabric@v8189.0.2,upstream@b8189`
     (the CLI refs ride inside the tokens; `addon` = the published npm prebuild;
     `addon@candidate`/`addon@baseline` are reserved until A2 lands).
   - **Model version:** bump the pinned commit in `config.cjs` (`SHA.*` / the blob's
     `source.sha`) — or just dispatch the new URL via `matrix_models`.
   - **Fixture images:** stored in a fixture object store (URI configured in the
     benchmark workflow), not git; you may download them separately for local tests.
     Regenerate with `build-fixture.cjs`, then upload `./images/` to that store; CI pulls
     them per run (needs the `release` environment for the OIDC role).

**3. Mode** — what's compared.
   - Config: `mode: 'two-models' | 'several-sources'`.
   - Dispatch: `-f matrix_mode=…` (every leg; forwarded to phones as device env).

**4. Preset** — task group: `smoke` (1 task, wiring check) · `cognitive` (5 VQA tasks × 5) ·
   `ocr` (1 light `ocr-page` doc — quick document-OCR check, fits the mobile session) ·
   `ocrhighmp` (all 5 high-MP `ocr-page` docs — desktop-oriented) ·
   `full` (cognitive + `ocr-small` + the 1 light `ocr-page`).
   - Config: `defaultPreset: '…'` (and the `presets` definitions: tasks/samples/repeats).
   - Dispatch: `-f matrix_preset=…` (every leg; forwarded to phones as device env).
     Keep mobile light (`base` or below); `full` risks the Device Farm session window.

**5. Desktop platforms × backends.**
   - Dispatch: `-f matrix_desktop=…` — any subset of `{linux,macos,macmini,windows}-{cpu,gpu}`
     (gpu = Vulkan on Linux/Windows, Metal on macOS/Mac mini).
   - Config: backends per preset via `devices` (`null` = both); env `NO_GPU=true`.

**6. Mobile devices × backends (AWS Device Farm).**
   - Dispatch: `-f matrix_mobile=s25,pixel9,iphone16,iphone17,iphone17pro` tokens, each
     optionally suffixed `-cpu`/`-gpu` (bare = both in one session). Empty = no mobile;
     two-models only — ignored for several-sources.

**7. Task set** — `scenarios.cjs` defines one `default` set: the 5 VQA tasks
   (textvqa/vizwiz/gqa/docvqa/ai2d) + the OCR tasks (ocr-small/ocr-page). Quality is
   reported per task, **not gated** (different models are compared, so there's no
   candidate-vs-baseline accuracy regression to gate on). OCR tasks score by CER/WER/BLEU
   in a separate table.
   - Dispatch: `-f matrix_scenarios=…` (single set today; forwarded to phones as device env).

**8. Samples / repeats / tasks.**
   - Samples — Config: preset `samplesPerTask`; Dispatch: `-f matrix_samples=N`.
   - Repeats / tasks — Config: preset `repeats` / `tasks`; (local env
     `QVAC_VLM_REPEATS` / `QVAC_VLM_TASKS` — no dispatch input).

**Dispatch inputs reference** (GitHub caps `workflow_dispatch` at 10 inputs — the set is full)

| input | overrides | purpose |
|---|---|---|
| `run_matrix` | — | **must be true** to run the matrix at all |
| `matrix_mode` | `config.mode` | `two-models` \| `several-sources` (every leg) |
| `matrix_preset` | `config.defaultPreset` | `smoke` \| `cognitive` \| `ocr` \| `ocrhighmp` \| `full` (every leg) |
| `matrix_models` | `config.defaultModels` | catalog names / `[label=]<llm-url>\|<mmproj-url>[@ctx=N]` / `json:[…]` (CONTRACT.md §3) |
| `matrix_sources` | — | builds under comparison: `addon` \| `fabric@<ref>` \| `upstream@<ref>` (`addon@candidate/baseline` reserved, A2) |
| `matrix_scenarios` | `config.defaultScenario` | task set (single `default` today) |
| `matrix_desktop` | — | desktop legs: `{linux,macos,macmini,windows}-{cpu,gpu}` (any subset) |
| `matrix_mobile` | — | mobile legs: `{s25,pixel9,iphone16,iphone17,iphone17pro}[-{cpu,gpu}]` (any subset; empty = none; two-models only) |
| `matrix_samples` | preset `samplesPerTask` | override samples/task, every leg (empty = default) |

**Example** — two-models, mixed leg selection, base preset, one ad-hoc model:

```bash
gh workflow run benchmark-vlm-model-comparison.yml --ref <branch> \
  -f run_matrix=true -f matrix_mode=two-models -f matrix_preset=full \
  -f matrix_models="qwen3.5-q8,challenger=https://huggingface.co/org/NewVLM-GGUF/resolve/<sha>/NewVLM-Q4_K_M.gguf|https://huggingface.co/org/NewVLM-GGUF/resolve/<sha>/mmproj-F16.gguf" \
  -f matrix_desktop=linux-cpu,linux-gpu,macos-gpu \
  -f matrix_mobile=s25-cpu,s25-gpu,iphone17
```

**Locally** you can run the harness directly under `bare` (desktop) by exporting
`QVAC_VLM_MATRIX=1` plus any `QVAC_VLM_*` overrides (`QVAC_VLM_MODE`, `QVAC_VLM_PRESET`,
`QVAC_VLM_MODELS`, `QVAC_VLM_SCENARIOS`, `QVAC_VLM_SAMPLES`, `QVAC_VLM_REPEATS`,
`QVAC_VLM_DEVICES`, `QVAC_VLM_TASKS`, `NO_GPU`); the several-sources CLIs are built and
driven by `cli-fixture-runner.cjs`. `node run-desktop.cjs --selfcheck` validates the
config/contract wiring without running any model.

---

## Metrics & report

Two metric families, one per inference: a quality score (matched to the task) and a set
of speed timings. The report rolls them up per (platform × backend × config).

**Quality** — one lmms-eval-style metric per task; the equal-weight mean across tasks is
"Overall %":

| metric | tasks | how |
|---|---|---|
| `vqa` | textvqa, vizwiz, gqa | normalized exact match vs the answer set (min(1, hits/3)) |
| `anls` | docvqa | Average Normalized Levenshtein Similarity (≥0.5) |
| `relaxed` | (chartqa) | numeric within ±5% or string match |
| `mc` | ai2d | the stated letter (explicit "answer: X" or a short letter-led reply) |

**Speed** — `mmproj` vision-encode ms (the headline for an mmproj quant; parsed from
llama.cpp native stderr), TTFT, decode TPS, wall ms.

**Report layout** — (1) **Highlights** (quality + speed at a glance), (2) **Details**
(models & origins with Source, HW/SW provenance, full matrices), (3) **Test Results**
(per-target pass counts), (4) **Image samples** (task → image → W×H).

---

## Extending

The benchmark is meant to grow. The three common changes:

- **Add tasks / refresh images.** `node build-fixture.cjs --per-task 3 --max-side 1024`
  iterates the HuggingFace datasets-server, **filters on resolution without
  downloading**, keeps only open-licensed datasets (allowlist), writes images to
  `./images/`, regenerates `fixture.data.cjs`, and updates `fixture.NOTICE.md`
  (per-image attribution). Adding a task = one manifest entry. **The images are not
  committed — they live in a fixture object store** (URI configured in the benchmark
  workflow); after regenerating, upload `./images/` to that store. CI syncs it →
  `images/` before each run (desktop and, before `stage.cjs`, mobile).
- **Change the models.** Edit `MODEL_1` / `MODEL_2` (two-models) or `SOURCES_MODEL`
  (several-sources) in `config.cjs` — give each blob a `source` descriptor. To compare
  two variants of one model, point both at the same `llm` and vary only the `mmproj`.
- **Add platforms.** Desktop: one case in the `matrix_desktop` → runner map
  (`dmatrix` step of the workflow `context` job). Mobile: one case in the device
  map (`mmatrix` step) — any phone available in the Device Farm fleet. No harness
  changes.

---

## Known limitations

- **several-sources is Linux-only.** `fabric-cli`/`upstream-cli` are native binaries
  built by the Linux legs; the mobile path runs an addon app, not arbitrary CLIs.
- **mmproj vision-encode time is unavailable on mobile.** It comes from llama.cpp's native
  stderr, which neither Android logcat nor the iOS console capture carries — the report
  shows `—` there and uses **TTFT** (which includes vision-encode) as the mobile proxy.
- **addon vs CLI prompt parity.** The addon API sends the image as its own `user` turn
  (~+11 tokens) vs the CLIs' single turn, so the *addon-vs-CLI* quality comparison is
  not strictly apples-to-apples. `fabric-cli` vs `upstream-cli` share an identical
  prompt and is the clean engine comparison. (True addon parity needs an addon-side
  single-turn API.)
- **MC (ai2d).** Only an explicit/short letter answer is scored; a reasoning paragraph
  with no stated choice scores 0 (by design — avoids grabbing a random letter from prose).
- **Registry source on mobile.** The P2P registry client isn't bundled into the mobile
  app; registry blobs are fetched via their pinned HTTPS origin (byte-identical) on
  every target.
- **Small n.** Defaults are 3 samples × 3 repeats; raise `samplesPerTask` for tighter
  quality estimates (borderline single-sample flips otherwise move the mean).

---

## Contract & parallel development

Active development (QVAC-19371 umbrella) is split into two independent workstreams —
**runner** (sources, methodology, metrics: `harness.cjs`, `models.cjs`, `sources.cjs`,
`methodology.cjs`, `run-desktop.cjs`, `config.cjs`, the workflow run jobs) and
**report** (scenarios, scoring, gate, views: `scenarios.cjs`, `aggregate.js`,
`combine.cjs`, `fixture*`, `score-check.cjs`, the workflow inputs + combine job).
They meet only at the frozen interface in **`CONTRACT.md`** (marker schema v2, env
vars, launch grammar); `markers-v2.sample.txt` is its executable sample — the report
side develops against it, `node run-desktop.cjs --selfcheck` validates it.

---

## Files

All in `packages/llm-llamacpp/benchmarks/vlm-benchmark/` unless noted:

| | |
|---|---|
| `CONTRACT.md`, `markers-v2.sample.txt` | **the frozen runner↔report contract** (marker schema v2, env vars, launch grammar) + its executable sample |
| `config.cjs` | run-side source of truth: modes, presets, model catalog, sources, methodology |
| `scenarios.cjs` | the task set (VQA + OCR) the benchmark runs — report-side owned |
| `models.cjs` | `matrix_models` grammar → canonical model specs (any model via two URLs) |
| `sources.cjs`, `methodology.cjs` | source tokens + measurement methodology helpers (A2/A3 build on these) |
| `run-desktop.cjs` | desktop run driver scaffold + `--selfcheck` contract guard |
| `combine.cjs` | combine driver: log discovery, host tagging, provenance, report, gate (B4) |
| `vlm-matrix.test.js`, `harness.cjs` | harness (loads models, emits markers) |
| `aggregate.js` | parses markers → report |
| `cli-fixture-runner.cjs` | runs the fixture through a native CLI (several-sources) |
| `cli-case-runner.js`, `stdout-parser.js`, `accuracy.js`, `utils.js`, `cli-source-config.js`, `build-cli-sources.js` | **vendored** native-CLI helpers — build + run fabric/upstream `llama-mtmd-cli` (several-sources). Self-contained; not imported from `vlm-performance` |
| `build-fixture.cjs` | open-licensed fixture generator |
| `fixture.data.cjs`, `fixture.NOTICE.md` | the frozen fixture manifest + attribution (images are in S3, synced into `images/` by CI) |
| `score-check.cjs` | offline metric-tuning harness — re-scores real predictions without re-running inference |
| `stage.cjs` | copies the above into `test/integration/` + `testAssets/` for the mobile build |
| `.github/workflows/benchmark-vlm-model-comparison.yml` | `run_matrix` jobs (desktop legs, mobile, combine) |

**Reused from the package** (on `main`, not copied): the addon (`../../index.js`) and
`ensureModel` (`../../test/integration/utils.js`). The several-sources native-CLI helpers
are **vendored** into this folder (above) so the benchmark is self-contained.
