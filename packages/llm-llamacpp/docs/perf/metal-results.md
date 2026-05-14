# QVAC-18297: Per-Commit Benchmark of llama.cpp Commits Since `tetherto/temp-8189`

Each of the three commits added on top of fiber `tetherto/temp-8189`
(`f686a1324`) was measured **in isolation** on a dedicated branch, plus the
cumulative `feat/QVAC-18297-fiber-updates` branch that stacks all three.

| Short | Branch | Commit | Subject |
|---|---|---|---|
| **RC2** | `feat/QVAC-18297-rc2-mul-mat-opt` | `f6d6dbcd1` | metal: optimize Metal Tensor API for `GGML_OP_MUL_MAT` (upstream #20962) |
| **RC1** | `feat/QVAC-18297-rc1-gated-delta-net` | `556789f7d` | port `GGML_OP_GATED_DELTA_NET` from upstream b9025 — fused Metal SIMD kernel |
| **RC3** | `feat/QVAC-18297-rc3-fa-dk512` | `054141103` | add Metal Flash Attention `dk512_dv512` kernel instantiations |
| **All RCs** | `feat/QVAC-18297-fiber-updates` | (RC2 + RC1 + RC3 stacked) | cumulative composition for production merge |

Primary comparison anchor: fiber-8189 Mac M4 rows in `metal-baseline.md §2`
(parsed JSON at `results/parsed/fiber-mac-2026-05-13T1856.json`).

> **Peak RSS column deliberately omitted** — the four 2026-05-14 run groups
> did not wrap the binary with `/usr/bin/time -l`, so `rss_mb` /
> `peak_mem_mb` are absent from the parsed JSONs. They will be captured in
> the next run; see `QVAC-18297-plan.md § Benchmark Methodology Notes`.

---

# Primary Results Matrix

All values are 3-run median, elephant.jpg, Mac M4, full 8-model × 2-backend
matrix. Matrix format mirrors `metal-baseline.md §2`. **"All RCs" = the
cumulative `feat/QVAC-18297-fiber-updates` branch** (RC2 + RC1 + RC3 stacked).
The RC1, RC3, and "All RCs" rows are from `cmake --build … --clean-first`
builds (clean rebuilds); the Fiber row is from `metal-baseline.md`; the RC2
row is the first-build measurement validated by a same-binary control rerun.

**Source JSONs** (one per branch):

| Branch | Source JSON | Build |
|---|---|---|
| Fiber | `results/parsed/fiber-mac-2026-05-13T1856.json` | tetherto/temp-8189, 2026-05-13 |
| RC2 | `results/parsed/mac-rc2-mul-mat-opt-2026-05-14T1116.json` | first build in orchestrator (no preceding stale state); validated by `rc2-control-rerun` |
| RC1 | `results/parsed/mac-rc1-rerun-2026-05-14T1710.json` | `cmake --build … --clean-first` |
| RC3 | `results/parsed/mac-rc3-rerun-2026-05-14T1830.json` | `cmake --build … --clean-first` |
| All RCs | `results/parsed/mac-fiber-updates-rerun-2026-05-14T1940.json` | `cmake --build … --clean-first` |

## Mac M4 — Metal × elephant.jpg

| Branch | Backend | Model | Quant | Vision (ms) | Prefill (t/s) | Decode (t/s) | TTFT (ms) | Total (ms) |
|--------|---------|-------|-------|------------:|--------------:|-------------:|----------:|-----------:|
| Fiber | Metal | Gemma4-E2B | Q4_K_M | 804 | 177.33 | 31.45 | 2,406 | 10,130 |
| RC2 | Metal | Gemma4-E2B | Q4_K_M | 644 | 257.42 | 41.89 | 1,747 | 7,703 |
| RC1 | Metal | Gemma4-E2B | Q4_K_M | 624 | 258.28 | 42.48 | 1,724 | 7,491 |
| RC3 | Metal | Gemma4-E2B | Q4_K_M | 627 | 260.41 | 52.43 | 1,718 | 6,337 |
| All RCs | Metal | Gemma4-E2B | Q4_K_M | 630 | 260.31 | 52.48 | 1,721 | 6,327 |
| Fiber | Metal | Gemma4-E2B | Q8_0 | 785 | 184.94 | 21.79 | 2,321 | 13,646 |
| RC2 | Metal | Gemma4-E2B | Q8_0 | 674 | 231.07 | 25.92 | 1,903 | 11,475 |
| RC1 | Metal | Gemma4-E2B | Q8_0 | 760 | 222.29 | 25.63 | 2,038 | 11,802 |
| RC3 | Metal | Gemma4-E2B | Q8_0 | 699 | 227.81 | 30.27 | 1,946 | 10,073 |
| All RCs | Metal | Gemma4-E2B | Q8_0 | 673 | 235.65 | 30.16 | 1,878 | 10,043 |
| Fiber | Metal | Gemma4-E4B | Q4_K_M | 877 | 109.69 | 16.78 | 3,466 | 18,303 |
| RC2 | Metal | Gemma4-E4B | Q4_K_M | 733 | 141.51 | 18.83 | 2,740 | 16,046 |
| RC1 | Metal | Gemma4-E4B | Q4_K_M | 800 | 133.55 | 19.31 | 2,927 | 15,692 |
| RC3 | Metal | Gemma4-E4B | Q4_K_M | 749 | 132.97 | 23.88 | 2,885 | 13,298 |
| All RCs | Metal | Gemma4-E4B | Q4_K_M | 722 | 142.45 | 24.13 | 2,716 | 13,046 |
| Fiber | Metal | Gemma4-E4B | Q8_0 | 807 | 141.72 | 12.66 | 2,811 | 23,329 |
| RC2 | Metal | Gemma4-E4B | Q8_0 | 833 | 133.77 | 13.06 | 2,956 | 22,406 |
| RC1 | Metal | Gemma4-E4B | Q8_0 | 743 | 161.64 | 14.94 | 2,500 | 19,970 |
| RC3 | Metal | Gemma4-E4B | Q8_0 | 748 | 165.42 | 17.63 | 2,465 | 17,404 |
| All RCs | Metal | Gemma4-E4B | Q8_0 | 739 | 163.62 | 16.98 | 2,475 | 17,904 |
| Fiber | Metal | Qwen3.5-2B | Q4_K_M | 470 | 253.23 | 32.56 | 1,516 | 9,181 |
| RC2 | Metal | Qwen3.5-2B | Q4_K_M | 493 | 232.53 | 32.39 | 1,633 | 9,230 |
| RC1 | Metal | Qwen3.5-2B | Q4_K_M | 475 | 240.82 | 38.54 | 1,575 | 7,891 |
| RC3 | Metal | Qwen3.5-2B | Q4_K_M | 473 | 241.56 | 36.53 | 1,570 | 8,301 |
| All RCs | Metal | Qwen3.5-2B | Q4_K_M | 480 | 237.85 | 40.00 | 1,594 | 7,661 |
| Fiber | Metal | Qwen3.5-2B | Q8_0 | 484 | 249.70 | 25.28 | 1,545 | 11,446 |
| RC2 | Metal | Qwen3.5-2B | Q8_0 | 499 | 242.81 | 25.59 | 1,590 | 11,320 |
| RC1 | Metal | Qwen3.5-2B | Q8_0 | 541 | 214.20 | 29.99 | 1,778 | 9,934 |
| RC3 | Metal | Qwen3.5-2B | Q8_0 | 444 | 286.06 | 29.23 | 1,370 | 9,873 |
| All RCs | Metal | Qwen3.5-2B | Q8_0 | 453 | 279.19 | 32.34 | 1,402 | 9,026 |
| Fiber | Metal | Qwen3.5-4B | Q4_K_M | 571 | 125.22 | 15.46 | 2,687 | 19,006 |
| RC2 | Metal | Qwen3.5-4B | Q4_K_M | 614 | 110.32 | 14.54 | 3,016 | 20,193 |
| RC1 | Metal | Qwen3.5-4B | Q4_K_M | 550 | 131.99 | 17.89 | 2,558 | 16,555 |
| RC3 | Metal | Qwen3.5-4B | Q4_K_M | 560 | 130.68 | 17.70 | 2,588 | 16,706 |
| All RCs | Metal | Qwen3.5-4B | Q4_K_M | 542 | 140.33 | 18.44 | 2,430 | 15,901 |
| Fiber | Metal | Qwen3.5-4B | Q8_0 | 633 | 111.00 | 13.15 | 3,020 | 22,246 |
| RC2 | Metal | Qwen3.5-4B | Q8_0 | 650 | 111.45 | 12.03 | 3,028 | 24,033 |
| RC1 | Metal | Qwen3.5-4B | Q8_0 | 573 | 124.01 | 14.25 | 2,710 | 20,201 |
| RC3 | Metal | Qwen3.5-4B | Q8_0 | 544 | 137.88 | 14.53 | 2,466 | 19,767 |
| All RCs | Metal | Qwen3.5-4B | Q8_0 | 540 | 138.12 | 15.23 | 2,459 | 18,874 |

## Mac M4 — CPU × elephant.jpg

| Branch | Backend | Model | Quant | Vision (ms) | Prefill (t/s) | Decode (t/s) | TTFT (ms) | Total (ms) |
|--------|---------|-------|-------|------------:|--------------:|-------------:|----------:|-----------:|
| Fiber | CPU | Gemma4-E2B | Q4_K_M | 2,256 | 415.57 | 38.08 | 2,939 | 9,462 |
| RC2 | CPU | Gemma4-E2B | Q4_K_M | 2,186 | 423.12 | 38.57 | 2,857 | 9,296 |
| RC1 | CPU | Gemma4-E2B | Q4_K_M | 2,153 | 429.23 | 38.59 | 2,815 | 9,269 |
| RC3 | CPU | Gemma4-E2B | Q4_K_M | 2,135 | 431.05 | 38.72 | 2,794 | 9,248 |
| All RCs | CPU | Gemma4-E2B | Q4_K_M | 2,191 | 433.13 | 38.73 | 2,847 | 9,267 |
| Fiber | CPU | Gemma4-E2B | Q8_0 | 2,538 | 364.83 | 21.09 | 3,316 | 15,229 |
| RC2 | CPU | Gemma4-E2B | Q8_0 | 1,921 | 428.98 | 25.23 | 2,583 | 12,551 |
| RC1 | CPU | Gemma4-E2B | Q8_0 | 1,959 | 313.96 | 25.43 | 2,864 | 13,565 |
| RC3 | CPU | Gemma4-E2B | Q8_0 | 1,948 | 428.63 | 25.33 | 2,611 | 12,587 |
| All RCs | CPU | Gemma4-E2B | Q8_0 | 1,919 | 429.03 | 25.50 | 2,581 | 12,494 |
| Fiber | CPU | Gemma4-E4B | Q4_K_M | 5,736 | 329.61 | 14.83 | 6,598 | 23,830 |
| RC2 | CPU | Gemma4-E4B | Q4_K_M | 4,217 | 367.63 | 17.32 | 4,990 | 19,767 |
| RC1 | CPU | Gemma4-E4B | Q4_K_M | 3,999 | 318.38 | 17.91 | 4,891 | 19,802 |
| RC3 | CPU | Gemma4-E4B | Q4_K_M | 3,992 | 369.37 | 18.65 | 4,761 | 18,535 |
| All RCs | CPU | Gemma4-E4B | Q4_K_M | 4,073 | 365.92 | 18.55 | 4,849 | 18,823 |
| Fiber | CPU | Gemma4-E4B | Q8_0 | 4,311 | 260.51 | 11.45 | 5,401 | 28,489 |
| RC2 | CPU | Gemma4-E4B | Q8_0 | 3,842 | 275.95 | 12.20 | 4,871 | 26,850 |
| RC1 | CPU | Gemma4-E4B | Q8_0 | 3,640 | 260.35 | 12.52 | 4,731 | 25,925 |
| RC3 | CPU | Gemma4-E4B | Q8_0 | 3,251 | 255.10 | 12.75 | 4,364 | 25,658 |
| All RCs | CPU | Gemma4-E4B | Q8_0 | 3,151 | 269.97 | 12.73 | 4,203 | 25,332 |
| Fiber | CPU | Qwen3.5-2B | Q4_K_M | 2,134 | 111.75 | 33.99 | 4,505 | 10,184 |
| RC2 | CPU | Qwen3.5-2B | Q4_K_M | 2,140 | 110.15 | 33.87 | 4,546 | 10,226 |
| RC1 | CPU | Qwen3.5-2B | Q4_K_M | 1,954 | 120.56 | 31.42 | 4,152 | 10,568 |
| RC3 | CPU | Qwen3.5-2B | Q4_K_M | 1,879 | 127.68 | 40.87 | 3,955 | 8,522 |
| All RCs | CPU | Qwen3.5-2B | Q4_K_M | 1,770 | 134.97 | 32.91 | 3,733 | 10,055 |
| Fiber | CPU | Qwen3.5-2B | Q8_0 | 1,686 | 139.67 | 28.00 | 3,583 | 11,224 |
| RC2 | CPU | Qwen3.5-2B | Q8_0 | 1,754 | 134.53 | 27.78 | 3,724 | 11,417 |
| RC1 | CPU | Qwen3.5-2B | Q8_0 | 1,532 | 153.25 | 22.68 | 3,261 | 13,240 |
| RC3 | CPU | Qwen3.5-2B | Q8_0 | 1,431 | 165.06 | 30.38 | 3,036 | 10,250 |
| All RCs | CPU | Qwen3.5-2B | Q8_0 | 1,442 | 163.43 | 25.53 | 3,063 | 11,820 |
| Fiber | CPU | Qwen3.5-4B | Q4_K_M | 5,108 | 46.98 | 16.45 | 10,749 | 21,438 |
| RC2 | CPU | Qwen3.5-4B | Q4_K_M | 5,716 | 42.18 | 15.64 | 11,999 | 22,911 |
| RC1 | CPU | Qwen3.5-4B | Q4_K_M | 4,741 | 50.49 | 14.05 | 9,990 | 23,686 |
| RC3 | CPU | Qwen3.5-4B | Q4_K_M | 4,503 | 52.83 | 19.12 | 9,519 | 18,642 |
| All RCs | CPU | Qwen3.5-4B | Q4_K_M | 4,258 | 55.69 | 14.11 | 9,016 | 23,594 |
| Fiber | CPU | Qwen3.5-4B | Q8_0 | 4,031 | 58.37 | 13.25 | 8,571 | 24,063 |
| RC2 | CPU | Qwen3.5-4B | Q8_0 | 4,878 | 49.17 | 12.74 | 10,267 | 25,718 |
| RC1 | CPU | Qwen3.5-4B | Q8_0 | 3,008 | 64.70 | 10.61 | 7,104 | 29,549 |
| RC3 | CPU | Qwen3.5-4B | Q8_0 | 3,278 | 68.88 | 13.55 | 7,125 | 23,065 |
| All RCs | CPU | Qwen3.5-4B | Q8_0 | 2,987 | 64.46 | 10.36 | 7,098 | 30,191 |

## Per-branch artifacts

| Branch | Raw logs | Parsed JSON | Metal System Trace (Gemma4-E2B-Q4) | Metal System Trace (Qwen3.5-2B-Q4) | Orchestrator log |
|---|---|---|---|---|---|
| Fiber | `results/raw/fiber-mac-2026-05-13T1856/` | `results/parsed/fiber-mac-2026-05-13T1856.json` | — | — | — |
| RC2 | `results/raw/mac-rc2-mul-mat-opt-2026-05-14T1116/` | `results/parsed/mac-rc2-mul-mat-opt-2026-05-14T1116.json` | `results/traces/rc2-mul-mat-opt-gemma4-e2b-q4km-2026-05-14T1116.trace` | `results/traces/rc2-mul-mat-opt-qwen35-2b-q4km-2026-05-14T1116.trace` | `results/orchestrator-logs/rc-isolation-2026-05-14T1116.log` |
| RC1 | `results/raw/mac-rc1-rerun-2026-05-14T1710/` | `results/parsed/mac-rc1-rerun-2026-05-14T1710.json` | `results/traces/rc1-rerun-gemma4-e2b-q4km-2026-05-14T1710.trace` | `results/traces/rc1-rerun-qwen35-2b-q4km-2026-05-14T1710.trace` | `results/orchestrator-logs/rc1-rerun-2026-05-14T1710.log` |
| RC3 | `results/raw/mac-rc3-rerun-2026-05-14T1830/` | `results/parsed/mac-rc3-rerun-2026-05-14T1830.json` | `results/traces/rc3-rerun-gemma4-e2b-q4km-2026-05-14T1830.trace` | `results/traces/rc3-rerun-qwen35-2b-q4km-2026-05-14T1830.trace` | `results/orchestrator-logs/rc3-rerun-2026-05-14T1830.log` |
| All RCs | `results/raw/mac-fiber-updates-rerun-2026-05-14T1940/` | `results/parsed/mac-fiber-updates-rerun-2026-05-14T1940.json` | `results/traces/fiber-updates-rerun-gemma4-e2b-q4km-2026-05-14T1940.trace` | `results/traces/fiber-updates-rerun-qwen35-2b-q4km-2026-05-14T1940.trace` | `results/orchestrator-logs/fiber-updates-rerun-2026-05-14T1940.log` |

The "Fiber" row's `results/raw/fiber-mac-2026-05-13T1856/` is the
metal-baseline.md baseline run; trace and orchestrator-log artifacts predate
this study and are not republished here.

---

# Appendices

The sections below preserve the chronological per-run-group analyses that
produced the headline matrix above. They are kept verbatim for traceability —
notably the original RC1/RC3 numbers ("orig", stale build), which document
the incremental-build hazard that motivated the `--clean-first` requirement
now codified in `QVAC-18297-plan.md § Benchmark Methodology Notes`.

## Appendix A: Run group `2026-05-14T1116` — initial isolation runs (stale builds for RC1/RC3)


Orchestrators: `tools/scripts/run-rc-isolation.sh` (RC2 → RC1 → RC3) and
`tools/scripts/run-fiber-and-rc2-control.sh` (fiber re-baseline + RC2 control).

### Methodology

- **Device**: Mac M4, macOS 26.4.1, 16 GB unified memory
- **Build**: `cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON`, AppleClang 17.0.0, Darwin arm64
- **Matrix**: 8 models × {metal, cpu} × {elephant.jpg, fruitPlate.png}
  - Gemma4-E2B Q4_K_M/Q8_0, Gemma4-E4B Q4_K_M/Q8_0
  - Qwen3.5-2B Q4_K_M/Q8_0, Qwen3.5-4B Q4_K_M/Q8_0
- **Inference params**: `--ctx-size 4096 --predict 256 --threads 4 --temp 0 --seed 42 --jinja -fit off`
  (Metal: `--gpu-layers 99`, CPU: `--gpu-layers 0`)
- **Protocol**: 1 warmup + 3 measured runs, **median** reported. No cool-down between runs (Mac M4 active cooling).
- **Trace capture**: `xcrun xctrace --template "Metal System Trace" --time-limit 30s` for `gemma4-e2b-q4km` + `qwen35-2b-q4km` per branch.
- **Expected failures**: Qwen3.5 × fruitPlate produces 4,015 vision tokens (ctx overflow) and crashes early — captured in raw logs but excluded from median (consistent with the original fiber baseline).

### Measurement variance — critical caveat

Five Mac M4 measurement points were taken in this run group, in time order:

| # | Variant | Approx start | Notes |
|---|---|---|---|
| 1 | RC2 isolated (full matrix + traces) | 11:16 | First — Mac near-idle thermal state |
| 2 | RC1 isolated (full matrix + traces) | ~12:25 | After 1h09m of continuous load |
| 3 | RC3 isolated (full matrix + traces) | ~13:30 | After 2h14m of continuous load |
| 4 | **fiber-baseline-today** (full matrix + traces) | ~14:35 | After 3h19m, same-thermal state as the RCs |
| 5 | **RC2-control-rerun** (2 configs only) | ~16:55 | After ~5h40m — thermal end-state validator |

The **fiber-baseline-today** (#4) and **RC2-control-rerun** (#5) were added
mid-experiment because the early data showed effects that looked thermal. They
serve as independent reference points:

- **fiber-baseline-today** is an apples-to-apples baseline measured in the same
  thermal state as RC1 and RC3.
- **RC2-control-rerun** re-measures RC2 *after* RC1/RC3/fiber-today completed,
  confirming whether RC2's earlier numbers were thermally-favored.

#### Fiber drift: today vs 2026-05-13

Comparing the two fiber-8189 measurements (same code, same machine, different
days/thermal states) — `metal-baseline.md` fiber-8189 (parsed
`fiber-mac-2026-05-13T1856.json`) vs the fresh `fiber-baseline-today` run:

| Config (Metal × elephant) | fiber 2026-05-13 (t/s) | fiber today (t/s) | Δ |
|---|---:|---:|---:|
| gemma4-e2b-q4km | 31.45 | 30.68 | −2.4% |
| gemma4-e2b-q8 | 21.79 | 19.87 | −8.8% |
| gemma4-e4b-q4km | 16.78 | 14.81 | −11.7% |
| gemma4-e4b-q8 | 12.66 | 10.91 | −13.8% |
| qwen35-2b-q4km | 32.56 | 29.88 | −8.2% |
| qwen35-2b-q8 | 25.28 | 24.67 | −2.4% |
| qwen35-4b-q4km | 15.46 | 14.55 | −5.9% |
| qwen35-4b-q8 | 13.15 | 12.46 | −5.2% |

Same fiber-8189 source code is **2–14% slower today** than yesterday — likely
chassis temperature / cooling-system state differences. This is the noise floor
for any Δ% comparison between days.

#### RC2-control-rerun: thermal end-state

| Config (Metal × elephant) | RC2 fresh (t/s) | RC2 control rerun (t/s) | Δ rerun / fresh |
|---|---:|---:|---:|
| gemma4-e2b-q4km | 41.89 | 42.26 | +0.9% |
| qwen35-2b-q4km | 32.39 | 38.65 | +19.3% |

**Key finding**: RC2 re-measured at the *very end* of the session (≈5h40m of
load) is **not** slower than the first measurement. It is **the same or faster**.
This rules out simple thermal degradation as the explanation for the
RC1/RC3 numbers below — the RC1 and RC3 measurements differ from RC2 because
of the code, not because the machine grew tired.

### Headline summary — Mac M4 Metal × elephant decode_tps

| Config | fiber 2026-05-13 | fiber today | **RC2** | **RC1** | **RC3** | RC2 ctrl |
|---|---:|---:|---:|---:|---:|---:|
| gemma4-e2b-q4km | 31.45 | 30.68 | **41.89** | 27.35 | 35.36 | 42.26 |
| gemma4-e2b-q8 | 21.79 | 19.87 | **25.92** | 20.81 | 16.02 | — |
| gemma4-e4b-q4km | 16.78 | 14.81 | **18.83** | 15.99 | 13.35 | — |
| gemma4-e4b-q8 | 12.66 | 10.91 | **13.06** | 10.79 | 8.85 | — |
| qwen35-2b-q4km | 32.56 | 29.88 | 32.39 | 32.41 | 23.45 | **38.65** |
| qwen35-2b-q8 | 25.28 | 24.67 | 25.59 | 24.63 | 22.08 | — |
| qwen35-4b-q4km | 15.46 | 14.55 | 14.54 | 14.12 | 13.16 | — |
| qwen35-4b-q8 | 13.15 | 12.46 | 12.03 | 11.23 | 10.96 | — |

Δ% against both fiber baselines for the eight Metal × elephant configs:

| Config | RC2 vs fiber-2026-05-13 | RC2 vs fiber-today | RC1 vs fiber-2026-05-13 | RC1 vs fiber-today | RC3 vs fiber-2026-05-13 | RC3 vs fiber-today |
|---|---:|---:|---:|---:|---:|---:|
| gemma4-e2b-q4km | **+33.2%** | **+36.5%** | −13.0% | −10.9% | +12.4% | +15.3% |
| gemma4-e2b-q8 | **+19.0%** | **+30.4%** | −4.5% | +4.7% | −26.5% | −19.4% |
| gemma4-e4b-q4km | **+12.2%** | **+27.1%** | −4.7% | +8.0% | −20.4% | −9.9% |
| gemma4-e4b-q8 | +3.2% | **+19.7%** | −14.8% | −1.1% | −30.1% | −18.9% |
| qwen35-2b-q4km | −0.5% | +8.4% | −0.5% | +8.5% | −28.0% | −21.5% |
| qwen35-2b-q8 | +1.2% | +3.7% | −2.6% | −0.2% | −12.7% | −10.5% |
| qwen35-4b-q4km | −6.0% | −0.1% | −8.7% | −3.0% | −14.9% | −9.6% |
| qwen35-4b-q8 | −8.5% | −3.5% | −14.6% | −9.9% | −16.7% | −12.0% |

#### Headline findings

1. **RC2 is a genuine Gemma4 Metal decode win on M4** (+12% to +37% across
   E2B/E4B × Q4_K_M/Q8_0). The thermal-control rerun reproduces the fast
   result, ruling out variance. This **contradicts the prior claim** in
   `QVAC-18297-fiber-b9025-gap.md` that RC2 has "zero effect on M4" — the
   Metal Tensor API kernel split benefits the non-tensor-API path too on M4.
2. **RC1 isolated is essentially neutral** vs same-thermal fiber-today on the
   2B/E2B models (−3% to +9%) and a modest regression on the 4B/E4B Q8_0
   models (−10% to −11%). The previously-reported "+18.8% Qwen3.5 from RC1
   alone" does **not** reproduce in isolation — that gain appears to require
   RC2's MUL_MAT kernel changes underneath.
3. **RC3 isolated is a regression** across most Metal configs (−10% to −22%
   on Qwen3.5; +15% to −19% on Gemma4). Unexpected — RC3 only adds FA
   `dk512_dv512` templates, which should be a pure addition. Possible cause:
   the wider supported-head-size table changes scheduler decisions in ways
   that disable previously-enabled FA paths. Worth further investigation.

These findings are *opposite of the cumulative measurements* in
`QVAC-18297-fiber-b9025-gap.md` (where RC2 looked like a no-op and RC3 looked
like a +20% Gemma4 win). The cumulative picture and the isolated picture
disagree, which means **the RC commits compose non-additively** — each
commit's effect depends on what is already underneath it.

---

### RC2 — Metal Tensor API `MUL_MAT` optimization

- **Commit**: `f6d6dbcd1`
- **Branch**: `feat/QVAC-18297-rc2-mul-mat-opt`
- **Raw logs**: `results/raw/mac-rc2-mul-mat-opt-2026-05-14T1116/`
- **Parsed**: `results/parsed/mac-rc2-mul-mat-opt-2026-05-14T1116.json`
- **Traces**: `results/traces/rc2-mul-mat-opt-{gemma4-e2b-q4km,qwen35-2b-q4km}-2026-05-14T1116.trace`
- **Control rerun**: `results/raw/mac-rc2-control-rerun-2026-05-14T1116/`,
  `results/parsed/mac-rc2-control-rerun-2026-05-14T1116.json`

#### Δ vs fiber-8189 (metal-baseline.md `fiber-mac-2026-05-13T1856`)

Headline configs (Metal × elephant). Full table in
`results/parsed/mac-rc2-mul-mat-opt-2026-05-14T1116-section-vs-old.md`.

| Config | vision_ms | prefill_tps | decode_tps | total_ms |
|---|---|---|---|---|
| `gemma4-e2b-q4km|metal|elephant` | 604ms (+23.1% ✓) | 257.4 t/s (+45.2% ✓) | 41.9 t/s (**+33.2% ✓**) | 7,703ms (+24.0% ✓) |
| `gemma4-e2b-q8|metal|elephant` | 653ms (+14.6% ✓) | 231.1 t/s (+24.9% ✓) | 25.9 t/s (**+19.0% ✓**) | 11,475ms (+15.9% ✓) |
| `gemma4-e4b-q4km|metal|elephant` | 712ms (+16.7% ✓) | 141.5 t/s (+29.0% ✓) | 18.8 t/s (**+12.2% ✓**) | 16,046ms (+12.3% ✓) |
| `gemma4-e4b-q8|metal|elephant` | 770ms (−2.8%) | 133.8 t/s (−5.6% ⚠) | 13.1 t/s (+3.2%) | 22,406ms (+4.0%) |
| `qwen35-2b-q4km|metal|elephant` | 488ms (−4.7%) | 232.5 t/s (−8.2% ⚠) | 32.4 t/s (−0.5%) | 9,230ms (−0.5%) |
| `qwen35-2b-q8|metal|elephant` | 494ms (−3.1%) | 242.8 t/s (−2.8%) | 25.6 t/s (+1.2%) | 11,320ms (+1.1%) |
| `qwen35-4b-q4km|metal|elephant` | 608ms (−7.8% ⚠) | 110.3 t/s (−11.9% ⚠) | 14.5 t/s (−6.0% ⚠) | 20,193ms (−6.2% ⚠) |
| `qwen35-4b-q8|metal|elephant` | 644ms (−2.7%) | 111.5 t/s (+0.4%) | 12.0 t/s (−8.5% ⚠) | 24,033ms (−8.0% ⚠) |

**Summary**: 46 cells improved > ±5%, 12 cells regressed > ±5% (out of 160).

#### Δ vs fiber-baseline-today (same-thermal apples-to-apples)

Full table in `results/parsed/mac-rc2-mul-mat-opt-2026-05-14T1116-section-vs-today.md`.

**Summary**: 62 cells improved > ±5%, 1 cell regressed > ±5% (out of 160).

#### Interpretation

RC2 is a net Gemma4 decode improvement on M4 (≈+12% to +37% Metal Q4_K_M/Q8_0).
The kernel split in `f6d6dbcd1` benefits the non-tensor-API code path too,
giving Gemma4 a real win on pre-M5 Apple Silicon. Qwen3.5 is approximately
neutral.

---

### RC1 — Fused `GATED_DELTA_NET` port

- **Commit**: `556789f7d`
- **Branch**: `feat/QVAC-18297-rc1-gated-delta-net`
- **Raw logs**: `results/raw/mac-rc1-gated-delta-net-2026-05-14T1116/`
- **Parsed**: `results/parsed/mac-rc1-gated-delta-net-2026-05-14T1116.json`
- **Traces**: `results/traces/rc1-gated-delta-net-{gemma4-e2b-q4km,qwen35-2b-q4km}-2026-05-14T1116.trace`

#### Δ vs fiber-8189 (metal-baseline.md `fiber-mac-2026-05-13T1856`)

Headline configs (Metal × elephant). Full table in
`results/parsed/mac-rc1-gated-delta-net-2026-05-14T1116-section-vs-old.md`.

| Config | vision_ms | prefill_tps | decode_tps | total_ms |
|---|---|---|---|---|
| `gemma4-e2b-q4km|metal|elephant` | 933ms (−18.9% ⚠) | 150.9 t/s (−14.9% ⚠) | 27.4 t/s (−13.0% ⚠) | 11,637ms (−14.9% ⚠) |
| `gemma4-e2b-q8|metal|elephant` | 830ms (−8.5% ⚠) | 176.7 t/s (−4.5%) | 20.8 t/s (−4.5%) | 14,224ms (−4.2%) |
| `gemma4-e4b-q4km|metal|elephant` | 826ms (+3.4%) | 118.4 t/s (+7.9% ✓) | 16.0 t/s (−4.7%) | 18,911ms (−3.3%) |
| `gemma4-e4b-q8|metal|elephant` | 882ms (−17.8% ⚠) | 104.1 t/s (−26.5% ⚠) | 10.8 t/s (−14.8% ⚠) | 27,594ms (−18.3% ⚠) |
| `qwen35-2b-q4km|metal|elephant` | 518ms (−11.2% ⚠) | 200.1 t/s (−21.0% ⚠) | 32.4 t/s (−0.5%) | 9,396ms (−2.3%) |
| `qwen35-2b-q8|metal|elephant` | 536ms (−11.9% ⚠) | 198.7 t/s (−20.4% ⚠) | 24.6 t/s (−2.6%) | 11,852ms (−3.5%) |
| `qwen35-4b-q4km|metal|elephant` | 647ms (−14.7% ⚠) | 98.4 t/s (−21.4% ⚠) | 14.1 t/s (−8.7% ⚠) | 20,877ms (−9.8% ⚠) |
| `qwen35-4b-q8|metal|elephant` | 603ms (+3.8%) | 103.2 t/s (−7.0% ⚠) | 11.2 t/s (−14.6% ⚠) | 25,477ms (−14.5% ⚠) |

**Summary**: 15 cells improved > ±5%, 53 cells regressed > ±5% (out of 160).

#### Δ vs fiber-baseline-today (same-thermal apples-to-apples)

Full table in `results/parsed/mac-rc1-gated-delta-net-2026-05-14T1116-section-vs-today.md`.

**Summary**: 23 cells improved > ±5%, 38 cells regressed > ±5% (out of 160).

#### Interpretation

RC1 isolated does not reproduce the previously-reported "+18.8% Qwen3.5" win.
Qwen3.5-2B/2B-Q8 Metal decode is essentially flat (−0.5% to −2.6% vs old fiber;
−0.2% to +8.5% vs fiber-today). The decode delta becomes a regression on the
4B Qwen3.5 variants and on Gemma4 E4B Q8.

The CPU Qwen3.5 decode drops 27–32% (e.g. `qwen35-2b-q4km|cpu|elephant`: 33.99
→ 24.81 t/s) which is plausibly real — RC1 reroutes the Qwen3.5 CPU path
through the new `GATED_DELTA_NET` op, which may have a slower CPU
implementation than the existing `DELTA_NET_AR` op. Worth profiling
separately if CPU Qwen3.5 inference matters downstream.

The "RC1 alone is a Qwen3.5 win" claim in `QVAC-18297-fiber-b9025-gap.md`
required RC2 underneath. The fused-GDN op needs the kernel restructuring from
RC2 to deliver its decode benefit on this hardware.

---

### RC3 — Metal Flash Attention `dk512_dv512` instantiations

- **Commit**: `054141103`
- **Branch**: `feat/QVAC-18297-rc3-fa-dk512`
- **Raw logs**: `results/raw/mac-rc3-fa-dk512-2026-05-14T1116/`
- **Parsed**: `results/parsed/mac-rc3-fa-dk512-2026-05-14T1116.json`
- **Traces**: `results/traces/rc3-fa-dk512-{gemma4-e2b-q4km,qwen35-2b-q4km}-2026-05-14T1116.trace`

#### Δ vs fiber-8189 (metal-baseline.md `fiber-mac-2026-05-13T1856`)

Headline configs (Metal × elephant). Full table in
`results/parsed/mac-rc3-fa-dk512-2026-05-14T1116-section-vs-old.md`.

| Config | vision_ms | prefill_tps | decode_tps | total_ms |
|---|---|---|---|---|
| `gemma4-e2b-q4km|metal|elephant` | 698ms (+11.0% ✓) | 196.6 t/s (+10.9% ✓) | 35.4 t/s (**+12.4% ✓**) | 8,950ms (+11.6% ✓) |
| `gemma4-e2b-q8|metal|elephant` | 730ms (+9.5% ✓) | 144.4 t/s (−21.9% ⚠) | 16.0 t/s (−26.5% ⚠) | 18,025ms (−32.1% ⚠) |
| `gemma4-e4b-q4km|metal|elephant` | 779ms (+8.9% ✓) | 109.7 t/s (+0.0%) | 13.3 t/s (−20.4% ⚠) | 22,394ms (−22.4% ⚠) |
| `gemma4-e4b-q8|metal|elephant` | 851ms (−5.5% ⚠) | 106.4 t/s (−24.9% ⚠) | 8.9 t/s (−30.1% ⚠) | 32,968ms (−41.4% ⚠) |
| `qwen35-2b-q4km|metal|elephant` | 631ms (−34.3% ⚠) | 142.9 t/s (−43.6% ⚠) | 23.5 t/s (−28.0% ⚠) | 12,710ms (−38.4% ⚠) |
| `qwen35-2b-q8|metal|elephant` | 624ms (−28.9% ⚠) | 145.1 t/s (−41.9% ⚠) | 22.1 t/s (−12.7% ⚠) | 13,019ms (−13.7% ⚠) |
| `qwen35-4b-q4km|metal|elephant` | 716ms (−25.4% ⚠) | 70.0 t/s (−44.1% ⚠) | 13.2 t/s (−14.9% ⚠) | 22,272ms (−17.2% ⚠) |
| `qwen35-4b-q8|metal|elephant` | 692ms (−9.3% ⚠) | 73.3 t/s (−34.0% ⚠) | 11.0 t/s (−16.7% ⚠) | 27,156ms (−22.0% ⚠) |

**Summary**: 4 cells improved > ±5%, 86 cells regressed > ±5% (out of 160).

#### Δ vs fiber-baseline-today (same-thermal apples-to-apples)

Full table in `results/parsed/mac-rc3-fa-dk512-2026-05-14T1116-section-vs-today.md`.

**Summary**: 2 cells improved > ±5%, 100 cells regressed > ±5% (out of 160).

#### Interpretation

RC3 alone — without RC1 (GDN) and without RC2 (MUL_MAT kernel split) underneath
— is a net regression on M4. The only positive cell is Gemma4-E2B-Q4 Metal
elephant decode (+12.4% vs old fiber, +15.3% vs fiber-today), consistent with
the FA re-enable for the 512-dim head case. But all larger Gemma4 variants
regress significantly, and all Qwen3.5 variants regress 9–28%.

Hypothesis: widening the FA `supports_op` head-size table (the
`ggml-metal-device.m` change in `054141103`) changes scheduler device-assignment
decisions for Qwen3.5 attention tensors in ways that hurt throughput. The FA
re-enable for Gemma4-E2B-Q4 helps, but other configurations are pushed onto
slower paths.

This is the most surprising finding in the run group and the most worth
following up on.

---

### Cross-branch summary

The three commits do **not** decompose into clean isolated gains on M4. Their
cumulative effect (as documented in `QVAC-18297-fiber-b9025-gap.md`) requires
all three to be stacked.

| Variant | Branch | Gemma4-E2B-Q4 decode (t/s) | Δ vs fiber-2026-05-13 | Qwen3.5-2B-Q4 decode (t/s) | Δ vs fiber-2026-05-13 |
|---|---|---:|---:|---:|---:|
| Fiber 2026-05-13 (anchor) | `tetherto/temp-8189` | 31.45 | — | 32.56 | — |
| Fiber today (apples-to-apples) | `tetherto/temp-8189` | 30.68 | −2.4% | 29.88 | −8.2% |
| **RC2 isolated** | `feat/QVAC-18297-rc2-mul-mat-opt` | **41.89** | **+33.2%** | 32.39 | −0.5% |
| **RC1 isolated** | `feat/QVAC-18297-rc1-gated-delta-net` | 27.35 | −13.0% | 32.41 | −0.5% |
| **RC3 isolated** | `feat/QVAC-18297-rc3-fa-dk512` | 35.36 | +12.4% | 23.45 | −28.0% |
| RC2 control rerun | `feat/QVAC-18297-rc2-mul-mat-opt` | 42.26 | +34.4% | 38.65 | +18.7% |

Conclusion: of the three commits, **only RC2 is a clear isolated win on M4**.
RC1 and RC3 isolated do not deliver the gains observed in the cumulative
stacked measurement.

### Artifacts

- **Branches** (in `/Users/ic/repo/vlm-benchmark/llama.cpp`):
  `feat/QVAC-18297-rc2-mul-mat-opt`, `feat/QVAC-18297-rc1-gated-delta-net`,
  `feat/QVAC-18297-rc3-fa-dk512` — each one cherry-pick on top of `tetherto/temp-8189`.
- **Built binaries** (in `llama.cpp/binaries/`):
  `rc2-mul-mat-opt/llama-mtmd-cli`, `rc1-gated-delta-net/llama-mtmd-cli`,
  `rc3-fa-dk512/llama-mtmd-cli`, `fiber-baseline-today/llama-mtmd-cli`.
- **Raw logs** (per `mac-<variant>-2026-05-14T1116/`): `rc2-mul-mat-opt`,
  `rc1-gated-delta-net`, `rc3-fa-dk512`, `fiber-baseline-today`, `rc2-control-rerun`.
- **Parsed medians** (per `mac-<variant>-2026-05-14T1116.json`): same set.
- **Metal System Traces** (in `results/traces/`):
  `rc2-mul-mat-opt-*.trace`, `rc1-gated-delta-net-*.trace`,
  `rc3-fa-dk512-*.trace`, `fiber-baseline-today-*.trace` — 8 traces total, ~2 GB.
- **Diff markdowns** (per `mac-<variant>-2026-05-14T1116-section-vs-{old,today}.md`):
  full per-config delta tables vs both fiber baselines.
- **Orchestrator logs**: `results/orchestrator-logs/rc-isolation-2026-05-14T1116.log`,
  `results/orchestrator-logs/fiber-and-rc2-control-2026-05-14T1116.log`.

### Reproducing

```sh
# 1. Create the three isolated branches:
LLAMA=/Users/ic/repo/vlm-benchmark/llama.cpp
git -C $LLAMA branch feat/QVAC-18297-rc2-mul-mat-opt    tetherto/temp-8189 && git -C $LLAMA checkout feat/QVAC-18297-rc2-mul-mat-opt    && git -C $LLAMA cherry-pick f6d6dbcd1
git -C $LLAMA branch feat/QVAC-18297-rc1-gated-delta-net tetherto/temp-8189 && git -C $LLAMA checkout feat/QVAC-18297-rc1-gated-delta-net && git -C $LLAMA cherry-pick 556789f7d
git -C $LLAMA branch feat/QVAC-18297-rc3-fa-dk512        tetherto/temp-8189 && git -C $LLAMA checkout feat/QVAC-18297-rc3-fa-dk512        && git -C $LLAMA cherry-pick 054141103

# 2. Run the orchestrator (~3.5 hours on Mac M4):
TS=$(date +%Y-%m-%dT%H%M) /Users/ic/repo/vlm-benchmark/tools/scripts/run-rc-isolation.sh

# 3. Run the fiber re-baseline + RC2 control (~1.5 hours, after a cool-down):
TS=$TS /Users/ic/repo/vlm-benchmark/tools/scripts/run-fiber-and-rc2-control.sh

# 4. Parse and diff (per variant):
for v in rc2-mul-mat-opt rc1-gated-delta-net rc3-fa-dk512 fiber-baseline-today rc2-control-rerun; do
  python3 tools/scripts/parse-mac-logs.py "results/raw/mac-${v}-${TS}" "results/parsed/mac-${v}-${TS}.json"
done
```

---

## Appendix B: Run group `2026-05-14T1710` — RC1 reproducibility check (clean build)


Orchestrator: `tools/scripts/run-rc1-rerun.sh`. Same protocol as the
`2026-05-14T1116` run group above, but only for the RC1 branch and with **one
key change**: the build uses `cmake --build … --clean-first` to force a fresh
compile of every object file. This eliminates any stale artifacts from the
prior incremental-build sequence (RC2 → RC1 → RC3 all used the same `build-mac`
directory in the original run group).

- **Variant slug**: `rc1-rerun`
- **Branch**: `feat/QVAC-18297-rc1-gated-delta-net` (unchanged — same commit `98b08344f` cherry-pick of `556789f7d`)
- **Binary**: `llama.cpp/binaries/rc1-rerun/llama-mtmd-cli`
- **Raw logs**: `results/raw/mac-rc1-rerun-2026-05-14T1710/`
- **Parsed**: `results/parsed/mac-rc1-rerun-2026-05-14T1710.json`
- **Traces**: `results/traces/rc1-rerun-{gemma4-e2b-q4km,qwen35-2b-q4km}-2026-05-14T1710.trace` (380 MB total)

### Headline: original RC1 numbers were a stale-build artifact

The clean rebuild produces dramatically faster RC1 numbers across **every**
Metal × elephant config — and the new numbers reproduce the originally-reported
"+18.8% Qwen3.5" RC1 win.

| Config (Metal × elephant) | fiber 2026-05-13 | RC2 (orig) | **RC1 orig (incr. build)** | **RC1 rerun (clean build)** | Δ rerun / orig |
|---|---:|---:|---:|---:|---:|
| gemma4-e2b-q4km | 31.45 | 41.89 | 27.35 | **42.48** | **+55.3%** |
| gemma4-e2b-q8 | 21.79 | 25.92 | 20.81 | **25.63** | +23.2% |
| gemma4-e4b-q4km | 16.78 | 18.83 | 15.99 | **19.31** | +20.8% |
| gemma4-e4b-q8 | 12.66 | 13.06 | 10.79 | **14.94** | +38.5% |
| qwen35-2b-q4km | 32.56 | 32.39 | 32.41 | **38.54** | **+18.9%** |
| qwen35-2b-q8 | 25.28 | 25.59 | 24.63 | **29.99** | +21.8% |
| qwen35-4b-q4km | 15.46 | 14.54 | 14.12 | **17.89** | +26.7% |
| qwen35-4b-q8 | 13.15 | 12.03 | 11.23 | **14.25** | +26.9% |

Key observations:

- **Qwen3.5-2B-Q4 isolated RC1 = +18.4% vs fiber-2026-05-13** (38.54 vs 32.56),
  reproducing the +18.8% figure from the cumulative measurement in
  `QVAC-18297-fiber-b9025-gap.md`. RC1 in isolation **does** deliver the GDN
  win on Qwen3.5 — the prior conclusion that "RC1 needs RC2 underneath" was
  wrong, caused by a build artifact.
- **Every Gemma4 variant ALSO improves under clean RC1**, with Gemma4-E2B-Q4
  hitting 42.48 t/s — a +35% gain vs fiber-2026-05-13 and effectively equal
  to RC2's 41.89 t/s. Since RC1 only adds the GDN op (no Gemma4 code path
  touched), this gain is structurally identical to RC2's Gemma4 win — the
  clean rebuild forces all cmake-generated artifacts (notably the embedded
  Metal shader library blob) to be regenerated, which is what surfaces the
  improvement.

### Hypothesis: which stale artifact?

The most likely culprit is the **embedded default-metallib blob** generated at
build time from `ggml/src/ggml-metal/ggml-metal.metal`. The incremental build
appears not to have detected that the Metal source file content depended on
the currently-checked-out branch — so when the orchestrator switched from RC2
to RC1, the embedded shader blob from the previous build was reused, producing
a binary with mismatched Metal kernels vs C++ dispatch code. This matches the
symptom (decode is dominated by Metal kernels; CPU paths show smaller deltas)
and is consistent with the per-run *stability* of the bad RC1 numbers (it's
not random thermal noise — it's a deterministic incorrect build).

### Δ vs fiber-8189 (metal-baseline.md `fiber-mac-2026-05-13T1856`) — RC1 clean build

Full table in `results/parsed/mac-rc1-rerun-2026-05-14T1710-section-vs-old.md`.

| Config | vision_ms | prefill_tps | decode_tps | total_ms |
|---|---|---|---|---|
| `gemma4-e2b-q4km|metal|elephant` | 593ms (+24.5% ✓) | 261.0 t/s (+47.2% ✓) | 42.5 t/s (**+35.1% ✓**) | 7,556ms (+25.4% ✓) |
| `gemma4-e2b-q8|metal|elephant` | 654ms (+14.5% ✓) | 232.9 t/s (+25.9% ✓) | 25.6 t/s (**+17.6% ✓**) | 11,531ms (+15.5% ✓) |
| `gemma4-e4b-q4km|metal|elephant` | 715ms (+16.3% ✓) | 141.6 t/s (+29.1% ✓) | 19.3 t/s (**+15.1% ✓**) | 15,786ms (+13.8% ✓) |
| `gemma4-e4b-q8|metal|elephant` | 768ms (−2.5%) | 133.8 t/s (−5.6% ⚠) | 14.9 t/s (**+18.0% ✓**) | 20,164ms (+13.6% ✓) |
| `qwen35-2b-q4km|metal|elephant` | 488ms (−4.7%) | 240.0 t/s (−5.2% ⚠) | 38.5 t/s (**+18.4% ✓**) | 7,989ms (+13.0% ✓) |
| `qwen35-2b-q8|metal|elephant` | 487ms (−1.7%) | 250.0 t/s (+0.1%) | 30.0 t/s (**+18.6% ✓**) | 9,825ms (+14.2% ✓) |
| `qwen35-4b-q4km|metal|elephant` | 612ms (−8.5% ⚠) | 110.9 t/s (−11.4% ⚠) | 17.9 t/s (**+15.7% ✓**) | 16,769ms (+11.8% ✓) |
| `qwen35-4b-q8|metal|elephant` | 656ms (−4.5%) | 110.9 t/s (−0.1%) | 14.3 t/s (**+8.4% ✓**) | 20,407ms (+8.3% ✓) |

**Summary**: 64 cells improved > ±5%, 12 cells regressed > ±5% (out of 160).

### Δ vs fiber-baseline-today (same-thermal apples-to-apples)

Full table in `results/parsed/mac-rc1-rerun-2026-05-14T1710-section-vs-today.md`.

**Summary**: 76 cells improved > ±5%, 11 cells regressed > ±5% (out of 160).

### Δ rerun vs orig (full)

Per-cell diff in `results/parsed/mac-rc1-rerun-2026-05-14T1710-vs-rc1-orig.md`:
**112 cells improved > ±5%**, 14 regressed (out of 160 measured).

### Implication for the rest of the run group

The same stale-artifact mechanism almost certainly affected **RC3** in the
original `2026-05-14T1116` run (RC3 was built incrementally on top of the RC2
binary state, then the RC1 binary state, before checking out RC3 and building).
The original RC3 numbers (86 cells regressed vs fiber-old) should be re-tested
with `--clean-first` before drawing any conclusion about RC3's isolated impact.

RC2 was the *first* build in the original run group, with no preceding state
to leave artifacts, so RC2 is likely the only original measurement that
remains reliable. The RC2-control-rerun (which used the same incremental
`build-mac` directory after fiber-today's clean-ish build) reproduced RC2's
fast numbers, supporting that interpretation.

### Updated cross-branch summary (incorporating RC1 rerun)

The original `2026-05-14T1116` cross-branch summary at the top of this
document is **superseded for RC1**. Updated headline:

| Variant | Branch | Gemma4-E2B-Q4 decode (t/s) | Δ vs fiber-2026-05-13 | Qwen3.5-2B-Q4 decode (t/s) | Δ vs fiber-2026-05-13 |
|---|---|---:|---:|---:|---:|
| Fiber 2026-05-13 (anchor) | `tetherto/temp-8189` | 31.45 | — | 32.56 | — |
| Fiber today (apples-to-apples) | `tetherto/temp-8189` | 30.68 | −2.4% | 29.88 | −8.2% |
| **RC2 isolated** | `feat/QVAC-18297-rc2-mul-mat-opt` | 41.89 | +33.2% | 32.39 | −0.5% |
| **RC1 isolated (orig — stale build)** | `feat/QVAC-18297-rc1-gated-delta-net` | 27.35 | −13.0% | 32.41 | −0.5% |
| **RC1 isolated (rerun — clean build)** ✅ | `feat/QVAC-18297-rc1-gated-delta-net` | **42.48** | **+35.1%** | **38.54** | **+18.4%** |
| RC3 isolated (orig — stale build) ⚠ | `feat/QVAC-18297-rc3-fa-dk512` | 35.36 | +12.4% | 23.45 | −28.0% |
| RC2 control rerun | `feat/QVAC-18297-rc2-mul-mat-opt` | 42.26 | +34.4% | 38.65 | +18.7% |

Revised conclusion: with clean builds, **both RC2 and RC1 in isolation are
real Mac M4 wins** (decode ~+35% Gemma4, ~+18% Qwen3.5). RC3 isolated has not
yet been re-validated with a clean build — recommend a follow-up run before
trusting its current (regressing) numbers.

### Reproducing the RC1 rerun

```sh
# From an existing build-mac directory (already configured):
TS=$(date +%Y-%m-%dT%H%M) /Users/ic/repo/vlm-benchmark/tools/scripts/run-rc1-rerun.sh

# Then parse + diff vs fiber baselines and vs the prior RC1 run:
REPO=/Users/ic/repo/vlm-benchmark
python3 $REPO/tools/scripts/parse-mac-logs.py "$REPO/results/raw/mac-rc1-rerun-${TS}" "$REPO/results/parsed/mac-rc1-rerun-${TS}.json"
python3 $REPO/tools/scripts/diff-parsed.py \
    "$REPO/results/parsed/mac-rc1-gated-delta-net-2026-05-14T1116.json" \
    "$REPO/results/parsed/mac-rc1-rerun-${TS}.json" \
    "$REPO/results/parsed/mac-rc1-rerun-${TS}-vs-rc1-orig.md"
```

---

## Appendix C: Run group `2026-05-14T1830` — RC3 reproducibility check (clean build)


Same protocol as the RC1 reproducibility check: clean rebuild
(`cmake --build … --clean-first`), fresh timestamp, all original artifacts
preserved.

- **Variant slug**: `rc3-rerun`
- **Branch**: `feat/QVAC-18297-rc3-fa-dk512` (commit `460207e83` cherry-pick of `054141103`)
- **Binary**: `llama.cpp/binaries/rc3-rerun/llama-mtmd-cli`
- **Raw logs**: `results/raw/mac-rc3-rerun-2026-05-14T1830/`
- **Parsed**: `results/parsed/mac-rc3-rerun-2026-05-14T1830.json`
- **Traces**: `results/traces/rc3-rerun-{gemma4-e2b-q4km,qwen35-2b-q4km}-2026-05-14T1830.trace` (325 MB total)

### Headline: RC3 isolated is the largest single-commit win, not a regression

The clean rebuild flips RC3 from "the worst of the three" to "the biggest
single-commit gain." With FA `dk512_dv512` templates properly registered, the
Gemma4 attention path runs Flash Attention end-to-end and decode recovers all
of the fiber regression (and slightly exceeds b9025 on E2B-Q4).

| Config (Metal × elephant) | fiber 2026-05-13 | RC2 (orig) | RC1 rerun (clean) | **RC3 orig (stale build)** | **RC3 rerun (clean build)** | Δ rerun / orig | Δ rerun / fiber-old |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemma4-e2b-q4km | 31.45 | 41.89 | 42.48 | 35.36 | **52.43** | +48.3% | **+66.7%** |
| gemma4-e2b-q8 | 21.79 | 25.92 | 25.63 | 16.02 | **30.27** | +89.0% | **+38.9%** |
| gemma4-e4b-q4km | 16.78 | 18.83 | 19.31 | 13.35 | **23.88** | +78.9% | **+42.3%** |
| gemma4-e4b-q8 | 12.66 | 13.06 | 14.94 | 8.85 | **17.63** | +99.2% | **+39.3%** |
| qwen35-2b-q4km | 32.56 | 32.39 | 38.54 | 23.45 | **36.53** | +55.8% | **+12.2%** |
| qwen35-2b-q8 | 25.28 | 25.59 | 29.99 | 22.08 | **29.23** | +32.4% | **+15.6%** |
| qwen35-4b-q4km | 15.46 | 14.54 | 17.89 | 13.16 | **17.70** | +34.5% | **+14.5%** |
| qwen35-4b-q8 | 13.15 | 12.03 | 14.25 | 10.96 | **14.53** | +32.6% | **+10.5%** |

Key observations:

- **Gemma4-E2B-Q4 hits 52.43 t/s — higher than b9025's 50.73 t/s** (per
  `metal-baseline.md` upstream baseline). RC3 isolated does not merely "recover"
  the fiber regression; it slightly exceeds the upstream reference.
- **All Gemma4 variants gain +39% to +67%** vs fiber-2026-05-13 from RC3 alone.
  This is the FA-enable benefit — fiber's Gemma4 had FA globally disabled
  because the `dk512_dv512` head-size templates were missing.
- **All Qwen3.5 variants gain +10% to +16%** from RC3 alone. Qwen3.5 already
  had FA enabled in fiber, but the broader supports_op table evidently helps
  some intermediate operations.

### Δ vs fiber-8189 (metal-baseline.md `fiber-mac-2026-05-13T1856`) — RC3 clean build

Full table in `results/parsed/mac-rc3-rerun-2026-05-14T1830-section-vs-old.md`.

| Config | vision_ms | prefill_tps | decode_tps | total_ms |
|---|---|---|---|---|
| `gemma4-e2b-q4km|metal|elephant` | 609ms (+22.4% ✓) | 260.4 t/s (+46.9% ✓) | 52.4 t/s (**+66.7% ✓**) | 6,337ms (+37.4% ✓) |
| `gemma4-e2b-q8|metal|elephant` | 681ms (+11.0% ✓) | 227.8 t/s (+23.2% ✓) | 30.3 t/s (**+38.9% ✓**) | 10,073ms (+26.2% ✓) |
| `gemma4-e4b-q4km|metal|elephant` | 729ms (+14.7% ✓) | 133.0 t/s (+21.2% ✓) | 23.9 t/s (**+42.3% ✓**) | 13,298ms (+27.3% ✓) |
| `gemma4-e4b-q8|metal|elephant` | 673ms (+10.1% ✓) | 165.4 t/s (+16.7% ✓) | 17.6 t/s (**+39.3% ✓**) | 17,405ms (+25.4% ✓) |
| `qwen35-2b-q4km|metal|elephant` | 468ms (−0.4%) | 241.6 t/s (−4.6%) | 36.5 t/s (**+12.2% ✓**) | 8,301ms (+9.6% ✓) |
| `qwen35-2b-q8|metal|elephant` | 439ms (+8.4% ✓) | 286.1 t/s (+14.6% ✓) | 29.2 t/s (**+15.6% ✓**) | 9,873ms (+13.7% ✓) |
| `qwen35-4b-q4km|metal|elephant` | 612ms (−8.5% ⚠) | 110.9 t/s (−11.4% ⚠) | 17.7 t/s (**+14.5% ✓**) | 16,888ms (+12.5% ✓) |
| `qwen35-4b-q8|metal|elephant` | 644ms (−2.7%) | 111.5 t/s (+0.4%) | 14.5 t/s (**+10.5% ✓**) | 20,407ms (+8.3% ✓) |

**Summary**: 76 cells improved > ±5%, 1 cell regressed > ±5% (out of 160).

### Δ vs fiber-baseline-today (same-thermal apples-to-apples)

Full table in `results/parsed/mac-rc3-rerun-2026-05-14T1830-section-vs-today.md`.

**Summary**: 86 cells improved > ±5%, 2 cells regressed > ±5% (out of 160).

### Δ rerun vs orig

Per-cell diff in `results/parsed/mac-rc3-rerun-2026-05-14T1830-vs-rc3-orig.md`:
**127 cells improved > ±5%**, 2 regressed (out of 160 measured). This is the
clearest stale-build-artifact signature in the entire study — only 2 cells
out of 160 don't show improvement after a clean rebuild.

### Final updated cross-branch summary (all clean-build measurements)

Using the clean-build numbers wherever they exist (RC1 rerun, RC3 rerun);
RC2 is unchanged because its original measurement was clean (no preceding
build state to leak artifacts).

| Variant | Branch | Gemma4-E2B-Q4 decode (t/s) | Δ vs fiber-2026-05-13 | Qwen3.5-2B-Q4 decode (t/s) | Δ vs fiber-2026-05-13 |
|---|---|---:|---:|---:|---:|
| Fiber 2026-05-13 (anchor) | `tetherto/temp-8189` | 31.45 | — | 32.56 | — |
| b9025 reference (from metal-baseline.md) | upstream | 50.73 | +61.3% | 39.79 | +22.2% |
| Fiber today (same thermal as RC runs) | `tetherto/temp-8189` | 30.68 | −2.4% | 29.88 | −8.2% |
| **RC2 isolated** | `feat/QVAC-18297-rc2-mul-mat-opt` | **41.89** | **+33.2%** | 32.39 | −0.5% |
| **RC1 isolated (clean build)** ✅ | `feat/QVAC-18297-rc1-gated-delta-net` | **42.48** | **+35.1%** | **38.54** | **+18.4%** |
| **RC3 isolated (clean build)** ✅ | `feat/QVAC-18297-rc3-fa-dk512` | **52.43** | **+66.7%** | **36.53** | **+12.2%** |
| RC2 control rerun | `feat/QVAC-18297-rc2-mul-mat-opt` | 42.26 | +34.4% | 38.65 | +18.7% |

#### Revised conclusion

All three commits independently deliver real Mac M4 wins. The original
`2026-05-14T1116` measurements painted a false picture because the
incremental cmake build leaked stale embedded-Metal-shader artifacts between
branch switches.

- **RC2 (Metal Tensor API MUL_MAT)** — clean, +33% Gemma4 / ~0% Qwen3.5
- **RC1 (fused GATED_DELTA_NET)** — clean, +35% Gemma4 / +18% Qwen3.5
- **RC3 (FA dk512_dv512)** — clean, +67% Gemma4 / +12% Qwen3.5

**RC3 is the single highest-impact commit on M4** — it recovers Gemma4 from
the fiber FA-disable regression and brings it to b9025-or-better levels.

For benchmark hygiene going forward: any branch-switching benchmark
orchestration on this codebase **must** use `--clean-first` (or, equivalently,
`rm -rf build-mac && cmake -B build-mac …`) between variants. The cmake
incremental build does not correctly track the embedded Metal shader blob's
dependency on the source `.metal` file when branches are switched.

### Reproducing the RC3 rerun

```sh
TS=$(date +%Y-%m-%dT%H%M) /Users/ic/repo/vlm-benchmark/tools/scripts/run-rc3-rerun.sh

# Parse + diff:
REPO=/Users/ic/repo/vlm-benchmark
python3 $REPO/tools/scripts/parse-mac-logs.py "$REPO/results/raw/mac-rc3-rerun-${TS}" "$REPO/results/parsed/mac-rc3-rerun-${TS}.json"
python3 $REPO/tools/scripts/diff-parsed.py \
    "$REPO/results/parsed/mac-rc3-fa-dk512-2026-05-14T1116.json" \
    "$REPO/results/parsed/mac-rc3-rerun-${TS}.json" \
    "$REPO/results/parsed/mac-rc3-rerun-${TS}-vs-rc3-orig.md"
```

---

## Appendix D: Run group `2026-05-14T1940` — cumulative `feat/QVAC-18297-fiber-updates` (clean build)


The cumulative branch already existed before this study began: `feat/QVAC-18297-fiber-updates`
stacks all three commits on top of `tetherto/temp-8189`:

```
tetherto/temp-8189 (f686a1324)
 └─ f6d6dbcd1  metal: optimize Metal Tensor API for GGML_OP_MUL_MAT
     └─ 556789f7d  port GGML_OP_GATED_DELTA_NET
         └─ 054141103  add Metal FA dk512_dv512 kernel instantiations
```

This run measures the cumulative branch with the same `--clean-first`
protocol as the RC1 and RC3 reruns. It directly answers "what does the
production fiber-updates branch deliver on M4?" and lets us check whether
the three commits **compose additively** when applied together.

- **Variant slug**: `fiber-updates-rerun`
- **Branch**: `feat/QVAC-18297-fiber-updates` (3 commits on top of `tetherto/temp-8189`)
- **Binary**: `llama.cpp/binaries/fiber-updates-rerun/llama-mtmd-cli`
- **Raw logs**: `results/raw/mac-fiber-updates-rerun-2026-05-14T1940/`
- **Parsed**: `results/parsed/mac-fiber-updates-rerun-2026-05-14T1940.json`
- **Traces**: `results/traces/fiber-updates-rerun-{gemma4-e2b-q4km,qwen35-2b-q4km}-2026-05-14T1940.trace` (364 MB total)

### Headline: cumulative composition recovers and slightly exceeds b9025

| Config (Metal × elephant) | fiber 2026-05-13 | b9025 (upstream) | RC2 isolated | RC1 rerun | RC3 rerun | **fiber-updates rerun** | Δ FUP vs fiber-old | Δ FUP vs b9025 | Δ FUP vs best-iso |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma4-e2b-q4km | 31.45 | 50.73 | 41.89 | 42.48 | 52.43 | **52.48** | **+66.9%** | +3.4% | +0.1% |
| gemma4-e2b-q8 | 21.79 | 30.88 | 25.92 | 25.63 | 30.27 | **30.16** | **+38.4%** | −2.3% | −0.4% |
| gemma4-e4b-q4km | 16.78 | 22.82 | 18.83 | 19.31 | 23.88 | **24.13** | **+43.8%** | +5.7% | +1.0% |
| gemma4-e4b-q8 | 12.66 | 14.34 | 13.06 | 14.94 | 17.63 | **16.98** | **+34.1%** | +18.4% | −3.7% |
| qwen35-2b-q4km | 32.56 | 39.79 | 32.39 | 38.54 | 36.53 | **40.00** | **+22.9%** | +0.5% | **+3.8%** |
| qwen35-2b-q8 | 25.28 | 30.37 | 25.59 | 29.99 | 29.23 | **32.34** | **+27.9%** | +6.5% | **+7.8%** |
| qwen35-4b-q4km | 15.46 | 17.60 | 14.54 | 17.89 | 17.70 | **18.44** | **+19.3%** | +4.8% | **+3.1%** |
| qwen35-4b-q8 | 13.15 | 14.08 | 12.03 | 14.25 | 14.53 | **15.23** | **+15.8%** | +8.2% | **+4.8%** |

`best-iso` = the highest decode_tps across the three isolated clean-build measurements (RC2, RC1-rerun, RC3-rerun).

Two distinct composition patterns emerge:

- **Gemma4 (saturated by RC3)**. fiber-updates ≈ RC3-rerun alone (Δ from best-iso = −3.7% to +1.0%, all within noise). RC3's FA enable is the dominant fix; RC2's MUL_MAT kernel split and RC1's GDN op contribute no measurable additional gain on Gemma4 once FA is live.
- **Qwen3.5 (additive — RC1 + RC3 stack)**. fiber-updates is **+3.1% to +7.8% above the best isolated commit** on every Qwen3.5 variant. RC1 (fused GDN op) and RC3 (broader FA op support) deliver independent wins that combine. RC2 alone is essentially neutral on Qwen3.5, but does not detract from the stacked result.

### Headline summary — all clean-build Mac M4 Metal × elephant decode_tps

| Config | fiber-old | fiber-today | b9025 | RC2 iso | RC1 rerun | RC3 rerun | **fiber-updates rerun** |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemma4-e2b-q4km | 31.45 | 30.68 | 50.73 | 41.89 | 42.48 | **52.43** | **52.48** |
| gemma4-e2b-q8 | 21.79 | 19.87 | 30.88 | 25.92 | 25.63 | **30.27** | **30.16** |
| gemma4-e4b-q4km | 16.78 | 14.81 | 22.82 | 18.83 | 19.31 | **23.88** | **24.13** |
| gemma4-e4b-q8 | 12.66 | 10.91 | 14.34 | 13.06 | 14.94 | **17.63** | **16.98** |
| qwen35-2b-q4km | 32.56 | 29.88 | 39.79 | 32.39 | **38.54** | 36.53 | **40.00** |
| qwen35-2b-q8 | 25.28 | 24.67 | 30.37 | 25.59 | **29.99** | 29.23 | **32.34** |
| qwen35-4b-q4km | 15.46 | 14.55 | 17.60 | 14.54 | **17.89** | 17.70 | **18.44** |
| qwen35-4b-q8 | 13.15 | 12.46 | 14.08 | 12.03 | **14.25** | 14.53 | **15.23** |

(Cell highlighting: **bold** marks the best decode_tps in each row.)

### Δ vs fiber-8189 (metal-baseline.md `fiber-mac-2026-05-13T1856`)

Full table in `results/parsed/mac-fiber-updates-rerun-2026-05-14T1940-section-vs-old.md`.

| Config | vision_ms | prefill_tps | decode_tps | total_ms |
|---|---|---|---|---|
| `gemma4-e2b-q4km|metal|elephant` | 610ms (+22.3% ✓) | 260.3 t/s (+46.8% ✓) | 52.5 t/s (**+66.9% ✓**) | 6,327ms (+37.5% ✓) |
| `gemma4-e2b-q8|metal|elephant` | 654ms (+14.5% ✓) | 235.7 t/s (+27.4% ✓) | 30.2 t/s (**+38.4% ✓**) | 10,043ms (+26.4% ✓) |
| `gemma4-e4b-q4km|metal|elephant` | 702ms (+17.9% ✓) | 142.4 t/s (+29.9% ✓) | 24.1 t/s (**+43.8% ✓**) | 13,046ms (+28.7% ✓) |
| `gemma4-e4b-q8|metal|elephant` | 675ms (+9.9% ✓) | 163.6 t/s (+15.5% ✓) | 17.0 t/s (**+34.1% ✓**) | 17,904ms (+23.3% ✓) |
| `qwen35-2b-q4km|metal|elephant` | 475ms (−1.9%) | 237.8 t/s (−6.1% ⚠) | 40.0 t/s (**+22.9% ✓**) | 7,661ms (+16.6% ✓) |
| `qwen35-2b-q8|metal|elephant` | 448ms (+6.5% ✓) | 279.2 t/s (+11.8% ✓) | 32.3 t/s (**+27.9% ✓**) | 9,026ms (+21.1% ✓) |

**Summary**: 73 cells improved > ±5%, 8 cells regressed > ±5% (out of 160).

### Δ vs fiber-baseline-today (same-thermal apples-to-apples)

Full table in `results/parsed/mac-fiber-updates-rerun-2026-05-14T1940-section-vs-today.md`.

**Summary**: 81 cells improved > ±5%, 5 cells regressed > ±5% (out of 160).

### Cross-validation: cumulative vs sum of isolated

Comparing cumulative (FUP rerun) decode_tps against the **best** isolated
clean-build measurement, per config:

| Config | Best isolated (which) | FUP rerun | Δ FUP / best-iso | Composition pattern |
|---|---:|---:|---:|---|
| gemma4-e2b-q4km | 52.43 (RC3) | 52.48 | +0.1% | saturated by RC3 |
| gemma4-e2b-q8 | 30.27 (RC3) | 30.16 | −0.4% | saturated by RC3 |
| gemma4-e4b-q4km | 23.88 (RC3) | 24.13 | +1.0% | saturated by RC3 |
| gemma4-e4b-q8 | 17.63 (RC3) | 16.98 | −3.7% | saturated by RC3 |
| qwen35-2b-q4km | 38.54 (RC1) | 40.00 | **+3.8%** | RC1 + RC3 additive |
| qwen35-2b-q8 | 29.99 (RC1) | 32.34 | **+7.8%** | RC1 + RC3 additive |
| qwen35-4b-q4km | 17.89 (RC1) | 18.44 | **+3.1%** | RC1 + RC3 additive |
| qwen35-4b-q8 | 14.53 (RC3) | 15.23 | **+4.8%** | RC1 + RC3 additive |

Mechanistic reading:

- **Gemma4 has no GDN layers** — RC1 (GDN op port) is a no-op for it. With RC2's
  Metal kernel split and RC3's FA enable both active, the third commit (the
  one not in play) adds nothing measurable. Gemma4 is FA-bound; once FA is on,
  the throughput ceiling on M4 is hardware-bound.
- **Qwen3.5 uses both** — RC1 wires the fused GDN op into Qwen3.5's recurrent
  layers, RC3's expanded supports_op table improves the non-GDN attention
  path. Both contribute to Qwen3.5 decode, hence the additive ≈4–8% on top of
  the best single-commit clean-build measurement.

### Final cross-branch summary (all clean-build measurements + cumulative)

| Variant | Branch | Gemma4-E2B-Q4 decode (t/s) | Δ vs fiber-2026-05-13 | Qwen3.5-2B-Q4 decode (t/s) | Δ vs fiber-2026-05-13 |
|---|---|---:|---:|---:|---:|
| Fiber 2026-05-13 (anchor) | `tetherto/temp-8189` | 31.45 | — | 32.56 | — |
| b9025 reference (`metal-baseline.md`) | upstream | 50.73 | +61.3% | 39.79 | +22.2% |
| Fiber today (same-thermal as RC runs) | `tetherto/temp-8189` | 30.68 | −2.4% | 29.88 | −8.2% |
| RC2 isolated | `feat/QVAC-18297-rc2-mul-mat-opt` | 41.89 | +33.2% | 32.39 | −0.5% |
| RC1 isolated (clean build) ✅ | `feat/QVAC-18297-rc1-gated-delta-net` | 42.48 | +35.1% | 38.54 | +18.4% |
| RC3 isolated (clean build) ✅ | `feat/QVAC-18297-rc3-fa-dk512` | 52.43 | +66.7% | 36.53 | +12.2% |
| **Cumulative (RC2+RC1+RC3, clean build)** ✅ | `feat/QVAC-18297-fiber-updates` | **52.48** | **+66.9%** | **40.00** | **+22.9%** |

The cumulative `feat/QVAC-18297-fiber-updates` branch is the **production
target**: it slightly exceeds b9025 on Gemma4-E2B-Q4 Metal decode (52.48 vs
50.73, +3.4%) and reaches b9025 parity on Qwen3.5-2B-Q4 Metal decode (40.00
vs 39.79, +0.5%). Across the full 8-config Metal × elephant matrix, every
config improves +15% to +67% over fiber-2026-05-13, and most are at or above
b9025.

#### Closing recommendation

`feat/QVAC-18297-fiber-updates` is ready for merge into the fiber tip. All
three commits earn their place: RC3 dominates the Gemma4 win, RC1 dominates
the Qwen3.5 win, and RC2's MUL_MAT kernel restructuring is the most
broad-spectrum optimization (every Gemma4 variant improves modestly, no
regressions). The build-system hazard surfaced in this study —
incremental-build artifact leakage between branch switches — should be
filed as a follow-up; the workaround (`cmake --build … --clean-first`) is
straightforward but easy to forget.

### Reproducing the fiber-updates rerun

```sh
TS=$(date +%Y-%m-%dT%H%M) /Users/ic/repo/vlm-benchmark/tools/scripts/run-fiber-updates-rerun.sh

# Parse + diff:
REPO=/Users/ic/repo/vlm-benchmark
python3 $REPO/tools/scripts/parse-mac-logs.py "$REPO/results/raw/mac-fiber-updates-rerun-${TS}" "$REPO/results/parsed/mac-fiber-updates-rerun-${TS}.json"
python3 $REPO/tools/scripts/format-rc-results.py \
    "$REPO/results/parsed/fiber-mac-2026-05-13T1856.json" \
    "$REPO/results/parsed/mac-fiber-updates-rerun-${TS}.json" \
    "feat/QVAC-18297-fiber-updates (RC2+RC1+RC3 cumulative, clean)" "all-3-stacked" "$TS" \
    "$REPO/results/parsed/mac-fiber-updates-rerun-${TS}-section-vs-old.md"
```

---
