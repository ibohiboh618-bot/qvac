/**
 * Vision Prefix Cache — Demo (no KV cache; A/B: cache disabled vs enabled)
 * -----------------------------------------------------------------------
 * Same as vision-cache-demo.ts, but with the KV cache DISABLED so the vision
 * prefix cache is the only cache in play. Showcases the @qvac/llm-llamacpp
 * vision prefix cache entirely through the @qvac/sdk facade by running the SAME
 * two questions on the SAME image twice — once with the cache OFF, once with it
 * ON — and comparing them:
 *   - Flow A (cache DISABLED): both calls re-encode the image (CLIP + mmproj
 *     projection); nothing is cached, so the repeat call stays slow.
 *   - Flow B (cache ENABLED): call 1 is a cache MISS (encode + store); call 2
 *     is a cache HIT (encode + projection skipped) and TTFT drops sharply.
 * Every call is stateless (no `kvCache`, KV reset between calls), so the Flow B
 * call-2 speedup is attributable solely to the vision prefix cache — there is
 * no KV-prefix reuse involved at all.
 *
 * vision_cache is a load-time-only config, so the two flows use two model loads
 * (Flow A unloads, then Flow B reloads the same model with the cache on). The
 * final summary prints the questions (identical across flows), the steps
 * performed, and a disabled-vs-enabled comparison per call.
 *
 * Run (from packages/sdk, after `npm run build`):
 *   bun run bare:example dist/examples/vision-cache-demo-no-kv-cache.js [imagePath]
 *
 * Defaults (override via env / argv):
 *   QVAC_DEMO_LLM     local Gemma 4 E2B GGUF      (~/repo/models/...)
 *   QVAC_DEMO_MMPROJ  local Gemma 4 E2B mmproj    (~/repo/models/...)
 *   QVAC_DEMO_DEVICE  "cpu" (default) | "gpu"
 *   argv[2]           image path (default: llm-llamacpp/media/elephant.jpg)
 */

import {
  completion,
  loadModel,
  unloadModel,
  plugins,
  type CompletionStats,
} from "@qvac/sdk";
import { llmPlugin } from "@/server/bare/plugins/llamacpp-completion/plugin";

// Register the llama.cpp completion plugin in the bare worker before any SDK
// call (the bare-targeted SDK ships no plugins implicitly).
plugins([llmPlugin]);

const env = process.env as Record<string, string | undefined>;
const HOME = env["HOME"] ?? "/home/tether";

const LLM_MODEL =
  env["QVAC_DEMO_LLM"] ??
  `${HOME}/repo/models/gemma-4-E2B-it/gemma-4-E2B-it-Q4_K_M.gguf`;
const PROJECTION_MODEL =
  env["QVAC_DEMO_MMPROJ"] ??
  `${HOME}/repo/models/gemma-4-E2B-it/mmproj-F16.gguf`;
const DEVICE = env["QVAC_DEMO_DEVICE"] ?? "cpu";
const IMAGE_PATH =
  process.argv[2] ??
  `${HOME}/repo/qvac/packages/llm-llamacpp/media/elephant.jpg`;

// Identical questions across both flows so the comparison is apples-to-apples.
const Q_MISS = "What animal is in this image? Answer in one word."; // call 1
const Q_HIT = "Identify the animal. One word."; // call 2

interface Turn {
  label: string;
  question: string;
  text: string;
  stats: CompletionStats | undefined;
}

interface Flow {
  miss: Turn; // call 1 (Q_MISS)
  hit: Turn; // call 2 (Q_HIT)
}

/**
 * One stateless completion on the given image + prompt. Streams tokens to
 * stdout and returns the aggregated text plus the runtime stats.
 */
async function runTurn(
  modelId: string,
  label: string,
  prompt: string,
): Promise<Turn> {
  console.log(`\n── ${label} ──`);
  console.log(`Q: ${prompt}`);
  process.stdout.write("A: ");

  const history = [
    {
      role: "user",
      content: prompt,
      attachments: [{ path: IMAGE_PATH }],
    },
  ];

  // KV cache disabled: `kvCache` is omitted, so every call is stateless (KV
  // reset between calls) and there is no KV-prefix reuse — the vision prefix
  // cache is the only cache that can speed up the Flow B repeat call.
  //
  // The reasoning channel is disabled at load time (reasoning_budget: 0 in the
  // model config) so the small predict budget yields a direct one-word answer
  // instead of being consumed by thinking tokens (gemma-4 emits reasoning that
  // the SDK strips from content).
  const result = completion({
    modelId,
    history,
    stream: true,
  });

  let text = "";
  for await (const token of result.tokenStream) {
    process.stdout.write(token);
    text += token;
  }
  console.log("");

  const stats = await result.stats;
  return { label, question: prompt, text, stats };
}

/**
 * Run one full flow: load the model with the vision cache on/off, ask both
 * questions on the same image (stateless), then unload. Because vision_cache is
 * a load-time-only config, switching it between flows requires this reload.
 */
async function runFlow(visionCacheEnabled: boolean): Promise<Flow> {
  const state = visionCacheEnabled ? "ENABLED" : "DISABLED";
  console.log(`\n🚀 Flow — vision cache ${state}: loading Gemma 4 E2B…`);

  const modelId = await loadModel({
    modelSrc: LLM_MODEL,
    modelType: "llamacpp-completion",
    modelConfig: {
      ctx_size: 4096,
      predict: 64,
      device: DEVICE,
      reasoning_budget: 0,
      vision_cache: visionCacheEnabled,
      vision_cache_budget_mb: 100,
      projectionModelSrc: PROJECTION_MODEL,
    },
    onProgress: (progress) =>
      console.log(`  loading: ${progress.percentage.toFixed(1)}%`),
  });
  console.log(`✅ Model ready: ${modelId}`);

  // Same image, same two questions in every flow (KV cache disabled — stateless).
  const miss = await runTurn(modelId, `cache ${state} — call 1`, Q_MISS);
  const hit = await runTurn(modelId, `cache ${state} — call 2`, Q_HIT);

  await unloadModel({ modelId, clearStorage: false });
  return { miss, hit };
}

function fmtMs(ms: number | undefined): string {
  return ms === undefined ? "n/a" : `${Math.round(ms)} ms`;
}

/** Collapse whitespace and clip to a single short line for table cells. */
function oneLine(text: string, max = 40): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat;
}

/** One table row: a label plus the cache-DISABLED and cache-ENABLED cells. */
function row(label: string, disabledCell: string, enabledCell: string): string {
  return `  ${label.padEnd(21)}: ${disabledCell.padEnd(20)}${enabledCell}`;
}

function printSummary(disabled: Flow, enabled: Flow): void {
  console.log(
    "\n====== Vision Prefix Cache (no KV cache) — disabled vs enabled ======",
  );
  console.log(`Image : ${IMAGE_PATH}`);
  console.log(`Device: ${DEVICE}`);
  console.log(`Model : ${LLM_MODEL}`);

  console.log("\nQuestions (identical across both flows):");
  console.log(`  call 1: "${Q_MISS}"`);
  console.log(`  call 2: "${Q_HIT}"`);

  console.log("\nSteps performed:");
  console.log("  Flow A — vision cache DISABLED:");
  console.log("    1. Loaded Gemma 4 E2B with vision_cache: false.");
  console.log("    2. [call 1] asked Q1 → full CLIP + mmproj encode (no cache).");
  console.log(
    "    3. [call 2] asked Q2 → full CLIP + mmproj encode AGAIN (no cache to hit).",
  );
  console.log("    4. Unloaded the model.");
  console.log("  Flow B — vision cache ENABLED (budget 100 MB):");
  console.log("    5. Reloaded Gemma 4 E2B with vision_cache: true.");
  console.log("    6. [call 1] asked Q1 → cache MISS, image encoded + stored.");
  console.log(
    "    7. [call 2] asked Q2 → cache HIT, CLIP encode + mmproj projection skipped.",
  );
  console.log("    8. Unloaded the model.");

  console.log("");
  console.log(`  ${"".padEnd(21)}  ${"cache DISABLED".padEnd(20)}cache ENABLED`);

  // call 1 (Q_MISS)
  console.log("  -- call 1 (Q1) --");
  console.log(
    row(
      "  answer",
      oneLine(disabled.miss.text, 18),
      oneLine(enabled.miss.text, 18),
    ),
  );
  console.log(
    row(
      "  TTFT",
      fmtMs(disabled.miss.stats?.timeToFirstToken),
      fmtMs(enabled.miss.stats?.timeToFirstToken),
    ),
  );
  console.log(
    row(
      "  visionCacheHits",
      String(disabled.miss.stats?.visionCacheHits ?? "n/a"),
      String(enabled.miss.stats?.visionCacheHits ?? "n/a"),
    ),
  );

  // call 2 (Q_HIT)
  console.log("  -- call 2 (Q2) --");
  console.log(
    row(
      "  answer",
      oneLine(disabled.hit.text, 18),
      oneLine(enabled.hit.text, 18),
    ),
  );
  console.log(
    row(
      "  TTFT",
      fmtMs(disabled.hit.stats?.timeToFirstToken),
      fmtMs(enabled.hit.stats?.timeToFirstToken),
    ),
  );
  console.log(
    row(
      "  visionCacheHits",
      String(disabled.hit.stats?.visionCacheHits ?? "n/a"),
      String(enabled.hit.stats?.visionCacheHits ?? "n/a"),
    ),
  );

  // Headline: the repeat call (call 2) — disabled re-encodes, enabled hits.
  const disabledRepeat = disabled.hit.stats?.timeToFirstToken;
  const enabledRepeat = enabled.hit.stats?.timeToFirstToken;
  const pct =
    disabledRepeat !== undefined &&
    enabledRepeat !== undefined &&
    disabledRepeat > 0
      ? `${((100 * (disabledRepeat - enabledRepeat)) / disabledRepeat).toFixed(1)}% faster`
      : "n/a";

  console.log("");
  console.log(
    `  → Repeat call (call 2) TTFT: ${fmtMs(disabledRepeat)} (disabled) vs ${fmtMs(enabledRepeat)} (enabled) — ${pct} with the cache.`,
  );

  const isHit = (enabled.hit.stats?.visionCacheHits ?? 0) > 0;
  console.log(
    isHit
      ? "  ✅ Vision cache hit confirmed in the ENABLED flow (encode + projection skipped)."
      : "  ⚠️  No vision cache hit recorded in the ENABLED flow — check the addon build / config.",
  );
  console.log("=====================================================================");
}

try {
  // Flow A first (cache disabled = baseline), then Flow B (cache enabled).
  // vision_cache is load-time only, so each flow is its own model load.
  const disabled = await runFlow(false);
  const enabled = await runFlow(true);

  printSummary(disabled, enabled);

  // The SDK keeps a persistent bare worker alive after unload (so further
  // models can be loaded), which holds the event loop open. A plain
  // process.exit(0) is not enough here: Bare's graceful exit blocks on the
  // addon's lingering native handle (see QVAC-18197 / worker-core
  // scheduleForceExit). Instead signal the SDK's own bounded shutdown path —
  // SIGTERM triggers shutdownBareDirectWorker, which runs full cleanup,
  // releases the worker lock, and force-exits via SIGKILL if the native
  // handle stalls the exit. This is the same path as Ctrl+C.
  process.kill(process.pid, "SIGTERM");
} catch (error) {
  console.error("❌ Error:", error);
  process.exit(1);
}
