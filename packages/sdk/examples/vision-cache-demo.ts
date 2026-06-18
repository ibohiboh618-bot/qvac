/**
 * Vision Prefix Cache — Demo (cache off/on, multi-turn, agent-loop eviction)
 * -------------------------------------------------------------------------
 * Showcases the @qvac/llm-llamacpp vision prefix cache entirely through the
 * @qvac/sdk facade with five scenarios on the SAME image:
 *   - Scenario A (cache DISABLED): two stateless turns; both re-encode the image
 *     (CLIP + mmproj projection), so the repeat turn stays slow.
 *   - Scenario B (cache ENABLED): two stateless turns; turn 1 is a cache MISS
 *     (encode + store), turn 2 is a cache HIT (encode skipped) → TTFT drops.
 *   - Scenario C (multi-turn conversation): a growing conversation; image attached on
 *     turn 1, text follow-ups on turns 2-3 (KV reuse, no vision re-hit), then
 *     turn 4 re-sends the image → vision cache HIT.
 *   - Scenario D (agent loop, cache DISABLED): under a persistent session with a small
 *     ctx, the image is attached on a non-first turn, then filler turns slide it out
 *     of the KV window; with the cache off the re-attach has nothing to hit and
 *     re-encodes the image (the baseline for the workload — use-cases Flow 5).
 *   - Scenario E (agent loop, cache ENABLED): the exact same loop as D but with the
 *     cache on, so the re-attach HITS the vision cache and skips the re-encode. D vs E
 *     is the apples-to-apples cost of the vision cache for this workload (like A/B).
 *
 * KV cache is enabled on every turn. In Scenarios A/B each turn is a separate
 * single-turn completion with a DIFFERENT prompt (auto key), so there is no
 * cross-turn KV reuse — the Scenario B turn-2 speedup is the vision prefix cache.
 * Scenarios C/D/E grow under a shared session key; the vision cache hits whenever the
 * image is re-attached (Scenario C turn 4, Scenario E after eviction), keyed by image
 * hash and independent of whether the image still occupies the KV window — while
 * Scenario D shows the same re-attach paying the full re-encode with the cache off.
 *
 * vision_cache is a load-time-only config, so each scenario is its own model load
 * (load → run → unload). The final summary prints the steps performed, the prompts,
 * the disabled-vs-enabled comparison, and the multi-turn results.
 *
 * Run (from packages/sdk, after `npm run build`):
 *   bun run bare:example dist/examples/vision-cache-demo.js [imagePath]
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
  deleteCache,
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

// Identical prompts across both stateless scenarios so the A/B is apples-to-apples.
const Q_MISS = "What animal is in this image? Answer in one word."; // turn 1
const Q_HIT = "Identify the animal. One word."; // turn 2

// Scenario C follow-ups. The image is sent only on turn 1; these reference "it",
// so they only resolve because the model remembers turn 1 via the conversation.
const Q_TURN2 = "What color is it? One word.";
const Q_TURN3 = "Is it big or small? One word.";
// Turn 4 RE-ATTACHES the same image (with a different prompt). Re-sending the image
// bytes makes the addon look up the vision cache again, which should HIT.
const Q_TURN4 = "Look at the image again — name the animal. One word.";

// Scenario D (agent loop / context eviction). Persistent session key + an image
// attached on a NON-FIRST turn so sliding can evict it, then a re-attach.
const AGENT_KEY = "vision-cache-demo-agent";
const Q_AGENT_INTRO = "Hello! I'll ask about an image soon."; // turn 1 (text, protected)
const Q_AGENT_IMG = "What animal is in this image? One word."; // turn 2 (image)
const Q_AGENT_REASK = "Look at that image again — name the animal. One word."; // re-attach
// ~400-token pad: each filler grows the context in large steps so it either fits
// with generation headroom or clearly overflows (triggering a clean slide) —
// never landing in the narrow "dead zone" where the prefill fits but there is no
// room to generate. n_discarded (>= pad + headroom) keeps post-slide nPast below
// that zone. With temp 0 the trajectory is deterministic, so this stays stable.
const FILLER_PAD = "Background note to keep in mind for later. ".repeat(43);
const FILLER_QS = [
  "What is the capital of France? One word.",
  "Name a primary color. One word.",
  "Name an ocean. One word.",
];

// Fixed KV-cache session key for the Scenario C conversation. Using a known key
// (instead of an auto key) lets the demo delete it up front so each run starts
// from a clean session — KV sessions persist on disk and a stale one would
// otherwise be reloaded and overflow the context on re-runs.
const CONV_KEY = "vision-cache-demo-conversation";

// History message shape accepted by the SDK (completionParamsSchema): only the
// first user turn carries the image attachment.
interface HistoryMsg {
  role: string;
  content: string;
  attachments?: { path: string }[];
}

interface Turn {
  label: string;
  question: string;
  text: string;
  stats: CompletionStats | undefined;
}

interface Scenario {
  miss: Turn; // turn 1 (Q_MISS)
  hit: Turn; // turn 2 (Q_HIT)
}

interface ConvTurn {
  turn: number;
  question: string;
  text: string;
  stats: CompletionStats | undefined;
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

  // KV cache enabled (auto key from history). Each turn is a separate
  // single-turn completion with a DIFFERENT prompt, so the auto-generated
  // keys differ and there is no cross-turn KV-prefix reuse — the vision prefix
  // cache remains what drives the Scenario B turn-2 speedup.
  //
  // The reasoning channel is disabled at load time (reasoning_budget: 0 in the
  // model config) rather than per-request: the KV-cache path primes a
  // system-prompt cache up front, and a per-request toggle would not apply to
  // that priming, leaving the thinking model emitting no visible answer.
  const result = completion({
    modelId,
    history,
    stream: true,
    kvCache: true,
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
 * Run one full scenario: load the model with the vision cache on/off, ask both
 * prompts on the same image (stateless), then unload. Because vision_cache is
 * a load-time-only config, switching it between scenarios requires this reload.
 */
async function runScenario(visionCacheEnabled: boolean): Promise<Scenario> {
  const state = visionCacheEnabled ? "ENABLED" : "DISABLED";
  console.log(`\n🚀 Scenario — vision cache ${state}: loading Gemma 4 E2B…`);

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

  // Same image, same two prompts in every scenario (KV cache enabled per turn).
  const miss = await runTurn(modelId, `cache ${state} — turn 1`, Q_MISS);
  const hit = await runTurn(modelId, `cache ${state} — turn 2`, Q_HIT);

  await unloadModel({ modelId, clearStorage: false });
  return { miss, hit };
}

/**
 * One turn of the multi-turn conversation. Streams tokens, then reads the final
 * result. Returns the displayable turn plus `cacheableAssistantContent` — the
 * canonical assistant string the caller must push back into history verbatim so
 * the next turn's KV-cache lookup hits.
 */
async function runConvTurn(
  modelId: string,
  turn: number,
  question: string,
  history: HistoryMsg[],
  cacheKey: string,
): Promise<{ result: ConvTurn; cacheable: string | undefined }> {
  console.log(`\n── conversation turn ${turn} ──`);
  console.log(`Q: ${question}`);
  process.stdout.write("A: ");

  const run = completion({ modelId, history, stream: true, kvCache: cacheKey });

  let streamed = "";
  for await (const token of run.tokenStream) {
    process.stdout.write(token);
    streamed += token;
  }
  console.log("");

  const final = await run.final;
  return {
    result: { turn, question, text: final.contentText || streamed, stats: final.stats },
    cacheable: final.cacheableAssistantContent,
  };
}

/**
 * Scenario C: a multi-turn conversation about the SAME image, under a shared KV-cache
 * session key. The image is attached on turn 1, then turns 2-3 are text-only
 * follow-ups that reuse the whole prefix — including the image tokens — via the
 * KV cache (the image is not re-evaluated, so the vision cache does not re-hit).
 * Turn 4 RE-ATTACHES the same image with a different prompt: the image is
 * evaluated again, so the vision cache is looked up and HITS (the stored
 * embedding is reused, skipping CLIP + projection).
 */
async function runConversation(): Promise<ConvTurn[]> {
  console.log(
    "\n🚀 Scenario C — multi-turn conversation (image on turn 1, re-sent on turn 4): loading Gemma 4 E2B…",
  );

  const modelId = await loadModel({
    modelSrc: LLM_MODEL,
    modelType: "llamacpp-completion",
    modelConfig: {
      ctx_size: 4096,
      predict: 64,
      device: DEVICE,
      reasoning_budget: 0,
      vision_cache: true,
      vision_cache_budget_mb: 100,
      projectionModelSrc: PROJECTION_MODEL,
    },
    onProgress: (progress) =>
      console.log(`  loading: ${progress.percentage.toFixed(1)}%`),
  });
  console.log(`✅ Model ready: ${modelId}`);

  // Start from a clean conversation: drop any session left on disk by a prior
  // run so a stale, oversized KV state isn't reloaded (which would overflow ctx).
  await deleteCache({ kvCacheKey: CONV_KEY });

  // Image attached ONLY on turn 1; follow-ups reuse the conversation via KV cache.
  const history: HistoryMsg[] = [
    { role: "user", content: Q_MISS, attachments: [{ path: IMAGE_PATH }] },
  ];

  const turns: ConvTurn[] = [];

  const t1 = await runConvTurn(modelId, 1, Q_MISS, history, CONV_KEY);
  turns.push(t1.result);
  history.push({ role: "assistant", content: t1.cacheable ?? t1.result.text });

  history.push({ role: "user", content: Q_TURN2 });
  const t2 = await runConvTurn(modelId, 2, Q_TURN2, history, CONV_KEY);
  turns.push(t2.result);
  history.push({ role: "assistant", content: t2.cacheable ?? t2.result.text });

  history.push({ role: "user", content: Q_TURN3 });
  const t3 = await runConvTurn(modelId, 3, Q_TURN3, history, CONV_KEY);
  turns.push(t3.result);
  history.push({ role: "assistant", content: t3.cacheable ?? t3.result.text });

  // Turn 4 RE-ATTACHES the same image → re-evaluated → vision cache lookup (HIT).
  history.push({
    role: "user",
    content: Q_TURN4,
    attachments: [{ path: IMAGE_PATH }],
  });
  const t4 = await runConvTurn(modelId, 4, Q_TURN4, history, CONV_KEY);
  turns.push(t4.result);

  // Leave no residue so re-runs stay reproducible.
  await deleteCache({ kvCacheKey: CONV_KEY });
  await unloadModel({ modelId, clearStorage: false });
  return turns;
}

/**
 * Scenario D (use-cases Flow 5): agent loop where the image survives context eviction.
 * Under a persistent session key, the image is attached on a NON-FIRST turn (so it
 * is outside the protected first message), then several filler turns grow the
 * conversation past the small ctx_size — the KV window slides and the image's
 * tokens are evicted. Re-attaching the SAME image then HITS the vision cache
 * (keyed by image hash, independent of KV residency) — the case neither the KV
 * cache nor sliding context can cover.
 *
 * Parameterized by `visionCacheEnabled` so the exact same agent loop can be run
 * with the cache OFF (Scenario D, the baseline): everything else is identical (KV
 * sliding still evicts the image), but the re-attach has no cache to hit and must
 * re-encode the image — the apples-to-apples cost of the vision cache for this
 * workload, contrasted against the cache-ON run (Scenario E).
 */
async function runAgentLoop(
  scenario: string,
  visionCacheEnabled: boolean,
  cacheKey: string,
): Promise<{
  imageTurn: ConvTurn;
  reattach: ConvTurn;
  fillers: number;
}> {
  const state = visionCacheEnabled ? "ENABLED" : "DISABLED";
  console.log(
    `\n🚀 Scenario ${scenario} — agent loop, image survives context eviction (vision cache ${state}): loading Gemma 4 E2B…`,
  );

  const modelId = await loadModel({
    modelSrc: LLM_MODEL,
    modelType: "llamacpp-completion",
    modelConfig: {
      ctx_size: 1024,
      // n_discarded > 0 enables the sliding window: when the conversation fills
      // ctx_size, this many tokens are dropped from after the protected first
      // message (evicting the turn-2 image). Sized generously so post-slide nPast
      // leaves room for the next prompt + generation (else the A3 guard throws).
      n_discarded: 640,
      // Small predict (one-word answers) shrinks the A3 generation headroom, and
      // temp 0 makes nPast grow deterministically run-to-run — together they keep
      // the conversation off the overflow "dead zone" where the prefill fits but
      // there is no room for generation (which would throw without sliding).
      predict: 32,
      temp: 0,
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

  await deleteCache({ kvCacheKey: cacheKey });

  const history: HistoryMsg[] = [];
  let turn = 0;

  // Turn 1: plain text → the protected first message (never slid out).
  turn += 1;
  history.push({ role: "user", content: Q_AGENT_INTRO });
  const t1 = await runConvTurn(modelId, turn, Q_AGENT_INTRO, history, cacheKey);
  history.push({ role: "assistant", content: t1.cacheable ?? t1.result.text });

  // Turn 2: image on a NON-FIRST turn → evictable. Cache MISS (encode + store).
  turn += 1;
  history.push({
    role: "user",
    content: Q_AGENT_IMG,
    attachments: [{ path: IMAGE_PATH }],
  });
  const img = await runConvTurn(modelId, turn, Q_AGENT_IMG, history, cacheKey);
  history.push({ role: "assistant", content: img.cacheable ?? img.result.text });

  // Filler turns: unrelated padded Q&A that grows the context until the KV window
  // slides and the turn-2 image is evicted.
  for (const q of FILLER_QS) {
    turn += 1;
    history.push({ role: "user", content: `${FILLER_PAD}${q}` });
    const f = await runConvTurn(modelId, turn, q, history, cacheKey);
    history.push({ role: "assistant", content: f.cacheable ?? f.result.text });
  }

  // Final turn: re-attach the SAME image (now evicted from KV). With the cache on
  // this HITS (no re-encode); with it off the image is re-encoded from scratch.
  turn += 1;
  history.push({
    role: "user",
    content: Q_AGENT_REASK,
    attachments: [{ path: IMAGE_PATH }],
  });
  const re = await runConvTurn(modelId, turn, Q_AGENT_REASK, history, cacheKey);

  await deleteCache({ kvCacheKey: cacheKey });
  await unloadModel({ modelId, clearStorage: false });
  return {
    imageTurn: img.result,
    reattach: re.result,
    fillers: FILLER_QS.length,
  };
}

function fmtMs(ms: number | undefined): string {
  return ms === undefined ? "n/a" : `${Math.round(ms)} ms`;
}

/** "N% faster" of `v` relative to `base` (both TTFTs), or "n/a". */
function pctFaster(base: number | undefined, v: number | undefined): string {
  return base !== undefined && v !== undefined && base > 0
    ? `${((100 * (base - v)) / base).toFixed(1)}% faster`
    : "n/a";
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

// Width of the summary banner / dividers.
const SUMMARY_W = 78;
const SUMMARY_BAR = "=".repeat(SUMMARY_W);

/** Scenario C/D row: a label plus one cell per turn, shown side by side. */
function convRow(label: string, cells: string[]): string {
  return `  ${label.padEnd(19)}: ${cells.map((c) => c.padEnd(13)).join("")}`;
}

function printSummary(
  disabled: Scenario,
  enabled: Scenario,
  conversation: ConvTurn[],
  agentEnabled: { imageTurn: ConvTurn; reattach: ConvTurn; fillers: number },
  agentDisabled: { imageTurn: ConvTurn; reattach: ConvTurn; fillers: number },
): void {
  console.log(`\n===== Vision Prefix Cache ${"=".repeat(SUMMARY_W - 26)}`);
  console.log(`Image : ${IMAGE_PATH}`);
  console.log(`Device: ${DEVICE}`);
  console.log(`Model : ${LLM_MODEL}`);

  console.log("\nSteps performed:");
  console.log("  Scenario A — vision cache DISABLED:");
  console.log("    1. Loaded Gemma 4 E2B with vision_cache: false.");
  console.log("    2. [turn 1] asked P1 → full CLIP + mmproj encode (no cache).");
  console.log(
    "    3. [turn 2] asked P2 → full CLIP + mmproj encode AGAIN (no cache to hit).",
  );
  console.log("    4. Unloaded the model.");
  console.log("  Scenario B — vision cache ENABLED (budget 100 MB):");
  console.log("    5. Reloaded Gemma 4 E2B with vision_cache: true.");
  console.log("    6. [turn 1] asked P1 → cache MISS, image encoded + stored.");
  console.log(
    "    7. [turn 2] asked P2 → cache HIT, CLIP encode + mmproj projection skipped.",
  );
  console.log("    8. Unloaded the model.");
  console.log("  Scenario C — multi-turn conversation (vision + KV cache):");
  console.log("    9.  Reloaded Gemma 4 E2B with vision_cache: true.");
  console.log("    10. [turn 1] asked about the image (attached once) → cache MISS, encoded.");
  console.log(
    "    11. [turns 2-3] text follow-ups → prefix (incl. image) reused via KV cache.",
  );
  console.log(
    "    12. [turn 4] re-attached the SAME image → vision cache HIT (no re-encode).",
  );
  console.log("    13. Unloaded the model.");
  console.log("  Scenario D — agent loop, image survives context eviction (vision cache DISABLED, ctx_size 1024):");
  console.log("    14. Loaded with small ctx and vision_cache: false; turn 1 text (protected), turn 2 image.");
  console.log(
    "    15. Ran filler turns until the KV window slid out the turn-2 image.",
  );
  console.log(
    "    16. Re-attached the SAME image → re-encoded from scratch (no cache to hit).",
  );
  console.log("    17. Unloaded the model.");
  console.log("  Scenario E — same agent loop with vision cache ENABLED (ctx_size 1024):");
  console.log("    18. Reloaded with vision_cache: true; turn 1 text, turn 2 image → MISS (stored).");
  console.log(
    "    19. Ran the same filler turns until the KV window slid out the image.",
  );
  console.log(
    "    20. Re-attached the SAME image → vision cache HIT despite KV eviction.",
  );
  console.log("    21. Unloaded the model.");

  console.log("\nPrompts (identical across both stateless scenarios):");
  console.log(`  turn 1 (P1): "${Q_MISS}"`);
  console.log(`  turn 2 (P2): "${Q_HIT}"`);

  console.log("");
  console.log(`  ${"".padEnd(21)}  ${"cache DISABLED".padEnd(20)}cache ENABLED`);

  // turn 1 (Q_MISS)
  console.log("  -- turn 1 (P1) --");
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

  // turn 2 (Q_HIT)
  console.log("  -- turn 2 (P2) --");
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

  // Headline: the repeat turn (turn 2) — disabled re-encodes, enabled hits.
  const disabledRepeat = disabled.hit.stats?.timeToFirstToken;
  const enabledRepeat = enabled.hit.stats?.timeToFirstToken;
  const isHit = (enabled.hit.stats?.visionCacheHits ?? 0) > 0;
  console.log("");
  console.log(
    `  → turn 2: ${fmtMs(disabledRepeat)} → ${fmtMs(enabledRepeat)} (${pctFaster(disabledRepeat, enabledRepeat)}) ${isHit ? "✅ vision cache hit" : "⚠️ no hit"}`,
  );

  // Scenario C — multi-turn conversation, turns shown side by side.
  console.log(
    "\nScenario C — multi-turn conversation (image on turn 1, re-sent on turn 4):",
  );
  console.log("  Prompts:");
  for (const t of conversation) {
    const tag = t.turn === conversation.length ? "  (image re-attached)" : "";
    console.log(`    turn ${t.turn}: "${t.question}"${tag}`);
  }
  console.log("");
  console.log(
    `  ${"".padEnd(19)}  ${conversation.map((t) => `turn ${t.turn}`.padEnd(13)).join("")}`,
  );
  console.log(
    convRow(
      "answer",
      conversation.map((t) => oneLine(t.text, 11)),
    ),
  );
  console.log(
    convRow(
      "TTFT",
      conversation.map((t) => fmtMs(t.stats?.timeToFirstToken)),
    ),
  );
  console.log(
    convRow(
      "visionCacheHits",
      conversation.map((t) => String(t.stats?.visionCacheHits ?? "n/a")),
    ),
  );
  console.log("");
  const convFirst = conversation[0]?.stats?.timeToFirstToken; // turn 1 (encode)
  const prevTtft =
    conversation[conversation.length - 2]?.stats?.timeToFirstToken; // turn 3
  const resend = conversation[conversation.length - 1]; // turn 4 (image re-sent)
  const resendTtft = resend?.stats?.timeToFirstToken;
  const resendHit = (resend?.stats?.visionCacheHits ?? 0) > 0;
  console.log(
    `  → turns 2-3 reuse via KV cache; turn ${resend?.turn} re-sends image → vision cache ${resendHit ? "HIT" : "miss"}.`,
  );
  console.log(
    `  → turn ${resend?.turn} vs turn ${(resend?.turn ?? 0) - 1}: ${fmtMs(resendTtft)} vs ${fmtMs(prevTtft)} (${pctFaster(prevTtft, resendTtft)}); vs turn 1: ${pctFaster(convFirst, resendTtft)}.`,
  );

  // Scenarios D & E — same agent loop, vision cache DISABLED vs ENABLED (mirrors A/B).
  console.log(
    "\nScenarios D & E — agent loop, image survives context eviction (vision cache DISABLED vs ENABLED):",
  );
  console.log(
    `  ctx_size 1024; image on turn 2, then ${agentEnabled.fillers} filler turns overflow the context (KV slides), then re-attach.`,
  );
  console.log("  Prompts:");
  console.log(`    turn 1 (text)     : "${Q_AGENT_INTRO}"`);
  console.log(`    turn 2 (image)    : "${Q_AGENT_IMG}"`);
  console.log(
    `    filler turns      : ${agentEnabled.fillers} unrelated padded Q&A (context filler)`,
  );
  console.log(`    re-attach (image) : "${Q_AGENT_REASK}"`);
  console.log("");
  console.log(
    `  ${"".padEnd(21)}  ${"cache DISABLED (D)".padEnd(20)}cache ENABLED (E)`,
  );

  // turn 2 — image attached (a MISS either way: cache off can't hit; cache on stores).
  console.log("  -- image turn (MISS) --");
  console.log(
    row(
      "  answer",
      oneLine(agentDisabled.imageTurn.text, 18),
      oneLine(agentEnabled.imageTurn.text, 18),
    ),
  );
  console.log(
    row(
      "  TTFT",
      fmtMs(agentDisabled.imageTurn.stats?.timeToFirstToken),
      fmtMs(agentEnabled.imageTurn.stats?.timeToFirstToken),
    ),
  );
  console.log(
    row(
      "  visionCacheHits",
      String(agentDisabled.imageTurn.stats?.visionCacheHits ?? "n/a"),
      String(agentEnabled.imageTurn.stats?.visionCacheHits ?? "n/a"),
    ),
  );

  // re-attach — the divergence: cache off re-encodes, cache on HITS.
  console.log("  -- re-attach (after KV eviction) --");
  console.log(
    row(
      "  answer",
      oneLine(agentDisabled.reattach.text, 18),
      oneLine(agentEnabled.reattach.text, 18),
    ),
  );
  console.log(
    row(
      "  TTFT",
      fmtMs(agentDisabled.reattach.stats?.timeToFirstToken),
      fmtMs(agentEnabled.reattach.stats?.timeToFirstToken),
    ),
  );
  console.log(
    row(
      "  visionCacheHits",
      String(agentDisabled.reattach.stats?.visionCacheHits ?? "n/a"),
      String(agentEnabled.reattach.stats?.visionCacheHits ?? "n/a"),
    ),
  );

  // Headline: the re-attach — disabled re-encodes, enabled hits the cache.
  const disabledReattach = agentDisabled.reattach.stats?.timeToFirstToken;
  const enabledReattach = agentEnabled.reattach.stats?.timeToFirstToken;
  const agentHit = (agentEnabled.reattach.stats?.visionCacheHits ?? 0) > 0;
  console.log("");
  console.log(
    `  → re-attach: ${fmtMs(disabledReattach)} → ${fmtMs(enabledReattach)} (${pctFaster(disabledReattach, enabledReattach)}) ${agentHit ? "✅ vision cache hit" : "⚠️ no hit"}`,
  );
  console.log(
    "  → image evicted from KV by sliding in both; only the cache-ENABLED run skips the re-encode.",
  );
  console.log(SUMMARY_BAR);
}

try {
  // Start every run from a clean slate. KV-cache sessions persist on disk keyed
  // by (auto key from history) + modelId; vision_cache is stripped before the
  // modelId/config hash, so Scenarios A and B share a key and re-runs would otherwise
  // reload — and keep growing — the same session until it overflows the context.
  await deleteCache({ all: true });

  // Scenario A first (cache disabled = baseline), then Scenario B (cache enabled).
  // vision_cache is load-time only, so each scenario is its own model load.
  const disabled = await runScenario(false);
  const enabled = await runScenario(true);
  const conversation = await runConversation();
  // Scenario D (cache DISABLED, baseline) then Scenario E (same loop, cache ENABLED)
  // so the re-attach can be compared head-to-head, mirroring the A/B comparison.
  // Distinct session keys keep the two runs independent on disk.
  const agentDisabled = await runAgentLoop("D", false, `${AGENT_KEY}-off`);
  const agentEnabled = await runAgentLoop("E", true, `${AGENT_KEY}-on`);

  printSummary(disabled, enabled, conversation, agentEnabled, agentDisabled);

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
