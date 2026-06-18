/**
 * Vision Prefix Cache — Demo (evicted image, NEW text-only question)
 * ------------------------------------------------------------------
 * A variant of Scenario D from vision-cache-demo.ts. The setup is identical —
 * under a persistent session with a small ctx, the image is attached on a
 * NON-FIRST turn, then filler turns slide it out of the KV window — but the
 * final turn is DIFFERENT: instead of re-attaching the image, the user simply
 * asks a NEW text-only question that refers back to the (now evicted) image.
 *
 * The point this makes: the vision prefix cache is keyed by image bytes, so it
 * is only ever consulted when an image is present in the prompt. A text-only
 * follow-up carries no image, so:
 *   - the cache is NOT looked up (visionCacheHits stays 0, no encode either), and
 *   - the model can only answer from what is still in the KV window after the
 *     slide — the protected first message plus the most recent turns. The image
 *     embeddings are gone, and so (likely) is the assistant's early description
 *     of it. The model therefore answers from residual/degraded context — it is
 *     no longer "looking" at the image.
 *
 * Contrast with Scenario D (vision-cache-demo.ts): there the final turn
 * RE-ATTACHES the image, the cache HITS, and the image is restored cheaply. Here
 * nothing re-attaches it, so the cache cannot help — re-sending the bytes is the
 * only way to recover an evicted image.
 *
 * KV cache is enabled (persistent session key) so the conversation grows and the
 * sliding window can evict the image. vision_cache is on, but it stays idle on
 * the text-only final turn.
 *
 * Run (from packages/sdk, after `npm run build`):
 *   bun run bare:example dist/examples/vision-cache-demo-evicted.js [imagePath]
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

// Persistent session key + an image attached on a NON-FIRST turn so sliding can
// evict it. The final turn is a NEW text-only question — the image is NOT re-sent.
const AGENT_KEY = "vision-cache-demo-evicted";
const Q_AGENT_INTRO = "Hello! I'll ask about an image soon."; // turn 1 (text, protected)
const Q_AGENT_IMG = "What animal is in this image? One word."; // turn 2 (image)
// Final turn: a NEW text-only question that refers back to the evicted image.
// No attachment — so the vision cache is never consulted, and the model can only
// answer from whatever survived in the KV window after sliding.
const Q_EVICTED_TEXT = "What color was the animal in that image? One word.";
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

// History message shape accepted by the SDK (completionParamsSchema).
interface HistoryMsg {
  role: string;
  content: string;
  attachments?: { path: string }[];
}

interface ConvTurn {
  turn: number;
  question: string;
  text: string;
  stats: CompletionStats | undefined;
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
    result: {
      turn,
      question,
      text: final.contentText || streamed,
      stats: final.stats,
    },
    cacheable: final.cacheableAssistantContent,
  };
}

/**
 * Agent loop where the image is evicted, then a NEW text-only question is asked
 * (the image is NOT re-attached). Returns the image turn and the final text turn
 * so the summary can contrast them.
 */
async function runEvictedQuestion(): Promise<{
  imageTurn: ConvTurn;
  evictedAsk: ConvTurn;
  fillers: number;
}> {
  console.log(
    "\n🚀 Evicted image, NEW text-only question: loading Gemma 4 E2B…",
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
      vision_cache: true,
      vision_cache_budget_mb: 100,
      projectionModelSrc: PROJECTION_MODEL,
    },
    onProgress: (progress) =>
      console.log(`  loading: ${progress.percentage.toFixed(1)}%`),
  });
  console.log(`✅ Model ready: ${modelId}`);

  await deleteCache({ kvCacheKey: AGENT_KEY });

  const history: HistoryMsg[] = [];
  let turn = 0;

  // Turn 1: plain text → the protected first message (never slid out).
  turn += 1;
  history.push({ role: "user", content: Q_AGENT_INTRO });
  const t1 = await runConvTurn(modelId, turn, Q_AGENT_INTRO, history, AGENT_KEY);
  history.push({ role: "assistant", content: t1.cacheable ?? t1.result.text });

  // Turn 2: image on a NON-FIRST turn → evictable. Cache MISS (encode + store).
  turn += 1;
  history.push({
    role: "user",
    content: Q_AGENT_IMG,
    attachments: [{ path: IMAGE_PATH }],
  });
  const img = await runConvTurn(modelId, turn, Q_AGENT_IMG, history, AGENT_KEY);
  history.push({ role: "assistant", content: img.cacheable ?? img.result.text });

  // Filler turns: unrelated padded Q&A that grows the context until the KV window
  // slides and the turn-2 image is evicted.
  for (const q of FILLER_QS) {
    turn += 1;
    history.push({ role: "user", content: `${FILLER_PAD}${q}` });
    const f = await runConvTurn(modelId, turn, q, history, AGENT_KEY);
    history.push({ role: "assistant", content: f.cacheable ?? f.result.text });
  }

  // Final turn: a NEW text-only question about the (now evicted) image. The image
  // is NOT re-attached, so the vision cache is never consulted and the model can
  // only answer from residual KV context.
  turn += 1;
  history.push({ role: "user", content: Q_EVICTED_TEXT });
  const ask = await runConvTurn(
    modelId,
    turn,
    Q_EVICTED_TEXT,
    history,
    AGENT_KEY,
  );

  await deleteCache({ kvCacheKey: AGENT_KEY });
  await unloadModel({ modelId, clearStorage: false });
  return {
    imageTurn: img.result,
    evictedAsk: ask.result,
    fillers: FILLER_QS.length,
  };
}

function fmtMs(ms: number | undefined): string {
  return ms === undefined ? "n/a" : `${Math.round(ms)} ms`;
}

/** Collapse whitespace and clip to a single short line for table cells. */
function oneLine(text: string, max = 40): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat;
}

// Width of the summary banner / dividers.
const SUMMARY_W = 78;
const SUMMARY_BAR = "=".repeat(SUMMARY_W);

/** A label plus one cell per turn, shown side by side. */
function convRow(label: string, cells: string[]): string {
  return `  ${label.padEnd(19)}: ${cells.map((c) => c.padEnd(15)).join("")}`;
}

function printSummary(agent: {
  imageTurn: ConvTurn;
  evictedAsk: ConvTurn;
  fillers: number;
}): void {
  console.log(`\n===== Vision Prefix Cache ${"=".repeat(SUMMARY_W - 26)}`);
  console.log(`Image : ${IMAGE_PATH}`);
  console.log(`Device: ${DEVICE}`);
  console.log(`Model : ${LLM_MODEL}`);

  console.log("\nSteps performed:");
  console.log(
    "  Evicted image, NEW text-only question (ctx_size 1024, vision_cache on):",
  );
  console.log("    1. Loaded with a small ctx; turn 1 text → protected first message.");
  console.log("    2. [turn 2] attached the image → cache MISS, image encoded + stored.");
  console.log(
    "    3. Ran filler turns until the KV window slid out the turn-2 image.",
  );
  console.log(
    "    4. [final turn] asked a NEW text-only question — image NOT re-attached.",
  );
  console.log("    5. Unloaded the model.");

  console.log("\nPrompts:");
  console.log(`    turn 1 (text)      : "${Q_AGENT_INTRO}"`);
  console.log(`    turn 2 (image)     : "${Q_AGENT_IMG}"`);
  console.log(
    `    filler turns       : ${agent.fillers} unrelated padded Q&A (context filler)`,
  );
  console.log(`    final (text only)  : "${Q_EVICTED_TEXT}"`);

  console.log("");
  console.log(`  ${"".padEnd(19)}  ${"image turn".padEnd(15)}new text Q`);
  console.log(
    convRow("answer", [
      oneLine(agent.imageTurn.text, 13),
      oneLine(agent.evictedAsk.text, 13),
    ]),
  );
  console.log(
    convRow("TTFT", [
      fmtMs(agent.imageTurn.stats?.timeToFirstToken),
      fmtMs(agent.evictedAsk.stats?.timeToFirstToken),
    ]),
  );
  console.log(
    convRow("visionCacheHits", [
      String(agent.imageTurn.stats?.visionCacheHits ?? "n/a"),
      String(agent.evictedAsk.stats?.visionCacheHits ?? "n/a"),
    ]),
  );

  // Print the final turn's answer in full (untruncated) so the degraded,
  // image-less response is readable rather than clipped in the side-by-side cell.
  console.log("");
  console.log(`  final turn Q: ${Q_EVICTED_TEXT}`);
  console.log(`  final turn A: ${agent.evictedAsk.text.replace(/\s+/g, " ").trim()}`);

  console.log("");
  const askHit = (agent.evictedAsk.stats?.visionCacheHits ?? 0) > 0;
  console.log(
    "  → the final turn carries NO image, so the vision cache is never consulted",
  );
  console.log(
    `    (visionCacheHits stays ${agent.evictedAsk.stats?.visionCacheHits ?? "n/a"}${askHit ? " — unexpected" : ""}); the stored embedding sits idle.`,
  );
  console.log(
    "  → with the image evicted from the KV window, the model answers only from",
  );
  console.log(
    "    residual context — it is no longer looking at the image. Re-attaching the",
  );
  console.log(
    "    image (see Scenario D) is the only way to recover it via a cache HIT.",
  );
  console.log(SUMMARY_BAR);
}

try {
  // Start from a clean slate so a stale on-disk session isn't reloaded.
  await deleteCache({ all: true });

  const agent = await runEvictedQuestion();
  printSummary(agent);

  // The SDK keeps a persistent bare worker alive after unload, holding the event
  // loop open. A plain process.exit(0) blocks on the addon's lingering native
  // handle (QVAC-18197); SIGTERM triggers the SDK's bounded shutdown path
  // (shutdownBareDirectWorker → cleanup → SIGKILL fallback). Same path as Ctrl+C.
  process.kill(process.pid, "SIGTERM");
} catch (error) {
  console.error("❌ Error:", error);
  process.exit(1);
}
