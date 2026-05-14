/**
 * Dynamic-tools on-device measurement flow.
 *
 * This turns the "What This Means On Device" section from
 * docs/blog/dynamic-tooling-kv-cache.md into a runnable comparison:
 *
 * 1. Run the same four-turn assistant workload with a static 12-tool catalog.
 * 2. Run it again with dynamic tools, passing only the tools each turn needs.
 * 3. Run a second pair where both static and dynamic receive only per-turn required tools.
 * 4. Print measured completion stats for both comparison pairs.
 *
 * Run with:
 *   bun run build
 *   bun run bare:example dist/examples/llamacpp-dynamic-tools-device-flow.js
 */
import { z } from "zod";
import {
  completion,
  deleteCache,
  loadModel,
  unloadModel,
  type CompletionEvent,
  type CompletionStats,
  type ToolCall,
  type ToolInput,
  QWEN3_1_7B_INST_Q4,
  VERBOSITY,
} from "@qvac/sdk";

type ChatMessage = {
  role: string;
  content: string;
};

type ToolName =
  | "get_current_location"
  | "search_nearby_restaurants"
  | "check_open_hours"
  | "create_calendar_hold"
  | "read_calendar"
  | "estimate_travel_time"
  | "send_message"
  | "save_note"
  | "get_weather"
  | "search_files"
  | "summarize_document"
  | "control_device";

type TurnSpec = {
  label: string;
  prompt: string;
  tools: ToolName[];
};

type CompletionMeasurement = {
  stats: CompletionStats | undefined;
  wallMs: number;
  toolCalls: ToolCall[];
  text: string;
};

type TurnMeasurement = {
  turn: number;
  label: string;
  toolCount: number;
  completionCalls: number;
  toolCalls: number;
  wallMs: number;
  ttftMs: number | undefined;
  cacheTokens: number | undefined;
  generatedTokens: number | undefined;
  tokensPerSecond: number | undefined;
};

type ScenarioResult = {
  label: string;
  turns: TurnMeasurement[];
};

const locationSchema = z.object({
  precision: z
    .enum(["city", "neighborhood", "street"])
    .describe("How precise the app should make the user location."),
});

const restaurantSearchSchema = z.object({
  area: z.string().describe("Neighborhood, address, or landmark to search near."),
  cuisine: z.string().describe("Cuisine or dining style requested by the user."),
  partySize: z.number().describe("Number of diners."),
});

const openHoursSchema = z.object({
  restaurant: z.string().describe("Restaurant name from the candidate list."),
  date: z.string().describe("Local date to check, formatted as YYYY-MM-DD."),
  time: z.string().describe("Local time to check, formatted as HH:mm."),
});

const calendarHoldSchema = z.object({
  title: z.string().describe("Calendar title for the temporary hold."),
  startTime: z.string().describe("Local start time in ISO-like format."),
  durationMinutes: z.number().describe("Length of the hold in minutes."),
});

const readCalendarSchema = z.object({
  date: z.string().describe("Local date to inspect, formatted as YYYY-MM-DD."),
});

const travelTimeSchema = z.object({
  origin: z.string().describe("Starting point."),
  destination: z.string().describe("Destination point."),
  mode: z.enum(["walk", "transit", "drive"]).describe("Transport mode."),
});

const sendMessageSchema = z.object({
  recipient: z.string().describe("Contact name or phone number."),
  message: z.string().describe("Message body to send."),
});

const saveNoteSchema = z.object({
  title: z.string().describe("Short note title."),
  body: z.string().describe("Note body with the details worth remembering."),
});

const weatherSchema = z.object({
  city: z.string().describe("City name."),
});

const fileSearchSchema = z.object({
  query: z.string().describe("Search query over local user files."),
});

const summarizeDocumentSchema = z.object({
  documentId: z.string().describe("Local document identifier."),
});

const controlDeviceSchema = z.object({
  device: z.string().describe("Smart device name."),
  action: z.enum(["on", "off", "dim"]).describe("Requested device action."),
});

const broadToolCatalog: Record<ToolName, ToolInput> = {
  get_current_location: {
    name: "get_current_location",
    description:
      "Resolve the user's current on-device location for local planning, routing, and nearby search.",
    parameters: locationSchema,
  },
  search_nearby_restaurants: {
    name: "search_nearby_restaurants",
    description:
      "Search locally indexed restaurant listings near a location with cuisine, price, distance, and rating filters.",
    parameters: restaurantSearchSchema,
  },
  check_open_hours: {
    name: "check_open_hours",
    description:
      "Check whether a restaurant is open at a requested local date and time, including last seating guidance.",
    parameters: openHoursSchema,
  },
  create_calendar_hold: {
    name: "create_calendar_hold",
    description:
      "Create a temporary calendar hold so the user can confirm a plan before it becomes a committed event.",
    parameters: calendarHoldSchema,
  },
  read_calendar: {
    name: "read_calendar",
    description:
      "Read the user's local calendar for a given date to find free time, conflicts, and travel buffers.",
    parameters: readCalendarSchema,
  },
  estimate_travel_time: {
    name: "estimate_travel_time",
    description:
      "Estimate route duration between two local places for walking, transit, or driving.",
    parameters: travelTimeSchema,
  },
  send_message: {
    name: "send_message",
    description:
      "Draft and send a short message through the user's local messaging app after user intent is clear.",
    parameters: sendMessageSchema,
  },
  save_note: {
    name: "save_note",
    description:
      "Save durable user-facing notes that should remain available after ephemeral tool outputs are compacted.",
    parameters: saveNoteSchema,
  },
  get_weather: {
    name: "get_weather",
    description:
      "Fetch current weather for a city, including temperature, precipitation, and short-term outdoor comfort.",
    parameters: weatherSchema,
  },
  search_files: {
    name: "search_files",
    description:
      "Search local files and workspace documents by semantic query without uploading private content.",
    parameters: fileSearchSchema,
  },
  summarize_document: {
    name: "summarize_document",
    description:
      "Summarize a local document by identifier while preserving sensitive content on device.",
    parameters: summarizeDocumentSchema,
  },
  control_device: {
    name: "control_device",
    description:
      "Control an allowed smart-home or device action when the user has granted local permission.",
    parameters: controlDeviceSchema,
  },
};

const allTools = Object.values(broadToolCatalog);

const turns: TurnSpec[] = [
  {
    label: "dinner plan",
    prompt:
      "Plan dinner near my hotel tonight. Use get_current_location, search_nearby_restaurants, and check_open_hours before answering.",
    tools: [
      "get_current_location",
      "search_nearby_restaurants",
      "check_open_hours",
    ],
  },
  {
    label: "calendar hold",
    prompt:
      "If dinner works, find a free slot and create a tentative calendar hold. Use read_calendar and create_calendar_hold.",
    tools: ["read_calendar", "create_calendar_hold"],
  },
  {
    label: "travel and message",
    prompt:
      "Work out travel time from here to the restaurant and message Sam the plan. Use get_current_location, estimate_travel_time, and send_message.",
    tools: ["get_current_location", "estimate_travel_time", "send_message"],
  },
  {
    label: "save summary",
    prompt:
      "Save the final dinner plan as a note I can reopen later. Use save_note.",
    tools: ["save_note"],
  },
];

function toolsFor(names: ToolName[]) {
  return names.map((name) => broadToolCatalog[name]);
}

function executeToolCall(call: ToolCall) {
  switch (call.name as ToolName) {
    case "get_current_location":
      return "Current location: Hotel Aurora, Shibuya, Tokyo.";
    case "search_nearby_restaurants":
      return "Candidates: Kissa Sora (7 min walk, Japanese small plates), Nami Bistro (11 min walk, seafood), Lantern Table (14 min walk, casual).";
    case "check_open_hours":
      return "Kissa Sora is open tonight until 22:30, with last seating at 21:30.";
    case "read_calendar":
      return "Calendar is free tonight from 19:00 to 21:30, with no travel conflicts.";
    case "create_calendar_hold":
      return "Created tentative calendar hold: Dinner near hotel, 19:30-21:00.";
    case "estimate_travel_time":
      return "Estimated travel time: 7 minutes walking, 4 minutes by taxi, transit not recommended.";
    case "send_message":
      return "Message sent to Sam: Dinner at Kissa Sora around 19:30, close to the hotel.";
    case "save_note":
      return "Saved note: Tokyo dinner plan - Kissa Sora at 19:30, 7 minute walk from Hotel Aurora.";
    case "get_weather":
      return "Tokyo weather: 18C, dry, light breeze.";
    case "search_files":
      return "No matching local files were needed for this dinner task.";
    case "summarize_document":
      return "No document summary was needed for this dinner task.";
    case "control_device":
      return "No device action was needed for this dinner task.";
    default:
      return `Unknown tool: ${call.name}`;
  }
}

function addOptional(left: number | undefined, right: number | undefined) {
  if (left === undefined) return right;
  if (right === undefined) return left;
  return left + right;
}

function averageDefined(values: Array<number | undefined>) {
  const numbers = values.filter((value): value is number => value !== undefined);
  if (numbers.length === 0) return undefined;
  return numbers.reduce((total, value) => total + value, 0) / numbers.length;
}

function latestDefined(values: Array<number | undefined>) {
  const numbers = values.filter((value): value is number => value !== undefined);
  return numbers.at(-1);
}

function formatNumber(value: number | undefined, digits = 1) {
  if (value === undefined) return "-";
  return value.toFixed(digits);
}

async function measureCompletion(params: {
  modelId: string;
  history: ChatMessage[];
  tools: ToolInput[];
  kvCache: string;
}) {
  const start = Date.now();
  const result = completion({
    modelId: params.modelId,
    history: params.history,
    tools: params.tools,
    kvCache: params.kvCache,
    stream: true,
    generationParams: {
      temp: 0,
      seed: 42,
      predict: 192,
    },
  });

  let text = "";
  const toolCalls: ToolCall[] = [];

  for await (const event of result.events) {
    handleEvent(event, toolCalls);
    if (event.type === "contentDelta") {
      text += event.text;
      process.stdout.write(event.text);
    }
  }

  const final = await result.final;
  return {
    text: final.contentText || text,
    stats: final.stats,
    toolCalls: final.toolCalls,
    wallMs: Date.now() - start,
  };
}

function handleEvent(event: CompletionEvent, toolCalls: ToolCall[]) {
  if (event.type !== "toolCall") return;
  toolCalls.push(event.call);
  console.log(
    `\n-> tool call: ${event.call.name}(${JSON.stringify(event.call.arguments)})`,
  );
}

async function runTurn(params: {
  turnIndex: number;
  modelId: string;
  history: ChatMessage[];
  tools: ToolInput[];
  kvCache: string;
}) {
  const measurements: CompletionMeasurement[] = [];

  for (let step = 0; step < 3; step++) {
    const measurement = await measureCompletion({
      modelId: params.modelId,
      history: params.history,
      tools: params.tools,
      kvCache: params.kvCache,
    });
    measurements.push(measurement);

    params.history.push({
      role: "assistant",
      content: measurement.text,
    });

    if (measurement.toolCalls.length === 0) break;

    for (const call of measurement.toolCalls) {
      const toolResult = executeToolCall(call);
      console.log(`\n<- tool result: ${toolResult}`);
      params.history.push({
        role: "tool",
        content: toolResult,
      });
    }
    console.log("\nassistant final answer:");
  }

  return summarizeTurn({
    turnIndex: params.turnIndex,
    toolCount: params.tools.length,
    measurements,
  });
}

function summarizeTurn(params: {
  turnIndex: number;
  toolCount: number;
  measurements: CompletionMeasurement[];
}) {
  const stats = params.measurements.map((measurement) => measurement.stats);
  const ttftMs = stats.reduce<number | undefined>(
    (total, stat) => addOptional(total, stat?.timeToFirstToken),
    undefined,
  );
  const generatedTokens = stats.reduce<number | undefined>(
    (total, stat) => addOptional(total, stat?.generatedTokens),
    undefined,
  );

  return {
    turn: params.turnIndex + 1,
    label: turns[params.turnIndex]!.label,
    toolCount: params.toolCount,
    completionCalls: params.measurements.length,
    toolCalls: params.measurements.reduce(
      (total, measurement) => total + measurement.toolCalls.length,
      0,
    ),
    wallMs: params.measurements.reduce(
      (total, measurement) => total + measurement.wallMs,
      0,
    ),
    ttftMs,
    cacheTokens: latestDefined(stats.map((stat) => stat?.cacheTokens)),
    generatedTokens,
    tokensPerSecond: averageDefined(
      stats.map((stat) => stat?.tokensPerSecond),
    ),
  };
}

async function runScenario(params: {
  label: string;
  toolsMode: "static" | "dynamic";
  cacheKey: string;
  toolsForTurn: (turn: TurnSpec) => ToolInput[];
}) {
  let modelId: string | undefined;
  try {
    await deleteCache({ kvCacheKey: params.cacheKey });
    modelId = await loadModel({
      modelSrc: QWEN3_1_7B_INST_Q4,
      modelType: "llm",
      modelConfig: {
        ctx_size: 8192,
        tools: true,
        toolsMode: params.toolsMode,
        verbosity: VERBOSITY.ERROR,
      },
      onProgress: (progress) =>
        console.log(`Loading: ${progress.percentage.toFixed(1)}%`),
    });

    const history: ChatMessage[] = [
      {
        role: "system",
        content:
          "You are a local-first personal assistant. Use the named tools when the user explicitly asks for them, then give a concise final answer.",
      },
    ];
    const results: TurnMeasurement[] = [];

    console.log(`\n=== ${params.label} ===`);
    for (let index = 0; index < turns.length; index++) {
      const turn = turns[index]!;
      const turnTools = params.toolsForTurn(turn);
      history.push({ role: "user", content: turn.prompt });

      console.log(
        `\nTurn ${index + 1}: ${turn.label} (${turnTools.length} tools passed)`,
      );
      const turnResult = await runTurn({
        turnIndex: index,
        modelId,
        history,
        tools: turnTools,
        kvCache: params.cacheKey,
      });
      results.push(turnResult);
      printTurnStats(turnResult);
    }

    return { label: params.label, turns: results };
  } finally {
    if (modelId) {
      await unloadModel({ modelId, clearStorage: false });
    }
  }
}

function printTurnStats(turn: TurnMeasurement) {
  console.log(
    [
      "\nturn stats:",
      `tools=${turn.toolCount}`,
      `completionCalls=${turn.completionCalls}`,
      `toolCalls=${turn.toolCalls}`,
      `ttftMs=${formatNumber(turn.ttftMs)}`,
      `tokensPerSecond=${formatNumber(turn.tokensPerSecond)}`,
      `cacheTokens=${formatNumber(turn.cacheTokens, 0)}`,
      `generatedTokens=${formatNumber(turn.generatedTokens, 0)}`,
      `wallMs=${formatNumber(turn.wallMs, 0)}`,
    ].join(" "),
  );
}

function printComparison(staticResult: ScenarioResult, dynamicResult: ScenarioResult) {
  console.log(
    `\n=== Measured comparison: ${staticResult.label} vs ${dynamicResult.label} ===`,
  );
  console.log(
    "turn | static tools | dynamic tools | static TTFT ms | dynamic TTFT ms | static cache tokens | dynamic cache tokens | static t/s | dynamic t/s",
  );
  console.log(
    "-----|--------------|---------------|----------------|-----------------|---------------------|----------------------|------------|------------",
  );

  for (let index = 0; index < staticResult.turns.length; index++) {
    const staticTurn = staticResult.turns[index]!;
    const dynamicTurn = dynamicResult.turns[index]!;
    console.log(
      [
        staticTurn.turn,
        staticTurn.toolCount,
        dynamicTurn.toolCount,
        formatNumber(staticTurn.ttftMs),
        formatNumber(dynamicTurn.ttftMs),
        formatNumber(staticTurn.cacheTokens, 0),
        formatNumber(dynamicTurn.cacheTokens, 0),
        formatNumber(staticTurn.tokensPerSecond),
        formatNumber(dynamicTurn.tokensPerSecond),
      ].join(" | "),
    );
  }

  console.log(
    "\nUse these measured values to replace or validate the planning numbers in the blog post for this machine, model build, and prompt/template setup.",
  );
}

try {
  const staticBroadResult = await runScenario({
    label: "static broad toolbox",
    toolsMode: "static",
    cacheKey: "device-flow-static-tools",
    toolsForTurn: () => allTools,
  });

  const dynamicNarrowResult = await runScenario({
    label: "dynamic narrow toolbox",
    toolsMode: "dynamic",
    cacheKey: "device-flow-dynamic-tools",
    toolsForTurn: (turn) => toolsFor(turn.tools),
  });

  const staticNarrowResult = await runScenario({
    label: "static per-turn required tools",
    toolsMode: "static",
    cacheKey: "device-flow-static-per-turn-tools",
    toolsForTurn: (turn) => toolsFor(turn.tools),
  });

  const dynamicNarrowMatchedResult = await runScenario({
    label: "dynamic per-turn required tools",
    toolsMode: "dynamic",
    cacheKey: "device-flow-dynamic-per-turn-tools",
    toolsForTurn: (turn) => toolsFor(turn.tools),
  });

  printComparison(staticBroadResult, dynamicNarrowResult);
  printComparison(staticNarrowResult, dynamicNarrowMatchedResult);
  await deleteCache({ kvCacheKey: "device-flow-static-tools" });
  await deleteCache({ kvCacheKey: "device-flow-dynamic-tools" });
  await deleteCache({ kvCacheKey: "device-flow-static-per-turn-tools" });
  await deleteCache({ kvCacheKey: "device-flow-dynamic-per-turn-tools" });
  process.exit(0);
} catch (error) {
  console.error("❌ Error:", error);
  process.exit(1);
}
