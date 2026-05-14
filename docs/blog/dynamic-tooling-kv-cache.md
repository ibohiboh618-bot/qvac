# Dynamic Tooling & KV Cache Management

Your local LLM now receives a tailored toolbox for every interaction, with automatic KV cache clearing to maintain high-speed inference.

Agentic applications tend to grow tool catalogs quickly. A personal assistant might have weather, calendar, file search, notes, reminders, device actions, workspace search, and app-specific commands. But any single user turn usually needs only a small slice of that toolbox.

When every possible tool is sent on every turn, local inference pays for it. The model has more prompt to prefill, the active context gets noisier, and long-running conversations become harder to keep fast. If the app tries to swap tools mid-session, cache reuse can also become fragmented because the prompt shape changed.

QVAC now gives application developers a better primitive: dynamic tools with KV cache compaction.

## What Changed

SDK users can load an LLM with dynamic tools enabled:

```typescript
const modelId = await loadModel({
  modelSrc: QWEN3_1_7B_INST_Q4,
  modelType: "llm",
  modelConfig: {
    ctx_size: 4096,
    tools: true,
    toolsMode: "dynamic",
  },
});
```

After that, each `completion()` call can provide only the tools relevant to the current turn while continuing to use the same `kvCache` key.

Under the hood, the SDK maps dynamic mode to the LLM addon's `tools_compact` behavior. Tool definitions are anchored after the latest user or tool message. When the model finishes a tool-call chain, the transient tool block, intermediate assistant tool-call messages, and tool responses are compacted out of the KV cache. The conversation can continue from the useful parts: the user prompt and the final assistant answer.

That means a local app can keep one fast conversation cache without pinning one giant toolbox to the whole session.

## A Same-Session Example

Imagine a personal assistant app. The user first asks about the weather, then asks for a horoscope, then asks for today's date. In a static setup, the app may be tempted to send every possible assistant tool all the time, even though each turn only needs one narrow capability.

With dynamic tools, the app can rotate the available tools by turn:

```typescript
const kvCache = "assistant-session-123";

const weatherRun = completion({
  modelId,
  kvCache,
  stream: true,
  history: [
    { role: "system", content: "You are a helpful personal assistant." },
    { role: "user", content: "What's the weather in Tokyo?" },
  ],
  tools: [
    {
      name: "get_weather",
      description: "Get current weather for a city",
      parameters: weatherSchema,
    },
  ],
});

const weatherFinal = await weatherRun.final;
const previousAssistantAnswer =
  weatherFinal.cacheableAssistantContent ?? weatherFinal.contentText;

const horoscopeRun = completion({
  modelId,
  kvCache,
  stream: true,
  history: [
    { role: "assistant", content: previousAssistantAnswer },
    { role: "user", content: "Now check my horoscope for Aquarius." },
  ],
  tools: [
    {
      name: "get_horoscope",
      description: "Get today's horoscope for an astrological sign",
      parameters: horoscopeSchema,
    },
  ],
});

const horoscopeFinal = await horoscopeRun.final;
```

The important part is that both turns use the same `kvCache` key, but each turn gets a different tool list. The weather tool does not have to remain active just because the previous turn needed it. The horoscope turn gets its own small toolbox.

A full runnable SDK example uses this same pattern across weather, horoscope, and date tools, including the follow-up completion loop used after tool results are added to history.

## A Tool-Chain Example

Dynamic tools also help when the model needs several tool calls before it can answer. Consider a local travel assistant:

```text
User: "Plan dinner near my hotel tonight."

The model can issue a chain:
1. get_current_location
2. search_nearby_restaurants
3. check_open_hours
4. create_calendar_hold
5. final answer
```

While that chain is active, the model still needs the tool definitions and intermediate tool responses. The KV cache may temporarily look like this:

```text
[system]
-> [user: plan dinner]
-> [tools: location, restaurant search, hours, calendar]
-> [assistant: call get_current_location]
-> [tool: current location]
-> [assistant: call search_nearby_restaurants]
-> [tool: restaurant candidates]
-> [assistant: call check_open_hours]
-> [tool: open restaurants]
-> [assistant: call create_calendar_hold]
-> [tool: hold created]
-> [assistant: final dinner plan]
```

Once the assistant returns the final answer, the chain has done its job. Dynamic tooling compacts the cache back to the durable conversation state:

```text
[system]
-> [user: plan dinner]
-> [assistant: final dinner plan]
```

The next user turn can now bring a different toolbox, such as file search or note-taking tools, without carrying the restaurant-search chain forward in active KV context. The user still sees the final dinner plan in the conversation, but the local model does not keep re-paying for the transient tool definitions and raw tool outputs.

## What This Means On Device

The exact numbers depend on the model, quantization, prompt template, tool schemas, and device. To make this concrete, here are sample measurements from a four-turn on-device assistant workload on a MacBook M4 Pro using a Qwen3 1.7B Instruct Q4 model. The same prompts were run across static and dynamic tool modes, with both a broad-catalog case and a per-turn-required-tools case.

### Sample bench: static broad toolbox vs dynamic narrow toolbox

| turn | static tools | dynamic tools | static TTFT ms | dynamic TTFT ms | static cache tokens | dynamic cache tokens | static t/s | dynamic t/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 12 | 3 | 894.1 | 342.0 | 1526 | 62 | 141.5 | 150.5 |
| 2 | 12 | 2 | 38.6 | 322.8 | 1749 | 287 | 139.5 | 149.9 |
| 3 | 12 | 3 | 192.1 | 362.6 | 2248 | 517 | 136.5 | 146.7 |
| 4 | 12 | 1 | 94.0 | 246.7 | 2569 | 736 | 132.7 | 144.8 |

### Sample bench: static per-turn required tools vs dynamic per-turn required tools

| turn | static tools | dynamic tools | static TTFT ms | dynamic TTFT ms | static cache tokens | dynamic cache tokens | static t/s | dynamic t/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 3 | 351.4 | 344.5 | 686 | 62 | 150.3 | 150.1 |
| 2 | 2 | 2 | 359.6 | 319.0 | 773 | 287 | 149.4 | 150.1 |
| 3 | 3 | 3 | 524.3 | 363.3 | 1081 | 517 | 146.7 | 144.8 |
| 4 | 1 | 1 | 550.9 | 247.3 | 1107 | 736 | 146.2 | 146.1 |

These are sample values, not universal benchmark claims. The useful pattern is:

- Dynamic compaction keeps cache tokens lower across turns, so the active context stays smaller.
- Smaller active context tends to preserve or slightly improve inference throughput (`tokensPerSecond`) in these runs.
- Dynamic mode can show higher TTFT on later turns when compared with static system tools that were already prefetched into cache, because dynamic mode reintroduces per-turn tool blocks.
- The first turn can still favor dynamic mode strongly when the alternative is a broad static catalog with many irrelevant tools.
- When both modes receive only required tools per turn, dynamic still keeps cache growth lower and can keep TTFT flatter later in the conversation.
- For smaller local models, narrowing the per-turn tool surface can also improve call quality: selecting from 2 to 3 relevant tools is often easier than selecting from 12 mixed tools, which can reduce wrong-tool picks and improve end-to-end task reliability.

So the trade-off is explicit: dynamic mode buys context efficiency and often steadier decode performance, while static broad headers can win TTFT on warmed turns precisely because they keep the full tool header resident.

For the dinner-planning example, assume the app has a broad assistant catalog with 12 tools, and each tool definition averages 250 to 500 prompt tokens once descriptions and JSON schemas are included. A static broad toolbox can add roughly 3,000 to 6,000 tool tokens to the session. The actual dinner task only needs four tools, or about 1,000 to 2,000 tool tokens.

That difference matters most for time to first token. On a mobile-class device, prompt prefill is often the part users feel before the model starts responding. If avoiding a broad toolbox saves 2,000 to 4,000 tokens of prompt work, then at an illustrative 100 to 400 prompt tokens per second, that is roughly 5 to 40 seconds of TTFT pressure removed on turns where the app would otherwise need to re-prefill or re-prime a broad tool context.

It also matters for memory and storage. KV cache cost scales with token count, model size, layer count, KV precision, and context settings. A useful mental model is:

```text
cache pressure ~= cached tokens * KV bytes per token
```

For small local LLMs, KV cache can still land in the tens to low hundreds of KB per cached token depending on configuration. That means 1,000 to 3,000 transient tool-chain tokens can translate into tens or hundreds of MB of RAM while active, and similar pressure when persisted to device storage. Dynamic compaction removes those transient tool definitions and raw tool outputs after the final answer, keeping the cache focused on durable conversation state.

These are not benchmark claims. They are planning numbers for app developers: if the product is mobile-first, every unnecessary tool token competes with TTFT, RAM, battery, and cache storage.

## Why This Helps Context Efficiency

The benefit is easiest to see by comparing common agentic flows:

| Flow | Tool Scope | Context Behavior | Best Fit |
| --- | --- | --- | --- |
| Static broad toolbox | Many tools available every turn | Tool definitions can add repeated prefill cost and context noise | Short sessions, stable tool sets, simple integrations |
| Forked agent flow | A branch inherits or copies context | Useful isolation, but the branch may duplicate context and later needs reconciliation | Exploring alternate reasoning paths or experiments |
| Spawned agent flow | A specialized worker gets its own task context | Clean delegation, but usually with orchestration overhead and a separate execution context | Background tasks, parallel work, specialized subagents |
| QVAC dynamic tools | Only relevant tools for the current turn | One local conversation cache continues while transient tool chains are compacted | Application runtimes where the same user session stays active but tools change by screen, permission, route, or task |

Fork and spawn are powerful orchestration patterns in products like Claude and Cursor. They are useful when work should happen in another reasoning branch or another worker. QVAC dynamic tooling solves a different layer of the problem: the inference/runtime layer inside an app that wants the same local model session to continue, but does not want every tool to follow every turn.

In other words, fork and spawn organize agents. Dynamic tools plus KV compaction organize what the model has to carry while it is generating.

## What Happens Under the Hood

With `toolsMode: "dynamic"`, tools are not treated as a permanent session header. They are attached near the turn that needs them. During a tool-call chain, the tool definitions stay available so the model can request one or more tool calls and then consume the tool responses.

Right now, this compact dynamic-tools path is supported for Qwen3 models. On other model architectures, the underlying `tools_compact` flag is ignored by the LLM addon.

Conceptually, a completed chain starts like this:

```text
[user] -> [tools] -> [assistant tool call] -> [tool response] -> [assistant final]
```

After the chain resolves, the cache is compacted to the durable conversation result:

```text
[user] -> [assistant final]
```

This keeps the KV cache focused on conversation state rather than transient execution details. The next turn can reuse the conversation cache and provide a different tool block.

There is one important constraint: compacted tool responses are not preserved for future reasoning. That is usually what you want for ephemeral calls like "what is today's date?" or "what is the current weather?" If a tool fetched content the model must reason about later, the app should either re-run the tool when needed or summarize the important result into the conversation.

## When To Use Dynamic Tools

Use dynamic tools when:

- You are running a supported Qwen3 LLM.
- The app has a large or changing tool catalog.
- Different screens, routes, workspaces, or permission scopes expose different actions.
- Tool responses are mostly ephemeral and can be re-fetched.
- Long conversations should stay fast through KV cache reuse.
- Local-first inference should avoid repeated full-history and full-tool prefill.

Use static tools when:

- The conversation is short.
- The same small tool set is needed for most turns.
- The model needs to keep reasoning over previous raw tool responses.
- The simplicity of one stable session toolbox is more valuable than per-turn specialization.

## The Developer Impact

Dynamic tooling makes local agents feel less like prompt stuffing and more like an application runtime. The app decides what tools make sense right now. The model receives a smaller, more relevant toolbox. The KV cache keeps the durable conversation fast, while completed tool-call chains are cleaned up automatically.

For users, the result is a local LLM that can feel more adaptive without giving up the responsiveness that KV caching makes possible.
