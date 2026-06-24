'use strict'

// Parses metrics from the addon's stdout/stderr stream. These lines are
// emitted by llama.cpp/llama-mtmd at info-or-higher verbosity; the
// benchmark runs the addon with config.verbosity='2' to keep them in
// the captured stream. If the addon ever stops surfacing these lines
// at the JS layer, surface a `visionEncodeMs` field on stats instead.

// Use /g + matchAll so we sum across every "image slice encoded" line
// llama.cpp emits — dynamic-resolution VLMs (Qwen-VL, InternVL, etc.)
// emit one line per tile, and reporting just the first one undercounts
// the actual encoder workload. visionEncodeSliceCount surfaces the
// tile count so it shows up in the report alongside the summed time.
const VISION_ENCODE_REGEX = /image (?:slice )?encoded in\s+(\d+(?:\.\d+)?)\s*ms/gi
// Pulled verbatim from Ian's Metal plan §5.7 — same llama.cpp output
// shape on every platform.
// Prompt eval must be matched BEFORE decode eval since "prompt eval time"
// contains "eval time". The prompt regex is anchored with "prompt" prefix;
// the decode regex uses a negative lookbehind to skip "prompt eval" lines.
const PROMPT_EVAL_REGEX = /prompt eval time\s*=\s*(\d+(?:\.\d+)?)\s*ms\s*\/\s*(\d+)\s+tokens\s*\([^)]*?(\d+(?:\.\d+)?)\s+tokens per second\)/i
const EVAL_TIME_REGEX = /(?<!prompt )eval time\s*=\s*(\d+(?:\.\d+)?)\s*ms\s*\/\s*(\d+)\s+(?:tokens|runs)\s*\([^)]*?(\d+(?:\.\d+)?)\s+tokens per second\)/i
const LOAD_TIME_REGEX = /load time\s*=\s*(\d+(?:\.\d+)?)\s*ms/i
const TOTAL_TIME_REGEX = /total time\s*=\s*(\d+(?:\.\d+)?)\s*ms/i

function parseStdoutMetrics (text) {
  if (!text) return {}
  const out = {}

  const visMatches = [...text.matchAll(VISION_ENCODE_REGEX)]
  if (visMatches.length) {
    out.visionEncodeMs = visMatches.reduce((sum, m) => sum + Number(m[1]), 0)
    out.visionEncodeSliceCount = visMatches.length
  }

  const prompt = text.match(PROMPT_EVAL_REGEX)
  if (prompt) {
    out.promptEvalMs = Number(prompt[1])
    out.promptTokens = Number(prompt[2])
    if (prompt[3]) out.promptTps = Number(prompt[3])
  }

  const eval_ = text.match(EVAL_TIME_REGEX)
  if (eval_) {
    out.decodeMs = Number(eval_[1])
    out.decodeTokens = Number(eval_[2])
    out.decodeTps = Number(eval_[3])
  }

  const load = text.match(LOAD_TIME_REGEX)
  if (load) out.loadMs = Number(load[1])

  const total = text.match(TOTAL_TIME_REGEX)
  if (total) out.totalMs = Number(total[1])

  return out
}

// QVAC-21318: extract the TOTAL KV-cache size (MiB) from the native llama.cpp logs for
// the current run. `logs` is the line array captured by attachSpecLogger (or any
// iterable of log strings). A model can allocate MORE THAN ONE KV cache — e.g. Gemma-4
// has a full-attention cache plus a separate sliding-window cache, each printing its own
// `llama_kv_cache: size = … MiB` line — so we SUM them rather than take one. When `cfg`
// is given, only lines echoing the expected `K (<k>)` / `V (<v>)` quant tags are summed,
// so a previous run's KV line (the buffered flush when the global logger is reinstalled)
// is never counted. Exact-duplicate lines (Android logcat double-prints) are summed once.
// Example line:
//   llama_kv_cache: size = 12.91 MiB (512 cells, 28 layers, ...), K (q8_0): 7.4 MiB, V (q4_0): 5.5 MiB
function parseKvCacheMiB (logs, cfg) {
  const sizeRe = /llama_kv_cache(?:_unified)?:\s*size\s*=\s*([\d.]+)\s*MiB/
  const kTag = cfg && cfg.k ? `K (${cfg.k}` : null
  const vTag = cfg && cfg.v ? `V (${cfg.v}` : null
  const lines = Array.isArray(logs) ? logs : String(logs == null ? '' : logs).split(/\r?\n/)
  const tagged = []
  const untagged = []
  const seen = new Set()
  for (const line of lines) {
    const match = line && line.match(sizeRe)
    if (!match) continue
    if (seen.has(line)) continue // drop exact-duplicate flushes (logcat double-print)
    seen.add(line)
    const mib = parseFloat(match[1])
    if (kTag && vTag && line.includes(kTag) && line.includes(vTag)) tagged.push(mib)
    else if (!kTag) untagged.push(mib)
  }
  const pick = tagged.length ? tagged : (kTag ? [] : untagged)
  if (!pick.length) return null
  return pick.reduce((a, b) => a + b, 0)
}

module.exports = { parseStdoutMetrics, parseKvCacheMiB }
