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
// QVAC-21372: the qvac-fabric A1 build of llama-mtmd-cli emits explicit timing markers
// (and may NOT print the standard `llama_perf_context_print` lines). Parse them as a
// fallback so the several-sources fabric-cli leg still yields TTFT / decode for the
// hybrid (QVAC_PREFILL_CPU off-vs-on) comparison.
//   QVAC_TIMING prefill_ttft_ms=<vision+prefill ms> prompt_tokens=<n>
//   QVAC_TIMING decode_ms=<ms> decode_tokens=<n> decode_tps=<tps>
const QVAC_TTFT_REGEX = /QVAC_TIMING\s+prefill_ttft_ms=(\d+(?:\.\d+)?)\s+prompt_tokens=(\d+)/i
const QVAC_DECODE_REGEX = /QVAC_TIMING\s+decode_ms=(\d+(?:\.\d+)?)\s+decode_tokens=(\d+)\s+decode_tps=(\d+(?:\.\d+)?)/i

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

  // QVAC-21372 fallback: fill any phase the llama_perf lines didn't provide from the
  // fabric A1 markers. prefill_ttft_ms is the full TTFT (vision + prefill); map it to
  // promptEvalMs so the CLI runner's ttft = visionEncodeMs(0) + promptEvalMs is correct.
  const qttft = text.match(QVAC_TTFT_REGEX)
  if (qttft) {
    if (out.promptEvalMs == null) out.promptEvalMs = Number(qttft[1])
    if (out.promptTokens == null) out.promptTokens = Number(qttft[2])
  }
  const qdec = text.match(QVAC_DECODE_REGEX)
  if (qdec) {
    if (out.decodeMs == null) out.decodeMs = Number(qdec[1])
    if (out.decodeTokens == null) out.decodeTokens = Number(qdec[2])
    if (out.decodeTps == null) out.decodeTps = Number(qdec[3])
  }

  return out
}

module.exports = { parseStdoutMetrics }
