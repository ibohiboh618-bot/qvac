'use strict'

const fs = require('bare-fs')
const path = require('bare-path')
const { round, average, stddev, cartesianProduct } = require('./math')
const { stripSurroundingQuotes, normalizeArgValue } = require('./utils')

const PROMPT_CASES = ['long', 'ctx-filling', 'span-fill']
const PROMPTS_PER_CASE = 1

const SWEEP_OVERRIDE_KEYS = [
  'quantization',
  'sampling-preset',
  'prompt-case',
  'device',
  'ctx-size',
  'threads',
  'batch-size',
  'ubatch-size',
  'flash-attn',
  'cache-type-k',
  'cache-type-v'
]

const CONFIG_METADATA_KEYS = new Set([
  'sampling-preset'
])

const FAKE_PRODUCTIVITY_TOOLS = [
  {
    type: 'function',
    name: 'run_cli',
    description: 'Run a safe local command and return stdout.',
    parameters: {
      type: 'object',
      properties: {
        command: { type: 'string', description: 'Command to run without shell metacharacters.' }
      },
      required: ['command']
    }
  },
  {
    type: 'function',
    name: 'web_search',
    description: 'Search the web for public information.',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string' },
        limit: { type: 'number' }
      },
      required: ['query']
    }
  },
  {
    type: 'function',
    name: 'google_calendar',
    description: 'Find or create calendar events.',
    parameters: {
      type: 'object',
      properties: {
        operation: { type: 'string' },
        title: { type: 'string' },
        date: { type: 'string' },
        time: { type: 'string' },
        attendees: { type: 'array', items: { type: 'string' } }
      },
      required: ['operation', 'title', 'date', 'time']
    }
  },
  {
    type: 'function',
    name: 'google_drive',
    description: 'Search Google Drive files.',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string' },
        mimeType: { type: 'string' },
        folder: { type: 'string' }
      },
      required: ['query']
    }
  },
  {
    type: 'function',
    name: 'github',
    description: 'Read GitHub PRs, issues, and checks.',
    parameters: {
      type: 'object',
      properties: {
        operation: { type: 'string' },
        repo: { type: 'string' },
        number: { type: 'number' },
        query: { type: 'string' }
      },
      required: ['operation', 'repo']
    }
  },
  {
    type: 'function',
    name: 'slack',
    description: 'Search Slack messages or post a message to a channel.',
    parameters: {
      type: 'object',
      properties: {
        operation: { type: 'string' },
        channel: { type: 'string' },
        text: { type: 'string' },
        query: { type: 'string' }
      },
      required: ['operation', 'channel']
    }
  },
  {
    type: 'function',
    name: 'asana',
    description: 'Create or inspect Asana tasks.',
    parameters: {
      type: 'object',
      properties: {
        operation: { type: 'string' },
        project: { type: 'string' },
        title: { type: 'string' },
        dueDate: { type: 'string' },
        assignee: { type: 'string' }
      },
      required: ['operation', 'project', 'title']
    }
  },
  {
    type: 'function',
    name: 'gmail',
    description: 'Draft or search email messages.',
    parameters: {
      type: 'object',
      properties: {
        operation: { type: 'string' },
        to: { type: 'string' },
        subject: { type: 'string' },
        body: { type: 'string' },
        query: { type: 'string' }
      },
      required: ['operation']
    }
  },
  {
    type: 'function',
    name: 'google_sheets',
    description: 'Read or append rows in Google Sheets.',
    parameters: {
      type: 'object',
      properties: {
        operation: { type: 'string' },
        spreadsheet: { type: 'string' },
        sheet: { type: 'string' },
        values: { type: 'array', items: { type: 'string' } }
      },
      required: ['operation', 'spreadsheet', 'sheet']
    }
  }
]

const RUNTIME_SWEEP_KEYS = [
  'device',
  'ctx-size',
  'threads',
  'batch-size',
  'ubatch-size',
  'flash-attn',
  'cache-type-k',
  'cache-type-v'
]

function splitCsvArg (value, key) {
  const normalizedInput = normalizeArgValue(value)
  if (normalizedInput === true || normalizedInput == null || normalizedInput === '') {
    throw new Error(`Missing value for --${key}. Expected comma-separated values.`)
  }
  const parts = String(normalizedInput)
    .split(',')
    .map((v) => stripSurroundingQuotes(v).trim())
    .filter(Boolean)
  if (parts.length === 0) {
    throw new Error(`Empty value for --${key}. Expected comma-separated values.`)
  }
  return parts
}

function buildSweepFromArgs (baseSweep, args) {
  const nextSweep = {}
  const overrideKeys = []
  for (const [key, values] of Object.entries(baseSweep)) {
    nextSweep[key] = Array.isArray(values) ? values.slice() : values
  }

  for (const key of SWEEP_OVERRIDE_KEYS) {
    if (!Object.prototype.hasOwnProperty.call(args, key)) continue
    const rawValues = splitCsvArg(args[key], key)
    nextSweep[key] = rawValues.map((v) => String(v))
    overrideKeys.push(key)
  }

  Object.defineProperty(nextSweep, '__overrideKeys', {
    value: new Set(overrideKeys),
    enumerable: false
  })
  return nextSweep
}

function resolveSamplingPreset (modelDef, name) {
  const presets = modelDef.samplingPresets || {}
  if (!Object.prototype.hasOwnProperty.call(presets, name)) {
    throw new Error(`Unknown sampling preset "${name}" for model "${modelDef.id}"`)
  }
  const preset = presets[name]
  return {
    name,
    description: preset.description || '',
    config: preset.config || {}
  }
}

function isToolPromptCase (promptCase) {
  return String(promptCase || '').startsWith('tool-')
}

function runtimeConfigWithPreset (defaults, preset, promptCase) {
  const runtimeConfig = {
    ...defaults,
    ...preset.config,
    'sampling-preset': preset.name
  }
  if (isToolPromptCase(promptCase)) {
    runtimeConfig.tools = 'true'
  }
  return runtimeConfig
}

function applyExplicitRuntimeSweepOverrides (runtimeConfig, sweep) {
  const overrideKeys = sweep.__overrideKeys || new Set()
  for (const key of RUNTIME_SWEEP_KEYS) {
    if (!overrideKeys.has(key)) continue
    const values = sweep[key]
    if (!Array.isArray(values) || values.length === 0) continue
    runtimeConfig[key] = values[0]
  }
  return runtimeConfig
}

function ensureDir (dirPath) {
  fs.mkdirSync(dirPath, { recursive: true })
}

function resolveModelName (modelDef, quantization) {
  return modelDef.quantizationFiles[quantization] || null
}

function checkModelExists (modelDir, modelName) {
  return fs.existsSync(path.join(modelDir, modelName))
}

function buildCases (modelDef, sweep) {
  const baseQuant = Array.isArray(modelDef.quantizations) ? modelDef.quantizations[0] : null
  const defaults = modelDef.defaults || {}
  if (baseQuant == null) {
    throw new Error(`No baseline quantization configured for model "${modelDef.id}"`)
  }
  const supportedQuants = (sweep.quantization || [])
    .filter((quant) => !!resolveModelName(modelDef, quant))

  if (supportedQuants.length === 0) {
    throw new Error(`No supported quantizations found for model "${modelDef.id}"`)
  }

  const devices = sweep.device || []
  const ctxSizes = sweep['ctx-size'] || []
  const batchSizes = sweep['batch-size'] || []
  const ubatchSizes = sweep['ubatch-size'] || []
  const flashAttnValues = sweep['flash-attn'] || []
  const threadsValues = sweep.threads || []
  const cacheTypeKValues = sweep['cache-type-k'] || []
  const cacheTypeVValues = sweep['cache-type-v'] || []
  const promptCases = sweep['prompt-case'] || PROMPT_CASES
  const samplingPresetNames = sweep['sampling-preset'] || ['qvac-current']
  const samplingPresets = samplingPresetNames.map((name) => resolveSamplingPreset(modelDef, name))
  const baselinePreset = samplingPresets.find((preset) => preset.name === 'qvac-current') ||
    resolveSamplingPreset(modelDef, 'qvac-current')

  const cases = []
  for (const promptCase of promptCases) {
    const runtimeConfig = applyExplicitRuntimeSweepOverrides(
      runtimeConfigWithPreset(defaults, baselinePreset, promptCase),
      sweep
    )
    cases.push({
      caseId: `${modelDef.id}__q=${baseQuant}__preset=${baselinePreset.name}__baseline-defaults__pc=${promptCase}`,
      parameter: 'sampling-preset',
      value: baselinePreset.name,
      samplingPreset: baselinePreset.name,
      promptCase,
      quantization: baseQuant,
      modelName: resolveModelName(modelDef, baseQuant),
      runtimeConfig,
      isBaseline: true
    })
  }

  if (devices.length > 0 && ctxSizes.length > 0 && batchSizes.length > 0 && ubatchSizes.length > 0 &&
      flashAttnValues.length > 0 &&
      threadsValues.length > 0 && cacheTypeKValues.length > 0 && cacheTypeVValues.length > 0) {
    const combos = cartesianProduct([
      supportedQuants,
      samplingPresets,
      devices,
      ctxSizes,
      batchSizes,
      ubatchSizes,
      flashAttnValues,
      threadsValues,
      cacheTypeKValues,
      cacheTypeVValues
    ])

    for (const [quantization, samplingPreset, device, ctxSize, batchSize, ubatchSize, flashAttn, threads, cacheTypeK, cacheTypeV] of combos) {
      if (Number(ubatchSize) > Number(batchSize)) {
        continue // Skip combinations where ubatchSize is greater than batchSize
      }
      for (const promptCase of promptCases) {
        const runtimeConfig = {
          ...runtimeConfigWithPreset(defaults, samplingPreset, promptCase),
          device,
          'ctx-size': ctxSize,
          'batch-size': batchSize,
          'ubatch-size': ubatchSize,
          'flash-attn': flashAttn,
          threads,
          'cache-type-k': cacheTypeK,
          'cache-type-v': cacheTypeV
        }
        const caseId = `${modelDef.id}__q=${quantization}__preset=${samplingPreset.name}__dev=${device}__ctx=${ctxSize}__bs=${batchSize}__ubs=${ubatchSize}__fa=${flashAttn}__t=${threads}__ck=${cacheTypeK}__cv=${cacheTypeV}`
        cases.push({
          caseId: `${caseId}__pc=${promptCase}`,
          parameter: 'sampling-preset',
          value: samplingPreset.name,
          samplingPreset: samplingPreset.name,
          promptCase,
          quantization,
          modelName: resolveModelName(modelDef, quantization),
          runtimeConfig,
          isBaseline: false
        })
      }
    }
  }

  cases.sort((a, b) => Number(b.isBaseline) - Number(a.isBaseline))
  return cases
}

function isAdaptivePromptId (promptId) {
  return String(promptId || '').startsWith('ctx-filling__ctx=') ||
    String(promptId || '').startsWith('batch-spanning__ctx=')
}

function selectPromptForCase (allPrompts, runtimeConfig, promptCase) {
  const byId = new Map(allPrompts.map((p) => [p.id, p]))
  const ctx = String(runtimeConfig['ctx-size'])
  const batch = String(runtimeConfig['batch-size'])
  const ctxId = `ctx-filling__ctx=${ctx}`
  const batchId = `batch-spanning__ctx=${ctx}__bs=${batch}`
  const promptId = promptCase === 'ctx-filling'
    ? ctxId
    : (promptCase === 'span-fill' ? batchId : 'long')
  const selectedPromptId = byId.has(promptCase) ? promptCase : promptId
  if (!byId.has(selectedPromptId)) {
    throw new Error(
      `Missing required prompt id "${selectedPromptId}" in prompt file. ` +
      'Run `npm run prepare:prompts` (or pass --prompts-file with exact variants).'
    )
  }
  return byId.get(selectedPromptId)
}

function getAdaptiveBaselineKey (promptId) {
  return isAdaptivePromptId(promptId) ? String(promptId) : null
}

function validatePromptObject (prompt, contextLabel) {
  if (!prompt || typeof prompt !== 'object') {
    throw new Error(`${contextLabel} must be an object`)
  }
  if (typeof prompt.id !== 'string' || !prompt.id.trim()) {
    throw new Error(`${contextLabel} must have a non-empty string 'id'`)
  }
  if (!Array.isArray(prompt.messages)) {
    throw new Error(`${contextLabel} must have a 'messages' array`)
  }
  for (let j = 0; j < prompt.messages.length; j++) {
    const msg = prompt.messages[j]
    if (!msg || typeof msg !== 'object') {
      throw new Error(`${contextLabel} message at index ${j} must be an object`)
    }
    if (typeof msg.role !== 'string' || !msg.role.trim()) {
      throw new Error(`${contextLabel} message at index ${j} must have a non-empty string 'role'`)
    }
    if (typeof msg.content !== 'string') {
      throw new Error(`${contextLabel} message at index ${j} must have a string 'content'`)
    }
  }
  if (prompt.tools != null && !Array.isArray(prompt.tools) && prompt.tools !== 'qwen-preset-fake-tools') {
    throw new Error(`${contextLabel} 'tools' must be an array or "qwen-preset-fake-tools" when present`)
  }
  if (prompt.expectedTool != null && typeof prompt.expectedTool !== 'object') {
    throw new Error(`${contextLabel} 'expectedTool' must be an object when present`)
  }
}

function buildPromptMessages (prompt) {
  const tools = prompt.tools === 'qwen-preset-fake-tools'
    ? FAKE_PRODUCTIVITY_TOOLS
    : prompt.tools
  if (!Array.isArray(tools) || tools.length === 0) return prompt.messages.slice()
  const messages = []
  let insertAt = 0
  while (insertAt < prompt.messages.length && prompt.messages[insertAt].role === 'system') {
    messages.push(prompt.messages[insertAt])
    insertAt++
  }
  for (const tool of tools) messages.push(tool)
  for (let i = insertAt; i < prompt.messages.length; i++) messages.push(prompt.messages[i])
  return messages
}

function extractToolCalls (outputText) {
  const text = String(outputText || '')
  const calls = []
  const qwen35Regex = /<tool_call>\s*<function=([^>\n]+)>\s*([\s\S]*?)<\/function>\s*<\/tool_call>/g
  let qwen35Match = null
  while ((qwen35Match = qwen35Regex.exec(text)) != null) {
    const args = {}
    const paramsRaw = qwen35Match[2]
    const paramRegex = /<parameter=([^>\n]+)>\s*([\s\S]*?)\s*<\/parameter>/g
    let paramMatch = null
    while ((paramMatch = paramRegex.exec(paramsRaw)) != null) {
      args[paramMatch[1].trim()] = paramMatch[2].trim()
    }
    calls.push({
      name: qwen35Match[1].trim(),
      arguments: args,
      parseError: null
    })
  }

  const regexes = [
    /<tool_call>\s*({[\s\S]*?})\s*<\/tool_call>/g,
    /<tool-call>\s*({[\s\S]*?})\s*<\/tool-call>/g
  ]

  for (const regex of regexes) {
    let match = null
    while ((match = regex.exec(text)) != null) {
      try {
        const parsed = JSON.parse(match[1])
        const name = parsed.name || parsed.function || parsed.tool || null
        const args = parsed.arguments || parsed.args || parsed.parameters || {}
        calls.push({ name, arguments: args, parseError: null })
      } catch (error) {
        calls.push({ name: null, arguments: null, parseError: error.message || String(error) })
      }
    }
  }

  return calls
}

function hasRepeatedNgram (outputText, ngramSize = 6, threshold = 4) {
  const tokens = String(outputText || '').trim().split(/\s+/).filter(Boolean)
  if (tokens.length < ngramSize * threshold) return false
  const counts = new Map()
  for (let i = 0; i <= tokens.length - ngramSize; i++) {
    const key = tokens.slice(i, i + ngramSize).join(' ').toLowerCase()
    const count = (counts.get(key) || 0) + 1
    if (count >= threshold) return true
    counts.set(key, count)
  }
  return false
}

function evaluateToolExpectation (prompt, outputText) {
  const expected = prompt.expectedTool
  if (!expected) return null
  const calls = extractToolCalls(outputText)
  const parseErrors = calls.filter((call) => call.parseError).map((call) => call.parseError)
  const expectedName = expected.name || null
  const matchingCall = calls.find((call) => call.name === expectedName) || null
  const requiredArgs = Array.isArray(expected.requiredArgs) ? expected.requiredArgs : []
  const args = matchingCall && matchingCall.arguments && typeof matchingCall.arguments === 'object'
    ? matchingCall.arguments
    : {}
  const missingArgs = requiredArgs.filter((key) => !Object.prototype.hasOwnProperty.call(args, key))
  const forbiddenArgPatterns = expected.forbiddenArgPatterns || {}
  const forbiddenMatches = []
  for (const [argName, pattern] of Object.entries(forbiddenArgPatterns)) {
    if (args[argName] == null) continue
    const re = new RegExp(pattern)
    if (re.test(String(args[argName]))) forbiddenMatches.push(argName)
  }
  const requiredArgPatterns = expected.requiredArgPatterns || {}
  const patternMismatches = []
  for (const [argName, pattern] of Object.entries(requiredArgPatterns)) {
    if (args[argName] == null) {
      patternMismatches.push(argName)
      continue
    }
    const re = new RegExp(pattern, 'i')
    if (!re.test(String(args[argName]))) patternMismatches.push(argName)
  }

  return {
    emitted: calls.length > 0,
    expectedName,
    matchedName: matchingCall ? matchingCall.name : null,
    nameMatched: Boolean(matchingCall),
    requiredArgs,
    missingArgs,
    parseErrors,
    forbiddenMatches,
    patternMismatches,
    passed: calls.length > 0 && Boolean(matchingCall) && missingArgs.length === 0 &&
      parseErrors.length === 0 && forbiddenMatches.length === 0 && patternMismatches.length === 0
  }
}

function analyzeFailureModes (outputText, metrics, runtimeConfig) {
  const text = String(outputText || '')
  const generatedTokens = metrics && metrics.generatedTokens != null ? Number(metrics.generatedTokens) : null
  const nPredict = runtimeConfig && runtimeConfig['n-predict'] != null ? Number(runtimeConfig['n-predict']) : null
  const hasThinkOpen = text.includes('<think>')
  const hasThinkClose = text.includes('</think>')
  const afterThink = hasThinkClose ? text.slice(text.lastIndexOf('</think>') + '</think>'.length).trim() : ''
  return {
    reachedNPredict: Number.isFinite(generatedTokens) && Number.isFinite(nPredict) && generatedTokens >= nPredict,
    missingThinkClose: hasThinkOpen && !hasThinkClose,
    noFinalAnswerAfterThinking: hasThinkOpen && (!hasThinkClose || afterThink.length === 0),
    repeatedNgram: hasRepeatedNgram(text)
  }
}

function evaluatePromptOutput (prompt, outputText, metrics, runtimeConfig) {
  const failureModes = analyzeFailureModes(outputText, metrics, runtimeConfig)
  const tool = evaluateToolExpectation(prompt, outputText)
  return {
    failureModes,
    tool
  }
}

function aggregateRunMetrics (runMetrics) {
  const loadMsValues = runMetrics.map((x) => x.loadMs).filter((x) => x != null)
  const runMsValues = runMetrics.map((x) => x.runMs).filter((x) => x != null)
  const unloadMsValues = runMetrics.map((x) => x.unloadMs).filter((x) => x != null)
  const ttftMsValues = runMetrics.map((x) => x.ttftMs).filter((x) => x != null)
  const tpsValues = runMetrics.map((x) => x.tps).filter((x) => x != null)
  const msPerTokenValues = runMetrics.map((x) => x.runMsPerGeneratedToken).filter((x) => x != null)
  const firstPromptTokens = runMetrics.find((x) => x.promptTokens != null)?.promptTokens ?? null
  const firstGeneratedTokens = runMetrics.find((x) => x.generatedTokens != null)?.generatedTokens ?? null

  return {
    repeats: runMetrics.length,
    loadMsMean: round(average(loadMsValues), 3),
    runMsMean: round(average(runMsValues), 3),
    unloadMsMean: round(average(unloadMsValues), 3),
    loadMsStd: round(stddev(loadMsValues), 3),
    runMsStd: round(stddev(runMsValues), 3),
    unloadMsStd: round(stddev(unloadMsValues), 3),
    ttftMsMean: round(average(ttftMsValues), 3),
    ttftMsStd: round(stddev(ttftMsValues), 3),
    tpsMean: round(average(tpsValues), 3),
    tpsStd: round(stddev(tpsValues), 3),
    promptTokens: firstPromptTokens,
    generatedTokens: firstGeneratedTokens,
    runMsPerGeneratedTokenMean: round(average(msPerTokenValues), 6),
    runMsPerGeneratedTokenStd: round(stddev(msPerTokenValues), 6)
  }
}

function loadPromptsFromFile (filePath) {
  const parsed = JSON.parse(fs.readFileSync(filePath, 'utf8'))
  if (!Array.isArray(parsed)) {
    throw new Error(`Invalid prompts JSON at ${filePath}; expected array`)
  }
  for (let i = 0; i < parsed.length; i++) {
    validatePromptObject(parsed[i], `Prompt at index ${i}`)
  }
  return parsed
}

function loadPreviousCaseRecords (resultsDir, currentJsonlPath) {
  const recordsByCaseKey = new Map()
  let files = []
  try {
    files = fs.readdirSync(resultsDir)
      .filter((name) => /^llm-parameter-sweep-\d{8}-\d{6}\.jsonl$/.test(name))
      .sort()
  } catch {
    return recordsByCaseKey
  }

  for (const name of files) {
    const absPath = path.join(resultsDir, name)
    if (absPath === currentJsonlPath) continue
    let raw = ''
    try {
      raw = fs.readFileSync(absPath, 'utf8')
    } catch {
      continue
    }
    const lines = raw.split('\n').filter(Boolean)
    for (const line of lines) {
      let parsed = null
      try {
        parsed = JSON.parse(line)
      } catch {
        continue
      }
      if (!parsed || !parsed.modelId || !parsed.caseId) continue
      recordsByCaseKey.set(`${parsed.modelId}:${parsed.caseId}`, parsed)
    }
  }
  return recordsByCaseKey
}

function seedBaselineCachesFromRecord (record, baselineOutputs, adaptiveBaselineOutputs) {
  if (!record || !record.isBaseline) return
  const promptResults = Array.isArray(record.promptResults) ? record.promptResults : []
  for (const promptResult of promptResults) {
    if (!promptResult || !promptResult.promptId) continue
    const promptId = String(promptResult.promptId)
    const outputText = typeof promptResult.outputText === 'string'
      ? promptResult.outputText
      : null
    if (outputText == null) continue
    baselineOutputs[promptId] = outputText
    const adaptiveKey = getAdaptiveBaselineKey(promptId)
    if (adaptiveKey) {
      adaptiveBaselineOutputs[adaptiveKey] = outputText
    }
  }
}

module.exports = {
  PROMPT_CASES,
  PROMPTS_PER_CASE,
  SWEEP_OVERRIDE_KEYS,
  CONFIG_METADATA_KEYS,
  splitCsvArg,
  buildSweepFromArgs,
  buildPromptMessages,
  ensureDir,
  resolveModelName,
  checkModelExists,
  buildCases,
  isAdaptivePromptId,
  selectPromptForCase,
  getAdaptiveBaselineKey,
  validatePromptObject,
  evaluatePromptOutput,
  aggregateRunMetrics,
  loadPromptsFromFile,
  loadPreviousCaseRecords,
  seedBaselineCachesFromRecord
}
