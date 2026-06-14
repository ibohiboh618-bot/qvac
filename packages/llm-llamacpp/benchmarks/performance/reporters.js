'use strict'

const { truncateText } = require('./progress')
const { round, average } = require('./math')

function tsFileStamp () {
  const d = new Date()
  const yyyy = String(d.getFullYear())
  const mm = String(d.getMonth() + 1).padStart(2, '0')
  const dd = String(d.getDate()).padStart(2, '0')
  const hh = String(d.getHours()).padStart(2, '0')
  const mi = String(d.getMinutes()).padStart(2, '0')
  const ss = String(d.getSeconds()).padStart(2, '0')
  return `${yyyy}${mm}${dd}-${hh}${mi}${ss}`
}

function compactPromptErrors (promptResults) {
  if (!Array.isArray(promptResults)) return []
  const out = []
  for (const item of promptResults) {
    if (!item || !item.error) continue
    out.push({
      promptId: item.promptId,
      error: truncateText(item.error, 300),
      vramError: Boolean(item.vramError)
    })
  }
  return out
}

function compactQualityChecks (promptResults) {
  if (!Array.isArray(promptResults)) return []
  const out = []
  for (const item of promptResults) {
    const checks = item && item.qualityChecks ? item.qualityChecks : null
    if (!checks) continue
    out.push({
      promptId: item.promptId,
      toolPassed: checks.tool ? Boolean(checks.tool.passed) : null,
      loopFlags: checks.failureModes || null
    })
  }
  return out
}

function formatQualityChecks (item) {
  const checks = compactQualityChecks(item.promptResults)
  if (checks.length === 0) return ''
  return checks.map((check) => {
    const parts = [check.promptId]
    if (check.toolPassed != null) {
      parts.push(`tool=${check.toolPassed ? 'pass' : 'fail'}`)
      const promptResult = (item.promptResults || []).find((p) => p.promptId === check.promptId)
      const tool = promptResult && promptResult.qualityChecks ? promptResult.qualityChecks.tool : null
      if (tool && !tool.passed) {
        const reasons = []
        if (!tool.emitted) reasons.push('not-emitted')
        if (!tool.nameMatched) reasons.push(`name=${tool.matchedName || 'none'}`)
        if (Array.isArray(tool.missingArgs) && tool.missingArgs.length > 0) reasons.push(`missing=${tool.missingArgs.join('+')}`)
        if (Array.isArray(tool.patternMismatches) && tool.patternMismatches.length > 0) reasons.push(`pattern=${tool.patternMismatches.join('+')}`)
        if (Array.isArray(tool.forbiddenMatches) && tool.forbiddenMatches.length > 0) reasons.push(`forbidden=${tool.forbiddenMatches.join('+')}`)
        if (Array.isArray(tool.parseErrors) && tool.parseErrors.length > 0) reasons.push('parse-error')
        if (reasons.length > 0) parts.push(`reason=${reasons.join(',')}`)
      }
    }
    const flags = check.loopFlags || {}
    const activeFlags = Object.entries(flags)
      .filter(([, value]) => Boolean(value))
      .map(([key]) => key)
    if (activeFlags.length > 0) parts.push(`flags=${activeFlags.join('+')}`)
    return parts.join(':')
  }).join('; ')
}

function mdCell (value) {
  if (value == null) return ''
  return String(value).replace(/\|/g, '\\|').replace(/\n/g, '<br>')
}

function metric (item, key) {
  return item && item.metrics && item.metrics[key] != null ? item.metrics[key] : null
}

function quantile (values, q) {
  const sorted = values.slice().sort((a, b) => a - b)
  if (sorted.length === 0) return null
  if (sorted.length === 1) return sorted[0]
  const pos = (sorted.length - 1) * q
  const base = Math.floor(pos)
  const rest = pos - base
  if (sorted[base + 1] == null) return sorted[base]
  return sorted[base] + rest * (sorted[base + 1] - sorted[base])
}

function valuesSummary (values, digits = 3) {
  const finite = values.map(Number).filter((value) => Number.isFinite(value))
  if (finite.length === 0) {
    return { median: null, min: null, max: null, p90: null }
  }
  return {
    median: round(quantile(finite, 0.5), digits),
    min: round(Math.min(...finite), digits),
    max: round(Math.max(...finite), digits),
    p90: round(quantile(finite, 0.9), digits)
  }
}

function attemptRecords (item) {
  const out = []
  const promptResults = Array.isArray(item.promptResults) ? item.promptResults : []
  for (const prompt of promptResults) {
    const repeatResults = Array.isArray(prompt.repeatResults) ? prompt.repeatResults : []
    if (repeatResults.length > 0) {
      for (const repeatResult of repeatResults) {
        out.push({
          metrics: repeatResult.metrics || null,
          qualityChecks: repeatResult.qualityChecks || null
        })
      }
      continue
    }
    out.push({
      metrics: prompt.metrics || item.metrics || null,
      qualityChecks: prompt.qualityChecks || null
    })
  }
  if (out.length === 0 && item.metrics) {
    out.push({ metrics: item.metrics, qualityChecks: null })
  }
  return out
}

function groupAttempts (items) {
  const out = []
  for (const item of items) out.push(...attemptRecords(item))
  return out
}

function attemptMetricValues (attempts, key) {
  return attempts
    .map((attempt) => attempt.metrics ? Number(attempt.metrics[key]) : NaN)
    .filter((value) => Number.isFinite(value))
}

function promptToolCounts (items) {
  let total = 0
  let passed = 0
  for (const item of items) {
    const promptResults = Array.isArray(item.promptResults) ? item.promptResults : []
    for (const prompt of promptResults) {
      const repeatResults = Array.isArray(prompt.repeatResults) ? prompt.repeatResults : []
      const repeatToolResults = repeatResults
        .map((repeatResult) => repeatResult && repeatResult.qualityChecks ? repeatResult.qualityChecks.tool : null)
        .filter(Boolean)
      if (repeatToolResults.length > 0) {
        for (const tool of repeatToolResults) {
          total++
          if (tool.passed) passed++
        }
        continue
      }
      const tool = prompt && prompt.qualityChecks ? prompt.qualityChecks.tool : null
      if (!tool) continue
      total++
      if (tool.passed) passed++
    }
  }
  return { total, passed }
}

function promptFlagCount (items, flagName) {
  let count = 0
  for (const attempt of groupAttempts(items)) {
    const modes = attempt.qualityChecks ? attempt.qualityChecks.failureModes : null
    if (modes && modes[flagName]) count++
  }
  return count
}

function completionCounts (items) {
  let total = 0
  let passed = 0
  for (const attempt of groupAttempts(items)) {
    const modes = attempt.qualityChecks ? attempt.qualityChecks.failureModes : null
    if (!modes) continue
    total++
    if (!modes.reachedNPredict && !modes.missingThinkClose && !modes.noFinalAnswerAfterThinking) passed++
  }
  return { total, passed }
}

function baselineAgreement (items) {
  const values = items
    .filter((item) => !item.isBaseline && item.qualityJudge != null)
    .map((item) => Number(item.qualityJudge))
    .filter((value) => Number.isFinite(value))
  return round(average(values), 3)
}

function agreementStatus (item) {
  if (item.isBaseline) return 'reference'
  if (item.baselineAgreementStatus) return item.baselineAgreementStatus
  if (item.qualityJudge != null) return 'scored'
  if (item.status !== 'ok' || item.error) return 'case-error'
  return 'unscored'
}

function judgeCoverage (models) {
  const out = {
    references: 0,
    candidates: 0,
    scored: 0,
    unscored: 0,
    errors: 0
  }
  for (const model of models || []) {
    for (const item of model.cases || []) {
      if (item.isBaseline) {
        out.references++
        continue
      }
      out.candidates++
      const status = agreementStatus(item)
      if (status === 'scored' || status === 'exact-match') out.scored++
      else if (status === 'case-error' || status === 'prompt-error' || status === 'judge-null') out.errors++
      else out.unscored++
    }
  }
  return out
}

function highVarianceCases (models) {
  const out = []
  for (const model of models || []) {
    for (const item of model.cases || []) {
      const runMean = Number(metric(item, 'runMsMean'))
      const runStd = Number(metric(item, 'runMsStd'))
      if (!Number.isFinite(runMean) || !Number.isFinite(runStd) || runMean <= 0) continue
      const ratio = runStd / runMean
      if (ratio < 0.5) continue
      out.push({
        modelId: model.modelId,
        quantization: item.quantization || '',
        samplingPreset: item.samplingPreset || item.value || '',
        promptCase: item.promptCase || '',
        runMean: round(runMean, 3),
        runStd: round(runStd, 3),
        ratio: round(ratio, 3),
        approxMin: round(Math.max(0, runMean - runStd), 3),
        approxMax: round(runMean + runStd, 3),
        tpsMean: metric(item, 'tpsMean'),
        tpsStd: metric(item, 'tpsStd'),
        generatedTokens: metric(item, 'generatedTokens')
      })
    }
  }
  return out
}

function aggregateCases (models) {
  const groups = new Map()
  for (const model of models || []) {
    for (const item of model.cases || []) {
      if (item.isBaseline) continue
      const key = [
        model.modelId,
        item.quantization || '',
        item.samplingPreset || item.value || ''
      ].join('\u0000')
      if (!groups.has(key)) {
        groups.set(key, {
          modelId: model.modelId,
          quantization: item.quantization || '',
          samplingPreset: item.samplingPreset || item.value || '',
          cases: []
        })
      }
      groups.get(key).cases.push(item)
    }
  }
  return [...groups.values()]
}

function appendMetadata (lines, report) {
  lines.push(`- Started: ${report.startedAt}`)
  lines.push(`- Finished: ${report.finishedAt}`)
  lines.push(`- Repeats per case: ${report.repeats}`)
  lines.push('- Sweep mode: full-grid')
  lines.push(`- Prompt variants per case: ${report.promptsCount} (one prompt for each Prompt Case row)`)
  if (report.totalCases != null) lines.push(`- Cases: ${report.totalCases}`)
  if (report.totalPlannedRuns != null) lines.push(`- Planned runs: ${report.totalPlannedRuns}`)
  if (report.totalCompletedRuns != null) lines.push(`- Completed runs: ${report.totalCompletedRuns}`)
  lines.push(`- Case records: ${report.jsonlPath}`)
  if (report.benchmarkCommand) lines.push(`- Benchmark command: \`${report.benchmarkCommand}\``)
  if (report.judge && report.judge.command) lines.push(`- Judge command: \`${report.judge.command}\``)
  if (report.judge && report.judge.modelId) {
    lines.push(`- Judge model: ${report.judge.modelId} (${report.judge.quantization || 'unknown'})`)
  }
  if (report.judge && report.judge.rubric) lines.push(`- Judge rubric: ${report.judge.rubric}`)
  lines.push('- Baseline Agreement: similarity to the qvac-current reference output, not a semantic quality score.')
  const coverage = judgeCoverage(report.models)
  lines.push(
    `- Baseline Agreement coverage: ${coverage.scored}/${coverage.candidates} candidate rows scored; ` +
    `${coverage.references} reference rows skipped; ${coverage.unscored} candidate rows unscored; ${coverage.errors} judge/case errors.`
  )
  if (report.runtime) {
    const runtime = report.runtime
    const parts = []
    if (runtime.addonSource) parts.push(`addonSource=${runtime.addonSource}`)
    if (runtime.platform) parts.push(`platform=${runtime.platform}`)
    if (runtime.arch) parts.push(`arch=${runtime.arch}`)
    if (runtime.cpuModel) parts.push(`cpu=${runtime.cpuModel}`)
    if (parts.length > 0) lines.push(`- Runtime: ${parts.join(', ')}`)
  }
  if (report.sweep) lines.push(`- Sweep dimensions: ${JSON.stringify(report.sweep)}`)
}

function appendAggregateSummary (lines, report) {
  const groups = aggregateCases(report.models)
  if (groups.length === 0) return

  lines.push('## Aggregate Summary')
  lines.push('')
  lines.push('| Model | Quantization | Sampling Preset | Cases OK | Tool Attempts Pass | Completion Pass | NPredict Hits | Run Median | Run P90 | Run Min | Run Max | TTFT Median | TPS Median | ms/Token Median | Generated Tokens Median | Generated Tokens Min | Generated Tokens Max | Baseline Agreement |')
  lines.push('|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
  for (const group of groups) {
    const cases = group.cases
    const okCount = cases.filter((item) => item.status === 'ok').length
    const tool = promptToolCounts(cases)
    const toolCell = tool.total > 0 ? `${tool.passed}/${tool.total}` : ''
    const completion = completionCounts(cases)
    const completionCell = completion.total > 0 ? `${completion.passed}/${completion.total}` : ''
    const attempts = groupAttempts(cases)
    const run = valuesSummary(attemptMetricValues(attempts, 'runMs'))
    const ttft = valuesSummary(attemptMetricValues(attempts, 'ttftMs'))
    const tps = valuesSummary(attemptMetricValues(attempts, 'tps'))
    const msPerToken = valuesSummary(attemptMetricValues(attempts, 'runMsPerGeneratedToken'), 6)
    const tokens = valuesSummary(attemptMetricValues(attempts, 'generatedTokens'), 1)
    lines.push(
      `| ${mdCell(group.modelId)} | ${mdCell(group.quantization)} | ${mdCell(group.samplingPreset)}` +
      ` | ${okCount}/${cases.length} | ${toolCell} | ${completionCell} | ${promptFlagCount(cases, 'reachedNPredict')}` +
      ` | ${run.median ?? ''} | ${run.p90 ?? ''} | ${run.min ?? ''} | ${run.max ?? ''}` +
      ` | ${ttft.median ?? ''} | ${tps.median ?? ''} | ${msPerToken.median ?? ''}` +
      ` | ${tokens.median ?? ''} | ${tokens.min ?? ''} | ${tokens.max ?? ''}` +
      ` | ${baselineAgreement(cases) != null ? baselineAgreement(cases).toFixed(3) : ''} |`
    )
  }
  lines.push('')
}

function appendHighVarianceCases (lines, report) {
  const cases = highVarianceCases(report.models)
  if (cases.length === 0) return
  lines.push('## High Variance Cases')
  lines.push('')
  lines.push('Cases where `runMsStd / runMsMean >= 0.5`; approximate range is `mean - std` to `mean + std`.')
  lines.push('')
  lines.push('| Model | Quantization | Sampling Preset | Prompt Case | Run Mean | Run Std | Std/Mean | Approx Run Range | TPS Mean | TPS Std | Generated Tokens |')
  lines.push('|---|---|---|---|---:|---:|---:|---|---:|---:|---:|')
  for (const item of cases) {
    lines.push(
      `| ${mdCell(item.modelId)} | ${mdCell(item.quantization)} | ${mdCell(item.samplingPreset)} | ${mdCell(item.promptCase)}` +
      ` | ${item.runMean} | ${item.runStd} | ${item.ratio} | ${item.approxMin}-${item.approxMax}` +
      ` | ${item.tpsMean ?? ''} | ${item.tpsStd ?? ''} | ${item.generatedTokens ?? ''} |`
    )
  }
  lines.push('')
}

function toMarkdown (report) {
  const lines = []
  lines.push('# LLM Parameter Sweep Benchmark Report')
  lines.push('')
  appendMetadata(lines, report)
  lines.push('')
  appendAggregateSummary(lines, report)
  appendHighVarianceCases(lines, report)
  for (const model of report.models) {
    lines.push(`## Raw Cases: ${model.modelId}`)
    lines.push('| Role | Quantization | Sampling Preset | Temp | Top P | Top K | Presence Penalty | Repeat Penalty | Reasoning Budget | Device | Ctx Size | N Predict | Batch Size | Ubatch Size | Flash Attn | Threads | Cache K | Cache V | Prompt Case | Status | Run Mean | Run Std | TTFT Mean | TTFT Std | TPS Mean | TPS Std | ms/Token | ms/Token Std | Generated Tokens | Quality Match | Baseline Agreement | Agreement Status | Checks | Error |')
    lines.push('|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|')
    for (const item of model.cases) {
      const runtimeConfig = item.runtimeConfig || {}
      const quality = item.qualityMatch != null ? item.qualityMatch.toFixed(3) : ''
      const roleCell = item.isBaseline ? 'reference' : 'candidate'
      const qualityJudge = item.isBaseline
        ? 'reference'
        : (item.qualityJudge != null ? item.qualityJudge.toFixed(3) : '')
      const quantizationCell = item.quantization ?? ''
      const samplingPresetCell = item.samplingPreset || runtimeConfig['sampling-preset'] || ''
      const deviceCell = runtimeConfig.device != null ? String(runtimeConfig.device) : ''
      const ctxSizeCell = runtimeConfig['ctx-size'] != null ? String(runtimeConfig['ctx-size']) : ''
      const nPredictCell = runtimeConfig['n-predict'] != null ? String(runtimeConfig['n-predict']) : ''
      const batchSizeCell = runtimeConfig['batch-size'] != null ? String(runtimeConfig['batch-size']) : ''
      const ubatchSizeCell = runtimeConfig['ubatch-size'] != null ? String(runtimeConfig['ubatch-size']) : ''
      const flashAttnCell = runtimeConfig['flash-attn'] != null ? String(runtimeConfig['flash-attn']) : ''
      const threadsCell = runtimeConfig.threads != null ? String(runtimeConfig.threads) : ''
      const cacheKCell = runtimeConfig['cache-type-k'] != null ? String(runtimeConfig['cache-type-k']) : ''
      const cacheVCell = runtimeConfig['cache-type-v'] != null ? String(runtimeConfig['cache-type-v']) : ''
      const errorCell = item.error && item.error.message
        ? truncateText(item.error.message, 120)
        : ''
      const checksCell = truncateText(formatQualityChecks(item), 160)
      lines.push(
        `| ${roleCell} | ${quantizationCell} | ${samplingPresetCell}` +
        ` | ${runtimeConfig.temp ?? ''} | ${runtimeConfig['top-p'] ?? ''} | ${runtimeConfig['top-k'] ?? ''}` +
        ` | ${runtimeConfig['presence-penalty'] ?? ''} | ${runtimeConfig['repeat-penalty'] ?? ''} | ${runtimeConfig['reasoning-budget'] ?? ''}` +
        ` | ${deviceCell} | ${ctxSizeCell} | ${nPredictCell} | ${batchSizeCell} | ${ubatchSizeCell} | ${flashAttnCell} | ${threadsCell} | ${cacheKCell} | ${cacheVCell}` +
        ` | ${item.promptCase ?? ''} | ${item.status ?? ''}` +
        ` | ${item.metrics?.runMsMean ?? ''} | ${item.metrics?.runMsStd ?? ''}` +
        ` | ${item.metrics?.ttftMsMean ?? ''} | ${item.metrics?.ttftMsStd ?? ''}` +
        ` | ${item.metrics?.tpsMean ?? ''} | ${item.metrics?.tpsStd ?? ''}` +
        ` | ${item.metrics?.runMsPerGeneratedTokenMean ?? ''} | ${item.metrics?.runMsPerGeneratedTokenStd ?? ''}` +
        ` | ${item.metrics?.generatedTokens ?? ''} | ${quality} | ${qualityJudge} | ${agreementStatus(item)} | ${checksCell} | ${errorCell} |`
      )
    }
    lines.push('')
  }
  lines.push('')
  return `${lines.join('\n')}\n`
}

module.exports = {
  tsFileStamp,
  compactPromptErrors,
  compactQualityChecks,
  toMarkdown
}
