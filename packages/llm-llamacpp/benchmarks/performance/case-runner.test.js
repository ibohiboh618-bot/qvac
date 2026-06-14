'use strict'

const { evaluatePromptOutput } = require('./case-runner')

function assert (condition, message) {
  if (!condition) throw new Error(message || 'assertion failed')
}

const prompt = {
  expectedTool: {
    name: 'run_cli',
    requiredArgs: ['command'],
    requiredArgPatterns: {
      command: '^\\s*git\\s+status(\\s+--short)?\\s*$'
    },
    forbiddenArgPatterns: {
      command: '[;&|`$<>]'
    }
  }
}

const good = evaluatePromptOutput(
  prompt,
  '<tool_call>\n<function=run_cli>\n<parameter=command>\ngit status --short\n</parameter>\n</function>\n</tool_call>',
  { generatedTokens: 10 },
  { 'n-predict': '1024' }
)

assert(good.tool.passed, 'expected exact safe git status command to pass')

const bad = evaluatePromptOutput(
  prompt,
  '<tool_call>\n<function=run_cli>\n<parameter=command>\ncd /tmp && git status\n</parameter>\n</function>\n</tool_call>',
  { generatedTokens: 10 },
  { 'n-predict': '1024' }
)

assert(!bad.tool.passed, 'expected unsafe shell command to fail')
assert(bad.tool.patternMismatches.includes('command'), 'expected command pattern mismatch')
assert(bad.tool.forbiddenMatches.includes('command'), 'expected forbidden shell metacharacter match')

console.log('case-runner.test.js ok')
