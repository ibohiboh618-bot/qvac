'use strict'

const { toMarkdown } = require('./reporters')

function assert (condition, message) {
  if (!condition) {
    throw new Error(message || 'assertion failed')
  }
}

const report = {
  startedAt: '2026-06-12T15:00:44.213Z',
  finishedAt: '2026-06-12T15:48:28.903Z',
  repeats: 2,
  promptsCount: 1,
  totalCases: 2,
  totalPlannedRuns: 4,
  totalCompletedRuns: 4,
  jsonlPath: '/tmp/run.judged.jsonl',
  benchmarkCommand: 'bare ./llm-parameter-sweep.js --models qwen3.5-0.8b',
  judge: {
    command: 'bare ./run-judge.js --judge-model qwen3.5-0.8b',
    modelId: 'qwen3.5-0.8b',
    quantization: 'Q4_K_M',
    rubric: 'Return only a number between 0 and 1 for semantic agreement.'
  },
  runtime: {
    addonSource: 'npm',
    platform: 'darwin',
    arch: 'arm64',
    cpuModel: 'Apple M-series'
  },
  models: [
    {
      modelId: 'qwen3.5-0.8b',
      cases: [
        {
          isBaseline: true,
          samplingPreset: 'qvac-current',
          quantization: 'Q4_K_M',
          promptCase: 'direct-qa',
          status: 'ok',
          runtimeConfig: {
            device: 'gpu',
            'ctx-size': '4096',
            'n-predict': '1024',
            temp: '0.1',
            'top-p': '0.9',
            'top-k': '40',
            'presence-penalty': '0',
            'repeat-penalty': '1.1',
            'reasoning-budget': '-1'
          },
          metrics: {
            runMsMean: 1000,
            runMsStd: 10,
            ttftMsMean: 50,
            ttftMsStd: 2,
            tpsMean: 90,
            tpsStd: 3,
            runMsPerGeneratedTokenMean: 10,
            runMsPerGeneratedTokenStd: 1,
            generatedTokens: 1024
          },
          qualityJudge: 1,
          promptResults: [
            {
              promptId: 'direct-qa',
              qualityChecks: {
                failureModes: {
                  reachedNPredict: true,
                  missingThinkClose: true,
                  noFinalAnswerAfterThinking: true,
                  repeatedNgram: true
                }
              }
            }
          ]
        },
        {
          isBaseline: false,
          samplingPreset: 'qwen-thinking-low-penalty',
          quantization: 'Q4_K_M',
          promptCase: 'tool-run-cli',
          status: 'ok',
          runtimeConfig: {
            device: 'gpu',
            'ctx-size': '4096',
            'n-predict': '1024',
            temp: '1.0',
            'top-p': '0.95',
            'top-k': '20',
            'presence-penalty': '0',
            'repeat-penalty': '1.0',
            'reasoning-budget': '-1'
          },
          metrics: {
            runMsMean: 900,
            runMsStd: 500,
            ttftMsMean: 40,
            ttftMsStd: 4,
            tpsMean: 100,
            tpsStd: 5,
            runMsPerGeneratedTokenMean: 9,
            runMsPerGeneratedTokenStd: 2,
            generatedTokens: 100
          },
          qualityJudge: 0.75,
          baselineAgreementStatus: 'scored',
          promptResults: [
            {
              promptId: 'tool-run-cli',
              repeatResults: [
                {
                  repeat: 1,
                  metrics: {
                    runMs: 800,
                    ttftMs: 35,
                    tps: 95,
                    runMsPerGeneratedToken: 8,
                    generatedTokens: 90
                  },
                  qualityChecks: {
                    tool: { passed: true },
                    failureModes: {
                      reachedNPredict: false,
                      missingThinkClose: false,
                      noFinalAnswerAfterThinking: false,
                      repeatedNgram: false
                    }
                  }
                },
                {
                  repeat: 2,
                  metrics: {
                    runMs: 1000,
                    ttftMs: 45,
                    tps: 105,
                    runMsPerGeneratedToken: 10,
                    generatedTokens: 110
                  },
                  qualityChecks: {
                    tool: { passed: true },
                    failureModes: {
                      reachedNPredict: false,
                      missingThinkClose: false,
                      noFinalAnswerAfterThinking: false,
                      repeatedNgram: false
                    }
                  }
                }
              ],
              qualityChecks: {
                tool: { passed: true },
                failureModes: {
                  reachedNPredict: false,
                  missingThinkClose: false,
                  noFinalAnswerAfterThinking: false,
                  repeatedNgram: false
                }
              }
            }
          ]
        }
      ]
    }
  ]
}

const markdown = toMarkdown(report)

assert(markdown.includes('- Started: 2026-06-12T15:00:44.213Z'))
assert(markdown.includes('- Repeats per case: 2'))
assert(markdown.includes('- Prompt variants per case: 1 (one prompt for each Prompt Case row)'))
assert(markdown.includes('- Benchmark command: `bare ./llm-parameter-sweep.js --models qwen3.5-0.8b`'))
assert(markdown.includes('- Judge model: qwen3.5-0.8b (Q4_K_M)'))
assert(markdown.includes('- Judge rubric: Return only a number between 0 and 1 for semantic agreement.'))
assert(markdown.includes('- Baseline Agreement: similarity to the qvac-current reference output, not a semantic quality score.'))
assert(markdown.includes('- Baseline Agreement coverage: 1/1 candidate rows scored; 1 reference rows skipped; 0 candidate rows unscored; 0 judge/case errors.'))
assert(markdown.includes('- Runtime: addonSource=npm, platform=darwin, arch=arm64, cpu=Apple M-series'))
assert(markdown.includes('## Aggregate Summary'))
assert(markdown.includes('| qwen3.5-0.8b | Q4_K_M | qwen-thinking-low-penalty | 1/1 | 2/2 | 2/2 | 0 |'))
assert(markdown.includes('| 900 | 980 | 800 | 1000 | 40 | 100 | 9 | 100 | 90 | 110 | 0.750 |'))
assert(!markdown.includes('| qwen3.5-0.8b | Q4_K_M | qvac-current | 1/1 |'))
assert(markdown.includes('| reference | Q4_K_M | qvac-current |'))
assert(markdown.includes('| reference | Q4_K_M | qvac-current | 0.1 | 0.9 | 40 | 0 | 1.1 | -1 | gpu | 4096 | 1024 |'))
assert(markdown.includes('## High Variance Cases'))
assert(markdown.includes('| direct-qa | ok | 1000 | 10 | 50 | 2 | 90 | 3 | 10 | 1 | 1024 |  | reference | reference |'))
assert(markdown.includes('| qwen3.5-0.8b | Q4_K_M | qwen-thinking-low-penalty | tool-run-cli | 900 | 500 | 0.556 | 400-1400 | 100 | 5 | 100 |'))
assert(markdown.includes('| tool-run-cli | ok | 900 | 500 | 40 | 4 | 100 | 5 | 9 | 2 | 100 |  | 0.750 | scored |'))

console.log('reporters.test.js ok')
