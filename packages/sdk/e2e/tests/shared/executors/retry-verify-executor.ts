import type { TestResult } from "@tetherto/qvac-test-suite";

/**
 * Each attempt and reload sleeps this long so the memory poller (samples every
 * ~200-500ms) captures multiple samples per phase. Without it these mock tests
 * finish in single-digit ms, the reload boundary collapses onto the end
 * boundary, and the Memory tab cannot split attempt 1 / attempt 2.
 */
const PHASE_DELAY_MS = 1200;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Executor that covers four distinct retry-mechanism scenarios:
 *
 *   retry-verify-pass          — always passes (no retry involved)
 *   retry-verify-flaky         — attempt 1 fails, attempt 2 passes  (retryOnFailure: true)
 *   retry-verify-fail-no-retry — always fails, retry disabled        (retryOnFailure: false)
 *   retry-verify-fail-both     — attempt 1 fails, attempt 2 also fails (retryOnFailure: true)
 *
 * No real model or ResourceManager is involved. Call counters are reset in
 * teardown so each scenario starts fresh on subsequent suite runs.
 */
export class RetryVerifyExecutor {
  readonly pattern = /^retry-verify-/;

  private readonly callCounts = new Map<string, number>();

  async execute(testId: string): Promise<TestResult> {
    const count = this.callCounts.get(testId) ?? 0;
    this.callCounts.set(testId, count + 1);

    await sleep(PHASE_DELAY_MS);

    switch (testId) {
      case "retry-verify-pass":
        return { passed: true, output: "[retry-verify] always passes" };

      case "retry-verify-flaky":
        if (count === 0) {
          return {
            passed: false,
            output: "[retry-verify] attempt 1: intentional failure — reload retry should follow",
          };
        }
        return {
          passed: true,
          output: "[retry-verify] attempt 2: passed after reload",
        };

      case "retry-verify-fail-no-retry":
        return {
          passed: false,
          output: "[retry-verify] always fails — no retry configured",
        };

      case "retry-verify-fail-both":
        return {
          passed: false,
          output: `[retry-verify] attempt ${count + 1}: fails on every attempt`,
        };

      default:
        return { passed: false, output: `[retry-verify] unknown testId: ${testId}` };
    }
  }

  // reload() only sleeps so the attempt-2 memory window spans real samples;
  // the call counter naturally increments through both attempts.
  async reload(_testId: string): Promise<void> {
    await sleep(PHASE_DELAY_MS);
  }

  async teardown(testId: string): Promise<void> {
    this.callCounts.delete(testId);
  }
}
