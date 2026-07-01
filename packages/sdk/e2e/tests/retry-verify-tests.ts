import type { TestDefinition } from "@tetherto/qvac-test-suite";

/**
 * Framework verification tests covering all four retry-mechanism outcomes.
 * These tests do NOT exercise any real SDK model functionality — they exist
 * solely to confirm the retry pipeline and reporting look correct end-to-end.
 *
 *   retry-verify-pass          → ✅ success        (no retry)
 *   retry-verify-flaky         → ❌ + ↩ RETRY:✓   (fails attempt 1, passes attempt 2)
 *   retry-verify-fail-no-retry → ❌ failure        (no retry configured)
 *   retry-verify-fail-both     → ❌ + ↩ RETRY:✗   (fails both attempts)
 */
export const retryVerifyTests: TestDefinition[] = [
  {
    testId: "retry-verify-pass",
    params: {},
    expectation: { validation: "type", expectedType: "string" },
    retryOnFailure: false,
    metadata: {
      category: "retry-verify",
      dependency: "none",
      estimatedDurationMs: 500,
    },
  },
  {
    testId: "retry-verify-flaky",
    params: {},
    expectation: { validation: "type", expectedType: "string" },
    retryOnFailure: true,
    metadata: {
      category: "retry-verify",
      dependency: "none",
      estimatedDurationMs: 500,
    },
  },
  {
    testId: "retry-verify-fail-no-retry",
    params: {},
    expectation: { validation: "type", expectedType: "string" },
    retryOnFailure: false,
    metadata: {
      category: "retry-verify",
      dependency: "none",
      estimatedDurationMs: 500,
    },
  },
  {
    testId: "retry-verify-fail-both",
    params: {},
    expectation: { validation: "type", expectedType: "string" },
    retryOnFailure: true,
    metadata: {
      category: "retry-verify",
      dependency: "none",
      estimatedDurationMs: 500,
    },
  },
];
