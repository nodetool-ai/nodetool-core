import { describe, expect, it } from "vitest";
import { evaluateCanaryGates } from "../src/gates.js";

describe("evaluateCanaryGates", () => {
  it("passes when metrics are within thresholds", () => {
    const result = evaluateCanaryGates({
      baselineErrorRate: 0.01,
      candidateErrorRate: 0.011,
      baselineCompletionRate: 0.99,
      candidateCompletionRate: 0.985,
      baselineP95Ms: 1000,
      candidateP95Ms: 1100,
      outputParityScore: 0.995,
    });

    expect(result.pass).toBe(true);
    expect(result.reasons).toHaveLength(0);
  });

  it("fails when parity and latency regress", () => {
    const result = evaluateCanaryGates({
      baselineErrorRate: 0.01,
      candidateErrorRate: 0.02,
      baselineCompletionRate: 0.99,
      candidateCompletionRate: 0.95,
      baselineP95Ms: 1000,
      candidateP95Ms: 1300,
      outputParityScore: 0.95,
    });

    expect(result.pass).toBe(false);
    expect(result.reasons.length).toBeGreaterThanOrEqual(3);
  });
});
