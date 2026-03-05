import { describe, it, expect } from "vitest";

describe("index barrel exports", () => {
  it("exports diffMessageStreams from diff module", async () => {
    const mod = await import("../src/index.js");
    expect(mod.diffMessageStreams).toBeTypeOf("function");
  });

  it("exports runShadowComparison from shadow module", async () => {
    const mod = await import("../src/index.js");
    expect(mod.runShadowComparison).toBeTypeOf("function");
  });

  it("exports evaluateCanaryGates and DEFAULT_THRESHOLDS from gates module", async () => {
    const mod = await import("../src/index.js");
    expect(mod.evaluateCanaryGates).toBeTypeOf("function");
    expect(mod.DEFAULT_THRESHOLDS).toBeDefined();
    expect(mod.DEFAULT_THRESHOLDS.maxErrorRateDelta).toBeTypeOf("number");
  });
});
