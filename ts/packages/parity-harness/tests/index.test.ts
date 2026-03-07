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

  it("exports compareModelSchemas from model-parity module", async () => {
    const mod = await import("../src/index.js");
    expect(mod.compareModelSchemas).toBeTypeOf("function");
  });

  it("exports compareApiRoutes from api-parity module", async () => {
    const mod = await import("../src/index.js");
    expect(mod.compareApiRoutes).toBeTypeOf("function");
  });

  it("exports compareCliCommands from cli-parity module", async () => {
    const mod = await import("../src/index.js");
    expect(mod.compareCliCommands).toBeTypeOf("function");
  });

  it("exports compareLibraryClasses and snakeToCamel from library-parity module", async () => {
    const mod = await import("../src/index.js");
    expect(mod.compareLibraryClasses).toBeTypeOf("function");
    expect(mod.snakeToCamel).toBeTypeOf("function");
  });
});
