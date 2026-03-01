import { describe, expect, it } from "vitest";
import { runShadowComparison } from "../src/shadow.js";

describe("runShadowComparison", () => {
  it("runs both commands and reports no drift for matching JSONL", async () => {
    const script = "console.log(JSON.stringify({type:'job_update',status:'running'}));";

    const result = await runShadowComparison(
      { cmd: "node", args: ["-e", script] },
      { cmd: "node", args: ["-e", script] }
    );

    expect(result.python.exitCode).toBe(0);
    expect(result.ts.exitCode).toBe(0);
    expect(result.drifts).toEqual([]);
  });
});
