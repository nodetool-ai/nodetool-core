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

  it("filters out non-JSON and non-object lines from messages", async () => {
    const script = `
      console.log("plain text");
      console.log("not json {");
      console.log(JSON.stringify({type:'job_update',status:'running'}));
      console.log(JSON.stringify({noTypeField: true}));
      console.log(JSON.stringify(42));
    `;

    const result = await runShadowComparison(
      { cmd: "node", args: ["-e", script] },
      { cmd: "node", args: ["-e", script] }
    );

    // Only the line with type:'job_update' should be in messages
    expect(result.python.messages).toHaveLength(1);
    expect(result.python.messages[0]).toEqual({ type: "job_update", status: "running" });
    // stdoutLines should contain all non-empty lines
    expect(result.python.stdoutLines.length).toBeGreaterThan(1);
  });

  it("captures stderr output", async () => {
    const script = `console.error("error msg"); console.log(JSON.stringify({type:'done'}));`;

    const result = await runShadowComparison(
      { cmd: "node", args: ["-e", script] },
      { cmd: "node", args: ["-e", script] }
    );

    expect(result.python.stderr).toContain("error msg");
  });

  it("reports non-zero exit code", async () => {
    const script = `console.log(JSON.stringify({type:'done'})); process.exit(1);`;

    const result = await runShadowComparison(
      { cmd: "node", args: ["-e", script] },
      { cmd: "node", args: ["-e", "console.log(JSON.stringify({type:'done'}))"] }
    );

    expect(result.python.exitCode).toBe(1);
    expect(result.ts.exitCode).toBe(0);
  });

  it("rejects when command does not exist", async () => {
    await expect(
      runShadowComparison(
        { cmd: "nonexistent_command_xyz_12345", args: [] },
        { cmd: "node", args: ["-e", "console.log('hi')"] }
      )
    ).rejects.toThrow();
  });

  it("detects drift between different outputs", async () => {
    const pyScript = `console.log(JSON.stringify({type:'output_update',value:1}));`;
    const tsScript = `console.log(JSON.stringify({type:'output_update',value:2}));`;

    const result = await runShadowComparison(
      { cmd: "node", args: ["-e", pyScript] },
      { cmd: "node", args: ["-e", tsScript] }
    );

    expect(result.drifts).toHaveLength(1);
    expect(result.drifts[0].category).toBe("output_drift");
  });

  it("passes env to spawned commands", async () => {
    const script = `console.log(JSON.stringify({type:'output_update',value:process.env.TEST_VAR}));`;

    const result = await runShadowComparison(
      { cmd: "node", args: ["-e", script], env: { TEST_VAR: "hello" } },
      { cmd: "node", args: ["-e", script], env: { TEST_VAR: "hello" } }
    );

    expect(result.python.messages[0]).toEqual({ type: "output_update", value: "hello" });
    expect(result.drifts).toHaveLength(0);
  });
});
