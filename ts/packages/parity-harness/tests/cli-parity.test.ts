import { describe, expect, it } from "vitest";
import { compareCliCommands, type CliCommand } from "../src/cli-parity.js";

describe("compareCliCommands", () => {
  it("returns pass=true when commands match", () => {
    const commands: CliCommand[] = [
      { name: "serve", is_group: false, params: [] },
      { name: "run", is_group: false, params: [{ name: "workflow_id", type: "STRING", required: true }] },
    ];

    const report = compareCliCommands(commands, [...commands]);
    expect(report.pass).toBe(true);
    expect(report.summary.matched).toBe(2);
  });

  it("detects commands in Python but not TS", () => {
    const pyCommands: CliCommand[] = [
      { name: "serve", is_group: false, params: [] },
      { name: "deploy", is_group: true, params: [] },
    ];
    const tsCommands: CliCommand[] = [
      { name: "serve", is_group: false, params: [] },
    ];

    const report = compareCliCommands(pyCommands, tsCommands);
    expect(report.summary.pythonOnly).toBe(1);
    expect(report.drifts).toHaveLength(1);
    expect(report.drifts[0].command).toBe("deploy");
  });

  it("detects parameter differences in matched commands", () => {
    const pyCommands: CliCommand[] = [
      {
        name: "run",
        is_group: false,
        params: [
          { name: "workflow_id", type: "STRING", required: true },
          { name: "verbose", type: "BOOL", required: false },
        ],
      },
    ];
    const tsCommands: CliCommand[] = [
      {
        name: "run",
        is_group: false,
        params: [{ name: "workflow_id", type: "STRING", required: true }],
      },
    ];

    const report = compareCliCommands(pyCommands, tsCommands);
    expect(report.summary.matched).toBe(1);
    // Should note the missing "verbose" param
    expect(report.drifts.some((d) => d.message.includes("verbose"))).toBe(true);
  });

  it("reports TS-only commands as info", () => {
    const pyCommands: CliCommand[] = [];
    const tsCommands: CliCommand[] = [
      { name: "chat", is_group: false, params: [] },
    ];

    const report = compareCliCommands(pyCommands, tsCommands);
    expect(report.pass).toBe(true);
    expect(report.drifts[0].severity).toBe("info");
  });
});
