import { describe, it, expect } from "vitest";

describe("agents index exports", () => {
  it("exports all public API", async () => {
    const mod = await import("../src/index.js");
    expect(mod).toBeDefined();

    // Tools
    expect(mod.Tool).toBeDefined();
    expect(mod.FinishStepTool).toBeDefined();
    expect(mod.CalculatorTool).toBeDefined();
    expect(mod.ReadFileTool).toBeDefined();
    expect(mod.WriteFileTool).toBeDefined();
    expect(mod.ListDirectoryTool).toBeDefined();
    expect(mod.DownloadFileTool).toBeDefined();
    expect(mod.HttpRequestTool).toBeDefined();
    expect(mod.RunCodeTool).toBeDefined();
    expect(mod.GoogleSearchTool).toBeDefined();
    expect(mod.GoogleNewsTool).toBeDefined();
    expect(mod.GoogleImagesTool).toBeDefined();
    expect(mod.BrowserTool).toBeDefined();
    expect(mod.ScreenshotTool).toBeDefined();
    expect(mod.htmlToText).toBeDefined();
    expect(mod.registerTool).toBeDefined();
    expect(mod.resolveTool).toBeDefined();
    expect(mod.listTools).toBeDefined();
    expect(mod.getAllTools).toBeDefined();

    // Utilities
    expect(mod.extractJSON).toBeDefined();

    // Core execution
    expect(mod.StepExecutor).toBeDefined();

    // Agents
    expect(mod.BaseAgent).toBeDefined();
    expect(mod.SimpleAgent).toBeDefined();

    // Planning & orchestration
    expect(mod.TaskPlanner).toBeDefined();
    expect(mod.TaskExecutor).toBeDefined();
  });
});
